"""质检自查：识别乱码/低置信内容，复核重试，剩余问题显式标注。"""
import re
import unicodedata
from typing import List, Tuple

from .models import Block, Line
from .vision_ocr import StripProvider, ocr_strip

# 明确的乱码信号
_BAD_CHARS = re.compile(r'[�-]')  # 替换符 / 私用区
_CID_RE = re.compile(r'\(cid:\d+\)')
# 目录条目：引导点会拉低 OCR 置信度，但清洗后的文本本身没问题
_TOC_LIKE = re.compile(r'^.{1,40}[　.\s。·•．…⋯：:\-－—]\d{1,3}$')
# 数学公式信号：OCR 处理分式/上下标很不可靠，这类低置信块直接截图
_MATH_RE = re.compile(r'[=≤≥×÷√∑∏∝≈≠＋＝]|[+*/^]\s*\d|\d\s*[+*/]')


def garbled_score(text: str) -> float:
    """0~1，越高越可疑。"""
    if not text:
        return 0.0
    if _BAD_CHARS.search(text) or _CID_RE.search(text):
        return 1.0
    n_total = 0
    n_odd = 0
    for ch in text:
        if ch.isspace() or ch.isascii():
            continue
        n_total += 1
        cat = unicodedata.category(ch)
        # 只有控制符/私用区/未定义字符算乱码；数学符号(Sm)等是正常内容
        if cat in ("Cc", "Cf", "Co", "Cn", "Cs"):
            n_odd += 1
    if n_total == 0:
        return 0.0
    return n_odd / n_total


def recheck_ocr_blocks(blocks: List[Block], provider: StripProvider,
                       conf_thresh: float = 0.45) -> List[str]:
    """对低置信度的 OCR 块放大 2 倍重新识别复核。

    复核结果与原结果一致 -> 通过；明显更优(置信度提升) -> 替换；
    仍然可疑 -> 打上 low_confidence 标记，导出时显式提示。
    """
    notes = []
    for blk in blocks:
        if blk.kind not in ("para", "heading") or blk.confidence >= conf_thresh:
            continue
        y0, y1 = int(blk.bbox[1]) - 6, int(blk.bbox[3]) + 6
        try:
            img = provider.get_strip(max(0, y0), min(provider.height, y1))
            # 只看本块的 x 范围，避免同一行其他栏的文字混进复核结果
            x0 = max(0, int(blk.bbox[0]) - 8)
            x1 = min(img.width, int(blk.bbox[2]) + 8)
            if x1 - x0 > 20:
                img = img.crop((x0, 0, x1, img.height))
            img2 = img.resize((img.width * 2, img.height * 2))
            lines = ocr_strip(img2, upscale_to=0)
        except Exception as e:
            notes.append(f"复核失败 y={y0}: {e}")
            blk.flags.append("low_confidence")
            continue
        if lines:
            from .assemble import cluster_rows, _join_lines, clean_leaders
            new_text = clean_leaders(_join_lines(cluster_rows(lines)))
            new_conf = min(l.conf for l in lines)
            norm_old = re.sub(r'\s+', '', blk.text)
            norm_new = re.sub(r'\s+', '', new_text)
            if norm_new == norm_old:
                # 两次独立识别结果一致，可信
                blk.confidence = max(blk.confidence, new_conf, conf_thresh)
                continue
            elif new_conf > blk.confidence + 0.15 and len(norm_new) >= len(norm_old) * 0.6:
                notes.append(f"复核修正: “{blk.text[:20]}…” -> “{new_text[:20]}…”")
                blk.text = new_text
                blk.confidence = new_conf
                if new_conf >= conf_thresh:
                    continue
        # 目录条目（引导点天然拉低置信度），文本干净即放行
        if _TOC_LIKE.match(blk.text) and garbled_score(blk.text) == 0:
            continue
        blk.flags.append("low_confidence")
    return notes


def image_fallback(blocks: List[Block], provider: StripProvider
                   ) -> Tuple[List[Block], List[str]]:
    """复核后仍不可信的内容（公式/图示文字）不输出错误文本，改为截原图。

    连续多个低置信块（典型的图示/复杂公式区域）合并成一张截图；
    单个低置信块只在带数学符号或乱码时转图。
    """
    notes = []
    out: List[Block] = []
    group: List[Block] = []

    def _is_bad(b: Block) -> bool:
        return (b.kind in ("para", "heading") and b.confidence <= 0.35
                and ("low_confidence" in b.flags or "garbled" in b.flags))

    def flush_group():
        if not group:
            return
        single = len(group) == 1
        if single and not (_MATH_RE.search(group[0].text)
                           or garbled_score(group[0].text) > 0):
            out.extend(group)
            group.clear()
            return
        y0 = min(b.bbox[1] for b in group) - 8
        y1 = max(b.bbox[3] for b in group) + 8
        x0 = max(0, min(b.bbox[0] for b in group) - 60)
        x1 = min(provider.width, max(b.bbox[2] for b in group) + 60)
        blk = Block(kind="image", page=group[0].page,
                    bbox=(x0, max(0, y0), x1, min(provider.height, y1)),
                    flags=["auto_image"])
        notes.append(f"低置信区域已转为截图(y={int(y0)}): "
                     + " / ".join(b.text[:18] for b in group[:3]))
        out.append(blk)
        group.clear()

    for b in sorted(blocks, key=lambda b: b.bbox[1]):
        if _is_bad(b):
            if group and b.bbox[1] - max(g.bbox[3] for g in group) > 250:
                flush_group()
            group.append(b)
        else:
            flush_group()
            out.append(b)
    flush_group()

    # 吸收：紧挨着截图区的残余低置信文本（公式的分子/分母标签等）并入截图
    imgs = [b for b in out if b.kind == "image" and "auto_image" in b.flags]
    if imgs:
        absorbed = set()
        for b in out:
            if not _is_bad(b):
                continue
            for im in imgs:
                if (b.bbox[1] < im.bbox[3] + 160 and b.bbox[3] > im.bbox[1] - 160):
                    im.bbox = (max(0, min(im.bbox[0], b.bbox[0] - 60)),
                               min(im.bbox[1], b.bbox[1] - 8),
                               min(provider.width, max(im.bbox[2], b.bbox[2] + 60)),
                               max(im.bbox[3], b.bbox[3] + 8))
                    absorbed.add(id(b))
                    notes.append(f"并入相邻截图: {b.text[:18]}")
                    break
        out = [b for b in out if id(b) not in absorbed]
    out.sort(key=lambda b: b.bbox[1])  # 吸收可能改变 bbox，重排保证视觉顺序
    return out, notes


# 高频 OCR 形近字混淆对：(误识字, 正确候选)。仅在全文词频投票支持时才改写
_CONFUSION = [("惺", "1"), ("项", "顶"), ("顶", "项"), ("己", "已"),
              ("酒", "洒"), ("凋", "调"), ("令", "或"), ("涨", "胀"),
              ("O", "0")]


def doc_vote_fix(blocks: List[Block]) -> List[str]:
    """文档内多数派投票纠错：形近字误识（封项值→封顶值、惺星→1星）。

    对每个可疑字，比较"原二元组"与"替换后二元组"在全文中的出现频次，
    替换形显著占优（≥3次 且 ≥3倍）才改写——绝不凭空猜测。
    """
    from collections import Counter

    def texts_of(b: Block):
        if b.kind in ("para", "heading"):
            yield b.text, ("text", None)
        else:
            if b.rows:  # 普通表格和图注网格都要处理
                for i, r in enumerate(b.rows):
                    for j, c in enumerate(r):
                        yield c, ("cell", (i, j))
            if b.kind == "image" and b.text:
                yield b.text, ("text", None)

    bigrams = Counter()
    for b in blocks:
        for t, _ in texts_of(b):
            for i in range(len(t) - 1):
                bigrams[t[i:i + 2]] += 1

    pair_map = {}
    for a, c in _CONFUSION:
        pair_map.setdefault(a, []).append(c)

    notes = []
    # 非词硬替换（错误占多数时词频投票会失效，这些组合在任何语境下都不是词）
    hard = {"酒漏": "洒漏"}
    from .vision_ocr import _normalize_ocr_text

    def fix(t: str) -> str:
        t = _normalize_ocr_text(t)  # 拼接环节可能再次引入数字空格等问题
        for bad, good in hard.items():
            t = t.replace(bad, good)
        out = list(t)
        for i, ch in enumerate(t):
            for repl in pair_map.get(ch, ()):
                olds = [t[max(0, i - 1):i + 1], t[i:i + 2]]
                news = [o.replace(ch, repl, 1) for o in olds]
                so = max((bigrams[o] for o in olds if len(o) == 2), default=0)
                sn = max((bigrams[n] for n in news if len(n) == 2), default=0)
                if sn >= 3 and sn >= 3 * max(so, 1):
                    out[i] = repl
                    notes.append(f"词频纠错: {t[max(0,i-1):i+2]} -> "
                                 f"{''.join(out[max(0,i-1):i+2])}")
                    break
        return "".join(out)

    for b in blocks:
        if b.kind in ("para", "heading"):
            b.text = fix(b.text)
        else:
            if b.rows:
                b.rows = [[fix(c) for c in r] for r in b.rows]
            if b.kind == "image" and b.text:
                b.text = fix(b.text)
    return notes


def qa_scan(blocks: List[Block]) -> Tuple[List[str], int]:
    """全文复读：返回 (问题列表, 可疑块数)。"""
    issues = []
    n_flag = 0
    for i, blk in enumerate(blocks):
        texts = []
        if blk.kind in ("para", "heading"):
            texts.append(blk.text)
        elif blk.kind == "table" and blk.rows:
            texts.extend(c for r in blk.rows for c in r)
        for t in texts:
            s = garbled_score(t)
            if s >= 0.5 and len(t) >= 2:
                issues.append(f"块{i} 疑似乱码({s:.0%}): {t[:40]}")
                if "garbled" not in blk.flags:
                    blk.flags.append("garbled")
        if "low_confidence" in blk.flags:
            n_flag += 1
            issues.append(f"块{i} 低置信度({blk.confidence:.2f}): "
                          f"{(blk.text or '')[:40]}")
    return issues, n_flag
