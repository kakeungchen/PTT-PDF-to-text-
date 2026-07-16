"""质检自查：识别乱码/低置信内容，复核重试，剩余问题显式标注。"""
import re
import unicodedata
from typing import List, Tuple

from .models import Block, Line
from .normalize import (
    _is_reliable_repaired_structured_table,
    is_noise_text,
    looks_truncated,
    table_suspect_score,
)
from .vision_ocr import StripProvider, ocr_strip

# 明确的乱码信号
_BAD_CHARS = re.compile(r'[�-]')  # 替换符 / 私用区
_CID_RE = re.compile(r'\(cid:\d+\)')
# 目录条目：引导点会拉低 OCR 置信度，但清洗后的文本本身没问题
_TOC_LIKE = re.compile(r'^.{1,40}[　.\s。·•．…⋯：:\-－—]\d{1,3}$')
_CLEAN_SECTION_HEADING = re.compile(
    r'^[一二三四五六七八九十]{1,3}[、.．]\s*'
    r'[0-9A-Za-z一-鿿（）()《》【】\s/-]{2,24}[.．。…⋯]?$')
_CLEAN_FORM_FIELD = re.compile(r'^\d{1,2}[.．]\s*站点ID[:：]\s*_?站点名称$')
# 数学公式信号：低置信且属于视觉公式时转入公式块；纯文本公式保留文本。
_MATH_RE = re.compile(r'[=≤≥×÷√∑∏∝≈≠＋＝]|[+*/^]\s*\d|\d\s*[+*/]')
_TEXT_FORMULA_RE = re.compile(r'^[\u4e00-\u9fffA-Za-z0-9（）()、/+＋*× _{}]+[=＝].{2,}$')
_WATERMARK_IN_TEXT = re.compile(
    r'(?i)(?:bm[_\\-]*ch|chenj|chenjia|i?ang[o0]l|cheny|陈加强|陈加\d?)')
_SUSPECT_OCR_TEXT = re.compile(r'[�俁仴雲抇讳冏昇銷埃門哭伿怯奂]')


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
            if _contains_watermark_noise(new_text):
                blk.flags.append("low_confidence")
                continue
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
        if _is_clean_low_confidence_text(blk.text):
            blk.confidence = max(blk.confidence, conf_thresh)
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
        group_text = "\n".join(b.text for b in group if b.text)
        if _looks_like_table_continuation_fragment(group_text):
            out.extend(group)
            group.clear()
            return
        if _looks_like_multiline_text_formula(group_text):
            out.extend(group)
            group.clear()
            return
        single = len(group) == 1
        if single and not _should_image_fallback_single(group[0]):
            out.extend(group)
            group.clear()
            return
        y0 = min(b.bbox[1] for b in group) - 8
        y1 = max(b.bbox[3] for b in group) + 8
        x0 = max(0, min(b.bbox[0] for b in group) - 60)
        x1 = min(provider.width, max(b.bbox[2] for b in group) + 60)
        text = group_text
        flags = ["auto_image"]
        if (any(_MATH_RE.search(b.text or "") for b in group)
                and not _looks_like_table_continuation_fragment(text)):
            flags.append("formula")
        blk = Block(kind="image", page=group[0].page, text=text,
                    bbox=(x0, max(0, y0), x1, min(provider.height, y1)),
                    flags=flags)
        notes.append(f"低置信区域已转为截图(y={int(y0)}): "
                     + " / ".join(b.text[:18] for b in group[:3]))
        out.append(blk)
        group.clear()

    for b in sorted(blocks, key=lambda b: b.bbox[1]):
        if b.kind in ("para", "heading") and _is_pure_noise_text(b.text):
            notes.append(f"移除疑似水印片段: {b.text[:18]}")
            continue
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


def _contains_watermark_noise(text: str) -> bool:
    return bool(is_noise_text(text) or _WATERMARK_IN_TEXT.search(text or ""))


def _is_clean_low_confidence_text(text: str) -> bool:
    s = (text or "").strip()
    if not s or garbled_score(s) > 0 or _SUSPECT_OCR_TEXT.search(s):
        return False
    if _contains_watermark_noise(s):
        return False
    if _TOC_LIKE.match(s):
        return True
    return bool(_CLEAN_SECTION_HEADING.match(s) or _CLEAN_FORM_FIELD.match(s))


def _is_pure_noise_text(text: str) -> bool:
    s = re.sub(r'\s+', '', text or "")
    if not s:
        return True
    if is_noise_text(s):
        return True
    if len(s) <= 28 and re.fullmatch(r'[#_()（）|\\/.·.\-A-Za-z0-9]+', s) \
            and re.search(r'(?i)ang|chen|bm|10901|1901', s):
        return True
    if len(s) <= 8 and re.fullmatch(r'[一二三四五六七八九十]?[A-Za-z0-9]{2,}', s):
        return True
    if len(s) <= 20 and _WATERMARK_IN_TEXT.search(s):
        return True
    return False


def _should_image_fallback_single(blk: Block) -> bool:
    text = blk.text or ""
    compact = re.sub(r'\s+', '', text)
    if _looks_like_text_formula(text):
        return False
    if _MATH_RE.search(text) and not _looks_like_text_formula(text):
        return True
    if garbled_score(text) > 0:
        return True
    if _SUSPECT_OCR_TEXT.search(text):
        return True
    if _contains_watermark_noise(text):
        return True
    if blk.confidence <= 0.35 and len(compact) >= 18:
        return True
    return False


def _looks_like_text_formula(text: str) -> bool:
    s = re.sub(r'\s+', '', text or "")
    if not s or "\n" in (text or ""):
        return False
    if not _TEXT_FORMULA_RE.match(s):
        return False
    if re.search(r"SUM|Σ|sqrt|\\frac|[上下]标", s, re.I):
        return False
    if len(re.findall(r"[=＝]", s)) > 1:
        return False
    # Plain business/rule formulas read left-to-right; dense variable-only
    # expressions are more likely to need LaTeX handling.
    chinese = len(re.findall(r"[\u4e00-\u9fff]", s))
    return chinese >= 4 and len(s) <= 160


def _looks_like_multiline_text_formula(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines or len(lines) > 4:
        return False
    compact = re.sub(r'\s+', '', "".join(lines))
    if len(compact) > 220 or len(compact) < 10:
        return False
    if len(re.findall(r"[=＝]", compact)) != 1:
        return False
    if re.search(r"SUM|Σ|\\frac|_\{|[A-Za-z]\s*/\s*[A-Za-z]", compact, re.I):
        return False
    chinese = len(re.findall(r"[\u4e00-\u9fff]", compact))
    if chinese < 8:
        return False
    return bool(re.search(r"[+＋]", compact))


def _looks_like_table_continuation_fragment(text: str) -> bool:
    """Detect table body fragments that contain formula-like symbols.

    Long policy/rule tables often contain definitions such as ``占比=...``.
    When a dense row is low-confidence, treating it as a visual formula creates
    a misleading review marker and can hide the table text.  These fragments
    usually mention repeated table entities and do not look like one centered
    standalone fraction.
    """
    raw = text or ""
    compact = re.sub(r"\s+", "", raw)
    if len(compact) < 60:
        return False
    visual_formula_terms = (
        "复合准时率", "配送原因未完成率", "承托比", "虚假点送达率",
        "复合超时时长", "站点组体验总得分", "特殊场景完成单占比",
        "客诉虚假点送达率",
    )
    if any(term in compact for term in visual_formula_terms):
        return False
    table_terms = (
        "站点", "骑手", "排班", "时段", "出勤", "合格", "考核",
        "班次", "烽火台", "目标值", "数据查询", "数据来源",
    )
    term_hits = sum(1 for term in table_terms if term in compact)
    bracket_hits = compact.count("【") + compact.count("】")
    column_gap_hits = len(re.findall(r"[^\n]{2,}\s{2,}[^\n]{2,}", raw))
    label_hits = len(re.findall(r"(?:数|率|值|占比|时段|骑手|站点)[】）)]?", compact))
    return term_hits >= 4 and (bracket_hits >= 3 or column_gap_hits >= 2 or label_hits >= 8)


def _is_safe_low_confidence_formula(text: str) -> bool:
    s = re.sub(r'\s+', '', text or "")
    if not s:
        return False
    if _looks_like_text_formula(text) or _looks_like_multiline_text_formula(text):
        return True
    # Long left-to-right scoring expressions are often low-confidence because
    # OCR sees dense symbols, but they are still readable text formulas rather
    # than visual fractions that require review.
    if ("得分" in s and "权重" in s and re.search(r"W[1-4]", s)
            and re.search(r"[=＝]", s)):
        return True
    if re.search(r"\d+%[*×]\d+", s) and re.search(r"[=＝]\d", s):
        return True
    return False


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
        if _is_protected_native_pdf_table(b):
            return
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
                if ch == "O":
                    prev = t[i - 1] if i > 0 else ""
                    nxt = t[i + 1] if i + 1 < len(t) else ""
                    if ((prev.isascii() and prev.isalpha())
                            or (nxt.isascii() and nxt.isalpha())):
                        continue
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
        if _is_protected_native_pdf_table(b):
            continue
        if b.kind in ("para", "heading"):
            b.text = fix(b.text)
        else:
            if b.rows:
                b.rows = [[fix(c) for c in r] for r in b.rows]
            if b.kind == "image" and b.text:
                b.text = fix(b.text)
    return notes


def _is_protected_native_pdf_table(blk: Block) -> bool:
    if "native_pdf_table" not in (blk.flags or []) or not blk.rows:
        return False
    joined = " ".join(" ".join(c for c in row if c) for row in blk.rows)
    if not joined.strip():
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
    if cjk > max(8, len(joined) * 0.08):
        return False
    terms = (
        "Model", "Overall", "Edit", "Metric", "Pages", "TPS",
        "DS-OCR", "Unlimited-OCR", "DeepSeek", "OCRVerse", "Nanonets",
    )
    return any(term in joined for term in terms)


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
            if _is_safe_low_confidence_formula(blk.text or ""):
                continue
            n_flag += 1
            issues.append(f"块{i} 低置信度({blk.confidence:.2f}): "
                          f"{(blk.text or '')[:40]}")
        verified_table = (
            "table_repaired_verified" in (blk.flags or [])
            or (blk.kind == "table"
                and _is_reliable_repaired_structured_table(blk))
        )
        if ("table_low_confidence" in blk.flags
                and not _is_protected_native_pdf_table(blk)
                and not verified_table):
            n_flag += 1
            issues.append(f"块{i} 表格疑似列错位，建议人工复核")
        elif (blk.kind == "table" and blk.rows and not verified_table
              and not _is_protected_native_pdf_table(blk)):
            score = table_suspect_score(blk.rows)
            if score >= 3:
                if "table_low_confidence" not in blk.flags:
                    blk.flags.append("table_low_confidence")
                n_flag += 1
                issues.append(f"块{i} 表格疑似列错位(score={score})，建议人工复核")
        if "possible_truncation" in blk.flags:
            n_flag += 1
            issues.append(f"块{i} 疑似截断: {(blk.text or '')[:50]}")
        elif blk.kind in ("para", "heading") and looks_truncated(blk.text):
            if "possible_truncation" not in blk.flags:
                blk.flags.append("possible_truncation")
            n_flag += 1
            issues.append(f"块{i} 疑似截断: {(blk.text or '')[:50]}")
    return issues, n_flag
