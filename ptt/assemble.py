"""把 OCR 行重组为结构化块：去页眉页脚/水印 → 行聚类 → 表格/标题/段落。

针对"长截图拼接式 PDF"（飞书/钉钉导出）做了专门处理：
- 周期性页眉页脚带检测：跨"虚拟页"规律重复的行（文档编号、防泄密声明、
  logo、页码）整带清除，正文中偶然重复的内容不受影响。
- 图片/表格区域检测在清除之前的完整行集上做覆盖判断，避免把页边空白
  当成插图。
"""
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .models import Block, Line
from .normalize import (clean_table_noise_rows, is_noise_text,
                        looks_truncated, table_suspect_score)
from .vision_ocr import StripProvider, ocr_strip

# 仅由引导点/装饰符组成的行（目录里的 ……… 之类）
_LEADER_RE = re.compile(r'^[\s.。·•．…⋯\-_—－]+$')
_DIGITS_RE = re.compile(r'^\d{1,4}$')
# 中文编号标题
_PAT_CN = re.compile(r'^[（(]?[一二三四五六七八九十]{1,3}[、）)．.]')
_PAT_CHAPTER = re.compile(r'^第[一二三四五六七八九十百\d]{1,3}[章节条部分]')
_PAT_NUM = re.compile(r'^\d{1,2}([.．]\d{1,2}){0,3}\s*[、.．\s]')
# 列表项起始（强制另起一段）
_LIST_START = re.compile(
    r'^(\d{1,2}[.．]\d{1,2}|[a-zA-Z][.、)）]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫]|[•·▪]|[（(]\d{1,2}[)）])')
_TOC_ENTRY = re.compile(r'.+?[.。·•．…⋯\-_\s]*\d{1,3}$')
# 纯数学碎片（分式被 OCR 拆出的分子/分母，如 "P"、"W+P"、"C+5*Y"）
_MATH_FRAG = re.compile(r'^[A-Za-z0-9+\-*/×÷().,%\s≤≥<>＋－＊]+$')
_ACCOUNT_WATERMARK = re.compile(r'(?i)(?:bm)?[a-z]{3,}[a-z0-9_]*\d{1,}')
_COMMON_CN_NOISE_WORDS = ("美团", "配送", "标准", "站点", "管理", "合作", "商",
                          "申诉", "检核", "区域", "规则", "内容")


def _norm(text: str) -> str:
    return re.sub(r'\s+', '', text)


def _fuzzy(text: str) -> str:
    """只留字母数字汉字——锚点比对要容忍 OCR 标点噪声。"""
    return re.sub(r'[^0-9A-Za-z一-鿿]', '', text)


# ---------------- 周期性页眉页脚带 ----------------

class PageBands:
    def __init__(self, bands: List[Tuple[float, float]], anchors: set):
        self.bands = bands          # 清除带 [(y0,y1)...]
        self.anchor_keys = anchors  # 锚点文本（归一化）

    def in_band(self, cy: float) -> bool:
        return any(a <= cy <= b for a, b in self.bands)


def detect_page_bands(lines: List[Line], page_h: float,
                      hlines: Optional[List["HLine"]] = None
                      ) -> Tuple[PageBands, List[str]]:
    """找跨虚拟页周期性重复的行（页眉/页脚锚点），生成清除带。

    清除带从锚点向上下各扩 55px，但不越过表格水平线——紧贴页脚的
    表格末行不能被吞掉。
    """
    groups = defaultdict(list)
    for ln in lines:
        key = _fuzzy(ln.text)  # 模糊键：OCR 标点噪声不应拆散同一页眉
        if len(key) >= 4:
            groups[key].append(ln)
    anchors = set()
    band_src: List[Line] = []
    notes = []
    periods = []  # (出现次数, 周期)
    for key, grp in groups.items():
        if len(grp) < 3:
            continue
        xs = [g.x0 for g in grp]
        if max(xs) - min(xs) > 60:
            continue
        ys = sorted(g.cy for g in grp)
        spread = ys[-1] - ys[0]
        if spread < page_h * 0.3:
            continue
        gaps = np.diff(ys)
        if len(gaps) and (gaps.min() < 500 or gaps.std() / max(gaps.mean(), 1) > 0.45):
            continue
        anchors.add(key)
        band_src.extend(grp)
        if len(gaps):
            periods.append((len(grp), float(np.median(gaps))))
        notes.append(f"{grp[0].text[:28]}（{len(grp)}次）")

    watermark_keys, watermark_notes = _detect_repeated_watermarks(groups, page_h)
    anchors.update(watermark_keys)
    notes.extend(watermark_notes)

    # 第二遍：已知页周期 P 后，间隔为 P 整数倍的重复行也是锚点
    # （页眉偶尔与 logo 误读黏成一行时，单独出现的次数会变少且间隔不均）
    P = max(periods)[1] if periods else 0  # 取出现最多的锚点组的周期
    if P > 500:
        for key, grp in groups.items():
            if key in anchors or len(grp) < 3:
                continue
            xs = [g.x0 for g in grp]
            if max(xs) - min(xs) > 60:
                continue
            ys = sorted(g.cy for g in grp)
            gaps = np.diff(ys)
            if len(gaps) and all(
                    min(g % P, P - (g % P)) < 70 for g in gaps) and gaps.min() > 500:
                anchors.add(key)
                band_src.extend(grp)
                notes.append(f"{grp[0].text[:28]}（{len(grp)}次,周期对齐）")
    # 按周期补全：某页的锚点若与 logo 黏成一行（文本不同），该页就缺一个
    # 清除带，这里按周期 P 把每个锚点带补到所有页上
    synth: List[Tuple[float, float]] = []
    if P > 500:
        by_key = defaultdict(list)
        for ln in band_src:
            by_key[_fuzzy(ln.text)].append(ln)
        for key, grp in by_key.items():
            grp.sort(key=lambda l: l.y0)
            ys = [l.y0 for l in grp]
            h = max(l.h for l in grp)
            base = ys[0] % P
            k = 0
            while base + k * P < page_h:
                yk = base + k * P
                k += 1
                if any(abs(yk - y) < 120 for y in ys):
                    continue
                synth.append((yk, yk + h))

    hys = sorted(h.y for h in hlines) if hlines else []
    bands = []
    for y0, y1 in ([(ln.y0, ln.y1) for ln in band_src] + synth):
        a, b = y0 - 55, y1 + 55
        lows = [y for y in hys if y <= y0]
        if lows:
            a = max(a, lows[-1] + 3)
        highs = [y for y in hys if y >= y1]
        if highs:
            b = min(b, highs[0] - 3)
        bands.append((a, b))
    bands.sort()
    merged: List[List[float]] = []
    for a, b in bands:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return PageBands([tuple(m) for m in merged], anchors), notes


def _detect_repeated_watermarks(groups, page_h: float) -> Tuple[set, List[str]]:
    """识别斜向/散落账号水印。

    这类水印不在固定页眉页脚带内，位置会横跨页面；只有账号形态或由账号
    推导出的短姓名/前缀在页面大范围重复时才删除。
    """
    keys = set()
    notes: List[str] = []
    account_roots = set()
    name_candidates = defaultdict(int)

    def spread_ok(grp, min_y=0.12) -> bool:
        if len(grp) < 3:
            return False
        ys = [g.cy for g in grp]
        return max(ys) - min(ys) > page_h * min_y

    for key, grp in groups.items():
        if len(grp) < 3 or not spread_ok(grp):
            continue
        if _ACCOUNT_WATERMARK.search(key):
            keys.add(key)
            m = _ACCOUNT_WATERMARK.search(key)
            root = re.sub(r'\d+$', '', m.group(0)) if m else key
            if len(root) >= 4:
                account_roots.add(root.lower())
            sample = grp[0].text[:24]
            notes.append(f"疑似账号水印 {sample}（{len(grp)}次）")
            for ln in grp:
                for run in re.findall(r'[一-鿿]{2,4}', ln.text):
                    if not any(w in run for w in _COMMON_CN_NOISE_WORDS):
                        name_candidates[_fuzzy(run)] += 1

    if not keys:
        return keys, notes

    for key, grp in groups.items():
        if key in keys or len(grp) < 3 or not spread_ok(grp):
            continue
        lower = key.lower()
        if len(lower) >= 4 and any(root.startswith(lower) or lower.startswith(root)
                                   for root in account_roots):
            keys.add(key)
            notes.append(f"疑似账号水印片段 {grp[0].text[:24]}（{len(grp)}次）")
            continue
        if name_candidates.get(key, 0) >= 2 and len(grp) >= 3:
            keys.add(key)
            notes.append(f"疑似姓名水印 {grp[0].text[:24]}（{len(grp)}次）")
    return keys, notes


def strip_header_footer(lines: List[Line], bands: PageBands
                        ) -> Tuple[List[Line], int]:
    """清除：锚点行（全部）+ 清除带内的短行/页码/装饰行（logo 误读、页码）。"""
    kept, removed = [], 0
    for ln in lines:
        key = _norm(ln.text)
        fkey = _fuzzy(ln.text)
        # 锚点本身，或行内黏连了锚点文本（logo 误读 + 文档编号连成一行）
        if fkey in bands.anchor_keys or any(
                len(a) >= 8 and a in fkey for a in bands.anchor_keys):
            removed += 1
            continue
        if bands.in_band(ln.cy) and (
                len(key) <= 14 or ln.conf < 0.45
                or _DIGITS_RE.match(key) or _LEADER_RE.match(ln.text)):
            removed += 1
            continue
        kept.append(ln)
    return kept, removed


def drop_noise(lines: List[Line]) -> List[Line]:
    return [l for l in lines
            if not _LEADER_RE.match(l.text) and not is_noise_text(l.text)]


def clean_leaders(text: str) -> str:
    """目录行：把引导点压成全角空格；去掉纯尾部引导点。"""
    text = re.sub(r'\s*[.。·•．…⋯\-_]{2,}\s*(\d{1,3})\s*$', r'　\1', text)
    text = re.sub(r'[.。·•．…⋯\-_]{3,}\s*$', '', text)
    return text.strip()


def cluster_rows(lines: List[Line]) -> List[List[Line]]:
    """按竖直重叠把行聚成"视觉行"（同一水平行上的多段文字）。"""
    rows: List[List[Line]] = []
    for ln in sorted(lines, key=lambda l: l.y0):
        placed = False
        if rows:
            last = rows[-1]
            cy = sum(l.cy for l in last) / len(last)
            h = max(l.h for l in last)
            if abs(ln.cy - cy) < h * 0.6:
                last.append(ln)
                placed = True
        if not placed:
            rows.append([ln])
    for r in rows:
        r.sort(key=lambda l: l.x0)
    return rows


# ---------------- 表格检测（基于表格线） ----------------

def _runs_to_centers(mask: np.ndarray, merge_gap: int = 4) -> List[int]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    centers = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            centers.append((start + prev) // 2)
            start = i
        prev = i
    centers.append((start + prev) // 2)
    return centers


class HLine:
    def __init__(self, y, x0, x1):
        self.y, self.x0, self.x1 = y, x0, x1


class TableRegion:
    def __init__(self, y0, y1, hy, vx, x0, x1):
        self.y0, self.y1 = y0, y1
        self.hy = hy  # 绝对 y
        self.vx = vx
        self.x0, self.x1 = x0, x1


def _ruling_mask(img: Image.Image, dark: int = 232) -> np.ndarray:
    """表格线/底纹像素（灰度暗于阈值）。色块与细线在几何上再区分。

    阈值取 232：要兼容浅灰色边框的表格；误检由"长连续段"几何约束兜底。
    """
    return np.asarray(img.convert("L")) < dark


def _max_run(row: np.ndarray, merge_gap: int = 3):
    """一行布尔数组里最长的连续 True 段，返回 (长度, x0, x1)。"""
    idx = np.where(row)[0]
    if len(idx) == 0:
        return 0, 0, 0
    splits = np.where(np.diff(idx) > merge_gap)[0]
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [len(idx) - 1]))
    lens = idx[ends] - idx[starts]
    k = int(np.argmax(lens))
    return int(lens[k]), int(idx[starts[k]]), int(idx[ends[k]])


def filter_hlines_by_text(hlines: List[HLine], lines: List[Line]) -> List[HLine]:
    """剔除从文字中间穿过的伪水平线（彩色底纹里的 JPEG 噪声会产生线状条带，
    真正的表格边框不会与文字相交）。"""
    out = []
    for h in hlines:
        crossed = any(l.y0 + 4 < h.y < l.y1 - 4
                      and l.x0 < h.x1 and l.x1 > h.x0 for l in lines)
        if not crossed:
            out.append(h)
    return out


def _detect_hlines(provider: StripProvider, chunk_h: int = 4000,
                   dark: int = 232) -> List[HLine]:
    """检测水平表格线（支持窄表）：找每行最长连续暗线段。"""
    H, W = provider.height, provider.width
    min_len = max(220, int(W * 0.18))
    raw: List[HLine] = []
    overlap = 40
    s = 0
    while s < H:
        e = min(s + chunk_h, H)
        img = provider.get_strip(s, e)
        scale = img.width / W
        mask = _ruling_mask(img)
        counts = mask.sum(axis=1)
        for y in np.where(counts >= min_len * scale)[0]:
            ln, x0, x1 = _max_run(mask[y])
            if ln >= min_len * scale:
                raw.append(HLine(int(s + y / scale), int(x0 / scale), int(x1 / scale)))
        if e >= H:
            break
        s = e - overlap
    raw.sort(key=lambda h: h.y)
    # 相邻几像素同属一条粗线 -> 合并
    out: List[HLine] = []
    for h in raw:
        if out and h.y - out[-1].y <= 8 and _x_overlap(h, out[-1]) > 0.5:
            out[-1].x0 = min(out[-1].x0, h.x0)
            out[-1].x1 = max(out[-1].x1, h.x1)
        else:
            out.append(h)
    return out


def _x_overlap(a, b) -> float:
    inter = min(a.x1, b.x1) - max(a.x0, b.x0)
    return inter / max(1, min(a.x1 - a.x0, b.x1 - b.x0))


def _x_match(a, b) -> float:
    """重叠占较长线的比例：1.0 表示两条线 x 范围几乎一致。"""
    inter = min(a.x1, b.x1) - max(a.x0, b.x0)
    return inter / max(1, a.x1 - a.x0, b.x1 - b.x0)


def _vertical_bridge(provider: StripProvider, y0: int, y1: int,
                     x0: int, x1: int, dark: int = 232) -> bool:
    """两条水平线之间是否有贯通的垂直线（同一张表的高单元格）。"""
    if y1 - y0 < 8:
        return True
    img = provider.get_strip(y0 + 3, y1 - 3)
    scale = img.width / provider.width
    a, b = int(x0 * scale), int(x1 * scale) + 1
    seg = _ruling_mask(img)[:, a:b]
    if seg.size == 0:
        return False
    return bool((seg.mean(axis=0) > 0.55).any())


def _band_coverage(y0: float, y1: float, bands: PageBands) -> float:
    if y1 <= y0:
        return 1.0
    cov = 0.0
    for a, b in bands.bands:
        cov += max(0.0, min(b, y1) - max(a, y0))
    return cov / (y1 - y0)


def _gap_has_content(lines: List[Line], y0: float, y1: float,
                     bands: PageBands) -> bool:
    """两条水平线之间是否存在真正的正文（页眉页脚/页码/装饰除外）。"""
    for ln in lines:
        if not (y0 + 5 < ln.cy < y1 - 5):
            continue
        if bands.in_band(ln.cy):
            continue
        key = _fuzzy(ln.text)
        if key in bands.anchor_keys or _DIGITS_RE.match(_norm(ln.text)) \
                or _LEADER_RE.match(ln.text):
            continue
        if len(key) >= 2:
            return True
    return False


def find_table_regions(provider: StripProvider, bands: PageBands,
                       lines: Optional[List[Line]] = None,
                       dark: int = 232,
                       hlines: Optional[List[HLine]] = None) -> List[TableRegion]:
    """水平线 -> 表格区域。合并规则：
    间距小 / 区间内有贯通垂直线（高单元格） / 隔着分页带且中间无正文（跨页表格）。
    """
    lines = lines or []
    if hlines is None:
        hlines = _detect_hlines(provider, dark=dark)
    groups: List[List[HLine]] = []
    cur: List[HLine] = []
    for h in hlines:
        if cur:
            prev = cur[-1]
            gap = h.y - prev.y
            same = False
            if _x_overlap(h, prev) > 0.55:
                if gap < 100:  # 紧邻的线（粗边框/紧凑行距）
                    same = True
                elif gap < 2400 and _vertical_bridge(
                        provider, prev.y, h.y, max(h.x0, prev.x0),
                        min(h.x1, prev.x1), dark):
                    same = True
                elif (gap < 1400 and _x_match(h, prev) > 0.85
                      and _band_coverage(prev.y, h.y, bands) > 0.25
                      and not _gap_has_content(lines, prev.y, h.y, bands)):
                    same = True  # 跨分页带、x 范围一致、中间无正文的同表延续
            if not same:
                groups.append(cur)
                cur = []
        cur.append(h)
    if cur:
        groups.append(cur)

    out = []
    for grp in groups:
        if len(grp) < 3:
            continue
        y0, y1 = grp[0].y, grp[-1].y
        x0 = min(h.x0 for h in grp)
        x1 = max(h.x1 for h in grp)
        img = provider.get_strip(max(0, y0 - 5), min(provider.height, y1 + 5))
        scale = img.width / provider.width
        mask = _ruling_mask(img)[:, int(x0 * scale):int(x1 * scale) + 1]
        # 垂直边界判定：在某一"行"(相邻水平线之间)内贯通 >=80% 行高。
        # 细的贯通段是表格线；宽的贯通段是底纹色块，取其左右边缘当列边界。
        # 再要求出现在足够多的行里（排除汉字竖笔画的干扰）。
        hys = [h.y for h in grp]
        votes = np.zeros(mask.shape[1])
        nrows_counted = 0
        for i in range(len(hys) - 1):
            a = int((hys[i] - y0 + 5) * scale) + 3
            b = int((hys[i + 1] - y0 + 5) * scale) - 3
            if b - a < 8:
                continue
            band = mask[a:b]
            if band.sum() == 0:
                continue  # 跨页空白行不参与投票
            nrows_counted += 1
            colmask = band.mean(axis=0) > 0.8
            idx = np.where(colmask)[0]
            if len(idx) == 0:
                continue
            splits = np.where(np.diff(idx) > 2)[0]
            starts = np.concatenate(([0], splits + 1))
            ends = np.concatenate((splits, [len(idx) - 1]))
            for s_i, e_i in zip(starts, ends):
                ra, rb = idx[s_i], idx[e_i]
                if rb - ra <= 10:
                    votes[(ra + rb) // 2] += 1
                else:  # 底纹色块：左右边缘是单元格分界
                    votes[ra] += 1
                    votes[rb] += 1
        if nrows_counted == 0:
            continue
        need = max(1.0, nrows_counted * 0.4)
        # 聚类候选位置（±6px 内合并投票）
        vx = []
        pos = np.where(votes > 0)[0]
        cur, cur_v = [], 0.0
        for p in pos:
            if cur and p - cur[-1] > 6:
                if cur_v >= need:
                    vx.append(int(x0 + np.mean(cur) / scale))
                cur, cur_v = [], 0.0
            cur.append(p)
            cur_v += votes[p]
        if cur and cur_v >= need:
            vx.append(int(x0 + np.mean(cur) / scale))
        if len(vx) >= 2:
            out.append(TableRegion(y0, y1, hys, vx, x0, x1))
    return out


def build_table_block(region: TableRegion, lines: List[Line], page: int,
                      provider: Optional[StripProvider] = None,
                      bands: Optional[PageBands] = None
                      ) -> Tuple[Optional[Block], set]:
    """把落在表格区域内的 OCR 行按网格分配到单元格。"""
    hy, vx = region.hy, region.vx
    if len(hy) < 2 or len(vx) < 2:
        return None, set()
    nrows, ncols = len(hy) - 1, len(vx) - 1
    if ncols > 12 or nrows > 500:
        return None, set()

    used_ids = _table_line_ids(region, lines)
    line_rows, line_conf = _table_rows_from_lines(region, lines)
    rows, conf = line_rows, line_conf
    flags: List[str] = []

    line_score = table_suspect_score(line_rows)
    if provider is not None and _should_cell_ocr(region, lines, line_rows, line_score):
        cell_rows, cell_conf = _table_rows_from_cell_ocr(provider, region, bands)
        if cell_rows:
            cell_score = table_suspect_score(cell_rows)
            line_fill = _filled_cells(line_rows)
            cell_fill = _filled_cells(cell_rows)
            enough_cell_text = (not line_rows
                                or cell_fill >= max(2, int(line_fill * 0.35)))
            if (enough_cell_text and (cell_score <= line_score
                    or (line_score >= 3 and cell_fill >= max(2, int(line_fill * 0.45)))
                    or _has_cross_column_lines(region, lines))):
                rows, conf = cell_rows, min(line_conf, cell_conf)
                flags.append("cell_ocr_table")

    if not rows:
        return None, set()
    rows = clean_table_noise_rows(rows)
    if not rows:
        return None, set()
    final_score = table_suspect_score(rows)
    if final_score >= 3:
        flags.append("table_low_confidence")
        flags.append("table_fallback")
        blk = Block(kind="image", rows=rows, page=page, confidence=conf,
                    bbox=(region.x0, region.y0, region.x1, region.y1),
                    flags=flags)
        return blk, used_ids
    blk = Block(kind="table", rows=rows, page=page, confidence=conf,
                bbox=(region.x0, region.y0, region.x1, region.y1),
                flags=flags)
    return blk, used_ids


def _table_line_ids(region: TableRegion, lines: List[Line]) -> set:
    used = set()
    for ln in lines:
        if not (region.y0 - 6 <= ln.cy <= region.y1 + 6):
            continue
        if ln.x1 < region.x0 - 20 or ln.x0 > region.x1 + 20:
            continue
        used.add(id(ln))
    return used


def _table_rows_from_lines(region: TableRegion, lines: List[Line]
                           ) -> Tuple[List[List[str]], float]:
    hy, vx = region.hy, region.vx
    nrows, ncols = len(hy) - 1, len(vx) - 1
    grid = [[[] for _ in range(ncols)] for _ in range(nrows)]
    n_used = 0
    conf = 1.0
    for ln in lines:
        if not (region.y0 - 5 <= ln.cy <= region.y1 + 5):
            continue
        cx = (ln.x0 + ln.x1) / 2
        if not (region.x0 - 20 <= cx <= region.x1 + 20):
            continue  # 窄表旁边的正文不属于表格
        ri = _bucket(ln.cy, hy)
        ci = _bucket(cx, vx)
        if ri is None or ci is None:
            continue
        grid[ri][ci].append(ln)
        n_used += 1
        conf = min(conf, ln.conf)
    if n_used < 2:
        return [], 1.0
    rows = []
    for r in grid:
        row = []
        for cell in r:
            cell.sort(key=lambda l: (l.y0, l.x0))
            row.append(_join_cell(cluster_rows(cell)) if cell else "")
        rows.append(row)
    return _clean_table_rows(rows), conf


def _clean_table_rows(rows: List[List[str]]) -> List[List[str]]:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    # 丢弃全空的列（底纹色块两条边缘会多出一条假分界线）
    keep = [j for j in range(ncol) if any(r[j].strip() for r in rows)]
    rows = [[r[j] for j in keep] for r in rows]
    if not rows or not rows[0]:
        return []
    # 缝合互补的拆裂行：文字骑在网格线上时一行会被拆成上下两半，
    # 仅当两行的非空单元格完全错开时合并（不会误并正常的续行）
    i = 0
    while i + 1 < len(rows):
        a, b = rows[i], rows[i + 1]
        a_set = {j for j, c in enumerate(a) if c.strip()}
        b_set = {j for j, c in enumerate(b) if c.strip()}
        if a_set and b_set and not (a_set & b_set):
            rows[i:i + 2] = [[a[j] if j in a_set else b[j]
                              for j in range(len(a))]]
        else:
            i += 1
    return rows


def _filled_cells(rows: List[List[str]]) -> int:
    return sum(1 for r in rows for c in r if c.strip())


def _has_cross_column_lines(region: TableRegion, lines: List[Line]) -> bool:
    """Vision 有时会把相邻列识别成同一行，这是表格串列的强信号。"""
    if len(region.vx) < 3:
        return False
    for ln in lines:
        if not (region.y0 - 5 <= ln.cy <= region.y1 + 5):
            continue
        for x in region.vx[1:-1]:
            if ln.x0 + 6 < x < ln.x1 - 6:
                return True
    return False


def _should_cell_ocr(region: TableRegion, lines: List[Line],
                     rows: List[List[str]], score: int) -> bool:
    nrows, ncols = len(region.hy) - 1, len(region.vx) - 1
    if nrows * ncols > 160 or ncols > 8:
        return False
    if not rows:
        return True
    if score >= 2 or _has_cross_column_lines(region, lines):
        return True
    # 窄列长文本表格容易把换行串到旁列，宁愿慢一点重扫单元格。
    col_widths = [region.vx[i + 1] - region.vx[i] for i in range(ncols)]
    return bool(rows and nrows >= 4 and ncols >= 3 and min(col_widths) < 0.16 * max(region.x1 - region.x0, 1))


def _table_rows_from_cell_ocr(provider: StripProvider, region: TableRegion,
                              bands: Optional[PageBands] = None
                              ) -> Tuple[List[List[str]], float]:
    rows: List[List[str]] = []
    conf = 1.0
    for i in range(len(region.hy) - 1):
        y0 = max(0, int(region.hy[i]) + 2)
        y1 = min(provider.height, int(region.hy[i + 1]) - 2)
        if y1 - y0 < 8:
            rows.append([""] * (len(region.vx) - 1))
            continue
        strip = provider.get_strip(y0, y1)
        row = []
        for j in range(len(region.vx) - 1):
            x0 = max(0, int(region.vx[j]) + 2)
            x1 = min(provider.width, int(region.vx[j + 1]) - 2)
            if x1 - x0 < 10:
                row.append("")
                continue
            cell = strip.crop((x0, 0, x1, strip.height))
            if not _cell_has_ink(cell):
                row.append("")
                continue
            try:
                cell_lines = ocr_strip(cell, upscale_to=1800)
            except Exception:
                row.append("")
                conf = min(conf, 0.0)
                continue
            cell_lines = [
                l for l in cell_lines
                if re.search(r'[0-9A-Za-z一-鿿Σ∑]', l.text)
                and not is_noise_text(l.text)
                and not (bands and bands.in_band(y0 + l.cy))
            ]
            if not cell_lines:
                row.append("")
                continue
            conf = min(conf, *(l.conf for l in cell_lines))
            row.append(_join_cell(cluster_rows(cell_lines)))
        rows.append(row)
    return _clean_table_rows(rows), conf


def _cell_has_ink(img: Image.Image) -> bool:
    if img.width < 8 or img.height < 8:
        return False
    pad_x = min(3, max(0, img.width // 12))
    pad_y = min(3, max(0, img.height // 12))
    core = img.crop((pad_x, pad_y, img.width - pad_x, img.height - pad_y))
    arr = np.asarray(core.convert("L"))
    if arr.size == 0:
        return False
    dark = (arr < 185).mean()
    mid = ((arr >= 185) & (arr < 235)).mean()
    return bool(dark > 0.0015 or (dark > 0.0006 and mid > 0.01))


def _bucket(v: float, edges: List[int]) -> Optional[int]:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return None


_CELL_BREAK = re.compile(
    r'^(\d{1,2}\.|[a-zA-Z][.、)）]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫]|[•·▪]|'
    r'[（(]\d{1,2}[)）]|场景[一二三四五六]|条件[一二三四五六]|情形[一二三四五六])')


def _join_cell(rows: List[List[Line]]) -> str:
    """单元格内多行合并：列表式条目前换行，其余直接连接。"""
    out = ""
    for row in rows:
        seg = (" ".join(l.text for l in row) if len(row) > 1
               else row[0].text).strip()
        if not out:
            out = seg
        elif _CELL_BREAK.match(seg):
            out += "\n" + seg
        elif out[-1:].isascii() and out[-1:] not in "，。；：、）)" and seg[:1].isascii():
            out += " " + seg
        else:
            out += seg
    return out


def _join_lines(rows: List[List[Line]]) -> str:
    """多行合并为一段：中文直接连接，ASCII 交界处补空格，同行多段用空格。"""
    parts = []
    for row in rows:
        seg = " ".join(l.text for l in row) if len(row) > 1 else row[0].text
        parts.append(seg.strip())
    out = ""
    for p in parts:
        if not out:
            out = p
        elif out[-1:].isascii() and out[-1:] not in "，。；：、）)" and p[:1].isascii():
            out += " " + p
        else:
            out += p
    return out


def structure_caption(cap_lines: List[Line]
                      ) -> Tuple[List[str], Optional[List[List[str]]], List[str]]:
    """把图示里的 OCR 文字按几何结构重建：返回 (图题行, 列对齐网格, 散行)。

    图示（架构图/流程图/金字塔）的标签天然按列分布，把 x 区间重叠的段聚成列、
    同一水平线的段对齐成行，输出表格远比按 y 排序的一坨文字可读。
    """
    rows_c = cluster_rows(cap_lines)
    if not rows_c:
        return [], None, []
    x_lo = min(l.x0 for l in cap_lines)
    x_hi = max(l.x1 for l in cap_lines)
    Wc = max(1.0, x_hi - x_lo)
    titles = []
    # 顶部的孤立单段行是图题
    while rows_c and len(rows_c[0]) == 1 and len(rows_c[0][0].text) <= 30:
        titles.append(rows_c[0][0].text.strip())
        rows_c = rows_c[1:]
    segs = [l for r in rows_c for l in r]
    if not segs:
        return titles, None, []
    # 跨多列的宽段（整行说明文字）不参与列聚类，单独成散行
    extras = [l for l in segs if l.x1 - l.x0 > 0.6 * Wc]
    segs = [l for l in segs if l.x1 - l.x0 <= 0.6 * Wc]
    flat_extras = [l.text.strip() for l in sorted(extras, key=lambda l: l.y0)]
    if not segs:
        return titles, None, flat_extras

    cols: List[List[float]] = []
    for l in sorted(segs, key=lambda l: l.x0):
        for c in cols:
            inter = min(c[1], l.x1) - max(c[0], l.x0)
            if inter > 0.3 * min(c[1] - c[0], l.x1 - l.x0):
                c[0] = min(c[0], l.x0)
                c[1] = max(c[1], l.x1)
                break
        else:
            cols.append([l.x0, l.x1])
    cols.sort()
    if not (2 <= len(cols) <= 6):
        flat = [_join_lines([r]) for r in rows_c]
        return titles, None, flat

    def col_of(l):
        best, bi = 0.0, 0
        for i, c in enumerate(cols):
            inter = min(c[1], l.x1) - max(c[0], l.x0)
            if inter > best:
                best, bi = inter, i
        return bi

    grid = []
    for r in rows_c:
        cells = [""] * len(cols)
        for l in r:
            if l in extras:
                continue
            ci = col_of(l)
            cells[ci] = (cells[ci] + " " + l.text).strip() if cells[ci] else l.text
        if any(cells):
            grid.append(cells)
    keepc = [j for j in range(len(cols)) if any(g[j] for g in grid)]
    if len(keepc) < 2 or not grid:
        flat = [_join_lines([r]) for r in rows_c]
        return titles, None, flat
    grid = [[g[j] for j in keepc] for g in grid]
    return titles, grid, flat_extras


# ---------------- 图片/图表区域检测 ----------------

def find_figure_regions(provider: StripProvider, coverage_lines: List[Line],
                        tables: List[TableRegion], bands: PageBands,
                        min_gap: int = 140, ink_thresh: float = 0.005
                        ) -> List[Tuple[int, int]]:
    """找出没有文字但有内容（图表/截图/公式）的竖直区段。

    覆盖判断用"清除前"的全部行 + 表格区 + 页眉页脚带，避免把被清掉的
    页边区域误当插图。
    """
    H = provider.height
    covered = [[ln.y0 - 8, ln.y1 + 8] for ln in coverage_lines]
    covered += [[t.y0, t.y1] for t in tables]
    covered += [list(b) for b in bands.bands]
    covered.sort()
    merged: List[List[float]] = []
    for a, b in covered:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps = []
    prev = 0.0
    for a, b in merged + [[H, H]]:
        if a - prev >= min_gap:
            gaps.append((int(prev), int(a)))
        prev = max(prev, b)
    figures = []
    for a, b in gaps:
        img = provider.get_strip(a, b)
        if (np.asarray(img.convert("L")) < 200).mean() >= ink_thresh:
            figures.append((a, b))
    return figures


# ---------------- 总装 ----------------

def assemble_blocks(provider: StripProvider, lines: List[Line], page: int = 0,
                    detect_tables: bool = True) -> Tuple[List[Block], List[str]]:
    """OCR 行 -> 结构化块列表（按 y 排序），返回 (块, 提示信息)。"""
    notes = []
    all_lines = list(lines)
    hlines = _detect_hlines(provider) if detect_tables else []
    hlines = filter_hlines_by_text(hlines, lines)
    bands, band_notes = detect_page_bands(lines, provider.height, hlines)
    tables = (find_table_regions(provider, bands, lines, hlines=hlines)
              if detect_tables else [])
    lines, n_removed = strip_header_footer(lines, bands)
    if band_notes:
        notes.append(f"已清除页眉/页脚/水印 {n_removed} 行: " + "；".join(band_notes[:6]))
    lines = drop_noise(lines)

    table_blocks = []
    in_table = set()
    for tr in tables:
        blk, used = build_table_block(tr, lines, page, provider, bands)
        if blk:
            table_blocks.append(blk)
            in_table |= used

    free_lines = [l for l in lines if id(l) not in in_table]
    figures = find_figure_regions(provider, all_lines, tables, bands)

    rows = cluster_rows(free_lines)

    # 图示/流程图的文字标签 -> 并入截图区。两类信号：
    # ① 稀疏多段行（一行被切成多个互相远离的短段）
    # ② 短标签行聚类（连续多行"短、窄、远离正文左边距"，如金字塔/流程图标注）
    W0 = provider.width
    wide_x0s = [min(l.x0 for l in r) for r in rows
                if max(l.x1 for l in r) - min(l.x0 for l in r) > 0.5 * W0]
    margin = float(np.median(wide_x0s)) if wide_x0s else 0.0

    def _is_heading_row(text: str) -> bool:
        return bool(_PAT_CHAPTER.match(text) or _PAT_CN.match(text)
                    or (_PAT_NUM.match(text) and len(text) <= 30))

    diag_ids = set()

    # 分式公式优先于图示聚类：含 "=" 的行上下紧挨着孤立的短数学碎片
    # （分子/分母被 OCR 拆开），线性化必然失真 -> 整体截图
    formula_bands = []
    row_spans = [(min(l.y0 for l in r), max(l.y1 for l in r),
                  max(l.h for l in r), _join_lines([r])) for r in rows]
    for i, (y0, y1, h, text) in enumerate(row_spans):
        if "=" not in text and "＝" not in text:
            continue
        frags = []
        for j, (fy0, fy1, fh, ft) in enumerate(row_spans):
            if j == i:
                continue
            if len(ft) <= 8 and _MATH_FRAG.match(ft) and (
                    -0.5 * h <= fy0 - y1 <= 1.8 * h or
                    -0.5 * h <= y0 - fy1 <= 1.8 * h):
                frags.append(j)
        if frags:
            a = min(y0, *(row_spans[j][0] for j in frags))
            b = max(y1, *(row_spans[j][1] for j in frags))
            formula_bands.append((a - 4, b + 4))
            diag_ids.update(id(rows[j][0]) for j in frags)
            diag_ids.update(id(l) for l in rows[i])

    diag_bands = []
    label_flags = []  # (y0, y1, row, is_sparse, is_label)
    for row in rows:
        if id(row[0]) in diag_ids:
            label_flags.append(None)
            continue
        text = _join_lines([row])
        # 目录行（标题…页码）和标题行不是图示标签
        if (_TOC_ENTRY.match(_norm(text)) or re.search(r'[.。·•．…⋯]{4,}', text)
                or _is_heading_row(text)):
            label_flags.append(None)
            continue
        x0 = min(l.x0 for l in row)
        span = max(l.x1 for l in row) - x0
        tw = sum(l.x1 - l.x0 for l in row)
        sparse = (len(row) >= 3 and span > 0
                  and tw / span < (0.7 if len(row) >= 4 else 0.55))
        label = (len(_norm(text)) <= 12 and span < 0.3 * W0
                 and x0 > margin + 60)
        if sparse:
            diag_bands.append((min(l.y0 for l in row), max(l.y1 for l in row)))
            diag_ids.update(id(l) for l in row)
        label_flags.append((min(l.y0 for l in row), max(l.y1 for l in row),
                            row, sparse, label))
    # 短标签聚类：相邻（<260px）的标签行 >=3 行，或含稀疏行 -> 图示带
    cluster = []
    for item in label_flags + [None]:
        hit = item is not None and (item[3] or item[4])
        if hit and (not cluster or item[0] - cluster[-1][1] < 260):
            cluster.append(item)
            continue
        if len(cluster) >= 3 or (cluster and any(c[3] for c in cluster)):
            diag_bands.append((cluster[0][0], cluster[-1][1]))
            for c in cluster:
                diag_ids.update(id(l) for l in c[2])
        cluster = [item] if hit else []

    def _merge_regions(regs):
        """合并相邻区域，但不得跨越仍留在正文里的行（标题/段落是屏障）。"""
        barriers = sorted(min(l.y0 for l in r) + max(l.h for l in r) / 2
                          for r in rows)
        regs = sorted(regs)
        out_r = []
        for a, b in regs:
            if out_r and a - out_r[-1][1] <= 160 and not any(
                    out_r[-1][1] - 5 < y < a + 5 for y in barriers):
                out_r[-1] = (out_r[-1][0], max(out_r[-1][1], b))
            else:
                out_r.append((a, b))
        return out_r

    if diag_bands or formula_bands:
        rows = [r for r in rows if id(r[0]) not in diag_ids]
        figures = _merge_regions(figures + diag_bands)
        # 紧贴图示区的零散行（半截标签/稀疏多段/低置信碎片）也吸收进去
        changed = True
        while changed:
            changed = False
            for r in rows:
                text = _join_lines([r])
                if _is_heading_row(text):
                    continue  # 标题永远留在正文
                ry0 = min(l.y0 for l in r)
                ry1 = max(l.y1 for l in r)
                span = max(l.x1 for l in r) - min(l.x0 for l in r)
                tw = sum(l.x1 - l.x0 for l in r)
                sparse = len(r) >= 2 and span > 0 and tw / span < 0.6
                lowconf = min(l.conf for l in r) < 0.45
                if not (len(_norm(text)) <= 10 or sparse or lowconf):
                    continue
                for i, (a, b) in enumerate(figures):
                    if ry1 > a - 90 and ry0 < b + 90:
                        figures[i] = (min(a, ry0), max(b, ry1))
                        rows.remove(r)
                        diag_ids.update(id(l) for l in r)
                        changed = True
                        break
                if changed:
                    break
        figures = _merge_regions(figures)
        # 公式带如与图示区重叠则并入图示，否则单独成块（不附图中文字）
        formula_bands = _merge_regions(formula_bands)
        standalone_formula = []
        for fa, fb in formula_bands:
            hit = False
            for i, (a, b) in enumerate(figures):
                if fb > a - 60 and fa < b + 60:
                    figures[i] = (min(a, fa), max(b, fb))
                    hit = True
                    break
            if not hit:
                standalone_formula.append((fa, fb))
        figures = _merge_regions(figures)
    else:
        standalone_formula = []

    heights = [max(l.h for l in r) for r in rows]
    med_h = float(np.median(heights)) if heights else 20
    W = provider.width

    blocks: List[Block] = list(table_blocks)
    kept_ids = {id(l) for r in rows for l in r}
    for (a, b) in figures:
        # 图中文字：只收录被并入截图的行，不与正文重复；按几何结构重建
        cap = [l for l in free_lines
               if a <= l.cy <= b and l.conf >= 0.3 and id(l) not in kept_ids
               and re.search(r'[0-9A-Za-z一-鿿]', l.text)]  # 纯符号(箭头等)不进图注
        titles, grid, flat = structure_caption(cap) if cap else ([], None, [])
        blocks.append(Block(kind="image", bbox=(0, a, W, b), page=page,
                            text="\n".join(titles + flat), rows=grid))
    for (a, b) in standalone_formula:
        cap = [l for l in free_lines
               if a <= l.cy <= b and l.conf >= 0.25
               and re.search(r'[0-9A-Za-z一-鿿=＝+*/()（）<>≤≥]', l.text)]
        text = "\n".join(l.text for l in sorted(cap, key=lambda l: (l.y0, l.x0)))
        blocks.append(Block(kind="image", bbox=(0, a, W, b), page=page,
                            text=text, flags=["formula"]))

    paras = _rows_to_paragraphs(rows, med_h, W, page)
    blocks.extend(paras)
    blocks = _merge_table_fallback_fragments(blocks, W)
    blocks.sort(key=lambda b: b.bbox[1])
    return blocks, notes


def _merge_table_fallback_fragments(blocks: List[Block],
                                    page_w: float) -> List[Block]:
    """把紧邻低置信表格截图的表格残片一起保留为截图。

    长截图式 PDF 里，一张逻辑表有时被检测成多个表格区域，中间夹出
    少量"场景/金额/包括但不限于"之类的残片。如果这些残片仍作为正文
    输出，阅读者会误以为它们是可信文字；并入截图更保真。
    """
    ordered = sorted(blocks, key=lambda b: (b.page, b.bbox[1]))
    out: List[Block] = []
    i = 0
    while i < len(ordered):
        blk = ordered[i]
        if not _is_table_fallback_block(blk):
            out.append(blk)
            i += 1
            continue

        run = [blk]
        saw_fragment = False
        last = blk
        j = i + 1
        while j < len(ordered) and ordered[j].page == blk.page:
            nxt = ordered[j]
            gap = nxt.bbox[1] - last.bbox[3]
            if _is_table_fallback_block(nxt):
                if gap > 900 and not saw_fragment:
                    break
                run.append(nxt)
                last = nxt
                j += 1
                continue
            if gap <= 1200 and _is_table_fragment_block(nxt):
                run.append(nxt)
                saw_fragment = True
                last = nxt
                j += 1
                continue
            break

        if saw_fragment:
            x0 = 0
            y0 = min(b.bbox[1] for b in run)
            x1 = page_w
            y1 = max(b.bbox[3] for b in run)
            merged_rows: List[List[str]] = []
            merged_texts: List[str] = []
            for part in run:
                if part.rows:
                    merged_rows.extend(part.rows)
                elif part.text:
                    merged_rows.append(["内容", part.text])
                text = _flatten_block_text(part).strip()
                if text:
                    merged_texts.append(text)
            flags = ["table_low_confidence", "table_fallback",
                     "merged_table_fallback"]
            out.append(Block(kind="image", page=blk.page,
                             text="\n".join(merged_texts),
                             rows=merged_rows or None,
                             confidence=min(b.confidence for b in run),
                             bbox=(x0, y0, x1, y1), flags=flags))
        else:
            out.append(blk)
        i = j
    fallback_refs = [b for b in out if _is_table_fallback_block(b)]
    final: List[Block] = []
    for blk in out:
        if (_is_table_fragment_block(blk)
                and _near_table_fallback(blk, fallback_refs)
                and not _is_table_fallback_block(blk)):
            rows = blk.rows
            if not rows and blk.text:
                rows = [["内容", blk.text]]
            final.append(Block(kind="image", page=blk.page,
                               text=_flatten_block_text(blk),
                               rows=rows,
                               confidence=blk.confidence,
                               bbox=(0, blk.bbox[1], page_w, blk.bbox[3]),
                               flags=["table_low_confidence", "table_fallback",
                                      "fragment_table_fallback"]))
        else:
            final.append(blk)
    return final


def _near_table_fallback(blk: Block, refs: List[Block]) -> bool:
    for ref in refs:
        if ref.page != blk.page:
            continue
        gap = max(ref.bbox[1] - blk.bbox[3], blk.bbox[1] - ref.bbox[3], 0)
        if gap <= 2400:
            return True
    return False


def _is_table_fallback_block(blk: Block) -> bool:
    return blk.kind == "image" and "table_fallback" in blk.flags


def _flatten_block_text(blk: Block) -> str:
    if blk.rows:
        return " ".join(" ".join(c for c in row if c) for row in blk.rows)
    return blk.text or ""


def _is_table_fragment_block(blk: Block) -> bool:
    if blk.kind not in ("para", "heading", "table", "image"):
        return False
    text = _flatten_block_text(blk).strip()
    compact = _norm(text)
    if len(compact) < 4:
        return False
    if compact.startswith("案例说明"):
        return False
    if blk.kind == "image":
        return bool((blk.rows and table_suspect_score(blk.rows) >= 2)
                    or (blk.rows and _table_fragment_signal(text) >= 2)
                    or _table_fragment_signal(text) >= 3)
    if blk.kind == "table":
        rows = blk.rows or []
        if table_suspect_score(rows) >= 2:
            return True
        return bool(re.search(r'场景\s*\d|场景[一二三四五六]', text)
                    and re.search(r'\d+\s*元\s*/', text))

    score = _table_fragment_signal(text)
    if blk.kind == "heading" and score < 3:
        return False
    return score >= 3


def _table_fragment_signal(text: str) -> int:
    compact = _norm(text)
    score = 0
    if re.search(r'\d+\s*元\s*/(?:站|人|项|次|月|天)', text):
        score += 2
    if re.search(r'场景\s*\d|场景[一二三四五六]|餐箱|消毒|看板|责任承担|'
                 r'承担违约|整改|不达标|虚假', text):
        score += 1
    if re.search(r'标准站系统|基础建设', text):
        score += 3
    if re.search(r'整改不达标|双倍违约金|烟感状态|'
                 r'功能区|配置|卧室|客厅|承担', text):
        score += 2
    if re.search(r'卧室|客厅|站点各功能区|明火|大功率电器|小太阳|'
                 r'电丝炉|禁止存放', text):
        score += 2
    if "烟感状态" in text and "双倍违约金" in text:
        score += 2
    if "包括但不限于" in text:
        score += 2
    if looks_truncated(text):
        score += 1
    if re.match(r'^\d{1,2}[.．]\s*包括但不限于', text):
        score += 3
    if re.search(r'\d{3,5}\s*$', compact) and re.search(
            r'审核|判定|处置|提交|违约|承担', compact):
        score += 2
    return score


def _rows_to_paragraphs(rows: List[List[Line]], med_h: float, page_w: float,
                        page: int) -> List[Block]:
    blocks: List[Block] = []
    para: List[List[Line]] = []
    toc_mode = False
    toc_miss = 0

    def flush():
        nonlocal toc_mode
        if not para:
            return
        text = clean_leaders(_join_lines(para))
        if not text:
            para.clear()
            return
        h = max(l.h for r in para for l in r)
        conf = min(l.conf for r in para for l in r)
        y0 = min(l.y0 for r in para for l in r)
        y1 = max(l.y1 for r in para for l in r)
        x0 = min(l.x0 for r in para for l in r)
        x1 = max(l.x1 for r in para for l in r)
        if toc_mode:
            kind, level = "para", 0
        else:
            kind, level = _classify_heading(text, h, med_h, len(para), x1 - x0, page_w)
        blocks.append(Block(kind=kind, text=text, level=level, page=page,
                            bbox=(x0, y0, x1, y1), confidence=conf))
        para.clear()

    prev_row = None
    for row in rows:
        text = _join_lines([row])
        ntext = _norm(text)
        y0 = min(l.y0 for l in row)
        h = max(l.h for l in row)
        width = max(l.x1 for l in row) - min(l.x0 for l in row)

        # 目录模式进出
        if ntext in ("目录", "目錄"):
            flush()
            toc_mode = True
            toc_miss = 0
            blocks.append(Block(kind="heading", text="目录", level=1, page=page,
                                bbox=(0, y0, page_w, max(l.y1 for l in row))))
            prev_row = row
            continue
        if toc_mode:
            if _DIGITS_RE.match(ntext):
                prev_row = row
                continue  # 目录里的孤立页码
            if _TOC_ENTRY.match(ntext) or _LEADER_RE.match(text):
                flush()
                para.append(row)
                flush()
                prev_row = row
                continue
            toc_miss += 1
            if toc_miss >= 2:
                toc_mode = False

        is_heading = (_classify_heading(text, h, med_h, 1, width, page_w)[0]
                      == "heading")
        force_new = bool(_LIST_START.match(text))
        if prev_row is not None and para:
            prev_y1 = max(l.y1 for l in prev_row)
            if (y0 - prev_y1) > h * 1.25 or is_heading or force_new:
                flush()
        para.append(row)
        if is_heading:
            flush()
        prev_row = row
    flush()
    return blocks


def _classify_heading(text: str, h: float, med_h: float, nrows: int,
                      width: float, page_w: float) -> Tuple[str, int]:
    if nrows > 1 or len(text) > 40:
        return "para", 0
    if "=" in text or "＝" in text:  # 公式行绝不是标题
        return "para", 0
    if width > page_w * 0.72:  # 满宽行是段落续行，不是标题
        return "para", 0
    if text[-1:] in "，、；,;":
        return "para", 0
    has_colon = text[-1:] in "：:"  # 带冒号的行仅在有编号时算标题
    big = h > med_h * 1.35
    if _PAT_CHAPTER.match(text):
        return "heading", 1
    if _PAT_CN.match(text):
        return "heading", 2
    m = _PAT_NUM.match(text)
    if m and len(text) <= 30:
        head = m.group(0).rstrip('、.．\t ')
        depth = head.count('.') + head.count('．')
        return "heading", min(3 + depth, 4)
    if has_colon:
        return "para", 0
    if big and len(text) <= 24:
        return "heading", 1
    return "para", 0
