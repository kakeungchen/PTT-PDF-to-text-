"""文本型 PDF 提取：直接读文字层，过滤水印/页眉页脚，按目标格式处理图片。"""
import io
import os
import re
from collections import defaultdict
from contextlib import suppress
from typing import List, Tuple

import fitz
from PIL import Image

from .models import Block
from .vision_ocr import ocr_strip

_PAGE_NUM_RE = re.compile(r'^[-–—\s]*(第?\s*\d+\s*页?|\d+\s*/\s*\d+|[ivxIVX]+)[-–—\s]*$')
_IMAGE_FORMULA_TOKEN_RE = re.compile(
    r"[=＝≤≥×÷√∑Σ]|SUM|\\frac|"
    r"(?:[A-Za-z]\s*[/]\s*[A-Za-z])|(?:\d+\s*/\s*\d+)",
    re.I,
)
_IMAGE_FORMULA_LABEL_RE = re.compile(
    r"公式|计算口径|计算路径|率|占比|承托比|时长|得分|SUM|Σ|分子|分母"
)
_NUM_HEADING_RE = re.compile(r"^(\d{1,2}(?:[.．]\d{1,2}){0,4})[.．]?\s*\S+")
_CN_HEADING_RE = re.compile(r"^[一二三四五六七八九十]{1,3}[、.．]\s*\S+")
_CAPTION_START_RE = re.compile(r"^(?:Figure|Fig\.?|Table|图|表)\s*\d+\s*(?:[|:：.]|\s)", re.I)


def _is_light(color_int: int) -> bool:
    r, g, b = (color_int >> 16) & 255, (color_int >> 8) & 255, color_int & 255
    return (0.299 * r + 0.587 * g + 0.114 * b) > 190


def _span_rotated(line_dir) -> bool:
    dx, dy = line_dir
    return abs(dy) > 0.08  # 非水平文字（典型的斜向水印）


def collect_repeated(doc: fitz.Document) -> set:
    """跨页重复出现且位置接近的文本 -> 页眉/页脚/水印候选。"""
    seen = defaultdict(list)
    npages = len(doc)
    for pno in range(npages):
        d = doc[pno].get_text("dict")
        for blk in d.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                text = "".join(s["text"] for s in ln.get("spans", [])).strip()
                key = re.sub(r'\d+', '#', re.sub(r'\s+', '', text))
                if len(key) >= 2:
                    y = ln["bbox"][1]
                    seen[key].append((pno, round(y / 20)))
    repeated = set()
    if npages >= 3:
        for key, occ in seen.items():
            pages = {p for p, _ in occ}
            ys = defaultdict(int)
            for _, y in occ:
                ys[y] += 1
            if len(pages) >= max(3, int(npages * 0.6)) and max(ys.values()) >= len(pages) * 0.8:
                repeated.add(key)
    return repeated


def extract_text_page(doc: fitz.Document, pno: int, repeated: set,
                      assets_dir: str, asset_prefix: str,
                      preserve_images: bool = True,
                      detect_formula_images: bool = True,
                      ) -> Tuple[List[Block], List[str]]:
    page = doc[pno]
    notes = []
    blocks: List[Block] = []
    H = page.rect.height

    # 先收集本页水印文字（用于清洗表格单元格——find_tables 会把叠在
    # 表格上的水印一起抽出来）
    wm_tokens = set()
    d0 = page.get_text("dict")
    for blk in d0.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            key = re.sub(r'\d+', '#', re.sub(r'\s+', '', text))
            rot = _span_rotated(ln.get("dir", (1, 0)))
            light = spans and all(_is_light(s.get("color", 0)) for s in spans)
            if (rot and (key in repeated or len(doc) < 3)) or (key in repeated and light):
                for tok in re.split(r'\s+', text):
                    if len(tok) >= 2:
                        wm_tokens.add(tok)

    def scrub_cell(cell: str) -> str:
        parts = []
        for seg in cell.split("\n"):
            s = seg.strip()
            if s and any(s == t or (len(s) >= 2 and s in t) for t in wm_tokens):
                continue
            parts.append(seg)
        return "\n".join(parts).strip()

    raw_vector_areas = _detect_vector_figure_regions(page, [])
    caption_vector_areas = _caption_figure_regions(page, raw_vector_areas)

    # 表格（PyMuPDF 内置检测）。文本型 PDF 的标准表格常会被
    # find_tables 拆成多个横向碎片，因此先合并相邻碎片，再按文字坐标
    # 重建行列，避免论文表格被压扁成一串数字。
    table_areas = []
    try:
        tabs = page.find_tables()
        native_tables = []
        for t in tabs.tables:
            table_rect = fitz.Rect(t.bbox)
            if any(_rect_overlap_ratio(table_rect, vr) > 0.45
                   for vr in caption_vector_areas):
                continue
            rows = [[scrub_cell((c or "").strip()) for c in r] for r in t.extract()]
            rows = [r for r in rows if any(r)]
            if rows:
                native_tables.append((table_rect, rows, list(getattr(t, "cells", []) or [])))
        for table_rect, rows, cells in _merge_native_table_fragments(page, native_tables):
            rebuilt = _reconstruct_native_table_rows(page, table_rect, rows, cells)
            blocks.append(Block(kind="table", rows=rebuilt or rows, page=pno,
                                bbox=tuple(table_rect),
                                flags=["native_pdf_table"]))
            table_areas.append(table_rect)
    except Exception as e:
        notes.append(f"p{pno+1} 表格检测失败: {e}")

    vector_areas = _dedupe_regions(caption_vector_areas + [
        r for r in raw_vector_areas
        if not any(_rect_overlap_ratio(r, table) > 0.45 for table in table_areas)
    ])
    visual_skip_areas = list(vector_areas)
    if vector_areas and not preserve_images:
        notes.append(f"p{pno+1} Markdown 跳过非公式图示 {len(vector_areas)} 处")
    if vector_areas and preserve_images:
        os.makedirs(assets_dir, exist_ok=True)
        for i, r in enumerate(vector_areas):
            try:
                image_path = os.path.join(
                    assets_dir, f"{asset_prefix}_p{pno+1}_figure{i+1}.png")
                _save_page_image_crop(page, r, image_path)
                blocks.append(Block(kind="image", image_path=image_path,
                                    page=pno, bbox=(r.x0, r.y0, r.x1, r.y1),
                                    flags=["vector_figure"]))
            except Exception as e:
                notes.append(f"p{pno+1} 图示裁切失败: {e}")

    # 内嵌图片：Markdown 只保留疑似公式图片的 OCR 文本；Word 保留裁切图。
    skipped_images = 0
    for i, info in enumerate(page.get_image_info(xrefs=True)):
        xref = info.get("xref", 0)
        if not info.get("bbox"):
            continue
        r = fitz.Rect(info["bbox"])
        if r.width < 30 or r.height < 30:  # 装饰小图标跳过
            continue
        if any(_rect_overlap_ratio(r, vr) > 0.65 for vr in vector_areas):
            continue
        if _rect_area(r) / max(_rect_area(page.rect), 1) < 0.75:
            visual_skip_areas.append(r)
        formula_text = ""
        if detect_formula_images and _image_bbox_may_contain_formula(r, page.rect):
            formula_text = _detect_formula_text_in_image(page, r)
        image_path = ""
        if preserve_images:
            try:
                os.makedirs(assets_dir, exist_ok=True)
                image_path = os.path.join(
                    assets_dir, f"{asset_prefix}_p{pno+1}_img{i+1}.png")
                _save_page_image_crop(page, r, image_path)
            except Exception as e:
                notes.append(f"p{pno+1} 图片裁切失败 xref={xref}: {e}")
                image_path = ""
        if formula_text:
            flags = ["formula", "formula_latex"]
            blocks.append(Block(kind="image", image_path=image_path, text=formula_text,
                                page=pno, bbox=(r.x0, r.y0, r.x1, r.y1),
                                flags=flags))
        elif image_path:
            blocks.append(Block(kind="image", image_path=image_path, page=pno,
                                bbox=(r.x0, r.y0, r.x1, r.y1)))
        else:
            skipped_images += 1
    if skipped_images:
        notes.append(f"p{pno+1} Markdown 跳过非公式图片 {skipped_images} 张")

    # 文本（跳过水印 / 页眉页脚 / 表格区域内文字）
    d = page.get_text("dict")
    removed_wm = 0
    sizes = []
    items = []  # (y0, x0, x1, y1, size, text, bold)
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            bbox = fitz.Rect(ln["bbox"])
            if any(bbox.intersects(t) for t in table_areas):
                continue
            if _line_inside_visual_area(bbox, visual_skip_areas):
                continue
            key = re.sub(r'\d+', '#', re.sub(r'\s+', '', text))
            in_margin = bbox.y1 < H * 0.08 or bbox.y0 > H * 0.92
            if key in repeated and in_margin:
                continue  # 页眉页脚
            if in_margin and _PAGE_NUM_RE.match(text):
                continue  # 页码
            if key in repeated and (_span_rotated(ln.get("dir", (1, 0)))
                                    or all(_is_light(s.get("color", 0)) for s in spans)):
                removed_wm += 1
                continue  # 水印：跨页重复 且 (倾斜 或 浅色)
            if _span_rotated(ln.get("dir", (1, 0))) and len(doc) < 3:
                removed_wm += 1
                continue  # 单页文档没有跨页信号，倾斜文字直接视为水印
            size = max(s["size"] for s in spans)
            bold = any(s.get("flags", 0) & 16 for s in spans)
            sizes.append(size)
            items.append((bbox.y0, bbox.x0, bbox.x1, bbox.y1, size, text, bold))
    if removed_wm:
        notes.append(f"p{pno+1} 移除疑似水印文字 {removed_wm} 处")

    body = _mode(sizes) if sizes else 11
    items.sort(key=lambda t: (t[0], t[1]))
    para_lines: List[Tuple] = []
    pending_caption_lines: List[Tuple] = []
    pending_math_lines: List[Tuple] = []

    def flush():
        if not para_lines:
            return
        text = ""
        for *_xy, _s, t, _b in para_lines:
            if text and text[-1:].isascii() and t[:1].isascii():
                text += " " + t
            else:
                text += t
        text = _repair_text_pdf_spacing(_repair_wrapped_hyphenation(text))
        size = max(p[4] for p in para_lines)
        bold = all(p[6] for p in para_lines)
        x0 = min(p[1] for p in para_lines)
        x1 = max(p[2] for p in para_lines)
        y0 = para_lines[0][0]; y1 = max(p[3] for p in para_lines)
        kind, level = "para", 0
        if (not _looks_like_math_fragment(text)
                and (size > body * 1.15
                     or (bold and len(text) <= 40 and len(para_lines) == 1))):
            kind = "heading"
            fallback = 1 if size > body * 1.6 else (2 if size > body * 1.3 else 3)
            level = _heading_level_from_text(text, fallback)
        blocks.append(Block(kind=kind, text=text, level=level, page=pno,
                            bbox=(x0, y0, x1, y1)))
        para_lines.clear()

    def flush_pending_caption():
        if not pending_caption_lines:
            return
        text = ""
        for *_xy, _s, t, _b in pending_caption_lines:
            if text and text[-1:].isascii() and t[:1].isascii():
                text += " " + t
            else:
                text += t
        text = _repair_text_pdf_spacing(_repair_wrapped_hyphenation(text))
        x0 = min(p[1] for p in pending_caption_lines)
        x1 = max(p[2] for p in pending_caption_lines)
        y0 = pending_caption_lines[0][0]
        y1 = max(p[3] for p in pending_caption_lines)
        blocks.append(Block(kind="para", text=text, page=pno,
                            bbox=(x0, y0, x1, y1),
                            flags=["caption"]))
        pending_caption_lines.clear()

    def flush_pending_math():
        if not pending_math_lines:
            return
        text = "\n".join(p[5] for p in pending_math_lines if p[5]).strip()
        x0 = min(p[1] for p in pending_math_lines)
        x1 = max(p[2] for p in pending_math_lines)
        y0 = pending_math_lines[0][0]
        y1 = max(p[3] for p in pending_math_lines)
        blocks.append(Block(kind="para", text=text, page=pno,
                            bbox=(x0, y0, x1, y1),
                            flags=["formula_text"]))
        pending_math_lines.clear()

    prev = None
    for it in items:
        y0, x0, x1, y1, size, text, bold = it
        if pending_math_lines:
            math_prev = pending_math_lines[-1]
            gap = y0 - math_prev[3]
            if gap <= size * 1.3 and (
                    _is_standalone_math_line(text, x0, x1, page.rect.width)
                    or _is_equation_number_line(text)):
                pending_math_lines.append(it)
                prev = None
                continue
            flush_pending_math()

        if pending_caption_lines:
            cap_prev = pending_caption_lines[-1]
            gap = y0 - cap_prev[3]
            same_caption_column = abs(x0 - pending_caption_lines[0][1]) < 24
            if (gap <= size * 1.35 and same_caption_column
                    and not _is_caption_start(text)
                    and not _NUM_HEADING_RE.match(text)
                    and not _CN_HEADING_RE.match(text)):
                pending_caption_lines.append(it)
                continue
            if not para_lines:
                flush_pending_caption()

        if _is_caption_start(text):
            if para_lines:
                # Text often wraps around a figure.  Defer the caption until
                # the surrounding paragraph naturally ends so the caption does
                # not split a sentence in the final Markdown.
                pending_caption_lines.append(it)
            else:
                pending_caption_lines.append(it)
                if text.rstrip().endswith((".", "。")):
                    flush_pending_caption()
            prev = None
            continue

        if _is_standalone_math_line(text, x0, x1, page.rect.width):
            flush()
            flush_pending_caption()
            pending_math_lines.append(it)
            prev = None
            continue

        if prev is not None:
            gap = y0 - prev[3]
            size_changed = abs(size - prev[4]) > 1
            keep_sentence = _continues_unfinished_text_line(prev[5], text)
            if (gap > size * 0.8 or size_changed) and not keep_sentence:
                flush()
                flush_pending_caption()
                flush_pending_math()
        para_lines.append(it)
        prev = it
    flush()
    flush_pending_math()
    flush_pending_caption()

    _promote_adjacent_math_fragments(blocks)
    _split_trailing_numbered_headings(blocks)
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return blocks, notes


def _is_caption_start(text: str) -> bool:
    return bool(_CAPTION_START_RE.match((text or "").strip()))


def _is_equation_number_line(text: str) -> bool:
    return bool(re.fullmatch(r"\(?\d{1,3}\)?", re.sub(r"\s+", "", text or "")))


def _is_standalone_math_line(text: str, x0: float, x1: float,
                             page_width: float) -> bool:
    clean = re.sub(r"\s+", "", text or "")
    if not _looks_like_math_fragment(text):
        return False
    if len(clean) > 120:
        return False
    if re.search(r"[\u4e00-\u9fff]", clean):
        return False
    words = re.findall(r"[A-Za-z]{3,}", text or "")
    strong_math_symbols = len(re.findall(r"[=≈≤≥∑Σ√∪∩{}→←↔↦]|\b(?:exp|max|min|SUM)\b", text or "", re.I))
    if len(words) >= 4 and strong_math_symbols < 2:
        return False
    width = max(0.0, x1 - x0)
    centered_or_short = (
        width <= page_width * 0.68
        or x0 >= page_width * 0.25
        or (x0 > page_width * 0.12 and x1 < page_width * 0.88)
    )
    return centered_or_short and bool(
        re.search(r"[=≈≤≥∑Σ√→←↔↦]|[𝑎-𝑧𝐴-𝑍α-ωΑ-Ω]", clean)
        or re.fullmatch(r"(?:Í?exp|sqrt|log|min|max|sum|SUM)", clean)
    )


def _promote_adjacent_math_fragments(blocks: List[Block]) -> None:
    formula_blocks = [
        b for b in blocks
        if b.kind == "para" and "formula_text" in b.flags
    ]
    if not formula_blocks:
        return
    for blk in blocks:
        if blk.kind != "para" or "formula_text" in blk.flags:
            continue
        if not _looks_like_short_math_fragment_block(blk.text):
            continue
        if any(_math_blocks_are_near(blk, other) for other in formula_blocks):
            blk.flags.append("formula_text")


def _split_trailing_numbered_headings(blocks: List[Block]) -> None:
    """Recover headings that PyMuPDF glued to the previous paragraph tail."""
    out: List[Block] = []
    heading_tail = re.compile(
        r"^(?P<body>.+?[.!?。])\s+"
        r"(?P<head>\d{1,2}(?:[.．]\d{1,2}){0,4}[.．]?\s+"
        r"[A-Z][^\n]{2,90})$"
    )
    for blk in blocks:
        text = (blk.text or "").strip()
        if blk.kind != "para" or "\n" in text:
            out.append(blk)
            continue
        match = heading_tail.match(text)
        if not match:
            out.append(blk)
            continue
        body = match.group("body").strip()
        head = match.group("head").strip()
        title_words = re.findall(r"[A-Za-z0-9]+", head)
        if len(body) < 80 or not (2 <= len(title_words) <= 8):
            out.append(blk)
            continue
        if re.search(r"\b(?:Table|Figure|Fig)\s+\d+\b", head, re.I):
            out.append(blk)
            continue
        x0, y0, x1, y1 = blk.bbox
        split_y = max(y0, y1 - 18)
        out.append(Block(kind="para", text=body, page=blk.page,
                         bbox=(x0, y0, x1, split_y),
                         flags=list(blk.flags)))
        level = _heading_level_from_text(head, 3)
        out.append(Block(kind="heading", text=head, level=level,
                         page=blk.page, bbox=(x0, split_y, x1, y1)))
    blocks[:] = out


def _looks_like_short_math_fragment_block(text: str) -> bool:
    clean = _strip_pdf_control_chars(re.sub(r"\s+", "", text or ""))
    if not clean or len(clean) > 48:
        return False
    if re.search(r"[\u4e00-\u9fff]", clean):
        return False
    if re.search(r"[=≈≤≥∑Σ√⊤⊥]|[𝑎-𝑧𝐴-𝑍α-ωΑ-Ω]", clean):
        return True
    return bool(re.fullmatch(r"(?:Í|Í?exp|sqrt|log|min|max|sum|SUM|[(),.;0-9]+)", clean))


def _math_blocks_are_near(a: Block, b: Block) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    vertical_gap = max(0.0, max(ay0, by0) - min(ay1, by1))
    if vertical_gap > 18:
        return False
    horizontal_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    return horizontal_gap <= 80


def _strip_pdf_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")


def _continues_unfinished_text_line(prev_text: str, next_text: str) -> bool:
    prev = (prev_text or "").strip()
    nxt = (next_text or "").strip()
    if not prev or not nxt:
        return False
    if prev.endswith((".", "。", "!", "！", "?", "？", ":", "：", ";", "；")):
        return False
    if _is_caption_start(nxt) or _NUM_HEADING_RE.match(nxt) or _CN_HEADING_RE.match(nxt):
        return False
    if re.match(r"^[a-z]", nxt):
        return True
    return False


def _repair_wrapped_hyphenation(text: str) -> str:
    """Join words split by PDF line-end hyphenation."""
    return re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", text or "")


_MATH_TEXT_CHARS = "𝑎-𝑧𝐴-𝑍α-ωΑ-Ω"


def _repair_text_pdf_spacing(text: str) -> str:
    """Add missing spaces between inline math variables and prose words."""
    if not text:
        return ""
    text = re.sub(r"\s+—(?=\S)", "—", text)
    text = re.sub(rf"(?<=\))(?=[{_MATH_TEXT_CHARS}])", " ", text)
    text = re.sub(rf"(?<=[{_MATH_TEXT_CHARS}])(?=[a-z])", " ", text)
    text = re.sub(rf"(?<=[{_MATH_TEXT_CHARS}])(?=[A-Z][a-z])", " ", text)
    return text


def _merge_native_table_fragments(page: fitz.Page, tables):
    if not tables:
        return []
    items = sorted(tables, key=lambda item: (item[0].y0, item[0].x0))
    groups = []
    for rect, rows, cells in items:
        placed = False
        for group in groups:
            grect = group["rect"]
            same_width = (
                abs(rect.x0 - grect.x0) < 18
                and abs(rect.x1 - grect.x1) < 18
            )
            gap = rect.y0 - grect.y1
            if same_width and -2 <= gap <= 36:
                group["rect"] = grect | rect
                group["parts"].append((rect, rows, cells))
                placed = True
                break
        if not placed:
            groups.append({"rect": fitz.Rect(rect), "parts": [(rect, rows, cells)]})

    merged = []
    for group in groups:
        rect = group["rect"]
        rows = []
        cells = []
        for _rect, part_rows, part_cells in group["parts"]:
            rows.extend(part_rows)
            cells.extend(part_cells)
        merged.append((rect, rows, cells))
    return merged


def _reconstruct_native_table_rows(page: fitz.Page, rect: fitz.Rect,
                                   fallback_rows: List[List[str]],
                                   cells) -> List[List[str]]:
    words = _words_in_rect(page, rect)
    if not words:
        return fallback_rows
    line_rows = _word_grid_rows(rect, words, cells)
    if not line_rows:
        return fallback_rows
    line_rows = _drop_empty_table_rows(line_rows)
    if not line_rows:
        return fallback_rows
    normalized = _normalize_scientific_table_rows(line_rows)
    return normalized or line_rows


def _words_in_rect(page: fitz.Page, rect: fitz.Rect):
    expanded = _expand_rect(rect, 2)
    out = []
    for word in page.get_text("words"):
        wrect = fitz.Rect(word[:4])
        if not expanded.intersects(wrect):
            continue
        text = (word[4] or "").strip()
        if not text:
            continue
        out.append((wrect, text))
    return sorted(out, key=lambda item: (item[0].y0, item[0].x0))


def _word_grid_rows(rect: fitz.Rect, words, cells) -> List[List[str]]:
    lines = []
    for wrect, text in words:
        cy = (wrect.y0 + wrect.y1) / 2
        for line in lines:
            if abs(cy - line["cy"]) <= 3.8:
                line["words"].append((wrect, text))
                count = len(line["words"])
                line["cy"] = (line["cy"] * (count - 1) + cy) / count
                break
        else:
            lines.append({"cy": cy, "words": [(wrect, text)]})
    lines.sort(key=lambda line: line["cy"])

    boundaries = _table_boundaries_from_cells(rect, cells)
    if not _boundaries_are_wide_enough(boundaries, lines):
        boundaries = _table_boundaries_from_word_anchors(rect, lines, boundaries)
    if len(boundaries) < 3:
        return []

    rows: List[List[str]] = []
    for line in lines:
        cols = [[] for _ in range(len(boundaries) - 1)]
        for wrect, text in sorted(line["words"], key=lambda item: item[0].x0):
            idx = _column_index_for_word(wrect, boundaries)
            if 0 <= idx < len(cols):
                cols[idx].append((wrect.x0, text))
        rows.append([
            _join_cell_words(parts)
            for parts in cols
        ])
    return rows


def _table_boundaries_from_cells(rect: fitz.Rect, cells) -> List[float]:
    xs = {round(rect.x0, 1), round(rect.x1, 1)}
    for cell in cells or []:
        if not cell:
            continue
        xs.add(round(cell[0], 1))
        xs.add(round(cell[2], 1))
    out = sorted(xs)
    if len(out) < 3:
        return []
    return out


def _boundaries_are_wide_enough(boundaries: List[float], lines) -> bool:
    if len(boundaries) >= 7:
        return True
    max_value_tokens = 0
    for line in lines:
        value_tokens = sum(
            1 for _wrect, text in line["words"]
            if re.match(r"^[↑↓]?\d+(?:[.,]\d+)?%?$|^[-–—]$|^\d+B$", text)
        )
        max_value_tokens = max(max_value_tokens, value_tokens)
    return len(boundaries) - 1 >= max_value_tokens + 1


def _table_boundaries_from_word_anchors(rect: fitz.Rect, lines,
                                        base_boundaries: List[float]) -> List[float]:
    value_xs = []
    for line in lines:
        numeric = [
            wrect.x0 for wrect, text in line["words"]
            if re.match(r"^[↑↓]?\d+(?:[.,]\d+)?%?$|^\d{1,4}$|^\d+B$|^[-–—]$", text)
        ]
        if len(numeric) >= 3:
            value_xs.extend(numeric)
    value_anchors = _cluster_x_positions(value_xs, tolerance=8)
    if not value_anchors:
        return base_boundaries

    first_value = value_anchors[0]
    left_anchors = [
        x for x in base_boundaries[:-1]
        if x < first_value - 15 and x >= rect.x0 - 1
    ]
    if not left_anchors:
        left_anchors = [rect.x0]
    anchors = _cluster_x_positions(left_anchors + value_anchors, tolerance=7)
    if len(anchors) < 2:
        return base_boundaries

    boundaries = [rect.x0]
    for left, right in zip(anchors, anchors[1:]):
        boundaries.append((left + right) / 2)
    boundaries.append(rect.x1)
    boundaries = sorted(set(round(x, 1) for x in boundaries))
    return boundaries if len(boundaries) > len(base_boundaries) else base_boundaries


def _cluster_x_positions(xs: List[float], tolerance: float = 7) -> List[float]:
    if not xs:
        return []
    clusters: List[List[float]] = []
    for x in sorted(xs):
        if clusters and abs(x - (sum(clusters[-1]) / len(clusters[-1]))) <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _column_index_for_word(wrect: fitz.Rect, boundaries: List[float]) -> int:
    cx = (wrect.x0 + wrect.x1) / 2
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] - 0.5 <= cx <= boundaries[idx + 1] + 0.5:
            return idx
    return len(boundaries) - 2 if cx > boundaries[-1] else 0


def _join_cell_words(parts) -> str:
    if not parts:
        return ""
    parts = sorted(parts)
    text = ""
    prev = None
    for x0, token in parts:
        if text and prev is not None and x0 - prev > 2:
            text += " "
        text += token
        prev = x0 + max(len(token), 1) * 3.2
    return re.sub(r"\s+", " ", text).strip()


def _drop_empty_table_rows(rows: List[List[str]]) -> List[List[str]]:
    return [row for row in rows if any(cell.strip() for cell in row)]


def _normalize_scientific_table_rows(rows: List[List[str]]) -> List[List[str]]:
    compact = " ".join(" ".join(cell for cell in row if cell) for row in rows)
    if "Metric" in compact and "Pages" in compact and "Distinct-20" in compact:
        return _normalize_metric_pages_table(rows)
    if "TPS" in compact and "Model" in compact and "Deepseek OCR" in compact:
        return _normalize_tps_table(rows)
    if "Model" in compact and "Edit" in compact and "DS-OCR" in compact and "UOW" in compact:
        return _normalize_subcategory_table(rows)
    if "Model" in compact and "Overall" in compact and "Unlimited-OCR" in compact:
        return _normalize_omnidocbench_table(rows)
    return rows


def _normalize_metric_pages_table(rows: List[List[str]]) -> List[List[str]]:
    values = []
    body = []
    for row in rows:
        text = " ".join(row)
        nums = re.findall(r"40\+|\b\d+\b", text)
        if "Pages" in text and nums:
            values = nums
            continue
        label_parts = []
        value_start = len(row)
        for idx, cell in enumerate(row):
            if _cell_starts_with_table_values(cell):
                value_start = idx
                break
            if cell.strip() and "Metric" not in cell:
                label_parts.append(cell.strip())
        label = " ".join(label_parts).strip()
        row_values = re.findall(r"\d+(?:\.\d+)?%?", " ".join(row[value_start:]))
        if label and row_values:
            label = re.sub(r"\s+", " ", label)
            body.append([label] + row_values[:len(values)])
    if values and body:
        return [["Metric"] + values] + body
    return rows


def _cell_starts_with_table_values(cell: str) -> bool:
    tokens = [tok for tok in re.split(r"\s+", (cell or "").strip()) if tok]
    if not tokens:
        return False
    numeric = 0
    for tok in tokens:
        if re.fullmatch(r"[↑↓]?\d+(?:\.\d+)?%?|40\+", tok):
            numeric += 1
    return numeric >= 1 and numeric >= max(1, len(tokens) - 1)


def _normalize_tps_table(rows: List[List[str]]) -> List[List[str]]:
    lengths = []
    body = []
    for row in rows:
        text = " ".join(row)
        nums = re.findall(r"\b\d{3,4}\b", text)
        if ("TPS" in text or "Model" in text) and len(nums) >= 3:
            lengths = nums
            continue
        first_value_idx = None
        for i, cell in enumerate(row):
            if re.search(r"\d+\.\d+", cell):
                first_value_idx = i
                break
        if first_value_idx is None:
            continue
        model = " ".join(cell.strip() for cell in row[:first_value_idx] if cell.strip())
        values = re.findall(r"\d+(?:\.\d+)?", " ".join(row[first_value_idx:]))
        if model and values:
            body.append([model] + values[:len(lengths)])
    if lengths and body:
        return [["Model"] + lengths] + body
    return rows


def _normalize_subcategory_table(rows: List[List[str]]) -> List[List[str]]:
    header = [
        "Model", "Edit", "PPT", "Academic Paper", "Book",
        "Colorful Textbook", "Exam Paper", "Magazine", "Newspaper",
        "Note", "Research Report",
    ]
    body = []
    last_model = ""
    for idx, row in enumerate(rows):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        metric = cells[1] if len(cells) > 1 else ""
        if cells[0] and cells[0] != "Model" and metric not in {"Text", "R-order"}:
            last_model = cells[0]
            continue
        if metric in {"Text", "R-order"}:
            model = "" if cells[0] == "Model" else cells[0]
            if not model and metric == "Text":
                model = _nearby_model(rows, idx)
            model = model or last_model or _nearby_model(rows, idx)
            values = re.findall(r"\d+(?:\.\d+)?", " ".join(cells[2:]))
            if model and len(values) >= 6:
                body.append([model, metric] + values[:9])
                last_model = model
    if body:
        return [header] + body
    return rows


def _nearby_model(rows: List[List[str]], idx: int) -> str:
    for offset in (1, -1, 2, -2):
        pos = idx + offset
        if 0 <= pos < len(rows):
            cells = [cell.strip() for cell in rows[pos]]
            if (cells and cells[0] and cells[0] != "Model"
                    and not re.search(r"\d+\.\d+|Text|R-order", " ".join(cells[1:]))):
                return cells[0]
    return ""


def _normalize_omnidocbench_table(rows: List[List[str]]) -> List[List[str]]:
    header = [
        "Model", "Size", "Overall ↑", "Text Edit ↓", "Formula CDM ↑",
        "Table TEDs ↑", "Table TEDSs ↑", "Read-order Edit ↓",
    ]
    body = []
    for row in rows:
        cells = [cell.strip() for cell in row]
        text = " ".join(cells)
        if not text or "Model" in cells or "Overall" in text:
            continue
        if re.search(r"End-to-end\s+Model", text):
            body.append([re.sub(r"\s+", " ", text).strip()] + [""] * (len(header) - 1))
            continue
        if not cells[0]:
            if any(cells[2:]):
                body.append([""] * 2 + cells[2:len(header)])
            continue
        if len(cells) >= 6 and re.search(r"\d", " ".join(cells[2:])):
            body.append((cells + [""] * len(header))[:len(header)])
    if body:
        return [header] + body
    return rows


def _heading_level_from_text(text: str, fallback: int) -> int:
    clean = (text or "").strip()
    m = _NUM_HEADING_RE.match(clean)
    if m:
        depth = m.group(1).replace("．", ".").count(".") + 1
        return max(2, min(6, depth + 1))
    if _CN_HEADING_RE.match(clean):
        return 2
    if clean in {"Abstract", "Contents", "References", "Conclusion"}:
        return 2
    return max(1, min(6, fallback))


def _detect_vector_figure_regions(page: fitz.Page,
                                  table_areas: List[fitz.Rect]) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    page_rect = page.rect
    for drawing in page.get_drawings():
        raw = drawing.get("rect")
        if raw is None:
            continue
        r = fitz.Rect(raw)
        if r.width < 1 and r.height < 1:
            continue
        # Page rules and separators are not figures.
        if (r.height <= 1 and r.width > page_rect.width * 0.6
                and (r.y0 < page_rect.height * 0.08
                     or r.y0 > page_rect.height * 0.92)):
            continue
        rects.append(_expand_rect(r, 3))
    if not rects:
        return []

    components: List[Tuple[fitz.Rect, int]] = []
    for r in rects:
        placed = False
        for idx, (comp, count) in enumerate(components):
            if _expand_rect(comp, 12).intersects(r):
                components[idx] = (comp | r, count + 1)
                placed = True
                break
        if not placed:
            components.append((fitz.Rect(r), 1))

    changed = True
    while changed:
        changed = False
        merged: List[Tuple[fitz.Rect, int]] = []
        for comp, count in components:
            for idx, (other, other_count) in enumerate(merged):
                if _expand_rect(other, 12).intersects(comp):
                    merged[idx] = (other | comp, other_count + count)
                    changed = True
                    break
            else:
                merged.append((comp, count))
        components = merged

    regions: List[fitz.Rect] = []
    page_area = max(_rect_area(page_rect), 1)
    for comp, count in components:
        comp = _expand_rect(comp, 6) & page_rect
        if count < 8:
            continue
        if comp.width < page_rect.width * 0.18 or comp.height < page_rect.height * 0.04:
            continue
        if _rect_area(comp) / page_area < 0.006:
            continue
        if any(_rect_overlap_ratio(comp, table) > 0.45 for table in table_areas):
            continue
        regions.append(comp)
    return _dedupe_regions(regions)


def _caption_figure_regions(page: fitz.Page,
                            vector_regions: List[fitz.Rect]) -> List[fitz.Rect]:
    if not vector_regions:
        return []
    captions: List[fitz.Rect] = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        text = "".join(
            s["text"] for ln in blk.get("lines", []) for s in ln.get("spans", [])
        ).strip()
        if re.match(r"^(?:Figure|Fig\.?|图|表)\s*\d+", text, re.I):
            captions.append(fitz.Rect(blk["bbox"]))
    regions: List[fitz.Rect] = []
    page_rect = page.rect
    for cap in captions:
        above = [
            r for r in vector_regions
            if r.y1 <= cap.y0 + 8 and r.y1 >= cap.y0 - page_rect.height * 0.36
        ]
        if not above:
            continue
        y0 = max(0, min(r.y0 for r in above) - 16)
        y1 = max(y0 + 1, cap.y0 - 2)
        full_width_caption = cap.x0 < page_rect.width * 0.2 and cap.x1 > page_rect.width * 0.75
        if full_width_caption:
            x0, x1 = page_rect.width * 0.08, page_rect.width * 0.92
        else:
            x0 = max(0, min(cap.x0, *(r.x0 for r in above)) - 16)
            x1 = min(page_rect.width, max(cap.x1, *(r.x1 for r in above)) + 16)
        regions.append(fitz.Rect(x0, y0, x1, y1))
    return regions


def _dedupe_regions(regions: List[fitz.Rect]) -> List[fitz.Rect]:
    out: List[fitz.Rect] = []
    for region in sorted(regions, key=lambda r: (_rect_area(r), r.y0), reverse=True):
        if any(_rect_overlap_ratio(region, kept) > 0.75 for kept in out):
            continue
        out.append(region)
    return sorted(out, key=lambda r: (r.y0, r.x0))


def _line_inside_visual_area(bbox: fitz.Rect,
                             regions: List[fitz.Rect]) -> bool:
    if not regions:
        return False
    for region in regions:
        if not bbox.intersects(region):
            continue
        if _rect_overlap_ratio(bbox, region) >= 0.55:
            return True
        cx = (bbox.x0 + bbox.x1) / 2
        cy = (bbox.y0 + bbox.y1) / 2
        if region.x0 <= cx <= region.x1 and region.y0 <= cy <= region.y1:
            return True
    return False


def _expand_rect(rect: fitz.Rect, pad: float) -> fitz.Rect:
    return fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)


def _rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.x1 - rect.x0) * max(0.0, rect.y1 - rect.y0)


def _rect_overlap_ratio(inner: fitz.Rect, outer: fitz.Rect) -> float:
    inter = inner & outer
    return _rect_area(inter) / max(_rect_area(inner), 1.0)


def _looks_like_math_fragment(text: str) -> bool:
    clean = re.sub(r"\s+", "", text or "")
    if not clean or len(clean) > 80:
        return False
    if re.search(r"[\u4e00-\u9fff]", clean):
        return False
    if re.fullmatch(r"[𝑎-𝑧𝐴-𝑍α-ωΑ-Ω]|[→←↔↦]+\d*[.]?", clean):
        return True
    if re.fullmatch(r"(?:exp|sqrt|log|min|max|sum|SUM)", clean):
        return True
    math_chars = len(re.findall(r"[=+\-*/×÷√∑Σ≤≥→←↔↦(){}\[\]|𝑎-𝑧𝐴-𝑍α-ωΑ-Ω]", clean))
    ascii_letters = len(re.findall(r"[A-Za-z]", clean))
    return math_chars >= 2 and math_chars + ascii_letters >= max(3, len(clean) * 0.55)


def _mode(vals):
    c = defaultdict(int)
    for v in vals:
        c[round(v)] += 1
    return max(c.items(), key=lambda kv: kv[1])[0]


def _image_bbox_may_contain_formula(rect: fitz.Rect,
                                    page_rect: fitz.Rect) -> bool:
    if rect.width < 60 or rect.height < 18:
        return False
    page_area = max(page_rect.width * page_rect.height, 1)
    area_ratio = (rect.width * rect.height) / page_area
    if area_ratio > 0.75:
        return False
    # Standalone visual formulas are usually compact and horizontal.  Large
    # screenshots/diagrams may still be checked, but decorative slivers are not.
    return rect.width / max(rect.height, 1) >= 1.2 or rect.height <= page_rect.height * 0.28


def _detect_formula_text_in_image(page: fitz.Page, rect: fitz.Rect) -> str:
    with suppress(Exception):
        img = _render_page_crop(page, rect, max_side=1300)
        lines = ocr_strip(img, upscale_to=1500)
        text = "\n".join(ln.text for ln in lines if ln.text).strip()
        if _image_ocr_text_looks_like_formula(text):
            return text
    return ""


def _image_ocr_text_looks_like_formula(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 4:
        return False
    if not _IMAGE_FORMULA_TOKEN_RE.search(compact):
        return False
    # Avoid keeping ordinary screenshots just because they contain "1/2".
    if not _IMAGE_FORMULA_LABEL_RE.search(compact) and not re.search(
            r"[A-Za-z]{1,3}[_\u4e00-\u9fff]*\s*[=＝/]", compact):
        return False
    return True


def _save_page_image_crop(page: fitz.Page, rect: fitz.Rect, path: str) -> None:
    pix = _page_crop_pixmap(page, rect, max_side=1800)
    pix.save(path)


def _render_page_crop(page: fitz.Page, rect: fitz.Rect,
                      max_side: int = 1300) -> Image.Image:
    pix = _page_crop_pixmap(page, rect, max_side=max_side)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _page_crop_pixmap(page: fitz.Page, rect: fitz.Rect,
                      max_side: int = 1600) -> fitz.Pixmap:
    longest = max(rect.width, rect.height, 1)
    scale = max(1.0, min(2.5, max_side / longest))
    mat = fitz.Matrix(scale, scale)
    return page.get_pixmap(matrix=mat, clip=rect, alpha=False)


def builtin_check_cases():
    import tempfile

    def make_pdf_with_image(path: str) -> None:
        doc = fitz.open()
        page = doc.new_page(width=300, height=220)
        page.insert_text((36, 40), "文本型 PDF 正文", fontsize=12)
        img = Image.new("RGB", (80, 40), "white")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        page.insert_image(fitz.Rect(36, 70, 156, 130), stream=buf.getvalue())
        doc.save(path)
        doc.close()

    def make_pdf_with_vector_figure(path: str) -> None:
        doc = fitz.open()
        page = doc.new_page(width=360, height=300)
        page.insert_text((36, 40), "Body paragraph should stay", fontsize=11)
        figure = fitz.Rect(60, 80, 300, 170)
        page.draw_rect(figure)
        for offset in range(12):
            y = 90 + offset * 6
            page.draw_line((70, y), (290, y))
        page.insert_text((92, 110), "Diagram Label Should Not Export", fontsize=8)
        page.insert_text((36, 190), "Figure 1 | Caption should stay.", fontsize=9)
        doc.save(path)
        doc.close()

    def make_pdf_with_numbered_headings(path: str) -> None:
        doc = fitz.open()
        page = doc.new_page(width=360, height=300)
        page.insert_text((36, 40), "1. Parent Section", fontsize=14)
        page.insert_text((36, 68), "Parent paragraph text.", fontsize=10)
        page.insert_text((36, 100), "1.1 Child Section", fontsize=13)
        page.insert_text((36, 128), "Child paragraph text.", fontsize=10)
        page.insert_text((36, 160), "1.1.1 Grandchild Section", fontsize=13)
        page.insert_text((36, 188), "Grandchild paragraph text.", fontsize=10)
        doc.save(path)
        doc.close()

    def make_pdf_with_caption_inside_wrapped_text(path: str) -> None:
        doc = fitz.open()
        page = doc.new_page(width=420, height=260)
        page.insert_text((36, 40), "3.4.3. Kernel study", fontsize=13)
        page.insert_text((36, 84), "The spike crosses a certain alignment boundary, causing an abrupt", fontsize=10)
        page.insert_text((250, 140), "Figure 3 | The latency of the Flash Attention v3", fontsize=9)
        page.insert_text((250, 153), "kernel as decoding length increases.", fontsize=9)
        page.insert_text((36, 158), "drop in data transfer efficiency; this issue also does not arise.", fontsize=10)
        doc.save(path)
        doc.close()

    def make_pdf_with_standalone_math(path: str) -> None:
        doc = fitz.open()
        page = doc.new_page(width=420, height=260)
        page.insert_text((36, 40), "Before formula text:", fontsize=10)
        page.insert_text((170, 68), "C_MHA(T) = L_m + T.", fontsize=11)
        page.insert_text((350, 70), "(5)", fontsize=9)
        page.insert_text((36, 98), "After formula text should start a new paragraph.", fontsize=10)
        doc.save(path)
        doc.close()

    def case_markdown_skips_plain_text_pdf_images() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            make_pdf_with_image(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, notes = extract_text_page(
                doc, 0, set(), os.path.join(td, "assets"), "sample",
                preserve_images=False, detect_formula_images=False)
            doc.close()
            assert not any(b.kind == "image" for b in blocks)
            assert any("跳过非公式图片" in note for note in notes)
            assert not os.path.exists(os.path.join(td, "assets"))

    def case_word_preserves_text_pdf_images() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            assets = os.path.join(td, "assets")
            make_pdf_with_image(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, _notes = extract_text_page(
                doc, 0, set(), assets, "sample",
                preserve_images=True, detect_formula_images=False)
            doc.close()
            image_blocks = [b for b in blocks if b.kind == "image"]
            assert len(image_blocks) == 1
            assert image_blocks[0].image_path.endswith(".png")
            assert os.path.exists(image_blocks[0].image_path)

    def case_image_formula_text_detection() -> None:
        assert _image_ocr_text_looks_like_formula("承托比 = R / W")
        assert _image_ocr_text_looks_like_formula("复合准时率=1-C/W")
        assert not _image_ocr_text_looks_like_formula("公司 Logo 2026")

    def case_markdown_skips_vector_figure_text_but_keeps_caption() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            make_pdf_with_vector_figure(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, notes = extract_text_page(
                doc, 0, set(), os.path.join(td, "assets"), "sample",
                preserve_images=False, detect_formula_images=False)
            doc.close()
            all_text = "\n".join(b.text for b in blocks if b.text)
            assert "Body paragraph should stay" in all_text
            assert "Figure 1" in all_text
            assert "Diagram Label Should Not Export" not in all_text
            assert not any(b.kind == "image" for b in blocks)
            assert any("跳过非公式图示" in note for note in notes)

    def case_word_preserves_vector_figure() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            assets = os.path.join(td, "assets")
            make_pdf_with_vector_figure(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, _notes = extract_text_page(
                doc, 0, set(), assets, "sample",
                preserve_images=True, detect_formula_images=False)
            doc.close()
            image_blocks = [b for b in blocks if "vector_figure" in b.flags]
            assert len(image_blocks) == 1
            assert os.path.exists(image_blocks[0].image_path)

    def case_numbered_text_pdf_headings_keep_hierarchy() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            make_pdf_with_numbered_headings(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, _notes = extract_text_page(
                doc, 0, set(), os.path.join(td, "assets"), "sample",
                preserve_images=False, detect_formula_images=False)
            doc.close()
            levels = {
                b.text: b.level for b in blocks
                if b.kind == "heading" and "Section" in b.text
            }
            assert levels["1. Parent Section"] == 2
            assert levels["1.1 Child Section"] == 3
            assert levels["1.1.1 Grandchild Section"] == 4

    def case_caption_does_not_split_wrapped_body_text() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            make_pdf_with_caption_inside_wrapped_text(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, _notes = extract_text_page(
                doc, 0, set(), os.path.join(td, "assets"), "sample",
                preserve_images=False, detect_formula_images=False)
            doc.close()
            texts = [b.text for b in blocks if b.text]
            body_idx = next(i for i, text in enumerate(texts) if "alignment boundary" in text)
            caption_idx = next(i for i, text in enumerate(texts) if text.startswith("Figure 3 |"))
            assert body_idx < caption_idx
            assert "causing an abrupt drop in data transfer efficiency" in texts[body_idx]
            assert "Figure 3 | The latency" in texts[caption_idx]

    def case_standalone_math_line_not_merged_into_body() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            make_pdf_with_standalone_math(pdf_path)
            doc = fitz.open(pdf_path)
            blocks, _notes = extract_text_page(
                doc, 0, set(), os.path.join(td, "assets"), "sample",
                preserve_images=False, detect_formula_images=False)
            doc.close()
            texts = [b.text for b in blocks if b.text]
            assert any(text == "Before formula text:" for text in texts)
            math = next(text for text in texts if "C_MHA" in text)
            assert "After formula" not in math
            assert "(5)" in math
            assert any(text.startswith("After formula text") for text in texts)

    def case_math_symbols_in_sentence_do_not_make_formula_block() -> None:
        assert not _is_standalone_math_line(
            "attention weight from token 𝑡 to position 𝑗∈N (𝑡) is computed as",
            70, 408, 595)
        assert _is_standalone_math_line("C_MHA(T) = L_m + T.", 170, 350, 420)
        assert _is_standalone_math_line("exp", 257, 274, 595)
        assert _is_standalone_math_line("√𝑑𝑘", 284, 297, 595)
        assert _is_standalone_math_line("→0.", 320, 343, 595)
        assert _is_standalone_math_line("𝑇", 298, 304, 595)
        assert not _is_standalone_math_line("Therefore, for long-sequence decoding", 70, 524, 595)

    def case_adjacent_short_math_fragments_promoted() -> None:
        blocks = [
            Block(kind="para", text="𝑡k𝑗\nexp", flags=["formula_text"],
                  bbox=(257, 565, 299, 580)),
            Block(kind="para", text="√𝑑𝑘", bbox=(284, 569, 297, 584)),
            Block(kind="para", text="\x11 , (3) \x10 q⊤", bbox=(291, 581, 525, 599)),
            Block(kind="para", text="Í", bbox=(252, 593, 261, 604)),
            Block(kind="para", text="normal sentence with token 𝑡", bbox=(70, 620, 525, 635)),
        ]
        _promote_adjacent_math_fragments(blocks)
        assert "formula_text" in blocks[1].flags
        assert "formula_text" in blocks[2].flags
        assert "formula_text" in blocks[3].flags
        assert "formula_text" not in blocks[4].flags

    def case_trailing_numbered_heading_split_from_paragraph() -> None:
        paragraph = (
            "For documents with complex layouts such as PPT, newspapers, "
            "magazines, and note, Unlimited OCR shows no disadvantage either, "
            "further demonstrating that replacing all standard attention with "
            "R-SWA for LLM-decoder is complete and sound for parsing tasks. "
            "5.4. Long-horizon Parsing"
        )
        blocks = [Block(kind="para", text=paragraph, bbox=(70, 184, 525, 550))]
        _split_trailing_numbered_headings(blocks)
        assert len(blocks) == 2
        assert blocks[0].kind == "para"
        assert blocks[0].text.endswith("parsing tasks.")
        assert blocks[1].kind == "heading"
        assert blocks[1].text == "5.4. Long-horizon Parsing"

    def case_wrapped_hyphenated_words_repaired() -> None:
        text = "types of docu- ment elements and pro- vides readable output"
        fixed = _repair_wrapped_hyphenation(text)
        assert fixed == "types of document elements and provides readable output"
        assert _repair_wrapped_hyphenation("one-shot R-SWA end-to-end") == "one-shot R-SWA end-to-end"

    def case_inline_math_variable_spacing_repaired() -> None:
        text = "preceding 𝑛output tokens and 𝑚with fixed cache. (9)𝑇Therefore"
        fixed = _repair_text_pdf_spacing(text)
        assert fixed == "preceding 𝑛 output tokens and 𝑚 with fixed cache. (9) 𝑇 Therefore"
        assert _repair_text_pdf_spacing("𝐶MHA(𝑇) = 𝐿𝑚+ 𝑇") == "𝐶MHA(𝑇) = 𝐿𝑚+ 𝑇"
        assert _repair_text_pdf_spacing("attention mechanism —beyond OCR") == "attention mechanism—beyond OCR"

    def case_unfinished_line_continuation_allowed_across_size_change() -> None:
        assert _continues_unfinished_text_line("Codes and model", "weights are available")
        assert not _continues_unfinished_text_line("Codes and model.", "weights are available")
        assert not _continues_unfinished_text_line("Codes and model", "5.4 Long-horizon Parsing")
        assert not _continues_unfinished_text_line("Contents", "1 Introduction")
        assert not _continues_unfinished_text_line("8. Conclusion", "In this technical report")

    def case_metric_pages_table_rebuilt_from_text_pdf_fragments() -> None:
        rows = [
            ["Pages", "2 5 10 15 20 40+"],
            ["Metric", ""],
            ["Distinct-20 ↑", "99.76% 99.78% 97.49% 99.92% 98.73% 96.08%"],
            ["Distinct-35 ↑", "99.87% 99.98% 99.83% 99.99% 99.89% 96.90%"],
            ["Edit Distance ↓", "0.0362 0.0452 0.0526 0.0787 0.0572 0.1069"],
        ]
        fixed = _normalize_metric_pages_table(rows)
        assert fixed[0] == ["Metric", "2", "5", "10", "15", "20", "40+"]
        assert fixed[1][0] == "Distinct-20 ↑"
        assert fixed[1][-1] == "96.08%"
        assert fixed[3][0] == "Edit Distance ↓"

    def case_subcategory_table_rebuilt_from_text_pdf_fragments() -> None:
        rows = [
            ["Model", "Edit ↓", "Academic Colorful Exam Research"],
            ["DS-OCR", "", ""],
            ["", "Text", "0.052 0.028 0.022 0.130 0.074 0.049 0.131 0.145 0.015"],
            ["", "R-order", "0.052 0.021 0.040 0.125 0.083 0.101 0.217 0.089 0.016"],
            ["DS-OCR 2", "", ""],
            ["", "Text", "0.031 0.013 0.033 0.053 0.047 0.026 0.139 0.068 0.008"],
            ["", "R-order", "0.025 0.013 0.027 0.066 0.048 0.100 0.176 0.035 0.011"],
        ]
        fixed = _normalize_subcategory_table(rows)
        assert fixed[0][:2] == ["Model", "Edit"]
        assert fixed[1][:3] == ["DS-OCR", "Text", "0.052"]
        assert fixed[2][:3] == ["DS-OCR", "R-order", "0.052"]
        assert fixed[3][:3] == ["DS-OCR 2", "Text", "0.031"]

    def case_tps_table_rebuilt_from_text_pdf_fragments() -> None:
        rows = [
            ["TPS", "256 512 1024 2048 3072 4096 6144"],
            ["Model", ""],
            ["Deepseek OCR", "7229.32 7468.27 7422.50 7166.85 6790.72 6430.21 5822.87"],
            ["Unlimited OCR", "7229.52 7714.78 7840.94 7881.11 7881.93 7905.18 7847.71"],
        ]
        fixed = _normalize_tps_table(rows)
        assert fixed[0] == ["Model", "256", "512", "1024", "2048", "3072", "4096", "6144"]
        assert fixed[1][:3] == ["Deepseek OCR", "7229.32", "7468.27"]
        assert fixed[2][-1] == "7847.71"

    return [
        ("text_extract.markdown_skips_plain_text_pdf_images", case_markdown_skips_plain_text_pdf_images),
        ("text_extract.word_preserves_text_pdf_images", case_word_preserves_text_pdf_images),
        ("text_extract.image_formula_text_detection", case_image_formula_text_detection),
        ("text_extract.markdown_skips_vector_figure_text_but_keeps_caption", case_markdown_skips_vector_figure_text_but_keeps_caption),
        ("text_extract.word_preserves_vector_figure", case_word_preserves_vector_figure),
        ("text_extract.numbered_text_pdf_headings_keep_hierarchy", case_numbered_text_pdf_headings_keep_hierarchy),
        ("text_extract.caption_does_not_split_wrapped_body_text", case_caption_does_not_split_wrapped_body_text),
        ("text_extract.standalone_math_line_not_merged_into_body", case_standalone_math_line_not_merged_into_body),
        ("text_extract.math_symbols_in_sentence_do_not_make_formula_block", case_math_symbols_in_sentence_do_not_make_formula_block),
        ("text_extract.adjacent_short_math_fragments_promoted", case_adjacent_short_math_fragments_promoted),
        ("text_extract.trailing_numbered_heading_split_from_paragraph", case_trailing_numbered_heading_split_from_paragraph),
        ("text_extract.wrapped_hyphenated_words_repaired", case_wrapped_hyphenated_words_repaired),
        ("text_extract.inline_math_variable_spacing_repaired", case_inline_math_variable_spacing_repaired),
        ("text_extract.unfinished_line_continuation_allowed_across_size_change", case_unfinished_line_continuation_allowed_across_size_change),
        ("text_extract.metric_pages_table_rebuilt_from_text_pdf_fragments", case_metric_pages_table_rebuilt_from_text_pdf_fragments),
        ("text_extract.subcategory_table_rebuilt_from_text_pdf_fragments", case_subcategory_table_rebuilt_from_text_pdf_fragments),
        ("text_extract.tps_table_rebuilt_from_text_pdf_fragments", case_tps_table_rebuilt_from_text_pdf_fragments),
    ]
