"""文本型 PDF 提取：直接读文字层，过滤水印/页眉页脚，保留内嵌图片与表格。"""
import os
import re
from collections import defaultdict
from typing import List, Tuple

import fitz

from .models import Block

_PAGE_NUM_RE = re.compile(r'^[-–—\s]*(第?\s*\d+\s*页?|\d+\s*/\s*\d+|[ivxIVX]+)[-–—\s]*$')


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
                      assets_dir: str, asset_prefix: str
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

    # 表格（PyMuPDF 内置检测）
    table_areas = []
    try:
        tabs = page.find_tables()
        for t in tabs.tables:
            rows = [[scrub_cell((c or "").strip()) for c in r] for r in t.extract()]
            rows = [r for r in rows if any(r)]
            if rows:
                blocks.append(Block(kind="table", rows=rows, page=pno,
                                    bbox=tuple(t.bbox)))
                table_areas.append(fitz.Rect(t.bbox))
    except Exception as e:
        notes.append(f"p{pno+1} 表格检测失败: {e}")

    # 内嵌图片
    os.makedirs(assets_dir, exist_ok=True)
    for i, info in enumerate(page.get_image_info(xrefs=True)):
        xref = info.get("xref", 0)
        if not xref:
            continue
        r = fitz.Rect(info["bbox"])
        if r.width < 30 or r.height < 30:  # 装饰小图标跳过
            continue
        try:
            img = doc.extract_image(xref)
            path = os.path.join(assets_dir, f"{asset_prefix}_p{pno+1}_img{i+1}.{img['ext']}")
            with open(path, "wb") as f:
                f.write(img["image"])
            blocks.append(Block(kind="image", image_path=path, page=pno,
                                bbox=(r.x0, r.y0, r.x1, r.y1)))
        except Exception as e:
            notes.append(f"p{pno+1} 图片提取失败 xref={xref}: {e}")

    # 文本（跳过水印 / 页眉页脚 / 表格区域内文字）
    d = page.get_text("dict")
    removed_wm = 0
    sizes = []
    items = []  # (y0, x0, size, text, bold)
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
            items.append((bbox.y0, bbox.x0, bbox.y1, size, text, bold))
    if removed_wm:
        notes.append(f"p{pno+1} 移除疑似水印文字 {removed_wm} 处")

    body = _mode(sizes) if sizes else 11
    items.sort(key=lambda t: (t[0], t[1]))
    para_lines: List[Tuple] = []

    def flush():
        if not para_lines:
            return
        text = ""
        for *_xy, _s, t, _b in para_lines:
            if text and text[-1:].isascii() and t[:1].isascii():
                text += " " + t
            else:
                text += t
        size = max(p[3] for p in para_lines)
        bold = all(p[5] for p in para_lines)
        y0 = para_lines[0][0]; y1 = max(p[2] for p in para_lines)
        kind, level = "para", 0
        if size > body * 1.15 or (bold and len(text) <= 40 and len(para_lines) == 1):
            kind = "heading"
            level = 1 if size > body * 1.6 else (2 if size > body * 1.3 else 3)
        blocks.append(Block(kind=kind, text=text, level=level, page=pno,
                            bbox=(0, y0, page.rect.width, y1)))
        para_lines.clear()

    prev = None
    for it in items:
        y0, x0, y1, size, text, bold = it
        if prev is not None:
            gap = y0 - prev[2]
            size_changed = abs(size - prev[3]) > 1
            if gap > size * 0.8 or size_changed:
                flush()
        para_lines.append(it)
        prev = it
    flush()

    blocks.sort(key=lambda b: b.bbox[1])
    return blocks, notes


def _mode(vals):
    c = defaultdict(int)
    for v in vals:
        c[round(v)] += 1
    return max(c.items(), key=lambda kv: kv[1])[0]
