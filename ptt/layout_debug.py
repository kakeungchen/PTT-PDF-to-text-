"""Optional layout ledger and visual overlays for conversion QA."""
import json
import os
import re
import tempfile
from typing import Callable
from typing import Dict, List, Tuple

import fitz
from PIL import Image, ImageDraw

from .models import Block, DocResult


_COLORS: Dict[str, Tuple[int, int, int]] = {
    "text": (37, 99, 235),
    "table": (34, 197, 94),
    "formula": (239, 68, 68),
    "image": (168, 85, 247),
    "discarded": (107, 114, 128),
}


def block_region_type(block: Block) -> str:
    if block.kind == "table" or "table_fallback" in block.flags or block.rows:
        return "table"
    if block.kind == "image" and "formula" in block.flags:
        return "formula"
    if block.kind == "image":
        return "image"
    return "text"


def block_export_status(block: Block) -> str:
    if block.kind in ("heading", "para", "list"):
        return "text"
    if block.kind == "table" or block.rows:
        if block.image_path:
            return "table+original_image"
        return "table"
    if "formula" in block.flags:
        return "formula_or_original_image" if block.image_path else "formula_text"
    if block.image_path:
        return "original_image"
    return "text"


def build_layout_ledger(result: DocResult) -> Dict[str, object]:
    pages: Dict[int, List[Block]] = {}
    for block in sorted(result.blocks, key=lambda b: (b.page, b.bbox[1], b.bbox[0])):
        pages.setdefault(block.page, []).append(block)

    page_entries = []
    for page_index in sorted(pages):
        entries = []
        for order, block in enumerate(pages[page_index], start=1):
            entries.append({
                "order": order,
                "type": block_region_type(block),
                "kind": block.kind,
                "bbox": [round(float(v), 2) for v in block.bbox],
                "confidence": round(float(block.confidence), 4),
                "flags": list(block.flags),
                "export_status": block_export_status(block),
                "text_preview": _preview(block),
            })
        page_entries.append({
            "page": page_index + 1,
            "regions": entries,
        })
    return {
        "source": result.meta.get("source", ""),
        "title": result.meta.get("title", ""),
        "pages": page_entries,
    }


def export_debug_layout(pdf_path: str, result: DocResult, out_dir: str,
                        safe_name: str) -> List[str]:
    """Write a MinerU-style layout JSON and per-page overlay PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    layout_dir = os.path.join(out_dir, f"{safe_name}_layout")
    os.makedirs(layout_dir, exist_ok=True)

    ledger = build_layout_ledger(result)
    json_path = os.path.join(out_dir, f"{safe_name}_layout.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)

    outputs = [json_path]
    try:
        doc = fitz.open(pdf_path)
        for pno in range(len(doc)):
            blocks = [b for b in result.blocks if b.page == pno]
            if not blocks:
                continue
            image = _render_page_for_blocks(doc[pno], blocks)
            _draw_blocks(image, blocks)
            path = os.path.join(layout_dir, f"page-{pno + 1:03d}.png")
            image.save(path)
            outputs.append(path)
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return outputs


def _preview(block: Block) -> str:
    if block.text:
        text = block.text
    elif block.rows:
        text = " ".join(" ".join(cell for cell in row if cell) for row in block.rows[:3])
    else:
        text = ""
    return re.sub(r"\s+", " ", text).strip()[:120]


def _render_page_for_blocks(page: fitz.Page, blocks: List[Block]) -> Image.Image:
    rect = page.rect
    max_x = max([rect.width] + [float(b.bbox[2]) for b in blocks])
    max_y = max([rect.height] + [float(b.bbox[3]) for b in blocks])
    coord_scale = max(max_x / max(rect.width, 1), max_y / max(rect.height, 1), 1.0)
    render_scale = min(coord_scale, 3.0)
    image_infos = []
    try:
        image_infos = page.get_image_info()
    except Exception:
        image_infos = []
    if any(info.get("width", 0) > 65000 or info.get("height", 0) > 65000
           for info in image_infos):
        return _blank_layout_canvas(max_x, max_y)

    est_w = int(rect.width * render_scale)
    est_h = int(rect.height * render_scale)
    if est_w > 65000 or est_h > 65000 or est_w * est_h > 40_000_000:
        return _blank_layout_canvas(max_x, max_y)

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale),
                             alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return _blank_layout_canvas(max_x, max_y)
    image.info["coord_width"] = max_x
    image.info["coord_height"] = max_y
    return image


def _blank_layout_canvas(coord_width: float, coord_height: float) -> Image.Image:
    coord_width = max(float(coord_width), 1.0)
    coord_height = max(float(coord_height), 1.0)
    width = min(max(int(coord_width), 400), 2400)
    height = min(max(int(coord_height * width / coord_width), 300), 65000)
    image = Image.new("RGB", (width, height), "white")
    image.info["coord_width"] = coord_width
    image.info["coord_height"] = coord_height
    return image


def _draw_blocks(image: Image.Image, blocks: List[Block]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    coord_w = float(image.info.get("coord_width") or image.width)
    coord_h = float(image.info.get("coord_height") or image.height)
    sx = image.width / max(coord_w, 1)
    sy = image.height / max(coord_h, 1)

    for order, block in enumerate(sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])), start=1):
        typ = block_region_type(block)
        color = _COLORS.get(typ, _COLORS["text"])
        x0, y0, x1, y1 = block.bbox
        rect = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
        draw.rectangle(rect, outline=(*color, 255), width=3)
        draw.rectangle([rect[0], max(0, rect[1] - 18), rect[0] + 80, rect[1]],
                       fill=(*color, 210))
        draw.text((rect[0] + 4, max(0, rect[1] - 16)),
                  f"{order} {typ}", fill=(255, 255, 255, 255))


def builtin_check_cases() -> List[Tuple[str, Callable[[], None]]]:
    def case_debug_layout_exports_json_and_png() -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            doc = fitz.open()
            page = doc.new_page(width=300, height=200)
            page.insert_text((40, 60), "hello")
            doc.save(pdf_path)
            doc.close()

            result = DocResult(
                meta={"source": pdf_path, "title": "sample", "pages": 1},
                blocks=[
                    Block(kind="para", text="hello", page=0,
                          bbox=(35, 45, 120, 70)),
                    Block(kind="table", rows=[["A", "B"], ["1", "2"]],
                          page=0, bbox=(35, 90, 180, 140)),
                ],
            )
            outputs = export_debug_layout(pdf_path, result, td, "sample")
            assert any(path.endswith("_layout.json") for path in outputs)
            assert any(path.endswith("page-001.png") for path in outputs)
            with open(os.path.join(td, "sample_layout.json"), encoding="utf-8") as f:
                data = json.load(f)
            assert data["pages"][0]["regions"][0]["type"] == "text"
            assert data["pages"][0]["regions"][1]["type"] == "table"

    return [
        ("layout_debug.exports_json_and_png", case_debug_layout_exports_json_and_png),
    ]
