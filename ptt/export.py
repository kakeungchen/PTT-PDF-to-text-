"""导出：Block 列表 -> Markdown / Word(.docx)。"""
import os
import re
from typing import List

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

from .models import Block, DocResult

# 列表式段落（①、a.、1.1、•…）不做首行缩进
_NO_INDENT = re.compile(
    r'^(\d{1,2}(\.\d{1,2})+|[a-zA-Z][.、)）]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫]|[•·▪\->（(【]|'
    r'场景[一二三四五六]|条件[一二三四五六]|情形[一二三四五六]|注意|备注|特殊说明)')


def _md_escape_text(s: str) -> str:
    """转义会被 Markdown 解析的符号（OCR 文本里的 * _ 等是字面字符）。"""
    return re.sub(r'([*_`\[\]])', r'\\\1', s)


def _md_escape_cell(s: str) -> str:
    return _md_escape_text(s).replace("|", "\\|").replace("\n", "<br>")


def export_markdown(result: DocResult, out_path: str) -> str:
    lines: List[str] = []
    title = result.meta.get("title")
    def emit_table(rows_data):
        ncol = max(len(r) for r in rows_data)
        rows = [r + [""] * (ncol - len(r)) for r in rows_data]
        lines.append("| " + " | ".join(_md_escape_cell(c) for c in rows[0]) + " |")
        lines.append("|" + "---|" * ncol)
        for r in rows[1:]:
            lines.append("| " + " | ".join(_md_escape_cell(c) for c in r) + " |")

    for b in result.blocks:
        if b.kind == "heading":
            lines.append("#" * max(1, min(b.level or 1, 6)) + " "
                         + _md_escape_text(b.text))
        elif b.kind == "para":
            text = _md_escape_text(b.text)
            if "low_confidence" in b.flags:
                text += "  <!-- 识别置信度低，建议人工核对 -->"
            lines.append(text)
        elif b.kind == "table" and b.rows:
            emit_table(b.rows)
        elif b.kind == "image" and b.image_path:
            rel = os.path.relpath(b.image_path, os.path.dirname(out_path) or ".")
            label = "公式" if "formula" in b.flags else "图片"
            lines.append(f"![{label}](<{rel}>)")
            # 图中识别出的文字（按版面结构还原；公式截图除外——其文本不可靠）
            if ((b.text or b.rows) and "formula" not in b.flags
                    and "auto_image" not in b.flags):
                lines.append("")
                lines.append("**图中文字（自动识别）**")
                for ln in (b.text or "").splitlines():
                    if ln.strip():
                        lines.append("")
                        lines.append(_md_escape_text(ln.strip()))
                if b.rows:
                    lines.append("")
                    emit_table(b.rows)
        lines.append("")
    md = "\n".join(lines).strip() + "\n"
    md = re.sub(r"\n{3,}", "\n\n", md)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path


def _set_cn_font(run, size=None, bold=None):
    run.font.name = "Times New Roman"
    r = run._element.rPr.rFonts
    r.set(qn("w:eastAsia"), "宋体")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def _shade_cell(cell, fill="EFEFEF"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def export_docx(result: DocResult, out_path: str) -> str:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.4
    style.paragraph_format.space_after = Pt(6)

    for b in result.blocks:
        if b.kind == "heading":
            p = doc.add_heading(level=max(1, min(b.level or 1, 4)))
            run = p.add_run(b.text)
            run.font.name = "Times New Roman"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
            run.font.color.rgb = RGBColor(0, 0, 0)
            sizes = {1: 16, 2: 14, 3: 13, 4: 12}
            run.font.size = Pt(sizes.get(b.level, 12))
            run.font.bold = True
        elif b.kind == "para":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            if len(b.text) > 25 and not _NO_INDENT.match(b.text):
                p.paragraph_format.first_line_indent = Pt(22)  # 首行缩进两字符
            run = p.add_run(b.text)
            _set_cn_font(run)
            if "low_confidence" in b.flags:
                from docx.enum.text import WD_COLOR_INDEX
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif b.kind == "table" and b.rows:
            ncol = max(len(r) for r in b.rows)
            t = doc.add_table(rows=len(b.rows), cols=ncol)
            t.style = "Table Grid"
            for i, row in enumerate(b.rows):
                for j in range(ncol):
                    cell = t.cell(i, j)
                    cell.text = ""
                    if i == 0 and len(b.rows) > 1:
                        _shade_cell(cell)
                    parts = (row[j] if j < len(row) else "").split("\n")
                    for k, part in enumerate(parts):
                        p = (cell.paragraphs[0] if k == 0
                             else cell.add_paragraph())
                        run = p.add_run(part)
                        _set_cn_font(run, size=10, bold=(i == 0))
            doc.add_paragraph()
        elif b.kind == "image" and b.image_path and os.path.exists(b.image_path):
            try:
                doc.add_picture(b.image_path, width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if ((b.text or b.rows) and "formula" not in b.flags
                        and "auto_image" not in b.flags):
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap = "图中文字（自动识别）"
                    if b.text:
                        cap += "：" + "；".join(
                            s.strip() for s in b.text.splitlines() if s.strip())
                    run = p.add_run(cap)
                    _set_cn_font(run, size=9)
                    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
                    if b.rows:
                        ncol = max(len(r) for r in b.rows)
                        t = doc.add_table(rows=len(b.rows), cols=ncol)
                        t.style = "Table Grid"
                        for i, row in enumerate(b.rows):
                            for j in range(ncol):
                                cell = t.cell(i, j)
                                cell.text = ""
                                run = cell.paragraphs[0].add_run(
                                    row[j] if j < len(row) else "")
                                _set_cn_font(run, size=8)
                                run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
                        doc.add_paragraph()
            except Exception:
                p = doc.add_paragraph()
                _set_cn_font(p.add_run(f"[图片: {os.path.basename(b.image_path)}]"))
    doc.save(out_path)
    return out_path
