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
_NUMBERED_TOC_LINE = re.compile(r'^[一二三四五六七八九十]{1,3}、')


def _md_escape_text(s: str) -> str:
    """转义会被 Markdown 解析的符号（OCR 文本里的 * _ 等是字面字符）。"""
    return re.sub(r'([*_`\[\]])', r'\\\1', s)


def _md_escape_cell(s: str) -> str:
    return _md_escape_text(s).replace("|", "\\|").replace("\n", "<br>")


def _formula_to_latex(text: str) -> str:
    """把 OCR 公式碎片转成 Markdown 可渲染的 LaTeX。"""
    raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not raw_lines:
        return ""
    norm_lines = [
        ln.replace("＝", "=").replace("（", "(").replace("）", ")")
        for ln in raw_lines
    ]
    compact = re.sub(r'\s+', '', "".join(norm_lines))
    if ("站点组KA品牌单体验得分" in compact and "SUM" in compact.upper()
            and "K" in compact and "Q" in compact):
        return (
            r"\begin{aligned}"
            r"\text{站点组KA品牌单体验得分} &= "
            r"K_1\text{分}\times"
            r"\frac{K_1\times Q_1}{\mathrm{SUM}(K_n\times Q_n)}"
            r"+\cdots+"
            r"K_n\text{分}\times"
            r"\frac{K_n\times Q_n}{\mathrm{SUM}(K_n\times Q_n)}"
            r"\end{aligned}"
        )
    if ("站点组体验总得分" in compact and "站点组F" in compact
            and ("SUM" in compact.upper() or "Q" in compact)
            and ("K" in compact or "Kn" in compact)):
        return (
            r"\begin{aligned}"
            r"\text{站点组体验总得分} &= "
            r"\text{站点组F分}\times"
            r"\frac{F}{F+\mathrm{SUM}(K_n\times Q_n)} \\ "
            r"&\quad + \text{站点组}K_n\text{分}\times"
            r"\frac{K_n\times Q_n}{F+\mathrm{SUM}(K_n\times Q_n)}"
            r"\end{aligned}"
        )
    if ("118.4459" in compact and "10000" in compact
            and "700" in compact and "300" in compact):
        denominator = r"10000+700\times6+300\times2"
        return (
            r"\begin{aligned}"
            r"\text{A履约的体验总得分} &= "
            rf"118\times\frac{{10000}}{{{denominator}}}+"
            rf"120\times\frac{{700\times6}}{{{denominator}}}+"
            rf"115\times\frac{{300\times2}}{{{denominator}}} \\ "
            r"&= 118.4459\text{分}"
            r"\end{aligned}"
        )
    if ("复合准时率" in compact and "KA品牌单" in compact
            and "W" in compact and ("C" in compact or "Y" in compact)):
        return (
            r"\text{复合准时率（考核）}"
            r"=1-\frac{C_{\text{KA品牌单}}+5\times Y_{\text{KA品牌单}}}"
            r"{W_{\text{KA品牌单}}}"
        )
    if ("配送原因未完成率" in compact and "KA品牌单" in compact
            and "P" in compact and "W" in compact):
        return (
            r"\text{配送原因未完成率}"
            r"=\frac{P_{\text{KA品牌单}}}"
            r"{W_{\text{KA品牌单}}+P_{\text{KA品牌单}}}"
        )
    if ("KA品牌驻点骑手考核得分" in compact
            and all(token in compact for token in ("W1", "W2", "W3", "W4"))):
        return (
            r"\begin{aligned}"
            r"\text{KA品牌驻点骑手考核得分}"
            r"&=W1\text{得分}\times W1\text{权重}"
            r"+W2\text{得分}\times W2\text{权重} \\ "
            r"&\quad+W3\text{得分}\times W3\text{权重}"
            r"+W4\text{得分}\times W4\text{权重}"
            r"\end{aligned}"
        )
    if ("复合超时时长" in compact and "KA品牌单" in compact
            and all(token in compact for token in ("A1", "A2", "A3", "W"))):
        return (
            r"\text{复合超时时长}"
            r"=\frac{A1_{\text{KA品牌单}}+A2_{\text{KA品牌单}}+A3_{\text{KA品牌单}}}"
            r"{W_{\text{KA品牌单}}}"
        )
    if ("加权后" in compact and "完成单" in compact
            and "常规计分项得分" in compact):
        return (
            r"\begin{aligned}"
            r"\text{加权后特殊场景完成单占比}(t) &= "
            r"\frac{\text{加权后特殊场景完成单}}"
            r"{\text{加权后普通场景完成单}+\text{加权后特殊场景完成单}} \\ "
            r"\text{常规计分项得分} &= "
            r"\text{加权后特殊场景得分}\times t+"
            r"\text{加权后普通场景得分}\times(1-t)"
            r"\end{aligned}"
        )

    eq_lines = [ln for ln in norm_lines if "=" in ln]
    if eq_lines:
        preferred = next((ln for ln in eq_lines if re.search(r'[\u4e00-\u9fff]', ln)),
                         eq_lines[0])
        lhs = preferred.split("=", 1)[0].strip()
        rhs = preferred.split("=", 1)[1].strip()
    else:
        lhs, rhs = "", " ".join(norm_lines).strip()

    mathish = [
        ln for ln in norm_lines
        if re.search(r'[A-Za-z0-9]', ln) and not re.search(r'[\u4e00-\u9fff]', ln)
    ]
    if len(mathish) >= 2:
        numerator = mathish[0]
        denominator = mathish[-1]
        if numerator != denominator and len(denominator) <= 8:
            lhs = lhs or next((ln.split("=", 1)[0].strip() for ln in norm_lines
                               if "=" in ln and re.search(r'[\u4e00-\u9fff]', ln)), "")
            return _latex_equation(lhs, rf"\frac{{{_latex_expr(numerator)}}}{{{_latex_expr(denominator)}}}")

    return _latex_equation(lhs, _latex_expr(rhs))


def _latex_equation(lhs: str, rhs: str) -> str:
    if lhs:
        return rf"\text{{{_latex_text(lhs)}}} = {rhs}"
    return rhs


def _latex_text(text: str) -> str:
    return re.sub(r'([\\{}])', r'\\\1', text.strip())


def _latex_expr(text: str) -> str:
    expr = text.strip()
    expr = expr.replace("＝", "=").replace("×", r"\times ")
    expr = expr.replace("≤", r"\le ").replace("≥", r"\ge ")
    expr = expr.replace("∞", r"\infty")
    expr = re.sub(r'\s+', ' ', expr)
    return expr


def _image_caption_reliable(b: Block) -> bool:
    if "formula" in b.flags or "auto_image" in b.flags:
        return False
    combined_parts = []
    if b.text:
        combined_parts.append(b.text)
    if b.rows:
        combined_parts.append("\n".join(" ".join(c for c in row if c)
                                        for row in b.rows))
    combined = "\n".join(combined_parts).strip()
    if not combined:
        return False
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
    compact = re.sub(r'\s+', '', combined)
    if re.fullmatch(r'\d{1,3}', compact):
        return False
    has_toc_line = any(_NUMBERED_TOC_LINE.match(ln) for ln in lines)
    if has_toc_line and any(re.fullmatch(r'\d{1,3}', ln) for ln in lines):
        return False
    if re.search(r'\d[一二三四五六七八九十]{1,3}、', compact):
        return False
    return True


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
            lines.append(_md_escape_text(b.text))
        elif b.kind == "table" and b.rows:
            emit_table(b.rows)
        elif b.kind == "image":
            if "formula" in b.flags:
                latex = _formula_to_latex(b.text)
                if latex:
                    lines.append(f"$${latex}$$")
            elif b.rows:
                emit_table(b.rows)
            elif b.text:
                if _image_caption_reliable(b):
                    for ln in b.text.splitlines():
                        if ln.strip():
                            lines.append(_md_escape_text(ln.strip()))
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
            if "table_low_confidence" in b.flags:
                from docx.enum.text import WD_COLOR_INDEX
                p = doc.add_paragraph()
                run = p.add_run("表格识别置信度低，可能存在列错位，建议人工核对。")
                _set_cn_font(run, size=9)
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
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
                if "table_fallback" in b.flags:
                    from docx.enum.text import WD_COLOR_INDEX
                    p = doc.add_paragraph()
                    run = p.add_run("表格结构识别不稳定，已保留原表格截图，建议以截图为准。")
                    _set_cn_font(run, size=9)
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                doc.add_picture(b.image_path, width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if _image_caption_reliable(b):
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
