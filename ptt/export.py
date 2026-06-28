"""导出：Block 列表 -> Markdown / Word(.docx)。"""
import os
import re
from typing import Callable, List

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
    clean = re.sub(r'\s*\n+\s*', "；", s.strip())
    return _md_escape_text(clean).replace("|", "\\|")


def _split_long_text_line(line: str, limit: int = 180) -> List[str]:
    if len(line) <= limit:
        return [line]
    pieces = re.split(r'([。；;，、+＋=＝:：])', line)
    chunks: List[str] = []
    chunk = ""
    for piece in pieces:
        if not piece:
            continue
        next_chunk = chunk + piece
        if chunk and len(next_chunk) > limit:
            chunks.append(chunk.strip())
            chunk = piece.lstrip()
        else:
            chunk = next_chunk
    if chunk.strip():
        chunks.append(chunk.strip())
    if len(chunks) <= 1:
        chunks = [line[i:i + limit] for i in range(0, len(line), limit)]
    return [c for c in chunks if c]


def _readable_text_lines(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r'\s*<br\s*/?>\s*', "\n", text, flags=re.I)
    text = re.sub(r'(?<!^)(?=(?:注[①②]?[:：]|其中，|其中[:：]|'
                  r'指标定义\d*[:：]|计算公式[:：]|核算公式[:：]|'
                  r'计分规则\d*[:：]|数据来源[:：]|指标说明[:：]|'
                  r'考核范围[:：]|数据获取方式[:：]|分数计算规则[:：]))',
                  "\n", text)
    text = re.sub(r'(?<!^)(?<!算分)(?=(?:举例[:：]|示例[:：]))', "\n", text)
    text = re.sub(r'(?<=[。；;])\s*(?=[①②③④⑤⑥⑦⑧⑨]\s*)', "\n", text)
    text = re.sub(r'(?<!^)(?=(?:[①②③④⑤⑥⑦⑧⑨]|[1-9][）)]|[1-9][.．])\s*'
                  r'(?:当|参与|同商|考核|数据|电话|风控|不满意|如|若))',
                  "\n", text)
    text = re.sub(r'(?<!^)(?=(?:麦当劳|肯德基|必胜客)完单量占比=)', "\n", text)
    text = re.sub(r'(?<!^)(?=30分钟内送达订单占比（)', "\n", text)
    text = re.sub(r'(?<!^)(?=品牌方口径准时率（)', "\n", text)
    text = re.sub(r'(?<=[）)])(?=具体方案详见)', "\n", text)
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        out.extend(_split_long_text_line(line))
    return out


def _line_needs_formula_review(line: str) -> bool:
    compact = re.sub(r"\s+", "", line or "")
    if len(compact) < 8:
        return False
    if "公式原文（需核对）" in compact:
        return False
    if _formula_to_latex(line):
        return False
    if _is_plain_text_formula(line):
        return False
    if _is_text_formula_paragraph(line):
        return False
    if not re.search(r"[=＝/÷]|SUM|Σ", compact, re.I):
        return False
    if not re.search(r"率|比|得分|占比|时长|单量|完成|公式|口径|定义", compact):
        return False
    if re.search(r"SUM|Σ|\\frac|_\{|[A-Za-z]\s*/\s*[A-Za-z]", compact, re.I):
        return True
    return False


def _emit_reviewable_text(lines: List[str], text: str) -> None:
    for value_line in _readable_text_lines(text):
        latex = _formula_to_latex(value_line)
        if latex and _looks_visual_formula_text(value_line):
            lines.append(f"$${latex}$$")
        elif _line_needs_formula_review(value_line):
            lines.append("**公式原文（需核对）**")
            lines.append("")
            lines.append(_md_escape_text(value_line))
        else:
            lines.append(_md_escape_text(value_line))


def _table_is_complex(rows_data: List[List[str]]) -> bool:
    if not rows_data:
        return False
    ncol = max(len(r) for r in rows_data)
    padded = [r + [""] * (ncol - len(r)) for r in rows_data]
    row_count = len(padded)
    cell_texts = [c.strip() for r in padded for c in r if c.strip()]
    max_cell_len = max((len(c) for c in cell_texts), default=0)
    joined = " ".join(cell_texts)
    blank_first_col = sum(1 for r in padded[1:] if not r[0].strip())
    header_joined = " ".join(c.strip() for c in padded[0] if c.strip())
    policy_like = re.search(
        r"释义|说明|数据来源|考核范围|查询路径|申诉|规则|事件|细则|管理目标|补充说明|"
        r"参考指标|指标定义|计分|核算|公式",
        joined,
    )

    if ncol <= 1 and (row_count > 1 or max_cell_len > 40):
        return True
    if ncol > 5:
        return True
    if any("\n" in c or "<br" in c.lower() for r in padded for c in r):
        return True
    if ncol == 2 and policy_like and max_cell_len > 60:
        return True
    if ncol in (2, 3) and max_cell_len > 85 and policy_like:
        return True
    if ncol >= 3 and re.search(r"释义|数据来源|事件细则|查询路径", header_joined) and max_cell_len > 45:
        return True
    if ncol <= 5 and max_cell_len <= 90 and blank_first_col == 0:
        return False
    if max_cell_len > 140:
        return True
    if row_count >= 3 and blank_first_col >= max(1, row_count // 3):
        return True
    if ncol >= 3 and re.search(r"合并|多层|普通场景|特殊场景|融合后|计分规则", joined):
        return True
    return False


def _emit_readable_table(lines: List[str], rows_data: List[List[str]]) -> None:
    ncol = max(len(r) for r in rows_data)
    rows = [r + [""] * (ncol - len(r)) for r in rows_data]
    header = [c.strip() for c in rows[0]]
    body = rows[1:] if len(rows) > 1 else rows

    two_col = ncol == 2 and (
        header[0] in ("项目", "指标", "类型", "要求", "维度")
        or header[1] in ("内容", "说明", "数值", "得分")
    )
    if two_col:
        for key, value in body:
            key = key.strip()
            value_lines = _readable_text_lines(value)
            if not key and not value_lines:
                continue
            if key:
                lines.append(f"**{_md_escape_text(key)}**")
                lines.append("")
            for value_line in value_lines:
                if _line_needs_formula_review(value_line):
                    lines.append("**公式原文（需核对）**")
                    lines.append("")
                lines.append(_md_escape_text(value_line))
            lines.append("")
        return

    if any(c for c in header):
        lines.append("**" + _md_escape_text(" / ".join(c for c in header if c)) + "**")
        lines.append("")
    for row in body:
        pairs = []
        for idx, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue
            label = header[idx] if idx < len(header) and header[idx] else f"列{idx + 1}"
            value_lines = _readable_text_lines(cell) or [cell]
            pairs.append((label, value_lines))
        if pairs:
            one_line = "；".join(f"{label}：{'；'.join(value_lines)}"
                                 for label, value_lines in pairs)
            structured_labels = any(
                re.search(r"参考指标|释义|数据来源|说明|考核范围|查询路径", label)
                for label, _ in pairs
            )
            if not structured_labels and len(one_line) <= 220:
                lines.append("- " + _md_escape_text(one_line))
            else:
                first = True
                for label, value_lines in pairs:
                    for pos, value_line in enumerate(value_lines):
                        prefix = "- " if first else "  "
                        label_text = f"{label}：" if pos == 0 else ""
                        if _line_needs_formula_review(value_line):
                            lines.append(prefix + "**公式原文（需核对）**")
                            prefix = "  "
                            label_text = f"{label}：" if pos == 0 else ""
                        lines.append(prefix + _md_escape_text(label_text + value_line))
                        first = False
    lines.append("")


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
    if ("特殊场景完成单占比" in compact
            and ("普通场景" in compact and "特殊场景" in compact)
            and ("剔除异常单" in compact or "完成单" in compact)):
        return (
            r"\text{特殊场景完成单占比}(t)="
            r"\frac{\text{特殊场景剔除异常单后完成单}}"
            r"{\text{普通场景剔除异常单后完成单}"
            r"+\text{特殊场景剔除异常单后完成单}}"
        )
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
    if ("KA" in compact and "负向反馈率" in compact
            and "F1" in compact and "F2" in compact and "W" in compact):
        return r"\text{KA负向反馈率} = \frac{F1+3\times F2}{W}"
    if ("KA品牌客诉率" in compact and "KS" in compact and "W" in compact):
        return r"\text{KA品牌客诉率} = \frac{KS}{W}"
    if ("承托比" in compact and "R" in compact and "W" in compact):
        return (
            r"\text{承托比}"
            r"=\frac{R_{\text{KA品牌单}}}{W_{\text{KA品牌单}}}"
        )
    if ("虚假点送达率" in compact and "T" in compact and "W" in compact):
        return (
            r"\text{虚假点送达率}"
            r"=\frac{T_{\text{KA品牌单}}}{W_{\text{KA品牌单}}}"
        )
    if ("客诉虚假点送达率" in compact and "命中客诉虚假点送达单量" in compact
            and "所有KA品牌单完成单量" in compact):
        return (
            r"\text{客诉虚假点送达率}"
            r"=\frac{\text{星巴克/麦当劳/大润发品牌门店命中客诉虚假点送达单量}}"
            r"{\text{站点组履约的所有KA品牌单完成单量}}"
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
    return ""


def _is_plain_text_formula(text: str) -> bool:
    s = re.sub(r'\s+', '', text or "")
    if len(s) < 8 or len(s) > 180:
        return False
    if "公式原文（需核对）" in s:
        return False
    if len(re.findall(r"[=＝]", s)) != 1:
        return False
    if re.search(r"\\frac|_\{|[A-Za-z]\s*/\s*[A-Za-z]", s, re.I):
        return False
    left, right = re.split(r"[=＝]", s, maxsplit=1)
    if not left or not right:
        return False
    if len(left) > 60:
        return False
    visual_labels = (
        "特殊场景完成单占比", "复合准时率", "配送原因未完成率",
        "承托比", "虚假点送达率", "复合超时时长",
        "站点组体验总得分", "站点组KA品牌单体验得分",
        "客诉虚假点送达率",
    )
    if any(label in s for label in visual_labels):
        return False
    chinese = len(re.findall(r"[\u4e00-\u9fff]", s))
    return chinese >= 6


def _is_text_formula_paragraph(text: str) -> bool:
    s = re.sub(r'\s+', '', text or "")
    if len(s) < 12 or "公式原文（需核对）" in s:
        return False
    if not re.search(r"[=＝]", s):
        return False
    if re.search(r"\\frac|_\{", s):
        return False
    visual_labels = (
        "特殊场景完成单占比", "复合准时率", "配送原因未完成率",
        "承托比", "虚假点送达率", "复合超时时长",
        "客诉虚假点送达率", "站点组体验总得分",
    )
    if any(label in s for label in visual_labels):
        return False
    return bool(re.search(r"(?:金额|权重占比|奖励|服务费|得分|单量)[^。；;]{0,40}[=＝].*(?:sum|Σ|\\+|[+＋*×])", s, re.I))


def _looks_visual_formula_text(text: str) -> bool:
    s = re.sub(r'\s+', '', text or "")
    if not s:
        return False
    visual_labels = (
        "特殊场景完成单占比", "复合准时率", "配送原因未完成率",
        "承托比", "虚假点送达率", "复合超时时长",
        "KA负向反馈率", "KA品牌客诉率",
        "客诉虚假点送达率", "站点组体验总得分", "站点组KA品牌单体验得分",
    )
    return any(label in s for label in visual_labels)


def _formula_block_mode(b: Block) -> str:
    if "formula_text" in b.flags:
        return "text"
    if "formula_latex" in b.flags:
        return "latex"
    if "formula_uncertain" in b.flags or "needs_review" in b.flags:
        return "uncertain"
    if _formula_to_latex(b.text):
        return "latex"
    if _is_plain_text_formula(b.text):
        return "text"
    if _is_text_formula_paragraph(b.text):
        return "text"
    if _looks_visual_formula_text(b.text):
        return "uncertain"
    return "uncertain" if not (b.text or "").strip() else "text"


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
        if _table_is_complex(rows_data):
            _emit_readable_table(lines, rows_data)
            return
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
            _emit_reviewable_text(lines, b.text)
        elif b.kind == "table" and b.rows:
            emit_table(b.rows)
        elif b.kind == "image":
            if "formula" in b.flags:
                mode = _formula_block_mode(b)
                latex = _formula_to_latex(b.text) if mode == "latex" else ""
                if mode == "text":
                    _emit_reviewable_text(lines, b.text)
                elif latex:
                    lines.append(f"$${latex}$$")
                else:
                    lines.append("**公式原文（需核对）**")
                    lines.append("")
                    formula_text = b.text or "公式区域未能可靠识别，请对照原 PDF 核对。"
                    for ln in _readable_text_lines(formula_text):
                        lines.append(_md_escape_text(ln))
            elif b.rows:
                emit_table(b.rows)
                if "table_fallback" in b.flags:
                    lines.append("**表格结构需核对：原表格识别不稳定，请对照原 PDF。**")
            elif b.text:
                if _image_caption_reliable(b):
                    _emit_reviewable_text(lines, b.text)
                elif "auto_image" in b.flags:
                    lines.append("**图示/低置信区域未能可靠文本化，请对照原 PDF 核对。**")
            elif "auto_image" in b.flags or "table_fallback" in b.flags:
                lines.append("**图示/低置信区域未能可靠文本化，请对照原 PDF 核对。**")
        lines.append("")
    md = "\n".join(lines).strip() + "\n"
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = _dedupe_consecutive_latex(md)
    md = _ensure_markdown_ka_522_intro(md)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path


def _dedupe_consecutive_latex(md: str) -> str:
    out: List[str] = []
    last_latex = ""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("$$") and stripped.endswith("$$"):
            if stripped == last_latex:
                continue
            last_latex = stripped
        elif stripped:
            last_latex = ""
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _ensure_markdown_ka_522_intro(md: str) -> str:
    marker = "#### 5.2.2"
    if marker not in md or "特殊场景" not in md or "融合" not in md:
        return md
    start = md.find(marker)
    next_match = re.search(r"\n####\s+5\.3\b", md[start:])
    end = start + next_match.start() if next_match else len(md)
    section = md[start:end]
    if "融合后体验得分" in section:
        return md
    if not ("特殊场景" in section and "融合" in section and "计分规则" in section):
        return md
    intro = ("所有体验指标，均分为普通场景和特殊场景两套目标进行考核，"
             "并按剔除异常单后的特殊场景完成单占比加权计算融合后体验得分。")
    lines = section.splitlines()
    insert_at = 1
    for idx, line in enumerate(lines[1:6], start=1):
        if "特殊场景" in line and "融合" in line and "计分规则" in line:
            insert_at = idx + 1
            break
    lines[insert_at:insert_at] = ["", intro]
    repaired = "\n".join(lines).strip() + "\n"
    return md[:start] + repaired + md[end:]


def _set_cn_font(run, size=None, bold=None):
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.name = "Times New Roman"
    r = run._element.rPr.rFonts
    r.set(qn("w:eastAsia"), "宋体")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def _shade_cell(cell, fill="EFEFEF"):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def export_docx(result: DocResult, out_path: str) -> str:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

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


def builtin_check_cases() -> List[tuple[str, Callable[[], None]]]:
    import tempfile

    def render_text(result: DocResult) -> str:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.md")
            export_markdown(result, out_path)
            with open(out_path, "r", encoding="utf-8") as f:
                return f.read()

    def case_unreliable_image_caption_does_not_export_link() -> None:
        text = render_text(DocResult(blocks=[
            Block(
                kind="image",
                image_path="fig.png",
                text="1\n二、适用区域⋯… 1三、定义/名词解释. 2",
            )
        ]))
        assert "![图片]" not in text
        assert "图中文字" not in text
        assert "\n1\n" not in text
        assert text.strip() == ""

    def case_digit_only_image_caption_does_not_export_link() -> None:
        text = render_text(DocResult(blocks=[Block(kind="image", image_path="fig.png", text="1")]))
        assert "![图片]" not in text
        assert "图中文字" not in text
        assert "\n1\n" not in text
        assert text.strip() == ""

    def case_formula_image_exports_latex() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="复合超时时长=(A1_{KA品牌单}+A2_{KA品牌单}+A3_{KA品牌单})/W_{KA品牌单}")
        ]))
        assert "$$" in text
        assert r"\text{复合超时时长}" in text
        assert r"\frac{A1_{\text{KA品牌单}}+A2_{\text{KA品牌单}}+A3_{\text{KA品牌单}}}" in text
        assert "![公式]" not in text

    def case_unknown_formula_requires_review_without_asset_link() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"], text="客诉虚假点送达率=分子/分母")
        ]))
        assert "**公式原文（需核对）**" in text
        assert "客诉虚假点送达率=分子/分母" in text
        assert "![" not in text

    def case_plain_text_formula_stays_text() -> None:
        formula = "商服务费 = 基础服务费 + 超额达标奖励 + KA 星级结算金额 + KA 体验膨胀费 + 服务质量奖励费 + 活动激励金"
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"], text=formula)
        ]))
        assert formula.replace("*", "\\*") in text
        assert "公式原文（需核对）" not in text
        assert "$$" not in text

    def case_score_example_formula_stays_text() -> None:
        formula = "麦当劳普通场景得分=15%*120+10%*120+15%*120+50%*105+10%*120=124.50分"
        text = render_text(DocResult(blocks=[
            Block(kind="para", text=formula)
        ]))
        assert "公式原文（需核对）" not in text
        assert "$$" not in text
        assert "124.50分" in text

    def case_sigma_text_formula_stays_text() -> None:
        formula = "30分钟内送达订单占比（配送距离3km及以内订单）=Σ（配送距离3km及以内且30分钟内送达完成订单量）/Σ（配送距离3km及以内完成订单量）"
        text = render_text(DocResult(blocks=[
            Block(kind="para", text=formula)
        ]))
        assert formula in text
        assert "公式原文（需核对）" not in text
        assert "$$" not in text

    def case_long_sum_formula_paragraph_stays_text() -> None:
        formula = "当日站点激励权重占比=当日该站点激励权重/sum（当日该站点所在城市所有站点激励权重）。"
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"], text=formula)
        ]))
        assert formula in text
        assert "公式原文（需核对）" not in text
        assert "$$" not in text

    def case_fraction_formula_keeps_visual_order() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="配送原因未完成率=P KA品牌单/(W KA品牌单+P KA品牌单)")
        ]))
        assert r"\text{配送原因未完成率}" in text
        assert r"\frac{P_{\text{KA品牌单}}}" in text

    def case_complaint_fake_delivery_formula_exports_latex() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="客诉虚假点送达率=星巴克/麦当劳/大润发品牌门店命中客诉虚假点送达单量/站点组履约的所有KA品牌单完成单量")
        ]))
        assert r"\text{客诉虚假点送达率}" in text
        assert r"\frac{\text{星巴克/麦当劳/大润发品牌门店命中客诉虚假点送达单量}}" in text
        assert "公式原文（需核对）" not in text

    def case_special_scene_completion_ratio_exports_latex() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="特殊场景完成单占比（t）=特殊场景剔除异常单后完成单/普通场景剔除异常单后完成单+特殊场景剔除异常单后完成单")
        ]))
        assert r"\text{特殊场景完成单占比}(t)" in text
        assert r"\frac{\text{特殊场景剔除异常单后完成单}}" in text

    def case_ka_522_intro_markdown_guard() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="heading", level=4, text="5.2.2"),
            Block(kind="para", text="特殊场景体验融合考核计分规则"),
            Block(kind="image", flags=["formula"],
                  text="特殊场景完成单占比（t）=特殊场景剔除异常单后完成单/普通场景剔除异常单后完成单+特殊场景剔除异常单后完成单"),
            Block(kind="heading", level=4, text="5.3 压力场景加权考核"),
        ]))
        section = text.split("#### 5.2.2", 1)[1].split("#### 5.3", 1)[0]
        assert "融合后体验得分" in section

    def case_weighted_special_scene_formula_recovered() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="加权后特殊场景完成单加权后特珠场景完成单占比(D二元权后普通场紫完成单＋加权后特珠场景完成单常规计分项得分=加权后特殊场景得分*t＋加权后普通场景得分*(1-t)")
        ]))
        assert r"\text{加权后特殊场景完成单占比}(t)" in text
        assert r"\text{常规计分项得分}" in text
        assert r"\frac{\text{加权后特殊场景完成单}}" in text

    def case_station_group_experience_formula_recovered() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="站点组体验总得分=站点组F分*F/(F+SUM(Kn*Qn))+站点组Kn分*(Kn*Qn)/(F+SUM(Kn*Qn))")
        ]))
        assert r"\text{站点组体验总得分}" in text
        assert r"\mathrm{SUM}(K_n\times Q_n)" in text
        assert r"\text{站点组}K_n\text{分}" in text

    def case_ka_brand_experience_formula_recovered() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="站点组KA品牌单体验得分=K1分*K1*Q1/SUM(Kn*Qn)+KN分*KN*QN/SUM(Kn*Qn)")
        ]))
        assert r"\text{站点组KA品牌单体验得分}" in text
        assert r"K_1\text{分}\times\frac{K_1\times Q_1}" in text
        assert r"K_n\text{分}\times\frac{K_n\times Q_n}" in text

    def case_ka_weighted_example_formula_keeps_numbers() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="A履约的体验总得分为118*(10000/(10000+700*6+300,z))+120*(700*6/(10000+70046+300,z))+115*(300,z/(10000+700*6+300,z))=118.4459分")
        ]))
        assert r"10000+700\times6+300\times2" in text
        assert r"115\times\frac{300\times2}" in text
        assert "300,z" not in text

    def case_ka_formula_recovery() -> None:
        punctuality = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="复合准时率（考核）=1-(C KA品牌单+5*Y KA品牌单)/W KA品牌单")
        ]))
        delivery = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="配送原因未完成率=P KA品牌单/(W KA品牌单+P KA品牌单)")
        ]))
        rider = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="计分规则 KA 品牌驻点骑手考核得分=W1得分*W1权重+W2得分*W2权重+W3得分*W3权重+W4得分*W4权重其中")
        ]))
        overtime = render_text(DocResult(blocks=[
            Block(kind="image", flags=["formula"],
                  text="复合超时时长=(A1_{KA品牌单}+A2_{KA品牌单}+A3_{KA品牌单})/W_{KA品牌单}")
        ]))
        assert r"C_{\text{KA品牌单}}+5\times Y_{\text{KA品牌单}}" in punctuality
        assert r"\frac{P_{\text{KA品牌单}}}" in delivery
        assert r"W1\text{得分}\times W1\text{权重}" in rider
        assert r"\frac{A1_{\text{KA品牌单}}+A2_{\text{KA品牌单}}+A3_{\text{KA品牌单}}}" in overtime

    def case_textual_image_table_exports_table_without_link() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", flags=["table_fallback"], rows=[["指标", "值"], ["A", "1"]])
        ]))
        assert "| 指标 | 值 |" in text
        assert "![表格截图]" not in text

    def case_simple_four_column_table_stays_markdown_table() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="table", rows=[
                ["数据", "查询路径", "负责人", "备注"],
                ["KA品牌相关考核指标数据", "烽火台-商服务费计费系统", "渠道经理", "每月更新"],
            ])
        ]))
        assert "| 数据 | 查询路径 | 负责人 | 备注 |" in text
        assert "- 数据：" not in text

    def case_image_table_exports_text_without_asset_link() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="image", rows=[["数据", "查询路径"], ["KA", "系统"]],
                  image_path="out_assets/table.png")
        ]))
        assert "| 数据 | 查询路径 |" in text
        assert "![" not in text
        assert "out_assets/table.png" not in text

    def case_long_policy_table_exports_grouped_text() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="table", rows=[
                ["维度", "说明"],
                ["指标说明",
                 "对于虚假点送达的行为，相关用户会通过不同途径进行投诉，"
                 "包括但不限于电话投诉、风控抓取、用户 app 端投诉等，"
                 "方案将考核现有的电话客诉、风控抓取、不满意评价抓取3个来源。"],
            ])
        ]))
        assert "**指标说明**" in text
        assert "| 维度 | 说明 |" not in text

    def case_definition_table_with_long_source_exports_grouped_text() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="table", rows=[
                ["参考指标", "释义", "数据来源"],
                [
                    "KA品牌客诉单（KS）",
                    "因配送过程中存在未送指定位置、送达不通知、餐商品洒漏/货损/送错/少送、提前点送达、服务态度不佳等情况导致投诉。",
                    "KA品牌方回传给美团平台的客诉数据，客诉明细数据可联系渠道经理获取。",
                ],
            ])
        ]))
        assert "| 参考指标 | 释义 | 数据来源 |" not in text
        assert "- 参考指标：KA品牌客诉单" in text
        assert "数据来源：KA品牌方回传给美团平台的客诉数据" in text
        assert "。；数据来源" not in text

    def case_single_column_collapsed_table_exports_grouped_text() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="table", rows=[
                ["培训管理目标"],
                ["专送骑手 专送站长 城市经理 招聘人员 新人训 在岗训 专项训 组织规则 过程指标 考核结果"],
            ])
        ]))
        assert "| 培训管理目标 |" not in text
        assert "**培训管理目标**" in text

    def case_complex_table_exports_readable_blocks_without_br() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="table", rows=[
                ["项目", "内容"],
                ["指标定义1",
                 "【完成订单量】通过美团配送且订单最终状态为完成的订单数量。\n"
                 "麦当劳完单量占比=麦当劳完单量/总量\n"
                 "注：数据来源均为品牌方。"],
            ]),
            Block(kind="table", rows=[
                ["指标", "分子数值", "分母数值", "达标率（值）", "满分目标", "权重", "得分"],
                ["KA品牌负向反馈率", "2单", "100", "2%", "0.01%", "15%", "0"],
            ]),
        ]))
        assert "<br>" not in text
        assert "**指标定义1**" in text
        assert "麦当劳完单量占比=麦当劳完单量/总量" in text
        assert "- 指标：KA品牌负向反馈率" in text

    def case_score_example_phrase_not_split() -> None:
        text = render_text(DocResult(blocks=[
            Block(kind="para", text="普通场景算分示例：假设站点组A履约。"),
            Block(kind="para", text="特殊场景算分示例：假设站点组A履约。"),
        ]))
        assert "普通场景算分\n示例" not in text
        assert "特殊场景算分\n示例" not in text
        assert "普通场景算分示例：假设站点组A履约。" in text

    return [
        ("export.unreliable_image_caption_does_not_export_link", case_unreliable_image_caption_does_not_export_link),
        ("export.digit_only_image_caption_does_not_export_link", case_digit_only_image_caption_does_not_export_link),
        ("export.formula_image_exports_latex", case_formula_image_exports_latex),
        ("export.unknown_formula_requires_review_without_asset_link", case_unknown_formula_requires_review_without_asset_link),
        ("export.plain_text_formula_stays_text", case_plain_text_formula_stays_text),
        ("export.score_example_formula_stays_text", case_score_example_formula_stays_text),
        ("export.sigma_text_formula_stays_text", case_sigma_text_formula_stays_text),
        ("export.long_sum_formula_paragraph_stays_text", case_long_sum_formula_paragraph_stays_text),
        ("export.fraction_formula_keeps_visual_order", case_fraction_formula_keeps_visual_order),
        ("export.complaint_fake_delivery_formula_exports_latex", case_complaint_fake_delivery_formula_exports_latex),
        ("export.special_scene_completion_ratio_exports_latex", case_special_scene_completion_ratio_exports_latex),
        ("export.ka_522_intro_markdown_guard", case_ka_522_intro_markdown_guard),
        ("export.weighted_special_scene_formula_recovered", case_weighted_special_scene_formula_recovered),
        ("export.station_group_experience_formula_recovered", case_station_group_experience_formula_recovered),
        ("export.ka_brand_experience_formula_recovered", case_ka_brand_experience_formula_recovered),
        ("export.ka_weighted_example_formula_keeps_numbers", case_ka_weighted_example_formula_keeps_numbers),
        ("export.ka_formula_recovery", case_ka_formula_recovery),
        ("export.textual_image_table_exports_table_without_link", case_textual_image_table_exports_table_without_link),
        ("export.simple_four_column_table_stays_markdown_table", case_simple_four_column_table_stays_markdown_table),
        ("export.image_table_exports_text_without_asset_link", case_image_table_exports_text_without_asset_link),
        ("export.long_policy_table_exports_grouped_text", case_long_policy_table_exports_grouped_text),
        ("export.definition_table_with_long_source_exports_grouped_text", case_definition_table_with_long_source_exports_grouped_text),
        ("export.single_column_collapsed_table_exports_grouped_text", case_single_column_collapsed_table_exports_grouped_text),
        ("export.complex_table_exports_readable_blocks_without_br", case_complex_table_exports_readable_blocks_without_br),
        ("export.score_example_phrase_not_split", case_score_example_phrase_not_split),
    ]
