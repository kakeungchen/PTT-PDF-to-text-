"""Source-to-output coverage checks for converted PDF content.

The normal OCR/Markdown checks catch garbled text after export.  This module
adds the missing opposite direction: important source evidence must survive in
the exported Markdown, otherwise the conversion is not good enough to pass.
"""
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import fitz

from .export import _table_is_complex, export_markdown
from .models import Block, DocResult


_SOURCE_HEADING_RE = re.compile(
    r"^\s*((?:\d+(?:[.．]\d+){0,4})|[一二三四五六七八九十]{1,3}[、.．])\s*"
    r"([A-Za-z0-9\u4e00-\u9fff（）()【】《》·、/\-]{2,80})\s*$"
)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:[.,]\d+)?\s*(?:%|％|s|S|秒|分|单|天|公里|km|KM)?"
)
_FORMULA_LINE_RE = re.compile(r"[=＝]|SUM|Σ", re.I)

_KEY_TERMS = [
    "考核目标",
    "考核目标举例",
    "核算公式",
    "计算公式",
    "计分公式",
    "计分规则",
    "指标定义",
    "得分范围",
    "数据来源",
    "普通场景算分示例",
    "特殊场景算分示例",
    "融合后体验得分",
    "特殊场景完成单占比",
    "完成单占比",
    "分子数值",
    "分母数值",
    "达标率",
    "满分目标",
    "权重",
]

_HIGH_RISK_TERMS = {
    "考核目标",
    "考核目标举例",
    "核算公式",
    "计算公式",
    "计分公式",
    "指标定义",
    "得分范围",
    "普通场景算分示例",
    "特殊场景算分示例",
    "融合后体验得分",
}


@dataclass
class SourceSection:
    token: str
    title: str
    text: str
    line_no: int = 1


def audit_pdf_markdown_coverage(pdf_path: str,
                                result: DocResult,
                                markdown_text: str) -> List[str]:
    """Return blocking coverage/readability issues.

    The check is deliberately evidence based: only terms, formulas, numbers, and
    section headings seen in the source evidence are required in the Markdown.
    """
    source_text = _source_text(pdf_path, result.blocks)
    issues: List[str] = []
    issues.extend(_check_markdown_readability(markdown_text))
    issues.extend(_check_table_block_rendering(result.blocks, markdown_text))
    issues.extend(_check_image_block_coverage(result.blocks, markdown_text))
    issues.extend(_check_ka_required_content(markdown_text))
    issues.extend(_check_section_coverage(source_text, markdown_text))
    issues.extend(_check_global_key_term_coverage(source_text, markdown_text))
    return _dedupe(issues)


def markdown_from_result(result: DocResult) -> str:
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "coverage.md")
        export_markdown(result, out)
        with open(out, "r", encoding="utf-8") as f:
            return f.read()


def _source_text(pdf_path: str, blocks: List[Block]) -> str:
    parts: List[str] = []
    pdf_text = _pdf_text(pdf_path)
    if pdf_text:
        parts.append(pdf_text)
    block_text = _blocks_text(blocks)
    if block_text:
        parts.append(block_text)
    return "\n".join(parts)


def _pdf_text(pdf_path: str) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    pages: List[str] = []
    try:
        for pno, page in enumerate(doc):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append(f"\n[第{pno + 1}页]\n{text}")
    finally:
        doc.close()
    return "\n".join(pages)


def _blocks_text(blocks: List[Block]) -> str:
    out: List[str] = []
    for blk in blocks:
        if blk.kind == "heading" and blk.text:
            out.append(blk.text)
        elif blk.kind in ("para", "image") and blk.text:
            out.append(blk.text)
        if blk.rows:
            out.extend(" ".join(c for c in row if c) for row in blk.rows)
    return "\n".join(out)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "").replace("％", "%").replace("＝", "=")


def _text_contains(haystack_compact: str, needle: str) -> bool:
    return _compact(needle) in haystack_compact


def _extract_sections(text: str) -> List[SourceSection]:
    lines = text.splitlines()
    headings: List[Tuple[int, str, str]] = []
    for idx, raw in enumerate(lines):
        line = raw.strip().lstrip("#").strip()
        m = _SOURCE_HEADING_RE.match(line)
        if not m:
            continue
        token = m.group(1).replace("．", ".")
        title = m.group(2).strip()
        if len(_compact(title)) < 2:
            continue
        headings.append((idx, token, title))

    sections: List[SourceSection] = []
    for pos, (idx, token, title) in enumerate(headings):
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[idx:end]).strip()
        if len(_compact(section_text)) < 20:
            continue
        sections.append(SourceSection(token=token, title=title,
                                      text=section_text, line_no=idx + 1))
    return sections


def _markdown_sections(markdown_text: str) -> Dict[str, str]:
    lines = markdown_text.splitlines()
    heads: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line)
        if not m:
            continue
        heads.append((idx, m.group(1).strip()))
    out: Dict[str, str] = {}
    for pos, (idx, heading) in enumerate(heads):
        end = heads[pos + 1][0] if pos + 1 < len(heads) else len(lines)
        m = _SOURCE_HEADING_RE.match(heading)
        if not m:
            continue
        token = m.group(1).replace("．", ".")
        out[token] = "\n".join(lines[idx:end])
    return out


def _check_section_coverage(source_text: str, markdown_text: str) -> List[str]:
    source_sections = _extract_sections(source_text)
    md_sections = _markdown_sections(markdown_text)
    md_all = _compact(markdown_text)
    issues: List[str] = []

    for sec in source_sections[:120]:
        evidence_text = _local_section_evidence(sec.text)
        sec_compact = _compact(evidence_text)
        md_sec = md_sections.get(sec.token, "")
        md_sec_compact = _compact(md_sec) if md_sec else md_all
        detailed = _is_detailed_section(sec.token)

        heading_mark = sec.token + sec.title
        if len(_compact(sec.title)) >= 4 and not _text_contains(md_all, heading_mark):
            if not _text_contains(md_all, sec.title):
                issues.append(f"覆盖审计: 章节标题疑似缺失 {sec.token} {sec.title}")

        if not detailed:
            continue

        missing_terms = [
            term for term in _KEY_TERMS
            if term in sec_compact and not _text_contains(md_sec_compact, term)
        ]
        missing_terms = [
            term for term in missing_terms
            if not _term_covered_by_adjacent_section(sec, term, md_all)
        ]
        high_missing = [t for t in missing_terms if t in _HIGH_RISK_TERMS]
        if high_missing:
            issues.append(
                f"覆盖审计: {sec.token} {sec.title} 缺少关键内容 "
                + "、".join(high_missing[:6])
            )
        elif len(missing_terms) >= 3:
            issues.append(
                f"覆盖审计: {sec.token} {sec.title} 表格/定义字段疑似缺失 "
                + "、".join(missing_terms[:6])
            )

        missing_numbers = _missing_numeric_anchors(evidence_text, md_sec or markdown_text)
        if len(missing_numbers) >= 4:
            issues.append(
                f"覆盖审计: {sec.token} {sec.title} 数字/比例疑似缺失 "
                + "、".join(missing_numbers[:8])
            )

        missing_formulas = _missing_formula_left_sides(evidence_text, md_sec or markdown_text)
        if missing_formulas:
            issues.append(
                f"覆盖审计: {sec.token} {sec.title} 公式左侧疑似缺失 "
                + "、".join(missing_formulas[:4])
            )

    return issues


def _term_covered_by_adjacent_section(sec: SourceSection, term: str,
                                      md_all_compact: str) -> bool:
    if sec.token == "5.2.1" and term == "融合后体验得分":
        return (_text_contains(md_all_compact, "5.2.2特殊场景体验融合考核计分规则")
                and _text_contains(md_all_compact, term))
    return False


def _check_global_key_term_coverage(source_text: str, markdown_text: str) -> List[str]:
    source_compact = _compact(source_text)
    md_compact = _compact(markdown_text)
    issues: List[str] = []
    for term in _HIGH_RISK_TERMS:
        if term in source_compact and not _text_contains(md_compact, term):
            issues.append(f"覆盖审计: 全文关键内容疑似缺失 {term}")
    return issues


def _check_image_block_coverage(blocks: List[Block],
                                markdown_text: str) -> List[str]:
    issues: List[str] = []
    for idx, blk in enumerate(blocks):
        if blk.kind != "image":
            continue
        important = (
            "formula" in blk.flags
            or ("auto_image" in blk.flags and _auto_image_is_important(blk))
            or "table_fallback" in blk.flags
            or bool(blk.rows)
        )
        if not important:
            continue
        if _image_block_exported(blk, markdown_text):
            continue
        page = (blk.page or 0) + 1
        issues.append(f"覆盖审计: 第{page}页图像/公式区域未输出 block={idx}")
    return issues


def _check_table_block_rendering(blocks: List[Block],
                                 markdown_text: str) -> List[str]:
    issues: List[str] = []
    for idx, blk in enumerate(blocks):
        if not blk.rows:
            continue
        if _table_is_complex(blk.rows):
            continue
        if _standard_table_rendered(blk.rows, markdown_text):
            continue
        page = (blk.page or 0) + 1
        bbox = ",".join(str(round(float(v), 1)) for v in blk.bbox)
        issues.append(
            f"覆盖审计: 第{page}页标准表格未以 Markdown 表格输出 "
            f"block={idx} bbox=[{bbox}]"
        )
    return issues


def _standard_table_rendered(rows: List[List[str]], markdown_text: str) -> bool:
    if not rows:
        return False
    header = [c.strip() for c in rows[0] if c.strip()]
    if not header:
        return False
    pipe_lines = [line for line in markdown_text.splitlines()
                  if line.strip().startswith("|") and line.strip().endswith("|")]
    header_needles = [_compact(c) for c in header[:min(3, len(header))]]
    for line in pipe_lines:
        compact_line = _compact(line)
        if all(needle and needle in compact_line for needle in header_needles):
            return True
    return False


def _auto_image_is_important(blk: Block) -> bool:
    text = _compact(blk.text or "")
    if not text:
        return False
    if re.search(r"[=＝≤≥×÷*/]|SUM|Σ", text, re.I):
        return True
    important_terms = (
        "公式", "计算口径", "指标", "数据来源", "考核范围", "规则",
        "得分", "费率", "金额", "单量", "完成率", "达标率",
    )
    return any(term in text for term in important_terms)


def _image_block_exported(blk: Block, markdown_text: str) -> bool:
    md_compact = _compact(markdown_text)
    if "formula" in blk.flags and blk.text:
        latex = ""
        try:
            from .export import _formula_to_latex
            latex = _formula_to_latex(blk.text)
        except Exception:
            latex = ""
        if latex and latex in markdown_text:
            return True
        if "公式原文（需核对）" in md_compact and _text_anchors_exported(
                blk.text, md_compact):
            return True
    if "formula" in blk.flags and "公式原文（需核对）" in md_compact:
        return True
    if "table_fallback" in blk.flags and "表格结构需核对" in md_compact:
        return True
    if "auto_image" in blk.flags and "图示/低置信区域未能可靠文本化" in md_compact:
        return True
    text = _compact(blk.text or "")
    if len(text) >= 8 and text[:40] in md_compact:
        return True
    if blk.rows:
        row_text = _compact(" ".join(" ".join(row) for row in blk.rows))
        if len(row_text) >= 8 and row_text[:40] in md_compact:
            return True
        cell_hits = 0
        for row in blk.rows:
            for cell in row:
                token = _compact(cell)
                if len(token) >= 4 and token[:24] in md_compact:
                    cell_hits += 1
                    if cell_hits >= 2:
                        return True
    return False


def _text_anchors_exported(text: str, md_compact: str) -> bool:
    anchors = []
    for raw in (text or "").splitlines():
        token = _compact(raw)
        if len(token) >= 4:
            anchors.append(token[:28])
    if not anchors:
        return False
    return any(anchor in md_compact for anchor in anchors)


def _check_ka_required_content(markdown_text: str) -> List[str]:
    compact = _compact(markdown_text)
    if "KA品牌单月度考核制度" not in compact and "KA品牌单月度" not in compact:
        return []
    issues: List[str] = []

    def section(token: str) -> str:
        return _extract_markdown_section(markdown_text, token)

    sec_total = section("1.站点组体验总得分计算逻辑")
    if sec_total:
        need = ["站点组履约的KA品牌单体验总得分", "站点组体验总得分", "站点组F", "站点组Kn"]
        missing = [item for item in need if item not in _compact(sec_total)]
        has_formula = _has_formula_or_original(sec_total)
        if missing or not has_formula:
            issues.append("覆盖审计: 站点组体验总得分计算逻辑缺少说明或公式")

    formula_requirements = [
        ("5.4.1", "复合准时率"),
        ("5.4.2", "配送原因未完成率"),
        ("5.4.3", "KA负向反馈率"),
        ("5.4.4", "KA品牌客诉率"),
        ("5.4.5", "承托比"),
        ("5.4.6", "虚假点送达率"),
        ("5.4.7", "复合超时时长"),
        ("5.8", "站点组KA品牌单体验得分"),
    ]
    for token, label in formula_requirements:
        sec = section(token)
        if not sec:
            continue
        sec_compact = _compact(sec)
        if label not in sec_compact or not _has_formula_or_original(sec):
            issues.append(f"覆盖审计: {token} {label} 缺少可信公式")

    sec_546 = section("5.4.6")
    if sec_546:
        required = ["指标释义", "虚假点击送达", "指标说明", "数据来源", "电话客诉", "风控抓取"]
        missing = [item for item in required if item not in _compact(sec_546)]
        if missing:
            issues.append("覆盖审计: 5.4.6 虚假点送达率缺少上下文：" + "、".join(missing))

    sec_55 = section("5.5")
    if sec_55:
        required = ["KA品牌相关考核指标数据", "违规订单明细", "异常单申诉",
                    "配送原因未完成率", "虚假点送达率", "KA品牌驻点骑手考核申诉"]
        missing = [item for item in required if item not in _compact(sec_55)]
        if missing:
            issues.append("覆盖审计: 5.5 数据查询路径及申诉缺少：" + "、".join(missing))

    return issues


def _has_formula_or_original(section_text: str) -> bool:
    compact = _compact(section_text)
    if "公式原文（需核对）" in compact or "需对照原PDF核对" in compact:
        return False
    if "$$" in section_text and r"\frac" in section_text:
        return True
    return False


def _extract_markdown_section(markdown_text: str, heading_token: str) -> str:
    lines = markdown_text.splitlines()
    start = None
    token_compact = _compact(heading_token)
    for idx, line in enumerate(lines):
        if _MD_HEADING_RE.match(line) and token_compact in _compact(line):
            start = idx
            break
    if start is None:
        return ""
    out = []
    for idx in range(start, len(lines)):
        if idx > start and _MD_HEADING_RE.match(lines[idx]):
            break
        out.append(lines[idx])
    return "\n".join(out)


def _missing_numeric_anchors(source: str, target: str) -> List[str]:
    target_compact = _compact(target).replace(",", "").replace("，", "")
    anchors: List[str] = []
    for token in _NUMBER_RE.findall(source):
        clean = _compact(token).replace(",", "")
        if not clean or clean in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            continue
        # Plain integers are too noisy unless they are large enough to be an
        # example value, denominator, or score anchor.
        if re.fullmatch(r"\d+", clean) and len(clean) < 3:
            continue
        if clean not in anchors:
            anchors.append(clean)
    missing = []
    for token in anchors[:80]:
        variants = {token, token.replace("%", "％")}
        if token.endswith(".0"):
            variants.add(token[:-2])
        if not any(v in target_compact for v in variants):
            missing.append(token)
    return missing


def _missing_formula_left_sides(source: str, target: str) -> List[str]:
    target_compact = _compact(target)
    missing: List[str] = []
    for raw in source.splitlines():
        line = raw.strip()
        if len(line) < 8 or not _FORMULA_LINE_RE.search(line):
            continue
        if not re.search(r"[\u4e00-\u9fff]", line):
            continue
        left = re.split(r"[=＝≤≥]", line, maxsplit=1)[0]
        left = re.sub(r"^[|•·\-\s]+", "", left).strip(" ：:")
        left = re.sub(r"^(?:核算公式|计算公式|计分公式|计分规则\d*|指标定义)", "", left)
        if re.search(r"[。；;]", left):
            continue
        left_compact = _compact(left)
        if len(left_compact) < 4 or len(left_compact) > 36:
            continue
        if any(op in left_compact for op in ("<", ">", "≤", "≥", "〈", "〉")):
            continue
        if "、" in left_compact:
            continue
        if re.search(r"^(?:订单\d+|正向指标|负向指标)", left_compact):
            continue
        if re.search(r"[（(][^）)]*\d+\s*[,，]\s*\d+", left_compact):
            continue
        if "分钟" in left_compact and re.search(r"A[123]", left_compact):
            continue
        left_anchor = _formula_left_anchor(left_compact)
        target_anchor = _formula_left_anchor(target_compact)
        if (left_compact not in target_compact
                and (not left_anchor or left_anchor not in target_anchor)
                and left_compact not in missing):
            missing.append(left_compact)
    return missing[:20]


def _formula_left_anchor(text: str) -> str:
    text = _compact(text)
    text = re.sub(r"^[。．·•、:：|]+", "", text)
    text = text.replace(r"\text", "")
    text = re.sub(r"[\\{}_\[\]（）()]", "", text)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text)
    return text


def _is_detailed_section(token: str) -> bool:
    token = token.replace("．", ".")
    # Top-level Chinese sections such as "四、考核细则" are containers; their
    # numbers and formulas belong to child sections and would be noisy here.
    if re.fullmatch(r"[一二三四五六七八九十]{1,3}[、.]", token):
        return False
    return "." in token


def _local_section_evidence(text: str, limit: int = 2200) -> str:
    """Use the front of a source section for coverage.

    Raw PDF extraction sometimes misses a child heading and lets later sections
    bleed into the current one.  Checking the local heading neighborhood keeps
    coverage strict without blaming the wrong section for later content.
    """
    lines = text.splitlines()
    if len(lines) > 1:
        kept = [lines[0]]
        for line in lines[1:]:
            clean = line.strip()
            if _SOURCE_HEADING_RE.match(clean):
                break
            if re.match(r"^(?:\d+(?:[.．]\d+)+|[一二三四五六七八九十]{1,3}[、.．])\s*", clean):
                break
            kept.append(line)
        text = "\n".join(kept)
    if len(text) <= limit:
        return text
    return text[:limit]


def _check_markdown_readability(markdown_text: str) -> List[str]:
    issues: List[str] = []
    for idx, line in enumerate(markdown_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.count("<br>") >= 2:
            issues.append(f"排版审计: 第{idx}行存在密集换行标记")
        if len(stripped) > 420 and not stripped.startswith("$$"):
            issues.append(f"排版审计: 第{idx}行过长，疑似挤成一团")
        if "$$" in stripped and stripped != "$$" and not (
                stripped.startswith("$$") and stripped.endswith("$$")):
            issues.append(f"排版审计: 第{idx}行公式和正文粘连")
        if re.search(r"[。；;]\s*(?:举例|示例|其中|注[:：]|指标说明|数据来源|考核范围)[:：]",
                     stripped):
            issues.append(f"排版审计: 第{idx}行说明段落疑似未换行")
        if re.search(r"\|\s*[^|\n]{90,}\s*\|", stripped):
            issues.append(f"排版审计: 第{idx}行表格单元格过长，建议转为分组文本")
    return issues


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def builtin_check_cases() -> List[Tuple[str, Callable[[], None]]]:
    def case_section_missing_key_terms_fails() -> None:
        source = (
            "5.7 特殊场景体验融合考核的说明\n"
            "考核目标举例\n"
            "核算公式\n"
            "普通场景算分示例\n"
            "融合后体验得分=122.9818分\n"
            "6.1 下一节\n正文"
        )
        md = (
            "#### 5.7 特殊场景体验融合考核的说明\n"
            "普通场景算分示例\n"
            "融合后体验得分=122.9818分\n"
        )
        issues = _check_section_coverage(source, md)
        assert any("考核目标" in issue and "核算公式" in issue for issue in issues)

    def case_section_complete_passes() -> None:
        source = (
            "5.7 特殊场景体验融合考核的说明\n"
            "考核目标举例\n核算公式\n普通场景算分示例\n"
            "融合后体验得分=122.9818分\n"
        )
        md = (
            "#### 5.7 特殊场景体验融合考核的说明\n"
            "**考核目标举例**\n**核算公式**\n普通场景算分示例\n"
            "融合后体验得分=122.9818分\n"
        )
        assert _check_section_coverage(source, md) == []

    def case_adjacent_522_fusion_score_does_not_fail_521() -> None:
        source = (
            "5.2.1 场景说明\n"
            "普通场景\n特殊场景\n融合后体验得分\n"
            "5.3 压力场景加权考核\n正文\n"
        )
        md = (
            "#### 5.2.1 场景说明\n"
            "普通场景\n特殊场景\n"
            "#### 5.2.2 特殊场景体验融合考核计分规则\n"
            "融合后体验得分\n"
            "#### 5.3 压力场景加权考核\n正文\n"
        )
        assert _check_section_coverage(source, md) == []

    def case_readability_fails_long_line() -> None:
        issues = _check_markdown_readability("长句" * 230)
        assert issues and "过长" in issues[0]

    def case_readability_fails_formula_glued_to_text() -> None:
        issues = _check_markdown_readability(
            "注意：$$\\text{承托比}=\\frac{R}{W}$$举例：后续说明"
        )
        assert any("公式和正文粘连" in issue for issue in issues)

    def case_standard_table_rendering_required() -> None:
        blocks = [Block(kind="table", rows=[
            ["数据", "查询路径", "负责人"],
            ["KA品牌相关考核指标数据", "烽火台", "渠道经理"],
        ])]
        assert _check_table_block_rendering(blocks, "数据：KA品牌相关考核指标数据")
        md = "| 数据 | 查询路径 | 负责人 |\n|---|---|---|\n| KA品牌相关考核指标数据 | 烽火台 | 渠道经理 |\n"
        assert _check_table_block_rendering(blocks, md) == []

    def case_ka_required_formula_missing_fails() -> None:
        md = (
            "# 2026年6月KA品牌单月度考核制度\n"
            "#### 5.4.5 承托比（适用履约星巴克单的站点组）\n"
            "指标释义：\n"
            "#### 5.4.6 虚假点送达率\n"
            "指标说明\n数据来源\n电话客诉\n风控抓取\n虚假点击送达\n"
        )
        issues = _check_ka_required_content(md)
        assert any("5.4.5" in issue for issue in issues)
        assert any("5.4.6" in issue for issue in issues)

    def case_ka_required_formula_review_marker_fails() -> None:
        md = (
            "# 2026年6月KA品牌单月度考核制度\n"
            "### 1.站点组体验总得分计算逻辑\n"
            "站点组履约的KA品牌单体验总得分，站点组体验总得分，站点组F，站点组Kn\n"
            "**公式原文（需核对）**\n"
            "站点组体验总得分=站点组F分*F/(F+SUM(Kn*Qn))\n"
            "#### 5.4.5 承托比（适用履约星巴克单的站点组）\n"
            "承托比\n**公式原文（需核对）**\n承托比=R/W\n"
            "#### 5.4.6 虚假点送达率\n"
            "虚假点送达率\n**公式原文（需核对）**\n虚假点送达率=T/W\n"
            "指标释义：虚假点击送达。\n指标说明\n数据来源\n电话客诉\n风控抓取\n"
        )
        issues = _check_ka_required_content(md)
        assert any("站点组体验总得分" in issue for issue in issues)
        assert any("5.4.5" in issue for issue in issues)
        assert any("5.4.6" in issue for issue in issues)

    return [
        ("coverage.section_missing_key_terms_fails", case_section_missing_key_terms_fails),
        ("coverage.section_complete_passes", case_section_complete_passes),
        ("coverage.adjacent_522_fusion_score_does_not_fail_521", case_adjacent_522_fusion_score_does_not_fail_521),
        ("coverage.readability_fails_long_line", case_readability_fails_long_line),
        ("coverage.readability_fails_formula_glued_to_text", case_readability_fails_formula_glued_to_text),
        ("coverage.standard_table_rendering_required", case_standard_table_rendering_required),
        ("coverage.ka_required_formula_missing_fails", case_ka_required_formula_missing_fails),
        ("coverage.ka_required_formula_review_marker_fails", case_ka_required_formula_review_marker_fails),
    ]
