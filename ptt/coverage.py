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
    raw_blocks = result.meta.get("raw_blocks") or result.blocks
    raw_regions = result.meta.get("raw_regions") or []
    source_text = _source_text(pdf_path, raw_blocks)
    raw_region_text = _raw_regions_text(raw_regions)
    if raw_region_text:
        source_text = "\n".join(part for part in (source_text, raw_region_text) if part)
    issues: List[str] = []
    issues.extend(_check_review_markers(markdown_text))
    issues.extend(_check_toc_body_consistency(markdown_text))
    issues.extend(_check_heading_sequence_gaps(markdown_text))
    issues.extend(_check_markdown_readability(markdown_text))
    issues.extend(_check_table_block_rendering(result.blocks, markdown_text))
    issues.extend(_check_image_block_coverage(result.blocks, markdown_text))
    issues.extend(_check_raw_block_coverage(raw_blocks, markdown_text))
    issues.extend(_check_raw_region_coverage(raw_regions, markdown_text))
    issues.extend(_check_raw_policy_table_item_coverage(raw_blocks, raw_regions, markdown_text))
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


def _raw_regions_text(raw_regions: List[dict]) -> str:
    parts: List[str] = []
    for region in raw_regions:
        text = str(region.get("text") or "").strip()
        if text:
            parts.append(text)
        rows = region.get("rows") or []
        for row in rows:
            if isinstance(row, (list, tuple)):
                row_text = " ".join(str(c) for c in row if c)
                if row_text:
                    parts.append(row_text)
    return "\n".join(parts)


def _check_review_markers(markdown_text: str) -> List[str]:
    issues: List[str] = []
    markers = [
        "图示/低置信区域未能可靠文本化",
        "低置信区域（需核对）",
        "表格结构需核对",
        "公式原文（需核对）",
    ]
    for marker in markers:
        if marker in markdown_text:
            issues.append(f"覆盖审计: Markdown 仍包含需核对占位：{marker}")
    return issues


def _check_raw_region_coverage(raw_regions: List[dict],
                               markdown_text: str) -> List[str]:
    if not raw_regions:
        return []
    md_compact = _compact(markdown_text)
    issues: List[str] = []
    for idx, region in enumerate(raw_regions):
        text = str(region.get("text") or "")
        rows = region.get("rows") or []
        if rows:
            row_text = "\n".join(
                " ".join(str(c) for c in row if c)
                for row in rows if isinstance(row, (list, tuple))
            )
            text = "\n".join(part for part in (text, row_text) if part)
        compact = _compact(text)
        if not _is_high_risk_region(region, compact):
            continue
        anchors = _region_anchors(text)
        missing = _missing_seen_terms(compact, md_compact, anchors)
        if missing:
            page = int(region.get("page") or 0) + 1
            bbox = _format_bbox(region.get("bbox"))
            issues.append(
                f"覆盖审计: 第{page}页原始区域{idx}未完整进入 Markdown "
                f"bbox=[{bbox}] 缺少：" + "、".join(missing[:6])
            )
        if len(issues) >= 20:
            break
    return issues


def _is_high_risk_region(region: dict, compact: str) -> bool:
    if not compact or len(compact) < 4:
        return False
    kind = str(region.get("kind") or "")
    stage = str(region.get("stage") or "")
    flags = set(region.get("flags") or [])
    raw_text = str(region.get("text") or "").strip()
    m_heading = _SOURCE_HEADING_RE.match(raw_text)
    if m_heading and _is_probable_source_heading(m_heading.group(1), m_heading.group(2)):
        return True
    if stage == "ocr_line":
        return False
    if flags & {"auto_image", "table_fallback", "formula", "table_low_confidence"}:
        return True
    if kind in {"table", "image"} and len(compact) >= 12:
        return True
    high_terms = (
        "检核", "管控规则", "违规说明", "消毒", "装备", "早会",
        "考核规则", "考核目标", "核算公式", "计算公式", "指标定义",
        "数据来源", "查询路径", "申诉", "责任承担", "违约", "整改",
    )
    if any(term in compact for term in high_terms):
        return True
    return False


def _region_anchors(text: str) -> List[str]:
    anchors = [
        "线下检核", "线上视频检核", "站点线上早会", "骑手餐箱消毒",
        "骑手装备检核", "区域商管控规则", "物防设施管理", "责任承担",
        "查询路径", "申诉路径", "整改", "违约", "检核项目", "内容",
        "考核方式", "考核指标", "天气等级", "考核规则", "考核目标",
        "普通场景算分示例", "特殊场景算分示例", "融合后体验得分",
        "指标", "分子数值", "分母数值", "达标率", "满分目标", "权重",
        "数据来源", "普通场景", "特殊场景", "备注",
    ]
    stripped = text.strip()
    m = _SOURCE_HEADING_RE.match(stripped)
    if m and _is_probable_source_heading(m.group(1), m.group(2)):
        anchors.extend([m.group(1), m.group(2)])
    return _dedupe(anchors)


def _format_bbox(value) -> str:
    if not value:
        return ""
    try:
        return ",".join(str(round(float(v), 1)) for v in value)
    except Exception:
        return str(value)


def _check_raw_block_coverage(raw_blocks: List[Block],
                              markdown_text: str) -> List[str]:
    issues: List[str] = []
    if not raw_blocks:
        return issues
    issues.extend(_check_ka_57_raw_rule_coverage(raw_blocks, markdown_text))
    issues.extend(_check_raw_high_risk_block_coverage(raw_blocks, markdown_text))
    return issues


def _check_raw_policy_table_item_coverage(raw_blocks: List[Block],
                                          raw_regions: List[dict],
                                          markdown_text: str) -> List[str]:
    """Ensure Chinese policy-table item labels survive normalization/export.

    These tables often use merged category cells.  If the repair/export path only
    keeps the long content text, the Markdown can look plausible while losing the
    actual ``检核项目`` such as ``1.健康证`` or ``23.选址安全``.
    """
    items = _raw_policy_items(raw_blocks, raw_regions)
    if not items:
        return []
    md_compact = _compact(markdown_text)
    missing: List[str] = []
    for item_no, item_name, source_label in items:
        if _policy_item_exported(md_compact, item_no, item_name):
            continue
        if source_label not in missing:
            missing.append(source_label)
        if len(missing) >= 12:
            break
    if not missing:
        return []
    return ["覆盖审计: 原始政策表检核项目未进入 Markdown：" + "、".join(missing)]


def _raw_policy_items(raw_blocks: List[Block],
                      raw_regions: List[dict]) -> List[Tuple[str, str, str]]:
    seen = set()
    out: List[Tuple[str, str, str]] = []

    def add_from_text(text: str) -> None:
        if not _looks_like_policy_source(text):
            return
        for item_no, item_name in _extract_policy_item_labels(text):
            key = (item_no, item_name)
            if key in seen:
                continue
            seen.add(key)
            out.append((item_no, item_name, f"{item_no}.{item_name}"))

    for block in raw_blocks:
        if block.rows:
            text = "\n".join(" ".join(str(c) for c in row if c) for row in block.rows)
            if not _looks_like_policy_source(text):
                continue
            header_idx, item_idx = _policy_table_item_column(block.rows)
            if item_idx is not None:
                for row in block.rows[header_idx + 1:]:
                    cells = [str(c or "") for c in row]
                    pieces = []
                    if item_idx < len(cells):
                        pieces.append(cells[item_idx])
                    # Merged-cell damage can push the real item label into a
                    # neighboring content cell, so inspect the whole row too.
                    pieces.append(" ".join(cells))
                    for item_no, item_name in _extract_policy_item_labels(" ".join(pieces)):
                        key = (item_no, item_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append((item_no, item_name, f"{item_no}.{item_name}"))
            else:
                add_from_text(text)
        elif block.text:
            add_from_text(block.text)

    for region in raw_regions:
        text = str(region.get("text") or "")
        rows = region.get("rows") or []
        if rows:
            row_text = "\n".join(
                " ".join(str(c) for c in row if c)
                for row in rows if isinstance(row, (list, tuple))
            )
            text = "\n".join(part for part in (text, row_text) if part)
        add_from_text(text)

    return out


def _looks_like_policy_source(text: str) -> bool:
    compact = _compact(text)
    if len(compact) < 20:
        return False
    header_like = (
        "责任承担" in compact
        and ("检核项目" in compact or "核项目" in compact or "检查项目" in compact)
    )
    if header_like:
        return True
    if "责任承担" in compact and len(_extract_policy_item_labels(text)) >= 2:
        return True
    return False


def _policy_table_item_column(rows: List[List[str]]) -> Tuple[int, Optional[int]]:
    for ridx, row in enumerate(rows[:5]):
        cells = [_compact(str(c)) for c in row]
        joined = "".join(cells)
        if "责任承担" not in joined and "检核项目" not in joined:
            continue
        for idx, cell in enumerate(cells):
            if "检核项目" in cell or cell in {"项目", "核项目", "检查项目"}:
                return ridx, idx
    return 0, None


_POLICY_ITEM_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[.．]\s*"
    r"([A-Za-z\u4e00-\u9fff（）()/、]{1,28})"
)


def _extract_policy_item_labels(text: str) -> List[Tuple[str, str]]:
    labels: List[Tuple[str, str]] = []
    for match in _POLICY_ITEM_RE.finditer(text or ""):
        item_no = match.group(1)
        name = _clean_policy_item_name(match.group(2))
        if not _policy_item_name_is_meaningful(name):
            continue
        labels.append((item_no, name))
    return _dedupe_tuple(labels)


def _clean_policy_item_name(text: str) -> str:
    name = _compact(text)
    name = re.split(
        r"(?:责任承担|内容|一级分类|检核项目|分类|备注|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)",
        name,
        maxsplit=1,
    )[0]
    name = name.strip("：:;；,，。.")
    # Common merged-cell joins from station-standard tables.
    for prefix in ("安全台账", "选址安全", "视频监控", "标准站建设", "标准站",
                   "健康证", "手提式灭火器", "站点烟感", "站点用电",
                   "安全通道", "充电区选址", "充电区维保要求", "形象装备",
                   "内容交流"):
        if name.startswith(prefix):
            return prefix
    if len(name) > 12:
        name = name[:12]
    return name


def _policy_item_name_is_meaningful(name: str) -> bool:
    if len(name) < 2:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", name):
        return False
    noise = {
        "内容", "项目", "分类", "一级分类", "检核项目", "责任承担", "说明",
        "责任", "承担", "整改", "违约金", "标准", "建设", "配置", "安全",
    }
    if name in noise:
        return False
    if re.fullmatch(r"[A-Za-z]+", name) and len(name) <= 3:
        return False
    return True


def _policy_item_exported(md_compact: str, item_no: str, item_name: str) -> bool:
    for name in _policy_item_name_variants(item_name):
        variants = [
            f"{item_no}.{name}",
            f"{item_no}．{name}",
            f"{item_no}、{name}",
            f"{item_no}{name}",
        ]
        if any(v in md_compact for v in variants):
            return True
    return False


def _policy_item_name_variants(name: str) -> List[str]:
    compact = _compact(name)
    variants = [compact]
    if compact.startswith("安全台账"):
        variants.append("安全台账")
    if compact.startswith("选址安全"):
        variants.append("选址安全")
    if compact.startswith("站点感"):
        variants.append("站点")
    if compact.startswith("标准站"):
        variants.append("标准站")
    if len(compact) >= 4:
        variants.append(compact[:3])
    elif len(compact) == 3:
        variants.append(compact[:2])
    return _dedupe(variants)


def _dedupe_tuple(items: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out: List[Tuple[str, str]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _check_ka_57_raw_rule_coverage(raw_blocks: List[Block],
                                   markdown_text: str) -> List[str]:
    section_blocks = _raw_section_blocks(raw_blocks, "5.7", "5.8")
    if not section_blocks:
        return []
    raw_text = "\n".join(_block_text_for_coverage(b) for b in section_blocks)
    raw_compact = _compact(raw_text)
    if ("考核规则" not in raw_compact
            and not ("普通场景考核" in raw_compact and "特殊场景考核" in raw_compact)):
        return []
    md_57 = _extract_markdown_section(markdown_text, "5.7") or markdown_text
    md_compact = _compact(md_57)
    required_if_seen = [
        "普通场景", "特殊场景", "备注",
        "KA品牌负向反馈率", "负向反馈率", "虚假点送达率", "虚假点送达",
        "配送原因未完成率", "配送原因未完成", "复合准时率",
        "承托比", "复合超时时长", "KA品牌客诉率", "品牌客诉率",
        "距离≤3公里", "距离<3公里", "距离>3公里", "距离≥3公里",
        "天气等级为10", "天气等级20", "天气等级为20", "天气等级为30",
        "正常天气单", "恶劣天气单", "40天气免责", "240天气免责",
        "HD尾单", "专送兜底",
    ]
    missing = _missing_seen_terms(raw_compact, md_compact, required_if_seen)
    if missing:
        return ["覆盖审计: 5.7 原始考核规则明细未进入 Markdown：" + "、".join(missing[:8])]
    return []


def _check_raw_high_risk_block_coverage(raw_blocks: List[Block],
                                        markdown_text: str) -> List[str]:
    md_compact = _compact(markdown_text)
    issues: List[str] = []
    for idx, block in enumerate(raw_blocks):
        text = _block_text_for_coverage(block)
        compact = _compact(text)
        if not _is_high_risk_raw_block(block, compact):
            continue
        anchors = _raw_block_anchors(text)
        missing = _missing_seen_terms(compact, md_compact, anchors)
        if len(missing) >= 2:
            issues.append(
                f"覆盖审计: 原始区块{idx}重要内容疑似丢失 "
                + "、".join(missing[:6])
            )
    return issues[:12]


def _raw_section_blocks(blocks: List[Block], start_token: str,
                        end_token: str) -> List[Block]:
    start = None
    for idx, block in enumerate(blocks):
        compact = _compact(_block_text_for_coverage(block))
        if start_token.replace(".", "") in compact.replace(".", ""):
            start = idx
            break
    if start is None:
        return []
    end = len(blocks)
    end_norm = end_token.replace(".", "")
    for idx in range(start + 1, len(blocks)):
        compact = _compact(_block_text_for_coverage(blocks[idx]))
        if end_norm in compact.replace(".", ""):
            end = idx
            break
    return blocks[start:end]


def _block_text_for_coverage(block: Block) -> str:
    parts: List[str] = []
    if block.text:
        parts.append(block.text)
    if block.rows:
        parts.extend(" ".join(c for c in row if c) for row in block.rows)
    return "\n".join(parts)


def _is_high_risk_raw_block(block: Block, compact: str) -> bool:
    if not compact or len(compact) < 20:
        return False
    if block.kind == "heading":
        return False
    high_terms = (
        "考核规则", "考核目标", "核算公式", "计算公式", "指标定义",
        "数据来源", "普通场景算分示例", "特殊场景算分示例",
        "融合后体验得分", "降星规则", "查询路径",
    )
    if any(term in compact for term in high_terms):
        return True
    if block.rows and any(term in compact for term in ("指标", "规则", "备注", "得分", "权重")):
        return True
    return False


def _raw_block_anchors(text: str) -> List[str]:
    anchors = [
        "考核方式", "考核指标", "天气等级", "考核规则", "考核目标",
        "考核目标举例", "普通场景体验满分目标", "特殊场景体验满分目标",
        "核算公式", "普通场景算分示例", "特殊场景算分示例", "融合后体验得分",
        "指标", "分子数值", "分母数值", "达标率", "满分目标", "权重", "得分",
        "普通场景", "特殊场景", "备注", "数据来源", "查询路径",
        "KA品牌负向反馈率", "虚假点送达率", "虚假点送达",
        "配送原因未完成率", "复合准时率", "承托比", "复合超时时长",
        "KA品牌客诉率", "40天气免责", "HD尾单", "专送兜底",
    ]
    return anchors + _missing_numeric_anchors(text, "")


def _missing_seen_terms(source_compact: str, target_compact: str,
                        terms: List[str]) -> List[str]:
    out: List[str] = []
    for term in terms:
        compact = _compact(term)
        if not compact or compact not in source_compact:
            continue
        variants = _coverage_variants(compact)
        if not any(v in target_compact for v in variants):
            out.append(term)
    return _dedupe(out)


def _coverage_variants(compact: str) -> List[str]:
    variants = {compact}
    if compact.endswith("率"):
        variants.add(compact[:-1])
    variants.add(compact.replace("≤", "<=").replace("≥", ">="))
    variants.add(compact.replace("<=", "≤").replace(">=", "≥"))
    variants.add(compact.replace("天气等级为", "天气等级"))
    if compact.startswith("天气等级") and not compact.startswith("天气等级为"):
        variants.add(compact.replace("天气等级", "天气等级为", 1))
    variants.add(compact.replace("KA品牌", "KA"))
    variants.add(compact.replace("240天气", "40天气或240天气"))
    if "特殊场景" in compact:
        variants.add(compact.replace("特殊场景", "特场景"))
    if "特场景" in compact:
        variants.add(compact.replace("特场景", "特殊场景"))
    return [v for v in variants if v]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "").replace("％", "%").replace("＝", "=")


def _text_contains(haystack_compact: str, needle: str) -> bool:
    compact = _compact(needle)
    return any(v in haystack_compact for v in _coverage_variants(compact))


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
        if not _is_probable_source_heading(token, title):
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


def _is_probable_source_heading(token: str, title: str) -> bool:
    title_compact = _compact(title)
    if len(re.findall(r"[\u4e00-\u9fff]", title_compact)) < 2:
        return False
    if re.fullmatch(r"20\d{2}", token or "") and re.match(r"年\d{1,2}月", title_compact):
        return False
    if re.search(r"\d{4}年\d{1,2}月\d{1,2}日", token + title_compact):
        return False
    if re.fullmatch(r"\d+", token or ""):
        if title_compact.startswith(("分钟", "秒", "公里", "km", "KM")):
            return False
        if any(term in title_compact for term in ("订单占比", "口径准时率", "完成订单量")):
            return False
    return True


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


_TOC_LINE_RE = re.compile(
    r"^\s*((?:\d+(?:[.．]\d+)*)|[一二三四五六七八九十]{1,3}[、.．])\s*"
    r"(.+?)\s*[.。·•．…⋯\s]*\d{1,3}\s*$"
)


def _split_toc_and_body(markdown_text: str) -> Tuple[List[str], List[str]]:
    lines = markdown_text.splitlines()
    toc_start = None
    for idx, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+目录\s*$", line.strip()) or line.strip() == "目录":
            toc_start = idx
            break
    if toc_start is None:
        return [], lines
    toc_lines: List[str] = []
    seen_entry = False
    end = len(lines)
    for idx in range(toc_start + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            if seen_entry:
                toc_lines.append(lines[idx])
            continue
        if _TOC_LINE_RE.match(stripped):
            seen_entry = True
            toc_lines.append(lines[idx])
            continue
        if seen_entry:
            end = idx
            break
    return toc_lines, lines[end:]


def _extract_toc_entries(markdown_text: str) -> List[Tuple[str, str]]:
    toc_lines, _ = _split_toc_and_body(markdown_text)
    entries: List[Tuple[str, str]] = []
    for line in toc_lines:
        m = _TOC_LINE_RE.match(line.strip())
        if not m:
            continue
        token = m.group(1).replace("．", ".")
        title = re.sub(r"[.。·•．…⋯\s]+$", "", m.group(2)).strip()
        title = re.sub(r"\s+", "", title)
        if len(_compact(title)) < 2:
            continue
        entries.append((token, title))
    return entries


def _check_toc_body_consistency(markdown_text: str) -> List[str]:
    entries = _extract_toc_entries(markdown_text)
    if len(entries) < 4:
        return []
    _, body_lines = _split_toc_and_body(markdown_text)
    body_compact = _compact("\n".join(body_lines))
    issues: List[str] = []
    for token, title in entries:
        title_compact = _compact(title)
        if len(title_compact) <= 2 and re.fullmatch(r"目的|附则", title_compact):
            continue
        token_title = _compact(token + title)
        variants = {
            title_compact,
            token_title,
            _compact(token.replace("．", ".") + "." + title),
            _compact(token.replace("．", ".") + " " + title),
        }
        if not any(v and v in body_compact for v in variants):
            issues.append(f"覆盖审计: 目录章节正文疑似缺失 {token}{title}")
    return issues[:20]


def _check_heading_sequence_gaps(markdown_text: str) -> List[str]:
    _, body_lines = _split_toc_and_body(markdown_text)
    tokens: List[str] = []
    for raw in body_lines:
        line = raw.strip().lstrip("#").strip()
        m = re.match(r"^(\d+(?:[.．]\d+){1,3})\s*", line)
        if m:
            tokens.append(m.group(1).replace("．", "."))
    issues: List[str] = []
    by_parent: Dict[str, List[int]] = {}
    for token in tokens:
        parts = token.split(".")
        if len(parts) < 2:
            continue
        parent = ".".join(parts[:-1])
        try:
            child = int(parts[-1])
        except ValueError:
            continue
        by_parent.setdefault(parent, []).append(child)
    for parent, children in by_parent.items():
        present = set(children)
        ordered = []
        for child in children:
            if child not in ordered:
                ordered.append(child)
        for prev, cur in zip(ordered, ordered[1:]):
            if cur > prev + 1:
                missing = []
                for n in range(prev + 1, cur):
                    if n in present:
                        continue
                    token = f"{parent}.{n}"
                    if any(t.startswith(token + ".") for t in tokens):
                        continue
                    missing.append(token)
                if not missing:
                    continue
                issues.append(
                    "覆盖审计: 正文章节编号跳号，疑似缺失 "
                    + "、".join(missing[:5])
                )
    return issues[:12]


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
            or _image_rows_are_important(blk)
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
    row_text = ""
    if blk.rows:
        row_text = " ".join(" ".join(str(c) for c in row if c) for row in blk.rows)
    text = _compact(" ".join(part for part in (blk.text or "", row_text) if part))
    if not text:
        return False
    if re.search(r"[=＝≤≥×÷*/]|SUM|Σ", text, re.I):
        return True
    important_terms = (
        "公式", "计算口径", "指标", "数据来源", "考核范围", "规则",
        "得分", "费率", "金额", "单量", "完成率", "达标率",
    )
    return any(term in text for term in important_terms)


def _image_rows_are_important(blk: Block) -> bool:
    if not blk.rows:
        return False
    row_text = _compact(" ".join(" ".join(str(c) for c in row if c) for row in blk.rows))
    if len(row_text) < 12:
        return False
    if _auto_image_is_important(blk):
        return True
    non_empty_cells = [
        _compact(str(c))
        for row in blk.rows
        for c in row
        if _compact(str(c))
    ]
    if len(blk.rows) >= 2 and len(non_empty_cells) >= 4:
        return True
    return any(len(cell) >= 8 for cell in non_empty_cells)


def _image_block_exported(blk: Block, markdown_text: str) -> bool:
    md_compact = _compact(markdown_text)
    if _formula_fragment_replaced_by_latex(blk.text or "", markdown_text):
        return True
    if "formula" in blk.flags and blk.text:
        latex = ""
        try:
            from .export import _formula_to_latex
            latex = _formula_to_latex(blk.text)
        except Exception:
            latex = ""
        if latex and latex in markdown_text:
            return True
        if _text_anchors_exported(blk.text, md_compact):
            return True
    text = _compact(blk.text or "")
    if len(text) >= 8 and text[:40] in md_compact:
        return True
    if blk.text and _text_anchors_exported(blk.text, md_compact, min_hits=2):
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


def _formula_fragment_replaced_by_latex(text: str, markdown_text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    replacements = [
        (("配送原因未定成率", "PKA"), r"\text{配送原因未完成率}"),
        (("承托比", "RKA"), r"\text{承托比}"),
        (("虚假点送达率", "TKA"), r"\text{虚假点送达率}"),
        (("A1", "A2", "A3"), r"\text{复合超时时长}"),
        (("站点组KA品牌单体验得分", "K1"), r"\text{站点组KA品牌单体验得分}"),
    ]
    for needles, latex_label in replacements:
        if all(needle in compact for needle in needles) and latex_label in markdown_text:
            return True
    return False


def _text_anchors_exported(text: str, md_compact: str, min_hits: int = 1) -> bool:
    anchors = []
    for raw in (text or "").splitlines():
        token = _compact(raw)
        if len(token) >= 4:
            anchors.append(token[:28])
    if not anchors:
        return False
    hits = sum(1 for anchor in anchors if anchor in md_compact)
    return hits >= min_hits


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
    for match in _NUMBER_RE.finditer(source):
        token = match.group(0)
        clean = _compact(token).replace(",", "")
        if not clean or clean in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            continue
        if clean.endswith("天") and source[match.end():match.end() + 1] == "气":
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
        if re.fullmatch(r"\d+(?:[.,]\d+)?[sS]", token):
            variants.add(token[:-1])
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
        left = re.sub(r"^(?:考核项|项目|指标|内容|列\d+)[:：]?", "", left)
        left = re.sub(
            r"^.*?(?:结果计算公式|计算方式|核算公式|计算公式|计分公式)[:：]?",
            "",
            left,
        )
        if re.search(r"[。；;]", left):
            continue
        left_compact = _compact(left)
        if len(left_compact) < 4 or len(left_compact) > 36:
            continue
        if any(op in left_compact for op in ("<", ">", "≤", "≥", "〈", "〉", "＜", "＞")):
            continue
        if "、" in left_compact:
            continue
        if re.search(r"^(?:订单\d+|正向指标|负向指标)", left_compact):
            continue
        if re.search(r"[（(][^）)]*\d+\s*[,，]\s*\d+", left_compact):
            continue
        if "分钟" in left_compact and re.search(r"A[123]", left_compact):
            continue
        if "品牌体验得分" in left_compact:
            continue
        if "每日服务质量奖励费" in left_compact and "5星" in left_compact:
            continue
        if re.search(r"SUM|Σ", left_compact, re.I):
            continue
        if re.search(r"K[nN].*Q[nN]|Q[nN].*K[nN]", left_compact):
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
        if re.search(r"[。；;]\s*(?:举例\d*[:：]|举例如下[:：]|示例|其中|"
                     r"注[:：]|指标说明|数据来源|考核范围|"
                     r"特殊说明[:：]|同时满足如下条件|【[^】]{2,24}】[:：]|"
                     r"条件[一二三四五六七八九十][:：]|[1-9][.．]\s*[^0-9]|PS[:：]|备注[:：])",
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

    def case_date_range_not_source_heading() -> None:
        assert not _is_probable_source_heading("2026", "年06月01日-2026年06月30日")

    def case_formula_field_label_not_left_side() -> None:
        source = (
            "2.1 考核框架\n"
            "结果计算公式 异商混送最终得分=压力场景考核得分\n"
            "2.2 压力场景考核得分\n"
            "计算方式 月维度恶劣天气得分=Z日维度恶劣天气得分/恶劣天气天数\n"
        )
        md = (
            "#### 2.1 考核框架\n"
            "结果计算公式：异商混送最终得分=压力场景考核得分\n"
            "#### 2.2 压力场景考核得分\n"
            "计算方式：月维度恶劣天气得分=Z日维度恶劣天气得分/恶劣天气天数\n"
        )
        assert _check_section_coverage(source, md) == []

    def case_heading_gap_duplicate_child_does_not_fail() -> None:
        md = (
            "# 文档\n\n"
            "### 2. 具体要求\n\n"
            "#### 2.1 合法合规经营\n\n"
            "2.2.4 合作商需严格审核配送人员身份资质。\n\n"
            "### 2.2配送人员权益保障\n\n"
            "2.2.1 合作商需明确薪酬构成。\n\n"
            "2.2.2 合作商需准时足额发放。\n\n"
            "2.2.3 不得恶意克扣保险理赔金。\n\n"
            "2.2.4 合作商不得向配送人员销售商品获取不正当利益。\n\n"
            "2.2.5 合作商不得随意删除账号。\n"
        )
        assert _check_heading_sequence_gaps(md) == []

    def case_raw_ka_57_rule_detail_missing_fails() -> None:
        raw_blocks = [
            Block(kind="heading", text="5.7 特殊场景体验融合考核的说明", level=4),
            Block(kind="table", rows=[
                ["指标", "普通场景考核", "特殊场景考核", "备注"],
                ["复合超时时长", "距离≤3公里正常天气单", "距离>3公里正常天气单；恶劣天气单", "40天气免责；HD尾单&专送兜底"],
            ]),
            Block(kind="heading", text="5.8 下一节", level=4),
        ]
        result = DocResult(blocks=[], meta={"raw_blocks": raw_blocks})
        md = (
            "#### 5.7 特殊场景体验融合考核的说明\n"
            "**考核规则**\n"
            "以运单首次调度时的天气等级判定为依据。\n"
        )
        issues = audit_pdf_markdown_coverage("", result, md)
        assert any("5.7 原始考核规则明细" in issue for issue in issues)

    def case_raw_ka_57_rule_detail_complete_passes() -> None:
        raw_blocks = [
            Block(kind="heading", text="5.7 特殊场景体验融合考核的说明", level=4),
            Block(kind="table", rows=[
                ["指标", "普通场景考核", "特殊场景考核", "备注"],
                ["复合超时时长", "距离≤3公里正常天气单", "距离>3公里正常天气单；恶劣天气单", "40天气免责；HD尾单&专送兜底"],
            ]),
            Block(kind="heading", text="5.8 下一节", level=4),
        ]
        result = DocResult(blocks=[], meta={"raw_blocks": raw_blocks})
        md = (
            "#### 5.7 特殊场景体验融合考核的说明\n"
            "| 指标 | 普通场景考核 | 特殊场景考核 | 备注 |\n"
            "|---|---|---|---|\n"
            "| 复合超时时长 | 距离≤3公里正常天气单 | 距离>3公里正常天气单；恶劣天气单 | 40天气免责；HD尾单&专送兜底 |\n"
        )
        assert audit_pdf_markdown_coverage("", result, md) == []

    def case_raw_policy_table_item_missing_fails() -> None:
        raw_blocks = [
            Block(kind="table", rows=[
                ["一级分类", "检核项目", "内容", "责任承担"],
                ["健康证", "1.健康证", "健康证存在虚假。", "需承担违约责任"],
                ["标准站", "3.视频监控", "未购买视频监控且未报备。", "200元/项/次"],
            ]),
        ]
        result = DocResult(blocks=[], meta={"raw_blocks": raw_blocks})
        md = (
            "- 内容：健康证存在虚假。\n"
            "  责任承担：需承担违约责任\n"
            "- 一级分类：标准站\n"
            "  内容：未购买视频监控且未报备。\n"
            "  责任承担：200元/项/次\n"
        )
        issues = audit_pdf_markdown_coverage("", result, md)
        assert any("1.健康证" in issue and "3.视频监控" in issue for issue in issues)

    def case_raw_policy_table_item_grouped_text_passes() -> None:
        raw_blocks = [
            Block(kind="table", rows=[
                ["一级分类", "检核项目", "内容", "责任承担"],
                ["健康证", "1.健康证", "健康证存在虚假。", "需承担违约责任"],
                ["站", "建设", "标准 2.标准站 ①线上与线下地址准确。", "200元/项/次"],
                ["安全台账", "22.安全台账宿舍23.选址安全",
                 "此条规则只适用于广东省加盟站点。1.宿舍实际地址一致。",
                 "300元/项/次"],
            ]),
        ]
        result = DocResult(blocks=[], meta={"raw_blocks": raw_blocks})
        md = (
            "- 一级分类：健康证\n"
            "  检核项目：1.健康证\n"
            "  内容：健康证存在虚假。\n"
            "  责任承担：需承担违约责任\n"
            "- 一级分类：标准站\n"
            "  检核项目：2.标准站建设\n"
            "  内容：①线上与线下地址准确。\n"
            "  责任承担：200元/项/次\n"
            "- 一级分类：安全台账\n"
            "  检核项目：22.安全台账\n"
            "  内容：此条规则只适用于广东省加盟站点。\n"
            "  责任承担：300元/项/次\n"
            "- 一级分类：站外宿舍\n"
            "  检核项目：23.选址安全\n"
            "  内容：1.宿舍实际地址一致。\n"
            "  责任承担：300元/项/次\n"
        )
        assert audit_pdf_markdown_coverage("", result, md) == []

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

    def case_tiny_image_row_fragment_not_blocking() -> None:
        blocks = [Block(kind="image", page=0, rows=[["知用户", "户", "效通知"]])]
        assert _check_image_block_coverage(blocks, "") == []

    def case_multiline_image_text_covered_by_anchors() -> None:
        blocks = [Block(kind="image", page=0, flags=["auto_image"], text=(
            "禁止入内\n"
            "高校范围内封\n"
            "违规取消\n"
            "火车站无法进入"
        ))]
        md = "禁止入内\n高校范围内封闭，违规取消。\n"
        assert _check_image_block_coverage(blocks, md) == []

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
        ("coverage.date_range_not_source_heading", case_date_range_not_source_heading),
        ("coverage.formula_field_label_not_left_side", case_formula_field_label_not_left_side),
        ("coverage.heading_gap_duplicate_child_does_not_fail", case_heading_gap_duplicate_child_does_not_fail),
        ("coverage.raw_ka_57_rule_detail_missing_fails", case_raw_ka_57_rule_detail_missing_fails),
        ("coverage.raw_ka_57_rule_detail_complete_passes", case_raw_ka_57_rule_detail_complete_passes),
        ("coverage.raw_policy_table_item_missing_fails", case_raw_policy_table_item_missing_fails),
        ("coverage.raw_policy_table_item_grouped_text_passes", case_raw_policy_table_item_grouped_text_passes),
        ("coverage.readability_fails_long_line", case_readability_fails_long_line),
        ("coverage.readability_fails_formula_glued_to_text", case_readability_fails_formula_glued_to_text),
        ("coverage.standard_table_rendering_required", case_standard_table_rendering_required),
        ("coverage.tiny_image_row_fragment_not_blocking", case_tiny_image_row_fragment_not_blocking),
        ("coverage.multiline_image_text_covered_by_anchors", case_multiline_image_text_covered_by_anchors),
        ("coverage.ka_required_formula_missing_fails", case_ka_required_formula_missing_fails),
        ("coverage.ka_required_formula_review_marker_fails", case_ka_required_formula_review_marker_fails),
    ]
