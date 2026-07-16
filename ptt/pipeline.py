"""主管线：逐页判型 -> 提取 -> 质检 -> 导出。"""
import os
import re
import shutil
from copy import deepcopy
from typing import Callable, List, Optional

import fitz

from . import assemble, coverage, qa, text_extract
from .export import export_docx, export_markdown
from .models import Block, DocResult
from .normalize import (
    _is_reliable_repaired_structured_table,
    finalize_visual_order_repairs,
    normalize_blocks,
)
from .vision_ocr import (EmbeddedImageProvider, RenderProvider, StripProvider,
                         ocr_provider, ocr_strip)

ProgressFn = Callable[[str, float], None]  # (消息, 0~1)


def _append_raw_block_regions(result: DocResult, blocks: List[Block],
                              stage: str) -> None:
    regions = result.meta.setdefault("raw_regions", [])
    for idx, blk in enumerate(blocks):
        parts = []
        if blk.text:
            parts.append(blk.text)
        if blk.rows:
            parts.extend(" ".join(c for c in row if c) for row in blk.rows)
        regions.append({
            "stage": stage,
            "index": idx,
            "kind": blk.kind,
            "text": "\n".join(parts),
            "rows": deepcopy(blk.rows) if blk.rows else None,
            "bbox": tuple(blk.bbox),
            "page": blk.page,
            "confidence": blk.confidence,
            "flags": list(blk.flags),
        })


def _append_raw_line_regions(result: DocResult, lines, page: int,
                             stage: str = "ocr_line") -> None:
    regions = result.meta.setdefault("raw_regions", [])
    for idx, line in enumerate(lines):
        regions.append({
            "stage": stage,
            "index": idx,
            "kind": "line",
            "text": line.text,
            "rows": None,
            "bbox": (line.x0, line.y0, line.x1, line.y1),
            "page": page,
            "confidence": line.conf,
            "flags": [],
        })


def _rescan_complex_table_regions(result: DocResult, provider: StripProvider,
                                  blocks: List[Block]) -> int:
    """Rescan risky table regions with Apple Vision at the table scale.

    A full-page pass can split a narrow percentage or formula cell across
    neighbouring columns.  The second pass keeps the same Vision OCR engine,
    but gives the table its own crop and a stable target width.  It is recorded
    only in the in-memory raw ledger and is never emitted as a debug artifact.
    """
    from .export import _table_is_complex

    count = 0
    for block in blocks:
        flags = set(block.flags or [])
        if block.kind not in {"image", "table"}:
            continue
        if not flags.intersection({
                "cell_ocr_table", "table_fallback", "table_low_confidence"}):
            continue
        risky = block.kind == "image" or not block.rows or len(block.rows) < 2
        risky = risky or _table_is_complex(block.rows or [])
        if not risky:
            continue
        y0 = max(0, int(block.bbox[1]) - 18)
        y1 = min(provider.height, int(block.bbox[3]) + 18)
        if y1 - y0 < 40:
            continue
        try:
            strip = provider.get_strip(y0, y1)
            lines = ocr_strip(strip)
        except Exception as exc:
            result.warnings.append(
                f"复杂表格二次OCR失败（第{block.page + 1}页，y={y0}-{y1}）：{exc}")
            continue
        for line in lines:
            line.y0 += y0
            line.y1 += y0
        if lines:
            _append_raw_line_regions(result, lines, block.page,
                                     stage="ocr_table_rescan")
            count += 1
    return count


def _table_line_candidates(raw_regions: List[dict], page: int,
                           bbox) -> List[tuple]:
    """Return de-duplicated table lines, preferring the table rescan."""
    x0, y0, x1, y1 = bbox
    candidates = []
    for region in raw_regions:
        if (region.get("stage") not in {"ocr_line", "ocr_table_rescan"}
                or int(region.get("page") or 0) != page):
            continue
        rx0, ry0, rx1, ry1 = region.get("bbox") or (0, 0, 0, 0)
        cy = (ry0 + ry1) / 2
        cx = (rx0 + rx1) / 2
        text = str(region.get("text") or "").strip()
        if not text or not (y0 - 8 <= cy <= y1 + 8
                            and x0 - 12 <= cx <= x1 + 12):
            continue
        candidates.append({
            "y": ry0, "x": rx0, "cy": cy, "cx": cx, "text": text,
            "stage": region.get("stage"),
            "confidence": float(region.get("confidence") or 0),
        })
    # A same-position line from the dedicated crop wins over the full-page
    # line.  Nearby columns remain separate because their x centers differ.
    selected = []
    for candidate in sorted(candidates, key=lambda item: (
            0 if item["stage"] == "ocr_table_rescan" else 1,
            item["y"], item["x"])):
        duplicate = None
        for index, current in enumerate(selected):
            if (abs(candidate["cy"] - current["cy"]) <= 18
                    and abs(candidate["cx"] - current["cx"]) <= 110):
                duplicate = index
                break
        if duplicate is None:
            selected.append(candidate)
        elif (candidate["stage"] == "ocr_table_rescan"
              and selected[duplicate]["stage"] != "ocr_table_rescan"):
            selected[duplicate] = candidate
    return sorted(selected, key=lambda item: (item["y"], item["x"]))


def _restore_complex_table_lines(blocks: List[Block], raw_regions: List[dict],
                                 raw_blocks: Optional[List[Block]] = None) -> int:
    """Use ordered OCR lines when a low-confidence table grid is destructive.

    Apple Vision usually sees every line in a long policy table, but a guessed
    cell grid can splice neighbouring columns together and silently lose the
    beginning or end of a rule.  For genuinely complex fallback tables, the
    ordered line stream is safer and remains readable in Markdown.  Simple and
    native text-PDF tables keep their structured rows.
    """
    from .export import _table_is_complex

    restored = 0
    furniture_re = re.compile(
        r"保密(?:资料|原则)|未经书面允许|第三方提供|美团配送|准时好用|MTPS[-A-Z0-9]*/?",
        re.I,
    )
    page_no_re = re.compile(r"^\s*\d{1,3}\s*$")
    raw_blocks = raw_blocks or []
    for block in blocks:
        flags = set(block.flags or [])
        if block.kind not in {"image", "table"}:
            continue
        source_table = block
        if not block.rows:
            # image_fallback may have already discarded the rows.  Match the
            # final placeholder back to the pre-fallback raw block by page and
            # visual center so the content still has a recoverable destination.
            candidates = [raw for raw in raw_blocks
                          if raw.page == block.page and raw.rows
                          and _bbox_near(block.bbox, raw.bbox)]
            if candidates:
                source_table = min(
                    candidates,
                    key=lambda raw: _bbox_distance(block.bbox, raw.bbox),
                )
            else:
                continue
        source_flags = set(source_table.flags or []) | flags
        if not source_flags.intersection({"cell_ocr_table", "table_fallback", "table_low_confidence"}):
            continue
        risky_table = (_table_is_complex(source_table.rows or [])
                       or block.kind == "image"
                       or len(source_table.rows or []) < 2)
        if not risky_table:
            continue

        # A low-confidence score is not proof that the grid is unusable.  In
        # long policy tables the score is commonly caused by long, multi-line
        # cells.  When the recovered rows still have a stable policy schema,
        # keep that structure and let the Markdown exporter render it as
        # grouped fields.  Flattening it into OCR line order is what caused
        # labels such as ``14.手提式灭火器`` to separate from their content.
        if _structured_policy_rows_are_safe(source_table.rows or []):
            block.kind = "table"
            block.text = ""
            block.rows = deepcopy(source_table.rows or [])
            block.flags = [flag for flag in block.flags
                           if flag not in {"cell_ocr_table", "table_fallback",
                                            "table_low_confidence",
                                            "merged_table_fallback",
                                            "fragment_table_fallback",
                                            "table_trailing_continuation"}]
            block.flags.extend(["structured_policy_table",
                                "table_repaired_verified"])
            block.confidence = max(block.confidence, 0.7)
            restored += 1
            continue

        score_text = _structured_score_example_text(source_table.rows or [])
        if score_text:
            block.kind = "para"
            block.text = score_text
            block.rows = None
            block.flags = [flag for flag in block.flags
                           if flag not in {"cell_ocr_table", "table_fallback",
                                            "table_low_confidence",
                                            "merged_table_fallback",
                                            "fragment_table_fallback",
                                            "table_trailing_continuation",
                                            "raw_table_text_fallback"}]
            block.flags.extend(["structured_score_table",
                                "table_repaired_verified"])
            block.confidence = max(block.confidence, 0.75)
            restored += 1
            continue

        x0, y0, x1, y1 = block.bbox
        selected_records = _table_line_candidates(raw_regions, block.page,
                                                  block.bbox)
        selected_records = [item for item in selected_records
                            if not furniture_re.search(item["text"])
                            and not page_no_re.fullmatch(item["text"])]
        if len(selected_records) < 2:
            continue
        raw_text = "\n".join(item["text"] for item in selected_records)
        raw_text = _restore_table_semantic_labels(
            raw_text, source_table.rows or [])
        structured_text = "\n".join(
            " ".join(str(cell) for cell in row if cell)
            for row in (source_table.rows or [])
        )
        # Require a meaningful source line stream.  The raw stream may be
        # shorter than the flattened cells for wide tables, so compare both
        # character volume and the number of visual lines.
        if len(re.sub(r"\s+", "", raw_text)) < 8:
            continue
        if (len(raw_text) < len(structured_text) * 0.22
                and len(selected_records) < 6):
            continue
        used_rescan = any(item["stage"] == "ocr_table_rescan"
                          for item in selected_records)
        supplement = _structured_table_supplement(source_table.rows or [])
        # The dedicated crop is the authoritative line stream for this table;
        # do not append stale numbers from the first, column-confused grid.
        if supplement and not used_rescan:
            raw_text += "\n\n表格关键字段补充：\n" + supplement
        block.kind = "para"
        block.text = raw_text
        block.rows = None
        block.flags = [flag for flag in block.flags
                       if flag not in {"cell_ocr_table", "table_fallback",
                                       "table_low_confidence", "table_repaired_verified"}]
        block.flags.append("raw_table_text_fallback")
        block.confidence = max(block.confidence, 0.75)
        restored += 1

    # A fallback image can be removed entirely by a later noise pass.  The
    # raw ledger still knows its visual bounds, so add a destination for any
    # complex table region that has no surviving block nearby.
    additions: List[Block] = []
    for raw in raw_blocks:
        if not raw.rows or raw.kind not in {"image", "table"}:
            continue
        raw_flags = set(raw.flags or [])
        if not raw_flags.intersection({"cell_ocr_table", "table_fallback", "table_low_confidence"}):
            continue
        if not _table_is_complex(raw.rows):
            continue
        if any(existing.page == raw.page
               and _bbox_near(existing.bbox, raw.bbox)
               for existing in blocks):
            continue
        x0, y0, x1, y1 = raw.bbox
        selected_records = _table_line_candidates(raw_regions, raw.page,
                                                  raw.bbox)
        selected_records = [item for item in selected_records
                            if not furniture_re.search(item["text"])
                            and not page_no_re.fullmatch(item["text"])]
        if len(selected_records) < 2:
            continue
        text = "\n".join(item["text"] for item in selected_records)
        text = _restore_table_semantic_labels(text, raw.rows)
        if len(re.sub(r"\s+", "", text)) < 8:
            continue
        used_rescan = any(item["stage"] == "ocr_table_rescan"
                          for item in selected_records)
        supplement = _structured_table_supplement(raw.rows)
        if supplement and not used_rescan:
            text += "\n\n表格关键字段补充：\n" + supplement
        additions.append(Block(
            kind="para",
            text=text,
            bbox=tuple(raw.bbox),
            page=raw.page,
            confidence=max(raw.confidence, 0.75),
            flags=["raw_table_text_fallback", "coverage_rescue"],
        ))
    if additions:
        blocks.extend(additions)
        restored += len(additions)
    return restored


def _structured_policy_rows_are_safe(rows: List[List[str]]) -> bool:
    """Return whether a fallback table still has a trustworthy row schema.

    This is intentionally stricter than merely checking for a header.  A
    policy table is eligible only when its item/content/responsibility columns
    are explicit and most body rows retain at least two populated cells.  The
    exporter can then format the rows without guessing a new column mapping.
    """
    if not rows or len(rows) < 2:
        return False
    ncol = max(len(row) for row in rows)
    # Row-spanned cells are represented by shorter rows (for example the
    # trailing ``特殊说明`` row has only two populated columns).  Padding is
    # safe here; rejecting those rows would force the whole table back into a
    # destructive line stream.
    if ncol < 3 or ncol > 6 or any(len(row) > ncol for row in rows):
        return False
    normalized_rows = [list(row) + [""] * (ncol - len(row)) for row in rows]
    compact_header = {re.sub(r"\s+", "", str(cell or ""))
                      for cell in normalized_rows[0]}
    has_item = bool(compact_header & {"检核项目", "项目", "类型", "考核项目"})
    has_content = bool(compact_header & {"内容", "说明", "释义"})
    has_responsibility = bool(compact_header & {
        "责任承担", "承担责任", "违约责任", "整改结果"
    })
    if not (has_item and has_content and has_responsibility):
        return False
    body = rows[1:]
    populated = sum(
        1 for row in normalized_rows[1:]
        if sum(bool(re.sub(r"\s+", "", str(cell or ""))) for cell in row) >= 2
    )
    if populated / max(1, len(body)) < 0.72:
        return False
    joined = "".join(re.sub(r"\s+", "", str(cell or ""))
                     for row in normalized_rows for cell in row)
    if not joined or any(token in joined for token in ("【保密资料", "MTPS-ZS-")):
        return False
    # The item column must contain real numbered or named entries; otherwise
    # this is likely a header fragment rather than a recoverable table.
    numbered = len(re.findall(r"(?<!\d)\d{1,2}\s*[.．、]\s*[\u4e00-\u9fffA-Za-z]", joined))
    named = sum(
        1 for row in normalized_rows[1:]
        if any(re.search(r"[\u4e00-\u9fffA-Za-z]{2,}", str(cell or ""))
               for cell in row[:min(2, len(row))])
    )
    return numbered >= 2 or named >= max(2, int(len(body) * 0.45))


def _structured_table_supplement(rows: List[List[str]]) -> str:
    """Keep row labels adjacent to their important numeric evidence."""
    lines: List[str] = []
    for row in rows:
        cells = [str(cell or '').strip() for cell in row]
        if not cells:
            continue
        label = cells[0]
        if not label:
            continue
        values = " ".join(cells[1:])
        numbers = re.findall(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:%|％|pp|分钟|小时|天|单)?",
            values,
            flags=re.I,
        )
        if numbers:
            lines.append(f"{label}：数值 {'；'.join(dict.fromkeys(numbers))}")
        else:
            lines.append(label)
    return "\n".join(lines)


def _restore_table_semantic_labels(text: str, rows: List[List[str]]) -> str:
    """Restore a table label that Vision often reduces to ``定义``.

    A label is part of the source meaning, not decorative table furniture.
    When the source grid has an explicit definition row and the rescan only
    returns the short cell label, make that context visible before export.
    """
    compact = re.sub(r"\s+", "", text or "")
    if "指标定义" in compact:
        return text
    labels = {
        re.sub(r"\s+", "", str(row[0] or ""))
        for row in rows
        if row and row[0]
    }
    if not (labels & {"定义", "指标定义", "指标定义1", "指标定义2", "指标定义3",
                      "指标定义4", "指标定义5"}):
        return text
    return "指标定义：\n" + text


def _bbox_near(left, right) -> bool:
    """Match OCR fallback blocks without requiring identical grid bounds."""
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    return abs(lcx - rcx) <= 900 and abs(lcy - rcy) <= 1800


def _bbox_distance(left, right) -> float:
    lcx, lcy = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rcx, rcy = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    return abs(lcx - rcx) + abs(lcy - rcy)


def _page_provider(doc: fitz.Document, pno: int) -> StripProvider:
    """图片页取图：整页单图则原图解码（保真且兼容超长JPEG），否则渲染。"""
    page = doc[pno]
    imgs = page.get_images()
    if len(imgs) == 1:
        try:
            info = page.get_image_info()
            if info:
                r = fitz.Rect(info[0]["bbox"])
                cover = (r.width * r.height) / (page.rect.width * page.rect.height)
                if cover > 0.9:
                    raw = doc.extract_image(imgs[0][0])["image"]
                    return EmbeddedImageProvider(raw)
        except Exception:
            pass
    return RenderProvider(page)


def convert(pdf_path: str, out_dir: str, formats=("md",),
            progress: Optional[ProgressFn] = None) -> dict:
    """转换单个 PDF。返回结果摘要 dict（供 CLI/GUI/Agent 使用）。"""
    def report(msg, frac):
        if progress:
            progress(msg, frac)

    name = os.path.splitext(os.path.basename(pdf_path))[0]
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    os.makedirs(out_dir, exist_ok=True)
    assets_dir = os.path.join(out_dir, f"{safe}_assets")

    doc = fitz.open(pdf_path)
    result = DocResult(meta={"source": pdf_path, "title": name,
                             "pages": len(doc)})
    repeated = text_extract.collect_repeated(doc)
    n = len(doc)
    fig_count = 0

    for pno in range(n):
        page = doc[pno]
        text_chars = len(page.get_text().strip())
        base = pno / n
        if text_chars >= 50:
            report(f"第 {pno+1}/{n} 页：文本提取", base)
            blocks, notes = text_extract.extract_text_page(
                doc, pno, repeated, assets_dir, safe,
                preserve_images=("docx" in formats),
                detect_formula_images=("md" in formats))
            result.meta.setdefault("raw_blocks", []).extend(deepcopy(blocks))
            _append_raw_block_regions(result, blocks, "text_extract")
            result.blocks.extend(blocks)
            result.warnings.extend(notes)
        else:
            report(f"第 {pno+1}/{n} 页：OCR 识别", base)
            provider = _page_provider(doc, pno)
            lines = ocr_provider(
                provider,
                progress=lambda i, m: report(
                    f"第 {pno+1}/{n} 页：OCR {i}/{m} 片", base + (i / m) * 0.7 / n))
            _append_raw_line_regions(result, lines, pno)
            blocks, notes = assemble.assemble_blocks(provider, lines, page=pno)
            result.warnings.extend(notes)
            _append_raw_block_regions(result, blocks, "assembled_pre_recheck")
            report(f"第 {pno+1}/{n} 页：复核低置信内容", base + 0.75 / n)
            notes2 = qa.recheck_ocr_blocks(blocks, provider)
            result.warnings.extend(notes2)
            raw_page_blocks = deepcopy(blocks)
            blocks, notes3 = qa.image_fallback(blocks, provider)
            result.warnings.extend(notes3)
            rescanned = _rescan_complex_table_regions(result, provider, blocks)
            if rescanned:
                result.warnings.append(
                    f"复杂表格按区域二次OCR（{rescanned}个区域）")
            # 图片区域落盘
            os.makedirs(assets_dir, exist_ok=True)
            for blk in blocks:
                if blk.kind == "image" and not blk.image_path:
                    fig_count += 1
                    x0, y0, x1, y1 = (int(blk.bbox[0]), int(blk.bbox[1]),
                                      int(blk.bbox[2]), int(blk.bbox[3]))
                    img = provider.get_strip(max(0, y0 - 4), y1 + 4)
                    x0 = max(0, min(img.width, x0))
                    x1 = max(x0 + 1, min(img.width, x1))
                    if x1 - x0 < img.width:
                        img = img.crop((x0, 0, x1, img.height))
                    path = os.path.join(assets_dir, f"{safe}_fig{fig_count}.png")
                    img.save(path)
                    blk.image_path = path
            result.blocks.extend(blocks)
            result.meta.setdefault("raw_blocks", []).extend(raw_page_blocks)

    # 全文词频投票纠错（形近字：封项值→封顶值、惺星→1星 等）
    restored_tables = _restore_complex_table_lines(
        result.blocks, result.meta.get("raw_regions") or [],
        result.meta.get("raw_blocks") or [])
    if restored_tables:
        result.warnings.append(f"复杂表格按原始OCR行降级保留（{restored_tables}）")
    vote_notes = qa.doc_vote_fix(result.blocks)
    if vote_notes:
        result.warnings.extend(vote_notes[:10])
    if "raw_blocks" not in result.meta:
        result.meta["raw_blocks"] = deepcopy(result.blocks)
    norm_notes = normalize_blocks(result.blocks)
    if norm_notes:
        result.warnings.extend(norm_notes[:20])

    # 封面标题常被换行书写成多个一级标题 -> 合并为一个
    blks = result.blocks
    while (len(blks) >= 2 and blks[0].kind == "heading" and blks[0].level == 1
           and blks[1].kind == "heading" and blks[1].level == 1
           and blks[0].page == blks[1].page
           and blks[1].bbox[1] - blks[0].bbox[3]
               < (blks[0].bbox[3] - blks[0].bbox[1]) * 1.2):
        blks[0].text += blks[1].text
        blks[0].bbox = (blks[0].bbox[0], blks[0].bbox[1],
                        blks[1].bbox[2], blks[1].bbox[3])
        del blks[1]

    result.blocks.sort(key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
    final_order_notes = finalize_visual_order_repairs(result.blocks)
    if final_order_notes:
        result.warnings.extend(final_order_notes[:20])

    report("质检复读", 0.92)
    issues, n_flag = qa.qa_scan(result.blocks)
    # 顺序自检：输出必须严格按原文档视觉顺序（页码、再纵坐标）
    prev_key = None
    for b in result.blocks:
        key = (b.page, b.bbox[1])
        if prev_key is not None and key < prev_key:
            issues.append(f"顺序异常: 页{b.page+1} y={b.bbox[1]:.0f} 出现在更早内容之前")
        prev_key = key

    outputs = []
    if "md" in formats:
        p = os.path.join(out_dir, f"{safe}.md")
        export_markdown(result, p)
        outputs.append(p)
    if "docx" in formats:
        p = os.path.join(out_dir, f"{safe}.docx")
        export_docx(result, p)
        outputs.append(p)

    md_paths = [p for p in outputs if p.endswith(".md")]
    if md_paths:
        try:
            with open(md_paths[0], "r", encoding="utf-8") as f:
                md_text = f.read()
        except OSError:
            md_text = ""
    else:
        md_text = coverage.markdown_from_result(result)
    coverage_issues = coverage.audit_pdf_markdown_coverage(
        pdf_path, result, md_text)
    if coverage_issues:
        issues.extend(coverage_issues)
    if md_paths:
        try:
            from .audit import scan_markdown
            md_audit = scan_markdown(md_paths[0])
            for issue in md_audit.get("issues", []):
                issue_type = issue.get("type", "Markdown审计")
                for match in issue.get("matches", [])[:5]:
                    line = match.get("line", "?")
                    text = match.get("text", "")
                    issues.append(f"Markdown审计: {issue_type} 第{line}行 {text}")
        except Exception as exc:
            issues.append(f"Markdown审计无法完成: {exc}")
    issues.extend(_blocking_issues_from_warnings(
        result.warnings,
        has_markdown=bool(md_paths),
        blocks=result.blocks,
        markdown_text=md_text,
    ))
    blocking_issues: List[str] = []
    qa_warnings: List[str] = []
    for issue in issues:
        if "表格疑似列错位" in issue:
            # A complex two-column policy table can legitimately score high
            # because its cells contain long multi-line definitions.  If the
            # repaired rows are complete and content-aware, the original QA
            # warning is stale rather than evidence of a missing column.
            match = re.search(r"块(\d+)", issue)
            if match:
                index = int(match.group(1))
                if (0 <= index < len(result.blocks)
                        and _is_reliable_repaired_structured_table(
                            result.blocks[index])):
                    continue
            qa_warnings.append(
                issue.replace("建议人工复核", "已进入覆盖审计复核")
            )
            blocking_issues.append(f"表格结构需复核: {issue}")
            continue
        blocking_issues.append(issue)

    # Markdown 只输出单文件；裁切图仅作 OCR/复核中间产物。
    # docx 图片在保存时已内嵌，也可以安全清理。
    if os.path.isdir(assets_dir):
        shutil.rmtree(assets_dir, ignore_errors=True)
    report("完成", 1.0)
    doc.close()

    return {
        "source": pdf_path,
        "outputs": outputs,
        "pages": n,
        "blocks": len(result.blocks),
        "warnings": result.warnings,
        "qa_issues": blocking_issues,
        "qa_warnings": qa_warnings,
        "blocking_qa": blocking_issues,
        "quality_ok": not blocking_issues,
        "flagged_blocks": n_flag,
    }


def _blocking_issues_from_warnings(warnings: List[str],
                                   has_markdown: bool = True,
                                   blocks: Optional[List[Block]] = None,
                                   markdown_text: str = "") -> List[str]:
    issues: List[str] = []
    if not has_markdown:
        return issues
    for warning in warnings:
        if "低置信区域已转为截图" in warning:
            if _fallback_warning_resolved(
                    warning, blocks or [], markdown_text):
                continue
            issues.append(
                "单文件Markdown存在未输出的低置信截图区域: "
                + warning
            )
    return issues


def _fallback_warning_resolved(warning: str, blocks: List[Block],
                               markdown_text: str) -> bool:
    """A stale screenshot warning is harmless only if final Markdown covers it.

    Low-confidence warnings are emitted before normalization and formula/table
    reconstruction.  The warning must still block when the final region has no
    destination, but should not keep a correctly reconstructed LaTeX formula or
    verified table in a permanent failure state.
    """
    if not markdown_text:
        return False
    match = re.search(r'y=(-?\d+)', warning)
    y = float(match.group(1)) if match else None
    candidates: List[Block] = []
    if y is not None:
        for blk in blocks:
            if blk.bbox[1] - 700 <= y <= blk.bbox[3] + 700:
                candidates.append(blk)

    for blk in candidates:
        if "discarded_toc_fragment" in (blk.flags or []):
            return True
        if ("table_low_confidence" in (blk.flags or [])
                or "possible_truncation" in (blk.flags or [])):
            continue
        if coverage._image_block_exported(blk, markdown_text):
            return True

    # A duplicate fallback fragment may have been removed after its complete
    # paragraph survived elsewhere.  In that case, test the warning snippet
    # itself against the final Markdown anchors.
    snippet = warning.split("): ", 1)[-1]
    snippet = "\n".join(part.strip() for part in snippet.split(" / ")
                          if part.strip())
    if snippet:
        probe = Block(kind="image", text=snippet, flags=["auto_image"])
        if coverage._image_block_exported(probe, markdown_text):
            return True
    return False


def _structured_score_example_text(rows: List[List[str]]) -> str:
    """Rebuild a multi-level score-example table without flattening columns.

    Vision can preserve every cell in a wide table while returning the cells
    in a three-column reading stream.  The stream is still recoverable when
    it contains the score headers, at least three metric rows, and the final
    fusion summary.  This helper keeps the source values but renders them as
    explicit field/value lines instead of guessing a visual Markdown grid.
    """
    if not rows or len(rows) < 7:
        return ""
    joined = re.sub(r"\s+", "", "".join(
        str(cell or "") for row in rows for cell in row
    ))
    required = ("分子数值", "达标率", "权重", "得分")
    if not all(term in joined for term in required):
        return ""
    if "分母" not in joined and "分乓" not in joined:
        return ""
    if "融合后体验得分" not in joined:
        return ""

    number_re = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|秒|s|S|单|分)?")
    metric_rows: List[str] = []
    formula_lines: List[str] = []
    score_terms: List[Tuple[str, str]] = []
    fusion_rows: List[Tuple[str, str]] = []
    metric_count = 0
    in_fusion = False

    for row in rows[1:]:
        cells = [str(cell or "").strip() for cell in row]
        if not any(cells):
            continue
        row_text = " ".join(cell for cell in cells if cell)
        compact_row = re.sub(r"\s+", "", row_text)

        if ("麦当劳特殊场景得分" in row_text
                or "特殊场景得分=" in compact_row):
            formula = re.sub(r"\s+", "", row_text)
            formula_lines.append(formula)
            continue

        if (cells[0] == "融合后体验得分"
                or "融合后体验得分" in compact_row
                or "普通场景剔除后完成单" in compact_row):
            in_fusion = True

        if in_fusion:
            left = cells[1] if len(cells) > 1 else cells[0]
            right = cells[2] if len(cells) > 2 else ""
            left = re.sub(r"^指标", "", left).strip()
            right = re.sub(r"^数值", "", right).strip()
            if left and right:
                fusion_rows.append((left, right))
            elif left:
                fusion_rows.append((left, ""))
            continue

        if len(cells) < 2:
            continue
        left = " ".join(cell for cell in cells[:2] if cell).strip()
        right = " ".join(cells[2:]).strip()
        left_numbers = [re.sub(r"\s+", "", value)
                        for value in number_re.findall(left)]
        right_numbers = [re.sub(r"\s+", "", value)
                         for value in number_re.findall(right)]
        if len(left_numbers) < 2 or len(right_numbers) < 4:
            continue

        metric = number_re.sub(" ", left)
        metric = re.sub(r"一般超时|严重超时", " ", metric)
        metric = metric.replace("分乓", "分母")
        metric = re.sub(r"\s+", " ", metric).strip()
        if not metric:
            continue
        if len(left_numbers) >= 3 and ("准时率" in metric
                                       or "一般超时" in left
                                       or "严重超时" in left):
            numerator = f"一般超时{left_numbers[0]}；严重超时{left_numbers[-1]}"
            denominator = left_numbers[-2]
        else:
            numerator = left_numbers[0]
            denominator = left_numbers[1]
        metric_rows.append(
            "- 指标：{}；分子数值：{}；分母数值：{}；达标率（值）：{}；"
            "满分目标：{}；权重：{}；得分：{}".format(
                metric, numerator, denominator, *right_numbers[:4]
            )
        )
        score_terms.append((right_numbers[2], right_numbers[3]))
        metric_count += 1

    if metric_count < 3 or not fusion_rows:
        return ""
    if formula_lines and len(score_terms) >= 3:
        raw_formula = formula_lines[0]
        label_match = re.match(r"(.+?)=", raw_formula)
        total_match = re.search(r"=\s*(\d+(?:\.\d+)?)\s*分", raw_formula)
        if label_match and total_match:
            formula_lines = [
                label_match.group(1) + "="
                + "+".join(f"{weight}*{score}"
                           for weight, score in score_terms)
                + f"={total_match.group(1)}分"
            ]
    out = ["**特殊场景算分示例明细**", ""]
    out.extend(metric_rows)
    if formula_lines:
        out.extend(["", "**特殊场景得分公式**", ""])
        out.extend(line.replace("*", r"\*") for line in formula_lines)
    out.extend(["", "**融合后体验得分核算**", ""])
    for key, value in fusion_rows:
        if value:
            out.append(f"- {key}：{value}")
        else:
            out.append(f"- {key}")
    return "\n".join(out)
