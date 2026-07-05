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
from .normalize import normalize_blocks
from .vision_ocr import (EmbeddedImageProvider, RenderProvider, StripProvider,
                         ocr_provider)

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


def _append_raw_line_regions(result: DocResult, lines, page: int) -> None:
    regions = result.meta.setdefault("raw_regions", [])
    for idx, line in enumerate(lines):
        regions.append({
            "stage": "ocr_line",
            "index": idx,
            "kind": "line",
            "text": line.text,
            "rows": None,
            "bbox": (line.x0, line.y0, line.x1, line.y1),
            "page": page,
            "confidence": line.conf,
            "flags": [],
        })


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
    issues.extend(_blocking_issues_from_warnings(result.warnings, has_markdown=bool(md_paths)))
    blocking_issues: List[str] = []
    qa_warnings: List[str] = []
    for issue in issues:
        if "表格疑似列错位" in issue:
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
                                   has_markdown: bool = True) -> List[str]:
    issues: List[str] = []
    if not has_markdown:
        return issues
    for warning in warnings:
        if "低置信区域已转为截图" in warning:
            issues.append(
                "单文件Markdown存在未输出的低置信截图区域: "
                + warning
            )
    return issues
