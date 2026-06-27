"""主管线：逐页判型 -> 提取 -> 质检 -> 导出。"""
import os
import re
import shutil
from typing import Callable, List, Optional

import fitz

from . import assemble, qa, text_extract
from .export import export_docx, export_markdown
from .models import Block, DocResult
from .normalize import normalize_blocks
from .vision_ocr import (EmbeddedImageProvider, RenderProvider, StripProvider,
                         ocr_provider)

ProgressFn = Callable[[str, float], None]  # (消息, 0~1)


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


def convert(pdf_path: str, out_dir: str, formats=("md", "docx"),
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
                doc, pno, repeated, assets_dir, safe)
            result.blocks.extend(blocks)
            result.warnings.extend(notes)
        else:
            report(f"第 {pno+1}/{n} 页：OCR 识别", base)
            provider = _page_provider(doc, pno)
            lines = ocr_provider(
                provider,
                progress=lambda i, m: report(
                    f"第 {pno+1}/{n} 页：OCR {i}/{m} 片", base + (i / m) * 0.7 / n))
            blocks, notes = assemble.assemble_blocks(provider, lines, page=pno)
            result.warnings.extend(notes)
            report(f"第 {pno+1}/{n} 页：复核低置信内容", base + 0.75 / n)
            notes2 = qa.recheck_ocr_blocks(blocks, provider)
            result.warnings.extend(notes2)
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

    # 全文词频投票纠错（形近字：封项值→封顶值、惺星→1星 等）
    vote_notes = qa.doc_vote_fix(result.blocks)
    if vote_notes:
        result.warnings.extend(vote_notes[:10])
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

    # Markdown 现在文本化图片/公式/表格，不再依赖外部 assets；
    # docx 图片已在保存时内嵌，导出结束后也可删除临时图片目录。
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
        "qa_issues": issues,
        "flagged_blocks": n_flag,
    }
