"""Apple Vision OCR 封装 + 超长页分片识别。

所有坐标统一为"页像素空间"（y 向下）。两条取图路线：
- EmbeddedImageProvider：页面就是一张内嵌大图（长截图 PDF），用 macOS
  ImageIO 原生解码（libjpeg 解不开 >65500px 的超长 JPEG，ImageIO 可以）。
- RenderProvider：常规图片页/混合页，用 PyMuPDF 按区域渲染。
"""
import io
from typing import List, Optional

import fitz
import Quartz
import Vision
from Foundation import NSData
from PIL import Image, ImageFile

from .models import Line

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

STRIP_H = 1800       # 分片高度（像素）
OVERLAP = 240        # 相邻分片重叠区
TARGET_W = 4200      # OCR 前放大到约此宽度（实测显著降低字符级误识，且不变慢）
MAX_SCALE = 2.5


def _cgimage_from_png(png_bytes):
    data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    if src is None:
        return None
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


import re as _re

_DEC_SPACE = _re.compile(r'(\d)\.\s+(\d)')


def _normalize_ocr_text(t: str) -> str:
    """确定性的字符级纠错：只改无歧义的 OCR 习惯性错误。"""
    t = _DEC_SPACE.sub(r'\1.\2', t)          # 小数点后多余空格: 0. 1 -> 0.1
    # 数字上下文里的字母 O -> 0
    t = _re.sub(r'(?<=[\d［\[（(,，.])\s?O(?=[\d,，.．%］\]」)])', '0', t)
    t = _re.sub(r'(?<=[则为是于])O(?=分)', '0', t)
    # 公式里的乘号/字母混淆: M*×1% -> M*X1%
    t = _re.sub(r'(?<=\*)×', 'X', t)
    t = _re.sub(r'×(?=\d{1,2}%)', 'X', t)
    # 括号配对修复: ［…」 -> ［…］
    if '［' in t and '」' in t and '］' not in t:
        t = t.replace('」', '］')
    if '「' in t and '］' in t and '［' not in t:
        t = t.replace('「', '［')
    # ［］ 被误识为 L/J: "L-0.1,0.1J" -> "［-0.1,0.1］"
    t = _re.sub(r'(?<![A-Za-z])L(-?\d[\d.,，\s]*)[J」](?![A-Za-z])',
                r'［\1］', t)
    return t


def ocr_png(png_bytes: bytes, langs=("zh-Hans", "en-US")) -> List[Line]:
    """对一张 PNG 做 OCR，返回像素坐标的行列表（坐标相对该图，y 向下）。"""
    data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(list(langs))
    req.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR 失败: {err}")
    img = Image.open(io.BytesIO(png_bytes))
    W, H = img.size
    lines = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        text = _normalize_ocr_text(str(cand[0].string()).strip())
        if not text:
            continue
        bb = obs.boundingBox()  # 归一化坐标，原点在左下
        x0 = bb.origin.x * W
        y1 = (1 - bb.origin.y) * H
        y0 = (1 - bb.origin.y - bb.size.height) * H
        x1 = (bb.origin.x + bb.size.width) * W
        lines.append(Line(text=text, conf=float(obs.confidence()),
                          x0=x0, y0=y0, x1=x1, y1=y1))
    return lines


class StripProvider:
    """按 y 区间提供页面图像切片（PIL Image），统一像素坐标。"""
    width = 0
    height = 0

    def get_strip(self, y0: int, y1: int) -> Image.Image:
        raise NotImplementedError


class EmbeddedImageProvider(StripProvider):
    """整页就是一张内嵌图：ImageIO 原生解码后按需裁切。"""

    def __init__(self, raw_bytes: bytes):
        data = NSData.dataWithBytes_length_(raw_bytes, len(raw_bytes))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        self._img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if self._img is None:
            raise ValueError("ImageIO 无法解码内嵌图片")
        self.width = Quartz.CGImageGetWidth(self._img)
        self.height = Quartz.CGImageGetHeight(self._img)

    def get_strip(self, y0: int, y1: int) -> Image.Image:
        y0 = max(0, int(y0)); y1 = min(self.height, int(y1))
        rect = Quartz.CGRectMake(0, y0, self.width, y1 - y0)
        crop = Quartz.CGImageCreateWithImageInRect(self._img, rect)
        return _cgimage_to_pil(crop)


class RenderProvider(StripProvider):
    """常规页面：PyMuPDF 按 clip 渲染，缩放到约 TARGET_W*1.25 宽。"""

    def __init__(self, page: fitz.Page, target_w: int = 2000):
        self.page = page
        self.zoom = target_w / page.rect.width
        self.width = int(page.rect.width * self.zoom)
        self.height = int(page.rect.height * self.zoom)

    def get_strip(self, y0: int, y1: int) -> Image.Image:
        y0 = max(0, int(y0)); y1 = min(self.height, int(y1))
        clip = fitz.Rect(0, y0 / self.zoom, self.page.rect.width, y1 / self.zoom)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = self.page.get_pixmap(matrix=mat, clip=clip)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _cgimage_to_pil(cgimg) -> Image.Image:
    """CGImage -> PIL，经由 PNG 编码（避免手撕像素格式）。"""
    mdata = Quartz.CFDataCreateMutable(None, 0)
    dest = Quartz.CGImageDestinationCreateWithData(mdata, "public.png", 1, None)
    Quartz.CGImageDestinationAddImage(dest, cgimg, None)
    Quartz.CGImageDestinationFinalize(dest)
    return Image.open(io.BytesIO(bytes(mdata))).convert("RGB")


def _pil_to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def ocr_strip(img: Image.Image, langs=("zh-Hans", "en-US"),
              upscale_to: int = TARGET_W) -> List[Line]:
    """OCR 单个切片；放大到目标宽度提升识别率，坐标换算回原尺寸。"""
    scale = 1.0
    if upscale_to and img.width < upscale_to:
        scale = min(upscale_to / img.width, MAX_SCALE)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    lines = ocr_png(_pil_to_png(img), langs)
    if scale != 1.0:
        for ln in lines:
            ln.x0 /= scale; ln.x1 /= scale; ln.y0 /= scale; ln.y1 /= scale
    return lines


def ocr_provider(provider: StripProvider, langs=("zh-Hans", "en-US"),
                 progress=None) -> List[Line]:
    """分片 OCR 整页并在重叠区去重合并，返回页像素坐标的全部行。"""
    H = provider.height
    step = STRIP_H - OVERLAP
    starts = list(range(0, max(H - OVERLAP, 1), step))
    all_lines: List[Line] = []
    for i, s in enumerate(starts):
        e = min(s + STRIP_H, H)
        strip = provider.get_strip(s, e)
        lines = ocr_strip(strip, langs)
        for ln in lines:
            ln.y0 += s; ln.y1 += s
        # 重叠区去重：除首尾片外，只接收中心点落在本片"专属区"的行
        lo = s + OVERLAP / 2 if i > 0 else 0
        hi = e - OVERLAP / 2 if e < H else H
        all_lines.extend(ln for ln in lines if lo <= ln.cy < hi)
        if progress:
            progress(i + 1, len(starts))
    all_lines.sort(key=lambda l: (l.y0, l.x0))
    return all_lines
