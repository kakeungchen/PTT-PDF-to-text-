"""统一中间结构：每页提取结果都先变成 Block 列表，再统一导出。"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# bbox 均为 (x0, y0, x1, y1)，单位是该页的"像素空间"（OCR 路线）或 pt（文本路线）


@dataclass
class Block:
    kind: str  # heading / para / table / image / list
    text: str = ""
    level: int = 0  # heading 层级 1-4
    rows: Optional[List[List[str]]] = None  # table 单元格
    image_path: str = ""  # image 块对应的资源文件
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    page: int = 0
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)  # 质检警告，如 low_confidence


@dataclass
class Line:
    """OCR 识别出的一行文字（页像素坐标，y 向下）。"""
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def h(self):
        return self.y1 - self.y0


@dataclass
class DocResult:
    blocks: List[Block] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
