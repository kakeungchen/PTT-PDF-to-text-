"""ptt — macOS 27 风格的本地 PDF OCR 图形界面。"""

from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import QFileInfo, Qt, QThread, Signal, QSize
from PySide6.QtGui import QCursor, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFileIconProvider,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .pipeline import convert


_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
_APP_ICON = os.path.join(_ASSETS, "logo-ocr.png")


QSS = """
* {
    font-family: ".AppleSystemUIFont", "PingFang SC", "Helvetica Neue", sans-serif;
    color: #1B1D23;
}
QMainWindow, QWidget#root { background: #F5F7FA; }
QToolTip {
    color: #FFFFFF; background: #24272E; border: none;
    border-radius: 7px; padding: 6px 8px; font-size: 12px;
}

/* Brand and navigation */
QLabel#brand { color: #17191F; font-size: 25px; font-weight: 700; }
QLabel#brandSub { color: #737B8A; font-size: 12px; font-weight: 500; }
QLabel#privacy { color: #667083; font-size: 12px; }
QFrame#segmented {
    background: rgba(236, 239, 244, 0.86);
    border: 1px solid #DCE1E9;
    border-radius: 14px;
}
QPushButton#tabButton {
    min-height: 36px; padding: 0 17px;
    color: #6D7482; background: transparent;
    border: 1px solid transparent; border-radius: 11px;
    font-size: 13px; font-weight: 600;
}
QPushButton#tabButton[active="true"] {
    color: #1A1D23; background: rgba(255, 255, 255, 0.96);
    border-color: #E1E5EB;
}
QPushButton#tabButton:hover { color: #1668E8; }

/* Drop surface */
QFrame#dropZone {
    background: rgba(255, 255, 255, 0.88);
    border: 1.5px dashed #C9D1DE;
    border-radius: 20px;
}
QFrame#dropZone:hover {
    background: #FFFFFF; border-color: #67A0FA;
}
QLabel#dropTitle { color: #252830; font-size: 20px; font-weight: 650; }
QLabel#dropHint { color: #8A92A1; font-size: 13px; }
QPushButton#outline {
    color: #1769E8; background: rgba(255,255,255,0.9);
    border: 1.5px solid #5B95F3; border-radius: 11px;
    min-height: 38px; padding: 0 22px;
    font-size: 13px; font-weight: 650;
}
QPushButton#outline:hover { background: #F2F7FF; border-color: #1769E8; }
QPushButton#outline:pressed { background: #E7F0FF; }

/* File queue */
QFrame#queuePanel {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #DDE2E9;
    border-radius: 17px;
}
QLabel#tableHeader {
    color: #7B8392; font-size: 11px; font-weight: 650;
}
QListWidget#fileList {
    background: transparent; border: none; outline: none; padding: 0;
}
QListWidget#fileList::item {
    background: transparent; border: none; margin: 0; padding: 0;
}
QListWidget#fileList::item:selected { background: #F4F7FC; }
QWidget#fileRow { background: transparent; border-top: 1px solid #EDF0F4; }
QLabel#fileName { color: #272A31; font-size: 13px; font-weight: 560; }
QLabel#fileMeta { color: #969EAB; font-size: 11px; }
QLabel#pageCount, QLabel#rowStatus { color: #697283; font-size: 12px; }
QFrame#statusDot { border: none; border-radius: 4px; background: #AEB6C3; }
QFrame#statusDot[state="processing"] { background: #3478F6; }
QFrame#statusDot[state="completed"] { background: #2DBA62; }
QFrame#statusDot[state="failed"] { background: #E5484D; }
QProgressBar#rowProgress {
    background: #E7EAF0; border: none; border-radius: 3px;
}
QProgressBar#rowProgress::chunk {
    background: #3478F6; border-radius: 3px;
}
QPushButton#iconButton {
    background: transparent; border: none; border-radius: 9px; padding: 5px;
}
QPushButton#iconButton:hover { background: #EEF2F8; }
QPushButton#iconButton:disabled { background: transparent; }

/* Bottom control dock */
QFrame#controlDock {
    background: rgba(255, 255, 255, 0.93);
    border: 1px solid #DCE1E9;
    border-radius: 18px;
}
QLabel#sectionLabel {
    color: #727B8A; font-size: 11px; font-weight: 650;
}
QCheckBox#formatCheck {
    color: #30343C; font-size: 13px; font-weight: 550; spacing: 8px;
    background: #F7F9FC; border: 1px solid #E0E5ED;
    border-radius: 10px; padding: 8px 11px;
}
QCheckBox#formatCheck:hover { border-color: #A9C6F5; background: #F3F7FD; }
QPushButton#destination {
    min-height: 38px; padding: 0 12px;
    color: #4D5563; background: #F7F9FC;
    border: 1px solid #E0E5ED; border-radius: 10px;
    font-size: 12px; text-align: left;
}
QPushButton#destination:hover { border-color: #A9C6F5; background: #F3F7FD; }
QPushButton#primary {
    min-height: 54px; min-width: 220px;
    color: #FFFFFF; font-size: 16px; font-weight: 700;
    border: none; border-radius: 14px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #3D7EF6, stop:1 #1267E8);
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #4B89FA, stop:1 #1F72ED);
}
QPushButton#primary:pressed { background: #125FCF; }
QPushButton#primary:disabled {
    color: #A7AFBC; background: #E7EAF0;
}
QFrame#divider { background: #E5E8ED; border: none; }

/* Status and history */
QLabel#globalStatus { color: #737C8A; font-size: 12px; }
QProgressBar#overallProgress {
    background: #E3E7ED; border: none; border-radius: 3px;
}
QProgressBar#overallProgress::chunk { background: #3478F6; border-radius: 3px; }
QLabel#pageTitle { color: #1C1F25; font-size: 25px; font-weight: 700; }
QLabel#pageSubtitle { color: #858D9B; font-size: 13px; }
QTextEdit#historyLog {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #DDE2E9; border-radius: 17px;
    color: #4E5765; font-family: "SF Mono", Menlo, monospace;
    font-size: 12px; padding: 16px;
    selection-background-color: #CFE1FF;
}
QScrollBar:vertical { background: transparent; width: 8px; margin: 3px; }
QScrollBar::handle:vertical {
    background: #C8CED8; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #AEB7C5; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


def _repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _human_size(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _pdf_page_count(path: str) -> int | None:
    try:
        import fitz

        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


class Worker(QThread):
    progress = Signal(str, float)
    file_started = Signal(int)
    file_progress = Signal(int, float)
    file_done = Signal(int, bool)
    all_done = Signal(int, int)
    log = Signal(str)

    def __init__(self, files, formats, out_dir):
        super().__init__()
        self.files, self.formats, self.out_dir = files, formats, out_dir

    def run(self):
        ok = fail = 0
        n = len(self.files)
        for i, path in enumerate(self.files):
            name = os.path.basename(path)
            out_dir = self.out_dir or os.path.join(
                os.path.dirname(os.path.abspath(path)), "转换结果"
            )
            self.file_started.emit(i)
            self.log.emit(f"开始转换  {name}")

            def report(message, fraction, index=i, filename=name):
                self.file_progress.emit(index, fraction)
                self.progress.emit(
                    f"{index + 1}/{n} · {filename} · {message}",
                    (index + fraction) / n,
                )

            try:
                res = convert(
                    path,
                    out_dir,
                    formats=self.formats,
                    progress=report,
                )
                ok += 1
                self.file_done.emit(i, True)
                for output in res["outputs"]:
                    self.log.emit(f"完成      {output}")
                if res["flagged_blocks"]:
                    self.log.emit(
                        f"需核对    {res['flagged_blocks']} 处低置信内容已标注，建议人工复核"
                    )
                if not res.get("quality_ok", True):
                    self.log.emit("质量提醒  建议对照 PDF 复核以下内容")
                    for issue in res.get("qa_issues", [])[:8]:
                        self.log.emit(f"          {issue}")
            except Exception as exc:
                fail += 1
                self.file_done.emit(i, False)
                self.log.emit(f"转换失败  {name} · {exc}")
        self.all_done.emit(ok, fail)


class DropZone(QFrame):
    pick = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMinimumHeight(190)
        self.setAccessibleName("PDF 文件拖放区域")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        layout.addStretch()

        icon = QLabel()
        icon.setAlignment(Qt.AlignCenter)
        icon.setPixmap(QIcon(_APP_ICON).pixmap(QSize(42, 42)))
        layout.addWidget(icon)

        title = QLabel("将 PDF 拖到这里")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        hint = QLabel("支持文本型、扫描型与超长截图 PDF")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        choose = QPushButton("选择文件")
        choose.setObjectName("outline")
        choose.setCursor(QCursor(Qt.PointingHandCursor))
        choose.clicked.connect(self.pick)
        choose.setAccessibleName("选择 PDF 文件")
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(choose)
        row.addStretch()
        layout.addSpacing(4)
        layout.addLayout(row)
        layout.addStretch()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pick.emit()
        super().mousePressEvent(event)


class FileRow(QWidget):
    remove_requested = Signal(str)

    def __init__(self, path: str, pages: int | None, parent=None):
        super().__init__(parent)
        self.path = path
        self.setObjectName("fileRow")
        self.setMinimumHeight(66)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 12, 8)
        layout.setSpacing(12)

        icon = QLabel()
        native_icon = QFileIconProvider().icon(QFileInfo(path))
        if native_icon.isNull():
            native_icon = self.style().standardIcon(QStyle.SP_FileIcon)
        icon.setPixmap(native_icon.pixmap(QSize(28, 28)))
        icon.setFixedWidth(30)
        layout.addWidget(icon)

        info = QVBoxLayout()
        info.setSpacing(2)
        name = QLabel(os.path.basename(path))
        name.setObjectName("fileName")
        name.setToolTip(path)
        name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meta = QLabel(_human_size(path))
        meta.setObjectName("fileMeta")
        info.addWidget(name)
        info.addWidget(meta)
        layout.addLayout(info, 1)

        page_text = "—" if pages is None else str(pages)
        self.pages = QLabel(page_text)
        self.pages.setObjectName("pageCount")
        self.pages.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pages.setFixedWidth(72)
        layout.addWidget(self.pages)

        self.status_box = QWidget()
        status_layout = QHBoxLayout(self.status_box)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)
        self.dot = QFrame()
        self.dot.setObjectName("statusDot")
        self.dot.setProperty("state", "waiting")
        self.dot.setFixedSize(8, 8)
        self.progress = QProgressBar()
        self.progress.setObjectName("rowProgress")
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(92, 6)
        self.progress.hide()
        self.status = QLabel("等待中")
        self.status.setObjectName("rowStatus")
        status_layout.addWidget(self.dot)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.status)
        status_layout.addStretch()
        self.status_box.setFixedWidth(250)
        layout.addWidget(self.status_box)

        self.remove = QPushButton()
        self.remove.setObjectName("iconButton")
        self.remove.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.remove.setIconSize(QSize(15, 15))
        self.remove.setFixedSize(30, 30)
        self.remove.setToolTip("移除文件")
        self.remove.setAccessibleName(f"移除 {os.path.basename(path)}")
        self.remove.clicked.connect(lambda: self.remove_requested.emit(self.path))
        layout.addWidget(self.remove)

    def set_state(self, state: str, fraction: float = 0.0):
        self.dot.setProperty("state", state)
        _repolish(self.dot)
        if state == "processing":
            self.progress.show()
            self.progress.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
            self.status.setText(f"转换中 {round(fraction * 100)}%")
        elif state == "completed":
            self.progress.hide()
            self.status.setText("转换完成")
        elif state == "failed":
            self.progress.hide()
            self.status.setText("转换失败")
        else:
            self.progress.hide()
            self.status.setText("等待中")

    def set_locked(self, locked: bool):
        self.remove.setDisabled(locked)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ptt — 本地 OCR")
        self.resize(1120, 800)
        self.setMinimumSize(860, 680)
        self.setAcceptDrops(True)
        self.files: list[str] = []
        self.file_rows: dict[str, FileRow] = {}
        self.custom_out: str | None = None
        self.last_out_dirs: set[str] = set()
        self.worker: Worker | None = None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        shell = QVBoxLayout(root)
        shell.setContentsMargins(26, 22, 26, 20)
        shell.setSpacing(15)

        shell.addLayout(self._build_header())

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_converter_page())
        self.pages.addWidget(self._build_history_page())
        shell.addWidget(self.pages, 1)
        self.set_tab(0)

    def _build_header(self):
        header = QHBoxLayout()
        header.setSpacing(12)

        logo = QLabel()
        logo.setPixmap(QIcon(_APP_ICON).pixmap(QSize(58, 58)))
        logo.setFixedSize(62, 62)
        logo.setAccessibleName("ptt OCR 应用图标")
        header.addWidget(logo)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(1)
        brand = QLabel("ptt")
        brand.setObjectName("brand")
        sub = QLabel("本地 OCR")
        sub.setObjectName("brandSub")
        privacy = QLabel("文件仅在这台 Mac 上处理")
        privacy.setObjectName("privacy")
        brand_col.addWidget(brand)
        brand_col.addWidget(sub)
        brand_col.addSpacing(2)
        brand_col.addWidget(privacy)
        header.addLayout(brand_col)
        header.addStretch()

        segmented = QFrame()
        segmented.setObjectName("segmented")
        nav = QHBoxLayout(segmented)
        nav.setContentsMargins(4, 4, 4, 4)
        nav.setSpacing(2)
        self.tab_convert = QPushButton("转换")
        self.tab_history = QPushButton("转换记录")
        for button in (self.tab_convert, self.tab_history):
            button.setObjectName("tabButton")
            button.setCursor(QCursor(Qt.PointingHandCursor))
            nav.addWidget(button)
        self.tab_convert.clicked.connect(lambda: self.set_tab(0))
        self.tab_history.clicked.connect(lambda: self.set_tab(1))
        header.addWidget(segmented)
        return header

    def _build_converter_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.drop_zone = DropZone()
        self.drop_zone.pick.connect(self.pick_files)
        layout.addWidget(self.drop_zone, 2)

        self.queue_panel = QFrame()
        self.queue_panel.setObjectName("queuePanel")
        queue_layout = QVBoxLayout(self.queue_panel)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(18, 11, 12, 10)
        header.setSpacing(12)
        file_head = QLabel("文件")
        file_head.setObjectName("tableHeader")
        page_head = QLabel("页数")
        page_head.setObjectName("tableHeader")
        page_head.setFixedWidth(72)
        status_head = QLabel("状态")
        status_head.setObjectName("tableHeader")
        status_head.setFixedWidth(250)
        action_spacer = QWidget()
        action_spacer.setFixedWidth(30)
        header.addWidget(file_head, 1)
        header.addWidget(page_head)
        header.addWidget(status_head)
        header.addWidget(action_spacer)
        queue_layout.addLayout(header)

        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setSelectionMode(QListWidget.NoSelection)
        self.file_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.file_list.setMinimumHeight(76)
        self.file_list.setMaximumHeight(224)
        queue_layout.addWidget(self.file_list)
        self.queue_panel.hide()
        layout.addWidget(self.queue_panel, 0)

        layout.addWidget(self._build_control_dock())

        status_row = QHBoxLayout()
        status_row.setContentsMargins(2, 0, 2, 0)
        status_row.setSpacing(8)
        self.global_dot = QFrame()
        self.global_dot.setObjectName("statusDot")
        self.global_dot.setProperty("state", "completed")
        self.global_dot.setFixedSize(8, 8)
        self.status = QLabel("就绪 · 转换完成后自动质检并标注低置信内容")
        self.status.setObjectName("globalStatus")
        self.overall_progress = QProgressBar()
        self.overall_progress.setObjectName("overallProgress")
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setTextVisible(False)
        self.overall_progress.setFixedSize(160, 6)
        self.overall_progress.hide()
        status_row.addWidget(self.global_dot)
        status_row.addWidget(self.status)
        status_row.addStretch()
        status_row.addWidget(self.overall_progress)
        layout.addLayout(status_row)
        return page

    def _build_control_dock(self):
        dock = QFrame()
        dock.setObjectName("controlDock")
        dock_layout = QHBoxLayout(dock)
        dock_layout.setContentsMargins(17, 14, 17, 14)
        dock_layout.setSpacing(18)

        format_col = QVBoxLayout()
        format_col.setSpacing(7)
        format_label = QLabel("输出格式")
        format_label.setObjectName("sectionLabel")
        format_col.addWidget(format_label)
        checks = QHBoxLayout()
        checks.setSpacing(8)
        self.cb_md = QCheckBox("Markdown  .md")
        self.cb_docx = QCheckBox("Word  .docx")
        for box in (self.cb_md, self.cb_docx):
            box.setObjectName("formatCheck")
            box.setCursor(QCursor(Qt.PointingHandCursor))
            checks.addWidget(box)
        self.cb_md.setChecked(True)
        self.cb_docx.setChecked(False)
        format_col.addLayout(checks)
        dock_layout.addLayout(format_col)

        dock_layout.addWidget(self._vertical_divider())

        out_col = QVBoxLayout()
        out_col.setSpacing(7)
        out_label = QLabel("输出位置")
        out_label.setObjectName("sectionLabel")
        out_col.addWidget(out_label)
        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        self.btn_out = QPushButton("源文件旁的「转换结果」")
        self.btn_out.setObjectName("destination")
        self.btn_out.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.btn_out.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_out.setToolTip("选择自定义输出文件夹")
        self.btn_out.clicked.connect(self.pick_out_dir)
        self.btn_reset = QPushButton()
        self.btn_reset.setObjectName("iconButton")
        self.btn_reset.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )
        self.btn_reset.setFixedSize(34, 34)
        self.btn_reset.setToolTip("恢复默认输出位置")
        self.btn_reset.clicked.connect(self.reset_out_dir)
        self.btn_reset.hide()
        out_row.addWidget(self.btn_out, 1)
        out_row.addWidget(self.btn_reset)
        out_col.addLayout(out_row)
        dock_layout.addLayout(out_col, 1)

        dock_layout.addWidget(self._vertical_divider())

        self.btn_start = QPushButton("开始转换")
        self.btn_start.setObjectName("primary")
        self.btn_start.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_start.setAccessibleName("开始转换 PDF")
        dock_layout.addWidget(self.btn_start)
        return dock

    def _build_history_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        title = QLabel("转换记录")
        title.setObjectName("pageTitle")
        subtitle = QLabel("转换进度、输出文件与质量提醒会保留在这里")
        subtitle.setObjectName("pageSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        title_row.addLayout(title_col)
        title_row.addStretch()
        self.btn_open = QPushButton("打开输出文件夹")
        self.btn_open.setObjectName("outline")
        self.btn_open.setEnabled(False)
        self.btn_open.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_open.clicked.connect(self.open_out)
        title_row.addWidget(self.btn_open)
        layout.addLayout(title_row)
        layout.addSpacing(8)

        self.logbox = QTextEdit()
        self.logbox.setObjectName("historyLog")
        self.logbox.setReadOnly(True)
        self.logbox.setPlaceholderText("还没有转换记录")
        layout.addWidget(self.logbox, 1)
        return page

    @staticmethod
    def _vertical_divider():
        line = QFrame()
        line.setObjectName("divider")
        line.setFixedSize(1, 52)
        return line

    def set_tab(self, index: int):
        self.pages.setCurrentIndex(index)
        for i, button in enumerate((self.tab_convert, self.tab_history)):
            button.setProperty("active", "true" if i == index else "false")
            _repolish(button)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(
            url.toLocalFile().lower().endswith(".pdf")
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.toLocalFile().lower().endswith(".pdf")
        ]
        self.add_files(paths)
        event.acceptProposedAction()

    def add_files(self, paths):
        for path in paths:
            if not path or path in self.files or not path.lower().endswith(".pdf"):
                continue
            self.files.append(path)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 66))
            self.file_list.addItem(item)
            row = FileRow(path, _pdf_page_count(path))
            row.remove_requested.connect(self.remove_file)
            self.file_list.setItemWidget(item, row)
            self.file_rows[path] = row
        self._refresh_queue()

    def remove_file(self, path: str):
        if self.worker and self.worker.isRunning():
            return
        for index in range(self.file_list.count()):
            item = self.file_list.item(index)
            widget = self.file_list.itemWidget(item)
            if widget is not None and widget.path == path:
                self.file_list.takeItem(index)
                break
        if path in self.files:
            self.files.remove(path)
        self.file_rows.pop(path, None)
        self._refresh_queue()

    def _refresh_queue(self):
        has_files = bool(self.files)
        self.queue_panel.setVisible(has_files)
        self.btn_start.setEnabled(has_files)
        count = len(self.files)
        self.btn_start.setText("开始转换" if not count else f"开始转换 · {count} 个文件")

    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择 PDF", "", "PDF 文件 (*.pdf)"
        )
        self.add_files(paths)

    def pick_out_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if directory:
            self.custom_out = directory
            self.btn_out.setText(os.path.basename(directory) or directory)
            self.btn_out.setToolTip(directory)
            self.btn_reset.show()

    def reset_out_dir(self):
        self.custom_out = None
        self.btn_out.setText("源文件旁的「转换结果」")
        self.btn_out.setToolTip("选择自定义输出文件夹")
        self.btn_reset.hide()

    def _selected_formats(self):
        return [
            fmt
            for fmt, checkbox in (("md", self.cb_md), ("docx", self.cb_docx))
            if checkbox.isChecked()
        ]

    def start(self):
        if not self.files:
            QMessageBox.information(self, "选择文件", "请先添加 PDF 文件")
            return
        formats = self._selected_formats()
        if not formats:
            QMessageBox.information(self, "选择格式", "请至少选择一种输出格式")
            return

        self.set_tab(0)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("正在转换…")
        self.btn_out.setEnabled(False)
        self.cb_md.setEnabled(False)
        self.cb_docx.setEnabled(False)
        for row in self.file_rows.values():
            row.set_state("waiting")
            row.set_locked(True)

        self.global_dot.setProperty("state", "processing")
        _repolish(self.global_dot)
        self.overall_progress.setValue(0)
        self.overall_progress.show()
        self.status.setText("正在准备本地 OCR…")

        self.last_out_dirs = (
            {self.custom_out}
            if self.custom_out
            else {
                os.path.join(os.path.dirname(os.path.abspath(path)), "转换结果")
                for path in self.files
            }
        )
        self.worker = Worker(list(self.files), tuple(formats), self.custom_out)
        self.worker.progress.connect(self.on_progress)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_progress.connect(self.on_file_progress)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.log.connect(self.logbox.append)
        self.worker.all_done.connect(self.on_done)
        self.worker.start()

    def on_file_started(self, index: int):
        if 0 <= index < len(self.files):
            self.file_rows[self.files[index]].set_state("processing", 0.0)

    def on_file_progress(self, index: int, fraction: float):
        if 0 <= index < len(self.files):
            self.file_rows[self.files[index]].set_state("processing", fraction)

    def on_file_done(self, index: int, success: bool):
        if 0 <= index < len(self.files):
            self.file_rows[self.files[index]].set_state(
                "completed" if success else "failed"
            )

    def on_progress(self, message: str, fraction: float):
        self.status.setText(message)
        self.overall_progress.setValue(int(fraction * 1000))

    def on_done(self, ok: int, fail: int):
        self.btn_start.setEnabled(True)
        self.btn_start.setText(f"再次转换 · {len(self.files)} 个文件")
        self.btn_out.setEnabled(True)
        self.cb_md.setEnabled(True)
        self.cb_docx.setEnabled(True)
        for row in self.file_rows.values():
            row.set_locked(False)
        self.btn_open.setEnabled(bool(self.last_out_dirs))
        self.overall_progress.setValue(1000)
        self.global_dot.setProperty("state", "failed" if fail else "completed")
        _repolish(self.global_dot)
        summary = f"转换完成 · 成功 {ok} 个"
        if fail:
            summary += f"，失败 {fail} 个"
        self.status.setText(summary)

    def open_out(self):
        for directory in self.last_out_dirs:
            if directory and os.path.isdir(directory):
                subprocess.run(["open", directory], check=False)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ptt")
    app.setApplicationDisplayName("ptt — 本地 OCR")
    app.setWindowIcon(QIcon(_APP_ICON))
    app.setStyleSheet(QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
