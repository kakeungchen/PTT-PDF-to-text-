"""ptt — PDF to Text 图形界面（深色科技风）。"""
import os
import subprocess
import sys

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QStackedWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from .pipeline import convert

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")

QSS = """
* { font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif; }
QMainWindow, #root { background: #0D0F15; }

#brand      { color: #F2F5FA; font-size: 24px; font-weight: 800; letter-spacing: 1px; }
#brandSub   { color: #6B7689; font-size: 11px; letter-spacing: 3px; font-weight: 700; }
#privacy    { color: #34D399; font-size: 12px; font-weight: 600;
              background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.25);
              border-radius: 10px; padding: 4px 10px; }

#dropCard   { background: #12151D; border: 1.5px dashed #2A3142; border-radius: 16px; }
#dropCard:hover { border-color: #5B8CFF; }
#dropTitle  { color: #C9D2E1; font-size: 16px; font-weight: 700; }
#dropHint   { color: #5E6A7E; font-size: 12px; }

QListWidget { background: #12151D; border: 1.5px solid #1D2230; border-radius: 16px;
              color: #C9D2E1; font-size: 13px; padding: 8px; outline: none; }
QListWidget::item { background: #181C27; border-radius: 10px; padding: 10px 12px;
                    margin: 3px 2px; }
QListWidget::item:selected { background: #1F2535; color: #FFFFFF;
                             border: 1px solid #3A4566; }
QListWidget::item:hover { background: #1C2130; }

QCheckBox   { color: #AEB9CC; font-size: 13px; font-weight: 600; spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 6px;
                       border: 1.5px solid #323A4D; background: #141823; }
QCheckBox::indicator:checked { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                       stop:0 #3EE6FF, stop:1 #A855F7); border-color: transparent; }

#pathLabel  { color: #8A95A8; font-size: 12px; background: #141823;
              border: 1px solid #1F2533; border-radius: 9px; padding: 7px 12px; }
QPushButton#ghost { color: #AEB9CC; background: #161B26; border: 1px solid #262D3E;
                    border-radius: 9px; padding: 7px 14px; font-size: 12px; font-weight: 600; }
QPushButton#ghost:hover { border-color: #5B8CFF; color: #DCE4F2; }
QPushButton#ghost:disabled { color: #4A5366; }

QPushButton#primary { color: white; font-size: 15px; font-weight: 800; border: none;
    border-radius: 13px; padding: 13px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2BB3FF, stop:0.5 #5B8CFF, stop:1 #9D5CF6); }
QPushButton#primary:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #45C4FF, stop:0.5 #6F9BFF, stop:1 #AC70FF); }
QPushButton#primary:disabled { background: #232938; color: #5A6478; }

QProgressBar { background: #161B26; border: none; border-radius: 3px; }
QProgressBar::chunk { border-radius: 3px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3EE6FF, stop:1 #A855F7); }

#status     { color: #76829A; font-size: 12px; }
QTextEdit   { background: #0F121A; border: 1px solid #1B2130; border-radius: 12px;
              color: #93A0B8; font-family: "SF Mono", Menlo, monospace; font-size: 11px;
              padding: 8px; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: #2A3142; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


class Worker(QThread):
    progress = Signal(str, float)
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
                os.path.dirname(os.path.abspath(path)), "转换结果")
            try:
                self.log.emit(f"▸ 开始转换 {name}")
                res = convert(path, out_dir, formats=self.formats,
                              progress=lambda m, f: self.progress.emit(
                                  f"({i+1}/{n}) {name} · {m}", (i + f) / n))
                ok += 1
                for o in res["outputs"]:
                    self.log.emit(f"  ✓ {o}")
                if res["flagged_blocks"]:
                    self.log.emit(f"  ⚠ {res['flagged_blocks']} 处低置信内容已标注"
                                  "（Word 中为黄色高亮），建议人工核对")
                if not res.get("quality_ok", True):
                    self.log.emit("  ⚠ 质量审计发现问题，建议打开 Markdown 对照 PDF 复核")
                    for issue in res.get("qa_issues", [])[:8]:
                        self.log.emit(f"    - {issue}")
            except Exception as e:
                fail += 1
                self.log.emit(f"  ✗ 失败: {e}")
        self.all_done.emit(ok, fail)


class DropArea(QStackedWidget):
    """空状态显示引导卡片，有文件后显示列表。点击空卡片可选择文件。"""
    pick = Signal()

    def __init__(self):
        super().__init__()
        card = QFrame()
        card.setObjectName("dropCard")
        card.setCursor(QCursor(Qt.PointingHandCursor))
        v = QVBoxLayout(card)
        v.addStretch()
        t1 = QLabel("把 PDF 拖到这里")
        t1.setObjectName("dropTitle")
        t1.setAlignment(Qt.AlignCenter)
        t2 = QLabel("或点击选择文件 · 支持文本型 / 扫描型 / 超长截图 PDF")
        t2.setObjectName("dropHint")
        t2.setAlignment(Qt.AlignCenter)
        v.addWidget(t1)
        v.addSpacing(6)
        v.addWidget(t2)
        v.addStretch()
        card.mousePressEvent = lambda e: self.pick.emit()
        self.list = QListWidget()
        self.addWidget(card)
        self.addWidget(self.list)

    def refresh(self, has_files: bool):
        self.setCurrentIndex(1 if has_files else 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ptt — PDF to Text")
        self.resize(820, 700)
        self.setMinimumSize(660, 560)
        self.setAcceptDrops(True)
        self.files = []
        self.custom_out = None   # None = 默认（源文件旁的"转换结果"）
        self.last_out_dirs = set()

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(14)

        # ---- 品牌区 ----
        head = QHBoxLayout()
        logo = QLabel()
        icon = QIcon(os.path.join(_ASSETS, "logo.svg"))
        logo.setPixmap(icon.pixmap(QSize(46, 46)))
        head.addWidget(logo)
        head.addSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        brand = QLabel("ptt")
        brand.setObjectName("brand")
        sub = QLabel("PDF  ›  TEXT")
        sub.setObjectName("brandSub")
        col.addWidget(brand)
        col.addWidget(sub)
        head.addLayout(col)
        head.addStretch()
        privacy = QLabel("⛨ 100% 本地处理 · 文件不出设备")
        privacy.setObjectName("privacy")
        head.addWidget(privacy)
        lay.addLayout(head)

        # ---- 文件区 ----
        self.drop = DropArea()
        self.drop.pick.connect(self.pick_files)
        self.drop.setMinimumHeight(170)
        lay.addWidget(self.drop, stretch=3)

        # ---- 选项区 ----
        opts = QHBoxLayout()
        opts.setSpacing(14)
        self.cb_docx = QCheckBox("Word (.docx)")
        self.cb_docx.setChecked(False)
        self.cb_md = QCheckBox("Markdown (.md)")
        self.cb_md.setChecked(True)
        opts.addWidget(self.cb_docx)
        opts.addWidget(self.cb_md)
        opts.addStretch()
        btn_add = QPushButton("＋ 添加文件")
        btn_add.setObjectName("ghost")
        btn_add.clicked.connect(self.pick_files)
        btn_clear = QPushButton("清空")
        btn_clear.setObjectName("ghost")
        btn_clear.clicked.connect(self.clear_files)
        opts.addWidget(btn_add)
        opts.addWidget(btn_clear)
        lay.addLayout(opts)

        # ---- 输出目录 ----
        out_row = QHBoxLayout()
        out_row.setSpacing(10)
        self.path_label = QLabel("输出位置：默认（每个 PDF 旁的「转换结果」文件夹）")
        self.path_label.setObjectName("pathLabel")
        btn_out = QPushButton("自定义…")
        btn_out.setObjectName("ghost")
        btn_out.clicked.connect(self.pick_out_dir)
        btn_reset = QPushButton("恢复默认")
        btn_reset.setObjectName("ghost")
        btn_reset.clicked.connect(self.reset_out_dir)
        out_row.addWidget(self.path_label, stretch=1)
        out_row.addWidget(btn_out)
        out_row.addWidget(btn_reset)
        lay.addLayout(out_row)

        # ---- 行动区 ----
        act = QHBoxLayout()
        act.setSpacing(10)
        self.btn_start = QPushButton("开始转换")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start)
        self.btn_open = QPushButton("打开输出文件夹")
        self.btn_open.setObjectName("ghost")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self.open_out)
        act.addWidget(self.btn_start, stretch=3)
        act.addWidget(self.btn_open, stretch=1)
        lay.addLayout(act)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        lay.addWidget(self.bar)
        self.status = QLabel("就绪 · 宁慢勿错：转换后自动质检，低置信内容显式标注")
        self.status.setObjectName("status")
        lay.addWidget(self.status)

        self.logbox = QTextEdit()
        self.logbox.setReadOnly(True)
        lay.addWidget(self.logbox, stretch=2)
        self.worker = None

    # ---- 拖拽 ----
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile().lower().endswith(".pdf")]
        self.add_files(paths)

    # ---- 文件管理 ----
    def add_files(self, paths):
        for p in paths:
            if p and p not in self.files:
                self.files.append(p)
                item = QListWidgetItem(f"📄  {os.path.basename(p)}")
                item.setToolTip(p)
                self.drop.list.addItem(item)
        self.drop.refresh(bool(self.files))

    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择 PDF", "", "PDF 文件 (*.pdf)")
        self.add_files(paths)

    def clear_files(self):
        self.files.clear()
        self.drop.list.clear()
        self.drop.refresh(False)

    # ---- 输出目录 ----
    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if d:
            self.custom_out = d
            self.path_label.setText(f"输出位置：{d}")

    def reset_out_dir(self):
        self.custom_out = None
        self.path_label.setText("输出位置：默认（每个 PDF 旁的「转换结果」文件夹）")

    # ---- 转换 ----
    def start(self):
        if not self.files:
            QMessageBox.information(self, "提示", "请先添加 PDF 文件")
            return
        formats = [f for f, cb in (("md", self.cb_md), ("docx", self.cb_docx))
                   if cb.isChecked()]
        if not formats:
            QMessageBox.information(self, "提示", "请至少勾选一种输出格式")
            return
        self.btn_start.setEnabled(False)
        self.btn_start.setText("转换中…")
        self.last_out_dirs = ({self.custom_out} if self.custom_out else
                              {os.path.join(os.path.dirname(os.path.abspath(p)), "转换结果")
                               for p in self.files})
        self.worker = Worker(list(self.files), tuple(formats), self.custom_out)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.logbox.append)
        self.worker.all_done.connect(self.on_done)
        self.worker.start()

    def on_progress(self, msg, frac):
        self.status.setText(msg)
        self.bar.setValue(int(frac * 1000))

    def on_done(self, ok, fail):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始转换")
        self.btn_open.setEnabled(True)
        self.bar.setValue(1000)
        self.status.setText(f"完成 · 成功 {ok} 个" + (f"，失败 {fail} 个" if fail else ""))

    def open_out(self):
        for d in self.last_out_dirs:
            if os.path.isdir(d):
                subprocess.run(["open", d])


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ptt")
    app.setWindowIcon(QIcon(os.path.join(_ASSETS, "logo.svg")))
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
