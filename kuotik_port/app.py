from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .helpview import HelpView, help_root
from .theme import app_style

HELP_ROOT = help_root()

HELP_BOOKS = ["project1_3", "project2_3", "project3_3", "project4_3",
              "project5_3", "project6_3", "project7_3", "project8_3"]

LABS = [
    ("Работа 1 — Периодические сигналы", "lab1"),
    ("Работа 2 — Непериодические сигналы", "lab2"),
    ("Работа 3 — Дискретизация по времени", "lab3"),
    ("Работа 4 — Случайные сигналы", "lab4"),
    ("Работа 5 — Рациональное кодирование", "lab5"),
    ("Работа 6 — Пропускная способность", "lab6"),
    ("Работа 7 — Коды БЧХ", "lab7"),
    ("Работа 8 — Циклические коды", "lab8"),
]


def _placeholder(title: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lbl = QLabel(f"{title}\n\nперенос ещё не сделан")
    lbl.setStyleSheet("color:#666; font-size:15px;")
    lay.addWidget(lbl)
    return w


def _lab_pages() -> dict:
    from .labs import lab1
    pages = {"lab1": lab1.Lab1Page}
    try:
        from .labs import lab2
        pages["lab2"] = lab2.Lab2Page
    except ImportError:
        pass
    try:
        from .labs import lab3
        pages["lab3"] = lab3.Lab3Page
    except ImportError:
        pass
    try:
        from .labs import lab4
        pages["lab4"] = lab4.Lab4Page
    except ImportError:
        pass
    try:
        from .labs import lab5
        pages["lab5"] = lab5.Lab5Page
    except ImportError:
        pass
    try:
        from .labs import lab6
        pages["lab6"] = lab6.Lab6Page
    except ImportError:
        pass
    try:
        from .labs import lab7
        pages["lab7"] = lab7.Lab7Page
    except ImportError:
        pass
    try:
        from .labs import lab8
        pages["lab8"] = lab8.Lab8Page
    except ImportError:
        pass
    return pages


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("КУ ОТИК — macOS")
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setFixedWidth(330)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setTextElideMode(Qt.ElideRight)
        for title, _ in LABS:
            self.nav.addItem(title)
        self.nav.addItem("Справка")
        self.stack = QStackedWidget()
        builders = _lab_pages()
        for title, key in LABS:
            self.stack.addWidget(builders[key]() if key in builders else _placeholder(title))
        self.help = HelpView(HELP_ROOT)
        self.stack.addWidget(self.help)
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.title = QLabel("")
        self.title.setStyleSheet("font-size:15px; font-weight:600; color:#123c6e;")
        self.btn_help = QPushButton("Справка по работе")
        self.btn_help.clicked.connect(self.open_lab_help)
        bar.addWidget(self.title, 1)
        bar.addWidget(self.btn_help)
        rlay.addLayout(bar)
        rlay.addWidget(self.stack, 1)
        row.addWidget(self.nav)
        row.addWidget(right, 1)
        self.setCentralWidget(body)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.currentRowChanged.connect(self._on_row)
        self.nav.setCurrentRow(0)
        self._on_row(0)
        self.resize(1180, 780)


    def _on_row(self, row: int):
        if 0 <= row < len(LABS):
            self.title.setText(LABS[row][0])
            self.btn_help.setEnabled(True)
        else:
            self.title.setText("Справка")
            self.btn_help.setEnabled(False)

    def open_lab_help(self):
        row = self.nav.currentRow()
        if 0 <= row < len(HELP_BOOKS):
            self.help.open_book(HELP_BOOKS[row])
            self.nav.setCurrentRow(self.nav.count() - 1)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("КУ ОТИК")
    app_style(app)
    win = Main()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
