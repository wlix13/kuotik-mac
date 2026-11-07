"""КУ ОТИК для macOS: список работ слева, вкладки работы справа, справка внутри."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QStackedWidget, QVBoxLayout, QWidget)

from .helpview import HelpView, help_root

HELP_ROOT = help_root()

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
    return pages


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("КУ ОТИК — macOS")
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setMaximumWidth(300)
        for title, _ in LABS:
            self.nav.addItem(title)
        self.nav.addItem("Справка")
        self.stack = QStackedWidget()
        builders = _lab_pages()
        for title, key in LABS:
            self.stack.addWidget(builders[key]() if key in builders else _placeholder(title))
        self.stack.addWidget(HelpView(HELP_ROOT))
        row.addWidget(self.nav)
        row.addWidget(self.stack, 1)
        self.setCentralWidget(body)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.resize(1180, 780)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("КУ ОТИК")
    win = Main()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
