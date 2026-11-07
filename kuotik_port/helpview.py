"""Справка КУ ОТИК: распакованные .chm показываются как HTML в QTextBrowser."""

from __future__ import annotations

import html
import os
import re
import zipfile
from pathlib import Path

from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout, QWidget

TITLES = {
    "project": "Справка по заданиям",
    "projectl1": "Лекции, ч. I",
    "projectl2": "Лекции, ч. II",
    "projectz": "Сборник задач",
    "annotation": "Аннотация",
    "uchprog": "Программа курса",
    "calpl5": "Календарный план ч. I",
    "calpl6": "Календарный план ч. II",
}
PARTS = {1: "теория", 2: "подготовка", 3: "порядок выполнения"}


def _title(name: str) -> str:
    if name in TITLES:
        return TITLES[name]
    m = re.fullmatch(r"project(\d)_(\d)", name)
    if m:
        return f"Работа {m.group(1)}: {PARTS.get(int(m.group(2)), name)}"
    return name


def _books(root: Path) -> list[tuple[str, str]]:
    """Все распакованные книги справки, сначала общие, затем по номерам работ."""
    have = sorted(d.name for d in root.iterdir() if d.is_dir())
    order = ["project", "projectl1", "projectl2", "projectz", "annotation", "uchprog"] + \
            [f"project{i}_{j}" for i in range(1, 9) for j in (1, 2, 3)]
    names = [n for n in order if n in have] + [n for n in have if n not in order]
    return [(_title(n), n) for n in names]


def help_root() -> Path | None:
    """Справка: env KUOTIK_HELP, затем reference/chm рядом, затем распаковка bundled help.zip в кэш."""
    env = os.environ.get("KUOTIK_HELP")
    if env and Path(env).is_dir():
        return Path(env)
    local = Path(__file__).resolve().parent.parent / "reference" / "chm"
    if local.is_dir():
        return local
    cache = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / "help"
    stamp = cache / ".unpacked"
    if stamp.exists():
        return cache
    bundle = Path(__file__).resolve().parent / "assets" / "help.zip"
    if not bundle.exists():
        return None
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        z.extractall(cache)
    stamp.write_text("ok", encoding="utf-8")
    return cache


def _book_dir(root: Path, name: str) -> Path | None:
    for cand in (root / name, root / f"{name}.chm"):
        if cand.is_dir():
            return cand
    return None


def _pages(book: Path) -> list[tuple[str, Path]]:
    out = []
    for p in sorted(book.rglob("*.htm*")):
        title = p.stem
        try:
            head = p.read_text(encoding="cp1251", errors="replace")[:4000]
            m = re.search(r"<title[^>]*>(.*?)</title>", head, re.I | re.S)
            if m:
                title = html.unescape(m.group(1)).strip() or title
        except OSError:
            pass
        out.append((title, p))
    return out


class HelpView(QWidget):
    def __init__(self, root: Path | None):
        super().__init__()
        self.root = root
        lay = QVBoxLayout(self)
        if root is None:
            lay.addWidget(QLabel("Справка не найдена: положите reference/chm рядом с приложением"))
            return
        top = QHBoxLayout()
        top.addWidget(QLabel("Книга:"))
        self.books = QComboBox()
        for title, name in _books(root):
            if _book_dir(root, name):
                self.books.addItem(title, name)
        top.addWidget(self.books, 1)
        self.pages = QComboBox()
        top.addWidget(QLabel("Раздел:"))
        top.addWidget(self.pages, 2)
        lay.addLayout(top)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(self._anchor)
        lay.addWidget(self.view)
        self.books.currentIndexChanged.connect(self._load_book)
        self.pages.currentIndexChanged.connect(self._load_page)
        if self.books.count():
            self._load_book()

    def open_book(self, name: str, page_hint: str | None = None) -> bool:
        """Выбрать книгу по имени каталога (project5_3) и, если задан, раздел по подстроке заголовка."""
        idx = self.books.findData(name)
        if idx < 0:
            return False
        self.books.setCurrentIndex(idx)
        if page_hint:
            for i, (title, path) in enumerate(getattr(self, "_items", [])):
                if page_hint.lower() in title.lower() or page_hint.lower() in path.name.lower():
                    self.pages.setCurrentIndex(i)
                    break
        return True

    def _load_book(self):
        name = self.books.currentData()
        book = _book_dir(self.root, name) if name else None
        self.pages.clear()
        self._items: list[tuple[str, Path]] = _pages(book) if book else []
        for title, _ in self._items:
            self.pages.addItem(title)
        if self._items:
            self._load_page()

    def _load_page(self):
        idx = self.pages.currentIndex()
        if 0 <= idx < len(self._items):
            self._show(self._items[idx][1])

    def _show(self, path: Path):
        try:
            body = path.read_text(encoding="cp1251", errors="replace")
        except OSError:
            body = "<p>не читается</p>"
        self._current = path
        self.view.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + "/"))
        self.view.setHtml(body)

    def _anchor(self, url: QUrl):
        target = Path(url.toLocalFile()) if url.isLocalFile() else None
        if target and target.exists():
            self._show(target)
