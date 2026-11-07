"""Справка КУ ОТИК: распакованные .chm показываются как HTML в QTextBrowser."""

from __future__ import annotations

import html
import os
import re
import zipfile
from pathlib import Path

from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout, QWidget

BOOKS = [
    ("Справка по заданиям", "project"),
    ("Лекции, ч. I", "projectl1"),
    ("Лекции, ч. II", "projectl2"),
    ("Сборник задач", "projectz"),
    ("Аннотация", "annotation"),
    ("Программа курса", "uchprog"),
]


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
        for title, name in BOOKS:
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
        base = QUrl.fromLocalFile(str(path.parent) + "/").toString()
        tag = f'<base href="{base}">'
        if "<head" in body.lower():
            body = re.sub(r"(<head[^>]*>)", r"\1" + tag, body, count=1, flags=re.I)
        else:
            body = tag + body
        self._current = path
        self.view.setHtml(body)

    def _anchor(self, url: QUrl):
        target = Path(url.toLocalFile()) if url.isLocalFile() else None
        if target and target.exists():
            self._show(target)
