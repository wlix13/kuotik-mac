"""Теория работы: страница справки вместо сокращённого пересказа."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from .helpview import help_root


def _page(book: str) -> Path | None:
    root = help_root()
    if root is None:
        return None
    d = root / book
    if not d.is_dir():
        return None
    pages = sorted(p for p in d.glob("*.htm*"))
    return pages[0] if pages else None


def theory_page(book: str) -> tuple[str, QUrl] | None:
    """HTML страницы справки и её база: формулы и картинки лежат рядом с .htm."""
    p = _page(book)
    if p is None:
        return None
    return p.read_text(encoding="cp1251", errors="replace"), QUrl.fromLocalFile(str(p.parent) + "/")


def theory_html(book: str) -> str | None:
    got = theory_page(book)
    return got[0] if got else None


def apply_theory(view: QTextBrowser, book: str, fallback: str) -> None:
    """Показать страницу справки в готовом QTextBrowser, с базой для картинок."""
    got = theory_page(book)
    if got:
        view.document().setBaseUrl(got[1])
        view.setHtml(got[0])
    else:
        view.setHtml(fallback)


def theory_tab(book: str, fallback: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    view = QTextBrowser()
    view.setOpenExternalLinks(False)
    got = theory_page(book)
    if got:
        view.document().setBaseUrl(got[1])
        view.setHtml(got[0])
    else:
        view.setHtml(fallback)
    lay.addWidget(view)
    return w
