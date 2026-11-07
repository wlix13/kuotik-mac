"""Единый вид приложения: шрифты, рамки, отступы — ближе к исходным формам VCL."""

from __future__ import annotations

PAGE_CSS = """<style>
body { font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
       font-size: 13px; line-height: 1.5; color: #1c1c1c; margin: 16px 20px; }
p { margin: 0 0 10px 0; }
h1, h2, h3, h4 { color: #123c6e; margin: 16px 0 8px 0; }
h1 { font-size: 19px; } h2 { font-size: 17px; } h3 { font-size: 15px; }
img { vertical-align: middle; }
table { border-collapse: collapse; margin: 10px 0; }
td, th { border: 1px solid #c9c9c9; padding: 4px 8px; font-size: 12px; }
th { background: #eef2f7; }
pre { background: #f6f6f6; border: 1px solid #e0e0e0; padding: 10px; }
a { color: #16599c; }
</style>"""

APP_QSS = """
QWidget { font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
          font-size: 13px; color: #1c1c1c; }
QMainWindow, QDialog { background: #f0f0f0; }

QListWidget { background: #ffffff; border: 1px solid #a8a8a8; padding: 2px; font-size: 12px; }
QListWidget::item { padding: 6px 7px; }
QListWidget::item:selected { background: #cce4f7; color: #10365f; }

QTabWidget::pane { border: 1px solid #a8a8a8; background: #ffffff; top: -1px; }
QTabBar::tab { background: #e8e8e8; border: 1px solid #a8a8a8; border-bottom: none;
               padding: 6px 12px; margin-right: 2px; }
QTabBar::tab:selected { background: #ffffff; font-weight: 600; color: #10365f;
                        border-bottom: 2px solid #0078d7; }
QTabBar::tab:hover:!selected { background: #f2f2f2; }

QPushButton { background: #e1e1e1; border: 1px solid #adadad; padding: 5px 14px; }
QPushButton:hover { background: #e5f1fb; border-color: #0078d7; }
QPushButton:pressed { background: #cce4f7; }
QPushButton:default { border-color: #0078d7; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff; border: 1px solid #abadb3; padding: 3px 6px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #0078d7; }
QComboBox::drop-down { width: 18px; border-left: 1px solid #abadb3; }

QGroupBox { border: 1px solid #c4c4c4; margin-top: 10px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; color: #24405e; }

QTableWidget, QTableView { background: #ffffff; border: 1px solid #a8a8a8;
                           gridline-color: #dcdcdc; alternate-background-color: #f7fafd; }
QHeaderView::section { background: #eef2f7; border: 1px solid #c9c9c9; padding: 4px 6px; }

QTextBrowser { background: #ffffff; border: 1px solid #a8a8a8; }
QLabel#status { color: #14624a; }
"""


def app_style(app) -> None:
    """Fusion одинаково выглядит на macOS и Linux, дальше правит QSS."""
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)


def style_page(html: str) -> str:
    """Вставить общий CSS страниц справки в HTML перед разметкой."""
    return PAGE_CSS + html
