from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from .forms import MASK_HINTS, Node

HANDLED = {}


def _int(node: Node, key: str, default: int = 0) -> int:
    v = node.get(key, default)
    return int(v) if isinstance(v, (int, float)) else default


def _place(w: QWidget, node: Node) -> None:
    w.setGeometry(_int(node, "Left"), _int(node, "Top"),
                  max(_int(node, "Width"), 1), max(_int(node, "Height"), 1))


def build(node: Node, parent: QWidget | None = None, assets: dict | None = None) -> QWidget:
    cls = node.cls
    w: QWidget
    if cls == "TPageControl":
        w = QTabWidget(parent)
        w.setDocumentMode(True)
        for kid in node.kids:
            if kid.cls == "TTabSheet":
                page = QWidget()
                page.setObjectName(kid.name)
                for gk in kid.kids:
                    build(gk, page, assets)
                w.addTab(page, kid.caption.strip() or kid.name)
        _place_tab(w, node, parent)
        return w
    if cls in ("TLabel", "TStaticText"):
        w = QLabel(node.caption, parent)
        w.setWordWrap(False)
    elif cls == "TButton":
        w = QPushButton(node.caption.replace("&", ""), parent)
        HANDLED[(node.name, node.get("OnClick"))] = w
    elif cls == "TMaskEdit":
        w = QLineEdit(parent)
        w.setText(node.get("Text", "") if isinstance(node.get("Text"), str) else "")
        w.setPlaceholderText(MASK_HINTS.get(str(node.get("EditMask", "")), ""))
        HANDLED[(node.name, node.get("OnExit"))] = w
    elif cls == "TEdit":
        w = QLineEdit(parent)
        if isinstance(node.get("Text"), str):
            w.setText(node.get("Text"))
    elif cls == "TSpinEdit":
        w = QSpinBox(parent)
        v = node.get("Value", 0)
        w.setValue(int(v) if isinstance(v, (int, float)) else 0)
    elif cls == "TGroupBox":
        w = QGroupBox(node.caption, parent)
        for kid in node.kids:
            build(kid, w, assets)
    elif cls == "TPanel":
        w = QFrame(parent)
        w.setFrameShape(QFrame.StyledPanel)
        for kid in node.kids:
            build(kid, w, assets)
    elif cls == "TTabSheet":
        w = QWidget(parent)
        w.setObjectName(node.name)
        for kid in node.kids:
            build(kid, w, assets)
    elif cls == "TImage":
        w = QLabel(parent)
        w.setAlignment(Qt.AlignCenter)
        w.setStyleSheet("border:1px dashed #bbb; color:#888;")
        pix = _pixmap(node, assets)
        if pix is not None:
            w.setPixmap(pix)
        else:
            w.setText(f"TImage {node.name}")
    elif cls == "TMemo":
        w = QTextEdit(parent)
        lines = node.get("Lines.Strings")
        if isinstance(lines, list):
            w.setPlainText("\n".join(str(x) for x in lines))
    elif cls == "TRadioButton":
        w = QRadioButton(node.caption, parent)
    elif cls == "TCheckBox":
        w = QCheckBox(node.caption, parent)
    elif cls == "TComboBox":
        w = QComboBox(parent)
        items = node.get("Items.Strings")
        if isinstance(items, list):
            w.addItems([str(i) for i in items])
    else:
        w = QLabel(f"[{cls}]", parent)
        w.setStyleSheet("color:#b00; border:1px solid #eee;")
    _place(w, node)
    w.setObjectName(node.name)
    if parent is not None:
        w.show()
    return w


def _place_tab(w: QTabWidget, node: Node, parent: QWidget | None) -> None:
    if parent is None:
        w.setGeometry(0, 0, _int(node, "Width", 792), _int(node, "Height", 556))
    else:
        _place(w, node)


def _pixmap(node: Node, assets: dict | None) -> QPixmap | None:
    if not assets:
        return None
    name = str(node.get("Picture") or "")
    for key, path in assets.items():
        if key.lower() in name.lower():
            pix = QPixmap(str(path))
            return pix if not pix.isNull() else None
    return None


def form_window(node: Node, assets: dict | None = None) -> QMainWindow:
    win = QMainWindow()
    win.setWindowTitle(node.caption.strip())
    host = QWidget()
    for kid in node.kids:
        build(kid, host, assets)
    win.setCentralWidget(host)
    win.resize(_int(node, "Width", 792), _int(node, "Height", 556))
    return win
