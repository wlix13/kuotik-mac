from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from kuotik_port.app import Main

out = Path(os.environ.get("SHOTS") or Path(tempfile.gettempdir()) / "shots")
out.mkdir(exist_ok=True)

app = QApplication([])
win = Main()
win.show()
app.processEvents()
print(f"pages: {win.stack.count()}  nav: {win.nav.count()}")

for i in range(win.nav.count()):
    win.nav.setCurrentRow(i)
    app.processEvents()
    print(f"  {i}: {win.nav.item(i).text()}")

win.nav.setCurrentRow(0)
app.processEvents()
page = win.stack.widget(0)
for t in range(page.tabs.count()):
    page.tabs.setCurrentIndex(t)
    app.processEvents()
    print(f"  lab1 tab {t}: {page.tabs.tabText(t)}")

from kuotik_port.labs.lab1 import PRESETS
from kuotik_port.mathlib.spectra import window_max

for name in PRESETS:
    page._load_preset(name=name)
    app.processEvents()
    print(f"  preset ok: {name} max={window_max(page.spec):.3f}")

page.tabs.setCurrentIndex(0)
app.processEvents()
win.grab().save(str(out / "lab1-theory.png"))
page.tabs.setCurrentIndex(1)
app.processEvents()
win.grab().save(str(out / "lab1-harmonics.png"))
page.tabs.setCurrentIndex(2)
app.processEvents()
win.grab().save(str(out / "lab1-signal.png"))
page.tabs.setCurrentIndex(3)
app.processEvents()
win.grab().save(str(out / "lab1-input.png"))
page.tabs.setCurrentIndex(4)
app.processEvents()
win.grab().save(str(out / "lab1-result.png"))

win.nav.setCurrentRow(8)
app.processEvents()
win.grab().save(str(out / "help.png"))
print("shots written to", out)
