from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtWidgets import QApplication

from kuotik_port.labs.lab2 import Lab2Page
from kuotik_port.mathlib import pulses as p

OUT = Path(os.environ.get("SHOTS", "/tmp/shots-lab2"))
OUT.mkdir(parents=True, exist_ok=True)

def curves(cv):
    return [ln for ln in cv.ax.lines if not ln.get_label().startswith("_")]


TABS = ["Общие сведения", "S(w)", "Q(w)", "Энергия в спектре", "Ряд Котельникова"]

app = QApplication([])
page = Lab2Page()
page.resize(1100, 760)
page.show()
app.processEvents()

got = [page.tabs.tabText(i) for i in range(page.tabs.count())]
assert got == TABS, got
print("вкладки:", ", ".join(got))

for i, name in enumerate(TABS):
    page.tabs.setCurrentIndex(i)
    app.processEvents()
    shot = OUT / f"{i + 1:02d} {name}.png"
    page.grab().save(str(shot))
    print(f"  снимок {shot}")

page.tabs.setCurrentIndex(1)
page.s_h.setValue(2.0)
page.s_b.setValue(1.0)
page.s_t0.setValue(0.0)
page.bt_s_up.click()
app.processEvents()
y = curves(page.cv_s)[0].get_ydata()
assert abs(y[0] - 2.0) < 1e-9, y[0]
print(f"S(0) при h = 2, β = 1: {y[0]:.6f} = h/β")

page.s_t0.setValue(2.0)
page.bt_s_up.click()
app.processEvents()
assert len(curves(page.cv_s)) == 2
assert np.allclose(curves(page.cv_s)[0].get_ydata(), curves(page.cv_s)[1].get_ydata())
print("S(ω) не зависит от t0: кривые совпали")

page.s_b.setValue(0.5)
page.bt_s_up.click()
app.processEvents()
assert len(curves(page.cv_s)) == 3
assert abs(curves(page.cv_s)[2].get_ydata()[0] - 4.0) < 1e-9
assert page.cv_s.ax.get_legend() is not None
print("семейство S(ω): три кривые, S(0) для β = 0.5 равно 4")

page.bt_s_clear.click()
page.s_h.setValue(2.0)
page.s_b.setValue(0.0)
page.bt_s_up.click()
app.processEvents()
line = curves(page.cv_s)[0]
w, s = line.get_xdata(), line.get_ydata()
i = int(np.abs(w - 0.5).argmin())
assert abs(s[i] - 2.0 / w[i]) < 1e-9, (w[i], s[i])
print(f"сигнал включения: S({w[i]:.3f}) = {s[i]:.4f} = h/ω")
assert "∞" in page.status.text(), page.status.text()

page.tabs.setCurrentIndex(2)
page.q_h.setValue(2.0)
page.q_b.setValue(1.0)
page.q_t0.setValue(0.0)
page.bt_q_up.click()
page.q_t0.setValue(1.0)
page.bt_q_up.click()
app.processEvents()
assert len(curves(page.cv_q)) == 2
wq = curves(page.cv_q)[1].get_xdata()
tq = curves(page.cv_q)[1].get_ydata()
k = int(np.abs(wq - 1.0).argmin())
want = -wq[k] * 1.0 - math.atan(wq[k] / 1.0)
assert abs(tq[k] - want) < 1e-9, (tq[k], want)
print(f"Θ({wq[k]:.3f}) при t0 = 1, β = 1: {tq[k]:.4f} рад")

page.bt_q_clear.click()
page.q_b.setValue(0.0)
page.q_t0.setValue(0.5)
page.bt_q_up.click()
app.processEvents()
line = curves(page.cv_q)[0]
w, t = line.get_xdata(), line.get_ydata()
k = int(np.abs(w - 1.0).argmin())
assert abs(t[k] - (-0.5 * w[k] - math.pi / 2)) < 1e-9, t[k]
print("сигнал включения: Θ(ω) = −ω·t0 − π/2")

page.tabs.setCurrentIndex(3)
page.e_h.setValue(2.0)
page.e_b.setValue(0.4)
page.e_lam.setValue(0.9)
page.bt_e_up.click()
app.processEvents()
wc = page.wc
dt0 = page.dt0
assert abs(wc - p.wc_of(0.9, 0.4)) < 1e-12
assert abs(p.lam(wc, 0.4) - 0.9) < 1e-12, p.lam(wc, 0.4)
assert abs(dt0 - math.pi / wc) < 1e-12
assert round(wc, 2) == 2.53, wc
assert page.e_wc.text() == f"ωc = {wc:.2f}", page.e_wc.text()
assert page.e_dt.text() == f"Δt = {dt0:.2f}", page.e_dt.text()
assert page.k_min.text() == f"Min = {dt0 / 2:.2f}", page.k_min.text()
assert page.k_max.text() == f"Max = {1.5 * dt0:.2f}", page.k_max.text()
assert abs(page.dt.value() - dt0) < 0.01
print(f"λ = 0.9, β = 0.4 → ωc = {wc:.4f}, λ(ωc) = {p.lam(wc, 0.4):.6f}, Δt0 = {dt0:.4f} = π/ωc")

page.e_b.setValue(1.0)
page.bt_e_up.click()
page.e_b.setValue(2.0)
page.bt_e_up.click()
app.processEvents()
assert len(page.e_curves) == 3
assert len(curves(page.cv_e)) >= 3
print(f"семейство λ(ωc): β = {', '.join(f'{b:g}' for b in page.e_curves)}")

page.e_b.setValue(0.4)
page.bt_e_up.click()
app.processEvents()
page.tabs.setCurrentIndex(4)
page.dt.setValue(round(dt0, 2))
page.bt_k_up.click()
app.processEvents()
assert "в диапазоне" in page.status.text(), page.status.text()
ks = np.arange(1, 3)
s_exact = p.kotelnikov(ks * dt0, 2.0, 0.4, wc, dt0)
assert np.allclose(s_exact, p.x(ks * dt0, 2.0, 0.4, 0.0), atol=1e-9), s_exact
line_sum = curves(page.cv_k)[1]
ts, ys = line_sum.get_xdata(), line_sum.get_ydata()
dt_in = page.dt.value()
idx = [int(np.abs(ts - v).argmin()) for v in ks * dt0]
for i in idx:
    assert abs(ys[i] - p.kotelnikov(np.array([ts[i]]), 2.0, 0.4, wc, dt_in)[0]) < 1e-9
peak = float(ys.max())
assert abs(peak - 2.0) < 0.12, peak
print(f"ряд при Δt ≈ Δt0: пик суммы {peak:.3f}, отсчёты на месте")

page.dt.setValue(round(dt0 / 3, 3))
page.bt_k_up.click()
app.processEvents()
assert "вне" in page.status.text(), page.status.text()
assert "меньше" in page.status.text()
warn_small = page.status.text()
page.tabs.setCurrentIndex(4)
page.grab().save(str(OUT / "06 Ряд Котельникова — Δt вне диапазона.png"))
page.dt.setValue(round(3 * dt0, 3))
page.bt_k_up.click()
app.processEvents()
assert "вне" in page.status.text() and "больше" in page.status.text(), page.status.text()
print("предупреждения Δt:", warn_small[:48], "…")

from kuotik_port.app import Main

win = Main()
win.show()
app.processEvents()
lab2 = win.stack.widget(1)
assert isinstance(lab2, Lab2Page), type(lab2)
for i in range(lab2.tabs.count()):
    lab2.tabs.setCurrentIndex(i)
    app.processEvents()
win.nav.setCurrentRow(1)
app.processEvents()
win.grab().save(str(OUT / "07 Главное окно — ЛР2.png"))
print("страница ЛР2 подхвачена главным окном, вкладка:", win.nav.item(1).text())
print("готово:", OUT)
