"""ЛР2: непериодические сигналы — вкладки как в КУ ОТИК, расчёт из mathlib/pulses.py."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import (QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..mathlib import pulses as p
from ..plot import Canvas

W_MAX = 3.0                                      # ось ω графиков программы, рад/с

THEORY = """<h3>Экспоненциальный импульс</h3>
<p>x(t) = h·e<sup>−β(t − t0)</sup> при t ≥ t0, x(t) = 0 при t &lt; t0. h — высота, β &gt; 0 — скорость
затухания, t0 — момент включения.</p>
<h4>Спектральная плотность</h4>
<p>S(jω) = ∫x(t)e<sup>−jωt</sup>dt = h·e<sup>−jωt0</sup>/(β + jω) = S(ω)·e<sup>jΘ(ω)</sup>.</p>
<p>S(ω) = h/√(β² + ω²) — модуль; S(0) = h/β; от t0 не зависит.</p>
<p>Θ(ω) = −ω·t<sub>0</sub> − arctg(ω/β) — аргумент: t0 даёт линейный набег, β — уход к −π/2.</p>
<p>Сигнал включения — предел β → 0 (в программе β = 00.00): x(t) = h при t ≥ t0,
S(ω) = h/ω, Θ(ω) = −ω·t0 − π/2.</p>
<h4>Энергия в спектре</h4>
<p>Э = ∫x²dt = h²/(2β) = (1/π)∫₀<sup>∞</sup>S²(ω)dω. Доля энергии в полосе 0…ωc:
λ(ωc) = (2/π)·arctg(ωc/β), откуда ωc = β·tg(πλ/2), шаг дискретизации Δt = π/ωc.</p>
<h4>Ряд Котельникова</h4>
<p>x̃(t) = Σ<sub>k≥0</sub> x(kΔt)·sin(ωc(t − kΔt))/(ωc(t − kΔt)). Программа берёт ωc с вкладки
«Энергия в спектре» и принимает Δt из [Δt0/2; 3Δt0/2], Δt0 = π/ωc. При Δt = Δt0 сумма проходит
через отсчёты.</p>
<h4>Порядок выполнения</h4>
<ol><li>Обучающие вкладки.</li><li>Семейства S(ω) по h, β, t0.</li><li>Семейства Θ(ω).</li>
<li>Сигнал включения (β = 0).</li><li>Интегральная кривая энергии λ(ωc).</li>
<li>Ряд Котельникова: Δt = Min, Δt0, Max.</li></ol>
<p>«Обновить» добавляет кривую к нарисованным, «Очистить» стирает график вкладки.</p>"""


def _lam(w, b):
    """λ(ωc) для массива ωc; p.lam — только для скаляров."""
    return 2 / math.pi * np.arctan(np.asarray(w, float) / b)


class Lab2Page(QWidget):
    def __init__(self):
        super().__init__()
        self.s_curves: list[tuple[float, float, float]] = []
        self.q_curves: list[tuple[float, float, float]] = []
        self.e_curves: list[float] = []
        self.wc = 0.0
        self.dt0 = 0.0
        self.k_h = 2.0
        self.k_b = 1.0
        self.tabs = QTabWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs, 1)
        self.tabs.addTab(self._theory_tab(), "Общие сведения")
        self.tabs.addTab(self._s_tab(), "S(w)")
        self.tabs.addTab(self._q_tab(), "Q(w)")
        self.tabs.addTab(self._energy_tab(), "Энергия в спектре")
        self.tabs.addTab(self._kot_tab(), "Ряд Котельникова")
        self.status = QLabel("Задайте параметры импульса и нажмите «Обновить».")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#334; padding:2px 6px;")
        lay.addWidget(self.status)
        self._energy_update()
        self.s_curves.append((self.s_h.value(), self.s_b.value(), self.s_t0.value()))
        self._s_draw()
        self.q_curves.append((self.q_h.value(), self.q_b.value(), self.q_t0.value()))
        self._q_draw()
        self._say("«Обновить» добавляет кривую к нарисованным, «Очистить» стирает график вкладки.")

    # --- вкладки -------------------------------------------------------------

    def _theory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        view.setHtml(THEORY)
        lay.addWidget(view)
        return w

    def _s_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Исследование модуля спектральной плотности экспоненциального импульса"))
        self.cv_s = Canvas(3.0)
        lay.addWidget(self.cv_s, 1)
        box, spins = self._param_box((("h", 0.0, 99.99, 2.0, 0.1),
                                      ("β", 0.0, 99.99, 1.0, 0.1),
                                      ("t0", -99.99, 99.99, 0.0, 0.1)))
        self.s_h, self.s_b, self.s_t0 = spins
        lay.addWidget(box)
        self.bt_s_up, self.bt_s_clear, row = self._buttons(self._s_update, self._s_clear)
        lay.addLayout(row)
        return w

    def _q_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Исследование аргумента спектральной плотности экспоненциального импульса"))
        self.cv_q = Canvas(3.0)
        lay.addWidget(self.cv_q, 1)
        box, spins = self._param_box((("h", 0.0, 99.99, 2.0, 0.1),
                                      ("β", 0.0, 99.99, 1.0, 0.1),
                                      ("t0", -99.99, 99.99, 0.0, 0.1)))
        self.q_h, self.q_b, self.q_t0 = spins
        lay.addWidget(box)
        self.bt_q_up, self.bt_q_clear, row = self._buttons(self._q_update, self._q_clear)
        lay.addLayout(row)
        return w

    def _energy_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Исследование интегральной кривой распределения энергии"))
        self.cv_e = Canvas(3.0)
        lay.addWidget(self.cv_e, 1)
        box, spins = self._param_box((("h", 0.0, 99.99, 2.0, 0.1),
                                      ("β", 0.01, 99.99, 1.0, 0.1),
                                      ("λ", 0.01, 0.99, 0.9, 0.05)))
        self.e_h, self.e_b, self.e_lam = spins
        lay.addWidget(box)
        out = QGroupBox("Результат")
        grid = QGridLayout(out)
        self.e_wc = QLabel("ωc = —")
        self.e_dt = QLabel("Δt = —")
        self.e_e = QLabel("Э = —")
        for i, lab in enumerate((self.e_wc, self.e_dt, self.e_e)):
            grid.addWidget(lab, 0, i)
        lay.addWidget(out)
        self.bt_e_up, self.bt_e_clear, row = self._buttons(self._energy_update, self._energy_clear)
        lay.addLayout(row)
        return w

    def _kot_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Разложение экспоненциального импульса в ряд Котельникова"))
        self.cv_k = Canvas(3.0)
        lay.addWidget(self.cv_k, 1)
        box = QGroupBox("Интервал дискретизации")
        grid = QGridLayout(box)
        self.k_min = QLabel("Min = —")
        self.k_max = QLabel("Max = —")
        self.dt = self._spin(0.01, 99.99, 1.0, 0.01, 2)
        grid.addWidget(QLabel("Δt"), 0, 0)
        grid.addWidget(self.dt, 0, 1)
        grid.addWidget(self.k_min, 0, 2)
        grid.addWidget(self.k_max, 0, 3)
        grid.setColumnStretch(4, 1)
        lay.addWidget(box)
        self.bt_k_up, self.bt_k_clear, row = self._buttons(self._kot_update, self._kot_clear)
        lay.addLayout(row)
        return w

    # --- построение контролов -------------------------------------------------

    @staticmethod
    def _spin(lo: float, hi: float, val: float, step: float = 0.1, dec: int = 2) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setValue(val)
        s.setMaximumWidth(96)
        return s

    def _param_box(self, specs) -> tuple[QGroupBox, list[QDoubleSpinBox]]:
        box = QGroupBox("Параметры экспоненциального импульса")
        grid = QGridLayout(box)
        spins: list[QDoubleSpinBox] = []
        for i, (name, lo, hi, val, step) in enumerate(specs):
            grid.addWidget(QLabel(name), 0, i)
            s = self._spin(lo, hi, val, step)
            grid.addWidget(s, 1, i)
            spins.append(s)
        grid.setColumnStretch(len(specs), 1)
        return box, spins

    @staticmethod
    def _buttons(on_update, on_clear) -> tuple[QPushButton, QPushButton, QHBoxLayout]:
        row = QHBoxLayout()
        upd = QPushButton("Обновить")
        upd.clicked.connect(on_update)
        clr = QPushButton("Очистить")
        clr.clicked.connect(on_clear)
        row.addWidget(upd)
        row.addWidget(clr)
        row.addStretch(1)
        return upd, clr, row

    def _say(self, text: str, warn: bool = False):
        self.status.setText(text)
        self.status.setStyleSheet("color:#9a2b1e; padding:2px 6px;" if warn
                                  else "color:#334; padding:2px 6px;")

    # --- S(ω) ----------------------------------------------------------------

    def _s_update(self):
        hp = (self.s_h.value(), self.s_b.value(), self.s_t0.value())
        if hp not in self.s_curves:
            self.s_curves.append(hp)
        self._s_draw()
        h, b, t0 = hp
        self._say(f"S(w): h = {h:g}, β = {b:g}, t0 = {t0:g}; S(0) = "
                  + ("∞ (сигнал включения)" if b == 0 else f"{h / b:.3f}"))

    def _s_clear(self):
        self.s_curves.clear()
        self._s_draw()
        self._say("S(w): график очищен.")

    def _s_draw(self):
        self.cv_s.clear("S(ω) = h/√(β² + ω²)", "ω, рад/с", "S(ω)")
        top = 0.0
        for h, b, t0 in self.s_curves:
            w = np.linspace(0.0 if b > 0 else 0.02, W_MAX, 1200)
            y = p.S(w, h, b)
            self.cv_s.ax.plot(w, y, linewidth=1.6, label=f"h = {h:g}, β = {b:g}, t0 = {t0:g}")
            top = max(top, float(np.nanmax(np.where(w >= 0.3, y, 0.0))))
            if b > 0:
                top = max(top, float(p.S(0.0, h, b)))
        if self.s_curves:
            self.cv_s.ax.set_xlim(0, W_MAX)
            self.cv_s.ax.set_ylim(0, 1.12 * top)
            self.cv_s.ax.legend(fontsize=8, loc="upper right")
        self.cv_s.draw_idle()

    # --- Θ(ω) ----------------------------------------------------------------

    def _q_update(self):
        hp = (self.q_h.value(), self.q_b.value(), self.q_t0.value())
        if hp not in self.q_curves:
            self.q_curves.append(hp)
        self._q_draw()
        h, b, t0 = hp
        self._say(f"Q(w): h = {h:g}, β = {b:g}, t0 = {t0:g}; Θ(0) = "
                  f"{float(p.theta(0.0, h, b, t0)) if b > 0 else -math.pi / 2:+.3f} рад")

    def _q_clear(self):
        self.q_curves.clear()
        self._q_draw()
        self._say("Q(w): график очищен.")

    def _q_draw(self):
        self.cv_q.clear("Θ(ω) = −ω·t0 − arctg(ω/β)", "ω, рад/с", "Θ, рад")
        for h, b, t0 in self.q_curves:
            w = np.linspace(0.01 if b <= 0 else 0.0, W_MAX, 1200)
            self.cv_q.ax.plot(w, p.theta(w, h, b, t0), linewidth=1.6,
                              label=f"h = {h:g}, β = {b:g}, t0 = {t0:g}")
        if self.q_curves:
            self.cv_q.ax.set_xlim(0, W_MAX)
            self.cv_q.ax.legend(fontsize=8, loc="lower left")
        self.cv_q.draw_idle()

    # --- энергия -------------------------------------------------------------

    def _energy_update(self):
        h, b, lam = self.e_h.value(), self.e_b.value(), self.e_lam.value()
        wc = p.wc_of(lam, b)
        self.wc, self.dt0, self.k_h, self.k_b = wc, p.dt_of(wc), h, b
        if b not in self.e_curves:
            self.e_curves.append(b)
        self.e_wc.setText(f"ωc = {wc:.2f}")
        self.e_dt.setText(f"Δt = {self.dt0:.2f}")
        self.e_e.setText(f"Э = {p.energy(h, b):.2f}")
        self.k_min.setText(f"Min = {self.dt0 / 2:.2f}")
        self.k_max.setText(f"Max = {1.5 * self.dt0:.2f}")
        self.dt.setValue(round(self.dt0, 2))
        self.cv_e.clear(f"λ(ωc) = (2/π)·arctg(ωc/β), λ = {lam:g}", "ωc, рад/с", "λ")
        w = np.linspace(0.01, 8.0, 1200)
        for bb in self.e_curves:
            self.cv_e.ax.plot(w, _lam(w, bb), linewidth=1.6, label=f"β = {bb:g}")
        self.cv_e.ax.axhline(lam, color="#888", linestyle="--", linewidth=0.9)
        self.cv_e.ax.axvline(wc, color="#b3261e", linestyle=":", linewidth=0.9)
        self.cv_e.ax.plot([wc], [lam], "o", color="#b3261e")
        self.cv_e.ax.annotate(f"ωc = {wc:.2f}", (wc, lam), xytext=(6, -14),
                              textcoords="offset points", fontsize=9)
        self.cv_e.ax.set_xlim(0, 8)
        self.cv_e.ax.set_ylim(0, 1.02)
        self.cv_e.ax.legend(fontsize=8, loc="upper left")
        self.cv_e.draw_idle()
        self._kot_draw()
        self._say(f"Энергия: β = {b:g}, λ = {lam:g}; ωc = {wc:.2f}, Δt0 = {self.dt0:.2f}.")

    def _energy_clear(self):
        self.e_curves.clear()
        self._energy_update()
        self._say("Энергия: график очищен.")

    # --- ряд Котельникова ----------------------------------------------------

    def _kot_update(self):
        if self.wc <= 0:
            self._say("Ряд Котельникова: сначала нажмите «Обновить» на «Энергии в спектре».", warn=True)
            return
        dt = self.dt.value()
        lo, hi = p.kotelnikov_range(self.dt0)
        self._kot_draw()
        if dt < lo - 1e-9:
            self._say(f"Δt = {dt:.2f} вне [{lo:.2f}; {hi:.2f}] — меньше Δt0/2: сумма завышена, "
                      "отсчёты избыточны.", warn=True)
        elif dt > hi + 1e-9:
            self._say(f"Δt = {dt:.2f} вне [{lo:.2f}; {hi:.2f}] — больше 3Δt0/2: отсчётов не хватает, "
                      "форма теряется.", warn=True)
        else:
            self._say(f"Δt = {dt:.2f} в диапазоне [{lo:.2f}; {hi:.2f}], Δt/Δt0 = {dt / self.dt0:.2f}.")

    def _kot_clear(self):
        self.cv_k.clear("Ряд Котельникова: x(t) и сумма отсчётов", "t, с", "x(t)")
        self.cv_k.draw_idle()
        self._say("Ряд Котельникова: график очищен.")

    def _kot_draw(self):
        t = np.linspace(-1.5, 3.0, 3001)
        h, b, wc, dt = self.k_h, self.k_b, self.wc, self.dt.value()
        self.cv_k.clear(f"Ряд Котельникова: h = {h:g}, β = {b:g}, ωc = {wc:.2f}, Δt = {dt:.2f}",
                        "t, с", "x(t)")
        xs = p.x(t, h, b, 0.0)
        self.cv_k.ax.plot(t, xs, color="#0b5", linewidth=2.0, label="x(t)")
        if wc > 0 and dt > 0:
            s = p.kotelnikov(t, h, b, wc, dt)
            self.cv_k.ax.plot(t, s, color="#b3261e", linewidth=1.3, label="сумма ряда")
            k = np.arange(0, int(3.0 / dt) + 1)
            self.cv_k.ax.plot(k * dt, p.x(k * dt, h, b, 0.0), "o", ms=3, color="#1a73e8",
                              label="отсчёты")
            top = max(float(xs.max()), float(s.max()))
        else:
            top = float(xs.max())
        self.cv_k.ax.set_xlim(-1.5, 3.0)
        self.cv_k.ax.set_ylim(-0.25, 1.15 * top)
        self.cv_k.ax.legend(fontsize=8, loc="upper right")
        self.cv_k.draw_idle()
