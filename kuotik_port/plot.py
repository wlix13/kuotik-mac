"""Matplotlib canvases for lab pages: signal, harmonic set, spectrum table."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Canvas(FigureCanvasQTAgg):
    def __init__(self, height: float = 3.4, dpi: int = 100):
        self.fig = Figure(figsize=(7.4, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style()

    def _style(self):
        self.ax.grid(True, color="#d9d9d9", linewidth=0.6)
        self.ax.axhline(0, color="#666", linewidth=0.8)
        self.ax.axvline(0, color="#666", linewidth=0.8)
        for s in ("top", "right"):
            self.ax.spines[s].set_visible(False)

    def clear(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        self.ax.clear()
        self._style()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def signal(self, spec, n: int | None = None, title: str = ""):
        """Сумма первых n гармоник (по умолчанию всех 10) на интервале графика программы."""
        t = np.linspace(-0.75, 0.75, 2001)
        n = n or len(spec.amps)
        part = spec.__class__(spec.key, spec.title, spec.formula, spec.a0_half,
                              spec.amps[:n], spec.phases[:n])
        y = part.synth(t)
        self.clear(title or spec.title, "t/T", "x(t)")
        self.ax.plot(t, y, color="#0b5", linewidth=1.6)
        self.ax.set_xlim(-0.75, 0.75)
        self.ax.annotate(f"max {y.max():.3f}", xy=(t[y.argmax()], y.max()),
                         xytext=(4, 6), textcoords="offset points", fontsize=9)
        self.draw_idle()

    def harmonics(self, spec, n: int, title: str = ""):
        """Каждая гармоника отдельно, как на вкладке «Гармоники»."""
        t = np.linspace(-0.75, 0.75, 2001)
        self.clear(title or "Гармоники", "t/T", "x(t)")
        colors = ["#b3261e", "#1a73e8", "#188038", "#8e24aa", "#f9ab00", "#00acc1", "#6d4c41",
                  "#c2185b", "#3949ab", "#7cb342"]
        for k in range(n):
            a, q = spec.amps[k], spec.phases[k]
            if a <= 1e-12:
                continue
            self.ax.plot(t, a * np.cos(2 * np.pi * (k + 1) * t + q), linewidth=1.2,
                         color=colors[k % len(colors)], label=f"k={k + 1}")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="upper right")
        self.draw_idle()

    def bars(self, amps, phases, title: str = "Спектр"):
        """Столбцы |A_k| с подписями фаз — то, что вводится во «Ввод спектра»."""
        self.clear(title, "k", "|A_k|")
        k = np.arange(1, len(amps) + 1)
        self.ax.bar(k, amps, color="#1a73e8", width=0.6)
        self.ax.set_xticks(k)
        top = max(amps) if len(amps) and max(amps) > 0 else 1.0
        self.ax.set_ylim(0, top * 1.22)
        for kk, a, q in zip(k, amps, phases):
            if a > 1e-9:
                self.ax.annotate(f"{q:+.3f}", (kk, a), textcoords="offset points", xytext=(0, 4),
                                 ha="center", fontsize=7)
        self.draw_idle()
