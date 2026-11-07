from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..mathlib import spectra as sp
from ..plot import Canvas
from ..theory import apply_theory

THEORY = """<h3>Ряд Фурье в форме программы</h3>
<p>x(t) = A<sub>0</sub>/2 + Σ |A<sub>k</sub>|·cos(kω<sub>0</sub>t + Θ<sub>k</sub>), ω<sub>0</sub> = 2π/T, k = 1…10.</p>
<p>a<sub>k</sub> = (2/T)∫x·cos(kω<sub>0</sub>t)dt, b<sub>k</sub> = (2/T)∫x·sin(kω<sub>0</sub>t)dt,
|A<sub>k</sub>| = √(a<sub>k</sub>²+b<sub>k</sub>²), Θ<sub>k</sub> = atan2(−b<sub>k</sub>, a<sub>k</sub>).</p>
<p>В столбце k = 0 таблицы — A<sub>0</sub>/2 (именно её вводят в поле «A0/2»). Фазы — в радианах,
в программе поле вмещает ±9.999.</p>
<h4>Порядок выполнения</h4>
<ol><li>Меандр: гармоники и их сумма.</li><li>Постоянный уровень.</li><li>Гармонический сигнал.</li>
<li>Треугольные импульсы.</li><li>Пила.</li><li>Прямоугольные импульсы.</li><li>Произвольный спектр.</li></ol>
<p>Доп. задание: сдвиг пилы на T/4 и удвоение частоты треугольника — см. список сигналов.</p>"""

PRESETS = {
    "Меандр (h = 1)": lambda: sp.meander(1.0),
    "Постоянный уровень": lambda: sp.constant(1.0),
    "Гармонический сигнал": lambda: sp.harmonic(2.0),
    "Треугольник (h = 2)": lambda: sp.triangle(2.0),
    "Пила (h = 2)": lambda: sp.sawtooth(2.0),
    "Прямоугольные импульсы": lambda: sp.rect_pulses(2.0, 4.0),
    "Произвольный: равные амплитуды": lambda: sp.flat(0.5, 0.5),
    "Пила, сдвиг влево на T/4": lambda: sp.shifted(sp.sawtooth(1.0), -0.25),
    "Треугольник, частота ×2": lambda: sp.freq_scaled(sp.triangle(2.0), 2),
}


class Lab1Page(QWidget):
    def __init__(self):
        super().__init__()
        self.spec = sp.triangle(2.0)
        self.tabs = QTabWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        self.tabs.addTab(self._theory_tab(), "Общие сведения")
        self.tabs.addTab(self._harmonics_tab(), "Гармоники")
        self.tabs.addTab(self._signal_tab(), "Сигнал")
        self.tabs.addTab(self._input_tab(), "Ввод спектра")
        self.tabs.addTab(self._result_tab(), "Сигнал с заданным спектром")
        self._load_preset(default=True)


    def _theory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        apply_theory(view, "project1_1", THEORY)
        lay.addWidget(view)
        return w

    def _harmonics_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Число гармоник:"))
        self.n_harm = QSpinBox()
        self.n_harm.setRange(1, 10)
        self.n_harm.setValue(5)
        row.addWidget(self.n_harm)
        btn = QPushButton("Обновить")
        btn.clicked.connect(self._draw_harmonics)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.cv_harm = Canvas(3.2)
        lay.addWidget(self.cv_harm)
        return w

    def _signal_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Число гармоник:"))
        self.n_sig = QSpinBox()
        self.n_sig.setRange(1, 10)
        self.n_sig.setValue(5)
        row.addWidget(self.n_sig)
        btn = QPushButton("Обновить")
        btn.clicked.connect(self._draw_signal)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.cv_sig = Canvas(3.2)
        lay.addWidget(self.cv_sig)
        return w

    def _input_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        left = QVBoxLayout()
        presets = QComboBox()
        presets.addItems(list(PRESETS))
        presets.currentTextChanged.connect(lambda _: self._load_preset(name=presets.currentText()))
        left.addWidget(QLabel("Сигнал:"))
        left.addWidget(presets)

        box = QGroupBox("Спектр: |A_k| и Θ_k (рад)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        head = ("k", "|A_k|", "Θ_k")
        for col, text in enumerate(head):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#24405e; font-weight:600;")
            grid.addWidget(lbl, 0, col)
        self.a0 = QLineEdit("0.000")
        self.amp_edits: list[QLineEdit] = []
        self.phase_edits: list[QLineEdit] = []
        grid.addWidget(QLabel("A0/2"), 1, 0)
        self.a0.setAlignment(Qt.AlignRight)
        self.a0.setFixedWidth(84)
        grid.addWidget(self.a0, 1, 1)
        for k in range(1, 11):
            grid.addWidget(QLabel(str(k)), 1 + k, 0)
            a, q = QLineEdit("0.000"), QLineEdit("+0.000")
            for e in (a, q):
                e.setAlignment(Qt.AlignRight)
                e.setFixedWidth(84)
            self.amp_edits.append(a)
            self.phase_edits.append(q)
            grid.addWidget(a, 1 + k, 1)
            grid.addWidget(q, 1 + k, 2)
        left.addWidget(box)

        row = QHBoxLayout()
        clear = QPushButton("Обнулить")
        clear.clicked.connect(self._zero)
        apply_ = QPushButton("Обновить сигнал")
        apply_.clicked.connect(self._apply_input)
        row.addWidget(clear)
        row.addWidget(apply_)
        row.addStretch(1)
        left.addLayout(row)
        left.addStretch(1)

        self.cv_in = Canvas(3.0)
        lay.addLayout(left, 0)
        lay.addWidget(self.cv_in, 1)
        return w

    def _result_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        btn = QPushButton("Обновить")
        btn.clicked.connect(self._draw_result)
        row.addWidget(btn)
        self.max_lbl = QLabel("")
        row.addWidget(self.max_lbl)
        row.addStretch(1)
        lay.addLayout(row)
        self.cv_res = Canvas(4.2)
        lay.addWidget(self.cv_res)
        return w


    def _load_preset(self, name: str | None = None, default: bool = False):
        key = name or "Треугольник (h = 2)"
        self.spec = PRESETS[key]()
        self.a0.setText(f"{self.spec.a0_half:+.3f}")
        for a, q, ae, qe in zip(self.spec.amps, self.spec.phases, self.amp_edits, self.phase_edits):
            ae.setText(f"{a:.3f}")
            qe.setText(f"{q:+.3f}")
        self._draw_harmonics()
        self._draw_signal()
        self._draw_result()
        self._draw_input()

    def _read_input(self) -> sp.Spectrum:
        def num(text: str, default: float = 0.0) -> float:
            try:
                return float(text.replace(",", ".").replace("+", "") or default)
            except ValueError:
                return default
        return sp.Spectrum("custom", "Спектр из полей", "", num(self.a0.text()),
                           [num(e.text()) for e in self.amp_edits],
                           [num(e.text()) for e in self.phase_edits])

    def _apply_input(self):
        self.spec = self._read_input()
        self._draw_result()
        self._draw_input()

    def _zero(self):
        self.a0.setText("0.000")
        for e in self.amp_edits:
            e.setText("0.000")
        for e in self.phase_edits:
            e.setText("+0.000")

    def _draw_harmonics(self):
        self.cv_harm.harmonics(self.spec, self.n_harm.value(), f"{self.spec.title}: гармоники")

    def _draw_signal(self):
        self.cv_sig.signal(self.spec, self.n_sig.value(), f"{self.spec.title}: сумма {self.n_sig.value()} гармоник")

    def _draw_input(self):
        self.cv_in.bars(self.spec.amps, self.spec.phases, "Ввод спектра: |A_k| и Θ_k")

    def _draw_result(self):
        self.cv_res.signal(self.spec, None, f"{self.spec.title}: сигнал по заданному спектру")
        self.max_lbl.setText(f"максимум на −0.75T…0.75T: {sp.window_max(self.spec):.3f}")
