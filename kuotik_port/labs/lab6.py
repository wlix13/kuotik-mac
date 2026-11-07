from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..plot import Canvas
from ..theory import apply_theory

TABS = (
    "Общие сведения",
    "ДК",
    "ДСКС",
    "ДСКС реализующий заданный циклический код",
    "С = F [p1,p2]",
    "C = F [p,q]",
)
FIELD_MASK = "0.00000"
FIVE = 5
EPS = 1e-10
SURFACE_N = 15
CODES = {"КОД (7,4)": (7, 4), "КОД (15,11)": (15, 11), "КОД (31,26)": (31, 26)}
DEFAULT_CODE = "КОД (7,4)"
P_OSH = 0.05

MSG_P_GT1 = "Условные вероятности p1 и p2 не могут быть больше 1. Повторите ввод."
MSG_PQ_GT1 = "Сумма p и q не может быть больше 1. Повторите ввод."
MSG_BAD = "Введенное значение неверно"
MSG_UNDEF = "Неопределенность"

THEORY = """<h3>Информационная пропускная способность двоичного канала</h3>
<p>Канал преобразует входные события x ∈ X в выходные y ∈ Y по условному распределению
p(y/x). Канал стационарен и без памяти, если p(y/x) одинаково для всех последовательных
событий. Пропускная способность — максимум среднего количества информации на символ:</p>
<p style="margin-left:18px">C = max<sub>{p(x)}</sub> H(X;Y) = max<sub>{p(x)}</sub> [H(Y) − H(Y/X)],&nbsp;&nbsp;
H(Y) = −Σ p(y)·log₂p(y),&nbsp;&nbsp; H(Y/X) = −Σ p(x)·Σ p(y/x)·log₂p(y/x).</p>
<p>Величина H(X;Y) — взаимная энтропия; ненадежность передачи H(X/Y) = H(X) − H(X;Y).
По теореме Шеннона при скорости источника R &lt; C существует кодирование со сколь угодно
малой частотой ошибок, при R &gt; C потери в канале не могут быть меньше R − C.</p>

<h4>Двоичный канал (ДК) без памяти</h4>
<p>Матрица переходных вероятностей и граф:</p>
<p style="margin-left:18px">p(y/x) = [[1−p1, p1], [p2, 1−p2]],&nbsp;&nbsp;
p(0/0) = 1−p1 = p<sub>пп</sub>,&nbsp; p(1/0) = p1 = p<sub>нп</sub>,&nbsp;
p(1/1) = 1−p2 = p<sub>пп</sub>,&nbsp; p(0/1) = p2 = p<sub>нп</sub>.</p>
<p>При z = P(x1), a = s = 1−p1−p2:</p>
<p style="margin-left:18px">P(y1) = a·z + p2,&nbsp;&nbsp;
H(Y) = −(az+p2)·log₂(az+p2) − (1−p1−az)·log₂(1−p1−az),<br>
H(Y/X) = −z(1−p1)·log₂(1−p1) − z·p1·log₂p1 − (1−z)p2·log₂p2 − (1−z)(1−p2)·log₂(1−p2).</p>
<p>Экстремум ∂H(X;Y)/∂z = 0 даёт промежуточные величины (логарифмы натуральные):</p>
<p style="margin-left:18px">B = p1·ln p1 + (1−p1)·ln(1−p1) − p2·ln p2 − (1−p2)·ln(1−p2),<br>
D = 1 + exp(−B/a),&nbsp;&nbsp; z₀ = (1 − p2·D)/(a·D),&nbsp;&nbsp;
Z = P(x1) = z₀,&nbsp;&nbsp; C = H(X;Y)|<sub>z=z₀</sub>.</p>
<p>Двоичный симметричный канал (ДСК) — частный случай p1 = p2 = p:</p>
<p style="margin-left:18px">C = 1 + p·log₂p + (1−p)·log₂(1−p) = 1 − H(p).</p>

<h4>Двоичный симметричный канал со стиранием (ДСКС)</h4>
<p style="margin-left:18px">p(y/x) = [[1−p−q, p, q], [p, 1−p−q, q]],<br>
p(0/0) = p(1/1) = 1−p−q = p<sub>пп</sub> — правильный приём,<br>
p(0/1) = p(1/0) = p = p<sub>но</sub> — необнаруженная ошибка,<br>
p(c/0) = p(c/1) = q = p<sub>оо</sub> — обнаруженная ошибка (стирание).</p>
<p>Канал симметричен по входу, оптимальное распределение равномерно (z₀ = 0.5), и</p>
<p style="margin-left:18px">C = max H(Y) + Σ p<sub>j</sub>·log₂p<sub>j</sub> =
(1−q)·[1 − log₂(1−q)] + (1−p−q)·log₂(1−p−q) + p·log₂p.</p>
<p>Из условий p<sub>но</sub> + p<sub>оо</sub> + p<sub>пп</sub> = 1 и известных длины кода N и
числа информационных разрядов k находят вероятности для циклического кода (N, k):</p>
<p style="margin-left:18px">p<sub>но</sub> = 2<sup>−(N−k)</sup>·p<sub>ош</sub>,&nbsp;&nbsp;
p<sub>оо</sub> = [1 − 2<sup>−(N−k)</sup>]·p<sub>ош</sub>,&nbsp;&nbsp;
p<sub>пп</sub> = 1 − p<sub>но</sub> − p<sub>оо</sub>.</p>

<h4>Пример из справки: табл. 6.1 (p ош = 0.05)</h4>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>(N,k)</th><th>2<sup>−(N−k)</sup></th><th>1 − 2<sup>−(N−k)</sup></th>
<th>p но</th><th>p оо</th><th>C ДСКС</th></tr>
<tr><td>(7,4)</td><td>0.125</td><td>0.875</td><td>0.00625</td><td>0.04375</td><td>0.90190</td></tr>
<tr><td>(15,11)</td><td>0.0625</td><td>0.9375</td><td>0.00313</td><td>0.04688</td><td>0.92283</td></tr>
<tr><td>(31,26)</td><td>0.03125</td><td>0.96875</td><td>0.00156</td><td>0.04844</td><td>0.93486</td></tr>
</table>
<p>Если в те же формулы подставить <i>p</i> = p ош и <i>q</i> = 2<sup>−(N−k)</sup> (как в столбцах
таблицы), получаются 0.59850, 0.65588 и 0.68470. Столбец C справки (0.94532, 0.99954,
0.99999) выше 1 − q и ни одной из формул не воспроизводится — числа взяты из другого
расчёта; ниже считаются значения по формулам 6.1–6.3.</p>

<h4>Порядок выполнения</h4>
<ol>
<li>Выбрать условные вероятности p1, p2 и рассчитать C ДК; проверить расчёт на вкладке «ДК».</li>
<li>Выбрать p, q и рассчитать C ДСКС на вкладке «ДСКС».</li>
<li>Построить поверхности C = F[p1,p2] и C = F[p,q] (кнопки «Грaфика», вкладки 5 и 6).</li>
<li>Вычислить C ДСКС для циклического кода (N, k) с обнаруживающей способностью.</li>
<li>Вычислить C того же кода с обнаруживающей и корректирующей способностью.</li>
<li>Сопоставить C ДСКС с контролем и C ДК без контроля при тех же данных.</li>
</ol>
<p>Проверка поверхности ускоряется на боковых гранях «куба» p1 + p2 = 1 (p + q = 1):
там s = 1 − p1 − p2 обращается в нуль и C = 0.</p>"""


def bin_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def nat_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return MSG_UNDEF
    return f"{value:.5f}"


def parse_field(text: str) -> float | None:
    s = (text or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


class DKResult:

    def __init__(self, p1: float, p2: float):
        self.p1, self.p2 = p1, p2
        self.s = 1.0 - p1 - p2
        e1, e2 = (p1, p2) if self.s > 0.0 else (1.0 - p1, 1.0 - p2)
        self.e1, self.e2 = e1, e2
        self.es = 1.0 - e1 - e2
        if self.es <= EPS:
            self.b = self.d = 0.0
            self.zo = 0.5
            self.py1 = 0.5
            self.c = 0.0
            self.defined = self.s > EPS
            return
        self.defined = self.s > EPS
        self.b = nat_entropy(e2) - nat_entropy(e1)
        self.d = 1.0 + math.exp(-self.b / self.es)
        self.zo = (1.0 - e2 * self.d) / (self.es * self.d)
        self.zo = min(max(self.zo, 0.0), 1.0)
        self.py1 = e2 + self.es * self.zo
        self.c = max(0.0, min(1.0, self.hy() - self.hyx()))

    def hy(self) -> float:
        return bin_entropy(self.py1)

    def hyx(self) -> float:
        return self.zo * bin_entropy(self.e1) + (1.0 - self.zo) * bin_entropy(self.e2)

    def hx(self) -> float:
        return bin_entropy(self.zo)

    @property
    def info(self) -> float:
        return self.c

    @property
    def redundancy(self) -> float:
        return max(0.0, 1.0 - self.c)

    def __repr__(self) -> str:
        return (f"DKResult(p1={self.p1}, p2={self.p2}, s={self.s:.6f}, "
                f"B={self.b:.6f}, D={self.d:.6f}, Zo={self.zo:.6f}, C={self.c:.6f})")


class DSKSResult:

    def __init__(self, p: float, q: float):
        self.p, self.q = p, q
        self.s = 1.0 - p - q
        self.zo = 0.5
        self.defined = self.s > EPS
        self.c = dsks_capacity(p, q) if self.defined else 0.0

    def hx(self) -> float:
        return 1.0

    def hy(self) -> float:
        if self.q >= 1.0:
            return 0.0
        half = (1.0 - self.q) / 2.0
        return -2.0 * half * math.log2(half) - self.q * math.log2(self.q)

    def hyx(self) -> float:
        total = 0.0
        for value in (self.s, self.p, self.q):
            if value > 0.0:
                total -= value * math.log2(value)
        return total

    @property
    def info(self) -> float:
        return self.c

    @property
    def redundancy(self) -> float:
        return max(0.0, 1.0 - self.c)

    def __repr__(self) -> str:
        return f"DSKSResult(p={self.p}, q={self.q}, s={self.s:.6f}, C={self.c:.6f})"


def dk_capacity(p1: float, p2: float) -> float:
    return DKResult(p1, p2).c


def dk_capacity_legacy(p1: float, p2: float) -> float:
    r = DKResult(p1, p2)
    if not r.defined:
        return 0.0
    hyx = r.zo * bin_entropy(r.e2) + (1.0 - r.zo) * bin_entropy(r.e1)
    return max(0.0, min(1.0, r.hy() - hyx))


def dsks_capacity(p: float, q: float) -> float:
    s = 1.0 - p - q
    if s <= EPS:
        return 0.0
    if q >= 1.0:
        return 0.0
    one_q = 1.0 - q
    value = (one_q * (1.0 - math.log2(one_q)) if one_q > 0 else 0.0)
    value += s * math.log2(s) + (p * math.log2(p) if p > 0 else 0.0)
    return max(0.0, min(1.0, value))


def mutual_info(p1: float, p2: float, z: float) -> float:
    z = min(max(z, 0.0), 1.0)
    py1 = z * (1.0 - p1) + (1.0 - z) * p2
    return bin_entropy(py1) - (z * bin_entropy(p1) + (1.0 - z) * bin_entropy(p2))


def best_z(p1: float, p2: float, steps: int = 20001) -> tuple[float, float]:
    best = (-1.0, 0.5)
    for i in range(1, steps):
        z = i / steps
        value = mutual_info(p1, p2, z)
        if value > best[0]:
            best = (value, z)
    return best


def code_probs(n: int, k: int, p_osh: float = P_OSH) -> tuple[float, float, float]:
    share = 2.0 ** (-(n - k))
    p_no = share * p_osh
    p_oo = (1.0 - share) * p_osh
    return p_no, p_oo, 1.0 - p_no - p_oo


def surface(which: str, n: int = SURFACE_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(axis, axis)
    func = dk_capacity if which == "dk" else dsks_capacity
    flat = [func(float(x), float(y)) if x + y < 1.0 - EPS else np.nan
            for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    return grid_x, grid_y, np.array(flat).reshape(grid_x.shape)


def surface_rows(which: str) -> list[tuple[float, float, float]]:
    xs = np.linspace(0.0, 0.99, SURFACE_N)
    ys = np.linspace(0.0, 0.99, SURFACE_N + 2)
    func = dk_capacity if which == "dk" else dsks_capacity
    rows: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            ok = x + y < 1.0 - EPS
            rows.append((float(x), float(y), func(float(x), float(y)) if ok else 0.0))
    return rows


def _mono(text: str = "", size: int = 12) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-family:Menlo,Consolas,monospace; font-size:{size}px;")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


class NumField(QLineEdit):

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.setInputMask(FIELD_MASK)
        self.setText(f"{value:.5f}")
        self.setAlignment(Qt.AlignRight)
        self.setFixedWidth(96)
        self.setStyleSheet("font-family:Menlo,Consolas,monospace; font-size:14px; padding:2px;")

    def value(self) -> float | None:
        return parse_field(self.text())

    def set_value(self, value: float) -> None:
        self.setText(f"{value:.5f}")


class Title(QLabel):
    def __init__(self, text: str, size: int = 14):
        super().__init__(text)
        self.setStyleSheet(f"font-weight:bold; font-size:{size}px; padding:2px;")


class ResultRow(QWidget):

    def __init__(self, caption: str, width: int = 0):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.caption = QLabel(caption)
        self.caption.setStyleSheet("font-size:13px;")
        if width:
            self.caption.setFixedWidth(width)
        self.value = _mono("—", 13)
        lay.addWidget(self.caption)
        lay.addWidget(self.value)
        lay.addStretch(1)

    def set(self, value: float | str) -> None:
        self.value.setText(value if isinstance(value, str) else fmt(value))

    def clear(self) -> None:
        self.value.setText("—")


class DKPanel(QWidget):

    def __init__(self, page: "Lab6Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(Title("Двоичный канал связи (ДК)", 15))

        grid = QGridLayout()
        grid.addWidget(QLabel("p1 ="), 0, 0, Qt.AlignRight)
        self.f_p1 = NumField(0.0)
        grid.addWidget(self.f_p1, 0, 1)
        grid.addWidget(QLabel("p2 ="), 0, 2, Qt.AlignRight)
        self.f_p2 = NumField(0.0)
        grid.addWidget(self.f_p2, 0, 3)
        grid.setColumnStretch(4, 1)
        lay.addLayout(grid)

        lay.addWidget(Title("Промежуточные результаты :", 13))
        self.r_s = ResultRow("s = 1 − p1 − p2 =", 200)
        self.r_b = ResultRow("B = p1·ln p1 + (1−p1)·ln(1−p1) − p2·ln p2 − (1−p2)·ln(1−p2) =", 430)
        self.r_d = ResultRow("D = 1 + exp(−B/s) =", 200)
        self.r_zo = ResultRow("Zo = (1 − p2·D) / (s·D) =", 230)
        for row in (self.r_s, self.r_b, self.r_d, self.r_zo):
            lay.addWidget(row)

        lay.addWidget(Title("Информационная пропускная способность ДК :", 13))
        self.r_c = ResultRow("C = H(X;Y) =", 160)
        self.r_z = ResultRow("Z = P(x1) = Zo =", 160)
        lay.addWidget(self.r_c)
        lay.addWidget(self.r_z)

        lay.addWidget(Title("Информационные характеристики :", 13))
        self.r_hx = ResultRow("H(X) = H(Zo) =", 160)
        self.r_hy = ResultRow("H(Y) =", 160)
        self.r_hyx = ResultRow("H(Y|X) = Zo·H(p1) + (1−Zo)·H(p2) =", 320)
        self.r_i = ResultRow("I(X;Y) = H(Y) − H(Y|X) =", 230)
        self.r_r = ResultRow("Избыточность 1 − C =", 190)
        for row in (self.r_hx, self.r_hy, self.r_hyx, self.r_i, self.r_r):
            lay.addWidget(row)
        lay.addStretch(1)

        btns = QHBoxLayout()
        self.btn_calc = QPushButton("Вычислить")
        self.btn_graph = QPushButton("Грaфика")
        self.btn_surface = QPushButton("С = F [p1,p2]")
        for b in (self.btn_calc, self.btn_graph, self.btn_surface):
            b.setMinimumHeight(32)
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.lbl_check = QLabel("Проконтролировать поверхность:")
        self.lbl_check.setStyleSheet("font-size:13px; color:#333;")
        lay.addWidget(self.lbl_check)

        self.btn_calc.clicked.connect(self.compute)
        self.btn_graph.clicked.connect(lambda: page.show_graph("dk"))
        self.btn_surface.clicked.connect(lambda: page.tabs.setCurrentIndex(4))
        self.f_p1.editingFinished.connect(self.compute)
        self.f_p2.editingFinished.connect(self.compute)

    def compute(self) -> float | None:
        p1, p2 = self.f_p1.value(), self.f_p2.value()
        if p1 is None or p2 is None:
            self.clear()
            self.page.set_status(MSG_BAD, ok=False)
            return None
        if p1 > 1.0 or p2 > 1.0:
            self.clear()
            self.page.set_status(MSG_P_GT1, ok=False)
            return None
        if 1.0 - p1 - p2 <= EPS:
            for row in (self.r_s, self.r_b, self.r_d, self.r_zo, self.r_c, self.r_z,
                        self.r_hx, self.r_hy, self.r_hyx, self.r_i, self.r_r):
                row.set(MSG_UNDEF)
            self.page.set_status(MSG_PQ_GT1, ok=False)
            return None
        res = DKResult(p1, p2)
        self.show_result(res)
        self.page.set_status(
            f"ДК: p1 = {p1:.5f}, p2 = {p2:.5f}; s = {res.s:.5f}, B = {res.b:.5f}, "
            f"D = {res.d:.5f}, Zo = {res.zo:.5f}; C = {res.c:.5f} бит/символ", ok=True)
        return res.c

    def show_result(self, res: DKResult) -> None:
        self.r_s.set(res.s)
        self.r_b.set(res.b)
        self.r_d.set(res.d)
        self.r_zo.set(res.zo)
        self.r_c.set(res.c)
        self.r_z.set(res.zo)
        self.r_hx.set(res.hx())
        self.r_hy.set(res.hy())
        self.r_hyx.set(res.hyx())
        self.r_i.set(res.info)
        self.r_r.set(res.redundancy)

    def clear(self) -> None:
        for row in (self.r_s, self.r_b, self.r_d, self.r_zo, self.r_c, self.r_z,
                    self.r_hx, self.r_hy, self.r_hyx, self.r_i, self.r_r):
            row.clear()


class DSKSPanel(QWidget):

    def __init__(self, page: "Lab6Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(Title("Двоичный симметричный канал связи со стиранием (ДСКС)", 15))

        grid = QGridLayout()
        grid.addWidget(QLabel("p ="), 0, 0, Qt.AlignRight)
        self.f_p = NumField(0.0)
        grid.addWidget(self.f_p, 0, 1)
        grid.addWidget(QLabel("q ="), 0, 2, Qt.AlignRight)
        self.f_q = NumField(0.0)
        grid.addWidget(self.f_q, 0, 3)
        grid.setColumnStretch(4, 1)
        lay.addLayout(grid)

        lay.addWidget(Title("Промежуточные результаты :", 13))
        self.r_s = ResultRow("s = 1 − p − q =", 200)
        self.r_zo = ResultRow("Zo = 0.5 (канал симметричен по входу) =", 300)
        lay.addWidget(self.r_s)
        lay.addWidget(self.r_zo)
        lay.addWidget(Title("Информационная пропускная способность ДCКC :", 13))
        self.r_c = ResultRow("C = H(X;Y) =", 160)
        self.r_z = ResultRow("Z = P(x1) = Zo =", 160)
        lay.addWidget(self.r_c)
        lay.addWidget(self.r_z)

        lay.addWidget(Title("Информационные характеристики :", 13))
        self.r_hx = ResultRow("H(X) =", 160)
        self.r_hy = ResultRow("H(Y) =", 160)
        self.r_hyx = ResultRow("H(Y|X) = H(1−p−q, p, q) =", 250)
        self.r_i = ResultRow("I(X;Y) = H(Y) − H(Y|X) =", 230)
        self.r_r = ResultRow("Избыточность 1 − C =", 190)
        for row in (self.r_hx, self.r_hy, self.r_hyx, self.r_i, self.r_r):
            lay.addWidget(row)
        lay.addStretch(1)

        btns = QHBoxLayout()
        self.btn_calc = QPushButton("Вычислить")
        self.btn_graph = QPushButton("Грaфика")
        self.btn_surface = QPushButton("C = F [p,q]")
        for b in (self.btn_calc, self.btn_graph, self.btn_surface):
            b.setMinimumHeight(32)
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.lbl_check = QLabel("Проконтролировать поверхность :")
        self.lbl_check.setStyleSheet("font-size:13px; color:#333;")
        lay.addWidget(self.lbl_check)

        self.btn_calc.clicked.connect(self.compute)
        self.btn_graph.clicked.connect(lambda: page.show_graph("dsks"))
        self.btn_surface.clicked.connect(lambda: page.tabs.setCurrentIndex(5))
        self.f_p.editingFinished.connect(self.compute)
        self.f_q.editingFinished.connect(self.compute)

    def compute(self) -> float | None:
        p, q = self.f_p.value(), self.f_q.value()
        if p is None or q is None:
            self.clear()
            self.page.set_status(MSG_BAD, ok=False)
            return None
        if p > 1.0 or q > 1.0:
            self.clear()
            self.page.set_status(MSG_PQ_GT1, ok=False)
            return None
        if 1.0 - p - q <= EPS:
            for row in (self.r_s, self.r_zo, self.r_c, self.r_z,
                        self.r_hx, self.r_hy, self.r_hyx, self.r_i, self.r_r):
                row.set(MSG_UNDEF)
            self.page.set_status(MSG_PQ_GT1, ok=False)
            return None
        res = DSKSResult(p, q)
        self.show_result(res)
        self.page.set_status(
            f"ДСКС: p = {res.p:.5f}, q = {res.q:.5f}; s = {res.s:.5f}, Zo = 0.5; "
            f"C = {res.c:.5f} бит/символ", ok=True)
        return res.c

    def show_result(self, res: DSKSResult) -> None:
        self.r_s.set(res.s)
        self.r_zo.set(res.zo)
        self.r_c.set(res.c)
        self.r_z.set(res.zo)
        self.r_hx.set(res.hx())
        self.r_hy.set(res.hy())
        self.r_hyx.set(res.hyx())
        self.r_i.set(res.info)
        self.r_r.set(res.redundancy)

    def clear(self) -> None:
        for row in (self.r_s, self.r_zo, self.r_c, self.r_z,
                    self.r_hx, self.r_hy, self.r_hyx, self.r_i, self.r_r):
            row.clear()


class TaskPanel(QWidget):

    TITLES = {
        1: ("Циклический код реализует только обнаруживающую способность :",
            ("p =", "q =")),
        2: ("Циклический код реализует обнаруживающую и корректирующую способность :",
            ("p =", "q =")),
        3: ("Сопоставьте информационную пропускную способность ДСКС с контролем и\n"
            "информационную пропускную способность ДК без контроля при тех же данных",
            ("p1 =", "p2 =")),
    }

    def __init__(self, page: "Lab6Page", mode: int):
        super().__init__()
        self.page = page
        self.mode = mode
        caption, names = self.TITLES[mode]
        lay = QVBoxLayout(self)
        head = QLabel(caption)
        head.setWordWrap(True)
        head.setStyleSheet("font-size:13px;")
        lay.addWidget(head)

        grid = QGridLayout()
        grid.addWidget(QLabel(names[0]), 0, 0, Qt.AlignRight)
        self.f_a = NumField(0.0)
        grid.addWidget(self.f_a, 0, 1)
        grid.addWidget(QLabel(names[1]), 1, 0, Qt.AlignRight)
        self.f_b = NumField(0.0)
        grid.addWidget(self.f_b, 1, 1)
        grid.addWidget(QLabel("C ="), 0, 2, Qt.AlignRight)
        self.l_c = _mono("—", 14)
        self.l_c.setMinimumWidth(90)
        grid.addWidget(self.l_c, 0, 3, 2, 1)
        self.btn_calc = QPushButton("Вычислить")
        self.btn_calc.setMinimumHeight(32)
        grid.addWidget(self.btn_calc, 0, 4, 2, 1)
        grid.setColumnStretch(5, 1)
        lay.addLayout(grid)
        lay.addStretch(1)

        self.btn_calc.clicked.connect(self.compute)

    def params(self) -> tuple[float, float] | None:
        a, b = self.f_a.value(), self.f_b.value()
        if a is None or b is None:
            self.page.set_status(MSG_BAD, ok=False)
            return None
        if self.mode == 3:
            if a > 1.0 or b > 1.0:
                self.page.set_status(MSG_P_GT1, ok=False)
                return None
            if 1.0 - a - b <= EPS:
                self.page.set_status(MSG_PQ_GT1, ok=False)
                return None
        elif a > 1.0 or b > 1.0 or 1.0 - a - b <= EPS:
            self.page.set_status(MSG_PQ_GT1, ok=False)
            return None
        return a, b

    def value(self) -> float | None:
        a, b = self.f_a.value(), self.f_b.value()
        if a is None or b is None:
            return None
        if self.mode == 3:
            return dk_capacity(a, b)
        return dsks_capacity(a, b)

    def compute(self) -> float | None:
        params = self.params()
        if params is None:
            self.l_c.setText("—")
            return None
        value = self.value()
        if value is None:
            self.l_c.setText("—")
            self.page.set_status(MSG_BAD, ok=False)
            return None
        self.l_c.setText(fmt(value))
        self.page.task_solved(self.mode, value)
        return value

    def fill_from_code(self, n: int, k: int, p_osh: float = P_OSH) -> None:
        p_no, p_oo, _ = code_probs(n, k, p_osh)
        if self.mode == 3:
            self.f_a.set_value(p_no)
            self.f_b.set_value(p_oo)
        else:
            self.f_a.set_value(p_no)
            self.f_b.set_value(p_oo)


class CodePanel(QWidget):

    def __init__(self, page: "Lab6Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(Title("ДСКС при реализации заданного циклического кода", 15))

        head = QHBoxLayout()
        self.cb_code = QComboBox()
        self.cb_code.addItems(list(CODES))
        self.lbl_code = Title("КОД (N,k)", 13)
        self.btn_fill = QPushButton("Заполнить p и q по коду")
        head.addWidget(self.cb_code)
        head.addWidget(self.lbl_code)
        head.addWidget(self.btn_fill)
        head.addStretch(1)
        lay.addLayout(head)

        self.stack = QStackedWidget()
        self.tasks = {m: TaskPanel(page, m) for m in (1, 2, 3)}
        for m in (1, 2, 3):
            self.stack.addWidget(self.tasks[m])
        lay.addWidget(self.stack, 1)

        self.result_panel = QWidget()
        rl = QVBoxLayout(self.result_panel)
        self.lbl_formula = QLabel("")
        self.lbl_formula.setStyleSheet("font-family:Menlo,Consolas,monospace; font-size:13px;")
        self.lbl_solved = QLabel("Задача решена верно")
        self.lbl_solved.setStyleSheet("font-size:14px; font-weight:bold; color:#137333;")
        self.btn_next = QPushButton("Продолжить")
        self.btn_next.setMinimumHeight(32)
        rl.addWidget(self.lbl_formula)
        rl.addWidget(self.lbl_solved)
        rl.addWidget(self.btn_next, 0, Qt.AlignRight)
        rl.addStretch(1)
        self.result_panel.setVisible(False)
        lay.addWidget(self.result_panel)

        self.cb_code.currentTextChanged.connect(self.on_code)
        self.btn_fill.clicked.connect(self.fill)
        self.btn_next.clicked.connect(self.next_task)
        self.on_code(DEFAULT_CODE)

    def on_code(self, name: str) -> None:
        n, k = CODES.get(name, CODES[DEFAULT_CODE])
        self.lbl_code.setText(name)
        self.page.set_status(f"Код {name}: N = {n}, k = {k}, r = N − k = {n - k}, "
                             f"2^−(N−k) = {2.0 ** (-(n - k)):.5f}", ok=True)

    def fill(self) -> None:
        n, k = CODES.get(self.cb_code.currentText(), CODES[DEFAULT_CODE])
        p_no, p_oo, p_pp = code_probs(n, k)
        for task in self.tasks.values():
            task.fill_from_code(n, k)
        self.page.set_status(
            f"{self.cb_code.currentText()}, p ош = {P_OSH:.2f}: p но = {p_no:.5f}, "
            f"p оо = {p_oo:.5f}, p пп = {p_pp:.5f}", ok=True)

    def task(self, mode: int) -> TaskPanel:
        return self.tasks[mode]

    def next_task(self) -> None:
        index = self.stack.currentIndex()
        self.result_panel.setVisible(False)
        self.stack.setVisible(True)
        if index >= 2:
            self.stack.setCurrentIndex(0)
            self.page.set_status("Все три задачи выполнены; вкладка возвращена к задаче 1",
                                 ok=True)
        else:
            self.stack.setCurrentIndex(index + 1)
            self.page.set_status(f"Задача {index + 2} из 3", ok=True)


class SurfaceTab(QWidget):

    def __init__(self, which: str, caption: str):
        super().__init__()
        self.which = which
        lay = QVBoxLayout(self)
        lay.addWidget(Title(caption, 14))
        self.canvas = Canvas(3.6)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas, 1)
        self.draw()

    def draw(self) -> None:
        _plot_surface(self.canvas, self.which)


def _plot_surface(canvas: Canvas, which: str) -> object:
    grid_x, grid_y, grid_z = surface(which)
    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111, projection="3d")
    canvas.ax = ax
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", rstride=1, cstride=1,
                           linewidth=0.15, antialiased=True, alpha=0.92)
    ax.set_xlabel("p1" if which == "dk" else "p", fontsize=8)
    ax.set_ylabel("p2" if which == "dk" else "q", fontsize=8)
    ax.set_zlabel("C, бит/символ", fontsize=8)
    ax.set_zlim(0.0, 1.0)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=24, azim=-128)
    ax.text(0.0, 0.0, 1.05, "C = 1 при " + ("p1 = p2 = 0" if which == "dk" else "p = q = 0"),
            fontsize=7, color="#333")
    canvas.fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.09)
    canvas.draw_idle()
    return surf


class GraphPanel(QWidget):

    def __init__(self, page: "Lab6Page"):
        super().__init__()
        self.page = page
        self.which = "dk"
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.lbl_channel = Title("Канал", 15)
        self.lbl_name = QLabel("ДК")
        self.lbl_name.setStyleSheet("font-size:14px; font-weight:bold;")
        top.addWidget(self.lbl_channel)
        top.addWidget(self.lbl_name)
        top.addStretch(1)
        lay.addLayout(top)

        body = QHBoxLayout()
        left = QVBoxLayout()
        self.canvas = Canvas(4.2)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.canvas, 1)
        self.axis_row = QHBoxLayout()
        self.captions = [QLabel("Ось 1"), QLabel("Ось 2"), QLabel("C")]
        for i, lbl in enumerate(self.captions):
            lbl.setStyleSheet("font-size:11px; color:#444;")
            self.axis_row.addWidget(lbl)
            sep = QLabel("·")
            sep.setStyleSheet("font-size:11px; color:#999;")
            self.axis_row.addWidget(sep)
        self.lbl_zero = QLabel("0")
        self.lbl_one = QLabel("1")
        for lbl in (self.lbl_zero, self.lbl_one):
            lbl.setStyleSheet("font-size:10px; color:#777;")
            self.axis_row.addWidget(lbl)
        self.axis_row.addStretch(1)
        left.addLayout(self.axis_row)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_reset = QPushButton("Сброс")
        self.btn_exit = QPushButton("Выxод")
        for b in (self.btn_reset, self.btn_exit):
            b.setMinimumHeight(32)
            row.addWidget(b)
        right.addLayout(row)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)

        self.btn_reset.clicked.connect(self.reset)
        self.btn_exit.clicked.connect(page.close_graph)

    def load(self, which: str) -> None:
        self.which = which
        self.lbl_name.setText("ДК" if which == "dk" else "ДCКC")
        _plot_surface(self.canvas, which)
        self.fill_table(which)

    def fill_table(self, which: str) -> None:
        names = ("p1", "p2", "C") if which == "dk" else ("p", "q", "C")
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["№", *names])
        rows = surface_rows(which)
        self.table.setRowCount(len(rows))
        for r, (x, y, c) in enumerate(rows):
            cells = [str(r + 1), f"{x:.4f}", f"{y:.4f}", f"{c:.5f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, col, item)
        self.table.setHorizontalHeaderLabels(["№", *names])
        self.table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignCenter)
        for col in range(1, 4):
            self.table.horizontalHeaderItem(col).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def reset(self) -> None:
        self.table.setRowCount(0)
        self.canvas.fig.clear()
        self.canvas.ax = self.canvas.fig.add_subplot(111, projection="3d")
        self.canvas.ax.set_zlim(0.0, 1.0)
        self.canvas.draw_idle()
        self.page.set_status("Таблица и поверхность сброшены", ok=True)


class Lab6Page(QWidget):
    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.tabs = QTabWidget()
        self.stack.addWidget(self.tabs)
        self.graph_panel = GraphPanel(self)
        self.stack.addWidget(self.graph_panel)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.dk = DKPanel(self)
        self.dsks = DSKSPanel(self)
        self.code = CodePanel(self)
        self.surface_dk = SurfaceTab("dk", "Информационная пропускная способность каналов ДСКС :")
        self.surface_dsks = SurfaceTab("dsks", "Информационная пропускная способность каналов ДК :")

        self.tabs.addTab(self._theory_tab(), TABS[0])
        self.tabs.addTab(self.dk, TABS[1])
        self.tabs.addTab(self.dsks, TABS[2])
        self.tabs.addTab(self.code, TABS[3])
        self.tabs.addTab(self.surface_dk, TABS[4])
        self.tabs.addTab(self.surface_dsks, TABS[5])

        self.active_task = 1
        self.dk.f_p1.set_value(0.05)
        self.dk.f_p2.set_value(0.10)
        self.dsks.f_p.set_value(0.05)
        self.dsks.f_q.set_value(0.125)
        self.set_status("Готово: введите вероятности и нажмите «Вычислить»", ok=True)
        self.dk.compute()
        self.dsks.compute()
        self.fill_tasks()


    def _theory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        apply_theory(view, "project6_1", THEORY)
        lay.addWidget(view)
        return w

    def set_status(self, text: str, ok: bool = True) -> None:
        self.status.setText(text)
        bg = "#f4f4f4" if ok else "#fce8e6"
        self.status.setStyleSheet(f"padding:4px; background:{bg};")

    def fill_tasks(self) -> None:
        name = self.code.cb_code.currentText()
        n, k = CODES.get(name, CODES[DEFAULT_CODE])
        for task in self.code.tasks.values():
            task.fill_from_code(n, k)
        self.code.tasks[3].f_a.set_value(0.05)
        self.code.tasks[3].f_b.set_value(0.10)

    def task_solved(self, mode: int, value: float) -> None:
        self.active_task = mode
        self.code.result_panel.setVisible(True)
        self.code.stack.setVisible(False)
        formula = {
            1: "C = (1−q)(1−log₂(1−q)) + (1−p−q)log₂(1−p−q) + p·log₂p",
            2: "C = (1−q)(1−log₂(1−q)) + (1−p−q)log₂(1−p−q) + p·log₂p",
            3: "C = H(X;Y) при Zo = (1 − p2·D)/(s·D)",
        }[mode]
        self.code.lbl_formula.setText(formula)
        self.code.lbl_solved.setText(f"Задача {mode} решена верно: C = {fmt(value)}")
        self.set_status(f"Задача {mode}: C = {fmt(value)} бит/символ", ok=True)


    def show_graph(self, which: str) -> None:
        self.graph_panel.load(which)
        self.stack.setCurrentWidget(self.graph_panel)
        self.set_status("Поверхность " + ("C = F[p1,p2] ДК" if which == "dk"
                                         else "C = F[p,q] ДСКС"), ok=True)

    def close_graph(self) -> None:
        self.stack.setCurrentWidget(self.tabs)
        self.set_status("Возврат к вкладкам формы", ok=True)
