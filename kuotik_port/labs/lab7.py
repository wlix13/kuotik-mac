"""ЛР7: корректирующие коды БЧХ — форма Form8 (15 вкладок).

Две ветви формы: «Построение кодов БЧХ» (теория, ввод N и T, таблица минимальных
многочленов, порождающая матрица) и «Исправление ошибок» (сведения CS и PGZ,
ввод A(x)/G(x)/T, вектор ошибок, методы CS и PGZ, результаты, поле GF(2^m),
матрица H). Расчёт — GF(2^m) над GF(2), минимальные многочлены через
циклотомические классы, декодирование PGZ по синдромам и CS циклическим сдвигом.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QFont, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..plot import Canvas

TABS = ("Построение кодов БЧХ", "Исправление ошибок")
TABS_BUILD = ("Общие сведения", "Ввод данных", "Таблица", "Порождающая матрица")
TABS_FIX = (
    "Сведения CS",
    "Сведения PGZ",
    "Ввод данных",
    "Вектор ошибок",
    "Метод CS",
    "Метод PGZ",
    "Результаты",
    "Построенное поле",
    "Матрица H",
)
N_CHOICES = (3, 5, 7, 9, 15, 21, 31, 63)
T_MAX = 7
PRIM = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011}
# минимальные многочлены F_i(x) по справке (табл. 7.2): i*C0 → код
MSG_NO_CODE = "БЧХ не существует, т.к. G(X) не укладывается в длину кода N"
MSG_NOT_CODE = ("Комбинация A(X) нацело на G(X) не делится, т.е. он не принадлежит "
                "циклическому коду")
MSG_T_BIG = "Код исправляет не более T ошибок включительно, а Вы пытаетесь задать больше"
MSG_BAD_BITS = "Введенное значение неверно: ожидаются двоичные символы 0/1"
MSG_NO_A = "Необходимо ввести A(X) и G(X) !"
MSG_BAD_G = "Порождающий многочлен должен быть ненулевым и иметь степень r <= N"

THEORY_BUILD = """<h3>Корректирующие коды Боуза–Чоудхури–Хоквингема</h3>
<p>Код БЧХ — циклический код над GF(2), построенный по элементам расширенного поля
Галуа. Поле GF(q) — множество с операциями сложения и умножения; для q = 2^m оно
строится по примитивному многочлену p(z), а примитивный элемент α порождает все
ненулевые элементы: α<sup>0</sup>, α<sup>1</sup>, …, α<sup>2<sup>m</sup>−2</sup>.</p>
<p>Минимальный многочлен F(x) элемента β над GF(2) — простой многочлен наименьшей
степени, для которого F(β) = 0. Он равен произведению (x + β<sup>2<sup>i</sup></sup>)
по всем сопряжённым β<sup>2<sup>i</sup></sup> (циклотомический класс), поэтому его
степень не больше m и он всегда имеет коэффициенты 0/1.</p>

<h4>Длина кода и параметр m</h4>
<p style="margin-left:18px">N = 2<sup>m</sup> − 1,&nbsp;&nbsp;
m = log₂(N + 1)&nbsp;&nbsp; или &nbsp;&nbsp;m = log₂(N·C₀ + 1),<br>
C₀ = (2<sup>m</sup> − 1)/N — коэффициент, при котором N·C₀ + 1 есть степень двойки.
Из набора N ∈ {3, 5, 7, 9, 15, 21, 31, 63} получаются m = 2, 4, 3, 6, 4, 6, 5, 6
и C₀ = 1, 3, 1, 7, 1, 3, 1, 1.</p>

<h4>Порождающий многочлен</h4>
<p style="margin-left:18px">G(x) = НОК [F<sub>1·C₀</sub>(x), F<sub>3·C₀</sub>(x), …,
F<sub>Φ·C₀</sub>(x)],&nbsp;&nbsp; Φ ≤ D − 2,&nbsp;&nbsp; D ≥ 2T + 1,<br>
r = deg G(x) ≤ m·T,&nbsp;&nbsp; k = N − r,&nbsp;&nbsp; M = [N, k].</p>
<p>Индекс Φ пробегает нечётные значения 1, 3, 5, …, 2T − 1: расширенный код БЧХ
исправляет T ошибок тогда и только тогда, когда его порождающий многочлен имеет
корнями α<sup>C₀</sup>, α<sup>3C₀</sup>, …, α<sup>(2T−1)C₀</sup>; чётные степени
добавляются автоматически как квадраты нечётных. Если r &gt; N (или k &lt; 1), код
БЧХ с такими N и T не существует.</p>
<p>Строки порождающей матрицы — остатки от деления x<sup>N−1−i</sup> на G(x),
дописанные к единичной информационной части (систематический код):
строка i = 1 в разряде N−1−i, далее r проверочных разрядов.</p>

<h4>Пример справки: N = 21, T = 2</h4>
<p style="margin-left:18px">m = log₂(21·3 + 1) = 6,&nbsp; C₀ = 3,&nbsp; Φ = D − 2 = 3,<br>
G(x) = НОК[F₃, F₉] = НОК[x⁶+x⁴+x²+x+1, x³+x²+1] =<br>
= x⁹ + x⁸ + x⁷ + x⁵ + x⁴ + x + 1 = 1110110011,<br>
r = deg G = 9,&nbsp; k = 21 − 9 = 12,&nbsp; M = [21, 12].</p>
<p>Остатки для строк матрицы (рис. 7.3 справки): 111011001, 100110101, 101000011,
101111000, 010111100, 001011110, 000101111, 111001110, 011100111, 110101010,
011010101, 110110011.</p>

<h4>Кодирование и синдромы</h4>
<p style="margin-left:18px">A(x) = u(x)·G(x),&nbsp;&nbsp; A″(x) = A(x) + E(x),<br>
S<sub>i</sub> = A″(α<sup>i·C₀</sup>) = e(α<sup>i·C₀</sup>),&nbsp; i = 1…2T,&nbsp;&nbsp;
S<sub>2i</sub> = S<sub>i</sub><sup>2</sup>.</p>
<p>Так как α<sup>i·C₀</sup> — корни G(x), любой кодовый вектор даёт нулевой синдром,
и синдром зависит только от вектора ошибок. Ненулевой синдром означает обнаруженную
ошибку; проверочная матрица H составлена из строк (1, α<sup>i·C₀</sup>, …,
α<sup>i·C₀(N−1)</sup>), i = 1…2T.</p>

<h4>Порядок выполнения</h4>
<ol>
<li>Задать N и T на вкладке «Ввод данных», вычислить m, C₀, Φ, r и G(x).</li>
<li>На вкладке «Таблица» взять из столбца m минимальные многочлены F<sub>i·C₀</sub>
и проверить ими порождающий многочлен.</li>
<li>Получить порождающую матрицу и сложить по модулю 2 любую комбинацию её строк —
это кодовая комбинация A(x) для второй части.</li>
<li>Задать вектор ошибок E(x), получить принятую комбинацию A″(x) = A(x) + E(x).</li>
<li>Исправить ошибки методом циклического сдвига (CS) и методом PGZ, сравнить
результаты и построенное поле GF(2<sup>m</sup>) с матрицей H.</li>
</ol>"""

THEORY_CS = """<h3>Метод циклического сдвига (CYCLIC SHIFT)</h3>
<p>Передавалась комбинация A(x) кода БЧХ (N, k) с порождающим многочленом G(x),
исправляющим ошибки кратности до T. В канале принята A″(x) = A(x) + E(x), где E(x) —
многочлен ошибки.</p>
<ol>
<li>Делим A″(x) на G(x) и вычисляем вес W полученного остатка.</li>
<li>Если W &gt; T, циклически сдвигаем делимое влево на один разряд и снова делим на G(x).</li>
<li>Процедура повторяется, пока вес остатка не станет W ≤ T.</li>
<li>Складываем последнее делимое с остатком по модулю 2 и сдвигаем полученную
комбинацию вправо на столько разрядов, на сколько сдвигали влево.</li>
</ol>
<p>Полученная комбинация — истинная переданная. Для кода (21, 12), T = 2,
A(x) = 110000000000011101100, E(x) = 110000000000000000000:</p>
<pre> A″(X)=000000000000011101100
 цикл 1: остаток 11101100,   W = 5 &gt; 2 → сдвиг влево на 1
 цикл 2: остаток 111011000,  W = 5 &gt; 2 → сдвиг влево на 1
 цикл 3: остаток 11,         W = 2 ≤ 2
 A″(X), сдвинутая влево на 2: 000000000001110110000
 сумма с остатком:            000000000001110110011
 сдвиг вправо на 2:           110000000000011101100 = A(X)</pre>
<p>Метод имеет ограниченное применение: если соседние единицы вектора ошибки стоят
на одинаковом расстоянии друг от друга с учётом циклического сдвига, алгоритм
зацикливается. Например, для кода БЧХ (15, 5) с α⁴ + α + 1 и вектором
E(x) = x¹⁴ + x¹² + x¹⁰ + x⁸ + x⁶ + x⁴ + x² + 1 исправить ошибку не удаётся.</p>"""

THEORY_PGZ = """<h3>Алгоритм Питерсона–Горенстейна–Цирлера (PGZ)</h3>
<p>Элементы α<sup>C₀</sup>, α<sup>3C₀</sup>, …, α<sup>(2T−1)C₀</sup> — корни G(x),
поэтому они корни любого многочлена A(x) кода. Значит, скалярные произведения
A·(1, α<sup>i·C₀</sup>, …) равны нулю, а векторы (1, α<sup>i·C₀</sup>, …,
α<sup>i·C₀(N−1)</sup>) образуют строки проверочной матрицы H.</p>
<p style="margin-left:18px">S = A″·H<sup>T</sup> = (A + E)·H<sup>T</sup> = E·H<sup>T</sup>,
&nbsp;&nbsp;S<sub>i</sub> = e(α<sup>i·C₀</sup>) = e₀ + e₁α<sup>i·C₀</sup> + … +
e<sub>N−1</sub>α<sup>i·C₀(N−1)</sup>.</p>
<p>Пусть реальная кратность ошибки ν ≤ T, локаторы X<sub>j</sub> = α<sup>C₀·p<sub>j</sub></sup>
(p<sub>j</sub> — позиции ошибок). Неизвестные коэффициенты полинома локаторов
s(z) = 1 + s₁z + … + s<sub>ν</sub>z<sup>ν</sup> находятся из системы Ньютона:</p>
<p style="margin-left:18px">M<sub>ν</sub>·(s₁, …, s<sub>ν</sub>)<sup>T</sup> =
(S<sub>ν+1</sub>, …, S<sub>2ν</sub>)<sup>T</sup>,&nbsp;&nbsp;
M<sub>ν</sub> = [S<sub>ν+i−j</sub>],&nbsp; i, j = 1…ν.</p>
<ol>
<li>Полагаем ν = T и вычисляем синдромы S₁…S<sub>2T</sub> (S<sub>2i</sub> = S<sub>i</sub>²).</li>
<li>Если det M<sub>ν</sub> ≠ 0 — решаем систему (в форме — методом Крамера).</li>
<li>Если det M<sub>ν</sub> = 0, матрица редуцируется (вычёркиваются последняя строка
и столбец), ν уменьшается, пока определитель не станет ненулевым: ошибок меньше T.</li>
<li>Составляем s(z) = 1 + s₁z + … + s<sub>ν</sub>z<sup>ν</sup> и подставляем в него все
ненулевые элементы поля. Корни s(z) — величины, обратные локаторам; позиции ошибок
p = −log<sub>α<sup>C₀</sup></sub>(корень).</li>
<li>Разряды, отвечающие локаторам, инвертируем. Если число корней меньше степени
s(z), ошибка неисправима — комбинация только обнаружена.</li>
</ol>
<p>Пример справки: код (15, 5, 7), p(x) = x⁴ + x + 1, g(x) = 10100110111,
u(x) = 101100100011110, e(x) = 001000001000001, r(x) = 100100101011111;
S₁ = α, S₂ = α², S₃ = α⁸, S₄ = α⁴, S₅ = 1, S₆ = α; корни локатора дают позиции
0, 6, 12, после инверсии разрядов получается u(x).</p>"""


# --- GF(2^m) и многочлены над GF(2) -------------------------------------------

class GF:
    """Поле GF(2^m): таблицы степеней и логарифмов по примитивному многочлену."""

    def __init__(self, m: int, prim: int | None = None):
        self.m = m
        self.prim = prim if prim is not None else PRIM[m]
        self.n = (1 << m) - 1
        self.exp = [0] * (2 * self.n)
        self.log = [0] * (1 << m)
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << m):
                x ^= self.prim
            x &= (1 << m) - 1
        for i in range(self.n, 2 * self.n):
            self.exp[i] = self.exp[i - self.n]

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % self.n]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("деление на нуль в GF(2^m)")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % self.n]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("обратного элемента у нуля нет")
        return self.exp[(-self.log[a]) % self.n]

    def pow(self, a: int, k: int) -> int:
        if a == 0:
            return 0
        return self.exp[(self.log[a] * k) % self.n]

    def name(self, a: int) -> str:
        """Элемент в виде степени примитивного элемента."""
        if a == 0:
            return "0"
        e = self.log[a]
        return "1" if e == 0 else f"α^{e}"

    def poly(self, a: int) -> str:
        """Элемент как многочлен от z."""
        if a == 0:
            return "0"
        return fmt_poly(a, "z")

    def table(self) -> list[tuple[int, int, str, str]]:
        """Строки: степень, элемент, многочлен, двоичный код."""
        return [
            (i, self.exp[i], self.poly(self.exp[i]), bin(self.exp[i])[2:].rjust(self.m, "0"))
            for i in range(self.n)
        ]


def poly_deg(p: int) -> int:
    return p.bit_length() - 1


def poly_mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def poly_mod(a: int, g: int) -> int:
    dg = poly_deg(g)
    while a and poly_deg(a) >= dg:
        a ^= g << (poly_deg(a) - dg)
    return a


def poly_divmod(a: int, g: int) -> tuple[int, int]:
    dg = poly_deg(g)
    q = 0
    while a and poly_deg(a) >= dg:
        s = poly_deg(a) - dg
        q |= 1 << s
        a ^= g << s
    return q, a


def poly_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, poly_mod(a, b)
    return a


def poly_lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return poly_mul(poly_divmod(a, poly_gcd(a, b))[0], b)


def bits(value: int, width: int | None = None) -> str:
    s = bin(max(value, 0))[2:]
    return s.rjust(width, "0") if width else s


def fmt_poly(p: int, var: str = "x") -> str:
    """Многочлен из битовой маски: 1011 → x^3 + x + 1."""
    if p == 0:
        return "0"
    terms = []
    for i in range(poly_deg(p), -1, -1):
        if not (p >> i) & 1:
            continue
        if i == 0:
            terms.append("1")
        elif i == 1:
            terms.append(var)
        else:
            terms.append(f"{var}^{i}")
    return " + ".join(terms)


def parse_bits(text: str) -> int | None:
    """Строка 0/1 → число; None, если есть другие символы."""
    s = (text or "").strip().replace(" ", "")
    if not s or any(c not in "01" for c in s):
        return None
    return int(s, 2)


def minimal_poly(gf: GF, power: int, N: int) -> tuple[int, list[int]]:
    """Минимальный многочлен α^power над GF(2) и его циклотомический класс."""
    conj: list[int] = []
    cur = power % gf.n
    while cur not in conj:
        conj.append(cur)
        cur = (cur * 2) % gf.n
    coeffs = [1]                      # по возрастанию степеней, коэффициенты GF
    for c in conj:
        val = gf.exp[c]
        new = [0] * (len(coeffs) + 1)
        for i, cf in enumerate(coeffs):
            new[i] ^= gf.mul(cf, val)
            new[i + 1] ^= cf
        coeffs = new
    assert all(c in (0, 1) for c in coeffs), coeffs
    p = 0
    for c in reversed(coeffs):
        p = (p << 1) | c
    return p, conj


@dataclass
class BCHModel:
    """Параметры кода БЧХ (N, T) и всё, что из них следует."""

    N: int
    T: int
    m: int = 0
    C0: int = 1
    D: int = 0
    exists: bool = True
    error: str = ""
    minpolys: list[tuple[int, int, list[int]]] = field(default_factory=list)
    G: int = 0
    gf: GF | None = None

    @property
    def Phi(self) -> int:
        return 2 * self.T - 1

    @property
    def r(self) -> int:
        return poly_deg(self.G) if self.G else 0

    @property
    def k(self) -> int:
        return self.N - self.r

    @property
    def r_limit(self) -> int:
        return self.m * self.T

    @property
    def exponents(self) -> list[int]:
        return [i * self.C0 for i in range(1, 2 * self.T, 2)]

    def generator(self) -> int:
        return self.G

    def matrix(self) -> list[int]:
        """Систематическая порождающая матрица: k строк по N разрядов."""
        rows = []
        for i in range(self.k):
            info = 1 << (self.N - 1 - i)
            rows.append(info | poly_mod(info, self.G))
        return rows

    def encode(self, message: int) -> int:
        """Систематическое кодирование: u(x)·x^r + остаток от деления на G(x)."""
        word = message << self.r
        return word | poly_mod(word, self.G)

    def encode_poly(self, message: int) -> int:
        """Несистематическое кодирование A(x) = u(x)·G(x)."""
        return poly_mul(message, self.G)


def build_bch(N: int, T: int) -> BCHModel:
    """Код БЧХ по N и T: m, C0, минимальные многочлены, G(x), r, k."""
    model = BCHModel(N=N, T=T, D=2 * T + 1)
    m = next((mm for mm in range(2, 13) if ((1 << mm) - 1) % N == 0), None)
    if m is None:
        model.exists = False
        model.error = "N не представимо как 2^m − 1 с множителем"
        return model
    model.m = m
    model.C0 = ((1 << m) - 1) // N
    gf = GF(m)
    model.gf = gf
    polys: list[tuple[int, int, list[int]]] = []
    G = 1
    for i in range(1, 2 * T, 2):
        p, conj = minimal_poly(gf, i * model.C0, N)
        polys.append((i, p, conj))
        G = poly_lcm(G, p)
    model.minpolys = polys
    model.G = G
    if poly_deg(G) > N:
        model.exists = False
        model.error = MSG_NO_CODE
    elif model.k < 1:
        model.exists = False
        model.error = MSG_NO_CODE
    elif poly_mod((1 << N) ^ 1, G) != 0:
        model.exists = False
        model.error = "G(x) не делит x^N + 1"
    return model


def syndromes(word: int, model: BCHModel) -> list[int]:
    """S_i = A(α^(i·C0)), i = 1…2T."""
    gf, N, C0 = model.gf, model.N, model.C0
    alpha = gf.exp[1]
    out = []
    for i in range(1, 2 * model.T + 1):
        s = 0
        a = gf.pow(alpha, i * C0)
        for j in range(N):
            if (word >> j) & 1:
                s ^= gf.pow(a, j)
        out.append(s)
    return out


def h_matrix(model: BCHModel) -> list[list[int]]:
    """Проверочная матрица 2T × N в степенях α."""
    gf, N, C0 = model.gf, model.N, model.C0
    alpha = gf.exp[1]
    rows = []
    for i in range(1, 2 * model.T + 1):
        a = gf.pow(alpha, i * C0)
        rows.append([gf.pow(a, j) for j in range(N)])
    return rows


def gf_det(mat: list[list[int]], gf: GF) -> int:
    """Определитель над GF(2^m) разложением по первой строке."""
    n = len(mat)
    if n == 0:
        return 1
    if n == 1:
        return mat[0][0]
    total = 0
    for j in range(n):
        if mat[0][j] == 0:
            continue
        minor = [[mat[i][k] for k in range(n) if k != j] for i in range(1, n)]
        total ^= gf.mul(mat[0][j], gf_det(minor, gf))
    return total


def gf_solve_cramer(A: list[list[int]], b: list[int], gf: GF) -> list[int] | None:
    """Решение системы A·s = b по правилу Крамера; None при нулевом определителе."""
    D = gf_det(A, gf)
    if D == 0:
        return None
    out = []
    n = len(A)
    for j in range(n):
        M = [row[:] for row in A]
        for i in range(n):
            M[i][j] = b[i]
        out.append(gf.div(gf_det(M, gf), D))
    return out


@dataclass
class PGZResult:
    """Итог PGZ: синдромы, локатор, позиции ошибок, исправленная комбинация."""

    ok: bool
    detected: bool = False
    nu: int = 0
    syndromes: list[int] = field(default_factory=list)
    matrix: list[list[int]] = field(default_factory=list)
    det: int = 0
    coeffs: list[int] = field(default_factory=list)
    roots: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    corrected: int | None = None
    steps: list[str] = field(default_factory=list)


def pgz_decode(word: int, model: BCHModel) -> PGZResult:
    """PGZ: синдромы → матрица Ньютона → Крамер → корни локатора → инверсия разрядов."""
    gf, N, T = model.gf, model.N, model.T
    S = [0] + syndromes(word, model)          # с 1
    res = PGZResult(ok=False, syndromes=S[1:])
    if all(v == 0 for v in S[1:]):
        res.ok = True
        res.nu = 0
        res.corrected = word
        res.steps.append("Синдромы нулевые: ошибок нет")
        return res
    res.detected = True
    nu = T
    while nu > 0:
        M = [[S[nu + i - j] for j in range(1, nu + 1)] for i in range(1, nu + 1)]
        d = gf_det(M, gf)
        res.steps.append(f"ν = {nu}: det M = {gf.name(d)}")
        if d != 0:
            break
        nu -= 1
    res.nu = nu
    if nu == 0:
        res.steps.append("Все определители нулевые: исправление невозможно")
        return res
    M = [[S[nu + i - j] for j in range(1, nu + 1)] for i in range(1, nu + 1)]
    b = [S[nu + i] for i in range(1, nu + 1)]
    res.matrix = M
    res.det = gf_det(M, gf)
    coeffs = gf_solve_cramer(M, b, gf)
    if coeffs is None:
        res.steps.append("Система Ньютона не решена")
        return res
    res.coeffs = coeffs
    res.steps.append("s(z) = 1 + " + " + ".join(
        f"{gf.name(c)}·z^{i}" for i, c in enumerate(coeffs, start=1)))
    for k in range(nu + 1, T + 1):
        lhs = S[k]
        rhs = 0
        for j, cf in enumerate(coeffs, start=1):
            if k - j >= 1:
                rhs ^= gf.mul(cf, S[k - j])
        if lhs != rhs:
            res.steps.append("Синдромы не согласованы с локатором: ошибка неисправима")
            return res
    roots = []
    for p in range(N):
        z = gf.exp[(-model.C0 * p) % gf.n]
        val = 1
        zp = 1
        for cf in coeffs:
            zp = gf.mul(zp, z)
            val ^= gf.mul(cf, zp)
        if val == 0:
            roots.append(p)
    res.roots = roots
    res.steps.append("Корни s(z): " + ", ".join(f"α^{-model.C0 * p}" for p in roots))
    if len(roots) != nu:
        res.steps.append(f"Корней {len(roots)} из {nu}: ошибка неисправима")
        return res
    res.positions = sorted(roots)
    corrected = word
    for p in roots:
        corrected ^= 1 << p
    res.corrected = corrected
    res.ok = True
    res.steps.append("Разряды " + ", ".join(str(p) for p in res.positions) + " инвертированы")
    return res


@dataclass
class CSCycle:
    """Один цикл CS: сдвиг, делимое, остаток, его вес."""

    shift: int
    dividend: int
    remainder: int
    weight: int


@dataclass
class CSResult:
    """Итог метода циклического сдвига."""

    ok: bool
    detected: bool = False
    cycles: list[CSCycle] = field(default_factory=list)
    shifts: int = 0
    remainder: int = 0
    shifted: int = 0
    summed: int = 0
    corrected: int | None = None
    looped: bool = False
    steps: list[str] = field(default_factory=list)


def cs_decode(word: int, G: int, N: int, T: int) -> CSResult:
    """CS: деление на G(x), сдвиги влево, пока вес остатка > T, обратный сдвиг."""
    res = CSResult(ok=False)
    if G == 0 or poly_deg(G) > N:
        res.steps.append("Порождающий многочлен не задан")
        return res
    cur = word
    mask = (1 << N) - 1
    for shift in range(N + 1):
        rem = poly_mod(cur, G)
        w = bin(rem).count("1")
        res.cycles.append(CSCycle(shift, cur, rem, w))
        res.steps.append(f"цикл {shift + 1}: остаток {bits(rem)}, вес W = {w}")
        if w <= T:
            res.shifts = shift
            res.remainder = rem
            res.shifted = cur
            res.summed = cur ^ rem
            fixed = ((res.summed >> shift) | (res.summed << (N - shift))) & mask if shift else res.summed
            res.corrected = fixed
            res.ok = True
            res.detected = w != 0 or shift != 0
            return res
        if shift == N - 1:
            break
        cur = ((cur << 1) | (cur >> (N - 1))) & mask
    res.looped = True
    res.steps.append("За N сдвигов остаток с весом W ≤ T не получен: метод CS зациклился")
    return res


def min_distance(model: BCHModel, limit: int = 1 << 16) -> int:
    """Минимальное расстояние перебором кодовых слов (для малых k)."""
    if model.k > 16 or (1 << model.k) > limit:
        raise ValueError("перебор слишком велик")
    best = model.N + 1
    rows = model.matrix()
    for combo in range(1, 1 << model.k):
        word = 0
        for i in range(model.k):
            if (combo >> i) & 1:
                word ^= rows[i]
        w = bin(word).count("1")
        if 0 < w < best:
            best = w
            if best == 1:
                break
    return best


# --- виджеты ------------------------------------------------------------------

def _mono_font() -> QFont:
    """Моноширинный шрифт для двоичных комбинаций."""
    font = QFont("Menlo")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(11)
    return font


def _mono(text: str = "", size: int = 12) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-family: Menlo, monospace; font-size: {size}px;")
    return lbl


class BitsEdit(QLineEdit):
    """Поле двоичной комбинации фиксированной длины N."""

    def __init__(self, length: int, value: int = 0):
        super().__init__()
        self._len = length
        self.setFont(_mono_font())
        self._apply(length)
        self.set_bits(value)

    def _apply(self, length: int) -> None:
        self.setMaxLength(length)
        self.setValidator(QRegularExpressionValidator(QRegularExpression(f"[01]{{0,{length}}}")))
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance("0" * length) + 28)

    def set_length(self, length: int) -> None:
        self._len = length
        self._apply(length)

    def bits(self) -> int | None:
        return parse_bits(self.text())

    def set_bits(self, value: int) -> None:
        self.setText(bits(value, self._len))

    def is_complete(self) -> bool:
        return len(self.text()) == self._len and self.bits() is not None


class PolyEdit(QLineEdit):
    """Поле многочлена в виде двоичной маски."""

    def __init__(self, value: int = 0):
        super().__init__()
        self.setValidator(QRegularExpressionValidator(QRegularExpression("[01]{1,64}")))
        self.setFont(_mono_font())
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance("0" * 24) + 24)
        self.set_value(value)

    def value(self) -> int | None:
        return parse_bits(self.text())

    def set_value(self, value: int) -> None:
        self.setText(bits(value))


class Title(QLabel):
    def __init__(self, text: str, size: int = 14):
        super().__init__(text)
        self.setStyleSheet(f"font-weight: 600; font-size: {size}px;")


class ResultRow(QWidget):
    """Строка «подпись → значение»."""

    def __init__(self, caption: str, width: int = 0):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.caption = _mono(caption)
        if width:
            self.caption.setFixedWidth(width)
        self.value = _mono("—")
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.caption)
        lay.addWidget(self.value, 1)

    def set(self, value: float | str) -> None:
        self.value.setText(value if isinstance(value, str) else f"{value:.5f}")

    def clear(self) -> None:
        self.value.setText("—")


# --- вкладка «Ввод данных» (построение) ---------------------------------------

class BuildInputPanel(QWidget):
    """N из набора, T = 1…7, расчёт m, C₀, Φ, r, G(x) и Memo2 протокола."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(_mono("Ваша задача состоит в построении порождающей матрицы кодов БЧХ", 13))
        lay.addWidget(_mono("Введите N из набора {3, 5, 7, 9, 15, 21, 31, 63} и Т из условия, "
                            "что 1<= T <=7", 12))

        top = QHBoxLayout()
        top.addWidget(_mono("N =", 13))
        self.cb_n = QComboBox()
        self.cb_n.addItems([str(n) for n in N_CHOICES])
        self.cb_n.setCurrentText("21")
        top.addWidget(self.cb_n)
        top.addSpacing(24)
        top.addWidget(_mono("T =", 13))
        self.sp_t = QSpinBox()
        self.sp_t.setRange(1, T_MAX)
        self.sp_t.setValue(2)
        top.addWidget(self.sp_t)
        top.addSpacing(24)
        self.btn_calc = QPushButton("Вычислить")
        self.btn_reset = QPushButton("Сброс")
        top.addWidget(self.btn_calc)
        top.addWidget(self.btn_reset)
        top.addStretch(1)
        lay.addLayout(top)
        lay.addWidget(_mono("Новые параметры задавать после сброса предыдущих результатов", 11))

        lay.addWidget(_mono("m можно вычислить по формуле:", 12))
        lay.addWidget(_mono("m = log2 ( N+1 )     или      m = log2 ( N · C0 + 1 )", 12))

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(2)
        self.r_m = ResultRow("m =", 30)
        self.r_phi = ResultRow("Индекс Ф <= D - 2", 190)
        self.r_n = ResultRow("Длина кода  N =", 190)
        self.r_mt = ResultRow("Ограничение  m * T =", 190)
        self.r_c0 = ResultRow("Коэффициент  C0 =", 190)
        grid.addWidget(self.r_m, 0, 0)
        grid.addWidget(self.r_phi, 0, 1)
        grid.addWidget(self.r_n, 0, 2)
        grid.addWidget(self.r_mt, 0, 3)
        grid.addWidget(self.r_c0, 0, 4)

        self.r_r = ResultRow("Число проверочных разрядов   r = deg G(x)", 330)
        self.r_k = ResultRow("Число информационных разрядов   k = N - r", 330)
        self.r_d = ResultRow("Минимальное расстояние Хэмминга   D >= 2*T+1", 380)
        grid.addWidget(self.r_r, 1, 0, 1, 2)
        grid.addWidget(self.r_k, 1, 2, 1, 2)
        grid.addWidget(self.r_d, 1, 4)
        lay.addLayout(grid)

        lay.addWidget(_mono("G(x) = НОК [F 1*Co (x), F 3*Co (x), ... , F i*Co (x), ... , "
                            "F Ф*Co (x)]", 12))
        self.r_g = ResultRow("G(x) =", 50)
        lay.addWidget(self.r_g)
        self.val_g = _mono("—")
        self.val_g.setWordWrap(True)
        self.val_g.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.val_g)

        self.memo = QPlainTextEdit()
        self.memo.setReadOnly(True)
        self.memo.setFixedHeight(150)
        self.memo.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        lay.addWidget(self.memo, 1)

        self.btn_calc.clicked.connect(self.compute)
        self.btn_reset.clicked.connect(self.reset)

    def params(self) -> tuple[int, int]:
        return int(self.cb_n.currentText()), int(self.sp_t.value())

    def compute(self) -> BCHModel:
        N, T = self.params()
        model = build_bch(N, T)
        self.page.set_model(model)
        self.show_model(model)
        return model

    def show_model(self, model: BCHModel) -> None:
        self.r_m.set(str(model.m))
        self.r_phi.set(str(model.Phi))
        self.r_n.set(str(model.N))
        self.r_mt.set(f"{model.m} * {model.T} = {model.r_limit}")
        self.r_c0.set(str(model.C0))
        self.r_d.set(f"D = {model.D}")
        if not model.exists:
            self.r_r.set("—")
            self.r_k.set("—")
            self.r_g.set("—")
            self.val_g.setText(model.error)
            self.memo.setPlainText(f"N = {model.N}, T = {model.T}\n{model.error}")
            self.page.set_status(model.error, ok=False)
            return
        self.r_r.set(f"r = {model.r}")
        self.r_k.set(f"k = {model.k}")
        self.r_g.set(bits(model.G))
        self.val_g.setText(f"{fmt_poly(model.G)}   =  {bits(model.G)}"
                           f"   (степень {model.r}, k = {model.k})")
        lines = [
            f"N = {model.N}, T = {model.T}",
            f"m = {model.m}, C0 = {model.C0} = (2^{model.m} - 1)/N",
            f"Ф = 2*T - 1 = {model.Phi}, D >= 2*T + 1 = {model.D}",
            f"r = deg G(x) = {model.r} <= m*T = {model.r_limit}",
            f"k = N - r = {model.k}, код ({model.N}, {model.k})",
            "G(x) = " + fmt_poly(model.G),
            "G(x) = " + bits(model.G),
        ]
        for i, p, conj in model.minpolys:
            lines.append(f"F {i}*Co = {bits(p)} = {fmt_poly(p)}"
                         f"  (сопряжённые: {', '.join('α^%d' % c for c in conj)})")
        self.memo.setPlainText("\n".join(lines))
        self.page.set_status(
            f"Код БЧХ ({model.N}, {model.k}): r = {model.r}, G(x) = {bits(model.G)}", ok=True)

    def reset(self) -> None:
        for row in (self.r_m, self.r_phi, self.r_n, self.r_mt, self.r_c0, self.r_r, self.r_k, self.r_d):
            row.clear()
        self.r_g.clear()
        self.val_g.setText("—")
        self.memo.clear()
        self.page.set_status("Результаты построения сброшены", ok=True)


# --- вкладка «Таблица» --------------------------------------------------------

class TablePanel(QWidget):
    """Столбец m таблицы минимальных многочленов и восемь полей F_i*Co."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(_mono("По полученному значению m, найдите в таблице соответствующий столбец", 12))
        lay.addWidget(_mono(" и введите коды, содержащиеся в этом столбце, в соответствии с", 12))
        lay.addWidget(_mono("номерами  i*Co , которые Вы получили выше.", 12))

        self.tbl = QTableWidget(T_MAX, 3)
        self.tbl.setHorizontalHeaderLabels(["i*Co", "F i*Co (x)", "двоичный код"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setFixedHeight(210)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.tbl)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.fields: list[PolyEdit] = []
        for row, i in enumerate(range(1, 16, 2)):
            lbl = _mono(f"F {i}*Co =", 12)
            fld = PolyEdit()
            fld.setMinimumWidth(120)
            self.fields.append(fld)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(fld, row, 1)
        self.btn_calc = QPushButton("Вычислить")
        grid.addWidget(self.btn_calc, 0, 2, 2, 1)
        self.r_nk = ResultRow("Порождающий многочлен кода БЧХ (N, k)", 330)
        self.r_g = ResultRow("G(X) =", 60)
        grid.addWidget(self.r_nk, 2, 2, 1, 2)
        grid.addWidget(self.r_g, 3, 2, 1, 2)
        lay.addLayout(grid)
        lay.addWidget(_mono("Примечание: нулевой код заменяем единицей", 11))
        lay.addStretch(1)

        self.btn_calc.clicked.connect(self.compute)

    def fill(self, model: BCHModel) -> None:
        """Заполнить столбец m: F_i*Co для текущих N и T."""
        self.tbl.setRowCount(max(len(model.minpolys), 1))
        for row, (i, p, conj) in enumerate(model.minpolys):
            values = [f"{i}*{model.C0}", fmt_poly(p), bits(p)]
            for col, text in enumerate(values):
                self.tbl.setItem(row, col, QTableWidgetItem(text))
        for row, fld in enumerate(self.fields):
            fld.setEnabled(True)
            fld.set_value(model.minpolys[row][1] if row < len(model.minpolys) else 1)
        self.r_nk.set(f"({model.N}, {model.k})" if model.exists else "—")
        self.r_g.set(bits(model.G) if model.exists else "—")

    def compute(self) -> int | None:
        """LCM введённых многочленов → G(x); проверка делимости x^N + 1."""
        model = self.page.model
        polys = []
        for row, fld in enumerate(self.fields):
            if not fld.isEnabled():
                continue
            v = fld.value()
            if v is None:
                self.page.set_status(MSG_BAD_BITS, ok=False)
                return None
            polys.append(v or 1)                 # нулевой код заменяем единицей
        if not polys:
            self.page.set_status(MSG_BAD_G, ok=False)
            return None
        G = 1
        for p in polys:
            G = poly_lcm(G, p)
        if G == 0 or poly_deg(G) > model.N:
            self.page.set_status(MSG_BAD_G, ok=False)
            return None
        if poly_mod((1 << model.N) ^ 1, G) != 0:
            self.page.set_status("G(x) не делит x^N + 1: многочлены выбраны неверно", ok=False)
            return None
        self.page.set_generator(G)
        self.r_nk.set(f"({model.N}, {model.N - poly_deg(G)})")
        self.r_g.set(bits(G))
        self.page.set_status(
            f"G(x) = {bits(G)} = {fmt_poly(G)}, r = {poly_deg(G)}, "
            f"k = {model.N - poly_deg(G)}", ok=True)
        return G


# --- вкладка «Порождающая матрица» --------------------------------------------

class MatrixPanel(QWidget):
    """Порождающая матрица и сложение выбранных строк в A(x)."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(Title("Порождающая матрица:", 13))
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        self.text.setMinimumHeight(150)
        lay.addWidget(self.text, 1)

        lay.addWidget(_mono("Сложите по модулю 2 выбранные строки — получится кодовая "
                            "комбинация A(x) для второй части:", 11))
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["строка", "информационная часть", "проверочные разряды"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        lay.addWidget(self.tbl, 1)

        row = QHBoxLayout()
        self.btn_sum = QPushButton("Сложить выбранные строки → A(x)")
        self.btn_all = QPushButton("Снять отметки")
        row.addWidget(self.btn_sum)
        row.addWidget(self.btn_all)
        row.addStretch(1)
        lay.addLayout(row)
        self.btn_sum.clicked.connect(self.sum_rows)
        self.btn_all.clicked.connect(self.clear_marks)

    def fill(self, model: BCHModel) -> None:
        rows = model.matrix() if model.exists else []
        lines = [f"M = [{model.N}, {model.k}]   G(x) = {bits(model.G)}"] if model.exists else []
        self.tbl.setRowCount(len(rows))
        for i, word in enumerate(rows):
            info = bits(word >> model.r, model.k)
            check = bits(word & ((1 << model.r) - 1), model.r)
            lines.append(f"{info}:{check}   = {bits(word, model.N)}")
            item = QTableWidgetItem(str(i + 1))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.tbl.setItem(i, 0, item)
            self.tbl.setItem(i, 1, QTableWidgetItem(info))
            self.tbl.setItem(i, 2, QTableWidgetItem(check))
        self.text.setPlainText("\n".join(lines))

    def marked(self) -> list[int]:
        out = []
        for i in range(self.tbl.rowCount()):
            if self.tbl.item(i, 0).checkState() == Qt.Checked:
                out.append(i)
        return out

    def clear_marks(self) -> None:
        for i in range(self.tbl.rowCount()):
            self.tbl.item(i, 0).setCheckState(Qt.Unchecked)

    def sum_rows(self) -> int | None:
        model = self.page.model
        idx = self.marked()
        if not idx:
            self.page.set_status("Отметьте хотя бы одну строку матрицы", ok=False)
            return None
        rows = model.matrix()
        word = 0
        for i in idx:
            word ^= rows[i]
        self.page.set_code(word)
        self.page.set_status(
            f"A(x) = {bits(word, model.N)} (сложено строк: {len(idx)}) → вкладка «Ввод данных»",
            ok=True)
        return word


# --- вторая часть: ввод данных ------------------------------------------------

class FixInputPanel(QWidget):
    """A(x), G(x) и T для исправления ошибок; «Далее >>» и «Сброс»."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(_mono("Введите кодовую комбинацию длиной N на основе первой части "
                            "данной работы.", 12))
        lay.addWidget(_mono("Для этого сложите по модулю 2 любую комбинацию строк ранее "
                            "полученной", 12))
        lay.addWidget(_mono("порождающей матрицы и Вы получите искомую кодовую комбинацию.", 12))

        lay.addWidget(_mono("A(x)=", 13))
        self.f_a = BitsEdit(21)
        lay.addWidget(self.f_a)

        lay.addWidget(_mono("Введите порождающий многочлен, который Вы получили в первой", 12))
        lay.addWidget(_mono("части лабораторной работы", 12))
        lay.addWidget(_mono("G(x)=", 13))
        self.f_g = PolyEdit()
        lay.addWidget(self.f_g)

        self.r_info = ResultRow("Информационные разряды  k =", 220)
        self.r_check = ResultRow("Проверочные разряды  r = deg G(x)", 220)
        lay.addWidget(self.r_info)
        lay.addWidget(self.r_check)

        row = QHBoxLayout()
        row.addWidget(_mono("Задайте максимальную кратность исправляемых ошибок :", 12))
        self.sp_t = QSpinBox()
        self.sp_t.setRange(1, T_MAX)
        self.sp_t.setValue(2)
        row.addWidget(self.sp_t)
        row.addStretch(1)
        self.btn_next = QPushButton("Далее  >>")
        self.btn_reset = QPushButton("Сброс")
        row.addWidget(self.btn_next)
        row.addWidget(self.btn_reset)
        lay.addLayout(row)
        lay.addStretch(1)

        self.btn_next.clicked.connect(self.next_step)
        self.btn_reset.clicked.connect(self.reset)

    def next_step(self) -> bool:
        model = self.page.model
        a = self.f_a.bits()
        g = self.f_g.value()
        if a is None or not self.f_a.is_complete():
            self.page.set_status(MSG_BAD_BITS, ok=False)
            return False
        if not g:
            self.page.set_status(MSG_NO_A, ok=False)
            return False
        if poly_mod(a, g) != 0:
            self.page.set_status(MSG_NOT_CODE, ok=False)
            return False
        if int(self.sp_t.value()) > model.T:
            self.page.set_status(MSG_T_BIG, ok=False)
            return False
        self.page.set_generator(g)
        self.page.set_code(a)
        r = poly_deg(g)
        self.r_info.set(bits(a >> r, model.N - r))
        self.r_check.set(bits(a & ((1 << r) - 1), r))
        self.page.fix_tabs.setCurrentIndex(3)
        self.page.set_status(
            f"A(x) = {bits(a, model.N)} принята: делится на G(x) нацело, T = {self.sp_t.value()}",
            ok=True)
        return True

    def reset(self) -> None:
        self.f_a.clear()
        self.f_g.clear()
        self.r_info.clear()
        self.r_check.clear()
        self.page.set_status("Ввод второй части сброшен", ok=True)


class ErrorVectorPanel(QWidget):
    """E(x) и принятая комбинация A″(x) = A(x) + E(x)."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(_mono("Задайте многочлен ошибок длиной в N двоичных символов, где 0 -", 12))
        lay.addWidget(_mono("соответствует правильно переданному разряду,   1- ошибочному.", 12))
        lay.addWidget(_mono("E(x) =", 13))
        self.f_e = BitsEdit(21)
        lay.addWidget(self.f_e)

        lay.addWidget(_mono("В результате действия помех в канале связи на приемном конце", 12))
        lay.addWidget(_mono("была принята комбинация", 12))
        self.r_ap = ResultRow("A''(x)=", 70)
        lay.addWidget(self.r_ap)

        lay.addWidget(_mono("Общее количество исправляемых ошибок, заданной кратности   T ,", 12))
        lay.addWidget(_mono("вычисляется по формуле:  W(E) <= T.", 12))
        lay.addWidget(_mono("Ваша задача - локализовать ошибки и получить истинную "
                            "переданную комбинацию.", 12))
        row = QHBoxLayout()
        self.btn_send = QPushButton("Передать")
        row.addWidget(self.btn_send)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        self.btn_send.clicked.connect(self.send)

    def send(self) -> int | None:
        model = self.page.model
        e = self.f_e.bits()
        if e is None or not self.f_e.is_complete():
            self.page.set_status(MSG_BAD_BITS, ok=False)
            return None
        if self.page.code is None:
            self.page.set_status(MSG_NO_A, ok=False)
            return None
        word = self.page.code ^ e
        self.page.set_received(word, e)
        self.r_ap.set(bits(word, model.N))
        self.page.set_status(
            f"A''(x) = {bits(word, model.N)}; вес вектора ошибок W(E) = {bin(e).count('1')}",
            ok=True)
        return word


# --- методы CS и PGZ ----------------------------------------------------------

class CSPanel(QWidget):
    """«Динамика» — протокол циклов CS, «Результаты» — итог."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.btn_dyn = QPushButton("Динамика")
        self.btn_res = QPushButton("Результаты")
        row.addWidget(self.btn_dyn)
        row.addWidget(self.btn_res)
        row.addStretch(1)
        lay.addLayout(row)
        self.memo = QPlainTextEdit()
        self.memo.setReadOnly(True)
        self.memo.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        lay.addWidget(self.memo, 1)
        self.canvas = Canvas(height=3.0)
        lay.addWidget(self.canvas, 1)
        self.btn_dyn.clicked.connect(self.dynamics)
        self.btn_res.clicked.connect(self.results)

    def dynamics(self) -> CSResult | None:
        res = self.page.run_cs()
        if res is None:
            return None
        self.memo.setPlainText("\n".join(res.steps))
        self.draw(res)
        return res

    def results(self) -> None:
        res = self.page.cs_result or self.page.run_cs()
        if res is None:
            return
        self.memo.setPlainText("\n".join(res.steps))
        self.draw(res)
        self.page.fill_results(res)
        self.page.fix_tabs.setCurrentIndex(6)

    def draw(self, res: CSResult) -> None:
        self.canvas.clear("Вес остатка по циклам сдвига", "цикл", "W")
        w = [c.weight for c in res.cycles]
        x = list(range(1, len(w) + 1))
        self.canvas.ax.bar(x, w, color="#1a73e8", width=0.55)
        self.canvas.ax.axhline(self.page.model.T, color="#b3261e", linestyle="--",
                               linewidth=1.2)
        if x:
            self.canvas.ax.set_xticks(x)
            self.canvas.ax.set_xlim(0.4, x[-1] + 0.6)
        self.canvas.ax.annotate(f"T = {self.page.model.T}",
                                xy=(x[-1] if x else 1, self.page.model.T),
                                xytext=(2, 4), textcoords="offset points", fontsize=8)
        for xi, wi in zip(x, w):
            self.canvas.ax.annotate(str(wi), (xi, wi), ha="center",
                                    xytext=(0, 3), textcoords="offset points", fontsize=8)
        self.canvas.draw_idle()


class PGZPanel(QWidget):
    """«Запуск PGZ», таблица синдромов, протокол и переходы к полю и матрице H."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Запуск PGZ")
        self.btn_field = QPushButton("Поле GF(2^m)")
        self.btn_h = QPushButton("Матрица H")
        row.addWidget(self.btn_run)
        row.addStretch(1)
        row.addWidget(self.btn_field)
        row.addWidget(self.btn_h)
        lay.addLayout(row)
        body = QHBoxLayout()
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["S i", "значение"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setFixedWidth(190)
        body.addWidget(self.tbl)
        self.memo = QPlainTextEdit()
        self.memo.setReadOnly(True)
        self.memo.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        body.addWidget(self.memo, 1)
        lay.addLayout(body, 1)
        self.btn_run.clicked.connect(self.run)
        self.btn_field.clicked.connect(lambda: page.fix_tabs.setCurrentIndex(7))
        self.btn_h.clicked.connect(lambda: page.fix_tabs.setCurrentIndex(8))

    def run(self) -> PGZResult | None:
        res = self.page.run_pgz()
        if res is None:
            return None
        self.tbl.setRowCount(len(res.syndromes))
        gf = self.page.model.gf
        for i, s in enumerate(res.syndromes):
            self.tbl.setItem(i, 0, QTableWidgetItem(f"S{i + 1}"))
            self.tbl.setItem(i, 1, QTableWidgetItem(f"{gf.name(s)}"))
        lines = [f"S = ({', '.join(gf.name(s) for s in res.syndromes)})"]
        lines += res.steps
        self.memo.setPlainText("\n".join(lines))
        return res


class ResultsPanel(QWidget):
    """Итог CS: сдвиг влево, сумма с остатком, обратный сдвиг, переданное сообщение."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        self.l_shift = _mono("Комбинация A″(x), циклически сдвинутая влево на — разрядов", 12)
        self.r_shifted = ResultRow("→", 24)
        self.l_sum = _mono("Сумма циклически сдвинутой комбинации A″(x) и полученного остатка:", 12)
        self.r_sum = ResultRow("→", 24)
        self.l_back = _mono("Наконец, проведем обратный циклический сдвиг этой суммы вправо на "
                            "— разрядов.", 12)
        self.l_done = _mono("Вы получили переданное сообщение:", 13)
        self.val = _mono("—", 14)
        self.val.setWordWrap(True)
        self.val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.verdict = _mono("", 12)
        for w in (self.l_shift, self.r_shifted, self.l_sum, self.r_sum, self.l_back,
                  self.l_done, self.val, self.verdict):
            lay.addWidget(w)
        lay.addStretch(1)

    def fill(self, res: CSResult, model: BCHModel) -> None:
        self.l_shift.setText("Комбинация A″(x), циклически сдвинутая влево на "
                             f"{res.shifts} разрядов")
        self.r_shifted.set(bits(res.shifted, model.N))
        self.l_sum.setText("Сумма циклически сдвинутой комбинации A″(x) и полученного "
                           f"остатка {bits(res.remainder)}:")
        self.r_sum.set(bits(res.summed, model.N))
        self.l_back.setText("Наконец, проведем обратный циклический сдвиг этой суммы вправо на "
                            f"{res.shifts} разрядов.")
        if res.ok:
            self.val.setText(f"{bits(res.corrected, model.N)}  =  {fmt_poly(res.corrected)}")
        else:
            self.val.setText("исправить не удалось")
        verdict = self.page.verdict()
        self.verdict.setText(verdict)

    def clear(self) -> None:
        self.val.setText("—")
        self.verdict.setText("")


class FieldPanel(QWidget):
    """Поле GF(2^m) по примитивному многочлену: степени, многочлены, коды."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        self.top = _mono("Поле GF(2^m) по полиному: —", 13)
        lay.addWidget(self.top)
        body = QHBoxLayout()
        self.tables = []
        for _ in range(3):
            t = QTableWidget(0, 2)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            t.verticalHeader().setVisible(False)
            body.addWidget(t)
            self.tables.append(t)
        lay.addLayout(body, 1)
        for t, head in zip(self.tables, (("i", "α^i"), ("α^i", "многочлен"), ("α^i", "код"))):
            t.setHorizontalHeaderLabels(list(head))

    def fill(self, model: BCHModel) -> None:
        gf = model.gf
        self.top.setText(f"Поле GF(2^{gf.m}) по полиному: p(z) = {fmt_poly(gf.prim, 'z')} "
                         f"= {bits(gf.prim)}")
        rows = gf.table()
        for t in self.tables:
            t.setRowCount(len(rows))
        for i, (e, val, poly, code) in enumerate(rows):
            self.tables[0].setItem(i, 0, QTableWidgetItem(str(e)))
            self.tables[0].setItem(i, 1, QTableWidgetItem(f"α^{e}"))
            self.tables[1].setItem(i, 0, QTableWidgetItem(f"α^{e}"))
            self.tables[1].setItem(i, 1, QTableWidgetItem(poly))
            self.tables[2].setItem(i, 0, QTableWidgetItem(f"α^{e}"))
            self.tables[2].setItem(i, 1, QTableWidgetItem(code))


class HMatrixPanel(QWidget):
    """Матрица H: строки в степенях α и карта ненулевых элементов."""

    def __init__(self, page: "Lab7Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        lay.addWidget(_mono("Матрица H: строка i = (1, α^(i·C0), …, α^(i·C0·(N−1))), i = 1…2T", 12))
        self.memo = QPlainTextEdit()
        self.memo.setReadOnly(True)
        self.memo.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        lay.addWidget(self.memo, 1)
        self.canvas = Canvas(height=2.6)
        lay.addWidget(self.canvas, 1)

    def fill(self, model: BCHModel) -> None:
        gf = model.gf
        H = h_matrix(model)
        lines = []
        for i, row in enumerate(H, start=1):
            lines.append(f"{i}*C0: " + " ".join(gf.name(v) for v in row))
        self.memo.setPlainText("\n".join(lines))
        self.canvas.fig.clear()
        self.canvas.ax = self.canvas.fig.add_subplot(111)
        self.canvas._style()
        self.canvas.ax.set_title("Матрица H: показатели степеней α")
        self.canvas.ax.set_xlabel("разряд j")
        self.canvas.ax.set_ylabel("строка i")
        grid = np.array([[gf.log[v] for v in row] for row in H], dtype=float)
        im = self.canvas.ax.imshow(grid, aspect="auto", cmap="viridis")
        self.canvas.fig.colorbar(im, ax=self.canvas.ax, fraction=0.03, pad=0.01,
                                 label="log α")
        self.canvas.ax.set_yticks(range(len(H)))
        self.canvas.ax.set_yticklabels([f"{i}·C0" for i in range(1, 2 * model.T + 1)], fontsize=7)
        self.canvas.draw_idle()


# --- страница -----------------------------------------------------------------

class Lab7Page(QWidget):
    """Form8: две ветви формы, 4 + 9 вкладок, общее состояние модели."""

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.model: BCHModel = build_bch(21, 2)
        self.code: int | None = None
        self.received: int | None = None
        self.error_word: int | None = None
        self.cs_result: CSResult | None = None
        self.pgz_result: PGZResult | None = None

        self.build_tabs = QTabWidget()
        self.fix_tabs = QTabWidget()
        self.tabs.addTab(self.build_tabs, TABS[0])
        self.tabs.addTab(self.fix_tabs, TABS[1])

        self.inp = BuildInputPanel(self)
        self.table = TablePanel(self)
        self.matrix = MatrixPanel(self)
        self.fix_in = FixInputPanel(self)
        self.errvec = ErrorVectorPanel(self)
        self.cs = CSPanel(self)
        self.pgz = PGZPanel(self)
        self.results = ResultsPanel(self)
        self.field = FieldPanel(self)
        self.hmat = HMatrixPanel(self)

        self.build_tabs.addTab(self._theory(THEORY_BUILD), TABS_BUILD[0])
        self.build_tabs.addTab(self.inp, TABS_BUILD[1])
        self.build_tabs.addTab(self.table, TABS_BUILD[2])
        self.build_tabs.addTab(self.matrix, TABS_BUILD[3])

        self.fix_tabs.addTab(self._theory(THEORY_CS), TABS_FIX[0])
        self.fix_tabs.addTab(self._theory(THEORY_PGZ), TABS_FIX[1])
        self.fix_tabs.addTab(self.fix_in, TABS_FIX[2])
        self.fix_tabs.addTab(self.errvec, TABS_FIX[3])
        self.fix_tabs.addTab(self.cs, TABS_FIX[4])
        self.fix_tabs.addTab(self.pgz, TABS_FIX[5])
        self.fix_tabs.addTab(self.results, TABS_FIX[6])
        self.fix_tabs.addTab(self.field, TABS_FIX[7])
        self.fix_tabs.addTab(self.hmat, TABS_FIX[8])

        self.inp.cb_n.currentTextChanged.connect(self._on_params)
        self.inp.sp_t.valueChanged.connect(self._on_params)
        self.inp.btn_calc.clicked.connect(self._on_calc)

        self.apply_model(self.model)
        self.set_status("Готово: выберите N и T и нажмите «Вычислить»", ok=True)

    # --- сборка/состояние ---------------------------------------------------

    def _theory(self, html: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        view.setHtml(html)
        lay.addWidget(view)
        return w

    def _on_params(self) -> None:
        N, T = self.inp.params()
        if N != self.model.N or T != self.model.T:
            self.set_status("Параметры изменены: нажмите «Вычислить»", ok=True)

    def _on_calc(self) -> None:
        model = self.inp.compute()
        self.apply_model(model)

    def set_model(self, model: BCHModel) -> None:
        self.model = model
        self.code = None
        self.received = None
        self.error_word = None
        self.cs_result = None
        self.pgz_result = None

    def apply_model(self, model: BCHModel) -> None:
        """Перестроить все зависимые вкладки под модель."""
        self.model = model
        N = model.N
        for w in (self.fix_in.f_a, self.errvec.f_e):
            w.set_length(N)
            w.set_bits(0)
        self.fix_in.f_g.set_value(model.G if model.exists else 0)
        self.fix_in.sp_t.setValue(model.T if model.exists else 1)
        self.fix_in.r_info.clear()
        self.fix_in.r_check.clear()
        self.table.fill(model)
        self.matrix.fill(model)
        if model.gf is not None:
            self.field.fill(model)
            self.hmat.fill(model)
        if model.exists:
            self.set_status(f"Код БЧХ ({N}, {model.k}): G(x) = {bits(model.G)}", ok=True)
        else:
            self.set_status(model.error, ok=False)

    def set_generator(self, G: int) -> None:
        self.model.G = G
        self.fix_in.f_g.set_value(G)
        self.matrix.fill(self.model)

    def set_code(self, word: int) -> None:
        """Готовая кодовая комбинация → поля второй части."""
        self.code = word
        self.fix_in.f_a.set_bits(word)
        r = self.model.r
        if self.model.exists and 0 < r < self.model.N:
            self.fix_in.r_info.set(bits(word >> r, self.model.N - r))
            self.fix_in.r_check.set(bits(word & ((1 << r) - 1), r))
        self.received = None
        self.error_word = None
        self.results.clear()

    def set_received(self, word: int, error: int) -> None:
        self.received = word
        self.error_word = error
        self.cs_result = None
        self.pgz_result = None

    def run_cs(self) -> CSResult | None:
        if self.received is None:
            self.set_status("Сначала введите E(x) и нажмите «Передать»", ok=False)
            return None
        self.cs_result = cs_decode(self.received, self.model.G, self.model.N,
                                   min(self.fix_in.sp_t.value(), self.model.T))
        self.set_status(
            f"CS: W = {self.cs_result.cycles[-1].weight}, сдвигов {self.cs_result.shifts}"
            if self.cs_result.ok else "CS: остаток с весом W ≤ T не найден", ok=self.cs_result.ok)
        return self.cs_result

    def run_pgz(self) -> PGZResult | None:
        if self.received is None:
            self.set_status("Сначала введите E(x) и нажмите «Передать»", ok=False)
            return None
        self.pgz_result = pgz_decode(self.received, self.model)
        self.results.verdict.setText(self.verdict())
        self.set_status(
            "PGZ: ошибки в разрядах " + ", ".join(str(p) for p in self.pgz_result.positions)
            if self.pgz_result.ok else "PGZ: ошибка обнаружена, но не исправлена",
            ok=self.pgz_result.ok)
        return self.pgz_result

    def fill_results(self, res: CSResult) -> None:
        self.results.fill(res, self.model)

    def verdict(self) -> str:
        """Вердикт по последнему исправлению."""
        if self.received is None:
            return ""
        corrected = None
        if self.cs_result is not None and self.cs_result.ok:
            corrected = self.cs_result.corrected
        elif self.pgz_result is not None and self.pgz_result.ok:
            corrected = self.pgz_result.corrected
        if corrected is None:
            return "Обнаружено: ошибка есть, исправить не удалось"
        if not self.error_word:
            return "Ошибок не обнаружено: синдромы нулевые"
        if self.code is not None and corrected == self.code:
            return "Обнаружено и исправлено: получена исходная комбинация"
        return "Обнаружено и исправлено, но комбинация отличается от исходной"

    def set_status(self, text: str, ok: bool = True) -> None:
        self.status.setText(text)
        bg = "#f4f4f4" if ok else "#fce8e6"
        self.status.setStyleSheet(f"padding:4px; background:{bg};")
