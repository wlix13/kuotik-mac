"""ЛР5: сжатие двоичных n-блоков — форма Form6.

Методы: КДС (длины серий), АПК (адресно-позиционное), ИПП (итерации простых
подстановок, Блох), УК (универсальное, Бабкин). Поля и числа повторяют исходную
форму: N — длина блока, K — число единиц, L — длина слова, P(K) = K/N, Kc = N/L.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..plot import Canvas

MAX_N = 64
SENS_N = 12                          # полный перебор расположений единиц до этой длины
SENS_MAX = 600                       # предел числа точек на графике чувствительности
METHODS = ("apk", "kds", "ipp", "uk")
TITLES = {"apk": "АПК", "kds": "КДС", "ipp": "ИПП", "uk": "УК"}
LONG = {
    "apk": "МЕТОД  АДРЕСНО-ПОЗИЦИОННОГО  КОДИРОВАНИЯ",
    "kds": "МЕТОД  КОДИРОВАНИЯ  ДЛИН  СЕРИЙ",
    "ipp": "МЕТОД  ИТЕРАЦИЙ  ПРОСТЫХ  ПОДСТАНОВОК  (автор Блох)",
    "uk": "МЕТОД  УНИВЕРСАЛЬНОГО  КОДИРОВАНИЯ  (автор Бабкин)",
}
FIELDS = {
    "apk": ("N", "K", "L(N)", "P(K)", "Kc"),
    "kds": ("N", "K", "L(N)", "P(K)", "Kc"),
    "ipp": ("N", "K", "P(K)", "Kc"),
    "uk": ("N", "K", "B(K)", "L(N,K)", "P(K)", "Kc"),
}
COLORS = {"apk": "#1a73e8", "kds": "#b3261e", "ipp": "#188038", "uk": "#8e24aa"}
HINT = {
    "apk": "Передаются позиции единиц (расстояние от начала блока), каждая — L бит.",
    "kds": "Передаются длины серий нулей перед каждой единицей и хвост блока, каждая — L бит.",
    "ipp": "Подстановки 00→0, 01→11, 1→10 повторяются, пока слово укорачивается.",
    "uk": "Передаётся пара (K, b): число единиц и номер блока по биномиальным коэффициентам.",
}

DEFAULT_BLOCK = "01101101"
BLOCKS = {1: "01101101", 2: "10101010"}

THEORY = """<h3>Рациональное кодирование двоичного источника</h3>
<p>Источник порождает независимые символы 1 и 0 с вероятностями p и 1−p. Поток режется на
n-блоки, каждому блоку сопоставляется кодовое слово из L двоичных символов. Энтропия двоичного
источника H(p) = −p·log₂p − (1−p)·log₂(1−p) бит/символ; при неизвестной статистике её оценивают
по частоте единиц в блоке: p = K/N.</p>
<p>Длина слова на символ L/N, избыточность R = L/N − H, эффективность η = H·N/L,
коэффициент сжатия K<sub>сж</sub> = N/L. При полностью известной статистике достижимо
K<sub>сж</sub> = 1/H (граница Шеннона); методы с частично или полностью неизвестной статистикой
лежат ниже этой границы.</p>
<p>Классификация обратимых методов: статистика известна полностью — Шеннон–Фано, Хаффман;
известна частично — КДС, АПК, ИПП; неизвестна — универсальное кодирование (УК).</p>

<h4>1. КДС — кодирование длин серий</h4>
<p>Высоковероятный символ — нуль. Блок разбивается на серии нулей; серия длины s = 1…n получает
префиксный код 0<sup>s−1</sup>1 длины s, а серия n+1 (блок из одних нулей) — код 0 длины 1.
В форме длина серии передаётся равномерным кодом из L = ⌈log₂(n+1)⌉ бит: для блока с K единицами
передаются K длин (расстояние до первой единицы и промежутки между единицами) и, если блок не
кончается единицей, длина хвоста + 1. Число серий t = K (блок кончается единицей) или K+1,
длина слова L·t, K<sub>сж</sub> = N/(L·t). Блок из одних нулей — одна серия со значением 0, L бит.</p>

<h4>2. АПК — адресно-позиционное кодирование</h4>
<p>Передаются не высоковероятные символы, а позиции редких единиц относительно начала блока.
Позиции нумеруются 1…n и кодируются равномерно L = ⌈log₂(n+1)⌉ битами. Слово — K чисел по L бит,
длина L·K, K<sub>сж</sub> = N/(L·K). Длина слова зависит только от K, не от расположения единиц.</p>

<h4>3. ИПП — итерации простых подстановок (Блох)</h4>
<p>К строке многократно применяется подстановка 00 → 0, 01 → 11, 1 → 10 (одиночный нуль в конце
строки остаётся нулём). Итерации идут, пока длина строки убывает или строка меняется при той же
длине; при удлинении сохраняется предыдущий результат, всего не более 10 итераций. Длина слова —
длина результата, K<sub>сж</sub> = N/L<sub>слов</sub>. Метод сжимает блоки с длинными сериями нулей
и чувствителен к расположению единиц: 00…011…1 и 1…10…0 дают разные длины.</p>

<h4>4. УК — универсальное кодирование (Бабкин)</h4>
<p>В n-блоке K единиц на позициях i₁ &lt; i₂ &lt; … &lt; i<sub>K</sub>. Номер блока</p>
<p style="margin-left:18px">b = Σ<sub>j=1…K</sub> C(i<sub>j</sub>−1, j),&nbsp;&nbsp;
0 ≤ b ≤ C(n,K)−1,&nbsp;&nbsp; C(n,m) = 0 при n &lt; m.</p>
<p>Соответствие блоков и пар (K, b) взаимно однозначно. Слово — K в l₁ = ⌈log₂(n+1)⌉ битах,
затем b в l₂ = ⌈log₂C(n,K)⌉ битах (при K = 0, K = n или n = 1 полагают l₂ = 0):
L = l₁ + l₂, K<sub>сж</sub> = N/L. Декодирование: в столбце j = K ищут наибольшее C(i−1, j) ≤ b,
вычитают его и переходят к j−1.</p>

<h4>Пример из справки: n = 8, блок 01101101</h4>
<p>K = 5, единицы на позициях 2, 3, 5, 6, 8; L = ⌈log₂9⌉ = 4.</p>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Метод</th><th>Слово</th><th>L</th><th>K<sub>сж</sub></th></tr>
<tr><td>АПК</td><td>0010 0011 0101 0110 1000</td><td>20</td><td>0.400</td></tr>
<tr><td>КДС</td><td>0010 0001 0010 0001 0010 (серии 2,1,2,1,2)</td><td>20</td><td>0.400</td></tr>
<tr><td>ИПП</td><td>01101101 (подстановка даёт 1110111011 — длиннее, результат сохранён)</td>
<td>8</td><td>1.000</td></tr>
<tr><td>УК</td><td>0101 100000</td><td>10</td><td>0.800</td></tr>
</table>
<p>УК: b = C(1,1)+C(2,2)+C(4,3)+C(5,4)+C(7,5) = 1+1+4+5+21 = 32;
L = 4 + ⌈log₂C(8,5)⌉ = 4 + ⌈log₂56⌉ = 4 + 6 = 10. Декодирование 32: 32−21 = 11, 11−5 = 6,
6−4 = 2, 2−1 = 1, 1−1 = 0 — позиции 8, 6, 5, 3, 2.</p>

<h4>Оценка длины для n = 100 (табл. 5.2 справки)</h4>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>K</th><th>1</th><th>2</th><th>3</th><th>4</th><th>10</th><th>20</th></tr>
<tr><td>l₁+l₂</td><td>7+7</td><td>7+13</td><td>7+18</td><td>7+23</td><td>7+45</td><td>7+70</td></tr>
<tr><td>K<sub>сж</sub></td><td>7.14</td><td>5.00</td><td>4.00</td><td>3.33</td><td>1.92</td><td>1.30</td></tr>
</table>
<p>Точный расчёт ⌈log₂C(100,K)⌉ для K = 4, 10, 20 даёт 22, 44, 69 (на единицу меньше табличных) —
в методичке округление выполнено с запасом; для K = 1, 2, 3 таблица совпадает точно.</p>

<h4>Порядок выполнения</h4>
<ol>
<li>Выбрать n-блок небольшой длины, сжать его методами КДС, АПК, ИПП и УК.</li>
<li>Проверить декодирование каждого слова — оно должно вернуть исходный блок.</li>
<li>Сопоставить L, K<sub>сж</sub>, избыточность и эффективность методов.</li>
<li>Построить семейства K<sub>сж</sub>(p) при двух фиксированных n.</li>
<li>Исследовать чувствительность методов к расположению единиц при фиксированной частоте p.</li>
<li>Сравнить K<sub>сж</sub> с границей 1/H(p) и объяснить проигрыш методов с неполной статистикой.</li>
</ol>"""


# --- математика ---------------------------------------------------------------

def ceil_log2(m: int) -> int:
    """⌈log₂ m⌉ точно, без плавающей точки."""
    return (m - 1).bit_length() if m > 1 else 0


def unit_len(n: int) -> int:
    """L = ⌈log₂(n+1)⌉ — длина позиции/серии в битах, как в форме."""
    return ceil_log2(n + 1)


def bin_str(x: int, width: int) -> str:
    """Двоичная запись x ровно в width бит (width = 0 — пустая строка)."""
    return format(x, f"0{width}b") if width > 0 else ""


def bin_entropy(p: float) -> float:
    """H(p) = −p·log₂p − (1−p)·log₂(1−p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def positions(block: str) -> list[int]:
    """Позиции единиц, нумерация с 1."""
    return [i + 1 for i, c in enumerate(block) if c == "1"]


def check_block(block: str) -> str:
    """Валидация n-блока; возвращает текст ошибки или пустую строку."""
    if not block:
        return "Не выбран исследуемый блок!"
    if any(c not in "01" for c in block):
        return "Допустимы только символы 0 и 1"
    if len(block) > MAX_N:
        return f"Неверная длина блока: {len(block)} > {MAX_N}"
    return ""


# --- методы -------------------------------------------------------------------

def apk_encode(block: str) -> tuple[str, int, list[int]]:
    """АПК: позиции единиц по L бит. Возвращает (слово, L, позиции)."""
    n = len(block)
    pos = positions(block)
    unit = unit_len(n)
    if not pos:
        return bin_str(0, unit), unit, []
    return "".join(bin_str(p, unit) for p in pos), unit, pos


def apk_decode(word: str, unit: int, n: int, k: int) -> str:
    """Обратное АПК: k позиций по unit бит."""
    if unit <= 0 or len(word) != unit * max(k, 1):
        return ""
    vals = [int(word[i:i + unit], 2) for i in range(0, len(word), unit)]
    if k == 0:
        return "0" * n if vals == [0] else ""
    if any(not 1 <= v <= n for v in vals) or any(b <= a for a, b in zip(vals, vals[1:])):
        return ""
    out = ["0"] * n
    for v in vals:
        out[v - 1] = "1"
    return "".join(out)


def kds_series(block: str) -> list[int]:
    """Длины серий: до первой единицы, промежутки, хвост + 1 (0 — блок из нулей)."""
    n = len(block)
    pos = positions(block)
    if not pos:
        return [0]
    vals = [pos[0]] + [pos[i] - pos[i - 1] for i in range(1, len(pos))]
    if pos[-1] != n:
        vals.append(n - pos[-1] + 1)
    return vals


def kds_encode(block: str) -> tuple[str, int, list[int]]:
    """КДС: длины серий по L бит. Возвращает (слово, L, серии)."""
    unit = unit_len(len(block))
    vals = kds_series(block)
    return "".join(bin_str(v, unit) for v in vals), unit, vals


def kds_decode(word: str, unit: int, n: int, k: int) -> str:
    """Обратное КДС при известном числе единиц k."""
    if unit <= 0 or len(word) % unit:
        return ""
    vals = [int(word[i:i + unit], 2) for i in range(0, len(word), unit)]
    if k == 0:
        return "0" * n if vals == [0] else ""
    if len(vals) < k:
        return ""
    out = ["0"] * n
    cur = 0
    for v in vals[:k]:
        if v <= 0:
            return ""
        cur += v
        if cur > n:
            return ""
        out[cur - 1] = "1"
    tail = vals[k:]
    if tail:
        if len(tail) != 1 or cur + tail[0] - 1 != n:
            return ""
    elif cur != n:
        return ""
    return "".join(out)


def ipp_step(s: str) -> str:
    """Одна подстановка: 00→0, 01→11, 1→10; одиночный нуль в конце — 0."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == "1":
            out.append("10")
            i += 1
        elif i == n - 1:
            out.append("0")
            i += 1
        elif s[i + 1] == "0":
            out.append("0")
            i += 2
        else:
            out.append("11")
            i += 2
    return "".join(out)


def ipp_encode(block: str) -> tuple[str, list[str]]:
    """ИПП: итерации до прекращения сжатия (не более 10). Возвращает (слово, шаги)."""
    s = block
    trace = [s]
    steps = 0
    while True:
        r = ipp_step(s)
        if len(r) < len(s):
            s = r
            steps += 1
        elif len(r) == len(s):
            if r == s:
                break
            s = r
            steps += 1
        else:
            break
        trace.append(s)
        if steps >= 10:
            break
    return s, trace


def _ipp_rev(r: str, target: int) -> str:
    """Обратный шаг подстановки при известной длине результата target."""
    out: list[str] = []
    i, n = 0, len(r)
    while i < n:
        if r[i] == "1":
            if i + 1 >= n:
                return ""
            out.append("01" if r[i + 1] == "1" else "1")
            i += 2
        else:
            out.append("00")
            i += 1
    total = sum(len(x) for x in out)
    if total == target:
        return "".join(out)
    if total == target + 1 and out and out[-1] == "00" and r.endswith("0"):
        out[-1] = "0"
        return "".join(out) if sum(len(x) for x in out) == target else ""
    return ""


def ipp_decode(word: str, trace: list[str]) -> str:
    """Обратный проход по шагам; длины шагов берутся из трассы кодирования."""
    s = word
    for target in reversed([len(t) for t in trace[:-1]]):
        s = _ipp_rev(s, target)
        if not s:
            return ""
    return s


def uk_number(block: str) -> int:
    """b = Σ C(i_j − 1, j) по позициям единиц."""
    return sum(math.comb(p - 1, j) for j, p in enumerate(positions(block), 1))


def uk_encode(block: str) -> tuple[str, int, int, int]:
    """УК: (слово, l1, l2, b)."""
    n = len(block)
    k = block.count("1")
    l1 = unit_len(n)
    l2 = 0 if (n == 1 or k == 0 or k == n) else ceil_log2(math.comb(n, k))
    b = uk_number(block)
    return bin_str(k, l1) + bin_str(b, l2), l1, l2, b


def uk_decode(word: str, n: int, l1: int, l2: int) -> str:
    """Обратное УК: номер b разлагается по биномиальным коэффициентам."""
    if l1 <= 0 or len(word) != l1 + l2:
        return ""
    k = int(word[:l1], 2)
    if k > n:
        return ""
    b = int(word[l1:], 2) if l2 else 0
    if k == 0:
        return "0" * n if b == 0 else ""
    if b >= math.comb(n, k):
        return ""
    out = ["0"] * n
    for j in range(k, 0, -1):
        i = j
        while i + 1 <= n and math.comb(i, j) <= b:
            i += 1
        if math.comb(i - 1, j) > b:
            return ""
        out[i - 1] = "1"
        b -= math.comb(i - 1, j)
    return "".join(out) if b == 0 else ""


@dataclass
class Code:
    """Результат сжатия одного блока одним методом."""
    method: str
    block: str
    word: str
    l_unit: int = 0
    extra: dict = field(default_factory=dict)
    decoded: str = ""
    trace: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.block)

    @property
    def k(self) -> int:
        return self.block.count("1")

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def length(self) -> int:
        return len(self.word)

    @property
    def kc(self) -> float:
        return self.n / self.length if self.length else 0.0

    @property
    def entropy(self) -> float:
        return bin_entropy(self.p)

    @property
    def redundancy(self) -> float:
        return self.length / self.n - self.entropy if self.n else 0.0

    @property
    def efficiency(self) -> float:
        return self.entropy * self.n / self.length if self.length else 0.0

    @property
    def ok(self) -> bool:
        return bool(self.decoded) and self.decoded == self.block

    @property
    def limit_kc(self) -> float:
        """Граница Шеннона Kсж = 1/H(p) для двоичного источника."""
        h = self.entropy
        return math.inf if h <= 0 else 1.0 / h

    @property
    def class_kc(self) -> float:
        """Предел для класса блоков с данным K: n/log₂C(n,K) (энтропия класса)."""
        c = math.comb(self.n, self.k)
        return math.inf if c <= 1 else self.n / math.log2(c)


def encode(method: str, block: str) -> Code:
    """Сжатие блока одним методом + проверка декодирования."""
    n, k = len(block), block.count("1")
    if method == "apk":
        word, unit, _ = apk_encode(block)
        code = Code(method, block, word, unit, {"parts": max(k, 1)})
        code.decoded = apk_decode(word, unit, n, k)
    elif method == "kds":
        word, unit, series = kds_encode(block)
        code = Code(method, block, word, unit, {"series": series, "parts": len(series)})
        code.decoded = kds_decode(word, unit, n, k)
    elif method == "ipp":
        word, trace = ipp_encode(block)
        code = Code(method, block, word, 0, {"steps": len(trace) - 1}, trace=trace)
        code.decoded = ipp_decode(word, trace)
    elif method == "uk":
        word, l1, l2, b = uk_encode(block)
        code = Code(method, block, word, l1, {"l1": l1, "l2": l2, "b": b})
        code.decoded = uk_decode(word, n, l1, l2)
    else:
        raise ValueError(method)
    return code


def encode_all(block: str) -> dict[str, Code]:
    return {m: encode(m, block) for m in METHODS}


def expected_length(method: str, block: str) -> int:
    """Предельная длина слова по формуле метода (КДС, АПК, УК — точно)."""
    n, k = len(block), block.count("1")
    if method == "apk":
        return unit_len(n) * max(k, 1)
    if method == "kds":
        return unit_len(n) * len(kds_series(block))
    if method == "uk":
        _, l1, l2, _ = uk_encode(block)
        return l1 + l2
    if method == "ipp":
        return len(ipp_encode(block)[0])
    raise ValueError(method)


def group_word(code: Code) -> str:
    """Слово с пробелами: по частям метода (АПК/КДС/УК), у ИПП — по 8 бит."""
    w = code.word
    if code.method == "ipp":
        return " ".join(w[i:i + 8] for i in range(0, len(w), 8))
    if code.method == "uk":
        l1, l2 = code.extra["l1"], code.extra["l2"]
        return (w[:l1] + (" " + w[l1:l1 + l2] if l2 else "")).strip()
    unit = code.l_unit or len(w)
    return " ".join(w[i:i + unit] for i in range(0, len(w), unit))


def code_report(code: Code) -> str:
    """Текст «Результат сжатия» — как Memo в форме."""
    n, k = code.n, code.k
    lines = [f"Исходный N-блок : {code.block}"]
    if code.method == "apk":
        pos = " ".join(str(p) for p in positions(code.block)) or "—"
        lines.append(f"Позиции единиц  : {pos}")
        lines.append(f"Кодовое слово   : {group_word(code)}")
        lines.append(f"Длина слова     : {code.length} бит ({code.extra['parts']} × {code.l_unit})")
    elif code.method == "kds":
        ser = " ".join(str(v) for v in code.extra["series"])
        lines.append(f"Длины серий     : {ser}")
        lines.append(f"Кодовое слово   : {group_word(code)}")
        lines.append(f"Длина слова     : {code.length} бит ({code.extra['parts']} × {code.l_unit})")
    elif code.method == "ipp":
        lines.append(f"Кодовое слово   : {group_word(code)}")
        lines.append(f"Длина слова     : {code.length} бит (N = {n}, итераций {code.extra['steps']})")
    else:
        l1, l2 = code.extra["l1"], code.extra["l2"]
        lines.append(f"Номер блока B(K): {code.extra['b']}  (0…C(N,K)−1 = {math.comb(n, k) - 1})")
        lines.append(f"Кодовое слово   : {group_word(code)}")
        lines.append(f"Длина слова     : {code.length} бит ({l1} + {l2})")
    lines.append(f"K = {k}, P(K) = {code.p:.4f}, Kсж = {code.kc:.3f}, "
                 f"R = {code.redundancy:+.3f} бит/симв, η = {code.efficiency:.3f}")
    lines.append("Граница 1/H(p)  : "
                 + ("∞" if math.isinf(code.limit_kc) else f"{code.limit_kc:.3f}"))
    lines.append("Предел n/log₂C  : "
                 + ("∞" if math.isinf(code.class_kc) else f"{code.class_kc:.3f}"))
    mark = "совпало" if code.ok else "не совпало"
    lines.append(f"Декодирование   : {code.decoded or '—'} — {mark}")
    return "\n".join(lines)


# --- интерфейс ----------------------------------------------------------------

def _mono(text: str = "", size: int = 12) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-family:Menlo,Consolas,monospace; font-size:{size}px;")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


class MethodPanel(QWidget):
    """Вкладка одного метода: параметры, слово, проверка декодирования."""

    def __init__(self, page: "Lab5Page", method: str):
        super().__init__()
        self.page = page
        self.method = method
        lay = QVBoxLayout(self)

        head = QLabel(LONG[method])
        head.setAlignment(Qt.AlignCenter)
        head.setStyleSheet("font-weight:bold; font-size:15px; padding:4px;")
        lay.addWidget(head)

        hint = QLabel(HINT[method])
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#444;")
        lay.addWidget(hint)

        body = QHBoxLayout()
        left = QGroupBox("Основные параметры :")
        grid = QGridLayout(left)
        self.vals: dict[str, QLabel] = {}
        for row, name in enumerate(FIELDS[method]):
            grid.addWidget(QLabel(name), row, 0)
            val = _mono("0")
            self.vals[name] = val
            grid.addWidget(val, row, 1)
        grid.setRowStretch(len(FIELDS[method]), 1)

        right = QGroupBox("Результат сжатия")
        rl = QVBoxLayout(right)
        self.memo = QPlainTextEdit()
        self.memo.setReadOnly(True)
        self.memo.setStyleSheet("font-family:Menlo,Consolas,monospace; font-size:12px;")
        rl.addWidget(self.memo, 1)

        body.addWidget(left, 1)
        body.addWidget(right, 2)
        lay.addLayout(body, 1)

        btns = QHBoxLayout()
        self.btn_encode = QPushButton("Кодировать")
        self.btn_back = QPushButton("Вернуться")
        self.btn_results = QPushButton("Результаты")
        self.btn_graph = QPushButton("График")
        for b in (self.btn_encode, self.btn_back, self.btn_results, self.btn_graph):
            b.setMinimumHeight(30)
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.verdict = QLabel("")
        self.verdict.setStyleSheet("padding:3px;")
        lay.addWidget(self.verdict)

        self.btn_encode.clicked.connect(page.encode_all)
        self.btn_results.clicked.connect(page.show_results)
        self.btn_back.clicked.connect(lambda: page.tabs.setCurrentIndex(1))
        self.btn_graph.clicked.connect(lambda: page.tabs.setCurrentIndex(6))

    def show_code(self, code: Code) -> None:
        v = {
            "N": str(code.n),
            "K": str(code.k),
            "L(N)": str(code.l_unit),
            "L(N,K)": str(code.length),
            "B(K)": str(code.extra.get("b", 0)),
            "P(K)": f"{code.p:.4f}",
            "Kc": f"{code.kc:.3f}",
        }
        for name, lbl in self.vals.items():
            lbl.setText(v.get(name, "—"))
        self.memo.setPlainText(code_report(code))
        if code.ok:
            self.verdict.setText("Декодирование совпало с исходным блоком")
            self.verdict.setStyleSheet("padding:3px; background:#e6f4ea; color:#137333;")
        else:
            self.verdict.setText("Декодирование НЕ совпало с исходным блоком")
            self.verdict.setStyleSheet("padding:3px; background:#fce8e6; color:#c5221f;")

    def clear(self) -> None:
        for lbl in self.vals.values():
            lbl.setText("0")
        self.memo.clear()
        self.verdict.setText("")


class Lab5Page(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.results: dict[str, Code] = {}
        self.blocks = dict(BLOCKS)
        self.active = 1

        self.panels = {m: MethodPanel(self, m) for m in METHODS}
        self.apk, self.kds, self.ipp, self.uk = (self.panels[m] for m in METHODS)

        self.tabs.addTab(self._theory_tab(), "Общие сведения")
        self.tabs.addTab(self._input_tab(), "Ввод данных")
        self.tabs.addTab(self.panels["apk"], "АПК")
        self.tabs.addTab(self.panels["kds"], "КДС")
        self.tabs.addTab(self.panels["ipp"], "ИПП")
        self.tabs.addTab(self.panels["uk"], "УК")
        self.tabs.addTab(self._graphs_tab(), "Графики")

        self.edit.setText(DEFAULT_BLOCK)
        self.set_status("Готово: введите n-блок и нажмите «Кодировать»", ok=True)
        self.encode_all()

    # --- вкладки ---------------------------------------------------------------

    def _theory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        view.setHtml(THEORY)
        lay.addWidget(view)
        return w

    def _input_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        title = QLabel("Выберите исследуемый блок")
        title.setStyleSheet("font-weight:bold; font-size:14px;")
        lay.addWidget(title)

        row = QHBoxLayout()
        self.rb1 = QPushButton("Блок №1")
        self.rb2 = QPushButton("Блок №2")
        self.btn_reset_block = QPushButton("Сброс блока")
        self.btn_reset_all = QPushButton("Сброс всей информации")
        for b in (self.rb1, self.rb2, self.btn_reset_block, self.btn_reset_all):
            b.setMinimumHeight(30)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        ruler = QLabel("".join(f"{i:<10}" for i in range(1, 65, 10)))
        ruler.setStyleSheet("font-family:Menlo,Consolas,monospace; font-size:18px; color:#666;"
                            " padding-left:8px;")
        lay.addWidget(ruler)
        ticks = QLabel(("|" + "·" * 9) * 7 + "|")
        ticks.setStyleSheet("font-family:Menlo,Consolas,monospace; font-size:18px; color:#999;"
                            " padding-left:8px;")
        lay.addWidget(ticks)

        self.edit = QLineEdit()
        self.edit.setMaxLength(MAX_N)
        self.edit.setValidator(QRegularExpressionValidator(QRegularExpression("[01]*")))
        self.edit.setStyleSheet("font-family:Menlo,Consolas,monospace; font-size:18px; padding:6px;")
        self.edit.setPlaceholderText("например 01101101")
        lay.addWidget(self.edit)

        # линейка позиций тем же шрифтом, что и строка блока
        for lbl in (ruler, ticks):
            lbl.setFont(self.edit.font())
            lbl.setStyleSheet("color:#777; padding:0px;")
        self.edit.setTextMargins(3, 0, 0, 0)

        self.btn_encode = QPushButton("Кодировать всеми методами")
        self.btn_encode.setMinimumHeight(32)
        lay.addWidget(self.btn_encode)

        self.lbl_info = _mono("")
        lay.addWidget(self.lbl_info)

        hint = QLabel("Длина блока 1…64, только символы 0 и 1. Единицы — редкий символ, нули — "
                      "высоковероятный. «Блок №1» и «Блок №2» загружают сохранённые блоки.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        lay.addWidget(hint)
        lay.addStretch(1)

        self.rb1.clicked.connect(lambda: self.load_block(1))
        self.rb2.clicked.connect(lambda: self.load_block(2))
        self.btn_reset_block.clicked.connect(self.reset_block)
        self.btn_reset_all.clicked.connect(self.reset_all)
        self.btn_encode.clicked.connect(self.encode_all)
        self.edit.returnPressed.connect(self.encode_all)
        return w

    def _graphs_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        ctl = QHBoxLayout()
        self.n1 = QSpinBox()
        self.n1.setRange(2, 32)
        self.n1.setValue(8)
        self.n2 = QSpinBox()
        self.n2.setRange(2, 32)
        self.n2.setValue(16)
        self.k_sens = QSpinBox()
        self.k_sens.setRange(0, 8)
        self.k_sens.setValue(4)
        self.btn_draw = QPushButton("Построить")
        ctl.addWidget(QLabel("n семейства"))
        ctl.addWidget(self.n1)
        ctl.addWidget(QLabel("второе n"))
        ctl.addWidget(self.n2)
        ctl.addWidget(QLabel("K для чувствительности"))
        ctl.addWidget(self.k_sens)
        ctl.addWidget(self.btn_draw)
        ctl.addStretch(1)
        lay.addLayout(ctl)

        self.cv_curves = Canvas(3.0)
        self.cv_curves.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cv_sens = Canvas(2.8)
        self.cv_sens.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.cv_curves, 1)
        lay.addWidget(self.cv_sens, 1)

        self.tbl_cmp = QTableWidget(0, 7)
        self.tbl_cmp.setHorizontalHeaderLabels(["Метод", "L, бит", "Kсж", "R, бит/симв", "η",
                                                "1/H(p)", "n/log₂C"])
        self.tbl_cmp.verticalHeader().setVisible(False)
        self.tbl_cmp.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_cmp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_cmp.setMaximumHeight(140)
        lay.addWidget(self.tbl_cmp)

        self.n1.valueChanged.connect(self._n_changed)
        self.n2.valueChanged.connect(self.draw_graphs)
        self.k_sens.valueChanged.connect(self.draw_graphs)
        self.btn_draw.clicked.connect(self.draw_graphs)
        return w

    # --- данные ----------------------------------------------------------------

    def block(self) -> str:
        return self.edit.text().strip()

    def set_block(self, text: str) -> dict[str, Code]:
        self.edit.setText(text)
        return self.encode_all()

    def load_block(self, idx: int) -> None:
        self.active = idx
        self.edit.setText(self.blocks.get(idx, DEFAULT_BLOCK))
        self.encode_all()

    def set_status(self, text: str, ok: bool = True) -> None:
        self.status.setText(text)
        bg = "#f4f4f4" if ok else "#fce8e6"
        self.status.setStyleSheet(f"padding:4px; background:{bg};")

    def reset_block(self) -> None:
        self.edit.clear()
        self.results = {}
        for panel in self.panels.values():
            panel.clear()
        self.lbl_info.setText("")
        self.tbl_cmp.setRowCount(0)
        self.set_status("Блок сброшен", ok=True)

    def reset_all(self) -> None:
        self.blocks = dict(BLOCKS)
        self.active = 1
        self.reset_block()
        self.set_status("Накопленные данные уничтожены !", ok=True)

    def encode_all(self) -> dict[str, Code]:
        text = self.block()
        err = check_block(text)
        if err:
            self.results = {}
            for panel in self.panels.values():
                panel.clear()
            self.lbl_info.setText("")
            self.tbl_cmp.setRowCount(0)
            self.set_status(err, ok=False)
            return {}
        self.blocks[self.active] = text
        self.results = encode_all(text)
        n, k = len(text), text.count("1")
        self.lbl_info.setText(
            f"N = {n}, K = {k}, p = {k / n:.4f}, L = ⌈log₂(N+1)⌉ = {unit_len(n)}")
        for m, code in self.results.items():
            self.panels[m].show_code(code)
        bad = [TITLES[m] for m, c in self.results.items() if not c.ok]
        head = (f"n = {n}, k = {k}, p = {k / n:.4f}; "
                + "; ".join(f"{TITLES[m]} {c.length} бит, Kсж = {c.kc:.3f}"
                            for m, c in self.results.items()))
        tail = "декодирование совпало" if not bad else "не совпало: " + ", ".join(bad)
        self.set_status(f"{head}; {tail}", ok=not bad)
        self.fill_table()
        self.draw_graphs()
        return self.results

    def show_results(self) -> None:
        if not self.results:
            self.set_status("Нет результатов: сначала закодируйте блок", ok=False)
            return
        best = min(self.results.values(), key=lambda c: c.length)
        lim = "∞" if math.isinf(best.limit_kc) else f"{best.limit_kc:.3f}"
        self.set_status(f"Лучший метод: {TITLES[best.method]}, L = {best.length} бит, "
                        f"Kсж = {best.kc:.3f}; граница 1/H = {lim}", ok=True)

    def fill_table(self) -> None:
        self.tbl_cmp.setRowCount(len(self.results))
        for r, (m, c) in enumerate(self.results.items()):
            cells = [TITLES[m], str(c.length), f"{c.kc:.3f}", f"{c.redundancy:+.3f}",
                     f"{c.efficiency:.3f}",
                     "∞" if math.isinf(c.limit_kc) else f"{c.limit_kc:.3f}",
                     "∞" if math.isinf(c.class_kc) else f"{c.class_kc:.3f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_cmp.setItem(r, col, item)

    # --- графики ---------------------------------------------------------------

    def _n_changed(self) -> None:
        self.k_sens.setMaximum(self.n1.value())
        self.draw_graphs()

    @staticmethod
    def canonical_block(n: int, k: int) -> str:
        """Канонический блок: k единиц, затем нули — хвост серии учитывается."""
        return "1" * k + "0" * (n - k)

    def family(self, method: str, n: int) -> tuple[list[float], list[float]]:
        """Kсж(p) при фиксированном n; p = 0 (блок из нулей) — вырожденный случай."""
        xs, ys = [], []
        for k in range(1, n + 1):
            code = encode(method, self.canonical_block(n, k))
            xs.append(k / n)
            ys.append(code.kc)
        return xs, ys

    def draw_graphs(self) -> None:
        if not hasattr(self, "cv_curves"):
            return
        n1, n2 = self.n1.value(), self.n2.value()
        cv = self.cv_curves
        cv.clear(f"Семейства Kсж(p): n = {n1} и n = {n2} (единицы в начале блока, p > 0)",
                 "p = K/N", "Kсж = N/L")
        ps = np.linspace(0.001, 0.999, 240)
        cv.ax.plot(ps, [1.0 / bin_entropy(float(p)) for p in ps], color="#666", linewidth=2.0,
                   linestyle=":", label="граница 1/H(p)")
        top = 0.0
        for method in METHODS:
            x1, y1 = self.family(method, n1)
            cv.ax.plot(x1, y1, color=COLORS[method], linewidth=1.8,
                       label=f"{TITLES[method]}, n={n1}")
            x2, y2 = self.family(method, n2)
            cv.ax.plot(x2, y2, color=COLORS[method], linewidth=1.0, linestyle="--", alpha=0.75,
                       label=f"{TITLES[method]}, n={n2}")
            top = max(top, max(y1), max(y2))
        cv.ax.set_xlim(0.0, 1.0)
        cv.ax.set_ylim(0, min(max(2.0, 1.06 * top), 8.0))
        cv.ax.legend(fontsize=6.5, ncol=3, loc="upper right")
        cv.draw_idle()

        n = min(n1, SENS_N)
        k = min(self.k_sens.value(), n)
        cv2 = self.cv_sens
        cv2.clear(f"Чувствительность к расположению единиц: n = {n}, K = {k}",
                  "номер расположения", "Kсж")
        blocks = self.arrangements(n, k)
        xs = np.arange(len(blocks))
        values: list[float] = []
        for method in METHODS:
            ys = [encode(method, b).kc for b in blocks]
            values.extend(ys)
            cv2.ax.plot(xs, ys, color=COLORS[method], linewidth=1.4, marker="o", markersize=2.5,
                        label=TITLES[method])
        cv2.ax.set_xlim(-0.5, max(1.5, len(blocks) - 0.5))
        lo, hi = min(values), max(values)
        pad = max(0.08 * (hi - lo), 0.05 * max(abs(hi), 1e-9))
        cv2.ax.set_ylim(max(0.0, lo - pad), hi + pad)
        cv2.ax.legend(fontsize=8, loc="lower right")
        cv2.draw_idle()

    @staticmethod
    def arrangements(n: int, k: int) -> list[str]:
        """Все расположения k единиц; при большом C(n,K) — равномерная выборка."""
        total = math.comb(n, k)
        if total <= SENS_MAX:
            out = []
            for ones in itertools.combinations(range(n), k):
                row = ["0"] * n
                for i in ones:
                    row[i] = "1"
                out.append("".join(row))
            return out
        rng = np.random.default_rng(5)
        seen: set[str] = set()
        while len(seen) < SENS_MAX:
            row = ["0"] * n
            for i in rng.choice(n, size=k, replace=False):
                row[int(i)] = "1"
            seen.add("".join(row))
        return sorted(seen)


def compare_blocks(block_a: str, block_b: str) -> list[str]:
    """Длины слов двух блоков одной длины и K — для отчёта по чувствительности."""
    out = []
    for m in METHODS:
        ca, cb = encode(m, block_a), encode(m, block_b)
        out.append(f"{TITLES[m]}: {ca.length} / {cb.length} бит")
    return out
