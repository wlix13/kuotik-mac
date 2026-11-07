from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QFont, QRegularExpressionValidator
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
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..plot import Canvas
from ..theory import apply_theory

TABS = ("Общие сведения", "Ввод данных", "Описание модели",
        "Модель кодера", "Модель декодера")

CHANNELS = (1, 2, 3, 4, 5)

MSG_ZERO = "Код принят без ошибок!"
MSG_ONE = ("Код принят с ошибкой! Для передачи кода потребителю и исправления "
           "одиночной ошибки необходимо продолжить процесс декодирования.")
MSG_DONE = "Процесс декодирования завершен!"
MSG_CHECK = ("Код принят! Состояние триггеров соответствует контрольным разрядам. "
             "Теперь необходимо выдвинуть контрольные разряды. Обратите внимание на "
             "состояние ключей.")
MSG_FORMED = "Информационные и контрольные разряды сформированы!"
MSG_NEED_INFO = "Введите информационные разряды: ожидаются двоичные символы 0/1"
MSG_NEED_ERR = "Введите вектор ошибки: ожидаются двоичные символы 0/1"
MSG_MULTI = ("Синдром не нулевой — ошибка обнаружена. Кратность ошибки больше "
             "единицы: код Хэмминга её только обнаруживает")
MSG_MISCORR = ("Синдром указывает на разряд {pos}, но исправленная комбинация "
               "не совпадает с переданной — ошибка кратности больше единицы")

THEORY = """<h3>Циклические коды Хэмминга</h3>
<p>Кодирование и декодирование циклических кодов выполняют многотактные линейные
схемы — фильтры (сдвигающие регистры с обратными связями). Основные элементы —
задержка на один такт и сумматор по модулю 2.</p>

<h4>Матрица связей</h4>
<p>Фильтр из r элементов задержки описывается матрицей связей M = ‖m<sub>ij</sub>‖,
где m<sub>ij</sub> = 1, если выход i-го элемента задержки связан со входом j-го.
Вектор-столбцам M отвечают полиномы G<sub>j</sub>(x); если G<sub>j</sub>(x) кратны
порождающему многочлену g(x), схема делит входной полином на g(x).</p>
<p>Состояние фильтра в следующем такте: s(t+1) = M<sup>T</sup>·s(t). Входная цепь
строится так, что s(t+1) = M<sup>T</sup>·s(t) + b(t)·e, то есть к содержимому
фильтра прибавляется входной символ и выполняется один сдвиг. После поступления
всех символов a(x) содержимое фильтра равно остатку a(x) mod g(x).</p>

<h4>Кодер</h4>
<p>Ключ в 1-м положении: информационные разряды идут на вход фильтра и в линию.
Затем ключ переключается во 2-е положение: обратная связь размыкается, контрольные
разряды выдвигаются в линию, и к концу передачи фильтр обнуляется.</p>
<p>Разделимый (систематический) код: A(x) = u(x)·x<sup>n−k</sup> + R(x), где
R(x) = [u(x)·x<sup>n−k</sup>] mod g(x). Многоканальный (v-канальный) кодер содержит
v входов и v выходов; чтобы остаток не требовал коррекции, информационный полином
умножают на x<sup>pad</sup>, pad = (−k) mod v — перед младшим информационным
разрядом приписывают pad фиктивных нулей.</p>

<h4>Декодер</h4>
<p>Принятая комбинация A″(x) делится на g(x). Нулевой остаток — ошибок нет
(«Код принят правильно, если [A<sub>i</sub>(x)·x<sup>n−k</sup> + R<sub>i</sub>(x)]
= 0 mod G(x)»). Ненулевой остаток — обнаруживаемая ошибка. При автономной работе
(вход нулевой) состояния фильтра равны x<sup>t</sup>·s(x) mod g(x), поэтому вычет
вида ошибки воспроизводится точно через n тактов. Для одиночной ошибки в разряде p
синдром равен x<sup>p</sup> mod g(x); логическая схема (схема совпадения) узнаёт
этот вычет и инвертирует разряд.</p>
<p>В v-канальном декодере логическая схема имеет v выходов: на каждом такте
автономной работы выход канала распознаёт свой вычет. Порог мажоритарного элемента
— ⌊v/2⌋+1 совпадений.</p>

<h4>Свойства кода</h4>
<p>g(x) должен делить x<sup>n</sup>+1 без остатка и иметь g(0) = 1. Для кода
Хэмминга n = 2<sup>r</sup>−1, k = n−r, d<sub>min</sub> = 3: одиночные ошибки
исправляются, двойные обнаруживаются. Код совершенный: все 2<sup>r</sup>−1
ненулевых синдромов заняты одиночными ошибками, поэтому двойная ошибка даёт
синдром одного из разрядов и «исправляется» в другое кодовое слово — порт
показывает это сравнением с переданной комбинацией.</p>

<h4>Порядок выполнения</h4>
<ol>
<li>Выберите число каналов v и порождающий многочлен g(x).</li>
<li>Введите информационные разряды и нажмите «Рассчитать остаток».</li>
<li>На вкладке «Модель кодера» пройдите такты кнопкой «Следующий такт».</li>
<li>Введите вектор ошибки, нажмите «Декодировать», пройдите такты декодера.</li>
<li>Сверьте синдром с таблицей порогов и убедитесь в исправлении разряда.</li>
</ol>
"""


@dataclass(frozen=True)
class CodeParams:

    n: int
    k: int
    g: int
    title: str

    @property
    def r(self) -> int:
        return self.n - self.k


POLYNOMIALS: tuple[CodeParams, ...] = (
    CodeParams(3, 1, 0b111, "(3,1)     x²+x+1"),
    CodeParams(7, 4, 0b1011, "(7,4)     x³+x+1"),
    CodeParams(7, 4, 0b1101, "(7,4)     x³+x²+1"),
    CodeParams(15, 11, 0b10011, "(15,11)   x⁴+x+1"),
    CodeParams(15, 11, 0b11001, "(15,11)   x⁴+x³+1"),
    CodeParams(31, 26, 0b100101, "(31,26)   x⁵+x²+1"),
    CodeParams(31, 26, 0b101001, "(31,26)   x⁵+x³+1"),
)


def poly_deg(p: int) -> int:
    return p.bit_length() - 1


def poly_mod(a: int, g: int) -> int:
    r = poly_deg(g)
    if g == 0:
        raise ValueError("нулевой порождающий многочлен")
    while a and poly_deg(a) >= r:
        a ^= g << (poly_deg(a) - r)
    return a


def bits(value: int, width: int | None = None) -> str:
    if width is None:
        width = max(1, value.bit_length())
    return format(value, f"0{width}b")[-width:] if width else ""


def fmt_poly(p: int, var: str = "x") -> str:
    if p == 0:
        return "0"
    out: list[str] = []
    for e in range(poly_deg(p), -1, -1):
        if not (p >> e) & 1:
            continue
        if e == 0:
            out.append("1")
        elif e == 1:
            out.append(var)
        else:
            out.append(f"{var}<sup>{e}</sup>")
    return "+".join(out)


def parse_bits(text: str) -> int | None:
    text = text.strip()
    if not text or any(c not in "01" for c in text):
        return None
    return int(text, 2)


def rot(value: int, length: int, shift: int) -> int:
    shift %= length
    if shift == 0:
        return value
    mask = (1 << length) - 1
    return ((value << shift) | (value >> (length - shift))) & mask


class CyclicCode:

    def __init__(self, params: CodeParams):
        self.p = params
        self.n = params.n
        self.k = params.k
        self.r = params.r
        self.g = params.g
        self.gtail = self.g & ((1 << self.r) - 1)


    def pad(self, channels: int = 1) -> int:
        return (-self.k) % channels if channels > 1 else 0

    def length(self, channels: int = 1) -> int:
        return self.n

    def clocks(self, channels: int = 1) -> int:
        return self.pad(channels) + self.n

    def threshold(self, channels: int = 1) -> int:
        return channels // 2 + 1


    def check_bits(self, message: int, channels: int = 1) -> int:
        return poly_mod(message << self.r, self.g)

    def encode(self, message: int, channels: int = 1) -> int:
        return (message << self.r) | poly_mod(message << self.r, self.g)

    def info_part(self, word: int, channels: int = 1) -> int:
        return word >> self.r

    def check_part(self, word: int, channels: int = 1) -> int:
        return word & ((1 << self.r) - 1)


    def syndrome(self, word: int) -> int:
        return poly_mod(word, self.g)

    def error_table(self, channels: int = 1) -> list[tuple[int, int]]:
        return [(p, poly_mod(1 << p, self.g)) for p in range(self.length(channels))]

    def syndromes_by_value(self, channels: int = 1) -> dict[int, int]:
        return {s: p for p, s in self.error_table(channels)}

    def trace(self, word: int, channels: int = 1) -> list[int]:
        length = self.length(channels)
        rem = 0
        out: list[int] = [0] * self.pad(channels)
        for t in range(length):
            b = (word >> (length - 1 - t)) & 1
            rem = (rem << 1) | b
            if rem >> self.r:
                rem ^= self.g
            out.append(rem)
        return out

    def autonomous(self, syndrome: int, clocks: int | None = None,
                   channels: int = 1) -> list[int]:
        clocks = self.clocks(channels) if clocks is None else clocks
        out: list[int] = []
        s = syndrome
        for _ in range(clocks):
            out.append(s)
            s <<= 1
            if s >> self.r:
                s ^= self.g
        return out

    def thresholds(self, channels: int = 1) -> list[tuple[int, int, int, int]]:
        rows: list[tuple[int, int, int, int]] = []
        for p, s in self.error_table(channels):
            rows.append((p % channels, p // channels, p, s))
        return rows

    def decode(self, word: int, channels: int = 1,
               truth: int | None = None) -> "DecodeResult":
        length = self.length(channels)
        syn = self.syndrome(word)
        table = self.syndromes_by_value(channels)
        threshold = self.threshold(channels)
        if syn == 0:
            return DecodeResult(syndrome=syn, clean=True, position=None, corrected=word,
                                length=length, votes=0, threshold=threshold,
                                failed=False, miscorrected=False, channel=None,
                                clock=None)

        votes: dict[int, int] = {}
        for c in range(channels):
            s_c = self.syndrome(rot(word, length, c))
            p_c = table.get(s_c)
            if p_c is None:
                continue
            pos = (p_c - c) % length
            votes[pos] = votes.get(pos, 0) + 1
        best = max(votes, key=lambda p: (votes[p], -p)) if votes else None
        nvotes = votes.get(best, 0) if best is not None else 0

        if best is None or nvotes < threshold:
            return DecodeResult(syndrome=syn, clean=False, position=None,
                                corrected=word, length=length, votes=nvotes,
                                threshold=threshold, failed=True,
                                miscorrected=False, channel=None, clock=None)

        corrected = word ^ (1 << best)
        miscorrected = self.syndrome(corrected) != 0
        if truth is not None and corrected != truth:
            miscorrected = True
        return DecodeResult(syndrome=syn, clean=False, position=best,
                            corrected=corrected, length=length, votes=nvotes,
                            threshold=threshold, failed=False,
                            miscorrected=miscorrected, channel=best % channels,
                            clock=best // channels)


@dataclass
class DecodeResult:

    syndrome: int
    clean: bool
    position: int | None
    corrected: int
    length: int
    votes: int
    threshold: int
    failed: bool
    miscorrected: bool
    channel: int | None
    clock: int | None

    @property
    def pos_num(self) -> int | None:
        return None if self.position is None else self.position + 1


def _mono_font() -> QFont:
    font = QFont("Menlo")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(11)
    return font


def _mono(text: str = "", size: int = 12) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-family: Menlo, monospace; font-size: {size}px;")
    return lbl


class BitsEdit(QLineEdit):

    def __init__(self, length: int, value: int = 0, read_only: bool = False):
        super().__init__()
        self._len = length
        self.setFont(_mono_font())
        self._apply(length)
        self.setReadOnly(read_only)
        self.set_bits(value)

    def _apply(self, length: int) -> None:
        self.setMaxLength(length)
        self.setValidator(QRegularExpressionValidator(
            QRegularExpression(f"[01]{{0,{length}}}")))
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


class Ruler(QLabel):

    def __init__(self, length: int):
        super().__init__()
        self.setStyleSheet("font-family: Menlo, monospace; font-size: 10px; color: #555;")
        self.set_length(length)

    def set_length(self, length: int) -> None:
        parts = [str(i + 1).rjust(2) for i in range(length - 1, -1, -1)]
        self.setText(" ".join(parts))


class RegisterTable(QTableWidget):

    def __init__(self, headers: tuple[str, ...] = ("такт", "вход", "состояние", "порог")):
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(list(headers))
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;"
                           " QTableWidget { gridline-color: #ddd; }")

    def fill(self, rows: list[tuple[str, str, str, str]]) -> None:
        self.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                self.setItem(i, j, QTableWidgetItem(text))


class InputPanel(QWidget):

    def __init__(self, page: "Lab8Page"):
        super().__init__()
        self.page = page
        outer = QVBoxLayout(self)
        outer.addWidget(_mono(" Данные, необходимые для построения модели кодера и декодера", 13))

        top = QHBoxLayout()
        gb_ch = QGroupBox("Количество каналов")
        vlay = QVBoxLayout(gb_ch)
        self.rb_channels: list[QRadioButton] = []
        names = {1: "Одноканальное         устройство",
                 2: "Двухканальное          устройство",
                 3: "Трехканальное          устройство",
                 4: "Четырехканальное  устройство",
                 5: "Пятиканальное        устройство"}
        for v in CHANNELS:
            rb = QRadioButton(names[v])
            rb.setChecked(v == 1)
            vlay.addWidget(rb)
            self.rb_channels.append(rb)
            rb.toggled.connect(lambda on, n=v: on and page.set_channels(n))
        gb_ch.setLayout(vlay)
        top.addWidget(gb_ch)

        gb_poly = QGroupBox("Порождающий многочлен")
        play = QVBoxLayout(gb_poly)
        self.rb_poly: list[QRadioButton] = []
        for i, prm in enumerate(POLYNOMIALS):
            rb = QRadioButton(prm.title)
            rb.setChecked(i == 1)
            play.addWidget(rb)
            self.rb_poly.append(rb)
            rb.toggled.connect(lambda on, n=i: on and page.set_params(n))
        gb_poly.setLayout(play)
        top.addWidget(gb_poly)
        outer.addLayout(top)

        outer.addWidget(_mono(" Данные, необходимые для запуска  процесса кодирования "
                              "и декодирования", 13))
        gb_seq = QGroupBox("Передаваемая последовательность")
        grid = QGridLayout(gb_seq)
        self.l_info = _mono("Информационные разряды", 12)
        self.l_rem = _mono("Остаток", 12)
        self.l_err = _mono("Вектор ошибки", 12)
        self.ruler_info = Ruler(4)
        self.ruler_rem = Ruler(3)
        self.ruler_err = Ruler(7)
        self.f_info = BitsEdit(4)
        self.f_rem = BitsEdit(3)
        self.f_err = BitsEdit(7)
        self.btn_rem = QPushButton("Рассчитать остаток")
        self.btn_dec = QPushButton("Декодировать")
        grid.addWidget(self.l_info, 0, 0)
        grid.addWidget(self.l_rem, 0, 2)
        grid.addWidget(self.ruler_info, 1, 0)
        grid.addWidget(self.ruler_rem, 1, 2)
        grid.addWidget(self.f_info, 2, 0)
        grid.addWidget(self.f_rem, 2, 2)
        grid.addWidget(self.l_err, 3, 0)
        grid.addWidget(self.ruler_err, 4, 0)
        grid.addWidget(self.f_err, 5, 0)
        grid.addWidget(self.btn_rem, 2, 3)
        grid.addWidget(self.btn_dec, 5, 3)
        grid.setColumnStretch(0, 1)
        gb_seq.setLayout(grid)
        outer.addWidget(gb_seq)

        self.note = _mono("", 11)
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        outer.addStretch(1)

        self.btn_rem.clicked.connect(page.calc_remainder)
        self.btn_dec.clicked.connect(page.decode)

    def sync_lengths(self, code: CyclicCode, channels: int) -> None:
        self.f_info.set_length(code.k)
        self.f_rem.set_length(code.r)
        self.f_err.set_length(code.length(channels))
        self.ruler_info.set_length(code.k)
        self.ruler_rem.set_length(code.r)
        self.ruler_err.set_length(code.length(channels))


class ModelPanel(QWidget):

    def __init__(self, page: "Lab8Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        gb_g = QGroupBox("Порождающий многочлен")
        glay = QVBoxLayout(gb_g)
        self.formula = _mono("G(x) = 1 + G1·x + G2·x² + ... + Gr·x^r", 13)
        self.gtext = _mono("—", 12)
        self.gmask = _mono("—", 12)
        glay.addWidget(self.formula)
        glay.addWidget(self.gtext)
        glay.addWidget(self.gmask)
        gb_g.setLayout(glay)
        lay.addWidget(gb_g)

        gb_m = QGroupBox("Матрица связей элементов памяти фильтра")
        mlay = QVBoxLayout(gb_m)
        self.mhead = _mono("M =", 12)
        self.mat = QTableWidget(0, 0)
        self.mat.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mat.verticalHeader().setVisible(False)
        self.mat.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mat.setMaximumHeight(210)
        mlay.addWidget(self.mhead)
        mlay.addWidget(self.mat)
        gb_m.setLayout(mlay)
        lay.addWidget(gb_m)

        gb_f = QGroupBox("Связи входа и выхода")
        flay = QVBoxLayout(gb_f)
        self.ftext = QPlainTextEdit()
        self.ftext.setReadOnly(True)
        self.ftext.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        flay.addWidget(self.ftext)
        gb_f.setLayout(flay)
        lay.addWidget(gb_f, 1)

    def fill(self, code: CyclicCode, channels: int) -> None:
        g = code.g
        r = code.r
        terms = ["1"]
        for e in range(1, r + 1):
            if (g >> e) & 1:
                terms.append("x" if e == 1 else f"x^{e}")
        self.gtext.setText("G(x) = " + " + ".join(terms)
                           + f"   ({bits(g, r + 1)}, степень r = {r})")
        self.gmask.setText(f"Многочлен-маска g = {bits(g, r + 1)}, "
                           f"G(0) = {g & 1}, k = n − r = {code.k}")

        rows = []
        for i in range(r):
            row = []
            for j in range(r):
                if j == 0:
                    row.append((g >> i) & 1)
                else:
                    row.append(1 if j == i + 1 else 0)
            rows.append(row)
        self.mat.setRowCount(r)
        self.mat.setColumnCount(r)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.mat.setItem(i, j, item)
        self.mat.setVerticalHeaderLabels([f"T{i + 1}" for i in range(r)])

        lines = ["Матрица связей между входом и n-k элементами задержки фильтра:",
                 "F*  =  [ " + "  ".join(str((g >> i) & 1) for i in range(r)) + " ]",
                 "        (коэффициенты G1 … Gr, единица — обратная связь замкнута)", ""]
        for v in range(1, min(channels, 5) + 1):
            lines.append(f"Матрица связей элементов памяти фильтра "
                         f"в {v}-ой (v-ой) степени:")
            lines.append(self._mat_pow_text(rows, v))
            lines.append("")
        lines.append(f"Число каналов v = {channels}, порог мажоритария = "
                     f"{code.threshold(channels)} = ⌊v/2⌋+1")
        lines.append(f"Длина слова при v = {channels}: L = k + pad + r = "
                     f"{code.k} + {code.pad(channels)} + {code.r} = "
                     f"{code.length(channels)}")
        self.ftext.setPlainText("\n".join(lines))

    @staticmethod
    def _mat_pow_text(rows: list[list[int]], power: int) -> str:
        size = len(rows)
        m = [row[:] for row in rows]
        for _ in range(power - 1):
            m = [[sum(m[i][k] * rows[k][j] for k in range(size)) % 2
                  for j in range(size)] for i in range(size)]
        out = []
        for row in m:
            out.append("      " + "  ".join(str(v) for v in row))
        return "\n".join(out)


class CoderPanel(QWidget):

    def __init__(self, page: "Lab8Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.src_title = _mono("Исходный код", 12)
        self.dst_title = _mono("Полученный код", 12)
        row.addWidget(self.src_title)
        row.addStretch(1)
        row.addWidget(self.dst_title)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.src = _mono("—", 13)
        self.src.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dst = _mono("—", 13)
        self.dst.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row2.addWidget(self.src)
        row2.addStretch(1)
        row2.addWidget(self.dst)
        lay.addLayout(row2)

        self.formula = _mono("R i(x) = [A i(x) · x^(n−k)] mod G(x)", 12)
        lay.addWidget(self.formula)
        self.clock_lbl = _mono("Такт: 0 из 0", 12)
        lay.addWidget(self.clock_lbl)

        body = QHBoxLayout()
        self.tbl = RegisterTable(("такт", "вход", "состояние", "разряд"))
        body.addWidget(self.tbl, 1)
        right = QVBoxLayout()
        self.canvas = Canvas(2.4)
        right.addWidget(self.canvas)
        self.memo = QPlainTextEdit()
        self.memo.setReadOnly(True)
        self.memo.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        right.addWidget(self.memo)
        body.addLayout(right, 1)
        lay.addLayout(body, 1)

        self.state_lbl = _mono("Содержимое фильтра после поступления k информационных "
                               "разрядов: —", 12)
        self.state_lbl.setWordWrap(True)
        lay.addWidget(self.state_lbl)
        self.verdict = _mono("", 12)
        self.verdict.setWordWrap(True)
        lay.addWidget(self.verdict)

        self.btn_next = QPushButton("Следующий такт")
        lay.addWidget(self.btn_next)
        self.btn_next.clicked.connect(page.coder_next)

    def fill(self, code: CyclicCode, channels: int, message: int | None,
             word: int | None, states: list[int] | None) -> None:
        L = code.length(channels)
        self.src.setText("—" if message is None else bits(message, code.k))
        self.dst.setText("—" if word is None else bits(word, L))
        self.formula.setText(f"R i(x) = [A i(x) · x^{code.r + code.pad(channels)}] "
                             f"mod G(x)")
        total = L if states else 0
        self.clock_lbl.setText(f"Такт: {self.page.coder_step} из {total}")
        rows = []
        shown = states[: self.page.coder_step] if states else []
        pad = code.pad(channels)
        n_info = pad + code.k
        for t, st in enumerate(shown):
            if t < pad:
                rows.append((str(t + 1), "0", bits(st, code.r), "фикт."))
                continue
            tt = t - pad
            b = (word >> (L - 1 - tt)) & 1 if word is not None else 0
            kind = "инф." if t < n_info else "контр."
            rows.append((str(t + 1), str(b), bits(st, code.r), kind))
        self.tbl.fill(rows)
        memo = []
        for t, st in enumerate(shown):
            if t < pad:
                memo.append(f"такт {t + 1}: фиктивный нуль, состояние {bits(st, code.r)}")
            else:
                tt = t - pad
                b = (word >> (L - 1 - tt)) & 1 if word is not None else 0
                memo.append(f"такт {t + 1}: вход {b}, состояние {bits(st, code.r)}")
        if word is not None:
            memo.append("")
            memo.append(f"A(x)      = {bits(code.info_part(word, channels), code.k)}")
            memo.append(f"R(x)      = {bits(code.check_part(word, channels), code.r)}")
            memo.append(f"слово     = {bits(word, L)}")
            memo.append("ключ 2: контрольные разряды выдвинуты, фильтр обнулён"
                        if memo and shown and shown[-1] == 0 else "фильтр не обнулён")
        self.memo.setPlainText("\n".join(memo))

        if states:
            done = self.page.coder_step >= len(states)
            k_idx = min(pad + code.k, len(states)) - 1
            self.state_lbl.setText(
                "Содержимое фильтра после поступления k информационных разрядов: "
                + bits(states[k_idx], code.r)
                + (f"   (после всех тактов: {bits(states[-1], code.r)})" if done else ""))
            self.verdict.setText(MSG_FORMED if done else MSG_CHECK)
        else:
            self.state_lbl.setText("Содержимое фильтра после поступления k "
                                   "информационных разрядов: —")
            self.verdict.setText(self.page.coder_verdict or "")
        self.draw(code, states, shown)

    def draw(self, code: CyclicCode, states: list[int] | None,
             shown: list[int]) -> None:
        self.canvas.clear("Состояние регистра по тактам", "такт", "s(t)")
        if not shown:
            self.canvas.draw_idle()
            return
        x = list(range(1, len(shown) + 1))
        self.canvas.ax.step(x, shown, where="mid", color="#1a73e8", linewidth=1.8)
        self.canvas.ax.plot(x, shown, "o", color="#1a73e8", markersize=4)
        for xi, yi in zip(x, shown):
            self.canvas.ax.annotate(bits(yi, code.r), (xi, yi), textcoords="offset points",
                                    xytext=(0, 6), ha="center", fontsize=7)
        self.canvas.ax.set_xticks(x)
        self.canvas.ax.set_xlim(0.5, max(len(shown), 1) + 0.5)
        self.canvas.draw_idle()


class DecoderPanel(QWidget):

    def __init__(self, page: "Lab8Page"):
        super().__init__()
        self.page = page
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(_mono("Исходный код", 12))
        row.addStretch(1)
        row.addWidget(_mono("Полученный код", 12))
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.src = _mono("—", 13)
        self.src.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dst = _mono("—", 13)
        self.dst.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row2.addWidget(self.src)
        row2.addStretch(1)
        row2.addWidget(self.dst)
        lay.addLayout(row2)

        self.formula = _mono(" [A i(x) · x^(n−k) + R i(x)] = 0 mod G(x)", 12)
        lay.addWidget(self.formula)
        self.syn_lbl = _mono("Синдром S(x) = —", 12)
        lay.addWidget(self.syn_lbl)
        self.clock_lbl = _mono("Такт: 0 из 0", 12)
        lay.addWidget(self.clock_lbl)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(_mono("Состояния фильтра", 11))
        self.tbl = RegisterTable()
        left.addWidget(self.tbl, 1)
        body.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(_mono("Пороги схем совпадения", 11))
        self.thr = QTableWidget(0, 4)
        self.thr.setHorizontalHeaderLabels(["канал", "такт", "разряд", "порог"])
        self.thr.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.thr.verticalHeader().setVisible(False)
        self.thr.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thr.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        right.addWidget(self.thr, 1)
        self.canvas = Canvas(2.2)
        right.addWidget(self.canvas)
        body.addLayout(right, 1)
        lay.addLayout(body, 1)

        self.verdict = _mono(MSG_ZERO, 12)
        self.verdict.setWordWrap(True)
        lay.addWidget(self.verdict)
        self.btn_next = QPushButton("Следующий такт")
        lay.addWidget(self.btn_next)
        self.btn_next.clicked.connect(page.decoder_next)

    def fill(self, code: CyclicCode, channels: int, sent: int | None,
             received: int | None, res: DecodeResult | None,
             states: list[int] | None) -> None:
        L = code.length(channels)
        self.src.setText("—" if sent is None else bits(sent, L))
        self.dst.setText("—" if received is None else bits(received, L))
        self.formula.setText(f" [A i(x) · x^{code.r + code.pad(channels)} + R i(x)] "
                             f"= 0 mod G(x)")

        rows = []
        shown = states[: self.page.decoder_step] if states else []
        pad = code.pad(channels)
        for t, st in enumerate(shown):
            if t < pad:
                rows.append((str(t + 1), "0", bits(st, code.r), bits(0, code.r)))
                continue
            tt = t - pad
            b = (received >> (L - 1 - tt)) & 1 if received is not None else 0
            p = L - 1 - tt
            pattern = poly_mod(1 << p, code.g)
            mark = "==" if (res is not None and res.position == p) else ""
            rows.append((str(t + 1), str(b), bits(st, code.r),
                         bits(pattern, code.r) + mark))
        self.tbl.fill(rows)
        total = len(states) if states else 0
        self.clock_lbl.setText(f"Такт: {self.page.decoder_step} из {total}")

        if res is None:
            self.syn_lbl.setText("Синдром S(x) = —")
            self.thr.setRowCount(0)
            self.verdict.setText(self.page.decoder_verdict or "")
            self.canvas.clear("Синдром по тактам", "такт", "S(t)")
            self.canvas.draw_idle()
            return

        self.syn_lbl.setText(f"Синдром S(x) = {bits(res.syndrome, code.r)} = "
                             f"{res.syndrome} (разряд {res.pos_num or '—'})")
        self._fill_thresholds(code, channels)
        self.verdict.setText(self.verdict_text(res, channels))
        self._draw(res, code, shown)

    @staticmethod
    def verdict_text(res: DecodeResult, channels: int) -> str:
        if res.clean:
            return MSG_ZERO
        if res.failed:
            return MSG_MULTI
        if res.miscorrected:
            return MSG_MISCORR.format(pos=res.pos_num)
        return (f"{MSG_ONE} Исправлен разряд {res.pos_num} "
                f"(канал {(res.channel or 0) + 1}); голосов {res.votes} "
                f"из {channels}, порог {res.threshold}.")

    def _fill_thresholds(self, code: CyclicCode, channels: int) -> None:
        rows = code.thresholds(channels)
        self.thr.setRowCount(len(rows))
        for i, (ch, clk, p, s) in enumerate(rows):
            for j, text in enumerate((str(ch + 1), str(clk + 1), str(p + 1),
                                      bits(s, code.r))):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.thr.setItem(i, j, item)

    def _draw(self, res: DecodeResult, code: CyclicCode, shown: list[int]) -> None:
        self.canvas.clear("Синдром по тактам (автономный режим)", "такт", "S(t)")
        prog = self.page.syndrome_progress or []
        x = list(range(1, len(prog) + 1))
        if not prog:
            self.canvas.draw_idle()
            return
        self.canvas.ax.step(x, prog, where="mid", color="#b3261e", linewidth=1.8)
        self.canvas.ax.plot(x, prog, "o", color="#b3261e", markersize=4)
        if res.position is not None:
            t = code.pad(self.page.channels) + (res.length - 1 - res.position) % res.length
            if t < len(prog):
                self.canvas.ax.plot([t + 1], [prog[t]], "o", markersize=11,
                                    markerfacecolor="none",
                                    markeredgecolor="#188038", markeredgewidth=2)
                self.canvas.ax.annotate("порог", (t + 1, prog[t]),
                                        textcoords="offset points", xytext=(6, 6),
                                        fontsize=8, color="#188038")
        self.canvas.ax.set_xticks(x)
        self.canvas.ax.set_xlim(0.5, max(len(prog), 1) + 0.5)
        self.canvas.draw_idle()


class Lab8Page(QWidget):

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.status = QLabel("—")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.params: CodeParams = POLYNOMIALS[1]
        self.code = CyclicCode(self.params)
        self.channels = 1

        self.message: int | None = None
        self.word: int | None = None
        self.error: int | None = None
        self.received: int | None = None
        self.result: DecodeResult | None = None
        self.coder_states: list[int] | None = None
        self.decoder_states: list[int] | None = None
        self.syndrome_progress: list[int] | None = None
        self.coder_step = 0
        self.decoder_step = 0
        self.coder_verdict = ""
        self.decoder_verdict = ""

        self.inp = InputPanel(self)
        self.model = ModelPanel(self)
        self.coder = CoderPanel(self)
        self.decoder = DecoderPanel(self)

        self.tabs.addTab(self._theory(THEORY), TABS[0])
        self.tabs.addTab(self.inp, TABS[1])
        self.tabs.addTab(self.model, TABS[2])
        self.tabs.addTab(self.coder, TABS[3])
        self.tabs.addTab(self.decoder, TABS[4])

        self.apply_model()
        self.set_status(f"Код ({self.code.n}, {self.code.k}), g(x) = "
                        f"{bits(self.code.g)}: выберите каналы и многочлен, "
                        f"введите информационные разряды", ok=True)


    def _theory(self, html: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        apply_theory(view, "project8_1", html)
        lay.addWidget(view)
        return w

    def set_status(self, text: str, ok: bool = True) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color: {'#188038' if ok else '#b3261e'}; font-size: 12px; padding: 4px;")


    def set_params(self, index: int) -> None:
        params = POLYNOMIALS[index]
        if params == self.params:
            return
        self.params = params
        self.code = CyclicCode(params)
        self.reset()
        self.apply_model()
        self.set_status(f"Код ({params.n}, {params.k}), g(x) = {bits(params.g, params.r + 1)}"
                        f" — введите информационные разряды", ok=True)

    def set_channels(self, channels: int) -> None:
        if channels == self.channels:
            return
        self.channels = channels
        self.reset()
        self.apply_model()
        self.set_status(f"Каналов: {channels}, длина слова L = {self.code.length(channels)}, "
                        f"порог мажоритария {self.code.threshold(channels)}", ok=True)

    def reset(self) -> None:
        self.message = None
        self.word = None
        self.error = None
        self.received = None
        self.result = None
        self.coder_states = None
        self.decoder_states = None
        self.syndrome_progress = None
        self.coder_step = 0
        self.decoder_step = 0
        self.coder_verdict = ""
        self.decoder_verdict = ""

    def apply_model(self) -> None:
        code = self.code
        self.inp.sync_lengths(code, self.channels)
        self.inp.f_rem.set_bits(0)
        self.inp.f_err.set_bits(0)
        self.inp.note.setText(
            f"Код ({code.n}, {code.k}): r = n − k = {code.r}, d min = 3, "
            f"каналов {self.channels}, pad = {code.pad(self.channels)}, "
            f"длина слова L = {code.length(self.channels)}. "
            f"Многочлен g(x) = {bits(code.g, code.r + 1)} делит x^{code.n} + 1.")
        self.model.fill(code, self.channels)
        self.coder.fill(code, self.channels, self.message, self.word, self.coder_states)
        self.decoder.fill(code, self.channels, self.word, self.received, self.result,
                          self.decoder_states)


    def _read_info(self) -> int | None:
        text = self.inp.f_info.text().strip()
        value = parse_bits(text)
        if value is None:
            self.set_status(MSG_NEED_INFO, ok=False)
            return None
        return value

    def _read_error(self) -> int | None:
        text = self.inp.f_err.text().strip()
        if not text:
            return 0
        value = parse_bits(text)
        if value is None:
            self.set_status(MSG_NEED_ERR, ok=False)
            return None
        return value

    def calc_remainder(self) -> int | None:
        message = self._read_info()
        if message is None:
            return None
        code = self.code
        L = code.length(self.channels)
        if message >= (1 << code.k):
            self.set_status(MSG_NEED_INFO, ok=False)
            return None
        self.message = message
        rem = code.check_bits(message, self.channels)
        self.word = code.encode(message, self.channels)
        self.inp.f_rem.set_length(code.r)
        self.inp.f_rem.set_bits(rem)
        self.coder_states = code.trace(self.word, self.channels)
        self.coder_step = 0
        self.coder_verdict = ""
        self.received = None
        self.result = None
        self.decoder_states = None
        self.decoder_step = 0
        self.decoder_verdict = ""
        self.coder.fill(code, self.channels, self.message, self.word, self.coder_states)
        self.decoder.fill(code, self.channels, self.word, None, None, None)
        self.set_status(
            f"A(x) = {bits(message, code.k)}, R(x) = {bits(rem, code.r)} = {rem}, "
            f"кодовое слово {bits(self.word, L)} ({L} разрядов). "
            f"Синдром слова = {code.syndrome(self.word)}", ok=True)
        return rem

    def decode(self) -> DecodeResult | None:
        if self.word is None:
            if self.calc_remainder() is None:
                return None
        error = self._read_error()
        if error is None:
            return None
        code = self.code
        L = code.length(self.channels)
        if error >= (1 << L):
            self.set_status(MSG_NEED_ERR, ok=False)
            return None
        self.error = error
        self.received = self.word ^ error
        res = code.decode(self.received, self.channels, truth=self.word)
        self.result = res
        self.decoder_states = code.trace(self.received, self.channels)
        self.syndrome_progress = code.autonomous(res.syndrome, None, self.channels)
        self.decoder_step = 0
        self.decoder_verdict = ""
        self.decoder.fill(code, self.channels, self.word, self.received, res,
                          self.decoder_states)
        verdict = DecoderPanel.verdict_text(res, self.channels)
        ok = res.clean or (not res.failed and not res.miscorrected)
        self.set_status(verdict, ok=ok)
        return res


    def coder_next(self) -> None:
        if not self.coder_states:
            self.set_status("Сначала нажмите «Рассчитать остаток»", ok=False)
            return
        if self.coder_step < len(self.coder_states):
            self.coder_step += 1
        self.coder.fill(self.code, self.channels, self.message, self.word,
                        self.coder_states)
        self.set_status(f"Такт кодера {self.coder_step} из {len(self.coder_states)}: "
                        f"состояние {bits(self.coder_states[self.coder_step - 1], self.code.r)}",
                        ok=True)

    def decoder_next(self) -> None:
        if not self.decoder_states:
            self.set_status("Сначала нажмите «Декодировать»", ok=False)
            return
        if self.decoder_step < len(self.decoder_states):
            self.decoder_step += 1
        self.decoder.fill(self.code, self.channels, self.word, self.received,
                          self.result, self.decoder_states)
        res = self.result
        if res is not None and res.position is not None and \
                self.decoder_step - 1 >= res.position:
            self.set_status(f"{MSG_DONE} Исправлен разряд {res.pos_num}: "
                            f"{bits(res.corrected, res.length)}", ok=True)
        else:
            self.set_status(f"Такт декодера {self.decoder_step} из "
                            f"{len(self.decoder_states)}", ok=True)
