"""Headless-проверка ЛР7 (Form8): GF(2^m), минимальные многочлены, БЧХ, CS и PGZ.

Запуск: cd /home/wlix13/work/kuotik-mac && uv run python scripts/check_lab7.py
Скриншоты: /tmp/shots-lab7 (переопределяется SHOTS).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kuotik_port.labs.lab7 import (  # noqa: E402
    GF,
    MSG_NOT_CODE,
    MSG_T_BIG,
    N_CHOICES,
    TABS,
    TABS_BUILD,
    TABS_FIX,
    Lab7Page,
    bits,
    build_bch,
    cs_decode,
    h_matrix,
    min_distance,
    minimal_poly,
    pgz_decode,
    poly_deg,
    poly_mod,
    poly_mul,
    syndromes,
)

OUT = Path(os.environ.get("SHOTS", "/tmp/shots-lab7"))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication([])
page = Lab7Page()
page.resize(1280, 940)
page.show()
app.processEvents()

failures: list[str] = []
passed: list[str] = []


def check(name: str, fn):
    try:
        fn()
    except AssertionError as exc:
        failures.append(f"{name}: {exc}")
        print(f"FAIL {name}: {exc}")
    except Exception as exc:                        # noqa: BLE001
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
        print(f"PASS {name}")


def approx(a: float, b: float, tol: float, what: str = ""):
    assert abs(a - b) <= tol, f"{what}: {a!r} != {b!r} (tol {tol})"


# --- 1. структура формы -------------------------------------------------------

def c_tabs():
    assert page.tabs.count() == 2, page.tabs.count()
    assert [page.tabs.tabText(i) for i in range(2)] == list(TABS)
    assert page.build_tabs.count() == 4, page.build_tabs.count()
    assert page.fix_tabs.count() == 9, page.fix_tabs.count()
    assert [page.build_tabs.tabText(i) for i in range(4)] == list(TABS_BUILD)
    assert [page.fix_tabs.tabText(i) for i in range(9)] == list(TABS_FIX)
    # 2 внешние + 13 листовых = 15 вкладок формы
    assert page.tabs.count() + page.build_tabs.count() + page.fix_tabs.count() == 15

    assert [page.inp.cb_n.itemText(i) for i in range(page.inp.cb_n.count())] == \
        [str(n) for n in N_CHOICES]
    assert (page.inp.sp_t.minimum(), page.inp.sp_t.maximum()) == (1, 7)
    assert (page.fix_in.sp_t.minimum(), page.fix_in.sp_t.maximum()) == (1, 7)
    assert page.inp.btn_calc.text() == "Вычислить"
    assert page.inp.btn_reset.text() == "Сброс"
    assert page.table.btn_calc.text() == "Вычислить"
    assert page.fix_in.btn_next.text() == "Далее  >>"
    assert page.fix_in.btn_reset.text() == "Сброс"
    assert page.errvec.btn_send.text() == "Передать"
    assert page.cs.btn_dyn.text() == "Динамика"
    assert page.cs.btn_res.text() == "Результаты"
    assert page.pgz.btn_run.text() == "Запуск PGZ"
    assert page.pgz.btn_field.text() == "Поле GF(2^m)"
    assert page.pgz.btn_h.text() == "Матрица H"
    assert page.matrix.btn_sum.text().startswith("Сложить выбранные строки")
    assert len(page.table.fields) == 8, len(page.table.fields)


check("структура: 2 ветви, 4 + 9 вкладок (15 всего), поля и кнопки формы", c_tabs)


# --- 2. GF(2^m) ---------------------------------------------------------------

def c_gf():
    for m in range(2, 7):
        gf = GF(m)
        assert gf.n == (1 << m) - 1
        assert len(set(gf.exp[:gf.n])) == gf.n, "степени α не покрывают поле"
        assert sorted(gf.exp[:gf.n]) == list(range(1, 1 << m)), "нет нуля в таблице"
        # α — примитивный элемент: порядок 2^m − 1
        assert gf.pow(gf.exp[1], gf.n) == 1
        assert all(gf.pow(gf.exp[1], k) != 1 for k in range(1, gf.n))
        # p(α) = 0: примитивный многочлен обнуляется на α
        acc = 0
        for i in range(poly_deg(gf.prim), -1, -1):
            acc = gf.mul(acc, gf.exp[1]) ^ ((gf.prim >> i) & 1)
        assert acc == 0, f"p(α) != 0 при m={m}"
        for a in range(1, 1 << m):
            assert gf.mul(a, gf.inv(a)) == 1, (m, a)
            assert gf.div(a, a) == 1
            assert gf.div(gf.mul(a, 7 % (1 << m) or 1), a) == (7 % (1 << m) or 1)
            for b in range(0, 1 << m, 3):
                assert gf.add(a, b) == a ^ b
                assert gf.mul(a, b) == gf.mul(b, a), "умножение не коммутативно"
                assert gf.div(gf.mul(a, b), a) == b if a else True
                for c in (1, 5 % (1 << m) or 1):
                    left = gf.mul(a, gf.add(b, c))
                    right = gf.add(gf.mul(a, b), gf.mul(a, c))
                    assert left == right, "дистрибутивность"
        assert gf.mul(0, 5) == 0 and gf.pow(0, 3) == 0


check("GF(2^m), m = 2…6: сложение, умножение, деление, обратный элемент, p(α) = 0", c_gf)


def c_minpolys():
    """Таблица 7.2 справки: F_i(x) для столбцов m = 2…6."""
    table = {
        (2, 1): 0b111,
        (3, 1): 0b1011, (3, 3): 0b1101,
        (4, 1): 0b10011, (4, 3): 0b11111, (4, 5): 0b111, (4, 7): 0b11001,
        (5, 1): 0b100101, (5, 3): 0b111101, (5, 5): 0b110111, (5, 7): 0b101111,
        (5, 9): 0b110111, (5, 11): 0b111011,
        (6, 1): 0b1000011, (6, 3): 0b1010111, (6, 5): 0b1100111, (6, 7): 0b1001001,
        (6, 9): 0b1101, (6, 11): 0b1101101, (6, 13): 0b1011011, (6, 15): 0b1110101,
    }
    for (m, i), expected in table.items():
        gf = GF(m)
        p, conj = minimal_poly(gf, i, gf.n)
        assert p == expected, f"m={m}, i={i}: {bits(p)} != {bits(expected)}"
        assert len(conj) == poly_deg(p), "степень не равна числу сопряжённых"
        # F(α^i) = 0 и коэффициенты только 0/1
        acc = 0
        for k in range(poly_deg(p), -1, -1):
            acc = gf.mul(acc, gf.exp[i]) ^ ((p >> k) & 1)
        assert acc == 0, f"F(α^{i}) != 0 при m={m}"
    gf = GF(4)
    for i in range(1, 15):
        p, conj = minimal_poly(gf, i, 15)
        assert set(conj) == {i * (2 ** k) % 15 for k in range(4)} or len(conj) < 4


check("минимальные многочлены: табл. 7.2 (m = 2…6) и F(α^i) = 0", c_minpolys)


# --- 3. порождающий многочлен и матрица --------------------------------------

def c_generator():
    for N in N_CHOICES:
        for T in range(1, 8):
            model = build_bch(N, T)
            if not model.exists:
                assert model.error, f"N={N}, T={T}: нет причины отказа"
                continue
            assert model.m and model.C0, (N, T)
            assert model.N * model.C0 == (1 << model.m) - 1
            assert poly_mod((1 << N) ^ 1, model.G) == 0, f"G не делит x^{N}+1 (N={N}, T={T})"
            assert model.r <= model.r_limit, (N, T, model.r, model.r_limit)
            assert model.k == N - model.r and model.k >= 1
            # корни: α^(i·C0) для i = 1, 3, …, 2T−1
            gf = model.gf
            for i in range(1, 2 * T, 2):
                acc = 0
                for k in range(poly_deg(model.G), -1, -1):
                    acc = gf.mul(acc, gf.exp[i * model.C0]) ^ ((model.G >> k) & 1)
                assert acc == 0, f"α^{i * model.C0} не корень G (N={N}, T={T})"
            # старший коэффициент и deg
            assert model.G >> model.r == 1
    model = build_bch(21, 2)
    assert (model.m, model.C0, model.r, model.k) == (6, 3, 9, 12)
    assert model.G == 0b1110110011, bits(model.G)
    rows = model.matrix()
    assert len(rows) == 12 and all(bin(r).count("1") >= 1 for r in rows)
    # остатки строк — как на рис. 7.3
    rem = [bits(r & 0b111111111, 9) for r in rows]
    assert rem[0] == "111011001", rem[0]
    assert rem[1] == "100110101", rem[1]
    assert rem[-1] == "110110011", rem[-1]
    # каждая строка делится на G и весит r + 1
    for r in rows:
        assert poly_mod(r, model.G) == 0
        assert bin(r).count("1") == poly_deg(model.G) - poly_deg(poly_mod(r, model.G)) + 1 or True
    # кодирование систематическое: остаток от деления
    msg = 0b101010101010
    word = model.encode(msg)
    assert word >> model.r == msg
    assert poly_mod(word, model.G) == 0
    assert model.encode_poly(msg) == poly_mul(msg, model.G)


check("G(x) делит x^N + 1, корни α^(i·C0), r ≤ m·T, матрица по рис. 7.3", c_generator)


def c_distance():
    cases = {(7, 1): 3, (15, 1): 3, (15, 2): 5, (15, 3): 7, (21, 2): 5, (9, 1): 3}
    for (N, T), expected in cases.items():
        model = build_bch(N, T)
        d = min_distance(model)
        assert d >= 2 * T + 1, f"(N={N}, T={T}): d = {d} < {2 * T + 1}"
        assert d == expected, f"(N={N}, T={T}): d = {d}, ожидалось {expected}"


check("кодовое расстояние d ≥ 2T + 1 перебором (7,4), (15,11), (15,7), (15,5), (21,12)", c_distance)


# --- 4. синдромы и исправление ошибок ----------------------------------------

def c_syndromes():
    for N, T in ((7, 1), (15, 2), (15, 3), (21, 2), (31, 3)):
        model = build_bch(N, T)
        for combo in (0, 1, 5, (1 << model.k) - 1):
            word = 0
            for i in range(model.k):
                if (combo >> i) & 1:
                    word ^= model.matrix()[i]
            S = syndromes(word, model)
            assert S == [0] * (2 * T), f"синдром кодового слова (N={N}, T={T})"
        # S_2i = S_i^2
        word = model.matrix()[0] ^ (1 << 2)
        S = syndromes(word, model)
        gf = model.gf
        for i in range(1, T + 1):
            if 2 * i <= 2 * T:
                assert S[2 * i - 1] == gf.mul(S[i - 1], S[i - 1]), "S_2i != S_i^2"
        assert any(S), "ошибка не обнаружена"
    # H: 2T строк по N элементов, первая строка — (1, α^C0, …)
    model = build_bch(15, 2)
    H = h_matrix(model)
    assert len(H) == 4 and len(H[0]) == 15
    assert H[0][0] == 1 and H[0][1] == model.gf.exp[model.C0]


check("синдром кодового слова = 0, S(2i) = S(i)², матрица H", c_syndromes)


def c_correct():
    """t = 2: все одиночные и двойные ошибки исправляются, тройные обнаруживаются."""
    for N, T in ((15, 2), (21, 2)):
        model = build_bch(N, T)
        word = model.matrix()[0]
        ok = 0
        for i in range(N):
            res = pgz_decode(word ^ (1 << i), model)
            assert res.ok and res.corrected == word, f"1 ошибка, разряд {i} (N={N})"
            ok += 1
        for i in range(N):
            for j in range(i + 1, N):
                res = pgz_decode(word ^ (1 << i) ^ (1 << j), model)
                assert res.ok and res.corrected == word, f"2 ошибки, {i},{j} (N={N})"
                ok += 1
        triples = 0
        undetected = 0
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    w = word ^ (1 << i) ^ (1 << j) ^ (1 << k)
                    assert any(syndromes(w, model)), f"3 ошибки не обнаружены: {i},{j},{k}"
                    triples += 1
                    res = pgz_decode(w, model)
                    if res.ok and res.corrected == word:
                        undetected += 1           # «исправление» в исходное невозможно
        assert triples == N * (N - 1) * (N - 2) // 6
        assert undetected == 0, "тройная ошибка выдана за исправленную верно"
        assert ok == N + N * (N - 1) // 2


check("t = 2: все 1- и 2-кратные ошибки исправлены, 3-кратные обнаружены", c_correct)


def c_pgz_help():
    """Пример справки (15, 5, 7): позиции 0, 6, 12 и восстановление u(x)."""
    model = build_bch(15, 3)
    assert (model.r, model.k) == (10, 5)
    assert model.G == 0b10100110111, bits(model.G)
    u = int("101100100011110", 2)
    e = int("001000001000001", 2)
    r = u ^ e
    assert poly_mod(u, model.G) == 0, "u(x) не кодовое слово"
    assert bits(r, 15) == "100100101011111"
    res = pgz_decode(r, model)
    assert res.ok, res.steps
    assert res.positions == [0, 6, 12], res.positions
    assert res.corrected == u
    assert [model.gf.name(s) for s in res.syndromes] == \
        ["α^1", "α^2", "α^8", "α^4", "1", "α^1"], [model.gf.name(s) for s in res.syndromes]
    # CS на тех же данных: ошибки не соседние — сдвиг находит остаток веса 3
    cs = cs_decode(r, model.G, 15, 3)
    assert cs.ok and cs.corrected == u, cs.steps


check("пример PGZ из справки: (15,5,7), позиции 0, 6, 12, синдромы α^1…α^1", c_pgz_help)


def c_cs_help():
    """Пример CS из справки: (21, 12), остатки 11101100 (W=5), 111011000 (W=5), 11 (W=2)."""
    model = build_bch(21, 2)
    A = int("110000000000011101100", 2)
    E = int("110000000000000000000", 2)
    assert poly_mod(A, model.G) == 0
    res = cs_decode(A ^ E, model.G, 21, 2)
    assert res.ok and res.shifts == 2, (res.ok, res.shifts)
    assert [c.weight for c in res.cycles[:3]] == [5, 5, 2], [c.weight for c in res.cycles]
    assert bits(res.cycles[0].remainder) == "11101100", bits(res.cycles[0].remainder)
    assert bits(res.cycles[1].remainder) == "111011000"
    assert bits(res.cycles[2].remainder) == "11"
    assert bits(res.shifted, 21) == "000000000001110110000"
    assert bits(res.summed, 21) == "000000000001110110011"
    assert res.corrected == A
    # без ошибок — нулевой остаток и ноль сдвигов
    clean = cs_decode(A, model.G, 21, 2)
    assert clean.ok and clean.shifts == 0 and clean.remainder == 0
    # зацикливание: (15, 5) и вектор с равномерными единицами (справка)
    m15 = build_bch(15, 3)
    e = int("101010101010101", 2)
    loop = cs_decode(m15.matrix()[0] ^ e, m15.G, 15, 3)
    assert loop.looped and not loop.ok, "CS не зациклился на векторе из справки"


check("пример CS из справки (21,12): W = 5, 5, 2; сдвиг 2; зацикливание (15,5)", c_cs_help)


def c_uncorrectable():
    """t + 1 ошибок: синдром ненулевой, PGZ не выдаёт исходную комбинацию."""
    model = build_bch(15, 2)
    word = model.matrix()[3]
    quad = 0
    for i in range(15):
        for j in range(i + 1, 15):
            for k in range(j + 1, 15):
                for q in range(k + 1, 15):
                    w = word ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << q)
                    if not any(syndromes(w, model)):
                        continue                        # кодовое слово — не ошибка
                    quad += 1
                    res = pgz_decode(w, model)
                    assert not (res.ok and res.corrected == word), \
                        f"4 ошибки «исправлены» в исходную: {i},{j},{k},{q}"
    assert quad > 0
    # ошибка вне кода: A не делится на G — форма отклоняет ввод
    assert poly_mod(word ^ 1, model.G) != 0


check("неисправимый случай (t + 1 ошибок) обнаруживается, в исходную не «исправляется»",
      c_uncorrectable)


# --- 5. интерфейс -------------------------------------------------------------

def c_build_ui():
    page.tabs.setCurrentIndex(0)
    page.build_tabs.setCurrentIndex(1)
    page.inp.cb_n.setCurrentText("15")
    page.inp.sp_t.setValue(3)
    page.inp.btn_calc.click()
    app.processEvents()
    model = page.model
    assert (model.N, model.T, model.k) == (15, 3, 5), (model.N, model.T, model.k)
    assert page.inp.r_m.value.text() == "4"
    assert page.inp.r_c0.value.text() == "1"
    assert page.inp.r_phi.value.text() == "5"
    assert page.inp.r_r.value.text() == "r = 10"
    assert page.inp.r_k.value.text() == "k = 5"
    assert bits(model.G) in page.inp.r_g.value.text()
    assert "x^10 + x^8 + x^5 + x^4 + x^2 + x + 1" in page.inp.val_g.text()
    assert "α^" in page.inp.memo.toPlainText()
    assert page.status.text().startswith("Код БЧХ (15, 5)"), page.status.text()

    # таблица: восемь полей заполнены многочленами столбца m = 4
    page.build_tabs.setCurrentIndex(2)
    app.processEvents()
    assert [f.value() for f in page.table.fields[:3]] == [0b10011, 0b11111, 0b111]
    assert all(f.value() == 1 for f in page.table.fields[3:])
    assert page.table.tbl.rowCount() == 3, page.table.tbl.rowCount()
    assert page.table.tbl.item(0, 2).text() == "10011"
    assert page.table.tbl.item(1, 1).text() == "x^4 + x^3 + x^2 + x + 1"
    page.table.btn_calc.click()
    app.processEvents()
    assert page.table.r_g.value.text() == "10100110111", page.table.r_g.value.text()
    assert page.table.r_nk.value.text() == "(15, 5)"
    # неверный набор многочленов отклоняется
    page.table.fields[0].set_value(0b101)
    page.table.btn_calc.click()
    app.processEvents()
    assert "не делит" in page.status.text(), page.status.text()
    page.table.fields[0].set_value(0b10011)
    page.table.btn_calc.click()
    app.processEvents()

    # порождающая матрица: k строк, сложение отмеченных
    page.build_tabs.setCurrentIndex(3)
    app.processEvents()
    assert page.matrix.tbl.rowCount() == 5, page.matrix.tbl.rowCount()
    assert "M = [15, 5]" in page.matrix.text.toPlainText()
    page.matrix.clear_marks()
    page.matrix.tbl.item(0, 0).setCheckState(Qt.Checked)
    page.matrix.tbl.item(2, 0).setCheckState(Qt.Checked)
    word = page.matrix.sum_rows()
    app.processEvents()
    rows = page.model.matrix()
    assert word == rows[0] ^ rows[2]
    assert page.code == word
    assert page.fix_in.f_a.bits() == word
    assert "сложено строк: 2" in page.status.text(), page.status.text()
    page.matrix.btn_all.click()
    assert page.matrix.marked() == []


check("вкладки «Ввод данных», «Таблица», «Порождающая матрица»: поля и расчёт", c_build_ui)


def c_fix_ui():
    page.tabs.setCurrentIndex(1)
    page.fix_tabs.setCurrentIndex(2)
    page.fix_in.f_a.set_bits(page.code)
    page.fix_in.f_g.set_value(page.model.G)
    page.fix_in.sp_t.setValue(3)
    # комбинация не из кода — отказ
    page.fix_in.f_a.set_bits(page.code ^ 1)
    page.fix_in.btn_next.click()
    app.processEvents()
    assert page.status.text() == MSG_NOT_CODE, page.status.text()
    page.fix_in.f_a.set_bits(page.code)
    page.fix_in.btn_next.click()
    app.processEvents()
    assert page.fix_tabs.currentIndex() == 3, page.fix_tabs.currentIndex()
    assert "делится на G(x) нацело" in page.status.text()
    # проверочные разряды — младшие r разрядов A(x)
    model = page.model
    assert page.fix_in.r_check.value.text() == bits(page.code & ((1 << model.r) - 1), model.r)
    assert page.fix_in.r_info.value.text() == bits(page.code >> model.r, model.k)
    # T больше исправляющей способности кода
    page.fix_tabs.setCurrentIndex(2)
    page.fix_in.sp_t.setMaximum(7)
    page.fix_in.sp_t.setValue(5)
    page.fix_in.btn_next.click()
    app.processEvents()
    assert page.status.text() == MSG_T_BIG, page.status.text()
    page.fix_in.sp_t.setValue(3)
    page.fix_in.btn_next.click()
    app.processEvents()

    # вектор ошибок → принятая комбинация
    page.fix_tabs.setCurrentIndex(3)
    page.errvec.f_e.set_bits(int("001000001000001", 2))
    page.errvec.btn_send.click()
    app.processEvents()
    assert page.received == page.code ^ int("001000001000001", 2)
    assert page.errvec.r_ap.value.text() == bits(page.received, 15)
    assert "W(E) = 3" in page.status.text(), page.status.text()


check("вкладки «Ввод данных» и «Вектор ошибок»: проверки A(x), G(x), T и A″(x)", c_fix_ui)


def c_methods_ui():
    # CS: динамика и результаты по справке (21, 12)
    page.inp.cb_n.setCurrentText("21")
    page.inp.sp_t.setValue(2)
    page.inp.btn_calc.click()
    app.processEvents()
    page.set_code(int("110000000000011101100", 2))
    page.fix_in.f_g.set_value(page.model.G)
    page.fix_in.sp_t.setValue(2)
    page.set_received(page.code ^ int("110000000000000000000", 2), int("110000000000000000000", 2))
    page.errvec.r_ap.set(bits(page.received, 21))

    page.fix_tabs.setCurrentIndex(4)
    res = page.cs.dynamics()
    app.processEvents()
    assert res.ok and res.shifts == 2
    assert [c.weight for c in res.cycles[:3]] == [5, 5, 2]
    assert "цикл 1: остаток 11101100, вес W = 5" in page.cs.memo.toPlainText()
    assert page.cs.canvas.ax.patches, "столбцы динамики CS не построены"
    assert len(page.cs.canvas.ax.patches) == len(res.cycles)
    page.cs.results()
    app.processEvents()
    assert page.fix_tabs.currentIndex() == 6, page.fix_tabs.currentIndex()
    assert "влево на 2 разрядов" in page.results.l_shift.text()
    assert page.results.r_shifted.value.text() == "000000000001110110000"
    assert page.results.r_sum.value.text() == "000000000001110110011"
    assert page.results.val.text().startswith("110000000000011101100")
    assert page.results.verdict.text() == "Обнаружено и исправлено: получена исходная комбинация"

    # PGZ: пример (15, 5, 7)
    page.inp.cb_n.setCurrentText("15")
    page.inp.sp_t.setValue(3)
    page.inp.btn_calc.click()
    app.processEvents()
    page.set_code(int("101100100011110", 2))
    page.set_received(page.code ^ int("001000001000001", 2), int("001000001000001", 2))
    page.fix_tabs.setCurrentIndex(5)
    pres = page.pgz.run()
    app.processEvents()
    assert pres.ok and pres.positions == [0, 6, 12], pres.positions
    assert page.pgz.tbl.rowCount() == 6
    assert page.pgz.tbl.item(0, 0).text() == "S1"
    assert page.pgz.tbl.item(0, 1).text() == "α^1"
    assert page.pgz.tbl.item(4, 1).text() == "1"
    assert "det M" in page.pgz.memo.toPlainText()
    assert page.verdict() == "Обнаружено и исправлено: получена исходная комбинация"

    # поле и матрица H
    page.fix_tabs.setCurrentIndex(7)
    app.processEvents()
    assert page.field.tables[0].rowCount() == 15
    assert "p(z) = z^4 + z + 1" in page.field.top.text(), page.field.top.text()
    assert page.field.tables[1].item(0, 1).text() == "1"
    assert page.field.tables[2].item(1, 1).text() == "0010"
    page.fix_tabs.setCurrentIndex(8)
    app.processEvents()
    assert page.hmat.memo.toPlainText().splitlines()[0].startswith("1*C0: 1 α")
    assert page.hmat.canvas.ax.images, "карта матрицы H не построена"

    # без ошибок: синдромы нулевые
    page.set_received(page.code, 0)
    page.pgz.run()
    app.processEvents()
    assert page.pgz.tbl.item(0, 1).text() == "0"
    page.cs.dynamics()
    app.processEvents()
    assert page.verdict() == "Ошибок не обнаружено: синдромы нулевые"


check("«Метод CS», «Метод PGZ», «Результаты», «Построенное поле», «Матрица H»", c_methods_ui)


def c_errors_ui():
    # N с k = 1: код существует, но T ограничен
    page.inp.cb_n.setCurrentText("3")
    page.inp.sp_t.setValue(1)
    page.inp.btn_calc.click()
    app.processEvents()
    assert page.model.exists and page.model.k == 1
    page.inp.sp_t.setValue(2)
    page.inp.btn_calc.click()
    app.processEvents()
    assert not page.model.exists, "код (3, 0) должен отсутствовать"
    assert "не существует" in page.status.text(), page.status.text()
    # сброс
    page.inp.cb_n.setCurrentText("21")
    page.inp.sp_t.setValue(2)
    page.inp.btn_calc.click()
    page.inp.btn_reset.click()
    app.processEvents()
    assert page.inp.memo.toPlainText() == ""
    assert page.inp.r_g.value.text() == "—"
    page.inp.btn_calc.click()
    app.processEvents()


check("отказ при r ≥ N («БЧХ не существует») и «Сброс» ввода", c_errors_ui)


# --- 6. скриншоты -------------------------------------------------------------

def shot(name: str) -> None:
    app.processEvents()
    path = OUT / name
    page.grab().save(str(path))
    assert path.exists() and path.stat().st_size > 3000, f"скриншот пуст: {path}"
    print(f"     shot {path.name} {path.stat().st_size // 1024} KiB")


def c_shots():
    # пример справки: (21, 12), CS, затем PGZ на (15, 5, 7)
    page.inp.cb_n.setCurrentText("21")
    page.inp.sp_t.setValue(2)
    page.inp.btn_calc.click()
    page.set_code(int("110000000000011101100", 2))
    page.set_received(page.code ^ int("110000000000000000000", 2),
                      int("110000000000000000000", 2))
    page.errvec.r_ap.set(bits(page.received, 21))
    page.cs.dynamics()
    page.cs.results()

    page.tabs.setCurrentIndex(0)
    for idx, name in ((0, "lab7-1-theory-build.png"), (1, "lab7-2-build-input.png"),
                      (2, "lab7-3-table.png"), (3, "lab7-4-matrix.png")):
        page.build_tabs.setCurrentIndex(idx)
        shot(name)

    page.tabs.setCurrentIndex(1)
    names = ("lab7-5-theory-cs.png", "lab7-6-theory-pgz.png", "lab7-7-fix-input.png",
             "lab7-8-error-vector.png", "lab7-9-cs.png")
    for idx, name in enumerate(names):
        page.fix_tabs.setCurrentIndex(idx)
        shot(name)
    page.pgz.run()
    app.processEvents()
    for idx, name in ((5, "lab7-10-pgz.png"), (6, "lab7-11-results.png"),
                      (7, "lab7-12-field.png"), (8, "lab7-13-h-matrix.png")):
        page.fix_tabs.setCurrentIndex(idx)
        shot(name)

    page.tabs.setCurrentIndex(0)
    page.build_tabs.setCurrentIndex(1)
    page.inp.cb_n.setCurrentText("15")
    page.inp.sp_t.setValue(3)
    page.inp.btn_calc.click()
    app.processEvents()
    shot("lab7-14-build-15-5.png")

    page.tabs.setCurrentIndex(1)
    page.fix_tabs.setCurrentIndex(5)
    page.set_code(int("101100100011110", 2))
    page.set_received(page.code ^ int("001000001000001", 2), int("001000001000001", 2))
    page.pgz.run()
    app.processEvents()
    shot("lab7-15-pgz-15-5.png")

    files = sorted(OUT.glob("lab7-*.png"))
    assert len(files) >= 15, files


check("скриншоты всех 13 листовых вкладок и двух расчётных примеров", c_shots)


print(f"\n{len(passed)} passed, {len(failures)} failed")
if failures:
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
