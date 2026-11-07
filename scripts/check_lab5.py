from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtWidgets import QApplication

from kuotik_port.labs.lab5 import (
    MAX_N,
    METHODS,
    TITLES,
    Lab5Page,
    apk_decode,
    apk_encode,
    bin_entropy,
    ceil_log2,
    check_block,
    encode,
    encode_all,
    expected_length,
    group_word,
    ipp_decode,
    ipp_encode,
    kds_decode,
    kds_encode,
    kds_series,
    uk_decode,
    uk_encode,
    unit_len,
)

OUT = Path(os.environ.get("SHOTS", "/tmp/shots-lab5"))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication([])
page = Lab5Page()
page.resize(1280, 900)
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
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
        print(f"PASS {name}")


def approx(a: float, b: float, tol: float, what: str = ""):
    assert abs(a - b) <= tol, f"{what}: {a!r} != {b!r} (tol {tol})"


def all_blocks(n: int) -> list[str]:
    return ["".join(bits) for bits in itertools.product("01", repeat=n)]


def c_tabs():
    names = ["Общие сведения", "Ввод данных", "АПК", "КДС", "ИПП", "УК", "Графики"]
    assert page.tabs.count() == 7, page.tabs.count()
    assert [page.tabs.tabText(i) for i in range(7)] == names, \
        [page.tabs.tabText(i) for i in range(7)]
    assert page.apk.vals.keys() == {"N", "K", "L(N)", "P(K)", "Kc"}
    assert page.kds.vals.keys() == {"N", "K", "L(N)", "P(K)", "Kc"}
    assert page.ipp.vals.keys() == {"N", "K", "P(K)", "Kc"}
    assert page.uk.vals.keys() == {"N", "K", "B(K)", "L(N,K)", "P(K)", "Kc"}


check("структура: 7 вкладок Form6 и поля методов", c_tabs)


def c_unit_len():
    assert [unit_len(n) for n in (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 100)] == \
        [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]
    for n in range(1, 200):
        assert unit_len(n) == math.ceil(math.log2(n + 1)), n
        assert unit_len(n) == (n + 1 - 1).bit_length(), n
    assert [unit_len(2 ** m - 1) for m in range(1, 7)] == [1, 2, 3, 4, 5, 6]


check("L = ⌈log₂(n+1)⌉ целочисленно, включая n = 2^m − 1", c_unit_len)


def c_decode_small():
    for n in range(1, 11):
        for block in all_blocks(n):
            for m in METHODS:
                code = encode(m, block)
                assert code.ok, f"{TITLES[m]}: {block} → {code.word} → {code.decoded}"
                assert code.decoded == block, (m, block, code.decoded)


check("декодирование: все блоки n = 1…10 возвращаются всеми методами", c_decode_small)


def c_decode_random():
    rng = np.random.default_rng(11)
    for n in (12, 16, 23, 32, 47, 64):
        for _ in range(12):
            p = float(rng.uniform(0.02, 0.95))
            block = "".join("1" if rng.random() < p else "0" for _ in range(n))
            for m in METHODS:
                code = encode(m, block)
                assert code.ok, f"{TITLES[m]} n={n}: {block} → {code.word} → {code.decoded}"


check("декодирование: случайные блоки до n = 64", c_decode_random)


def c_lengths():
    for n in range(1, 13):
        for block in all_blocks(n):
            k = block.count("1")
            unit = unit_len(n)
            for m in ("apk", "kds", "uk"):
                code = encode(m, block)
                assert code.length == expected_length(m, block), (m, block, code.length)
            assert encode("apk", block).length == unit * max(k, 1), block
            series = kds_series(block)
            assert encode("kds", block).length == unit * len(series), block
            if k == 0:
                assert series == [0], block
            code = encode("uk", block)
            l1, l2 = code.extra["l1"], code.extra["l2"]
            assert code.length == l1 + l2, block
            if k in (0, n) or n == 1:
                assert l2 == 0, (block, l2)
            else:
                assert l2 == ceil_log2(math.comb(n, k)), (block, l2)


check("длина слова = K·L (АПК), t·L (КДС), l₁+l₂ (УК)", c_lengths)


def c_ipp_no_lengthen():
    for n in range(1, 13):
        for block in all_blocks(n):
            code = encode("ipp", block)
            assert code.length <= n, (block, code.word)
            assert code.length == len(code.word), block
    code = encode("ipp", "0" * 30 + "1" + "0" * 30)
    assert code.length < 61, code.word


check("ИПП не удлиняет строку (L ≤ N) и сжимает редкие единицы", c_ipp_no_lengthen)


SRC = "01101101"


def c_reference_example():
    assert len(SRC) == 8 and SRC.count("1") == 5
    assert unit_len(8) == 4
    apk = encode("apk", SRC)
    kds = encode("kds", SRC)
    ipp = encode("ipp", SRC)
    uk = encode("uk", SRC)
    assert apk.word == "00100011010101101000", apk.word
    assert kds.word == "00100001001000010010", kds.word
    assert kds.extra["series"] == [2, 1, 2, 1, 2], kds.extra["series"]
    assert ipp.word == SRC and ipp.length == 8, (ipp.word, ipp.length)
    assert uk.extra["b"] == 32, uk.extra["b"]
    assert (uk.extra["l1"], uk.extra["l2"]) == (4, 6), uk.extra
    assert uk.word == "0101100000", uk.word
    assert (apk.length, kds.length, ipp.length, uk.length) == (20, 20, 8, 10)
    approx(apk.kc, 0.4, 1e-12, "Kсж АПК")
    approx(kds.kc, 0.4, 1e-12, "Kсж КДС")
    approx(ipp.kc, 1.0, 1e-12, "Kсж ИПП")
    approx(uk.kc, 0.8, 1e-12, "Kсж УК")
    for code in (apk, kds, ipp, uk):
        assert code.ok, code.word


check("пример справки n=8: b=32, 4+6=10 бит, Kсж УК=0.8, слова", c_reference_example)


def c_reference_table_52():
    l1 = unit_len(100)
    assert l1 == 7, l1
    for k, l2, total in ((1, 7, 14), (2, 13, 20), (3, 18, 25)):
        block = "1" * k + "0" * (100 - k)
        code = encode("uk", block)
        assert code.extra["l2"] == l2, (k, code.extra)
        assert code.length == total, (k, code.length)
        approx(code.kc, 100 / total, 1e-12, f"Kсж n=100 K={k}")
    approx(100 / 14, 7.14, 0.005, "7.14")
    approx(100 / 20, 5.0, 1e-12, "5.00")
    approx(100 / 25, 4.0, 1e-12, "4.00")


check("табл. 5.2 справки: n=100 → 7+7, 7+13, 7+18 и Kсж 7.14 / 5.00 / 4.00",
      c_reference_table_52)


def c_edge_cases():
    zero, ones = "0" * 8, "1" * 8
    assert encode("apk", zero).word == "0000" and encode("kds", zero).word == "0000"
    assert encode("uk", zero).word == "0000" and encode("ipp", zero).word == "0"
    assert encode("ipp", zero).length == 1
    assert encode("uk", ones).extra["l2"] == 0 and encode("uk", ones).length == 4
    assert encode("apk", ones).length == 32 and encode("kds", ones).length == 32
    for block in ("0", "1", "01", "10", "00", "11"):
        for m in METHODS:
            assert encode(m, block).ok, f"{TITLES[m]} на {block}"
    assert encode("uk", "1").extra["l2"] == 0


check("крайние случаи: K=0, K=N, N=1", c_edge_cases)


def c_bounds():
    for n in range(2, 11):
        for block in all_blocks(n):
            k = block.count("1")
            for m in ("apk", "kds", "uk"):
                code = encode(m, block)
                assert code.kc <= code.class_kc + 1e-9, \
                    f"{TITLES[m]} {block}: Kсж {code.kc} > предела {code.class_kc}"
    for n in (8, 10, 12):
        for k in range(1, n):
            blocks = [b for b in all_blocks(n) if b.count("1") == k]
            avg = sum(encode("ipp", b).kc for b in blocks) / len(blocks)
            bound = n / math.log2(math.comb(n, k))
            assert avg <= bound + 1e-9, (n, k, avg, bound)
    for p in (0.1, 0.25, 0.5, 0.75):
        assert 1 / bin_entropy(p) >= 1.0 - 1e-12, p


check("Kсж не выше предела класса n/log₂C; ИПП — в среднем", c_bounds)


def c_entropy_monotone():
    left = [bin_entropy(p) for p in (0.01, 0.1, 0.25, 0.5)]
    right = [bin_entropy(p) for p in (0.5, 0.75, 0.9)]
    assert all(a < b - 1e-12 for a, b in zip(left, left[1:])), left
    assert all(a > b + 1e-12 for a, b in zip(right, right[1:])), right
    approx(bin_entropy(0.5), 1.0, 1e-12, "H(0.5)")
    approx(bin_entropy(0.25), bin_entropy(0.75), 1e-12, "симметрия H(p) = H(1−p)")
    assert bin_entropy(0.0) == 0.0 and bin_entropy(1.0) == 0.0


check("H(p): симметрия и максимум при p = 0.5", c_entropy_monotone)


def positions_of_blocks(n: int, k: int) -> list[str]:
    out = []
    for ones in itertools.combinations(range(n), k):
        row = ["0"] * n
        for i in ones:
            row[i] = "1"
        out.append("".join(row))
    return out


def c_sensitivity():
    blocks = positions_of_blocks(10, 1)
    lens = {m: {encode(m, b).length for b in blocks} for m in METHODS}
    assert lens["apk"] == {4}, lens["apk"]
    assert lens["uk"] == {8}, lens["uk"]
    assert lens["kds"] == {4, 8}, lens["kds"]
    assert len(lens["ipp"]) > 1, lens["ipp"]
    assert lens["ipp"] == {3, 4, 5, 6}, lens["ipp"]
    tail = [b for b in blocks if b.endswith("0")]
    assert all(encode("kds", b).length == 8 for b in tail), "нет хвостовой серии"
    assert encode("kds", "0000000001").length == 4

    blocks8 = positions_of_blocks(8, 4)
    lens8 = {m: {encode(m, b).length for b in blocks8} for m in METHODS}
    assert lens8["apk"] == {16} and lens8["uk"] == {11}, lens8
    assert lens8["kds"] == {16, 20}, lens8["kds"]
    assert encode("kds", "11110000").length == 20
    assert encode("kds", "00001111").length == 16


check("чувствительность к расположению: АПК/УК постоянны, КДС и ИПП — нет", c_sensitivity)


def c_page_numbers():
    page.set_block(SRC)
    res = page.results
    assert set(res) == set(METHODS), res
    assert all(c.ok for c in res.values())
    assert page.apk.vals["L(N)"].text() == "4"
    assert page.apk.vals["Kc"].text() == "0.400"
    assert page.kds.vals["N"].text() == "8" and page.kds.vals["K"].text() == "5"
    assert page.ipp.vals["P(K)"].text() == "0.6250"
    assert page.uk.vals["B(K)"].text() == "32"
    assert page.uk.vals["L(N,K)"].text() == "10"
    memo = page.uk.memo.toPlainText()
    assert "0101 100000" in memo, memo
    assert "Декодирование   : 01101101 — совпало" in memo, memo
    assert "не совпало" not in page.apk.memo.toPlainText().replace("Декодирование", "")
    text = page.status.text()
    assert "Kсж" in text and "совпало" in text and "n = 8" in text, text
    assert page.lbl_info.text().startswith("N = 8, K = 5"), page.lbl_info.text()
    for panel in page.panels.values():
        assert "совпало" in panel.verdict.text(), panel.verdict.text()


check("страница: параметры N, K, L, P(K), Kc, B(K) и статус", c_page_numbers)


def c_table():
    page.set_block(SRC)
    assert page.tbl_cmp.rowCount() == 4, page.tbl_cmp.rowCount()
    names = [page.tbl_cmp.item(r, 0).text() for r in range(4)]
    assert names == ["АПК", "КДС", "ИПП", "УК"], names
    rows = {page.tbl_cmp.item(r, 0).text(): [page.tbl_cmp.item(r, c).text() for c in range(7)]
            for r in range(4)}
    assert rows["УК"][1] == "10" and rows["УК"][2] == "0.800", rows["УК"]
    assert rows["АПК"][1] == "20" and rows["КДС"][1] == "20", rows
    assert rows["ИПП"][1] == "8", rows["ИПП"]
    assert rows["УК"][5] == "1.048", rows["УК"][5]
    assert rows["УК"][6] == "1.378", rows["УК"][6]


check("таблица сравнения: L, Kсж, R, η, 1/H(p), n/log₂C", c_table)


def c_validation():
    page.set_block("")
    assert page.results == {}, page.results
    assert page.status.text() == "Не выбран исследуемый блок!", page.status.text()
    assert page.tbl_cmp.rowCount() == 0
    assert page.apk.memo.toPlainText() == ""

    page.edit.setText("01x1")
    page.encode_all()
    assert "Допустимы только символы 0 и 1" in page.status.text(), page.status.text()

    assert page.edit.maxLength() == MAX_N, page.edit.maxLength()
    assert "Неверная длина блока" in check_block("1" * (MAX_N + 1))
    assert check_block("1" * MAX_N) == ""
    page.edit.setText("1" * (MAX_N + 1))
    assert len(page.block()) == MAX_N, len(page.block())
    page.edit.clear()
    page.edit.insert("0101")
    assert page.block() == "0101", page.block()
    page.set_block(SRC)


check("валидация: пустой блок, недопустимые символы, длина > 64", c_validation)


def c_buttons():
    page.reset_all()
    page.rb1.click()
    app.processEvents()
    assert page.block() == "01101101", page.block()
    page.rb2.click()
    app.processEvents()
    assert page.block() == "10101010", page.block()
    assert page.results["ipp"].length == 8
    page.set_block("001100")
    assert page.blocks[2] == "001100", page.blocks
    page.btn_reset_block.click()
    app.processEvents()
    assert page.block() == ""
    assert page.status.text() == "Блок сброшен", page.status.text()
    assert page.apk.memo.toPlainText() == ""
    page.btn_reset_all.click()
    app.processEvents()
    assert page.status.text() == "Накопленные данные уничтожены !", page.status.text()
    assert page.blocks == {1: "01101101", 2: "10101010"}, page.blocks


check("кнопки «Блок №1/№2», «Сброс блока», «Сброс всей информации»", c_buttons)


def c_graphs():
    page.set_block(SRC)
    page.n1.setValue(8)
    page.n2.setValue(16)
    page.k_sens.setValue(1)
    app.processEvents()
    cv = page.cv_curves
    assert "Семейства" in cv.ax.get_title(), cv.ax.get_title()
    labeled = [ln for ln in cv.ax.lines if not ln.get_label().startswith("_")]
    assert len(labeled) == 2 * len(METHODS) + 1, len(labeled)
    labels = [t.get_text() for t in cv.ax.get_legend().get_texts()]
    assert any("граница 1/H" in t for t in labels), labels
    assert all(any(TITLES[m] in t for t in labels) for m in METHODS), labels
    apk_line = [ln for ln in labeled if ln.get_label().startswith("АПК, n=8")][0]
    y = dict(zip(apk_line.get_xdata(), apk_line.get_ydata()))
    approx(float(y[0.5]), 0.5, 1e-12, "Kсж АПК при p = 0.5, n = 8")

    cv2 = page.cv_sens
    assert "Чувствительность" in cv2.ax.get_title(), cv2.ax.get_title()
    labeled2 = [ln for ln in cv2.ax.lines if not ln.get_label().startswith("_")]
    assert len(labeled2) == len(METHODS), len(labeled2)
    flat = [ln for ln in labeled2 if ln.get_label() == "АПК"][0]
    assert len(set(np.round(flat.get_ydata(), 9))) == 1, "АПК не постоянен по расположениям"


check("графики: семейства Kсж(p), граница 1/H и разброс по расположениям", c_graphs)


def c_group_word():
    page.set_block(SRC)
    assert group_word(page.results["uk"]) == "0101 100000"
    assert group_word(page.results["apk"]) == "0010 0011 0101 0110 1000"
    assert group_word(page.results["kds"]) == "0010 0001 0010 0001 0010"
    assert group_word(page.results["ipp"]) == "01101101"


check("разбивка слова по частям метода", c_group_word)


def c_helpers():
    word, unit, pos = apk_encode(SRC)
    assert (word, unit, pos) == ("00100011010101101000", 4, [2, 3, 5, 6, 8])
    assert apk_decode(word, unit, 8, 5) == SRC
    assert apk_decode("0000", 4, 8, 0) == "0" * 8
    assert apk_decode("1111", 4, 8, 0) == ""
    word, unit, series = kds_encode(SRC)
    assert (word, unit, series) == ("00100001001000010010", 4, [2, 1, 2, 1, 2])
    assert kds_decode(word, unit, 8, 5) == SRC
    assert kds_decode("111", 4, 8, 5) == ""
    word, l1, l2, b = uk_encode(SRC)
    assert (word, l1, l2, b) == ("0101100000", 4, 6, 32)
    assert uk_decode(word, 8, l1, l2) == SRC
    assert uk_decode("1111111111", 8, 4, 6) == ""
    word, trace = ipp_encode("001100")
    assert word == "11110" and len(trace) == 3, (word, trace)
    assert ipp_decode(word, trace) == "001100"
    assert encode_all("001100")["kds"].length == 9


check("функции методов: слова, отказы декодера, трасса ИПП", c_helpers)


def shot(name: str) -> None:
    app.processEvents()
    path = OUT / name
    page.grab().save(str(path))
    assert path.exists() and path.stat().st_size > 2000, f"скриншот пуст: {path}"
    print(f"     shot {path.name} {path.stat().st_size // 1024} KiB")


def c_shots():
    page.set_block(SRC)
    for idx, name in enumerate(("lab5-1-theory.png", "lab5-2-input.png", "lab5-3-apk.png",
                                "lab5-4-kds.png", "lab5-5-ipp.png", "lab5-6-uk.png",
                                "lab5-7-graphs.png")):
        page.tabs.setCurrentIndex(idx)
        shot(name)
    page.n1.setValue(10)
    page.k_sens.setValue(1)
    page.tabs.setCurrentIndex(6)
    shot("lab5-8-graphs-sensitivity.png")
    page.tabs.setCurrentIndex(3)
    page.kds.btn_encode.click()
    shot("lab5-9-kds-encode.png")


check("скриншоты всех вкладок", c_shots)


print(f"\n{len(passed)} passed, {len(failures)} failed")
if failures:
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("OK; скриншоты:", OUT)
