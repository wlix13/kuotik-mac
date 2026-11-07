"""Headless-проверка ЛР6 (Form7): C ДК/ДСК/ДСКС, максимум по входу, границы, числа справки.

Запуск: cd /home/wlix13/work/kuotik-mac && uv run python scripts/check_lab6.py
Скриншоты: /tmp/shots-lab6 (переопределяется SHOTS).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kuotik_port.labs.lab6 import (  # noqa: E402
    CODES,
    FIELD_MASK,
    MSG_BAD,
    MSG_P_GT1,
    MSG_PQ_GT1,
    MSG_UNDEF,
    TABS,
    DKResult,
    DSKSResult,
    Lab6Page,
    best_z,
    bin_entropy,
    code_probs,
    dk_capacity,
    dk_capacity_legacy,
    dsks_capacity,
    fmt,
    mutual_info,
    nat_entropy,
    parse_field,
    surface,
    surface_rows,
)

OUT = Path(os.environ.get("SHOTS", "/tmp/shots-lab6"))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication([])
page = Lab6Page()
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
    assert page.tabs.count() == 6, page.tabs.count()
    names = [page.tabs.tabText(i) for i in range(6)]
    assert tuple(names) == TABS, names
    assert "циклический код" in names[3], names[3]
    for edit in (page.dk.f_p1, page.dk.f_p2, page.dsks.f_p, page.dsks.f_q):
        assert edit.inputMask() == FIELD_MASK, edit.inputMask()
        assert len(edit.text()) == len(FIELD_MASK), edit.text()
    assert page.dk.btn_calc.text() == "Вычислить"
    assert page.dsks.btn_calc.text() == "Вычислить"
    assert page.dk.btn_graph.text() == "Грaфика"           # написание формы
    assert page.dk.btn_surface.text() == "С = F [p1,p2]"
    assert page.dsks.btn_surface.text() == "C = F [p,q]"
    assert page.graph_panel.btn_reset.text() == "Сброс"
    assert page.graph_panel.btn_exit.text() == "Выxод"
    assert page.code.tasks[1].btn_calc.text() == "Вычислить"
    assert page.code.btn_next.text() == "Продолжить"
    assert page.code.lbl_solved.text() == "Задача решена верно"


check("структура: 6 вкладок Form7, маски полей и кнопки", c_tabs)


def c_labels_dk():
    assert page.dk.r_s.caption.text().startswith("s = 1 − p1 − p2"), page.dk.r_s.caption.text()
    assert page.dk.r_b.caption.text().startswith("B = p1·ln p1"), page.dk.r_b.caption.text()
    assert page.dk.r_d.caption.text().startswith("D = 1 + exp(−B/s)"), page.dk.r_d.caption.text()
    assert page.dk.r_zo.caption.text().startswith("Zo = (1 − p2·D)"), page.dk.r_zo.caption.text()
    assert page.dk.r_c.caption.text().startswith("C = H(X;Y)"), page.dk.r_c.caption.text()
    assert page.dk.r_z.caption.text().startswith("Z = P(x1) = Zo"), page.dk.r_z.caption.text()
    assert page.dsks.r_s.caption.text().startswith("s = 1 − p − q"), page.dsks.r_s.caption.text()
    assert page.dsks.r_zo.caption.text().startswith("Zo = 0.5"), page.dsks.r_zo.caption.text()
    # подписи поверхностей дословно из DFM (в форме перепутаны местами)
    assert page.tabs.tabText(4) == "С = F [p1,p2]"
    titles = [w.text() for w in page.surface_dk.findChildren(type(page.dk.r_s.caption))]
    assert any("ДСКС" in t for t in titles), titles
    assert page.dk.lbl_check.text() == "Проконтролировать поверхность:"
    assert page.dsks.lbl_check.text().startswith("Проконтролировать поверхность")
    assert [c.text() for c in page.graph_panel.captions] == ["Ось 1", "Ось 2", "C"]
    assert page.graph_panel.lbl_zero.text() == "0" and page.graph_panel.lbl_one.text() == "1"


check("подписи полей 1:1 с DFM (s, B, D, Zo, C, Z = P(x1) = Zo)", c_labels_dk)


# --- 2. ДСК: C = 1 − H(p) -----------------------------------------------------

def c_bsc():
    for p in (0.0, 0.01, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 0.9, 0.99):
        approx(dk_capacity(p, p), 1.0 - bin_entropy(p), 1e-12, f"C ДСК p={p}")
    approx(dk_capacity(0.5, 0.5), 0.0, 1e-12, "C при p=0.5")
    approx(dk_capacity(0.0, 0.0), 1.0, 1e-12, "C при p1=p2=0")
    # ДСКС при q = 0 — тот же двоичный симметричный канал
    for p in (0.01, 0.05, 0.2, 0.4):
        approx(dsks_capacity(p, 0.0), 1.0 - bin_entropy(p), 1e-12, f"ДСКС(p={p}, q=0)")
    # ДСКС при p = 0 — канал со стиранием, C = 1 − q
    for q in (0.0, 0.125, 0.5, 0.9):
        approx(dsks_capacity(0.0, q), 1.0 - q, 1e-12, f"ДСКС(p=0, q={q})")
    # симметрия ДК по перестановке p1 и p2
    for p1, p2 in ((0.05, 0.2), (0.01, 0.4), (0.3, 0.6)):
        approx(dk_capacity(p1, p2), dk_capacity(p2, p1), 1e-12, f"p1<->p2 {p1}/{p2}")


check("C = 1 − H(p) для ДСК; ДСКС(p,0) и ДСКС(0,q); симметрия по p1/p2", c_bsc)


# --- 3. I(X;Y) = H(Y) − H(Y|X) ------------------------------------------------

def c_mutual_info():
    rng = np.random.default_rng(7)
    for _ in range(200):
        p1 = float(rng.uniform(0.0, 0.45))
        p2 = float(rng.uniform(0.0, 0.45))
        z = float(rng.uniform(0.0, 1.0))
        py1 = z * (1.0 - p1) + (1.0 - z) * p2
        py2 = 1.0 - py1
        hy = -(py1 * math.log2(py1) if py1 > 0 else 0.0) - \
             (py2 * math.log2(py2) if py2 > 0 else 0.0)
        hyx = z * bin_entropy(p1) + (1.0 - z) * bin_entropy(p2)
        approx(mutual_info(p1, p2, z), hy - hyx, 1e-12, "H(Y) − H(Y|X)")
    # ручной счёт одного узла: p1 = 0.1, p2 = 0.2, z = 0.3
    p1, p2, z = 0.1, 0.2, 0.3
    py1 = z * (1 - p1) + (1 - z) * p2
    assert abs(py1 - 0.41) < 1e-12, py1
    expected = bin_entropy(0.41) - (0.3 * bin_entropy(0.1) + 0.7 * bin_entropy(0.2))
    approx(mutual_info(p1, p2, z), expected, 1e-12, "узел 0.1/0.2/z=0.3")


check("I(X;Y) = H(Y) − H(Y|X) по определению, 200 случайных каналов", c_mutual_info)


# --- 4. C — максимум по входному распределению --------------------------------

def c_max_over_input():
    pairs = [(0.0, 0.0), (0.05, 0.05), (0.05, 0.125), (0.1, 0.2), (0.2, 0.3),
             (0.0, 0.3), (0.3, 0.05), (0.45, 0.45), (0.01, 0.9), (0.4, 0.5)]
    for p1, p2 in pairs:
        res = DKResult(p1, p2)
        best_value, best_point = best_z(p1, p2, steps=40001)
        approx(res.c, best_value, 2e-4, f"C != max I при p1={p1}, p2={p2}")
        approx(res.zo, best_point, 2e-3, f"Zo != argmax при p1={p1}, p2={p2}")
        # ни одна точка сетки не даёт больше C
        for z in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
            assert mutual_info(p1, p2, z) <= res.c + 1e-9, (p1, p2, z)
        # и C достигается в Zo
        approx(mutual_info(p1, p2, res.zo), res.c, 1e-9, f"I(Zo) != C при {p1}/{p2}")
    assert DKResult(0.0, 0.0).c == 1.0
    assert DKResult(0.3, 0.7).c == 0.0            # s = 0 — канал не пропускает


check("C — максимум H(X;Y) по z, достигается в Zo = (1−p2·D)/(s·D)", c_max_over_input)


def c_dsks_max():
    for p, q in ((0.0, 0.0), (0.05, 0.125), (0.01, 0.5), (0.2, 0.3), (0.0, 0.9)):
        s = 1.0 - p - q
        # H(Y) при равномерном входе: P(0) = P(1) = (1−q)/2, P(c) = q
        p0 = (1.0 - q) / 2.0
        hy = -2 * p0 * math.log2(p0) - (q * math.log2(q) if q > 0 else 0.0)
        hrow = -(s * math.log2(s) if s > 0 else 0.0) - \
               (p * math.log2(p) if p > 0 else 0.0) - \
               (q * math.log2(q) if q > 0 else 0.0)
        approx(dsks_capacity(p, q), hy - hrow, 1e-12, f"C ДСКС {p}/{q}")
    # Zo = 0.5 — оптимум: сдвиг распределения не увеличивает H(Y)
    p, q = 0.05, 0.125
    base = dsks_capacity(p, q)
    hrow = -(1 - p - q) * math.log2(1 - p - q) - p * math.log2(p) - q * math.log2(q)
    for z in (0.4, 0.45, 0.49, 0.51, 0.55, 0.6):
        p0 = z * (1 - p - q) + (1 - z) * p
        p1_ = z * p + (1 - z) * (1 - p - q)
        est = -p0 * math.log2(p0) - p1_ * math.log2(p1_) - q * math.log2(q) - hrow
        assert est <= base + 1e-9, (z, est, base)


check("C ДСКС = H(Y)max − H(p,q,1−p−q), Zo = 0.5 — оптимум", c_dsks_max)


# --- 5. границы I ≥ 0, I ≤ min(H(X), H(Y)) ------------------------------------

def c_bounds():
    rng = np.random.default_rng(19)
    for _ in range(300):
        p1 = float(rng.uniform(0.0, 0.5))
        p2 = float(rng.uniform(0.0, 0.5))
        z = float(rng.uniform(0.0, 1.0))
        p1, p2 = min(p1, 0.5 - p2), p2
        i = mutual_info(p1, p2, z)
        hx = bin_entropy(z)
        hy = bin_entropy(z * (1 - p1) + (1 - z) * p2)
        assert i >= -1e-12, (p1, p2, z, i)
        assert i <= min(hx, hy) + 1e-12, (p1, p2, z, i, hx, hy)
    for p1, p2 in ((0.05, 0.125), (0.1, 0.2), (0.3, 0.05)):
        c = dk_capacity(p1, p2)
        assert 0.0 <= c <= 1.0, c
        assert c <= min(bin_entropy(0.5), bin_entropy(0.5)) + 1e-12, c
    for p, q in ((0.05, 0.125), (0.2, 0.3), (0.01, 0.5)):
        c = dsks_capacity(p, q)
        assert 0.0 <= c <= 1.0 - q + 1e-12, c
    # избыточность 1 − C — доля недоиспользованной пропускной способности
    for p, q in ((0.05, 0.125), (0.1, 0.2)):
        assert 0.0 <= 1.0 - dsks_capacity(p, q) <= 1.0


check("0 ≤ I ≤ min(H(X), H(Y)); C ∈ [0, 1]; C ≤ 1 − q для ДСКС", c_bounds)


def c_nat_entropy():
    approx(nat_entropy(0.5), math.log(2), 1e-12, "h(0.5)")
    approx(nat_entropy(0.1), -0.1 * math.log(0.1) - 0.9 * math.log(0.9), 1e-12, "h(0.1)")
    assert nat_entropy(0.0) == 0.0 and nat_entropy(1.0) == 0.0
    # B и D из формы при p1 = 0.05, p2 = 0.1
    res = DKResult(0.05, 0.1)
    expected_b = nat_entropy(0.1) - nat_entropy(0.05)
    approx(res.b, expected_b, 1e-12, "B")
    approx(res.d, 1.0 + math.exp(-expected_b / res.s), 1e-12, "D")
    approx(res.zo, (1 - 0.1 * res.d) / (res.s * res.d), 1e-12, "Zo")
    approx(res.s, 0.85, 1e-12, "s")


check("B = h(p2) − h(p1) в натах, D = 1 + exp(−B/s), Zo по формуле формы", c_nat_entropy)


# --- 6. числа справки ---------------------------------------------------------

REF = {  # (N, k): p_ош, q = 2^−(N−k) в справке, 1 − q в справке
    (7, 4): (0.05, 0.125, 0.875),
    (15, 11): (0.05, 0.063, 0.937),
    (31, 26): (0.05, 0.031, 0.969),
}
REF_C = {(7, 4): 0.94532, (15, 11): 0.99954, (31, 26): 0.99999}


def c_reference_table():
    for (n, k), (p_osh, q_ref, one_q_ref) in REF.items():
        share = 2.0 ** (-(n - k))
        approx(share, q_ref, 1e-3, f"2^−({n}−{k}) для q справки")
        approx(1.0 - share, one_q_ref, 1e-3, f"1 − 2^−({n}−{k})")
        p_no, p_oo, p_pp = code_probs(n, k, p_osh)
        approx(p_no, share * p_osh, 1e-12, "p_но")
        approx(p_oo, (1 - share) * p_osh, 1e-12, "p_оо")
        approx(p_no + p_oo + p_pp, 1.0, 1e-12, "сумма вероятностей кода")
    # вариант (7,4): p_но = 0.00625, p_оо = 0.04375 при p_ош = 0.05
    p_no, p_oo, p_pp = code_probs(7, 4, 0.05)
    approx(p_no, 0.00625, 1e-12, "p_но (7,4)")
    approx(p_oo, 0.04375, 1e-12, "p_оо (7,4)")
    approx(p_pp, 0.95, 1e-12, "p_пп (7,4)")
    assert set(CODES) == {"КОД (7,4)", "КОД (15,11)", "КОД (31,26)"}, CODES


check("табл. 6.1: q = 2^−(N−k), 1 − q, p_но = q·p_ош, p_оо = (1−q)·p_ош", c_reference_table)


def c_reference_c():
    # C по формулам 6.1–6.3 для p_ош = 0.05 и q = 2^−(N−k)
    values = {}
    for (n, k), (p_osh, q_ref, _) in REF.items():
        share = 2.0 ** (-(n - k))
        values[(n, k)] = dsks_capacity(p_osh, share)
    approx(values[(7, 4)], 0.59850, 5e-5, "C (7,4) ДСКС(p,q)")
    approx(values[(15, 11)], 0.65588, 5e-5, "C (15,11)")
    approx(values[(31, 26)], 0.68470, 5e-5, "C (31,26)")
    assert values[(7, 4)] < values[(15, 11)] < values[(31, 26)], values
    # ДСКС при q = 0.125 не может дать больше 1 − q = 0.875
    assert values[(7, 4)] <= 1.0 - 0.125 + 1e-12
    # столбец C справки недостижим ни одной моделью канала — фиксируем расхождение
    for (n, k), manual in REF_C.items():
        share = 2.0 ** (-(n - k))
        p_no, p_oo, _ = code_probs(n, k, 0.05)
        variants = (dsks_capacity(0.05, share), dsks_capacity(p_no, p_oo),
                    dk_capacity(p_no, p_oo), dk_capacity(0.05, share))
        gaps = [abs(v - manual) for v in variants]
        assert min(gaps) > 0.02, f"{n},{k}: справка {manual} воспроизведена моделью"
        assert manual > 1.0 - share, f"{n},{k}: C справки выше 1 − q"
    # а с p_но, p_оо расчёт даёт заметно больше, чем с (p_ош, q)
    p_no, p_oo, _ = code_probs(7, 4, 0.05)
    approx(dsks_capacity(p_no, p_oo), 0.90190, 5e-5, "C (7,4) по p_но, p_оо")
    assert dsks_capacity(p_no, p_oo) > dsks_capacity(0.05, 0.125)


check("столбец C табл. 6.1 формулами не воспроизводится (расхождение зафиксировано)",
      c_reference_c)


def c_legacy():
    # оригинал (0x4fc898) путает веса в H(Y/X): при p1 ≠ p2 занижает C
    approx(dk_capacity_legacy(0.05, 0.125), 0.57157, 5e-5, "C оригинала 0.57157")
    approx(dk_capacity(0.05, 0.125), 0.58173, 5e-5, "истинный максимум 0.58173")
    assert dk_capacity_legacy(0.05, 0.125) < dk_capacity(0.05, 0.125)
    for p in (0.0, 0.1, 0.25, 0.4):
        approx(dk_capacity_legacy(p, p), dk_capacity(p, p), 1e-12, f"p1=p2={p}")


check("сверка с оригиналом: 0.57157 (как в форме) против 0.58173 (максимум)", c_legacy)


# --- 7. ввод и валидация ------------------------------------------------------

def c_parse():
    assert parse_field("0.05000") == 0.05
    assert parse_field(" 1.00000 ") == 1.0
    assert parse_field("0,12500") == 0.125
    assert parse_field("") is None
    assert parse_field("   ") is None
    assert parse_field("-0.10000") is None
    assert fmt(0.5817325) == "0.58173"
    assert fmt(1.0) == "1.00000"
    assert fmt(0.0) == "0.00000"


check("разбор полей маски 0.00000 и формат Str(x:6:5)", c_parse)


def c_validation():
    page.dk.f_p1.set_value(1.5)
    page.dk.f_p2.set_value(0.1)
    page.dk.btn_calc.click()
    app.processEvents()
    assert page.status.text() == MSG_P_GT1, page.status.text()
    page.dk.f_p1.set_value(0.6)
    page.dk.f_p2.set_value(0.5)
    page.dk.btn_calc.click()
    app.processEvents()
    assert page.status.text() == MSG_PQ_GT1, page.status.text()
    assert page.dk.r_c.value.text() == MSG_UNDEF, page.dk.r_c.value.text()

    page.dsks.f_p.set_value(0.7)
    page.dsks.f_q.set_value(0.4)
    page.dsks.btn_calc.click()
    app.processEvents()
    assert page.status.text() == MSG_PQ_GT1, page.status.text()
    assert page.dsks.r_c.value.text() == MSG_UNDEF, page.dsks.r_c.value.text()

    page.dk.f_p1.setText("")
    page.dk.btn_calc.click()
    app.processEvents()
    assert page.status.text() == MSG_BAD, page.status.text()

    page.dk.f_p1.set_value(0.05)
    page.dk.f_p2.set_value(0.1)
    page.dk.btn_calc.click()
    app.processEvents()
    assert page.dk.r_c.value.text() == "0.62093", page.dk.r_c.value.text()


check("валидация: p > 1, p1 + p2 ≥ 1, пустое поле; возврат к 0.05/0.10", c_validation)


# --- 8. страница: числа в подписях --------------------------------------------

def c_page_dk():
    page.dk.f_p1.set_value(0.05)
    page.dk.f_p2.set_value(0.10)
    page.dk.btn_calc.click()
    app.processEvents()
    res = DKResult(0.05, 0.10)
    assert page.dk.r_s.value.text() == "0.85000", page.dk.r_s.value.text()
    assert page.dk.r_b.value.text() == fmt(res.b), page.dk.r_b.value.text()
    assert page.dk.r_d.value.text() == fmt(res.d)
    assert page.dk.r_zo.value.text() == fmt(res.zo)
    assert page.dk.r_c.value.text() == "0.62093", page.dk.r_c.value.text()
    assert page.dk.r_z.value.text() == page.dk.r_zo.value.text()
    assert "C = 0.62093" in page.status.text(), page.status.text()
    approx(float(page.dk.r_zo.value.text()), res.zo, 5e-6)

    page.dsks.f_p.set_value(0.05)
    page.dsks.f_q.set_value(0.125)
    page.dsks.btn_calc.click()
    app.processEvents()
    assert page.dsks.r_s.value.text() == "0.82500", page.dsks.r_s.value.text()
    assert page.dsks.r_zo.value.text() == "0.50000"
    assert page.dsks.r_c.value.text() == "0.59850", page.dsks.r_c.value.text()
    assert page.dsks.r_z.value.text() == "0.50000"
    assert "0.59850" in page.status.text(), page.status.text()


check("страница ДК/ДСКС: s, B, D, Zo, C и статус с пятью знаками", c_page_dk)


def c_page_info():
    """H(X), H(Y), H(Y|X), I(X;Y) и избыточность на вкладках ДК и ДСКС."""
    page.dk.f_p1.set_value(0.05)
    page.dk.f_p2.set_value(0.10)
    page.dk.compute()
    app.processEvents()
    res = DKResult(0.05, 0.10)
    hx, hy, hyx = res.hx(), res.hy(), res.hyx()
    approx(float(page.dk.r_hx.value.text()), hx, 5e-6, "H(X) на вкладке")
    approx(float(page.dk.r_hy.value.text()), hy, 5e-6, "H(Y)")
    approx(float(page.dk.r_hyx.value.text()), hyx, 5e-6, "H(Y|X)")
    approx(float(page.dk.r_i.value.text()), hy - hyx, 5e-6, "I(X;Y)")
    approx(float(page.dk.r_r.value.text()), 1.0 - res.c, 5e-6, "избыточность")
    approx(float(page.dk.r_i.value.text()), float(page.dk.r_c.value.text()), 5e-6,
           "I(X;Y) = C")
    assert 0.0 <= hx <= 1.0 and abs(hx - bin_entropy(res.zo)) < 1e-12
    assert hy >= hyx - 1e-12

    page.dsks.f_p.set_value(0.05)
    page.dsks.f_q.set_value(0.125)
    page.dsks.compute()
    app.processEvents()
    d = DSKSResult(0.05, 0.125)
    assert page.dsks.r_hx.value.text() == "1.00000", page.dsks.r_hx.value.text()
    approx(float(page.dsks.r_hy.value.text()), d.hy(), 5e-6, "H(Y) ДСКС")
    approx(float(page.dsks.r_hyx.value.text()), d.hyx(), 5e-6, "H(Y|X) ДСКС")
    approx(float(page.dsks.r_hy.value.text()) - float(page.dsks.r_hyx.value.text()),
           float(page.dsks.r_c.value.text()), 1e-5, "H(Y) − H(Y|X) = C (ДСКС)")
    approx(float(page.dsks.r_r.value.text()), 1.0 - d.c, 5e-6, "избыточность ДСКС")
    # H(Y) = H(Y|X) + C по определению взаимной энтропии
    approx(d.hy() - d.hyx(), d.c, 1e-12, "ДСКС: H(Y) − H(Y|X) = C")


check("H(X), H(Y), H(Y|X), I(X;Y) = C и избыточность 1 − C на вкладках", c_page_info)


def c_tasks():
    code = page.code
    code.cb_code.setCurrentText("КОД (7,4)")
    app.processEvents()
    assert code.lbl_code.text() == "КОД (7,4)", code.lbl_code.text()
    code.btn_fill.click()
    app.processEvents()
    t1, t2, t3 = code.tasks[1], code.tasks[2], code.tasks[3]
    assert t1.f_a.value() == 0.00625 and t1.f_b.value() == 0.04375, \
        (t1.f_a.value(), t1.f_b.value())

    t1.btn_calc.click()
    app.processEvents()
    expected = dsks_capacity(0.00625, 0.04375)
    assert t1.l_c.text() == fmt(expected), (t1.l_c.text(), fmt(expected))
    assert not code.result_panel.isHidden(), "ResultPanel не показан"
    assert "Задача 1 решена верно" in code.lbl_solved.text(), code.lbl_solved.text()
    assert "1−log₂(1−q)" in code.lbl_formula.text(), code.lbl_formula.text()

    code.btn_next.click()
    app.processEvents()
    assert code.stack.currentIndex() == 1, code.stack.currentIndex()
    assert code.result_panel.isHidden()
    t2.btn_calc.click()
    app.processEvents()
    assert t2.l_c.text() == fmt(expected)
    code.btn_next.click()
    app.processEvents()
    assert code.stack.currentIndex() == 2, code.stack.currentIndex()
    assert page.status.text().startswith("Задача 3"), page.status.text()

    # задача 3 — ДК без контроля при тех же данных
    t3.f_a.set_value(0.00625)
    t3.f_b.set_value(0.04375)
    t3.btn_calc.click()
    app.processEvents()
    c_dk = dk_capacity(0.00625, 0.04375)
    assert t3.l_c.text() == fmt(c_dk), (t3.l_c.text(), fmt(c_dk))
    assert c_dk < expected, (c_dk, expected)      # контроль повышает C
    code.btn_next.click()
    app.processEvents()
    assert code.stack.currentIndex() == 0, code.stack.currentIndex()

    # сброс ввода задачи: p + q > 1 отклоняется
    t1.f_a.set_value(0.8)
    t1.f_b.set_value(0.4)
    t1.btn_calc.click()
    app.processEvents()
    assert page.status.text() == MSG_PQ_GT1, page.status.text()


check("задачи кода: p_но/p_оо, C, «Задача решена верно», «Продолжить», валидация",
      c_tasks)


# --- 9. поверхности и панель графики ------------------------------------------

def c_surface_grid():
    grid_x, grid_y, grid_z = surface("dk")
    assert grid_x.shape == grid_z.shape == (15, 15), grid_x.shape
    mask = grid_x + grid_y < 1.0 - 1e-9
    assert np.isfinite(grid_z[mask]).all()
    assert np.isnan(grid_z[~mask]).all()
    assert grid_z[mask].min() >= 0.0 and grid_z[mask].max() <= 1.0 + 1e-12
    i0, j0 = 0, 0
    approx(float(grid_z[i0, j0]), 1.0, 1e-12, "C(0,0) на поверхности")
    gx2, gy2, gz2 = surface("dsks")
    approx(float(gz2[0, 0]), 1.0, 1e-12, "C ДСКС(0,0)")
    approx(float(gz2[3, 0]), dsks_capacity(float(gx2[3, 0]), float(gy2[3, 0])), 1e-12, "край q")
    rows = surface_rows("dk")
    assert len(rows) == 255, len(rows)
    assert all(0.0 <= c <= 1.0 for _, _, c in rows), "C вне [0,1] в таблице"


check("поверхности: NaN вне s > 0, C ∈ [0, 1], таблица 255 строк", c_surface_grid)


def c_surfaces_ui():
    for idx, tab in ((4, page.surface_dk), (5, page.surface_dsks)):
        page.tabs.setCurrentIndex(idx)
        app.processEvents()
        ax = tab.canvas.ax
        assert ax.name == "3d", ax.name
        assert ax.get_zlabel().startswith("C"), ax.get_zlabel()
        assert ax.collections, "поверхность не построена"
    assert page.tabs.tabText(5) == "C = F [p,q]"


check("вкладки 5 и 6: 3D-поверхности C = F[p1,p2] и C = F[p,q]", c_surfaces_ui)


def c_graph_panel():
    page.dk.btn_graph.click()
    app.processEvents()
    assert page.stack.currentWidget() is page.graph_panel, "панель графики не показана"
    gp = page.graph_panel
    assert gp.lbl_name.text() == "ДК", gp.lbl_name.text()
    assert gp.table.rowCount() == 255, gp.table.rowCount()
    assert gp.table.columnCount() == 4, gp.table.columnCount()
    headers = [gp.table.horizontalHeaderItem(i).text() for i in range(4)]
    assert headers == ["№", "p1", "p2", "C"], headers
    assert gp.table.item(0, 0).text() == "1"
    assert gp.table.item(0, 1).text() == "0.0000"
    assert gp.table.item(0, 3).text() == "1.00000", gp.table.item(0, 3).text()
    assert gp.canvas.ax.name == "3d"

    gp.btn_reset.click()
    app.processEvents()
    assert gp.table.rowCount() == 0, gp.table.rowCount()
    gp.btn_exit.click()
    app.processEvents()
    assert page.stack.currentWidget() is page.tabs, "«Выxод» не вернул вкладки"

    page.dsks.btn_graph.click()
    app.processEvents()
    assert gp.lbl_name.text() == "ДCКC", gp.lbl_name.text()
    headers = [gp.table.horizontalHeaderItem(i).text() for i in range(4)]
    assert headers == ["№", "p", "q", "C"], headers
    row = [(gp.table.item(0, c).text()) for c in range(4)]
    assert row[3] == "1.00000", row
    gp.btn_exit.click()
    app.processEvents()


check("Panel8: «Канал», таблица №/p1/p2/C (№/p/q/C), «Сброс», «Выxод»", c_graph_panel)


# --- 10. скриншоты ------------------------------------------------------------

def shot(name: str) -> None:
    app.processEvents()
    path = OUT / name
    page.grab().save(str(path))
    assert path.exists() and path.stat().st_size > 2000, f"скриншот пуст: {path}"
    print(f"     shot {path.name} {path.stat().st_size // 1024} KiB")


def c_shots():
    page.dk.f_p1.set_value(0.05)
    page.dk.f_p2.set_value(0.10)
    page.dsks.f_p.set_value(0.05)
    page.dsks.f_q.set_value(0.125)
    page.dk.compute()
    page.dsks.compute()
    page.stack.setCurrentWidget(page.tabs)
    names = ("lab6-1-theory.png", "lab6-2-dk.png", "lab6-3-dsks.png",
             "lab6-4-code.png", "lab6-5-surface-dk.png", "lab6-6-surface-dsks.png")
    for idx, name in enumerate(names):
        page.tabs.setCurrentIndex(idx)
        shot(name)
    page.code.cb_code.setCurrentText("КОД (7,4)")
    page.code.btn_fill.click()
    page.code.stack.setCurrentIndex(0)
    page.code.result_panel.setVisible(False)
    page.code.stack.setVisible(True)
    page.code.tasks[1].compute()
    page.tabs.setCurrentIndex(3)
    shot("lab6-7-code-solved.png")
    page.code.btn_next.click()
    page.code.tasks[2].compute()
    shot("lab6-8-code-task2.png")
    page.dk.btn_graph.click()
    shot("lab6-9-graph-dk.png")
    page.dsks.btn_graph.click()
    shot("lab6-10-graph-dsks.png")
    page.close_graph()


check("скриншоты всех вкладок и панели графики", c_shots)


print(f"\n{len(passed)} passed, {len(failures)} failed")
if failures:
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
