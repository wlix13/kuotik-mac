"""Headless-проверка ЛР4 (Form4): вкладки, пороги трёх правил, вероятности, АКФ/СПМ, фильтр."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
from PySide6.QtWidgets import QApplication           # noqa: E402

from kuotik_port.labs.lab4 import (KINDS, Lab4Page, acf_biased,  # noqa: E402
                                   acf_rc, likelihood_ratio, matched_output,
                                   matched_snr, mean_risk, p_error, pdf, periodogram,
                                   posterior, power_from_psd, psd_from_acf, psd_rc,
                                   qfunc, threshold)

OUT = Path(os.environ.get("SHOTS", "/tmp/shots-lab4"))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication([])
page = Lab4Page()
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
    except Exception as exc:                       # noqa: BLE001
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
        print(f"PASS {name}")


def approx(a: float, b: float, tol: float, what: str = ""):
    assert abs(a - b) <= tol, f"{what}: {a!r} != {b!r} (tol {tol})"


def bisect_root(fn, lo: float, hi: float) -> float:
    flo = fn(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- структура ---------------------------------------------------------------

def c_tabs():
    titles = ["Задача1", "Задача2", "Задача3", "Исследование"]
    assert page.tabs.count() == 4, page.tabs.count()
    assert [page.tabs.tabText(i) for i in range(4)] == titles
    counts = [page.pc1.count(), page.pc2.count(), page.pc3.count(), page.pc4.count()]
    assert counts == [4, 2, 2, 2], counts
    inner = [page.pc1.tabText(i) for i in range(4)]
    assert inner == ["Общие сведения", "Спектральные характеристики", "Ввод данных",
                     "Отображение функций"], inner
    assert page.pc4.tabText(0) == "Ввод данных" and page.pc4.tabText(1) == "Отображение функции"


check("структура: 4 внешние вкладки, 4/2/2/2 внутренних", c_tabs)


# --- нормировка плотностей и отношения правдоподобия -------------------------

def c_normalization():
    a, m, s = 1.7, -0.4, 0.9
    x = np.linspace(m - 12 * s, m + a + 12 * s, 200001)
    approx(float(np.trapezoid(pdf(x, m, s), x)), 1.0, 1e-6, "∫f(x|a1)")
    approx(float(np.trapezoid(pdf(x, m + a, s), x)), 1.0, 1e-6, "∫f(x|a2)")
    # ∫L(x)·f(x|a1)dx = ∫f(x|a2)dx = 1 — L есть отношение плотностей
    approx(float(np.trapezoid(likelihood_ratio(x, a, m, s) * pdf(x, m, s), x)), 1.0, 1e-6,
           "∫L·f1")
    # L монотонна по x и равна 1 в m + a/2
    vals = likelihood_ratio(np.array([m, m + a / 2, m + a]), a, m, s)
    assert vals[0] < vals[1] < vals[2], vals
    approx(float(vals[1]), 1.0, 1e-12, "L(m+a/2)")


check("нормировка плотностей и L(x)=f2/f1", c_normalization)


# --- порог: максимум правдоподобия ------------------------------------------

def c_threshold_maxlik():
    a, m, s = 2.0, 0.5, 1.0
    xp = threshold(a, m, s, 1.0)
    approx(xp, m + a / 2, 1e-12, "Xп = m + a/2")
    approx(xp, 1.5, 1e-12, "Xп численно")
    approx(float(likelihood_ratio(xp, a, m, s)), 1.0, 1e-12, "L(Xп)=1")
    page.tabs.setCurrentIndex(0)
    page.pc1.setCurrentIndex(2)
    page.t1.a.setValue(a)
    page.t1.m.setValue(m)
    page.t1.sigma.setValue(s)
    page.t1.lam.setValue(1.0)
    page.t1.compute()
    approx(page.t1.xp_calc, xp, 1e-12, "порог на странице")
    assert "Xп = 1.500" in page.t1.summary(), page.t1.summary()


check("порог максимума правдоподобия: Xп = m + a/2", c_threshold_maxlik)


# --- порог: идеальный наблюдатель -------------------------------------------

def c_threshold_ideal():
    a, m, s, p1, p2 = 2.0, 0.0, 1.0, 0.8, 0.2
    xp = threshold(a, m, s, p1 / p2)
    approx(xp, a / 2 + s ** 2 * math.log(4.0) / a, 1e-12, "Xп идеального наблюдателя")
    root = bisect_root(lambda x: p1 * float(pdf(x, m, s)) - p2 * float(pdf(x, m + a, s)),
                       m - 10 * s, m + a + 10 * s)
    approx(xp, root, 1e-6, "корень p1·f1 = p2·f2")
    page.t1.p1.setValue(p1)
    page.t2.p1.setValue(p1)
    page.t2.p2.setValue(p2)
    page.t2.a.setValue(a)
    page.t2.m.setValue(m)
    page.t2.sigma.setValue(s)
    page.t2.compute()
    approx(page.t2.lam_value(), 4.0, 1e-12, "λ = p1/p2")
    approx(page.t2.xp_calc, root, 1e-6, "порог на странице")


check("порог идеального наблюдателя: λ = p1/p2", c_threshold_ideal)


# --- порог: минимальный риск -------------------------------------------------

def c_threshold_risk():
    a, m, s, p1, p2, r12, r21 = 2.0, 0.0, 1.0, 0.5, 0.5, 0.1, 1.0
    lam = p1 * r12 / (p2 * r21)
    approx(lam, 0.1, 1e-12, "λ = p1·r12/(p2·r21)")
    xp = threshold(a, m, s, lam)
    root = bisect_root(
        lambda x: p1 * r12 * float(pdf(x, m, s)) - p2 * r21 * float(pdf(x, m + a, s)),
        m - 10 * s, m + a + 10 * s)
    approx(xp, root, 1e-6, "корень p1·r12·f1 = p2·r21·f2")
    approx(xp, a / 2 + s ** 2 * math.log(0.1) / a, 1e-9, "Xп аналитически")
    page.t3.a.setValue(a)
    page.t3.m.setValue(m)
    page.t3.sigma.setValue(s)
    page.t3.p1.setValue(p1)
    page.t3.p2.setValue(p2)
    page.t3.r12.setValue(r12)
    page.t3.r21.setValue(r21)
    page.t3.compute()
    approx(page.t3.xp_calc, root, 1e-6, "порог на странице")
    assert "λ = 0.100" in page.t3.summary(), page.t3.summary()


check("порог минимального риска: λ = p1·r12/(p2·r21)", c_threshold_risk)


# --- вероятность ошибки и риск ----------------------------------------------

def c_probabilities():
    a, m, s, p1, p2, r12, r21 = 1.5, -0.2, 0.8, 0.7, 0.3, 2.0, 5.0
    xp = 0.9
    x = np.linspace(m - 12 * s, m + a + 12 * s, 400001)
    num = (p1 * float(np.trapezoid(np.where(x > xp, pdf(x, m, s), 0.0), x))
           + p2 * float(np.trapezoid(np.where(x < xp, pdf(x, m + a, s), 0.0), x)))
    approx(p_error(a, m, s, p1, p2, xp), num, 1e-6, "Pош формула/интеграл")
    r_num = (p1 * r12 * float(np.trapezoid(np.where(x > xp, pdf(x, m, s), 0.0), x))
             + p2 * r21 * float(np.trapezoid(np.where(x < xp, pdf(x, m + a, s), 0.0), x)))
    approx(mean_risk(a, m, s, p1, p2, r12, r21, xp), r_num, 1e-5, "риск формула/интеграл")
    # Q-функция
    approx(qfunc(0.0), 0.5, 1e-12, "Q(0)")
    approx(qfunc(1.0), 0.5 * math.erfc(1 / math.sqrt(2)), 1e-12, "Q(1)")
    approx(qfunc(-1.0), 1 - qfunc(1.0), 1e-12, "Q(-1)")


check("Pош и риск: формула совпадает с интегралом плотностей", c_probabilities)


def c_optimum():
    a, m, s, p1, p2 = 1.2, 0.1, 1.0, 0.6, 0.4
    lam = p1 / p2
    xp = threshold(a, m, s, lam)
    base = p_error(a, m, s, p1, p2, xp)
    for dx in (-0.5, -0.2, -0.05, 0.05, 0.2, 0.5):
        assert p_error(a, m, s, p1, p2, xp + dx) > base, f"Xп+{dx} даёт меньшую ошибку"
    grid = np.linspace(m - 4 * s, m + a + 4 * s, 20001)
    errs = [p_error(a, m, s, p1, p2, v) for v in grid]
    assert abs(grid[int(np.argmin(errs))] - xp) <= grid[1] - grid[0], "минимум не в Xп"


check("Xп* минимизирует Pош", c_optimum)


# --- апостериорные вероятности ----------------------------------------------

def c_posterior():
    a, m, s = 1.0, 0.0, 1.0
    p1, p2 = 0.9, 0.1
    hi = posterior(a, m, s, p1, p2, m + 30 * s)
    lo = posterior(a, m, s, p1, p2, m - 30 * s)
    approx(hi[0], p1, 1e-6, "P(a1|X1) при Xп→+∞")
    approx(lo[0], 1.0, 1e-6, "P(a1|X1) при Xп→−∞")
    # монотонность по порогу: шире область X1 — апостериорная падает к p(a1)
    grid = np.linspace(m - 4 * s, m + a + 4 * s, 200)
    pa1 = [posterior(a, m, s, p1, p2, xp)[0] for xp in grid]
    assert np.all(np.diff(pa1) < 0), "P(a1|X1) не убывает с ростом порога"
    assert np.all(np.diff(pa1) > -0.01), "скачок апостериорной"
    for xp in (-2.0, -0.5, 0.0, 0.5, 2.0):
        pa1v, pa2v = posterior(a, m, s, p1, p2, xp)
        assert 0.0 <= pa1v <= 1.0 and 0.0 <= pa2v <= 1.0, (pa1v, pa2v)
    # оптимальный порог: апостериорные не ниже априорных внутри своих областей
    xp = threshold(a, m, s, p1 / p2)
    pa1v, pa2v = posterior(a, m, s, p1, p2, xp)
    assert pa1v >= p1 - 1e-12, f"P(a1|X1)={pa1v} < p1={p1}"
    assert pa2v >= p2 - 1e-12, f"P(a2|X2)={pa2v} < p2={p2}"


check("апостериорные вероятности: пределы и допустимость", c_posterior)


# --- кнопки -----------------------------------------------------------------

def c_buttons():
    page.t1.a.setValue(2.0)
    page.t1.m.setValue(0.0)
    page.t1.sigma.setValue(1.0)
    page.t1.lam.setValue(1.0)
    page.t1.compute()
    page.t1.xp.setValue(round(page.t1.xp_calc, 3))
    page.t1.btn_compute.click()
    assert page.t1.ok, page.t1.verdict.text()
    assert "Верно" in page.t1.verdict.text(), page.t1.verdict.text()
    page.t1.xp.setValue(page.t1.xp_calc + 0.5)
    page.t1.btn_compute.click()
    assert not page.t1.ok, page.t1.verdict.text()
    assert "Неверно" in page.t1.verdict.text(), page.t1.verdict.text()
    page.t1.btn_clear.click()
    assert page.t1.verdict.text() == "", page.t1.verdict.text()
    approx(page.t1.xp.value(), 0.0, 1e-12, "Xп после «Обнулить»")
    page.t1.btn_upd.click()
    assert page.t1.xp_calc > 0, page.t1.xp_calc


check("кнопки «Проверить», «Обнулить», «Обновить»", c_buttons)


# --- спектральные характеристики --------------------------------------------

def c_wiener_khinchin():
    sigma2, alpha = 4.0, 200.0
    w = np.array([10.0, 50.0, 100.0, 200.0, 400.0, 800.0])
    num = psd_from_acf(sigma2, alpha, w)
    th = psd_rc(w, sigma2, alpha)
    err = float(np.max(np.abs(num - th) / th))
    assert err < 0.02, f"Φ(ω) из R(τ): относительная ошибка {err}"
    # ∫Φ(ω)dω/2π = σ² = R(0)
    ww = np.linspace(-2000 * alpha, 2000 * alpha, 200001)
    power = power_from_psd(ww, psd_rc(ww, sigma2, alpha))
    approx(power, sigma2, 0.01 * sigma2, "мощность через СПМ")
    approx(float(acf_rc(0.0, sigma2, alpha)), sigma2, 1e-12, "R(0)=σ²")


check("Винер–Хинчин: Φ(ω) из R(τ) и ∫Φdω/2π = R(0)", c_wiener_khinchin)


def c_rc_process():
    page.tabs.setCurrentIndex(0)
    page.pc1.setCurrentIndex(1)
    page.spec_kind.setCurrentText(KINDS[1])
    page.spec_sigma.setValue(1.0)
    page.spec_alpha.setValue(200.0)
    page.spec_fd.setValue(10000.0)
    page.spec_n.setCurrentText("1024")
    page.draw_spectral()
    app.processEvents()
    kind, x, fs = page.spectral_data()
    assert kind == KINDS[1] and len(x) == 1024
    r = acf_biased(x)
    approx(float(r[0]), 1.0, 0.10, "R(0) выборочная")
    tau = np.arange(6) / fs
    th = acf_rc(tau, 1.0, 200.0)
    dev = float(np.max(np.abs(r[:6] - th)))
    assert dev < 0.12, f"АКФ АР-1 расходится с σ²e^(−ατ): {dev}"
    # средняя мощность ≈ дисперсия ≈ R(0)
    approx(float((x ** 2).mean()), float(r[0]), 0.05, "мощность процесса")


check("стационарный процесс: R(0)=σ², АКФ ≈ σ²e^(−ατ)", c_rc_process)


def c_white_noise():
    page.spec_kind.setCurrentText(KINDS[2])
    page.spec_sigma.setValue(2.0)
    page.spec_fd.setValue(10000.0)
    page.spec_n.setCurrentText("1024")
    page.draw_spectral()
    app.processEvents()
    _, x, fs = page.spectral_data()
    r = acf_biased(x)
    approx(float(r[0]), 4.0, 0.5, "R(0) белого шума")
    f, s = periodogram(x, fs)
    mean = float(s[1:-1].mean())
    expected = 2 * 4.0 / fs
    assert abs(mean - expected) / expected < 0.2, f"средний уровень СПМ {mean} vs {expected}"
    # ровность: разброс одного бина ~ уровню (хи-квадрат с 2 степенями свободы)
    assert float(s[1:-1].std() / mean) < 2.0, "СПМ не ровная"


check("белый шум: R(0)=σ², СПМ ровная, уровень 2σ²/Fд", c_white_noise)


def c_const_level():
    page.spec_kind.setCurrentText(KINDS[0])
    page.spec_amp.setValue(3.0)
    page.draw_spectral()
    app.processEvents()
    _, x, fs = page.spectral_data()
    assert np.allclose(x, 3.0), "постоянный уровень задан неверно"
    r = acf_biased(x)
    approx(float(r[0]), 9.0, 1e-9, "R(0)=A²")
    f, s = periodogram(x, fs)
    total = float((s * (fs / len(x))).sum())
    approx(total, 9.0, 1e-6, "мощность по периодограмме = A²")


check("постоянный уровень: R(0)=A², мощность A²", c_const_level)


# --- согласованный фильтр ----------------------------------------------------

def c_matched_filter():
    a, t_dur, n0 = 1.0, 1e-3, 0.01
    fs = max(2000.0, 200.0 / t_dur)
    dt = 1.0 / fs
    t = np.arange(0, 2.2 * t_dur, dt)
    s = np.where((t >= 0) & (t < t_dur), a, 0.0)
    y = matched_output(t, s, s, dt, t_dur)
    k = int(np.argmax(np.abs(y)))
    approx(float(t[k]), t_dur, dt, "максимум выхода при t0=T")
    approx(float(y[k]), a * a * t_dur, 1e-9, "y(t0)=A²T=E")
    # SNR = 2E/N0 точно и как (∫s·h)²/((N0/2)∫h²)
    e = float(np.trapezoid(s * s, t))
    snr = e ** 2 / ((n0 / 2) * float(np.trapezoid(s * s, t)))
    # сетка даёт O(dt) поправку к интегралу, поэтому 1 %
    approx(snr, matched_snr(a, t_dur, n0), 1e-2 * matched_snr(a, t_dur, n0), "SNR = 2E/N0")
    # фильтр вдвое короче — хуже
    short = np.where((t >= 0) & (t < t_dur / 2), a, 0.0)
    snr_alt = (float(np.trapezoid(s * short, t)) ** 2
               / ((n0 / 2) * float(np.trapezoid(short * short, t))))
    assert snr_alt < snr, f"короткий фильтр дал {snr_alt} >= {snr}"
    # страница рисует этот же пик
    page.mf_a.setValue(a)
    page.mf_t.setValue(t_dur * 1e3)
    page.mf_n0.setValue(n0)
    page.draw_mf()
    app.processEvents()
    tit = page.cv_mf.ax.get_title()
    assert "Согласованный" in tit, tit
    page.mf_noise.setChecked(True)
    page.draw_mf()
    page.mf_noise.setChecked(False)
    page.draw_mf()


check("согласованный фильтр: пик в t0, SNR=2E/N0, короче — хуже", c_matched_filter)


# --- очистка -----------------------------------------------------------------

def c_clear():
    page.t2.xp.setValue(1.234)
    page.t2.verdict.setText("мусор")
    page.clear_spectral()
    app.processEvents()
    assert "нет данных" in page.cv_psd.ax.get_title(), page.cv_psd.ax.get_title()
    assert page.spec_caption.text() == "", page.spec_caption.text()
    page.clear()
    app.processEvents()
    assert page.status.text() == "Данные очищены", page.status.text()
    assert page.t2.verdict.text() == "", page.t2.verdict.text()


check("«Очистить»: графики и поля пусты", c_clear)


# --- вкладки и скриншоты -----------------------------------------------------

def shot(name: str):
    page.redraw_all()
    app.processEvents()
    path = OUT / name
    page.grab().save(str(path))
    assert path.exists() and path.stat().st_size > 1000, f"скриншот пуст: {path}"
    print(f"     shot {path} {path.stat().st_size // 1024} KiB")


# рабочее состояние для скриншотов
for panel in (page.t1, page.t2, page.t3, page.t4):
    panel.compute()
page.draw_spectral()
app.processEvents()

for i in range(page.tabs.count()):
    page.tabs.setCurrentIndex(i)
    inner = page.tabs.widget(i)
    for j in range(inner.count()):
        inner.setCurrentIndex(j)
        app.processEvents()
        shot(f"lab4-{i + 1}-{j + 1}-{inner.tabText(j).replace(' ', '-')}.png")
    assert inner.count() in (2, 4)

page.tabs.setCurrentIndex(0)
page.pc1.setCurrentIndex(2)
page.t1.xp.setValue(round(page.t1.xp_calc, 3))
page.t1.btn_compute.click()
shot("lab4-verdict.png")

page.pc1.setCurrentIndex(1)
page.spec_kind.setCurrentText(KINDS[1])
page.draw_spectral()
shot("lab4-spectral-rc.png")

page.tabs.setCurrentIndex(3)
page.pc4.setCurrentIndex(0)
page.t4.compute()
shot("lab4-study.png")

print(f"\n{len(passed)} passed, {len(failures)} failed")
if failures:
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("OK; скриншоты:", OUT)
