from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtWidgets import QApplication

from kuotik_port.labs.lab3 import KINDS, Lab3Page, dtft, fold_freq, spectrum_one_sided

OUT = Path(os.environ.get("SHOTS", "/tmp/shots-lab3"))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication([])
page = Lab3Page()
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
    else:
        passed.append(name)
        print(f"PASS {name}")


def approx(a: float, b: float, tol: float, what: str = ""):
    assert abs(a - b) <= tol, f"{what}: {a!r} != {b!r} (tol {tol})"


def set_harm(*, n: int = 256, amp: float = 1.0, freq: float = 125.0,
             dt: float = 1.0, quant: float = 0.0, wc: float | None = None):
    page.cb_kind.setCurrentText(KINDS[0])
    page.cb_n.setCurrentText(str(n))
    page.sp_amp.setValue(amp)
    page.sp_freq.setValue(freq)
    page.sp_phase.setValue(0.0)
    page.sp_dt.setValue(dt)
    page.sp_quant.setValue(quant)
    page.sp_wc.setValue(math.pi / (dt * 1e-3) if wc is None else wc)
    page.model()
    app.processEvents()


def assert_equal(a, b):
    assert a == b, f"{a!r} != {b!r}"


expected_tabs = ["Общие сведения", "Схема установки", "Осциллограф", "Спектр амплитуд",
                 "Корреляционная функция", "Спектральная плотность мощности"]
check("вкладки формы", lambda: (
    assert_equal(page.tabs.count(), 6),
    assert_equal([page.tabs.tabText(i) for i in range(6)], expected_tabs)))


def c_sampling():
    set_harm(dt=1.0)
    approx(page.fd, 1000.0, 1e-9, "Fд при Δt=1 мс")
    approx(page.fd_value, 1000.0, 1e-9, "поле Fд")
    approx(page.fd / 2, 500.0, 1e-9, "Fн")
    page.sp_dt.setValue(0.25)
    page.model()
    approx(page.fd, 4000.0, 1e-9, "Fд при Δt=0.25 мс")


check("частота дискретизации и граница Найквиста", c_sampling)


def c_harmonic_peak():
    set_harm(freq=125.0, amp=1.0, dt=1.0, n=256)
    f, amp, spec = page.spectrum_data()
    k = int(np.argmax(amp))
    assert abs(f[k] - 125.0) <= page.fd / page.p.n, f"пик {f[k]} Гц"
    approx(amp[k], 1.0, 1e-6, "амплитуда пика")
    for i in range(0, 12):
        z = dtft(page.x, page.p.dt_ms * 1e-3, f[i])[0]
        assert abs(z - spec[i]) < 1e-9, f"ДПФ bin {i}: {z} != {spec[i]}"


check("пик спектра гармоники 125 Гц, A = 1 В", c_harmonic_peak)


def c_copies():
    set_harm(freq=125.0)
    x, dt, fd = page.x, page.p.dt_ms * 1e-3, page.fd
    for f0 in (37.3, 111.1, 480.0):
        base = dtft(x, dt, f0)[0]
        plus = dtft(x, dt, f0 + fd)[0]
        minus = dtft(x, dt, f0 - fd)[0]
        twice = dtft(x, dt, f0 + 2 * fd)[0]
        for name, val in (("+Fд", plus), ("−Fд", minus), ("+2Fд", twice)):
            assert abs(base - val) < 1e-9, f"X({f0}) != X({f0}{name}): {abs(base - val)}"
    f, amp, _ = page.spectrum_data()
    assert np.allclose(amp, spectrum_one_sided(x, fd)[1]), "спектр на вкладке не совпал с rfft"


check("периодичность спектра: X(f+Fд)=X(f)", c_copies)


def c_alias():
    set_harm(freq=1250.0, dt=1.0)
    f, amp, _ = page.spectrum_data()
    k = int(np.argmax(amp))
    approx(f[k], 250.0, page.fd / page.p.n, "частота-образ")
    approx(amp[k], 1.0, 1e-5, "амплитуда образа")
    approx(fold_freq(1250.0, 1000.0), 250.0, 1e-12, "fold_freq")
    approx(fold_freq(750.0, 1000.0), 250.0, 1e-12, "fold_freq 750")
    assert "Котельникова" in page.status.text() and "алиасинг" in page.status.text(), \
        page.status.text()


check("наложение: 1250 Гц при Fд=1000 → 250 Гц + предупреждение", c_alias)


def c_restore():
    set_harm(freq=125.0, dt=1.0, n=256)
    xr = page.restore(page.t)
    err = float(np.abs(xr - page.x).max())
    assert err < 1e-9, f"ряд не проходит через отсчёты: {err}"
    td = page.t_dense
    mid = (td > 0.25 * td[-1]) & (td < 0.75 * td[-1])
    err_mid = float(np.abs(page.xr[mid] - page.x_cont[mid]).max())
    assert err_mid < 5e-3, f"ошибка восстановления в середине {err_mid}"
    page.sp_dt.setValue(2.0)
    page.sp_wc.setValue(math.pi / 2e-3)
    page.model()
    err2 = float(np.abs(page.restore(page.t) - page.x).max())
    assert err2 < 1e-9, f"Δt=2 мс: {err2}"


check("восстановление при Wc=π/Δt проходит через отсчёты", c_restore)


def c_quant():
    set_harm(freq=125.0, dt=1.0, quant=0.1)
    err = float(np.abs(page.xq - page.x).max())
    assert err <= 0.05 + 1e-12, f"ошибка квантования {err} > Δ/2"
    steps = np.unique(np.round(page.xq / 0.1, 9))
    assert np.allclose(steps, np.round(steps)), "отсчёты не кратны кванту"


check("квантование: max|xq−x| ≤ Δ/2", c_quant)


def c_acf():
    set_harm(freq=125.0, amp=1.0, dt=1.0)
    tau, r = page.acf_data()
    approx(r[0], 0.5, 1e-9, "R(0)")
    i = int(np.argmin(np.abs(tau - 2e-3)))
    assert abs(r[i]) < 2e-3, f"R(T/4) = {r[i]}"
    near = tau <= 4e-3
    th = 0.5 * (1 - tau[near] / (page.p.n * page.p.dt_ms * 1e-3)) * np.cos(2 * np.pi * 125.0 * tau[near])
    dev = float(np.abs(r[near] - th).max())
    assert dev < 2e-3, f"АКФ не совпала с (A²/2)(1−τ/T)cos: {dev}"


check("АКФ синусоиды: R(0)=A²/2, R(T/4)=0", c_acf)


def c_noise_psd():
    page.cb_kind.setCurrentText(KINDS[2])
    page.cb_n.setCurrentText("1024")
    page.sp_n_var.setValue(0.25)
    page.sp_n_mean.setValue(0.0)
    page.sp_dt.setValue(1.0)
    page.model()
    app.processEvents()
    var = float(page.x.var())
    approx(var, 0.25, 0.03, "выборочная дисперсия")
    f, s = page.psd_data()
    mean = float(s[1:-1].mean())
    expected = 2 * 0.25 / page.fd
    assert abs(mean - expected) / expected < 0.15, f"СПМ {mean} vs {expected}"
    total = float((s * (page.fd / page.p.n)).sum())
    power = float((page.x ** 2).mean())
    assert abs(total - power) / power < 0.02, f"Парсеваль: {total} vs {power}"


check("белый шум: дисперсия и СПМ ≈ 2σ²/Fд", c_noise_psd)


def c_pulses():
    page.cb_kind.setCurrentText(KINDS[1])
    page.cb_n.setCurrentText("256")
    page.sp_im_amp.setValue(2.0)
    page.sp_im_per.setValue(20.0)
    page.sp_im_dur.setValue(5.0)
    page.sp_dt.setValue(1.0)
    page.model()
    app.processEvents()
    assert set(np.unique(page.x)) <= {0.0, 2.0}, "импульсы не двухуровневые"
    approx(float(page.lbl_fup.text().split()[0]), 200.0, 0.1, "Fв = 1/τ")
    page.sp_im_per.setValue(1000.0)
    page.model()
    idx = np.flatnonzero(page.x > 0)
    assert idx.size == 5, f"одиночный импульс: {idx.size} отсчётов"


check("прямоугольные импульсы: уровни, Fв=1/τ, одиночный импульс", c_pulses)


def c_mixed():
    page.cb_kind.setCurrentText(KINDS[3])
    page.sp_freq.setValue(125.0)
    page.sp_im_per.setValue(20.0)
    page.sp_im_dur.setValue(5.0)
    page.sp_n_var.setValue(0.04)
    page.model()
    app.processEvents()
    assert page.x is not None and page.x.std() > 0
    assert "Смешанный" in page.cv_scope.ax.get_title()
    assert page.xr is not None


check("смешанный сигнал считается и строится", c_mixed)


def c_clear():
    set_harm(freq=125.0)
    page.clear()
    app.processEvents()
    assert page.x is None and page.t is None and page.xr is None, "данные не очищены"
    assert page.status.text() == "Данные очищены", page.status.text()
    page.model()
    app.processEvents()
    assert page.x is not None, "после очистки модель не пересчиталась"


check("Очистить / Обновить", c_clear)


def shot(name: str):
    app.processEvents()
    path = OUT / name
    page.grab().save(str(path))
    assert path.exists() and path.stat().st_size > 1000, f"скриншот пуст: {path}"
    print(f"     shot {path} {path.stat().st_size // 1024} KiB")


for i in range(page.tabs.count()):
    page.tabs.setCurrentIndex(i)
    app.processEvents()
    shot(f"lab3-{i + 1}-tab.png")

set_harm(freq=125.0)
page.tabs.setCurrentIndex(2)
shot("lab3-scope-harmonic.png")

set_harm(freq=1250.0)
page.tabs.setCurrentIndex(3)
shot("lab3-spectrum-alias.png")

page.cb_kind.setCurrentText(KINDS[1])
page.tabs.setCurrentIndex(2)
shot("lab3-scope-pulses.png")

page.cb_kind.setCurrentText(KINDS[2])
page.tabs.setCurrentIndex(5)
shot("lab3-psd-noise.png")

page.cb_kind.setCurrentText(KINDS[3])
page.tabs.setCurrentIndex(4)
shot("lab3-acf-mixed.png")

page.tabs.setCurrentIndex(0)

print(f"\n{len(passed)} passed, {len(failures)} failed")
if failures:
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("OK; скриншоты:", OUT)
