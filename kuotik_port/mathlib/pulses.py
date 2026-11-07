"""Экспоненциальный импульс ЛР2 (рис. 2.5): x(t) = h·e^(−β(t − t0)) при t ≥ t0, 0 при t < t0.

Спектральная плотность (теория, формулы 2.4–2.5):
    S(jω) = ∫x(t)e^(−jωt)dt = h·e^(−jωt0)/(β + jω) = A(ω) − jB(ω) = S(ω)·e^(jΘ(ω)),
    S(ω) = h/√(β² + ω²),   Θ(ω) = −ωt0 − arctg(ω/β).
Сигнал включения — предел β → 0: x(t) = h при t ≥ t0; S(ω) = h/ω, Θ(ω) = −ωt0 − π/2 (ω > 0).
Энергия: Э = ∫x²dt = h²/(2β) = (1/π)∫₀^∞ S²dω (равенство Парсеваля); доля энергии в полосе 0…ωc:
    λ(ωc) = (2/π)·arctg(ωc/β)  →  ωc = β·tg(πλ/2),  Δt = π/ωc.
Ряд Котельникова так, как его строит программа (проверено по графикам): ωc — из «Энергии в спектре»,
Δt — введённое, отсчёты с t = 0 (t0 = 0):  x̃(t) = Σ_{k≥0} x(kΔt)·sin(ωc(t − kΔt))/(ωc(t − kΔt)).
Программа принимает Δt только из [Δt0/2; 3Δt0/2], Δt0 = π/ωc.
Каждая формула сверяется с численным интегрированием (check()).
"""
import math

import numpy as np

PI = math.pi


def x(t, h, b, t0):
    t = np.asarray(t, float)
    return np.where(t >= t0, h * np.exp(-b * (t - t0)), 0.0)


def S(w, h, b):
    w = np.asarray(w, float)
    with np.errstate(divide="ignore"):
        return h / np.sqrt(b * b + w * w)


def theta(w, h, b, t0):
    """Аргумент спектральной плотности, рад (без приведения к [−π, π): так рисует программа)."""
    w = np.asarray(w, float)
    return -w * t0 - (np.arctan(w / b) if b > 0 else np.where(w > 0, PI / 2, np.nan))


def AB(w, h, b, t0):
    """A(ω), B(ω): S(jω) = A − jB."""
    w = np.asarray(w, float)
    d = b * b + w * w
    return (h * (b * np.cos(w * t0) - w * np.sin(w * t0)) / d,
            h * (b * np.sin(w * t0) + w * np.cos(w * t0)) / d)


def energy(h, b):
    return h * h / (2 * b)


def lam(wc, b):
    return 2 / PI * math.atan(wc / b)


def wc_of(lam_, b):
    return b * math.tan(PI * lam_ / 2)


def dt_of(wc):
    return PI / wc


def kotelnikov_range(dt0):
    return dt0 / 2, 3 * dt0 / 2


def kotelnikov(t, h, b, wc, dt, kmax=400):
    """Сумма ряда так, как её рисует программа (t0 = 0)."""
    t = np.asarray(t, float)
    k = np.arange(kmax)
    return (h * np.exp(-b * k * dt) * np.sinc(wc * (t[:, None] - k * dt) / PI)).sum(1)


def check():
    """Формулы против численного интегрирования."""
    for h, b, t0 in ((2, 1, 0), (2, 0.5, 1), (3, 2, 2), (1, 0.4, -1)):
        t = np.linspace(t0, t0 + 60 / b, 2_000_001)
        dt = t[1] - t[0]
        xs = x(t, h, b, t0)
        for w in (0.0, 0.6, 1.2, 2.4):
            e = np.exp(-1j * w * t)
            Sj = (xs * e).sum() * dt - (xs[0] * e[0] + xs[-1] * e[-1]) * dt / 2     # трапеции
            assert abs(abs(Sj) - S(w, h, b)) < 1e-4, (h, b, t0, w, abs(Sj), S(w, h, b))
            if w:
                dphi = (np.angle(Sj) - theta(w, h, b, t0) + PI) % (2 * PI) - PI
                assert abs(dphi) < 1e-4, (h, b, t0, w, np.angle(Sj), theta(w, h, b, t0))
                A, B = AB(w, h, b, t0)
                assert abs(Sj - (A - 1j * B)) < 1e-4
        Ex = (xs ** 2).sum() * dt - (xs[0] ** 2 + xs[-1] ** 2) * dt / 2
        assert abs(Ex - energy(h, b)) < 1e-3, (Ex, energy(h, b))
        wv = np.linspace(0, 4000 * b, 4_000_001)                        # Парсеваль: (1/π)∫₀^∞ S²dω
        Ew = np.trapezoid(S(wv, h, b) ** 2, wv) / PI
        assert abs(Ew - energy(h, b)) / energy(h, b) < 1e-3, (Ew, energy(h, b))
        for l in (0.5, 0.9, 0.95):
            wc = wc_of(l, b)
            part = np.trapezoid(S(np.linspace(0, wc, 200001), h, b) ** 2, np.linspace(0, wc, 200001)) / PI
            assert abs(part / energy(h, b) - l) < 1e-6, (l, part / energy(h, b))
    # при Δt = π/ωc ряд Котельникова проходит через отсчёты
    wc = wc_of(0.9, 0.4)
    dt0 = dt_of(wc)
    k = np.arange(1, 6)
    assert np.allclose(kotelnikov(k * dt0, 2, 0.4, wc, dt0), x(k * dt0, 2, 0.4, 0), atol=1e-9)
