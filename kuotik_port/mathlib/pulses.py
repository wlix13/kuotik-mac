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
    w = np.asarray(w, float)
    return -w * t0 - (np.arctan(w / b) if b > 0 else np.where(w > 0, PI / 2, np.nan))


def AB(w, h, b, t0):
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
    t = np.asarray(t, float)
    k = np.arange(kmax)
    return (h * np.exp(-b * k * dt) * np.sinc(wc * (t[:, None] - k * dt) / PI)).sum(1)


def check():
    for h, b, t0 in ((2, 1, 0), (2, 0.5, 1), (3, 2, 2), (1, 0.4, -1)):
        t = np.linspace(t0, t0 + 60 / b, 2_000_001)
        dt = t[1] - t[0]
        xs = x(t, h, b, t0)
        for w in (0.0, 0.6, 1.2, 2.4):
            e = np.exp(-1j * w * t)
            Sj = (xs * e).sum() * dt - (xs[0] * e[0] + xs[-1] * e[-1]) * dt / 2
            assert abs(abs(Sj) - S(w, h, b)) < 1e-4, (h, b, t0, w, abs(Sj), S(w, h, b))
            if w:
                dphi = (np.angle(Sj) - theta(w, h, b, t0) + PI) % (2 * PI) - PI
                assert abs(dphi) < 1e-4, (h, b, t0, w, np.angle(Sj), theta(w, h, b, t0))
                A, B = AB(w, h, b, t0)
                assert abs(Sj - (A - 1j * B)) < 1e-4
        Ex = (xs ** 2).sum() * dt - (xs[0] ** 2 + xs[-1] ** 2) * dt / 2
        assert abs(Ex - energy(h, b)) < 1e-3, (Ex, energy(h, b))
        wv = np.linspace(0, 4000 * b, 4_000_001)
        Ew = np.trapezoid(S(wv, h, b) ** 2, wv) / PI
        assert abs(Ew - energy(h, b)) / energy(h, b) < 1e-3, (Ew, energy(h, b))
        for l in (0.5, 0.9, 0.95):
            wc = wc_of(l, b)
            part = np.trapezoid(S(np.linspace(0, wc, 200001), h, b) ** 2, np.linspace(0, wc, 200001)) / PI
            assert abs(part / energy(h, b) - l) < 1e-6, (l, part / energy(h, b))
    wc = wc_of(0.9, 0.4)
    dt0 = dt_of(wc)
    k = np.arange(1, 6)
    assert np.allclose(kotelnikov(k * dt0, 2, 0.4, wc, dt0), x(k * dt0, 2, 0.4, 0), atol=1e-9)
