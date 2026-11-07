"""Спектры сигналов ЛР1 в форме программы КУ ОТИК:

    x(t) = A0/2 + Σ |A_k|·cos(k·ω0·t + Θ_k),  ω0 = 2π/T,  k = 1..10,
    a_k = (2/T)∫x(t)cos(kω0t)dt,  b_k = (2/T)∫x(t)sin(kω0t)dt,  |A_k| = √(a_k² + b_k²),  Θ_k = −arctg(b_k/a_k)
    (с учётом четверти: Θ_k = atan2(−b_k, a_k), так как a_k·cos − b_k·sin = |A_k|·cos(· + Θ_k)).

Время — в долях периода (T = 1). Фазы приводятся к [−π, π): поле фазы в программе вмещает только ±9.999.
Каждый аналитический спектр сверяется с численным интегрированием (check()).
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

N = 10                                     # гармоник в программе
PI = math.pi


def wrap(theta):
    return (theta + PI) % (2 * PI) - PI


@dataclass
class Spectrum:
    key: str                               # для имён файлов
    title: str                             # по-русски, для таблиц и подписей
    formula: str                           # x(t) на периоде
    a0_half: float
    amps: list = field(default_factory=lambda: [0.0] * N)     # |A_k|, k = 1..N
    phases: list = field(default_factory=lambda: [0.0] * N)   # Θ_k, рад
    x: object = None                       # исходный сигнал x(t), t в долях T (для численной проверки)

    def synth(self, t):
        t = np.asarray(t, float)
        return self.a0_half + sum(a * np.cos(2 * PI * k * t + q)
                                  for k, (a, q) in enumerate(zip(self.amps, self.phases), 1))

    def table(self):
        """Строки таблицы задания: k = 0..10, A_k (для k = 0 — A0/2), Θ_k."""
        return [(0, self.a0_half, 0.0)] + [(k, a, q) for k, (a, q) in enumerate(zip(self.amps, self.phases), 1)]


def from_ab(key, title, formula, a0_half, a, b, x=None):
    amps, phases = [], []
    for ak, bk in zip(a, b):
        amp = math.hypot(ak, bk)
        amps.append(amp)
        phases.append(wrap(math.atan2(-bk, ak)) if amp > 1e-12 else 0.0)
    return Spectrum(key, title, formula, a0_half, amps, phases, x)


def numeric(x, n=N, m=400_000):
    """A0/2, a_k, b_k численно (метод средних прямоугольников по периоду [−T/2, T/2))."""
    t = (np.arange(m) + 0.5) / m - 0.5
    v = x(t)
    k = np.arange(1, n + 1)[:, None]
    return v.mean(), 2 * (v * np.cos(2 * PI * k * t)).mean(1), 2 * (v * np.sin(2 * PI * k * t)).mean(1)


def check(s, tol=1e-4):
    """Аналитический спектр против численного интегрирования исходного сигнала."""
    if s.x is None:
        return
    c, a, b = numeric(s.x)
    ref = from_ab("", "", "", c, a, b)
    assert abs(ref.a0_half - s.a0_half) < tol, (s.key, "A0/2", ref.a0_half, s.a0_half)
    for k, (A, Q, rA, rQ) in enumerate(zip(s.amps, s.phases, ref.amps, ref.phases), 1):
        assert abs(A - rA) < tol, (s.key, k, A, rA)
        if A > 1e-3:
            assert abs(wrap(Q - rQ)) < 1e-3, (s.key, k, Q, rQ)


# ---------- сигналы задания ----------

def constant(c):
    return Spectrum("const", "Постоянный уровень", f"x(t) = {c:g}", c, x=lambda t: np.full_like(t, c))


def harmonic(amp, phase=0.0, k=1):
    amps, phases = [0.0] * N, [0.0] * N
    amps[k - 1], phases[k - 1] = amp, wrap(phase)
    return Spectrum("harmonic", "Гармонический сигнал", f"x(t) = {amp:g}·cos({k}ω0t{phase:+.3f})", 0.0, amps, phases,
                    x=lambda t: amp * np.cos(2 * PI * k * t + phase))


def triangle(h):
    """Рис. 1.4: x(t) = (2h/T)|t| на [−T/2, T/2) — чётная: b_k = 0,
    a_k = (8h/T²)∫₀^{T/2} t·cos(kω0t)dt = 2h((−1)^k − 1)/(πk)² = −4h/(πk)² для нечётных k, 0 для чётных."""
    a = [-4 * h / (PI * k) ** 2 if k % 2 else 0.0 for k in range(1, N + 1)]
    return from_ab("triangle", "Униполярные треугольные импульсы (рис. 1.4)", f"x(t) = (2h/T)|t|, h = {h:g}",
                   h / 2, a, [0.0] * N, x=lambda t: 2 * h * np.abs(t))


def sawtooth(h):
    """Рис. 1.5: x(t) = (2h/T)t на [−T/2, T/2) — нечётная: a_k = 0, A0 = 0,
    b_k = (8h/T²)∫₀^{T/2} t·sin(kω0t)dt = −2h(−1)^k/(πk) = 2h(−1)^(k+1)/(πk)."""
    b = [2 * h * (-1) ** (k + 1) / (PI * k) for k in range(1, N + 1)]
    return from_ab("sawtooth", "Колебание пилообразной формы (рис. 1.5)", f"x(t) = (2h/T)t, h = {h:g}",
                   0.0, [0.0] * N, b, x=lambda t: 2 * h * t)


def meander(h):
    """Рис. 1.2 (пример из теории): b_k = 4h/(πk) для нечётных k."""
    b = [4 * h / (PI * k) if k % 2 else 0.0 for k in range(1, N + 1)]
    return from_ab("meander", "Меандр (рис. 1.2)", f"x(t) = ±h, h = {h:g}", 0.0, [0.0] * N, b,
                   x=lambda t: np.where(t >= 0, h, -h))


def rect_pulses(h, q):
    """Прямоугольные импульсы высоты h, длительности τ = T/q (скважность q), импульс по центру t = 0:
    A0/2 = h/q, a_k = (2h/(πk))·sin(πk/q), b_k = 0."""
    a = [2 * h / (PI * k) * math.sin(PI * k / q) for k in range(1, N + 1)]
    return from_ab("rect", "Прямоугольные импульсы", f"x(t) = h при |t| < τ/2, h = {h:g}, τ = T/{q:g}",
                   h / q, a, [0.0] * N, x=lambda t: np.where(np.abs(t) < 0.5 / q, h, 0.0))


def flat(level, a0_half):
    """Произвольный спектр: все A_k = level, Θ_k = 0 → x(t) = A0/2 + level·Σcos(kω0t) (ядро Дирихле:
    узкие импульсы с периодом T)."""
    return Spectrum("arbitrary", "Произвольный спектр: равные амплитуды", f"A0/2 = {a0_half:g}, A_k = {level:g}, Θ_k = 0",
                    a0_half, [level] * N, [0.0] * N)


# ---------- доп. задание ----------

def shifted(s, d):
    """x(t − d·T): Θ_k → Θ_k − 2πk·d (d < 0 — сдвиг влево, опережение; d > 0 — вправо, запаздывание)."""
    x = (lambda t: s.x(((np.asarray(t) - d) + 0.5) % 1 - 0.5)) if s.x else None
    word = "влево" if d < 0 else "вправо"
    frac = {0.25: "T/4", 0.5: "T/2", 0.125: "T/8"}.get(abs(d), f"{abs(d):g}T")
    return replace(s, key=f"{s.key}_shift", title=f"{s.title}, сдвиг {word} на {frac}",
                   formula=f"{s.formula}; x(t {'+' if d < 0 else '−'} {frac}): Θ_k {'+' if d < 0 else '−'} k·{2 * PI * abs(d):.4f}",
                   phases=[wrap(q - 2 * PI * k * d) if a > 1e-12 else 0.0
                           for k, (a, q) in enumerate(zip(s.amps, s.phases), 1)], x=x)


def freq_scaled(s, m):
    """x(m·t): частота в m раз выше — гармоника k исходного сигнала встаёт на место m·k (до 10-й)."""
    amps, phases = [0.0] * N, [0.0] * N
    for k in range(1, N // m + 1):
        amps[m * k - 1], phases[m * k - 1] = s.amps[k - 1], s.phases[k - 1]
    x = (lambda t: s.x(((m * np.asarray(t)) + 0.5) % 1 - 0.5)) if s.x else None
    return replace(s, key=f"{s.key}_x{m}", title=f"{s.title}, частота ×{m}",
                   formula=f"{s.formula}; x({m}t): A'_{{{m}k}} = A_k, Θ'_{{{m}k}} = Θ_k", amps=amps, phases=phases, x=x)


def window_max(s):
    """Максимум суммы 10 гармоник на интервале графика программы (−0.75T…0.75T) — программа его подписывает."""
    t = np.linspace(-0.75, 0.75, 30001)
    return float(s.synth(t).max())


# ---------- примеры с фото (перепроверка) ----------

PHOTOS = {
    "Задачи.jpg, левое окно (треугольник, h = 2)": (
        triangle(2), 1.000, [0.811, 0, 0.090, 0, 0.032, 0, 0.017, 0, 0.010, 0],
        [-3.140, 0, -3.140, 0, -3.140, 0, -3.140, 0, -3.140, 0]),
    "Задачи.jpg, правое окно (пила, h = 2)": (
        sawtooth(2), 0.000, [1.273, 0.637, 0.424, 0.318, 0.255, 0.212, 0.181, 0.159, 0.141, 0.127],
        [-1.570, 1.570] * 5),
    "Сдвиг на Т делённое на4.jpg (пила, h = 1, сдвиг влево на T/4)": (
        shifted(sawtooth(1), -0.25), 0.000, [0.636, 0.318, 0.212, 0.159, 0.127, 0.106, 0.090, 0.079, 0.075, 0.063],
        [-0.000, 4.712, 3.142, 7.853, 6.284, 4.707, 9.425, 8.047, 6.280, 4.704]),
}


def compare_photo(ref, a0, amps, phases, exact=0.0006, minor=0.002):
    """Отличия значений с фото от верных: [(поле, на фото, верно, ошибка?)].
    Верная фаза берётся в той же ветви 2π, что и на фото (4.707 сравнивается с 4.712, а не с −1.571).
    Отличие ≤ minor — усечение (π → 3.14, π/2 → 1.57, 0.1819 → 0.181), больше — ошибка."""
    diffs = []
    rows = [("A0/2", a0, ref.a0_half)] + [(f"A{k}", amps[k - 1], ref.amps[k - 1]) for k in range(1, N + 1)]
    rows += [(f"Θ{k}", phases[k - 1], phases[k - 1] + wrap(ref.phases[k - 1] - phases[k - 1]))
             for k in range(1, N + 1) if ref.amps[k - 1] > 1e-3]
    for name, got, true in rows:
        if abs(got - true) > exact:
            diffs.append((name, got, true, abs(got - true) > minor))
    return diffs
