"""ЛР4: случайные сигналы и оптимальный приём — форма Form4.

Разделы: спектральные характеристики случайных процессов (реализации, АКФ, СПМ,
согласованный фильтр) и синтез обнаружителя постоянного сигнала для трёх правил.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..plot import Canvas
from ..theory import apply_theory

SQRT2 = math.sqrt(2.0)
KINDS = ("Постоянный уровень", "Стационарный (RC)", "«Белый» шум")
NS = (256, 512, 1024, 2048, 4096)

THEORY = """<h3>Математические модели случайных сигналов и элементы оптимального приёма</h3>
<p>Сигнал до приёма — случайный процесс X(t), после приёма известна одна его реализация.
Усреднение комплексной спектральной плотности по реализациям даёт нуль, поэтому вводят
спектральную плотность мощности — преобразование Фурье корреляционной функции:</p>
<p style="margin-left:18px">Φ(ω) = ∫R(τ)·e<sup>−jωτ</sup>dτ = 2∫₀<sup>∞</sup>R(τ)cos(ωτ)dτ,
&nbsp; R(τ) = (1/π)∫₀<sup>∞</sup>Φ(ω)cos(ωτ)dω&nbsp; — преобразования Хинчина–Винера.</p>
<p>При τ = 0: R(0) = (1/π)∫₀<sup>∞</sup>Φ(ω)dω — средняя мощность процесса.</p>
<p>Задача оптимального приёма — использовать сведения о сигнале, помехе и канале для роста
вероятности правильного приёма. Фильтр улучшает отношение сигнал/шум, решающее устройство
выполняет обнаружение, различение или восстановление.</p>
<p>Обнаружение: пространство признаков делится на области X<sub>1</sub> (сигнала нет) и
X<sub>2</sub> (сигнал есть), граница — решающая поверхность. Вместо многомерной поверхности
сравнивают отношение правдоподобия L = f(X/a<sub>2</sub>)/f(X/a<sub>1</sub>) с порогом λ.</p>
<p>Качество распознавания: P<sub>ош</sub> = Σ p(a<sub>i</sub>)·P(принято X<sub>j≠i</sub>/a<sub>i</sub>),
средний риск r = Σ<sub>i≠j</sub> p(a<sub>i</sub>)·r<sub>ij</sub>·P(X<sub>j</sub>/a<sub>i</sub>).</p>
<h4>Критерии</h4>
<ol>
<li>Максимум правдоподобия (Фишер): λ = 1.</li>
<li>Идеального наблюдателя (Зигерт–Котельников): λ = p(a<sub>1</sub>)/p(a<sub>2</sub>).</li>
<li>Минимального риска (Байес): λ = p(a<sub>1</sub>)·r<sub>12</sub>/(p(a<sub>2</sub>)·r<sub>21</sub>).</li>
</ol>
<p>Порог по отсчёту при гауссовой помехе N(m, σ): X<sub>п</sub> = m + a/2 + (σ²/a)·ln λ.</p>
<h4>Порядок выполнения</h4>
<ol>
<li>Изучить спектральные характеристики: реализации, АКФ, СПМ.</li>
<li>Вывести выражения порога для трёх решающих правил.</li>
<li>Задача 1: критерий максимума правдоподобия, найти X<sub>п</sub>.</li>
<li>Задача 2: критерий идеального наблюдателя, найти X<sub>п</sub>.</li>
<li>Задача 3: критерий минимального риска, найти X<sub>п</sub>.</li>
<li>Исследование: влияние априорных данных на решение, P<sub>ош</sub> и риск.</li>
<li>Согласованный фильтр: максимум отношения сигнал/шум в момент t<sub>0</sub>.</li>
</ol>"""

MEMOS = {
    1: ("ЗАДАЧА 1. Обнаружение постоянного сигнала x(t) = a на фоне аддитивной помехи с нормальным "
        "распределением (m, σ). Приём — однократный отсчёт. Синтез по критерию МАКСИМУМА "
        "ПРАВДОПОДОБИЯ (λ = 1); определить пороговый уровень Xп."),
    2: ("ЗАДАЧА 2. То же, но известны априорные вероятности отсутствия p(a1) и присутствия p(a2). "
        "Синтез по критерию ИДЕАЛЬНОГО НАБЛЮДАТЕЛЯ (λ = p(a1)/p(a2)); определить Xп."),
    3: ("ЗАДАЧА 3. Известны p(a1), p(a2) и условные риски r12, r21. Синтез по критерию "
        "МИНИМАЛЬНОГО РИСКА (λ = p(a1)·r12/(p(a2)·r21)); определить Xп."),
    4: ("ИССЛЕДОВАНИЕ. Зависимость функций правдоподобия L(ai), их отношения L = L(a2)/L(a1) и "
        "апостериорных вероятностей от параметров. Найти P(ai/Xi), среднюю ошибку Pош и риск r."),
}

CRITERIA = {1: "максимум правдоподобия", 2: "идеальный наблюдатель", 3: "минимальный риск",
            4: "исследование"}


# --- математика ---------------------------------------------------------------

def phi(u):
    """Плотность стандартного нормального закона (скаляр или массив)."""
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def cdf(u):
    """Φ(u) — функция распределения N(0,1)."""
    return 0.5 * math.erfc(-u / SQRT2)


def qfunc(u):
    """Q(u) = 1 − Φ(u)."""
    return 0.5 * math.erfc(u / SQRT2)


def pdf(x, m, sigma):
    return phi((np.asarray(x, dtype=float) - m) / sigma) / sigma


def likelihood_ratio(x, a, m, sigma):
    """L(x) = f(x|a2)/f(x|a1); монотонна, L = 1 при x = m + a/2."""
    return np.exp(a * (np.asarray(x, dtype=float) - m) / sigma ** 2 - a * a / (2 * sigma ** 2))


def threshold(a, m, sigma, lam):
    """Xп из L(Xп) = λ; при λ → ∞ порог уходит в бесконечность."""
    if not math.isfinite(lam) or lam <= 0:
        return float("inf")
    return m + a / 2 + sigma ** 2 * math.log(lam) / a


def lam_ideal(p1, p2):
    return p1 / p2 if p2 > 1e-12 else float("inf")


def lam_risk(p1, p2, r12, r21):
    den = p2 * r21
    return p1 * r12 / den if den > 1e-12 else float("inf")


def p_error(a, m, sigma, p1, p2, xp):
    """Средняя вероятность ошибки распознавания."""
    xp = _finite_xp(xp)
    return p1 * qfunc((xp - m) / sigma) + p2 * cdf((xp - m - a) / sigma)


def mean_risk(a, m, sigma, p1, p2, r12, r21, xp):
    xp = _finite_xp(xp)
    return p1 * r12 * qfunc((xp - m) / sigma) + p2 * r21 * cdf((xp - m - a) / sigma)


def posterior(a, m, sigma, p1, p2, xp):
    """(P(a1|X1), P(a2|X2)) — апостериорные вероятности правильных решений."""
    xp = _finite_xp(xp)
    u1, u2 = (xp - m) / sigma, (xp - m - a) / sigma
    f1, f2 = cdf(u1), cdf(u2)
    den1 = p1 * f1 + p2 * f2
    den2 = p1 * (1 - f1) + p2 * (1 - f2)
    pa1 = p1 * f1 / den1 if den1 > 0 else 0.0
    pa2 = p2 * (1 - f2) / den2 if den2 > 0 else 0.0
    return pa1, pa2


def _finite_xp(xp):
    if not math.isfinite(xp):
        return 1e6 if xp > 0 else -1e6
    return xp


# --- спектральные характеристики ---------------------------------------------

def acf_rc(tau, sigma2, alpha):
    return sigma2 * np.exp(-alpha * np.abs(np.asarray(tau, dtype=float)))


def psd_rc(w, sigma2, alpha):
    """Двусторонняя СПМ RC-процесса, В²·с/рад."""
    w = np.asarray(w, dtype=float)
    return 2 * alpha * sigma2 / (alpha ** 2 + w ** 2)


def psd_from_acf(sigma2, alpha, w, tmax_factor: float = 20.0, n: int = 40001):
    """Численное Φ(ω) = ∫R(τ)e^{−jωτ}dτ по symmetric сетке — проверка Винера–Хинчина."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    tmax = tmax_factor / alpha
    tau = np.linspace(-tmax, tmax, n)
    r = acf_rc(tau, sigma2, alpha)
    return np.trapezoid(r[None, :] * np.cos(w[:, None] * tau[None, :]), tau, axis=1)


def power_from_psd(w, s):
    """(1/2π)∫Φ(ω)dω = R(0)."""
    return float(np.trapezoid(s, w) / (2 * math.pi))


def ar1(n: int, sigma: float, alpha: float, fs: float, rng=None):
    """Стационарный процесс с R(τ)=σ²e^{−α|τ|}: y[n]=ρy[n−1]+σ√(1−ρ²)w[n]."""
    rng = rng or np.random.default_rng(0)
    rho = math.exp(-alpha / fs)
    w = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = sigma * w[0]
    k = sigma * math.sqrt(max(1 - rho ** 2, 0.0))
    for i in range(1, n):
        y[i] = rho * y[i - 1] + k * w[i]
    return y


def acf_biased(x):
    """R[m] = (1/N)Σx[n]x[n+m]."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return np.correlate(x, x, "full")[n - 1:] / n


def periodogram(x, fs):
    """Односторонняя СПМ, В²/Гц: S[k]=|X[k]|²/(N·Fд), внутренние бины ×2."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    spec = np.fft.rfft(x) / n
    df = fs / n
    s = np.abs(spec) ** 2 / df
    s[1:-1] *= 2
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, s


def matched_snr(a, t_dur, n0):
    """SNR на выходе согласованного фильтра в момент t0: 2E/N0, E = a²T."""
    return 2 * a * a * t_dur / n0


def matched_output(t, x, s, dt, t0=None):
    """y(t) = ∫x(τ)h(t−τ)dτ, h(τ)=s(t0−τ); согласованный фильтр — свёртка с s(t0−t)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    t0 = float(t[-1]) if t0 is None else float(t0)
    h = np.interp(t0 - t, t, s, left=0.0, right=0.0)
    return np.convolve(x, h)[: len(t)] * dt


# --- вспомогательные виджеты ---------------------------------------------------

def _spin(lo, hi, val, step, suffix="", decimals=3) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(step)
    sp.setValue(val)
    sp.setSuffix(suffix)
    sp.setKeyboardTracking(False)
    sp.setMinimumWidth(96)
    return sp


def _canvas(height: float) -> Canvas:
    cv = Canvas(height)
    cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    cv.setMinimumHeight(int(height * 72))
    return cv


def _form(rows) -> QWidget:
    w = QWidget()
    f = QFormLayout(w)
    f.setContentsMargins(4, 2, 4, 2)
    for label, widget in rows:
        f.addRow(label, widget)
    return w


class TaskPanel:
    """Задачи 1–3 и «Исследование»: вкладки «Ввод данных» и «Отображение функций»."""

    def __init__(self, page: "Lab4Page", mode: int):
        self.page = page
        self.mode = mode
        self.ok = False
        self.xp_calc = 0.0
        self.p_err = 0.0
        self.risk = 0.0
        self.post = (0.0, 0.0)

        self.a = _spin(0.01, 100.0, 2.0, 0.1, " В")
        self.m = _spin(-100.0, 100.0, 0.0, 0.1, " В")
        self.sigma = _spin(0.01, 100.0, 1.0, 0.1, " В")
        self.p1 = _spin(0.0, 1.0, 0.5, 0.05, "")
        self.p2 = _spin(0.0, 1.0, 0.5, 0.05, "")
        self.r12 = _spin(0.0, 100.0, 1.0, 0.1, "")
        self.r21 = _spin(0.0, 100.0, 1.0, 0.1, "")
        self.xp = _spin(-999.0, 999.0, 0.0, 0.1, " В")
        self.sample = _spin(-999.0, 999.0, 0.0, 0.1, " В")
        self.lam = _spin(0.001, 1000.0, 1.0, 0.1, "") if mode == 1 else None
        self.lam_lbl = QLabel("—")
        self.verdict = QLabel("")

        self.pages: list[tuple[str, QWidget]] = [
            ("Ввод данных", self._data_tab()),
            ("Отображение функции" if mode == 4 else "Отображение функций", self._plot_tab()),
        ]
        self._connect()
        self.compute()

    # --- сборка ---------------------------------------------------------------

    def _data_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        memo = QLabel(MEMOS[self.mode])
        memo.setWordWrap(True)
        memo.setFrameShape(QFrame.StyledPanel)
        memo.setStyleSheet("background:#eef3f8; padding:6px;")
        lay.addWidget(memo)

        box = QGroupBox("Введите значения параметров")
        g = QGridLayout(box)
        lam_title = ("λ — порог отношения правдоподобия" if self.mode == 1
                     else "λ — вычисляемый порог отношения правдоподобия")
        rows = [("a — амплитуда сигнала", self.a, "m — мат. ожидание помехи", self.m),
                ("σ — СКО помехи", self.sigma, lam_title, self._lam_row())]
        if self.mode >= 2:
            rows.append(("p(a1) — вероятность отсутствия", self.p1,
                         "p(a2) — вероятность присутствия", self.p2))
        if self.mode >= 3:
            rows.append(("r12 — риск пропуска", self.r12, "r21 — риск ложной тревоги", self.r21))
        if self.mode == 4:
            rows.append(("X — реализация на входе", self.sample, "", None))
        for r, (l1, w1, l2, w2) in enumerate(rows):
            g.addWidget(QLabel(l1), r, 0)
            g.addWidget(w1, r, 1)
            if w2 is not None:
                g.addWidget(QLabel(l2), r, 2)
                g.addWidget(w2, r, 3)
        g.setColumnStretch(4, 1)
        lay.addWidget(box)

        thr = QGroupBox("Введите значение порога")
        h = QHBoxLayout(thr)
        h.addWidget(QLabel("Xп, В:"))
        h.addWidget(self.xp)
        h.addWidget(self.verdict)
        h.addStretch(1)
        lay.addWidget(thr)

        btns = QHBoxLayout()
        if self.mode == 4:
            self.btn_compute = QPushButton("Вычислить")
            self.btn_compute.clicked.connect(self.compute)
        else:
            self.btn_compute = QPushButton("Проверить")
            self.btn_compute.clicked.connect(self.check)
        self.btn_clear = QPushButton("Обнулить")
        self.btn_clear.clicked.connect(self.clear)
        btns.addWidget(self.btn_compute)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.res_lb: dict[str, QLabel] = {}
        if self.mode == 4:
            res = QGroupBox("Результаты")
            rf = QFormLayout(res)
            for key, title in (("xp", "Порог Xп, В"), ("lam", "λ"), ("pa1", "P(a1/X1)"),
                               ("pa2", "P(a2/X2)"), ("pe", "Средняя ошибка Pош"),
                               ("risk", "Средний риск r"), ("decision", "Решение по X")):
                lab = QLabel("—")
                lab.setStyleSheet("font-family:monospace;")
                self.res_lb[key] = lab
                rf.addRow(title, lab)
            lay.addWidget(res)
        lay.addStretch(1)
        return w

    def _lam_row(self):
        if self.lam is not None:
            return self.lam
        lbl = QLabel("λ = p(a1)/p(a2)" if self.mode == 2 else "λ = p(a1)·r12/(p(a2)·r21)")
        lbl.setStyleSheet("color:#555;")
        return lbl

    def _plot_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        cap = QLabel("Функции правдоподобия L(a1), L(a2), их отношение L = L(a2)/L(a1) "
                     "и пороговый уровень Xп.")
        cap.setWordWrap(True)
        lay.addWidget(cap)
        row = QHBoxLayout()
        self.cv = _canvas(3.8)
        self.ax2 = self.cv.ax.twinx()
        row.addWidget(self.cv, 1)

        panel = QGroupBox("Параметры синтеза")
        pf = QFormLayout(panel)
        self.lb = {}
        items = [("A", "A = a, В"), ("M", "M = m, В"), ("G", "G = σ, В")]
        if self.mode >= 2:
            items += [("p1", "p(a1)"), ("p2", "p(a2)")]
        if self.mode >= 3:
            items += [("r12", "r12"), ("r21", "r21")]
        items += [("lam", "λ"), ("xp", "Xп, В")]
        if self.mode == 4:
            items += [("pa1", "P(a1/X1)"), ("pa2", "P(a2/X2)")]
        items += [("pe", "Pош"), ("risk", "r")]
        for key, title in items:
            lab = QLabel("—")
            lab.setStyleSheet("font-family:monospace;")
            self.lb[key] = lab
            pf.addRow(title, lab)
        row.addWidget(panel)
        lay.addLayout(row, 1)

        self.btn_upd = QPushButton("Обновить")
        self.btn_upd.clicked.connect(self.compute)
        lay.addWidget(self.btn_upd)
        return w

    def _connect(self):
        for sp in (self.a, self.m, self.sigma, self.p1, self.p2, self.r12, self.r21, self.xp,
                   self.sample):
            sp.editingFinished.connect(self.compute)
        if self.lam is not None:
            self.lam.editingFinished.connect(self.compute)

    # --- расчёт ---------------------------------------------------------------

    def lam_value(self) -> float:
        if self.mode == 1:
            assert self.lam is not None
            return float(self.lam.value())
        if self.mode == 2:
            return lam_ideal(self.p1.value(), self.p2.value())
        return lam_risk(self.p1.value(), self.p2.value(), self.r12.value(), self.r21.value())

    def compute(self):
        a, m, s = self.a.value(), self.m.value(), self.sigma.value()
        p1, p2 = self.p1.value(), self.p2.value()
        lam = self.lam_value()
        self.xp_calc = threshold(a, m, s, lam)
        self.p_err = p_error(a, m, s, p1, p2, self.xp_calc)
        self.risk = mean_risk(a, m, s, p1, p2, self.r12.value(), self.r21.value(), self.xp_calc)
        self.post = posterior(a, m, s, p1, p2, self.xp_calc)
        self._update_labels()
        self.draw()
        self.page.set_status(self.summary())

    def summary(self) -> str:
        lam = self.lam_value()
        lam_s = "∞" if not math.isfinite(lam) else f"{lam:.3f}"
        xp = self.xp_calc
        xp_s = "∞" if not math.isfinite(xp) else f"{xp:.3f} В"
        text = (f"Задача {self.mode} ({CRITERIA[self.mode]}): a = {self.a.value():.3f} В, "
                f"m = {self.m.value():.3f} В, σ = {self.sigma.value():.3f} В, λ = {lam_s}, "
                f"Xп = {xp_s}, Pош = {self.p_err:.4f}")
        if self.mode >= 2:
            text += f", p(a1) = {self.p1.value():.3f}, p(a2) = {self.p2.value():.3f}"
        if self.mode >= 3:
            text += f", r12 = {self.r12.value():.2f}, r21 = {self.r21.value():.2f}, r = {self.risk:.4f}"
        if self.mode == 4:
            accepted = "a2" if self.sample.value() > self.xp_calc else "a1"
            text += (f", X = {self.sample.value():.3f} В → принята {accepted}, "
                     f"P(a1/X1) = {self.post[0]:.3f}, P(a2/X2) = {self.post[1]:.3f}")
        return text

    def _update_labels(self):
        lam = self.lam_value()
        vals = {
            "A": f"{self.a.value():.3f}",
            "M": f"{self.m.value():.3f}",
            "G": f"{self.sigma.value():.3f}",
            "p1": f"{self.p1.value():.3f}",
            "p2": f"{self.p2.value():.3f}",
            "r12": f"{self.r12.value():.2f}",
            "r21": f"{self.r21.value():.2f}",
            "lam": "∞" if not math.isfinite(lam) else f"{lam:.3f}",
            "xp": "∞" if not math.isfinite(self.xp_calc) else f"{self.xp_calc:.3f}",
            "pa1": f"{self.post[0]:.3f}",
            "pa2": f"{self.post[1]:.3f}",
            "pe": f"{self.p_err:.4f}",
            "risk": f"{self.risk:.4f}",
        }
        for key, lab in self.lb.items():
            lab.setText(vals.get(key, "—"))
        if self.res_lb:
            accepted = "a2 (X > Xп)" if self.sample.value() > self.xp_calc else "a1 (X ≤ Xп)"
            vals["decision"] = accepted
            for key, lab in self.res_lb.items():
                lab.setText(vals.get(key, "—"))

    def check(self):
        """Сверка введённого Xп с расчётным; форма маски — 3 знака."""
        if not math.isfinite(self.xp_calc):
            self.ok = False
            self.verdict.setText("Порог не определён: λ или a вне области")
            self.verdict.setStyleSheet("color:#b3261e; font-weight:bold;")
            self.page.set_status(self.verdict.text())
            return
        diff = abs(self.xp.value() - self.xp_calc)
        self.ok = diff <= 1e-3
        if self.ok:
            self.verdict.setText(f"Верно: Xп = {self.xp_calc:.3f} В")
            self.verdict.setStyleSheet("color:#188038; font-weight:bold;")
        else:
            self.verdict.setText(f"Неверно: введено {self.xp.value():.3f}, ожидается "
                                 f"{self.xp_calc:.3f} В")
            self.verdict.setStyleSheet("color:#b3261e; font-weight:bold;")
        self.page.set_status(self.summary() + " · " + self.verdict.text())

    def clear(self):
        for sp, val in ((self.a, 2.0), (self.m, 0.0), (self.sigma, 1.0), (self.p1, 0.5),
                        (self.p2, 0.5), (self.r12, 1.0), (self.r21, 1.0), (self.xp, 0.0),
                        (self.sample, 0.0)):
            sp.setValue(val)
        if self.lam is not None:
            self.lam.setValue(1.0)
        self.ok = False
        self.verdict.setText("")
        self.compute()
        self.page.set_status(f"Задача {self.mode}: поля обнулены")

    # --- графики --------------------------------------------------------------

    def draw(self):
        cv = self.cv
        a, m, s = self.a.value(), self.m.value(), self.sigma.value()
        xp = self.xp_calc
        lo, hi = m - 4.5 * s, m + a + 4.5 * s
        x = np.linspace(lo, hi, 1201)
        f1, f2 = pdf(x, m, s), pdf(x, m + a, s)
        cv.clear(f"Функции правдоподобия и порог ({CRITERIA[self.mode]})", "x, В", "f(x|a), 1/В")
        if math.isfinite(xp) and lo < xp < hi:
            cv.ax.fill_between(x, 0, np.where(x > xp, f1, 0), color="#b3261e", alpha=0.12)
            cv.ax.fill_between(x, 0, np.where(x < xp, f2, 0), color="#1a73e8", alpha=0.12)
            cv.ax.axvline(xp, color="#444", lw=1.2, ls="--")
            cv.ax.annotate(f"Xп = {xp:.3f} В", (xp, 0), xytext=(5, 8),
                           textcoords="offset points", fontsize=8)
        cv.ax.plot(x, f1, color="#1a73e8", lw=1.6, label="L(a1) = f(x|a1)")
        cv.ax.plot(x, f2, color="#b3261e", lw=1.6, label="L(a2) = f(x|a2)")
        if self.mode == 4:
            cv.ax.plot([self.sample.value()], [0], marker="v", ms=7, color="#f9ab00",
                       label="реализация X")

        ax2 = self.ax2
        ax2.clear()
        if self.mode == 4:
            pa1 = [posterior(a, m, s, self.p1.value(), self.p2.value(), xi)[0] for xi in x]
            ax2.plot(x, pa1, color="#188038", lw=1.3, ls="-.")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("P(a1|x)", fontsize=8, color="#188038")
            ax2.tick_params(labelsize=7, colors="#188038")
        else:
            ax2.semilogy(x, likelihood_ratio(x, a, m, s), color="#188038", lw=1.1, ls="-.",
                         label="L = L(a2)/L(a1)")
            ax2.set_ylabel("L", fontsize=8, color="#188038")
            ax2.tick_params(labelsize=7, colors="#188038")
            ax2.set_ylim(1e-6, 1e6)
        h1, l1 = cv.ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        cv.ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        cv.draw_idle()


class Lab4Page(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding:4px; background:#f4f4f4;")

        self.t1 = TaskPanel(self, 1)
        self.t2 = TaskPanel(self, 2)
        self.t3 = TaskPanel(self, 3)
        self.t4 = TaskPanel(self, 4)

        self.pc1 = QTabWidget()
        self.pc1.addTab(self._theory_tab(), "Общие сведения")
        self.pc1.addTab(self._spectral_tab(), "Спектральные характеристики")
        for title, w in self.t1.pages:
            self.pc1.addTab(w, title)

        self.pc2 = self._inner(self.t2)
        self.pc3 = self._inner(self.t3)
        self.pc4 = self._inner(self.t4)

        self.tabs.addTab(self.pc1, "Задача1")
        self.tabs.addTab(self.pc2, "Задача2")
        self.tabs.addTab(self.pc3, "Задача3")
        self.tabs.addTab(self.pc4, "Исследование")

        lay.addWidget(self.status)
        self.draw_spectral()
        self.set_status("Готово: задайте параметры и нажмите «Проверить»")

    @staticmethod
    def _inner(panel: TaskPanel) -> QTabWidget:
        pc = QTabWidget()
        for title, w in panel.pages:
            pc.addTab(w, title)
        return pc

    def _theory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        apply_theory(view, "project4_1", THEORY)
        lay.addWidget(view)
        return w

    # --- спектральные характеристики -----------------------------------------

    def _spectral_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        cap = QLabel("Реализации, корреляционные функции и спектральные плотности мощности "
                     "трёх процессов; СПМ и АКФ связаны преобразованиями Хинчина–Винера.")
        cap.setWordWrap(True)
        lay.addWidget(cap)

        top = QHBoxLayout()
        proc = QGroupBox("Процесс")
        g = QGridLayout(proc)
        self.spec_kind = QComboBox()
        self.spec_kind.addItems(KINDS)
        self.spec_sigma = _spin(0.01, 50.0, 1.0, 0.1, " В")
        self.spec_alpha = _spin(1.0, 1e6, 200.0, 10.0, " 1/с", 1)
        self.spec_amp = _spin(0.1, 20.0, 1.0, 0.1, " В")
        self.spec_n = QComboBox()
        self.spec_n.addItems([str(n) for n in NS])
        self.spec_n.setCurrentText("1024")
        self.spec_fd = _spin(100.0, 1e6, 10000.0, 1000.0, " Гц", 0)
        g.addWidget(QLabel("Тип"), 0, 0)
        g.addWidget(self.spec_kind, 0, 1, 1, 3)
        g.addWidget(QLabel("σ, В"), 1, 0)
        g.addWidget(self.spec_sigma, 1, 1)
        g.addWidget(QLabel("A, В"), 1, 2)
        g.addWidget(self.spec_amp, 1, 3)
        g.addWidget(QLabel("α, 1/с"), 2, 0)
        g.addWidget(self.spec_alpha, 2, 1)
        g.addWidget(QLabel("Fд, Гц"), 2, 2)
        g.addWidget(self.spec_fd, 2, 3)
        g.addWidget(QLabel("N"), 3, 0)
        g.addWidget(self.spec_n, 3, 1)
        top.addWidget(proc)

        mf = QGroupBox("Согласованный фильтр")
        f = QGridLayout(mf)
        self.mf_a = _spin(0.1, 10.0, 1.0, 0.1, " В")
        self.mf_t = _spin(0.01, 100.0, 1.0, 0.1, " мс")
        self.mf_n0 = _spin(1e-6, 100.0, 0.01, 0.01, " В²/Гц", 6)
        self.mf_noise = QCheckBox("Шум на входе")
        f.addWidget(QLabel("A, В"), 0, 0)
        f.addWidget(self.mf_a, 0, 1)
        f.addWidget(QLabel("T, мс"), 1, 0)
        f.addWidget(self.mf_t, 1, 1)
        f.addWidget(QLabel("N0, В²/Гц"), 2, 0)
        f.addWidget(self.mf_n0, 2, 1)
        f.addWidget(self.mf_noise, 3, 0, 1, 2)
        top.addWidget(mf)

        btns = QVBoxLayout()
        self.btn_spec = QPushButton("Обновить")
        self.btn_spec.clicked.connect(self.draw_spectral)
        self.btn_spec_clear = QPushButton("Очистить")
        self.btn_spec_clear.clicked.connect(self.clear_spectral)
        btns.addWidget(self.btn_spec)
        btns.addWidget(self.btn_spec_clear)
        btns.addStretch(1)
        top.addLayout(btns)
        top.addStretch(1)
        lay.addLayout(top)

        grid = QGridLayout()
        self.cv_real = _canvas(2.4)
        self.cv_acf = _canvas(2.4)
        self.cv_psd = _canvas(2.4)
        self.cv_mf = _canvas(2.4)
        self.ax_mf2 = self.cv_mf.ax.twinx()
        self.spec_caption = QLabel("")
        grid.addWidget(self.cv_real, 0, 0)
        grid.addWidget(self.cv_acf, 0, 1)
        grid.addWidget(self.cv_psd, 1, 0)
        grid.addWidget(self.cv_mf, 1, 1)
        lay.addLayout(grid, 1)
        lay.addWidget(self.spec_caption)

        self.spec_kind.currentTextChanged.connect(lambda _: self.draw_spectral())
        self.spec_n.currentTextChanged.connect(lambda _: self.draw_spectral())
        for sp in (self.spec_sigma, self.spec_alpha, self.spec_amp, self.spec_fd,
                   self.mf_a, self.mf_t, self.mf_n0):
            sp.editingFinished.connect(self.draw_spectral)
        self.mf_noise.stateChanged.connect(lambda _: self.draw_mf())
        return w

    def spectral_data(self):
        """(kind, x, fs) — реализация процесса по текущим полям."""
        kind = self.spec_kind.currentText()
        n = int(self.spec_n.currentText())
        fs = float(self.spec_fd.value())
        sigma = float(self.spec_sigma.value())
        alpha = float(self.spec_alpha.value())
        amp = float(self.spec_amp.value())
        rng = np.random.default_rng(0)
        if kind == KINDS[0]:
            x = np.full(n, amp)
        elif kind == KINDS[1]:
            x = ar1(n, sigma, alpha, fs, rng)
        else:
            x = sigma * rng.standard_normal(n)
        return kind, x, fs

    def _spec_theory(self, kind: str):
        """(R(τ), Φ(ω), подпись единиц) для текущего процесса."""
        sigma2 = float(self.spec_sigma.value()) ** 2
        alpha = float(self.spec_alpha.value())
        amp = float(self.spec_amp.value())
        if kind == KINDS[0]:
            return (lambda tau: np.full_like(np.asarray(tau, float), amp ** 2),
                    lambda w: np.zeros_like(np.asarray(w, float)), "В²")
        if kind == KINDS[1]:
            return (lambda tau: acf_rc(tau, sigma2, alpha),
                    lambda w: psd_rc(w, sigma2, alpha), "В²")
        return (lambda tau: sigma2 * (np.asarray(tau) == 0),
                lambda w: np.full_like(np.asarray(w, float), sigma2), "В²/рад·с")

    def draw_spectral(self):
        kind, x, fs = self.spectral_data()
        n = len(x)
        fs = float(fs)
        dt = 1.0 / fs
        tau = np.arange(len(acf_biased(x))) / fs
        r_emp = acf_biased(x)
        f, s = periodogram(x, fs)

        cv = self.cv_real
        cv.clear(f"Реализация: {kind}", "t, мс", "x, В")
        cv.ax.plot(np.arange(n) / fs * 1e3, x, color="#1a73e8", lw=1.0)
        cv.ax.set_xlim(0, min(n * dt * 1e3, 200.0))

        cv = self.cv_acf
        cv.clear(f"Корреляционная функция: {kind}", "τ, мс", "R, В²")
        cv.ax.plot(tau * 1e3, r_emp, color="#188038", lw=1.5, label="выборочная")
        r_th, _, _ = self._spec_theory(kind)
        tt = np.linspace(0, tau[-1], 800)
        cv.ax.plot(tt * 1e3, r_th(tt), "--", color="#b3261e", lw=1.2, label="теория")
        cv.ax.set_xlim(0, min(tau[-1] * 1e3, 20 / float(self.spec_alpha.value()) * 1e3))
        cv.ax.legend(fontsize=7)
        r0 = float(r_emp[0])
        cv.ax.annotate(f"R(0) = {r0:.3f} В²", (0, r0), xytext=(10, -12), va="top",
                       textcoords="offset points", fontsize=8)

        cv = self.cv_psd
        if kind == KINDS[0]:
            # у постоянного уровня спектр — дельта на нуле: рисуем линейчатый спектр
            amp2 = float(self.spec_amp.value()) ** 2
            cv.clear(f"СПМ: {kind} (линейчатый спектр)", "f, Гц", "P, В²")
            cv.ax.vlines(0.0, 0.0, amp2, color="#8e24aa", lw=2.5)
            cv.ax.annotate(f"2πA²·δ(ω): вся мощность {amp2:.3f} В² на ω = 0",
                           (0.0, amp2), xytext=(12, -4), textcoords="offset points", fontsize=8,
                           color="#b3261e")
            cv.ax.set_xlim(0, fs / 2)
            cv.ax.set_ylim(0, amp2 * 1.3)
            cv.draw_idle()
        else:
            cv.clear(f"СПМ: {kind}", "f, Гц", "S, В²/Гц")
            for ln in list(cv.ax.lines):
                if np.any(np.asarray(ln.get_ydata()) <= 0):
                    ln.remove()
            pos = s > 0
            cv.ax.set_autoscaley_on(False)
            cv.ax.semilogy(f[pos], s[pos], color="#8e24aa", lw=1.0, label="периодограмма")
            if kind == KINDS[1]:
                alpha = float(self.spec_alpha.value())
                sigma2 = float(self.spec_sigma.value()) ** 2
                ff = np.linspace(1.0, fs / 2, 800)
                cv.ax.semilogy(ff, 4 * alpha * sigma2 / (alpha ** 2 + (2 * np.pi * ff) ** 2),
                               "--", color="#b3261e", lw=1.2,
                               label="4ασ²/(α²+(2πf)²)")
            else:
                cv.ax.axhline(2 * float(self.spec_sigma.value()) ** 2 / fs, color="#b3261e",
                              ls="--", lw=1.0, label="2σ²/Fд")
            cv.ax.set_xlim(0, fs / 2)
            cv.ax.legend(fontsize=7)
            if pos.sum() > 1:
                cv.ax.set_ylim(float(s[pos].min()) * 0.5, float(s[pos].max()) * 2)
            cv.draw_idle()

        self.draw_mf()
        self.spec_caption.setText(
            f"σ = {self.spec_sigma.value():.3f} В, α = {self.spec_alpha.value():.1f} 1/с, "
            f"N = {n}, Fд = {fs:.0f} Гц; выборочные R(0) = {r0:.3f} В², "
            f"средняя мощность (1/N)Σx² = {float((x ** 2).mean()):.3f} В²")
        self.set_status(f"Спектральные характеристики: {kind}, R(0) = {r0:.4f} В², "
                        f"P = {(x ** 2).mean():.4f} В²")

    def draw_mf(self):
        a = float(self.mf_a.value())
        t_dur = float(self.mf_t.value()) * 1e-3
        n0 = float(self.mf_n0.value())
        fs = max(2000.0, 200.0 / t_dur)
        dt = 1.0 / fs
        t = np.arange(0, 2.2 * t_dur, dt)
        s = np.where((t >= 0) & (t < t_dur), a, 0.0)
        rng = np.random.default_rng(1)
        sigma_n = math.sqrt(n0 * fs / 2)
        x = s + (sigma_n * rng.standard_normal(len(t)) if self.mf_noise.isChecked() else 0.0)
        y = matched_output(t, x, s, dt, t_dur)
        cv = self.cv_mf
        cv.clear("Согласованный фильтр: s(t) и y(t)", "t, мс", "y(t), В²·с")
        cv.ax.plot(t * 1e3, y, color="#b3261e", lw=1.6, label="y(t)")
        if self.mf_noise.isChecked():
            cv.ax.plot(t * 1e3, x, color="#c9d6e8", lw=0.6, label="x(t)")
        k = int(np.argmax(np.abs(y)))
        cv.ax.axvline(t_dur * 1e3, color="#188038", ls="--", lw=1.0)
        cv.ax.annotate(f"t0 = {t_dur * 1e3:.3f} мс, y = {y[k]:.4f}", (t[k] * 1e3, y[k]),
                       xytext=(6, 4), textcoords="offset points", fontsize=8)
        cv.ax.legend(fontsize=7, loc="upper left")

        ax2 = self.ax_mf2
        ax2.clear()
        ax2.plot(t * 1e3, s, color="#1a73e8", lw=1.1, ls="--", label="s(t), В")
        ax2.set_ylabel("s(t), В", fontsize=8, color="#1a73e8")
        ax2.tick_params(labelsize=7, colors="#1a73e8")
        ax2.set_ylim(0, a * 1.3)
        h1, l1 = cv.ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        cv.ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        cv.draw_idle()

    def clear_spectral(self):
        for cv, title in ((self.cv_real, "Реализация"), (self.cv_acf, "Корреляционная функция"),
                          (self.cv_psd, "СПМ"), (self.cv_mf, "Согласованный фильтр")):
            cv.clear(f"{title}: нет данных", "", "")
            cv.draw_idle()
        self.ax_mf2.clear()
        self.ax_mf2.set_ylabel("")
        self.spec_caption.setText("")
        self.set_status("Спектральные характеристики очищены")

    # --- общее ----------------------------------------------------------------

    def set_status(self, text: str):
        self.status.setText(text)

    def canvases(self) -> list[Canvas]:
        return [self.cv_real, self.cv_acf, self.cv_psd, self.cv_mf,
                self.t1.cv, self.t2.cv, self.t3.cv, self.t4.cv]

    def redraw_all(self):
        """Синхронная перерисовка: draw_idle может не успеть к grab()."""
        for cv in self.canvases():
            cv.draw()

    def clear(self):
        for panel in (self.t1, self.t2, self.t3, self.t4):
            panel.clear()
        self.clear_spectral()
        self.set_status("Данные очищены")
