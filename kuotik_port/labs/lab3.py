"""ЛР3: дискретизация непрерывных сигналов по времени — форма fSpektr."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
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

KINDS = ("Гарм. сигнал", "Прям. импульсы", "Шум", "Смешанный")
LENGTHS = (64, 128, 256, 512, 1024)

THEORY = """<h3>Дискретизация непрерывных сигналов по времени</h3>
<p>Теорема Котельникова: функцию со спектром от 0 до F<sub>в</sub> можно передать с любой точностью
числами, следующими через Δt ≤ 1/(2F<sub>в</sub>). Ряд Котельникова–Шеннона:
x(t) = Σ x[n]·sinc((t − nΔt)/Δt), x[n] = x(nΔt), F<sub>д</sub> = 1/Δt.</p>
<p>ДПФ: X[k] = Σ<sub>n</sub> x[n]·e<sup>−j2πkn/N</sup>, шаг Δf = F<sub>д</sub>/N. Спектр дискретного
сигнала периодичен с периодом F<sub>д</sub>; граница Найквиста F<sub>н</sub> = F<sub>д</sub>/2.</p>
<p>АКФ: R[m] = (1/N)·Σ x[n]·x[n+m]. СПМ (периодограмма): S[k] = |X[k]|²/(N·F<sub>д</sub>), В²/Гц;
по теореме Хинчина–Винера СПМ и АКФ — пара преобразования Фурье.</p>
<p>Идеальный ФНЧ с срезом W<sub>c</sub>: h(t) = (W<sub>c</sub>/π)·sinc(W<sub>c</sub>t/π),
восстановление x<sub>r</sub>(t) = Δt·Σ x[n]·h(t − nΔt). При W<sub>c</sub> = π/Δt ряд проходит через
все отсчёты.</p>
<h4>Порядок выполнения</h4>
<ol><li>Ввести параметры гармонического сигнала.</li>
<li>Снять показания на входе и выходе дискретизатора.</li>
<li>Оценить восстановление при двух режимах ФНЧ.</li>
<li>Сравнить спектр и АКФ с теорией.</li>
<li>Исследовать влияние N и Δt.</li>
<li>Получить импульс ВЧ колебаний (частота много больше 1/T).</li>
<li>Повторить для последовательности импульсов, одиночного импульса и пачки.</li>
<li>Ввести параметры гауссова шума; проанализировать СПМ.</li>
<li>Смешанный сигнал: повторить п. 8.</li></ol>
<p>Единицы: амплитуда — В, частота — Гц, интервал и период — мс, W<sub>c</sub> — рад/с,
АКФ — В², СПМ — В²/Гц.</p>"""

SCHEMA = """<h3>Схема установки</h3>
<pre style="font-size:13px; line-height:1.5">
  ┌──────────────┐   x(t)   ┌──────────────┐   x[n]   ┌──────────────┐   x_восст(t)
  │  Генератор   │ ───────► │     АЦП      │ ───────► │  Идеальный   │ ─────────────►
  │ гарм./имп./  │          │ дискретизация│          │     ФНЧ      │
  │ шум/смеш.    │          │  и квантов.  │          │  срез Wc     │
  └──────────────┘          └──────────────┘          └──────────────┘
         │                         │                         │
         └───────────────┬─────────┴─────────────┬───────────┘
                         ▼                       ▼
                  ┌──────────────┐        ┌──────────────┐
                  │ Осциллограф  │        │ ЭВМ: спектр, │
                  │ вход/выход   │        │ АКФ, СПМ     │
                  └──────────────┘        └──────────────┘
</pre>
<p>Генератор задаёт N отсчётов сигнала. АЦП берёт отсчёты через Δt и квантует по уровню Δ.
Идеальный ФНЧ восстанавливает сигнал; на ЭВМ считаются ДПФ, АКФ и СПМ.</p>"""


@dataclass
class Params:
    kind: str = KINDS[0]
    n: int = 256
    amp: float = 1.0
    freq: float = 50.0
    phase_ms: float = 0.0
    imp_amp: float = 1.0
    imp_period_ms: float = 20.0
    imp_dur_ms: float = 5.0
    noise_mean: float = 0.0
    noise_var: float = 0.25
    dt_ms: float = 1.0
    quant: float = 0.0
    wc: float = 0.0          # 0 → π/Δt


def harmonic(t, p: Params):
    return p.amp * np.sin(2 * np.pi * p.freq * (t - p.phase_ms * 1e-3))


def pulses(t, p: Params):
    period = max(p.imp_period_ms, 1e-9) * 1e-3
    dur = p.imp_dur_ms * 1e-3
    return np.where(np.mod(t, period) < dur, p.imp_amp, 0.0)


def noise(n: int, p: Params, rng) -> np.ndarray:
    return p.noise_mean + math.sqrt(max(p.noise_var, 0.0)) * rng.standard_normal(n)


def continuous(t, p: Params):
    """Непрерывная (детерминированная) часть; None для чистого шума."""
    if p.kind == KINDS[0]:
        return harmonic(t, p)
    if p.kind == KINDS[1]:
        return pulses(t, p)
    if p.kind == KINDS[3]:
        return harmonic(t, p) + pulses(t, p)
    return None


def sampled(p: Params, rng) -> tuple[np.ndarray, np.ndarray]:
    dt = p.dt_ms * 1e-3
    t = np.arange(p.n) * dt
    x = continuous(t, p)
    if p.kind == KINDS[2]:
        x = noise(p.n, p, rng)
    elif p.kind == KINDS[3]:
        x = x + noise(p.n, p, rng)
    return t, x


def quantize(x: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return x.copy()
    return np.round(x / step) * step


def ideal_lpf_ir(t, wc: float):
    wc = max(wc, 1e-12)
    return (wc / np.pi) * np.sinc(wc * t / np.pi)


def reconstruct(t, x: np.ndarray, dt: float, wc: float) -> np.ndarray:
    """Идеальный ФНЧ: свёртка отсчётов с h(t)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    grid = t[:, None] - (np.arange(len(x)) * dt)[None, :]
    return dt * (np.asarray(x, dtype=float)[None, :] * ideal_lpf_ir(grid, wc)).sum(axis=1)


def fold_freq(f: float, fd: float) -> float:
    """Частота-образ при дискретизации: свёртка спектра в 0…Fд/2."""
    r = abs(f) % fd
    return fd - r if r > fd / 2 else r


def spectrum_one_sided(x: np.ndarray, fd: float):
    n = len(x)
    spec = np.fft.rfft(x) / n
    amp = np.abs(spec)
    amp[1:] *= 2
    f = np.fft.rfftfreq(n, 1 / fd)
    return f, amp, spec


def dtft(x: np.ndarray, dt: float, f):
    """ДПФ на произвольной частоте — для проверки периодичности."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = np.arange(len(x))
    return (x[None, :] * np.exp(-2j * np.pi * n[None, :] * dt * f[:, None])).sum(axis=1) / len(x)


def acf(x: np.ndarray, dt: float):
    n = len(x)
    r = np.correlate(x, x, "full")[n - 1:] / n
    tau = np.arange(n) * dt
    return tau[: n // 2 + 1], r[: n // 2 + 1]


def psd_one_sided(x: np.ndarray, fd: float):
    n = len(x)
    spec = np.fft.rfft(x) / n
    df = fd / n
    s = np.abs(spec) ** 2 / df
    s[1:-1] *= 2
    f = np.fft.rfftfreq(n, 1 / fd)
    return f, s


def upper_freq(p: Params, fd: float) -> float:
    if p.kind == KINDS[0]:
        return p.freq
    if p.kind == KINDS[1]:
        return 1 / (max(p.imp_dur_ms, 1e-9) * 1e-3)
    if p.kind == KINDS[2]:
        return fd / 2
    return max(p.freq, 1 / (max(p.imp_dur_ms, 1e-9) * 1e-3))


class Lab3Page(QWidget):
    def __init__(self):
        super().__init__()
        self._rng = np.random.default_rng(0)
        self.p = Params()
        self.t = self.x = self.xq = None
        self.t_dense = self.x_cont = self.xr = None
        self.wc_eff = math.pi / (self.p.dt_ms * 1e-3)
        self.fd = 1 / (self.p.dt_ms * 1e-3)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.tabs.addTab(self._theory_tab(), "Общие сведения")
        self.tabs.addTab(self._schema_tab(), "Схема установки")
        self.tabs.addTab(self._scope_tab(), "Осциллограф")
        self.tabs.addTab(self._spectrum_tab(), "Спектр амплитуд")
        self.tabs.addTab(self._acf_tab(), "Корреляционная функция")
        self.tabs.addTab(self._psd_tab(), "Спектральная плотность мощности")

        lay.addWidget(self._controls())
        self.status = QLabel("—")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.model()

    # --- вкладки -------------------------------------------------------------

    def _theory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        view.setHtml(THEORY)
        lay.addWidget(view)
        return w

    def _schema_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTextBrowser()
        view.setHtml(SCHEMA)
        lay.addWidget(view)
        return w

    def _scope_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Время/дел:"))
        self.cb_tdiv = QComboBox()
        self.cb_tdiv.addItems(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50"])
        self.cb_tdiv.setCurrentText("5")
        row.addWidget(self.cb_tdiv)
        row.addWidget(QLabel("Вольт/дел:"))
        self.cb_vdiv = QComboBox()
        self.cb_vdiv.addItems(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
        self.cb_vdiv.setCurrentText("0.5")
        row.addWidget(self.cb_vdiv)
        self.cb_filter = QCheckBox("Режим фильтра")
        self.cb_filter.setChecked(True)
        row.addWidget(self.cb_filter)
        self.cb_components = QCheckBox("Показывать составляющие")
        row.addWidget(self.cb_components)
        row.addStretch(1)
        lay.addLayout(row)
        self.cv_scope = self._canvas(3.4)
        lay.addWidget(self.cv_scope)
        self.lbl_scope = QLabel("")
        lay.addWidget(self.lbl_scope)
        self.cb_tdiv.currentTextChanged.connect(lambda _: self._draw_scope())
        self.cb_vdiv.currentTextChanged.connect(lambda _: self._draw_scope())
        self.cb_filter.stateChanged.connect(lambda _: self.model())
        self.cb_components.stateChanged.connect(lambda _: self._draw_scope())
        return w

    def _spectrum_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_spec = QLabel("")
        lay.addWidget(self.lbl_spec)
        self.cv_spec = self._canvas(4.0)
        lay.addWidget(self.cv_spec)
        return w

    def _acf_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_acf = QLabel("")
        lay.addWidget(self.lbl_acf)
        self.cv_acf = self._canvas(4.0)
        lay.addWidget(self.cv_acf)
        return w

    def _psd_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_psd = QLabel("")
        lay.addWidget(self.lbl_psd)
        self.cv_psd = self._canvas(4.0)
        lay.addWidget(self.cv_psd)
        return w

    # --- панель управления ---------------------------------------------------

    def _controls(self) -> QWidget:
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 4, 6, 2)
        outer.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        gen = QGroupBox("Генератор")
        g = QGridLayout(gen)
        g.addWidget(QLabel("Длина сигнала"), 0, 0)
        self.cb_n = QComboBox()
        self.cb_n.addItems([str(n) for n in LENGTHS])
        self.cb_n.setCurrentText("256")
        g.addWidget(self.cb_n, 0, 1)
        g.addWidget(QLabel("Тип сигнала"), 1, 0)
        self.cb_kind = QComboBox()
        self.cb_kind.addItems(KINDS)
        g.addWidget(self.cb_kind, 1, 1)
        row.addWidget(gen)

        pars = QGroupBox("Параметры сигнала")
        pl = QVBoxLayout(pars)
        self.param_tabs = QTabWidget()
        self.param_tabs.setMaximumHeight(112)
        self.sp_amp = self._spin(0, 10, 1.0, 0.1, " В")
        self.sp_freq = self._spin(0, 100000, 50.0, 10, " Гц")
        self.sp_phase = self._spin(0, 1000, 0.0, 1, " мс")
        self.sp_im_amp = self._spin(0, 10, 1.0, 0.1, " В")
        self.sp_im_per = self._spin(0, 100000, 20.0, 1, " мс")
        self.sp_im_dur = self._spin(0, 10000, 5.0, 1, " мс")
        self.sp_n_mean = self._spin(-100, 100, 0.0, 0.1, " В")
        self.sp_n_var = self._spin(0, 100, 0.25, 0.05, " В²")
        self.param_tabs.addTab(self._form((("Амплитуда", self.sp_amp), ("Частота", self.sp_freq),
                                           ("Фаза", self.sp_phase))), "Гарм. сигнал")
        self.param_tabs.addTab(self._form((("Амплитуда", self.sp_im_amp), ("Период", self.sp_im_per),
                                           ("Длительность", self.sp_im_dur))), "Импульсы")
        self.param_tabs.addTab(self._form((("Дисперсия", self.sp_n_var),
                                           ("Мат. ожидание", self.sp_n_mean))), "Шум")
        pl.addWidget(self.param_tabs)
        self.param_tabs.setUsesScrollButtons(False)
        row.addWidget(pars, 1)

        adc = QGroupBox("Дискретизация и квантование")
        a = QFormLayout(adc)
        self.sp_dt = self._spin(0.01, 9999.99, 1.0, 0.1, " мс")
        self.sp_dt.setDecimals(2)
        a.addRow("Интервал дискретизации", self.sp_dt)
        self.lbl_fd = QLabel("1000.0 Гц")
        a.addRow("Частота дискретизации", self.lbl_fd)
        self.sp_quant = self._spin(0, 100, 0.0, 0.01, " В")
        self.sp_quant.setDecimals(3)
        a.addRow("Размер одного кванта", self.sp_quant)
        row2.addWidget(adc)

        flt = QGroupBox("Фильтр низких частот")
        fl = QVBoxLayout(flt)
        self.sp_wc = self._spin(0.01, 1e7, 3141.59, 50, " рад/с")
        self.sp_wc.setDecimals(2)
        fl.addWidget(QLabel("Ограничение спектра, Wc"))
        fl.addWidget(self.sp_wc)
        btn_pi = QPushButton("Wc = π/Δt")
        btn_pi.clicked.connect(self._set_wc_nyquist)
        fl.addWidget(btn_pi)
        row2.addWidget(flt)

        proc = QGroupBox("Параметры процесса")
        pr = QFormLayout(proc)
        self.lbl_wc_opt = QLabel("—")
        self.lbl_dt_opt = QLabel("—")
        self.lbl_fup = QLabel("—")
        pr.addRow("Wc = 2πFв", self.lbl_wc_opt)
        pr.addRow("Интервал", self.lbl_dt_opt)
        pr.addRow("Fв", self.lbl_fup)
        row2.addWidget(proc)
        row2.addStretch(1)

        btns = QVBoxLayout()
        self.btn_update = QPushButton("Обновить")
        self.btn_update.clicked.connect(self.model)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.clear)
        btns.addWidget(self.btn_update)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        row.addLayout(btns)
        outer.addLayout(row)
        outer.addLayout(row2)

        self.sp_dt.valueChanged.connect(self._sync_fd)
        self.cb_kind.currentTextChanged.connect(self._kind_changed)
        self.cb_n.currentTextChanged.connect(lambda _: self.model())
        for sp in (self.sp_amp, self.sp_freq, self.sp_phase, self.sp_im_amp, self.sp_im_per,
                   self.sp_im_dur, self.sp_n_mean, self.sp_n_var, self.sp_quant, self.sp_wc):
            sp.editingFinished.connect(self.model)
        return box

    @staticmethod
    def _canvas(height: float) -> Canvas:
        cv = Canvas(height)
        cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.setMinimumHeight(int(height * 90))
        return cv

    @staticmethod
    def _spin(lo: float, hi: float, val: float, step: float, suffix: str) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(2)
        sp.setSingleStep(step)
        sp.setValue(val)
        sp.setSuffix(suffix)
        sp.setKeyboardTracking(False)
        return sp

    @staticmethod
    def _form(rows) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 2, 4, 2)
        for label, widget in rows:
            f.addRow(label, widget)
        return w

    # --- данные --------------------------------------------------------------

    def _read_params(self) -> Params:
        return Params(kind=self.cb_kind.currentText(), n=int(self.cb_n.currentText()),
                      amp=self.sp_amp.value(), freq=self.sp_freq.value(),
                      phase_ms=self.sp_phase.value(), imp_amp=self.sp_im_amp.value(),
                      imp_period_ms=self.sp_im_per.value(), imp_dur_ms=self.sp_im_dur.value(),
                      noise_mean=self.sp_n_mean.value(), noise_var=self.sp_n_var.value(),
                      dt_ms=self.sp_dt.value(), quant=self.sp_quant.value(),
                      wc=self.sp_wc.value())

    def _kind_changed(self, kind: str):
        idx = {KINDS[0]: 0, KINDS[1]: 1, KINDS[2]: 2}.get(kind)
        if idx is not None:
            self.param_tabs.setCurrentIndex(idx)
        self.model()

    def _set_wc_nyquist(self):
        self.sp_wc.setValue(math.pi / (self.sp_dt.value() * 1e-3))
        self.model()

    def _sync_fd(self):
        fd = 1000.0 / self.sp_dt.value()
        self.lbl_fd.setText(f"{fd:.1f} Гц")

    @property
    def fd_value(self) -> float:
        return 1000.0 / self.sp_dt.value()

    def model(self):
        """Пересчитать сигнал и все графики по текущим полям."""
        self._rng = np.random.default_rng(0)
        self.p = self._read_params()
        dt = self.p.dt_ms * 1e-3
        self.fd = 1 / dt
        self.wc_eff = self.p.wc if self.p.wc > 0 else math.pi / dt
        self.t, self.x = sampled(self.p, self._rng)
        self.xq = quantize(self.x, self.p.quant)
        self.t_dense = np.linspace(0, (self.p.n - 1) * dt, min(4000, 8 * self.p.n))
        cont = continuous(self.t_dense, self.p)
        self.x_cont = np.interp(self.t_dense, self.t, self.x) if cont is None else cont
        use_filter = self.cb_filter.isChecked()
        self.xr = (reconstruct(self.t_dense, self.xq, dt, self.wc_eff) if use_filter else None)
        self._sync_fd()
        fup = upper_freq(self.p, self.fd)
        self.lbl_wc_opt.setText(f"{2 * math.pi * fup:.1f} рад/с")
        self.lbl_dt_opt.setText(f"{1000 / (2 * fup):.3f} мс")
        self.lbl_fup.setText(f"{fup:.1f} Гц")
        self._update_status(fup)
        self._draw_all()

    def _update_status(self, fup: float):
        nyq = self.fd / 2
        parts = [f"Fд = {self.fd:.1f} Гц", f"Fн = {nyq:.1f} Гц", f"Fв = {fup:.1f} Гц",
                 f"Wc = {self.wc_eff:.1f} рад/с", f"N = {self.p.n}",
                 f"T = {self.p.n * self.p.dt_ms:.1f} мс"]
        text = " · ".join(parts)
        if fup > nyq + 1e-9:
            text += (f"<br><span style='color:#b3261e'><b>Нарушение теоремы Котельникова:</b> "
                     f"Fв = {fup:.1f} Гц &gt; Fн = {nyq:.1f} Гц — спектр накладывается (алиасинг)</span>")
        self.status.setText(text)

    def restore(self, t) -> np.ndarray:
        """Восстановленный сигнал в точках t (для проверок)."""
        dt = self.p.dt_ms * 1e-3
        return reconstruct(t, self.xq, dt, self.wc_eff)

    def spectrum_data(self):
        return spectrum_one_sided(self.x, self.fd)

    def acf_data(self):
        return acf(self.x, self.p.dt_ms * 1e-3)

    def psd_data(self):
        return psd_one_sided(self.x, self.fd)

    def clear(self):
        self.t = self.x = self.xq = None
        self.x_cont = self.xr = None
        for cv, title in ((self.cv_scope, "Осциллограф"), (self.cv_spec, "Спектр амплитуд"),
                          (self.cv_acf, "Корреляционная функция"),
                          (self.cv_psd, "Спектральная плотность мощности")):
            cv.clear(f"{title}: нет данных", "", "")
            cv.draw_idle()
        self.lbl_scope.setText("")
        self.lbl_spec.setText("")
        self.lbl_acf.setText("")
        self.lbl_psd.setText("")
        self.status.setText("Данные очищены")

    # --- рисование -----------------------------------------------------------

    def _draw_all(self):
        self._draw_scope()
        self._draw_spectrum()
        self._draw_acf()
        self._draw_psd()

    def _draw_scope(self):
        if self.x is None:
            return
        cv = self.cv_scope
        cv.clear(f"Осциллограф: {self.p.kind}", "t, мс", "x, В")
        t_ms = self.t * 1e3
        cv.ax.plot(self.t_dense * 1e3, self.x_cont, color="#0b5", lw=1.6, label="непрерывный")
        cv.ax.vlines(t_ms, 0, self.xq, color="#1a73e8", lw=0.8, alpha=0.7)
        cv.ax.plot(t_ms, self.xq, "o", ms=3.2, color="#1a73e8", label="дискретный")
        if self.p.quant > 0:
            cv.ax.plot(t_ms, self.xq, "_", ms=7, color="#8e24aa", label="квантованный")
        if self.xr is not None:
            cv.ax.plot(self.t_dense * 1e3, self.xr, "--", color="#b3261e", lw=1.2,
                       label="восстановленный")
            if self.cb_components.isChecked():
                dt = self.p.dt_ms * 1e-3
                for n in range(min(12, self.p.n)):
                    term = dt * self.xq[n] * ideal_lpf_ir(self.t_dense - n * dt, self.wc_eff)
                    cv.ax.plot(self.t_dense * 1e3, term, color="#f9ab00", lw=0.6, alpha=0.5)
        cv.ax.legend(fontsize=8, loc="upper right")
        tdiv = float(self.cb_tdiv.currentText())
        vdiv = float(self.cb_vdiv.currentText())
        span = max(10 * tdiv, self.p.dt_ms)
        cv.ax.set_xlim(0, min(span, self.p.n * self.p.dt_ms))
        cv.ax.set_ylim(-4 * vdiv, 4 * vdiv)
        err = (np.abs(self.xr - self.x_cont).max() if self.xr is not None else float("nan"))
        self.lbl_scope.setText(f"Отсчётов: {self.p.n}; ошибка восстановления "
                               f"max|x_восст − x| = {err:.3f} В" if self.xr is not None else
                               f"Отсчётов: {self.p.n}; фильтр выключен")
        cv.draw_idle()

    def _draw_spectrum(self):
        if self.x is None:
            return
        f, amp, _ = self.spectrum_data()
        cv = self.cv_spec
        cv.clear(f"Спектр амплитуд: ДПФ, N = {self.p.n}, Fд = {self.fd:.0f} Гц",
                 "f, Гц", "A, В")
        for shift, color, alpha in ((0.0, "#1a73e8", 1.0), (self.fd, "#9aa0a6", 0.85),
                                    (-self.fd, "#9aa0a6", 0.55)):
            cv.ax.vlines(f + shift, 0, amp, color=color, lw=1.0, alpha=alpha)
        cv.ax.axvline(self.fd / 2, color="#b3261e", ls=":", lw=1.0)
        cv.ax.annotate("Fд/2", (self.fd / 2, 0), xytext=(3, 4), textcoords="offset points",
                       fontsize=8, color="#b3261e")
        cv.ax.axvline(self.fd, color="#666", ls="--", lw=0.8)
        cv.ax.annotate("Fд", (self.fd, 0), xytext=(3, 4), textcoords="offset points", fontsize=8)
        cv.ax.set_xlim(0, 2 * self.fd)
        k = int(np.argmax(amp))
        cv.ax.annotate(f"{f[k]:.1f} Гц, {amp[k]:.2f} В", (f[k], amp[k]),
                       xytext=(6, 4), textcoords="offset points", fontsize=8)
        cv.draw_idle()
        self.lbl_spec.setText(f"Пик: {f[k]:.1f} Гц, {amp[k]:.2f} В; Δf = {self.fd / self.p.n:.2f} Гц; "
                              f"серая часть — копия спектра со сдвигом Fд")

    def _draw_acf(self):
        if self.x is None:
            return
        tau, r = self.acf_data()
        cv = self.cv_acf
        cv.clear("Корреляционная функция дискретного сигнала", "τ, мс", "R, В²")
        cv.ax.plot(tau * 1e3, r, color="#188038", lw=1.5)
        cv.ax.annotate(f"R(0) = {r[0]:.3f} В²", (0, r[0]), xytext=(6, -2),
                       textcoords="offset points", fontsize=8)
        cv.ax.set_xlim(0, tau[-1] * 1e3)
        cv.draw_idle()
        self.lbl_acf.setText(f"R(0) = {r[0]:.4f} В²; первый нуль при τ ≈ "
                             f"{self._first_zero(tau, r):.2f} мс")

    @staticmethod
    def _first_zero(tau, r) -> float:
        idx = np.where(np.diff(np.signbit(r)))[0]
        return float(tau[idx[0]] * 1e3) if len(idx) else float("nan")

    def _draw_psd(self):
        if self.x is None:
            return
        f, s = self.psd_data()
        cv = self.cv_psd
        cv.clear("Спектральная плотность мощности (периодограмма)", "f, Гц", "S, В²/Гц")
        # лог. ось несовместима с нулевыми линиями из Canvas._style
        for ln in list(cv.ax.lines):
            if np.any(np.asarray(ln.get_ydata()) <= 0):
                ln.remove()
        pos = s > 0
        cv.ax.set_autoscaley_on(False)      # нулевая sticky-линия иначе ломает log
        cv.ax.semilogy(f[pos], s[pos], color="#8e24aa", lw=1.4)
        cv.ax.set_ylim(float(s[pos].min()) * 0.5, float(s[pos].max()) * 2)
        if self.p.kind in (KINDS[2], KINDS[3]):
            exp = 2 * max(self.p.noise_var, 0) / self.fd
            cv.ax.axhline(exp, color="#b3261e", ls="--", lw=0.9)
            cv.ax.annotate(f"2σ²/Fд = {exp:.4g}", (0, exp), xytext=(4, 4),
                           textcoords="offset points", fontsize=8, color="#b3261e")
        cv.ax.set_xlim(0, self.fd / 2)
        cv.draw_idle()
        mean = float(s[1:-1].mean()) if len(s) > 2 else float(s.mean())
        self.lbl_psd.setText(f"Средний уровень S = {mean:.4g} В²/Гц; N = {self.p.n}; "
                             f"Δf = {self.fd / self.p.n:.2f} Гц")
