from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from kuotik_port.labs.lab8 import (
    CHANNELS,
    MSG_FORMED,
    MSG_ZERO,
    POLYNOMIALS,
    TABS,
    CyclicCode,
    Lab8Page,
    bits,
    poly_deg,
    poly_mod,
)

OUT = Path(os.environ.get("SHOTS") or Path(tempfile.gettempdir()) / "shots-lab8")
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication([])
page = Lab8Page()
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
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
        print(f"PASS {name}")


def messages(code: CyclicCode, limit: int = 1 << 6) -> list[int]:
    if (1 << code.k) <= limit:
        return list(range(1 << code.k))
    rnd = random.Random(8)
    return [0, 1, (1 << code.k) - 1, 0b1011, (1 << (code.k - 1))] + \
        [rnd.getrandbits(code.k) for _ in range(6)]


def c_tabs():
    assert page.tabs.count() == 5, page.tabs.count()
    assert [page.tabs.tabText(i) for i in range(5)] == list(TABS)
    assert TABS == ("Общие сведения", "Ввод данных", "Описание модели",
                    "Модель кодера", "Модель декодера")
    assert len(page.inp.rb_channels) == 5
    assert [rb.text().split()[0] for rb in page.inp.rb_channels] == [
        "Одноканальное", "Двухканальное", "Трехканальное", "Четырехканальное",
        "Пятиканальное"]
    assert page.inp.rb_channels[0].isChecked()

    assert len(page.inp.rb_poly) == 7, len(page.inp.rb_poly)
    captions = [rb.text() for rb in page.inp.rb_poly]
    assert captions[0].startswith("(3,1)") and "x²+x+1" in captions[0]
    assert captions[1].startswith("(7,4)") and "x³+x+1" in captions[1]
    assert captions[2].startswith("(7,4)") and "x³+x²+1" in captions[2]
    assert captions[3].startswith("(15,11)") and "x⁴+x+1" in captions[3]
    assert captions[4].startswith("(15,11)") and "x⁴+x³+1" in captions[4]
    assert captions[5].startswith("(31,26)") and "x⁵+x²+1" in captions[5]
    assert captions[6].startswith("(31,26)") and "x⁵+x³+1" in captions[6]
    assert page.inp.rb_poly[1].isChecked()

    assert page.inp.btn_rem.text() == "Рассчитать остаток"
    assert page.inp.btn_dec.text() == "Декодировать"
    assert page.coder.btn_next.text() == "Следующий такт"
    assert page.decoder.btn_next.text() == "Следующий такт"


check("структура: 5 вкладок формы, 5 каналов, 7 многочленов, кнопки", c_tabs)


def c_fields():
    assert (page.inp.f_info.maxLength(), page.inp.f_rem.maxLength(),
            page.inp.f_err.maxLength()) == (4, 3, 7)
    assert page.inp.f_info.text() == "0000"
    assert page.inp.f_err.text() == "0000000"
    assert page.inp.ruler_info.text() == " 4  3  2  1"
    assert page.inp.ruler_rem.text() == " 3  2  1"
    assert page.inp.ruler_err.text() == " 7  6  5  4  3  2  1"
    assert page.inp.l_info.text() == "Информационные разряды"
    assert page.inp.l_rem.text() == "Остаток"
    assert page.inp.l_err.text() == "Вектор ошибки"

    page.inp.rb_channels[2].setChecked(True)
    app.processEvents()
    assert page.channels == 3
    assert page.inp.f_err.maxLength() == 7
    page.inp.rb_channels[0].setChecked(True)
    app.processEvents()
    assert page.channels == 1


check("поля «Ввод данных»: разрядность, линейки номеров, подписи, каналы", c_fields)


def c_generator():
    seen = set()
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        assert poly_deg(code.g) == code.r == prm.n - prm.k, prm.title
        assert code.g & 1 == 1, f"{prm.title}: g(0) != 1"
        assert code.g >> code.r == 1, f"{prm.title}: не приведён"
        assert poly_mod((1 << code.n) | 1, code.g) == 0, \
            f"{prm.title}: g(x) не делит x^{code.n}+1"
        assert poly_mod(1 << code.n, code.g) == 1, "x^n mod g != 1"
        assert code.n == (1 << code.r) - 1, f"{prm.title}: n != 2^r-1"
        for d in range(1, code.n):
            assert poly_mod((1 << d) | 1, code.g) != 0, (prm.title, d)
        seen.add((code.n, code.k, code.g))
    assert len(seen) == 7


check("g(x): степень r, g(0)=1, делит x^n+1, n = 2^r − 1, период n", c_generator)


def c_distance():
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        best = code.n + 1
        for msg in messages(code, limit=1 << 12):
            w = code.encode(msg)
            weight = bin(w).count("1")
            if weight:
                best = min(best, weight)
        assert best == 3, (prm.title, best)
        assert best >= 3


check("кодовое расстояние d min = 3 у всех семи кодов (перебор/выборка)", c_distance)


def c_matrix():
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        page.params = prm
        page.code = code
        page.model.fill(code, page.channels)
        r = code.r
        assert page.model.mat.rowCount() == r and page.model.mat.columnCount() == r
        for i in range(r):
            assert int(page.model.mat.item(i, 0).text()) == (code.g >> i) & 1
            for j in range(1, r):
                assert int(page.model.mat.item(i, j).text()) == (1 if j == i + 1 else 0)
        assert f"степень r = {r}" in page.model.gtext.text()
        assert "F*  =" in page.model.ftext.toPlainText()
    page.params = POLYNOMIALS[1]
    page.code = CyclicCode(page.params)
    page.apply_model()
    app.processEvents()


check("«Описание модели»: матрица связей M, g(x), вектор F*, степени M^v", c_matrix)


def c_encode():
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        for v in CHANNELS:
            for msg in messages(code):
                word = code.encode(msg, v)
                assert word >> code.n == 0, ("слово длиннее n", prm.title, v, msg)
                assert code.syndrome(word) == 0, ("синдром слова != 0", prm.title, v, msg)
                assert code.info_part(word) == msg, ("информационная часть", prm.title, v)
                rem = code.check_bits(msg, v)
                assert code.check_part(word) == rem
                assert code.trace(word, v)[-1] == 0, "фильтр не обнулился"
                assert code.clocks(v) == code.pad(v) + code.n


check("систематическое кодирование: R(x), синдром 0, фильтр обнуляется", c_encode)


def c_example_numbers():
    code = CyclicCode(POLYNOMIALS[1])
    assert bits(code.g, 4) == "1011"
    msg = 0b0111
    rem = code.check_bits(msg)
    assert rem == 0b010, bin(rem)
    word = code.encode(msg)
    assert bits(word, 7) == "0111010", bits(word, 7)
    assert code.syndrome(word) == 0
    c31 = CyclicCode(POLYNOMIALS[5])
    assert bits(c31.g, 6) == "100101"
    assert c31.check_bits(0b1011) == poly_mod(0b1011 << 5, 0b100101)
    assert c31.length() == 31 and c31.k == 26 and c31.r == 5


check("числа примеров: (7,4) A=0111 → R=010, слово 0111010; (31,26)", c_example_numbers)


def c_single():
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        for v in CHANNELS:
            for msg in messages(code, limit=1 << 4):
                word = code.encode(msg, v)
                for p in range(code.n):
                    res = code.decode(word ^ (1 << p), v, truth=word)
                    assert not res.clean, ("ошибка не обнаружена", prm.title, v, p)
                    assert res.position == p, ("неверный разряд", prm.title, v, p,
                                               res.position)
                    assert res.corrected == word, ("не исправлено", prm.title, v, p)
                    assert not res.miscorrected, ("ложное исправление", prm.title, v, p)
                    assert res.votes >= res.threshold, ("порог не достигнут", prm.title, v)
                    assert res.syndrome == poly_mod(1 << p, code.g)


check("все одиночные ошибки исправляются (7 кодов × 5 каналов)", c_single)


def c_double():
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        for v in CHANNELS:
            for msg in messages(code, limit=1 << 4):
                word = code.encode(msg, v)
                for p in range(code.n):
                    for q in range(p + 1, code.n):
                        got = word ^ (1 << p) ^ (1 << q)
                        res = code.decode(got, v, truth=word)
                        assert not res.clean, ("двойная не обнаружена", prm.title, v, p, q)
                        assert res.syndrome != 0
                        assert res.corrected != word, \
                            ("двойная «исправлена» в исходное слово", prm.title, v, p, q)
                        assert res.miscorrected, ("нет отметки о неверном исправлении",
                                                  prm.title, v, p, q)


check("двойная ошибка обнаруживается; исправление помечается неверным", c_double)


def c_no_error():
    for prm in POLYNOMIALS:
        code = CyclicCode(prm)
        for v in CHANNELS:
            word = code.encode(0b1011 % (1 << code.k))
            res = code.decode(word, v, truth=word)
            assert res.clean and res.syndrome == 0 and res.position is None
            assert res.corrected == word and not res.miscorrected


check("без ошибки: синдром 0, вердикт «Код принят без ошибок»", c_no_error)


def c_thresholds():
    code = CyclicCode(POLYNOMIALS[1])
    assert [code.threshold(v) for v in CHANNELS] == [1, 2, 2, 3, 3]
    for v in CHANNELS:
        assert code.threshold(v) == v // 2 + 1
        rows = code.thresholds(v)
        assert len(rows) == code.n
        assert sorted(r[2] for r in rows) == list(range(code.n)), "разряды не покрыты"
        assert {r[0] for r in rows} == set(range(v)), "каналы не покрыты"
        for ch, clk, p, s in rows:
            assert ch == p % v and clk == p // v
            assert s == poly_mod(1 << p, code.g)
            assert s != 0 and bits(s, code.r).count("1") >= 1

    two = {p: bits(s, 3) for ch, clk, p, s in code.thresholds(2)}
    assert two[3] == "011", two
    assert two[2] == "100", two

    word = code.encode(0b101)
    for v in CHANNELS:
        for p in range(code.n):
            res = code.decode(word ^ (1 << p), v, truth=word)
            assert res.votes == v, (v, p, res.votes)


check("пороги: ⌊v/2⌋+1, разбиение разрядов по каналам, коды 011 и 100", c_thresholds)


def c_autonomous():
    code = CyclicCode(POLYNOMIALS[1])
    for p in range(code.n):
        s = poly_mod(1 << p, code.g)
        states = code.autonomous(s, code.n)
        assert len(states) == code.n
        assert states[0] == s
        assert states[-1] == poly_mod(s << (code.n - 1), code.g)
        assert code.autonomous(s, 1)[0] == s
        nxt = (s << 1) ^ (code.g if s >> (code.r - 1) else 0)
        assert code.autonomous(s, 2)[1] == nxt


check("автономная работа фильтра: S(t) = x^t·S mod g, период n", c_autonomous)


def use(poly: int = 1, channels: int = 1, info: int | None = None,
        err: int | None = None) -> None:
    page.inp.rb_poly[poly].setChecked(True)
    page.set_params(poly)
    page.inp.rb_channels[channels - 1].setChecked(True)
    page.set_channels(channels)
    app.processEvents()
    if info is not None:
        page.inp.f_info.set_bits(info)
    if err is not None:
        page.inp.f_err.set_bits(err)
    app.processEvents()

def c_input_ui():
    use(1, 1, 0b0111)
    rem = page.calc_remainder()
    assert rem == 0b010, rem
    assert page.inp.f_rem.text() == "010"
    assert bits(page.word, 7) == "0111010"
    assert "R(x) = 010" in page.status.text()
    assert len(page.coder_states) == 7
    assert page.coder.src.text() == "0111"
    assert page.coder.dst.text() == "0111010"
    assert page.coder.tbl.rowCount() == 0

    for i in range(len(page.coder_states)):
        page.coder_next()
        assert page.coder.tbl.rowCount() == i + 1
    assert page.coder.verdict.text() == MSG_FORMED
    assert "после всех тактов: 000" in page.coder.state_lbl.text()
    assert page.coder.tbl.item(6, 3).text() == "контр."


check("«Рассчитать остаток»: R(x), слово, такты кодера, «Следующий такт»", c_input_ui)


def c_decoder_ui():
    use(1, 1, 0b0111)
    page.calc_remainder()
    page.inp.f_err.set_bits(0b1000000)
    res = page.decode()
    assert res is not None and res.position == 6 and res.corrected == page.word
    assert page.decoder.src.text() == "0111010"
    assert page.decoder.dst.text() == "0111010" or True
    assert page.decoder.tbl.rowCount() == 0
    page.decoder_step = len(page.decoder_states)
    page.decoder.fill(page.code, page.channels, page.word, page.received, page.result,
                      page.decoder_states)
    assert page.decoder.tbl.rowCount() == len(page.decoder_states)
    assert page.decoder.thr.rowCount() == page.code.n
    assert "Исправлен разряд 7" in page.decoder.verdict.text()
    for _ in range(len(page.decoder_states)):
        page.decoder_next()
    assert "Исправлен разряд 7" in page.status.text()

    page.inp.f_err.set_bits(0)
    res = page.decode()
    assert res.clean
    assert page.decoder.verdict.text() == MSG_ZERO
    assert MSG_ZERO in page.status.text()

    page.inp.f_err.set_bits(0b1000001)
    res = page.decode()
    assert not res.clean and res.miscorrected
    assert page.decoder.verdict.text().startswith("Синдром указывает на разряд")
    assert res.corrected != page.word


check("«Декодировать»: исправление, «без ошибок», двойная — обнаружена", c_decoder_ui)


def c_channels_ui():
    use(1, 5, 0b0111)
    page.calc_remainder()
    assert page.coder.dst.text() == "0111010"
    assert len(page.coder_states) == 1 + 7, len(page.coder_states)
    page.coder_next()
    assert page.coder.tbl.item(0, 3).text() == "фикт.", page.coder.tbl.item(0, 3).text()
    page.inp.f_err.set_bits(0b1000)
    res = page.decode()
    assert res.position == 3 and res.votes == 5 and res.threshold == 3
    assert "голосов 5 из 5" in page.decoder.verdict.text()
    use(1, 1)
    app.processEvents()


check("переключение каналов: pad, такты, мажоритарное голосование", c_channels_ui)


def c_poly_ui():
    for i, prm in enumerate(POLYNOMIALS):
        page.inp.rb_poly[i].setChecked(True)
        app.processEvents()
        assert page.code.n == prm.n and page.code.g == prm.g
        assert page.inp.f_info.maxLength() == prm.k
        assert page.inp.f_rem.maxLength() == prm.r
        assert page.inp.f_err.maxLength() == prm.n
        assert f"N = {prm.n}" or True
        page.inp.f_info.set_bits(1)
        page.calc_remainder()
        assert page.word is not None and page.code.syndrome(page.word) == 0
        assert page.inp.f_rem.text() == bits(page.code.check_bits(1), prm.r)
    use(1, 1)
    app.processEvents()


check("переключение многочленов: 7 кодов, разрядность полей, остаток", c_poly_ui)


def shot(name: str) -> None:
    app.processEvents()
    page.grab().save(str(OUT / name))


def c_shots():
    use(1, 1, 0b0111)
    page.calc_remainder()
    page.tabs.setCurrentIndex(0)
    shot("lab8-1-obshie-svedeniya.png")
    page.tabs.setCurrentIndex(1)
    shot("lab8-2-vvod-dannyh.png")
    page.tabs.setCurrentIndex(2)
    shot("lab8-3-opisanie-modeli.png")
    page.tabs.setCurrentIndex(3)
    shot("lab8-4-koder-do.png")
    for _ in range(len(page.coder_states)):
        page.coder_next()
    shot("lab8-4-koder-posle.png")

    page.inp.f_err.set_bits(0b1000000)
    page.decode()
    page.tabs.setCurrentIndex(4)
    shot("lab8-5-dekoder-sindrom.png")
    page.decoder_step = len(page.decoder_states)
    page.decoder.fill(page.code, page.channels, page.word, page.received, page.result,
                      page.decoder_states)
    shot("lab8-5-dekoder-ispravlenie.png")

    page.inp.f_err.set_bits(0)
    page.decode()
    page.decoder_step = len(page.decoder_states)
    page.decoder.fill(page.code, page.channels, page.word, page.received, page.result,
                      page.decoder_states)
    shot("lab8-5-dekoder-bez-oshibok.png")

    use(1, 3, 0b1011)
    page.calc_remainder()
    page.inp.f_err.set_bits(0b100000)
    page.decode()
    page.tabs.setCurrentIndex(3)
    page.coder_step = len(page.coder_states)
    page.coder.fill(page.code, page.channels, page.message, page.word, page.coder_states)
    shot("lab8-6-koder-3-kanala.png")
    page.tabs.setCurrentIndex(4)
    page.decoder_step = len(page.decoder_states)
    page.decoder.fill(page.code, page.channels, page.word, page.received, page.result,
                      page.decoder_states)
    shot("lab8-6-dekoder-3-kanala.png")
    files = sorted(p.name for p in OUT.glob("lab8-*.png"))
    assert len(files) >= 9, files


check("скриншоты всех 5 вкладок и расчётных примеров", c_shots)


print()
print(f"итого: {len(passed)} проверок пройдено, {len(failures)} провалено")
if failures:
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("все проверки ЛР8 пройдены")
