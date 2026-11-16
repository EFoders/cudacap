"""cudacap's OWN antenna path (READANT + ANTCALC) against the Fortran.

Every deck in ``golden/dumps/luffy`` puts the sample.00 isotrope at both
ends, and the replay tests inject the captured /cantenna/, so before these
decks the type-13 / type-14 file path -- ``readant.read_antenna`` +
``engine.antcalc_cant`` -- had never been compared with voacapl.  The decks in
``golden/p2p/antenna_decks`` (``tools/oracle/antenna/gen_antenna_decks.py``,
MANIFEST.tsv says what each covers) vary exactly that: type 14 at TX / RX /
both, type 13 at the TX with the beam on, 30, 90 and 180 degrees off the
path, OFFAZIM across the 359 -> 0 wrap, the long-path azimuth flip, type 13
at the RX, and a 3..20 MHz MINFREQ..MAXFREQ.  They were captured through the
LUFFY shim into ``golden/dumps/luffy_antenna``.

From the RAW deck -- CIRCUIT, ANTENNA and FREQUENCY cards, the stock ``.VOA``
files copied into ``tests/data/antennas`` -- the test builds the /cantenna/
with cudacap's own builder and never reads the captured one, then asserts:

* ``test_cantenna_vs_antcalc`` -- the built /cantenna/ equals ANTCALC's, read
  back by DECRED, field for field (see CANTENNA EQUALITY);
* ``test_son_vs_oracle`` -- one fresh circuit per hour through
  ``build_circuit`` + LUFFY reproduces the captured /SON/ with the bounds of
  ``tests/test_engine.py`` (hop counts exact);
* ``test_predict_vs_oracle`` -- the same through the public ``predict()``,
  for a subset of decks.

CANTENNA EQUALITY
-----------------
ANTCALC does not hand DECRED its tables in memory: it writes ``gainNN.dat``
(antcalc.for:336-347: XFQS/XFQE ``f5.0``, BEAMMAIN/OFFAZIM ``f7.2``, gains
clamped at -99.99 and written ``f7.3``, efficiency ``f6.2``) and DECRED reads
it back (decred.for:300-320).  So the captured /cantenna/ is that text round
trip, and the comparison is EXACT (float32 ==) after applying the same
round trip to the engine's values where the engine does not already:

* type-13 gain rows 1..30 and every efficiency: exact as built
  (``load_p2p_t13`` models the f7.3 / f6.2 round trip itself);
* TYPE 14 -- exact.  This suite found that ``load_p2p`` used to install the
  float32 sum ``table + PARM(1)`` WITHOUT the gainNN.dat round trip, which
  is the identity for PARM(1) = 0 (sample.14) but not for sample.14x's
  PARM(1) = 3.00 (-9.782 + 3 is float32 -6.7819996 where DECRED reads back
  -6.782; 511 of 2730 cells 1-2 ULP off, at most 1.5e-5 dB on /SON/).
  ``load_p2p`` now models the round trip; the check below still allows the
  raw value within 2 ULP and requires exact equality after the round trip,
  so it holds either way;
* XFQS / XFQE / BEAMMAIN: exact;
* OFFAZIM: equal to the engine's value written ``f7.2`` and read back.  The
  engine keeps the unrounded azimuth (it is the angle the installed slice
  was interpolated at, which ANTCALC also uses before writing it); nothing
  downstream reads OFFAZIM except the area-coverage sentinel -999, so the
  quantisation has no effect on GAIN;
* PWRKW / PWRDBA of the TRANSMIT entry: exact.  The receive entry's PWRKW is
  the RX card's power field in the Fortran (DECRED reads it for every card,
  decred.for:286-292) and 0 in the engine; it is never read (PWRDB matches
  ITR 1 only), so it is not compared;
* an isotropic end (sample.00, type 0) is built by the engine as
  ``antcalc_cant``'s ``None`` end plus the card gain as ``tx_gos`` /
  ``rx_gos``.  ANTCALC installs the constant gain over MINFREQ..MAXFREQ
  with XFQS/XFQE bracketing that range; the engine's isotrope covers 0..30
  with every row at the gain.  The two agree over the card's range, which
  is what is asserted (identical GAIN for every frequency in it).
"""

import functools
import pathlib
import time

import numpy as np
import pytest

from voacap_gpu import engine as _engine
from voacap_gpu.engine import (POLE_DEFAULT, antcalc_cant, build_circuit,
                               predict, predict_circuits)
from voacap_gpu.scalar.antennastate import NELEV, NFREQ
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.readant import read_antenna
from voacap_gpu.scalar.redmap import redmap

from tests.scalar.dumputil import ulp_diff_floored
from tests.scalar.test_luffy_replay import _rows
from tests.test_engine import (SON_ABS_BOUND, ULP_BAR, _SON_NUM,
                               _deck_method, _system_from_row)

REPO = pathlib.Path(__file__).resolve().parents[1]
DECKDIR = REPO / "golden" / "p2p" / "antenna_decks"
DUMPS = REPO / "golden" / "dumps" / "luffy_antenna"
ANTDIR = REPO / "tests" / "data" / "antennas"

DECKS = sorted(p.name[:-7] for p in DUMPS.glob("*_in.csv"))

# predict() end to end on these (one per antenna arm: type 14 both ends,
# type 13 off-path, the wrap, the long path, type 13 at the RX, the 3..20
# range) -- the rest go through build_circuit only, to keep the CPU run short.
PREDICT_DECKS = ("t14both_fbahel_m10_s160", "t13off90_toksyd_m01_s15",
                 "t13wrap360_madmos_m10_s100", "t13long_lonakl_m01_s100",
                 "t13rx_fbahel_m07_s100_v20", "range320_latok_m10_s160")

ISOTROPE = "samples/sample.00"

BACKENDS = ("cpu", "cuda")


def _need(backend):
    if backend == "cuda" and not _engine._cuda_usable():
        pytest.skip("no usable CUDA device")


# ---------------------------------------------------------------------------
# Raw deck parsing (the Fortran card formats)
# ---------------------------------------------------------------------------

def _fx(s):
    """A Fortran F-edit field: blanks ignored, all-blank is 0."""
    s = s.replace(" ", "")
    return float(s) if s else 0.0


@functools.lru_cache(maxsize=None)
def parse_deck(deck):
    """The cards of ``<deck>.dat`` that the antenna path consumes.

    CIRCUIT is antcalc.for's FORMAT(10x,f5.2,a1,3(f9.2,a1),2x,a) -- the path
    letter at column 49, 'L' the long great circle -- and ANTENNA is
    FORMAT(10x,4i5,f10.3,1x,a21,1x,f5.1,f10.4) (antcalc.for:118,
    decred.for:1510).  FREQUENCY is (10x,11f5.2).
    """
    out = {"ants": {}, "coeff": "CCIR"}
    with open(DECKDIR / f"{deck}.dat") as f:
        for line in f:
            line = line.rstrip("\n").ljust(80)
            key = line[:10]
            if key.startswith("CIRCUIT"):
                tlat, tns = _fx(line[10:15]), line[15]
                tlon, tew = _fx(line[16:25]), line[25]
                rlat, rns = _fx(line[26:35]), line[35]
                rlon, rew = _fx(line[36:45]), line[45]
                out["tx"] = (-tlat if tns == "S" else tlat,
                             -tlon if tew == "W" else tlon)
                out["rx"] = (-rlat if rns == "S" else rlat,
                             -rlon if rew == "W" else rlon)
                out["long"] = line[48] == "L"
            elif key.startswith("ANTENNA"):
                itr = int(line[10:15])
                out["ants"][itr] = dict(
                    idx=int(line[15:20]), fmin=int(line[20:25]),
                    fmax=int(line[25:30]), design=_fx(line[30:40]),
                    file=line[41:62].strip(), beam=_fx(line[63:68]),
                    power=_fx(line[68:78]))
            elif key.startswith("FREQUENCY"):
                fr = [_fx(line[10 + 5 * i:15 + 5 * i]) for i in range(11)]
                out["freqs"] = [F32(x) for x in fr if x > 0.0]
            elif key.startswith("COEFFS"):
                out["coeff"] = line[10:14]
            elif key.startswith("MONTH"):
                out["month"] = int(_fx(line[15:20]))
            elif key.startswith("SUNSPOT"):
                out["ssn"] = _fx(line[10:15])
    return out


def _iso_gain(itr, card):
    """The constant gain ANTCALC gives a sample.00 isotrope (type 0).

    PARM(1) = designfreq (antcalc.for:254), except that on the RECEIVE card
    a non-zero power field replaces it (antcalc.for:119).
    """
    if itr == 2 and card["power"] != 0.0:
        return card["power"]
    return card["design"]


def _parse_voa(card):
    if card["file"] == ISOTROPE:
        return None
    return read_antenna(ANTDIR / card["file"].split("/")[-1])


@functools.lru_cache(maxsize=None)
def build_cant(deck):
    """The /cantenna/ from the raw ANTENNA cards, with cudacap's builder."""
    d = parse_deck(deck)
    tx, rx = d["ants"][1], d["ants"][2]
    return antcalc_cant(
        _parse_voa(tx), _parse_voa(rx), d["tx"], d["rx"],
        power_kw=tx["power"],
        tx_gos=_iso_gain(1, tx) if tx["file"] == ISOTROPE else 0.0,
        rx_gos=_iso_gain(2, rx) if rx["file"] == ISOTROPE else 0.0,
        tx_bearing=tx["beam"], rx_bearing=rx["beam"], long_path=d["long"],
        tx_freq_range=(tx["fmin"], tx["fmax"]),
        rx_freq_range=(rx["fmin"], rx["fmax"]), freqs=d["freqs"])


def _f72(x):
    """A REAL*4 written f7.2 by ANTCALC and read back by DECRED."""
    return F32(float(f"{float(F32(x)):7.2f}"))


def _gainfile(row):
    """Gains through gainNN.dat: clamp at -99.99, write f7.3, read back."""
    row = np.maximum(np.asarray(row, np.float32), F32(-99.99))
    return np.array([F32(float(f"{float(g):7.3f}")) for g in row], np.float32)


# ---------------------------------------------------------------------------
# (a) /cantenna/ equality
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (DUMPS / "META.txt").exists(),
                    reason="antenna oracle dumps missing")
def test_antenna_files_are_the_oracle_bytes():
    """tests/data/antennas holds the exact files ANTCALC read for the capture.

    run_antenna_dumps.sh records their SHA-256 in META.txt; the copies here
    must hash the same, or the comparison below is against other inputs.
    """
    import hashlib
    recorded = {}
    for line in (DUMPS / "META.txt").read_text().splitlines():
        tok = line.split()
        if len(tok) == 2 and tok[1].startswith("samples/"):
            recorded[tok[1].split("/")[-1]] = tok[0]
    used = {parse_deck(d)["ants"][i]["file"].split("/")[-1]
            for d in DECKS for i in (1, 2)} - {ISOTROPE.split("/")[-1]}
    assert used and used <= set(recorded)
    for name in sorted(used):
        got = hashlib.sha256((ANTDIR / name).read_bytes()).hexdigest()
        assert got == recorded[name], f"tests/data/antennas/{name} drifted"


@pytest.mark.skipif(not DECKS, reason="antenna oracle dumps missing")
@pytest.mark.parametrize("deck", DECKS)
def test_cantenna_vs_antcalc(deck):
    d = parse_deck(deck)
    cant = build_cant(deck)
    cap = {}
    for v in _rows(DUMPS / f"{deck}_ant.csv"):
        cap.setdefault(int(v["iant"]), {})[int(v["ifreq"])] = v
    assert sorted(cap) == [1, 2] and cant.numants == 2

    for iat in (1, 2):
        card = d["ants"][iat]
        rows = cap[iat]
        v1 = rows[1]
        assert int(cant.iats[iat]) == int(v1["itr"]) == iat
        lo, hi = card["fmin"], card["fmax"]
        assert F32(v1["xfqs"]) == F32(lo) and F32(v1["xfqe"]) == F32(hi)
        assert F32(v1["beammain"]) == F32(card["beam"])
        # OFFAZIM: ANTCALC's f7.2 write, DECRED's read (see module doc)
        got_off = F32(cant.offazim[iat])
        assert _f72(got_off) == F32(v1["offazim"]), (
            f"{deck} iant {iat}: OFFAZIM {got_off!r} (f7.2 "
            f"{_f72(got_off)!r}) vs ANTCALC {F32(v1['offazim'])!r}")

        iso = card["file"] == ISOTROPE
        if iso:
            # the engine's isotrope covers the card's range (see module doc)
            assert cant.xfqs[iat] <= lo and cant.xfqe[iat] >= hi
            span = range(lo, hi + 1)
        else:
            assert F32(cant.xfqs[iat]) == F32(lo)
            assert F32(cant.xfqe[iat]) == F32(hi)
            span = range(1, NFREQ + 1)
        jant = None if iso else _parse_voa(card).jant
        bad = []
        for ifreq in span:
            v = rows[ifreq]
            exp = np.array([F32(v[f"g{j:02d}"]) for j in range(1, NELEV + 1)],
                           np.float32)
            got = np.asarray(cant.array[ifreq, 1:NELEV + 1, iat], np.float32)
            if jant == 14:
                # KNOWN DEVIATION (see module doc, TYPE 14): load_p2p skips
                # the gainNN.dat f7.3 round trip, so a table + PARM(1) sum
                # that leaves the 3-decimal grid is 1-2 ULP off ANTCALC's.
                # Applying the round trip must make it exact; the raw value
                # must be within 2 floored ULP.
                ulp = max((ulp_diff_floored(g, e) for g, e in zip(got, exp)),
                          default=0.0)
                if ulp > 2.0:
                    bad.append(f"f{ifreq}: raw type-14 gain {ulp:.0f} ULP off")
                got = _gainfile(got)
            if not np.array_equal(got, exp):
                k = int(np.flatnonzero(got != exp)[0])
                bad.append(f"f{ifreq} elev {k}: {got[k]!r} vs {exp[k]!r}")
            if F32(cant.aeff[ifreq, iat]) != F32(v["aeff"]):
                bad.append(f"f{ifreq} aeff {cant.aeff[ifreq, iat]!r} vs "
                           f"{F32(v['aeff'])!r}")
        assert not bad, f"{deck} iant {iat}: " + "; ".join(bad[:6])

    # the transmitter power (PWRDB reads ITR 1 only)
    assert F32(cant.pwrkw[1]) == F32(cap[1][1]["pwrkw"])
    assert F32(cant.pwrdba[1]) == F32(cap[1][1]["pwrdba"]), (
        f"{deck}: PWRDBA {cant.pwrdba[1]!r} vs {F32(cap[1][1]['pwrdba'])!r}")


# ---------------------------------------------------------------------------
# (b) /SON/ end to end
# ---------------------------------------------------------------------------

def _inputs(deck):
    d = parse_deck(deck)
    ins = _rows(DUMPS / f"{deck}_in.csv")
    v0 = ins[0]
    nf = sum(1 for i in range(1, 12) if v0[f"frel{i:02d}"] > 0)
    # the card's frequencies are LUFFY's FREL 1..nf
    assert [F32(v0[f"frel{i:02d}"]) for i in range(1, nf + 1)] == d["freqs"]
    assert int(v0["month"]) == d["month"] and F32(v0["ssn"]) == F32(d["ssn"])
    return d, ins, _system_from_row(v0), _deck_method(v0)


@functools.lru_cache(maxsize=None)
def run_deck(deck, backend):
    """Fresh circuit per hour, the engine-built cant, through LUFFY."""
    d, ins, system, method = _inputs(deck)
    cant = build_cant(deck)
    st = redmap(_engine._COEFFS_DIR, d["coeff"], d["month"], F32(d["ssn"]))
    circuits, ipflags = [], []
    for v in ins:
        c, ip = build_circuit(d["tx"], d["rx"], d["month"], F32(d["ssn"]),
                              int(v["it"]), d["freqs"], method=method,
                              system=system, cant=cant, coeff=d["coeff"],
                              pole=POLE_DEFAULT, npsl=int(d["long"]), iono=st)
        circuits.append(c)
        ipflags.append(ip)
    t0 = time.perf_counter()
    _, status = predict_circuits(circuits, ipflags, backend=backend)
    secs = time.perf_counter() - t0
    nf = len(d["freqs"])
    son = {nm: np.array([np.asarray(getattr(c.son, nm)[1:nf + 1], np.float32)
                         for c in circuits]) for nm in _SON_NUM}
    son["nhp"] = np.array([[int(c.son.nhp[j]) for j in range(1, nf + 1)]
                           for c in circuits], np.int32)
    son["it"] = np.array([int(c.time.it) for c in circuits], np.int32)
    return son, np.asarray(status, np.int32), secs


def _compare_son(deck, label, son, status, outs):
    """Hop counts exact; every numeric field inside test_engine's bounds."""
    assert status.tolist() == [0] * len(outs), f"{deck}: status {status}"
    H, nf = son["nhp"].shape
    assert H == len(outs)
    worst = {nm: (0.0, None) for nm in _SON_NUM}
    total = exact = 0
    for h, vout in enumerate(outs):
        it = int(son["it"][h])
        for j in range(nf):
            assert int(son["nhp"][h, j]) == int(vout[f"nhp{j + 1:02d}"]), (
                f"{deck} nhp{j + 1} h{it}: hop count differs")
            for nm in _SON_NUM:
                g = F32(son[nm][h, j])
                e = F32(vout[f"{nm}{j + 1:02d}"])
                where = f"{nm}{j + 1} h{it}: {g!r} vs {e!r}"
                d = ulp_diff_floored(g, e)
                total += 1
                exact += d == 0.0
                assert d <= ULP_BAR, f"{deck} {where} ({d:.0f} ULP)"
                a = abs(float(g) - float(e))
                if a > worst[nm][0]:
                    worst[nm] = (a, where)
                assert a <= SON_ABS_BOUND[nm], (
                    f"{deck} {where}: |d| {a:.3e} > {SON_ABS_BOUND[nm]:g}")
    print(f"\n{deck} [{label}]: {exact}/{total} bit-exact "
          f"({100 * exact / total:.2f}%); worst |d| "
          + ", ".join(f"{nm} {worst[nm][0]:.1e}" for nm in _SON_NUM))
    return worst


@pytest.mark.skipif(not DECKS, reason="antenna oracle dumps missing")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("deck", DECKS)
def test_son_vs_oracle(deck, backend):
    """Raw cards -> engine cant -> build_circuit -> LUFFY vs captured /SON/."""
    _need(backend)
    outs = _rows(DUMPS / f"{deck}_out.csv")
    son, status, _secs = run_deck(deck, backend)
    _compare_son(deck, backend, son, status, outs)


@pytest.mark.skipif(not DECKS, reason="antenna oracle dumps missing")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("deck", [d for d in PREDICT_DECKS if d in DECKS])
def test_predict_vs_oracle(deck, backend):
    """The public predict() from the raw cards and .VOA paths, end to end."""
    _need(backend)
    d, ins, system, method = _inputs(deck)
    tx, rx = d["ants"][1], d["ants"][2]
    tx_iso, rx_iso = tx["file"] == ISOTROPE, rx["file"] == ISOTROPE
    system.power_kw = tx["power"]
    system.tx_gos = _iso_gain(1, tx) if tx_iso else 0.0
    system.rx_gos = _iso_gain(2, rx) if rx_iso else 0.0
    hours = [int(v["it"]) for v in ins]
    r = predict(d["tx"], d["rx"], freqs=d["freqs"], month=d["month"],
                ssn=d["ssn"], hours=hours,
                tx_antenna=None if tx_iso else ANTDIR / tx["file"].split("/")[-1],
                rx_antenna=None if rx_iso else ANTDIR / rx["file"].split("/")[-1],
                tx_bearing=tx["beam"], rx_bearing=rx["beam"], system=system,
                method=method, coeff=d["coeff"], pole=POLE_DEFAULT,
                long_path=d["long"],
                tx_antenna_freq_range=(tx["fmin"], tx["fmax"]),
                rx_antenna_freq_range=(rx["fmin"], rx["fmax"]),
                backend=backend)
    outs = _rows(DUMPS / f"{deck}_out.csv")
    assert r.valid.all(), f"{deck}: invalid cells {np.argwhere(~r.valid)}"
    son = {"sndb": r.snr, "dbw": r.dbw, "dbu": r.field_dbu,
           "reliab": r.reliab, "nhp": r.nhp, "it": r.hours}
    # predict exposes a subset of /SON/; compare that subset with the bounds
    fields = ("sndb", "dbw", "dbu", "reliab")
    nf = len(d["freqs"])
    for h, vout in enumerate(outs):
        assert int(r.hours[h]) == int(ins[h]["it"])
        for j in range(nf):
            assert int(r.nhp[h, j]) == int(vout[f"nhp{j + 1:02d}"])
            for nm in fields:
                g, e = F32(son[nm][h, j]), F32(vout[f"{nm}{j + 1:02d}"])
                assert abs(float(g) - float(e)) <= SON_ABS_BOUND[nm], (
                    f"{deck} {nm}{j + 1} h{r.hours[h]}: {g!r} vs {e!r}")
    # and predict() agrees exactly with the build_circuit path above
    son_bc, _status, _ = run_deck(deck, backend)
    for nm in fields:
        assert np.array_equal(np.asarray(son[nm], np.float32), son_bc[nm]), nm
