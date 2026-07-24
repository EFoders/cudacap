"""Stage A RELBIL vs the tier-B oracle dump (reliability + combined modes).

RELBIL is the first routine reading /allMODE/ rather than /ZON/, so the cases
are lists of up to twenty candidate modes.  Every input is injected.

Three things the dump is built to catch, none of which shows up in the
arithmetic:

* the **two-pass** most-reliable-mode selection.  A single pass finds the
  wrong mode on some orderings -- the source says so -- so the test includes
  the ordering where a one-pass port would diverge.
* the **stale read**: when the only mode has a virtual height at or below
  70 km, RELBIL assigns none of D10R/D50R/D90R, and SNRLW/SNRUP/SNPR then
  return whatever the previous frequency left in /DON/.  The oracle seeds
  those three per case so the stale value is a known number rather than
  whatever the last case happened to leave.
* the **sentinels**: the oracle presets every dumped output to -8888 (NREL
  -1, DSLF/DSUF -7777) before each call, and the Python side seeds the same,
  so a field RELBIL never writes is compared as untouched rather than as
  coincidentally-zero.

Measured: 390 cases x 69 columns = 26,910 cells, ALL BIT-EXACT.  As with
GENOIS there is no residual to characterise, so one ULP bar as a libm safety
valve plus the exactness floor is the whole tolerance story.  One case does
divide by a zero decile spread and gets IEEE infinity; the Fortran does the
same and FNORML's 5-sigma clamp absorbs it, so it lands bit-exact too.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.allmodestate import AllModeState
from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.noisestate import AnoisState
from voacap_gpu.scalar.relbil import relbil
from voacap_gpu.scalar.signalstate import SigdState
from voacap_gpu.scalar.sonstate import LAYTYP, SonState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("relbil")

RELBIL_ULP_BAR = 4.0            # measured worst 0.0 -- a libm safety valve
# copies of an input, or set to a constant: no arithmetic can drift them
RELBIL_EXACT = {"nrel", "modecode", "nhp", "angle", "cprob", "vhigh",
                "delay", "rneff", "xnynois", "du_nois", "dl_nois",
                "gaint", "gainr", "dslf", "dsuf"}

SENTINEL = F32(-8888.0)
SCRATCH_SENTINEL = F32(-7777.0)
NM = 20

SCALARS = ["ifx", "freq", "rsn", "lufp", "du", "dl", "rcnse", "nmmod",
           "pwrdba", "txlo", "txhi", "d10r", "d50r", "d90r"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "hn", "sigpow", "fldst", "tgain",
          "rgain", "b", "timed", "tloss", "prob", "eff", "nmode"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

SON_COLS = ["angle", "cprob", "dblos", "dblosl", "dblosu", "dbu", "delay",
            "dbw", "nhp", "xnynois", "reliab", "sndb", "snpr", "snrlw",
            "snrup", "vhigh", "rneff", "du_nois", "dl_nois", "snxx",
            "gaint", "gainr"]


def _cantenna(txlo, txhi, pwrdba) -> CantennaState:
    """Only PWRDB is reached from RELBIL, so nothing but the ranges matter."""
    cant = CantennaState()
    cant.numants = 1
    cant.iats[1] = 1
    cant.xfqs[1], cant.xfqe[1] = F32(txlo), F32(txhi)
    cant.pwrdba[1] = F32(pwrdba)
    return cant


def _run(inp):
    """One case: build the injected state and call RELBIL."""
    v = dict(zip(COLUMNS, inp))
    allm = AllModeState()
    allm.nmmod = int(float(v["nmmod"]))
    allm.nrel = -1
    for i in range(1, NM + 1):
        for a in ARRAYS:
            val = float(v[f"{a}{i}"])
            if a == "nmode":
                allm.nmode[i] = int(val)
            else:
                getattr(allm, a)[i] = F32(val)
        allm.crel[i] = SENTINEL
        allm.rely[i] = SENTINEL
    anois = AnoisState(du=F32(v["du"]), dl=F32(v["dl"]),
                       rcnse=F32(v["rcnse"]))
    sigd = SigdState(dslf=SCRATCH_SENTINEL, dsuf=SCRATCH_SENTINEL)
    son = SonState(d10r=F32(v["d10r"]), d50r=F32(v["d50r"]),
                   d90r=F32(v["d90r"]))
    for name in SON_COLS:
        arr = getattr(son, name)
        arr[1:] = int(SENTINEL) if name == "nhp" else SENTINEL
    ifx = int(float(v["ifx"]))
    relbil(allm, anois, sigd, son,
           _cantenna(v["txlo"], v["txhi"], v["pwrdba"]),
           ifx, F32(v["freq"]), F32(v["rsn"]), int(float(v["lufp"])))
    got = {f"crel{i:02d}": allm.crel[i] for i in range(1, NM + 1)}
    got.update({f"rely{i:02d}": allm.rely[i] for i in range(1, NM + 1)})
    got["nrel"] = F32(allm.nrel)
    got["d10r"], got["d50r"], got["d90r"] = son.d10r, son.d50r, son.d90r
    for name in SON_COLS:
        got[name] = F32(getattr(son, name)[ifx])
    got["dslf"], got["dsuf"] = sigd.dslf, sigd.dsuf
    label = son.mode[ifx]
    got["modecode"] = F32(LAYTYP.index(label) + 1 if label in LAYTYP else 0)
    return got, allm, son


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_relbil_dumps.sh")
def test_relbil_vs_oracle():
    header, inputs, rows = DUMP
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    seen = {"no_modes": 0, "one_mode": 0, "many_modes": 0, "dead_mode": 0,
            "stale_deciles": 0, "not_the_max": 0, "clamped_low": 0,
            "clamped_high": 0, "lufp_under_50": 0, "lufp_over_50": 0}
    labels = set()
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got, allm, son = _run(inp)
        v = dict(zip(COLUMNS, inp))
        nmmod = int(float(v["nmmod"]))
        lufp = int(float(v["lufp"]))
        seen["no_modes"] += nmmod <= 0
        seen["one_mode"] += nmmod == 1
        seen["many_modes"] += nmmod > 1
        seen["dead_mode"] += any(float(v[f"hp{i}"]) <= 70.0
                                 for i in range(1, max(nmmod, 1) + 1))
        seen["stale_deciles"] += nmmod == 1 and float(v["hp1"]) <= 70.0
        seen["not_the_max"] += nmmod > 1 and allm.nrel != int(
            np.argmax([allm.rely[i] for i in range(1, nmmod + 1)])) + 1
        if nmmod > 1:
            seen["clamped_low"] += son.dblosl[int(float(v["ifx"]))] == 0.2
            seen["clamped_high"] += son.dblosu[int(float(v["ifx"]))] == 30.0
        if nmmod > 0:
            seen["lufp_under_50"] += lufp < 50
            seen["lufp_over_50"] += lufp >= 50
            labels.add(son.mode[int(float(v["ifx"]))])
        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            base = name.rstrip("0123456789")
            if base in RELBIL_EXACT or base in ("crel", "rely"):
                assert g == e, (f"case {k} {name}: got {g!r} expected {e!r} "
                                f"(must be bit-exact)")
                total += 1
                exact += 1
                continue
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= RELBIL_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert labels == set(LAYTYP), f"unreached layer label: {labels}"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def _one_mode_state(n, rely_driver, hn, sn):
    """An n-mode /allMODE/ whose reliabilities are ordered by `rely_driver`."""
    allm = AllModeState()
    allm.nmmod = n
    for i in range(1, n + 1):
        allm.hp[i] = F32(300.0)
        allm.sn[i] = F32(sn[i - 1])
        allm.hn[i] = F32(hn[i - 1])
        allm.tllow[i] = F32(9.0)
        allm.tlhgh[i] = F32(8.0)
        allm.sigpow[i] = F32(-120.0)
        allm.fldst[i] = F32(30.0)
        allm.nmode[i] = 3
        allm.prob[i] = F32(0.9)
    del rely_driver
    return allm


def _relbil(allm, rsn=24.0, lufp=90):
    son = SonState()
    relbil(allm, AnoisState(du=F32(9.7), dl=F32(6.0), rcnse=F32(-145.0)),
           SigdState(), son, _cantenna(2.0, 30.0, 40.0), 3, F32(14.0),
           F32(rsn), lufp)
    return son


def test_second_pass_can_walk_away_from_the_maximum():
    """The 140 loop compares against a best it has already moved.

    Modes 2 and 5 are both near the top; 5 is the outright maximum but 2 has
    fewer hops and is within 0.05, so the second pass replaces the maximum.
    A port that ran only the second pass, starting from mode 1, would land
    somewhere else entirely -- which is exactly what the source's added
    first pass exists to prevent.
    """
    allm = _one_mode_state(5, None, hn=[3.0, 1.0, 3.0, 3.0, 2.0],
                           sn=[10.0, 24.3, 11.0, 12.0, 24.4])
    _relbil(allm)
    rely = [float(allm.rely[i]) for i in range(1, 6)]
    assert np.argmax(rely) + 1 == 5, "mode 5 should have the max reliability"
    assert allm.nrel == 2, "fewer hops wins inside the 0.05 window"


def test_low_virtual_height_is_not_a_mode():
    """HP <= 70 km gets reliability 0.001 and never enters the selection."""
    allm = _one_mode_state(3, None, hn=[1.0, 2.0, 3.0], sn=[5.0, 40.0, 6.0])
    allm.hp[2] = F32(70.0)                    # the boundary is inclusive
    _relbil(allm)
    assert allm.rely[2] == F32(0.001) and allm.crel[2] == F32(0.001)
    assert allm.nrel != 2, "a dead mode must not win despite the best S/N"
    assert allm.crel[1] == F32(1000.0), "live modes get CREL 1000"


def test_stale_deciles_survive_a_dead_only_mode():
    """With no live mode RELBIL assigns none of D10R/D50R/D90R."""
    allm = _one_mode_state(1, None, hn=[1.0], sn=[22.0])
    allm.hp[1] = F32(50.0)
    son = SonState(d10r=F32(3.5), d50r=F32(12.0), d90r=F32(7.25))
    relbil(allm, AnoisState(du=F32(9.7), dl=F32(6.0), rcnse=F32(-145.0)),
           SigdState(), son, _cantenna(2.0, 30.0, 40.0), 3, F32(14.0),
           F32(24.0), 90)
    assert (son.d10r, son.d50r, son.d90r) == (F32(3.5), F32(12.0), F32(7.25))
    assert son.snrlw[3] == F32(3.5) and son.snrup[3] == F32(7.25)


def test_no_modes_returns_untouched():
    """NREL keeps whatever it held -- and before the first successful call
    that is the BLOCK DATA 1 (blkdat.for:181), not 0.  MPATH and SERPRB index
    /allMODE/ with it either way, so slot 1 rather than the padded slot 0."""
    allm = AllModeState()
    allm.nmmod = 0
    allm.crel[1] = SENTINEL
    son = SonState(d10r=F32(1.0))
    relbil(allm, AnoisState(), SigdState(), son, CantennaState(),
           3, F32(14.0), F32(24.0), 90)
    assert allm.crel[1] == SENTINEL and allm.nrel == 1
    assert son.reliab[3] == 0.0 and son.mode[3] == ""


def test_layer_label_out_of_range_is_refused():
    """NMODE outside 1..5 indexes LAYTYP out of bounds in the Fortran."""
    allm = _one_mode_state(1, None, hn=[1.0], sn=[22.0])
    allm.nmode[1] = 7
    with pytest.raises(IndexError):
        _relbil(allm)
