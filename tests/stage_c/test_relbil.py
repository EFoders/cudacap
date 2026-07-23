"""RELBIL Stage C vs Stage A, on the relbil oracle grid.

The reliability combiner: per-mode normal tests, the two-pass most-reliable-
mode selection, and the random-phase power sum that collapses up to twenty
modes into one /SON/ column.  Whole-struct compares on all three mutated
states (/allMODE/, /SIGD/, /SON/), every array seeded with a sentinel so an
untouched slot is provably untouched -- which is the only way to pin the two
things this routine is really about:

* the **stale /DON/ triple**.  When the only mode's virtual height is at or
  below 70 km RELBIL assigns none of D10R/D50R/D90R, and SNRLW/SNRUP/SNPR
  then read whatever the previous frequency left there.  The entry state
  seeds all three per case so the stale value is a known number.
* the **LAYTYP label**.  Stage A writes a 2-char label; Stage C writes the
  LAYTYP INDEX (state.py LABEL_FIELDS), so MODE(IFX) must come out as
  NMODE(IR) - 1 and every other slot must stay at the empty-label -1.

Where Stage A raises IndexError on an out-of-range NMODE, Stage C returns
status 1 with the state exactly as Stage A leaves it at the raise; that is
tested against Stage A's own post-exception state.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.allmodestate import AllModeState
from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.noisestate import AnoisState
from voacap_gpu.stage_a.relbil import relbil
from voacap_gpu.stage_a.signalstate import SigdState
from voacap_gpu.stage_a.sonstate import LAYTYP, SonState
from voacap_gpu.stage_c.state import (pack_allmode, pack_anois, pack_sigd,
                                      pack_son)

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("relbil")
SENTINEL = F32(-8888.0)
SCRATCH_SENTINEL = F32(-7777.0)
NM = 20
NA = 2                          # numants + 1, the /cantenna/ extent

SCALARS = ["ifx", "freq", "rsn", "lufp", "du", "dl", "rcnse", "nmmod",
           "pwrdba", "txlo", "txhi", "d10r", "d50r", "d90r"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "hn", "sigpow", "fldst", "tgain",
          "rgain", "b", "timed", "tloss", "prob", "eff", "nmode"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

# every /SON/ float array, so a slot RELBIL does not write is provably stale
SON_ARRAYS = ["angle", "angler", "cprob", "dblos", "dblosl", "dblosu", "dbu",
              "delay", "dbw", "xnynois", "probmp", "reliab", "sndb", "snpr",
              "snrlw", "snrup", "sprob", "vhigh", "rneff", "du_nois",
              "dl_nois", "snxx", "gaint", "gainr"]

_SRC = r'''
#include "relbil.cuh"

extern "C" __global__ void drive(AllModeState* allm, const AnoisState* anois,
                                 SigdState* sigd, SonState* son,
                                 const int* iats, const float* xfqs_all,
                                 const float* xfqe_all,
                                 const float* pwrdba_all, const int* ifx,
                                 const float* freq, const float* rsn,
                                 const int* lufp, int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    status[i] = relbil(&allm[i], &anois[i], &sigd[i], &son[i], iats,
                       xfqs_all + i * 2, xfqe_all + i * 2, pwrdba_all + i * 2,
                       1, ifx[i], freq[i], rsn[i], lufp[i]);
}
'''


def _cantenna(txlo, txhi, pwrdba) -> CantennaState:
    """Only PWRDB is reached from RELBIL, so nothing but the ranges matter."""
    cant = CantennaState()
    cant.numants = 1
    cant.iats[1] = 1
    cant.xfqs[1], cant.xfqe[1] = F32(txlo), F32(txhi)
    cant.pwrdba[1] = F32(pwrdba)
    return cant


def _entry(v):
    """The entry state for one dump row, exactly as the Stage A test builds."""
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
    anois = AnoisState(du=F32(v["du"]), dl=F32(v["dl"]), rcnse=F32(v["rcnse"]))
    sigd = SigdState(dslf=SCRATCH_SENTINEL, dsuf=SCRATCH_SENTINEL)
    son = SonState(d10r=F32(v["d10r"]), d50r=F32(v["d50r"]),
                   d90r=F32(v["d90r"]), d10s=SENTINEL, d50s=SENTINEL,
                   d90s=SENTINEL)
    for name in SON_ARRAYS:
        getattr(son, name)[1:] = SENTINEL
    son.nhp[1:] = int(SENTINEL)
    return allm, anois, sigd, son


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_relbil_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    cases = [dict(zip(COLUMNS, inp)) for inp in inputs]
    n = len(cases)

    p_allm, p_anois, p_sigd, p_son = [], [], [], []
    e_allm, e_sigd, e_son = [], [], []
    xfqs = np.zeros((n, NA), np.float32)
    xfqe = np.zeros((n, NA), np.float32)
    pwrdba = np.zeros((n, NA), np.float32)
    for k, v in enumerate(cases):
        allm, anois, sigd, son = _entry(v)
        cant = _cantenna(v["txlo"], v["txhi"], v["pwrdba"])
        xfqs[k] = cant.xfqs[:NA]
        xfqe[k] = cant.xfqe[:NA]
        pwrdba[k] = cant.pwrdba[:NA]
        p_allm.append(pack_allmode(allm))
        p_anois.append(pack_anois(anois))
        p_sigd.append(pack_sigd(sigd))
        p_son.append(pack_son(son))
        relbil(allm, anois, sigd, son, cant, int(float(v["ifx"])),
               F32(v["freq"]), F32(v["rsn"]), int(float(v["lufp"])))
        e_allm.append(pack_allmode(allm))
        e_sigd.append(pack_sigd(sigd))
        e_son.append(pack_son(son))
    exp_allm = np.stack(e_allm)
    exp_sigd = np.stack(e_sigd)
    exp_son = np.stack(e_son)

    ifx = np.array([int(float(v["ifx"])) for v in cases], np.int32)
    freq = np.array([F32(v["freq"]) for v in cases], np.float32)
    rsn = np.array([F32(v["rsn"]) for v in cases], np.float32)
    lufp = np.array([int(float(v["lufp"])) for v in cases], np.int32)
    nmmod = np.array([int(float(v["nmmod"])) for v in cases], np.int32)
    iats = np.array([0, 1], np.int32)
    status = np.full(n, -1, np.int32)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(p_allm).view(np.uint8),
                       np.stack(p_anois).view(np.uint8),
                       np.stack(p_sigd).view(np.uint8),
                       np.stack(p_son).view(np.uint8),
                       iats, xfqs.ravel(), xfqe.ravel(), pwrdba.ravel(),
                       ifx, freq, rsn, lufp, status])
    got_allm = fetch(dev[0]).view(exp_allm.dtype)
    got_anois = fetch(dev[1]).view(np.stack(p_anois).dtype)
    got_sigd = fetch(dev[2]).view(exp_sigd.dtype)
    got_son = fetch(dev[3]).view(exp_son.dtype)

    assert (fetch(dev[12]) == 0).all(), "no dump lane may refuse"
    for name in exp_allm.dtype.names:
        assert_bitexact(got_allm[name], exp_allm[name], f"allm.{name}")
    for name in exp_sigd.dtype.names:
        assert_bitexact(got_sigd[name], exp_sigd[name], f"sigd.{name}")
    for name in exp_son.dtype.names:
        assert_bitexact(got_son[name], exp_son[name], f"son.{name}")
    for name in got_anois.dtype.names:      # /ANOIS/ is read-only here
        assert_bitexact(got_anois[name], np.stack(p_anois)[name],
                        f"anois.{name} (read-only)")

    # the persisted /DON/ triple carries across a dead-only mode
    dead_only = (nmmod == 1) & np.array(
        [F32(v["hp1"]) <= F32(70.0) for v in cases])
    assert dead_only.any(), "no dead-only-mode row: the stale read is untested"
    pre_son = np.stack(p_son)
    for name in ("d10r", "d50r", "d90r"):
        assert_bitexact(exp_son[name][dead_only], pre_son[name][dead_only],
                        f"son.{name} must survive a dead-only mode")
    live = ifx[dead_only]
    assert np.array_equal(exp_son["snrlw"][dead_only, live],
                          pre_son["d10r"][dead_only])
    assert np.array_equal(exp_son["snrup"][dead_only, live],
                          pre_son["d90r"][dead_only])

    # the label field really is the LAYTYP INDEX, and only slot IFX moved
    run = nmmod > 0
    got_lbl = got_son["mode"][np.arange(n), ifx]
    want = np.array([int(float(cases[k][f"nmode{exp_allm['nrel'][k]}"])) - 1
                     for k in range(n) if run[k]])
    assert np.array_equal(got_lbl[run], want), "MODE != NMODE(IR) - 1"
    assert (got_lbl[run] >= 0).all() and (got_lbl[run] <= 4).all()
    assert set(got_lbl[run].tolist()) == set(range(len(LAYTYP))), \
        "not every LAYTYP index is produced"
    assert (got_lbl[~run] == -1).all(), "an untouched column got a label"
    for name in ("mdl", "moder"):
        assert (got_son[name] == -1).all(), f"son.{name} is not RELBIL's"

    # branch coverage, exactly the arms the Stage A test pins
    seen = {
        "no_modes": int((nmmod <= 0).sum()),
        "one_mode": int((nmmod == 1).sum()),
        "many_modes": int((nmmod > 1).sum()),
        "dead_mode": sum(any(F32(v[f"hp{i}"]) <= F32(70.0) for i in
                             range(1, max(int(float(v["nmmod"])), 1) + 1))
                         for v in cases),
        "stale_deciles": int(dead_only.sum()),
        "not_the_max": 0,
        "clamped_low": 0,
        "clamped_high": 0,
        "lufp_under_50": int((lufp[run] < 50).sum()),
        "lufp_over_50": int((lufp[run] >= 50).sum()),
    }
    for k in range(n):
        if nmmod[k] > 1:
            rely = exp_allm["rely"][k, 1:nmmod[k] + 1]
            seen["not_the_max"] += int(exp_allm["nrel"][k]
                                       != int(np.argmax(rely)) + 1)
            seen["clamped_low"] += int(exp_son["dblosl"][k, ifx[k]] == F32(0.2))
            seen["clamped_high"] += int(
                exp_son["dblosu"][k, ifx[k]] == F32(30.0))
    for arm, cnt in seen.items():
        assert cnt, f"branch never reached: {arm} ({seen})"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_layer_label_out_of_range_returns_status_1():
    """NMODE outside 1..5 indexes LAYTYP out of bounds in the Fortran.

    Stage A raises IndexError partway through the tail; Stage C must return
    status 1 having written EXACTLY what Stage A wrote before raising, so the
    comparison is against Stage A's own post-exception state.
    """
    def build(nmode1):
        allm = AllModeState()
        allm.nmmod = 1
        allm.hp[1] = F32(300.0)
        allm.sn[1] = F32(22.0)
        allm.hn[1] = F32(1.0)
        allm.tllow[1] = F32(9.0)
        allm.tlhgh[1] = F32(8.0)
        allm.sigpow[1] = F32(-120.0)
        allm.fldst[1] = F32(30.0)
        allm.prob[1] = F32(0.9)
        allm.nmode[1] = nmode1
        anois = AnoisState(du=F32(9.7), dl=F32(6.0), rcnse=F32(-145.0))
        sigd = SigdState(dslf=SCRATCH_SENTINEL, dsuf=SCRATCH_SENTINEL)
        son = SonState()
        for name in SON_ARRAYS:
            getattr(son, name)[1:] = SENTINEL
        son.nhp[1:] = int(SENTINEL)
        return allm, anois, sigd, son

    cases = [7, 3]                      # refused, then a clean control lane
    p_allm, p_anois, p_sigd, p_son = [], [], [], []
    e_allm, e_sigd, e_son = [], [], []
    for nm in cases:
        allm, anois, sigd, son = build(nm)
        p_allm.append(pack_allmode(allm))
        p_anois.append(pack_anois(anois))
        p_sigd.append(pack_sigd(sigd))
        p_son.append(pack_son(son))
        cant = _cantenna(2.0, 30.0, 40.0)
        try:
            relbil(allm, anois, sigd, son, cant, 3, F32(14.0), F32(24.0), 90)
        except IndexError:
            assert nm == 7
        else:
            assert nm == 3
        e_allm.append(pack_allmode(allm))
        e_sigd.append(pack_sigd(sigd))
        e_son.append(pack_son(son))
    exp_allm, exp_sigd = np.stack(e_allm), np.stack(e_sigd)
    exp_son = np.stack(e_son)

    n = len(cases)
    cant = _cantenna(2.0, 30.0, 40.0)
    status = np.full(n, -1, np.int32)
    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(p_allm).view(np.uint8),
                       np.stack(p_anois).view(np.uint8),
                       np.stack(p_sigd).view(np.uint8),
                       np.stack(p_son).view(np.uint8),
                       np.array([0, 1], np.int32),
                       np.tile(cant.xfqs[:NA].astype(np.float32),
                               (n, 1)).ravel(),
                       np.tile(cant.xfqe[:NA].astype(np.float32),
                               (n, 1)).ravel(),
                       np.tile(cant.pwrdba[:NA].astype(np.float32),
                               (n, 1)).ravel(),
                       np.full(n, 3, np.int32), np.full(n, F32(14.0),
                                                        np.float32),
                       np.full(n, F32(24.0), np.float32),
                       np.full(n, 90, np.int32), status])
    assert fetch(dev[12]).tolist() == [1, 0]
    for j, exp in ((0, exp_allm), (2, exp_sigd), (3, exp_son)):
        got = fetch(dev[j]).view(exp.dtype)
        for name in exp.dtype.names:
            assert_bitexact(got[name], exp[name], f"[{j}] {name}")
    # the refused lane stopped exactly where Stage A raised
    assert exp_son["cprob"][0, 3] == F32(0.9)       # written before the raise
    assert exp_son["mode"][0, 3] == -1              # the label was not
    assert exp_son["snpr"][0, 3] == SENTINEL        # nor the tail
