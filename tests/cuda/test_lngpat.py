"""LNGPAT Stage C vs Stage A, on the lngpat oracle grid.

Where the long path's ONE mode is built: SELTXR's chosen ray at each end,
averaged into a single geometry and priced end to end through CONVH, GETTOP,
TABS, BABS, PRBMUF and PWRDB -- all six real, so this pins the composition as
well as the arithmetic.  Against the oracle that costs a ULP budget (BABS's
AF multiplied back out by DF*RZ, and PTOT's near-total cosine cancellation --
see tests/stage_a/test_lngpat.py); against Stage A it costs nothing, and the
bar is bit-identity on the whole of /allMODE/ and /SIGD/, with /RON//GEOG/,
/LPATH/, /LOSX/, /REFLX/, /MUFS/ and /ANOIS/ compared against their entry
copies to prove nothing else moved.

BABS and TABS disagree about a virtual height of exactly 70 km and LNGPAT
calls both with the same HPX; the branch counters below carry that arm, as
the Stage A test does.

Two structural claims get their own witness, both driven through the device:
the LTXRGM <= 0 early return writes NOTHING (every output keeps a distinctive
sentinel, including NMMOD), and slot 2 of /allMODE/ is not a second mode --
B(2) and NMODE(2) are the reception angle and receiving layer of the SAME one.

Where Stage A raises ModeIndexError -- NMODE at one end outside 1..4, so
YMUF(IS) reads off the end -- Stage C returns status 1 with /allMODE/ and
/SIGD/ exactly as Stage A leaves them at the raise.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.lngpat import ModeIndexError, lngpat
from voacap_gpu.cuda.state import (ALLMODE_DTYPE, ANOIS_DTYPE, LAYER_DTYPE,
                                      LOSX_DTYPE, LPATH_DTYPE, MUF_DTYPE,
                                      REFLX_DTYPE, SIGD_DTYPE, pack_allmode,
                                      pack_anois, pack_layer, pack_losx,
                                      pack_lpath, pack_muf, pack_reflx,
                                      pack_sigd)

from tests.scalar.dumputil import load_dump
from tests.scalar.test_lngpat import COLUMNS, OUT, SENTINEL, _arms, _build
from tests.scalar.test_lngpat import _plain
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)
from tests.cuda.test_settxr import unview

pytestmark = requires_cuda

DUMP = load_dump("lngpat")
NA = 21                 # /cantenna/ stride: NANT + 1, 1-based

_SRC = r'''
#include "lngpat.cuh"

extern "C" __global__ void drive(const LayerState* st, const LpathState* lpath,
                                 const LosxState* losx,
                                 const ReflxState* reflx, const MufState* muf,
                                 AllModeState* allm, SigdState* sigd,
                                 const AnoisState* anois, const int* iats,
                                 const float* xfqs, const float* xfqe,
                                 const float* pwrdba, const int* numants,
                                 int na, const float* freq, const float* gcd,
                                 const int* itxrcp2, int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    status[i] = lngpat(&st[i], &lpath[i], &losx[i], &reflx[i], &muf[i],
                       &allm[i], &sigd[i], &anois[i], iats + i * na,
                       xfqs + i * na, xfqe + i * na, pwrdba + i * na,
                       numants[i], freq[i], gcd[i], itxrcp2[i]);
}
'''

RO = (("st", LAYER_DTYPE), ("lpath", LPATH_DTYPE), ("losx", LOSX_DTYPE),
      ("reflx", REFLX_DTYPE), ("muf", MUF_DTYPE), ("anois", ANOIS_DTYPE))
RW = (("allm", ALLMODE_DTYPE), ("sigd", SIGD_DTYPE))
SLOT = {"st": 0, "lpath": 1, "losx": 2, "reflx": 3, "muf": 4, "allm": 5,
        "sigd": 6, "anois": 7}


def run_batch(vs):
    n = len(vs)
    keys = [k for k, _ in RO] + [k for k, _ in RW]
    pack = {"st": pack_layer, "lpath": pack_lpath, "losx": pack_losx,
            "reflx": pack_reflx, "muf": pack_muf, "anois": pack_anois,
            "allm": pack_allmode, "sigd": pack_sigd}
    pre = {k: [] for k in keys}
    exp = {k: [] for k, _ in RW}
    iats = np.zeros((n, NA), np.int32)
    xfqs = np.zeros((n, NA), np.float32)
    xfqe = np.zeros((n, NA), np.float32)
    pwrdba = np.zeros((n, NA), np.float32)
    numants = np.zeros(n, np.int32)
    freq = np.zeros(n, np.float32)
    gcd = np.zeros(n, np.float32)
    itxrcp2 = np.zeros(n, np.int32)
    exp_status = np.zeros(n, np.int32)
    for i, v in enumerate(vs):
        (st, lpath, losx, reflx, muf, allm, sigd, anois, cant,
         k2) = _build(v)
        obj = {"st": st, "lpath": lpath, "losx": losx, "reflx": reflx,
               "muf": muf, "anois": anois, "allm": allm, "sigd": sigd}
        iats[i] = cant.iats[:NA]
        xfqs[i], xfqe[i] = cant.xfqs[:NA], cant.xfqe[:NA]
        pwrdba[i] = cant.pwrdba[:NA]
        numants[i], itxrcp2[i] = cant.numants, k2
        freq[i], gcd[i] = F32(v["freq"]), F32(v["gcd"])
        for k in keys:
            pre[k].append(pack[k](obj[k]))
        try:
            lngpat(st, lpath, losx, reflx, muf, allm, sigd, anois, cant,
                   freq[i], gcd[i], k2)
        except ModeIndexError:
            exp_status[i] = 1
        for k, _ in RW:
            exp[k].append(pack[k](obj[k]))          # expected POST state
    pre = {k: np.stack(v) for k, v in pre.items()}
    exp = {k: np.stack(v) for k, v in exp.items()}

    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [pre["st"].view(np.uint8), pre["lpath"].view(np.uint8),
                       pre["losx"].view(np.uint8),
                       pre["reflx"].view(np.uint8),
                       pre["muf"].view(np.uint8), pre["allm"].view(np.uint8),
                       pre["sigd"].view(np.uint8),
                       pre["anois"].view(np.uint8),
                       iats.ravel(), xfqs.ravel(), xfqe.ravel(),
                       pwrdba.ravel(), numants, np.int32(NA),
                       freq, gcd, itxrcp2, status])
    got = {k: unview(fetch(dev[SLOT[k]]), dt) for k, dt in RO + RW}
    return got, exp, pre, fetch(dev[17]), exp_status


def assert_all_bitexact(got, exp, pre, tag=""):
    for key, dt in RW:
        for name in dt.names:
            assert_bitexact(got[key][name], exp[key][name],
                            f"{tag}{key}.{name}")
    for key, dt in RO:
        for name in dt.names:
            assert_bitexact(got[key][name], pre[key][name],
                            f"{tag}{key}.{name} (read-only)")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_lngpat_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    vs = [dict(zip(COLUMNS, (float(x) for x in inp))) for inp in inputs]
    got, exp, pre, status, exp_status = run_batch(vs)

    assert_bitexact(status, exp_status, "lngpat.status")
    assert (status == 0).all(), (
        f"lanes refused: {np.flatnonzero(status).tolist()}")
    assert_all_bitexact(got, exp, pre)

    # ---- branch coverage: the Stage A test's own arm counter, replayed ----
    seen = dict.fromkeys(
        ["early_return", "ran", "hpx_is_70", "dikm_at_or_below_-1",
         "dikm_above_-1", "over_the_top", "no_over_the_top", "df_floored",
         "babs_dead", "babs_live", "tabs_dead", "tabs_live", "fhop_floored",
         "fhop_free", "hn_floored", "hn_free", "tllow_clamped", "tllow_free",
         "tlhgh_clamped", "tlhgh_free", "layers_same", "layers_differ",
         "pwrdb_found", "pwrdb_fallback", "prob_tx_wins", "prob_rx_wins",
         "prob_tie"], 0)
    for i, v in enumerate(vs):
        _arms(v, {"tllow1": exp["allm"]["tllow"][i, 1],
                  "tlhgh1": exp["allm"]["tlhgh"][i, 1]}, seen)
    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"


def test_the_early_return_writes_nothing():
    """LTXRGM <= 0 at either end means SELTXR never chose a ray.  Every
    output keeps its sentinel -- including NMMOD, which is NOT set to 1."""
    vs = [_plain(l1=0), _plain(l2=0), _plain(l1=-1, l2=-1), _plain()]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="early ")
    for i in (0, 1, 2):
        allm, sigd = got["allm"][i], got["sigd"][i]
        assert allm["nmmod"] == -4444
        assert sigd["dslf"] == F32(-3333.0) and sigd["dsuf"] == F32(-2222.0)
        assert allm["prob"][1] == F32(0.8), "not even the AMIN1 on the last line"
        for name in OUT:
            if name in ("nmmod", "prob1", "dslf", "dsuf", "b1", "b2",
                        "nmode1", "nmode2"):
                continue
            assert allm[name[:-1]][1] == SENTINEL, name
        assert allm["b"][1] == SENTINEL and allm["b"][2] == SENTINEL
        assert allm["nmode"][1] == -8888 and allm["nmode"][2] == -8888
    assert got["allm"][3]["nmmod"] == 1, "... and the control lane ran"


def test_slot_2_carries_the_same_mode_and_prob_collapses():
    """NMMOD is 1, yet B(2) and NMODE(2) are written: they are the RECEPTION
    angle and receiving layer of the one mode.  PROB is the same shape --
    SETTXR leaves the two ends' MUFdays in slots 1 and 2 and the last line
    takes the smaller."""
    vs = [_plain(delfx1=8.0, delfx2=11.0, imode1=3, imode2=1),
          _plain(prob1=0.8, prob2=0.6),
          _plain(prob1=0.2, prob2=0.9)]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="slot2 ")
    allm = got["allm"]
    assert allm[0]["nmmod"] == 1
    assert allm[0]["b"][1] == F32(8.0) and allm[0]["b"][2] == F32(11.0)
    assert allm[0]["nmode"][1] == 3 and allm[0]["nmode"][2] == 1
    assert allm[1]["prob"][1] == F32(0.6)
    assert allm[2]["prob"][1] == F32(0.2)
    # /SIGD/ DSLF and DSUF are path parameters clobbered as scratch
    assert got["sigd"][0]["dslf"] != F32(-3333.0)
    assert got["sigd"][0]["dsuf"] != F32(-2222.0)


def test_one_layer_at_both_ends_skips_the_averaging_pass():
    """With NMODE(1) == NMODE(2) the second PRBMUF pass is skipped outright,
    so DSLF is that layer's spread rather than an average of two."""
    vs = [_plain(imode1=3, imode2=3),
          _plain(imode1=3, imode2=3, ymuf1=99.0, yfot1=88.0),
          _plain(imode1=3, imode2=1),
          _plain(imode1=1, imode2=1)]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="layers ")
    same, solo, mixed, one = [got["sigd"][i]["dslf"] for i in range(4)]
    assert same == solo, "layer 1's numbers cannot leak in"
    assert mixed != same
    assert abs(float(mixed) - 0.5 * (float(same) + float(one))) < 1e-5


def test_a_ray_with_no_layer_is_refused():
    """NMODE outside 1..4 indexes YMUF off the end of a four-element array.
    A ray SELTXR chose but FINDF never filled leaves IMODE at zero -- and the
    refusal can come on either pass, so DSLF/DSUF hold pass one's values when
    it is the RECEIVING layer that has none."""
    vs = [_plain(imode1=0), _plain(imode2=0), _plain(imode1=5), _plain()]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert exp_status.tolist() == [1, 1, 1, 0], "Stage A did not refuse"
    assert_bitexact(status, exp_status, "lngpat.status")
    assert_all_bitexact(got, exp, pre, tag="refused ")
    # lane 0 refuses on the FIRST pass, before DSLF/DSUF are written at all
    assert got["sigd"][0]["dslf"] == F32(-3333.0)
    assert got["sigd"][0]["dsuf"] == F32(-2222.0)
    # lane 1 refuses on the SECOND: the transmit layer's spread survives
    assert got["sigd"][1]["dslf"] != F32(-3333.0)
    assert got["sigd"][1]["dsuf"] != F32(-2222.0)
    # ... and both got as far as SN(1), which is where Stage A raises
    for i in (0, 1, 2):
        assert got["allm"][i]["nmmod"] == 1
        assert got["allm"][i]["sn"][1] != SENTINEL
        assert got["allm"][i]["tllow"][1] == SENTINEL, "the tail never ran"


def test_a_vanishing_ionospheric_distance_costs_nothing():
    """Two ways to reach the same zero: DIKM at or below -1 km sets both
    losses to zero outright, while DIKM just above it gets there the long
    way -- GETTOP refuses, DM is zero, DF is floored to zero, and BABS and
    TABS both return their 1000 dB/km marker into a leg of no length."""
    base = 0.5 * (3200.0 + 3400.0) / 6370.0
    vs = [_plain(gcd=base - 4000.0 / 6370.0),
          _plain(gcd=base - 0.5 / 6370.0)]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="vanishing ")
    for i in (0, 1):
        tl = float(got["allm"][i]["tloss"][1])
        fs = float(got["allm"][i]["fslos"][1])
        assert tl - fs < 100.0, "no 1000 dB/km leg leaked into the loss"
