"""ESMOD Stage C vs Stage A, on the esmod oracle grid.

The sporadic-E modes: /ZON/ slots 4 and 5, a thin mirror at HS rather than a
refraction through a profile.  Every slot of the entry /ZON/ is seeded with the
oracle's 12345. sentinel (NMODE -9), exactly as the Stage A test does, so the
whole-struct compare proves three things at once: the arithmetic in slots 4-5,
that the unconditional presets survive all four early returns, and that slots
1, 2, 3, 6 and 7 are still carrying the sentinel afterwards.

Two ESMOD details the compare pins rather than merely exercises:

* ``PROB = PROS**IHOP`` is an INTEGER exponent, so it is gfortran's powi
  binary exponentiation, not pow32 -- a pow32 would differ in the last bit.
* a two-hop path runs ``IHOP = 2,2``: ONE iteration, writing slot **4** and
  leaving slot 5 at its presets.  Slot 4 is "the first mode found", not "the
  one-hop mode".  Covered by the dump grid and again by an explicit case.

Where Stage A raises through GAIN, Stage C returns status 3 or 4; the dump
grid drives neither, which is asserted rather than assumed.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.esmod import esmod
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.noisestate import AnoisState
from voacap_gpu.scalar.raysstate import RaysState
from voacap_gpu.scalar.readant import read_antenna
from voacap_gpu.scalar.signalstate import SigdState
from voacap_gpu.scalar.zonstate import ZonState
from voacap_gpu.cuda.state import (pack_anois, pack_layer, pack_muf,
                                      pack_rays, pack_sigd, pack_zon)

from tests.scalar.dumputil import ANTENNAS_DIR, load_dump
from tests.scalar.test_esmod import COLUMNS, FIELDS, SENTINEL, SENTINEL_NMODE
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)
from tests.cuda.test_regmod import shared_tables

pytestmark = requires_cuda

DUMP = load_dump("esmod")
RZ = F32(6370.0)

_SRC = r'''
#include "esmod.cuh"

extern "C" __global__ void drive(const LayerState* st, const MufState* muf,
                                 const SigdState* sigd,
                                 const AnoisState* anois,
                                 const RaysState* rays, const float* array,
                                 const float* aeff, const int* iats,
                                 const float* xfqs_all, const float* xfqe_all,
                                 const float* offazim,
                                 const float* pwrdba_all, int numants,
                                 ZonState* zon, const float* freq,
                                 const int* jmode, const float* gcd,
                                 const float* gcdkm, const float* amind,
                                 int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int na = numants + 1;
    // rays_view_of() is the stride-1 view: byte-for-byte the packed struct
    // this test has always built, which is what proves the SoA refactor inert.
    status[i] = esmod(&st[i], &muf[i], &sigd[i], &anois[i],
                      rays_view_of(&rays[i]),
                      array, aeff, iats, xfqs_all + i * na, xfqe_all + i * na,
                      offazim, pwrdba_all + i * na, numants, &zon[i],
                      freq[i], jmode[i], gcd[i], gcdkm[i], amind[i]);
}
'''

_ANT = None


def _cantenna(txlo, txhi, rxlo, rxhi, pwrdba) -> CantennaState:
    """sample.14 as both /cantenna/ terminals, exactly as the Stage A test."""
    global _ANT
    if _ANT is None:
        _ANT = read_antenna(ANTENNAS_DIR / "sample.14").p2p_tables()
    array, aeff = _ANT
    cant = CantennaState()
    cant.numants = 2
    cant.iats[1], cant.iats[2] = 1, 2
    cant.xfqs[1], cant.xfqe[1] = F32(txlo), F32(txhi)
    cant.xfqs[2], cant.xfqe[2] = F32(rxlo), F32(rxhi)
    cant.offazim[1] = cant.offazim[2] = F32(0.0)
    for iat in (1, 2):
        cant.array[1:31, 1:92, iat] = array
        cant.aeff[1:31, iat] = aeff
    cant.pwrdba[1] = F32(pwrdba)
    return cant


def _seeded_zon() -> ZonState:
    """All seven slots at the oracle's sentinel, not zero."""
    zon = ZonState()
    for name in FIELDS:
        if name == "nmode":
            zon.nmode[1:8] = SENTINEL_NMODE
        else:
            getattr(zon, name)[1:8] = SENTINEL
    return zon


def _build(inp):
    """Every state ESMOD reads, from one case row -- the Stage A test's shape."""
    v = dict(zip(COLUMNS, inp))
    state = LayerState()
    state.km = int(float(v["km"]))
    state.kfx = 1
    for i in range(1, 6):
        state.fs[1, i] = F32(v[f"fs1_{i}"])
        state.fs[2, i] = F32(v[f"fs2_{i}"])
        state.fs[3, i] = F32(v[f"fs3_{i}"])
        state.hs[i] = F32(v[f"hs{i}"])
        state.fi[1, i] = F32(v[f"fi1_{i}"])
        state.gyz[i] = F32(v[f"gyz{i}"])
        state.sigpat[i] = F32(v[f"sigpat{i}"])
        state.epspat[i] = F32(v[f"epspat{i}"])
    muf = MufState()
    muf.sigl[4] = F32(v["sigl4"])
    muf.sigu[4] = F32(v["sigu4"])
    muf.ymuf[4] = F32(v["ymuf4"])
    muf.yfot[4] = F32(v["yfot4"])
    muf.yhpf[4] = F32(v["yhpf4"])
    sigd = SigdState(acav=F32(v["acav"]), afe=F32(v["afe"]), bfe=F32(v["bfe"]),
                     asm=F32(v["asm"]), dsl=F32(v["dsl"]), dsu=F32(v["dsu"]))
    anois = AnoisState(rcnse=F32(v["rcnse"]))
    jmode = int(float(v["jmode"]))
    rays = RaysState()
    rays.ifob[1, 3, jmode] = int(float(v["ifob13"]))
    cant = _cantenna(v["txlo"], v["txhi"], v["rxlo"], v["rxhi"], v["pwrdba"])
    gcdkm = F32(v["gcdkm"])
    return (state, muf, sigd, anois, rays, cant, F32(v["freq"]), jmode,
            gcdkm / RZ, gcdkm, F32(v["amind"]))


def _run_batch(built):
    n = len(built)
    cant0 = built[0][5]
    na = cant0.numants + 1
    p_st, p_muf, p_sigd, p_anois, p_rays, p_zon = ([] for _ in range(6))
    e_zon = []
    xfqs = np.zeros((n, na), np.float32)
    xfqe = np.zeros((n, na), np.float32)
    pwrdba = np.zeros((n, na), np.float32)
    freq = np.zeros(n, np.float32)
    jmode = np.zeros(n, np.int32)
    gcd = np.zeros(n, np.float32)
    gcdkm = np.zeros(n, np.float32)
    amind = np.zeros(n, np.float32)
    for i, b in enumerate(built):
        st, muf, sigd, anois, rays, cant, f, k, g, gk, am = b
        xfqs[i], xfqe[i], pwrdba[i] = cant.xfqs[:na], cant.xfqe[:na], \
            cant.pwrdba[:na]
        freq[i], jmode[i], gcd[i], gcdkm[i], amind[i] = f, k, g, gk, am
        p_st.append(pack_layer(st))
        p_muf.append(pack_muf(muf))
        p_sigd.append(pack_sigd(sigd))
        p_anois.append(pack_anois(anois))
        p_rays.append(pack_rays(rays))
        zon = _seeded_zon()
        p_zon.append(pack_zon(zon))
        esmod(st, muf, sigd, anois, rays, cant, zon, f, k, g, gk, am)
        e_zon.append(pack_zon(zon))
    exp_zon = np.stack(e_zon)

    array, aeff, iats, offazim = shared_tables(cant0)
    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(p_st).view(np.uint8),
                       np.stack(p_muf).view(np.uint8),
                       np.stack(p_sigd).view(np.uint8),
                       np.stack(p_anois).view(np.uint8),
                       np.stack(p_rays).view(np.uint8),
                       array, aeff, iats, xfqs.ravel(), xfqe.ravel(), offazim,
                       pwrdba.ravel(), np.int32(cant0.numants),
                       np.stack(p_zon).view(np.uint8),
                       freq, jmode, gcd, gcdkm, amind, status])
    got_zon = fetch(dev[13]).view(exp_zon.dtype)
    return got_zon, exp_zon, fetch(dev[19]), np.stack(p_zon)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_esmod_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    built = [_build(inp) for inp in inputs]
    n = len(built)
    got_zon, exp_zon, status, pre_zon = _run_batch(built)

    assert (status == 0).all(), (
        f"lanes refused: {np.flatnonzero(status).tolist()}")
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")

    # the slots ESMOD must not touch still carry the entry sentinel
    for s in (1, 2, 3, 6, 7):
        for name in FIELDS:
            assert_bitexact(got_zon[name][:, s], pre_zon[name][:, s],
                            f"zon.{name}[{s}] (not ESMOD's slot)")

    # branch coverage, exactly the arms the Stage A test pins
    hn4 = exp_zon["hn"][:, 4]
    hn5 = exp_zon["hn"][:, 5]
    seen = {
        "one_hop": int((hn4 == F32(1.0)).sum()),
        "two_hop": int((hn4 == F32(2.0)).sum()),
        "no_mode": int((hn4 == F32(-1.0)).sum()),
        "amind_cut": int(((hn4 == F32(-1.0)) & (hn5 > 0)).sum()),
        "slot5_live": int((hn5 > 0).sum()),
        "slot5_preset": int((hn5 == F32(-1.0)).sum()),
    }
    for arm, cnt in seen.items():
        assert cnt, f"branch never reached: {arm} ({seen})"

    # a live slot (ABPS off its 1000. preset, ADV forced to 0 as Es has no
    # deviative loss) and a preset-only slot must both be in the set
    live_slots = sum(int(exp_zon["abps"][i, s] != F32(1000.0)
                         and exp_zon["adv"][i, s] == F32(0.0))
                     for i in range(n) for s in (4, 5))
    assert live_slots, "no live Es slot at all"
    assert (exp_zon["nmode"][:, 4] == 4).any(), "no computed Es mode"
    assert (exp_zon["nmode"][:, 5] == 5).any(), "no preset-only slot 5"


def test_two_hop_path_writes_slot_4_not_slot_5():
    """IHOP = 2,2 is ONE iteration, and it fills slot 4, not slot 5.

    The detail most likely to be mis-ported, and the three early returns come
    along for free: the too-far case leaves both slots at their presets.
    """
    state = LayerState()
    state.km = 1
    state.hs[1] = F32(110.0)
    state.fs[1, 1], state.fs[2, 1], state.fs[3, 1] = (F32(4.0), F32(5.0),
                                                      F32(7.0))
    state.fi[1, 1] = F32(3.0)
    state.gyz[1] = F32(1.2)
    state.sigpat[1], state.epspat[1] = F32(5.0), F32(80.0)
    muf = MufState()
    muf.sigl[4], muf.sigu[4] = F32(0.15), F32(0.20)
    muf.ymuf[4], muf.yfot[4], muf.yhpf[4] = F32(14.0), F32(11.0), F32(17.0)
    sigd = SigdState(acav=F32(0.12), afe=F32(0.5), bfe=F32(1.5), asm=F32(3.0),
                     dsl=F32(8.0), dsu=F32(9.0))
    rays = RaysState()
    rays.ifob[1, 3, 1] = 2500
    anois = AnoisState(rcnse=F32(-145.0))
    cant = _cantenna(2.0, 30.0, 2.0, 30.0, 40.0)

    built = [(state, muf, sigd, anois, rays, cant, F32(10.0), 1,
              F32(g) / RZ, F32(g), F32(0.0))
             for g in (1500.0, 3500.0, 9000.0)]
    got_zon, exp_zon, status, _ = _run_batch(built)
    assert status.tolist() == [0, 0, 0]
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")
    assert got_zon["hn"][0, 4] == F32(1.0) and got_zon["hn"][0, 5] == F32(2.0)
    assert got_zon["hn"][1, 4] == F32(2.0), "the two-hop mode belongs in slot 4"
    assert got_zon["hn"][1, 5] == F32(-1.0), "slot 5 must keep its presets"
    assert got_zon["nmode"][1, 5] == 5
    assert got_zon["tloss"][1, 5] == F32(1000.0)
    assert got_zon["hn"][2, 4] == F32(-1.0) and got_zon["hn"][2, 5] == F32(-1.0)


def test_early_return_still_presets_both_slots():
    """FS(2,K) <= 0 returns before any arithmetic; the markers must be there."""
    state = LayerState()
    state.km = 1
    state.hs[1] = F32(110.0)
    state.fs[2, 1] = F32(-1.0)                  # no Es layer -> early return
    rays = RaysState()
    rays.ifob[1, 3, 1] = 2500
    built = [(state, MufState(), SigdState(), AnoisState(), rays,
              _cantenna(2.0, 30.0, 2.0, 30.0, 40.0), F32(10.0), 1,
              F32(0.3), F32(1900.0), F32(3.0))]
    got_zon, exp_zon, status, pre_zon = _run_batch(built)
    assert status.tolist() == [0]
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")
    for s in (4, 5):
        assert got_zon["hn"][0, s] == F32(-1.0) and got_zon["nmode"][0, s] == 5
        assert got_zon["crel"][0, s] == F32(-1000.0)    # ESREG's get +1000
        assert not any(got_zon[n][0, s] == SENTINEL for n in FIELDS)
    for s in (1, 2, 3, 6, 7):
        assert got_zon["abps"][0, s] == SENTINEL, f"ESMOD touched slot {s}"
