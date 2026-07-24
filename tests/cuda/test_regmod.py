"""REGMOD Stage C vs Stage A, on the regmod oracle grid.

The per-mode loss budget: the deepest transcendental chain in the program --
COS/SIN/SQRT geometry, two ``**1.98`` powers, ALOG/EXP for the collision
frequency and the E-bending term, up to six ALOG10s through PRBMUF, then seven
comparably sized terms summed into XTLOS.  Against the ORACLE that costs a ULP
budget (see tests/scalar/test_regmod.py); against Stage A it costs nothing,
because both sides evaluate every transcendental in binary64 and round once --
so the bar here is the ordinary Stage C one, bit-identity on every /ZON/ field
of every slot, including the ones REGMOD never writes.

The whole ZonState is compared, not just the outputs: slot 7 and every unused
slot must come back carrying exactly what Stage A left (the HN/HP = -1 preset
and untouched zeros), which is what pins "REGMOD wrote nothing it should not".
The read-only states are compared against their own entry copies for the same
reason.

Where Stage A raises RegmodError -- MUFday probability 0, the Fortran's STOP --
Stage C returns status 1 with /ZON/ exactly as Stage A leaves it at the raise.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.regmod import RegmodError, regmod
from voacap_gpu.scalar.zonstate import ZonState
from voacap_gpu.cuda.state import (pack_anois, pack_layer, pack_modes,
                                      pack_muf, pack_rays, pack_sigd,
                                      pack_zon)

from tests.scalar.dumputil import load_dump
from tests.scalar.test_regmod import COLUMNS
from tests.scalar.test_regmod import _build as _build_case
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("regmod")

_SRC = r'''
#include "regmod.cuh"

extern "C" __global__ void drive(const LayerState* st, const ModesState* modes,
                                 const MufState* muf, const SigdState* sigd,
                                 const AnoisState* anois,
                                 const RaysState* rays, const float* array,
                                 const float* aeff, const int* iats,
                                 const float* xfqs_all, const float* xfqe_all,
                                 const float* offazim,
                                 const float* pwrdba_all, int numants,
                                 ZonState* zon, const float* freq,
                                 const int* jmode, const float* gcd,
                                 int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int na = numants + 1;
    status[i] = regmod(&st[i], &modes[i], &muf[i], &sigd[i], &anois[i],
                       // rays_view_of(): the stride-1 view, byte-for-byte the
                       // packed struct this test has always built.
                       rays_view_of(&rays[i]),
                       array, aeff, iats, xfqs_all + i * na,
                       xfqe_all + i * na, offazim, pwrdba_all + i * na,
                       numants, &zon[i], freq[i], jmode[i], gcd[i]);
}
'''


def shared_tables(cant):
    """ARRAY / AEFF / IATS / OFFAZIM are one circuit's, so they are shared."""
    na = cant.numants + 1
    return (np.ascontiguousarray(cant.array[:, :, :na]).ravel(),
            np.ascontiguousarray(cant.aeff[:, :na]).ravel(),
            np.ascontiguousarray(cant.iats[:na]).astype(np.int32),
            np.ascontiguousarray(cant.offazim[:na]).astype(np.float32))


def _run_batch(built, expect_status=0):
    """Drive the device function once per built case; return (got, exp) zon."""
    n = len(built)
    cant0 = built[0][6]
    na = cant0.numants + 1
    p_st, p_modes, p_muf, p_sigd, p_anois, p_rays, p_zon = ([] for _ in range(7))
    e_zon = []
    xfqs = np.zeros((n, na), np.float32)
    xfqe = np.zeros((n, na), np.float32)
    pwrdba = np.zeros((n, na), np.float32)
    freq = np.zeros(n, np.float32)
    jmode = np.zeros(n, np.int32)
    gcd = np.zeros(n, np.float32)
    for i, (st, modes, muf, sigd, anois, rays, cant, f, k, g) in enumerate(built):
        xfqs[i] = cant.xfqs[:na]
        xfqe[i] = cant.xfqe[:na]
        pwrdba[i] = cant.pwrdba[:na]
        freq[i], jmode[i], gcd[i] = f, k, g
        p_st.append(pack_layer(st))
        p_modes.append(pack_modes(modes))
        p_muf.append(pack_muf(muf))
        p_sigd.append(pack_sigd(sigd))
        p_anois.append(pack_anois(anois))
        p_rays.append(pack_rays(rays))
        zon = ZonState()
        p_zon.append(pack_zon(zon))
        try:
            regmod(st, modes, muf, sigd, anois, rays, cant, zon, f, k, g)
        except RegmodError:
            assert expect_status, "an unexpected Stage A refusal"
        e_zon.append(pack_zon(zon))                 # expected POST state
    exp_zon = np.stack(e_zon)

    array, aeff, iats, offazim = shared_tables(cant0)
    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(p_st).view(np.uint8),
                       np.stack(p_modes).view(np.uint8),
                       np.stack(p_muf).view(np.uint8),
                       np.stack(p_sigd).view(np.uint8),
                       np.stack(p_anois).view(np.uint8),
                       np.stack(p_rays).view(np.uint8),
                       array, aeff, iats, xfqs.ravel(), xfqe.ravel(), offazim,
                       pwrdba.ravel(), np.int32(cant0.numants),
                       np.stack(p_zon).view(np.uint8),
                       freq, jmode, gcd, status])
    got_zon = fetch(dev[14]).view(exp_zon.dtype)
    got_status = fetch(dev[18])
    pre = {"st": np.stack(p_st), "modes": np.stack(p_modes),
           "muf": np.stack(p_muf), "sigd": np.stack(p_sigd),
           "anois": np.stack(p_anois), "rays": np.stack(p_rays)}
    got_ro = {"st": fetch(dev[0]).view(pre["st"].dtype),
              "modes": fetch(dev[1]).view(pre["modes"].dtype),
              "muf": fetch(dev[2]).view(pre["muf"].dtype),
              "sigd": fetch(dev[3]).view(pre["sigd"].dtype),
              "anois": fetch(dev[4]).view(pre["anois"].dtype),
              "rays": fetch(dev[5]).view(pre["rays"].dtype)}
    return got_zon, exp_zon, got_status, pre, got_ro


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_regmod_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    built = [_build_case(row) for row in inputs]
    n = len(built)
    got_zon, exp_zon, status, pre, got_ro = _run_batch(built)

    assert (status == 0).all(), (
        f"lanes refused: {np.flatnonzero(status).tolist()}")
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")
    for key in pre:
        for name in pre[key].dtype.names:
            assert_bitexact(got_ro[key][name], pre[key][name],
                            f"{key}.{name} (read-only)")

    # ---- branch coverage, the arms the Stage A test's grid was built for ----
    cases = [dict(zip(COLUMNS, (float(x) for x in row))) for row in inputs]
    seen = {"f_layer": 0, "de_layer": 0, "de_xnsq_floor": 0, "de_xnsq_profile":
            0, "es_obscuration": 0, "no_es_obscuration": 0, "low_mufday": 0,
            "one_hop": 0, "many_hops": 0, "eight_hops": 0, "unused_slot": 0,
            "tx_out_of_range": 0, "decile_clamped": 0, "slot7_untouched": 0}
    filled = 0
    for i, (v, b) in enumerate(zip(cases, built)):
        st, modes, muf, sigd, anois, rays, cant, freq, k, gcd = b
        ihop = int(gcd / modes.ghop + F32(0.01))
        seen["one_hop"] += ihop == 1
        seen["many_hops"] += ihop > 1
        seen["eight_hops"] += ihop >= 8
        seen["tx_out_of_range"] += not (v["txlo"] <= v["freq"] <= v["txhi"])
        seen["es_obscuration"] += st.fs[2, k] > 0.0
        seen["no_es_obscuration"] += st.fs[2, k] <= 0.0
        seen["slot7_untouched"] += exp_zon["hp"][i, 7] == F32(-1.0)
        for im in range(1, 7):
            if modes.hpmod[im, k] <= 0.0:
                seen["unused_slot"] += 1
                continue
            if st.fi[1, k] - modes.fvmod[im, k] < 0.0:
                seen["f_layer"] += 1
            else:
                seen["de_layer"] += 1
                if sigd.htloss - modes.htmod[im, k] <= 0.0:
                    seen["de_xnsq_floor"] += 1
                else:
                    seen["de_xnsq_profile"] += 1
            p = exp_zon["prob"][i, im]
            seen["low_mufday"] += bool(F32(0.0) < p < F32(0.0001))
            seen["decile_clamped"] += bool(
                exp_zon["tllow"][i, im] == F32(25.0)
                or exp_zon["tlhgh"][i, im] == F32(25.0))
        filled += int(exp_zon["hn"][i, 1] > 0 or exp_zon["hn"][i, 2] > 0)
    for arm, cnt in seen.items():
        assert cnt, f"branch never reached: {arm} ({seen})"
    assert filled > n // 2, f"only {filled} of {n} cases produced a mode"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_slot7_and_unused_modes_keep_the_preset():
    """Slot 7 is ESMOD's; an unused mode keeps HN/HP = -1, as /MODES/ marks."""
    _, inputs, _ = DUMP
    st, modes, muf, sigd, anois, rays, cant, freq, k, gcd = _build_case(
        inputs[0])
    for im in range(1, 7):
        modes.hpmod[im, k] = F32(-1.0)              # every slot unused
    got_zon, exp_zon, status, _, _ = _run_batch(
        [(st, modes, muf, sigd, anois, rays, cant, freq, k, gcd)])
    assert status.tolist() == [0]
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")
    assert (got_zon["hn"][0, 1:8] == F32(-1.0)).all()
    assert (got_zon["hp"][0, 1:8] == F32(-1.0)).all()
    assert (got_zon["tloss"][0, 1:8] == F32(0.0)).all()


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_zero_mufday_returns_status_1():
    """PRBMUF returns a hard 0 when no median is placed; the Fortran STOPs.

    Stage A raises RegmodError partway through the first mode; Stage C must
    return status 1 having written EXACTLY what Stage A wrote before raising,
    so the comparison is against Stage A's own post-exception /ZON/.
    """
    _, inputs, _ = DUMP
    refused = _build_case(inputs[0])
    for i in range(1, 5):
        refused[2].ymuf[i] = F32(0.0)       # FSET <= 0 -> a step at FGO
        refused[2].fvmuf[i] = F32(1.0)      # ... and FREQ above it -> 0
    control = _build_case(inputs[0])        # a clean lane in the same launch
    got_zon, exp_zon, status, _, _ = _run_batch([refused, control],
                                                expect_status=1)
    assert status.tolist() == [1, 0]
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")
    # the refused lane stopped exactly where Stage A raised: the presets and
    # the early per-mode writes are there, the tail is not
    assert exp_zon["hp"][0, 1] == F32(-1.0)         # never reached the tail
    assert exp_zon["fslos"][0, 1] != F32(0.0)       # but the budget started
    assert exp_zon["tloss"][0, 1] == F32(0.0)
