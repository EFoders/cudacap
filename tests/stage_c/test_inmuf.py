"""INMUF Stage C vs Stage A, on the inmuf oracle grid.

The subtlest routine in the program, and the deepest: it rewrites /MODES/ in
place, may overwrite and then restore five /MUFS/ decile arrays, and finishes
by running REGMOD on whatever it produced -- so this test is simultaneously an
INMUF test and an end-to-end REGMOD integration test.  Three states are
compared whole (/MODES/, /MUFS/, /ZON/) plus NUMMOD and the status, and the
read-only states are compared against their entry copies.

The traps, each pinned by the compare rather than trusted:

* **MODMUF is a Fortran LAYER NUMBER**, so DELMUF(MODMUF) is an ordinary
  1-based read.  Reading it as a 0-based row sends lanes down the
  zero-distance path because DELMUF(3) can be ~90.
* **the stage-3 guard is the Fortran test NEGATED**, never the operator
  flipped: ``IF(FREQ-ESD+EPS) 240,160,160`` skips only when the value is
  NEGATIVE, and a ray that cannot reflect gives ESD = NaN, where ``< 0`` is
  false -- so the Fortran INSERTS a NaN mode.  Any NaN Stage A produces has to
  come back as the same NaN bit pattern; ``assert_bitexact`` compares float
  bit patterns, so that is exactly what is checked.  MEASURED: this grid
  produces no NaN at all, so the arm is written correctly but NOT exercised
  here -- the test pins the count at zero so a future grid that does reach it
  cannot pass by accident.
* **stage 1 does not clear the slots above INUM**, and **stage 3 mutates GHOP**
  for every mode, not just the inserted one.  Both show up as whole-/MODES/
  differences if lost.
* **/MUFS/ is restored on the way out** -- but NOT when REGMOD refuses, because
  Stage A's exception escapes before the restore runs.  The out-of-bounds lane
  below tests exactly that.

Exactly one grid case drives GAIN out of the bottom of its table (a real
Fortran defect INMUF is known to trigger, via a negative take-off angle from
the stage-3 Martyn recompute).  Stage A raises IndexError there; Stage C
returns REGMOD's status 3 with every state as Stage A left it at the raise.
INMUF's own two refusals (status 5 and 6) are asserted UNREACHABLE on this
grid rather than claimed as covered.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.inmuf import inmuf
from voacap_gpu.stage_a.zonstate import ZonState
from voacap_gpu.stage_c.state import (pack_anois, pack_layer, pack_modes,
                                      pack_muf, pack_rays, pack_sigd,
                                      pack_zon)

from tests.stage_a.dumputil import load_dump
from tests.stage_a.test_inmuf import _build as _build_case
from tests.stage_a.test_inmuf import _negative_takeoff
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)
from tests.stage_c.test_regmod import shared_tables

pytestmark = requires_cuda

DUMP = load_dump("inmuf")

_SRC = r'''
#include "inmuf.cuh"

extern "C" __global__ void drive(const LayerState* st, ModesState* modes,
                                 MufState* muf, const SigdState* sigd,
                                 const AnoisState* anois,
                                 const RaysState* rays, const float* array,
                                 const float* aeff, const int* iats,
                                 const float* xfqs_all, const float* xfqe_all,
                                 const float* offazim,
                                 const float* pwrdba_all, int numants,
                                 ZonState* zon, const float* freq,
                                 const int* jmode, const float* gcd,
                                 const int* ihop, int* nummod, int* status,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int na = numants + 1;
    status[i] = inmuf(&st[i], &modes[i], &muf[i], &sigd[i], &anois[i],
                      // rays_view_of(): the stride-1 view, byte-for-byte the
                      // packed struct this test has always built.
                      rays_view_of(&rays[i]),
                      array, aeff, iats, xfqs_all + i * na,
                      xfqe_all + i * na, offazim, pwrdba_all + i * na,
                      numants, &zon[i], freq[i], jmode[i], gcd[i], ihop[i],
                      &nummod[i]);
}
'''


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_inmuf_bit_exact_vs_stage_a():
    _, inputs, rows = DUMP
    n = len(inputs)
    built = [_build_case(inp, exp) for inp, exp in zip(inputs, rows)]
    cant0 = built[0][6]
    na = cant0.numants + 1

    p_st, p_modes, p_muf, p_sigd, p_anois, p_rays, p_zon = ([] for _ in
                                                            range(7))
    e_modes, e_muf, e_zon = [], [], []
    xfqs = np.zeros((n, na), np.float32)
    xfqe = np.zeros((n, na), np.float32)
    pwrdba = np.zeros((n, na), np.float32)
    freq = np.zeros(n, np.float32)
    jmode = np.zeros(n, np.int32)
    gcd = np.zeros(n, np.float32)
    ihop = np.zeros(n, np.int32)
    exp_nummod = np.zeros(n, np.int32)
    exp_status = np.zeros(n, np.int32)
    for i, b in enumerate(built):
        st, modes, muf, sigd, anois, rays, cant, f, k, g, hopc = b
        xfqs[i], xfqe[i], pwrdba[i] = (cant.xfqs[:na], cant.xfqe[:na],
                                       cant.pwrdba[:na])
        freq[i], jmode[i], gcd[i], ihop[i] = f, k, g, hopc
        p_st.append(pack_layer(st))
        p_modes.append(pack_modes(modes))
        p_muf.append(pack_muf(muf))
        p_sigd.append(pack_sigd(sigd))
        p_anois.append(pack_anois(anois))
        p_rays.append(pack_rays(rays))
        zon = ZonState()
        p_zon.append(pack_zon(zon))
        try:
            exp_nummod[i] = inmuf(st, modes, muf, sigd, anois, rays, cant, zon,
                                  f, k, g, hopc)
        except IndexError:
            exp_status[i] = 3               # GAIN's out-of-bounds table read
            exp_nummod[i] = -1              # Stage A loses NUMMOD to the raise
        e_modes.append(pack_modes(modes))
        e_muf.append(pack_muf(muf))
        e_zon.append(pack_zon(zon))
    exp_modes = np.stack(e_modes)
    exp_muf = np.stack(e_muf)
    exp_zon = np.stack(e_zon)
    pre = {"st": np.stack(p_st), "sigd": np.stack(p_sigd),
           "anois": np.stack(p_anois), "rays": np.stack(p_rays)}

    array, aeff, iats, offazim = shared_tables(cant0)
    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    nummod = np.full(n, -1, np.int32)
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
                       freq, jmode, gcd, ihop, nummod, status])
    got_modes = fetch(dev[1]).view(exp_modes.dtype)
    got_muf = fetch(dev[2]).view(exp_muf.dtype)
    got_zon = fetch(dev[14]).view(exp_zon.dtype)
    got_nummod = fetch(dev[19])
    got_status = fetch(dev[20])

    assert_bitexact(got_status, exp_status, "status")
    ok = exp_status == 0
    assert_bitexact(got_nummod[ok], exp_nummod[ok], "nummod")
    for name in exp_modes.dtype.names:
        assert_bitexact(got_modes[name], exp_modes[name], f"modes.{name}")
    for name in exp_muf.dtype.names:
        assert_bitexact(got_muf[name], exp_muf[name], f"muf.{name}")
    for name in exp_zon.dtype.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"zon.{name}")
    for key in pre:
        got_ro = fetch(dev[{"st": 0, "sigd": 3, "anois": 4,
                            "rays": 5}[key]]).view(pre[key].dtype)
        for name in pre[key].dtype.names:
            assert_bitexact(got_ro[name], pre[key][name],
                            f"{key}.{name} (read-only)")

    # The NaN-mode insert.  Measured on this grid: ZERO NaN cells, so the
    # pathological arm is NOT exercised here -- recorded as a fact rather than
    # claimed as coverage.  Both counts are still compared so that a future
    # grid which does reach it fails loudly instead of quietly agreeing on
    # "no NaN", and assert_bitexact above would then compare the bit patterns.
    nan_of = ("delmod", "hpmod", "htmod", "fvmod", "afmod", "ghop")
    nan_exp = sum(int(np.isnan(exp_modes[f]).sum()) for f in nan_of)
    nan_got = sum(int(np.isnan(got_modes[f]).sum()) for f in nan_of)
    assert nan_got == nan_exp, (f"NaN modes: got {nan_got} expected {nan_exp};"
                                " the stage-3 guard was flipped, not negated")
    assert nan_exp == 0, ("the grid now reaches the NaN insert; keep the "
                          "bit-pattern compare and drop this pin")

    # ---- coverage, mirroring the Stage A test's asserts --------------------
    out_of_bounds = sum(_negative_takeoff(exp) for exp in rows)
    assert out_of_bounds == 1, (
        f"{out_of_bounds} cases drive GAIN out of bounds; the grid pins this "
        f"at exactly 1 so a change in the count is noticed")
    assert int((exp_status == 3).sum()) == 1, "the refusal must be that lane"
    inserted = int((exp_nummod[ok] > 0).sum())
    assert inserted > n // 3, f"only {inserted} of {n} cases produced a mode"

    # INMUF's own two refusals: still unreachable, not merely untested
    assert not (got_status == 5).any(), "ITMOD outside 1..3 became reachable"
    assert not (got_status == 6).any(), (
        "the zero-distance /MODES/ row-7 write became reachable")

    # /MUFS/ really is put back, and the refused lane's really is not
    pre_muf = np.stack(p_muf)
    for name in ("sigl", "sigu", "yfot", "yhpf", "ymuf"):
        assert_bitexact(got_muf[name][ok], pre_muf[name][ok],
                        f"muf.{name} (restored)")
    bad = np.flatnonzero(~ok)
    changed = any(not np.array_equal(got_muf[name][bad], pre_muf[name][bad])
                  for name in ("sigl", "sigu", "yfot", "yhpf", "ymuf"))
    assert changed, ("the refused lane restored /MUFS/; Stage A's exception "
                     "escapes before the restore, so it must not")

    # the two insert paths and the zero-distance construction all appear
    modmuf = np.array([b[2].modmuf for b in built])
    delmuf_mm = np.array([float(b[2].delmuf[b[2].modmuf]) for b in built])
    nhop_mm = np.array([int(b[2].nhopmf[b[2].modmuf]) for b in built])
    seen = {
        "modmuf_is_a_layer_number": int((modmuf >= 1).sum()),
        "zero_distance": int((delmuf_mm >= 89.9).sum()),
        "hop_matches_circuit": int(((delmuf_mm < 89.9)
                                    & (ihop == nhop_mm)).sum()),
        "stage3_per_hop_muf": int(((delmuf_mm < 89.9)
                                   & (ihop != nhop_mm)).sum()),
        "ghop_mutated": int((got_modes["ghop"]
                             != np.stack(p_modes)["ghop"]).sum()),
    }
    for arm, cnt in seen.items():
        assert cnt, f"branch never reached: {arm} ({seen})"
