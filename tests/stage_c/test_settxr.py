"""SETTXR Stage C vs Stage A, on the settxr oracle grid.

The widest routine in the port: all 45 rays at each of two control areas,
523 output columns per case against the oracle.  Against Stage A the bar is
the ordinary Stage C one -- bit-identity on every field of every state, so
the comparison is whole-struct rather than column-by-column: /LOSX/ (including
EFFLP and the two dead columns), /REFLX/ (whose ray 1 SETTXR OVERWRITES with
the over-the-MUF fallback, so the ray table it reads is also an output), and
/allMODE/ PROB.  /RON//GEOG/, /MUFS/, /SIGD/ and /RAYS/ are read-only and are
compared against their own entry copies to prove nothing else moved.

The Stage A findings all get their own witness here, driven through the
device: AOFX is the one /LOSX/ column not preset per ray, the fallback
clobbers ray 1, the existence scan gives up at the FIRST dead ray, EFFLP is
the receiver end only, PROB is the same number for every ray, and GML/FHP are
marked unset across all 45 rays regardless of IAFTXR.

Where Stage A raises, Stage C returns a status with the state exactly as
Stage A leaves it at the raise: 1 and 2 for the two MODMUF sites (the
over-the-MUF fallback's DELMUF read and the pricing loop's YMUF read, which
are reachable independently), and GAIN's 3 and 4 propagated as REGMOD does.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.settxr import ModmufError, settxr
from voacap_gpu.stage_c.state import (ALLMODE_DTYPE, LAYER_DTYPE, LOSX_DTYPE,
                                      MUF_DTYPE, RAYS_DTYPE, REFLX_DTYPE,
                                      SIGD_DTYPE, pack_allmode, pack_layer,
                                      pack_losx, pack_muf, pack_rays,
                                      pack_reflx, pack_sigd)

from tests.stage_a.dumputil import load_dump
from tests.stage_a.test_settxr import (AOFX_SENTINEL, COLUMNS,
                                       EFFLP_SENTINEL, NO_GAIN, _build,
                                       _cantenna, _plain)
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("settxr")
NRAY = 45

_SRC = r'''
#include "settxr.cuh"

extern "C" __global__ void drive(const LayerState* st, const MufState* muf,
                                 const SigdState* sigd, ReflxState* reflx,
                                 LosxState* losx, const RaysState* rays,
                                 const float* array, const float* aeff,
                                 const int* iats, const float* xfqs_all,
                                 const float* xfqe_all, const float* offazim,
                                 int numants, AllModeState* allm,
                                 const int* itxrcp, const float* freq,
                                 const float* gcdkm, const float* amind,
                                 int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int na = numants + 1;
    status[i] = settxr(&st[i], &muf[i], &sigd[i], &reflx[i], &losx[i],
                       // rays_view_of(): the stride-1 view, byte-for-byte the
                       // packed struct this test has always built.
                       rays_view_of(&rays[i]),
                       array, aeff, iats, xfqs_all + i * na,
                       xfqe_all + i * na, offazim, numants, &allm[i],
                       &itxrcp[i * 3], freq[i], gcdkm[i], amind[i]);
}
'''


def unview(raw, dtype):
    """A flat uint8 buffer back to an (n,) structured array."""
    return raw.reshape(-1, dtype.itemsize).view(dtype).reshape(-1)


def _stage_a_status(fn):
    """Run a Stage A call, mapping its refusals onto the Stage C status."""
    try:
        fn()
    except ModmufError as e:            # ... a subclass of IndexError
        return 1 if "DELMUF" in str(e) else 2
    except IndexError:
        return 3                        # GAIN's out-of-bounds table read
    except NotImplementedError:
        return 4                        # GAIN's area-coverage antenna
    return 0


def run_batch(vs, mutate=None):
    """Drive SETTXR once per case; return got/expected/entry state."""
    n = len(vs)
    cants = []
    states = []
    for v in vs:
        cant = _cantenna(v)
        if mutate is not None:
            mutate(cant)
        cants.append(cant)
        states.append(_build(v))
    na = cants[0].numants + 1

    pre = {k: [] for k in ("st", "muf", "sigd", "reflx", "losx", "rays",
                           "allm")}
    exp = {k: [] for k in ("reflx", "losx", "allm")}
    xfqs = np.zeros((n, na), np.float32)
    xfqe = np.zeros((n, na), np.float32)
    itxrcp = np.zeros((n, 3), np.int32)
    freq = np.zeros(n, np.float32)
    gcdkm = np.zeros(n, np.float32)
    amind = np.zeros(n, np.float32)
    exp_status = np.zeros(n, np.int32)
    for i, (v, cant) in enumerate(zip(vs, cants)):
        st, muf, sigd, reflx, losx, rays, allm, itx, _k2 = states[i]
        xfqs[i], xfqe[i] = cant.xfqs[:na], cant.xfqe[:na]
        itxrcp[i] = itx
        freq[i], gcdkm[i], amind[i] = (F32(v["freq"]), F32(v["gcdkm"]),
                                       F32(v["amind"]))
        pre["st"].append(pack_layer(st))
        pre["muf"].append(pack_muf(muf))
        pre["sigd"].append(pack_sigd(sigd))
        pre["reflx"].append(pack_reflx(reflx))
        pre["losx"].append(pack_losx(losx))
        pre["rays"].append(pack_rays(rays))
        pre["allm"].append(pack_allmode(allm))
        exp_status[i] = _stage_a_status(
            lambda: settxr(st, muf, sigd, reflx, losx, rays, cant, allm, itx,
                           freq[i], gcdkm[i], amind[i]))
        exp["reflx"].append(pack_reflx(reflx))      # expected POST state
        exp["losx"].append(pack_losx(losx))
        exp["allm"].append(pack_allmode(allm))
    pre = {k: np.stack(v) for k, v in pre.items()}
    exp = {k: np.stack(v) for k, v in exp.items()}

    c0 = cants[0]
    array = np.ascontiguousarray(c0.array[:, :, :na]).ravel()
    aeff = np.ascontiguousarray(c0.aeff[:, :na]).ravel()
    iats = np.ascontiguousarray(c0.iats[:na]).astype(np.int32)
    offazim = np.ascontiguousarray(c0.offazim[:na]).astype(np.float32)

    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [pre["st"].view(np.uint8), pre["muf"].view(np.uint8),
                       pre["sigd"].view(np.uint8),
                       pre["reflx"].view(np.uint8),
                       pre["losx"].view(np.uint8), pre["rays"].view(np.uint8),
                       array, aeff, iats, xfqs.ravel(), xfqe.ravel(), offazim,
                       np.int32(c0.numants), pre["allm"].view(np.uint8),
                       itxrcp, freq, gcdkm, amind, status])
    got = {"st": unview(fetch(dev[0]), LAYER_DTYPE),
           "muf": unview(fetch(dev[1]), MUF_DTYPE),
           "sigd": unview(fetch(dev[2]), SIGD_DTYPE),
           "reflx": unview(fetch(dev[3]), REFLX_DTYPE),
           "losx": unview(fetch(dev[4]), LOSX_DTYPE),
           "rays": unview(fetch(dev[5]), RAYS_DTYPE),
           "allm": unview(fetch(dev[13]), ALLMODE_DTYPE)}
    return got, exp, pre, fetch(dev[18]), exp_status


def assert_all_bitexact(got, exp, pre, tag=""):
    for key, dt in (("reflx", REFLX_DTYPE), ("losx", LOSX_DTYPE),
                    ("allm", ALLMODE_DTYPE)):
        for name in dt.names:
            assert_bitexact(got[key][name], exp[key][name],
                            f"{tag}{key}.{name}")
    for key, dt in (("st", LAYER_DTYPE), ("muf", MUF_DTYPE),
                    ("sigd", SIGD_DTYPE), ("rays", RAYS_DTYPE)):
        for name in dt.names:
            assert_bitexact(got[key][name], pre[key][name],
                            f"{tag}{key}.{name} (read-only)")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_settxr_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    vs = [dict(zip(COLUMNS, (float(x) for x in inp))) for inp in inputs]
    n = len(vs)
    got, exp, pre, status, exp_status = run_batch(vs)

    assert_bitexact(status, exp_status, "settxr.status")
    assert (status == 0).all(), (
        f"lanes refused: {np.flatnonzero(status).tolist()}")
    assert_all_bitexact(got, exp, pre)

    # ---- branch coverage, exactly the arms the Stage A test counts --------
    seen = dict.fromkeys(
        ["scan_stops_at_dead_ray", "scan_skips_short_hop",
         "scan_finds_a_mode", "scan_exhausts_45_rays", "fallback_fired",
         "fallback_skipped", "price_skipped", "price_ran",
         "aofx_sentinel_survives", "gain_matched", "gain_no_antenna_match",
         "es_obscuration", "no_es_obscuration", "f_layer", "de_layer",
         "de_xnsq_floor", "de_xnsq_profile", "km_1", "km_3", "km_5",
         "modmuf_1", "modmuf_2", "modmuf_3", "modmuf_4",
         "itxrcp2_2", "itxrcp2_3"], 0)
    hp0, de0, gd0 = (pre["reflx"]["hpflx"], pre["reflx"]["delfx"],
                     pre["reflx"]["gdflx"])
    fi0, htl0 = pre["st"]["fi"], pre["sigd"]["htloss"]
    for i, v in enumerate(vs):
        k2 = int(v["itxrcp2"])
        dend = min(F32(4000.0), F32(v["gcdkm"]))
        amind = F32(v["amind"])
        for k in (1, k2):
            found = False
            exhausted = True
            for ia in range(1, NRAY + 1):
                if hp0[i][ia, k] - F32(70.0) < 0.0:
                    seen["scan_stops_at_dead_ray"] += 1
                    exhausted = False
                    break
                if dend / gd0[i][ia, k] - F32(0.9) < 0.0:
                    seen["scan_skips_short_hop"] += 1
                    continue
                if de0[i][ia, k] - amind >= 0.0:
                    seen["scan_finds_a_mode"] += 1
                    found = True
                    exhausted = False
                    break
            if exhausted:
                seen["scan_exhausts_45_rays"] += 1
            seen["fallback_fired" if not found else "fallback_skipped"] += 1
            if not found:
                assert exp["reflx"]["hpflx"][i][1, k] == F32(v["hpmuf"]), (
                    "ray 1 overwritten by the over-the-MUF fallback")
            for ia in range(1, NRAY + 1):
                if exp["losx"]["tgainx"][i][ia, k] == NO_GAIN:
                    seen["price_skipped"] += 1
                    if exp["losx"]["aofx"][i][ia, k] == AOFX_SENTINEL:
                        seen["aofx_sentinel_survives"] += 1
                    continue
                seen["price_ran"] += 1
                seen["gain_no_antenna_match" if
                     exp["losx"]["tgainx"][i][ia, k] == F32(0.0)
                     else "gain_matched"] += 1
                seen["no_es_obscuration" if
                     exp["losx"]["aofx"][i][ia, k] == F32(0.0)
                     else "es_obscuration"] += 1
                # which layer branch the ray took, on the POST ray table (the
                # fallback may have rewritten ray 1 before pricing)
                fv = exp["reflx"]["fvflx"][i][ia, k]
                ht = exp["reflx"]["htflx"][i][ia, k]
                if fi0[i][1, k] < fv:
                    seen["f_layer"] += 1
                else:
                    seen["de_layer"] += 1
                    seen["de_xnsq_floor" if htl0[i] <= ht
                         else "de_xnsq_profile"] += 1
        seen[f"km_{int(v['km'])}"] += 1
        seen[f"modmuf_{int(v['modmuf'])}"] += 1
        seen[f"itxrcp2_{k2}"] += 1
    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"
    assert n == len(vs)


def test_the_stage_a_findings_survive_the_port():
    """One launch carrying the six directed cases the Stage A test builds."""
    vs = [_plain(nfill1=3, nfill2=3),                       # 0: AOFX preset
          _plain(nfill1=0, nfill2=0),                       # 1: fallback
          _plain(hpflx0_1=100.0, hpflx0_2=100.0),           # 2: ray 1 kept
          _plain(hpflx0_1=10.0, dhpflx_1=40.0, nfill1=12),  # 3: first dead ray
          _plain(rxlo=20.0, rxhi=30.0),                     # 4: no RX antenna
          _plain(nfill1=1, nfill2=1),                       # 5: one ray priced
          _plain(nfill1=12, nfill2=12)]                     # 6: twelve rays
    got, exp, pre, status, exp_status = run_batch(vs)
    assert_bitexact(status, exp_status, "settxr.status")
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="directed ")

    losx, reflx, allm = got["losx"], got["reflx"], got["allm"]
    # AOFX is the one /LOSX/ column SETTXR does not preset
    for ia in (4, 20, 45):
        assert losx["tgainx"][0][ia, 1] == NO_GAIN
        assert losx["andvx"][0][ia, 1] == F32(1000.0)
        assert losx["advx"][0][ia, 1] == F32(1000.0)
        assert losx["grlosx"][0][ia, 1] == F32(0.0)
        assert losx["aofx"][0][ia, 1] == AOFX_SENTINEL, "untouched"
    # the over-the-MUF fallback overwrites ray 1 of /REFLX/
    assert reflx["delfx"][1][1, 1] == F32(6.0)
    assert reflx["hpflx"][1][1, 1] == F32(320.0)
    assert reflx["htflx"][1][1, 1] == F32(280.0)
    assert reflx["fvflx"][1][1, 1] == F32(6.5)
    assert reflx["afflx"][1][1, 1] == F32(1.1)
    assert reflx["gdflx"][1][1, 1] == F32(15000.0) / F32(4.0)
    assert reflx["imode"][1][1, 1] == 3
    assert reflx["hpflx"][2][1, 1] == F32(100.0), "ray 1's own virtual height"
    # the existence scan gives up at the FIRST dead ray, not the first live one
    assert reflx["hpflx"][3][1, 1] == F32(320.0), "ray 1 at 10 km -> fallback"
    # EFFLP is the receiver end only, and 0 when no RX antenna covers FREQ
    assert losx["efflp"][6][1] != EFFLP_SENTINEL
    assert losx["efflp"][4][1] == F32(0.0)
    # ... and case 2's ray 1 clears 70 km but not AMIND, so it is never
    # priced and EFFLP keeps the driver's sentinel
    assert losx["efflp"][2][1] == EFFLP_SENTINEL
    # PROB is the same number for every ray
    assert allm["prob"][5][1] == allm["prob"][6][1]
    assert allm["prob"][5][2] == allm["prob"][6][2]
    # GML and FHP are marked unset across all 45 rays, not just priced ones
    for k in (1, 3):
        for ia in (1, 23, 45):
            assert reflx["gml"][0][ia, k] == F32(-999.0)
            assert reflx["fhp"][0][ia, k] == F32(999.0)


def test_a_missing_controlling_layer_is_refused_at_both_sites():
    """MODMUF indexes DELMUF in the fallback and YMUF in the pricing loop,
    and the two are reachable independently: with no ray passing the
    existence tests the fallback refuses first (status 1), with a live ray
    the fallback is skipped entirely and the refusal comes from the PROS
    call instead (status 2)."""
    vs = []
    for modmuf in (0, 5, -1):
        vs.append(_plain(modmuf=modmuf, nfill1=0, nfill2=0))
        vs.append(_plain(modmuf=modmuf, hpflx0_1=100.0, hpflx0_2=100.0))
    vs.append(_plain())                     # a clean lane in the same launch
    got, exp, pre, status, exp_status = run_batch(vs)
    assert exp_status.tolist() == [1, 2, 1, 2, 1, 2, 0], (
        "the two MODMUF sites are not being reached independently")
    assert_bitexact(status, exp_status, "settxr.status")
    assert_all_bitexact(got, exp, pre, tag="modmuf ")
    # the status-2 lane got as far as pricing its first live ray (ray 2 --
    # ray 1 clears 70 km but not AMIND) before refusing at the PROS call
    assert got["losx"]["grlosx"][1][2, 1] != F32(0.0)
    assert got["losx"]["tgainx"][1][2, 1] != NO_GAIN, "the gain was assigned"
    assert got["losx"]["andvx"][1][2, 1] != F32(1000.0), "absorption, no XLS"
    # the status-1 lane refused before pricing, but did mark GML/FHP unset
    assert got["reflx"]["gml"][0][1, 1] == F32(-999.0)
    assert got["reflx"]["fhp"][0][1, 1] == F32(999.0)


def test_gain_refusals_propagate_as_status_3_and_4():
    """GAIN's missing bottom clamp (Stage A: IndexError) and its unported
    area-coverage antenna (Stage A: NotImplementedError) reach SETTXR through
    the antenna-gain call, and must arrive as 3 and 4 exactly as REGMOD's."""
    below = [_plain(freq=0.5, txlo=0.0, rxlo=0.0), _plain()]
    got, exp, pre, status, exp_status = run_batch(below)
    assert exp_status.tolist() == [3, 0], "Stage A did not raise IndexError"
    assert_bitexact(status, exp_status, "settxr.status")
    assert_all_bitexact(got, exp, pre, tag="oob ")
    assert got["losx"]["tgainx"][0][1, 1] == NO_GAIN, "TGAINX kept its preset"

    def area(cant):
        cant.offazim[2] = F32(-999.0)       # entry 2 is the receiver

    got, exp, pre, status, exp_status = run_batch([_plain()], mutate=area)
    assert exp_status.tolist() == [4]
    assert_bitexact(status, exp_status, "settxr.status")
    assert_all_bitexact(got, exp, pre, tag="area ")
    # the transmitter end priced normally before the receiver end refused
    assert got["losx"]["tgainx"][0][1, 1] != NO_GAIN
    assert got["losx"]["efflp"][0][1] == EFFLP_SENTINEL, "never reached"
