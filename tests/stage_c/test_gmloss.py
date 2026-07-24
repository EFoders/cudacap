"""GMLOSS Stage C vs Stage A, on the gmloss oracle grid.

GMLOSS is a wrapper with two surviving effects -- it presets all seven /ZON/
slots to the model's "no mode" markers (GENOIS reads the block afterwards)
and it calls SETTXR -- plus a TXRGML loop whose result nothing in voacapw
ever reads.  The loop is ported anyway, because "write-only" is a claim about
the whole program, and the dump is what makes it checkable.

The comparison is whole-struct on everything SETTXR touches (/LOSX/, /REFLX/,
/allMODE/) plus /ZON/ and /LPATH/, with the read-only blocks compared against
their entry copies.  The two thresholds GMLOSS owns get their own directed
witness: a ray at EXACTLY 70 km is priced here (``IF(HPFLX-70.) 110,105,105``
-- SELTXR's ``.LE. 70`` kills the same ray), and so is a ray exactly on AMIND.
TXRGML is deliberately NOT preset, so a ray failing either test keeps whatever
the previous frequency left there; the driver's sentinel pins that.

Where SETTXR refuses, GMLOSS returns the same status with /ZON/ already
preset and the TXRGML loop not run -- exactly where Stage A's exception
leaves it.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.gmloss import gmloss
from voacap_gpu.stage_a.lpathstate import LpathState
from voacap_gpu.stage_a.settxr import ModmufError
from voacap_gpu.stage_a.zonstate import ZonState
from voacap_gpu.stage_c.state import (ALLMODE_DTYPE, LAYER_DTYPE, LOSX_DTYPE,
                                      LPATH_DTYPE, MUF_DTYPE, RAYS_DTYPE,
                                      REFLX_DTYPE, SIGD_DTYPE, ZON_DTYPE,
                                      pack_allmode, pack_layer, pack_losx,
                                      pack_lpath, pack_muf, pack_rays,
                                      pack_reflx, pack_sigd, pack_zon)

from tests.stage_a.dumputil import load_dump
from tests.stage_a.test_gmloss import COLUMNS, TXRGML_SENTINEL, ZON_FIELDS
from tests.stage_a.test_settxr import _build, _cantenna, _plain
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)
from tests.stage_c.test_settxr import unview

pytestmark = requires_cuda

DUMP = load_dump("gmloss")
NRAY = 45
ZON_MARK = F32(12345.0)

_SRC = r'''
#include "gmloss.cuh"

extern "C" __global__ void drive(const LayerState* st, const MufState* muf,
                                 const SigdState* sigd, ReflxState* reflx,
                                 LosxState* losx, const RaysState* rays,
                                 const float* array, const float* aeff,
                                 const int* iats, const float* xfqs_all,
                                 const float* xfqe_all, const float* offazim,
                                 int numants, AllModeState* allm,
                                 ZonState* zon, LpathState* lpath,
                                 const int* itxrcp, const float* freq,
                                 const float* gcdkm, const float* amind,
                                 int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int na = numants + 1;
    status[i] = gmloss(&st[i], &muf[i], &sigd[i], &reflx[i], &losx[i],
                       // rays_view_of(): the stride-1 view, byte-for-byte the
                       // packed struct this test has always built.
                       rays_view_of(&rays[i]),
                       array, aeff, iats, xfqs_all + i * na,
                       xfqe_all + i * na, offazim, numants, &allm[i],
                       &zon[i], &lpath[i], &itxrcp[i * 3], freq[i],
                       gcdkm[i], amind[i]);
}
'''

RO = (("st", LAYER_DTYPE), ("muf", MUF_DTYPE), ("sigd", SIGD_DTYPE),
      ("rays", RAYS_DTYPE))
RW = (("reflx", REFLX_DTYPE), ("losx", LOSX_DTYPE), ("allm", ALLMODE_DTYPE),
      ("zon", ZON_DTYPE), ("lpath", LPATH_DTYPE))


def _marked_zon():
    """A distinctive value in every slot GMLOSS presets, so "written" and
    "already held the marker" are different observations."""
    zon = ZonState()
    for im in range(1, 8):
        for name in ZON_FIELDS:
            getattr(zon, name)[im] = ZON_MARK
        zon.nmode[im] = -9
    return zon


def run_batch(vs):
    n = len(vs)
    cants = [_cantenna(v) for v in vs]
    states = [_build(v) for v in vs]
    na = cants[0].numants + 1

    keys = ("st", "muf", "sigd", "reflx", "losx", "rays", "allm", "zon",
            "lpath")
    pre = {k: [] for k in keys}
    exp = {k: [] for k, _ in RW}
    xfqs = np.zeros((n, na), np.float32)
    xfqe = np.zeros((n, na), np.float32)
    itxrcp = np.zeros((n, 3), np.int32)
    freq = np.zeros(n, np.float32)
    gcdkm = np.zeros(n, np.float32)
    amind = np.zeros(n, np.float32)
    exp_status = np.zeros(n, np.int32)
    for i, (v, cant) in enumerate(zip(vs, cants)):
        st, muf, sigd, reflx, losx, rays, allm, itx, _k2 = states[i]
        zon = _marked_zon()
        # DELOPT/GMIN/YMIN are SELTXR's, not GMLOSS's; carried so the whole
        # /LPATH/ struct can be compared.  Absent from a hand-built case, the
        # blkdat defaults stand.
        lpath = LpathState(delopt=F32(v.get("delopt", 3.0)),
                           gmin=F32(v.get("gmin", 3.0)),
                           ymin=F32(v.get("ymin", 0.1)))
        lpath.txrgml[1:, 1:] = TXRGML_SENTINEL
        xfqs[i], xfqe[i] = cant.xfqs[:na], cant.xfqe[:na]
        itxrcp[i] = itx
        freq[i], gcdkm[i], amind[i] = (F32(v["freq"]), F32(v["gcdkm"]),
                                       F32(v["amind"]))
        packed = {"st": pack_layer(st), "muf": pack_muf(muf),
                  "sigd": pack_sigd(sigd), "reflx": pack_reflx(reflx),
                  "losx": pack_losx(losx), "rays": pack_rays(rays),
                  "allm": pack_allmode(allm), "zon": pack_zon(zon),
                  "lpath": pack_lpath(lpath)}
        for k in keys:
            pre[k].append(packed[k])
        try:
            gmloss(st, muf, sigd, reflx, losx, rays, cant, allm, zon, lpath,
                   itx, freq[i], gcdkm[i], amind[i])
        except ModmufError as e:
            exp_status[i] = 1 if "DELMUF" in str(e) else 2
        except IndexError:
            exp_status[i] = 3
        except NotImplementedError:
            exp_status[i] = 4
        post = {"reflx": pack_reflx(reflx), "losx": pack_losx(losx),
                "allm": pack_allmode(allm), "zon": pack_zon(zon),
                "lpath": pack_lpath(lpath)}
        for k, _ in RW:
            exp[k].append(post[k])
    pre = {k: np.stack(v) for k, v in pre.items()}
    exp = {k: np.stack(v) for k, v in exp.items()}

    c0 = cants[0]
    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [pre["st"].view(np.uint8), pre["muf"].view(np.uint8),
                       pre["sigd"].view(np.uint8),
                       pre["reflx"].view(np.uint8),
                       pre["losx"].view(np.uint8), pre["rays"].view(np.uint8),
                       np.ascontiguousarray(c0.array[:, :, :na]).ravel(),
                       np.ascontiguousarray(c0.aeff[:, :na]).ravel(),
                       np.ascontiguousarray(c0.iats[:na]).astype(np.int32),
                       xfqs.ravel(), xfqe.ravel(),
                       np.ascontiguousarray(c0.offazim[:na]),
                       np.int32(c0.numants), pre["allm"].view(np.uint8),
                       pre["zon"].view(np.uint8),
                       pre["lpath"].view(np.uint8),
                       itxrcp, freq, gcdkm, amind, status])
    slot = {"st": 0, "muf": 1, "sigd": 2, "reflx": 3, "losx": 4, "rays": 5,
            "allm": 13, "zon": 14, "lpath": 15}
    got = {k: unview(fetch(dev[slot[k]]), dt)
           for k, dt in RO + RW}
    return got, exp, pre, fetch(dev[20]), exp_status


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
def test_gmloss_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    vs = [dict(zip(COLUMNS, (float(x) for x in inp))) for inp in inputs]
    got, exp, pre, status, exp_status = run_batch(vs)

    assert_bitexact(status, exp_status, "gmloss.status")
    assert (status == 0).all(), (
        f"lanes refused: {np.flatnonzero(status).tolist()}")
    assert_all_bitexact(got, exp, pre)

    # ---- branch coverage ------------------------------------------------
    # The first five arms are the Stage A test's, counted on the ENTRY ray
    # table across all 45 rays (a statement about the case grid); the rest
    # are GMLOSS's own loop, which runs over 1..IAFTXR of the table SETTXR
    # has already rewritten.
    seen = dict.fromkeys(
        ["gm_hp_under70", "gm_hp_exactly70", "gm_hp_over70", "gm_del_under",
         "gm_del_at_or_over", "txrgml_written", "txrgml_skipped_hp",
         "txrgml_skipped_del", "txrgml_sentinel_survives"], 0)
    hp0, de0 = pre["reflx"]["hpflx"], pre["reflx"]["delfx"]
    hp1, de1 = exp["reflx"]["hpflx"], exp["reflx"]["delfx"]
    naf = exp["reflx"]["iaftxr"]
    txr = exp["lpath"]["txrgml"]
    for i, v in enumerate(vs):
        amind = F32(v["amind"])
        for jj, k in ((1, 1), (2, int(v["itxrcp2"]))):
            for ia in range(1, NRAY + 1):
                hp, de = hp0[i][ia, k], de0[i][ia, k]
                seen["gm_hp_under70"] += hp < F32(70.0)
                seen["gm_hp_exactly70"] += hp == F32(70.0)
                seen["gm_hp_over70"] += hp > F32(70.0)
                seen["gm_del_under"] += de < amind
                seen["gm_del_at_or_over"] += de >= amind
                if ia > naf[i][k]:
                    seen["txrgml_sentinel_survives"] += (
                        txr[i][ia, jj] == TXRGML_SENTINEL)
                    continue
                if hp1[i][ia, k] - F32(70.0) < 0.0:
                    seen["txrgml_skipped_hp"] += 1
                elif de1[i][ia, k] - amind < 0.0:
                    seen["txrgml_skipped_del"] += 1
                else:
                    seen["txrgml_written"] += 1
    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"


def test_the_thresholds_gmloss_owns():
    """A ray at EXACTLY 70 km is priced, and so is one exactly on AMIND;
    a ray failing either test keeps whatever TXRGML already held."""
    ramp = {"hpflx0_1": 70.0, "dhpflx_1": 0.0, "hpflx0_2": 70.0,
            "dhpflx_2": 0.0, "delfx0_1": 3.0, "ddelfx_1": 0.0,
            "delfx0_2": 3.0, "ddelfx_2": 0.0, "gdflx0_1": 4000.0,
            "dgdflx_1": 0.0, "gdflx0_2": 4000.0, "dgdflx_2": 0.0,
            "nfill1": 3, "nfill2": 3, "amind": 3.0}
    below = dict(ramp, hpflx0_1=69.9, hpflx0_2=69.9)
    under = dict(ramp, delfx0_1=2.9, delfx0_2=2.9)
    vs = [_plain(delopt=3.0, gmin=3.0, ymin=0.1, **ramp),
          _plain(delopt=3.0, gmin=3.0, ymin=0.1, **below),
          _plain(delopt=3.0, gmin=3.0, ymin=0.1, **under)]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert_bitexact(status, exp_status, "gmloss.status")
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="threshold ")

    txr = got["lpath"]["txrgml"]
    for jj in (1, 2):
        for ia in (1, 2, 3):
            assert txr[0][ia, jj] != TXRGML_SENTINEL, "exactly 70 km LIVES"
        # In the two failing tables NO ray clears SETTXR's existence scan, so
        # the over-the-MUF fallback rewrites ray 1 (320 km, 6 deg) and GMLOSS
        # prices that; rays 2 and 3 are the ones that stay under the bar.
        for ia in (2, 3):
            assert txr[1][ia, jj] == TXRGML_SENTINEL, "69.9 km does not"
            assert txr[2][ia, jj] == TXRGML_SENTINEL, "just under AMIND"
        assert txr[1][1, jj] != TXRGML_SENTINEL, "the over-the-MUF ray"
        assert txr[2][1, jj] != TXRGML_SENTINEL
        # rays past IAFTXR are never even looked at
        assert txr[0][4, jj] == TXRGML_SENTINEL, "TXRGML is not preset"
        assert txr[0][NRAY, jj] == TXRGML_SENTINEL


def test_gmloss_presets_every_zon_slot():
    """The long path never fills /ZON/ with modes -- it works in /allMODE/ --
    but GENOIS runs afterwards and reads the block, so the markers matter."""
    got, exp, pre, status, exp_status = run_batch([_plain()])
    assert (status == 0).all()
    assert_all_bitexact(got, exp, pre, tag="preset ")
    zon = got["zon"][0]
    for im in range(1, 8):
        assert zon["tloss"][im] == F32(1000.0), "a loss that was not computed"
        assert zon["obf"][im] == F32(1000.0)
        assert zon["prob"][im] == F32(0.001), "a probability"
        assert zon["spro"][im] == F32(0.001)
        assert zon["sigpow"][im] == F32(-1000.0), "a level"
        assert zon["sn"][im] == F32(-1000.0)
        assert zon["hp"][im] == F32(-1.0), "a height"
        assert zon["timed"][im] == F32(-1.0), "a delay"
        assert zon["eff"][im] == F32(0.0)
        assert zon["nmode"][im] == 5, "LAYTYP(5) = ' N', no layer"
    # slot 0 is 1-based padding: the preset loop starts at 1, so it is still
    # the freshly constructed zero (the marker loop skips it too)
    assert zon["tloss"][0] == F32(0.0) and zon["nmode"][0] == 0


def test_a_settxr_refusal_stops_before_the_txrgml_loop():
    """MODMUF outside 1..4 makes SETTXR refuse; GMLOSS passes the status
    straight out with /ZON/ already preset and TXRGML untouched."""
    vs = [_plain(modmuf=0, nfill1=0, nfill2=0),
          _plain(modmuf=0, hpflx0_1=100.0, hpflx0_2=100.0),
          _plain()]
    got, exp, pre, status, exp_status = run_batch(vs)
    assert exp_status.tolist() == [1, 2, 0]
    assert_bitexact(status, exp_status, "gmloss.status")
    assert_all_bitexact(got, exp, pre, tag="refused ")
    for i in (0, 1):
        assert got["zon"][i]["tloss"][1] == F32(1000.0), "/ZON/ still preset"
        assert (got["lpath"][i]["txrgml"][1:, 1:] == TXRGML_SENTINEL).all()
