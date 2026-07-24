"""Stage A FINDF vs the tier-B oracle dump (area coverage / ray-set search).

Each case builds the ionogram (LECDEN+GENION) and a synthetic AFAC ramp, then
injects the oracle's own reflectrix IFOB (dumped in ifob.csv) and ray count
NANG rather than recomputing them with the Python FOBBY.  That is deliberate:
FOBBY's ``RCOSD = RZ*COS`` carries the [[f32-libm-noise]] floor, and because
IFOB is quantised to integer kHz a 1-ULP COS difference can flip a value by
1 kHz, which FINDF then AMPLIFIES near a cusp (a small SLOPD makes the
interpolation SLOPE swing) into a ~0.1% ground-distance difference.  That is an
inherent sensitivity of the reflectrix design, already covered by FOBBY's own
tier-B; injecting bit-identical IFOB isolates FINDF's own logic.  The residual
is then only the Martyn COS/ASIN and PENANG ACOS floors on the ground distance
(plus GENION's tiny HTRUE/HPRIM floor for IEDP<0), all small.

FINDF fills the /REFLX/ ray table for area 1; the test compares the skip/max
scalars + IAFTXR and each ray's DELFX/GDFLX/HPFLX/HTFLX/FVFLX/AFFLX/IMODE (only
slots 1..IAFTXR -- higher slots are left stale/zero).

Measured over 170 cases = 31033 cells: ALL bit-exact, which pins the GO TO
state machine (layer/angle/cusp dispatch, the reflectrix search and its
interpolation, and the Martyn correction) branch for branch.  The bar still
allows the COS/ASIN/ACOS floor for safety, but none is observed.
"""

import csv

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.findf import findf
from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.raysstate import RaysState
from voacap_gpu.scalar.reflxstate import ReflxState

from .dumputil import GOLDEN, load_dump, ulp_diff_floored

DUMP = load_dump("findf")

_SCALARS = ["iaftxr", "dskpkm", "delskp", "hpskp", "htskp", "fvskp", "iskp",
            "dmaxkm"]
_ARRAYS = ["delfx", "gdflx", "hpflx", "htflx", "fvflx", "afflx", "imode"]


def _load_ifob():
    p = GOLDEN / "findf" / "ifob.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return [[int(x) for x in row] for row in csv.reader(f)]


IFOB_ROWS = _load_ifob()


def _run_case(inp, ifob_row):
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (float(x) for x in inp[0:9])
    iedp = int(inp[9])
    amind, freq = float(inp[11]), float(inp[12])
    s = LayerState(km=1)
    s.iedp = iedp
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    for ih in range(1, 31):
        s.afac[ih, 1] = F32(0.1) + F32(0.03) * F32(ih)
    r = RaysState()
    r.nang = ifob_row[0]                       # oracle NANG
    flat = ifob_row[1:]                        # 40x30 IFOB, IA outer, IH inner
    for ia in range(1, 41):
        for ih in range(1, 31):
            r.ifob[ia, ih, 1] = flat[(ia - 1) * 30 + (ih - 1)]
    reflx = ReflxState()
    findf(s, reflx, r, 1, freq, amind)
    return reflx


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_findf_dumps.sh")
def test_findf_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 8 + 7 * 45
    total = exact = 0
    worst = 0.0
    worst_where = ""
    for kk, (inp, exp, ifob_row) in enumerate(zip(inputs, rows, IFOB_ROWS)):
        reflx = _run_case(inp, ifob_row)
        scal = [reflx.iaftxr[1], reflx.dskpkm[1], reflx.delskp[1],
                reflx.hpskp[1], reflx.htskp[1], reflx.fvskp[1], reflx.iskp[1],
                reflx.dmaxkm[1]]
        iaftxr_exp = int(round(float(exp[0])))
        for i, name in enumerate(_SCALARS):
            e = np.float32(exp[i])
            d = ulp_diff_floored(scal[i], e)
            if d > worst:
                worst, worst_where = d, f"case {kk} {name}: {scal[i]!r} vs {e!r}"
            assert d <= 8, (f"case {kk} {name}: got {scal[i]!r} expected {e!r} "
                            f"({d:.2f} floored ULP) in {inp}")
            total += 1
            exact += d == 0.0
        arrays = [reflx.delfx, reflx.gdflx, reflx.hpflx, reflx.htflx,
                  reflx.fvflx, reflx.afflx, reflx.imode]
        for b, arr in enumerate(arrays):
            base = 8 + b * 45
            for j in range(1, iaftxr_exp + 1):
                e = np.float32(exp[base + j - 1])
                d = ulp_diff_floored(arr[j, 1], e)
                if d > worst:
                    worst = d
                    worst_where = (f"case {kk} {_ARRAYS[b]}[{j}]: "
                                   f"{arr[j, 1]!r} vs {e!r}")
                assert d <= 8, (f"case {kk} {_ARRAYS[b]}[{j}]: got {arr[j, 1]!r}"
                                f" expected {e!r} ({d:.2f} floored ULP)")
                total += 1
                exact += d == 0.0
    assert exact / total >= 0.99, (f"exact {exact}/{total} "
                                   f"(worst {worst:.2f} @ {worst_where})")
