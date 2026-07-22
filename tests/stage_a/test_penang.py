"""Stage A PENANG vs the tier-B oracle dump (per-layer penetration angle).

Each case builds the ionogram from controlled FI/YI/HI (+IEDP), presets
DSKPKM/DMAXKM as FINDF does, sets FREQ, and runs PENANG, comparing DELPEN(1..3)
and the DSKPKM/DMAXKM skip/max flags.  The reflection cosine CDEL is built from
correctly-rounded ops (/, *, sqrt) so it is bit-identical to the oracle; the
only residual would be ACOS's ~1-ULP [[f32-libm-noise]] floor on DELPEN (plus,
for IEDP<0, GENION's tiny HTRUE integration floor).  DSKPKM/DMAXKM are exact
(the 10000/0 preset, or -1 when the F2 nose fully penetrates).

Measured over 200 cases = 1000 cells: all bit-exact -- ACOS32 matches this
gfortran build's runtime acosf on every input (CDEL stays <= .999999, away from
the ill-conditioned pole).  The bar still allows the ACOS floor for safety.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_a.penang import penang
from voacap_gpu.stage_a.reflxstate import ReflxState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("penang")


def _run_case(inp):
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (float(x) for x in inp[0:9])
    iedp = int(inp[9])
    freq = float(inp[10])
    s = LayerState(km=1)
    s.iedp = iedp
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    r = ReflxState()
    r.dskpkm[1] = F32(10000.0)
    r.dmaxkm[1] = F32(0.0)
    penang(s, r, 1, freq)
    return [r.delpen[1, 1], r.delpen[2, 1], r.delpen[3, 1],
            r.dskpkm[1], r.dmaxkm[1]]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_penang_dumps.sh")
def test_penang_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["delpen1", "delpen2", "delpen3", "dskpkm", "dmaxkm"]
    total = exact = 0
    worst = 0.0
    worst_where = ""
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for name, g, e_str in zip(header, got, exp):
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst:
                worst, worst_where = d, f"case {k} {name}: {g!r} vs {e!r}"
            assert d <= 4, (f"case {k} {name}: got {g!r} expected {e!r} "
                            f"({d:.2f} floored ULP) in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.98, (f"exact {exact}/{total} "
                                   f"(worst {worst:.2f} @ {worst_where})")
