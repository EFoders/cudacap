"""Stage A FOBBY vs. the tier-B oracle dump (Module-4 reflectrix).

Each case builds the ionogram from controlled FI/YI/HI (+IEDP), fixes a single
take-off angle (NANG=1, ANG(1)=ang), runs FOBBY, and compares the 30 oblique
operating frequencies IFOB (integer kHz).  Measured over 180 cases x 30 =
5400 cells (angles across the whole ANG range, both IEDP branches): every
value is bit-exact.  IFOB is a REAL->INT truncation of 1000*FREQ, so the only
theoretically possible divergence is a 1-kHz flip when the COS floor
([[f32-libm-noise]]) nudges FREQ across an integer boundary — not observed
here, but the bar allows it.
"""

import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.fobby import fobby
from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.raysstate import RaysState

from .dumputil import load_dump

DUMP = load_dump("fobby")


def _run_case(inp):
    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp, ang) = (
        float(x) for x in inp)
    s = LayerState(km=1)
    s.iedp = int(round(iedp))
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    rays = RaysState(nang=1)
    rays.ang[1] = F32(ang)
    fobby(s, rays, 1)
    return [int(rays.ifob[1, ih, 1]) for ih in range(1, 31)]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_fobby_dumps.sh")
def test_fobby_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 30
    total = exact = 0
    worst = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for name, e_str, g in zip(header, exp, got):
            e = int(e_str)
            d = abs(g - e)
            worst = max(worst, d)
            assert d <= 1, (f"case {k} {name}: got {g} expected {e} "
                            f"(diff {d} kHz), in {inp}")
            total += 1
            exact += d == 0
    assert exact / total >= 0.999, f"exact {exact}/{total} (worst {worst} kHz)"
