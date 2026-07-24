"""Stage A ALOSFV vs. the tier-B oracle dump (Module-4 deviative loss).

Each case builds the ionogram from controlled FI/YI/HI (+IEDP), injects the
per-layer MUF heights HPMUF/HTMUF, runs ALOSFV, and compares AFAC(1..30).
Measured over 160 cases x 30 = 4800 cells (both IEDP branches, F1
present/absent, 1666 cells nonzero so the collision-frequency EXP path is
exercised): AFAC is bit-exact 4800/4800.  Only EXP carries the
[[f32-libm-noise]] floor and the HPRIM-HTRUE-HM cancellation could amplify it,
so the bar allows a small residual, but none is observed here.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.alosfv import alosfv
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.mufstate import MufState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("alosfv")


def _run_case(inp):
    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp,
     hp1, hp2, hp3, ht1, ht2, ht3) = (float(x) for x in inp)
    s = LayerState(km=1)
    s.iedp = int(round(iedp))
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    m = MufState()
    m.hpmuf[1], m.hpmuf[2], m.hpmuf[3] = F32(hp1), F32(hp2), F32(hp3)
    m.htmuf[1], m.htmuf[2], m.htmuf[3] = F32(ht1), F32(ht2), F32(ht3)
    alosfv(s, m, 1)
    return [s.afac[ih, 1] for ih in range(1, 31)]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_alosfv_dumps.sh")
def test_alosfv_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 30
    total = exact = 0
    worst = 0.0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for name, e_str, g in zip(header, exp, got):
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            worst = max(worst, d)
            assert d <= 8, (f"case {k} {name}: got {g!r} expected {e!r} "
                            f"({d:.2f} floored ULP), in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.99, f"exact {exact}/{total} (worst {worst:.1f})"
