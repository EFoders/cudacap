"""Stage A DISTXY vs. the tier-B oracle dump.

DISTXY's real work happens inside DAZEL (float64 internals), so the
float32 interface narrowing absorbs libm noise completely: measured 100%
bit-exact on all 296 cases x 16 fields at generation.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.dazel import Azel
from voacap_gpu.stage_a.distxy import distxy
from voacap_gpu.stage_a.f32math import F32

from .dumputil import load_dump, ulp_diff

DUMP = load_dump("distxy")

AZEL_FIELDS = ["ztlat", "ztlon", "ztht", "zrlat", "zrlon", "zrht", "ztaz",
               "zraz", "ztelv", "zrelv", "zd", "zdgc", "ztakof", "zrakof"]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_mod1_dumps.sh")
def test_distxy_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["rlatd", "rlongd"] + AZEL_FIELDS
    total = 0
    exact = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        z = Azel()
        rlatd, rlongd = distxy(z, int(inp[0]), int(inp[1]), F32(inp[2]),
                               F32(inp[3]), F32(inp[4]), F32(inp[5]),
                               int(inp[6]))
        got = [rlatd, rlongd] + [getattr(z, f) for f in AZEL_FIELDS]
        for h, gv, ev in zip(header, got, exp):
            d = ulp_diff(gv, np.float32(ev))
            assert d <= 1.0, (f"case {k} {h}: got {gv!r} expected "
                              f"{np.float32(ev)!r} ({d:.2f} ULP), in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.98, f"bit-exact rate {exact}/{total}"
