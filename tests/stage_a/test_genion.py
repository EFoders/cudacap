"""Stage A GENION vs. the tier-B oracle dump (Module-4 ionogram).

Each case injects controlled FI/YI/HI + a per-case IEDP, runs the same
LECDEN -> GENION chain the oracle driver does, and compares the 30-point
ionogram (FVERT vertical-sounding frequencies, HPRIM virtual heights, HTRUE
true heights) field-for-field.

Measured over 140 cases x 90 fields (both branches: 67 full-Gaussian, 73
fast): FVERT bit-exact 4200/4200; HTRUE 4199/4200 (worst 1 floored-ULP);
HPRIM 4198/4200, worst 19 floored-ULP (~0.15 m) — the GETHP Gaussian-
integration [[f32-libm-noise]] floor, which only shows up on the virtual
height of the lowest D-E sample frequencies (a razor-thin, poorly-conditioned
integration range) in the full-Gaussian branch.  FVERT is pure REAL*4
arithmetic over the critical frequencies (one correctly-rounded SQRT), so it
must be bit-exact.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("genion")

# FVERT is transcendental-free (bar one SQRT) -> bit-exact; the height fields
# carry at most the GETHP/BENDY/PEN libm floor (worst 19 floored-ULP on HPRIM
# at the low D-E samples).
FVERT_FIELDS = [f"fvert{i:02d}" for i in range(1, 31)]
HTRUE_BAR = 4
HPRIM_BAR = 32


def _bar(name):
    if name.startswith("fvert"):
        return 2
    if name.startswith("htrue"):
        return HTRUE_BAR
    return HPRIM_BAR


def _run_case(inp):
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp = (float(x) for x in inp)
    s = LayerState(km=1)
    s.iedp = int(round(iedp))
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    out = {}
    for i in range(1, 31):
        out[f"fvert{i:02d}"] = s.fvert[i, 1]
        out[f"hprim{i:02d}"] = s.hprim[i, 1]
        out[f"htrue{i:02d}"] = s.htrue[i, 1]
    return out


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genion_dumps.sh")
def test_genion_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 90
    total = exact = 0
    exact_by_field = {h: 0 for h in header}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for name, e_str in zip(header, exp):
            e = np.float32(e_str)
            bar = _bar(name)
            d = ulp_diff_floored(got[name], e)
            assert d <= bar, (
                f"case {k} {name}: got {got[name]!r} expected {e!r} "
                f"({d:.2f} floored ULP > {bar}), in {inp}")
            total += 1
            hit = d == 0.0
            exact += hit
            exact_by_field[name] += hit

    n = len(inputs)
    for name in FVERT_FIELDS:
        assert exact_by_field[name] == n, (
            f"{name} (a vertical-sounding frequency) must be bit-exact: "
            f"{exact_by_field[name]}/{n}")
    assert exact / total >= 0.99, f"bit-exact rate {exact}/{total}"
