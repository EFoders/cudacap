"""Stage A profile/bending primitives vs. the tier-B oracle dump.

Each case feeds controlled FI/YI/HI (no coefficient chain) through the
oracle's own LECDEN + GETHP + BENDY + PEN, isolating the primitives from
the VERSY libm floor.  Measured at generation (148 cases): the full 50-point
HTR height grid and FNSQ plasma-frequency profile are bit-exact (7400/7400
each), and the GETHP/BENDY/PEN probes are within 1 ULP.

FSECV is only *assigned* by LECDEN when an F1 layer is present (it is the
valley base frequency, or -1 for a linear F1); with no F1 the Fortran never
touches it, so the dumped value is stale COMMON state from the previous
case — undefined output, compared here only when FI(2) > 0.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.bendy import bendy
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.gethp import gethp
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.pen import pen

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("profile")

PROBE = ["hp1", "ht1", "hp2", "ht2", "hp3", "ht3", "hp4", "ht4",
         "bendy", "pen"]


def _bar(name):
    if name.startswith("htr"):
        return 2       # pure arithmetic grid — exact
    if name.startswith("fnsq"):
        return 8       # SQRT-exact; D-region EXP may wobble 1 ULP
    if name in ("hi2", "yi2"):
        return 4
    if name == "fsecv":
        return 8
    return 8           # GETHP / BENDY / PEN probes


def _run_case(inp):
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (F32(x) for x in inp)
    s = LayerState(km=1)
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = fi1, fi2, fi3
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = yi1, yi2, yi3
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = hi1, hi2, hi3
    lecden(s, 1)
    fp = s.fi[3, 1]
    hp1, ht1 = gethp(s, F32(0.30) * fp)
    hp2, ht2 = gethp(s, F32(0.60) * fp)
    hp3, ht3 = gethp(s, F32(0.90) * fp)
    hp4, ht4 = gethp(s, F32(0.99) * fp)
    out = {f"htr{i:02d}": s.htr[i, 1] for i in range(1, 51)}
    out.update({f"fnsq{i:02d}": s.fnsq[i, 1] for i in range(1, 51)})
    out["hi2"], out["yi2"], out["fsecv"] = s.hi[2, 1], s.yi[2, 1], s.fsecv[1]
    for name, v in zip(PROBE, [hp1, ht1, hp2, ht2, hp3, ht3, hp4, ht4,
                               bendy(s, 3, 1, F32(0.90) * fp),
                               pen(s, 1, 1, F32(0.90) * fp)]):
        out[name] = v
    return out


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_profile_dumps.sh")
def test_profile_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 113
    total = exact = 0
    htr_exact = fnsq_exact = 0
    htr_n = fnsq_n = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        f1_present = float(inp[1]) > 0.0
        got = _run_case(inp)
        for name, e_str in zip(header, exp):
            if name == "fsecv" and not f1_present:
                continue  # undefined output (stale COMMON) when no F1
            e = np.float32(e_str)
            d = ulp_diff_floored(got[name], e)
            assert d <= _bar(name), (
                f"case {k} {name}: got {got[name]!r} expected {e!r} "
                f"({d:.2f} floored ULP > {_bar(name)}), in {inp}")
            total += 1
            hit = d == 0.0
            exact += hit
            if name.startswith("htr"):
                htr_n += 1
                htr_exact += hit
            elif name.startswith("fnsq"):
                fnsq_n += 1
                fnsq_exact += hit

    # the profile itself must be bit-clean (arithmetic + correctly-rounded
    # SQRT); only the D-region EXP could nudge FNSQ off, so allow a sliver
    assert htr_exact == htr_n, f"HTR grid not bit-exact: {htr_exact}/{htr_n}"
    assert fnsq_exact / fnsq_n >= 0.99, f"FNSQ rate {fnsq_exact}/{fnsq_n}"
    assert exact / total >= 0.98, f"bit-exact rate {exact}/{total}"
