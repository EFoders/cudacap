"""Stage A GEOM vs. the tier-B oracle dump.

The dump driver stubs NOISY's land/sea lookup with a longitude rule that
``wld_rule`` below mirrors exactly (the real map lookup is module 7).

Pass bars come from the measured generation run plus a conditioning
analysis.  GEOM is pure REAL*4: a 1-ULP cross-libm difference in a
float32 sin/cos seeds every downstream comparison.  Most fields are
bit-exact (overall ~98%), but the sample-area longitude comes through
ACOS whose error is amplified by 1/sqrt(1-qcos^2); the worst dump case
(receiver at -89.95 deg, sample 5 only 1000 km from it) has
qcos=0.99999636 — a ~370x amplification that turns one seed ULP into
~4e-5 rad of longitude *at a near-pole point where longitude is nearly
degenerate*.  The same Fortran built by a different compiler would
scatter comparably, so the bars accept it; the bit-exact-rate guardrails
keep the bars from hiding structural breakage.

Fields are compared with a scale-floored ULP (error against the field's
O(1) natural scale — see dumputil.ulp_diff_floored); km, sigpat and
epspat (branch outcomes) must be exact.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.geom import geom

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("geom")

# measured worst at generation: btrd 3, alatd 4, clong2 18, gyz2 10,
# gyz5 41, glat5 52, gmdip5 67, clong5 368 (the conditioning outlier)
BARS = {
    "dlong": 4, "gcd": 4, "gcdkm": 4, "btr": 8, "btrd": 8, "brtd": 8,
    "alatd": 8, "rlatd": 0,
    "rd": 4, "clat": 8, "glat": [8, 8, 16, 8, 96],
    "clong": [32, 32, 32, 32, 512], "gyz": [8, 16, 16, 16, 96],
    "gmdip": [8, 8, 16, 8, 128],
}


def wld_rule(rfltd, clgd):
    """Mirror of the NOISY stub in dump_geom.f: east hemisphere land."""
    return F32(1.0) if clgd >= F32(180.0) else F32(-1.0)


def bar_for(name):
    if name in BARS:
        return BARS[name]
    base, idx = name[:-1], int(name[-1]) - 1
    b = BARS[base]
    return b[idx] if isinstance(b, list) else b


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_mod1_dumps.sh")
def test_geom_vs_oracle():
    header, inputs, rows = DUMP
    total = 0
    exact = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        r = geom(F32(inp[2]), F32(inp[3]), F32(inp[4]), F32(inp[5]),
                 int(inp[6]), F32(inp[0]), F32(inp[1]), wld_rule)
        got = {"km": r.km, "dlong": r.dlong, "gcd": r.gcd,
               "gcdkm": r.gcdkm, "btr": r.btr, "btrd": r.btrd,
               "brtd": r.brtd, "alatd": r.alatd, "rlatd": r.rlatd}
        for base, arr in [("rd", r.rd), ("clat", r.clat),
                          ("clong", r.clong), ("glat", r.glat),
                          ("gyz", r.gyz), ("gmdip", r.gmdip),
                          ("sigpat", r.sigpat), ("epspat", r.epspat)]:
            for j in range(5):
                got[f"{base}{j+1}"] = arr[j]

        assert got["km"] == int(exp[0]), f"case {k}: km branch differs"
        for h, ev in zip(header[1:], exp[1:]):
            ev = np.float32(ev)
            if h.startswith(("sigpat", "epspat")):
                assert got[h] == ev, (
                    f"case {k} {h}: land/sea branch differs ({got[h]!r} "
                    f"vs {ev!r}), in {inp}")
                continue
            d = ulp_diff_floored(got[h], ev)
            assert d <= bar_for(h), (
                f"case {k} {h}: got {got[h]!r} expected {ev!r} "
                f"({d:.2f} floored ULP > {bar_for(h)}), in {inp}")
            total += 1
            exact += d == 0.0
    # guardrail: loose bars must not hide systematic drift
    assert exact / total >= 0.96, f"bit-exact rate {exact}/{total}"


@pytest.mark.skipif(DUMP is None, reason="no oracle dump")
def test_geom_branch_coverage():
    """The dump grid must keep exercising every structural branch."""
    _, inputs, rows = DUMP
    kms = {int(e[0]) for e in rows}
    assert kms == {1, 3, 5}
    sig_idx = None
    header = DUMP[0]
    land = sea = 0
    for e in rows:
        if sig_idx is None:
            sig_idx = header.index("sigpat1")
        v = np.float32(e[sig_idx])
        land += v == np.float32(0.001)
        sea += v == np.float32(5.0)
    assert land and sea, "land/sea seam not exercised on both sides"
    assert any(int(i[6]) == 1 for i in inputs), "no long-path cases"
