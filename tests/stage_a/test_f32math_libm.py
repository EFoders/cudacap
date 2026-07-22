"""Characterize f32math's transcendentals against the raw glibc dump.

This pins the noise floor described in docs/f32_libm.md: our
correctly-rounded (float64-evaluate, round-once) sin/cos/acos/atan agree
with the runtime glibc the oracle links on the large majority of
arguments and are never worse than 1 ULP.  If a numpy/platform change
ever pushed that past 1 ULP, or the agreement rate collapsed, the fidelity
story in the trig-heavy dump tests (GEOM, VERSY) would be wrong — this
catches it directly.
"""

import csv
import struct

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import acos32, atan32, cos32, sin32

from .dumputil import GOLDEN


def _i2f(i):
    return struct.unpack("f", struct.pack("i", int(i)))[0]


def _load():
    d = GOLDEN / "libm"
    if not (d / "expected.csv").exists():
        return None
    with open(d / "expected.csv", newline="") as f:
        r = csv.reader(f)
        assert next(r) == ["xbits", "sinbits", "cosbits", "acosbits",
                           "atanbits"]
        return [tuple(_i2f(v) for v in row) for row in r]


ROWS = _load()


def _ulp(got, exp):
    got = np.float32(got)
    exp = np.float32(exp)
    if got == exp:
        return 0.0
    return abs(float(got) - float(exp)) / float(
        np.spacing(np.abs(exp) + np.float32(1e-30)))


@pytest.mark.skipif(ROWS is None, reason="libm dump missing — run "
                    "tools/oracle/dumps/run_libm_dump.sh")
@pytest.mark.parametrize("fn_name,col,helper,domain", [
    ("sin", 1, sin32, None),
    ("cos", 2, cos32, None),
    ("acos", 3, acos32, (-1.0, 1.0)),
    ("atan", 4, atan32, None),
])
def test_noise_floor(fn_name, col, helper, domain):
    worst = 0.0
    total = exact = 0
    for row in ROWS:
        x = row[0]
        if not np.isfinite(x):
            continue
        if domain is not None and not (domain[0] <= x <= domain[1]):
            continue
        exp = row[col]
        d = _ulp(helper(x), exp)
        assert d <= 1.0, f"{fn_name}({x!r}): {d:.2f} ULP vs glibc"
        worst = max(worst, d)
        total += 1
        exact += d == 0.0
    # correctly-rounded vs 1-ULP-accurate glibc: most agree, some differ
    assert exact / total >= 0.97, (
        f"{fn_name} agreement {exact}/{total} unexpectedly low")
