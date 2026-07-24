"""Stage A DAZEL vs. the tier-B oracle dump (golden/dumps/dazel/).

Pass bar (PLAN.md section 0): the oracle dump was produced by gfortran/glibc
in WSL while this test usually runs under Windows CPython, so transcendental
libm results can differ by ~1 ULP in float64. After narrowing to the REAL*4
interface that almost always vanishes — every field must match within 1
float32 ULP, and the overwhelming majority must be bit-exact.
"""

import csv
import pathlib

import numpy as np
import pytest

from voacap_gpu.scalar.dazel import Azel, dazel

GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "golden" / "dumps" / "dazel"

FIELDS = [
    "ztlat", "ztlon", "ztht", "zrlat", "zrlon", "zrht", "ztaz", "zraz",
    "ztelv", "zrelv", "zd", "zdgc", "ztakof", "zrakof",
]


def load_cases():
    if not (GOLDEN / "expected.csv").exists():
        pytest.skip("oracle dump missing — run tools/oracle/dumps/run_dazel_dump.sh")
    with open(GOLDEN / "cases.csv", newline="") as f:
        inputs = [row for row in csv.reader(f)]
    with open(GOLDEN / "expected.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0] == "mode" and header[1:] == FIELDS
        expected = [row for row in reader]
    assert len(inputs) == len(expected)
    return list(zip(inputs, expected))


CASES = load_cases() if (GOLDEN / "expected.csv").exists() else []


def run_case(inp):
    mode = int(inp[0])
    tlat, tlon, tht, rlat, rlon, rht, taz, dgc = (np.float32(v) for v in inp[1:])
    z = Azel(ztlat=tlat, ztlon=tlon, ztht=tht, zrlat=rlat, zrlon=rlon,
             zrht=rht, ztaz=taz, zdgc=dgc)
    dazel(z, mode)
    return z


def ulp_diff(got: np.float32, exp: np.float32) -> float:
    if got == exp:
        return 0.0
    return abs(float(got) - float(exp)) / float(np.spacing(np.abs(exp) + np.float32(1e-30)))


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_dazel_case(idx):
    inp, exp_row = CASES[idx]
    z = run_case(inp)
    exp = {name: np.float32(v) for name, v in zip(FIELDS, exp_row[1:])}
    for name in FIELDS:
        got = getattr(z, name)
        assert ulp_diff(got, exp[name]) <= 1.0, (
            f"case {idx} mode {inp[0]} field {name}: got {got!r} expected {exp[name]!r}"
            f" (inputs {inp[1:]})"
        )


def test_bit_exact_rate():
    """Guardrail: cross-libm noise may flip the odd last bit, but if more
    than 2% of fields differ at all, something structural is wrong."""
    if not CASES:
        pytest.skip("no oracle dump")
    total = 0
    exact = 0
    for inp, exp_row in CASES:
        z = run_case(inp)
        for name, v in zip(FIELDS, exp_row[1:]):
            total += 1
            if getattr(z, name) == np.float32(v):
                exact += 1
    rate = exact / total
    assert rate >= 0.98, f"bit-exact rate {rate:.4%} ({exact}/{total})"
