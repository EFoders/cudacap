"""Shared helpers for tier-B dump comparisons."""

import csv
import pathlib

import numpy as np

import voacap_gpu

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = REPO / "golden" / "dumps"

# The vendored Fortran tree.  Dev-only (removed from the distributed tree in
# 477dad7); tests that genuinely need it are marked ``oracle`` and skipped by
# tests/conftest.py when it is absent.
REFERENCE = REPO / "reference"
REFERENCE_ITSHFBC = REFERENCE / "voacapl" / "voacapl" / "itshfbc"
REFERENCE_COEFFS_DIR = REFERENCE_ITSHFBC / "coeffs"
REFERENCE_ANTENNAS_DIR = REFERENCE_ITSHFBC / "antennas" / "samples"

# The CCIR/URSI coefficients the PACKAGE ships (engine._COEFFS_DIR; pinned
# byte-identical to the vendored .asc sources by tests/test_packaging.py), so
# the dump tests need no reference/ tree.
COEFFS_DIR = pathlib.Path(voacap_gpu.__file__).resolve().parent / "data" / "coeffs"

# Byte-exact copies of the stock voacapl sample antennas the tests read
# (provenance in tests/data/README.md).  Only the small ones are copied;
# sample.13 (269 KB) is read from REFERENCE_ANTENNAS_DIR by an oracle test.
ANTENNAS_DIR = REPO / "tests" / "data" / "antennas"


def load_dump(name):
    """Returns (header, input rows, expected rows) or None if missing."""
    d = GOLDEN / name
    if not (d / "expected.csv").exists():
        return None
    with open(d / "cases.csv", newline="") as f:
        inputs = list(csv.reader(f))
    with open(d / "expected.csv", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    assert len(inputs) == len(rows)
    return header, inputs, rows


def ulp_diff(got, exp) -> float:
    """|got-exp| in units of float32 spacing at exp (0 when bit-equal)."""
    got = np.float32(got)
    exp = np.float32(exp)
    if got == exp:
        return 0.0
    return abs(float(got) - float(exp)) / float(
        np.spacing(np.abs(exp) + np.float32(1e-30)))


def ulp_diff_floored(got, exp, floor=1.0) -> float:
    """ULP against the field's natural scale: spacing at max(|exp|, floor).

    For quantities whose natural scale is O(1) (radians, MHz), a raw ULP
    metric explodes when the expected value happens to sit near zero even
    though the absolute error is tiny; flooring the spacing measures error
    against the field scale instead.
    """
    got = np.float32(got)
    exp = np.float32(exp)
    if got == exp:
        return 0.0
    denom = float(np.spacing(np.float32(max(abs(float(exp)), floor))))
    return abs(float(got) - float(exp)) / denom
