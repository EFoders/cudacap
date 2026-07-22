"""Shared helpers for tier-B dump comparisons."""

import csv
import pathlib

import numpy as np

GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "golden" / "dumps"


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
