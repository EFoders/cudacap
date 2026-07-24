"""Stage B (PyTorch) DAZEL vs. the tier-B oracle dump and vs. Stage A.

Same pass bar as Stage A (see tests/stage_a/test_dazel.py): every field
within 1 float32 ULP of the gfortran oracle, and ≥98% bit-exact.  Runs on
CPU always and on CUDA when available (the Stage B production target).
"""

import csv
import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.dazel import Azel, dazel as dazel_a
from voacap_gpu.batched.dazel import AzelBatch, dazel as dazel_b

GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "golden" / "dumps" / "dazel"

FIELDS = [
    "ztlat", "ztlon", "ztht", "zrlat", "zrlon", "zrht", "ztaz", "zraz",
    "ztelv", "zrelv", "zd", "zdgc", "ztakof", "zrakof",
]
IN_COLS = ["ztlat", "ztlon", "ztht", "zrlat", "zrlon", "zrht", "ztaz", "zdgc"]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def load_cases():
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


def make_batch(rows, device):
    """Build an AzelBatch from case-input rows (mode dropped), outputs zeroed
    exactly as the Fortran dump driver zeroes them."""
    n = len(rows)
    z = AzelBatch.zeros(n, device=device)
    cols = np.array(
        [[np.float32(v) for v in r[1:]] for r in rows], dtype=np.float32
    )
    for j, name in enumerate(IN_COLS):
        setattr(z, name, torch.from_numpy(cols[:, j].copy()).to(device))
    return z


def ulp_diffs(got: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Elementwise |got-exp| in units of exp's float32 spacing (0 where equal)."""
    eq = got == exp
    denom = np.spacing(np.abs(exp) + np.float32(1e-30)).astype(np.float64)
    d = np.abs(got.astype(np.float64) - exp.astype(np.float64)) / denom
    d[eq] = 0.0
    return d


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", [0, 1])
def test_vs_oracle_dump(device, mode):
    if not CASES:
        pytest.skip("oracle dump missing — run tools/oracle/dumps/run_dazel_dump.sh")
    rows = [(inp, exp) for inp, exp in CASES if int(inp[0]) == mode]
    assert rows, f"no mode-{mode} cases in dump"
    z = make_batch([inp for inp, _ in rows], device)
    dazel_b(z, mode)
    total = 0
    exact = 0
    for j, name in enumerate(FIELDS):
        got = getattr(z, name).cpu().numpy()
        exp = np.array([np.float32(e[1:][j]) for _, e in rows], dtype=np.float32)
        d = ulp_diffs(got, exp)
        worst = int(np.argmax(d))
        assert d[worst] <= 1.0, (
            f"[{device}] mode {mode} field {name} case {worst}: "
            f"got {got[worst]!r} expected {exp[worst]!r} ({d[worst]:.2f} ULP), "
            f"inputs {rows[worst][0][1:]}"
        )
        total += d.size
        exact += int((d == 0.0).sum())
    assert exact / total >= 0.98, f"bit-exact rate {exact}/{total}"


def _lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return np.float32(lo + (state >> 11) / 2**53 * (hi - lo))

    return rnd


def random_rows(n_mode0=400, n_mode1=200, seed=20260722):
    """Fixed-seed random cases (dump-generator ranges) for the A/B cross-check."""
    rnd = _lcg(seed)
    rows = []
    for _ in range(n_mode0):
        rows.append([
            "0", rnd(-90, 90), rnd(-360, 360), rnd(0, 3000),
            rnd(-90, 90), rnd(-360, 360), rnd(0, 3000), "0", "0",
        ])
    for _ in range(n_mode1):
        rows.append([
            "1", rnd(-89, 89), rnd(-180, 180), rnd(0, 3000),
            "0", "0", rnd(0, 3000), rnd(0, 360), rnd(100, 19000),
        ])
    return rows


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", [0, 1])
def test_vs_stage_a_random(device, mode):
    """Lane-for-lane agreement with Stage A on inputs beyond the dump."""
    rows = [r for r in random_rows() if int(r[0]) == mode]
    z = make_batch(rows, device)
    dazel_b(z, mode)
    got = {name: getattr(z, name).cpu().numpy() for name in FIELDS}
    total = 0
    exact = 0
    for i, r in enumerate(rows):
        za = Azel(**{
            name: np.float32(v) for name, v in zip(IN_COLS, r[1:])
        })
        dazel_a(za, mode)
        for name in FIELDS:
            e = getattr(za, name)
            g = got[name][i]
            d = ulp_diffs(np.array([g]), np.array([e]))[0]
            assert d <= 1.0, (
                f"[{device}] mode {mode} case {i} field {name}: "
                f"stage B {g!r} vs stage A {e!r} ({d:.2f} ULP), inputs {r[1:]}"
            )
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.98, f"bit-exact rate {exact}/{total}"


@pytest.mark.parametrize("device", DEVICES)
def test_mode1_preserves_zd_zdgc_ztaz(device):
    """Mode 1 must leave zd/zdgc (and the ztaz input) untouched."""
    rows = [r for r in random_rows(n_mode0=0, n_mode1=32) if r[0] == "1"]
    z = make_batch(rows, device)
    zd0 = z.zd.clone()
    zdgc0 = z.zdgc.clone()
    ztaz0 = z.ztaz.clone()
    dazel_b(z, 1)
    assert torch.equal(z.zd, zd0)
    assert torch.equal(z.zdgc, zdgc0)
    assert torch.equal(z.ztaz, ztaz0)
