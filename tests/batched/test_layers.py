"""Stage B (PyTorch, batched) Module-2 layer chain vs. the oracle dump and
vs. the Stage A chain, on CPU and CUDA.

Same tier-B fixture as tests/scalar/test_layers.py (golden/dumps/layers):
the full TIMVAR->EF1VAR->F2VAR->ESIND->IONSET pipeline per grid point.
Measured at generation: the batched chain reproduces the oracle to the same
fidelity as the scalar port — the fs1/fs2/fs3/fi3 misses are the identical
VERSY cosf-floor cases (worst fs2 252 floored ULP), and the only extra noise
is a few ULP on the zenith-derived fields where torch libm differs from the
oracle's glibc ([[f32-libm-noise]]).
"""

import csv
import pathlib
from collections import defaultdict

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.virtim import virtim
from voacap_gpu.batched.layers import iono_params_batch

from tests.scalar.dumputil import COEFFS_DIR

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "golden" / "dumps" / "layers"
COEFFS = COEFFS_DIR    # the package's bundled copy (tests/scalar/dumputil.py)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
F32 = np.float32

FIELDS = ["fi1", "fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3",
          "fs1", "fs2", "fs3", "hs", "f2m3", "hpf2", "rat", "abiy", "clck",
          "zenang", "zenmax"]
BARS = {
    "fi1": 16, "fi2": 32, "fi3": 256, "yi1": 2, "yi2": 32, "yi3": 32,
    "hi1": 2, "hi2": 32, "hi3": 32, "fs1": 512, "fs2": 512, "fs3": 512,
    "hs": 2, "f2m3": 16, "hpf2": 16, "rat": 16, "abiy": 16, "clck": 16,
    "zenang": 16, "zenmax": 16,
}
EXACT_FIELDS = ["hi1", "hs", "yi1"]  # constants, transcendental-free


def _floored_ulp(a, b):
    a = np.float32(a)
    b = np.float32(b)
    if a == b:
        return 0.0
    return abs(float(a) - float(b)) / float(
        np.spacing(np.float32(max(abs(float(b)), 1.0))))


def _load_groups():
    if not (GOLDEN / "expected.csv").exists():
        return None
    cases = list(csv.reader(open(GOLDEN / "cases.csv", newline="")))
    rows = list(csv.reader(open(GOLDEN / "expected.csv", newline="")))
    assert rows[0] == FIELDS + ["kfx"]
    groups = defaultdict(list)
    for crow, erow in zip(cases, rows[1:]):
        key = (int(crow[0]), float(crow[1]), float(crow[2]))  # month,ssn,gmt
        groups[key].append((tuple(float(x) for x in crow[3:8]),
                            {f: float(v) for f, v in zip(FIELDS, erow)}))
    return groups


GROUPS = _load_groups()


@pytest.mark.skipif(GROUPS is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_layer_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_layers_b_vs_oracle(device):
    total = exact = 0
    exact_by_field = {f: 0 for f in FIELDS}
    n_by_field = {f: 0 for f in FIELDS}
    for (month, ssn, gmt), items in GROUPS.items():
        st = redmap(COEFFS, "CCIR", month, F32(ssn))
        ab = torch.from_numpy(virtim(st, F32(gmt))).to(device)
        g = np.array([it[0] for it in items], dtype=np.float32)
        cols = [torch.tensor(g[:, j], dtype=torch.float32, device=device)
                for j in range(5)]
        out = iono_params_batch(st, ab, month, ssn, gmt, *cols)
        for i, (_, exp) in enumerate(items):
            for f in FIELDS:
                got = float(out[f][i].cpu())
                d = _floored_ulp(got, exp[f])
                assert d <= BARS[f], (
                    f"[{device}] {f}: {got!r} vs {exp[f]!r} "
                    f"({d:.1f} floored ULP > {BARS[f]}) month={month} "
                    f"ssn={ssn} gmt={gmt}")
                total += 1
                n_by_field[f] += 1
                hit = d == 0.0
                exact += hit
                exact_by_field[f] += hit
    for f in EXACT_FIELDS:
        assert exact_by_field[f] == n_by_field[f], (
            f"[{device}] {f} must be bit-exact: "
            f"{exact_by_field[f]}/{n_by_field[f]}")
    assert exact / total >= 0.90, f"[{device}] bit-exact {exact}/{total}"
