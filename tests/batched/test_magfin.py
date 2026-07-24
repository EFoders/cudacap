"""Stage B (PyTorch, batched) MAGVAR/MAGFIN vs. the tier-B oracle dump, on
CPU and CUDA.

Same fixture as tests/scalar/test_magvar.py.  gyz and clg (clg is a pure
float32 add) stay bit-exact/1-ULP; gmdip is a dip angle that can pass
through zero, so it uses the scale-floored metric — the raw ULP explodes
there while the absolute error stays ~1e-8 rad (the atan-chain libm floor,
a touch wider for torch than numpy but negligible on the radian scale).
"""

import csv
import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.batched.magfin import magvar_batch

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "golden" / "dumps" / "magvar"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
HEADER = ["gyz", "gmdip", "clg"]


def _load():
    if not (GOLDEN / "expected.csv").exists():
        return None
    inp = list(csv.reader(open(GOLDEN / "cases.csv", newline="")))
    with open(GOLDEN / "expected.csv", newline="") as f:
        r = csv.reader(f)
        assert next(r) == HEADER
        exp = list(r)
    return inp, exp


DUMP = _load()


def _ulp(a, b, floor=None):
    a = np.float32(a)
    b = np.float32(b)
    if a == b:
        return 0.0
    denom = np.abs(b) if floor is None else np.float32(max(abs(float(b)), floor))
    return abs(float(a) - float(b)) / float(
        np.spacing(np.float32(denom) + np.float32(1e-30)))


# All three are O(1)-scale (MHz / radians) and gmdip passes through zero, so
# the scale-floored metric is used throughout; worst observed is 1 ULP.
@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_mod1_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_magvar_b_vs_oracle(device):
    inp, exp = DUMP
    cenlat = torch.tensor([np.float32(r[0]) for r in inp], dtype=torch.float32,
                          device=device)
    cenlg = torch.tensor([np.float32(r[1]) for r in inp], dtype=torch.float32,
                         device=device)
    gyz, gmdip, clg = magvar_batch(cenlat, cenlg)
    got = {"gyz": gyz.cpu().numpy(), "gmdip": gmdip.cpu().numpy(),
           "clg": clg.cpu().numpy()}
    total = exact = 0
    clg_exact = 0
    for k, e in enumerate(exp):
        for j, h in enumerate(HEADER):
            d = _ulp(got[h][k], np.float32(e[j]), 1.0)
            assert d <= 2.0, (f"[{device}] case {k} {h}: got {got[h][k]!r} "
                              f"expected {np.float32(e[j])!r} ({d:.2f} ULP), "
                              f"in {inp[k]}")
            total += 1
            exact += d == 0.0
        clg_exact += got["clg"][k] == np.float32(e[2])
    # clg is a pure float32 add — must be exact; gyz/gmdip carry the 1-ULP
    # sin/cos/atan libm floor (torch vs the oracle's glibc)
    assert clg_exact == len(exp), f"[{device}] clg not bit-exact"
    assert exact / total >= 0.90, f"[{device}] bit-exact {exact}/{total}"
