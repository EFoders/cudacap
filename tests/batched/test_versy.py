"""Stage B (PyTorch) VERSY vs. the tier-B oracle dump and vs. Stage A.

Runs on CPU always and CUDA when available (the Stage B production target).
Same fidelity bar as the scalar port (tests/scalar/test_versy.py): the
per-gamma floored-ULP bars encode the VERSY cosf-cancellation floor.

Measured at generation: on BOTH devices Stage B reproduces Stage A /the
oracle to identical fidelity (1336/1344 bit-exact, worst 397 floored ULP on
gamma1) — the batched GPU evaluation adds no error of its own; the eight
misses are the same [[f32-libm-noise]] glibc-cosf cases the scalar port has.
"""

import csv
import pathlib
from collections import defaultdict

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.versy import versy as versy_a
from voacap_gpu.scalar.virtim import virtim
from voacap_gpu.batched.versy import versy_batch

from tests.scalar.dumputil import COEFFS_DIR

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "golden" / "dumps" / "versy"
COEFFS = COEFFS_DIR    # the package's bundled copy (tests/scalar/dumputil.py)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
BARS = {1: 512, 2: 512, 3: 512, 4: 128, 5: 8, 6: 8}
F32 = np.float32


def _floored_ulp(a, b):
    a = np.float32(a)
    b = np.float32(b)
    if a == b:
        return 0.0
    return abs(float(a) - float(b)) / float(
        np.spacing(np.float32(max(abs(float(b)), 1.0))))


def _load_groups():
    """Group dump cases by (coeff,month,ssn,gmt) so one REDMAP+VIRTIM feeds
    a batch of grid points."""
    if not (GOLDEN / "expected.csv").exists():
        return None
    cases = list(csv.reader(open(GOLDEN / "cases.csv", newline="")))
    with open(GOLDEN / "expected.csv", newline="") as f:
        r = csv.reader(f)
        assert next(r) == [f"gamma{i}" for i in range(1, 7)]
        exp = list(r)
    groups = defaultdict(list)
    for crow, erow in zip(cases, exp):
        key = (int(crow[0]), int(crow[1]), float(crow[2]), float(crow[3]))
        groups[key].append((float(crow[4]), float(crow[5]), float(crow[6]),
                            [float(x) for x in erow]))
    return groups


GROUPS = _load_groups()


@pytest.mark.skipif(GROUPS is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_ionomap_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_versy_b_vs_oracle(device):
    exact = total = 0
    for (icoeff, month, ssn, gmt), pts in GROUPS.items():
        coeff = "URSI" if icoeff == 1 else "CCIR"
        st = redmap(COEFFS, coeff, month, F32(ssn))
        ab = torch.from_numpy(virtim(st, F32(gmt))).to(device)
        clat = torch.tensor([p[0] for p in pts], dtype=torch.float32,
                            device=device)
        clong = torch.tensor([p[1] for p in pts], dtype=torch.float32,
                             device=device)
        gmdip = torch.tensor([p[2] for p in pts], dtype=torch.float32,
                             device=device)
        g = versy_batch(st.ikim, ab, clat, clong, gmdip, 1, 6).cpu().numpy()
        for i, p in enumerate(pts):
            for iz in range(1, 7):
                d = _floored_ulp(g[i, iz], p[3][iz - 1])
                assert d <= BARS[iz], (
                    f"[{device}] gamma{iz}: {g[i, iz]!r} vs {p[3][iz-1]!r} "
                    f"({d:.1f} floored ULP > {BARS[iz]})")
                total += 1
                exact += d == 0.0
    assert exact / total >= 0.99, f"[{device}] bit-exact {exact}/{total}"


def _lcg(seed):
    s = seed

    def rnd(lo, hi):
        nonlocal s
        s = (s * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (s >> 11) / 2**53 * (hi - lo)
    return rnd


@pytest.mark.skipif(not COEFFS.exists(), reason="coeffs missing")
@pytest.mark.parametrize("device", DEVICES)
def test_versy_b_vs_stage_a_random(device):
    """Lane-for-lane agreement with the scalar port beyond the dump grid."""
    rnd = _lcg(20260726)
    pts = [(rnd(-1.55, 1.55), rnd(-3.14159, 3.14159), rnd(-1.5, 1.5))
           for _ in range(300)]
    st = redmap(COEFFS, "CCIR", 4, F32(120))
    ab_np = virtim(st, F32(9.0))
    ab = torch.from_numpy(ab_np).to(device)
    clat = torch.tensor([p[0] for p in pts], dtype=torch.float32, device=device)
    clong = torch.tensor([p[1] for p in pts], dtype=torch.float32, device=device)
    gmdip = torch.tensor([p[2] for p in pts], dtype=torch.float32, device=device)
    gb = versy_batch(st.ikim, ab, clat, clong, gmdip, 1, 6).cpu().numpy()
    exact = total = 0
    for i, (cl, cg, dp) in enumerate(pts):
        ga = versy_a(st, ab_np, F32(cl), F32(cg), F32(dp), 1, 6)
        for iz in range(1, 7):
            d = _floored_ulp(gb[i, iz], ga[iz])
            assert d <= BARS[iz], (
                f"[{device}] pt{i} gamma{iz}: B {gb[i, iz]!r} vs A {ga[iz]!r} "
                f"({d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    # CPU matches Stage A exactly; CUDA tracks it to the libm floor
    floor = 0.999 if device == "cpu" else 0.99
    assert exact / total >= floor, f"[{device}] bit-exact {exact}/{total}"
