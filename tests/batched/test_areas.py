"""Stage B batched SETLNG + SELMOD vs the SAME tier-B oracle dumps Stage A uses.

Both routines are pure control flow -- SETLNG copies, SELMOD compares -- so
there is no arithmetic to drift and every cell must match EXACTLY on every
device.  That makes these the cheapest tests in Stage B and also the ones with
the least excuse for a bar.

The whole dump is one batch, which is the only thing that makes them
interesting.  Each dump case carries its own KFX, so all three of SETLNG's
copy patterns and all four of SELMOD's arms are live in a single call -- which
is precisely what the scalar code never has to handle, and precisely what an
index-map formulation could get wrong while still passing a uniform batch.

SETLNG's sentinels are distinct exact integers below 2**24, so every witness
identifies the slot it came from.  A copy from the wrong AREA and a copy from
the wrong ROW give different wrong answers, and a missing copy gives a third.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.batched.areas import NAREA, _pattern, replicate, \
    selmod_batch, setlng_sources

from tests.scalar.dumputil import load_dump

SETLNG = load_dump("setlng")
SELMOD = load_dump("selmod")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# the injection formula and witness order mirror dump_setlng.f exactly
_SCALARS = ["clat", "clong", "glat", "rd", "gyz", "rat", "gmdip", "clck",
            "abiy", "artic", "sigpat", "epspat", "hs"]
_LAYERS = ["fs", "fi", "yi", "hi"]
_IONO = ["hprim", "htrue", "fvert", "afac"]
_IFOB_PAIRS = [(1, 1), (40, 1), (1, 30), (40, 30), (20, 15)]
NIFOB_A, NIFOB_H = 40, 30


def _inject(salts, device):
    """The sentinel state, as (…, 5, N) tensors."""
    n = len(salts)
    s = torch.as_tensor(np.asarray(salts, dtype=np.float32), device=device)
    a = torch.arange(1, NAREA + 1, dtype=torch.float32,
                     device=device).unsqueeze(1)                    # (5,1)
    scal = {name: (i + 1) * 10000 + a * 1000 + s
            for i, name in enumerate(_SCALARS)}
    il3 = torch.arange(1, 4, dtype=torch.float32,
                       device=device).view(3, 1, 1)
    lay = {name: j * 100000 + a * 10000 + il3 * 1000 + s
           for j, name in enumerate(_LAYERS, start=1)}
    il30 = torch.arange(1, 31, dtype=torch.float32,
                        device=device).view(30, 1, 1)
    iono = {name: j * 1000000 + a * 100000 + il30 * 1000 + s
            for j, name in enumerate(_IONO, start=1)}
    ia = torch.arange(1, NIFOB_A + 1, dtype=torch.float32,
                      device=device).view(NIFOB_A, 1, 1, 1)
    ih = torch.arange(1, NIFOB_H + 1, dtype=torch.float32,
                      device=device).view(1, NIFOB_H, 1, 1)
    ifob = (a * 100000 + ih * 1000 + ia * 10 + s).expand(
        NIFOB_A, NIFOB_H, NAREA, n)
    return scal, lay, iono, ifob.contiguous()


def _witnesses(scal, lay, iono, ifob):
    """The same 151 cells the Fortran dump driver records, in its order."""
    out = []
    for name in _SCALARS:
        out.extend(scal[name][a - 1] for a in (2, 3, 4, 5))
    for name in _LAYERS:
        out.extend(lay[name][il - 1, a - 1]
                   for il in (1, 2, 3) for a in (2, 3, 4, 5))
    for name in _IONO:
        out.extend(iono[name][il - 1, a - 1]
                   for il in (1, 15, 30) for a in (2, 3, 4))
    for (ia, il) in _IFOB_PAIRS:
        out.extend(ifob[ia - 1, il - 1, a - 1] for a in (2, 3, 4))
    return out


@pytest.mark.skipif(SETLNG is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_setlng_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_setlng_batch_vs_oracle(device):
    header, inputs, rows = SETLNG
    assert len(header) == 151
    kfx = torch.as_tensor([int(inp[0]) for inp in inputs], device=device)
    scal, lay, iono, ifob = _inject([int(inp[1]) for inp in inputs], device)

    src_s, src_i = setlng_sources(kfx)
    scal = {k: replicate(v, src_s) for k, v in scal.items()}
    lay = {k: replicate(v, src_s) for k, v in lay.items()}
    iono = {k: replicate(v, src_i) for k, v in iono.items()}
    ifob = replicate(ifob.reshape(NIFOB_A * NIFOB_H, NAREA, -1),
                     src_i).reshape(NIFOB_A, NIFOB_H, NAREA, -1)

    got = _witnesses(scal, lay, iono, ifob)
    assert len(got) == 151
    got = np.stack([g.cpu().numpy() for g in got])          # (151, N)
    want = np.array([[np.float32(x) for x in exp] for exp in rows],
                    dtype=np.float32).T
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"[{device}] {len(bad)} witnesses differ; first is case "
        f"{bad[0][1]} kfx={int(kfx[bad[0][1]])} col{bad[0][0] + 1:03d}: "
        f"got {got[tuple(bad[0])]!r} expected {want[tuple(bad[0])]!r}")

    seen = sorted({int(k) for k in kfx})
    assert set(_pattern(kfx).cpu().numpy().tolist()) == {0, 1, 2}, (
        f"KFX values {seen} do not cover all three copy patterns")


@pytest.mark.skipif(SELMOD is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_selmod_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_selmod_batch_vs_oracle(device):
    _, inputs, rows = SELMOD
    cols = np.array([[float(x) for x in inp] for inp in inputs],
                    dtype=np.float32).T
    kfx = torch.as_tensor(cols[0].astype(np.int64), device=device)
    fi = [torch.as_tensor(c, device=device) for c in cols[1:]]
    got = selmod_batch(kfx, *fi).cpu().numpy()
    want = np.array([int(round(float(exp[0]))) for exp in rows])
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"[{device}] case {bad[0][0]} kfx={int(kfx[bad[0][0]])}: JMODE "
        f"{got[bad[0][0]]} vs {want[bad[0][0]]}")

    assert set(np.unique(want)) == {1, 2, 3}, (
        f"the dump only ever selects {sorted(set(want.tolist()))}; every arm "
        "has to be reachable for this to test the selection")
    # the F2 tie-break is the one arm a coarse dump could miss entirely
    delfi = (fi[3] - fi[4]).abs().cpu().numpy()
    assert int((delfi < 0.01).sum()) >= 1, (
        "no case has the two path ends tied on F2, so the E-layer tie-break "
        "is never exercised")
