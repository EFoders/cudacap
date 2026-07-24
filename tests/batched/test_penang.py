"""Stage B batched PENANG vs the SAME tier-B oracle dump Stage A uses.

Every case of the dump is loaded as one lane of one batch, which is the point:
PENANG is a nest of early returns, and the batched version computes all of them
and selects.  A batch whose lanes take DIFFERENT arms is what tests that the
selection is right -- a uniform batch would pass no matter which arm the code
actually wired up.  The coverage assertions below make that explicit rather
than hoping the dump happens to be varied.

Only the IEDP < 0 cases are used, because Stage B's GENION implements only
that model (deliberately -- see its docstring).  The dump is about half each
way, so this still leaves ~99 lanes.

MEASURED: 486/495 exact on CPU (98.2%, worst 1 ULP), 451/495 on CUDA (91.1%,
worst 2 ULP).  Stage A gets 1000/1000 on this same dump because its ACOS32 is
matched to this gfortran build's acosf; torch's ACOS is a different function
and differs by a ULP here and there, so the residual is libm, not the port.
Two ULP on an angle in degrees is 4e-6 degrees.  The exact FRACTION is still
worth asserting: a wrong branch selection would move a lane by TENS of
degrees, not by two ULP, so the fraction collapsing is the signal.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.batched.genion import genion_batch
from voacap_gpu.batched.penang import penang_batch
from voacap_gpu.batched.profile import lecden_batch

from tests.scalar.dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("penang")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_NAMES = ["delpen1", "delpen2", "delpen3", "dskpkm", "dmaxkm"]
ULP_BAR = 4.0
EXACT_FLOOR = {"cpu": 0.95, "cuda": 0.85}   # measured 0.982 / 0.911


def _cases():
    """The IEDP < 0 subset, as columns of one batch."""
    _, inputs, rows = DUMP
    keep = [(inp, exp) for inp, exp in zip(inputs, rows)
            if int(round(float(inp[9]))) < 0]
    cols = np.array([[float(x) for x in inp[:9]] + [float(inp[10])]
                     for inp, _ in keep], dtype=np.float32).T
    return cols, [exp for _, exp in keep]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_penang_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_penang_batch_vs_oracle(device):
    cols, expected = _cases()
    t = [torch.as_tensor(c, device=device) for c in cols]
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, freq = t
    htr, fnsq, _, _, _ = lecden_batch(fi1, fi2, fi3, yi1, yi2, yi3,
                                      hi1, hi2, hi3)
    fvert, _, htrue = genion_batch(htr, fnsq, fi1, fi2, fi3)

    delpen, nose = penang_batch(htrue, fvert, fi2, freq)
    # FINDF presets these and PENANG only ever flags them negative
    dskpkm = torch.where(nose, -torch.ones_like(freq),
                         torch.full_like(freq, 10000.0))
    dmaxkm = torch.where(nose, -torch.ones_like(freq),
                         torch.zeros_like(freq))
    got = [delpen[1], delpen[2], delpen[3], dskpkm, dmaxkm]
    got = [g.cpu().numpy() for g in got]

    total = exact = 0
    worst, where = 0.0, ""
    for i, exp in enumerate(expected):
        for name, arr, e_str in zip(_NAMES, got, exp):
            e = np.float32(e_str)
            d = ulp_diff_floored(arr[i], e)
            if d > worst:
                worst, where = d, f"[{device}] lane {i} {name}: {arr[i]!r} vs {e!r}"
            assert d <= ULP_BAR, (
                f"[{device}] lane {i} {name}: {arr[i]!r} vs {e!r} "
                f"({d:.2f} floored ULP)")
            total += 1
            exact += d == 0.0

    # the arms actually have to be exercised, or the selection proves nothing
    penetrated = sum(1 for e in expected if abs(float(e[0]) - 89.9) < 1e-3)
    no_f1 = int((fi2 <= 0).sum())
    assert penetrated >= 1, "no lane has the E layer penetrating"
    assert 0 < no_f1 < len(expected), (
        f"{no_f1}/{len(expected)} lanes lack an F1 ledge; need both arms")
    # NOT asserted here: the F2-nose veto.  Every case in this dump has
    # DSKPKM = 10000, so the oracle never exercises it and claiming coverage
    # would be false.  It is covered against Stage A instead, below.
    assert int(nose.sum()) == 0, (
        "a dump case now vetoes the area; the -1 flag is reachable from the "
        "oracle after all and belongs in this comparison")
    assert exact / total >= EXACT_FLOOR[device], (
        f"exact {exact}/{total}, worst {worst:.2f} at {where}")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_the_f2_nose_veto_matches_stage_a():
    """The one arm the oracle dump cannot reach.

    Label 175 fires when even the F2 NOSE lets the ray through -- DELPEN(3)
    forced to zero and DSKPKM/DMAXKM flagged -1, which tells FINDF the area is
    dead at this frequency.  It needs an operating frequency far above the F2
    critical (CDEL > 0.999999 wants FVERT(30) below about 0.3*FREQ), and every
    dump case is well below that, so the comparison here is against Stage A
    rather than against the Fortran.  That is weaker -- it shows the two
    transliterations agree, not that both are right -- and it is the strongest
    claim available without a new dump.
    """
    from voacap_gpu.scalar.genion import genion
    from voacap_gpu.scalar.layerstate import LayerState
    from voacap_gpu.scalar.lecden import lecden
    from voacap_gpu.scalar.penang import penang
    from voacap_gpu.scalar.reflxstate import ReflxState

    cols, _ = _cases()
    cols = cols[:, :12].copy()
    cols[9] = 8.0 * cols[2]                 # FREQ well past the F2 critical
    t = [torch.as_tensor(c) for c in cols]
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, freq = t
    htr, fnsq, _, _, _ = lecden_batch(fi1, fi2, fi3, yi1, yi2, yi3,
                                      hi1, hi2, hi3)
    fvert, _, htrue = genion_batch(htr, fnsq, fi1, fi2, fi3)
    delpen, nose = penang_batch(htrue, fvert, fi2, freq)
    assert int(nose.sum()) == nose.numel(), (
        "the synthesised cases were meant to penetrate and do not")

    for i in range(cols.shape[1]):
        s = LayerState(km=1)
        s.iedp = -1
        for lay in range(1, 4):
            s.fi[lay, 1] = np.float32(cols[lay - 1, i])
            s.yi[lay, 1] = np.float32(cols[lay + 2, i])
            s.hi[lay, 1] = np.float32(cols[lay + 5, i])
        lecden(s, 1)
        genion(s, 1)
        r = ReflxState()
        r.dskpkm[1] = np.float32(10000.0)
        penang(s, r, 1, float(cols[9, i]))
        assert int(r.dskpkm[1]) == -1 and float(r.delpen[3, 1]) == 0.0
        for lay in range(1, 4):
            got = float(delpen[lay][i])
            want = float(r.delpen[lay, 1])
            assert got == want, (
                f"lane {i} DELPEN({lay}): {got!r} vs Stage A {want!r}")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """The ionogram is built once on the CPU and copied, so this isolates
    PENANG's own arithmetic from LECDEN's EXP noise."""
    cols, _ = _cases()
    t = [torch.as_tensor(c) for c in cols]
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, freq = t
    htr, fnsq, _, _, _ = lecden_batch(fi1, fi2, fi3, yi1, yi2, yi3,
                                      hi1, hi2, hi3)
    fvert, _, htrue = genion_batch(htr, fnsq, fi1, fi2, fi3)
    outs = []
    for dev in ("cpu", "cuda"):
        dp, nose = penang_batch(htrue.to(dev), fvert.to(dev), fi2.to(dev),
                                freq.to(dev))
        outs.append((dp.cpu().numpy(), nose.cpu().numpy()))
    assert np.array_equal(outs[0][1], outs[1][1]), (
        "the F2-nose penetration flag differs between devices; it is a pure "
        "comparison and must not")
    d = np.abs(outs[0][0].astype(np.float64) - outs[1][0].astype(np.float64))
    sp = np.spacing(np.maximum(np.abs(outs[1][0]), 1.0))
    assert (d / sp).max() <= ULP_BAR, f"DELPEN worst {(d / sp).max():.2f} ULP"
