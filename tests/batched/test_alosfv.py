"""Stage B batched ALOSFV vs the SAME tier-B oracle dump Stage A uses.

The whole dump is one batch, so the F1-present and F1-absent arms run side by
side and the selection between them is under test rather than assumed.

What this really checks is the CF CARRY.  AFAC is not thirty independent
cells: CF survives each loop and becomes the leading coefficient of the next,
which is what makes the loss continuous across the E->F1 and F1->F2 cusps.
Get the carry wrong -- thread the CLAMPED value back in as the coefficient,
say, instead of holding A(2) fixed across the two-cell bridge -- and cells
11..12 and everything downstream of them drift, while the E layer stays
perfect.  So the test asserts a nonzero count in the upper rows as well as
agreement: a port that zeroed them would otherwise pass on the many cells that
are legitimately zero.

Only the IEDP < 0 cases are used, Stage B's GENION implementing only that
model.

MEASURED: 2575/2580 exact on CPU (99.8%), 2490/2580 on CUDA (96.5%), worst
2 ULP either way.  Stage A is 4800/4800 on this dump because its EXP32 is
matched to this gfortran build's expf; torch's EXP is a different function.
The CUDA gap is EXP and nothing else, which is measured rather than assumed:
running ALOSFV alone on both devices from ONE CPU-built ionogram reproduces
the same 96.4% / 2 ULP, so it is not an inherited difference in HTRUE/HPRIM.

Two ULP on a loss factor is nothing.  The exact FRACTION is worth asserting
anyway, because it would collapse rather than drift if the CF carry were
wrong -- which is exactly how the YI(2) bug in this file's setup showed up.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.batched.alosfv import alosfv_batch
from voacap_gpu.batched.genion import genion_batch
from voacap_gpu.batched.profile import lecden_batch

from tests.scalar.dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("alosfv")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ULP_BAR = 8.0
# CUDA is lower purely because torch's device EXP is a different function
# from its host one -- see the module docstring, where that is measured
# rather than assumed.
EXACT_FLOOR = {"cpu": 0.99, "cuda": 0.90}   # measured 0.998 / 0.965


def _cases():
    _, inputs, rows = DUMP
    keep = [(inp, exp) for inp, exp in zip(inputs, rows)
            if int(round(float(inp[9]))) < 0]
    cols = np.array([[float(x) for x in inp[:9]]
                     + [float(x) for x in inp[10:16]]
                     for inp, _ in keep], dtype=np.float32).T
    return cols, [exp for _, exp in keep]


def _args(device):
    """Everything ALOSFV needs, with the chain run on `device`."""
    cols, expected = _cases()
    t = [torch.as_tensor(c, device=device) for c in cols]
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = t[:9]
    hpmuf = torch.stack(t[9:12])
    htmuf = torch.stack(t[12:15])
    htr, fnsq, _, yi2_n, _ = lecden_batch(fi1, fi2, fi3, yi1, yi2, yi3,
                                          hi1, hi2, hi3)
    _, hprim, htrue = genion_batch(htr, fnsq, fi1, fi2, fi3)
    # LECDEN NARROWS YI(2) IN PLACE when the F1 ledge has to be reconciled
    # with the E layer, and ALOSFV reads /RON/ afterwards -- so the F1 arm's
    # collision-frequency scale height is the ADJUSTED one, not the input.
    # Feeding the input value here made rows 14..17 disagree by up to 5% on
    # the lanes LECDEN touched, while the clamped rows above them still
    # matched, which is what made it look like noise rather than a wiring bug.
    yi = torch.stack([yi1, yi2_n, yi3])
    return (htrue, hprim, fi2, yi, hi1, hpmuf, htmuf), expected


def _run(device):
    args, expected = _args(device)
    return alosfv_batch(*args), expected, args[2]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_alosfv_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_alosfv_batch_vs_oracle(device):
    afac_t, expected, fi2 = _run(device)
    afac = afac_t.cpu().numpy()
    total = exact = 0
    worst, where = 0.0, ""
    for i, exp in enumerate(expected):
        for ih in range(30):
            e = np.float32(exp[ih])
            d = ulp_diff_floored(afac[ih, i], e)
            if d > worst:
                worst, where = d, (f"lane {i} afac[{ih + 1}]: "
                                   f"{afac[ih, i]!r} vs {e!r}")
            assert d <= ULP_BAR, (
                f"[{device}] lane {i} afac[{ih + 1}]: {afac[ih, i]!r} vs "
                f"{e!r} ({d:.2f} floored ULP)")
            total += 1
            exact += d == 0.0

    no_f1 = int((fi2 <= 0).sum())
    assert 0 < no_f1 < len(expected), (
        f"{no_f1}/{len(expected)} lanes lack an F1 ledge; need both arms")
    # the CF carry only shows up above the E layer, so those rows have to be
    # doing something -- otherwise agreement on a table of zeros proves nothing
    assert np.count_nonzero(afac[10:]) >= 100, (
        f"only {np.count_nonzero(afac[10:])} nonzero cells above the E layer; "
        "the cusp-continuity path is not being exercised")
    assert np.count_nonzero(afac[:10]) >= 10, (
        "no E-layer point reflects above the MUF height in any lane, so CF is "
        "never set and the E->F cusp coefficient is untested")
    assert exact / total >= EXACT_FLOOR[device], (
        f"[{device}] exact {exact}/{total}, worst {worst:.2f} at {where}")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """ONE CPU-built ionogram fed to both devices, so this isolates ALOSFV's
    own EXP from LECDEN's and GETHP's.  It is what says the CUDA residual in
    the oracle comparison above belongs to this routine rather than upstream.
    """
    args, _ = _args("cpu")
    a = alosfv_batch(*args).numpy()
    b = alosfv_batch(*[x.cuda() for x in args]).cpu().numpy()
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    sp = np.spacing(np.maximum(np.abs(b), 1.0))
    assert (d / sp).max() <= ULP_BAR, f"AFAC worst {(d / sp).max():.2f} ULP"
    assert (a == b).mean() >= 0.90, (
        f"only {(a == b).mean():.3f} of cells are bit-equal across devices; "
        "measured 0.964 and it is EXP alone")
