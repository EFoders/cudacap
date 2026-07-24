"""Stage B batched GENFAM vs the SAME tier-B oracle dump Stage A uses.

The whole dump is one batch.  It spans all twelve months, so the FAM/DUD
tables are fed in their per-lane form -- see the module docstring in
``batched/genfam.py`` for why that form exists; ``test_shared_tables_agree``
is what says the shared form production actually uses computes the same
thing, so the extra path is not an untested second implementation.

GENFAM has no data-dependent branch worth selecting on: every step is a
multiply-accumulate.  What it does have is INDEXING -- the IBK hemisphere
shift and the two frequency clamps -- and those fail loudly rather than
subtly.  A wrong column or a swapped clamp is a dB-sized error, not a
ULP-sized one, so a tight bar here is exactly what lets the GENOIS residual
next door be read as arithmetic rather than as bookkeeping.

MEASURED, whole dump as one batch, IDENTICAL on CPU and CUDA:

    field   exact       worst floored ULP
    fa      639/640     1.00   (case 198, freq 3.8686, month 5)
    dua     640/640     0
    dla     640/640     0
    dms     640/640     0
    dus     640/640     0
    dls     640/640     0
    total   3839/3840 = 0.99974

That is Stage A's number, cell for cell: Stage A measures 3839/3840 with the
same single miss, 1 ULP in FA on the ``2.**X`` that forms the second Horner
abscissa.  It reproduces because the two transcendentals here are evaluated
in float64 and rounded once to float32 on both stages, so torch's f32 libm
never enters -- which is also why CUDA and CPU are bit-identical instead of
differing at the [[f32-libm-noise]] floor the way ALOSFV's EXP does.

The five decile outputs are asserted at a floor of 0.999 rather than
bitwise: their only transcendental is the ALOG10 abscissa and they are in
fact exact on every case here, but a double-precision ALOG10 that moved by
one ULP in an unlucky place could still tip a float32 rounding, and that is
libm noise rather than a port defect.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.batched.genfam import genfam_batch

from tests.scalar.dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("genfam")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

ULP_BAR = 2.0                   # measured worst 1.00, on FA alone
# Both devices measured identically; the split is kept because the rule that
# CPU is not automatically bit-exact in Stage B still holds -- it is only the
# float64-evaluate-once transcendentals that make it so here.
EXACT_FLOOR = {"cpu": {"fa": 0.995, "_other": 0.999},
               "cuda": {"fa": 0.995, "_other": 0.999}}


def _batch():
    """The whole dump as one batch, with per-lane FAM/DUD (12 months in it)."""
    header, inputs, rows = DUMP
    months = sorted({int(r[0]) for r in inputs})
    sts = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0)) for m in months}
    fam = np.stack([sts[int(r[0])].fam for r in inputs], axis=-1)
    dud = np.stack([sts[int(r[0])].dud for r in inputs], axis=-1)
    cols = {
        "fam": fam, "dud": dud,
        "y2": np.array([float(r[1]) for r in inputs], dtype=np.float32),
        "iblk": np.array([int(r[2]) for r in inputs], dtype=np.int64),
        "freq": np.array([float(r[3]) for r in inputs], dtype=np.float32),
        "z": np.array([float(r[4]) for r in inputs], dtype=np.float32),
    }
    expected = np.array([[np.float32(c) for c in r] for r in rows],
                        dtype=np.float32).T
    return header, cols, expected


def _args(cols, device):
    return tuple(torch.as_tensor(cols[n], device=device)
                 for n in ("fam", "dud", "y2", "iblk", "freq", "z"))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genfam_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_genfam_batch_vs_oracle(device):
    header, cols, expected = _batch()
    assert header == ["fa", "dua", "dla", "dms", "dus", "dls"]
    got = genfam_batch(*_args(cols, device))

    for k, name in enumerate(header):
        g = got[k].cpu().numpy()
        e = expected[k]
        assert g.dtype == np.float32
        worst, where = 0.0, ""
        for i in range(len(e)):
            d = ulp_diff_floored(g[i], e[i])
            if d > worst:
                worst, where = d, f"lane {i}: {g[i]!r} vs {e[i]!r}"
            assert d <= ULP_BAR, (
                f"[{device}] {name} lane {i}: {g[i]!r} vs {e[i]!r} "
                f"({d:.2f} floored ULP)")
        frac = float((g == e).mean())
        floor = EXACT_FLOOR[device].get(name, EXACT_FLOOR[device]["_other"])
        assert frac >= floor, (
            f"[{device}] {name} exact {frac:.5f}, worst {worst:.2f} "
            f"at {where}")

    # ---- coverage --------------------------------------------------------
    y2, iblk, freq = cols["y2"], cols["iblk"], cols["freq"]
    blocks = set(np.unique(iblk + 6 * (y2 < 0)).tolist())
    assert blocks == set(range(1, 13)), f"unreached table column: {blocks}"
    # both clamps and the unclamped region have to be present, or the two
    # abscissae are agreeing by never being asked to differ
    assert (freq <= 10.0).sum() > 20, "no lane below the sigma-Fam clamp"
    assert ((freq > 10.0) & (freq <= 20.0)).sum() > 20, (
        "no lane between the two clamps, where x5 and x20 must DIFFER")
    assert (freq > 20.0).sum() > 20, "no lane above the decile clamp"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_frequency_clamps_hold_the_curve_ends(device):
    """Report 322's decile curves stop at 20 MHz, sigma-Fam's at 10.

    Above those the polynomials are evaluated at the endpoint rather than
    extrapolated, so the deciles are flat above 20 MHz and DMS above 10 --
    while FA keeps tracking frequency through both.  One batch: the lane
    below the clamp and the lanes above it are computed together, which is
    the arrangement that would catch a clamp applied to the whole batch on
    the strength of one lane's frequency.
    """
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    fq = np.array([20.0, 25.0, 40.0, 55.0, 10.0, 12.0, 18.0],
                  dtype=np.float32)
    n = len(fq)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    fa, dua, dla, dms, dus, dls = genfam_batch(
        t(st.fam), t(st.dud), t(np.full(n, 0.5, np.float32)),
        t(np.full(n, 3, np.int64)), t(fq), t(np.full(n, 60.0, np.float32)))
    fa, dua, dms = (x.cpu().numpy() for x in (fa, dua, dms))
    dla, dus, dls = (x.cpu().numpy() for x in (dla, dus, dls))

    for i in (1, 2, 3):                     # 25, 40, 55 MHz
        for name, v in (("dua", dua), ("dla", dla), ("dms", dms),
                        ("dus", dus), ("dls", dls)):
            assert v[i] == v[0], f"{name} moved above 20 MHz at {fq[i]}"
        assert fa[i] != fa[0], "FA must still track frequency"
    for i in (5, 6):                        # 12, 18 MHz vs 10
        assert dms[i] == dms[4], f"DMS moved above 10 MHz at {fq[i]}"
        assert dua[i] != dua[4], "DUA must still track frequency to 20 MHz"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_southern_hemisphere_shifts_the_table_column(device):
    """Only the SIGN of Y2 matters, and it selects blocks 7..12.

    Lanes 0..1 are north, 2..3 south, all asking for block 2 at 8 MHz -- so
    a port that resolved IBK once for the batch, or that let the magnitude
    of the latitude leak in, splits here.
    """
    st = redmap(COEFFS_DIR, "CCIR", 2, F32(100.0))
    y2 = np.array([0.01, 1.4, -0.01, -1.4], dtype=np.float32)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    out = genfam_batch(t(st.fam), t(st.dud), t(y2),
                       t(np.full(4, 2, np.int64)),
                       t(np.full(4, 8.0, np.float32)),
                       t(np.full(4, 70.0, np.float32)))
    for v in out:
        v = v.cpu().numpy()
        assert v[0] == v[1], "latitude MAGNITUDE changed the column"
        assert v[2] == v[3], "latitude MAGNITUDE changed the column"
    # the DUD row is the observable proof the six-column shift happened;
    # GENFAM has no way to ask for column 8 directly
    assert out[1].cpu().numpy()[0] != out[1].cpu().numpy()[2]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_shared_tables_agree_with_per_lane_tables():
    """The (14,12) form production uses and the (14,12,N) form the dump needs
    must be the same routine, not two.  Month 6's cases, both ways, bitwise."""
    _, cols, _ = _batch()
    _, inputs, _ = DUMP
    idx = [i for i, r in enumerate(inputs) if int(r[0]) == 6]
    assert len(idx) > 50
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    rest = [torch.as_tensor(cols[n][idx]) for n in ("y2", "iblk", "freq", "z")]
    shared = genfam_batch(torch.as_tensor(st.fam), torch.as_tensor(st.dud),
                          *rest)
    perlane = genfam_batch(torch.as_tensor(cols["fam"][..., idx]),
                           torch.as_tensor(cols["dud"][..., idx]), *rest)
    for name, a, b in zip(["fa", "dua", "dla", "dms", "dus", "dls"],
                          shared, perlane):
        assert bool((a == b).all()), f"{name} differs between table forms"


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """Inputs built ONCE on CPU and moved, so this isolates GENFAM's own
    ALOG10 and powf.  Bitwise, deliberately: both are evaluated in float64
    and rounded once, so there is no f32 libm difference to absorb and a
    tolerance here would hide a real device split."""
    _, cols, _ = _batch()
    args = _args(cols, "cpu")
    a = genfam_batch(*args)
    b = genfam_batch(*[x.cuda() for x in args])
    for name, x, y in zip(["fa", "dua", "dla", "dms", "dus", "dls"], a, b):
        assert bool((x == y.cpu()).all()), (
            f"{name}: {int((x != y.cpu()).sum())} lanes differ across devices")
