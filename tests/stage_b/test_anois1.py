"""Stage B batched ANOIS1/NOISY vs the SAME tier-B oracle dump Stage A uses.

All 189 cases go in as ONE batch, so every noise block, both longitude
branches and the no-data-base early return run side by side and the SELECTION
between them is under test rather than assumed.

TWO KINDS OF FIELD, TWO KINDS OF BAR.  CC/TM/KJ/JK are the block-selection
arithmetic: a truncation, an integer centre time and three-way comparisons.
Nothing there is approximate -- a lane that picks the wrong block is wrong by
a whole map -- so they are asserted BITWISE, with no tolerance at all, and
they pass bitwise on both devices.  ATNU/ATNY are the Fourier sums and carry
the [[f32-libm-noise]] SIN/COS floor, amplified by an angle-addition
recurrence run out to 29 harmonics and a 435-term accumulation.

MEASURED over 189 lanes (CC/TM/KJ/JK over the 180 with a data base):

    field   CPU exact      worst    CUDA exact     worst
    atnu    165/189 .873   2.0 ULP  144/189 .762   3.0 ULP
    atny    163/189 .862   3.0 ULP  154/189 .815   4.0 ULP
    cc/tm   180/180 1.000  0        180/180 1.000  0
    kj/jk   180/180 1.000  0        180/180 1.000  0

CPU-vs-CUDA on identical CPU-built inputs: ATNU 74.6% bit-equal, ATNY 78.8%,
worst 3 and 4 floored ULP; CC/TM/KJ/JK bit-identical.

THE RESIDUAL IS THE LIBM PAIR AND NOTHING ELSE, and that is measured, not
asserted by hand-waving: ``test_series_order_is_exact_given_stage_a_sincos``
substitutes Stage A's gfortran-matched SIN32/COS32 for torch's and the batched
NOISY then reproduces Stage A BIT FOR BIT, 60/60 on two different maps.  So
the accumulation order, the recurrence and the constants are exact; only the
two transcendental calls differ.  Stage A scores 98.5% on this dump for the
same reason in reverse -- its SIN/COS are matched to this gfortran build's
and torch's are different functions.

NOT COVERED BY THE DUMP: the KJ=7/8 map limits (LM=15, LN=10).  ANOIS1 only
ever asks for blocks 1..6, which all use LM=29/LN=15, so the dump cannot reach
the other branch -- ``test_dump_never_reaches_the_kj8_limits`` asserts that,
so this note stays true if the dump grows.  The KJ=8 arm is instead checked
against Stage A directly in ``test_kj8_limits_vs_stage_a``, which is the
WEAKER claim: two transliterations of the same Fortran agreeing, with no
independent oracle behind it.
"""

import numpy as np
import pytest
import torch

import voacap_gpu.stage_b.noisy as noisy_mod
from voacap_gpu.stage_a.f32math import F32, cos32, sin32
from voacap_gpu.stage_a.noisy import noisy as noisy_stage_a
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_b.anois1 import anois1_batch
from voacap_gpu.stage_b.noisy import limits_for_kj, noisy_batch

from tests.stage_a.dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("anois1")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# ATNU/ATNY only.  Measured worst is 3 (CPU) / 4 (CUDA) floored ULP on a
# quantity in dB above kTb, i.e. tens of dB -- so 12 is a wide margin that
# still catches a real wiring error, which would be orders of magnitude out.
ULP_BAR = 12.0
# CUDA is lower purely because torch's device SIN/COS are different functions
# from its host ones; see the module docstring, where that is measured.
EXACT_FLOOR = {"cpu": 0.80, "cuda": 0.62}       # measured 0.862 / 0.762


def _inputs():
    """The dump as numpy columns, plus the per-lane REDMAP overlays.

    FAKP/FAKABP/F2D depend only on the month (the SSN blend in REDMAP touches
    the ionospheric coefficients, not the noise maps), so twelve REDMAPs cover
    189 lanes.  They are stacked to (29,16,6,N)/(2,6,N) here to exercise the
    per-lane path; a real grid shares one map and passes (29,16,6).
    """
    _, inputs, rows = DUMP
    months = sorted({int(r[0]) for r in inputs})
    maps = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0)) for m in months}
    izf = np.array([int(r[6]) for r in inputs])
    col = lambda i: np.array(  # noqa: E731
        [float(r[i]) for r in inputs], dtype=np.float32)
    cols = dict(
        fakp=np.stack([maps[int(r[0])].fakp for r in inputs], axis=-1),
        fakabp=np.stack([maps[int(r[0])].fakabp for r in inputs], axis=-1),
        # IZF forces the no-data-base arm without disturbing the REDMAP cache
        f2d111=np.array([0.0 if int(r[6])
                         else float(maps[int(r[0])].f2d[0, 0, 0])
                         for r in inputs], dtype=np.float32),
        gmtr=col(2), rlat=col(3), rlong=col(4), rlongd=col(5),
    )
    fields = ["atnu", "atny", "cc", "tm", "kj", "jk"]
    expected = {h: np.array([np.float32(r[i]) for r in rows], dtype=np.float32)
                for i, h in enumerate(fields)}
    return cols, expected, izf


def _run(cols, device):
    t = lambda a: torch.as_tensor(  # noqa: E731
        np.ascontiguousarray(a), device=device)
    return anois1_batch(t(cols["fakp"]), t(cols["fakabp"]), t(cols["f2d111"]),
                        t(cols["gmtr"]), t(cols["rlong"]), t(cols["rlongd"]),
                        t(cols["rlat"]))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_anois1_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_block_selection_is_bitwise(device):
    """CC/TM/KJ/JK are integer/copy logic -- no tolerance is allowed here.

    Only the lanes with a data base are checked: on the early return the
    Fortran leaves the COMMON holding whatever the previous call put there,
    which is a stale value no port can or should reproduce.
    """
    cols, expected, izf = _inputs()
    out = _run(cols, device)
    keep = izf == 0
    assert keep.sum() and (~keep).sum(), "need both arms of the IZF branch"
    for name in ("cc", "tm", "kj", "jk"):
        got = out[name].cpu().numpy().astype(np.float32)[keep]
        exp = expected[name][keep]
        bad = np.flatnonzero(got != exp)
        assert bad.size == 0, (
            f"[{device}] {name} differs on lanes {bad[:5].tolist()}: "
            f"{got[bad[:5]].tolist()} vs {exp[bad[:5]].tolist()} -- this "
            "field is integer/branch logic and must be bit-exact")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_noise_maps_vs_oracle(device):
    """ATNU/ATNY: the NOISY Fourier sums, SIN/COS floor and all."""
    cols, expected, izf = _inputs()
    out = _run(cols, device)
    for name in ("atnu", "atny"):
        got = out[name].cpu().numpy().astype(np.float32)
        exp = expected[name]
        d = np.array([ulp_diff_floored(g, e) for g, e in zip(got, exp)])
        i = int(d.argmax())
        assert d.max() <= ULP_BAR, (
            f"[{device}] {name} lane {i}: {got[i]!r} vs {exp[i]!r} "
            f"({d[i]:.2f} floored ULP)")
        frac = float((d == 0).mean())
        assert frac >= EXACT_FLOOR[device], (
            f"[{device}] {name} exact {frac:.3f}, worst {d.max():.2f} at "
            f"lane {i}")
    # the early return really does zero the noise, exactly
    z = izf != 0
    assert z.sum() == 9, f"{z.sum()} no-data-base lanes; the dump had 9"
    for name in ("atnu", "atny"):
        v = out[name].cpu().numpy()[z]
        assert np.all(v == 0.0), f"[{device}] {name} not zeroed on IZF lanes"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_dump_exercises_every_branch():
    """The batch must genuinely reach every arm, or agreement proves little."""
    cols, expected, izf = _inputs()
    out = _run(cols, "cpu")
    kj = out["kj"].numpy()[izf == 0]
    jk = out["jk"].numpy()[izf == 0]
    cc = out["cc"].numpy()[izf == 0]
    tm = out["tm"].numpy()[izf == 0]
    assert set(kj.tolist()) == set(range(1, 7)), f"KJ only reaches {set(kj)}"
    assert set(jk.tolist()) == set(range(1, 7)), f"JK only reaches {set(jk)}"
    # IF(CC-22.): the truncation arm AND the literal-6 shortcut
    assert (cc >= 22.0).sum() and (cc < 22.0).sum()
    # IF(CC-TM) 115,120,125 -- all three, including the exact-equality middle
    # arm, which sets JK = KJ and collapses GENOIS's interpolation
    assert (cc < tm).sum() and (cc > tm).sum()
    assert (cc == tm).sum() >= 2, "no lane sits exactly on a block centre"
    assert (kj[cc == tm] == jk[cc == tm]).all()
    # both cyclic wraps: JK 0 -> 6 and JK 7 -> 1
    assert ((kj == 1) & (jk == 6)).sum(), "the JK=0 -> 6 wrap is unexercised"
    assert ((kj == 6) & (jk == 1)).sum(), "the JK=7 -> 1 wrap is unexercised"
    # IF(RLONG): west (radians, folded into 0..360) and east (degrees)
    assert (cols["rlong"] < 0).sum() and (cols["rlong"] >= 0).sum()


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_dump_never_reaches_the_kj8_limits():
    """ANOIS1 asks only for blocks 1..6, so LM/LN is always 29/15 here.

    Asserted rather than asserted-in-prose so the docstring's coverage note
    cannot quietly go stale if the dump is regenerated.
    """
    cols, _, izf = _inputs()
    out = _run(cols, "cpu")
    blocks = set(out["kj"].tolist()) | set(out["jk"].tolist())
    assert blocks <= set(range(1, 7))
    assert {limits_for_kj(int(b)) for b in blocks} == {(29, 15)}


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_shared_map_matches_per_lane_map(device):
    """The (29,16,6) grid-shared form and the (29,16,6,N) per-lane form are
    the same arithmetic, and must be bit-identical.

    This is the API guarantee that lets a real area run avoid materialising a
    per-lane copy of the coefficient map -- 1.8 GB at a million grid points.
    It falls out of broadcasting, which is exactly why it is worth pinning:
    nothing would complain if a future edit forced the lane axis.
    """
    cols, _, _ = _inputs()
    st = redmap(COEFFS_DIR, "CCIR", 3, F32(100.0))
    n = len(cols["gmtr"])
    shared = dict(cols, fakp=st.fakp, fakabp=st.fakabp,
                  f2d111=np.float32(st.f2d[0, 0, 0]))
    tiled = dict(shared,
                 fakp=np.repeat(st.fakp[..., None], n, axis=-1),
                 fakabp=np.repeat(st.fakabp[..., None], n, axis=-1),
                 f2d111=np.full(n, st.f2d[0, 0, 0], dtype=np.float32))
    a = _run(shared, device)
    b = _run(tiled, device)
    for name in a:
        assert torch.equal(a[name], b[name]), f"[{device}] {name} differs"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_lanes_are_independent(device):
    """A lane's answer must not depend on which other lanes share its batch.

    Obvious from the source -- every operation here is elementwise and NOTHING
    reduces across the lane axis -- and precisely because it is obvious it is
    the kind of thing a future edit breaks without anyone looking.  It is also
    the premise the work-sharing in ``stage_b.dedupe`` rests on: ANOIS1 is
    keyed by (receiver site, receiver mean time), so a driver may evaluate the
    distinct sites as a smaller batch and read the answers back, and that is
    only sound if a smaller batch computes the same bits.

    Asserted BITWISE on every field, on an arbitrary scattered subset rather
    than a prefix, so a reduction that happened to be a no-op at the edges
    would still show.
    """
    cols, _, _ = _inputs()
    n = len(cols["gmtr"])
    pick = np.arange(3, n, 7)
    assert len(pick) > 10
    full = _run(cols, device)
    sub = _run({k: (v[..., pick] if k in ("fakp", "fakabp") else v[pick])
                for k, v in cols.items()}, device)
    for name in full:
        got = sub[name]
        exp = full[name][torch.as_tensor(pick, device=got.device)]
        assert torch.equal(got, exp), (
            f"[{device}] {name} depends on the batch it is evaluated in: "
            f"{int((got != exp).sum())}/{len(pick)} lanes differ")


def test_kj8_limits_vs_stage_a():
    """The LM=15/LN=10 arm, against Stage A -- the WEAKER claim.

    No oracle dump reaches it (see above), so this is two transliterations of
    the same Fortran agreeing with each other rather than an independent
    check.  It is still worth having: it pins that the truncated series reads
    the CONSTANT column P(J,16) and not P(J,LN+1), which is the one thing the
    shorter limits make easy to get wrong.
    """
    st = redmap(COEFFS_DIR, "CCIR", 3, F32(100.0))
    rng = np.random.default_rng(7)
    xla = rng.uniform(-89.0, 89.0, 40).astype(np.float32)
    ceg = rng.uniform(0.0, 360.0, 40).astype(np.float32)
    lm, ln = limits_for_kj(8)
    assert (lm, ln) == (15, 10)
    ref = np.array([noisy_stage_a(st, 8, x, c) for x, c in zip(xla, ceg)],
                   dtype=np.float32)
    for device in DEVICES:
        t = lambda a: torch.as_tensor(  # noqa: E731
        np.ascontiguousarray(a), device=device)
        got = noisy_batch(t(st.hmym), t(np.float32(st.abmap[0, 1])),
                          t(np.float32(st.abmap[1, 1])),
                          t(xla), t(ceg), lm, ln).cpu().numpy()
        d = np.array([ulp_diff_floored(g, e) for g, e in zip(got, ref)])
        assert d.max() <= 8.0, f"[{device}] KJ=8 worst {d.max():.2f} ULP"
        # measured 38/40 CPU, 34/40 CUDA -- the SIN/COS floor again
        assert (d == 0).mean() >= 0.70, (
            f"[{device}] KJ=8 exact {(d == 0).mean():.3f}")


@pytest.mark.parametrize("device", DEVICES)
def test_series_order_is_exact_given_stage_a_sincos(monkeypatch, device):
    """Swap torch's SIN/COS for Stage A's and the whole residual disappears.

    This is what turns "the difference is libm" from an excuse into a
    measurement.  The angle-addition recurrence, the 435-term accumulation
    order, the P(J,16) constant column and the ALF + BET*Q normalisation all
    have to be exactly right for this to come out bit-identical -- reassociate
    the inner sum, or use a dot product, and it will not.
    """
    st = redmap(COEFFS_DIR, "CCIR", 3, F32(100.0))
    rng = np.random.default_rng(1)
    xla = rng.uniform(-89.0, 89.0, 60).astype(np.float32)
    ceg = rng.uniform(0.0, 360.0, 60).astype(np.float32)

    def _apply(fn):
        def wrapped(t):
            v = np.array([fn(x) for x in t.detach().cpu().numpy().ravel()],
                         dtype=np.float32).reshape(t.shape)
            return torch.as_tensor(v, device=t.device)
        return wrapped

    monkeypatch.setattr(noisy_mod, "_sin", _apply(sin32))
    monkeypatch.setattr(noisy_mod, "_cos", _apply(cos32))

    t = lambda a: torch.as_tensor(  # noqa: E731
        np.ascontiguousarray(a), device=device)
    maps = ((1, st.fakp[:, :, 0], (st.fakabp[0, 0], st.fakabp[1, 0])),
            (8, st.hmym, (st.abmap[0, 1], st.abmap[1, 1])))
    for kj, p, ab in maps:
        lm, ln = limits_for_kj(kj)
        ref = np.array([noisy_stage_a(st, kj, x, c) for x, c in zip(xla, ceg)],
                       dtype=np.float32)
        got = noisy_batch(t(p), t(np.float32(ab[0])), t(np.float32(ab[1])),
                          t(xla), t(ceg), lm, ln).cpu().numpy()
        assert np.array_equal(got, ref), (
            f"[{device}] KJ={kj}: {(got != ref).sum()}/{len(ref)} lanes "
            "differ with Stage A's own SIN/COS in place -- the residual is "
            "NOT just libm; the series order or a constant is wrong")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """ONE set of CPU-built inputs fed to both devices, so nothing upstream
    can be mistaken for this routine's own difference -- ANOIS1 has no
    upstream anyway (REDMAP is a file read), which makes this a clean read on
    torch's host vs device SIN/COS."""
    cols, _, _ = _inputs()
    a = _run(cols, "cpu")
    b = _run(cols, "cuda")
    for name in ("cc", "tm", "kj", "jk"):
        assert torch.equal(a[name], b[name].cpu()), (
            f"{name} differs across devices; it is integer/branch logic and "
            "contains no transcendental call")
    for name, floor in (("atnu", 0.60), ("atny", 0.60)):
        x = a[name].numpy()
        y = b[name].cpu().numpy()
        sp = np.spacing(np.maximum(np.abs(y), np.float32(1.0)))
        d = np.abs(x.astype(np.float64) - y.astype(np.float64)) / sp
        assert d.max() <= ULP_BAR, f"{name} worst {d.max():.2f} ULP"
        assert (x == y).mean() >= floor, (
            f"only {(x == y).mean():.3f} of {name} is bit-equal across "
            "devices; measured 0.746/0.788 and it is SIN/COS alone")
