"""Stage B leaf routines the rest of the freq loop is built on.

XLIN, FNORML, PRBMUF, GAIN (both branches) and PWRDB.  They are shared by
SIGDIS, REGMOD, INMUF, ESMOD, GENOIS and RELBIL, so they land ahead of all of
them and are pinned here rather than inside each caller's test.

WHERE AN ORACLE DUMP EXISTS IT IS USED (gain, gainant, prbmuf); XLIN and
FNORML have none, so they are compared against Stage A, which does have one
via its callers.  That is the weaker claim and is labelled as such -- it shows
the two transliterations agree, not that both are right.

Almost all of this is bit-exact by construction: table interpolation and a
polynomial, no libm, no cancellation.  The one exception is GAIN's ground
branch, which is COS/SIN/ATAN/ASIN/ALOG end to end.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.fnorml import fnorml
from voacap_gpu.stage_a.readant import read_antenna
from voacap_gpu.stage_a.xlin import xlin
from voacap_gpu.stage_b.gain import (BatchedCantenna, antenna_gain_batch,
                                     ground_loss_batch, pwrdb_batch)
from voacap_gpu.stage_b.normal import fnorml_batch, prbmuf_batch
from voacap_gpu.stage_b.xlin import xlin_batch

from tests.stage_a.dumputil import ANTENNAS_DIR, load_dump, ulp_diff_floored

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GAIN = load_dump("gain")
GAINANT = load_dump("gainant")
PRBMUF = load_dump("prbmuf")
SAMPLE = ANTENNAS_DIR / "sample.14"


# --------------------------------------------------------------------- XLIN

def _tables(rng, n, ntab=30):
    """Tables that are deliberately NOT monotone.

    XLIN exists because the ionogram has a valley in it, so a table that only
    rises would exercise one of its three segment kinds.  These rise, fall and
    hold flat, and every lane gets a different shape.
    """
    xn = np.zeros((ntab, n), dtype=np.float32)
    yn = np.zeros((ntab, n), dtype=np.float32)
    for i in range(n):
        base = np.cumsum(rng.uniform(-1.0, 2.0, ntab)).astype(np.float32)
        base[i % ntab] = base[max(i % ntab - 1, 0)]     # a flat segment
        xn[:, i] = base
        yn[:, i] = (base * np.float32(1.7) + np.float32(i)).astype(np.float32)
    return xn, yn


@pytest.mark.parametrize("device", DEVICES)
def test_xlin_batch_matches_stage_a(device):
    rng = np.random.default_rng(20260723)
    n, ntab = 96, 30
    xn, yn = _tables(rng, n, ntab)
    # spread the probe across, below and above the table so both endpoint
    # holds are hit as well as the interior
    x = np.linspace(-3.0, 25.0, n, dtype=np.float32)

    got = xlin_batch(*[torch.as_tensor(a, device=device)
                       for a in (x, xn, yn)], ntab).cpu().numpy()

    below = above = interior = 0
    for i in range(n):
        xn1 = np.zeros((ntab + 1, 2), dtype=np.float32)
        yn1 = np.zeros((ntab + 1, 2), dtype=np.float32)
        xn1[1:, 1], yn1[1:, 1] = xn[:, i], yn[:, i]
        want = xlin(x[i], xn1, yn1, ntab, 1)
        assert got[i] == np.float32(want), (
            f"[{device}] lane {i}: {got[i]!r} vs Stage A {want!r}")
        below += want == yn[0, i]
        above += want == yn[ntab - 1, i]
        interior += want not in (yn[0, i], yn[ntab - 1, i])
    assert below and above and interior, (
        f"coverage: {below} below the table, {above} above, {interior} "
        "interior -- all three must occur or the clamps are untested")


def test_xlin_holds_the_endpoints_rather_than_extrapolating():
    """Outside the table the Fortran returns YN(1) or YN(NTAB) flat.  Getting
    this wrong is invisible in the interior and wrong by a lot at the ends."""
    xn = torch.tensor([[1.0], [2.0], [3.0]])
    yn = torch.tensor([[10.0], [20.0], [30.0]])
    assert float(xlin_batch(torch.tensor([-5.0]), xn, yn)) == 10.0
    assert float(xlin_batch(torch.tensor([99.0]), xn, yn)) == 30.0
    assert float(xlin_batch(torch.tensor([2.5]), xn, yn)) == 25.0


# ------------------------------------------------------------------- FNORML

@pytest.mark.parametrize("device", DEVICES)
def test_fnorml_batch_matches_stage_a(device):
    """No libm, only the Hastings quartic, so bit-exact is the right bar.
    The sweep runs past the +-5 sigma clamp on both sides."""
    z = np.linspace(-8.0, 8.0, 401, dtype=np.float32)
    got = fnorml_batch(torch.as_tensor(z, device=device)).cpu().numpy()
    for i, zi in enumerate(z):
        assert got[i] == np.float32(fnorml(zi)), (
            f"[{device}] z={zi}: {got[i]!r} vs Stage A {fnorml(zi)!r}")
    assert (got[:20] == got[0]).all() and (got[-20:] == got[-1]).all(), (
        "the 5-sigma clamp is not holding flat")


# ------------------------------------------------------------------- PRBMUF

@pytest.mark.skipif(PRBMUF is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_prbmuf_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_prbmuf_batch_vs_oracle(device):
    _, inputs, rows = PRBMUF
    cols = np.array([[float(x) for x in inp] for inp in inputs],
                    dtype=np.float32).T
    fmhz, fgo, fset, _il, sigl, sigu = cols
    t = [torch.as_tensor(c, device=device) for c in (fmhz, fgo, fset, sigl,
                                                     sigu)]
    got = prbmuf_batch(*t).cpu().numpy()
    want = np.array([np.float32(r[0]) for r in rows], dtype=np.float32)
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"[{device}] {len(bad)} of {len(want)} differ; first case "
        f"{bad[0][0]}: {got[bad[0][0]]!r} vs {want[bad[0][0]]!r}")
    assert int((fset <= 0).sum()) >= 1, (
        "no case has FSET <= 0, so the step-at-FGO degenerate arm is untested")


# --------------------------------------------------------------------- GAIN

@pytest.mark.skipif(GAIN is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_ground_loss_batch_vs_oracle(device):
    """COS/SIN/ATAN/ASIN/ALOG end to end, so this is the one leaf with a
    genuine floor -- five different libm functions feeding one LOG of a sum of
    squares.  Stage A measures 98.4% exact / 2 ULP against this same dump
    because its cos32/sin32/atan32/asin32/log32 are matched to this gfortran
    build; torch's are five different functions, and the result is 330/374
    (88.2%) on CPU and 327/374 on CUDA, worst 5 ULP.  A dB of ground loss is
    quoted to a tenth, so 5 ULP at ~2 dB is nine orders of magnitude spare."""
    _, inputs, rows = GAIN
    cols = np.array([[float(x) for x in inp] for inp in inputs],
                    dtype=np.float32).T
    _ig, sigma, er, delta, fmc = cols
    got = ground_loss_batch(*[torch.as_tensor(c, device=device)
                              for c in (sigma, er, delta, fmc)]).cpu().numpy()
    want = np.array([np.float32(r[0]) for r in rows], dtype=np.float32)

    worst = max(ulp_diff_floored(g, w) for g, w in zip(got, want))
    exact = int((got == want).sum())
    assert worst <= 32.0, f"[{device}] worst {worst:.1f} floored ULP"
    assert exact / len(want) >= 0.85, (       # measured 0.882 / 0.874
        f"[{device}] exact {exact}/{len(want)}, worst {worst:.1f} ULP")
    assert int((sigma <= 0).sum()) >= 1 and (got[sigma <= 0] == 0).all(), (
        "the 'no ground data' arm must return GAIN's preset 0")
    assert int((delta <= 1e-8).sum()) >= 1, (
        "no grazing case, so the flat 6 dB substitution is untested")


def _cantenna():
    """The three entries dump_gainant.f installs, built the same way."""
    array, aeff = read_antenna(SAMPLE).p2p_tables()
    c = CantennaState()
    for iat in (1, 2, 3):
        c.array[1:31, 1:92, iat] = array
        c.aeff[1:31, iat] = aeff
        c.offazim[iat] = np.float32(0.0)
    c.array[1:31, 1:92, 3] = array + np.float32(3.0)
    c.aeff[1:31, 3] = aeff + np.float32(0.5)
    c.numants = 3
    c.iats[1], c.xfqs[1], c.xfqe[1] = 1, np.float32(2.0), np.float32(15.0)
    c.iats[2], c.xfqs[2], c.xfqe[2] = 2, np.float32(2.0), np.float32(30.0)
    c.iats[3], c.xfqs[3], c.xfqe[3] = 1, np.float32(15.0), np.float32(30.0)
    c.pwrdba[1], c.pwrdba[2], c.pwrdba[3] = (np.float32(40.0),
                                             np.float32(0.0),
                                             np.float32(47.0))
    return c


@pytest.mark.skipif(GAINANT is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_antenna_gain_batch_vs_oracle(device):
    """Multiply-add over a table -- no transcendental, no cancellation -- so
    bit-exact is the right bar on both devices.

    The dump mixes ITR=1 and ITR=2 cases; the terminal is a property of the
    CALL rather than of a lane, so they are run as two batches.  Which is the
    honest shape: an area run asks for the transmitter gain at every grid
    point, not for a different terminal per point.
    """
    _, inputs, rows = GAINANT
    cant = BatchedCantenna.from_stage_a(_cantenna(), device)
    cols = np.array([[float(x) for x in inp] for inp in inputs],
                    dtype=np.float32)
    want = np.array([[np.float32(r[0]), np.float32(r[1])] for r in rows],
                    dtype=np.float32)

    matched = 0
    for itr in (1, 2):
        sel = cols[:, 0].astype(int) == itr
        if not sel.any():
            continue
        delta = torch.as_tensor(cols[sel, 1], device=device)
        fmc = torch.as_tensor(cols[sel, 2], device=device)
        rain, eff = antenna_gain_batch(cant, itr, delta, fmc)
        rain, eff = rain.cpu().numpy(), eff.cpu().numpy()
        assert np.array_equal(rain, want[sel, 0]), (
            f"[{device}] ITR={itr} rain differs on "
            f"{int((rain != want[sel, 0]).sum())} of {int(sel.sum())} cases")
        assert np.array_equal(eff, want[sel, 1]), (
            f"[{device}] ITR={itr} eff differs")
        matched += int((eff != 0).sum())
    assert matched > len(rows) // 2, (
        f"only {matched} cases selected an antenna; a grid that matches "
        "nothing passes trivially")


@pytest.mark.parametrize("device", DEVICES)
def test_gain_refuses_the_index_the_fortran_reads_out_of_bounds(device):
    """GAIN clamps only the TOP of each axis, so FMC < 1 or DELTA < 0 reads
    ARRAY below its first element and picks up the previous antenna's table.
    INMUF is known to generate exactly that.  A negative index in torch would
    silently wrap to the far end of the axis, which is worse, so this raises.
    """
    cant = BatchedCantenna.from_stage_a(_cantenna(), device)
    ok = torch.tensor([0.1, 0.2], device=device)
    fmc = torch.tensor([10.0, 12.0], device=device)
    antenna_gain_batch(cant, 1, ok, fmc)                    # baseline: fine
    with pytest.raises(IndexError, match="outside the table"):
        antenna_gain_batch(cant, 1, torch.tensor([0.1, -0.5], device=device),
                           fmc)


@pytest.mark.parametrize("device", DEVICES)
def test_a_lane_no_antenna_covers_is_zero_rather_than_a_crash(device):
    """The batched early return is a select AFTER the fact, so the gather has
    to survive lanes it is going to discard.

    The scalar routine returns 0 dB on `iat == 0` before it computes an index
    at all, so it never sees this.  Batched, a lane at 55 MHz -- which GENOIS
    genuinely asks for, and which no honest antenna entry covers -- still has
    its frequency index computed, and 55 indexes ARRAY(30,...) off the end.
    Clamping the DISCARDED lanes is the fix; the matched ones still raise.
    """
    cant = BatchedCantenna.from_stage_a(_cantenna(), device)
    fmc = torch.tensor([10.0, 55.0, 31.0, 0.5], device=device)
    delta = torch.tensor([0.2, 0.2, 0.2, 0.2], device=device)
    rain, eff = antenna_gain_batch(cant, 1, delta, fmc)
    assert float(rain[0]) != 0.0, "the in-band lane should still have gain"
    assert list(rain[1:].cpu().numpy()) == [0.0, 0.0, 0.0], (
        "lanes no entry covers must read as GAIN's preset 0 dB")
    assert list(eff[1:].cpu().numpy()) == [0.0, 0.0, 0.0]


# -------------------------------------------------------------------- PWRDB

@pytest.mark.parametrize("device", DEVICES)
def test_pwrdb_batch_picks_the_transmitter_entry_covering_the_frequency(device):
    """The 2..30 MHz clamp applies to the RANGE TEST ONLY, and a miss falls
    back to 30 dBW rather than to zero -- the opposite of GAIN's convention,
    which is exactly the kind of thing a shared helper gets wrong."""
    cant = BatchedCantenna.from_stage_a(_cantenna(), device)
    f = torch.tensor([1.0, 5.0, 15.0, 20.0, 50.0], device=device)
    got = pwrdb_batch(cant, f).cpu().numpy()
    # entry 1 covers 2..15 (40 dBW), entry 3 covers 15..30 (47); entry 2 is
    # the receiver and must never be selected even though it covers 2..30
    assert list(got) == [40.0, 40.0, 40.0, 47.0, 47.0], list(got)
    empty = BatchedCantenna.from_stage_a(CantennaState(), device)
    assert float(pwrdb_batch(empty, torch.tensor([10.0], device=device))) \
        == 30.0, "a miss must fall back to 1 kW, not to 0"
