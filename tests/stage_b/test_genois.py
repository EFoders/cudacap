"""Stage B batched GENOIS vs the SAME tier-B oracle dump Stage A uses.

MEASURED: 570 cases x 21 columns = 11,970 cells, ALL BIT-EXACT, on CPU AND ON
CUDA.  Worst floored ULP 0.0 in every field on both devices.  That is Stage A's
result reproduced exactly, and it is worth stating plainly for the same reason
it was worth stating there: the routine is twelve EXPs, three powers of ten,
two natural logs and two square roots deep, it mixes REAL*4 and REAL*8 at
nearly every step, and getting zero drift out of it means three things are
modelled right that would each have shown up as a nonzero residual otherwise --

* which quantities are REAL*8 (A, V and the sigmas) and which are not; every
  EXP feeding the sums is REAL*4 and only its RESULT widens,
* ``**2`` is an INTEGER exponent (a multiply) while ``10.**x`` is a REAL one
  (powf); routing the first through a pow drifts,
* ``4.34294`` and ``4.34294d0`` are different constants and GENOIS uses BOTH.

CUDA matching CPU bitwise is a fourth thing, and it is not luck: the REAL*4
transcendentals are evaluated in float64 and rounded once, exactly as Stage A's
``f32math`` does, so torch's f32 libm -- which is what puts ALOSFV 2 ULP apart
across devices -- never enters.  The ULP bars below therefore exist only as a
safety valve so a future libm change reports as a small drift rather than a
hard failure; the exactness floors are what actually guard the arithmetic.

THE DUMP IS REPLAYED AS TWO BATCHES, NOT ONE, and that is forced rather than
chosen.  The receive antenna is a property of the CIRCUIT, not of the grid
point -- one coverage run is one transmitter and one receiver -- so
``BatchedCantenna`` is a single table shared by every lane.  The dump varies
the receiver band across its cases (2-30 MHz on 477 of them, 12-18 MHz on the
other 93, which is how the no-matching-entry arm gets exercised), so it splits
into exactly two antennas and therefore two batches.  Every other branch is
live inside each of them; the coverage assertions are on the union.

Fields asserted BIT-EXACT unconditionally because they are copies or integers
rather than computed: ZNOISE (a function of the NOISE card alone), the six EFF
slots (GAIN's output broadcast), METHOD_USED and MODE_N2.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genois import OldNoiseModel
from voacap_gpu.stage_a.readant import read_antenna
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_b.gain import BatchedCantenna
from voacap_gpu.stage_b.genois import genois_batch

from tests.stage_a.dumputil import (ANTENNAS_DIR, COEFFS_DIR, load_dump,
                                    ulp_diff_floored)

DUMP = load_dump("genois")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

ULP_BAR = 4.0                   # measured worst 0.0 -- a libm safety valve
EXACT_FLOOR = {"cpu": 0.999, "cuda": 0.999}     # measured 1.000000 on both
# copies and flags: no arithmetic in their derivation, so no tolerance
BITEXACT = {"znoise", "method_used", "mode_n2",
            "eff1", "eff2", "eff3", "eff4", "eff5", "eff6", "eff7"}


def _cantenna(rxlo, rxhi, device):
    """sample.14 as /cantenna/ entry 2, the receiver, over [rxlo, rxhi]."""
    ant = read_antenna(ANTENNAS_DIR / "sample.14")
    cant = CantennaState()
    cant.numants = 2
    cant.iats[1] = 1
    cant.iats[2] = 2
    cant.xfqs[1], cant.xfqe[1] = F32(2.0), F32(30.0)
    cant.xfqs[2], cant.xfqe[2] = F32(rxlo), F32(rxhi)
    cant.offazim[1] = cant.offazim[2] = F32(0.0)
    array, aeff = ant.p2p_tables()
    cant.array[1:31, 1:92, 2] = array
    cant.aeff[1:31, 2] = aeff
    return BatchedCantenna.from_stage_a(cant, device)


def _empty_cantenna(device):
    """No entries at all: GAIN matches nothing and returns a 0 dB isotrope."""
    return BatchedCantenna.from_stage_a(CantennaState(), device)


def _groups():
    """The dump split by receive-antenna band -- see the module docstring."""
    _, inputs, _ = DUMP
    bands = sorted({(float(r[12]), float(r[13])) for r in inputs})
    return [(b, [i for i, r in enumerate(inputs)
                 if (float(r[12]), float(r[13])) == b]) for b in bands]


def _run_group(idx, device):
    """One band's cases as one batch."""
    _, inputs, _ = DUMP
    rr = [inputs[i] for i in idx]
    sts = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0))
           for m in sorted({int(r[0]) for r in rr})}

    def col(c, dtype=np.float32):
        return torch.as_tensor(
            np.array([dtype(r[c]) if dtype is not np.int64
                      else np.int64(int(r[c])) for r in rr], dtype=dtype),
            device=device)

    fam = torch.as_tensor(np.stack([sts[int(r[0])].fam for r in rr], -1),
                          device=device)
    dud = torch.as_tensor(np.stack([sts[int(r[0])].dud for r in rr], -1),
                          device=device)
    band = (float(rr[0][12]), float(rr[0][13]))
    # month,izf,freq,noise,rlat,atnu,atny,cc,tm,kj,jk,fi3,rxlo,rxhi
    return genois_batch(
        fam, dud, col(1, np.int64).bool(), col(9, np.int64),
        col(10, np.int64), col(5), col(6), col(7), col(8), col(4), col(2),
        col(3, np.int64), col(11), _cantenna(*band, device))


def _whole_dump(device):
    """Both batches, re-ordered back to the dump's own case order."""
    order, parts = [], []
    for _, idx in _groups():
        order += idx
        parts.append(_run_group(idx, device))
    keys = parts[0].keys()
    got = {k: torch.cat([p[k] for p in parts]).cpu().numpy() for k in keys}
    inv = np.argsort(np.array(order))
    return {k: v[inv] for k, v in got.items()}


def _column(got, name):
    """The dump column `name` out of the returned dict.

    EFF 1..6 are one value broadcast; slot 7 belongs to ESMOD and GENOIS never
    writes it, so the oracle's own eff7 is the untouched preset.
    """
    if name == "eff7":
        return None
    if name.startswith("eff"):
        return got["eff"]
    if name in ("method_used", "mode_n2"):
        return got[name].astype(np.float32)
    return got[name]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genois_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_genois_batch_vs_oracle(device):
    header, inputs, rows = DUMP
    got = _whole_dump(device)
    expected = np.array([[np.float32(c) for c in r] for r in rows],
                        dtype=np.float32)

    total = exact = 0
    worst, where = 0.0, ""
    for c, name in enumerate(header):
        g = _column(got, name)
        e = expected[:, c]
        if g is None:                       # eff7: GENOIS must not write it
            assert (e == 0.0).all(), (
                "the oracle's EFF(7) is not its preset, so GENOIS is writing "
                "past the six mode slots it owns")
            continue
        for i in range(len(e)):
            if name in BITEXACT:
                assert g[i] == e[i], (
                    f"[{device}] case {i} {name}: {g[i]!r} vs {e[i]!r} "
                    f"(must be bit-exact) in {inputs[i]}")
                total += 1
                exact += 1
                continue
            d = ulp_diff_floored(g[i], e[i])
            if d > worst:
                worst, where = d, f"case {i} {name} {inputs[i]}"
            assert d <= ULP_BAR, (
                f"[{device}] case {i} {name}: {g[i]!r} vs {e[i]!r} "
                f"({abs(float(g[i]) - float(e[i])):.3e} abs, "
                f"{d:.2f} floored ULP) in {inputs[i]}")
            total += 1
            exact += d == 0.0
    assert exact / total >= EXACT_FLOOR[device], (
        f"[{device}] exact {exact}/{total}, worst {worst:.2f} at {where}")

    # ---- coverage, over the union of the two batches ---------------------
    izf = np.array([int(r[1]) for r in inputs])
    noise = np.array([int(r[3]) for r in inputs])
    seen = {
        "nodata": int((izf != 0).sum()),
        "database": int((izf == 0).sum()),
        "man_level": int((noise > 0).sum()),
        "man_default": int((noise == 0).sum()),
        "man_index": int((noise < 0).sum()),
        "caruana": int((got["method_used"] == 1).sum()),
        "spaulding": int((got["method_used"] == 2).sum()),
        "no_galactic": int((got["mode_n2"] == 0).sum()),
        "galactic": int((got["mode_n2"] == 1).sum()),
        "no_ant": int((got["eff"] == 0).sum()),
        "ant": int((got["eff"] != 0).sum()),
    }
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    # Caruana's test is entered on 79 lanes and caps a sigma on 66 of them;
    # the other 13 find DXX2/DXX3 non-positive and leave Spaulding alone.
    # Both sub-arms are therefore live, which is what makes the > 12 dB
    # bound and the > 0 bound separable.
    assert 60 <= seen["caruana"] <= 75, (
        f"{seen['caruana']} capped lanes; measured 66 -- if this moved, the "
        "Caruana bound or the > 0 guard has shifted, not the arithmetic")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_old_noise_model_is_refused():
    """NYEAR ' 2089' selects GENOIS_OLD -- superseded physics, not ported.

    The same exception class Stage A raises, so a caller cannot tell the two
    stages apart by how they refuse.
    """
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    one = torch.zeros(1)
    with pytest.raises(OldNoiseModel):
        genois_batch(torch.as_tensor(st.fam), torch.as_tensor(st.dud),
                     torch.zeros(1, dtype=torch.bool),
                     torch.ones(1, dtype=torch.long),
                     torch.ones(1, dtype=torch.long),
                     one, one, one, one, one, one + 10.0,
                     torch.full((1,), -2, dtype=torch.long), one,
                     _empty_cantenna("cpu"), nyear=" 2089")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_galactic_noise_is_dropped_below_the_f2_critical(device):
    """Galactic noise cannot penetrate at or below foF2, so it is zeroed --
    and the boundary is INCLUSIVE.

    All three frequencies in one batch against the same foF2, which is the
    arrangement that catches a mask resolved once for the batch.
    """
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    freq = np.array([8.99, 9.0, 9.01], dtype=np.float32)
    n = 3
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    f = lambda v: t(np.full(n, v, np.float32))       # noqa: E731
    out = genois_batch(
        t(st.fam), t(st.dud), t(np.zeros(n, bool)),
        t(np.full(n, 3, np.int64)), t(np.full(n, 3, np.int64)),
        f(70.0), f(68.0), f(10.0), f(10.0), f(0.5), t(freq),
        t(np.full(n, -2, np.int64)), f(9.0), _empty_cantenna(device))
    mode = out["mode_n2"].cpu().numpy()
    gnos = out["gnos"].cpu().numpy()
    assert mode.tolist() == [0, 0, 1], f"mode_n2 {mode.tolist()}"
    assert gnos[0] == np.float32(-204.0) and gnos[1] == np.float32(-204.0)
    assert gnos[2] != np.float32(-204.0)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_no_data_base_uses_the_fixed_deciles(device):
    """F2D(1,1,1) < 0 skips GENFAM: the atmospheric level vanishes but its
    deciles are still the fixed 9/7 dB, so the spread does not collapse.

    Both arms in one batch.  GENFAM is still evaluated on the no-data lanes
    and thrown away -- it is pure arithmetic and cannot produce a NaN to leak
    through the select -- so this also checks the discarded arm stays
    discarded.
    """
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    f = lambda v: t(np.full(2, v, np.float32))       # noqa: E731
    out = genois_batch(
        t(st.fam), t(st.dud), t(np.array([True, False])),
        t(np.full(2, 3, np.int64)), t(np.full(2, 3, np.int64)),
        f(70.0), f(68.0), f(10.0), f(10.0), f(0.5), f(10.0),
        t(np.full(2, -2, np.int64)), f(4.0), _empty_cantenna(device))
    at = out["xn_atnos"].cpu().numpy()
    assert at[0] == np.float32(-204.0), "no-data lane is not 0 dB above kTb"
    assert at[1] != np.float32(-204.0), "the data-base lane collapsed too"
    assert out["du"].cpu().numpy()[0] > 0 and out["dl"].cpu().numpy()[0] > 0
    # 9 and 7 are both under 12, so Spaulding stands on the no-data lane
    assert out["method_used"].cpu().numpy()[0] == 2


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_man_made_categories_clamp_at_four(device):
    """MIN0(4,MA) makes CCIR categories 5 and 6 UNREACHABLE.

    The DATA statements carry six entries but NOISE < -4 is folded onto the
    'remote' category, so ``_XNINT[4:]``/``_CONN[4:]`` are dead table.  This
    asserts they stay dead -- if the clamp were dropped, NOISE = -5 would pick
    up the 83.2 dB 'noisy' level and diverge by 30 dB.  NOISE = 0 is the
    -164 dBW default, which lands on category 4 by a different path (label
    114, not the clamp), so it must agree as well.
    """
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    noise = np.array([-4, -5, -6, -9, 0, -1], dtype=np.int64)
    n = len(noise)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    f = lambda v: t(np.full(n, v, np.float32))       # noqa: E731
    out = genois_batch(
        t(st.fam), t(st.dud), t(np.zeros(n, bool)),
        t(np.full(n, 3, np.int64)), t(np.full(n, 3, np.int64)),
        f(70.0), f(68.0), f(10.0), f(10.0), f(0.5), f(10.0),
        t(noise), f(4.0), _empty_cantenna(device))
    x = out["xnoise"].cpu().numpy()
    z = out["znoise"].cpu().numpy()
    assert (x[:5] == x[0]).all(), (
        f"categories beyond 4 are not clamped: xnoise {x.tolist()}")
    assert (z[:5] == z[0]).all(), f"znoise {z.tolist()}"
    assert x[5] != x[0], "category 1 must differ from category 4"


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """Inputs built ONCE on CPU and moved, so this isolates GENOIS's own
    arithmetic from anything upstream.  BITWISE on every field, deliberately:
    the REAL*4 transcendentals go through float64 and round once, so there is
    no f32 libm difference to absorb and a tolerance would hide a real split.
    """
    _, idx = _groups()[0]
    a = _run_group(idx, "cpu")
    b = _run_group(idx, "cuda")
    for name in a:
        x, y = a[name], b[name].cpu()
        assert bool((x == y).all()), (
            f"{name}: {int((x != y).sum())} lanes differ across devices")
