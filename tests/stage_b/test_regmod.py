"""Stage B batched REGMOD vs the SAME tier-B oracle dump Stage A uses.

All 262 cases go in as ONE batch, one lane each.  Every input is INJECTED from
the dump rather than recomputed, exactly as Stage A's tier-B does: REGMOD
interpolates no upstream quantised table, so the FINDF/IFOB replay rule does
not apply and injection buys branch control instead -- D-E against F-layer
dispatch and the D-E collision-frequency sub-branch, Es obscuration present
and absent, the low-MUFday extra loss, 1-hop through 8-hop paths, one to five
sample areas, dead mode slots, and a frequency outside the transmitter's range.

THE BARS ARE PER FIELD, because the residual has two distinct shapes and a
single bar would hide both.  Measured over 36,680 cells (7 slots x 262 lanes x
20 fields), CPU 36097/36680 = 98.41% bit-exact, CUDA 34618/36680 = 94.38%:

    field    cpu exact  worst ULP  worst abs | cuda exact  worst ULP  worst abs
    b eff hn hp nmode rgain tgain      EXACT, every cell, on both devices
    prob       99.95%       0.01   1.3e-09   |   99.84%      26.00   3.1e-06
    tllow      99.84%     225.00   4.3e-04   |   99.62%     224.00   4.3e-04
    tlhgh      99.89%      59.00   1.1e-04   |   99.73%       8.00   2.9e-06
    obf        99.95%     119.00   5.7e-05   |   80.43%       1.00   4.8e-07
    fslos      98.42%       4.00   3.1e-05   |   90.19%       2.00   1.5e-05
    fldst      97.16%     728.00   5.6e-03   |   89.64%     320.00   8.5e-04
    sigpow     97.16%     364.00   5.6e-03   |   90.73%      56.00   8.5e-04
    sn         97.16%    5824.00   5.6e-03   |   90.73%     256.00   8.5e-04
    tloss      97.16%     364.00   5.6e-03   |   90.51%      34.00   8.5e-04
    timed      96.07%      39.00   1.1e-04   |   86.86%      24.00   5.0e-05
    adv        95.42%       3.00   1.9e-06   |   89.26%       3.00   3.8e-06
    abps       95.26%      63.00   6.0e-05   |   90.08%      19.00   1.8e-05
    grlos      94.77%       5.00   9.5e-07   |   89.97%       6.00   9.5e-07

WHY BOTH BARS ARE NEEDED IS VISIBLE IN THAT TABLE.  SN's worst cell is 5824
floored ULP and 5.6e-3 dB at the same time: it lands at -15 dB after being
built from ~200 dB terms, so the relative metric explodes while the absolute
error is a thousandth of a decibel.  Conversely OBF on CUDA is only 80% exact
yet never wrong by more than 4.8e-7 dB -- lots of last-bit noise, no drift.
Either bar alone would tell a false story.

TWO CANCELLATIONS PRODUCE THE WHOLE RESIDUAL, and they are separable:

* ``1 - COS(PSI)`` in PATH.  At a 700 km hop that leaves 0.0015 from operands
  near 1, a ~660x magnification of COS's last bit, and RZ*(1-COS PSI) is then
  a few percent of the numerator.  Measured PATH relative error 8e-7..3e-6,
  which is exactly TIMED's residual and, through 20*ALOG10, FSLOS's 3e-5 dB.
* ``1 - SPHET**2`` in CPHET, and this one is the big one.  A 1 ULP difference
  in CDEL moves CPHET by 1.8e-7, XMUF by 4.5e-7 relative -- and PRBMUF's
  normal tail turns that into a 2.1e-4 RELATIVE move in P, which
  ``-10*ALOG10(P)/CPHET`` amplifies again into 2.8e-3 dB of XLS, times HOP.
  That is the 5.6e-3 dB worst cell in TLOSS/SIGPOW/FLDST/SN, and it was
  verified term by term: Stage A is bit-exact against the oracle on that same
  cell, so every bit of it is torch's libm differing from gfortran's.

RGAIN/TGAIN/EFF STAY BIT-EXACT, which is worth stating because it was not
obvious: GAIN's ground-reflection branch does call torch's ATAN/ASIN/ALOG and
its answer (GRLOS) does drift, but the antenna branch is a pure bilinear
interpolation of the /cantenna/ tables with no libm at all, so the two gains
come back exact on both devices.  B/HP/HN/NMODE are copies or counts.

WHAT THIS DUMP CANNOT REACH is asserted rather than assumed, in
``test_the_unreachable_branches_are_still_unreachable``: the -7 floor under
XFAC (no lane gets MUFday below 1e-7), the 1e-6 floor under the over-the-MUF
probability (unreachable by construction -- PRBMUF's quartic bottoms out at
5.83e-6, so only its FSET <= 0 step branch can go lower, and every FSET column
in this dump is positive), the 1e-6 floor under CPHET, and the non-positive
MUFday STOP.  The last one gets its own synthetic test instead.

THE ANTENNA IS PER-LANE HERE, unlike production.  ``BatchedCantenna`` documents
XFQS/XFQE/PWRDBA as (A+1,) because a real area-coverage run is one transmitter
seen from many grid points.  The oracle varied the transmitter's power on 221
of the 262 cases and its frequency window on one, so the fixture gives those
three fields an extra lane axis, (A+1, N).  Every use of them inside
``gain.py`` is elementwise against an (N,) frequency, so it broadcasts
unmodified -- the shared ARRAY/AEFF tables and the per-entry IATS/OFFAZIM keep
their production shapes.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.readant import read_antenna
from voacap_gpu.stage_b.gain import BatchedCantenna
from voacap_gpu.stage_b.regmod import FIELDS, RegmodError, regmod_batch

from tests.stage_a.dumputil import ANTENNAS_DIR, load_dump

DUMP = load_dump("regmod")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SAMPLE = ANTENNAS_DIR / "sample.14"
NANT = 20

# the dump's 103 injected columns, mirroring gen_regmod_cases.py COLUMNS
_SCALARS = ["jmode", "freq", "gcdkm", "ghopkm", "km", "fi1", "gyz", "fs1",
            "fs2", "hs", "ifob13", "acav", "htloss", "xnuz", "hnu", "xve",
            "afe", "bfe", "asm", "dsl", "dsu", "rcnse"]
_QUADS = ["sigl", "sigu", "ymuf", "yfot", "yhpf", "hpmuf", "htmuf", "fvmuf"]
_SIXES = ["delmod", "hpmod", "htmod", "fvmod", "itmod", "afmod"]
_FIVES = ["sigpat", "epspat"]
COLUMNS = (_SCALARS
           + [f"{n}{i}" for n in _QUADS for i in range(1, 5)]
           + [f"{n}{i}" for n in _SIXES for i in range(1, 7)]
           + [f"{n}{i}" for n in _FIVES for i in range(1, 6)]
           + ["pwrdba", "txlo", "txhi"])
_SIGD = ("acav", "htloss", "xnuz", "hnu", "xve", "afe", "bfe", "asm",
         "dsl", "dsu")

# Fields that only copy or count, so no arithmetic can drift: the radiation
# angle and virtual height come straight from /MODES/, the layer index and hop
# count are integers, and the two gains plus the efficiency are GAIN's own
# bilinear table lookup -- no libm on that path.  Same set Stage A pins.
_EXACT = {"b", "eff", "hn", "hp", "nmode", "rgain", "tgain"}

# (floored ULP, absolute dB -- ms for TIMED, none for PROB) per field.  Set
# from the measured worst in the module docstring with 4x-10x headroom; see
# there for why neither bar alone is enough.
_BARS = {"abps": (256.0, 5.0e-4), "adv": (32.0, 5.0e-5),
         "fldst": (4096.0, 0.05), "fslos": (64.0, 5.0e-4),
         "grlos": (64.0, 5.0e-5), "obf": (512.0, 1.0e-3),
         # PROB's absolute bar is deliberately tight: the low-MUFday penalty
         # switches on at PROB < 1e-4, so an error that size could flip a
         # branch rather than just shade a number.
         "prob": (256.0, 2.0e-5), "sigpow": (2048.0, 0.05),
         "sn": (32768.0, 0.05), "timed": (256.0, 2.0e-3),
         "tllow": (1024.0, 5.0e-3), "tlhgh": (512.0, 5.0e-3),
         "tloss": (2048.0, 0.05)}
EXACT_FLOOR = {"cpu": 0.96, "cuda": 0.90}       # measured 0.984 / 0.944

_ANT = None


def _tables():
    global _ANT
    if _ANT is None:
        _ANT = read_antenna(SAMPLE).p2p_tables()
    return _ANT


def _cantenna(txlo, txhi, pwrdba, device):
    """The two entries dump_regmod.f installs, with a PER-LANE TX window.

    Entry 1 is the transmitter over [txlo, txhi] at pwrdba dBW, entry 2 the
    receiver over 2-30 MHz -- the same /cantenna/ Stage A's fixture builds,
    except that txlo/txhi/pwrdba differ per case, so those three fields carry
    an extra lane axis.  See the module docstring.
    """
    array, aeff = _tables()
    c = CantennaState()
    for iat in (1, 2):
        c.array[1:31, 1:92, iat] = array
        c.aeff[1:31, iat] = aeff
        c.offazim[iat] = F32(0.0)
    c.numants = 2
    c.iats[1], c.iats[2] = 1, 2
    cant = BatchedCantenna.from_stage_a(c, device)
    n = len(txlo)
    z = torch.zeros((NANT + 1, n), dtype=torch.float32, device=device)
    cant.xfqs, cant.xfqe, cant.pwrdba = z.clone(), z.clone(), z.clone()
    cant.xfqs[1] = torch.as_tensor(txlo, device=device)
    cant.xfqe[1] = torch.as_tensor(txhi, device=device)
    cant.xfqs[2], cant.xfqe[2] = 2.0, 30.0
    cant.pwrdba[1] = torch.as_tensor(pwrdba, device=device)
    return cant


def _batch(device="cpu", inputs=None):
    """The whole dump as one batch, every column injected.

    `inputs` overrides the case rows, which is how test_inmuf.py reuses this
    builder: INMUF's dump has REGMOD's columns as its own leading block, so
    sharing the fixture keeps the two from drifting apart.  `expected` is
    meaningless when it is overridden and the caller ignores it.
    """
    _, dump_inputs, rows = DUMP
    inputs = dump_inputs if inputs is None else inputs
    n = len(inputs)
    col = {name: np.array([np.float32(r[j]) for r in inputs], dtype=np.float32)
           for j, name in enumerate(COLUMNS)}
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32),  # noqa
                                  device=device)
    # (7, N) with the Fortran 1-based mode index, matching fdist_batch
    stack6 = lambda name: torch.cat(  # noqa: E731
        [torch.zeros(1, n, device=device),
         torch.stack([t(col[f"{name}{i}"]) for i in range(1, 7)])])
    stack = lambda name, m: torch.stack(  # noqa: E731
        [t(col[f"{name}{i}"]) for i in range(1, m + 1)])

    modes = {name: stack6(name) for name in _SIXES}
    modes["itmod"] = modes["itmod"].to(torch.int32)
    modes["ghop"] = t(col["ghopkm"] / np.float32(6370.0))
    args = dict(
        modes=modes,
        muf={name: stack(name, 4) for name in _QUADS},
        sigd={name: t(col[name]) for name in _SIGD},
        cant=_cantenna(col["txlo"], col["txhi"], col["pwrdba"], device),
        sigpat=stack("sigpat", 5), epspat=stack("epspat", 5),
        km=torch.as_tensor(col["km"].astype(np.int64), device=device),
        fi1=t(col["fi1"]), gyz=t(col["gyz"]), fs1=t(col["fs1"]),
        fs2=t(col["fs2"]), hs=t(col["hs"]), ifob13=t(col["ifob13"]),
        rcnse=t(col["rcnse"]), freq=t(col["freq"]),
        gcd=t(col["gcdkm"] / np.float32(6370.0)))
    expected = (np.array([[np.float32(x) for x in r] for r in rows],
                         dtype=np.float32)
                if inputs is dump_inputs else None)
    return args, expected, col


def _to(args, device):
    """Move an already-built batch, so both devices see IDENTICAL bits in."""
    move = lambda v: v.to(device)  # noqa: E731
    cant = args["cant"]
    new = BatchedCantenna(
        array=move(cant.array), aeff=move(cant.aeff), iats=move(cant.iats),
        xfqs=move(cant.xfqs), xfqe=move(cant.xfqe),
        offazim=move(cant.offazim), pwrdba=move(cant.pwrdba),
        numants=cant.numants)
    out = {}
    for k, v in args.items():
        if k == "cant":
            out[k] = new
        elif isinstance(v, dict):
            out[k] = {kk: move(vv) for kk, vv in v.items()}
        else:
            out[k] = move(v)
    return out


def _ulp(got, want):
    """ulp_diff_floored, vectorised: |d| / spacing(max(|want|, 1))."""
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    denom = np.spacing(np.maximum(np.abs(want), np.float32(1.0)))
    return np.where(got == want, 0.0, d / denom)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_regmod_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_regmod_batch_vs_oracle(device):
    args, expected, _ = _batch(device)
    out = regmod_batch(**args)
    cells = exact = 0
    for j, name in enumerate(FIELDS):
        got = out[name].cpu().numpy().astype(np.float32)      # (7, N)
        want = expected[:, j * 7:(j + 1) * 7].T               # (7, N)
        assert got.shape == want.shape
        hit = got == want
        cells += hit.size
        exact += int(hit.sum())
        if name in _EXACT:
            bad = np.argwhere(~hit)
            assert not len(bad), (
                f"[{device}] {name} drifted at slot {bad[0][0] + 1} lane "
                f"{bad[0][1]}: {got[tuple(bad[0])]!r} vs "
                f"{want[tuple(bad[0])]!r} — nothing on this field's path "
                "calls libm, so it is a wiring bug, not a floor")
            continue
        ulp_bar, abs_bar = _BARS[name]
        ulp = _ulp(got, want)
        err = np.abs(got.astype(np.float64) - want.astype(np.float64))
        for metric, bar, label in ((ulp, ulp_bar, "floored ULP"),
                                   (err, abs_bar, "absolute")):
            i = np.unravel_index(np.argmax(metric), metric.shape)
            assert metric[i] <= bar, (
                f"[{device}] {name} at slot {i[0] + 1} lane {i[1]}: "
                f"{got[i]!r} vs {want[i]!r} — {metric[i]:.3g} {label}, "
                f"bar {bar:g}")
    assert exact / cells >= EXACT_FLOOR[device], (
        f"[{device}] only {exact}/{cells} = {100 * exact / cells:.2f}% of "
        "cells are bit-exact")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_the_batch_reaches_every_branch():
    """A per-lane select only proves something on a batch that diverges."""
    args, _, col = _batch()
    modes, sigd = args["modes"], args["sigd"]
    out = regmod_batch(**args)
    act = torch.stack([modes["hpmod"][i] > 0.0 for i in range(1, 7)])
    is_f = torch.stack([args["fi1"] - modes["fvmod"][i] < 0.0
                        for i in range(1, 7)])
    sub = torch.stack([sigd["htloss"] - modes["htmod"][i] <= 0.0
                       for i in range(1, 7)])
    has_es = (args["fs2"] > 0.0).expand(6, -1)
    n = lambda m: int((m & act).sum())  # noqa: E731

    assert n(act) == 1123, "the dump changed shape"
    assert int((~act).sum()) == 449, "the dump changed shape"
    # the two structural dispatches and the D-E collision-frequency sub-branch
    assert n(is_f) == 855, "F-layer modes"
    assert n(~is_f) == 268, "D-E modes"
    assert n(~is_f & sub) == 260, "D-E with the nominal XNSQ"
    assert n(~is_f & ~sub) == 8, "D-E with the EXP collision profile"
    # sporadic-E obscuration of an F mode, present and absent
    assert n(is_f & has_es) == 799 and n(is_f & ~has_es) == 56
    # ... including 411 cells where PROS saturates its 0.90 clamp
    assert n(out["obf"][:6] >= 9.99) == 411
    # the low-MUFday extra loss, and its upper XFAC clamp at -4
    assert n(out["prob"][:6] < 1.0e-4) == 270
    # DEL clamped at 89.99 degrees
    delmod = torch.stack([modes["delmod"][i] for i in range(1, 7)])
    assert n(delmod > 89.99) == 5
    # the TLLOW/TLHGH ceiling at 25 dB
    assert n(out["tllow"][:6] == 25.0) == 141
    assert n(out["tlhgh"][:6] == 25.0) == 208
    # 1..8 hops, 1..5 sample areas (so the KM mask on the ground-loss average
    # is exercised in every shape), and a lane with no mode at all
    hop = torch.trunc(args["gcd"] / modes["ghop"] + 0.01)
    assert set(hop.tolist()) == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0}
    assert set(args["km"].tolist()) == {1, 2, 3, 4, 5}
    assert int((~act.any(0)).sum()) == 1
    # a frequency outside the transmitter's window: GAIN returns 0 dB and
    # PWRDB falls back to 30 dBW on those lanes
    miss = (args["freq"] < args["cant"].xfqs[1]) | (args["freq"]
                                                    > args["cant"].xfqe[1])
    assert int(miss.sum()) == 2
    assert n(out["tgain"][:6] == 0.0) == 12


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_the_unreachable_branches_are_still_unreachable():
    """Honest coverage: four guards this dump CANNOT exercise.

    Asserting they stay unreachable is the useful half -- if a regenerated
    dump ever does reach one, this fails and says so instead of quietly
    leaving the branch untested.
    """
    args, _, _ = _batch()
    out = regmod_batch(**args)
    act = torch.stack([args["modes"]["hpmod"][i] > 0.0 for i in range(1, 7)])
    # 1. XFAC's lower clamp: nothing gets MUFday below 1e-7
    assert int(((out["prob"][:6] < 1.0e-7) & act).sum()) == 0
    # 2. the non-positive MUFday STOP (covered synthetically below)
    assert int(((out["prob"][:6] <= 0.0) & act).sum()) == 0
    # 3. the 1e-6 floor under the over-the-MUF probability.  PRBMUF's quartic
    #    bottoms out at 0.5/(1+5*(...))**4 = 5.83e-6 for |Z/SIG| clamped at 5,
    #    so the ONLY route below 1e-6 is its FSET <= 0 step branch -- and
    #    every FSET column here is positive, which is what this checks.
    for name in ("ymuf", "yfot", "yhpf"):
        assert bool((args["muf"][name] > 0.0).all()), f"{name} has a zero row"
    # 4. CPHET's 1e-6 floor: no mode reflects at grazing incidence
    for im in range(1, 7):
        d2r = F32(0.01745329251)
        cdel = torch.cos(torch.clamp(d2r * args["modes"]["delmod"][im],
                                     max=float(F32(89.99) * d2r)))
        sphet = 6370.0 * cdel / (6370.0 + args["modes"]["htmod"][im])
        assert int(((1.0 - sphet * sphet < 1.0e-6) & act[im - 1]).sum()) == 0


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_slot7_and_unused_modes_keep_the_preset():
    """Slot 7 is ESMOD's; an unused mode keeps HN/HP = -1, as /MODES/ marks."""
    args, _, _ = _batch()
    out = regmod_batch(**args)
    assert bool((out["hn"][6] == -1.0).all()), "slot 7 was written"
    assert bool((out["hp"][6] == -1.0).all()), "slot 7 was written"
    assert bool((out["tloss"][6] == 0.0).all())

    dead = dict(args)
    dead["modes"] = dict(args["modes"])
    dead["modes"]["hpmod"] = torch.full_like(args["modes"]["hpmod"], -1.0)
    out = regmod_batch(**dead)
    assert bool((out["hn"] == -1.0).all()) and bool((out["hp"] == -1.0).all())
    for name in FIELDS:
        if name in ("hn", "hp"):
            continue
        assert bool((out[name] == 0).all()), f"{name} written for a dead slot"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_zero_mufday_is_refused():
    """PRBMUF returns a hard 0 when no median is placed; the Fortran STOPs.

    Unreachable from the dump, so it is provoked: FSET <= 0 collapses PRBMUF
    to a step at FGO, and a FREQ above FGO puts the step at zero.
    """
    args, _, _ = _batch()
    args["muf"] = dict(args["muf"])
    args["muf"]["ymuf"] = torch.zeros_like(args["muf"]["ymuf"])
    args["muf"]["fvmuf"] = torch.ones_like(args["muf"]["fvmuf"])
    with pytest.raises(RegmodError, match="MUFday"):
        regmod_batch(**args)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_nonpositive_frequency_is_refused():
    """The other Fortran STOP.  One bad lane in 262 must stop the batch --
    silently returning NaN for it is the failure this project cannot afford.
    """
    args, _, _ = _batch()
    freq = args["freq"].clone()
    live = args["modes"]["hpmod"][1:7].max(dim=0).values > 0.0
    freq[int(torch.nonzero(live)[0])] = 0.0
    args["freq"] = freq
    with pytest.raises(RegmodError, match="frequency"):
        regmod_batch(**args)
    # ... but a dead lane never reaches the Fortran's test, so a zero there is
    # not an error, exactly as the scalar code has it
    args["freq"] = args["freq"].clone()
    args["freq"][int(torch.nonzero(~live)[0])] = 0.0
    args["freq"][int(torch.nonzero(live)[0])] = 10.0
    regmod_batch(**args)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_a_lane_does_not_depend_on_its_batch_neighbours():
    """REGMOD is elementwise over lanes, so this should be free -- but the
    ground-loss average masks on a PER-LANE KM and the /MUFS/ rows come from a
    PER-LANE ISMOD gather, and either one gets this wrong by leaking a
    neighbour's row.  A uniform batch would not notice.
    """
    args, _, _ = _batch()
    whole = regmod_batch(**args)
    for i in (0, 7, 131, 160, 261):
        one = {}
        for k, v in args.items():
            if k == "cant":
                one[k] = v
            elif isinstance(v, dict):
                one[k] = {kk: (vv[..., i:i + 1] if vv.dim() > 1
                               else vv[i:i + 1]) for kk, vv in v.items()}
            else:
                one[k] = v[..., i:i + 1]
        # the cantenna's per-lane columns have to be sliced too
        c = one["cant"]
        one["cant"] = BatchedCantenna(
            array=c.array, aeff=c.aeff, iats=c.iats, offazim=c.offazim,
            xfqs=c.xfqs[:, i:i + 1], xfqe=c.xfqe[:, i:i + 1],
            pwrdba=c.pwrdba[:, i:i + 1], numants=c.numants)
        got = regmod_batch(**one)
        for name in FIELDS:
            assert np.array_equal(whole[name][:, i:i + 1].cpu().numpy(),
                                  got[name].cpu().numpy()), (
                f"lane {i} {name} depends on its batch neighbours")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """Same bits in on both devices, so only the libm may differ.

    The DISCRETE and table-lookup fields must agree exactly across devices --
    a mode either exists or it does not, and the antenna interpolation has no
    transcendental in it -- while the loss budget is allowed its libm floor.
    """
    args = _batch("cpu")[0]
    cpu = regmod_batch(**args)
    cuda = regmod_batch(**_to(args, "cuda"))
    for name in FIELDS:
        a = cpu[name].numpy()
        b = cuda[name].cpu().numpy()
        if name in _EXACT:
            assert np.array_equal(a, b), f"{name} differs between devices"
            continue
        ulp_bar, abs_bar = _BARS[name]
        ulp = _ulp(a, b)
        err = np.abs(a.astype(np.float64) - b.astype(np.float64))
        assert ulp.max() <= ulp_bar, f"{name} worst {ulp.max():.1f} ULP"
        assert err.max() <= abs_bar, f"{name} worst {err.max():.3g} absolute"
