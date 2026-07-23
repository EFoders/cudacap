"""Stage B batched INMUF vs the SAME tier-B oracle dump Stage A uses.

INMUF inserts the modes FDIST could not find -- above the MUF there are none
in the reflectrix, and at short distances the ray table does not reach -- and
then runs REGMOD on the result.  So this covers both its own bookkeeping and
the integration with everything under it, which is most of the freq loop.

THE FIRST 126 COLUMNS OF THE DUMP ARE INJECTED INPUTS, NOT RESULTS: the
ionogram INMUF's XLIN interpolates, and FI/HI **as LECDEN and GENION leave
them**.  That second part is the one that bites.  The chain ADJUSTS the
critical frequencies and layer heights it is handed, and INMUF's layer tests
read the adjusted values, so feeding back the case file's originals puts a
mode in the wrong layer.  The rule Stage A drew from it holds here too:
injecting PART of a routine's upstream state is worse than injecting none.

Two lockstep-flavoured hazards this file exists to catch, since INMUF is a
per-lane cursor over four stages:

  * the row-0 sink leaking into a real slot -- covered by the
    batch-neighbours test, which is the one that found the same bug in FDIST;
  * stage 3's /MUFS/ overwrite not being undone.  The dump's MUFS block is
    checked for exactly that: YMUF/SIGL/YFOT/SIGU/YHPF must come back as they
    went in on every lane that took stage 3.

MEASURED, whole dump as one batch: see the bars below.  The residual is
REGMOD's, arriving through two extra amplifiers -- INMUF reaches deeper losses
(FLDST to -206 dB against REGMOD's tens) and its inserted modes carry XLIN
plus a second Martyn correction ahead of everything REGMOD then does.

REGMOD's exact-field set does NOT carry over, and assuming it would is the
trap.  There, B/HP/EFF/RGAIN/TGAIN merely copied injected inputs; here INMUF
COMPUTES DELMOD (ATAN(TDEL)*R2D) and HPMOD (HPVER + the Martyn correction), so
B and HP inherit that drift and the gains inherit it through GAIN's elevation
index.  What survives is the genuinely integral: the hop count and the layer
tags.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_b.gain import BatchedCantenna
from voacap_gpu.stage_b.inmuf import inmuf_batch

from tests.stage_a.dumputil import load_dump, ulp_diff_floored
from tests.stage_b.test_regmod import (COLUMNS as REGMOD_COLUMNS, _BARS,
                                       _batch as _regmod_batch, _to)

DUMP = load_dump("inmuf")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

COLUMNS = (list(REGMOD_COLUMNS) + ["ihop", "modmuf"]
           + [f"delmuf{i}" for i in range(1, 5)]
           + [f"afmuf{i}" for i in range(1, 5)]
           + [f"nhopmf{i}" for i in range(1, 5)]
           + ["fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3", "iedp"])

IONO = ["fvert", "hprim", "htrue", "afac"]
MODEF = ["delmod", "hpmod", "htmod", "fvmod", "itmod", "afmod"]
MUFF = ["ymuf", "sigl", "yfot", "sigu", "yhpf"]

_O_IONO, _O_FIHI, _O_NUMMOD, _O_GHOP = 0, 120, 126, 127
_O_MODES = 128
_O_MUFS = _O_MODES + 36
_O_ZON = _O_MUFS + 20
# the dump's own /ZON/ column order, read off expected.csv's header rather
# than assumed -- ADV sits second, not fifteenth
ZONF = ["abps", "adv", "b", "eff", "fldst", "fslos", "grlos", "hn", "hp",
        "nmode", "obf", "prob", "rgain", "sigpow", "sn", "tgain", "timed",
        "tllow", "tlhgh", "tloss"]

# Integral quantities: no arithmetic can move them by a fraction.  ITMOD and
# NUMMOD are INMUF's own; HN and NMODE come back through /ZON/.
_EXACT = {"nummod", "itmod", "hn", "nmode"}
# (floored ULP, absolute) for INMUF's own outputs.  MEASURED: 13,620/13,630
# exact on CPU -- NUMMOD, GHOP, HTMOD, FVMOD, AFMOD, ITMOD and every one of
# the five restored /MUFS/ tables are bit-exact, and the only drift is 7 cells
# of DELMOD (6 ULP, 2.3e-5 deg) and 3 of HPMOD (13 ULP, 3.2e-3 km -- three
# METRES of virtual height).  Both come out of stage 3's ATAN/COS/SIN.
_ULP_BAR = 64.0
_ABS_BAR = 0.05
# INMUF's OWN outputs and the /ZON/ block REGMOD produced are scored
# SEPARATELY.  Pooling them buries INMUF's own near-perfect bookkeeping under
# 32,900 cells of inherited REGMOD residual, and it is INMUF that is on trial
# here -- REGMOD has its own dump and its own test.
EXACT_FLOOR = {"cpu": 0.995, "cuda": 0.99}      # measured 0.99927 CPU
ZON_FLOOR = {"cpu": 0.95, "cuda": 0.90}         # measured 0.958 CUDA

# REGMOD's exact set does NOT carry over -- see the module docstring -- so
# /ZON/ is checked against REGMOD's own per-field bars, widened once because
# INMUF reaches deeper losses and prefixes XLIN plus a second Martyn
# correction.  The precise work on /ZON/ is test_regmod.py's; what this adds
# is that REGMOD ran on the modes INMUF actually produced.
_ZON_WIDEN = 8.0
# The five /ZON/ fields REGMOD treats as EXACT and INMUF cannot.  There they
# only copied injected inputs; here INMUF computes DELMOD and HPMOD, so B and
# HP carry that drift directly and EFF/RGAIN/TGAIN inherit it through GAIN's
# frequency and elevation indices.  Measured worst is 204 ULP / 2.4e-5 dB on
# RGAIN, which is four orders below anything a prediction quotes.
_ZON_INHERITED = {"b": (2048.0, 1.0e-3), "eff": (2048.0, 1.0e-3),
                  "hp": (2048.0, 1.0e-2), "rgain": (2048.0, 1.0e-3),
                  "tgain": (2048.0, 1.0e-3)}


def _build(device="cpu"):
    """The whole dump as one batch.  Inputs come from BOTH files: the case
    row for the scalars, the expected row for the ionogram and FI/HI."""
    _, inputs, rows = DUMP
    n = len(inputs)
    exp = np.array([[np.float32(x) for x in r] for r in rows],
                   dtype=np.float32)
    col = {name: np.array([np.float32(r[j]) for r in inputs],
                          dtype=np.float32)
           for j, name in enumerate(COLUMNS)}

    # REGMOD's builder reads only its own leading columns, so it can be
    # reused verbatim -- which also keeps the two fixtures from drifting.
    args, _, _ = _regmod_batch(device, inputs=[r[:len(REGMOD_COLUMNS)]
                                               for r in inputs])
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32),  # noqa
                                  device=device)

    for i in range(1, 5):
        args["muf"].setdefault("delmuf", torch.zeros(4, n, device=device))
        args["muf"].setdefault("afmuf", torch.zeros(4, n, device=device))
        args["muf"].setdefault("nhopmf", torch.zeros(4, n, device=device))
    args["muf"]["delmuf"] = torch.stack([t(col[f"delmuf{i}"])
                                         for i in range(1, 5)])
    args["muf"]["afmuf"] = torch.stack([t(col[f"afmuf{i}"])
                                        for i in range(1, 5)])
    args["muf"]["nhopmf"] = torch.stack([t(col[f"nhopmf{i}"])
                                         for i in range(1, 5)])
    args["muf"]["modmuf"] = t(col["modmuf"])

    state = {}
    for j, name in enumerate(IONO):
        state[name] = torch.stack(
            [t(exp[:, _O_IONO + j * 30 + ih]) for ih in range(30)])
    # FI/HI as the oracle's LECDEN+GENION left them -- see the docstring
    state["fi"] = torch.stack(
        [t(exp[:, _O_FIHI + i]) for i in range(3)]
        + [torch.zeros(n, device=device)])
    state["hi"] = torch.stack(
        [t(exp[:, _O_FIHI + 3 + i]) for i in range(3)]
        + [torch.zeros(n, device=device)])

    args["state"] = state
    args["ihop"] = torch.as_tensor(col["ihop"].astype(np.int64),
                                   device=device)
    args["jmode"] = torch.as_tensor(col["jmode"].astype(np.int64),
                                    device=device)
    return args, exp, col


def _negative_takeoff(exp):
    """Lanes where the oracle's own /MODES/ holds a used slot at a negative
    take-off angle.

    Stage 3's Martyn recompute sets DELMOD = ATAN(TDEL)*R2D, and TDEL goes
    negative when COS(PSI) < RZ/(RZ+HPNOW).  REGMOD hands that straight to
    GAIN, whose elevation index J = INT(DELTD)+1 goes <= 0 -- and GAIN clamps
    only the top of the axis, so the Fortran reads ARRAY out of bounds and
    picks up a neighbouring antenna's table.  Deterministic and meaningless.

    The port refuses that index, so /ZON/ cannot be compared on such a lane.
    /MODES/ still can: INMUF writes it in full before calling REGMOD.
    """
    delmod = exp[:, _O_MODES:_O_MODES + 6]
    hpmod = exp[:, _O_MODES + 6:_O_MODES + 12]
    return ((hpmod > 0.0) & (delmod < 0.0)).any(axis=1)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_inmuf_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_inmuf_batch_vs_oracle(device):
    args, exp, col = _build(device)
    ub = _negative_takeoff(exp)
    keep = torch.as_tensor(~ub, device=device)
    if not bool(keep.all()):
        # Drop the out-of-bounds lanes rather than the whole batch: they are
        # the Fortran reading past its antenna table, which this port refuses
        # by policy.  Asserted below that there are only a handful.
        args = _select_lanes(args, keep)
        exp = exp[~ub]

    nummod, zon = inmuf_batch(**args)
    got_modes = {name: args["modes"][name].cpu().numpy() for name in MODEF}
    got_muf = {name: args["muf"][name].cpu().numpy() for name in MUFF}
    got_zon = {name: v.cpu().numpy() for name, v in zon.items()}
    nummod = nummod.cpu().numpy()

    own, zt = [0, 0], [0, 0]
    worst, worst_abs, where = 0.0, 0.0, ""

    def _check(name, g, e, bars=None, tally=None):
        nonlocal worst, worst_abs, where
        tally = own if tally is None else tally
        g, e = np.float32(g), np.float32(e)
        tally[0] += 1
        if g == e:
            tally[1] += 1
            return
        assert name not in _EXACT, (
            f"[{device}] {name} drifted: {g!r} vs {e!r}; it is integral and "
            "nothing here can move it by a fraction")
        d, a = ulp_diff_floored(g, e), abs(float(g) - float(e))
        ub, ab = bars if bars else (_ULP_BAR, _ABS_BAR)
        if d > worst:
            where = f"{name}: {g!r} vs {e!r}"
        worst, worst_abs = max(worst, d), max(worst_abs, a)
        assert d <= ub and a <= ab, (
            f"[{device}] {name}: {g!r} vs {e!r} ({d:.0f} ULP, {a:.2e} abs)")

    for i in range(len(exp)):
        _check("nummod", nummod[i], exp[i, _O_NUMMOD])
        # /MODES/ GHOP is in RADIANS both in the dump and in the port; the
        # km form only appears as an INPUT column of the case file.
        _check("ghop", float(args["modes"]["ghop"][i]), exp[i, _O_GHOP])
        for j, name in enumerate(MODEF):
            for m in range(6):
                _check(name, got_modes[name][m + 1, i],
                       exp[i, _O_MODES + j * 6 + m])
        for j, name in enumerate(MUFF):
            for lay in range(4):
                _check(name, got_muf[name][lay, i],
                       exp[i, _O_MUFS + j * 4 + lay])
        for j, name in enumerate(ZONF):
            bars = _BARS.get(name)
            bars = ((bars[0] * _ZON_WIDEN, bars[1] * _ZON_WIDEN) if bars
                    else _ZON_INHERITED.get(name, (_ULP_BAR, _ABS_BAR)))
            for s in range(7):
                _check(name, got_zon[name][s, i], exp[i, _O_ZON + j * 7 + s],
                       bars, zt)

    assert int(ub.sum()) <= 8, (
        f"{int(ub.sum())} of {len(ub)} lanes drive GAIN out of bounds; that "
        "was 1 in Stage A's run and a jump means the port changed, not the "
        "dump")
    assert own[1] / own[0] >= EXACT_FLOOR[device], (
        f"[{device}] INMUF's own outputs: exact {own[1]}/{own[0]}, worst "
        f"{worst:.0f} ULP / {worst_abs:.2e} abs at {where}")
    assert zt[1] / zt[0] >= ZON_FLOOR[device], (
        f"[{device}] /ZON/ after INMUF: exact {zt[1]}/{zt[0]} -- REGMOD's "
        "own residual, but a collapse here means INMUF fed it wrong modes")


def _select_lanes(args, keep):
    """Narrow an already-built batch to a lane mask.

    The antenna needs narrowing too, but only its three per-lane fields --
    this fixture varies transmit power and the band edges across the dump,
    which BatchedCantenna allows, while ARRAY/AEFF stay shared.
    """
    out = {}
    for k, v in args.items():
        if k == "cant":
            out[k] = BatchedCantenna(
                array=v.array, aeff=v.aeff, iats=v.iats, offazim=v.offazim,
                numants=v.numants,
                **{f: (getattr(v, f)[:, keep] if getattr(v, f).dim() == 2
                       else getattr(v, f))
                   for f in ("xfqs", "xfqe", "pwrdba")})
        elif isinstance(v, dict):
            out[k] = {kk: vv[..., keep] if vv.dim() else vv
                      for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v[..., keep]
        else:
            out[k] = v
    return out


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_stage_three_puts_the_muf_distributions_back():
    """Stage 3 OVERWRITES YMUF/SIGL/YFOT/SIGU/YHPF so REGMOD sees the per-hop
    MUF, then has to undo it.  A batched port can get this wrong in a way the
    scalar one cannot: restore every lane and the lanes that did not take
    stage 3 are fine but so is a bug; restore none and the caller's /MUFS/ is
    quietly corrupted for the next frequency."""
    args, exp, _ = _build("cpu")
    before = {n: args["muf"][n].clone() for n in MUFF}
    ub = _negative_takeoff(exp)
    args = _select_lanes(args, torch.as_tensor(~ub))
    before = {n: v[:, ~ub] for n, v in before.items()}
    inmuf_batch(**args)
    for n in MUFF:
        assert torch.equal(args["muf"][n], before[n]), (
            f"{n} came back changed; stage 3's overwrite was not undone")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_a_lane_does_not_depend_on_its_batch_neighbours():
    """The cursor-and-sink failure mode: a mode written at row 0 by one lane
    must never surface in another's slot.  This is the test that caught the
    equivalent bug in FDIST."""
    args, exp, _ = _build("cpu")
    ub = _negative_takeoff(exp)
    args = _select_lanes(args, torch.as_tensor(~ub))
    n = args["freq"].shape[0]
    whole_n, whole_z = inmuf_batch(**args)
    whole_modes = {k: v.clone() for k, v in args["modes"].items()}

    for i in (0, 5, n // 2, n - 1):
        one_args, _, _ = _build("cpu")
        one_args = _select_lanes(one_args, torch.as_tensor(~ub))
        sel = torch.zeros(n, dtype=torch.bool)
        sel[i] = True
        one_args = _select_lanes(one_args, sel)
        one_n, one_z = inmuf_batch(**one_args)
        assert int(one_n) == int(whole_n[i]), f"lane {i} NUMMOD differs"
        for name in MODEF + ["ghop"]:
            a = whole_modes[name][..., i:i + 1]
            assert torch.equal(a, one_args["modes"][name]), (
                f"lane {i} {name} depends on its batch neighbours")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """Identical bits in, so the only difference allowed out is libm.  The
    INTEGRAL results -- how many modes, which layer each is -- are pure
    comparison and cursor arithmetic and must agree exactly."""
    args, exp, _ = _build("cpu")
    ub = torch.as_tensor(~_negative_takeoff(exp))
    args = _select_lanes(args, ub)
    n_cpu, _ = inmuf_batch(**args)
    modes_cpu = {k: v.clone() for k, v in args["modes"].items()}

    args2, _, _ = _build("cpu")
    args2 = _to(_select_lanes(args2, ub), "cuda")
    n_cuda, _ = inmuf_batch(**args2)
    assert torch.equal(n_cpu, n_cuda.cpu()), "NUMMOD differs between devices"
    # read the MOVED dict: inmuf_batch mutates in place, so the pre-move
    # CPU copy would still hold the untouched input and compare equal by
    # accident on a port that never wrote itmod at all
    assert torch.equal(modes_cpu["itmod"],
                       args2["modes"]["itmod"].cpu()), "ITMOD differs"
