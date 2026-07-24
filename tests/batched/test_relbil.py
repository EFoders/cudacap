"""Stage B batched RELBIL vs the SAME tier-B oracle dump Stage A uses.

All 390 cases are ONE batch, so every branch runs beside every other and the
selection between them is under test rather than assumed: lanes with no modes
sit next to lanes with twenty, dead modes next to live ones, LUFP below 50 next
to LUFP above it.

WHAT THE BATCHING ACTUALLY RISKS, and therefore what this asserts hardest:

* the **two-pass mode selection**.  Pass one is an argmax with FIRST-WINS ties;
  pass two is a sequential walk whose XREL/XHN/XSN carry moves as it goes, and
  which can therefore land somewhere other than the maximum.  A port that
  turned either into a plain reduction, or that got the tie-break backwards,
  changes NREL -- and NREL is the index every copy-only field is gathered at,
  so it shows up as ANGLE, GAINT, VHIGH and friends all going wrong together.
  Those fields are asserted BITWISE for exactly that reason, and the dump is
  asserted to contain cases where the walk moves away from the maximum.
* the **stale read**.  D10R/D50R/D90R live in /DON/, persist across calls, and
  RELBIL assigns none of them when every mode is dead -- SNRLW/SNRUP/SNPR then
  report the PREVIOUS frequency's numbers.  The batched form threads the triple
  in and out, and the oracle seeds it per case so the stale value is a known
  number.  ``test_stale_deciles_survive_a_dead_only_mode`` pins it directly.
* the **sentinels**.  The oracle presets every dumped output to -8888 (NREL -1,
  DSLF/DSUF -7777) before each call and this seeds the same, so a field RELBIL
  never writes is compared as untouched rather than as coincidentally-zero.

MEASURED, whole dump as one batch, 390 cases x 69 columns = 26,910 cells:

    cpu    26,897 exact (99.95%), worst 8.0 floored ULP
    cuda   26,883 exact (99.90%), worst 8.0 floored ULP

Stage A is 26,910/26,910 on this dump.  THE ENTIRE RESIDUAL IS THE COMBINED
DISTRIBUTION and nothing else: it lands only on DBLOSL/DBLOSU and the six
quantities derived from them, only on lanes with more than one mode, and it
comes from ``10.**ZEXP`` and ``ALOG10`` -- torch's are different functions from
this gfortran build's.  SQRT is IEEE-correctly-rounded on both, which is why
RELY, CREL and every arm-A decile are bit-exact everywhere.  DBLOSL is
``|SIGMED - 10*log10(XDSLW) - DXDSLW|``, a cancellation of ~-83 dB quantities
down to ~15, so it multiplies a 1-ULP log10 difference by about five.  Eight
ULP on a decile spread in dB is nothing.

The exact FRACTION is asserted anyway, because it would COLLAPSE rather than
drift if the mode selection or the sum ordering were wrong.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.sonstate import LAYTYP
from voacap_gpu.batched.gain import BatchedCantenna, pwrdb_batch
from voacap_gpu.batched.relbil import SON_NAMES, relbil_batch

from tests.scalar.dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("relbil")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

NM = 20
SENTINEL = np.float32(-8888.0)
SCRATCH_SENTINEL = np.float32(-7777.0)

SCALARS = ["ifx", "freq", "rsn", "lufp", "du", "dl", "rcnse", "nmmod",
           "pwrdba", "txlo", "txhi", "d10r", "d50r", "d90r"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "hn", "sigpow", "fldst", "tgain",
          "rgain", "b", "timed", "tloss", "prob", "eff", "nmode"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

# copies of an input, an integer, or set to a constant: no arithmetic can drift
# them, and SQRT is correctly rounded so RELY and CREL cannot either.
EXACT = {"nrel", "modecode", "nhp", "angle", "cprob", "vhigh", "delay",
         "rneff", "xnynois", "du_nois", "dl_nois", "gaint", "gainr",
         "dslf", "dsuf", "crel", "rely"}
# the combined-distribution fields: 10.**ZEXP and ALOG10, plus the DBLOSL
# cancellation that amplifies them.  See the module docstring.
ULP_BAR = 32.0                          # measured worst 8.0 either device
EXACT_FLOOR = {"cpu": 0.997, "cuda": 0.995}     # measured 0.99952 / 0.99900


def _pwrdb_per_lane(freq, txlo, txhi, pwrdba):
    """PWRDB with a PER-LANE one-entry /cantenna/.

    ``gain.pwrdb_batch`` takes a SHARED antenna table, because a Stage B batch
    is one transmitter seen from many grid points.  The oracle varies the
    transmitter BAND and POWER per case -- 302 distinct powers over 390 cases,
    and cases whose operating frequency falls outside their own band -- so no
    shared table can express it.  Stage A's harness fabricates a fresh
    one-entry CantennaState per case for the same reason; this is that,
    batched.  ``test_pwrdb_helper_matches_gain`` pins it against the real
    routine on a table that IS shared, including the no-match fallback.
    """
    f = torch.clamp(freq, min=2.0, max=30.0)    # FOR THE RANGE TEST ONLY
    return torch.where((txlo <= f) & (f <= txhi), pwrdba,
                       torch.full_like(f, 30.0))   # 1 kW when nothing matches


def _cols():
    _, inputs, _ = DUMP
    return {name: np.array([float(r[j]) for r in inputs], dtype=np.float64)
            for j, name in enumerate(COLUMNS)}


def _args(device):
    """Everything RELBIL needs, the whole dump as one batch on `device`."""
    cols = _cols()
    n = len(cols["ifx"])

    def f(name):
        return torch.as_tensor(cols[name].astype(np.float32), device=device)

    def i(name):
        return torch.as_tensor(cols[name].astype(np.int64), device=device)

    modes = {}
    for a in ARRAYS:
        integer = a == "nmode"
        t = torch.zeros((NM + 1, n),
                        dtype=torch.int64 if integer else torch.float32,
                        device=device)
        for m in range(1, NM + 1):
            t[m] = (i if integer else f)(f"{a}{m}")
        modes[a] = t
    # the oracle presets CREL/RELY per case, so a slot RELBIL never writes
    # compares as untouched rather than as coincidentally-zero
    modes["crel"] = torch.full((NM + 1, n), float(SENTINEL),
                               dtype=torch.float32, device=device)
    modes["rely"] = modes["crel"].clone()

    sent = torch.full((n,), float(SENTINEL), dtype=torch.float32,
                      device=device)
    son_prev = {}
    for name in SON_NAMES:
        if name == "nhp":
            son_prev[name] = torch.full((n,), int(SENTINEL),
                                        dtype=torch.int64, device=device)
        elif name == "modecode":
            # MODE(IFX) is a CHARACTER; untouched it is blank, which the dump
            # writes as LAYTYP index 0
            son_prev[name] = torch.zeros(n, dtype=torch.int64, device=device)
        else:
            son_prev[name] = sent
    scratch = torch.full((n,), float(SCRATCH_SENTINEL), dtype=torch.float32,
                         device=device)
    pwrdb = _pwrdb_per_lane(f("freq"), f("txlo"), f("txhi"), f("pwrdba"))
    return dict(
        modes=modes, nmmod=i("nmmod"), du=f("du"), dl=f("dl"),
        rcnse=f("rcnse"), pwrdb=pwrdb, freq=f("freq"), rsn=f("rsn"),
        lufp=i("lufp"), d10r=f("d10r"), d50r=f("d50r"), d90r=f("d90r"),
        dslf=scratch, dsuf=scratch,
        nrel_prev=torch.full((n,), -1, dtype=torch.int64, device=device),
        son_prev=son_prev), cols


def _flat(allm, son):
    """The batched outputs keyed by the dump's column names."""
    got = {f"crel{m:02d}": allm["crel"][m] for m in range(1, NM + 1)}
    got.update({f"rely{m:02d}": allm["rely"][m] for m in range(1, NM + 1)})
    got["nrel"] = allm["nrel"]
    got.update(son)
    return {k: v.cpu().numpy().astype(np.float32) for k, v in got.items()}


def _run(device):
    args, cols = _args(device)
    allm, son = relbil_batch(**args)
    return _flat(allm, son), cols


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_relbil_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_relbil_batch_vs_oracle(device):
    header, _, rows = DUMP
    got, cols = _run(device)
    total = exact = 0
    worst, where = 0.0, ""
    for j, name in enumerate(header):
        base = name.rstrip("0123456789")
        e = np.array([np.float32(r[j]) for r in rows], dtype=np.float32)
        g = got[name]
        for lane in range(len(e)):
            if base in EXACT:
                assert g[lane] == e[lane], (
                    f"[{device}] lane {lane} {name}: got {g[lane]!r} expected "
                    f"{e[lane]!r} (must be bit-exact: it is a copy, an "
                    "integer, a constant or a correctly-rounded SQRT)")
                total += 1
                exact += 1
                continue
            d = ulp_diff_floored(g[lane], e[lane])
            if d > worst:
                worst, where = d, f"lane {lane} {name}"
            assert d <= ULP_BAR, (
                f"[{device}] lane {lane} {name}: got {g[lane]!r} expected "
                f"{e[lane]!r} ({d:.2f} floored ULP)")
            total += 1
            exact += d == 0.0

    assert exact / total >= EXACT_FLOOR[device], (
        f"[{device}] exact {exact}/{total}, worst {worst:.2f} at {where}")

    # ---- coverage: every branch has to be reached by the batch ------------
    nmmod = cols["nmmod"].astype(int)
    lufp = cols["lufp"].astype(int)
    hp = np.stack([cols[f"hp{m}"] for m in range(1, NM + 1)])
    live = np.arange(1, NM + 1)[:, None] <= nmmod[None, :]
    rely = np.stack([got[f"rely{m:02d}"] for m in range(1, NM + 1)])
    argmax = np.where(live, rely, -np.inf).argmax(axis=0) + 1
    seen = {
        "no_modes": (nmmod <= 0).sum(),
        "one_mode": (nmmod == 1).sum(),
        "many_modes": (nmmod > 1).sum(),
        "dead_mode": ((hp <= 70.0) & live).any(axis=0).sum(),
        "stale_deciles": ((nmmod == 1) & (cols["hp1"] <= 70.0)).sum(),
        # the second pass walking away from the outright maximum -- the thing
        # the source added its first pass to make safe
        "not_the_max": ((nmmod > 1) & (got["nrel"] != argmax)).sum(),
        "clamped_low": ((nmmod > 1)
                        & (got["dblosl"] == np.float32(0.2))).sum(),
        "clamped_high": ((nmmod > 1)
                         & (got["dblosu"] == np.float32(30.0))).sum(),
        "lufp_under_50": ((nmmod > 0) & (lufp < 50)).sum(),
        "lufp_over_50": ((nmmod > 0) & (lufp >= 50)).sum(),
        # both arms of the transmitter-power lookup
        "pwrdb_matched": ((cols["txlo"] <= np.clip(cols["freq"], 2.0, 30.0))
                          & (np.clip(cols["freq"], 2.0, 30.0)
                             <= cols["txhi"])).sum(),
        "pwrdb_fallback": ((cols["txlo"] > np.clip(cols["freq"], 2.0, 30.0))
                           | (np.clip(cols["freq"], 2.0, 30.0)
                              > cols["txhi"])).sum(),
    }
    # the equal-hops arm of the tie-break: two live modes within 0.05 of each
    # other in BOTH reliability and hop count, so the median S/N decides
    hn = np.stack([cols[f"hn{m}"] for m in range(1, NM + 1)])
    close = np.zeros(len(nmmod), bool)
    for p in range(NM):
        for q in range(p + 1, NM):
            close |= (live[p] & live[q]
                      & (np.abs(rely[p] - rely[q]) < np.float32(0.05))
                      & (np.abs(hn[p] - hn[q]) <= np.float32(0.05)))
    seen["equal_hops_tiebreak"] = close.sum()
    for arm, count in seen.items():
        assert count, f"branch never reached: {arm} ({seen})"

    # NOT COVERED, and provably NOT COVERABLE: the four `.GT.0.` guards on
    # XDSLW/XSIGS/XDSUP/XFLD.  Each sum is referred to its own maximum term,
    # so that term's ZEXP is exactly 0, clears the -10 floor and puts 1.0 into
    # the sum -- every sum is >= 1 whenever the lane has a mode at all.  The
    # else arms (DBLOSL/DBLOSU = 0, SIGMED/DBU = -500) are dead code in the
    # Fortran, not merely unexercised by this dump.  Asserted as the property
    # that makes them dead rather than claimed as branch coverage.
    # NOT COVERED, same reason: the `if(xrel.lt..00000001) xrel=.00000001`
    # floor in the second pass.  FNORML clamps |Z| at 5, so the smallest
    # reliability it can return is ~5.8e-6, and a dead mode gets 0.001 --
    # XREL can never fall below 1e-8.  Deleting the floor leaves all 26,910
    # cells intact, which is the measurement behind this assertion.
    assert np.where(live, rely, 1.0).min() > 1e-8, (
        "a reliability below the XREL floor appeared; the floor is reachable "
        "after all and deleting it would no longer be a no-op")

    sigpow = np.stack([cols[f"sigpow{m}"] for m in range(1, NM + 1)])
    tllow = np.stack([cols[f"tllow{m}"] for m in range(1, NM + 1)])
    tlhgh = np.stack([cols[f"tlhgh{m}"] for m in range(1, NM + 1)])
    fldst = np.stack([cols[f"fldst{m}"] for m in range(1, NM + 1)])
    some = nmmod > 0
    for what, table in (("XSIGS", sigpow), ("XFLD", fldst),
                        ("XDSLW", sigpow - tllow),
                        ("XDSUP", sigpow + tlhgh)):
        top = np.where(live, table, -np.inf).max(axis=0)[some]
        assert (top > -1000.0).all(), (
            f"{what}'s maximum term falls below the AMAX1 seed of -1000, so "
            "its .GT.0. guard is reachable after all")

    labels = {int(c) for c, m in zip(got["modecode"], nmmod) if m > 0}
    assert labels == set(range(1, len(LAYTYP) + 1)), (
        f"unreached layer label: {sorted(labels)}")
    # a lane with no modes must come back completely untouched
    none = nmmod <= 0
    assert (got["nrel"][none] == -1).all(), "NREL written with no modes"
    assert (got["dslf"][none] == SCRATCH_SENTINEL).all()
    assert (got["reliab"][none] == SENTINEL).all()


def _one_lane(hp, sn, hn, nmode=3, nmmod=None, lufp=90, rsn=24.0,
              device="cpu"):
    """A minimal single-lane /allMODE/, the batched twin of Stage A's
    ``_one_mode_state``.  Every list is per mode, 1-based once padded."""
    n = 1
    nm = len(hp) if nmmod is None else nmmod
    modes = {}
    flat = {"hp": hp, "sn": sn, "hn": hn,
            "tllow": [9.0] * len(hp), "tlhgh": [8.0] * len(hp),
            "sigpow": [-120.0] * len(hp), "fldst": [30.0] * len(hp),
            "prob": [0.9] * len(hp)}
    for a in ARRAYS:
        integer = a == "nmode"
        t = torch.zeros((NM + 1, n),
                        dtype=torch.int64 if integer else torch.float32,
                        device=device)
        for m in range(1, len(hp) + 1):
            if a == "nmode":
                t[m] = nmode
            elif a in flat:
                t[m] = float(np.float32(flat[a][m - 1]))
        modes[a] = t
    z = torch.zeros(n, dtype=torch.float32, device=device)
    return dict(
        modes=modes,
        nmmod=torch.full((n,), nm, dtype=torch.int64, device=device),
        du=z + 9.7, dl=z + 6.0, rcnse=z - 145.0, pwrdb=z + 40.0,
        freq=z + 14.0, rsn=z + float(np.float32(rsn)),
        lufp=torch.full((n,), lufp, dtype=torch.int64, device=device),
        d10r=z.clone(), d50r=z.clone(), d90r=z.clone(),
        dslf=z.clone(), dsuf=z.clone())


def test_second_pass_can_walk_away_from_the_maximum():
    """The 140 loop compares against a best it has already moved.

    Modes 2 and 5 are both near the top; 5 is the outright maximum but 2 has
    fewer hops and is within 0.05, so the second pass replaces the maximum.
    A batched port that collapsed the walk into an argmax, or that started it
    from mode 1 instead of from IRmax, lands somewhere else.  Run as THREE
    lanes, so what is under test is the DIVERGENCE between them rather than
    three separate calls: same modes, three hop layouts, all three arms of the
    comparison in one batch.
    """
    sn = [10.0, 24.3, 11.0, 12.0, 24.4]
    lanes = [
        _one_lane(hp=[300.0] * 5, sn=sn, hn=[3.0, 1.0, 3.0, 3.0, 2.0]),
        _one_lane(hp=[300.0] * 5, sn=sn, hn=[3.0, 4.0, 3.0, 3.0, 2.0]),
        _one_lane(hp=[300.0] * 5, sn=sn, hn=[2.0] * 5),
    ]
    args = {k: ({m: torch.cat([lane[k][m] for lane in lanes], dim=-1)
                 for m in lanes[0][k]} if isinstance(lanes[0][k], dict)
                else torch.cat([lane[k] for lane in lanes], dim=-1))
            for k in lanes[0]}
    allm, _ = relbil_batch(**args)
    rely = np.array([allm["rely"][m].numpy() for m in range(1, 6)])
    assert (rely.argmax(axis=0) + 1 == 5).all(), "mode 5 has the max rely"
    assert int(allm["nrel"][0]) == 2, "fewer hops wins inside the 0.05 window"
    assert int(allm["nrel"][1]) == 5, "more hops does not displace the max"
    # lane 2 takes the EQUAL-HOPS arm instead, where the median S/N decides --
    # and mode 5 already holds it, so the walk evaluates that test and stays
    assert int(allm["nrel"][2]) == 5, "equal hops, and mode 5 has the best S/N"


def test_first_wins_when_reliabilities_tie():
    """Pass one is a strict ``.GT.``, so the LOWEST-indexed maximum wins.

    THE ORACLE DUMP CANNOT DISTINGUISH first-wins from last-wins here --
    flipping the comparison to ``>=`` leaves all 26,910 cells intact, because
    reliabilities computed from continuous inputs do not tie.  Dead modes DO:
    they all get exactly 0.001.  So a lane whose modes are all dead, all with
    the same hop count and the same median S/N, is a genuine tie that the
    second pass has no grounds to break, and it pins the direction.
    """
    args = _one_lane(hp=[50.0] * 3, sn=[5.0] * 3, hn=[2.0] * 3)
    for m, angle in enumerate((11.0, 22.0, 33.0), start=1):
        args["modes"]["b"][m] = angle
    allm, son = relbil_batch(**args)
    assert (allm["rely"][1:4, 0] == F32(0.001)).all(), "all three tie"
    assert int(allm["nrel"][0]) == 1, "the first maximum, not the last"
    assert son["angle"][0] == F32(11.0), "and every copy field follows NREL"


def test_low_virtual_height_is_not_a_mode():
    """HP <= 70 km gets reliability 0.001 and never enters the selection."""
    args = _one_lane(hp=[300.0, 70.0, 300.0], sn=[5.0, 40.0, 6.0],
                     hn=[1.0, 2.0, 3.0])              # boundary is inclusive
    allm, _ = relbil_batch(**args)
    assert allm["rely"][2, 0] == F32(0.001)
    assert allm["crel"][2, 0] == F32(0.001)
    assert int(allm["nrel"][0]) != 2, "a dead mode must not win on S/N"
    assert allm["crel"][1, 0] == F32(1000.0), "live modes get CREL 1000"


def test_stale_deciles_survive_a_dead_only_mode():
    """With no live mode RELBIL assigns none of D10R/D50R/D90R, so the
    /DON/ triple -- and SNRLW/SNRUP through it -- still reports the PREVIOUS
    frequency's spread.  The batched form has to thread it, not recompute it.
    """
    args = _one_lane(hp=[50.0], sn=[22.0], hn=[1.0])
    args["d10r"] = args["d10r"] + F32(3.5)
    args["d50r"] = args["d50r"] + F32(12.0)
    args["d90r"] = args["d90r"] + F32(7.25)
    _, son = relbil_batch(**args)
    assert (float(son["d10r"][0]), float(son["d50r"][0]),
            float(son["d90r"][0])) == (3.5, 12.0, 7.25)
    assert son["snrlw"][0] == F32(3.5) and son["snrup"][0] == F32(7.25)


def test_no_modes_leaves_the_lane_alone():
    """NMMOD <= 0 is a Fortran RETURN before anything is written.  NREL keeps
    whatever it held -- and before the first successful call that is BLOCK
    DATA's 1, not 0, because MPATH and SERPRB index /allMODE/ with it."""
    args = _one_lane(hp=[300.0], sn=[22.0], hn=[1.0], nmmod=0)
    args["modes"]["crel"] = torch.full((NM + 1, 1), float(SENTINEL))
    allm, _ = relbil_batch(**args)
    assert allm["crel"][1, 0] == F32(SENTINEL)
    assert int(allm["nrel"][0]) == 1, "the BLOCK DATA default, not 0"


def test_layer_label_out_of_range_is_refused():
    """NMODE outside 1..5 indexes LAYTYP out of bounds in the Fortran, and in
    torch a bad index silently wraps to the far end of the axis."""
    args = _one_lane(hp=[300.0], sn=[22.0], hn=[1.0], nmode=7)
    with pytest.raises(IndexError):
        relbil_batch(**args)


def test_pwrdb_helper_matches_gain():
    """The per-lane PWRDB helper is ``gain.pwrdb_batch`` for a table that
    happens to be shared -- including the 30 dBW no-match fallback, which is
    the arm the oracle dump reaches through its narrow-band cases."""
    cant = CantennaState()
    cant.numants = 1
    cant.iats[1] = 1
    cant.xfqs[1], cant.xfqe[1] = F32(20.0), F32(25.0)
    cant.pwrdba[1] = F32(40.0)
    freq = torch.tensor([14.0, 22.0, 1.0, 29.0], dtype=torch.float32)
    ref = pwrdb_batch(BatchedCantenna.from_stage_a(cant), freq)
    got = _pwrdb_per_lane(freq, torch.full_like(freq, 20.0),
                          torch.full_like(freq, 25.0),
                          torch.full_like(freq, 40.0))
    assert torch.equal(ref, got), f"{ref} vs {got}"
    assert (ref == 30.0).any() and (ref == 40.0).any(), "need both arms"


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """ONE CPU-built batch fed to both devices, so this isolates RELBIL's own
    ``10.**ZEXP`` and ALOG10 from anything upstream.  It is what says the CUDA
    gap in the oracle comparison belongs to this routine.
    """
    args, _ = _args("cpu")
    a = _flat(*relbil_batch(**args))

    def to_cuda(v):
        if torch.is_tensor(v):
            return v.cuda()
        if isinstance(v, dict):
            return {k: x.cuda() for k, x in v.items()}
        return v

    b = _flat(*relbil_batch(**{k: to_cuda(v) for k, v in args.items()}))
    for name in a:
        base = name.rstrip("0123456789")
        x, y = a[name], b[name]
        if base in EXACT:
            assert (x == y).all(), (
                f"{name} differs across devices; it has no libm call in it")
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        sp = np.spacing(np.maximum(np.abs(y), 1.0))
        assert (d / sp).max() <= ULP_BAR, (
            f"{name} worst {(d / sp).max():.2f} ULP across devices")
