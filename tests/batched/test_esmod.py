"""Stage B batched ESMOD vs the SAME tier-B oracle dump Stage A uses.

All 466 dump cases are loaded as 466 lanes of ONE batch -- they are genuinely
independent here, unlike ALLMODES -- and the lanes diverge in every way the
routine can diverge: 1 to 5 sample areas, one-hop / two-hop / no-mode / cut by
the minimum take-off angle, and both early returns.

MEASURED, 21,436 cells (23 fields x 2 slots x 466 lanes):

    device   exact            worst floored ULP   worst absolute
    cpu      21,269 (99.22%)  2,648               6.3e-4
    cuda     21,055 (98.22%)  478                 1.87

The two columns disagree about which is "worse" because they drift in
different places.  CPU's 2,648 ULP is RGAIN at 0.334 dB: the take-off angle
comes out 9e-5 degrees from gfortran's and the antenna table interpolation
turns that into 4e-4 dB, so the ULP count is large only because the value is
small (the floored metric divides by the spacing at 1.0).

CUDA's 1.87 is more interesting and is the one number here worth explaining.
It lands on TLOSS = 52,935 dB -- a mode that does not exist in any physical
sense -- and its chain is: ``COS(PSI)`` differs by 1 ULP, so ``SECS`` and then
``ESD = FS(2,K)*SECS`` differ by 1 ULP; ``PRBMUF`` immediately subtracts
``FREQ - ESD``, 5.18 - 5.06, which magnifies that by ~40; the Hastings quartic
magnifies the relative error four times more; and ``PROS**(-0.7)`` then runs
it into a REF of 26,000 dB, doubled by the two-hop factor.  5e-5 relative in
PROS becomes 1.87 dB absolute.  It is a conditioning property of the far tail,
not of the port -- CPU reproduces the oracle bit-for-bit on that same lane.

BIT-EXACT FIELDS.  ADV, CREL, HN, HP, NMODE, PROB, RELY and SPRO are asserted
bitwise on both devices, because none of them touches the trig chain: they are
constants, presets, a copy of HS, or (PROB) a pure polynomial of the inputs.
EFF joins them because GAIN's efficiency interpolates over FREQUENCY ALONE and
never sees the elevation.  Stage A also lists RGAIN/TGAIN as exact; here they
cannot be, because the elevation they are looked up at is a torch ATAN result.

The ADX quirks -- counted twice in the loss and once in ABPS -- get their own
test rather than being taken on trust from the dump: perturbing AFE by a known
amount moves ABPS by that amount and TLOSS by twice it per hop.

NOT COVERED BY THE DUMP: the degenerate-geometry refusal (SDMAX <= 0, i.e.
HS <= 0).  All 466 cases have IHSRT >= 1, which the oracle test asserts is
still true; the refusal itself is exercised by a constructed case.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.readant import read_antenna
from voacap_gpu.batched.esmod import esmod_batch
from voacap_gpu.batched.gain import BatchedCantenna

from tests.scalar.dumputil import ANTENNAS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("esmod")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

RZ = np.float32(6370.0)

ULP_BAR = 8192.0                # measured worst 2,648 (CPU), 478 (CUDA)
EXACT_FLOOR = {"cpu": 0.985,    # measured 0.99221
               "cuda": 0.975}   # measured 0.98223
# no dependence on the trig chain: constants, presets, a copy of an input, a
# pure polynomial, or an interpolation over frequency alone
EXACT = ("adv", "crel", "eff", "hn", "hp", "nmode", "prob", "rely", "spro")

FIELDS = ("abps", "adv", "b", "crel", "eff", "fldst", "fslos", "grlos", "hn",
          "hp", "nmode", "obf", "prob", "rely", "rgain", "sigpow", "sn",
          "spro", "tgain", "timed", "tllow", "tlhgh", "tloss")
SLOTS = (4, 5)

# column order in gen_esmod_cases.py
COLUMNS = (
    ["jmode", "freq", "gcdkm", "amind", "km", "ifob13",
     "acav", "afe", "bfe", "asm", "dsl", "dsu", "rcnse",
     "sigl4", "sigu4", "ymuf4", "yfot4", "yhpf4"]
    + [f"fs1_{i}" for i in range(1, 6)]
    + [f"fs2_{i}" for i in range(1, 6)]
    + [f"fs3_{i}" for i in range(1, 6)]
    + [f"hs{i}" for i in range(1, 6)]
    + [f"fi1_{i}" for i in range(1, 6)]
    + [f"gyz{i}" for i in range(1, 6)]
    + [f"sigpat{i}" for i in range(1, 6)]
    + [f"epspat{i}" for i in range(1, 6)]
    + ["pwrdba", "txlo", "txhi", "rxlo", "rxhi"]
)

_ANT = None


def _cantenna(device, txlo, txhi, rxlo, rxhi, pwrdba):
    """sample.14 as both terminals, over PER-LANE frequency ranges and power.

    ``BatchedCantenna`` documents ARRAY/AEFF as shared across lanes and they
    are -- one circuit has one antenna pattern.  XFQS/XFQE/PWRDBA are a
    different matter: GAIN only ever indexes them with a PYTHON int (the
    entry number, in the DO-10 scan and in PWRDB), never with the per-lane
    entry tensor, so giving them a lane axis works and is what this dump
    needs -- the case generator randomised the receiver's band and the
    transmitter power per case.
    """
    global _ANT
    if _ANT is None:
        _ANT = read_antenna(ANTENNAS_DIR / "sample.14").p2p_tables()
    array, aeff = _ANT
    cs = CantennaState()
    cs.numants = 2
    cs.iats[1], cs.iats[2] = 1, 2
    for iat in (1, 2):
        cs.array[1:31, 1:92, iat] = array
        cs.aeff[1:31, iat] = aeff
    cant = BatchedCantenna.from_stage_a(cs, device)
    n = txlo.shape[0]
    z = lambda: torch.zeros((3, n), dtype=torch.float32,  # noqa: E731
                            device=device)
    cant.xfqs, cant.xfqe, cant.pwrdba = z(), z(), z()
    cant.xfqs[1], cant.xfqe[1] = txlo, txhi
    cant.xfqs[2], cant.xfqe[2] = rxlo, rxhi
    cant.pwrdba[1] = pwrdba
    return cant


def _load(device):
    """The whole dump as one batch of 466 lanes."""
    _, inputs, rows = DUMP
    n = len(rows)
    v = {name: torch.tensor([np.float32(r[j]) for r in inputs],
                            dtype=torch.float32, device=device)
         for j, name in enumerate(COLUMNS)}
    st = lambda pre: torch.stack(  # noqa: E731
        [v[f"{pre}{i}"] for i in range(1, 6)])
    fs = torch.stack([torch.stack([v[f"fs{b}_{i}"] for i in range(1, 6)])
                      for b in (1, 2, 3)])
    gcdkm = v["gcdkm"]
    args = dict(
        km=v["km"].to(torch.long), fs=fs, hs=st("hs"), fi1=st("fi1_"),
        gyz=st("gyz"), sigpat=st("sigpat"), epspat=st("epspat"),
        acav=v["acav"], afe=v["afe"], bfe=v["bfe"], asm=v["asm"],
        dsl=v["dsl"], dsu=v["dsu"], rcnse=v["rcnse"], sigl4=v["sigl4"],
        sigu4=v["sigu4"], ymuf4=v["ymuf4"], yfot4=v["yfot4"],
        yhpf4=v["yhpf4"], ifob13=v["ifob13"],
        cant=_cantenna(device, v["txlo"], v["txhi"], v["rxlo"], v["rxhi"],
                       v["pwrdba"]),
        freq=v["freq"], gcd=gcdkm / torch.tensor(RZ, device=device),
        gcdkm=gcdkm, amind=v["amind"])
    return args, rows, v


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_esmod_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_esmod_batch_vs_oracle(device):
    header, _, rows = DUMP
    args, rows, v = _load(device)
    out = {k: t.cpu().numpy() for k, t in esmod_batch(**args).items()}

    total = exact = 0
    worst_ulp = worst_abs = 0.0
    where = ""
    drifted = set()
    for name in FIELDS:
        for s in SLOTS:
            col = header.index(f"{name}{s}")
            e = np.array([np.float32(r[col]) for r in rows], dtype=np.float32)
            g = out[name][s - 4].astype(np.float32)
            if name in EXACT:
                bad = np.argwhere(g != e)
                assert not len(bad), (
                    f"[{device}] {name}{s} lane {bad[0][0]}: {g[bad[0][0]]!r} "
                    f"vs {e[bad[0][0]]!r} -- must be bit-exact, it never "
                    "touches the trig chain")
            u = np.array([ulp_diff_floored(a, b) for a, b in zip(g, e)])
            a = np.abs(g.astype(np.float64) - e.astype(np.float64))
            i = int(u.argmax())
            if u[i] > worst_ulp:
                worst_ulp = float(u[i])
                where = f"{name}{s} lane {i}: {g[i]!r} vs {e[i]!r}"
            worst_abs = max(worst_abs, float(a.max()))
            assert u.max() <= ULP_BAR, (
                f"[{device}] {name}{s} lane {int(u.argmax())}: "
                f"{g[u.argmax()]!r} vs {e[u.argmax()]!r} "
                f"({u.max():.1f} floored ULP)")
            if u.max():
                drifted.add(name)
            total += len(e)
            exact += int((u == 0).sum())
    assert not (drifted & set(EXACT))
    assert exact / total >= EXACT_FLOOR[device], (
        f"[{device}] exact {exact}/{total} = {exact / total:.5f}, worst "
        f"{worst_ulp:.1f} ULP at {where}, worst {worst_abs:.3e} absolute")

    # ---- coverage: every arm of the routine has to be reached -------------
    hn = out["hn"]
    seen = {
        "one hop in slot 4": int((hn[0] == 1).sum()),
        "two hops in slot 4": int((hn[0] == 2).sum()),
        "no mode at all": int(((hn[0] == -1) & (hn[1] == -1)).sum()),
        "slot 5 live": int((hn[1] > 0).sum()),
        # a one-hop mode cut by AMIND still CONSUMES slot 4
        "AMIND cut slot 4": int(((hn[0] == -1) & (hn[1] > 0)).sum()),
        "TLLOW clamped at 25": int((out["tllow"] == 25.0).sum()),
        "TLHGH clamped at 25": int((out["tlhgh"] == 25.0).sum()),
        "unmatched antenna": int(((hn > 0) & (out["rgain"] == 0.0)).sum()),
    }
    # the early returns, recomputed from the inputs
    km, fs = args["km"], args["fs"]
    k = torch.zeros(km.shape[0], dtype=torch.long, device=device)
    fk = fs[1, 0]
    for i in range(1, 5):
        take = (i < km) & (fk - fs[1, i] > 0.0)
        k = torch.where(take, torch.full_like(k, i), k)
        fk = torch.where(take, fs[1, i], fk)
    fsdead = torch.minimum(v["ifob13"] / 1000.0,
                           torch.tensor(3.0, device=device))
    alive = (v["freq"] > fsdead) & (fk > 0)
    hsk = torch.gather(args["hs"], 0, k.unsqueeze(0)).squeeze(0)
    rz = torch.tensor(RZ, device=device)
    sdmax = 2.0 * rz * (np.float32(1.570796326)
                        - torch.asin(1.0 / (1.0 + hsk / rz)))
    ihsrt = torch.trunc(v["gcdkm"] / sdmax + 1.0).to(torch.long)
    seen["FREQ <= FSDEAD return"] = int((v["freq"] <= fsdead).sum())
    seen["FS(2,K) <= 0 return"] = int((fk <= 0).sum())
    seen["IHSRT > 2 return"] = int((alive & (ihsrt > 2)).sum())
    seen["IHSRT == 1"] = int((alive & (ihsrt == 1)).sum())
    seen["IHSRT == 2"] = int((alive & (ihsrt == 2)).sum())
    seen["K moved off area 1"] = int((k > 0).sum())
    for arm, count in seen.items():
        assert count, f"[{device}] branch never reached: {arm} ({seen})"
    assert len(set(km.tolist())) == 5, "not every sample-area count is present"
    # NOT COVERED: the SDMAX <= 0 refusal.  The dump cannot reach it, and
    # that is asserted rather than assumed; a constructed case covers it.
    assert bool((ihsrt >= 1).all()), (
        "the dump now reaches IHSRT < 1 -- the refusal in esmod_batch is no "
        "longer dead and needs covering from the dump")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_a_lane_must_not_depend_on_its_batch_neighbours():
    """The two hops scatter through a sink row; a dirty sink or a leaked
    cursor would make a lane's answer depend on who else is in the batch."""
    args, rows, _ = _load("cpu")
    whole = {k: t.numpy() for k, t in esmod_batch(**args).items()}
    for i in (0, 79, 120, 152, 399, len(rows) - 1):
        one = dict(args)
        for name, val in args.items():
            if torch.is_tensor(val):
                one[name] = val[..., i:i + 1].contiguous()
        cant = args["cant"]
        one["cant"] = type(cant)(
            array=cant.array, aeff=cant.aeff, iats=cant.iats,
            xfqs=cant.xfqs[:, i:i + 1], xfqe=cant.xfqe[:, i:i + 1],
            offazim=cant.offazim, pwrdba=cant.pwrdba[:, i:i + 1],
            numants=cant.numants)
        single = {k: t.numpy() for k, t in esmod_batch(**one).items()}
        for name, got in single.items():
            assert np.array_equal(whole[name][..., i:i + 1], got), (
                f"lane {i} {name} depends on its batch neighbours")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """The mode SELECTION -- which slot, how many hops, whether a mode exists
    at all -- is comparison only and must agree exactly.  Only the computed
    losses may move, and then by the documented trig/pow floor."""
    outs = []
    for device in ("cpu", "cuda"):
        args, _, _ = _load(device)
        outs.append({k: t.cpu().numpy()
                     for k, t in esmod_batch(**args).items()})
    for name in ("hn", "hp", "nmode", "adv", "crel", "rely", "spro", "prob",
                 "eff"):
        assert np.array_equal(outs[0][name], outs[1][name]), f"{name} differs"
    for name in FIELDS:
        a, b = outs[0][name], outs[1][name]
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        sp = np.spacing(np.maximum(np.abs(b), np.float32(1.0)))
        assert (d / sp).max() <= ULP_BAR, (
            f"{name} worst {(d / sp).max():.1f} floored ULP between devices")


def _constructed(device, gcdkm, **over):
    """A hand-built batch: one Es layer, sample.14 at both ends, 2-30 MHz."""
    gcdkm = torch.tensor(gcdkm, dtype=torch.float32, device=device)
    n = gcdkm.shape[0]
    f = lambda x: torch.full((n,), float(x), dtype=torch.float32,  # noqa: E731
                             device=device)
    a = lambda x: torch.stack(  # noqa: E731
        [f(x[i]) for i in range(5)])
    args = dict(
        km=torch.ones(n, dtype=torch.long, device=device),
        fs=torch.stack([a([4.0, 0, 0, 0, 0]), a([5.0, 0, 0, 0, 0]),
                        a([7.0, 0, 0, 0, 0])]),
        hs=a([110.0, 0, 0, 0, 0]), fi1=a([3.0, 0, 0, 0, 0]),
        gyz=a([1.2, 0, 0, 0, 0]), sigpat=a([5.0, 0, 0, 0, 0]),
        epspat=a([80.0, 0, 0, 0, 0]),
        acav=f(0.12), afe=f(0.5), bfe=f(1.5), asm=f(3.0), dsl=f(8.0),
        dsu=f(9.0), rcnse=f(-145.0), sigl4=f(0.15), sigu4=f(0.20),
        ymuf4=f(14.0), yfot4=f(11.0), yhpf4=f(17.0), ifob13=f(2500.0),
        cant=_cantenna(device, f(2.0), f(30.0), f(2.0), f(30.0), f(40.0)),
        freq=f(10.0), gcd=gcdkm / torch.tensor(RZ, device=device),
        gcdkm=gcdkm, amind=f(0.0))
    args.update({k: f(x) for k, x in over.items()})
    return args


@pytest.mark.parametrize("device", DEVICES)
def test_a_two_hop_path_writes_slot_4_not_slot_5(device):
    """IH counts DO-150 TRIPS, so ``IHOP = 2,2`` is one trip writing slot 4.

    Slot 4 is "the first mode found", not "the one-hop mode".  This is the
    detail most likely to be mis-ported, and running the three geometries as
    three LANES also checks that they do not interfere.
    """
    out = esmod_batch(**_constructed(device, [1500.0, 3500.0, 9000.0]))
    hn = out["hn"].cpu().numpy()
    assert hn[0, 0] == 1.0 and hn[1, 0] == 2.0    # IHSRT = 1 -> hops 1 and 2
    assert hn[0, 1] == 2.0, "the two-hop mode belongs in slot 4"
    assert hn[1, 1] == -1.0, "slot 5 must keep its presets"
    assert int(out["nmode"][1, 1]) == 5
    assert float(out["tloss"][1, 1]) == 1000.0
    assert hn[0, 2] == -1.0 and hn[1, 2] == -1.0  # IHSRT > 2 -> no Es mode


@pytest.mark.parametrize("device", DEVICES)
def test_the_presets_survive_every_early_return(device):
    """All four returns must still leave slots 4 and 5 readable."""
    args = _constructed(device, [1500.0, 1500.0, 9000.0, 1500.0])
    args["fs"][1, 0, 1] = -1.0          # FS(2,K) <= 0
    args["freq"][0] = 1.0               # FREQ <= FSDEAD
    args["amind"][3] = 89.0             # every hop cut by the angle
    out = {k: t.cpu().numpy() for k, t in esmod_batch(**args).items()}
    for lane in (0, 1, 2, 3):
        for row in (0, 1):
            assert out["hn"][row, lane] == -1.0
            assert out["hp"][row, lane] == -1.0
            assert out["nmode"][row, lane] == 5
            assert out["tloss"][row, lane] == 1000.0
            # ESREG's slots get +1000; the two routines disagree on purpose
            assert out["crel"][row, lane] == -1000.0
            assert out["eff"][row, lane] == 0.0
            assert out["prob"][row, lane] == np.float32(0.001)


@pytest.mark.parametrize("device", DEVICES)
def test_adx_counts_twice_in_the_loss_and_once_in_abps(device):
    """The Es-below-E correction is added to SABPS *and again* inside XTLOS.

    Perturbing AFE by a known amount is a clean probe of it: ADX moves by
    exactly that amount on the lanes where the branch is live, so ABPS moves
    by the same and TLOSS by twice it per hop.  That also proves both arms of
    ``IF(FI(1,K).GT.SFVMOD)`` are reachable -- the lanes whose ABPS does not
    move are the ones taking the other one.
    """
    d = 1000.0
    base = _constructed(device, [1500.0, 1500.0])
    base["fi1"][0, 0] = 5.0             # lane 0: FI(1,K) over SFVMOD, 2 hops
    base["fi1"][0, 1] = 0.5             # lane 1: FI(1,K) far below SFVMOD
    out0 = {k: t.cpu().numpy() for k, t in esmod_batch(**base).items()}
    bumped = dict(base)
    bumped["afe"] = base["afe"] + d
    out1 = {k: t.cpu().numpy() for k, t in esmod_batch(**bumped).items()}

    for row, hop in ((0, 1.0), (1, 2.0)):
        assert out0["hn"][row, 0] == hop
        assert (out1["abps"][row, 0] - out0["abps"][row, 0]
                == pytest.approx(d, rel=1e-6)), (
            "ADX must reach ABPS exactly once")
        assert (out1["tloss"][row, 0] - out0["tloss"][row, 0]
                == pytest.approx(2.0 * hop * d, rel=1e-6)), (
            "ADX must reach TLOSS twice per hop")
        # the other arm: FI(1,K) below SFVMOD leaves ADX at zero
        assert out1["abps"][row, 1] == out0["abps"][row, 1]
        assert out1["tloss"][row, 1] == out0["tloss"][row, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_the_lowest_es_critical_frequency_picks_the_sample_area(device):
    """DO 95 takes the MINIMUM FS(2,.), not the maximum: the weakest link
    along the path bounds the mode.  And only the first KM areas are looked
    at, so a lower value beyond KM must be ignored."""
    args = _constructed(device, [1500.0, 1500.0])
    args["km"] = torch.tensor([3, 1], device=device)
    for lane in (0, 1):
        args["hs"][:, lane] = torch.tensor([110.0, 90.0, 250.0, 70.0, 60.0],
                                           device=device)
        args["fs"][1, :, lane] = torch.tensor([5.0, 9.0, 3.0, 1.0, 1.0],
                                              device=device)
        args["fs"][0, :, lane] = 4.0
        args["fs"][2, :, lane] = 7.0
        args["fi1"][:, lane] = 3.0
        args["gyz"][:, lane] = 1.2
        args["sigpat"][:, lane] = 5.0
        args["epspat"][:, lane] = 80.0
    out = esmod_batch(**args)
    # KM = 3 -> area 3 (HS 250) wins on FS(2,.) = 3; the smaller 1.0 in areas
    # 4 and 5 is out of range.  KM = 1 -> area 1 (HS 110).
    assert float(out["hp"][0, 0]) == 250.0
    assert float(out["hp"][0, 1]) == 110.0


@pytest.mark.parametrize("device", DEVICES)
def test_a_degenerate_es_height_is_refused(device):
    """HS <= 0 makes SDMAX <= 0, and the Fortran's DO 150 would then run far
    more than the two hops /ZON/ has room for.  Unreachable from the dump."""
    args = _constructed(device, [1500.0])
    args["hs"][0] = 0.0
    with pytest.raises(ValueError, match="SDMAX"):
        esmod_batch(**args)
