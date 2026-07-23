"""Stage B batched SERPRB + MPATH vs the SAME tier-B oracle dump Stage A uses.

Both read /allMODE/ and write one /SON/ column, and they run back to back in
LUFFY, so the dump drives them TOGETHER in that order -- which is what pins
that neither disturbs the other's view of the mode table.  NREL is injected
rather than produced, keeping both independent of RELBIL's mode selection: a
failure here is these two routines, not the one before them.

All 375 cases are ONE batch, which is the point: LUFP is per lane, so SERPRB's
two tails run side by side and the ``torch.where`` between them is under test
rather than assumed, and DMP/PMP/METHOD/MSPEC/GCDKM are per lane, so MPATH's
two early RETURNs become masks that have to coexist with the lanes that run.

The two routines disagree about what a dead mode is, which is worth pinning:
for SERPRB a slot is dead at HP <= 70 km, for MPATH at HP <= 0.  A slot between
those thresholds is live for one and dead for the other, and the case grid puts
modes there -- asserted below as ``live_for_one_only``.

MEASURED, whole dump as one batch, 375 cases x 26 columns = 9,750 cells:

    cpu    9,750 exact (100%), worst 0.0 ULP
    cuda   9,750 exact (100%), worst 0.0 ULP

i.e. the same ALL-BIT-EXACT Stage A gets, on both devices.  There is no
transcendental in this pair at all: SERPRB's only libm call is SQRT, which IEEE
754 requires to be correctly rounded, so torch's and gfortran's agree bit for
bit; FNORML is a pure polynomial; MPATH is comparisons and a maximum.  The ULP
bar below is therefore a libm safety valve and nothing more, and the exactness
floor is set at 1.0 because anything less would be a real regression rather
than noise.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_b.mpath import mpath_batch
from voacap_gpu.stage_b.serprb import NoModesError, serprb_batch

from tests.stage_a.dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("serprb")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

NM = 20
SENTINEL = np.float32(-8888.0)
SCALARS = ["ifx", "rsn", "lufp", "du", "dl", "sigm", "sygu", "sygl",
           "ads", "sus", "sls", "nmmod", "nrel", "dmp", "pmp", "gcdkm",
           "method", "mspec"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "sigpow", "timed", "rely"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

# SQRT is correctly rounded and FNORML is a polynomial, so every column here
# is reachable bit-exactly.  The bar exists only so a libm change surfaces as
# a number rather than as a wall of failures.
ULP_BAR = 4.0                                   # measured worst 0.0
EXACT_FLOOR = {"cpu": 1.0, "cuda": 1.0}         # measured 1.0 / 1.0


def _cols():
    _, inputs, _ = DUMP
    return {name: np.array([float(r[j]) for r in inputs], dtype=np.float64)
            for j, name in enumerate(COLUMNS)}


def _args(device):
    cols = _cols()
    n = len(cols["ifx"])

    def f(name):
        return torch.as_tensor(cols[name].astype(np.float32), device=device)

    def i(name):
        return torch.as_tensor(cols[name].astype(np.int64), device=device)

    modes = {}
    for a in ARRAYS:
        t = torch.zeros((NM + 1, n), dtype=torch.float32, device=device)
        for m in range(1, NM + 1):
            t[m] = f(f"{a}{m}")
        modes[a] = t
    # the oracle presets SPRO per case, so a slot SERPRB never writes compares
    # as untouched rather than as coincidentally-zero
    modes["spro"] = torch.full((NM + 1, n), float(SENTINEL),
                               dtype=torch.float32, device=device)
    serprb_args = dict(
        modes=modes, nmmod=i("nmmod"), du=f("du"), dl=f("dl"), sigm=f("sigm"),
        sygu=f("sygu"), sygl=f("sygl"), ads=f("ads"), sus=f("sus"),
        sls=f("sls"), rsn=f("rsn"), lufp=i("lufp"))
    mpath_args = dict(
        modes=modes, nmmod=i("nmmod"), nrel=i("nrel"), dmp=f("dmp"),
        pmp=f("pmp"), gcdkm=f("gcdkm"), method=i("method"),
        mspec=i("mspec"))
    return serprb_args, mpath_args, cols


def _run(device):
    """SERPRB then MPATH, in LUFFY's order and on the same mode table."""
    serprb_args, mpath_args, cols = _args(device)
    out = serprb_batch(**serprb_args)
    probmp = mpath_batch(**mpath_args)
    got = {f"spro{m:02d}": out["spro"][m] for m in range(1, NM + 1)}
    got["sprob"] = out["sprob"]
    got["probmp"] = probmp
    for name in ("d10s", "d50s", "d90s", "spr"):
        got[name] = out[name]
    flat = {k: v.cpu().numpy().astype(np.float32) for k, v in got.items()}
    return flat, cols


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_serprb_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_serprb_mpath_batch_vs_oracle(device):
    header, _, rows = DUMP
    got, cols = _run(device)
    total = exact = 0
    worst, where = 0.0, ""
    for j, name in enumerate(header):
        e = np.array([np.float32(r[j]) for r in rows], dtype=np.float32)
        g = got[name]
        for lane in range(len(e)):
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

    # ---- coverage --------------------------------------------------------
    nmmod = cols["nmmod"].astype(int)
    lufp = cols["lufp"].astype(int)
    hp = np.stack([cols[f"hp{m}"] for m in range(1, NM + 1)])
    live = np.arange(1, NM + 1)[:, None] <= nmmod[None, :]
    area = ((cols["method"] == 20) & (cols["mspec"] == 121)
            & (cols["gcdkm"] > 7000.0))
    seen = {
        "lufp_under_50": (lufp < 50).sum(),
        "lufp_over_50": (lufp >= 50).sum(),
        "dead_mode": ((hp <= 70.0) & live).any(axis=0).sum(),
        "all_dead": (((hp <= 70.0) | ~live) & live.any(axis=0)).all(
            axis=0).sum(),
        # a slot SERPRB calls dead and MPATH calls alive
        "live_for_one_only": ((hp > 0.0) & (hp <= 70.0) & live).any(
            axis=0).sum(),
        "mp_off": ((cols["dmp"] <= 0.0) | (cols["pmp"] <= 0.0)).sum(),
        "mp_area_cut": area.sum(),
        "mp_found": (got["probmp"] > np.float32(0.001)).sum(),
        "mp_none": (got["probmp"] == np.float32(0.001)).sum(),
    }
    for arm, count in seen.items():
        assert count, f"branch never reached: {arm} ({seen})"
    # NMMOD is never 0 in this dump (the META says so: SERPRB reads
    # uninitialised locals there), so the refusal below cannot be covered from
    # the oracle and is exercised synthetically instead.
    assert (nmmod > 0).all(), "the dump now reaches the UB case; re-check"
    # slots above NMMOD must come back untouched
    assert all(got[f"spro{m:02d}"][lane] == SENTINEL
               for lane in range(len(nmmod))
               for m in range(nmmod[lane] + 1, NM + 1))


def _modes(n_modes, lanes=1, device="cpu", **kw):
    """A minimal /allMODE/ for `lanes` lanes, all sharing one mode layout."""
    modes = {}
    default = {"hp": 300.0, "sn": 20.0, "tllow": 9.0, "tlhgh": 8.0,
               "sigpow": -120.0, "timed": 11.0, "rely": 0.5}
    for a in ARRAYS:
        t = torch.zeros((NM + 1, lanes), dtype=torch.float32, device=device)
        for m in range(1, n_modes + 1):
            v = kw.get(a, [default[a]] * n_modes)[m - 1]
            t[m] = float(np.float32(v))
        modes[a] = t
    return modes


def _noise(lanes, device="cpu"):
    z = torch.zeros(lanes, dtype=torch.float32, device=device)
    return dict(du=z + 9.7, dl=z + 6.0, sigm=z + 5.4, sygu=z + 1.5,
                sygl=z + 1.5, ads=z + 5.6, sus=z + 0.6, sls=z + 1.63)


def _serprb(modes, lanes=1, nmmod=2, lufp=90, rsn=24.0, device="cpu"):
    z = torch.zeros(lanes, dtype=torch.float32, device=device)
    return serprb_batch(
        modes=modes,
        nmmod=torch.full((lanes,), nmmod, dtype=torch.int64, device=device),
        rsn=z + float(np.float32(rsn)),
        lufp=torch.full((lanes,), lufp, dtype=torch.int64, device=device),
        **_noise(lanes, device))


def _mpath(modes, lanes=1, nmmod=2, nrel=1, dmp=0.85, pmp=10.0, gcdkm=3000.0,
           method=30, mspec=0, device="cpu"):
    z = torch.zeros(lanes, dtype=torch.float32, device=device)
    ones = torch.ones(lanes, dtype=torch.int64, device=device)
    return mpath_batch(
        modes=modes,
        nmmod=ones * nmmod, nrel=ones * nrel, dmp=z + float(np.float32(dmp)),
        pmp=z + float(np.float32(pmp)), gcdkm=z + float(np.float32(gcdkm)),
        method=ones * method, mspec=ones * mspec)


def test_no_modes_is_refused():
    """With NMMOD <= 0 the Fortran reads uninitialised D10SA/D50SA locals.
    One bad lane poisons the batch, so the refusal is on ``.any()``."""
    modes = _modes(2, lanes=3)
    with pytest.raises(NoModesError):
        serprb_batch(
            modes=modes,
            nmmod=torch.tensor([2, 0, 2]),
            rsn=torch.full((3,), 24.0), lufp=torch.full((3,), 90,
                                                        dtype=torch.int64),
            **_noise(3))


def test_the_two_routines_disagree_about_a_dead_mode():
    """SERPRB kills a mode at HP <= 70 km; MPATH only at HP <= 0.  Both run
    on the SAME table in the same order LUFFY uses them."""
    modes = _modes(2, hp=[300.0, 40.0], timed=[11.0, 30.0], rely=[0.5, 0.77])
    out = _serprb(modes)
    assert out["spro"][2, 0] == F32(0.001), "40 km is not a mode for SERPRB"
    probmp = _mpath(modes)
    assert probmp[0] == F32(0.77), "but it still counts as multipath"


def test_multipath_keeps_modes_OUTSIDE_the_delay_tolerance():
    """The header comment says "within"; the code skips within and keeps
    outside, which is what multipath interference actually means.  Both cases
    as two lanes of one batch, so the mask is what decides, not the call."""
    modes = _modes(2, lanes=2, timed=[11.0, 11.5], rely=[0.5, 0.9])
    modes["timed"][2, 1] = 14.0             # lane 1 is 3 ms away, lane 0 0.5
    probmp = _mpath(modes, lanes=2)
    assert probmp[0] == F32(0.001), "0.5 ms apart is not multipath"
    assert probmp[1] == F32(0.9), "3 ms apart is"


def test_multipath_not_requested_and_the_area_cut():
    modes = _modes(2, lanes=4, timed=[11.0, 30.0], rely=[0.5, 0.9])
    ones = torch.ones(4, dtype=torch.int64)
    probmp = mpath_batch(
        modes=modes, nmmod=ones * 2, nrel=ones,
        dmp=torch.tensor([0.0, 0.85, 0.85, 0.85]),
        pmp=torch.tensor([10.0, 0.0, 10.0, 10.0]),
        gcdkm=torch.tensor([3000.0, 3000.0, 7001.0, 7001.0]),
        method=torch.tensor([30, 30, 20, 30]),
        mspec=torch.tensor([0, 0, 121, 121]))
    assert (probmp[:3] == F32(0.001)).all(), (
        "DMP <= 0, PMP <= 0 and the METHOD 20 / MSPEC 121 / >7000 km area cut "
        "each suppress multipath entirely")
    assert probmp[3] == F32(0.9), "METHOD 30 is not cut at 7000 km"


def test_d90s_is_a_copy_of_d10s_not_an_upper_decile():
    out = _serprb(_modes(2))
    assert out["d90s"][0] == out["d10s"][0]
    assert out["d10s"][0] > out["d50s"][0]
    assert out["spr"][0] == 0.0, "/TON/ SPR is zeroed and never touched again"


def test_both_lufp_tails_in_one_batch():
    """XLH flips sign across LUFP 50 and four of the five spreads change with
    it, so the two arms are computed in full and selected -- the reference
    level moves DOWN from the median above 50% and UP below it."""
    modes = _modes(1, lanes=2)
    out = serprb_batch(
        modes=modes, nmmod=torch.tensor([1, 1]), rsn=torch.full((2,), 24.0),
        lufp=torch.tensor([90, 10]), **_noise(2))
    sn = float(modes["sn"][1, 0])
    assert float(out["d50s"][0]) < sn < float(out["d50s"][1])


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """ONE CPU-built batch fed to both devices.  Nothing in this pair calls a
    device-dependent libm, so this is asserted BITWISE rather than to a
    tolerance -- a difference here would be a real reassociation, not noise.
    """
    a, _ = _run("cpu")
    b, _ = _run("cuda")
    for name in a:
        assert (a[name] == b[name]).all(), (
            f"{name} differs between CPU and CUDA; SQRT is correctly rounded "
            "and FNORML is a polynomial, so there is nothing here that may")
