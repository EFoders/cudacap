"""Stage B batched SIGDIS_orig vs the SAME tier-B oracle dump Stage A uses.

All 130 cases are one batch, so the three KFX values (57 / 34 / 39 lanes) run
side by side and the per-lane length of the sample-area average is genuinely
under test rather than assumed.  That average is the point of the port: each
lane sums a different number of areas, in area order, in float32.

**THE IONOGRAM COMES FROM STAGE A, DELIBERATELY.**  SIGDIS reads HTRUE/FVERT
only through the XLIN lookup that produces XVE, and half the dump's cases run
IEDP = 0, which ``genion_batch`` refuses (it implements the IEDP < 0 model
only).  Building the ionogram with Stage A's LECDEN+GENION -- themselves
bit-exact against their own oracle dumps -- keeps all 130 cases in play and,
more importantly, makes the residual reported below SIGDIS's own rather than
an inherited one.  The same arrays are handed to both devices.

MEASURED, 130 lanes x (18 path outputs + ABIY/ARTIC for the active areas)
= 2824 cells:

    CPU   2814/2824 exact (99.65%), worst 1.0 floored ULP
    CUDA  2754/2824 exact (97.52%), worst 4.0 floored ULP

and the residual has a clean shape, which is why the bars are per field:

  * BITWISE on both devices, no tolerance allowed: AGLAT FEAV HTLOSS XVE
    XNUZ HNU AFE BFE ADJ SU SL ADS SUS SLS ARTIC1..3.  These are constants,
    table copies, sums of table cells over KFX, and XLIN -- no libm and no
    cancellation, so a residual would be a wiring bug.
  * ACAV and ABIY1..3 carry EXP.  CPU 126..127/130 and 71/73, 38/39;
    CUDA 103..112/130 and 59/73, 33/39.  Worst 1 ULP either way.
  * ASM, DSU and DSL carry ALOG10, amplified by the deliberate cancellation
    of ESLSM and F2LSM against themselves.  CPU 130/130 (this gfortran's
    log10f and torch's host log10 agree on every one of these arguments);
    CUDA 127..129/130, worst 4 ULP on DSU.

Stage A is 2824/2824 on this dump because its ``f32math`` helpers are matched
to this gfortran build's expf/log10f; torch's are different functions, on the
host as well as the device, so the CPU column is not expected to be 1.000
either and its bar is set below what was measured rather than at it.

ONE ASSOCIATION BUG THIS CAUGHT.  Writing AFE as ``1.359 * ((FEAV-0.5)/1.5)``
-- the obvious factoring, since AFE and BFE share the ramp -- put AFE 1 ULP
out on 5 lanes and BFE on 12, ON CPU TOO.  The Fortran multiplies first and
divides second.  That is the whole reason AFE/BFE are asserted bitwise here:
the tidy version passes any tolerance bar.

WHAT THIS BATCH DOES NOT REACH, stated rather than glossed:

  * **The AFE = BFE = 0 arm (FEAV <= 0.5).**  FEAV is the path-average E
    critical frequency and the dump generator draws FI(1) from [1, 4] MHz, so
    no lane can get there.  ``test_branch_coverage`` asserts the dump still
    cannot, so this note stays true, and ``test_afe_zero_arm_vs_stage_a``
    checks that arm against **Stage A** on a doctored input.  That is the
    weaker claim -- two ports agreeing, not a port agreeing with the Fortran.
  * **SYSSY's no-data (ITSA-1) fallback.**  REDMAP always succeeds here, so
    F2D(1,1,1) > 0 on every lane.  ``tests/batched/test_syssy.py`` covers
    that arm against the oracle directly, on 21 lanes.
  * SIGDIS_alex (the 1/3/5 area stride) and the 'I'/'a' variants without the
    0.1 absorption floor are not ported at all -- the oracle's version string
    ends 'W'.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.constants import R2D
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.sigdis import sigdis
from voacap_gpu.scalar.signalstate import SigdState, TonState
from voacap_gpu.batched.sigdis import sigdis_batch

from tests.scalar.dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("sigdis")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NA = 3                          # the dump injects three sample areas

# Argument order of sigdis_batch, so the raw dict can be splatted.
ORDER = ("fi1", "glat", "clck", "kfx", "htrue", "fvert", "jmode", "gcdkm",
         "sys", "perr", "has_data", "yfot", "yhpf", "ymuf", "sigl", "sigu")

# Constants, table copies, sums of table cells, and XLIN.  No libm, no
# cancellation -> no tolerance.
BITWISE = ("aglat", "feav", "htloss", "xve", "xnuz", "hnu", "afe", "bfe",
           "adj", "su", "sl", "ads", "sus", "sls",
           "artic1", "artic2", "artic3")
EXP_FIELDS = ("acav", "abiy1", "abiy2", "abiy3")    # -0.04 + EXP(...)
LOG_FIELDS = ("asm", "dsu", "dsl")                  # -10*ALOG10(...)

ULP_BAR = 16.0                                      # measured worst 4.0
# Measured: EXP fields 0.969..0.977 on CPU, 0.792..0.862 on CUDA; LOG fields
# 1.000 on CPU, 0.977..1.000 on CUDA.  Bars sit below, not on, those numbers.
EXP_FLOOR = {"cpu": 0.90, "cuda": 0.70}
LOG_FLOOR = {"cpu": 0.95, "cuda": 0.90}


def _layer_state(inp):
    """A Stage A LayerState with the case's areas injected and ionograms built.

    Exactly the setup the Stage A test uses: LECDEN+GENION for k <= KFX, the
    slots above KFX left at zero (which is what the oracle driver's fresh
    COMMON holds too).
    """
    s = LayerState(kfx=int(inp[2]))
    s.iedp = int(inp[5])
    for a in range(1, NA + 1):
        off = 6 + (a - 1) * 11
        v = [float(x) for x in inp[off:off + 11]]
        s.fi[1, a], s.fi[2, a], s.fi[3, a] = (F32(x) for x in v[0:3])
        s.yi[1, a], s.yi[2, a], s.yi[3, a] = (F32(x) for x in v[3:6])
        s.hi[1, a], s.hi[2, a], s.hi[3, a] = (F32(x) for x in v[6:9])
        s.glat[a], s.clck[a] = F32(v[9]), F32(v[10])
    for k in range(1, s.kfx + 1):
        lecden(s, k)
        genion(s, k)
    return s


def _muf_state(inp):
    m = MufState()
    (yf3, yf4, yh3, yh4, ym3, ym4, sl3, sl4, su3, su4) = (
        F32(float(x)) for x in inp[39:49])
    m.yfot[3], m.yfot[4] = yf3, yf4
    m.yhpf[3], m.yhpf[4] = yh3, yh4
    m.ymuf[3], m.ymuf[4] = ym3, ym4
    m.sigl[3], m.sigl[4] = sl3, sl4
    m.sigu[3], m.sigu[4] = su3, su4
    return m


def _build():
    """Every SIGDIS input for the whole dump, as numpy, on the host."""
    _, inputs, _ = DUMP
    n = len(inputs)
    z = lambda *s: np.zeros(s, np.float32)          # noqa: E731
    raw = dict(fi1=z(NA, n), glat=z(NA, n), clck=z(NA, n),
               htrue=z(30, NA, n), fvert=z(30, NA, n), gcdkm=z(n),
               sys=z(9, 16, 6, n), perr=z(9, 4, 6, n),
               yfot=z(4, n), yhpf=z(4, n), ymuf=z(4, n),
               sigl=z(4, n), sigu=z(4, n),
               kfx=np.zeros(n, np.int64), jmode=np.zeros(n, np.int64),
               has_data=np.zeros(n, bool))
    for i, inp in enumerate(inputs):
        s = _layer_state(inp)
        m = _muf_state(inp)
        st = redmap(COEFFS_DIR, "CCIR", int(inp[0]), F32(float(inp[1])))
        raw["kfx"][i] = s.kfx
        raw["jmode"][i] = int(inp[3])
        raw["gcdkm"][i] = np.float32(float(inp[4]))
        raw["sys"][..., i] = st.sys
        raw["perr"][..., i] = st.perr
        raw["has_data"][i] = st.f2d[0, 0, 0] > 0.0
        for a in range(1, NA + 1):
            raw["fi1"][a - 1, i] = s.fi[1, a]
            raw["glat"][a - 1, i] = s.glat[a]
            raw["clck"][a - 1, i] = s.clck[a]
            raw["htrue"][:, a - 1, i] = s.htrue[1:31, a]
            raw["fvert"][:, a - 1, i] = s.fvert[1:31, a]
        for name, arr in (("yfot", m.yfot), ("yhpf", m.yhpf),
                          ("ymuf", m.ymuf), ("sigl", m.sigl),
                          ("sigu", m.sigu)):
            raw[name][2, i], raw[name][3, i] = arr[3], arr[4]
    return raw


RAW = _build() if DUMP is not None else None


def _flat(sigd, ton, area):
    """The 24 dump columns as one name -> (N,) numpy dict."""
    out = {k: v.detach().cpu().numpy() for k, v in sigd.items()}
    out.update({k: v.detach().cpu().numpy() for k, v in ton.items()})
    for a in range(NA):
        out[f"abiy{a + 1}"] = area["abiy"][a].detach().cpu().numpy()
        out[f"artic{a + 1}"] = area["artic"][a].detach().cpu().numpy()
    return out


def _run(device):
    args = [torch.as_tensor(RAW[k], device=device) for k in ORDER]
    return _flat(*sigdis_batch(*args))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_sigdis_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_sigdis_batch_vs_oracle(device):
    header, inputs, rows = DUMP
    assert len(header) == 24
    got = _run(device)
    kfx = RAW["kfx"]

    exact = {name: [0, 0] for name in header}       # [exact, counted]
    worst = {name: (0.0, "") for name in header}
    for f, name in enumerate(header):
        # ABIY/ARTIC above KFX are meaningless in BOTH ports: the Fortran
        # leaves the previous circuit's values there, this one leaves a
        # freshly computed value for an area nobody asked about.
        area_of = int(name[-1]) if name[:-1] in ("abiy", "artic") else 0
        for i, exp in enumerate(rows):
            if area_of and area_of > kfx[i]:
                continue
            e = np.float32(exp[f])
            d = ulp_diff_floored(got[name][i], e)
            if d > worst[name][0]:
                worst[name] = (d, f"lane {i}: {got[name][i]!r} vs {e!r}")
            if name in BITWISE:
                assert d == 0.0, (
                    f"[{device}] lane {i} {name}: got {got[name][i]!r} "
                    f"expected {e!r} ({d:.2f} floored ULP) -- this field has "
                    f"no libm and no cancellation, so it must be bitwise")
            else:
                assert d <= ULP_BAR, (
                    f"[{device}] lane {i} {name}: got {got[name][i]!r} "
                    f"expected {e!r} ({d:.2f} floored ULP), in {inputs[i]}")
            exact[name][0] += d == 0.0
            exact[name][1] += 1

    for name in EXP_FIELDS:
        e, t = exact[name]
        assert e / t >= EXP_FLOOR[device], (
            f"[{device}] {name} exact {e}/{t}, worst {worst[name][0]:.2f} at "
            f"{worst[name][1]} -- EXP should cost about 1 lane in 5 at most")
    for name in LOG_FIELDS:
        e, t = exact[name]
        assert e / t >= LOG_FLOOR[device], (
            f"[{device}] {name} exact {e}/{t}, worst {worst[name][0]:.2f} at "
            f"{worst[name][1]}")

    tot_e = sum(v[0] for v in exact.values())
    tot_n = sum(v[1] for v in exact.values())
    assert tot_n == 2824, f"counted {tot_n} cells, expected 2824"
    assert tot_e / tot_n >= (0.99 if device == "cpu" else 0.95), (
        f"[{device}] exact {tot_e}/{tot_n}")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_branch_coverage():
    """Every arm the docstring claims is reached -- and the two it says are
    unreachable still are, so the docstring cannot quietly go stale."""
    _, inputs, _ = DUMP
    got = _run("cpu")
    kfx, jmode, gcdkm = RAW["kfx"], RAW["jmode"], RAW["gcdkm"]

    # the variable-length average: all three KFX values, side by side
    for k in (1, 2, 3):
        assert (kfx == k).sum() >= 20, (
            f"only {(kfx == k).sum()} lanes at KFX={k}")
    # JMODE used as an area index -- all three columns get gathered
    for j in (1, 2, 3):
        assert (jmode == j).sum() >= 5, f"JMODE={j} never selected"
    assert (jmode <= kfx).all(), (
        "a lane has JMODE > KFX, so XVE would read an ionogram column nobody "
        "built; the dump generator promises JMODE <= KFX")
    # SYSSY's distance code, both sides of the 2500 km threshold
    assert (gcdkm > 2500).sum() >= 20 and (gcdkm <= 2500).sum() >= 20

    # Es present / absent -- the ESLSM and both decile PES masks
    for name in ("ymuf", "yfot", "yhpf"):
        es = RAW[name][3]
        assert (es > 0).sum() >= 20 and (es <= 0).sum() >= 20, (
            f"{name}(4) is never both present and absent")

    # the FTAB latitude ladder, all three arms
    glavd = got["aglat"] * np.float32(R2D)
    assert (glavd <= 40).sum() >= 10
    assert ((glavd > 40) & (glavd <= 50)).sum() >= 10
    assert (glavd > 50).sum() >= 10

    # the AMAX1 floors on the signal adjustments
    assert (got["asm"] == 0.0).sum() >= 3, "ASM never clamps at 0"
    assert (got["dsu"] == np.float32(0.5)).sum() >= 3, "DSU never clamps"
    assert (got["dsl"] == np.float32(1.0)).sum() >= 3, "DSL never clamps"
    # the Greg Hand absorption floor -- the whole reason this is the 'W' port
    active = np.arange(NA)[:, None] < kfx[None, :]
    abiy = np.stack([got[f"abiy{a + 1}"] for a in range(NA)])
    assert ((abiy == np.float32(0.1)) & active).sum() >= 3, (
        "no area's ABIY hits the 0.1 floor, so the 'W' branch is untested")

    # --- and the two arms that are NOT reachable, asserted so ---------------
    assert (got["feav"] <= 0.5).sum() == 0, (
        "a lane now reaches FEAV <= 0.5, so the AFE=BFE=0 arm IS covered by "
        "this dump; update the module docstring, which says it is not")
    assert RAW["has_data"].all(), (
        "a lane now has no coefficient data, so SYSSY's ITSA-1 fallback IS "
        "covered here; update the module docstring")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_afe_zero_arm_vs_stage_a():
    """The FEAV <= 0.5 arm, against STAGE A rather than the oracle.

    THIS IS THE WEAKER CLAIM and is labelled as such: it says two Python
    transliterations agree, not that either matches the Fortran.  The dump
    cannot reach FEAV <= 0.5 (see the docstring), so the input is doctored --
    the case's E critical frequencies are dropped to 0.3 MHz AFTER the
    ionogram is built, which is physically inconsistent but leaves HTRUE and
    FVERT identical between the two ports so only the AFE/BFE branch moves.
    """
    _, inputs, _ = DUMP
    keep = [inp for inp in inputs if int(inp[2]) == 3][:4]
    assert keep, "no KFX=3 case to doctor"
    tiny = F32(0.3)

    exp_afe, exp_bfe, states = [], [], []
    for inp in keep:
        s = _layer_state(inp)
        for a in range(1, NA + 1):
            s.fi[1, a] = tiny                       # after LECDEN/GENION
        sigd, ton = SigdState(), TonState()
        st = redmap(COEFFS_DIR, "CCIR", int(inp[0]), F32(float(inp[1])))
        sigdis(s, _muf_state(inp), st, sigd, ton, float(inp[4]), int(inp[3]))
        exp_afe.append(sigd.afe)
        exp_bfe.append(sigd.bfe)
        states.append((s, st, inp))

    n = len(keep)
    args = {k: (RAW[k][..., :n] if RAW[k].ndim > 1 else RAW[k][:n]).copy()
            for k in ORDER}
    for i, (s, st, inp) in enumerate(states):
        args["kfx"][i] = s.kfx
        args["jmode"][i] = int(inp[3])
        args["gcdkm"][i] = np.float32(float(inp[4]))
        args["sys"][..., i] = st.sys
        args["perr"][..., i] = st.perr
        args["has_data"][i] = st.f2d[0, 0, 0] > 0.0
        m = _muf_state(inp)
        for name, arr in (("yfot", m.yfot), ("yhpf", m.yhpf),
                          ("ymuf", m.ymuf), ("sigl", m.sigl),
                          ("sigu", m.sigu)):
            args[name][2, i], args[name][3, i] = arr[3], arr[4]
        for a in range(1, NA + 1):
            args["fi1"][a - 1, i] = tiny
            args["glat"][a - 1, i] = s.glat[a]
            args["clck"][a - 1, i] = s.clck[a]
            args["htrue"][:, a - 1, i] = s.htrue[1:31, a]
            args["fvert"][:, a - 1, i] = s.fvert[1:31, a]

    got = _flat(*sigdis_batch(*[torch.as_tensor(args[k]) for k in ORDER]))
    assert (got["feav"] <= 0.5).all(), "the doctored input missed the arm"
    for i in range(n):
        assert got["afe"][i] == np.float32(exp_afe[i]) == np.float32(0.0)
        assert got["bfe"][i] == np.float32(exp_bfe[i]) == np.float32(0.0)


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """SIGDIS alone, from ONE host-built input set moved across.

    Everything upstream -- the ionograms, the coefficient tables -- is built
    once on the CPU and copied, so a difference here belongs to SIGDIS and
    not to LECDEN/GENION.  It is what says the CUDA column in the oracle
    comparison is this routine's EXP and ALOG10 and nothing else.
    """
    a = _run("cpu")
    b = _run("cuda")
    for name in BITWISE:
        assert np.array_equal(a[name], b[name]), (
            f"{name} differs across devices; it has no libm in it")
    for name in EXP_FIELDS + LOG_FIELDS:
        d = np.abs(a[name].astype(np.float64) - b[name].astype(np.float64))
        sp = np.spacing(np.maximum(np.abs(b[name]), 1.0))
        assert (d / sp).max() <= ULP_BAR, (
            f"{name} worst {(d / sp).max():.2f} ULP across devices")
