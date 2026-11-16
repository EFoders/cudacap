"""What -DVOACAP_FAST_MATH costs, measured on the 504-circuit-hour replay.

Stage C's default arithmetic convention evaluates every transcendental in
binary64 and rounds ONCE to binary32 (``cuda/f32math.cuh``).  That is what
makes it bit-identical to Stage A, and it is not negotiable as a default --
``test_luffy_replay.py`` holds it at 100%.  But this is a consumer card where
binary64 runs at ~1/64 the binary32 rate, and LUFFY is dense with
cos/sin/atan/asin/acos/exp/log/log10/pow, so the convention is expensive.
``-DVOACAP_FAST_MATH`` compiles the SAME headers against CUDA's native f32
libm instead.  This file is the price tag.

THE SPEED (kernel only, batch already resident, one thread per circuit-hour,
this GPU, interleaved A/B, minimum of five trials on a contended machine)::

    circuit-hours      504     8,064    32,256
    strict      ms  127.72     75.09    121.49
    fast        ms   42.74     31.65     64.21
    speedup          2.99x     2.37x     1.89x

Register count is unchanged at 168, so nothing is bought with occupancy; the
win is the f64 transcendental units themselves, and it is LARGEST on a small
batch because there the kernel is latency-bound on its own serial dependency
chain and the f64 calls are the long links in it.

THE PRICE
---------
Per-helper, over 400k arguments (``test_fastmath_helper_ulp``)::

    sqrt32   0 ulp  100.000% exact      atan32  1 ulp  96.7%
    log32    1 ulp   94.8%              pow32   1 ulp  93.7%
    acos32   1 ulp   90.6%              sin32   1 ulp  80.1%
    cos32    1 ulp   81.1%              asin32  1 ulp  74.5%
    exp32    2 ulp   69.6%              log10_32 2 ulp  66.2%

``sqrt32`` is FREE and provably so: ``sqrtf`` under --prec-sqrt=true is
correctly rounded, and so is rounding a correctly rounded binary64 sqrt down
to binary32 (53 >= 2*24+2, so no double-rounding case exists).  It is asserted
at zero ulp, not bounded.

End to end over 504 circuit-hours x every field of CircuitState and /RAYS/ --
7,012,152 cells::

    6,965,858 bit-exact (99.3398%)
    worst struct  anois 81.55%,  son 87.97%,  ndiag 95.52%, allm 96.38%
    muf, frq, time and every loose scalar: 100.0000%
    worst absolute deviation anywhere  0.0509 (REFLX GDFLX/DSKPKM, km)
    worst deviation in a dB column     0.0127 dB (/allMODE/ SIGPOW, SN, FLDST)

Headline outputs::

    MUF / FOT / XLUF / HPF / EMUF / F1MUF / F2MUF / ESMUF   100% BIT-EXACT
    DBU     73.0% exact, worst 2.4e-4 dB      DBLOS  80.6%, worst 2.4e-4 dB
    SNDB    73.5% exact, worst 2.4e-4 dB      RELIAB 82.0%, worst 2.7e-6
    SPROB   74.1% exact, worst 1.1e-6         SIGPOW 92.0%, worst 1.3e-2 dB

NO DISCRETE DECISION MOVES, and that is the finding that matters
---------------------------------------------------------------
Every integer field is bit-identical except FOBBY's IFOB: NHP (the hop count
of the mode RELBIL picked), JMODE, MODE, MODER, MDL, NMODE in both /ZON/ and
/allMODE/, MODMUF, NHOPMF, IMODE, ISKP, IAFTXR, ITMOD, LTXRGM, ITXRCP, the
noise-model selectors and NANG are ALL exact in all 504 circuit-hours.  Fast
math moves the numbers attached to the chosen mode; it does not choose a
different mode.

THE KNOWN AMPLIFIER, COUNTED NOT ESTIMATED
------------------------------------------
FOBBY truncates ``1000.*FREQ`` to integer kHz, so a 1-ulp cosine there becomes
a whole-kHz step and FINDF interpolates on it.  Stage A's replay measured that
as the DOMINANT residual against the real Fortran.  Counted here over the
830,940 cells FOBBY actually writes:

    strict build vs Stage A       0 flips
    fast-math build vs Stage A   96 flips (0.0116%), every one exactly 1 kHz,
                                 touching 61 of 504 circuit-hours

For scale, ``tests/scalar/test_luffy_replay.py`` measured Stage A's OWN IFOB
against the Fortran's at 139 flips in 55,440 entries -- **0.251%**.  So fast
math adds a kHz-flip rate roughly TWENTY TIMES SMALLER than the one the port
already carries against the oracle, and none of the 96 propagated into a
discrete decision.

CONTEXT: THE EXISTING FLOOR
---------------------------
Stage A against the actual Fortran is 98.705% bit-exact over 132,048 cells,
worst deviation 5.4e-3 dB (DBLOSU, 7.8e-4 relative).  Fast math against Stage
A is 99.34% exact with a worst HEADLINE deviation of 2.4e-4 dB -- more than an
order of magnitude INSIDE the error the port already has against the thing it
is modelling.  Only /allMODE/ SIGPOW's 1.3e-2 dB exceeds the Fortran floor's
5.4e-3, and it does so on a per-mode table column rather than a headline
column, at 2.3x, against a 99%-of-Fortran acceptance bar.

THE BARS BELOW ARE SET FROM THOSE MEASUREMENTS, deliberately above them (5x to
40x), so a toolchain or libm change that DEGRADES fast math fails loudly while
ordinary rebuild noise does not.  The discrete assertions are the exception:
they are zero, because "no mode-selection flip" is a claim, not a tolerance.
"""

import contextlib

import numpy as np
import pytest

import voacap_gpu.cuda.driver as drv
from voacap_gpu.cuda import luffy_rigs
from voacap_gpu.scalar import f32math as fm
from voacap_gpu.cuda.compile import load
from voacap_gpu.cuda.state import RAYS_DTYPE

from tests.scalar.test_luffy_replay import DECKS
from tests.cuda.harness import fetch, launch_rows, requires_cuda
from tests.cuda.test_f32math import _SRC as F32MATH_SRC, _cases
from tests.cuda.test_luffy_replay import _fields, _get, _replay

pytestmark = requires_cuda

FAST_OPT = ("-DVOACAP_FAST_MATH",)

# --- the measured bars -----------------------------------------------------
# Every one is set ABOVE the observed worst (the comment gives the
# measurement), so this file fails on degradation, not on rebuild noise.
MAX_ULP = 4                     # observed 2 (exp32, log10_32); 1 elsewhere
MAX_ULP_POW = 8                 # observed 1; CUDA documents powf at 8
MIN_EXACT_OVERALL = 0.985       # observed 0.993398
MIN_EXACT_PER_STRUCT = 0.75     # observed worst 0.8155 (/ANOIS/)
MAX_ABS_ANYWHERE = 0.5          # observed 0.050903 (REFLX GDFLX, km)
MAX_IFOB_FLIP_FRAC = 5.0e-4     # observed 1.155e-4 (96 of 830,940)
MAX_IFOB_KHZ = 2                # observed 1 kHz, always exactly 1

# field -> worst permitted |deviation|, in that field's own unit.
HEADLINE_BARS = {
    "muf.emuf": 0.01, "muf.f1muf": 0.01, "muf.f2muf": 0.01,      # MHz, all
    "muf.esmuf": 0.01, "muf.allmuf": 0.01, "muf.fot": 0.01,      # observed 0
    "muf.xluf": 0.01, "muf.hpf": 0.01,
    "son.dbu": 5.0e-3,          # dB,  observed 2.4e-4
    "son.dblos": 5.0e-3,        # dB,  observed 2.4e-4
    "son.sndb": 5.0e-3,         # dB,  observed 2.4e-4
    "son.reliab": 1.0e-4,       #      observed 2.7e-6
    "son.sprob": 1.0e-4,        #      observed 1.1e-6
    "allm.sigpow": 0.1,         # dB,  observed 1.3e-2
    "zon.sigpow": 0.1,          # dB,  observed 2.2e-3
}

# The integer fields that MUST NOT MOVE.  rays.ifob is excluded and bounded
# separately: it is the one discrete quantity fast math is allowed to disturb,
# because FOBBY's kHz truncation is a quantiser, not a decision.
DISCRETE_EXEMPT = {"rays.ifob"}


# ---------------------------------------------------------------------------
# compiling and launching the fast-math build
# ---------------------------------------------------------------------------

_FAST = {}


def fast_kernel_module(groups=1, fast_math=False):
    """The LUFFY batch module, compiled with -DVOACAP_FAST_MATH.

    Same source, same strict options (--fmad=false and friends still hold --
    fast math is about libm, not about FMA contraction), one extra -D.

    The signature mirrors ``driver.load_kernel``, which ``stage_luffy`` now
    calls with ``groups``/``fast_math``; ``fast_math`` is ignored because this
    shim's whole job is to force it on.  ``groups`` IS honoured, so the A/B
    still works if a caller packs circuit-hours per warp.
    """
    key = int(groups)
    if key not in _FAST:
        src = getattr(drv, "_KERNEL_SRC", None)
        assert src is not None, ("stage_c.driver no longer exposes "
                                 "_KERNEL_SRC; this A/B needs the same "
                                 "translation unit the strict build uses")
        opts = FAST_OPT + ((f"-DVC_WARP_GROUPS={key}",) if key != 1 else ())
        _FAST[key] = load(src, extra_opts=opts)
    return _FAST[key]


@contextlib.contextmanager
def fast_math():
    """Make ``driver.stage_luffy`` pick up the fast-math module.

    The driver is not modified: ``stage_luffy`` looks ``load_kernel`` up in
    its own module globals (``cuda/luffy_rigs.py``; ``driver`` is only a
    re-exporting facade) on every call, so swapping it THERE redirects the
    launch and nothing else.
    """
    orig = luffy_rigs.load_kernel
    luffy_rigs.load_kernel = fast_kernel_module
    try:
        yield
    finally:
        luffy_rigs.load_kernel = orig


_RUNS = {}


def _both_builds():
    """(strict, fast) results of the whole 504-circuit-hour replay, cached."""
    if "v" in _RUNS:
        return _RUNS["v"]
    v = _replay()
    args = (v["entries"], v["entries_rays"], v["ipflags"],
            drv.shared_iono_tables(v["ionos"]), v["iono_idx"],
            drv.shared_antenna_tables(v["cants"]), v["ant_idx"])
    strict = drv.launch_luffy(*args)
    with fast_math():
        fast = drv.launch_luffy(*args)
    _RUNS["v"] = (v, strict, fast)
    return _RUNS["v"]


def _walk(v, res):
    """(label, got, expected) for every leaf field of both state buffers."""
    got, got_rays, _status = res
    for dt, g_all, e_all, prefix in ((drv.CIRCUIT_DTYPE, got, v["exits"], ""),
                                     (RAYS_DTYPE, got_rays, v["exits_rays"],
                                      "rays.")):
        for label, path in _fields(dt):
            yield (prefix + label, np.asarray(_get(g_all, path)),
                   np.asarray(_get(e_all, path)))


def _same(g, e):
    if g.dtype.kind == "f":
        return g.view(np.uint32) == e.view(np.uint32)
    return g == e


# ---------------------------------------------------------------------------
# 1. the helpers themselves
# ---------------------------------------------------------------------------

def _ulp(a, b):
    """Floored ULP distance between two float32 arrays."""
    def key(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, np.int64(-2147483648) - i, i)
    return np.abs(key(a) - key(b))


def test_fastmath_helper_ulp():
    """Each helper's error against the binary64 convention, bounded."""
    x = _cases(400_000, seed=17)
    n = len(x)
    mod = load(F32MATH_SRC, extra_opts=FAST_OPT)
    outs = [np.zeros(n, np.float32) for _ in range(9)]
    dev = launch_rows(mod.get_function("unary"), n, [x] + outs)
    got = [fetch(d) for d in dev[1:10]]

    p = np.float32(np.abs(x)) + np.float32(1e-6)
    u = np.fmod(x, np.float32(1.0))
    pairs = [("sin32", fm.sin32, x), ("cos32", fm.cos32, x),
             ("atan32", fm.atan32, x), ("exp32", fm.exp32, x),
             ("log32", fm.log32, p), ("log10_32", fm.log10_32, p),
             ("sqrt32", fm.sqrt32, p), ("asin32", fm.asin32, u),
             ("acos32", fm.acos32, u)]
    print("\nfast-math helper cost vs the binary64 convention:")
    worst = {}
    for g, (name, f, arg) in zip(got, pairs):
        d = _ulp(g, f(arg))
        worst[name] = int(d.max())
        print(f"  {name:<10} max {int(d.max())} ulp, "
              f"{100.0 * (d == 0).mean():6.2f}% bit-exact, "
              f"mean {d.mean():.4f} ulp")

    # sqrt32 is EXACT, and provably so rather than empirically: sqrtf is
    # correctly rounded under --prec-sqrt=true, and a correctly rounded
    # binary64 sqrt of a binary32 argument cannot double-round wrong
    # (53 >= 2*24+2).  Switching it is free speed, so it is asserted at zero.
    assert worst["sqrt32"] == 0, (
        f"sqrtf is no longer bit-identical to the binary64 path "
        f"({worst['sqrt32']} ulp) -- --prec-sqrt=true may have been lost")

    for name, w in worst.items():
        assert w <= MAX_ULP, (
            f"{name} degraded to {w} ulp under fast math (bar {MAX_ULP}); "
            "the whole accuracy budget in this file was measured at <= 2")

    rng = np.random.default_rng(11)
    px = rng.uniform(0.01, 50.0, 200_000).astype(np.float32)
    py = rng.uniform(-3.0, 3.0, 200_000).astype(np.float32)
    out = np.zeros_like(px)
    dev = launch_rows(mod.get_function("pow_probe"), len(px), [px, py, out])
    d = _ulp(fetch(dev[2]), fm.pow32(px, py))
    print(f"  {'pow32':<10} max {int(d.max())} ulp, "
          f"{100.0 * (d == 0).mean():6.2f}% bit-exact")
    assert int(d.max()) <= MAX_ULP_POW, f"pow32 {int(d.max())} ulp"


def test_default_build_is_untouched_by_the_macro():
    """The macro is OPT-IN: without -D, the helpers are still bit-exact.

    ``test_f32math.py`` already proves this for the default build; what is
    added here is that the default build and the fast one are DIFFERENT, so a
    header edit that accidentally made fast math the default (or made the
    macro a no-op) cannot pass silently.
    """
    x = _cases(50_000, seed=23)
    n = len(x)
    outs = [np.zeros(n, np.float32) for _ in range(9)]
    strict = load(F32MATH_SRC)
    fastm = load(F32MATH_SRC, extra_opts=FAST_OPT)
    ds = launch_rows(strict.get_function("unary"), n, [x] + outs)
    df = launch_rows(fastm.get_function("unary"), n, [x] + outs)
    s_cos, f_cos = fetch(ds[2]), fetch(df[2])
    assert (s_cos.view(np.uint32) == fm.cos32(x).view(np.uint32)).all(), (
        "the DEFAULT build is no longer bit-exact against Stage A")
    assert not (s_cos.view(np.uint32) == f_cos.view(np.uint32)).all(), (
        "-DVOACAP_FAST_MATH changed nothing -- the macro guard is dead")


# ---------------------------------------------------------------------------
# 2. the replay, both builds
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_strict_build_is_the_anchor():
    """The A/B is only meaningful if the strict side is still perfect."""
    v, strict, _fast = _both_builds()
    bad = [(lab, int((~_same(g, e)).sum()))
           for lab, g, e in _walk(v, strict) if not _same(g, e).all()]
    assert not bad, f"the DEFAULT build drifted from Stage A: {bad[:10]}"


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_fastmath_no_discrete_decision_changes():
    """THE headline question: does fast math ever pick a different answer?

    Continuous dB columns drifting in the last bits is noise.  A different hop
    count, layer or mode is a DIFFERENT ANSWER, and no percentage captures
    that -- so every integer field is required exact, with FOBBY's kHz
    quantiser the single documented exception, bounded in its own test.
    """
    v, _strict, fast = _both_builds()
    moved = []
    checked = 0
    for label, g, e in _walk(v, fast):
        if g.dtype.kind == "f" or label in DISCRETE_EXEMPT:
            continue
        checked += 1
        n = int((g != e).sum())
        if n:
            j = int(np.flatnonzero((g != e).ravel())[0])
            moved.append(f"{label} {n}/{g.size} (e.g. {e.ravel()[j]} -> "
                         f"{g.ravel()[j]})")
    print(f"\n{checked} discrete fields checked, {len(moved)} moved")
    assert not moved, (
        "fast math CHANGED A DISCRETE DECISION -- this is not a tolerance "
        "question, the model is answering differently: " + "; ".join(moved))


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_fastmath_fobby_khz_flip_rate():
    """FOBBY's ``IFOB = 1000.*FREQ`` truncation, counted rather than estimated.

    A 1-ulp cosine tips the truncation wherever 1000*FREQ lands within about
    1e-3 of an integer, and FINDF then interpolates the ionogram on the
    flipped kHz -- Stage A's replay bisected exactly this as the DOMINANT
    residual against the real Fortran (0.251% of entries there).  So the flip
    is counted here over the cells FOBBY actually writes, and bounded.
    """
    v, strict, fast = _both_builds()
    e = np.asarray(v["exits_rays"]["ifob"])
    nang = np.asarray(v["exits_rays"]["nang"])
    n = e.shape[0]

    # the cells FOBBY writes: IA in 1..NANG, IH in 1..30, every sample area
    # that has a table at all.  (This set turns out to be exactly the nonzero
    # set, which is the cheap cross-check.)
    written = np.zeros(e.shape, bool)
    for i in range(n):
        for k in range(6):
            if e[i, :, :, k].any():
                written[i, 1:nang[i] + 1, 1:31, k] = True
    ncells = int(written.sum())
    assert ncells == int((e != 0).sum()), "IFOB live-cell model disagrees"

    s_flip = np.asarray(strict[1]["ifob"]) != e
    f_flip = np.asarray(fast[1]["ifob"]) != e
    assert int(s_flip.sum()) == 0, "the strict build flipped an IFOB kHz"
    assert not (f_flip & ~written).any(), "an IFOB flip outside FOBBY's range"

    nf = int(f_flip.sum())
    frac = nf / ncells
    d = np.abs(np.asarray(fast[1]["ifob"]).astype(np.int64) - e.astype(np.int64))
    hours = int(f_flip.reshape(n, -1).any(axis=1).sum())
    print(f"\nFOBBY IFOB over {ncells:,} written cells: strict 0 flips, "
          f"fast {nf} ({100.0 * frac:.4f}%), max {int(d.max())} kHz, "
          f"{hours}/{n} circuit-hours touched")
    print("  Stage A's own IFOB vs the Fortran's: 139/55,440 = 0.2507% "
          "(tests/scalar/test_luffy_replay.py)")

    assert frac <= MAX_IFOB_FLIP_FRAC, (
        f"FOBBY kHz flip rate {100 * frac:.4f}% exceeds the "
        f"{100 * MAX_IFOB_FLIP_FRAC:.4f}% bar; measured 0.0116%")
    assert int(d.max()) <= MAX_IFOB_KHZ, (
        f"an IFOB flip of {int(d.max())} kHz -- measured flips are all 1, so "
        "this is a bigger cosine error than fast math is meant to cost")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_fastmath_exactness_and_worst_deviation():
    """Per-struct bit-exact fractions and the worst deviation anywhere."""
    v, _strict, fast = _both_builds()
    per_struct = {}
    worst = (0.0, "", None)
    for label, g, e in _walk(v, fast):
        same = _same(g, e)
        block = label.split(".")[0]
        h, t = per_struct.get(block, (0, 0))
        per_struct[block] = (h + int(same.sum()), t + int(same.size))
        if g.dtype.kind == "f" and not same.all():
            gd, ed = g.astype(np.float64), e.astype(np.float64)
            fin = np.isfinite(gd) & np.isfinite(ed)
            dv = np.where(fin, np.abs(gd - ed), 0.0)
            if dv.max() > worst[0]:
                j = int(np.argmax(dv))
                worst = (float(dv.max()), label,
                         (float(ed.ravel()[j]), float(gd.ravel()[j])))
    hits = sum(h for h, _ in per_struct.values())
    tots = sum(t for _, t in per_struct.values())
    frac = hits / tots
    print(f"\nfast math vs Stage A: {hits:,}/{tots:,} cells bit-exact "
          f"({100.0 * frac:.4f}%)")
    for block, (h, t) in sorted(per_struct.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        if h < t:
            print(f"  {block:<8s} {h:>9d}/{t:<9d} {100.0 * h / t:8.4f}%")
    print(f"worst absolute deviation {worst[0]:.6g} in {worst[1]} "
          f"({worst[2][0]:.7g} -> {worst[2][1]:.7g})")

    assert frac >= MIN_EXACT_OVERALL, (
        f"only {100 * frac:.4f}% of cells bit-exact under fast math; "
        f"measured 99.3398%, bar {100 * MIN_EXACT_OVERALL}%")
    low = {b: h / t for b, (h, t) in per_struct.items()
           if h / t < MIN_EXACT_PER_STRUCT}
    assert not low, (
        f"struct(s) below the {100 * MIN_EXACT_PER_STRUCT}% per-struct bar: "
        + ", ".join(f"{b} {100 * f:.2f}%" for b, f in low.items()))
    assert worst[0] <= MAX_ABS_ANYWHERE, (
        f"worst deviation {worst[0]:.6g} in {worst[1]} exceeds "
        f"{MAX_ABS_ANYWHERE}; measured 0.0509 (REFLX GDFLX, km)")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_fastmath_headline_outputs():
    """MUF, LUF, signal power, reliability, SNR and service probability.

    These are the columns a user reads.  Each carries its own bar in its own
    unit; the MUF family is currently BIT-EXACT and its 0.01 MHz bar is there
    to catch a regression, not to license drift.
    """
    v, _strict, fast = _both_builds()
    seen = {}
    for label, g, e in _walk(v, fast):
        if label not in HEADLINE_BARS:
            continue
        gd, ed = g.astype(np.float64), e.astype(np.float64)
        fin = np.isfinite(gd) & np.isfinite(ed)
        dv = np.where(fin, np.abs(gd - ed), 0.0)
        same = _same(g, e)
        seen[label] = (float(dv.max()), float(same.mean()), int(g.size))
    missing = set(HEADLINE_BARS) - set(seen)
    assert not missing, f"headline field(s) not found in the state: {missing}"

    print(f"\n{'field':<14} {'exact%':>9} {'worst |d|':>12} {'bar':>9}")
    over = []
    for label, bar in HEADLINE_BARS.items():
        d, ex, n = seen[label]
        print(f"{label:<14} {100.0 * ex:>8.4f}% {d:>12.5g} {bar:>9g}")
        if d > bar:
            over.append(f"{label} {d:.5g} > {bar:g}")
    assert not over, "headline output(s) past their bar: " + "; ".join(over)
