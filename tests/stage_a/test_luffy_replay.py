"""Stage A LUFFY end-to-end replay against the real voacapl.

The other dumps inject one routine's inputs.  This one cannot: LUFFY builds
its own ionosphere from the coefficient maps, so its "inputs" are most of the
program.  ``tools/oracle/dumps/dump_luffy.f`` therefore shims LUFFY *into* the
real binary -- card reading, REDMAP, GEOM, SANG, IONSET, CURMUF and the hour
loop all run untouched -- and records the state at entry and at exit for 10
short-path decks x 24 hours.  What this replays is composition: eighteen
routines, each already bit-exact on its own tier-B dump, running as a driver.

**One Circuit per deck, reused across the 24 hours.**  Fortran COMMON persists
between calls and /SON/ is only partly rewritten each one: every deck here
gives 9 frequencies, so FREL(10), FREL(11) and FREL(13) are zero and LUFFY
skips those columns with ``go to 265``, leaving the previous hour's numbers in
place.  A fresh Circuit per row would zero them and disagree.  The R decile
triple (D10R/D50R/D90R) carries over the same way.

/SON/ MODE and MDL are CHARACTER and feed only the output tables, so they are
not captured and not compared; everything else in the common is.

Not captured, rebuilt Python-side: the coefficient maps (REDMAP, validated by
its own dump) and the ionogram HPRIM/HTRUE/FVERT/AFAC, which LUFFY itself
fills by calling GENION and ALOSFV on the control area -- and every reader of
them takes the same area index, so no stale slot is ever read.  /ALPHA/ NYEAR
is not captured either; it selects the pre-1989 noise model on the exact
string " 2089", and these decks are 1994, so the Circuit default "" is right.

WHAT THE REPLAY FOUND
---------------------
*/allMODE/ NREL and NMMOD start at 1, not 0.*  ``blkdat.for:181`` initialises
both, and the capture caught it at the first RELBIL call of the first hour.
RELBIL returns without assigning NREL when NMMOD is non-positive, and MPATH
and SERPRB then index the mode arrays with the stale value, so the difference
is between "slot 1" and "the padding slot 0".  Fixed in ``allmodestate.py``.

THE RESIDUAL
------------
Measured over 16 decks x 24 hours x 262 columns = 100,608 cells: **99,395
bit-exact (98.79%)**, per deck between 100.0% (lonpar_m01_s100_v30) and 95.90%
(tanbel_m07_s160_v30).  Worst disagreement 5.4e-3 dB on a signal lower decile
(latok row 22, DBLOSU(8): 6.919716 vs 6.925117), 7.8e-4 relative, 11,328
floored ULP.  Median mismatch 4 ULP.

**The long path is no worse than the short path.**  The six IPFLAG-200 decks
land between 98.36% and 99.81%, inside the short path's own spread -- so
joining two independently-chosen rays across 11,000 to 21,700 km accumulates
no more error than sweeping hop counts over one ionosphere does.

**No discrete decision ever differs.**  NHP -- the hop count of the mode
RELBIL picked -- and JMODE are bit-exact in all 5,376 discrete cells across
all sixteen decks, 2,016 of them long-path, so the mode selection, the hop
sweep, the layer choice and SELTXR's ray choice at both ends all agree; only
the numbers attached to the chosen mode move.  That is asserted below,
because a regression that flipped a branch would show up there first and is
qualitatively different from arithmetic noise.

Bisected against the Fortran -- an instrumented LUFFY+FINDF+RELBIL shim built
the same way as the capture, run over tanbel_m07_s160_v30, latok_m01_s100_v20
and fbahel_m01_s100_v20 -- the residual is the [[f32-libm-noise]] floor with
two amplifiers and no third cause:

1. *FOBBY quantises to integer kHz.*  ``IFOB = 1000.*FREQ`` truncates, and
   FREQ comes through ``COS``.  Where 1000*FREQ lands within ~0.04 of an
   integer the 1-ULP cosf disagreement tips the truncation: 58 of 24,480
   entries for tanbel_m07, 41 of 13,680 for latok, 40 of 17,280 for fbahel --
   every one of the 139 with a fractional part above 0.96 or below 0.05.
   FINDF then interpolates the ionogram with ``slope = (JFHZ-IFOB(ih)) /
   (IFOB(ih+1)-IFOB(ih))``, so one kHz moves HPFLX/HTFLX/FVFLX by ~1e-3
   relative.  The fobby.py docstring already names this flip; the replay
   measures what it costs.  Force IFOB to the oracle's values and FINDF's
   entire ray table goes bit-exact for all 720 calls, taking fbahel from
   99.67% to 99.94%, latok from 97.93% to 99.13%.
2. *FDIST and GAIN subtract nearly-equal angles.*  With the ray table forced
   bit-exact, what is left in /allMODE/ is 1-30 ULP in B, TIMED, GRLOS and
   PROB.  FDIST's ``DELMOD = (PIO2 - PHE - THET)*R2D`` turns a 1-ULP error in
   ``ATAN``'s 1.4 rad result into 7 ULP of an 8-degree take-off angle -- the
   observed shift in B is 6.7e-6 deg, which is one float32 ULP at 1.4 rad
   expressed in degrees, to two figures.

Everything above FOBBY composes cleanly: the ionogram (FI/YI/HI after LECDEN's
F1 fit, FVERT, HPRIM, HTRUE, AFAC and the HTR/FNSQ profile) is bit-exact for
all 72 hours bisected, so LECDEN, GENION and ALOSFV are exonerated and the
divergence starts at FOBBY's truncation and nowhere earlier.

The bars below therefore sit well above the measured worst rather than on it,
and the exactness floor is per deck.  A tighter set would be tracking the
libm floor, not the port.
"""

import csv

import numpy as np
import pytest

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.circuit import Circuit
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.luffy import luffy
from voacap_gpu.stage_a.redmap import redmap

from .dumputil import COEFFS_DIR, GOLDEN, ulp_diff_floored

DUMPS = GOLDEN / "luffy"
DECKS = sorted(p.name[:-7] for p in DUMPS.glob("*_in.csv"))

# measured worst 11,328 floored ULP / 7.8e-4 relative / 95.90% exact per deck
ULP_BAR = 25000.0
REL_BAR = 3.0e-3
EXACT_FLOOR = 0.94

NAREA = 5                       # sample areas, /RON/ K=1..5
NLAYER = 3                      # E, F1, F2
NANT = 20
NELEV = 91                      # ARRAY's elevation axis, /cantenna/
NCOL = 262                      # /SON/ 13 x 20 + XLUF + JMODE

# /SON/ in the order LUFHDR writes it -- interleaved per index I, so the CSV
# header runs angle01,angler01,...,rneff01,angle02,...  Read from the file
# rather than assumed; this tuple only names the Circuit fields.
SON_FIELDS = ("angle", "angler", "cprob", "dblos", "dblosl", "dblosu", "dbu",
              "delay", "dbw", "nhp", "xnynois", "probmp", "reliab", "sndb",
              "snpr", "snrlw", "snrup", "sprob", "vhigh", "rneff")
# the columns that encode a decision rather than a magnitude
DISCRETE = tuple(f"nhp{i:02d}" for i in range(1, 14)) + ("jmode",)


def _rows(path):
    """CSV -> list of {column: float}.

    The Fortran's ``ES18.9E3,999(',',ES18.9E3)`` leaves a trailing comma on
    every data line (format control emits the literal before discovering the
    iolist is spent), so the last field is empty; zip against the header
    drops it.
    """
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        return [dict(zip(header, (float(x) for x in row))) for row in r]


def _load(deck):
    if not (DUMPS / f"{deck}_out.csv").exists():
        return None
    with open(DUMPS / f"{deck}_out.csv", newline="") as f:
        out_header = [h.strip() for h in next(csv.reader(f))]
    return (_rows(DUMPS / f"{deck}_in.csv"), _rows(DUMPS / f"{deck}_out.csv"),
            _rows(DUMPS / f"{deck}_ant.csv"), out_header)


def _antennas(ant_rows) -> CantennaState:
    """/cantenna/ + /pantenna/ as ANTCALC left them.

    Injected rather than rebuilt from the .VOA file because OFFAZIM -- the
    angle between the antenna's main beam and the path -- is derived by
    ANTCALC from the ANTENNA card's beam heading and the path bearing, and
    nothing in the antenna file carries it.  ``load_p2p`` would leave 0 where
    the real value is (for lonpar) 58.1 degrees.  COND and DIEL are on the
    common and captured, but the ported GAIN reads ground constants from
    /GEOG/ SIGPAT/EPSPAT instead, so they have no slot here.
    """
    cant = CantennaState()
    for v in ant_rows:
        iant = int(v["iant"])
        ifreq = int(v["ifreq"])
        cant.numants = max(cant.numants, iant)
        cant.iats[iant] = int(v["itr"])
        cant.xfqs[iant] = F32(v["xfqs"])
        cant.xfqe[iant] = F32(v["xfqe"])
        cant.beammain[iant] = F32(v["beammain"])
        cant.offazim[iant] = F32(v["offazim"])
        cant.pwrkw[iant] = F32(v["pwrkw"])
        cant.pwrdba[iant] = F32(v["pwrdba"])
        cant.aeff[ifreq, iant] = F32(v["aeff"])
        for j in range(1, NELEV + 1):        # g01..g91 are elevations 1..91
            cant.array[ifreq, j, iant] = F32(v[f"g{j:02d}"])
    assert cant.numants <= NANT
    return cant


def _apply(c, v) -> None:
    """Install one captured entry state into the Circuit."""
    lay, muf, rays, tim, frq = c.layers, c.muf, c.rays, c.time, c.frq

    # /SSP/, /CONTRL/, /ION/, /OUTPRT/ and the loose scalars.  NPSL (GEOM's
    # through-the-antipode flag), FLUX (the F10.7 index the output header
    # prints) and PWR are captured and have no reader below LUFFY -- PWR in
    # particular is superseded by /pantenna/ PWRDBA, which is what PWRDB
    # reads -- so none of the three gets a Circuit field.
    lay.month = int(v["month"])
    lay.ssn = F32(v["ssn"])
    c.method = int(v["method"])
    c.mspec = int(v["mspec"])
    c.lufp = int(v["lufp"])
    c.noise = int(v["noise"])
    c.want_service_prob = v["linbot14"] > 0

    lay.km = int(v["km"])
    lay.kfx = int(v["kfx"])
    lay.iedp = int(v["iedp"])
    rays.nang = int(v["nang"])

    tim.it = int(v["it"])
    tim.gmt = F32(v["gmt"])
    tim.gmtr = F32(v["gmtr"])

    # /DON/
    c.gcd = F32(v["gcd"])
    c.gcdkm = F32(v["gcdkm"])
    c.amind = F32(v["amind"])
    c.rsn = F32(v["rsn"])
    c.dmp = F32(v["dmp"])
    c.pmp = F32(v["pmp"])
    c.rlat = F32(v["rlat"])
    c.rlong = F32(v["rlong"])
    c.rlongd = F32(v["rlongd"])

    # /MUFS/ -- the hour-indexed slots at IT, then the per-layer arrays
    it = tim.it
    for name in ("emuf", "f1muf", "f2muf", "esmuf", "allmuf", "fot", "hpf",
                 "angmuf"):
        getattr(muf, name)[it] = F32(v[name])
    muf.modmuf = int(v["modmuf"])
    for ly in range(1, 5):
        for name in ("sigl", "sigu", "delmuf", "hpmuf", "htmuf", "fvmuf",
                     "afmuf", "yfot", "yhpf", "ymuf"):
            getattr(muf, name)[ly] = F32(v[f"{name}{ly:02d}"])
        muf.nhopmf[ly] = int(v[f"nhopmf{ly:02d}"])

    # /RON/, /ES/, /MFAC/, /GEOG/ -- per sample area
    for k in range(1, NAREA + 1):
        for name in ("clat", "clong", "glat", "rd", "gyz", "rat", "gmdip",
                     "clck", "abiy", "artic", "sigpat", "epspat", "f2m3",
                     "hpf2", "zenang", "zenmax"):
            getattr(lay, name)[k] = F32(v[f"{name}{k:02d}"])
        lay.hs[k] = F32(v[f"hs{k:02d}"])
        for ly in range(1, NLAYER + 1):
            # fi{L}{K} is FI(L,K): layer inner, area outer, as LUFHDR writes
            lay.fi[ly, k] = F32(v[f"fi{ly}{k}"])
            lay.yi[ly, k] = F32(v[f"yi{ly}{k}"])
            lay.hi[ly, k] = F32(v[f"hi{ly}{k}"])
            lay.fs[ly, k] = F32(v[f"fs{ly}{k}"])
    for ly in range(1, NLAYER + 1):
        lay.fsecv[ly] = F32(v[f"fsecv{ly:02d}"])
        lay.ielect[ly] = int(v[f"ielect{ly:02d}"])
    for ly in range(1, 5):
        lay.psc[ly] = F32(v[f"psc{ly:02d}"])

    # /FRQ/ the user frequency list (slot 14 is DECRED's flag, not a
    # frequency) and /RAYS/ the take-off angle scan
    for i in range(1, 15):
        frq.frel[i] = F32(v[f"frel{i:02d}"])
    for i in range(1, 41):
        rays.ang[i] = F32(v[f"ang{i:02d}"])


def _got(c) -> dict:
    g = {f"{name}{i:02d}": getattr(c.son, name)[i]
         for name in SON_FIELDS for i in range(1, 14)}
    g["xluf"] = c.muf.xluf[c.time.it]
    g["jmode"] = c.frq.jmode
    return g


def replay(deck):
    """Yield (row index, got, expected row, header) for one deck's 24 hours.

    The Circuit is built once and reused so the COMMON persistence is
    reproduced.  Shared with the measurement scripts.
    """
    loaded = _load(deck)
    if loaded is None:
        return
    ins, outs, ants, header = loaded
    assert len(ins) == len(outs)
    months = {int(v["month"]) for v in ins}
    ssns = {float(v["ssn"]) for v in ins}
    assert len(months) == 1 and len(ssns) == 1, "one map set per deck"
    coeff = "URSI" if "_ursi" in deck else "CCIR"
    c = Circuit(iono=redmap(COEFFS_DIR, coeff, months.pop(), F32(ssns.pop())))
    c.cant = _antennas(ants)
    # SETOUT's preset, once before the hour loop (setout.for:29).  On the
    # system-performance arm LUFFY never assigns XLUF, so -1 is what the
    # capture records for all 24 hours -- the column proves the driver leaves
    # it alone, not that it computes it.
    c.muf.xluf[:] = F32(-1.0)
    for k, (vin, vout) in enumerate(zip(ins, outs)):
        _apply(c, vin)
        luffy(c, int(vin["ipflag"]))
        yield k, _got(c), vout, header


@pytest.mark.skipif(not DECKS, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_luffy_dumps.sh")
@pytest.mark.parametrize("deck", DECKS)
def test_luffy_replay_vs_oracle(deck):
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    for k, got, vout, header in replay(deck):
        for name in header:
            g, e = got[name], np.float32(vout[name])
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"{deck} row {k} {name}"
            where = (f"{deck} row {k} (hour {k + 1}) {name}: got {g!r} "
                     f"expected {e!r}")
            assert d <= ULP_BAR, (
                f"{where} ({abs(float(g) - float(e)):.3e} abs, {d:.1f} "
                "floored ULP)")
            if d:
                assert name not in DISCRETE, (
                    f"{where} -- a discrete decision changed, which is not "
                    "the arithmetic noise floor")
                assert float(e) != 0.0, f"{where} -- expected an exact zero"
                rel = abs(float(g) - float(e)) / abs(float(e))
                assert rel <= REL_BAR, f"{where} ({rel:.2e} relative)"
            total += 1
            exact += d == 0.0
    assert total == 24 * NCOL, f"{deck}: {total} cells"
    assert exact / total >= EXACT_FLOOR, (
        f"{deck}: exact {exact}/{total} (worst {worst_ulp:.1f} ULP "
        f"at {worst_at})")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_the_capture_covers_the_branches_that_matter():
    """A replay is only worth its coverage.  These are the decisions inside
    LUFFY that the captured decks have to exercise between them.

    ``short_path`` and ``long_path`` are the two that would be easiest to lose
    by accident: the runner clears every CSV before it writes, so capturing a
    subset of decks silently deletes the rest, and a replay over short-path
    decks alone would still pass while validating half the driver.
    """
    seen = {"one_area": 0, "three_areas": 0, "five_areas": 0,
            "mspec_121": 0, "mspec_0": 0, "method_25": 0,
            "empty_frequency_slot": 0, "short_path": 0, "long_path": 0}
    for deck in DECKS:
        v = _rows(DUMPS / f"{deck}_in.csv")[0]
        seen["one_area"] += int(v["kfx"]) == 1
        seen["three_areas"] += int(v["kfx"]) == 2
        seen["five_areas"] += int(v["kfx"]) == 3
        seen["mspec_121"] += int(v["mspec"]) == 121
        seen["mspec_0"] += int(v["mspec"]) == 0
        seen["method_25"] += int(v["method"]) == 25
        seen["empty_frequency_slot"] += any(
            v[f"frel{i:02d}"] <= 0.0 for i in range(1, 12))
        seen["short_path"] += int(v["ipflag"]) in (100, 300)
        seen["long_path"] += int(v["ipflag"]) in (200, 400)
    for arm, n in seen.items():
        assert n, f"no deck exercises {arm} ({seen})"
