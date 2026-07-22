"""LUFFY — one hour of system performance, over the whole frequency list.

This is the driver every routine ported so far hangs off.  It runs once per
hour and does three things in order: pick the controlling sample area and
build its ionosphere; loop over the eleven frequencies; and for each one,
sweep the plausible hop counts, collect every mode that exists, and hand the
collection to RELBIL for a single answer.

    SELMOD                        which sample area controls the path
    ANOIS1                        noise at 1 MHz at the receiver
    LECDEN/GENION/FOBBY/ALOSFV    electron profile -> ionogram -> reflectrix
    SETLNG, SIGDIS                fill the other areas, set the loss spread
    for each frequency:
        ALLMODES(0)               clear the accumulator
        GENOIS                    noise at this frequency
        FINDF                     ray set for this frequency
        for each hop count:
            FDIST, INMUF          up to six modes, plus the over-the-MUF one
            ALLMODES              accumulate them
        ESMOD, ESREG              the sporadic-E slots
        ALLMODES(2)               accumulate those
        GENOIS, RELBIL            noise again, then the combined distribution
        SERPRB, MPATH             service and multipath probability

IPFLAG selects the model: 100 short-path system performance, 200 long-path,
300 short-path LUF, 400 long-path LUF.  **Only the short-path arms (100 and
300) are implemented here**; the long-path body raises.

Four notes on the original that shape this port.

**IPFG is re-read from IPFLAG at the top of every frequency iteration.**  The
entry code may have raised it to 200 (line 90, "use long path method") to
select the long-path *setup* for a path beyond the smoothing distance, but
that assignment does not survive into the loop, so the per-frequency *body*
reverts to whatever the caller asked for.  A path can therefore run the
long-path area selection and the short-path mode calculation.  Reachable only
from the CIRAF driver: HFMUFS independently forces IPFLAG to 200 for METHOD 30
beyond 10000 km (via ``meth``, which it sets to 30 whenever MSPEC is 121), so
for point-to-point the two agree and the quirk stays dormant.

**Label 300's search for the best reliability does not update its running
maximum.**  REL is set from RELIAB(1) and never reassigned inside the loop, so
IG ends up being the LAST frequency that beats slot 1 rather than the best
one.  The source flags this itself, in an inline comment asking whether the
test should have been ``RELIAB(IF).gt.RELIAB(IG)``.  Transliterated as
written; see ``_no_luf``.

**Two locals are dead**: IHMX (an upper hop bound computed from NHOPMF and
never read) and IXHP (a hop counter incremented and never read).  Noted where
they would go rather than transcribed.

**The 7000-10000 km recycle needs the long path.**  With MSPEC 121 in that
band the routine runs the short path, jumps back to label 83 with IPFLAG
forced to 200, runs the long path, and blends the two.  That is METHOD 30's
normal behaviour for a long-haul circuit, not an exotic corner.  The /SON/
columns the blend consumes are saved per frequency into (13,2) temporaries at
lines 330-369; that save is deliberately NOT transcribed here, because
nothing this port can run would ever read it back.  It lands with the long
path.
"""

from .allmodes import allmodes
from .alosfv import alosfv
from .anois1 import anois1
from .esmod import esmod
from .esreg import esreg
from .f32math import F32
from .fdist import fdist
from .findf import findf
from .fobby import fobby
from .frqcom import frqcom
from .genion import genion
from .genois import genois
from .inmuf import inmuf
from .lecden import lecden
from .mpath import mpath
from .relbil import relbil
from .selmod import selmod
from .serprb import serprb
from .setlng import setlng
from .sigdis import sigdis

NFREQ = 12                      # DO 265 IFX=1,12 -- 11 frequencies plus the MUF
I_LUF = 13                      # FREA(13)/MDL(13)/NHP(13), the summary slot
_SMOOTH_MIN = F32(7000.0)
_SMOOTH_MAX = F32(10000.0)
_PCT = F32(0.01)
_ES_FVAL = F32(999.0)           # ALLMODES(2,FVAL): the sentinel "frequency"
_MAX_EXTRA_HOPS = 2             # IHSTP = MIN0(IHMAX, IHSRT+2)


class LongPathNotImplemented(NotImplementedError):
    """The long-path arm (GMLOSS + SELTXR + LNGPAT) is not ported yet."""


def luffy(circ, ipflag: int) -> None:
    """Run one hour over the frequency complement.  Mutates `circ` throughout.

    circ: Circuit; ipflag: 100 short-path system performance or 300 short-path
    LUF.  /SON/ is the result -- one column per frequency -- plus /MUFS/ XLUF
    and /FRQ/ FREA(13) on the LUF arm.
    """
    c = circ
    mspec_temp = c.mspec                     # restored at label 999
    try:
        _luffy(c, ipflag)
    finally:
        c.mspec = mspec_temp


def _luffy(c, ipflag: int) -> None:
    ipfg = ipflag
    if c.mspec == 121 and c.gcdkm > _SMOOTH_MAX:
        ipfg = 200                           # long-path SETUP only; see docstring
        c.mspec = 0                          # beyond the smoothing distance

    pluf = F32(0.0)
    if ipfg > 200:
        # .....FREQUENCY COMPLEMENT, and the required reliability
        frqcom(c.muf, c.frq.frea, 0, c.time.it)
        pluf = _PCT * F32(c.lufp)

    # .....SELECT CONTROLLING SAMPLE AREA
    c.frq.jmode = selmod(c.layers)
    # .....GET NOISE VALUE AT 1 MHZ FOR RCVR SITE
    anois1(c.anois, c.iono, c.time.gmtr, c.rlong, c.rlongd, c.rlat)

    k = _select_area(c, ipfg)                # labels 83, 84, 85
    # .....CAN RUN FOR SHORT PATHS ALSO, SO SET ALL SAMPLE AREAS
    setlng(c.layers, c.rays)
    # .....SET ABSORPTION LOSS PARAMETERS, ADJUST SIGNAL DISTRIBUTION TABLES
    sigdis(c.layers, c.muf, c.iono, c.sigd, c.ton, c.gcdkm, c.frq.jmode)

    ihmin = _min_hops(c.muf.nhopmf)          # IHMX is computed here and dead

    # .....START FREQUENCY LOOP
    c.son.nhp[I_LUF] = -1
    c.son.mdl[I_LUF] = "S"                   # XMDL, set with K at label 83
    for ifx in range(1, NFREQ + 1):
        c.son.mdl[ifx] = " "

    for ifx in range(1, NFREQ + 1):          # DO 265 IFX=1,12
        if _frequency(c, ifx, ipflag, k, ihmin, pluf):
            _luf_found(c, ifx, pluf)         # label 165
            return
    # .....END OF FREQUENCY LOOP
    _after_loop(c, ipflag)


def _after_loop(c, ipflag: int) -> None:
    """What happens once every frequency has been done."""
    if c.method == 25:
        return
    if ipflag >= 300:
        _no_luf(c)                           # label 300
        return
    # .....SMOOTHING ONLY FOR DISTANCES 7000 - 10000 KM.  MSPEC was zeroed on
    # entry past 10000 km, so that case leaves here too.
    if c.mspec != 121 or not _SMOOTH_MIN <= c.gcdkm <= _SMOOTH_MAX:
        return
    raise LongPathNotImplemented(
        f"METHOD 30 at {float(c.gcdkm):.0f} km recycles into the long-path "
        "model and blends the two results; GMLOSS, SELTXR and LNGPAT are not "
        "ported")


def _select_area(c, ipfg: int) -> int:
    """Labels 83-85: choose the control area and build its ionosphere."""
    if ipfg not in (100, 300):
        raise LongPathNotImplemented(
            f"IPFG {ipfg} selects the long-path model (GMLOSS + SELTXR + "
            "LNGPAT); only the short-path arms 100 and 300 are ported")
    # .....SET FLAG TO INDICATE SHORT PATH MODEL
    c.jlong = -1
    k = c.frq.jmode

    while True:
        # .....ELECTRON DENSITY PROFILE -> IONOGRAM -> REFLECTRIX -> DEV LOSS
        lecden(c.layers, k)
        genion(c.layers, k)
        fobby(c.layers, c.rays, k)
        alosfv(c.layers, c.muf, k)
        if ipfg == 100 or k > 1:
            return k
        # Label 85.  Short-path LUF with JMODE == 1 falls in here and rebuilds
        # the ionosphere at the receiver index, leaving THAT one in place.
        k = c.layers.kfx
        c.frq.itxrcp[2] = c.layers.kfx
        if c.mspec == 121 and c.frq.jmode == k:
            return k
        if c.layers.kfx == 1:
            c.frq.itxrcp[2] = 2
            return k


def _min_hops(nhopmf) -> int:
    """IHMIN over the three layer MUFs, skipping F1 when it carries no modes."""
    ihmin = int(nhopmf[1])
    if 0 < nhopmf[2] < ihmin:
        ihmin = int(nhopmf[2])
    if nhopmf[3] < ihmin:
        ihmin = int(nhopmf[3])
    return ihmin


def _frequency(c, ifx: int, ipflag: int, k: int, ihmin: int, pluf) -> bool:
    """One trip through DO 265.  Returns True when the LUF has been found."""
    ipfg = ipflag                            # line 179; see the module docstring
    if ipfg < 300:
        # .....PRESET IN COMMON/SON/
        c.son.nhp[ifx] = -1
        c.son.sprob[ifx] = F32(0.0)
        c.son.probmp[ifx] = F32(0.0)
        if c.frq.frel[ifx] <= 0.0:
            return False                     # no frequency in this slot
        freq = c.frq.frel[ifx]
    else:
        freq = c.frq.frea[ifx]
        c.son.reliab[ifx] = F32(0.0)

    freq = allmodes(c.zon, c.allm, 0, freq)  # clear the accumulator
    c.frq.freq = freq
    # .....NOISE DISTRIBUTION
    _genois(c, freq)

    if ipfg not in (100, 300):
        raise LongPathNotImplemented(
            "the long-path frequency body needs FINDF at both ends plus "
            "GMLOSS, SELTXR and LNGPAT")

    # .....RAY SET TABLE
    findf(c.layers, c.reflx, c.rays, k, freq, c.amind)
    if c.reflx.dmaxkm[k] <= 0.0:
        if ipfg == 300:
            return False
        # .....ONLY ONE OVER THE MUF MODE
        ihsrt = ihstp = ihmin
    else:
        # .....UP TO THREE HOPS
        ihsrt = int(c.gcdkm / c.reflx.dmaxkm[k] + F32(1.0))
        ihmax = int(c.gcdkm / c.reflx.dskpkm[k])   # DSKPKM starts at 10000 km
        if ihsrt < ihmin:
            ihsrt = ihmin
        if ihmax < ihsrt:
            ihmax = ihsrt
        ihstp = min(ihmax, ihsrt + _MAX_EXTRA_HOPS)
        if ihsrt > ihmin:
            ihsrt = max(ihmin, ihstp - _MAX_EXTRA_HOPS)

    # .....HOP LOOP (IXHP counts the trips here and is never read)
    for ihop in range(ihsrt, ihstp + 1):
        hop = F32(ihop)
        c.modes.ghop = c.gcd / hop           # .....HOP DISTANCE
        # .....FIND UP TO SIX MODES
        fdist(c.reflx, c.modes, k, freq, c.amind)
        inmuf(c.layers, c.modes, c.muf, c.sigd, c.anois, c.rays, c.cant,
              c.zon, freq, c.frq.jmode, c.gcd, ihop)
        # .....SAVE ALL MODES AVAILABLE
        allmodes(c.zon, c.allm, ipfg, hop)

    # .....UP TO 2 ES MODES, THEN THE (DEAD) ES-F MODE
    esmod(c.layers, c.muf, c.sigd, c.anois, c.rays, c.cant, c.zon,
          freq, c.frq.jmode, c.gcd, c.gcdkm, c.amind)
    esreg(c.zon)
    allmodes(c.zon, c.allm, 2, _ES_FVAL)

    # .....COMBINED RELIABILITY
    _genois(c, freq)
    relbil(c.allm, c.anois, c.sigd, c.son, c.cant, ifx, freq, c.rsn, c.lufp)

    if ipfg >= 300:
        # .....TEST RELIABILITY
        return bool(c.son.reliab[ifx] >= pluf)

    if c.want_service_prob:                  # IF( LINBOT(14) ) 285,285,280
        serprb(c.allm, c.anois, c.ton, c.son, ifx, c.rsn, c.lufp)
    # IPFG is 100 here: 200 would set the receiver end from NMODE(2)/B(2).
    # .....CALCULATE MULTIPATH
    mpath(c.allm, c.son, ifx, c.dmp, c.pmp, c.gcdkm, c.method, c.mspec)
    # OUTALL(IFX) is presentation only under METHOD 25 -- out of scope.
    return False


def _luf_found(c, ifx: int, pluf) -> None:
    """Label 165.  The first frequency to make the grade IS the LUF; below it
    the answer lies between this slot and the one under it, so interpolate.

    The divisor cannot vanish: every slot 1..12 is tested on this arm (the
    "no frequency here" skip belongs to the system-performance arm), so
    RELIAB(IFX-1) is strictly below PLUF and RELIAB(IFX) at or above it.
    """
    frea = c.frq.frea
    if ifx == 1:                             # 1st frequency is good
        c.muf.xluf[c.time.it] = c.frq.freq
    else:
        # .....NO ITERATION, SELECT THE FREQUENCY COMPLEMENT FIRST
        flow = frea[ifx - 1]
        fhigh = frea[ifx]
        rlow = c.son.reliab[ifx - 1]
        rhigh = c.son.reliab[ifx]
        # .....USE LINEAR INTERPOLATION
        c.muf.xluf[c.time.it] = (flow + (fhigh - flow) * (pluf - rlow)
                                 / (rhigh - rlow))
    frea[I_LUF] = c.muf.xluf[c.time.it]


def _no_luf(c) -> None:
    """Label 300.  Nothing reached the required reliability, so report the
    best frequency negated -- except that the comparison never updates REL,
    so "best" means "the last one that beat slot 1".  See the module docstring.
    """
    ig = 1
    rel = c.son.reliab[1]                    # ... and never reassigned
    for i in range(2, NFREQ + 1):            # DO 160 IF = 2,12
        if c.son.reliab[i] > rel:
            ig = i
    c.muf.xluf[c.time.it] = -c.frq.frea[ig]
    c.frq.frea[I_LUF] = c.frq.frea[ig]


def _genois(c, freq) -> None:
    genois(c.iono, c.layers, c.anois, c.ton, c.zon, c.ndiag, c.cant,
           freq, c.noise, c.rlat, c.nyear)
