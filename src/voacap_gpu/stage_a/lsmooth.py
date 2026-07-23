"""The long-path/short-path blend, from a VOA memo dated 15 January 1991.

Between 7000 and 10000 km neither propagation model is right.  Below that
range the short-path model is: the signal takes the great circle and hops.
Above it the long-path model is: the two ends are far enough apart that they
have to be treated separately.  In between, both are defensible and they
disagree, so VOACAP runs BOTH and interpolates between them by distance.

LUFFY does that by running its whole frequency loop twice -- once with
IPFLAG 100 and once with 200 -- saving each pass's /SON/ column into a
(13,2) temporary, and then combining them here.  ``save_pass`` is the saving
half, called from inside the frequency loop; ``blend`` is the combining half,
called after the second pass.

**The blend is on POWER, not on decibels.**  DELPOW is the long path's signal
lower decile minus the short path's, in dB.  If the long path is weaker
(DELPOW < 0) the short path is simply taken -- no interpolation at all, on the
principle that the higher signal level wins.  Otherwise the excess is
converted back to a power ratio, interpolated linearly on DISINT (0 at 7000
km, 1 at 10000), and converted back:

    SMOOTH = SSPLD + 10*log10( DISINT*(10^(DELPOW/10) - 1) + 1 )

At DISINT 0 that is SSPLD exactly and at DISINT 1 it is SLPLD exactly, so the
two ends of the range join the pure models continuously.  The guard at
DELPOW > 375 dB clamps 10^37.5 to keep the exponential from overflowing
float32, which it would at about 38.5.

Three things about the original worth stating.

**Exactly 10000 km takes the long path with no smoothing at all.**  The outer
guard admits ``GCDKM <= 10000`` but the smoothing test is ``GCDKM < 10000``,
so a circuit at exactly the boundary selects the long path and stops.  The
MDL tag it reports is 'L', not 'M'.

**The 'M' tag is written twice and can be overwritten.**  MDL(13) is set to
'M' for the whole hour before the loop, then each frequency's own MDL(IF) is
set to 'S' or 'L' by the selection and only upgraded to 'M' if that frequency
actually smoothed.  So an hour tagged 'M' can contain frequencies tagged 'S'.

**Only the smoothed arm re-runs the physics.**  When DELPOW < 0, or beyond
10000 km, the /SON/ column is copied wholesale from whichever pass won and
nothing is recomputed.  When it smooths, the routine rebuilds /allMODE/ slot 1
from the blended power and calls GENOIS, RELBIL and SERPRB again -- so the
reliability of a smoothed frequency comes from the blend, not from either
pass.
"""

from dataclasses import dataclass, field

import numpy as np

from .f32math import F32, log10_32, pow32
from .genois import genois
from .pwrdb import pwrdb
from .relbil import relbil
from .serprb import serprb
from .sonstate import LAYTYP

NFREQ = 12
_SMOOTH_MIN = F32(7000.0)
_SMOOTH_MAX = F32(10000.0)
_SPAN = F32(3000.0)
_MAX_DELPOW = F32(375.0)        # ... above which 10**(DELPOW/10) overflows f32
_CLAMPED = F32(37.5)
_FIELD_DB = F32(107.2)          # the field-strength constant, RELBIL's too

# The /SON/ columns saved per pass and copied back wholesale by the winner.
_SON_SAVED = ("angle", "cprob", "dblos", "dblosl", "dblosu", "delay", "dbw",
              "vhigh", "dbu", "reliab", "sndb", "snpr", "snrlw", "snrup",
              "sprob", "probmp")


def _pair():
    return np.zeros((NFREQ + 2, 3), dtype=np.float32)


@dataclass
class SmoothStore:
    """LUFFY's "TEMPORARY STORAGE" -- one column per frequency per pass.

    Index [ifx, idx] with idx 1 for the short-path pass and 2 for the long,
    which is what ``(IPFLAG+1)/100`` evaluates to in the original.
    """

    son: dict = field(default_factory=lambda: {
        n: _pair() for n in _SON_SAVED})
    nhp: np.ndarray = field(
        default_factory=lambda: np.zeros((NFREQ + 2, 3), dtype=np.int32))
    mode: list = field(
        default_factory=lambda: [["", ""] for _ in range(NFREQ + 2)])
    tgain: np.ndarray = field(default_factory=_pair)
    rgain: np.ndarray = field(default_factory=_pair)
    # The long pass alone also keeps these, for the arm that selects it.
    zdu: np.ndarray = field(default_factory=lambda: np.zeros(NFREQ + 2,
                                                             dtype=np.float32))
    zdl: np.ndarray = field(default_factory=lambda: np.zeros(NFREQ + 2,
                                                             dtype=np.float32))
    inmode: np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ + 2, dtype=np.int32))
    zangler: np.ndarray = field(default_factory=lambda: np.zeros(
        NFREQ + 2, dtype=np.float32))
    zmoder: list = field(default_factory=lambda: [""] * (NFREQ + 2))


def save_pass(c, store, ifx: int, ipflag: int) -> None:
    """Stash this frequency's answer for the pass that just ran (lines 330-369).

    Called from inside the frequency loop, once per frequency per pass.  Only
    runs under MSPEC 121 -- the original guards the whole block on it, so a
    non-smoothing run pays nothing.
    """
    idx = (ipflag + 1) // 100           # 100 -> 1, 200 -> 2
    for name in _SON_SAVED:
        store.son[name][ifx, idx] = getattr(c.son, name)[ifx]
    store.nhp[ifx, idx] = c.son.nhp[ifx]
    store.mode[ifx][idx - 1] = c.son.mode[ifx]
    # NB slot 1 of /allMODE/, not slot IFX: these are the CHOSEN mode's gains.
    store.tgain[ifx, idx] = c.allm.tgain[1]
    store.rgain[ifx, idx] = c.allm.rgain[1]

    if ipflag == 200:
        # .....SAVE LONG PATH PARAMETERS
        store.zdu[ifx] = c.anois.du
        store.zdl[ifx] = c.anois.dl
        store.inmode[ifx] = c.allm.nmode[1]
        store.zangler[ifx] = c.son.angler[ifx]
        store.zmoder[ifx] = c.son.moder[ifx]


def blend(c, store) -> None:
    """Combine the two passes into one answer per frequency (lines 410-522)."""
    c.son.mdl[13] = "M"
    for ifx in range(1, NFREQ + 1):         # DO 290 IF=1,12
        c.son.mdl[ifx] = " "
        freq = c.frq.frel[ifx]
        if freq <= 0.0:
            continue
        c.frq.freq = freq
        # The signal's lower decile, per pass.  DBLOSL's sign is not trusted:
        # the ABS was added 9/14/94 "per VOA".
        slpld = store.son["dbw"][ifx, 2] - abs(store.son["dblosl"][ifx, 2])
        sspld = store.son["dbw"][ifx, 1] - abs(store.son["dblosl"][ifx, 1])
        delpow = slpld - sspld

        if delpow < 0.0:
            # .....USE SHORT PATH -- the higher signal level wins outright
            idx = 1
            c.son.mdl[ifx] = "S"
            c.son.moder[ifx] = store.mode[ifx][0]
            c.son.angler[ifx] = store.son["angle"][ifx, 1]
            # TGAIN and RGAIN are /allMODE/ arrays indexed by MODE everywhere
            # else; here the original indexes them by FREQUENCY.  Both run to
            # 20 so it is legal, and the long arm below goes back to slot 1.
            c.allm.tgain[ifx] = store.tgain[ifx, 1]
            c.allm.rgain[ifx] = store.rgain[ifx, 1]
            # ... and these two are what actually gets printed
            c.son.gaint[ifx] = c.allm.tgain[ifx]
            c.son.gainr[ifx] = c.allm.rgain[ifx]
            c.son.snxx[ifx] = c.rsn - store.son["snpr"][ifx, 1]
        else:
            # .....USE LONG PATH, OR SMOOTH IF UNDER 10000 KM
            idx = 2
            c.son.mdl[ifx] = "L"
            c.son.moder[ifx] = store.zmoder[ifx]
            c.son.angler[ifx] = store.zangler[ifx]
            c.anois.du = store.zdu[ifx]
            c.anois.dl = store.zdl[ifx]
            c.allm.b[1] = store.son["angle"][ifx, 2]
            c.allm.tllow[1] = store.son["dblosl"][ifx, 2]
            c.allm.tlhgh[1] = store.son["dblosu"][ifx, 2]
            c.allm.timed[1] = store.son["delay"][ifx, 2]
            c.allm.hn[1] = F32(store.nhp[ifx, 2])
            c.allm.hp[1] = store.son["vhigh"][ifx, 2]
            c.allm.prob[1] = store.son["cprob"][ifx, 2]
            c.allm.nmode[1] = store.inmode[ifx]
            c.allm.tgain[1] = store.tgain[ifx, 2]
            c.allm.rgain[1] = store.rgain[ifx, 2]
            c.allm.eff[1] = c.son.rneff[ifx]

        # .....COPY THE WINNING PASS'S /SON/ COLUMN
        c.son.mode[ifx] = store.mode[ifx][idx - 1]
        for name in _SON_SAVED:
            getattr(c.son, name)[ifx] = store.son[name][ifx, idx]
        c.son.nhp[ifx] = int(store.nhp[ifx, idx])
        c.anois.rcnse = c.son.xnynois[ifx]

        if delpow >= 0.0 and c.gcdkm < _SMOOTH_MAX:
            _smooth(c, ifx, freq, sspld, delpow)


def _smooth(c, ifx: int, freq, sspld, delpow) -> None:
    """Interpolate the two models on power and redo the physics from there."""
    c.son.mdl[ifx] = "M"
    disint = (c.gcdkm - _SMOOTH_MIN) / _SPAN
    if delpow > _MAX_DELPOW:
        delx = pow32(F32(10.0), _CLAMPED)   # ... avoid the overflow problem
    else:
        delx = pow32(F32(10.0), delpow / F32(10.0))
    smooth = sspld + F32(10.0) * log10_32(disint * (delx - F32(1.0))
                                          + F32(1.0))
    c.allm.sigpow[1] = smooth + c.allm.tllow[1]
    pdb = pwrdb(c.cant, freq)
    c.allm.tloss[1] = pdb - c.allm.sigpow[1]
    c.allm.sn[1] = c.allm.sigpow[1] - c.anois.rcnse - c.allm.eff[1]
    c.allm.fldst[1] = (_FIELD_DB + pdb + F32(20.0) * log10_32(freq)
                       - c.allm.tloss[1] - c.allm.rgain[1])
    c.allm.nmmod = 1
    genois(c.iono, c.layers, c.anois, c.ton, c.zon, c.ndiag, c.cant,
           freq, c.noise, c.rlat, c.nyear)
    relbil(c.allm, c.anois, c.sigd, c.son, c.cant, ifx, freq, c.rsn, c.lufp)
    if c.want_service_prob:
        serprb(c.allm, c.anois, c.ton, c.son, ifx, c.rsn, c.lufp)


def wants_smoothing(c, ipflag: int) -> bool:
    """Whether this circuit runs both models and blends them.

    The band is INCLUSIVE at both ends here; the smoothing itself excludes
    10000 km, so a circuit exactly there runs both passes and then takes the
    long path whole.
    """
    return (c.mspec == 121
            and _SMOOTH_MIN <= c.gcdkm <= _SMOOTH_MAX)
