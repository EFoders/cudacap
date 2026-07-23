"""GMLOSS -- "the gain minus loss function", and mostly a wrapper.

Sixty lines, of which the last twelve compute a quantity NOTHING EVER READS.

TXRGML is assigned here and nowhere else in voacapw, and no routine reads it:
the gain-minus-loss the subroutine is named for is computed per ray, per end,
and discarded.  (SELTXR computes its own GNOW from the same three columns a
moment later, and stores it in GML -- which is itself read only by LNGPAT's
debug listing.)  So GMLOSS has exactly two surviving effects:

1. it presets all seven /ZON/ slots to the model's "no mode" markers, and
2. it calls SETTXR.

The presets matter even though the long path never fills /ZON/ with modes --
it works in /allMODE/ instead -- because GENOIS runs afterwards and reads the
block.  They are the usual mixed vocabulary: 1000 dB for a loss that was not
computed, 0.001 for a probability, -1000 for a level, -1 for a height or a
delay, and NMODE 5, the ' N' entry in LAYTYP.

The TXRGML loop is transliterated anyway, because "write-only" is a claim
about the whole program and the dump is what makes it checkable rather than
asserted.  Note it does NOT preset TXRGML: a ray that fails either test keeps
whatever the previous frequency left there.

**GMLOSS agrees with SETTXR about exactly 70 km and disagrees with SELTXR.**
``IF(HPFLX(IA,K)-70.) 110,105,105`` keeps a ray at exactly 70; SELTXR's
``HPFLX .LE. 70`` kills it.  Likewise ``IF(DELFX-AMIND) 110,106,106`` keeps a
ray exactly on the minimum angle.

float32; no libm calls of its own -- everything transcendental is inside
SETTXR.
"""

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.settxr import settxr

_NZON = 7
_MODE_HEIGHT = F32(70.0)


def _preset(zon) -> None:
    """DO 240: every /ZON/ slot back to its "no mode" marker."""
    for im in range(1, _NZON + 1):
        zon.obf[im] = F32(1000.0)
        zon.adv[im] = F32(1000.0)
        zon.fslos[im] = F32(1000.0)
        zon.tloss[im] = F32(1000.0)
        zon.abps[im] = F32(1000.0)
        zon.eff[im] = F32(0.0)
        zon.grlos[im] = F32(1000.0)
        zon.rgain[im] = F32(0.0)
        zon.tgain[im] = F32(0.0)
        zon.hn[im] = F32(-1.0)
        zon.prob[im] = F32(0.001)
        zon.crel[im] = F32(0.001)
        zon.rely[im] = F32(0.001)
        zon.spro[im] = F32(0.001)
        zon.fldst[im] = F32(-1000.0)
        zon.sigpow[im] = F32(-1000.0)
        zon.sn[im] = F32(-1000.0)
        zon.timed[im] = F32(-1.0)
        zon.hp[im] = F32(-1.0)
        zon.b[im] = F32(-1.0)
        zon.nmode[im] = 5           # LAYTYP(5) = ' N', i.e. no layer


def gmloss(state, muf, sigd, reflx, losx, rays, cant, allm, zon, lpath,
           itxrcp, freq, gcdkm, amind) -> None:
    """Preset /ZON/, run SETTXR, then fill TXRGML (which nothing reads).

    Arguments are SETTXR's plus `zon` (ZonState, preset here) and `lpath`
    (LpathState, TXRGML written here).
    """
    # .....PRESET ALL ARRAYS
    _preset(zon)
    # .....TRANSMITTER/RECEIVER ENDS
    settxr(state, muf, sigd, reflx, losx, rays, cant, allm, itxrcp,
           freq, gcdkm, amind)
    amind = F32(amind)
    for jj in (1, 2):
        k = int(itxrcp[jj])
        for ia in range(1, int(reflx.iaftxr[k]) + 1):
            # .....AT LEAST AN OVER THE MUF
            if reflx.hpflx[ia, k] - _MODE_HEIGHT < 0.0:
                continue                    # ... so exactly 70 LIVES
            if reflx.delfx[ia, k] - amind < 0.0:
                continue                    # ... and so does exactly AMIND
            # .....GAIN-LOSS, TRANSMITTER /RECEIVER ENDS
            lpath.txrgml[ia, jj] = losx.tgainx[ia, k] - (
                losx.andvx[ia, k] + losx.advx[ia, k])
