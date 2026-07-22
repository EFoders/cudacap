"""Stage A transliteration of F2VAR
(reference/voacapl/src/voacapw/f2var.for).

Evaluates the F2 layer parameters for every sample area from VERSY(4,5):
M(3000)F2, the 0.834-critical virtual height, the F2 critical frequency,
then the E-layer retardation and the (possibly twilight-corrected) F1->F2
merge that can move HI(2)/YI(2).  Consumes the E/F1 slots EF1VAR set and
the RAT/ZEN values TIMVAR set.  REAL*4 throughout; the GOTO web
(110/210/215/220/225/115/120) is preserved as nested branches.
"""

from voacap_gpu.stage_a.f32math import F32, log32
from voacap_gpu.stage_a.versy import versy

_XF1 = F32(1.1)
_DELZ = F32(2.0)


def _ret_term(fac):
    """FAC * ALOG((FAC+1)/(FAC-1)) with FAC already floored — the shared
    retardation kernel; caller finishes the (.-2.)*semithickness scaling."""
    return fac * log32((fac + F32(1.0)) / (fac - F32(1.0)))


def f2var(state, st, ab):
    """state: LayerState; st: IonoState; ab: virtim output."""
    for ii in range(1, state.km + 1):  # DO 125
        gamma = versy(st, ab, state.clat[ii], state.clong[ii],
                      state.gmdip[ii], 4, 5)
        state.f2m3[ii] = gamma[5]                     # M(3000) factor
        state.hpf2[ii] = F32(1490.0) / gamma[5] - F32(176.0)
        state.fi[3, ii] = (gamma[4] + F32(0.5) * state.gyz[ii]) * state.psc[3]
        zmax = state.zenmax[ii]
        z = state.zenang[ii]
        # F1 must be less than F2
        state.fi[2, ii] = min(state.fi[2, ii], F32(state.fi[3, ii] - F32(0.2)))
        fc = F32(0.834) * state.fi[3, ii]
        ec = state.fi[1, ii]
        fcec = fc / ec
        fcec = max(fcec, F32(1.1))
        # E layer retardation
        ret = _ret_term(fcec)
        ret = (ret - F32(2.0)) * state.yi[1, ii]
        fc1 = state.fi[2, ii]
        if fc1 > 0.0:  # 110 (else jump straight to 120)
            ffec = fc / fc1
            if zmax - _DELZ - z < 0.0:  # 215: near day-night twilight
                ffec = max(ffec, _XF1)
                zn = zmax - _DELZ
                hn = F32(165.0) + F32(0.6428) * zn
                yn = hn * (state.yi[2, ii] / state.hi[2, ii])
                rfn = _ret_term(ffec)
                rfn = F32(0.5) * yn * (rfn - F32(2.0))
                sz = (z - zn) / _DELZ
                rft = rfn * (F32(1.0) - sz)
                # F2 without F1
                hm = state.hpf2[ii] - ret
                ym = hm / state.rat[ii]
                dhn = (hm - ym) - (hn - yn)
                if dhn > 0.0:  # 220: bottom of F1 goes to bottom of F2
                    dh = dhn * (F32(1.0) - sz)
                    state.hi[2, ii] = (hm - ym) - dh + state.yi[2, ii]
                    if fc1 - fc > 0.0:  # 225: force F1 YM to F2 YM
                        y1max = yn + (ym - yn) * (
                            fc1 / state.fi[3, ii] - F32(0.834)) / F32(0.166)
                        state.yi[2, ii] = yn + (y1max - yn) * sz
                        state.hi[2, ii] = (hm - ym) - dh + state.yi[2, ii]
            else:  # 210: no problem (daytime)
                ffec = max(ffec, _XF1)
                rft = _ret_term(ffec)
                rft = F32(0.5) * state.yi[2, ii] * (rft - F32(2.0))
            ret = ret + rft  # 115
        # 120
        state.hi[3, ii] = state.hpf2[ii] - ret
        state.yi[3, ii] = state.hi[3, ii] / state.rat[ii]
