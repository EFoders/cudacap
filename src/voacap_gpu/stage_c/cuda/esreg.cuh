// ESREG — would compute a mixed Es-F mode.  It does not.  It presets; the
// CUDA C twin of stage_a/esreg.py.  The Fortran's 280 lines of mixed-mode
// physics sit behind an unconditional RETURN ("ERRORS IN THE CODE BELOW"),
// so the whole reachable behaviour is presetting /ZON/ slots 6 and 7 to the
// no-mode markers.  Note CREL is +1000 here where ESMOD's Es slots get
// -1000 — transliterated as found.

#ifndef VOACAP_ESREG_CUH
#define VOACAP_ESREG_CUH

#include "state.cuh"

__device__ inline void esreg(ZonState* zon) {
    for (int im = 6; im <= 7; ++im) {       // DO 100, IMS = 6, IME = 7
        zon->abps[im]   = 1000.0f;
        zon->crel[im]   = 1000.0f;
        zon->eff[im]    = 0.0f;
        zon->fldst[im]  = -1000.0f;
        zon->grlos[im]  = 1000.0f;
        zon->hn[im]     = -1.0f;
        zon->hp[im]     = -1.0f;
        zon->prob[im]   = 0.001f;
        zon->rely[im]   = 0.001f;
        zon->rgain[im]  = 0.0f;
        zon->tgain[im]  = 0.0f;
        zon->timed[im]  = -1.0f;
        zon->tloss[im]  = 1000.0f;
        zon->b[im]      = -1.0f;
        zon->fslos[im]  = 1000.0f;
        zon->adv[im]    = 1000.0f;
        zon->obf[im]    = 1000.0f;
        zon->nmode[im]  = 5;
        zon->tllow[im]  = 10.0f;
        zon->tlhgh[im]  = 10.0f;
        zon->sigpow[im] = -1000.0f;
        zon->sn[im]     = -1000.0f;
        zon->spro[im]   = 0.001f;
    }
}

#endif  // VOACAP_ESREG_CUH
