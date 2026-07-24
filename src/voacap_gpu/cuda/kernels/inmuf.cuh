// INMUF — insert over-the-MUF and zero-distance modes, then run REGMOD; the
// CUDA C twin of stage_a/inmuf.py.  FDIST only finds modes that appear in the
// reflectrix.  Above the MUF there are none, and at very short distances the
// ray table does not reach, yet the prediction still needs a mode to attach a
// loss to.  INMUF supplies them so the frequency sweep degrades through the
// MUF instead of falling off a cliff at it.
//
// Four stages:
//   1. COMPACT the /MODES/ list, dropping anything below 70 km virtual height
//      and packing what survives into slots 1..INUM.
//   2. INSERT THE CIRCUIT MUF MODE for any layer that has none, when the
//      frequency is at or above that layer's MUF and the hop count matches.
//   3. RECOMPUTE A PER-HOP MUF for layers whose MUF was established at a
//      different hop count, via the ionogram (XLIN) plus the same Martyn
//      spherical correction CURMUF uses, and insert a mode where the frequency
//      clears it.  This stage OVERWRITES YMUF/SIGL/YFOT/SIGU/YHPF, which are
//      saved and restored around the REGMOD call.
//   4. Or, for a very short path, SYNTHESISE A ZERO-DISTANCE MODE straight off
//      the ionogram at 90 degrees take-off.
//
// Five things preserved deliberately because each looks like a bug:
//
//   * The compaction in stage 1 does NOT clear the slots above INUM, so a
//     vacated slot keeps its previous occupant.  With FDIST's contiguous input
//     nothing is left behind, but the trailing NUMMOD count scans all six
//     slots and would see a stale mode if it were.
//   * Stage 3 assigns GHOP = 2*PSI, mutating the hop distance in /MODES/ that
//     REGMOD then reads.  The insert changes the geometry for EVERY mode, not
//     just the inserted one.
//   * MODMUF is a Fortran LAYER NUMBER, so DELMUF(MODMUF)/NHOPMF(MODMUF) are
//     ordinary 1-based reads of the /MUFS/ layer tables, not a row offset.
//   * The stage-3 guard is `IF(FREQ-ESD+EPS) 240,160,160` — skip when the
//     value is NEGATIVE.  A ray that cannot reflect gives SPHE > 1 and hence
//     ESD = NaN, where `< 0` is FALSE, so the Fortran deliberately INSERTS a
//     NaN mode.  The test is written as the Fortran's, negated, never with the
//     operator flipped: `>= 0` would silently drop exactly those lanes.
//   * ISAV is assigned twice and never read — dead, and kept only for shape.
//
// The INFO-gated writes to unit 99 are diagnostics that compute nothing
// reaching /MODES/ or /ZON/, so they are not ported.
//
// Where Stage A RAISES, this returns a status.  Codes 1..4 are REGMOD's, and
// are propagated VERBATIM with /MUFS/ LEFT UNRESTORED, because Stage A's
// exception escapes before the `if ireset:` restore runs.  INMUF adds:
//   5 = /MODES/ ITMOD outside 1..3 while compacting — Stage A indexes its
//       4-element INM list and raises IndexError (a negative ITMOD would wrap
//       in Python instead; that is not reproduced, and is refused too),
//   6 = the zero-distance stage would write /MODES/ row 7 — Stage A raises
//       IndexError on the (7, 4) array.
// Both are asserted UNREACHABLE on the oracle grid, not merely untested.
//
// *nummod_out is written before REGMOD runs, exactly where Stage A computes
// NUMMOD; it is only meaningful when the returned status is 0, since Stage A
// loses the value to the exception otherwise.
//
// Rule 4b: RZ is the only compile-time denominator here (XHP), and goes
// through div32.  No REAL*8 anywhere.
//
// float32; COS/SIN/SQRT/ATAN carry the [[f32-libm-noise]] floor.

#ifndef VOACAP_INMUF_CUH
#define VOACAP_INMUF_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "raysview.cuh"
#include "regmod.cuh"
#include "state.cuh"
#include "xlin.cuh"

// Arguments match regmod() plus `ihop`, the hop count under test, and
// `nummod_out`.  `modes` and `muf` are MUTATED; `muf` is restored on the way
// out if stage 3 changed it and REGMOD did not refuse.
__device__ inline int inmuf(const LayerState* state, ModesState* modes,
                            MufState* muf, const SigdState* sigd,
                            const AnoisState* anois, const RaysView& rays,
                            const float* cant_array, const float* cant_aeff,
                            const int* cant_iats, const float* cant_xfqs,
                            const float* cant_xfqe, const float* cant_offazim,
                            const float* cant_pwrdba, int numants,
                            ZonState* zon, float freq, int jmode, float gcd,
                            int ihop, int* nummod_out) {
    const float eps = 0.4001f;
    const float hmin = 70.0f;
    int k = jmode;
    int ireset = 0;
    int inm[4] = {0, -1, -1, -1};       // INM(1..3): is layer ILY present?

    // --- 1. compact the mode list ------------------------------------------
    int inum = 0;
    for (int im = 1; im <= 6; ++im) {
        if (modes->hpmod[im * 4 + k] - hmin < 0.0f)     // IF(...) 125,100,100
            continue;
        inum += 1;
        int it = modes->itmod[im * 4 + k];
        if (it < 0 || it > 3)
            return 5;                   // Stage A indexes INM out of range
        inm[it] = 1;
        modes->delmod[inum * 4 + k] = modes->delmod[im * 4 + k];
        modes->hpmod[inum * 4 + k] = modes->hpmod[im * 4 + k];
        modes->htmod[inum * 4 + k] = modes->htmod[im * 4 + k];
        modes->fvmod[inum * 4 + k] = modes->fvmod[im * 4 + k];
        modes->afmod[inum * 4 + k] = modes->afmod[im * 4 + k];
        modes->itmod[inum * 4 + k] = modes->itmod[im * 4 + k];
        // NB: slots above INUM are deliberately left as they were
    }
    int isav = inum;

    float osigl[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float osigu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oyfot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oyhpf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oymuf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // very short distance (take-off angle >= 89.9) -> zero-distance mode
    int zero_distance = (muf->delmuf[muf->modmuf] >= 89.9f);

    if (!zero_distance) {
        // --- 2. insert the circuit MUF mode for any missing layer ----------
        for (int ily = 1; ily <= 3; ++ily) {
            if (ily == 2 && state->fi[2 * 6 + k] <= 0.0f)   // no F1 layer
                continue;
            if (inum >= 5)                                  // all layers in
                break;
            if (inm[ily] < 0) {
                if ((freq + eps) >= muf->ymuf[ily]
                        && ihop == muf->nhopmf[ily]) {
                    inum += 1;
                    modes->delmod[inum * 4 + k] = muf->delmuf[ily];
                    modes->hpmod[inum * 4 + k] = muf->hpmuf[ily];
                    modes->htmod[inum * 4 + k] = muf->htmuf[ily];
                    modes->fvmod[inum * 4 + k] = muf->fvmuf[ily];
                    modes->afmod[inum * 4 + k] = muf->afmuf[ily];
                    modes->itmod[inum * 4 + k] = ily;
                    inm[ily] = 1;
                }
            }
        }
        if (inum > isav)
            isav = inum;                // ISAV is never read again — dead

        if (ihop == muf->nhopmf[muf->modmuf]) {
            // a mode already covers this hop order -> straight to the budget;
            // none at all -> fall back to the zero-distance construction
            zero_distance = (inum <= 0);
        } else {
            // --- 3. per-hop MUF, above the circuit MUF for this hop order --
            ireset = 1;
            for (int il = 1; il <= 3; ++il) {
                osigl[il] = muf->sigl[il];
                osigu[il] = muf->sigu[il];
                oyfot[il] = muf->yfot[il];
                oyhpf[il] = muf->yhpf[il];
                oymuf[il] = muf->ymuf[il];
            }
            for (int jh = 1; jh <= 3; ++jh) {
                if (inum >= 5)                                  // all in
                    break;
                if (inm[jh] > 0)                        // layer already has one
                    continue;
                if (jh == 2 && state->fi[2 * 6 + k] <= 0.0f)    // no F1 layer
                    continue;
                if (ihop <= muf->nhopmf[jh])
                    continue;
                float fv = muf->fvmuf[jh];
                float ht = muf->htmuf[jh];
                float hpver = xlin(fv, state->fvert, state->hprim, 30, k, 6);
                float xhp = div32(hpver - ht, VC_RZ);
                float hpnow = hpver;
                float hop = (float)ihop;
                float psi = 0.5f * gcd / hop;
                float tdel = (cos32(psi) - VC_RZ / (VC_RZ + hpnow))
                             / sin32(psi);
                float cdel = 1.0f / sqrt32(1.0f + tdel * tdel);
                float sphe = VC_RZ * cdel / (VC_RZ + ht);
                float secp = 1.0f / sqrt32(1.0f - sphe * sphe);
                float esd = fv * secp;
                // correction to Martyn's theorem, as in CURMUF
                float fob1 = esd;
                float xmut = sphe * sphe;
                float xfsq = fob1 * fob1
                             / (state->fi[jh * 6 + k] * state->fi[jh * 6 + k]);
                float sph = xfsq * xmut * xhp
                            * (ht + 2.0f * (VC_RZ + ht) * xhp);
                hpnow = hpver + sph;
                tdel = (cos32(psi) - VC_RZ / (VC_RZ + hpnow)) / sin32(psi);
                cdel = 1.0f / sqrt32(1.0f + tdel * tdel);
                sphe = VC_RZ * cdel / (VC_RZ + ht);
                secp = 1.0f / sqrt32(1.0f - sphe * sphe);
                esd = fv * secp;
                if (freq - esd + eps < 0.0f)    // IF(...) 240,160,160; NaN
                    continue;                   // is NOT negative -> inserted
                // a new MUF has been found and can be used
                muf->ymuf[jh] = esd;
                muf->sigl[jh] = esd * osigl[jh] / oymuf[jh];
                muf->yfot[jh] = muf->ymuf[jh] - 1.28f * muf->sigl[jh];
                muf->sigu[jh] = esd * osigu[jh] / oymuf[jh];
                muf->yhpf[jh] = muf->ymuf[jh] + 1.28f * muf->sigu[jh];
                inum += 1;
                modes->delmod[inum * 4 + k] = atan32(tdel) * VC_R2D;
                modes->hpmod[inum * 4 + k] = hpnow;
                modes->htmod[inum * 4 + k] = muf->htmuf[jh];
                modes->fvmod[inum * 4 + k] = muf->fvmuf[jh];
                modes->afmod[inum * 4 + k] = muf->afmuf[jh];
                modes->itmod[inum * 4 + k] = jh;
                modes->ghop = 2.0f * psi;   // mutates the hop for REGMOD
            }
        }
    }

    // --- 4. zero-distance mode straight off the ionogram --------------------
    if (zero_distance) {
        inum += 1;
        float fv = freq - 0.001f;
        // find FV in the ionogram
        int ihx = 30;
        int hit = 0;                    // 0 = above the table, 260, or 265
        for (int ihh = 2; ihh <= 30; ++ihh) {
            ihx = ihh;
            if (fv - state->fvert[ihh * 6 + k] < 0.0f) {   // IF(...) 260,265,255
                hit = 260;
                break;
            }
            if (fv - state->fvert[ihh * 6 + k] == 0.0f) {
                hit = 265;
                break;
            }
        }
        if (hit && inum > 6)
            return 6;                   // Stage A: IndexError on /MODES/ row
        if (hit == 260) {
            // interpolate between table values
            float slope = state->fvert[ihx * 6 + k]
                          - state->fvert[(ihx - 1) * 6 + k];
            if (slope > 0.01f) {
                float slp = (fv - state->fvert[(ihx - 1) * 6 + k]) / slope;
                modes->delmod[inum * 4 + k] = 90.0f;
                modes->hpmod[inum * 4 + k] =
                    state->hprim[(ihx - 1) * 6 + k]
                    + (state->hprim[ihx * 6 + k]
                       - state->hprim[(ihx - 1) * 6 + k]) * slp;
                modes->htmod[inum * 4 + k] =
                    state->htrue[(ihx - 1) * 6 + k]
                    + (state->htrue[ihx * 6 + k]
                       - state->htrue[(ihx - 1) * 6 + k]) * slp;
                modes->fvmod[inum * 4 + k] = fv;
                modes->afmod[inum * 4 + k] =
                    state->afac[(ihx - 1) * 6 + k]
                    + (state->afac[ihx * 6 + k]
                       - state->afac[(ihx - 1) * 6 + k]) * slp;
            }
        } else if (hit == 265) {
            // exact hit in the FVERT table, so use its entries
            modes->delmod[inum * 4 + k] = 90.0f;
            modes->hpmod[inum * 4 + k] = state->hprim[ihx * 6 + k];
            modes->htmod[inum * 4 + k] = state->htrue[ihx * 6 + k];
            modes->fvmod[inum * 4 + k] = state->fvert[ihx * 6 + k];
            modes->afmod[inum * 4 + k] = state->afac[ihx * 6 + k];
        }
        if (hit) {                      // 270: assign the mode a layer
            for (int ily = 1; ily <= 3; ++ily) {
                if (ily == 2 && state->fi[2 * 6 + k] <= 0.0f)
                    continue;
                if (modes->htmod[inum * 4 + k] <= state->hi[ily * 6 + k]) {
                    modes->itmod[inum * 4 + k] = ily;
                    inm[ily] = 1;
                    break;
                }
            }
        }
    }

    // --- 275: count the modes, run the loss budget, put /MUFS/ back ---------
    int nummod = 0;
    for (int ilm = 1; ilm <= 6; ++ilm) {
        if (modes->hpmod[ilm * 4 + k] < 0.0f)
            continue;
        nummod += 1;
    }
    *nummod_out = nummod;
    int st = regmod(state, modes, muf, sigd, anois, rays, cant_array,
                    cant_aeff, cant_iats, cant_xfqs, cant_xfqe, cant_offazim,
                    cant_pwrdba, numants, zon, freq, jmode, gcd);
    if (st != 0)
        return st;                      // /MUFS/ stays as Stage A leaves it
    if (ireset) {
        for (int il = 1; il <= 3; ++il) {
            muf->sigl[il] = osigl[il];
            muf->sigu[il] = osigu[il];
            muf->yfot[il] = oyfot[il];
            muf->yhpf[il] = oyhpf[il];
            muf->ymuf[il] = oymuf[il];
        }
    }
    return 0;
}

#endif  // VOACAP_INMUF_CUH
