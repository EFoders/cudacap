// IONSET — consistency pass over the layer parameters at each sample area;
// the CUDA C twin of scalar/ionset.py.
//
// Two stages, in this order:
//
//   1. PACKING (105/115/120).  KM sample areas collapse into KFX = 1, 2 or 3
//      "controlling" slots by the traditional rule — E layer taken from the
//      closest sample area, F layer from the further one.  The copies are
//      pure index moves and their ORDER IS LOAD-BEARING: FI(3,1) takes
//      FI(3,2) *before* the DO 122 loop overwrites FI(3,2) with FI(3,3), and
//      the DO 122 loop reads column 3 *before* column 3 is refilled from
//      columns 4/5.  Written out sequentially exactly as Stage A does.
//
//   2. CONSISTENCY (125..155).  For each controlling slot: an F1 layer that
//      has merged into E (FOF1 - FOE <= 0.2) or into F2 (FOF2 - FOF1 <= 0.2)
//      is zeroed outright (frequency, semithickness and height); otherwise
//      its height is capped at the F2 height.  Finally the F2 semithickness
//      is floored so the layer bottom stays above the E layer by 2 km.
//
// Integer control flow plus REAL*4 AMIN1 (Python min -> pymin); no
// transcendentals, no division, so this routine has no libm or strength-
// reduction exposure at all — every case must be bit-exact.
//
// The FI/YI/HI arrays are the LayerState (4,6) row-major members, so Fortran
// FI(i,k) is fi[i * 6 + k] with the 1-based padding (row 0, column 0) intact.

#ifndef VOACAP_IONSET_CUH
#define VOACAP_IONSET_CUH

#include "f32math.cuh"
#include "state.cuh"

// state: LayerState.  Reads KM, writes KFX and rewrites the FI/YI/HI
// controlling slots in place.  No return value (Stage A mutates the state).
__device__ inline void ionset(LayerState* state) {
    float* fi = state->fi;
    float* yi = state->yi;
    float* hi = state->hi;
    int km = state->km;
    if (km <= 1) {                              // 105
        state->kfx = 1;
    } else if (km <= 3) {                       // 115
        state->kfx = 2;
        fi[3 * 6 + 1] = fi[3 * 6 + 2];
        yi[3 * 6 + 1] = yi[3 * 6 + 2];
        hi[3 * 6 + 1] = hi[3 * 6 + 2];
        fi[1 * 6 + 2] = fi[1 * 6 + 3];
        yi[1 * 6 + 2] = yi[1 * 6 + 3];
        hi[1 * 6 + 2] = hi[1 * 6 + 3];
        fi[2 * 6 + 2] = fi[2 * 6 + 3];
        yi[2 * 6 + 2] = yi[2 * 6 + 3];
        hi[2 * 6 + 2] = hi[2 * 6 + 3];
    } else {                                    // 120
        state->kfx = 3;
        fi[3 * 6 + 1] = fi[3 * 6 + 2];
        yi[3 * 6 + 1] = yi[3 * 6 + 2];
        hi[3 * 6 + 1] = hi[3 * 6 + 2];
        for (int i = 1; i <= 3; ++i) {          // DO 122
            fi[i * 6 + 2] = fi[i * 6 + 3];
            yi[i * 6 + 2] = yi[i * 6 + 3];
            hi[i * 6 + 2] = hi[i * 6 + 3];
        }
        fi[1 * 6 + 3] = fi[1 * 6 + 5];
        yi[1 * 6 + 3] = yi[1 * 6 + 5];
        hi[1 * 6 + 3] = hi[1 * 6 + 5];
        fi[2 * 6 + 3] = fi[2 * 6 + 5];
        yi[2 * 6 + 3] = yi[2 * 6 + 5];
        hi[2 * 6 + 3] = hi[2 * 6 + 5];
        fi[3 * 6 + 3] = fi[3 * 6 + 4];
        yi[3 * 6 + 3] = yi[3 * 6 + 4];
        hi[3 * 6 + 3] = hi[3 * 6 + 4];
    }

    // 125: consistency pass over the KFX controlling slots
    if (state->kfx < 1)                         // 165 (KFX is 1..3 above)
        return;
    for (int k = 1; k <= state->kfx; ++k) {     // DO 155
        if (fi[2 * 6 + k] > 0.0f) {             // 130 (F1 present)
            if (fi[2 * 6 + k] - fi[1 * 6 + k] - 0.2f <= 0.0f) {
                fi[2 * 6 + k] = 0.0f;           // 135: F1 merged into E
                yi[2 * 6 + k] = 0.0f;
                hi[2 * 6 + k] = 0.0f;
            } else if (fi[3 * 6 + k] - fi[2 * 6 + k] - 0.2f <= 0.0f) {
                fi[2 * 6 + k] = 0.0f;           // 135: merged into F2
                yi[2 * 6 + k] = 0.0f;
                hi[2 * 6 + k] = 0.0f;
            } else {                            // 145
                // Stage A keeps this dead local (the Fortran recomputes HDIF
                // at 150 before any use); carried for line fidelity only.
                float hdif_dead = hi[2 * 6 + k] - hi[1 * 6 + k] + 1.0f;
                (void)hdif_dead;
                hi[2 * 6 + k] = pymin(hi[2 * 6 + k], hi[3 * 6 + k]);
            }
        }
        // 150
        float hdif = hi[3 * 6 + k] - hi[1 * 6 + k] - 2.0f;
        yi[3 * 6 + k] = pymin(yi[3 * 6 + k], hdif);
    }
}

#endif  // VOACAP_IONSET_CUH
