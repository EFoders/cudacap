"""Stage A transliteration of IONSET
(reference/voacapl/src/voacapw/ionset.for).

Checks consistency of the layer parameters and packs them into the KFX
"controlling" slots using the traditional rule (E layer from the closest
sample area, F layer from the further).  Zeroes an F1 layer that has merged
into E or F2, and floors the F2 semithickness.  Integer control flow +
REAL*4 AMIN1; no transcendentals.
"""

from voacap_gpu.scalar.f32math import F32


def ionset(state):
    fi, yi, hi = state.fi, state.yi, state.hi
    km = state.km
    if km <= 1:  # 105
        state.kfx = 1
    elif km <= 3:  # 115
        state.kfx = 2
        fi[3, 1], yi[3, 1], hi[3, 1] = fi[3, 2], yi[3, 2], hi[3, 2]
        fi[1, 2], yi[1, 2], hi[1, 2] = fi[1, 3], yi[1, 3], hi[1, 3]
        fi[2, 2], yi[2, 2], hi[2, 2] = fi[2, 3], yi[2, 3], hi[2, 3]
    else:  # 120
        state.kfx = 3
        fi[3, 1], yi[3, 1], hi[3, 1] = fi[3, 2], yi[3, 2], hi[3, 2]
        for i in range(1, 4):  # DO 122
            fi[i, 2], yi[i, 2], hi[i, 2] = fi[i, 3], yi[i, 3], hi[i, 3]
        fi[1, 3], yi[1, 3], hi[1, 3] = fi[1, 5], yi[1, 5], hi[1, 5]
        fi[2, 3], yi[2, 3], hi[2, 3] = fi[2, 5], yi[2, 5], hi[2, 5]
        fi[3, 3], yi[3, 3], hi[3, 3] = fi[3, 4], yi[3, 4], hi[3, 4]

    # 125: consistency pass over the KFX controlling slots
    if state.kfx < 1:  # 165
        return
    for k in range(1, state.kfx + 1):  # DO 155
        if fi[2, k] > 0.0:  # 130 (F1 present)
            if fi[2, k] - fi[1, k] - F32(0.2) <= 0.0:  # 135: F1 merged into E
                fi[2, k] = F32(0.0)
                yi[2, k] = F32(0.0)
                hi[2, k] = F32(0.0)
            elif fi[3, k] - fi[2, k] - F32(0.2) <= 0.0:  # 135: merged into F2
                fi[2, k] = F32(0.0)
                yi[2, k] = F32(0.0)
                hi[2, k] = F32(0.0)
            else:  # 145
                _hdif = hi[2, k] - hi[1, k] + F32(1.0)  # dead (reset at 150)
                hi[2, k] = min(hi[2, k], hi[3, k])
        # 150
        hdif = hi[3, k] - hi[1, k] - F32(2.0)
        yi[3, k] = min(yi[3, k], F32(hdif))
