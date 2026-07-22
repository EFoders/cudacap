"""Stage A transliteration of LECDEN
(reference/voacapl/src/voacapw/lecden.for).

Builds the electron-density profile HTR(50,1)/FNSQ(50,1) (true height km,
plasma frequency squared MHz^2) from the E/F1/F2 layer parameters for
sample area K, taking the pointwise maximum of the D-region exponential
tail, the parabolic E, a linear valley, the parabolic-or-linear F1, and
the parabolic F2.  May mutate HI(2)/YI(2)/FSECV during the F1 fit.  REAL*4;
only SQRT (correctly rounded) and EXP (mild [[f32-libm-noise]]) appear.

The arithmetic-IF control flow (labels 185/190/194/195/205/215 for the F1
fit, 225..375 for the per-height assembly) is preserved branch-for-branch.
"""

from voacap_gpu.stage_a.f32math import F32, exp32, sqrt32


def lecden(state, k):
    fi, yi, hi = state.fi, state.yi, state.hi
    htr, fnsq, fsecv = state.htr, state.fnsq, state.fsecv

    # external electron-density profile in use -> nothing to build
    for i in range(1, 4):  # DO 90
        if state.ielect[i] > 0:  # 85
            return

    xlow = F32(0.8516)
    hz = hi[1, k] - yi[1, k]
    xup = F32(0.98) * fi[1, k] / fi[3, k]
    hlow = hz + yi[1, k] * (F32(1.0) + sqrt32(F32(1.0) - xlow * xlow))
    hte = hi[1, k] + yi[1, k]
    hb2 = hi[3, k] - yi[3, k]
    fce = fi[1, k] * fi[1, k]
    fc2 = fi[3, k] * fi[3, k]
    hup = hb2 + yi[3, k] * (F32(1.0) - sqrt32(F32(1.0) - xup * xup))
    fup = xup * xup * fc2
    flow = xlow * xlow * fce
    asp = F32(0.0)
    if hup - hlow > 0.0:  # 165
        asp = (fup - flow) / (hup - hlow)

    # ---- F1-layer fit (may move HI(2)/YI(2)) ----
    lin = 0
    s1 = F32(0.0)
    fc1 = F32(0.0)
    hb1 = hi[2, k] - yi[2, k]        # line 42 (original hi/yi)
    ht1 = hi[2, k] + yi[2, k]        # line 43 (original hi/yi)
    if fi[2, k] > 0.0:  # 185
        fc1 = fi[2, k] * fi[2, k]
        htw = hb2 + yi[3, k] * (F32(1.0) - sqrt32(F32(1.0) - fc1 / fc2))
        if htw > hi[2, k] + F32(0.001):  # -> 205
            lin = 0
            yi[2, k] = min(yi[2, k], F32(hi[2, k] - hi[1, k] + F32(1.0)))
            fsecv[k] = F32(-1.0)
        else:  # 190
            ys = htw - hb1
            ys = max(F32(1.0), ys)
            s1 = fc1 / ys
            hi[2, k] = htw
            yi[2, k] = ys
            if hb2 - hb1 < 0.0:  # 194 -> 205
                yi[2, k] = hi[2, k] - hb2
                lin = 0
                yi[2, k] = min(yi[2, k], F32(hi[2, k] - hi[1, k] + F32(1.0)))
                fsecv[k] = F32(-1.0)
            else:  # 195
                lin = 1
                fsecv[k] = sqrt32(flow)
                denom = F32(1.0) - (fi[1, k] / fi[2, k]) ** 2
                denom = max(F32(0.17), denom)
                yb = (hi[2, k] - hi[1, k]) / denom
                if ys - yb >= 0.0:  # 401 (else 402 falls through)
                    ys = yb
                    yi[2, k] = ys
                    s1 = fc1 / ys
                    hb1 = hi[1, k]
                ht1 = htw  # 415

    # ---- D-region tail + the HTR height grid ----
    hd = F32(70.0)
    xtail = F32(0.85)
    hex = hi[1, k] - xtail * yi[1, k]
    fnx = F32(1.0) - xtail * xtail
    alp = F32(2.0) * (hi[1, k] - hex) / (fnx * yi[1, k] * yi[1, k])
    fsq = fnx * exp32(-alp * (hex - hd))
    htr[1, 1] = hd
    htr[5, 1] = hex
    hdif = (htr[5, 1] - htr[1, 1]) * F32(0.25)
    hdif = max(F32(0.0), hdif)
    htr[4, 1] = htr[5, 1] - min(F32(1.0), hdif)
    htr[2, 1] = htr[1, 1] + hdif
    htr[3, 1] = (htr[2, 1] + htr[4, 1]) * F32(0.5)
    htr[11, 1] = hi[1, k]  # E below nose
    hdif = (htr[11, 1] - htr[5, 1]) / F32(6.0)
    for ih in range(6, 11):  # DO 100
        htr[ih, 1] = htr[ih - 1, 1] + hdif
    htr[17, 1] = hi[1, k] + yi[1, k]  # E above nose
    hdif = (htr[17, 1] - htr[11, 1]) / F32(6.0)
    for ih in range(12, 17):  # DO 105
        htr[ih, 1] = htr[ih - 1, 1] + hdif
    htr[11, 1] = F32(0.5) * (htr[10, 1] + htr[12, 1])
    htr[50, 1] = hi[3, k]
    two_layers = fi[2, k] > 0.0
    if two_layers:  # 125
        hb2 = hi[3, k] - yi[3, k]
        hb1b = hi[2, k] - yi[2, k] + F32(0.00001)
        if hb2 - hb1b <= 0.0:  # 110 (treat as no F1)
            two_layers = False
    if not two_layers:  # 110: F2 layer, no F1 layer
        htr[18, 1] = hi[3, k] - yi[3, k]
        hdif = (htr[50, 1] - htr[18, 1]) / F32(32.0)
        for ih in range(19, 50):  # DO 115
            htr[ih, 1] = htr[ih - 1, 1] + hdif
    else:  # 126: F1 and F2 layers
        htr[18, 1] = hi[2, k] - yi[2, k]
        htr[18, 1] = max(htr[18, 1], F32(htr[17, 1] + F32(1.0)))
        htr[28, 1] = hi[2, k]
        hdif = (htr[28, 1] - htr[18, 1]) / F32(10.0)
        for ih in range(19, 28):  # DO 135
            htr[ih, 1] = htr[ih - 1, 1] + hdif
        hdif = (htr[50, 1] - htr[28, 1]) / F32(22.0)
        for ih in range(29, 50):  # DO 145
            htr[ih, 1] = htr[ih - 1, 1] + hdif

    # ---- per-height density (max of the layer contributions) ----
    hb1 = max(hi[1, k], F32(hi[2, k] - yi[2, k]))  # line 143
    hi1k, yi1k = hi[1, k], yi[1, k]
    hi2k, yi2k = hi[2, k], yi[2, k]
    hi3k, yi3k = hi[3, k], yi[3, k]
    has_f1 = fi[2, k] > 0.0
    for ih in range(1, 51):  # DO 375
        fnd = fne = fn1 = fn2 = fnval = F32(0.0)
        h = htr[ih, 1]
        if h - hlow > 0.0:  # 225
            if h - hup < 0.0:  # 235 linear valley
                fnval = fup + asp * (h - hup)
        # 245
        if h - hex < 0.0:  # 275 exponential D-E
            fnd = fce * fsq * exp32(alp * (h - hd))
        elif h - hte <= 0.0:  # 260 parabolic E
            z = (h - hi1k) / yi1k
            fne = fce * (F32(1.0) - z * z)
        # 285
        if has_f1:  # 290
            if hb1 - h <= 0.0 and h - ht1 <= 0.0:  # 295 -> 300
                if lin <= 0:  # 305 parabolic F1
                    z = (h - hi2k) / yi2k
                    fn1 = fc1 * (F32(1.0) - z * z)
                else:  # 335 linear F1
                    fn1 = s1 * (h - (hi2k - yi2k))
        # 345
        if hb2 - h <= 0.0:  # 355 parabolic F2
            z = (h - hi3k) / yi3k
            fn2 = fc2 * (F32(1.0) - z * z)
        fnsq[ih, 1] = max(fnd, fne, fnval, fn1, fn2)  # 365
