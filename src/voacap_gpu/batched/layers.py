"""Stage B (PyTorch, batched) Module-2 layer chain: the coefficient maps
turned into E/F1/F2/Es parameters at every grid point.

Composes the batched VERSY (batched/versy.py) with batched ports of
NOISY(8), TIMVAR, EF1VAR, F2VAR, ESIND and IONSET (KM=1 per grid point).
The scalar chain's data-dependent branches — EF1VAR's F1 present/absent
(Z<=ZMAX), F2VAR's twilight / F1->F2 merge, IONSET's consistency zeroing —
are evaluated whole-batch and merged with torch.where, exactly the DAZEL
technique (unselected lanes may be inf/NaN and are discarded).  The
IKIM-driven map loops inside VERSY/NOISY stay data-independent, so they need
no masking.

Precision mirrors the Stage A ports: VERSY keeps its real*8 basis, the layer
algebra is REAL*4 (float32 tensors, float64-evaluate-once transcendentals).
On CUDA the transcendentals are CUDA libm, tracking Stage A/the oracle to
the [[f32-libm-noise]] floor.
"""

import numpy as np
import torch

from voacap_gpu.batched.versy import versy_batch

_F32 = torch.float32
_F64 = torch.float64

# /CON/ constants as their exact float32 values (python floats stay "weak").
D2R = float(np.float32(0.01745329251))
R2D = float(np.float32(57.295779513))
_HR2RAD = float(np.float32(0.261799387))
_LON_HALF = float(np.float32(0.0087266466))
_LAT_D2R = float(np.float32(0.01745329252))
_BETAE = float(np.float32(5.5))
_BETAF1 = float(np.float32(4.0))
_XF1 = float(np.float32(1.1))
_DELZ = float(np.float32(2.0))

# SUN(2,12) subsolar declinations (deg), from blkdat.
_SUN1 = [-23.05, -17.30, -7.88, 4.26, 14.84, 21.93,
         23.15, 18.20, 8.55, -2.90, -14.20, -21.66]
_SUN2 = [-17.31, -7.89, 4.21, 14.83, 21.93, 23.45,
         18.23, 8.68, -2.86, -14.16, -21.68, -23.45]


def _cos32(x):
    return torch.cos(x.to(_F64)).to(_F32)


def _sin32(x):
    return torch.sin(x.to(_F64)).to(_F32)


def _acos32(x):
    return torch.arccos(x.to(_F64)).to(_F32)


def _exp32(x):
    return torch.exp(x.to(_F64)).to(_F32)


def _log32(x):
    return torch.log(x.to(_F64)).to(_F32)


def _sqrt(x):
    return torch.sqrt(x)  # float32 IEEE, correctly rounded on CPU+CUDA


def noisy8_batch(hmym, abmap, xla, ceg):
    """NOISY(KJ=8) RAT map, batched.  hmym: (29,16) float32 tensor (P slice
    8); abmap: (2,3) float32 tensor; xla/ceg: (N,) float32 (deg)."""
    lm, ln = 15, 10
    alf = float(abmap[0, 1])
    bet = float(abmap[1, 1])
    # longitude half-angle sine harmonics
    q = _LON_HALF * ceg
    c1 = _cos32(q)
    s1 = _sin32(q)
    n = xla.shape[0]
    sx = [None] * (ln + 1)
    sx[1] = s1
    cx = c1
    for k in range(2, ln + 1):  # DO 115
        tx = sx[k - 1]
        sx[k] = tx * c1 + cx * s1
        cx = cx * c1 - tx * s1
    zz = [None] * (lm + 1)
    for j in range(1, lm + 1):  # DO 125
        r = torch.zeros(n, dtype=_F32, device=xla.device)
        for k in range(1, ln + 1):  # DO 120
            r = r + sx[k] * hmym[j - 1, k - 1]
        zz[j] = r + hmym[j - 1, 15]  # + P(J,16,KJ)
    # latitude series (in-place sin(k*Q) recurrence)
    q = _LAT_D2R * (xla + float(np.float32(90.0)))
    s1 = _sin32(q)
    c1 = _cos32(q)
    sx1 = s1
    cx = c1
    r = torch.zeros(n, dtype=_F32, device=xla.device)
    for k in range(1, lm + 1):  # DO 130
        r = r + sx1 * zz[k]
        ss = sx1 * c1 + cx * s1
        cx = cx * c1 - sx1 * s1
        sx1 = ss
    return r + alf + bet * q


def _ret_term(fac):
    return fac * _log32((fac + 1.0) / (fac - 1.0))


def iono_params_batch(st, ab, month, ssn, gmt, clat, clong, glat, gmdip,
                      gyz, psc=(1.0, 1.0, 1.0, 0.7)):
    """Batched TIMVAR->EF1VAR->F2VAR->ESIND->IONSET at N grid points.

    st: IonoState; ab: (319,) float32 tensor (VIRTIM output); month/ssn/gmt
    scalars; clat/clong/glat/gmdip/gyz: (N,) float32 tensors (radians).
    Returns a dict of (N,) float32 tensors (foE=fi1, foF1=fi2, foF2=fi3,
    M3000=f2m3, Es deciles fs1..3, semithicknesses, heights, rat, ...).
    """
    device = clat.device
    ssn = float(np.float32(ssn))
    gmt = float(np.float32(gmt))
    psc1, psc2, psc3, psc4 = (float(np.float32(p)) for p in psc)
    z0 = torch.zeros_like(clat)

    gamma = versy_batch(st.ikim, ab, clat, clong, gmdip, 1, 6)  # (N,7)
    anew = [float(v) for v in st.anew]
    bnew = [float(v) for v in st.bnew]
    achi = [float(v) for v in st.achi]
    bchi = [float(v) for v in st.bchi]
    hmym = torch.as_tensor(np.asarray(st.hmym), dtype=_F32, device=device)
    abmap = torch.as_tensor(np.asarray(st.abmap), dtype=_F32, device=device)

    # ---- TIMVAR: local time + solar zenith angle ----
    ssl = (float(np.float32(180.0)) - float(np.float32(15.0)) * gmt) * D2R
    s1m = float(np.float32(_SUN1[month - 1]))
    s2m = float(np.float32(_SUN2[month - 1]))
    cendog = (clat - D2R * s1m).abs()
    ssp2 = s2m * D2R
    cencat = (clat - ssp2).abs()
    ssp = torch.where(cendog - cencat > 0.0, torch.full_like(clat, s1m * D2R),
                      torch.full_like(clat, ssp2))
    clock = gmt + clong / _HR2RAD
    clock = torch.where(float(np.float32(24.0)) - clock < 0.0, clock - 24.0,
                        torch.where(clock <= 0.0, clock + 24.0, clock))
    zang = clong - ssl
    cycen = (_sin32(clat) * _sin32(ssp)
             + _cos32(clat) * _cos32(ssp) * _cos32(zang))
    cycen = _acos32(cycen)

    # ---- EF1VAR: E and F1 parameters ----
    g6 = gamma[:, 6]
    # midnight fallback (scalar in ssn): F32(0.36)*sqrt(1 + 0.0098*ssn)
    g6_mid = float(np.float32(0.36) * np.sqrt(
        np.float32(1.0) + np.float32(0.0098) * np.float32(ssn)))
    g6 = torch.where(g6 - 0.36 < 0.0, torch.full_like(g6, g6_mid), g6)
    fi1 = g6 * psc1
    hi1 = torch.full_like(clat, float(np.float32(110.0)))
    yi1 = hi1 / _BETAE
    cycen = cycen.abs()
    zdeg = cycen * R2D
    cosz = _cos32(cycen)
    cosdi = _cos32(gmdip)
    zmax = (achi[0] + bchi[0] * ssn + (achi[1] + bchi[1] * ssn) * cosdi)
    zmax = zmax * torch.ones_like(clat)
    zenmax = zmax
    zenang = zdeg
    f1 = ((anew[0] + bnew[0] * ssn) + (anew[1] + bnew[1] * ssn) * cosz
          + (anew[2] + bnew[2] * ssn) * cosz * cosz) * psc2
    hi2_day = float(np.float32(165.0)) + float(np.float32(0.6428)) * zdeg
    yi2_day = hi2_day / _BETAF1
    f1_present = zdeg <= zmax
    fi2 = torch.where(f1_present, f1, z0)
    hi2 = torch.where(f1_present, hi2_day, z0)
    yi2 = torch.where(f1_present, yi2_day, z0)

    # ---- TIMVAR continued: RAT (NOISY 8) and absorption index ----
    gm = glat.abs() * R2D - float(np.float32(45.0))
    zn2 = torch.where(clock - 12.0 < 0.0, -zdeg, zdeg) + float(np.float32(180.0))
    ym = noisy8_batch(hmym, abmap, gm, zn2)
    rat = torch.where(ym - 2.0 < 0.0, torch.full_like(ym, 2.0), ym)
    abiy = (float(np.float32(-0.04))
            + _exp32(float(np.float32(-2.937))
                     + float(np.float32(0.8445)) * fi1))

    # ---- F2VAR: F2 parameters (may move HI2/YI2) ----
    g4, g5 = gamma[:, 4], gamma[:, 5]
    f2m3 = g5
    hpf2 = float(np.float32(1490.0)) / g5 - float(np.float32(176.0))
    fi3 = (g4 + float(np.float32(0.5)) * gyz) * psc3
    fi2 = torch.minimum(fi2, fi3 - float(np.float32(0.2)))
    fc = float(np.float32(0.834)) * fi3
    fcec = torch.clamp(fc / fi1, min=float(np.float32(1.1)))
    ret = (_ret_term(fcec) - 2.0) * yi1
    fc1 = fi2
    has_f1 = fc1 > 0.0
    fc1_safe = torch.where(has_f1, fc1, torch.ones_like(fc1))
    ffec = torch.clamp(fc / fc1_safe, min=_XF1)
    twilight = (zmax - _DELZ - zenang) < 0.0
    # day path (label 210)
    rft_day = float(np.float32(0.5)) * yi2 * (_ret_term(ffec) - 2.0)
    # twilight path (label 215)
    zn = zmax - _DELZ
    hn = float(np.float32(165.0)) + float(np.float32(0.6428)) * zn
    hi2_safe = torch.where(hi2 != 0.0, hi2, torch.ones_like(hi2))
    yn = hn * (yi2 / hi2_safe)
    rfn = float(np.float32(0.5)) * yn * (_ret_term(ffec) - 2.0)
    sz = (zenang - zn) / _DELZ
    rft_twi = rfn * (1.0 - sz)
    hm = hpf2 - ret
    ymv = hm / rat
    dhn = (hm - ymv) - (hn - yn)
    dh = dhn * (1.0 - sz)
    hi2_220 = (hm - ymv) - dh + yi2
    y1max = yn + (ymv - yn) * (fc1 / fi3 - float(np.float32(0.834))) / float(
        np.float32(0.166))
    yi2_225 = yn + (y1max - yn) * sz
    hi2_225 = (hm - ymv) - dh + yi2_225
    # merge twilight sub-branches (220: dhn>0; 225: also fc1-fc>0)
    do220 = twilight & (dhn > 0.0)
    do225 = do220 & (fc1 - fc > 0.0)
    yi2 = torch.where(has_f1 & do225, yi2_225, yi2)
    hi2 = torch.where(has_f1 & do225, hi2_225,
                      torch.where(has_f1 & do220, hi2_220, hi2))
    rft = torch.where(twilight, rft_twi, rft_day)
    ret = torch.where(has_f1, ret + rft, ret)
    hi3 = hpf2 - ret
    yi3 = hi3 / rat

    # ---- ESIND: sporadic-E deciles ----
    g1, g2, g3 = gamma[:, 1], gamma[:, 2], gamma[:, 3]
    fs1 = g3 * psc4
    fs2 = g2 * psc4
    fs3 = g1 * psc4
    hs = torch.full_like(clat, float(np.float32(110.0)))

    # ---- IONSET (KM=1): consistency ----
    merge_e = fi2 - fi1 - float(np.float32(0.2)) <= 0.0
    merge_f2 = fi3 - fi2 - float(np.float32(0.2)) <= 0.0
    zero_f1 = has_f1 & (merge_e | merge_f2)
    adjust = has_f1 & (~merge_e) & (~merge_f2)
    hi2 = torch.where(adjust, torch.minimum(hi2, hi3), hi2)
    fi2 = torch.where(zero_f1, z0, fi2)
    yi2 = torch.where(zero_f1, z0, yi2)
    hi2 = torch.where(zero_f1, z0, hi2)
    yi3 = torch.minimum(yi3, hi3 - hi1 - float(np.float32(2.0)))

    return {
        "fi1": fi1, "fi2": fi2, "fi3": fi3, "yi1": yi1, "yi2": yi2,
        "yi3": yi3, "hi1": hi1, "hi2": hi2, "hi3": hi3, "fs1": fs1,
        "fs2": fs2, "fs3": fs3, "hs": hs, "f2m3": f2m3, "hpf2": hpf2,
        "rat": rat, "abiy": abiy, "clck": clock, "zenang": zenang,
        "zenmax": zenmax,
    }
