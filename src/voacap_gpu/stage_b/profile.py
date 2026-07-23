"""Stage B (PyTorch, batched) electron-density profile — LECDEN.

``area_iono_map`` gives the layer parameters at every grid point.  This turns
them into the thing the rest of the model actually reads: a 50-point profile
of plasma frequency squared against true height, at every point at once.

The Fortran builds that profile as a pointwise MAXIMUM of five overlapping
layer contributions -- a D-region exponential tail, a parabolic E, a linear
valley bridging E to F2, a parabolic-or-linear F1, and a parabolic F2 -- each
one masked to the height range where it is defined.  That structure is what
makes the routine batch cleanly: the max is elementwise, so once the five
candidates are evaluated over the whole (50, N) grid the assembly is one
``torch.amax``.  A contribution outside its range contributes 0, exactly as
the scalar code leaves its accumulator at 0.

Two parts need the compute-both-and-select treatment (the DAZEL technique
already used for EF1VAR/F2VAR/IONSET in ``layers.py``):

**The F1 fit** decides between four outcomes -- no F1 at all, a parabolic F1
with the layer moved, a linear F1, and a linear F1 whose semithickness is
re-derived from the E-layer separation.  Those are nested arithmetic IFs on
the data, so every grid point can take a different one.  All four are
evaluated and merged with ``torch.where``.  The fit also MUTATES HI(2)/YI(2)
and writes FSECV, so the batched version returns the updated values rather
than writing through -- a caller that needs the Fortran's aliasing gets it by
using what comes back.

**The height grid** has two shapes, with and without an F1 layer, differing in
where the 18..50 points are distributed.  Both are built and selected.

TWO WAYS TO GET THE HEIGHT LADDER WRONG, both of which this hit.  The Fortran
walks ``htr[ih] = htr[ih-1] + hdif`` in float32, so neither of the obvious
vectorisations reproduces it: ``start + i*hdif`` rounds once where the
original rounds i times, and ``cumsum`` is free to use a parallel scan that
associates differently -- it put CPU and CUDA 1 ULP apart on identical input.
``_ladder`` therefore accumulates sequentially, looping over the height index
only (at most 31 steps), which costs kernel launches rather than anything per
grid point.  And see ``_k``: dividing by a Python float on CUDA does not round
like a float32 divide, which moved HDIF itself.

Both bugs surfaced as CPU/CUDA disagreements in a quantity built from nothing
but adds and divides -- which is why the device test asserts bitwise equality
there rather than a tolerance.

Precision follows Stage A: float32 throughout, EXP and SQRT only.
"""

import numpy as np
import torch

_F32 = torch.float32

NHT = 50                        # HTR/FNSQ points
_XLOW = float(np.float32(0.8516))
_XUP_FAC = float(np.float32(0.98))
_XTAIL = float(np.float32(0.85))
_HD = float(np.float32(70.0))   # D-region base height, km
_MIN_DENOM = float(np.float32(0.17))
_EPS_F1 = float(np.float32(0.001))
_EPS_HB = float(np.float32(0.00001))


def _k(value, like):
    """A constant as a float32 SCALAR TENSOR on the right device.

    Not cosmetic.  Dividing a float32 tensor by a Python float on CUDA does
    not give the same bits as dividing by a float32 tensor: the scalar is a
    double, and that path rounds differently from a true single-precision
    divide.  Measured on ``(377.234589-188.384598)/22`` -- CPU, numpy and a
    float32-tensor divisor all give 0x6f580941, a CUDA Python-float divisor
    gives 0x70580941.  One ULP, but it accumulates through the height ladder
    and put the two devices 1 ULP apart at nine of the fifty heights.

    Multiplication by an exact power of two is safe either way; every other
    constant operand goes through here.
    """
    return torch.tensor(float(np.float32(value)), dtype=_F32,
                        device=like.device)


def _sqrt(x):
    return torch.sqrt(x.to(_F32))


def _exp32(x):
    return torch.exp(x.to(_F32))


def _ladder(start, hdif, n):
    """``h[i] = h[i-1] + hdif`` for i = 1..n, as an (n, N) tensor.

    SEQUENTIALLY accumulated, deliberately.  Neither ``start + i*hdif`` nor
    ``cumsum`` reproduces the Fortran: the multiply rounds once where the
    Fortran rounds i times, and cumsum is free to use a parallel scan that
    associates differently -- which showed up immediately as a 1-ULP split
    between CPU and CUDA on the same input.  The loop is over the height
    index only (at most 31 steps), so it costs a few kernel launches rather
    than anything per grid point.
    """
    rows = []
    h = start
    for _ in range(n):
        h = h + hdif
        rows.append(h)
    return torch.stack(rows)


def lecden_batch(fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3):
    """Batched LECDEN.  All arguments are (N,) float32 tensors.

    Returns ``(htr, fnsq, hi2, yi2, fsecv)`` where htr and fnsq are (50, N)
    -- true height km and plasma frequency squared MHz^2 -- and the last
    three are the F1 values the fit may have moved, (N,) each.

    The IELECT early return (an externally supplied profile) is not modelled:
    it is a per-run input card, not per-grid-point state, so a caller with
    IELECT set should not be calling this at all.
    """
    one = torch.ones_like(fi1)
    zero = torch.zeros_like(fi1)

    hz = hi1 - yi1
    xup = _XUP_FAC * fi1 / fi3
    hlow = hz + yi1 * (one + _sqrt(one - _XLOW * _XLOW))
    hte = hi1 + yi1
    hb2 = hi3 - yi3
    fce = fi1 * fi1
    fc2 = fi3 * fi3
    hup = hb2 + yi3 * (one - _sqrt(one - xup * xup))
    fup = xup * xup * fc2
    flow = torch.full_like(fi1, float(np.float32(_XLOW * _XLOW))) * fce
    asp = torch.where(hup - hlow > 0, (fup - flow) / (hup - hlow), zero)

    # ---- F1-layer fit (label 185 onward) --------------------------------
    has_f1 = fi2 > 0
    fc1 = torch.where(has_f1, fi2 * fi2, zero)
    # HTW is only meaningful where F1 exists; clamp the ratio so the sqrt of
    # a negative never appears in the unused lane.
    ratio = torch.where(has_f1, fc1 / fc2, zero)
    htw = hb2 + yi3 * (one - _sqrt(torch.clamp(one - ratio, min=0.0)))

    hb1_0 = hi2 - yi2                    # the ORIGINAL hi/yi, lines 42-43
    ht1_0 = hi2 + yi2

    # --- outcome A (label 205): F1 stays put, semithickness clipped -------
    yi2_A = torch.minimum(yi2, hi2 - hi1 + one)
    # --- the 190 branch: the layer moves down to HTW ----------------------
    ys = torch.clamp(htw - hb1_0, min=1.0)
    hi2_B = htw
    # --- outcome C (label 194 -> 205): F1 base below F2 base -------------
    yi2_C = torch.minimum(hi2_B - hb2, hi2_B - hi1 + one)
    # --- outcome D (label 195): linear F1 --------------------------------
    # (FI(1)/FI(2))**2 with an INTEGER exponent: gfortran emits a multiply,
    # so square it explicitly rather than through pow, which is a different
    # function and rounds differently on CUDA than on CPU.
    _r = fi1 / torch.where(has_f1, fi2, one)
    denom = torch.clamp(one - _r * _r, min=_MIN_DENOM)
    yb = (hi2_B - hi1) / denom
    take_yb = ys - yb >= 0                       # label 401
    ys_D = torch.where(take_yb, yb, ys)
    # Label 401 also sets HB1 = HI(1,K), which is DEAD: HB1 is recomputed as
    # max(HI(1), HI(2)-YI(2)) before the density loop and never read in
    # between.  Not transcribed; the recomputation below is the live one.

    stays = htw > hi2 + _EPS_F1                  # 185 -> 205
    below = hb2 - hb1_0 < 0                      # 194 -> 205

    lin = has_f1 & ~stays & ~below               # only outcome D is linear
    hi2_n = torch.where(has_f1 & ~stays, hi2_B, hi2)
    yi2_n = torch.where(
        has_f1,
        torch.where(stays, yi2_A,
                    torch.where(below, yi2_C, ys_D)),
        yi2)
    s1 = torch.where(lin, fc1 / ys_D, zero)
    ht1 = torch.where(lin, htw, ht1_0)           # label 415
    fsecv = torch.where(has_f1 & ~stays & ~below, _sqrt(flow),
                        torch.where(has_f1, -one, zero))

    # ---- D-region tail ---------------------------------------------------
    hex_ = hi1 - _XTAIL * yi1
    fnx = torch.full_like(fi1, float(np.float32(1.0 - _XTAIL * _XTAIL)))
    alp = _k(2.0, fi1) * (hi1 - hex_) / (fnx * yi1 * yi1)
    fsq = fnx * _exp32(-alp * (hex_ - _HD))

    # ---- the height grid -------------------------------------------------
    htr = torch.empty((NHT, fi1.shape[0]), dtype=_F32, device=fi1.device)
    htr[0] = _HD
    htr[4] = hex_
    hdif = torch.clamp((htr[4] - htr[0]) * 0.25, min=0.0)
    htr[3] = htr[4] - torch.clamp(hdif, max=1.0)
    htr[1] = htr[0] + hdif
    htr[2] = (htr[1] + htr[3]) * 0.5
    # E below the nose.  The nose itself (1-based 11) is HI(1) and the ladder
    # only fills 1-based 6..10 -- five steps, not six.  Running it one step
    # further would overwrite the endpoint it was scaled to reach.
    htr[10] = hi1
    hdif = (htr[10] - htr[4]) / _k(6.0, htr)
    htr[5:10] = _ladder(htr[4], hdif, 5)
    # E above the nose: 1-based 12..16, with 17 held at HI(1)+YI(1).
    htr[16] = hi1 + yi1
    hdif = (htr[16] - htr[10]) / _k(6.0, htr)
    htr[11:16] = _ladder(htr[10], hdif, 5)
    htr[10] = 0.5 * (htr[9] + htr[11])   # after the ladders, as the Fortran
    htr[49] = hi3

    # Two shapes for points 18..49 (1-based 18..50).
    two_layers = has_f1 & (hb2 - (hi2_n - yi2_n + _EPS_HB) > 0)

    # -- no F1: one even run from the F2 base to the F2 nose --------------
    base_1 = hi3 - yi3
    hd_1 = (htr[49] - base_1) / _k(32.0, htr)
    run_1 = _ladder(base_1, hd_1, 31)                    # 1-based 19..49

    # -- F1 and F2: a run to the F1 nose, then a run to the F2 nose -------
    base_2 = torch.maximum(hi2_n - yi2_n, htr[16] + 1.0)
    mid_2 = hi2_n
    hd_2a = (mid_2 - base_2) / _k(10.0, htr)
    run_2a = _ladder(base_2, hd_2a, 9)                   # 1-based 19..27
    hd_2b = (htr[49] - mid_2) / _k(22.0, htr)
    run_2b = _ladder(mid_2, hd_2b, 21)                   # 1-based 29..49

    htr[17] = torch.where(two_layers, base_2, base_1)
    sel = two_layers.unsqueeze(0)
    htr[18:27] = torch.where(sel, run_2a, run_1[:9])
    htr[27] = torch.where(two_layers, mid_2, run_1[9])
    htr[28:49] = torch.where(sel, run_2b, run_1[10:31])

    # ---- per-height density, the pointwise max of five contributions ----
    h = htr                                              # (50, N)
    hb1 = torch.maximum(hi1, hi2_n - yi2_n)              # line 143

    def b(x):
        return x.unsqueeze(0)

    # linear valley, between HLOW and HUP
    fnval = torch.where((h - b(hlow) > 0) & (h - b(hup) < 0),
                        b(fup) + b(asp) * (h - b(hup)),
                        torch.zeros_like(h))
    # exponential D-E below HEX, parabolic E from there to HTE
    fnd = torch.where(h - b(hex_) < 0,
                      b(fce) * b(fsq) * _exp32(b(alp) * (h - _HD)),
                      torch.zeros_like(h))
    ze = (h - b(hi1)) / b(yi1)
    fne = torch.where((h - b(hex_) >= 0) & (h - b(hte) <= 0),
                      b(fce) * (1.0 - ze * ze),
                      torch.zeros_like(h))
    # F1: parabolic or linear, between HB1 and HT1
    z1 = (h - b(hi2_n)) / b(yi2_n)
    f1_par = b(fc1) * (1.0 - z1 * z1)
    f1_lin = b(s1) * (h - b(hi2_n - yi2_n))
    in_f1 = b(has_f1) & (b(hb1) - h <= 0) & (h - b(ht1) <= 0)
    fn1 = torch.where(in_f1, torch.where(b(lin), f1_lin, f1_par),
                      torch.zeros_like(h))
    # parabolic F2, above the F2 base
    z2 = (h - b(hi3)) / b(yi3)
    fn2 = torch.where(b(hb2) - h <= 0, b(fc2) * (1.0 - z2 * z2),
                      torch.zeros_like(h))

    fnsq = torch.amax(torch.stack((fnd, fne, fnval, fn1, fn2)), dim=0)
    return htr, fnsq, hi2_n, yi2_n, fsecv
