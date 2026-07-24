"""Stage B (PyTorch, batched) deviative loss — ALOSFV.

AFAC is the extra absorption a ray pays for the distance it travels INSIDE a
layer rather than reflecting cleanly off it: the average collision frequency,
falling exponentially with height, times the group-minus-phase path excess.
Only modes reflecting above the height of the MUF pay it, so most of the table
is zero.  FINDF carries AFAC through into the ray set, which is why this sits
next to it.

THE SHAPE THAT MATTERS IS THE CF CARRY.  ALOSFV is not thirty independent
cells: ``CF`` survives each loop and is reused as the leading coefficient of
the next, which is how the loss is forced continuous across the E->F1 and
F1->F2 cusps.  It is a running value updated only on the cells that took the
active branch -- a last-active-wins carry, not a reduction with a name.

So the port keeps the height index as a PYTHON loop over thirty rows and
batches only across grid points.  Thirty unrolled iterations of a handful of
elementwise kernels is nothing next to the lane count, and it makes the carry
fall out for free: ``cf = where(active, cf_here, cf)`` is the assignment the
Fortran performs, one row at a time, with no reformulation to get wrong.

The F1-present and F1-absent arms are computed in full and selected, the usual
Stage B trade.  They differ in more than a bound -- the no-F1 arm reads YI(3)
where the F1 arm reads YI(2), and clamps CF at 0.5 rather than 0.05 in its
upper loop -- so there is no merging them.

TWO FORTRAN QUIRKS ARE PRESERVED DELIBERATELY.  The F1 arm's 14..20 loop
scales by A(3) immediately after setting A(2), which is harmless only because
both happen to be 0.1 in the DATA statement.  And CF is left UNDEFINED if no
E-layer point reflects above HTMUF(1), so the E->F cusp coefficient is then
whatever the compiler left behind; here that is a well-defined 0.0, which is
the only place this port cannot be bit-faithful to a value the Fortran never
really had.  Realistic MUF heights always set it.

float32.  EXP is the one libm call, carrying the [[f32-libm-noise]] floor,
amplified by the HPRIM - HTRUE - HM cancellation.
"""

import torch

from voacap_gpu.batched.profile import _k

_F32 = torch.float32
NFV = 30

_A1 = 0.2                       # DATA A/0.2,0.1,0.1/
_A_DEFAULT = 0.1
_ZEXP_FLOOR = -10.0             # AMAX1(ZEXP,-10.), the underflow guard


def alosfv_batch(htrue, hprim, fi2, yi, hi1, hpmuf, htmuf):
    """Batched ALOSFV.

    htrue/hprim: (30, N) ionogram heights from ``genion_batch``.
    fi2: (N,) F1 critical frequency; <= 0 means no F1 ledge.
    yi: (3, N) layer semi-thicknesses, rows 0..2 for Fortran YI(1..3).
        **YI(2) must be LECDEN's adjusted value**, not the input one.  LECDEN
        narrows it in place when the F1 ledge has to be reconciled with the E
        layer, and the Fortran reads /RON/ afterwards, so the F1 arm's scale
        height is the post-LECDEN one; ``lecden_batch`` returns it as
        ``yi2_n`` for exactly this.  Passing the input value shifts rows
        14..17 by up to 5% on the lanes LECDEN touched while leaving
        everything else right, which reads as noise rather than as a bug.
    hi1: (N,) E-layer height HI(1).
    hpmuf/htmuf: (3, N) virtual and true MUF heights, rows 0..2 for layers 1..3.

    Returns AFAC as a (30, N) float32 tensor, Fortran IH = 1..30 as rows 0..29.
    """
    n = htrue.shape[1]
    dev = htrue.device
    k2 = _k(-2.0, htrue)
    ka1, kad = _k(_A1, htrue), _k(_A_DEFAULT, htrue)
    zero = torch.zeros(n, dtype=_F32, device=dev)

    def cf_at(ih, hz, y, lead):
        """A(lead) * EXP(-2*(HTRUE(IH)-HZ)/Y), floored as the Fortran floors
        it.  `ih` is the Fortran 1-based height index."""
        zexp = k2 * (htrue[ih - 1] - hz) / y
        return lead * torch.exp(torch.clamp(zexp, min=_ZEXP_FLOOR))

    def loss(ih, cf, hm):
        return cf * (hprim[ih - 1] - htrue[ih - 1] - hm)

    afac = [zero] * (NFV + 1)           # 1-based; slot 0 unused

    # ---- E layer, IH 1..10 ------------------------------------------------
    hm1 = hpmuf[0] - htmuf[0]
    hz = hi1 - yi[0]
    cf = zero
    for ih in range(1, 11):
        active = htrue[ih - 1] - htmuf[0] > 0.0
        here = cf_at(ih, hz, yi[0], ka1)
        afac[ih] = torch.where(active, loss(ih, here, hm1), zero)
        cf = torch.where(active, here, cf)      # the cusp carry
    cf_e = cf

    def _upper(y_low, y_high, hz_hi_row, htm, htp, low_floor, hi_floor,
               hi_lo, hi_hi):
        """The shape both arms share above the E layer.

        A two-cell bridge off the E nose whose leading coefficient is the E
        layer's last CF (that is the continuity), a hard zero at the cusp
        itself, then the layer proper gated on the MUF height.  Returns the
        rows it fills plus the CF left over, which the F1 arm hands on to F2.
        """
        out = {}
        hz_lo = htrue[10]                       # HTRUE(11)
        hm = hprim[12] - htrue[12]              # HPRIM(13)-HTRUE(13)
        # A(2) is set ONCE from the E layer's carry and both cells scale by
        # it; only CF itself is overwritten per cell.  Threading the clamped
        # CF back in as the leading coefficient would compound it.
        c = cf_e
        for ih in (11, 12):
            c = torch.clamp(cf_at(ih, hz_lo, y_low, cf_e), min=low_floor)
            out[ih] = loss(ih, c, hm)
        out[13] = zero
        hm = htp - htm
        for ih in range(hi_lo, hi_hi + 1):
            active = htrue[ih - 1] - htm > 0.0
            here = torch.clamp(cf_at(ih, hz_hi_row, y_high, kad),
                               min=hi_floor)
            out[ih] = torch.where(active, loss(ih, here, hm), zero)
            c = torch.where(active, here, c)
        return out, c

    # ---- arm A: no F1 ledge (label 115) -- E straight into F2 -------------
    # YI(3) throughout, and the upper loop floors CF at 0.5, not 0.05.
    a_rows, _ = _upper(yi[2], yi[2], htrue[10], htmuf[2], hpmuf[2],
                       0.05, 0.5, 14, 30)

    # ---- arm B: F1 present (label 140) -- E -> F1, then F1 -> F2 ----------
    b_rows, cf_f1 = _upper(yi[1], yi[1], htrue[10], htmuf[1], hpmuf[1],
                           0.05, 0.05, 14, 20)
    # F1 -> F2: the same bridge again, now off the F1 nose with CF carried
    hz2 = htrue[20]                             # HTRUE(21)
    hm2 = hprim[22] - htrue[22]                 # HPRIM(23)-HTRUE(23)
    for ih in (21, 22):
        c = torch.clamp(cf_at(ih, hz2, yi[2], cf_f1), min=0.05)
        b_rows[ih] = loss(ih, c, hm2)
    b_rows[23] = zero
    hm2 = hpmuf[2] - htmuf[2]
    for ih in range(24, 31):
        active = htrue[ih - 1] - htmuf[2] > 0.0
        here = torch.clamp(cf_at(ih, hz2, yi[2], kad), min=0.05)
        b_rows[ih] = torch.where(active, loss(ih, here, hm2), zero)

    # ---- select the arm, then floor at zero (DO 205) ----------------------
    has_f1 = fi2 > 0.0
    out = torch.empty((NFV, n), dtype=_F32, device=dev)
    for ih in range(1, NFV + 1):
        if ih <= 10:
            v = afac[ih]
        else:
            v = torch.where(has_f1, b_rows[ih], a_rows[ih])
        out[ih - 1] = torch.clamp(v, min=0.0)
    return out
