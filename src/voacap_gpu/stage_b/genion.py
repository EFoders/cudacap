"""Stage B (PyTorch, batched) ionogram — GENION.

Lays out thirty vertical-sounding frequencies across the D-E tail, the E nose
and the F region, then fills the virtual and true reflection height at each
one.  With ``lecden_batch`` for the profile and ``gethp_batch`` for the
integration, this closes the chain from a lat/lon grid to a GPU-resident
ionogram at every point.

The frequency layout is a set of anchors -- the E critical scaled by 0.957,
0.99 and 1.05, the F1 critical by 0.99 and 1.01, the F2 critical by 0.92,
0.96, 0.98 and 0.99 -- with even ladders filling the gaps between them.  The
anchors either side of each critical frequency are what put sounding points
tight around the cusps, where the ionogram bends fastest.  The gap-filling has
two shapes depending on whether an F1 layer exists, and as in ``profile.py``
both are built and selected.

**Only the IEDP < 0 model is implemented**, which is the one that runs.  IEDP
is -1 in blkdat and is changed only by the INTEGRATE card, which none of the
golden decks carry; the alternative is a faster approximation (a fixed
ten-point parabolic-E table, true heights by lookup, virtual heights from the
BENDY/PEN formula) that would need three more routines batched to no current
purpose.  Refused explicitly rather than silently treated as the Gaussian one,
because the two give different ionograms.
"""

import numpy as np
import torch

from voacap_gpu.stage_b.gethp import gethp_batch
from voacap_gpu.stage_b.profile import _k, _ladder

_F32 = torch.float32

NFV = 30                        # vertical-sounding frequencies
_XTAIL = float(np.float32(0.85))    # must agree with LECDEN
_F_FLOOR = float(np.float32(0.01))  # FVERT(1), the bottom of the sweep


def genion_batch(htr, fnsq, fi1, fi2, fi3, iedp=-1):
    """Batched GENION.

    htr/fnsq: (50, N) from ``lecden_batch``.  fi1/fi2/fi3: (N,) E, F1 and F2
    critical frequencies -- fi2 <= 0 means no F1 layer.  Returns
    ``(fvert, hprim, htrue)``, each (30, N).
    """
    if iedp >= 0:
        raise NotImplementedError(
            f"IEDP is {iedp}: the fast parabolic-E model (genion.for label "
            "705) needs XLIN, BENDY and PEN batched.  Only the Gaussian "
            "model (IEDP < 0, blkdat's default) is implemented")

    n = fi1.shape[0]
    fv = torch.empty((NFV, n), dtype=_F32, device=fi1.device)

    # ---- D-E region tail -------------------------------------------------
    fex = fi1 * torch.sqrt(_k(1.0 - _XTAIL * _XTAIL, fi1)).expand_as(fi1)
    fv[0] = _F_FLOOR
    fv[3] = fex
    fdif = (fv[3] - fv[0]) / _k(3.0, fi1)
    fv[1:3] = _ladder(fv[0], fdif, 2)
    # ---- E region nose (the guess for the E MUF) -------------------------
    fv[8] = _k(0.957, fi1) * fi1
    fv[9] = _k(0.99, fi1) * fi1
    fdif = (fv[8] - fv[3]) / _k(5.0, fi1)
    fv[4:8] = _ladder(fv[3], fdif, 4)
    # ---- the E-F cusp and the F region nose ------------------------------
    fv[10] = _k(1.05, fi1) * fi1
    fv[29] = _k(0.99, fi1) * fi3
    fv[28] = _k(0.98, fi1) * fi3
    fv[27] = _k(0.96, fi1) * fi3
    fv[26] = _k(0.92, fi1) * fi3

    # ---- the gap between the cusps, in two shapes ------------------------
    has_f1 = (fi2 > 0).unsqueeze(0)
    # no F1: one even run from the E-F cusp to the F2 shoulder
    d_no = (fv[26] - fv[10]) / _k(16.0, fi1)
    run_no = _ladder(fv[10], d_no, 15)                  # 1-based 12..26
    # F1 present: a run up to the F1 nose, the F1-F2 cusp, then a run on
    f1_nose = _k(0.99, fi1) * fi2
    d_a = (f1_nose - fv[10]) / _k(9.0, fi1)
    run_a = _ladder(fv[10], d_a, 8)                     # 1-based 12..19
    f1_cusp = _k(1.01, fi1) * fi2
    d_b = (fv[26] - f1_cusp) / _k(6.0, fi1)
    run_b = _ladder(f1_cusp, d_b, 5)                    # 1-based 22..26

    fv[11:19] = torch.where(has_f1, run_a, run_no[:8])
    fv[19] = torch.where(has_f1.squeeze(0), f1_nose, run_no[8])
    fv[20] = torch.where(has_f1.squeeze(0), f1_cusp, run_no[9])
    fv[21:26] = torch.where(has_f1, run_b, run_no[10:15])

    # ---- virtual and true heights ----------------------------------------
    hprim, htrue = gethp_batch(htr, fnsq, fv)
    return fv, hprim, htrue
