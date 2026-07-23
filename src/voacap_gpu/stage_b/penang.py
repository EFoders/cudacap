"""Stage B (PyTorch, batched) penetration take-off angles — PENANG.

DELPEN(IL) is the take-off angle above which a ray at the operating frequency
punches through layer IL instead of reflecting from it.  FINDF needs it before
it starts scanning: it is both the gate on whether a layer is worth searching
at all and the cusp angle FINDF splices into the angle scan.

Scalar PENANG is a nest of early returns -- the E layer penetrating short-
circuits F1 and F2, an absent F1 makes F1 share E's angle, F1 penetrating
short-circuits F2 -- so the batched version computes every arm and selects.
The arms are cheap (a divide, a sqrt, an ACOS) and the nesting is only three
deep, so there is nothing to be gained by being clever about it.

TWO GUARDS EXIST ONLY TO KEEP THE UNTAKEN ARMS FINITE.  ``sqrt(1 - FRAT)`` is
evaluated on lanes where FRAT > 1 and ``ACOS(CDEL)`` on lanes where CDEL > 1;
both are NaN there and ``torch.where`` would discard them, but NaN spreading
through a later arithmetic chain is a poor thing to rely on.  Each is clamped
at exactly the bound its own branch test already guarantees, so on the lanes
that DO take the arm the clamp is a no-op and the value is unchanged.

float32.  ACOS carries the [[f32-libm-noise]] floor and is ill-conditioned as
CDEL approaches 1 -- but the branch that would expose that (CDEL > 0.999999)
takes the DELPEN = 0 path instead, so the pole is never evaluated.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.constants import R2D, RZ
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
_CDEL_MAX = float(np.float32(0.999999))
_FRAT_MAX = float(np.float32(0.9999))

# The ionogram indices PENANG probes, as 0-based rows of a (30, N) table.
_E_CUSP = 10 - 1
_F1_CUSP = 20 - 1
_F2 = (28 - 1, 29 - 1, 30 - 1)


def _cusp_angle(htrue_row, fvert_row, freq, kc):
    """The layer cusp angle, plus whether the layer reflects at all.

    Returns ``(delpen, reflects)``.  ``delpen`` is meaningless where
    ``reflects`` is false -- the caller substitutes 89.9 there, as the Fortran
    does by jumping past this.
    """
    frat = fvert_row / freq
    frat = frat * frat
    reflects = frat - kc["frat_max"] < 0.0
    # clamped at the bound `reflects` already enforces: a no-op where it holds
    root = torch.sqrt(torch.clamp(kc["one"] - frat, min=0.0))
    cdel = (kc["rz"] + htrue_row) * root / kc["rz"]
    delpen = torch.where(
        cdel - kc["cdel_max"] <= 0.0,
        torch.acos(torch.clamp(cdel, max=_CDEL_MAX)) * kc["r2d"],
        torch.zeros_like(cdel))
    return delpen, reflects


def penang_batch(htrue, fvert, fi2, freq):
    """Batched PENANG.

    htrue/fvert: (30, N) from ``genion_batch``.  fi2: (N,) F1 critical
    frequency, <= 0 meaning no F1 ledge.  freq: (N,) operating frequency, MHz.

    Returns ``(delpen, nose_penetrates)`` with delpen a (4, N) float32 table
    indexed by the Fortran's 1-based layer number (row 0 unused) and
    ``nose_penetrates`` a (N,) bool -- the flag that makes the caller set
    DSKPKM/DMAXKM to -1 because even the F2 nose lets the ray through.
    """
    n = freq.shape[0]
    dev = freq.device
    # Every constant goes through `_k`: a float32 tensor combined with a
    # PYTHON float is not reliably a float32 operation on CUDA, which is the
    # trap that cost three bit-exactness bugs earlier in Stage B.
    kc = {"rz": _k(RZ, freq), "r2d": _k(R2D, freq), "one": _k(1.0, freq),
          "cdel_max": _k(_CDEL_MAX, freq), "frat_max": _k(_FRAT_MAX, freq),
          "eps": _k(0.0001, freq)}
    rz = kc["rz"]

    d_e, e_reflects = _cusp_angle(htrue[_E_CUSP], fvert[_E_CUSP], freq, kc)
    d_f1, f1_reflects = _cusp_angle(htrue[_F1_CUSP], fvert[_F1_CUSP], freq, kc)
    has_f1 = fi2 > 0.0

    # ---- F2: label 155 ---------------------------------------------------
    # Below its own critical the F2 layer reflects everything, so 89.9 stands
    # in for "no penetration angle".  Above it, the nose is probed at three
    # heights because the thick F2 layer is not spherically symmetric enough
    # for the top sample alone to give the highest refractive height.
    f2_pen = fvert[_F2[2]] - freq + kc["eps"] < 0.0
    xm = []
    for row in _F2:
        t = fvert[row] / freq
        xm.append((rz + htrue[row]) * torch.sqrt(
            torch.clamp(kc["one"] - t * t, min=0.0)))
    xm28, xm29, xm30 = xm
    # IF(XM30-XM29) 161,157,157 then IF(XMxx-XM28) -- both arms reduce to
    # "the higher of the two candidates, unless XM28 beats it"
    hi = torch.where(xm30 - xm29 < 0.0, xm29, xm30)
    cdel = torch.where(hi - xm28 < 0.0, xm28, hi) / rz
    nose_reflects = cdel - kc["cdel_max"] <= 0.0
    d_f2 = torch.where(
        f2_pen,
        torch.where(nose_reflects,
                    torch.acos(torch.clamp(cdel, max=_CDEL_MAX)) * kc["r2d"],
                    torch.zeros_like(cdel)),
        torch.full((n,), float(np.float32(89.9)), dtype=_F32, device=dev))

    # ---- stitch the early returns back together --------------------------
    c89_9 = torch.full((n,), float(np.float32(89.9)), dtype=_F32, device=dev)
    c90 = torch.full((n,), 90.0, dtype=_F32, device=dev)

    delpen = torch.zeros((4, n), dtype=_F32, device=dev)
    delpen[1] = torch.where(e_reflects, d_e, c89_9)
    # no F1 ledge: F1 inherits the E angle; F1 present but penetrated: 89.9
    f1_ang = torch.where(has_f1, torch.where(f1_reflects, d_f1, c89_9),
                         delpen[1])
    delpen[2] = torch.where(e_reflects, f1_ang, c90)
    # F2 is only reached when E reflects and F1 either is absent or reflects
    reach_f2 = e_reflects & (~has_f1 | f1_reflects)
    delpen[3] = torch.where(reach_f2, d_f2, c90)

    return delpen, reach_f2 & f2_pen & ~nose_reflects
