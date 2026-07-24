"""Stage B (PyTorch, batched) virtual/true reflection heights — GETHP.

This is the expensive part of building an ionogram, and therefore the part
worth putting on a GPU.  For every vertical-sounding frequency it integrates
1/mu over the electron-density profile with a 20-point Gauss-Legendre rule
evaluated as a folded 40-point pair, and every one of those 40 nodes needs a
plasma-frequency lookup into the profile table.  Thirty frequencies per grid
point is 1,200 table searches per point before anything else happens.

THE TWO SEARCHES ARE NOT THE SAME SHAPE, and treating them the same would be
either wrong or slow:

* **The node lookup** (label 535..625) walks HTR, which is a height ladder --
  strictly increasing by construction, and asserted so here.  On a monotone
  table the Fortran's "first segment satisfying the arithmetic IF" reduces
  exactly to a bracketing search, so it becomes one ``searchsorted`` instead
  of a 49-step scan.  That is the difference between 40 kernels and 1,960 per
  frequency block.  The reduction is checked, not assumed: ``_require_monotone``
  raises if the ladder ever comes back out of order, rather than silently
  returning a bracketing answer for a table that does not bracket.

* **The true-height search** (label 445..495) walks FNSQ, which is the
  pointwise MAXIMUM of five layer contributions and therefore NOT monotone --
  it rises through the E layer, dips through the valley, and rises again
  through the F region.  A bracketing search would pick the wrong crossing, so
  this one keeps the faithful first-match scan over all 49 segments, with a
  running "already found" mask.  It costs 49 steps, but it runs once per
  frequency where the node lookup runs forty times, so it is 1/40th of the
  work.

Both searches have a flat-segment case that returns a table value directly
rather than interpolating -- the interpolation would divide by zero there.
The denominators are clamped away from zero in the discarded lane so no NaN
is produced at all, rather than relying on ``where`` to discard one.

NPL is 1 (set in voacapw.for), which makes ``ZG**NPL`` the identity and
``ZG**(NPL-1)`` exactly 1.  Both are folded away here; a build with NPL != 1
would need the powers back, so it is refused rather than silently mismodelled.

Precision follows Stage A: float32 throughout, SQRT only -- so GETHP is
bit-clean and this should match Stage A exactly on both devices.
"""

import numpy as np
import torch

from voacap_gpu.scalar.datr import NPL, NT, TWDIV, WT, XNPL, XT

_F32 = torch.float32
NHT = 50
_YSQ_CAP = float(np.float32(0.9999))

if NPL != 1:                    # pragma: no cover - guards a build-time change
    raise NotImplementedError(
        f"/DATR/ NPL is {NPL}; the batched GETHP folds away ZG**NPL and "
        "ZG**(NPL-1) on the assumption NPL == 1 (voacapw.for)")


def _k(value, like):
    """A constant as a float32 scalar tensor -- see stage_b/profile.py._k."""
    return torch.tensor(float(np.float32(value)), dtype=_F32,
                        device=like.device)


def _require_monotone(htr):
    """The node lookup below is a bracketing search, which is only equivalent
    to the Fortran's scan while HTR increases."""
    if not bool((htr[1:] > htr[:-1]).all()):
        bad = int(torch.argwhere(~(htr[1:] > htr[:-1]))[0][0]) + 1
        raise ValueError(
            f"HTR is not strictly increasing (first at height index {bad}); "
            "the batched node lookup brackets and would pick the wrong "
            "segment")


def _true_height(htr, fnsq, fr):
    """HT where FR crosses FNSQ, first match over the 49 segments.

    htr/fnsq are (50, N); fr is (F, N).  Returns (F, N).  FNSQ is not
    monotone, so this is the faithful scan rather than a bracketing search.
    """
    found = torch.zeros_like(fr, dtype=torch.bool)
    out = htr[NHT - 1].expand_as(fr).clone()      # the fall-through, HTR(50)
    for ih in range(NHT - 1):
        f0, f1 = fnsq[ih].unsqueeze(0), fnsq[ih + 1].unsqueeze(0)
        h0, h1 = htr[ih].unsqueeze(0), htr[ih + 1].unsqueeze(0)
        d = f0 - f1
        den = f1 - f0
        safe = torch.where(den == 0, torch.ones_like(den), den)
        interp = h0 + (fr - f0) * (h1 - h0) / safe
        inc = (d < 0) & ((f0 - fr > 0) | (fr - f1 < 0))
        flat = (d == 0) & (fr - f0 == 0)
        dec = (d > 0) & (f0 - fr >= 0) & (fr - f1 > 0)
        hit = inc | flat | dec
        val = torch.where(flat, h0.expand_as(fr), interp)
        take = hit & ~found
        out = torch.where(take, val, out)
        found = found | take
    return out


def _interp_fnsq(htr_t, fnsq_t, zi_t):
    """Plasma frequency squared at height ZI, by bracketing HTR.

    htr_t/fnsq_t are (N, 50) -- transposed so the table is the innermost
    dimension searchsorted needs.  zi_t is (N, F).  Returns (N, F).
    """
    h0col = htr_t[:, :1]
    hlast = htr_t[:, -1:]
    # ih = the first segment with ZI < HTR(ih+1); on an increasing table that
    # is exactly what the arithmetic IF at label 585/600 selects.
    idx = torch.searchsorted(htr_t.contiguous(), zi_t.contiguous(),
                             right=True) - 1
    idx = idx.clamp(0, NHT - 2)
    h0 = torch.gather(htr_t, 1, idx)
    h1 = torch.gather(htr_t, 1, idx + 1)
    f0 = torch.gather(fnsq_t, 1, idx)
    f1 = torch.gather(fnsq_t, 1, idx + 1)
    den = h1 - h0
    safe = torch.where(den == 0, torch.ones_like(den), den)
    out = torch.where(den == 0, f0, f0 + (zi_t - h0) * (f1 - f0) / safe)
    # the two ends: at or below HTR(1) the profile floor, at or above HTR(50)
    # the scan finds nothing and falls through to FNSQ(50)
    out = torch.where(zi_t - h0col <= 0, fnsq_t[:, :1].expand_as(out), out)
    return torch.where(zi_t >= hlast, fnsq_t[:, -1:].expand_as(out), out)


def gethp_batch(htr, fnsq, fxx):
    """Batched GETHP.

    htr/fnsq: (50, N) float32, the profile from ``lecden_batch``.
    fxx: (F, N) float32, vertical-sounding frequencies (MHz).
    Returns ``(hpx, htx)``, each (F, N) -- virtual and true height in km.
    """
    _require_monotone(htr)
    fr = fxx * fxx
    h1 = htr[0].unsqueeze(0)                      # HTR(1), the profile base

    ht = _true_height(htr, fnsq, fr) - h1
    hrmz = ht * _k(TWDIV, htr) * _k(XNPL, htr)

    htr_t = htr.transpose(0, 1)
    fnsq_t = fnsq.transpose(0, 1)
    twdiv = _k(TWDIV, htr)
    one = torch.ones_like(fr)
    hp = torch.zeros_like(fr)
    for ig in range(1, NT + 1):
        acc = torch.zeros_like(fr)
        for sign in (-1.0, +1.0):
            # ZG = 1 -+ XT(ig); with NPL == 1, ZG**NPL is ZG and ZG**(NPL-1)
            # is exactly 1, so the mu factor is just 1/sqrt(1-YSQ).
            zg = one + _k(sign * float(XT[ig]), htr)
            zi = ht * (one - twdiv * zg) + h1
            ysq = _interp_fnsq(htr_t, fnsq_t,
                               zi.transpose(0, 1)).transpose(0, 1) / fr
            ysq = torch.clamp(ysq, max=_YSQ_CAP)
            acc = acc + one / torch.sqrt(one - ysq)
        hp = hp + _k(float(WT[ig]), htr) * acc

    hpx = h1 + hrmz * hp
    htx = h1 + ht
    # Label 435: a frequency below the profile floor reflects at HTR(1).
    below = fr - fnsq[0].unsqueeze(0) <= 0
    base = h1.expand_as(fr)
    return torch.where(below, base, hpx), torch.where(below, base, htx)
