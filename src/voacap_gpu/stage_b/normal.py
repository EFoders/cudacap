"""Stage B (PyTorch, batched) normal distribution — FNORML and PRBMUF.

Two spellings of the same Hastings quartic (AMS 55 26.2.17), kept apart
because the Fortran keeps them apart: PRBMUF INLINES the polynomial rather
than calling FNORML, and prbmuf.for says so in a comment.  Merging them here
would be a simplification of the source, not of the port -- and the two are
validated against different dumps.

WHY ``1 - fnorml(z)`` IS SPELLED OUT BY CALLERS and not folded in: for Z >= 0
the Fortran computes ``PX = 1 - QX`` and the caller then computes ``1 - PX``,
so the value round-trips through two float32 subtractions and does not in
general come back to QX.  Shortcutting it changes the last bit.

Both are elementwise, so batching is the identity transformation on the
arithmetic.  float32 throughout; no libm at all, only the polynomial, so both
are bit-exact against Stage A on either device.
"""

import numpy as np
import torch

from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
_C = (0.196854, 0.115194, 0.000344, 0.019527)
_CLAMP = 5.0
_SIG_FLOOR = float(np.float32(0.001))


def _hastings(yp, k):
    """0.5 / (1 + yp*(c1 + yp*(c2 + yp*(c3 + yp*c4))))**4, folded to |yp|."""
    qx = k["one"] + yp * (k["c1"] + yp * (k["c2"] + yp * (k["c3"]
                                                         + yp * k["c4"])))
    qx = qx * qx * qx * qx
    return k["half"] * (k["one"] / qx)


def _consts(like):
    return {"one": _k(1.0, like), "half": _k(0.5, like),
            "c1": _k(_C[0], like), "c2": _k(_C[1], like),
            "c3": _k(_C[2], like), "c4": _k(_C[3], like)}


def fnorml_batch(ypx):
    """P(X <= ypx) for a standard normal X.  ypx: (N,) -> (N,)."""
    k = _consts(ypx)
    neg = ypx < 0.0
    yp = torch.clamp(torch.where(neg, -ypx, ypx), max=_CLAMP)
    qx = _hastings(yp, k)
    return torch.where(neg, qx, k["one"] - qx)


def prbmuf_batch(fmhz, fgo, fset, sigl, sigu):
    """Batched PRBMUF: P(the operating frequency exceeds this layer's MUF).

    fmhz: (N,) operating frequency.  fgo: (N,) the layer's median frequency.
    fset: (N,) the frequency the deciles are relative to; <= 0 means no median
    was placed and the result degrades to a step at FGO.
    sigl/sigu: (N,) the layer's lower and upper decile spreads, ALREADY
    selected from /MUFS/ -- the caller knows which layer it is asking about,
    and passing the row instead of the table keeps this free of the layer
    indexing that differs between SIGDIS and REGMOD.
    """
    k = _consts(fmhz)
    z = fmhz - fgo
    sig = torch.clamp(fgo * torch.where(z <= 0.0, sigl, sigu) / fset,
                      min=_SIG_FLOOR)
    zs = z / sig
    neg = zs < 0.0
    yp = torch.clamp(torch.where(neg, -zs, zs), max=_CLAMP)
    qx = _hastings(yp, k)
    qx = torch.where(neg, k["one"] - qx, qx)
    # FSET <= 0: no median placed, so the distribution collapses to a step.
    # SIG is computed unconditionally above and divides by FSET, which is zero
    # on exactly those lanes -- the result is discarded here, never read.
    step = torch.where(z <= 0.0, k["one"], torch.zeros_like(qx))
    return torch.where(fset > 0.0, qx, step)
