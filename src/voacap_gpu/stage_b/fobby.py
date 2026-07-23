"""Stage B (PyTorch, batched) reflectrix table — FOBBY.

For each take-off angle and each of the thirty vertical-sounding frequencies,
the secant law gives the oblique operating frequency that reflects at that
angle.  FINDF later searches the resulting table to place every mode, so this
is the bridge from "what the ionosphere looks like straight up" to "what it
does for a path".

Pure elementwise given the ionogram: no search, no branch except a clamp, so
it batches over (angles x frequencies x grid points) with nothing to select.

**THE OUTPUT IS TRUNCATED TO INTEGER kHz**, and that quantisation is the
single largest amplifier of float32 noise in the whole model.  The end-to-end
replay measured it: about 1 in 400 entries has ``1000*FREQ`` within 0.04 of an
integer, where a 1-ULP disagreement in COS tips the truncation by a whole kHz,
and FINDF's interpolation then turns that into ~1e-3 relative error in the ray
table.  Forcing IFOB to the oracle's values made FINDF bit-exact across 720
calls.  Nothing here can fix that -- it is inherent to truncating a
transcendental -- but it is why the test below checks the kHz integers rather
than the float frequencies, and why a 1-ULP cos difference is worth more
attention in this routine than anywhere else.

True height, not virtual, so there is no ionospheric K factor: the secant law
is applied at the height the ray actually reflects from.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.constants import D2R, RZ
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
_MIN_SQCOS = float(np.float32(0.000001))
_KHZ = float(np.float32(1000.0))


def fobby_batch(ang, fvert, htrue, return_khz_float=False):
    """Batched FOBBY.

    ang: (A,) take-off angles in degrees -- the restricted scan SANG produced.
    fvert/htrue: (F, N) from ``genion_batch``.
    Returns IFOB, an int32 (A, F, N) tensor of oblique frequencies in kHz.

    With ``return_khz_float`` it also returns the pre-truncation ``1000*FREQ``.
    That is what makes the quantisation testable: an entry that disagrees with
    another implementation by a kHz should have a fractional part sitting
    against 0 or 1, and if it does not, the arithmetic is wrong rather than
    merely rounded differently.
    """
    rz = _k(RZ, fvert)
    delta = ang.to(_F32) * _k(D2R, fvert)
    rcosd = (rz * torch.cos(delta)).reshape(-1, 1, 1)      # (A,1,1)

    sphe = rcosd / (rz + htrue).unsqueeze(0)               # (A,F,N)
    sqcos = torch.clamp(1.0 - sphe * sphe, min=_MIN_SQCOS)
    freq = fvert.unsqueeze(0) / torch.sqrt(sqcos)
    khz = _k(_KHZ, fvert) * freq
    # REAL->INTEGER in Fortran truncates toward zero; trunc, not floor.
    ifob = torch.trunc(khz).to(torch.int32)
    return (ifob, khz) if return_khz_float else ifob
