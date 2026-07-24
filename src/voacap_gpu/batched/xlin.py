"""Stage B (PyTorch, batched) table interpolation — XLIN.

Piecewise-linear interpolation of YN against XN, tolerant of a table that is
not monotone: it walks left to right and takes the FIRST segment that brackets
X, handling increasing, flat and decreasing runs separately, and clamps to the
endpoints outside the table.  SIGDIS and INMUF both use it to read a height off
the ionogram at a given frequency, where the F-region valley makes the table
genuinely non-monotone -- which is why it cannot become a ``searchsorted`` the
way GETHP's node lookup could.

FIRST-MATCH BECOMES A DONE MASK.  The scalar routine returns out of the loop;
the batched one runs all NTAB-1 segments for every lane and keeps the first
that hits.  That is pure waste on lanes that matched early -- but NTAB is 30,
the body is four elementwise kernels, and the alternative is lockstep for a
loop whose trip count is a compile-time constant.  Not worth it.

float32, no transcendentals, so bit-exact against Stage A on both devices.
"""

import torch

_F32 = torch.float32


def xlin_batch(x, xn, yn, ntab=None):
    """Batched XLIN.

    x: (N,) the abscissa to look up.
    xn/yn: (T, N) the table, Fortran index 1..NTAB as rows 0..T-1.
    ntab: how much of the table is live; defaults to all of it.

    Returns (N,).  Outside the table the endpoints are held flat, which is the
    Fortran's behaviour and not an extrapolation.
    """
    if ntab is None:
        ntab = xn.shape[0]
    if ntab < 2 or ntab > xn.shape[0]:
        raise ValueError(f"ntab={ntab} outside the table's {xn.shape[0]} rows")

    # IF(XN(1)-X) 3,3,2 -- above the first entry, hold YN(1)
    done = xn[0] - x > 0.0
    out = torch.where(done, yn[0], torch.zeros_like(x))

    for i in range(ntab - 1):               # Fortran J = 1..NTAB-1, I = J
        lo, hi = xn[i], xn[i + 1]
        d = lo - hi
        denom = hi - lo
        # only read on segments where it is nonzero; a flat segment takes
        # YN(I) instead, so the guard changes no answer
        safe = torch.where(denom == 0.0, torch.ones_like(denom), denom)
        interp = yn[i] + (x - lo) * (yn[i + 1] - yn[i]) / safe

        rising = (d < 0.0) & ((lo - x > 0.0) | (x - hi < 0.0))
        flat = (d == 0.0) & (x - lo == 0.0)
        falling = (d > 0.0) & (lo - x >= 0.0) & (x - hi > 0.0)

        hit = ~done & (rising | flat | falling)
        out = torch.where(hit, torch.where(flat, yn[i], interp), out)
        done = done | hit

    # XLIN = YN(ISMAX): past the end of the table, hold the last entry
    return torch.where(done, out, yn[ntab - 1])
