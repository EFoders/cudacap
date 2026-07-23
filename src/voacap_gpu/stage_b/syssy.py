"""Stage B (PyTorch, batched) excess-system-loss table lookup — SYSSY.

The CCIR SYS(9,16,6) and PERR(9,4,6) tables say how much loss the model is
missing and how wrong it is likely to be, as a function of where and when the
ray bounces: geomagnetic latitude in nine 5-degree bands from 40 degrees up,
local time in eight three-hour blocks (doubled to sixteen, southern hemisphere
in the upper half), and a short/long distance code.  SYSSY turns (G, PT, NN)
into three SYS cells and three PERR cells.  There is no arithmetic in it worth
the name: everything after the index computation is a table read and two
divides by 1.28.

**THE WHOLE ROUTINE IS AN INDEX COMPUTATION, SO THAT IS WHAT BATCHES.**  KJ,
LOL and LD are float32 expressions truncated to integers, and the truncation
is the part that has to be right -- a boundary case that lands one cell over
picks a different decile, not a slightly different number.  The port keeps the
expressions in float32 (via ``_k``, so a CUDA divide by 5.0 rounds like a
float32 divide and not like a double one) and truncates with a cast, which is
Fortran's round-toward-zero.  The three arithmetic-IF clamps become
``torch.where``.

**THE TABLES CAN BE SHARED OR PER-LANE.**  In a real run every grid point is
the same month, so SYS/PERR is one (9, 16, 6) tensor and the lookup is a plain
advanced index.  The oracle dump sweeps all twelve months, so the test wants a
different table per lane; passing (9, 16, 6, N) switches the lookup to a
gather along the flattened table axis.  Both spellings read the same flat
offset, so there is only one index computation either way.

The no-data fallback (F2D(1,1,1) <= 0, the ITSA-1 constants) is a per-run
condition rather than a per-grid-point one, but it is cheap to carry per lane
and the dump exercises it, so it is a mask like everything else.  **Note its
SL divides by 1.29 where the table branch divides by 1.28** -- that asymmetry
is in syssy.for and is preserved, not a typo here.

No transcendentals anywhere, so this is bit-exact against the oracle on both
devices; anything else is a bug in the indexing rather than a libm floor.
"""

import torch

from voacap_gpu.stage_b.lockstep import gather_rows
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32

_NKJ, _NLOL, _NNN = 9, 16, 6        # SYS(9,16,6)
_NLD, _NND = 4, 6                   # PERR(9,4,6)
_SYS_CELLS = _NKJ * _NLOL * _NNN
_PERR_CELLS = _NKJ * _NLD * _NND

# SYSSY's own coarse degrees-per-radian (5.729577E1), NOT the /CON/ R2D.
_RAD2DEG = 5.729577e1


def _flatten(table, cells):
    """(a, b, c) or (a, b, c, N) -> (cells,) or (cells, N), C-order."""
    if table.dim() == 3:
        return table.reshape(cells)
    if table.dim() == 4:
        return table.reshape(cells, table.shape[3])
    raise ValueError(f"table must be 3-D (shared) or 4-D (per lane), "
                     f"got shape {tuple(table.shape)}")


def _pick(flat, idx):
    """One cell per lane: shared (cells,) or per-lane (cells, N) table."""
    return flat[idx] if flat.dim() == 1 else gather_rows(flat, idx)


def syssy_batch(sys, perr, has_data, g, pt, nn):
    """Batched SYSSY.

    sys: (9, 16, 6) or (9, 16, 6, N) float32 -- the CCIR excess-system-loss
        table, Fortran SYS(KJ, LOL, NN) as [kj-1, lol-1, nn-1].
    perr: (9, 4, 6) or (9, 4, 6, N) -- the prediction-error table.
    has_data: (N,) bool or a Python bool -- ``F2D(1,1,1) > 0``, i.e. a
        coefficient set was loaded.  False selects the ITSA-1 constants.
    g: (N,) geomagnetic latitude of the reflection area, RADIANS.  Its SIGN
        picks the hemisphere half of the table.
    pt: (N,) reflection-area local time, hours.
    nn: (N,) long (or a Python int) -- the distance code, 2 short / 5 long.

    Returns ``(fm, su, sl, fmp, sup, slp)``, each (N,) float32.
    """
    n = g.shape[0]
    dev = g.device
    if not torch.is_tensor(nn):
        nn = torch.full((n,), int(nn), dtype=torch.long, device=dev)
    nn = nn.to(torch.long)
    if not torch.is_tensor(has_data):
        has_data = torch.full((n,), bool(has_data), dtype=torch.bool,
                              device=dev)

    # ---- the index arithmetic (labels 95..150) ---------------------------
    # J = PT + .5, a REAL-to-INTEGER assignment: truncation toward zero, which
    # is what the cast does.  PT is a local time in [0, 24), so J is in [0, 24]
    # and every downstream index lands inside its table (see the clamps below).
    j = (pt + _k(0.5, pt)).to(torch.long)
    jf = j.to(_F32)

    gg = torch.abs(g) * _k(_RAD2DEG, g)
    # KJ = ((GG-40.)/5.) + 1.5 -- float32 all the way, then truncate.  Below
    # 32.5 degrees the expression is negative and the cast truncates toward
    # zero rather than flooring, but both land at or below 0 and the clamp
    # takes them to 1, so the distinction cannot change an answer.
    kj = ((gg - _k(40.0, g)) / _k(5.0, g) + _k(1.5, g)).to(torch.long)
    kj = torch.clamp(kj, 1, _NKJ)                   # labels 100 and 110

    south = g < 0.0                                 # IF(G) 125,120,120
    zl = torch.zeros_like(kj)
    kk = torch.where(south, zl + 8, zl)
    nd = torch.where(south, zl + 3, zl)

    # LJ = FLOAT(J)/3. + .67, then LJ==0 wraps to the 21-24h block (label 135).
    lj = (jf / _k(3.0, g) + _k(0.67, g)).to(torch.long)
    lj = torch.where(lj < 1, torch.full_like(lj, 8), lj)
    lol = lj + kk
    # LD = FLOAT(J)/6. + .67, same wrap (label 145).
    ld = (jf / _k(6.0, g) + _k(0.67, g)).to(torch.long)
    ld = torch.where(ld < 1, torch.full_like(ld, _NLD), ld)

    # Index safety only.  For PT in [0, 24.5) LJ is in 1..8 and KK in {0,8},
    # so LOL is already in 1..16 and LD already in 1..4; the Fortran would
    # simply run off the end of /TWO/ outside that range, and clamping is a
    # kinder way to be wrong than a torch indexing error.
    lol = torch.clamp(lol, 1, _NLOL)
    ld = torch.clamp(ld, 1, _NLD)

    # ---- the table reads (label 150) -------------------------------------
    sys_f = _flatten(sys, _SYS_CELLS)
    perr_f = _flatten(perr, _PERR_CELLS)
    base_s = (kj - 1) * (_NLOL * _NNN) + (lol - 1) * _NNN + (nn - 1)
    base_p = (kj - 1) * (_NLD * _NND) + (ld - 1) * _NND + nd

    fm_t = _pick(sys_f, base_s)                     # SYS(KJ,LOL,NN)
    du_t = _pick(sys_f, base_s + 1)                 # SYS(KJ,LOL,NN+1)
    dl_t = _pick(sys_f, base_s - 1)                 # SYS(KJ,LOL,NN-1)
    fmp_t = _pick(perr_f, base_p)                   # PERR(KJ,LD,ND+1)
    sup_t = _pick(perr_f, base_p + 1)               # PERR(KJ,LD,ND+2)
    slp_t = _pick(perr_f, base_p + 2)               # PERR(KJ,LD,ND+3)
    dcl = _k(1.28, g)
    su_t = du_t / dcl
    sl_t = dl_t / dcl

    # ---- the no-data fallback (label 90) ---------------------------------
    # DL/1.29, not DL/1.28.  syssy.for really does use a different divisor in
    # this arm; it changes SL by ~0.8% and only ever runs with no coefficient
    # file loaded, which is why nobody has ever noticed.
    fm_0 = _k(9.0, g)
    su_0 = _k(9.0, g) / dcl
    sl_0 = _k(4.0, g) / _k(1.29, g)
    fmp_0, sup_0, slp_0 = _k(5.6, g), _k(0.60, g), _k(1.63, g)

    def sel(tab, fixed):
        return torch.where(has_data, tab, fixed.expand(n))

    return (sel(fm_t, fm_0), sel(su_t, su_0), sel(sl_t, sl_0),
            sel(fmp_t, fmp_0), sel(sup_t, sup_0), sel(slp_t, slp_0))
