"""Stage B (PyTorch, batched) 1 MHz atmospheric noise at the RX — ANOIS1.

The CCIR atmospheric-noise atlas is stored as six maps, one per 4-hour block
of LOCAL time.  ANOIS1's whole job is to decide WHICH TWO of those six apply
at the receiver: KJ is the block the receiver mean time CC falls in, and JK is
the neighbouring block on the side CC leans toward, wrapped cyclically (block
0 is block 6, block 7 is block 1).  It then evaluates the noise map at the
receiver latitude/longitude for both, giving the primary ATNU and the
secondary ATNY; GENOIS interpolates between them.

TWO DIFFERENT KINDS OF ARITHMETIC LIVE HERE, and they get different treatment.

*The block selection is integer.*  KJ, TM and JK come out of truncation and
three-way comparisons, and there is nothing approximate about them -- a lane
that lands in the wrong block is wrong by a whole map, not by an ULP.  The
tests assert those BITWISE.  The three-way IF(CC-TM) keeps its middle arm:
CC == TM exactly does happen (the dump has lanes at 2.0, 6.0, 10.0, 14.0,
18.0 hours, sitting on a block centre), and that arm chooses JK = KJ, i.e.
the secondary map is the primary one and GENOIS's interpolation collapses.
Folding it into either neighbour would be wrong on exactly those lanes.

*The map values carry the SIN/COS floor* of the Fourier series -- see
``noisy.py``.

WHY ALL SIX BLOCKS ARE EVALUATED.  Only two are needed per lane, but KJ and
JK are per-lane, so the alternative is gathering a per-lane copy of the
coefficient map.  That map is (29, 16) and the gather would materialise
(29, 16, N) floats -- 1.8 GB at a million grid points -- to save 4/6 of a few
thousand elementwise ops.  The usual case is a map SHARED by the entire grid
(REDMAP depends only on month and SSN), so evaluating all six and selecting
the two results keeps the storage at O(N) and the code at one path.  It also
means ATNY costs nothing beyond ATNU.

THE NO-DATA-BASE EARLY RETURN.  ``IF(F2D(1,1,1)) 90,90,100`` bails out with
the noise zeroed when no long-term ionospheric data base was loaded; GENOIS
recognises the zero and substitutes -204 dB.  Batched, that is a select on
ATNU/ATNY.  CC/TM/KJ/JK are still RETURNED COMPUTED on those lanes, where the
Fortran leaves them holding whatever the previous call left in the COMMON --
a stale value this port could not reproduce even if it wanted to, and one no
caller reads (the Stage A oracle test skips those four fields for exactly
that reason).  The dump does reach this arm: 9 of its 189 cases.

WHAT THIS IS KEYED BY: (RECEIVER SITE, RECEIVER MEAN TIME).  Nothing else.
Read the signature and there is no transmitter in it and no frequency in it --
GMTR, RLONG, RLONGD and RLAT are the receiver's, and FAKP/FAKABP/F2D111 are
properties of the month and the coefficient set.  The Fortran agrees: LUFFY
calls ANOIS1 once, before the frequency loop opens (luffy.for:104), and the
only path-dependent thing computed before it -- SELMOD's controlling sample
area -- is not among its inputs.

That matters for an area run, where the "receiver" is the grid point.  A
naive driver evaluates one lane per (circuit, hour), and with 100
transmitters over the same grid, 99 of every 100 lanes recompute an answer
bit-for-bit identical to one already in flight -- six 435-term Fourier sums
each.  The honest key is (grid point, hour), so::

    from voacap_gpu.batched.dedupe import (compose, unique_inverse,
                                           representatives, take, expand)

    keys = compose(grid_index, hour_index)       # both (N,) int64
    uniq, inv = unique_inverse(keys)
    rep = representatives(inv, len(uniq))        # (U,) lanes to actually run

    args = take({"gmtr": gmtr, "rlong": rlong, "rlongd": rlongd,
                 "rlat": rlat, "f2d111": f2d111}, rep, n_lanes=len(inv))
    compact = anois1_batch(fakp, fakabp, args["f2d111"], args["gmtr"],
                           args["rlong"], args["rlongd"], args["rlat"])
    out = expand(compact, inv)                   # back to N lanes, exactly

``expand`` is an index read, so ``out`` is bit-identical to calling
``anois1_batch`` on all N lanes -- see ``tests/batched/test_dedupe.py``, which
asserts that on this routine's own oracle inputs.  FAKP/FAKABP go through
unchanged in the shared (29,16,6) form; in the per-lane (29,16,6,N) form they
need ``take(..., dim=-1)`` as well.

GENOIS IS NOT SHAREABLE THE SAME WAY, despite also being transmitter-free in
appearance -- see the matching section in ``genois.py`` for where the line
falls and why.

State: the Fortran writes COMMON /ANOIS/.  This returns a dict instead, the
Stage B convention (see ``findf_batch``).

float32, mirroring Stage A.
"""

import torch

from voacap_gpu.scalar.constants import R2D
from voacap_gpu.batched.noisy import LIMITS_DEFAULT, noisy_batch
from voacap_gpu.batched.profile import _k

_F32 = torch.float32
_I64 = torch.int64

NBLOCK = 6                      # six 4-hour local-time noise blocks


def anois1_batch(fakp, fakabp, f2d111, gmtr, rlong, rlongd, rlat):
    """Batched ANOIS1.

    fakp:   atmospheric-noise coefficient maps, ``fakp[j, k, b]`` for Fortran
            P(J+1, K+1, B+1), b = 0..5.  Shape (29, 16, 6) for a map shared by
            the whole batch, or (29, 16, 6, N) when it varies per lane.
    fakabp: ABP normalisation, ``fakabp[i, b]``, i = 0/1 for ALF/BET.  Shape
            (2, 6) or (2, 6, N).
    f2d111: F2D(1,1,1), the long-term-data-base flag.  Scalar tensor or (N,);
            <= 0 zeroes the noise.
    gmtr:   receiver mean (local) time, hours.  (N,)
    rlong:  receiver longitude, RADIANS, signed -- the sign is what selects
            the east/west branch, so it is not redundant with rlongd.
    rlongd: receiver east longitude, DEGREES, used when rlong >= 0.
    rlat:   receiver latitude, radians.

    Returns a dict with atnu, atny, cc, tm (float32 (N,)) and kj, jk (int64
    (N,), values 1..6 -- Fortran block numbers, not indices).
    """
    cc = gmtr.to(_F32)

    # ---- KJ: the block CC falls in ----------------------------------------
    # KJ=CC/4.+1. into an INTEGER, i.e. truncation toward zero.  CC >= 22
    # would give 6.5 and truncate to 6 anyway, but the Fortran short-circuits
    # it with a literal 6, so both arms exist; the guarded arm is evaluated
    # unconditionally and is finite everywhere (CC is a clock time).
    kj_calc = torch.trunc(cc / _k(4.0, cc) + _k(1.0, cc)).to(_I64)
    kj = torch.where(cc - _k(22.0, cc) < 0.0,        # IF(CC-22.) 105,110,110
                     kj_calc, torch.full_like(kj_calc, NBLOCK))

    # ---- TM: the centre time of block KJ, and JK: the block CC leans to ----
    tm = (4 * kj - 2).to(_F32)          # TM=4*KJ-2, integer then promoted
    d = cc - tm                         # IF(CC-TM) 115,120,125
    jk = torch.where(d < 0.0, kj - 1,
                     torch.where(d == 0.0, kj, kj + 1))
    # cyclic wrap: IF(JK) 135,135,140 then IF(JK-6) 150,150,145
    jk = torch.where(jk <= 0, torch.full_like(jk, NBLOCK), jk)
    jk = torch.where(jk - NBLOCK > 0, torch.ones_like(jk), jk)

    # ---- receiver position in the map's coordinates ------------------------
    # IF(RLONG) 155,160,160.  West longitudes are converted from radians and
    # folded into 0..360; east ones are taken from the degrees argument
    # directly, which is not the same rounding -- deliberately, it is what the
    # Fortran does.
    r2d = _k(float(R2D), cc)
    ceg = torch.where(rlong < 0.0,
                      _k(360.0, cc) + rlong.to(_F32) * r2d,
                      rlongd.to(_F32))
    xla = rlat.to(_F32) * r2d

    # ---- all six blocks, then select KJ and JK -----------------------------
    lm, ln = LIMITS_DEFAULT             # KJ 1..6 is never the KJ=8 map
    anos = [noisy_batch(fakp[:, :, b], fakabp[0, b], fakabp[1, b],
                        xla, ceg, lm, ln) for b in range(NBLOCK)]

    zero = torch.zeros_like(cc)
    atnu = zero
    atny = zero
    for b in range(NBLOCK):             # pure selects -- bit-exact
        atnu = torch.where(kj == b + 1, anos[b], atnu)
        atny = torch.where(jk == b + 1, anos[b], atny)

    # ---- 90: no long-term data base -> zero the noise ----------------------
    has_base = (f2d111.to(_F32) > 0.0)
    atnu = torch.where(has_base, atnu, zero)
    atny = torch.where(has_base, atny, zero)

    return {"atnu": atnu, "atny": atny, "cc": cc, "tm": tm,
            "kj": kj, "jk": jk}
