"""Stage B (PyTorch, batched) atmospheric-noise frequency law — GENFAM.

ANOIS1 gives Fam at 1 MHz for a local-time block; GENFAM carries it to the
operating frequency and supplies the spread statistics that go with it, all
out of the CCIR Report 322 tables REDMAP loads into /TWO/:

* ``FA``          -- Fam at FREQ, from the FAM(14,12) coefficients.  Two nested
  degree-6 Horner passes (PZ from FAM 1..7, PX from FAM 8..14): the first at
  the fixed abscissa -0.75 to invert the 1 MHz value back into the curve's own
  variable CZ, the second at U = (8*2**log10(FREQ) - 11)/4.
* ``DUA``/``DLA`` -- the upper and lower decile ratios,
* ``DUS``/``DLS`` -- their standard deviations,
* ``DMS``         -- sigma-Fam, the standard deviation of Fam itself,

the last five from quartics in log10(FREQ) held in DUD(5,12,5).

THE BATCHING IS TRIVIAL AND THE INDEXING IS NOT.  Every arithmetic step is a
multiply-accumulate over the lane axis, so there is not a single data-dependent
branch to select on -- but the column each lane reads is per-lane, because IBK
is the receiver's local-time block and the hemisphere shift is on the sign of
the receiver's latitude.  Both vary across a coverage grid.  So the whole port
is: resolve IBK to a column index per lane, gather, then run the same eleven
multiply-adds the scalar code runs.

TWO FREQUENCY CLAMPS, AND THEY ARE THE EASY THING TO GET WRONG.  Report 322's
decile curves stop at 20 MHz and sigma-Fam's stops at 10, so the abscissa is
pinned at log10(20) above 20 MHz for ALL FIVE outputs, and again at 1.0 above
10 MHz for DMS ALONE.  The scalar code does that by assigning to the loop's X
in two places, which reads as one clamp being a refinement of the other; here
they are two separate tensors, ``x20`` feeding outputs 1..4 and ``x5`` feeding
output 5, because a batched version cannot smuggle the state through the loop.
Getting them the wrong way round is a dB-sized error, not a ULP-sized one --
genfam.for's own 1993 rewrite comment records the previous version having them
index-confused and DUA/DLA/DUS/DLS wrong by up to +-4 dB at 20 MHz.

THE TABLES ARE SHARED ACROSS LANES IN PRODUCTION and per-lane in the test.  A
run has one month, so FAM/DUD are a single (14,12) / (5,12,5) table on the
device and only the column INDEX is per-lane.  The oracle dump spans all twelve
months, though, so ``_fam_col``/``_dud_col`` also accept a trailing lane axis;
that form exists solely so the dump replays as one batch and production never
builds it (14*12*N floats is not something to materialise over a grid).

float32.  ALOG10 and the ``2.**X`` power are the only transcendentals and both
carry the [[f32-libm-noise]] floor; both are evaluated in float64 and rounded
once to float32, which is what Stage A's ``log10_32``/``pow32`` do, so the two
stages agree bit-for-bit rather than merely closely.
"""

import numpy as np
import torch

from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
_F64 = torch.float64

_U0 = -0.75                     # the fixed first-pass abscissa
# ALOG10(20.) as gfortran folds it: the float32 nearest to log10(20).
_LOG10_20 = float(np.float32(np.log10(np.float64(np.float32(20.0)))))


def _log10_32(x):
    """REAL*4 ALOG10: evaluated in float64, rounded once to float32."""
    return torch.log10(x.to(_F64)).to(_F32)


def _pow32(base, exp):
    """REAL*4 ``base**exp`` with a REAL exponent -- gfortran emits powf.

    The exponent is ALREADY float32 here and must stay that way through the
    widening; that is the whole point of Stage A's ``pow32``.  Do not route
    ``x**2`` through this -- an INTEGER exponent is a multiply in gfortran.
    """
    return torch.pow(base.to(_F64), exp.to(_F64)).to(_F32)


def _fam_col(fam, j):
    """FAM's column `j` per lane -> (14, N).  See the module docstring for
    why the per-lane (14, 12, N) form is accepted at all."""
    if fam.dim() == 2:
        return fam[:, j]
    lane = torch.arange(j.shape[0], device=j.device)
    return fam[:, j, lane]


def _dud_col(dud, j):
    """DUD's column `j` per lane -> (5, 5, N), axes (coefficient, output, lane)."""
    if dud.dim() == 3:
        return dud[:, j, :].permute(0, 2, 1)
    lane = torch.arange(j.shape[0], device=j.device)
    # two advanced indices separated by a slice: the broadcast axis goes FIRST,
    # so this is (N, 5, 5) = (lane, coefficient, output) before the permute
    return dud[:, j, :, lane].permute(1, 2, 0)


def genfam_batch(fam, dud, y2, iblk, freq, z):
    """Batched GENFAM.

    fam: (14, 12) float32 /TWO/ FAM, or (14, 12, N) per-lane.
    dud: (5, 12, 5) float32 /TWO/ DUD, or (5, 12, 5, N) per-lane.
    y2:   (N,) receiver latitude in radians -- ONLY ITS SIGN is used.
    iblk: (N,) int64 local-time block 1..6.
    freq: (N,) float32 MHz.
    z:    (N,) float32 Fam at 1 MHz (ANOIS1's output).

    Returns ``(fa, dua, dla, dms, dus, dls)``, each (N,) float32 -- the
    Fortran's argument order, which is NOT the order the DUD loop produces
    them in.
    """
    # southern hemisphere -> blocks 7..12.  The sign alone decides; magnitude
    # never enters, so a lane exactly on the equator reads the northern column.
    ibk = iblk + 6 * (y2 < 0.0).to(iblk.dtype)
    j = ibk - 1
    famj = _fam_col(fam, j)
    dudj = _dud_col(dud, j)

    k2, k4, k8, k11 = (_k(2.0, freq), _k(4.0, freq),
                       _k(8.0, freq), _k(11.0, freq))

    def _pzpx(u1):
        """The paired Horner evaluations at abscissa U1 (label 110)."""
        pz = u1 * famj[0] + famj[1]
        px = u1 * famj[7] + famj[8]
        for i in range(3, 8):           # DO 115 I=3,7
            pz = u1 * pz + famj[i - 1]
            px = u1 * px + famj[i + 6]
        return pz, px

    x = _log10_32(freq)
    u = (k8 * _pow32(k2, x) - k11) / k4
    pz, px = _pzpx(_k(_U0, freq))       # KOP=1: recover the curve variable CZ
    cz = z * pz + px
    cz = z + z - cz
    pz, px = _pzpx(u)                   # KOP=2: evaluate at this frequency
    fa = cz * pz + px

    # Report 322's decile curves end at 20 MHz, sigma-Fam's at 10.  The scalar
    # code walks one X through the loop and reassigns it; batched, that is two
    # abscissae -- x20 for outputs 1..4, x5 for output 5.  Above 20 MHz both
    # clamps fire and DMS ends up at 1.0, which is what the scalar order gives.
    x20 = torch.where(freq > 20.0, _k(_LOG10_20, freq), x)
    x5 = torch.where(freq > 10.0, _k(1.0, freq), x20)

    v = []
    for i in range(1, 6):               # DO 145
        xi = x5 if i == 5 else x20
        y = dudj[0, i - 1]
        for jj in range(2, 6):          # DO 140
            y = y * xi + dudj[jj - 1, i - 1]
        v.append(y)
    dua, dla, dus, dls, dms = v
    return fa, dua, dla, dms, dus, dls
