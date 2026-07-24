"""Stage B (PyTorch, batched) port of DAZEL.

Element-wise identical to the Stage A transliteration
(src/voacap_gpu/scalar/dazel.py) up to libm rounding of float64
transcendentals, which the float32 interface narrowing absorbs — the same
argument (and the same 1-ULP/98%-bit-exact test bar) used to validate
Stage A against the gfortran oracle dump.

Vectorization notes, in Stage A order:
- The DELON ±360 normalization loops (labels 12/16/18) become one masked
  while-loop.  Every ±360.0 step is exact in float64 for these operands, and
  an element that starts below -180 can never overshoot above +180 (nor vice
  versa), so the combined loop reproduces the per-element Fortran loops.
- The three mode-0 branches (same point / same longitude / general) are
  computed on the whole batch and combined with torch.where.  Unselected
  lanes may produce inf/NaN (e.g. the general path divides by sgc = 0 on a
  same-longitude lane); torch.where discards them without arithmetic.
- The same-point branch (labels 24-35) RETURNs before the elevation block
  and does not write ZTAKOF/ZRAKOF, so those lanes keep their input values.
- Mode 1 writes zrlat/zrlon/ztelv/zrelv/ztakof/zrakof only; zd and zdgc are
  untouched (zdgc is a mode-1 input).

All arithmetic is done in float64 tensors (matching IMPLICIT DOUBLE
PRECISION(A-L,N-Y)) and narrowed to float32 exactly at the Fortran
assignment sites.  Python-float constants are exact float64; constants that
the Fortran writes as REAL*4 literals are pre-widened through float32.
Multiplication order follows Stage A (left-to-right) so rounding sequences
match lane-for-lane.
"""

from dataclasses import dataclass, fields

import numpy as np
import torch

# DATA statements (dazel.for:53-54); exact float64 values.
PI = 3.141592653589793238462643
RERTH = 6370.0
DTOR = 0.01745329252
RTOD = 57.29577951

# REAL*4 literals widened through float32 (as Fortran mixed-mode does).
_C1E5 = float(np.float32(1.0e-5))
_C89_9 = float(np.float32(-89.9))
_C89_9P = float(np.float32(89.9))


@dataclass
class AzelBatch:
    """COMMON/AZEL/ as float32 tensors of shape (n,), one lane per case."""

    ztlat: torch.Tensor
    ztlon: torch.Tensor
    ztht: torch.Tensor
    zrlat: torch.Tensor
    zrlon: torch.Tensor
    zrht: torch.Tensor
    ztaz: torch.Tensor
    zraz: torch.Tensor
    ztelv: torch.Tensor
    zrelv: torch.Tensor
    zd: torch.Tensor
    zdgc: torch.Tensor
    ztakof: torch.Tensor
    zrakof: torch.Tensor

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v.dtype != torch.float32:
                raise TypeError(f"{f.name} must be float32, got {v.dtype}")

    @classmethod
    def zeros(cls, n: int, device=None) -> "AzelBatch":
        return cls(**{
            f.name: torch.zeros(n, dtype=torch.float32, device=device)
            for f in fields(cls)
        })


def dazel(z: AzelBatch, mode: int) -> None:
    """Run DAZEL over every lane; mode is uniform for the batch."""
    if mode == 1:
        _mode1(z)
        return
    _mode0(z)


def _norm_delon(delon: torch.Tensor) -> torch.Tensor:
    # Labels 12/16/18: normalize into [-180, 180] by exact ±360 steps.
    while True:
        lo = delon < -180.0
        hi = delon > 180.0
        if not bool((lo | hi).any()):
            return delon
        delon = torch.where(
            lo, delon + 360.0, torch.where(hi, delon - 360.0, delon)
        )


def _mode0(z: AzelBatch) -> None:
    # 56-65: load interface into doubles; clamp latitudes off the poles
    tlats = z.ztlat.double()
    tlons = z.ztlon.double()
    thts = (z.ztht * 1.0e-3).double()  # REAL*4 multiply, then widen
    rlats = z.zrlat.double()
    rlons = z.zrlon.double()
    rhts = (z.zrht * 1.0e-3).double()
    tlats = torch.where(tlats <= -90.0, _C89_9, tlats)
    tlats = torch.where(tlats >= 90.0, _C89_9P, tlats)
    rlats = torch.where(rlats <= -90.0, _C89_9, rlats)
    rlats = torch.where(rlats >= 90.0, _C89_9P, rlats)

    # 66-75
    delat = rlats - tlats
    adlat = delat.abs()
    delon = _norm_delon(rlons - tlons)
    adlon = delon.abs()
    delht = rhts - thts

    near_lon = (adlon - _C1E5) <= 0.0            # 77
    same_pt = near_lon & ((adlat - _C1E5) <= 0.0)  # 78 → labels 24-35
    dneg = delht < 0.0
    dzero = delht == 0.0

    # ---- 40-50: same longitude, distinct latitudes
    z180 = torch.full_like(delat, 180.0)
    z0 = torch.zeros_like(delat)
    taz_sl = torch.where(delat <= 0.0, z180, z0)   # 42/45
    raz_sl = torch.where(delat <= 0.0, z0, z180)
    gc_sl = adlat * DTOR                            # 50
    sgc_sl = torch.sin(0.5 * gc_sl)
    d_sl = torch.sqrt(
        delht * delht
        + 4.0 * (RERTH + thts) * (RERTH + rhts) * sgc_sl * sgc_sl
    )

    # ---- 55-65: distinct longitudes (general path)
    west = delon <= 0.0                             # 56/60
    wlat = torch.where(west, rlats, tlats) * DTOR
    elat = torch.where(west, tlats, rlats) * DTOR
    # 65: azimuths at W and E
    sdlat = torch.sin(0.5 * adlat * DTOR)
    sdlon = torch.sin(0.5 * adlon * DTOR)
    sadln = torch.sin(adlon * DTOR)
    cwlat = torch.cos(wlat)
    celat = torch.cos(elat)
    p = 2.0 * (sdlat * sdlat + sdlon * sdlon * cwlat * celat)
    sgc_g = torch.sqrt(p * (2.0 - p))
    sdlat = torch.sin(elat - wlat)  # reused as sin(delta-lat), per Fortran
    cwaz = (2.0 * celat * torch.sin(wlat) * sdlon * sdlon + sdlat) / sgc_g
    swaz = sadln * celat / sgc_g
    waz = torch.atan2(swaz, cwaz) * RTOD
    ceaz = (2.0 * cwlat * torch.sin(elat) * sdlon * sdlon - sdlat) / sgc_g
    seaz = sadln * cwlat / sgc_g
    eaz = torch.atan2(seaz, ceaz) * RTOD
    eaz = 360.0 - eaz
    taz_g = torch.where(west, eaz, waz)             # 141-147
    raz_g = torch.where(west, waz, eaz)
    # 152-155: straight-line distance and great-circle angle
    d_g = torch.sqrt(
        delht * delht + 2.0 * (RERTH + thts) * (RERTH + rhts) * p
    )
    cgc = 1.0 - p
    gc_g = torch.atan2(sgc_g, cgc)

    # ---- branch merge, then the shared elevation block
    gc = torch.where(near_lon, gc_sl, gc_g)
    d = torch.where(near_lon, d_sl, d_g)
    telv, relv, tak_t, tak_r = _elevations(gc, d, delht, thts, rhts)

    # ---- 24-35: same-point early-return values
    n90 = torch.full_like(delat, -90.0)
    p90 = torch.full_like(delat, 90.0)
    telv_sp = torch.where(dneg, n90, torch.where(dzero, z0, p90))
    relv_sp = torch.where(dneg, p90, torch.where(dzero, z0, n90))

    # ---- assign outputs (narrow to REAL*4 at the assignment sites)
    z.ztaz = torch.where(same_pt, z0, torch.where(near_lon, taz_sl, taz_g)).float()
    z.zraz = torch.where(same_pt, z0, torch.where(near_lon, raz_sl, raz_g)).float()
    z.zd = torch.where(same_pt, delht.abs(), d).float()
    z.zdgc = torch.where(same_pt, z0, gc * RERTH).float()  # 140
    z.ztelv = torch.where(same_pt, telv_sp, telv).float()
    z.zrelv = torch.where(same_pt, relv_sp, relv).float()
    # same-point lanes RETURN before 171-183: takeoff fields keep inputs
    z.ztakof = torch.where(same_pt, z.ztakof, tak_t.float())
    z.zrakof = torch.where(same_pt, z.zrakof, tak_r.float())


def _elevations(gc, d, delht, thts, rhts):
    """Labels 142-194 (minus the zdgc store): returns float64
    (telv, relv, takof_t, takof_r) already ordered by the delht sign."""
    # 142-150: order end points by height
    pos = delht >= 0.0  # 145
    aht = torch.where(pos, rhts, thts)
    bht = torch.where(pos, thts, rhts)
    saelv = (
        0.5
        * (d * d + delht.abs() * (RERTH + aht + RERTH + bht))
        / (d * (RERTH + aht))
    )
    arg = torch.clamp(1.0 - saelv * saelv, min=0.0)
    aelv = torch.atan2(saelv, torch.sqrt(arg))
    belv = (aelv - gc) * RTOD
    aelv = -aelv * RTOD
    # 171-183: take-off angles for 4/3 earth radius
    r4thd = RERTH * 4.0 / 3.0
    gc = 0.75 * gc
    sgc = torch.sin(0.5 * gc)
    p = 2.0 * sgc * sgc
    aalt = r4thd + aht
    balt = r4thd + bht
    da = torch.sqrt(delht * delht + 2.0 * aalt * balt * p)
    saelv = 0.5 * (da * da + delht.abs() * (aalt + balt)) / (da * aalt)
    arg = torch.clamp(1.0 - saelv * saelv, min=0.0)
    atakof = torch.atan(saelv / torch.sqrt(arg))
    btakof = (atakof - gc) * RTOD
    atakof = -atakof * RTOD
    # 184-194: order outputs by delht sign (151/155)
    neg = delht < 0.0
    telv = torch.where(neg, aelv, belv)
    relv = torch.where(neg, belv, aelv)
    tak_t = torch.where(neg, atakof, btakof)
    tak_r = torch.where(neg, btakof, atakof)
    return telv, relv, tak_t, tak_r


def _mode1(z: AzelBatch) -> None:
    # 198-217: end point from bearing and distance
    tlatr = z.ztlat.double() * DTOR
    tazr = z.ztaz.double() * DTOR
    gc = z.zdgc.double() / RERTH
    colat = PI / 2.0 - tlatr
    cosco = torch.cos(colat)
    sinco = torch.sin(colat)
    cosgc = torch.cos(gc)
    singc = torch.sin(gc)
    cosb = cosco * cosgc + sinco * singc * torch.cos(tazr)
    arg = torch.clamp(1.0 - cosb * cosb, min=0.0)
    b = torch.atan2(torch.sqrt(arg), cosb)
    arc = (cosgc - cosco * cosb) / (sinco * torch.sin(b))
    arg = torch.clamp(1.0 - arc * arc, min=0.0)
    rdlon = torch.atan2(torch.sqrt(arg), arc)
    zrlat32 = ((PI / 2.0 - b.abs()) * RTOD).float()
    drlat = zrlat32.double()  # 214-215: sign transfer round-trips REAL*4
    z.zrlat = torch.copysign(drlat, cosb).float()
    rl_abs = rdlon.abs() * RTOD
    zrlon_plus = (z.ztlon.double() + rl_abs).float()
    zrlon_minus = (z.ztlon.double() - rl_abs).float()
    z.zrlon = torch.where(z.ztaz > 180.0, zrlon_minus, zrlon_plus)  # 217
    # 218-223: distance for the elevation block, then join mode 0 at 142.
    # Mode 1 does NOT write zd or zdgc (D stays internal).
    thts = (z.ztht * 1.0e-3).double()
    rhts = (z.zrht * 1.0e-3).double()
    delht = rhts - thts
    sgc = torch.sin(0.5 * gc)
    d = torch.sqrt(
        delht * delht
        + 4.0 * (RERTH + thts) * (RERTH + rhts) * sgc * sgc
    )
    telv, relv, tak_t, tak_r = _elevations(gc, d, delht, thts, rhts)
    z.ztelv = telv.float()
    z.zrelv = relv.float()
    z.ztakof = tak_t.float()
    z.zrakof = tak_r.float()
