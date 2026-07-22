"""GAIN -- antenna power gain and ground reflection loss (Module 5).

gain.for is two independent routines behind one entry point, selected by ITR:

* ``ITR < 0``  -- ground reflection loss for sample area |ITR|, from the
  Fresnel reflection coefficients of the ground constants SIGPAT (conductivity,
  mhos/m) and EPSPAT (relative permittivity) in /GEOG/.  Self-contained maths,
  no tables.  Implemented here as :func:`ground_loss`.
* ``ITR >= 0`` -- antenna gain: a bilinear interpolation of the /cantenna/
  tables over [frequency 1..30 MHz][elevation 0..90 deg] (plus azimuth, for
  area-coverage antennas), returning gain and efficiency.  That half needs the
  .VOA antenna tables loaded and lands with them; only the runtime lookup path
  is in scope (MoM pattern synthesis is not).

GAIN is the gate for the rest of LUFFY's frequency loop -- REGMOD, ESMOD,
ESREG, GENOIS and SETTXR all call it.  The ground half has no table
dependency, so it lands first.

float32; COS/SIN/ATAN/ASIN/ALOG all carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.antennastate import AREA_COVERAGE
from voacap_gpu.stage_a.constants import R2D
from voacap_gpu.stage_a.f32math import (F32, asin32, atan32, cos32, log32,
                                        sin32, sqrt32)

_TWO = F32(2.0)
_HALF = F32(0.5)


def ground_loss(state, ig: int, delta, fmc) -> F32:
    """GAIN's ITR<0 branch: ground reflection loss in dB for sample area IG.

    state: LayerState (SIGPAT/EPSPAT); delta: elevation angle (radians);
    fmc: frequency (MHz).  Returns RAIN, the loss magnitude; the companion EFF
    output is 0 on this path (GAIN presets it and never touches it here).
    A non-positive conductivity means "no ground data" and returns 0.
    """
    delta = F32(delta)
    fmc = F32(fmc)
    sigma = state.sigpat[ig]
    er = state.epspat[ig]
    if sigma <= 0.0:
        return F32(0.0)          # RAIN keeps its preset 0
    relta = delta
    x = F32(18000.0) * sigma / fmc
    t = cos32(relta)
    q = sin32(relta)
    r = q * q
    s = r * r
    ert = er - t * t
    rho = sqrt32(ert * ert + x * x)
    rho12 = sqrt32(rho)
    alpha = -atan32(x / ert)
    u = er * er + x * x
    v = sqrt32(u)
    asxv = asin32(x / v)
    # vertical-polarisation reflection coefficient
    cv = (sqrt32(rho * rho + u * u * s
                 - _TWO * rho * u * r * cos32(alpha + _TWO * asxv))
          / (rho + u * r + _TWO * rho12 * v * q * cos32(alpha * _HALF + asxv)))
    # horizontal-polarisation reflection coefficient
    ch = (sqrt32(rho * rho + s - _TWO * rho * r * cos32(alpha))
          / (rho + r + _TWO * rho12 * q * cos32(alpha * _HALF)))
    rain = F32(4.3429) * log32(_HALF * (ch * ch + cv * cv))
    rain = abs(rain)
    if delta <= F32(0.00000001):
        rain = F32(6.0)
    return rain


def _select(cant, itr: int, fmc) -> int:
    """GAIN's DO-10 scan: first entry for terminal `itr` covering `fmc`.

    Returns the 1-based /cantenna/ index, or 0 for "no match" -- which GAIN
    treats as a 0 dB isotrope rather than an error.
    """
    for iat in range(1, cant.numants + 1):
        if cant.iats[iat] != itr:
            continue
        if cant.xfqs[iat] <= fmc and fmc <= cant.xfqe[iat]:
            return iat
    return 0


def antenna_gain(cant, itr: int, delta, fmc):
    """GAIN's ITR>=0 branch: bilinear gain lookup.  Returns (RAIN, EFF).

    cant: CantennaState; itr: 1 = transmitter, 2 = receiver; delta: elevation
    angle (radians); fmc: frequency (MHz).  RAIN is gain in dB here, not a loss.

    Interpolates ARRAY over frequency (1 MHz cells, index = INT(FMC)) and
    elevation (1 degree cells, index = INT(DELTD)+1), and AEFF over frequency
    alone.  Both axes clamp only at the TOP (MIN0(30,..), MIN0(91,..)), so the
    top cell degenerates to a flat hold rather than extrapolating.

    There is no matching bottom clamp: FMC < 1 or DELTA < 0 indexes ARRAY at 0
    or below, which is out of bounds in the Fortran and reads adjacent COMMON.
    Rather than reproduce that (Python would silently wrap to the far end of
    the axis), this raises -- callers are expected to honour XFQS/XFQE and pass
    a non-negative elevation.
    """
    delta = F32(delta)
    fmc = F32(fmc)
    iat = _select(cant, itr, fmc)
    if iat == 0:
        return F32(0.0), F32(0.0)               # RAIN/EFF keep their preset 0
    if cant.offazim[iat] == AREA_COVERAGE:
        raise NotImplementedError(
            "area-coverage antennas (OFFAZIM == -999) use /Careaant/ "
            "IARRAY360; point-to-point mode is ported first")
    i = int(fmc)                                # REAL->INT truncates toward 0
    ip1 = min(30, i + 1)
    xfmc = fmc - F32(i)
    deltd = delta * R2D
    j = int(deltd + F32(1.0))
    j = min(j, 91)
    jp1 = min(91, j + 1)
    xdelta = deltd - F32(j - 1)
    if i < 1 or j < 1:
        raise IndexError(
            f"GAIN index below the table: fmc={float(fmc)} -> I={i}, "
            f"delta={float(delta)} rad -> J={j} (Fortran reads out of bounds "
            f"here; both must be >= 1)")
    array = cant.array
    xx = array[i, j, iat]
    yx = array[ip1, j, iat]
    xy = array[i, jp1, iat]
    yy = array[ip1, jp1, iat]
    rx = xx + xfmc * (yx - xx)
    ry = xy + xfmc * (yy - xy)
    rain = rx + xdelta * (ry - rx)
    aeff = cant.aeff
    eff = aeff[i, iat] + xfmc * (aeff[ip1, iat] - aeff[i, iat])
    return rain, eff
