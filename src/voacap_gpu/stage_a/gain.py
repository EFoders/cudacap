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
