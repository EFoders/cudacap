"""Stage B (PyTorch, batched) antenna gain, ground loss and power — GAIN, PWRDB.

GAIN is the gate for the rest of LUFFY's frequency loop: REGMOD, ESMOD, ESREG,
GENOIS and SETTXR all call it, so it lands before any of them.  It is two
independent routines behind one entry point:

* ITR < 0  -- ground reflection loss from the Fresnel coefficients of the
  sample area's ground constants.  Self-contained, no tables.
* ITR >= 0 -- antenna gain, a bilinear interpolation of the /cantenna/ tables
  over [frequency 1..30 MHz][elevation 0..90 deg], plus efficiency over
  frequency alone.

THE ANTENNA TABLES ARE SHARED ACROSS LANES, not per-lane.  A circuit has one
transmit and one receive antenna, and an area-coverage run is one transmitter
seen from many grid points, so ARRAY/AEFF stay a single (31, 92, A) table and
only the INDICES into it are per-lane.  ``BatchedCantenna`` is that shared
table moved to the device once; build it from a Stage A ``CantennaState`` and
hand it to every routine that needs a gain.

THE MISSING BOTTOM CLAMP IS STILL REFUSED.  GAIN clamps only the TOP of each
axis (MIN0(30,..) / MIN0(91,..)), so FMC < 1 or DELTA < 0 reads ARRAY out of
bounds in the Fortran and picks up whatever COMMON follows it -- a real defect
INMUF is known to trigger.  Stage A raises rather than reproducing it, and so
does this, because a negative index in torch would silently wrap to the far
end of the axis.  The check costs one device sync per call; that is the price
of not silently returning a neighbouring antenna's pattern.

float32; COS/SIN/ATAN/ASIN/ALOG all carry the [[f32-libm-noise]] floor.
"""

from dataclasses import dataclass

import numpy as np
import torch

from voacap_gpu.stage_a.antennastate import AREA_COVERAGE
from voacap_gpu.stage_a.constants import R2D
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
_ZERO_ELEV = float(np.float32(0.00000001))
_GRAZING_LOSS = 6.0
NFREQ, NELEV = 30, 91


@dataclass
class BatchedCantenna:
    """/cantenna/ + /pantenna/ on the device, shared by every lane.

    array: (31, 92, A+1) gain tables, 1-based in all three axes.
    aeff:  (31, A+1) efficiencies.  iats/xfqs/xfqe/offazim/pwrdba: (A+1,).
    """

    array: torch.Tensor
    aeff: torch.Tensor
    iats: torch.Tensor
    xfqs: torch.Tensor
    xfqe: torch.Tensor
    offazim: torch.Tensor
    pwrdba: torch.Tensor
    numants: int

    @classmethod
    def from_stage_a(cls, cant, device="cpu"):
        t = lambda a, d=_F32: torch.as_tensor(  # noqa: E731
            np.asarray(a), dtype=d, device=device)
        return cls(array=t(cant.array), aeff=t(cant.aeff),
                   iats=t(cant.iats, torch.long), xfqs=t(cant.xfqs),
                   xfqe=t(cant.xfqe), offazim=t(cant.offazim),
                   pwrdba=t(cant.pwrdba), numants=int(cant.numants))


def _select(cant, itr, fmc):
    """GAIN's DO-10 scan, as a first-match fold over the antenna entries.

    Returns the 1-based /cantenna/ index per lane, or 0 for "no match" --
    which GAIN treats as a 0 dB isotrope rather than an error.  The entry
    count is a handful, so unrolling it beats any cleverer search.
    """
    out = torch.zeros_like(fmc, dtype=torch.long)
    for iat in range(1, cant.numants + 1):
        if int(cant.iats[iat]) != itr:
            continue                        # terminal is a per-ENTRY property
        hit = (out == 0) & (cant.xfqs[iat] <= fmc) & (fmc <= cant.xfqe[iat])
        out = torch.where(hit, torch.full_like(out, iat), out)
    return out


def ground_loss_batch(sigpat, epspat, delta, fmc):
    """GAIN's ITR<0 branch: ground reflection loss in dB, per lane.

    sigpat/epspat: (N,) conductivity (mhos/m) and relative permittivity of the
    sample area.  delta: (N,) elevation angle in RADIANS.  fmc: (N,) MHz.
    A non-positive conductivity means "no ground data" and returns 0, which is
    GAIN's preset rather than a computed answer.
    """
    k = {n: _k(v, fmc) for n, v in
         (("two", 2.0), ("half", 0.5), ("k18000", 18000.0),
          ("log10e", 4.3429), ("graze", _GRAZING_LOSS))}
    x = k["k18000"] * sigpat / fmc
    t = torch.cos(delta)
    q = torch.sin(delta)
    r = q * q
    s = r * r
    ert = epspat - t * t
    rho = torch.sqrt(ert * ert + x * x)
    rho12 = torch.sqrt(rho)
    alpha = -torch.atan(x / ert)
    u = epspat * epspat + x * x
    v = torch.sqrt(u)
    asxv = torch.asin(x / v)
    cv = (torch.sqrt(rho * rho + u * u * s
                     - k["two"] * rho * u * r * torch.cos(alpha
                                                          + k["two"] * asxv))
          / (rho + u * r
             + k["two"] * rho12 * v * q * torch.cos(alpha * k["half"] + asxv)))
    ch = (torch.sqrt(rho * rho + s - k["two"] * rho * r * torch.cos(alpha))
          / (rho + r + k["two"] * rho12 * q * torch.cos(alpha * k["half"])))
    rain = torch.abs(k["log10e"] * torch.log(k["half"] * (ch * ch + cv * cv)))
    rain = torch.where(delta <= _ZERO_ELEV, k["graze"], rain)
    return torch.where(sigpat <= 0.0, torch.zeros_like(rain), rain)


def antenna_gain_batch(cant, itr, delta, fmc):
    """GAIN's ITR>=0 branch: bilinear gain lookup.  Returns (rain, eff).

    itr: 1 = transmitter, 2 = receiver (a scalar; the terminal is a property
    of the call, not of the lane).  delta: (N,) elevation in RADIANS.
    fmc: (N,) MHz.  RAIN is a gain in dB here, not a loss.

    Both axes clamp only at the TOP, so the top cell holds flat rather than
    extrapolating.  See the module docstring for why the missing bottom clamp
    raises instead.
    """
    iat = _select(cant, itr, fmc)
    matched = iat > 0
    if bool((cant.offazim[iat] == AREA_COVERAGE)[matched].any()):
        raise NotImplementedError(
            "area-coverage antennas (OFFAZIM == -999) use /Careaant/ "
            "IARRAY360; point-to-point mode is ported first")

    i = torch.trunc(fmc).to(torch.long)      # REAL->INT truncates toward zero
    deltd = delta * _k(R2D, fmc)
    j = torch.trunc(deltd + _k(1.0, fmc)).to(torch.long).clamp(max=NELEV)
    # I is NOT clamped at the top either -- only IP1 gets MIN0(30,..) -- so an
    # entry whose XFQE reaches past 30 MHz would read past the table the same
    # way a negative index reads before it.  Same policy, same refusal.
    bad = matched & ((i < 1) | (j < 1) | (i > NFREQ))
    if bool(bad.any()):
        raise IndexError(
            f"GAIN index outside the table on {int(bad.sum())} lane(s): the "
            "Fortran reads out of bounds here and picks up a neighbouring "
            f"antenna's pattern.  I in [{int(i[bad].min())}, "
            f"{int(i[bad].max())}], J in [{int(j[bad].min())}, "
            f"{int(j[bad].max())}]; need 1 <= I <= {NFREQ} and J >= 1, so "
            "honour XFQS/XFQE and pass a non-negative elevation")
    xfmc = fmc - i.to(_F32)
    xdelta = deltd - (j - 1).to(_F32)

    # UNMATCHED LANES STILL GET GATHERED, so their indices have to be made
    # safe -- and they are genuinely out of range, because FMC is whatever the
    # caller asked for on a lane no antenna covers (GENOIS reaches 55 MHz).
    # The scalar routine never sees this: it returns 0 dB on `iat == 0` BEFORE
    # computing an index at all.  Batched, the early return becomes a select
    # after the fact, so the gather has to survive the lanes it will discard.
    iz = torch.where(matched, i, torch.zeros_like(i)).clamp(0, NFREQ)
    jz = torch.where(matched, j, torch.zeros_like(j)).clamp(0, NELEV)
    ip1 = (iz + 1).clamp(max=NFREQ)
    jp1 = (jz + 1).clamp(max=NELEV)
    xx = cant.array[iz, jz, iat]
    yx = cant.array[ip1, jz, iat]
    xy = cant.array[iz, jp1, iat]
    yy = cant.array[ip1, jp1, iat]
    rx = xx + xfmc * (yx - xx)
    ry = xy + xfmc * (yy - xy)
    rain = rx + xdelta * (ry - rx)
    a0 = cant.aeff[iz, iat]
    eff = a0 + xfmc * (cant.aeff[ip1, iat] - a0)

    zero = torch.zeros_like(rain)
    return torch.where(matched, rain, zero), torch.where(matched, eff, zero)


def pwrdb_batch(cant, freq):
    """Transmitter power in dBW at `freq` (MHz), per lane.

    The frequency is clamped to 2..30 MHz FOR THE RANGE TEST ONLY.  30 dBW
    (1 kW) is the fallback when no transmitter entry matches -- unlike GAIN,
    which returns 0 for a miss, this one has a non-zero default.
    """
    f = torch.clamp(freq, min=2.0, max=30.0)
    out = torch.full_like(f, 30.0)
    found = torch.zeros_like(f, dtype=torch.bool)
    for i in range(1, cant.numants + 1):
        if int(cant.iats[i]) != 1:
            continue                        # not the transmitter
        hit = ~found & (cant.xfqs[i] <= f) & (f <= cant.xfqe[i])
        out = torch.where(hit, cant.pwrdba[i], out)
        found = found | hit
    return out
