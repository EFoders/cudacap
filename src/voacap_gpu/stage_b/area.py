"""Stage B (PyTorch, batched) Area-mode ionospheric-parameter map.

Composes the batched geometry (MAGVAR gyrofrequency/dip + the GEOM
geomagnetic-latitude formula) with the batched Module-2 layer chain into a
single per-grid-point kernel: given a lat/lon grid and a month/SSN/hour, it
returns the E/F1/F2/Es layer parameters at every point in one pass — the
first end-to-end GPU-resident VOACAP product.

Each grid point is its own sample area (KM=1), so no path/hop geometry is
involved; this is the ionospheric state map, the massively-parallel core
the whole GPU effort targets.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.virtim import virtim
from voacap_gpu.stage_b.layers import (D2R, _acos32, _cos32, _sin32,
                                       iono_params_batch)
from voacap_gpu.stage_b.magfin import magvar_batch

_PIO2 = float(np.float32(1.570796326))
# Oracle geomagnetic north pole (itshfbc/database/north_pole.txt).
POLE_LAT = 79.5
POLE_LON = -69.0


def _geomag_lat(clat, clong, pole_lat, pole_lon):
    """GEOM's geomagnetic latitude (geom.for:155-163), batched."""
    glt = float(np.float32(pole_lat)) * D2R
    glg = float(np.float32(pole_lon)) * D2R
    qcos = (_sin32(torch.full_like(clat, glt)) * _sin32(clat)
            + _cos32(torch.full_like(clat, glt)) * _cos32(clat)
            * _cos32(clong - glg))
    qcos = torch.clamp(qcos, -1.0, 1.0)
    return _PIO2 - _acos32(qcos)


def area_iono_map(st, ab, month, ssn, gmt, lat_deg, lon_deg,
                  pole_lat=POLE_LAT, pole_lon=POLE_LON):
    """st: IonoState; ab: (319,) float32 tensor (VIRTIM output on the target
    device); lat_deg/lon_deg: (N,) float32 tensors of grid coordinates.
    Returns the iono_params dict of (N,) tensors (foE=fi1, foF1=fi2,
    foF2=fi3, M3000=f2m3, Es median=fs2, heights, ...)."""
    clat = lat_deg * D2R
    clong = lon_deg * D2R
    gyz, gmdip, _ = magvar_batch(clat, clong)
    glat = _geomag_lat(clat, clong, pole_lat, pole_lon)
    return iono_params_batch(st, ab, month, ssn, gmt, clat, clong, glat,
                             gmdip, gyz)


def build_map(coeffs_dir, coeff, month, ssn, gmt, lat_deg, lon_deg,
              device="cpu"):
    """Convenience: load coefficients (CPU) + VIRTIM, move AB to `device`,
    and evaluate the map over the given grid tensors."""
    st = redmap(coeffs_dir, coeff, month, np.float32(ssn))
    ab = torch.from_numpy(virtim(st, np.float32(gmt))).to(device)
    lat = torch.as_tensor(lat_deg, dtype=torch.float32, device=device)
    lon = torch.as_tensor(lon_deg, dtype=torch.float32, device=device)
    return st, area_iono_map(st, ab, month, ssn, gmt, lat, lon)
