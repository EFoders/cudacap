"""Stage B Area-mode iono map vs. the Stage A per-point composition, CPU+CUDA.

area_iono_map wires the batched geometry (MAGVAR + the GEOM geomagnetic-
latitude formula) into the batched layer chain.  Every piece is already
validated against the oracle (test_magfin, test_layers, test_geom); this
confirms the composition matches the scalar chain lane-for-lane over a grid,
to the same VERSY/libm floor (worst on foF2/Es).
"""

import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.constants import D2R, PIO2
from voacap_gpu.scalar.esind import esind
from voacap_gpu.scalar.f2var import f2var
from voacap_gpu.scalar.f32math import F32, acos32, cos32, sin32
from voacap_gpu.scalar.ionset import ionset
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.magfin import magvar
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.timvar import timvar
from voacap_gpu.scalar.virtim import virtim
from voacap_gpu.batched.area import POLE_LAT, POLE_LON, area_iono_map

from tests.scalar.dumputil import COEFFS_DIR

ROOT = pathlib.Path(__file__).resolve().parents[2]
COEFFS = COEFFS_DIR    # the package's bundled copy (tests/scalar/dumputil.py)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MONTH, SSN, GMT = 4, 120.0, 12.0

# key products; bars mirror the layer-chain VERSY floor (fs/fi3) + small libm
BARS = {"fi1": 16, "fi2": 32, "fi3": 256, "f2m3": 16, "fs2": 512,
        "hi3": 32, "yi3": 32, "hpf2": 16, "rat": 16}


def _floored(a, b):
    a = np.float32(a)
    b = np.float32(b)
    if a == b:
        return 0.0
    return abs(float(a) - float(b)) / float(
        np.spacing(np.float32(max(abs(float(b)), 1.0))))


def _stage_a_point(st, ab, latd, lond):
    clat = F32(latd) * D2R
    clong = F32(lond) * D2R
    gyz, gmdip, _ = magvar(clat, clong)
    glt = F32(POLE_LAT) * D2R
    glg = F32(POLE_LON) * D2R
    q = (sin32(glt) * sin32(clat)
         + cos32(glt) * cos32(clat) * cos32(clong - glg))
    q = np.float32(min(1.0, max(-1.0, float(q))))
    glat = PIO2 - acos32(q)
    s = LayerState(km=1, month=MONTH, ssn=F32(SSN))
    s.clat[1], s.clong[1], s.glat[1] = clat, clong, glat
    s.gmdip[1], s.gyz[1] = gmdip, gyz
    timvar(s, st, ab, GMT)
    f2var(s, st, ab)
    esind(s, st, ab)
    ionset(s)
    return {"fi1": s.fi[1, 1], "fi2": s.fi[2, 1], "fi3": s.fi[3, 1],
            "f2m3": s.f2m3[1], "fs2": s.fs[2, 1], "hi3": s.hi[3, 1],
            "yi3": s.yi[3, 1], "hpf2": s.hpf2[1], "rat": s.rat[1]}


@pytest.mark.skipif(not COEFFS.exists(), reason="coeffs missing")
@pytest.mark.parametrize("device", DEVICES)
def test_area_map_vs_stage_a(device):
    st = redmap(COEFFS, "CCIR", MONTH, F32(SSN))
    ab_np = virtim(st, F32(GMT))
    lats = np.arange(-80, 81, 13, dtype=np.float32)
    lons = np.arange(-180, 180, 20, dtype=np.float32)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    latf, lonf = la.ravel(), lo.ravel()

    ab = torch.from_numpy(ab_np).to(device)
    out = area_iono_map(st, ab, MONTH, SSN, GMT,
                        torch.tensor(latf, device=device),
                        torch.tensor(lonf, device=device))
    total = exact = 0
    for i in range(len(latf)):
        ref = _stage_a_point(st, ab_np, latf[i], lonf[i])
        for k, bar in BARS.items():
            d = _floored(float(out[k][i].cpu()), ref[k])
            assert d <= bar, (f"[{device}] ({latf[i]},{lonf[i]}) {k}: "
                              f"{float(out[k][i].cpu())!r} vs {ref[k]!r} "
                              f"({d:.1f} floored ULP > {bar})")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.85, f"[{device}] bit-exact {exact}/{total}"
