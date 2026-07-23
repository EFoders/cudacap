"""Stage B batched FOBBY vs. the Stage A scalar reflectrix, CPU+CUDA.

Completes the GPU chain: map -> profile -> ionogram -> reflectrix.

IFOB is truncated to integer kHz, so a 1-ULP difference in the underlying
frequency is either invisible (it lands on the same integer) or costs a whole
kHz (it does not).  There is no in-between, and a ULP bar on the float would
call both of those a near-miss.

So the bar here is three claims instead of one: the flips are RARE (measured
0.248% CPU, 0.038% CUDA, bar 1%), every one is EXACTLY one kHz, and every one
sits against an integer BOUNDARY.  That last is the causal check -- it is what
separates a quantisation artefact from an arithmetic error, because a routine
that was actually wrong would move entries by more than a kHz or move ones
sitting in the middle of their interval.
"""

import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.fobby import fobby
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_b.area import build_map
from voacap_gpu.stage_b.fobby import fobby_batch
from voacap_gpu.stage_b.genion import NFV, genion_batch
from voacap_gpu.stage_b.profile import lecden_batch

ROOT = pathlib.Path(__file__).resolve().parents[2]
COEFFS = ROOT / "reference" / "voacapl" / "voacapl" / "itshfbc" / "coeffs"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MONTH, SSN, GMT = 4, 120.0, 12.0
PARAMS = ("fi1", "fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3")
# a realistic restricted scan: SANG narrows the 0..90 sweep for the distance
ANGLES = np.array([0.1, 1.0, 3.0, 6.0, 10.0, 16.0, 24.0, 35.0, 50.0, 74.0],
                  dtype=np.float32)


def _grid(n_lat=5, n_lon=7):
    lat = np.linspace(-60.0, 60.0, n_lat, dtype=np.float32)
    lon = np.linspace(-180.0, 120.0, n_lon, dtype=np.float32)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return la.ravel(), lo.ravel()


def _chain(device):
    _, params = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, *_grid(),
                          device=device)
    htr, fnsq, hi2, yi2, _ = lecden_batch(*[params[k] for k in PARAMS])
    fv, _, ht = genion_batch(htr, fnsq, params["fi1"], params["fi2"],
                             params["fi3"])
    return fv, ht


FLIP_RATE_BAR = 0.01            # measured 0.248% on CPU, 0.038% on CUDA
_NEAR_INT = 0.05                # how close to a boundary a flip must sit


def _stage_a_ifob(fv_np, ht_np):
    out = np.zeros((len(ANGLES), NFV, fv_np.shape[1]), dtype=np.int64)
    for i in range(fv_np.shape[1]):
        st = LayerState()
        st.km = 1
        st.fvert[1:NFV + 1, 1] = fv_np[:, i]
        st.htrue[1:NFV + 1, 1] = ht_np[:, i]
        rays = RaysState()
        rays.nang = len(ANGLES)
        rays.ang[1:len(ANGLES) + 1] = ANGLES
        fobby(st, rays, 1)
        out[:, :, i] = rays.ifob[1:len(ANGLES) + 1, 1:NFV + 1, 1]
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_fobby_batch_matches_stage_a_up_to_the_khz_quantisation(device):
    """Torch's COS and glibc's cosf differ by an ULP here and there, and the
    truncation turns that into a whole kHz or nothing at all.

    So the bar is not integer equality.  It is: the flips are RARE, every one
    of them is exactly one kHz, and every one sits against an integer boundary
    -- which is what distinguishes a quantisation artefact from an arithmetic
    error.  An implementation that was actually wrong would move entries by
    more than a kHz, or move ones sitting in the middle of their interval.
    """
    fv, ht = _chain(device)
    ang = torch.as_tensor(ANGLES, device=fv.device)
    ifob_t, khz_t = fobby_batch(ang, fv, ht, return_khz_float=True)
    ifob = ifob_t.cpu().numpy().astype(np.int64)
    khz = khz_t.cpu().numpy()
    ref = _stage_a_ifob(fv.cpu().numpy(), ht.cpu().numpy())

    delta = ifob - ref
    flips = np.argwhere(delta != 0)
    rate = len(flips) / delta.size
    assert np.abs(delta).max() <= 1, (
        f"[{device}] an entry moved {np.abs(delta).max()} kHz; the "
        "quantisation can only ever move one")
    assert rate <= FLIP_RATE_BAR, (
        f"[{device}] {len(flips)}/{delta.size} entries flipped "
        f"({100 * rate:.3f}%)")
    frac = np.abs(khz - np.round(khz))
    for ia, j, i in flips:
        assert frac[ia, j, i] <= _NEAR_INT, (
            f"[{device}] lane {i} angle {ANGLES[ia]} freq {j + 1} flipped "
            f"while sitting {frac[ia, j, i]:.3f} from an integer -- that is "
            "an arithmetic difference, not a rounding boundary")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu():
    """The ionogram is built once and copied, so this isolates FOBBY.

    COS is the one libm call, and the truncation means a 1-ULP disagreement
    either vanishes or costs a whole kHz.  Measured: 30 of 10,500 entries move
    between devices, every one by exactly a kHz -- torch's CUDA cosf is a
    different function from its CPU one, the same way both differ from glibc.
    """
    fv, ht = _chain("cpu")
    ang = torch.as_tensor(ANGLES)
    outs = [fobby_batch(ang.to(d), fv.to(d), ht.to(d)).cpu().numpy()
            .astype(np.int64) for d in ("cpu", "cuda")]
    delta = outs[0] - outs[1]
    rate = np.count_nonzero(delta) / delta.size
    assert np.abs(delta).max() <= 1, (
        f"an entry moved {np.abs(delta).max()} kHz between devices")
    assert rate <= FLIP_RATE_BAR, (
        f"IFOB differs between CPU and CUDA in {np.count_nonzero(delta)}/"
        f"{delta.size} entries ({100 * rate:.3f}%)")


def test_a_grazing_ray_is_clamped_not_negative():
    """SQCOS = 1 - SPHE^2 goes negative for a ray that cannot reflect at all;
    the Fortran clamps it to 1e-6 rather than taking the root of a negative,
    which pins the oblique frequency at a huge finite value instead of NaN."""
    fv = torch.full((1, 1), 5.0)
    ht = torch.full((1, 1), 300.0)
    ifob = fobby_batch(torch.tensor([0.0]), fv, ht)
    assert int(ifob[0, 0, 0]) > 0 and np.isfinite(float(ifob[0, 0, 0]))
