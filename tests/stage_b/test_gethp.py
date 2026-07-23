"""Stage B batched GETHP vs. the Stage A scalar integration, CPU+CUDA.

The profiles come from the batched LECDEN over a real global grid, and the
frequencies from the same layer parameters, so the (profile, frequency) pairs
exercised here are the ones an ionogram actually asks for.

GETHP is bit-clean in Stage A -- only SQRT, which is correctly rounded -- so
the bar is bit-exactness, on both devices.  The batched version reduces one of
its two table searches to a bracketing lookup, and the whole point of the bar
being this tight is that the reduction has to be exactly equivalent, not
approximately.
"""

import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.gethp import gethp
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_b.area import build_map
from voacap_gpu.stage_b.gethp import gethp_batch
from voacap_gpu.stage_b.profile import lecden_batch

ROOT = pathlib.Path(__file__).resolve().parents[2]
COEFFS = ROOT / "reference" / "voacapl" / "voacapl" / "itshfbc" / "coeffs"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MONTH, SSN, GMT = 4, 120.0, 12.0
PARAMS = ("fi1", "fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3")


def _grid(n_lat=7, n_lon=11):
    lat = np.linspace(-70.0, 70.0, n_lat, dtype=np.float32)
    lon = np.linspace(-180.0, 150.0, n_lon, dtype=np.float32)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return la.ravel(), lo.ravel()


def _profiles(device):
    _, params = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, *_grid(),
                          device=device)
    htr, fnsq, _, _, _ = lecden_batch(*[params[k] for k in PARAMS])
    return params, htr, fnsq


def _freqs(params, nf=12):
    """A frequency sweep spanning the whole profile, per lane.

    The two ends GETHP handles specially are worth reaching deliberately.
    Below the profile FLOOR means FR <= FNSQ(1), and FNSQ(1) is the D-region
    density at 70 km -- of order 3e-4 MHz^2, so it takes a frequency around
    0.017 MHz, far below anything in the sweep proper.  The first row is
    pinned there rather than left to chance.
    """
    lo = 0.2 * params["fi1"]
    hi = 1.05 * params["fi3"]
    t = torch.linspace(0.0, 1.0, nf - 1, dtype=torch.float32,
                       device=lo.device).unsqueeze(1)
    sweep = lo.unsqueeze(0) + t * (hi - lo).unsqueeze(0)
    floor = torch.full_like(sweep[:1], 0.005)     # under FNSQ(1) everywhere
    return torch.cat((floor, sweep), dim=0)


def _stage_a(htr, fnsq, i, f):
    st = LayerState()
    st.km = 1
    st.htr[1:51, 1] = htr[:, i]
    st.fnsq[1:51, 1] = fnsq[:, i]
    return gethp(st, F32(f))


@pytest.mark.parametrize("device", DEVICES)
def test_gethp_batch_matches_stage_a_bit_exactly(device):
    params, htr, fnsq = _profiles(device)
    fxx = _freqs(params)
    hpx, htx = (t.cpu().numpy() for t in gethp_batch(htr, fnsq, fxx))
    h_np, f_np = htr.cpu().numpy(), fnsq.cpu().numpy()
    fx_np = fxx.cpu().numpy()

    seen = {"below_floor": 0, "in_profile": 0, "above_top": 0}
    nf, n = fx_np.shape
    for i in range(n):
        for j in range(nf):
            f = fx_np[j, i]
            a_hp, a_ht = _stage_a(h_np, f_np, i, f)
            seen["below_floor"] += F32(f) * F32(f) <= f_np[0, i]
            seen["above_top"] += F32(f) * F32(f) > f_np[:, i].max()
            seen["in_profile"] += (f_np[0, i] < F32(f) * F32(f)
                                   <= f_np[:, i].max())
            assert np.float32(hpx[j, i]) == np.float32(a_hp), (
                f"[{device}] lane {i} freq {f:.4f}: HPX {hpx[j, i]!r} "
                f"vs {a_hp!r}")
            assert np.float32(htx[j, i]) == np.float32(a_ht), (
                f"[{device}] lane {i} freq {f:.4f}: HTX {htx[j, i]!r} "
                f"vs {a_ht!r}")
    for arm, c in seen.items():
        assert c, f"the sweep never reaches {arm} ({seen})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu_bitwise():
    """GETHP touches only SQRT, which is correctly rounded on both devices,
    so there is no libm floor here to excuse a difference.

    The profile is built ONCE and copied to both devices rather than built on
    each.  Building it per device would feed in LECDEN's own cross-device exp
    noise and this test would be measuring that instead -- it read as a GETHP
    difference the first time round, and was not one.
    """
    params, htr, fnsq = _profiles("cpu")
    fxx = _freqs(params)
    outs = []
    for device in ("cpu", "cuda"):
        args = [t.to(device) for t in (htr, fnsq, fxx)]
        outs.append([t.cpu().numpy() for t in gethp_batch(*args)])
    for name, a, b in zip(("hpx", "htx"), *outs):
        assert np.array_equal(a, b), f"{name} differs between CPU and CUDA"


def test_a_non_monotone_height_ladder_is_refused():
    """The node lookup brackets, which is only equivalent to the Fortran's
    scan while HTR increases -- so a table that does not increase has to be
    refused rather than quietly bracketed."""
    _, htr, fnsq = _profiles("cpu")
    htr = htr.clone()
    htr[20, 0], htr[21, 0] = htr[21, 0].clone(), htr[20, 0].clone()
    with pytest.raises(ValueError, match="not strictly increasing"):
        gethp_batch(htr, fnsq, torch.full_like(htr[:1], 5.0))
