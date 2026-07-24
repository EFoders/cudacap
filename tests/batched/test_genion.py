"""Stage B batched GENION vs. the Stage A scalar ionogram, CPU+CUDA.

Closes the chain: iono map -> profile -> ionogram, all GPU-resident.  The
layer parameters come from the real coefficient maps over a global grid, so
both frequency-layout shapes (with and without an F1 layer) appear, which is
asserted rather than hoped for.

FVERT is pure arithmetic and must be BIT-EXACT.  HPRIM and HTRUE come through
GETHP, which is bit-clean too, so they get the same bar -- the profile they
integrate is the one thing that can carry noise, and it is passed in
identically rather than rebuilt per device.
"""

import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.batched.area import build_map
from voacap_gpu.batched.genion import NFV, genion_batch
from voacap_gpu.batched.profile import lecden_batch

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


def _chain(device):
    _, params = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, *_grid(),
                          device=device)
    htr, fnsq, hi2, yi2, _ = lecden_batch(*[params[k] for k in PARAMS])
    return params, htr, fnsq, hi2, yi2


def _stage_a(params, htr, fnsq, hi2, yi2, i):
    """Scalar GENION on lane i, with the profile LECDEN actually produced.

    HI(2)/YI(2) are the values the F1 fit MOVED, not the map's originals --
    LECDEN mutates them in place and GENION reads them afterwards, so feeding
    the originals here would compare against an ionogram the model never
    builds.
    """
    st = LayerState()
    st.km = 1
    st.iedp = -1
    for ly, key in ((1, "1"), (2, "2"), (3, "3")):
        st.fi[ly, 1] = np.float32(params[f"fi{key}"][i])
        st.yi[ly, 1] = np.float32(params[f"yi{key}"][i])
        st.hi[ly, 1] = np.float32(params[f"hi{key}"][i])
    st.hi[2, 1] = np.float32(hi2[i])
    st.yi[2, 1] = np.float32(yi2[i])
    st.htr[1:51, 1] = htr[:, i]
    st.fnsq[1:51, 1] = fnsq[:, i]
    genion(st, 1)
    return st


@pytest.mark.parametrize("device", DEVICES)
def test_genion_batch_matches_stage_a_bit_exactly(device):
    params, htr, fnsq, hi2, yi2 = _chain(device)
    fv, hp, ht = (t.cpu().numpy()
                  for t in genion_batch(htr, fnsq, params["fi1"],
                                        params["fi2"], params["fi3"]))
    p = {k: params[k].cpu().numpy() for k in PARAMS}
    h_np, f_np = htr.cpu().numpy(), fnsq.cpu().numpy()
    hi2_np, yi2_np = hi2.cpu().numpy(), yi2.cpu().numpy()

    seen = {"with_f1": 0, "without_f1": 0}
    for i in range(len(p["fi1"])):
        seen["with_f1"] += p["fi2"][i] > 0.0
        seen["without_f1"] += p["fi2"][i] <= 0.0
        a = _stage_a(p, h_np, f_np, hi2_np, yi2_np, i)
        for j in range(NFV):
            assert np.float32(fv[j, i]) == np.float32(a.fvert[j + 1, 1]), (
                f"[{device}] lane {i} FVERT({j + 1}): {fv[j, i]!r} vs "
                f"{a.fvert[j + 1, 1]!r}")
            assert np.float32(hp[j, i]) == np.float32(a.hprim[j + 1, 1]), (
                f"[{device}] lane {i} HPRIM({j + 1}): {hp[j, i]!r} vs "
                f"{a.hprim[j + 1, 1]!r}")
            assert np.float32(ht[j, i]) == np.float32(a.htrue[j + 1, 1]), (
                f"[{device}] lane {i} HTRUE({j + 1}): {ht[j, i]!r} vs "
                f"{a.htrue[j + 1, 1]!r}")
    for arm, c in seen.items():
        assert c, f"the grid never reaches {arm} ({seen})"


def test_the_sounding_frequencies_ascend():
    """FVERT is a sweep: FOBBY applies the secant law along it and FINDF
    interpolates between neighbours, so an out-of-order pair would put the
    reflectrix out of sequence."""
    params, htr, fnsq, _, _ = _chain("cpu")
    fv = genion_batch(htr, fnsq, params["fi1"], params["fi2"],
                      params["fi3"])[0].numpy()
    d = np.diff(fv, axis=0)
    assert (d > 0).all(), (
        f"{(d <= 0).sum()} non-ascending steps, first at FVERT index "
        f"{int(np.argwhere(d <= 0)[0][0]) + 1}")


def test_the_fast_model_is_refused():
    """IEDP >= 0 selects a different, cheaper height model.  It is not the
    Gaussian one, so it must not be silently treated as it."""
    params, htr, fnsq, _, _ = _chain("cpu")
    with pytest.raises(NotImplementedError, match="IEDP"):
        genion_batch(htr, fnsq, params["fi1"], params["fi2"], params["fi3"],
                     iedp=1)
