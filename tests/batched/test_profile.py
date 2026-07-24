"""Stage B batched LECDEN vs. the Stage A scalar profile, CPU+CUDA.

The layer parameters come from the real coefficient maps at a global grid, so
the profiles exercised here are the ones the model actually builds rather than
synthetic ones -- including the awkward lanes where the F1 fit moves the layer
or drops it entirely.

Stage A is the reference.  Both implementations are float32 with the same EXP
and SQRT, so the bar is tight: the height grid is pure add/divide and should
be BIT-EXACT, and the densities carry only the exp32 floor.
"""

import pathlib

import numpy as np
import pytest
import torch

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.batched.area import build_map
from voacap_gpu.batched.profile import NHT, lecden_batch

ROOT = pathlib.Path(__file__).resolve().parents[2]
COEFFS = ROOT / "reference" / "voacapl" / "voacapl" / "itshfbc" / "coeffs"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MONTH, SSN, GMT = 4, 120.0, 12.0

HTR_ULP_BAR = 0.0                # the grid is add/divide only -- bit-exact
FNSQ_ULP_BAR = 8.0               # densities carry exp32


def _grid(n_lat=13, n_lon=24):
    lat = np.linspace(-80.0, 80.0, n_lat, dtype=np.float32)
    lon = np.linspace(-180.0, 165.0, n_lon, dtype=np.float32)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return la.ravel(), lo.ravel()


def _floored(a, b, floor=1.0):
    a, b = np.float32(a), np.float32(b)
    if a == b:
        return 0.0
    denom = float(np.spacing(np.float32(max(abs(float(b)), floor))))
    return abs(float(a) - float(b)) / denom


def _stage_a(p, i):
    """Run the scalar LECDEN on lane `i` of the batched parameter map."""
    st = LayerState()
    st.km = 1
    for ly, key in ((1, "1"), (2, "2"), (3, "3")):
        st.fi[ly, 1] = F32(p[f"fi{key}"][i])
        st.yi[ly, 1] = F32(p[f"yi{key}"][i])
        st.hi[ly, 1] = F32(p[f"hi{key}"][i])
    lecden(st, 1)
    return st


@pytest.mark.parametrize("device", DEVICES)
def test_lecden_batch_matches_stage_a(device):
    st, params = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, *_grid(),
                           device=device)
    p = {k: v.cpu().numpy() for k, v in params.items()}
    args = [params[k] for k in ("fi1", "fi2", "fi3", "yi1", "yi2", "yi3",
                                "hi1", "hi2", "hi3")]
    htr, fnsq, hi2, yi2, fsecv = (t.cpu().numpy()
                                  for t in lecden_batch(*args))

    n = len(p["fi1"])
    seen = {"f1_present": 0, "f1_absent": 0, "f1_moved": 0, "f1_linear": 0,
            "two_layer_grid": 0, "single_layer_grid": 0}
    worst_h = worst_f = 0.0
    at_h = at_f = None
    for i in range(n):
        a = _stage_a(p, i)
        seen["f1_present"] += p["fi2"][i] > 0.0
        seen["f1_absent"] += p["fi2"][i] <= 0.0
        seen["f1_moved"] += float(hi2[i]) != float(p["hi2"][i])
        seen["f1_linear"] += float(fsecv[i]) > 0.0
        # the two height-grid shapes differ in where point 28 sits
        seen["two_layer_grid"] += abs(float(htr[27, i])
                                      - float(hi2[i])) < 1e-3
        seen["single_layer_grid"] += abs(float(htr[27, i])
                                         - float(hi2[i])) >= 1e-3
        for ih in range(NHT):
            d = _floored(htr[ih, i], a.htr[ih + 1, 1])
            if d > worst_h:
                worst_h, at_h = d, f"lane {i} htr[{ih + 1}]"
            assert d <= HTR_ULP_BAR, (
                f"lane {i} htr[{ih + 1}]: {htr[ih, i]!r} vs "
                f"{a.htr[ih + 1, 1]!r} ({d:.1f} ULP)")
            d = _floored(fnsq[ih, i], a.fnsq[ih + 1, 1])
            if d > worst_f:
                worst_f, at_f = d, f"lane {i} fnsq[{ih + 1}]"
            assert d <= FNSQ_ULP_BAR, (
                f"lane {i} fnsq[{ih + 1}]: {fnsq[ih, i]!r} vs "
                f"{a.fnsq[ih + 1, 1]!r} ({d:.1f} ULP)")
        assert _floored(hi2[i], a.hi[2, 1]) == 0.0, f"lane {i} HI(2)"
        assert _floored(yi2[i], a.yi[2, 1]) == 0.0, f"lane {i} YI(2)"
        assert _floored(fsecv[i], a.fsecv[1]) == 0.0, f"lane {i} FSECV"
    for arm, c in seen.items():
        assert c, f"the grid never reaches {arm} ({seen})"
    assert worst_h == 0.0, f"height grid drifted at {at_h}"
    print(f"[{device}] {n} lanes, htr exact, fnsq worst {worst_f:.1f} ULP "
          f"at {at_f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu():
    """Everything that is pure arithmetic must agree BITWISE across devices;
    only the exp-bearing densities may move, and then by at most 2 ULP.

    Splitting the assertion this way is what made the two real bugs here
    visible.  Both showed up as CPU/CUDA splits in HTR, which is nothing but
    adds and divides and therefore has no excuse to differ: a cumsum ladder
    that reassociated, and a division by a Python float, which on CUDA does
    not round like a float32 divide.  A single blanket ULP bar would have
    absorbed both silently.

    What is left is torch.exp disagreeing between the devices in the
    D-region tail -- 734 of 15,600 cells, all in the bottom four heights,
    worst 2 ULP.  That is the same [[f32-libm-noise]] floor the rest of the
    project documents, not a porting error.
    """
    grid = _grid()
    outs = []
    for device in ("cpu", "cuda"):
        _, params = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, *grid,
                              device=device)
        args = [params[k] for k in ("fi1", "fi2", "fi3", "yi1", "yi2", "yi3",
                                    "hi1", "hi2", "hi3")]
        outs.append([t.cpu().numpy() for t in lecden_batch(*args)])
    names = ("htr", "fnsq", "hi2", "yi2", "fsecv")
    for name, a, b in zip(names, *outs):
        if name == "fnsq":
            continue
        assert np.array_equal(a, b), (
            f"{name} differs between CPU and CUDA, but it is pure arithmetic")
    a, b = outs[0][1], outs[1][1]
    sp = np.spacing(np.maximum(np.abs(b), np.float32(1.0)))
    ulp = np.abs(a.astype(np.float64) - b.astype(np.float64)) / sp
    assert ulp.max() <= 4.0, f"fnsq worst {ulp.max():.2f} ULP across devices"
    rows = set(np.argwhere(a != b)[:, 0].tolist())
    assert rows <= {0, 1, 2, 3}, (
        f"exp noise should be confined to the D-region tail, saw rows {rows}")


def test_the_height_grid_is_monotonic():
    """HTR is a height ladder: FINDF and GETHP both walk it in order, so a
    non-increasing step would put the profile out of sequence."""
    _, params = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, *_grid())
    args = [params[k] for k in ("fi1", "fi2", "fi3", "yi1", "yi2", "yi3",
                                "hi1", "hi2", "hi3")]
    htr = lecden_batch(*args)[0].numpy()
    diffs = np.diff(htr, axis=0)
    assert (diffs > 0).all(), (
        f"{(diffs <= 0).sum()} non-increasing steps, first at height index "
        f"{int(np.argwhere(diffs <= 0)[0][0]) + 1}")
