"""Stage B batched FINDF vs the SAME tier-B oracle dump Stage A uses.

The whole IEDP < 0 subset is loaded as one batch, so the lockstep machine runs
on lanes that genuinely diverge: 21 distinct ray counts from 9 to 44, all
three layers appearing in IMODE, half the lanes with an F1 ledge and half
without.  A lockstep port that quietly kept lanes in step would pass a uniform
batch and fail this one, which is why the divergence is asserted rather than
assumed.

IFOB IS INJECTED FROM THE ORACLE, exactly as Stage A's tier-B does, and for the
same reason: FOBBY truncates the reflectrix to integer kHz, so a 1-ULP COS
difference upstream either vanishes or moves an entry a whole kHz, and near a
cusp a small SLOPD turns that into a ~0.1% ground-distance shift.  That
sensitivity belongs to the reflectrix design and is covered by FOBBY's own
tier-B; injecting bit-identical IFOB isolates FINDF's state machine, which is
what this file is about.

THE BARS ARE PER FIELD, because the residual has a shape and a single bar
would hide it.  Measured on CPU (15262/15467 = 98.7% overall; CUDA 94.8%):

    iaftxr iskp imode afflx htflx fvflx   EXACT, every cell
    delfx                                 99.7%, worst 1.9e-6 deg
    hpflx                                 98.1%, worst 3.1e-5 km
    gdflx  dskpkm dmaxkm                  92.9%, worst 9.2 METRES in ~1500 km

That split is the result.  Every DISCRETE quantity -- how many rays were
found, which layer each belongs to, which ray is the skip ray -- is exact, so
the state machine dispatches identically to the Fortran on all 88 lanes.  So
are the pure-table quantities, which pins the height search and its
interpolation.  Only the Martyn geometry drifts, and only because
GDR = 2*RZ*(PIO2 - DEL - PHE) subtracts operands near 1.5 to leave ~0.12, a
~13x magnification of ASIN's floor.  Stage A is bit-exact here because its
ASIN32 is matched to this gfortran build's asinf; torch's ASIN is a different
function.  Nine metres is not a number worth defending against.

DELSKP/HTSKP/FVSKP/HPSKP being exact is the load-bearing part of that claim:
they are the skip RAY's own values, so if the drift in GDFLX had ever changed
which ray won the DSKPKM comparison, they would move by a whole ray.
"""

import csv

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.sang import ANG as SANG_ANG
from voacap_gpu.stage_b.findf import findf_batch
from voacap_gpu.stage_b.genion import genion_batch
from voacap_gpu.stage_b.profile import lecden_batch

from tests.stage_a.dumputil import GOLDEN, load_dump, ulp_diff_floored

DUMP = load_dump("findf")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_SCALARS = ["iaftxr", "dskpkm", "delskp", "hpskp", "htskp", "fvskp", "iskp",
            "dmaxkm"]
_ARRAYS = ["delfx", "gdflx", "hpflx", "htflx", "fvflx", "afflx", "imode"]
NANG = 40
NFV = 30

# DISCRETE fields.  A floating-point floor cannot move an integer count or a
# layer tag by a fraction -- it either dispatches the same way or it does
# not -- so any drift here is a wiring bug and these are compared bitwise on
# every device.  AFFLX joins them because the test injects a synthetic ramp,
# which is the same bits everywhere.
_EXACT = {"iaftxr", "iskp", "imode", "afflx"}
# HTFLX and FVFLX are copied or linearly interpolated out of the ionogram
# with no libm call of FINDF's own, so they are exact too -- but only when
# the ionogram feeding them is itself bit-identical to the oracle's, which is
# true of the CPU chain and not of the CUDA one.  Claiming them as exact on
# CUDA would be claiming something about GETHP, not about FINDF.
_EXACT_CPU_ONLY = {"htflx", "fvflx"}
# (floored ULP, absolute) in each field's own units -- km, degrees, MHz.
_BARS = {"gdflx": (256.0, 0.05), "dskpkm": (256.0, 0.05),
         "dmaxkm": (256.0, 0.05), "delskp": (8.0, 1.0e-4),
         "delfx": (8.0, 1.0e-4), "hpflx": (32.0, 0.01),
         "hpskp": (32.0, 0.01), "htskp": (8.0, 0.01),
         "fvskp": (8.0, 1.0e-4), "htflx": (8.0, 0.01),
         "fvflx": (8.0, 1.0e-4)}
EXACT_FLOOR = {"cpu": 0.97, "cuda": 0.85}   # measured 0.987 / 0.948


def _load_ifob():
    p = GOLDEN / "findf" / "ifob.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return [[int(x) for x in row] for row in csv.reader(f)]


IFOB_ROWS = _load_ifob()


def _batch(device):
    """The IEDP < 0 cases as one batch, with the oracle's IFOB and NANG."""
    _, inputs, rows = DUMP
    keep = [(inp, exp, ifr) for inp, exp, ifr in zip(inputs, rows, IFOB_ROWS)
            if int(round(float(inp[9]))) < 0]
    n = len(keep)
    cols = np.array([[float(x) for x in inp[:9]] + [float(inp[11]),
                                                    float(inp[12])]
                     for inp, _, _ in keep], dtype=np.float32).T
    t = [torch.as_tensor(c, device=device) for c in cols]
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, amind, freq = t

    htr, fnsq, _, _, _ = lecden_batch(fi1, fi2, fi3, yi1, yi2, yi3,
                                      hi1, hi2, hi3)
    fvert, hprim, htrue = genion_batch(htr, fnsq, fi1, fi2, fi3)
    # the same synthetic AFAC ramp Stage A's tier-B injects -- ALOSFV has its
    # own dump, and a ramp makes a misread height index obvious
    ih = torch.arange(1, NFV + 1, device=device, dtype=torch.float32)
    afac = (0.1 + 0.03 * ih).unsqueeze(1).expand(NFV, n).contiguous()

    ang = torch.as_tensor(np.asarray(SANG_ANG, dtype=np.float32),
                          device=device).unsqueeze(1).expand(NANG, n)
    nang = torch.tensor([ifr[0] for _, _, ifr in keep], device=device)
    ifob = torch.zeros((NANG, NFV, n), dtype=torch.int32, device=device)
    for i, (_, _, ifr) in enumerate(keep):
        ifob[:, :, i] = torch.as_tensor(
            np.asarray(ifr[1:], dtype=np.int32).reshape(NANG, NFV),
            device=device)

    args = (htrue, hprim, fvert, afac, fi2, fi3, ang.contiguous(), nang,
            ifob, freq, amind)
    return args, [exp for _, exp, _ in keep], fi2


def _check(device, name, got, want, total_exact):
    e = np.float32(want)
    g = np.float32(got)
    total_exact[0] += 1
    if g == e:
        total_exact[1] += 1
        return 0.0
    exact_here = _EXACT | (_EXACT_CPU_ONLY if device == "cpu" else set())
    assert name not in exact_here, (
        f"[{device}] {name} drifted: {g!r} vs {e!r}.  Nothing in FINDF's "
        "derivation of it calls libm, so this is a wiring bug, not a floor")
    ulp_bar, abs_bar = _BARS[name]
    d = ulp_diff_floored(g, e)
    assert d <= ulp_bar and abs(float(g) - float(e)) <= abs_bar, (
        f"[{device}] {name}: {g!r} vs {e!r} ({d:.1f} floored ULP, "
        f"{abs(float(g) - float(e)):.2e} absolute)")
    return d


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_findf_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_findf_batch_vs_oracle(device):
    args, expected, fi2 = _batch(device)
    rays, scalars = findf_batch(*args)
    got_s = {k: v.cpu().numpy() for k, v in scalars.items()}
    got_r = {k: v.cpu().numpy() for k, v in rays.items()}

    tally = [0, 0]                          # (compared, bit-exact)
    worst, where = 0.0, ""
    ray_counts = {}
    for i, exp in enumerate(expected):
        want_n = int(round(float(exp[0])))
        ray_counts[want_n] = ray_counts.get(want_n, 0) + 1
        for j, name in enumerate(_SCALARS):
            d = _check(device, name, got_s[name][i], exp[j], tally)
            if d > worst:
                worst, where = d, f"lane {i} {name}"
        for b, name in enumerate(_ARRAYS):
            base = 8 + b * 45
            arr = got_r[name]
            for j in range(1, want_n + 1):
                d = _check(device, name, arr[j, i], exp[base + j - 1], tally)
                if d > worst:
                    worst, where = d, f"lane {i} {name}[{j}]"

    # lockstep proves nothing on a batch that walks in step
    assert len(ray_counts) >= 10, (
        f"[{device}] the batch barely diverges: {len(ray_counts)} distinct "
        f"ray counts {sorted(ray_counts)}")
    assert 0 < int((fi2 <= 0).sum()) < len(expected), (
        "need both the F1-present and F1-absent layer dispatches")
    assert set(np.unique(got_r["imode"][1:])) >= {1, 2, 3}, (
        "not every layer is represented, so a layer transition is untested")
    # NOT asserted: PENANG's -1 veto of the whole area.  No case in this dump
    # reaches it (every DMAXKM is positive); it is covered in test_penang.py
    # against Stage A, which is the strongest claim available without a dump.
    assert int((got_s["dmaxkm"] < 0).sum()) == 0, (
        "a dump case now vetoes the area; the -1 path is reachable from the "
        "oracle after all and belongs in this comparison")
    assert tally[1] / tally[0] >= EXACT_FLOOR[device], (
        f"[{device}] exact {tally[1]}/{tally[0]}, worst {worst:.1f} ULP at "
        f"{where}")


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing")
def test_a_lane_does_not_depend_on_its_batch_neighbours():
    """The failure mode lockstep invites: a cursor or the row-0 sink leaking
    across lanes looks fine on a uniform batch and wrong on a mixed one.

    It is a real risk here rather than a theoretical one -- every lane rides
    along until the SLOWEST finishes, so a lane that stopped after nine rays
    keeps scattering into the sink for hundreds more steps while its
    neighbours are still working.
    """
    args, expected, _ = _batch("cpu")
    whole_r, whole_s = findf_batch(*args)
    for i in (0, 3, 17, len(expected) - 1):
        one = [a[..., i:i + 1] if a.dim() > 1 else a[i:i + 1] for a in args]
        r, s = findf_batch(*one)
        for name, v in list(r.items()) + list(s.items()):
            w = whole_r[name] if name in whole_r else whole_s[name]
            assert np.array_equal(w[..., i:i + 1].cpu().numpy(),
                                  v.cpu().numpy()), (
                f"lane {i} {name} depends on its batch neighbours")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None
                    or IFOB_ROWS is None, reason="no CUDA")
def test_cuda_matches_cpu():
    """The ray SELECTION -- which layers are searched, how many rays come out,
    which layer each belongs to -- is comparison and integer arithmetic, so it
    must agree exactly across devices.  Only the geometry may move.
    """
    outs = []
    for device in ("cpu", "cuda"):
        args, _, _ = _batch(device)
        r, s = findf_batch(*args)
        outs.append(({k: v.cpu().numpy() for k, v in r.items()},
                     {k: v.cpu().numpy() for k, v in s.items()}))
    for name in ("iaftxr", "iskp"):
        assert np.array_equal(outs[0][1][name], outs[1][1][name]), \
            f"{name} differs between devices"
    assert np.array_equal(outs[0][0]["imode"], outs[1][0]["imode"]), \
        "IMODE differs between devices"
    for name in ("delfx", "gdflx", "hpflx", "htflx", "fvflx"):
        a, b = outs[0][0][name], outs[1][0][name]
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        sp = np.spacing(np.maximum(np.abs(b), 1.0))
        assert (d / sp).max() <= _BARS[name][0], \
            f"{name} worst {(d / sp).max():.1f} ULP"


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None
                    or IFOB_ROWS is None, reason="no CUDA")
def test_the_cuda_graph_path_is_bit_for_bit_the_eager_path():
    """Capture-and-replay is the same kernels in the same order, so any
    difference means the capture is not running what the eager loop runs.

    It also catches the capture-specific trap: capture needs a warmup, the
    warmup genuinely ADVANCES the state machine, and failing to restore the
    buffers afterwards yields an answer that starts a few steps in.
    """
    args, _, _ = _batch("cuda")
    eager = findf_batch(*args, use_cuda_graph=False)
    graph = findf_batch(*args, use_cuda_graph=True)
    for side in (0, 1):
        for name in eager[side]:
            assert np.array_equal(eager[side][name].cpu().numpy(),
                                  graph[side][name].cpu().numpy()), \
                f"{name} differs between eager and graph"
