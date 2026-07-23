"""Stage B batched FDIST vs the SAME tier-B oracle dump Stage A uses.

All 180 dump cases are loaded as 180 lanes of one batch, so the lockstep state
machine is checked against the Fortran on a batch whose lanes genuinely
diverge -- different numbers of modes, different table cursors, different exit
points.  That is the property worth testing: a lockstep port that quietly
kept every lane in step would pass a uniform batch and fail this one.

Comparing against the oracle rather than against Stage A also means Stage B is
not inheriting Stage A's answer; the two are independent transliterations
checked against the same third thing.

MEASURED: 6,406/6,480 exact on CPU (98.86%), 6,432/6,480 on CUDA (99.26%).
Worst 115 floored ULP, and ONLY EVER in DELMOD and FVMOD -- the two quantities
that go through the cancellation.  DELMOD = (PIO2 - PHE - THET)*R2D subtracts
two ~1.57 rad operands to leave ~0.06, a ~25x magnification of the ATAN floor,
and FVMOD inherits it through COS(DELMOD*D2R).  Every other column is
bit-exact on every lane.

115 ULP sounds alarming and is not: at a take-off angle of 3.3 degrees it is
2.7e-5 degrees.  The absolute bar below says that in the units that mean
something, since a ULP count on an angle near zero is a poor guide.

Stage A measures 28 ULP worst on this same dump against the same oracle; the
difference is torch's ATAN against glibc's, not a difference in the port.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_b.fdist import NMODE, NRAY, fdist_batch

from tests.stage_a.dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("fdist")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_RAY_BASE = 2                       # dmaxkm, ghop, then the 7 ray arrays
_OUT_BASE = _RAY_BASE + 7 * NRAY    # index of nummod
_NAMES = ["delmod", "hpmod", "htmod", "fvmod", "itmod", "afmod"]
ULP_BAR = 256.0                     # measured worst 115 (CPU)
ABS_BAR = 1.0e-4                    # ... which is 2.7e-5 deg of take-off angle
EXACT_FLOOR = 0.98                  # measured 0.9886 (CPU), 0.9926 (CUDA)


def _load(device):
    """The whole dump as one batch: (46, 180) ray tables and (180,) scalars."""
    _, inputs, rows = DUMP
    n = len(rows)
    ray = {name: torch.zeros((NRAY + 1, n), dtype=torch.float32)
           for name in ("delfx", "gdflx", "hpflx", "htflx", "fvflx", "afflx")}
    imode = torch.zeros((NRAY + 1, n), dtype=torch.float32)
    dmaxkm = torch.zeros(n, dtype=torch.float32)
    ghop = torch.zeros(n, dtype=torch.float32)
    freq = torch.zeros(n, dtype=torch.float32)
    amind = torch.zeros(n, dtype=torch.float32)
    for i, (inp, exp) in enumerate(zip(inputs, rows)):
        amind[i] = float(inp[11])
        freq[i] = float(inp[12])
        dmaxkm[i] = float(exp[0])
        ghop[i] = float(exp[1])
        for b, name in enumerate(("delfx", "gdflx", "hpflx", "htflx",
                                  "fvflx", "afflx")):
            base = _RAY_BASE + b * NRAY
            ray[name][1:, i] = torch.tensor(
                [float(exp[base + j]) for j in range(NRAY)])
        base = _RAY_BASE + 6 * NRAY
        imode[1:, i] = torch.tensor(
            [round(float(exp[base + j])) for j in range(NRAY)])
    args = [ray["delfx"], ray["hpflx"], ray["htflx"], ray["gdflx"],
            ray["fvflx"], ray["afflx"], imode, dmaxkm, ghop, freq, amind]
    return [t.to(device) for t in args], rows


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_fdist_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_fdist_batch_vs_oracle(device):
    args, rows = _load(device)
    out = fdist_batch(*args)
    delmod, hpmod, htmod, fvmod, afmod, itmod, nummod = (
        t.cpu().numpy() for t in out)
    arrays = [delmod, hpmod, htmod, fvmod, itmod, afmod]

    total = exact = 0
    worst = 0.0
    where = ""
    lanes_by_nummod = {}
    for i, exp in enumerate(rows):
        want = int(round(float(exp[_OUT_BASE])))
        assert int(nummod[i]) == want, (
            f"[{device}] lane {i} NUMMOD: {int(nummod[i])} vs {want}")
        lanes_by_nummod[want] = lanes_by_nummod.get(want, 0) + 1
        for b, arr in enumerate(arrays):
            base = _OUT_BASE + 1 + b * NMODE
            for j in range(1, NMODE + 1):
                e = np.float32(exp[base + j - 1])
                d = ulp_diff_floored(arr[j, i], e)
                if d > worst:
                    worst, where = d, (f"lane {i} {_NAMES[b]}[{j}]: "
                                       f"{arr[j, i]!r} vs {e!r}")
                assert d <= ULP_BAR and abs(float(arr[j, i]) - float(e)) <= ABS_BAR, (
                    f"[{device}] lane {i} {_NAMES[b]}[{j}]: {arr[j, i]!r} "
                    f"vs {e!r} ({d:.1f} floored ULP, "
                    f"{abs(float(arr[j, i]) - float(e)):.2e} abs)")
                if d:
                    assert _NAMES[b] in ("delmod", "fvmod"), (
                        f"[{device}] {_NAMES[b]} drifted; only the two "
                        "quantities behind the ATAN cancellation should")
                total += 1
                exact += d == 0.0
    # the batch has to actually diverge, or lockstep proves nothing
    assert len(lanes_by_nummod) >= 3, (
        f"[{device}] the batch barely diverges: mode counts "
        f"{lanes_by_nummod}")
    assert exact / total >= EXACT_FLOOR, (
        f"[{device}] exact {exact}/{total}, worst {worst:.1f} ULP at {where}")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_the_lockstep_batch_matches_running_the_lanes_one_at_a_time():
    """A lane's answer must not depend on who it shares a batch with.

    This is the failure mode lockstep invites: a cursor or mask that leaks
    across lanes looks fine on a uniform batch and wrong on a mixed one.
    """
    args, rows = _load("cpu")
    whole = [t.cpu().numpy() for t in fdist_batch(*args)]
    for i in (0, 7, 41, len(rows) - 1):
        single = [t.cpu().numpy() for t in
                  fdist_batch(*[a[..., i:i + 1] for a in args])]
        names = ["delmod", "hpmod", "htmod", "fvmod", "afmod", "itmod",
                 "nummod"]          # the RETURN order, not the dump's
        for name, w, s in zip(names, whole, single):
            assert np.array_equal(w[..., i:i + 1], s), (
                f"lane {i} {name} depends on its batch neighbours")


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """The mode SELECTION -- which rays bracket the hop, how many modes come
    out -- is pure comparison and must agree exactly.  Only the interpolated
    values may move, and then by the trig floor."""
    outs = []
    for device in ("cpu", "cuda"):
        args, _ = _load(device)
        outs.append([t.cpu().numpy() for t in fdist_batch(*args)])
    assert np.array_equal(outs[0][6], outs[1][6]), "NUMMOD differs"
    assert np.array_equal(outs[0][5], outs[1][5]), "ITMOD differs"
    for name, a, b in zip(_NAMES, outs[0], outs[1]):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        sp = np.spacing(np.maximum(np.abs(b), 1.0))
        assert (d / sp).max() <= ULP_BAR, f"{name} worst {(d / sp).max():.1f}"


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_the_cuda_graph_path_is_bit_for_bit_the_eager_path():
    """Capturing the step and replaying it is the same kernels in the same
    order, so there is no room for a numerical difference -- any would mean
    the capture is not running what the eager loop runs.

    It is also the test that catches the capture-specific trap: CUDA graph
    capture needs a warmup, the warmup genuinely ADVANCES the state machine,
    and forgetting to restore the buffers afterwards yields an answer that
    starts a few steps in.  That failure is invisible on a batch whose lanes
    all finish early and obvious here.
    """
    args, _ = _load("cuda")
    eager = [t.cpu().numpy() for t in fdist_batch(*args, use_cuda_graph=False)]
    graph = [t.cpu().numpy() for t in fdist_batch(*args, use_cuda_graph=True)]
    names = ["delmod", "hpmod", "htmod", "fvmod", "afmod", "itmod", "nummod"]
    for name, e, g in zip(names, eager, graph):
        assert np.array_equal(e, g), f"{name} differs between eager and graph"
