"""Shared helpers for Stage C tests: compile, move, fetch, compare.

The pattern every Stage C test follows (see test_xlin.py for the worked
example): build a case set, compute expected values by calling STAGE A
in-process, run the device function once per thread through a tiny
``extern "C" __global__`` wrapper, and assert bit-identity per case per
output.  ``assert_bitexact`` is deliberately unforgiving — a mismatch prints
the worst offenders and fails; tolerances are not a thing in Stage C.
"""

import numpy as np
import pytest

try:
    import cupy as cp
    _HAVE_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                    # pragma: no cover
    cp = None
    _HAVE_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAVE_CUDA, reason="no CUDA device")


def compile_test(source):
    from voacap_gpu.stage_c.compile import load
    return load(source)


def launch_rows(kernel, n, args, block=128):
    """One thread per case: launch `kernel` over n threads with `args`.

    numpy arrays are moved to the device automatically; everything else is
    passed through (wrap scalars in np.int32/np.float32 yourself so the ABI
    is explicit).  Returns the device copies so outputs can be fetched.
    """
    dev = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    grid = ((n + block - 1) // block,)
    kernel(grid, (block,), tuple(dev) + (np.int32(n),))
    cp.cuda.runtime.deviceSynchronize()
    return dev


def fetch(dev_arr):
    return cp.asnumpy(dev_arr)


def assert_bitexact(got, expected, name, cases=None):
    """Require got == expected bitwise, elementwise; report the worst cells."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.shape == expected.shape, (name, got.shape, expected.shape)
    if got.dtype.kind == "f":
        same = got.view(np.uint32) == expected.view(np.uint32)
        # -0.0 vs 0.0 and NaN-vs-same-NaN count as equal only bitwise, which
        # is exactly the standard Stage C holds.
    else:
        same = got == expected
    if bool(same.all()):
        return
    bad = np.flatnonzero(~same.ravel())
    lines = []
    for j in bad[:10]:
        e = expected.ravel()[j]
        g = got.ravel()[j]
        tag = f" case={cases[j]}" if cases is not None and j < len(cases) else ""
        lines.append(f"  [{j}] got {g!r} expected {e!r}{tag}")
    raise AssertionError(
        f"{name}: {bad.size}/{same.size} cells differ from Stage A\n"
        + "\n".join(lines))
