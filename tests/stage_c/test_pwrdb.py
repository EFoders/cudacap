"""PWRDB Stage C vs Stage A — transmitter power lookup.

No golden dump exists (PWRDB has no floating-point arithmetic, only
comparisons and a table read), so the case set is the Stage A test's scenario
matrix — the 30 dBW default, receiver entries skipped, first match wins,
inclusive band edges, and the 2-30 MHz clamp applying to the range test only
— each driven across a shared frequency sweep, bit-exact against Stage A
in-process.
"""

import numpy as np

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.pwrdb import pwrdb

from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "pwrdb.cuh"

extern "C" __global__ void drive(const int* iats, const float* xfqs,
                                 const float* xfqe, const float* pwrdba,
                                 int numants, const float* freq, float* out,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pwrdb(iats, xfqs, xfqe, pwrdba, numants, freq[i]);
}
'''


def _cant(*entries) -> CantennaState:
    """entries: (iats, xfqs, xfqe, pwrdba) tuples, installed from index 1."""
    c = CantennaState()
    for i, (iats, lo, hi, p) in enumerate(entries, start=1):
        c.iats[i] = iats
        c.xfqs[i] = F32(lo)
        c.xfqe[i] = F32(hi)
        c.pwrdba[i] = F32(p)
    c.numants = len(entries)
    return c


# the Stage A test's scenarios, one per assertion family it carries
SCENARIOS = [
    ("empty", CantennaState()),
    ("out_of_range", _cant((1, 2.0, 10.0, 47.0))),
    ("rx_then_tx", _cant((2, 2.0, 30.0, 55.0), (1, 2.0, 30.0, 43.0))),
    ("rx_only", _cant((2, 2.0, 30.0, 55.0))),
    ("first_match", _cant((1, 2.0, 15.0, 41.0), (1, 2.0, 30.0, 52.0))),
    ("inclusive_edges", _cant((1, 5.0, 15.0, 44.0))),
    ("clamp_full_band", _cant((1, 2.0, 30.0, 46.0))),
    ("clamp_miss_low", _cant((1, 3.0, 30.0, 46.0))),
    ("clamp_miss_high", _cant((1, 2.0, 25.0, 46.0))),
]

# every frequency the Stage A assertions probe, plus a random sweep
_EDGES = [0.5, 1.9, 2.0, 4.999, 5.0, 10.0, 14.0, 15.0, 15.001, 20.0,
          25.0, 30.0, 45.0]


def _freqs():
    rng = np.random.default_rng(11)
    return np.concatenate([np.float32(_EDGES),
                           rng.uniform(0.1, 45.0, 2000).astype(np.float32)])


def test_pwrdb_bit_exact_across_scenarios():
    freqs = _freqs()
    mod = compile_test(_SRC)
    kernel = mod.get_function("drive")
    seen_default = seen_power = False
    for name, cant in SCENARIOS:
        expected = np.array([pwrdb(cant, f) for f in freqs], np.float32)
        na = cant.numants + 1
        out = np.zeros_like(freqs)
        dev = launch_rows(kernel, len(freqs),
                          [np.ascontiguousarray(cant.iats[:na]).astype(
                               np.int32),
                           np.ascontiguousarray(cant.xfqs[:na]),
                           np.ascontiguousarray(cant.xfqe[:na]),
                           np.ascontiguousarray(cant.pwrdba[:na]),
                           np.int32(cant.numants), freqs, out])
        assert_bitexact(fetch(dev[6]), expected, f"pwrdb[{name}]", cases=freqs)
        seen_default |= bool((expected == np.float32(30.0)).any())
        seen_power |= bool((expected != np.float32(30.0)).any())
    # the scenario matrix must reach both the fallback and a table hit
    assert seen_default and seen_power


def test_pwrdb_branch_values():
    """The Stage A test's pointwise claims, replayed through the kernel."""
    mod = compile_test(_SRC)
    kernel = mod.get_function("drive")

    def run(cant, freq):
        na = cant.numants + 1
        out = np.zeros(1, np.float32)
        dev = launch_rows(kernel, 1,
                          [np.ascontiguousarray(cant.iats[:na]).astype(
                               np.int32),
                           np.ascontiguousarray(cant.xfqs[:na]),
                           np.ascontiguousarray(cant.xfqe[:na]),
                           np.ascontiguousarray(cant.pwrdba[:na]),
                           np.int32(cant.numants),
                           np.float32([freq]), out])
        return fetch(dev[6])[0]

    assert run(CantennaState(), 14.0) == np.float32(30.0)
    assert run(_cant((1, 2.0, 10.0, 47.0)), 20.0) == np.float32(30.0)
    # receiver entries are skipped
    assert run(_cant((2, 2.0, 30.0, 55.0), (1, 2.0, 30.0, 43.0)),
               14.0) == np.float32(43.0)
    assert run(_cant((2, 2.0, 30.0, 55.0)), 14.0) == np.float32(30.0)
    # first matching entry wins
    two = _cant((1, 2.0, 15.0, 41.0), (1, 2.0, 30.0, 52.0))
    assert run(two, 10.0) == np.float32(41.0)
    assert run(two, 20.0) == np.float32(52.0)
    # inclusive at both ends
    mid = _cant((1, 5.0, 15.0, 44.0))
    assert run(mid, 5.0) == np.float32(44.0)
    assert run(mid, 15.0) == np.float32(44.0)
    assert run(mid, 4.999) == np.float32(30.0)
    assert run(mid, 15.001) == np.float32(30.0)
    # the clamp applies to the range test, never the result
    full = _cant((1, 2.0, 30.0, 46.0))
    assert run(full, 0.5) == np.float32(46.0)
    assert run(full, 1.9) == np.float32(46.0)
    assert run(full, 45.0) == np.float32(46.0)
    assert run(_cant((1, 3.0, 30.0, 46.0)), 0.5) == np.float32(30.0)
    assert run(_cant((1, 2.0, 25.0, 46.0)), 45.0) == np.float32(30.0)
