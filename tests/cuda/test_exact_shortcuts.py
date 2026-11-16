"""The kernel's bit-exact shortcuts, each proved against the expression it
replaces.

Three performance changes avoid binary64 work without changing a single
result bit.  Each is checked here against the ORIGINAL expression, on the
device, under the strict compile options:

* ``cos32_deg`` (cuda/kernels/angcos.cuh) -- ``cos32(deg * VC_D2R)`` with the
  forty /RAYS/ take-off angles answered from a table.  Checked over ALL 2^32
  binary32 inputs, and the table re-derived entry by entry.
* ``sqrt32`` (cuda/kernels/f32math.cuh) -- the binary32 ``sqrtf`` with the
  binary64 path's NaN bit pattern for negative arguments, against
  ``(float)sqrt((double)x)``, over ALL 2^32 binary32 inputs.
* ``ground_loss_sum`` (cuda/kernels/gain.cuh) -- GAIN's ground-loss DO loop
  with SIN(DELTA) hoisted, areas with identical ground constants reusing one
  evaluation, and the elevation-independent half (the ASIN) memoised across
  calls, against the literal per-area loop, on random states with and
  without repeated ground types.
"""

import numpy as np
import pytest

from tests.cuda.harness import requires_cuda

pytestmark = requires_cuda

_EXHAUSTIVE = r'''
#include "angcos.cuh"
#include "constants.cuh"
#include "f32math.cuh"

__device__ __forceinline__ float sqrt_ref_(float x) {
    return (float)sqrt((double)x);
}

extern "C" __global__ void exhaust(int which, unsigned long long* nbad,
                                   unsigned* first) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x
                           + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long k = g; k < (1ull << 32); k += stride) {
        unsigned u = (unsigned)k;
        float x = __uint_as_float(u);
        float a, b;
        if (which == 0) {
            float rad = x * VC_D2R;
            a = cos32_deg(x, rad);
            b = cos32(rad);
        } else {
            a = sqrt32(x);
            b = sqrt_ref_(x);
        }
        if (__float_as_uint(a) != __float_as_uint(b)) {
            unsigned long long c = atomicAdd(nbad, 1ull);
            if (c < 8) first[c] = u;
        }
    }
}

extern "C" __global__ void table(unsigned* got, unsigned* want) {
    int i = threadIdx.x;
    if (i >= 40) return;
    got[i] = VC_SANG_COS[i];
    want[i] = __float_as_uint(cos32(SANG_ANG[i] * VC_D2R));
}
'''


def _module(src):
    from voacap_gpu.cuda.compile import load
    return load(src)


def _exhaust(which):
    import cupy as cp
    f = _module(_EXHAUSTIVE).get_function("exhaust")
    nbad = cp.zeros(1, np.uint64)
    first = cp.zeros(8, np.uint32)
    f((84 * 64,), (256,), (np.int32(which), nbad, first))
    cp.cuda.runtime.deviceSynchronize()
    n = int(nbad[0])
    return n, [hex(int(v)) for v in cp.asnumpy(first)[:min(n, 8)]]


def test_angle_cosine_table_is_the_device_cosine():
    import cupy as cp
    got = cp.zeros(40, np.uint32)
    want = cp.zeros(40, np.uint32)
    _module(_EXHAUSTIVE).get_function("table")((1,), (64,), (got, want))
    assert np.array_equal(cp.asnumpy(got), cp.asnumpy(want))


def test_cos32_deg_matches_cos32_on_every_binary32():
    n, first = _exhaust(0)
    assert n == 0, f"{n} mismatches, first {first}"


def test_sqrt32_matches_binary64_sqrt_on_every_binary32():
    n, first = _exhaust(1)
    assert n == 0, f"{n} mismatches, first {first}"


_GROUND = r'''
#include "gain.cuh"

extern "C" __global__ void ground(const LayerState* st, const float* delta,
                                  const float* fmc, float* lit, float* sum,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const LayerState* s = &st[i];
    float y = 0.0f;
    for (int ig = 1; ig <= s->km; ++ig)
        y = y + ground_loss(s, ig, delta[i], fmc[i]);
    lit[i] = y;
    // the memo carries across calls, as it does across rays in SETTXR:
    // evaluate a different elevation first, then the one under test
    GroundMemo m;
    ground_memo_init(&m);
    float d0 = delta[(i + 1) % n];
    (void)ground_loss_sum(s, d0, cos32(d0), fmc[i], &m);
    sum[i] = ground_loss_sum(s, delta[i], cos32(delta[i]), fmc[i], &m);
}
'''


@pytest.mark.parametrize("dup", [False, True])
def test_ground_loss_sum_matches_the_literal_loop(dup):
    import cupy as cp
    from voacap_gpu.cuda.state import LAYER_DTYPE
    rng = np.random.default_rng(31 + dup)
    n = 20000
    st = np.zeros(n, LAYER_DTYPE)
    st["km"] = rng.integers(1, 6, n)
    if dup:                     # GEOM's two ground types, plus "no data"
        kinds = np.array([(0.001, 4.0), (5.0, 80.0), (0.0, 0.0)], np.float32)
        pick = rng.integers(0, 3, (n, 6))
        st["sigpat"] = kinds[pick, 0]
        st["epspat"] = kinds[pick, 1]
    else:
        st["sigpat"] = rng.uniform(-0.5, 6.0, (n, 6)).astype(np.float32)
        st["epspat"] = rng.uniform(1.0, 90.0, (n, 6)).astype(np.float32)
    delta = rng.uniform(-0.01, 1.6, n).astype(np.float32)
    delta[::97] = 0.0
    fmc = rng.uniform(1.0, 30.0, n).astype(np.float32)
    lit = cp.zeros(n, np.float32)
    got = cp.zeros(n, np.float32)
    f = _module(_GROUND).get_function("ground")
    f(((n + 127) // 128,), (128,),
      (cp.asarray(st.view(np.uint8)), cp.asarray(delta), cp.asarray(fmc),
       lit, got, np.int32(n)))
    a, b = cp.asnumpy(lit), cp.asnumpy(got)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
