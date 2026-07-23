"""The Stage C arithmetic contract, tested directly.

Two separate claims, and the second is the one that guards everybody else:

1. Every f32math.cuh helper is bit-identical to stage_a/f32math.py — the
   f64-evaluate-round-once convention transfers to CUDA's binary64 libm.
2. ``a*b + c`` compiled through compile.load() is NOT contracted to an FMA.
   NVRTC contracts by default; --fmad=false is what turns it off, and this
   sweep is the statistical tripwire: over a million random triples, a
   contracted build disagrees with numpy's separately-rounded result on a
   large fraction of cases, so zero disagreement means the flag held.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a import f32math as fm

from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "f32math.cuh"

extern "C" __global__ void unary(const float* x, float* sin_o, float* cos_o,
                                 float* atan_o, float* exp_o, float* log_o,
                                 float* log10_o, float* sqrt_o, float* asin_o,
                                 float* acos_o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    sin_o[i] = sin32(v);
    cos_o[i] = cos32(v);
    atan_o[i] = atan32(v);
    exp_o[i] = exp32(v);
    float p = fabsf(v) + 1e-6f;         // positive domain for the logs/sqrt
    log_o[i] = log32(p);
    log10_o[i] = log10_32(p);
    sqrt_o[i] = sqrt32(p);
    float u = fmodf(v, 1.0f);           // [-1, 1) domain
    asin_o[i] = asin32(u);
    acos_o[i] = acos32(u);
}

extern "C" __global__ void fmad_probe(const float* a, const float* b,
                                      const float* c, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i] + c[i];
}

extern "C" __global__ void pow_probe(const float* x, const float* y,
                                     float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pow32(x[i], y[i]);
}
'''


def _cases(n=200_000, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-30.0, 30.0, n).astype(np.float32)
    # edge dust: exact zeros, tiny, near pi multiples
    x[:8] = np.float32([0.0, -0.0, 1e-30, -1e-30, 3.1415926, -3.1415926,
                        6.2831853, 1.5707963])
    return x


def test_round_once_transfers_bit_exactly():
    x = _cases()
    n = len(x)
    mod = compile_test(_SRC)
    outs = [np.zeros(n, np.float32) for _ in range(9)]
    dev = launch_rows(mod.get_function("unary"), n, [x] + outs)
    got = [fetch(d) for d in dev[1:10]]

    p = np.float32(np.abs(x)) + np.float32(1e-6)
    u = np.fmod(x, np.float32(1.0))
    pairs = [(fm.sin32, x), (fm.cos32, x), (fm.atan32, x), (fm.exp32, x),
             (fm.log32, p), (fm.log10_32, p), (fm.sqrt32, p),
             (fm.asin32, u), (fm.acos32, u)]
    names = ["sin32", "cos32", "atan32", "exp32", "log32", "log10_32",
             "sqrt32", "asin32", "acos32"]
    for g, (f, arg), name in zip(got, pairs, names):
        assert_bitexact(g, f(arg), name, cases=x)


def test_pow32_matches_stage_a():
    rng = np.random.default_rng(11)
    x = rng.uniform(0.01, 50.0, 100_000).astype(np.float32)
    y = rng.uniform(-3.0, 3.0, 100_000).astype(np.float32)
    y[:2] = np.float32([1.98, 0.5])
    mod = compile_test(_SRC)
    out = np.zeros_like(x)
    dev = launch_rows(mod.get_function("pow_probe"), len(x), [x, y, out])
    assert_bitexact(fetch(dev[2]), fm.pow32(x, y), "pow32")


def test_fmad_is_off():
    rng = np.random.default_rng(3)
    n = 1_000_000
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = rng.standard_normal(n).astype(np.float32)
    mod = compile_test(_SRC)
    out = np.zeros(n, np.float32)
    dev = launch_rows(mod.get_function("fmad_probe"), n, [a, b, c, out])
    want = a * b + c                     # numpy: two separate roundings
    assert_bitexact(fetch(dev[3]), want, "a*b+c (fmad tripwire)")
