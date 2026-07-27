"""SANG Stage C vs Stage A — the take-off angle count NANG.

SANG is a bucketed table lookup on great-circle distance with a minimum
take-off angle, so the interesting cases are the bucket edges: the
truncation of ``GCDKM/2000. + 1.`` turns a 1-ULP difference into a whole
different NANG, which is why the port divides through div32.
"""

import numpy as np

from voacap_gpu.scalar.sang import sang

from tests.cuda.harness import (compile_test, fetch, launch_rows,
                                   requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "sang.cuh"

extern "C" __global__ void drive(const float* gcdkm, const float* amind,
                                 int* nang, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nang[i] = sang(gcdkm[i], amind[i]);
}
'''


def test_sang_bit_exact():
    rng = np.random.default_rng(11)
    # a dense sweep plus every 2000 km bucket edge, exactly and +-1 ULP
    edges = []
    for k in range(0, 22):
        e = np.float32(k * 2000.0)
        edges += [e, np.nextafter(e, np.float32(0)),
                  np.nextafter(e, np.float32(1e9))]
    gcd = np.concatenate([
        np.array(edges, np.float32),
        rng.uniform(0.0, 42000.0, 20_000).astype(np.float32),
    ])
    amind = np.concatenate([
        np.full(len(edges), 3.0, np.float32),
        rng.uniform(0.0, 30.0, 20_000).astype(np.float32),
    ])

    expected = np.array([sang(g, a) for g, a in zip(gcd, amind)], np.int32)

    mod = compile_test(_SRC)
    out = np.zeros(len(gcd), np.int32)
    dev = launch_rows(mod.get_function("drive"), len(gcd),
                      [gcd, amind, out])
    got = fetch(dev[2])

    bad = np.nonzero(got != expected)[0]
    assert bad.size == 0, (
        f"SANG differs on {bad.size}/{len(gcd)} cases; first: "
        f"gcdkm={gcd[bad[0]]!r} amind={amind[bad[0]]!r} "
        f"got={got[bad[0]]} expected={expected[bad[0]]}")

    # the cases must actually span several buckets, not just one
    assert len(set(expected.tolist())) > 3, "case set spans several NANG values"
