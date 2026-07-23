"""The Stage C state ABI offset proof: Python packed layout == C struct.

For every struct in state.cuh a generated ``extern "C" __global__`` kernel
reads each member of the struct (through the compiler's own offsets) and
writes it, flattened in declaration order, into a float array.  The test
packs a Stage A instance whose every slot carries a distinct sentinel,
ships the raw bytes to the device, and asserts the readback is bit-identical
to the flattened packed values — so a single mis-ordered, mis-sized or
mis-typed field anywhere in state.cuh / state.py fails loudly.  A second
kernel asserts ``sizeof(<Struct>) == dtype.itemsize`` for every struct, and
a CPU round-trip checks unpack is pack's exact inverse (labels included).
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.allmodestate import AllModeState
from voacap_gpu.stage_a.frqstate import FrqState
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.losxstate import LosxState
from voacap_gpu.stage_a.lpathstate import LpathState
from voacap_gpu.stage_a.modesstate import ModesState
from voacap_gpu.stage_a.mufstate import MufState
from voacap_gpu.stage_a.noisestate import AnoisState, NoiseDiagState
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_a.reflxstate import ReflxState
from voacap_gpu.stage_a.signalstate import SigdState, TonState
from voacap_gpu.stage_a.sonstate import LAYTYP, SonState
from voacap_gpu.stage_a.timestate import TimeState
from voacap_gpu.stage_a.zonstate import ZonState

from voacap_gpu.stage_c.state import (C_RENAMES, DTYPES, LABEL_FIELDS, PACK,
                                      UNPACK)

from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   requires_cuda)

pytestmark = requires_cuda

CLASSES = {
    "AllModeState": AllModeState,
    "FrqState": FrqState,
    "LayerState": LayerState,
    "LosxState": LosxState,
    "LpathState": LpathState,
    "ModesState": ModesState,
    "MufState": MufState,
    "AnoisState": AnoisState,
    "NoiseDiagState": NoiseDiagState,
    "RaysState": RaysState,
    "ReflxState": ReflxState,
    "SigdState": SigdState,
    "TonState": TonState,
    "SonState": SonState,
    "TimeState": TimeState,
    "ZonState": ZonState,
}


def _field_info(dtype, name):
    """(base kind 'f'/'i', subarray shape or (), flattened length)."""
    sub = dtype.fields[name][0]
    if sub.shape:
        return sub.base.kind, sub.shape, int(np.prod(sub.shape))
    return sub.kind, (), 1


def _kernel_source():
    """One read-back kernel per struct plus the sizeof kernel.

    Generated from the dtypes: the kernel indexes the C struct BY MEMBER
    NAME, so the compiler supplies the C offsets — any divergence from the
    Python dtype (order, size, type, or a missing/renamed member) either
    fails to compile or fails the bit-compare.
    """
    lines = ['#include "state.cuh"', ""]
    for sname, dt in DTYPES.items():
        lines.append(f'extern "C" __global__ void read_{sname.lower()}('
                     f'const {sname}* s, float* out) {{')
        lines.append("    int j = 0;")
        for fname in dt.names:
            kind, _, n = _field_info(dt, fname)
            cname = C_RENAMES.get(fname, fname)
            cast = "(float)" if kind == "i" else ""
            if n > 1:
                lines.append(f"    for (int i = 0; i < {n}; ++i) "
                             f"out[j++] = {cast}s->{cname}[i];")
            else:
                lines.append(f"    out[j++] = {cast}s->{cname};")
        lines.append("}")
        lines.append("")
    lines.append('extern "C" __global__ void struct_sizes(int* out) {')
    for i, sname in enumerate(DTYPES):
        lines.append(f"    out[{i}] = (int)sizeof({sname});")
    lines.append("}")
    return "\n".join(lines)


def _fill_sentinels(obj, dtype, labels=()):
    """Distinct sentinel in every 4-byte slot, exactly representable in f32.

    Floats get counter + 0.25 (exact, and never collides with an int slot);
    ints get the counter itself; label lists cycle through -1..4 so every
    LAYTYP index and the empty label are exercised.  The largest struct has
    < 2^22 slots, so every value round-trips through (float) exactly.
    """
    c = 1
    for name in dtype.names:
        kind, shape, n = _field_info(dtype, name)
        vals = np.arange(c, c + n, dtype=np.int64)
        c += n
        if name in labels:
            idx = (vals % 6) - 1                      # -1..4
            setattr(obj, name,
                    ["" if i < 0 else LAYTYP[int(i)] for i in idx])
        elif shape:
            arr = getattr(obj, name)
            if kind == "i":
                arr[...] = vals.reshape(shape).astype(np.int32)
            else:
                arr[...] = (vals.astype(np.float64) + 0.25
                            ).reshape(shape).astype(np.float32)
        elif kind == "i":
            setattr(obj, name, int(vals[0]))
        else:
            setattr(obj, name, np.float32(float(vals[0]) + 0.25))
    return obj


def _expected_flat(buf, dtype):
    """The packed buffer flattened field-by-field, ints cast to float."""
    parts = [np.asarray(buf[name]).ravel().astype(np.float32)
             for name in dtype.names]
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def module():
    return compile_test(_kernel_source())


@pytest.mark.parametrize("sname", list(DTYPES))
def test_struct_layout_bit_exact(module, sname):
    import cupy as cp
    dt = DTYPES[sname]
    labels = LABEL_FIELDS.get(sname, ())
    obj = _fill_sentinels(CLASSES[sname](), dt, labels)
    buf = PACK[sname](obj)
    assert buf.shape == () and buf.dtype == dt

    dev_struct = cp.asarray(np.atleast_1d(buf).view(np.uint8))
    nwords = dt.itemsize // 4
    dev_out = cp.zeros(nwords, dtype=cp.float32)
    module.get_function(f"read_{sname.lower()}")(
        (1,), (1,), (dev_struct, dev_out))
    cp.cuda.runtime.deviceSynchronize()

    expected = _expected_flat(buf, dt)
    assert expected.size == nwords
    assert_bitexact(fetch(dev_out), expected, f"{sname} layout")


def test_struct_sizes(module):
    import cupy as cp
    dev_out = cp.zeros(len(DTYPES), dtype=cp.int32)
    module.get_function("struct_sizes")((1,), (1,), (dev_out,))
    cp.cuda.runtime.deviceSynchronize()
    got = fetch(dev_out)
    expected = np.array([dt.itemsize for dt in DTYPES.values()],
                        dtype=np.int32)
    assert_bitexact(got, expected, "sizeof", cases=list(DTYPES))
    # every struct really is all-4-byte members: itemsize divisible by 4
    assert all(dt.itemsize % 4 == 0 for dt in DTYPES.values())


@pytest.mark.parametrize("sname", list(DTYPES))
def test_pack_unpack_round_trip(sname):
    dt = DTYPES[sname]
    labels = LABEL_FIELDS.get(sname, ())
    src = _fill_sentinels(CLASSES[sname](), dt, labels)
    buf = PACK[sname](src)
    dst = UNPACK[sname](buf, CLASSES[sname]())
    for name in dt.names:
        a, b = getattr(src, name), getattr(dst, name)
        if name in labels:
            assert a == b, (sname, name)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and bool((a == b).all()), (sname, name)
        else:
            assert type(b) in (int, np.float32)
            assert a == b, (sname, name)
    # and the sentinel generator really produced distinct slots (label
    # fields excepted: they only have the six LAYTYP values to draw from)
    parts = [np.asarray(buf[name]).ravel().astype(np.float32)
             for name in dt.names if name not in labels]
    flat = np.concatenate(parts)
    assert np.unique(flat).size == flat.size
