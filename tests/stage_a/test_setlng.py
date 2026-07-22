"""Stage A SETLNG vs the tier-B oracle dump (sample-area replication).

SETLNG does no arithmetic -- it copies per-area scalars, the layer parameters
FS/FI/YI/HI and the ionogram HPRIM/HTRUE/FVERT/AFAC/IFOB between sample-area
slots, choosing the pattern from KFX.  Each case injects a distinct
exact-integer sentinel (all < 2**24, hence exact in float32) into every
touched cell of areas 1..5, runs SETLNG, and checks 151 witness cells covering
every destination range.  Because the sentinels are exact integers the copy is
bit-exact by construction; the test's real job is pinning the control flow
(the KFX branch) and the deliberately asymmetric index ranges, so every
witness must match exactly.  The injection formula and witness order mirror
tools/oracle/dumps/dump_setlng.f exactly.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_a.setlng import setlng

from .dumputil import load_dump

DUMP = load_dump("setlng")

_SCALARS = ["clat", "clong", "glat", "rd", "gyz", "rat", "gmdip", "clck",
            "abiy", "artic", "sigpat", "epspat", "hs"]
_SCAL_OFF = {name: (i + 1) * 10000 for i, name in enumerate(_SCALARS)}
_LAYERS = ["fs", "fi", "yi", "hi"]
_IONO = ["hprim", "htrue", "fvert", "afac"]
_IFOB_PAIRS = [(1, 1), (40, 1), (1, 30), (40, 30), (20, 15)]


def _inject(s, r, salt):
    for a in range(1, 6):
        for name in _SCALARS:
            getattr(s, name)[a] = F32(_SCAL_OFF[name] + a * 1000 + salt)
        for j, name in enumerate(_LAYERS, start=1):
            arr = getattr(s, name)
            for il in range(1, 4):
                arr[il, a] = F32(j * 100000 + a * 10000 + il * 1000 + salt)
        for j, name in enumerate(_IONO, start=1):
            arr = getattr(s, name)
            for il in range(1, 31):
                arr[il, a] = F32(j * 1000000 + a * 100000 + il * 1000 + salt)
        for il in range(1, 31):
            for ia in range(1, 41):
                r.ifob[ia, il, a] = a * 100000 + il * 1000 + ia * 10 + salt


def _witnesses(s, r):
    got = []
    for name in _SCALARS:
        arr = getattr(s, name)
        got.extend(arr[a] for a in (2, 3, 4, 5))
    for name in _LAYERS:
        arr = getattr(s, name)
        got.extend(arr[il, a] for il in (1, 2, 3) for a in (2, 3, 4, 5))
    for name in _IONO:
        arr = getattr(s, name)
        got.extend(arr[il, a] for il in (1, 15, 30) for a in (2, 3, 4))
    for (ia, il) in _IFOB_PAIRS:
        got.extend(r.ifob[ia, il, a] for a in (2, 3, 4))
    return got


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_setlng_dumps.sh")
def test_setlng_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 151
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        kfx, salt = int(inp[0]), int(inp[1])
        s = LayerState(kfx=kfx)
        r = RaysState()
        _inject(s, r, salt)
        setlng(s, r)
        got = _witnesses(s, r)
        assert len(got) == 151
        for i, (g, e_str) in enumerate(zip(got, exp)):
            e = np.float32(e_str)
            assert np.float32(g) == e, (
                f"case {k} kfx={kfx} salt={salt} col{i + 1:03d}: "
                f"got {g!r} expected {e!r}")
