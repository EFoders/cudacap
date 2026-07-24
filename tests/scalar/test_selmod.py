"""Stage A SELMOD vs the tier-B oracle dump (controlling sample-area select).

SELMOD is pure integer branching on KFX and FI comparisons, so JMODE is exact.
The 143 cases cover KFX 1..5, the KFX==2 E-critical tiebreak, and the KFX>=3
F2-critical decision with |DELFI| straddling the 0.01 MHz threshold.
"""

import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.selmod import selmod

from .dumputil import load_dump

DUMP = load_dump("selmod")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_selmod_dumps.sh")
def test_selmod_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["jmode"]
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        kfx = int(inp[0])
        fi11, fi12, fi13, fi31, fi33 = (float(x) for x in inp[1:6])
        s = LayerState(kfx=kfx)
        s.fi[1, 1], s.fi[1, 2], s.fi[1, 3] = F32(fi11), F32(fi12), F32(fi13)
        s.fi[3, 1], s.fi[3, 3] = F32(fi31), F32(fi33)
        got = selmod(s)
        assert got == int(exp[0]), (f"case {k}: got {got} expected {exp[0]} "
                                    f"in {inp}")
