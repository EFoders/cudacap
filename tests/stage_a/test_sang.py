"""Stage A SANG vs. the tier-B oracle dump.

NANG is an integer table index; the comparisons feeding it sit away from
representability knife-edges in the case grid, so the result must match
the oracle exactly on every case (measured 357/357 at generation).
"""

import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.sang import sang

from .dumputil import load_dump

DUMP = load_dump("sang")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_mod1_dumps.sh")
def test_sang_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["nang"]
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = sang(F32(inp[0]), F32(inp[1]))
        assert got == int(exp[0]), (
            f"case {k}: gcdkm={inp[0]} amind={inp[1]}: "
            f"got {got} expected {exp[0]}")
