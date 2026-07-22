"""Stage A SYSSY vs the tier-B oracle dump (system-loss decile table).

SYSSY is a pure table lookup -- the outputs are direct SYS/PERR cells (or the
cells over 1.28) with no transcendentals -- so given the same tables it is
bit-exact.  The Python SYS/PERR come from redmap() reading the vendored ASCII
coefficients (proven byte-identical to the oracle binaries by test_coeffs), so
the test's real job is confirming the KJ/LOL/LD/ND integer index arithmetic
picks the same cells the Fortran does across the 470 cases (both hemispheres,
the local-time block boundaries, NN in {2,5}, and the no-data fallback).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.syssy import syssy

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("syssy")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_syssy_dumps.sh")
def test_syssy_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["fm", "su", "sl", "fmp", "sup", "slp"]
    total = exact = 0
    worst = 0.0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        month, g_rad, pt, nn_s, izf_s = inp
        st = redmap(COEFFS_DIR, "CCIR", int(month), F32(100.0))
        if int(izf_s):  # force the no-data branch without touching the cache
            st = SimpleNamespace(f2d=np.zeros((1, 1, 1), dtype=np.float32),
                                 sys=st.sys, perr=st.perr)
        got = syssy(st, float(g_rad), float(pt), int(nn_s))
        for name, g, e_str in zip(header, got, exp):
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            worst = max(worst, d)
            assert d <= 1, (f"case {k} {name}: got {g!r} expected {e!r} "
                            f"({d:.2f} floored ULP) in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.999, f"exact {exact}/{total} (worst {worst:.2f})"
