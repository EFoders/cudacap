"""Stage A FDIST vs the tier-B oracle dump (up to six modes per hop).

The dump carries both the /REFLX/ ray table FDIST read and the /MODES/ output,
so the test injects the ray table verbatim rather than rebuilding it through
FINDF/FOBBY -- keeping FDIST's tier-B decoupled from FOBBY's IFOB quantization
floor (see test_findf).  The residual is then only FDIST's own SIN/COS/ATAN
[[f32-libm-noise]] floor on the interpolated modes.

Compares NUMMOD and all six mode slots' DELMOD/HPMOD/HTMOD/FVMOD/ITMOD/AFMOD
(the driver presets the mode arrays so unwritten slots are deterministic).

Measured over 180 cases (205 modes found) = 6480 cells: 99.97% bit-exact,
worst 28 floored ULP.  The two non-exact cells are the known ATAN floor
amplified by CANCELLATION: DELMOD = (PIO2 - PHE - THET)*R2D subtracts operands
of ~1.57 rad to leave ~0.06 rad, a ~25x magnification, so a 1-ULP ATAN
difference lands as ~7e-6 deg; FVMOD inherits it through COS(DELMOD*D2R).
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.fdist import fdist
from voacap_gpu.stage_a.modesstate import ModesState
from voacap_gpu.stage_a.reflxstate import ReflxState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("fdist")

_MODE_ARRAYS = ["delmod", "hpmod", "htmod", "fvmod", "itmod", "afmod"]
_RAY_BASE = 2               # dmaxkm, ghop, then the 7 ray arrays
_OUT_BASE = _RAY_BASE + 7 * 45   # index of nummod


def _run_case(inp, exp):
    amind, freq = float(inp[11]), float(inp[12])
    reflx = ReflxState()
    reflx.dmaxkm[1] = np.float32(exp[0])
    for b, arr in enumerate([reflx.delfx, reflx.gdflx, reflx.hpflx,
                             reflx.htflx, reflx.fvflx, reflx.afflx]):
        base = _RAY_BASE + b * 45
        for j in range(1, 46):
            arr[j, 1] = np.float32(exp[base + j - 1])
    base = _RAY_BASE + 6 * 45
    for j in range(1, 46):
        reflx.imode[j, 1] = int(round(float(exp[base + j - 1])))
    modes = ModesState()
    modes.ghop = np.float32(exp[1])   # GHOP the oracle used
    nummod = fdist(reflx, modes, 1, freq, amind)
    return nummod, modes


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_fdist_dumps.sh")
def test_fdist_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 2 + 7 * 45 + 1 + 6 * 6
    total = exact = 0
    worst = 0.0
    worst_where = ""
    for kk, (inp, exp) in enumerate(zip(inputs, rows)):
        nummod, modes = _run_case(inp, exp)
        e = np.float32(exp[_OUT_BASE])
        assert nummod == int(round(float(e))), (
            f"case {kk} nummod: got {nummod} expected {e!r} in {inp}")
        total += 1
        exact += 1
        arrays = [modes.delmod, modes.hpmod, modes.htmod, modes.fvmod,
                  modes.itmod, modes.afmod]
        for b, arr in enumerate(arrays):
            base = _OUT_BASE + 1 + b * 6
            for j in range(1, 7):
                e = np.float32(exp[base + j - 1])
                d = ulp_diff_floored(arr[j, 1], e)
                if d > worst:
                    worst = d
                    worst_where = (f"case {kk} {_MODE_ARRAYS[b]}[{j}]: "
                                   f"{arr[j, 1]!r} vs {e!r}")
                assert d <= 64, (f"case {kk} {_MODE_ARRAYS[b]}[{j}]: got "
                                 f"{arr[j, 1]!r} expected {e!r} "
                                 f"({d:.2f} floored ULP)")
                total += 1
                exact += d == 0.0
    assert exact / total >= 0.99, (f"exact {exact}/{total} "
                                   f"(worst {worst:.2f} @ {worst_where})")
