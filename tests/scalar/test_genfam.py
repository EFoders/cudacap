"""Stage A GENFAM vs the tier-B oracle dump.

GENFAM is two nested Horner evaluations over FAM(14,12) plus five quartics
over DUD(5,12,5), so the only transcendentals are the ALOG10 that forms the
abscissa and the ``2.**X`` inside it -- everything downstream is
multiply-accumulate.  That makes it a good place to separate table indexing
from arithmetic: if the IBK hemisphere shift or either frequency clamp were
wrong the error would be dB-sized, not ULP-sized, so a tight bar here is what
lets GENOIS's own residual be read as arithmetic rather than bookkeeping.

The Python tables come from redmap() reading the vendored ASCII coefficients,
proven byte-identical to the oracle binaries by test_coeffs.

Measured: 3839/3840 cells bit-exact.  All five decile outputs are exact on
every case -- their only transcendental is the ALOG10 abscissa, which
f32math.log10_32 reproduces exactly here.  The single miss is 1 ULP in FA, on
the ``2.**X`` that forms the second Horner abscissa: powf is the one call in
this routine where glibc's 1-ULP accuracy shows through ([[f32-libm-noise]]).
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.genfam import genfam
from voacap_gpu.scalar.redmap import redmap

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("genfam")

GENFAM_ULP_BAR = 2.0            # measured worst 1.0, on FA alone


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genfam_dumps.sh")
def test_genfam_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["fa", "dua", "dla", "dms", "dus", "dls"]
    total = exact = 0
    worst = 0.0
    worst_at = None
    blocks = set()
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        month, y2, iblk, freq, z = inp
        st = redmap(COEFFS_DIR, "CCIR", int(month), F32(100.0))
        got = genfam(st, F32(y2), int(iblk), F32(freq), F32(z))
        blocks.add(int(iblk) + (6 if float(y2) < 0.0 else 0))
        for name, g, e_str in zip(header, got, exp):
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst:
                worst, worst_at = d, f"case {k} {name} {inp}"
            assert d <= GENFAM_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({d:.2f} floored ULP) in {inp}")
            total += 1
            exact += d == 0.0
    assert blocks == set(range(1, 13)), f"unreached table column: {blocks}"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst:.2f} at {worst_at})")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_frequency_clamps_hold_the_curve_ends():
    """Report 322's curves stop at 20 MHz, and sigma-Fam's at 10.

    Above those the polynomials must be evaluated at the endpoint, not
    extrapolated -- the deciles are flat above 20 MHz and DMS above 10.
    """
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    at20 = genfam(st, F32(0.5), 3, F32(20.0), F32(60.0))
    for freq in (25.0, 40.0, 55.0):
        above = genfam(st, F32(0.5), 3, F32(freq), F32(60.0))
        assert above[1:] == at20[1:], f"deciles moved above 20 MHz at {freq}"
        assert above[0] != at20[0], "FA must still track frequency"
    at10 = genfam(st, F32(0.5), 3, F32(10.0), F32(60.0))
    for freq in (12.0, 18.0):
        above = genfam(st, F32(0.5), 3, F32(freq), F32(60.0))
        assert above[3] == at10[3], f"DMS moved above 10 MHz at {freq}"
        assert above[1] != at10[1], "DUA must still track frequency to 20 MHz"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_southern_hemisphere_shifts_the_table_column():
    """Only the SIGN of Y2 matters, and it selects blocks 7..12."""
    st = redmap(COEFFS_DIR, "CCIR", 2, F32(100.0))
    north = genfam(st, F32(0.01), 2, F32(8.0), F32(70.0))
    assert genfam(st, F32(1.4), 2, F32(8.0), F32(70.0)) == north
    south = genfam(st, F32(-0.01), 2, F32(8.0), F32(70.0))
    assert genfam(st, F32(-1.4), 2, F32(8.0), F32(70.0)) == south
    assert south != north
    # block 2 southern == block 8 northern would be a shift of 6 columns, but
    # GENFAM has no way to ask for column 8 directly -- the DUD row is the
    # observable check that the shift happened at all
    assert south[1] != north[1]
