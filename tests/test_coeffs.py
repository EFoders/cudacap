"""Module 0 ingestion vs. the oracle-generated coefficient binaries.

The Python parsers read the vendored ASCII sources; the golden binaries in
golden/coeffs/ were produced from the SAME sources by voacapl's own
converters compiled with the oracle gfortran.  Every array must round-trip
bit-exact: serializing the parsed arrays in Fortran memory order must
reproduce the binary record payloads byte for byte.
"""

import pathlib

import numpy as np
import pytest

from voacap_gpu.coeffs import (
    COEFF_READS, load_coeff_month, load_fof2_atlas, load_fof2_daily,
)
from voacap_gpu.fortran_io import read_direct_records, read_sequential_records

from tests.scalar.dumputil import COEFFS_DIR, REFERENCE_COEFFS_DIR

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The monthly coeffNN.asc and fof2CCIR/URSI.asc sources are read from the
# package's bundled copy (byte-identical to the vendored ones, pinned by
# test_packaging.py).  fof2daly.asc is not bundled -- predict() never reads it
# -- so the daily test needs the vendored tree and is marked ``oracle``.
REF = COEFFS_DIR
GOLD = ROOT / "golden" / "coeffs"

pytestmark = pytest.mark.skipif(
    not (GOLD / "coeff01.bin").exists(),
    reason="golden binaries missing — run tools/oracle/copy_coeff_bins.sh",
)

# The WRITE statements of coeffbin.f: 9 records, each the concatenation of
# these arrays in Fortran memory order.
BIN_RECORDS = [
    ["ikim"],
    ["fakp", "fakabp"],
    ["dud", "fam", "sys"],
    ["xfm3cf"],
    ["f2d", "perr"],
    ["anew", "bnew", "achi", "bchi", "fakmap", "abmap"],
    ["xesmcf", "xpmap"],
    ["xeslcf", "xesucf"],
    ["xercof"],
]


def fbytes(a: np.ndarray) -> bytes:
    return np.asarray(a).tobytes(order="F")


@pytest.mark.parametrize("month", range(1, 13))
def test_coeff_month_bit_exact(month):
    arrays = load_coeff_month(REF / f"coeff{month:02d}.asc")
    assert arrays["month"] == month
    records = read_sequential_records(GOLD / f"coeff{month:02d}.bin")
    assert len(records) == len(BIN_RECORDS)
    for rec, names in zip(records, BIN_RECORDS):
        built = b"".join(fbytes(arrays[n]) for n in names)
        assert built == rec, f"record {names} differs for month {month}"


@pytest.mark.parametrize("stem", ["fof2CCIR", "fof2URSI"])
def test_fof2_atlas_bit_exact(stem):
    atlas = load_fof2_atlas(REF / f"{stem}.asc")
    assert atlas.shape == (12, 13, 76, 2) and atlas.dtype == np.float32
    records = read_direct_records(GOLD / f"{stem}.da", recl=7904)
    assert len(records) == 12
    for m, rec in enumerate(records):
        assert fbytes(atlas[m]) == rec, f"{stem} month {m + 1} differs"


@pytest.mark.oracle
def test_fof2_daily_bit_exact():
    daily = load_fof2_daily(REFERENCE_COEFFS_DIR / "fof2daly.asc")
    assert daily.shape == (9, 13, 76, 2) and daily.dtype == np.float32
    records = read_sequential_records(GOLD / "fof2daly.bin")
    assert len(records) == 9
    for s, rec in enumerate(records):
        assert fbytes(daily[s]) == rec, f"daily set {s + 1} differs"


def test_known_values():
    """Spot checks pinning the Fortran-order index mapping (values visible
    at the top of coeff01.asc)."""
    a = load_coeff_month(REF / "coeff01.asc")
    assert a["ikim"][0, 0] == 10 and a["ikim"][9, 0] == 2
    assert a["ikim"][0, 1] == 10 and a["ikim"][9, 5] == 4
    assert a["dud"][0, 0, 0] == np.float32(0.60209268)
    total = sum(
        int(np.prod(s)) for _, arrs in COEFF_READS for _, s in arrs
    )
    assert total == 9620  # 60 ints + 9560 floats per month
