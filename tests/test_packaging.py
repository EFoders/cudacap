"""The package must be self-contained and distributable.

``voacap_gpu`` ships two kinds of non-.py asset that an installed copy cannot
run without: the NVRTC CUDA kernels (``cuda/kernels/*.cuh``) and the CCIR/URSI
coefficient data (``data/coeffs/*.asc``) that ``predict()`` reads to build the
ionosphere.  These tests pin that they are present, that the engine reads the
BUNDLED coefficients rather than the vendored ``reference/`` tree, and that the
bundled copy is byte-identical to the oracle source it was taken from -- so a
future reference update cannot silently desync the shipped data.
"""

import hashlib
import pathlib

import pytest

import voacap_gpu
from voacap_gpu.engine import _COEFFS_DIR

from tests.scalar.dumputil import REFERENCE_ANTENNAS_DIR, REFERENCE_COEFFS_DIR

PKG = pathlib.Path(voacap_gpu.__file__).resolve().parent
# Repo-relative (same root tests/conftest.py checks for the oracle skip).
REFERENCE = REFERENCE_COEFFS_DIR

_MONTHS = [f"coeff{m:02d}.asc" for m in range(1, 13)]
_ATLASES = ["fof2CCIR.asc", "fof2URSI.asc"]


def test_cuda_kernels_are_inside_the_package():
    kernels = list((PKG / "cuda" / "kernels").glob("*.cuh"))
    assert len(kernels) >= 44, (
        f"only {len(kernels)} .cuh kernels under the package; the NVRTC build "
        "compiles these at runtime, so an installed copy needs them all")


def test_coefficients_are_bundled_in_the_package():
    d = PKG / "data" / "coeffs"
    missing = [f for f in _MONTHS + _ATLASES if not (d / f).exists()]
    assert not missing, f"coefficient data not bundled: {missing}"


def test_engine_reads_the_bundled_coefficients_not_reference():
    """predict()'s coeff dir must be inside the package, so reference/ (the
    dev-only oracle tree) is never a runtime dependency."""
    assert _COEFFS_DIR == PKG / "data" / "coeffs", _COEFFS_DIR
    assert _COEFFS_DIR.is_dir()


# SHA-256 of the vendored sources, taken from
# ``git show 477dad7^:reference/voacapl/voacapl/itshfbc/coeffs/<name>``.  The
# oracle test below compares bytes directly when reference/ is present; this
# pin keeps the same guarantee on a clean clone without it.
_SOURCE_SHA256 = {
    "coeff01.asc": "3f3426f9563a3575030a61db6eb74fdc607845a053bd97a3c792078b6ad4207d",
    "coeff02.asc": "1f39682221c1d1205809a30ee75eb8caed15c26b38559f3e331d7a0f5c56eff4",
    "coeff03.asc": "91a5af9b8752a14708b7fd0e93cf199793073dde471ef457d8de15b071d46d72",
    "coeff04.asc": "aac3d66e82eea0f52d8b89e60b27d8d2fe6a25450786b06c84e314265a82473c",
    "coeff05.asc": "2537e08a0107b17416cd97efe1c1d6ff25b5bf7b626b44bc6f71c94fbbb5c74e",
    "coeff06.asc": "d3d78777e03f42021f1c3c3d95dd6afaf59c43150084b78c3d0aa81749aeb9f8",
    "coeff07.asc": "b9e90a9c23d662b28c4efa3ec49ee3740974547610d9651fec671276fd0435c6",
    "coeff08.asc": "c8ee445541102aadcb495da878ac59be3828ff2615566154b8a3db17175f2e12",
    "coeff09.asc": "a3800180fd672911bc10a0ee433f089480fa2ef4c5e4bf1f8fa1b8ff7c1a31b3",
    "coeff10.asc": "27141e321eb6e3706c025d8232fec7d740ff610ff89a9bbd4b89f5cdda10f2f3",
    "coeff11.asc": "82cc57469e7d523a1ae26afff57b5a665eb10fef5f44a26f906d88048d150930",
    "coeff12.asc": "ac7ed6cf1816a3d21e166c8e4559889a9ae240cc16827a5e38e3bf172ac206e7",
    "fof2CCIR.asc": "070f985b4ac4ade091bf2c8631cec710c9d2544e9f82f90054e3f923356ad44f",
    "fof2URSI.asc": "ade385b2da94eab7b893b112be2ecd602da0a338e16973e7476feae28208237b",
}


@pytest.mark.parametrize("name", _MONTHS + _ATLASES)
def test_bundled_coefficients_match_the_pinned_source_digest(name):
    """Byte-identical to the vendored source by SHA-256, so the shipped data
    (which the dump tests now read) cannot drift even without reference/."""
    bundled = (PKG / "data" / "coeffs" / name).read_bytes()
    assert hashlib.sha256(bundled).hexdigest() == _SOURCE_SHA256[name], (
        f"{name} differs from the vendored oracle source -- re-copy it")


@pytest.mark.oracle
@pytest.mark.parametrize("name", _MONTHS + _ATLASES)
def test_bundled_coefficients_match_the_oracle_source(name):
    """Byte-identical to the vendored source, so the shipped data cannot drift
    from what the tier-B dumps were validated against."""
    bundled = (PKG / "data" / "coeffs" / name).read_bytes()
    source = (REFERENCE / name).read_bytes()
    assert bundled == source, f"{name} differs from reference/ -- re-copy it"


_TEST_DATA = pathlib.Path(__file__).resolve().parent / "data"
_SAMPLE_ANTENNAS = ["sample.11", "sample.13", "sample.14", "sample.14x"]


@pytest.mark.oracle
@pytest.mark.parametrize("name", _SAMPLE_ANTENNAS)
def test_copied_sample_antennas_match_the_oracle_source(name):
    """tests/data/antennas holds byte-exact copies of the stock samples the
    oracle read (see tests/data/README.md); keep them from drifting."""
    copied = (_TEST_DATA / "antennas" / name).read_bytes()
    source = (REFERENCE_ANTENNAS_DIR / name).read_bytes()
    assert copied == source, f"tests/data/antennas/{name} differs from reference/"
