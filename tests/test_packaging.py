"""The package must be self-contained and distributable.

``voacap_gpu`` ships two kinds of non-.py asset that an installed copy cannot
run without: the NVRTC CUDA kernels (``cuda/kernels/*.cuh``) and the CCIR/URSI
coefficient data (``data/coeffs/*.asc``) that ``predict()`` reads to build the
ionosphere.  These tests pin that they are present, that the engine reads the
BUNDLED coefficients rather than the vendored ``reference/`` tree, and that the
bundled copy is byte-identical to the oracle source it was taken from -- so a
future reference update cannot silently desync the shipped data.
"""

import pathlib

import pytest

import voacap_gpu
from voacap_gpu.engine import _COEFFS_DIR

PKG = pathlib.Path(voacap_gpu.__file__).resolve().parent
REFERENCE = (PKG.parents[1] / "reference" / "voacapl" / "voacapl" / "itshfbc"
             / "coeffs")

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


@pytest.mark.skipif(not REFERENCE.is_dir(),
                    reason="vendored reference tree absent (fine in a "
                    "distributed install; this guard is dev-only)")
@pytest.mark.parametrize("name", _MONTHS + _ATLASES)
def test_bundled_coefficients_match_the_oracle_source(name):
    """Byte-identical to the vendored source, so the shipped data cannot drift
    from what the tier-B dumps were validated against."""
    bundled = (PKG / "data" / "coeffs" / name).read_bytes()
    source = (REFERENCE / name).read_bytes()
    assert bundled == source, f"{name} differs from reference/ -- re-copy it"
