"""NVRTC compilation for Stage C — CUDA C, one thread per Fortran call.

Stage C is the third form of each routine: Stage A is the scalar reference,
Stage B the batched-tensor port, and Stage C compiles the Stage A control
flow *directly* as CUDA C, one thread per circuit.  There is no nvcc or MSVC
on this machine; everything compiles at runtime through NVRTC (shipped in the
pip wheels ``cupy-cuda12x[ctk]``), which handles device code only — exactly
what we need, since kernels launch from Python.

THE OPTION SET BELOW IS THE FIDELITY CONTRACT.  ``--fmad=false`` is the one
that bites: NVRTC contracts ``a*b + c`` into a fused multiply-add by
default, which skips the intermediate rounding and silently breaks
bit-exactness against Stage A on essentially every polynomial in the
program.  ``tests/stage_c/test_f32math.py`` carries a statistical tripwire
for it.  Never compile a Stage C source through a bare ``cp.RawModule``.
"""

import hashlib
import pathlib

import cupy as cp

CUDA_DIR = pathlib.Path(__file__).resolve().parent / "kernels"


def _header_fingerprint():
    """One hash over every header, so editing a .cuh invalidates the cache.

    CuPy keys its kernel cache on the SOURCE STRING only — #include contents
    are resolved by NVRTC and never hashed, so an edited header would keep
    serving the stale binary.  Injecting this fingerprint as a comment makes
    the cache key move with the headers.  Content-hashed (not mtime) so a
    checkout/branch switch that restores identical bytes still hits.
    """
    h = hashlib.sha1()
    for p in sorted(CUDA_DIR.glob("*.cuh")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()

# --fmad=false: no FMA contraction (intermediate rounding must happen).
# --prec-div / --prec-sqrt: IEEE-correct f32 division and sqrt (these are the
# defaults today, pinned so a toolchain change cannot move them silently).
STRICT_OPTS = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true",
               "--std=c++17", "-I" + str(CUDA_DIR))


def load(source: str, extra_opts: tuple = ()) -> cp.RawModule:
    """Compile CUDA C with the strict option set; includes resolve in cuda/.

    ``source`` is the full text of a .cu translation unit (usually a handful
    of ``#include "<routine>.cuh"`` lines plus ``extern "C" __global__`` test
    or driver kernels).  CuPy caches the build keyed on source+options, so
    repeated test runs pay compilation once.
    """
    stamped = f"// headers: {_header_fingerprint()}\n" + source
    return cp.RawModule(code=stamped, options=STRICT_OPTS + tuple(extra_opts))
