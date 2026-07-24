# Stage C conventions — CUDA C, one thread per Fortran call

Stage C compiles the Stage A control flow directly as CUDA C `__device__`
functions, to be composed into one kernel where **one thread runs one whole
LUFFY call** (one circuit-hour), state in registers/local memory. Stage A
(`src/voacap_gpu/scalar/`) is the authoritative source for every port: it is
line-faithful to the Fortran and validated bit-exact (or at the documented
libm floor) against the oracle. **Port Stage A, not the Fortran** — Stage A
has already resolved every Fortran-semantics question (NINT rounding, COMMON
persistence, 1-based padding, dead locals), and the tests hold Stage C to
bit-identity with it.

## The fidelity bar

**Bit-identical to Stage A, per output, per case.** Both sides evaluate
transcendentals in binary64 and round once to binary32, so there is no libm
excuse between them (verified: 100k-sample sweeps of the f32math helpers
agree exactly on this GPU). A test that finds a non-bit-exact cell has found
a porting bug until proven otherwise; the only acceptable residual is a
demonstrated CUDA-vs-numpy binary64 libm divergence, which must be counted,
printed, and asserted rare — never waved through with a tolerance.

## Arithmetic discipline

1. **Every literal is `f`-suffixed.** `2.0 * x` in C promotes to double and
   rounds differently — this is the #1 way to lose bit-exactness. `0.5f * x`.
2. **Compile only through `voacap_gpu.stage_c.compile.load()`**, which pins
   `--fmad=false --prec-div=true --prec-sqrt=true`. Never a bare RawModule.
3. **Transcendentals via `f32math.cuh`** (`cos32`, `exp32`, `pow32`, ...) —
   never `cosf`/`__cosf`/`powf`.
4. **Python `max`/`min` become `pymax`/`pymin`** (NaN: first argument wins),
   never `fmaxf`/`fminf`.
4b. **Division by a constant goes through `div32`.** NVRTC strength-reduces
   `x / 6370.0f` (any compile-time-constant denominator, even through a
   const-propagated local) into multiply-by-reciprocal *despite*
   `--prec-div=true` — 1 ULP wrong, and ill-conditioned callers amplify it.
   Plain `/` is fine only when the denominator is a genuinely runtime value.
   The hazard is identical in binary64: a REAL*8 region dividing by a
   constant needs `div64` (`__ddiv_rn`). Note also that a promoted binary32
   literal (`(double)4.34294f`) and a true binary64 one (`4.34294`) are
   DIFFERENT numbers — GENOIS uses both, in different expressions, and they
   must stay distinct.
5. Fortran `IFIX`/`INT` truncation is C `(int)x`. For anything subtler
   (NINT, MOD on negatives), mirror the exact expression Stage A wrote.
6. Expression shape is part of the answer: keep Stage A's association order
   (`a + b * c / d` stays literal), keep `x * x` as a multiply, keep
   "pointless" double subtractions like `1 - (1 - q)`.

## Data layout

- A numpy `(rows, cols)` array becomes a **flat row-major `float*` with an
  explicit `ncols` parameter**: element `[i, k]` is `p[i * ncols + k]`.
  1-based padding (unused row/column 0) is preserved exactly.
- State dataclasses (`FrqState`, `SonState`, ...) become structs in
  `state.cuh`, same field names, same array shapes flattened row-major,
  fixed-size arrays by value inside the struct. `np.int32` fields are `int`.
- Scalars pass by value; Stage A functions returning tuples return a small
  struct or write through pointers — mirror the Stage A signature otherwise,
  same function name in lower case.

## File layout

- One header per Stage A module: `cuda/<name>.cuh`, include-guarded
  `VOACAP_<NAME>_CUH`, including only `f32math.cuh`/`constants.cuh`/
  `state.cuh` and its declared dependencies.
- Tests in `tests/cuda/test_<name>.py`, following `test_xlin.py` /
  `test_fnorml.py` as the pattern: a tiny `extern "C" __global__` wrapper
  kernel drives the device function once per thread; inputs come from the
  routine's golden dump (`tests/scalar/dumputil.load_dump`) when one
  exists, otherwise from the case generator its Stage A test uses; expected
  values are computed **by calling Stage A in-process** on the same cases;
  the assert is `bit-exact, all cases, all outputs`, plus the branch-coverage
  asserts the Stage A test carries.

## Process rules

- Run tests with `uv run pytest tests/cuda/test_<name>.py -x -q`
  (`uv run python`; there is no bare `python` on this machine).
- Do not run any git command. Do not modify `reference/` (read-only vendored
  Fortran), `scalar/`, `batched/`, or another chunk's files.
- Report measured results honestly: numbers of cases, bit-exact fractions,
  any branch asserted unreachable.
