# VOACAP-GPU Rewrite Plan

Status: draft for review. Nothing in `reference/` is modified by this plan; a new
implementation is built alongside it.

## 0. Decisions locked in for this plan

- **Primary target: point-to-point (P2P) mode first.** Matches the existing oracle
  test case (`reference/voacapl/tests/p2p/test01.dat` / `test01.out`). Area coverage
  mode (the actual GPU payoff) comes after P2P is validated end-to-end.
- **Fidelity bar: bit-exact (practical interpretation and precision strategy below).**
- **Oracle build: WSL + gfortran**, building `reference/voacapl` unmodified as the
  ground-truth binary.
- **Antenna modeling: precomputed static data only.** The method-of-moments wire
  solver (`hfmufesw/`, `wp10dwin/curtain*`, `mutual.for`, `resist.for`, `react.for`,
  `sim.for`) is out of scope. We port only the runtime gain-lookup/interpolation path
  (`wp10dwin/gainterp.for`, `gainrel.for`, `readant.for`, etc.) and consume the
  shipped pattern files under `reference/itshfbc/antennas/default/*.VOA` as fixed
  input data.
- **Framework: PyTorch, not JAX.** Switched from the initial proposal — more
  mature documentation/ecosystem, and no functional-purity constraints to fight
  while transliterating ~130 files of COMMON-block-coupled Fortran. Batching
  across circuits/hours/frequencies later uses ordinary tensor broadcasting or
  `torch.vmap`; it isn't a hard requirement to write everything as pure functions
  from day one the way JAX would demand.
- **Precision: mixed, mirroring the Fortran's own per-module choices — not
  blanket float64.** This reverses the original assumption that "bit-exact"
  implied float64 everywhere; see below.

### On "bit-exact"

True ULP-for-ULP matching across gfortran's math library and PyTorch/CUDA's GPU
math intrinsics (sin/cos/atan/log/exp) is not guaranteed at any precision —
different libraries round transcendental functions differently, regardless of
how carefully the code is translated. The practical, achievable, and actually
meaningful bar is:

- **Output-level:** byte-for-byte identical printed tables (VOACAP's output is
  fixed-format text, e.g. `F5.2`, `F9.2` — a handful of decimal digits per field).
  This is a real, checkable, binary pass/fail bar and it's what an operator
  reading the report actually cares about.
- **Internal-value level:** intermediate quantities (MUF, angles, field
  strength, loss terms) matched to a tight relative tolerance rather than
  assumed bit-identical — target ~1e-6 relative for float32-computed
  quantities (a few ULP), ~1e-9 for the float64 exceptions below. Both are
  tighter than anything that could ever show up in the printed output, so this
  is belt-and-suspenders, not the actual pass/fail gate.

**Empirical confirmation (2026-07-21):** the upstream `tests/p2p/test01.out`
fixture turns out to have been generated on Windows by a different compiler
lineage (backslash antenna paths inside the output text give it away). Our
gfortran 15 oracle build matches it except for ±1 in the last printed digit
in roughly 1% of numeric cells — exactly the cross-compiler REAL*4 rounding
signature predicted above, observed before a single line was ported. The
project therefore pins its own oracle outputs (under `golden/`, with the
toolchain recorded in `META.txt`) as the byte-exact bar; the upstream fixture
is advisory only, and `tools/oracle/build_oracle.sh` enforces exactly this:
determinism check, byte-exact match against the pinned fixture, advisory
diff against upstream.

### Precision strategy: mirror the Fortran, don't blanket-upgrade

You asked whether 32 bits will be enough precision. Short answer: **yes — and
we know that because the reference Fortran already thinks so.** VOACAP's own
author made that call decades ago, deliberately, in the specific places where
it mattered. Checked across all 188 `.for`/`.f95` files in `voacapl/src`:

- Only **three files** — `voacapw/dazel.for`, `voa_lib/dazel1.for`,
  `wp10dwin/dazel0.for` (three call-mode variants of the same great-circle
  geometry routine, DAZEL) — declare `IMPLICIT DOUBLE PRECISION` for their
  entire working set. The routine's own header comment explains why: its
  inputs/outputs are single precision ("ALL OF THE ABOVE PARAMETERS ... ARE
  SINGLE PRECISION"), but internally it needs double precision because the
  spherical-trig formulas it uses lose precision catastrophically for
  near-coincident points — the same comment warns the routine is
  "unpredictable for separations less than 0.00005 degrees, about 5 meters"
  *even in double precision*. That's a textbook catastrophic-cancellation
  conditioning issue, not a "the physics needs more digits" issue.
- A small number of individually-flagged scalars/temporaries elsewhere use
  `REAL*8` for specific, named reasons: `versy.for` casts a few variables to
  `real*8` with the inline comment "to solve floating point underflows";
  `genois.for` uses `real*8` for a handful of noise-variance intermediate
  terms; a few COMMON-block fields in `fice_ssn.hdr` (`flat`, `flon`, `fof2` —
  a 5-element station-coordinate/foF2 table) are `REAL*8` while their
  neighbors in the same header are `REAL*4`. `csz1.for` (`hfmufesw`, the
  antenna method-of-moments solver) also uses `real*8` — already out of scope
  per the antenna decision above.
- **Everything else** — the CCIR/URSI ionospheric coefficient evaluation, the
  MUF/FOT mode search proper, absorption and field-strength/loss budgeting,
  the noise model apart from those specific temporaries, and the reliability
  statistics — is `REAL*4` (Fortran's default `REAL`), by the original
  author's explicit choice, running in operational use for decades.

So the plan is a **mixed-precision port that mirrors this exactly**: `float32`
by default everywhere, `float64` only for DAZEL/DAZEL0/DAZEL1 and the handful
of named scalars above. This isn't just "probably good enough" — it's actually
the *more* faithful choice for the bit-exact goal, not a compromise against it:
matching a `REAL*4` Fortran calculation means replicating single-precision
rounding at every step, not avoiding it by computing in higher precision and
hoping it happens to round the same way at the end. Blanket float64 could
plausibly make our numbers *more correct* while making them match gfortran's
printed output *less* often — the opposite of the actual goal.

This also lines up with the hardware reality: consumer NVIDIA GPUs (GeForce/
RTX) deliberately throttle FP64 throughput to a fraction of FP32 (commonly
1/32-1/64), a CUDA hardware limit that applies regardless of PyTorch vs. JAX.
Blanket float64 would have quietly undercut most of the speedup this whole
project is for. Precision fidelity and GPU throughput point the same direction
here instead of trading off against each other.

The three DAZEL variants collapse into one ported module anyway (Section 5,
Group 1) — same routine, different mode argument — so in practice this is
"one geometry module in float64, everything else float32," not a sprawling
mixed-precision bookkeeping problem.

## 1. Goals / non-goals

**Goals**
- A PyTorch implementation of VOACAP P2P prediction, validated against the
  gfortran oracle on a broad matrix of inputs, structured so batching (via
  tensor broadcasting or `torch.vmap`) over hours/frequencies/months/circuits
  is natural — setting up Area mode as a near-term follow-on, not a rewrite.
- A regression test suite durable enough to catch translation errors module by
  module, not just at the end-to-end level.

**Non-goals (for this phase)**
- Area coverage mode itself (planned for, not built yet).
- Antenna pattern synthesis (MoM solver).
- GUI / area-map plotting / ICEPAC or other non-VOACAP tools bundled in `itshfbc/`.
- Further precision tuning beyond mirroring the Fortran's per-module choices
  (Section 0) — e.g. chasing float16/bfloat16, or trying to shrink the
  float64 exceptions (DAZEL, the `versy.for`/`genois.for` temporaries) down to
  float32. The mixed-precision baseline is adopted now; squeezing it further
  is a later exercise if profiling ever shows it's worth it.

## 2. Repository & environment setup

- `git init` the project (it currently isn't a repo). `reference/` is vendored,
  read-only, and untouched by the port — keep it committed for reproducibility of
  the oracle, with its own license files intact.
- **License: RESOLVED — unlicensed for now** (user decision 2026-07-21,
  reversing a brief same-day GPL choice). No LICENSE file ships; default
  all-rights-reserved applies. For the record, the upstream situation imposes
  nothing on this choice: the original VOACAP code is a US Government
  (NTIA/ITS) work, public domain in the US; J.A. Watson's voacapl port
  modifications are CC0; only `dst2csv.f90`/`dst2ascii.f90` (export utilities
  we don't use) are GPLv3. So the license can be freely revisited later, and
  there is no obligation inherited from transliterating the reference closely.
- WSL: install gfortran, build `voacapl` per its own README (`automake
  --add-missing && autoreconf && ./configure && make`), confirm `test01.dat` run
  reproduces `test01.out` exactly. This is step zero — if we can't reproduce the
  existing oracle output, nothing downstream is trustworthy.
- Python side: `src`-layout package, `pyproject.toml`, PyTorch (CUDA build) +
  NumPy + pytest. No global dtype flag needed — PyTorch tensors are float32 by
  default already, and dtype is set explicitly per module per the
  mixed-precision strategy in Section 0.

## 3. Testing methodology

Three tiers, from coarse to fine-grained:

**A. End-to-end golden files (the safety net).**
A generator script produces a matrix of VOACAP input decks (card-image format
per `test01.dat`) sweeping: frequency (2-30 MHz steps), hour (0-23), month (1-12),
sunspot number (0, 10, 50, 100, 150, 200), path length/bearing (short <500km,
medium, long >10000km, crossing the equator vs. not, both hemispheres, polar
paths), antenna selection (a few from `antennas/default`), noise level, and
output method/table selection. Each deck is run through the WSL oracle to produce
a reference `.out`. These (input, output) pairs are committed as fixtures. This
is the regression suite that must stay green forever.

**B. Module-level unit tests (the actual debugging tool).**
End-to-end diffs alone won't be enough to debug a ~15k-line port — when the
final table mismatches, tracing it back through a dozen COMMON-block-coupled
subroutines is exactly the kind of debugging that stalls this kind of project.
Instead, for each module in Section 5, before porting: write a small Fortran test
driver (or instrument the existing routine with a debug dump to a side file)
that calls the routine directly with controlled inputs and records its outputs
(e.g., call `HFMUFS` for one hour/frequency and dump MUF, FOT, mode, hop count).
Each ported module is validated against its own oracle dump *before* being wired
into the full chain.

**C. Property tests (catches what the matrix didn't happen to cover).**
Physics invariants independent of the Fortran oracle: great-circle bearing
symmetry, monotonic absorption vs. frequency, reciprocity of a circuit run
forwards vs. backwards, etc. Cheap insurance against a translation bug that
happens to pass every golden-file case in the matrix.

**Comparison harness.** A Python runner that: invokes the oracle (via WSL
subprocess) and the Python/PyTorch implementation on the same input, parses both
fixed-width output tables into structured records, and diffs field-by-field —
exact string match on printed fields by default (satisfies the output-level bit-
exactness bar), with a documented tolerance escape hatch only where a field is
provably affected by transcendental-function library differences (Section 0).

## 4. Two-stage technique per module

Given the fidelity bar and the COMMON-block-heavy source, each module is ported
in two stages rather than going straight to PyTorch tensors:

- **Stage A — scalar Python/NumPy transliteration.** Line-for-line translation
  preserving Fortran control flow, loop structure, and arithmetic order (not a
  "clean" refactor). Validated against the module's oracle dump (tier B) to
  near machine precision. This isolates "is the physics/algorithm right" from
  "is the vectorization right" — trying to debug both at once against 15k lines
  of COMMON-block state is how this kind of project stalls.
- **Stage B — PyTorch vectorization.** Replace Python control flow with
  tensorized equivalents (`torch.where`, masked updates, fixed-trip-count loops
  over pre-allocated tensors — mirroring the Fortran's own hard-coded iteration
  limits), replace COMMON-block/mutable state with explicit function arguments
  and return values, and add a batch dimension (even if unused immediately) so
  broadcasting/`torch.vmap` over circuits/hours/frequencies works later
  without another rewrite. Dtype follows the per-module precision map (Section
  0) — float32 by default, float64 only where the Fortran itself uses it.
  Validated against Stage A (should match near machine precision) *and*
  against the same oracle dump.

Stage A implementations are kept in the test suite permanently as a continuous
cross-check against Stage B, not deleted once Stage B passes — any future
refactor of the PyTorch side gets caught against a second independent
implementation, cheaply.

## 5. Module inventory & port order

Bottom-up by dependency, so each stage only depends on already-validated code.

| # | Group | Key files | Notes |
|---|-------|-----------|-------|
| 0 | Data ingestion | `voacapl/itshfbc/coeffs/*.asc` (coeff01-12, fof2CCIR, fof2URSI, fof2daly), `CGLALO.DAT`, `antennas/default/*.VOA` | De-risked during setup: voacapl ships the CCIR/URSI coefficients as **ASCII** sources plus the converter programs (`coeffbin.f`, `coefbinw.f`, `fof2bin.f`, `fof2binw.f`) that generate the runtime `.bin` files at install time — the converters are the format spec. We parse the ASCII directly and skip the platform-dependent binary format entirely (using the oracle-generated `.bin` only as a cross-check). Output: versioned `.npz` arrays. |
| 1 | Math/geometry utils | `voa_lib/*`, `distxy.for`, `dazel.for`, `cang.for`, `sang.for`, `geom.for` | Pure math, no ionospheric physics. Warm-up module for the Stage A/B workflow. `dazel.for` (+ its `dazel0.for`/`dazel1.for` variants) is the one piece of this group that stays float64, matching the Fortran's own `IMPLICIT DOUBLE PRECISION` — see Section 0. Everything else here is float32. |
| 2 | Ionospheric coefficients | `lecden.for`, `genfam.for`, `genion.for`, `ionset.for`, `hfciraf.for` | Spherical-harmonic-like expansion of the CCIR/URSI numerical maps (foF2, M(3000)F2, foE). Highest translation risk in the whole project — dense numerical summation, easy to get subtly wrong and still look plausible. Also the best vmap candidate later (evaluated at every lat/lon/month/hour grid point). |
| 3 | MUF/FOT & mode search | `hfmufs.for`, `hfmufs2.for`, `inmuf.for`, `curmuf.for`, `nommuf.for`, `prbmuf.for` | Iterative root-finding/mode selection, GOTO-heavy. Translating GOTO-driven iteration to `lax.while_loop`/`fori_loop` with matching iteration caps is the main structural challenge here. |
| 4 | Path/ray geometry | `mpath.for`, `bendy.for`, `gettop.for`, `luffy.for` | Virtual heights, hop distances, elevation angles per mode. |
| 5 | Antenna gain lookup | `wp10dwin/gainterp.for`, `gainrel.for`, `readant.for`, `antfile.for` | Interpolation only, per Section 0 scope decision — consumes static `.VOA` pattern files. |
| 6 | Field strength / loss | `gmloss.for`, `esreg.for`, `ef1var.for`, `f2dis.for`, `fdist.for` | |
| 7 | Noise model | `anois1.for`, `noisy.for`, `genois.for` | `genois.for` has a handful of `REAL*8` variance temporaries — keep those float64, matching the original; rest of the group is float32. |
| 8 | Reliability / statistics | `serprb.for`, `sigdis.for` | |
| 9 | Circuit orchestration | equivalent of `decred.for` + `batch.for` driving one circuit across requested hours/frequencies | First point where end-to-end golden-file tests (tier A) become meaningful. |
| 10 | Output formatting | `outtab.for`, `outall.for`, `tabbod.for` | Must hit the output-level bit-exact bar against `test01.out`'s fixed-width tables. |
| 11 | CLI | reads the same card-image format as `voacapl` (compatibility, so existing decks/tests work unmodified) | |
| 12 | *(stretch, after P2P is solid)* | vmap across hours × frequencies × months × ssn for one circuit, then generalize to the Area-mode grid | This is where the actual GPU speedup shows up — P2P alone (24 hours × ~11 frequencies) is too small to benefit much. Explicitly deferred until P2P is validated. |

## 6. Tooling

- Python: `src`-layout package, `pyproject.toml`, PyTorch (CUDA), NumPy, pytest,
  optionally `hypothesis` for tier-C property tests. `torch.compile` is a later,
  optional performance lever once correctness is locked down — not needed to
  get GPU parallelism in the first place.
- WSL: gfortran + `voacapl` build, plus a script that drives it over the golden
  input matrix and archives outputs as fixtures.
- Git: init now; commit `reference/` as vendored/read-only; `.gitignore` for
  build artifacts (`voacapl` build dir, WSL-built binaries, generated `.npz`
  coefficient caches).
- CI: local-only for now (no remote configured) — `pytest` + oracle-diff as the
  gate before any module is considered "done."

## 7. Risks flagged

- **Bit-exactness ceiling** — addressed in Section 0; the harness is built
  around the practical (output-level exact, internal-level tight-tolerance)
  interpretation from day one, and precision (float32 vs. float64) is assigned
  per module by mirroring the Fortran rather than assumed uniform, so neither
  of these gets relitigated later.
- **CCIR/URSI coefficient format** — ~~undocumented binary~~ resolved: ASCII
  sources + converter programs ship inside voacapl (see Module 0). Remaining
  work is reading the converters to document array shapes/ordering, not
  reverse-engineering an opaque binary.
- **COMMON-block global state** — ~100+ files share state through COMMON blocks;
  this is the single biggest source of subtle porting bugs. The Stage A/B split
  and per-module oracle dumps (tier B) exist specifically to contain this risk.
- **License** — resolved: unlicensed for now (see Section 2); revisit freely
  later, nothing is inherited from the public-domain/CC0 reference.

## 8. Immediate next steps once you approve

All original items 1–7 are DONE as of 2026-07-22: repo + oracle + golden
matrix (99 decks) + scaffold + Module 0 ingestion (bit-exact vs oracle
binaries, docs/coeff_format.md) + precision map + Module 1 (DAZEL Stage
A+B 100% bit-exact; DISTXY/SANG/MAGVAR/MAGFIN/GEOM Stage A validated
against tier-B dumps; `cang.for` skipped — it belongs to the separate
hfmufesw tool, not the voacapl P2P binary).

Next up (module table order, Section 5):

1. Module 2 — ionospheric parameter maps (highest risk): trace how
   voacapw consumes the /ccir/ arrays (GENION/F2VAR etc.), port the
   spherical-harmonic map evaluation Stage A with tier-B dumps.
2. Stage B (PyTorch) for the float32 Module-1 group once its consumers
   exist (GEOM/MAGVAR batching is mechanical; DAZEL Stage B is the
   template).
3. Module 3 — MUF/FOT (method-9 golden decks in golden/p2p/matrix are
   the early end-to-end targets).
4. Modules 4+ per the Section 5 table.
