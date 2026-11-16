# Changelog

## 0.2.0 — 2026-11-16

Performance release. Results in the default (bit-exact) mode are unchanged:
`tests/cuda/test_bitexact_gate.py` pins SHA-256 digests of every output field
over world-grid, random-order, METHOD 20/25/30, CCIR/URSI+Es, long-path and
type-13/14 antenna workloads, generated before the changes.

### Faster

- LUFFY kernel 2.5× faster (bit-exact), 2.9× with fast-math: per-thread state
  in local memory laid out for coalesced access, plus exact shortcuts (binary32
  square root proved equal to the binary64 path on all 2^32 inputs, ground-loss
  reuse across identical sample areas, a cosine table for the 40 fixed ray
  angles, bit-keyed memos). World grid end to end: 0.76 s bit-exact at chunk
  32,768 × 2 (was 1.35 s).
- `predict()` builds the circuit on the device: one circuit × 24 h × 11
  frequencies in about 6.5 ms warm (was 82–178 ms).

### Added

- `predict_batch` (`engine.predict_batch`, `voacap_gpu.batch`): many circuits in
  one job, `(N, H, F)` arrays, internal warp-coherent ordering, results in the
  caller's order; `r.circuit(i)` equals `predict()` bit for bit.
- Slim outputs: `outputs=(...)` on `SetupLuffy.run` / `PipelinedSetupLuffy.run`
  returns only the named /SON/ columns (101 MB instead of 785 MB per world
  grid), bit-identical to the same columns of the full result.
- On-device grid reduction: `gridjob.DeviceCellReducer` / `gridjob.run_grid`
  reproduce `CellReducer` (order-dependent, strict ties, winner-frequency
  accumulation) on the GPU; nothing per-row crosses the bus.
- `engine.coeffs_dir()` (public replacement for `engine._COEFFS_DIR`, which
  still works) and `engine.clear_device_cache()`.
- Tests: the bit-exact gate, the six sweep regression decks, status-code
  consistency, exhaustive checks of the exact shortcuts.

### Changed (callers may notice)

- **Status codes are globally unique.** `code // 100` names the routine (table
  in `cuda/kernels/status.cuh`; `engine.STATUS_*`). 0 = success and -1 = not
  written are unchanged; every non-zero refusal has a new number (e.g. the
  setup kernel's bad-hour refusal 3 → 1301).
- **Device memory:** the faster kernel reserves about 3.5 GB of local memory
  per process. On a 16 GB card use chunk 32,768 × 2 or smaller; 65,536 × 2
  overflows a card shared with a desktop. `-DVC_LUFFY_GLOBAL_STATE` restores
  the old layout at about 65% of the speed.
- `predict(fast_math=True)` now builds geometry, ionosphere and MUF with the
  fast-math libm too (as `PipelinedSetupLuffy` always did), so its results
  move slightly (MUF ~3e-6 relative) against 0.1.0's fast-math `predict`.
  Bit-exact mode is unaffected.
- `real_world_compare.py` and `tools/results/fortran_baseline.md` are marked
  LEGACY; `tools/bench_vs_fortran/` is the like-for-like benchmark.
- **Module layout.** `engine.py` and `cuda/driver.py` are now facades over
  smaller modules (`system`, `status`, `antennas`, `circuit`, `backends`,
  `results`, `rigcache`, `endpoint`; `cuda/abi`, `checks`, `tables`,
  `kernels`, `results`, `luffy_rigs`, `setup_rigs`). Every name importable
  from either facade before still is (pinned by `tests/test_engine_facade.py`
  and `tests/cuda/test_driver_facade.py`). Assigning to or monkeypatching a
  facade attribute (`engine.build_circuit = ...`, `engine._COEFFS_DIR = ...`,
  `driver.load_kernel = ...`) no longer affects the code: patch the module
  that looks the name up (see each facade's docstring). `predict()` no longer
  detects a patched `build_circuit`/`predict_circuits` and falls back to the
  host path.

## 0.1.0 — 2026-11-15

The first release after a full review. For valid inputs the numbers are the
same as 0.0.4 except for the three deliberate fixes under **Changed results**;
this was checked bit-for-bit on Prop-SL-shaped workloads (world grids, METHOD
25/30, fast-math, type-13/14 antennas, ICEPAC mode, 160 m).

### Fixed

- **Race in the multi-stream GPU runners** (`PipelinedSetupLuffy`,
  `PipelinedLuffy`, also `SetupLuffy`/`LuffyArena`). A runner built after an
  earlier one was freed could read stale device memory and return wrong rows,
  mostly with status 0. 0.0.4 gave different answers run to run on identical
  inputs in that pattern (up to ~1000 dB, with hop counts and MUFs changing).
  Every constructor and table upload now finishes its default-stream work
  before the work streams use it.
- `predict()` discarded the kernel's per-circuit status; `Results.status` was
  always 0. It now carries the real status.
- Type-14 antennas with a non-zero header gain (PARM(1)) skipped the Fortran's
  3-decimal gain round trip (1–2 ULP).
- Transmitter power in dBW is computed in float32, as the Fortran does
  (was float64; 1 ULP on some powers).

### Changed results (deliberate; all closer to the Fortran)

- Type-14 antennas now cover only the ANTENNA card's MINFREQ..MAXFREQ rows
  (default 2..30 MHz), as voacapl's ANTCALC does. **Below 2 MHz a type-14
  pattern is no longer used** (0 dBi, like the Fortran). Pass
  `tx_freq_range=(1, 30)` / `tx_antenna_freq_range=(1, 30)` to keep it on 160 m.
- The two float32 fixes above (at most ~3e-5 dB).

### Breaking or behaviour changes for callers

- **Input validation.** Out-of-range inputs raise `ValueError` instead of
  returning plausible numbers: frequency outside 1–30 MHz, month outside
  1..12, non-integer hour or hour outside 1..24, SSN outside 0–300, latitude
  outside ±90, non-finite values, `power_kw <= 0`, reliability outside 1..99,
  non-positive noise other than the category codes 0/−1..−4, and a `coeff`
  other than exactly `'CCIR'`/`'URSI'`. The device runners check their
  parameter arrays the same way (real exceptions, not `assert`).
- **Longitude is wrapped, not refused**, on the host and device paths (190 →
  −170; ±180 kept), so a map panned past the date line keeps working.
- `model=` given with a conflicting `coeff=` or non-default `System.fprob`
  raises instead of silently overriding it.
- `Results.at(hour, freq)` needs an exact frequency (it used to snap to the
  nearest) and raises `KeyError` for a missing hour or frequency.
- `Results.valid` (new) marks cells LUFFY did not compute; their numeric
  values are NaN instead of preset zeros (which read as 0 dBW).
- `System.max_delay_ms` / `max_spread_db` are deprecated: their names were
  crossed (`max_delay_ms` set the power tolerance in dB). They keep the old
  mapping and emit a `FutureWarning`; use `multipath_delay_ms` (DMP, ms) and
  `multipath_power_db` (PMP, dB).
- New warnings: fractional noise or reliability (truncated, as the Fortran
  does), a type-13 antenna without a bearing, a bearing that looks like
  radians, an installed antenna peak above 30 dBi.
- `build_circuit` no longer overwrites the OFFAZIM of a caller-supplied
  /cantenna/ (numerically inert for point-to-point gain).
- `gridjob.CellReducer.feed` treats non-finite samples as failed circuits.
- **PyTorch is optional**: install the `batched` extra for
  `voacap_gpu.batched`. CuPy stays a hard dependency.

### Added

- `predict(..., backend="cpu" | "auto")`: the scalar reference as a CPU
  backend, bit-identical to CUDA on every oracle deck (about 1000× slower).
- Antenna options: `tx_antenna_freq_range` / `rx_antenna_freq_range`,
  `tx_table_absolute` / `rx_table_absolute`.
- Oracle coverage: 12 antenna-pattern decks, 13 more ICEPAC-mode decks, a
  2,200-deck randomised sweep (`docs/accuracy_sweep.md`); CPU CI workflow;
  tests run on a clean clone without the `reference/` tree.
- `NOTICE`, units and accuracy sections in the README, like-for-like
  GPU-vs-Fortran benchmark (`tools/bench_vs_fortran/`).

### Clarified

- `model="icepac"` is Prop-SL's ICEPAC mode on the VOACAP engine (URSI +
  sporadic E), **not** the ITS ICEPAC program (no ICED model; Q/Kp ignored).
