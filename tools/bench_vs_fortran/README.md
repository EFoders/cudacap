# GPU vs Fortran on the same workload

This is the current, like-for-like benchmark. (`tools/real_world_compare.py`
and `tools/results/fortran_baseline.md` are LEGACY. They replay pre-built
circuits and divide by a 28-frequency, process-per-circuit baseline, so their
ratios cannot be compared with these.)

Both sides use the same workload definition (`gen_bench_decks.py`): an Ottawa
transmitter, receivers on the 2-degree world grid, 9 frequencies, month 7,
SSN 120, CCIR, METHOD 30, `System()` defaults, 1 kW, isotropic antennas at
both ends, and hours 1-24.

1. `uv run python tools/bench_vs_fortran/gen_bench_decks.py` writes `decks/`.
   That is 640 world-grid circuits, plus 64 one-hour decks used to size process
   start-up. `decks/` is generated and not committed.
2. `uv run python tools/bench_vs_fortran/bench_gpu.py` (quiet machine, GPU)
   times three things:
   * the world grid on the production device path (`PipelinedSetupLuffy`),
     bit-exact and fast-math. It starts from raw (lat, lon, hour) parameters,
     with the setup chain on the device, and the timing covers both PCIe legs;
   * the same 640 circuits voacapl runs, in both modes;
   * `predict()` latency on both backends.

   It writes `results_<date>_gpu.json` and prints the summary table. It never
   overwrites an existing file without `--force`. Use `--out` for a scratch
   run and `--note` to record conditions.
3. From PowerShell: `wsl bash "<repo>/tools/bench_vs_fortran/bench_fortran.sh"`.
   This needs the oracle built by `tools/oracle/build_oracle.sh`. It runs
   voacapl serially and 32-way parallel, one process per 24-hour circuit (how
   voacapl runs a deck), and writes `results_<date>_fortran.txt`. Arithmetic
   uses `awk` only, because `bc` is not in the WSL image.
4. `uv run python tools/bench_vs_fortran/summary.py` prints the like-for-like
   table from the newest pair of result files (or `--gpu` / `--fortran`).

## Units

* **circuit-hours/s**: one circuit-hour is one receiver at one hour, with all
  9 frequencies.
* **µs per (circuit, hour, frequency)**: wall time divided by
  circuit-hours x 9. This is the unit to use when comparing against a deck
  with a different number of frequencies. For example, the legacy baseline's
  "1.44 µs/eval" is a 28-frequency deck.

voacapl's numbers include one process start-up per circuit (~3 ms, sized from
the one-hour decks). That is what running voacapl over many circuits costs.
The "compute only" row subtracts it, as (t(24 h) - t(1 h)) / 23 per hour.

## Results, 2026-11-15 (RTX 5080, i9-14900KF, quiet machine)

`results_2026-11-15_fortran.txt`, `results_2026-11-15_gpu.json`:

| what | circuit-hours/s | µs per (c, h, f) | vs voacapl 32 threads |
|---|---:|---:|---:|
| voacapl, 32 threads, 640 circuits | 63,734 | 1.743 | 1.00x |
| voacapl, 1 thread, incl. process start-up | 3,483 | 31.9 | 0.05x |
| voacapl, 1 thread, compute only (est.) | 6,345 | 17.5 | 0.10x |
| GPU, world grid (395,304 circuit-hours), bit-exact | 292,330 | 0.380 | **4.59x** |
| GPU, world grid, fast-math | 370,578 | 0.300 | **5.81x** |
| GPU, the same 640 circuits, bit-exact | 152,398 | 0.729 | 2.39x |

The 640-circuit row is the strictly identical workload. At 15,360
circuit-hours it is too small to fill the GPU, so the world-grid rows are the
throughput at scale. The 2026-11-15 GPU file predates the fast-math 640-circuit
run. `bench_gpu.py` now records it, so the next quiet-machine run adds that
row. `predict()` latency for one circuit x 24 h x 9 f: 143.6 ms on CUDA and
1,492 ms on CPU.
