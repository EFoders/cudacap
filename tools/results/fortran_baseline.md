# LEGACY: Fortran oracle baseline from 2026-07-23

> **LEGACY. Do not quote these numbers as the GPU-vs-Fortran comparison.**
> The current like-for-like benchmark is
> [`tools/bench_vs_fortran/`](../bench_vs_fortran/README.md). On 2026-11-15
> (`results_2026-11-15_*`), 32-thread voacapl ran 63,700 circuit-hours/s
> (1.74 µs per circuit, hour and frequency). The GPU world grid ran 292k
> circuit-hours/s bit-exact and 371k fast-math, which is 4.6x and 5.8x.
>
> **What this file is.** It records voacapl timed on `bench28.dat`. That is a
> single circuit over 24 hours with 28 frequencies, run as one process per
> circuit, so each timing includes one process start-up (~3 ms).
>
> **Why ratios built on it are not like-for-like.**
> `tools/real_world_compare.py` divides its GPU time by
> `FORT_32CORE_MS_CH = 0.97/24` ms per circuit-hour from this table. Its GPU
> side, however, replays pre-built 9-frequency circuits with setup untimed. Per
> circuit-hour, that charges Fortran for about 3x the frequencies, as well as
> setup and start-up that the GPU side skips. This is where the old "~9x / ~15x
> vs Fortran 32-core" figures came from. Normalised per (circuit, hour,
> frequency), this table's 1.44 µs/eval is close to the current 1.74.
>
> The table below is kept unchanged as history.

## Original note: MEASURED, not assumed

voacapl-oracle/prefix/bin/voacapl on bench28.dat (1 circuit, 24 h, 28 freq =
672 evals), WSL, 32-core box, 2026-07-23.  Per-circuit numbers include
process startup, which is what bulk running actually pays (one process per
circuit).

| cores | ms/circuit | us/eval | scaling |
|------:|-----------:|--------:|--------:|
|     1 |      16.68 |   24.82 |    1.0x |
|     8 |       2.20 |    3.27 |    7.6x |
|    32 |       0.97 |    1.44 |   17.2x |

- 1-core confirms the stored 17.5 ms baseline (measured 16.68).
- 32-core scaling is SUB-LINEAR (17x, not 32x) because each circuit is a
  separate voacapl process and process spawn does not parallelise.  This is
  the honest whole-machine competitor for the GPU.
- us/eval is the unit the GPU benchmark uses; ms/circuit is what a user
  waiting on one prediction feels.
