# GPU vs Fortran on the same workload

1. `uv run python tools/bench_vs_fortran/gen_bench_decks.py` writes `decks/`
   (640 world-grid circuits, Ottawa transmitter, 9 frequencies; plus 64
   one-hour decks to size process start-up). `decks/` is generated, not committed.
2. `uv run python tools/bench_vs_fortran/bench_gpu.py` (quiet machine, GPU):
   world-grid production path bit-exact and fast-math, the same 640 circuits,
   and `predict()` latency on both backends -> `bench_gpu.json`.
3. From PowerShell: `wsl bash "<repo>/tools/bench_vs_fortran/bench_fortran.sh"`
   (needs the oracle built by `tools/oracle/build_oracle.sh`): serial and
   32-way parallel voacapl, one process per circuit.

Results from 2026-11-15 (RTX 5080, i9-14900KF) are in `results_2026-11-15_*`
and summarised in the top-level README's Performance section.
