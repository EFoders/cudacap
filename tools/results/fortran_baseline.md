# Fortran oracle baseline — MEASURED (not assumed)

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
