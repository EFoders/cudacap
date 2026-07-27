# CUDACAP (voacap-gpu)

A GPU rewrite of **VOACAP** (Voice of America Coverage Analysis Program) HF ionospheric propagation prediction, validated bit-for-bit against the original Fortran. Since ICEPAC is built on the same engine, ICEPAC has also been made available. One CUDA thread runs one whole circuit-hour, so tens of thousands of circuits compute concurrently on a single GPU. The theoretical increase in performance is much higher than what we have achieved today, due to the nature of how IONCAP was written originally in Fortran. Some effort has been put in to restructure the engine without impacting correctness. At this time, I cannot guarantee the exact match to VOACAP, as that would require more testing for edge cases, etc. 

This project was done in support of Prop-SL, an HF, LF and VLF propagation platform made while working in strategic HF communications. Prop-SL is property of the Department of National Defence, and the Government of Canada, but this project was made in my own time to support it and is free to build upon, within the confines of the MIT Licence. Claude was used throughout to translate Fortran, and to speedup testing and development. This said, I have reviewed the code and structure extensively to confirm its validity. 

The point-to-point prediction engine is ported end to end and runs on the GPU. On an RTX 5080 it computes a
world-scale batch **~15× faster than the original Fortran across all 32 cores**
of my machine (and bit-exact against a single-core run when fast-math is
off). See the two architecture documents:

- **[docs/architecture_fortran.md](docs/architecture_fortran.md)** — how the
  original scalar/sequential Fortran VOACAP works.
- **[docs/architecture_gpu.md](docs/architecture_gpu.md)** — how this GPU
  version works, and why it is faster.

## The three implementations

The same physics exists three times, each validated against the one before it.
The folder names say what each *is*; the commit history calls them Stage A/B/C.

| Folder | Stage | What it is |
|---|---|---|
| `src/voacap_gpu/scalar/` | A | Line-faithful NumPy transliteration of the Fortran. Slow, but the **validation oracle** — every GPU result is checked against it. |
| `src/voacap_gpu/batched/` | B | PyTorch, area-mode: builds ionospheric maps over a lat/lon grid on the GPU. |
| `src/voacap_gpu/cuda/` | C | **The production path.** CUDA C compiled at runtime by NVRTC, one thread (or warp) per circuit-hour. Device sources in `cuda/kernels/`. |

Why bit-exactness matters: `scalar/` is 98.7% bit-exact against the real
Fortran over 132,048 cells, with **every discrete decision — mode, hop count,
MUF — exact**. Because `cuda/` is in turn bit-identical to `scalar/`, the GPU
answers carry the same guarantee transitively.

## The endpoint

`src/voacap_gpu/engine.py` is the public entry point: raw circuit parameters
in, propagation results out. It composes the setup chain (great-circle
geometry → sample areas → per-hour ionosphere and circuit MUF) on the host,
then runs the GPU kernel.

```python
from voacap_gpu.engine import predict, System

res = predict(
    tx=(35.68, 139.77),          # Tokyo   (lat, lon; S/W negative)
    rx=(-33.87, 151.21),         # Sydney
    freqs=[7, 14, 21],           # MHz
    month=7, ssn=160,            # July, sunspot number
    hours=range(1, 25),          # UTC
    tx_antenna="…/samples/sample.14",   # tabulated pattern (or None = isotropic)
    tx_bearing=90.0, rx_bearing=270.0,  # main-beam azimuth -> OFFAZIM
    model="voacap",                     # or "icepac" (see note below)
    system=System(power_kw=1.0, required_snr=24.0,
                  required_reliability=90, man_made_noise="residential"),
)
res.muf                          # circuit MUF by hour
res.at(12, 14)                   # {'snr','dbw','reliab','field_dbu','mode',…}
```

The interface accepts the Prop-SL point-to-point parameter surface: antenna
pattern, **beam bearing/azimuth** (→ OFFAZIM), transmit power, required S/N and
reliability, man-made-noise category, and per-end gain offsets. Anything not
yet ported **raises `NotImplementedError`** rather than returning approximate
numbers — e.g. `model="icepac"` (see the engine docstring for the full list of
what is validated vs refused).

## Performance

Measured on my machine (RTX 5080 GPU, 32-core CPU for the Fortran), a
161,280-circuit-hour workload, whole wall-clock including host prep and PCIe:

| | total | vs Fortran 32-core |
|---|---|---|
| GPU, pipelined + fast-math (not bit-exact) | ~422 ms | **~15×** |
| GPU, pipelined (bit-exact) | ~715 ms | ~9× |
| Fortran, 32 cores | ~6,518 ms | 1× |
| Fortran, 1 core | ~112,090 ms | 1/17× |

Fast-math trades the binary64-round-once transcendentals for native f32 libm:
1.9–3.0× faster, still 99.34% of cells bit-exact, and **no discrete decision
changes**. It is opt-in (`fast_math=True`); the default stays bit-exact.
Baselines: [tools/results/fortran_baseline.md](tools/results/fortran_baseline.md);
reproduce with `uv run python tools/real_world_compare.py`.

## Layout

- `reference/` — vendored, read-only upstream Fortran (never modified):
  - `reference/voacapl/` — the gfortran port we validate against (the *oracle*).
  - `reference/itshfbc/` — the ITSHFBC data + antenna samples.
- `src/voacap_gpu/` — `scalar/`, `batched/`, `cuda/`, and `engine.py`.
- `tests/` — three-tier suite: `scalar/`, `batched/`, `cuda/` mirror the source;
  golden end-to-end decks, per-module oracle dumps, and property tests.
- `tools/` — benchmarks (`bench_stage_c.py`, `real_world_compare.py`), the
  oracle build/run scripts (`tools/oracle/`), and measured results.
- `docs/` — the two architecture documents plus format/precision notes.

## Requirements

- Python via `uv`.
- An NVIDIA GPU. The CUDA path compiles device code at runtime through NVRTC,
  shipped in the `cupy-cuda12x[ctk]` wheels.

## Testing

    uv run pytest            # the full three-tier suite (839 tests)

The headline test is `tests/cuda/test_luffy_replay.py`: it runs 504
circuit-hours through the GPU kernel and asserts **all 7,012,152 output cells
are bit-identical to the scalar reference.**

## License
original VOACAP is a US Government (NTIA/ITS) public-domain work.
