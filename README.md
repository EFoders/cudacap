# CUDACAP (voacap-gpu)

A GPU rewrite of **VOACAP** (Voice of America Coverage Analysis Program) HF ionospheric propagation prediction, validated against the original Fortran. ICEPAC is built on the same engine, so an ICEPAC configuration is also available (see [ICEPAC](#icepac)). One CUDA thread runs one whole circuit-hour, so tens of thousands of circuits compute concurrently on a single GPU. The theoretical increase in performance is much higher than what we have achieved today, due to the nature of how IONCAP was written originally in Fortran. Some effort has been put in to restructure the engine without impacting correctness. At this time, I cannot guarantee the exact match to VOACAP, as that would require more testing for edge cases, etc. See [Accuracy](#accuracy) for what has and has not been checked.

This project was done in support of Prop-SL, an HF, LF and VLF propagation platform made while working in strategic HF communications. Prop-SL is property of the Department of National Defence, and the Government of Canada, but this project was made in my own time to support it and is free to build upon, within the confines of the MIT Licence. Claude was used throughout to translate Fortran, and to speedup testing and development. This said, I have reviewed the code and structure extensively to confirm its validity. 

The point-to-point prediction engine is ported end to end and runs on the GPU.
On an RTX 5080 the production path computes about 290,000 circuit-hours per
second (9 frequencies each, bit-exact mode; 370,000 with fast-math), about
4.6× all 32 threads of an i9-14900KF running the Fortran on the same work;
see [Performance](#performance) for what that includes. See the two architecture documents:

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
| `src/voacap_gpu/batched/` | B | PyTorch, area-mode: builds ionospheric maps over a lat/lon grid on the GPU. Not used by `predict()` or the CUDA path. |
| `src/voacap_gpu/cuda/` | C | **The production path.** CUDA C compiled at runtime by NVRTC (via CuPy), one thread (or warp) per circuit-hour. Device sources in `cuda/kernels/`. |

`cuda/` is bit-identical to `scalar/` on the replay cases, and `scalar/` is
compared against the real Fortran. The numbers are in [Accuracy](#accuracy).

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
    month=7, ssn=160,            # July, smoothed sunspot number
    hours=range(1, 25),          # UTC, 1..24
    tx_antenna="…/samples/sample.14",   # tabulated pattern (or None = isotropic)
    tx_bearing=90.0, rx_bearing=270.0,  # main-beam azimuth, degrees true -> OFFAZIM
    model="voacap",                     # or "icepac"
    system=System(power_kw=1.0,
                  required_snr=48.0,         # dB-Hz (about 13 dB in 3 kHz)
                  required_reliability=90,   # percent
                  man_made_noise=145),       # -dBW/Hz at 3 MHz (145 = residential)
)
res.muf                          # circuit MUF by hour
res.status                       # per-hour kernel status, 0 = OK
res.at(12, 14)                   # {'snr','dbw','reliab','field_dbu','mode',…}
```

`tx_antenna` points at a VOACAP/ITSHFBC sample antenna file (`samples/sample.14`
in a VOACAP or voacapl install). That file is **not shipped in this repo**;
use `tx_antenna=None` for an isotropic antenna, or point it at your own
type-13/14 `.VOA` file. The `System()` defaults are the golden-deck test
values (for example 73 dB-Hz required SNR, 0.1° minimum take-off angle,
1 kW), not recommended operating values.

The interface accepts the Prop-SL point-to-point parameter surface: antenna
pattern, **beam bearing/azimuth** (→ OFFAZIM), transmit power, required S/N and
reliability, man-made noise, and per-end gain offsets. Anything not ported
**raises `NotImplementedError`** rather than returning approximate numbers:
METHOD other than 20/25/30, more than 11 frequencies, synthesised antenna
types 1–12, area-coverage antennas, the old noise model and local-time hours.
Out-of-range inputs raise `ValueError` (see
[Units and conventions](#units-and-conventions)).

### ICEPAC mode (not the ITS ICEPAC program)

`model="icepac"` reproduces Prop-SL's ICEPAC setting as voacapl runs it: the
**VOACAP engine** with URSI coefficients and sporadic E enabled (FPROB
`1.00 1.00 1.00 0.70`). It is **not** the ITS ICEPAC program. voacapl contains
no ICED ionospheric-profile model, and a Q/Kp index has no effect (voacapl
reads a Q on the SUNSPOT card as a second month's sunspot number and ignores
it). This mode is validated against voacapl on 14 oracle decks
(`golden/dumps/luffy_icepac/`): 343 km to the ~21,700 km long path; auroral,
polar, southern high-latitude and equatorial circuits; all four seasons;
SSN 15–160; METHOD 20, 25 and 30. The same absolute bounds as VOACAP mode
apply (worst 1.3e-3 dB), on CPU and CUDA. One near-meridional circuit
(Ushuaia–Rothera, Δlon 0.17°) exceeds the take-off-angle bound (0.0065° vs
0.002°) through float32 conditioning in the great-circle geometry; VOACAP mode
shows the same on that path, and the test marks it as an expected failure. It
is not validated against `Icepacw.exe` or any ICED-based prediction.

### Multipath fields

`System.multipath_delay_ms` is VOACAP's DMP (maximum multipath delay, ms) and
`System.multipath_power_db` is PMP (multipath power tolerance, dB). They only
affect the multipath probability, which `Results` does not currently return.
The older names `max_delay_ms` and `max_spread_db` are deprecated: they keep
their old, crossed mapping (`max_delay_ms` sets PMP, `max_spread_db` sets DMP)
so existing callers get the same numbers, and they emit a `FutureWarning`.

## Units and conventions

### Inputs

| Input | Unit / convention | Accepted range |
|---|---|---|
| `tx`, `rx` | (lat, lon) in degrees; south and west negative | lat −90..90 (else `ValueError`); lon any finite value, wrapped into −180..180 by whole turns (190 → −170; ±180 kept) |
| `freqs` | MHz, 1 to 11 values per call | 1–30 MHz |
| `month` | calendar month | integer 1..12 |
| `ssn` | smoothed sunspot number; the CCIR/URSI maps are interpolated or extrapolated linearly | 0–300 |
| `hours` | UTC, 1-based, as in VOACAP | integers 1..24 |
| `tx_bearing`, `rx_bearing` | antenna main-beam azimuth, **degrees true** (not radians). `None` points the beam along the path at that end. OFFAZIM = path azimuth − beam heading, wrapped to [0, 360) | — |
| `tx_antenna`, `rx_antenna` | type-13 or type-14 `.VOA` file, or `None` for 0 dBi isotropic | file antennas cover whole-MHz rows 2–30 by default (`tx_antenna_freq_range` / `rx_antenna_freq_range`, the ANTENNA card's MINFREQ/MAXFREQ); a requested frequency outside the antenna's range raises `ValueError` |
| antenna table values | dB **relative to header parameter 1**: the header gain is added to every table value (the VOACAP ANTCALC rule). For a table already in absolute dBi, set `tx_table_absolute` / `rx_table_absolute` to skip that addition. Efficiency in dB | — |
| `System.power_kw` | transmitter power, kW (converted to dBW as 30 + 10·log10 kW) | > 0 |
| `System.required_snr` | required SNR, **dB in a 1 Hz bandwidth (dB-Hz)**. To convert from an SNR in bandwidth B Hz, add 10·log10(B) (34.8 dB for 3 kHz) | — |
| `System.required_reliability` | reliability percentile, percent (integer) | — |
| `System.man_made_noise` | a **positive** number in −dBW/Hz at 3 MHz (145 residential, 150 rural, 164 quiet rural). Negative values are accepted only as CCIR category codes: −1 industrial, −2 residential, −3 rural, −4 remote | > 0, or −1..−4 |
| `System.min_takeoff_deg` | minimum take-off angle, degrees; ≤ 0 becomes 3.0 (as the Fortran DECRED does) | — |
| `System.multipath_delay_ms` | DMP, maximum multipath delay, ms | — |
| `System.multipath_power_db` | PMP, multipath power tolerance, dB | — |
| `System.tx_gos`, `System.rx_gos` | dB added to every value of that end's gain table | — |
| `System.fprob` | FPROB multipliers on the E, F1, F2 and Es critical frequencies; a non-zero 4th value turns sporadic E on | — |
| `System.bandwidth`, `System.modulation` | accepted and **ignored**; fold bandwidth into `required_snr` yourself | — |
| `pole` | geomagnetic north pole (lat, lon), degrees. Default (78.5, −69.0) is voacapl's built-in fallback, which the oracle used; a stock voacapl install reads 79.5° N from `north_pole.txt` | — |

### Outputs (`Results`)

| Output | Unit / convention |
|---|---|
| `hours` | UTC, 1..24, shape `(H,)` |
| `freqs` | MHz, shape `(F,)` |
| `snr` | median SNR, **dB-Hz** (1 Hz bandwidth), shape `(H, F)` |
| `dbw` | median signal power at the receiver, dBW |
| `field_dbu` | median field strength, dB(µV/m) |
| `reliab` | circuit reliability, **0..1** (not percent) |
| `nhp` | number of hops of the dominant mode |
| `mode` | 2-character reflecting-layer label (`'F2'`, `' E'`, …) |
| `muf`, `fot`, `hpf` | circuit MUF and its 90 % / 10 % decile frequencies, MHz, shape `(H,)` |
| `luf` | lowest *requested* frequency that meets the required reliability, MHz; NaN if none does. This is not VOACAP's LUF |
| `status` | per-hour kernel status, shape `(H,)`; 0 = computed, non-zero = refused by the kernel |
| `valid` | boolean mask, shape `(H, F)`; False where LUFFY computed nothing. The numeric tables are NaN there |

`Results.at(hour, freq)` needs an exact frequency from `freqs`; any other value
raises. Above the MUF the kernel still returns numbers (very low SNR and
signal power, very small reliability), and these are marked valid because
LUFFY did compute them.

## Accuracy

The oracle is voacapl built with gfortran 15.2.1 (glibc), with LUFFY's entry
and exit state captured by a shim. Comparisons use `predict()`'s
configuration (raw inputs, a fresh circuit per hour) on both backends; CPU and
CUDA are bit-identical throughout.

**Curated oracle decks (asserted by the test suite):**

| Deck set | Decks | Hop counts | Worst \|Δ\| signal | Worst \|Δ\| reliability |
|---|---|---|---|---|
| VOACAP, isotropic (`golden/dumps/luffy/`) | 21 | identical | 1.66e-3 dB | 3.4e-5 |
| ICEPAC mode (`golden/dumps/luffy_icepac/`) | 14 | identical | 1.3e-3 dB | 2.5e-5 |
| Antenna patterns (`golden/dumps/luffy_antenna/`) | 12 | identical | 1.0e-3 dB | 1.5e-5 |

The antenna decks check cudacap's own READANT + ANTCALC path for stock
type-13 and type-14 `.VOA` files: type 14 at the transmitter, receiver and
both; type-13 beams on, 30°, 90° and 180° off the path, across the 359°→0°
wrap and on the long path; and a 3–20 MHz antenna range. The built gain
tables equal ANTCALC's exactly. `tests/test_engine.py` holds the asserted
bounds (signal fields ≤ 0.01 dB, reliability ≤ 2e-4, take-off angle ≤ 0.002°).

**Randomised sweep (not in CI).** 2,200 random decks, 317,304 hour ×
frequency cells: the whole globe including the poles, 100–19,900 km plus long
paths, METHOD 20/25/30, CCIR and URSI with and without sporadic E, SSN 0–200,
and random SYSTEM and antenna settings.

| Quantity | Result |
|---|---|
| Bit-identical values | 93% |
| Hop counts agree | 99.999% of cells (4 cells differ, one deck) |
| 99.9th-percentile \|Δ\| | 0.004 dB signal/SNR; 3e-4° take-off angle; 7e-6 reliability |
| Worst single cell | 0.1 dB, almost all on over-the-MUF modes below −150 dBW |

Every disagreement traces to float32 rounding differences between glibc's
trig functions and cudacap's correctly rounded host math, amplified by the
model's own conditioning (near-pole endpoints, long paths near 2π, an
integer truncation in FOBBY, one hop-count `INT()` at exactly 2.0). None is a
port error. The 0.042% of cells beyond the CI bounds are why the sweep's bounds
are reported as distributions, not asserted. Generator, runner, comparison and
full report: `tools/oracle/sweep/` and
[docs/accuracy_sweep.md](docs/accuracy_sweep.md).

The "98.7% bit-exact" figure in the docs is different: it is the LUFFY routine
**replayed from captured Fortran input state** (21 cases, 132,048 cells), not
`predict()` from coordinates. In that replay every discrete decision (mode,
hop count, layer) is exact. `cuda/` is bit-identical to `scalar/` on the same
replay (`tests/cuda/test_luffy_replay.py`).

**Not covered by any oracle comparison:**

- the real ITS ICEPAC program (ICED model, Q/Kp); see ICEPAC mode above;
- synthesised antenna types (1–12, which the engine refuses) and antenna
  files other than the stock type-13/14 samples;
- frequencies below 2 MHz;
- SSN above 200;
- method 9 (the engine refuses it) and area mode;
- disturbed high-latitude conditions (VOACAP models monthly medians only);
- the printed `.out` matrix in `golden/p2p/matrix/outputs/`;
- the device-side setup path against the oracle directly (it is compared with
  the host setup, which is compared with the oracle).

## Performance

Measured 2026-11-15 on one machine: NVIDIA RTX 5080 (16 GB) under Windows 11;
the Fortran oracle (voacapl, gfortran 15.2.1 `-O2`) in WSL2 on the same
Intel i9-14900KF (24 cores / 32 threads). **Identical workload on both
sides:** transmitter Ottawa, receivers on the 2° world grid, 9 frequencies,
July, SSN 120, CCIR, METHOD 30, `System()` defaults, 1 kW isotropic.
Throughput is in circuit-hours per second (one path, one hour, all 9
frequencies). Scripts and the raw results are in `tools/bench_vs_fortran/`
(`gen_bench_decks.py`, then `bench_gpu.py` and, in WSL, `bench_fortran.sh`).

| Configuration | Workload | Wall time | Circuit-hours/s |
|---|---|---|---|
| GPU, production path, bit-exact (chunk 65,536 × 2) | 395,304 (world grid) | 1.35 s | **292,000** |
| GPU, production path, fast-math (chunk 65,536 × 2) | 395,304 | 1.07 s | 371,000 |
| GPU, production path, bit-exact | 15,360 (640 circuits × 24 h) | 0.10 s | 152,000 |
| Fortran, 32 processes in parallel | 15,360 (same 640 circuits) | 0.24 s | 63,700 |
| Fortran, 1 process at a time | 1,536 (64 circuits × 24 h) | 0.44 s | 3,480 |
| `predict()`, CUDA backend, one circuit × 24 h | 24 | 144 ms | 167 |
| `predict()`, CPU backend, one circuit × 24 h | 24 | 1.49 s | 16 |

"Production path" is `PipelinedSetupLuffy` (what the Prop-SL service uses):
raw coordinates in, geometry and ionosphere built on the device, results back
on the host. Bit-exact mode used 8.6 GB of device memory at chunk 65,536 × 2
streams (5.1 GB at 32,768 × 2, about 10% slower).

What this means:

- At world-grid scale the GPU is about **4.6× faster bit-exact and 5.8× with
  fast-math than all 32 CPU threads** running the Fortran on the same work.
  At 15,360 circuit-hours the GPU is under-filled and the gap is 2.4×.
- The Fortran figures include one process start-up per circuit, as voacapl
  runs one deck per process. A 1-hour deck takes 3.3 ms and a 24-hour deck
  6.9 ms, so roughly 3 ms of each run is start-up. Without it, 32-thread
  Fortran would be at most about 1.8× faster than measured, and the GPU's
  lead at world-grid scale would shrink to about 2.5×.
- `predict()` is the convenience API and is **host-bound**: one circuit takes
  144 ms on CUDA, about 20× slower than one Fortran process. Use the device
  rigs (`PipelinedSetupLuffy`) for anything larger than a handful of
  circuits. The CPU backend is about 200× slower than the Fortran per
  circuit-hour; it exists for machines without a GPU and for CI.

**Older replay benchmark.** `tools/real_world_compare.py` replays 161,280
captured, pre-built circuit-hours (host gather + PCIe + kernel; setup is not
timed). Re-run on 2026-11-15: pipelined bit-exact 732 ms, pipelined
fast-math 445 ms. Its "~9× / ~15× vs Fortran 32-core" ratios divide by a
stored Fortran baseline from a 28-frequency deck
([tools/results/fortran_baseline.md](tools/results/fortran_baseline.md)),
not by the like-for-like measurement above. Against the measured 32-thread
Fortran rate (63,700 circuit-hours/s, so 2.5 s for 161,280), the replay is
3.5× (bit-exact) and 5.7× (fast-math).

Fast-math trades the binary64-round-once transcendentals for native f32 libm.
It is 1.3–1.6× faster on the production path, keeps 99.34% of cells
bit-exact, and changes no discrete decision in the tests. It is opt-in
(`fast_math=True`); the default stays bit-exact.

## Layout

- `src/voacap_gpu/` — `scalar/`, `batched/`, `cuda/`, `engine.py`, and
  `data/coeffs/` (the bundled CCIR/URSI coefficient files).
- `golden/` — oracle outputs from voacapl: end-to-end decks and outputs
  (`p2p/`), per-routine and LUFFY dumps (`dumps/`), coefficient binaries
  (`coeffs/`).
- `tests/` — three-tier suite: `scalar/`, `batched/`, `cuda/` mirror the source;
  golden end-to-end decks, per-module oracle dumps, and property tests.
- `tools/` — benchmarks (`bench_stage_c.py`, `real_world_compare.py`), the
  oracle build/run scripts (`tools/oracle/`), and measured results.
- `docs/` — the two architecture documents plus format/precision notes.

The vendored Fortran tree `reference/` (voacapl and ITSHFBC) is **no longer
in the repo**. It was removed in commit `477dad7`. For development it can be
restored from history:

    git checkout 477dad7^ -- reference

Do not commit it back.

## Requirements and install

- Python via `uv`.
- For the production path, an NVIDIA GPU. The CUDA path compiles device code
  at runtime through NVRTC, shipped in the `cupy-cuda12x[ctk]` wheels (a hard
  dependency for now; about 1.6 GB of wheels on Linux).
- PyTorch is **optional**. Only `batched/` (Stage B) and the
  `tools/bench_stage_b.py`-style scripts use it.

```bash
uv sync                      # default: NumPy + CuPy (bundled CUDA 12 toolkit), no PyTorch
uv sync --extra batched      # also PyTorch (CUDA 12.8 build, several GB) for voacap_gpu.batched
```

With pip: `pip install .` or `pip install ".[batched]"`. pip does not read
uv's index settings, so for a CUDA build of torch on Windows install it first
from https://download.pytorch.org/whl/cu128.

### CPU backend

`predict(..., backend="cpu")` runs LUFFY with the scalar reference
(`voacap_gpu.scalar.luffy`) instead of the CUDA kernel. It never imports
CuPy, so it works without a GPU. It is bit-identical to the CUDA kernel on
every captured oracle case, but roughly 1000× slower (about 20–70 ms per
circuit-hour), so it suits a few circuits, not area grids. `backend="auto"`
uses CUDA when a device is usable and otherwise falls back to the CPU with a
warning. Refused hours get the same `Results.status` code and `valid` mask on
both backends; status 99 marks a CPU-only refusal with no kernel counterpart.
`fast_math=True` is CUDA-only. The default stays `backend="cuda"`.

## Testing

```bash
uv run pytest -q -m "not oracle"   # no reference/ needed; this is what CI runs
uv run pytest -q                    # full suite
```

Tests read the packaged coefficients (`src/voacap_gpu/data/coeffs`), the
oracle captures in `golden/`, and two stock sample antennas copied into
`tests/data/` (see its README). A few tests marked `oracle` need the vendored
Fortran tree `reference/`; they skip with a reason when it is absent. To run
them, restore it (see [Layout](#layout)) with `git checkout 477dad7^ -- reference`.

`tests/batched/` is skipped when torch is not installed. CUDA tests skip when
no GPU or driver is present; engine tests then still run on the CPU backend.
CI (`.github/workflows/ci.yml`) runs the non-oracle suite on a CPU-only Linux
runner.

`tests/test_engine.py` runs all 22 captured oracle cases end to end from raw
coordinates on both backends, with absolute bounds (hop counts exact; signal
fields ≤ 0.01 dB; reliability ≤ 2e-4); see the table at the top of that file.

The headline test is `tests/cuda/test_luffy_replay.py`: it runs 504
circuit-hours through the GPU kernel and asserts **all 7,012,152 output cells
are bit-identical to the scalar reference.**

## License

cudacap's own code is released under the MIT License; see [LICENSE](LICENSE).

It is a port of VOACAP/ICEPAC, which NTIA/ITS developed as a work of the US
Government; the original is not subject to copyright protection in the US.
The port was validated against J. A. Watson's voacapl, whose modifications are
released under CC0. The bundled CCIR/URSI coefficient data in
`src/voacap_gpu/data/coeffs/` comes from the VOACAP/ITSHFBC distribution via
voacapl.

See [NOTICE](NOTICE) for the attributions and the original NTIA/ITS
disclaimer.
