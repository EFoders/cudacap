# CUDACAP (voacap-gpu)

A GPU rewrite of **VOACAP** (Voice of America Coverage Analysis Program) HF ionospheric propagation prediction, validated against the original Fortran. ICEPAC is built on the same engine, so an ICEPAC configuration is also available (see [ICEPAC](#icepac)). One CUDA thread runs one whole circuit-hour, so tens of thousands of circuits compute concurrently on a single GPU. The theoretical increase in performance is much higher than what we have achieved today, due to the nature of how IONCAP was written originally in Fortran. Some effort has been put in to restructure the engine without impacting correctness. At this time, I cannot guarantee the exact match to VOACAP, as that would require more testing for edge cases, etc. See [Accuracy](#accuracy) for what has and has not been checked.

This project was done in support of Prop-SL, an HF, LF and VLF propagation platform made while working in strategic HF communications. Prop-SL is property of the Department of National Defence, and the Government of Canada, but this project was made in my own time to support it and is free to build upon, within the confines of the MIT Licence. Claude was used throughout to translate Fortran, and to speedup testing and development. This said, I have reviewed the code and structure extensively to confirm its validity. 

The point-to-point prediction engine is ported end to end and runs on the GPU.
On an RTX 5080 the production path computes about 520,000 circuit-hours per
second (9 frequencies each, METHOD 30, bit-exact; 780,000 with fast-math),
about 8× all 32 threads of an i9-14900KF running the Fortran on the same work;
see [Performance](#performance) for what that includes and the memory it needs. See the two architecture documents:

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
| `status` | per-hour kernel status, shape `(H,)`; 0 = computed, non-zero = refused. Codes are globally unique: `code // 100` names the refusing routine (table in `src/voacap_gpu/cuda/kernels/status.cuh`, mirrored as `engine.STATUS_*`) |
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

**Randomised sweep.** The full sweep is not run in CI, but its six most
extreme decks are (`tests/test_sweep_regression.py`, `golden/dumps/luffy_sweep/`),
end to end on CPU and CUDA, against sweep-sized bounds (signal ≤ 0.25 dB,
reliability ≤ 1e-3, take-off angle ≤ 0.04°) with the one documented hop-count
flip pinned, plus a SHA-256 of cudacap's own output per deck. The sweep itself:
2,200 random decks, 317,304 hour ×
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
frequencies). Scripts and raw results: `tools/bench_vs_fortran/` (and
`tools/kernel_perf/` for the kernel profile).

| Configuration | Workload (circuit-hours) | Bit-exact | Fast-math |
|---|---|---|---|
| GPU, production path, chunk 32,768 × 2 | 395,304 (world grid) | **523,000/s** (0.76 s) | **776,000/s** (0.51 s) |
| GPU, production path, chunk 16,384 × 2 | 15,360 (640 circuits × 24 h) | 231,000/s | 334,000/s |
| Fortran, 32 processes in parallel | 15,360 (same 640 circuits) | 63,700/s | — |
| Fortran, 1 process at a time | 1,536 (64 circuits) | 3,480/s | — |
| `predict()`, CUDA, one circuit × 24 h × 11 freqs (warm) | 24 | 6.5 ms | — |
| `predict()`, CPU backend, the same | 24 | 1.06 s | — |

**Area maps (METHOD 25, the Prop-SL service's grid setting), world grid ×
24 h:** full rows plus host reduction (the service today) 0.44 s; slim rows
plus host reduction 0.31 s; reduction on the GPU (`gridjob.run_grid`) 0.24 s.
METHOD 25 skips the long-path model, so it is faster than the METHOD 30 rows
above.

What this means:

- At world-grid scale the GPU is about **8× faster bit-exact and 12× with
  fast-math than all 32 CPU threads** running the Fortran on the same work. On
  the small 640-circuit job it is about 3.6× / 5.2×.
- The Fortran figures include one process start-up per circuit (about 3 ms of
  each 6.9 ms run), as voacapl runs one deck per process; without it 32-thread
  Fortran would be at most about 1.8× faster than measured.
- Phase 4 made the LUFFY kernel 2.5× faster (bit-exact; same results to the
  bit, checked by `tests/cuda/test_bitexact_gate.py`): per-thread state now
  lives in local memory laid out for coalesced access, plus exact shortcuts
  (a binary32 square root proved equal on all 2³² inputs, ground-loss reuse, a
  cosine table for the 40 fixed ray angles, bit-keyed memos). The remaining
  limit is binary64 transcendentals (fast-math avoids them).

**Device memory.** The faster kernel reserves about **3.5 GB of local memory**
per process (CUDA sizes it for every resident thread), on top of the chunk
buffers. On a 16 GB card shared with a desktop, chunk 65,536 × 2 streams
(about 15 GB in total) overflows and Windows pages GPU memory, halving speed;
**use chunk 32,768 × 2 (about 10 GB) or smaller** on 16 GB. Building with
`-DVC_LUFFY_GLOBAL_STATE` keeps the old global-memory layout (about 3.9 GB at
chunk 32,768) at about 65% of the speed.

**Batch API.** `engine.predict_batch` / `voacap_gpu.batch.predict_batch` runs
many circuits in one GPU job (one TX to many RX, or explicit pairs) and
returns `(N, H, F)` arrays; `r.circuit(i)` equals `predict()` for circuit i
bit for bit. It reorders circuits internally so those taking the same kernel
branches share a warp (path-length class, then hour, then position) and
returns answers in your order: a shuffled world grid runs 2.3× faster than
unordered (302,000 vs 133,000/s at its default chunk). 200 circuits take
about 33× less time than a `predict()` loop.

**Legacy replay benchmark.** `tools/real_world_compare.py` replays pre-built
captured circuits (setup not timed) against a stored 28-frequency Fortran
baseline; both are marked LEGACY and the script now also prints the
like-for-like table.

## Layout

- `src/voacap_gpu/` — `scalar/`, `batched/`, `cuda/`, `engine.py`, and
  `data/coeffs/` (the bundled CCIR/URSI coefficient files). `engine.py` is a
  facade: the endpoint's code lives in `system.py`, `status.py`,
  `antennas.py`, `circuit.py`, `backends.py`, `results.py`, `rigcache.py` and
  `endpoint.py`, and every name is still importable from `voacap_gpu.engine`.
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
