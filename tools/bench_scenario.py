"""Project a real workload: 100 transmitters over a world grid, 24 h, 3-30 MHz.

Measures the per-lane cost of each pipeline stage at a batch size big enough to
be compute-bound, then multiplies out the work decomposition.  It is a
PROJECTION, not an end-to-end run -- there is no composed driver yet, and at 1
degree resolution the largest case is 4.4 billion evaluations, which nothing is
going to run as a benchmark.  What makes it more than arithmetic is that the
rates are measured on this machine at a batch size in the flat part of the
curve, and the decomposition is the one the physics actually has.

THE DECOMPOSITION IS THE POINT, and it is not uniform:

  * the IONOSPHERE at a point depends on (lat, lon, month, SSN, UT) and on
    NOTHING ELSE -- not on which transmitter is being considered, and not on
    the operating frequency.  So a global map is one evaluation per
    (grid point, hour), shared by all 100 transmitters and all 28 frequencies.
  * the REFLECTRIX (the oblique-frequency table) likewise depends only on the
    ionosphere and the take-off angle scan, so it is per (sample area, hour).
  * the RAY SEARCH is per (circuit, hour, frequency): FINDF looks for a
    specific operating frequency in that table.
  * the LOSS BUDGET is per (circuit, hour, frequency, hop).

Counting the ionosphere per CIRCUIT instead of per GRID POINT costs a factor
of 100 here, which is why both are reported.  Nothing in the current code
enforces either choice -- there is no driver yet -- so this is a design input,
not a measurement of a decision already made.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from bench_stage_b import bench_back, bench_findf, bench_front  # noqa: E402

HOURS = 24
FREQS = 28                      # 3..30 MHz, 1 MHz step
ANTENNAS = 100
# average live hops per (circuit, hour, frequency); the loss budget runs once
# per hop order, and VOACAP's sweep is IHSRT..IHSTP
HOPS = 3.0
# the back half is REGMOD plus INMUF, ALLMODES, GENOIS, RELBIL, SERPRB and
# MPATH.  REGMOD is the largest by some way, so 3x REGMOD is used as the
# stand-in -- an ESTIMATE, and labelled as one in the output.
BACK_MULTIPLIER = 3.0

# one circuit, 24 h x 28 freq, METHOD 30, measured on this machine's WSL build
FORTRAN_MS_PER_CIRCUIT = 17.5


def _grid_points(deg):
    """A lat/lon grid at `deg` resolution: poles included, longitude wrapped."""
    nlat = int(round(180.0 / deg)) + 1
    nlon = int(round(360.0 / deg))
    return nlat * nlon


def _fmt(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 86400:.1f} days"


def measure(n, device):
    """Per-lane microseconds for each stage, at a batch size that is past the
    launch-bound knee."""
    print(f"  measuring at {n:,} lanes on {device} ...", flush=True)
    t0 = time.perf_counter()
    front = bench_front(n, device) * 1000.0 / n
    findf = bench_findf(n, device) * 1000.0 / n
    back = bench_back(n, device) * 1000.0 / n
    print(f"  ... {time.perf_counter() - t0:.0f} s: front {front:.3f} "
          f"findf {findf:.3f} back {back:.3f} us/lane", flush=True)
    return front, findf, back


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="batch size to measure the per-lane rates at")
    ap.add_argument("--cpu-n", type=int, default=100_000)
    args = ap.parse_args()

    print(f"\nScenario: {ANTENNAS} transmitters x world grid x {HOURS} h x "
          f"{FREQS} freqs (3-30 MHz)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rates = {}
    if torch.cuda.is_available():
        rates["cuda"] = measure(args.n, "cuda")
    rates["cpu"] = measure(args.cpu_n, "cpu")

    hdr = (f"{'res':>5} {'grid pts':>10} {'circuits':>12} "
           f"{'evaluations':>15} {'Fortran':>11}")
    for dev in rates:
        hdr += f" {dev + ' (map/pt)':>13} {dev + ' (map/ckt)':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for deg in (1, 2, 3, 4):
        pts = _grid_points(deg)
        circuits = ANTENNAS * pts
        evals = circuits * HOURS * FREQS
        fortran = circuits * HOURS * FREQS / (HOURS * FREQS) \
            * FORTRAN_MS_PER_CIRCUIT / 1000.0
        row = (f"{deg:>4}d {pts:>10,} {circuits:>12,} {evals:>15,} "
               f"{_fmt(fortran):>11}")
        for dev, (front, findf, back) in rates.items():
            # the ionosphere shared across transmitters and frequencies
            iono_shared = pts * HOURS
            # ... or recomputed per circuit, which is what a naive port does
            iono_perckt = circuits * HOURS
            ray = circuits * HOURS * FREQS
            loss = ray * HOPS
            base = (ray * findf + loss * back * BACK_MULTIPLIER) / 1e6
            row += (f" {_fmt(base + iono_shared * front / 1e6):>13}"
                    f" {_fmt(base + iono_perckt * front / 1e6):>14}")
        print(row)

    print(f"\nFortran baseline: {FORTRAN_MS_PER_CIRCUIT} ms per circuit "
          f"({HOURS} h x {FREQS} freq, METHOD 30), single-threaded, measured.")
    print(f"Back half taken as {BACK_MULTIPLIER}x REGMOD -- an ESTIMATE; the "
          "other six routines are not separately timed here.")
    print(f"Hops per (circuit, hour, freq) assumed {HOPS}.")
    print("'map/pt' shares the ionosphere across transmitters and "
          "frequencies; 'map/ckt' recomputes it per circuit.")


if __name__ == "__main__":
    main()
