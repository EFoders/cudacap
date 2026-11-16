"""Fortran benchmark decks with the SAME workload as the GPU benchmark:
TX Ottawa, RX sampled from the 2-degree world grid, 9 frequencies, month 7,
SSN 120, CCIR, METHOD 30, System() defaults (RSN 73, noise 145, AMIND 0.1,
LUFP 90, PMP 3, DMP 0.1), 1 kW, isotropic 0 dBi both ends, hours 1..24.

Writes  single/NNNN.dat  (one circuit per deck = one voacapl process each)
and     onehour/NNNN.dat (the first 64 at TIME 12 12, to size process start-up).
"""
import pathlib, sys
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "decks"   # generated; not committed
N_SINGLE = 640
FREQ = " 3.50 5.00 7.0010.0014.0018.0021.0024.9028.00 0.00 0.00"
TLAT, TLON = 45.42, -75.70


def ns(v, pos, neg):
    return (abs(v), pos if v >= 0 else neg)


def circuit_card(rlat, rlon):
    tl, tn = ns(TLAT, "N", "S"); to, te = ns(TLON, "E", "W")
    rl, rn = ns(rlat, "N", "S"); ro, re = ns(rlon, "E", "W")
    return (f"CIRCUIT   {tl:5.2f}{tn}{to:9.2f}{te}{rl:9.2f}{rn}"
            f"{ro:9.2f}{re}  S     0")


HEAD = [
    "COMMENT   cudacap benchmark deck",
    "LINEMAX      55       number of lines-per-page",
    "COEFFS    CCIR",
    "TIME          1   24    1    1",
    "MONTH      1994 7.00",
    "SUNSPOT    120.",
    "LABEL     OTTAWA              GRID",
]
BODY = [
    "SYSTEM       1. 145. 0.10  90. 73.0 3.00 0.10",
    "FPROB      1.00 1.00 1.00 0.00",
    "ANTENNA       1    1    2   30     0.000[samples/sample.00    ]"
    "  0.0    1.0000",
    "ANTENNA       2    2    2   30     0.000[samples/sample.00    ]"
    "  0.0    0.0000",
    f"FREQUENCY {FREQ}",
    "METHOD       30    0",
]


def main():
    rng = np.random.default_rng(7)
    # 2-degree world grid, as gridjob.grid_cells builds it (avoid the poles,
    # where VOACAP's GEOM nudges coordinates)
    lats = np.arange(-88.0, 89.0, 2.0)
    lons = np.arange(-180.0, 180.0, 2.0)
    pts = [(la, lo) for la in lats for lo in lons]
    pick = rng.choice(len(pts), N_SINGLE, replace=False)
    sel = [pts[i] for i in pick]
    (OUT / "single").mkdir(parents=True, exist_ok=True)
    for i, (la, lo) in enumerate(sel):
        lines = HEAD + [circuit_card(la, lo)] + BODY + ["EXECUTE", "QUIT"]
        (OUT / "single" / f"{i:04d}.dat").write_text("\n".join(lines) + "\n", newline="\n")
    # one-hour variants (TIME 12 12): t(1 h) ~ process start-up + one hour,
    # so per-hour compute ~ (t(24 h) - t(1 h)) / 23
    (OUT / "onehour").mkdir(parents=True, exist_ok=True)
    head1 = [("TIME         12   12    1    1" if l.startswith("TIME") else l) for l in HEAD]
    for i, (la, lo) in enumerate(sel[:64]):
        lines = head1 + [circuit_card(la, lo)] + BODY + ["EXECUTE", "QUIT"]
        (OUT / "onehour" / f"{i:04d}.dat").write_text("\n".join(lines) + "\n", newline="\n")
    np.save(OUT / "rx_points.npy", np.array(sel, np.float64))
    print(f"wrote {N_SINGLE} 24-hour + 64 one-hour decks to {OUT}")


if __name__ == "__main__":
    main()
