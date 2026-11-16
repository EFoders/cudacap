# luffy_sweep: candidate regression decks from the accuracy sweep

Six decks picked from the 2,200-deck randomised sweep (docs/accuracy_sweep.md,
seed 20261115). Each is the deck (`<deck>.dat`, written by
`tools/oracle/sweep/gen_sweep.py`) plus its LUFFY capture (`_in.csv`,
`_out.csv`, `_ant.csv`), which `tools/oracle/sweep/run_sweep.sh` made
through `tools/oracle/dumps/run_luffy_dumps.sh`'s shim, with gfortran 15.2.1
at `-O2 -std=legacy -fno-sign-zero`. The CSV formats are the ones in
`golden/dumps/luffy/META.txt`.

Every deck uses isotropic `samples/sample.00` antennas. The TX isotrope gain
comes from the TX ANTENNA card's design-frequency field, the RX isotrope gain
from the RX card's power field, and the TX power from the TX card's power
field (antcalc.for:119,136).

All six break at least one bound in `tests/test_engine.py` (`SON_ABS_BOUND`),
or disagree on a hop count. In every case the cause is the float32 libm floor:
glibc's `cosf`, `sinf` and `acosf` round differently from cudacap's correctly
rounded host math, and a routine's conditioning amplifies that difference. None
of the six is a port bug. If they go into CI, they need their own wider bounds,
or the glibc emulation described in docs/f32_libm.md. Do not just add them to
`ALL_DECKS`.

"setup" means the error enters before LUFFY: LUFFY replayed from the captured
entry state matches the oracle. "LUFFY" means the same replay already shows the
error.

| deck | what it exercises | worst cell (cudacap vs voacapl) | cause |
|---|---|---|---|
| sw0112 | 25.6N 121.8E to 23.15S 2.87E, 13,877 km, M25, URSI + Es 0.70, Jan, SSN 87. The only cell that breaks all nine bounds at once, on a usable-ish signal. | h7, 24.55 MHz: dbw -140.333 vs -140.431 (0.098 dB), reliab 0.04369 vs 0.04323, angle 6.297 vs 6.316 deg, delay 49.275 vs 49.281 ms, vhigh 453.99 vs 454.64 km | LUFFY. A 1-ULP `cosf` flip in FOBBY's integer-kHz truncation reproduces it to 3 figures. |
| sw1539 | 68.78S 66.96W to 51.52S 144.83E, long path (`L`), M25, URSI + Es 0.70. Over-the-MUF at 2.07 MHz with 26 hops. This is the largest \|d\| in the sweep. | h22, 2.07 MHz: dbu -1011.376 vs -1011.490 (0.113). dbw is the -500 sentinel on both sides. | LUFFY. A 1-ULP FOBBY `cosf` flip reproduces it to 4 figures. |
| sw0008 | 47.80S 58.75E to 52.78S 59.07E. A 554 km circuit taken the long way (`L`, 39,470 km, nearly 2 pi). M30, URSI + Es 0.70. | h3, 23.47 MHz: dblos 240.151 vs 240.180 (0.029 dB). 12 cells over. | setup. GEOM's `acosf` gives a GCD 1 ULP apart. BTR divides by sin(GCD) = -0.09, so the bearing moves 1.1e-4 rad and every control point moves with it (CLONG04 differs by 1,100 ULP). `geom_libm.py` with glibc reproduces the capture bit for bit. |
| sw1704 | 0.60N 80.81W to 72.03N 155.49E, 11,037 km, M20, URSI. The only hop-count disagreements in the sweep. | h8-h11, 15.12 MHz: nhp 1 vs 2, with dblos within 2e-3 dB | setup, then a discrete flip. GCD is 1 ULP apart (`acosf`), so the long-path hop number `HN = 0.5*GCD/(DR+DT)` is 1.9999999 against 2.0, and `INT(HN)` truncates differently. |
| sw1714 | 27.56N 30.14E to 26.45N 30.52E, 129 km NVIS (angle 78.6 deg), M20, URSI + Es 0.70, SSN 17. This is the strongest signal that breaks a bound. | h4, 3.81 MHz: dbw -45.921 vs -45.964 (0.042 dB), angle 78.596 vs 78.601 deg | setup. Ionosphere-chain trig noise, with GEOM exact to 1 ULP. Reproduced by 1-ULP noise in any one of `sinf`, `cosf`, `acosf` or `atanf`. |
| sw0759 | 32.94N 57.16E to 19.82N 122.63W, 14,146 km, M25, CCIR + Es 0.70, SSN 8. This deck has the most over-bound cells (47 of 132). | h1, 22.40 MHz: dblos 701.018 vs 701.073. h3: dblos 447.60 vs 447.69 (0.087 dB). | setup. GEOM control points differ by up to 1,307 ULP, again from glibc `acosf`/`cosf` (reproduced bit for bit). |

To reproduce a row:

    uv run python tools/oracle/sweep/diagnose.py --decks golden/dumps/luffy_sweep \
        --dumps golden/dumps/luffy_sweep sw0112 --hours 7
