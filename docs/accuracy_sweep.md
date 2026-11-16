# Accuracy sweep: cudacap vs voacapl over 2,200 random circuits

This sweep generated 2,200 random point-to-point decks, each covering a full
24 hours. Each deck ran through the real voacapl, and through cudacap rebuilt
from the deck's raw cards. Every /SON/ output was then compared.

**Result.** 92.97% of compared values are bit-identical, and hop counts agree
in 99.9987% of cells. The CPU backend equals CUDA bit for bit. Every
CI bound in `tests/test_engine.py` holds at the 99.9th percentile. None holds
as a worst case over the sweep: 132 of 317,304 cells (0.042%, in 42 decks)
exceed at least one bound, and 4 cells (all in one deck) disagree on the hop
count. Every one of these outliers traces to the float32 libm floor. glibc's
`cosf`, `sinf` and `acosf` round differently from cudacap's correctly rounded
host math on a small fraction of arguments, and either GEOM's or FOBBY's
conditioning amplifies that difference. The sweep found no port bug in the
propagation path. It found one small f32-fidelity difference in the engine's
own antenna builder (PWRDBA, see the antenna section).

## Terms

* **deck**: one voacapl input file, i.e. one circuit for 24 hours.
* **cell**: one (deck, hour, frequency slot). The sweep has 317,304 cells.
* **|d|**: the absolute difference |cudacap - voacapl|, computed in float64
  from the two float32 values.
* **bit-exact**: the two float32 values are identical. The headline fraction
  is taken over the nine bounded fields x all cells.
* **bound**: the per-field absolute bound `SON_ABS_BOUND` asserted by
  `tests/test_engine.py::test_end_to_end_vs_oracle`. Signal fields (dbw,
  sndb, dbu, dblos, snpr) 0.01 dB, reliab 2e-4, angle 2e-3 deg, delay 3e-3 ms,
  vhigh 0.4 km.
* **floored ULP**: |d| divided by the float32 spacing at max(|oracle|, 1).
  This is the test's `ULP_BAR` metric (25,000).
* **regime**: which LUFFY model ran. *short* is IPFLAG 100. *long* is IPFLAG
  200, for paths of 10,000 km or more. *smooth* is METHOD 30 at
  7,000-10,000 km, where LUFFY runs both models and blends them.
* **replay**: the scalar LUFFY run from the oracle's captured LUFFY *entry*
  state, rather than from cudacap's own setup chain. If the replay matches the
  oracle and the end-to-end run does not, the error entered in the setup chain.
* **libm floor**: the 1-ULP disagreement between glibc's float32
  transcendentals and correctly rounded ones. See docs/f32_libm.md.

## What was swept

`tools/oracle/sweep/gen_sweep.py`, **seed 20261115**, 2,200 decks. Card
formats are those of `tools/oracle/golden_decks/gen_matrix.py`.

| input | range | how drawn |
|---|---|---|
| TX latitude | +-89.5 deg | 1/3 beyond +-60, 1/4 within +-15, rest area-uniform |
| RX | random bearing from TX | great-circle distance from six equal buckets spanning 100-19,900 km (log-uniform below 1,000 km) |
| path | S / L | 10% long path (`L`, NPSL 1). These give up to 39,500 km |
| month, SSN | 1-12, 0-200 | uniform integers |
| METHOD | 20 / 25 / 30 | uniform |
| coefficients, FPROB Es | CCIR / URSI, 0.00 / 0.70 | independent, uniform |
| frequencies | 1-11 of them, 2.00-30.00 MHz | distinct. 80% ascending, 20% in random order |
| SYSTEM | noise 140-165, min take-off 0.10-10.00 deg, reliability 50-99%, required SNR 30.0-90.0 dB-Hz, PMP 0.50-10.00 dB, DMP 0.05-5.00 ms | uniform |
| antennas | `samples/sample.00` isotrope at both ends | TX gain -10..+20 dBi (TX card's design-frequency field). RX gain -10..+20 dBi (RX card's power field). TX power 0.01-1000 kW, log-uniform. Beam headings random |

Coverage: 987 decks have an endpoint beyond +-60 deg. 1,018 have an endpoint
within +-15 deg. The geometry splits into 1,481 short, 604 long and 115 smooth
decks. Every input is inside cudacap's validated ranges.

**Oracle.** `~/voacapl-oracle` built with GNU Fortran (GCC) 15.2.1 20260123
(Red Hat 15.2.1-7), flags `-cpp -w -fno-backslash -ffixed-line-length-none
-fno-sign-zero -g -O2 -std=legacy -fallow-argument-mismatch`, taken from the
real link line, on Fedora 42 in WSL2. LUFFY was shimmed in by
`tools/oracle/dumps/run_luffy_dumps.sh` (8 shards, each with its own itshfbc
copy). **2,200 of 2,200 decks captured**, 24 LUFFY rows each. No deck
crashed and none was skipped.

**cudacap side** (`tools/oracle/sweep/compare_sweep.py`). Each deck is
parsed in DECRED's fixed columns. Each hour gets a fresh `build_circuit`, all
24 sharing one REDMAP, with the captured /cantenna/. This is
`tests/test_engine.py::run_deck`'s configuration, and it reuses that file's
`_rows`, `_antennas` and `_deck_method`. Everything runs on CUDA through
`predict_circuits`, 100 decks per launch. The parsed card values were checked
against the captured entry state (rsn, amind, lufp, pmp, dmp, noise, psc,
month, SSN, frequencies, method, NPSL, hours 1-24), with no mismatch on any
deck.

## Distributions

All 317,304 cells, numeric /SON/ fields. Every hour returned status 0.

| field | units | max | p99.9 | p99 | median | bit-exact | bound | field-cells over |
|---|---|---|---|---|---|---|---|---|
| dbw | dBW | 0.0982 | 0.00366 | 0.000412 | 0 | 92.21% | 0.01 | 110 |
| sndb | dB-Hz | 0.0982 | 0.00366 | 0.000412 | 0 | 92.14% | 0.01 | 110 |
| dbu | dB(uV/m) | 0.113 | 0.00366 | 0.000412 | 0 | 91.64% | 0.01 | 114 |
| dblos | dB | 0.0982 | 0.00366 | 0.000412 | 0 | 92.48% | 0.01 | 110 |
| snpr | dB | 0.0982 | 0.00366 | 0.000412 | 0 | 91.81% | 0.01 | 111 |
| reliab | fraction | 0.00046 | 6.85e-06 | 5.96e-07 | 0 | 96.98% | 2e-4 | 1 |
| angle | deg | 0.0188 | 0.000325 | 4.77e-05 | 0 | 94.73% | 2e-3 | 19 |
| delay | ms | 0.00648 | 7.63e-05 | 1.53e-05 | 0 | 92.80% | 3e-3 | 1 |
| vhigh | km | 0.652 | 0.00708 | 0.00119 | 0 | 91.94% | 0.4 | 1 |
| angler | deg | 0.00235 | 4.86e-05 | 0 | 0 | 99.38% | - | |
| cprob | fraction | 0.000336 | 6.44e-06 | 1.24e-06 | 0 | 96.22% | - | |
| dblosl | dB | 0.0371 | 0.000337 | 3.05e-05 | 0 | 96.53% | - | |
| dblosu | dB | 0.0193 | 0.00043 | 3.81e-05 | 0 | 95.79% | - | |
| snrlw | dB | 0.033 | 0.00024 | 2.29e-05 | 0 | 96.48% | - | |
| snrup | dB | 0.0188 | 0.000388 | 3.15e-05 | 0 | 95.66% | - | |
| sprob | fraction | 0.00015 | 6.14e-06 | 4.77e-07 | 0 | 97.21% | - | |
| xnynois | dBW | 1.53e-05 | 1.53e-05 | 0 | 0 | 99.90% | - | |
| probmp | fraction | 3.6e-05 | 0 | 0 | 0 | 99.92% | - | |
| rneff | dB | 0 | 0 | 0 | 0 | 100.00% | - | |

* Hop count (`nhp`): 317,300 of 317,304 cells agree (99.9987%).
* Bit-exact over the nine bounded fields: 92.97%.
* Floored ULP: the worst is 205,882, and 14 cells exceed `ULP_BAR`. All 14
  are deep over-the-MUF losses or snpr values near zero.
* Over-bound cells are mostly deep over-the-MUF modes. Of the 115 cells over a
  signal bound, 111 have an oracle dbw below -150 dBW and 99 below -200 dBW.
  The worst relative error on dblos is 5.2e-4. The strongest signal that breaks
  a bound is sw1714 h4 at 3.81 MHz: dbw -45.921 vs -45.964.

### Breakdowns

"cells over" counts cells exceeding at least one bound, counted once per
field.

**By METHOD**

| METHOD | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 736 | 105,576 | 93.14% | 99.996% | 0.042 | 0.00017 | 0.0053 | 0.0011 | 0.17 | 90 |
| 25 | 703 | 104,088 | 93.28% | 100% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 291 |
| 30 | 761 | 107,640 | 92.51% | 100% | 0.073 | 0.00019 | 0.0042 | 0.0025 | 0.13 | 196 |

**By regime**

| regime | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| short | 1,481 | 215,472 | 93.01% | 100% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 318 |
| smooth (7,000-10,000 km, M30) | 115 | 16,296 | 91.48% | 100% | 0.073 | 0.00019 | 0.0014 | 0.00049 | 0.086 | 47 |
| long (10,000 km or more) | 604 | 85,536 | 93.15% | 99.995% | 0.048 | 0.00017 | 0.0038 | 0.0025 | 0.13 | 212 |

**By propagation path length** (GEOM's GCDKM. The long way round counts as its full length)

| km | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-1,000 | 308 | 44,304 | 94.08% | 100% | 0.042 | 2.8e-05 | 0.0053 | 0.0011 | 0.17 | 11 |
| 1,000-3,000 | 332 | 46,776 | 93.27% | 100% | 0.0053 | 0.00011 | 0.0017 | 0.00013 | 0.039 | 0 |
| 3,000-7,000 | 345 | 50,136 | 91.98% | 100% | 0.047 | 9.9e-05 | 0.017 | 0.002 | 0.3 | 19 |
| 7,000-10,000 | 330 | 46,800 | 92.53% | 100% | 0.073 | 0.00019 | 0.0029 | 0.00055 | 0.086 | 63 |
| 10,000-15,000 | 342 | 48,096 | 92.48% | 99.992% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 276 |
| 15,000-20,000 | 319 | 47,112 | 93.22% | 100% | 0.018 | 6.5e-05 | 0.0009 | 0.00064 | 0.076 | 45 |
| 20,000 or more (long path) | 224 | 34,080 | 93.53% | 100% | 0.048 | 0.00017 | 0.0038 | 0.0025 | 0.13 | 163 |

**By coefficients / Es**

| coeff, FPROB Es | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| CCIR, 0.00 | 551 | 79,056 | 93.01% | 100% | 0.023 | 0.00017 | 0.0049 | 0.00095 | 0.12 | 99 |
| CCIR, 0.70 | 576 | 83,112 | 92.72% | 100% | 0.087 | 0.00019 | 0.0051 | 0.0017 | 0.14 | 346 |
| URSI, 0.00 | 551 | 79,920 | 93.87% | 99.995% | 0.048 | 0.00011 | 0.0034 | 0.00063 | 0.12 | 23 |
| URSI, 0.70 | 522 | 75,216 | 92.26% | 100% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 109 |

**By the larger endpoint |latitude|**

| deg | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-20 | 465 | 66,648 | 95.43% | 100% | 0.047 | 8.1e-05 | 0.0034 | 0.00095 | 0.12 | 37 |
| 20-40 | 394 | 55,920 | 93.46% | 100% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 285 |
| 40-60 | 354 | 51,672 | 91.77% | 100% | 0.029 | 0.00017 | 0.0042 | 0.0025 | 0.13 | 74 |
| 60-75 | 523 | 77,448 | 92.54% | 99.995% | 0.014 | 0.00011 | 0.0049 | 0.00071 | 0.12 | 32 |
| 75-90 | 464 | 65,616 | 91.51% | 100% | 0.073 | 0.00019 | 0.0026 | 0.00064 | 0.086 | 149 |

**By SSN**

| SSN | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-50 | 519 | 73,104 | 92.83% | 100% | 0.087 | 0.00017 | 0.0053 | 0.0017 | 0.17 | 337 |
| 50-100 | 521 | 76,560 | 91.58% | 99.995% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 18 |
| 100-150 | 545 | 79,344 | 93.79% | 100% | 0.057 | 8.1e-05 | 0.017 | 0.002 | 0.3 | 70 |
| 150-200 | 615 | 88,296 | 93.56% | 100% | 0.073 | 0.00019 | 0.0049 | 0.0025 | 0.13 | 152 |

**By path flag**

| CIRCUIT | decks | cells | bit-exact | nhp agree | max dbw | max reliab | max angle | max delay | max vhigh | cells over |
|---|---|---|---|---|---|---|---|---|---|---|
| short (`S`) | 1,976 | 283,224 | 92.90% | 99.999% | 0.098 | 0.00046 | 0.019 | 0.0065 | 0.65 | 414 |
| long (`L`) | 224 | 34,080 | 93.53% | 100% | 0.048 | 0.00017 | 0.0038 | 0.0025 | 0.13 | 163 |

No input dimension isolates the outliers. The cell counts are lumpy because a
few decks dominate: sw0759 alone has 47 of the 132 cells, which is why "CCIR,
0.70" and "SSN 0-50" lead. Equatorial paths are the most bit-exact group
(95.4%). sw0112, the worst cell, sits in the 10,000-15,000 km, METHOD 25,
URSI + Es and 20-40 deg rows.

## CPU backend equals CUDA

100 random decks (2,400 circuit-hours, seed 20261116) ran again on
`backend="cpu"`. All 100 are identical to the CUDA run, bit for bit, across all
19 /SON/ numeric fields, `nhp` and status.

## cudacap's own antenna builder

200 random decks (seed 20261117) were rebuilt with `build_circuit(cant=None)`.
The engine then calls `isotropic_cant(power_kw, tx_gos, rx_gos)` with the
card's TX power and isotrope gains, and sets OFFAZIM from the card's beam
headings.

* Against the oracle: max |d| is dbw 0.0225 dB, reliab 4.8e-5, angle 2.6e-3
  deg, delay 4.9e-4 ms and vhigh 0.061 km, with no hop-count disagreement.
  Each maximum is the same as the captured-antenna run's maximum on the same
  200 decks.
* Against the captured-antenna run: 172 of 200 decks are identical. In the
  other 28, the only input difference is **PWRDBA**, off by 1 ULP (49.159992 vs
  49.159996 for 82.41 kW). The cause is that `engine._finish_cant` computes
  `30 + 10*log10(kW)` in float64 and rounds once. DECRED/ANTCALC compute
  `30. + 10.*ALOG10(PWRKW)` in REAL*4, and
  `F32(F32(30) + F32(10)*log10_32(F32(kW)))` reproduces the oracle PWRDBA on
  200 of 200 decks. PWRDBA differs on 47 decks, and 28 of those change an
  output, by at most a few 1e-5 dB. This is a genuine but negligible port
  difference, and the one fix this sweep suggests.
* Gain tables for 2-30 MHz match the capture exactly. Two differences cannot
  affect the sweep. The engine's isotrope spans 0-30 MHz, while ANTCALC
  installs only the card's 2-30 MHz rows and leaves 0 dBi below them, so the
  two differ only for frequencies under 2 MHz, which the sweep does not use.
  OFFAZIM differs by up to 0.08 deg, because ANTCALC stores it through an F7.2
  field and takes its bearing from DAZEL0, not GEOM. OFFAZIM is inert for an
  isotrope.

## Outliers and their causes

42 decks have a cell over a bound or a hop-count disagreement (the full list
is in `outliers.tsv`). Each was bisected with `tools/oracle/sweep/diagnose.py`
into two classes. *setup* means LUFFY replayed from the captured entry state
matches the oracle, so the error enters before LUFFY. *LUFFY* means the replay
already shows the error. Each was then put through
`tools/oracle/sweep/libm_noise.py`, which re-runs the hour with 1-ULP random
perturbations of the float32 trig functions, to test whether noise of exactly
the libm's kind reaches the observed |d|.

**LUFFY class: 28 decks.** These are sw0080, 0112, 0229, 0426, 0449, 0492,
0634, 0779, 0842, 0909, 1025, 1104, 1244, 1267, 1325, 1371, 1389, 1416, 1432,
1539, 1629, 1641, 1789, 1929, 2046, 2085, 2173 and 2197. In 18 of them every
setup field agrees with the capture to under one floored ULP, so cudacap and
voacapl enter LUFFY with essentially identical state and still disagree. Restricting the noise to `cos32` inside
FOBBY alone reproduces the observed |d| in 27 of the 28 decks, to 2-4
significant figures. For example: sw1539 0.11344 observed vs 0.11346 from
noise, sw1025 0.016838 vs 0.016838, sw0080 0.004925 vs 0.004925. The remaining
deck, sw1104, reaches 0.78 of its observed |d|. This is the mechanism
`tests/scalar/test_luffy_replay.py` documents: FOBBY's `IFOB = 1000.*FREQ`
truncates to integer kHz, a 1-ULP `cosf` disagreement tips the truncation, and
FINDF's ionogram interpolation turns one kHz into ~1e-3 relative in the ray
table. Over the MUF, where loss rises steeply with frequency, that becomes
hundredths of a dB.

**Setup class: 14 decks.** These are sw0008, 0168, 0173, 0265, 0759, 0768,
0892, 0943, 1171, 1275, 1462, 1704, 1714 and 1835. With trig noise in the
whole chain, every one reaches at least its observed |d| (ratio 1.0-180), and
sw1704's hop count flips under noise. For GEOM the floor is proven exactly.
`tools/oracle/sweep/geom_libm.py` restates geom.for's distance, bearing and
control-point formulas in float32 with a pluggable libm. With glibc's real
`sinf`/`cosf`/`acosf`, called through ctypes in WSL, it reproduces the
oracle's GCD, CLAT and CLONG **bit for bit on all 2,200 decks**. With
correctly rounded trig it reproduces cudacap's GEOM bit for bit on all 2,200
decks. The two disagree on 229 decks. So every GEOM difference in the sweep is
the libm, and none is the port. The amplifiers:

* **Near-2pi long paths**: sw0008 (554 km taken the long way, 39,470 km),
  sw1462 (144 km) and sw1835 (428 km). BTR =
  ACOS((sin RLAT - sin TLAT cos GCD)/(cos TLAT sin GCD)) divides by
  sin GCD of about -0.09. In sw0008, a 1-ULP GCD (6.1961946 vs 6.1961951 from
  `acosf`) moves the bearing by 1.1e-4 rad. Every control point moves with it
  (CLONG04 1.0714808 vs 1.0713496), and the worst cell is dblos 240.151 vs
  240.180 at h3, 23.47 MHz.
* **Near-polar endpoints and near-unity ACOS arguments** in the control-point
  longitude: sw0168 (85.0N), sw1275 (86.6N), sw0892 (84.9S) and sw0759 (CLONG
  off by up to 3.1e-4 rad on a 14,146 km path). The derivative of ACOS near
  +-1 multiplies a 1-ULP input difference by hundreds.
* **Ionosphere-chain trig** downstream of an essentially exact GEOM: sw1714,
  sw0173 and sw0265. In sw1714, a 129 km NVIS path at 78.6 deg take-off, noise
  in any one of `sinf`, `cosf`, `acosf` or `atanf` reproduces the 0.042 dB.

**The hop-count disagreement** is one deck, sw1704 (0.60N 80.81W to 72.03N
155.49E, 11,037 km, METHOD 20, URSI). At h8-h11, 15.12 MHz, cudacap reports
nhp 1 and voacapl reports 2, while dbw agrees to 2e-3 dB. GCD is 1 ULP apart
(1.7327126 vs 1.7327125, `acosf`, reproduced by `geom_libm.py`). On the long
path `HN = 0.5*GCD/(DR+DT)` comes out 1.9999999 against exactly 2.0, and RELBIL's
`INT(HN)` truncates differently. The replay from the captured entry state
gives 2.0, matching the oracle. This is a discrete flip caused by the libm. It
is a reporting difference only: the signal values are unaffected.

**Most informative cells.**

| deck | path | hour, MHz | values (cudacap vs voacapl) | class |
|---|---|---|---|---|
| sw0112 | 25.6N 121.8E to 23.15S 2.87E, 13,877 km, M25, URSI + Es | h7, 24.55 | all nine fields over: dbw -140.333 vs -140.431, reliab 0.04369 vs 0.04323, angle 6.297 vs 6.316, delay 49.275 vs 49.281, vhigh 453.99 vs 454.64 | LUFFY, FOBBY `cosf` |
| sw1539 | 68.78S 66.96W to 51.52S 144.83E, `L`, M25, URSI + Es | h22, 2.07 | dbu -1011.376 vs -1011.490 (largest \|d\|, dbw at the -500 sentinel) | LUFFY, FOBBY `cosf` |
| sw1025 | 4.34S 138.40E to 29.81S 149.32E, 3,056 km, M25, URSI + Es | h19, 8.09 | angle 22.960 vs 22.944 deg (setup bit-exact) | LUFFY, FOBBY `cosf` |
| sw1714 | 27.56N 30.14E to 26.45N 30.52E, 129 km, M20, URSI + Es | h4, 3.81 | dbw -45.921 vs -45.964 (strongest signal over a bound) | setup, ionosphere trig |
| sw0008 | 554 km circuit, long way (39,470 km), M30 | h3, 23.47 | dblos 240.151 vs 240.180 | setup, GEOM `acosf` |
| sw1704 | 11,037 km, M20 | h8-h11, 15.12 | nhp 1 vs 2 | setup, `INT(HN)` flip |

None of the outliers sits at an input-domain edge that cudacap refuses or
mishandles. All are inside the validated ranges, and no status was non-zero.
The only "edge" flavour is geometric: long paths close to 2pi and endpoints
within ~5 deg of a pole, where GEOM's formulas are ill-conditioned in float32
on both sides.

## CI bounds over the sweep

| field | bound | sweep worst | sweep p99.9 | holds? |
|---|---|---|---|---|
| dbw / sndb / dblos / snpr | 0.01 dB | 0.098 dB | 0.0037 dB | no (110-111 cells) |
| dbu | 0.01 dB | 0.113 dB | 0.0037 dB | no (114 cells) |
| reliab | 2e-4 | 4.6e-4 | 6.9e-6 | no (1 cell, sw0112) |
| angle | 2e-3 deg | 0.0188 deg | 3.3e-4 deg | no (19 cells) |
| delay | 3e-3 ms | 6.5e-3 ms | 7.6e-5 ms | no (1 cell, sw0112) |
| vhigh | 0.4 km | 0.65 km | 0.0071 km | no (1 cell, sw0112) |
| `ULP_BAR` | 25,000 | 205,882 | - | no (14 cells) |
| nhp exact | - | 4 cells differ | - | no (sw1704) |

The bounds were set at ~5x the worst over 22 hand-picked decks. That is
right for those decks, but it is not a worst case over the input domain. The
same libm-floor mechanism produces about 10x larger excursions on rare cells,
nearly all of them over the MUF. Every bound holds at the 99.9th percentile.
The margins there are 2.7x for the signal fields, 6x for angle, and 29-56x for
reliab, delay and vhigh. A bound set
that covered this sweep at ~2x its worst would be: signal fields 0.25 dB,
angle 0.04 deg, delay 0.015 ms, reliab 1e-3, vhigh 1.5 km. Whether to widen
the bounds, keep them for the curated decks only, or emulate glibc's float32
trig (docs/f32_libm.md) is a project decision, so this sweep does not change
the tests.

Candidate regression decks are in `golden/dumps/luffy_sweep/`: sw0112, sw1539,
sw0008, sw1704, sw1714 and sw0759. Each has its `.dat` and its three CSVs, and
the README there says what each one exercises.

## Timings

RTX 5080, shared with other jobs during the run. The host is Windows 11 with
32 logical cores. voacapl runs under WSL2 on the same machine. A sweep deck
averages 6.0 frequencies.

| what | time |
|---|---|
| voacapl, stock binary, serial, per deck (24 hours) | mean 8.8 ms, median 7.6 ms, p99 17 ms, 19.3 s for all 2,200, i.e. 0.37 ms per circuit-hour |
| voacapl + LUFFY shim (CSV capture), 8 parallel shards | 14.5 ms per deck mean, 7.1 s wall for all 2,200 |
| cudacap `predict_circuits` on CUDA, whole sweep, excluding host setup | 29.0 s for 52,800 circuit-hours (549 us per circuit-hour), in 22 launches of 2,400 circuit-hours |
| of which `launch_luffy` (H2D + kernel + D2H) | 2.7 s (51 us per circuit-hour) |
| of which per-circuit Python `pack_circuit` / `unpack_circuit` | 6.9 s / 8.5 s (the rest is glue inside `predict_circuits`) |
| host setup (deck parse + REDMAP + 24 `build_circuit` per deck) | 239.9 s (4.5 ms per circuit-hour) |
| cudacap, whole sweep, including host setup | 268.9 s (5.1 ms per circuit-hour) |
| cudacap CPU backend (scalar LUFFY), 2,400 circuit-hours | 23.8 ms per circuit-hour (28.2 ms in an earlier identical run) |

Reading oracle CSVs over the WSL share (88 s) is excluded everywhere. On this
per-circuit path the GPU kernel is not the bottleneck: the Python setup chain
and per-circuit packing dominate, and end to end the path is slower than one
voacapl core. The device setup path (`setup_config` / `gridjob`) exists to
avoid exactly that host work.

## Reproduce

In WSL, from the repo (`/mnt/c/.../VOACAP GPU`; put these in a `.sh` file
if invoking from Windows):

    python3 tools/oracle/sweep/gen_sweep.py --out ~/sweep/decks      # seed 20261115, 2200 decks
    SWEEP_REFERENCE=/path/to/reference bash tools/oracle/sweep/run_sweep.sh
        # -> ~/sweep/dumps: CSVs, RUNLOG.tsv, STOCK_TIMING.tsv, META.txt
        # env: SWEEP_DECKS SWEEP_DEST SWEEP_JOBS(8) SWEEP_STOCK_N SWEEP_LIMIT

In Windows, from the repo root:

    uv run python tools/oracle/sweep/compare_sweep.py \
        --decks //wsl.localhost/FedoraLinux-42/home/$USER/sweep/decks \
        --dumps //wsl.localhost/FedoraLinux-42/home/$USER/sweep/dumps \
        --out build/sweep/results
        # summary.json, tables.md, decks.tsv, outliers.tsv, cells.npz
        # --cpu-n 100 --own-n 200 --chunk 100 by default

Bisecting an outlier:

    uv run python tools/oracle/sweep/diagnose.py --decks D --dumps D sw0112 --hours 7
    uv run python tools/oracle/sweep/diagnose.py --decks D --dumps D --table sw0008 sw1704 ...
    uv run python tools/oracle/sweep/libm_noise.py --decks D --dumps D \
        --outliers build/sweep/results/outliers.tsv --mode replay \
        --funcs cos32 --modules fobby --p 0.3 --seeds 48
    wsl python3 tools/oracle/sweep/geom_libm.py --check ~/sweep/decks ~/sweep/dumps sw0008

The six regression decks in `golden/dumps/luffy_sweep/` work directly as
`--decks` / `--dumps` for `diagnose.py`.
