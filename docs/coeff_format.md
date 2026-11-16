# CCIR/URSI coefficient data — format and ingestion (Module 0)

The ionospheric coefficient data VOACAP depends on ships with voacapl as
**ASCII sources** plus small Fortran converter programs, all vendored under
`reference/voacapl/voacapl/itshfbc/coeffs/`. The converters are the format
specification; nothing here was reverse-engineered from binaries.

> The `reference/` tree is no longer in the working tree (removed in commit
> `477dad7`); restore it with `git checkout 477dad7^ -- reference`. The ASCII
> coefficient files the engine reads at runtime are bundled in
> `src/voacap_gpu/data/coeffs/`.

| ASCII source | Converter | Binary product (oracle) | Consumed by |
|---|---|---|---|
| `coeff01.asc` … `coeff12.asc` | `coeffbin.f` | `coeffXX.bin` (sequential unformatted, 38,552 B) | ICEPAC/VOACAP/REC533 monthly load |
| `fof2CCIR.asc`, `fof2URSI.asc` | `fof2bin.f` | `fof2CCIR.da`, `fof2URSI.da` (direct access, recl=7904, 12 records) | foF2 maps, method-dependent CCIR/URSI choice |
| `fof2daly.asc` | `fof2bin.f` | `fof2daly.bin` (sequential unformatted, 9 records) | daily-foF2 option |

The `*w.bin` / `*.daw` twins (from `coefbinw.f`/`fof2binw.f`) exist for the
old Salford FTN77 Windows build; compiled with gfortran they are
byte-identical to the plain files and are ignored here.

Our ingestion lives in `src/voacap_gpu/coeffs.py` and parses the ASCII
directly (the binaries are never needed at runtime).
`tests/test_coeffs.py` proves every parsed array **bit-exact** against the
oracle-generated binaries pinned in `golden/coeffs/`
(`tools/oracle/copy_coeff_bins.sh` refreshes them after a
`build_oracle.sh` run).

## Text conventions

- Every array is preceded by a **name record** — a text line starting with
  the array spec (e.g. `DUD(5,12,5)`); the converters verify these tags and
  so does the parser.
- Real values use `FORMAT(6E15.8)`: six **fixed 15-character fields** per
  line, 8 significant digits, leading-dot mantissa (` .60209268E+00`,
  `-.41046128E+01`). Negative signs can fill the field's leading blank, so
  fields must be sliced by position, not whitespace-split.
- `fof2daly.asc` values are instead read **list-directed** (`READ (21,*)`)
  — whitespace-separated tokens (mixed `0.378737E+01` / `-.716914E-01`
  styles).
- Integer array IKIM uses `FORMAT(10I5)`: ten 5-character fields per line.
- All reals are REAL*4 (float32); IKIM is INTEGER (int32).

## `coeffXX.asc` (one per month)

Line 1: `month=MM ASCII Ionospheric coefficient file (except foF2)` — MM in
columns 7–8 (`FORMAT(6X,I2)`), verified against the file name.

Then 17 name-tagged READ groups, in this order (shapes are Fortran
dimensions; a group like `ANEW…BCHI` is ONE read sharing lines):

| Group | Values | Note |
|---|---|---|
| `IKIM(10,6)` | 60 int | per-array coefficient counts / limits |
| `DUD(5,12,5)` | 300 | |
| `FAM(14,12)` | 168 | |
| `SYS(9,16,6)` | 864 | |
| `FAKP(29,16,6)` | 2784 | |
| `FAKABP(2,6)` | 12 | |
| `XFM3CF(9,49,2)` | 882 | M(3000)F2 factor coefficients |
| `ANEW(3),BNEW(3),ACHI(2),BCHI(2)` | 10 | one read: 2 lines (6+4) |
| `FAKMAP(29,16)` | 464 | |
| `ABMAP(2,3)` | 6 | |
| `F2D(16,6,6)` | 576 | |
| `PERR(9,4,6)` | 216 | |
| `XESMCF(7,61,2)` | 854 | Es-median coefficients |
| `XPMAP(29,16,2)` | 928 | |
| `XESLCF(5,55,2)` | 550 | Es-lower-decile |
| `XESUCF(5,55,2)` | 550 | Es-upper-decile |
| `XERCOF(9,22,2)` | 396 | E-region coefficients |

Total 60 integers + 9,560 reals per month. Values stream in Fortran
column-major order; `coeffs.py` reshapes with `order='F'` so
`dud[i-1, j-1, k-1] == DUD(i,j,k)`.

Physical interpretation (established CCIR-atlas structure; to be confirmed
routine-by-routine in Module 2): trailing dimensions of size 2 are the two
solar-activity levels the maps are defined at, and the `X*CF` sets are
(diurnal Fourier order × geographic function × activity level).

### `coeffXX.bin` record layout (what coeffbin.f writes)

Nine sequential records, each the plain concatenation of arrays in memory
order (gfortran framing: little-endian `u4 len ⋯ u4 len`):

1. `IKIM`
2. `FAKP, FAKABP`
3. `DUD, FAM, SYS`
4. `XFM3CF`
5. `F2D, PERR`
6. `ANEW, BNEW, ACHI, BCHI, FAKMAP, ABMAP`
7. `XESMCF, XPMAP`
8. `XESLCF, XESUCF`
9. `XERCOF`

Payload 38,480 B + 9×8 B markers = 38,552 B. Records 1–4 are shared by
ICEPAC/VOACAP/REC533; 5–9 are ICEPAC/VOACAP only. `XF2COF` is in the
`/ccir/` COMMON but is **not** in these files — foF2 comes from the files
below.

## `fof2CCIR.asc` / `fof2URSI.asc` → `XF2COF(13,76,2)`

Line 1: title. Then 12 blocks: `month=MM` line + 1,976 values (330 lines,
6E15.8). Each month converts to one direct-access record of
recl = 13·76·2·4 = **7,904 bytes** (12 records, 94,848 B, no framing).
`load_fof2_atlas` returns float32 `(12, 13, 76, 2)` indexed `[month-1]`.

The 76 geographic functions × 13 diurnal Fourier terms are the standard
CCIR (Oslo 1966) / URSI-88 foF2 map representation, at the two activity
levels R12 = 0 and 100.

## `fof2daly.asc` → 9 sets of `XF2COF(13,76,2)`

Line 1: title; then 9 sets, each a one-line set title + 1,976
list-directed values. Written as 9 sequential records (71,208 B total).
`load_fof2_daily` returns float32 `(9, 13, 76, 2)`.

## Precision note

The ASCII decimals (8 significant digits) are parsed as
`np.float32(float(s))` — widen to binary64, then narrow. For ≤9-digit
decimals in this magnitude range the double rounding cannot differ from
gfortran's direct decimal→binary32 conversion, and the bit-exact fixture
comparison in `tests/test_coeffs.py` (~180k values) confirms it
empirically.
