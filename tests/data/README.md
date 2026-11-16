# Test data

Small input files the tests read, copied byte-exact (CRLF line endings
included) from the vendored voacapl tree that was removed from this
repository in commit `477dad7`. Copying them means a clean clone can run the
suite without `reference/`.

| File | Size (bytes) | SHA-256 | Original path (at `477dad7^`) |
|---|---|---|---|
| `antennas/sample.11` | 970 | `3d739a19809f5674124f3bb579eafefebb57637fb4ab6dec04116061a6417060` | `reference/voacapl/voacapl/itshfbc/antennas/samples/sample.11` |
| `antennas/sample.14` | 22625 | `9b60054ddacc84db313eb738c95c863e9a763e2158d4ecd0a6a7c139fb28df27` | `reference/voacapl/voacapl/itshfbc/antennas/samples/sample.14` |
| `antennas/sample.13` | 269150 | `2e70b96023b4dbbfe37992ce8b57f15aad3491c9a9efac7bf37ad5be61130282` | `reference/voacapl/voacapl/itshfbc/antennas/samples/sample.13` |
| `antennas/sample.14x` | 22625 | `7667680e46eb45f1b6380161c833fa58b7993f5f65aadef2aa9ede851fb7ef53` | `reference/voacapl/voacapl/itshfbc/antennas/samples/sample.14x` |

`sample.11` and `sample.14` were extracted with
`git show 477dad7^:<original path> > tests/data/antennas/<name>`.
`sample.13` (type 13, 360 azimuths x 91 elevations, PARM(1) 0) and
`sample.14x` (sample.14 with PARM(1) "Max Gain dBi" 3.00 and efficiency
-4.70) were copied with `cp` from the oracle's installed data,
`~/itshfbc/antennas/samples/` in the WSL oracle environment -- the tree the
gfortran voacapl reads at run time.  Their SHA-256 equals that of the
vendored originals at `477dad7^` (checked with `git show ... | sha256sum`).
They are what ANTCALC read when `golden/dumps/luffy_antenna` was captured
(the hashes are also recorded in that directory's `META.txt`), and
`tests/test_antenna_oracle.py` parses them with cudacap's own READANT.

## Provenance

These are stock sample antenna files from the ITSHFBC/VOACAP distribution, as
vendored with voacapl (https://github.com/jawatson/voacapl, commit
`c12a98b348c39f36f2659b8b3aa97b86284c8a1d`). The original VOACAP/ITSHFBC
code and data are a US Government (NTIA/ITS) work and are not subject to
copyright in the US. J.A. Watson's voacapl port changes are CC0.

They are the same bytes the gfortran oracle read when it generated the
`golden/dumps` fixtures, so the tier-B comparisons stay bit-exact.

## Not copied

- `coeffs/fof2daly.asc` (234437 bytes) is too large to duplicate here. The
  tests that need it are marked `oracle` and read it from `reference/`.
  (`sample.13` used to be listed here too; it is now copied, above, for the
  antenna oracle test.  `tests/scalar/test_readant.py` still reads its own
  copy from `reference/`.)
- The CCIR/URSI coefficient files (`coeffNN.asc`, `fof2CCIR.asc`,
  `fof2URSI.asc`) are not copied here because the package ships them in
  `src/voacap_gpu/data/coeffs`, byte-identical to the vendored sources.

To restore the vendored tree for oracle work:
`git checkout 477dad7^ -- reference`.
