#!/usr/bin/env bash
# Pin the oracle-generated coefficient binaries as golden fixtures.
#
# build_oracle.sh runs voacapl's converters (coeffbin, fof2bin — built by
# `make` from reference sources) over the vendored ASCII coefficient files,
# leaving binaries in ~/itshfbc/coeffs/.  This script copies the ones the
# Python ingestion tests validate against into golden/coeffs/ and records
# hashes + toolchain in META.txt.
#
# Run from Windows:  wsl bash tools/oracle/copy_coeff_bins.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$HOME/itshfbc/coeffs"
DST="$REPO/golden/coeffs"

FILES=(coeff{01,02,03,04,05,06,07,08,09,10,11,12}.bin
       fof2CCIR.da fof2URSI.da fof2daly.bin)

for f in "${FILES[@]}"; do
    [[ -f "$SRC/$f" ]] || { echo "missing $SRC/$f — run build_oracle.sh first" >&2; exit 1; }
done

mkdir -p "$DST"
for f in "${FILES[@]}"; do
    cp "$SRC/$f" "$DST/$f"
done

{
    echo "Oracle-generated coefficient binaries (gfortran unformatted)."
    echo "Produced by voacapl's coeffbin/fof2bin converters from the vendored"
    echo "ASCII sources in reference/voacapl/voacapl/itshfbc/coeffs/."
    echo "Layout: docs/coeff_format.md.  Regenerate: build_oracle.sh, then this script."
    echo
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gfortran: $(gfortran --version | head -1)"
    echo "uname: $(uname -srm)"
    echo
    (cd "$DST" && sha256sum "${FILES[@]}")
} > "$DST/META.txt"

echo "pinned ${#FILES[@]} files into golden/coeffs/"
