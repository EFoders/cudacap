#!/bin/bash
# run_readant_dumps.sh — tier-B dump for READANT's table paths (jant 11/13/14).
# The driver re-executes readant.for's READ statements on the stock sample
# antenna files, so nothing from voacapl needs linking.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_readant_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SAMPLES="$HOME/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/readant"
mkdir -p "$WORK"

# tag path — one antenna file per line
cat >"$WORK/readant_cases.txt" <<EOF
s11 $SAMPLES/sample.11
s13 $SAMPLES/sample.13
s14 $SAMPLES/sample.14
EOF

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_readant "$SCRIPT_DIR/dump_readant.f"
./dump_readant

DEST="$REPO_ROOT/golden/dumps/readant"
mkdir -p "$DEST"
cp readant_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_readant.f (readant.for READ statements, verbatim)"
  echo "inputs: antennas/samples/sample.{11,13,14} -- the tests read the same"
  echo "        files from reference/, which are byte-identical to these"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "readant: $(($(wc -l <"$DEST/expected.csv") - 1)) values -> golden/dumps/readant/"
