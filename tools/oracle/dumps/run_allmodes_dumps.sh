#!/bin/bash
# run_allmodes_dumps.sh — tier-B dump for ALLMODES (/ZON/ -> /allMODE/).
# The case file is one long CALL SEQUENCE, not independent cases: /allMODE/
# accumulates and the driver never clears it between records.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_allmodes_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/allmodes"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_allmodes_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_allmodes "$SCRIPT_DIR/dump_allmodes.f" \
    "$VSRC/allmodes.for"
./dump_allmodes

DEST="$REPO_ROOT/golden/dumps/allmodes"
mkdir -p "$DEST"
cp allmodes_cases.csv "$DEST/cases.csv"
cp allmodes_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_allmodes.f + allmodes"
  echo "inputs: iflg,fval + /ZON/ 23 fields x 7 (see gen_allmodes_cases.py)"
  echo "outputs: /allMODE/ 23 arrays x 20, /ZON/ HP x 7, nmmod, freq"
  echo "NOTE: records are a CALL SEQUENCE -- /allMODE/ carries across them"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "allmodes: $(($(wc -l <"$DEST/expected.csv") - 1)) calls -> golden/dumps/allmodes/"
