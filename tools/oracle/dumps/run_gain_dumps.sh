#!/bin/bash
# run_gain_dumps.sh — tier-B dump for GAIN's ground-reflection branch (ITR<0).
# That branch touches only /GEOG/ SIGPAT/EPSPAT, so no antenna tables and no
# coeffs are needed; gain.for is linked alone.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_gain_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/gain"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_gain_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_gain "$SCRIPT_DIR/dump_gain.f" "$VSRC/gain.for"
./dump_gain

DEST="$REPO_ROOT/golden/dumps/gain"
mkdir -p "$DEST"
cp gain_cases.csv "$DEST/cases.csv"
cp gain_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_gain.f + gain (ground-reflection branch only)"
  echo "inputs: ig,sigma,er,delta_rad,fmc injected into /GEOG/ SIGPAT/EPSPAT"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "gain: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/gain/"
