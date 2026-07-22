#!/bin/bash
# run_setlng_dumps.sh — tier-B dump for SETLNG (sample-area replication).
# SETLNG is a self-contained COMMON-block routine with no callees; the driver
# injects distinct exact-integer sentinels into every touched cell of areas
# 1..5, runs SETLNG, and dumps 151 witness cells covering every destination
# range (see dump_setlng.f).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_setlng_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/setlng"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_setlng_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_setlng "$SCRIPT_DIR/dump_setlng.f" "$VSRC/setlng.for"
./dump_setlng

DEST="$REPO_ROOT/golden/dumps/setlng"
mkdir -p "$DEST"
cp setlng_cases.csv "$DEST/cases.csv"
cp setlng_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_setlng.f + setlng"
  echo "inputs: kfx,salt -> distinct exact-int sentinels in areas 1..5"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "setlng: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/setlng/"
