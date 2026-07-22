#!/bin/bash
# run_dazel_dump.sh — generate the DAZEL tier-B oracle dump in WSL.
#
# Compiles dump_dazel.f against the vendored dazel.for with the SAME flags
# as the oracle build, runs it over the committed case grid, and copies the
# results into golden/dumps/dazel/ for the pytest suite.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_dazel_dump.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WORK="$HOME/voacapl-oracle/dumps/dazel"
mkdir -p "$WORK"

# Regenerate the case grid (deterministic; must match the committed copy).
python3 "$SCRIPT_DIR/gen_dazel_cases.py" "$WORK/dazel_cases.csv"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch"
gfortran $FLAGS -o "$WORK/dump_dazel" \
    "$SCRIPT_DIR/dump_dazel.f" \
    "$REPO_ROOT/reference/voacapl/src/voacapw/dazel.for"

cd "$WORK"
./dump_dazel

DEST="$REPO_ROOT/golden/dumps/dazel"
mkdir -p "$DEST"
cp "$WORK/dazel_cases.csv" "$DEST/cases.csv"
cp "$WORK/dazel_dump.csv" "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "source: reference/voacapl/src/voacapw/dazel.for"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"

echo "dump complete: $(wc -l <"$DEST/expected.csv") lines -> golden/dumps/dazel/"
