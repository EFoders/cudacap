#!/bin/bash
# run_selmod_dumps.sh — tier-B dump for SELMOD (controlling sample-area select).
# Self-contained /RON//FRQ/ routine (no callees, no coeffs).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_selmod_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/selmod"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_selmod_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_selmod "$SCRIPT_DIR/dump_selmod.f" "$VSRC/selmod.for"
./dump_selmod

DEST="$REPO_ROOT/golden/dumps/selmod"
mkdir -p "$DEST"
cp selmod_cases.csv "$DEST/cases.csv"
cp selmod_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_selmod.f + selmod"
  echo "inputs: kfx,fi11,fi12,fi13,fi31,fi33"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "selmod: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/selmod/"
