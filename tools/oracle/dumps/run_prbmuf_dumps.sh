#!/bin/bash
# run_prbmuf_dumps.sh — tier-B dump for the PRBMUF probability function.
# PRBMUF is a self-contained /MUFS/ routine (no callees, no coeffs); the driver
# injects SIGL(IL)/SIGU(IL) per case.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_prbmuf_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/prbmuf"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_prbmuf_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_prbmuf "$SCRIPT_DIR/dump_prbmuf.f" "$VSRC/prbmuf.for"
./dump_prbmuf

DEST="$REPO_ROOT/golden/dumps/prbmuf"
mkdir -p "$DEST"
cp prbmuf_cases.csv "$DEST/cases.csv"
cp prbmuf_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_prbmuf.f + prbmuf"
  echo "inputs: fmhz,fgo,fset,il,siglil,siguil (SIGL/SIGU injected)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "prbmuf: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/prbmuf/"
