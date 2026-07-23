#!/bin/bash
# run_gettop_dumps.sh — tier-B dump for GETTOP, the over-the-top distance:
# how much of a long path the ray flies above the E layer (the night-day-night
# case, where a daylit middle reflects what the dark ends let through).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_gettop_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/gettop"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_gettop_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_gettop "$SCRIPT_DIR/dump_gettop.f" "$VSRC/gettop.for"
./dump_gettop

DEST="$REPO_ROOT/golden/dumps/gettop"
mkdir -p "$DEST"
cp gettop_cases.csv "$DEST/cases.csv"
cp gettop_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_gettop.f + gettop"
  echo "inputs: 9 columns — delopt,freq,dikm,hi1,hi2,hi3,fi1,fi2,fi3."
  echo "        HI/FI are the E layer only, HI(1,jf)/FI(1,jf) for the three"
  echo "        control areas.  DELOPT is injected rather than left at"
  echo "        blkdat's 3 degrees so the grid can reach the sin^2 clamp."
  echo "outputs: GMKM, the over-the-top distance in km (0 = none qualifies)"
  echo "NOTE: the generator emulates GETTOP's float32 arithmetic so it can"
  echo "      SEARCH for inputs that land each boundary on exactly zero."
  echo "      At physical frequencies the two 0.001 MHz slack boundaries"
  echo "      cannot be hit exactly (float32 0.001 is an odd multiple of"
  echo "      2^-33 and FREQ-FP is a multiple of ulp(FP)), so those exact"
  echo "      cases sit at kilohertz-scale FP instead; both sides are pinned"
  echo "      at physical scale by stepping FREQ one ULP at a time."
  echo "NOTE: the sin^2 clamp at 0.9999 is UNREACHABLE with real geometry"
  echo "      (DELOPT 3 degrees, E layer near 110 km gives sin^2 = 0.964)."
  echo "      Probed with DELOPT 0 / near-zero heights and with negative"
  echo "      heights, where without the clamp SQRT would take a negative."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "gettop: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/gettop/"
