#!/bin/bash
# run_lngabs_dumps.sh — tier-B dump for BABS + TABS + CONVH, the three pure
# leaves LNGPAT calls (absorbing-leg loss, over-the-top leg loss, and the
# geometrical convergence factor).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_lngabs_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/lngabs"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_lngabs_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_lngabs "$SCRIPT_DIR/dump_lngabs.f" \
    "$VSRC/babs.for" "$VSRC/tabs.for" "$VSRC/convh.for"
./dump_lngabs

DEST="$REPO_ROOT/golden/dumps/lngabs"
mkdir -p "$DEST"
cp lngabs_cases.csv "$DEST/cases.csv"
cp lngabs_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_lngabs.f + babs + tabs + convh"
  echo "inputs: 10 columns — freq,gyz3,acav,del,hp,df,dm,gd,phe,ray."
  echo "        FREQ/GYZ(3)/ACAV are injected into the commons; DEL and HP"
  echo "        are shared by all three routines exactly as LNGPAT shares"
  echo "        them.  /CON/ is DATA-initialised in the driver from blkdat's"
  echo "        own values rather than linking blkdat for two constants."
  echo "outputs: AF (BABS), AM (TABS), CH and PTOT (CONVH)"
  echo "NOTE: FREQ+GYZ(3) is never negative (powf of a negative base with a"
  echo "      fractional exponent is NaN) and DEL stays below PIO2 (where"
  echo "      PSI underflows to zero and XMHOP is infinite).  RAY >= 0 and"
  echo "      DEL inside a right angle keep CONVH's ALOG10 argument positive."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "lngabs: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/lngabs/"
