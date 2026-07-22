#!/bin/bash
# run_geotim_dumps.sh — tier-B dump for GEOTIM + CNGTIM, the UT/LMT
# conversions that open every hour of the prediction.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_geotim_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/geotim"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_geotim_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_geotim "$SCRIPT_DIR/dump_geotim.f" \
    "$VSRC/geotim.for" "$VSRC/cngtim.for"
./dump_geotim

DEST="$REPO_ROOT/golden/dumps/geotim"
mkdir -p "$DEST"
cp geotim_cases.csv "$DEST/cases.csv"
cp geotim_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_geotim.f + geotim + cngtim"
  echo "inputs: 13 injected columns (see gen_geotim_cases.py COLUMNS):"
  echo "        JT, ITIM, TLONG/RLONG (radians), KM, CLONG(1..5), and a"
  echo "        standalone CNGTIM triple (TIME, XTHETA in degrees, ISW)."
  echo "outputs: CNGTIM's rewritten TIME and its day-change return, then"
  echo "         GEOTIM's IT, GMT, GMTR, UTIME(IT), XLMT(IT), CLCK(1..5)."
  echo "NOTE: the standalone CNGTIM call is made FIRST and on its own copies,"
  echo "      so it cannot disturb the slots GEOTIM then fills."
  echo "NOTE: GEOTIM discards every day-change CNGTIM returns -- a prediction"
  echo "      hour carries no date.  Only the direct call can observe it."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "geotim: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/geotim/"
