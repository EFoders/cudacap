#!/bin/bash
# run_frqcom_dumps.sh — tier-B dump for FRQCOM + SETLUF, the driver's two ends
# of the frequency list, driven in HFMUFS's order.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_frqcom_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/frqcom"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_frqcom_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_frqcom "$SCRIPT_DIR/dump_frqcom.f" \
    "$VSRC/frqcom.for" "$VSRC/setluf.for"
./dump_frqcom

DEST="$REPO_ROOT/golden/dumps/frqcom"
mkdir -p "$DEST"
cp frqcom_cases.csv "$DEST/cases.csv"
cp frqcom_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_frqcom.f + frqcom + setluf"
  echo "inputs: 31 injected columns (see gen_frqcom_cases.py COLUMNS):"
  echo "        IT, IFREQ, the five /MUFS/ slots at IT, LUFP, FREL(1..11),"
  echo "        a seed for FREL(13) and RELIAB(1..11)."
  echo "outputs: QFREL(1..13) -- FREL as FRQCOM left it, slot 13 included so"
  echo "         the case-1 arm can be seen NOT zeroing it -- then SETLUF's"
  echo "         XLUF(IT) and FREL(13)."
  echo "NOTE: case 3 is filtered to NE <= 10.  Beyond that the Fortran writes"
  echo "      FREA(NE+1) past the end of the common block; the port raises."
  echo "NOTE: FRQCOM case 6 is unreachable -- its guard is (FHIGH-2.)-FLOW"
  echo "      with both DATA constants, i.e. the constant 36."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "frqcom: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/frqcom/"
