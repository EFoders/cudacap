#!/bin/bash
# run_esmod_dumps.sh — tier-B dump for ESMOD + ESREG (the sporadic-E /ZON/
# slots).  Both are driven in LUFFY's order on disjoint slots.  All inputs are
# injected, so only the routines and their callees need linking.
# /cantenna/ is loaded from the VENDORED antenna samples, not a copy.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_esmod_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
SAMPLES="$REPO_ROOT/reference/voacapl/voacapl/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/esmod"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_esmod_cases.py" "$WORK"
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/esmod.for" "$VSRC/esreg.for" "$VSRC/prbmuf.for"
      "$VSRC/gain.for" "$VSRC/pwrdb.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_esmod "$SCRIPT_DIR/dump_esmod.f" "${srcs[@]}"
./dump_esmod

DEST="$REPO_ROOT/golden/dumps/esmod"
mkdir -p "$DEST"
cp esmod_cases.csv "$DEST/cases.csv"
cp esmod_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_esmod.f + esmod + esreg + prbmuf + gain + pwrdb"
  echo "inputs: 63 injected columns (see gen_esmod_cases.py COLUMNS)"
  echo "        /cantenna/ from antennas/samples/sample.14, TX entry 1 over"
  echo "        [txlo,txhi], RX entry 2 over [rxlo,rxhi]"
  echo "outputs: /ZON/ 23 fields x slots 4..7, after ESMOD then ESREG"
  echo "NOTE: all 7 slots are preset to 12345. (NMODE -9) before the calls,"
  echo "      so an unwritten field is visibly untouched rather than zero"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "esmod: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/esmod/"
