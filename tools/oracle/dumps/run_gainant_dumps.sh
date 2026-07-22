#!/bin/bash
# run_gainant_dumps.sh — tier-B dump for GAIN's antenna branch (ITR>=0, P2P).
# The driver injects sample.14 into /cantenna/ itself, so ANTCALC's card-deck
# parsing (out of scope) is not needed and gain.for links alone.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_gainant_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
SAMPLES="$HOME/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/gainant"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_gainant_cases.py" "$WORK"
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_gainant "$SCRIPT_DIR/dump_gainant.f" "$VSRC/gain.for"
./dump_gainant

DEST="$REPO_ROOT/golden/dumps/gainant"
mkdir -p "$DEST"
cp gainant_cases.csv "$DEST/cases.csv"
cp gainant_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_gainant.f + gain (antenna branch, ITR>=0)"
  echo "inputs: itr,delta_rad,fmc; /cantenna/ loaded from antennas/samples/sample.14"
  echo "        (the tests read the byte-identical copy under reference/)"
  echo "entries: 1=TX 2-15MHz, 2=RX 2-30MHz, 3=TX 15-30MHz (+3 dB, EFF+0.5)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "gainant: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/gainant/"
