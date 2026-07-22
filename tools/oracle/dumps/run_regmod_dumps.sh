#!/bin/bash
# run_regmod_dumps.sh — tier-B dump for REGMOD (per-mode loss budget -> /ZON/).
# All inputs are injected, so only REGMOD and its three callees need linking.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_regmod_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
SAMPLES="$HOME/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/regmod"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_regmod_cases.py" "$WORK"
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/regmod.for" "$VSRC/prbmuf.for" "$VSRC/gain.for" "$VSRC/pwrdb.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_regmod "$SCRIPT_DIR/dump_regmod.f" "${srcs[@]}"
./dump_regmod

DEST="$REPO_ROOT/golden/dumps/regmod"
mkdir -p "$DEST"
cp regmod_cases.csv "$DEST/cases.csv"
cp regmod_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_regmod.f + regmod + prbmuf + gain + pwrdb"
  echo "inputs: 103 injected columns (see gen_regmod_cases.py COLUMNS)"
  echo "        /cantenna/ from antennas/samples/sample.14, TX entry 1 over"
  echo "        [txlo,txhi], RX entry 2 over 2-30 MHz"
  echo "outputs: /ZON/ 20 fields x 7 mode slots"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "regmod: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/regmod/"
