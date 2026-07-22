#!/bin/bash
# run_serprb_dumps.sh — tier-B dump for SERPRB + MPATH (service probability
# and multipath probability), driven together in LUFFY's order.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_serprb_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/serprb"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_serprb_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_serprb "$SCRIPT_DIR/dump_serprb.f" \
    "$VSRC/serprb.for" "$VSRC/mpath.for" "$VSRC/fnorml.for"
./dump_serprb

DEST="$REPO_ROOT/golden/dumps/serprb"
mkdir -p "$DEST"
cp serprb_cases.csv "$DEST/cases.csv"
cp serprb_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_serprb.f + serprb + mpath + fnorml"
  echo "inputs: 158 injected columns (see gen_serprb_cases.py COLUMNS):"
  echo "        18 scalars + 7 /allMODE/ arrays x 20 slots.  NREL is injected"
  echo "        rather than produced, so neither routine depends on RELBIL."
  echo "outputs: /allMODE/ SPRO x 20, /SON/ SPROB+PROBMP at IFX, the /DON/"
  echo "         D10S/D50S/D90S triple, /TON/ SPR"
  echo "NOTE: NMMOD is never 0 -- SERPRB reads uninitialised locals there"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "serprb: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/serprb/"
