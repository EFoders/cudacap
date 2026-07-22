#!/bin/bash
# run_inmuf_dumps.sh — tier-B dump for INMUF (over-the-MUF / zero-distance
# modes, then REGMOD).  LECDEN+GENION build a real ionogram, which is dumped
# as an injected input because INMUF's XLIN interpolates it.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_inmuf_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
SAMPLES="$HOME/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/inmuf"
mkdir -p "$WORK"

( cd "$SCRIPT_DIR" && python3 gen_inmuf_cases.py "$WORK" )
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/inmuf.for" "$VSRC/regmod.for" "$VSRC/prbmuf.for" \
      "$VSRC/gain.for" "$VSRC/pwrdb.for" "$VSRC/xlin.for" \
      "$VSRC/genion.for" "$VSRC/lecden.for" "$VSRC/gethp.for" \
      "$VSRC/bendy.for" "$VSRC/pen.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_inmuf "$SCRIPT_DIR/dump_inmuf.f" "${srcs[@]}"
./dump_inmuf

DEST="$REPO_ROOT/golden/dumps/inmuf"
mkdir -p "$DEST"
cp inmuf_cases.csv "$DEST/cases.csv"
cp inmuf_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_inmuf.f + inmuf + regmod + prbmuf + gain + pwrdb + xlin"
  echo "         + genion + lecden + gethp + bendy + pen"
  echo "inputs: 126 injected columns (see gen_inmuf_cases.py COLUMNS)"
  echo "        plus the post-chain FVERT/HPRIM/HTRUE/AFAC ionogram AND the"
  echo "        post-chain FI/HI (LECDEN+GENION adjust them), the FIRST"
  echo "        126 columns of expected.csv -- injected INPUTS, not results"
  echo "outputs: nummod, ghop, /MODES/ 6x6, /MUFS/ 5x4, /ZON/ 20x7"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "inmuf: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/inmuf/"
