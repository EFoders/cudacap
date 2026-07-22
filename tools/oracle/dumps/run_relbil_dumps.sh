#!/bin/bash
# run_relbil_dumps.sh — tier-B dump for RELBIL (reliability + combined modes).
# blkdat.for is linked so LAYTYP (the layer labels) comes from the real source
# rather than being transcribed into the driver.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_relbil_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/relbil"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_relbil_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/relbil.for" "$VSRC/fnorml.for" "$VSRC/pwrdb.for"
      "$VSRC/blkdat.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_relbil "$SCRIPT_DIR/dump_relbil.f" "${srcs[@]}"
./dump_relbil

DEST="$REPO_ROOT/golden/dumps/relbil"
mkdir -p "$DEST"
cp relbil_cases.csv "$DEST/cases.csv"
cp relbil_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_relbil.f + relbil + fnorml + pwrdb + blkdat"
  echo "inputs: 314 injected columns (see gen_relbil_cases.py COLUMNS):"
  echo "        14 scalars + 15 /allMODE/ arrays x 20 slots"
  echo "        D10R/D50R/D90R are seeded as well as dumped -- they live in"
  echo "        /DON/, persist across calls, and RELBIL does not always"
  echo "        assign them"
  echo "outputs: /allMODE/ CREL+RELY x 20, NREL, the D*R triple, /SON/ +"
  echo "         /DUDL_NOIS/ + /sncom/ + /cgains/ at column IFX, the /SIGD/"
  echo "         DSLF/DSUF scratch, and MODE(IFX) as its 1..5 LAYTYP index"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "relbil: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/relbil/"
