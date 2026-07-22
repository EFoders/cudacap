#!/bin/bash
# run_fobby_dumps.sh — tier-B dump for FOBBY (Module-4 reflectrix table).
# FOBBY + its ionogram callees GENION/LECDEN/GETHP/BENDY/PEN/XLIN are plain
# COMMON-block F77 routines (no modules, no runtime coeffs).  The driver sets
# /RAYS/ ANG(1)+NANG directly (SANG is validated separately) and /CON/, feeds
# controlled FI/YI/HI + IEDP + a single angle per case.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_fobby_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/fobby"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_fobby_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/fobby.for" "$VSRC/genion.for" "$VSRC/lecden.for" \
      "$VSRC/gethp.for" "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/xlin.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_fobby "$SCRIPT_DIR/dump_fobby.f" "${srcs[@]}"
./dump_fobby

DEST="$REPO_ROOT/golden/dumps/fobby"
mkdir -p "$DEST"
cp fobby_cases.csv "$DEST/cases.csv"
cp fobby_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_fobby.f + fobby/genion/lecden/gethp/bendy/pen/xlin"
  echo "inputs: controlled FI/YI/HI + IEDP + one take-off angle per case"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "fobby: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/fobby/"
