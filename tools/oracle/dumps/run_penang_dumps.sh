#!/bin/bash
# run_penang_dumps.sh — tier-B dump for PENANG (per-layer penetration angle).
# PENANG + its ionogram callees GENION/LECDEN/GETHP/BENDY/PEN/XLIN are plain
# COMMON-block F77 routines (no modules, no runtime coeffs); the driver builds
# the ionogram from controlled FI/YI/HI and sets FREQ.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_penang_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/penang"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_penang_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/penang.for" "$VSRC/genion.for" "$VSRC/lecden.for" \
      "$VSRC/gethp.for" "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/xlin.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_penang "$SCRIPT_DIR/dump_penang.f" "${srcs[@]}"
./dump_penang

DEST="$REPO_ROOT/golden/dumps/penang"
mkdir -p "$DEST"
cp penang_cases.csv "$DEST/cases.csv"
cp penang_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_penang.f + penang/genion/lecden/gethp/bendy/pen/xlin"
  echo "inputs: controlled FI/YI/HI + IEDP + FREQ"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "penang: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/penang/"
