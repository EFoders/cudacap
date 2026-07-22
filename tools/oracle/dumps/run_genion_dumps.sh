#!/bin/bash
# run_genion_dumps.sh — tier-B dump for GENION (Module-4 ionogram).
# GENION + its callees LECDEN/GETHP/BENDY/PEN/XLIN are plain COMMON-block F77
# routines (no modules, no runtime coeffs), fed controlled FI/YI/HI + a
# per-case IEDP to isolate the ionogram from the coefficient chain and to
# exercise both the full-Gaussian and fast height models.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_genion_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/genion"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_genion_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/genion.for" "$VSRC/lecden.for" "$VSRC/gethp.for" \
      "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/xlin.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_genion "$SCRIPT_DIR/dump_genion.f" "${srcs[@]}"
./dump_genion

DEST="$REPO_ROOT/golden/dumps/genion"
mkdir -p "$DEST"
cp genion_cases.csv "$DEST/cases.csv"
cp genion_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_genion.f + genion/lecden/gethp/bendy/pen/xlin"
  echo "inputs: controlled FI/YI/HI + per-case IEDP (no coefficient chain)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "genion: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/genion/"
