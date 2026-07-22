#!/bin/bash
# run_alosfv_dumps.sh — tier-B dump for ALOSFV (Module-4 deviative loss).
# ALOSFV + its ionogram callees GENION/LECDEN/GETHP/BENDY/PEN/XLIN are plain
# COMMON-block F77 routines (no modules, no runtime coeffs).  The driver
# injects the per-layer MUF heights HPMUF/HTMUF directly (CURMUF is validated
# separately) plus controlled FI/YI/HI + IEDP.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_alosfv_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/alosfv"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_alosfv_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/alosfv.for" "$VSRC/genion.for" "$VSRC/lecden.for" \
      "$VSRC/gethp.for" "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/xlin.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_alosfv "$SCRIPT_DIR/dump_alosfv.f" "${srcs[@]}"
./dump_alosfv

DEST="$REPO_ROOT/golden/dumps/alosfv"
mkdir -p "$DEST"
cp alosfv_cases.csv "$DEST/cases.csv"
cp alosfv_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_alosfv.f + alosfv/genion/lecden/gethp/bendy/pen/xlin"
  echo "inputs: controlled FI/YI/HI + IEDP + injected HPMUF/HTMUF"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "alosfv: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/alosfv/"
