#!/bin/bash
# run_fdist_dumps.sh — tier-B dump for FDIST (up to six modes per hop).
# Runs the full ray-set chain (LECDEN/GENION/SANG/FOBBY/FINDF/PENANG) to build
# a realistic /REFLX/ table, then FDIST.  Plain COMMON-block F77, no coeffs.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_fdist_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/fdist"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_fdist_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/fdist.for" "$VSRC/findf.for" "$VSRC/penang.for" "$VSRC/fobby.for" \
      "$VSRC/sang.for" "$VSRC/genion.for" "$VSRC/lecden.for" "$VSRC/gethp.for" \
      "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/xlin.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_fdist "$SCRIPT_DIR/dump_fdist.f" "${srcs[@]}"
./dump_fdist

DEST="$REPO_ROOT/golden/dumps/fdist"
mkdir -p "$DEST"
cp fdist_cases.csv "$DEST/cases.csv"
cp fdist_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_fdist.f + fdist/findf/penang/fobby/sang/genion/lecden/gethp/bendy/pen/xlin"
  echo "inputs: FI/YI/HI+IEDP+GCDKM+AMIND+FREQ+GHOP; /REFLX/ dumped as the injected input"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "fdist: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/fdist/"
