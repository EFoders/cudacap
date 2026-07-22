#!/bin/bash
# run_findf_dumps.sh — tier-B dump for FINDF (area coverage / ray-set search).
# FINDF + its callees PENANG and the setup chain SANG/FOBBY/GENION/LECDEN/
# GETHP/BENDY/PEN/XLIN are plain COMMON-block F77 routines (no modules, no
# runtime coeffs); the driver injects a synthetic AFAC ramp.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_findf_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/findf"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_findf_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/findf.for" "$VSRC/penang.for" "$VSRC/fobby.for" "$VSRC/sang.for" \
      "$VSRC/genion.for" "$VSRC/lecden.for" "$VSRC/gethp.for" \
      "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/xlin.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_findf "$SCRIPT_DIR/dump_findf.f" "${srcs[@]}"
./dump_findf

DEST="$REPO_ROOT/golden/dumps/findf"
mkdir -p "$DEST"
cp findf_cases.csv "$DEST/cases.csv"
cp findf_dump.csv "$DEST/expected.csv"
cp findf_ifob.csv "$DEST/ifob.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_findf.f + findf/penang/fobby/sang/genion/lecden/gethp/bendy/pen/xlin"
  echo "inputs: controlled FI/YI/HI + IEDP + GCDKM + AMIND + FREQ (AFAC ramp injected)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "findf: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/findf/"
