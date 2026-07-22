#!/bin/bash
# run_libm_dump.sh — raw REAL*4 intrinsic dump (the f32math emulation
# target: the glibc flt-32 sinf/cosf/acosf/atanf the oracle links).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_libm_dump.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK="$HOME/voacapl-oracle/dumps/libm"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_libm_cases.py" "$WORK/libm_cases.csv"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch"
cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_libm "$SCRIPT_DIR/dump_libm.f"
./dump_libm

DEST="$REPO_ROOT/golden/dumps/libm"
mkdir -p "$DEST"
cp libm_cases.csv "$DEST/cases.csv"
cp libm_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  ldd ./dump_libm | grep -o 'libm[^ ]*' | head -1 || true
  echo "flags: $FLAGS"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "libm: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/libm/"
