#!/bin/bash
# run_anois1_dumps.sh — tier-B dump for ANOIS1 (1 MHz atmospheric noise).
# Links ANOIS1 + NOISY + a real REDMAP (for the /TWO/ noise maps and F2D) and
# the modules REDMAP needs; requires the converted coefficients under ~/itshfbc.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_anois1_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/anois1"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_anois1_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_anois1 \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" "$MODS/cversn.f95" \
    "$SCRIPT_DIR/dump_anois1.f" "$VSRC/anois1.for" "$VSRC/noisy.for" \
    "$VSRC/redmap.for"
./dump_anois1

DEST="$REPO_ROOT/golden/dumps/anois1"
mkdir -p "$DEST"
cp anois1_cases.csv "$DEST/cases.csv"
cp anois1_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_anois1.f + anois1/noisy/redmap + modules"
  echo "inputs: month,ssn,gmtr,rlat,rlong,rlongd,izf; REDMAP loads noise maps"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "anois1: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/anois1/"
