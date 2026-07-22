#!/bin/bash
# run_syssy_dumps.sh — tier-B dump for SYSSY (system-loss decile table lookup).
# Links SYSSY + a real REDMAP (for the SYS/PERR tables in /TWO/) and the
# modules REDMAP needs; requires the converted coefficients under ~/itshfbc.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_syssy_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/syssy"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_syssy_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_syssy \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" "$MODS/cversn.f95" \
    "$SCRIPT_DIR/dump_syssy.f" "$VSRC/syssy.for" "$VSRC/redmap.for"
./dump_syssy

DEST="$REPO_ROOT/golden/dumps/syssy"
mkdir -p "$DEST"
cp syssy_cases.csv "$DEST/cases.csv"
cp syssy_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_syssy.f + syssy/redmap + modules"
  echo "inputs: month,g_rad,pt,nn,izf; REDMAP loads SYS/PERR (SSN=100 CCIR)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "syssy: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/syssy/"
