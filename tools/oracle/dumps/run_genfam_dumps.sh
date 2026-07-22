#!/bin/bash
# run_genfam_dumps.sh — tier-B dump for GENFAM (atmospheric-noise vs frequency).
# Links GENFAM + a real REDMAP (for the FAM/DUD tables in /TWO/) and the
# modules REDMAP needs; requires the converted coefficients under ~/itshfbc.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_genfam_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/genfam"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_genfam_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_genfam \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" "$MODS/cversn.f95" \
    "$SCRIPT_DIR/dump_genfam.f" "$VSRC/genfam.for" "$VSRC/redmap.for"
./dump_genfam

DEST="$REPO_ROOT/golden/dumps/genfam"
mkdir -p "$DEST"
cp genfam_cases.csv "$DEST/cases.csv"
cp genfam_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_genfam.f + genfam/redmap + modules"
  echo "inputs: month,y2_rad,iblk,freq,z; REDMAP loads FAM/DUD (SSN=100 CCIR,"
  echo "        neither table is SSN-blended so only MONTH selects them)"
  echo "outputs: fa,dua,dla,dms,dus,dls"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "genfam: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/genfam/"
