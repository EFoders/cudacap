#!/bin/bash
# run_genois_dumps.sh — tier-B dump for GENOIS (combined receiver noise).
# Links GENOIS + GENFAM + GAIN + a real REDMAP (for /TWO/) and the modules
# REDMAP needs; requires the converted coefficients under ~/itshfbc.
# /cantenna/ is loaded from the VENDORED antenna samples, not a copy.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_genois_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"
SAMPLES="$REPO_ROOT/reference/voacapl/voacapl/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/genois"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_genois_cases.py" "$WORK"
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_genois \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" "$MODS/cversn.f95" \
    "$SCRIPT_DIR/dump_genois.f" "$VSRC/genois.for" "$VSRC/genfam.for" \
    "$VSRC/gain.for" "$VSRC/redmap.for"
./dump_genois

DEST="$REPO_ROOT/golden/dumps/genois"
mkdir -p "$DEST"
cp genois_cases.csv "$DEST/cases.csv"
cp genois_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_genois.f + genois/genfam/gain/redmap + modules"
  echo "inputs: month,izf,freq,noise,rlat,atnu,atny,cc,tm,kj,jk,fi3,rxlo,rxhi"
  echo "        REDMAP loads F2D/FAM/DUD (SSN=100 CCIR); IZF forces"
  echo "        F2D(1,1,1) negative for the no-data-base branch"
  echo "        /cantenna/ entry 2 = antennas/samples/sample.14 over [rxlo,rxhi]"
  echo "outputs: /ANOIS/ rcnse,du,dl,sigm,sygu,sygl; /TON/ gnos,xnoise,znoise;"
  echo "         /ZON/ eff x 7; the /c_noise*/ breakdown + method_used"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "genois: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/genois/"
