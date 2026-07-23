#!/bin/bash
# run_settxr_dumps.sh — tier-B dump for SETTXR, the long path's transmitter
# and receiver end tables: absorption, deviative loss, sporadic-E obscuration,
# ground loss and antenna gain for all 45 rays at each control area.
#
# settxr.for is a DISPATCHER over SETTXR_ORIG and SETTXR_ALEX, chosen on
# VERSN(8:8).  The driver plants the vendored database/version.w32 string
# ("16.1207W"), so the dispatcher picks the ORIGINAL — what the stock program
# runs, and what the port implements.  cversn.f95 is compiled for that.
#
# /cantenna/ is loaded from the VENDORED antenna samples, not a copy.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_settxr_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
VMOD="$REPO_ROOT/reference/voacapl/src/modules"
SAMPLES="$REPO_ROOT/reference/voacapl/voacapl/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/settxr"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_settxr_cases.py" "$WORK"
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -c "$VMOD/cversn.f95"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_settxr "$SCRIPT_DIR/dump_settxr.f" cversn.o \
    "$VSRC/settxr.for" "$VSRC/gain.for" "$VSRC/prbmuf.for"
./dump_settxr

DEST="$REPO_ROOT/golden/dumps/settxr"
mkdir -p "$DEST"
cp settxr_cases.csv "$DEST/cases.csv"
cp settxr_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_settxr.f + cversn + settxr + gain + prbmuf"
  echo "inputs: 83 columns (see gen_settxr_cases.py COLUMNS).  The /REFLX/"
  echo "        ray tables are generated from an injected BASE and STEP per"
  echo "        column per control area over NFILL rays, the rest zeroed;"
  echo "        every ray column is a threshold input somewhere, so a ramp"
  echo "        that crosses the threshold puts rays on both sides of it."
  echo "        /cantenna/ from antennas/samples/sample.14, TX entry 1 over"
  echo "        [txlo,txhi], RX entry 2 over [rxlo,rxhi]."
  echo "outputs: 523 columns — ANDVX/ADVX/AOFX/GRLOSX/TGAINX x 45 rays x 2"
  echo "         control areas, EFFlp x 45, ray 1 of the six /REFLX/ columns"
  echo "         plus IMODE at both areas (the over-the-MUF fallback"
  echo "         OVERWRITES it), GML/FHP at rays 1/23/45, and PROB(1..2)"
  echo "NOTE: AOFX is preset to -8888 and SETTXR is expected NOT to reset it."
  echo "      TGAINX/ANDVX/ADVX/GRLOSX are preset to -7777 and it does reset"
  echo "      all four at the top of every ray, so a dead ray shows their"
  echo "      presets and AOFX's SENTINEL side by side."
  echo "NOTE: held out of the grid — MODMUF outside 1..4 (indexes DELMUF and"
  echo "      YMUF off the end; the port raises), GDFLX == 0 (divide by zero"
  echo "      in the hop count), HTFLX <= 0 (SQRT of a negative, and also the"
  echo "      only route to CPHET's 1e-8 floor, which is therefore"
  echo "      unreachable for a live ray), FI(1,K) == 0, HS(K) == 0,"
  echo "      HNU == 0, XVE <= 0 and YMUF <= 0."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "settxr: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/settxr/"
