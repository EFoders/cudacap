#!/bin/bash
# run_gmloss_dumps.sh — tier-B dump for GMLOSS + SELTXR, driven in LUFFY's own
# order: GMLOSS (which presets /ZON/ and calls SETTXR) and then SELTXR, which
# picks the one ray per end that LNGPAT will use.
#
# Both are dispatchers over an original and an "Alex" variant chosen on
# VERSN(8:8); the driver plants the vendored version.w32 string so both select
# the ORIGINAL, which is what the port implements.
#
# /cantenna/ is loaded from the VENDORED antenna samples, not a copy.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_gmloss_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
VMOD="$REPO_ROOT/reference/voacapl/src/modules"
SAMPLES="$REPO_ROOT/reference/voacapl/voacapl/itshfbc/antennas/samples"

WORK="$HOME/voacapl-oracle/dumps/gmloss"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_gmloss_cases.py" "$WORK"
cp "$SAMPLES/sample.14" "$WORK/ant14.voa"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -c "$VMOD/cversn.f95"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_gmloss "$SCRIPT_DIR/dump_gmloss.f" cversn.o \
    "$VSRC/gmloss.for" "$VSRC/seltxr.for" "$VSRC/settxr.for" \
    "$VSRC/gain.for" "$VSRC/prbmuf.for"
./dump_gmloss

DEST="$REPO_ROOT/golden/dumps/gmloss"
mkdir -p "$DEST"
cp gmloss_cases.csv "$DEST/cases.csv"
cp gmloss_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_gmloss.f + cversn + gmloss + seltxr + settxr + gain"
  echo "         + prbmuf   (all callees linked real, not stubbed)"
  echo "inputs: 87 columns — SETTXR's grid (imported from gen_settxr_cases,"
  echo "        so the two stay in step) plus SELTXR's DELOPT, GMIN, YMIN"
  echo "        and the /INFORM/ mask.  The /REFLX/ ray tables are generated"
  echo "        from a base and a step per column per control area."
  echo "outputs: 314 columns — /ZON/ 21 fields at slots 1 and 7 (the preset"
  echo "         loop's ends), TXRGML x 45 x 2, LTXRGM x 2, and GML/FHP x 45"
  echo "         x 2 control areas"
  echo "NOTE: TXRGML is preset to -8888 because GMLOSS does NOT preset it."
  echo "      /ZON/ is preset to -9999 for the same reason in reverse: the"
  echo "      preset loop is most of what GMLOSS actually accomplishes."
  echo "NOTE: INFO bit 4 is carried deliberately.  It makes SELTXR run its"
  echo "      whole selection twice, and the second pass inherits XHPM and"
  echo "      YHPM from the first, so the diagnostic flag can change the"
  echo "      selected ray.  Both settings are in the grid."
  echo "NOTE: held out of the grid, on top of SETTXR's own exclusions —"
  echo "      IAFTXR at exactly 45 with every ray existing and none clearing"
  echo "      AMIND drives SELTXR's find-first search to ray 46, which the"
  echo "      Fortran reads out of the column.  NFILL is capped at 44."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "gmloss: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/gmloss/"
