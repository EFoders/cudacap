#!/bin/bash
# run_lngpat_dumps.sh — tier-B dump for LNGPAT, the long path's single mode:
# SELTXR's one chosen ray at each end, averaged into one geometry and priced
# end to end (convergence, over-the-top segment, absorption, gains, deciles).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_lngpat_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/lngpat"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_lngpat_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_lngpat "$SCRIPT_DIR/dump_lngpat.f" \
    "$VSRC/lngpat.for" "$VSRC/babs.for" "$VSRC/tabs.for" "$VSRC/convh.for" \
    "$VSRC/gettop.for" "$VSRC/prbmuf.for" "$VSRC/pwrdb.for"
./dump_lngpat

DEST="$REPO_ROOT/golden/dumps/lngpat"
mkdir -p "$DEST"
cp lngpat_cases.csv "$DEST/cases.csv"
cp lngpat_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_lngpat.f + lngpat + babs + tabs + convh + gettop"
  echo "         + prbmuf + pwrdb   (all callees linked real, not stubbed)"
  echo "inputs: 65 columns (see gen_lngpat_cases.py COLUMNS) — 6 integers"
  echo "        plus the two chosen rays' geometry and losses, the /MUFS/"
  echo "        four-layer distributions, the GETTOP E layer, and the loose"
  echo "        scalars.  Only two slots of each (45,3) table are touched, so"
  echo "        two values per table are injected rather than whole arrays;"
  echo "        they cannot collide because ITXRCP(2) is never 1."
  echo "outputs: /allMODE/ NMMOD + slot 1 of twenty arrays + slot 2's B and"
  echo "         NMODE + PROB(1), and /SIGD/ DSLF/DSUF, which LNGPAT clobbers"
  echo "NOTE: every output is preset to a distinctive sentinel (-8888, -4444,"
  echo "      -3333, -2222) so the LTXRGM <= 0 early return is visible as"
  echo "      untouched rather than as a written zero."
  echo "NOTE: held out of the grid — NMODE outside 1..4 (indexes YMUF off the"
  echo "      end of a four-element array; the port raises), HPFLX <= 0"
  echo "      (SQRT of a negative), GDFLX == 0 (divide by zero in HN/FHOP),"
  echo "      and YMUF/YFOT/YHPF <= 0 (PRBMUF's step branch can return 0 and"
  echo "      make -10*ALOG10(P) infinite)."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "lngpat: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/lngpat/"
