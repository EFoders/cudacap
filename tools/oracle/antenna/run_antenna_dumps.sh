#!/bin/bash
# run_antenna_dumps.sh -- capture the antenna oracle decks through the LUFFY
# shim (tools/oracle/dumps/run_luffy_dumps.sh), isolated from the main replay
# set: own deck dir, own itshfbc copy (so concurrent captures do not share
# run/voacapx.dat), own shim work dir, own destination.
#
# Usage (from Windows PowerShell, repo root):
#   wsl bash "/mnt/c/.../VOACAP GPU/tools/oracle/antenna/run_antenna_dumps.sh"
# Regenerate the decks first with
#   uv run python tools/oracle/antenna/gen_antenna_decks.py
#
# LUFFY_REFERENCE must point at a vendored tree with voacapl/src (the repo's
# reference/ by default; override it in the environment as for the shim).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DECKS="$REPO_ROOT/golden/p2p/antenna_decks"

ITS="${LUFFY_ITSHFBC:-$HOME/itshfbc_ant}"
[[ -d "$ITS" ]] || cp -r "$HOME/itshfbc" "$ITS"

mapfile -t NAMES < <(cd "$DECKS" && ls *.dat | sed 's/\.dat$//' | sort)
[[ ${#NAMES[@]} -gt 0 ]] || { echo "no decks in $DECKS" >&2; exit 1; }

LUFFY_DECKS="$DECKS" \
LUFFY_ITSHFBC="$ITS" \
LUFFY_WORK="${LUFFY_WORK:-$HOME/voacapl-oracle/dumps/ant}" \
LUFFY_DEST="${LUFFY_DEST:-golden/dumps/luffy_antenna}" \
    bash "$REPO_ROOT/tools/oracle/dumps/run_luffy_dumps.sh" "${NAMES[@]}"

# Antenna provenance: the exact files ANTCALC read for these decks.
DEST="$REPO_ROOT/${LUFFY_DEST:-golden/dumps/luffy_antenna}"
case "${LUFFY_DEST:-}" in /*) DEST="$LUFFY_DEST" ;; esac
{ echo "antenna files (sha256, from $ITS/antennas):"
  (cd "$ITS/antennas" && sha256sum samples/sample.00 samples/sample.13 \
      samples/sample.14 samples/sample.14x | sed 's/^/  /')
} >> "$DEST/META.txt"
