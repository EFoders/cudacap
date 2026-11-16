#!/bin/bash
# run_icepac_dumps.sh -- capture the ICEPAC-mode oracle decks through the
# LUFFY shim (tools/oracle/dumps/run_luffy_dumps.sh), without touching the
# committed tanbel_m07_s160_icepac capture or its META.txt.
#
# The shim CLEARS every CSV in its destination, so it is pointed at a
# scratch dest and the new <deck>_in/_out/_ant.csv are copied into
# golden/dumps/luffy_icepac/ afterwards; its META.txt is kept alongside as
# META_icepac_decks.txt.
#
# Usage (from Windows):  wsl bash "/mnt/c/.../tools/oracle/icepac/run_icepac_dumps.sh" [deck ...]
# Default: every deck in golden/p2p/icepac_decks/ (gen_icepac_decks.py).
#
# Overrides: ICE_REFERENCE (vendored tree with voacapl/src; default the repo's
# reference/), ICE_ITSHFBC (private itshfbc copy, made from ~/itshfbc if
# missing), ICE_WORK (shim build dir), ICE_TMPDEST (scratch capture dir).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DECKS="$REPO_ROOT/golden/p2p/icepac_decks"
FINAL="$REPO_ROOT/golden/dumps/luffy_icepac"
ITSHFBC="${ICE_ITSHFBC:-$HOME/itshfbc_ice}"
TMPDEST="${ICE_TMPDEST:-$HOME/voacapl-oracle/dumps/ice_dest}"

[[ -d "$ITSHFBC" ]] || cp -r "$HOME/itshfbc" "$ITSHFBC"

DECK_LIST=("$@")
if [[ ${#DECK_LIST[@]} -eq 0 ]]; then
    for f in "$DECKS"/*.dat; do DECK_LIST+=("$(basename "$f" .dat)"); done
fi

mkdir -p "$TMPDEST"
LUFFY_DECKS="$DECKS" \
LUFFY_REFERENCE="${ICE_REFERENCE:-$REPO_ROOT/reference}" \
LUFFY_ITSHFBC="$ITSHFBC" \
LUFFY_WORK="${ICE_WORK:-$HOME/voacapl-oracle/dumps/ice}" \
LUFFY_DEST="$TMPDEST" \
    bash "$REPO_ROOT/tools/oracle/dumps/run_luffy_dumps.sh" "${DECK_LIST[@]}"

mkdir -p "$FINAL"
for name in "${DECK_LIST[@]}"; do
    [[ "$name" == tanbel_m07_s160_icepac ]] && { echo "refusing to overwrite $name" >&2; exit 1; }
    for part in in out ant; do
        cp "$TMPDEST/${name}_${part}.csv" "$FINAL/${name}_${part}.csv"
    done
done
{ cat "$TMPDEST/META.txt"
  echo "deck source: golden/p2p/icepac_decks/ (tools/oracle/icepac/gen_icepac_decks.py)"
  echo "deck config: COEFFS URSI, FPROB 1.00 1.00 1.00 0.70 (Prop-SL ICEPAC mode)"
  echo "voacapl: $(ls -l --time-style=+%F "$HOME/voacapl-oracle/prefix/bin/voacapl" | awk '{print $6}') build at ~/voacapl-oracle/prefix/bin/voacapl"
  echo "sha256(voacapl): $(sha256sum "$HOME/voacapl-oracle/prefix/bin/voacapl" | cut -d' ' -f1)"
  echo "captured via: tools/oracle/icepac/run_icepac_dumps.sh (scratch dest, then copied;"
  echo "              tanbel_m07_s160_icepac and META.txt are a separate, earlier capture)"
} > "$FINAL/META_icepac_decks.txt"
echo "icepac: ${#DECK_LIST[@]} decks -> $FINAL"
