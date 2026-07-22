#!/usr/bin/env bash
# Run every golden-matrix deck through the WSL oracle and archive outputs.
#
# For each golden/p2p/matrix/decks/*.dat: run voacapl twice (determinism
# gate), then archive the byte-exact output under
# golden/p2p/matrix/outputs/<name>.out.  Records toolchain + hashes in
# golden/p2p/matrix/META.txt.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/golden_decks/run_matrix.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
DECKS="$REPO/golden/p2p/matrix/decks"
OUTS="$REPO/golden/p2p/matrix/outputs"
BIN="$HOME/voacapl-oracle/prefix/bin/voacapl"
RUN="$HOME/itshfbc/run"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

[[ -x "$BIN" ]] || { echo "oracle not built — run tools/oracle/build_oracle.sh" >&2; exit 1; }
compgen -G "$DECKS/*.dat" >/dev/null || { echo "no decks — run gen_matrix.py" >&2; exit 1; }

mkdir -p "$OUTS"
n=0
for deck in "$DECKS"/*.dat; do
    name="$(basename "$deck" .dat)"
    cp "$deck" "$RUN/voacapx.dat"
    "$BIN" "$HOME/itshfbc" voacapx.dat voacapx.out >/dev/null
    cp "$RUN/voacapx.out" "$WORK/run1.out"
    "$BIN" "$HOME/itshfbc" voacapx.dat voacapx.out >/dev/null
    if ! cmp -s "$WORK/run1.out" "$RUN/voacapx.out"; then
        echo "FAIL: non-deterministic output for $name" >&2
        exit 1
    fi
    cp "$RUN/voacapx.out" "$OUTS/$name.out"
    n=$((n + 1))
done

{
    echo "Golden-deck matrix outputs — voacapl oracle, byte-exact bar."
    echo "Decks: golden/p2p/matrix/decks (gen_matrix.py); each deck ran twice"
    echo "and both runs were byte-identical before archiving."
    echo
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cat "$HOME/voacapl-oracle/BUILD_INFO.txt"
    echo
    (cd "$REPO/golden/p2p/matrix" && sha256sum decks/*.dat outputs/*.out)
} > "$REPO/golden/p2p/matrix/META.txt"

echo "archived $n deck outputs into golden/p2p/matrix/outputs/"
