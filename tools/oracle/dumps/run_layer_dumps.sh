#!/bin/bash
# run_layer_dumps.sh — Module-2 tier-B dump for the layer-parameter chain
# TIMVAR -> F2VAR -> ESIND -> IONSET (with EF1VAR, VERSY, NOISY(8)).
#
# Links the real voacapw routines: same production FFLAGS and module files
# as run_ionomap_dumps.sh.  REDMAP reads $HOME/itshfbc/coeffs at runtime.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_layer_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/layers"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_layer_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

srcs=("$VSRC/redmap.for" "$VSRC/virtim.for" "$VSRC/versy.for" \
      "$VSRC/timvar.for" "$VSRC/ef1var.for" "$VSRC/f2var.for" \
      "$VSRC/esind.for" "$VSRC/ionset.for" "$VSRC/noisy.for")

cd "$WORK"
# module sources must precede their users
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_layers \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" \
    "$SCRIPT_DIR/dump_layers.f" "${srcs[@]}"
./dump_layers

DEST="$REPO_ROOT/golden/dumps/layers"
mkdir -p "$DEST"
cp layers_cases.csv "$DEST/cases.csv"
cp layers_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_layers.f + redmap/virtim/versy/timvar/ef1var/f2var/esind/ionset/noisy + modules"
  echo "coeffs: \$HOME/itshfbc/coeffs (w-twins; byte-identical to golden/coeffs)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "layers: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/layers/"
