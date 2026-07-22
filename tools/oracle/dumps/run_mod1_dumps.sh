#!/bin/bash
# run_mod1_dumps.sh — generate the Module-1 tier-B oracle dumps in WSL:
# MAGVAR(+MAGFIN), DISTXY(+DAZEL), SANG, GEOM(+MAGVAR/MAGFIN, NOISY stub).
#
# Same pattern as run_dazel_dump.sh: compile each dump driver against the
# vendored routines with the oracle flags, run over the deterministic case
# grids, copy fixtures into golden/dumps/<name>/.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_mod1_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/mod1"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_mod1_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch"

cd "$WORK"
for name in magvar distxy sang geom; do
    case $name in
        magvar) srcs=("$VSRC/magvar.for" "$VSRC/magfin.for");;
        distxy) srcs=("$VSRC/distxy.for" "$VSRC/dazel.for");;
        sang)   srcs=("$VSRC/sang.for");;
        geom)   srcs=("$VSRC/geom.for" "$VSRC/magvar.for" "$VSRC/magfin.for");;
    esac
    # shellcheck disable=SC2086
    gfortran $FLAGS -o "dump_$name" "$SCRIPT_DIR/dump_$name.f" "${srcs[@]}"
    "./dump_$name"
    DEST="$REPO_ROOT/golden/dumps/$name"
    mkdir -p "$DEST"
    cp "${name}_cases.csv" "$DEST/cases.csv"
    cp "${name}_dump.csv" "$DEST/expected.csv"
    { gfortran --version | head -1
      echo "flags: $FLAGS"
      echo "sources: dump_$name.f ${srcs[*]#"$REPO_ROOT/"}"
      date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
    } > "$DEST/META.txt"
    echo "$name: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/$name/"
done
