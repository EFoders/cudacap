#!/bin/bash
# run_sweep.sh -- capture the accuracy-sweep decks through the LUFFY shim.
#
# Runs in WSL.  Builds one shim per shard by calling the stock
# tools/oracle/dumps/run_luffy_dumps.sh once (on that shard's first deck), each
# shard with its OWN itshfbc copy and work dir, then drives that shim over the
# shard's decks itself.  It does not reuse run_luffy_dumps.sh's loop because
# that loop aborts the whole run on the first deck that crashes or captures
# nothing; here such a deck is logged and skipped.
#
#   wsl bash tools/oracle/sweep/run_sweep.sh
#
# Environment (defaults in brackets):
#   SWEEP_DECKS      deck dir from gen_sweep.py        [~/sweep/decks]
#   SWEEP_DEST       output dir (NOT the repo)         [~/sweep/dumps]
#   SWEEP_JOBS       parallel shards                   [8]
#   SWEEP_REFERENCE  vendored reference tree           [$REPO/reference]
#   SWEEP_STOCK_N    decks to time on the STOCK voacapl, serially [all]
#   SWEEP_TIMEOUT    per-deck timeout, s               [600]
#   SWEEP_LIMIT      only the first N decks (smoke runs) [all]
#
# Writes $SWEEP_DEST/<deck>_{in,out,ant}.csv, RUNLOG.tsv (deck, status,
# seconds, LUFFY rows; status ok / crash:<rc> / nocapture), STOCK_TIMING.tsv
# and META.txt (gfortran version and flags, from the shim build).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DECKS="${SWEEP_DECKS:-$HOME/sweep/decks}"
DEST="${SWEEP_DEST:-$HOME/sweep/dumps}"
JOBS="${SWEEP_JOBS:-8}"
REFERENCE="${SWEEP_REFERENCE:-$REPO_ROOT/reference}"
STOCK_N="${SWEEP_STOCK_N:-100000}"
TMO="${SWEEP_TIMEOUT:-600}"
SHARDS="$HOME/sweep/shards"
STOCK_BIN="$HOME/voacapl-oracle/prefix/bin/voacapl"

[[ -d "$DECKS" ]] || { echo "no deck dir $DECKS" >&2; exit 1; }
mapfile -t ALL < <(cd "$DECKS" && ls sw*.dat | sed 's/\.dat$//' | sort)
echo "sweep: ${#ALL[@]} decks, $JOBS shards -> $DEST"
[[ -n "${SWEEP_LIMIT:-}" ]] && ALL=("${ALL[@]:0:$SWEEP_LIMIT}")
mkdir -p "$DEST" "$SHARDS"
dt() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4f", b - a }'; }
rm -f "$DEST"/*.csv "$DEST"/RUNLOG.tsv

shard() {
    local j=$1; shift
    local itshfbc="$SHARDS/itshfbc_$j" work="$SHARDS/work_$j"
    local log="$SHARDS/log_$j.tsv"
    rm -rf "$itshfbc"; cp -r "$HOME/itshfbc" "$itshfbc"
    : > "$log"
    # build the shim (and get its META.txt) with the stock capture script
    if ! LUFFY_DECKS="$DECKS" LUFFY_REFERENCE="$REFERENCE" \
         LUFFY_ITSHFBC="$itshfbc" LUFFY_WORK="$work" \
         LUFFY_DEST="$SHARDS/warm_$j" \
         bash "$REPO_ROOT/tools/oracle/dumps/run_luffy_dumps.sh" "$1" \
         > "$SHARDS/build_$j.txt" 2>&1; then
        # the warm-up deck itself may be the one that fails: keep the shim
        [[ -x "$work/voacapl_shim" ]] || { echo "shard $j: shim build failed" >&2; return 1; }
    fi
    cd "$work" || return 1
    local name t0 t1 rc rows st
    for name in "$@"; do
        rm -f luffy_in.csv luffy_out.csv luffy_ant.csv
        cp "$DECKS/$name.dat" "$itshfbc/run/voacapx.dat"
        t0=$(date +%s.%N)
        timeout "$TMO" ./voacapl_shim "$itshfbc" voacapx.dat voacapx.out \
            >/dev/null 2>"$work/stderr.txt"
        rc=$?
        t1=$(date +%s.%N)
        rows=0
        if [[ $rc -ne 0 ]]; then
            st="crash:$rc"
        elif [[ ! -s luffy_in.csv || ! -s luffy_out.csv || ! -s luffy_ant.csv ]]; then
            st="nocapture"
        else
            st="ok"
            rows=$(($(wc -l <luffy_out.csv) - 1))
            cp luffy_in.csv "$DEST/${name}_in.csv"
            cp luffy_out.csv "$DEST/${name}_out.csv"
            cp luffy_ant.csv "$DEST/${name}_ant.csv"
        fi
        printf '%s\t%s\t%.4f\t%d\n' "$name" "$st" "$(dt "$t0" "$t1")" "$rows" >> "$log"
    done
}

T0=$(date +%s.%N)
for ((j = 0; j < JOBS; j++)); do
    mine=()
    for ((k = j; k < ${#ALL[@]}; k += JOBS)); do mine+=("${ALL[k]}"); done
    shard "$j" "${mine[@]}" &
done
wait
T1=$(date +%s.%N)

{ printf 'deck\tstatus\tseconds\trows\n'; cat "$SHARDS"/log_*.tsv | sort; } > "$DEST/RUNLOG.tsv"
nok=$(grep -c $'\tok\t' "$DEST/RUNLOG.tsv")
echo "captured $nok / ${#ALL[@]} decks in $(dt "$T0" "$T1") s wall"
grep -v $'\tok\t' "$DEST/RUNLOG.tsv" | tail -n +2 | sed 's/^/  NOT OK: /'

# The stock (unshimmed) voacapl, serially, on the first STOCK_N decks (default all): the
# clean per-deck oracle time (the shim adds CSV dumping, and the shards
# compete for cores).
st_itshfbc="$SHARDS/itshfbc_stock"
rm -rf "$st_itshfbc"; cp -r "$HOME/itshfbc" "$st_itshfbc"
printf 'deck\trc\tseconds\n' > "$DEST/STOCK_TIMING.tsv"
for name in "${ALL[@]:0:$STOCK_N}"; do
    cp "$DECKS/$name.dat" "$st_itshfbc/run/voacapx.dat"
    t0=$(date +%s.%N)
    (cd "$st_itshfbc/run" && timeout "$TMO" "$STOCK_BIN" "$st_itshfbc" voacapx.dat voacapx.out >/dev/null 2>&1)
    rc=$?
    t1=$(date +%s.%N)
    printf '%s\t%d\t%.4f\n' "$name" "$rc" "$(dt "$t0" "$t1")" >> "$DEST/STOCK_TIMING.tsv"
done

{ head -2 "$SHARDS/warm_0/META.txt"
  echo "oracle: $STOCK_BIN (stock); LUFFY shim built per shard by"
  echo "        tools/oracle/dumps/run_luffy_dumps.sh from $REFERENCE"
  echo "decks: $DECKS ($(head -1 "$DECKS/MANIFEST.tsv"))"
  echo "captured: $nok / ${#ALL[@]} decks ($JOBS shards, $(dt "$T0" "$T1") s wall)"
  echo "files: <deck>_in.csv / _out.csv / _ant.csv as run_luffy_dumps.sh;"
  echo "       RUNLOG.tsv per-deck status and shim wall time;"
  echo "       STOCK_TIMING.tsv stock voacapl wall time, serial"
  uname -srm
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
cat "$DEST/META.txt"
