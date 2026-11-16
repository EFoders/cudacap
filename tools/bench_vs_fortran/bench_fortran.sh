#!/bin/bash
# Fortran voacapl timing on the same workload as the GPU benchmark.
# One process per circuit (how voacapl runs a deck). Run from PowerShell:
#   wsl bash "/mnt/c/.../tools/bench_vs_fortran/bench_fortran.sh"
# Output is also written to results_<date>_fortran.txt next to this script
# (OUT=... to override; an existing file is never overwritten). Only awk is
# used for arithmetic: bc is not installed in the WSL image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/results_$(date +%F)_fortran.txt}"
if [ -e "$OUT" ]; then echo "$OUT exists; set OUT=... to write elsewhere" >&2; exit 1; fi
NF=9   # frequencies per circuit-hour (gen_bench_decks.py FREQ card)
BIN="$HOME/voacapl-oracle/prefix/bin/voacapl"
NW=${NW:-32}

run_one() { cp "$2" "$1/run/bench.dat"; "$BIN" "$1" bench.dat bench.out >/dev/null 2>&1; }
now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", b-a}'; }

bench() {
  B="$HOME/bench_itshfbc"; rm -rf "$B"; mkdir -p "$B"
  for w in $(seq 0 $((NW - 1))); do cp -r "$HOME/itshfbc" "$B/w$w"; done
  L="$HOME/bench_decks"; rm -rf "$L"; cp -r "$HERE/decks" "$L"

  cpu=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')
  echo "cpu: $cpu ($(nproc) logical), workers: $NW"
  echo "gfortran: $(gfortran --version | head -1)"

  for rep in 1 2 3; do
    t0=$(now); for f in "$L/single"/00[0-5]*.dat "$L/single"/006[0-3].dat; do run_one "$B/w0" "$f"; done; t1=$(now)
    echo "serial_24h_x64 rep$rep: $(elapsed "$t0" "$t1") s"
    t0=$(now); for f in "$L/onehour"/*.dat; do run_one "$B/w0" "$f"; done; t1=$(now)
    echo "serial_1h_x64 rep$rep: $(elapsed "$t0" "$t1") s"
  done
  # sanity: last 24 h output has 24 hourly FREQ rows
  run_one "$B/w0" "$L/single/0000.dat"; echo "freq_rows_in_a_24h_deck_output: $(grep -c " FREQ$" "$B/w0/run/bench.out" || true)"

  par=""
  for rep in 1 2 3; do
    t0=$(now)
    for w in $(seq 0 $((NW - 1))); do
      ( i=0; for f in "$L/single"/*.dat; do
          if (( i % NW == w )); then run_one "$B/w$w" "$f"; fi; i=$((i + 1)); done ) &
    done; wait
    t1=$(now)
    t=$(elapsed "$t0" "$t1"); par="$par $t"
    echo "parallel_24h_x640 rep$rep: $t s"
  done

  # best parallel rep -> circuit-hours/s and us per (circuit, hour, freq)
  echo "$par" | awk -v nf="$NF" '{
    for (i = 1; i <= NF; i++) { t = $i + 0; if (best == "" || t < best) best = t }
    ch = 640 * 24
    printf "summary_parallel_best: %.3f s, %.0f circuit-hours/s, %.3f us per (circuit, hour, freq)\n",
           best, ch / best, 1e6 * best / (ch * nf) }'
}

bench 2>&1 | tee "$OUT"
