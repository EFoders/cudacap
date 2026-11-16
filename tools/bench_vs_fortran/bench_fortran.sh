#!/bin/bash
# Fortran voacapl timing on the same workload as the GPU benchmark.
# One process per circuit (how voacapl runs a deck). Run from PowerShell:
#   wsl bash "/mnt/c/.../bench/bench_fortran.sh"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HOME/voacapl-oracle/prefix/bin/voacapl"
NW=${NW:-32}
B="$HOME/bench_itshfbc"; rm -rf "$B"; mkdir -p "$B"
for w in $(seq 0 $((NW - 1))); do cp -r "$HOME/itshfbc" "$B/w$w"; done
L="$HOME/bench_decks"; rm -rf "$L"; cp -r "$HERE/decks" "$L"

run_one() { cp "$2" "$1/run/bench.dat"; "$BIN" "$1" bench.dat bench.out >/dev/null 2>&1; }
now() { date +%s.%N; }
cpu=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')
echo "cpu: $cpu ($(nproc) logical), workers: $NW"
echo "gfortran: $(gfortran --version | head -1)"

for rep in 1 2 3; do
  t0=$(now); for f in "$L/single"/00[0-5]*.dat "$L/single"/006[0-3].dat; do run_one "$B/w0" "$f"; done; t1=$(now)
  echo "serial_24h_x64 rep$rep: $(awk -v a="$t0" -v b="$t1" "BEGIN{printf \"%.3f\", b-a}") s"
  t0=$(now); for f in "$L/onehour"/*.dat; do run_one "$B/w0" "$f"; done; t1=$(now)
  echo "serial_1h_x64 rep$rep: $(awk -v a="$t0" -v b="$t1" "BEGIN{printf \"%.3f\", b-a}") s"
done
# sanity: last 24 h output has 24 hourly FREQ rows
run_one "$B/w0" "$L/single/0000.dat"; echo "freq_rows_in_a_24h_deck_output: $(grep -c " FREQ$" "$B/w0/run/bench.out" || true)"

for rep in 1 2 3; do
  t0=$(now)
  for w in $(seq 0 $((NW - 1))); do
    ( i=0; for f in "$L/single"/*.dat; do
        if (( i % NW == w )); then run_one "$B/w$w" "$f"; fi; i=$((i + 1)); done ) &
  done; wait
  t1=$(now)
  echo "parallel_24h_x640 rep$rep: $(awk -v a="$t0" -v b="$t1" "BEGIN{printf \"%.3f\", b-a}") s"
done
