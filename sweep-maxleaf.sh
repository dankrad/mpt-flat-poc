#!/usr/bin/env bash
# Sweep max_leaf to measure the deep-promotion tax vs write-amp / frontier RAM.
# Each run preloads $KEYS uniform keys at one max_leaf size and prints the
# milestone + final summary (which now include promotion events/children/sizes,
# the promotion share of write bytes, and write-amp B/key).
set -euo pipefail
cd "$(dirname "$0")"

KEYS="${KEYS:-40000000}"
BATCH="${BATCH:-10000}"
SIZES="${SIZES:-1 2 4 8 16 32}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="sweep-maxleaf-${STAMP}.log"
export TMPDIR="${TMPDIR:-/tmp}/mpt-sweep-${STAMP}"
mkdir -p "$TMPDIR"

cargo build --release --bench large >/dev/null 2>&1

echo "sweep: keys=$KEYS batch=$BATCH sizes=[$SIZES] tmpdir=$TMPDIR" | tee "$LOG"
for kib in $SIZES; do
  echo -e "\n\n############## max_leaf = ${kib} KiB ##############" | tee -a "$LOG"
  LARGE_PRELOAD="$KEYS" LARGE_BATCH="$BATCH" LARGE_MAX_LEAF_KIB="$kib" \
    cargo bench --bench large 2>&1 | grep -v -E '^\s*(Compiling|Finished|Running|warning:)' | tee -a "$LOG"
  rm -f "$TMPDIR"/*.flat "$TMPDIR"/*.values "$TMPDIR"/*.meta 2>/dev/null || true
done
echo -e "\nsweep complete -> $LOG"
rm -rf "$TMPDIR"
