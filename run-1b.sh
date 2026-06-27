#!/usr/bin/env bash
# Launch the instrumented 1B-key run (branch dense-pack-gc) with one command.
#
#   ./run-1b.sh /path/with/250GB [num_keys]
#
# Builds the bench, runs the preload at the standard config, tees a timestamped
# log into the data dir, and persists a reopenable checkpoint there. SIGINT/
# SIGTERM are forwarded so a kill still persists before exit (kill -9 does not).
set -euo pipefail

DATA_DIR="${1:?usage: ./run-1b.sh <data-dir-with-~250GB> [num_keys]}"
KEYS="${2:-1000000000}"            # default 1B; pass e.g. 600000000 for a shorter run
BATCH=10000
MAX_LEAF_KIB=16

cd "$(dirname "$0")"
mkdir -p "$DATA_DIR"

# Warn (don't block) if the volume looks short on space.
AVAIL_GB="$(df -Pk "$DATA_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
NEED_GB=$(( KEYS / 5000000 ))      # ~200GB at 1B, scales with key count
echo "data dir: $DATA_DIR   free: ${AVAIL_GB} GiB   (need ~${NEED_GB} GiB for ${KEYS} keys)"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  echo "WARNING: less free space than the estimate — the run may fill the disk." >&2
fi

echo "building release bench (first build compiles RocksDB; takes a few minutes)..."
cargo build --release --bench large

LOG="$DATA_DIR/run-${KEYS}-$(date +%Y%m%d-%H%M%S).log"
echo "config: ${KEYS} keys, batch ${BATCH}, max_leaf ${MAX_LEAF_KIB} KiB, persist on"
echo "logging to: $LOG"
echo

TMPDIR="$DATA_DIR" \
LARGE_PRELOAD="$KEYS" \
LARGE_BATCH="$BATCH" \
LARGE_MAX_LEAF_KIB="$MAX_LEAF_KIB" \
LARGE_PERSIST=1 \
  cargo bench --bench large 2>&1 | tee "$LOG"

echo
echo "done. checkpoint: $DATA_DIR/mpt-checkpoint.flat (+ .meta + .values)"
echo "drill down without rebuilding:"
echo "  cargo run --release --example reopen -- $DATA_DIR/mpt-checkpoint.flat 300"
