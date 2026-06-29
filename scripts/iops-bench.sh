#!/usr/bin/env bash
# Characterize the SSD's random/sequential I/O (the device, not the page cache),
# via the `iops` example (O_DIRECT on Linux / F_NOCACHE on macOS). Reports the
# read IOPS-vs-block-size and IOPS-vs-queue-depth curves, plus single- vs
# multi-stream write bandwidth — the numbers behind the insert-path tuning
# (e.g. ~5 KiB random reads hit the device's IOPS plateau; concurrent appends
# contend below a single sequential stream).
#
#   scripts/iops-bench.sh
#       env: GIB (test-file size, default 12)   OPS (ops per run, default 2,000,000)
#
# Writes a temp file next to $TMPDIR — point that at a real SSD (O_DIRECT is
# unsupported on tmpfs, and tmpfs is RAM anyway). The file is filled then removed.
set -euo pipefail
cd "$(dirname "$0")/.."

GIB="${GIB:-12}"
OPS="${OPS:-2000000}"
IOPS="./target/release/examples/iops"

cargo build --release --example iops

echo "### random-READ vs block size (64 threads; device, uncached) ###"
for B in 4096 8192 16384 49152; do
  printf -- '-- block=%-6s --  ' "$B"
  "$IOPS" 64 "$GIB" "$OPS" "$B" 2>&1 | awk '/rand read/{print}'
done

echo "### random-READ vs queue depth (8 KiB block) ###"
for T in 8 16 32 64 128; do
  printf -- '-- threads=%-4s --  ' "$T"
  "$IOPS" "$T" "$GIB" "$OPS" 8192 2>&1 | awk '/rand read/{print}'
done

echo "### WRITE: single-stream (seq) vs multi-thread (rand), by block size ###"
for B in 131072 1048576; do
  echo "-- block=$B, 64 threads --"
  "$IOPS" 64 "$GIB" 1000000 "$B" 2>&1 | grep -E "seq write|rand write"
done
