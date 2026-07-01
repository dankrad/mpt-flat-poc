#!/usr/bin/env bash
# Build a 1,000,000,000-key baseline checkpoint with the fast method:
#   - RAM-build at the start: fresh leaves live in RAM (no flat I/O / no GC), the
#     top branch is bootstrapped so the first batch parallelizes, and values go
#     into a vector-memtable RocksDB tuned for bulk load.
#   - 100M-key insert batches.
#   - Spill the in-RAM leaves to disk once the process footprint crosses 30 GiB,
#     then finish on the disk path (bounded GC).
#   - 16 KiB leaves, 64 workers, values written with the WAL disabled.
#
# This is the configuration that built the 1B baseline in ~1420 s (~1.4 us/key),
# root 4ebe8c88…5ddc.
#
#   Usage: scripts/build-baseline-1b.sh <output-dir>
#
# Produces <output-dir>/ckpt.flat, ckpt.flat.meta, ckpt.flat.values/. Reopen/verify
# with:  cargo run --release --example reopen -- <output-dir>/ckpt.flat
#
# Note: MPT_RAM_BUILD_GIB=30 is the spill ceiling — it must trip before RAM gets
# tight; raise it on a larger-RAM box to stay in the faster RAM-build phase longer.
# The build needs ~150 GiB of free disk at peak and settles to ~85 GiB.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:?usage: build-baseline-1b.sh <output-dir>}"
mkdir -p "$OUT"

cargo build --release --example buildpersist

MPT_RAM_BUILD=1 MPT_RAM_BUILD_GIB=30 MPT_BUILD_BATCH=100000000 \
MAX_LEAF_KIB=16 MPT_WORKERS=64 MPT_NO_WAL=1 \
  ./target/release/examples/buildpersist 1000000000 "$OUT/ckpt.flat"
