#!/usr/bin/env bash
# Benchmark sustained 10k-key-batch inserts into a baseline checkpoint using the
# full fused fast path:
#   one-writer + opportunistic GC (evacuation fused into the foreground read,
#   relocations written by the parallel writer) + no-WAL + async values, 64 workers,
#   GC evac threshold 0.30.
#
# Runs against a throwaway copy-on-write clone of the baseline (the baseline is left
# untouched). Warms 25M keys first to drive GC to its steady state, then measures
# 10M keys in 10k batches and prints the us/key, the one-writer device read/write
# split, and the gc-evac breakdown.
#
#   Usage: scripts/bench-fused.sh <baseline-dir>
#
# <baseline-dir> is the directory holding ckpt.flat (e.g. the output of
# build-baseline-1b.sh). The clone is made next to it (same volume, so the COW
# clone is instant and space-free on APFS/reflink filesystems) and removed on exit.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:?usage: bench-fused.sh <baseline-dir>}"
CK="$SRC/ckpt.flat"
[ -f "$CK" ] || { echo "no ckpt.flat in '$SRC'"; exit 1; }

# Clone next to the source so cp -c / --reflink stays on the same volume.
CLONE="$(cd "$SRC/.." && pwd)/.bench-fused.$$"
trap 'rm -rf "$CLONE"' EXIT
mkdir -p "$CLONE"
clone()    { cp -c  "$1" "$2" 2>/dev/null || cp --reflink=auto "$1" "$2" 2>/dev/null || cp "$1" "$2"; }
clonedir() { cp -cR "$1" "$2" 2>/dev/null || cp -R --reflink=auto "$1" "$2" 2>/dev/null || cp -R "$1" "$2"; }
clone    "$CK"        "$CLONE/ckpt.flat"
clone    "$CK.meta"   "$CLONE/ckpt.flat.meta"
clonedir "$CK.values" "$CLONE/ckpt.flat.values"

cargo build --release --example profins

echo "bench-fused: 25M warmup + 10M measured (10k batches) on a clone of $CK"
env MPT_WORKERS=64 MPT_ONE_WRITER=1 MPT_GC_OPP=1 MPT_GC_OPP_UTIL=0.30 \
    MPT_NO_WAL=1 MPT_ASYNC_VALUES=1 MPT_PROF_WARMUP=25000000 \
  ./target/release/examples/profins "$CLONE/ckpt.flat" 10000000 8600000000 10000
