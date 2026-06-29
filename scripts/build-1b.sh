#!/usr/bin/env bash
# Optimized 1B-key build -> checkpoint, with good defaults.
#
#   scripts/build-1b.sh <checkpoint.flat> [n_keys]
#
# RAM-build mode: builds in RAM at ~1.8 us/key until the memory footprint crosses
# the spill threshold, then spills to disk and finishes the tail in disk mode. The
# `buildpersist` binary has mimalloc compiled in (the system allocator uses ~2.3x
# the RAM). GC is left ON so the disk tail's leaf rewrites don't balloon the file.
#
# The one knob that matters most is the spill threshold: higher = more of the build
# stays in RAM (faster). We default it to ~65% of total RAM (leaving headroom for
# RocksDB + page cache + the spill transient). Override any setting via the env,
# e.g.  MPT_RAM_BUILD_GIB=160 MPT_WORKERS=96 scripts/build-1b.sh out.flat
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:?usage: build-1b.sh <checkpoint.flat> [n_keys]}"
N="${2:-1000000000}"

# Total RAM in GiB (Linux /proc/meminfo, else macOS sysctl).
if [ -r /proc/meminfo ]; then
  RAM_GIB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
else
  RAM_GIB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
fi

: "${MPT_RAM_BUILD:=1}"
: "${MPT_RAM_BUILD_GIB:=$(( RAM_GIB * 65 / 100 ))}"   # ~65% of RAM
: "${MAX_LEAF_KIB:=16}"
: "${MPT_WORKERS:=64}"                                 # read queue depth for the disk tail

echo "build: N=$N  out=$OUT"
echo "  RAM=${RAM_GIB}GiB  spill@${MPT_RAM_BUILD_GIB}GiB  max_leaf=${MAX_LEAF_KIB}KiB  workers=${MPT_WORKERS}"

cargo build --release --example buildpersist

exec env \
  MPT_RAM_BUILD="$MPT_RAM_BUILD" \
  MPT_RAM_BUILD_GIB="$MPT_RAM_BUILD_GIB" \
  MAX_LEAF_KIB="$MAX_LEAF_KIB" \
  MPT_WORKERS="$MPT_WORKERS" \
  ./target/release/examples/buildpersist "$N" "$OUT"
