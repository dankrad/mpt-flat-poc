#!/usr/bin/env bash
# Fast-build a trie of any size to a reopenable checkpoint.
#
#   scripts/build-tree.sh <checkpoint.flat> [n_keys]      # default 1,000,000,000
#
# Uses RAM-build mode: builds entirely in RAM at ~1.8 us/key until the process
# memory footprint crosses a spill threshold, then spills to the flat file and
# finishes the remainder in disk mode. So the more RAM the box has, the more of
# the build stays fast (a box with > ~footprint RAM never spills). The
# `buildpersist` binary links mimalloc (the system allocator uses ~2.3x the RAM).
#
# The knob that matters most is the spill threshold, defaulted to ~65% of total
# RAM (leaving headroom for RocksDB + page cache + the spill transient). GC stays
# ON for the disk-mode tail so leaf rewrites don't balloon the file.
#
# Override anything via the env, e.g.:
#   MPT_RAM_BUILD_GIB=160 MAX_LEAF_KIB=16 MPT_WORKERS=96 \
#     scripts/build-tree.sh /data/tree.flat 2000000000
#
# Output: <checkpoint.flat> + <...>.meta (manifest) + <...>.values/ (RocksDB).
# Reopen with FlatMpt::open, or grow/benchmark it with the sibling scripts.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:?usage: build-tree.sh <checkpoint.flat> [n_keys]}"
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
: "${MPT_NO_WAL:=1}"                                   # values durable via persist's flush

echo "build: N=$N  out=$OUT"
echo "  RAM=${RAM_GIB}GiB  spill@${MPT_RAM_BUILD_GIB}GiB  max_leaf=${MAX_LEAF_KIB}KiB  workers=${MPT_WORKERS}"

cargo build --release --example buildpersist

exec env \
  MPT_RAM_BUILD="$MPT_RAM_BUILD" \
  MPT_RAM_BUILD_GIB="$MPT_RAM_BUILD_GIB" \
  MAX_LEAF_KIB="$MAX_LEAF_KIB" \
  MPT_WORKERS="$MPT_WORKERS" \
  MPT_NO_WAL="$MPT_NO_WAL" \
  ./target/release/examples/buildpersist "$N" "$OUT"
