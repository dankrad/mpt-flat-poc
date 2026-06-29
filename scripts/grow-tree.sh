#!/usr/bin/env bash
# Grow an existing tree by inserting more keys, then checkpoint it in place.
#
#   scripts/grow-tree.sh <checkpoint.flat> <n_more> [start_key] [batch_size]
#       defaults: start_key=2,000,000,000   batch_size=10,000
#
# Inserts `n_more` fresh keys (numbered from start_key) into the checkpoint and
# persists the result. start_key must be at/above the tree's existing key count
# to ADD keys — overlapping numbers overwrite. This MODIFIES the checkpoint.
#
# Default path keeps GC ON so the file stays bounded as leaf rewrites create
# garbage (slower, ~device-bound). For a faster grow that trades file size for
# speed, opt into the bulk fast path (skips GC -> file grows):
#   MPT_ONE_WRITER=1 MPT_ASYNC_VALUES=1 MPT_GC_DISABLE=1 \
#     scripts/grow-tree.sh tree.flat 100000000
set -euo pipefail
cd "$(dirname "$0")/.."

CK="${1:?usage: grow-tree.sh <checkpoint.flat> <n_more> [start_key] [batch_size]}"
N="${2:?n_more required}"
START="${3:-2000000000}"
BATCH="${4:-10000}"

: "${MPT_WORKERS:=64}"
: "${MPT_NO_WAL:=1}"

cargo build --release --example foldbench
echo "grow: +$N keys (from $START, batch=$BATCH) into $CK, then persist"
exec env MPT_PERSIST=1 \
  MPT_WORKERS="$MPT_WORKERS" \
  MPT_NO_WAL="$MPT_NO_WAL" \
  ${MPT_ONE_WRITER:+MPT_ONE_WRITER="$MPT_ONE_WRITER"} \
  ${MPT_ASYNC_VALUES:+MPT_ASYNC_VALUES="$MPT_ASYNC_VALUES"} \
  ${MPT_GC_DISABLE:+MPT_GC_DISABLE="$MPT_GC_DISABLE"} \
  ./target/release/examples/foldbench "$CK" "$N" "$START" "$BATCH"
