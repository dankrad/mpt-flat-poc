#!/usr/bin/env bash
# Benchmark batch inserts into an existing tree, reporting us/key.
#
#   scripts/batch-bench.sh <checkpoint.flat> [n_keys] [batch_size]
#       defaults: n_keys=10,000,000   batch_size=10,000
#
# Runs against a throwaway COW clone of the checkpoint (APFS clone / reflink, so
# it's instant and space-free), leaving your real tree untouched. Inserts N fresh
# keys in `batch_size` batches and prints per-batch + overall us/key.
#
# Uses the full fast path by default (one-writer + no-WAL + async values + 64
# read workers, GC off — safe since the clone is discarded). Override via env,
# e.g. compare paths:  MPT_ONE_WRITER=0 scripts/batch-bench.sh tree.flat
set -euo pipefail
cd "$(dirname "$0")/.."

CK="${1:?usage: batch-bench.sh <checkpoint.flat> [n_keys] [batch_size]}"
N="${2:-10000000}"
BATCH="${3:-10000}"
START="${START:-9000000000}"   # fresh keys, beyond a typical build's range
NAME="$(basename "$CK")"
CLONE="${TMPDIR:-/tmp}/batch-bench.$$"
trap 'rm -rf "$CLONE"' EXIT

# Copy-on-write clone where supported (macOS APFS `cp -c`, Linux reflink), else a
# plain copy. Files: <name>, <name>.meta, <name>.values/.
clone() { cp -c "$1" "$2" 2>/dev/null || cp --reflink=auto "$1" "$2" 2>/dev/null || cp "$1" "$2"; }
clonedir() { cp -cR "$1" "$2" 2>/dev/null || cp -R --reflink=auto "$1" "$2" 2>/dev/null || cp -R "$1" "$2"; }
mkdir -p "$CLONE"
clone    "$CK"         "$CLONE/$NAME"
clone    "$CK.meta"    "$CLONE/$NAME.meta"
clonedir "$CK.values"  "$CLONE/$NAME.values"

: "${MPT_WORKERS:=64}"
: "${MPT_ONE_WRITER:=1}"
: "${MPT_NO_WAL:=1}"
: "${MPT_ASYNC_VALUES:=1}"
: "${MPT_GC_DISABLE:=1}"

cargo build --release --example foldbench
echo "batch-bench: $N keys, batch=$BATCH, into a clone of $CK"
env MPT_WORKERS="$MPT_WORKERS" MPT_ONE_WRITER="$MPT_ONE_WRITER" MPT_NO_WAL="$MPT_NO_WAL" \
    MPT_ASYNC_VALUES="$MPT_ASYNC_VALUES" MPT_GC_DISABLE="$MPT_GC_DISABLE" \
    ./target/release/examples/foldbench "$CLONE/$NAME" "$N" "$START" "$BATCH"
