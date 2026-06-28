#!/usr/bin/env bash
# Controlled A/B for the streaming-keccak change: same 40M/16KiB/8-worker
# cache-resident config, old (vec! per hash) vs new (streamed into the sponge).
set -uo pipefail
cd "$(dirname "$0")"
OUT="/tmp/parse-ab-$(date +%H%M%S).log"; echo "$OUT" > /tmp/parse-ab-logpath.txt
export TMPDIR="/tmp/mpt-parse-ab-$$"; mkdir -p "$TMPDIR"

STASHED=0
cleanup(){ [ "$STASHED" = 1 ] && git stash pop >/dev/null 2>&1 || true; }
trap cleanup EXIT

run(){ LARGE_PRELOAD=40000000 LARGE_BATCH=10000 LARGE_MAX_LEAF_KIB=16 MPT_WORKERS=8 \
  cargo bench --bench large 2>&1 | grep -vE '^[[:space:]]*(Compiling|Finished|Running)'
  rm -f "$TMPDIR"/*.flat "$TMPDIR"/*.values "$TMPDIR"/*.meta 2>/dev/null || true; }

{
  echo "##### OLD (committed d0e4663) #####"
  if git stash push -m parse-ab -- benches/large.rs examples/reopen.rs src/lib.rs; then STASHED=1; fi
  cargo build --release --bench large 2>&1 | tail -1
  run
  echo "##### NEW (presize serialize + lazy leaves) #####"
  git stash pop && STASHED=0
  cargo build --release --bench large 2>&1 | tail -1
  run
} > "$OUT" 2>&1
echo "done -> $OUT"
