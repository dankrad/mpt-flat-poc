#!/usr/bin/env bash
# A/B the flat-file read path on the persisted ~177M base: buffered vs direct
# I/O at QD8 and QD32. Each run gets a fresh APFS clone (cp -c, instant, COW) so
# the inserts never touch the pristine base. NOTE: at ~37 GB this base is
# cache-resident, so direct (which bypasses the cache) is EXPECTED to lose here —
# this measures the cache-resident penalty + direct's QD scaling, not the
# disk-bound win (which needs file > RAM).
set -uo pipefail
cd "$(dirname "$0")"
BASE=$(cat /tmp/mpt-1b-base-datadir.txt)
CKPT="$BASE/mpt-checkpoint.flat"
OUT="/tmp/direct-ab-$(date +%H%M%S).log"; echo "$OUT" > /tmp/direct-ab-logpath.txt

cargo build --release --example reopen >/dev/null 2>&1

runone() {
  local label="$1" direct="$2" workers="$3"
  local C="/tmp/ab-clone-$$"
  rm -rf "$C"; mkdir -p "$C"
  cp -c  "$CKPT"        "$C/mpt-checkpoint.flat"
  cp -c  "$CKPT.meta"   "$C/mpt-checkpoint.flat.meta"
  cp -cR "$CKPT.values" "$C/mpt-checkpoint.flat.values"
  echo "########## $label  (MPT_DIRECT_IO=$direct MPT_WORKERS=$workers) ##########"
  MPT_DIRECT_IO="$direct" MPT_WORKERS="$workers" \
    ./target/release/examples/reopen "$C/mpt-checkpoint.flat" 100
  echo
  rm -rf "$C"
}

{
  runone "buffered QD8"  0 8
  runone "buffered QD32" 0 32
  runone "direct   QD8"  1 8
  runone "direct   QD32" 1 32
} > "$OUT" 2>&1
echo "done -> $OUT"
