#!/usr/bin/env bash
#
# Build and run the large-trie benchmark (benches/large.rs): preload N keys,
# then time 1000 new inserts and 1000 overwrites against the result.
#
# ── Prereqs (one-time) ────────────────────────────────────────────────────────
#   Rust toolchain:
#     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#     . "$HOME/.cargo/env"
#   A C/C++ toolchain + libclang (RocksDB builds from source):
#     Debian/Ubuntu:  sudo apt-get install -y build-essential clang libclang-dev
#     macOS:          xcode-select --install
#
# ── Important ─────────────────────────────────────────────────────────────────
#   The benchmark writes its data (flat file + RocksDB) to $TMPDIR. On Linux
#   /tmp is frequently tmpfs (RAM-backed) — for a large run you MUST point
#   TMPDIR at a real disk with enough free space:
#       export TMPDIR=/mnt/bigdisk/tmp
#   Footprint ≈ 410 B/key at 8 KiB leaves (~460 GB for 1B); ~190 GB at 32 KiB.
#   In-RAM index is ~1.5 GB at 480M keys (grows sub-linearly).
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   ./scripts/run-large-bench.sh                       # defaults below
#   PRELOAD=1000000000 BATCH=10000 ./scripts/run-large-bench.sh
#   MAX_LEAF_KIB=32 PRELOAD=1000000000 ./scripts/run-large-bench.sh
#   PROFILE=1 ./scripts/run-large-bench.sh             # + per-category time breakdown
#
#   Long run, backgrounded with a live log:
#     PRELOAD=1000000000 BATCH=10000 nohup ./scripts/run-large-bench.sh \
#         > ~/mpt-1b.log 2>&1 &
#     tail -f ~/mpt-1b.log
#
# ── Knobs (env vars) ──────────────────────────────────────────────────────────
#   PRELOAD       keys to build                       (default 1000000)
#   BATCH         batch size; 0 = one-by-one inserts  (default 10000)
#   MAX_LEAF_KIB  max disk-leaf size in KiB           (default 8)
#   PROFILE       set to 1 for the time-attribution breakdown
#
set -euo pipefail

# Run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

PRELOAD="${PRELOAD:-1000000}"
BATCH="${BATCH:-10000}"
MAX_LEAF_KIB="${MAX_LEAF_KIB:-8}"

# Make cargo available if it was installed via rustup in this shell's profile.
if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

tmp="${TMPDIR:-/tmp}"
echo "TMPDIR = $tmp  (DB is written here — make sure it has enough free disk)"
df -h "$tmp" 2>/dev/null || true
echo "config: PRELOAD=$PRELOAD  BATCH=$BATCH  MAX_LEAF_KIB=$MAX_LEAF_KIB"
echo

features=()
if [ "${PROFILE:-0}" = "1" ]; then
    features=(--features profiling)
fi

LARGE_PRELOAD="$PRELOAD" \
LARGE_BATCH="$BATCH" \
LARGE_MAX_LEAF_KIB="$MAX_LEAF_KIB" \
    cargo bench --bench large "${features[@]}"
