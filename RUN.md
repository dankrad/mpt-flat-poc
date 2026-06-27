# Remote 1B run — launch + read guide

Branch: `dense-pack-gc` (batched writes + dense 256 B packing + inline self-tuning
GC). Verifies the thesis: dense packing shrinks the live working set toward
RAM-resident so the device-read half of Phase B caches away.

## Prereqs
- **Disk:** ~250 GB free under `$TMPDIR`. At 1B/16 KiB the flat file settles at
  roughly `live / util` ≈ 78 GiB / 0.5 ≈ ~160 GiB (+ high-water slack), plus the
  RocksDB value store (~33 GiB). Give it headroom.
- **RAM:** the thesis is that ~78 GiB live fits in RAM. More RAM = more of the
  working set caches.
- Rust toolchain; `cargo build --release --bench large` (RocksDB builds C++, takes
  a few minutes the first time).

## Launch
```sh
# Pick a data dir with space; the checkpoint + values land here.
export TMPDIR=/path/with/250GB
cargo build --release --bench large
LARGE_PRELOAD=1000000000 \
LARGE_BATCH=10000 \
LARGE_MAX_LEAF_KIB=16 \
LARGE_PERSIST=1 \
  cargo bench --bench large 2>&1 | tee /path/with/250GB/run1b.log
```
- `LARGE_PERSIST=1` writes a reopenable checkpoint (`mpt-checkpoint.flat` +
  `.meta` + `.values`) at the end.
- **Kill-safe:** SIGINT/SIGTERM flushes + persists before exit (so `kill` mid-run
  leaves a reopenable checkpoint). `kill -9` does not.
- Config logged at start; expect `max_leaf=16 KiB (target 8192, min_promote 8192)`.

## Reading the per-10M-key milestone block
```
[ 200M]   1234s 7.2µs/key | flat 96.0G live 75.1G util 58% | leaves 16777216 avg 4700B (60 k/leaf) | RAM 760M / 1.1M nodes
          phase ms/batch: A 14 | B 290 | GC 60 | C 6   (B 80%, GC 17%)
          B-work µs/key: pread 70 parse 1 rebuild 3 serialize 2 pwrite 40 lock 0
          GC: 0.45 reloc/key, 1500 regs/batch, 60 ms/batch, R=4096, free_regions 9000
```
What to watch:
- **live vs flat / util** — the headline. Does `live` stay ≪ RAM (cacheable)?
  Does `util` settle near `TARGET_UTIL` (0.60), or sit lower with `R` maxed
  (controller can't keep up — see GC)?
- **phase split (A/B/GC/C)** — GC is now its own phase (runs between B and C).
  If GC% is large, the build is GC-bound.
- **B-work µs/key** (thread-µs, summed over workers) — the device/CPU split:
  `pread`+`pwrite` = device; `parse`+`rebuild`+`serialize` = CPU; `lock` =
  alloc contention. Device-bound ⇒ pread/pwrite dominate; if the working set
  caches, pread collapses.
- **GC line** — `reloc/key` is the GC write-amplification (live data relocated
  per inserted key); `R` is the controller's cleaning rate (`R=8192` = maxed,
  i.e. it can't hold the target); `regs/batch` actually reclaimed; `free_regions`
  in the reuse pool.

## Tunables (src/lib.rs, near `REGION_PAGES`) if you want to sweep
- `TARGET_UTIL` (0.60) — file size vs GC pressure. Note: during a write-heavy
  build the controller often can't reach it (R maxes) — see `GC_DESIGN.md`.
- `EVAC_MAX_UTIL` (0.30) — only clean regions ≥70% garbage. Lower = cheaper
  cleaning, bigger file; the strongest knob on build GC cost.
- `GC_R_MAX` (8192) — per-batch cleaning cap (bounds the GC stall).
- `REGION_PAGES` (8 = 128 KiB) — reclaim + write-coalesce unit.

## After the run
- Final summary prints flat/live/util/garbage, leaves, leaf-page histogram,
  in-RAM index, and cumulative split/write/GC stats.
- Reopen for a device-bound drill-down without rebuilding:
  `cargo run --release --example reopen -- $TMPDIR/mpt-checkpoint.flat 300`
  (inserts fresh batches, prints the same Phase-B breakdown).
- Compare against the in-place baseline (`paged-nodes`): **30 µs/key, 863 MiB
  RAM, 256 GiB flat, 74.6 GiB live** at 1B.
