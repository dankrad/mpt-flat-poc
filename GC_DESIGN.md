# Batched writes + dense packing + inline self-tuning GC

Replaces the earlier separate-pass GC sketch (branch `batched-write-gc`). Grounded
in the 1B drill-down + IOPS/bandwidth sweeps:

- Phase B at 1B is **97% device I/O** (pwrite 61%, pread 36%), 3% CPU.
- Random 16 KiB writes are *not* slow (175k IOPS); **bandwidth** rises with block
  size (16K→48–64K ≈ 2316→2900 MB/s, ~1.25×, then plateaus). The workload writes
  at only ~970 MB/s — ~40% of the 16K ceiling — because each worker reads-then-
  writes serially.
- The 256 GiB file holds only **74.6 GiB live** (leaves ~29% full): per-leaf
  16 KiB page padding both inflates writes ~3.4× and bloats the working set past
  RAM (so every `pread` is a device round-trip).

So the win is **(a) coalesce writes into large sequential `pwrite`s** and **(b)
pack records densely** to cut write volume *and* shrink the working set toward
RAM-resident (killing most of the 36% read cost). GC is needed because dense
append-and-supersede creates garbage; we make GC **inline and self-tuning**.

## Storage model

- **Address unit = 256 B.** `DiskPtr { unit: u32, len: u32 }`, byte offset =
  `unit * 256`. Records are packed at 256 B alignment (≤256 B slack/record vs the
  current ~11.6 KiB page padding). 256 B × u32 ⇒ 1 TiB addressable (file ≈ 125 GiB,
  fine). **`DiskPtr` stays 8 bytes** — only the unit's meaning changes (16384→256),
  so the serialized layout (two u32s) is unchanged.
- **Region = 64 KiB** (256 units) — the bandwidth sweet spot and the GC reclaim
  unit. Per-region live-byte counts drive cleaning. A record never straddles a
  region boundary (pad to the next region if it would).
- **Physical writes are ≥16 KiB-aligned.** A batch packs all its output records
  densely into one buffer and writes it in 16 KiB-aligned chunks (round the tail
  up to 16 KiB; ≤16 KiB slack/batch) — no sub-page RMW penalty, while records
  inside are 256 B-dense.
- **Records are length-delimited** (`u32` len prefix) so a region can be scanned
  record-by-record during GC without consulting the frontier.

## Inline GC (folded into the batch)

The frontier is the source of truth for liveness; GC rides the work Phase B/C
already do, so there is **no separate pass and no full-frontier walk**.

Per `insert_batch`:
1. **Phase A** routes the batch's keys to their leaves (as today) and records the
   foreground target prefixes.
2. **Select victims:** pop the `R` least-live regions (emptiest-first) from a
   live-fraction–bucketed index (O(1) per pick; excludes the write head + free
   pool). `R` is set by the controller below.
3. **Phase B (parallel):** for each foreground group, read+rewrite the leaf as
   today. *Additionally*, for each victim region: one **sequential 64 KiB read**;
   walk its length-delimited records; for each, take its `prefix`, look it up in
   the frontier (read-only in Phase B), and if the frontier `DiskPtr` still points
   here it's **live** — queue it for relocation (verbatim bytes). Skip records
   that are stale, or whose prefix is a foreground target this batch (the
   foreground rewrite supersedes them).
4. **Write:** pack all outputs — foreground rewrites + relocated records — densely
   and append as large 16 KiB-aligned `pwrite`s, assigning each a `DiskPtr`.
5. **Phase C (serial):** install every new pointer into the frontier —
   `install_at_key` for foreground groups, `install_by_prefix` for relocated
   records. Relocations are verbatim ⇒ same hash ⇒ **root unchanged**, no hash
   invalidation. Then mark fully-evacuated victim regions free → pool; next batch's
   writes reuse the pool before extending the file.

Relocation reads are sequential (one read/region); relocation writes are folded
into the batch's packed writes (no extra write ops). Liveness is O(records in the
R victim regions), not O(frontier).

## Self-tuning cleaning rate (target 60% utilization)

Utilization `u = live / active`, where `active = end_page − free_pool` (the
in-use footprint; the free pool is reclaimed space we haven't reused yet). Cleaning
more shrinks `active` (raises `u`); writing consumes the pool / extends the file
(lowers `u`). So `R` (regions cleaned per batch) is the control knob and `u`
responds to it directly.

Proportional controller, per batch:
```
u   = live / active
err = TARGET_UTIL - u            // >0 ⇒ under-utilized (too much garbage) ⇒ clean more
R   = clamp(R + round(GAIN * err), 0, R_MAX)
```
- `TARGET_UTIL = 0.60` → file settles at ≈ live/0.6 ≈ 1.67× live ≈ ~125 GiB.
- `err > 0` (u below target) ⇒ R increases ⇒ more cleaning ⇒ free pool grows ⇒
  file stops extending ⇒ u rises. `err < 0` ⇒ R decreases. Self-stabilizes at 60%.
- `GAIN` (regions per unit error, e.g. ~a few thousand so a 10% miss moves R by a
  few hundred) and `R_MAX` (per-batch stall bound) are **tunable** — sweep against
  the large run.

Cost note: at 60% the emptiest regions are still ~40–50% live, so GC write-
amplification is ~0.7–1× (the price for a small, RAM-resident file). Emptiest-first
selection + the controller keep it minimal; raising `TARGET_UTIL` trades file size
for less GC.

## Implementation stages (each builds + tests green; root must stay identical)

GC-first (lower risk, measurable sooner), then the format-rippling pack change.

- **Stage 1 — region allocator + batched writes + inline GC + controller, still
  page-granular (16 KiB records).** Append allocator over 64 KiB regions with
  per-region live counts; coalesced batched `pwrite`s; inline victim evacuation in
  Phase B; `install_by_prefix` in Phase C; free-pool reuse; the 60% controller.
  Manifest recomputes region liveness from the frontier on open. Validates the
  *full* GC mechanism + the write-bandwidth path without touching `DiskPtr`. Verify
  `batchcheck` root unchanged + a churn test holding the file near 1.67× live.
- **Stage 2 — dense packing.** `DiskPtr` unit 256 B, length-delimited records,
  256 B-aligned packing inside the 16 KiB-aligned batch write, records don't
  straddle regions. This is the working-set/read win on top of Stage 1. Verify
  root unchanged + flat-size/live ratio drops toward ~1.0 + working set ≈ RAM.
- **Stage 3 — measure + tune** `GAIN`/`R_MAX`/region size against a fresh large
  run; compare µs/key and the Phase-B device split to the 30 µs/key baseline.

## Invariants

`DiskPtr` 8 bytes; records verbatim across relocation ⇒ Merkle root unchanged;
frontier is the sole liveness authority; physical writes ≥16 KiB-aligned; a record
never straddles a region.
