# Log-structured writes + segment GC

**Status:** implemented on branch `batched-write-gc`, **not yet built or tested**
(written while a 1B preload run held the SSD; build + test in the morning — see
"Morning checklist" at the end).

## Problem

Today every insert rewrites a leaf record to a *new* free-list location and frees
the old one. The free list hands out best-fit holes scattered across the file, so
the device sees **random 16 KiB writes** — the measured ceiling is ~168 k IOPS,
and that write IOP is the dominant cost of an insert at scale (Phase B was ~83 %
pwrite in the 130 M device-bound run).

The `iops` sweep showed **sequential** 16 KiB writes run ~3× faster than random.
So the win is to place writes sequentially. A throwaway "append every batch, never
reclaim" prototype confirmed the speedup but grew the file ~9× (every superseded
record stays forever). The missing half is **garbage collection**.

## Key insight: sequential placement comes from the *allocator*, not from coalescing

The prototype conflated two ideas. They're separable:

1. **Sequential placement** — append new records at a moving "head" instead of
   scattering them into free-list holes. *This alone* makes the device see a
   contiguous write region even when many worker threads each issue their own
   `pwrite`, because the allocator hands out consecutive page ranges. This is the
   primary win.
2. **Syscall coalescing** — pack several records into one `pwrite`. A secondary
   win (fewer syscalls); already prototyped as `write_batch`.

Both are kept, but the allocator change is what matters.

## Design: segmented log-structured store

### Segments

The file is a sequence of fixed-size **segments** of `SEGMENT_PAGES` pages
(4096 pages × 16 KiB = **64 MiB**). All writes append to the current **head**
segment at a moving `next_page`. When a batch wouldn't fit in the head's
remainder, the head is finalized and a new head is opened — from the **free-segment
pool** if non-empty (reuse), otherwise a fresh segment at the file end (extend).
A single batch never straddles a segment boundary, so every record's segment is
exactly `page / SEGMENT_PAGES`.

`DiskPtr { page, len }` is **unchanged** — it still addresses bytes; the segment
is derived arithmetically.

### Liveness accounting

`SegmentAlloc` tracks `live[seg]` = live pages in each segment.

- **alloc** (`write_payload` / `write_batch`): bump `next_page`, `live[head] +=
  pages`, raise the file high-water.
- **free** (record superseded by a rewrite/promotion): `live[seg(ptr)] -=
  pages`. The space is *not* reused immediately — it becomes garbage until the
  whole segment is reclaimed.

### Garbage collection (evacuation)

The frontier is the **single source of truth for liveness**: every
`RamChild::Disk` is a live record; everything else in the file is garbage. So GC
needs no on-disk record headers and no separate reverse index — it reads the live
set straight off the in-RAM frontier.

A GC pass:

1. **Select victims.** Under the `SegmentAlloc` lock, pick Full segments (not the
   head, not already free) whose live fraction is below `EVAC_MAX_LIVE_FRAC`
   (evacuating a near-full segment relocates a lot for little gain), lowest-live
   first, up to `MAX_EVAC_SEGS` per pass.
2. **Relocate.** Walk the frontier once. For every `RamChild::Disk{ptr}` whose
   segment is a victim: read the record's raw bytes, append them at the head
   (new `DiskPtr`, same bytes ⇒ same hash ⇒ **root unchanged**), update the
   child's `ptr` in place, and adjust `live` (old segment down, new segment up).
3. **Reclaim.** Each victim now has `live == 0`; push it to the free-segment pool
   for reuse. (Defensive: only reclaim segments that actually reached 0, so an
   accounting bug can never free a segment that still has a live record.)

One frontier walk evacuates *all* victims at once, so the `O(frontier)` walk cost
amortizes over `MAX_EVAC_SEGS` reclaimed segments. The relocation reads are
random (scattered live records) but the relocation writes are sequential
(appended to the head) — GC trades a burst of background random reads for
sequential writes and reclaimed space.

### Trigger & steady state

After each batch, if the file is `< GC_TARGET_UTIL` live (i.e. garbage exceeds
`1 - GC_TARGET_UTIL` of the file) and the file is past a small floor
(`GC_MIN_PAGES`), run one GC pass. Appends reuse free segments before extending,
so once GC produces free segments as fast as rewrites create garbage, the file
size **stabilizes** at roughly `live / GC_TARGET_UTIL`. With `GC_TARGET_UTIL =
0.5` the on-disk file is ~2× the live bytes — the space overhead we pay for
sequential writes.

GC runs **serially between batches**, where `insert_batch` holds `&mut self`
(hence `&mut frontier`), so it never races the Phase-B workers and the frontier
mutation needs no locking.

## Persistence

The manifest drops the `FreeList` and keeps `cfg`, `upper`, `end_page`. On
`open`, `SegmentAlloc` is **recomputed from the frontier**: walk it, bucket each
live `DiskPtr` into `live[seg]`; mark all existing segments Full; open a fresh
head segment at the file end. This wastes at most the tail of the last
pre-checkpoint segment (≤ 64 MiB) and avoids any new serialized structure (and
the version skew that comes with it). It also means the **old `FreeList`-format
checkpoints cannot be reopened by this branch** — evaluate it with fresh runs.

## Tunables (all in one place near `SegmentAlloc`)

| const | value | meaning |
|---|---|---|
| `SEGMENT_PAGES` | 4096 (64 MiB) | reclamation granularity |
| `GC_TARGET_UTIL` | 0.5 | live/file floor before GC fires → ~2× space |
| `EVAC_MAX_LIVE_FRAC` | 0.5 | only evacuate segments below this live fraction |
| `MAX_EVAC_SEGS` | 32 (2 GiB) | victims per pass — amortizes the frontier walk |
| `GC_MIN_PAGES` | 4·SEG | don't GC tiny files |

## Cost model — and the read-strategy caveat

Per inserted record, comparing the two designs (device-bound, uncached):

| | in-place | log + GC v1 | log + GC v2 |
|---|---|---|---|
| foreground read | 1 random | 1 random | 1 random |
| foreground write | 1 **random** | 1 **seq** | 1 **seq** |
| GC read (to hold ~`UTIL`) | — | ~1 **random** | ~1 **seq** (1 big read/seg) |
| GC write | — | ~1 seq | ~1 seq |

At `GC_TARGET_UTIL = 0.5`, holding the file at 2× live means GC eventually
relocates ~1 live record per inserted record. The danger: **v1 reads each live
record with its own random `pread`**, so it adds a random read per insert. A
random read (~6–7 µs) costs about what the sequential-write *saved*, so v1 can
come out flat or worse at scale despite the sequential writes. This is the most
important thing to measure in the morning.

**v2 fixes it:** read each victim segment with one sequential 64 MiB read into
RAM and relocate live records from that buffer. GC reads become sequential, and
the model tilts clearly positive. v2 needs two frontier walks (collect victim
ptrs immutably → relocate from buffers, recording an old→new `DiskPtr` map →
re-walk to retarget pointers), because you can't hold a `&mut` to many pointer
slots across one walk. v1 ships first because it's simple and obviously correct;
v2 is a contained follow-up if v1 regresses.

Higher `GC_TARGET_UTIL` (e.g. 0.66) reduces GC frequency/volume at the cost of
more space — another knob to sweep.

- **GC walk:** `O(frontier)` per pass, amortized over `MAX_EVAC_SEGS` victims.
  Watch the pass cadence at scale: if GC fires every few batches the walk
  dominates — raise `MAX_EVAC_SEGS` and/or `GC_TARGET_UTIL`. A reverse index
  (segment → live ptrs) would remove the walk entirely but roughly duplicates the
  frontier in RAM; deferred.

## What this does NOT change

- `DiskPtr`, the record byte format, the Merkle hashing, the frontier structure,
  the lazy reader, Phase A/B/C decomposition. GC is a new serial maintenance step
  after Phase C; the write path swaps its allocator and defaults to coalesced
  appends.

## Morning checklist

1. Ideally after the 1B run finishes (so the build/test I/O doesn't compete):
   `cargo build --release && cargo test --release` (expect 10 pass), fix any
   compile nits.
2. `cargo run --release --example batchcheck 200000 10000` — **root + flat match
   must be YES** (GC and sequential placement must not change the root).
3. Add a GC-stress check: many overwrites of a small key set should hold the file
   near `live / GC_TARGET_UTIL` instead of growing unboundedly (watch
   `free_bytes` / `free_regions`, now "garbage" / "free segments").
4. Re-run the reopen-stress and a fresh large run; compare µs/key and the Phase-B
   pwrite share against the in-place baseline.
