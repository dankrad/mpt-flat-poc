# Paged-node storage design

Status: **proposal** (no code yet). Branch `paged-nodes`, forked from
`ram-optimization` @ `691ac51`.

## 1. Why

The current design stores each disk subtree as one monolithic blob (`DiskSubtree`)
and pushes a per-leaf reference (`RamChild::Disk { ptr, root }`) into the RAM
frontier. At scale this hits a **radix-16 split cascade**:

- When a leaf reaches `max_leaf_bytes` it splits into **all 16** next-nibble
  children at once. A whole generation of leaves does this ~simultaneously
  (`leaves` jumps to 16ⁿ).
- Each split mints a swarm of tiny children. Measured at 8 KiB:
  `split: 1,067,705 new @ 555 B` — ~3 keys each.
- Page-aligned, each ~555 B child grabs a full 4 KiB page → ~86% padding waste.
- Each tiny child is also a frontier entry → RAM balloons (11.8 GiB observed
  at 870M vs ~0.5–1.5 GiB ideal; flat file ~5× ideal).

The byte-granular variant avoids padding but fragments the free list
(6.5M regions at 480M) and eventually needs compaction. Neither is good.

## 2. Goal / non-goals

**Goal:** bound RAM ~independently of key count (target ~0.5–1.5 GiB at 1B),
kill the cascade and the padding swarm, while keeping ~1 read per lookup and
the same Merkle root as the canonical radix-16 MPT.

**Non-goals:** changing the *logical* trie (the radix-16 structure is fixed),
changing the value store (RocksDB unchanged), unbounded-RAM full-index designs.

> **Phase 2a finding (done, `c05d971`).** The root was *not* storage-independent:
> RAM and disk hashed the same node type with different domain tags (ext 1 vs 4,
> branch 2 vs 5), so the root secretly depended on `max_leaf_bytes` / where the
> RAM–disk boundary fell. Since the paged design *moves* that boundary, the tags
> were unified (ext=4, branch=5 everywhere) so the root is a pure function of the
> key set — order-, config-, and layout-independent. This **changes the absolute
> root vs `ram-optimization`** (different tags); both stay internally consistent.
> Locked in by `root_is_independent_of_leaf_size`.

## 3. The unit: a *page node* record

Replaces `DiskSubtree`. A page node is a subtree rooted at a nibble-prefix,
stored as one variable-size record laid out in three logical parts:

```
+-----------------------------------------------------------+
| (1) HEADER                                                |
|     prefix nibbles (extension, if any)                    |
|     for each of 16 branch slots:                          |
|        child_digest : [u8;32]        (Merkle hash)        |
|        locator      : Empty                               |
|                     | Inline { off:u32, len:u32 }  -> (2) |
|                     | Overflow { ptr:DiskPtr }     -> (3) |
+-----------------------------------------------------------+
| (2) INLINE AREA                                           |
|     small child subtrees, packed back-to-back,            |
|     each located by (off,len) from the header             |
+-----------------------------------------------------------+
```

**(3) Overflow** is *not* bytes in this record — overflow children are their
own page-node records elsewhere in the flat file, reached via `Overflow{ptr}`.
Each is recursively a (1)/(2)/(3) page node.

The header is small and bounded: ≤16×(32 + ~10) ≈ **~700 B**, regardless of how
much data hangs below it. That bound is what makes the design work.

## 4. RAM model — pointers, not digests

The RAM frontier holds, per page-node record, exactly what it holds today:
`RamChild::Disk { ptr, root }` — a pointer plus the *record's own* root digest
(needed by its parent to hash). It does **not** cache the header's 16 child
digests; those live on disk in (1).

Because small children are **packed** (many per record), the number of records
is far below the number of leaves in the current design — so the frontier
shrinks proportionally. RAM is bounded by the count of *records*, not *keys*.

The split that matters:
- **Navigation + parent-hashing** → RAM (`ptr` + `root`), small.
- **Child digests (authentication within a record)** → disk header (1), read
  on demand when we touch the record.

### Where the RAM/disk boundary falls — adaptive, via `min_promote`

There is **no fixed-depth knob.** The boundary is the same kind of adaptive,
data-driven boundary the engine already uses, refined by one rule:

> A branch lives in the **RAM frontier iff all its children are "fat"
> (≥ `min_promote`)** — each child then deserves its own record, so the branch is
> a RAM node whose children are records. A branch that has any **small** children
> (< `min_promote`) is instead **packed into one disk record** (small children
> inline; fat children as `Overflow` pointers).

Consequences, all desirable and all adaptive:
- The frontier covers the **upper tree** (where subtrees are big → fat children)
  and stops exactly where subtrees shrink below `min_promote` — which is *where
  the old design's swarm began*. Instead of fanning that level into 16 tiny RAM
  disk-leaves, we pack it into one record.
- **Bounded RAM:** packed records hold ~`max/leaf` keys (~80 at 8 KiB) instead of
  ~3, so ~15–25× fewer records ⇒ ~15–25× smaller frontier (~0.75 GB at 1B vs
  11.8 GB). The frontier stays adaptive but small *because the leaves are fat*.
- **~1 read** for the common case (RAM navigation down to the boundary record,
  then one read). An overflow hop is added only for a fat child *below* the
  boundary, and the chain there is short — its length is set by the
  `min_promote : max_leaf` ratio (~1–2), and tunable.
- `min_promote` is the single dial: bigger ⇒ frontier stops higher (less RAM,
  slightly deeper disk chains); smaller ⇒ frontier reaches deeper (more RAM,
  shallower chains).

## 5. The sibling-hash flow (the question that drove this design)

When we update one child and must recompute the record's branch hash, we need
the *other 15* child digests. They are in header (1), which we already read to
get to the child. **No extra read, no RAM cache.** Concretely, an insert:

1. Navigate RAM frontier → reach `RamChild::Disk{ptr}` → read record `(1)+(2)`.
2. Route by next nibble:
   - `Inline` → the child subtree is in (2) we just read; insert into it.
   - `Overflow{ptr}` → recurse into that record (its `(1)+(2)`).
3. Recompute the touched child's digest; write it into header slot's
   `child_digest`; recompute the header branch hash = this record's root.
4. Persist: write `(1)+(2)` back; set the RAM `root` for this record.
5. Bubble the new root up the RAM frontier (existing incremental re-hash).

Sibling digests for step 3 come from the header read in step 1. ✔

## 6. Migration (where `min_promote`/`target`/`max` finally do real work)

Per-record budget, mapped onto the existing `Config`:

| Config field        | role in paged design                                        |
|---------------------|-------------------------------------------------------------|
| `max_leaf_bytes`    | hard cap on a record `(1)+(2)`; over it ⇒ migrate out       |
| `target_leaf_bytes` | migrate inline children to overflow until back under this   |
| `min_promote_bytes` | a child must be ≥ this to earn its *own* overflow record    |

Two distinct triggers (do **not** conflate them):

- **Proactive promotion — gated by `min_promote`.** When an inline child *grows*
  to ≥ `min_promote`, give it its own overflow record promptly, even if the
  record isn't full. These are the common, low-padding promotions (the child is
  already sizeable, so its page is well-filled). Children below `min_promote`
  stay packed in (2) — this is what avoids minting ~555 B records.

- **Forced shedding — ignores `min_promote`.** If a record still exceeds `max`
  (only happens when it's packed with many *sub-`min_promote`* children), it
  **must** shed: promote the **largest** child, then the next largest, until the
  record is ≤ `target`. `min_promote` does *not* gate this — progress is
  mandatory, so a full record is never stuck.

  *Why this is safe (it was the obvious deadlock):*
  - **Converges:** each forced promotion removes one child; header is bounded
    (~700 B). Worst case, all 16 promote and the record becomes a *bare 16-way
    branch header* (≤ `target`) — a normal MPT branch over 16 child records.
  - **No swarm:** one child at a time, not 16-at-once — incremental, not a
    synchronized cascade.
  - **Transient padding amortizes:** a forced-promoted child starts ~`max/16`
    but under uniform load grows toward `max` before *it* sheds, so a record
    averages ~1 full page over its life. Padding is only on young records.
  - **Skew** (all data under one nibble) pushes data down a level; hashed/uniform
    keys diverge within a few nibbles, and key length (64 nibbles) + extension
    compression bound the chain.

- **Depth comes from overflow chains, not header splits.** The header branch is
  always 16-way; a child that keeps growing becomes its own page node with its
  own (1)/(2)/(3), recursively. No 16-at-once cascade — migration is incremental
  (a few children per insert), spread over time.

So `min_promote` is a *preference* for voluntary promotion, not a hard floor that
can block a full record.

This means we can **keep page-alignment** for low fragmentation: records are now
either packed-full (inline) or ≥ `min_promote` (overflow), so padding is a small
fraction instead of ~86%.

## 7. Read amplification

- **Inline-child lookup:** 1 read (the whole `(1)+(2)` record).
- **Overflow-child lookup:** read this record (its (2) wasted), then recurse one
  record deeper. The overflow chain depth ≈ how many size-bands the key passes;
  shallow in practice. *(Open question 9.1: read header-only first to skip the
  wasted (2) read when records are large.)*
- **Update:** same record reads as the matching lookup, + the writes for the one
  touched child and the (small) header. Less write-amp than today, since we
  rewrite header + one child, not a whole monolithic leaf.

## 8. Persistence

Unchanged in shape: the RAM frontier (pointers + root digests) is checkpointed
to `.meta` as today; page-node records are self-describing on disk (header
carries digests + locators), values stay in the `.values` RocksDB. On reopen,
restore the frontier from the manifest; page nodes load on demand. Invariant:
a record's header branch hash == the root digest cached for it in RAM/manifest.

## 9. Open questions

1. **Header-only reads.** Read `(1)` alone (cheap) then `(2)` only if the routed
   child is inline, vs always reading `(1)+(2)`? Depends on record size vs the
   inline-hit rate. Start with always-`(1)+(2)`; measure.
2. **Overflow allocation.** Each overflow child = its own record (simplest), vs a
   managed shared overflow area. Start with own-record.
3. **De-migration.** If an overflow child shrinks below `min_promote` (deletes),
   pull it back inline? Defer — PoC is insert/overwrite-heavy.
4. **Record size unit.** `max_leaf_bytes` default (8 KiB) vs larger. Larger ⇒
   fewer records ⇒ smaller frontier, more bytes rewritten per insert. Re-sweep
   once it runs.

## 10. Correctness strategy

The root must be a pure function of the key set — storage layout does not change
the (logical) trie. Gates on every phase:
- `examples/batchcheck.rs`: batch == one-by-one (root + ideally flat size).
- `root_is_independent_of_leaf_size` (lib test): same keys, very different
  `max_leaf_bytes`, identical root. This is the real layout-independence check —
  it moves the boundary the most. (Supersedes the earlier "== `ram-optimization`
  root" idea, which no longer holds after the 2a tag unification.)
- For overflow specifically: an **all-inline build == a build that forces
  overflow** (tiny `min_promote`) must give the same root.

## 11. Phased plan

1. **(this doc)** — format + invariants. ✔
2a. **Storage-independent root.** Unify hash tags so the root doesn't depend on
   the RAM/disk boundary. ✔ (`c05d971`, `root_is_independent_of_leaf_size`)
2b. **`Node::Overflow` foundation.** Wire format + hash contract + round-trip
   test. ✔ (`c35cb0c`)
2c. **Record-crossing insert + adaptive boundary** — *next*. At a split, promote
   a branch to RAM only if all children are fat (≥ `min_promote`); otherwise keep
   it as a packed disk record and shed fat children to `Overflow`. Record-level
   insert traverses overflow edges (read → recurse → rewrite → re-hash up).
   **Gate:** all-inline build (huge `min_promote`) == forced-overflow build (tiny
   `min_promote`) — same root; plus `batchcheck` + `root_is_independent_of_leaf_size`.
3. **Migration tuning + stats.** Forced/proactive shedding to `target`; add
   `overflow_records` / `migrations` stats and a read-depth probe.
4. **Batch + parallel.** Re-integrate `insert_batch` over the new format (note the
   constraint: parallel precompute may *read* overflow records but writes stay
   serial).
5. **Scale.** Run `benches/large.rs` through the old cascade points (40M, 560M);
   confirm `avg_leaf`/`split`/`free_reg`/RAM stay healthy and track the
   ~150 GB / ~0.5 GB ideal.
