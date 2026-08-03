# Flat-file trie storage design

Status: **implemented** — this describes the engine as it exists on the
mainline. It descends from the earlier *paged-node storage* proposal (see
**History** at the end for what shipped and what diverged).

## 1. The problem

A Merkle Patricia Trie over billions of keys cannot keep its nodes
individually addressable on disk without paying O(depth) random IOs per
update (the conventional trie-on-pages design), and cannot keep them all in
RAM. The design goal: **one random read per accessed key** — the same read
execution needs for the value — with commitment recomputed from that read,
and RAM bounded by the count of *records*, not keys.

## 2. The unit: a `DiskSubtree` record

A record is one subtree, compact-serialized as `[u32 len][payload]` and
placed at a 256 B-aligned offset (`DiskPtr { unit, len }`) inside a sequence
of 128 KiB regions. A record's payload is its node tree, recursively:

- `Leaf { path, value }` — values live **in** the trie (Ethereum leaf RLP);
  there is no external value store. Bytecode is the one exception
  (content-addressed code store, never read during hashing).
- `Extension` / `Branch` — interior structure, each carrying its node
  reference (`NodeRef`: hash, or the RLP itself when < 32 bytes, per
  Ethereum's inline-node rule).
- `Overflow { ptr, root }` — a child subtree living in its **own** record.
  Hash-transparent: a branch hashes identically whether a child is inline
  or overflowed, so layout never affects the root.
- `Account { fields, storage }` — a state-trie account leaf with its
  **storage subtree nested inline**: the account's `storage_root` is
  computed from the nested tree, never fetched. Records inside a storage
  subtree are addressed by **composite paths** (64 account nibbles ‖
  storage nibbles).

Every record stores its **full composite frontier path**, so any record is
independently relocatable — the property the garbage collector's
verify-then-install step depends on (asserted by `MPT_GC_ASSERT_PATHS=1`).

Defaults: `max_leaf_bytes` 16 KiB (hard cap), `target_leaf_bytes` 8 KiB.
(`min_promote_bytes` survives in `Config` but is vestigial — nothing in the
operative path reads it; kept only by legacy tests.)

## 3. The RAM frontier

The frontier holds every node whose subtree exceeds one record, as
`RamNode::{Branch, Extension}` plus promoted `Account` nodes. A branch slot
(`RamChild`, size-guarded ≤ 56 B) is one of:

- `Ram(Box<RamNode>)` — deeper frontier structure;
- `Disk { ptr, root }` — a record pointer plus that record's own root
  digest (what the parent needs to hash). Child digests *inside* a record
  are **not** cached in RAM — they come along with the record read;
- `Mem(Arc<[u8]>)` — an in-RAM record: RAM-build mode during bulk
  bootstrap, and the hot-record cache (Mem-on-rewrite with deferred
  write-back) in steady state.

Because a frontier terminal covers a whole record (~10² keys), frontier
size is Θ(n / record-capacity) — **linear in n with a sub-byte-per-key
constant**, dominated by the one slot + amortized parent-branch share per
record. The trie's log-depth does not multiply memory: node count per
level shrinks geometrically going up, so the log-many upper levels sum to
a constant factor.

### The boundary: promote-on-max

There is no fixed-depth knob and no size-class rule. A record that exceeds
`max_leaf_bytes` on rewrite **promotes**: its top branch is lifted into the
frontier and its children become records (oversized children keep
splitting). The boundary therefore tracks the data adaptively — dense
subtrees push the frontier deeper, sparse ones stay packed. `Overflow`
edges cover the remaining case of a fat child *below* a still-packed
parent, keeping the parent record bounded without promoting it.

## 4. The sibling-hash flow

The reason the whole design works: updating a key requires the sibling
hashes along its path, and they are already in the bytes being read.

1. Route through the RAM frontier to `Disk { ptr, .. }`; read the record
   (one IO — the same IO that fetches the value for execution).
2. Parse **lazily**: only nodes on the touched path are expanded;
   untouched sibling subtrees stay `Raw` (zero-copy slices of the shared
   read buffer) with their cached `NodeRef`s, and are written back
   verbatim.
3. Re-hash only the touched path inside the record; recompute the record
   root; splice the new `{ptr, root}` into the frontier and bubble the
   root up through the frontier's cached hash cells (RAM-only).

No stored mid-tree hash pages, no extra reads, per-update hashing
proportional to the touched path — independent of record size.

## 5. The block API

`apply_block(ops) -> (root, inverse_diff)` is the operative interface: a
per-block batch of `SetAccount / DeleteAccount / WipeStorage / SetStorage /
DeleteStorage`.

- **Phase A (route):** group ops per record via the frontier; an advisory
  pre-read (`prefetch_block`) warms the records a pending block will touch.
- **Phase B (parallel):** per-record reads fanned across the top branch's
  16 disjoint subtrees and a deep pread queue; apply, re-hash, rewrite or
  promote. Deletion re-folds structure canonically (a branch left with one
  child merges into its survivor).
- **Phase C (install):** splice results, recompute the root once, emit the
  exact inverse ops for reorg rollback.

## 6. Writes, regions, GC

Records are never updated in place: every rewrite **appends** — one
batched sequential write per apply. Old copies become garbage inside
their regions; per-region liveness is bookkept at write time, and freed
regions **stage** until the current apply's read-ahead window closes
before reuse (prevents read-after-free aliasing within a batch).

The garbage collector runs **off the block critical path** as a split
design: `gc_collect` on a pinned snapshot (lock-free: pick low-utilization
regions, read victims in parallel, filter to live records) and
`gc_install` under the writer (re-verify each candidate against the live
frontier via its stored path; relocate or discard). Embedders schedule
collect into idle windows. Known structural limit: at sustained
device-saturating write load, reclaim competes with fresh appends for the
same bandwidth — the space bound degrades gracefully (file grows) rather
than stalling the block path.

## 7. Persistence

Checkpoint-based. The flat file holds the data; the index — frontier
structure, pointers, cached hashes, allocator high-water — is RAM-only and
serialized to the `.meta` manifest by `persist()` (spill in-RAM records,
fsync the file, write the manifest atomically). A crash reopens at the
last checkpoint; node integrations pair this with a root-verified
`<flat>.height` sidecar so a torn checkpoint is detected rather than
followed, and re-derive the gap from the chain.

## 8. Hashing

Mainnet-exact keccak/RLP (`src/eth.rs`), validated against the official
`ethereum/tests` vectors. The root is a pure function of the key set —
independent of record size, promotion history, and layout — gated by
`root_is_independent_of_leaf_size` and the batch-vs-one-by-one parity
tests. (The PoC-era domain-tag scheme is gone; this trie hashes byte-for-
byte like Ethereum's.)

## 9. Known limits / future directions

- **Write amplification** is the design's structural cost: a whole record
  rewrites for any change inside it. It is paid deliberately — sequential
  appends instead of random page writes — and the GC exists to reclaim it.
  Record size (`MAX_LEAF_KIB`) is the tuning dial: bytes-per-write vs
  frontier size vs read fatness.
- **Hash-transplant follower** (planned): persist externally-computed
  node hashes instead of re-deriving them in the apply, removing the
  apply-side hashing from the follower's critical path.
- **Two-level disk variant** (design sketch): for key counts where even
  the linear frontier outgrows RAM, insert an intermediate disk record of
  two 16-ary levels (256 record pointers, ~11 KiB — fits the same IO
  unit). Exactly 2 IOs per accessed key; frontier constant divides by the
  256 fanout; write bytes roughly 1.7× (the intermediate rewrites when a
  child record's root changes). Extends the design by two orders of
  magnitude of keys for one extra read.

## 10. History

This document supersedes the *paged-node storage* proposal (the previous
revision of this file). What survived from it: the `Overflow` edge and its
hash-transparency contract, the pointers-not-digests RAM model, the
sibling-hashes-from-the-record-read principle, and the bounded-RAM goal.
What diverged: the header/inline-area/locator record format was never
built — `DiskSubtree` remained, with lazy `Raw` parsing providing the
sibling property; the `min_promote` adaptive boundary and two-trigger
migration gave way to promote-on-max; the RocksDB value store was removed
entirely (values moved into the leaves for the Ethereum-exact model); and
PoC domain-tag hashing was replaced by mainnet keccak/RLP.
