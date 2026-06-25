# mpt-flat-poc

A proof-of-concept **Merkle Patricia Trie (MPT)** with a *flat* storage layout: a
small in-RAM "frontier" of the trie sits on top of larger **subtrees serialized
into a single flat file**, while the actual key→value payloads live in an
embedded **RocksDB** store. It explores how to keep an authenticated key/value
trie mostly on disk while bounding RAM usage and keeping per-insert hashing
work proportional to what actually changed.

This is a benchmarking/learning artifact, not a production database. Keys are
fixed 32-byte hashes (64 nibbles); values are arbitrary byte strings.

---

## Architecture at a glance

```
                    FlatMpt (src/lib.rs)
   ┌───────────────────────────────────────────────────────────┐
   │                                                             │
   │   upper: RamNode            ← in-RAM trie "frontier"        │
   │   ┌───────────┐               (Branch / Extension nodes,    │
   │   │  Branch   │                each caching its own hash)   │
   │   └─────┬─────┘                                             │
   │      ┌──┴───────────┐                                       │
   │  RamChild::Disk   RamChild::Ram                             │
   │   { ptr, root }     (Box<RamNode>)                          │
   │        │                                                    │
   └────────┼────────────────────────────────────────────────-─┘
            │ DiskPtr { offset, len }
            ▼
   store: FlatFile  ──────────────────────  values: rocksdb::DB
   ┌──────────────────────────────┐         ┌──────────────────────┐
   │ flat file of DiskSubtree      │         │  key → value bytes    │
   │ records  [len][compact bytes] │         │  (the trie only ever  │
   │ + FreeList (reuse of holes)   │         │   stores value_hash)  │
   └──────────────────────────────┘         └──────────────────────┘
```

There are **three storage tiers**, each with a distinct job:

### 1. The RAM frontier (`RamNode` / `upper`)
The top of the trie is held in memory. Its nodes are `Branch` (16-way) and
`Extension` (shared-nibble path) — the same shapes as a classic MPT. A branch
slot points either to another in-RAM node (`RamChild::Ram`) or to a subtree that
has been pushed to disk (`RamChild::Disk`, holding a `DiskPtr`, the subtree's
cached `root` hash, and its byte size).

The frontier stays small on purpose: large subtrees are written to disk and
represented by a single `RamChild::Disk` pointer. `ram_nodes()` reports the
current frontier size, and several tests assert it stays bounded.

### 2. The flat file (`FlatFile` / `store`)
Disk subtrees are compact-encoded `DiskSubtree` records appended to one flat file
as `[len: u32][payload]`. A `DiskPtr { offset, len }` addresses a record. The
payload keeps the subtree's cached Merkle hashes, but avoids bincode's enum,
`Vec`, `Option`, and `Box` overhead for the hot disk records.

Crucially, the file is **not** purely append-only: a `FreeList` tracks regions
vacated by rewritten/split subtrees. New writes prefer a best-fit free region
(splitting off any remainder) and only extend the file when nothing fits; freed
regions are coalesced with their neighbours. This keeps the file from growing
unboundedly as the same keys are overwritten.

### 3. The value store (`rocksdb::DB` / `values`)
The trie itself only ever manipulates a `value_hash` (a keccak of the value).
The real value bytes are kept in an embedded RocksDB instance that lives in a
sibling `<flatfile>.values` directory. Inserts buffer values in a small RAM
overlay and flush them to RocksDB as a single `WriteBatch` every `VALUE_BATCH`
inserts (and on `flush`/`persist`); `get_value` consults the overlay first, so
reads always observe the latest write. Batching the writes is ~10% faster than
one `put` per insert.

### Persistence (`persist` / `open` and the `.meta` manifest)
The flat file and value store hold their data on disk, but the *index* tying
them together — the `upper` frontier (structure, disk pointers, cached hashes),
the `FreeList`, and the high-water `end` — lives only in RAM. `persist()`
checkpoints that index into a sibling `<flatfile>.meta` manifest: it fsyncs the
flat file, then writes the bincode-serialized `Manifest` atomically (temp file +
rename). `open(path)` reverses this — it loads the manifest and **reattaches** to
the existing flat file and RocksDB without truncating them, fully restoring a
writable trie (cached hashes and all). `create()` remains the from-scratch path
(it truncates the flat file and recreates the value store).

---

## How an insert flows

`FlatMpt::insert(key, value)`:

1. `value_hash = hash_leaf_value(value)`, then buffer `key -> value` in the RAM
   overlay (flushed to RocksDB in `WriteBatch` chunks).
2. `insert_ram` walks/mutates the RAM frontier down to the relevant branch slot:
   - **Empty slot** → create a single-entry `DiskSubtree`, write it, install a `RamChild::Disk`.
   - **`RamChild::Disk`** → read the subtree, **incrementally insert** the new
     entry into its `Node` tree (`node_insert`), and either rewrite it in place
     (reusing freed space) or, if it exceeds `Config::max_leaf_bytes`,
     `split_subtree` it back up into the RAM frontier.
   - **`RamChild::Ram`** → recurse.
3. `store.flush()` and return the new root via `root()`.

### Hashing & memoization (why per-insert hashing is "essential")
Recomputing the whole trie hash on every insert is the naive cost. This PoC
avoids it on both tiers:

- **Disk subtrees:** every `Node` caches its own Merkle hash, computed once at
  construction (`make_leaf` / `make_extension` / `make_branch`) and serialized to
  disk. `node_insert` mutates a subtree in place and recomputes hashes **only
  along the changed root-to-leaf path**; untouched sibling subtrees are reused
  verbatim with their cached hashes.
- **RAM frontier:** `RamNode` caches its hash in a `Cell`. An insert calls
  `invalidate_ram` as it descends, clearing caches only on the touched path;
  `hash_ram`/`root` then recompute just those and reuse the rest. Disk children
  contribute their already-cached `root`, so they're never re-hashed.
- The empty-node hash `keccak(&[0])` is a constant, computed once (`empty_hash`).

The net effect: a steady-state insert performs only the keccak calls that are
genuinely new (≈ path length), not a count that scales with subtree size. The
`hashaudit` example verifies this empirically (0% redundant hashing).

---

## Repository layout

| Path | What it does |
|------|--------------|
| [`src/lib.rs`](src/lib.rs) | The entire engine. See the component map below. |
| [`benches/insert.rs`](benches/insert.rs) | Criterion throughput benchmark — 1000 inserts under random, sequential-hashed, and shared-prefix key distributions. Run with `cargo bench --bench insert`. |
| [`benches/profile.rs`](benches/profile.rs) | Wall-clock **attribution** benchmark: splits insert/read time across hashing, (de)serialization, file IO, flush, and RocksDB. Run with `cargo bench --bench profile --features profiling`. |
| [`benches/large.rs`](benches/large.rs) | Steady-state benchmark: preload N keys (`LARGE_PRELOAD`, optional `LARGE_BATCH`/`LARGE_MAX_LEAF_KIB`), then time new inserts + overwrites, logging per-10M stats. Easiest via the script below. |
| [`examples/hashcount.rs`](examples/hashcount.rs) | Diagnostic: prints keccak calls per individual insert, showing how hashing scales. `cargo run --release --example hashcount --features profiling`. |
| [`examples/hashaudit.rs`](examples/hashaudit.rs) | Diagnostic: classifies each keccak call of an insert as essential / recomputed-unchanged / duplicate, to prove hashing is minimal. `cargo run --release --example hashaudit --features profiling`. |
| [`examples/diskusage.rs`](examples/diskusage.rs) | Diagnostic: reports the flat-file index footprint (bytes/entry) for N inserts. `cargo run --release --example diskusage [N]`. |
| [`examples/sizecheck.rs`](examples/sizecheck.rs) | Diagnostic: reports flat-file length/free bytes/RAM nodes for the three benchmark key distributions. `cargo run --release --example sizecheck`. |
| [`scripts/run-large-bench.sh`](scripts/run-large-bench.sh) | Build + run `benches/large.rs` with env knobs (`PRELOAD`, `BATCH`, `MAX_LEAF_KIB`, `PROFILE`); documents prereqs and the `TMPDIR`-must-be-a-real-disk caveat for large runs. |
| [`Cargo.toml`](Cargo.toml) | Dependencies and the `profiling` feature flag. |

### `src/lib.rs` component map

Public API and types:
- **`FlatMpt`** — the top-level database. Fields: `cfg`, `store` (`FlatFile`),
  `upper` (`RamNode`), `values` (`rocksdb::DB`).
  - `create(path, cfg)` — fresh DB (truncates the flat file, recreates the value store).
  - `open(path)` — reopen a previously `persist`ed DB (reattaches, no truncation).
  - `persist()` — checkpoint the RAM frontier + free list to the `.meta` manifest.
  - `flush()` — flush buffered values to RocksDB without a full checkpoint.
  - `insert(key, value) -> Hash` — insert/overwrite, returns the new root.
  - `get_value(key) -> Result<Option<Vec<u8>>>` — read from RocksDB.
  - `root() -> Hash` — memoized Merkle root of the whole trie.
  - `ram_nodes()`, `flat_file_len()`, `free_bytes()`, `free_regions()`,
    `disk_accesses_for_key()` — observability helpers used by tests/benches.
- **`Config`** — leaf-size thresholds: `target_leaf_bytes`, `max_leaf_bytes`
  (rewrite vs. split), `min_promote_bytes` (promote to its own disk record vs.
  fold into a remainder).
- **`Hash` / `Key`** — `[u8; 32]` aliases. **`DiskPtr`** — `{ offset, len }`.
- **`prof`** — opt-in (`--features profiling`) wall-clock attribution + a keccak
  audit hook. Compiles to zero-cost no-ops when the feature is off.

Internal storage:
- **`FreeList`** — coalescing, best-fit allocator over freed flat-file regions.
- **`FlatFile`** — the flat file plus its `FreeList` and high-water `end`;
  `write_payload` / `read` / `free` / `flush` / `sync` of `DiskSubtree` records.
- **`Manifest` / `ManifestRef`** — the serialized checkpoint (`cfg` + `upper` +
  `free` + `end`) read/written by `open` / `persist` via the `.meta` sidecar.

Internal trie:
- **`Node`** (`Empty` / `Leaf` / `Extension` / `Branch`) — a disk subtree's
  Merkle structure; each non-trivial variant caches its `hash`. Serialized.
- **`DiskSubtree`** — `{ prefix, node }`: the compact-encoded node plus the
  nibble prefix it is rooted at. Splits derive `(Key, value_hash)` entries from
  the node on demand.
- **`RamChild`** — `Ram(Box<RamNode>)` or `Disk { ptr, root, bytes }`.
- **`RamNode`** (`Empty` / `Extension` / `Branch`) — RAM frontier nodes with a
  `Cell`-cached hash.

Key functions:
- `insert_ram` — frontier walk/mutation; `invalidate_ram` clears path caches.
- `node_insert` — incremental insertion into a disk `Node` (path-only re-hash).
- `build_node` / `make_*` / `single_entry_node` — canonical node construction
  with hash caching.
- `split_subtree` — turn an oversized disk leaf back into a RAM branch frontier.
- `hash_ram` / `hash_node` / `hash_join` / `hash_leaf_value` / `keccak` /
  `empty_hash` — the hashing layer (keccak-256).

---

## Building & running

```bash
cargo test                                            # unit tests (debug build also
                                                      # cross-checks incremental vs. full rebuild)
cargo bench --bench insert                            # throughput
cargo bench --bench profile --features profiling      # time attribution
cargo run --release --example hashcount --features profiling   # hashes per insert
cargo run --release --example hashaudit --features profiling   # essential-hashing audit
```

The `profiling` feature gates all instrumentation; with it off (the default)
the hot path carries no measurement overhead.

---

## Known limitations / non-goals

- **Persistence is checkpoint-based, not continuous.** The RAM frontier and free
  list are only durable as of the last `persist()`. A crash after inserts but
  before `persist()` reopens at the previous checkpoint (any newer flat-file
  records are orphaned, and the value store may hold unreferenced values). There
  is no write-ahead log for the trie index.
- **Per-insert `flush()` is not an `fsync`** (only `persist()` fsyncs the flat
  file). Crash durability between checkpoints is not provided.
- **Write amplification.** Each insert into a disk leaf rewrites the whole compact
  subtree record. The compact format keeps this cheaper than bincode, but the
  single-record rewrite remains the central cost of the design.
- **Splits rebuild.** An overflowing leaf is rebuilt from its entries (its hashes
  recomputed) — incremental hashing covers ordinary inserts, not splits.
- **PoC value model.** Keys must be 32 bytes; values are duplicated between the
  trie's `value_hash` and the RocksDB payload.
