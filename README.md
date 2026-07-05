# mpt-flat-poc

> **This branch (`reth-integration`): Ethereum mainnet pathway.** On top of the
> base engine described below, this branch makes the trie Ethereum-*exact* and
> turns it into reth's state commitment:
>
> - `src/eth.rs` — keccak/RLP hashing identical to mainnet (validated against
>   ethereum/tests), values stored in leaves (no separate KV store on this path);
> - nested account model — one unified trie, account fields + storage subtrees
>   in-tree (`Node::Account`);
> - `apply_block(ops) -> (root, inverse_diff)` — per-block batched state
>   transition with reorg rollback, plus deletion with canonical collapse;
> - bulk ingest: `insert_batch_accounts` / `FlatMpt::create_ram_build`
>   (mainnet: 147.6M accounts + 535.4M slots reconstructed to reth's exact
>   state root);
> - key examples: `rethload_nested` (build a checkpoint from reth's hashed
>   tables), `readbench` (point reads), `blockbench` / `ethbench` /
>   `hotcontracts` (apply workloads), `replay` (diff-corpus replay).
>
> **To run it against a live reth mainnet node** — shadow verification or as
> the node's sole state commitment (merkle stages removed) — follow the runbook
> in [dankrad/flatmpt-exex](https://github.com/dankrad/flatmpt-exex), which
> pins all versions and includes the reth patch and benchmark instructions.
> Headline numbers there: 20.9× faster state-root updates over 200k mainnet
> blocks; execution→Finish gap 76.5 min → 84 s.

A proof-of-concept **Merkle Patricia Trie (MPT)** with a *flat* storage layout: a
small in-RAM "frontier" of the trie sits on top of larger **subtrees packed into a
single flat file**, while the actual key→value payloads live in an embedded
**RocksDB** store. It explores how to keep an authenticated key/value trie mostly
on disk while bounding RAM, keeping per-insert hashing proportional to what
actually changed, and building/updating trees of ~10⁹ keys fast.

This is a benchmarking/learning artifact, not a production database. Keys are
fixed 32-byte hashes (64 nibbles); values are arbitrary byte strings.

---

## Quick start

**Start here.** These are the two primary scripts — each takes a single argument (a
directory) and reproduces the headline results end-to-end:

```bash
# 1. Build the 1B-key baseline the fast way (RAM-build, 100M-key batches, spill at
#    30 GiB) → <dir>/ckpt.flat   (~1.4 us/key)
scripts/build-baseline-1b.sh /data/baseline

# 2. Benchmark the fused fast path on a throwaway COW clone of it — one-writer +
#    opportunistic GC + parallel writer + no-WAL + async values; prints us/key +
#    the device read/write split + the gc-evac breakdown.
scripts/bench-fused.sh /data/baseline
```

General-purpose variants (arbitrary size, custom batches, in-place growth):

```bash
scripts/build-tree.sh  /data/tree.flat 1000000000      # build a tree of any size
scripts/batch-bench.sh /data/tree.flat 10000000 10000  # benchmark batch inserts on a clone
scripts/grow-tree.sh   /data/tree.flat 100000000       # grow a tree + re-checkpoint in place
```

A checkpoint is three siblings: `tree.flat` (packed subtrees), `tree.flat.meta`
(the manifest/index), and `tree.flat.values/` (RocksDB). Reopen with
`FlatMpt::open`. The scripts auto-build the needed binaries and set good default
tuning; every knob is overridable via the environment (see **Tuning** below).

> **Disk note:** run with `$TMPDIR`/output on a real SSD, not tmpfs. A 1B-key tree
> is ~85 GiB flat + ~34 GiB values.

---

## Architecture at a glance

```
                    FlatMpt (src/lib.rs)
   ┌───────────────────────────────────────────────────────────┐
   │   upper: RamNode            ← in-RAM trie "frontier"        │
   │   ┌───────────┐               (Branch / Extension, each     │
   │   │  Branch   │                caching its own hash)        │
   │   └─────┬─────┘                                             │
   │     ┌───┴────────────┬───────────────┐                     │
   │  RamChild::Disk   RamChild::Ram   RamChild::Mem             │
   │   {ptr, root}     (Box<RamNode>)   (Arc<[u8]>, RAM-build)   │
   │        │                                                    │
   └────────┼────────────────────────────────────────────────-─┘
            │ DiskPtr { unit, len }   (256 B-aligned)
            ▼
   store: FlatFile  ──────────────────────  values: rocksdb::DB
   ┌──────────────────────────────┐         ┌──────────────────────┐
   │ 128 KiB regions of records    │         │  key → value bytes    │
   │ [len][compact subtree], dense │         │  (the trie stores     │
   │ 256 B packing + region GC     │         │   only a leaf hash)   │
   └──────────────────────────────┘         └──────────────────────┘
```

Three storage tiers:

### 1. The RAM frontier (`RamNode` / `upper`)
The top of the trie is held in memory as `Branch` (16-way) and `Extension`
(shared-nibble) nodes, each caching its Merkle hash. A branch slot points to
another in-RAM node (`RamChild::Ram`), to a disk-resident subtree
(`RamChild::Disk { ptr, root }`), or — during a RAM build — to an in-RAM leaf
held as its own `Arc<[u8]>` (`RamChild::Mem`). The frontier stays bounded (~0.9
B/key at 1B): large subtrees live on disk behind a single pointer.

### 2. The flat file (`FlatFile` / `store`)
Disk subtrees are compact-encoded `DiskSubtree` records (`[u32 len][payload]`),
**densely packed at 256 B-aligned offsets** (a `DiskPtr { unit, len }` addresses
one). The file is a sequence of **128 KiB regions**; a log-structured allocator
appends records and an inline, self-tuning **garbage collector** evacuates the
emptiest regions to reclaim space, so the file doesn't grow unboundedly under
overwrite/split churn.

### 3. The value store (`rocksdb::DB` / `values`)
The trie stores only a leaf hash — `keccak(key ‖ value)` — so the real value bytes
live in an embedded RocksDB in `<flatfile>.values/`. Inserts buffer values in a RAM
overlay and flush them as one `WriteBatch`; `get_value` checks the overlay first so
reads see the latest write. The store is tuned for write-only bulk load
(`value_db_opts`): a **vector memtable** makes the flush an O(1) append (the sort is
deferred to the background SST flush), avoiding the skiplist memtable's
cache-miss-heavy random inserts.

### Persistence (`persist` / `open`, the `.meta` manifest)
The flat file and value store hold the data, but the *index* tying them together
— the frontier (structure, disk pointers, cached hashes) and the allocator's
high-water mark — lives in RAM. `persist()` checkpoints it: it spills any in-RAM
(`Mem`) leaves to the flat file, flushes the RocksDB memtable to SST, fsyncs the
flat file, then writes the bincode `Manifest` atomically (temp + rename).
`open(path)` reattaches to the existing files without truncating, restoring a
writable trie (cached hashes and all). A crash reopens at the last checkpoint.

---

## How inserts flow

`insert_batch(entries)` is the workhorse (single `insert` is a 1-element batch):

1. **Phase A (route):** compute each leaf hash (`keccak(key‖value)`), buffer the
   value for RocksDB, and walk the RAM frontier to find the disk leaf each key lands
   in, grouping keys per leaf — the hashing and the frontier walks are fanned across
   cores.
2. **Phase B (per-leaf, parallel):** each group reads its leaf record, applies
   its keys (`record_node_insert` — re-hashing only the touched path), and either
   rewrites the leaf or, if it exceeds `max_leaf_bytes`, **promotes** it into more
   frontier structure.
3. **Phase C (install):** splice the new records into the frontier and recompute
   the root once.

**RAM-build mode** (`MPT_RAM_BUILD=1`, used by `build-tree.sh`): new/rewritten
leaves live in RAM as their own `Arc`s with no flat-file I/O or GC. The batch is
partitioned by top nibble and the insert is **fanned across the top branch's 16
disjoint child subtrees** (one thread each, no shared store, no lock); a fresh
tree is bootstrapped into a 16-way branch with a single serial insert so even the
first batch parallelizes. When the process footprint crosses a threshold it
**spills** the leaves to disk and reverts to the disk path. Building huge trees is
fast: a 1B-key tree (16 KiB leaves, 100M-key batches) builds at ~1.4 µs/key —
pure-RAM until the footprint crosses the spill ceiling, then the disk path with
bounded GC (~7 GiB flat growth per 100M-key batch).

**The disk path** is read-bandwidth-bound (see **Tuning**): per-key compute is
hidden under the read I/O and the device's random-read bandwidth is the ceiling.
The main write costs — *append contention* and the *synchronous value write* — are
taken off the read path by a single sequential writer and by overlapping the value
write with the reads.

### Hashing & memoization
Recomputing the whole trie hash per insert is the naive cost; this PoC avoids it.
Records are parsed **lazily** — untouched child subtrees stay `Raw` (zero-copy
slices) with their cached hashes reused — so an insert re-hashes only the
root-to-leaf path it changed, *independent of leaf size*. The RAM frontier caches
each node's hash in a `Cell` and invalidates only the touched path. Net: a
steady-state insert performs ≈ path-length keccak calls, not a count that scales
with subtree size.

---

## Performance & tuning

Measured on a Micron 7500 PRO NVMe SSD, inserting fresh keys into the out-of-RAM
1B-key tree:

- **Raw ingest (GC off, `batch-bench.sh`):** ~6.3 µs/key (10k-key batches).
- **Space-bounded (GC on, `bench-fused.sh`):** ~7.0 µs/key (300k-key batches) — the
  fused opportunistic GC holds the flat file steady (`flat_grow=0`) for ~0.7 µs/key
  over the GC-off path.
- **1B-key build (`build-baseline-1b.sh`):** ~1.4 µs/key (RAM-build until the spill,
  then the bounded-GC disk path).

The disk path is **read-bandwidth-bound**: per-key compute is hidden under the read
I/O and the device's random-read bandwidth (~5 GB/s at these block sizes) is the
ceiling. The knobs below target that — a deep read queue, tight reads, and moving
the value write and append contention off the read path.

### Tuning knobs (environment variables)

| Var | Default | Effect |
|-----|---------|--------|
| `MPT_WORKERS` | `192` | Phase-B read queue depth (each worker issues one blocking `pread`). 192 is the measured sweet spot on NVMe — read throughput keeps rising with concurrency up to the buffered-read IOPS ceiling. |
| `MPT_FOLD` | on | Sort the per-leaf groups by file offset so each worker's reads ascend in place; `=0` reverts to unsorted per-leaf reads. |
| `MPT_FOLD_GAP_KIB` | `0` | Coalesce consecutive leaf reads across on-disk gaps ≤ this (KiB) into one `pread`. Default 0 = read only the touched leaves; raising it trades reading the dead in-between bytes for fewer, larger reads — a win only when touched leaves are densely placed. |
| `MPT_GC_OPP` / `MPT_GC_OPP_UTIL` | off / 0.30 | Opportunistic GC: evacuate only the touched, under-util regions, fused into the foreground read. Sustainable bounded file at lower cost than the global cost-benefit GC. |
| `MPT_ONE_WRITER` | off | Many parallel readers fold leaves → payloads; **one** writer appends them in a single sequential `write_batch`. Removes inter-worker append contention (64 concurrent appends hit ~1.1 GB/s vs one stream's ~3.5). ~1.5×. **Skips inline GC** — for bulk/gc-off use. |
| `MPT_NO_WAL` | off | Write values with the RocksDB WAL disabled; durability via `persist`'s memtable flush (same model as the manifest). |
| `MPT_ASYNC_VALUES` | off | (one-writer path) write the batch's values on a thread concurrent with Phase B, joined per batch — hides the value-write CPU under the I/O-bound reads. Pair with `MPT_NO_WAL`. |
| `MPT_RAM_BUILD` / `MPT_RAM_BUILD_GIB` | off / 85·45 | RAM-build mode and its spill threshold (GiB of process footprint; macOS/Linux defaults). `build-baseline-1b.sh` sets 30. |
| `MAX_LEAF_KIB` | 16 | Leaf-size target (build time); sets `Config`. |
| `MPT_DIRECT_IO` | off | O_DIRECT/F_NOCACHE reads. **Loses** here (bypasses cache hits + readahead). |

> The one-writer / no-WAL / async-values flags are **opt-in**; the default path is
> unchanged. `build-tree.sh` enables RAM-build + no-WAL; `batch-bench.sh` enables
> the full fast path; `grow-tree.sh` defaults to the bounded (GC-on) path.

---

## Repository layout

| Path | What it does |
|------|--------------|
| [`src/lib.rs`](src/lib.rs) | The entire engine (see the component map below). |
| [`scripts/build-tree.sh`](scripts/build-tree.sh) | **Fast-build a tree of any size** → checkpoint (RAM-build + spill). |
| [`scripts/build-baseline-1b.sh`](scripts/build-baseline-1b.sh) | ⭐ **PRIMARY — build the 1B baseline** the fast way (RAM-build, 100M-key batches, spill at 30 GiB) → `<dir>/ckpt.flat`. Single arg: output dir. |
| [`scripts/bench-fused.sh`](scripts/bench-fused.sh) | ⭐ **PRIMARY — benchmark the fused fast path** (one-writer + opportunistic GC + parallel writer + no-WAL + async values) on a COW clone; 25M warmup → 15M measured (300k batches). Single arg: baseline dir. |
| [`scripts/batch-bench.sh`](scripts/batch-bench.sh) | **Benchmark batch inserts** into a tree, on a throwaway COW clone; prints us/key. |
| [`scripts/grow-tree.sh`](scripts/grow-tree.sh) | **Grow a tree** by N more keys and re-checkpoint it in place. |
| [`scripts/run-large-bench.sh`](scripts/run-large-bench.sh) | Build + run `benches/large.rs` (preload + timed inserts/overwrites); documents prereqs. |
| [`scripts/iops-bench.sh`](scripts/iops-bench.sh) | **Characterize the SSD** (via `examples/iops`): read IOPS vs block size / queue depth, and single- vs multi-stream write bandwidth — the device numbers behind the tuning. |
| [`examples/buildpersist.rs`](examples/buildpersist.rs) | Build N keys (RAM-build aware) + persist, no post-phases; per-10M rate + footprint. Driven by `build-tree.sh`. |
| [`examples/foldbench.rs`](examples/foldbench.rs) | Insert N keys into an existing checkpoint in batches; per-batch + overall us/key; `MPT_PERSIST=1` to checkpoint. Driven by `batch-bench.sh` / `grow-tree.sh`. |
| [`examples/profins.rs`](examples/profins.rs) | Profiling harness: per-phase + device-busy breakdown of batch inserts (read/write/value), for tuning. |
| [`examples/reopen.rs`](examples/reopen.rs) | Reopen a checkpoint and exercise reads/`disk_accesses_for_key`. |
| [`examples/iops.rs`](examples/iops.rs) | Device characterization: random/sequential read+write IOPS & bandwidth (O_DIRECT) and mmap reads, across block size / thread count. |
| [`examples/hashaudit.rs`](examples/hashaudit.rs), [`hashcount.rs`](examples/hashcount.rs) | Diagnostics: prove per-insert hashing is minimal (`--features profiling`). |
| [`examples/diskusage.rs`](examples/diskusage.rs), [`sizecheck.rs`](examples/sizecheck.rs), [`batchcheck.rs`](examples/batchcheck.rs) | Diagnostics: index footprint, file/free/RAM sizing, batch-vs-one-by-one parity. |
| [`benches/insert.rs`](benches/insert.rs) | Criterion throughput (random / sequential / shared-prefix). |
| [`benches/profile.rs`](benches/profile.rs) | Wall-clock attribution (`--features profiling`). |
| [`benches/large.rs`](benches/large.rs) | Steady-state preload + inserts/overwrites with per-10M stats. |

### `src/lib.rs` component map

- **`FlatMpt`** — top-level DB. `create` / `open` / `persist` / `flush` /
  `insert` / `insert_batch` / `get_value` / `root`, plus observability helpers
  (`ram_nodes`, `flat_file_len`, `free_bytes`, `disk_accesses_for_key`).
  `process_footprint_bytes()` reports the true committed footprint (counts
  compressed/swapped memory — drives the RAM-build spill).
- **`Config`** — `target_leaf_bytes` / `max_leaf_bytes` / `min_promote_bytes`.
- **`Hash` / `Key`** = `[u8; 32]`; **`DiskPtr` `{ unit, len }`** (256 B units).
- **`FlatFile`** — the flat file + `RegionAlloc` (log-structured 128 KiB-region
  allocator with per-region liveness); `read_payload` / `write_payload` /
  `write_batch` / `free` / region GC (`select_victims` / `evacuate_regions`).
- **`RamNode`** (`Empty`/`Extension`/`Branch`) + **`RamChild`** (`Ram` / `Disk` /
  `Mem`) — the frontier; **`Node`** / **`DiskSubtree`** — a disk subtree's Merkle
  structure (lazily parsed; `Raw` children stay zero-copy).
- Insert internals: `insert_batch` (Phase A/B/C), `process_chunk_coalesced` /
  `process_chunk_fold` (readers), `fold_group`, `record_node_insert`,
  `promote_record_to_ram`, `spill_mem` (RAM-build → disk), `process_opportunistic`
  (fused GC). Hashing: `hash_node` / streaming keccak.
- **`prof`** / **`stats`** — opt-in wall-clock attribution + always-on phase
  counters (compile to no-ops / cheap atomics).

---

## Building & running

```bash
cargo test                                           # unit tests (incl. batch-vs-one-by-one,
                                                     #   RAM-build vs disk-build root parity)
cargo bench --bench insert                           # throughput
cargo bench --bench profile --features profiling     # time attribution
scripts/build-tree.sh /data/tree.flat 100000000      # build a 100M tree
scripts/batch-bench.sh /data/tree.flat               # benchmark inserts into it
```

A C/C++ toolchain + libclang is needed (RocksDB builds from source):
`apt-get install build-essential clang libclang-dev` (Debian) / `xcode-select
--install` (macOS).

---

## Known limitations / non-goals

- **Persistence is checkpoint-based.** The frontier/index is durable only as of
  the last `persist()`; a crash reopens at the previous checkpoint. No WAL for the
  trie index (and `MPT_NO_WAL` extends that model to the value store).
- **The one-writer path skips inline GC** — it's an opt-in bulk/gc-off fast path;
  folding GC into it (so it can be the default) is a follow-up.
- **Write amplification.** Each insert into a disk leaf rewrites the whole compact
  record; dense packing + the sequential writer keep it cheap, but it remains the
  design's central write cost.
- **PoC value model.** Keys are 32 bytes; the trie commits to a leaf hash
  (`keccak(key‖value)`) while the value bytes live separately in RocksDB.
