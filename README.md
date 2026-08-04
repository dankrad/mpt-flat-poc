# mpt-flat-poc — flat-MPT Ethereum state commitment

An Ethereum-**exact** Merkle Patricia Trie with a *flat* storage layout, built
to serve as a node's state commitment: a small in-RAM trie "frontier" sits on
top of larger subtrees packed into a **single flat file**. Account fields and
storage slots live *in* the leaves — keccak/RLP identical to mainnet (validated
against ethereum/tests and live mainnet headers) — so re-hashing never consults
an external store. Bytecode is the one thing outside the trie: a
content-addressed append log keyed by `code_hash` (never read during hashing).

It is the state store behind two full-node integrations:

- **reth / Ethereum mainnet** — a reth 2.3 fork computes every block's state
  root from this engine inside payload validation (strict header comparison; a
  mismatch rejects the block), with the hashing/merkle stages removed. Runbook,
  reth patch, and recovery tools:
  [dankrad/flatmpt-exex](https://github.com/dankrad/flatmpt-exex).
- **tempo** — the node's sole state store (EVM reads, sparse commitment,
  persistence) for the 1B-account benchmarks.

Current results for both are under **End-to-end results** below.

---

## The state model

One unified secure trie, keyed by `keccak256(address)`:

- **Account leaves** carry `nonce / balance / code_hash` plus a **nested
  per-account storage subtree** (`Node::Account`): the storage trie is packed
  into the same flat file, its records addressed by composite paths (64
  account nibbles ‖ storage nibbles). Hot contracts' storage structure is
  **promoted** into the RAM frontier like any other hot structure, so huge
  contracts don't serialize behind one record.
- **`apply_block(ops) -> (root, inverse_diff)`** — one batched state
  transition per block. Ops are `SetAccount` / `DeleteAccount` /
  `WipeStorage` / `SetStorage` / `DeleteStorage`; the returned inverse ops
  roll the block back exactly (reorg support). Deletion re-folds structure
  canonically (a branch left with one child merges into its survivor, etc.).
- **Cursors + reveal** (`src/cursor.rs`, `src/reveal.rs`) — ordered account
  and per-account storage leaf cursors (the shape of reth's `HashedCursor`
  interfaces) as stateless successor walks, plus direct reveal-node
  extraction for sparse-trie overlays: the path's nodes are copied out with
  their precomputed child hashes — no keccak, no proof RLP round-trip.
- **Bulk bootstrap** — `create_ram_build` + `insert_batch_accounts` stream
  sorted account/storage TSVs into a checkpoint. Full mainnet (400.2M
  accounts + 1.604B slots) builds in ~3 h and reproduces the header's exact
  state root.

## Architecture at a glance

```
                    FlatMpt (src/lib.rs)
   ┌─────────────────────────────────────────────────────────────┐
   │   upper: RamNode            ← in-RAM trie "frontier"        │
   │   ┌───────────┐               (Branch / Extension / Account,│
   │   │  Branch   │                each caching its own hash)   │
   │   └─────┬─────┘                                             │
   │     ┌───┴────────────┬───────────────┐                      │
   │  RamChild::Disk   RamChild::Ram   RamChild::Mem             │
   │   {ptr, root}     (Box<RamNode>)   (Arc<[u8]>, RAM-build    │
   │        │                            + hot-record cache)     │
   └────────┼──────────────────────────────────────────────────-─┘
            │ DiskPtr { unit, len }   (256 B-aligned)
            ▼
   store: FlatFile                        state.rs code store
   ┌───────────────────────────────┐       ┌───────────────────────┐
   │ 128 KiB regions of records    │       │ append log keyed by   │
   │ [len][compact subtree], dense │       │ code_hash (bytecode   │
   │ 256 B packing + region GC     │       │ only; not hashed)     │
   └───────────────────────────────┘       └───────────────────────┘
```

### 1. The RAM frontier (`RamNode` / `upper`)
The top of the trie is held in memory as `Branch` (16-way), `Extension`
(shared-nibble), and `Account` (promoted-account) nodes, each caching its
Merkle hash. A slot points to another in-RAM node (`RamChild::Ram`), to a
disk-resident subtree (`RamChild::Disk { ptr, root }`), or to an in-RAM record
(`RamChild::Mem` — RAM builds and the hot-record cache). The frontier stays
bounded (~0.9 B/key at 1B keys): large subtrees live on disk behind a single
pointer.

### 2. The flat file (`FlatFile` / `store`)
Disk subtrees are compact-encoded `DiskSubtree` records (`[u32 len][payload]`),
**densely packed at 256 B-aligned offsets** (a `DiskPtr { unit, len }`
addresses one). The file is a sequence of **128 KiB regions**; a
log-structured allocator appends records densely, and every record stores its
**full composite frontier path**, so any record is independently relocatable
(asserted by `MPT_GC_ASSERT_PATHS=1`). Freed regions **stage** until the
current apply's read-ahead window closes before they can be reused.

### 3. Persistence (`persist` / `open`, the `.meta` manifest)
The flat file holds the data; the *index* — frontier structure, disk pointers,
cached hashes, allocator high-water — lives in RAM. `persist()` checkpoints
it: spills any in-RAM records, fsyncs the flat file, then writes the bincode
`Manifest` atomically (temp + rename). `open(path)` reattaches without
truncating; a crash reopens at the last checkpoint. The node integrations
store a root-verified `<flat>.height` beside it so a torn checkpoint is
detected rather than followed.

---

## How a block applies

`apply_block(ops)`:

1. **Phase A (route):** walk the RAM frontier to find the disk record each
   key lands in, grouping ops per record; an advisory **pre-read pass**
   (`prefetch_block`, 192-deep) warms the records a pending block will touch.
2. **Phase B (per-record, parallel):** each group reads its record (reads
   fanned across the top branch's 16 disjoint subtrees and a deep pread
   queue), applies its ops re-hashing only the touched paths, and rewrites —
   or **promotes** the record into more frontier structure if it outgrows
   `max_leaf_bytes`. Rewritten-often records stay in RAM (**hot-record
   cache**, Mem-on-rewrite with deferred write-back, budget `MPT_HOT_GIB`).
3. **Phase C (install):** splice the new records into the frontier, recompute
   the root once, and emit the inverse diff.

The disk path is **read-bandwidth-bound**: per-key compute hides under the
read I/O. The write costs — append contention and record rewrites — are kept
off the read path by batched sequential appends.

### Background GC

Overwrite/split churn strands garbage in old regions; GC keeps the file
bounded **off the block critical path**, as a split design:

- **`FlatSnapshot::gc_collect`** — lock-free on a pinned snapshot: picks the
  emptiest regions (utilization below `BG_EVAC_MAX_UTIL`), reads victims in
  parallel, filters to live records.
- **`FlatMpt::gc_install`** — brief pass under the writer that re-verifies
  each candidate against the live frontier before relocating (returns
  `(installed, discarded)`; stale candidates are simply dropped).

The embedders schedule collect into idle windows (tempo: between-block dead
time; reth fork: a 500 ms try-lock loop) with an emergency mode when file
utilization drops too low. Known limit: at sustained flood saturation on a
single NVMe, reclaim (~75 MB/s) trails garbage production (~100–125 MB/s)
while costing ~7% tps — bounding a long-lived follower file remains the open
engineering front (see results below).

---

## End-to-end results

### tempo node, 1B-account random workload (July 2026)

Full-node comparison on the truly-random workload: 1B keccak-signable
accounts, sender AND recipient uniformly random, single token, 30k offered
TPS, identical clean-golden datadir and node args. "Flat" = this engine as
the node's single state store (`TEMPO_NO_STATE_KV=1`, EVM reads + sparse
commitment + persistence all through the flat MPT); "stock" = unmodified
MDBX path. Worst persist = longest Saving->Saved engine-persistence span.

30-minute writer-stress legs, plus a no-commitment ceiling: "stock,
no commitment" runs `--debug.skip-state-root` — no state root computed, no
sparse-trie/proof work spawned, no trie writes; the header carries the
parent's root. It bounds what ANY commitment scheme could achieve on this
box (r160-r163, 2026-07-31; reproduces the r157-r159 numbers within 1%):

| | avg tps | p50 / p99 block | worst persist | IOPS (r + w) | MB/s (r + w) | RAM frontier |
|---|---|---|---|---|---|---|
| stock, no commitment | 14,680 | **0.38 s / 1.6 s** | 244 s | 75.5k + 58.8k | 427 + 327 | — |
| flat, no gc | 11,465 | 2.4 s / 9.6 s | 31 s | 99.6k + **2.3k** | 962 + 475 | 0.73 GiB |
| flat + gc   | 10,791 | 2.7 s / 9.4 s | **34 s** | 97.8k + **2.2k** | 947 + 474 | 0.73 -> 0.75 GiB |
| stock       | 4,687  | 2.1 s / 20.4 s | **847 s** | 126.9k + 13.5k | 565 + 375 | — |

Read against the ceiling: stock's commitment costs it 68% of the
achievable throughput; flat's costs 22-27% — flat delivers 73-78% of the
no-commitment maximum while root-checking every block (flat/sparse
cross-check, zero mismatches; ops dumps replay-verify offline). The
persist column shows a second, commitment-independent stock bottleneck:
even with commitment off, MDBX hashed-state writes stall persistence for
244 s worst-case, while the flat legs (which replace those writes too,
`TEMPO_NO_STATE_KV=1`) stay at ~30 s. RAM frontier = the in-RAM trie
index (measured as the persisted manifest): ~0.78 B/account; stock has no
equivalent resident index (its trie lives in MDBX pages).

IOPS = device ops/s on the working NVMe over each leg's first 200 s (the
sampling protocol behind the earlier tables). The write *shape* is the
story: the flat legs push ~475 MB/s through ~2.2k writes/s (large
sequential appends), the no-commitment leg pushes 327 MB/s through 58.8k
writes/s (small random MDBX pages) — flat writes ~26x fewer, ~30x larger
ops for more bandwidth. Committed stock is read-dominated (126.9k r/s):
trie-node lookups on top of state reads. Its low write-op rate is NOT
commitment being write-cheap — normalized by throughput it writes ~80
KB/tx vs the no-commitment leg's ~22 KB/tx (375 vs 327 MB/s absolute at a
third of the tps); the ops are fewer and larger (~28 KB vs ~5.6 KB)
because the lagging persist pipeline flushes big accumulated batches
whose sorted trie/hashed updates coalesce into contiguous page runs.

Margin note from the re-run: at this load both state pipelines run near
block-production speed, and one attempt of each comparison leg failed
before its clean pass — stock OOM-killed at minute 5 (55 GB anon RSS:
cold-start trie work lagged, persistence backed up, executed blocks piled
in RAM), and flat+gc aborted at minute 28 when a production burst put the
builder 64 blocks ahead of the shadow and overflowed the retained-overlay
window (apply pace was identical to its passing run — 2.30 vs 2.23 s per
~162k-op block). The flat-side fix is the hash-transplant follower (persist
sparse-computed hashes instead of re-deriving them in the apply).

### tempo node, 1B-account cold workload (July 2026)

The original 1B benchmark shape for comparison: **4,000 sender accounts**
(vs ~1B random senders above), recipients uniformly random over a 250M
cold range, block-0 golden datadir, 4-token bloat; same 30-min protocol,
node args, and binary as the random suite (r164-r167, 2026-07-31):

| | avg tps | p50 / p99 block | worst persist | IOPS (r + w) | MB/s (r + w) |
|---|---|---|---|---|---|
| stock, no commitment | 27,708 | **1.0 s / 1.6 s** | 20 s | 41.9k + 46.9k | 253 + 250 |
| flat, no gc | 16,500 | 1.7 s / 5.0 s | 53 s | 49.3k + **1.9k** | 399 + 384 |
| flat + gc   | 16,209 | 1.7 s / 4.0 s | **59 s** | 49.2k + **2.1k** | 406 + 423 |
| stock       | 6,487  | 1.6 s / 9.3 s | **408 s** | 79.6k + 22.8k | 407 + 188 |

With the lighter sender set the execution ceiling nearly doubles (27,708,
near-saturating the 30k offered rate) — and the commitment gap WIDENS:
stock keeps only 23% of the achievable throughput (vs 32% on the random
workload), flat keeps 58-60% and stays 2.5x stock. The gc tax is ~2% here
(vs ~6% random): fewer unique writers per block means less record churn
to reclaim. Same write-shape story as the random suite: flat moves ~420
MB/s through ~2k writes/s of large appends; the no-commitment ceiling
does 46.9k/s of small MDBX page writes for 250 MB/s.

One caveat on the flat+gc leg: the cold golden (`golden-1b.flat`, built
July 5) predates the composite-prefix fix, so its records carry old
storage-local paths the GC collect walk cannot see. The leg runs with
`MPT_GC_ASSERT_PATHS` off — the GC pins such regions instead of
relocating them (the designed graceful path), slightly understating
reclaim vs a freshly built tree. Root cross-checks passed on every block
in both flat legs.

### Ethereum mainnet: 10,000-block replay, flat vs stock (August 2026)

The direct head-to-head on a reth 2.3 fork (mainnet at ~block 25.65M,
~400M accounts / 1.6B slots, single NVMe, 62 GB box): the same node
synced the same 10,000 mainnet blocks (25,642,905-25,652,904) twice from
the same starting state — once with flat as the commitment backend
(`RETH_FLATMPT_ROOT=1`: state roots computed from the flat MPT with a
strict header comparison, hashing/merkle stages disabled), once as
vanilla stock (hashed-state mode: execution maintains `HashedAccounts`/
`HashedStorages`; incremental `MerkleExecute` is the commitment work on
top). Fixed-range `--debug.tip/--debug.max-block` sync, caches dropped
before each leg, every leg root-verified (flat: each batch against its
tip header; stock: `MerkleExecute` against the target header):

| | total wall | execution | commitment | history indexes |
|---|---|---|---|---|
| flat  | **871 s** | 668 s | **60.5 s** (4.73M ops, 6.1 ms/block) | 135 s |
| stock | 939 s | 420 s | 489 s (48.9 ms/block incremental) | 25 s |

Flat syncs the range 7% faster end to end — while doing MORE work (its
leg also re-downloaded bodies + senders, 26 s, which stock's leg
skipped). The commitment itself is 8.1x cheaper (60.5 vs 489 s), and the
6.1 ms/block batch figure matches the ~7 ms/block the engine sustains
live at the tip. The interesting asymmetry: flat's execution and index
stages run SLOWER (668 vs 420 s, 135 vs 25 s) because the flat applies
overlap them and compete for the same NVMe — flat spends its savings
buying back contention, and still nets ahead. (A second-pass MDBX
page-reuse effect may also flatter stock's execution: its leg re-executed
into pages the flat leg had already allocated.)

The RAM frontier for the full mainnet state (400M accounts + 1.6B storage
slots) is **1.56 GiB at build, 1.73 GiB on the live follower** — ~0.8 B
per key, linear in key count (measured as the persisted manifest).

---

## Bootstrapping a mainnet checkpoint

```bash
# 1. Export reth's HashedAccounts/HashedStorages to TSV (paged; see script header)
scripts/reth-export.sh <reth-datadir> <out-dir>

# 2. Sort both TSVs by key (GNU sort), then build + root-verify the checkpoint
MPT_RAM_BUILD=1 MPT_RAM_BUILD_GIB=45 \
  cargo run --release --example rethload_nested -- \
  <out-dir>/accounts.tsv <out-dir>/storages.tsv /data/tip-nested.flat <stateRoot>
```

`rethload_nested` merge-joins the sorted streams, batches whole accounts
(fields + all slots) through `insert_batch_accounts`, verifies the root
against the block's real `stateRoot`, and persists a reopenable checkpoint.
To run that checkpoint under a live node, follow the
[flatmpt-exex](https://github.com/dankrad/flatmpt-exex) runbook.

---

## Tuning knobs (environment variables)

| Var | Default | Effect |
|-----|---------|--------|
| `MPT_WORKERS` | `192` | Phase-B read queue depth (each worker issues one blocking `pread`) — the measured NVMe sweet spot. |
| `MPT_FOLD` / `MPT_FOLD_GAP_KIB` | on / `0` | Sort per-record reads by file offset; optionally coalesce reads across gaps ≤ N KiB into one `pread`. |
| `MPT_PREFETCH` | on | `=0` disables the `apply_block` pre-read pass. |
| `MPT_HOT_RECORDS` / `MPT_HOT_GIB` | on / budget | Mem-on-rewrite hot-record cache with deferred write-back; `MPT_HOT_RECORDS=0` disables (records rewrite to disk every time). |
| `MPT_BATCHED_WRITES` | on | One sequential append batch per apply; `=0` writes per record (A/B comparison). |
| `MPT_GC_DISABLE` | off | Kill switch for all inline GC (A/B comparison). |
| `MPT_GC_OPP` / `MPT_GC_OPP_UTIL` | off / 0.30 | Opportunistic GC fused into the foreground read: evacuate touched, under-utilized regions. |
| `MPT_GC_R_MAX` / `MPT_GC_GAIN` | — | Reclaim controller: max regions per call / ramp rate (for high-write-amp bulk loads). |
| `MPT_GC_ASSERT_PATHS` / `MPT_GC_VERIFY` / `MPT_GC_LOG` | off | GC diagnostics: assert every collected record's stored path resolves in the live frontier; verify/log installs. |
| `MPT_RAM_BUILD` / `MPT_RAM_BUILD_GIB` | off / 85·45 | RAM-build mode for bulk bootstrap and its spill threshold (GiB of process footprint). |
| `MAX_LEAF_KIB` | 16 | Record-size target at build time (loader examples). |
| `MPT_DIRECT_IO` | off | O_DIRECT reads. **Loses** here (bypasses cache hits + readahead). |

---

## Repository layout

| Path | What it does |
|------|--------------|
| [`src/lib.rs`](src/lib.rs) | The engine: frontier, flat file, `apply_block`, GC (see the component map). |
| [`src/eth.rs`](src/eth.rs) | Mainnet-exact keccak/RLP node hashing (validated against ethereum/tests). |
| [`src/state.rs`](src/state.rs) | Typed account API over the trie + the content-addressed code store. |
| [`src/cursor.rs`](src/cursor.rs) | Ordered account/storage leaf cursors + branch-node cursors (reth `TrieCursor`/`HashedCursor` backing). |
| [`src/reveal.rs`](src/reveal.rs) | Direct reveal-node extraction for sparse-trie overlays (no keccak, no proof RLP). |
| [`scripts/reth-export.sh`](scripts/reth-export.sh) | Export reth's hashed tables to TSV for `rethload_nested`. |
| [`examples/rethload_nested.rs`](examples/rethload_nested.rs) | Build + root-verify a mainnet checkpoint from the TSVs. |
| [`examples/replay.rs`](examples/replay.rs), [`replaymod.rs`](examples/replaymod.rs) | Replay a recorded diff corpus block-by-block, verifying roots offline. |
| [`examples/blockbench.rs`](examples/blockbench.rs), [`applyprofile.rs`](examples/applyprofile.rs), [`applysweep.rs`](examples/applysweep.rs) | `apply_block` throughput/IO-split benchmarks on real corpora. |
| [`examples/ethbench.rs`](examples/ethbench.rs), [`ethfused.rs`](examples/ethfused.rs), [`hotcontracts.rs`](examples/hotcontracts.rs) | Synthetic EVM-shaped apply workloads (1B-scale baselines). |
| [`examples/readbench.rs`](examples/readbench.rs) | Point-read latency/throughput on a checkpoint. |
| [`examples/gcprobe.rs`](examples/gcprobe.rs), [`gcdrain.rs`](examples/gcdrain.rs) | GC oracle (region/liveness audit) and offline drain-to-target-utilization. |
| `examples/probe*.rs`, [`rootaudit.rs`](examples/rootaudit.rs), [`flatdump.rs`](examples/flatdump.rs), [`tsvdiff.rs`](examples/tsvdiff.rs) | Forensics: single-key/slot probes, root audits, record dumps, TSV diffing. |
| [`examples/corpusdump.rs`](examples/corpusdump.rs), [`applydump.rs`](examples/applydump.rs), [`opdrop.rs`](examples/opdrop.rs), [`opsbisect.rs`](examples/opsbisect.rs), [`reprospill.rs`](examples/reprospill.rs), [`keccaklines.rs`](examples/keccaklines.rs) | Corpus forensics: dump/filter recorded op streams, bisect a diverging block's ops, spill repro, address hashing. |

### `src/lib.rs` component map

- **`FlatMpt`** — top-level DB: `create` / `create_ram_build` / `open` /
  `persist` / `root`, the block API (`apply_block` / `prefetch_block`), bulk
  ingest (`insert_batch_accounts`), GC (`snapshot().gc_collect` →
  `gc_install`), cursors (`account_cursor` / `storage_cursor`), plus
  observability (`ram_nodes`, `flat_file_len`, `free_bytes`,
  `process_footprint_bytes`).
- **`StateOp`** — the per-block op vocabulary (`SetAccount`, `DeleteAccount`,
  `WipeStorage`, `SetStorage`, `DeleteStorage`).
- **`Config`** — `target_leaf_bytes` / `max_leaf_bytes` / `min_promote_bytes`.
- **`FlatFile`** — flat file + `RegionAlloc` (log-structured 128 KiB-region
  allocator, per-region liveness, staged frees).
- **`RamNode`** (`Empty`/`Extension`/`Branch`/`Account`) + **`RamChild`**
  (`Ram`/`Disk`/`Mem`) — the frontier; **`Node`** / **`DiskSubtree`** — a disk
  subtree's Merkle structure (lazily parsed; untouched children stay zero-copy
  with cached hashes).
- **`prof`** / **`stats`** — opt-in wall-clock attribution + cheap always-on
  phase counters.

---

## Building & running

```bash
cargo test          # unit + equivalence tests (eth hashing vectors, apply/inverse
                    #   round-trips, GC relocation, cursor order, RAM/disk parity)
```

Pure Rust — no external toolchain dependencies.

---

## Known limitations / non-goals

- **Persistence is checkpoint-based.** The frontier/index is durable only as
  of the last `persist()`; a crash reopens at the previous checkpoint (the
  node integrations pair this with a root-verified height file and
  re-backfill the gap).
- **Write amplification.** Each op into a disk record rewrites the whole
  compact record; dense packing, batched appends, and the hot-record cache
  keep it cheap, but it remains the design's central write cost — and the
  source of the GC's work.
- **GC vs sustained flood.** At single-NVMe write saturation, reclaim can
  trail garbage production; bounding a long-lived follower file without
  touching the block critical path is the active engineering front.
- **Single-process, single-writer.** One writer owns the trie; concurrent
  access is snapshot-based (GC collect, prefetch, cursors).
