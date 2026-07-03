# Plan: FlatMpt as reth's primary state backend

**End state:** a reth node where `EthState`/`FlatMpt` serves (a) canonical state-root
computation via reth's `CustomStateRoot` hook and (b) latest-state execution reads via a
`StateProvider`, with MDBX retained for everything else (headers, bodies, receipts,
changesets/history). Plus a defensible A/B benchmark against stock reth's sparse trie.

**Verified integration seam (reth 2.3.0, `/mnt2/reth-src` @ 9384bc53):**
`with_custom_state_root` on the engine payload validator
(`crates/engine/tree/src/tree/payload_validator.rs:269`), type
`Arc<dyn Fn(CustomStateRootInput) -> ProviderResult<(B256, TrieUpdates)> + Send + Sync>`
(`:2374`). `plan_state_root_computation` (`:1757`) checks it FIRST, short-circuiting the
sparse-trie strategy entirely. Input carries `block`, `parent_block`, execution `output`,
and `hashed_state` (`LazyHashedPostState`). => custom node binary, **no reth fork**.
Bonus: `root_elapsed` times whichever strategy runs with the same clock at the same call
site — stock vs custom numbers are apples-to-apples by construction.

**Two shaping decisions:**

1. **FlatMpt tracks reth's *persisted* head, not the tip.** Reth's in-memory canonical
   state overlays the unpersisted window (~2 blocks) and absorbs shallow reorgs.
   Validation-time root = apply (accumulated unpersisted ancestor diffs + this block's
   diff) → read root → revert via inverse diff. Commit only when reth persists.
   Sidesteps the non-canonical-sibling problem without a persistent overlay tree.
2. **Values-in-trie makes execution the prefetch.** Every SSTORE reads the old value
   (refund accounting), so once reads are served from FlatMpt, execution loads exactly
   the records the root fold needs. Until then (shadow / root-only phases), an explicit
   untimed pre-read pass emulates it.

---

## WP0 — Groundwork (1 week, mostly parallel)

- **`DiskPtr` ceiling: deferred (watch item).** `unit: u32` × 256 B = 1 TiB. The earlier
  panic at ~70M accounts came from the pathological batched-load path (28× write-amp,
  661 GiB high-water); RAM-build packs 90M accounts into 14 GiB, so full mainnet dense
  state ≈ 50–100 GiB and steady-state churn with opportunistic GC held file growth at 0.
  ~10× headroom — do not widen now. Add a loud warning when high-water crosses ~700 GiB.
- **Baseline curves:** µs/key vs batch size (2k/5k/10k/20k/300k) on the nested model,
  warm vs cold page cache; `profins` `READ_PARSE`/`SERIALIZE` split — decides WP4
  priorities with data.
- **Start reth tip sync immediately** — operational critical path (weeks). Historical
  work proceeds meanwhile via ExEx backfill.
- Library hygiene as-we-go: per-instance knobs into `Config` (env stays as override).

## WP1 — Deletion (2–3 weeks) · gate: randomized oracle parity

- `LeafOp::Delete` through the `record_node_insert` descent: remove leaf, MPT collapse
  rules (branch w/ one survivor → extension/leaf merge). In-record collapse is local;
  hard cases are **cross-boundary**: collapse pulling a sibling out of a `Disk` record
  (read, re-prefix, rewrite) and frontier collapse incl. demotion of promoted
  `RamChild::Account` storage when it shrinks.
- **Account deletion** (EIP-158/selfdestruct): drop account leaf + entire nested storage
  subtree, free records to GC. **Storage-wipe as a first-class op** — reth `BundleState`
  wipe semantics incl. destroy-then-recreate in one block.
- Slot→zero = delete in the nested storage trie; empty storage → `EMPTY_ROOT`.
- Tests: property-based insert/delete sequences vs `eth::root` oracle; official trietest
  delete vectors; promote→delete→demote round-trips; persist/reopen; GC reclaim.

## WP2 — Batched two-level block updates (1–2 weeks) · gate: block-shaped batches match oracle

- `apply_block(BlockUpdate)` → root. `BlockUpdate` = account set/delete + storage
  set/delete/wipe with reth-`BundleState` intra-batch semantics. Phase A routes
  two-level keys, groups per account; parallelism as today.
- **Return the inverse diff** (old values — Phase B reads them anyway) so rollback is
  free. WP5's validate-then-revert flow depends on this.

## WP3 — Shadow ExEx + mainnet-scale nested load (2–3 weeks, overlaps WP2) · gate: root parity over ≥100k real blocks

- Extend `rethload` to the **nested model** (needs WP2) under RAM-build (mandatory —
  28× write-amp finding), from the `reth-hashexport` TSVs.
- ExEx follower: `BundleState` → `BlockUpdate` (deletes/wipes/code), apply, assert
  root == header, log timing; reverts via inverse diffs. Use **ExEx backfill** during
  the ongoing sync for millions of historical blocks.
- **Capture every per-block diff to a corpus file** — backbone of WP4/WP6 offline replay.
- Doubles as the deletion soak test on real mainnet data.

## WP4 — Optimizations for a fair comparison (3–4 weeks, data-driven, interleaved)

Ranked; each gated on measured improvement over the WP3 corpus:

1. **Hot-record cache** — parsed records in RAM, heat-triggered promotion (reuse
   `promote_record_to_ram` with a touch-count predicate), LFU-bounded ~1–4 GiB,
   **deferred write-back** flushed at `persist()`/eviction. Kills
   pread+parse+serialize+write for the hot set; cuts steady-state write-amp + GC churn.
   Counterpart of reth's preserved sparse trie — without it the comparison is unfair to
   our own design.
2. **Small-batch efficiency** — persistent worker pool, no per-batch spawn/sort
   overhead. Target: 10k-key block within ~2× of the 300k-batch µs/key.
3. **Pre-read pass** — untimed prefetch of touched records before the timed apply; in
   WP5 it hides I/O before the validation-path apply.
4. **Tail work:** parallelize Phase C install/root-fold if profiling shows serial cost;
   multi-buffer keccak only if hashing dominates post-cache.
5. **GC in the one-writer path** — the fused fast path currently skips inline GC;
   steady state must not choose between speed and bounded disk.

## WP5 — reth primary backend (4–8 weeks) · gate: follows tip N days, zero divergence

- **5a. Sync-safety:** coarse `RwLock<EthState>` provider (readers during execution,
  writer for the once-per-block apply). Avoids re-litigating the `Cell` hash caches;
  revisit only if contention measured.
- **5b. Root-only integration (big milestone, smallest step):** custom node binary sets
  `with_custom_state_root`. Closure: assert lineage from FlatMpt persisted head → apply
  accumulated unpersisted diffs + block diff → root → revert → return
  `(root, TrieUpdates::default())`.
- **5c. Cut the trie tables:** stop maintaining HashedAccounts/HashedStorages/
  AccountsTrie/StoragesTrie (measure I/O saved — part of the honest total-cost story).
  Keep PlainState + changesets for reorgs/history/recovery. `eth_getProof` unsupported —
  documented non-goal.
- **5d. Commit-at-persistence:** hook reth's persistence of blocks out of the memory
  window to commit into FlatMpt; `persist()` every ~100 blocks. Crash recovery: replay
  MDBX changesets from FlatMpt checkpoint → reth persisted head.
- **5e. Serve execution reads** from FlatMpt `StateProvider` (latest state; reth's
  in-memory overlays layer above unchanged; historical stays MDBX). First in
  cross-check mode (sample-compare vs PlainState) — a read divergence is a consensus
  bug, so it earns its own soak.

## WP6 — Benchmark (2–3 weeks, overlapping)

**Configs:** stock reth · FlatMpt root-only (5b) · FlatMpt reads-too (5e) · offline
corpus replay for microbenchmarks.

**Headline:** per-block `root_elapsed`, same clock, same blocks — p50/p99/max. Then:
newPayload end-to-end latency; sustained throughput re-executing a fixed 100k-block
range; RSS + page cache under **cgroup RAM ladders (8/16/32/64 GiB)** — where the
bounded-RAM claim is proven or killed; disk I/O bytes + write-amp incl. reth's
trie-table persistence vs FlatMpt flat+GC writes; on-disk size.

**Scenarios:** live tip follow (≥1 week per config); deterministic historical replay;
**adversarial cold-touch blocks** (tens of thousands of cold slots — the
worst-case-bounded claim is the protocol-relevant one); state-growth projection on a
synthetically doubled state.

**Pre-registered fairness rules:** same box + NVMe; warmup excluded; both sides tuned;
reth gets its recommended RAM in the headline (RAM ladder is the second axis, not a
gotcha); reth timeout-fallbacks count as reth latency, error-fallbacks reported
separately; both critical-path and total-work views published.

## Timeline & risks

~3–4.5 months, one person: WP1–3 ≈ 5–7 weeks to continuous real-data correctness,
WP5 ≈ 1–2 months, WP4/WP6 interleaved.

Risks, in order:
1. **BundleState wipe/recreate semantics** — where a silent root divergence would live;
   test viciously.
2. **Reth tip sync wall-clock** — start today.
3. **DiskPtr ceiling** — fix in WP0 before any big load.
4. RwLock contention on execution reads — measure before engineering around.
5. Deep reorgs beyond retained inverse diffs → resync from checkpoint (acceptable, loud).

Deliberate non-goals: trie-index WAL (checkpoint + changeset replay suffices),
historical state in FlatMpt, proof serving.
