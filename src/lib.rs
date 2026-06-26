use anyhow::{Result, anyhow, bail};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{File, OpenOptions},
    io::Write,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::Mutex,
    sync::atomic::{AtomicU64, Ordering},
};

/// Number of buffered values before the overlay is flushed to RocksDB as one
/// `WriteBatch`, amortizing per-`put` overhead.
const VALUE_BATCH: usize = 256;

/// Flat-file allocation granularity. Records are page-aligned and occupy a whole
/// number of pages, so each leaf read/write is a single positioned I/O over a
/// contiguous page-aligned extent, and the free list only ever deals in whole
/// pages (which collapses the size distribution and curbs fragmentation).
const PAGE: u64 = 4096;

pub type Hash = [u8; 32];
pub type Key = [u8; 32];

/// Debug instrumentation for the split/write path (cheap relaxed atomics).
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static WRITES: AtomicU64 = AtomicU64::new(0);
    pub static SPLITS: AtomicU64 = AtomicU64::new(0);
    pub static MAX_RECORD: AtomicU64 = AtomicU64::new(0);
    pub static MIN_SPLIT_TRIGGER: AtomicU64 = AtomicU64::new(u64::MAX);
    pub static MAX_SPLIT_TRIGGER: AtomicU64 = AtomicU64::new(0);
    /// Histogram of written-record sizes, bucketed by page count (index 0..=16).
    pub static PAGE_HIST: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
    /// Leaves emitted by `split_subtree` (the children a split produces) and
    /// their total bytes — sample the delta between milestones to see the
    /// average size of *freshly split-created* leaves over an interval.
    pub static SPLIT_LEAVES: AtomicU64 = AtomicU64::new(0);
    pub static SPLIT_LEAF_BYTES: AtomicU64 = AtomicU64::new(0);

    /// Cumulative wall-time (ns) in each `insert_batch` phase, plus the number of
    /// batches — sample the delta between milestones to see where batch time goes.
    /// Phase A = dedup + value-hash + flush + routing, B = parallel per-record
    /// work, C = frontier install + fresh keys + root recompute + flush.
    pub static PHASE_A_NS: AtomicU64 = AtomicU64::new(0);
    pub static PHASE_B_NS: AtomicU64 = AtomicU64::new(0);
    pub static PHASE_C_NS: AtomicU64 = AtomicU64::new(0);
    pub static BATCHES: AtomicU64 = AtomicU64::new(0);

    pub fn on_batch(a_ns: u64, b_ns: u64, c_ns: u64) {
        PHASE_A_NS.fetch_add(a_ns, Relaxed);
        PHASE_B_NS.fetch_add(b_ns, Relaxed);
        PHASE_C_NS.fetch_add(c_ns, Relaxed);
        BATCHES.fetch_add(1, Relaxed);
    }

    /// Phase-B sub-breakdown, summed across all worker threads (so the total
    /// exceeds Phase-B wall time by the parallel speedup — only the ratios are
    /// meaningful): READ = fetching the record, REBUILD = applying keys
    /// (`record_node_insert`: keccak + structure), FINAL = migrate + split/promote
    /// + serialize + write.
    pub static B_READ_NS: AtomicU64 = AtomicU64::new(0);
    pub static B_REBUILD_NS: AtomicU64 = AtomicU64::new(0);
    pub static B_FINAL_NS: AtomicU64 = AtomicU64::new(0);

    pub fn on_group(read_ns: u64, rebuild_ns: u64, final_ns: u64) {
        B_READ_NS.fetch_add(read_ns, Relaxed);
        B_REBUILD_NS.fetch_add(rebuild_ns, Relaxed);
        B_FINAL_NS.fetch_add(final_ns, Relaxed);
    }

    /// Within the write path, separate the time spent acquiring+holding the
    /// free-list lock (`alloc` + `free`, where threads contend) from the actual
    /// positioned write (`pwrite`, which is lock-free). Summed across threads.
    pub static W_LOCK_NS: AtomicU64 = AtomicU64::new(0);
    pub static W_PWRITE_NS: AtomicU64 = AtomicU64::new(0);

    pub fn on_alloc_lock(ns: u64) {
        W_LOCK_NS.fetch_add(ns, Relaxed);
    }
    pub fn on_pwrite(ns: u64) {
        W_PWRITE_NS.fetch_add(ns, Relaxed);
    }

    pub fn on_write(total: usize) {
        WRITES.fetch_add(1, Relaxed);
        MAX_RECORD.fetch_max(total as u64, Relaxed);
        let pages = total.div_ceil(super::PAGE as usize).min(16);
        PAGE_HIST[pages].fetch_add(1, Relaxed);
    }

    pub fn on_split(trigger: usize) {
        SPLITS.fetch_add(1, Relaxed);
        MIN_SPLIT_TRIGGER.fetch_min(trigger as u64, Relaxed);
        MAX_SPLIT_TRIGGER.fetch_max(trigger as u64, Relaxed);
    }

    /// Record one leaf produced by a split, with its record size in bytes.
    pub fn on_split_leaf(bytes: usize) {
        SPLIT_LEAVES.fetch_add(1, Relaxed);
        SPLIT_LEAF_BYTES.fetch_add(bytes as u64, Relaxed);
    }

    pub fn reset() {
        WRITES.store(0, Relaxed);
        SPLITS.store(0, Relaxed);
        MAX_RECORD.store(0, Relaxed);
        MIN_SPLIT_TRIGGER.store(u64::MAX, Relaxed);
        MAX_SPLIT_TRIGGER.store(0, Relaxed);
        SPLIT_LEAVES.store(0, Relaxed);
        SPLIT_LEAF_BYTES.store(0, Relaxed);
        PHASE_A_NS.store(0, Relaxed);
        PHASE_B_NS.store(0, Relaxed);
        PHASE_C_NS.store(0, Relaxed);
        BATCHES.store(0, Relaxed);
        B_READ_NS.store(0, Relaxed);
        B_REBUILD_NS.store(0, Relaxed);
        B_FINAL_NS.store(0, Relaxed);
        W_LOCK_NS.store(0, Relaxed);
        W_PWRITE_NS.store(0, Relaxed);
        for a in &PAGE_HIST {
            a.store(0, Relaxed);
        }
    }

    pub fn dump() -> String {
        let min_trig = MIN_SPLIT_TRIGGER.load(Relaxed);
        let mut s = format!(
            "writes={} splits={} max_record={}B split_trigger=[{}..{}]B  pages:",
            WRITES.load(Relaxed),
            SPLITS.load(Relaxed),
            MAX_RECORD.load(Relaxed),
            if min_trig == u64::MAX { 0 } else { min_trig },
            MAX_SPLIT_TRIGGER.load(Relaxed),
        );
        for (p, a) in PAGE_HIST.iter().enumerate() {
            let c = a.load(Relaxed);
            if c > 0 {
                s += &format!(" {p}p={c}");
            }
        }
        s
    }
}

/// Opt-in wall-clock profiler.
///
/// Each [`Cat`] is a *leaf* primitive — none of the instrumented regions nests
/// inside another — so the buckets are mutually exclusive and their sum is the
/// time spent in hashing / serialization / IO. Whatever the timed workload's
/// wall clock has left over is "trie/CPU": nibble math, tree restructuring,
/// buffer assembly, free-list bookkeeping (and the profiler's own overhead).
///
/// Enabled by the `profiling` cargo feature; otherwise every hook is a no-op
/// zero-sized guard and compiles away entirely.
pub mod prof {
    /// Human-readable labels, indexed by `Cat as usize`.
    pub const CATS: [&str; 8] = [
        "keccak (hashing)",
        "subtree serialize",
        "subtree deserialize",
        "flat-file read (syscall)",
        "flat-file write (syscall)",
        "flat-file flush",
        "rocksdb put (value store)",
        "rocksdb get (value store)",
    ];

    #[derive(Clone, Copy)]
    pub enum Cat {
        Keccak = 0,
        Serialize = 1,
        Deserialize = 2,
        FileRead = 3,
        FileWrite = 4,
        Flush = 5,
        ValuePut = 6,
        ValueGet = 7,
    }

    #[cfg(feature = "profiling")]
    mod imp {
        use super::Cat;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Instant;

        pub const ENABLED: bool = true;

        static NANOS: [AtomicU64; 8] = [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];
        static COUNT: [AtomicU64; 8] = [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];

        pub struct Guard {
            cat: usize,
            start: Instant,
        }

        impl Drop for Guard {
            fn drop(&mut self) {
                let elapsed = self.start.elapsed().as_nanos() as u64;
                NANOS[self.cat].fetch_add(elapsed, Ordering::Relaxed);
                COUNT[self.cat].fetch_add(1, Ordering::Relaxed);
            }
        }

        /// Start timing `cat`; the returned guard records on drop.
        pub fn scope(cat: Cat) -> Guard {
            Guard {
                cat: cat as usize,
                start: Instant::now(),
            }
        }

        pub fn reset() {
            for i in 0..8 {
                NANOS[i].store(0, Ordering::Relaxed);
                COUNT[i].store(0, Ordering::Relaxed);
            }
        }

        /// Per-category `(total_nanos, call_count)`.
        pub fn snapshot() -> [(u64, u64); 8] {
            std::array::from_fn(|i| {
                (
                    NANOS[i].load(Ordering::Relaxed),
                    COUNT[i].load(Ordering::Relaxed),
                )
            })
        }

        use std::cell::RefCell;
        thread_local! {
            // `Some` while an audit window is open; each keccak output is logged.
            static AUDIT: RefCell<Option<Vec<[u8; 32]>>> = const { RefCell::new(None) };
        }

        /// Begin recording keccak outputs (clears any prior window).
        pub fn audit_start() {
            AUDIT.with(|a| *a.borrow_mut() = Some(Vec::new()));
        }

        /// End recording and return the outputs produced since `audit_start`.
        pub fn audit_take() -> Vec<[u8; 32]> {
            AUDIT.with(|a| a.borrow_mut().take().unwrap_or_default())
        }

        #[inline]
        pub fn record(output: [u8; 32]) {
            AUDIT.with(|a| {
                if let Some(log) = a.borrow_mut().as_mut() {
                    log.push(output);
                }
            });
        }
    }

    #[cfg(not(feature = "profiling"))]
    mod imp {
        use super::Cat;

        pub const ENABLED: bool = false;

        pub struct Guard;

        #[inline(always)]
        pub fn scope(_: Cat) -> Guard {
            Guard
        }
        pub fn reset() {}
        pub fn snapshot() -> [(u64, u64); 8] {
            [(0, 0); 8]
        }
        #[inline(always)]
        pub fn record(_: [u8; 32]) {}
        pub fn audit_start() {}
        pub fn audit_take() -> Vec<[u8; 32]> {
            Vec::new()
        }
    }

    pub use imp::{ENABLED, Guard, audit_start, audit_take, record, reset, scope, snapshot};
}

/// A `u32` first-page index (16 TiB of addressable file at 4 KiB pages) plus the
/// record's exact byte length — eight bytes instead of the twelve a `{u64, u32}`
/// byte offset would take. It appears per frontier leaf (RAM) and per overflow
/// child (disk), so the four bytes matter at scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskPtr {
    pub page: u32,
    pub len: u32,
}

impl DiskPtr {
    /// Byte offset of the record in the flat file.
    fn offset(&self) -> u64 {
        self.page as u64 * PAGE
    }
    /// Whole pages the record occupies.
    fn pages(&self) -> u32 {
        pages_for(self.len)
    }
}

/// Tracks reclaimed regions of the flat file so new records can be placed into
/// holes left by rewritten/split subtrees instead of always extending the file.
///
/// The unit is **pages**, not bytes: `by_offset` maps a free region's first page
/// index to its length in pages. (The structure itself is unit-agnostic; the
/// caller — [`FlatFile`] — works exclusively in pages.) Quantizing to pages keeps
/// the size distribution tiny, so freed holes match new requests far more often
/// and fragmentation stays low.
///
/// Regions are kept non-overlapping and coalesced (adjacent free regions are
/// merged on `free`). Two indexes are maintained in lock-step: `by_offset` for
/// coalescing with neighbours, and `by_size` (keyed by `(len, offset)`) so that
/// best-fit allocation is O(log n) rather than a linear scan.
///
/// Only `by_offset` is serialized; `by_size` is rebuilt via [`reindex`] on load.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct FreeList {
    /// first page index -> length in pages of each free region.
    by_offset: BTreeMap<u64, u32>,
    /// (length, first page) of each free region, for size-ordered best-fit lookup.
    #[serde(skip)]
    by_size: BTreeSet<(u32, u64)>,
}

impl FreeList {
    fn insert_region(&mut self, offset: u64, len: u32) {
        self.by_offset.insert(offset, len);
        self.by_size.insert((len, offset));
    }

    fn remove_region(&mut self, offset: u64, len: u32) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(len, offset));
    }

    /// Reserve `need` bytes from a free region if one is large enough.
    /// Returns the offset of the allocation, leaving any remainder free.
    fn alloc(&mut self, need: u32) -> Option<u64> {
        // Best fit: smallest region with len >= need, in O(log n).
        let (len, offset) = self.by_size.range((need, 0)..).next().copied()?;
        self.remove_region(offset, len);
        let remainder = len - need;
        if remainder > 0 {
            self.insert_region(offset + need as u64, remainder);
        }
        Some(offset)
    }

    /// Mark `[offset, offset + len)` as free, coalescing with neighbours.
    fn free(&mut self, offset: u64, len: u32) {
        let mut start = offset;
        let mut size = len as u64;

        // Merge with the region immediately preceding this one.
        let pred = self
            .by_offset
            .range(..start)
            .next_back()
            .map(|(&off, &len)| (off, len));
        if let Some((prev_off, prev_len)) = pred {
            if prev_off + prev_len as u64 == start {
                start = prev_off;
                size += prev_len as u64;
                self.remove_region(prev_off, prev_len);
            }
        }
        // Merge with the region immediately following this one.
        if let Some(next_len) = self.by_offset.get(&(start + size)).copied() {
            self.remove_region(start + size, next_len);
            size += next_len as u64;
        }

        self.insert_region(start, size as u32);
    }

    fn total(&self) -> u64 {
        self.by_offset.values().map(|&len| len as u64).sum()
    }

    fn region_count(&self) -> usize {
        self.by_offset.len()
    }

    /// Rebuild the size index from `by_offset` (after deserialization).
    fn reindex(&mut self) {
        self.by_size = self.by_offset.iter().map(|(&off, &len)| (len, off)).collect();
    }
}

/// Append-mostly flat file of compact-serialized [`DiskSubtree`] records, with a
/// [`FreeList`] so that space freed by rewrites can be reused.
///
/// Thread-safe: positioned `pread`/`pwrite` go through a shared `&File`, and the
/// only mutable state — the free list and the high-water mark — is behind a
/// `Mutex`/atomic. That lets `insert_batch` run many independent record updates
/// concurrently (each touches a disjoint subtree; only the brief allocation step
/// is serialized).
#[derive(Debug)]
struct FlatFile {
    file: File,
    free: Mutex<FreeList>,
    /// High-water mark in pages: the next page index where fresh appends land.
    /// Always ≥ every live/free region, so `fetch_add` hands out fresh,
    /// non-overlapping extents without touching the free-list lock.
    end_page: AtomicU64,
}

/// Pages needed to hold a record of `record_bytes` (length prefix + payload).
fn pages_for(record_bytes: u32) -> u32 {
    record_bytes.div_ceil(PAGE as u32)
}

impl FlatFile {
    fn new(file: File) -> Self {
        Self {
            file,
            free: Mutex::new(FreeList::default()),
            end_page: AtomicU64::new(0),
        }
    }

    /// Store an already-encoded subtree payload in a page-aligned record,
    /// preferring a reclaimed free region over extending the file. The payload is
    /// written verbatim (no length prefix — its exact size lives in the returned
    /// [`DiskPtr`]) in a single positioned `pwrite`, occupying `ceil(len/PAGE)`
    /// whole pages. Reads fetch exactly `len` bytes, so the unused tail of the
    /// last page is never read and needn't be zeroed. Safe to call concurrently:
    /// allocation holds the free-list lock only briefly; the `pwrite` is lock-free.
    fn write_payload(&self, payload: &[u8]) -> Result<DiskPtr> {
        let total = payload.len() as u32;
        stats::on_write(total as usize);
        let pages = pages_for(total);
        // Reuse a freed region if one fits; otherwise extend the file. Both yield
        // a page range disjoint from every other in-flight allocation. Time the
        // lock-held alloc separately from the (lock-free) pwrite to expose
        // free-list contention between the parallel batch workers.
        let lt = std::time::Instant::now();
        let reused = self.free.lock().unwrap().alloc(pages);
        let page = reused.unwrap_or_else(|| self.end_page.fetch_add(pages as u64, Ordering::SeqCst));
        stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
        if page > u32::MAX as u64 {
            bail!("flat file exceeds the 16 TiB DiskPtr addressing limit");
        }
        let page = page as u32;

        let _g = prof::scope(prof::Cat::FileWrite);
        let wt = std::time::Instant::now();
        (&self.file).write_all_at(payload, page as u64 * PAGE)?;
        stats::on_pwrite(wt.elapsed().as_nanos() as u64);
        Ok(DiskPtr { page, len: total })
    }

    fn read(&self, ptr: DiskPtr) -> Result<DiskSubtree> {
        read_record(&self.file, ptr)
    }

    fn free(&self, ptr: DiskPtr) {
        let lt = std::time::Instant::now();
        self.free
            .lock()
            .unwrap()
            .free(ptr.page as u64, ptr.pages());
        stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
    }

    fn end_page(&self) -> u64 {
        self.end_page.load(Ordering::SeqCst)
    }

    fn flush(&self) -> Result<()> {
        let _g = prof::scope(prof::Cat::Flush);
        Ok((&self.file).flush()?)
    }

    /// Flush and fsync the flat file to disk (used before a manifest checkpoint
    /// so the manifest never references data that hasn't reached storage).
    fn sync(&self) -> Result<()> {
        (&self.file).flush()?;
        self.file.sync_all()?;
        Ok(())
    }
}

/// Read and decode one record by pointer, using only `&File` (a positioned
/// `pread` + decode) so it can run on any thread. `read_exact_at` is a `pread`,
/// which is safe to call concurrently on the same file from multiple threads.
fn read_record(file: &File, ptr: DiskPtr) -> Result<DiskSubtree> {
    let mut record = vec![0u8; ptr.len as usize];
    {
        let _g = prof::scope(prof::Cat::FileRead);
        file.read_exact_at(&mut record, ptr.offset())?;
    }
    let _g = prof::scope(prof::Cat::Deserialize);
    deserialize_subtree(&record)
}




#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub target_leaf_bytes: usize,
    pub max_leaf_bytes: usize,
    pub min_promote_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            target_leaf_bytes: 4 * 1024,
            max_leaf_bytes: 8 * 1024,
            min_promote_bytes: 2 * 1024,
        }
    }
}

// Each non-trivial node caches its own Merkle hash, computed once at
// construction and persisted to disk. This lets a rewrite recompute only the
// hashes on the path it actually changed (see `node_insert`), instead of
// re-hashing the whole subtree. All keys are full 64-nibble paths, so leaves
// only ever sit at depth 64 and branches never carry a value.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node {
    Empty,
    Leaf {
        /// Remaining key nibbles from this leaf's position to depth 64 (a merged
        /// extension+leaf). The full key is `position_prefix ++ path`, recovered
        /// from the tree position, so the key isn't stored; `value_hash` is folded
        /// into the position-independent `hash`, so it isn't stored either.
        path: Vec<u8>,
        hash: Hash,
    },
    Extension {
        path: Vec<u8>,
        child: Box<Node>,
        hash: Hash,
    },
    Branch {
        children: [Option<Box<Node>>; 16],
        hash: Hash,
    },
    /// A child subtree that lives in its *own* flat-file record rather than
    /// inline in this one (the "(3) overflow" of the paged-node design). `root`
    /// is that subtree's Merkle hash — identical to what an inline node's `hash`
    /// would be — so a branch hashes the same whether a child is inline or
    /// overflowed. The bytes at `ptr` are themselves a [`DiskSubtree`] record,
    /// which may recursively contain further `Overflow` children.
    Overflow {
        ptr: DiskPtr,
        root: Hash,
    },
}

// A subtree is fully described by its `node`; the flat list of entries it holds
// is derived on demand (`node_entries`) when a split needs to regroup them.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskSubtree {
    prefix: Vec<u8>,
    node: Node,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RamChild {
    Ram(Box<RamNode>),
    // `root` is the subtree's Merkle hash; the on-disk record size is recoverable
    // from `ptr.len`, so it isn't stored here.
    Disk { ptr: DiskPtr, root: Hash },
}

// RAM-frontier nodes cache their hash in an interior-mutable `Cell` so that
// `hash_ram`/`root` can memoize. An insert invalidates only the caches along
// the path it touches (`invalidate_ram`), so recomputing the root re-hashes
// just that path — every other node returns its cached value, and disk children
// contribute their already-cached `root`.
//
// Children are an inline 16-slot array: frontier branches are dense in practice
// (a near-complete 16-ary tree over the disk leaves), so a sparse representation
// would only add per-branch heap allocations without shrinking anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum RamNode {
    Empty,
    Extension {
        path: Vec<u8>,
        child: Box<RamNode>,
        hash: Cell<Option<Hash>>,
    },
    // No `value`: keys are full 64-nibble paths, so none ever terminates at a
    // (necessarily shallower) frontier branch.
    Branch {
        children: [Option<RamChild>; 16],
        hash: Cell<Option<Hash>>,
    },
}

impl Default for RamNode {
    fn default() -> Self {
        Self::Empty
    }
}

/// Breakdown of the in-RAM index footprint (see [`FlatMpt::ram_report`]).
#[derive(Debug, Clone, Copy)]
pub struct RamReport {
    /// Branch/Extension nodes in the trie frontier.
    pub frontier_nodes: usize,
    /// Heap bytes for those frontier nodes (accurate: allocation size + paths).
    pub frontier_bytes: usize,
    /// Free regions tracked by the flat-file allocator.
    pub free_regions: usize,
    /// Stored-data bytes for the free list (excludes BTree container overhead).
    pub free_list_bytes: usize,
    /// Values buffered but not yet flushed to RocksDB.
    pub overlay_entries: usize,
    /// Heap bytes held by the value overlay (keys + value buffers).
    pub overlay_bytes: usize,
}

impl RamReport {
    /// Estimated total in-RAM index bytes.
    pub fn total_bytes(&self) -> usize {
        self.frontier_bytes + self.free_list_bytes + self.overlay_bytes
    }
}

#[derive(Debug)]
pub struct FlatMpt {
    cfg: Config,
    store: FlatFile,
    upper: RamNode,
    /// Disk-backed key -> value store. Holds the actual values; the trie only
    /// ever deals in `value_hash`.
    values: DB,
    /// Buffer of values not yet flushed to `values`. Flushed in one batch every
    /// `VALUE_BATCH` inserts (and on `persist`); read by `get_value` first so
    /// reads always observe the latest write.
    overlay: HashMap<Key, Vec<u8>>,
    /// Path of the flat file; the value store and manifest are derived from it.
    path: PathBuf,
}

/// On-disk checkpoint of everything that otherwise lives only in RAM: the trie
/// frontier, the flat-file free list, and the high-water mark. Together with the
/// flat file and the value store this is enough to fully reconstruct a `FlatMpt`.
#[derive(Serialize)]
struct ManifestRef<'a> {
    cfg: &'a Config,
    upper: &'a RamNode,
    free: &'a FreeList,
    end_page: u64,
}

#[derive(Deserialize)]
struct Manifest {
    cfg: Config,
    upper: RamNode,
    free: FreeList,
    end_page: u64,
}

impl FlatMpt {
    pub fn create(path: impl AsRef<Path>, cfg: Config) -> Result<Self> {
        if cfg.min_promote_bytes == 0 || cfg.min_promote_bytes > cfg.max_leaf_bytes {
            bail!("invalid split thresholds");
        }
        let path = path.as_ref();
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;

        // RocksDB instance lives in a sibling directory. `create` is a fresh
        // start, so discard any leftover store from a previous run at this path.
        let values_path = values_path(path);
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let _ = DB::destroy(&opts, &values_path);
        let values = DB::open(&opts, &values_path)?;

        Ok(Self {
            cfg,
            store: FlatFile::new(file),
            upper: RamNode::Empty,
            values,
            overlay: HashMap::new(),
            path: path.to_path_buf(),
        })
    }

    /// Reopen a database previously written with [`FlatMpt::persist`]. Reattaches
    /// to the existing flat file and value store (no truncation) and restores the
    /// RAM frontier and free list from the `.meta` manifest.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let meta = meta_path(path);
        let bytes = std::fs::read(&meta)
            .map_err(|e| anyhow!("no manifest at {}: {e}", meta.display()))?;
        let Manifest {
            cfg,
            upper,
            mut free,
            end_page,
        } = bincode::deserialize(&bytes)?;
        // The size index isn't serialized; rebuild it from the offset map.
        free.reindex();

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let values = DB::open(&opts, values_path(path))?;

        Ok(Self {
            cfg,
            store: FlatFile {
                file,
                free: Mutex::new(free),
                end_page: AtomicU64::new(end_page),
            },
            upper,
            values,
            overlay: HashMap::new(),
            path: path.to_path_buf(),
        })
    }

    /// Flush buffered values to RocksDB and the flat file's writer. Call this to
    /// make all preceding inserts visible in the value store without a full
    /// [`persist`](Self::persist) checkpoint.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_values()?;
        self.store.flush()
    }

    /// Flush the buffered value overlay to RocksDB as a single batch.
    fn flush_values(&mut self) -> Result<()> {
        if self.overlay.is_empty() {
            return Ok(());
        }
        let _g = prof::scope(prof::Cat::ValuePut);
        let mut batch = WriteBatch::default();
        for (key, value) in &self.overlay {
            batch.put(key, value);
        }
        self.values.write(batch)?;
        self.overlay.clear();
        Ok(())
    }

    /// Checkpoint the in-RAM state to disk so the database can later be reopened
    /// with [`FlatMpt::open`]. Flushes buffered values, fsyncs the flat file, then
    /// writes the manifest atomically (temp file + rename) so a crash can't leave
    /// a torn manifest.
    pub fn persist(&mut self) -> Result<()> {
        self.flush_values()?;
        self.store.sync()?;
        let free = self.store.free.lock().unwrap();
        let manifest = ManifestRef {
            cfg: &self.cfg,
            upper: &self.upper,
            free: &free,
            end_page: self.store.end_page(),
        };
        let bytes = bincode::serialize(&manifest)?;
        drop(free);

        let meta = meta_path(&self.path);
        let mut tmp = meta.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &meta)?;
        Ok(())
    }

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<Hash> {
        let value_hash = hash_leaf_value(&value);
        self.overlay.insert(key, value);
        if self.overlay.len() >= VALUE_BATCH {
            self.flush_values()?;
        }
        let cfg = self.cfg.clone();
        insert_ram(
            &mut self.store,
            &cfg,
            &mut self.upper,
            Vec::new(),
            key,
            value_hash,
        )?;
        self.store.flush()?;
        Ok(self.root())
    }

    /// Insert/overwrite many key/value pairs at once. Equivalent in result to
    /// calling [`insert`](Self::insert) for each pair (last value wins on a
    /// duplicate key within the batch), but far cheaper: values are written to
    /// RocksDB in one batch, and the trie is updated by grouping keys per route
    /// so every touched disk leaf is read/rebuilt/written exactly once and every
    /// node is re-hashed at most once. Returns the new root.
    pub fn insert_batch(&mut self, entries: Vec<(Key, Vec<u8>)>) -> Result<Hash> {
        if entries.is_empty() {
            return Ok(self.root());
        }
        let t_a = std::time::Instant::now();
        // Dedup (last write wins) and compute leaf value-hashes; buffer values.
        let mut leaves: BTreeMap<Key, Hash> = BTreeMap::new();
        for (key, value) in entries {
            let value_hash = hash_leaf_value(&value);
            self.overlay.insert(key, value);
            leaves.insert(key, value_hash);
        }
        self.flush_values()?;
        let cfg = self.cfg.clone();

        // Phase A (serial, read-only): route each key to the frontier disk leaf
        // it lands in, grouping keys per leaf. Keys with no existing leaf create
        // fresh structure and are applied serially afterwards.
        let mut groups: HashMap<u32, (DiskPtr, Key, Vec<(Key, Hash)>)> = HashMap::new();
        let mut fresh: Vec<(Key, Hash)> = Vec::new();
        for (key, value_hash) in leaves {
            match find_disk_ptr_key(&self.upper, &key, 0) {
                Some(ptr) => {
                    groups
                        .entry(ptr.page)
                        .or_insert_with(|| (ptr, key, Vec::new()))
                        .2
                        .push((key, value_hash));
                }
                None => fresh.push((key, value_hash)),
            }
        }
        let groups: Vec<(DiskPtr, Key, Vec<(Key, Hash)>)> = groups.into_values().collect();
        let a_ns = t_a.elapsed().as_nanos() as u64;
        let t_b = std::time::Instant::now();

        // Phase B (parallel): each group reads its record, applies its keys
        // (record_node_insert + migrate + possible promotion), and produces the
        // replacement RamChild — all the per-record CPU + I/O off the serial path.
        // Groups touch disjoint subtrees; the store is thread-safe.
        let store = &self.store;
        let work = |(ptr, rep, keys): &(DiskPtr, Key, Vec<(Key, Hash)>)| {
            Ok::<_, anyhow::Error>((*rep, process_group(store, &cfg, *ptr, keys)?))
        };
        let results: Vec<(Key, RamChild)> = if groups.len() < 64 {
            groups.iter().map(work).collect::<Result<_>>()?
        } else {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8);
            let chunk = groups.len().div_ceil(threads);
            std::thread::scope(|scope| {
                let handles: Vec<_> = groups
                    .chunks(chunk)
                    .map(|c| scope.spawn(|| c.iter().map(work).collect::<Result<Vec<_>>>()))
                    .collect();
                let mut out = Vec::with_capacity(groups.len());
                for h in handles {
                    out.extend(h.join().expect("batch group thread panicked")?);
                }
                Ok::<_, anyhow::Error>(out)
            })?
        };

        let b_ns = t_b.elapsed().as_nanos() as u64;
        let t_c = std::time::Instant::now();

        // Phase C (serial): splice each group's result into the frontier, then
        // create structure for the brand-new keys. Recompute the root once.
        for (rep, new_child) in results {
            install_at_key(&mut self.upper, &rep, 0, new_child);
        }
        for (key, value_hash) in fresh {
            insert_ram(&self.store, &cfg, &mut self.upper, Vec::new(), key, value_hash)?;
        }
        self.store.flush()?;
        let root = self.root();
        stats::on_batch(a_ns, b_ns, t_c.elapsed().as_nanos() as u64);
        Ok(root)
    }

    pub fn get_value(&self, key: &Key) -> Result<Option<Vec<u8>>> {
        let _g = prof::scope(prof::Cat::ValueGet);
        // Buffered writes win over what's already in RocksDB.
        if let Some(value) = self.overlay.get(key) {
            return Ok(Some(value.clone()));
        }
        Ok(self.values.get(key)?)
    }

    pub fn root(&self) -> Hash {
        hash_ram(&self.upper)
    }

    pub fn ram_nodes(&self) -> usize {
        count_ram_nodes(&self.upper)
    }

    /// Number of disk leaves (`RamChild::Disk`) the frontier points at.
    pub fn disk_leaves(&self) -> usize {
        count_disk_leaves(&self.upper)
    }

    /// Size summary of the live disk leaves, from their `DiskPtr`s (RAM-only, no
    /// reads): count, total record bytes, and a page-count histogram. Average
    /// leaf size = `total_bytes / count` — it collapses when a split cascade
    /// replaces full leaves with swarms of near-empty ones.
    pub fn leaf_stats(&self) -> LeafStats {
        let mut stats = LeafStats::default();
        collect_leaf_stats(&self.upper, &mut stats);
        stats
    }

    /// Logical size of the flat file (high-water mark). Stays flat across
    /// rewrites when freed space is reused.
    pub fn flat_file_len(&self) -> u64 {
        self.store.end_page() * PAGE
    }

    /// Total bytes currently held in the flat file's free list.
    pub fn free_bytes(&self) -> u64 {
        self.store.free.lock().unwrap().total() * PAGE
    }

    /// Number of distinct free regions tracked in the flat file.
    pub fn free_regions(&self) -> usize {
        self.store.free.lock().unwrap().region_count()
    }

    /// Heap held by the in-RAM index — the part of the database that is *not*
    /// on disk: the trie frontier, the free list, and the unflushed value
    /// overlay. Excludes the OS page cache and RocksDB's own (C++) memory.
    pub fn ram_report(&self) -> RamReport {
        let frontier_nodes = count_ram_nodes(&self.upper);
        let frontier_bytes = frontier_bytes(&self.upper);
        let free_regions = self.store.free.lock().unwrap().region_count();
        // by_offset (u64->u32) and by_size ((u32,u64)) both hold one entry per
        // region; this is the stored-data size and omits BTree node overhead.
        let free_list_bytes = free_regions
            * (std::mem::size_of::<(u64, u32)>() + std::mem::size_of::<(u32, u64)>());
        let overlay_entries = self.overlay.len();
        let overlay_bytes: usize = self
            .overlay
            .iter()
            .map(|(k, v)| k.len() + v.capacity())
            .sum();
        RamReport {
            frontier_nodes,
            frontier_bytes,
            free_regions,
            free_list_bytes,
            overlay_entries,
            overlay_bytes,
        }
    }

    /// Number of flat-file records read to reach `key` (1 + the overflow-chain
    /// depth on its path). 0 if no disk record is addressed for the key.
    pub fn disk_accesses_for_key(&mut self, key: &Key) -> Result<usize> {
        let nibbles = key_nibbles(key);
        let Some(mut ptr) = find_disk_ptr(&self.upper, &nibbles, 0) else {
            return Ok(0);
        };
        let mut reads = 0;
        loop {
            let subtree = self.store.read(ptr)?;
            reads += 1;
            match follow_key(&subtree.node, subtree.prefix.len(), key) {
                PathEnd::Overflow(next) => ptr = next,
                PathEnd::Inline(true) => return Ok(reads),
                PathEnd::Inline(false) => bail!("key not found in addressed disk subtree"),
            }
        }
    }
}

/// Sibling path for the RocksDB value store, e.g. `db.flat` -> `db.flat.values`.
fn values_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".values");
    path.with_file_name(name)
}

/// Sibling path for the manifest, e.g. `db.flat` -> `db.flat.meta`.
fn meta_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".meta");
    path.with_file_name(name)
}

fn insert_ram(
    store: &FlatFile,
    cfg: &Config,
    node: &mut RamNode,
    prefix: Vec<u8>,
    key: Key,
    value_hash: Hash,
) -> Result<()> {
    let nibbles = key_nibbles(&key);
    // This node (or its subtree) is about to change, so its cached hash is stale.
    invalidate_ram(node);
    match node {
        RamNode::Empty => {
            let idx = nibbles[prefix.len()] as usize;
            // Build the leaf at the branch-slot depth (prefix + slot nibble), the
            // same depth every other code path uses, so the representation — and
            // therefore the hash — is independent of how the leaf was created.
            let mut child_prefix = prefix;
            child_prefix.push(idx as u8);
            let subtree = subtree_from_entries(child_prefix, vec![(key, leaf_hash(key, value_hash))]);
            let (payload, _) = serialize_subtree(&subtree)?;
            let ptr = store.write_payload(&payload)?;
            let mut children = empty_children();
            children[idx] = Some(RamChild::Disk { ptr, root: hash_node(&subtree.node) });
            *node = RamNode::Branch {
                children,
                hash: Cell::new(None),
            };
            Ok(())
        }
        RamNode::Extension { path, child, .. } => {
            let common = common_prefix(path, &nibbles[prefix.len()..]);
            if common < path.len() {
                let old = std::mem::replace(node, RamNode::Empty);
                let RamNode::Extension {
                    path: old_path,
                    child: old_child,
                    ..
                } = old
                else {
                    unreachable!();
                };
                let mut children = empty_children();
                let old_idx = old_path[common] as usize;
                let old_remainder = old_path[common + 1..].to_vec();
                children[old_idx] = Some(RamChild::Ram(if old_remainder.is_empty() {
                    old_child
                } else {
                    Box::new(RamNode::Extension {
                        path: old_remainder,
                        child: old_child,
                        hash: Cell::new(None),
                    })
                }));

                let new_idx = nibbles[prefix.len() + common] as usize;
                let mut new_prefix = prefix.clone();
                new_prefix.extend_from_slice(&old_path[..common]);
                new_prefix.push(new_idx as u8);
                let subtree = subtree_from_entries(new_prefix, vec![(key, leaf_hash(key, value_hash))]);
                let (payload, _) = serialize_subtree(&subtree)?;
                let ptr = store.write_payload(&payload)?;
                children[new_idx] = Some(RamChild::Disk { ptr, root: hash_node(&subtree.node) });

                let branch = RamNode::Branch {
                    children,
                    hash: Cell::new(None),
                };
                *node = if common == 0 {
                    branch
                } else {
                    RamNode::Extension {
                        path: old_path[..common].to_vec(),
                        child: Box::new(branch),
                        hash: Cell::new(None),
                    }
                };
                Ok(())
            } else {
                let mut next_prefix = prefix;
                next_prefix.extend_from_slice(path);
                insert_ram(store, cfg, child, next_prefix, key, value_hash)
            }
        }
        RamNode::Branch { children, .. } => {
            if prefix.len() == nibbles.len() {
                bail!("key terminates at a frontier branch; keys must be distinct and fixed-length");
            }
            let idx = nibbles[prefix.len()] as usize;
            let mut child_prefix = prefix;
            child_prefix.push(idx as u8);
            match &mut children[idx] {
                Some(RamChild::Ram(child)) => {
                    insert_ram(store, cfg, child, child_prefix, key, value_hash)
                }
                Some(RamChild::Disk { ptr, root }) => {
                    let mut subtree = store.read(*ptr)?;
                    let old_ptr = *ptr;
                    // Incremental insert, crossing any overflow edges on the key's
                    // path (re-hashing only that path). Then shed children to
                    // overflow if the record outgrew `max` — the record stays on
                    // disk (one frontier entry) instead of promoting a RAM branch.
                    record_node_insert(store, cfg, &mut subtree.node, subtree.prefix.len(), key, value_hash)?;
                    migrate_record(store, cfg, &mut subtree)?;
                    if count_overflow_children(&subtree.node) >= PROMOTE_AT_OVERFLOW {
                        // Majority of children externalized: lift this record into
                        // the RAM frontier so its (now mostly-fat) children become
                        // first-class frontier entries — shallower reads/rewrites.
                        store.free(old_ptr);
                        children[idx] = Some(promote_record_to_ram(store, subtree)?);
                    } else {
                        let (payload, _) = serialize_subtree(&subtree)?;
                        // Old record is dead; reclaim before writing so the rewrite
                        // can reuse the same region when it still fits.
                        store.free(old_ptr);
                        *ptr = store.write_payload(&payload)?;
                        *root = hash_node(&subtree.node);
                    }
                    Ok(())
                }
                None => {
                    let subtree = subtree_from_entries(child_prefix, vec![(key, leaf_hash(key, value_hash))]);
                    let (payload, _) = serialize_subtree(&subtree)?;
                    let ptr = store.write_payload(&payload)?;
                    children[idx] = Some(RamChild::Disk { ptr, root: hash_node(&subtree.node) });
                    Ok(())
                }
            }
        }
    }
}


fn subtree_from_entries(prefix: Vec<u8>, entries: Vec<(Key, Hash)>) -> DiskSubtree {
    let node = build_node(&entries, prefix.len());
    DiskSubtree { prefix, node }
}

/// The `i`-th nibble of a key (i in 0..64), without allocating.
fn nibble_at(key: &Key, i: usize) -> u8 {
    let byte = key[i / 2];
    if i % 2 == 0 { byte >> 4 } else { byte & 0x0f }
}






// --- Disk-node constructors: compute and cache the node hash exactly once. ---

/// Position-independent leaf hash: `keccak(3 ‖ key ‖ value_hash)`. It commits to
/// the *full* key and the value, so it never changes when the leaf moves to a
/// different position in the tree (only the stored `path` does) — which is what
/// lets a divergence re-home a leaf without re-hashing it.
fn leaf_hash(key: Key, value_hash: Hash) -> Hash {
    let mut bytes = vec![3];
    bytes.extend_from_slice(&key);
    bytes.extend_from_slice(&value_hash);
    keccak(&bytes)
}

/// A leaf holding the suffix `path` (key nibbles from its position to depth 64)
/// and its precomputed `hash`.
fn leaf_node(path: Vec<u8>, hash: Hash) -> Node {
    Node::Leaf { path, hash }
}

fn make_extension(path: Vec<u8>, child: Node) -> Node {
    let hash = hash_join(4, &path, &hash_node(&child));
    Node::Extension {
        path,
        child: Box::new(child),
        hash,
    }
}

/// keccak(5 ‖ h0 ‖ … ‖ h15) over the 16 child digests (empty slots use the empty
/// hash). An `Overflow` child contributes its `root` via `hash_node`, so a branch
/// hashes identically whether a child is inline or overflowed.
fn branch_hash(children: &[Option<Box<Node>>; 16]) -> Hash {
    let mut bytes = vec![5];
    for child in children {
        bytes.extend_from_slice(&child.as_ref().map(|c| hash_node(c)).unwrap_or_else(empty_hash));
    }
    keccak(&bytes)
}

fn make_branch(children: [Option<Box<Node>>; 16]) -> Node {
    // Disk-side branches never carry a value (every key is a full 64-nibble path).
    let hash = branch_hash(&children);
    Node::Branch { children, hash }
}

/// Canonical node for a subtree holding exactly one entry at `depth`.
/// Canonical node for a single entry at `depth`: a bare leaf carrying the key's
/// suffix and its (already-computed) leaf hash. `leaf_hash` is the value from
/// [`leaf_hash`], not a raw value hash.
fn single_entry_node(key: Key, leaf_hash: Hash, depth: usize) -> Node {
    leaf_node(key_nibbles(&key)[depth..].to_vec(), leaf_hash)
}

fn build_node(entries: &[(Key, Hash)], depth: usize) -> Node {
    if entries.is_empty() {
        return Node::Empty;
    }
    if entries.len() == 1 {
        let (key, value_hash) = entries[0];
        return single_entry_node(key, value_hash, depth);
    }

    let nibbles: Vec<Vec<u8>> = entries.iter().map(|(key, _)| key_nibbles(key)).collect();
    let mut common = 0;
    while depth + common < 64 {
        let nibble = nibbles[0][depth + common];
        if nibbles.iter().all(|ks| ks[depth + common] == nibble) {
            common += 1;
        } else {
            break;
        }
    }
    if common > 0 {
        let path = nibbles[0][depth..depth + common].to_vec();
        return make_extension(path, build_node(entries, depth + common));
    }

    let mut grouped: [Vec<(Key, Hash)>; 16] = std::array::from_fn(|_| Vec::new());
    for (i, entry) in entries.iter().enumerate() {
        let idx = nibbles[i].get(depth).copied().unwrap_or(0) as usize;
        grouped[idx].push(*entry);
    }
    let mut children = empty_box_children();
    for (idx, group) in grouped.into_iter().enumerate() {
        if !group.is_empty() {
            children[idx] = Some(Box::new(build_node(&group, depth + 1)));
        }
    }
    make_branch(children)
}


/// Outcome of following a key's nibble path through one record's (inline) node.
enum PathEnd {
    /// The path reaches an `Overflow` child; continue in the record at `ptr`.
    Overflow(DiskPtr),
    /// The path terminates within this record; `bool` is whether the key is present.
    Inline(bool),
}

/// Follow `key`'s nibble path through `node` (rooted at `depth`), stopping at the
/// first `Overflow` edge (so it never recurses into one). Only walks the slot the
/// key routes to — siblings (including overflow siblings) are not visited.
fn follow_key(node: &Node, depth: usize, key: &Key) -> PathEnd {
    match node {
        Node::Empty => PathEnd::Inline(false),
        // The leaf holds the key's suffix; it matches iff that suffix equals the
        // remaining nibbles of `key` from here.
        Node::Leaf { path, .. } => PathEnd::Inline(key_nibbles(key)[depth..] == path[..]),
        Node::Extension { path, child, .. } => {
            let nibbles = key_nibbles(key);
            if nibbles.get(depth..depth + path.len()) == Some(path.as_slice()) {
                follow_key(child, depth + path.len(), key)
            } else {
                PathEnd::Inline(false)
            }
        }
        Node::Branch { children, .. } => {
            let idx = nibble_at(key, depth) as usize;
            match children[idx].as_deref() {
                None => PathEnd::Inline(false),
                Some(Node::Overflow { ptr, .. }) => PathEnd::Overflow(*ptr),
                Some(child) => follow_key(child, depth + 1, key),
            }
        }
        Node::Overflow { ptr, .. } => PathEnd::Overflow(*ptr),
    }
}

/// Like [`node_insert`], but the subtree may contain `Overflow` children (which
/// live only at branch slots). Crossing one reads, recurses into, migrates, and
/// rewrites that child record, then updates the `Overflow{ptr, root}` in place.
/// Pure-inline subtrees are handled exactly as `node_insert` does, so the
/// resulting structure and hashes are identical to an all-inline build.
fn record_node_insert(
    store: &FlatFile,
    cfg: &Config,
    node: &mut Node,
    depth: usize,
    key: Key,
    value_hash: Hash,
) -> Result<()> {
    let nibbles = key_nibbles(&key);
    let lh = leaf_hash(key, value_hash);
    let updated = match std::mem::replace(node, Node::Empty) {
        Node::Empty => single_entry_node(key, lh, depth),
        Node::Leaf { path, hash } => {
            let remaining = &nibbles[depth..];
            let common = common_prefix(&path, remaining);
            if common == path.len() {
                // Same key (both suffixes run to depth 64): overwrite the value.
                debug_assert_eq!(path.as_slice(), remaining);
                leaf_node(path, lh)
            } else {
                // The new key diverges partway along the leaf's suffix. Keep the
                // old leaf under a shorter suffix (its full key — and so its hash —
                // is unchanged) and add the new leaf alongside.
                let mut children = empty_box_children();
                let old_idx = path[common] as usize;
                children[old_idx] = Some(Box::new(leaf_node(path[common + 1..].to_vec(), hash)));
                let new_idx = remaining[common] as usize;
                children[new_idx] =
                    Some(Box::new(single_entry_node(key, lh, depth + common + 1)));
                let branch = make_branch(children);
                if common == 0 {
                    branch
                } else {
                    make_extension(path[..common].to_vec(), branch)
                }
            }
        }
        Node::Extension { path, mut child, .. } => {
            let common = common_prefix(&path, &nibbles[depth..]);
            if common == path.len() {
                record_node_insert(store, cfg, &mut child, depth + path.len(), key, value_hash)?;
                make_extension(path, *child)
            } else {
                // Diverges partway along the extension — no overflow edge here.
                let mut children = empty_box_children();
                let old_idx = path[common] as usize;
                let old_rest = path[common + 1..].to_vec();
                children[old_idx] = Some(Box::new(if old_rest.is_empty() {
                    *child
                } else {
                    make_extension(old_rest, *child)
                }));
                let new_idx = nibbles[depth + common] as usize;
                children[new_idx] =
                    Some(Box::new(single_entry_node(key, lh, depth + common + 1)));
                let branch = make_branch(children);
                if common == 0 {
                    branch
                } else {
                    make_extension(path[..common].to_vec(), branch)
                }
            }
        }
        Node::Branch { mut children, .. } => {
            let idx = nibbles[depth] as usize;
            match children[idx].as_deref_mut() {
                Some(Node::Overflow { ptr, root }) => {
                    // Recurse into the overflow record, then rewrite it.
                    let mut sub = store.read(*ptr)?;
                    record_node_insert(store, cfg, &mut sub.node, sub.prefix.len(), key, value_hash)?;
                    migrate_record(store, cfg, &mut sub)?;
                    let (payload, _) = serialize_subtree(&sub)?;
                    store.free(*ptr);
                    *ptr = store.write_payload(&payload)?;
                    *root = hash_node(&sub.node);
                }
                Some(child) => record_node_insert(store, cfg, child, depth + 1, key, value_hash)?,
                None => {
                    children[idx] = Some(Box::new(single_entry_node(key, lh, depth + 1)));
                }
            }
            make_branch(children)
        }
        Node::Overflow { .. } => unreachable!("overflow is only reached via its parent branch slot"),
    };
    *node = updated;
    Ok(())
}

/// `&mut` access to the top branch's children (descending through a leading
/// extension). `None` if the record holds no branch (a 0/1-entry record).
fn top_branch_children_mut(node: &mut Node) -> Option<&mut [Option<Box<Node>>; 16]> {
    match node {
        Node::Branch { children, .. } => Some(children),
        Node::Extension { child, .. } => top_branch_children_mut(child),
        _ => None,
    }
}

/// Recompute cached hashes for the record's leading structure after a top-branch
/// child slot changed (e.g. an inline child became an `Overflow`).
fn rehash_top(node: &mut Node) {
    match node {
        Node::Branch { children, hash, .. } => *hash = branch_hash(children),
        Node::Extension { path, child, hash } => {
            rehash_top(child);
            *hash = hash_join(4, path, &hash_node(child));
        }
        _ => {}
    }
}

/// The nibble prefix of the record's top branch (record prefix + leading
/// extension path), or `None` if the record holds no branch.
fn top_branch_prefix(subtree: &DiskSubtree) -> Option<Vec<u8>> {
    match &subtree.node {
        Node::Branch { .. } => Some(subtree.prefix.clone()),
        Node::Extension { path, child, .. } if matches!(**child, Node::Branch { .. }) => {
            let mut p = subtree.prefix.clone();
            p.extend_from_slice(path);
            Some(p)
        }
        _ => None,
    }
}

/// Shed top-branch children of `subtree` into their own `Overflow` records until
/// it fits (the "(2)→(3) migration"):
///  - **Proactive:** any inline child whose own record would be ≥ `min_promote`
///    is moved out (it deserves its own record).
///  - **Forced:** while the record still exceeds `max_leaf_bytes`, the *largest*
///    inline child is moved out (ignoring `min_promote`) until ≤ `target`.
/// Converges — worst case every child becomes an `Overflow` and the record is a
/// bare branch header. Root-preserving: an `Overflow{root}` contributes the same
/// digest the inline child did.
fn migrate_record(store: &FlatFile, cfg: &Config, subtree: &mut DiskSubtree) -> Result<()> {
    let Some(branch_prefix) = top_branch_prefix(subtree) else {
        return Ok(());
    };
    let child_prefix_len = branch_prefix.len() + 1;
    loop {
        // Size-only (no allocation): the record, and each inline child as its
        // own record. Most inserts shed nothing, so this must stay cheap.
        let total = record_size(subtree.prefix.len(), &subtree.node);
        let children = top_branch_children_mut(&mut subtree.node).unwrap();
        let mut inline: Vec<(usize, usize)> = Vec::new();
        for (i, slot) in children.iter().enumerate() {
            if let Some(boxed) = slot {
                if !matches!(**boxed, Node::Overflow { .. }) {
                    inline.push((i, record_size(child_prefix_len, boxed)));
                }
            }
        }

        // Pick a child to shed: proactive first, then forced if still over max.
        let shed = inline
            .iter()
            .find(|(_, b)| *b >= cfg.min_promote_bytes)
            .map(|(i, _)| *i)
            .or_else(|| {
                if total > cfg.max_leaf_bytes {
                    inline.iter().max_by_key(|(_, b)| *b).map(|(i, _)| *i)
                } else {
                    None
                }
            });
        let Some(idx) = shed else { return Ok(()) };

        // Move children[idx] out to its own record and replace with an Overflow.
        let child = children[idx].take().unwrap();
        let mut cp = branch_prefix.clone();
        cp.push(idx as u8);
        let child_sub = DiskSubtree { prefix: cp, node: *child };
        let (payload, _) = serialize_subtree(&child_sub)?;
        let ptr = store.write_payload(&payload)?;
        let root = hash_node(&child_sub.node);
        let children = top_branch_children_mut(&mut subtree.node).unwrap();
        children[idx] = Some(Box::new(Node::Overflow { ptr, root }));
        rehash_top(&mut subtree.node);
    }
}

/// Once this many of a packed record's 16 top-branch children have been
/// externalized to `Overflow` records, the branch is "branchy" enough to earn a
/// place in the RAM frontier — promote it (see [`promote_record_to_ram`]). A
/// strict majority (8/16) lifts the frontier early enough to keep read/rewrite
/// depth shallow, while the children left inline are by then just under
/// `min_promote` (so writing them out wastes little space).
const PROMOTE_AT_OVERFLOW: usize = 8;

/// Count the `Overflow` children of a record's top branch (through a leading
/// extension).
fn count_overflow_children(node: &Node) -> usize {
    match node {
        Node::Branch { children, .. } => children
            .iter()
            .flatten()
            .filter(|c| matches!(***c, Node::Overflow { .. }))
            .count(),
        Node::Extension { child, .. } => count_overflow_children(child),
        _ => 0,
    }
}

/// Promote a packed disk record into a RAM-frontier node: its top branch becomes
/// a `RamNode::Branch` (re-wrapped in a `RamNode::Extension` if the record had a
/// leading extension), and every child becomes a `RamChild::Disk` — `Overflow`
/// children keep their existing record, inline children are written out to their
/// own records. Root-preserving: the RAM node hashes identically to the disk
/// record (unified tags, unchanged child roots). The caller frees the old record.
fn promote_record_to_ram(store: &FlatFile, subtree: DiskSubtree) -> Result<RamChild> {
    let DiskSubtree { prefix, node } = subtree;
    let (ext_path, branch_node) = match node {
        Node::Branch { .. } => (Vec::new(), node),
        Node::Extension { path, child, .. } => (path, *child),
        _ => unreachable!("promote called on a record without a top branch"),
    };
    let Node::Branch { children, .. } = branch_node else {
        unreachable!("a leading extension must wrap a branch");
    };
    let mut branch_prefix = prefix;
    branch_prefix.extend_from_slice(&ext_path);

    let mut ram_children = empty_children();
    for (i, slot) in children.into_iter().enumerate() {
        let Some(boxed) = slot else { continue };
        let child = match *boxed {
            // Already its own record — reuse it verbatim.
            Node::Overflow { ptr, root } => RamChild::Disk { ptr, root },
            // Inline small child — write it out (it's < min_promote, fits one record).
            other => {
                let mut cp = branch_prefix.clone();
                cp.push(i as u8);
                let root = hash_node(&other);
                let child_sub = DiskSubtree { prefix: cp, node: other };
                let (payload, _) = serialize_subtree(&child_sub)?;
                let ptr = store.write_payload(&payload)?;
                RamChild::Disk { ptr, root }
            }
        };
        ram_children[i] = Some(child);
    }
    let branch = RamNode::Branch {
        children: ram_children,
        hash: Cell::new(None),
    };
    if ext_path.is_empty() {
        Ok(RamChild::Ram(Box::new(branch)))
    } else {
        Ok(RamChild::Ram(Box::new(RamNode::Extension {
            path: ext_path,
            child: Box::new(branch),
            hash: Cell::new(None),
        })))
    }
}

/// Apply a whole group of keys (all routing to the disk record at `ptr`) and
/// produce the replacement `RamChild` for that frontier slot. Pure w.r.t. the
/// frontier — it only reads/writes the (disjoint) record subtree via the
/// thread-safe `store`, so groups for different frontier leaves run concurrently.
fn process_group(
    store: &FlatFile,
    cfg: &Config,
    ptr: DiskPtr,
    keys: &[(Key, Hash)],
) -> Result<RamChild> {
    let t = std::time::Instant::now();
    let mut subtree = store.read(ptr)?;
    let read_ns = t.elapsed().as_nanos() as u64;

    let depth = subtree.prefix.len();
    let t = std::time::Instant::now();
    for (key, value_hash) in keys {
        record_node_insert(store, cfg, &mut subtree.node, depth, *key, *value_hash)?;
    }
    let rebuild_ns = t.elapsed().as_nanos() as u64;

    let t = std::time::Instant::now();
    migrate_record(store, cfg, &mut subtree)?;
    let out = if count_overflow_children(&subtree.node) >= PROMOTE_AT_OVERFLOW {
        store.free(ptr);
        promote_record_to_ram(store, subtree)
    } else {
        let (payload, _) = serialize_subtree(&subtree)?;
        store.free(ptr);
        let new_ptr = store.write_payload(&payload)?;
        Ok(RamChild::Disk {
            ptr: new_ptr,
            root: hash_node(&subtree.node),
        })
    };
    stats::on_group(read_ns, rebuild_ns, t.elapsed().as_nanos() as u64);
    out
}

/// Splice a batch result into the frontier: navigate `key`'s route to the
/// `RamChild::Disk` slot it lands in and replace it with `new`, invalidating the
/// cached hash of every node on the path. Returns whether the slot was found.
fn install_at_key(node: &mut RamNode, key: &Key, depth: usize, new: RamChild) -> bool {
    match node {
        RamNode::Empty => false,
        RamNode::Extension { path, child, hash } => {
            let done = install_at_key(child, key, depth + path.len(), new);
            if done {
                hash.set(None);
            }
            done
        }
        RamNode::Branch { children, hash } => {
            let idx = nibble_at(key, depth) as usize;
            let done = match children[idx].as_mut() {
                Some(RamChild::Disk { .. }) => {
                    children[idx] = Some(new);
                    true
                }
                Some(RamChild::Ram(child)) => install_at_key(child, key, depth + 1, new),
                None => false,
            };
            if done {
                hash.set(None);
            }
            done
        }
    }
}

/// On-disk byte size of `node` (matches exactly what [`write_node`] emits), with
/// no allocation — a cheap size pass used by migration to decide shedding
/// without serializing every child into a throwaway buffer.
fn node_size(node: &Node) -> usize {
    match node {
        Node::Empty => 1,
        Node::Leaf { path, .. } => 1 + (1 + path.len().div_ceil(2)) + 32,
        Node::Extension { path, child, .. } => {
            1 + (1 + path.len().div_ceil(2)) + 32 + node_size(child)
        }
        Node::Branch { children, .. } => {
            1 + 2 + 32 + children.iter().flatten().map(|c| node_size(c)).sum::<usize>()
        }
        Node::Overflow { .. } => 1 + 8 + 4 + 32,
    }
}

/// Total on-disk record size for a `DiskSubtree { prefix, node }` — equal to the
/// `total` [`serialize_subtree`] would return, but allocation-free.
fn record_size(prefix_len: usize, node: &Node) -> usize {
    // path-len byte(1) + path bytes + node. No magic/version/length framing — the
    // record is just the prefix path followed by the node tree.
    (1 + prefix_len.div_ceil(2)) + node_size(node)
}

/// Serialize a subtree to its on-disk payload (the prefix path followed by the
/// node tree — no magic/version/length framing; the size is carried by the
/// addressing [`DiskPtr`]). Returns the payload and its exact byte length.
fn serialize_subtree(subtree: &DiskSubtree) -> Result<(Vec<u8>, usize)> {
    let _g = prof::scope(prof::Cat::Serialize);
    let mut payload = Vec::new();
    write_nibble_path(&mut payload, &subtree.prefix)?;
    write_node(&mut payload, &subtree.node)?;
    let total = payload.len();
    Ok((payload, total))
}

fn deserialize_subtree(payload: &[u8]) -> Result<DiskSubtree> {
    let mut reader = CompactReader::new(payload);
    let prefix = reader.read_nibble_path()?;
    let node = reader.read_node()?;
    // Reads fetch exactly `DiskPtr::len` bytes, so the record must consume all of
    // them; anything left over signals corruption.
    if !reader.is_finished() {
        bail!("trailing bytes in flat-file record");
    }
    Ok(DiskSubtree { prefix, node })
}

fn write_node(out: &mut Vec<u8>, node: &Node) -> Result<()> {
    match node {
        Node::Empty => out.push(0),
        Node::Leaf { path, hash } => {
            out.push(1);
            write_nibble_path(out, path)?;
            out.extend_from_slice(hash);
        }
        Node::Extension { path, child, hash } => {
            out.push(2);
            write_nibble_path(out, path)?;
            out.extend_from_slice(hash);
            write_node(out, child)?;
        }
        Node::Branch {
            children,
            hash,
        } => {
            out.push(3);
            let mut bitmap = 0u16;
            for (idx, child) in children.iter().enumerate() {
                if child.is_some() {
                    bitmap |= 1 << idx;
                }
            }
            out.extend_from_slice(&bitmap.to_le_bytes());
            out.extend_from_slice(hash);
            for child in children.iter().flatten() {
                write_node(out, child)?;
            }
        }
        Node::Overflow { ptr, root } => {
            out.push(4);
            out.extend_from_slice(&ptr.page.to_le_bytes());
            out.extend_from_slice(&ptr.len.to_le_bytes());
            out.extend_from_slice(root);
        }
    }
    Ok(())
}

fn write_nibble_path(out: &mut Vec<u8>, path: &[u8]) -> Result<()> {
    if path.len() > u8::MAX as usize {
        bail!("nibble path too long");
    }
    if path.iter().any(|&nibble| nibble > 0x0f) {
        bail!("invalid nibble path");
    }
    out.push(path.len() as u8);
    for pair in path.chunks(2) {
        let high = pair[0] << 4;
        let low = pair.get(1).copied().unwrap_or(0);
        out.push(high | low);
    }
    Ok(())
}

struct CompactReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> CompactReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_finished(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow!("compact subtree offset overflow"))?;
        if end > self.bytes.len() {
            bail!("truncated compact subtree record");
        }
        let bytes = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_hash(&mut self) -> Result<Hash> {
        let bytes = self.read_bytes(32)?;
        let mut hash = [0; 32];
        hash.copy_from_slice(bytes);
        Ok(hash)
    }

    fn read_nibble_path(&mut self) -> Result<Vec<u8>> {
        let len = self.read_u8()? as usize;
        if len > 64 {
            bail!("compact subtree nibble path too long");
        }
        let mut path = Vec::with_capacity(len);
        for _ in 0..len.div_ceil(2) {
            let byte = self.read_u8()?;
            path.push(byte >> 4);
            if path.len() < len {
                path.push(byte & 0x0f);
            }
        }
        Ok(path)
    }

    fn read_node(&mut self) -> Result<Node> {
        match self.read_u8()? {
            0 => Ok(Node::Empty),
            1 => {
                let path = self.read_nibble_path()?;
                let hash = self.read_hash()?;
                Ok(Node::Leaf { path, hash })
            }
            2 => {
                let path = self.read_nibble_path()?;
                let hash = self.read_hash()?;
                let child = Box::new(self.read_node()?);
                Ok(Node::Extension { path, child, hash })
            }
            3 => {
                let bitmap = self.read_u16()?;
                let hash = self.read_hash()?;
                let mut children = empty_box_children();
                for (idx, slot) in children.iter_mut().enumerate() {
                    if bitmap & (1 << idx) != 0 {
                        *slot = Some(Box::new(self.read_node()?));
                    }
                }
                Ok(Node::Branch { children, hash })
            }
            4 => {
                let page = self.read_u32()?;
                let len = self.read_u32()?;
                let root = self.read_hash()?;
                Ok(Node::Overflow {
                    ptr: DiskPtr { page, len },
                    root,
                })
            }
            tag => bail!("invalid compact subtree node tag {tag}"),
        }
    }
}


fn find_disk_ptr(node: &RamNode, nibbles: &[u8], depth: usize) -> Option<DiskPtr> {
    match node {
        RamNode::Empty => None,
        RamNode::Extension { path, child, .. } => {
            if nibbles.get(depth..depth + path.len()) == Some(path.as_slice()) {
                find_disk_ptr(child, nibbles, depth + path.len())
            } else {
                None
            }
        }
        RamNode::Branch { children, .. } => {
            let idx = *nibbles.get(depth)? as usize;
            match children[idx].as_ref()? {
                RamChild::Ram(child) => find_disk_ptr(child, nibbles, depth + 1),
                RamChild::Disk { ptr, .. } => Some(*ptr),
            }
        }
    }
}

/// Route `key` to the disk leaf it currently lives in (or `None` if it would
/// land on an empty/absent slot). Allocation-free variant of [`find_disk_ptr`],
/// used to collect leaves for parallel prefetch.
fn find_disk_ptr_key(node: &RamNode, key: &Key, depth: usize) -> Option<DiskPtr> {
    match node {
        RamNode::Empty => None,
        RamNode::Extension { path, child, .. } => {
            if depth + path.len() <= 64
                && path.iter().enumerate().all(|(i, &p)| nibble_at(key, depth + i) == p)
            {
                find_disk_ptr_key(child, key, depth + path.len())
            } else {
                None
            }
        }
        RamNode::Branch { children, .. } => {
            if depth >= 64 {
                return None;
            }
            match children[nibble_at(key, depth) as usize].as_ref()? {
                RamChild::Ram(child) => find_disk_ptr_key(child, key, depth + 1),
                RamChild::Disk { ptr, .. } => Some(*ptr),
            }
        }
    }
}

/// Drop the cached hash of `node` (if any). Called as the insert descends, so
/// exactly the nodes on the touched path are invalidated and later recomputed.
fn invalidate_ram(node: &RamNode) {
    match node {
        RamNode::Extension { hash, .. } | RamNode::Branch { hash, .. } => hash.set(None),
        RamNode::Empty => {}
    }
}

fn hash_ram(node: &RamNode) -> Hash {
    match node {
        RamNode::Empty => empty_hash(),
        RamNode::Extension { path, child, hash } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            // Tag 4 == the disk-side extension tag (`make_extension`). Using one
            // tag per node type across RAM and disk makes a node's hash depend
            // only on its structure, never on which side of the storage boundary
            // it currently lives on — see `root_is_independent_of_leaf_size`.
            let computed = hash_join(4, path, &hash_ram(child));
            hash.set(Some(computed));
            computed
        }
        RamNode::Branch { children, hash } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            // Tag 5 == the disk-side branch tag (`make_branch`); see above.
            let mut bytes = vec![5];
            for child in children {
                let h = match child {
                    Some(RamChild::Ram(node)) => hash_ram(node),
                    Some(RamChild::Disk { root, .. }) => *root,
                    None => empty_hash(),
                };
                bytes.extend_from_slice(&h);
            }
            let computed = keccak(&bytes);
            hash.set(Some(computed));
            computed
        }
    }
}

/// Disk-node hash accessor: returns the cached hash (computed at construction).
fn hash_node(node: &Node) -> Hash {
    match node {
        Node::Empty => empty_hash(),
        Node::Leaf { hash, .. } | Node::Extension { hash, .. } | Node::Branch { hash, .. } => {
            *hash
        }
        // The overflowed subtree's root is exactly the hash the inline node would
        // have had — that equivalence is what keeps the root storage-independent.
        Node::Overflow { root, .. } => *root,
    }
}

fn hash_join(tag: u8, path: &[u8], child: &Hash) -> Hash {
    let mut bytes = vec![tag, path.len() as u8];
    bytes.extend_from_slice(path);
    bytes.extend_from_slice(child);
    keccak(&bytes)
}

fn hash_leaf_value(value: &[u8]) -> Hash {
    let mut bytes = vec![6];
    bytes.extend_from_slice(value);
    keccak(&bytes)
}

fn keccak(bytes: &[u8]) -> Hash {
    let _g = prof::scope(prof::Cat::Keccak);
    let output: Hash = Keccak256::digest(bytes).into();
    prof::record(output);
    output
}

/// Hash of the empty/absent node. It is a constant, so compute it once instead
/// of re-running keccak for every empty child slot of every branch we hash.
fn empty_hash() -> Hash {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Hash> = OnceLock::new();
    *EMPTY.get_or_init(|| keccak(&[0]))
}

fn key_nibbles(key: &Key) -> Vec<u8> {
    key.iter()
        .flat_map(|byte| [byte >> 4, byte & 0x0f])
        .collect()
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}


fn empty_children() -> [Option<RamChild>; 16] {
    std::array::from_fn(|_| None)
}

fn empty_box_children() -> [Option<Box<Node>>; 16] {
    std::array::from_fn(|_| None)
}

/// Size summary of the live disk leaves (see [`FlatMpt::leaf_stats`]).
#[derive(Debug, Default, Clone)]
pub struct LeafStats {
    pub count: usize,
    pub total_bytes: u64,
    /// Histogram by page count: index `p` (1..=8, 8 = "8 or more") -> #leaves.
    pub page_hist: [u64; 9],
}

impl LeafStats {
    pub fn avg_bytes(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.total_bytes / self.count as u64
        }
    }
}

fn collect_leaf_stats(node: &RamNode, stats: &mut LeafStats) {
    match node {
        RamNode::Empty => {}
        RamNode::Extension { child, .. } => collect_leaf_stats(child, stats),
        RamNode::Branch { children, .. } => {
            for child in children.iter().flatten() {
                match child {
                    RamChild::Disk { ptr, .. } => {
                        stats.count += 1;
                        stats.total_bytes += ptr.len as u64;
                        let pages = ((ptr.len as u64).div_ceil(PAGE).max(1)).min(8) as usize;
                        stats.page_hist[pages] += 1;
                    }
                    RamChild::Ram(n) => collect_leaf_stats(n, stats),
                }
            }
        }
    }
}

fn count_disk_leaves(node: &RamNode) -> usize {
    match node {
        RamNode::Empty => 0,
        RamNode::Extension { child, .. } => count_disk_leaves(child),
        RamNode::Branch { children, .. } => children
            .iter()
            .flatten()
            .map(|c| match c {
                RamChild::Disk { .. } => 1,
                RamChild::Ram(n) => count_disk_leaves(n),
            })
            .sum(),
    }
}

fn count_ram_nodes(node: &RamNode) -> usize {
    match node {
        RamNode::Empty => 0,
        RamNode::Extension { child, .. } => 1 + count_ram_nodes(child),
        RamNode::Branch { children, .. } => {
            1 + children
                .iter()
                .filter_map(|child| match child {
                    Some(RamChild::Ram(node)) => Some(count_ram_nodes(node)),
                    _ => None,
                })
                .sum::<usize>()
        }
    }
}

/// Heap bytes of the frontier rooted at `node`. Each node occupies
/// `size_of::<RamNode>()` (the enum is sized to its largest variant, so this is
/// the real allocation size of every boxed node); `Disk` children are inline in
/// their parent branch's array, so only `Ram` children add a recursive cost.
fn frontier_bytes(node: &RamNode) -> usize {
    let mut total = std::mem::size_of::<RamNode>();
    match node {
        RamNode::Empty => {}
        RamNode::Extension { path, child, .. } => {
            total += path.capacity();
            total += frontier_bytes(child);
        }
        RamNode::Branch { children, .. } => {
            for child in children.iter().flatten() {
                if let RamChild::Ram(node) = child {
                    total += frontier_bytes(node);
                }
            }
        }
    }
    total
}

pub fn hashed_key(input: impl AsRef<[u8]>) -> Key {
    keccak(input.as_ref())
}

pub fn hex(hash: Hash) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn assert_root_changes(old: Hash, new: Hash) -> Result<()> {
    if old == new {
        Err(anyhow!("root did not change"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn db(cfg: Config) -> FlatMpt {
        FlatMpt::create(NamedTempFile::new().unwrap().path(), cfg).unwrap()
    }

    #[test]
    fn batch_insert_matches_one_by_one() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let pairs: Vec<(Key, Vec<u8>)> = (0..3000u64)
            .map(|i| (hashed_key(i.to_le_bytes()), vec![i as u8; 40]))
            .collect();

        // Reference: one-by-one inserts.
        let mut one = db(cfg.clone());
        for (k, v) in &pairs {
            one.insert(*k, v.clone()).unwrap();
        }

        // Batched in uneven chunks (crosses leaf/split boundaries).
        let mut batched = db(cfg.clone());
        for chunk in pairs.chunks(137) {
            batched.insert_batch(chunk.to_vec()).unwrap();
        }

        assert_eq!(one.root(), batched.root(), "batch root must equal one-by-one");
        for (k, v) in &pairs {
            assert_eq!(batched.get_value(k).unwrap(), Some(v.clone()));
        }

        // Within-batch duplicate: last value wins.
        let key = hashed_key("dup");
        batched
            .insert_batch(vec![(key, b"first".to_vec()), (key, b"second".to_vec())])
            .unwrap();
        assert_eq!(batched.get_value(&key).unwrap(), Some(b"second".to_vec()));
    }

    #[test]
    fn root_is_independent_of_leaf_size() {
        // The Merkle root must be a pure function of the key set — independent of
        // `max_leaf_bytes`, i.e. of where the RAM/disk storage boundary falls.
        // Tiny leaves push almost everything into the RAM frontier (many splits);
        // huge leaves keep it all in one disk subtree. Same keys => same root.
        // (This FAILS under the old split-tag scheme and is the precondition for
        // the paged-node design, where overflow records move that boundary.)
        let tiny = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 1024,
            min_promote_bytes: 256,
        };
        let huge = Config {
            target_leaf_bytes: 32 * 1024,
            max_leaf_bytes: 64 * 1024,
            min_promote_bytes: 16 * 1024,
        };
        let mut a = db(tiny);
        let mut b = db(huge);
        for i in 0..5000u64 {
            let k = hashed_key(i.to_le_bytes());
            a.insert(k, vec![7u8; 32]).unwrap();
            b.insert(k, vec![7u8; 32]).unwrap();
        }
        assert_eq!(
            a.root(),
            b.root(),
            "root depends on leaf size — hash is not storage-independent",
        );
        // Sanity: the two really took different storage paths. Tiny leaves force
        // on-disk overflow chains (>1 read for some keys); huge leaves keep every
        // top-nibble subtree inline in one record (always 1 read).
        let a_max = (0..5000u64)
            .map(|i| a.disk_accesses_for_key(&hashed_key(i.to_le_bytes())).unwrap())
            .max()
            .unwrap();
        assert!(a_max > 1, "tiny-leaf build should use overflow chains, got {a_max}");
        assert_eq!(
            b.disk_accesses_for_key(&hashed_key(0u64.to_le_bytes())).unwrap(),
            1,
            "huge-leaf build should be all-inline",
        );
    }

    #[test]
    fn overflow_node_round_trips_and_hashes_as_its_root() {
        // A branch with one inline leaf child and one Overflow child must:
        //  (a) survive serialize -> deserialize unchanged, and
        //  (b) hash identically whether that child is inline or overflowed
        //      (the Overflow.root equals the inline node's hash).
        let key = hashed_key("x");
        let inline_child = leaf_node(vec![5, 6, 7], leaf_hash(key, [9u8; 32]));
        let inline_hash = hash_node(&inline_child);

        // Build branch B1 with the child inline at slot 3.
        let mut c1 = empty_box_children();
        c1[3] = Some(Box::new(inline_child));
        let branch_inline = make_branch(c1);

        // Build branch B2 with the same child as an Overflow pointer at slot 3.
        let mut c2 = empty_box_children();
        c2[3] = Some(Box::new(Node::Overflow {
            ptr: DiskPtr { page: 1, len: 200 },
            root: inline_hash,
        }));
        let branch_overflow = make_branch(c2);

        // (b) Same branch hash regardless of where the child lives.
        assert_eq!(hash_node(&branch_inline), hash_node(&branch_overflow));

        // (a) Round-trip the overflow-bearing subtree.
        let sub = DiskSubtree { prefix: vec![1, 2], node: branch_overflow };
        let (payload, _) = serialize_subtree(&sub).unwrap();
        let back = deserialize_subtree(&payload[..payload.len()]).unwrap();
        assert_eq!(hash_node(&back.node), hash_node(&sub.node));
        match back.node {
            Node::Branch { children, .. } => match children[3].as_deref() {
                Some(Node::Overflow { ptr, root }) => {
                    assert_eq!(*ptr, DiskPtr { page: 1, len: 200 });
                    assert_eq!(*root, inline_hash);
                }
                other => panic!("slot 3 not an Overflow: {other:?}"),
            },
            other => panic!("not a branch: {other:?}"),
        }
    }

    #[test]
    fn insertion_updates_root_and_value_store() {
        let mut db = db(Config::default());
        let old = db.root();
        let key = hashed_key("alice");
        let new = db.insert(key, b"100".to_vec()).unwrap();
        assert_root_changes(old, new).unwrap();
        assert_eq!(db.get_value(&key).unwrap(), Some(b"100".to_vec()));
        assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
    }

    #[test]
    fn repeated_insert_overwrites_value_hash() {
        let mut db = db(Config::default());
        let key = hashed_key("alice");
        let root1 = db.insert(key, b"100".to_vec()).unwrap();
        let root2 = db.insert(key, b"200".to_vec()).unwrap();
        assert_ne!(root1, root2);
        assert_eq!(db.get_value(&key).unwrap(), Some(b"200".to_vec()));
        assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
    }

    #[test]
    fn persists_and_reopens_frontier() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.flat");
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let keys: Vec<Key> = (0..300u64).map(|i| hashed_key(i.to_le_bytes())).collect();

        let (root, flat_len, free_bytes, ram_nodes) = {
            let mut db = FlatMpt::create(&path, cfg).unwrap();
            for (i, key) in keys.iter().enumerate() {
                db.insert(*key, vec![i as u8; 40]).unwrap();
            }
            db.persist().unwrap();
            (db.root(), db.flat_file_len(), db.free_bytes(), db.ram_nodes())
        }; // drop: close the flat file and RocksDB

        let mut db = FlatMpt::open(&path).unwrap();
        // The frontier, free list, and root all survived the reopen.
        assert_eq!(db.root(), root);
        assert_eq!(db.flat_file_len(), flat_len);
        assert_eq!(db.free_bytes(), free_bytes);
        assert_eq!(db.ram_nodes(), ram_nodes);
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(db.get_value(key).unwrap(), Some(vec![i as u8; 40]));
            // With tiny leaves, some keys sit behind an overflow chain (>=1 read).
            assert!(db.disk_accesses_for_key(key).unwrap() >= 1);
        }

        // And the reopened database is fully writable.
        let new_key = hashed_key("inserted-after-reopen");
        let new_root = db.insert(new_key, b"value".to_vec()).unwrap();
        assert_ne!(new_root, root);
        assert_eq!(db.get_value(&new_key).unwrap(), Some(b"value".to_vec()));
    }

    #[test]
    fn values_round_trip_through_disk_store() {
        let mut db = db(Config::default());
        let key = hashed_key("alice");
        db.insert(key, b"hello world".to_vec()).unwrap();
        // Reads come straight back out of the on-disk RocksDB store.
        assert_eq!(db.get_value(&key).unwrap(), Some(b"hello world".to_vec()));
        assert_eq!(db.get_value(&hashed_key("absent")).unwrap(), None);
    }

    #[test]
    fn reuses_freed_flat_file_space_on_overwrite() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        let keys: Vec<Key> = (0..200u64).map(|i| hashed_key(i.to_le_bytes())).collect();
        for key in &keys {
            db.insert(*key, vec![1; 32]).unwrap();
        }
        let len_after_build = db.flat_file_len();

        // Overwrite every value many times. Each overwrite rewrites a disk
        // subtree of the same size, freeing the old region and reusing it, so
        // the flat file must not keep growing.
        for round in 0..20u8 {
            for key in &keys {
                db.insert(*key, vec![round; 32]).unwrap();
            }
        }
        let len_after_churn = db.flat_file_len();

        // Without reuse this would grow ~20x; allow generous slack for any
        // transient remainder fragments while still proving space is recycled.
        assert!(
            len_after_churn <= len_after_build + len_after_build / 2,
            "flat file grew from {len_after_build} to {len_after_churn} despite reuse"
        );
        // All values are still retrievable and current after the churn.
        for key in &keys {
            assert_eq!(db.get_value(key).unwrap(), Some(vec![19u8; 32]));
        }
    }

    #[test]
    fn splits_large_disk_leaf_into_overflow_records() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        for i in 0..200u64 {
            db.insert(hashed_key(i.to_le_bytes()), vec![i as u8; 32])
                .unwrap();
        }
        // Always-pack: growth is absorbed by on-disk overflow records rather than
        // RAM-branch promotion, so the frontier stays shallow while some keys sit
        // behind an overflow chain (>1 read).
        assert!(
            db.ram_nodes() < 20,
            "frontier should stay shallow, got {}",
            db.ram_nodes()
        );
        let max_reads = (0..200u64)
            .map(|i| db.disk_accesses_for_key(&hashed_key(i.to_le_bytes())).unwrap())
            .max()
            .unwrap();
        assert!(max_reads > 1, "expected overflow chains, max reads={max_reads}");
        // Every key is still reachable.
        for i in [0u64, 33, 99, 199] {
            assert!(db.disk_accesses_for_key(&hashed_key(i.to_le_bytes())).unwrap() >= 1);
        }
    }

    #[test]
    fn long_shared_prefix_does_not_materialize_many_ram_nodes() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        for i in 0..80u8 {
            let mut key = [0u8; 32];
            key[0] = 0xab;
            key[1] = 0xcd;
            key[2] = 0xef;
            key[31] = i;
            db.insert(key, vec![i; 32]).unwrap();
        }
        assert!(db.ram_nodes() < 20, "ram_nodes={}", db.ram_nodes());
    }
}
