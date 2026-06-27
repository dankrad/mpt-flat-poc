use anyhow::{Result, anyhow, bail};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::{
    cell::Cell,
    collections::{BTreeMap, HashMap},
    fs::{File, OpenOptions},
    io::Write,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::Arc,
    sync::Mutex,
    sync::atomic::{AtomicU64, Ordering},
};

/// Number of buffered values before the overlay is flushed to RocksDB as one
/// `WriteBatch`, amortizing per-`put` overhead.
const VALUE_BATCH: usize = 256;

/// Flat-file allocation granularity. Records are page-aligned and occupy a whole
/// number of pages, and `write_payload` zero-pads each record to its full page
/// extent, so every write is a whole, page-aligned device write. 16 KiB matches
/// this SSD's write indirection unit (and the Apple-Silicon OS page): sub-16 KiB
/// writes incur a read-modify-write penalty (~47k IOPS) while full 16 KiB-aligned
/// writes do not (~168k IOPS) — so the page size directly sets the write ceiling.
const PAGE: u64 = 16384;

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

    /// Time spent in `serialize_subtree` (building the record payload — walks the
    /// whole leaf), a subset of the "finalize" bucket. Summed across threads.
    pub static B_SERIALIZE_NS: AtomicU64 = AtomicU64::new(0);
    pub fn on_serialize(ns: u64) {
        B_SERIALIZE_NS.fetch_add(ns, Relaxed);
    }

    /// Split of the Phase-B read (`B_READ_NS`) into the device `pread`
    /// (`B_READ_IO_NS`) and the lazy spine parse (`B_READ_PARSE_NS`). Summed over
    /// threads — exposes how much of "read" is the SSD vs CPU at scale.
    pub static B_READ_IO_NS: AtomicU64 = AtomicU64::new(0);
    pub static B_READ_PARSE_NS: AtomicU64 = AtomicU64::new(0);
    pub fn on_read_io(ns: u64) {
        B_READ_IO_NS.fetch_add(ns, Relaxed);
    }
    pub fn on_read_parse(ns: u64) {
        B_READ_PARSE_NS.fetch_add(ns, Relaxed);
    }

    /// Inline GC: batches that ran a pass, regions reclaimed, records relocated,
    /// and time spent in evacuation (read region + relocate), summed.
    pub static GC_PASSES: AtomicU64 = AtomicU64::new(0);
    pub static GC_REGIONS: AtomicU64 = AtomicU64::new(0);
    pub static GC_RELOCATED: AtomicU64 = AtomicU64::new(0);
    pub static GC_NS: AtomicU64 = AtomicU64::new(0);
    pub fn on_gc(regions: u64, relocated: u64, ns: u64) {
        if regions > 0 || relocated > 0 {
            GC_PASSES.fetch_add(1, Relaxed);
        }
        GC_REGIONS.fetch_add(regions, Relaxed);
        GC_RELOCATED.fetch_add(relocated, Relaxed);
        GC_NS.fetch_add(ns, Relaxed);
    }

    /// Phase-C sub-breakdown (serial): INSTALL = splice each group's result into
    /// the frontier + create structure for brand-new keys, ROOT = recompute the
    /// frontier hashes (`hash_ram` over the invalidated path — the keccac-heavy
    /// part), FLUSH = the value-store flush.
    pub static C_INSTALL_NS: AtomicU64 = AtomicU64::new(0);
    pub static C_ROOT_NS: AtomicU64 = AtomicU64::new(0);
    pub static C_FLUSH_NS: AtomicU64 = AtomicU64::new(0);
    pub fn on_phase_c(install_ns: u64, root_ns: u64, flush_ns: u64) {
        C_INSTALL_NS.fetch_add(install_ns, Relaxed);
        C_ROOT_NS.fetch_add(root_ns, Relaxed);
        C_FLUSH_NS.fetch_add(flush_ns, Relaxed);
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
        C_INSTALL_NS.store(0, Relaxed);
        C_ROOT_NS.store(0, Relaxed);
        C_FLUSH_NS.store(0, Relaxed);
        W_LOCK_NS.store(0, Relaxed);
        W_PWRITE_NS.store(0, Relaxed);
        B_SERIALIZE_NS.store(0, Relaxed);
        B_READ_IO_NS.store(0, Relaxed);
        B_READ_PARSE_NS.store(0, Relaxed);
        GC_PASSES.store(0, Relaxed);
        GC_REGIONS.store(0, Relaxed);
        GC_RELOCATED.store(0, Relaxed);
        GC_NS.store(0, Relaxed);
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
        s += &format!(
            "  gc: passes={} regions_reclaimed={} relocated={} ms={}",
            GC_PASSES.load(Relaxed),
            GC_REGIONS.load(Relaxed),
            GC_RELOCATED.load(Relaxed),
            GC_NS.load(Relaxed) / 1_000_000,
        );
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

/// On-disk record framing: a `u32` little-endian payload length precedes the
/// payload, so GC can scan a region record-by-record without the frontier. The
/// payload itself starts at `offset() + RECORD_HDR`.
const RECORD_HDR: u32 = 4;

impl DiskPtr {
    /// Byte offset of the record's length prefix in the flat file.
    fn offset(&self) -> u64 {
        self.page as u64 * PAGE
    }
    /// Whole pages the framed record (header + payload) occupies.
    fn pages(&self) -> u32 {
        pages_for(RECORD_HDR + self.len)
    }
}

/// Reclaim + write-coalesce unit: the flat file is a sequence of 128 KiB regions
/// (8 × 16 KiB pages). 128 KiB sits on the device's write-bandwidth plateau and a
/// whole-region write is 16 KiB-aligned (no sub-page RMW penalty); a region is
/// also the unit GC reclaims.
const REGION_PAGES: u64 = 8;

/// Inline-GC controller (see GC_DESIGN.md). The cleaning rate `R` (victim regions
/// per batch) is adjusted each batch to hold `live / active` at `TARGET_UTIL`.
const TARGET_UTIL: f64 = 0.60;
/// Proportional gain: regions added to `R` per unit of utilization error. A 10%
/// miss moves `R` by ~`GC_GAIN/10`. Tunable.
const GC_GAIN: f64 = 4000.0;
/// Per-batch cap on regions evacuated (bounds the GC stall). Tunable.
const GC_R_MAX: usize = 8192;
/// Don't GC until the file is at least this big (avoid churn on tiny files).
const GC_MIN_PAGES: u64 = 64 * REGION_PAGES;
/// Never evacuate a region fuller than this (relocation cost not worth the space).
const EVAC_MAX_UTIL: f64 = 0.75;

/// Log-structured page allocator over fixed regions. New records append to a
/// moving head region (sequential, coalesced writes); per-region live-page counts
/// let space be reclaimed a whole region at a time. A region whose live count
/// reaches zero returns to `free_regions` and is reused before the file is
/// extended, so the file size stays bounded. (Stage 1a reclaims only regions that
/// become fully dead on their own; Stage 1b adds active evacuation.)
///
/// Not serialized — recomputed from the frontier on [`FlatMpt::open`].
#[derive(Debug, Default)]
struct RegionAlloc {
    /// Live pages per region; index = region number = page / REGION_PAGES.
    live: Vec<u32>,
    /// Batch epoch each region was last opened for writing — drives age-aware
    /// (cost-benefit) victim selection.
    epoch_of: Vec<u32>,
    /// Current batch epoch (bumped once per batch).
    epoch: u32,
    /// Region currently being appended to.
    head_region: u64,
    /// Next free page (absolute) within the head region.
    next_page: u64,
    /// Fully-dead regions, reusable as the next head.
    free_regions: Vec<u64>,
}

impl RegionAlloc {
    fn region_of(page: u64) -> u64 {
        page / REGION_PAGES
    }

    fn ensure_region(&mut self, r: u64) {
        if r as usize >= self.live.len() {
            self.live.resize(r as usize + 1, 0);
            self.epoch_of.resize(r as usize + 1, 0);
        }
    }

    /// Advance the batch epoch (called once per `insert_batch`).
    fn bump_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Open a new head region: reuse a reclaimed one if available, else a fresh
    /// region at the current file end. Stamps the region with the current epoch.
    fn open_new_head(&mut self, end_page: &AtomicU64) {
        let r = self
            .free_regions
            .pop()
            .unwrap_or_else(|| end_page.load(Ordering::SeqCst).div_ceil(REGION_PAGES));
        self.head_region = r;
        self.next_page = r * REGION_PAGES;
        self.ensure_region(r);
        self.epoch_of[r as usize] = self.epoch;
    }

    /// Reserve `pages` consecutive pages at the head, opening a new head region
    /// first if the run wouldn't fit. The whole run stays within one region, so
    /// each record's region (`page / REGION_PAGES`) is well-defined.
    fn alloc(&mut self, pages: u32, end_page: &AtomicU64) -> u64 {
        debug_assert!(pages as u64 <= REGION_PAGES);
        let region_end = self.head_region * REGION_PAGES + REGION_PAGES;
        if self.live.is_empty() || self.next_page + pages as u64 > region_end {
            self.open_new_head(end_page);
        }
        let page = self.next_page;
        self.next_page += pages as u64;
        self.live[self.head_region as usize] += pages;
        end_page.fetch_max(self.next_page, Ordering::SeqCst);
        page
    }

    /// Mark a record's pages dead; reclaim the region once it is fully dead.
    fn free(&mut self, page: u64, pages: u32) {
        let r = Self::region_of(page) as usize;
        if r >= self.live.len() {
            return;
        }
        let was = self.live[r];
        self.live[r] = was.saturating_sub(pages);
        if was > 0 && self.live[r] == 0 && r as u64 != self.head_region {
            self.free_regions.push(r as u64);
        }
    }

    fn live_pages(&self) -> u64 {
        self.live.iter().map(|&p| p as u64).sum()
    }

    fn free_region_pages(&self) -> u64 {
        self.free_regions.len() as u64 * REGION_PAGES
    }

    /// Pick up to `max` victim regions by **cost-benefit** score (Sprite LFS):
    /// `score = (1 - u) * age / (1 + u)`, where `u` is the region's utilization and
    /// `age` is batches since it was written. This favors *old, settled* regions
    /// that are mostly garbage (lasting reclaim, little relocation) and skips
    /// freshly-emptied *hot* regions — which, in a uniform build, empty themselves
    /// as their leaves migrate to the head and get reclaimed for free. Excludes the
    /// head, already-free, full, and too-full (`> EVAC_MAX_UTIL`) regions. Simple
    /// O(regions) scan; bucket in a later pass if it shows up.
    fn select_victims(&self, max: usize) -> Vec<u64> {
        if max == 0 {
            return Vec::new();
        }
        let cap_live = (REGION_PAGES as f64 * EVAC_MAX_UTIL) as u32;
        let free: std::collections::HashSet<u64> = self.free_regions.iter().copied().collect();
        let mut cands: Vec<(f64, u64)> = self
            .live
            .iter()
            .enumerate()
            .filter_map(|(r, &live)| {
                let r = r as u64;
                if live == 0 || live > cap_live || r == self.head_region || free.contains(&r) {
                    return None;
                }
                let u = live as f64 / REGION_PAGES as f64;
                let age = self.epoch.wrapping_sub(self.epoch_of[r as usize]) as f64;
                let score = (1.0 - u) * age / (1.0 + u);
                Some((score, r))
            })
            .collect();
        // Highest score first.
        cands.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        cands.truncate(max);
        cands.into_iter().map(|(_, r)| r).collect()
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
    seg: Mutex<RegionAlloc>,
    /// High-water mark in pages: one past the largest page ever written. Atomic so
    /// reporting can read it without the region lock.
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
            seg: Mutex::new(RegionAlloc::default()),
            end_page: AtomicU64::new(0),
        }
    }

    /// Append an already-encoded subtree payload as a page-aligned record at the
    /// head region (sequential placement). Written verbatim (no length prefix — its
    /// size lives in the returned [`DiskPtr`]) in one positioned `pwrite` of
    /// `ceil(len/PAGE)` whole pages; reads fetch exactly `len` bytes, so the padded
    /// tail needn't be zeroed beyond the payload. Safe to call concurrently: the
    /// allocation holds the region lock only briefly; the `pwrite` is lock-free.
    fn write_payload(&self, payload: &[u8]) -> Result<DiskPtr> {
        let total = payload.len() as u32;
        stats::on_write(total as usize);
        let pages = pages_for(RECORD_HDR + total);
        let lt = std::time::Instant::now();
        let page = self.seg.lock().unwrap().alloc(pages, &self.end_page);
        stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
        if page + pages as u64 > u32::MAX as u64 {
            bail!("flat file exceeds the 16 TiB DiskPtr addressing limit");
        }
        let page = page as u32;

        // Frame: [u32 payload len][payload][zero pad to page].
        let mut record = vec![0u8; pages as usize * PAGE as usize];
        record[..4].copy_from_slice(&total.to_le_bytes());
        record[4..4 + payload.len()].copy_from_slice(payload);

        let _g = prof::scope(prof::Cat::FileWrite);
        let wt = std::time::Instant::now();
        (&self.file).write_all_at(&record, page as u64 * PAGE)?;
        stats::on_pwrite(wt.elapsed().as_nanos() as u64);
        Ok(DiskPtr { page, len: total })
    }

    /// Coalesce several records into contiguous appended `pwrite`s, each ≤ one
    /// region (so a write is 16 KiB-aligned and stays on the bandwidth plateau, and
    /// no record straddles a region boundary). Returns a `DiskPtr` per payload.
    fn write_batch(&self, payloads: &[&[u8]]) -> Result<Vec<DiskPtr>> {
        let mut ptrs = Vec::with_capacity(payloads.len());
        let mut i = 0;
        while i < payloads.len() {
            // Take a run of records whose page-sum fits in one region.
            let mut run_pages = 0u32;
            let mut j = i;
            while j < payloads.len() {
                let pc = pages_for(RECORD_HDR + payloads[j].len() as u32);
                if j > i && run_pages + pc > REGION_PAGES as u32 {
                    break;
                }
                run_pages += pc;
                j += 1;
            }
            let lt = std::time::Instant::now();
            let page_start = self.seg.lock().unwrap().alloc(run_pages, &self.end_page);
            stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
            if page_start + run_pages as u64 > u32::MAX as u64 {
                bail!("flat file exceeds the 16 TiB DiskPtr addressing limit");
            }
            let mut buf = vec![0u8; run_pages as usize * PAGE as usize];
            let mut page = page_start;
            let mut off = 0usize;
            for p in &payloads[i..j] {
                let pc = pages_for(RECORD_HDR + p.len() as u32);
                // Frame: [u32 payload len][payload], page-aligned within the run.
                buf[off..off + 4].copy_from_slice(&(p.len() as u32).to_le_bytes());
                buf[off + 4..off + 4 + p.len()].copy_from_slice(p);
                ptrs.push(DiskPtr {
                    page: page as u32,
                    len: p.len() as u32,
                });
                stats::on_write(p.len());
                page += pc as u64;
                off += pc as usize * PAGE as usize;
            }
            let _g = prof::scope(prof::Cat::FileWrite);
            let wt = std::time::Instant::now();
            (&self.file).write_all_at(&buf, page_start * PAGE)?;
            stats::on_pwrite(wt.elapsed().as_nanos() as u64);
            i = j;
        }
        Ok(ptrs)
    }

    fn read(&self, ptr: DiskPtr) -> Result<DiskSubtree> {
        read_record(&self.file, ptr)
    }

    /// Lazy read: parse only the spine; child subtrees stay `Raw`. Used by the
    /// insert path, where a record is touched on one key's path per call.
    fn read_lazy(&self, ptr: DiskPtr) -> Result<DiskSubtree> {
        let mut record = vec![0u8; ptr.len as usize];
        {
            let _g = prof::scope(prof::Cat::FileRead);
            let it = std::time::Instant::now();
            // Payload starts just past the framing header.
            (&self.file).read_exact_at(&mut record, ptr.offset() + RECORD_HDR as u64)?;
            stats::on_read_io(it.elapsed().as_nanos() as u64);
        }
        let _g = prof::scope(prof::Cat::Deserialize);
        // `Arc::from(Vec)` reuses the allocation (no copy); Raw children then
        // share it as zero-copy slices.
        let pt = std::time::Instant::now();
        let out = deserialize_subtree_lazy(Arc::from(record));
        stats::on_read_parse(pt.elapsed().as_nanos() as u64);
        out
    }

    /// Read a record's exact payload bytes (no parse) — used by GC to relocate a
    /// record verbatim. The relocated copy is re-framed by `write_payload`.
    fn read_raw(&self, ptr: DiskPtr) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; ptr.len as usize];
        (&self.file).read_exact_at(&mut buf, ptr.offset() + RECORD_HDR as u64)?;
        Ok(buf)
    }

    /// Read one whole region (`REGION_PAGES` pages) for GC scanning. Pages past the
    /// file end read as zeros (sparse), which the scan treats as an empty tail.
    fn read_region(&self, region: u64) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; REGION_PAGES as usize * PAGE as usize];
        let off = region * REGION_PAGES * PAGE;
        // Tolerate a short read at the very end of the file (sparse tail).
        match (&self.file).read_exact_at(&mut buf, off) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {}
            Err(e) => return Err(e.into()),
        }
        Ok(buf)
    }

    fn free(&self, ptr: DiskPtr) {
        let lt = std::time::Instant::now();
        self.seg.lock().unwrap().free(ptr.page as u64, ptr.pages());
        stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
    }

    fn end_page(&self) -> u64 {
        self.end_page.load(Ordering::SeqCst)
    }

    /// Live pages and pages held in reclaimed-but-unused regions.
    fn live_and_free_pages(&self) -> (u64, u64) {
        let seg = self.seg.lock().unwrap();
        (seg.live_pages(), seg.free_region_pages())
    }

    /// Total dead pages in the file (everything not currently live).
    fn garbage_pages(&self) -> u64 {
        let (live, _) = self.live_and_free_pages();
        self.end_page().saturating_sub(live)
    }

    /// Number of fully-reclaimed regions available for reuse.
    fn free_region_count(&self) -> usize {
        self.seg.lock().unwrap().free_regions.len()
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
        // Payload starts just past the framing header.
        file.read_exact_at(&mut record, ptr.offset() + RECORD_HDR as u64)?;
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
            target_leaf_bytes: 8 * 1024,
            max_leaf_bytes: 16 * 1024,
            min_promote_bytes: 8 * 1024,
        }
    }
}

// Each non-trivial node caches its own Merkle hash, computed once at
// construction and persisted to disk. This lets a rewrite recompute only the
// hashes on the path it actually changed (see `node_insert`), instead of
// re-hashing the whole subtree. All keys are full 64-nibble paths, so leaves
// only ever sit at depth 64 and branches never carry a value.
// Not serde-serialized: the flat-file format is the custom `write_node`/lazy
// reader, and the manifest stores `RamNode`. (Raw holds an `Arc`, which serde
// wouldn't derive anyway.)
#[derive(Debug, Clone)]
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
    /// A still-serialized child subtree: a zero-copy `[off, off+len)` slice of the
    /// shared record buffer plus its root `hash`. Produced by the *lazy* reader so
    /// that, to change one key, we only parse the nodes on that key's path —
    /// untouched sibling subtrees stay `Raw` (no byte copy on read) and are written
    /// back verbatim. Expanded one level on demand by `record_node_insert`.
    Raw {
        buf: Arc<[u8]>,
        off: usize,
        len: usize,
        hash: Hash,
    },
}

#[derive(Debug, Clone)]
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
/// A frontier node's cached hash. It's a plain `Cell` (no atomic overhead on the
/// hot serial path), but declared `Sync` so the root re-hash can run across
/// threads: the frontier is a tree of uniquely-owned (`Box`) nodes, so when we
/// split it into disjoint subtrees each node's cache is touched by exactly one
/// thread — there is never concurrent access to the same cell.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct HashCell(Cell<Option<Hash>>);

// SAFETY: the only multi-threaded reader is `hash_ram_parallel`, which hands each
// thread a disjoint subtree of the uniquely-owned frontier; no two threads ever
// reach the same `HashCell`.
unsafe impl Sync for HashCell {}

impl HashCell {
    fn new(v: Option<Hash>) -> Self {
        HashCell(Cell::new(v))
    }
    fn get(&self) -> Option<Hash> {
        self.0.get()
    }
    fn set(&self, v: Option<Hash>) {
        self.0.set(v)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RamNode {
    Empty,
    Extension {
        path: Vec<u8>,
        child: Box<RamNode>,
        hash: HashCell,
    },
    // No `value`: keys are full 64-nibble paths, so none ever terminates at a
    // (necessarily shallower) frontier branch.
    Branch {
        children: [Option<RamChild>; 16],
        hash: HashCell,
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
    /// Inline-GC cleaning rate: victim regions to evacuate per batch, adjusted by
    /// the proportional controller to hold utilization at `TARGET_UTIL`.
    gc_regions: usize,
}

/// On-disk checkpoint of everything that otherwise lives only in RAM: the trie
/// frontier and the high-water mark. The region allocator's liveness is recomputed
/// from the frontier on [`FlatMpt::open`] (the frontier is the source of truth for
/// which records are live), so the manifest stays minimal and format-stable.
#[derive(Serialize)]
struct ManifestRef<'a> {
    cfg: &'a Config,
    upper: &'a RamNode,
    end_page: u64,
}

#[derive(Deserialize)]
struct Manifest {
    cfg: Config,
    upper: RamNode,
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
            gc_regions: 0,
        })
    }

    /// Reopen a database previously written with [`FlatMpt::persist`]. Reattaches
    /// to the existing flat file and value store (no truncation), restores the RAM
    /// frontier, and rebuilds the region allocator's liveness by walking the
    /// frontier. A fresh head region is opened past the file end so appends resume
    /// cleanly (wasting at most the tail of the last region).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let meta = meta_path(path);
        let bytes = std::fs::read(&meta)
            .map_err(|e| anyhow!("no manifest at {}: {e}", meta.display()))?;
        let Manifest {
            cfg,
            upper,
            end_page,
        } = bincode::deserialize(&bytes)?;

        // Rebuild per-region liveness from the frontier, then open a fresh head
        // region at the next region boundary past the file end.
        let num_regions = end_page.div_ceil(REGION_PAGES);
        let mut alloc = RegionAlloc {
            live: vec![0u32; num_regions as usize],
            // Reopened regions are "old" (epoch 0); new regions get growing epochs,
            // so the existing data ages relative to fresh writes.
            epoch_of: vec![0u32; num_regions as usize],
            ..RegionAlloc::default()
        };
        recompute_live(&upper, &mut alloc.live);
        alloc.head_region = num_regions;
        alloc.next_page = num_regions * REGION_PAGES;
        alloc.ensure_region(alloc.head_region);
        let new_end = alloc.next_page;

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let values = DB::open(&opts, values_path(path))?;

        Ok(Self {
            cfg,
            store: FlatFile {
                file,
                seg: Mutex::new(alloc),
                end_page: AtomicU64::new(new_end),
            },
            upper,
            values,
            overlay: HashMap::new(),
            path: path.to_path_buf(),
            gc_regions: 0,
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
        let manifest = ManifestRef {
            cfg: &self.cfg,
            upper: &self.upper,
            end_page: self.store.end_page(),
        };
        let bytes = bincode::serialize(&manifest)?;

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
        // Advance the GC epoch so regions written this batch are "age 0" and the
        // cost-benefit cleaner leaves them alone until they settle.
        self.store.seg.lock().unwrap().bump_epoch();
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
        let batched = batched_writes();
        let results: Vec<(Key, RamChild)> = if groups.len() < 64 {
            process_chunk(store, &cfg, &groups, batched)?
        } else {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8);
            let chunk = groups.len().div_ceil(threads);
            std::thread::scope(|scope| {
                let handles: Vec<_> = groups
                    .chunks(chunk)
                    .map(|c| scope.spawn(|| process_chunk(store, &cfg, c, batched)))
                    .collect();
                let mut out = Vec::with_capacity(groups.len());
                for h in handles {
                    out.extend(h.join().expect("batch group thread panicked")?);
                }
                Ok::<_, anyhow::Error>(out)
            })?
        };

        let b_ns = t_b.elapsed().as_nanos() as u64;

        // Inline GC (after Phase B, so this batch's foreground frees are already
        // applied): evacuate the live records out of the emptiest regions and
        // collect their relocations to install alongside the foreground results.
        // The pages this batch is already rewriting are skipped (deduped).
        let fg_pages: std::collections::HashSet<u32> =
            groups.iter().map(|(ptr, _, _)| ptr.page).collect();
        let t_gc = std::time::Instant::now();
        let r = self.gc_rate();
        let victims = self.store.seg.lock().unwrap().select_victims(r);
        let reloc = if victims.is_empty() {
            Vec::new()
        } else {
            evacuate_regions(&self.store, &self.upper, &victims, &fg_pages)?
        };
        stats::on_gc(victims.len() as u64, reloc.len() as u64, t_gc.elapsed().as_nanos() as u64);

        let t_c = std::time::Instant::now();

        // Phase C (serial): splice each group's result into the frontier, retarget
        // the relocated records' pointers, then create structure for the brand-new
        // keys. Recompute the root once.
        for (rep, new_child) in results {
            install_at_key(&mut self.upper, &rep, 0, new_child);
        }
        for (prefix, new_ptr) in reloc {
            install_ptr_by_prefix(&mut self.upper, &prefix, 0, new_ptr);
        }
        for (key, value_hash) in fresh {
            insert_ram(&self.store, &cfg, &mut self.upper, Vec::new(), key, value_hash)?;
        }
        let install_ns = t_c.elapsed().as_nanos() as u64;
        let t_flush = std::time::Instant::now();
        self.store.flush()?;
        let flush_ns = t_flush.elapsed().as_nanos() as u64;
        let t_root = std::time::Instant::now();
        let root = self.root();
        let root_ns = t_root.elapsed().as_nanos() as u64;
        stats::on_batch(a_ns, b_ns, t_c.elapsed().as_nanos() as u64);
        stats::on_phase_c(install_ns, root_ns, flush_ns);
        Ok(root)
    }

    /// Proportional controller for the inline-GC cleaning rate. Nudges
    /// `gc_regions` (victims/batch) toward holding `live / active` at
    /// `TARGET_UTIL`: below target ⇒ too much garbage ⇒ clean more; above ⇒ ease
    /// off. Returns the rate to use this batch (0 until the file passes the floor).
    fn gc_rate(&mut self) -> usize {
        let end = self.store.end_page();
        if end < GC_MIN_PAGES {
            return 0;
        }
        let (live, free_pages) = self.store.live_and_free_pages();
        let active = end.saturating_sub(free_pages).max(1);
        let u = live as f64 / active as f64;
        let adj = ((TARGET_UTIL - u) * GC_GAIN).round() as i64;
        let r = (self.gc_regions as i64 + adj).clamp(0, GC_R_MAX as i64) as usize;
        self.gc_regions = r;
        r
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
        hash_ram_parallel(&self.upper)
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

    /// Logical size of the flat file (high-water mark). Under the log-structured
    /// allocator it grows with garbage until regions are reclaimed and reused.
    pub fn flat_file_len(&self) -> u64 {
        self.store.end_page() * PAGE
    }

    /// Dead (non-live) bytes in the flat file. (Kept under the historical name.)
    pub fn free_bytes(&self) -> u64 {
        self.store.garbage_pages() * PAGE
    }

    /// Number of fully-reclaimed regions available for reuse.
    pub fn free_regions(&self) -> usize {
        self.store.free_region_count()
    }

    /// Heap held by the in-RAM index — the part of the database that is *not*
    /// on disk: the trie frontier, the region allocator, and the unflushed value
    /// overlay. Excludes the OS page cache and RocksDB's own (C++) memory.
    pub fn ram_report(&self) -> RamReport {
        let frontier_nodes = count_ram_nodes(&self.upper);
        let frontier_bytes = frontier_bytes(&self.upper);
        let free_regions = self.store.free_region_count();
        // The region allocator is one u32 of live-count per region; report that.
        let free_list_bytes =
            self.store.seg.lock().unwrap().live.len() * std::mem::size_of::<u32>();
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
                hash: HashCell::new(None),
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
                        hash: HashCell::new(None),
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
                    hash: HashCell::new(None),
                };
                *node = if common == 0 {
                    branch
                } else {
                    RamNode::Extension {
                        path: old_path[..common].to_vec(),
                        child: Box::new(branch),
                        hash: HashCell::new(None),
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
                    let mut subtree = store.read_lazy(*ptr)?;
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
    // RLP-style sparse encoding: a 16-bit presence bitmap followed by only the
    // present children's hashes, so an absent slot costs ~0 bytes instead of a
    // 32-byte `empty_hash`. The bitmap + ordered hashes still uniquely determine
    // the branch, so this stays collision-resistant; it just shrinks the keccak
    // input (513 bytes fixed -> 3 + 32*popcount) for the sparse branches that
    // dominate, cutting the path-rehash work.
    let mut bytes = vec![5];
    let bitmap = branch_bitmap(children.iter().map(|c| c.is_some()));
    bytes.extend_from_slice(&bitmap.to_le_bytes());
    for child in children.iter().flatten() {
        bytes.extend_from_slice(&hash_node(child));
    }
    keccak(&bytes)
}

/// Pack a child-presence iterator (16 slots) into a little-endian bitmap.
fn branch_bitmap(present: impl Iterator<Item = bool>) -> u16 {
    let mut bitmap = 0u16;
    for (i, p) in present.enumerate() {
        if p {
            bitmap |= 1 << i;
        }
    }
    bitmap
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
        // `follow_key` runs only on fully-parsed records (the disk_accesses probe).
        Node::Raw { .. } => unreachable!("follow_key on a lazily-parsed record"),
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
    // Expand a lazily-unparsed subtree one level before navigating into it
    // (children become `Raw` again, so deeper untouched subtrees stay unparsed).
    if let Node::Raw { buf, off, len, .. } = node {
        *node = parse_node_lazy(buf, *off, *len)?;
    }
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
                    let mut sub = store.read_lazy(*ptr)?;
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
        Node::Raw { .. } => unreachable!("Raw is expanded before the match"),
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
        hash: HashCell::new(None),
    };
    if ext_path.is_empty() {
        Ok(RamChild::Ram(Box::new(branch)))
    } else {
        Ok(RamChild::Ram(Box::new(RamNode::Extension {
            path: ext_path,
            child: Box::new(branch),
            hash: HashCell::new(None),
        })))
    }
}

/// Outcome of applying a group's keys, before the new leaf record is written.
enum GroupOut {
    /// The record was promoted into a RAM frontier branch (no deferred write).
    Promoted(RamChild),
    /// The rewritten leaf record's payload + its root, to be written by the caller
    /// (in place, or coalesced into a batched contiguous append).
    Leaf { payload: Vec<u8>, root: Hash },
}

/// Coalesce up to this many leaf writes per pending flush. `write_batch` further
/// splits a flush so no single `pwrite` exceeds one region (`REGION_PAGES`).
const BATCH_LEAVES: usize = 8;

/// Whether to coalesce leaf writes into batched `pwrite`s. The append allocator
/// places every write sequentially regardless; batching just cuts syscalls. On by
/// default; `MPT_BATCHED_WRITES=0` writes per-record for comparison.
fn batched_writes() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_BATCHED_WRITES").ok().as_deref() != Some("0"))
}

/// Apply a whole group of keys (all routing to the disk record at `ptr`), free
/// the old record, and return the outcome — *without* writing the new leaf (the
/// caller writes it, in place or batched). Reads/writes only the disjoint record
/// subtree via the thread-safe `store`, so groups run concurrently.
fn process_group(
    store: &FlatFile,
    cfg: &Config,
    ptr: DiskPtr,
    keys: &[(Key, Hash)],
) -> Result<GroupOut> {
    let t = std::time::Instant::now();
    let mut subtree = store.read_lazy(ptr)?;
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
        GroupOut::Promoted(promote_record_to_ram(store, subtree)?)
    } else {
        let st = std::time::Instant::now();
        let (payload, _) = serialize_subtree(&subtree)?;
        stats::on_serialize(st.elapsed().as_nanos() as u64);
        store.free(ptr);
        GroupOut::Leaf {
            payload,
            root: hash_node(&subtree.node),
        }
    };
    stats::on_group(read_ns, rebuild_ns, t.elapsed().as_nanos() as u64);
    Ok(out)
}

/// Process a chunk of groups, returning the replacement `(rep_key, RamChild)` for
/// each. With `batched`, leaf writes are coalesced into contiguous appended
/// batches of `BATCH_LEAVES` (one `pwrite` per batch); otherwise each is written
/// in place (reusing freed space).
fn process_chunk(
    store: &FlatFile,
    cfg: &Config,
    chunk: &[(DiskPtr, Key, Vec<(Key, Hash)>)],
    batched: bool,
) -> Result<Vec<(Key, RamChild)>> {
    let mut out = Vec::with_capacity(chunk.len());
    let mut pending: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
    for (ptr, rep, keys) in chunk {
        match process_group(store, cfg, *ptr, keys)? {
            GroupOut::Promoted(rc) => out.push((*rep, rc)),
            GroupOut::Leaf { payload, root } if batched => {
                pending.push((*rep, root, payload));
                if pending.len() >= BATCH_LEAVES {
                    flush_leaf_batch(store, &mut pending, &mut out)?;
                }
            }
            GroupOut::Leaf { payload, root } => {
                let new_ptr = store.write_payload(&payload)?;
                out.push((*rep, RamChild::Disk { ptr: new_ptr, root }));
            }
        }
    }
    flush_leaf_batch(store, &mut pending, &mut out)?;
    Ok(out)
}

/// Write all pending leaf payloads as one contiguous appended record batch.
fn flush_leaf_batch(
    store: &FlatFile,
    pending: &mut Vec<(Key, Hash, Vec<u8>)>,
    out: &mut Vec<(Key, RamChild)>,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let ptrs = {
        let payloads: Vec<&[u8]> = pending.iter().map(|(_, _, p)| p.as_slice()).collect();
        store.write_batch(&payloads)?
    };
    for ((rep, root, _), ptr) in pending.drain(..).zip(ptrs) {
        out.push((rep, RamChild::Disk { ptr, root }));
    }
    Ok(())
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
            // tag + bitmap + hash + child-length table (u32 per present child) + children.
            let n = children.iter().flatten().count();
            1 + 2 + 32 + n * 4 + children.iter().flatten().map(|c| node_size(c)).sum::<usize>()
        }
        Node::Overflow { .. } => 1 + 4 + 4 + 32,
        // Raw is already serialized — its byte length is its on-disk size.
        Node::Raw { len, .. } => *len,
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
            // Child-length table: a u32 per present child, so a reader can jump to
            // child `i` by summing lengths 0..i instead of scanning siblings. Not
            // hashed (the branch hash is over the child digests), so it doesn't
            // affect the root. Reserve the slots, then backfill from the serialized
            // lengths as each child is written.
            let n = bitmap.count_ones() as usize;
            let table_pos = out.len();
            out.resize(table_pos + n * 4, 0);
            let mut ti = 0;
            for child in children.iter().flatten() {
                let start = out.len();
                write_node(out, child)?;
                let len = (out.len() - start) as u32;
                out[table_pos + ti * 4..table_pos + ti * 4 + 4].copy_from_slice(&len.to_le_bytes());
                ti += 1;
            }
        }
        Node::Overflow { ptr, root } => {
            out.push(4);
            out.extend_from_slice(&ptr.page.to_le_bytes());
            out.extend_from_slice(&ptr.len.to_le_bytes());
            out.extend_from_slice(root);
        }
        // A Raw subtree is already its own `write_node` bytes — emit verbatim.
        Node::Raw { buf, off, len, .. } => out.extend_from_slice(&buf[*off..*off + *len]),
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
                // Skip the child-length table; the full parse reads children
                // sequentially and doesn't need it.
                let n = bitmap.count_ones() as usize;
                let _ = self.read_bytes(n * 4)?;
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

/// Lazy reader over a shared `Arc<[u8]>` record buffer. Positions are absolute
/// offsets into the buffer, so `Raw` children are zero-copy `(buf, off, len)`
/// slices — no per-sibling byte copy on read.
struct LazyReader {
    buf: Arc<[u8]>,
    pos: usize,
}

impl LazyReader {
    fn new(buf: Arc<[u8]>) -> Self {
        Self { buf, pos: 0 }
    }
    fn at(buf: Arc<[u8]>, pos: usize) -> Self {
        Self { buf, pos }
    }

    fn take(&mut self, len: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow!("compact subtree offset overflow"))?;
        if end > self.buf.len() {
            bail!("truncated compact subtree record");
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn peek_u8(&self) -> Result<u8> {
        self.buf
            .get(self.pos)
            .copied()
            .ok_or_else(|| anyhow!("truncated compact subtree record"))
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn hash(&mut self) -> Result<Hash> {
        let mut h = [0u8; 32];
        h.copy_from_slice(self.take(32)?);
        Ok(h)
    }
    fn nibble_path(&mut self) -> Result<Vec<u8>> {
        let len = self.u8()? as usize;
        if len > 64 {
            bail!("compact subtree nibble path too long");
        }
        let mut path = Vec::with_capacity(len);
        for _ in 0..len.div_ceil(2) {
            let byte = self.u8()?;
            path.push(byte >> 4);
            if path.len() < len {
                path.push(byte & 0x0f);
            }
        }
        Ok(path)
    }

    /// Parse this node and its extension/branch spine; branch child subtrees stay
    /// `Raw`. Children are located via the branch's length table (no sibling scan).
    fn node(&mut self) -> Result<Node> {
        match self.u8()? {
            0 => Ok(Node::Empty),
            1 => {
                let path = self.nibble_path()?;
                let hash = self.hash()?;
                Ok(Node::Leaf { path, hash })
            }
            2 => {
                let path = self.nibble_path()?;
                let hash = self.hash()?;
                let child = Box::new(self.node()?);
                Ok(Node::Extension { path, child, hash })
            }
            3 => {
                let bitmap = self.u16()?;
                let hash = self.hash()?;
                let n = bitmap.count_ones() as usize;
                let mut lens = [0u32; 16];
                for l in lens.iter_mut().take(n) {
                    *l = self.u32()?;
                }
                let mut children = empty_box_children();
                let mut ti = 0;
                for (idx, slot) in children.iter_mut().enumerate() {
                    if bitmap & (1 << idx) == 0 {
                        continue;
                    }
                    let len = lens[ti] as usize;
                    ti += 1;
                    // ext/branch subtrees become zero-copy `Raw` — jump over them via
                    // the table (no scan), reading only the child's header hash. Small
                    // terminal nodes (leaf/overflow/empty) are parsed fully.
                    match self.peek_u8()? {
                        2 | 3 => {
                            let off = self.pos;
                            let hash = extract_hash(&self.buf[off..off + len])?;
                            self.pos += len;
                            *slot = Some(Box::new(Node::Raw {
                                buf: self.buf.clone(),
                                off,
                                len,
                                hash,
                            }));
                        }
                        _ => *slot = Some(Box::new(self.node()?)),
                    }
                }
                Ok(Node::Branch { children, hash })
            }
            4 => {
                let page = self.u32()?;
                let len = self.u32()?;
                let root = self.hash()?;
                Ok(Node::Overflow {
                    ptr: DiskPtr { page, len },
                    root,
                })
            }
            tag => bail!("invalid compact subtree node tag {tag}"),
        }
    }
}

/// Read a node's root hash from the front of its serialized `bytes` — a shallow
/// header parse (tag + path/bitmap), no recursion into the subtree.
fn extract_hash(bytes: &[u8]) -> Result<Hash> {
    let mut r = CompactReader::new(bytes);
    match r.read_u8()? {
        0 => Ok(empty_hash()),
        1 | 2 => {
            let _ = r.read_nibble_path()?;
            r.read_hash()
        }
        3 => {
            let _ = r.read_u16()?;
            r.read_hash()
        }
        4 => {
            let _ = r.read_bytes(4 + 4)?;
            r.read_hash()
        }
        tag => bail!("invalid compact subtree node tag {tag}"),
    }
}

/// Expand a `Raw` one level: parse the node at `buf[off..off+len]`, leaving its
/// children `Raw` over the same shared buffer.
fn parse_node_lazy(buf: &Arc<[u8]>, off: usize, len: usize) -> Result<Node> {
    let mut r = LazyReader::at(buf.clone(), off);
    let node = r.node()?;
    if r.pos != off + len {
        bail!("trailing bytes in raw node");
    }
    Ok(node)
}

/// Like [`deserialize_subtree`], but only parses the spine down to the top
/// branch; child subtrees stay `Raw` (zero-copy slices of the record buffer).
fn deserialize_subtree_lazy(buf: Arc<[u8]>) -> Result<DiskSubtree> {
    let end = buf.len();
    let mut r = LazyReader::new(buf);
    let prefix = r.nibble_path()?;
    let node = r.node()?;
    if r.pos != end {
        bail!("trailing bytes in flat-file record");
    }
    Ok(DiskSubtree { prefix, node })
}

/// Parse just a record payload's leading nibble-path (its `prefix`) — the path to
/// its slot in the frontier — without parsing the subtree. Used by GC to locate a
/// scanned record's frontier pointer.
fn parse_prefix(payload: &[u8]) -> Result<Vec<u8>> {
    CompactReader::new(payload).read_nibble_path()
}

/// Retarget the `DiskPtr` at `prefix`'s frontier slot to `new_ptr`, leaving its
/// cached hash untouched (relocation is verbatim ⇒ the subtree root is unchanged).
/// Returns whether a `Disk` slot was found and updated.
fn install_ptr_by_prefix(node: &mut RamNode, prefix: &[u8], depth: usize, new_ptr: DiskPtr) -> bool {
    match node {
        RamNode::Empty => false,
        RamNode::Extension { path, child, .. } => {
            if prefix.get(depth..depth + path.len()) == Some(path.as_slice()) {
                install_ptr_by_prefix(child, prefix, depth + path.len(), new_ptr)
            } else {
                false
            }
        }
        RamNode::Branch { children, .. } => {
            let idx = match prefix.get(depth) {
                Some(&i) => i as usize,
                None => return false,
            };
            match children[idx].as_mut() {
                Some(RamChild::Disk { ptr, .. }) => {
                    *ptr = new_ptr;
                    true
                }
                Some(RamChild::Ram(child)) => {
                    install_ptr_by_prefix(child, prefix, depth + 1, new_ptr)
                }
                None => false,
            }
        }
    }
}

/// Evacuate the live records out of `victims` (inline GC). For each victim region:
/// one sequential read, then scan its framed records; relocate each *live*,
/// non-foreground record verbatim (so its hash/root is unchanged) into a coalesced
/// batched write, and free the old copy (which drops the region's live count to 0,
/// returning it to the free pool). Liveness and the dedup against this batch's
/// foreground rewrites both come from the frontier. Returns `(prefix, new_ptr)` for
/// the relocated records, to be installed into the frontier by the caller.
fn evacuate_regions(
    store: &FlatFile,
    upper: &RamNode,
    victims: &[u64],
    fg_pages: &std::collections::HashSet<u32>,
) -> Result<Vec<(Vec<u8>, DiskPtr)>> {
    // Collect (prefix, payload, old_ptr) for each live record in the victims.
    let mut live: Vec<(Vec<u8>, Vec<u8>, DiskPtr)> = Vec::new();
    for &region in victims {
        let buf = store.read_region(region)?;
        let base_page = region * REGION_PAGES;
        let mut p = 0usize;
        while p + 4 <= buf.len() {
            let len = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            if len == 0 {
                break; // unwritten tail of the region
            }
            let rec_pages = pages_for(RECORD_HDR + len) as usize;
            let end = p + 4 + len as usize;
            if end > buf.len() {
                break; // defensive: never expected (records don't straddle regions)
            }
            let page = (base_page + (p / PAGE as usize) as u64) as u32;
            // Skip foreground-target records — the batch's own rewrite supersedes
            // them — and stale/garbage records (liveness via the frontier).
            if !fg_pages.contains(&page) {
                let payload = &buf[p + 4..end];
                if let Ok(prefix) = parse_prefix(payload) {
                    if find_disk_ptr(upper, &prefix, 0) == Some(DiskPtr { page, len }) {
                        live.push((prefix, payload.to_vec(), DiskPtr { page, len }));
                    }
                }
            }
            p += rec_pages * PAGE as usize;
        }
    }
    if live.is_empty() {
        return Ok(Vec::new());
    }
    // Relocate verbatim (one coalesced batched write), then free the old copies.
    let payloads: Vec<&[u8]> = live.iter().map(|(_, pl, _)| pl.as_slice()).collect();
    let new_ptrs = store.write_batch(&payloads)?;
    let mut reloc = Vec::with_capacity(live.len());
    for ((prefix, _, old), new) in live.into_iter().zip(new_ptrs) {
        store.free(old);
        reloc.push((prefix, new));
    }
    Ok(reloc)
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
            // Tag 5 == the disk-side branch tag (`make_branch`); see above. Same
            // sparse encoding as `branch_hash`: bitmap + present children's hashes
            // only, so RAM and disk branches with the same structure still hash
            // identically (storage-independent root).
            let mut bytes = vec![5];
            let bitmap = branch_bitmap(children.iter().map(|c| c.is_some()));
            bytes.extend_from_slice(&bitmap.to_le_bytes());
            for child in children.iter().flatten() {
                let h = match child {
                    RamChild::Ram(node) => hash_ram(node),
                    RamChild::Disk { root, .. } => *root,
                };
                bytes.extend_from_slice(&h);
            }
            let computed = keccak(&bytes);
            hash.set(Some(computed));
            computed
        }
    }
}

/// Parallel root recompute: re-hash the top branch's child subtrees on separate
/// threads, then combine. The frontier is a tree of uniquely-owned nodes, so the
/// subtrees are disjoint — each thread only touches its own nodes' caches (see
/// [`HashCell`]). Cached nodes short-circuit, so an unchanged batch costs nothing
/// and only the invalidated paths re-hash. The non-branch spine and the per-child
/// subtrees use the ordinary serial [`hash_ram`].
fn hash_ram_parallel(node: &RamNode) -> Hash {
    match node {
        RamNode::Empty => empty_hash(),
        RamNode::Extension { path, child, hash } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            // One child: parallelism is at the branch below, so recurse parallel.
            let computed = hash_join(4, path, &hash_ram_parallel(child));
            hash.set(Some(computed));
            computed
        }
        RamNode::Branch { children, hash } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            // Fan out only when several child subtrees are actually stale. A lone
            // dirty path (e.g. one-by-one inserts re-hashing after every key) is far
            // cheaper to walk serially than to spawn a thread pool for; without this
            // guard root() pays ~16 thread spawns per call.
            let stale = children.iter().flatten().filter(|c| ram_child_stale(c)).count();
            if stale < 2 {
                return hash_ram(node);
            }
            // At the top of a deep frontier the children are themselves large Ram
            // subtrees, so a thread per present child fans the keccak-heavy re-hash
            // across cores. Cached children / Disk pointers resolve in O(1).
            let child_hashes: Vec<Hash> = std::thread::scope(|scope| {
                let handles: Vec<_> = children
                    .iter()
                    .flatten()
                    .map(|child| scope.spawn(move || ram_child_hash(child)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("frontier hash thread panicked"))
                    .collect()
            });
            let mut bytes = vec![5];
            let bitmap = branch_bitmap(children.iter().map(|c| c.is_some()));
            bytes.extend_from_slice(&bitmap.to_le_bytes());
            for h in &child_hashes {
                bytes.extend_from_slice(h);
            }
            let computed = keccak(&bytes);
            hash.set(Some(computed));
            computed
        }
    }
}

/// Hash of a frontier child (serial): a Ram subtree re-hashes, a Disk pointer
/// returns its stored root.
fn ram_child_hash(child: &RamChild) -> Hash {
    match child {
        RamChild::Ram(node) => hash_ram(node),
        RamChild::Disk { root, .. } => *root,
    }
}

/// Whether a child needs real re-hash work — a Ram subtree whose cached hash was
/// invalidated. Disk pointers and already-cached subtrees are O(1).
fn ram_child_stale(child: &RamChild) -> bool {
    match child {
        RamChild::Ram(node) => match node.as_ref() {
            RamNode::Empty => false,
            RamNode::Extension { hash, .. } | RamNode::Branch { hash, .. } => hash.get().is_none(),
        },
        RamChild::Disk { .. } => false,
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
        // A lazily-unparsed subtree carries its own root hash.
        Node::Raw { hash, .. } => *hash,
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

/// Bucket every live `DiskPtr` in the frontier into per-region live-page counts
/// (rebuilds [`RegionAlloc`] liveness on reopen). The frontier is the source of
/// truth for which records are live.
fn recompute_live(node: &RamNode, live: &mut [u32]) {
    match node {
        RamNode::Empty => {}
        RamNode::Extension { child, .. } => recompute_live(child, live),
        RamNode::Branch { children, .. } => {
            for c in children.iter().flatten() {
                match c {
                    RamChild::Ram(n) => recompute_live(n, live),
                    RamChild::Disk { ptr, .. } => {
                        let r = RegionAlloc::region_of(ptr.page as u64) as usize;
                        if r < live.len() {
                            live[r] += ptr.pages();
                        }
                    }
                }
            }
        }
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

        let _ = (flat_len, free_bytes);
        let mut db = FlatMpt::open(&path).unwrap();
        // The frontier and root survive the reopen. The region allocator is
        // recomputed from the frontier (fresh head past the file end), so
        // flat_file_len / free_bytes legitimately differ and aren't asserted.
        assert_eq!(db.root(), root);
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
    fn active_gc_bounds_file_under_churn() {
        // Build a file well past the GC floor, then overwrite every key many
        // times. Without active GC the high-water would grow ~per round; the
        // inline cleaner must reclaim regions for reuse so it stays bounded.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.flat");
        let mut db = FlatMpt::create(&path, Config::default()).unwrap();
        let n: u64 = 150_000;
        let build = |db: &mut FlatMpt, v: u8| {
            for chunk in (0..n).step_by(10_000) {
                let batch: Vec<(Key, Vec<u8>)> = (chunk..(chunk + 10_000).min(n))
                    .map(|i| (hashed_key(i.to_le_bytes()), vec![v; 32]))
                    .collect();
                db.insert_batch(batch).unwrap();
            }
        };
        build(&mut db, 0);
        let built = db.flat_file_len();
        assert!(
            built > GC_MIN_PAGES * PAGE,
            "build {built} below the GC floor — raise n"
        );

        let rounds = 12u8;
        for round in 1..=rounds {
            build(&mut db, round);
        }
        let churned = db.flat_file_len();

        // Active GC must have run and held the file far below the ~12x growth that
        // churn would otherwise cause.
        assert!(
            stats::GC_PASSES.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "GC never ran"
        );
        assert!(
            churned < built * 3,
            "file ballooned {built} -> {churned} despite GC"
        );
        // Every value is current and the root still resolves.
        for i in (0..n).step_by(7_001) {
            assert_eq!(
                db.get_value(&hashed_key(i.to_le_bytes())).unwrap(),
                Some(vec![rounds; 32])
            );
        }
        let _ = db.root();
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
