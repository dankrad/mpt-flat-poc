pub mod eth;
pub mod state;

use alloy_primitives::{B256, U256};
use anyhow::{Result, anyhow, bail};
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
    /// Total payload bytes written (sum of record sizes; excludes page/coalesce
    /// padding) — divide by keys for the logical write amplification.
    pub static WRITE_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Deep-promotion accounting: number of promotion events, the child records
    /// they wrote out, and those children's total bytes. `PROMOTE_CHILD_BYTES /
    /// WRITE_BYTES` is the share of write volume spent on promotion rebalancing.
    pub static PROMOTE_EVENTS: AtomicU64 = AtomicU64::new(0);
    pub static PROMOTE_CHILDREN: AtomicU64 = AtomicU64::new(0);
    pub static PROMOTE_CHILD_BYTES: AtomicU64 = AtomicU64::new(0);
    pub fn on_promote(children: u64, bytes: u64) {
        PROMOTE_EVENTS.fetch_add(1, Relaxed);
        PROMOTE_CHILDREN.fetch_add(children, Relaxed);
        PROMOTE_CHILD_BYTES.fetch_add(bytes, Relaxed);
    }
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
    // Phase-A sub-split: value-hash + map-build vs the routing walks.
    pub static A_BUILD_NS: AtomicU64 = AtomicU64::new(0);
    pub static A_ROUTE_NS: AtomicU64 = AtomicU64::new(0);

    pub fn on_batch(a_ns: u64, b_ns: u64, c_ns: u64) {
        PHASE_A_NS.fetch_add(a_ns, Relaxed);
        PHASE_B_NS.fetch_add(b_ns, Relaxed);
        PHASE_C_NS.fetch_add(c_ns, Relaxed);
        BATCHES.fetch_add(1, Relaxed);
    }
    pub fn on_phase_a_split(build_ns: u64, route_ns: u64) {
        A_BUILD_NS.fetch_add(build_ns, Relaxed);
        A_ROUTE_NS.fetch_add(route_ns, Relaxed);
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

    /// Fused-GC evacuation accounting (the `process_fold_gc` path). For each
    /// candidate region read to evacuate: `REGIONS` counts it, `BYTES_READ` adds
    /// the whole region read (the full 128 KiB scan), `LIVE_BYTES` adds the live
    /// records found in it (foreground + survivors — its true utilization),
    /// `RELOC_BYTES` adds just the relocated survivors' payloads, and `READ_NS`
    /// the region-read time. Derived: read-amp = `BYTES_READ / RELOC_BYTES` (bytes
    /// scanned per useful byte moved), evac util = `LIVE_BYTES / BYTES_READ`,
    /// reloc write share = `RELOC_BYTES / WRITE_BYTES`. These pinpoint whether the
    /// full-region read or the relocation volume is the cost to cut.
    pub static GC_EVAC_REGIONS: AtomicU64 = AtomicU64::new(0);
    pub static GC_EVAC_BYTES_READ: AtomicU64 = AtomicU64::new(0);
    pub static GC_EVAC_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GC_RELOC_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GC_EVAC_READ_NS: AtomicU64 = AtomicU64::new(0);
    pub fn on_evac(regions: u64, bytes_read: u64, live_bytes: u64, reloc_bytes: u64, read_ns: u64) {
        GC_EVAC_REGIONS.fetch_add(regions, Relaxed);
        GC_EVAC_BYTES_READ.fetch_add(bytes_read, Relaxed);
        GC_EVAC_LIVE_BYTES.fetch_add(live_bytes, Relaxed);
        GC_RELOC_BYTES.fetch_add(reloc_bytes, Relaxed);
        GC_EVAC_READ_NS.fetch_add(read_ns, Relaxed);
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

    /// One-writer path: wall time the device spends in the read phase (all readers)
    /// vs the write phase (single writer). `(OW_READ_NS + OW_WRITE_NS) / total wall`
    /// is the device-busy fraction — how close to "always reading or writing".
    pub static OW_READ_NS: AtomicU64 = AtomicU64::new(0);
    pub static OW_WRITE_NS: AtomicU64 = AtomicU64::new(0);
    pub fn on_one_writer(read_ns: u64, write_ns: u64) {
        OW_READ_NS.fetch_add(read_ns, Relaxed);
        OW_WRITE_NS.fetch_add(write_ns, Relaxed);
    }

    pub fn on_write(total: usize) {
        WRITES.fetch_add(1, Relaxed);
        WRITE_BYTES.fetch_add(total as u64, Relaxed);
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
        WRITE_BYTES.store(0, Relaxed);
        PROMOTE_EVENTS.store(0, Relaxed);
        PROMOTE_CHILDREN.store(0, Relaxed);
        PROMOTE_CHILD_BYTES.store(0, Relaxed);
        SPLITS.store(0, Relaxed);
        MAX_RECORD.store(0, Relaxed);
        MIN_SPLIT_TRIGGER.store(u64::MAX, Relaxed);
        MAX_SPLIT_TRIGGER.store(0, Relaxed);
        SPLIT_LEAVES.store(0, Relaxed);
        SPLIT_LEAF_BYTES.store(0, Relaxed);
        PHASE_A_NS.store(0, Relaxed);
        A_BUILD_NS.store(0, Relaxed);
        A_ROUTE_NS.store(0, Relaxed);
        PHASE_B_NS.store(0, Relaxed);
        PHASE_C_NS.store(0, Relaxed);
        BATCHES.store(0, Relaxed);
        B_READ_NS.store(0, Relaxed);
        B_REBUILD_NS.store(0, Relaxed);
        B_FINAL_NS.store(0, Relaxed);
        C_INSTALL_NS.store(0, Relaxed);
        C_ROOT_NS.store(0, Relaxed);
        C_FLUSH_NS.store(0, Relaxed);
        OW_READ_NS.store(0, Relaxed);
        OW_WRITE_NS.store(0, Relaxed);
        W_LOCK_NS.store(0, Relaxed);
        W_PWRITE_NS.store(0, Relaxed);
        B_SERIALIZE_NS.store(0, Relaxed);
        B_READ_IO_NS.store(0, Relaxed);
        B_READ_PARSE_NS.store(0, Relaxed);
        GC_PASSES.store(0, Relaxed);
        GC_REGIONS.store(0, Relaxed);
        GC_RELOCATED.store(0, Relaxed);
        GC_NS.store(0, Relaxed);
        GC_EVAC_REGIONS.store(0, Relaxed);
        GC_EVAC_BYTES_READ.store(0, Relaxed);
        GC_EVAC_LIVE_BYTES.store(0, Relaxed);
        GC_RELOC_BYTES.store(0, Relaxed);
        GC_EVAC_READ_NS.store(0, Relaxed);
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
        "value put (unused)",
        "value get (trie point-read)",
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

/// A `u32` index in `ADDR_UNIT` (256 B) units — records are densely packed at
/// 256 B alignment, so this addresses ~1 TiB of file — plus the record's exact
/// byte length. Eight bytes instead of the twelve a `{u64, u32}` byte offset would
/// take; it appears per frontier leaf (RAM) and per overflow child (disk), so the
/// four bytes matter at scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskPtr {
    pub unit: u32,
    pub len: u32,
}

/// On-disk record framing: a `u32` little-endian payload length precedes the
/// payload, so GC can scan a region record-by-record without the frontier. The
/// payload itself starts at `offset() + RECORD_HDR`.
const RECORD_HDR: u32 = 4;

/// Dense-packing address granularity: records are placed at 256 B-aligned offsets
/// (vs the 16 KiB `PAGE` they used to be padded to), cutting the per-record waste
/// from ~71% to a few percent and shrinking the working set toward RAM-resident.
const ADDR_UNIT: u64 = 256;
/// 256 B units per 16 KiB page (physical write alignment) and per 128 KiB region.
const UNITS_PER_PAGE: u64 = PAGE / ADDR_UNIT;
const REGION_UNITS: u64 = REGION_PAGES * UNITS_PER_PAGE;
const REGION_BYTES: usize = (REGION_PAGES * PAGE) as usize;

/// 256 B units needed to hold `bytes` (a framed record).
fn units_for(bytes: u32) -> u32 {
    (bytes as u64).div_ceil(ADDR_UNIT) as u32
}

impl DiskPtr {
    /// Byte offset of the record's length prefix in the flat file.
    fn offset(&self) -> u64 {
        self.unit as u64 * ADDR_UNIT
    }
    /// 256 B units the framed record (header + payload) occupies.
    fn units(&self) -> u32 {
        units_for(RECORD_HDR + self.len)
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
const EVAC_MAX_UTIL: f64 = 0.30;

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
    fn region_of_unit(unit: u64) -> u64 {
        unit / REGION_UNITS
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

    /// Reserve a page-aligned run of `run_pages` pages at the head (the physical
    /// write is 16 KiB-aligned), crediting `live_units` of dense live data to the
    /// region. Opens a new head region first if the run wouldn't fit, so the run
    /// stays within one region. Returns the start page. `live_units` is the
    /// records' true 256 B footprint — the page-rounding pad is left as garbage.
    fn alloc(&mut self, run_pages: u32, live_units: u32, end_page: &AtomicU64) -> u64 {
        debug_assert!(run_pages as u64 <= REGION_PAGES);
        let region_end = self.head_region * REGION_PAGES + REGION_PAGES;
        if self.live.is_empty() || self.next_page + run_pages as u64 > region_end {
            self.open_new_head(end_page);
        }
        let page = self.next_page;
        self.next_page += run_pages as u64;
        self.live[self.head_region as usize] += live_units;
        end_page.fetch_max(self.next_page, Ordering::SeqCst);
        page
    }

    /// Mark a record's units dead; reclaim the region once it is fully dead.
    fn free(&mut self, unit: u64, units: u32) {
        let r = Self::region_of_unit(unit) as usize;
        if r >= self.live.len() {
            return;
        }
        let was = self.live[r];
        self.live[r] = was.saturating_sub(units);
        if was > 0 && self.live[r] == 0 && r as u64 != self.head_region {
            self.free_regions.push(r as u64);
        }
    }

    fn live_units(&self) -> u64 {
        self.live.iter().map(|&u| u as u64).sum()
    }

    fn free_region_units(&self) -> u64 {
        self.free_regions.len() as u64 * REGION_UNITS
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
        let cap_live = (REGION_UNITS as f64 * EVAC_MAX_UTIL) as u32;
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
                let u = live as f64 / REGION_UNITS as f64;
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

    /// Opportunistic victim selection: from the regions this batch already read a
    /// leaf out of (`touched` — so their bytes are likely still page-cache-hot from
    /// the foreground reads), pick those under `max_util` live. GC then only ever
    /// re-reads regions the foreground already paid to fetch — no separate cold
    /// random-read pass (the cost that doesn't scale with read QD) — and piggybacks
    /// the relocation. Cheap and self-limiting: it cleans exactly the regions the
    /// insert churn is emptying.
    fn select_opportunistic(&self, touched: &std::collections::HashSet<u64>, max_util: f64) -> Vec<u64> {
        let cap_live = (REGION_UNITS as f64 * max_util) as u32;
        touched
            .iter()
            .copied()
            .filter(|&r| {
                let live = self.live.get(r as usize).copied().unwrap_or(0);
                live > 0 && live <= cap_live && r != self.head_region && !self.free_regions.contains(&r)
            })
            .collect()
    }
}

/// Direct-I/O alignment for buffer address, file offset, and length. 4096 covers
/// Linux `O_DIRECT` on ext4/xfs (and is a divisor of `PAGE`, so page-aligned
/// writes already satisfy it). macOS `F_NOCACHE` imposes no alignment, but we use
/// the same aligned path on both so it's exercised by the macOS test suite.
const DIO_ALIGN: u64 = 4096;

/// Bypass the page cache for flat-file I/O (`MPT_DIRECT_IO=1`), decided once at
/// open time. Direct reads go straight to the device, which scales far better
/// under many threads when the working set exceeds RAM — there's no per-file
/// page-cache fault lock to serialize on (see `examples/mmapseg`, and the
/// O_DIRECT-vs-buffered gap in `examples/iops`). The trade is losing the page
/// cache for data that *does* fit RAM, so this is a disk-bound-only win.
fn direct_io() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_DIRECT_IO").as_deref() == Ok("1"))
}

/// Open the flat file, optionally with the page cache bypassed (Linux `O_DIRECT`
/// open flag; macOS `F_NOCACHE` via `fcntl`).
fn open_flat(path: &Path, create: bool, direct: bool) -> Result<File> {
    let mut o = OpenOptions::new();
    o.read(true).write(true);
    if create {
        o.create(true).truncate(true);
    }
    #[cfg(target_os = "linux")]
    if direct {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_DIRECT);
    }
    let f = o.open(path)?;
    #[cfg(target_os = "macos")]
    if direct {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    let _ = direct;
    Ok(f)
}

/// A heap buffer aligned to [`DIO_ALIGN`] (and zero-initialized) — required for
/// the O_DIRECT buffer-address constraint. Derefs to `[u8]`.
struct AlignedBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    len: usize,
}
impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len.max(1), DIO_ALIGN as usize).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned alloc failed");
        Self { ptr, layout, len }
    }
}
impl std::ops::Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// A read/write buffer that is a plain `Vec` for buffered I/O or a [`DIO_ALIGN`]-
/// aligned buffer for direct I/O. Derefs to `[u8]` so the framing code is the
/// same either way.
enum IoBuf {
    Heap(Vec<u8>),
    Aligned(AlignedBuf),
}
impl IoBuf {
    fn zeroed(len: usize, direct: bool) -> Self {
        if direct {
            IoBuf::Aligned(AlignedBuf::new(len))
        } else {
            IoBuf::Heap(vec![0u8; len])
        }
    }
}
impl std::ops::Deref for IoBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            IoBuf::Heap(v) => v,
            IoBuf::Aligned(a) => a,
        }
    }
}
impl std::ops::DerefMut for IoBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            IoBuf::Heap(v) => v,
            IoBuf::Aligned(a) => a,
        }
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
    /// Page cache bypassed (O_DIRECT / F_NOCACHE) ⇒ reads/writes use aligned
    /// buffers and offset/length widening.
    direct: bool,
}

/// Pages needed to hold a record of `record_bytes` (length prefix + payload).
fn pages_for(record_bytes: u32) -> u32 {
    record_bytes.div_ceil(PAGE as u32)
}

impl FlatFile {
    fn new(file: File, direct: bool) -> Self {
        Self {
            file,
            seg: Mutex::new(RegionAlloc::default()),
            end_page: AtomicU64::new(0),
            direct,
        }
    }

    /// Read `len` payload bytes at file offset `off`. With direct I/O the request
    /// is widened to [`DIO_ALIGN`] boundaries into an aligned buffer (O_DIRECT
    /// requires aligned offset+length+buffer) and the payload copied out; buffered
    /// I/O reads exactly `len` into a tight `Vec` (kept zero-copy via `Arc`).
    fn read_payload(&self, off: u64, len: usize) -> Result<Vec<u8>> {
        if self.direct {
            let lo = off & !(DIO_ALIGN - 1);
            let hi = (off + len as u64 + DIO_ALIGN - 1) & !(DIO_ALIGN - 1);
            let mut abuf = AlignedBuf::new((hi - lo) as usize);
            (&self.file).read_exact_at(&mut abuf, lo)?;
            let pad = (off - lo) as usize;
            Ok(abuf[pad..pad + len].to_vec())
        } else {
            let mut v = vec![0u8; len];
            (&self.file).read_exact_at(&mut v, off)?;
            Ok(v)
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
        let run_pages = pages_for(RECORD_HDR + total);
        let units = units_for(RECORD_HDR + total);
        let lt = std::time::Instant::now();
        let page = self.seg.lock().unwrap().alloc(run_pages, units, &self.end_page);
        stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
        let unit = page * UNITS_PER_PAGE;
        if unit + units as u64 > u32::MAX as u64 {
            bail!("flat file exceeds the DiskPtr addressing limit");
        }

        // Frame: [u32 payload len][payload][zero pad to page].
        let mut record = IoBuf::zeroed(run_pages as usize * PAGE as usize, self.direct);
        record[..4].copy_from_slice(&total.to_le_bytes());
        record[4..4 + payload.len()].copy_from_slice(payload);

        let _g = prof::scope(prof::Cat::FileWrite);
        let wt = std::time::Instant::now();
        (&self.file).write_all_at(&record, page * PAGE)?;
        stats::on_pwrite(wt.elapsed().as_nanos() as u64);
        Ok(DiskPtr { unit: unit as u32, len: total })
    }

    /// Coalesce several records into contiguous appended `pwrite`s, packed densely
    /// at 256 B alignment, each run ≤ one region (so the write is 16 KiB-aligned and
    /// stays on the bandwidth plateau, and no record straddles a region). Returns a
    /// `DiskPtr` per payload.
    fn write_batch(&self, payloads: &[&[u8]]) -> Result<Vec<DiskPtr>> {
        let aligned = |bytes: usize| units_for(bytes as u32) as usize * ADDR_UNIT as usize;
        let mut ptrs = Vec::with_capacity(payloads.len());
        let mut i = 0;
        while i < payloads.len() {
            // Pack a run of records (256 B-aligned) whose total fits in one region.
            let mut run_bytes = 0usize;
            let mut j = i;
            while j < payloads.len() {
                let rec = aligned(RECORD_HDR as usize + payloads[j].len());
                if j > i && run_bytes + rec > REGION_BYTES {
                    break;
                }
                run_bytes += rec;
                j += 1;
            }
            let run_pages = pages_for(run_bytes as u32);
            let live_units: u32 = payloads[i..j]
                .iter()
                .map(|p| units_for(RECORD_HDR + p.len() as u32))
                .sum();
            let lt = std::time::Instant::now();
            let page_start = self.seg.lock().unwrap().alloc(run_pages, live_units, &self.end_page);
            stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
            let base_unit = page_start * UNITS_PER_PAGE;
            if base_unit + (run_pages as u64 * UNITS_PER_PAGE) > u32::MAX as u64 {
                bail!("flat file exceeds the DiskPtr addressing limit");
            }
            let mut buf = IoBuf::zeroed(run_pages as usize * PAGE as usize, self.direct);
            let mut off = 0usize; // dense byte offset within the run
            for p in &payloads[i..j] {
                // Frame: [u32 payload len][payload], 256 B-aligned within the run.
                buf[off..off + 4].copy_from_slice(&(p.len() as u32).to_le_bytes());
                buf[off + 4..off + 4 + p.len()].copy_from_slice(p);
                ptrs.push(DiskPtr {
                    unit: (base_unit + (off / ADDR_UNIT as usize) as u64) as u32,
                    len: p.len() as u32,
                });
                stats::on_write(p.len());
                off += aligned(RECORD_HDR as usize + p.len());
            }
            let _g = prof::scope(prof::Cat::FileWrite);
            let wt = std::time::Instant::now();
            (&self.file).write_all_at(&buf, page_start * PAGE)?;
            stats::on_pwrite(wt.elapsed().as_nanos() as u64);
            i = j;
        }
        Ok(ptrs)
    }

    /// Like [`write_batch`] but fans the per-run `pwrite`s across worker threads.
    /// All allocation happens first in one locked pass (each run gets a region and
    /// its records' `ptrs` are assigned); then the writes run lock-free in parallel,
    /// each run targeting a distinct region. The single-writer `write_batch` is best
    /// for sequential end-appends (one monotonic stream ~= device seq rate), but the
    /// bounded-file GC path reuses freed regions scattered across the file, and
    /// scattered writes are queue-depth-bound — many concurrent writers to distinct
    /// offsets hit the device's multi-thread write rate (~6-10 GB/s) instead of the
    /// single-stream ~3.5. No tail contention since every run is a different region.
    fn write_batch_parallel(&self, payloads: &[&[u8]]) -> Result<Vec<DiskPtr>> {
        let aligned = |bytes: usize| units_for(bytes as u32) as usize * ADDR_UNIT as usize;
        let mut ptrs: Vec<DiskPtr> = vec![DiskPtr { unit: 0, len: 0 }; payloads.len()];
        // Plan + allocate pass (single, locked): pack runs (each ≤ one region),
        // reserve a region per run, and fill in every record's ptr.
        let mut runs: Vec<(u64, u32, usize, usize)> = Vec::new(); // (page_start, run_pages, i, j)
        {
            let lt = std::time::Instant::now();
            let mut seg = self.seg.lock().unwrap();
            let mut i = 0;
            while i < payloads.len() {
                let mut run_bytes = 0usize;
                let mut j = i;
                while j < payloads.len() {
                    let rec = aligned(RECORD_HDR as usize + payloads[j].len());
                    if j > i && run_bytes + rec > REGION_BYTES {
                        break;
                    }
                    run_bytes += rec;
                    j += 1;
                }
                let run_pages = pages_for(run_bytes as u32);
                let live_units: u32 = payloads[i..j]
                    .iter()
                    .map(|p| units_for(RECORD_HDR + p.len() as u32))
                    .sum();
                let page_start = seg.alloc(run_pages, live_units, &self.end_page);
                let base_unit = page_start * UNITS_PER_PAGE;
                if base_unit + (run_pages as u64 * UNITS_PER_PAGE) > u32::MAX as u64 {
                    bail!("flat file exceeds the DiskPtr addressing limit");
                }
                let mut off = 0usize;
                for k in i..j {
                    ptrs[k] = DiskPtr {
                        unit: (base_unit + (off / ADDR_UNIT as usize) as u64) as u32,
                        len: payloads[k].len() as u32,
                    };
                    off += aligned(RECORD_HDR as usize + payloads[k].len());
                }
                runs.push((page_start, run_pages, i, j));
                i = j;
            }
            stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
        }
        // Write pass (parallel, lock-free): each thread pwrites its runs to distinct
        // region offsets.
        let threads = worker_count();
        let chunk = runs.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = runs
                .chunks(chunk)
                .map(|rc| {
                    scope.spawn(move || -> Result<()> {
                        for &(page_start, run_pages, i, j) in rc {
                            let mut buf =
                                IoBuf::zeroed(run_pages as usize * PAGE as usize, self.direct);
                            let mut off = 0usize;
                            for p in &payloads[i..j] {
                                buf[off..off + 4].copy_from_slice(&(p.len() as u32).to_le_bytes());
                                buf[off + 4..off + 4 + p.len()].copy_from_slice(p);
                                stats::on_write(p.len());
                                off += units_for(RECORD_HDR + p.len() as u32) as usize
                                    * ADDR_UNIT as usize;
                            }
                            let _g = prof::scope(prof::Cat::FileWrite);
                            let wt = std::time::Instant::now();
                            (&self.file).write_all_at(&buf, page_start * PAGE)?;
                            stats::on_pwrite(wt.elapsed().as_nanos() as u64);
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                h.join().expect("parallel writer thread panicked")?;
            }
            Ok(())
        })?;
        Ok(ptrs)
    }

    fn read(&self, ptr: DiskPtr) -> Result<DiskSubtree> {
        let record = self.read_payload(ptr.offset() + RECORD_HDR as u64, ptr.len as usize)?;
        deserialize_subtree(&record)
    }

    /// Lazy read: parse only the spine; child subtrees stay `Raw`. Used by the
    /// insert path, where a record is touched on one key's path per call.
    fn read_lazy(&self, ptr: DiskPtr) -> Result<DiskSubtree> {
        let record = {
            let _g = prof::scope(prof::Cat::FileRead);
            let it = std::time::Instant::now();
            // Payload starts just past the framing header.
            let r = self.read_payload(ptr.offset() + RECORD_HDR as u64, ptr.len as usize)?;
            stats::on_read_io(it.elapsed().as_nanos() as u64);
            r
        };
        let _g = prof::scope(prof::Cat::Deserialize);
        // `Arc::from(Vec)` reuses the allocation (no copy); Raw children then
        // share it as zero-copy slices.
        let pt = std::time::Instant::now();
        let out = deserialize_subtree_lazy(Arc::from(record));
        stats::on_read_parse(pt.elapsed().as_nanos() as u64);
        out
    }

    /// Read one whole region (`REGION_PAGES` pages) for GC scanning. Pages past the
    /// file end read as zeros (sparse), which the scan treats as an empty tail.
    fn read_region(&self, region: u64) -> Result<IoBuf> {
        let mut buf = IoBuf::zeroed(REGION_PAGES as usize * PAGE as usize, self.direct);
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
        self.seg.lock().unwrap().free(ptr.unit as u64, ptr.units());
        stats::on_alloc_lock(lt.elapsed().as_nanos() as u64);
    }

    fn end_page(&self) -> u64 {
        self.end_page.load(Ordering::SeqCst)
    }

    /// Live units and units held in reclaimed-but-unused regions.
    fn live_and_free_units(&self) -> (u64, u64) {
        let seg = self.seg.lock().unwrap();
        (seg.live_units(), seg.free_region_units())
    }

    /// Total dead bytes in the file (everything not currently live).
    fn garbage_bytes(&self) -> u64 {
        let (live_units, _) = self.live_and_free_units();
        (self.end_page() * UNITS_PER_PAGE)
            .saturating_sub(live_units)
            * ADDR_UNIT
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
        /// from the tree position, so the key isn't stored.
        path: Vec<u8>,
        /// The leaf's value bytes. Ethereum's leaf reference is `ref(RLP([
        /// hex_prefix(path), value]))`, so it depends on `path`; when a split
        /// re-homes the leaf to a shallower depth its `path` shrinks and `nref`
        /// must be recomputed — which needs `value` here, in the trie.
        value: Vec<u8>,
        nref: NodeRef,
    },
    Extension {
        path: Vec<u8>,
        child: Box<Node>,
        nref: NodeRef,
    },
    Branch {
        children: [Option<Box<Node>>; 16],
        nref: NodeRef,
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
    /// shared record buffer plus its `nref`. Produced by the *lazy* reader so
    /// that, to change one key, we only parse the nodes on that key's path —
    /// untouched sibling subtrees stay `Raw` (no byte copy on read) and are written
    /// back verbatim. Expanded one level on demand by `record_node_insert`.
    Raw {
        buf: Arc<[u8]>,
        off: usize,
        len: usize,
        nref: NodeRef,
    },
    /// A state-trie **account** leaf (Phase 4): the account fields plus its storage
    /// trie nested *inline* in this node, so the whole thing is one unified trie and
    /// re-hashing is self-contained. The leaf's value is `RLP([nonce, balance,
    /// storage_root, code_hash])`, where `storage_root` is the Merkle root of the
    /// nested `storage` subtree — computed here, never fetched from a store. Small
    /// storage packs in the same record; large-contract promotion is future work.
    Account {
        /// Remaining `keccak(address)` nibbles from this leaf's position to depth 64.
        path: Vec<u8>,
        nonce: u64,
        balance: U256,
        code_hash: Hash,
        /// The account's storage trie (ordinary `Node`s over `keccak(slot)` paths,
        /// with `RLP(U256)` values); `Node::Empty` when the account has no storage.
        storage: Box<Node>,
        /// Cached storage-trie root (`EMPTY_ROOT` when empty) — lets an account-field
        /// write skip re-walking storage.
        storage_root: Hash,
        /// Cached reference of this account leaf (over the account RLP).
        nref: NodeRef,
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
    /// A **promoted account**: an account whose storage grew past a record, lifted
    /// into the frontier so its storage trie spans multiple records (see
    /// [`PromotedAccount`]). Boxed so this rare variant doesn't inflate `RamChild`'s
    /// size — every frontier branch holds a `[Option<RamChild>; 16]`, so an unboxed
    /// ~100-byte variant would ~double the whole in-RAM frontier footprint.
    Account(Box<PromotedAccount>),
    /// RAM-build leaf: the serialized record bytes held as their *own* heap object
    /// (an `Arc<[u8]>`), with no flat-file I/O. Reads clone the `Arc` (lock-free
    /// refcount bump); a rewrite drops the old `Arc` (malloc reclaims) and installs
    /// a new one. Spilled to a `Disk` record before persist (never serialized into
    /// a manifest). Same Merkle root as the equivalent `Disk` leaf, so a leaf
    /// hashes identically either way.
    Mem(MemLeaf),
}

/// The payload of a promoted account (boxed inside [`RamChild::Account`]). `storage`
/// is the account's storage trie as a RAM subtree whose leaves are `Disk` storage
/// records — so the inter-record pointers live in RAM and GC relocates them like any
/// other frontier record (no on-disk pointer rewriting). The account leaf's hash is
/// `leaf_ref(path, RLP([nonce, balance, storage_root, code_hash]))`, where
/// `storage_root = hash_ram(storage)`; recomputed on demand (cheap) while the storage
/// subtree memoizes its own hash. `balance`/`code_hash` are stored as bytes so the
/// manifest (serde) needs no `U256` serde feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromotedAccount {
    /// Remaining `keccak(address)` nibbles from this leaf's frontier position.
    path: Vec<u8>,
    nonce: u64,
    balance: [u8; 32],
    code_hash: Hash,
    storage: Box<RamNode>,
}

/// A `Mem` leaf's bytes (the same `[prefix-path][node]` payload a disk record
/// holds) and cached root. `Arc<[u8]>` isn't serde-serializable without the `rc`
/// feature, and a `Mem` leaf must be spilled before any manifest write anyway, so
/// these impls deliberately error as a guard. Appending the `Mem` variant keeps
/// bincode's existing `Ram`/`Disk` indices, so old manifests still load.
#[derive(Debug, Clone)]
struct MemLeaf {
    bytes: Arc<[u8]>,
    root: Hash,
}
impl Serialize for MemLeaf {
    fn serialize<S: serde::Serializer>(&self, _: S) -> std::result::Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "RamChild::Mem must be spilled to disk before persist",
        ))
    }
}
impl<'de> Deserialize<'de> for MemLeaf {
    fn deserialize<D: serde::Deserializer<'de>>(_: D) -> std::result::Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "manifest unexpectedly contains a RamChild::Mem leaf",
        ))
    }
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
}

impl RamReport {
    /// Estimated total in-RAM index bytes.
    pub fn total_bytes(&self) -> usize {
        self.frontier_bytes + self.free_list_bytes
    }
}

#[derive(Debug)]
pub struct FlatMpt {
    cfg: Config,
    store: FlatFile,
    upper: RamNode,
    /// Path of the flat file; the manifest is derived from it.
    path: PathBuf,
    /// Inline-GC cleaning rate: victim regions to evacuate per batch, adjusted by
    /// the proportional controller to hold utilization at `TARGET_UTIL`.
    gc_regions: usize,
    /// RAM-build mode: new/rewritten leaves live in RAM (`RamChild::Mem`, each its
    /// own `Arc`) with no flat-file I/O or GC. Flips to `false` after the first
    /// spill. Enabled by `MPT_RAM_BUILD=1` at create time.
    ram_mode: bool,
    /// Resident-size ceiling (bytes) at which a RAM build spills to disk.
    spill_threshold: u64,
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
        let direct = direct_io();
        let file = open_flat(path, true, direct)?;

        let (ram_mode, spill_threshold) = ram_build_config();
        Ok(Self {
            cfg,
            store: FlatFile::new(file, direct),
            upper: RamNode::Empty,
            path: path.to_path_buf(),
            gc_regions: 0,
            ram_mode,
            spill_threshold,
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

        // Build the store first (reads need the file), then rebuild per-region
        // liveness by walking the frontier AND descending into each record's
        // overflow children. Finally open a fresh head region past the file end.
        let num_regions = end_page.div_ceil(REGION_PAGES);
        let direct = direct_io();
        let file = open_flat(path, false, direct)?;
        let store = FlatFile {
            file,
            seg: Mutex::new(RegionAlloc {
                live: vec![0u32; num_regions as usize],
                // Reopened regions are "old" (epoch 0); new regions get growing
                // epochs, so existing data ages relative to fresh writes.
                epoch_of: vec![0u32; num_regions as usize],
                ..RegionAlloc::default()
            }),
            end_page: AtomicU64::new(end_page),
            direct,
        };
        {
            let mut live = vec![0u32; num_regions as usize];
            recompute_live(&upper, &mut live);
            let mut alloc = store.seg.lock().unwrap();
            alloc.live = live;
            alloc.head_region = num_regions;
            alloc.next_page = num_regions * REGION_PAGES;
            alloc.ensure_region(num_regions);
        }
        store.end_page.store(num_regions * REGION_PAGES, Ordering::SeqCst);

        Ok(Self {
            cfg,
            store,
            upper,
            path: path.to_path_buf(),
            gc_regions: 0,
            // A reopened DB is disk-resident; RAM-build mode is for fresh creation.
            ram_mode: false,
            spill_threshold: ram_build_config().1,
        })
    }

    /// Flush the flat file's writer so all preceding inserts are on disk, without a
    /// full [`persist`](Self::persist) checkpoint.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()
    }

    /// Checkpoint the in-RAM state to disk so the database can later be reopened
    /// with [`FlatMpt::open`]. Spills in-RAM leaves, fsyncs the flat file, then
    /// writes the manifest atomically (temp file + rename) so a crash can't leave
    /// a torn manifest.
    pub fn persist(&mut self) -> Result<()> {
        // The manifest stores disk ptrs only; spill any in-RAM leaves to the file
        // first (they're not serializable and would be lost on reopen).
        self.spill_mem()?;
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
        self.insert_op(key, LeafOp::Value(value))
    }

    /// Insert an account leaf (creating or updating fields, preserving storage).
    pub fn insert_account(
        &mut self,
        key: Key,
        nonce: u64,
        balance: U256,
        code_hash: Hash,
    ) -> Result<Hash> {
        self.insert_op(key, LeafOp::Account { nonce, balance, code_hash })
    }

    /// Set one storage slot of the account at `key` (auto-creates an empty account
    /// if none exists). `slot` is the secure-trie storage key (`keccak(slot)`).
    pub fn insert_storage(&mut self, key: Key, slot: Key, value: Vec<u8>) -> Result<Hash> {
        self.insert_op(key, LeafOp::Storage { slot, value })
    }

    fn insert_op(&mut self, key: Key, op: LeafOp) -> Result<Hash> {
        let cfg = self.cfg.clone();
        let ram = self.ram_mode;
        insert_ram(&self.store, &cfg, &mut self.upper, Vec::new(), key, op, ram)?;
        self.store.flush()?;
        Ok(self.root())
    }

    /// Insert/overwrite many key/value pairs at once. Equivalent in result to
    /// calling [`insert`](Self::insert) for each pair (last value wins on a
    /// duplicate key within the batch), but far cheaper: the trie is updated by
    /// grouping keys per route so every touched disk leaf is read/rebuilt/written
    /// exactly once and every node is re-hashed at most once. Returns the new root.
    pub fn insert_batch(&mut self, entries: Vec<(Key, Vec<u8>)>) -> Result<Hash> {
        if entries.is_empty() {
            return Ok(self.root());
        }
        // Advance the GC epoch so regions written this batch are "age 0" and the
        // cost-benefit cleaner leaves them alone until they settle.
        self.store.seg.lock().unwrap().bump_epoch();
        let t_a = std::time::Instant::now();
        // Dedup entries into the per-leaf value map (last write wins). The value
        // now lives in its trie leaf; the Ethereum leaf hash is position-dependent,
        // so it can't be precomputed here — it's computed when the leaf is placed,
        // inside the parallel per-record fold of Phase B.
        let mut leaves: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
        for (key, value) in entries.into_iter() {
            leaves.insert(key, value);
        }
        let cfg = self.cfg.clone();

        // RAM-build fast path: leaves live in RAM (their own `Arc`s), so there's no
        // disk I/O and no GC. Parallelism comes from partitioning by top nibble and
        // fanning the serial insert across the top branch's disjoint child subtrees
        // — no shared store, no lock. Then maybe spill if over the memory ceiling.
        if self.ram_mode {
            self.insert_batch_ram(leaves)?;
            self.maybe_spill()?;
            return Ok(self.root());
        }

        // Phase A (read-only): route each key to the frontier disk leaf it lands
        // in, grouping keys per leaf. Keys with no existing leaf create fresh
        // structure and are applied serially afterwards. The routing walks are
        // parallelized across cores below for large batches.
        // The walks are independent read-only descents of `self.upper`, so fan
        // them across CPU threads and merge the per-thread partials afterward
        // (cheap — the merge does no trie walks). Phase A is CPU-bound, so size
        // to cores, not the (much larger) I/O worker count. A group's
        // representative key can be *any* key routing to that leaf, so merging
        // partials in arbitrary order is correct.
        let build_ns = t_a.elapsed().as_nanos() as u64; // dedup + value-hash + maps
        let t_route = std::time::Instant::now();
        let leaves_vec: Vec<(Key, Vec<u8>)> = leaves.into_iter().collect();
        let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let route_threads = ncpu.min(leaves_vec.len() / 1024).max(1);
        let (groups, fresh): (HashMap<u32, (DiskPtr, Key, Vec<(Key, Vec<u8>)>)>, Vec<(Key, Vec<u8>)>) =
            if route_threads > 1 {
                let upper = &self.upper;
                let chunk = leaves_vec.len().div_ceil(route_threads);
                let partials: Vec<(HashMap<u32, (DiskPtr, Key, Vec<(Key, Vec<u8>)>)>, Vec<(Key, Vec<u8>)>)> =
                    std::thread::scope(|scope| {
                        let handles: Vec<_> = leaves_vec
                            .chunks(chunk)
                            .map(|c| {
                                scope.spawn(move || {
                                    let mut g: HashMap<u32, (DiskPtr, Key, Vec<(Key, Vec<u8>)>)> =
                                        HashMap::new();
                                    let mut f: Vec<(Key, Vec<u8>)> = Vec::new();
                                    for (key, vh) in c {
                                        match find_disk_ptr_key(upper, key, 0) {
                                            Some(ptr) => g
                                                .entry(ptr.unit)
                                                .or_insert_with(|| (ptr, *key, Vec::new()))
                                                .2
                                                .push((*key, vh.clone())),
                                            None => f.push((*key, vh.clone())),
                                        }
                                    }
                                    (g, f)
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| h.join().expect("phase-A routing thread panicked"))
                            .collect()
                    });
                let mut groups: HashMap<u32, (DiskPtr, Key, Vec<(Key, Vec<u8>)>)> = HashMap::new();
                let mut fresh: Vec<(Key, Vec<u8>)> = Vec::new();
                for (g, f) in partials {
                    for (unit, (ptr, rep, kvs)) in g {
                        groups.entry(unit).or_insert_with(|| (ptr, rep, Vec::new())).2.extend(kvs);
                    }
                    fresh.extend(f);
                }
                (groups, fresh)
            } else {
                let mut groups: HashMap<u32, (DiskPtr, Key, Vec<(Key, Vec<u8>)>)> = HashMap::new();
                let mut fresh: Vec<(Key, Vec<u8>)> = Vec::new();
                for (key, value) in leaves_vec {
                    match find_disk_ptr_key(&self.upper, &key, 0) {
                        Some(ptr) => groups
                            .entry(ptr.unit)
                            .or_insert_with(|| (ptr, key, Vec::new()))
                            .2
                            .push((key, value)),
                        None => fresh.push((key, value)),
                    }
                }
                (groups, fresh)
            };
        let mut groups: Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)> = groups.into_values().collect();
        // Sort by file offset so each worker's chunk is a contiguous file range and
        // reads ascend in place — turning the per-leaf random reads into sequential
        // ones, and letting large batches coalesce neighbours into one big read.
        let coalesce = fold_coalesce();
        if coalesce && groups.len() >= 64 {
            groups.sort_unstable_by_key(|(ptr, _, _)| ptr.unit);
        }
        let a_ns = t_a.elapsed().as_nanos() as u64;
        stats::on_phase_a_split(build_ns, t_route.elapsed().as_nanos() as u64);
        let t_b = std::time::Instant::now();

        // Phase B (parallel): each group reads its record, applies its keys
        // (record_node_insert + migrate + possible promotion), and produces the
        // replacement RamChild — all the per-record CPU + I/O off the serial path.
        // Groups touch disjoint subtrees; the store is thread-safe. With many
        // groups (sorted above) each worker folds its file range via coalesced
        // multi-MB span reads (sequential) instead of one `pread` per leaf.
        let batched = batched_writes();
        // Opportunistic GC fuses evacuation into Phase B: candidate (touched,
        // under-util) regions are read once and serve both the insert fold and the
        // evacuation, so no region is read twice. Otherwise: normal Phase B, then a
        // separate evacuation pass over the rate-selected emptiest regions.
        let opp = gc_opportunistic() && groups.len() >= 64;
        let results: Vec<(Key, RamChild)>;
        let reloc: Vec<(Vec<u8>, DiskPtr)>;
        if one_writer() && groups.len() >= 64 {
            // Phase B as many parallel readers (read+fold to payloads) + ONE writer
            // that appends them all in a single sequential `write_batch` — avoids the
            // inter-worker append contention (concurrent appends run ~1.1 GB/s vs a
            // single stream's ~3.5). Phased read-all-then-write-all: overlapping the
            // two would contend for the one device.
            //
            // With opportunistic GC (`opp`): candidate (touched, under-util) regions
            // are read *once* and that read serves both the foreground fold and the
            // evacuation of the region's other live records — the relocated payloads
            // ride the same single writer, so GC adds no second read of those regions
            // and no append contention. Without `opp`: plain fold, no evacuation.
            let store = &self.store;
            let upper = &self.upper;
            // Read phase: readers fold to payloads (device reading), and — when
            // `opp` — also collect relocated survivors from low-util regions.
            let t_read = std::time::Instant::now();
            #[allow(clippy::type_complexity)]
            let (leaves, promoted, relocs): (
                Vec<(Key, Hash, Vec<u8>)>,
                Vec<(Key, RamChild)>,
                Vec<(Vec<u8>, Vec<u8>, DiskPtr)>,
            ) = std::thread::scope(|scope| -> Result<_> {
                let folded = if opp {
                    process_fold_gc(store, upper, &cfg, &groups)?
                } else {
                    let threads = worker_count();
                    let chunk = groups.len().div_ceil(threads);
                    let handles: Vec<_> = groups
                        .chunks(chunk)
                        .map(|c| scope.spawn(|| process_chunk_fold(store, &cfg, c)))
                        .collect();
                    let mut lv: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
                    let mut pr: Vec<(Key, RamChild)> = Vec::new();
                    for h in handles {
                        let (l, p) = h.join().expect("fold reader thread panicked")?;
                        lv.extend(l);
                        pr.extend(p);
                    }
                    (lv, pr, Vec::new())
                };
                Ok(folded)
            })?;
            let read_ns = t_read.elapsed().as_nanos() as u64;
            // Write phase: one sequential `write_batch` of every new foreground leaf
            // followed by every relocated survivor. Then assign the returned ptrs —
            // foreground -> frontier child, relocations -> (prefix, new_ptr) for the
            // Phase C install — and free the relocations' old ptrs (the foreground
            // old ptrs were already freed inside `fold_group`).
            let t_write = std::time::Instant::now();
            let nlv = leaves.len();
            let payloads: Vec<&[u8]> = leaves
                .iter()
                .map(|(_, _, p)| p.as_slice())
                .chain(relocs.iter().map(|(_, p, _)| p.as_slice()))
                .collect();
            // Fused GC reuses freed regions scattered across the file, so its writes
            // are random — parallel writers (distinct offsets, no tail contention)
            // beat the single stream there. Plain sequential appends stay single
            // (one monotonic stream already hits the device's seq rate; concurrent
            // appends to the tail only contend). `MPT_PARALLEL_WRITE=1` forces it on.
            let ptrs = if opp || parallel_write() {
                store.write_batch_parallel(&payloads)?
            } else {
                store.write_batch(&payloads)?
            };
            let write_ns = t_write.elapsed().as_nanos() as u64;
            stats::on_one_writer(read_ns, write_ns);
            let mut res: Vec<(Key, RamChild)> = promoted;
            res.reserve(nlv);
            for ((rep, root, _), ptr) in leaves.iter().zip(&ptrs[..nlv]) {
                res.push((*rep, RamChild::Disk { ptr: *ptr, root: *root }));
            }
            let mut reloc_out: Vec<(Vec<u8>, DiskPtr)> = Vec::with_capacity(relocs.len());
            for ((prefix, _, old), ptr) in relocs.iter().zip(&ptrs[nlv..]) {
                store.free(*old);
                reloc_out.push((prefix.clone(), *ptr));
            }
            if !relocs.is_empty() {
                stats::on_gc(0, relocs.len() as u64, 0);
            }
            results = res;
            reloc = reloc_out;
        } else if opp {
            let (res, rl) = process_opportunistic(&self.store, &self.upper, &cfg, &groups, batched)?;
            stats::on_gc(0, rl.len() as u64, 0);
            results = res;
            reloc = rl;
        } else {
            let store = &self.store;
            results = if groups.len() < 64 {
                process_chunk(store, &cfg, &groups, batched)?
            } else {
                let threads = worker_count();
                let chunk = groups.len().div_ceil(threads);
                std::thread::scope(|scope| {
                    let handles: Vec<_> = groups
                        .chunks(chunk)
                        .map(|c| {
                            scope.spawn(|| {
                                if coalesce {
                                    process_chunk_coalesced(store, &cfg, c, batched)
                                } else {
                                    process_chunk(store, &cfg, c, batched)
                                }
                            })
                        })
                        .collect();
                    let mut out = Vec::with_capacity(groups.len());
                    for h in handles {
                        out.extend(h.join().expect("batch group thread panicked")?);
                    }
                    Ok::<_, anyhow::Error>(out)
                })?
            };
            // Inline GC: evacuate live records out of the emptiest regions, skipping
            // the records this batch already rewrote (deduped by unit).
            let fg_units: std::collections::HashSet<u32> =
                groups.iter().map(|(ptr, _, _)| ptr.unit).collect();
            let t_gc = std::time::Instant::now();
            // GC runs every batch (including large/coalesced ones) so the file stays
            // bounded; `MPT_GC_DISABLE=1` turns it off explicitly for one-shot bulk
            // loads that don't care about reclaim.
            let r = self.gc_rate();
            let victims = self.store.seg.lock().unwrap().select_victims(r);
            reloc = if victims.is_empty() {
                Vec::new()
            } else {
                evacuate_regions(&self.store, &self.upper, &victims, &fg_units)?
            };
            stats::on_gc(victims.len() as u64, reloc.len() as u64, t_gc.elapsed().as_nanos() as u64);
        }

        let b_ns = t_b.elapsed().as_nanos() as u64;
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
        for (key, value) in fresh {
            // Disk path (RAM mode early-returns via the fan-out below), so new
            // structure is disk-backed.
            insert_ram(&self.store, &cfg, &mut self.upper, Vec::new(), key, LeafOp::Value(value), false)?;
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

    /// RAM-build insert: partition the (deduped) keys by top nibble and fan the
    /// serial `insert_ram` across the top branch's disjoint child subtrees, one
    /// thread per nibble. Each leaf lives under exactly one nibble ⇒ exactly one
    /// thread, so multiple keys to the same leaf are handled serially in that
    /// thread — no contention, no lock, no shared store (each leaf is its own
    /// `Arc`, freed by drop). Falls back to serial while the top isn't yet a branch
    /// (the tiny early phase).
    fn insert_batch_ram(&mut self, leaves: BTreeMap<Key, Vec<u8>>) -> Result<()> {
        let cfg = self.cfg.clone();
        let mut buckets: [Vec<(Key, Vec<u8>)>; 16] = std::array::from_fn(|_| Vec::new());
        for (key, vh) in leaves {
            buckets[nibble_at(&key, 0) as usize].push((key, vh));
        }
        // Fan the top into a 16-way branch with a single serial insert if it isn't
        // one yet, so the *rest* of even the first batch takes the parallel path.
        // Without this, a fresh build runs its entire first batch serially (a 100M
        // first batch = minutes on one core) just to bootstrap the top branch.
        if !matches!(self.upper, RamNode::Branch { .. }) {
            for b in buckets.iter_mut() {
                if let Some((key, vh)) = b.pop() {
                    insert_ram(&self.store, &cfg, &mut self.upper, Vec::new(), key, LeafOp::Value(vh), true)?;
                    break;
                }
            }
        }
        if matches!(self.upper, RamNode::Branch { .. }) {
            let store = &self.store;
            if let RamNode::Branch { children, hash } = &mut self.upper {
                hash.set(None); // top branch re-hashed after the parallel inserts
                std::thread::scope(|scope| -> Result<()> {
                    let cfg = &cfg;
                    let mut handles = Vec::new();
                    for (k, slot) in children.iter_mut().enumerate() {
                        let keys = std::mem::take(&mut buckets[k]);
                        if keys.is_empty() {
                            continue;
                        }
                        handles.push(scope.spawn(move || -> Result<()> {
                            for (key, vh) in keys {
                                insert_into_child(store, cfg, slot, vec![k as u8], key, LeafOp::Value(vh), true)?;
                            }
                            Ok(())
                        }));
                    }
                    for h in handles {
                        h.join().expect("RAM fan-out thread panicked")?;
                    }
                    Ok(())
                })?;
            }
        } else {
            let store = &self.store;
            for (key, vh) in buckets.into_iter().flatten() {
                insert_ram(store, &cfg, &mut self.upper, Vec::new(), key, LeafOp::Value(vh), true)?;
            }
        }
        Ok(())
    }

    /// If a RAM build has crossed the resident-size threshold, spill its in-RAM
    /// leaves to disk and revert to disk mode for subsequent batches.
    fn maybe_spill(&mut self) -> Result<()> {
        if self.ram_mode && process_footprint_bytes() >= self.spill_threshold {
            self.spill_mem()?;
        }
        Ok(())
    }

    /// Write every in-RAM `Mem` leaf to a disk record and retarget its frontier
    /// slot to the resulting `Disk` ptr, then revert to disk mode. Streams in chunks
    /// (one dense `write_batch` each) so transient memory stays bounded; ptrs are
    /// installed by prefix after the walk (releasing the `&mut upper` borrow first).
    fn spill_mem(&mut self) -> Result<()> {
        self.ram_mode = false;
        const CHUNK: usize = 8192;
        let mut buf = SpillBuf {
            prefixes: Vec::new(),
            payloads: Vec::new(),
            installs: Vec::new(),
        };
        spill_walk(&mut self.upper, Vec::new(), &self.store, &mut buf, CHUNK)?;
        flush_spill_chunk(&self.store, &mut buf)?;
        for (prefix, ptr) in buf.installs {
            install_ptr_by_prefix(&mut self.upper, &prefix, 0, ptr);
        }
        Ok(())
    }

    /// Proportional controller for the inline-GC cleaning rate. Nudges
    /// `gc_regions` (victims/batch) toward holding `live / active` at
    /// `TARGET_UTIL`: below target ⇒ too much garbage ⇒ clean more; above ⇒ ease
    /// off. Returns the rate to use this batch (0 until the file passes the floor).
    fn gc_rate(&mut self) -> usize {
        // Kill switch for A/B comparison: MPT_GC_DISABLE=1 turns off all inline
        // compaction (no victims selected, no relocation), so the flat file only
        // ever grows. Cached once.
        static GC_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *GC_OFF.get_or_init(|| std::env::var("MPT_GC_DISABLE").as_deref() == Ok("1")) {
            return 0;
        }
        let end = self.store.end_page();
        if end < GC_MIN_PAGES {
            return 0;
        }
        let (live_units, free_units) = self.store.live_and_free_units();
        let active = (end * UNITS_PER_PAGE).saturating_sub(free_units).max(1);
        let u = live_units as f64 / active as f64;
        let adj = ((TARGET_UTIL - u) * GC_GAIN).round() as i64;
        let r = (self.gc_regions as i64 + adj).clamp(0, GC_R_MAX as i64) as usize;
        self.gc_regions = r;
        r
    }

    pub fn get_value(&self, key: &Key) -> Result<Option<Vec<u8>>> {
        let _g = prof::scope(prof::Cat::ValueGet);
        // The value lives in its trie leaf (not a side store): walk the RAM
        // frontier to the record that owns this key, then descend that record's
        // node tree to the leaf.
        let nibbles = key_nibbles(key);
        ram_get(&self.store, &self.upper, &nibbles, 0)
    }

    /// Read a storage slot of the account at `account_key`: descend to the account
    /// leaf, then into its nested storage subtree at `slot_key`. `None` if the
    /// account or slot is absent, or the account is a raw (opaque-storage) leaf.
    pub fn get_storage(&self, account_key: &Key, slot_key: &Key) -> Result<Option<Vec<u8>>> {
        let nibbles = key_nibbles(account_key);
        ram_get_storage(&self.store, &self.upper, &nibbles, 0, slot_key)
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
        self.store.garbage_bytes()
    }

    /// Number of fully-reclaimed regions available for reuse.
    pub fn free_regions(&self) -> usize {
        self.store.free_region_count()
    }

    /// Live on-disk bytes — the dense footprint of the records the frontier points
    /// at (`live_units × 256`). This is the working set that reads must hit; the
    /// key scaling metric (vs `flat_file_len`, the high-water that includes
    /// garbage). Excludes the per-page write rounding within a region.
    pub fn live_bytes(&self) -> u64 {
        self.store.live_and_free_units().0 * ADDR_UNIT
    }

    /// Active-file utilization: `live / (end − reclaimed free regions)`. The
    /// inline-GC controller drives this toward `TARGET_UTIL`.
    pub fn utilization(&self) -> f64 {
        let (live_units, free_units) = self.store.live_and_free_units();
        let active = (self.store.end_page() * UNITS_PER_PAGE).saturating_sub(free_units);
        if active == 0 {
            0.0
        } else {
            live_units as f64 / active as f64
        }
    }

    /// Current GC cleaning rate (victim regions/batch the controller has settled on).
    pub fn gc_rate_current(&self) -> usize {
        self.gc_regions
    }

    /// Debug audit: (allocator's tracked live units, true live units recomputed
    /// from the frontier). They must be equal after a batch completes; a positive
    /// gap means records were superseded without `free()` — a live-accounting leak
    /// that creates unreclaimable "zombie" regions.
    pub fn audit_live_units(&self) -> (u64, u64) {
        let alloc_live = self.store.live_and_free_units().0;
        let num_regions = self.store.end_page().div_ceil(REGION_PAGES) as usize + 1;
        let mut live = vec![0u32; num_regions];
        recompute_live(&self.upper, &mut live);
        let true_live: u64 = live.iter().map(|&u| u as u64).sum();
        (alloc_live, true_live)
    }

    /// Heap held by the in-RAM index — the part of the database that is *not* on
    /// disk: the trie frontier and the region allocator. Excludes the OS page cache.
    pub fn ram_report(&self) -> RamReport {
        let frontier_nodes = count_ram_nodes(&self.upper);
        let frontier_bytes = frontier_bytes(&self.upper);
        let free_regions = self.store.free_region_count();
        // The region allocator is one u32 of live-count per region; report that.
        let free_list_bytes =
            self.store.seg.lock().unwrap().live.len() * std::mem::size_of::<u32>();
        RamReport {
            frontier_nodes,
            frontier_bytes,
            free_regions,
            free_list_bytes,
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

/// Point-read the value for `key` by walking the RAM frontier down to the record
/// that owns it, then descending that record's node tree. Returns `None` if the
/// key is absent. `depth` is the nibble index reached so far.
fn ram_get(store: &FlatFile, node: &RamNode, nibbles: &[u8], depth: usize) -> Result<Option<Vec<u8>>> {
    match node {
        RamNode::Empty => Ok(None),
        RamNode::Extension { path, child, .. } => {
            if nibbles[depth..].starts_with(path) {
                ram_get(store, child, nibbles, depth + path.len())
            } else {
                Ok(None)
            }
        }
        RamNode::Branch { children, .. } => {
            let idx = nibbles[depth] as usize;
            match &children[idx] {
                None => Ok(None),
                Some(RamChild::Ram(child)) => ram_get(store, child, nibbles, depth + 1),
                Some(RamChild::Disk { ptr, .. }) => {
                    let sub = store.read_lazy(*ptr)?;
                    node_get(store, &sub.node, nibbles, sub.prefix.len())
                }
                Some(RamChild::Mem(m)) => {
                    let sub = parse_payload_lazy(m.bytes.clone())?;
                    node_get(store, &sub.node, nibbles, sub.prefix.len())
                }
                // A generic value read of a promoted account returns its account RLP.
                // The child sits one nibble below this branch (like the Ram/Disk arms).
                Some(RamChild::Account(a)) => {
                    if nibbles[depth + 1..] == a.path[..] {
                        let sr = hash_ram(&a.storage);
                        Ok(Some(ram_account_rlp(a.nonce, &a.balance, &a.code_hash, sr)))
                    } else {
                        Ok(None)
                    }
                }
            }
        }
    }
}

/// Descend a record's node tree following `nibbles` from `depth`, returning the
/// leaf's value if the key is present. Expands `Raw`/`Overflow` edges on demand.
fn node_get(store: &FlatFile, node: &Node, nibbles: &[u8], depth: usize) -> Result<Option<Vec<u8>>> {
    match node {
        Node::Empty => Ok(None),
        Node::Leaf { path, value, .. } => {
            Ok((nibbles[depth..] == path[..]).then(|| value.clone()))
        }
        // A generic value read of an account key returns the account RLP (the leaf
        // value). Storage is read via the account's nested subtree (`storage_get`).
        Node::Account { path, nonce, balance, code_hash, storage_root, .. } => {
            Ok((nibbles[depth..] == path[..]).then(|| {
                eth::Account {
                    nonce: *nonce,
                    balance: *balance,
                    storage_root: B256::from(*storage_root),
                    code_hash: B256::from(*code_hash),
                }
                .rlp()
            }))
        }
        Node::Extension { path, child, .. } => {
            if nibbles[depth..].starts_with(path) {
                node_get(store, child, nibbles, depth + path.len())
            } else {
                Ok(None)
            }
        }
        Node::Branch { children, .. } => {
            let idx = nibbles[depth] as usize;
            match &children[idx] {
                Some(child) => node_get(store, child, nibbles, depth + 1),
                None => Ok(None),
            }
        }
        Node::Overflow { ptr, .. } => {
            let sub = store.read_lazy(*ptr)?;
            node_get(store, &sub.node, nibbles, sub.prefix.len())
        }
        Node::Raw { buf, off, len, .. } => {
            let n = parse_node_lazy(buf, *off, *len)?;
            node_get(store, &n, nibbles, depth)
        }
    }
}

/// Walk the RAM frontier toward the account at `nibbles`, then read `slot_key` from
/// its nested storage. Mirrors [`ram_get`] but descends into the account's storage.
fn ram_get_storage(
    store: &FlatFile,
    node: &RamNode,
    nibbles: &[u8],
    depth: usize,
    slot_key: &Key,
) -> Result<Option<Vec<u8>>> {
    match node {
        RamNode::Empty => Ok(None),
        RamNode::Extension { path, child, .. } => {
            if nibbles[depth..].starts_with(path) {
                ram_get_storage(store, child, nibbles, depth + path.len(), slot_key)
            } else {
                Ok(None)
            }
        }
        RamNode::Branch { children, .. } => match &children[nibbles[depth] as usize] {
            None => Ok(None),
            Some(RamChild::Ram(child)) => ram_get_storage(store, child, nibbles, depth + 1, slot_key),
            Some(RamChild::Disk { ptr, .. }) => {
                let sub = store.read_lazy(*ptr)?;
                node_get_storage(store, &sub.node, nibbles, sub.prefix.len(), slot_key)
            }
            Some(RamChild::Mem(m)) => {
                let sub = parse_payload_lazy(m.bytes.clone())?;
                node_get_storage(store, &sub.node, nibbles, sub.prefix.len(), slot_key)
            }
            // Promoted account (one nibble below this branch): descend its RAM
            // storage frontier with the slot key.
            Some(RamChild::Account(a)) => {
                if nibbles[depth + 1..] == a.path[..] {
                    ram_get(store, &a.storage, &key_nibbles(slot_key), 0)
                } else {
                    Ok(None)
                }
            }
        },
    }
}

/// Descend a record to the account leaf at `nibbles`, then read `slot_key` from its
/// nested storage subtree. `None` if the account/slot is absent or the leaf is a
/// raw (opaque-storage) value leaf rather than an account node.
fn node_get_storage(
    store: &FlatFile,
    node: &Node,
    nibbles: &[u8],
    depth: usize,
    slot_key: &Key,
) -> Result<Option<Vec<u8>>> {
    match node {
        Node::Empty | Node::Leaf { .. } => Ok(None),
        Node::Account { path, storage, .. } => {
            if nibbles[depth..] == path[..] {
                node_get(store, storage, &key_nibbles(slot_key), 0)
            } else {
                Ok(None)
            }
        }
        Node::Extension { path, child, .. } => {
            if nibbles[depth..].starts_with(path) {
                node_get_storage(store, child, nibbles, depth + path.len(), slot_key)
            } else {
                Ok(None)
            }
        }
        Node::Branch { children, .. } => match &children[nibbles[depth] as usize] {
            Some(child) => node_get_storage(store, child, nibbles, depth + 1, slot_key),
            None => Ok(None),
        },
        Node::Overflow { ptr, .. } => {
            let sub = store.read_lazy(*ptr)?;
            node_get_storage(store, &sub.node, nibbles, sub.prefix.len(), slot_key)
        }
        Node::Raw { buf, off, len, .. } => {
            let n = parse_node_lazy(buf, *off, *len)?;
            node_get_storage(store, &n, nibbles, depth, slot_key)
        }
    }
}

/// Sibling path for the manifest, e.g. `db.flat` -> `db.flat.meta`.
fn meta_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".meta");
    path.with_file_name(name)
}

/// Serialize a subtree and wrap the bytes in their own `Arc` — a RAM-build leaf.
fn make_mem_leaf(subtree: &DiskSubtree) -> Result<RamChild> {
    let root = hash_node(&subtree.node).finalize();
    let (payload, _) = serialize_subtree(subtree)?;
    Ok(RamChild::Mem(MemLeaf { bytes: Arc::from(payload), root }))
}

/// Build a leaf child: a RAM-resident `Mem` leaf in RAM-build mode (no I/O), or a
/// serialized `Disk` record otherwise.
fn make_leaf_child(store: &FlatFile, ram: bool, subtree: DiskSubtree) -> Result<RamChild> {
    if ram {
        make_mem_leaf(&subtree)
    } else {
        let root = hash_node(&subtree.node).finalize();
        let (payload, _) = serialize_subtree(&subtree)?;
        let ptr = store.write_payload(&payload)?;
        Ok(RamChild::Disk { ptr, root })
    }
}

/// Parse a `Mem` leaf's bytes (lazy: children stay `Raw`, zero-copy slices of the
/// `Arc`) — the in-RAM equivalent of [`FlatFile::read_lazy`].
fn parse_payload_lazy(bytes: Arc<[u8]>) -> Result<DiskSubtree> {
    deserialize_subtree_lazy(bytes)
}

/// Like [`promote_record_to_ram`], but the lifted branch's children become in-RAM
/// `Mem` leaves (each its own `Arc`) instead of serialized `Disk` records — no I/O.
fn promote_to_mem(subtree: DiskSubtree) -> Result<RamChild> {
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
        if let Some(boxed) = slot {
            let mut cp = branch_prefix.clone();
            cp.push(i as u8);
            ram_children[i] = Some(make_mem_leaf(&DiskSubtree { prefix: cp, node: *boxed })?);
        }
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

/// Insert `key` into a branch slot — the per-slot body of `insert_ram`'s branch
/// arm, factored out so the parallel RAM fan-out (which splits the top branch's
/// children across threads) shares the exact same logic. `ram` selects whether a
/// newly-created or rewritten leaf is a `Mem` (RAM) or `Disk` leaf.
fn insert_into_child(
    store: &FlatFile,
    cfg: &Config,
    slot: &mut Option<RamChild>,
    child_prefix: Vec<u8>,
    key: Key,
    op: LeafOp,
    ram: bool,
) -> Result<()> {
    match slot {
        Some(RamChild::Ram(child)) => insert_ram(store, cfg, child, child_prefix, key, op, ram),
        Some(RamChild::Mem(_)) => {
            // In-RAM leaf: parse its bytes, apply the key, re-serialize into a new
            // `Arc` (the old one drops here). No I/O, no shared store.
            let Some(RamChild::Mem(m)) = slot.take() else { unreachable!() };
            let mut subtree = parse_payload_lazy(m.bytes)?;
            let depth = subtree.prefix.len();
            record_node_insert(store, cfg, &mut subtree.node, depth, key, op)?;
            *slot = Some(if should_promote(cfg, &subtree) {
                promote_to_mem(subtree)?
            } else if should_promote_account(cfg, &subtree) {
                promote_account_to_ram(store, subtree)?
            } else {
                make_mem_leaf(&subtree)?
            });
            Ok(())
        }
        Some(RamChild::Disk { ptr, root }) => {
            let mut subtree = store.read_lazy(*ptr)?;
            let old_ptr = *ptr;
            record_node_insert(store, cfg, &mut subtree.node, subtree.prefix.len(), key, op)?;
            if should_promote(cfg, &subtree) {
                store.free(old_ptr);
                *slot = Some(promote_record_to_ram(store, subtree)?);
            } else if should_promote_account(cfg, &subtree) {
                store.free(old_ptr);
                *slot = Some(promote_account_to_ram(store, subtree)?);
            } else {
                let (payload, _) = serialize_subtree(&subtree)?;
                store.free(old_ptr);
                *ptr = store.write_payload(&payload)?;
                *root = hash_node(&subtree.node).finalize();
            }
            Ok(())
        }
        Some(RamChild::Account(_)) => {
            let nibbles = key_nibbles(&key);
            let (common, path_len) = match slot.as_ref() {
                Some(RamChild::Account(a)) => {
                    (common_prefix(&a.path, &nibbles[child_prefix.len()..]), a.path.len())
                }
                _ => unreachable!(),
            };
            if common == path_len {
                // Same account: apply the op in place (storage insert or field update).
                let Some(RamChild::Account(a)) = slot.as_mut() else { unreachable!() };
                match op {
                    LeafOp::Storage { slot: skey, value } => {
                        // Descend the account's RAM storage frontier with the slot key
                        // (storage records are Disk, so ram=false).
                        insert_ram(store, cfg, &mut a.storage, Vec::new(), skey, LeafOp::Value(value), false)?;
                    }
                    LeafOp::Account { nonce: n, balance: b, code_hash: ch } => {
                        a.nonce = n;
                        a.balance = b.to_be_bytes::<32>();
                        a.code_hash = ch;
                    }
                    LeafOp::Value(_) => bail!("value op on a promoted account leaf"),
                }
                Ok(())
            } else {
                // A different account diverges: split into a frontier branch —
                // re-home the promoted account under a shorter path, place the new
                // key alongside.
                let Some(RamChild::Account(mut a)) = slot.take() else { unreachable!() };
                let mut children = empty_children();
                let old_idx = a.path[common] as usize;
                let rehomed_path = a.path[common + 1..].to_vec();
                let split_path = a.path[..common].to_vec();
                a.path = rehomed_path;
                children[old_idx] = Some(RamChild::Account(a));
                let new_idx = nibbles[child_prefix.len() + common] as usize;
                let mut new_prefix = child_prefix.clone();
                new_prefix
                    .extend_from_slice(&nibbles[child_prefix.len()..child_prefix.len() + common + 1]);
                let node = place_new(store, cfg, key, op, new_prefix.len())?;
                children[new_idx] =
                    Some(make_leaf_child(store, ram, DiskSubtree { prefix: new_prefix, node })?);
                let branch = RamNode::Branch { children, hash: HashCell::new(None) };
                *slot = Some(if common == 0 {
                    RamChild::Ram(Box::new(branch))
                } else {
                    RamChild::Ram(Box::new(RamNode::Extension {
                        path: split_path,
                        child: Box::new(branch),
                        hash: HashCell::new(None),
                    }))
                });
                Ok(())
            }
        }
        None => {
            let node = place_new(store, cfg, key, op, child_prefix.len())?;
            let subtree = DiskSubtree { prefix: child_prefix, node };
            *slot = Some(make_leaf_child(store, ram, subtree)?);
            Ok(())
        }
    }
}

fn insert_ram(
    store: &FlatFile,
    cfg: &Config,
    node: &mut RamNode,
    prefix: Vec<u8>,
    key: Key,
    op: LeafOp,
    ram: bool,
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
            let node_new = place_new(store, cfg, key, op, child_prefix.len())?;
            let subtree = DiskSubtree { prefix: child_prefix, node: node_new };
            let mut children = empty_children();
            children[idx] = Some(make_leaf_child(store, ram, subtree)?);
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
                let node_new = place_new(store, cfg, key, op, new_prefix.len())?;
                let subtree = DiskSubtree { prefix: new_prefix, node: node_new };
                children[new_idx] = Some(make_leaf_child(store, ram, subtree)?);

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
                insert_ram(store, cfg, child, next_prefix, key, op, ram)
            }
        }
        RamNode::Branch { children, .. } => {
            if prefix.len() == nibbles.len() {
                bail!("key terminates at a frontier branch; keys must be distinct and fixed-length");
            }
            let idx = nibbles[prefix.len()] as usize;
            let mut child_prefix = prefix;
            child_prefix.push(idx as u8);
            insert_into_child(store, cfg, &mut children[idx], child_prefix, key, op, ram)
        }
    }
}


/// The `i`-th nibble of a key (i in 0..64), without allocating.
fn nibble_at(key: &Key, i: usize) -> u8 {
    let byte = key[i / 2];
    if i % 2 == 0 { byte >> 4 } else { byte & 0x0f }
}






// --- Disk-node constructors: compute and cache the node hash exactly once. ---

/// A leaf holding the suffix `path` (key nibbles from its position to depth 64)
/// and its `value`, computing its Ethereum node reference (`leaf_ref`). Recomputes
/// the reference from `path` + `value`, so callers can re-home a leaf to a new
/// depth just by handing it a shorter `path`.
fn leaf_node(path: Vec<u8>, value: Vec<u8>) -> Node {
    let nref = leaf_ref(&path, &value);
    Node::Leaf { path, value, nref }
}

/// Build an account leaf: compute its storage root from the nested `storage`
/// subtree, fold `RLP([nonce, balance, storage_root, code_hash])` into the leaf
/// value, and cache the storage root + node reference. Self-contained — hashing an
/// account never touches a value store.
fn account_node(path: Vec<u8>, nonce: u64, balance: U256, code_hash: Hash, storage: Node) -> Node {
    let storage_root = hash_node(&storage).finalize();
    let account_rlp = eth::Account {
        nonce,
        balance,
        storage_root: B256::from(storage_root),
        code_hash: B256::from(code_hash),
    }
    .rlp();
    let nref = leaf_ref(&path, &account_rlp);
    Node::Account {
        path,
        nonce,
        balance,
        code_hash,
        storage: Box::new(storage),
        storage_root,
        nref,
    }
}

/// The RLP of a promoted account, given its (already-computed) storage root.
fn ram_account_rlp(nonce: u64, balance: &[u8; 32], code_hash: &Hash, storage_root: Hash) -> Vec<u8> {
    eth::Account {
        nonce,
        balance: U256::from_be_bytes(*balance),
        storage_root: B256::from(storage_root),
        code_hash: B256::from(*code_hash),
    }
    .rlp()
}

/// Node reference of a promoted-account frontier leaf: fold the storage root
/// (`hash_ram(storage)`) into the account RLP, then the leaf reference over `path`.
fn ram_account_ref(
    path: &[u8],
    nonce: u64,
    balance: &[u8; 32],
    code_hash: &Hash,
    storage: &RamNode,
) -> NodeRef {
    let storage_root = hash_ram(storage);
    leaf_ref(path, &ram_account_rlp(nonce, balance, code_hash, storage_root))
}

fn make_extension(path: Vec<u8>, child: Node) -> Node {
    let nref = ext_ref(&path, &hash_node(&child));
    Node::Extension {
        path,
        child: Box::new(child),
        nref,
    }
}

/// Ethereum branch node reference over the 16 child slots (see `branch_ref`). An
/// `Overflow` child contributes its `root` via `hash_node`, so a branch references
/// identically whether a child is inline or overflowed.
fn branch_node_ref(children: &[Option<Box<Node>>; 16]) -> NodeRef {
    let bitmap = branch_bitmap(children.iter().map(|c| c.is_some()));
    branch_ref(bitmap, children.iter().flatten().map(|c| hash_node(c)))
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
    let nref = branch_node_ref(&children);
    Node::Branch { children, nref }
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
        Node::Leaf { path, .. } | Node::Account { path, .. } => {
            PathEnd::Inline(key_nibbles(key)[depth..] == path[..])
        }
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

/// What to place/update at a key's leaf position. The state trie inserts
/// `Account`/`Storage` ops (producing [`Node::Account`] leaves); a storage subtree
/// inserts `Value` ops (producing [`Node::Leaf`] value leaves) — the same descent
/// (`record_node_insert`) serves both levels.
enum LeafOp {
    /// Place/overwrite a generic value leaf (storage-trie slot, or a plain trie).
    Value(Vec<u8>),
    /// Place/update an account leaf, preserving any existing storage subtree.
    Account { nonce: u64, balance: U256, code_hash: Hash },
    /// Set one storage slot of the account at this key (auto-creates an empty
    /// account if none exists), by inserting into the account's nested storage.
    Storage { slot: Key, value: Vec<u8> },
}

/// Create a fresh node for a brand-new key at `depth` from a [`LeafOp`].
fn place_new(store: &FlatFile, cfg: &Config, key: Key, op: LeafOp, depth: usize) -> Result<Node> {
    let path = key_nibbles(&key)[depth..].to_vec();
    Ok(match op {
        LeafOp::Value(value) => leaf_node(path, value),
        LeafOp::Account { nonce, balance, code_hash } => {
            account_node(path, nonce, balance, code_hash, Node::Empty)
        }
        LeafOp::Storage { slot, value } => {
            // Storage on a missing account: create an empty account holding the slot.
            let mut storage = Node::Empty;
            record_node_insert(store, cfg, &mut storage, 0, slot, LeafOp::Value(value))?;
            account_node(path, 0, U256::ZERO, eth::EMPTY_CODE_HASH.0, storage)
        }
    })
}

/// Re-home an existing leaf (value or account) under a shorter suffix `path` after
/// a divergence — recomputing its reference (which depends on `path`).
fn rehome(node: Node, path: Vec<u8>) -> Node {
    match node {
        Node::Leaf { value, .. } => leaf_node(path, value),
        Node::Account { nonce, balance, code_hash, storage, .. } => {
            account_node(path, nonce, balance, code_hash, *storage)
        }
        _ => unreachable!("only leaf/account nodes are re-homed on divergence"),
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
    op: LeafOp,
) -> Result<()> {
    // Expand a lazily-unparsed subtree one level before navigating into it
    // (children become `Raw` again, so deeper untouched subtrees stay unparsed).
    if let Node::Raw { buf, off, len, .. } = node {
        *node = parse_node_lazy(buf, *off, *len)?;
    }
    let nibbles = key_nibbles(&key);
    let updated = match std::mem::replace(node, Node::Empty) {
        Node::Empty => place_new(store, cfg, key, op, depth)?,
        old @ (Node::Leaf { .. } | Node::Account { .. }) => {
            let path = match &old {
                Node::Leaf { path, .. } | Node::Account { path, .. } => path.clone(),
                _ => unreachable!(),
            };
            let remaining = &nibbles[depth..];
            let common = common_prefix(&path, remaining);
            if common == path.len() {
                // Same key (both suffixes run to depth 64): apply the op in place.
                debug_assert_eq!(path.as_slice(), remaining);
                apply_op(store, cfg, old, path, op)?
            } else {
                // The new key diverges partway along the leaf's suffix. Re-home the
                // old leaf under a shorter suffix and add the new node alongside.
                let mut children = empty_box_children();
                let old_idx = path[common] as usize;
                children[old_idx] = Some(Box::new(rehome(old, path[common + 1..].to_vec())));
                let new_idx = remaining[common] as usize;
                children[new_idx] =
                    Some(Box::new(place_new(store, cfg, key, op, depth + common + 1)?));
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
                record_node_insert(store, cfg, &mut child, depth + path.len(), key, op)?;
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
                    Some(Box::new(place_new(store, cfg, key, op, depth + common + 1)?));
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
                Some(Node::Overflow { .. }) => {
                    // Option B never creates on-disk overflow children; records are
                    // promoted into the RAM frontier instead. So this is unreachable
                    // on data written by this build.
                    unreachable!("on-disk Overflow under promote-on-max (option B)")
                }
                Some(child) => record_node_insert(store, cfg, child, depth + 1, key, op)?,
                None => {
                    children[idx] = Some(Box::new(place_new(store, cfg, key, op, depth + 1)?));
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

/// Apply a [`LeafOp`] to the existing node at an exact key match.
fn apply_op(store: &FlatFile, cfg: &Config, old: Node, path: Vec<u8>, op: LeafOp) -> Result<Node> {
    Ok(match (old, op) {
        // Overwrite a value leaf.
        (Node::Leaf { .. }, LeafOp::Value(value)) => leaf_node(path, value),
        // Update account fields, keeping the existing storage subtree.
        (Node::Account { storage, .. }, LeafOp::Account { nonce, balance, code_hash }) => {
            account_node(path, nonce, balance, code_hash, *storage)
        }
        // Set one storage slot: insert into the account's nested storage, re-fold.
        (
            Node::Account { nonce, balance, code_hash, mut storage, .. },
            LeafOp::Storage { slot, value },
        ) => {
            record_node_insert(store, cfg, &mut storage, 0, slot, LeafOp::Value(value))?;
            account_node(path, nonce, balance, code_hash, *storage)
        }
        _ => bail!("leaf op does not match the existing node kind at this key"),
    })
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

/// Whether a record should be **promoted** into the RAM frontier rather than kept
/// as a single packed disk leaf (option B). A record is promoted once it grows past
/// `max_leaf_bytes`: its top branch lifts into RAM and each child becomes its own
/// frontier `RamChild::Disk`. This replaces the old on-disk `Overflow` shedding —
/// which a compacting GC can't tolerate, because moving an overflow child would
/// require rewriting the on-disk pointer inside its (scattered) parent. With
/// promotion, every disk record is a frontier leaf and inter-record pointers live
/// in RAM, so GC can relocate any record by updating a RAM pointer.
fn should_promote(cfg: &Config, subtree: &DiskSubtree) -> bool {
    record_size(subtree.prefix.len(), &subtree.node) > cfg.max_leaf_bytes
        && top_branch_prefix(subtree).is_some()
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

    // Serialize the inline children and write them all in ONE dense batch (these
    // children are small — well under a page — so writing them individually would
    // page-pad each to 16 KiB and balloon the file). Overflow children (none under
    // option B) keep their existing record.
    let mut ram_children = empty_children();
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut batched_slots: Vec<(usize, Hash)> = Vec::new();
    for (i, slot) in children.into_iter().enumerate() {
        let Some(boxed) = slot else { continue };
        match *boxed {
            Node::Overflow { ptr, root } => {
                ram_children[i] = Some(RamChild::Disk { ptr, root });
            }
            other => {
                let mut cp = branch_prefix.clone();
                cp.push(i as u8);
                let root = hash_node(&other).finalize();
                let (payload, _) = serialize_subtree(&DiskSubtree { prefix: cp, node: other })?;
                payloads.push(payload);
                batched_slots.push((i, root));
            }
        }
    }
    let payload_refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    stats::on_promote(
        payload_refs.len() as u64,
        payload_refs.iter().map(|p| p.len() as u64).sum(),
    );
    let ptrs = store.write_batch(&payload_refs)?;
    for ((i, root), ptr) in batched_slots.into_iter().zip(ptrs) {
        ram_children[i] = Some(RamChild::Disk { ptr, root });
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

/// Whether a single-account record should be promoted: it exceeds `max_leaf_bytes`
/// and its account's storage has a top branch to lift (so the storage trie can span
/// multiple records with the pointers held in RAM — GC-relocatable). A record whose
/// top is a branch (multiple accounts / generic) uses [`should_promote`] instead.
fn should_promote_account(cfg: &Config, subtree: &DiskSubtree) -> bool {
    if record_size(subtree.prefix.len(), &subtree.node) <= cfg.max_leaf_bytes {
        return false;
    }
    match &subtree.node {
        Node::Account { storage, .. } => matches!(
            storage.as_ref(),
            Node::Branch { .. }
                | Node::Extension { .. } // extension always wraps a branch here
        ),
        _ => false,
    }
}

/// Lift a storage subtree (with a top branch) into a RAM frontier node whose
/// children are `Disk` storage records — the storage analogue of
/// [`promote_record_to_ram`]. The pointers now live in RAM, so GC can relocate any
/// storage record by updating a RAM pointer.
fn promote_storage_to_ram(store: &FlatFile, storage: Node) -> Result<Box<RamNode>> {
    match promote_record_to_ram(store, DiskSubtree { prefix: Vec::new(), node: storage })? {
        RamChild::Ram(ram) => Ok(ram),
        _ => unreachable!("promote_record_to_ram returns a Ram node"),
    }
}

/// Promote a single-account record into a `RamChild::Account`: the account's storage
/// trie is lifted into a RAM subtree (its top branch's children become `Disk`
/// storage records), and the account fields move into the frontier node. The account
/// leaf hashes identically (same fields + same storage root).
fn promote_account_to_ram(store: &FlatFile, subtree: DiskSubtree) -> Result<RamChild> {
    let Node::Account { path, nonce, balance, code_hash, storage, .. } = subtree.node else {
        unreachable!("promote_account_to_ram called on a non-account record");
    };
    let storage_ram = promote_storage_to_ram(store, *storage)?;
    Ok(RamChild::Account(Box::new(PromotedAccount {
        path,
        nonce,
        balance: balance.to_be_bytes::<32>(),
        code_hash,
        storage: storage_ram,
    })))
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

/// Fold (sequential-merge) tuning. When a batch touches many leaves, the disk
/// path sorts the per-leaf groups by file offset and reads consecutive ones in a
/// single `pread` spanning up to `FOLD_SPAN_BYTES`, coalescing across gaps
/// ≤ `fold_max_gap()`. Coalescing trades extra bytes read (the untouched records
/// in the gap) for fewer syscalls/seeks; it only wins when touched leaves are
/// dense, so the gap defaults to 0 (see `fold_max_gap`). All leaves in a span
/// share one buffer `Arc` and parse zero-copy. Inline GC runs on every batch
/// regardless of size (use `MPT_GC_DISABLE=1` for gc-off bulk loads).
const FOLD_SPAN_BYTES: u64 = 8 << 20; // ≤ 8 MiB per coalesced read
/// Coalesce consecutive leaf reads across on-disk gaps ≤ this many bytes
/// (default 0 = don't coalesce). Merging two touched leaves into one `pread`
/// also reads the untouched records between them, which only pays off when
/// touched leaves are densely/sequentially placed. For sparse random-key
/// workloads (the common case) it reads large spans of dead data: measured
/// ~2x slower end-to-end at a 1 MiB gap vs 0 on NVMe (35 KiB avg device read
/// vs ~8 KiB), because the device is read-bandwidth bound. Tunable via
/// `MPT_FOLD_GAP_KIB` for workloads with sequential insert locality.
fn fold_max_gap() -> u64 {
    use std::sync::OnceLock;
    static G: OnceLock<u64> = OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("MPT_FOLD_GAP_KIB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|kib| kib * 1024)
            .unwrap_or(0)
    })
}
const FOLD_WRITE_LEAVES: usize = 256; // flush threshold on the batched write path

/// Whether to coalesce leaf writes into batched `pwrite`s. The append allocator
/// places every write sequentially regardless; batching just cuts syscalls. On by
/// default; `MPT_BATCHED_WRITES=0` writes per-record for comparison.
fn batched_writes() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_BATCHED_WRITES").ok().as_deref() != Some("0"))
}

/// `MPT_ONE_WRITER=1`: split the disk batch path into many parallel readers that
/// read+fold leaves into payloads, then ONE writer that appends them all in a
/// single sequential `write_batch`. N concurrent appends contend on the file and
/// run *slower* than one stream (~1.1 vs ~3.5 GB/s measured), so funnelling the
/// writes through one stream is ~1.5x faster overall. Phased (read-all then
/// write-all) beats overlapping them — both share the one device. No inline GC yet.
/// `MPT_PARALLEL_WRITE=1`: in the one-writer path, fan the write phase's per-run
/// `pwrite`s across worker threads ([`write_batch_parallel`]) instead of one serial
/// stream. Helps when the writes are scattered (bounded-file GC reusing freed
/// regions) — random writes are queue-depth-bound. Opt-in for A/B.
fn parallel_write() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_PARALLEL_WRITE").as_deref() == Ok("1"))
}

fn one_writer() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_ONE_WRITER").as_deref() == Ok("1"))
}

/// Whether the disk batch path sorts groups by file offset and reads them via
/// coalesced multi-MB spans (the sequential-merge fold). On by default;
/// `MPT_FOLD=0` reverts to the original unsorted per-leaf `pread` path for A/B.
fn fold_coalesce() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_FOLD").ok().as_deref() != Some("0"))
}

/// `MPT_GC_OPP=1` switches inline GC to opportunistic mode: instead of selecting
/// the globally-emptiest regions (cold random reads that don't scale with read
/// QD), it evacuates only the regions this batch already read from that are under
/// `MPT_GC_OPP_UTIL` (default 0.30) live — those are page-cache-hot, so GC rides
/// the foreground reads. Overrides the rate-based selector and the bulk skip.
fn gc_opportunistic() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MPT_GC_OPP").as_deref() == Ok("1"))
}

fn gc_opp_util() -> f64 {
    use std::sync::OnceLock;
    static U: OnceLock<f64> = OnceLock::new();
    *U.get_or_init(|| {
        std::env::var("MPT_GC_OPP_UTIL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.30)
    })
}

/// RAM-build config read once: `(enabled, spill-threshold-bytes)`. `MPT_RAM_BUILD=1`
/// keeps fresh leaves in RAM (each its own `Arc`, no flat-file I/O or GC) until
/// memory footprint crosses the threshold, then spills to disk. `MPT_RAM_BUILD_GIB`
/// overrides the threshold. Defaults leave real headroom below installed RAM for
/// the page cache and the spill-time transient — the footprint per key runs higher
/// than a naive value-size estimate, so the threshold must trip before the box
/// gets tight.
fn ram_build_config() -> (bool, u64) {
    use std::sync::OnceLock;
    static CFG: OnceLock<(bool, u64)> = OnceLock::new();
    *CFG.get_or_init(|| {
        let on = std::env::var("MPT_RAM_BUILD").as_deref() == Ok("1");
        let default_gib: u64 = if cfg!(target_os = "macos") { 85 } else { 45 };
        let gib = std::env::var("MPT_RAM_BUILD_GIB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_gib);
        (on, gib * (1 << 30))
    })
}

/// Process memory footprint in bytes — the real committed memory *including*
/// compressed and swapped pages, not just what is currently resident in RAM.
///
/// This is what must drive spill decisions. Under memory pressure the resident
/// set is pinned at physical RAM while the true footprint keeps growing into the
/// compressor / swap, so a resident-only metric (`getrusage`'s `ru_maxrss`)
/// plateaus and never crosses the threshold — the build then thrashes into swap
/// instead of spilling. Measured directly in a 1B run: `ru_maxrss` sat at 35 GiB
/// while the real footprint was ~107 GiB (35 resident + 43 compressed + 29 swap).
///
/// macOS: `ri_phys_footprint` from `proc_pid_rusage` (what Activity Monitor's
/// "Memory" column reports; counts compressed memory). Linux: `VmRSS + VmSwap`
/// from `/proc/self/status`.
pub fn process_footprint_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut ri: libc::rusage_info_v0 = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V0,
                &mut ri as *mut _ as *mut libc::rusage_info_t,
            )
        };
        if rc == 0 { ri.ri_phys_footprint } else { 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // VmRSS + VmSwap from /proc/self/status (both in kB).
        let mut total = 0u64;
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                let rest = line
                    .strip_prefix("VmRSS:")
                    .or_else(|| line.strip_prefix("VmSwap:"));
                if let Some(rest) = rest {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) {
                        total += kb * 1024;
                    }
                }
            }
        }
        total
    }
}

/// Number of Phase-B worker threads (each issues one blocking pread at a time, so
/// this is effectively the read queue depth). Defaults to 192: the reads are
/// I/O-bound (threads block on pread), so we intentionally oversubscribe cores to
/// keep the device queue deep. Measured sweet spot on NVMe — read-phase µs/key
/// keeps falling to ~192 then regresses as the buffered-read IOPS ceiling
/// (~165K) saturates and extra threads only deepen the queue. `MPT_WORKERS=N`
/// overrides (e.g. to explore the QD curve or cap threads on a small box).
fn worker_count() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MPT_WORKERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(192)
    })
}

/// Apply a whole group of keys (all routing to the disk record at `ptr`), free
/// the old record, and return the outcome — *without* writing the new leaf (the
/// caller writes it, in place or batched). Reads/writes only the disjoint record
/// subtree via the thread-safe `store`, so groups run concurrently.
fn process_group(
    store: &FlatFile,
    cfg: &Config,
    ptr: DiskPtr,
    keys: &[(Key, Vec<u8>)],
) -> Result<GroupOut> {
    let t = std::time::Instant::now();
    let subtree = store.read_lazy(ptr)?;
    let read_ns = t.elapsed().as_nanos() as u64;
    fold_group(store, cfg, ptr, subtree, keys, read_ns)
}

/// Apply a group's keys to an already-read record subtree, free the old record,
/// and return the outcome (the leaf payload to write, or a promotion). Shared by
/// the per-leaf read path (`process_group`) and the coalesced span path
/// (`process_chunk_coalesced`); `read_ns` is the read time to attribute (already
/// amortized across the span on the coalesced path).
fn fold_group(
    store: &FlatFile,
    cfg: &Config,
    ptr: DiskPtr,
    mut subtree: DiskSubtree,
    keys: &[(Key, Vec<u8>)],
    read_ns: u64,
) -> Result<GroupOut> {
    let depth = subtree.prefix.len();
    let t = std::time::Instant::now();
    for (key, value) in keys {
        record_node_insert(store, cfg, &mut subtree.node, depth, *key, LeafOp::Value(value.clone()))?;
    }
    let rebuild_ns = t.elapsed().as_nanos() as u64;

    let t = std::time::Instant::now();
    let out = if should_promote(cfg, &subtree) {
        store.free(ptr);
        GroupOut::Promoted(promote_record_to_ram(store, subtree)?)
    } else {
        let st = std::time::Instant::now();
        let (payload, _) = serialize_subtree(&subtree)?;
        stats::on_serialize(st.elapsed().as_nanos() as u64);
        store.free(ptr);
        GroupOut::Leaf {
            payload,
            root: hash_node(&subtree.node).finalize(),
        }
    };
    stats::on_group(read_ns, rebuild_ns, t.elapsed().as_nanos() as u64);
    Ok(out)
}

/// Bulk-fold a chunk of groups **pre-sorted by `ptr.unit`** (file offset). Walks
/// the chunk coalescing consecutive leaves into spans of ≤ `FOLD_SPAN_BYTES`
/// (breaking at gaps > `FOLD_MAX_GAP`), reads each span in one `pread`, then folds
/// every leaf in it — parsing each record zero-copy from the shared span buffer
/// (`Arc` clone, no per-record copy). Updated leaves are written back via the same
/// batched-append path as `process_chunk`. Reads each touched leaf once, in file
/// order ⇒ sequential I/O; memory is bounded to one span + one pending write run.
fn process_chunk_coalesced(
    store: &FlatFile,
    cfg: &Config,
    chunk: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
    batched: bool,
) -> Result<Vec<(Key, RamChild)>> {
    let mut out = Vec::with_capacity(chunk.len());
    let mut pending: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
    let rec_end = |p: &DiskPtr| p.offset() + RECORD_HDR as u64 + p.len as u64;
    let mut i = 0;
    while i < chunk.len() {
        // Grow a read span over consecutive leaves while they stay close enough
        // (gap ≤ FOLD_MAX_GAP) and the span stays ≤ FOLD_SPAN_BYTES.
        let start = chunk[i].0.offset();
        let mut j = i;
        while j + 1 < chunk.len() {
            let gap = chunk[j + 1].0.offset().saturating_sub(rec_end(&chunk[j].0));
            if gap > fold_max_gap() || rec_end(&chunk[j + 1].0) - start > FOLD_SPAN_BYTES {
                break;
            }
            j += 1;
        }
        let span_len = (rec_end(&chunk[j].0) - start) as usize;
        let t = std::time::Instant::now();
        let span: Arc<[u8]> = Arc::from(store.read_payload(start, span_len)?);
        stats::on_read_io(t.elapsed().as_nanos() as u64);

        for (ptr, rep, keys) in &chunk[i..=j] {
            // Payload sits just past this record's length prefix within the span.
            let rel = (ptr.offset() - start) as usize + RECORD_HDR as usize;
            let subtree = deserialize_subtree_lazy_at(span.clone(), rel, ptr.len as usize)?;
            match fold_group(store, cfg, *ptr, subtree, keys, 0)? {
                GroupOut::Promoted(rc) => out.push((*rep, rc)),
                GroupOut::Leaf { payload, root } if batched => {
                    pending.push((*rep, root, payload));
                    if pending.len() >= FOLD_WRITE_LEAVES {
                        flush_leaf_batch(store, &mut pending, &mut out)?;
                    }
                }
                GroupOut::Leaf { payload, root } => {
                    let new_ptr = store.write_payload(&payload)?;
                    out.push((*rep, RamChild::Disk { ptr: new_ptr, root }));
                }
            }
        }
        i = j + 1;
    }
    flush_leaf_batch(store, &mut pending, &mut out)?;
    Ok(out)
}

/// Reader stage for the one-writer path: same coalesced span reads + fold as
/// `process_chunk_coalesced`, but instead of writing each new leaf it returns its
/// `(rep, root, payload)` for a single downstream writer to append sequentially —
/// concurrent appends from many workers contend and run slower than one stream.
/// (`fold_group` still frees the old record — cheap, uncontended.) Promotions are
/// written in-stage (rare) and returned as finished `RamChild`s.
#[allow(clippy::type_complexity)]
fn process_chunk_fold(
    store: &FlatFile,
    cfg: &Config,
    chunk: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
) -> Result<(Vec<(Key, Hash, Vec<u8>)>, Vec<(Key, RamChild)>)> {
    let mut leaves: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
    let mut promoted: Vec<(Key, RamChild)> = Vec::new();
    let rec_end = |p: &DiskPtr| p.offset() + RECORD_HDR as u64 + p.len as u64;
    let mut i = 0;
    while i < chunk.len() {
        let start = chunk[i].0.offset();
        let mut j = i;
        while j + 1 < chunk.len() {
            let gap = chunk[j + 1].0.offset().saturating_sub(rec_end(&chunk[j].0));
            if gap > fold_max_gap() || rec_end(&chunk[j + 1].0) - start > FOLD_SPAN_BYTES {
                break;
            }
            j += 1;
        }
        let span_len = (rec_end(&chunk[j].0) - start) as usize;
        let t = std::time::Instant::now();
        let span: Arc<[u8]> = Arc::from(store.read_payload(start, span_len)?);
        stats::on_read_io(t.elapsed().as_nanos() as u64);
        for (ptr, rep, keys) in &chunk[i..=j] {
            let rel = (ptr.offset() - start) as usize + RECORD_HDR as usize;
            let subtree = deserialize_subtree_lazy_at(span.clone(), rel, ptr.len as usize)?;
            match fold_group(store, cfg, *ptr, subtree, keys, 0)? {
                GroupOut::Promoted(rc) => promoted.push((*rep, rc)),
                GroupOut::Leaf { payload, root } => leaves.push((*rep, root, payload)),
            }
        }
        i = j + 1;
    }
    Ok((leaves, promoted))
}

/// Process a chunk of groups, returning the replacement `(rep_key, RamChild)` for
/// each. With `batched`, leaf writes are coalesced into contiguous appended
/// batches of `BATCH_LEAVES` (one `pwrite` per batch); otherwise each is written
/// in place (reusing freed space).
fn process_chunk(
    store: &FlatFile,
    cfg: &Config,
    chunk: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
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
                Some(RamChild::Disk { .. }) | Some(RamChild::Mem(_)) => {
                    children[idx] = Some(new);
                    true
                }
                Some(RamChild::Ram(child)) => install_at_key(child, key, depth + 1, new),
                // Promoted accounts aren't relocated as whole records (their storage
                // records are pinned from evacuation), so none is installed here.
                Some(RamChild::Account(_)) => false,
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
        Node::Leaf { path, value, nref } => {
            1 + (1 + path.len().div_ceil(2)) + 4 + value.len() + ref_size(nref)
        }
        Node::Extension { path, child, nref } => {
            1 + (1 + path.len().div_ceil(2)) + ref_size(nref) + node_size(child)
        }
        Node::Branch { children, nref } => {
            // tag + bitmap + ref + child-length table (u32 per present child) + children.
            let n = children.iter().flatten().count();
            1 + 2 + ref_size(nref) + n * 4 + children.iter().flatten().map(|c| node_size(c)).sum::<usize>()
        }
        // tag + path + nonce(8) + balance(32) + code_hash(32) + storage subtree.
        // The storage root + node ref are recomputed on read, not stored.
        Node::Account { path, storage, .. } => {
            1 + (1 + path.len().div_ceil(2)) + 8 + 32 + 32 + node_size(storage)
        }
        Node::Overflow { .. } => 1 + 4 + 4 + 32,
        // Raw is already serialized — its byte length is its on-disk size.
        Node::Raw { len, .. } => *len,
    }
}

/// On-disk size of a serialized node reference (matches [`write_ref`]).
fn ref_size(nref: &NodeRef) -> usize {
    1 + match nref {
        NodeRef::Hash(_) => 32,
        NodeRef::Inline(rlp) => 1 + rlp.len(),
    }
}

/// Serialize a node reference: tag `0` + 32-byte hash, or tag `1` + `u8` length +
/// the inline RLP (`< 32` bytes).
fn write_ref(out: &mut Vec<u8>, nref: &NodeRef) {
    match nref {
        NodeRef::Hash(h) => {
            out.push(0);
            out.extend_from_slice(h);
        }
        NodeRef::Inline(rlp) => {
            out.push(1);
            out.push(rlp.len() as u8);
            out.extend_from_slice(rlp);
        }
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
    // Pre-size exactly: `record_size` is an allocation-free walk (O(1) per `Raw`
    // child) equal to the final length, so the payload never reallocates as
    // `write_node` fills it — replacing ~log2(size) grow-and-copy doublings with
    // one exact allocation.
    let total = record_size(subtree.prefix.len(), &subtree.node);
    let mut payload = Vec::with_capacity(total);
    write_nibble_path(&mut payload, &subtree.prefix)?;
    write_node(&mut payload, &subtree.node)?;
    debug_assert_eq!(payload.len(), total, "record_size must equal serialized length");
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
        Node::Leaf { path, value, nref } => {
            out.push(1);
            write_nibble_path(out, path)?;
            out.extend_from_slice(&(value.len() as u32).to_le_bytes());
            out.extend_from_slice(value);
            write_ref(out, nref);
        }
        Node::Extension { path, child, nref } => {
            out.push(2);
            write_nibble_path(out, path)?;
            write_ref(out, nref);
            write_node(out, child)?;
        }
        Node::Branch {
            children,
            nref,
        } => {
            out.push(3);
            let mut bitmap = 0u16;
            for (idx, child) in children.iter().enumerate() {
                if child.is_some() {
                    bitmap |= 1 << idx;
                }
            }
            out.extend_from_slice(&bitmap.to_le_bytes());
            write_ref(out, nref);
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
            out.extend_from_slice(&ptr.unit.to_le_bytes());
            out.extend_from_slice(&ptr.len.to_le_bytes());
            out.extend_from_slice(root);
        }
        Node::Account { path, nonce, balance, code_hash, storage, .. } => {
            out.push(5);
            write_nibble_path(out, path)?;
            out.extend_from_slice(&nonce.to_le_bytes());
            out.extend_from_slice(&balance.to_be_bytes::<32>());
            out.extend_from_slice(code_hash);
            write_node(out, storage)?;
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

    /// Advance past a nibble path without materializing it — for header scans
    /// (e.g. [`extract_hash`]) that only need the trailing hash, not the path.
    fn skip_nibble_path(&mut self) -> Result<()> {
        let len = self.read_u8()? as usize;
        if len > 64 {
            bail!("compact subtree nibble path too long");
        }
        self.read_bytes(len.div_ceil(2))?;
        Ok(())
    }

    fn read_ref(&mut self) -> Result<NodeRef> {
        match self.read_u8()? {
            0 => Ok(NodeRef::Hash(self.read_hash()?)),
            1 => {
                let len = self.read_u8()? as usize;
                Ok(NodeRef::Inline(self.read_bytes(len)?.to_vec()))
            }
            t => bail!("invalid node-ref tag {t}"),
        }
    }

    fn read_node(&mut self) -> Result<Node> {
        match self.read_u8()? {
            0 => Ok(Node::Empty),
            1 => {
                let path = self.read_nibble_path()?;
                let vlen = self.read_u32()? as usize;
                let value = self.read_bytes(vlen)?.to_vec();
                let nref = self.read_ref()?;
                Ok(Node::Leaf { path, value, nref })
            }
            2 => {
                let path = self.read_nibble_path()?;
                let nref = self.read_ref()?;
                let child = Box::new(self.read_node()?);
                Ok(Node::Extension { path, child, nref })
            }
            3 => {
                let bitmap = self.read_u16()?;
                let nref = self.read_ref()?;
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
                Ok(Node::Branch { children, nref })
            }
            4 => {
                let unit = self.read_u32()?;
                let len = self.read_u32()?;
                let root = self.read_hash()?;
                Ok(Node::Overflow {
                    ptr: DiskPtr { unit, len },
                    root,
                })
            }
            5 => {
                let path = self.read_nibble_path()?;
                let nonce = u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap());
                let balance = U256::from_be_slice(self.read_bytes(32)?);
                let mut code_hash = [0u8; 32];
                code_hash.copy_from_slice(self.read_bytes(32)?);
                let storage = self.read_node()?;
                Ok(account_node(path, nonce, balance, code_hash, storage))
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
    fn node_ref(&mut self) -> Result<NodeRef> {
        match self.u8()? {
            0 => Ok(NodeRef::Hash(self.hash()?)),
            1 => {
                let len = self.u8()? as usize;
                Ok(NodeRef::Inline(self.take(len)?.to_vec()))
            }
            t => bail!("invalid node-ref tag {t}"),
        }
    }

    fn node(&mut self) -> Result<Node> {
        match self.u8()? {
            0 => Ok(Node::Empty),
            1 => {
                let path = self.nibble_path()?;
                let vlen = self.u32()? as usize;
                let value = self.take(vlen)?.to_vec();
                let nref = self.node_ref()?;
                Ok(Node::Leaf { path, value, nref })
            }
            2 => {
                let path = self.nibble_path()?;
                let nref = self.node_ref()?;
                let child = Box::new(self.node()?);
                Ok(Node::Extension { path, child, nref })
            }
            3 => {
                let bitmap = self.u16()?;
                let nref = self.node_ref()?;
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
                    // leaf/ext/branch subtrees become zero-copy `Raw` — jump over
                    // them via the table (no scan), reading only the child's header
                    // hash, with no per-child path allocation. A batch usually touches
                    // one child; `record_node_insert` expands that `Raw` on descent
                    // (others stay `Raw` and serialize back verbatim). Only the tiny
                    // terminal tags (empty/overflow) are parsed fully.
                    match self.peek_u8()? {
                        1 | 2 | 3 => {
                            let off = self.pos;
                            let nref = extract_ref(&self.buf[off..off + len])?;
                            self.pos += len;
                            *slot = Some(Box::new(Node::Raw {
                                buf: self.buf.clone(),
                                off,
                                len,
                                nref,
                            }));
                        }
                        _ => *slot = Some(Box::new(self.node()?)),
                    }
                }
                Ok(Node::Branch { children, nref })
            }
            4 => {
                let unit = self.u32()?;
                let len = self.u32()?;
                let root = self.hash()?;
                Ok(Node::Overflow {
                    ptr: DiskPtr { unit, len },
                    root,
                })
            }
            5 => {
                let path = self.nibble_path()?;
                let nonce = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
                let balance = U256::from_be_slice(self.take(32)?);
                let mut code_hash = [0u8; 32];
                code_hash.copy_from_slice(self.take(32)?);
                // Parse the storage subtree lazily too (its branch children stay
                // `Raw`); account_node reads their cached refs to fold the root.
                let storage = self.node()?;
                Ok(account_node(path, nonce, balance, code_hash, storage))
            }
            tag => bail!("invalid compact subtree node tag {tag}"),
        }
    }
}

/// Read a node's reference from the front of its serialized `bytes` — a shallow
/// header parse (tag + path/bitmap), no recursion into the subtree.
fn extract_ref(bytes: &[u8]) -> Result<NodeRef> {
    let mut r = CompactReader::new(bytes);
    match r.read_u8()? {
        0 => Ok(NodeRef::Hash(empty_hash())),
        1 => {
            // Leaf: tag, path, value (u32-len-prefixed), ref.
            r.skip_nibble_path()?;
            let vlen = r.read_u32()? as usize;
            let _ = r.read_bytes(vlen)?;
            r.read_ref()
        }
        2 => {
            r.skip_nibble_path()?;
            r.read_ref()
        }
        3 => {
            let _ = r.read_u16()?;
            r.read_ref()
        }
        4 => {
            let _ = r.read_bytes(4 + 4)?;
            Ok(NodeRef::Hash(r.read_hash()?))
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

/// Parse one record's subtree lazily from within a larger shared buffer — a
/// coalesced span read covering many records. The record payload occupies
/// `buf[off .. off+len]`; `Raw` children reference `buf` (every record in the span
/// shares the one allocation, so no per-record copy). Mirrors
/// [`deserialize_subtree_lazy`] but bounded to the record's sub-range.
fn deserialize_subtree_lazy_at(buf: Arc<[u8]>, off: usize, len: usize) -> Result<DiskSubtree> {
    let end = off
        .checked_add(len)
        .filter(|e| *e <= buf.len())
        .ok_or_else(|| anyhow!("span record out of range"))?;
    let mut r = LazyReader::at(buf, off);
    let prefix = r.nibble_path()?;
    let node = r.node()?;
    if r.pos != end {
        bail!("trailing bytes in span record");
    }
    Ok(DiskSubtree { prefix, node })
}

/// Parse just a record payload's leading nibble-path (its `prefix`) — the path to
/// its slot in the frontier — without parsing the subtree. Used by GC to locate a
/// scanned record's frontier pointer.
fn parse_prefix(payload: &[u8]) -> Result<Vec<u8>> {
    CompactReader::new(payload).read_nibble_path()
}

/// Accumulator for [`FlatMpt::spill_mem`]: a pending chunk of `Mem`-leaf byte
/// buffers to write, plus the `(prefix, disk-ptr)` retargets collected so far.
struct SpillBuf {
    prefixes: Vec<Vec<u8>>,
    payloads: Vec<Arc<[u8]>>,
    installs: Vec<(Vec<u8>, DiskPtr)>,
}

/// Write the pending chunk to the file (one dense `write_batch`) and queue each
/// record's `(prefix, disk-ptr)` for install, then clear the chunk.
fn flush_spill_chunk(store: &FlatFile, buf: &mut SpillBuf) -> Result<()> {
    if buf.payloads.is_empty() {
        return Ok(());
    }
    let refs: Vec<&[u8]> = buf.payloads.iter().map(|p| &p[..]).collect();
    let ptrs = store.write_batch(&refs)?;
    for (prefix, ptr) in buf.prefixes.drain(..).zip(ptrs) {
        buf.installs.push((prefix, ptr));
    }
    buf.payloads.clear();
    Ok(())
}

/// Walk the frontier; for each `Mem` leaf take its bytes and replace the slot with
/// a `Disk` placeholder carrying the correct root (the ptr is filled in after the
/// walk). Flushes every `chunk` records so only one chunk of payloads is resident.
fn spill_walk(node: &mut RamNode, prefix: Vec<u8>, store: &FlatFile, buf: &mut SpillBuf, chunk: usize) -> Result<()> {
    match node {
        RamNode::Empty => Ok(()),
        RamNode::Extension { path, child, .. } => {
            let mut next = prefix;
            next.extend_from_slice(path);
            spill_walk(child, next, store, buf, chunk)
        }
        RamNode::Branch { children, .. } => {
            for (i, slot) in children.iter_mut().enumerate() {
                match slot {
                    Some(RamChild::Ram(child)) => {
                        let mut cp = prefix.clone();
                        cp.push(i as u8);
                        spill_walk(child, cp, store, buf, chunk)?;
                    }
                    Some(RamChild::Mem(_)) => {
                        let Some(RamChild::Mem(m)) = slot.take() else { unreachable!() };
                        let mut cp = prefix.clone();
                        cp.push(i as u8);
                        *slot = Some(RamChild::Disk {
                            ptr: DiskPtr { unit: 0, len: m.bytes.len() as u32 },
                            root: m.root,
                        });
                        buf.prefixes.push(cp);
                        buf.payloads.push(m.bytes);
                        if buf.payloads.len() >= chunk {
                            flush_spill_chunk(store, buf)?;
                        }
                    }
                    Some(RamChild::Disk { .. }) | None => {}
                    // Promoted accounts are created on the disk path, so their storage
                    // records are already `Disk` (nothing to spill). RAM-build mode is
                    // not combined with account promotion.
                    Some(RamChild::Account(_)) => {}
                }
            }
            Ok(())
        }
    }
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
                Some(RamChild::Mem(_)) => false,
                Some(RamChild::Ram(child)) => {
                    install_ptr_by_prefix(child, prefix, depth + 1, new_ptr)
                }
                // A promoted account is not a relocatable state record; its storage
                // records are pinned from GC evacuation, so no reloc targets one here.
                Some(RamChild::Account(_)) => false,
                None => false,
            }
        }
    }
}

/// Evacuate the live records out of `victims` (inline GC). Under option B every
/// disk record is a frontier `RamChild::Disk` leaf (no on-disk overflow), so a
/// record is *live* iff the frontier still points at exactly its location. Scan
/// each victim region's framed records, relocate the live, non-foreground ones
/// **verbatim** (same bytes ⇒ unchanged hash/root) via one coalesced **dense**
/// `write_batch`, and free the old copies (dropping each victim region to 0 live so
/// it's reclaimed). Returns `(prefix, new_ptr)` for the relocated records.
fn evacuate_regions(
    store: &FlatFile,
    upper: &RamNode,
    victims: &[u64],
    fg_units: &std::collections::HashSet<u32>,
) -> Result<Vec<(Vec<u8>, DiskPtr)>> {
    let mut live: Vec<(Vec<u8>, Vec<u8>, DiskPtr)> = Vec::new(); // (prefix, payload, old_ptr)
    for &region in victims {
        let buf = store.read_region(region)?;
        let base_unit = region * REGION_UNITS;
        let mut p = 0usize; // byte offset within the region, always 256 B-aligned
        while p + 4 <= buf.len() {
            let len = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            if len == 0 {
                // Padding to a flush's page boundary; skip to the next page.
                let next = (p / PAGE as usize + 1) * PAGE as usize;
                if next <= p {
                    break;
                }
                p = next;
                continue;
            }
            let rec_units = units_for(RECORD_HDR + len) as usize;
            let end = p + 4 + len as usize;
            if end > buf.len() {
                break; // defensive: records never straddle a region
            }
            let unit = (base_unit + (p / ADDR_UNIT as usize) as u64) as u32;
            if !fg_units.contains(&unit) {
                let payload = &buf[p + 4..end];
                if let Ok(prefix) = parse_prefix(payload) {
                    if find_disk_ptr(upper, &prefix, 0) == Some(DiskPtr { unit, len }) {
                        live.push((prefix, payload.to_vec(), DiskPtr { unit, len }));
                    }
                }
            }
            p += rec_units * ADDR_UNIT as usize;
        }
    }
    if live.is_empty() {
        return Ok(Vec::new());
    }
    let payloads: Vec<&[u8]> = live.iter().map(|(_, pl, _)| pl.as_slice()).collect();
    let new_ptrs = store.write_batch(&payloads)?;
    let mut reloc = Vec::with_capacity(live.len());
    for ((prefix, _, old), new) in live.into_iter().zip(new_ptrs) {
        store.free(old);
        reloc.push((prefix, new));
    }
    Ok(reloc)
}

/// Opportunistic GC fused with the foreground read. For a touched, under-util
/// region this batch needs anyway, read the 128 KiB region ONCE and serve both
/// jobs from that single buffer: fold the batch's keys into the foreground leaves
/// found in it, and relocate the region's other still-live records — instead of
/// reading the 5 KiB foreground leaf and then re-reading the whole region for GC.
/// Returns the foreground replacements and the relocations to install.
fn evac_and_fold_region(
    store: &FlatFile,
    cfg: &Config,
    upper: &RamNode,
    region: u64,
    fg: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
    batched: bool,
) -> Result<(Vec<(Key, RamChild)>, Vec<(Vec<u8>, DiskPtr)>)> {
    let t = std::time::Instant::now();
    let buf = store.read_region(region)?;
    stats::on_read_io(t.elapsed().as_nanos() as u64);
    let base_unit = region * REGION_UNITS;
    let fg_by_unit: std::collections::HashMap<u32, &(DiskPtr, Key, Vec<(Key, Vec<u8>)>)> =
        fg.iter().map(|g| (g.0.unit, g)).collect();

    let mut results: Vec<(Key, RamChild)> = Vec::new();
    let mut leaf_pending: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
    let mut reloc_pending: Vec<(Vec<u8>, Vec<u8>, DiskPtr)> = Vec::new();

    let mut p = 0usize;
    while p + 4 <= buf.len() {
        let len = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
        if len == 0 {
            let next = (p / PAGE as usize + 1) * PAGE as usize;
            if next <= p {
                break;
            }
            p = next;
            continue;
        }
        let rec_units = units_for(RECORD_HDR + len) as usize;
        let end = p + 4 + len as usize;
        if end > buf.len() {
            break; // records never straddle a region
        }
        let unit = (base_unit + (p / ADDR_UNIT as usize) as u64) as u32;
        let ptr = DiskPtr { unit, len };
        if let Some(g) = fg_by_unit.get(&unit) {
            // Foreground leaf: fold this batch's keys. Copy just this record into an
            // Arc so the lazy parse's Raw children can borrow it (5 KiB, not 128).
            let rec: Arc<[u8]> = Arc::from(&buf[p + 4..end]);
            let subtree = deserialize_subtree_lazy_at(rec, 0, len as usize)?;
            match fold_group(store, cfg, ptr, subtree, &g.2, 0)? {
                GroupOut::Promoted(rc) => results.push((g.1, rc)),
                GroupOut::Leaf { payload, root } if batched => {
                    leaf_pending.push((g.1, root, payload));
                }
                GroupOut::Leaf { payload, root } => {
                    let np = store.write_payload(&payload)?;
                    results.push((g.1, RamChild::Disk { ptr: np, root }));
                }
            }
        } else {
            // Other live record in this low-util region: relocate it (skip stale).
            let payload = &buf[p + 4..end];
            if let Ok(prefix) = parse_prefix(payload) {
                if find_disk_ptr(upper, &prefix, 0) == Some(ptr) {
                    reloc_pending.push((prefix, payload.to_vec(), ptr));
                }
            }
        }
        p += rec_units * ADDR_UNIT as usize;
    }
    flush_leaf_batch(store, &mut leaf_pending, &mut results)?;

    let mut reloc = Vec::with_capacity(reloc_pending.len());
    if !reloc_pending.is_empty() {
        let payloads: Vec<&[u8]> = reloc_pending.iter().map(|(_, pl, _)| pl.as_slice()).collect();
        let new_ptrs = store.write_batch(&payloads)?;
        for ((prefix, _, old), new) in reloc_pending.into_iter().zip(new_ptrs) {
            store.free(old);
            reloc.push((prefix, new));
        }
    }
    Ok((results, reloc))
}

/// Phase B for opportunistic GC: candidate (touched, under-util) regions are read
/// once and processed by [`evac_and_fold_region`] (fused insert + evacuation); all
/// other groups take the normal coalesced-read fold. Both run in parallel. Returns
/// the foreground replacements and the GC relocations.
fn process_opportunistic(
    store: &FlatFile,
    upper: &RamNode,
    cfg: &Config,
    groups: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
    batched: bool,
) -> Result<(Vec<(Key, RamChild)>, Vec<(Vec<u8>, DiskPtr)>)> {
    let cand: std::collections::HashSet<u64> = {
        let touched: std::collections::HashSet<u64> =
            groups.iter().map(|(p, _, _)| p.unit as u64 / REGION_UNITS).collect();
        store
            .seg
            .lock()
            .unwrap()
            .select_opportunistic(&touched, gc_opp_util())
            .into_iter()
            .collect()
    };
    let mut by_region: std::collections::HashMap<u64, Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)>> =
        std::collections::HashMap::new();
    let mut normal: Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)> = Vec::new();
    for g in groups {
        let region = g.0.unit as u64 / REGION_UNITS;
        if cand.contains(&region) {
            by_region.entry(region).or_default().push(g.clone());
        } else {
            normal.push(g.clone());
        }
    }
    let mut region_jobs: Vec<(u64, Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)>)> =
        by_region.into_iter().collect();
    region_jobs.sort_unstable_by_key(|(r, _)| *r); // sequential region reads
    let threads = worker_count();

    std::thread::scope(|scope| -> Result<(Vec<(Key, RamChild)>, Vec<(Vec<u8>, DiskPtr)>)> {
        let nchunk = normal.len().div_ceil(threads).max(1);
        let normal_handles: Vec<_> = normal
            .chunks(nchunk)
            .map(|c| scope.spawn(|| process_chunk_coalesced(store, cfg, c, batched)))
            .collect();
        let rchunk = region_jobs.len().div_ceil(threads).max(1);
        let region_handles: Vec<_> = region_jobs
            .chunks(rchunk.max(1))
            .map(|rc| {
                scope.spawn(move || -> Result<(Vec<(Key, RamChild)>, Vec<(Vec<u8>, DiskPtr)>)> {
                    let mut res = Vec::new();
                    let mut rel = Vec::new();
                    for (region, fgs) in rc {
                        let (r, rl) = evac_and_fold_region(store, cfg, upper, *region, fgs, batched)?;
                        res.extend(r);
                        rel.extend(rl);
                    }
                    Ok((res, rel))
                })
            })
            .collect();

        let mut results = Vec::new();
        let mut reloc = Vec::new();
        for h in normal_handles {
            results.extend(h.join().expect("normal fold thread panicked")?);
        }
        for h in region_handles {
            let (r, rl) = h.join().expect("region evac thread panicked")?;
            results.extend(r);
            reloc.extend(rl);
        }
        Ok((results, reloc))
    })
}

/// Like [`evac_and_fold_region`] but **writes nothing** — it reads the region once
/// and returns the payloads to be written: foreground promotions, foreground new
/// leaves `(rep, root, payload)`, and relocations `(prefix, payload, old_ptr)`.
/// Used by the fused one-writer GC path, which funnels all of these into the
/// single sequential writer instead of each region thread writing on its own.
/// `fold_group` still frees the foreground records' old ptrs; the relocations'
/// old ptrs are freed by the caller after the single write commits.
#[allow(clippy::type_complexity)]
fn fold_region_collect(
    store: &FlatFile,
    cfg: &Config,
    upper: &RamNode,
    region: u64,
    fg: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
) -> Result<(
    Vec<(Key, RamChild)>,
    Vec<(Key, Hash, Vec<u8>)>,
    Vec<(Vec<u8>, Vec<u8>, DiskPtr)>,
)> {
    let t = std::time::Instant::now();
    let buf = store.read_region(region)?;
    let read_ns = t.elapsed().as_nanos() as u64;
    stats::on_read_io(read_ns);
    let base_unit = region * REGION_UNITS;
    let fg_by_unit: std::collections::HashMap<u32, &(DiskPtr, Key, Vec<(Key, Vec<u8>)>)> =
        fg.iter().map(|g| (g.0.unit, g)).collect();

    let mut promoted: Vec<(Key, RamChild)> = Vec::new();
    let mut leaves: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
    let mut relocs: Vec<(Vec<u8>, Vec<u8>, DiskPtr)> = Vec::new();
    // Evac instrumentation: live bytes found (true region utilization) and relocated
    // survivor bytes, against the full region read.
    let mut live_bytes: u64 = 0;
    let mut reloc_bytes: u64 = 0;

    let mut p = 0usize;
    while p + 4 <= buf.len() {
        let len = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
        if len == 0 {
            let next = (p / PAGE as usize + 1) * PAGE as usize;
            if next <= p {
                break;
            }
            p = next;
            continue;
        }
        let rec_units = units_for(RECORD_HDR + len) as usize;
        let end = p + 4 + len as usize;
        if end > buf.len() {
            break; // records never straddle a region
        }
        let unit = (base_unit + (p / ADDR_UNIT as usize) as u64) as u32;
        let ptr = DiskPtr { unit, len };
        if let Some(g) = fg_by_unit.get(&unit) {
            live_bytes += len as u64; // foreground record was live
            let rec: Arc<[u8]> = Arc::from(&buf[p + 4..end]);
            let subtree = deserialize_subtree_lazy_at(rec, 0, len as usize)?;
            match fold_group(store, cfg, ptr, subtree, &g.2, 0)? {
                GroupOut::Promoted(rc) => promoted.push((g.1, rc)),
                GroupOut::Leaf { payload, root } => leaves.push((g.1, root, payload)),
            }
        } else {
            // Other live record in this low-util region: relocate it (skip stale).
            let payload = &buf[p + 4..end];
            if let Ok(prefix) = parse_prefix(payload) {
                if find_disk_ptr(upper, &prefix, 0) == Some(ptr) {
                    live_bytes += len as u64;
                    reloc_bytes += len as u64;
                    relocs.push((prefix, payload.to_vec(), ptr));
                }
            }
        }
        p += rec_units * ADDR_UNIT as usize;
    }
    stats::on_evac(1, buf.len() as u64, live_bytes, reloc_bytes, read_ns);
    Ok((promoted, leaves, relocs))
}

/// Read phase for the fused one-writer + opportunistic-GC path. Like
/// [`process_opportunistic`] it splits groups into candidate (touched, under-util)
/// regions and the rest, reading each candidate region exactly once to fold the
/// foreground keys *and* evacuate the region's other live records — but it
/// **collects** all resulting payloads instead of writing them, so the caller can
/// append them in one sequential `write_batch`. Returns
/// `(foreground_leaves, promotions, relocations)`.
#[allow(clippy::type_complexity)]
fn process_fold_gc(
    store: &FlatFile,
    upper: &RamNode,
    cfg: &Config,
    groups: &[(DiskPtr, Key, Vec<(Key, Vec<u8>)>)],
) -> Result<(
    Vec<(Key, Hash, Vec<u8>)>,
    Vec<(Key, RamChild)>,
    Vec<(Vec<u8>, Vec<u8>, DiskPtr)>,
)> {
    let cand: std::collections::HashSet<u64> = {
        let touched: std::collections::HashSet<u64> =
            groups.iter().map(|(p, _, _)| p.unit as u64 / REGION_UNITS).collect();
        store
            .seg
            .lock()
            .unwrap()
            .select_opportunistic(&touched, gc_opp_util())
            .into_iter()
            .collect()
    };
    let mut by_region: std::collections::HashMap<u64, Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)>> =
        std::collections::HashMap::new();
    let mut normal: Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)> = Vec::new();
    for g in groups {
        let region = g.0.unit as u64 / REGION_UNITS;
        if cand.contains(&region) {
            by_region.entry(region).or_default().push(g.clone());
        } else {
            normal.push(g.clone());
        }
    }
    let mut region_jobs: Vec<(u64, Vec<(DiskPtr, Key, Vec<(Key, Vec<u8>)>)>)> =
        by_region.into_iter().collect();
    region_jobs.sort_unstable_by_key(|(r, _)| *r); // sequential region reads
    let threads = worker_count();

    #[allow(clippy::type_complexity)]
    std::thread::scope(
        |scope| -> Result<(
            Vec<(Key, Hash, Vec<u8>)>,
            Vec<(Key, RamChild)>,
            Vec<(Vec<u8>, Vec<u8>, DiskPtr)>,
        )> {
            let nchunk = normal.len().div_ceil(threads).max(1);
            let normal_handles: Vec<_> = normal
                .chunks(nchunk)
                .map(|c| scope.spawn(|| process_chunk_fold(store, cfg, c)))
                .collect();
            let rchunk = region_jobs.len().div_ceil(threads).max(1);
            #[allow(clippy::type_complexity)]
            let region_handles: Vec<_> = region_jobs
                .chunks(rchunk.max(1))
                .map(|rc| {
                    scope.spawn(move || -> Result<(
                        Vec<(Key, RamChild)>,
                        Vec<(Key, Hash, Vec<u8>)>,
                        Vec<(Vec<u8>, Vec<u8>, DiskPtr)>,
                    )> {
                        let mut pr = Vec::new();
                        let mut lv = Vec::new();
                        let mut rl = Vec::new();
                        for (region, fgs) in rc {
                            let (p, l, r) = fold_region_collect(store, cfg, upper, *region, fgs)?;
                            pr.extend(p);
                            lv.extend(l);
                            rl.extend(r);
                        }
                        Ok((pr, lv, rl))
                    })
                })
                .collect();

            let mut leaves: Vec<(Key, Hash, Vec<u8>)> = Vec::new();
            let mut promoted: Vec<(Key, RamChild)> = Vec::new();
            let mut relocs: Vec<(Vec<u8>, Vec<u8>, DiskPtr)> = Vec::new();
            for h in normal_handles {
                let (l, p) = h.join().expect("normal fold thread panicked")?;
                leaves.extend(l);
                promoted.extend(p);
            }
            for h in region_handles {
                let (p, l, r) = h.join().expect("region evac thread panicked")?;
                promoted.extend(p);
                leaves.extend(l);
                relocs.extend(r);
            }
            Ok((leaves, promoted, relocs))
        },
    )
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
                RamChild::Mem(_) | RamChild::Account(_) => None,
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
                RamChild::Mem(_) | RamChild::Account(_) => None,
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
            // Extension hashing is identical on the RAM and disk sides, so a
            // node's hash depends only on its structure, never on which side of
            // the storage boundary it currently lives on — see
            // `root_is_independent_of_leaf_size`. Frontier nodes are always ≥ 32
            // bytes (their children are whole records), so the ref finalizes to a
            // plain hash — inlining only happens inside records.
            let computed = ext_ref(path, &NodeRef::Hash(hash_ram(child))).finalize();
            hash.set(Some(computed));
            computed
        }
        RamNode::Branch { children, hash } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            // Same Ethereum branch encoding as the disk side (`branch_node_ref`), so
            // RAM and disk branches with the same structure hash identically
            // (storage-independent root).
            let bitmap = branch_bitmap(children.iter().map(|c| c.is_some()));
            let computed = branch_ref(
                bitmap,
                children.iter().flatten().map(|child| NodeRef::Hash(ram_child_hash(child))),
            )
            .finalize();
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
            let computed = ext_ref(path, &NodeRef::Hash(hash_ram_parallel(child))).finalize();
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
            let bitmap = branch_bitmap(children.iter().map(|c| c.is_some()));
            let computed =
                branch_ref(bitmap, child_hashes.into_iter().map(NodeRef::Hash)).finalize();
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
        RamChild::Mem(m) => m.root,
        RamChild::Account(a) => {
            ram_account_ref(&a.path, a.nonce, &a.balance, &a.code_hash, &a.storage).finalize()
        }
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
        // A promoted account re-folds cheaply, but its storage subtree may be stale.
        RamChild::Account(a) => match a.storage.as_ref() {
            RamNode::Empty => false,
            RamNode::Extension { hash, .. } | RamNode::Branch { hash, .. } => hash.get().is_none(),
        },
        // A Mem leaf's hash is current (set when its bytes were serialized).
        RamChild::Disk { .. } | RamChild::Mem(_) => false,
    }
}

/// A node's *reference* as embedded in its parent: `keccak256(RLP)` when the node's
/// RLP is ≥ 32 bytes, or the RLP inlined verbatim when it is `< 32` (Ethereum's
/// inline rule). Small storage-slot leaves and their tiny parents are the case that
/// makes `Inline` occur; state-trie nodes are always `Hash`.
#[derive(Clone, Debug, PartialEq, Eq)]
enum NodeRef {
    Hash(Hash),
    Inline(Vec<u8>),
}

impl NodeRef {
    /// The bytes this reference contributes as one item of its parent's RLP list:
    /// `RLP(hash)` (a 33-byte string) when hashed, or the node's own RLP verbatim
    /// when inlined.
    fn embed(&self) -> Vec<u8> {
        match self {
            NodeRef::Hash(h) => crate::eth::rlp_string(h),
            NodeRef::Inline(rlp) => rlp.clone(),
        }
    }
    /// The 32-byte hash this node exposes as a trie root: `keccak256(RLP)` — for an
    /// inlined node that means hashing its RLP (the root is always keccak'd, never
    /// inlined).
    fn finalize(&self) -> Hash {
        match self {
            NodeRef::Hash(h) => *h,
            NodeRef::Inline(rlp) => keccak(rlp),
        }
    }
}

/// Build a node reference from the node's RLP: inline it if `< 32` bytes, else hash.
fn mk_ref(rlp: Vec<u8>) -> NodeRef {
    if rlp.len() < 32 {
        NodeRef::Inline(rlp)
    } else {
        NodeRef::Hash(keccak(&rlp))
    }
}

/// Disk-node reference accessor: returns the cached reference (computed at
/// construction). A `Overflow`/cross-record child is always a large subtree, so its
/// reference is the hashed form.
fn hash_node(node: &Node) -> NodeRef {
    match node {
        Node::Empty => NodeRef::Hash(empty_hash()),
        Node::Leaf { nref, .. }
        | Node::Extension { nref, .. }
        | Node::Branch { nref, .. }
        | Node::Account { nref, .. } => nref.clone(),
        Node::Overflow { root, .. } => NodeRef::Hash(*root),
        Node::Raw { nref, .. } => nref.clone(),
    }
}

/// Ethereum extension node reference: `ref(RLP([hex_prefix(path, leaf=false),
/// embed(child)]))`.
fn ext_ref(path: &[u8], child: &NodeRef) -> NodeRef {
    mk_ref(crate::eth::rlp_list(&[
        crate::eth::rlp_string(&crate::eth::hex_prefix(path, false)),
        child.embed(),
    ]))
}

/// Ethereum leaf node reference: `ref(RLP([hex_prefix(path, leaf=true), value]))`.
/// `path` is the leaf's remaining nibbles at its current position, so this must be
/// recomputed whenever the leaf is re-homed to a different depth (see the split in
/// `record_node_insert`) — which is why the value now lives in the leaf.
fn leaf_ref(path: &[u8], value: &[u8]) -> NodeRef {
    mk_ref(crate::eth::rlp_list(&[
        crate::eth::rlp_string(&crate::eth::hex_prefix(path, true)),
        crate::eth::rlp_string(value),
    ]))
}


fn keccak(bytes: &[u8]) -> Hash {
    let _g = prof::scope(prof::Cat::Keccak);
    let output: Hash = Keccak256::digest(bytes).into();
    prof::record(output);
    output
}

/// Ethereum branch node reference: `ref(RLP([ref_0, …, ref_15, value]))`. `bitmap`
/// marks the present child slots (little-endian) and `child_refs` yields their
/// references in slot order; absent slots encode as the empty string `0x80`. All
/// keys are full 64-nibble paths, so no key terminates at a branch — the 17th
/// (value) slot is always the empty string.
fn branch_ref(bitmap: u16, mut child_refs: impl Iterator<Item = NodeRef>) -> NodeRef {
    let empty = crate::eth::rlp_string(&[]);
    let mut items: Vec<Vec<u8>> = Vec::with_capacity(17);
    for i in 0..16 {
        if bitmap & (1 << i) != 0 {
            let ch = child_refs.next().expect("bitmap/child-ref count mismatch");
            items.push(ch.embed());
        } else {
            items.push(empty.clone());
        }
    }
    items.push(empty); // 17th value slot: always empty (secure trie, full-length keys)
    mk_ref(crate::eth::rlp_list(&items))
}

/// Hash of the empty trie: `keccak256(RLP("")) = keccak256(0x80)` — Ethereum's
/// `EMPTY_ROOT`. Computed once.
fn empty_hash() -> Hash {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Hash> = OnceLock::new();
    *EMPTY.get_or_init(|| keccak(&crate::eth::rlp_string(&[])))
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
                    RamChild::Mem(m) => {
                        stats.count += 1;
                        let b = m.bytes.len() as u64;
                        stats.total_bytes += b;
                        let pages = (b.div_ceil(PAGE).max(1)).min(8) as usize;
                        stats.page_hist[pages] += 1;
                    }
                    RamChild::Ram(n) => collect_leaf_stats(n, stats),
                    RamChild::Account(a) => collect_leaf_stats(&a.storage, stats),
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
                RamChild::Mem(_) => 1,
                RamChild::Ram(n) => count_disk_leaves(n),
                RamChild::Account(a) => count_disk_leaves(&a.storage),
            })
            .sum(),
    }
}

/// Bucket every live record's units into per-region live counts (rebuilds
/// [`RegionAlloc`] liveness on reopen). Under option B every disk record is a
/// frontier `RamChild::Disk` leaf — there is no on-disk overflow — so the frontier
/// is the complete liveness map and this is a pure RAM walk (no record reads).
fn recompute_live(node: &RamNode, live: &mut [u32]) {
    match node {
        RamNode::Empty => {}
        RamNode::Extension { child, .. } => recompute_live(child, live),
        RamNode::Branch { children, .. } => {
            for c in children.iter().flatten() {
                match c {
                    RamChild::Ram(n) => recompute_live(n, live),
                    // Mem leaves occupy no disk units (spilled before any reopen).
                    RamChild::Mem(_) => {}
                    RamChild::Disk { ptr, .. } => {
                        let r = RegionAlloc::region_of_unit(ptr.unit as u64) as usize;
                        if r < live.len() {
                            live[r] += ptr.units();
                        }
                    }
                    // A promoted account's storage records are its own Disk leaves.
                    RamChild::Account(a) => recompute_live(&a.storage, live),
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

    /// Guard the frontier's per-node footprint. `RamChild` lives 16-per-branch in
    /// `RamNode`, so any growth multiplies across the whole in-RAM index (an unboxed
    /// ~104 B `Account` variant once nearly doubled it). `RamChild` is 56 B = its
    /// largest inline variant `Mem` (Arc<[u8]> 16 + root Hash 32 = 48) + an 8 B
    /// discriminant; the 32-byte leaf root is what lets the frontier re-hash without
    /// reading records. Rare variants (`Ram`, `Account`) are boxed to a pointer.
    #[test]
    fn frontier_node_stays_compact() {
        use std::mem::size_of;
        assert!(size_of::<RamChild>() <= 56, "RamChild grew to {}", size_of::<RamChild>());
        assert!(size_of::<RamNode>() <= 936, "RamNode grew to {}", size_of::<RamNode>());
        // Box the rare variants so they never set the width.
        assert_eq!(size_of::<Box<PromotedAccount>>(), 8);
    }

    /// The engine must reproduce Ethereum's exact MPT roots: the empty root, the
    /// official secure-trie account vector's known root, and — over an arbitrary
    /// key/value set — the `eth` reference oracle. All values here are ≥ 32 bytes,
    /// so every node's RLP is ≥ 32 bytes and the (not-yet-implemented) inline-`<32`
    /// rule never applies; agreement with the oracle (which does implement inline)
    /// then also proves no inline node arises in these tries.
    #[test]
    fn engine_root_matches_ethereum() {
        let dec = |s: &str| alloy_primitives::hex::decode(s).unwrap();

        // (a) Empty trie == Ethereum EMPTY_ROOT = keccak256(RLP("")).
        let mut mpt = db(Config::default());
        assert_eq!(
            hex(mpt.root()),
            "56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"
        );

        // (b) Official ethereum/tests hex_encoded_securetrie_test "test1": insert
        // keccak(address) -> RLP(account) and reproduce the known state root.
        let pairs = [
            ("a94f5374fce5edbc8e2a8697c15331677e6ebf0b", "f848018405f446a7a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ("095e7baea6a6c7c4c2dfeb977efac326af552d87", "f8440101a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a004bccc5d94f4d1f99aab44369a910179931772f2a5c001c3229f57831c102769"),
            ("d2571607e241ecf590ed94b12d87c94babe36db6", "f8440180a0ba4b47865c55a341a4a78759bb913cd15c3ee8eaf30a62fa8d1c8863113d84e8a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ("62c01474f089b07dae603491675dc5b5748f7049", "f8448080a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ("2adc25665018aa1fe0e6bc666dac8fc2697ff9ba", "f8478083019a59a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        ];
        let mut oracle_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for (addr, rlp) in pairs {
            let (addr, rlp) = (dec(addr), dec(rlp));
            mpt.insert(hashed_key(&addr), rlp.clone()).unwrap();
            oracle_entries.push((addr, rlp));
        }
        assert_eq!(
            hex(mpt.root()),
            "730a444e08ab4b8dee147c9b232fc52d34a223d600031c1e9d25bfc985cbd797"
        );
        assert_eq!(mpt.root().as_slice(), crate::eth::secure_root(&oracle_entries).as_slice());

        // (c) General equivalence over many keys via the batch path: engine root ==
        // eth oracle over the same (32-byte key used verbatim, value) set.
        let mut mpt2 = db(Config::default());
        let mut entries: Vec<(Key, Vec<u8>)> = Vec::new();
        for i in 0..3000u64 {
            let k = hashed_key(i.to_le_bytes());
            let mut v = vec![0xabu8; 40];
            v[..8].copy_from_slice(&i.to_le_bytes());
            entries.push((k, v));
        }
        mpt2.insert_batch(entries.clone()).unwrap();
        let oracle: Vec<(Vec<u8>, Vec<u8>)> =
            entries.iter().map(|(k, v)| (k.to_vec(), v.clone())).collect();
        assert_eq!(mpt2.root().as_slice(), crate::eth::root(&oracle).as_slice());
    }

    /// Storage-trie shape: 32-byte keys with *tiny* values, so deep leaves (and the
    /// branches above them) have RLP < 32 bytes and must be **inlined** into their
    /// parent per Ethereum's rule. Matching the oracle (which inlines) proves the
    /// engine's inline-`<32` node references are correct — the Phase-4a prerequisite
    /// for exact storage roots. Tiny leaves force many records so inline nodes span
    /// the disk path, not just one in-RAM record.
    #[test]
    fn engine_matches_oracle_with_inline_nodes() {
        let cfg = Config { target_leaf_bytes: 512, max_leaf_bytes: 1024, min_promote_bytes: 256 };
        // 1-3 byte values (like RLP(U256) storage slots) => sub-32-byte leaf/branch RLP.
        let entries: Vec<(Key, Vec<u8>)> = (0..8000u64)
            .map(|i| {
                let k = hashed_key(i.to_le_bytes());
                let v = vec![(i as u8) | 1; 1 + (i % 3) as usize]; // non-empty, 1..3 bytes
                (k, v)
            })
            .collect();
        let oracle: Vec<(Vec<u8>, Vec<u8>)> =
            entries.iter().map(|(k, v)| (k.to_vec(), v.clone())).collect();
        let want = crate::eth::root(&oracle);

        // Disk build (batched) and one-by-one must both match the inline-aware oracle.
        let mut batched = db(cfg.clone());
        for chunk in entries.chunks(97) {
            batched.insert_batch(chunk.to_vec()).unwrap();
        }
        assert_eq!(batched.root().as_slice(), want.as_slice(), "batched root vs oracle");

        let mut one = db(cfg);
        for (k, v) in &entries {
            one.insert(*k, v.clone()).unwrap();
        }
        assert_eq!(one.root().as_slice(), want.as_slice(), "one-by-one root vs oracle");

        // Values still retrievable through the inline nodes.
        for (k, v) in entries.iter().take(200) {
            assert_eq!(batched.get_value(k).unwrap().as_deref(), Some(v.as_slice()));
        }
    }

    #[test]
    fn ram_build_matches_disk_build() {
        // A RAM build (leaves held as their own Arcs, reached through the parallel
        // top-nibble fan-out) must yield the byte-identical Merkle root of a disk
        // build — before spilling, after spilling, and after persist+reopen. Tiny
        // leaves force heavy promotion so the path is well exercised.
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 1024,
            min_promote_bytes: 256,
        };
        const N: u64 = 20_000;
        let key = |i: u64| hashed_key(i.to_le_bytes());

        let mut disk = db(cfg.clone());
        for i in 0..N {
            disk.insert(key(i), vec![7u8; 32]).unwrap();
        }
        let root_disk = disk.root();

        // RAM build, no spill (huge threshold): every leaf stays a Mem(Arc).
        let tmp = NamedTempFile::new().unwrap();
        let mut ram = FlatMpt::create(tmp.path(), cfg.clone()).unwrap();
        ram.ram_mode = true;
        ram.spill_threshold = u64::MAX;
        let mut buf = Vec::new();
        for i in 0..N {
            buf.push((key(i), vec![7u8; 32]));
            if buf.len() == 1000 {
                ram.insert_batch(std::mem::take(&mut buf)).unwrap();
            }
        }
        ram.insert_batch(buf).unwrap();
        assert!(ram.ram_mode, "should not have spilled at u64::MAX threshold");
        assert_eq!(ram.store.end_page(), 0, "RAM build must not touch the flat file");
        assert_eq!(ram.root(), root_disk, "RAM build root differs from disk build");

        // Spill Mem -> disk; root unchanged, leaves now reachable on disk.
        ram.spill_mem().unwrap();
        assert!(!ram.ram_mode);
        assert_eq!(ram.root(), root_disk, "root changed across spill");
        assert_eq!(
            ram.disk_accesses_for_key(&key(0)).unwrap(),
            1,
            "spilled leaves should be reachable in one disk read",
        );

        // Persist + reopen round-trips to the same root.
        let path = tmp.path().to_path_buf();
        ram.persist().unwrap();
        drop(ram);
        let reopened = FlatMpt::open(&path).unwrap();
        assert_eq!(reopened.root(), root_disk, "reopened RAM-built DB root differs");
    }

    #[test]
    fn direct_io_records_round_trip() {
        // Validate the direct-I/O aligned read/write path (offset/length widening)
        // independent of the MPT_DIRECT_IO env: build a FlatFile with direct=true
        // and round-trip records that sit at 256 B-aligned (not 4096-aligned)
        // offsets with assorted lengths, several straddling 4096 boundaries.
        let tmp = NamedTempFile::new().unwrap();
        let f = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(tmp.path())
            .unwrap();
        let store = FlatFile::new(f, true);
        let payloads: Vec<Vec<u8>> = (0..40usize)
            .map(|i| {
                let n = 50 + i * 211; // spans sub-page to multi-page
                (0..n).map(|j| (i as u8).wrapping_add(j as u8)).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        let ptrs = store.write_batch(&refs).unwrap();
        for (p, ptr) in payloads.iter().zip(&ptrs) {
            let got = store
                .read_payload(ptr.offset() + RECORD_HDR as u64, ptr.len as usize)
                .unwrap();
            assert_eq!(&got, p, "direct aligned read must return the written payload");
        }
        // Single-record (page-padded) write path too.
        let single = vec![0x5au8; 9000];
        let ptr = store.write_payload(&single).unwrap();
        let got = store
            .read_payload(ptr.offset() + RECORD_HDR as u64, ptr.len as usize)
            .unwrap();
        assert_eq!(got, single);
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
    fn bulk_fold_coalesced_matches_one_by_one() {
        // The coalesced span-read fold (taken when a batch touches ≥ 64 leaves)
        // must produce the byte-identical root and values as one-by-one inserts.
        // Tiny leaves + many keys fan out to thousands of disk leaves across many
        // regions; churn (rewrites) leaves freed holes so spans have gaps — so this
        // exercises span growth, gap breaks, and zero-copy parse from a shared
        // buffer. A 4000-key batch touches well over 64 leaves ⇒ the bulk path.
        let cfg = Config {
            target_leaf_bytes: 128,
            max_leaf_bytes: 256,
            min_promote_bytes: 64,
        };
        let pairs: Vec<(Key, Vec<u8>)> = (0..12000u64)
            .map(|i| (hashed_key(i.to_le_bytes()), vec![i as u8; 24]))
            .collect();

        let mut one = db(cfg.clone());
        for (k, v) in &pairs {
            one.insert(*k, v.clone()).unwrap();
        }

        // Two large batches (each ≫ 64 groups) so the coalesced reader dominates,
        // plus an overwrite batch so some leaves are folded a second time.
        let mut bulk = db(cfg.clone());
        bulk.insert_batch(pairs[..6000].to_vec()).unwrap();
        bulk.insert_batch(pairs[6000..].to_vec()).unwrap();
        bulk.insert_batch(pairs[..4000].to_vec()).unwrap(); // re-fold existing leaves

        assert_eq!(one.root(), bulk.root(), "coalesced fold root must match one-by-one");
        assert!(
            count_disk_leaves(&bulk.upper) >= 64,
            "test must exercise the coalesced path (≥64 leaves)"
        );
        for (k, v) in &pairs {
            assert_eq!(bulk.get_value(k).unwrap(), Some(v.clone()));
        }

        // Survives persist + reopen.
        let path = NamedTempFile::new().unwrap().path().to_path_buf();
        let mut p = FlatMpt::create(&path, cfg.clone()).unwrap();
        for chunk in pairs.chunks(4096) {
            p.insert_batch(chunk.to_vec()).unwrap();
        }
        let root = p.root();
        p.persist().unwrap();
        drop(p);
        let reopened = FlatMpt::open(&path).unwrap();
        assert_eq!(reopened.root(), root, "root must survive reopen after a bulk fold");
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
        // Sanity: the two really took different storage paths. Tiny leaves promote
        // heavily (the trie fans out into the RAM frontier), so `a` has many more
        // frontier nodes than the huge-leaf `b`, which keeps everything in a few
        // packed records. Both reach any key in one disk read (no overflow chains).
        assert!(
            a.ram_nodes() > b.ram_nodes(),
            "tiny leaves should promote into a larger frontier: a={} b={}",
            a.ram_nodes(),
            b.ram_nodes(),
        );
        assert_eq!(
            a.disk_accesses_for_key(&hashed_key(0u64.to_le_bytes())).unwrap(),
            1,
            "option B reaches any key in one disk read",
        );
    }

    #[test]
    fn overflow_node_round_trips_and_hashes_as_its_root() {
        // A branch with one inline leaf child and one Overflow child must:
        //  (a) survive serialize -> deserialize unchanged, and
        //  (b) hash identically whether that child is inline or overflowed
        //      (the Overflow.root equals the inline node's hash).
        let _key = hashed_key("x");
        let inline_child = leaf_node(vec![5, 6, 7], vec![9u8; 32]);
        let inline_hash = hash_node(&inline_child).finalize();

        // Build branch B1 with the child inline at slot 3.
        let mut c1 = empty_box_children();
        c1[3] = Some(Box::new(inline_child));
        let branch_inline = make_branch(c1);

        // Build branch B2 with the same child as an Overflow pointer at slot 3.
        let mut c2 = empty_box_children();
        c2[3] = Some(Box::new(Node::Overflow {
            ptr: DiskPtr { unit: 1, len: 200 },
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
                    assert_eq!(*ptr, DiskPtr { unit: 1, len: 200 });
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
    fn overwrite_churn_keeps_values_current() {
        // Heavy overwrite churn stays correct. (File-size bounding under churn is a
        // GC property, covered by the active_gc/overflow_gc tests at a scale past
        // the GC floor; this small store sits below it, where the log-structured
        // allocator simply appends, so file size isn't asserted here.)
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        let keys: Vec<Key> = (0..200u64).map(|i| hashed_key(i.to_le_bytes())).collect();
        for round in 0..20u8 {
            for key in &keys {
                db.insert(*key, vec![round; 32]).unwrap();
            }
        }
        for key in &keys {
            assert_eq!(db.get_value(key).unwrap(), Some(vec![19u8; 32]));
        }
    }

    #[test]
    fn live_accounting_stays_consistent() {
        // Tiny leaves force heavy overflow-splitting/promotion — the regime where
        // the remote 8 KiB run leaked live units (zombie regions). The allocator's
        // live count must match the frontier's truth after every batch.
        let cfg = Config {
            target_leaf_bytes: 384,
            max_leaf_bytes: 512,
            min_promote_bytes: 256,
        };
        let dir = tempfile::tempdir().unwrap();
        let mut db = FlatMpt::create(dir.path().join("db.flat"), cfg).unwrap();
        let n: u64 = 200_000;
        for chunk in (0..n).step_by(5000) {
            let batch: Vec<(Key, Vec<u8>)> = (chunk..(chunk + 5000).min(n))
                .map(|i| (hashed_key(i.to_le_bytes()), vec![0u8; 16]))
                .collect();
            db.insert_batch(batch).unwrap();
            let (alloc, truth) = db.audit_live_units();
            assert_eq!(
                alloc, truth,
                "live accounting diverged after {} keys: allocator={alloc} frontier={truth} (leak={})",
                chunk + 5000,
                alloc as i64 - truth as i64,
            );
        }
    }

    #[test]
    fn overflow_gc_bounds_file() {
        // Small leaves force heavy overflow-splitting — the regime where overflow
        // child records (invisible to the frontier) made GC unable to reclaim their
        // regions, so the file ballooned (the remote hit util ~6%). With
        // overflow-aware GC the file must stay bounded.
        let cfg = Config {
            target_leaf_bytes: 768,
            max_leaf_bytes: 1024,
            min_promote_bytes: 512,
        };
        let dir = tempfile::tempdir().unwrap();
        let mut db = FlatMpt::create(dir.path().join("db.flat"), cfg).unwrap();
        let n: u64 = 3_000_000;
        for chunk in (0..n).step_by(10_000) {
            let batch: Vec<(Key, Vec<u8>)> = (chunk..(chunk + 10_000).min(n))
                .map(|i| (hashed_key(i.to_le_bytes()), vec![0u8; 16]))
                .collect();
            db.insert_batch(batch).unwrap();
        }
        assert!(
            db.flat_file_len() > GC_MIN_PAGES * PAGE,
            "file too small to have engaged GC"
        );
        // Pre-fix this collapsed toward ~6%; overflow-aware GC reclaims the overflow
        // regions, so utilization stays healthy.
        assert!(
            db.utilization() > 0.30,
            "file ballooned: util {:.0}% flat {} live {}",
            db.utilization() * 100.0,
            db.flat_file_len(),
            db.live_bytes(),
        );
        let (alloc, truth) = db.audit_live_units();
        assert_eq!(alloc, truth, "live accounting diverged: {alloc} vs {truth}");
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
    fn promotes_large_leaf_into_frontier() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        for i in 0..3000u64 {
            db.insert(hashed_key(i.to_le_bytes()), vec![i as u8; 32])
                .unwrap();
        }
        // Option B: a record exceeding max_leaf is promoted into the RAM frontier
        // (its children become frontier leaves) rather than shedding children to
        // on-disk overflow. So the frontier grows, and every key is reachable in
        // exactly ONE disk read — there are no overflow chains.
        assert!(
            db.ram_nodes() > 10,
            "tiny leaves should promote into the frontier, got {}",
            db.ram_nodes()
        );
        let max_reads = (0..3000u64)
            .map(|i| db.disk_accesses_for_key(&hashed_key(i.to_le_bytes())).unwrap())
            .max()
            .unwrap();
        assert_eq!(max_reads, 1, "option B has no overflow chains, max reads={max_reads}");
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
