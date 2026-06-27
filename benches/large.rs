//! Steady-state benchmark on a large trie: preload N keys, then measure the
//! cost of (a) 1000 brand-new inserts and (b) 1000 overwrites of existing keys.
//!
//!     cargo bench --bench large
//!     cargo bench --bench large --features profiling      # + time attribution
//!     LARGE_PRELOAD=100000 cargo bench --bench large       # smaller preload
//!
//! Uses a custom harness (not criterion) because the 1M-key DB must be built
//! once and then mutated in place — criterion's per-iteration setup can't.

use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key, prof};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

const SAMPLE: usize = 1000;
const ROUNDS: usize = 20;

/// Global allocator that tracks live Rust-heap bytes, so we can read the true
/// in-RAM footprint of the database (RocksDB's own C++ memory is not counted).
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            LIVE.fetch_add(new_size, Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live_heap() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Set by the SIGINT/SIGTERM handler; the preload loop checks it and persists the
/// database (flushing any buffered batch first) before exiting, so a killed run
/// leaves a reopenable checkpoint instead of losing the in-RAM frontier.
/// `kill -9` (SIGKILL) cannot be caught, so it still loses the frontier.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    // Async-signal-safe: only flip a flag; the main thread does the real work.
    INTERRUPTED.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // Cast through a fn pointer (not the fn item) to get a valid sighandler_t.
    let handler = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only performs an atomic store, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / 1_048_576.0
}

fn key_at(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

fn us_per(d: Duration, n: usize) -> f64 {
    d.as_nanos() as f64 / 1000.0 / n as f64
}

fn report(title: &str, rounds: &[Duration]) {
    let mut per_op: Vec<f64> = rounds.iter().map(|d| us_per(*d, SAMPLE)).collect();
    per_op.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_op[per_op.len() / 2];
    let mean = per_op.iter().sum::<f64>() / per_op.len() as f64;

    println!("=== {title} ===");
    println!(
        "  {} rounds x {} ops:  {:.2} µs/op median  ({:.2} mean, {:.2} min, {:.2} max)",
        rounds.len(),
        SAMPLE,
        median,
        mean,
        per_op[0],
        per_op[per_op.len() - 1],
    );

    if prof::ENABLED {
        let total_ops = (rounds.len() * SAMPLE) as f64;
        println!("  per-op breakdown (µs/op, summed over all rounds):");
        for (i, (nanos, count)) in prof::snapshot().iter().enumerate() {
            if *count == 0 {
                continue;
            }
            println!(
                "    {:<28} {:>7.3} µs/op",
                prof::CATS[i],
                *nanos as f64 / 1000.0 / total_ops,
            );
        }
    }
    println!();
}

fn main() {
    let preload: u64 = std::env::var("LARGE_PRELOAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    // Disk-leaf size in KiB (max). target = max/2, min_promote = max/2.
    // Larger leaves => fewer leaves => smaller frontier and less page padding,
    // at the cost of more bytes rewritten per insert.
    let max_kib: usize = std::env::var("LARGE_MAX_LEAF_KIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let max = max_kib * 1024;
    let cfg = Config {
        target_leaf_bytes: max / 2,
        max_leaf_bytes: max,
        min_promote_bytes: max / 2,
    };
    // Batch size for preload inserts; 0 = one-by-one. Must divide PROGRESS_EVERY.
    let batch: usize = std::env::var("LARGE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    println!(
        "config: max_leaf={} KiB (target {}, min_promote {}), batch={}\n",
        max_kib,
        max / 2,
        max / 2,
        if batch == 0 { "off".into() } else { batch.to_string() },
    );
    // With LARGE_PERSIST=1 the DB is built at a fixed, non-temporary path under
    // $TMPDIR (so the flat file + .values + .meta survive process exit and can be
    // reopened with FlatMpt::open). Otherwise a NamedTempFile auto-cleans on exit.
    let persist = std::env::var("LARGE_PERSIST").ok().as_deref() == Some("1");
    let tmp = NamedTempFile::new().unwrap();
    let db_path = if persist {
        std::env::temp_dir().join("mpt-checkpoint.flat")
    } else {
        tmp.path().to_path_buf()
    };
    let mut db = FlatMpt::create(&db_path, cfg).unwrap();
    install_signal_handlers();

    // ---- preload ----
    // Log running stats every PROGRESS_EVERY keys so a long (1B-key) run can be
    // tracked live: per-chunk insert rate, on-disk size, fragmentation, and the
    // in-RAM index footprint.
    const PROGRESS_EVERY: u64 = 10_000_000;
    mpt_flat_poc::stats::reset();
    let heap_before = live_heap();
    let t = Instant::now();
    let mut chunk_start = Instant::now();
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::new();
    // Previous-milestone split-leaf totals, to report the per-interval average
    // size of leaves freshly created by splitting.
    use std::sync::atomic::Ordering::Relaxed;
    use mpt_flat_poc::stats;
    // Per-interval deltas: read each counter, subtract the previous milestone's
    // value. Indices documented in `snap()`.
    let snap = || -> [u64; 14] {
        [
            stats::PHASE_A_NS.load(Relaxed),      // 0
            stats::PHASE_B_NS.load(Relaxed),      // 1
            stats::PHASE_C_NS.load(Relaxed),      // 2
            stats::BATCHES.load(Relaxed),         // 3
            stats::B_READ_IO_NS.load(Relaxed),    // 4  pread (device)
            stats::B_READ_PARSE_NS.load(Relaxed), // 5  parse (cpu)
            stats::B_REBUILD_NS.load(Relaxed),    // 6  rebuild (cpu)
            stats::B_SERIALIZE_NS.load(Relaxed),  // 7  serialize (cpu)
            stats::W_PWRITE_NS.load(Relaxed),     // 8  pwrite (device)
            stats::W_LOCK_NS.load(Relaxed),       // 9  alloc lock (contention)
            stats::GC_NS.load(Relaxed),           // 10 GC evacuate (read+relocate)
            stats::GC_RELOCATED.load(Relaxed),    // 11 records relocated
            stats::GC_REGIONS.load(Relaxed),      // 12 regions reclaimed
            stats::B_FINAL_NS.load(Relaxed),      // 13 migrate+serialize+promote
        ]
    };
    let mut prev = snap();
    for i in 0..preload {
        // Caught a SIGINT/SIGTERM: flush any buffered batch, persist a reopenable
        // checkpoint, and exit. (`std::process::exit` skips the NamedTempFile
        // drop, so even the temp-path flat file survives.)
        if INTERRUPTED.load(Relaxed) {
            eprintln!("\n[signal] persisting {i} keys before exit...");
            if !buf.is_empty() {
                db.insert_batch(std::mem::take(&mut buf)).unwrap();
            }
            db.flush().unwrap();
            db.persist().unwrap();
            println!(
                "persisted {i} keys to {} (reopen with FlatMpt::open)",
                db_path.display(),
            );
            std::io::stdout().flush().ok();
            std::process::exit(0);
        }
        if batch == 0 {
            db.insert(key_at(i), vec![0u8; 32]).unwrap();
        } else {
            buf.push((key_at(i), vec![0u8; 32]));
            if buf.len() >= batch {
                db.insert_batch(std::mem::take(&mut buf)).unwrap();
            }
        }
        let done = i + 1;
        if done % PROGRESS_EVERY == 0 && done < preload {
            let chunk = chunk_start.elapsed();
            let live = live_heap().saturating_sub(heap_before);
            let ls = db.leaf_stats();
            let cur = snap();
            let d: [u64; 14] = std::array::from_fn(|k| cur[k].saturating_sub(prev[k]));
            prev = cur;
            let dnb = d[3].max(1); // batches this interval
            let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
            let msb = |ns: u64| ns as f64 / 1e6 / dnb as f64; // ms/batch
            let uk = |ns: u64| ns as f64 / 1000.0 / PROGRESS_EVERY as f64; // thread-µs/key
            // Phase wall split now includes GC (which runs between B and C).
            let phase_tot = (d[0] + d[1] + d[10] + d[2]).max(1);
            println!(
                "  [{:>4}M] {:>6.0}s {:.1}µs/key | flat {:.1}G live {:.1}G util {:.0}% | leaves {} avg {}B ({:.1} k/leaf) | RAM {:.0}M / {} nodes",
                done / 1_000_000,
                t.elapsed().as_secs_f64(),
                chunk.as_micros() as f64 / PROGRESS_EVERY as f64,
                gib(db.flat_file_len()),
                gib(db.live_bytes()),
                db.utilization() * 100.0,
                ls.count,
                ls.avg_bytes(),
                done as f64 / ls.count.max(1) as f64,
                mib(live),
                db.ram_report().frontier_nodes,
            );
            println!(
                "          phase ms/batch: A {:.1} | B {:.1} | GC {:.1} | C {:.1}  (B {:.0}%, GC {:.0}%)",
                msb(d[0]), msb(d[1]), msb(d[10]), msb(d[2]),
                d[1] as f64 / phase_tot as f64 * 100.0,
                d[10] as f64 / phase_tot as f64 * 100.0,
            );
            println!(
                "          B-work µs/key: pread {:.1} parse {:.1} rebuild {:.1} serialize {:.1} pwrite {:.1} lock {:.1}",
                uk(d[4]), uk(d[5]), uk(d[6]), uk(d[7]), uk(d[8]), uk(d[9]),
            );
            println!(
                "          GC: {:.2} reloc/key, {} regs/batch, {:.1} ms/batch, R={}, free_regions {}",
                d[11] as f64 / PROGRESS_EVERY as f64,
                d[12] / dnb,
                msb(d[10]),
                db.gc_rate_current(),
                db.free_regions(),
            );
            std::io::stdout().flush().ok();
            chunk_start = Instant::now();
        }
    }
    if !buf.is_empty() {
        db.insert_batch(buf).unwrap();
    }
    db.flush().unwrap();
    let elapsed = t.elapsed();
    let heap_after = live_heap();

    // Optional checkpoint: LARGE_PERSIST=1 writes the .meta manifest so the built
    // database can be reopened with FlatMpt::open (e.g. to time phases at scale
    // without rebuilding). The flat file persists in $TMPDIR — don't delete it.
    if persist {
        let pt = Instant::now();
        db.persist().unwrap();
        println!(
            "  persisted checkpoint to {} in {:.1}s (reopen with FlatMpt::open)",
            db_path.display(),
            pt.elapsed().as_secs_f64(),
        );
    }

    println!(
        "preloaded {preload} keys in {:.2}s  ({:.2} µs/key)",
        elapsed.as_secs_f64(),
        us_per(elapsed, preload as usize),
    );
    println!(
        "  flat file: {:.1} MiB,  live: {:.1} MiB,  util: {:.0}%,  free_regions: {},  garbage: {:.1} MiB",
        db.flat_file_len() as f64 / 1_048_576.0,
        db.live_bytes() as f64 / 1_048_576.0,
        db.utilization() * 100.0,
        db.free_regions(),
        db.free_bytes() as f64 / 1_048_576.0,
    );
    let ls = db.leaf_stats();
    println!(
        "  leaves: {} (avg {} B = {:.1} keys/leaf), live {:.1} MiB vs flat {:.1} MiB",
        ls.count,
        ls.avg_bytes(),
        preload as f64 / ls.count.max(1) as f64,
        ls.total_bytes as f64 / 1_048_576.0,
        db.flat_file_len() as f64 / 1_048_576.0,
    );
    println!(
        "  leaf pages: {}",
        (1..=8)
            .map(|p| format!("{p}p={}", ls.page_hist[p]))
            .collect::<Vec<_>>()
            .join(" "),
    );

    // In-RAM index footprint: ground truth (live heap delta) + structural breakdown.
    let r = db.ram_report();
    let live = heap_after.saturating_sub(heap_before);
    println!(
        "  in-RAM index (live heap): {:.1} MiB total  ({:.1} B/key)",
        mib(live),
        live as f64 / preload as f64,
    );
    println!(
        "    frontier: {:.1} MiB ({} nodes),  free list: ~{:.1} MiB ({} regions),  overlay: {:.1} MiB",
        mib(r.frontier_bytes),
        r.frontier_nodes,
        mib(r.free_list_bytes),
        r.free_regions,
        mib(r.overlay_bytes),
    );
    println!("  split/write stats: {}", mpt_flat_poc::stats::dump());
    println!();

    // When building a checkpoint, stop here: phases 1/2 below would mutate the
    // flat file after persist() and leave the on-disk manifest stale.
    if persist {
        return;
    }

    // ---- phase 1: brand-new inserts ----
    let mut next_new = preload;
    let mut rounds = Vec::new();
    prof::reset();
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for _ in 0..SAMPLE {
            db.insert(key_at(next_new), vec![1u8; 32]).unwrap();
            next_new += 1;
        }
        db.flush().unwrap();
        rounds.push(t.elapsed());
    }
    report("1000 NEW inserts into the preloaded trie", &rounds);
    println!(
        "  (after phase 1: free_regions: {}, free: {:.1} MiB)\n",
        db.free_regions(),
        db.free_bytes() as f64 / 1_048_576.0,
    );

    // ---- phase 2: overwrite existing keys ----
    let mut rng = StdRng::seed_from_u64(99);
    let mut rounds = Vec::new();
    prof::reset();
    for r in 0..ROUNDS {
        // Pick the target keys outside the timed region.
        let targets: Vec<u64> = (0..SAMPLE).map(|_| rng.next_u64() % preload).collect();
        let value = vec![(r as u8).wrapping_add(1); 32];
        let t = Instant::now();
        for &i in &targets {
            db.insert(key_at(i), value.clone()).unwrap();
        }
        db.flush().unwrap();
        rounds.push(t.elapsed());
    }
    report("1000 OVERWRITES of random existing keys", &rounds);
}
