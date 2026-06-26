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
use std::sync::atomic::{AtomicUsize, Ordering};
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

    // Disk-leaf size in KiB (max). target = max/2, min_promote = max/4.
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
        min_promote_bytes: max / 4,
    };
    // Batch size for preload inserts; 0 = one-by-one. Must divide PROGRESS_EVERY.
    let batch: usize = std::env::var("LARGE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    println!(
        "config: max_leaf={} KiB (target {}, min {}), batch={}\n",
        max_kib,
        max / 2,
        max / 4,
        if batch == 0 { "off".into() } else { batch.to_string() },
    );
    let tmp = NamedTempFile::new().unwrap();
    let mut db = FlatMpt::create(tmp.path(), cfg).unwrap();

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
    for i in 0..preload {
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
            println!(
                "  [{:>4}M] {:>6.0}s | {:.2} µs/key | flat {:.1} GiB | leaves {} | avg_leaf {} B | free_reg {} | RAM {:.1} MiB",
                done / 1_000_000,
                t.elapsed().as_secs_f64(),
                chunk.as_micros() as f64 / PROGRESS_EVERY as f64,
                db.flat_file_len() as f64 / (1024.0 * 1024.0 * 1024.0),
                ls.count,
                ls.avg_bytes(),
                db.free_regions(),
                mib(live),
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

    println!(
        "preloaded {preload} keys in {:.2}s  ({:.2} µs/key)",
        elapsed.as_secs_f64(),
        us_per(elapsed, preload as usize),
    );
    println!(
        "  flat file: {:.1} MiB,  free_regions: {},  free: {:.1} MiB",
        db.flat_file_len() as f64 / 1_048_576.0,
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
