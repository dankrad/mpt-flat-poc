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
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

const SAMPLE: usize = 1000;
const ROUNDS: usize = 20;

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

    let cfg = Config {
        target_leaf_bytes: 4 * 1024,
        max_leaf_bytes: 8 * 1024,
        min_promote_bytes: 2 * 1024,
    };
    let tmp = NamedTempFile::new().unwrap();
    let mut db = FlatMpt::create(tmp.path(), cfg).unwrap();

    // ---- preload ----
    let t = Instant::now();
    for i in 0..preload {
        db.insert(key_at(i), vec![0u8; 32]).unwrap();
    }
    db.flush().unwrap();
    let elapsed = t.elapsed();
    println!(
        "preloaded {preload} keys in {:.2}s  ({:.2} µs/key)",
        elapsed.as_secs_f64(),
        us_per(elapsed, preload as usize),
    );
    println!(
        "  flat file: {:.1} MiB,  ram_nodes: {},  free_regions: {},  free: {:.1} MiB\n",
        db.flat_file_len() as f64 / 1_048_576.0,
        db.ram_nodes(),
        db.free_regions(),
        db.free_bytes() as f64 / 1_048_576.0,
    );

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
