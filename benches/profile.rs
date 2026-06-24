//! Wall-clock attribution benchmark: where does insert time actually go?
//!
//! Run with the `profiling` feature for the breakdown:
//!
//!     cargo bench --bench profile --features profiling
//!
//! Without the feature it still reports wall-clock throughput, but every
//! category reads as zero (the instrumentation compiles away).

use mpt_flat_poc::prof;
use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

const N: usize = 1000;

fn cfg() -> Config {
    Config {
        target_leaf_bytes: 4 * 1024,
        max_leaf_bytes: 8 * 1024,
        min_promote_bytes: 2 * 1024,
    }
}

fn make_db() -> FlatMpt {
    FlatMpt::create(NamedTempFile::new().unwrap().path(), cfg()).unwrap()
}

fn random_keys() -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..N)
        .map(|_| {
            let mut key = [0u8; 32];
            rng.fill_bytes(&mut key);
            key
        })
        .collect()
}

fn sequential_keys() -> Vec<Key> {
    (0..N as u64).map(|i| hashed_key(i.to_le_bytes())).collect()
}

fn shared_prefix_keys() -> Vec<Key> {
    (0..N as u16)
        .map(|i| {
            let mut key = [0u8; 32];
            key[..6].copy_from_slice(&[0xab, 0xcd, 0xef, 0x12, 0x34, 0x50]);
            key[30..].copy_from_slice(&i.to_be_bytes());
            key
        })
        .collect()
}

/// Time `ops` inserts of `keys` into a fresh DB and print the category split.
fn profile_inserts(title: &str, keys: &[Key]) {
    let mut db = make_db();
    prof::reset();
    let start = Instant::now();
    for key in keys {
        black_box(db.insert(*key, vec![1; 32]).unwrap());
    }
    let wall = start.elapsed();
    report(title, wall, keys.len());

    // Read every value back out of the store to characterise the lookup path.
    prof::reset();
    let start = Instant::now();
    let mut sink = 0u8;
    for key in keys {
        if let Some(value) = db.get_value(key).unwrap() {
            sink ^= value[0];
        }
    }
    black_box(sink);
    let wall = start.elapsed();
    report(&format!("{title} — value read-back"), wall, keys.len());
}

fn report(title: &str, wall: Duration, ops: usize) {
    let wall_ns = wall.as_nanos() as u64;
    println!("\n=== {title} ===");
    println!(
        "wall: {:.3} ms over {ops} ops  ({:.3} µs/op)",
        wall_ns as f64 / 1e6,
        wall_ns as f64 / 1e3 / ops as f64,
    );

    if !prof::ENABLED {
        println!("(profiling feature off — rerun with `--features profiling` for the breakdown)");
        return;
    }

    println!(
        "  {:<28} {:>10} {:>7} {:>11} {:>10}",
        "category", "total ms", "% wall", "calls", "µs/call",
    );
    let snapshot = prof::snapshot();
    let mut accounted = 0u64;
    for (i, (nanos, count)) in snapshot.iter().enumerate() {
        accounted += *nanos;
        if *count == 0 {
            continue;
        }
        println!(
            "  {:<28} {:>10.3} {:>6.1}% {:>11} {:>10.3}",
            prof::CATS[i],
            *nanos as f64 / 1e6,
            *nanos as f64 * 100.0 / wall_ns as f64,
            count,
            *nanos as f64 / 1e3 / *count as f64,
        );
    }
    let other = wall_ns.saturating_sub(accounted);
    println!(
        "  {:<28} {:>10.3} {:>6.1}% {:>11} {:>10}",
        "trie/CPU + overhead (rest)",
        other as f64 / 1e6,
        other as f64 * 100.0 / wall_ns as f64,
        "-",
        "-",
    );
}

fn main() {
    if !prof::ENABLED {
        println!(
            "NOTE: built without the `profiling` feature — only wall-clock totals are shown.\n\
             Rerun: cargo bench --bench profile --features profiling"
        );
    }
    profile_inserts("insert 1000 random", &random_keys());
    profile_inserts("insert 1000 sequential-hashed", &sequential_keys());
    profile_inserts("insert 1000 shared-prefix", &shared_prefix_keys());
}
