//! Correctness check: build N keys one-by-one vs in batches and compare the
//! Merkle root + flat-file size. They must match exactly.
//!
//!     cargo run --release --example batchcheck [N] [BATCH]

use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key, hex};
use std::time::Instant;
use tempfile::NamedTempFile;

fn key(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

fn build_one_by_one(n: u64, cfg: Config) -> (FlatMpt, NamedTempFile) {
    let tmp = NamedTempFile::new().unwrap();
    let mut db = FlatMpt::create(tmp.path(), cfg).unwrap();
    for i in 0..n {
        db.insert(key(i), vec![0u8; 32]).unwrap();
    }
    (db, tmp)
}

fn build_batched(n: u64, batch: usize, cfg: Config) -> (FlatMpt, NamedTempFile) {
    let tmp = NamedTempFile::new().unwrap();
    let mut db = FlatMpt::create(tmp.path(), cfg).unwrap();
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::with_capacity(batch);
    for i in 0..n {
        buf.push((key(i), vec![0u8; 32]));
        if buf.len() >= batch {
            db.insert_batch(std::mem::take(&mut buf)).unwrap();
        }
    }
    if !buf.is_empty() {
        db.insert_batch(buf).unwrap();
    }
    (db, tmp)
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let batch: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let cfg = Config::default();
    println!(
        "N={n}, batch={batch}, max_leaf={} KiB",
        cfg.max_leaf_bytes / 1024
    );

    let t = Instant::now();
    let (one, _t1) = build_one_by_one(n, cfg.clone());
    let (root_one, flat_one, ram_one) = (one.root(), one.flat_file_len(), one.ram_nodes());
    println!(
        "one-by-one: {:.1}s  root={}  flat={:.1} MiB  ram_nodes={}",
        t.elapsed().as_secs_f64(),
        hex(root_one),
        flat_one as f64 / 1_048_576.0,
        ram_one,
    );

    let t = Instant::now();
    let (bat, _t2) = build_batched(n, batch, cfg);
    let (root_bat, flat_bat, ram_bat) = (bat.root(), bat.flat_file_len(), bat.ram_nodes());
    println!(
        "batched:    {:.1}s  root={}  flat={:.1} MiB  ram_nodes={}",
        t.elapsed().as_secs_f64(),
        hex(root_bat),
        flat_bat as f64 / 1_048_576.0,
        ram_bat,
    );

    println!(
        "\nroot match: {}   flat match: {}",
        if root_one == root_bat { "YES" } else { "*** NO ***" },
        if flat_one == flat_bat { "YES" } else { "*** NO ***" },
    );
}
