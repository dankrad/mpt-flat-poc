//! Insert N fresh keys into an existing (disk-resident) checkpoint, in batches,
//! and report per-batch + overall throughput. Used by both `batch-bench.sh`
//! (benchmark a clone; no persist) and `grow-tree.sh` (grow the real tree;
//! `MPT_PERSIST=1` checkpoints the result at the end). New keys hash uniformly
//! across the trie's leaves. Tuning is via env (see README): MPT_WORKERS,
//! MPT_ONE_WRITER, MPT_NO_WAL, MPT_ASYNC_VALUES, MPT_GC_DISABLE, MPT_FOLD.
//!
//!   MPT_ONE_WRITER=1 MPT_NO_WAL=1 cargo run --release --example foldbench -- \
//!       /path/ckpt.flat 20000000 2000000000 10000
//!         args: <checkpoint> <n_keys> <start_index> <batch_size>

use mpt_flat_poc::{FlatMpt, Key, hashed_key, hex, process_footprint_bytes};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn key(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: foldbench <ckpt> <n> <start> <batch>");
    let n: u64 = a.next().expect("n").parse().unwrap();
    let start: u64 = a.next().map(|s| s.parse().unwrap()).unwrap_or(1_000_000_000);
    let batch: u64 = a.next().map(|s| s.parse().unwrap()).unwrap_or(5_000_000);

    let fold = std::env::var("MPT_FOLD").as_deref() != Ok("0");
    eprintln!(
        "open {path}  (MPT_FOLD={}  n={n}  start={start}  batch={batch})",
        if fold { "1 coalesced" } else { "0 random" }
    );
    // cfg (max_leaf etc.) is restored from the checkpoint manifest by open().
    let mut db = FlatMpt::open(&path).unwrap();
    eprintln!("root before = {}", hex(db.root()));

    let t = Instant::now();
    let mut done = 0u64;
    let mut i = start;
    while done < n {
        let this = batch.min(n - done);
        let bt = Instant::now();
        let entries: Vec<(Key, Vec<u8>)> = (i..i + this).map(|k| (key(k), vec![0u8; 32])).collect();
        db.insert_batch(entries).unwrap();
        done += this;
        i += this;
        eprintln!(
            "  +{:>4}M  {:>6.2}s  {:.2} us/key (batch)  mem {:.1} GiB  flat {:.1} GiB",
            done / 1_000_000,
            t.elapsed().as_secs_f64(),
            bt.elapsed().as_micros() as f64 / this as f64,
            process_footprint_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            db.flat_file_len() as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    eprintln!(
        "inserted {n} keys in {:.1}s ({:.2} us/key)\n  root after = {}",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_micros() as f64 / n as f64,
        hex(db.root()),
    );

    // grow-tree.sh sets MPT_PERSIST=1 to checkpoint the enlarged tree; the
    // benchmark path leaves the (cloned) checkpoint untouched.
    if std::env::var("MPT_PERSIST").as_deref() == Ok("1") {
        let ps = Instant::now();
        db.persist().unwrap();
        eprintln!(
            "persisted in {:.1}s, flat now {:.1} GiB",
            ps.elapsed().as_secs_f64(),
            db.flat_file_len() as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
}
