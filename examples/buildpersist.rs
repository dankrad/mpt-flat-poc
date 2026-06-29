//! Build N keys (RAM-build mode when MPT_RAM_BUILD=1) and persist a checkpoint,
//! with NO post-build phases — so the on-disk checkpoint stays pristine and can be
//! reopened to verify spill + checkpointing. Prints per-10M build rate, the build
//! root, persist (spill) time, and the resulting flat-file size.
//!
//!   MPT_RAM_BUILD=1 MPT_RAM_BUILD_GIB=300 \
//!     cargo run --release --example buildpersist -- 100000000 /path/ckpt.flat

use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key, hex, process_footprint_bytes};
use std::time::Instant;

// Diagnostic: does a churn-friendly allocator cut the build's footprint? The
// default system allocator may retain freed memory from the heavy per-insert
// Arc/Vec churn across the fan-out threads.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn key(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

// Real committed footprint (resident + compressed + swapped), the same metric
// that drives the spill trigger — NOT ru_maxrss, which plateaus at physical RAM
// once the OS starts compressing/swapping and hides the true growth.
fn mem_gib() -> f64 {
    process_footprint_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn main() {
    let n: u64 = std::env::args().nth(1).expect("usage: buildpersist <n> <path>").parse().unwrap();
    let path = std::env::args().nth(2).expect("usage: buildpersist <n> <path>");
    let max = std::env::var("MAX_LEAF_KIB").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize) * 1024;
    let cfg = Config { target_leaf_bytes: max / 2, max_leaf_bytes: max, min_promote_bytes: max / 2 };

    let mut db = FlatMpt::create(&path, cfg).unwrap();
    // Insert batch size. Large batches let the disk-tail fold coalesce its reads
    // into big sequential ones (much faster than 10k-sparse); GC still runs so the
    // file stays bounded. MPT_BUILD_BATCH overrides (default 1M).
    let b: u64 = std::env::var("MPT_BUILD_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    eprintln!("batch = {b}");
    let t = Instant::now();
    let mut last = Instant::now();
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::with_capacity(b.min(10_000_000) as usize);
    for i in 0..n {
        buf.push((key(i), vec![0u8; 32]));
        if buf.len() as u64 == b {
            db.insert_batch(std::mem::take(&mut buf)).unwrap();
        }
        if (i + 1) % 10_000_000 == 0 {
            eprintln!(
                "[{:>4}M] {:>6.0}s  {:.2} us/key (chunk)  mem {:.1} GiB  flat {:.1} GiB",
                (i + 1) / 1_000_000,
                t.elapsed().as_secs_f64(),
                last.elapsed().as_micros() as f64 / 10_000_000.0,
                mem_gib(),
                db.flat_file_len() as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            last = Instant::now();
        }
    }
    if !buf.is_empty() {
        db.insert_batch(buf).unwrap();
    }
    let build = t.elapsed();
    let root = db.root();
    eprintln!(
        "built {n} keys in {:.1}s ({:.2} us/key), mem {:.1} GiB\n  root={}",
        build.as_secs_f64(),
        build.as_micros() as f64 / n as f64,
        mem_gib(),
        hex(root),
    );
    let ps = Instant::now();
    db.persist().unwrap();
    eprintln!(
        "persist (spill Mem -> disk + manifest): {:.1}s, flat now {:.1} GiB",
        ps.elapsed().as_secs_f64(),
        db.flat_file_len() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    eprintln!("root after persist={}", hex(db.root()));
}
