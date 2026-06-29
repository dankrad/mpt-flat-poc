//! Sustained-insert steady-state probe. Insert N keys into an existing checkpoint
//! in 10k-key batches, printing a milestone every ~10M keys:
//!   - chunk us/key (does the rate hold, or was the short-bench speed a cache hit?)
//!   - logical flat size + utilization (does GC bound the file => steady state?)
//!   - GC reclaim + relocation rate per chunk (is GC keeping pace?)
//!   - process footprint.
//!
//!   cargo run --release --example sustain -- <ckpt.flat> <n_keys> [start_key]
//!
//! Run with GC ON (the default disk path) — gc-off would balloon the file. Tune
//! via env (MPT_WORKERS, MPT_FOLD, MPT_GC_OPP, MPT_NO_WAL, ...).

use mpt_flat_poc::{FlatMpt, Key, hashed_key, hex, process_footprint_bytes, stats};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn key(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: sustain <ckpt> <n> [start]");
    let n: u64 = a.next().expect("n").parse().unwrap();
    let start: u64 = a.next().map(|s| s.parse().unwrap()).unwrap_or(3_000_000_000);

    let mut db = FlatMpt::open(&path).unwrap();
    eprintln!(
        "open {path}\n  root before = {}\n  flat {:.1} GiB  live {:.1} GiB  util {:.0}%",
        hex(db.root()),
        db.flat_file_len() as f64 / GIB,
        db.live_bytes() as f64 / GIB,
        db.live_bytes() as f64 / db.flat_file_len().max(1) as f64 * 100.0,
    );

    const B: u64 = 10_000;
    const MILE: u64 = 10_000_000;
    let t = Instant::now();
    let mut last = Instant::now();
    let (mut lreg, mut lrel) = (0u64, 0u64);
    let mut i = start;
    let mut done = 0u64;
    let mut next = MILE;
    while done < n {
        let buf: Vec<(Key, Vec<u8>)> = (i..i + B).map(|k| (key(k), vec![0u8; 32])).collect();
        db.insert_batch(buf).unwrap();
        done += B;
        i += B;
        if done >= next {
            let reg = stats::GC_REGIONS.load(Relaxed);
            let rel = stats::GC_RELOCATED.load(Relaxed);
            let flat = db.flat_file_len() as f64 / GIB;
            let live = db.live_bytes() as f64 / GIB;
            eprintln!(
                "[{:>4}M] {:>6.0}s  {:.2} us/key  flat {:.1} GiB  util {:.0}%  free {:.1} GiB  gc:+{}reg +{:.2}reloc/key  mem {:.1} GiB",
                done / 1_000_000,
                t.elapsed().as_secs_f64(),
                last.elapsed().as_micros() as f64 / MILE as f64,
                flat,
                live / flat.max(0.001) * 100.0,
                db.free_bytes() as f64 / GIB,
                reg - lreg,
                (rel - lrel) as f64 / MILE as f64,
                process_footprint_bytes() as f64 / GIB,
            );
            last = Instant::now();
            (lreg, lrel) = (reg, rel);
            next += MILE;
        }
    }
    eprintln!(
        "done: {} keys, {:.2} us/key avg, flat {:.1} GiB, util {:.0}%\n  root after = {}",
        n,
        t.elapsed().as_micros() as f64 / n as f64,
        db.flat_file_len() as f64 / GIB,
        db.live_bytes() as f64 / db.flat_file_len().max(1) as f64 * 100.0,
        hex(db.root()),
    );
}
