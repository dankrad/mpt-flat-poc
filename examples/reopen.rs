//! Reopen a persisted database, verify it round-trips, and optionally run timed
//! batch inserts against it to break down where insert_batch spends its time.
//!
//!     cargo run --release --example reopen -- /path/to/db.flat [batches]
//!
//! With a `[batches]` count it inserts that many 10k-key batches of fresh keys
//! and reports the A/B/C phase split plus the Phase-B sub-breakdown
//! (read / rebuild / finalize), measured at the reopened database's scale.

use mpt_flat_poc::{FlatMpt, Key, hashed_key, hex, stats};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn key_at(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: reopen <db.flat> [batches]");
    let batches: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut db = FlatMpt::open(&path).expect("open checkpoint");
    println!(
        "reopened {path}\n  root={}\n  ram_nodes={}  flat={:.1} MiB",
        hex(db.root()),
        db.ram_nodes(),
        db.flat_file_len() as f64 / (1024.0 * 1024.0),
    );
    let mut present = 0;
    for i in [0u64, 1, 1000, 1_000_000] {
        if db.get_value(&key_at(i)).unwrap().is_some() {
            present += 1;
        }
    }
    println!("  spot-check: {present}/4 sampled keys present");

    if batches == 0 {
        return;
    }

    // Insert fresh keys (base well beyond any preload) in 10k-key batches, timing
    // the phases. The keys scatter across the existing leaves, so this is exactly
    // the steady-state batch-insert path at this database's scale.
    const B: u64 = 10_000;
    let base = 2_000_000_000u64;
    stats::reset();
    let t = Instant::now();
    for b in 0..batches {
        let buf: Vec<(Key, Vec<u8>)> = (0..B)
            .map(|j| (key_at(base + b * B + j), vec![0u8; 32]))
            .collect();
        db.insert_batch(buf).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    let n = batches * B;

    let (pa, pb, pc) = (
        stats::PHASE_A_NS.load(Relaxed),
        stats::PHASE_B_NS.load(Relaxed),
        stats::PHASE_C_NS.load(Relaxed),
    );
    let (rd, rb, fi) = (
        stats::B_READ_NS.load(Relaxed),
        stats::B_REBUILD_NS.load(Relaxed),
        stats::B_FINAL_NS.load(Relaxed),
    );
    let tot_p = (pa + pb + pc).max(1) as f64;
    let tot_b = (rd + rb + fi).max(1) as f64;
    let per = |ns: u64| ns as f64 / 1e6 / batches as f64; // ms/batch

    println!(
        "\n  {batches} batches x {B} keys = {n} inserts in {secs:.1}s ({:.1} µs/key)",
        secs * 1e6 / n as f64,
    );
    println!(
        "  phases ms/batch: A {:.1} ({:.0}%) | B {:.1} ({:.0}%) | C {:.1} ({:.0}%)",
        per(pa),
        pa as f64 / tot_p * 100.0,
        per(pb),
        pb as f64 / tot_p * 100.0,
        per(pc),
        pc as f64 / tot_p * 100.0,
    );
    println!(
        "  Phase B (summed over threads): read {:.0}% | rebuild {:.0}% | finalize(migrate+write) {:.0}%",
        rd as f64 / tot_b * 100.0,
        rb as f64 / tot_b * 100.0,
        fi as f64 / tot_b * 100.0,
    );
    // Within Phase B: serialize (builds the record payload — a full-leaf walk),
    // the free-list lock (alloc+free), and the pwrite. All summed over threads.
    let ser = stats::B_SERIALIZE_NS.load(Relaxed);
    let lock = stats::W_LOCK_NS.load(Relaxed);
    let pw = stats::W_PWRITE_NS.load(Relaxed);
    println!(
        "  of which (summed): serialize {:.0}% ({:.1} ms/batch) | free-list lock {:.0}% ({:.1}) | pwrite {:.0}% ({:.1})",
        ser as f64 / tot_b * 100.0,
        per(ser),
        lock as f64 / tot_b * 100.0,
        per(lock),
        pw as f64 / tot_b * 100.0,
        per(pw),
    );

    // Re-persist so the on-disk manifest matches the mutated flat file (otherwise
    // the next reopen reads freed/overwritten regions).
    db.persist().unwrap();
}
