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
    let tot_p = (pa + pb + pc).max(1) as f64;
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

    // Phase-B drill-down. Each counter is summed across the worker threads, so it
    // measures total CPU/IO *work* per key (thread-µs/key); dividing the summed
    // total by the Phase-B wall time gives the effective concurrency. The seven
    // components are disjoint and sum to the whole of Phase B:
    //   read = pread (device) + parse (cpu);  rebuild (cpu keccak);
    //   migrate = B_FINAL - serialize (cpu);  serialize (cpu);
    //   alloc-lock (contention);  pwrite (device).
    let io = stats::B_READ_IO_NS.load(Relaxed);
    let parse = stats::B_READ_PARSE_NS.load(Relaxed);
    let rebuild = stats::B_REBUILD_NS.load(Relaxed);
    let ser = stats::B_SERIALIZE_NS.load(Relaxed);
    let migrate = stats::B_FINAL_NS.load(Relaxed).saturating_sub(ser);
    let lock = stats::W_LOCK_NS.load(Relaxed);
    let pw = stats::W_PWRITE_NS.load(Relaxed);
    let summed = (io + parse + rebuild + migrate + ser + lock + pw).max(1);
    let uk = |ns: u64| ns as f64 / 1000.0 / n as f64; // thread-µs/key
    let pct = |ns: u64| ns as f64 / summed as f64 * 100.0;

    println!(
        "  Phase B work (thread-µs/key, summed over threads; effective concurrency {:.1}x):",
        summed as f64 / pb.max(1) as f64,
    );
    println!(
        "    read.pread (device) {:.2} ({:.0}%) | read.parse (cpu) {:.2} ({:.0}%)",
        uk(io), pct(io), uk(parse), pct(parse),
    );
    println!(
        "    rebuild (cpu)       {:.2} ({:.0}%) | migrate (cpu)     {:.2} ({:.0}%) | serialize (cpu) {:.2} ({:.0}%)",
        uk(rebuild), pct(rebuild), uk(migrate), pct(migrate), uk(ser), pct(ser),
    );
    println!(
        "    pwrite (device)     {:.2} ({:.0}%) | alloc-lock (cont) {:.2} ({:.0}%)",
        uk(pw), pct(pw), uk(lock), pct(lock),
    );
    println!(
        "  grouped: device(pread+pwrite) {:.2} ({:.0}%) | cpu(parse+rebuild+migrate+serialize) {:.2} ({:.0}%) | contention {:.2} ({:.0}%)",
        uk(io + pw), pct(io + pw),
        uk(parse + rebuild + migrate + ser), pct(parse + rebuild + migrate + ser),
        uk(lock), pct(lock),
    );
    // Phase C is serial today; split it to see what a parallel version could win.
    let (ci, cr, cf) = (
        stats::C_INSTALL_NS.load(Relaxed),
        stats::C_ROOT_NS.load(Relaxed),
        stats::C_FLUSH_NS.load(Relaxed),
    );
    println!(
        "  Phase C: install+fresh {:.1} ms/batch | root recompute {:.1} | flush {:.1}",
        per(ci),
        per(cr),
        per(cf),
    );
    // GC runs between Phase B and C and is timed separately (not in A/B/C above).
    let gc_ns = stats::GC_NS.load(Relaxed);
    let gc_reloc = stats::GC_RELOCATED.load(Relaxed);
    let gc_regs = stats::GC_REGIONS.load(Relaxed);
    println!(
        "  GC: {:.1} ms/batch ({:.0}% of wall) | {:.2} reloc/key | {} regions reclaimed",
        per(gc_ns),
        gc_ns as f64 / 1e9 / secs * 100.0,
        gc_reloc as f64 / n as f64,
        gc_regs,
    );

    // Re-persist only when asked (REOPEN_PERSIST=1). Skipping it keeps the
    // checkpoint pristine so the same 1B base can be reused for repeated A/B runs
    // (the appended records become orphans the next run's allocator overwrites).
    if std::env::var("REOPEN_PERSIST").as_deref() == Ok("1") {
        db.persist().unwrap();
    }
}
