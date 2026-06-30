//! Profile incremental inserts into an existing checkpoint and dump the phase
//! breakdown, for knob tuning (MPT_WORKERS / MPT_DIRECT_IO / MPT_FOLD /
//! MPT_GC_DISABLE / MPT_BATCHED_WRITES). Default batch = 10k.
//!
//!   MPT_WORKERS=32 cargo run --release --example profins -- \
//!       /tmp/ckpt.flat 2000000 3000000000 10000

use mpt_flat_poc::{FlatMpt, Key, hashed_key, process_footprint_bytes, stats};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn key(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: profins <ckpt> <n> <start> [batch]");
    let n: u64 = a.next().expect("n").parse().unwrap();
    let start: u64 = a.next().map(|s| s.parse().unwrap()).unwrap_or(3_000_000_000);
    let batch: u64 = a.next().map(|s| s.parse().unwrap()).unwrap_or(10_000);

    let mut db = FlatMpt::open(&path).unwrap();
    let mut i = start;
    // Optional warmup (MPT_PROF_WARMUP keys) inserted *before* measuring + resetting
    // stats — drives GC to its steady-state utilization so the evac breakdown
    // reflects the steady regime, not the dilute early phase (regions still >util).
    let warmup: u64 = std::env::var("MPT_PROF_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if warmup > 0 {
        let mut done = 0u64;
        while done < warmup {
            let this = batch.min(warmup - done);
            let entries: Vec<(Key, Vec<u8>)> = (i..i + this).map(|k| (key(k), vec![0u8; 32])).collect();
            db.insert_batch(entries).unwrap();
            done += this;
            i += this;
        }
        eprintln!("warmup: {warmup} keys, flat {:.1} GiB", db.flat_file_len() as f64 / (1024.0 * 1024.0 * 1024.0));
    }
    let flat0 = db.flat_file_len();
    stats::reset();
    let t = Instant::now();
    let mut done = 0u64;
    while done < n {
        let this = batch.min(n - done);
        let entries: Vec<(Key, Vec<u8>)> = (i..i + this).map(|k| (key(k), vec![0u8; 32])).collect();
        db.insert_batch(entries).unwrap();
        done += this;
        i += this;
    }
    let wall = t.elapsed().as_secs_f64();

    // One-writer device-busy: wall in read phase + write phase vs total.
    let ow_read = stats::OW_READ_NS.load(Relaxed);
    let ow_write = stats::OW_WRITE_NS.load(Relaxed);
    if ow_read + ow_write > 0 {
        let total_ns = wall * 1e9;
        println!(
            "\n  one-writer device: read {:.2} us/key + write {:.2} us/key = {:.0}% of wall busy ({:.0}% idle: route/install/value)",
            ow_read as f64 / 1000.0 / n as f64,
            ow_write as f64 / 1000.0 / n as f64,
            (ow_read + ow_write) as f64 / total_ns * 100.0,
            (1.0 - (ow_read + ow_write) as f64 / total_ns) * 100.0,
        );
    }

    // Fused-GC evacuation breakdown: where the GC cost actually goes.
    let ev_regions = stats::GC_EVAC_REGIONS.load(Relaxed);
    if ev_regions > 0 {
        let ev_read = stats::GC_EVAC_BYTES_READ.load(Relaxed);
        let ev_live = stats::GC_EVAC_LIVE_BYTES.load(Relaxed);
        let ev_reloc = stats::GC_RELOC_BYTES.load(Relaxed);
        let ev_read_ns = stats::GC_EVAC_READ_NS.load(Relaxed);
        let gib = 1024.0 * 1024.0 * 1024.0;
        let wbytes_all = stats::WRITE_BYTES.load(Relaxed).max(1);
        println!(
            "  gc-evac: {} regions  read {:.1} GiB ({:.0} KiB/region, util {:.0}%)  reloc {:.2} GiB  read-amp {:.1}x/reloc-byte  evac-read {:.2} us/key  reloc-write-share {:.0}%",
            ev_regions,
            ev_read as f64 / gib,
            ev_read as f64 / ev_regions as f64 / 1024.0,
            ev_live as f64 / ev_read.max(1) as f64 * 100.0,
            ev_reloc as f64 / gib,
            ev_read as f64 / ev_reloc.max(1) as f64,
            ev_read_ns as f64 / 1000.0 / n as f64,
            ev_reloc as f64 / wbytes_all as f64 * 100.0,
        );
    }

    let pa = stats::PHASE_A_NS.load(Relaxed);
    let pb = stats::PHASE_B_NS.load(Relaxed);
    let pc = stats::PHASE_C_NS.load(Relaxed);
    let phases = (pa + pb + pc).max(1) as f64;
    let a_build = stats::A_BUILD_NS.load(Relaxed);
    let a_route = stats::A_ROUTE_NS.load(Relaxed);
    let us = |x: u64| x as f64 / 1000.0 / n as f64;
    println!(
        "  A split (us/key): build(hash+maps)={:.3}  route(walks)={:.3}",
        us(a_build), us(a_route),
    );
    // Phase B is parallel: sub-buckets are summed across worker threads, so only
    // their internal ratios are meaningful. Report them as a share of B-summed.
    let brebuild = stats::B_REBUILD_NS.load(Relaxed);
    let bfinal = stats::B_FINAL_NS.load(Relaxed);
    let bio = stats::B_READ_IO_NS.load(Relaxed);
    let bparse = stats::B_READ_PARSE_NS.load(Relaxed);
    // Coalesced path records the span read in B_READ_IO_NS (not B_READ_NS), so use
    // io as the read bucket. rebuild + final are the per-record CPU.
    let bread = bio;
    let bsum = (bio + brebuild + bfinal).max(1) as f64;
    let bser = stats::B_SERIALIZE_NS.load(Relaxed);
    let wlock = stats::W_LOCK_NS.load(Relaxed);
    let wpwrite = stats::W_PWRITE_NS.load(Relaxed);
    let cinstall = stats::C_INSTALL_NS.load(Relaxed);
    let croot = stats::C_ROOT_NS.load(Relaxed);
    let cflush = stats::C_FLUSH_NS.load(Relaxed);
    let csum = (cinstall + croot + cflush).max(1) as f64;
    // Effective Phase-B concurrency = summed worker time / Phase-B wall.
    let b_conc = bsum / (pb.max(1) as f64);
    let writes = stats::WRITES.load(Relaxed);
    let wbytes = stats::WRITE_BYTES.load(Relaxed);

    let workers = std::env::var("MPT_WORKERS").unwrap_or_else(|_| "default".into());
    let direct = std::env::var("MPT_DIRECT_IO").unwrap_or_else(|_| "0".into());
    let fold = std::env::var("MPT_FOLD").unwrap_or_else(|_| "1".into());
    let gc = if std::env::var("MPT_GC_DISABLE").as_deref() == Ok("1") {
        "off".to_string()
    } else if std::env::var("MPT_GC_OPP").as_deref() == Ok("1") {
        format!("opp<{}", std::env::var("MPT_GC_OPP_UTIL").unwrap_or_else(|_| "0.30".into()))
    } else {
        "full".to_string()
    };
    let flat_grow = (db.flat_file_len().saturating_sub(flat0)) as f64 / (1024.0 * 1024.0 * 1024.0);
    let reloc = stats::GC_RELOCATED.load(Relaxed);

    println!(
        "\n=== {n} keys, batch={batch}  W={workers} direct={direct} fold={fold} gc={gc} ===\n\
         {:.2} us/key  ({:.1}s)   mem {:.1} GiB\n\
         phases (wall):  A={:.0}% (route)   B={:.0}% (per-record)   C={:.0}% (install+root+flush)\n\
         B-conc ~{:.1}x   B split (thread-sum): read={:.0}% rebuild={:.0}% final={:.0}%\n\
           read = io {:.0}% + parse {:.0}%   |   final incl serialize {:.0}%, wlock {:.0}%, pwrite {:.0}%  (of B-sum)\n\
         C split:  install={:.0}%  root={:.0}%  flush={:.0}%\n\
         writes={} write_amp={:.0} B/key   gc_ms={} reloc={} flat_grow={:.1}GiB ({:.0} B/key)",
        wall * 1e6 / n as f64,
        wall,
        process_footprint_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        pa as f64 / phases * 100.0,
        pb as f64 / phases * 100.0,
        pc as f64 / phases * 100.0,
        b_conc,
        bread as f64 / bsum * 100.0,
        brebuild as f64 / bsum * 100.0,
        bfinal as f64 / bsum * 100.0,
        bio as f64 / bsum * 100.0,
        bparse as f64 / bsum * 100.0,
        bser as f64 / bsum * 100.0,
        wlock as f64 / bsum * 100.0,
        wpwrite as f64 / bsum * 100.0,
        cinstall as f64 / csum * 100.0,
        croot as f64 / csum * 100.0,
        cflush as f64 / csum * 100.0,
        writes,
        wbytes as f64 / n as f64,
        ms(stats::GC_NS.load(Relaxed)) as u64,
        reloc,
        flat_grow,
        flat_grow * 1024.0 * 1024.0 * 1024.0 / n as f64,
    );
}
