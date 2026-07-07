//! IO-vs-compute split of tempo-shaped block applies.
//!
//! Builds a hot-contracts state (4 tokens x N slots), then measures per-apply:
//! engine phase nanos (requires `--features profiling`), rusage CPU time,
//! /proc/self/io bytes, and wall time — warm page cache vs cold (fadvise
//! DONTNEED on the flat file between applies), with the hot-record cache on
//! or off (MPT_HOT_RECORDS).
//!
//!   cargo run --release --features profiling --example applyprofile -- \
//!       <slots_per_contract> <blocks> <ops_per_block> [cold]

use mpt_flat_poc::{prof, Config, FlatMpt, Key, StateOp};
use sha3::{Digest, Keccak256};
use std::time::Instant;

fn h(data: &[u8]) -> Key {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Keccak256::digest(data));
    out
}

fn proc_io() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/io").unwrap();
    let get = |k: &str| {
        s.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };
    (get("read_bytes"), get("write_bytes"))
}

fn cpu_seconds() -> f64 {
    let mut ru = unsafe { std::mem::zeroed::<libc::rusage>() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

fn drop_cache(path: &std::path::Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(f) = std::fs::File::open(path) {
        unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
}

const CATS: [&str; 8] = [
    "keccak", "serialize", "deserialize", "file_read", "file_write", "flush", "value_put", "value_get",
];

fn main() {
    let slots: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(25_000_000);
    let blocks: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let ops: u64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(94_000);
    let cold = std::env::args().nth(4).as_deref() == Some("cold");
    assert!(prof::ENABLED, "build with --features profiling");

    let dir = std::path::PathBuf::from("/mnt2/applyprofile");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("p.flat");
    // Record-size sensitivity: MPT_PROFILE_LEAF_KB overrides target/max leaf bytes.
    let mut cfg = Config::default();
    if let Ok(kb) = std::env::var("MPT_PROFILE_LEAF_KB").map(|s| s.parse::<usize>().unwrap()) {
        cfg.target_leaf_bytes = kb * 1024;
        cfg.max_leaf_bytes = kb * 2048;
        cfg.min_promote_bytes = kb * 1024;
    }

    const TOKENS: u64 = 4;
    let empty_code = mpt_flat_poc::eth::EMPTY_CODE_HASH.0;
    eprintln!("building {TOKENS} x {slots} slots...");
    let t0 = Instant::now();
    let mut db = FlatMpt::create_ram_build(&path, cfg).unwrap();
    for t in 0..TOKENS {
        let key = h(format!("token{t}").as_bytes());
        let mut sl: Vec<(Key, Vec<u8>)> = (0..slots)
            .map(|s| {
                (
                    h(&(t * 1_000_000_000 + s).to_be_bytes()),
                    mpt_flat_poc::eth::storage_value_rlp(alloy_primitives::U256::from(s + 1)),
                )
            })
            .collect();
        sl.sort_by(|a, b| a.0.cmp(&b.0));
        db.insert_batch_accounts(vec![(key, mpt_flat_poc::AccountSeed {
            nonce: 1,
            balance: alloy_primitives::U256::ZERO,
            code_hash: h(b"code"),
            slots: sl,
        })])
        .unwrap();
    }
    // user accounts
    let users: Vec<(Key, mpt_flat_poc::AccountSeed)> = (0..8000u64)
        .map(|u| {
            (h(&u.to_be_bytes()), mpt_flat_poc::AccountSeed {
                nonce: 1,
                balance: alloy_primitives::U256::from(1u64 << 40),
                code_hash: empty_code,
                slots: Vec::new(),
            })
        })
        .collect();
    let mut users: Vec<_> = users;
    users.sort_by(|a, b| a.0.cmp(&b.0));
    db.insert_batch_accounts(users).unwrap();
    db.persist().unwrap();
    drop(db);
    let mut db = FlatMpt::open(&path).unwrap();
    eprintln!("built + reopened in {:.0}s; cold={cold} hot_records={}", t0.elapsed().as_secs_f64(),
        std::env::var("MPT_HOT_RECORDS").unwrap_or_else(|_| "default".into()));

    let mut rng: u64 = 0x9e3779b97f4a7c15;
    let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };

    println!("block,wall_ms,cpu_ms,io_read_mb,io_write_mb,{}", CATS.map(|c| format!("{c}_ms")).join(","));
    for b in 0..blocks {
        // tempo shape: per tx 2 slot updates on a random token + a user account bump
        let ntx = ops / 5;
        let mut block: Vec<(Key, StateOp)> = Vec::with_capacity(ops as usize);
        for _ in 0..ntx {
            let t = next() % TOKENS;
            let key = h(format!("token{t}").as_bytes());
            for _ in 0..2 {
                let s = next() % slots;
                block.push((key, StateOp::SetStorage {
                    slot: h(&(t * 1_000_000_000 + s).to_be_bytes()),
                    value: mpt_flat_poc::eth::storage_value_rlp(alloy_primitives::U256::from(next() | 1)),
                }));
            }
            let u = next() % 8000;
            block.push((h(&u.to_be_bytes()), StateOp::SetAccount {
                nonce: b + 2,
                balance: alloy_primitives::U256::from(1u64 << 40),
                code_hash: empty_code,
            }));
        }
        block.sort_by(|a, b| a.0.cmp(&b.0));

        if cold {
            drop_cache(&path);
        }
        use mpt_flat_poc::stats;
        use std::sync::atomic::Ordering::Relaxed;
        let s0 = [
            stats::WRITE_BYTES.load(Relaxed),
            stats::GC_RELOC_BYTES.load(Relaxed),
            stats::GC_EVAC_BYTES_READ.load(Relaxed),
            stats::PROMOTE_CHILD_BYTES.load(Relaxed),
            stats::SPLIT_LEAF_BYTES.load(Relaxed),
            stats::WRITES.load(Relaxed),
        ];
        prof::reset();
        let (r0, w0) = proc_io();
        let c0 = cpu_seconds();
        let t = Instant::now();
        db.apply_block(block).unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let c1 = cpu_seconds();
        let (r1, w1) = proc_io();
        let snap = prof::snapshot();
        let s1 = [
            stats::WRITE_BYTES.load(Relaxed),
            stats::GC_RELOC_BYTES.load(Relaxed),
            stats::GC_EVAC_BYTES_READ.load(Relaxed),
            stats::PROMOTE_CHILD_BYTES.load(Relaxed),
            stats::SPLIT_LEAF_BYTES.load(Relaxed),
            stats::WRITES.load(Relaxed),
        ];
        println!(
            "{b},{:.0},{:.0},{:.1},{:.1},{},payload={:.0},gc_reloc={:.0},gc_read={:.0},promote={:.0},split={:.0},nwrites={}",
            wall,
            (c1 - c0) * 1e3,
            (r1 - r0) as f64 / 1e6,
            (w1 - w0) as f64 / 1e6,
            snap.map(|(n, _)| format!("{:.0}", n as f64 / 1e6)).join(","),
            (s1[0] - s0[0]) as f64 / 1e6,
            (s1[1] - s0[1]) as f64 / 1e6,
            (s1[2] - s0[2]) as f64 / 1e6,
            (s1[3] - s0[3]) as f64 / 1e6,
            (s1[4] - s0[4]) as f64 / 1e6,
            s1[5] - s0[5],
        );
    }
}
