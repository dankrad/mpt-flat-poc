//! Sustained 300k-key-batch insert benchmark over the 1B Ethereum-shaped baseline —
//! the eth analogue of the fused bench. It simulates a typical EVM write workload:
//! most keys are SSTOREs to **existing** storage slots of **existing** contracts,
//! chosen with probability proportional to each contract's slot count (so the big
//! contracts take most of the writes), plus a small fraction of fresh slots grown
//! onto those same contracts and a few account updates.
//!
//! The baseline distribution is deterministic (seeded), so the workload can pick
//! keys that already exist without reading the trie. First run builds + persists the
//! baseline checkpoint (`<path>` + `<path>.meta`); later runs reopen it and go
//! straight to the fused phase.
//!
//!   MPT_WORKERS=192 MPT_GC_OPP=1 MPT_GC_OPP_UTIL=0.30 MPT_ONE_WRITER=1 \
//!   MPT_FUSED_BATCHES=40 \
//!     cargo run --release --example ethfused -- 7400000 /mnt2/user/ethckpt.flat

use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key, hex, process_footprint_bytes};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const SEED: u64 = 0xE7_1131; // MUST match ethbench so the distribution lines up.

fn account_key(id: u64) -> Key {
    let mut b = [0u8; 9];
    b[0] = 0x00;
    b[1..].copy_from_slice(&id.to_le_bytes());
    hashed_key(b)
}
fn slot_key(contract_id: u64, slot: u64) -> Key {
    let mut b = [0u8; 17];
    b[0] = 0x01;
    b[1..9].copy_from_slice(&contract_id.to_le_bytes());
    b[9..].copy_from_slice(&slot.to_le_bytes());
    hashed_key(b)
}
fn rand_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    rng.fill_bytes(&mut v);
    v
}
fn account_value(rng: &mut StdRng, is_contract: bool) -> Vec<u8> {
    rand_bytes(rng, if is_contract { 104 } else { 70 })
}
fn slot_value(rng: &mut StdRng) -> Vec<u8> {
    let len = rng.gen_range(1..=33);
    rand_bytes(rng, len)
}
fn mem_gib() -> f64 {
    process_footprint_bytes() as f64 / GIB
}
fn write_bytes() -> u64 {
    mpt_flat_poc::stats::WRITE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(7_400_000);
    let path = std::env::args().nth(2).expect("usage: ethfused <n> <checkpoint-path>");
    let max = std::env::var("MAX_LEAF_KIB").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize) * 1024;
    let cfg = Config { target_leaf_bytes: max / 2, max_leaf_bytes: max, min_promote_bytes: max / 2 };
    let build_batch: usize = std::env::var("MPT_BUILD_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let fused_batches: u64 = std::env::var("MPT_FUSED_BATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let fused_batch: usize = std::env::var("MPT_FUSED_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(300_000);
    // Fraction of each batch that inserts a *fresh* slot onto an existing contract
    // (rest overwrite existing slots/accounts). Models storage growth vs churn.
    let new_frac: f64 = std::env::var("MPT_NEW_FRAC").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05);

    let s = (n as f64).sqrt() as u64;
    let lo = (s / 10).max(1);
    let hi = (n / 10).max(lo + 1);

    // Regenerate the baseline distribution: per-large-contract slot counts + prefix
    // sums (identical to ethbench because of the shared seed and draw order).
    let mut rng = StdRng::seed_from_u64(SEED);
    let cnt: Vec<u64> = (0..s).map(|_| rng.gen_range(lo..=hi)).collect();
    let mut cum: Vec<u64> = Vec::with_capacity(s as usize + 1);
    cum.push(0);
    for c in &cnt {
        cum.push(cum.last().unwrap() + c);
    }
    let total_storage = *cum.last().unwrap();
    let total_accounts = n + 2 * s;
    let total_leaves = total_accounts + total_storage;
    let large_id = |j: usize| n + s + j as u64;

    eprintln!(
        "eth fused bench: n={n}, {s} large contracts, {total_storage} storage slots, \
         total leaves {total_leaves}\n  fused: {fused_batches} x {fused_batch}-key batches, \
         new-slot fraction {new_frac}",
    );

    // ---- Baseline: reopen if present, else build (same generation as ethbench) + persist ----
    let meta = format!("{path}.meta");
    let mut db = if Path::new(&meta).exists() {
        eprintln!("reopening baseline checkpoint {path} ...");
        let t = Instant::now();
        let db = FlatMpt::open(&path).unwrap();
        eprintln!("  reopened in {:.1}s  root={}", t.elapsed().as_secs_f64(), hex(db.root()));
        db
    } else {
        eprintln!("building baseline (no checkpoint at {meta}) ...");
        build_baseline(&path, cfg, n, s, &cnt, build_batch)
    };

    // Per-contract "next fresh slot" counter, weighted growth onto existing contracts.
    let mut next_new: Vec<u64> = cnt.clone();

    // ---- Fused phase: sustained 300k-key EVM-like batches ----
    eprintln!("\nfused phase:");
    let flat0 = db.flat_file_len();
    let wb0 = write_bytes();
    let mut us_per_key: Vec<f64> = Vec::new();
    let phase = Instant::now();

    for b in 0..fused_batches {
        let mut batch: Vec<(Key, Vec<u8>)> = Vec::with_capacity(fused_batch);
        for _ in 0..fused_batch {
            if rng.gen_bool(new_frac) {
                // Grow a fresh slot on an existing contract, weighted by size.
                let j = pick_contract(&mut rng, &cum, total_storage);
                let k = next_new[j];
                next_new[j] += 1;
                batch.push((slot_key(large_id(j), k), slot_value(&mut rng)));
            } else {
                // Overwrite an existing leaf, weighted by the baseline population.
                let r = rng.gen_range(0..total_leaves);
                if r < total_accounts {
                    let id = r;
                    let is_contract = id >= n;
                    batch.push((account_key(id), account_value(&mut rng, is_contract)));
                } else {
                    let si = r - total_accounts;
                    let j = cum.partition_point(|&c| c <= si) - 1;
                    let k = si - cum[j];
                    batch.push((slot_key(large_id(j), k), slot_value(&mut rng)));
                }
            }
        }
        let t = Instant::now();
        db.insert_batch(batch).unwrap();
        let us = t.elapsed().as_micros() as f64 / fused_batch as f64;
        us_per_key.push(us);
        let flat = db.flat_file_len();
        let live = db.live_bytes();
        eprintln!(
            "  batch {:>3}/{fused_batches}  {:.3} us/key  flat {:.1} GiB  live {:.1} GiB  util {:.1}%  RSS {:.1} GiB",
            b + 1,
            us,
            flat as f64 / GIB,
            live as f64 / GIB,
            if flat > 0 { live as f64 / flat as f64 * 100.0 } else { 0.0 },
            mem_gib(),
        );
    }

    // ---- Summary ----
    let total_ops = fused_batches * fused_batch as u64;
    let wall = phase.elapsed().as_secs_f64();
    let mut sorted = us_per_key.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = us_per_key.iter().sum::<f64>() / us_per_key.len() as f64;
    let p50 = sorted[sorted.len() / 2];
    let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
    let flat_growth = db.flat_file_len().saturating_sub(flat0);
    let wamp = if total_ops > 0 {
        (write_bytes() - wb0) as f64 / (total_ops as f64 * 40.0) // ~40 B avg logical write/key
    } else {
        0.0
    };
    eprintln!(
        "\nFUSED: {fused_batches} x {fused_batch} = {total_ops} inserts in {wall:.1}s\n\
         per-batch us/key: mean {mean:.3}  p50 {p50:.3}  p99 {p99:.3}  min {:.3}  max {:.3}\n\
         flat-file grew {:.2} GiB (now {:.2} GiB high-water, {:.2} GiB live, util {:.1}%)\n\
         fused write-amp ~{wamp:.1}x   RAM index {:.2} GiB   RSS {:.2} GiB   root={}",
        sorted[0],
        sorted[sorted.len() - 1],
        flat_growth as f64 / GIB,
        db.flat_file_len() as f64 / GIB,
        db.live_bytes() as f64 / GIB,
        if db.flat_file_len() > 0 { db.live_bytes() as f64 / db.flat_file_len() as f64 * 100.0 } else { 0.0 },
        db.ram_report().total_bytes() as f64 / GIB,
        mem_gib(),
        hex(db.root()),
    );
}

/// Pick a large contract weighted by its slot count (big contracts more likely).
fn pick_contract(rng: &mut StdRng, cum: &[u64], total_storage: u64) -> usize {
    let si = rng.gen_range(0..total_storage);
    cum.partition_point(|&c| c <= si) - 1
}

/// Build the 1B baseline exactly as ethbench does, then persist a reopenable
/// checkpoint. Storage values are tiny (1..33 B), accounts 70/104 B; code goes to a
/// side file (not part of the trie).
fn build_baseline(path: &str, cfg: Config, n: u64, s: u64, cnt: &[u64], batch: usize) -> FlatMpt {
    let mut rng = StdRng::seed_from_u64(SEED);
    // Re-draw cnt in the same order so the value RNG stream matches ethbench exactly.
    let _cnt: Vec<u64> = (0..s).map(|_| rng.gen_range((s / 10).max(1)..=(n / 10).max((s / 10) + 1))).collect();
    debug_assert_eq!(&_cnt[..], cnt);

    let mut code_file = OpenOptions::new().create(true).write(true).truncate(true)
        .open(format!("{path}.code")).unwrap();
    let mut db = FlatMpt::create(path, cfg).unwrap();
    let t = Instant::now();
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::with_capacity(batch);
    let mut done: u64 = 0;
    macro_rules! push {
        ($k:expr, $v:expr) => {{
            buf.push(($k, $v));
            done += 1;
            if buf.len() >= batch {
                db.insert_batch(std::mem::take(&mut buf)).unwrap();
                if done % 100_000_000 == 0 {
                    eprintln!("  build {}M leaves  {:.0}s  flat {:.0} GiB",
                        done / 1_000_000, t.elapsed().as_secs_f64(),
                        db.flat_file_len() as f64 / GIB);
                }
            }
        }};
    }
    for id in 0..n {
        push!(account_key(id), account_value(&mut rng, false));
    }
    for j in 0..s {
        push!(account_key(n + j), account_value(&mut rng, true));
        let code_len = rng.gen_range(1024..=20 * 1024);
        let code = rand_bytes(&mut rng, code_len);
        code_file.write_all(&code).unwrap();
    }
    for j in 0..s {
        let id = n + s + j;
        push!(account_key(id), account_value(&mut rng, true));
        for k in 0..cnt[j as usize] {
            push!(slot_key(id, k), slot_value(&mut rng));
        }
    }
    if !buf.is_empty() {
        db.insert_batch(std::mem::take(&mut buf)).unwrap();
    }
    code_file.flush().unwrap();
    eprintln!("  built {done} leaves in {:.0}s, persisting checkpoint ...", t.elapsed().as_secs_f64());
    let ps = Instant::now();
    db.persist().unwrap();
    eprintln!("  persisted in {:.0}s (root={})", ps.elapsed().as_secs_f64(), hex(db.root()));
    db
}
