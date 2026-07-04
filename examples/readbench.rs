//! Value-fetch benchmark on a nested mainnet checkpoint: random account and
//! storage-slot point reads, warm and cold-ish, at several thread counts,
//! with latency percentiles. Keys come from the TSV exports so every read hits
//! a real entry.
//!
//!   cargo run --release --example readbench -- <ckpt.flat> <accounts.tsv> <storages.tsv> <n-reads>

use mpt_flat_poc::{FlatMpt, Key};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn hex32(s: &str) -> Key {
    let mut k = [0u8; 32];
    let b = alloy_primitives::hex::decode(s.trim_start_matches("0x")).unwrap();
    k[32 - b.len()..].copy_from_slice(&b);
    k
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: readbench <flat> <accounts.tsv> <storages.tsv> <n-reads>");
    let accounts_tsv = args.next().unwrap();
    let storages_tsv = args.next().unwrap();
    let n_reads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);

    let accounts: Vec<Key> = BufReader::new(std::fs::File::open(&accounts_tsv).unwrap())
        .lines()
        .map(|l| hex32(l.unwrap().split('\t').next().unwrap()))
        .collect();
    let slots: Vec<(Key, Key)> = BufReader::new(std::fs::File::open(&storages_tsv).unwrap())
        .lines()
        .map(|l| {
            let l = l.unwrap();
            let mut it = l.split('\t');
            (hex32(it.next().unwrap()), hex32(it.next().unwrap()))
        })
        .collect();
    eprintln!("keys: {} accounts, {} slots", accounts.len(), slots.len());

    let db = FlatMpt::open(&flat).unwrap();

    // Warm pass first (also validates presence), then the measured passes.
    for phase in ["account", "storage"] {
        for threads in [1usize, 8, 32] {
            let counter = AtomicUsize::new(0);
            let t0 = Instant::now();
            let lat: Vec<Vec<f64>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|ti| {
                        let db = &db;
                        let accounts = &accounts;
                        let slots = &slots;
                        let counter = &counter;
                        scope.spawn(move || {
                            let mut rng = StdRng::seed_from_u64(0xF00D + ti as u64);
                            let mut lats = Vec::new();
                            loop {
                                let i = counter.fetch_add(1, Ordering::Relaxed);
                                if i >= n_reads {
                                    break;
                                }
                                let t = Instant::now();
                                let found = if phase == "account" {
                                    let k = &accounts[rng.gen_range(0..accounts.len())];
                                    db.get_value(k).unwrap().is_some()
                                } else {
                                    let (a, s) = &slots[rng.gen_range(0..slots.len())];
                                    db.get_storage(a, s).unwrap().is_some()
                                };
                                lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
                                assert!(found, "sampled key missing");
                            }
                            lats
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let all: Vec<f64> = lat.into_iter().flatten().collect();
            let wall = t0.elapsed().as_secs_f64();
            let n = all.len() as f64;
            println!(
                "{phase} reads x{threads:>2}: {:>9.0} reads/s   p50 {:>6.1} us  p90 {:>6.1} us  p99 {:>6.1} us  max {:>7.1} us",
                n / wall,
                pct(all.clone(), 0.50),
                pct(all.clone(), 0.90),
                pct(all.clone(), 0.99),
                pct(all, 1.0),
            );
        }
    }
}
