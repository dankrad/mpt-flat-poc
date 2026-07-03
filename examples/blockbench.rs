//! Per-block apply_block benchmark on a nested mainnet checkpoint: applies
//! synthetic block-shaped updates (slot overwrites weighted at real contracts,
//! fresh slots, account field updates, slot deletes, fresh accounts) and
//! reports µs/op vs ops-per-block — the WP0 baseline for the shadow follower.
//!
//! Keys are sampled from the same TSV exports the checkpoint was built from,
//! so every "existing" op hits a real account/slot.
//!
//!   cargo run --release --example blockbench -- \
//!     <ckpt.flat> <accounts.tsv> <storages.tsv> <ops-per-block> <n-blocks>

use alloy_primitives::U256;
use mpt_flat_poc::{FlatMpt, Key, StateOp, eth, hex, process_footprint_bytes};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn hex32(s: &str) -> Key {
    let mut k = [0u8; 32];
    let b = alloy_primitives::hex::decode(s.trim_start_matches("0x")).unwrap();
    k[32 - b.len()..].copy_from_slice(&b);
    k
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: blockbench <flat> <accounts.tsv> <storages.tsv> <ops/block> <blocks>");
    let accounts_tsv = args.next().unwrap();
    let storages_tsv = args.next().unwrap();
    let ops_per_block: usize = args.next().unwrap().parse().unwrap();
    let n_blocks: usize = args.next().unwrap().parse().unwrap();

    // Sample existing account keys and (account, slot) pairs from the TSVs.
    let sample_every = 37;
    let mut accounts: Vec<Key> = Vec::new();
    for (i, line) in BufReader::new(std::fs::File::open(&accounts_tsv).unwrap()).lines().enumerate() {
        if i % sample_every == 0 {
            accounts.push(hex32(line.unwrap().split('\t').next().unwrap()));
        }
        if accounts.len() >= 200_000 {
            break;
        }
    }
    let mut slots: Vec<(Key, Key)> = Vec::new();
    for (i, line) in BufReader::new(std::fs::File::open(&storages_tsv).unwrap()).lines().enumerate() {
        if i % sample_every == 0 {
            let l = line.unwrap();
            let mut it = l.split('\t');
            slots.push((hex32(it.next().unwrap()), hex32(it.next().unwrap())));
        }
        if slots.len() >= 400_000 {
            break;
        }
    }
    eprintln!("sampled {} accounts, {} slots", accounts.len(), slots.len());

    let mut db = FlatMpt::open(&flat).unwrap();
    let mut rng = StdRng::seed_from_u64(0xB33F);
    let mut times_us: Vec<f64> = Vec::new();
    let t0 = Instant::now();

    for b in 0..n_blocks {
        let mut ops: Vec<(Key, StateOp)> = Vec::with_capacity(ops_per_block);
        while ops.len() < ops_per_block {
            match rng.gen_range(0..100u32) {
                // Overwrite an existing slot (the dominant real-block op).
                0..=59 => {
                    let (a, s) = slots[rng.gen_range(0..slots.len())];
                    ops.push((a, StateOp::SetStorage {
                        slot: s,
                        value: eth::storage_value_rlp(U256::from(rng.r#gen::<u64>())),
                    }));
                }
                // Fresh slot on an existing contract.
                60..=74 => {
                    let (a, _) = slots[rng.gen_range(0..slots.len())];
                    let mut s = [0u8; 32];
                    rng.fill(&mut s);
                    ops.push((a, StateOp::SetStorage {
                        slot: s,
                        value: eth::storage_value_rlp(U256::from(rng.r#gen::<u64>())),
                    }));
                }
                // Account field update (balance/nonce churn).
                75..=89 => {
                    let a = accounts[rng.gen_range(0..accounts.len())];
                    ops.push((a, StateOp::SetAccount {
                        nonce: rng.gen_range(0..1_000_000),
                        balance: U256::from(rng.r#gen::<u128>()),
                        code_hash: eth::EMPTY_CODE_HASH.0,
                    }));
                }
                // Slot delete (SSTORE to zero).
                90..=94 => {
                    let (a, s) = slots[rng.gen_range(0..slots.len())];
                    ops.push((a, StateOp::DeleteStorage { slot: s }));
                }
                // Fresh account.
                _ => {
                    let mut a = [0u8; 32];
                    rng.fill(&mut a);
                    ops.push((a, StateOp::SetAccount {
                        nonce: 1,
                        balance: U256::from(rng.r#gen::<u64>()),
                        code_hash: eth::EMPTY_CODE_HASH.0,
                    }));
                }
            }
        }
        let t = Instant::now();
        let (_root, _inv) = db.apply_block(ops).unwrap();
        let us = t.elapsed().as_micros() as f64;
        times_us.push(us);
        if (b + 1) % 10 == 0 {
            eprintln!("  block {:>4}/{n_blocks}: {:.0} us  ({:.2} us/op)", b + 1, us, us / ops_per_block as f64);
        }
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tot: f64 = times_us.iter().sum();
    let pct = |p: f64| times_us[((times_us.len() as f64 * p) as usize).min(times_us.len() - 1)];
    println!(
        "ops/block {ops_per_block}  blocks {n_blocks}  wall {:.1}s\n\
         per-block ms: mean {:.1}  p50 {:.1}  p90 {:.1}  p99 {:.1}  max {:.1}\n\
         per-op us: mean {:.2}  root {}  RSS {:.1} GiB",
        t0.elapsed().as_secs_f64(),
        tot / times_us.len() as f64 / 1000.0,
        pct(0.50) / 1000.0,
        pct(0.90) / 1000.0,
        pct(0.99) / 1000.0,
        times_us[times_us.len() - 1] / 1000.0,
        tot / (times_us.len() * ops_per_block) as f64,
        hex(db.root()),
        process_footprint_bytes() as f64 / (1u64 << 30) as f64,
    );
}
