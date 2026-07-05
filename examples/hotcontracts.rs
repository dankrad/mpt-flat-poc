//! Tempo-shaped workload: a few token contracts absorb almost all storage
//! writes. 4 hot contracts × N slots each, plus user account churn — the
//! opposite of mainnet's spread access. Measures apply_block latency when the
//! per-account storage groups are huge and the top-level fan-out degenerates.
//!
//!   cargo run --release --example hotcontracts [slots_per_contract] [blocks]

use mpt_flat_poc::{Config, FlatMpt, Key, StateOp};
use sha3::{Digest, Keccak256};
use std::time::Instant;

fn h(data: &[u8]) -> Key {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Keccak256::digest(data));
    out
}

fn main() {
    let slots_per_contract: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let blocks: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hot.flat");
    let mut db = FlatMpt::create(&path, Config::default()).unwrap();

    const TOKENS: u64 = 4;
    const USERS: u64 = 8_000;
    let empty_code = mpt_flat_poc::eth::EMPTY_CODE_HASH.0;

    // Genesis: users + tokens with pre-populated balances.
    let mut ops: Vec<(Key, StateOp)> = Vec::new();
    for u in 0..USERS {
        ops.push((h(&u.to_be_bytes()), StateOp::SetAccount {
            nonce: 1, balance: alloy_primitives::U256::from(1u64 << 40), code_hash: empty_code,
        }));
    }
    for t in 0..TOKENS {
        let key = h(format!("token{t}").as_bytes());
        ops.push((key, StateOp::SetAccount {
            nonce: 1, balance: alloy_primitives::U256::ZERO, code_hash: h(b"code"),
        }));
        for s in 0..slots_per_contract {
            ops.push((key, StateOp::SetStorage {
                slot: h(&(t * 1_000_000_000 + s).to_be_bytes()),
                value: mpt_flat_poc::eth::storage_value_rlp(alloy_primitives::U256::from(s + 1)),
            }));
        }
    }
    let t0 = Instant::now();
    let (root, _) = db.apply_block(ops).unwrap();
    println!("genesis: {} tokens x {} slots + {} users in {:.2}s (root {})",
        TOKENS, slots_per_contract, USERS, t0.elapsed().as_secs_f64(), mpt_flat_poc::hex(root));

    // Blocks: 18k transfers -> 2 slot writes in a random token + sender nonce bump.
    let mut rng: u64 = 0x9e3779b97f4a7c15;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    for b in 0..blocks {
        const TXS: u64 = 18_000;
        let mut ops: Vec<(Key, StateOp)> = Vec::with_capacity((TXS * 4) as usize);
        for _ in 0..TXS {
            let t = next() % TOKENS;
            let key = h(format!("token{t}").as_bytes());
            for _ in 0..2 {
                let s = next() % slots_per_contract;
                ops.push((key, StateOp::SetStorage {
                    slot: h(&(t * 1_000_000_000 + s).to_be_bytes()),
                    value: mpt_flat_poc::eth::storage_value_rlp(alloy_primitives::U256::from(next() | 1)),
                }));
            }
            let u = next() % USERS;
            ops.push((h(&u.to_be_bytes()), StateOp::SetAccount {
                nonce: b + 2, balance: alloy_primitives::U256::from(1u64 << 40), code_hash: empty_code,
            }));
        }
        // Canonicalize like the tempo integration does.
        ops.sort_by(|a, b| a.0.cmp(&b.0));
        let n = ops.len();
        let t0 = Instant::now();
        let (_root, _inv) = db.apply_block(ops).unwrap();
        println!("block {b}: {n} ops in {} ms", t0.elapsed().as_millis());
    }
}
