//! Reconstruct a mainnet state root from reth's exported secure-trie leaves with
//! storage **nested in the trie** (the full engine model — unlike `rethload`,
//! which precomputes storage roots with the oracle and stores opaque accounts).
//!
//! Streams `accounts.tsv` + `storages.tsv` (sorted by keccak(addr)) in a
//! merge-join, batches whole accounts (fields + all slots) through
//! `insert_batch_accounts` (RAM-build recommended), verifies the root against
//! the block's real `stateRoot`, and persists a reopenable checkpoint.
//!
//!   MPT_RAM_BUILD=1 MPT_RAM_BUILD_GIB=45 \
//!     cargo run --release --example rethload_nested -- \
//!     /mnt2/accounts.tsv /mnt2/storages.tsv /mnt2/recon-nested.flat <stateRoot>

use alloy_primitives::{B256, U256};
use mpt_flat_poc::eth::{self, EMPTY_CODE_HASH};
use mpt_flat_poc::{AccountSeed, Config, FlatMpt, Key, hex, process_footprint_bytes};
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn hex32(s: &str) -> Key {
    let mut k = [0u8; 32];
    let b = alloy_primitives::hex::decode(s.trim_start_matches("0x")).unwrap();
    k[32 - b.len()..].copy_from_slice(&b);
    k
}

fn u256_hex(s: &str) -> U256 {
    let h = s.trim_start_matches("0x");
    if h.is_empty() { U256::ZERO } else { U256::from_str_radix(h, 16).unwrap() }
}

struct StoreRow {
    addr: Key,
    slot: Key,
    value_rlp: Vec<u8>,
}

fn parse_store(line: &str) -> StoreRow {
    let mut it = line.split('\t');
    let addr = hex32(it.next().unwrap());
    let slot = hex32(it.next().unwrap());
    let value_rlp = eth::storage_value_rlp(u256_hex(it.next().unwrap()));
    StoreRow { addr, slot, value_rlp }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let accounts_path =
        args.next().expect("usage: rethload_nested <accounts.tsv> <storages.tsv> <flat> <stateRoot>");
    let storages_path = args.next().expect("need storages.tsv");
    let flat_path = args.next().expect("need flat path");
    let want: B256 = args.next().expect("need expected stateRoot").parse().unwrap();

    let cfg = Config {
        target_leaf_bytes: 8 * 1024,
        max_leaf_bytes: 16 * 1024,
        min_promote_bytes: 8 * 1024,
    };
    let mut db = FlatMpt::create(&flat_path, cfg).unwrap();

    let mut accounts = BufReader::new(std::fs::File::open(&accounts_path).unwrap()).lines();
    let mut storages = BufReader::new(std::fs::File::open(&storages_path).unwrap()).lines();
    let mut pending: Option<StoreRow> = storages.next().map(|l| parse_store(&l.unwrap()));

    // Batch by total ops (accounts + slots) so huge contracts don't blow RAM.
    let batch_ops = 4_000_000usize;
    let mut batch: Vec<(Key, AccountSeed)> = Vec::new();
    let mut batch_op_count = 0usize;
    let (mut n_acc, mut n_contract, mut n_slots, mut orphan_slots) = (0u64, 0u64, 0u64, 0u64);
    let t = Instant::now();

    for line in accounts.by_ref() {
        let line = line.unwrap();
        let mut it = line.split('\t');
        let addr = hex32(it.next().unwrap());
        let nonce: u64 = it.next().unwrap().parse().unwrap();
        let balance = u256_hex(it.next().unwrap());
        let ch = it.next().unwrap();
        let code_hash = if ch == "null" { EMPTY_CODE_HASH.0 } else { hex32(ch) };

        while pending.as_ref().is_some_and(|r| r.addr < addr) {
            orphan_slots += 1;
            pending = storages.next().map(|l| parse_store(&l.unwrap()));
        }
        let mut slots: Vec<(Key, Vec<u8>)> = Vec::new();
        while pending.as_ref().is_some_and(|r| r.addr == addr) {
            let r = pending.take().unwrap();
            slots.push((r.slot, r.value_rlp));
            pending = storages.next().map(|l| parse_store(&l.unwrap()));
        }
        if !slots.is_empty() {
            n_contract += 1;
            n_slots += slots.len() as u64;
        }
        batch_op_count += 1 + slots.len();
        batch.push((addr, AccountSeed { nonce, balance, code_hash, slots }));
        n_acc += 1;

        if batch_op_count >= batch_ops {
            db.insert_batch_accounts(std::mem::take(&mut batch)).unwrap();
            batch_op_count = 0;
            eprintln!(
                "[{:>5.0}s] accounts {:.1}M  contracts {:.2}M  slots {:.1}M  mem {:.1} GiB  flat {:.1} GiB",
                t.elapsed().as_secs_f64(),
                n_acc as f64 / 1e6,
                n_contract as f64 / 1e6,
                n_slots as f64 / 1e6,
                process_footprint_bytes() as f64 / GIB,
                db.flat_file_len() as f64 / GIB,
            );
        }
    }
    if !batch.is_empty() {
        db.insert_batch_accounts(batch).unwrap();
    }
    while pending.is_some() {
        orphan_slots += 1;
        pending = storages.next().map(|l| parse_store(&l.unwrap()));
    }

    let root = db.root();
    let rr = db.ram_report();
    eprintln!(
        "\nLOADED {n_acc} accounts ({n_contract} with storage, {n_slots} slots, nested) in {:.1}s\n\
         orphan storage rows: {orphan_slots}\n\
         our state_root = {}\n\
         want           = {}\n\
         MATCH: {}\n\
         flat-file: {:.2} GiB live / {:.2} GiB high-water   RAM index: {:.2} GiB   RSS {:.2} GiB",
        t.elapsed().as_secs_f64(),
        hex(root),
        hex(want.0),
        root == want.0,
        db.live_bytes() as f64 / GIB,
        db.flat_file_len() as f64 / GIB,
        rr.total_bytes() as f64 / GIB,
        process_footprint_bytes() as f64 / GIB,
    );
    std::io::stdout().flush().ok();

    let ps = Instant::now();
    db.persist().unwrap();
    eprintln!("persisted reopenable checkpoint in {:.1}s", ps.elapsed().as_secs_f64());
    assert!(root == want.0, "state root mismatch");
}
