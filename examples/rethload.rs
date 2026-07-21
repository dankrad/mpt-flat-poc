//! Reconstruct a mainnet state root from reth's exported secure-trie leaves and
//! diff it against the block's real `stateRoot`.
//!
//! Input: `accounts.tsv` (keccak(addr) \t nonce \t balance_hex \t code_hash|null)
//! and `storages.tsv` (keccak(addr) \t keccak(slot) \t value_hex), both produced by
//! `scripts/reth-export.sh` and both sorted by keccak(addr) (MDBX key order).
//!
//! We stream both files in a merge-join on keccak(addr): for each account we gather
//! its contiguous run of storage rows, compute the account's `storage_root` with the
//! `eth` encoding (`eth::root` over keccak(slot) -> RLP(value)), fold it into the
//! account RLP, and build the state trie through the fast batched value path. Then
//! `root()` must equal the block's `stateRoot`.
//!
//!   cargo run --release --example rethload -- \
//!     /mnt/accounts.tsv /mnt/storages.tsv /mnt/state.flat \
//!     0x73a5463d90927bfdb0e3e9b719cfc70e6c5516d47847cf33bb7968fd70b27397

use alloy_primitives::{B256, U256};
use mpt_flat_poc::eth::{self, Account, EMPTY_CODE_HASH, EMPTY_ROOT};
use mpt_flat_poc::{Config, FlatMpt, Key, hex, process_footprint_bytes};
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn hex32(s: &str) -> Key {
    let mut k = [0u8; 32];
    let b = alloy_primitives::hex::decode(s.trim_start_matches("0x")).unwrap();
    // left-pad shouldn't be needed for 32-byte keys, but be defensive.
    k[32 - b.len()..].copy_from_slice(&b);
    k
}

fn u256_hex(s: &str) -> U256 {
    let h = s.trim_start_matches("0x");
    if h.is_empty() { U256::ZERO } else { U256::from_str_radix(h, 16).unwrap() }
}

/// One line of `storages.tsv` -> (addr_hash, slot_hash, RLP(value)).
struct StoreRow {
    addr: Key,
    slot: Vec<u8>,
    value_rlp: Vec<u8>,
}

fn parse_store(line: &str) -> StoreRow {
    let mut it = line.split('\t');
    let addr = hex32(it.next().unwrap());
    let slot = hex32(it.next().unwrap()).to_vec();
    let value_rlp = eth::storage_value_rlp(u256_hex(it.next().unwrap()));
    StoreRow { addr, slot, value_rlp }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let accounts_path = args.next().expect("usage: rethload <accounts.tsv> <storages.tsv> <flat> <stateRoot>");
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

    // One-row lookahead into the (sorted) storage stream.
    let mut pending: Option<StoreRow> = storages.next().map(|l| parse_store(&l.unwrap()));

    let batch_n = 1_000_000usize;
    let mut batch: Vec<(Key, Vec<u8>)> = Vec::with_capacity(batch_n);
    let (mut n_acc, mut n_contract, mut n_slots, mut orphan_slots) = (0u64, 0u64, 0u64, 0u64);
    let t = Instant::now();

    for line in accounts.by_ref() {
        let line = line.unwrap();
        let mut it = line.split('\t');
        let addr = hex32(it.next().unwrap());
        let nonce: u64 = it.next().unwrap().parse().unwrap();
        let balance = u256_hex(it.next().unwrap());
        let ch = it.next().unwrap();
        let code_hash = if ch == "null" { EMPTY_CODE_HASH } else { B256::from(hex32(ch)) };

        // Skip any storage rows that sort before this account (shouldn't happen —
        // storage implies an account exists).
        while pending.as_ref().is_some_and(|r| r.addr < addr) {
            orphan_slots += 1;
            pending = storages.next().map(|l| parse_store(&l.unwrap()));
        }
        // Gather this account's contiguous storage run and hash it into a root.
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        while pending.as_ref().is_some_and(|r| r.addr == addr) {
            let r = pending.take().unwrap();
            entries.push((r.slot, r.value_rlp));
            pending = storages.next().map(|l| parse_store(&l.unwrap()));
        }
        let storage_root = if entries.is_empty() {
            EMPTY_ROOT
        } else {
            n_contract += 1;
            n_slots += entries.len() as u64;
            eth::root(&entries)
        };

        let acct = Account { nonce, balance, storage_root, code_hash };
        batch.push((addr, acct.rlp()));
        n_acc += 1;
        if batch.len() >= batch_n {
            db.insert_batch(std::mem::take(&mut batch)).unwrap();
            batch.reserve(batch_n);
        }
        if n_acc % 5_000_000 == 0 {
            use std::sync::atomic::Ordering::Relaxed;
            eprintln!(
                "[{:>5.0}s] accounts {}M  contracts {}  slots {}M  mem {:.1} GiB  \
                 flat {:.1} GiB  live {:.1} GiB  free-regions {}  gc-rate {}  \
                 gc-passes {}  gc-victims {}  gc-reloc {}",
                t.elapsed().as_secs_f64(),
                n_acc / 1_000_000,
                n_contract,
                n_slots / 1_000_000,
                process_footprint_bytes() as f64 / GIB,
                db.flat_file_len() as f64 / GIB,
                db.live_bytes() as f64 / GIB,
                db.free_regions(),
                db.gc_rate_current(),
                mpt_flat_poc::stats::GC_PASSES.load(Relaxed),
                mpt_flat_poc::stats::GC_REGIONS.load(Relaxed),
                mpt_flat_poc::stats::GC_RELOCATED.load(Relaxed),
            );
        }
    }
    if !batch.is_empty() {
        db.insert_batch(batch).unwrap();
    }
    // Any storage rows left over have no account (unexpected).
    while pending.is_some() {
        orphan_slots += 1;
        pending = storages.next().map(|l| parse_store(&l.unwrap()));
    }

    let root = db.root();
    let rr = db.ram_report();
    eprintln!(
        "\nLOADED {n_acc} accounts ({n_contract} with storage, {n_slots} slots) in {:.1}s\n\
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

    // Persist a reopenable checkpoint (writes the frontier manifest) so the
    // reconstructed state can be reopened later for experiments.
    let ps = Instant::now();
    db.persist().unwrap();
    eprintln!("persisted reopenable checkpoint in {:.1}s", ps.elapsed().as_secs_f64());
}
