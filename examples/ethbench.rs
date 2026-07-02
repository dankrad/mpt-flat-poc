//! Ethereum-shaped state benchmark.
//!
//! Builds a synthetic state whose *leaf population* matches a typical Ethereum
//! distribution, then measures random overwrites and random insertions plus the
//! flat-file and RAM footprint. Parameterised by `n`:
//!
//!   * `n`        small (EOA) accounts            keccak(id) -> ~70 B account RLP
//!   * `sqrt(n)`  small contract accounts         + 1..20 KiB code (side store)
//!   * `sqrt(n)`  large contract accounts         each with U ~ uniform(sqrt(n)/10,
//!                                                 n/10) storage slots
//!   * storage slot                               keccak(id||slot) -> 1..33 B value
//!
//! CAVEATS (this is Phase-2 signal, not a mainnet-validated root):
//!   * The engine is still a flat Key->value trie (no account/storage/code model
//!     yet), so every leaf is an independent 32-byte-keyed entry. Storage slots are
//!     *scattered* (keccak(id||slot)) — the eventual nested design would pack a
//!     contract's slots together, so scattered is the pessimistic case for
//!     flat-file size and read amplification.
//!   * Values are size-representative random bytes (not RLP-validated); the root
//!     here does not correspond to any Ethereum root.
//!   * Code blobs go to a `<path>.code` side file (never hashed into the trie),
//!     modelling the future code_hash->bytecode store; its size is reported apart.
//!
//! Run with MPT_SKIP_VALUES=1 to preview the post-RocksDB-removal path (values
//! already live in the trie, so reads still work).
//!
//!   MPT_SKIP_VALUES=1 MPT_WORKERS=192 MAX_LEAF_KIB=16 \
//!     cargo run --release --example ethbench -- 1000000 /mnt/user/eth.flat

use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key, hex, process_footprint_bytes};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: f64 = 1024.0 * 1024.0;

#[derive(Clone, Copy)]
enum LeafKind {
    Eoa,
    Contract,
    Slot,
}

// Leaf-key domains (keep account keys and storage-slot keys in disjoint keyspaces).
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

// Account leaf value, sized like an RLP account: EOAs (empty storage+code) ~70 B,
// contracts (storage_root + code_hash present) ~104 B.
fn account_value(rng: &mut StdRng, is_contract: bool) -> Vec<u8> {
    rand_bytes(rng, if is_contract { 104 } else { 70 })
}
// Storage slot value: RLP(U256) is 1..33 bytes; small values are common.
fn slot_value(rng: &mut StdRng) -> Vec<u8> {
    let len = rng.gen_range(1..=33);
    rand_bytes(rng, len)
}

fn mem_gib() -> f64 {
    process_footprint_bytes() as f64 / GIB
}

fn report(db: &FlatMpt, label: &str, code_bytes: u64) {
    let flat = db.flat_file_len();
    let live = db.live_bytes();
    let free = db.free_bytes();
    let util = if flat > 0 { live as f64 / flat as f64 * 100.0 } else { 0.0 };
    let ls = db.leaf_stats();
    let rr = db.ram_report();
    eprintln!(
        "  [{label}]\n\
             flat-file:  {:.2} GiB high-water  |  live {:.2} GiB  free {:.2} GiB  util {:.1}%\n\
             leaves:     {} records, avg {} B/record\n\
             code store: {:.2} GiB\n\
             RAM index:  {:.2} GiB total (frontier {:.2} GiB, free-list {:.1} MiB)\n\
             process:    {:.2} GiB RSS",
        flat as f64 / GIB, live as f64 / GIB, free as f64 / GIB, util,
        ls.count, ls.avg_bytes(),
        code_bytes as f64 / GIB,
        rr.total_bytes() as f64 / GIB, rr.frontier_bytes as f64 / GIB, rr.free_list_bytes as f64 / MIB,
        mem_gib(),
    );
}

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let path = std::env::args().nth(2).expect("usage: ethbench <n> <path>");
    let max = std::env::var("MAX_LEAF_KIB").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize) * 1024;
    let cfg = Config { target_leaf_bytes: max / 2, max_leaf_bytes: max, min_promote_bytes: max / 2 };
    let batch: usize = std::env::var("MPT_BUILD_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let ops: u64 = std::env::var("MPT_BENCH_OPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);

    let s = (n as f64).sqrt() as u64; // small-contract count == large-contract count
    let lo = (s / 10).max(1);
    let hi = (n / 10).max(lo + 1);

    let mut rng = StdRng::seed_from_u64(0xE7_1131_u64);

    // Per-large-contract storage-slot counts + prefix sums (for weighted random pick).
    let cnt: Vec<u64> = (0..s).map(|_| rng.gen_range(lo..=hi)).collect();
    let mut cum: Vec<u64> = Vec::with_capacity(s as usize + 1);
    cum.push(0);
    for c in &cnt {
        cum.push(cum.last().unwrap() + c);
    }
    let total_storage = *cum.last().unwrap();
    let total_accounts = n + 2 * s;
    let total_leaves = total_accounts + total_storage;

    eprintln!(
        "eth-shaped state: n={n} small accounts, {s} small contracts (code 1-20 KiB), \
         {s} large contracts\n  storage slots: {total_storage} ({lo}..={hi} per large contract), \
         total leaves {total_leaves}\n  leaf cfg: {} KiB max, batch {batch}, ops {ops}",
        max / 1024,
    );

    let code_path = format!("{path}.code");
    let mut code_file = OpenOptions::new().create(true).write(true).truncate(true).open(&code_path).unwrap();
    let mut code_bytes: u64 = 0;

    let mut db = FlatMpt::create(&path, cfg).unwrap();

    // ---- Build phase: stream every leaf into insert_batch ----
    let t = Instant::now();
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::with_capacity(batch);
    let flush = |db: &mut FlatMpt, buf: &mut Vec<(Key, Vec<u8>)>, force: bool| {
        if buf.len() >= batch || (force && !buf.is_empty()) {
            db.insert_batch(std::mem::take(buf)).unwrap();
        }
    };

    // EOAs.
    for id in 0..n {
        buf.push((account_key(id), account_value(&mut rng, false)));
        flush(&mut db, &mut buf, false);
    }
    // Small contracts (+ code blobs to the side store).
    for j in 0..s {
        let id = n + j;
        buf.push((account_key(id), account_value(&mut rng, true)));
        flush(&mut db, &mut buf, false);
        let code_len = rng.gen_range(1024..=20 * 1024);
        let code = rand_bytes(&mut rng, code_len);
        code_file.write_all(&code).unwrap();
        code_bytes += code.len() as u64;
    }
    // Large contracts + their scattered storage slots.
    for j in 0..s {
        let id = n + s + j;
        buf.push((account_key(id), account_value(&mut rng, true)));
        flush(&mut db, &mut buf, false);
        for k in 0..cnt[j as usize] {
            buf.push((slot_key(id, k), slot_value(&mut rng)));
            flush(&mut db, &mut buf, false);
        }
    }
    flush(&mut db, &mut buf, true);
    code_file.flush().unwrap();
    db.flush().unwrap();
    let build = t.elapsed();
    eprintln!(
        "\nBUILD: {total_leaves} leaves in {:.1}s ({:.3} us/leaf)  root={}",
        build.as_secs_f64(),
        build.as_micros() as f64 / total_leaves as f64,
        hex(db.root()),
    );
    report(&db, "after build", code_bytes);

    // Weighted random existing key, tagged by leaf kind so the overwrite can pick a
    // value of the right size class. Weighting is by population (storage dominates).
    let pick_existing = |rng: &mut StdRng| -> (Key, LeafKind) {
        let r = rng.gen_range(0..total_leaves);
        if r < total_accounts {
            let id = r;
            let kind = if id < n { LeafKind::Eoa } else { LeafKind::Contract }; // [n, n+2s) contracts
            (account_key(id), kind)
        } else {
            let si = r - total_accounts;
            let j = cum.partition_point(|&c| c <= si) - 1; // contract owning slot si
            let k = si - cum[j];
            (slot_key(n + s + j as u64, k), LeafKind::Slot)
        }
    };

    // ---- Random overwrites: existing keys, fresh same-size-class values ----
    let flat0 = db.flat_file_len();
    let t = Instant::now();
    let wbatch = 100_000usize.min(batch);
    let mut done = 0u64;
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::with_capacity(wbatch);
    while done < ops {
        let (k, kind) = pick_existing(&mut rng);
        let v = match kind {
            LeafKind::Eoa => account_value(&mut rng, false),
            LeafKind::Contract => account_value(&mut rng, true),
            LeafKind::Slot => slot_value(&mut rng),
        };
        buf.push((k, v));
        done += 1;
        if buf.len() >= wbatch {
            db.insert_batch(std::mem::take(&mut buf)).unwrap();
        }
    }
    if !buf.is_empty() {
        db.insert_batch(buf).unwrap();
    }
    db.flush().unwrap();
    let wt = t.elapsed();
    eprintln!(
        "\nRANDOM OVERWRITES: {ops} ops in {:.1}s ({:.3} us/op)  flat +{:.2} GiB  root={}",
        wt.as_secs_f64(),
        wt.as_micros() as f64 / ops as f64,
        (db.flat_file_len().saturating_sub(flat0)) as f64 / GIB,
        hex(db.root()),
    );
    report(&db, "after overwrites", code_bytes);

    // ---- Random insertions: brand-new EOA accounts ----
    let flat1 = db.flat_file_len();
    let t = Instant::now();
    let mut next_id = total_accounts; // new account ids past the built range
    let mut done = 0u64;
    let mut buf: Vec<(Key, Vec<u8>)> = Vec::with_capacity(wbatch);
    while done < ops {
        buf.push((account_key(next_id), account_value(&mut rng, false)));
        next_id += 1;
        done += 1;
        if buf.len() >= wbatch {
            db.insert_batch(std::mem::take(&mut buf)).unwrap();
        }
    }
    if !buf.is_empty() {
        db.insert_batch(buf).unwrap();
    }
    db.flush().unwrap();
    let it = t.elapsed();
    eprintln!(
        "\nRANDOM INSERTIONS: {ops} new accounts in {:.1}s ({:.3} us/op)  flat +{:.2} GiB  root={}",
        it.as_secs_f64(),
        it.as_micros() as f64 / ops as f64,
        (db.flat_file_len().saturating_sub(flat1)) as f64 / GIB,
        hex(db.root()),
    );
    report(&db, "after insertions", code_bytes);

    let _ = std::fs::remove_file(&code_path);
}
