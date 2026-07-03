//! Small-scale repro of the nested-load pipeline: RAM-build insert_batch_accounts
//! in sorted contiguous batches (like the TSV loader), persist (spill), reopen,
//! then read everything back and apply update blocks. Run with:
//!   MPT_RAM_BUILD=1 MPT_RAM_BUILD_GIB=40 cargo run --release --example reprospill
use alloy_primitives::U256;
use mpt_flat_poc::{AccountSeed, Config, FlatMpt, Key, StateOp, eth, hex};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("repro.flat");
    let cfg = Config { target_leaf_bytes: 8 * 1024, max_leaf_bytes: 16 * 1024, min_promote_bytes: 8 * 1024 };
    let mut db = FlatMpt::create(&path, cfg).unwrap();
    let mut rng = StdRng::seed_from_u64(0x0DD);

    // Synthesize accounts; sort globally by key; feed in contiguous batches.
    let n = 10_000usize;
    let mut entries: Vec<(Key, AccountSeed)> = Vec::new();
    let mut oracle_accts: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for i in 0..n {
        let mut k = [0u8; 32];
        rng.fill(&mut k);
        let n_slots = match i % 1000 {
            0..=49 => rng.gen_range(1..40),    // small contracts
            50..=59 => rng.gen_range(200..1200), // promoting contracts
            60 => 400_000,                     // monster: multi-level recursive promotion
            _ => 0,
        };
        let mut slots = Vec::new();
        let mut sm = std::collections::BTreeMap::new();
        for _ in 0..n_slots {
            let mut sk = [0u8; 32];
            rng.fill(&mut sk);
            let v = eth::storage_value_rlp(U256::from(rng.r#gen::<u64>()));
            sm.insert(sk.to_vec(), v.clone());
            slots.push((sk, v));
        }
        let storage_root = if sm.is_empty() {
            eth::EMPTY_ROOT
        } else {
            let se: Vec<(Vec<u8>, Vec<u8>)> = sm.into_iter().collect();
            eth::root(&se)
        };
        let acct = eth::Account {
            nonce: i as u64,
            balance: U256::from(i as u64 * 3),
            storage_root,
            code_hash: alloy_primitives::B256::from([9u8; 32]),
        };
        oracle_accts.push((k.to_vec(), acct.rlp()));
        entries.push((k, AccountSeed { nonce: i as u64, balance: U256::from(i as u64 * 3), code_hash: [9u8; 32], slots }));
    }
    entries.sort_by_key(|(k, _)| *k);
    let all: Vec<(Key, Vec<(Key, Vec<u8>)>)> =
        entries.iter().map(|(k, s)| (*k, s.slots.clone())).collect();

    for chunk in entries.chunks(5000) {
        db.insert_batch_accounts(chunk.to_vec()).unwrap();
    }
    let want = eth::root(&oracle_accts).0;
    assert_eq!(db.root(), want, "root after sorted batched load");
    println!("load ok, root {}", hex(want));

    db.persist().unwrap();
    drop(db);
    let mut db = FlatMpt::open(&path).unwrap();
    assert_eq!(db.root(), want, "root after reopen");
    println!("reopen ok");

    // Read back every account and slot.
    for (i, (k, slots)) in all.iter().enumerate() {
        match db.get_value(k) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("account {i} missing: {}", hex(*k)),
            Err(e) => panic!("account {i} read error: {e} ({})", hex(*k)),
        }
        for (sk, v) in slots {
            match db.get_storage(k, sk) {
                Ok(Some(got)) => assert_eq!(&got, v, "slot value mismatch"),
                Ok(None) => panic!("slot missing: acct {} slot {}", hex(*k), hex(*sk)),
                Err(e) => panic!("slot read error: {e}"),
            }
        }
        if i % 5000 == 0 {
            println!("  read {i}");
        }
    }
    println!("reads ok");

    // Update blocks on the reopened checkpoint.
    for b in 0..5 {
        let mut ops: Vec<(Key, StateOp)> = Vec::new();
        for _ in 0..2000 {
            let (k, slots) = &all[rng.gen_range(0..all.len())];
            if slots.is_empty() || rng.gen_bool(0.3) {
                ops.push((*k, StateOp::SetAccount {
                    nonce: rng.r#gen::<u32>() as u64,
                    balance: U256::from(rng.r#gen::<u64>()),
                    code_hash: [9u8; 32],
                }));
            } else {
                let (sk, _) = &slots[rng.gen_range(0..slots.len())];
                ops.push((*k, StateOp::SetStorage {
                    slot: *sk,
                    value: eth::storage_value_rlp(U256::from(rng.r#gen::<u64>())),
                }));
            }
        }
        let (_root, _inv) = db.apply_block(ops).unwrap();
        println!("block {b} ok");
    }
    println!("ALL OK");
}
