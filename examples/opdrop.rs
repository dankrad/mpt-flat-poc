//! Synthetic op-drop reproducer shaped like the mainnet fused batch: a few
//! contracts receive tens of thousands of slot ops each (mass SETS, mass DELS,
//! delete-then-readd churn), plus a few hundred thousand plain SetAccounts.
//! ONE `apply_block`, then verify EVERY touched key against a sequential model
//! via `get_value`/`get_storage`.
//!
//!   cargo run --release --example opdrop -- <flat> <seed-accounts> <n-ops> [rng-seed]
//!
//! Exits nonzero (with counts) if the flat diverges from the model.

use mpt_flat_poc::{AccountSeed, Config, FlatMpt, Key, StateOp, hex};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn rand_key(rng: &mut StdRng) -> Key {
    let mut k = [0u8; 32];
    rng.fill(&mut k);
    k
}

/// Model of the touched slice of state, with engine-equivalent op semantics.
#[derive(Default)]
struct Model {
    acct: HashMap<Key, bool>,
    slot: HashMap<(Key, Key), Option<Vec<u8>>>,
}

impl Model {
    fn apply(&mut self, key: &Key, op: &StateOp, existed: bool) {
        match op {
            StateOp::SetAccount { .. } => {
                self.acct.insert(*key, true);
            }
            StateOp::DeleteAccount => {
                self.acct.insert(*key, false);
                for ((a, _), v) in self.slot.iter_mut() {
                    if a == key {
                        *v = None;
                    }
                }
            }
            StateOp::WipeStorage => {
                let present = *self.acct.entry(*key).or_insert(existed);
                if present {
                    for ((a, _), v) in self.slot.iter_mut() {
                        if a == key {
                            *v = None;
                        }
                    }
                }
            }
            StateOp::SetStorage { slot, value } => {
                self.acct.insert(*key, true);
                self.slot.insert((*key, *slot), Some(value.clone()));
            }
            StateOp::DeleteStorage { slot } => {
                let present = *self.acct.entry(*key).or_insert(existed);
                if present {
                    self.slot.insert((*key, *slot), None);
                }
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: opdrop <flat> <seed-accounts> <n-ops> [rng-seed]");
    let m: usize = args.next().expect("seed-accounts").parse().unwrap();
    let n: usize = args.next().expect("n-ops").parse().unwrap();
    let rng_seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0xD20F);
    let mut rng = StdRng::seed_from_u64(rng_seed);

    // ---- Seed M accounts via the normal bulk path.
    //   first 200 (spread through the keyspace by construction — keys are
    //   random): big contracts with 3000 slots each (promoted at default cfg)
    //   every 10th otherwise: small contract (4 slots)
    const NBIG: usize = 200;
    const BIG_SLOTS: usize = 3000;
    let t0 = std::time::Instant::now();
    let mut db = FlatMpt::create(&flat, Config::default()).unwrap();
    let mut seeded: Vec<Key> = Vec::with_capacity(m);
    let mut big: Vec<Key> = Vec::new();
    let mut big_slots: HashMap<Key, Vec<Key>> = HashMap::new();
    let mut small_slots: Vec<(Key, Key)> = Vec::new();
    let mut entries: Vec<(Key, AccountSeed)> = Vec::with_capacity(m);
    for i in 0..m {
        let k = rand_key(&mut rng);
        seeded.push(k);
        let n_slots = if i < NBIG { BIG_SLOTS } else if i % 10 == 0 { 4 } else { 0 };
        if n_slots == BIG_SLOTS {
            big.push(k);
        }
        let mut slots: Vec<(Key, Vec<u8>)> = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            let sk = rand_key(&mut rng);
            slots.push((sk, vec![0xAB, (i & 0xff) as u8, 1]));
            if n_slots == BIG_SLOTS {
                big_slots.entry(k).or_default().push(sk);
            } else {
                small_slots.push((k, sk));
            }
        }
        entries.push((
            k,
            AccountSeed {
                nonce: i as u64,
                balance: alloy_primitives::U256::from(i as u64 + 1),
                code_hash: [0u8; 32],
                slots,
            },
        ));
    }
    db.insert_batch_accounts(entries).unwrap();
    db.persist().unwrap();
    eprintln!(
        "seeded {m} accounts ({} big x {BIG_SLOTS} slots) in {:.1}s root {}",
        big.len(),
        t0.elapsed().as_secs_f64(),
        hex(db.root())
    );

    // ---- Build ONE block of ~N mixed ops, mainnet-fused-batch shaped:
    //   hot contracts:   big[0..20]  each gets 15k fresh SETS + overwrite of all
    //                                seeded slots
    //   mass-delete:     big[20..40] each gets DELS on ALL seeded slots
    //   churn:           big[40..60] DELS all seeded slots, then re-add 1/3 with
    //                                new values, then 5k fresh SETS
    //   the rest of N:   55% SETA new key, 20% SETA existing, 20% SETS random
    //                    seeded account fresh slot, 5% DELS of a seeded small
    //                    slot; a sprinkle of DELA
    let mut ops: Vec<(Key, StateOp)> = Vec::with_capacity(n + 800_000);
    for (ci, &a) in big.iter().take(20).enumerate() {
        for j in 0..15_000usize {
            let sk = rand_key(&mut rng);
            ops.push((a, StateOp::SetStorage { slot: sk, value: vec![0x11, ci as u8, (j & 0xff) as u8, 1] }));
        }
        for (j, sk) in big_slots[&a].iter().enumerate() {
            ops.push((a, StateOp::SetStorage { slot: *sk, value: vec![0x22, ci as u8, (j & 0xff) as u8, 1] }));
        }
    }
    for &a in big.iter().skip(20).take(20) {
        for sk in &big_slots[&a] {
            ops.push((a, StateOp::DeleteStorage { slot: *sk }));
        }
    }
    for (ci, &a) in big.iter().skip(40).take(20).enumerate() {
        for sk in &big_slots[&a] {
            ops.push((a, StateOp::DeleteStorage { slot: *sk }));
        }
        for (j, sk) in big_slots[&a].iter().enumerate().filter(|(j, _)| j % 3 == 0) {
            ops.push((a, StateOp::SetStorage { slot: *sk, value: vec![0x33, ci as u8, (j & 0xff) as u8, 1] }));
        }
        for j in 0..5_000usize {
            let sk = rand_key(&mut rng);
            ops.push((a, StateOp::SetStorage { slot: sk, value: vec![0x44, ci as u8, (j & 0xff) as u8, 1] }));
        }
    }
    let structured = ops.len();
    let mut slot_cursor = 0usize;
    let mut dela = 0usize;
    for i in 0..n {
        match i % 20 {
            0..=10 => {
                let k = rand_key(&mut rng);
                ops.push((
                    k,
                    StateOp::SetAccount {
                        nonce: i as u64,
                        balance: alloy_primitives::U256::from(7u64),
                        code_hash: [0u8; 32],
                    },
                ));
            }
            11..=14 => {
                // Update an existing plain account (skip the big contracts).
                let a = seeded[rng.gen_range(NBIG..seeded.len())];
                ops.push((
                    a,
                    StateOp::SetAccount {
                        nonce: 7777,
                        balance: alloy_primitives::U256::from(15u64),
                        code_hash: [0u8; 32],
                    },
                ));
            }
            15..=18 => {
                let a = seeded[rng.gen_range(NBIG..seeded.len())];
                let sk = rand_key(&mut rng);
                ops.push((a, StateOp::SetStorage { slot: sk, value: vec![0xCD, (i & 0xff) as u8, 1] }));
            }
            _ => {
                if dela < 40 && i % 400 == 19 {
                    dela += 1;
                    let a = seeded[rng.gen_range(NBIG..seeded.len())];
                    ops.push((a, StateOp::DeleteAccount));
                } else if slot_cursor < small_slots.len() {
                    let (a, sk) = small_slots[slot_cursor];
                    slot_cursor += 1;
                    ops.push((a, StateOp::DeleteStorage { slot: sk }));
                }
            }
        }
    }
    eprintln!("ops: {} structured (hot contracts) + {} scattered", structured, ops.len() - structured);

    // ---- Sequential model of the touched slice (engine semantics).
    let seeded_set: std::collections::HashSet<Key> = seeded.iter().copied().collect();
    let mut model = Model::default();
    for (k, op) in &ops {
        let existed = seeded_set.contains(k);
        model.apply(k, op, existed);
    }

    let t1 = std::time::Instant::now();
    let n_ops = ops.len();
    let (root, _inv) = db.apply_block(ops).unwrap();
    eprintln!("applied {n_ops} ops in {:.1}s root {}", t1.elapsed().as_secs_f64(), hex(root));

    // ---- Verify the touched slice against the model.
    let mut wrong_acct = 0usize;
    let mut wrong_slot = 0usize;
    let mut first_wrong: Option<Key> = None;
    let mut first_wrong_slot: Option<(Key, Key)> = None;
    for (k, want) in &model.acct {
        let got = db.get_value(k).unwrap().is_some();
        if got != *want {
            wrong_acct += 1;
            if first_wrong.is_none() {
                first_wrong = Some(*k);
            }
        }
    }
    for ((a, sk), want) in &model.slot {
        let want = if model.acct.get(a) == Some(&false) { &None } else { want };
        let got = db.get_storage(a, sk).unwrap();
        let ok = match want {
            Some(v) => got.as_deref() == Some(v.as_slice()),
            None => got.is_none(),
        };
        if !ok {
            wrong_slot += 1;
            if first_wrong_slot.is_none() {
                first_wrong_slot = Some((*a, *sk));
            }
        }
    }
    println!(
        "checked {} accounts ({wrong_acct} wrong), {} slots ({wrong_slot} wrong)",
        model.acct.len(),
        model.slot.len()
    );
    if let Some(k) = first_wrong {
        println!("first wrong account key: 0x{}", hex(k));
    }
    if let Some((a, sk)) = first_wrong_slot {
        println!("first wrong slot: acct 0x{} slot 0x{}", hex(a), hex(sk));
    }
    if wrong_acct + wrong_slot > 0 {
        std::process::exit(1);
    }
}
