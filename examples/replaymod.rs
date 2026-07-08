//! Forensic replay: apply batch 0 of a diff corpus (optionally filtered /
//! transformed) to a checkpoint flat, then verify every op landed via point
//! reads against a sequential model. The flat is NOT persisted, so the same
//! file can be reused across runs (the manifest still references the pristine
//! checkpoint).
//!
//!   replaymod <flat> <corpus> <mode> [K]
//!
//! modes: all | seta | nodels | nosets | onlystorage | smallgroups-K |
//!        biggroups-K | firsthalf | secondhalf | firstN-K
use mpt_flat_poc::{FlatMpt, Key, StateOp, hex};
use std::collections::HashMap;
use std::io::BufReader;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(serde::Serialize, serde::Deserialize)]
struct CorpusBlock {
    block: u64,
    gas_used: u64,
    ops: Vec<(Key, StateOp)>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: replaymod <flat> <corpus> <mode>");
    let corpus = args.next().expect("corpus");
    let mode = args.next().unwrap_or_else(|| "all".into());

    let mut rd = BufReader::new(std::fs::File::open(&corpus).unwrap());
    let cb: CorpusBlock = bincode::deserialize_from(&mut rd).unwrap();
    eprintln!("batch block {} with {} ops", cb.block, cb.ops.len());

    // Per-account slot-op counts (for group-size filters).
    let mut slot_ops: HashMap<Key, usize> = HashMap::new();
    for (k, op) in &cb.ops {
        if matches!(op, StateOp::SetStorage { .. } | StateOp::DeleteStorage { .. }) {
            *slot_ops.entry(*k).or_default() += 1;
        }
    }

    let (name, k) = match mode.rsplit_once('-') {
        Some((n, ks)) if ks.parse::<usize>().is_ok() => (n.to_string(), ks.parse::<usize>().unwrap()),
        _ => (mode.clone(), 0usize),
    };
    let total = cb.ops.len();
    let ops: Vec<(Key, StateOp)> = cb
        .ops
        .into_iter()
        .enumerate()
        .filter(|(i, (key, op))| match name.as_str() {
            "all" => true,
            "seta" => matches!(op, StateOp::SetAccount { .. }),
            "nodels" => !matches!(op, StateOp::DeleteStorage { .. }),
            "nosets" => !matches!(op, StateOp::SetStorage { .. }),
            "onlystorage" => matches!(op, StateOp::SetStorage { .. } | StateOp::DeleteStorage { .. }),
            "smallgroups" => slot_ops.get(key).copied().unwrap_or(0) <= k,
            "biggroups" => {
                matches!(op, StateOp::SetAccount { .. })
                    || slot_ops.get(key).copied().unwrap_or(0) > k
            }
            "firsthalf" => *i < total / 2,
            "secondhalf" => *i >= total / 2,
            "firstN" => *i < k,
            other => panic!("unknown mode {other}"),
        })
        .map(|(_, kv)| kv)
        .collect();
    eprintln!("mode {mode}: applying {} ops", ops.len());

    // Sequential model of expected point-read outcomes.
    let mut acct: HashMap<Key, bool> = HashMap::new();
    let mut slot: HashMap<(Key, Key), Option<Vec<u8>>> = HashMap::new();
    for (key, op) in &ops {
        match op {
            StateOp::SetAccount { .. } => {
                acct.insert(*key, true);
            }
            StateOp::DeleteAccount => {
                acct.insert(*key, false);
                for ((a, _), v) in slot.iter_mut() {
                    if a == key {
                        *v = None;
                    }
                }
            }
            StateOp::WipeStorage => {
                for ((a, _), v) in slot.iter_mut() {
                    if a == key {
                        *v = None;
                    }
                }
            }
            StateOp::SetStorage { slot: sk, value } => {
                acct.insert(*key, true);
                slot.insert((*key, *sk), Some(value.clone()));
            }
            StateOp::DeleteStorage { slot: sk } => {
                // Regardless of prior account existence, post-state read is None.
                slot.insert((*key, *sk), None);
            }
        }
    }

    let mut db = FlatMpt::open(&flat).unwrap();
    let root0 = db.root();
    eprintln!("checkpoint root {}", hex(root0));
    let t = std::time::Instant::now();
    let (root, inv) = db.apply_block(ops).unwrap();
    eprintln!("applied in {:.1}s root {}", t.elapsed().as_secs_f64(), hex(root));

    let mut wrong_acct = 0usize;
    let mut wrong_slot = 0usize;
    let mut samples: Vec<String> = Vec::new();
    for (key, want) in &acct {
        let got = db.get_value(key).unwrap().is_some();
        if got != *want {
            wrong_acct += 1;
            if samples.len() < 5 {
                samples.push(format!("acct 0x{} want_present={want} got={got}", hex(*key)));
            }
        }
    }
    for ((a, sk), want) in &slot {
        let want = if acct.get(a) == Some(&false) { &None } else { want };
        let got = db.get_storage(a, sk).unwrap();
        let ok = match want {
            Some(v) => got.as_deref() == Some(v.as_slice()),
            None => got.is_none(),
        };
        if !ok {
            wrong_slot += 1;
            if samples.len() < 10 {
                samples.push(format!("slot 0x{} / 0x{}", hex(*a), hex(*sk)));
            }
        }
    }
    for s in &samples {
        println!("WRONG {s}");
    }
    // Inverse round-trip: applying the inverse diff must restore the exact
    // checkpoint root. A sibling harmed by the forward apply (content the
    // batch never touched) is not covered by the inverse, so the root won't
    // come back — a detector for silent neighbor damage.
    let t = std::time::Instant::now();
    let (back, _) = db.apply_block(inv).unwrap();
    eprintln!(
        "inverse applied in {:.1}s root {} ({})",
        t.elapsed().as_secs_f64(),
        hex(back),
        if back == root0 { "RESTORED" } else { "NOT RESTORED" }
    );
    println!(
        "mode {mode}: checked {} accts ({wrong_acct} wrong), {} slots ({wrong_slot} wrong), root {} inverse={}",
        acct.len(),
        slot.len(),
        hex(root),
        if back == root0 { "restored" } else { "NOT-restored" }
    );
}
