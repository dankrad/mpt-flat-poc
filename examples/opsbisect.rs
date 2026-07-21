//! Delta-debug a dropped op: find the minimal op subset of a corpus segment
//! under which a target account's SetAccount fails to take effect.
//! In-place trials: apply subset -> probe target -> apply inverse to unwind.
//!
//!   opsbisect <ckpt.flat> <corpus> <block> <target-key-hex>

use mpt_flat_poc::{FlatMpt, Key, StateOp, hex};
use std::io::BufReader;

#[derive(serde::Serialize, serde::Deserialize)]
struct CorpusBlock {
    block: u64,
    gas_used: u64,
    ops: Vec<(Key, StateOp)>,
}

fn probe(db: &FlatMpt, key: &Key) -> Option<Vec<u8>> {
    db.get_value(key).unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: opsbisect <flat> <corpus> <block> <key>");
    let corpus = args.next().unwrap();
    let block: u64 = args.next().unwrap().parse().unwrap();
    let keyhex = args.next().unwrap();
    let mut key = [0u8; 32];
    key.copy_from_slice(&alloy_primitives::hex::decode(keyhex.trim_start_matches("0x")).unwrap());

    let mut rd = BufReader::new(std::fs::File::open(&corpus).unwrap());
    let ops: Vec<(Key, StateOp)> = loop {
        let cb: CorpusBlock = bincode::deserialize_from(&mut rd).expect("block not in corpus");
        if cb.block == block {
            break cb.ops;
        }
    };
    let target_idx = ops.iter().position(|(k, _)| *k == key).expect("target op not in segment");
    eprintln!("segment ops: {}, target at {}", ops.len(), target_idx);

    let mut db = FlatMpt::open(&flat).unwrap();
    let before = probe(&db, &key);
    eprintln!("target value before: {:?}", before.as_ref().map(|v| v.len()));

    // A trial applies `subset` (always containing the target op), probes, unwinds.
    let mut trial = |subset: Vec<(Key, StateOp)>, db: &mut FlatMpt| -> bool {
        let (_root, inverse) = db.apply_block(subset).unwrap();
        let after = probe(db, &key);
        let took = after != before;
        let (_r2, _inv2) = db.apply_block(inverse.clone()).unwrap();
        assert_eq!(probe(db, &key), before, "unwind failed");
        took
    };

    // Sanity: target alone.
    let solo = vec![ops[target_idx].clone()];
    let solo_ok = trial(solo, &mut db);
    eprintln!("solo apply takes effect: {solo_ok}");

    // Full set (expected to FAIL to take effect).
    let full_ok = trial(ops.clone(), &mut db);
    eprintln!("full-set apply takes effect: {full_ok}");
    if full_ok {
        eprintln!("cannot reproduce with full set — different mechanism");
        return;
    }

    // Delta-debug the co-op set: keep target always; halve the rest while the
    // failure (target NOT taking effect) persists.
    let mut rest: Vec<(Key, StateOp)> = ops
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx)
        .map(|(_, op)| op.clone())
        .collect();
    let target_op = ops[target_idx].clone();
    while rest.len() > 1 {
        let mid = rest.len() / 2;
        let (a, b) = rest.split_at(mid);
        let mut with_a: Vec<_> = a.to_vec();
        with_a.push(target_op.clone());
        let mut with_b: Vec<_> = b.to_vec();
        with_b.push(target_op.clone());
        if !trial(with_a, &mut db) {
            rest = a.to_vec();
        } else if !trial(with_b, &mut db) {
            rest = b.to_vec();
        } else {
            eprintln!("failure needs ops from BOTH halves at {} — stopping with current set", rest.len());
            break;
        }
        eprintln!("narrowed to {} co-ops", rest.len());
    }
    for (k, op) in rest.iter().take(8) {
        eprintln!("co-op: {} {:?}", hex(*k), std::mem::discriminant(op));
    }
    let (k, op) = &rest[0];
    eprintln!("first co-op detail: key={} op={:?}", hex(*k), op);
}
