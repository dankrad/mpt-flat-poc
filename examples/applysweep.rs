//! Apply-size sweep on a 1B flat with phase profiling: decomposes where
//! apply_block time goes as block size grows (the follower-superlinearity
//! question), and what share is keccak (the hash-transplant premise).
//! usage: applysweep <flat> (build with --features profiling)
use mpt_flat_poc::{prof, FlatMpt, Key, StateOp};

fn kec(s: &str) -> Key {
    alloy_primitives::keccak256(s.as_bytes()).0
}

fn main() -> anyhow::Result<()> {
    assert!(prof::ENABLED, "build with --features profiling");
    let flat = std::env::args().nth(1).expect("usage: applysweep <flat>");
    let mut db = FlatMpt::open(&flat)?;
    let contract = kec("bench-contract-anchor"); // synthetic; ops target random keys anyway

    for (round, n_ops) in [(0usize, 16_000usize), (1, 33_000), (2, 66_000), (3, 100_000), (4, 147_000), (5, 66_000)] {
        let n_acct = n_ops / 4;
        let mut ops: Vec<(Key, StateOp)> = Vec::with_capacity(n_ops);
        for i in 0..n_acct {
            let a = kec(&format!("sw-{round}-acct-{i}"));
            ops.push((a, StateOp::SetAccount {
                nonce: 1,
                balance: alloy_primitives::U256::from(1000u64 + i as u64),
                code_hash: [0u8; 32],
            }));
            for j in 0..3 {
                ops.push((contract, StateOp::SetStorage {
                    slot: kec(&format!("sw-{round}-slot-{i}-{j}")),
                    value: vec![0x82, (i & 0xff) as u8, j as u8],
                }));
            }
        }
        ops.truncate(n_ops);
        ops.sort_by(|a, b| a.0.cmp(&b.0));

        prof::reset();
        let t = std::time::Instant::now();
        let (_root, _inv) = db.apply_block(ops).map_err(|e| anyhow::anyhow!("{e:#}"))?;
        let wall_ms = t.elapsed().as_millis();
        let snap = prof::snapshot();
        let phases: Vec<String> = snap
            .iter()
            .enumerate()
            .filter(|(_, (nanos, _))| *nanos > 0)
            .map(|(i, (nanos, count))| {
                format!("{}={:.0}ms/{}", prof::CATS[i], *nanos as f64 / 1e6, count)
            })
            .collect();
        println!(
            "n_ops={n_ops:7} wall={wall_ms:6}ms  ({:.0} ops/ms)  {}",
            n_ops as f64 / wall_ms.max(1) as f64,
            phases.join(" ")
        );
    }
    Ok(())
}
