//! Apply a dumped ops JSON (from the exex's divergence forensics) to a flat
//! checkpoint and print every touched account's resulting leaf fields, for
//! comparison against eth_getProof truth.
//! usage: applydump <flat> <ops.json>
use mpt_flat_poc::{FlatMpt, Key, StateOp};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: applydump <flat> <ops.json>");
    let opsf = args.next().expect("usage: applydump <flat> <ops.json>");
    let ops: Vec<(Key, StateOp)> = serde_json::from_reader(std::fs::File::open(&opsf)?)?;
    let mut db = FlatMpt::open(&flat)?;
    eprintln!("pre-root  0x{}", mpt_flat_poc::hex(db.root()));
    let (root, _inv) = db.apply_block(ops.clone()).map_err(|e| anyhow::anyhow!("{e:#}"))?;
    eprintln!("post-root 0x{}", mpt_flat_poc::hex(root));
    let mut keys: Vec<Key> = ops.iter().map(|(k, _)| *k).collect();
    keys.sort_unstable();
    keys.dedup();
    for k in keys {
        match db.get_value(&k).map_err(|e| anyhow::anyhow!("{e:#}"))? {
            None => println!("0x{}\tABSENT", mpt_flat_poc::hex(k)),
            Some(rlp) => {
                // RLP([nonce, balance, storage_root, code_hash])
                let mut buf = rlp.as_slice();
                let h = alloy_rlp::Header::decode(&mut buf)?;
                anyhow::ensure!(h.list, "not a list");
                let nonce = <u64 as alloy_rlp::Decodable>::decode(&mut buf)?;
                let balance = <alloy_primitives::U256 as alloy_rlp::Decodable>::decode(&mut buf)?;
                let sroot = <alloy_primitives::B256 as alloy_rlp::Decodable>::decode(&mut buf)?;
                let ch = <alloy_primitives::B256 as alloy_rlp::Decodable>::decode(&mut buf)?;
                println!("0x{}\t{nonce}\t{balance:#x}\t{sroot:?}\t{ch:?}", mpt_flat_poc::hex(k));
            }
        }
    }
    Ok(())
}
