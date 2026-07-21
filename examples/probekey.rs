//! Point-read account keys from a flat file (forensics).
use mpt_flat_poc::{hex, FlatMpt};
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: probekey <flat> <key>...");
    let mpt = FlatMpt::open(&flat)?;
    for k in args {
        let mut key = [0u8; 32];
        alloy_primitives::hex::decode_to_slice(k.trim_start_matches("0x"), &mut key)?;
        match mpt.get_value(&key)? {
            Some(v) => println!("0x{} PRESENT rlp_len={}", hex(key), v.len()),
            None => println!("0x{} ABSENT", hex(key)),
        }
    }
    Ok(())
}
