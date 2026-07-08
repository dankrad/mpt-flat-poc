//! Point-read one storage slot + the owning account's RLP (forensics).
//!   probeslot <flat> <acct-key> <slot-key>
use mpt_flat_poc::{hex, FlatMpt};
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: probeslot <flat> <acct> <slot>");
    let mut a = [0u8; 32];
    let mut s = [0u8; 32];
    alloy_primitives::hex::decode_to_slice(args.next().unwrap().trim_start_matches("0x"), &mut a)?;
    alloy_primitives::hex::decode_to_slice(args.next().unwrap().trim_start_matches("0x"), &mut s)?;
    let mpt = FlatMpt::open(&flat)?;
    match mpt.get_value(&a)? {
        Some(v) => println!("acct PRESENT rlp={}", hex_bytes(&v)),
        None => println!("acct ABSENT"),
    }
    match mpt.get_storage(&a, &s)? {
        Some(v) => println!("slot PRESENT val={}", hex_bytes(&v)),
        None => println!("slot ABSENT/opaque"),
    }
    let _ = hex([0u8;32]);
    Ok(())
}
fn hex_bytes(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
