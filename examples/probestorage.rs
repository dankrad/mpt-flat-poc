//! Enumerate the first few storage slots of an account via the storage cursor
//! (forensics: distinguishes structured storage from an opaque leaf).
//!   probestorage <flat> <acct-key> [n]
use mpt_flat_poc::{hex, FlatMpt};
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: probestorage <flat> <acct> [n]");
    let mut a = [0u8; 32];
    alloy_primitives::hex::decode_to_slice(args.next().unwrap().trim_start_matches("0x"), &mut a)?;
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let mpt = FlatMpt::open(&flat)?;
    let mut cur = mpt.storage_cursor(&a);
    let mut count = 0usize;
    let mut entry = cur.seek(&[0u8; 32])?;
    while let Some((sk, v)) = entry {
        println!("slot 0x{} = {}", hex(sk), v.iter().map(|x| format!("{x:02x}")).collect::<String>());
        count += 1;
        if count >= n {
            println!("... (more)");
            break;
        }
        entry = cur.next()?;
    }
    if count == 0 {
        println!("NO SLOTS (opaque or empty storage)");
    }
    Ok(())
}
