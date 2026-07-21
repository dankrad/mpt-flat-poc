//! keccak256 each stdin line (hex address), print hashed key.
use std::io::{BufRead, Write};
fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let l = line.unwrap();
        let b = alloy_primitives::hex::decode(l.trim().trim_start_matches("0x")).unwrap();
        writeln!(out, "0x{}", mpt_flat_poc::hex(alloy_primitives::keccak256(&b).0)).unwrap();
    }
}
