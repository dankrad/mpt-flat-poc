//! Reopen a persisted database and verify it round-trips.
//!
//!     cargo run --release --example reopen -- /path/to/db.flat
//!
//! Prints the root, frontier size, and flat-file length, and spot-checks a few
//! keys built by the `large` bench (`hashed_key(i.to_le_bytes())`).

use mpt_flat_poc::{FlatMpt, hashed_key, hex};

fn main() {
    let path = std::env::args().nth(1).expect("usage: reopen <db.flat>");
    let db = FlatMpt::open(&path).expect("open checkpoint");
    println!(
        "reopened {path}\n  root={}\n  ram_nodes={}  flat={:.1} MiB",
        hex(db.root()),
        db.ram_nodes(),
        db.flat_file_len() as f64 / (1024.0 * 1024.0),
    );
    // Spot-check a handful of large-bench keys are present.
    let mut present = 0;
    for i in [0u64, 1, 1000, 1_000_000, 1_849_999] {
        if db.get_value(&hashed_key(i.to_le_bytes())).unwrap().is_some() {
            present += 1;
        }
    }
    println!("  spot-check: {present}/5 sampled keys present");
}
