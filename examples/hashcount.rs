//! Counts keccak calls per individual insert into a freshly-created DB, to see
//! how the hashing cost scales. Run with:
//!     cargo run --release --example hashcount --features profiling

use mpt_flat_poc::prof::{self, Cat};
use mpt_flat_poc::{Config, FlatMpt, hashed_key};
use tempfile::NamedTempFile;

fn main() {
    let cfg = Config {
        target_leaf_bytes: 4 * 1024,
        max_leaf_bytes: 8 * 1024,
        min_promote_bytes: 2 * 1024,
    };
    let mut db = FlatMpt::create(NamedTempFile::new().unwrap().path(), cfg).unwrap();

    let mut total = 0u64;
    for i in 0..1000u64 {
        prof::reset();
        db.insert(hashed_key(i.to_le_bytes()), vec![1; 32]).unwrap();
        let keccaks = prof::snapshot()[Cat::Keccak as usize].1;
        total += keccaks;
        if i < 6 || [9, 49, 99, 499, 999].contains(&i) {
            println!("insert #{i:<4} -> {keccaks:>4} keccak calls (ram_nodes={})", db.ram_nodes());
        }
    }
    println!("\ntotal keccak calls over 1000 inserts: {total} ({:.1}/insert avg)", total as f64 / 1000.0);
}
