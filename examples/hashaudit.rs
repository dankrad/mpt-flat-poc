//! Verifies whether the remaining hashing is *strictly essential*: i.e. whether
//! any keccak call recomputes a digest that was already computed (and therefore
//! could have been cached). keccak is collision-resistant, so equal outputs
//! imply equal inputs — recomputing an output already seen is, by definition,
//! avoidable work.
//!
//!     cargo run --release --example hashaudit --features profiling

use mpt_flat_poc::prof;
use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key};
use std::collections::HashSet;
use tempfile::NamedTempFile;

const WARM: u64 = 600;

fn key(i: u64) -> Key {
    hashed_key(i.to_le_bytes())
}

// Unique value per key so leaf-value hashes never collide by accident.
fn val(i: u64) -> Vec<u8> {
    let mut v = vec![0u8; 32];
    v[..8].copy_from_slice(&i.to_le_bytes());
    v
}

fn audit_one(db: &mut FlatMpt, ever: &HashSet<Key>, k: Key, v: Vec<u8>) {
    prof::audit_start();
    db.insert(k, v).unwrap();
    let calls = prof::audit_take();

    let mut seen_now = HashSet::new();
    let (mut essential, mut recomputed_unchanged, mut intra_dup) = (0u64, 0u64, 0u64);
    for h in &calls {
        if !seen_now.insert(*h) {
            intra_dup += 1; // already produced earlier in *this same* insert
        } else if ever.contains(h) {
            recomputed_unchanged += 1; // produced in an earlier insert; node unchanged
        } else {
            essential += 1; // genuinely new digest
        }
    }
    let total = calls.len() as u64;
    let redundant = recomputed_unchanged + intra_dup;
    println!("  total keccak calls       : {total}");
    println!("  essential (new digests)  : {essential}");
    println!("  recomputed-unchanged     : {recomputed_unchanged}");
    println!("  intra-insert duplicates  : {intra_dup}");
    println!(
        "  => redundant             : {redundant} / {total}  ({:.0}%)",
        redundant as f64 * 100.0 / total.max(1) as f64,
    );
}

fn main() {
    if !prof::ENABLED {
        eprintln!("build without `profiling`; rerun: cargo run --release --example hashaudit --features profiling");
        return;
    }

    let cfg = Config {
        target_leaf_bytes: 4 * 1024,
        max_leaf_bytes: 8 * 1024,
        min_promote_bytes: 2 * 1024,
    };
    let mut db = FlatMpt::create(NamedTempFile::new().unwrap().path(), cfg).unwrap();

    // Warm up, accumulating every digest ever computed.
    let mut ever: HashSet<Key> = HashSet::new();
    for i in 0..WARM {
        prof::audit_start();
        db.insert(key(i), val(i)).unwrap();
        for h in prof::audit_take() {
            ever.insert(h);
        }
    }

    println!("=== single steady-state insert (brand-new key into existing leaf) ===");
    audit_one(&mut db, &ever, key(1_000_000), val(1_000_000));

    println!("\n=== overwrite of an existing key (same key, new value) ===");
    let snapshot_ever = ever.clone();
    audit_one(&mut db, &snapshot_ever, key(42), val(999_999));
}
