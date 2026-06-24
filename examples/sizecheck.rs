use mpt_flat_poc::{Config, FlatMpt, Key, hashed_key};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use tempfile::NamedTempFile;

const N: usize = 1000;

fn cfg() -> Config {
    Config {
        target_leaf_bytes: 4 * 1024,
        max_leaf_bytes: 8 * 1024,
        min_promote_bytes: 2 * 1024,
    }
}

fn make_db() -> FlatMpt {
    FlatMpt::create(NamedTempFile::new().unwrap().path(), cfg()).unwrap()
}

fn random_keys() -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..N)
        .map(|_| {
            let mut key = [0u8; 32];
            rng.fill_bytes(&mut key);
            key
        })
        .collect()
}

fn sequential_keys() -> Vec<Key> {
    (0..N as u64).map(|i| hashed_key(i.to_le_bytes())).collect()
}

fn shared_prefix_keys() -> Vec<Key> {
    (0..N as u16)
        .map(|i| {
            let mut key = [0u8; 32];
            key[..6].copy_from_slice(&[0xab, 0xcd, 0xef, 0x12, 0x34, 0x50]);
            key[30..].copy_from_slice(&i.to_be_bytes());
            key
        })
        .collect()
}

fn measure(title: &str, keys: &[Key]) {
    let mut db = make_db();
    for key in keys {
        db.insert(*key, vec![1; 32]).unwrap();
    }
    db.flush().unwrap();
    println!(
        "{title}: flat_file_len={} free_bytes={} ram_nodes={}",
        db.flat_file_len(),
        db.free_bytes(),
        db.ram_nodes()
    );
}

fn main() {
    measure("random", &random_keys());
    measure("sequential_hashed", &sequential_keys());
    measure("shared_prefix", &shared_prefix_keys());
}
