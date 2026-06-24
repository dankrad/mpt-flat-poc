//! Measures the flat-file footprint of the trie index (keys + value-hashes +
//! structure; the values themselves live in RocksDB, not here).
//!
//!     cargo run --release --example diskusage [N]

use mpt_flat_poc::{Config, FlatMpt, hashed_key};
use tempfile::NamedTempFile;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);

    let cfg = Config {
        target_leaf_bytes: 4 * 1024,
        max_leaf_bytes: 8 * 1024,
        min_promote_bytes: 2 * 1024,
    };
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut db = FlatMpt::create(&path, cfg).unwrap();
    for i in 0..n {
        db.insert(hashed_key(i.to_le_bytes()), vec![0u8; 32]).unwrap();
    }

    let end = db.flat_file_len();
    let free = db.free_bytes();
    let live = end - free;
    let on_disk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    println!("entries            : {n}");
    println!(
        "flat_file_len (end): {end} bytes ({:.2} MiB)",
        end as f64 / 1_048_576.0
    );
    println!("free bytes         : {free}");
    println!(
        "live (end - free)  : {live} bytes  ({:.1} B/entry)",
        live as f64 / n as f64
    );
    println!("file on disk       : {on_disk} bytes");
}
