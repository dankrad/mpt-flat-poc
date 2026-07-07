//! Forensic hash audit of a persisted checkpoint: reopen it, recompute every
//! hash bottom-up from raw bytes (ignoring all cached layers), and report which
//! cached layer disagrees. Read-only — safe to run on a precious checkpoint.
//!
//!   cargo run --release --example rootaudit -- /mnt2/tip-25374199.flat [expectedRoot]

use mpt_flat_poc::{FlatMpt, hex};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: rootaudit <flat> [expectedRoot]");
    let want: Option<[u8; 32]> = args.next().map(|s| {
        alloy_primitives::hex::decode(s.trim_start_matches("0x"))
            .unwrap()
            .try_into()
            .unwrap()
    });

    let t = Instant::now();
    let db = FlatMpt::open(&path).unwrap();
    eprintln!("[{:>6.1}s] opened {path}", t.elapsed().as_secs_f64());
    let cached_root = db.root();
    eprintln!("[{:>6.1}s] cached root  = {}", t.elapsed().as_secs_f64(), hex(cached_root));

    let audit = db.audit_hashes().unwrap();
    eprintln!("[{:>6.1}s] audit done", t.elapsed().as_secs_f64());

    println!("true root    = {}", hex(audit.true_root));
    println!("cached root  = {}   ({})", hex(cached_root), if cached_root == audit.true_root { "matches recompute" } else { "STALE" });
    if let Some(w) = want {
        println!("expected     = {}   ({})", hex(w), if audit.true_root == w { "true root MATCHES expected" } else { "true root differs from expected" });
    }
    println!(
        "\nfrontier nodes {}  stale HashCells {}\n\
         disk leaves {}  bad Disk roots {}\n\
         mem leaves {}  bad Mem roots {}\n\
         bad record prefixes {}\n\
         promoted accounts {}\n\
         bad in-record nrefs {}  bad in-record storage_roots {}",
        audit.frontier_nodes,
        audit.stale_cells,
        audit.disk_leaves,
        audit.bad_disk_roots,
        audit.mem_leaves,
        audit.bad_mem_roots,
        audit.bad_prefixes,
        audit.accounts,
        audit.bad_record_nrefs,
        audit.bad_record_storage_roots,
    );
    if !audit.samples.is_empty() {
        println!("\nsamples:");
        for s in &audit.samples {
            println!("  {s}");
        }
    }
    println!("\nverdict: {}", if audit.clean() { "all cached hashes consistent" } else { "cached-hash corruption found" });
}
