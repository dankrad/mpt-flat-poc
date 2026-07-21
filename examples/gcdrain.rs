//! Drain-mode GC: collect/install cycles until no victim qualifies.
//! Validates relocation end-to-end on a real file (path-asserting under
//! MPT_GC_ASSERT_PATHS=1).
//!
//!   cargo run --release --example gcdrain -- <flat> [chunk]

use mpt_flat_poc::FlatMpt;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("flat path");
    let chunk: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(2048);
    let mut db = FlatMpt::open(&path)?;
    let t0 = Instant::now();
    let (mut cycles, mut installed, mut discarded) = (0u64, 0u64, 0u64);
    let start_len = db.flat_file_len();
    loop {
        let t = Instant::now();
        let batch = db.snapshot().gc_collect(chunk)?;
        if batch.is_empty() {
            break;
        }
        let (regions, items) = (batch.regions(), batch.len());
        let collect_ms = t.elapsed().as_millis();
        let t = Instant::now();
        let (ins, dis) = db.gc_install(batch)?;
        cycles += 1;
        installed += ins as u64;
        discarded += dis as u64;
        db.prefetch_clear(); // release staged regions at the cycle boundary
        if cycles % 20 == 0 {
            eprintln!(
                "cycle {cycles}: regions={regions} items={items} installed={ins} collect_ms={collect_ms} install_ms={} util={:.3} free_regions={}",
                t.elapsed().as_millis(),
                db.utilization(),
                db.free_regions(),
            );
        }
    }
    db.persist()?;
    println!(
        "drained: cycles={cycles} installed={installed} discarded={discarded} util={:.3} free_regions={} file {:.1} -> {:.1} GB, {:.0}s",
        db.utilization(),
        db.free_regions(),
        start_len as f64 / 1e9,
        db.flat_file_len() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
