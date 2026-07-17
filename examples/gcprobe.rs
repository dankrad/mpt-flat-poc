//! Read-only GC probe: open a flat file, pick victim regions, and classify
//! every record: parse failure / resolvable / dead / unresolvable, with
//! prefix-length histogram. No writes.
//!
//!   cargo run --release --example gcprobe -- <flat> [regions]

use mpt_flat_poc::FlatMpt;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("flat path");
    let n: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let db = FlatMpt::open(&path)?;
    let stats = db.gc_probe(n)?;
    println!("{stats}");
    Ok(())
}
