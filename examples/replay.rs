//! Replay a shadow-follower diff corpus against a checkpoint copy — the
//! offline A/B harness: identical real per-block inputs, timed apply_block,
//! repeatable across engine/tuning variants.
//!
//!   cargo run --release --example replay -- <ckpt.flat> <corpus> [max-blocks]

use mpt_flat_poc::{FlatMpt, Key, StateOp, hex};
use std::io::BufReader;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Must match the ExEx's corpus record.
#[derive(serde::Serialize, serde::Deserialize)]
struct CorpusBlock {
    block: u64,
    gas_used: u64,
    ops: Vec<(Key, StateOp)>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: replay <ckpt.flat> <corpus> [max-blocks]");
    let corpus = args.next().expect("need corpus path");
    let max_blocks: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let mut db = FlatMpt::open(&flat).unwrap();
    let mut rd = BufReader::new(std::fs::File::open(&corpus).unwrap());
    let mut times_us: Vec<f64> = Vec::new();
    let mut n_ops_total = 0u64;
    let t0 = Instant::now();
    let mut n = 0usize;
    loop {
        if n >= max_blocks {
            break;
        }
        let cb: CorpusBlock = match bincode::deserialize_from(&mut rd) {
            Ok(cb) => cb,
            Err(_) => break, // EOF
        };
        n_ops_total += cb.ops.len() as u64;
        let t = Instant::now();
        let (_root, _inv) = db.apply_block(cb.ops).unwrap();
        times_us.push(t.elapsed().as_micros() as f64);
        n += 1;
        if n % 100 == 0 {
            eprintln!("  block {} ({} replayed)", cb.block, n);
        }
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tot: f64 = times_us.iter().sum();
    let pct = |p: f64| times_us[((times_us.len() as f64 * p) as usize).min(times_us.len() - 1)];
    println!(
        "replayed {n} blocks ({n_ops_total} ops) in {:.1}s\n\
         per-block ms: mean {:.1}  p50 {:.1}  p90 {:.1}  p99 {:.1}  max {:.1}\n\
         per-op us: mean {:.2}  final root {}",
        t0.elapsed().as_secs_f64(),
        tot / times_us.len().max(1) as f64 / 1000.0,
        pct(0.50) / 1000.0,
        pct(0.90) / 1000.0,
        pct(0.99) / 1000.0,
        times_us.last().copied().unwrap_or(0.0) / 1000.0,
        tot / (n_ops_total.max(1)) as f64,
        hex(db.root()),
    );
}
