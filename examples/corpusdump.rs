//! Dump one block's ops from a shadow-follower diff corpus (forensics).
//!   cargo run --release --example corpusdump -- <corpus> <block> [verbose]
use mpt_flat_poc::{hex, Key, StateOp};
use serde::Deserialize;

#[derive(Deserialize)]
struct CorpusBlock {
    block: u64,
    #[allow(dead_code)]
    gas_used: u64,
    ops: Vec<(Key, StateOp)>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus = args.next().expect("usage: corpusdump <corpus> <block> [verbose]");
    let want_s = args.next().expect("block number or 'list'");
    if want_s == "list" {
        let f = std::io::BufReader::new(std::fs::File::open(&corpus).unwrap());
        let mut r = f;
        let mut i = 0u64;
        while let Ok(cb) = bincode::deserialize_from::<_, CorpusBlock>(&mut r) {
            println!("{} {} {}", i, cb.block, cb.ops.len());
            i += 1;
        }
        return;
    }
    let want: u64 = want_s.parse().unwrap();
    let verbose = args.next().is_some();
    let f = std::io::BufReader::new(std::fs::File::open(corpus).unwrap());
    let mut r = f;
    loop {
        let cb: CorpusBlock = match bincode::deserialize_from(&mut r) {
            Ok(cb) => cb,
            Err(_) => break,
        };
        if cb.block != want {
            continue;
        }
        let mut set_a = 0u64;
        let mut del_a = 0u64;
        let mut wipe = 0u64;
        let mut set_s = 0u64;
        let mut del_s = 0u64;
        for (key, op) in &cb.ops {
            match op {
                StateOp::SetAccount { nonce, balance, code_hash } => {
                    set_a += 1;
                    if verbose {
                        println!("SETA 0x{} nonce={} bal={} ch=0x{}", hex(*key), nonce, balance, hex(*code_hash));
                    }
                }
                StateOp::DeleteAccount => {
                    del_a += 1;
                    println!("DELA 0x{}", hex(*key));
                }
                StateOp::WipeStorage => {
                    wipe += 1;
                    println!("WIPE 0x{}", hex(*key));
                }
                StateOp::SetStorage { slot, .. } => {
                    set_s += 1;
                    if verbose {
                        println!("SETS 0x{} slot=0x{}", hex(*key), hex(*slot));
                    }
                }
                StateOp::DeleteStorage { slot } => {
                    del_s += 1;
                    println!("DELS 0x{} slot=0x{}", hex(*key), hex(*slot));
                }
            }
        }
        println!("block {want}: SetAccount={set_a} DeleteAccount={del_a} WipeStorage={wipe} SetStorage={set_s} DeleteStorage={del_s}");
        return;
    }
    eprintln!("block {want} not found in corpus");
}
