//! Dump the flat trie's account leaves as TSV (forensic diff vs reth's
//! HashedAccounts export). Same format as export-tsv/reth-export.sh.
use mpt_flat_poc::{hex, FlatMpt};
use std::io::{BufWriter, Write};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: flatdump <flat> <out.tsv>");
    let out = args.next().expect("usage: flatdump <flat> <out.tsv>");
    let mpt = FlatMpt::open(&flat)?;
    let mut w = BufWriter::with_capacity(8 << 20, std::fs::File::create(&out)?);
    let mut cur = mpt.account_cursor();
    let mut key = [0u8; 32];
    let mut n = 0u64;
    let t = std::time::Instant::now();
    while let Some(e) = cur.seek(&key)? {
        writeln!(
            w,
            "0x{}\t{}\t{:#x}\t0x{}",
            hex(e.key),
            e.nonce,
            e.balance,
            hex(e.code_hash)
        )?;
        n += 1;
        if n % 50_000_000 == 0 {
            eprintln!("{n} ({}s)", t.elapsed().as_secs());
        }
        // successor key
        key = e.key;
        let mut carry = true;
        for b in key.iter_mut().rev() {
            let (v, c) = b.overflowing_add(1);
            *b = v;
            if !c { carry = false; break; }
        }
        if carry { break; }
    }
    w.flush()?;
    eprintln!("done: {n} accounts in {}s", t.elapsed().as_secs());
    Ok(())
}
