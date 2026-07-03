//! Probe a nested checkpoint: read every sampled account and slot, reporting
//! the first failures (diagnostic for record round-trip bugs).
use mpt_flat_poc::{FlatMpt, Key};
use std::io::{BufRead, BufReader};

fn hex32(s: &str) -> Key {
    let mut k = [0u8; 32];
    let b = alloy_primitives::hex::decode(s.trim_start_matches("0x")).unwrap();
    k[32 - b.len()..].copy_from_slice(&b);
    k
}

static LAST: std::sync::Mutex<Option<(u64, Key)>> = std::sync::Mutex::new(None);

fn main() {
    std::thread::spawn(|| {
        let mut prev = None;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let cur = *LAST.lock().unwrap();
            if cur.is_some() && cur == prev {
                if let Some((i, k)) = cur {
                    eprintln!("STUCK at item {i}: key {}", mpt_flat_poc::hex(k));
                }
            }
            prev = cur;
        }
    });
    let mut args = std::env::args().skip(1);
    let flat = args.next().unwrap();
    let accounts_tsv = args.next().unwrap();
    let storages_tsv = args.next().unwrap();
    eprintln!("opening...");
    let db = FlatMpt::open(&flat).unwrap();
    eprintln!("opened");
    let (a, t) = db.audit_live_units();
    eprintln!("audited");
    println!("audit_live_units: alloc={a} true={t} match={}", a == t);

    let mut acc_err = 0u64;
    let mut acc_none = 0u64;
    let mut n = 0u64;
    for (i, line) in BufReader::new(std::fs::File::open(&accounts_tsv).unwrap()).lines().enumerate() {
        if i % 97 != 0 { continue; }
        let l = line.unwrap();
        let addr = hex32(l.split('\t').next().unwrap());
        n += 1;
        if n % 1000 == 0 { eprintln!("  acct {n}"); }
        LAST.lock().unwrap().replace((n, addr));
        match db.get_value(&addr) {
            Ok(Some(_)) => {}
            Ok(None) => { acc_none += 1; if acc_none < 3 { println!("account NONE: {}", mpt_flat_poc::hex(addr)); } }
            Err(e) => { acc_err += 1; if acc_err < 4 { println!("account ERR {}: {e}", mpt_flat_poc::hex(addr)); } }
        }
        if n >= 2_000 { break; }
    }
    println!("accounts probed {n}: errors {acc_err}, missing {acc_none}");

    let (mut s_err, mut s_none, mut m) = (0u64, 0u64, 0u64);
    let (mut tot_us, mut slow) = (0u64, 0u64);
    let mut worst: (u64, Key) = (0, [0u8; 32]);
    for (i, line) in BufReader::new(std::fs::File::open(&storages_tsv).unwrap()).lines().enumerate() {
        let _ = i;
        let l = line.unwrap();
        let mut it = l.split('\t');
        let addr = hex32(it.next().unwrap());
        let slot = hex32(it.next().unwrap());
        m += 1;
        if m % 1000 == 0 { eprintln!("  slot {m}"); }
        LAST.lock().unwrap().replace((m, addr));
        let t = std::time::Instant::now();
        let r = db.get_storage(&addr, &slot);
        let us = t.elapsed().as_micros() as u64;
        tot_us += us;
        if us > worst.0 { worst = (us, addr); }
        if us > 5000 { slow += 1; if slow < 6 { eprintln!("SLOW {us}us: {}", mpt_flat_poc::hex(addr)); } }
        match r {
            Ok(Some(_)) => {}
            Ok(None) => { s_none += 1; if s_none < 3 { println!("slot NONE: {} {}", mpt_flat_poc::hex(addr), mpt_flat_poc::hex(slot)); } }
            Err(e) => { s_err += 1; if s_err < 4 { println!("slot ERR {} {}: {e}", mpt_flat_poc::hex(addr), mpt_flat_poc::hex(slot)); } }
        }
        if m >= 30_000 { break; }
    }
    println!("slots probed {m}: errors {s_err}, missing {s_none}, slow(>5ms) {slow}, mean {}us, worst {}us at {}",
        tot_us / m.max(1), worst.0, mpt_flat_poc::hex(worst.1));
}
