//! Semantic diff of a checkpoint against the reth TSV export it was built from:
//! enumerate the trie in key order (16 parallel top-nibble subtrees) and
//! merge-compare against the sorted accounts.tsv/storages.tsv, classifying every
//! divergence (missing/extra accounts or slots, field/value mismatches).
//!
//!   cargo run --release --example tsvdiff -- \
//!     /mnt2/tip-25374199.flat /mnt/export-25374199/accounts.tsv /mnt/export-25374199/storages.tsv

use alloy_primitives::U256;
use mpt_flat_poc::eth::{self, EMPTY_CODE_HASH};
use mpt_flat_poc::{FlatMpt, Key, ScanEntry, hex};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn hex32(s: &str) -> Key {
    let mut k = [0u8; 32];
    let b = alloy_primitives::hex::decode(s.trim_start_matches("0x")).unwrap();
    k[32 - b.len()..].copy_from_slice(&b);
    k
}

fn u256_hex(s: &str) -> U256 {
    let h = s.trim_start_matches("0x");
    if h.is_empty() { U256::ZERO } else { U256::from_str_radix(h, 16).unwrap() }
}

/// First nibble of the key at the start of a TSV line ("0x<hex>...").
fn line_nibble(line: &str) -> u8 {
    u8::from_str_radix(&line[2..3], 16).unwrap()
}

/// Byte offset of the first full line whose key's top nibble is >= `nib`.
fn seek_nibble(path: &str, nib: u8) -> u64 {
    if nib == 0 {
        return 0;
    }
    let mut f = File::open(path).unwrap();
    let len = f.metadata().unwrap().len();
    let (mut lo, mut hi) = (0u64, len); // invariant: first-such-line offset in (lo, hi]
    while hi - lo > 1 << 20 {
        let mid = (lo + hi) / 2;
        f.seek(SeekFrom::Start(mid)).unwrap();
        let mut r = BufReader::new(&f);
        let mut skip = String::new();
        r.read_line(&mut skip).unwrap(); // partial line
        let mut line = String::new();
        if r.read_line(&mut line).unwrap() == 0 || line_nibble(&line) >= nib {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // Scan the last 1 MiB linearly for the exact boundary.
    f.seek(SeekFrom::Start(lo)).unwrap();
    let mut r = BufReader::new(&f);
    let mut pos = lo;
    let mut line = String::new();
    if lo > 0 {
        pos += r.read_line(&mut line).unwrap() as u64; // partial line
    }
    loop {
        line.clear();
        let n = r.read_line(&mut line).unwrap();
        if n == 0 || line_nibble(&line) >= nib {
            return pos;
        }
        pos += n as u64;
    }
}

struct AcctRow {
    key: Key,
    nonce: u64,
    balance: U256,
    code_hash: [u8; 32],
}

struct SlotRow {
    addr: Key,
    slot: Key,
    value_rlp: Vec<u8>,
}

/// Line reader over one top-nibble segment of a sorted TSV.
struct Segment {
    r: BufReader<File>,
    nib: u8,
    line: String,
}

impl Segment {
    fn new(path: &str, nib: u8) -> Self {
        let mut f = File::open(path).unwrap();
        f.seek(SeekFrom::Start(seek_nibble(path, nib))).unwrap();
        Segment { r: BufReader::with_capacity(4 << 20, f), nib, line: String::new() }
    }
    /// Next line within the segment (None at segment end / EOF).
    fn next_line(&mut self) -> Option<&str> {
        self.line.clear();
        if self.r.read_line(&mut self.line).unwrap() == 0 {
            return None;
        }
        let l = self.line.trim_end();
        if line_nibble(l) != self.nib { None } else { Some(l) }
    }
}

#[derive(Default)]
struct Diff {
    accounts: u64,
    slots: u64,
    acct_missing: u64,
    acct_extra: u64,
    acct_fields: u64,
    slot_missing: u64,
    slot_extra: u64,
    slot_value: u64,
    opaque: u64,
    samples: Vec<String>,
}

impl Diff {
    const MAX_SAMPLES: usize = 24;
    fn sample(&mut self, s: String) {
        if self.samples.len() < Self::MAX_SAMPLES {
            self.samples.push(s);
        }
    }
    fn total(&self) -> u64 {
        self.acct_missing + self.acct_extra + self.acct_fields + self.slot_missing
            + self.slot_extra + self.slot_value + self.opaque
    }
}

struct Comparator {
    acc: Segment,
    sto: Segment,
    pending_acc: Option<AcctRow>,
    pending_sto: Option<SlotRow>,
    d: Diff,
}

impl Comparator {
    fn new(accounts: &str, storages: &str, nib: u8) -> Self {
        let mut c = Comparator {
            acc: Segment::new(accounts, nib),
            sto: Segment::new(storages, nib),
            pending_acc: None,
            pending_sto: None,
            d: Diff::default(),
        };
        c.advance_acc();
        c.advance_sto();
        c
    }
    fn advance_acc(&mut self) {
        self.pending_acc = self.acc.next_line().map(|l| {
            let mut it = l.split('\t');
            let key = hex32(it.next().unwrap());
            let nonce: u64 = it.next().unwrap().parse().unwrap();
            let balance = u256_hex(it.next().unwrap());
            let ch = it.next().unwrap();
            let code_hash = if ch == "null" { EMPTY_CODE_HASH.0 } else { hex32(ch) };
            AcctRow { key, nonce, balance, code_hash }
        });
    }
    fn advance_sto(&mut self) {
        self.pending_sto = self.sto.next_line().map(|l| {
            let mut it = l.split('\t');
            let addr = hex32(it.next().unwrap());
            let slot = hex32(it.next().unwrap());
            let value_rlp = eth::storage_value_rlp(u256_hex(it.next().unwrap()));
            SlotRow { addr, slot, value_rlp }
        });
    }

    fn on_entry(&mut self, e: ScanEntry) {
        match e {
            ScanEntry::Account { key, nonce, balance, code_hash } => {
                self.d.accounts += 1;
                while let Some(row) = &self.pending_acc {
                    if row.key >= key {
                        break;
                    }
                    self.d.acct_missing += 1;
                    let k = row.key;
                    self.d.sample(format!("account {} in TSV, missing from trie", hex(k)));
                    self.advance_acc();
                }
                match &self.pending_acc {
                    Some(row) if row.key == key => {
                        if row.nonce != nonce || row.balance != balance || row.code_hash != code_hash
                        {
                            self.d.acct_fields += 1;
                            self.d.sample(format!(
                                "account {} fields: trie (n {nonce}, b {balance}, ch {}) vs tsv (n {}, b {}, ch {})",
                                hex(key), hex(code_hash), row.nonce, row.balance, hex(row.code_hash)
                            ));
                        }
                        self.advance_acc();
                    }
                    _ => {
                        self.d.acct_extra += 1;
                        self.d.sample(format!("account {} in trie, not in TSV", hex(key)));
                    }
                }
            }
            ScanEntry::Slot { account, slot, value } => {
                self.d.slots += 1;
                while let Some(row) = &self.pending_sto {
                    if (row.addr, row.slot) >= (account, slot) {
                        break;
                    }
                    self.d.slot_missing += 1;
                    let (a, s) = (row.addr, row.slot);
                    self.d.sample(format!("slot {}/{} in TSV, missing from trie", hex(a), hex(s)));
                    self.advance_sto();
                }
                match &self.pending_sto {
                    Some(row) if (row.addr, row.slot) == (account, slot) => {
                        if row.value_rlp != value {
                            self.d.slot_value += 1;
                            let tv = alloy_primitives::hex::encode(&row.value_rlp);
                            self.d.sample(format!(
                                "slot {}/{} value: trie {} vs tsv {}",
                                hex(account), hex(slot), alloy_primitives::hex::encode(&value), tv
                            ));
                        }
                        self.advance_sto();
                    }
                    _ => {
                        self.d.slot_extra += 1;
                        self.d.sample(format!("slot {}/{} in trie, not in TSV", hex(account), hex(slot)));
                    }
                }
            }
            ScanEntry::Opaque { key, .. } => {
                self.d.opaque += 1;
                self.d.sample(format!("opaque leaf {} in trie", hex(key)));
            }
        }
    }

    fn finish(mut self) -> Diff {
        while let Some(row) = &self.pending_acc {
            self.d.acct_missing += 1;
            let k = row.key;
            self.d.sample(format!("account {} in TSV, missing from trie (tail)", hex(k)));
            self.advance_acc();
        }
        while let Some(row) = &self.pending_sto {
            self.d.slot_missing += 1;
            let (a, s) = (row.addr, row.slot);
            self.d.sample(format!("slot {}/{} in TSV, missing from trie (tail)", hex(a), hex(s)));
            self.advance_sto();
        }
        self.d
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let flat = args.next().expect("usage: tsvdiff <flat> <accounts.tsv> <storages.tsv>");
    let accounts = args.next().expect("need accounts.tsv");
    let storages = args.next().expect("need storages.tsv");

    let t = Instant::now();
    let db = FlatMpt::open(&flat).unwrap();
    eprintln!("[{:>6.1}s] opened {flat}", t.elapsed().as_secs_f64());

    let diffs: Vec<Diff> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0u8..16)
            .map(|nib| {
                let (db, accounts, storages) = (&db, accounts.as_str(), storages.as_str());
                scope.spawn(move || {
                    let mut c = Comparator::new(accounts, storages, nib);
                    db.scan_top(nib, &mut |e| {
                        c.on_entry(e);
                        Ok(())
                    })
                    .unwrap();
                    let d = c.finish();
                    eprintln!(
                        "[{:>6.1}s] nibble {nib:x}: {} accounts {} slots, {} diffs",
                        t.elapsed().as_secs_f64(),
                        d.accounts,
                        d.slots,
                        d.total()
                    );
                    d
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut tot = Diff::default();
    println!("\nper-nibble: (accounts / slots / missing-acct / extra-acct / bad-fields / missing-slot / extra-slot / bad-value / opaque)");
    for (nib, d) in diffs.iter().enumerate() {
        println!(
            "  {nib:x}: {} / {} / {} / {} / {} / {} / {} / {} / {}",
            d.accounts, d.slots, d.acct_missing, d.acct_extra, d.acct_fields,
            d.slot_missing, d.slot_extra, d.slot_value, d.opaque
        );
        tot.accounts += d.accounts;
        tot.slots += d.slots;
        tot.acct_missing += d.acct_missing;
        tot.acct_extra += d.acct_extra;
        tot.acct_fields += d.acct_fields;
        tot.slot_missing += d.slot_missing;
        tot.slot_extra += d.slot_extra;
        tot.slot_value += d.slot_value;
        tot.opaque += d.opaque;
    }
    println!(
        "\nTOTAL: {} accounts, {} slots\n  missing accounts {}\n  extra accounts {}\n  field mismatches {}\n  missing slots {}\n  extra slots {}\n  value mismatches {}\n  opaque leaves {}",
        tot.accounts, tot.slots, tot.acct_missing, tot.acct_extra, tot.acct_fields,
        tot.slot_missing, tot.slot_extra, tot.slot_value, tot.opaque
    );
    println!("\nsamples:");
    for d in &diffs {
        for s in &d.samples {
            println!("  {s}");
        }
    }
    println!("\nverdict: {}", if tot.total() == 0 { "trie content matches TSV export" } else { "content divergence found" });
}
