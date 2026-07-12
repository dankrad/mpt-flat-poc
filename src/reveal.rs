//! Direct reveal-node extraction for sparse-trie overlays.
//!
//! A sparse commitment trie needs a key's *path revealed* before it can apply
//! an update: every node from the root to the leaf (or to the divergence
//! point, for absent keys), each with its children as hash refs. That is
//! exactly a Merkle proof's information — but generating it through the
//! generic proof machinery re-hashes subtrees bottom-up and round-trips
//! through proof RLP, reconstructing what this store already holds: records
//! and frontier nodes carry every child hash precomputed. This module walks
//! the path once and copies the nodes out. No keccak, no HashBuilder.
//!
//! Masks follow the same conventions as the trie cursors (`hash_mask`
//! excludes inline and promoted-account children; `tree_mask` is
//! conservative-true for overflow children, whose subtrees always root in a
//! branch or extension).

use crate::cursor::{with_account_storage, RecordMemo};
use crate::{
    eth, hash_ram, key_nibbles, parse_node_lazy, parse_payload_lazy, ram_child_hash, FlatFile,
    FlatMpt, FlatSnapshot, Hash, Key, Node, NodeRef, RamChild, RamNode,
};
use anyhow::{bail, Result};

/// A child reference copied out of the store.
#[derive(Debug, Clone)]
pub enum RevealRef {
    Hash(Hash),
    /// RLP of an inline (<32 B) child, embedded verbatim.
    Inline(Vec<u8>),
}

/// One node on a revealed path. Paths are absolute nibble positions within
/// the trie being revealed (account trie, or one account's storage trie).
#[derive(Debug)]
pub enum RevealNode {
    Branch {
        path: Vec<u8>,
        children: [Option<RevealRef>; 16],
        /// Children with a stored branch below (cursor `tree_mask` rule).
        tree_mask: u16,
        /// Children whose 32-byte hashes the cursors would serve (`hash_mask`).
        hash_mask: u16,
    },
    Extension {
        path: Vec<u8>,
        key: Vec<u8>,
        child: RevealRef,
    },
    Leaf {
        path: Vec<u8>,
        /// Remaining key nibbles from `path` to depth 64.
        key: Vec<u8>,
        /// RLP leaf value (account body or storage value).
        value: Vec<u8>,
    },
    EmptyRoot,
}

fn node_child_ref(node: &Node) -> Result<RevealRef> {
    Ok(match node {
        Node::Leaf { nref, .. }
        | Node::Account { nref, .. }
        | Node::Extension { nref, .. }
        | Node::Branch { nref, .. } => match nref {
            NodeRef::Hash(h) => RevealRef::Hash(*h),
            NodeRef::Inline(b) => RevealRef::Inline(b.clone()),
        },
        Node::Overflow { root, .. } => RevealRef::Hash(*root),
        Node::Raw { buf, off, len, .. } => node_child_ref(&parse_node_lazy(buf, *off, *len)?)?,
        Node::Empty => bail!("empty child has no ref"),
    })
}

/// tree/hash mask bits per cursor conventions, without reading overflow
/// records (`tree_mask` conservative-true for overflow).
fn node_child_masks(node: &Node) -> Result<(bool, bool)> {
    Ok(match node {
        Node::Empty => (false, false),
        Node::Leaf { nref, .. } | Node::Account { nref, .. } => {
            (false, matches!(nref, NodeRef::Hash(_)))
        }
        Node::Extension { child, nref, .. } => {
            let below = node_child_masks(child)?.0 || matches!(**child, Node::Branch { .. });
            (below, matches!(nref, NodeRef::Hash(_)))
        }
        Node::Branch { nref, .. } => (true, matches!(nref, NodeRef::Hash(_))),
        Node::Overflow { .. } => (true, true),
        Node::Raw { buf, off, len, .. } => node_child_masks(&parse_node_lazy(buf, *off, *len)?)?,
    })
}

struct Emitter<'s> {
    store: &'s FlatFile,
    memo: &'s RecordMemo,
    out: Vec<RevealNode>,
    /// Positions already emitted this call (targets share prefixes).
    seen: std::collections::HashSet<Vec<u8>>,
}

impl<'s> Emitter<'s> {
    fn emit_ram_branch(&mut self, path: &[u8], children: &[Option<RamChild>; 16]) {
        if !self.seen.insert(path.to_vec()) {
            return;
        }
        let mut out: [Option<RevealRef>; 16] = Default::default();
        let (mut tree, mut hash) = (0u16, 0u16);
        for (i, c) in children.iter().enumerate() {
            let Some(c) = c else { continue };
            out[i] = Some(RevealRef::Hash(ram_child_hash(c)));
            let (t, h) = match c {
                RamChild::Ram(sub) => (
                    matches!(**sub, RamNode::Branch { .. } | RamNode::Extension { .. }),
                    true,
                ),
                RamChild::Disk { .. } | RamChild::Mem(_) => (true, true),
                // Promoted accounts: cursors exclude them from hash_mask.
                RamChild::Account(_) => (false, false),
            };
            if t {
                tree |= 1 << i;
            }
            if h {
                hash |= 1 << i;
            }
        }
        self.out.push(RevealNode::Branch { path: path.to_vec(), children: out, tree_mask: tree, hash_mask: hash });
    }

    fn emit_node_branch(&mut self, path: &[u8], children: &[Option<Box<Node>>; 16]) -> Result<()> {
        if !self.seen.insert(path.to_vec()) {
            return Ok(());
        }
        let mut out: [Option<RevealRef>; 16] = Default::default();
        let (mut tree, mut hash) = (0u16, 0u16);
        for (i, c) in children.iter().enumerate() {
            let Some(c) = c else { continue };
            out[i] = Some(node_child_ref(c)?);
            let (t, h) = node_child_masks(c)?;
            if t {
                tree |= 1 << i;
            }
            if h {
                hash |= 1 << i;
            }
        }
        self.out.push(RevealNode::Branch { path: path.to_vec(), children: out, tree_mask: tree, hash_mask: hash });
        Ok(())
    }

    fn emit_ext(&mut self, path: &[u8], key: &[u8], child: RevealRef) {
        if self.seen.insert(path.to_vec()) {
            self.out.push(RevealNode::Extension { path: path.to_vec(), key: key.to_vec(), child });
        }
    }

    fn emit_leaf(&mut self, path: &[u8], key: &[u8], value: Vec<u8>) {
        if self.seen.insert(path.to_vec()) {
            self.out.push(RevealNode::Leaf { path: path.to_vec(), key: key.to_vec(), value });
        }
    }

    /// Walk one target through a record-level node tree.
    fn walk_node(&mut self, node: &Node, prefix: &mut Vec<u8>, target: &[u8]) -> Result<()> {
        match node {
            Node::Empty => Ok(()),
            Node::Leaf { path, value, .. } => {
                self.emit_leaf(prefix, path, value.clone());
                Ok(())
            }
            Node::Account { path, nonce, balance, code_hash, storage_root, .. } => {
                let acct = eth::Account::contract(
                    *nonce,
                    *balance,
                    (*storage_root).into(),
                    (*code_hash).into(),
                );
                self.emit_leaf(prefix, path, acct.rlp());
                Ok(())
            }
            Node::Extension { path, child, .. } => {
                self.emit_ext(prefix, path, node_child_ref(child)?);
                let plen = prefix.len();
                prefix.extend_from_slice(path);
                let on_path = target[plen..].starts_with(path);
                // On divergence the extension's child branch is still needed
                // (V2 reveals fold extensions into their child branch).
                self.walk_node_step(child, prefix, if on_path { target } else { &[] })?;
                prefix.truncate(plen);
                Ok(())
            }
            Node::Branch { children, .. } => {
                self.emit_node_branch(prefix, children)?;
                if prefix.len() >= target.len() {
                    return Ok(());
                }
                let nib = target[prefix.len()] as usize;
                let Some(child) = &children[nib] else { return Ok(()) };
                prefix.push(nib as u8);
                self.walk_node(child, prefix, target)?;
                prefix.pop();
                Ok(())
            }
            Node::Overflow { ptr, .. } => {
                let sub = self.memo.read(self.store, *ptr)?;
                let node = sub.node.clone();
                self.walk_node(&node, prefix, target)
            }
            Node::Raw { buf, off, len, .. } => {
                let n = parse_node_lazy(buf, *off, *len)?;
                self.walk_node(&n, prefix, target)
            }
        }
    }

    /// One descent step used for a diverged extension child: emit the child
    /// branch only (empty target stops recursion immediately below it).
    fn walk_node_step(&mut self, node: &Node, prefix: &mut Vec<u8>, target: &[u8]) -> Result<()> {
        if target.is_empty() {
            // emit this node only
            match node {
                Node::Branch { children, .. } => self.emit_node_branch(prefix, children),
                Node::Overflow { ptr, .. } => {
                    let sub = self.memo.read(self.store, *ptr)?;
                    let node = sub.node.clone();
                    self.walk_node_step(&node, prefix, &[])
                }
                Node::Raw { buf, off, len, .. } => {
                    let n = parse_node_lazy(buf, *off, *len)?;
                    self.walk_node_step(&n, prefix, &[])
                }
                other => self.walk_node(other, prefix, &[]),
            }
        } else {
            self.walk_node(node, prefix, target)
        }
    }

    /// Walk one target through the RAM frontier.
    fn walk_ram(&mut self, node: &RamNode, prefix: &mut Vec<u8>, target: &[u8]) -> Result<()> {
        match node {
            RamNode::Empty => {
                if prefix.is_empty() && self.seen.insert(Vec::new()) {
                    self.out.push(RevealNode::EmptyRoot);
                }
                Ok(())
            }
            RamNode::Extension { path, child, .. } => {
                self.emit_ext(
                    prefix,
                    path,
                    RevealRef::Hash(hash_ram(child)),
                );
                let plen = prefix.len();
                prefix.extend_from_slice(path);
                if target[plen..].starts_with(path) {
                    self.walk_ram(child, prefix, target)?;
                } else {
                    // diverged: still emit the child branch for the V2 fold
                    self.walk_ram_step(child, prefix)?;
                }
                prefix.truncate(plen);
                Ok(())
            }
            RamNode::Branch { children, .. } => {
                self.emit_ram_branch(prefix, children);
                if prefix.len() >= target.len() {
                    return Ok(());
                }
                let nib = target[prefix.len()] as usize;
                let Some(child) = &children[nib] else { return Ok(()) };
                prefix.push(nib as u8);
                let r = self.walk_ram_child(child, prefix, target);
                prefix.pop();
                r
            }
        }
    }

    fn walk_ram_step(&mut self, node: &RamNode, prefix: &mut Vec<u8>) -> Result<()> {
        if let RamNode::Branch { children, .. } = node {
            self.emit_ram_branch(prefix, children);
        }
        Ok(())
    }

    fn walk_ram_child(&mut self, child: &RamChild, prefix: &mut Vec<u8>, target: &[u8]) -> Result<()> {
        match child {
            RamChild::Ram(sub) => self.walk_ram(sub, prefix, target),
            RamChild::Disk { ptr, .. } => {
                let sub = self.memo.read(self.store, *ptr)?;
                let node = sub.node.clone();
                self.walk_node(&node, prefix, target)
            }
            RamChild::Mem(m) => {
                m.touch();
                let sub = parse_payload_lazy(m.bytes.clone())?;
                self.walk_node(&sub.node, prefix, target)
            }
            RamChild::Account(a) => {
                // Account leaf in the account trie.
                let acct = eth::Account::contract(
                    a.nonce,
                    alloy_primitives::U256::from_be_bytes(a.balance),
                    hash_ram(&a.storage).into(),
                    a.code_hash.into(),
                );
                self.emit_leaf(prefix, &a.path, acct.rlp());
                Ok(())
            }
        }
    }
}

/// Reveal-path nodes for `keys` in the account trie of the view
/// `(store, upper)` — the walk backing both the [`FlatMpt`] and
/// [`FlatSnapshot`] entry points.
fn reveal_account_paths(store: &FlatFile, upper: &RamNode, keys: &[Key]) -> Result<Vec<RevealNode>> {
    let memo = RecordMemo::new();
    let mut e = Emitter { store, memo: &memo, out: Vec::new(), seen: Default::default() };
    let mut prefix = Vec::with_capacity(64);
    for k in keys {
        prefix.clear();
        let nibbles = key_nibbles(k);
        e.walk_ram(upper, &mut prefix, &nibbles)?;
    }
    Ok(e.out)
}

/// Reveal-path nodes for `slots` within `account`'s storage trie of the view
/// `(store, upper)`.
fn reveal_storage_paths(
    store: &FlatFile,
    upper: &RamNode,
    account: &Key,
    slots: &[Key],
) -> Result<Option<Vec<RevealNode>>> {
    let memo = RecordMemo::new();
    let out = with_account_storage(store, upper, account, |store, storage| -> Result<Vec<RevealNode>> {
        let mut e = Emitter { store, memo: &memo, out: Vec::new(), seen: Default::default() };
        let mut prefix = Vec::with_capacity(64);
        for s in slots {
            prefix.clear();
            let nibbles = key_nibbles(s);
            match storage {
                crate::cursor::StorageRef::Node(node) => e.walk_node(node, &mut prefix, &nibbles)?,
                crate::cursor::StorageRef::Ram(ram) => e.walk_ram(ram, &mut prefix, &nibbles)?,
            }
        }
        Ok(e.out)
    })?;
    out.transpose()
}

impl FlatMpt {
    /// Reveal-path nodes for `keys` in the account trie (sorted or not).
    pub fn reveal_account_paths(&self, keys: &[Key]) -> Result<Vec<RevealNode>> {
        reveal_account_paths(&self.store, &self.upper, keys)
    }

    /// Reveal-path nodes for `slots` within `account`'s storage trie.
    /// `Ok(None)` when the account doesn't exist or carries opaque storage.
    pub fn reveal_storage_paths(&self, account: &Key, slots: &[Key]) -> Result<Option<Vec<RevealNode>>> {
        reveal_storage_paths(&self.store, &self.upper, account, slots)
    }
}

impl FlatSnapshot {
    /// Reveal-path nodes for `keys` in the account trie, at snapshot time.
    pub fn reveal_account_paths(&self, keys: &[Key]) -> Result<Vec<RevealNode>> {
        reveal_account_paths(&self.store, &self.root, keys)
    }

    /// Reveal-path nodes for `slots` within `account`'s storage trie, at
    /// snapshot time. `Ok(None)` when the account doesn't exist or carries
    /// opaque storage.
    pub fn reveal_storage_paths(&self, account: &Key, slots: &[Key]) -> Result<Option<Vec<RevealNode>>> {
        reveal_storage_paths(&self.store, &self.root, account, slots)
    }

    /// Point-read + reveal in one walk: `get_value` semantics for the value,
    /// plus the reveal-path nodes the walk visited anyway. Lets an executor's
    /// state read double as the commitment trie's reveal — one record fetch
    /// serves both consumers.
    pub fn get_value_reveal(&self, key: &Key) -> Result<(Option<Vec<u8>>, Vec<RevealNode>)> {
        let nodes = reveal_account_paths(&self.store, &self.root, std::slice::from_ref(key))?;
        Ok((extract_exact(&nodes, key), nodes))
    }

    /// Storage-slot analog of [`Self::get_value_reveal`]. `None` nodes when
    /// the account is absent or carries opaque storage — the value then comes
    /// from the plain read path so `get_storage` semantics are preserved.
    pub fn get_storage_reveal(
        &self,
        account: &Key,
        slot: &Key,
    ) -> Result<(Option<Vec<u8>>, Option<Vec<RevealNode>>)> {
        match reveal_storage_paths(&self.store, &self.root, account, std::slice::from_ref(slot))? {
            Some(nodes) => Ok((extract_exact(&nodes, slot), Some(nodes))),
            None => Ok((self.get_storage(account, slot)?, None)),
        }
    }
}

/// The exact-match leaf value for `key` among revealed path nodes: a leaf
/// whose `path ++ key` spans the full 64 nibbles of the target. Divergent
/// leaves (exclusion proofs) don't match — same `None` as the plain getters.
fn extract_exact(nodes: &[RevealNode], key: &Key) -> Option<Vec<u8>> {
    let target = key_nibbles(key);
    for n in nodes {
        if let RevealNode::Leaf { path, key: rest, value } = n {
            if path.len() + rest.len() == target.len()
                && target[..path.len()] == path[..]
                && target[path.len()..] == rest[..]
            {
                return Some(value.clone());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, StateOp, U256};
    use sha3::{Digest, Keccak256};

    fn h(data: &[u8]) -> Key {
        let mut out = [0u8; 32];
        out.copy_from_slice(&Keccak256::digest(data));
        out
    }

    /// get_value_reveal / get_storage_reveal must be byte-equivalent to the
    /// plain getters (the EVM consumes these values) and emit the same nodes
    /// as the reveal-only walk, for present keys, absent keys, and storage
    /// shapes from empty to split/promoted records.
    #[test]
    fn read_with_reveal_matches_plain_getters() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("rwr.flat");
        let mut db = FlatMpt::create(&path, Config::default()).unwrap();

        let mut ops: Vec<(Key, StateOp)> = Vec::new();
        let mut acct_keys = Vec::new();
        for a in 0..400u64 {
            let key = h(&a.to_be_bytes());
            acct_keys.push(key);
            ops.push((key, StateOp::SetAccount {
                nonce: a + 1,
                balance: U256::from(a * 7 + 1),
                code_hash: h(&[a as u8; 4]),
            }));
            // storage sizes: none, small, large (record split / promote)
            let n_slots = match a % 5 { 0 => 0, 1 => 3, 2 => 40, 3 => 400, _ => 1 };
            for s in 0..n_slots {
                let slot = h(&(a * 100_000 + s).to_be_bytes());
                ops.push((key, StateOp::SetStorage {
                    slot,
                    value: eth::storage_value_rlp(U256::from(s + 1)),
                }));
            }
        }
        db.apply_block(ops.clone()).unwrap();
        let snap = db.snapshot();

        for (i, key) in acct_keys.iter().enumerate() {
            let (v, nodes) = snap.get_value_reveal(key).unwrap();
            assert_eq!(v, snap.get_value(key).unwrap(), "account value {i}");
            assert!(v.is_some(), "account {i} must exist");
            let only = snap.reveal_account_paths(std::slice::from_ref(key)).unwrap();
            assert_eq!(format!("{nodes:?}"), format!("{only:?}"), "account nodes {i}");
        }
        // absent account keys (exclusion paths)
        for a in 0..50u64 {
            let key = h(&(1_000_000 + a).to_be_bytes());
            let (v, _) = snap.get_value_reveal(&key).unwrap();
            assert_eq!(v, snap.get_value(&key).unwrap());
            assert!(v.is_none());
        }
        // storage: present + absent slots across the shape spectrum
        for (i, key) in acct_keys.iter().enumerate() {
            let a = i as u64;
            let present = h(&(a * 100_000).to_be_bytes());
            let absent = h(&(a * 100_000 + 99_999).to_be_bytes());
            for slot in [present, absent] {
                let (v, _) = snap.get_storage_reveal(key, &slot).unwrap();
                assert_eq!(v, snap.get_storage(key, &slot).unwrap(), "slot of account {i}");
            }
        }
    }
}

#[cfg(test)]
mod probe {
    use super::*;
    use crate::stats;
    use sha3::{Digest, Keccak256};
    use std::sync::atomic::Ordering::Relaxed;

    /// Point-read cost probe: on the flat at `FLAT_PROBE`, sample existing
    /// storage slots uniformly (cursor-seek to random keys) and count device
    /// record reads + wall latency per `get_storage`. Run against a SCRATCH
    /// COPY with dropped caches for the cold numbers:
    ///   FLAT_PROBE=/mnt2/probe-1b.flat cargo test --release probe_point_read_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_point_read_cost() {
        let path = std::env::var("FLAT_PROBE").expect("set FLAT_PROBE");
        let n: usize = std::env::var("FLAT_PROBE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
        let db = FlatMpt::open(&path).expect("open probe flat");
        let snap = db.snapshot();

        // Accounts that own storage: scan the account cursor, keep those whose
        // storage cursor yields an entry.
        let mut owners: Vec<Key> = Vec::new();
        let mut c = snap.account_cursor();
        let mut k = [0u8; 32];
        for _ in 0..5000 {
            let Some(e) = c.seek(&k).unwrap() else { break };
            if snap.storage_cursor(&e.key).seek(&[0u8; 32]).unwrap().is_some() {
                owners.push(e.key);
                if owners.len() >= 64 { break; }
            }
            k = e.key;
            let mut carry = 1u16;
            for b in k.iter_mut().rev() {
                let v = *b as u16 + carry; *b = v as u8; carry = v >> 8;
                if carry == 0 { break; }
            }
        }
        assert!(!owners.is_empty(), "no storage-owning accounts found");
        eprintln!("storage-owning accounts sampled: {}", owners.len());

        // Uniform existing slots: seek the storage cursor to hash-derived keys.
        let mut lookups: Vec<(Key, Key)> = Vec::new();
        'outer: for i in 0..n * 2 {
            let owner = owners[i % owners.len()];
            let probe = {
                let mut out = [0u8; 32];
                out.copy_from_slice(&Keccak256::digest(format!("probe-{i}")));
                out
            };
            if let Some((slot, _)) = snap.storage_cursor(&owner).seek(&probe).unwrap() {
                lookups.push((owner, slot));
                if lookups.len() >= n { break 'outer; }
            }
        }
        eprintln!("lookups prepared: {} (cursor warm-up reads not counted)", lookups.len());

        // Cold measurement: the sampling seeks above warmed these records.
        let _ = std::process::Command::new("sudo")
            .args(["-n", "sh", "-c", "sync; echo 1 > /proc/sys/vm/drop_caches"])
            .status();
        let mut per_reads: Vec<u64> = Vec::with_capacity(lookups.len());
        let mut per_ns: Vec<u64> = Vec::with_capacity(lookups.len());
        let mut per_bytes: Vec<u64> = Vec::with_capacity(lookups.len());
        for (owner, slot) in &lookups {
            let r0 = stats::B_READ_IOS.load(Relaxed);
            let b0 = stats::B_READ_BYTES.load(Relaxed);
            let t = std::time::Instant::now();
            let v = snap.get_storage(owner, slot).unwrap();
            per_ns.push(t.elapsed().as_nanos() as u64);
            per_reads.push(stats::B_READ_IOS.load(Relaxed) - r0);
            per_bytes.push(stats::B_READ_BYTES.load(Relaxed) - b0);
            assert!(v.is_some(), "sampled slot must exist");
        }
        per_bytes.sort_unstable();
        per_reads.sort_unstable();
        per_ns.sort_unstable();
        let pct = |v: &Vec<u64>, p: usize| v[v.len() * p / 100];
        let mut hist = std::collections::BTreeMap::new();
        for r in &per_reads { *hist.entry(*r).or_insert(0u64) += 1; }
        eprintln!("record reads per get_storage: histogram {hist:?}");
        eprintln!(
            "reads p50/p90/p99/max: {}/{}/{}/{}",
            pct(&per_reads, 50), pct(&per_reads, 90), pct(&per_reads, 99), per_reads.last().unwrap()
        );
        eprintln!(
            "latency µs p50/p90/p99/max: {}/{}/{}/{}",
            pct(&per_ns, 50) / 1000, pct(&per_ns, 90) / 1000,
            pct(&per_ns, 99) / 1000, per_ns.last().unwrap() / 1000
        );
        eprintln!(
            "record bytes/read p50/p90/p99/max: {}/{}/{}/{}",
            pct(&per_bytes, 50), pct(&per_bytes, 90), pct(&per_bytes, 99),
            per_bytes.last().unwrap()
        );
    }
}
