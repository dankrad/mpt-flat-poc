//! Ordered leaf cursors over the flat trie — the backing for reth's
//! `HashedCursor` interfaces (and, later, trie-node cursors), so reth's
//! sparse-trie/state-root machinery can run directly on flat-MPT data.
//!
//! Design: stateless successor walks. `seek(k)` descends from the root and
//! returns the smallest leaf with key `>= k`; `next()` is
//! `seek(last_key + 1)`. Each descent re-reads its path, but a one-record
//! memo means consecutive keys parse each record once (~150-200 leaves per
//! record), and the intended consumer (root walks over changed prefixes,
//! sparse-trie boundary fetches) seeks in key order.
//!
//! All walks take `&FlatMpt` — read-only and safe alongside other readers.

use crate::*;
use std::cell::RefCell;
use std::rc::Rc;

/// An account leaf as surfaced by [`AccountCursor`].
#[derive(Debug, Clone, PartialEq)]
pub struct AccountEntry {
    pub key: Key,
    pub nonce: u64,
    pub balance: U256,
    pub code_hash: Hash,
    pub storage_root: Hash,
}

/// What a seek finds at a leaf position.
enum LeafOut {
    Account { nonce: u64, balance: U256, code_hash: Hash, storage_root: Hash },
    /// Raw leaf value bytes (plain value tries / storage slots).
    Value(Vec<u8>),
}

/// One-record parse memo shared by a cursor's descents.
struct RecordMemo(RefCell<Option<(u32, Rc<DiskSubtree>)>>);

impl RecordMemo {
    fn new() -> Self {
        Self(RefCell::new(None))
    }
    fn read(&self, store: &FlatFile, ptr: DiskPtr) -> Result<Rc<DiskSubtree>> {
        if let Some((unit, sub)) = self.0.borrow().as_ref() {
            if *unit == ptr.unit {
                return Ok(sub.clone());
            }
        }
        let sub = Rc::new(store.read_lazy(ptr)?);
        *self.0.borrow_mut() = Some((ptr.unit, sub.clone()));
        Ok(sub)
    }
}

/// Increment a 64-nibble key by one; `None` on overflow (end of keyspace).
fn key_successor(key: &Key) -> Option<Key> {
    let mut k = *key;
    for b in k.iter_mut().rev() {
        let (nb, carry) = b.overflowing_add(1);
        *b = nb;
        if !carry {
            return Some(k);
        }
    }
    None
}

fn nibbles_to_key(nibbles: &[u8]) -> Key {
    debug_assert_eq!(nibbles.len(), 64);
    let mut k = [0u8; 32];
    for (i, pair) in nibbles.chunks(2).enumerate() {
        k[i] = (pair[0] << 4) | pair[1];
    }
    k
}

/// Compare a subtree position `prefix` against `target`: is any leaf under
/// this prefix possibly `>= target`, and if so must we still constrain the
/// descent (`OnPath`) or take the subtree minimum (`AllGreater`)?
enum Rel {
    /// prefix is a prefix of target — constrained descent continues.
    OnPath,
    /// every leaf under prefix is > target — take the subtree's minimum.
    AllGreater,
    /// every leaf under prefix is < target — skip the subtree.
    AllLess,
}

fn rel(prefix: &[u8], target: &[u8]) -> Rel {
    let n = prefix.len().min(target.len());
    match prefix[..n].cmp(&target[..n]) {
        std::cmp::Ordering::Less => Rel::AllLess,
        std::cmp::Ordering::Greater => Rel::AllGreater,
        std::cmp::Ordering::Equal => Rel::OnPath,
    }
}

// ---------------------------------------------------------------------------
// Node-level (record) successor walk
// ---------------------------------------------------------------------------

/// Smallest leaf `>= target` within `node` at absolute nibble `prefix`.
/// `min_mode` short-circuits the target comparison (take the subtree minimum).
fn node_seek(
    store: &FlatFile,
    memo: &RecordMemo,
    node: &Node,
    prefix: &mut Vec<u8>,
    target: &[u8],
    min_mode: bool,
) -> Result<Option<(Vec<u8>, LeafOut)>> {
    match node {
        Node::Empty => Ok(None),
        Node::Leaf { path, value, .. } => {
            let plen = prefix.len();
            prefix.extend_from_slice(path);
            let out = if min_mode || prefix[..] >= target[..prefix.len().min(target.len())]
                && !matches!(rel(prefix, target), Rel::AllLess)
            {
                // Full-key comparison: leaf keys are always full depth.
                if min_mode || prefix[..] >= target[..] {
                    Some((prefix.clone(), LeafOut::Value(value.clone())))
                } else {
                    None
                }
            } else {
                None
            };
            prefix.truncate(plen);
            Ok(out)
        }
        Node::Account { path, nonce, balance, code_hash, storage_root, .. } => {
            let plen = prefix.len();
            prefix.extend_from_slice(path);
            let take = min_mode || prefix[..] >= target[..];
            let out = take.then(|| {
                (
                    prefix.clone(),
                    LeafOut::Account {
                        nonce: *nonce,
                        balance: *balance,
                        code_hash: *code_hash,
                        storage_root: *storage_root,
                    },
                )
            });
            prefix.truncate(plen);
            Ok(out)
        }
        Node::Extension { path, child, .. } => {
            let plen = prefix.len();
            prefix.extend_from_slice(path);
            let sub_min = min_mode
                || match rel(prefix, target) {
                    Rel::AllLess => {
                        prefix.truncate(plen);
                        return Ok(None);
                    }
                    Rel::AllGreater => true,
                    Rel::OnPath => false,
                };
            let out = node_seek(store, memo, child, prefix, target, sub_min)?;
            prefix.truncate(plen);
            Ok(out)
        }
        Node::Branch { children, .. } => {
            let depth = prefix.len();
            let start: u8 = if min_mode || depth >= target.len() { 0 } else { target[depth] };
            for i in start..16 {
                let Some(child) = &children[i as usize] else { continue };
                prefix.push(i);
                let sub_min = min_mode || i > start || depth >= target.len();
                let out = node_seek(store, memo, child, prefix, target, sub_min)?;
                prefix.pop();
                if out.is_some() {
                    return Ok(out);
                }
            }
            Ok(None)
        }
        Node::Overflow { ptr, .. } => {
            let sub = memo.read(store, *ptr)?;
            debug_assert_eq!(sub.prefix.len(), prefix.len());
            node_seek(store, memo, &sub.node, prefix, target, min_mode)
        }
        Node::Raw { buf, off, len, .. } => {
            let n = parse_node_lazy(buf, *off, *len)?;
            node_seek(store, memo, &n, prefix, target, min_mode)
        }
    }
}

// ---------------------------------------------------------------------------
// Frontier-level successor walk
// ---------------------------------------------------------------------------

fn ram_seek(
    store: &FlatFile,
    memo: &RecordMemo,
    node: &RamNode,
    prefix: &mut Vec<u8>,
    target: &[u8],
    min_mode: bool,
) -> Result<Option<(Vec<u8>, LeafOut)>> {
    match node {
        RamNode::Empty => Ok(None),
        RamNode::Extension { path, child, .. } => {
            let plen = prefix.len();
            prefix.extend_from_slice(path);
            let sub_min = min_mode
                || match rel(prefix, target) {
                    Rel::AllLess => {
                        prefix.truncate(plen);
                        return Ok(None);
                    }
                    Rel::AllGreater => true,
                    Rel::OnPath => false,
                };
            let out = ram_seek(store, memo, child, prefix, target, sub_min)?;
            prefix.truncate(plen);
            Ok(out)
        }
        RamNode::Branch { children, .. } => {
            let depth = prefix.len();
            let start: u8 = if min_mode || depth >= target.len() { 0 } else { target[depth] };
            for i in start..16 {
                let Some(child) = &children[i as usize] else { continue };
                prefix.push(i);
                let sub_min = min_mode || i > start || depth >= target.len();
                let out = ram_child_seek(store, memo, child, prefix, target, sub_min)?;
                prefix.pop();
                if out.is_some() {
                    return Ok(out);
                }
            }
            Ok(None)
        }
    }
}

fn ram_child_seek(
    store: &FlatFile,
    memo: &RecordMemo,
    child: &RamChild,
    prefix: &mut Vec<u8>,
    target: &[u8],
    min_mode: bool,
) -> Result<Option<(Vec<u8>, LeafOut)>> {
    match child {
        RamChild::Ram(sub) => ram_seek(store, memo, sub, prefix, target, min_mode),
        RamChild::Disk { ptr, .. } => {
            let sub = memo.read(store, *ptr)?;
            debug_assert_eq!(sub.prefix.len(), prefix.len());
            node_seek(store, memo, &sub.node, prefix, target, min_mode)
        }
        RamChild::Mem(m) => {
            m.touch();
            let sub = parse_payload_lazy(m.bytes.clone())?;
            node_seek(store, memo, &sub.node, prefix, target, min_mode)
        }
        RamChild::Account(a) => {
            let plen = prefix.len();
            prefix.extend_from_slice(&a.path);
            let take = min_mode || prefix[..] >= target[..];
            let out = take.then(|| {
                (
                    prefix.clone(),
                    LeafOut::Account {
                        nonce: a.nonce,
                        balance: U256::from_be_bytes(a.balance),
                        code_hash: a.code_hash,
                        storage_root: hash_ram_parallel(&a.storage),
                    },
                )
            });
            prefix.truncate(plen);
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Public cursors
// ---------------------------------------------------------------------------

/// Ordered cursor over the account trie's leaves.
pub struct AccountCursor<'a> {
    mpt: &'a FlatMpt,
    memo: RecordMemo,
    last: Option<Key>,
}

impl<'a> AccountCursor<'a> {
    pub fn seek(&mut self, key: &Key) -> Result<Option<AccountEntry>> {
        let target = key_nibbles(key);
        let mut prefix = Vec::with_capacity(64);
        let hit = ram_seek(&self.mpt.store, &self.memo, &self.mpt.upper, &mut prefix, &target, false)?;
        Ok(match hit {
            Some((nibbles, LeafOut::Account { nonce, balance, code_hash, storage_root })) => {
                let k = nibbles_to_key(&nibbles);
                self.last = Some(k);
                Some(AccountEntry { key: k, nonce, balance, code_hash, storage_root })
            }
            Some((nibbles, LeafOut::Value(v))) => {
                // Plain-value trie (non-account engines): surface the account
                // fields by decoding the leaf RLP.
                let k = nibbles_to_key(&nibbles);
                self.last = Some(k);
                let acct = eth::Account::decode(&v)
                    .map_err(|e| anyhow::anyhow!("account leaf RLP: {e}"))?;
                Some(AccountEntry {
                    key: k,
                    nonce: acct.nonce,
                    balance: acct.balance,
                    code_hash: acct.code_hash.0,
                    storage_root: acct.storage_root.0,
                })
            }
            None => {
                self.last = None;
                None
            }
        })
    }

    pub fn next(&mut self) -> Result<Option<AccountEntry>> {
        let Some(last) = self.last else { return Ok(None) };
        let Some(succ) = key_successor(&last) else {
            self.last = None;
            return Ok(None);
        };
        self.seek(&succ)
    }
}

/// Ordered cursor over one account's storage-slot leaves. Values are the
/// stored `RLP(U256)` slot encodings.
pub struct StorageCursor<'a> {
    mpt: &'a FlatMpt,
    memo: RecordMemo,
    account_key: Key,
    last: Option<Key>,
}

impl<'a> StorageCursor<'a> {
    pub fn seek(&mut self, slot_key: &Key) -> Result<Option<(Key, Vec<u8>)>> {
        let target = key_nibbles(slot_key);
        let hit = self
            .mpt
            .with_account_storage(&self.account_key, |store, storage| match storage {
                StorageRef::Node(node) => {
                    let mut prefix = Vec::with_capacity(64);
                    node_seek(store, &self.memo, node, &mut prefix, &target, false)
                }
                StorageRef::Ram(ram) => {
                    let mut prefix = Vec::with_capacity(64);
                    ram_seek(store, &self.memo, ram, &mut prefix, &target, false)
                }
            })?
            .transpose()?
            .flatten();
        Ok(match hit {
            Some((nibbles, LeafOut::Value(v))) => {
                let k = nibbles_to_key(&nibbles);
                self.last = Some(k);
                Some((k, v))
            }
            Some((_, LeafOut::Account { .. })) => {
                anyhow::bail!("account node inside a storage trie")
            }
            None => {
                self.last = None;
                None
            }
        })
    }

    pub fn next(&mut self) -> Result<Option<(Key, Vec<u8>)>> {
        let Some(last) = self.last else { return Ok(None) };
        let Some(succ) = key_successor(&last) else {
            self.last = None;
            return Ok(None);
        };
        self.seek(&succ)
    }
}

/// A view of an account's storage subtree for the cursor walk.
pub(crate) enum StorageRef<'n> {
    Node(&'n Node),
    Ram(&'n RamNode),
}

impl FlatMpt {
    /// Ordered cursor over account leaves.
    pub fn account_cursor(&self) -> AccountCursor<'_> {
        AccountCursor { mpt: self, memo: RecordMemo::new(), last: None }
    }

    /// Ordered cursor over `account_key`'s storage leaves.
    pub fn storage_cursor(&self, account_key: &Key) -> StorageCursor<'_> {
        StorageCursor { mpt: self, memo: RecordMemo::new(), account_key: *account_key, last: None }
    }
}

impl FlatMpt {
    /// Descend to `account_key`'s leaf and run `f` over its storage subtree.
    /// `Ok(None)` if the account doesn't exist (or is a plain-value leaf).
    pub(crate) fn with_account_storage<R>(
        &self,
        account_key: &Key,
        f: impl FnOnce(&FlatFile, StorageRef<'_>) -> R,
    ) -> Result<Option<R>> {
        let nibbles = key_nibbles(account_key);
        // Frontier descent.
        let mut node = &self.upper;
        let mut depth = 0usize;
        loop {
            match node {
                RamNode::Empty => return Ok(None),
                RamNode::Extension { path, child, .. } => {
                    if !nibbles[depth..].starts_with(path) {
                        return Ok(None);
                    }
                    depth += path.len();
                    node = child;
                }
                RamNode::Branch { children, .. } => {
                    match &children[nibbles[depth] as usize] {
                        None => return Ok(None),
                        Some(RamChild::Ram(sub)) => {
                            depth += 1;
                            node = sub;
                        }
                        Some(RamChild::Account(a)) => {
                            return Ok((nibbles[depth + 1..] == a.path[..])
                                .then(|| f(&self.store, StorageRef::Ram(&a.storage))));
                        }
                        Some(RamChild::Disk { ptr, .. }) => {
                            let sub = self.store.read_lazy(*ptr)?;
                            return record_account_storage(&self.store, &sub.node, &nibbles, sub.prefix.len(), f);
                        }
                        Some(RamChild::Mem(m)) => {
                            m.touch();
                            let sub = parse_payload_lazy(m.bytes.clone())?;
                            return record_account_storage(&self.store, &sub.node, &nibbles, sub.prefix.len(), f);
                        }
                    }
                }
            }
        }
    }
}

/// Record-level descent to the account leaf; runs `f` on its storage.
fn record_account_storage<R>(
    store: &FlatFile,
    node: &Node,
    nibbles: &[u8],
    depth: usize,
    f: impl FnOnce(&FlatFile, StorageRef<'_>) -> R,
) -> Result<Option<R>> {
    match node {
        Node::Empty | Node::Leaf { .. } => Ok(None),
        Node::Account { path, storage, .. } => {
            Ok((nibbles[depth..] == path[..]).then(|| f(store, StorageRef::Node(storage))))
        }
        Node::Extension { path, child, .. } => {
            if nibbles[depth..].starts_with(path) {
                record_account_storage(store, child, nibbles, depth + path.len(), f)
            } else {
                Ok(None)
            }
        }
        Node::Branch { children, .. } => match &children[nibbles[depth] as usize] {
            Some(child) => record_account_storage(store, child, nibbles, depth + 1, f),
            None => Ok(None),
        },
        Node::Overflow { ptr, .. } => {
            let sub = store.read_lazy(*ptr)?;
            record_account_storage(store, &sub.node, nibbles, sub.prefix.len(), f)
        }
        Node::Raw { buf, off, len, .. } => {
            let n = parse_node_lazy(buf, *off, *len)?;
            record_account_storage(store, &n, nibbles, depth, f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha3::{Digest, Keccak256};
    use std::collections::BTreeMap;

    fn h(data: &[u8]) -> Key {
        let mut out = [0u8; 32];
        out.copy_from_slice(&Keccak256::digest(data));
        out
    }

    #[test]
    fn cursors_enumerate_exactly_the_inserted_state() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("c.flat");
        let mut db = FlatMpt::create(&path, Config::default()).unwrap();

        // Reference model: accounts with varied storage sizes (0, small packed,
        // large enough to split/promote records).
        let mut accounts: BTreeMap<Key, (u64, U256)> = BTreeMap::new();
        let mut storages: BTreeMap<Key, BTreeMap<Key, Vec<u8>>> = BTreeMap::new();
        let mut ops: Vec<(Key, StateOp)> = Vec::new();
        for a in 0..300u64 {
            let key = h(&a.to_be_bytes());
            let nonce = a + 1;
            let balance = U256::from(a * 7 + 1);
            accounts.insert(key, (nonce, balance));
            ops.push((key, StateOp::SetAccount {
                nonce,
                balance,
                code_hash: eth::EMPTY_CODE_HASH.0,
            }));
            let nslots = match a % 7 {
                0 => 0,
                1..=4 => (a % 5) as u64 + 1,
                _ => 800, // splits into multiple records / promotes
            };
            let mut slots = BTreeMap::new();
            for s in 0..nslots {
                let slot = h(&(a * 1_000_000 + s).to_be_bytes());
                let value = eth::storage_value_rlp(U256::from(s + 1));
                slots.insert(slot, value.clone());
                ops.push((key, StateOp::SetStorage { slot, value }));
            }
            if !slots.is_empty() {
                storages.insert(key, slots);
            }
        }
        db.apply_block(ops).unwrap();

        let scan = |db: &FlatMpt| {
            // Full account scan.
            let mut cur = db.account_cursor();
            let mut seen: BTreeMap<Key, (u64, U256)> = BTreeMap::new();
            let mut entry = cur.seek(&[0u8; 32]).unwrap();
            while let Some(e) = entry {
                seen.insert(e.key, (e.nonce, e.balance));
                // Per-account storage scan.
                let mut sc = db.storage_cursor(&e.key);
                let mut sseen: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
                let mut s = sc.seek(&[0u8; 32]).unwrap();
                while let Some((k, v)) = s {
                    sseen.insert(k, v);
                    s = sc.next().unwrap();
                }
                match storages.get(&e.key) {
                    Some(expect) => assert_eq!(&sseen, expect, "storage scan mismatch"),
                    None => assert!(sseen.is_empty(), "unexpected storage"),
                }
                entry = cur.next().unwrap();
            }
            assert_eq!(seen, accounts, "account scan mismatch");
        };

        // Warm (Mem/frontier mix).
        scan(&db);
        // Mid-keyspace seek lands on the true successor.
        let mid = *accounts.keys().nth(150).unwrap();
        let e = db.account_cursor().seek(&mid).unwrap().unwrap();
        assert_eq!(e.key, mid);
        // After persist + reopen (all Disk records).
        db.persist().unwrap();
        drop(db);
        let db = FlatMpt::open(&path).unwrap();
        scan(&db);
    }
}

// ---------------------------------------------------------------------------
// Branch-node cursor (reth TrieCursor backing): pre-order enumeration of
// branch positions with per-child masks and cached hashes.
// ---------------------------------------------------------------------------

/// A branch node surfaced to reth's trie cursor: absolute nibble path plus
/// per-child classification.
#[derive(Debug, Clone, PartialEq)]
pub struct TrieNodeEntry {
    pub path: Vec<u8>,
    /// Children present.
    pub state_mask: u16,
    /// Children whose subtrees contain further branch nodes (servable by this
    /// cursor at a deeper path).
    pub tree_mask: u16,
    /// Children whose 32-byte reference hashes are in `hashes` (inline <32B
    /// references are excluded; consumers recompute those from leaves).
    pub hash_mask: u16,
    /// Hashes for `hash_mask` bits, in ascending nibble order.
    pub hashes: Vec<Hash>,
}

/// Child classification for mask construction.
struct ChildInfo {
    hash: Option<Hash>,
    has_branch_below: bool,
}

fn classify_node_child(store: &FlatFile, memo: &RecordMemo, node: &Node) -> Result<ChildInfo> {
    Ok(match node {
        Node::Empty => ChildInfo { hash: None, has_branch_below: false },
        Node::Leaf { nref, .. } | Node::Account { nref, .. } => ChildInfo {
            hash: match nref {
                NodeRef::Hash(h) => Some(*h),
                NodeRef::Inline(_) => None,
            },
            has_branch_below: false,
        },
        Node::Extension { child, nref, .. } => {
            let below = classify_node_child(store, memo, child)?;
            ChildInfo {
                hash: match nref {
                    NodeRef::Hash(h) => Some(*h),
                    NodeRef::Inline(_) => None,
                },
                has_branch_below: below.has_branch_below || matches!(**child, Node::Branch { .. }),
            }
        }
        Node::Branch { nref, .. } => ChildInfo {
            hash: match nref {
                NodeRef::Hash(h) => Some(*h),
                NodeRef::Inline(_) => None,
            },
            has_branch_below: true,
        },
        Node::Overflow { ptr, root } => {
            let sub = memo.read(store, *ptr)?;
            let below = classify_node_child(store, memo, &sub.node)?;
            ChildInfo {
                hash: Some(*root),
                has_branch_below: matches!(sub.node, Node::Branch { .. }) || below.has_branch_below,
            }
        }
        Node::Raw { buf, off, len, .. } => {
            let n = parse_node_lazy(buf, *off, *len)?;
            classify_node_child(store, memo, &n)?
        }
    })
}

fn classify_ram_child(store: &FlatFile, memo: &RecordMemo, child: &RamChild) -> Result<ChildInfo> {
    Ok(match child {
        RamChild::Ram(sub) => ChildInfo {
            hash: Some(hash_ram_parallel(sub)),
            has_branch_below: matches!(**sub, RamNode::Branch { .. } | RamNode::Extension { .. }),
        },
        RamChild::Disk { ptr, root } => {
            let sub = memo.read(store, *ptr)?;
            let below = classify_node_child(store, memo, &sub.node)?;
            ChildInfo {
                hash: Some(*root),
                has_branch_below: below.has_branch_below || matches!(sub.node, Node::Branch { .. }),
            }
        }
        RamChild::Mem(m) => {
            m.touch();
            let sub = parse_payload_lazy(m.bytes.clone())?;
            let below = classify_node_child(store, memo, &sub.node)?;
            ChildInfo {
                hash: Some(m.root),
                has_branch_below: below.has_branch_below || matches!(sub.node, Node::Branch { .. }),
            }
        }
        RamChild::Account(_) => ChildInfo { hash: None, has_branch_below: false },
    })
}

/// Pre-order successor over BRANCH positions: smallest branch whose absolute
/// path is `>= target` (lexicographic nibble order; a parent precedes its
/// children). Returns the entry with masks/hashes built from child refs.
fn node_branch_seek(
    store: &FlatFile,
    memo: &RecordMemo,
    node: &Node,
    prefix: &mut Vec<u8>,
    target: &[u8],
) -> Result<Option<TrieNodeEntry>> {
    match node {
        Node::Empty | Node::Leaf { .. } | Node::Account { .. } => Ok(None),
        Node::Extension { path, child, .. } => {
            let plen = prefix.len();
            prefix.extend_from_slice(path);
            // The subtree below this extension covers paths prefixed by `prefix`;
            // skip it entirely only if even its largest path is < target.
            let out = if prefix_range_all_less(prefix, target) {
                None
            } else {
                node_branch_seek(store, memo, child, prefix, target)?
            };
            prefix.truncate(plen);
            Ok(out)
        }
        Node::Branch { children, .. } => {
            // Pre-order: this branch itself first (if >= target)...
            if lex_ge(prefix, target) {
                return build_branch_entry(store, memo, prefix, |i| {
                    children[i].as_deref().map(|c| classify_node_child(store, memo, c))
                })
                .map(Some);
            }
            // ...then children in nibble order.
            for i in 0..16u8 {
                let Some(child) = &children[i as usize] else { continue };
                prefix.push(i);
                let out = if prefix_range_all_less(prefix, target) {
                    None
                } else {
                    node_branch_seek(store, memo, child, prefix, target)?
                };
                prefix.pop();
                if out.is_some() {
                    return Ok(out);
                }
            }
            Ok(None)
        }
        Node::Overflow { ptr, .. } => {
            let sub = memo.read(store, *ptr)?;
            node_branch_seek(store, memo, &sub.node, prefix, target)
        }
        Node::Raw { buf, off, len, .. } => {
            let n = parse_node_lazy(buf, *off, *len)?;
            node_branch_seek(store, memo, &n, prefix, target)
        }
    }
}

/// Lexicographic `prefix >= target` for nibble paths (parent-before-child order).
fn lex_ge(path: &[u8], target: &[u8]) -> bool {
    let n = path.len().min(target.len());
    match path[..n].cmp(&target[..n]) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        // Equal prefix: shorter-or-equal path sorts first; path >= target iff
        // path is at least as long as target.
        std::cmp::Ordering::Equal => path.len() >= target.len(),
    }
}

/// True if EVERY path under `prefix` (prefix itself and its descendants) is
/// lexicographically `< target` — the subtree can be skipped.
fn prefix_range_all_less(prefix: &[u8], target: &[u8]) -> bool {
    let n = prefix.len().min(target.len());
    match prefix[..n].cmp(&target[..n]) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        // prefix is a prefix of target (or equal): descendants may reach target.
        std::cmp::Ordering::Equal => false,
    }
}

/// Assemble a [`TrieNodeEntry`] at `path` from per-child classification.
fn build_branch_entry(
    store: &FlatFile,
    memo: &RecordMemo,
    path: &[u8],
    child: impl Fn(usize) -> Option<Result<ChildInfo>>,
) -> Result<TrieNodeEntry> {
    let _ = (store, memo);
    let mut state_mask = 0u16;
    let mut tree_mask = 0u16;
    let mut hash_mask = 0u16;
    let mut hashes = Vec::new();
    for i in 0..16usize {
        let Some(info) = child(i) else { continue };
        let info = info?;
        state_mask |= 1 << i;
        if info.has_branch_below {
            tree_mask |= 1 << i;
        }
        if let Some(h) = info.hash {
            hash_mask |= 1 << i;
            hashes.push(h);
        }
    }
    Ok(TrieNodeEntry { path: path.to_vec(), state_mask, tree_mask, hash_mask, hashes })
}

/// Frontier-level branch-position successor (mirrors [`node_branch_seek`]).
fn ram_branch_seek(
    store: &FlatFile,
    memo: &RecordMemo,
    node: &RamNode,
    prefix: &mut Vec<u8>,
    target: &[u8],
) -> Result<Option<TrieNodeEntry>> {
    match node {
        RamNode::Empty => Ok(None),
        RamNode::Extension { path, child, .. } => {
            let plen = prefix.len();
            prefix.extend_from_slice(path);
            let out = if prefix_range_all_less(prefix, target) {
                None
            } else {
                ram_branch_seek(store, memo, child, prefix, target)?
            };
            prefix.truncate(plen);
            Ok(out)
        }
        RamNode::Branch { children, .. } => {
            if lex_ge(prefix, target) {
                return build_branch_entry(store, memo, prefix, |i| {
                    children[i].as_ref().map(|c| classify_ram_child(store, memo, c))
                })
                .map(Some);
            }
            for i in 0..16u8 {
                let Some(child) = &children[i as usize] else { continue };
                prefix.push(i);
                let out = if prefix_range_all_less(prefix, target) {
                    None
                } else {
                    match child {
                        RamChild::Ram(sub) => ram_branch_seek(store, memo, sub, prefix, target)?,
                        RamChild::Disk { ptr, .. } => {
                            let sub = memo.read(store, *ptr)?;
                            node_branch_seek(store, memo, &sub.node, prefix, target)?
                        }
                        RamChild::Mem(m) => {
                            m.touch();
                            let sub = parse_payload_lazy(m.bytes.clone())?;
                            node_branch_seek(store, memo, &sub.node, prefix, target)?
                        }
                        // Account-trie walk stops at account leaves; their storage
                        // has its own cursor.
                        RamChild::Account(_) => None,
                    }
                };
                prefix.pop();
                if out.is_some() {
                    return Ok(out);
                }
            }
            Ok(None)
        }
    }
}

/// Ordered cursor over the account trie's BRANCH nodes (reth TrieCursor
/// backing). `next()` continues strictly after the last returned path.
pub struct TrieNodeCursor<'a> {
    mpt: &'a FlatMpt,
    memo: RecordMemo,
    last: Option<Vec<u8>>,
}

impl<'a> TrieNodeCursor<'a> {
    /// Smallest branch with path `>=` the given nibble path.
    pub fn seek(&mut self, path_nibbles: &[u8]) -> Result<Option<TrieNodeEntry>> {
        let mut prefix = Vec::with_capacity(64);
        let out = ram_branch_seek(&self.mpt.store, &self.memo, &self.mpt.upper, &mut prefix, path_nibbles)?;
        self.last = out.as_ref().map(|e| e.path.clone());
        Ok(out)
    }

    pub fn next(&mut self) -> Result<Option<TrieNodeEntry>> {
        let Some(mut succ) = self.last.clone() else { return Ok(None) };
        // Smallest path strictly greater than `last` in pre-order = last ++ [0].
        succ.push(0);
        self.seek(&succ)
    }
}

/// Ordered cursor over one account's storage BRANCH nodes.
pub struct StorageTrieNodeCursor<'a> {
    mpt: &'a FlatMpt,
    memo: RecordMemo,
    account_key: Key,
    last: Option<Vec<u8>>,
}

impl<'a> StorageTrieNodeCursor<'a> {
    pub fn seek(&mut self, path_nibbles: &[u8]) -> Result<Option<TrieNodeEntry>> {
        let out = self
            .mpt
            .with_account_storage(&self.account_key, |store, storage| {
                let mut prefix = Vec::with_capacity(64);
                match storage {
                    StorageRef::Node(node) => node_branch_seek(store, &self.memo, node, &mut prefix, path_nibbles),
                    StorageRef::Ram(ram) => ram_branch_seek(store, &self.memo, ram, &mut prefix, path_nibbles),
                }
            })?
            .transpose()?
            .flatten();
        self.last = out.as_ref().map(|e| e.path.clone());
        Ok(out)
    }

    pub fn next(&mut self) -> Result<Option<TrieNodeEntry>> {
        let Some(mut succ) = self.last.clone() else { return Ok(None) };
        succ.push(0);
        self.seek(&succ)
    }
}

impl FlatMpt {
    pub fn trie_node_cursor(&self) -> TrieNodeCursor<'_> {
        TrieNodeCursor { mpt: self, memo: RecordMemo::new(), last: None }
    }
    pub fn storage_trie_node_cursor(&self, account_key: &Key) -> StorageTrieNodeCursor<'_> {
        StorageTrieNodeCursor { mpt: self, memo: RecordMemo::new(), account_key: *account_key, last: None }
    }
}
