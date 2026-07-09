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

use crate::cursor::RecordMemo;
use crate::{
    eth, hash_ram, key_nibbles, parse_node_lazy, parse_payload_lazy, ram_child_hash, FlatFile,
    FlatMpt, Hash, Key, Node, NodeRef, RamChild, RamNode,
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

impl FlatMpt {
    /// Reveal-path nodes for `keys` in the account trie (sorted or not).
    pub fn reveal_account_paths(&self, keys: &[Key]) -> Result<Vec<RevealNode>> {
        let memo = RecordMemo::new();
        let mut e = Emitter { store: &self.store, memo: &memo, out: Vec::new(), seen: Default::default() };
        let mut prefix = Vec::with_capacity(64);
        for k in keys {
            prefix.clear();
            let nibbles = key_nibbles(k);
            e.walk_ram(&self.upper, &mut prefix, &nibbles)?;
        }
        Ok(e.out)
    }

    /// Reveal-path nodes for `slots` within `account`'s storage trie.
    /// `Ok(None)` when the account doesn't exist or carries opaque storage.
    pub fn reveal_storage_paths(&self, account: &Key, slots: &[Key]) -> Result<Option<Vec<RevealNode>>> {
        let memo = RecordMemo::new();
        let out = self.with_account_storage(account, |store, storage| -> Result<Vec<RevealNode>> {
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
}
