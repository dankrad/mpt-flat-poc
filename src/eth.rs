//! Ethereum-exact MPT encoding primitives: RLP node encoding, hex-prefix (compact)
//! path encoding, the "< 32 bytes ⇒ inline, else keccak256(RLP)" node-reference rule,
//! and a small **reference** trie-root builder used to validate those primitives
//! against the official `ethereum/tests` vectors.
//!
//! This is the encoding layer that the flat-file trie's hashing will be rewritten to
//! use (Phase 2+). The reference builder here (`root`/`secure_root`) is an oracle —
//! it constructs a fresh logical trie from a key/value set and hashes it Ethereum-style
//! — not the production trie.

use alloy_primitives::{keccak256, B256};
use std::collections::BTreeMap;

/// Hex-prefix (compact) encoding of a nibble path. The first nibble is a flag:
/// `0` extension-even, `1` extension-odd, `2` leaf-even, `3` leaf-odd; even paths get
/// a `0` padding nibble after the flag so the result is whole bytes.
pub fn hex_prefix(nibbles: &[u8], leaf: bool) -> Vec<u8> {
    let odd = nibbles.len() % 2 == 1;
    let flag: u8 = if leaf { 2 } else { 0 };
    let mut out = Vec::with_capacity(nibbles.len() / 2 + 1);
    if odd {
        out.push(((flag | 1) << 4) | nibbles[0]);
        for pair in nibbles[1..].chunks(2) {
            out.push((pair[0] << 4) | pair[1]);
        }
    } else {
        out.push(flag << 4);
        for pair in nibbles.chunks(2) {
            out.push((pair[0] << 4) | pair[1]);
        }
    }
    out
}

/// RLP-encode a byte string.
pub(crate) fn rlp_string(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + 9);
    alloy_rlp::Encodable::encode(&bytes, &mut out);
    out
}

/// RLP-encode a list whose items are already RLP-encoded elements.
pub(crate) fn rlp_list(items: &[Vec<u8>]) -> Vec<u8> {
    let payload: usize = items.iter().map(|i| i.len()).sum();
    let mut out = Vec::with_capacity(payload + 9);
    alloy_rlp::Header { list: true, payload_length: payload }.encode(&mut out);
    for it in items {
        out.extend_from_slice(it);
    }
    out
}

/// A node's *reference* as embedded in its parent: the node's RLP inlined verbatim if
/// it is `< 32` bytes, otherwise `keccak256(RLP)` encoded as a 32-byte RLP string.
fn node_ref(node_rlp: &[u8]) -> Vec<u8> {
    if node_rlp.len() < 32 {
        node_rlp.to_vec()
    } else {
        rlp_string(keccak256(node_rlp).as_slice())
    }
}

/// Longest common prefix length across a sorted, non-empty entry set (first vs last).
fn common_prefix_len(entries: &[(&[u8], &[u8])]) -> usize {
    let a = entries[0].0;
    let b = entries[entries.len() - 1].0;
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

/// RLP of the trie node spanning `entries` (sorted by nibble-key, distinct, non-empty
/// values). Recursively builds leaf / extension / branch nodes with Ethereum encoding.
fn encode_node(entries: &[(&[u8], &[u8])]) -> Vec<u8> {
    if entries.len() == 1 {
        let (k, v) = entries[0];
        return rlp_list(&[rlp_string(&hex_prefix(k, true)), rlp_string(v)]);
    }
    let cp = common_prefix_len(entries);
    if cp > 0 {
        let stripped: Vec<(&[u8], &[u8])> = entries.iter().map(|(k, v)| (&k[cp..], *v)).collect();
        let child = encode_node(&stripped);
        return rlp_list(&[rlp_string(&hex_prefix(&entries[0].0[..cp], false)), node_ref(&child)]);
    }
    // Branch: 16 child slots + a terminal value slot.
    let mut items: Vec<Vec<u8>> = Vec::with_capacity(17);
    for i in 0..16u8 {
        let bucket: Vec<(&[u8], &[u8])> = entries
            .iter()
            .filter(|(k, _)| !k.is_empty() && k[0] == i)
            .map(|(k, v)| (&k[1..], *v))
            .collect();
        if bucket.is_empty() {
            items.push(rlp_string(&[]));
        } else {
            items.push(node_ref(&encode_node(&bucket)));
        }
    }
    // Value at this branch (a key that terminates exactly here). For a secure trie
    // (all keys the same length) this is always empty.
    match entries.iter().find(|(k, _)| k.is_empty()) {
        Some((_, v)) => items.push(rlp_string(v)),
        None => items.push(rlp_string(&[])),
    }
    rlp_list(&items)
}

fn key_nibbles(key: &[u8]) -> Vec<u8> {
    let mut n = Vec::with_capacity(key.len() * 2);
    for b in key {
        n.push(b >> 4);
        n.push(b & 0x0f);
    }
    n
}

/// Ethereum trie root over `(key, value)` pairs (keys used verbatim → nibbles; last
/// write wins; empty value = absent). This is the non-secure trie (matches
/// `ethereum/tests` `TrieTests/trietest.json`).
pub fn root(entries: &[(Vec<u8>, Vec<u8>)]) -> B256 {
    let mut map: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for (k, v) in entries {
        if v.is_empty() {
            map.remove(&key_nibbles(k));
        } else {
            map.insert(key_nibbles(k), v.clone());
        }
    }
    if map.is_empty() {
        return keccak256([0x80u8]); // keccak256(RLP("")) — empty trie root
    }
    let refs: Vec<(&[u8], &[u8])> = map.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    keccak256(encode_node(&refs))
}

/// Secure-trie root: keys are `keccak256`-hashed before insertion (Ethereum state and
/// storage tries). Values are used verbatim (caller RLP-encodes accounts / storage).
pub fn secure_root(entries: &[(Vec<u8>, Vec<u8>)]) -> B256 {
    let hashed: Vec<(Vec<u8>, Vec<u8>)> = entries
        .iter()
        .map(|(k, v)| (keccak256(k).to_vec(), v.clone()))
        .collect();
    root(&hashed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    fn hx(s: &str) -> Vec<u8> {
        alloy_primitives::hex::decode(s).unwrap()
    }

    #[test]
    fn empty_trie_root() {
        // keccak256(RLP("")) = the canonical empty MPT root (also Ethereum's
        // EMPTY_ROOT / empty storage-trie root).
        let empty: B256 = "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"
            .parse()
            .unwrap();
        assert_eq!(root(&[]), empty);
        assert_eq!(keccak256([0x80u8]), empty);
    }

    #[test]
    fn official_secure_trie_accounts() {
        // ethereum/tests TrieTests/hex_encoded_securetrie_test.json "test1":
        // address -> RLP(account); keys are keccak-hashed (secure trie). This is the
        // Ethereum state-trie shape exactly.
        let pairs = [
            ("a94f5374fce5edbc8e2a8697c15331677e6ebf0b", "f848018405f446a7a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ("095e7baea6a6c7c4c2dfeb977efac326af552d87", "f8440101a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a004bccc5d94f4d1f99aab44369a910179931772f2a5c001c3229f57831c102769"),
            ("d2571607e241ecf590ed94b12d87c94babe36db6", "f8440180a0ba4b47865c55a341a4a78759bb913cd15c3ee8eaf30a62fa8d1c8863113d84e8a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ("62c01474f089b07dae603491675dc5b5748f7049", "f8448080a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ("2adc25665018aa1fe0e6bc666dac8fc2697ff9ba", "f8478083019a59a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a0c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        ];
        let entries: Vec<(Vec<u8>, Vec<u8>)> = pairs.iter().map(|(k, v)| (hx(k), hx(v))).collect();
        let want: B256 = "0x730a444e08ab4b8dee147c9b232fc52d34a223d600031c1e9d25bfc985cbd797"
            .parse()
            .unwrap();
        assert_eq!(secure_root(&entries), want);
    }

    #[test]
    fn hex_prefix_flags() {
        assert_eq!(hex_prefix(&[], true), vec![0x20]); // leaf, even (empty)
        assert_eq!(hex_prefix(&[0xa, 0xb], false), vec![0x00, 0xab]); // ext, even
        assert_eq!(hex_prefix(&[0xa], true), vec![0x3a]); // leaf, odd
        assert_eq!(hex_prefix(&[0x1, 0x2, 0x3], false), vec![0x11, 0x23]); // ext, odd
    }

    #[test]
    fn official_trietest_emptyvalues() {
        // ethereum/tests TrieTests/trietest.json "emptyValues", after the deletes are
        // applied the live set is {do, horse, doge, dog}.
        let entries = vec![
            (b("do"), b("verb")),
            (b("horse"), b("stallion")),
            (b("doge"), b("coin")),
            (b("dog"), b("puppy")),
        ];
        let want: B256 = "0x5991bb8c6514148a29db676a14ac506cd2cd5775ace63c30a4fe457715e9ac84"
            .parse()
            .unwrap();
        assert_eq!(root(&entries), want);
    }
}

/// Minimal hex helper for tests / diagnostics.
pub fn hex(bytes: impl AsRef<[u8]>) -> String {
    let mut s = String::with_capacity(bytes.as_ref().len() * 2);
    for b in bytes.as_ref() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
