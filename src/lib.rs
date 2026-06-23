use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub type Hash = [u8; 32];
pub type Key = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskPtr {
    pub offset: u64,
    pub len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub target_leaf_bytes: usize,
    pub max_leaf_bytes: usize,
    pub min_promote_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            target_leaf_bytes: 16 * 1024,
            max_leaf_bytes: 32 * 1024,
            min_promote_bytes: 8 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node {
    Empty,
    Leaf {
        key: Key,
        value_hash: Hash,
    },
    Extension {
        path: Vec<u8>,
        child: Box<Node>,
    },
    Branch {
        children: [Option<Box<Node>>; 16],
        value: Option<Hash>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskSubtree {
    prefix: Vec<u8>,
    node: Node,
    entries: Vec<(Key, Hash)>,
}

#[derive(Debug, Clone)]
enum RamChild {
    Ram(Box<RamNode>),
    Disk {
        ptr: DiskPtr,
        root: Hash,
        bytes: usize,
    },
}

#[derive(Debug, Clone)]
enum RamNode {
    Empty,
    Extension {
        path: Vec<u8>,
        child: Box<RamNode>,
    },
    Branch {
        children: [Option<RamChild>; 16],
        value: Option<Hash>,
    },
}

impl Default for RamNode {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Debug)]
pub struct FlatMpt {
    cfg: Config,
    file: File,
    upper: RamNode,
    values: BTreeMap<Key, Vec<u8>>,
}

impl FlatMpt {
    pub fn create(path: impl AsRef<Path>, cfg: Config) -> Result<Self> {
        if cfg.min_promote_bytes == 0 || cfg.min_promote_bytes > cfg.max_leaf_bytes {
            bail!("invalid split thresholds");
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;
        Ok(Self {
            cfg,
            file,
            upper: RamNode::Empty,
            values: BTreeMap::new(),
        })
    }

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<Hash> {
        let value_hash = hash_leaf_value(&value);
        self.values.insert(key, value);
        let cfg = self.cfg.clone();
        insert_ram(
            &mut self.file,
            &cfg,
            &mut self.upper,
            Vec::new(),
            key,
            value_hash,
        )?;
        self.file.flush()?;
        Ok(self.root())
    }

    pub fn get_value(&self, key: &Key) -> Option<&[u8]> {
        self.values.get(key).map(Vec::as_slice)
    }

    pub fn root(&self) -> Hash {
        hash_ram(&self.upper)
    }

    pub fn ram_nodes(&self) -> usize {
        count_ram_nodes(&self.upper)
    }

    pub fn disk_accesses_for_key(&mut self, key: &Key) -> Result<usize> {
        let nibbles = key_nibbles(key);
        let Some(ptr) = find_disk_ptr(&self.upper, &nibbles, 0) else {
            return Ok(0);
        };
        let subtree = read_subtree(&mut self.file, ptr)?;
        if subtree.entries.iter().any(|(k, _)| k == key) {
            Ok(1)
        } else {
            bail!("key not found in addressed disk subtree")
        }
    }
}

fn insert_ram(
    file: &mut File,
    cfg: &Config,
    node: &mut RamNode,
    prefix: Vec<u8>,
    key: Key,
    value_hash: Hash,
) -> Result<()> {
    let nibbles = key_nibbles(&key);
    match node {
        RamNode::Empty => {
            let idx = nibbles[prefix.len()] as usize;
            let subtree = subtree_from_entries(prefix, vec![(key, value_hash)]);
            let bytes = encoded_len(&subtree)?;
            let ptr = append_subtree(file, &subtree)?;
            *node = RamNode::Branch {
                children: empty_children(),
                value: None,
            };
            if let RamNode::Branch { children, .. } = node {
                children[idx] = Some(RamChild::Disk {
                    ptr,
                    root: hash_node(&subtree.node),
                    bytes,
                });
            }
            Ok(())
        }
        RamNode::Extension { path, child } => {
            let common = common_prefix(path, &nibbles[prefix.len()..]);
            if common < path.len() {
                let old = std::mem::replace(node, RamNode::Empty);
                let RamNode::Extension {
                    path: old_path,
                    child: old_child,
                } = old
                else {
                    unreachable!();
                };
                let mut branch = RamNode::Branch {
                    children: empty_children(),
                    value: None,
                };
                if let RamNode::Branch { children, .. } = &mut branch {
                    let old_idx = old_path[common] as usize;
                    let old_remainder = old_path[common + 1..].to_vec();
                    children[old_idx] = Some(RamChild::Ram(if old_remainder.is_empty() {
                        old_child
                    } else {
                        Box::new(RamNode::Extension {
                            path: old_remainder,
                            child: old_child,
                        })
                    }));

                    let new_idx = nibbles[prefix.len() + common] as usize;
                    let mut new_prefix = prefix.clone();
                    new_prefix.extend_from_slice(&old_path[..common]);
                    new_prefix.push(new_idx as u8);
                    let subtree = subtree_from_entries(new_prefix, vec![(key, value_hash)]);
                    let bytes = encoded_len(&subtree)?;
                    let ptr = append_subtree(file, &subtree)?;
                    children[new_idx] = Some(RamChild::Disk {
                        ptr,
                        root: hash_node(&subtree.node),
                        bytes,
                    });
                }
                *node = if common == 0 {
                    branch
                } else {
                    RamNode::Extension {
                        path: old_path[..common].to_vec(),
                        child: Box::new(branch),
                    }
                };
                Ok(())
            } else {
                let mut next_prefix = prefix;
                next_prefix.extend_from_slice(path);
                insert_ram(file, cfg, child, next_prefix, key, value_hash)
            }
        }
        RamNode::Branch { children, value } => {
            if prefix.len() == nibbles.len() {
                *value = Some(value_hash);
                return Ok(());
            }
            let idx = nibbles[prefix.len()] as usize;
            let mut child_prefix = prefix;
            child_prefix.push(idx as u8);
            match &mut children[idx] {
                Some(RamChild::Ram(child)) => {
                    insert_ram(file, cfg, child, child_prefix, key, value_hash)
                }
                Some(RamChild::Disk { ptr, root, bytes }) => {
                    let mut subtree = read_subtree(file, *ptr)?;
                    upsert_entry(&mut subtree.entries, key, value_hash);
                    subtree = subtree_from_entries(child_prefix.clone(), subtree.entries);
                    let new_bytes = encoded_len(&subtree)?;
                    if new_bytes <= cfg.max_leaf_bytes {
                        let new_ptr = append_subtree(file, &subtree)?;
                        *ptr = new_ptr;
                        *root = hash_node(&subtree.node);
                        *bytes = new_bytes;
                    } else {
                        children[idx] = Some(split_subtree(file, cfg, subtree)?);
                    }
                    Ok(())
                }
                None => {
                    let subtree = subtree_from_entries(child_prefix, vec![(key, value_hash)]);
                    let bytes = encoded_len(&subtree)?;
                    let ptr = append_subtree(file, &subtree)?;
                    children[idx] = Some(RamChild::Disk {
                        ptr,
                        root: hash_node(&subtree.node),
                        bytes,
                    });
                    Ok(())
                }
            }
        }
    }
}

fn split_subtree(file: &mut File, cfg: &Config, mut subtree: DiskSubtree) -> Result<RamChild> {
    let original_prefix_len = subtree.prefix.len();
    let shared = shared_prefix_after(&subtree.entries, original_prefix_len);
    if !shared.is_empty() {
        subtree.prefix.extend_from_slice(&shared);
        subtree = subtree_from_entries(subtree.prefix.clone(), subtree.entries);
    }

    let groups = group_by_next_nibble(&subtree.entries, subtree.prefix.len());
    let mut children = empty_children();
    let mut remainder = Vec::new();

    for (idx, entries) in groups.into_iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let mut child_prefix = subtree.prefix.clone();
        child_prefix.push(idx as u8);
        let child_subtree = subtree_from_entries(child_prefix.clone(), entries);
        let child_bytes = encoded_len(&child_subtree)?;
        if child_bytes > cfg.max_leaf_bytes {
            children[idx] = Some(split_subtree(file, cfg, child_subtree)?);
        } else if child_bytes >= cfg.min_promote_bytes {
            let ptr = append_subtree(file, &child_subtree)?;
            children[idx] = Some(RamChild::Disk {
                ptr,
                root: hash_node(&child_subtree.node),
                bytes: child_bytes,
            });
        } else {
            remainder.push((idx, child_subtree));
        }
    }

    for (idx, rem_subtree) in remainder {
        let ptr = append_subtree(file, &rem_subtree)?;
        children[idx] = Some(RamChild::Disk {
            ptr,
            root: hash_node(&rem_subtree.node),
            bytes: encoded_len(&rem_subtree)?,
        });
    }

    let branch = RamNode::Branch {
        children,
        value: None,
    };
    if shared.is_empty() {
        Ok(RamChild::Ram(Box::new(branch)))
    } else {
        Ok(RamChild::Ram(Box::new(RamNode::Extension {
            path: shared,
            child: Box::new(branch),
        })))
    }
}

fn subtree_from_entries(prefix: Vec<u8>, entries: Vec<(Key, Hash)>) -> DiskSubtree {
    let mut entries = entries;
    entries.sort_by_key(|(key, _)| *key);
    let node = build_node(&entries, prefix.len());
    DiskSubtree {
        prefix,
        node,
        entries,
    }
}

fn build_node(entries: &[(Key, Hash)], depth: usize) -> Node {
    if entries.is_empty() {
        return Node::Empty;
    }
    if entries.len() == 1 {
        let (key, value_hash) = entries[0];
        let path = key_nibbles(&key)[depth..].to_vec();
        return if path.is_empty() {
            Node::Leaf { key, value_hash }
        } else {
            Node::Extension {
                path,
                child: Box::new(Node::Leaf { key, value_hash }),
            }
        };
    }

    let nibbles: Vec<Vec<u8>> = entries.iter().map(|(key, _)| key_nibbles(key)).collect();
    let mut common = 0;
    while depth + common < 64 {
        let nibble = nibbles[0][depth + common];
        if nibbles.iter().all(|ks| ks[depth + common] == nibble) {
            common += 1;
        } else {
            break;
        }
    }
    if common > 0 {
        return Node::Extension {
            path: nibbles[0][depth..depth + common].to_vec(),
            child: Box::new(build_node(entries, depth + common)),
        };
    }

    let mut grouped: [Vec<(Key, Hash)>; 16] = std::array::from_fn(|_| Vec::new());
    for (i, entry) in entries.iter().enumerate() {
        let idx = nibbles[i].get(depth).copied().unwrap_or(0) as usize;
        grouped[idx].push(*entry);
    }
    let mut children = empty_box_children();
    for (idx, group) in grouped.into_iter().enumerate() {
        if !group.is_empty() {
            children[idx] = Some(Box::new(build_node(&group, depth + 1)));
        }
    }
    Node::Branch {
        children,
        value: None,
    }
}

fn append_subtree(file: &mut File, subtree: &DiskSubtree) -> Result<DiskPtr> {
    let payload = bincode::serialize(subtree)?;
    let offset = file.seek(SeekFrom::End(0))?;
    let len = payload.len() as u32;
    file.write_all(&len.to_le_bytes())?;
    file.write_all(&payload)?;
    Ok(DiskPtr {
        offset,
        len: len + 4,
    })
}

fn read_subtree(file: &mut File, ptr: DiskPtr) -> Result<DiskSubtree> {
    file.seek(SeekFrom::Start(ptr.offset))?;
    let mut len_bytes = [0; 4];
    file.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len + 4 != ptr.len as usize {
        bail!("flat-file record length mismatch");
    }
    let mut payload = vec![0; len];
    file.read_exact(&mut payload)?;
    Ok(bincode::deserialize(&payload)?)
}

fn encoded_len(subtree: &DiskSubtree) -> Result<usize> {
    Ok(bincode::serialize(subtree)?.len() + 4)
}

fn upsert_entry(entries: &mut Vec<(Key, Hash)>, key: Key, value_hash: Hash) {
    if let Some((_, old)) = entries.iter_mut().find(|(k, _)| k == &key) {
        *old = value_hash;
    } else {
        entries.push((key, value_hash));
    }
}

fn group_by_next_nibble(entries: &[(Key, Hash)], depth: usize) -> [Vec<(Key, Hash)>; 16] {
    let mut groups: [Vec<(Key, Hash)>; 16] = std::array::from_fn(|_| Vec::new());
    for entry in entries {
        let nibble = key_nibbles(&entry.0).get(depth).copied().unwrap_or(0) as usize;
        groups[nibble].push(*entry);
    }
    groups
}

fn find_disk_ptr(node: &RamNode, nibbles: &[u8], depth: usize) -> Option<DiskPtr> {
    match node {
        RamNode::Empty => None,
        RamNode::Extension { path, child } => {
            if nibbles.get(depth..depth + path.len()) == Some(path.as_slice()) {
                find_disk_ptr(child, nibbles, depth + path.len())
            } else {
                None
            }
        }
        RamNode::Branch { children, .. } => {
            let idx = *nibbles.get(depth)? as usize;
            match children[idx].as_ref()? {
                RamChild::Ram(child) => find_disk_ptr(child, nibbles, depth + 1),
                RamChild::Disk { ptr, .. } => Some(*ptr),
            }
        }
    }
}

fn hash_ram(node: &RamNode) -> Hash {
    match node {
        RamNode::Empty => keccak(&[0]),
        RamNode::Extension { path, child } => hash_join(1, path, &hash_ram(child)),
        RamNode::Branch { children, value } => {
            let mut bytes = vec![2];
            for child in children {
                let h = match child {
                    Some(RamChild::Ram(node)) => hash_ram(node),
                    Some(RamChild::Disk { root, .. }) => *root,
                    None => keccak(&[0]),
                };
                bytes.extend_from_slice(&h);
            }
            if let Some(v) = value {
                bytes.extend_from_slice(v);
            }
            keccak(&bytes)
        }
    }
}

fn hash_node(node: &Node) -> Hash {
    match node {
        Node::Empty => keccak(&[0]),
        Node::Leaf { key, value_hash } => {
            let mut bytes = vec![3];
            bytes.extend_from_slice(key);
            bytes.extend_from_slice(value_hash);
            keccak(&bytes)
        }
        Node::Extension { path, child } => hash_join(4, path, &hash_node(child)),
        Node::Branch { children, value } => {
            let mut bytes = vec![5];
            for child in children {
                bytes.extend_from_slice(
                    &child
                        .as_ref()
                        .map(|child| hash_node(child))
                        .unwrap_or_else(|| keccak(&[0])),
                );
            }
            if let Some(v) = value {
                bytes.extend_from_slice(v);
            }
            keccak(&bytes)
        }
    }
}

fn hash_join(tag: u8, path: &[u8], child: &Hash) -> Hash {
    let mut bytes = vec![tag, path.len() as u8];
    bytes.extend_from_slice(path);
    bytes.extend_from_slice(child);
    keccak(&bytes)
}

fn hash_leaf_value(value: &[u8]) -> Hash {
    let mut bytes = vec![6];
    bytes.extend_from_slice(value);
    keccak(&bytes)
}

fn keccak(bytes: &[u8]) -> Hash {
    Keccak256::digest(bytes).into()
}

fn key_nibbles(key: &Key) -> Vec<u8> {
    key.iter()
        .flat_map(|byte| [byte >> 4, byte & 0x0f])
        .collect()
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn shared_prefix_after(entries: &[(Key, Hash)], depth: usize) -> Vec<u8> {
    if entries.len() < 2 || depth >= 64 {
        return Vec::new();
    }
    let nibbles: Vec<Vec<u8>> = entries.iter().map(|(key, _)| key_nibbles(key)).collect();
    let mut len = 0;
    while depth + len < 64 {
        let nibble = nibbles[0][depth + len];
        if nibbles.iter().all(|ks| ks[depth + len] == nibble) {
            len += 1;
        } else {
            break;
        }
    }
    nibbles[0][depth..depth + len].to_vec()
}

fn empty_children() -> [Option<RamChild>; 16] {
    std::array::from_fn(|_| None)
}

fn empty_box_children() -> [Option<Box<Node>>; 16] {
    std::array::from_fn(|_| None)
}

fn count_ram_nodes(node: &RamNode) -> usize {
    match node {
        RamNode::Empty => 0,
        RamNode::Extension { child, .. } => 1 + count_ram_nodes(child),
        RamNode::Branch { children, .. } => {
            1 + children
                .iter()
                .filter_map(|child| match child {
                    Some(RamChild::Ram(node)) => Some(count_ram_nodes(node)),
                    _ => None,
                })
                .sum::<usize>()
        }
    }
}

pub fn hashed_key(input: impl AsRef<[u8]>) -> Key {
    keccak(input.as_ref())
}

pub fn hex(hash: Hash) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn assert_root_changes(old: Hash, new: Hash) -> Result<()> {
    if old == new {
        Err(anyhow!("root did not change"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn db(cfg: Config) -> FlatMpt {
        FlatMpt::create(NamedTempFile::new().unwrap().path(), cfg).unwrap()
    }

    #[test]
    fn insertion_updates_root_and_value_store() {
        let mut db = db(Config::default());
        let old = db.root();
        let key = hashed_key("alice");
        let new = db.insert(key, b"100".to_vec()).unwrap();
        assert_root_changes(old, new).unwrap();
        assert_eq!(db.get_value(&key), Some(&b"100"[..]));
        assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
    }

    #[test]
    fn repeated_insert_overwrites_value_hash() {
        let mut db = db(Config::default());
        let key = hashed_key("alice");
        let root1 = db.insert(key, b"100".to_vec()).unwrap();
        let root2 = db.insert(key, b"200".to_vec()).unwrap();
        assert_ne!(root1, root2);
        assert_eq!(db.get_value(&key), Some(&b"200"[..]));
        assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
    }

    #[test]
    fn splits_large_disk_leaf_into_ram_frontier() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        for i in 0..200u64 {
            db.insert(hashed_key(i.to_le_bytes()), vec![i as u8; 32])
                .unwrap();
        }
        assert!(db.ram_nodes() > 2);
        for i in [0u64, 33, 99, 199] {
            let key = hashed_key(i.to_le_bytes());
            assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
        }
    }

    #[test]
    fn long_shared_prefix_does_not_materialize_many_ram_nodes() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        for i in 0..80u8 {
            let mut key = [0u8; 32];
            key[0] = 0xab;
            key[1] = 0xcd;
            key[2] = 0xef;
            key[31] = i;
            db.insert(key, vec![i; 32]).unwrap();
        }
        assert!(db.ram_nodes() < 20, "ram_nodes={}", db.ram_nodes());
    }
}
