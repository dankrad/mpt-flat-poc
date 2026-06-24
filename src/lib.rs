use anyhow::{Result, anyhow, bail};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

/// Number of buffered values before the overlay is flushed to RocksDB as one
/// `WriteBatch`, amortizing per-`put` overhead.
const VALUE_BATCH: usize = 256;

pub type Hash = [u8; 32];
pub type Key = [u8; 32];

/// Opt-in wall-clock profiler.
///
/// Each [`Cat`] is a *leaf* primitive — none of the instrumented regions nests
/// inside another — so the buckets are mutually exclusive and their sum is the
/// time spent in hashing / serialization / IO. Whatever the timed workload's
/// wall clock has left over is "trie/CPU": nibble math, tree restructuring,
/// buffer assembly, free-list bookkeeping (and the profiler's own overhead).
///
/// Enabled by the `profiling` cargo feature; otherwise every hook is a no-op
/// zero-sized guard and compiles away entirely.
pub mod prof {
    /// Human-readable labels, indexed by `Cat as usize`.
    pub const CATS: [&str; 8] = [
        "keccak (hashing)",
        "bincode serialize",
        "bincode deserialize",
        "flat-file read (syscall)",
        "flat-file write (syscall)",
        "flat-file flush",
        "rocksdb put (value store)",
        "rocksdb get (value store)",
    ];

    #[derive(Clone, Copy)]
    pub enum Cat {
        Keccak = 0,
        Serialize = 1,
        Deserialize = 2,
        FileRead = 3,
        FileWrite = 4,
        Flush = 5,
        ValuePut = 6,
        ValueGet = 7,
    }

    #[cfg(feature = "profiling")]
    mod imp {
        use super::Cat;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Instant;

        pub const ENABLED: bool = true;

        static NANOS: [AtomicU64; 8] = [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];
        static COUNT: [AtomicU64; 8] = [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];

        pub struct Guard {
            cat: usize,
            start: Instant,
        }

        impl Drop for Guard {
            fn drop(&mut self) {
                let elapsed = self.start.elapsed().as_nanos() as u64;
                NANOS[self.cat].fetch_add(elapsed, Ordering::Relaxed);
                COUNT[self.cat].fetch_add(1, Ordering::Relaxed);
            }
        }

        /// Start timing `cat`; the returned guard records on drop.
        pub fn scope(cat: Cat) -> Guard {
            Guard {
                cat: cat as usize,
                start: Instant::now(),
            }
        }

        pub fn reset() {
            for i in 0..8 {
                NANOS[i].store(0, Ordering::Relaxed);
                COUNT[i].store(0, Ordering::Relaxed);
            }
        }

        /// Per-category `(total_nanos, call_count)`.
        pub fn snapshot() -> [(u64, u64); 8] {
            std::array::from_fn(|i| {
                (
                    NANOS[i].load(Ordering::Relaxed),
                    COUNT[i].load(Ordering::Relaxed),
                )
            })
        }

        use std::cell::RefCell;
        thread_local! {
            // `Some` while an audit window is open; each keccak output is logged.
            static AUDIT: RefCell<Option<Vec<[u8; 32]>>> = const { RefCell::new(None) };
        }

        /// Begin recording keccak outputs (clears any prior window).
        pub fn audit_start() {
            AUDIT.with(|a| *a.borrow_mut() = Some(Vec::new()));
        }

        /// End recording and return the outputs produced since `audit_start`.
        pub fn audit_take() -> Vec<[u8; 32]> {
            AUDIT.with(|a| a.borrow_mut().take().unwrap_or_default())
        }

        #[inline]
        pub fn record(output: [u8; 32]) {
            AUDIT.with(|a| {
                if let Some(log) = a.borrow_mut().as_mut() {
                    log.push(output);
                }
            });
        }
    }

    #[cfg(not(feature = "profiling"))]
    mod imp {
        use super::Cat;

        pub const ENABLED: bool = false;

        pub struct Guard;

        #[inline(always)]
        pub fn scope(_: Cat) -> Guard {
            Guard
        }
        pub fn reset() {}
        pub fn snapshot() -> [(u64, u64); 8] {
            [(0, 0); 8]
        }
        #[inline(always)]
        pub fn record(_: [u8; 32]) {}
        pub fn audit_start() {}
        pub fn audit_take() -> Vec<[u8; 32]> {
            Vec::new()
        }
    }

    pub use imp::{ENABLED, Guard, audit_start, audit_take, record, reset, scope, snapshot};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskPtr {
    pub offset: u64,
    pub len: u32,
}

/// Tracks reclaimed regions of the flat file so new records can be placed into
/// holes left by rewritten/split subtrees instead of always extending the file.
///
/// Regions are kept non-overlapping and coalesced (adjacent free regions are
/// merged on `free`). Two indexes are maintained in lock-step: `by_offset` for
/// coalescing with neighbours, and `by_size` (keyed by `(len, offset)`) so that
/// best-fit allocation is O(log n) rather than a linear scan — this matters a
/// lot once fragmentation grows the list to tens of thousands of regions.
///
/// Only `by_offset` is serialized; `by_size` is rebuilt via [`reindex`] on load.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct FreeList {
    /// offset -> length of each free region.
    by_offset: BTreeMap<u64, u32>,
    /// (length, offset) of each free region, for size-ordered best-fit lookup.
    #[serde(skip)]
    by_size: BTreeSet<(u32, u64)>,
}

impl FreeList {
    fn insert_region(&mut self, offset: u64, len: u32) {
        self.by_offset.insert(offset, len);
        self.by_size.insert((len, offset));
    }

    fn remove_region(&mut self, offset: u64, len: u32) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(len, offset));
    }

    /// Reserve `need` bytes from a free region if one is large enough.
    /// Returns the offset of the allocation, leaving any remainder free.
    fn alloc(&mut self, need: u32) -> Option<u64> {
        // Best fit: smallest region with len >= need, in O(log n).
        let (len, offset) = self.by_size.range((need, 0)..).next().copied()?;
        self.remove_region(offset, len);
        let remainder = len - need;
        if remainder > 0 {
            self.insert_region(offset + need as u64, remainder);
        }
        Some(offset)
    }

    /// Mark `[offset, offset + len)` as free, coalescing with neighbours.
    fn free(&mut self, offset: u64, len: u32) {
        let mut start = offset;
        let mut size = len as u64;

        // Merge with the region immediately preceding this one.
        let pred = self
            .by_offset
            .range(..start)
            .next_back()
            .map(|(&off, &len)| (off, len));
        if let Some((prev_off, prev_len)) = pred {
            if prev_off + prev_len as u64 == start {
                start = prev_off;
                size += prev_len as u64;
                self.remove_region(prev_off, prev_len);
            }
        }
        // Merge with the region immediately following this one.
        if let Some(next_len) = self.by_offset.get(&(start + size)).copied() {
            self.remove_region(start + size, next_len);
            size += next_len as u64;
        }

        self.insert_region(start, size as u32);
    }

    fn total(&self) -> u64 {
        self.by_offset.values().map(|&len| len as u64).sum()
    }

    fn region_count(&self) -> usize {
        self.by_offset.len()
    }

    /// Rebuild the size index from `by_offset` (after deserialization).
    fn reindex(&mut self) {
        self.by_size = self.by_offset.iter().map(|(&off, &len)| (len, off)).collect();
    }
}

/// Append-mostly flat file of serialized [`DiskSubtree`] records, with a
/// [`FreeList`] so that space freed by rewrites can be reused.
#[derive(Debug)]
struct FlatFile {
    file: File,
    free: FreeList,
    /// High-water mark: logical end of the file, where fresh appends land.
    end: u64,
}

impl FlatFile {
    fn new(file: File) -> Self {
        Self {
            file,
            free: FreeList::default(),
            end: 0,
        }
    }

    /// Store an already-serialized subtree payload, preferring a reclaimed free
    /// region over extending the file. Record layout is `[len: u32 LE][payload]`.
    /// Callers serialize once (via [`serialize_subtree`]) and reuse the size for
    /// their rewrite-vs-split decision, so the blob is never serialized twice.
    fn write_payload(&mut self, payload: &[u8]) -> Result<DiskPtr> {
        let len = payload.len() as u32;
        let total = len + 4;
        let offset = match self.free.alloc(total) {
            Some(offset) => offset,
            None => {
                let offset = self.end;
                self.end += total as u64;
                offset
            }
        };
        let _g = prof::scope(prof::Cat::FileWrite);
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(payload)?;
        Ok(DiskPtr { offset, len: total })
    }

    fn read(&mut self, ptr: DiskPtr) -> Result<DiskSubtree> {
        let payload = {
            let _g = prof::scope(prof::Cat::FileRead);
            self.file.seek(SeekFrom::Start(ptr.offset))?;
            let mut len_bytes = [0; 4];
            self.file.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            if len + 4 != ptr.len as usize {
                bail!("flat-file record length mismatch");
            }
            let mut payload = vec![0; len];
            self.file.read_exact(&mut payload)?;
            payload
        };
        let _g = prof::scope(prof::Cat::Deserialize);
        Ok(bincode::deserialize(&payload)?)
    }

    fn free(&mut self, ptr: DiskPtr) {
        self.free.free(ptr.offset, ptr.len);
    }

    fn flush(&mut self) -> Result<()> {
        let _g = prof::scope(prof::Cat::Flush);
        Ok(self.file.flush()?)
    }

    /// Flush and fsync the flat file to disk (used before a manifest checkpoint
    /// so the manifest never references data that hasn't reached storage).
    fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }
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

// Each non-trivial node caches its own Merkle hash, computed once at
// construction and persisted to disk. This lets a rewrite recompute only the
// hashes on the path it actually changed (see `node_insert`), instead of
// re-hashing the whole subtree. All keys are full 64-nibble paths, so leaves
// only ever sit at depth 64 and branches never carry a value.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node {
    Empty,
    Leaf {
        key: Key,
        value_hash: Hash,
        hash: Hash,
    },
    Extension {
        path: Vec<u8>,
        child: Box<Node>,
        hash: Hash,
    },
    Branch {
        children: [Option<Box<Node>>; 16],
        value: Option<Hash>,
        hash: Hash,
    },
}

// A subtree is fully described by its `node`; the flat list of entries it holds
// is derived on demand (`node_entries`) when a split needs to regroup them.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskSubtree {
    prefix: Vec<u8>,
    node: Node,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RamChild {
    Ram(Box<RamNode>),
    Disk {
        ptr: DiskPtr,
        root: Hash,
        bytes: usize,
    },
}

// RAM-frontier nodes cache their hash in an interior-mutable `Cell` so that
// `hash_ram`/`root` can memoize. An insert invalidates only the caches along
// the path it touches (`invalidate_ram`), so recomputing the root re-hashes
// just that path — every other node returns its cached value, and disk children
// contribute their already-cached `root`.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum RamNode {
    Empty,
    Extension {
        path: Vec<u8>,
        child: Box<RamNode>,
        hash: Cell<Option<Hash>>,
    },
    Branch {
        children: [Option<RamChild>; 16],
        value: Option<Hash>,
        hash: Cell<Option<Hash>>,
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
    store: FlatFile,
    upper: RamNode,
    /// Disk-backed key -> value store. Holds the actual values; the trie only
    /// ever deals in `value_hash`.
    values: DB,
    /// Buffer of values not yet flushed to `values`. Flushed in one batch every
    /// `VALUE_BATCH` inserts (and on `persist`); read by `get_value` first so
    /// reads always observe the latest write.
    overlay: HashMap<Key, Vec<u8>>,
    /// Path of the flat file; the value store and manifest are derived from it.
    path: PathBuf,
}

/// On-disk checkpoint of everything that otherwise lives only in RAM: the trie
/// frontier, the flat-file free list, and the high-water mark. Together with the
/// flat file and the value store this is enough to fully reconstruct a `FlatMpt`.
#[derive(Serialize)]
struct ManifestRef<'a> {
    cfg: &'a Config,
    upper: &'a RamNode,
    free: &'a FreeList,
    end: u64,
}

#[derive(Deserialize)]
struct Manifest {
    cfg: Config,
    upper: RamNode,
    free: FreeList,
    end: u64,
}

impl FlatMpt {
    pub fn create(path: impl AsRef<Path>, cfg: Config) -> Result<Self> {
        if cfg.min_promote_bytes == 0 || cfg.min_promote_bytes > cfg.max_leaf_bytes {
            bail!("invalid split thresholds");
        }
        let path = path.as_ref();
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;

        // RocksDB instance lives in a sibling directory. `create` is a fresh
        // start, so discard any leftover store from a previous run at this path.
        let values_path = values_path(path);
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let _ = DB::destroy(&opts, &values_path);
        let values = DB::open(&opts, &values_path)?;

        Ok(Self {
            cfg,
            store: FlatFile::new(file),
            upper: RamNode::Empty,
            values,
            overlay: HashMap::new(),
            path: path.to_path_buf(),
        })
    }

    /// Reopen a database previously written with [`FlatMpt::persist`]. Reattaches
    /// to the existing flat file and value store (no truncation) and restores the
    /// RAM frontier and free list from the `.meta` manifest.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let meta = meta_path(path);
        let bytes = std::fs::read(&meta)
            .map_err(|e| anyhow!("no manifest at {}: {e}", meta.display()))?;
        let Manifest {
            cfg,
            upper,
            mut free,
            end,
        } = bincode::deserialize(&bytes)?;
        // The size index isn't serialized; rebuild it from the offset map.
        free.reindex();

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let values = DB::open(&opts, values_path(path))?;

        Ok(Self {
            cfg,
            store: FlatFile { file, free, end },
            upper,
            values,
            overlay: HashMap::new(),
            path: path.to_path_buf(),
        })
    }

    /// Flush buffered values to RocksDB and the flat file's writer. Call this to
    /// make all preceding inserts visible in the value store without a full
    /// [`persist`](Self::persist) checkpoint.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_values()?;
        self.store.flush()
    }

    /// Flush the buffered value overlay to RocksDB as a single batch.
    fn flush_values(&mut self) -> Result<()> {
        if self.overlay.is_empty() {
            return Ok(());
        }
        let _g = prof::scope(prof::Cat::ValuePut);
        let mut batch = WriteBatch::default();
        for (key, value) in &self.overlay {
            batch.put(key, value);
        }
        self.values.write(batch)?;
        self.overlay.clear();
        Ok(())
    }

    /// Checkpoint the in-RAM state to disk so the database can later be reopened
    /// with [`FlatMpt::open`]. Flushes buffered values, fsyncs the flat file, then
    /// writes the manifest atomically (temp file + rename) so a crash can't leave
    /// a torn manifest.
    pub fn persist(&mut self) -> Result<()> {
        self.flush_values()?;
        self.store.sync()?;
        let manifest = ManifestRef {
            cfg: &self.cfg,
            upper: &self.upper,
            free: &self.store.free,
            end: self.store.end,
        };
        let bytes = bincode::serialize(&manifest)?;

        let meta = meta_path(&self.path);
        let mut tmp = meta.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &meta)?;
        Ok(())
    }

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<Hash> {
        let value_hash = hash_leaf_value(&value);
        self.overlay.insert(key, value);
        if self.overlay.len() >= VALUE_BATCH {
            self.flush_values()?;
        }
        let cfg = self.cfg.clone();
        insert_ram(
            &mut self.store,
            &cfg,
            &mut self.upper,
            Vec::new(),
            key,
            value_hash,
        )?;
        self.store.flush()?;
        Ok(self.root())
    }

    pub fn get_value(&self, key: &Key) -> Result<Option<Vec<u8>>> {
        let _g = prof::scope(prof::Cat::ValueGet);
        // Buffered writes win over what's already in RocksDB.
        if let Some(value) = self.overlay.get(key) {
            return Ok(Some(value.clone()));
        }
        Ok(self.values.get(key)?)
    }

    pub fn root(&self) -> Hash {
        hash_ram(&self.upper)
    }

    pub fn ram_nodes(&self) -> usize {
        count_ram_nodes(&self.upper)
    }

    /// Logical size of the flat file (high-water mark). Stays flat across
    /// rewrites when freed space is reused.
    pub fn flat_file_len(&self) -> u64 {
        self.store.end
    }

    /// Total bytes currently held in the flat file's free list.
    pub fn free_bytes(&self) -> u64 {
        self.store.free.total()
    }

    /// Number of distinct free regions tracked in the flat file.
    pub fn free_regions(&self) -> usize {
        self.store.free.region_count()
    }

    pub fn disk_accesses_for_key(&mut self, key: &Key) -> Result<usize> {
        let nibbles = key_nibbles(key);
        let Some(ptr) = find_disk_ptr(&self.upper, &nibbles, 0) else {
            return Ok(0);
        };
        let subtree = self.store.read(ptr)?;
        if node_contains(&subtree.node, key) {
            Ok(1)
        } else {
            bail!("key not found in addressed disk subtree")
        }
    }
}

/// Sibling path for the RocksDB value store, e.g. `db.flat` -> `db.flat.values`.
fn values_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".values");
    path.with_file_name(name)
}

/// Sibling path for the manifest, e.g. `db.flat` -> `db.flat.meta`.
fn meta_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".meta");
    path.with_file_name(name)
}

fn insert_ram(
    store: &mut FlatFile,
    cfg: &Config,
    node: &mut RamNode,
    prefix: Vec<u8>,
    key: Key,
    value_hash: Hash,
) -> Result<()> {
    let nibbles = key_nibbles(&key);
    // This node (or its subtree) is about to change, so its cached hash is stale.
    invalidate_ram(node);
    match node {
        RamNode::Empty => {
            let idx = nibbles[prefix.len()] as usize;
            // Build the leaf at the branch-slot depth (prefix + slot nibble), the
            // same depth every other code path uses, so the representation — and
            // therefore the hash — is independent of how the leaf was created.
            let mut child_prefix = prefix;
            child_prefix.push(idx as u8);
            let subtree = subtree_from_entries(child_prefix, vec![(key, value_hash)]);
            let (payload, bytes) = serialize_subtree(&subtree)?;
            let ptr = store.write_payload(&payload)?;
            *node = RamNode::Branch {
                children: empty_children(),
                value: None,
                hash: Cell::new(None),
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
        RamNode::Extension { path, child, .. } => {
            let common = common_prefix(path, &nibbles[prefix.len()..]);
            if common < path.len() {
                let old = std::mem::replace(node, RamNode::Empty);
                let RamNode::Extension {
                    path: old_path,
                    child: old_child,
                    ..
                } = old
                else {
                    unreachable!();
                };
                let mut branch = RamNode::Branch {
                    children: empty_children(),
                    value: None,
                    hash: Cell::new(None),
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
                            hash: Cell::new(None),
                        })
                    }));

                    let new_idx = nibbles[prefix.len() + common] as usize;
                    let mut new_prefix = prefix.clone();
                    new_prefix.extend_from_slice(&old_path[..common]);
                    new_prefix.push(new_idx as u8);
                    let subtree = subtree_from_entries(new_prefix, vec![(key, value_hash)]);
                    let (payload, bytes) = serialize_subtree(&subtree)?;
                    let ptr = store.write_payload(&payload)?;
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
                        hash: Cell::new(None),
                    }
                };
                Ok(())
            } else {
                let mut next_prefix = prefix;
                next_prefix.extend_from_slice(path);
                insert_ram(store, cfg, child, next_prefix, key, value_hash)
            }
        }
        RamNode::Branch {
            children, value, ..
        } => {
            if prefix.len() == nibbles.len() {
                *value = Some(value_hash);
                return Ok(());
            }
            let idx = nibbles[prefix.len()] as usize;
            let mut child_prefix = prefix;
            child_prefix.push(idx as u8);
            match &mut children[idx] {
                Some(RamChild::Ram(child)) => {
                    insert_ram(store, cfg, child, child_prefix, key, value_hash)
                }
                Some(RamChild::Disk { ptr, root, bytes }) => {
                    let mut subtree = store.read(*ptr)?;
                    let old_ptr = *ptr;
                    // Incremental: re-hash only the path from this leaf's root down
                    // to the changed entry; untouched siblings keep cached hashes.
                    node_insert(&mut subtree.node, subtree.prefix.len(), key, value_hash);
                    debug_assert_eq!(
                        hash_node(&subtree.node),
                        hash_node(&build_node(
                            &node_entries(&subtree.node),
                            subtree.prefix.len()
                        )),
                        "incremental node hash diverged from a full rebuild",
                    );
                    let (payload, new_bytes) = serialize_subtree(&subtree)?;
                    // The old record is now dead; reclaim it before writing so the
                    // rewrite can land back in the same region when it still fits.
                    store.free(old_ptr);
                    if new_bytes <= cfg.max_leaf_bytes {
                        let new_ptr = store.write_payload(&payload)?;
                        *ptr = new_ptr;
                        *root = hash_node(&subtree.node);
                        *bytes = new_bytes;
                    } else {
                        children[idx] = Some(split_subtree(store, cfg, subtree)?);
                    }
                    Ok(())
                }
                None => {
                    let subtree = subtree_from_entries(child_prefix, vec![(key, value_hash)]);
                    let (payload, bytes) = serialize_subtree(&subtree)?;
                    let ptr = store.write_payload(&payload)?;
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

fn split_subtree(store: &mut FlatFile, cfg: &Config, subtree: DiskSubtree) -> Result<RamChild> {
    let leaves = node_entries(&subtree.node);
    // Absorb any nibbles all entries still share into the prefix (becomes a RAM
    // extension), then fan the rest out by their next nibble.
    let shared = shared_prefix_after(&leaves, subtree.prefix.len());
    let mut prefix = subtree.prefix;
    prefix.extend_from_slice(&shared);

    let groups = group_by_next_nibble(&leaves, prefix.len());
    let mut children = empty_children();
    let mut remainder = Vec::new();

    for (idx, entries) in groups.into_iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let mut child_prefix = prefix.clone();
        child_prefix.push(idx as u8);
        let child_subtree = subtree_from_entries(child_prefix, entries);
        let (payload, child_bytes) = serialize_subtree(&child_subtree)?;
        if child_bytes > cfg.max_leaf_bytes {
            children[idx] = Some(split_subtree(store, cfg, child_subtree)?);
        } else if child_bytes >= cfg.min_promote_bytes {
            let ptr = store.write_payload(&payload)?;
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
        let (payload, bytes) = serialize_subtree(&rem_subtree)?;
        let ptr = store.write_payload(&payload)?;
        children[idx] = Some(RamChild::Disk {
            ptr,
            root: hash_node(&rem_subtree.node),
            bytes,
        });
    }

    let branch = RamNode::Branch {
        children,
        value: None,
        hash: Cell::new(None),
    };
    if shared.is_empty() {
        Ok(RamChild::Ram(Box::new(branch)))
    } else {
        Ok(RamChild::Ram(Box::new(RamNode::Extension {
            path: shared,
            child: Box::new(branch),
            hash: Cell::new(None),
        })))
    }
}

fn subtree_from_entries(prefix: Vec<u8>, entries: Vec<(Key, Hash)>) -> DiskSubtree {
    let node = build_node(&entries, prefix.len());
    DiskSubtree { prefix, node }
}

/// Collect every `(key, value_hash)` leaf in a node, in ascending key order.
fn node_entries(node: &Node) -> Vec<(Key, Hash)> {
    fn walk(node: &Node, out: &mut Vec<(Key, Hash)>) {
        match node {
            Node::Empty => {}
            Node::Leaf {
                key, value_hash, ..
            } => out.push((*key, *value_hash)),
            Node::Extension { child, .. } => walk(child, out),
            Node::Branch { children, .. } => {
                for child in children.iter().flatten() {
                    walk(child, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(node, &mut out);
    out
}

/// Whether a node's subtree holds `key` (used by the `disk_accesses_for_key` probe).
fn node_contains(node: &Node, key: &Key) -> bool {
    match node {
        Node::Empty => false,
        Node::Leaf { key: k, .. } => k == key,
        Node::Extension { child, .. } => node_contains(child, key),
        Node::Branch { children, .. } => {
            children.iter().flatten().any(|c| node_contains(c, key))
        }
    }
}

// --- Disk-node constructors: compute and cache the node hash exactly once. ---

fn make_leaf(key: Key, value_hash: Hash) -> Node {
    let mut bytes = vec![3];
    bytes.extend_from_slice(&key);
    bytes.extend_from_slice(&value_hash);
    Node::Leaf {
        key,
        value_hash,
        hash: keccak(&bytes),
    }
}

fn make_extension(path: Vec<u8>, child: Node) -> Node {
    let hash = hash_join(4, &path, &hash_node(&child));
    Node::Extension {
        path,
        child: Box::new(child),
        hash,
    }
}

fn make_branch(children: [Option<Box<Node>>; 16]) -> Node {
    // Disk-side branches never carry a value (every key is a full 64-nibble path).
    let mut bytes = vec![5];
    for child in &children {
        bytes.extend_from_slice(&child.as_ref().map(|c| hash_node(c)).unwrap_or_else(empty_hash));
    }
    Node::Branch {
        children,
        value: None,
        hash: keccak(&bytes),
    }
}

/// Canonical node for a subtree holding exactly one entry at `depth`.
fn single_entry_node(key: Key, value_hash: Hash, depth: usize) -> Node {
    let path = key_nibbles(&key)[depth..].to_vec();
    let leaf = make_leaf(key, value_hash);
    if path.is_empty() {
        leaf
    } else {
        make_extension(path, leaf)
    }
}

fn build_node(entries: &[(Key, Hash)], depth: usize) -> Node {
    if entries.is_empty() {
        return Node::Empty;
    }
    if entries.len() == 1 {
        let (key, value_hash) = entries[0];
        return single_entry_node(key, value_hash, depth);
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
        let path = nibbles[0][depth..depth + common].to_vec();
        return make_extension(path, build_node(entries, depth + common));
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
    make_branch(children)
}

/// Insert `(key, value_hash)` into an existing disk-subtree node in place,
/// recomputing cached hashes only for the nodes on the changed path. Every
/// untouched sibling subtree is left exactly as-is (hash included), which is
/// what makes the disk-side hashing strictly essential. `depth` is the nibble
/// depth of `node`. Produces the same canonical structure `build_node` would.
fn node_insert(node: &mut Node, depth: usize, key: Key, value_hash: Hash) {
    let nibbles = key_nibbles(&key);
    let updated = match std::mem::replace(node, Node::Empty) {
        Node::Empty => single_entry_node(key, value_hash, depth),
        Node::Leaf {
            key: leaf_key,
            value_hash: leaf_vh,
            ..
        } => {
            if leaf_key == key {
                make_leaf(key, value_hash)
            } else {
                // A bare leaf sits at depth 64, so two distinct 32-byte keys can
                // never both reach it — they diverge earlier, at a branch.
                debug_assert_ne!(leaf_key, key);
                let _ = leaf_vh;
                unreachable!("two distinct keys cannot share a full 64-nibble path");
            }
        }
        Node::Extension {
            path, mut child, ..
        } => {
            let common = common_prefix(&path, &nibbles[depth..]);
            if common == path.len() {
                node_insert(&mut child, depth + path.len(), key, value_hash);
                make_extension(path, *child)
            } else {
                // The new key diverges partway along the extension. Reuse the old
                // continuation verbatim (cached hashes intact) and add a leaf.
                let mut children = empty_box_children();
                let old_idx = path[common] as usize;
                let old_rest = path[common + 1..].to_vec();
                children[old_idx] = Some(Box::new(if old_rest.is_empty() {
                    *child
                } else {
                    make_extension(old_rest, *child)
                }));
                let new_idx = nibbles[depth + common] as usize;
                children[new_idx] =
                    Some(Box::new(single_entry_node(key, value_hash, depth + common + 1)));
                let branch = make_branch(children);
                if common == 0 {
                    branch
                } else {
                    make_extension(path[..common].to_vec(), branch)
                }
            }
        }
        Node::Branch { mut children, .. } => {
            let idx = nibbles[depth] as usize;
            match &mut children[idx] {
                Some(child) => node_insert(child, depth + 1, key, value_hash),
                None => {
                    children[idx] =
                        Some(Box::new(single_entry_node(key, value_hash, depth + 1)));
                }
            }
            make_branch(children)
        }
    };
    *node = updated;
}

/// Serialize a subtree once, returning the payload and the total on-disk record
/// size (`payload + 4`-byte length prefix). Callers use the size to decide
/// rewrite-vs-split and pass the same payload to [`FlatFile::write_payload`].
fn serialize_subtree(subtree: &DiskSubtree) -> Result<(Vec<u8>, usize)> {
    let _g = prof::scope(prof::Cat::Serialize);
    let payload = bincode::serialize(subtree)?;
    let total = payload.len() + 4;
    Ok((payload, total))
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
        RamNode::Extension { path, child, .. } => {
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

/// Drop the cached hash of `node` (if any). Called as the insert descends, so
/// exactly the nodes on the touched path are invalidated and later recomputed.
fn invalidate_ram(node: &RamNode) {
    match node {
        RamNode::Extension { hash, .. } | RamNode::Branch { hash, .. } => hash.set(None),
        RamNode::Empty => {}
    }
}

fn hash_ram(node: &RamNode) -> Hash {
    match node {
        RamNode::Empty => empty_hash(),
        RamNode::Extension { path, child, hash } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            let computed = hash_join(1, path, &hash_ram(child));
            hash.set(Some(computed));
            computed
        }
        RamNode::Branch {
            children,
            value,
            hash,
        } => {
            if let Some(cached) = hash.get() {
                return cached;
            }
            let mut bytes = vec![2];
            for child in children {
                let h = match child {
                    Some(RamChild::Ram(node)) => hash_ram(node),
                    Some(RamChild::Disk { root, .. }) => *root,
                    None => empty_hash(),
                };
                bytes.extend_from_slice(&h);
            }
            if let Some(v) = value {
                bytes.extend_from_slice(v);
            }
            let computed = keccak(&bytes);
            hash.set(Some(computed));
            computed
        }
    }
}

/// Disk-node hash accessor: returns the cached hash (computed at construction).
fn hash_node(node: &Node) -> Hash {
    match node {
        Node::Empty => empty_hash(),
        Node::Leaf { hash, .. } | Node::Extension { hash, .. } | Node::Branch { hash, .. } => {
            *hash
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
    let _g = prof::scope(prof::Cat::Keccak);
    let output: Hash = Keccak256::digest(bytes).into();
    prof::record(output);
    output
}

/// Hash of the empty/absent node. It is a constant, so compute it once instead
/// of re-running keccak for every empty child slot of every branch we hash.
fn empty_hash() -> Hash {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Hash> = OnceLock::new();
    *EMPTY.get_or_init(|| keccak(&[0]))
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
        assert_eq!(db.get_value(&key).unwrap(), Some(b"100".to_vec()));
        assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
    }

    #[test]
    fn repeated_insert_overwrites_value_hash() {
        let mut db = db(Config::default());
        let key = hashed_key("alice");
        let root1 = db.insert(key, b"100".to_vec()).unwrap();
        let root2 = db.insert(key, b"200".to_vec()).unwrap();
        assert_ne!(root1, root2);
        assert_eq!(db.get_value(&key).unwrap(), Some(b"200".to_vec()));
        assert_eq!(db.disk_accesses_for_key(&key).unwrap(), 1);
    }

    #[test]
    fn persists_and_reopens_frontier() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.flat");
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let keys: Vec<Key> = (0..300u64).map(|i| hashed_key(i.to_le_bytes())).collect();

        let (root, flat_len, free_bytes, ram_nodes) = {
            let mut db = FlatMpt::create(&path, cfg).unwrap();
            for (i, key) in keys.iter().enumerate() {
                db.insert(*key, vec![i as u8; 40]).unwrap();
            }
            db.persist().unwrap();
            (db.root(), db.flat_file_len(), db.free_bytes(), db.ram_nodes())
        }; // drop: close the flat file and RocksDB

        let mut db = FlatMpt::open(&path).unwrap();
        // The frontier, free list, and root all survived the reopen.
        assert_eq!(db.root(), root);
        assert_eq!(db.flat_file_len(), flat_len);
        assert_eq!(db.free_bytes(), free_bytes);
        assert_eq!(db.ram_nodes(), ram_nodes);
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(db.get_value(key).unwrap(), Some(vec![i as u8; 40]));
            assert_eq!(db.disk_accesses_for_key(key).unwrap(), 1);
        }

        // And the reopened database is fully writable.
        let new_key = hashed_key("inserted-after-reopen");
        let new_root = db.insert(new_key, b"value".to_vec()).unwrap();
        assert_ne!(new_root, root);
        assert_eq!(db.get_value(&new_key).unwrap(), Some(b"value".to_vec()));
    }

    #[test]
    fn values_round_trip_through_disk_store() {
        let mut db = db(Config::default());
        let key = hashed_key("alice");
        db.insert(key, b"hello world".to_vec()).unwrap();
        // Reads come straight back out of the on-disk RocksDB store.
        assert_eq!(db.get_value(&key).unwrap(), Some(b"hello world".to_vec()));
        assert_eq!(db.get_value(&hashed_key("absent")).unwrap(), None);
    }

    #[test]
    fn reuses_freed_flat_file_space_on_overwrite() {
        let cfg = Config {
            target_leaf_bytes: 512,
            max_leaf_bytes: 768,
            min_promote_bytes: 192,
        };
        let mut db = db(cfg);
        let keys: Vec<Key> = (0..200u64).map(|i| hashed_key(i.to_le_bytes())).collect();
        for key in &keys {
            db.insert(*key, vec![1; 32]).unwrap();
        }
        let len_after_build = db.flat_file_len();

        // Overwrite every value many times. Each overwrite rewrites a disk
        // subtree of the same size, freeing the old region and reusing it, so
        // the flat file must not keep growing.
        for round in 0..20u8 {
            for key in &keys {
                db.insert(*key, vec![round; 32]).unwrap();
            }
        }
        let len_after_churn = db.flat_file_len();

        // Without reuse this would grow ~20x; allow generous slack for any
        // transient remainder fragments while still proving space is recycled.
        assert!(
            len_after_churn <= len_after_build + len_after_build / 2,
            "flat file grew from {len_after_build} to {len_after_churn} despite reuse"
        );
        // All values are still retrievable and current after the churn.
        for key in &keys {
            assert_eq!(db.get_value(key).unwrap(), Some(vec![19u8; 32]));
        }
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
