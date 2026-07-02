//! Ethereum state layer over the flat-file trie: a typed account API plus a code
//! store, producing mainnet-exact state roots.
//!
//! The state trie is the existing [`FlatMpt`] keyed by `keccak256(address)` (a
//! secure trie) with the RLP-encoded [`Account`] as the leaf value — so the
//! account fields live *in* the trie and re-hashing never consults an external
//! store (see the crate's Ethereum-equivalence plan). Bytecode is the one thing
//! kept outside the trie: it is content-addressed by `code_hash` and **never read
//! during hashing** (only the 32-byte `code_hash` inside the account is), so a
//! simple append log suffices.
//!
//! Storage is still flat here (Phase 3): an account carries whatever `storage_root`
//! the caller sets. The nested per-account storage trie that computes that root
//! from packed storage leaves is Phase 4.

use crate::eth::{Account, EMPTY_CODE_HASH};
use crate::{FlatMpt, Key};
use alloy_primitives::{keccak256, Address, B256};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// The secure-trie key for an address: `keccak256(address)`.
fn account_key(addr: &Address) -> Key {
    keccak256(addr.as_slice()).0
}

/// Content-addressed bytecode store: an append log of `[code_hash:32][len:4][code]`
/// records with an in-RAM `code_hash -> (offset, len)` index rebuilt by scanning
/// the log on open. Write-once (a repeated hash is not re-appended). Never consulted
/// during trie hashing.
pub struct CodeStore {
    file: File,
    index: HashMap<B256, (u64, u32)>,
    end: u64,
}

impl CodeStore {
    fn open(path: PathBuf, truncate: bool) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(truncate)
            .open(&path)?;
        let mut index = HashMap::new();
        let mut end = 0u64;
        if !truncate {
            // Rebuild the index by scanning the log.
            file.seek(SeekFrom::Start(0))?;
            let mut hdr = [0u8; 36];
            loop {
                match read_exact_or_eof(&mut file, &mut hdr)? {
                    false => break,
                    true => {}
                }
                let hash = B256::from_slice(&hdr[..32]);
                let len = u32::from_le_bytes(hdr[32..36].try_into().unwrap());
                let data_off = end + 36;
                index.insert(hash, (data_off, len));
                end = data_off + len as u64;
                file.seek(SeekFrom::Start(end))?;
            }
        }
        file.seek(SeekFrom::Start(end))?;
        Ok(Self { file, index, end })
    }

    /// Store `code`, returning its `keccak256` hash. No-op (returns the hash) if the
    /// code is already present or empty (empty code => `EMPTY_CODE_HASH`, not stored).
    pub fn put(&mut self, code: &[u8]) -> Result<B256> {
        let hash = keccak256(code);
        if code.is_empty() || self.index.contains_key(&hash) {
            return Ok(hash);
        }
        let mut rec = Vec::with_capacity(36 + code.len());
        rec.extend_from_slice(hash.as_slice());
        rec.extend_from_slice(&(code.len() as u32).to_le_bytes());
        rec.extend_from_slice(code);
        self.file.write_all(&rec)?;
        let data_off = self.end + 36;
        self.index.insert(hash, (data_off, code.len() as u32));
        self.end = data_off + code.len() as u64;
        Ok(hash)
    }

    /// Fetch bytecode by hash (`EMPTY_CODE_HASH` -> empty code).
    pub fn get(&self, hash: &B256) -> Result<Option<Vec<u8>>> {
        if *hash == EMPTY_CODE_HASH {
            return Ok(Some(Vec::new()));
        }
        let Some(&(off, len)) = self.index.get(hash) else {
            return Ok(None);
        };
        let mut buf = vec![0u8; len as usize];
        // Positioned read so we don't disturb the append cursor.
        read_at(&self.file, off, &mut buf)?;
        Ok(Some(buf))
    }

    fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }
}

/// Ethereum state: the account state trie plus the code store.
pub struct EthState {
    trie: FlatMpt,
    code: CodeStore,
}

fn code_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".code");
    path.with_file_name(name)
}

impl EthState {
    /// Create a fresh state at `path` (trie) + `path.code` (bytecode log).
    pub fn create(path: impl AsRef<Path>, cfg: crate::Config) -> Result<Self> {
        let path = path.as_ref();
        let trie = FlatMpt::create(path, cfg)?;
        let code = CodeStore::open(code_path(path), true)?;
        Ok(Self { trie, code })
    }

    /// Reopen a persisted state.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let trie = FlatMpt::open(path)?;
        let code = CodeStore::open(code_path(path), false)?;
        Ok(Self { trie, code })
    }

    /// Insert/overwrite an account (keyed by `keccak256(address)`).
    pub fn set_account(&mut self, addr: &Address, acct: &Account) -> Result<()> {
        self.trie.insert(account_key(addr), acct.rlp())?;
        Ok(())
    }

    /// Bulk account upsert — one batched trie insert.
    pub fn set_accounts(&mut self, accounts: &[(Address, Account)]) -> Result<()> {
        let entries: Vec<(Key, Vec<u8>)> =
            accounts.iter().map(|(a, acct)| (account_key(a), acct.rlp())).collect();
        self.trie.insert_batch(entries)?;
        Ok(())
    }

    /// Fetch and decode an account.
    pub fn get_account(&self, addr: &Address) -> Result<Option<Account>> {
        match self.trie.get_value(&account_key(addr))? {
            Some(bytes) => Ok(Some(Account::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Store bytecode, returning its `code_hash` (to place in an account).
    pub fn set_code(&mut self, code: &[u8]) -> Result<B256> {
        self.code.put(code)
    }

    /// Fetch bytecode by `code_hash`.
    pub fn get_code(&self, code_hash: &B256) -> Result<Option<Vec<u8>>> {
        self.code.get(code_hash)
    }

    /// The state-trie root — the Ethereum `stateRoot`.
    pub fn state_root(&self) -> B256 {
        B256::from(self.trie.root())
    }

    /// Checkpoint the trie and flush the code log.
    pub fn persist(&mut self) -> Result<()> {
        self.trie.persist()?;
        self.code.flush()
    }
}

fn read_exact_or_eof(f: &mut File, buf: &mut [u8]) -> Result<bool> {
    let mut read = 0;
    while read < buf.len() {
        match f.read(&mut buf[read..])? {
            0 if read == 0 => return Ok(false), // clean EOF at a record boundary
            0 => anyhow::bail!("truncated code-store record"),
            n => read += n,
        }
    }
    Ok(true)
}

#[cfg(unix)]
fn read_at(f: &File, off: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{hex, U256};
    use tempfile::NamedTempFile;

    fn state() -> EthState {
        EthState::create(NamedTempFile::new().unwrap().path(), crate::Config::default()).unwrap()
    }

    #[test]
    fn assembled_state_reproduces_known_root() {
        // The official secure-trie account vector, assembled through the typed API,
        // must reproduce the known Ethereum state root.
        let accts = [
            ("a94f5374fce5edbc8e2a8697c15331677e6ebf0b", Account::eoa(1, U256::from(0x05f446a7u64))),
            ("095e7baea6a6c7c4c2dfeb977efac326af552d87",
             Account::contract(1, U256::from(1u64), crate::eth::EMPTY_ROOT,
                "0x04bccc5d94f4d1f99aab44369a910179931772f2a5c001c3229f57831c102769".parse().unwrap())),
            ("d2571607e241ecf590ed94b12d87c94babe36db6",
             Account::contract(1, U256::ZERO,
                "0xba4b47865c55a341a4a78759bb913cd15c3ee8eaf30a62fa8d1c8863113d84e8".parse().unwrap(),
                EMPTY_CODE_HASH)),
            ("62c01474f089b07dae603491675dc5b5748f7049", Account::eoa(0, U256::ZERO)),
            ("2adc25665018aa1fe0e6bc666dac8fc2697ff9ba", Account::eoa(0, U256::from(0x019a59u64))),
        ];
        let mut st = state();
        for (addr, acct) in &accts {
            let addr: Address = addr.parse().unwrap();
            st.set_account(&addr, acct).unwrap();
        }
        let want: B256 = "0x730a444e08ab4b8dee147c9b232fc52d34a223d600031c1e9d25bfc985cbd797"
            .parse()
            .unwrap();
        assert_eq!(st.state_root(), want);

        // Round-trip one account through get_account.
        let a0: Address = accts[0].0.parse().unwrap();
        assert_eq!(st.get_account(&a0).unwrap().unwrap(), accts[0].1);
        let missing: Address = "0x000000000000000000000000000000000000dead".parse().unwrap();
        assert_eq!(st.get_account(&missing).unwrap(), None);
    }

    #[test]
    fn code_store_round_trips() {
        let mut st = state();
        let code = hex::decode("6080604052348015600f57600080fd").unwrap();
        let h = st.set_code(&code).unwrap();
        assert_eq!(h, keccak256(&code));
        assert_eq!(st.set_code(&code).unwrap(), h, "idempotent on repeat");
        assert_eq!(st.get_code(&h).unwrap().unwrap(), code);
        assert_eq!(st.get_code(&EMPTY_CODE_HASH).unwrap().unwrap(), Vec::<u8>::new());
        let unknown: B256 = "0x1111111111111111111111111111111111111111111111111111111111111111".parse().unwrap();
        assert_eq!(st.get_code(&unknown).unwrap(), None);
    }

    #[test]
    fn code_store_reopen_rebuilds_index() {
        let tmp = NamedTempFile::new().unwrap();
        let (h1, h2);
        {
            let mut st = EthState::create(tmp.path(), crate::Config::default()).unwrap();
            h1 = st.set_code(b"contract-one").unwrap();
            h2 = st.set_code(b"contract-two-longer-bytecode").unwrap();
            st.persist().unwrap();
        }
        let st = EthState::open(tmp.path()).unwrap();
        assert_eq!(st.get_code(&h1).unwrap().unwrap(), b"contract-one");
        assert_eq!(st.get_code(&h2).unwrap().unwrap(), b"contract-two-longer-bytecode");
    }
}
