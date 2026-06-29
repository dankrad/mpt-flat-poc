//! Does splitting one file into N separate mmap mappings relieve the page-fault
//! lock in the *IO-bound* (major-fault) regime — the one that caps cold mmap?
//!
//! The file is larger than RAM and filled UNCACHED (F_NOCACHE), so the page
//! cache cannot hold it and reads are real major faults to the device. Each
//! mapping-count reads its own fresh quarter of the file, so the runs don't warm
//! one another. We compare random-read IOPS at 32 threads across N mappings,
//! against an O_DIRECT pread baseline on the same file. If more mappings scale,
//! the binding fault lock is per-mapping (VMA); if flat, it's per-file
//! (page cache / vm_object), shared by every mapping of the same file.
//!
//!     MMAPSEG_GIB=192 cargo run --release --example mmapseg

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

const BLOCK: usize = 8192;
const THREADS: usize = 32;
const OPS: u64 = 1_000_000;
const NS: [usize; 4] = [1, 4, 16, 64];

fn open_uncached(path: &std::path::Path, truncate: bool) -> File {
    let mut o = OpenOptions::new();
    o.create(true).read(true).write(true);
    if truncate {
        o.truncate(true);
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_DIRECT);
    }
    let f = o.open(path).unwrap();
    #[cfg(target_os = "macos")]
    unsafe {
        libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    f
}

fn main() {
    let gib: usize = std::env::var("MMAPSEG_GIB").ok().and_then(|s| s.parse().ok()).unwrap_or(192);
    let file_bytes = gib * (1 << 30);
    let path = std::env::temp_dir().join("mmapseg.bin");

    // Fill uncached: real blocks on disk, nothing left in the page cache.
    {
        let f = open_uncached(&path, true);
        f.set_len(file_bytes as u64).unwrap();
        let layout = std::alloc::Layout::from_size_align(1 << 20, 4096).unwrap();
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        let chunk = unsafe { std::slice::from_raw_parts(p, 1 << 20) };
        let mut o = 0u64;
        while (o as usize) < file_bytes {
            f.write_all_at(chunk, o).unwrap();
            o += 1 << 20;
        }
        f.sync_all().unwrap();
        unsafe { std::alloc::dealloc(p, layout) };
    }
    println!("{gib} GiB uncached file (> {}-ish GiB RAM), {THREADS} threads, {OPS} reads/run, 8 KiB:", 137);

    // O_DIRECT pread baseline (no page cache) on the same file.
    {
        let f = open_uncached(&path, false);
        let blocks = (file_bytes / BLOCK) as u64;
        let per = OPS / THREADS as u64;
        let t = Instant::now();
        std::thread::scope(|s| {
            for tid in 0..THREADS {
                let f = &f;
                s.spawn(move || {
                    let layout = std::alloc::Layout::from_size_align(BLOCK, 4096).unwrap();
                    let p = unsafe { std::alloc::alloc(layout) };
                    let b = unsafe { std::slice::from_raw_parts_mut(p, BLOCK) };
                    let mut st = 0x9e3779b97f4a7c15u64 ^ (tid as u64).wrapping_mul(0x100000001b3);
                    let mut acc = 0u64;
                    for _ in 0..per {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let off = ((st >> 16) % blocks) as usize * BLOCK;
                        f.read_exact_at(b, off as u64).unwrap();
                        acc += b[0] as u64;
                    }
                    std::hint::black_box(acc);
                    unsafe { std::alloc::dealloc(p, layout) };
                });
            }
        });
        let secs = t.elapsed().as_secs_f64();
        let n = per * THREADS as u64;
        println!("  {:>15} | {:>10.0} IOPS  {:>8.0} MB/s", "O_DIRECT pread", n as f64 / secs, n as f64 * BLOCK as f64 / 1e6 / secs);
    }

    // mmap as N segments, cold major faults; each run reads its own fresh quarter.
    let f = OpenOptions::new().read(true).open(&path).unwrap();
    for (ri, &n) in NS.iter().enumerate() {
        let seg = file_bytes / n;
        let maps: Vec<usize> = (0..n)
            .map(|i| {
                let m = unsafe {
                    libc::mmap(std::ptr::null_mut(), seg, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), (i * seg) as libc::off_t)
                };
                assert!(m != libc::MAP_FAILED, "mmap failed");
                unsafe { libc::madvise(m, seg, libc::MADV_RANDOM) };
                m as usize
            })
            .collect();
        let region = file_bytes / NS.len();
        let base = (ri * region) as u64;
        let rblocks = (region / BLOCK) as u64;
        let per = OPS / THREADS as u64;
        let maps_ref = &maps;
        let t = Instant::now();
        std::thread::scope(|s| {
            for tid in 0..THREADS {
                s.spawn(move || {
                    let mut st = 0x9e3779b97f4a7c15u64 ^ (tid as u64).wrapping_mul(0x100000001b3);
                    let mut buf = vec![0u8; BLOCK];
                    let mut acc = 0u64;
                    for _ in 0..per {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let off = (base + ((st >> 16) % rblocks) * BLOCK as u64) as usize;
                        let (si, lo) = (off / seg, off % seg);
                        if lo + BLOCK <= seg {
                            let m = unsafe { std::slice::from_raw_parts(maps_ref[si] as *const u8, seg) };
                            buf.copy_from_slice(&m[lo..lo + BLOCK]);
                            acc += buf[0] as u64;
                        }
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        let secs = t.elapsed().as_secs_f64();
        let nn = per * THREADS as u64;
        println!("  {:>10} maps | {:>10.0} IOPS  {:>8.0} MB/s", n, nn as f64 / secs, nn as f64 * BLOCK as f64 / 1e6 / secs);
        for m in &maps {
            unsafe { libc::munmap(*m as *mut libc::c_void, seg) };
        }
    }
    std::fs::remove_file(&path).ok();
}
