//! Isolate the cost of a *cached* read: syscall overhead vs the byte copy.
//! Single-threaded, random offsets over a page-cache-warm 256 MiB file. For each
//! size we time three things that read `S` bytes:
//!   pread  = read_exact_at  (syscall + copy out of the page cache)
//!   memcpy = copy_from_slice on a large in-RAM buffer (pure cold-RAM copy)
//!   mmap   = copy_from_slice on an mmap'd view of the file (copy, NO syscall)
//! pread - mmap ≈ the syscall cost; flat-across-sizes ⇒ syscall-dominated.
//!
//!     cargo run --release --example readcost

use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

const FILE: usize = 256 * 1024 * 1024;
const ITERS: usize = 400_000;

fn main() {
    let path = std::env::temp_dir().join("readcost.bin");
    let f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    f.set_len(FILE as u64).unwrap();
    let chunk = vec![0xABu8; 1 << 20];
    let mut o = 0u64;
    while (o as usize) < FILE {
        f.write_all_at(&chunk, o).unwrap();
        o += chunk.len() as u64;
    }
    // Warm the page cache.
    {
        let mut b = vec![0u8; 1 << 20];
        let mut o = 0u64;
        while (o as usize) < FILE {
            (&f).read_exact_at(&mut b, o).ok();
            o += b.len() as u64;
        }
    }
    // Large in-RAM buffer for the pure-copy baseline (cold CPU cache at random offsets).
    let src = vec![0xCDu8; FILE];
    // mmap the file (read-only, random-advised), then touch every page to fault it in.
    let map = unsafe {
        libc::mmap(std::ptr::null_mut(), FILE, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0)
    };
    assert!(map != libc::MAP_FAILED, "mmap failed");
    unsafe { libc::madvise(map, FILE, libc::MADV_RANDOM) };
    let mslice = unsafe { std::slice::from_raw_parts(map as *const u8, FILE) };
    {
        let mut s = 0u64;
        let mut i = 0;
        while i < FILE {
            s += mslice[i] as u64;
            i += 4096;
        }
        std::hint::black_box(s);
    }

    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut off = |s: usize| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 16) as usize % (FILE - s)
    };

    println!("256 MiB cached file, {ITERS} random reads each, single thread:");
    println!("  {:>7} | {:>10} {:>10} {:>10} | {:>12}", "size", "pread ns", "memcpy ns", "mmap ns", "syscall gap");
    for &s in &[64usize, 512, 4096, 5120, 16384] {
        let mut buf = vec![0u8; s];
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            (&f).read_exact_at(&mut buf, off(s) as u64).unwrap();
            acc += buf[0] as u64;
        }
        let pread = t.elapsed().as_nanos() as f64 / ITERS as f64;
        std::hint::black_box(acc);

        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            let o = off(s);
            buf.copy_from_slice(&src[o..o + s]);
            acc += buf[0] as u64;
        }
        let mc = t.elapsed().as_nanos() as f64 / ITERS as f64;
        std::hint::black_box(acc);

        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            let o = off(s);
            buf.copy_from_slice(&mslice[o..o + s]);
            acc += buf[0] as u64;
        }
        let mm = t.elapsed().as_nanos() as f64 / ITERS as f64;
        std::hint::black_box(acc);

        println!("  {:>6}B | {:>10.0} {:>10.0} {:>10.0} | {:>12.0}", s, pread, mc, mm, pread - mm);
    }
    // Thread-scaling at the record size (~5 KB): does the cached pread cost
    // balloon under concurrency (fd/vnode lock) while mmap stays flat?
    const S: usize = 5120;
    let map_addr = map as usize;
    println!("\nthread-scaling at {S} B (per-op ns; total {ITERS} reads split across threads):");
    println!("  {:>8} | {:>12} {:>12}", "threads", "pread ns", "mmap ns");
    for &t in &[1usize, 2, 4, 8, 16] {
        let per = ITERS / t;
        let pread_ns = {
            let start = Instant::now();
            std::thread::scope(|sc| {
                for tid in 0..t {
                    let f = &f;
                    sc.spawn(move || {
                        let mut buf = vec![0u8; S];
                        let mut rng = 0x1234567u64 ^ (tid as u64).wrapping_mul(0x9e3779b9);
                        let mut acc = 0u64;
                        for _ in 0..per {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let o = (rng >> 16) as usize % (FILE - S);
                            f.read_exact_at(&mut buf, o as u64).unwrap();
                            acc += buf[0] as u64;
                        }
                        std::hint::black_box(acc);
                    });
                }
            });
            start.elapsed().as_nanos() as f64 / (per * t) as f64
        };
        let mmap_ns = {
            let start = Instant::now();
            std::thread::scope(|sc| {
                for tid in 0..t {
                    sc.spawn(move || {
                        let m = unsafe { std::slice::from_raw_parts(map_addr as *const u8, FILE) };
                        let mut buf = vec![0u8; S];
                        let mut rng = 0x1234567u64 ^ (tid as u64).wrapping_mul(0x9e3779b9);
                        let mut acc = 0u64;
                        for _ in 0..per {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let o = (rng >> 16) as usize % (FILE - S);
                            buf.copy_from_slice(&m[o..o + S]);
                            acc += buf[0] as u64;
                        }
                        std::hint::black_box(acc);
                    });
                }
            });
            start.elapsed().as_nanos() as f64 / (per * t) as f64
        };
        println!("  {:>8} | {:>12.0} {:>12.0}", t, pread_ns, mmap_ns);
    }
    unsafe { libc::munmap(map, FILE) };
    std::fs::remove_file(&path).ok();
}
