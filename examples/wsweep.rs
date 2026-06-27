//! Multi-threaded WRITE-bandwidth sweep across block sizes — to settle whether
//! coalescing leaf writes into larger `pwrite`s raises throughput (higher MB/s
//! even if IOPS falls). Mirrors the batch workers: `threads` threads each write
//! their own contiguous region, sequentially and (for contrast) randomly within
//! it. Bypasses the page cache (F_NOCACHE) so it measures the device.
//!
//!     cargo run --release --example wsweep -- [threads] [file_gib]

use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

fn main() {
    let threads: usize = arg(1).unwrap_or(8);
    let file_gib: u64 = arg(2).unwrap_or(16);
    let total = file_gib * 1024 * 1024 * 1024;
    let path = std::env::temp_dir().join("wsweep.bin");
    let f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    f.set_len(total).unwrap();
    unsafe {
        libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    println!("write-bandwidth sweep: {threads} threads, {file_gib} GiB file, F_NOCACHE");
    println!(
        "  {:>7} | {:>10} {:>9} | {:>10} {:>9}",
        "block", "seq MB/s", "seq kIOPS", "rand MB/s", "rand kIOPS"
    );
    for &block in &[16384usize, 32768, 49152, 65536, 131072, 262144, 524288] {
        let per = (total / threads as u64) / block as u64; // blocks per thread
        let ops = per * threads as u64;
        let bytes = ops * block as u64;
        let seq = run(&f, threads, block, per, true);
        let rnd = run(&f, threads, block, per, false);
        let mbps = |s: f64| bytes as f64 / 1e6 / s;
        let kiops = |s: f64| ops as f64 / 1e3 / s;
        println!(
            "  {:>5}K | {:>10.0} {:>9.0} | {:>10.0} {:>9.0}",
            block / 1024,
            mbps(seq),
            kiops(seq),
            mbps(rnd),
            kiops(rnd),
        );
    }
    std::fs::remove_file(&path).ok();
}

fn run(f: &std::fs::File, threads: usize, block: usize, per: u64, seq: bool) -> f64 {
    let t = Instant::now();
    std::thread::scope(|s| {
        for tid in 0..threads {
            s.spawn(move || {
                let region = per * block as u64 * tid as u64; // disjoint per-thread region
                let buf = vec![0u8; block];
                let mut st = 0x9e3779b97f4a7c15u64 ^ (tid as u64).wrapping_mul(0x100000001b3);
                for i in 0..per {
                    let off = if seq {
                        region + i * block as u64
                    } else {
                        st = st
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        region + ((st >> 16) % per) * block as u64
                    };
                    f.write_all_at(&buf, off).unwrap();
                }
            });
        }
    });
    f.sync_all().unwrap();
    t.elapsed().as_secs_f64()
}

fn arg<T: std::str::FromStr>(i: usize) -> Option<T> {
    std::env::args().nth(i).and_then(|s| s.parse().ok())
}
