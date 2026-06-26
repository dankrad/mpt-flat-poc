//! Measure the SSD's random I/O rate under $TMPDIR, bypassing the page cache
//! (macOS F_NOCACHE), to compare against the MPT's per-insert I/O.
//!
//!     cargo run --release --example iops -- [threads] [file_gib] [ops] [block]
//!
//! Defaults: 8 threads (matching the batch workers), 16 GiB file, 1M ops,
//! 8192-byte block (a 2-page leaf). Pass block=4096 for a 1-page leaf.

use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

fn main() {
    let threads: usize = arg(1).unwrap_or(8);
    let file_gib: u64 = arg(2).unwrap_or(16);
    let ops: u64 = arg(3).unwrap_or(1_000_000);
    let block: usize = arg(4).unwrap_or(8192);
    let file_bytes = file_gib * 1024 * 1024 * 1024;
    let blocks = file_bytes / block as u64;

    let path = std::env::temp_dir().join("iops-test.bin");
    let f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    f.set_len(file_bytes).unwrap();
    // Bypass the unified buffer cache so we measure the device, not RAM.
    unsafe {
        libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    println!(
        "random {block}-byte I/O on {} ({file_gib} GiB file, {threads} threads, F_NOCACHE):",
        path.display()
    );

    // Sequential fill (allocates every block so reads hit real data).
    let t = Instant::now();
    {
        let buf = vec![0u8; block];
        let mut off = 0u64;
        while off < file_bytes {
            f.write_all_at(&buf, off).unwrap();
            off += block as u64;
        }
        f.sync_all().unwrap();
    }
    report("seq write ", blocks, block, t.elapsed().as_secs_f64());

    run(&f, "rand write", blocks, block, ops, threads, true);
    run(&f, "rand read ", blocks, block, ops, threads, false);

    std::fs::remove_file(&path).ok();
}

#[allow(clippy::too_many_arguments)]
fn run(f: &std::fs::File, label: &str, blocks: u64, block: usize, ops: u64, threads: usize, write: bool) {
    let per = ops / threads as u64;
    let t = Instant::now();
    std::thread::scope(|s| {
        for tid in 0..threads {
            s.spawn(move || {
                let mut st = 0x9e3779b97f4a7c15u64 ^ (tid as u64).wrapping_mul(0x100000001b3);
                let mut buf = vec![0u8; block];
                for _ in 0..per {
                    st = st
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let off = ((st >> 16) % blocks) * block as u64;
                    if write {
                        f.write_all_at(&buf, off).unwrap();
                    } else {
                        f.read_exact_at(&mut buf, off).unwrap();
                    }
                }
            });
        }
    });
    if write {
        f.sync_all().unwrap();
    }
    report(label, per * threads as u64, block, t.elapsed().as_secs_f64());
}

fn report(label: &str, n: u64, block: usize, secs: f64) {
    println!(
        "  {label}: {:>8.0} IOPS  {:>6.0} MB/s   ({n} ops, {secs:.1}s)",
        n as f64 / secs,
        n as f64 * block as f64 / 1e6 / secs,
    );
}

fn arg<T: std::str::FromStr>(i: usize) -> Option<T> {
    std::env::args().nth(i).and_then(|s| s.parse().ok())
}
