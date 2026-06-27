//! Measure the SSD's random I/O rate, bypassing the page cache so we measure the
//! device, not RAM. Linux: O_DIRECT; macOS: F_NOCACHE.
//!
//!     cargo run --release --example iops -- [threads] [file_gib] [ops] [block]
//!
//! Defaults: 8 threads (matching the batch workers), 16 GiB file, 1M ops,
//! 8192-byte block. Block must be a multiple of 4096 (O_DIRECT alignment).
//!
//! NOTE: run with $TMPDIR (or cwd) on the real SSD — O_DIRECT is unsupported on
//! tmpfs (and tmpfs is RAM anyway). The test file is created next to $TMPDIR.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::time::Instant;

/// O_DIRECT requires buffer, offset, and length aligned to the device's logical
/// block size; 4096 covers ext4/xfs (and is harmless on macOS).
const ALIGN: usize = 4096;

fn main() {
    let threads: usize = arg(1).unwrap_or(8);
    let file_gib: u64 = arg(2).unwrap_or(16);
    let ops: u64 = arg(3).unwrap_or(1_000_000);
    let block: usize = arg(4).unwrap_or(8192);
    assert!(
        block % ALIGN == 0,
        "block must be a multiple of {ALIGN} for O_DIRECT alignment"
    );
    let file_bytes = file_gib * 1024 * 1024 * 1024;
    let blocks = file_bytes / block as u64;

    let path = std::env::temp_dir().join("iops-test.bin");
    let f = open_uncached(&path);
    f.set_len(file_bytes).unwrap();
    println!(
        "random {block}-byte I/O on {} ({file_gib} GiB file, {threads} threads, uncached):",
        path.display()
    );

    // Sequential fill (allocates every block so reads hit real data).
    let t = Instant::now();
    {
        let buf = AlignedBuf::new(block);
        let mut off = 0u64;
        while off < file_bytes {
            f.write_all_at(buf.as_slice(), off).unwrap();
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
fn run(f: &File, label: &str, blocks: u64, block: usize, ops: u64, threads: usize, write: bool) {
    let per = ops / threads as u64;
    let t = Instant::now();
    std::thread::scope(|s| {
        for tid in 0..threads {
            s.spawn(move || {
                let mut st = 0x9e3779b97f4a7c15u64 ^ (tid as u64).wrapping_mul(0x100000001b3);
                let mut buf = AlignedBuf::new(block);
                for _ in 0..per {
                    st = st
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let off = ((st >> 16) % blocks) * block as u64;
                    if write {
                        f.write_all_at(buf.as_slice(), off).unwrap();
                    } else {
                        f.read_exact_at(buf.as_mut(), off).unwrap();
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

/// Open a file with the page cache bypassed for the device measurement.
fn open_uncached(path: &std::path::Path) -> File {
    let mut opts = OpenOptions::new();
    opts.create(true).read(true).write(true).truncate(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_DIRECT);
    }
    let f = opts.open(path).unwrap_or_else(|e| {
        panic!(
            "open {}: {e}\n  (Linux: O_DIRECT is unsupported on tmpfs/overlayfs — \
             point TMPDIR at a real SSD-backed path)",
            path.display()
        )
    });
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    f
}

/// A heap buffer aligned to `ALIGN`, required for O_DIRECT I/O.
struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, ALIGN).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned alloc failed");
        Self { ptr, layout, len }
    }
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}
