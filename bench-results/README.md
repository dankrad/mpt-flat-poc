# Benchmark results

## `1b-baseline-main.log`

A 1B-key `cargo bench --bench large` run on **`main`** (commit `0464dc6`), stopped
at **480M keys** after ~11.4 h. This is the baseline for the I/O optimizations
being developed on the `ram-optimization` branch.

**Build under test:** byte-granular free list (O(log n) best-fit), and each leaf
access split into a seek + length read + payload read (2 read / 2 write
syscalls). Records are *not* page-aligned. Config: `target=4K, max=8K, min=2K`.

### What it showed (per 10M-key chunk)

| keys | µs/key (chunk) | flat file | free_regions | in-RAM index |
|--:|--:|--:|--:|--:|
| 10M | 15.8 | 1.7 GiB | 340K | 89 MiB |
| 100M | ~28 | 16 GiB | ~1.5M | ~0.7 GiB |
| 470M | 145 | 78 GiB | 6.5M | 1.49 GiB |
| 480M | 146 | 80 GiB | 6.5M | 1.49 GiB |

Key findings:
- **In-RAM index grows *sub*-linearly past ~10M** (per-key RAM falls 9.4 → 3.2 B/key);
  1B projects to ~3 GB, not the ~20 GB extrapolated from <10M data.
- **Insert latency degraded ~9× (16 → 145 µs/key)** and is **I/O-bound**, confirmed
  by sampling the running process: ~78% of insert time in the `read()` syscall
  (page-cache misses once the 78 GB flat file + ~16 GB RocksDB exceed page cache),
  hashing only ~6%, process at 38% CPU.
- **Fragmentation**: free_regions grew to ~6.5M (byte-granular best-fit leaves
  many small unusable remainders).

### Optimizations being compared against this
1. Single positioned `pread`/`pwrite` per leaf access (was 2 syscalls + seek).
2. Page-aligned (4 KB) records.
3. Page-granular free list (collapses the size distribution → less fragmentation).
