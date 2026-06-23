use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use mpt_flat_poc::{Config, FlatMpt, hashed_key};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use tempfile::NamedTempFile;

fn make_db() -> FlatMpt {
    FlatMpt::create(
        NamedTempFile::new().unwrap().path(),
        Config {
            target_leaf_bytes: 4 * 1024,
            max_leaf_bytes: 8 * 1024,
            min_promote_bytes: 2 * 1024,
        },
    )
    .unwrap()
}

fn bench_inserts(c: &mut Criterion) {
    c.bench_function("insert_1000_random", |b| {
        b.iter_batched(
            || {
                let mut rng = StdRng::seed_from_u64(7);
                let mut keys = Vec::with_capacity(1000);
                for _ in 0..1000 {
                    let mut key = [0u8; 32];
                    rng.fill_bytes(&mut key);
                    keys.push(key);
                }
                (make_db(), keys)
            },
            |(mut db, keys)| {
                for key in keys {
                    db.insert(key, vec![1; 32]).unwrap();
                }
                db.root()
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("insert_1000_sequential_hashed", |b| {
        b.iter_batched(
            || (make_db(), 0u64..1000),
            |(mut db, range)| {
                for i in range {
                    db.insert(hashed_key(i.to_le_bytes()), vec![1; 32]).unwrap();
                }
                db.root()
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("insert_1000_shared_prefix", |b| {
        b.iter_batched(
            || {
                let mut keys = Vec::with_capacity(1000);
                for i in 0..1000u16 {
                    let mut key = [0u8; 32];
                    key[..6].copy_from_slice(&[0xab, 0xcd, 0xef, 0x12, 0x34, 0x50]);
                    key[30..].copy_from_slice(&i.to_be_bytes());
                    keys.push(key);
                }
                (make_db(), keys)
            },
            |(mut db, keys)| {
                for key in keys {
                    db.insert(key, vec![1; 32]).unwrap();
                }
                db.ram_nodes()
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_inserts);
criterion_main!(benches);
