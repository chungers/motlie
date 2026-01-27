# bench_vector Admin

This document describes the `bench_vector admin` tooling, how it maps to
vector storage internals, and current gaps/improvements.

## Summary

The admin command is a read-only diagnostics surface over the vector RocksDB
column families. It focuses on embedding metadata, lifecycle counts, HNSW graph
health, and consistency validation.

Primary entry points (CLI):

- `bench_vector admin stats --db-path <path> [--code <embedding>] [--json]`
- `bench_vector admin inspect --db-path <path> --code <embedding> [--json]`
- `bench_vector admin vectors --db-path <path> --code <embedding> [--state ...] [--sample N] [--vec-id N] [--json]`
- `bench_vector admin validate --db-path <path> [--code <embedding>] [--json]`
- `bench_vector admin rocksdb --db-path <path> [--json]`

## Data Sources

Admin pulls from these column families:

- `EmbeddingSpecs` (specs, model/dim/distance/storage)
- `GraphMeta` (entry point, max level, count, spec hash)
- `VecMeta` (lifecycle, max layer, created_at)
- `IdForward` / `IdReverse` (ID mapping)
- `IdAlloc` (next ID, free bitmap)
- `Edges` (neighbor links by layer)
- `BinaryCodes` (RaBitQ presence)
- `Pending` (async index queue)

## Implementation Notes / Limitations

- `admin stats` and `admin validate` iterate `EmbeddingSpecs` and then read
  other CFs per-embedding.
- `admin validate` uses **sampling** for ID mapping and orphan detection
  (first 1000 entries). It can miss inconsistencies.
- `admin rocksdb` estimates entry counts and sizes by scanning the first
  10,000 entries per CF. Sizes are **sampled bytes only**, not full DB size.
  The CLI prints `10000+` to indicate partial counts.
- Lifecycle counts roll `PendingDeleted` into `Deleted` for `stats`.

## Known Issues / Inconsistencies (from review)

1) **Stale EmbeddingSpecs after failed streaming index**  
   In `bench_vector index` (random streaming), the embedding is registered
   before the code checks for an existing index. If the check fails, the command
   aborts but leaves the embedding spec in `EmbeddingSpecs`, which then appears
   in `admin stats`/`embeddings list` with no vectors.

2) **Docs/examples still show query without `--embedding-code`**  
   The CLI now rejects implicit registration in `query`, but some examples still
   show `bench_vector query` without `--embedding-code`, which can lead to
   confusion if the dataset name does not match the registered embedding model.

3) **Validation count mismatch ambiguity**  
   `admin validate` compares `GraphMeta::Count` against
   `indexed + pending + deleted`. If `GraphMeta::Count` tracks only indexed
   vectors, this will produce a warning even when data is correct.

4) **RocksDB size reporting is not full size**  
   `admin rocksdb` reports sample sizes rather than true on-disk sizes. The
   current output can be misread as total size.

## Proposed Improvements

- **Add a read-only admin mode**: use read-only RocksDB handles to avoid
  contention with live workloads and reduce risk of accidental writes.
- **Lifecycle accounting**: show `PendingDeleted` as a separate count in
  `admin stats` rather than folding into `Deleted`.
- **Stronger validation**:
  - Reverse mapping checks (IdReverse -> IdForward)
  - VecMeta presence for all IdReverse entries
  - Vector payload existence for Indexed vectors
  - Optional full-scan mode with `--strict` (no sampling)
- **RocksDB stats**:
  - Use RocksDB properties (cf size, approximate counts) when available.
  - Clearly label sampled values vs. totals.
- **CLI examples**:
  - Update `bench_vector datasets` and README examples to show
    `--embedding-code` for `query` after `index`.
- **Index behavior**:
  - Defer embedding registration until after "fresh" checks pass, or clean up
    EmbeddingSpecs on abort.

