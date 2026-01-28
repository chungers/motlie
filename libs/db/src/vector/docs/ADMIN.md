# bench_vector Admin

This document describes the `bench_vector admin` tooling, how it maps to
vector storage internals, and current gaps/improvements.

## Summary

The admin command is a read-only diagnostics surface over the vector RocksDB
column families. It focuses on embedding metadata, lifecycle counts, HNSW graph
health, and consistency validation.

Primary entry points (CLI):

- `bench_vector admin stats --db-path <path> [--code <embedding>] [--json] [--secondary]`
- `bench_vector admin inspect --db-path <path> --code <embedding> [--json]`
- `bench_vector admin vectors --db-path <path> --code <embedding> [--state ...] [--sample N] [--vec-id N] [--json]`
- `bench_vector admin validate --db-path <path> [--code <embedding>] [--json] [--strict]`
- `bench_vector admin rocksdb --db-path <path> [--json] [--secondary]`

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
- **GAP:** `admin stats` does a full `VecMeta` scan per embedding for lifecycle
  counts (O(N) per embedding). This can be very expensive at 10M+ scale and
  blocks on TransactionDB reads unless `--secondary` is used.

  **Status** (claude, 2025-01-27, ACKNOWLEDGED): This is a known limitation.
  Mitigation: Use `--secondary` for non-blocking reads. Future improvement could
  maintain lifecycle counts in GraphMeta to avoid full scan.
- `admin validate` uses **sampling** for ID mapping and orphan detection
  (first 1000 entries). It can miss inconsistencies.
- **GAP:** `admin validate --strict` is a full scan and can be long-running /
  disruptive on large DBs; there is no progress reporting or early-exit flag.

  **Status** (claude, 2025-01-27, FIXED): Added progress reporting via stderr
  during strict validation (prints count every 10,000 entries).
- `admin rocksdb` estimates entry counts and sizes by scanning the first
  10,000 entries per CF. Sizes are **sampled bytes only**, not full DB size.
  The CLI prints `10000+` to indicate partial counts and the column header
  shows "Bytes (sampled)" to clarify this is not on-disk size.
- Lifecycle counts now show `PendingDeleted` as a separate field (FIXED).

## Known Issues / Inconsistencies (from review)

1) **Stale EmbeddingSpecs after failed streaming index** ✅ FIXED
   ~~In `bench_vector index` (random streaming), the embedding is registered
   before the code checks for an existing index.~~

   **Fix**: Embedding registration is now deferred until after validation checks
   pass. See `commands.rs` index function with inline comment.

2) **Docs/examples still show query without `--embedding-code`** ✅ FIXED
   ~~Some CLI examples are updated (`bench_vector datasets` output), but the README
   still shows `bench_vector query` without `--embedding-code` in multiple places.~~

   **Fix** (claude, 2025-01-27): Updated all query examples in:
   - `bins/bench_vector/src/main.rs` (docstring)
   - `bins/bench_vector/README.md` (2 examples)
   - `libs/db/src/vector/docs/BASELINE.md` (3 examples)
   - `libs/db/src/vector/docs/PHASE8.md` (2 examples)
   - `libs/db/src/vector/benchmark/README.md` (1 example)

3) **Validation count mismatch ambiguity** ✅ FIXED
   ~~`admin validate` compares `GraphMeta::Count` against
   `indexed + pending + deleted`.~~

   **Fix**: Now compares `GraphMeta::Count` against `indexed` count only, since
   the graph tracks indexed vectors. Message shows pending/deleted separately.

4) **RocksDB size reporting is not full size** ✅ FIXED
   ~~`admin rocksdb` reports sample sizes rather than true on-disk sizes.~~

   **Fix**: Added `is_sampled` field to `ColumnFamilyStats`. CLI shows "Bytes (sampled)"
   in header and uses "10000+" notation when data is sampled.

## Implemented Improvements

- ✅ **Read-only admin mode**: Added `--secondary` flag to `admin stats` and
  `admin rocksdb` for read-only access via RocksDB secondary instance. This
  avoids contention with live workloads and prevents accidental writes.

- ✅ **Lifecycle accounting**: `PendingDeleted` is now shown as a separate count
  in `admin stats` rather than being folded into `Deleted`.

- ✅ **RocksDB stats clarification**: Column header changed to "Bytes (sampled)"
  and `is_sampled` field added to indicate when values are from sampling.

- ✅ **Index behavior**: Embedding registration deferred until after fresh checks
  pass in streaming index mode.

## Additional Improvements Implemented

- ✅ **Stronger validation** (chungers, 2025-01-27):
  - Reverse mapping checks (IdReverse -> IdForward): `validate_reverse_id_mappings`
  - VecMeta presence for all IdReverse entries: `validate_vec_meta_presence`
  - Vector payload existence for Indexed vectors: `validate_vector_payloads`
  - Optional full-scan mode with `--strict` flag (no sampling)

- ✅ **RocksDB stats** (chungers, 2025-01-27):
  - Added `estimated_num_keys` field from `rocksdb.estimate-num-keys` property
  - Available in secondary mode (`--secondary`) which uses DB instead of TransactionDB

## Remaining Improvements (TODO)

- **Secondary mode for remaining commands**:
  - Extend `--secondary` support to `admin inspect`, `admin vectors`, and
    `admin validate` subcommands.
  - Note: This requires duplicating many internal helper functions to work with
    both `DB` and `TransactionDB`. The key benefit (non-blocking reads) is already
    available for `stats` and `rocksdb` commands.
- **GAP:** Secondary mode uses a temporary directory under `/tmp` but does not
  clean up secondary DB files. This may accumulate on repeated runs.

  **Status** (claude, 2025-01-27, FIXED): Secondary mode now cleans up temp
  directory on drop using a wrapper that removes the directory when finished.
