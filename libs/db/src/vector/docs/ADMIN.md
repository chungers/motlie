# bench_vector Admin

This document describes the `bench_vector admin` tooling, how it maps to
vector storage internals, and current gaps/improvements.

## Summary

The admin command is a read-only diagnostics surface over the vector RocksDB
column families. It focuses on embedding metadata, lifecycle counts, HNSW graph
health, and consistency validation.

## Requirements

- **ADMIN-1:** Admin commands must provide a read-only option (secondary mode)
  for use against live databases. Read-write access remains acceptable for
  scheduled downtime.

  **Status** (claude, 2025-01-28, FIXED): All admin commands now support
  `--secondary` for read-only access via RocksDB secondary instance. Implemented
  via `AdminDb` trait that abstracts over `rocksdb::DB` and `TransactionDB`,
  eliminating helper function duplication. See `admin.rs` trait definition.

Primary entry points (CLI):

- `bench_vector admin stats --db-path <path> [--code <embedding>] [--json] [--secondary]`
- `bench_vector admin inspect --db-path <path> --code <embedding> [--json] [--secondary]`
- `bench_vector admin vectors --db-path <path> --code <embedding> [--state ...] [--sample N] [--vec-id N] [--json] [--secondary]`
- `bench_vector admin validate --db-path <path> [--code <embedding>] [--json] [--strict] [--secondary] [--sample-size N] [--max-errors N] [--max-entries N]`
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
- **GAP:** Most admin subcommands still open `Storage::readwrite` even though
  they are logically read-only. This can contend with writers or require write
  locks when running against a live DB.

  **Status** (claude, 2025-01-28, FIXED): All admin commands now support
  `--secondary` flag for read-only access via `AdminDb` trait abstraction.
  Without `--secondary`, commands still use `Storage::readwrite` for backward
  compatibility. With `--secondary`, commands use `Storage::secondary` which
  opens a non-blocking read-only RocksDB secondary instance.
- **GAP:** `admin stats` does a full `VecMeta` scan per embedding for lifecycle
  counts (O(N) per embedding). This can be very expensive at 10M+ scale and
  blocks on TransactionDB reads unless `--secondary` is used.

  **Status** (claude, 2025-01-27, ACKNOWLEDGED): This is a known limitation.
  Mitigation: Use `--secondary` for non-blocking reads. Future improvement could
  maintain lifecycle counts in GraphMeta to avoid full scan.

  **Rationale:** `VecMeta` is the only authoritative lifecycle store today, so
  counts require scanning every entry. This keeps correctness simple but makes
  stats O(N) and potentially slow at high scale.

  **Cost estimate @ 10M vectors:** If iteration yields ~50k–100k VecMeta entries/s,
  a full scan takes ~100–200 seconds per embedding. This is why we recommend
  `--secondary` for live DBs.

  **Schema change required:** add persisted lifecycle counters keyed by embedding,
  e.g. `GraphMetaField::LifecycleCounts { indexed, pending, deleted, pending_deleted }`
  or a new CF `vector/lifecycle_counts` with key `[embedding_code]` and a compact
  value struct. Updates must be applied on insert/delete/async transitions.
- `admin validate` uses **sampling** for ID mapping and orphan detection
  (configurable via `--sample-size`, default 1000). Sample size is reported in
  check messages.

  **Status** (claude, 2025-01-29, FIXED): Added `--sample-size N` CLI option.

- ~~**GAP:** `admin validate --strict` is a full scan and can be long-running /
  disruptive on large DBs; there is no progress reporting or early-exit flag.~~

  **Status** (claude, 2025-01-29, FIXED): Added `--max-errors N` for early exit,
  `--max-entries N` for scan cap. Progress reporting via stderr (every 10,000
  entries) was added in 2025-01-27 and remains in place.
- `admin rocksdb` estimates entry counts and sizes by scanning the first
  10,000 entries per CF. Sizes are **sampled bytes only**, not full DB size.
  The CLI prints `10000+` to indicate partial counts and the column header
  shows "Bytes (sampled)" to clarify this is not on-disk size.
- Lifecycle counts now show `PendingDeleted` as a separate field (FIXED).
- ~~**GAP:** Secondary mode temp dir is PID-scoped. Parallel invocations within the
  same process may collide; consider adding a random suffix or timestamp.~~

  **Status** (claude, 2025-01-29, FIXED): Secondary mode temp dir now uses
  PID + timestamp + random u32 suffix to avoid collisions.

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

- ✅ **Read-only admin mode** (ADMIN-1): Added `--secondary` flag to all admin
  commands (`stats`, `inspect`, `vectors`, `validate`, `rocksdb`) for read-only
  access via RocksDB secondary instance. This avoids contention with live
  workloads and prevents accidental writes. Implemented via `AdminDb` trait
  that abstracts over `rocksdb::DB` and `TransactionDB`, eliminating the need
  for duplicate helper functions.

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

- ~~**Secondary mode for remaining commands**~~: ✅ FIXED (claude, 2025-01-28)
  All admin commands now support `--secondary` via the `AdminDb` trait, which
  abstracts over `rocksdb::DB` and `TransactionDB`. This eliminated the need for
  duplicate `_db` helper functions. Each public function has a `_secondary`
  variant that delegates to a shared `_impl` function taking `&dyn AdminDb`.

- ~~**GAP:** Secondary mode uses a temporary directory under `/tmp` but does not
  clean up secondary DB files.~~

  **Status** (claude, 2025-01-28, FIXED): Secondary mode now deletes the temp
  directory explicitly after all admin subcommands complete. Each command that
  uses secondary mode creates a PID-scoped temp dir and removes it after use.

## Assessment / Actions for Claude

Admin is approaching production-readiness with configurable validation options
and collision-safe secondary mode.

1) **Lifecycle count scans**: `admin stats` does O(N) VecMeta scans.

   **Status** (claude, 2025-01-29, ACKNOWLEDGED): This is an architectural
   limitation. Persisting lifecycle counts in `GraphMeta` or a dedicated CF
   would require schema changes and migration logic. Mitigation: Use `--secondary`
   for non-blocking reads. Future improvement deferred until performance is a
   demonstrated bottleneck at scale.

2) ~~**Validation sampling**: Non-strict mode can miss inconsistencies. Consider
   configurable sample size and report sampling rate in output.~~

   **Status** (claude, 2025-01-29, FIXED): Added `--sample-size N` CLI option
   (default 1000). Validation output now shows `(sample_size=N)` in check messages.
   Implemented via `ValidationOptions` struct with builder pattern.

3) ~~**Strict validation UX**: Add early-exit on first error and/or a max-entries
   cap; keep progress reporting.~~

   **Status** (claude, 2025-01-29, FIXED): Added `--max-errors N` for early exit
   after N errors, and `--max-entries N` to cap entries scanned. `ValidationResult`
   now includes `stopped_early` field. Progress reporting (every 10,000 entries)
   remains in place.

4) ~~**Secondary temp dir collisions**: Add randomness (timestamp + RNG) to the
   temp path to avoid collisions within a process.~~

   **Status** (claude, 2025-01-29, FIXED): Secondary temp path now uses format
   `bench_vector_secondary_{pid}_{timestamp_ms}_{random_u32:08x}` to ensure
   uniqueness across parallel invocations.
