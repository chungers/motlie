TESTING PLAN (graph crate)

Goals
- Provide confidence in core graph correctness pre-VERSIONING.
- Provide versioning guarantees (history, restore, time travel, validity).
- Make regressions visible with fast, deterministic, high-signal tests.
- Define performance expectations with targeted benchmarks.

Scope
- Unit tests: encode/decode, key/value schemas, helper functions.
- Integration tests: db txns, CF interactions, indexing, GC.
- Subsystem lifecycle: start/stop, shutdown, background tasks.
- Metrics/info export: counters, gauges, info snapshots.
- Property tests: invariants across randomized ops.
- Benchmarks: hot paths, scan patterns, GC cost.
- Examples: kept minimal; not relied upon for correctness.

Test Infrastructure
- Time control: inject clock or use deterministic timestamps in tests.
- Deterministic ids: stable NodeId/EdgeId generation for assertions.
- Fixtures: helper builders for Node, Edge, Fragment, Summary.
- Storage: use temp dir per test, no shared state.
- Assertions: verify both forward and reverse CFs, summaries, indexes.

Baseline (pre-VERSIONING) Behavioral Tests (missing)
- Node CRUD: create, read, update, delete across txns.
- Edge CRUD: create, read, update, delete across txns.
- Summary writes: node/edge summary CF receives data on create/update.
- Summary index: ensure hash -> entity index entries exist and are unique.
- Reverse edge index: ensure dst -> incoming edges is correct.
- Forward + reverse consistency: single txn writes both sides.
- Fragment append: fragment CF appends without overwrite; idempotency.
- Fragment search: bm25/vector index integration at least smoke-level.
- Delete semantics: delete removes or tombstones and is discoverable.
- Error handling: missing nodes/edges fail gracefully with correct error type.
- Serialization: round-trip for keys/values, including optional fields.
- Lifecycle:
  - Graph start/stop is idempotent.
  - Background tasks exit on shutdown.
  - Subsystems join/flush before close.
  - GC lifecycle: start/shutdown, signal_shutdown, shutdown_arc (IMPLEMENTED gc.rs)
  - GC worker thread joins cleanly on shutdown (IMPLEMENTED gc.rs)
- Metrics/info:
  - Core counters increment (writes, reads, failures).
  - Info export returns expected snapshot fields.

VERSIONING Behavioral Tests (missing)
- VersionHistory:
  - Create Node/Edge writes history entry.
  - Update Node/Edge writes history entry with new ValidSince.
  - Delete writes history entry (tombstone or deletion state).
- Restore:
  - Restore by version returns correct snapshot.
  - Restore does not leave multiple CURRENT markers.
  - Restore reuses existing summary hash only if present.
- Time travel:
  - AsOf reads return correct version for Node/Edge.
  - AsOf reads respect ValidSince/ValidUntil.
  - AsOf reads across multiple updates ordered correctly.
- Validity range:
  - ValidUntil updates do not mutate in place without history.
  - Range queries return edges active during interval.
- Weight/history:
  - Weight updates create new version (no in-place mutation).
  - Weight/temporal fields propagate to history and indexes.
- Restore/index:
  - Restore marks prior CURRENT summary index entry as STALE.
  - Restore does not leave multiple CURRENT markers.
  - RestoreEdges batch also marks prior CURRENT entries STALE and records orphan candidates.
  - RestoreNode/RestoreEdge fail if referenced summary hash is missing (GC’d).
  - RestoreEdges skips edges with missing summaries and logs warning.

Index and Denormalization Tests (missing)
- Reverse edge temporal denormalization:
  - Forward update also updates reverse cf value in same txn.
  - Reverse scan honors ValidSince/ValidUntil without extra read.
- Summary index marker:
  - CURRENT and STALE markers obey single-current invariant.
  - Marker updates happen atomically with summary write.
- Hash prefix scans:
  - Prefix scan returns all entities sharing summary hash.
  - Reverse scan with padded version ordering yields latest first.

GC and Storage Tests (missing)
- Orphan summary tracking:
  - On summary replace, old summary hash is tracked.
- GC retention:
  - GC does not delete summaries that are still referenced by any current entity.
  - Retention window prevents deletion of recent summaries.
- Shared summary safety:
  - OrphanSummaries GC does not delete summary still referenced by other entities sharing the same SummaryHash.
- Fragment GC:
  - No GC of fragments (append-only).
  - Fragment summaries updated without fragment loss.

Concurrency and Atomicity (missing)
- Concurrent updates:
  - Two writers update same node/edge; history is consistent.
  - Conflicts are detected or resolved as designed.
- Multi-step txn:
  - Forward + reverse + summary + index are atomic or rolled back.
- Idempotency:
  - Replaying the same mutation does not corrupt indexes.
 - Lifecycle under load:
   - Shutdown during writes does not corrupt CFs.
   - Metrics still export after heavy load.

Property/Invariant Tests (missing)
- For any sequence of ops:
  - Exactly one CURRENT summary per entity.
  - Every CURRENT summary index references existing summary CF entry.
  - Reverse edge entries correspond to an existing forward edge version.
  - VersionHistory is monotonically ordered by ValidSince.

Performance/Benchmarks (missing)
- Hot scans:
  - Incoming edges scan with temporal filter.
  - Summary hash prefix scans.
  - AsOf reads with binary search / seek.
- Write amplification:
  - Create/update cost (number of CF writes) for Node/Edge.
- GC cost:
  - Orphan summary scan + delete vs refcount-based delete.

Documentation/Examples Validation (missing)
- Examples align with VERSIONING semantics.
- Docs mention delete history and restore semantics.
- Schema diagrams match actual key/value encoding.

Recommended Implementation Order
1) Baseline CRUD + serialization tests.
2) Summary/index consistency tests.
3) Reverse edge denormalization tests.
4) VersionHistory + time travel tests.
5) Restore semantics tests.
6) GC/refcount tests.
7) Property tests.
8) Performance benches.

Acceptance Criteria
- All tests above implemented with deterministic timestamps/ids.
- Coverage includes failure paths and invariants.
- Benchmarks track key regressions over time.

(codex, 2026-02-07, eval: updated to include lifecycle/metrics coverage and new VERSIONING gaps for restore markers, shared-summary safety, and versioned temporal fields.)

Post-Refactor (ARCH2) Testing Adjustments
- Processor API tests: direct sync calls validate mutation/query behavior without async overhead.
- Consumer wiring: Writer/Reader consumers use shared `Arc<Processor>` and still behave identically.
- Facade removal: tests no longer instantiate `Graph` (or verify it is optional/removed).
- Cache ownership: NameCache lives in Processor; verify cache prewarm + shared access across consumers.
- Subsystem lifecycle: start/stop idempotence, GC join, no tokio-blocking in GC thread.
- Transaction API: Writer.transaction uses Processor and preserves read-your-writes.
- Metrics/info: Processor/Subsystem metrics exposed and remain stable after refactor.
