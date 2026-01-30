# SUBSYSTEM: Lifecycle Gaps and Plan

## Goal

Ensure the vector subsystem fully manages the lifecycle of all runtime
dependencies it creates or requires: Storage, caches, async updater, and
garbage collection.

## Current State (Observed)

- Storage + CFs: wired via `Subsystem` (RocksdbSubsystem/StorageSubsystem).
- Caches: `EmbeddingRegistry` prewarmed in `on_ready()`, `NavigationCache` shared
  by `Processor` and `AsyncGraphUpdater`.
- Async updater: started in `start_with_async()`, shutdown in `on_shutdown()`.
- Garbage collection: **not** owned or managed by `Subsystem`; must be started
  manually via `GarbageCollector::start()` and shut down manually.
- Consumer threads: mutation/query consumers are spawned but handles are not
  retained or joined during shutdown (rely on channel close + writer flush).

## Risks / Impact

- GC can be left running after storage shutdown unless callers manage it.
- Subsystem shutdown does not wait for consumer threads to exit (best-effort).
- Operational burden: users must remember to wire GC separately.

## Project Plan

### Phase 1: Add GC Lifecycle to Subsystem

- Add `gc: RwLock<Option<GarbageCollector>>` to `Subsystem`.
- Add `start_with_gc()` (or extend `start_with_async()` to accept `GcConfig`).
- Start GC when config is provided; store handle in subsystem.
- In `on_shutdown()`, call `gc.shutdown()` before writer flush.

Acceptance:
- GC is started only when configured.
- GC is shut down exactly once and before storage shutdown.

### Phase 2: Optional Consumer Lifecycle Tracking

- Store mutation/query join handles in `Subsystem` (optional).
- On shutdown, signal consumers via writer close and join handles with timeout.
- Add docs: shutdown is best-effort if no runtime available.

Acceptance:
- Shutdown does not hang; joins are time-bounded.
- Logs indicate join success or timeout.

### Phase 3: Documentation and Tests

- Update `docs/API.md` and `docs/PHASE5.md` with subsystem lifecycle summary.
- Add a test that `Subsystem::start_with_async()` + GC config shuts down cleanly.

Acceptance:
- Docs describe managed lifecycle including GC.
- Tests cover shutdown ordering (GC -> async updater -> writer flush).

## Open Questions

- Do we want GC enabled by default in `start_with_async()` or opt-in only?
- Should GC share a cache with Processor/AsyncUpdater (if needed), or remain isolated?
- Is join-with-timeout acceptable, or should shutdown be fully blocking?
