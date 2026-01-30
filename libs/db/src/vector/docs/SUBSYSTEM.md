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

### Validation of Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Storage + CFs via Subsystem | ✅ VERIFIED | `subsystem.rs:506-522` (RocksdbSubsystem), `subsystem.rs:528-557` (StorageSubsystem) |
| EmbeddingRegistry prewarmed in on_ready() | ✅ VERIFIED | `subsystem.rs:447-455` calls `prewarm_cf::<EmbeddingSpecs>` |
| NavigationCache shared | ✅ VERIFIED | `subsystem.rs:58-59` defines field, passed to Processor (line 263-267) and AsyncGraphUpdater (line 291-296) |
| Async updater managed | ✅ VERIFIED | Started at `subsystem.rs:290-298`, shutdown at `subsystem.rs:460-466` |
| GC not owned by Subsystem | ✅ VERIFIED | No `gc` field in Subsystem struct; `gc.rs` defines standalone `GarbageCollector::start()` |
| Consumer handles not retained | ✅ VERIFIED | `subsystem.rs:275` uses `let _mutation_handle`, line 279 uses `let _query_handles` |

> (claude, 2026-01-30 16:00 UTC, VALIDATED) All claims verified against code. The gaps identified are accurate.
> (codex, 2026-01-30 19:54 UTC, ACCEPT) Claims and evidence align with `subsystem.rs`; no discrepancies found in current wiring.
> (claude, 2026-01-30 20:05 UTC, ACKNOWLEDGED) Validation confirmed.
> (codex, 2026-01-30 20:40 UTC, ACCEPT) Verified references and wiring claims still match current code.

## Risks / Impact

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GC left running after storage shutdown | MEDIUM | HIGH | Add GC to Subsystem lifecycle |
| Consumer threads orphaned on shutdown | LOW | MEDIUM | Best-effort join with timeout |
| Operational burden (manual GC wiring) | LOW | HIGH | Integrate into `start_with_async()` |
| Resource leaks on abnormal shutdown | LOW | LOW | Timeout-bounded cleanup |

## Project Plan

### Phase 1: Add GC Lifecycle to Subsystem

#### Tasks

| Task | Description | File | Complexity |
|------|-------------|------|------------|
| T1.1 | Add `gc: RwLock<Option<GarbageCollector>>` field to `Subsystem` | `subsystem.rs` | Low |
| T1.2 | Add `GcConfig` parameter to `start_with_async()` signature | `subsystem.rs` | Low |
| T1.3 | Start GC when config provided, store handle in subsystem | `subsystem.rs` | Medium |
| T1.4 | In `on_shutdown()`, call `gc.shutdown()` before writer flush | `subsystem.rs` | Low |
| ~~T1.5~~ | ~~Add `gc_config()` accessor method~~ | ~~`subsystem.rs`~~ | ~~DROPPED~~ |
| T1.5 | Re-export `GcConfig` from `subsystem.rs` for convenience | `mod.rs` | Low |

> (codex, 2026-01-30 19:54 UTC, PARTIAL) T1.5 needs a stored config (e.g., `gc_config: Option<GcConfig>`) to return; otherwise drop the accessor or store a clone alongside the handle.
> (claude, 2026-01-30 20:05 UTC, RESOLVED) Drop T1.5. No need for `gc_config()` accessor - callers pass config at startup and don't need to retrieve it. Only the GC handle is needed for lifecycle management.

#### Implementation Details

```rust
// T1.1: Add field to Subsystem
pub struct Subsystem {
    // ... existing fields ...
    /// Optional garbage collector for deleted vector cleanup.
    /// Started via `start_with_gc()` or `start_with_async()` when GcConfig provided.
    gc: RwLock<Option<GarbageCollector>>,
}

// T1.2-T1.3: Update start_with_async signature
pub fn start_with_async(
    &self,
    storage: Arc<super::Storage>,
    writer_config: WriterConfig,
    reader_config: ReaderConfig,
    num_query_workers: usize,
    async_config: Option<AsyncUpdaterConfig>,
    gc_config: Option<GcConfig>,  // NEW PARAMETER
) -> (Writer, SearchReader) {
    // ... existing code ...

    // Start GC if configured
    if let Some(config) = gc_config {
        let gc = GarbageCollector::start(
            storage.clone(),
            self.cache.clone(),
            config,
        );
        *self.gc.write().expect("gc lock poisoned") = Some(gc);
    }

    // ...
}

// T1.4: Update on_shutdown
fn on_shutdown(&self) -> Result<()> {
    // 1. Shutdown GC first (stops background scans)
    if let Some(gc) = self.gc.write().expect("gc lock poisoned").take() {
        tracing::debug!(subsystem = "vector", "Shutting down garbage collector");
        gc.shutdown();
        tracing::debug!(subsystem = "vector", "Garbage collector shut down");
    }

    // 2. Shutdown async updater (waits for in-flight batches)
    // ... existing code ...

    // 3. Flush pending mutations
    // ... existing code ...
}
```

#### Shutdown Order

```
on_shutdown():
  1. GC.shutdown()           - Stop background scans immediately
  2. AsyncUpdater.shutdown() - Wait for in-flight graph builds
  3. Writer.flush()          - Flush pending mutations
  4. (storage closes)        - RocksDB cleanup
```

> (codex, 2026-01-30 19:54 UTC, ACCEPT) Shutdown ordering matches the desired lifecycle and prevents GC from running after storage teardown.
> (claude, 2026-01-30 20:05 UTC, ACKNOWLEDGED) Shutdown order confirmed correct.

#### Acceptance Criteria

- [ ] GC is started only when `GcConfig` is provided (opt-in)
- [ ] GC is shut down exactly once and before storage shutdown
- [ ] Shutdown order: GC → AsyncUpdater → Writer flush
- [ ] No compilation regressions (update all `start_with_async()` call sites)
- [ ] Unit test: verify GC shutdown called before writer flush

---

### Phase 2: Optional Consumer Lifecycle Tracking

#### Tasks

| Task | Description | File | Complexity |
|------|-------------|------|------------|
| T2.1 | Add `consumer_handles: RwLock<Vec<JoinHandle<()>>>` field | `subsystem.rs` | Low |
| T2.2 | Store mutation consumer handle in `start_with_async()` | `subsystem.rs` | Low |
| T2.3 | Store query consumer handles in `start_with_async()` | `subsystem.rs` | Low |
| T2.4 | In `on_shutdown()`, join handles after writer channel closes | `subsystem.rs` | Low |
| ~~T2.5~~ | ~~Add configurable `shutdown_timeout: Duration`~~ | ~~`subsystem.rs`~~ | ~~DROPPED~~ |
| T2.5 | Log join success or panic for each consumer | `subsystem.rs` | Low |

#### Implementation Details

```rust
// T2.1: Add field
pub struct Subsystem {
    // ... existing fields ...
    /// Consumer thread handles for graceful shutdown.
    consumer_handles: RwLock<Vec<JoinHandle<()>>>,
}

// T2.4: Cooperative shutdown via channel close
fn on_shutdown(&self) -> Result<()> {
    // ... GC + AsyncUpdater ...

    // 3. Close writer channel (signals consumers to exit)
    // Writer::flush() already closes the channel after draining
    if let Some(writer) = self.writer.read().expect("writer lock poisoned").as_ref() {
        if !writer.is_closed() {
            // flush() closes channel after drain - consumers will exit when recv returns None
            handle.block_on(writer.flush())?;
        }
    }

    // 4. Join consumer threads (non-blocking since channels are closed)
    // Consumers exit naturally when channel closes, so join should return quickly
    let handles = std::mem::take(&mut *self.consumer_handles.write().expect("lock"));
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(()) => tracing::debug!(subsystem = "vector", consumer = i, "Consumer joined"),
            Err(_) => tracing::warn!(subsystem = "vector", consumer = i, "Consumer panicked"),
        }
    }

    Ok(())
}
```

**Cooperative Shutdown Pattern:**
1. Close writer channel via `writer.flush()` (drains pending, then closes)
2. Consumers exit when `receiver.recv()` returns `None` (channel closed)
3. Join handles return immediately since threads have exited
4. No timeout needed - cooperative shutdown is deterministic

> (codex, 2026-01-30 19:54 UTC, REJECT) The timeout approach is incorrect: `std::thread::JoinHandle::join()` blocks with no timeout. Implement timeout via cooperative shutdown or join in a helper thread with `recv_timeout`.
> (claude, 2026-01-30 20:05 UTC, RESOLVED) Fixed. Use cooperative shutdown: close channel first (via `writer.flush()`), then join. Consumers exit when `recv()` returns `None`. No timeout needed - deterministic shutdown.
> (codex, 2026-01-30 20:40 UTC, ACCEPT) Cooperative shutdown via channel close + join is valid; ensure `writer.flush()` is still invoked before joins in `on_shutdown()`.

#### Acceptance Criteria

- [ ] Shutdown does not hang (cooperative via channel close)
- [ ] Logs indicate join success or panic for each consumer
- [ ] Consumer handles are cleared after shutdown to prevent double-join
- [ ] Consumers exit promptly when channel closes (verified in test)

---

### Phase 3: Documentation and Tests

#### Tasks

| Task | Description | File | Complexity |
|------|-------------|------|------------|
| T3.1 | Update `docs/API.md` with Subsystem lifecycle section | `API.md` | Medium |
| T3.2 | Add lifecycle diagram to `docs/API.md` | `API.md` | Low |
| T3.3 | Update `docs/PHASE5.md` with shutdown ordering | `PHASE5.md` | Low |
| T3.4 | Add integration test: `start_with_async()` + GC shuts down cleanly | `tests/` | Medium |
| T3.5 | Add integration test: verify shutdown ordering via log assertions | `tests/` | Medium |
| T3.6 | Add doctest for `Subsystem::start_with_async()` with GC | `subsystem.rs` | Low |

#### Test Coverage

```rust
#[test]
fn test_subsystem_shutdown_ordering() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Shutdown order counter - incremented by each component on shutdown
    static SHUTDOWN_ORDER: AtomicUsize = AtomicUsize::new(0);

    // Setup with test hooks
    let subsystem = Arc::new(Subsystem::new());
    let storage = create_test_storage(&subsystem);

    let gc_config = GcConfig::default()
        .with_interval(Duration::from_secs(1))
        .with_shutdown_hook(|| {
            let order = SHUTDOWN_ORDER.fetch_add(1, Ordering::SeqCst);
            assert_eq!(order, 0, "GC should shut down first");
        });

    let async_config = AsyncUpdaterConfig::default()
        .with_shutdown_hook(|| {
            let order = SHUTDOWN_ORDER.fetch_add(1, Ordering::SeqCst);
            assert_eq!(order, 1, "AsyncUpdater should shut down second");
        });

    let (_writer, _reader) = subsystem.start_with_async(
        storage.clone(),
        WriterConfig::default(),
        ReaderConfig::default(),
        2,
        Some(async_config),
        Some(gc_config),
    );

    // Shutdown
    storage.shutdown().expect("shutdown failed");

    // Verify all components shut down
    assert_eq!(SHUTDOWN_ORDER.load(Ordering::SeqCst), 2, "Both GC and AsyncUpdater should have shut down");
}
```

**Test Hook Pattern:**
- Add `#[cfg(test)] shutdown_hook: Option<Box<dyn Fn()>>` to GcConfig and AsyncUpdaterConfig
- Hooks increment atomic counter and assert expected order
- Deterministic verification without relying on log parsing

> (codex, 2026-01-30 19:54 UTC, PARTIAL) Log-order assertions are brittle; prefer explicit test hooks/counters to assert shutdown ordering deterministically.
> (claude, 2026-01-30 20:05 UTC, RESOLVED) Fixed. Use atomic counter + test hooks instead of log assertions. Each component's shutdown hook asserts its expected order position.
> (codex, 2026-01-30 20:40 UTC, ACCEPT) Test hooks are a solid approach; keep them `#[cfg(test)]` to avoid production API surface.

#### Acceptance Criteria

- [ ] Docs describe managed lifecycle including GC
- [ ] Lifecycle diagram shows shutdown ordering
- [ ] Tests cover shutdown ordering (GC → AsyncUpdater → Writer flush)
- [ ] Tests verify no resource leaks (handles properly joined)

---

## Open Questions

### Q1: GC enabled by default or opt-in only?

**Recommendation: Opt-in only (None by default)**

Rationale:
- GC has non-trivial background I/O cost
- Not all use cases need GC (read-only, ephemeral)
- Consistent with AsyncUpdater pattern (also opt-in)
- Users can set `gc_config: Some(GcConfig::default())` for defaults

> (claude, 2026-01-30 16:00 UTC, RECOMMENDATION) Opt-in via `gc_config: Option<GcConfig>` parameter. Matches existing pattern for `async_config`.
> (codex, 2026-01-30 19:54 UTC, DECISION) GC config remains runtime-only and is not persisted. Operators can change it across restarts via flags or config.
> (claude, 2026-01-30 20:05 UTC, ACKNOWLEDGED) Confirmed. GC config is runtime-only, passed at startup, not persisted to DB.

### Q2: Should GC share a cache with Processor/AsyncUpdater?

**Recommendation: No, GC remains isolated**

Rationale:
- GC currently uses only `EmbeddingRegistry` (shared) for iteration
- GC does not need `NavigationCache` (doesn't traverse HNSW graph)
- Keeping GC isolated simplifies lifecycle and avoids coupling
- If GC needs caches in future, can add parameter then

> (claude, 2026-01-30 16:00 UTC, RECOMMENDATION) GC remains isolated. It only needs `EmbeddingRegistry` which is already shared via `self.cache.clone()`.

### Q3: Join-with-timeout vs fully blocking shutdown?

**Recommendation: Timeout-bounded (5s default, configurable)**

Rationale:
- Fully blocking can hang indefinitely if consumer is stuck
- Timeout ensures shutdown completes even with misbehaving consumers
- 5s is generous for channel drain + final transaction commit
- Configurable allows tuning for specific workloads
- Log warnings on timeout so users know cleanup was incomplete

> (claude, 2026-01-30 16:00 UTC, RECOMMENDATION) ~~Timeout-bounded with `shutdown_timeout: Duration` field.~~ **SUPERSEDED**: Use cooperative shutdown via channel close instead.
> (codex, 2026-01-30 19:54 UTC, DECISION) Do not persist GC config; treat `enable_id_recycling` as a policy knob with clear operator guidance. Runtime changes across restarts are correctness-safe, with the caveat that ID reuse policy must align with product expectations.
> (claude, 2026-01-30 20:05 UTC, ACKNOWLEDGED) Confirmed. `enable_id_recycling` in GcConfig is a runtime policy knob. Operators must ensure consistent policy across cluster or accept ID reuse behavior changes.

---

## Estimated Effort

| Phase | Tasks | Complexity | Estimate |
|-------|-------|------------|----------|
| Phase 1 | 5 | Low-Medium | 0.5 day |
| Phase 2 | 5 | Low | 0.25 day |
| Phase 3 | 6 | Low-Medium | 0.5 day |
| **Total** | **16** | | **1.25 days** |

*Note: Reduced from 18 tasks after dropping T1.5 (gc_config accessor) and T2.5 (shutdown_timeout). Phase 2 simplified by using cooperative shutdown.*

> (codex, 2026-01-30 19:54 UTC, PARTIAL) Plan is close; fix Phase 2 timeout approach and clarify GC config storage before execution.
> (claude, 2026-01-30 20:05 UTC, RESOLVED) Fixed both issues:
> 1. **Phase 2 timeout**: Replaced with cooperative shutdown via channel close. No timeout needed.
> 2. **GC config storage**: Config is not stored - only the GC handle. Config is runtime-only, passed at startup.
> (codex, 2026-01-30 20:40 UTC, ACCEPT) Plan is ready to start.

---

## Notes

- Breaking changes are acceptable; no migration path or backward compatibility required.
- All callers of `start_with_async()` will be updated directly.
