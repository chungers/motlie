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
| T1.5 | Add `gc_config()` accessor method | `subsystem.rs` | Low |
| T1.6 | Re-export `GcConfig` from `subsystem.rs` for convenience | `mod.rs` | Low |

> (codex, 2026-01-30 19:54 UTC, PARTIAL) T1.5 needs a stored config (e.g., `gc_config: Option<GcConfig>`) to return; otherwise drop the accessor or store a clone alongside the handle.

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
| T2.4 | In `on_shutdown()`, join handles with timeout after writer close | `subsystem.rs` | Medium |
| T2.5 | Add configurable `shutdown_timeout: Duration` to Subsystem | `subsystem.rs` | Low |
| T2.6 | Log join success or timeout for each consumer | `subsystem.rs` | Low |

#### Implementation Details

```rust
// T2.1: Add field
pub struct Subsystem {
    // ... existing fields ...
    /// Consumer thread handles for graceful shutdown.
    consumer_handles: RwLock<Vec<JoinHandle<()>>>,
    /// Timeout for consumer shutdown joins.
    shutdown_timeout: Duration,
}

// T2.4: Join with timeout in on_shutdown
fn on_shutdown(&self) -> Result<()> {
    // ... GC + AsyncUpdater + Writer flush ...

    // 4. Join consumer threads with timeout
    let handles = std::mem::take(&mut *self.consumer_handles.write().expect("lock"));
    if !handles.is_empty() {
        tracing::debug!(subsystem = "vector", count = handles.len(), "Joining consumer threads");

        let deadline = Instant::now() + self.shutdown_timeout;
        for (i, handle) in handles.into_iter().enumerate() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                tracing::warn!(subsystem = "vector", consumer = i, "Shutdown timeout, abandoning join");
                break;
            }

            // Use park_timeout for bounded wait (thread::JoinHandle doesn't have timeout)
            // Best-effort: if channel closed, consumers will exit naturally
            match handle.join() {
                Ok(()) => tracing::debug!(subsystem = "vector", consumer = i, "Consumer joined"),
                Err(_) => tracing::warn!(subsystem = "vector", consumer = i, "Consumer panicked"),
            }
        }
    }

    Ok(())
}
```

> (codex, 2026-01-30 19:54 UTC, REJECT) The timeout approach is incorrect: `std::thread::JoinHandle::join()` blocks with no timeout. Implement timeout via cooperative shutdown or join in a helper thread with `recv_timeout`.

#### Acceptance Criteria

- [ ] Shutdown does not hang (joins are time-bounded)
- [ ] Logs indicate join success, timeout, or panic for each consumer
- [ ] Default timeout is reasonable (e.g., 5 seconds)
- [ ] Consumer handles are cleared after shutdown to prevent double-join

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
    // Setup
    let subsystem = Arc::new(Subsystem::new());
    let storage = create_test_storage(&subsystem);

    let gc_config = GcConfig::default().with_interval(Duration::from_secs(1));
    let async_config = AsyncUpdaterConfig::default();

    let (_writer, _reader) = subsystem.start_with_async(
        storage.clone(),
        WriterConfig::default(),
        ReaderConfig::default(),
        2,
        Some(async_config),
        Some(gc_config),
    );

    // Insert some vectors
    // ...

    // Shutdown should not panic and should complete within timeout
    let start = Instant::now();
    storage.shutdown().expect("shutdown failed");
    let elapsed = start.elapsed();

    assert!(elapsed < Duration::from_secs(10), "Shutdown took too long");

    // Verify shutdown ordering via tracing subscriber assertions
    // (GC shutdown log before AsyncUpdater shutdown log before Writer flush log)
}
```

> (codex, 2026-01-30 19:54 UTC, PARTIAL) Log-order assertions are brittle; prefer explicit test hooks/counters to assert shutdown ordering deterministically.

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

> (claude, 2026-01-30 16:00 UTC, RECOMMENDATION) Timeout-bounded with `shutdown_timeout: Duration` field. Default 5s. Log warnings on timeout.
> (codex, 2026-01-30 19:54 UTC, DECISION) Do not persist GC config; treat `enable_id_recycling` as a policy knob with clear operator guidance. Runtime changes across restarts are correctness-safe, with the caveat that ID reuse policy must align with product expectations.

---

## Estimated Effort

| Phase | Tasks | Complexity | Estimate |
|-------|-------|------------|----------|
| Phase 1 | 6 | Low-Medium | 0.5 day |
| Phase 2 | 6 | Low-Medium | 0.5 day |
| Phase 3 | 6 | Low-Medium | 0.5 day |
| **Total** | **18** | | **1.5 days** |

> (codex, 2026-01-30 19:54 UTC, PARTIAL) Plan is close; fix Phase 2 timeout approach and clarify GC config storage before execution.

---

## Notes

- Breaking changes are acceptable; no migration path or backward compatibility required.
- All callers of `start_with_async()` will be updated directly.
