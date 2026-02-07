# ARCH2: Align Graph Architecture with Vector Processor Pattern

---

## ⚠️ CRITICAL BUGS TO FIX DURING REFACTOR ⚠️

> (claude, 2026-02-07) These bugs exist in the current graph crate and MUST be
> fixed as part of this refactor. Do not close this work until all are resolved.

### BUG 1: GC Worker Handle Discarded

**File:** `libs/db/src/graph/subsystem.rs:234`

```rust
// CURRENT (BROKEN):
let _handle = gc.clone().spawn_worker();  // ← HANDLE DISCARDED!
*self.gc.write().expect("gc lock") = Some(gc);
```

**Impact:** The GC worker runs detached. Subsystem has no way to join it on shutdown.

**Fix:** Store handle in GarbageCollector struct (see vector::GarbageCollector pattern).
(codex, 2026-02-07, decision: verified in `libs/db/src/graph/subsystem.rs:234` — handle is discarded.)

---

### BUG 2: GC Shutdown Doesn't Join Worker

**File:** `libs/db/src/graph/subsystem.rs:470`

```rust
// CURRENT (BROKEN):
if let Some(gc) = self.gc.write().expect("gc lock").take() {
    gc.shutdown();  // ← SIGNALS FLAG BUT DOESN'T WAIT!
}
// Control returns immediately - worker may still be running!
// Storage may close while GC is mid-cycle!
```

**Impact:** Race condition. GC worker can outlive Storage, causing use-after-free or panics.

**Fix:** `shutdown(self)` must join worker thread before returning (see vector::GarbageCollector::shutdown).
(codex, 2026-02-07, decision: verified in `libs/db/src/graph/subsystem.rs:468-470` — shutdown does not join.)

---

### BUG 3: GC Uses Tokio Task for Blocking Work

**File:** `libs/db/src/graph/gc.rs:547`

```rust
// CURRENT (WRONG PATTERN):
pub fn spawn_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // ... blocking RocksDB operations inside async context!
    })
}
```

**Impact:** Blocking RocksDB I/O inside tokio task blocks the async runtime. Should use `std::thread` for blocking work.

**Fix:** Use `std::thread::spawn` and return `std::thread::JoinHandle<()>` (see vector::GarbageCollector::start).
(codex, 2026-02-07, decision: verified in `libs/db/src/graph/gc.rs:553` — uses `tokio::spawn` for blocking RocksDB work.)

---

### Verification Checklist

- [ ] `subsystem.rs` no longer discards GC worker handle
- [ ] `gc.shutdown()` blocks until worker thread completes
- [ ] GC uses `std::thread::spawn` not `tokio::spawn`
- [ ] Shutdown order: flush → consumers → GC (GC last)
- [ ] No panics or errors during `cargo test` shutdown sequences

---

## Problem

The graph crate currently mixes two roles:

- **Internal, synchronous execution** (direct RocksDB reads/writes)
- **Public, async channel APIs** (mutation/query mpsc/mpmc consumers)

In the vector crate, these roles are explicitly separated via an internal
`Processor` and public async `Writer`/`Reader` APIs that hide it. The graph
crate lacks this separation and instead uses `Graph` as both processor and
public entry point, which makes it harder to:

- Evolve internal logic without changing public APIs
- Add shared caches or execution state
- Maintain architectural consistency across subsystems

---

## Vector Crate Pattern (Reference)

### Internal, synchronous core

- `vector::processor::Processor` (pub(crate))
- Owns storage + caches + registries
- Used by both mutations and queries

### Public, async APIs

- `vector::writer::{Writer, Consumer}`
  - MPSC channel
  - `Consumer` holds `Arc<Processor>`
  - `spawn_mutation_consumer_with_storage(...)` constructs Processor internally

- `vector::reader::{Reader, Consumer, ProcessorConsumer}`
  - MPMC (flume)
  - `Reader` holds `Arc<Processor>` for `SearchKNN`
  - `spawn_query_consumers_with_storage(...)` constructs Processor internally

**Key point:** public API is async + channel-based; internal Processor is sync and hidden.

---

## Graph Crate Pattern (Current)

### Mutations

- `graph::writer::{Writer, Consumer}`
- `Consumer<P>` generic over a `Processor` trait
- `Graph` implements Processor and is used directly
- No internal `Processor` struct

### Queries

- `graph::reader::{Reader, Consumer}`
- `Consumer<P>` uses a Processor trait that only exposes `storage()`
- `Reader` does not hold a Processor

**Result:** Graph mixes internal sync execution with public async wiring.

---

## Recent Implementation Changes (as of 2026-02-07)

- **Subsystem-managed lifecycle**: `Subsystem::start(...)` now wires Writer/Reader, spawns consumers, optionally starts GC, and manages shutdown ordering (flush → join consumers → stop GC). `graph/subsystem.rs`
- **GC and RefCount (current + versioning changes)**: GC is integrated for stale index cleanup; VERSIONING adds OrphanSummaries CF + OrphanSummaryGc to defer deletion when RefCount→0 (retention window for rollback). This replaces inline deletion with an orphan index scan. `graph/gc.rs`, `graph/subsystem.rs`, `graph/schema.rs`
- **VERSIONING design underway**: VERSIONING introduces ValidSince/ValidUntil, VersionHistory CFs, bitemporal semantics (system time + business ActivePeriod), and orphan-index-based GC for rollback. This increases the amount of core sync logic that should live behind an internal Processor. `graph/docs/VERSIONING.md`
- **Public APIs unchanged**: Writer/Reader are still the public async channel APIs; `Graph` remains the processor used by consumers.

These changes improve robustness and lifecycle management but **do not yet introduce** an internal `graph::processor::Processor` or hide `Graph` behind the public APIs.

---

## Proposed Refactor (Match Vector Pattern)

### 1) Introduce `graph::processor::Processor` (pub(crate))

Responsibilities:

- Owns `Arc<Storage>`
- Owns shared caches (e.g., name cache, future graph caches)
- Exposes synchronous methods for core graph operations
- Encapsulates VERSIONING semantics (ValidSince/ValidUntil, VersionHistory writes, reverse-edge denormalization, summary index maintenance, RefCount + orphan index writes, and GC hooks)

### 2) Update graph mutation path

- `graph::writer::Consumer` holds `Arc<graph::processor::Processor>`
- Add helper functions mirroring vector:
  - `spawn_mutation_consumer_with_storage(...)`
  - `spawn_mutation_consumer_with_processor(...)` (pub(crate))
- Ensure mutation helpers cover VERSIONING workflows (content update, topology change, rollback, orphan index updates) without exposing sync internals

### 3) Update graph query path

- Add a Processor-backed query consumer (like `vector::ProcessorConsumer`)
- Provide reader construction helpers:
  - `create_reader_with_storage(...)`
  - `spawn_query_consumers_with_storage(...)`
- Centralize time-travel query logic inside Processor (current vs as_of filtering, reverse scans, version lookup)

### 4) Keep `Graph` as facade (optional)

- `Graph` can wrap an `Arc<Processor>` or `Arc<Storage>`
- Public API can remain stable while internal implementation migrates
- **Desired outcome:** the refactor should make it possible to remove `Graph` entirely over time, leaving `Processor` + public async `Writer`/`Reader` as the primary entry points
- VERSIONING increases the value of this separation by keeping complex temporal + GC logic off the async boundary

---

## Benefits

- **Encapsulation:** Internal sync logic is isolated
- **API Stability:** Public async APIs remain unchanged
- **Consistency:** Aligns graph with vector/fulltext patterns
- **Future-proof:** Easier to add caches or new execution modes
- **Simplification (success criterion):** Fewer public types and clearer ownership boundaries

---

## Summary

Vector’s architecture clearly separates synchronous internal execution
(`Processor`) from async public APIs (`Writer`/`Reader`). Refactoring graph
to follow the same pattern will make the system more consistent, easier to
extend, and safer to evolve without breaking public interfaces.

**Status:** Not implemented yet; recent changes are additive (GC/RefCount + lifecycle) and do not change the processor/public API boundary.

---

## Can `Graph` Be Removed Entirely?

Yes — it is structurally possible. The graph crate already routes most public
usage through async `Writer`/`Reader` channel APIs. If those APIs construct and
hold the internal `Processor` (as in the vector crate), `Graph` becomes a thin
facade with no unique responsibilities. At that point it can be deprecated and
eventually removed.

### Preconditions

- Any `Graph`-specific helpers must be moved to `Processor` or to the public
  async APIs.
- Call sites that currently construct `Graph::new(storage)` must have equivalent
  construction paths via `Processor` + `Writer`/`Reader` helpers.
- Tests/examples should be migrated to use the async APIs or direct `Processor`
  calls.

### Desired Outcome

`Processor` becomes the only synchronous core, and `Writer`/`Reader` become the
public entry points. `Graph` is no longer required for external users.

---

## Concrete Refactoring Proposal

> (claude, 2026-02-07) This section provides implementation details based on
> analysis of the vector::Processor pattern and current graph crate structure.

### Analysis: Vector Processor Design Pattern

The vector crate's `Processor` serves as a **central state hub** with these characteristics:

```
┌─────────────────────────────────────────────────────────────────┐
│                    vector::Processor                             │
├─────────────────────────────────────────────────────────────────┤
│ Owned State:                                                     │
│   - Arc<Storage>              // RocksDB access                  │
│   - Arc<EmbeddingRegistry>    // Embedding metadata              │
│   - DashMap<IdAllocators>     // Per-embedding ID allocation     │
│   - DashMap<RaBitQEncoders>   // Per-embedding quantization      │
│   - DashMap<HnswIndices>      // Per-embedding graph indices     │
│   - Arc<NavigationCache>      // Shared HNSW layer cache         │
│   - Arc<BinaryCodeCache>      // Shared RaBitQ code cache        │
├─────────────────────────────────────────────────────────────────┤
│ Sync API (pub(crate)):                                           │
│   - insert_vector()           // Single vector + HNSW            │
│   - insert_batch()            // Batch insert + HNSW             │
│   - delete_vector()           // Soft delete                     │
│   - search()                  // HNSW + rerank                   │
│   - search_with_config()      // Configurable search             │
│   - get_or_create_allocator() // Lazy allocator init             │
│   - get_or_create_index()     // Lazy HNSW init                  │
│   - get_or_create_encoder()   // Lazy RaBitQ init                │
└─────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
    ┌────▼─────┐                     ┌─────▼────┐
    │  Writer  │ (pub)               │  Reader  │ (pub)
    │  MPSC    │                     │  MPMC    │
    │  async   │                     │  async   │
    └────┬─────┘                     └─────┬────┘
         │                                 │
    ┌────▼─────┐                     ┌─────▼────┐
    │ Consumer │                     │ Consumer │
    │holds Arc │                     │holds Arc │
    │<Processor>                     │<Processor>
    └──────────┘                     └──────────┘
```

Key design decisions:
1. **Processor is pub(crate)** - never exposed to users
2. **Writer/Reader are pub** - users interact only with async channel APIs
3. **Processor owns all caches** - not Storage
4. **Lazy initialization** - DashMap for lock-free concurrent access
5. **Cache updates deferred** - applied after transaction commit

### Current Graph Structure (to be refactored)

```rust
// graph/mod.rs - Current
pub struct Graph {
    storage: Arc<Storage>,
}

#[async_trait]
impl writer::Processor for Graph {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;
}
```

Problems:
- `Graph` is public but should be internal
- `Graph` is thin wrapper with no caches
- Processor trait is async (unnecessary overhead for sync operations)
- NameCache lives in Storage, not Processor

---

## What Does NOT Change

> (claude, 2026-02-07) Critical: the public API surface remains identical.

### Unchanged Public Types and Methods

| Type | Method | Status |
|------|--------|--------|
| `Writer` | `send(Vec<Mutation>)` | **Unchanged** |
| `Writer` | `flush()` | **Unchanged** |
| `Writer` | `send_sync()` | **Unchanged** |
| `Writer` | `transaction()` | **Unchanged** |
| `Reader` | `send_query(Query)` | **Unchanged** |
| `Runnable` (mutations) | `run(self, &Writer)` | **Unchanged** |
| `Runnable<Reader>` (queries) | `run(self, &Reader, Duration)` | **Unchanged** |
| `Mutation` enum | All variants | **Unchanged** |
| `Query` enum | All variants | **Unchanged** |

### Runnable Trait Pattern (Unchanged)

The `Runnable` trait is syntactic sugar for channel operations:

```rust
// User writes:
AddNode { id, name, ... }.run(&writer).await?;

// Which calls:
writer.send(vec![Mutation::AddNode(...)]).await?;

// User writes:
NodeById::new(id, None).run(&reader, timeout).await?;

// Which calls:
reader.send_query(Query::NodeById(dispatch)).await?;
```

This pattern is **not affected** by the refactor. Users continue to:
1. Construct mutation/query types
2. Call `.run()` with Writer or Reader
3. Await the result

### What Changes (Internal Only)

The **only** change is inside `Consumer`:

```rust
// BEFORE: Consumer uses async Processor trait
impl<P: Processor> Consumer<P> {
    async fn run(mut self) {
        while let Some(mutations) = self.receiver.recv().await {
            // Async trait method (virtual dispatch + async overhead)
            self.processor.process_mutations(&mutations).await?;
        }
    }
}

// AFTER: Consumer uses concrete Processor struct
impl Consumer {
    async fn run(mut self) {
        while let Some(mutations) = self.receiver.recv().await {
            // Direct sync call (no trait, no async overhead)
            self.processor.process_mutations(&mutations)?;
        }
    }
}
```

**Benefits of sync Processor:**
- No async trait overhead (no Boxing, no vtable)
- Simpler error handling
- Matches vector crate pattern
- Processor methods can be called directly in tests

---

## Implementation Tasks

> (claude, 2026-02-07) Breaking change refactor - no migration/deprecation needed.

### Task 1: Create `graph::processor::Processor` struct

**File:** `libs/db/src/graph/processor.rs` (new)

```rust
// (claude, 2026-02-07) New internal processor for graph operations

/// Internal processor for synchronous graph operations.
///
/// This is the single source of truth for graph state:
/// - Storage access (RocksDB)
/// - Name cache (NameHash ↔ String deduplication)
/// - Future: versioning cache, summary cache
pub(crate) struct Processor {
    /// RocksDB storage (read-write mode required for mutations)
    storage: Arc<Storage>,

    /// Name cache for NameHash ↔ String resolution
    /// Moved from Storage to Processor for proper ownership
    name_cache: Arc<NameCache>,

    /// GC configuration (optional, for background cleanup)
    gc_config: Option<GraphGcConfig>,
}

impl Processor {
    /// Create a new Processor with storage.
    pub fn new(storage: Arc<Storage>) -> Self {
        // Extract or create name cache
        let name_cache = storage.cache().clone();
        Self {
            storage,
            name_cache,
            gc_config: None,
        }
    }

    /// Create with GC configuration.
    pub fn with_gc(mut self, config: GraphGcConfig) -> Self {
        self.gc_config = Some(config);
        self
    }

    // --- Storage Access ---

    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    pub fn transaction_db(&self) -> Result<&rocksdb::TransactionDB> {
        self.storage.transaction_db()
    }

    // --- Cache Access ---

    pub fn name_cache(&self) -> &Arc<NameCache> {
        &self.name_cache
    }

    // --- Synchronous Mutation API ---

    /// Process mutations synchronously within a transaction.
    /// This is the core mutation method called by Writer consumers.
    pub fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();

        for mutation in mutations {
            mutation.execute_with_cache(&txn, txn_db, &self.name_cache)?;
        }

        txn.commit()?;
        Ok(())
    }

    /// Execute a single mutation in a new transaction.
    pub fn execute_mutation(&self, mutation: &Mutation) -> Result<()> {
        self.process_mutations(std::slice::from_ref(mutation))
    }
}
```

**Rationale:**
- Mirrors vector::Processor ownership pattern
- Sync-only API (no async trait overhead)
- Name cache moved to Processor (single owner)
- Extensible for future caches (versioning, summaries)

---

### Task 2: Update `graph::writer` module

**File:** `libs/db/src/graph/writer.rs`

Changes:
1. Remove `Processor` async trait (replace with direct Processor struct usage)
2. Update `Consumer` to hold `Arc<processor::Processor>`
3. Add construction helpers

```rust
// (claude, 2026-02-07) Refactored writer module

use super::processor::Processor;

// REMOVE: pub trait Processor (async trait no longer needed)

/// Mutation consumer that processes batches from the channel.
pub(crate) struct Consumer {
    receiver: mpsc::Receiver<Vec<Mutation>>,
    processor: Arc<Processor>,  // Changed from generic P: Processor
    config: WriterConfig,
}

impl Consumer {
    pub fn new(
        receiver: mpsc::Receiver<Vec<Mutation>>,
        processor: Arc<Processor>,
        config: WriterConfig,
    ) -> Self {
        Self { receiver, processor, config }
    }

    /// Run the consumer loop (blocking).
    pub async fn run(mut self) {
        while let Some(mutations) = self.receiver.recv().await {
            // Sync call to processor (no async overhead)
            if let Err(e) = self.processor.process_mutations(&mutations) {
                tracing::error!(error = %e, "Failed to process mutations");
            }
        }
    }
}

/// Spawn a mutation consumer with storage.
///
/// This is the primary public construction helper.
/// Creates Processor internally and spawns consumer task.
pub fn spawn_mutation_consumer_with_storage(
    storage: Arc<Storage>,
    config: WriterConfig,
) -> (Writer, JoinHandle<()>) {
    let processor = Arc::new(Processor::new(storage));
    spawn_mutation_consumer_with_processor(processor, config)
}

/// Spawn a mutation consumer with existing processor.
///
/// Used when Processor is shared (e.g., with Reader).
pub(crate) fn spawn_mutation_consumer_with_processor(
    processor: Arc<Processor>,
    config: WriterConfig,
) -> (Writer, JoinHandle<()>) {
    let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
    let consumer = Consumer::new(receiver, processor, config.clone());
    let handle = tokio::spawn(async move { consumer.run().await });
    (Writer::new(sender), handle)
}
```

---

### Task 3: Update `graph::reader` module

**File:** `libs/db/src/graph/reader.rs`

Changes:
1. Remove `Processor` trait (use Storage directly or Arc<Processor>)
2. Update `Reader` to optionally hold `Arc<Processor>` for cache access
3. Add construction helpers

```rust
// (claude, 2026-02-07) Refactored reader module

use super::processor::Processor;

/// Query reader with MPMC channel.
#[derive(Clone)]
pub struct Reader {
    sender: flume::Sender<Query>,
    processor: Option<Arc<Processor>>,  // Optional for cache access
}

impl Reader {
    /// Get processor reference (for queries needing cache).
    pub(crate) fn processor(&self) -> Option<&Arc<Processor>> {
        self.processor.as_ref()
    }
}

/// Query consumer that processes queries from the channel.
pub(crate) struct Consumer {
    receiver: flume::Receiver<Query>,
    storage: Arc<Storage>,
    processor: Option<Arc<Processor>>,  // For cache-aware queries
}

impl Consumer {
    pub async fn run(self) {
        while let Ok(query) = self.receiver.recv_async().await {
            // Query execution uses storage + optional processor
            if let Err(e) = query.process(&self.storage, self.processor.as_ref()).await {
                tracing::error!(error = %e, "Query processing failed");
            }
        }
    }
}

/// Create a reader with storage.
pub fn create_reader_with_storage(storage: Arc<Storage>) -> Reader {
    let processor = Arc::new(Processor::new(storage.clone()));
    create_reader_with_processor(processor)
}

/// Create a reader with existing processor.
pub(crate) fn create_reader_with_processor(processor: Arc<Processor>) -> Reader {
    let (sender, _receiver) = flume::unbounded();
    Reader {
        sender,
        processor: Some(processor),
    }
}

/// Spawn query consumers with storage.
pub fn spawn_query_consumers_with_storage(
    storage: Arc<Storage>,
    num_workers: usize,
) -> (Reader, Vec<JoinHandle<()>>) {
    let processor = Arc::new(Processor::new(storage));
    spawn_query_consumers_with_processor(processor, num_workers)
}

/// Spawn query consumers with existing processor.
pub(crate) fn spawn_query_consumers_with_processor(
    processor: Arc<Processor>,
    num_workers: usize,
) -> (Reader, Vec<JoinHandle<()>>) {
    let (sender, receiver) = flume::unbounded();
    let storage = processor.storage().clone();

    let handles: Vec<_> = (0..num_workers)
        .map(|_| {
            let consumer = Consumer {
                receiver: receiver.clone(),
                storage: storage.clone(),
                processor: Some(processor.clone()),
            };
            tokio::spawn(async move { consumer.run().await })
        })
        .collect();

    let reader = Reader {
        sender,
        processor: Some(processor),
    };

    (reader, handles)
}
```

---

### Task 4: Update `graph::subsystem` module

**File:** `libs/db/src/graph/subsystem.rs`

Changes:
1. Use new construction helpers
2. Remove Graph usage
3. Share Processor between Writer and Reader

```rust
// (claude, 2026-02-07) Subsystem uses shared Processor

impl Subsystem {
    pub fn start(
        &mut self,
        storage: Arc<Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
    ) -> Result<()> {
        // Create shared processor
        let processor = Arc::new(Processor::new(storage.clone()));

        // Spawn mutation consumer with shared processor
        let (writer, writer_handle) = spawn_mutation_consumer_with_processor(
            processor.clone(),
            writer_config,
        );

        // Spawn query consumers with shared processor
        let (reader, reader_handles) = spawn_query_consumers_with_processor(
            processor.clone(),
            reader_config.num_workers,
        );

        self.writer = Some(writer);
        self.reader = Some(reader);
        self.processor = Some(processor);
        // ... rest of lifecycle management

        Ok(())
    }
}
```

---

### Task 5: Remove `Graph` struct

**File:** `libs/db/src/graph/mod.rs`

Changes:
1. Delete `pub struct Graph`
2. Delete `impl Processor for Graph`
3. Update module exports

```rust
// (claude, 2026-02-07) Graph removed - Processor is the internal core

// REMOVED:
// pub struct Graph { ... }
// impl Processor for Graph { ... }

// Public exports (unchanged for users):
pub use reader::{Reader, create_reader_with_storage, spawn_query_consumers_with_storage};
pub use writer::{Writer, spawn_mutation_consumer_with_storage, WriterConfig};
pub use mutation::*;
pub use query::*;

// Internal only:
pub(crate) use processor::Processor;
```

---

### Task 6: Update Transaction API

**File:** `libs/db/src/graph/transaction.rs`

The Transaction API currently gets storage from Writer. Update to use Processor:

```rust
// (claude, 2026-02-07) Transaction uses Processor for cache access

impl Writer {
    /// Create a transaction for read-your-writes semantics.
    pub fn transaction(&self) -> Result<Transaction<'_>> {
        let processor = self.processor.as_ref()
            .ok_or_else(|| anyhow!("Writer not configured with processor"))?;

        let txn_db = processor.transaction_db()?;
        let txn = txn_db.transaction();
        let name_cache = processor.name_cache().clone();

        Ok(Transaction::new(txn, txn_db, name_cache))
    }
}
```

---

## Migration Checklist

> (claude, 2026-02-07) Steps to complete the refactor.

- [ ] **Task 1:** Create `graph/processor.rs` with Processor struct
- [ ] **Task 2:** Refactor `graph/writer.rs` - remove trait, update Consumer
- [ ] **Task 3:** Refactor `graph/reader.rs` - remove trait, add processor
- [ ] **Task 4:** Update `graph/subsystem.rs` - use shared Processor
- [ ] **Task 5:** Remove `Graph` from `graph/mod.rs`
- [ ] **Task 6:** Update `graph/transaction.rs` - use Processor
- [ ] **Task 7:** Update tests - use new construction helpers
- [ ] **Task 8:** Update examples and benchmarks
- [ ] **Task 9:** Verify all tests pass
- [ ] **Task 10:** Update documentation

---

## Future Enhancements

> (claude, 2026-02-07) Additional improvements enabled by this refactor.

### Versioning Cache

With Processor owning caches, we can add a versioning cache:

```rust
pub(crate) struct Processor {
    // ... existing fields ...

    /// Cache for recent version lookups (point-in-time query optimization)
    version_cache: Arc<VersionCache>,
}

pub(crate) struct VersionCache {
    /// Recent node versions: (NodeId, Timestamp) → NodeCfValue
    nodes: DashMap<(Id, TimestampMilli), NodeCfValue>,
    /// Recent edge versions: (EdgeKey, Timestamp) → EdgeCfValue
    edges: DashMap<(EdgeKey, TimestampMilli), ForwardEdgeCfValue>,
}
```

### Summary Cache

For frequently accessed summaries:

```rust
pub(crate) struct SummaryCache {
    /// Node summaries by hash
    nodes: DashMap<SummaryHash, NodeSummary>,
    /// Edge summaries by hash
    edges: DashMap<SummaryHash, EdgeSummary>,
}
```

### Metrics and Observability

Processor can expose metrics:

```rust
impl Processor {
    pub fn metrics(&self) -> ProcessorMetrics {
        ProcessorMetrics {
            name_cache_size: self.name_cache.len(),
            mutations_processed: self.mutation_count.load(Ordering::Relaxed),
            // ...
        }
    }
}
```

---

## Success Criteria

> (claude, 2026-02-07) How to verify the refactor is complete.

1. **No public `Graph` type** - Users construct via `spawn_*` helpers only
2. **Processor is pub(crate)** - Internal implementation detail
3. **Shared caches** - NameCache owned by Processor, not Storage
4. **Consistent with vector** - Same construction patterns
5. **All tests pass** - No behavioral changes
6. **API stability** - Writer/Reader public interfaces unchanged

---

## Lifecycle Management

> (claude, 2026-02-07) Subsystem lifecycle following vector:: pattern.

### Vector Pattern (Reference)

The vector subsystem manages component lifecycle with explicit ownership:

```
┌─────────────────────────────────────────────────────────────────┐
│                     vector::Subsystem                            │
├─────────────────────────────────────────────────────────────────┤
│ Owned State:                                                     │
│   cache: Arc<EmbeddingRegistry>      // Pre-warmed on startup    │
│   nav_cache: Arc<NavigationCache>    // Shared HNSW cache        │
│   writer: RwLock<Option<Writer>>     // For shutdown flush       │
│   async_updater: RwLock<Option<AsyncGraphUpdater>>               │
│   gc: RwLock<Option<GarbageCollector>>                           │
│   consumer_handles: RwLock<Vec<JoinHandle>>                      │
├─────────────────────────────────────────────────────────────────┤
│ Lifecycle Methods:                                               │
│   new()           → Create with empty caches                     │
│   on_ready(db)    → Pre-warm caches from TransactionDB           │
│   start()         → Spawn consumers, start GC/updater            │
│   on_shutdown()   → Flush, join, cleanup (ordered)               │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design:**
- `RwLock<Option<T>>` for deferred initialization
- Components return `Self` (not JoinHandle) with embedded worker handles
- Shutdown methods take ownership (`fn shutdown(self)`) and join workers
- Explicit ordering: flush → async updater → consumers → GC

### Current Graph Issues

> (claude, 2026-02-07) Bugs identified in current graph::Subsystem.

**Issue 1: GC Worker Handle Discarded**
```rust
// Current (BROKEN):
let gc = Arc::new(GraphGarbageCollector::new(storage, config));
let _handle = gc.clone().spawn_worker();  // ← Handle discarded!
*self.gc.write() = Some(gc);
```

**Issue 2: GC Shutdown Doesn't Join**
```rust
// Current (BROKEN):
if let Some(gc) = self.gc.write().take() {
    gc.shutdown();  // Only signals flag, doesn't wait!
}
// Storage may close while GC worker is still running
```

**Issue 3: No Async Updater**
- Vector has `AsyncGraphUpdater` for two-phase inserts
- Graph has no equivalent for deferred index building

### Proposed Graph Subsystem

> (claude, 2026-02-07) Aligned with vector pattern.

```rust
// graph/subsystem.rs (refactored)

pub struct Subsystem {
    // --- Caches (created at construction) ---
    /// Name cache for NameHash ↔ String resolution
    name_cache: Arc<NameCache>,

    // --- Deferred Components (created at start) ---
    /// Processor for synchronous operations
    processor: RwLock<Option<Arc<Processor>>>,

    /// Writer for shutdown flush
    writer: RwLock<Option<Writer>>,

    /// Garbage collector with embedded worker
    gc: RwLock<Option<GraphGarbageCollector>>,

    /// Consumer task handles for shutdown join
    consumer_handles: RwLock<Vec<JoinHandle<Result<()>>>>,

    // --- Configuration ---
    prewarm_config: NameCacheConfig,
}
```

### Initialization Flow

> (claude, 2026-02-07) Step-by-step startup sequence.

```rust
impl Subsystem {
    /// Phase 1: Construction (no I/O)
    pub fn new() -> Self {
        Self {
            name_cache: Arc::new(NameCache::new()),
            processor: RwLock::new(None),
            writer: RwLock::new(None),
            gc: RwLock::new(None),
            consumer_handles: RwLock::new(Vec::new()),
            prewarm_config: NameCacheConfig::default(),
        }
    }

    /// Phase 2: Pre-warm caches from TransactionDB
    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        // Pre-warm name cache from Names CF
        let count = prewarm_cf::<Names, _>(
            db,
            self.prewarm_config.prewarm_limit,
            |key, value| {
                self.name_cache.insert(key.0, value.0.clone());
                Ok(())
            },
        )?;
        tracing::info!(count, "Pre-warmed name cache");
        Ok(())
    }

    /// Phase 3: Start all components
    pub fn start(
        &self,
        storage: Arc<Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
        num_query_workers: usize,
        gc_config: Option<GraphGcConfig>,
    ) -> (Writer, Reader) {
        // 1. Create shared Processor
        let processor = Arc::new(Processor::new_with_cache(
            storage.clone(),
            self.name_cache.clone(),
        ));
        *self.processor.write().unwrap() = Some(processor.clone());

        // 2. Spawn mutation consumer
        let (writer, mutation_handle) = spawn_mutation_consumer_with_processor(
            processor.clone(),
            writer_config,
        );

        // 3. Spawn query consumers
        let (reader, query_handles) = spawn_query_consumers_with_processor(
            processor.clone(),
            num_query_workers,
        );

        // 4. Store handles for shutdown
        {
            let mut handles = self.consumer_handles.write().unwrap();
            handles.push(mutation_handle);
            handles.extend(query_handles);
        }

        // 5. Register writer for shutdown flush
        *self.writer.write().unwrap() = Some(writer.clone());

        // 6. Start GC (returns Self with embedded worker)
        if let Some(config) = gc_config {
            let gc = GraphGarbageCollector::start(storage, config);
            *self.gc.write().unwrap() = Some(gc);
        }

        (writer, reader)
    }
}
```

### Shutdown Flow

> (claude, 2026-02-07) Ordered shutdown matching vector pattern.

```rust
impl RocksdbSubsystem for Subsystem {
    fn on_shutdown(&self) -> Result<()> {
        // Phase 1: Flush pending mutations
        // Closes channel, signals consumers to exit
        if let Some(writer) = self.writer.read().unwrap().as_ref() {
            if !writer.is_closed() {
                tracing::debug!("Flushing pending graph mutations");
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    let _ = handle.block_on(writer.flush());
                }
            }
        }

        // Phase 2: Join consumer tasks
        // Consumers exit when channel closes (recv returns None)
        let handles = std::mem::take(&mut *self.consumer_handles.write().unwrap());
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            for task_handle in handles {
                let _ = handle.block_on(task_handle);
            }
        }

        // Phase 3: Shutdown GC (takes ownership, joins worker)
        // CRITICAL: Must happen AFTER consumers to avoid concurrent access
        if let Some(gc) = self.gc.write().unwrap().take() {
            tracing::debug!("Shutting down graph garbage collector");
            gc.shutdown();  // Blocks until worker completes
        }

        // Phase 4: Drop Processor (releases caches)
        *self.processor.write().unwrap() = None;

        Ok(())
    }
}
```

### GC Worker Fix

> (claude, 2026-02-07) Fix GC to store and join worker handle.

```rust
// graph/gc.rs (refactored)

pub struct GraphGarbageCollector {
    storage: Arc<Storage>,
    config: GraphGcConfig,
    shutdown: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,  // ← Store handle!
}

impl GraphGarbageCollector {
    /// Start GC with embedded worker thread.
    /// Returns Self (not JoinHandle) for lifecycle control.
    pub fn start(storage: Arc<Storage>, config: GraphGcConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn worker thread (not tokio task - blocking work)
        let worker = {
            let storage = storage.clone();
            let shutdown = shutdown.clone();
            let interval = config.check_interval;

            std::thread::spawn(move || {
                while !shutdown.load(Ordering::SeqCst) {
                    if let Err(e) = Self::run_cleanup_cycle(&storage) {
                        tracing::error!(error = %e, "GC cycle failed");
                    }
                    std::thread::sleep(interval);
                }
                tracing::info!("GC worker exiting");
            })
        };

        Self {
            storage,
            config,
            shutdown,
            worker: Some(worker),
        }
    }

    /// Shutdown GC and wait for worker to complete.
    /// Takes ownership to prevent use-after-shutdown.
    pub fn shutdown(mut self) {
        tracing::info!("GC: initiating shutdown");
        self.shutdown.store(true, Ordering::SeqCst);

        if let Some(worker) = self.worker.take() {
            if let Err(e) = worker.join() {
                tracing::error!("GC worker panicked: {:?}", e);
            }
        }
        tracing::info!("GC: shutdown complete");
    }
}
```

### Cache Lifecycle

> (claude, 2026-02-07) How caches are created, shared, and cleaned up.

```
INITIALIZATION:
├─ Subsystem::new()
│  └─ name_cache = Arc::new(NameCache::new())     // Empty cache
│
├─ on_ready(db)
│  └─ Pre-warm name_cache from Names CF           // Fill from DB
│
├─ start(storage, ...)
│  └─ Processor::new_with_cache(storage, name_cache.clone())
│     └─ Processor holds Arc<NameCache>           // Shared reference

OPERATIONAL:
├─ Mutation Consumer
│  └─ processor.process_mutations()
│     └─ mutation.execute_with_cache(&txn, txn_db, &name_cache)
│        └─ write_name_to_cf_cached()             // Updates cache
│
├─ Query Consumer
│  └─ query.execute(storage, processor)
│     └─ resolve_name_from_cache()                // Reads cache

SHUTDOWN:
├─ on_shutdown()
│  ├─ Flush writer                                // Final mutations
│  ├─ Join consumers                              // Stop cache writes
│  ├─ Shutdown GC                                 // Stop GC reads
│  └─ Drop Processor                              // Release Arc
│
└─ Subsystem dropped
   └─ name_cache Arc refcount → 0                 // Cache freed
```

### TransactionDB Access Pattern

> (claude, 2026-02-07) How components access RocksDB.

```rust
// All DB access goes through Processor → Storage → TransactionDB

impl Processor {
    /// Get TransactionDB for mutations
    pub fn transaction_db(&self) -> Result<&TransactionDB> {
        self.storage.transaction_db()
    }

    /// Execute mutations in a transaction
    pub fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        let txn_db = self.transaction_db()?;
        let txn = txn_db.transaction();

        for mutation in mutations {
            mutation.execute_with_cache(&txn, txn_db, &self.name_cache)?;
        }

        txn.commit()?;
        Ok(())
    }
}

// Query consumers also access via Processor
impl QueryExecutor for NodeByIdDispatch {
    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Can use storage.db() for readonly
        // Or storage.transaction_db() for read-write
        // Processor provides cache access for name resolution
    }
}
```

### Lifecycle Timeline

> (claude, 2026-02-07) Complete lifecycle from start to shutdown.

```
PHASE 1: CONSTRUCTION (no I/O)
├─ Subsystem::new()
│  ├─ Create empty NameCache
│  ├─ Initialize RwLock<Option<T>> fields to None
│  └─ Return Subsystem

PHASE 2: STORAGE OPEN (RocksDB I/O)
├─ StorageBuilder::build()
│  ├─ Open TransactionDB with column families
│  ├─ Call subsystem.on_ready(db)
│  │  └─ Pre-warm NameCache from Names CF
│  └─ Return SharedStorage

PHASE 3: COMPONENT START (spawn tasks)
├─ subsystem.start(storage, configs)
│  ├─ Create Processor with storage + name_cache
│  ├─ Spawn mutation consumer (holds Arc<Processor>)
│  ├─ Spawn query consumers (hold Arc<Processor>)
│  ├─ Store JoinHandles for shutdown
│  ├─ Register Writer for flush
│  ├─ Start GC with embedded worker thread
│  └─ Return (Writer, Reader)

PHASE 4: OPERATIONAL
├─ User sends mutations via Writer
│  ├─ Consumer receives from channel
│  ├─ Processor.process_mutations() in transaction
│  └─ Cache updated after commit
├─ User sends queries via Reader
│  ├─ Consumer receives from channel
│  ├─ QueryExecutor uses Processor for cache
│  └─ Results sent via oneshot channel
└─ GC runs periodic cleanup cycles

PHASE 5: SHUTDOWN (ordered teardown)
├─ subsystem.on_shutdown()
│  ├─ [1] Writer.flush() → closes channel
│  ├─ [2] Join consumer handles → wait for drain
│  ├─ [3] GC.shutdown() → signal + join worker
│  └─ [4] Drop Processor → release cache Arc
└─ Subsystem dropped → NameCache freed

INVARIANTS:
- Consumers exit when channel closes (recv returns None)
- GC shutdown blocks until worker completes
- All JoinHandles awaited before storage closes
- No dangling references to TransactionDB
```

### Migration Checklist (Updated)

> (claude, 2026-02-07) Additional lifecycle tasks.

- [ ] **Task 7:** Fix GC to store worker handle (not discard)
- [ ] **Task 8:** Fix GC shutdown to join worker thread
- [ ] **Task 9:** Change GC from tokio task to std::thread (blocking work)
- [ ] **Task 10:** Add on_ready() pre-warm for NameCache
- [ ] **Task 11:** Update Subsystem.start() to create shared Processor
- [ ] **Task 12:** Verify shutdown ordering: flush → consumers → GC
