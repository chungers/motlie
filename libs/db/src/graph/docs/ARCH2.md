# ARCH2: Align Graph Architecture with Vector Processor Pattern

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

## Recent Implementation Changes (as of 2026-02-03)

- **Subsystem-managed lifecycle**: `Subsystem::start(...)` now wires Writer/Reader, spawns consumers, optionally starts GC, and manages shutdown ordering (flush → join consumers → stop GC). `graph/subsystem.rs`
- **GC and RefCount**: GC is integrated for stale index cleanup; summaries are content-addressed with RefCount deleted inline by mutations, removing orphan summary scans. `graph/gc.rs`, `graph/mutation.rs`, `graph/schema.rs`
- **Public APIs unchanged**: Writer/Reader are still the public async channel APIs; `Graph` remains the processor used by consumers.

These changes improve robustness and lifecycle management but **do not yet introduce** an internal `graph::processor::Processor` or hide `Graph` behind the public APIs.

---

## Proposed Refactor (Match Vector Pattern)

### 1) Introduce `graph::processor::Processor` (pub(crate))

Responsibilities:

- Owns `Arc<Storage>`
- Owns shared caches (e.g., name cache, future graph caches)
- Exposes synchronous methods for core graph operations

### 2) Update graph mutation path

- `graph::writer::Consumer` holds `Arc<graph::processor::Processor>`
- Add helper functions mirroring vector:
  - `spawn_mutation_consumer_with_storage(...)`
  - `spawn_mutation_consumer_with_processor(...)` (pub(crate))

### 3) Update graph query path

- Add a Processor-backed query consumer (like `vector::ProcessorConsumer`)
- Provide reader construction helpers:
  - `create_reader_with_storage(...)`
  - `spawn_query_consumers_with_storage(...)`

### 4) Keep `Graph` as facade (optional)

- `Graph` can wrap an `Arc<Processor>` or `Arc<Storage>`
- Public API can remain stable while internal implementation migrates
- **Desired outcome:** the refactor should make it possible to remove `Graph` entirely over time, leaving `Processor` + public async `Writer`/`Reader` as the primary entry points

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
