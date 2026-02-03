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

---

## Benefits

- **Encapsulation:** Internal sync logic is isolated
- **API Stability:** Public async APIs remain unchanged
- **Consistency:** Aligns graph with vector/fulltext patterns
- **Future-proof:** Easier to add caches or new execution modes

---

## Summary

Vector’s architecture clearly separates synchronous internal execution
(`Processor`) from async public APIs (`Writer`/`Reader`). Refactoring graph
to follow the same pattern will make the system more consistent, easier to
extend, and safer to evolve without breaking public interfaces.
