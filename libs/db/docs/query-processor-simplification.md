# Query Processor Simplification: Trait-Based Design

**Status**: 📋 Proposed Design
**Date**: 2025-11-14
**Context**: Analysis comparing mutation and query processor patterns

## Executive Summary

This document proposes a significant simplification of the query processor architecture by moving query execution logic from the `Processor` trait implementation into individual query types via a `QueryExecutor` trait. This brings the query system into alignment with the mutation system's design philosophy.

**Current State**:
- `query::Processor` trait: 8 methods (one per query type)
- `Graph` implementation: ~300 lines across 8 methods
- Pattern: Trait defines interface, Graph contains all fetch logic

**Proposed State**:
- `query::Processor` trait: 1 method (`storage()` accessor)
- `Graph` implementation: ~3 lines
- Pattern: Trait provides storage access, query types contain fetch logic

**Net Impact**: ~90% reduction in Processor trait complexity, better alignment with mutation pattern, improved maintainability.

---

## Table of Contents

1. [Background: Mutation vs Query Pattern Analysis](#background)
2. [Current Query Design](#current-design)
3. [Proposed Trait-Based Design](#proposed-design)
4. [Implementation Details](#implementation-details)
5. [Migration Strategy](#migration-strategy)
6. [Benefits and Trade-offs](#benefits-and-trade-offs)
7. [Code Examples](#code-examples)

---

## Background: Mutation vs Query Pattern Analysis {#background}

### Mutation Pattern (Current Implementation)

The mutation system uses a clean, minimal design:

```rust
// mutation.rs - Trait definition
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;
}

// mutation.rs - Enum with 4 variants
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddFragment(AddFragment),
    Invalidate(InvalidateArgs),
}

// graph.rs - Implementation (simple delegation)
#[async_trait::async_trait]
impl mutation::Processor for Graph {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        // Convert mutations to storage operations
        let operations = schema::Plan::create_batch(mutations)?;

        // Execute in single transaction
        // ... transaction logic
    }
}
```

**Key characteristics**:
- ✅ **Single trait method**: One entry point for all mutation types
- ✅ **Enum dispatch**: Mutations are enum variants
- ✅ **Logic separation**: `schema::Plan` contains conversion logic
- ✅ **Simple implementation**: Graph just orchestrates, doesn't contain business logic

### Query Pattern (Current Implementation)

The query system uses a more complex design:

```rust
// query.rs - Trait definition
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    async fn get_node_by_id(&self, query: &NodeByIdQuery)
        -> Result<(NodeName, NodeSummary)>;

    async fn get_edge_by_id(&self, query: &EdgeByIdQuery)
        -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>;

    async fn get_edge_summary_by_src_dst_name(&self, query: &EdgeSummaryBySrcDstNameQuery)
        -> Result<(Id, EdgeSummary)>;

    async fn get_fragments_by_id_time_range(&self, query: &FragmentsByIdTimeRangeQuery)
        -> Result<Vec<(TimestampMilli, FragmentContent)>>;

    async fn get_edges_from_node_by_id(&self, query: &EdgesFromNodeQuery)
        -> Result<Vec<(SrcId, EdgeName, DstId)>>;

    async fn get_edges_to_node_by_id(&self, query: &EdgesToNodeQuery)
        -> Result<Vec<(DstId, EdgeName, SrcId)>>;

    async fn get_nodes_by_name(&self, query: &NodesByNameQuery)
        -> Result<Vec<(NodeName, Id)>>;

    async fn get_edges_by_name(&self, query: &EdgesByNameQuery)
        -> Result<Vec<(EdgeName, Id)>>;
}

// query.rs - Enum with 8 variants
pub enum Query {
    NodeById(NodeByIdQuery),
    EdgeById(EdgeByIdQuery),
    // ... 6 more variants
}

// graph.rs - Implementation (~300 lines)
#[async_trait::async_trait]
impl query::Processor for Graph {
    async fn get_node_by_id(&self, query: &NodeByIdQuery)
        -> Result<(NodeName, NodeSummary)> {
        // 30+ lines of RocksDB fetch logic
        let key = schema::NodeCfKey(query.id);
        let key_bytes = schema::Nodes::key_to_bytes(&key);

        let value_bytes = if let Ok(db) = self.storage.db() {
            // ... fetch logic
        } else {
            // ... transaction mode logic
        };

        // ... deserialization and return
    }

    // ... 7 more methods with similar patterns
}
```

**Key characteristics**:
- ❌ **Eight trait methods**: One per query type
- ✅ **Enum dispatch**: Queries are enum variants (like mutations)
- ❌ **Logic in implementation**: Graph contains all fetch logic (~300 lines)
- ❌ **Complex implementation**: Implementers must provide 8 methods

### The Asymmetry Problem

| Aspect | Mutation | Query | Issue |
|--------|----------|-------|-------|
| Trait methods | 1 | 8 | High implementer burden |
| Implementation lines | ~30 | ~300 | Code scattered across methods |
| Logic location | `schema::Plan` | `Graph` impl | Inconsistent |
| Extensibility | Add to enum + Plan | Add method + impl | Breaking change |

**Core Issue**: Queries place business logic in the trait implementation, while mutations place it in helper types. This makes queries harder to implement and extend.

---

## Current Query Design {#current-design}

### Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│ Reader (Client)                                           │
│ - node_by_id(), edge_by_id(), etc.                       │
│ - Creates query structs with oneshot channels            │
└────────────────────┬─────────────────────────────────────┘
                     │ Sends Query enum via flume channel
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Consumer (Query dispatcher)                               │
│ - Receives Query enum                                     │
│ - Calls query.process_and_send(processor)                │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Query Enum Dispatch                                       │
│ match self {                                              │
│   Query::NodeById(q) => q.process_and_send(processor),   │
│   Query::EdgeById(q) => q.process_and_send(processor),   │
│   // ... 6 more variants                                 │
│ }                                                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ NodeByIdQuery::process_and_send()                        │
│ - Calls processor.get_node_by_id(self)                   │
│ - Sends result back via oneshot channel                  │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Processor Trait (8 methods)                              │
│ trait Processor {                                         │
│   async fn get_node_by_id(...) -> Result<...>;          │
│   async fn get_edge_by_id(...) -> Result<...>;          │
│   // ... 6 more methods                                  │
│ }                                                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Graph Implementation (~300 lines)                         │
│ impl Processor for Graph {                               │
│   async fn get_node_by_id(&self, query: &NodeByIdQuery) │
│       -> Result<(NodeName, NodeSummary)> {               │
│     // 30+ lines of RocksDB access code                  │
│     let key = schema::NodeCfKey(query.id);               │
│     let value_bytes = self.storage.db()                  │
│       .get_cf(cf, key_bytes)?;                           │
│     // ... deserialization                               │
│   }                                                       │
│   // ... 7 more methods                                  │
│ }                                                         │
└──────────────────────────────────────────────────────────┘
```

### Pattern Matching Locations

1. **Query enum dispatch** (query.rs:96-105)
   - Purpose: Route to correct query struct
   - Pattern: `match Query { NodeById(q) => q.process_and_send(), ... }`

2. **Processor trait boundary** (8 separate methods)
   - Purpose: Define interface for each query type
   - Pattern: One method per query variant

3. **Graph implementation** (8 method bodies)
   - Purpose: Execute storage operations
   - Pattern: Fetch logic embedded in each method

### Problems with Current Design

1. **High Implementation Burden**
   - New storage backends must implement 8 methods
   - Each method contains similar RocksDB boilerplate
   - Example: fulltext indexer must stub out all 8 methods

2. **Scattered Logic**
   - Related code split across 8 method bodies
   - Hard to see common patterns
   - Difficult to refactor shared logic

3. **Extensibility Issues**
   - Adding new query type = breaking change to trait
   - All implementers must update
   - Cannot add queries in external crates

4. **Inconsistent with Mutations**
   - Mutations: 1 trait method, logic in `schema::Plan`
   - Queries: 8 trait methods, logic in `Graph`
   - No clear design principle

---

## Proposed Trait-Based Design {#proposed-design}

### Core Concept

**Move query execution logic from `Processor` trait implementation to individual query types via a `QueryExecutor` trait.**

This mirrors how mutations work:
- **Mutations**: `schema::Plan::create_batch()` knows how to convert mutations
- **Queries**: Each query type knows how to execute itself

### New Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Reader (Client) - UNCHANGED                               │
│ - node_by_id(), edge_by_id(), etc.                       │
│ - Creates query structs with oneshot channels            │
└────────────────────┬─────────────────────────────────────┘
                     │ Sends Query enum
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Consumer - UNCHANGED                                      │
│ - Receives Query enum                                     │
│ - Calls query.process_and_send(processor)                │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Query Enum Dispatch - UNCHANGED                           │
│ match self {                                              │
│   Query::NodeById(q) => q.process_and_send(processor),   │
│   // ...                                                  │
│ }                                                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ NodeByIdQuery::process_and_send() - MODIFIED             │
│ - Calls self.execute(processor.storage()) ← NEW          │
│ - Sends result back via oneshot channel                  │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ QueryExecutor Trait - NEW                                 │
│ trait QueryExecutor {                                     │
│   type Output;                                            │
│   async fn execute(&self, storage: &Storage)             │
│       -> Result<Self::Output>;                           │
│ }                                                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ NodeByIdQuery Implementation - NEW LOCATION              │
│ impl QueryExecutor for NodeByIdQuery {                   │
│   type Output = (NodeName, NodeSummary);                 │
│   async fn execute(&self, storage: &Storage)             │
│       -> Result<Self::Output> {                          │
│     // 30+ lines of RocksDB access (MOVED HERE)          │
│     let key = schema::NodeCfKey(self.id);                │
│     let value_bytes = storage.db()                       │
│       .get_cf(cf, key_bytes)?;                           │
│     // ... deserialization                               │
│   }                                                       │
│ }                                                         │
│ // ... 7 more QueryExecutor implementations              │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Processor Trait - SIMPLIFIED                              │
│ trait Processor {                                         │
│   fn storage(&self) -> &Storage;  ← SINGLE METHOD        │
│ }                                                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Graph Implementation - TRIVIAL                            │
│ impl Processor for Graph {                               │
│   fn storage(&self) -> &Storage {                        │
│     &self.storage                                        │
│   }                                                       │
│ }                                                         │
└──────────────────────────────────────────────────────────┘
```

### Key Changes

1. **New `QueryExecutor` Trait**
   ```rust
   #[async_trait::async_trait]
   pub trait QueryExecutor: Send + Sync {
       type Output: Send;
       async fn execute(&self, storage: &Storage) -> Result<Self::Output>;
   }
   ```

2. **Simplified `Processor` Trait**
   ```rust
   pub trait Processor: Send + Sync {
       fn storage(&self) -> &Storage;
   }
   ```

3. **Query Types Implement `QueryExecutor`**
   - Each query struct implements `execute(&self, storage: &Storage)`
   - Fetch logic moves from `Graph` to query types

4. **Trivial `Graph` Implementation**
   ```rust
   impl Processor for Graph {
       fn storage(&self) -> &Storage {
           &self.storage
       }
   }
   ```

### Pattern Comparison

**Before** (8-method trait):
```
Query → Processor trait (8 methods) → Graph impl (8 methods with logic)
```

**After** (trait-based):
```
Query → QueryExecutor trait → Self::execute() → Storage
                                    ↓
Processor trait (1 method) → Graph impl (1 line) → Storage reference
```

---

## Implementation Details {#implementation-details}

### 1. QueryExecutor Trait

```rust
/// Trait that all query types implement to execute themselves
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// The type of result this query produces
    type Output: Send;

    /// Execute this query against the storage layer
    /// Each query type knows how to fetch its own data
    async fn execute(&self, storage: &Storage) -> Result<Self::Output>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}
```

### 2. Processor Trait Simplification

```rust
/// Minimal processor trait - just provides storage access
pub trait Processor: Send + Sync {
    /// Get access to the underlying storage
    fn storage(&self) -> &Storage;
}
```

### 3. QueryExecutor Implementations

Each query struct implements `QueryExecutor`:

```rust
// NodeByIdQuery
#[async_trait::async_trait]
impl QueryExecutor for NodeByIdQuery {
    type Output = (NodeName, NodeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let id = self.id;
        let key = schema::NodeCfKey(id);
        let key_bytes = schema::Nodes::key_to_bytes(&key);

        // Handle both readonly and readwrite modes
        let value_bytes = if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::Nodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::Nodes::CF_NAME
                ))?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Nodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::Nodes::CF_NAME
                ))?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", id))?;

        let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        Ok((value.0, value.1))
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

// EdgeByIdQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgeByIdQuery {
    type Output = (SrcId, DstId, EdgeName, EdgeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Implementation from current graph.rs:411-440
        // ...
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

// ... same pattern for all 8 query types
```

### 4. QueryWithTimeout Integration

```rust
#[async_trait::async_trait]
impl<T: QueryExecutor> QueryWithTimeout for T {
    type ResultType = T::Output;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Self::ResultType> {
        let result = tokio::time::timeout(
            self.timeout(),
            self.execute(processor.storage())
        ).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout())),
        }
    }

    fn timeout(&self) -> Duration {
        self.timeout()
    }
}
```

### 5. Graph Implementation

```rust
impl Processor for Graph {
    fn storage(&self) -> &Storage {
        &self.storage
    }
}
```

That's it! From ~300 lines down to 3 lines.

### 6. Query Enum (Unchanged)

The `Query` enum and its dispatch logic remain unchanged:

```rust
#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send<P: Processor>(self, processor: &P) {
        match self {
            Query::NodeById(q) => q.process_and_send(processor).await,
            Query::EdgeById(q) => q.process_and_send(processor).await,
            Query::EdgeSummaryBySrcDstName(q) => q.process_and_send(processor).await,
            Query::FragmentsByIdTimeRange(q) => q.process_and_send(processor).await,
            Query::EdgesFromNode(q) => q.process_and_send(processor).await,
            Query::EdgesToNode(q) => q.process_and_send(processor).await,
            Query::NodesByName(q) => q.process_and_send(processor).await,
            Query::EdgesByName(q) => q.process_and_send(processor).await,
        }
    }
}
```

---

## Migration Strategy {#migration-strategy}

### Phase 1: Add QueryExecutor Trait (Additive)

**Goal**: Introduce new trait without breaking existing code

1. **Add `QueryExecutor` trait to query.rs**
   ```rust
   #[async_trait::async_trait]
   pub trait QueryExecutor: Send + Sync {
       type Output: Send;
       async fn execute(&self, storage: &Storage) -> Result<Self::Output>;
       fn timeout(&self) -> Duration;
   }
   ```

2. **Implement `QueryExecutor` for each query type**
   - Move logic from `graph.rs` → query struct implementations
   - Keep original Processor methods intact (dual implementation)

3. **Add `storage()` method to `Processor` trait**
   ```rust
   pub trait Processor: Send + Sync {
       fn storage(&self) -> &Storage;

       // ... existing 8 methods remain
   }
   ```

4. **Update `Graph` to provide storage access**
   ```rust
   impl Processor for Graph {
       fn storage(&self) -> &Storage {
           &self.storage
       }

       // ... existing 8 method impls remain
   }
   ```

**Status**: ✅ No breaking changes, fully backward compatible

### Phase 2: Update QueryWithTimeout (Internal)

**Goal**: Switch to using `QueryExecutor::execute()`

1. **Modify `QueryWithTimeout::result` implementation**
   - Change from calling `processor.get_node_by_id()`
   - To calling `self.execute(processor.storage())`

2. **Test extensively**
   - Verify all query types work correctly
   - Check timeout behavior
   - Validate error handling

**Status**: ✅ No API changes, internal refactoring only

### Phase 3: Deprecate Old Methods (Gradual)

**Goal**: Signal intent to remove old Processor methods

1. **Add deprecation warnings to trait methods**
   ```rust
   pub trait Processor: Send + Sync {
       fn storage(&self) -> &Storage;

       #[deprecated(
           since = "0.x.0",
           note = "Use QueryExecutor trait instead. This method will be removed in 0.y.0"
       )]
       async fn get_node_by_id(...) -> Result<...>;

       // ... deprecate all 8 methods
   }
   ```

2. **Update documentation**
   - Explain migration path
   - Provide examples

3. **Wait one or more versions**
   - Allow downstream users to migrate
   - Gather feedback

### Phase 4: Remove Old Methods (Breaking)

**Goal**: Clean up deprecated code

1. **Remove 8 query-specific methods from `Processor` trait**
2. **Remove implementations from `Graph`**
3. **Update version number** (major bump if semver)

**Result**: Clean, simple trait with single `storage()` method

### Rollback Strategy

At any phase, can rollback by:
- Phase 2: Revert QueryWithTimeout changes
- Phase 1: Remove QueryExecutor implementations, keep Processor methods

No data migration required - this is purely code organization.

---

## Benefits and Trade-offs {#benefits-and-trade-offs}

### Benefits

#### 1. Alignment with Mutation Pattern

**Mutations**:
```rust
trait Processor {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;
}
// Logic in: schema::Plan::create_batch()
```

**Queries** (proposed):
```rust
trait Processor {
    fn storage(&self) -> &Storage;
}
// Logic in: impl QueryExecutor for each query type
```

✅ **Consistent philosophy**: Logic lives with types, not in central implementation

#### 2. Reduced Implementation Burden

**Before**:
```rust
impl Processor for MyStorage {
    async fn get_node_by_id(...) -> Result<...> { /* 30 lines */ }
    async fn get_edge_by_id(...) -> Result<...> { /* 30 lines */ }
    // ... 6 more methods
}
```

**After**:
```rust
impl Processor for MyStorage {
    fn storage(&self) -> &Storage { &self.storage }
}
```

✅ **90% less code** to implement Processor trait

#### 3. Better Code Organization

**Before**: Fetch logic scattered across 8 methods in graph.rs

**After**: Each query type is self-contained
- `NodeByIdQuery` struct + fields + `QueryExecutor` impl in one place
- Easy to find and understand
- Related code co-located

✅ **Improved locality of reference**

#### 4. Easier Testing

**Before**: Mock entire Processor trait (8 methods)
```rust
#[async_trait::async_trait]
impl Processor for MockProcessor {
    async fn get_node_by_id(...) { /* mock */ }
    async fn get_edge_by_id(...) { /* mock */ }
    // ... 6 more mocks
}
```

**After**: Mock storage or test queries directly
```rust
// Test query execution in isolation
let query = NodeByIdQuery::new(...);
let result = query.execute(&mock_storage).await?;
assert_eq!(result, expected);
```

✅ **Simpler test setup, more focused tests**

#### 5. Non-Breaking Extension

**Before**: Adding new query type = modify Processor trait
- All implementers must update
- Breaking change

**After**: Adding new query type
- Create struct + impl QueryExecutor
- Add to Query enum
- No Processor trait changes

✅ **Open/closed principle** - open for extension, closed for modification

#### 6. Storage Abstraction Flexibility

**Current**: Processor trait tightly couples to query types

**Proposed**: Processor just provides storage access
- Can swap storage implementations easily
- Can add caching layer transparently
- Can add observability without touching queries

✅ **Better separation of concerns**

### Trade-offs

#### 1. More Distributed Code

**Before**: All fetch logic in graph.rs (one file)

**After**: Fetch logic in query.rs (8 impl blocks)

⚠️ **Consideration**: Some prefer centralized logic

**Mitigation**:
- Each impl block is self-contained
- Easy to navigate with IDE (go to implementation)
- Better than 300-line file with 8 similar methods

#### 2. Less Granular Trait Bounds

**Before**: Can implement just `get_node_by_id()` in isolation

**After**: Must implement `storage()` which provides access to all data

⚠️ **Consideration**: Can't restrict to subset of queries via trait bounds

**Mitigation**:
- In practice, storage implementations support all queries
- Partial implementations were never needed
- Can add feature flags if needed: `#[cfg(feature = "node-queries")]`

#### 3. Slightly More Complex Query Types

**Before**: Query structs are simple data containers

**After**: Query structs also have execution logic

⚠️ **Consideration**: Mixing data and behavior

**Mitigation**:
- This is standard OOP practice (objects = data + methods)
- Actually improves cohesion
- Similar to how mutations work with `schema::Plan`

#### 4. Async Trait Overhead

**Before**: 8 async trait methods

**After**: Still 8 async trait methods (just different location)

⚠️ **Consideration**: No change in async trait overhead

**Mitigation**: Not a regression, just different organization

### Net Assessment

| Metric | Score | Rationale |
|--------|-------|-----------|
| **Code simplicity** | ✅✅✅ | 90% reduction in Processor impl |
| **Maintainability** | ✅✅ | Better organization, easier to understand |
| **Extensibility** | ✅✅✅ | Non-breaking query additions |
| **Testability** | ✅✅ | Simpler mocks, isolated testing |
| **Consistency** | ✅✅✅ | Aligns with mutation pattern |
| **Migration cost** | ⚠️ | Moderate effort, but non-breaking |

**Overall**: Strong positive impact, worth the migration effort.

---

## Code Examples {#code-examples}

### Example 1: Implementing a New Query Type

**Before** (8-method trait):

```rust
// 1. Add to Query enum
pub enum Query {
    // ... existing variants
    NodesByTag(NodesByTagQuery),  // NEW
}

// 2. Define query struct
pub struct NodesByTagQuery {
    pub tag: String,
    pub timeout: Duration,
    result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
}

// 3. Modify Processor trait (BREAKING CHANGE!)
pub trait Processor: Send + Sync {
    // ... existing 8 methods
    async fn get_nodes_by_tag(&self, query: &NodesByTagQuery)
        -> Result<Vec<(NodeName, Id)>>;  // NEW METHOD
}

// 4. Implement in Graph (and ALL other implementations!)
impl Processor for Graph {
    // ... existing 8 methods
    async fn get_nodes_by_tag(&self, query: &NodesByTagQuery)
        -> Result<Vec<(NodeName, Id)>> {
        // 40+ lines of fetch logic
        // ...
    }
}

// 5. Update every other Processor implementation
impl Processor for FulltextIndex {
    // ... existing 8 methods
    async fn get_nodes_by_tag(&self, query: &NodesByTagQuery)
        -> Result<Vec<(NodeName, Id)>> {
        unimplemented!("Fulltext doesn't support tag queries")
    }
}
```

**After** (trait-based):

```rust
// 1. Add to Query enum
pub enum Query {
    // ... existing variants
    NodesByTag(NodesByTagQuery),  // NEW
}

// 2. Define query struct
pub struct NodesByTagQuery {
    pub tag: String,
    pub timeout: Duration,
    result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
}

// 3. Implement QueryExecutor
#[async_trait::async_trait]
impl QueryExecutor for NodesByTagQuery {
    type Output = Vec<(NodeName, Id)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // 40+ lines of fetch logic
        // ...
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

// 4. Add QueryWithTimeout (via macro)
impl_query_processor!(NodesByTagQuery);

// DONE! No changes to Processor trait or Graph implementation needed!
```

✅ **Result**: 4 steps → 3 steps, no breaking changes, cleaner separation

### Example 2: Testing Query Execution

**Before**:

```rust
#[tokio::test]
async fn test_node_by_id() {
    // Must mock entire Processor trait
    struct MockProcessor;

    #[async_trait::async_trait]
    impl Processor for MockProcessor {
        async fn get_node_by_id(&self, query: &NodeByIdQuery)
            -> Result<(NodeName, NodeSummary)> {
            Ok((
                "test_node".to_string(),
                NodeSummary::new("summary".to_string()),
            ))
        }

        // Must stub 7 other methods
        async fn get_edge_by_id(&self, _: &EdgeByIdQuery)
            -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
            unimplemented!()
        }
        // ... 6 more stubs
    }

    let processor = MockProcessor;
    let query = NodeByIdQuery::new(...);
    let result = query.result(&processor).await?;
    assert_eq!(result.0, "test_node");
}
```

**After**:

```rust
#[tokio::test]
async fn test_node_by_id() {
    // Test query execution directly
    let storage = create_test_storage().await;

    // Insert test data
    let node_id = Id::new();
    insert_test_node(&storage, node_id, "test_node", "summary").await;

    // Execute query
    let query = NodeByIdQuery {
        id: node_id,
        timeout: Duration::from_secs(1),
        result_tx: /* ... */,
    };

    let result = query.execute(&storage).await?;
    assert_eq!(result.0, "test_node");
}
```

✅ **Result**: More focused test, easier to set up, tests actual storage logic

### Example 3: Implementing Processor for New Backend

**Before** (8 methods required):

```rust
struct RedisStorage {
    client: redis::Client,
}

#[async_trait::async_trait]
impl Processor for RedisStorage {
    async fn get_node_by_id(&self, query: &NodeByIdQuery)
        -> Result<(NodeName, NodeSummary)> {
        // Redis-specific fetch logic
    }

    async fn get_edge_by_id(&self, query: &EdgeByIdQuery)
        -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
        // Redis-specific fetch logic
    }

    async fn get_edge_summary_by_src_dst_name(&self, query: &EdgeSummaryBySrcDstNameQuery)
        -> Result<(Id, EdgeSummary)> {
        // Redis-specific fetch logic
    }

    // ... 5 more methods with Redis logic
}
```

**After** (1 method required):

```rust
struct RedisStorage {
    client: redis::Client,
}

impl Processor for RedisStorage {
    fn storage(&self) -> &Storage {
        &self.storage  // Wrap Redis in Storage abstraction
    }
}

// If Storage doesn't support Redis, just create an adapter:
struct RedisToStorageAdapter {
    redis: RedisStorage,
}

impl Storage for RedisToStorageAdapter {
    fn db(&self) -> Result<&DB> {
        // Implement RocksDB-compatible interface over Redis
    }

    fn transaction_db(&self) -> Result<&TransactionDB> {
        // Implement transaction interface
    }
}
```

✅ **Result**: Minimal implementation, focus on storage abstraction not query logic

### Example 4: Adding Caching Layer

**Before**: Must intercept all 8 Processor methods

```rust
struct CachedProcessor<P: Processor> {
    inner: P,
    cache: Arc<Mutex<HashMap<Id, CachedValue>>>,
}

#[async_trait::async_trait]
impl<P: Processor> Processor for CachedProcessor<P> {
    async fn get_node_by_id(&self, query: &NodeByIdQuery)
        -> Result<(NodeName, NodeSummary)> {
        if let Some(cached) = self.cache.lock().unwrap().get(&query.id) {
            return Ok(cached.clone());
        }
        let result = self.inner.get_node_by_id(query).await?;
        self.cache.lock().unwrap().insert(query.id, result.clone());
        Ok(result)
    }

    // ... repeat caching logic for 7 other methods
}
```

**After**: Cache at storage layer

```rust
struct CachedStorage {
    inner: Storage,
    cache: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl Storage for CachedStorage {
    fn db(&self) -> Result<&DB> {
        // Return cached DB wrapper
    }

    fn get_cf(&self, cf: &ColumnFamily, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(cached) = self.cache.lock().unwrap().get(key) {
            return Ok(Some(cached.clone()));
        }
        let result = self.inner.db()?.get_cf(cf, key)?;
        if let Some(value) = &result {
            self.cache.lock().unwrap().insert(key.to_vec(), value.clone());
        }
        Ok(result)
    }
}

impl Processor for Graph {
    fn storage(&self) -> &Storage {
        &self.storage  // Returns CachedStorage
    }
}
```

✅ **Result**: Single interception point, works for all queries automatically

---

## Conclusion

The proposed trait-based design brings query processing into alignment with the mutation pattern:

- **Minimal trait surface**: 1 method instead of 8
- **Logic with types**: Query execution lives in query types, not central implementation
- **Easy to extend**: Add queries without breaking changes
- **Better testing**: Mock storage, test queries in isolation
- **Simpler implementations**: 90% reduction in boilerplate

**Recommendation**: Proceed with phased migration as outlined in [Migration Strategy](#migration-strategy).

---

## Related Documents

- [reader.md](reader.md) - Reader API documentation
- [variable-length-fields-in-keys.md](variable-length-fields-in-keys.md) - Key design principles
- [../src/query.rs](../src/query.rs) - Current query implementation
- [../src/mutation.rs](../src/mutation.rs) - Mutation implementation for comparison
