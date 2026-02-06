# Mutation API Guide

**Status**: ✅ Current (as of 2025-11-19)

This guide covers the modern Mutation API introduced in v0.3.0, which provides a clean, type-driven interface for mutating the motlie graph database. The Mutation API follows the same design pattern as the Query API, ensuring consistency across the codebase.

## Table of Contents

- [Overview](#overview)
- [Design Rationale](#design-rationale)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Mutation Types](#mutation-types)
- [Common Patterns](#common-patterns)
- [Advanced Usage](#advanced-usage)
- [Migration Guide](#migration-guide)

## Overview

The Mutation API follows the same idiomatic Rust pattern as the Query API:

```rust
MutationType { ...fields }
    .run(&writer)
    .await?
```

### Key Features

1. **Type-Driven** - Mutation type determines behavior
2. **Consistent with Query API** - Same `.run()` pattern
3. **Zero-Cost Batching** - Use `mutations![]` macro or `MutationBatch`
4. **No Builder Boilerplate** - Direct mutation construction
5. **Async-First** - Built on `tokio` for concurrent mutations
6. **Composable** - Mutations are values that can be stored and reused

## Design Rationale

### Why Refactor?

The original mutation API used helper methods on the Writer:

```rust
// Old API
writer.add_node(AddNode { /* ... */ }).await?;
writer.add_edge(AddEdge { /* ... */ }).await?;
```

While functional, this approach had several drawbacks:

1. **Inconsistent with Query API** - Queries use `.run()`, mutations used helper methods
2. **Verbose API Surface** - One helper method per mutation type
3. **Less Composable** - Mutations were tightly coupled to the Writer
4. **Harder to Extend** - Adding new mutation types required new Writer methods

### Explored Options

We evaluated several approaches:

#### Option 1: async_trait for Vec<Mutation>

**Approach**: Implement `Runnable` directly on `Vec<Mutation>` using the `async_trait` crate.

```rust
#[async_trait::async_trait]
impl Runnable for Vec<Mutation> {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(self).await
    }
}
```

**Pros**:
- Familiar `vec![]` syntax
- Direct implementation on standard type

**Cons**:
- Requires `async_trait` dependency (though already present in codebase)
- Heap allocation overhead - every `.run()` call boxes the future (~50-100μs per 1,000 mutations)
- Loss of zero-cost abstraction - Rust async/await is normally stack-allocated

#### Option 2: MutationBatch Wrapper

**Approach**: Create a newtype wrapper `MutationBatch(Vec<Mutation>)` with native async methods.

```rust
pub struct MutationBatch(pub Vec<Mutation>);

impl Runnable for MutationBatch {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(self.0).await
    }
}
```

**Pros**:
- **Zero allocation overhead** - No boxing required
- **Better type safety** - Distinct type signals "this is a batch"
- **Extensible** - Can add validation, size limits, etc.
- **No external dependencies** - No need for `async_trait`
- **Future-proof** - Ready for when Rust stabilizes async in traits

**Cons**:
- Requires wrapper type instead of using `Vec` directly
- Slightly more verbose than `vec![]`

#### Option 3: mutations![] Macro

**Approach**: Provide a `vec![]`-like macro that returns `MutationBatch`.

```rust
#[macro_export]
macro_rules! mutations {
    () => { $crate::MutationBatch::new() };
    ($($mutation:expr),+ $(,)?) => {
        $crate::MutationBatch(vec![$($mutation.into()),+])
    };
}
```

**Pros**:
- **Zero overhead** - Macro expands to `MutationBatch(vec![...])`
- **Ergonomic** - Familiar syntax like `vec![]`
- **Auto-conversion** - Uses `.into()` to convert mutation types
- **Type safe** - Compiler checks mutation types

**Cons**:
- Adds macro to API surface (but macros are idiomatic in Rust)

### Final Decision

**We chose Option 2 + Option 3**: `MutationBatch` wrapper + `mutations![]` macro.

This provides:
- ✅ Zero-cost abstraction (no heap allocation)
- ✅ Ergonomic syntax via macro
- ✅ Type safety and extensibility
- ✅ Consistency with Query API pattern
- ✅ No external dependencies for core functionality

## Quick Start

```rust
use motlie_db::{Writer, AddNode, AddEdge, mutations, MutationRunnable, Id, TimestampMilli, EdgeSummary};

async fn example(writer: &Writer) -> anyhow::Result<()> {
    let alice_id = Id::new();
    let bob_id = Id::new();

    // Single mutation
    AddNode {
        id: alice_id,
        name: "Alice".to_string(),
        ts_millis: TimestampMilli::now(),
        valid_range: None,
    }
    .run(writer)
    .await?;

    // Batch mutations with macro
    mutations![
        AddNode {
            id: bob_id,
            name: "Bob".to_string(),
            ts_millis: TimestampMilli::now(),
            valid_range: None,
        },
        AddEdge {
            source_node_id: alice_id,
            target_node_id: bob_id,
            name: "follows".to_string(),
            ts_millis: TimestampMilli::now(),
            valid_range: None,
            summary: EdgeSummary::default(),
            weight: None,
        },
    ]
    .run(writer)
    .await?;

    Ok(())
}
```

## Core Concepts

### Writer

The `Writer` is a handle for sending mutations to the database:

```rust
use motlie_db::{Writer, create_mutation_writer};

// Create a writer (typically during app initialization)
let (writer, receiver) = create_mutation_writer(Default::default());
```

The `Writer` is:
- **Cloneable** - Can be shared across tasks/threads
- **Lightweight** - Just a channel sender
- **Thread-safe** - Safe to use from multiple tokio tasks

### Mutations

Mutations are lightweight value types that describe what changes to make:

```rust
// Construct a mutation (no I/O, just data)
let mutation = AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,
};

// Execute it (performs I/O)
mutation.run(&writer).await?;
```

### Runnable Trait

All mutation types implement the `Runnable` trait:

```rust
pub trait Runnable {
    async fn run(self, writer: &Writer) -> Result<()>;
}
```

This enables generic mutation execution and composability.

**Note**: To avoid naming conflicts with the Query API's `Runnable` trait, the mutation version is re-exported as `MutationRunnable`:

```rust
use motlie_db::MutationRunnable;  // For mutations
use motlie_db::QueryRunnable;     // For queries
```

In practice, you can import both as:
```rust
use motlie_db::mutation::Runnable as MutRunnable;
use motlie_db::query::Runnable as QueryRunnable;
```

Or just use the types directly without importing the trait (Rust's trait method resolution works automatically).

### Batching

Mutations can be batched for atomic execution:

```rust
// Using the mutations![] macro (recommended)
mutations![
    AddNode { /* ... */ },
    AddEdge { /* ... */ },
].run(&writer).await?;

// Manual construction
let mut batch = MutationBatch::new();
batch.push(AddNode { /* ... */ });
batch.push(AddEdge { /* ... */ });
batch.run(&writer).await?;

// Direct Vec<Mutation> via writer.send()
writer.send(vec![
    Mutation::AddNode(AddNode { /* ... */ }),
    Mutation::AddEdge(AddEdge { /* ... */ }),
]).await?;
```

## Mutation Types

### 1. AddNode

Add a new node to the graph:

```rust
use motlie_db::{AddNode, Id, TimestampMilli, MutationRunnable};

AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,  // Optional validity range
}
.run(&writer)
.await?;
```

**Fields**:
- `id: Id` - Unique identifier for the node (ULID)
- `name: String` - Human-readable name
- `ts_millis: TimestampMilli` - Creation timestamp
- `valid_range: Option<ValidRange>` - Optional validity period

### 2. AddEdge

Add a new edge between two nodes:

```rust
use motlie_db::{AddEdge, Id, TimestampMilli, EdgeSummary, MutationRunnable};

AddEdge {
    source_node_id: alice_id,
    target_node_id: bob_id,
    name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,
    summary: EdgeSummary::default(),
    weight: Some(1.0),
}
.run(&writer)
.await?;
```

**Fields**:
- `source_node_id: Id` - Source node
- `target_node_id: Id` - Target node
- `name: String` - Edge type/name
- `ts_millis: TimestampMilli` - Creation timestamp
- `valid_range: Option<ValidRange>` - Optional validity period
- `summary: EdgeSummary` - Summary information for the edge (fragment counts, timestamps)
- `weight: Option<f64>` - Optional weight for weighted graph algorithms

### 3. AddNodeFragment

Add time-series data associated with a node:

```rust
use motlie_db::{AddNodeFragment, Id, TimestampMilli, DataUrl, MutationRunnable};

AddNodeFragment {
    id: node_id,
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_text("Fragment data"),
    valid_range: None,
}
.run(&writer)
.await?;
```

**Fields**:
- `id: Id` - Node ID this fragment belongs to
- `ts_millis: TimestampMilli` - Fragment timestamp
- `content: DataUrl` - Fragment content (text, JSON, binary, etc.)
- `valid_range: Option<ValidRange>` - Optional validity period

### 4. AddEdgeFragment

Add time-series data associated with an edge:

```rust
use motlie_db::{AddEdgeFragment, Id, TimestampMilli, DataUrl, MutationRunnable};

AddEdgeFragment {
    src_id: alice_id,
    dst_id: bob_id,
    edge_name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_text("Fragment data"),
    valid_range: None,
}
.run(&writer)
.await?;
```

**Fields**:
- `src_id: Id` - Source node ID
- `dst_id: Id` - Destination node ID
- `edge_name: String` - Edge name/type
- `ts_millis: TimestampMilli` - Fragment timestamp
- `content: DataUrl` - Fragment content (text, JSON, binary, etc.)
- `valid_range: Option<ValidRange>` - Optional validity period

### 5. UpdateNodeValidSinceUntil

Update the temporal validity range of a node:

```rust
use motlie_db::{UpdateNodeValidSinceUntil, Id, TimestampMilli, ValidRange, MutationRunnable};

UpdateNodeValidSinceUntil {
    id: node_id,
    temporal_range: ValidRange {
        valid_from: TimestampMilli::now(),
        valid_to: Some(future_timestamp),
    },
    reason: "Updated validity period".to_string(),
}
.run(&writer)
.await?;
```

**Fields**:
- `id: Id` - Node ID to update
- `temporal_range: ValidRange` - New temporal validity range
- `reason: String` - Reason for the update

### 6. UpdateEdgeValidSinceUntil

Update the temporal validity range of an edge (using topology instead of edge ID):

```rust
use motlie_db::{UpdateEdgeValidSinceUntil, Id, TimestampMilli, ValidRange, MutationRunnable};

UpdateEdgeValidSinceUntil {
    src_id: alice_id,
    dst_id: bob_id,
    name: "follows".to_string(),
    temporal_range: ValidRange {
        valid_from: TimestampMilli::now(),
        valid_to: Some(future_timestamp),
    },
    reason: "Updated validity period".to_string(),
}
.run(&writer)
.await?;
```

**Fields**:
- `src_id: Id` - Source node ID
- `dst_id: Id` - Destination node ID
- `name: String` - Edge name/type
- `temporal_range: ValidRange` - New temporal validity range
- `reason: String` - Reason for the update

### 7. UpdateEdgeWeight

Update the weight of an edge (for weighted graph algorithms):

```rust
use motlie_db::{UpdateEdgeWeight, Id, MutationRunnable};

UpdateEdgeWeight {
    src_id: alice_id,
    dst_id: bob_id,
    name: "follows".to_string(),
    weight: 2.5,
}
.run(&writer)
.await?;
```

**Fields**:
- `src_id: Id` - Source node ID
- `dst_id: Id` - Destination node ID
- `name: String` - Edge name/type
- `weight: f64` - New weight value

## Common Patterns

### Pattern 1: Single Mutation

```rust
async fn create_user(
    writer: &Writer,
    name: String
) -> anyhow::Result<Id> {
    let user_id = Id::new();

    AddNode {
        id: user_id,
        name,
        ts_millis: TimestampMilli::now(),
        valid_range: None,
    }
    .run(writer)
    .await?;

    Ok(user_id)
}
```

### Pattern 2: Batched Mutations

```rust
async fn create_user_with_profile(
    writer: &Writer,
    name: String,
    bio: String
) -> anyhow::Result<Id> {
    let user_id = Id::new();

    mutations![
        AddNode {
            id: user_id,
            name,
            ts_millis: TimestampMilli::now(),
            valid_range: None,
        },
        AddNodeFragment {
            id: user_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(&bio),
            valid_range: None,
        },
    ]
    .run(writer)
    .await?;

    Ok(user_id)
}
```

### Pattern 3: Building Social Graph

```rust
async fn create_friendship(
    writer: &Writer,
    user1_id: Id,
    user2_id: Id
) -> anyhow::Result<()> {
    mutations![
        AddEdge {
            source_node_id: user1_id,
            target_node_id: user2_id,
            name: "follows".to_string(),
            ts_millis: TimestampMilli::now(),
            valid_range: None,
            summary: EdgeSummary::default(),
            weight: None,
        },
        AddEdge {
            source_node_id: user2_id,
            target_node_id: user1_id,
            name: "follows".to_string(),
            ts_millis: TimestampMilli::now(),
            valid_range: None,
            summary: EdgeSummary::default(),
            weight: None,
        },
    ]
    .run(writer)
    .await?;

    Ok(())
}
```

### Pattern 4: Programmatic Batch Construction

```rust
async fn bulk_import_users(
    writer: &Writer,
    users: Vec<(String, String)>  // (name, bio)
) -> anyhow::Result<()> {
    let mut batch = MutationBatch::new();

    for (name, bio) in users {
        let user_id = Id::new();

        batch.push(AddNode {
            id: user_id,
            name,
            ts_millis: TimestampMilli::now(),
            valid_range: None,
        });

        batch.push(AddNodeFragment {
            id: user_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(&bio),
            valid_range: None,
        });
    }

    batch.run(writer).await?;
    Ok(())
}
```

### Pattern 5: Conditional Mutations

```rust
async fn update_user_conditionally(
    writer: &Writer,
    user_id: Id,
    new_name: Option<String>,
    new_bio: Option<String>
) -> anyhow::Result<()> {
    let mut batch = MutationBatch::new();

    if let Some(name) = new_name {
        batch.push(AddNode {
            id: user_id,
            name,
            ts_millis: TimestampMilli::now(),
            valid_range: None,
        });
    }

    if let Some(bio) = new_bio {
        batch.push(AddNodeFragment {
            id: user_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(&bio),
            valid_range: None,
        });
    }

    if !batch.is_empty() {
        batch.run(writer).await?;
    }

    Ok(())
}
```

### Pattern 6: Temporal Validity

```rust
use motlie_db::schema::ValidRange;

async fn create_limited_time_offer(
    writer: &Writer,
    offer_name: String,
    valid_until: TimestampMilli
) -> anyhow::Result<Id> {
    let offer_id = Id::new();

    AddNode {
        id: offer_id,
        name: offer_name,
        ts_millis: TimestampMilli::now(),
        valid_range: Some(ValidRange {
            valid_from: TimestampMilli::now(),
            valid_to: Some(valid_until),
        }),
    }
    .run(writer)
    .await?;

    Ok(offer_id)
}
```

## Advanced Usage

### Generic Mutation Execution

```rust
use motlie_db::mutation::Runnable as MutRunnable;

async fn execute_with_retry<M>(
    mutation: M,
    writer: &Writer,
    max_attempts: usize,
) -> anyhow::Result<()>
where
    M: MutRunnable + Clone,
{
    for attempt in 1..=max_attempts {
        match mutation.clone().run(writer).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < max_attempts => {
                eprintln!("Attempt {} failed: {}", attempt, e);
                tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

// Usage
let mutation = AddNode { /* ... */ };
execute_with_retry(mutation, &writer, 3).await?;
```

### Mutation Composition

```rust
async fn create_post_with_tags(
    writer: &Writer,
    author_id: Id,
    content: String,
    tags: Vec<String>
) -> anyhow::Result<Id> {
    let post_id = Id::new();
    let ts = TimestampMilli::now();

    // Create post node
    AddNode {
        id: post_id,
        name: "post".to_string(),
        ts_millis: ts,
        valid_range: None,
    }
    .run(writer)
    .await?;

    // Add content fragment
    AddNodeFragment {
        id: post_id,
        ts_millis: ts,
        content: DataUrl::from_text(&content),
        valid_range: None,
    }
    .run(writer)
    .await?;

    // Create tags and edges
    let mut batch = MutationBatch::new();

    batch.push(AddEdge {
        source_node_id: author_id,
        target_node_id: post_id,
        name: "authored".to_string(),
        ts_millis: ts,
        valid_range: None,
        summary: EdgeSummary::default(),
        weight: None,
    });

    for tag in tags {
        let tag_id = Id::new();

        batch.push(AddNode {
            id: tag_id,
            name: tag,
            ts_millis: ts,
            valid_range: None,
        });

        batch.push(AddEdge {
            source_node_id: post_id,
            target_node_id: tag_id,
            name: "tagged".to_string(),
            ts_millis: ts,
            valid_range: None,
            summary: EdgeSummary::default(),
            weight: None,
        });
    }

    batch.run(writer).await?;

    Ok(post_id)
}
```

### Batch Size Management

```rust
async fn bulk_import_with_batching(
    writer: &Writer,
    items: Vec<String>,
    batch_size: usize
) -> anyhow::Result<()> {
    for chunk in items.chunks(batch_size) {
        let mut batch = MutationBatch::with_capacity(chunk.len());

        for name in chunk {
            batch.push(AddNode {
                id: Id::new(),
                name: name.clone(),
                ts_millis: TimestampMilli::now(),
                valid_range: None,
            });
        }

        batch.run(writer).await?;

        // Optional: Add delay between batches
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    Ok(())
}
```

## Migration Guide

### From Deprecated Writer API (v0.2.x)

**Before (Deprecated)**:
```rust
// Old API - helper methods
writer.add_node(AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,
}).await?;

writer.add_edge(AddEdge {
    id: Id::new(),
    source_node_id: alice_id,
    target_node_id: bob_id,
    name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,
}).await?;
```

**After (New API)**:
```rust
// New API - .run() pattern
AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,
}
.run(&writer)
.await?;

AddEdge {
    source_node_id: alice_id,
    target_node_id: bob_id,
    name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
    valid_range: None,
    summary: EdgeSummary::default(),
    weight: None,
}
.run(&writer)
.await?;
```

### Batching Migration

**Before**:
```rust
writer.send(vec![
    Mutation::AddNode(AddNode { /* ... */ }),
    Mutation::AddEdge(AddEdge { /* ... */ }),
]).await?;
```

**After**:
```rust
// Using mutations![] macro (recommended)
mutations![
    AddNode { /* ... */ },
    AddEdge { /* ... */ },
].run(&writer).await?;

// Or using MutationBatch
let mut batch = MutationBatch::new();
batch.push(AddNode { /* ... */ });
batch.push(AddEdge { /* ... */ });
batch.run(&writer).await?;
```

### Fragment Migration

**Before (using generic AddFragment)**:
```rust
AddFragment {
    id: entity_id,  // Could be node or edge
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_text("Fragment data"),
    valid_range: None,
}
.run(&writer)
.await?;
```

**After (separate types for nodes and edges)**:
```rust
// For node fragments
AddNodeFragment {
    id: node_id,
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_text("Fragment data"),
    valid_range: None,
}
.run(&writer)
.await?;

// For edge fragments
AddEdgeFragment {
    src_id: alice_id,
    dst_id: bob_id,
    edge_name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_text("Fragment data"),
    valid_range: None,
}
.run(&writer)
.await?;
```

### Edge Mutation Updates

**AddEdge Changes**:
- Removed: `id` field (edges no longer have unique IDs)
- Added: `summary: EdgeSummary` - Summary information for the edge
- Added: `weight: Option<f64>` - Optional weight for weighted graphs

**UpdateEdgeValidSinceUntil Changes**:
- Now uses topology (src_id, dst_id, name) instead of edge_id
- Example:
```rust
UpdateEdgeValidSinceUntil {
    src_id: alice_id,
    dst_id: bob_id,
    name: "follows".to_string(),
    temporal_range: ValidRange {
        valid_from: TimestampMilli::now(),
        valid_to: Some(future_timestamp),
    },
    reason: "Updated validity period".to_string(),
}
.run(&writer)
.await?;
```

**New Mutation: UpdateEdgeWeight**:
```rust
UpdateEdgeWeight {
    src_id: alice_id,
    dst_id: bob_id,
    name: "follows".to_string(),
    weight: 2.5,
}
.run(&writer)
.await?;
```

### Migration Checklist

1. ✅ Import new types:
   ```rust
   use motlie_db::{mutations, MutationBatch, MutationRunnable, EdgeSummary};
   ```

2. ✅ Update helper method calls:
   ```rust
   // Old: writer.add_node(AddNode { /* ... */ }).await?
   AddNode { /* ... */ }.run(&writer).await?

   // Old: writer.add_edge(AddEdge { /* ... */ }).await?
   AddEdge { /* ... */ }.run(&writer).await?
   ```

3. ✅ Update AddEdge to include new required fields:
   ```rust
   AddEdge {
       source_node_id: alice_id,
       target_node_id: bob_id,
       name: "follows".to_string(),
       ts_millis: TimestampMilli::now(),
       valid_range: None,
       summary: EdgeSummary::default(),  // New field
       weight: None,                     // New field
   }
   ```

4. ✅ Replace AddFragment with AddNodeFragment or AddEdgeFragment:
   ```rust
   // For nodes:
   AddNodeFragment { id: node_id, /* ... */ }

   // For edges:
   AddEdgeFragment { src_id, dst_id, edge_name, /* ... */ }
   ```

5. ✅ Update UpdateEdgeValidSinceUntil to use topology:
   ```rust
   // Old: used edge_id
   // New: uses src_id, dst_id, and name
   UpdateEdgeValidSinceUntil {
       src_id: alice_id,
       dst_id: bob_id,
       name: "follows".to_string(),
       temporal_range: /* ... */,
       reason: "...".to_string(),
   }
   ```

6. ✅ Update batch sends:
   ```rust
   // Old: writer.send(vec![Mutation::AddNode(...), ...]).await?
   mutations![AddNode { /* ... */ }, ...].run(&writer).await?
   ```

7. ✅ Suppress deprecation warnings during migration:
   ```rust
   #[allow(deprecated)]
   writer.add_node(AddNode { /* ... */ }).await?;
   ```

### Compatibility

The old Writer methods (`writer.add_node()`, etc.) are **deprecated but still functional**. You can migrate gradually:

```rust
// Both work during migration period:

// Old API (deprecated)
#[allow(deprecated)]
writer.add_node(node).await?;

// New API (recommended)
node.run(&writer).await?;
```

## Best Practices

1. **Use the mutations![] Macro for Batches**
   - Cleaner syntax than manual MutationBatch construction
   - Auto-conversion via `.into()`
   - Familiar `vec![]`-like syntax

2. **Batch Related Mutations**
   - Group related changes into atomic batches
   - Better performance (single transaction)
   - Ensures atomicity

3. **Set Appropriate Batch Sizes**
   - For bulk imports: 100-1,000 mutations per batch
   - For real-time updates: 1-10 mutations per batch
   - Monitor RocksDB transaction limits

4. **Use Temporal Validity When Needed**
   - Set `valid_to` for time-limited data
   - Leave None for permanent data
   - Remember: queries can filter by temporal validity

5. **Handle Errors Appropriately**
   - Mutations are atomic - failed batches roll back
   - Implement retry logic for transient failures
   - Log failures for debugging

## Performance Considerations

### Mutation Performance

- **Single mutation**: ~10-100µs (depends on storage backend)
- **Batched mutations**: ~50-500µs for 10-100 mutations
- **Bulk imports**: ~1-10ms for 1,000 mutations

### Batching Benefits

Batching provides significant performance improvements:

- **Individual mutations**: 10,000 mutations = ~1-10 seconds
- **Batched mutations** (100 per batch): 10,000 mutations = ~500ms - 1s
- **Speedup**: 2-10x faster with batching

### Memory Considerations

- `MutationBatch` has zero overhead beyond the `Vec<Mutation>` itself
- Each mutation struct is ~200-300 bytes
- 1,000 mutations ≈ 200-300 KB memory

## See Also

- [Query API Guide](query-api-guide.md) - Query API documentation
- [Writer API Reference](writer.md) - Complete Writer API documentation
- [Concurrency Guide](concurrency-and-storage-modes.md) - Threading and storage modes
- [Main README](../README.md) - Library overview
