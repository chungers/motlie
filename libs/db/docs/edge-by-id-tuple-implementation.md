# Implementation Plan: Transform `edge_summary_by_id()` → `edge_by_id()` Using Tuple

## Goal

Transform the current incomplete API:
```rust
edge_summary_by_id(id: Id) -> Result<EdgeSummary>
```

Into the complete API:
```rust
edge_by_id(id: Id) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>
```

## Design Decision: Use Tuple for EdgeCfValue

Following the pattern established by `NodeCfValue(NodeName, NodeSummary)`, we'll use a tuple:

```rust
// NodeCfValue pattern (existing):
struct NodeCfValue(NodeName, NodeSummary);

// EdgeCfValue pattern (proposed):
struct EdgeCfValue(SrcId, DstId, EdgeName, EdgeSummary);
```

This is **consistent** with the existing schema design and avoids introducing a different pattern for edges.

---

## Implementation Overview

### Key Changes

1. **Schema:** `EdgeCfValue` tuple: `(EdgeSummary)` → `(SrcId, DstId, EdgeName, EdgeSummary)`
2. **Query:** Type alias and return types updated
3. **Graph:** Processor returns full tuple
4. **Reader:** API renamed and return type updated
5. **Tests:** Updated to destructure topology

### Breaking Changes

⚠️ **YES** - This is a breaking schema and API change requiring:
- Database migration
- API consumers must update
- Tests must be updated

---

## Detailed Changes

### 1. Schema Changes (libs/db/src/schema.rs)

#### Change 1.1: Update EdgeCfValue tuple structure

**Location:** Line 92-93

```rust
// OLD:
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue(pub(crate) EdgeSummary);

// NEW:
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue(
    pub(crate) Id,          // source_id
    pub(crate) Id,          // dest_id
    pub(crate) EdgeName,    // edge name
    pub(crate) EdgeSummary, // edge summary/content
);
```

**Rationale:** Mirrors the `NodeCfValue(NodeName, NodeSummary)` pattern.

---

#### Change 1.2: Update Edges::record_from()

**Location:** Line 118-123

```rust
// OLD:
fn record_from(args: &AddEdge) -> (EdgeCfKey, EdgeCfValue) {
    let key = EdgeCfKey(args.id);
    let markdown = format!("<!-- id={} -->]\n# {}\n# Summary\n", args.id, args.name);
    let value = EdgeCfValue(EdgeSummary::new(markdown));
    (key, value)
}

// NEW:
fn record_from(args: &AddEdge) -> (EdgeCfKey, EdgeCfValue) {
    let key = EdgeCfKey(args.id);
    let markdown = format!("<!-- id={} -->]\n# {}\n# Summary\n", args.id, args.name);
    let value = EdgeCfValue(
        args.source_node_id,              // SrcId
        args.target_node_id,              // DstId
        EdgeName(args.name.clone()),      // EdgeName
        EdgeSummary::new(markdown),       // EdgeSummary
    );
    (key, value)
}
```

---

### 2. Query Changes (libs/db/src/query.rs)

#### Change 2.1: Rename type alias

**Location:** Line ~22

```rust
// OLD:
pub type EdgeSummaryByIdQuery = ByIdQuery<EdgeSummary>;

// NEW:
pub type EdgeByIdQuery = ByIdQuery<(Id, Id, EdgeName, EdgeSummary)>;
```

**Note:** Using `Id` directly instead of `SrcId`/`DstId` type aliases in the tuple for simplicity. The Reader API will use `SrcId`/`DstId` for clarity.

---

#### Change 2.2: Update Query enum variant

**Location:** Line ~10-17

```rust
// OLD:
pub enum Query {
    NodeById(NodeByIdQuery),
    EdgeSummaryById(EdgeSummaryByIdQuery),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameQuery),
    FragmentContentById(FragmentContentByIdQuery),
    EdgesFromNodeById(EdgesFromNodeByIdQuery),
    EdgesToNodeById(EdgesToNodeByIdQuery),
}

// NEW:
pub enum Query {
    NodeById(NodeByIdQuery),
    EdgeById(EdgeByIdQuery),  // Renamed
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameQuery),
    FragmentContentById(FragmentContentByIdQuery),
    EdgesFromNodeById(EdgesFromNodeByIdQuery),
    EdgesToNodeById(EdgesToNodeByIdQuery),
}
```

---

#### Change 2.3: Update sealed trait implementation

**Location:** Line ~43-52

```rust
// OLD:
mod sealed {
    use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
    use crate::{Id, TimestampMilli};

    pub trait ByIdQueryable {}
    impl ByIdQueryable for (NodeName, NodeSummary) {}
    impl ByIdQueryable for EdgeSummary {}
    impl ByIdQueryable for Vec<(TimestampMilli, FragmentContent)> {}
    impl ByIdQueryable for Vec<(Id, EdgeName, Id)> {}
}

// NEW:
mod sealed {
    use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
    use crate::{Id, TimestampMilli};

    pub trait ByIdQueryable {}
    impl ByIdQueryable for (NodeName, NodeSummary) {}
    impl ByIdQueryable for (Id, Id, EdgeName, EdgeSummary) {}  // Updated tuple
    impl ByIdQueryable for Vec<(TimestampMilli, FragmentContent)> {}
    impl ByIdQueryable for Vec<(Id, EdgeName, Id)> {}
}
```

---

#### Change 2.4: Update ByIdQueryable trait implementation

**Location:** Line ~78-87

```rust
// OLD:
#[async_trait::async_trait]
impl ByIdQueryable for EdgeSummary {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_edge_summary_by_id(query).await
    }
}

// NEW:
#[async_trait::async_trait]
impl ByIdQueryable for (Id, Id, EdgeName, EdgeSummary) {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_edge_by_id(query).await
    }
}
```

---

#### Change 2.5: Update Processor trait

**Location:** Line ~340-347

```rust
// OLD:
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    async fn get_node_by_id(&self, query: &ByIdQuery<(NodeName, NodeSummary)>)
        -> Result<(NodeName, NodeSummary)>;

    async fn get_edge_summary_by_id(&self, query: &ByIdQuery<EdgeSummary>)
        -> Result<EdgeSummary>;

    // ... other methods ...
}

// NEW:
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    async fn get_node_by_id(&self, query: &ByIdQuery<(NodeName, NodeSummary)>)
        -> Result<(NodeName, NodeSummary)>;

    async fn get_edge_by_id(&self, query: &ByIdQuery<(Id, Id, EdgeName, EdgeSummary)>)
        -> Result<(Id, Id, EdgeName, EdgeSummary)>;

    // ... other methods ...
}
```

---

#### Change 2.6: Update query consumer match arm

**Location:** Line ~417-425

```rust
// OLD:
match query {
    Query::NodeById(q) => {
        log::debug!("Processing NodeById: id={}", q.id);
        let result = q.result(&self.processor).await;
        q.send_result(result);
    }
    Query::EdgeSummaryById(q) => {
        log::debug!("Processing EdgeById: id={}", q.id);
        let result = q.result(&self.processor).await;
        q.send_result(result);
    }
    // ...
}

// NEW:
match query {
    Query::NodeById(q) => {
        log::debug!("Processing NodeById: id={}", q.id);
        let result = q.result(&self.processor).await;
        q.send_result(result);
    }
    Query::EdgeById(q) => {
        log::debug!("Processing EdgeById: id={}", q.id);
        let result = q.result(&self.processor).await;
        q.send_result(result);
    }
    // ...
}
```

---

#### Change 2.7: Update test mock processors

**Location:** Line ~473-488 and ~577-593

```rust
// OLD TestProcessor:
async fn get_edge_summary_by_id(
    &self,
    query: &EdgeSummaryByIdQuery,
) -> Result<EdgeSummary> {
    tokio::time::sleep(Duration::from_millis(10)).await;
    Ok(EdgeSummary::new(format!("Edge: {:?}", query.id)))
}

// NEW TestProcessor:
async fn get_edge_by_id(
    &self,
    query: &EdgeByIdQuery,
) -> Result<(Id, Id, EdgeName, EdgeSummary)> {
    tokio::time::sleep(Duration::from_millis(10)).await;
    Ok((
        query.id,                          // mock source
        Id::new(),                         // mock dest
        EdgeName("test_edge".to_string()), // mock edge name
        EdgeSummary::new(format!("Edge: {:?}", query.id)),
    ))
}

// Similar update for SlowProcessor
```

---

### 3. Graph Processor Changes (libs/db/src/graph.rs)

#### Change 3.1: Rename and update method implementation

**Location:** Line 418-447

```rust
// OLD:
async fn get_edge_summary_by_id(
    &self,
    query: &crate::query::EdgeSummaryByIdQuery,
) -> Result<EdgeSummary> {
    let id = query.id;
    let key = schema::EdgeCfKey(id);
    let key_bytes = schema::Edges::key_to_bytes(&key)
        .map_err(|e| anyhow::anyhow!("Failed to serialize key: {}", e))?;

    // Handle both readonly and readwrite modes
    let value_bytes = if let Ok(db) = self.storage.db() {
        let cf = db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
        })?;
        db.get_cf(cf, key_bytes)?
    } else {
        let txn_db = self.storage.transaction_db()?;
        let cf = txn_db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
        })?;
        txn_db.get_cf(cf, key_bytes)?
    };

    let value_bytes = value_bytes.ok_or_else(|| anyhow::anyhow!("Edge not found: {}", id))?;

    let value: schema::EdgeCfValue = schema::Edges::value_from_bytes(&value_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

    Ok(value.0)  // Returns only EdgeSummary
}

// NEW:
async fn get_edge_by_id(
    &self,
    query: &crate::query::EdgeByIdQuery,
) -> Result<(Id, Id, EdgeName, EdgeSummary)> {
    let id = query.id;
    let key = schema::EdgeCfKey(id);
    let key_bytes = schema::Edges::key_to_bytes(&key)
        .map_err(|e| anyhow::anyhow!("Failed to serialize key: {}", e))?;

    // Handle both readonly and readwrite modes
    let value_bytes = if let Ok(db) = self.storage.db() {
        let cf = db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
        })?;
        db.get_cf(cf, key_bytes)?
    } else {
        let txn_db = self.storage.transaction_db()?;
        let cf = txn_db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
        })?;
        txn_db.get_cf(cf, key_bytes)?
    };

    let value_bytes = value_bytes.ok_or_else(|| anyhow::anyhow!("Edge not found: {}", id))?;

    let value: schema::EdgeCfValue = schema::Edges::value_from_bytes(&value_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

    // Return full tuple: (source_id, dest_id, edge_name, summary)
    Ok((value.0, value.1, value.2, value.3))
}
```

**Key Change:** Return `(value.0, value.1, value.2, value.3)` instead of just `value.0`.

---

### 4. Reader API Changes (libs/db/src/reader.rs)

#### Change 4.1: Rename method and update signature

**Location:** Line 53-65

```rust
// OLD:
/// Query an edge by its ID
pub async fn edge_summary_by_id(&self, id: Id, timeout: Duration)
    -> Result<EdgeSummary>
{
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let query = EdgeSummaryByIdQuery::new(id, timeout, result_tx);

    self.sender
        .send_async(Query::EdgeSummaryById(query))
        .await
        .context("Failed to send query to reader queue")?;

    result_rx.await?
}

// NEW:
/// Query an edge by its ID (returns topology and summary)
/// Returns (source_id, dest_id, edge_name, summary)
pub async fn edge_by_id(
    &self,
    id: Id,
    timeout: Duration,
) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let query = EdgeByIdQuery::new(id, timeout, result_tx);

    self.sender
        .send_async(Query::EdgeById(query))
        .await
        .context("Failed to send query to reader queue")?;

    // Result is (Id, Id, EdgeName, EdgeSummary)
    // We return it with SrcId/DstId type aliases for clarity
    result_rx.await?
}
```

**Note:** The return type uses `SrcId` and `DstId` type aliases (which are both `Id`) to make the API self-documenting.

---

#### Change 4.2: Update imports

**Location:** Line 4-9

```rust
// OLD:
use crate::query::{
    DstId, EdgeSummaryByIdQuery, EdgeSummaryBySrcDstNameQuery, EdgesFromNodeByIdQuery,
    EdgesToNodeByIdQuery, FragmentContentByIdQuery, NodeByIdQuery, Query, SrcId,
};

// NEW:
use crate::query::{
    DstId, EdgeByIdQuery, EdgeSummaryBySrcDstNameQuery, EdgesFromNodeByIdQuery,
    EdgesToNodeByIdQuery, FragmentContentByIdQuery, NodeByIdQuery, Query, SrcId,
};
```

---

### 5. Public API Exports (libs/db/src/lib.rs)

#### Change 5.1: Update exported types

**Location:** Line ~13-16

```rust
// OLD:
pub use query::{
    DstId, EdgeSummaryByIdQuery, EdgeSummaryBySrcDstNameQuery, EdgesFromNodeByIdQuery,
    EdgesToNodeByIdQuery, FragmentContentByIdQuery, NodeByIdQuery, Query, SrcId,
};

// NEW:
pub use query::{
    DstId, EdgeByIdQuery, EdgeSummaryBySrcDstNameQuery, EdgesFromNodeByIdQuery,
    EdgesToNodeByIdQuery, FragmentContentByIdQuery, NodeByIdQuery, Query, SrcId,
};
```

---

### 6. Test Updates (libs/db/src/graph_tests.rs)

Update all tests that use `edge_summary_by_id()`:

```rust
// OLD pattern:
let summary = reader.edge_summary_by_id(edge_id, Duration::from_secs(5)).await?;
assert!(summary.content()?.contains("expected content"));

// NEW pattern:
let (src_id, dst_id, edge_name, summary) = reader
    .edge_by_id(edge_id, Duration::from_secs(5))
    .await?;

assert_eq!(src_id, expected_source_id);
assert_eq!(dst_id, expected_dest_id);
assert_eq!(edge_name.0, "expected_edge_name");
assert!(summary.content()?.contains("expected content"));
```

#### Specific Test Updates

Search for all occurrences of `edge_summary_by_id` in test files and update them.

---

## Database Migration Strategy

### Migration Required

The `EdgeCfValue` structure changes from 1 field to 4 fields:
- **OLD:** `EdgeCfValue(EdgeSummary)`
- **NEW:** `EdgeCfValue(SrcId, DstId, EdgeName, EdgeSummary)`

### Migration Script (Pseudo-code)

```rust
use anyhow::Result;
use rocksdb::DB;
use std::path::Path;

/// Migrate existing database to new EdgeCfValue schema
pub fn migrate_edge_cf(db_path: &Path) -> Result<()> {
    println!("Opening database at {:?}", db_path);
    let db = DB::open_cf(
        db_path,
        &["nodes", "edges", "forward_edges", "reverse_edges", "fragments"],
    )?;

    let edges_cf = db.cf_handle("edges")
        .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;
    let forward_cf = db.cf_handle("forward_edges")
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    println!("Scanning edges column family...");
    let mut migration_count = 0;
    let mut edges_to_update = Vec::new();

    // Collect all edges that need migration
    for item in db.iterator_cf(edges_cf, rocksdb::IteratorMode::Start) {
        let (key_bytes, old_value_bytes) = item?;

        // Deserialize edge ID from key
        let edge_id = deserialize_edge_key(&key_bytes)?;

        // Deserialize old value (single EdgeSummary)
        let old_summary = deserialize_old_edge_value(&old_value_bytes)?;

        // Find topology by scanning forward_edges CF
        if let Some((src_id, dst_id, edge_name)) =
            find_topology_in_forward_cf(&db, forward_cf, edge_id)?
        {
            edges_to_update.push((
                key_bytes.to_vec(),
                edge_id,
                src_id,
                dst_id,
                edge_name,
                old_summary,
            ));
            migration_count += 1;
        } else {
            eprintln!("Warning: No topology found for edge {}", edge_id);
        }
    }

    println!("Found {} edges to migrate", migration_count);

    // Update each edge with new schema
    for (key_bytes, edge_id, src_id, dst_id, edge_name, summary) in edges_to_update {
        // Create new EdgeCfValue tuple
        let new_value = EdgeCfValue(src_id, dst_id, edge_name, summary);
        let new_value_bytes = serialize_new_edge_value(&new_value)?;

        // Write back to edges CF
        db.put_cf(edges_cf, key_bytes, new_value_bytes)?;

        if migration_count % 100 == 0 {
            println!("Migrated {} edges...", migration_count);
        }
    }

    println!("Migration complete: {} edges updated", migration_count);
    Ok(())
}

/// Scan forward_edges CF to find topology for a given edge_id
fn find_topology_in_forward_cf(
    db: &DB,
    forward_cf: &rocksdb::ColumnFamily,
    target_edge_id: Id,
) -> Result<Option<(Id, Id, EdgeName)>> {
    // Scan all entries in forward_edges CF
    for item in db.iterator_cf(forward_cf, rocksdb::IteratorMode::Start) {
        let (key_bytes, value_bytes) = item?;

        // Deserialize value to get edge_id
        let edge_id: Id = deserialize_forward_edge_value(&value_bytes)?;

        if edge_id == target_edge_id {
            // Found it! Extract topology from key
            let (src, dst, name) = deserialize_forward_edge_key(&key_bytes)?;
            return Ok(Some((src.0, dst.0, name)));
        }
    }

    Ok(None)
}

// Helper deserialization functions
fn deserialize_edge_key(bytes: &[u8]) -> Result<Id> {
    #[derive(serde::Deserialize)]
    struct EdgeCfKey(Id);
    let key: EdgeCfKey = rmp_serde::from_slice(bytes)?;
    Ok(key.0)
}

fn deserialize_old_edge_value(bytes: &[u8]) -> Result<EdgeSummary> {
    #[derive(serde::Deserialize)]
    struct OldEdgeCfValue(EdgeSummary);
    let value: OldEdgeCfValue = rmp_serde::from_slice(bytes)?;
    Ok(value.0)
}

fn serialize_new_edge_value(value: &EdgeCfValue) -> Result<Vec<u8>> {
    Ok(rmp_serde::to_vec(value)?)
}

fn deserialize_forward_edge_value(bytes: &[u8]) -> Result<Id> {
    #[derive(serde::Deserialize)]
    struct ForwardEdgeCfValue(Id);
    let value: ForwardEdgeCfValue = rmp_serde::from_slice(bytes)?;
    Ok(value.0)
}

fn deserialize_forward_edge_key(bytes: &[u8]) -> Result<(EdgeSourceId, EdgeDestinationId, EdgeName)> {
    #[derive(serde::Deserialize)]
    struct ForwardEdgeCfKey(EdgeSourceId, EdgeDestinationId, EdgeName);
    let key: ForwardEdgeCfKey = rmp_serde::from_slice(bytes)?;
    Ok((key.0, key.1, key.2))
}
```

---

## Testing Strategy

### Unit Tests to Add

#### Test 1: Edge returns full topology

```rust
#[tokio::test]
async fn test_edge_by_id_returns_complete_topology() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_edge_by_id");

    // Setup
    let config = WriterConfig::default();
    let (writer, mutation_receiver) = create_mutation_writer(config.clone());
    let mutation_handle = spawn_graph_consumer(
        mutation_receiver,
        config.clone(),
        &db_path,
    );

    let (reader, query_receiver) = create_query_reader(ReaderConfig::default());
    let query_handle = spawn_query_consumer(query_receiver, config, &db_path);

    // Create edge with known topology
    let src_id = Id::new();
    let dst_id = Id::new();
    let edge_id = Id::new();

    writer.add_node(AddNode {
        id: src_id,
        name: "Alice".to_string(),
        ts_millis: TimestampMilli::now(),
    }).await.unwrap();

    writer.add_node(AddNode {
        id: dst_id,
        name: "Bob".to_string(),
        ts_millis: TimestampMilli::now(),
    }).await.unwrap();

    writer.add_edge(AddEdge {
        id: edge_id,
        source_node_id: src_id,
        target_node_id: dst_id,
        name: "follows".to_string(),
        ts_millis: TimestampMilli::now(),
    }).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Query edge and verify topology
    let (ret_src, ret_dst, ret_name, summary) = reader
        .edge_by_id(edge_id, Duration::from_secs(5))
        .await
        .unwrap();

    assert_eq!(ret_src, src_id, "Source ID should match");
    assert_eq!(ret_dst, dst_id, "Destination ID should match");
    assert_eq!(ret_name.0, "follows", "Edge name should match");
    assert!(summary.content().unwrap().contains("follows"));

    // Cleanup
    drop(writer);
    drop(reader);
    mutation_handle.await.unwrap().unwrap();
    query_handle.await.unwrap().unwrap();
}
```

#### Test 2: Edge not found error

```rust
#[tokio::test]
async fn test_edge_by_id_not_found() {
    // ... setup ...

    let result = reader.edge_by_id(Id::new(), Duration::from_secs(5)).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}
```

#### Test 3: Verify existing forward_and_reverse_edge_queries test still works

Update the existing test to also query edges by ID and verify topology.

---

## Summary of Changes

### Files Modified (7 files)

| File | Lines Changed | Type | Breaking? |
|------|---------------|------|-----------|
| `libs/db/src/schema.rs` | ~10 | Schema tuple update | ⚠️ YES |
| `libs/db/src/query.rs` | ~50 | Type aliases, trait, impl | ⚠️ YES |
| `libs/db/src/graph.rs` | ~10 | Method rename, return type | ⚠️ YES |
| `libs/db/src/reader.rs` | ~15 | Method rename, signature | ⚠️ YES |
| `libs/db/src/lib.rs` | ~2 | Export update | ⚠️ YES |
| `libs/db/src/graph_tests.rs` | ~50 | Test updates | ✅ No |
| Migration script (new file) | ~200 | New migration tool | ✅ No |

### Key Pattern: Tuple Consistency

The change maintains consistency with the existing schema pattern:

```rust
// Nodes (existing pattern):
struct NodeCfValue(NodeName, NodeSummary);
//                 ^^^^^^^^  ^^^^^^^^^^^
//                 identity  content

// Edges (new pattern):
struct EdgeCfValue(SrcId, DstId, EdgeName, EdgeSummary);
//                 ^^^^^  ^^^^^  ^^^^^^^^  ^^^^^^^^^^^
//                 topology      identity  content
```

Both use **tuple structs** with ordered fields representing different aspects of the entity.

---

## Implementation Checklist

- [ ] 1. Update `EdgeCfValue` tuple in `schema.rs`
- [ ] 2. Update `Edges::record_from()` in `schema.rs`
- [ ] 3. Update type alias `EdgeByIdQuery` in `query.rs`
- [ ] 4. Update `Query` enum variant in `query.rs`
- [ ] 5. Update sealed trait in `query.rs`
- [ ] 6. Update `ByIdQueryable` impl in `query.rs`
- [ ] 7. Update `Processor` trait in `query.rs`
- [ ] 8. Update consumer match arm in `query.rs`
- [ ] 9. Update test mocks in `query.rs`
- [ ] 10. Rename method in `graph.rs`
- [ ] 11. Update return statement in `graph.rs`
- [ ] 12. Rename method in `reader.rs`
- [ ] 13. Update imports in `reader.rs`
- [ ] 14. Update exports in `lib.rs`
- [ ] 15. Update all test assertions in `graph_tests.rs`
- [ ] 16. Add new topology verification tests
- [ ] 17. Write migration script
- [ ] 18. Test migration on sample database
- [ ] 19. Run full test suite
- [ ] 20. Test store example end-to-end
- [ ] 21. Update documentation

---

## Breaking Changes Impact

### Schema Level
- `EdgeCfValue` tuple structure changes
- Existing databases **incompatible** without migration
- Migration script **required** before upgrade

### API Level
- `edge_summary_by_id()` → `edge_by_id()`
- Return type: `EdgeSummary` → `(SrcId, DstId, EdgeName, EdgeSummary)`
- Consumers must update to destructure tuple

### Type System Level
- `EdgeSummaryByIdQuery` → `EdgeByIdQuery`
- Internal type changes propagate through query system

---

## Recommendation

**Proceed with implementation** using tuple approach because:

1. ✅ **Consistency:** Matches existing `NodeCfValue` tuple pattern
2. ✅ **Simplicity:** No new struct types, just extend existing pattern
3. ✅ **Performance:** Single O(1) lookup, no joins needed
4. ✅ **Completeness:** Fixes fundamental API gap
5. ⚠️ **Breaking:** Acceptable for early-stage database

The tuple approach is **cleaner** than introducing a new struct and maintains **design consistency** with the nodes schema.
