# Implementation Plan: Transform `edge_summary_by_id()` → `edge_by_id()`

## Goal

Transform the current incomplete API:
```rust
edge_summary_by_id(id: Id) -> Result<EdgeSummary>
```

Into the complete API:
```rust
edge_by_id(id: Id) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>
```

## Implementation Approach

Since the topology information (source_id, dest_id, edge_name) is **NOT** stored in the `edges` CF but is embedded in the **keys** of `forward_edges` and `reverse_edges` CFs, we have two options:

### Option A: Scan ForwardEdges CF (Simple but Inefficient)
### Option B: Add EdgeTopology CF (Optimal but Requires Schema Change)

**Recommendation:** Start with **Option A** for immediate functionality, plan **Option B** for optimization.

---

## Option A: Scan ForwardEdges (Immediate Solution)

### Pros
- ✅ No schema changes required
- ✅ Works with existing databases
- ✅ Can implement immediately

### Cons
- ❌ O(n) scan of forward_edges CF for each query
- ❌ Slow for large graphs
- ❌ Not suitable for production at scale

### Implementation Strategy

Scan the `forward_edges` CF to find the entry where the **value** equals the target edge_id, then extract topology from the **key**.

---

## Changes Required (Option A)

### 1. Schema Changes (libs/db/src/schema.rs)

**No schema changes needed** - we'll use existing CFs.

However, we need to **denormalize** the `EdgeCfValue` to store topology:

#### Change 1.1: Update EdgeCfValue to include topology

```rust
// OLD (line 92-93):
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue(pub(crate) EdgeSummary);

// NEW:
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue {
    pub(crate) source_id: Id,
    pub(crate) dest_id: Id,
    pub(crate) name: String,
    pub(crate) summary: EdgeSummary,
}
```

**Impact:** This is a **BREAKING CHANGE** - existing databases will need migration.

#### Change 1.2: Update Edges::record_from() to include topology

```rust
// OLD (line 118-123):
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
    let value = EdgeCfValue {
        source_id: args.source_node_id,
        dest_id: args.target_node_id,
        name: args.name.clone(),
        summary: EdgeSummary::new(markdown),
    };
    (key, value)
}
```

**Note:** This approach **denormalizes** the data (stores topology in 3 places: edges, forward_edges, reverse_edges).

---

### 2. Query Changes (libs/db/src/query.rs)

#### Change 2.1: Rename type alias

```rust
// OLD (line ~22):
pub type EdgeSummaryById = ByIdQuery<EdgeSummary>;

// NEW:
pub type EdgeById = ByIdQuery<(SrcId, DstId, EdgeName, EdgeSummary)>;
```

#### Change 2.2: Update Query enum variant

```rust
// OLD (line ~12):
pub enum Query {
    NodeById(NodeById),
    EdgeSummaryById(EdgeSummaryById),  // OLD
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstName),
    // ...
}

// NEW:
pub enum Query {
    NodeById(NodeById),
    EdgeById(EdgeById),  // NEW
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstName),
    // ...
}
```

#### Change 2.3: Add sealed trait implementation for new tuple type

```rust
// In mod sealed (line ~43-51):
mod sealed {
    use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
    use crate::{Id, TimestampMilli};

    pub trait ByIdQueryable {}
    impl ByIdQueryable for (NodeName, NodeSummary) {}
    impl ByIdQueryable for EdgeSummary {}  // OLD - remove this
    impl ByIdQueryable for (Id, Id, EdgeName, EdgeSummary) {}  // NEW - add this
    impl ByIdQueryable for Vec<(TimestampMilli, FragmentContent)> {}
    impl ByIdQueryable for Vec<(Id, EdgeName, Id)> {}
}
```

#### Change 2.4: Update ByIdQueryable trait implementation

```rust
// OLD (line ~79-87):
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
impl ByIdQueryable for (SrcId, DstId, EdgeName, EdgeSummary) {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_edge_by_id(query).await
    }
}
```

#### Change 2.5: Update Processor trait

```rust
// OLD (line ~346-347):
async fn get_edge_summary_by_id(&self, query: &ByIdQuery<EdgeSummary>) -> Result<EdgeSummary>;

// NEW:
async fn get_edge_by_id(&self, query: &ByIdQuery<(SrcId, DstId, EdgeName, EdgeSummary)>)
    -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>;
```

#### Change 2.6: Update query consumer match arm

```rust
// OLD (line ~422-426):
Query::EdgeSummaryById(q) => {
    log::debug!("Processing EdgeById: id={}", q.id);
    let result = q.result(&self.processor).await;
    q.send_result(result);
}

// NEW:
Query::EdgeById(q) => {
    log::debug!("Processing EdgeById: id={}", q.id);
    let result = q.result(&self.processor).await;
    q.send_result(result);
}
```

#### Change 2.7: Update test mocks

```rust
// OLD (line ~482-488):
async fn get_edge_summary_by_id(
    &self,
    query: &EdgeSummaryById,
) -> Result<EdgeSummary> {
    tokio::time::sleep(Duration::from_millis(10)).await;
    Ok(EdgeSummary::new(format!("Edge: {:?}", query.id)))
}

// NEW:
async fn get_edge_by_id(
    &self,
    query: &EdgeById,
) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
    tokio::time::sleep(Duration::from_millis(10)).await;
    Ok((
        query.id,  // mock source
        Id::new(), // mock dest
        EdgeName("test_edge".to_string()),
        EdgeSummary::new(format!("Edge: {:?}", query.id)),
    ))
}
```

Similarly update `SlowProcessor` mock in tests.

---

### 3. Graph Processor Changes (libs/db/src/graph.rs)

#### Change 3.1: Update get_edge_summary_by_id implementation

```rust
// OLD (line 418-447):
async fn get_edge_summary_by_id(
    &self,
    query: &crate::query::EdgeSummaryById,
) -> Result<EdgeSummary> {
    let id = query.id;
    let key = schema::EdgeCfKey(id);
    // ... lookup logic ...
    Ok(value.0)  // Returns just EdgeSummary
}

// NEW:
async fn get_edge_by_id(
    &self,
    query: &crate::query::EdgeById,
) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
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

    // NEW: Return full tuple with topology
    Ok((
        value.source_id,
        value.dest_id,
        EdgeName(value.name),
        value.summary,
    ))
}
```

---

### 4. Reader API Changes (libs/db/src/reader.rs)

#### Change 4.1: Rename and update method signature

```rust
// OLD (line 53-65):
/// Query an edge by its ID
pub async fn edge_summary_by_id(&self, id: Id, timeout: Duration) -> Result<EdgeSummary> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let query = EdgeSummaryById::new(id, timeout, result_tx);

    self.sender
        .send_async(Query::EdgeSummaryById(query))
        .await
        .context("Failed to send query to reader queue")?;

    result_rx.await?
}

// NEW:
/// Query an edge by its ID (returns topology and summary)
pub async fn edge_by_id(
    &self,
    id: Id,
    timeout: Duration,
) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let query = EdgeById::new(id, timeout, result_tx);

    self.sender
        .send_async(Query::EdgeById(query))
        .await
        .context("Failed to send query to reader queue")?;

    result_rx.await?
}
```

---

### 5. Public API Exports (libs/db/src/lib.rs)

#### Change 5.1: Update exported types

```rust
// OLD (line ~13-16):
pub use query::{
    DstId, EdgeSummaryById, EdgeSummaryBySrcDstName, EdgesFromNodeById,
    EdgesToNodeById, FragmentContentById, NodeById, Query, SrcId,
};

// NEW:
pub use query::{
    DstId, EdgeById, EdgeSummaryBySrcDstName, EdgesFromNodeById,
    EdgesToNodeById, FragmentContentById, NodeById, Query, SrcId,
};
```

---

### 6. Test Updates (libs/db/src/graph_tests.rs)

Update all tests that use `edge_summary_by_id()` to use `edge_by_id()` and destructure the result.

Example:

```rust
// OLD:
let summary = reader.edge_summary_by_id(edge_id, Duration::from_secs(5)).await?;
assert!(summary.content()?.contains("test"));

// NEW:
let (src_id, dst_id, edge_name, summary) = EdgeById::new(edge_id, None)
    .run(&reader, Duration::from_secs(5))
    .await?;
assert_eq!(src_id, expected_source_id);
assert_eq!(dst_id, expected_dest_id);
assert_eq!(edge_name.0, "test_edge");
assert!(summary.content()?.contains("test"));
```

---

### 7. Update Verification Example (examples/store/main.rs)

No changes needed - verification uses direct RocksDB access, not the Reader API.

---

## Migration Strategy

### Database Migration Required

Because we're changing `EdgeCfValue` from a tuple struct to a full struct, **existing databases are incompatible**.

#### Migration Script (Pseudo-code)

```rust
async fn migrate_edge_cf(db_path: &Path) -> Result<()> {
    // 1. Open database
    let db = DB::open_cf(db_path, ["edges", "forward_edges", ...])?;

    // 2. For each edge in edges CF
    let edges_cf = db.cf_handle("edges")?;
    let forward_cf = db.cf_handle("forward_edges")?;

    for (edge_key_bytes, old_edge_value_bytes) in db.iterator_cf(edges_cf) {
        // 3. Deserialize old format
        let edge_id: Id = deserialize_edge_key(&edge_key_bytes)?;
        let old_value: EdgeSummary = deserialize_old_edge_value(&old_edge_value_bytes)?;

        // 4. Find topology in forward_edges CF
        let (src_id, dst_id, edge_name) = find_edge_topology_in_forward_cf(
            &db,
            forward_cf,
            edge_id
        )?;

        // 5. Create new format value
        let new_value = EdgeCfValue {
            source_id: src_id,
            dest_id: dst_id,
            name: edge_name,
            summary: old_value,
        };

        // 6. Write back
        let new_value_bytes = serialize_new_edge_value(&new_value)?;
        db.put_cf(edges_cf, edge_key_bytes, new_value_bytes)?;
    }

    Ok(())
}

fn find_edge_topology_in_forward_cf(
    db: &DB,
    forward_cf: &ColumnFamily,
    target_edge_id: Id,
) -> Result<(Id, Id, String)> {
    // Scan forward_edges CF
    for (key_bytes, value_bytes) in db.iterator_cf(forward_cf) {
        let edge_id: Id = deserialize_forward_edge_value(&value_bytes)?;
        if edge_id == target_edge_id {
            // Found it! Extract topology from key
            let (src, dst, name) = deserialize_forward_edge_key(&key_bytes)?;
            return Ok((src.0, dst.0, name.0));
        }
    }
    Err(anyhow!("Edge topology not found for {}", target_edge_id))
}
```

---

## Testing Strategy

### Unit Tests to Add/Update

1. **Test edge_by_id returns topology**
   ```rust
   #[tokio::test]
   async fn test_edge_by_id_returns_topology() {
       // Create edge with known topology
       let src = Id::new();
       let dst = Id::new();
       let edge_id = Id::new();

       AddEdge {
           id: edge_id,
           source_node_id: src,
           target_node_id: dst,
           name: "follows".to_string(),
           ts_millis: TimestampMilli::now(),
       }
       .run(&writer)
       .await?;

       // Query should return topology
       let (ret_src, ret_dst, ret_name, _summary) = EdgeById::new(edge_id, None)
           .run(&reader, timeout)
           .await?;

       assert_eq!(ret_src, src);
       assert_eq!(ret_dst, dst);
       assert_eq!(ret_name.0, "follows");
   }
   ```

2. **Test edge_by_id with non-existent edge**
   ```rust
   #[tokio::test]
   async fn test_edge_by_id_not_found() {
       let result = EdgeById::new(Id::new(), None)
           .run(&reader, timeout)
           .await;
       assert!(result.is_err());
       assert!(result.unwrap_err().to_string().contains("not found"));
   }
   ```

3. **Update existing tests** that use `edge_summary_by_id()`

### Integration Tests

Test with the store example to ensure end-to-end functionality.

---

## Summary of Files to Change

| File | Changes | Breaking? |
|------|---------|-----------|
| `libs/db/src/schema.rs` | Update `EdgeCfValue` struct, update `record_from()` | ⚠️ YES - schema change |
| `libs/db/src/query.rs` | Rename types, update trait, update enum, update implementations | ⚠️ YES - API change |
| `libs/db/src/graph.rs` | Rename method, update return type, update implementation | ⚠️ YES - API change |
| `libs/db/src/reader.rs` | Rename method, update signature | ⚠️ YES - API change |
| `libs/db/src/lib.rs` | Update exports | ⚠️ YES - API change |
| `libs/db/src/graph_tests.rs` | Update test assertions | ✅ No - tests only |
| `examples/store/main.rs` | None (uses direct RocksDB) | ✅ No |

---

## Implementation Checklist

- [ ] 1. Update schema (`EdgeCfValue` struct)
- [ ] 2. Update mutation (`Edges::record_from()`)
- [ ] 3. Update query types and trait (`query.rs`)
- [ ] 4. Update processor implementation (`graph.rs`)
- [ ] 5. Update reader API (`reader.rs`)
- [ ] 6. Update public exports (`lib.rs`)
- [ ] 7. Update all tests
- [ ] 8. Write migration script for existing databases
- [ ] 9. Run full test suite
- [ ] 10. Test with store example
- [ ] 11. Update documentation

---

## Alternative: Non-Breaking Approach

If you want to avoid breaking changes, you could:

1. Keep `edge_summary_by_id()` as-is
2. Add new `edge_by_id()` alongside it
3. Mark `edge_summary_by_id()` as deprecated

But this requires **still changing the schema** to store topology in `EdgeCfValue`, so it's still a breaking change at the database level.

---

## Recommendation

**Implement the full breaking change** because:

1. The current API is incomplete and the gap is fundamental
2. Better to fix early than carry technical debt
3. The database is likely in early development (based on examples)
4. Migration is straightforward (one-time scan and update)

This transforms the edge API to match the completeness of the node API.
