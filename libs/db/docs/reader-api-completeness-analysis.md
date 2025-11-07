# Reader API Completeness Analysis

## Question: Is `node_by_name` the only gap for a complete Reader API?

**Short Answer:** No. There are several API gaps beyond `node_by_name` for a complete point-query interface.

## Current Reader API (6 methods)

```rust
impl Reader {
    // 1. Node queries
    pub async fn node_by_id(id: Id, timeout: Duration)
        -> Result<(NodeName, NodeSummary)>

    // 2. Edge queries
    pub async fn edge_summary_by_id(id: Id, timeout: Duration)
        -> Result<EdgeSummary>

    pub async fn edge_by_src_dst_name(source_id: Id, dest_id: Id, name: String, timeout: Duration)
        -> Result<(Id, EdgeSummary)>

    // 3. Navigation queries
    pub async fn edges_from_node_by_id(id: Id, timeout: Duration)
        -> Result<Vec<(SrcId, EdgeName, DstId)>>

    pub async fn edges_to_node_by_id(id: Id, timeout: Duration)
        -> Result<Vec<(DstId, EdgeName, SrcId)>>

    // 4. Fragment queries
    pub async fn fragments_by_id(id: Id, timeout: Duration)
        -> Result<Vec<(TimestampMilli, FragmentContent)>>
}
```

## Database Schema Analysis

### Column Families
1. **Nodes**: `NodeCfKey(Id)` → `NodeCfValue(NodeName, NodeSummary)`
2. **Edges**: `EdgeCfKey(Id)` → `EdgeCfValue(EdgeSummary)`
3. **ForwardEdges**: `ForwardEdgeCfKey(SrcId, DstId, EdgeName)` → `ForwardEdgeCfValue(EdgeId)`
4. **ReverseEdges**: `ReverseEdgeCfKey(DstId, SrcId, EdgeName)` → `ReverseEdgeCfValue(EdgeId)`
5. **Fragments**: `FragmentCfKey(Id, TimestampMilli)` → `FragmentCfValue(FragmentContent)`

### Indexing Structure
- **Nodes** indexed by: ID only
- **Edges** indexed by: ID (primary), (SrcId, DstId, Name) triple
- **Fragments** indexed by: (ID, Timestamp) composite key

### Notable: Node names are NOT indexed
The schema stores node names as part of the value in `NodeCfValue(NodeName, NodeSummary)`, not as a separate index. This means there's no efficient way to look up a node by name without scanning all nodes.

## API Gaps for Complete Point-Query Interface

### Gap 1: Node Lookup by Name ❌

**Missing:** `node_by_name(name: String, timeout: Duration) -> Result<Option<(Id, NodeSummary)>>`

**Use Case:**
- Applications that work with human-readable names (like the store example)
- User interfaces where users type node names
- Integration with external systems using names

**Current Workaround:**
- Application must maintain own name→ID mapping
- Or scan all nodes (requires bulk iteration, not in Reader API)

**Schema Impact:**
- Requires new index: "nodes_by_name" CF with `NodeNameKey(String)` → `NodeNameValue(Id)`
- Must be updated on every node creation/update

**Implementation Complexity:** Medium
- New CF for reverse index
- Mutation pipeline must write to both Nodes and NodesByName CFs
- Query is straightforward once index exists

---

### Gap 2: Edge Lookup by ID Returns Only Summary ❌

**Current:** `edge_summary_by_id(id, timeout) -> Result<EdgeSummary>`

**Missing:** Full edge details including topology

**Problem:** You get the edge's content summary, but NOT:
- Source node ID
- Destination node ID
- Edge name

**Use Case:**
- Given an edge ID, want to know what it connects
- Navigation: "show me the nodes this edge connects"
- Debugging: "what is edge X?"

**Proposed:** `edge_by_id(id: Id, timeout) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>`

**Current Workaround:**
- Query fragments or other metadata to infer topology
- No direct way to get this information

**Schema Support:**
- The information exists in ForwardEdges/ReverseEdges CFs
- Need to scan one of these CFs to find the key containing this edge ID
- Or maintain reverse index: EdgeId → (SrcId, DstId, EdgeName)

**Implementation Complexity:** Medium-High
- Requires scanning ForwardEdges CF by value (inefficient)
- Or adding new reverse index CF
- Better: Change `edge_summary_by_id` return type to include topology

---

### Gap 3: Fragment Queries Return Only Content, Not Association ❌

**Current:** `fragments_by_id(id, timeout) -> Result<Vec<(TimestampMilli, FragmentContent)>>`

**Problem:** You know an ID has fragments, but don't know if it's a node or edge

**Use Case:**
- Fragment viewer: "show fragment X in context"
- Debugging: "this fragment belongs to what?"
- Navigation: "show the entity this fragment describes"

**Proposed Additions:**

```rust
// Option A: Add metadata to return type
pub async fn fragments_by_id_with_type(id: Id, timeout: Duration)
    -> Result<(EntityType, Vec<(TimestampMilli, FragmentContent)>)>

enum EntityType {
    Node(NodeName, NodeSummary),
    Edge(SrcId, DstId, EdgeName, EdgeSummary),
}

// Option B: Separate queries
pub async fn entity_type(id: Id, timeout: Duration)
    -> Result<EntityType>
```

**Schema Support:**
- Must query both Nodes and Edges CFs
- Could be optimized with entity type prefix in ID

**Implementation Complexity:** Low-Medium
- Query Nodes CF first, if not found query Edges CF
- Or maintain entity type registry

---

### Gap 4: No Existence/Type Checking ❌

**Missing:**
```rust
pub async fn entity_exists(id: Id, timeout: Duration)
    -> Result<bool>

pub async fn entity_type(id: Id, timeout: Duration)
    -> Result<Option<EntityType>>

enum EntityType { Node, Edge }
```

**Use Case:**
- Validation: "does this ID exist before querying?"
- Type checking: "is this a node or edge ID?"
- Error handling: better error messages

**Current Workaround:**
- Try `node_by_id()`, if fails try `edge_summary_by_id()`
- Inefficient and relies on error handling for control flow

**Implementation Complexity:** Low
- Same as querying but only check existence
- Can be more efficient than full query

---

### Gap 5: Edge Navigation Returns IDs Only, Not Names ✅ (Partially Filled)

**Current:**
```rust
edges_from_node_by_id(id) -> Vec<(SrcId, EdgeName, DstId)>
edges_to_node_by_id(id) -> Vec<(DstId, EdgeName, SrcId)>
```

**Status:** ✅ These ARE complete for navigation use cases

**However:** If you want node names (not just IDs), you must:
1. Get edge list
2. For each DstId/SrcId, call `node_by_id()` to get name
3. This is N+1 queries

**Potential Enhancement (not critical):**
```rust
pub async fn edges_from_node_by_id_with_names(id: Id, timeout: Duration)
    -> Result<Vec<(SrcId, NodeName, EdgeName, DstId, NodeName)>>
```

**Trade-off:**
- More convenient but couples node and edge queries
- Current approach (separate queries) is more composable
- This is an optimization, not a gap

---

### Gap 6: No Reverse Fragment Lookup ❌

**Missing:** Given a timestamp or content, find the entity

```rust
pub async fn fragments_at_timestamp(ts: TimestampMilli, timeout: Duration)
    -> Result<Vec<(Id, FragmentContent)>>

pub async fn search_fragments(content_query: String, timeout: Duration)
    -> Result<Vec<(Id, TimestampMilli, FragmentContent)>>
```

**Use Case:**
- Timeline view: "show all fragments created at time T"
- Content search: "find entities with fragment containing X"
- Audit: "what was changed at this timestamp?"

**Current Workaround:** None (requires bulk iteration)

**Implementation Complexity:** High
- Requires new indexes
- Search requires full-text search (separate concern, has FullText consumer)

---

## Summary of Gaps

### Critical Gaps (block common use cases)

1. ❌ **`node_by_name(name: String)`** - Cannot lookup nodes by their human-readable names
   - **Impact:** High - most UIs work with names, not IDs
   - **Requires:** New index CF

2. ❌ **`edge_by_id()` returns topology** - Cannot get source/dest/name from edge ID alone
   - **Impact:** Medium - limits edge-centric queries
   - **Requires:** Schema change or reverse index

3. ❌ **`entity_type(id: Id)`** - Cannot determine if ID is node or edge
   - **Impact:** Medium - forces try/catch pattern for polymorphic IDs
   - **Requires:** Type registry or ID prefix convention

### Nice-to-Have Gaps (workarounds exist)

4. ⚠️ **Fragment context** - Fragments don't indicate if they're for node/edge
   - **Impact:** Low - can be inferred from separate queries
   - **Requires:** Joining node/edge queries

5. ⚠️ **`entity_exists(id: Id)`** - No efficient existence check
   - **Impact:** Low - can attempt query and handle error
   - **Requires:** Lightweight query variant

6. ⚠️ **Edge navigation with names** - Must do N+1 queries to get destination node names
   - **Impact:** Low - This is a performance optimization, not a functional gap
   - **Requires:** Denormalization or join logic

### Out of Scope (requires different API pattern)

7. ⛔ **Bulk iteration** - List all nodes/edges/fragments
   - See separate gap analysis document
   - Requires different API design (scanner/iterator)

8. ⛔ **Reverse fragment lookup** - Find entities by timestamp or content
   - Specialized search use case
   - Handled by FullText consumer for content search

## Recommended Additions for Completeness

### High Priority (Essential for ID-agnostic applications)

```rust
impl Reader {
    /// Lookup node by name (requires new index)
    pub async fn node_by_name(&self, name: String, timeout: Duration)
        -> Result<Option<(Id, NodeSummary)>>

    /// Get full edge details including topology
    pub async fn edge_by_id(&self, id: Id, timeout: Duration)
        -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>

    /// Check entity type
    pub async fn entity_type(&self, id: Id, timeout: Duration)
        -> Result<Option<EntityType>>
}

pub enum EntityType {
    Node { name: NodeName },
    Edge { source: Id, dest: Id, name: EdgeName },
}
```

### Medium Priority (Convenience and consistency)

```rust
impl Reader {
    /// Quick existence check
    pub async fn entity_exists(&self, id: Id, timeout: Duration)
        -> Result<bool>

    /// Get entity with fragments in one query
    pub async fn entity_with_fragments(&self, id: Id, timeout: Duration)
        -> Result<Entity>
}

pub enum Entity {
    Node {
        id: Id,
        name: NodeName,
        summary: NodeSummary,
        fragments: Vec<(TimestampMilli, FragmentContent)>,
    },
    Edge {
        id: Id,
        source: Id,
        dest: Id,
        name: EdgeName,
        summary: EdgeSummary,
        fragments: Vec<(TimestampMilli, FragmentContent)>,
    },
}
```

### Low Priority (Optimizations)

```rust
impl Reader {
    /// Get edges with resolved node names (denormalized view)
    pub async fn edges_from_node_with_names(&self, id: Id, timeout: Duration)
        -> Result<Vec<EdgeWithNames>>
}

pub struct EdgeWithNames {
    pub edge_id: Id,
    pub source_id: SrcId,
    pub source_name: NodeName,
    pub edge_name: EdgeName,
    pub dest_id: DstId,
    pub dest_name: NodeName,
}
```

## Implementation Roadmap

### Phase 1: Fix Critical Gaps
1. Implement `node_by_name()` with new NodesByName index CF
   - Update mutation pipeline to maintain index
   - Add query implementation
   - Tests for consistency

2. Change `edge_summary_by_id()` → `edge_by_id()` with topology
   - Breaking change, but better API
   - Or keep both for compatibility

3. Add `entity_type()` helper
   - Simple: try node query, then edge query
   - Or add entity type marker in schema

### Phase 2: Convenience Methods
1. Add `entity_exists()`
2. Add `entity_with_fragments()` for common join pattern

### Phase 3: Performance Optimizations
1. Add denormalized edge views with names (if benchmarks show need)
2. Consider caching layer for hot queries

## Conclusion

**No, `node_by_name` is not the only gap.** For a complete point-query Reader API, you need:

1. ✅ **Critical:** `node_by_name()` - Most important missing piece
2. ✅ **Critical:** `edge_by_id()` returning full topology - Cannot query edges properly without this
3. ✅ **Critical:** `entity_type()` - Need to know what an ID represents
4. ⚠️ **Nice-to-have:** `entity_exists()`, `entity_with_fragments()` - Convenience wrappers
5. ⛔ **Out of scope:** Bulk iteration, search - Different API pattern needed

The first three gaps (#1-3) are **fundamental** and should be addressed for API completeness. The others are quality-of-life improvements.
