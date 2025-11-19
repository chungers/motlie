# `edge_by_id()` - Detailed Explanation

## TL;DR

`edge_by_id()` is a **proposed** (currently missing) Reader API method that would return complete edge information including its topology (source, destination, edge name) when given an edge ID.

**Current state:** The Reader API only has `edge_summary_by_id()` which returns just the `EdgeSummary` content, **NOT** the edge's structural information.

**The Problem:** You cannot answer "What does this edge connect?" given only an edge ID.

---

## Current Implementation: `edge_summary_by_id()`

### What Exists Now

```rust
// In reader.rs line 54
pub async fn edge_summary_by_id(&self, id: Id, timeout: Duration)
    -> Result<EdgeSummary>
```

**What it returns:**
- `EdgeSummary` - Just the content/summary of the edge (markdown text)

**What it does NOT return:**
- Source node ID
- Destination node ID
- Edge name

### Example Usage (Current)

```rust
let edge_id = Id::new(); // You have an edge ID from somewhere

// Query the edge
let summary = reader.edge_summary_by_id(edge_id, timeout).await?;

// You get:
println!("Edge content: {}", summary.content()?);
// Output: "<!-- id=... -->\n# edge_name\n# Summary\n"

// You CANNOT get:
// - What is the source node?  ❌ Unknown
// - What is the dest node?    ❌ Unknown
// - What is the edge name?    ❌ Embedded in markdown, must parse
```

---

## The Database Schema for Edges

To understand the problem, we need to see how edges are stored:

### Three Column Families Store Edge Data

#### 1. **Edges CF** - Main edge storage (keyed by edge ID)

```rust
// Key
struct EdgeCfKey(Id)  // Just the edge ID

// Value
struct EdgeCfValue(EdgeSummary)  // Just the content summary
```

**Schema:**
```
edges CF:
  Key: edge_id (16 bytes)
  Value: EdgeSummary(DataUrl)
```

**What's stored:**
- ✅ Edge content/summary
- ❌ Source node ID
- ❌ Destination node ID
- ❌ Edge name

**This is what `edge_summary_by_id()` queries!**

---

#### 2. **ForwardEdges CF** - Source → Destination index

```rust
// Key (composite)
struct ForwardEdgeCfKey(
    EdgeSourceId,      // Source node ID
    EdgeDestinationId, // Dest node ID
    EdgeName,          // Edge name
)

// Value
struct ForwardEdgeCfValue(Id)  // Points to edge ID
```

**Schema:**
```
forward_edges CF:
  Key: (source_id, dest_id, edge_name)
  Value: edge_id
```

**Purpose:**
- Enables query: "Given source and destination nodes and edge name, what's the edge ID?"
- This is what `edge_by_src_dst_name()` uses!
- Also enables: "What edges go FROM this node?" (scan with prefix)

---

#### 3. **ReverseEdges CF** - Destination → Source index

```rust
// Key (composite)
struct ReverseEdgeCfKey(
    EdgeDestinationId, // Dest node ID
    EdgeSourceId,      // Source node ID
    EdgeName,          // Edge name
)

// Value
struct ReverseEdgeCfValue(Id)  // Points to edge ID
```

**Schema:**
```
reverse_edges CF:
  Key: (dest_id, source_id, edge_name)
  Value: edge_id
```

**Purpose:**
- Enables: "What edges come TO this node?" (scan with prefix)
- This is what `edges_to_node_by_id()` uses!

---

## The Problem: Topology is Not Accessible by Edge ID Alone

### Information Distribution

Given an edge with `edge_id = 12345`:

| Information | Where It's Stored | Accessible by Edge ID? |
|------------|-------------------|----------------------|
| Edge content/summary | `edges[edge_id]` | ✅ YES - via `edge_summary_by_id()` |
| Source node ID | `forward_edges` key, `reverse_edges` key | ❌ NO - ID is in the VALUE, not the KEY |
| Dest node ID | `forward_edges` key, `reverse_edges` key | ❌ NO - ID is in the VALUE, not the KEY |
| Edge name | `forward_edges` key, `reverse_edges` key | ❌ NO - name is in the VALUE, not the KEY |

### The Asymmetry

```rust
// ✅ You CAN do this (topology → edge ID):
let (edge_id, summary) = reader.edge_by_src_dst_name(
    source_id,  // Known
    dest_id,    // Known
    "follows",  // Known
    timeout
).await?;

// ❌ You CANNOT do the reverse (edge ID → topology):
let edge_id = Id::from_str("...")?;
let (source_id, dest_id, edge_name, summary) = EdgeById::new(edge_id, None)
    .run(&reader, timeout)
    .await?;
//                                               ^^^^^^^^^^^^^^
//                                               DOES NOT EXIST!
```

---

## Real-World Use Cases Where This Gaps Matters

### Use Case 1: Edge Inspector/Debugger

```rust
// User clicks on an edge in a graph visualization
let edge_id = user_selected_edge_id;

// Want to show:
// "Edge 'follows' from Alice to Bob"

// Current API - CANNOT DO THIS:
let summary = reader.edge_summary_by_id(edge_id, timeout).await?;
// Only get: content summary
// Missing: who's Alice? who's Bob? what's the relationship name?

// Proposed API - CAN DO THIS:
let (src_id, dst_id, edge_name, summary) = EdgeById::new(edge_id, None)
    .run(&reader, timeout)
    .await?;
let (src_name, _) = NodeById::new(src_id, None)
    .run(&reader, timeout)
    .await?;
let (dst_name, _) = NodeById::new(dst_id, None)
    .run(&reader, timeout)
    .await?;
println!("Edge '{}' from {} to {}", edge_name.0, src_name, dst_name);
```

### Use Case 2: Fragment Navigation

```rust
// You have fragments for an edge
let fragments = reader.fragments_by_id(edge_id, timeout).await?;

// Want to show context: "These fragments describe the 'likes' edge from Alice to Bob"

// Current API - CANNOT get context:
// You know it's an edge (not a node), but that's it

// Proposed API - CAN get context:
let (src_id, dst_id, edge_name, _) = EdgeById::new(edge_id, None)
    .run(&reader, timeout)
    .await?;
// Now can build contextual UI
```

### Use Case 3: Edge-Centric Workflow

```rust
// Application stores edge IDs in its own data structures
struct Recommendation {
    edge_id: Id,  // Reference to database edge
    score: f64,
}

// Later, need to display recommendations with full context
for rec in recommendations {
    // Current API - INCOMPLETE:
    let summary = reader.edge_summary_by_id(rec.edge_id, timeout).await?;
    // Can show content but not "who → who" or relationship type

    // Proposed API - COMPLETE:
    let (src, dst, name, summary) = EdgeById::new(rec.edge_id, None)
        .run(&reader, timeout)
        .await?;
    println!("{} -{:?}-> {} (score: {})", src, name, dst, rec.score);
}
```

### Use Case 4: Consistency Checking

```rust
// Verify edge data integrity
let edge_id = Id::new();

// Want to check: "Does this edge's source and dest nodes actually exist?"

// Current API - Cannot do this:
let summary = reader.edge_summary_by_id(edge_id, timeout).await?;
// Don't know what nodes to check!

// Proposed API - Can validate:
let (src_id, dst_id, edge_name, summary) = EdgeById::new(edge_id, None)
    .run(&reader, timeout)
    .await?;
let src_exists = NodeById::new(src_id, None)
    .run(&reader, timeout)
    .await
    .is_ok();
let dst_exists = NodeById::new(dst_id, None)
    .run(&reader, timeout)
    .await
    .is_ok();
if !src_exists || !dst_exists {
    eprintln!("Edge {} has dangling references!", edge_id);
}
```

---

## Proposed Solution: `edge_by_id()`

### API Signature

```rust
impl Reader {
    /// Query a full edge by its ID, including topology
    ///
    /// Returns the complete edge information:
    /// - source_id: The ID of the source node
    /// - dest_id: The ID of the destination node
    /// - edge_name: The name/type of the edge
    /// - summary: The edge's content summary
    ///
    /// # Example
    /// ```rust
    /// let (src, dst, name, summary) = EdgeById::new(edge_id, None)
    ///     .run(&reader, timeout)
    ///     .await?;
    /// println!("Edge '{}' connects {} to {}", name.0, src, dst);
    /// ```
    pub async fn edge_by_id(
        &self,
        id: Id,
        timeout: Duration
    ) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>
}
```

### Return Type Breakdown

```rust
(
    SrcId,        // Type alias for Id - the source node ID
    DstId,        // Type alias for Id - the destination node ID
    EdgeName,     // Struct wrapping String - the edge name/type
    EdgeSummary   // The edge content (markdown)
)
```

---

## Implementation Approaches

### Approach 1: Scan ForwardEdges CF (Inefficient)

```rust
async fn edge_by_id(&self, id: Id) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
    // 1. Get edge summary from edges CF
    let summary = self.edge_summary_by_id(id, timeout).await?;

    // 2. Scan forward_edges CF to find entry with this edge_id as VALUE
    let forward_edges_cf = db.cf_handle("forward_edges")?;
    for (key_bytes, value_bytes) in db.iterator_cf(forward_edges_cf, ...) {
        let edge_id: Id = deserialize_value(&value_bytes)?;
        if edge_id == id {
            // Found it! Deserialize the key to get topology
            let (src, dst, name) = deserialize_forward_edge_key(&key_bytes)?;
            return Ok((src, dst, name, summary));
        }
    }
    Err(anyhow!("Edge topology not found"))
}
```

**Problems:**
- ❌ O(n) scan of entire forward_edges CF
- ❌ Very slow for large databases
- ❌ Defeats the purpose of indexed storage

---

### Approach 2: Add Reverse Index CF (Recommended)

Create a new column family that maps edge ID → topology:

```rust
// New CF: EdgeTopology
struct EdgeTopologyCfKey(Id);  // Edge ID

struct EdgeTopologyCfValue(
    EdgeSourceId,      // Source node ID
    EdgeDestinationId, // Dest node ID
    EdgeName,          // Edge name
);

// Schema:
// edge_topology CF:
//   Key: edge_id
//   Value: (source_id, dest_id, edge_name)
```

**Implementation:**

```rust
async fn edge_by_id(&self, id: Id) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
    // 1. Query edge_topology CF for topology (O(1) lookup)
    let topology = self.get_edge_topology(id).await?;

    // 2. Query edges CF for summary (O(1) lookup)
    let summary = self.edge_summary_by_id(id, timeout).await?;

    Ok((topology.source_id, topology.dest_id, topology.name, summary))
}
```

**Benefits:**
- ✅ O(1) lookup - just as fast as current queries
- ✅ Clean separation of concerns
- ✅ No scanning required

**Cost:**
- Additional CF to maintain
- Mutation pipeline must write to 4 CFs instead of 3:
  - `edges` (summary)
  - `forward_edges` (source → dest index)
  - `reverse_edges` (dest → source index)
  - `edge_topology` (id → topology) **NEW**
- Slightly more storage

---

### Approach 3: Denormalize EdgeCfValue (Breaking Change)

Change the schema to store topology in the edges CF:

```rust
// OLD:
struct EdgeCfValue(EdgeSummary);

// NEW:
struct EdgeCfValue {
    source_id: Id,
    dest_id: Id,
    name: String,
    summary: EdgeSummary,
}
```

**Implementation:**

```rust
async fn edge_by_id(&self, id: Id) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
    // Single lookup in edges CF
    let value: EdgeCfValue = self.get_edge_value(id).await?;
    Ok((value.source_id, value.dest_id, EdgeName(value.name), value.summary))
}
```

**Benefits:**
- ✅ Single CF lookup - fastest possible
- ✅ No additional CFs needed
- ✅ Simpler schema

**Costs:**
- ❌ **Breaking change** - existing databases won't work
- ❌ Data duplication (topology stored in 3 places: edges, forward_edges, reverse_edges)
- ❌ Migration required for existing data

---

## Comparison with Similar API: `node_by_id()`

Notice the **symmetry** between nodes and edges:

### Node API (Complete)

```rust
pub async fn node_by_id(&self, id: Id, timeout: Duration)
    -> Result<(NodeName, NodeSummary)>
```

**Given a node ID, you get:**
- ✅ Node name (identifier/label)
- ✅ Node summary (content)

**Schema:**
```
nodes CF:
  Key: node_id
  Value: (NodeName, NodeSummary)  // Both in ONE lookup
```

---

### Edge API (Incomplete)

```rust
// Current - INCOMPLETE
pub async fn edge_summary_by_id(&self, id: Id, timeout: Duration)
    -> Result<EdgeSummary>

// Proposed - COMPLETE
pub async fn edge_by_id(&self, id: Id, timeout: Duration)
    -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>
```

**Given an edge ID, you should get:**
- ❌ Source node ID (topology) - **MISSING**
- ❌ Dest node ID (topology) - **MISSING**
- ❌ Edge name (identifier/label) - **MISSING**
- ✅ Edge summary (content) - current API

**Schema:**
```
edges CF:
  Key: edge_id
  Value: EdgeSummary  // Only content, NO topology!

forward_edges CF:
  Key: (src, dst, name)
  Value: edge_id  // Topology in KEY, not VALUE - wrong direction!
```

---

## The Asymmetry Problem

### Nodes: Lookup Works Both Ways ✅

```rust
// Name → ID (if we add node_by_name):
let (node_id, summary) = reader.node_by_name("Alice", timeout).await?;

// ID → Name (already exists):
let (node_name, summary) = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;
```

**Both directions accessible!**

---

### Edges: Lookup Only Works One Way ❌

```rust
// Topology → ID (exists):
let (edge_id, summary) = reader.edge_by_src_dst_name(
    alice_id,
    bob_id,
    "follows",
    timeout
).await?;

// ID → Topology (MISSING):
let (src_id, dst_id, name, summary) = EdgeById::new(edge_id, None)
    .run(&reader, timeout)
    .await?;
//                                    ^^^^^^^^^^^^^^^^^ DOES NOT EXIST
```

**Only one direction works!**

---

## Recommendation

### Option 1: Add `edge_by_id()` with Reverse Index (Recommended)

**Implementation:**
1. Add new CF: `edge_topology` with mapping `edge_id → (source, dest, name)`
2. Update mutation pipeline to write to this CF when creating edges
3. Add new query method: `edge_by_id()` that reads from both `edges` and `edge_topology`

**Migration:**
- Scan existing `forward_edges` CF to populate `edge_topology` CF
- One-time operation on database upgrade

**API:**
```rust
pub async fn edge_by_id(&self, id: Id, timeout: Duration)
    -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>
```

---

### Option 2: Rename and Keep Current API (Not Recommended)

If the current behavior is intentional:
- Rename `edge_summary_by_id()` to make it clear it's incomplete
- Document that topology requires separate query
- Maybe: `edge_content_by_id()` or `edge_summary_only_by_id()`

**API:**
```rust
// Clearer naming
pub async fn edge_content_by_id(&self, id: Id, timeout: Duration)
    -> Result<EdgeSummary>

// Still missing the topology query!
```

This doesn't solve the problem, just makes it more obvious.

---

## Summary

### What `edge_by_id()` Is

A **proposed** Reader API method that returns complete edge information:

```rust
pub async fn edge_by_id(&self, id: Id, timeout: Duration)
    -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>
```

### Why It's Needed

The current `edge_summary_by_id()` returns only content, not topology. You cannot determine:
- What the edge connects (source/destination)
- The edge's name/type
- How to navigate from this edge to its endpoint nodes

### How to Implement

Add reverse index CF: `edge_topology[edge_id] → (source, dest, name)`

### Impact

**High Priority** - This is a fundamental gap in the API. Edges are first-class entities in a graph database, but the current API treats them as second-class (content-only, no structure).
