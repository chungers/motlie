# HNSW2: Hannoy-Inspired Design for RocksDB

This document explores adapting [hannoy](https://github.com/nnethercott/hannoy)'s LMDB-backed HNSW design to RocksDB, focusing on compressed bitmap edge storage, zero-copy patterns, and efficient online updates.

## Background: Hannoy/Arroy Architecture

### Sources

- [From trees to graphs: speeding up vector search 10x with Hannoy](https://blog.kerollmops.com/from-trees-to-graphs-speeding-up-vector-search-10x-with-hannoy)
- [How Meilisearch Updates a Millions Vector Embeddings Database in Under a Minute](https://www.meilisearch.com/blog/how-meilisearch-updates-a-millions-vector-embeddings-database-in-under-a-minute/)
- [hannoy GitHub](https://github.com/nnethercott/hannoy)
- [arroy GitHub](https://github.com/meilisearch/arroy)

### Key Design Choices in Hannoy

| Aspect | Hannoy Design | Benefit |
|--------|---------------|---------|
| **Storage Backend** | LMDB (memory-mapped) | OS-managed caching, zero-copy reads |
| **Edge Storage** | Roaring bitmaps | ~200 bytes/vector vs ~12KB for explicit edges |
| **Concurrent Reads** | LMDB MVCC | Non-blocking readers |
| **Online Updates** | Incremental re-indexing | <1% vectors touched per update |
| **ID Management** | ConcurrentNodeIds with bitmap | Lock-free ID generation |
| **Multi-index** | u16 index prefix | Multiple indexes per DB |

### Hannoy Storage Schema (Inferred)

```
LMDB Database
├── vectors/{item_id}     → f32[dim] binary
├── edges/{node_id}       → RoaringBitmap (compressed neighbor IDs)
├── layers/{node_id}      → u8 (max layer for this node)
├── meta/entry_point      → item_id
├── meta/max_level        → u8
└── updated/{item_id}     → () (marker for incremental indexing)
```

### Why Roaring Bitmaps for Edges?

Traditional HNSW stores edges as explicit lists:
```
Node 42 edges: [17, 89, 234, 567, 890, 1023, ...]
Storage: 32 edges × 8 bytes = 256 bytes (just IDs)
With metadata: 32 edges × 24 bytes = 768 bytes
```

Roaring bitmap approach:
```
Node 42 edges: RoaringBitmap { 17, 89, 234, 567, 890, 1023, ... }
Storage: ~50-200 bytes (compressed, depends on density)
```

**Benefits**:
- 4-10× more compact for typical edge sets
- O(1) membership test: `bitmap.contains(neighbor_id)`
- Efficient set operations: `bitmap.union(other)`, `bitmap.intersection(filter)`
- Naturally supports filtered search (intersect with filter bitmap)

**Trade-off**:
- No distance stored with edges (must recompute or store separately)
- Iteration order is by ID, not by distance

---

## LMDB vs RocksDB: Key Differences

| Feature | LMDB | RocksDB | Implication |
|---------|------|---------|-------------|
| **Memory Model** | Memory-mapped | Block cache | RocksDB needs explicit caching |
| **Concurrency** | MVCC, single writer | Multi-writer with locking | RocksDB more flexible |
| **Zero-copy** | Yes (mmap) | No (copy to buffer) | Must minimize copies |
| **Compression** | None built-in | ZSTD, LZ4, etc. | RocksDB can compress vectors |
| **Column Families** | Single namespace | Multiple CFs | RocksDB can isolate data types |
| **Merge Operators** | None | Custom merge functions | RocksDB can update bitmaps atomically |
| **Transactions** | Full ACID | WriteBatch, OptimisticTxn | Different consistency model |

### Key Challenge: No Memory Mapping

LMDB's memory mapping provides:
1. OS-managed caching (LRU eviction handled by kernel)
2. Zero-copy reads (pointer directly into mapped file)
3. Automatic prefetching via `madvise`

RocksDB alternative:
1. Block cache with configurable size
2. Pinnable slices (avoid copy if data in cache)
3. ReadOptions with `fill_cache`, `verify_checksums`

---

## Proposed RocksDB Schema: HNSW2

### Column Families

```
┌─────────────────────────────────────────────────────────────────┐
│                     RocksDB Database                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: vectors                                               │   │
│  │ Key:   item_id (u32, 4 bytes)                             │   │
│  │ Value: f32[dim] binary (e.g., 4KB for 1024-dim)           │   │
│  │ Access: Point lookup, MultiGet, iterator scan             │   │
│  │ Options: ZSTD compression, 64KB blocks, bloom filter      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: edges                                                 │   │
│  │ Key:   node_id | layer (5 bytes: u32 + u8)                │   │
│  │ Value: RoaringBitmap serialized (variable, ~50-200 bytes) │   │
│  │ Access: Point lookup, merge for updates                   │   │
│  │ Options: No compression (already compact), merge operator │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: distances (OPTIONAL - for weighted search)            │   │
│  │ Key:   node_id | neighbor_id (8 bytes: u32 + u32)         │   │
│  │ Value: f32 distance (4 bytes)                             │   │
│  │ Access: Point lookup, batch get                           │   │
│  │ Options: LZ4 compression, small blocks                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: node_meta                                             │   │
│  │ Key:   node_id (4 bytes)                                  │   │
│  │ Value: { max_layer: u8, flags: u8 }                       │   │
│  │ Access: Point lookup                                      │   │
│  │ Options: In-memory (tiny), no compression                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: graph_meta                                            │   │
│  │ Key:   string ("entry_point", "max_level", "count", etc.) │   │
│  │ Value: Respective serialized values                       │   │
│  │ Access: Point lookup (rarely changes)                     │   │
│  │ Options: In-memory, no compression                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: pending_updates                                       │   │
│  │ Key:   node_id (4 bytes)                                  │   │
│  │ Value: UpdateType enum (1 byte: Insert/Delete/Modified)   │   │
│  │ Access: Iterator scan, delete after processing            │   │
│  │ Options: No compression, frequent writes                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Item ID Design

Use **u32 item IDs** instead of UUIDs:
- 4 bytes vs 16 bytes per reference
- Roaring bitmaps require integer IDs
- 4 billion vectors capacity (sufficient for most use cases)

ID allocation:
```rust
struct IdAllocator {
    next_id: AtomicU32,
    free_ids: RoaringBitmap,  // Reusable IDs from deletions
}

impl IdAllocator {
    fn allocate(&self) -> u32 {
        // First try to reuse a freed ID
        if let Some(id) = self.free_ids.select(0) {
            self.free_ids.remove(id);
            return id;
        }
        // Otherwise allocate fresh
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn free(&self, id: u32) {
        self.free_ids.insert(id);
    }
}
```

---

## Roaring Bitmap Merge Operator

RocksDB merge operators enable atomic bitmap updates without read-modify-write:

```rust
use roaring::RoaringBitmap;

enum EdgeOperation {
    Add(u32),           // Add single neighbor
    AddBatch(Vec<u32>), // Add multiple neighbors
    Remove(u32),        // Remove single neighbor
    Replace(Vec<u32>),  // Replace entire edge list
}

fn edge_merge_operator(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &[&[u8]],
) -> Option<Vec<u8>> {
    // Start with existing bitmap or empty
    let mut bitmap = existing
        .map(|b| RoaringBitmap::deserialize_from(b).unwrap())
        .unwrap_or_default();

    // Apply each operation in order
    for operand in operands {
        let op: EdgeOperation = bincode::deserialize(operand).unwrap();
        match op {
            EdgeOperation::Add(id) => { bitmap.insert(id); }
            EdgeOperation::AddBatch(ids) => {
                for id in ids { bitmap.insert(id); }
            }
            EdgeOperation::Remove(id) => { bitmap.remove(id); }
            EdgeOperation::Replace(ids) => {
                bitmap.clear();
                for id in ids { bitmap.insert(id); }
            }
        }
    }

    // Serialize result
    let mut buf = Vec::new();
    bitmap.serialize_into(&mut buf).unwrap();
    Some(buf)
}
```

**Usage**:
```rust
// Atomic add neighbor without reading first
db.merge_cf(edges_cf, &key, EdgeOperation::Add(neighbor_id).serialize())?;

// Atomic batch add
db.merge_cf(edges_cf, &key, EdgeOperation::AddBatch(vec![1, 2, 3]).serialize())?;
```

**Benefits**:
- No read-modify-write cycle
- Works with WriteBatch
- Concurrent writers don't conflict
- Crash-safe (logged in WAL)

---

## Distance Storage Options

Roaring bitmaps don't store distances with edges. Three options:

### Option A: Recompute Distances (Hannoy's Approach)

```rust
async fn get_neighbors_with_distances(
    db: &DB,
    node_id: u32,
    layer: u8,
    query: &[f32],
) -> Vec<(f32, u32)> {
    // Get neighbor IDs from bitmap
    let key = encode_key(node_id, layer);
    let bitmap: RoaringBitmap = db.get_cf(edges_cf, key)?
        .map(|b| RoaringBitmap::deserialize_from(&b).unwrap())
        .unwrap_or_default();

    // Batch fetch vectors
    let neighbor_ids: Vec<u32> = bitmap.iter().collect();
    let keys: Vec<_> = neighbor_ids.iter().map(|id| id.to_le_bytes()).collect();
    let vectors = db.multi_get_cf(vectors_cf, &keys);

    // Compute distances
    let mut results: Vec<(f32, u32)> = neighbor_ids.iter()
        .zip(vectors.iter())
        .filter_map(|(id, v)| {
            v.as_ref().ok().flatten().map(|vec| {
                let dist = euclidean_distance(query, &deserialize_vector(vec));
                (dist, *id)
            })
        })
        .collect();

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results
}
```

**Pros**: Minimal storage, always fresh distances
**Cons**: Compute overhead during search

### Option B: Separate Distance Column Family

Store distances in a separate CF:

```rust
// Key: node_id | neighbor_id (8 bytes)
// Value: f32 distance (4 bytes)

fn store_edge_with_distance(
    batch: &mut WriteBatch,
    node_id: u32,
    neighbor_id: u32,
    distance: f32,
) {
    // Add to bitmap
    batch.merge_cf(edges_cf,
        &encode_edge_key(node_id, layer),
        EdgeOperation::Add(neighbor_id).serialize());

    // Store distance
    batch.put_cf(distances_cf,
        &encode_distance_key(node_id, neighbor_id),
        &distance.to_le_bytes());
}
```

**Pros**: Precomputed distances, fast lookup
**Cons**: 2× storage for edges, must keep in sync

### Option C: Interleaved Bitmap+Distances

Custom format storing distances alongside bitmap:

```rust
struct EdgeList {
    neighbors: RoaringBitmap,
    distances: Vec<f32>,  // Same order as bitmap iteration
}

impl EdgeList {
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.neighbors.serialize_into(&mut buf).unwrap();
        buf.extend(bytemuck::cast_slice(&self.distances));
        buf
    }

    fn deserialize(data: &[u8]) -> Self {
        let bitmap_end = RoaringBitmap::serialized_size_in_bytes(data);
        let neighbors = RoaringBitmap::deserialize_from(&data[..bitmap_end]).unwrap();
        let distances: Vec<f32> = bytemuck::cast_slice(&data[bitmap_end..]).to_vec();
        Self { neighbors, distances }
    }
}
```

**Pros**: Single read for neighbors + distances
**Cons**: More complex updates, larger values

### Recommendation

**Use Option A (recompute)** for initial implementation:
- Simplest to implement
- Matches hannoy's approach
- Distance computation is fast with SIMD
- Can add Option B later for optimization

---

## Online Updates: Incremental Re-indexing

Hannoy's key innovation: update <1% of vectors per insert/delete.

### Update Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Insert Vector                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Allocate item_id from IdAllocator                           │
│  2. Store vector in vectors CF                                  │
│  3. Determine layer level (random, exponential)                 │
│  4. Greedy search to find neighbors at each layer               │
│  5. Add edges via merge operator (bitmap updates)               │
│  6. Add reverse edges via merge operator                        │
│  7. Mark affected nodes in pending_updates CF                   │
│  8. Return immediately (no inline pruning)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Background Maintenance Thread                   │
├─────────────────────────────────────────────────────────────────┤
│  Loop every N inserts or T seconds:                             │
│    1. Scan pending_updates CF                                   │
│    2. For each affected node:                                   │
│       a. Check if over M_max connections                        │
│       b. If yes, prune using RNG heuristic                      │
│       c. Update edges via merge operator                        │
│    3. Clear processed entries from pending_updates              │
│    4. Update graph_meta (entry_point if needed)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Delete Vector

```rust
async fn delete_vector(db: &DB, item_id: u32) -> Result<()> {
    let batch = WriteBatch::new();

    // 1. Get max layer for this node
    let max_layer = db.get_cf(node_meta_cf, &item_id.to_le_bytes())?
        .map(|b| b[0])
        .unwrap_or(0);

    // 2. Remove from each layer's edge lists
    for layer in 0..=max_layer {
        // Get current neighbors
        let key = encode_edge_key(item_id, layer);
        if let Some(bitmap_data) = db.get_cf(edges_cf, &key)? {
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_data)?;

            // Remove reverse edges from each neighbor
            for neighbor_id in bitmap.iter() {
                batch.merge_cf(edges_cf,
                    &encode_edge_key(neighbor_id, layer),
                    EdgeOperation::Remove(item_id).serialize());

                // Mark neighbor for potential orphan repair
                batch.put_cf(pending_updates_cf,
                    &neighbor_id.to_le_bytes(),
                    &[UpdateType::Modified as u8]);
            }
        }

        // Delete node's edge list
        batch.delete_cf(edges_cf, &key);
    }

    // 3. Delete vector data
    batch.delete_cf(vectors_cf, &item_id.to_le_bytes());

    // 4. Delete node metadata
    batch.delete_cf(node_meta_cf, &item_id.to_le_bytes());

    // 5. Free ID for reuse
    id_allocator.free(item_id);

    // 6. Atomic commit
    db.write(batch)?;

    Ok(())
}
```

### Orphan Repair (DiskANN-inspired)

When a node loses connections due to neighbor deletion:

```rust
async fn repair_orphans(db: &DB, batch_size: usize) -> Result<()> {
    // Scan pending_updates for Modified nodes
    let mut to_repair = Vec::new();
    let iter = db.iterator_cf(pending_updates_cf, IteratorMode::Start);

    for item in iter.take(batch_size) {
        let (key, value) = item?;
        if value[0] == UpdateType::Modified as u8 {
            to_repair.push(u32::from_le_bytes(key.try_into().unwrap()));
        }
    }

    // Batch process repairs
    let batch = WriteBatch::new();

    for node_id in to_repair {
        // Check connection count at layer 0
        let key = encode_edge_key(node_id, 0);
        let bitmap = db.get_cf(edges_cf, &key)?
            .map(|b| RoaringBitmap::deserialize_from(&b).unwrap())
            .unwrap_or_default();

        let min_connections = M / 2;  // Half of target degree

        if bitmap.len() < min_connections as u64 {
            // Find new neighbors via greedy search
            let vector = get_vector(db, node_id)?;
            let candidates = greedy_search(db, &vector, M * 2)?;

            // Add new edges
            for (_, neighbor_id) in candidates.iter().take(M) {
                batch.merge_cf(edges_cf, &key,
                    EdgeOperation::Add(*neighbor_id).serialize());
                batch.merge_cf(edges_cf,
                    &encode_edge_key(*neighbor_id, 0),
                    EdgeOperation::Add(node_id).serialize());
            }
        }

        // Clear pending update
        batch.delete_cf(pending_updates_cf, &node_id.to_le_bytes());
    }

    db.write(batch)?;
    Ok(())
}
```

---

## Comparison with Pure-RocksDB Design (README)

| Aspect | Pure-RocksDB (README) | HNSW2 (Hannoy-inspired) |
|--------|----------------------|-------------------------|
| **Edge Storage** | Packed list per node | Roaring bitmap per node |
| **Edge Size** | ~640 bytes (32×20) | ~100 bytes (compressed) |
| **Membership Test** | Linear scan | O(1) bitmap contains |
| **Distance Storage** | In edge list | Separate or recompute |
| **Set Operations** | Manual | Native bitmap ops |
| **Filtered Search** | Post-filter | Bitmap intersection |
| **Update Pattern** | Replace entire list | Merge operator |
| **Concurrency** | WriteBatch | Merge (no conflicts) |

### When to Use Which

**Use Pure-RocksDB when:**
- Need sorted neighbors by distance
- Edge count is small (<16)
- No filtered search needed
- Simpler implementation preferred

**Use HNSW2 (Bitmap) when:**
- Large edge counts (32-64)
- Filtered search is common
- Concurrent updates important
- Storage efficiency matters

---

## Filtered Search with Bitmaps

Key advantage of roaring bitmaps: native intersection with filter sets.

```rust
async fn filtered_search(
    db: &DB,
    query: &[f32],
    filter: &RoaringBitmap,  // Only search within these IDs
    k: usize,
    ef: usize,
) -> Result<Vec<(f32, u32)>> {
    let entry_point = get_entry_point(db)?;

    // If entry point is filtered out, find valid starting point
    let start = if filter.contains(entry_point) {
        entry_point
    } else {
        // Find any valid ID in filter that exists in graph
        filter.iter()
            .find(|id| db.get_cf(node_meta_cf, &id.to_le_bytes()).is_ok())
            .ok_or(Error::NoValidStartPoint)?
    };

    let mut visited = RoaringBitmap::new();
    let mut candidates = BinaryHeap::new();
    let mut results = Vec::new();

    // Initialize
    let start_vec = get_vector(db, start)?;
    let start_dist = euclidean_distance(query, &start_vec);
    candidates.push(Reverse((OrderedFloat(start_dist), start)));
    visited.insert(start);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= ef && dist.0 > results.last().unwrap().0 {
            break;
        }

        results.push((dist.0, node));

        // Get neighbors
        let neighbors = get_edge_bitmap(db, node, 0)?;

        // KEY: Intersect with filter BEFORE fetching vectors
        let valid_neighbors = neighbors.and(&filter);
        let unvisited: Vec<u32> = valid_neighbors.iter()
            .filter(|id| !visited.contains(*id))
            .collect();

        for &id in &unvisited {
            visited.insert(id);
        }

        // Batch fetch only filtered, unvisited vectors
        let vectors = multi_get_vectors(db, &unvisited)?;

        for (id, vector) in unvisited.iter().zip(vectors.iter()) {
            if let Some(vec) = vector {
                let dist = euclidean_distance(query, vec);
                candidates.push(Reverse((OrderedFloat(dist), *id)));
            }
        }
    }

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results.truncate(k);
    Ok(results)
}
```

**Performance**: Only vectors matching filter are fetched, dramatically reducing I/O for selective filters.

---

## Performance Estimates

### Storage Comparison (1M vectors, 1024-dim, M=32)

| Component | Pure-RocksDB | HNSW2 (Bitmap) | Savings |
|-----------|--------------|----------------|---------|
| Vectors | 4GB | 4GB | - |
| Edges (per node) | 640 bytes | 100 bytes | **84%** |
| Edges (total) | 640MB | 100MB | **540MB** |
| **Total** | **4.64GB** | **4.1GB** | **12%** |

### Operation Costs

| Operation | Pure-RocksDB | HNSW2 (Bitmap) |
|-----------|--------------|----------------|
| Check neighbor exists | O(n) scan | O(1) bitmap |
| Add neighbor | Replace list | Merge append |
| Remove neighbor | Replace list | Merge remove |
| Get all neighbors | 1 read | 1 read |
| Intersect with filter | O(n) scan | O(min(m,n)) |
| Concurrent updates | Lock or conflict | Lock-free merge |

### Expected Throughput

| Metric | Current | HNSW2 Estimate |
|--------|---------|----------------|
| Insert | 30-68/sec | **5,000-10,000/sec** |
| Search (unfiltered) | 45-108 QPS | **500-1,000 QPS** |
| Search (10% filter) | N/A | **800-1,500 QPS** |
| Memory (10M vectors) | 40GB | **<1GB** (block cache) |

---

## Implementation Roadmap

### Phase 1: Core Schema (1 week)
- [ ] Implement u32 ID allocator with roaring reuse
- [ ] Create column families (vectors, edges, node_meta, graph_meta)
- [ ] Implement roaring bitmap edge merge operator
- [ ] Basic get/put operations

### Phase 2: HNSW Construction (1-2 weeks)
- [ ] Port HNSW insert with bitmap edges
- [ ] Implement greedy search with bitmap neighbors
- [ ] Add WriteBatch for atomic operations
- [ ] Remove 10ms sleep with WriteBatchWithIndex

### Phase 3: Online Updates (1 week)
- [ ] Implement pending_updates tracking
- [ ] Background pruning thread
- [ ] Orphan repair logic
- [ ] Delete vector support

### Phase 4: Optimizations (1 week)
- [ ] MultiGet for batch vector fetching
- [ ] Filtered search with bitmap intersection
- [ ] Block cache tuning
- [ ] SIMD distance computation

### Phase 5: Benchmarking (1 week)
- [ ] Compare with current implementation
- [ ] Test at 100K, 1M, 10M scales
- [ ] Measure filtered search performance
- [ ] Profile and iterate

---

## Open Questions

1. **Distance caching**: Should we store distances separately or always recompute?
   - Recompute is simpler but adds CPU overhead
   - Separate CF adds storage but faster search

2. **Layer-specific bitmaps**: One bitmap per layer, or combined?
   - Per-layer is cleaner but more keys
   - Combined needs layer encoding in value

3. **Entry point updates**: How to handle when entry point is deleted?
   - Pick random high-level node?
   - Background recomputation?

4. **Filtered search starting point**: What if filter excludes entry point?
   - Linear scan of filter?
   - Secondary index by layer?

5. **Memory budget for block cache**: How much RAM to allocate?
   - Too small: excessive disk reads
   - Too large: competes with application

---

## References

- [hannoy](https://github.com/nnethercott/hannoy) - Production KV-backed HNSW
- [arroy](https://github.com/meilisearch/arroy) - LMDB-based ANN (tree-based predecessor)
- [roaring-rs](https://github.com/RoaringBitmap/roaring-rs) - Rust roaring bitmap implementation
- [heed](https://github.com/meilisearch/heed) - Safe LMDB wrapper for Rust
- [DiskANN](https://github.com/microsoft/DiskANN) - Microsoft's disk-based ANN
- [From trees to graphs: Hannoy blog post](https://blog.kerollmops.com/from-trees-to-graphs-speeding-up-vector-search-10x-with-hannoy)
