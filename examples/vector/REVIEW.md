# Engineering Review: Vector Search Architecture

**Author:** Gemini (AI Assistant)
**Date:** December 24, 2025
**Scope:** Review of `examples/vector/*.md` design documents and benchmarks.

---

## 1. Executive Summary

The proposed evolution from the current Proof-of-Concept (POC) to a Hybrid, Billion-Scale architecture is **sound and well-reasoned**, specifically regarding the transition to **RaBitQ** to satisfy the critical [`DATA-1`](./REQUIREMENTS.md#data-1) (no pre-training) constraint.

The current POC implementation serves its purpose as a baseline but confirms that a naive RocksDB-backed approach (with per-insert flushing) cannot meet the throughput targets (40 insert/s vs 5,000 insert/s target).

**Key Endorsement:** The **[HNSW2](./HNSW2.md) + [RaBitQ](./HYBRID.md)** combination is the most viable path to meeting the 1B scale requirement within the < 64GB RAM constraint while adhering to [`DATA-1`](./REQUIREMENTS.md#data-1).

**Key Risk:** The **ID Mapping (UUID ↔ u32)** required for HNSW2 and bitsets is a significant architectural bottleneck that needs careful design to avoid blowing the memory budget or killing latency.

---

## 2. Requirements Assessment

The requirements defined in [REQUIREMENTS.md](./REQUIREMENTS.md) are ambitious but realistic for a modern vector database.

| Requirement | Feasibility | Comment |
|-------------|-------------|---------|
| [**SCALE-1**](./REQUIREMENTS.md#scale-1) (1B Vectors) | **High** | Achievable only with disk-based storage and compression (RaBitQ). |
| [**SCALE-3**](./REQUIREMENTS.md#scale-3) (< 64GB RAM) | **Medium/High Risk** | Pure vectors + graph fits, but **ID Mapping** and **Bloom Filters** will eat significantly into this budget. |
| [**DATA-1**](./REQUIREMENTS.md#data-1) (No Training) | **Strict** | Correctly disqualifies standard PQ, ScaNN, and cluster-based IVF (unless random projection is used). |
| [**LAT-1**](./REQUIREMENTS.md#lat-1) (< 20ms Search) | **High** | Achievable with HNSW navigation + SIMD distance. |
| [**THR-1**](./REQUIREMENTS.md#thr-1) (5k Insert/s) | **Medium** | Requires moving away from synchronous flushing. The `WriteBatchWithIndex` proposal in HNSW2 is the correct solution. |

---

## 3. Design Evaluation

### 3.1 POC (Phase 1)
*   **Status:** Functional but slow.
*   **Critique:** The reliance on `flush()` for consistency is the correct root cause analysis for the low throughput. The storage schema (JSON strings) is inefficient (60% overhead).
*   **Verdict:** Good baseline for correctness, but strictly a prototype.

### 3.2 HNSW2 (Phase 2) - **Strongly Recommended**
*   **Roaring Bitmaps:** Using Roaring Bitmaps for edge lists is an excellent engineering choice. It reduces storage, improves cache locality, and enables fast set intersection for filtered search.
*   **Merge Operators:** Using RocksDB merge operators for edge updates is crucial for concurrency and reducing read-modify-write amplification.
*   **u32 IDs:** Moving to `u32` internal IDs is mandatory for Roaring Bitmaps and memory efficiency.
*   **Risk:** The translation layer between external UUIDs and internal u32 IDs is a critical path.

### 3.3 IVFPQ (Phase 3) - **Optional / Low Priority**
*   **Critique:** Standard IVF-PQ requires training centroids (k-means), which violates [`DATA-1`](./REQUIREMENTS.md#data-1). While `IVFPQ.md` mentions it as "optional," it introduces a dual-system architecture (RocksDB + GPU Index) that significantly increases operational complexity.
*   **Recommendation:** Deprioritize this phase. Focus resources on making the CPU-based `HNSW2 + RaBitQ` fast enough. Only pursue GPU acceleration if CPU metrics fall short *after* optimization, and even then, ensure the quantization method is training-free.

### 3.4 HYBRID (Phase 4)
*   **Evolution:** The shift from standard PQ to **RaBitQ** (Random Binary Quantization) in the design is the correct move for [`DATA-1`](./REQUIREMENTS.md#data-1) compliance.
*   **Architecture:** The "Navigation (HNSW) -> Compressed (RaBitQ) -> Exact (Disk)" funnel is a standard, proven pattern for disk-based vector search (similar to SPANN/DiskANN).

---

## 4. Performance & Benchmarks Review

The analysis in [PERF.md](./PERF.md) is thorough.

*   **Insert Bottleneck:** The diagnosis of the 10ms sleep/flush loop is correct. The projection that removing this will unlock orders of magnitude throughput is accurate, provided the RocksDB write path (WAL + Memtable) doesn't become the next bottleneck.
*   **Recall:** The 95.3% recall at 1M scale for HNSW is excellent and validates the graph implementation.
*   **Vamana:** The struggles with Vamana recall (requiring `L=200` to reach acceptable levels) aligns with industry experience; Vamana typically requires more expensive build parameters than HNSW for equivalent quality.

---

## 5. Critical Risks & Feasibility Analysis

### 5.1 The ID Mapping Problem ([SCALE-3](./REQUIREMENTS.md#scale-3) Risk)
At 1 Billion vectors, mapping 16-byte UUIDs to 4-byte u32 IDs is expensive.
*   **In-Memory HashMap:** 1B * (16 + 4 + overhead) ≈ 32GB+. This consumes **50% of your entire memory budget**.
*   **RocksDB Lookup:** Adds a disk seek (or block cache hit) to *every* insert and search result.
*   **Mitigation Strategy:**
    *   Use a specialized, memory-efficient compacted trie or perfect hash if IDs are static (they aren't).
    *   Use a heavy RocksDB Block Cache for the `id_mapping` column family and accept the latency penalty.
    *   **Recommendation:** Prototype the ID mapper *now* to verify memory usage. Do not leave this until Phase 4.

### 5.2 RocksDB Write Amplification
HNSW generates many small random writes (edge updates). RocksDB (LSM-Tree) converts these into sequential writes, but the *compaction* process eventually rewrites data multiple times.
*   **Risk:** Sustained insert throughput might degrade as the DB grows and compaction struggles to keep up.
*   **Mitigation:**
    *   Aggressive compaction tuning (Universal Compaction).
    *   Separate RocksDB instances or physical disks for "Vectors" (large blobs) vs "Edges" (small hot updates).

---

## 6. Detailed Technical Specifications

### 6.1 Transaction Model Refinement
*   **Observation:** `motlie_db` already implements a `Transaction` struct ([`libs/db/src/graph/transaction.rs`](../../../libs/db/src/graph/transaction.rs)) that wraps `rocksdb::Transaction`.
*   **Correction:** The HNSW2 design calls for `WriteBatchWithIndex`. This is functionally equivalent to what `rocksdb::Transaction` provides.
*   **Directive:** HNSW2 implementation MUST use the existing `motlie_db::Transaction` API.
    *   This API guarantees **Read-Your-Writes** visibility, which is the core requirement for the greedy search during insertion.
    *   HNSW2 requires **Merge Operators** (for RoaringBitmap updates). The implementation must ensure `motlie_db::Transaction` exposes `merge` operations or add them if missing.

### 6.2 ID Mapper Specification
The translation between external 16-byte UUIDs (`motlie_db::Id`) and internal 4-byte `u32` integers is a critical performance path.

**Requirements:**
1.  **Bi-directional:** `UUID -> u32` (for insert/query) and `u32 -> UUID` (for results).
2.  **Persistence:** Mappings must survive restarts (RocksDB backed).
3.  **Recycling:** Deleted IDs must be reused to keep the `u32` space compact (essential for RoaringBitmap performance).
4.  **Concurrency:** Thread-safe allocation.

**Architecture:**
*   **Column Families:**
    *   `id_map_fwd`: Key=`[UUID]`, Value=`[u32]` (Size: ~20B/entry)
    *   `id_map_rev`: Key=`[u32]`, Value=`[UUID]` (Size: ~20B/entry)
    *   `meta`: Key=`"next_id"`, Value=`[u32]`
    *   `free_list`: Key=`"reusable"`, Value=`RoaringBitmap`
*   **Caching Strategy:**
    *   **Hot Cache:** Small LRU (e.g., 1M entries) for active workload.
    *   **Block Cache:** Rely on RocksDB Block Cache for the tail. Do NOT load 1B IDs into RAM.

**Proposed Interface:**
```rust
pub struct IdMapper {
    db: Arc<DB>,
    next_id: AtomicU32,
    // Optional: Small LRU cache
    cache: Cache<Id, u32>,
}

impl IdMapper {
    /// Get existing u32 or allocate new one atomically
    pub fn get_or_allocate(&self, external_id: Id) -> Result<u32>;
    
    /// Reverse lookup for search results
    pub fn get_external(&self, internal_id: u32) -> Result<Option<Id>>;
    
    /// Mark ID as free for reuse
    pub fn free(&self, internal_id: u32) -> Result<()>;
}
```

### 6.3 Vector Subsystem Architecture
The vector subsystem should mirror the existing `graph` and `fulltext` module structure in `libs/db/src/`.

**Proposed Directory Layout:**
```
libs/db/src/vector/
├── mod.rs             # Module exports
├── config.rs          # VectorConfig
├── schema.rs          # Column Family definitions (Vectors, Edges, IDMap)
├── store.rs           # Low-level RocksDB wrapper (cf handles)
├── mapper.rs          # IdMapper implementation
├── quantization.rs    # RaBitQ codec
├── hnsw.rs            # HNSW graph logic (insert, search)
├── mutation.rs        # Mutation definitions (InsertVector, etc.)
├── query.rs           # Query definitions (NearestNeighbors)
├── writer.rs          # Mutation consumer/writer
└── reader.rs          # Query executor
```

**Integration Points:**
1.  **`libs/db/src/lib.rs`**: Add `vector` module. Update `StorageConfig` to include `VectorConfig`.
2.  **`libs/db/src/storage.rs`**: Initialize vector Column Families in `Storage::ready()`.
3.  **`libs/db/src/mutation.rs`**: Add `Vector(vector::mutation::Mutation)` variant to the unified `Mutation` enum.

---

## 7. Implementation Roadmap

The implementation should proceed in strict order of dependency and risk.

### Phase 1: Foundation & Risk Mitigation (Weeks 1-2)
**Goal:** Validate memory constraints and establish module structure.
1.  **Scaffold Vector Subsystem:** Create `libs/db/src/vector/` structure.
2.  **Implement ID Mapper:** Build the Bi-directional RocksDB-backed mapper with LRU caching.
3.  **Benchmark ID Mapper:** Prove that 1B ID mappings fit within memory/latency budgets (Crucial "Go/No-Go" gate).

### Phase 2: HNSW2 Core Engine (Weeks 3-4)
**Goal:** Solve the throughput bottleneck.
1.  **Merge Operators:** Implement RocksDB Merge Operator for `RoaringBitmap`.
2.  **Core HNSW:** Port insert/search logic to use `u32` IDs and `motlie_db::Transaction`.
3.  **Validation:** Verify [`REC-1`](./REQUIREMENTS.md#rec-1) (Recall) and [`THR-1`](./REQUIREMENTS.md#thr-1) (>5k inserts/s) on mid-sized datasets (10M).

### Phase 3: RaBitQ & Scale (Weeks 5-6)
**Goal:** Solve the storage/memory bottleneck for 1B scale ([`DATA-1`](./REQUIREMENTS.md#data-1)).
1.  **RaBitQ Codec:** Implement random rotation and binarization.
2.  **Storage:** Add `binary_codes` Column Family.
3.  **Search Pipeline:** Implement Hamming distance pre-filter + Exact Re-ranking.
4.  **Validation:** Verify Recall is maintained with binary quantization.

### Phase 4: Productionization (Weeks 7-8)
**Goal:** Operational stability and latency guarantees.
1.  **Async Updater:** Decouple write-path from graph maintenance (optional based on Phase 2 results).
2.  **SIMD:** Optimize distance calculations (AVX2/AVX-512).
3.  **Recovery:** Stress test crash recovery and WAL replay.

---

## 8. Alternative Proposals

### 6.1 Consideration: Lance-style Columnar Storage
Instead of storing vectors in RocksDB (Row-based), consider a simple append-only columnar file format for the `vectors` and `binary_codes` data.
*   **Pros:** Zero write amplification for vector data. Highly efficient for SIMD scans.
*   **Cons:** Loss of RocksDB's crash recovery/WAL (must implement your own).
*   **Verdict:** Stick with RocksDB for now for engineering simplicity, but keep this as an escape hatch if RocksDB write stalls become an issue at 100M+ scale.

### 6.2 Consideration: Extended-RaBitQ
`HYBRID.md` mentions this. If standard 1-bit RaBitQ (16 bytes for 128D) doesn't provide enough resolution for ranking, the 2-bit or 4-bit variants are a compliant upgrade path that doesn't require training.

---

## 7. Recommendations & Next Steps

1.  **Approve Phase 2 (HNSW2):** Proceed immediately. This is the critical path.
2.  **Mandatory Prototype: ID Mapper:** Build a standalone benchmark for the UUID<->u32 mapping at 1B scale. If it uses >20GB RAM, the entire memory budget is at risk.
3.  **Adopt RaBitQ:** Formally adopt RaBitQ as the compression standard. Discard standard PQ codebooks to simplify the architecture.
4.  **Refine "Delete" Strategy:** Deletions in HNSW are tricky. The `HNSW2` "orphan repair" strategy is complex. Consider a simpler "tombstone now, repair later" background process to keep insert latency low.
5.  **Simulate Disk Latency:** Ensure benchmarks include forced cold-cache scenarios to validate the "Re-ranking" latency budget (loading full vectors from disk).

**Reviewer:** Gemini

---

## Review / Feedback Alignment

**Author:** Claude Opus 4.5 (AI Assistant)
**Date:** December 25, 2025
**Scope:** Cross-review of Gemini's engineering analysis, alignment assessment, and additional considerations.

---

### 1. Executive Alignment Summary

**Overall Assessment:** I am **strongly aligned** with the Gemini review. The technical analysis is accurate, the risk identification is sound, and the proposed implementation roadmap is well-structured. The reviewer correctly identifies the critical path (HNSW2 + RaBitQ) and the key risk (ID Mapping memory budget).

| Gemini Recommendation | Alignment | Comment |
|----------------------|-----------|---------|
| HNSW2 + RaBitQ as primary path | ✅ **Full Agreement** | Correct given [DATA-1](./REQUIREMENTS.md#data-1) constraint |
| ID Mapper as critical risk | ✅ **Full Agreement** | Must prototype early |
| Transaction API over WriteBatchWithIndex | ✅ **Full Agreement** | Already noted in HNSW2.md header |
| Deprioritize IVFPQ (Phase 3) | ✅ **Full Agreement** | GPU adds complexity, violates [DATA-1](./REQUIREMENTS.md#data-1) |
| Phased implementation roadmap | ✅ **Full Agreement** | Risk-first ordering is correct |

---

### 2. Corrections and Clarifications

#### 2.1 Transaction API Already Addressed

The Gemini review correctly notes that `motlie_db::Transaction` exists and should be used. However, this has **already been documented** in [HNSW2.md](./HNSW2.md):

```
> **Note**: The Transaction API for read-your-writes is already implemented in
> `libs/db/src/graph/transaction.rs`. This obviates the need for `WriteBatchWithIndex`
> mentioned in hannoy's design. HNSW2 should use the existing Transaction API directly.
```

This note was added on 2025-12-25. The design documents are already aligned with this recommendation.

#### 2.2 Serialization Strategy Clarification

HNSW2.md now includes a detailed serialization strategy section that addresses storage efficiency concerns:

| Column Family | Serialization | Rationale |
|---------------|---------------|-----------|
| `vectors` | Raw bytes (`bytemuck`) | Zero overhead for f32 arrays |
| `edges` | Roaring native | Optimized compression + set operations |
| `node_meta` | **rkyv** | Zero-copy field access |
| `graph_meta` | **rkyv** | Zero-copy field access |

This addresses the "60% JSON overhead" critique in Section 3.1. The rkyv choice for metadata aligns with zero-copy patterns used in high-performance systems.

---

### 3. Nuanced Perspectives

#### 3.1 ID Mapping Memory Estimate

The Gemini estimate of "1B × (16 + 4 + overhead) ≈ 32GB+" for in-memory HashMap is **accurate for naive implementation** but potentially pessimistic with proper design:

**Alternative Architecture:**

1. **Dense u32 Space:** If we assign u32 IDs sequentially (0, 1, 2, ...), the reverse mapping (`u32 → UUID`) can be stored as a simple `Vec<[u8; 16]>`:
   - Memory: 1B × 16 bytes = **16 GB** (dense array, no hash overhead)
   - Lookups: O(1) array index

2. **Forward Mapping Stays in RocksDB:** The `UUID → u32` direction is only needed during insert. Since inserts are less frequent than searches, a RocksDB-backed lookup with aggressive block cache is acceptable.

3. **Hybrid Strategy:**
   - Reverse (`u32 → UUID`): Memory-mapped file or OS page cache
   - Forward (`UUID → u32`): RocksDB with bloom filters + 1M LRU cache

**Revised Estimate:** ~16-20 GB for reverse mapping, <1 GB for LRU cache = **~17-21 GB total**, well within 64 GB budget.

**Recommendation:** Still prototype early, but the 32GB estimate may be overly conservative.

#### 3.2 RocksDB Write Amplification Mitigation

The Gemini review correctly identifies write amplification risk. Additional mitigation strategies:

1. **FIFO Compaction for Vectors:** Since vectors are write-once (no updates), use FIFO compaction style for the `vectors` CF to minimize compaction overhead.

2. **Tiered Block Cache:** Allocate separate block cache budgets:
   - Edges CF: 4-8 GB (hot, frequently accessed)
   - Vectors CF: 2-4 GB (less hot, larger reads)
   - ID mapping: 4-8 GB (critical for latency)

3. **Rate Limiting:** RocksDB's `WriteController` can rate-limit writes during heavy compaction to prevent latency spikes.

#### 3.3 Direct Read Path for 1B Scale

At 1B vectors with <64GB RAM, the **read path becomes critical**. The design documents don't fully address cold-cache scenarios:

**Concern:** Re-ranking requires loading 20-100 full vectors from disk. At 512 bytes per vector (128D float32), this is 10-50 KB per query. With spinning disks, this could add 20-100ms latency.

**Recommendations:**
1. **SSD Requirement:** At 1B scale, SSDs are mandatory for re-ranking latency targets.
2. **Prefetch Strategy:** During beam search, prefetch likely re-ranking candidates.
3. **mmap for Vectors CF:** Consider memory-mapping the vectors column family for OS-managed caching.
4. **Alternative: Quantized Re-ranking:** Use 4-bit Extended-RaBitQ for intermediate re-ranking (64 bytes/vector), only fetch full vectors for final top-K.

---

### 4. Additional Considerations

#### 4.1 Temporal Aspects (motlie_db Unique Feature)

The motlie_db graph storage includes `ActivePeriod` support (see `libs/db/src/lib.rs:55-110`). This enables:

- **Vector Versioning:** Track when vectors were valid (useful for ML model updates)
- **Point-in-Time Search:** Query "what were the nearest neighbors as of timestamp T?"
- **Soft Deletes:** Mark vectors as invalid without physical deletion

**Recommendation:** Consider whether HNSW2 should inherit temporal semantics. If yes, edges need `valid_range` fields. If no, document this explicitly as a simplification for v1.

#### 4.2 API Compatibility with Existing motlie_db Patterns

The existing `motlie_db::graph::Transaction` API provides:

```rust
pub fn write<M: Into<Mutation>>(&mut self, mutation: M) -> Result<()>
pub fn read<Q: TransactionQueryExecutor>(&self, query: Q) -> Result<Q::Output>
pub fn commit(self) -> Result<()>
```

HNSW2 implementation should:
1. Define `InsertVector`, `DeleteVector` as mutation types
2. Define `GreedySearch`, `BeamSearch` as query types implementing `TransactionQueryExecutor`
3. Ensure merge operations are exposed or wrapped

**Gap Identified:** The current `Transaction` API may not expose `merge` operations. This should be verified and added if missing.

#### 4.3 Filtered Search Integration

The Gemini review endorses Roaring Bitmaps for filtered search. The implementation should consider integration with motlie_db's fulltext search:

**Use Case:** "Find 10 nearest vectors where the document contains 'machine learning'"

**Implementation:**
1. Fulltext query returns `RoaringBitmap` of matching node IDs
2. HNSW2 search intersects neighbor bitmaps with filter
3. Only filtered candidates are considered for distance computation

This requires the ID mapper to maintain consistency between fulltext index IDs and vector index IDs.

---

### 5. Revised Priority Assessment

Based on my analysis, I propose a slight refinement to the implementation roadmap:

| Phase | Gemini Proposal | My Refinement |
|-------|-----------------|---------------|
| 1 | Foundation + ID Mapper | ✅ Agree. Add: Define merge operator interface |
| 2 | HNSW2 Core | ✅ Agree. Add: Validate Transaction API merge support |
| 3 | RaBitQ & Scale | ✅ Agree. Add: Evaluate Extended-RaBitQ (2-4 bit) as fallback |
| 4 | Productionization | ✅ Agree. Add: Cold-cache benchmarking requirement |

**Additional Phase 0 Recommendation:** Before Phase 1, conduct a 1-day spike to:
1. Verify `rocksdb::Transaction` supports merge operations
2. Prototype RoaringBitmap merge operator
3. Benchmark RocksDB block cache for ID mapping workload

This de-risks the most uncertain technical components before committing to full implementation.

---

### 6. Open Questions for Project Owner

1. **Temporal Semantics:** Should vector search support temporal queries, or is this out of scope for v1?

2. **ID Strategy:** Is there flexibility on external IDs? If vectors could use u64 timestamps (like ULID) as primary keys, the mapping overhead could be reduced.

3. **Target Hardware:** Is SSD a requirement for production? The 1B scale projections assume SSD for re-ranking latency.

4. **Filtered Search Priority:** Is filtered search (hybrid fulltext + vector) a P0 requirement, or can it be deferred?

5. **Recall vs Latency Trade-off:** At 1B scale, is 95% recall with <50ms acceptable, or is higher recall worth higher latency?

---

### 7. Conclusion

The Gemini engineering review is **comprehensive and technically sound**. I endorse the proposed architecture (HNSW2 + RaBitQ) and implementation roadmap. The key risks are correctly identified:

1. **ID Mapping Memory:** Manageable with dense reverse mapping, but prototype early
2. **Write Amplification:** Mitigable with compaction tuning and CF separation
3. **Cold-Cache Latency:** Requires SSD and careful prefetching strategy

My primary additions are:
- Clarification that Transaction API guidance is already in HNSW2.md
- More optimistic ID mapping memory estimates with proper architecture
- Emphasis on cold-cache read path testing
- Integration considerations with motlie_db's temporal and fulltext features

**Recommendation:** Proceed with Phase 1 immediately. The architecture is sound.

**Reviewer:** Claude Opus 4.5

---

## Addendum: Design Decisions from Project Owner

**Date:** December 25, 2025

Based on clarifying discussion, the following design decisions have been made:

### A.1 Temporal Filtering Architecture

**Decision:** Vector index remains **temporal-agnostic**. Temporal visibility is enforced during re-ranking via graph RocksDB lookup.

**Rationale:** The vectors represent node/edge summaries and fragments. The graph storage is the source of truth for temporal validity. This separation of concerns keeps the vector index simpler and avoids synchronization complexity.

#### Architecture: Over-Fetch + Temporal Filter

```
Query: "Find 10 nearest neighbors valid at time T"

┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Vector Search (Temporal-Agnostic)                              │
│   Input: query_vector, ef=200                                           │
│   Output: ~200 candidates with (distance, internal_id)                  │
│   Complexity: O(log N × ef)                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2: ID Mapping (internal u32 → ULID)                               │
│   Lookup: reverse mapping array or mmap file                            │
│   Cost: ~1-10μs per candidate (O(1) array index)                        │
│   Total: 200 × 5μs = 1ms                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Temporal Visibility Check (Graph RocksDB)                      │
│   Query: MultiGet on graph CF for ActivePeriod                         │
│   Filter: is_valid_at(temporal_range, query_time)                       │
│   Cost: ~50-100μs for batch of 200 (MultiGet + block cache)             │
│   Output: ~N candidates that pass temporal filter                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Exact Re-ranking (if needed)                                   │
│   Load full vectors for top-M candidates                                │
│   Compute exact distances                                               │
│   Return top-K                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Latency Analysis

| Phase | Operation | Latency (SSD) | Notes |
|-------|-----------|---------------|-------|
| 1 | Vector search (ef=200) | 5-15ms | HNSW + RaBitQ |
| 2 | ID mapping (200 candidates) | 1ms | Dense array O(1) |
| 3 | Temporal filter (MultiGet) | 1-2ms | Graph block cache |
| 4 | Exact re-rank (50 vectors) | 5-10ms | Load from SSD |
| **Total** | | **12-28ms** | Well within targets |

The O(1) lookup overhead per candidate is negligible (~5μs) compared to the distance computation cost.

#### Over-Fetch Ratio Considerations

The over-fetch ratio (ef / k) depends on temporal selectivity:

| Temporal Selectivity | Expected Valid % | Recommended ef for k=10 |
|---------------------|------------------|-------------------------|
| Low (recent data) | 90%+ | ef = 20-50 |
| Medium (historical) | 50-90% | ef = 50-100 |
| High (narrow window) | 10-50% | ef = 100-200 |
| Very High | <10% | ef = 200-500 + progressive fetch |

**Progressive Fetch Strategy** for highly selective queries:

```rust
fn search_with_temporal_filter(
    query: &[f32],
    filter_time: TimestampMilli,
    k: usize,
) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut ef = 50;  // Start small

    while results.len() < k && ef <= 500 {
        // Fetch more candidates
        let candidates = vector_search(query, ef);

        // Map IDs and filter by temporal visibility
        let valid = candidates
            .into_iter()
            .filter(|c| {
                let ulid = id_mapper.get_external(c.internal_id);
                let range = graph.get_temporal_range(ulid);
                is_valid_at_time(&range, filter_time)
            })
            .collect::<Vec<_>>();

        results.extend(valid);
        ef *= 2;  // Double ef for next iteration
    }

    results.truncate(k);
    results
}
```

#### Benefits of This Architecture

1. **Simpler vector index**: No temporal metadata in edges, no sync issues
2. **Single source of truth**: Graph RocksDB is authoritative for visibility
3. **Flexible filtering**: Can add other filters (ACL, soft-delete) at same layer
4. **Cache-friendly**: Temporal lookups hit graph block cache (likely hot)
5. **Consistent with Tantivy pattern**: Fulltext search also defers to graph for visibility

#### Implementation Notes

- **Batch optimization**: Use `MultiGet` for graph lookups to amortize RocksDB overhead
- **Pipeline**: While computing distances for batch N, prefetch graph data for batch N+1
- **Cache warming**: Graph temporal ranges are small (~16 bytes), cache efficiently

---

### A.2 Internal ID Strategy

**Decision:** Vector IDs are internal, similar to Tantivy's `doc_id`. They map back to graph entity ULIDs.

**Implications:**

1. **No UUID in vector index**: Saves 12 bytes per entry (16-byte UUID → 4-byte u32)
2. **Mapping is internal concern**: API accepts/returns ULIDs, translation is hidden
3. **Consistent with existing patterns**: Tantivy already does this for fulltext

**ID Mapper Simplification:**

Since vectors are internal:
- Forward mapping (`ULID → u32`) only needed during insert
- Reverse mapping (`u32 → ULID`) needed for every search result
- Prioritize reverse mapping performance (dense array or mmap)

```rust
pub struct VectorIdMapper {
    // Forward: RocksDB-backed (insert path, less frequent)
    forward: Arc<DB>,  // CF: ulid_to_internal

    // Reverse: Memory-mapped file (search path, hot)
    reverse: Mmap,     // File: [ULID; max_vectors]

    // Allocation
    next_id: AtomicU32,
    free_list: RoaringBitmap,
}
```

---

### A.3 Hardware Requirements

**Decision:** SSD is a **requirement** for production at 1B scale.

This de-risks:
- Re-ranking latency (loading full vectors)
- Cold-cache graph lookups
- Write amplification during compaction

---

### A.4 Recall vs Latency Trade-off

**Decision:** Optimize for **recall over latency**.

Implications for default parameters:

| Parameter | Latency-Optimized | **Recall-Optimized (Chosen)** |
|-----------|-------------------|-------------------------------|
| ef_search | 50-100 | 200-500 |
| rerank_count | 20-50 | 100-200 |
| Extended-RaBitQ bits | 1 bit | 2-4 bits |
| Target recall@10 | 90-95% | **98%+** |
| Expected latency | 10-20ms | 30-50ms |

---

### A.5 Updated Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         motlie_db Unified Storage                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ Graph RocksDB  │  │ Fulltext       │  │ Vector Index               │ │
│  │ (Source of     │  │ (Tantivy)      │  │ (HNSW2 + RaBitQ)           │ │
│  │  Truth)        │  │                │  │                            │ │
│  ├────────────────┤  ├────────────────┤  ├────────────────────────────┤ │
│  │ • Nodes        │  │ • Text search  │  │ • Geometric similarity     │ │
│  │ • Edges        │  │ • doc_id → ULID│  │ • internal_id → ULID       │ │
│  │ • Fragments    │  │                │  │ • Temporal-agnostic        │ │
│  │ • ActivePeriod│  │                │  │                            │ │
│  │ • Visibility   │  │                │  │                            │ │
│  └───────┬────────┘  └───────┬────────┘  └─────────────┬──────────────┘ │
│          │                   │                         │                │
│          └───────────────────┴─────────────────────────┘                │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │  Unified Query      │                              │
│                    │  Executor           │                              │
│                    │  • Fulltext → IDs   │                              │
│                    │  • Vector → IDs     │                              │
│                    │  • Temporal Filter  │                              │
│                    │  • Graph Hydration  │                              │
│                    └─────────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### A.6 Revised Memory Budget (1B Vectors)

With design decisions applied:

| Component | Size | Notes |
|-----------|------|-------|
| HNSW2 edges (roaring) | ~4-8 GB | Roaring bitmaps, block cached |
| RaBitQ binary codes (2-bit) | ~32 GB | In block cache, prioritized |
| Reverse ID mapping (mmap) | 16 GB | `[ULID; 1B]` memory-mapped |
| Graph block cache | 8-12 GB | For temporal lookups |
| Forward ID mapping (RocksDB) | 1-2 GB | Block cache, insert path |
| **Total RAM** | **~50-60 GB** | Within 64 GB budget ✓ |

The dense reverse mapping (16 GB mmap) is the key optimization that makes this feasible.

**Reviewer:** Claude Opus 4.5
