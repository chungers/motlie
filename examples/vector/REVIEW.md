# Engineering Review: Vector Search Architecture

**Author:** Gemini (AI Assistant)
**Date:** December 24, 2025
**Scope:** Review of `examples/vector/*.md` design documents and benchmarks.

---

## 1. Executive Summary

The proposed evolution from the current Proof-of-Concept (POC) to a Hybrid, Billion-Scale architecture is **sound and well-reasoned**, specifically regarding the transition to **RaBitQ** to satisfy the critical `DATA-1` (no pre-training) constraint.

The current POC implementation serves its purpose as a baseline but confirms that a naive RocksDB-backed approach (with per-insert flushing) cannot meet the throughput targets (40 insert/s vs 5,000 insert/s target).

**Key Endorsement:** The **[HNSW2](./HNSW2.md) + [RaBitQ](./HYBRID.md)** combination is the most viable path to meeting the 1B scale requirement within the < 64GB RAM constraint while adhering to `DATA-1`.

**Key Risk:** The **ID Mapping (UUID ↔ u32)** required for HNSW2 and bitsets is a significant architectural bottleneck that needs careful design to avoid blowing the memory budget or killing latency.

---

## 2. Requirements Assessment

The requirements defined in [REQUIREMENTS.md](./REQUIREMENTS.md) are ambitious but realistic for a modern vector database.

| Requirement | Feasibility | Comment |
|-------------|-------------|---------|
| **SCALE-1 (1B Vectors)** | **High** | Achievable only with disk-based storage and compression (RaBitQ). |
| **SCALE-3 (< 64GB RAM)** | **Medium/High Risk** | Pure vectors + graph fits, but **ID Mapping** and **Bloom Filters** will eat significantly into this budget. |
| **DATA-1 (No Training)** | **Strict** | Correctly disqualifies standard PQ, ScaNN, and cluster-based IVF (unless random projection is used). |
| **LAT-1 (< 20ms Search)** | **High** | Achievable with HNSW navigation + SIMD distance. |
| **THR-1 (5k Insert/s)** | **Medium** | Requires moving away from synchronous flushing. The `WriteBatchWithIndex` proposal in HNSW2 is the correct solution. |

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
*   **Critique:** Standard IVF-PQ requires training centroids (k-means), which violates `DATA-1`. While `IVFPQ.md` mentions it as "optional," it introduces a dual-system architecture (RocksDB + GPU Index) that significantly increases operational complexity.
*   **Recommendation:** Deprioritize this phase. Focus resources on making the CPU-based `HNSW2 + RaBitQ` fast enough. Only pursue GPU acceleration if CPU metrics fall short *after* optimization, and even then, ensure the quantization method is training-free.

### 3.4 HYBRID (Phase 4)
*   **Evolution:** The shift from standard PQ to **RaBitQ** (Random Binary Quantization) in the design is the correct move for `DATA-1` compliance.
*   **Architecture:** The "Navigation (HNSW) -> Compressed (RaBitQ) -> Exact (Disk)" funnel is a standard, proven pattern for disk-based vector search (similar to SPANN/DiskANN).

---

## 4. Performance & Benchmarks Review

The analysis in [PERF.md](./PERF.md) is thorough.

*   **Insert Bottleneck:** The diagnosis of the 10ms sleep/flush loop is correct. The projection that removing this will unlock orders of magnitude throughput is accurate, provided the RocksDB write path (WAL + Memtable) doesn't become the next bottleneck.
*   **Recall:** The 95.3% recall at 1M scale for HNSW is excellent and validates the graph implementation.
*   **Vamana:** The struggles with Vamana recall (requiring `L=200` to reach acceptable levels) aligns with industry experience; Vamana typically requires more expensive build parameters than HNSW for equivalent quality.

---

## 5. Critical Risks & Feasibility Analysis

### 5.1 The ID Mapping Problem (SCALE-3 Risk)
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
3.  **Validation:** Verify `REC-1` (Recall) and `THR-1` (>5k inserts/s) on mid-sized datasets (10M).

### Phase 3: RaBitQ & Scale (Weeks 5-6)
**Goal:** Solve the storage/memory bottleneck for 1B scale (`DATA-1`).
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
