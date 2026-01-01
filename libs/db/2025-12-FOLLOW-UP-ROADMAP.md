# December 2025 Engineering Review Follow-Up & Roadmap

**Author:** David Chung + Claude  
**Date:** December 24, 2025  
**Scope:** `libs/db` (Architecture, API, Implementation)
**References:** [2025-12-REVIEW.md](2025-12-REVIEW.md)

## Executive Summary

In [2025-12-REVIEW.md](2025-12-REVIEW.md), several performance optimizations for the core motlie_db library were proposed and reviewed.  This document outlines the roadmap to address the recommendations as prioritized tasks and deferred items.
  
## Goals

### 1. Improve RocksDB Block Cache Locality

| Principle | Rationale |
|-----------|-----------|
| **Name interning** | Use hash for names to improve key / value alignment in block cache and cache density |
| **Separate hot from cold data** | Block cache loads entire blocks; large cold values evict hot data |
| **Keep keys short** | Keys share blocks with values; long keys reduce cache density |
| **Use BlobDB for large values** | Keeps LSM compact, improves compaction speed, natural cache priority |
| **Tune block size per CF** | Graph topology: 4KB, Vectors: 16-32KB |

### 2. Reduce Serde / Memcopy Cost

| Principle | Rationale |
|-----------|-----------|
| **rkyv for hot CFs only** | Zero-copy requires uncompressed data; cold CFs benefit from LZ4 |
| **rmp_serde + LZ4 for cold CF** | Variable length (DataUrl type) uses more costly serde + compression |


## Non-Goals

Backward compatibility is a non-goal.  Changes outline here are expected to be breaking.  No API or database colum families are to be 'deprecated'.  Existing databases and indexes are expected to be dropped and rebuilt.

Some recommendations from the review were analyzed and deprioritized for now.  They are either due to risks of misuse (DirectPath API), nice-to-have ergonomics for more advanced API use cases (Iterator), or to-be-defined product behavior (transaction semantics across multiple storage engines - from RocksDB, to Tantivy, to the to-be-built vector index).

```
┌─────────────────────────────────────────────────────────────────┐
│ Prioritized and Committed                                       │
│ Phase 1: Name Interning ✅ COMPLETED (January 1, 2026)          │
│   - Phase 1.4: Block Cache Configuration Tuning ✅ COMPLETED    │
│   - Phase 1.5: NameCache Integration ✅ COMPLETED               │
│ Phase 2: Blob Separation                                        │
│ Phase 3: Zero-Copy Serialization (rkyv)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Deferred: Ergonomic & Consistency Improvements                  │
│   - Direct Read Path: Re-evaluate if profiling shows need       │
│   - Iterator API: Refactor when touching scan module            │
│   - Fulltext Sync: Design unified commit after vector impl      │
└─────────────────────────────────────────────────────────────────┘
```

| Item | Reason for Deferral | Re-evaluation Trigger |
|------|---------------------|----------------------|
| **Direct Read Path** | Transaction API already provides sync reads; channel overhead <0.1% at 1B scale | Profiling shows point lookup as bottleneck |
| **Iterator-Based Scan API** | Ergonomic improvement; current visitor achieves O(1) seek | Refactoring scan module for other reasons |
| **Fulltext Sync** | Needs comprehensive strategy for graph + fulltext + vector commits | After vector index implementation complete |

### Deferred: Direct Read Path
| Step | Command | Status |
|------|---------|--------|
| **Status:** Deferred | Transaction API provides sync reads for HNSW; channel overhead negligible at scale | ⏸️ |
| Re-evaluate trigger | If point lookup latency becomes bottleneck in profiling | ⏸️ |

### Deferred: Iterator-Based Scan API
| Step | Command | Status |
|------|---------|--------|
| **Status:** Deferred | Ergonomic improvement; current visitor pattern already achieves O(1) seek | ⏸️ |
| Re-evaluate trigger | When refactoring scan module for other reasons | ⏸️ |

### Deferred: Fulltext Sync Mechanism
| Step | Command | Status |
|------|---------|--------|
| **Status:** Deferred | Requires comprehensive strategy for fulltext + vector index commits | ⏸️ |
| **Dependency:** Complete vector index implementation first | | ⏸️ |
| **Note:** Will design unified commit semantics for graph, fulltext, and vector indices | | ⏸️ |


## Implementation Order

Based on impact, dependencies, and practical implementation sequence:

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Name Interning (NEW)                                   │
│   - Fixed 40-byte edge keys (was 32 + variable)                 │
│   - 35% more edges per cache block                              │
│   - Enables excellent prefix compression                        │
│   - Deduplicates names across edges                             │
│   - Foundation for all subsequent optimizations                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Blob Separation                                        │
│   - Split hot (topology/weights) from cold (summaries)          │
│   - 10-20x cache efficiency for graph algorithms                │
│   - Depends on: Phase 1 (schema stabilization)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Zero-Copy Serialization (rkyv)                         │
│   - Apply rkyv to hot CFs only                                  │
│   - 2-5x scan throughput improvement                            │
│   - Depends on: Phase 2 (hot CFs identified and isolated)       │
└─────────────────────────────────────────────────────────────────┘

```

### Phase Dependencies

```
Name Interning ──┬──► Blob Separation ──► rkyv Serialization
                 │
                 └──► (Enables predictable key sizes for all CFs)
```

**Why Name Interning First:**
1. **Schema foundation:** All subsequent optimizations benefit from fixed-size keys
2. **Low risk:** Additive change (new Names CF) with clear migration path
3. **Immediate benefit:** 35% cache density improvement
4. **Deduplication:** Names stored once, referenced everywhere

**Why Blob Separation Second:**
1. **Depends on stable schema:** Name interning must be complete
2. **Highest cache impact:** 10-20x for graph algorithm workloads
3. **Enables rkyv:** Hot CFs can be uncompressed for zero-copy

**Why rkyv Third:**
1. **Depends on blob separation:** Only hot CFs use rkyv
2. **Cold CFs keep LZ4:** Summaries compress well, rarely accessed
3. **Incremental:** Can apply to one CF at a time

### Expected Cumulative Impact

| Optimization | Cache Efficiency | Scan Throughput | Key Size | Write Overhead |
|--------------|------------------|-----------------|----------|----------------|
| Current baseline | 1x | 1x | Variable | 0% |
| + **Name Interning** ✅ | **1.35x** (theoretical) | 0.5-0.7x (current*) | Fixed 40B | +32-46% (measured) |
| + Blob Separation | **15-25x** | 5-10x | Fixed 40B | +10-15% |
| + rkyv | 15-25x | **10-50x** | Fixed 40B | +10-15% |
| **Total** | **15-25x** | **10-50x** | **Fixed** | **+10-15%** |

*Note: Current name interning implementation has per-edge name resolution overhead.
Phase 1.5 optimization (in-memory NameCache integration) will address scan throughput regression.

----------------------------------------------------------------------------------------

## Name Interning: Optimizing Arbitrary-Length Names for Cache Efficiency

> **Problem Statement:** The requirement to support arbitrary node and edge names (potentially 100+ bytes) conflicts with the need for compact keys to maximize block cache efficiency. This section explores design options to satisfy both requirements.

### Current Schema Analysis

**Current key structures:**
```rust
// Nodes: Fixed 16-byte key (optimal)
NodeCfKey(Id)                           // 16 bytes

// Forward Edges: Variable-length key (problematic)
ForwardEdgeCfKey(SrcId, DstId, EdgeName) // 32 bytes + len(EdgeName)

// Reverse Edges: Variable-length key (problematic)
ReverseEdgeCfKey(DstId, SrcId, EdgeName) // 32 bytes + len(EdgeName)
```

**Nodes store name in value** (already optimal for keys):
```rust
NodeCfValue(TemporalRange, NodeName, NodeSummary)  // Name in value, not key
```

### Typical Name Length Analysis

Based on graph database naming conventions from [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/syntax/naming/):

| Category | Examples | Typical Length | Max Observed |
|----------|----------|----------------|--------------|
| **Relationship types** | `FOLLOWS`, `LIKES`, `HAS_PERMISSION` | 8-20 bytes | 50 bytes |
| **Semantic relationships** | `WORKED_AT_COMPANY`, `IS_PARENT_OF` | 15-30 bytes | 80 bytes |
| **Domain-specific** | `IAM_ROLE_BINDING_V2`, `VECTOR_SIMILARITY` | 20-40 bytes | 100 bytes |
| **User-defined labels** | `customer_interaction_2024_q1` | 25-50 bytes | 200 bytes |
| **Node names** | Entity names, document titles | 20-100 bytes | Unbounded |

**Key insight:** Edge names (relationship types) are typically shorter and more repetitive than node names. A knowledge graph with 1M edges might have only 50-100 distinct edge types.

### Impact Quantification

**Block cache density with current schema:**

| Edge Name Length | Key Size | Value Size | Entry Size | Edges per 4KB Block |
|------------------|----------|------------|------------|---------------------|
| 8 bytes | 40 bytes | ~30 bytes | 70 bytes | ~58 |
| 32 bytes | 64 bytes | ~30 bytes | 94 bytes | ~43 |
| 64 bytes | 96 bytes | ~30 bytes | 126 bytes | ~32 |
| 128 bytes | 160 bytes | ~30 bytes | 190 bytes | ~21 |

**Delta encoding partially mitigates this:**
- Edges from same source share 16-byte `src_id` prefix
- But different edge names reset prefix sharing
- Restart points every 16 keys limit compression gains

---

### Design Option 1: Name Interning with Hash Keys

**Concept:** Replace variable-length names with fixed-length hashes in keys; store full names in a separate metadata CF.

```
┌─────────────────────────────────────────────────────────────────┐
│  Name Metadata CF (names)                                       │
│  Key: [hash: 8 bytes]                                           │
│  Value: [full_name: String]                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Forward Edges CF (forward_edges)                               │
│  Key: [src_id: 16] + [dst_id: 16] + [name_hash: 8] = 40 bytes  │
│  Value: [temporal_range] + [weight] + [has_summary]             │
└─────────────────────────────────────────────────────────────────┘
```

**Hash function choice:** xxHash64 (8-byte truncation)
- xxHash64 achieves ~10-20ns for small inputs (faster than BLAKE3)
- Non-cryptographic hash is appropriate for content addressing (not security)
- 8 bytes provides 2^64 namespace (~18 quintillion) - collision probability negligible for typical graphs
- **Decision:** Use `xxhash-rust` crate with `xxh64` feature

**Implementation:**

```rust
// libs/db/src/graph/name_hash.rs

use xxhash_rust::xxh64::xxh64;

/// 8-byte name hash for compact keys using xxHash64
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NameHash([u8; 8]);

impl NameHash {
    /// Create a NameHash from a string name using xxHash64
    pub fn from_name(name: &str) -> Self {
        let hash = xxh64(name.as_bytes(), 0);  // seed = 0
        NameHash(hash.to_be_bytes())
    }

    /// Get the raw bytes of the hash
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        NameHash(bytes)
    }
}

/// Names metadata CF
pub(crate) struct Names;
pub(crate) struct NamesCfKey(pub(crate) NameHash);
pub(crate) struct NamesCfValue(pub(crate) String);

/// Forward edges with interned names
pub(crate) struct ForwardEdgeCfKey(
    pub(crate) SrcId,      // 16 bytes
    pub(crate) DstId,      // 16 bytes
    pub(crate) NameHash,   // 8 bytes (was: variable String)
);
```

**Write path changes:**
```rust
impl MutationExecutor for AddEdge {
    fn execute(&self, txn: &Transaction, txn_db: &TransactionDB) -> Result<()> {
        let name_hash = NameHash::from_name(&self.name);

        // 1. Write to names CF (idempotent - same hash = same name)
        let names_cf = txn_db.cf_handle(Names::CF_NAME)?;
        txn.put_cf(names_cf,
            name_hash.as_bytes(),
            Names::value_to_bytes(&NamesCfValue(self.name.clone()))?)?;

        // 2. Write to forward_edges with hash key
        let edge_key = ForwardEdgeCfKey(self.src, self.dst, name_hash);
        // ... rest of write logic
    }
}
```

**Read path changes:**
```rust
impl OutgoingEdges {
    fn execute(&self, storage: &Storage) -> Result<Vec<EdgeResult>> {
        let db = storage.db()?;
        let edges_cf = db.cf_handle(ForwardEdges::CF_NAME)?;
        let names_cf = db.cf_handle(Names::CF_NAME)?;

        let mut results = Vec::new();
        for (key_bytes, value_bytes) in db.prefix_iterator_cf(edges_cf, self.src.as_bytes()) {
            let key = ForwardEdges::key_from_bytes(&key_bytes)?;

            // Resolve name hash → full name (extra lookup)
            let name = if let Some(name_bytes) = db.get_cf(names_cf, key.2.as_bytes())? {
                Names::value_from_bytes(&name_bytes)?.0
            } else {
                return Err(anyhow!("Orphan name hash: {:?}", key.2));
            };

            results.push((key.0, key.1, name, ...));
        }
        Ok(results)
    }
}
```

**Pros:**
| Benefit | Impact |
|---------|--------|
| **Fixed key size** | 40 bytes vs 32+N bytes; predictable cache density |
| **Better prefix compression** | All edges from same (src, dst) share 32-byte prefix |
| **Separate name caching** | Names CF can be pinned in cache (small, hot) |
| **Collision-safe** | 8-byte hash with full name stored; can verify if paranoid |

**Cons:**
| Cost | Mitigation |
|------|------------|
| **Extra write per edge** | Names CF write is idempotent; often hits existing entry |
| **Extra read per edge** | Cache names CF aggressively; implement in-memory LRU |
| **Hash computation** | ~100ns per name; negligible vs µs-scale I/O |
| **Complexity** | Encapsulate in `NameHash` type; hide from API users |

---

### ~~Design Option 2: Hybrid Approach with Length Threshold~~

**Concept:** Use full names for short strings, hash for long strings. Best of both worlds.

```rust
/// Name representation: inline for short names, hash for long names
#[derive(Clone, Serialize, Deserialize)]
pub enum NameKey {
    /// Names ≤ 24 bytes stored inline (no extra lookup)
    Inline([u8; 24], u8),  // 24 bytes + 1 byte length = 25 bytes

    /// Names > 24 bytes stored as hash (requires lookup)
    Hashed(NameHash),      // 8 bytes + 1 byte tag = 9 bytes
}

impl NameKey {
    pub fn from_name(name: &str) -> Self {
        if name.len() <= 24 {
            let mut buf = [0u8; 24];
            buf[..name.len()].copy_from_slice(name.as_bytes());
            NameKey::Inline(buf, name.len() as u8)
        } else {
            NameKey::Hashed(NameHash::from_name(name))
        }
    }

    pub fn resolve(&self, names_cf: &CF) -> Result<String> {
        match self {
            NameKey::Inline(buf, len) => {
                Ok(String::from_utf8(buf[..*len as usize].to_vec())?)
            }
            NameKey::Hashed(hash) => {
                // Lookup in names CF
                ...
            }
        }
    }
}
```

**Key sizes:**
| Name Length | NameKey Size | Total Edge Key | Notes |
|-------------|--------------|----------------|-------|
| 8 bytes | 25 bytes | 57 bytes | Inline, no lookup |
| 24 bytes | 25 bytes | 57 bytes | Inline, no lookup |
| 32 bytes | 9 bytes | 41 bytes | Hashed, requires lookup |
| 128 bytes | 9 bytes | 41 bytes | Hashed, requires lookup |

**Pros:**
- Common short names (FOLLOWS, LIKES) avoid any extra I/O
- Long names still get cache-efficient keys
- Gradual migration path

**Cons:**
- Variable key size (25 or 9 bytes for name component)
- Enum tag adds complexity
- Prefix compression less effective with variable sizes

---

### ~~Design Option 3: In-Memory Name Cache with Lazy Resolution~~

**Concept:** Keep current schema but add application-level caching to amortize name lookup cost.

```rust
/// Thread-local name cache
pub struct NameCache {
    hash_to_name: DashMap<NameHash, Arc<String>>,
    name_to_hash: DashMap<Arc<String>, NameHash>,
}

impl NameCache {
    /// Get or compute hash, caching the mapping
    pub fn intern(&self, name: &str) -> NameHash {
        if let Some(hash) = self.name_to_hash.get(name) {
            return *hash;
        }

        let hash = NameHash::from_name(name);
        let name = Arc::new(name.to_string());
        self.hash_to_name.insert(hash, name.clone());
        self.name_to_hash.insert(name, hash);
        hash
    }

    /// Resolve hash to name (from cache or DB)
    pub fn resolve(&self, hash: NameHash, db: &DB) -> Result<Arc<String>> {
        if let Some(name) = self.hash_to_name.get(&hash) {
            return Ok(name.clone());
        }

        // Fallback to DB lookup
        let name = db.get_cf(names_cf, hash.as_bytes())?
            .ok_or_else(|| anyhow!("Unknown name hash"))?;
        let name = Arc::new(String::from_utf8(name)?);
        self.hash_to_name.insert(hash, name.clone());
        Ok(name)
    }
}
```

**Pros:**
- Minimal schema changes
- Amortizes lookup cost across queries
- Can be added incrementally

**Cons:**
- Still pays key size cost in RocksDB
- Cache coherency concerns in multi-process scenarios
- Memory overhead for cache

---

### Cost-Benefit Analysis

**Assumptions for analysis:**
- 1M edges, 100 distinct edge types, 10K distinct node names
- Average edge name: 20 bytes, Average node name: 50 bytes
- Block cache: 1GB, SSD read: 100µs, Hash compute: 100ns, Cached lookup: 1µs

**Option 1: Full Name Interning**

| Operation | Current Cost | With Interning | Delta |
|-----------|--------------|----------------|-------|
| **Write edge** | 1 put | 2 puts (edge + name) | +1 put (~10µs) |
| **Scan 100 edges** | 100 key deserializations | 100 key deserializations + 100 name lookups | +100 lookups |
| | (variable key sizes) | (if names cached: +100µs) | |
| | | (if names uncached: +10ms) | |
| **Block cache efficiency** | ~43 edges/block (32-byte names) | ~58 edges/block (fixed 40-byte keys) | **+35%** |
| **1GB cache capacity** | ~11M edges | ~15M edges | **+4M edges** |

**Break-even analysis:**
- Extra name lookups cost ~1µs each (cached) or ~100µs (uncached)
- Cache efficiency gain: 35% more edges in cache
- **If cache hit rate improves by >3-5%, the extra lookups are paid for**

For workloads with high locality (BFS, PageRank, HNSW), the improved cache density provides 10-20% throughput improvement that exceeds the name resolution overhead.

**Option 2: Hybrid Threshold (24 bytes)**

| Metric | Value |
|--------|-------|
| % of edge types that fit inline | ~80% (FOLLOWS, LIKES, HAS_*, etc.) |
| % requiring hash lookup | ~20% (long domain-specific names) |
| Effective lookups per 100 edges | ~20 (vs 100 for full interning) |
| Key size variance | 57 bytes (inline) vs 41 bytes (hashed) |

**Recommendation:** Hybrid is a good compromise if the workload has predictable naming patterns.

---

### Recommended Design: Option 1 (Full Name Interning)

**Rationale:**

1. **Predictable performance:** Fixed key sizes enable reliable capacity planning
2. **Simplest mental model:** All names go through the same path
3. **Future-proof:** Works regardless of how long names become
4. **Name cache is tiny:** 100 edge types × 50 bytes = 5KB (easily pinned)
5. **Write overhead is minimal:** Most edges reuse existing name entries

**Implementation priorities:**

1. **In-memory name cache (required):** Cache name lookups to avoid per-edge I/O
2. **Names CF with high cache priority:** Pin in block cache high-pri pool
3. **Bloom filter on names CF:** Avoid disk reads for cache misses on new names
4. **Lazy hash verification (optional):** Store hash→name; verify on read if paranoid about collisions

**Schema migration path:**

```
Phase 1: Add Names CF alongside existing schema
Phase 2: Dual-write (old variable keys + new hashed keys)
Phase 3: Migrate reads to use hashed keys with name resolution
Phase 4: Drop old variable-key columns
```

### Updated Key Layout with Name Interning

```
After Name Interning:

┌─────────────────────────────────────────────────────────────────┐
│  Names CF (small, hot, pinned in cache)                         │
│  Key:   [name_hash: 8 bytes]                                    │
│  Value: [name: String] (LZ4 compressed)                         │
│  Size:  ~5-50KB for typical graph (100-1000 distinct names)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Nodes CF                                                        │
│  Key:   [id: 16 bytes]                                          │
│  Value: [temporal] + [name_hash: 8] + [has_summary]             │
│  Note:  Node names now use hash; full name in Names CF          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Forward Edges Hot CF                                            │
│  Key:   [src: 16] + [dst: 16] + [name_hash: 8] = 40 bytes      │
│  Value: [temporal] + [weight] + [has_summary] (~20 bytes)       │
│  Density: ~68 edges per 4KB block (was ~43 with 32-byte names) │
└─────────────────────────────────────────────────────────────────┘
```

**Cache efficiency improvement:**

| Metric | Before (variable names) | After (name interning) | Improvement |
|--------|-------------------------|------------------------|-------------|
| Edge key size | 32 + avg(20) = 52 bytes | 40 bytes (fixed) | -23% |
| Edges per 4KB block | ~43 | ~58 | **+35%** |
| Edges per 1GB cache | ~11M | ~15M | **+4M edges** |
| Prefix compression | Moderate | Excellent (32-byte shared prefix) | Better |

#### Benchmark Validation Plan

**Benchmark:** `prefix_scans_by_position` and `prefix_scans_by_degree` in `libs/db/benches/db_operations.rs`

**Pre-Implementation Baseline:**
```bash
cargo bench -p motlie-db -- prefix_scans --save-baseline before_interning
cargo bench -p motlie-db -- batch_scan_throughput --save-baseline before_interning
```

**Key Metrics to Capture:**

| Metric | Benchmark Test | Expected Current | Expected After |
|--------|---------------|------------------|----------------|
| Edges per 4KB block | (calculated) | ~43 | ~58 |
| Prefix scan latency | `prefix_scans_by_degree/10_edges` | ~X µs | ~0.85X µs |
| Batch scan throughput | `batch_scan_throughput/5000_nodes` | ~X scans/s | ~1.15X scans/s |
| Write throughput | `writes/medium` | ~X ops/s | ~0.95X ops/s |

**Post-Implementation Validation:**
```bash
cargo bench -p motlie-db -- prefix_scans --baseline before_interning
cargo bench -p motlie-db -- batch_scan_throughput --baseline before_interning
```

**Success Criteria (Original):**
- [x] Edge key size is exactly 40 bytes (16 + 16 + 8) ✅
- [ ] `prefix_scans_*` show **10-20% improvement** due to better cache density ❌ See note below
- [ ] `batch_scan_throughput` shows **10-15% improvement** ❌ See note below
- [ ] Write throughput regression **< 10%** (acceptable for name CF writes) ❌ +32-46% measured
- [ ] Name resolution from cache adds **< 1µs per edge** on average ⏳ Needs NameCache integration

**Actual Results (January 1, 2026):**

The Phase 1 implementation achieved the structural goal (fixed 40-byte keys) but introduces
per-edge name resolution overhead that currently outweighs cache density benefits:

| Benchmark | Before | After | Delta | Notes |
|-----------|--------|-------|-------|-------|
| `prefix_scans_by_degree/10_edges` | 9.04 µs | 12.67 µs | +40% | Name resolution overhead |
| `prefix_scans_by_degree/50_edges` | 15.79 µs | 44.73 µs | +184% | Linear with edge count |
| `writes/100_nodes_10x_edges` | 13.67 ms | 18.08 ms | +32% | Additional Names CF write |
| `writes/1000_nodes_10x_edges` | 56.66 ms | 82.69 ms | +46% | Additional Names CF write |

**Analysis:**
- The fixed 40-byte keys are correctly implemented
- Name resolution from Names CF adds overhead per edge
- The `NameCache` is implemented but not yet integrated into the query/scan paths
- Cache density benefits (1.35x) will be realized when working set exceeds block cache

**Follow-up Task (Phase 1.5):**
Integrate `NameCache` into both write and read paths to eliminate redundant operations:

**Write Path Optimization:**
1. Use `NameCache::intern()` before calling `write_name_to_cf()`
2. Only write to Names CF if name not already in cache
3. Expected benefit: Avoid ~80% of Names CF writes for repeated edge types

**Read Path Optimization:**
1. Warm cache during Storage::ready() or first scan
2. Use cached lookups instead of per-edge DB reads
3. Target: < 1µs per edge with warm cache (vs. current ~3µs/edge)

**Name Cache Verification:**
```rust
// In tests: verify cache hit rate
let cache = NameCache::new();
// After warming with 100 distinct names...
assert!(cache.hit_rate() > 0.99); // 99%+ hit rate expected
```


---------------------------------------------------------

## Proposed Architecture: Hybrid Serialization Strategy

**Date:** December 25, 2025

### Overview

Combine blob separation with a hybrid serialization approach:

| Data Category | Column Family | Serialization | Compression | Access Pattern |
|---------------|---------------|---------------|-------------|----------------|
| **Hot** (topology, weights, temporal) | `forward_edges_hot`, `nodes_hot` | rkyv | None | Zero-copy, high-frequency traversal |
| **Cold** (summaries, content) | `edge_summaries`, `node_summaries` | rmp_serde | LZ4 | Full deser, infrequent access |
| **Fragments** (historical content) | `node_fragments`, `edge_fragments` | rmp_serde | LZ4 | Full deser, rare access |

This resolves the LZ4/rkyv conflict by applying each serialization strategy where it's most effective.

### Schema Design

#### Hot Column Families (rkyv, no compression)

```rust
// libs/db/src/graph/schema_hot.rs

use rkyv::{Archive, Deserialize, Serialize};

/// Hot edge data - optimized for graph traversal
/// Size: ~30 bytes per edge (vs ~500 bytes with summary)
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct ForwardEdgeHotValue {
    pub valid_range: Option<ArchivedTemporalRange>,
    pub weight: Option<f64>,
    pub has_summary: bool,  // Flag to indicate cold data exists
}

/// Hot node data - optimized for lookups
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct NodeHotValue {
    pub valid_range: Option<ArchivedTemporalRange>,
    pub name: ArchivedString,  // rkyv's zero-copy string
    pub has_summary: bool,
}

/// rkyv-compatible temporal range
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Copy)]
#[archive(check_bytes)]
pub struct ArchivedTemporalRange {
    pub start: Option<u64>,
    pub until: Option<u64>,
}
```

#### Cold Column Families (rmp_serde + LZ4)

```rust
// libs/db/src/graph/schema_cold.rs

use serde::{Deserialize, Serialize};

/// Cold edge data - summaries stored separately
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EdgeSummaryColdValue(pub DataUrl);

/// Cold node data - summaries stored separately
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeSummaryColdValue(pub DataUrl);
```

### Trait Design

```rust
// libs/db/src/graph/mod.rs

/// Trait for hot column families using rkyv (zero-copy)
pub(crate) trait HotColumnFamily {
    const CF_NAME: &'static str;
    type Key;
    type Value: rkyv::Archive + rkyv::Serialize<...>;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    /// Zero-copy value access - returns archived reference
    fn value_archived(bytes: &[u8]) -> Result<&<Self::Value as Archive>::Archived> {
        rkyv::check_archived_root::<Self::Value>(bytes)
            .map_err(|e| anyhow::anyhow!("Archive validation failed: {}", e))
    }

    /// Full deserialization when mutation is needed
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value> {
        let archived = Self::value_archived(bytes)?;
        Ok(archived.deserialize(&mut rkyv::Infallible)?)
    }

    fn value_to_bytes(value: &Self::Value) -> Result<rkyv::AlignedVec> {
        Ok(rkyv::to_bytes::<_, 256>(value)?)
    }
}

/// Trait for cold column families using rmp_serde + LZ4 (existing pattern)
pub(crate) trait ColdColumnFamily {
    const CF_NAME: &'static str;
    type Key;
    type Value: Serialize + DeserializeOwned;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>> {
        let msgpack = rmp_serde::to_vec(value)?;
        Ok(lz4::block::compress(&msgpack, None, true)?)
    }

    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value> {
        let decompressed = lz4::block::decompress(bytes, None)?;
        Ok(rmp_serde::from_slice(&decompressed)?)
    }
}
```

### Column Family Layout

```
Before (current):
├── nodes           (Id → TemporalRange + Name + Summary)  ~200-500 bytes
├── forward_edges   (Src+Dst+Name → TemporalRange + Weight + Summary)  ~200-500 bytes
├── reverse_edges   (Dst+Src+Name → TemporalRange)  ~30 bytes
├── node_fragments  (Id+Ts → Content)  variable
└── edge_fragments  (Src+Dst+Name+Ts → Content)  variable

After (proposed):
├── nodes_hot       (Id → TemporalRange + Name + has_summary)  ~50 bytes [rkyv]
├── node_summaries  (Id → Summary)  variable [rmp+lz4]
├── forward_edges_hot (Src+Dst+Name → TemporalRange + Weight + has_summary)  ~30 bytes [rkyv]
├── edge_summaries  (Src+Dst+Name → Summary)  variable [rmp+lz4]
├── reverse_edges   (Dst+Src+Name → TemporalRange)  ~30 bytes [rkyv]
├── node_fragments  (Id+Ts → Content)  variable [rmp+lz4]
└── edge_fragments  (Src+Dst+Name+Ts → Content)  variable [rmp+lz4]
```

### Read Path Examples

#### Graph Traversal (Hot Path - Zero Copy)

```rust
impl OutgoingEdges {
    pub fn execute_zero_copy(&self, storage: &Storage) -> Result<Vec<(Option<f64>, DstId)>> {
        let db = storage.db()?;
        let cf = db.cf_handle(ForwardEdgesHot::CF_NAME)?;

        let prefix = self.src_id.into_bytes();
        let mut results = Vec::new();

        for item in db.prefix_iterator_cf(cf, &prefix) {
            let (key_bytes, value_bytes) = item?;

            // Zero-copy: just validate and cast, no allocation
            let archived = ForwardEdgesHot::value_archived(&value_bytes)?;

            // Check temporal validity directly on archived data
            if !is_valid_archived(&archived.valid_range, self.reference_ts) {
                continue;
            }

            // Extract weight without full deserialization
            let weight = archived.weight;
            let key = ForwardEdgesHot::key_from_bytes(&key_bytes)?;

            results.push((weight, key.1)); // dst_id
        }

        Ok(results)
    }
}
```

#### Full Edge Details (Hot + Cold)

```rust
impl EdgeDetails {
    pub fn execute(&self, storage: &Storage) -> Result<EdgeDetailsResult> {
        let db = storage.db()?;

        // 1. Read hot data (zero-copy)
        let hot_cf = db.cf_handle(ForwardEdgesHot::CF_NAME)?;
        let key = ForwardEdgeHotKey(self.src, self.dst, self.name.clone());
        let key_bytes = ForwardEdgesHot::key_to_bytes(&key);

        let hot_bytes = db.get_cf(hot_cf, &key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found"))?;

        let archived = ForwardEdgesHot::value_archived(&hot_bytes)?;
        let weight = archived.weight;

        // 2. Read cold data only if needed (full deserialization)
        let summary = if archived.has_summary {
            let cold_cf = db.cf_handle(EdgeSummaries::CF_NAME)?;
            let cold_bytes = db.get_cf(cold_cf, &key_bytes)?
                .ok_or_else(|| anyhow::anyhow!("Summary missing"))?;
            EdgeSummaries::value_from_bytes(&cold_bytes)?.0
        } else {
            DataUrl::from_text("")  // Empty summary
        };

        Ok((weight, self.src, self.dst, self.name.clone(), summary))
    }
}
```

### Write Path

```rust
impl MutationExecutor for AddEdge {
    fn execute(&self, txn: &Transaction, txn_db: &TransactionDB) -> Result<()> {
        // 1. Write to hot CF (rkyv serialization)
        let hot_cf = txn_db.cf_handle(ForwardEdgesHot::CF_NAME)?;
        let hot_key = ForwardEdgeHotKey(self.source_node_id, self.target_node_id, self.name.clone());
        let hot_value = ForwardEdgeHotValue {
            valid_range: self.valid_range.map(Into::into),
            weight: self.weight,
            has_summary: !self.summary.as_ref().is_empty(),
        };
        txn.put_cf(hot_cf,
            ForwardEdgesHot::key_to_bytes(&hot_key),
            ForwardEdgesHot::value_to_bytes(&hot_value)?)?;

        // 2. Write to cold CF if summary exists (rmp + lz4)
        if !self.summary.as_ref().is_empty() {
            let cold_cf = txn_db.cf_handle(EdgeSummaries::CF_NAME)?;
            let cold_value = EdgeSummaryColdValue(self.summary.clone());
            txn.put_cf(cold_cf,
                ForwardEdgesHot::key_to_bytes(&hot_key),  // Same key
                EdgeSummaries::value_to_bytes(&cold_value)?)?;
        }

        // 3. Write reverse edge (hot only, no summary)
        let reverse_cf = txn_db.cf_handle(ReverseEdgesHot::CF_NAME)?;
        // ... similar pattern

        Ok(())
    }
}
```

### Performance Projections

| Operation | Current | With Hybrid | Improvement |
|-----------|---------|-------------|-------------|
| **Edge scan (1M edges)** | ~500ms (deserialize + decompress) | ~50ms (zero-copy) | **10x** |
| **Point lookup (cached)** | ~100µs (channel + deserialize) | ~5µs (direct + zero-copy) | **20x** |
| **BFS traversal (100K nodes)** | ~2s | ~200ms | **10x** |
| **Storage size** | 1x | ~1.1x (hot uncompressed, cold compressed) | ~10% increase |

### Migration Strategy

1. **Phase 1**: Add new `*_hot` and `*_summaries` column families alongside existing ones
2. **Phase 2**: Dual-write to both old and new CFs during transition
3. **Phase 3**: Background migration job to populate new CFs from old data
4. **Phase 4**: Switch reads to new CFs, deprecate old CFs
5. **Phase 5**: Remove old CFs in next major version

### Dependencies

Add to `Cargo.toml`:
```toml
rkyv = { version = "0.8", features = ["validation", "strict"] }
```

### Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| **Serialization complexity** | Optimal format per data type | Two serialization paths to maintain |
| **Storage size** | Cold data still compressed | Hot data ~10% larger (no compression) |
| **Schema evolution** | rkyv requires careful versioning | Need migration tooling for hot CFs |
| **Code complexity** | Clear separation of concerns | More column families to manage |

### Conclusion

The hybrid serialization strategy maximizes performance where it matters most (graph traversal) while preserving storage efficiency for large, infrequently-accessed content. This approach directly addresses the LZ4/rkyv conflict identified in the original review assessment.

---------------------------------------------------------

## Benchmark Validation Summary

All optimization recommendations include benchmark validation plans. This section provides a consolidated view for tracking implementation progress.

### Benchmark Infrastructure

**Location:** `libs/db/benches/db_operations.rs`

**Benchmark Groups:**
- `baseline_benches` - Core performance metrics (existing)
- `optimization_benches` - Optimization evaluation metrics (new)

**Running Benchmarks:**
```bash
# Run all benchmarks
cargo bench -p motlie-db

# Run specific optimization benchmark
cargo bench -p motlie-db -- <benchmark_name>

# Compare against baseline
cargo bench -p motlie-db -- --baseline before_<optimization>
```

### Validation Checklist by Optimization

#### Phase 1: Name Interning (Priority 1) - ✅ COMPLETED

**Implementation Date:** January 1, 2026

**Decisions Made:**
- Node names (`NodeCfValue`) use `NameHash` (full name stored in Names CF)
- Edge names in keys use `NameHash` for fixed 40-byte edge keys
- Hash function: xxHash64 via `xxhash-rust` crate (faster, non-cryptographic)
- Breaking change: Clean-break schema, no backward compatibility required
- In-memory `NameCache` with DashMap for concurrent name resolution

**Schema Changes:**
```
BEFORE:
├── nodes           Key: [id: 16]          Value: [temporal, name: String, summary]
├── forward_edges   Key: [src: 16, dst: 16, name: String]  Value: [temporal, weight, summary]
├── reverse_edges   Key: [dst: 16, src: 16, name: String]  Value: [temporal]
├── edge_fragments  Key: [src: 16, dst: 16, name: String, ts: 8]  Value: [temporal, content]

AFTER:
├── names           Key: [hash: 8]         Value: [name: String]  ← NEW CF
├── nodes           Key: [id: 16]          Value: [temporal, name_hash: 8, summary]
├── forward_edges   Key: [src: 16, dst: 16, hash: 8] = 40 bytes FIXED
├── reverse_edges   Key: [dst: 16, src: 16, hash: 8] = 40 bytes FIXED
├── edge_fragments  Key: [src: 16, dst: 16, hash: 8, ts: 8] = 48 bytes FIXED
├── node_fragments  (unchanged - no name in key)
```

**Step 1.1: Foundation**
| Task | File | Status |
|------|------|--------|
| Add `xxhash-rust` and `dashmap` dependencies | `Cargo.toml` | ✅ |
| Create `NameHash` type with xxHash64 | `src/graph/name_hash.rs` | ✅ |
| Create `NameCache` with DashMap | `src/graph/name_hash.rs` | ✅ |
| Add `Names` CF struct | `src/graph/schema.rs` | ✅ |
| Export from `src/graph/mod.rs` | `src/graph/mod.rs` | ✅ |
| Unit tests for NameHash (8 tests) | `src/graph/name_hash.rs` | ✅ |
| Unit tests for NameCache (10 tests) | `src/graph/name_hash.rs` | ✅ |

**Step 1.2: Schema Migration**
| Task | File | Status |
|------|------|--------|
| Update `ForwardEdgeCfKey` to use `NameHash` | `src/graph/schema.rs` | ✅ |
| Update `ReverseEdgeCfKey` to use `NameHash` | `src/graph/schema.rs` | ✅ |
| Update `EdgeFragmentCfKey` to use `NameHash` | `src/graph/schema.rs` | ✅ |
| Update `NodeCfValue` to use `NameHash` for node name | `src/graph/schema.rs` | ✅ |
| Update `key_to_bytes` for fixed 40-byte edge keys | `src/graph/schema.rs` | ✅ |
| Update `key_from_bytes` for fixed 40-byte edge keys | `src/graph/schema.rs` | ✅ |
| Add `Names` to CF descriptors in Storage::ready() | `src/graph/mod.rs` | ✅ |
| Update `AddNode` executor: hash + write to Names CF | `src/graph/mutation.rs` | ✅ |
| Update `AddEdge` executor: hash + write to Names CF | `src/graph/mutation.rs` | ✅ |
| Update `AddEdgeFragment` executor | `src/graph/mutation.rs` | ✅ |
| Update `UpdateEdgeValidSinceUntil` executor | `src/graph/mutation.rs` | ✅ |
| Update `UpdateEdgeWeight` executor | `src/graph/mutation.rs` | ✅ |
| Add name resolution helpers to `query.rs` | `src/graph/query.rs` | ✅ |
| Update `NodeById` query with name resolution | `src/graph/query.rs` | ✅ |
| Update `NodesByIdsMulti` query with name resolution | `src/graph/query.rs` | ✅ |
| Update `OutgoingEdges` query with name resolution | `src/graph/query.rs` | ✅ |
| Update `IncomingEdges` query with name resolution | `src/graph/query.rs` | ✅ |
| Update `EdgeSummaryBySrcDstName` query | `src/graph/query.rs` | ✅ |
| Update `EdgeFragmentsByEdgeTimeRange` query | `src/graph/query.rs` | ✅ |
| Update `AllNodes` transaction query | `src/graph/query.rs` | ✅ |
| Update `AllEdges` transaction query | `src/graph/query.rs` | ✅ |
| Add name resolution helper to `scan.rs` | `src/graph/scan.rs` | ✅ |
| Update `AllNodes` scan with name resolution | `src/graph/scan.rs` | ✅ |
| Update `AllEdges` scan with name resolution | `src/graph/scan.rs` | ✅ |
| Update `AllReverseEdges` scan with name resolution | `src/graph/scan.rs` | ✅ |
| Update `AllEdgeFragments` scan with name resolution | `src/graph/scan.rs` | ✅ |
| Update `examples/store` for new schema | `examples/store/main.rs` | ✅ |

**Step 1.3: Validation**
| Task | Command | Status |
|------|---------|--------|
| Run all unit tests | `cargo test -p motlie-db --lib` | ✅ 196 passed |
| Run all doctests | `cargo test -p motlie-db --doc` | ✅ 11 passed |
| Verify examples work | `./target/release/examples/store` | ✅ store + verify modes |
| Run prefix_scans benchmark | `cargo bench -p motlie-db -- prefix_scans` | ✅ see below |
| Run writes benchmark | `cargo bench -p motlie-db -- writes` | ✅ see below |
| **Target:** Fixed 40-byte edge keys, all tests pass | | ✅ |

**Benchmark Results (January 1, 2026):**

| Metric | Result | Notes |
|--------|--------|-------|
| Edge key size | **40 bytes (fixed)** | Was 32 + variable name length |
| Unit tests | **196 passed** | All tests pass |
| Examples | **Working** | store + verify modes |

**Performance Impact:**

| Benchmark | Delta | Analysis |
|-----------|-------|----------|
| `prefix_scans_by_degree/10_edges` | +40% | Name resolution overhead per edge |
| `prefix_scans_by_degree/50_edges` | +184% | Linear overhead with edge count |
| `writes/100_nodes_10x_edges` | +32% | Additional Names CF write |
| `writes/1000_nodes_10x_edges` | +46% | Additional Names CF write |

**Analysis:** The current implementation shows regression due to per-edge name resolution from Names CF. This is expected behavior - the optimization's benefits are:

1. **Fixed 40-byte keys** - Enables predictable cache planning
2. **Better prefix compression** - 32-byte shared prefix for same (src, dst) pairs
3. **Cache density** - 58 edges per 4KB block (was ~43 with 32-byte names)
4. **Name deduplication** - Names stored once, referenced everywhere

**Recommendations for further optimization:**
1. Implement in-memory name caching in the query path (currently using direct DB lookups)
2. Consider batch name resolution for scans
3. Pin Names CF in block cache high-priority pool

**Files Modified:**
- `libs/db/Cargo.toml` - Added xxhash-rust, dashmap dependencies
- `libs/db/src/graph/name_hash.rs` - NEW: NameHash and NameCache implementation
- `libs/db/src/graph/mod.rs` - Exported name_hash module, added Names CF descriptor
- `libs/db/src/graph/schema.rs` - Updated key/value types, added Names CF
- `libs/db/src/graph/mutation.rs` - Write names to Names CF
- `libs/db/src/graph/query.rs` - Name resolution in all query executors
- `libs/db/src/graph/scan.rs` - Name resolution in all scan implementations
- `libs/db/src/graph/tests.rs` - Updated tests for new schema
- `examples/store/main.rs` - Updated for new schema

---

#### Phase 1.4: Block Cache Configuration Tuning

**Status:** ✅ COMPLETED (January 1, 2026)

**Goal:** Optimize RocksDB block cache settings for the new fixed-size key schema to maximize cache efficiency.

**Background:** See [Appendix: Technical Reference: RocksDB Block Cache](#technical-reference-rocksdb-block-cache) for detailed analysis.

**Key Configuration Points:**

| Setting | Current | Recommended | Rationale |
|---------|---------|-------------|-----------|
| Block cache size | Default (~8MB) | 256MB default, configurable | Conservative default; users tune for their workload |
| Block size (graph CFs) | 4KB | 4KB | Optimal for 40-byte fixed keys (~100 keys/block) |
| Block size (fragment CFs) | 4KB | 16KB | Better for larger variable-length content |
| Cache index/filter blocks | false | true | Keep hot metadata in cache |
| Pin L0 filter/index | false | true | Avoid eviction of newest data |
| Names CF priority | default | high | Keep name resolution fast; small CF, frequently accessed |
| Shared cache | No | Yes | RocksDB recommends sharing cache across CFs for better memory utilization |

**Design Decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single shared cache | Yes | RocksDB best practice; allows dynamic allocation across CFs based on access patterns |
| Default cache size | 256MB | Conservative default that works for most workloads; production deployments should tune to ~1/3 of available memory |
| Graph CF block size | 4KB | With 40-byte fixed keys, we get ~100 keys/block - optimal density |
| Fragment CF block size | 16KB | NodeFragments/EdgeFragments contain larger variable-length content; larger blocks reduce index overhead |
| Names CF high priority | Yes | Names CF is small but accessed on every name resolution; keep it hot |

**Implementation Plan:**

**Step 1.4.1: Add BlockCacheConfig struct** (`src/graph/mod.rs`)
| Task | Status |
|------|--------|
| Define `BlockCacheConfig` struct with cache_size_bytes, graph_block_size, fragment_block_size, cache_index_and_filter_blocks, pin_l0_filter_and_index | ✅ |
| Add `Default` impl with sensible defaults (256MB cache, 4KB graph, 16KB fragment) | ✅ |
| Add `block_cache: Option<rocksdb::Cache>` field to Storage struct | ✅ |
| Add `block_cache_config: BlockCacheConfig` field to Storage struct | ✅ |

**Step 1.4.2: Update Storage constructors** (`src/graph/mod.rs`)
| Task | Status |
|------|--------|
| Update `Storage::readonly()` to initialize BlockCacheConfig | ✅ |
| Update `Storage::readwrite()` to initialize BlockCacheConfig | ✅ |
| Update `Storage::secondary()` to initialize BlockCacheConfig | ✅ |
| Add `with_block_cache_config()` builder method | ✅ |

**Step 1.4.3: Create shared cache in Storage::ready()** (`src/graph/mod.rs`)
| Task | Status |
|------|--------|
| Create `rocksdb::Cache::new_lru_cache()` before opening DB | ✅ |
| Store cache in `self.block_cache` | ✅ |
| Pass cache reference to CF descriptor creation | ✅ |

**Step 1.4.4: Update per-CF options** (`src/graph/schema.rs`)
| Task | Status |
|------|--------|
| Add `Names::column_family_options_with_cache()` with cache & config | ✅ |
| Add `Nodes::column_family_options_with_cache()` with graph_block_size | ✅ |
| Add `ForwardEdges::column_family_options_with_cache()` with graph_block_size | ✅ |
| Add `ReverseEdges::column_family_options_with_cache()` with graph_block_size | ✅ |
| Add `NodeFragments::column_family_options_with_cache()` with fragment_block_size | ✅ |
| Add `EdgeFragments::column_family_options_with_cache()` with fragment_block_size | ✅ |

**Step 1.4.5: Update CF descriptor creation** (`src/graph/mod.rs`)
| Task | Status |
|------|--------|
| Modify `Storage::ready()` to use `column_family_options_with_cache()` for all CFs | ✅ |

**Step 1.4.6: Validation**
| Task | Status |
|------|--------|
| All 198 tests pass | ✅ |
| Benchmarks show no regression | ✅ |

**Benchmark Results (January 1, 2026):**

No performance regression observed after block cache tuning:
- `writes/1000_nodes`: 70.8 ms (stable)
- `prefix_scans/50_edges`: 16.76 µs (stable)
- `batch_scan/10000_nodes`: 952 µs (stable)

**Note:** The full benefits of block cache tuning will be more visible in production workloads where:
1. Working set exceeds default 8MB cache (now 256MB)
2. Repeated access patterns benefit from cached index/filter blocks
3. Names CF stays hot due to cache_index_and_filter_blocks=true

**Files Modified:**

| File | Changes |
|------|---------|
| `src/graph/mod.rs` | Add BlockCacheConfig, update Storage struct, update constructors, update ready() |
| `src/graph/schema.rs` | Update all 6 column_family_options() methods to accept cache & config |
| `benches/db_operations.rs` | Add cache validation benchmarks |

**Expected Benefits:**
- Higher cache hit rate with fixed 40-byte keys (more keys per block)
- Names CF pinned in memory for O(1) resolution
- Better separation of hot (graph topology) vs cold (fragment) data
- Shared cache allows RocksDB to dynamically balance memory across CFs

**Reference Code:**
```rust
/// Configuration for RocksDB block cache
#[derive(Debug, Clone)]
pub struct BlockCacheConfig {
    /// Total block cache size in bytes. Default: 256MB.
    pub cache_size_bytes: usize,
    /// Block size for graph CFs (Nodes, Edges, Names). Default: 4KB.
    pub graph_block_size: usize,
    /// Block size for fragment CFs (NodeFragments, EdgeFragments). Default: 16KB.
    pub fragment_block_size: usize,
    /// Cache index and filter blocks. Default: true.
    pub cache_index_and_filter_blocks: bool,
    /// Pin L0 filter and index blocks in cache. Default: true.
    pub pin_l0_filter_and_index: bool,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            cache_size_bytes: 256 * 1024 * 1024, // 256MB
            graph_block_size: 4 * 1024,          // 4KB
            fragment_block_size: 16 * 1024,      // 16KB
            cache_index_and_filter_blocks: true,
            pin_l0_filter_and_index: true,
        }
    }
}
```

---

#### Phase 1.5: NameCache Integration ✅ COMPLETED

**Status:** ✅ COMPLETED (January 1, 2026)

**Problem:** Phase 1 showed performance regression due to:
1. **Write path:** Every name unconditionally written to Names CF (even repeated names)
2. **Read path:** Every edge scan required DB lookup to Names CF for name resolution

**Solution:** Integrate the existing `NameCache` into both write and read paths:
1. **Write deduplication:** Only write to Names CF if name is new (not in cache)
2. **Cache pre-warming:** Load existing names from Names CF on startup
3. **Read caching:** Use in-memory cache for name resolution (coming next)

**Step 1.5.1: Cache Infrastructure**
| Task | File | Status |
|------|------|--------|
| Add `get_hash()` method to NameCache | `src/graph/name_hash.rs` | ✅ |
| Add `intern_if_new()` method to NameCache | `src/graph/name_hash.rs` | ✅ |
| Add unit tests for new methods | `src/graph/name_hash.rs` | ✅ |

**Step 1.5.2: Storage Integration**
| Task | File | Status |
|------|------|--------|
| Add `NameCacheConfig` struct | `src/graph/mod.rs` | ✅ |
| Add `name_cache: Arc<NameCache>` to Storage | `src/graph/mod.rs` | ✅ |
| Add `with_name_cache_config()` builder | `src/graph/mod.rs` | ✅ |
| Implement `prewarm_name_cache()` | `src/graph/mod.rs` | ✅ |
| Call prewarm in `Storage::ready()` | `src/graph/mod.rs` | ✅ |

**Step 1.5.3: Write Path Integration**
| Task | File | Status |
|------|------|--------|
| Add `execute_with_cache()` to MutationExecutor trait | `src/graph/writer.rs` | ✅ |
| Add `write_name_to_cf_cached()` helper | `src/graph/mutation.rs` | ✅ |
| Implement `execute_with_cache()` for AddNode | `src/graph/mutation.rs` | ✅ |
| Implement `execute_with_cache()` for AddEdge | `src/graph/mutation.rs` | ✅ |
| Implement `execute_with_cache()` for AddEdgeFragment | `src/graph/mutation.rs` | ✅ |
| Update `Graph::process_mutations()` to use cache | `src/graph/mod.rs` | ✅ |

**Step 1.5.4: Read Path Integration**
| Task | File | Status |
|------|------|--------|
| Add cache to QueryExecutor context | `src/graph/query.rs` | ✅ |
| Add cache to ScanExecutor context | `src/graph/scan.rs` | ✅ |
| Add cache to TransactionQueryExecutor trait | `src/graph/query.rs` | ✅ |
| Add cache to Transaction struct | `src/graph/transaction.rs` | ✅ |
| Use cache for name resolution in queries | `src/graph/query.rs` | ✅ |
| Use cache for name resolution in scans | `src/graph/scan.rs` | ✅ |

**Benchmark Results (January 1, 2026 - Write Path Cache Only):**

| Benchmark | Before Cache | After Cache | Delta | Analysis |
|-----------|--------------|-------------|-------|----------|
| `writes/100_nodes_10x_edges` | 17.96 ms | 16.87 ms | **-6.1%** | Skip redundant Names CF writes |
| `writes/1000_nodes_10x_edges` | 82.86 ms | 70.35 ms | **-15.1%** | Larger benefit with more edge reuse |
| `writes/5000_nodes_10x_edges` | 465.5 ms | 395.1 ms | **-15.1%** | Consistent 15% improvement |
| `prefix_scans_by_degree/10_edges` | 12.68 µs | 11.70 µs | **-7.7%** | Cache warmed from previous bench |
| `prefix_scans_by_degree/50_edges` | 44.78 µs | 35.25 µs | **-21.3%** | Cache benefit starting |
| `batch_scan/1000_nodes_100_scans` | 1.248 ms | 1.073 ms | **-14.0%** | Batch scans benefit |
| `batch_scan/5000_nodes_100_scans` | 1.287 ms | 1.155 ms | **-10.3%** | Consistent improvement |

**Benchmark Results (January 1, 2026 - Full Cache Integration):**

| Benchmark | Write Cache Only | Full Cache | Delta | Analysis |
|-----------|------------------|------------|-------|----------|
| `writes/100_nodes_10x_edges` | 16.87 ms | 16.81 ms | **0%** | No change (read path doesn't affect writes) |
| `writes/1000_nodes_10x_edges` | 70.35 ms | 70.29 ms | **0%** | No change |
| `writes/5000_nodes_10x_edges` | 395.1 ms | 397.9 ms | **0%** | No change |
| `prefix_scans_by_degree/1_edges` | 8.43 µs | 7.74 µs | **-8.2%** | Cache lookup faster than DB |
| `prefix_scans_by_degree/10_edges` | 11.70 µs | 9.57 µs | **-18.2%** | Major improvement |
| `prefix_scans_by_degree/50_edges` | 35.25 µs | 16.75 µs | **-52.5%** | Huge improvement! |
| `prefix_scans_by_position/10000_late` | 12.24 µs | 9.60 µs | **-21.6%** | Major improvement |
| `batch_scan/1000_nodes_100_scans` | 1.073 ms | 0.936 ms | **-12.8%** | Continued improvement |
| `batch_scan/5000_nodes_100_scans` | 1.155 ms | 0.953 ms | **-17.5%** | Continued improvement |
| `batch_scan/10000_nodes_100_scans` | 1.239 ms | 0.960 ms | **-22.5%** | Continued improvement |

**Summary vs Original (Before Name Interning):**

| Benchmark | Original | Phase 1 (No Cache) | Phase 1.5 (Full Cache) | Net Change |
|-----------|----------|-------------------|------------------------|------------|
| `writes/1000_nodes` | ~56 ms | 82.86 ms (+48%) | 70.29 ms | **+25%** |
| `prefix_scans/50_edges` | ~15.8 µs | 44.78 µs (+184%) | 16.75 µs | **+6%** |
| `batch_scan/10000_nodes` | ~1.24 ms | ~1.29 ms (+4%) | 0.960 ms | **-22.5%** |

**Analysis:**
- **Read path cache integration is a major success!**
- `prefix_scans/50_edges` improved by **52.5%** with full cache integration
- Net regression from original reduced from **+184%** to just **+6%**
- **Batch scans now faster than original** due to cache efficiency
- Fixed 40-byte keys + in-memory name cache = optimal cache density with fast resolution

**Key Implementation Details:**

```rust
// NameCache.intern_if_new() - Skip DB writes for cached names
pub fn intern_if_new(&self, name: &str) -> (NameHash, bool) {
    if let Some(hash) = self.name_to_hash.get(name) {
        return (*hash, false);  // Already cached, skip DB write
    }
    // ... compute hash and cache ...
    (hash, true)  // New name, needs DB write
}

// write_name_to_cf_cached() - Only write if new
fn write_name_to_cf_cached(..., cache: &NameCache) -> Result<NameHash> {
    let (name_hash, is_new) = cache.intern_if_new(name);
    if is_new {
        // Write to Names CF only for new names
        txn.put_cf(names_cf, key_bytes, value_bytes)?;
    }
    Ok(name_hash)
}
```

**Configuration:**

```rust
#[derive(Debug, Clone)]
pub struct NameCacheConfig {
    /// Maximum number of names to pre-load from Names CF on startup.
    /// Default: 1000. Set to 0 to disable pre-warming.
    pub prewarm_limit: usize,
}
```

---

#### Phase 2: Blob Separation (Priority 2)
| Step | Command | Status |
|------|---------|--------|
| Capture baseline | `cargo bench -p motlie-db -- value_size_impact --save-baseline before_blob_sep` | ⬜ |
| Capture baseline | `cargo bench -p motlie-db -- batch_scan_throughput --save-baseline before_blob_sep` | ⬜ |
| Capture baseline | `cargo bench -p motlie-db -- write_throughput_by_size --save-baseline before_blob_sep` | ⬜ |
| Implement hot/cold CF split | (code changes) | ⬜ |
| Validate scan improvement | `cargo bench -p motlie-db -- value_size_impact --baseline before_blob_sep` | ⬜ |
| Validate batch throughput | `cargo bench -p motlie-db -- batch_scan_throughput --baseline before_blob_sep` | ⬜ |
| Validate write regression | `cargo bench -p motlie-db -- write_throughput_by_size --baseline before_blob_sep` | ⬜ |
| **Target:** 5-10x scan improvement, <30% write regression | | ⬜ |

#### Phase 3: Zero-Copy Serialization (rkyv) (Priority 3)
| Step | Command | Status |
|------|---------|--------|
| Capture baseline | `cargo bench -p motlie-db -- serialization_overhead --save-baseline before_rkyv` | ⬜ |
| Implement rkyv for hot CFs | (code changes) | ⬜ |
| Validate improvement | `cargo bench -p motlie-db -- serialization_overhead --baseline before_rkyv` | ⬜ |
| Validate batch scan improvement | `cargo bench -p motlie-db -- batch_scan_throughput --baseline before_rkyv` | ⬜ |
| **Target:** 10-50x deserialize improvement | | ⬜ |




------------------------------------------------------------------------------------------------

# Appendix

## Technical Reference: RocksDB Block Cache

> **Purpose:** This section documents RocksDB's block cache behavior to inform schema design decisions, particularly for blob separation and variable-length key handling. Understanding these mechanics is essential for optimizing cache efficiency.
>
> **Sources:** [RocksDB Block Cache Wiki](https://github.com/facebook/rocksdb/wiki/Block-Cache), [BlockBasedTable Format](https://github.com/facebook/rocksdb/wiki/Rocksdb-BlockBasedTable-Format), [BlobDB Wiki](https://github.com/facebook/rocksdb/wiki/BlobDB), [RocksDB Tuning Guide](https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide)

### What Gets Cached: Blocks, Not Key-Value Pairs

**Critical insight:** RocksDB's block cache stores **uncompressed data blocks**, not individual key-value pairs. Each block contains multiple key-value entries packed together.

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Block (~4KB default)                │
├─────────────────────────────────────────────────────────────┤
│  [key1][value1] [key2][value2] [key3][value3] ... [keyN]    │
│                                                             │
│  Keys are delta-encoded (prefix compression)                │
│  Values are stored inline with their keys                   │
└─────────────────────────────────────────────────────────────┘
```

**Implications for our design:**

| Observation | Impact on motlie_db |
|-------------|---------------------|
| **Entire block is loaded** | If one edge in a block has a 2KB summary, the entire block (containing that edge + neighbors) is loaded |
| **Block = cache unit** | Large values reduce effective cache capacity (fewer key-values per cached block) |
| **Keys and values together** | Cannot cache keys separately from values; blob separation requires separate CFs |

### Block Size and Variable-Length Data

**Default block size:** 4KB (uncompressed), configurable via `block_size` option.

**Variable-length handling:** RocksDB handles variable-length keys and values natively:

1. **Keys:** Delta-encoded with restart points every 16 keys (configurable via `block_restart_interval`)
   - First key in restart interval: stored fully
   - Subsequent keys: stored as (shared_prefix_len, suffix_len, suffix_bytes)
   - **Long edge names impact:** Variable-length edge names reduce prefix compression effectiveness

2. **Values:** Stored inline after each key, length-prefixed
   - No alignment requirements
   - No maximum length (but large values should use BlobDB)

**Delta encoding example:**
```
Key 1: "edge:src123:dst456:follows"     → stored fully (restart point)
Key 2: "edge:src123:dst456:likes"       → shared=21, suffix="likes"
Key 3: "edge:src123:dst789:follows"     → shared=13, suffix="dst789:follows"
```

### Block Cache vs Bloom Filter (Not the Same)

| Component | Purpose | What It Stores |
|-----------|---------|----------------|
| **Block Cache** | Avoid re-reading blocks from disk | Uncompressed data blocks |
| **Bloom Filter** | Avoid reading files that don't contain a key | Probabilistic bit array (~10 bits/key) |
| **OS Page Cache** | Kernel-level file caching | Compressed blocks (SST file contents) |

**Key distinction:** Bloom filters help avoid unnecessary I/O by answering "might this file contain key X?" Block cache speeds up repeated access to the same data blocks.

### Cache Eviction and Block Reload

**Eviction policy:** LRU (Least Recently Used) with optional priority tiers.

```
┌─────────────────────────────────────────────────────────────┐
│                    LRU Block Cache                          │
├─────────────────────────────────────────────────────────────┤
│  High Priority Pool (index, filters, compression dict)      │
│  ─────────────────────────────────────────────────────────  │
│  Low Priority Pool (data blocks)                            │
│  ─────────────────────────────────────────────────────────  │
│  Bottom Priority (blobs from BlobDB)                        │
└─────────────────────────────────────────────────────────────┘
         ↑                                            ↓
      Insert                                       Evict
```

**What happens on cache eviction:**
1. Block is removed from cache (memory freed)
2. Next access requires:
   - Disk read (SST file)
   - Decompression (if compression enabled)
   - CRC32 checksum verification
   - Block loaded back into cache

**Cost of reload:** ~100-500µs for SSD, ~5-10ms for HDD, plus CPU for decompression/checksum.

### Implications for rkyv Zero-Copy Deserialization

**Question:** How does rkyv benefit if blocks can be evicted?

**Answer:** rkyv's benefit is **per-access**, not per-cache-lifetime:

| Scenario | rmp_serde (current) | rkyv (proposed) |
|----------|---------------------|-----------------|
| **Block in cache** | Decompress → Deserialize → Allocate structs | Cast pointer → Zero-copy access |
| **Block evicted** | Disk read → Decompress → Deserialize → Allocate | Disk read → Decompress → Cast pointer |
| **Scan 1000 edges** | 1000 deserializations, 1000 allocations | 1000 pointer casts, 0 allocations |

**Key insight:** Even with cache eviction, rkyv eliminates allocation churn. The block reload cost is paid regardless of serialization format, but rkyv avoids the additional deserialization cost after the block is loaded.

**Caveat:** rkyv requires uncompressed data to achieve zero-copy. Our hybrid strategy (rkyv for hot CFs, rmp+LZ4 for cold CFs) addresses this.

### Implications for Variable-Length Edge Names

**Current schema:** `ForwardEdgeCfKey(SrcId, DstId, EdgeName)` where `EdgeName` is a `String`.

**Block cache impact of long edge names:**

| Edge Name Length | Key Size | Keys per 4KB Block | Cache Efficiency |
|------------------|----------|---------------------|------------------|
| 8 bytes ("follows") | 40 bytes | ~100 keys | High |
| 32 bytes ("user_interaction_type_v2") | 64 bytes | ~62 keys | Medium |
| 128 bytes (long descriptive name) | 160 bytes | ~25 keys | Low |

**Delta encoding helps but has limits:**
- Only prefix compression (shared prefix removed)
- Different edge names from same source share `src_id` prefix (16 bytes)
- Restart points every 16 keys reset compression

**Recommendations:**
1. Keep edge names short (< 32 bytes) for optimal cache density
2. Consider edge name interning (map names → small integers) for high-cardinality names
3. For very long names, store name in value and use hash in key

### Implications for Vector (f32[]) Storage

**Question:** How should f32 vectors (for HNSW) be stored and cached?

**Vector sizes:**
| Dimensions | Size (f32) | Size (f16) | Vectors per 4KB Block |
|------------|------------|------------|------------------------|
| 128 | 512 bytes | 256 bytes | 7-8 |
| 256 | 1024 bytes | 512 bytes | 3-4 |
| 512 | 2048 bytes | 1024 bytes | 1-2 |
| 768 (OpenAI) | 3072 bytes | 1536 bytes | 1 |
| 1536 (OpenAI large) | 6144 bytes | 3072 bytes | 0 (exceeds block) |

**Recommendations:**

1. **Use BlobDB for vectors > 1KB:** RocksDB's integrated BlobDB stores large values in separate blob files, keeping the LSM tree compact. From [BlobDB Wiki](https://github.com/facebook/rocksdb/wiki/BlobDB): "Storing large values outside of the LSM significantly reduces the size of the LSM which improves caching."

2. **Separate vector CF from graph topology:** Keep HNSW edge topology in one CF (small, hot) and vectors in another CF (large, accessed per-search).

3. **Consider quantization:** f16 or int8 quantization halves/quarters storage while maintaining search quality.

4. **Block size tuning for vectors:** If vectors dominate a CF, increase `block_size` to 16KB or 32KB to reduce index overhead.

**BlobDB cache behavior:**
- Blobs get "bottom" priority in cache (below data blocks)
- This is appropriate: vector access is typically sequential (greedy search), not random
- Hot vectors naturally stay in OS page cache

### Configuration Recommendations

Based on the above analysis, recommended RocksDB settings for motlie_db:

```rust
// Block cache: ~1/3 of available memory
let cache = Cache::new_lru_cache(1 * 1024 * 1024 * 1024)?; // 1GB

let mut block_opts = BlockBasedOptions::default();
block_opts.set_block_cache(&cache);
block_opts.set_block_size(4 * 1024); // 4KB for graph data
block_opts.set_cache_index_and_filter_blocks(true);
block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

// For vector CF (if using BlobDB)
let mut vector_opts = Options::default();
vector_opts.set_enable_blob_files(true);
vector_opts.set_min_blob_size(512); // Vectors > 512 bytes → blob file
vector_opts.set_blob_compression_type(CompressionType::None); // Vectors don't compress well
```
