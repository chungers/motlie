# Unified Sharding and Routing Design

This document describes a unified sharding architecture that leverages semantic similarity
to co-locate related data across the vector, graph, and fulltext subsystems.

## Table of Contents

1. [Motivation](#motivation)
2. [Core Insight: Semantic Co-location](#core-insight-semantic-co-location)
3. [Architecture Overview](#architecture-overview)
4. [Deployment Model](#deployment-model)
5. [Live Rebalancing](#live-rebalancing)
6. [IVF Routing Layer](#ivf-routing-layer)
7. [Per-Shard Storage](#per-shard-storage)
8. [Multi-Index Tenancy](#multi-index-tenancy)
9. [Query Routing](#query-routing)
10. [Cross-Shard Operations](#cross-shard-operations)
11. [Storage Schema](#storage-schema)
12. [Implementation Phases](#implementation-phases)

---

## Motivation

### Single-Node Scale Limits

Based on observed benchmarks (see [PHASE8.md](../vector/docs/PHASE8.md)):

| Scale | Feasibility | Notes |
|-------|-------------|-------|
| 10M vectors | Practical | 64GB RAM, NVMe, M=16 |
| 50M vectors | Stretch | Requires cache tuning, async indexing |
| 100M+ vectors | Requires sharding | 128GB+ RAM or distributed |
| 1B vectors | Not single-node | Sharding/tiering mandatory |

### Why Unified Sharding?

Traditional approach shards each subsystem independently:
- Vectors by embedding similarity (IVF)
- Graph by ID hash or connectivity
- Fulltext by term distribution

**Problem**: Related data scattered across shards. A hybrid query like
"find similar documents AND their citations" requires cross-shard joins.

**Solution**: Partition by **semantic similarity** and co-locate all modalities.

---

## Core Insight: Semantic Co-location

### Observation: Data Correlation

| Relationship | Implication |
|--------------|-------------|
| Similar embeddings → related content | Vector neighbors likely graph neighbors |
| Graph-connected nodes → often similar topics | Co-locate graph clusters with vector clusters |
| Similar text → similar embeddings | Fulltext and vector locality align |

### Example: Documents about "machine learning"

```
Documents in same semantic cluster:
├── Have similar embeddings (cluster in vector space)
├── Often cite each other (graph edges)
├── Share terminology ("neural", "gradient", "model")
└── Should live on same shard
```

### Benefit: Local Hybrid Queries

```
Query: "Find papers similar to X that cite papers by author Y"

Without co-location:                    With co-location:
1. Vector search → shards A,B,C         1. Vector search → shards A,B (semantic)
2. For each result: graph lookup        2. Local graph traversal on each shard
   → scatter to random shards           3. Local author filter
3. Filter by author → more scatter      4. Merge results
Latency: O(results × shard_hops)        Latency: O(1) shard hops
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Unified Sharded Architecture                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Unified Router                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │
│  │  │ IVF Routing │  │ Term→Shard  │  │ Entity→Shard│               │ │
│  │  │ (centroids) │  │   Index     │  │   Cache     │               │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                     │
│          ▼                   ▼                   ▼                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐           │
│  │   Shard 0     │   │   Shard 1     │   │   Shard N     │           │
│  │  (Topic A)    │   │  (Topic B)    │   │  (Topic N)    │           │
│  ├───────────────┤   ├───────────────┤   ├───────────────┤           │
│  │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │           │
│  │ │  Vector   │ │   │ │  Vector   │ │   │ │  Vector   │ │           │
│  │ │ Processor │ │   │ │ Processor │ │   │ │ Processor │ │           │
│  │ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │           │
│  │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │           │
│  │ │   Graph   │ │   │ │   Graph   │ │   │ │   Graph   │ │           │
│  │ │  Storage  │ │   │ │  Storage  │ │   │ │  Storage  │ │           │
│  │ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │           │
│  │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │           │
│  │ │ Fulltext  │ │   │ │ Fulltext  │ │   │ │ Fulltext  │ │           │
│  │ │  Index    │ │   │ │  Index    │ │   │ │  Index    │ │           │
│  │ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │           │
│  │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │           │
│  │ │ Entities  │ │   │ │ Entities  │ │   │ │ Entities  │ │           │
│  │ │ (source)  │ │   │ │ (source)  │ │   │ │ (source)  │ │           │
│  │ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │           │
│  └───────────────┘   └───────────────┘   └───────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Principle: Each Shard is Complete

Each shard contains the **same subsystem implementations** currently in use:
- `vector::Processor` - unchanged HNSW + RaBitQ
- `graph::Graph` - unchanged RocksDB storage with 8 CFs
- `fulltext::Index` - unchanged Tantivy index

The sharding layer sits **above** these subsystems, not within them.

---

## Deployment Model

### Two-Binary Architecture

The simplest deployment separates routing from storage:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Production Deployment                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Router Binary                               │   │
│  │                    (motlie-router)                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              Router's Own RocksDB                        │    │   │
│  │  │  • routing/centroids     (IVF cluster centers)          │    │   │
│  │  │  • routing/shard_map     (centroid → shard assignment)  │    │   │
│  │  │  • routing/entity_shard  (entity → shard cache)         │    │   │
│  │  │  • routing/migrations    (in-progress rebalances)       │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│              gRPC/HTTP to shard nodes (public API)                     │
│                              │                                          │
│     ┌────────────────────────┼────────────────────────┐                │
│     ▼                        ▼                        ▼                │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│  │ Node Binary  │     │ Node Binary  │     │ Node Binary  │           │
│  │ (motlie-db)  │     │ (motlie-db)  │     │ (motlie-db)  │           │
│  │              │     │              │     │              │           │
│  │  Shard 0     │     │  Shard 1     │     │  Shard 2     │           │
│  │  ┌────────┐  │     │  ┌────────┐  │     │  ┌────────┐  │           │
│  │  │Vector  │  │     │  │Vector  │  │     │  │Vector  │  │           │
│  │  │Graph   │  │     │  │Graph   │  │     │  │Graph   │  │           │
│  │  │Fulltext│  │     │  │Fulltext│  │     │  │Fulltext│  │           │
│  │  └────────┘  │     │  └────────┘  │     │  └────────┘  │           │
│  └──────────────┘     └──────────────┘     └──────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Component | Binary | Changes Required |
|-----------|--------|------------------|
| **Node (Shard)** | `motlie-db` | **None** - existing single-node binary unchanged |
| **Router** | `motlie-router` | **New** - routing logic + own RocksDB |
| **Communication** | gRPC/HTTP | Uses existing public APIs of each crate |

### Node Binary (Unchanged)

Each node is the **current single-node implementation**:

```rust
// Existing public API - no changes needed
impl NodeService {
    // Vector operations
    pub async fn insert_vector(&self, req: InsertVectorRequest) -> Result<InsertVectorResponse>;
    pub async fn insert_batch(&self, req: InsertBatchRequest) -> Result<InsertBatchResponse>;
    pub async fn search(&self, req: SearchRequest) -> Result<SearchResponse>;
    pub async fn delete_vector(&self, req: DeleteVectorRequest) -> Result<DeleteVectorResponse>;
    pub async fn get_vector(&self, req: GetVectorRequest) -> Result<GetVectorResponse>;

    // Graph operations
    pub async fn add_node(&self, req: AddNodeRequest) -> Result<AddNodeResponse>;
    pub async fn add_edge(&self, req: AddEdgeRequest) -> Result<AddEdgeResponse>;
    pub async fn query_outgoing(&self, req: OutgoingEdgesRequest) -> Result<EdgesResponse>;
    pub async fn query_incoming(&self, req: IncomingEdgesRequest) -> Result<EdgesResponse>;

    // Fulltext operations
    pub async fn index_document(&self, req: IndexDocumentRequest) -> Result<IndexResponse>;
    pub async fn search_text(&self, req: TextSearchRequest) -> Result<TextSearchResponse>;
}
```

### Router Binary (New)

The router is a **separate process** with its own storage:

```rust
pub struct Router {
    // Router's own RocksDB for routing metadata
    routing_db: Arc<RocksDB>,

    // Connections to shard nodes (via public API)
    shard_clients: Vec<ShardClient>,

    // In-memory routing state (loaded from routing_db)
    centroids: Vec<Vec<f32>>,
    shard_map: Vec<ShardId>,
    entity_cache: LruCache<EntityId, ShardId>,

    // Migration state
    active_migrations: HashMap<MigrationId, MigrationState>,
}

impl Router {
    /// Route insert to appropriate shard
    pub async fn insert(&self, entity: Entity) -> Result<EntityId> {
        // 1. Compute nearest centroid
        let centroid_id = self.find_nearest_centroid(&entity.embedding);
        let shard_id = self.shard_map[centroid_id];

        // 2. Forward to shard using its public API
        let client = &self.shard_clients[shard_id];
        let result = client.insert_vector(InsertVectorRequest {
            embedding: entity.embedding.clone(),
            id: entity.id,
            vector: entity.vector.clone(),
        }).await?;

        // 3. Cache entity→shard mapping
        self.entity_cache.insert(entity.id, shard_id);
        self.routing_db.put(entity_shard_key(entity.id), shard_id)?;

        Ok(result.entity_id)
    }

    /// Fan out search to relevant shards
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        // 1. Find shards to query (IVF routing)
        let shard_ids = self.route_by_embedding(&query.embedding, query.nprobe);

        // 2. Parallel search on each shard (using public API)
        let futures: Vec<_> = shard_ids.iter()
            .map(|&shard_id| {
                let client = &self.shard_clients[shard_id];
                client.search(SearchRequest {
                    embedding: query.embedding.clone(),
                    k: query.k,
                    ef_search: query.ef_search,
                })
            })
            .collect();

        let shard_results = futures::future::try_join_all(futures).await?;

        // 3. Merge results
        self.merge_search_results(shard_results, query.k)
    }
}
```

### Scaling: Adding More Nodes

```
Initial: 1 node (no router needed, direct access)
┌──────────────┐
│   Node 0     │  ← Client connects directly
│  (all data)  │
└──────────────┘

Growth: Add router + split to 2 nodes
┌──────────────┐
│    Router    │  ← Client connects to router
└──────┬───────┘
       │
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌──────┐
│Node 0│ │Node 1│  ← Data rebalanced across nodes
└──────┘ └──────┘

Scale: Add more nodes as needed
┌──────────────┐
│    Router    │
└──────┬───────┘
       │
   ┌───┼───┬───┐
   ▼   ▼   ▼   ▼
┌────┐┌────┐┌────┐┌────┐
│ N0 ││ N1 ││ N2 ││ N3 │
└────┘└────┘└────┘└────┘
```

---

## Live Rebalancing

### Challenge: Continuous Service During Migration

Rebalancing must support:
- **Inserts continue** - new data routed correctly
- **Searches continue** - results from both old and new locations
- **No data loss** - atomic cutover
- **Incremental** - migrate partition by partition

### Rebalancing Strategy: Dual-Write + Background Copy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Live Rebalancing Workflow                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: PREPARE                                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ • Select partition P to migrate (shard 0 → shard 2)            │    │
│  │ • Mark partition P as "migrating" in routing_db                │    │
│  │ • Record: source_shard=0, dest_shard=2, status=MIGRATING       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  Phase 2: DUAL-WRITE (new inserts go to both)                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Router behavior for partition P:                                │    │
│  │ • INSERT: Write to BOTH shard 0 AND shard 2                    │    │
│  │ • SEARCH: Query BOTH shards, deduplicate results               │    │
│  │ • DELETE: Delete from BOTH shards                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  Phase 3: BACKGROUND COPY (existing data)                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Background task (using public APIs):                            │    │
│  │ • Read entities from shard 0 (source) via get_vector/query     │    │
│  │ • Write to shard 2 (dest) via insert_vector                    │    │
│  │ • Track progress: copied_count, last_id                        │    │
│  │ • Rate limit to avoid overloading shards                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  Phase 4: CUTOVER (atomic routing switch)                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ • Verify all entities copied (count matches)                   │    │
│  │ • Update shard_map: partition P → shard 2 (atomic)             │    │
│  │ • Mark migration COMPLETE                                      │    │
│  │ • Stop dual-write, route only to shard 2                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  Phase 5: CLEANUP (garbage collection)                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ • Background: Delete partition P data from shard 0             │    │
│  │ • Uses public delete API                                       │    │
│  │ • Can be deferred/batched for efficiency                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Router State Machine for Migrations

```rust
#[derive(Clone, Debug)]
pub enum MigrationPhase {
    Preparing,
    DualWrite,      // New writes go to both shards
    Copying,        // Background copy in progress
    Verifying,      // Verifying copy completeness
    CuttingOver,    // Atomic routing switch
    Cleaning,       // Deleting from source
    Complete,
    Failed(String),
}

pub struct Migration {
    id: MigrationId,
    partition_id: PartitionId,
    source_shard: ShardId,
    dest_shard: ShardId,
    phase: MigrationPhase,

    // Progress tracking
    total_entities: u64,
    copied_entities: u64,
    last_copied_id: Option<EntityId>,

    // Timestamps
    started_at: Timestamp,
    phase_started_at: Timestamp,
}

impl Router {
    /// Handle insert during migration
    async fn insert_with_migration(&self, entity: Entity, partition: PartitionId) -> Result<EntityId> {
        if let Some(migration) = self.active_migrations.get(&partition) {
            match migration.phase {
                MigrationPhase::DualWrite | MigrationPhase::Copying | MigrationPhase::Verifying => {
                    // Write to BOTH shards
                    let source_future = self.shard_clients[migration.source_shard]
                        .insert_vector(entity.clone().into());
                    let dest_future = self.shard_clients[migration.dest_shard]
                        .insert_vector(entity.clone().into());

                    // Both must succeed
                    let (source_result, dest_result) = tokio::try_join!(source_future, dest_future)?;
                    Ok(source_result.entity_id)
                }
                MigrationPhase::CuttingOver | MigrationPhase::Cleaning | MigrationPhase::Complete => {
                    // Route to destination only
                    self.shard_clients[migration.dest_shard]
                        .insert_vector(entity.into())
                        .await
                        .map(|r| r.entity_id)
                }
                _ => {
                    // Route to source (migration not yet active)
                    self.shard_clients[migration.source_shard]
                        .insert_vector(entity.into())
                        .await
                        .map(|r| r.entity_id)
                }
            }
        } else {
            // No migration, normal routing
            let shard_id = self.route_partition(partition);
            self.shard_clients[shard_id].insert_vector(entity.into()).await.map(|r| r.entity_id)
        }
    }

    /// Handle search during migration
    async fn search_with_migration(&self, query: SearchQuery, partitions: Vec<PartitionId>) -> Result<Vec<SearchResult>> {
        let mut shard_ids = HashSet::new();

        for partition in &partitions {
            if let Some(migration) = self.active_migrations.get(partition) {
                match migration.phase {
                    MigrationPhase::DualWrite | MigrationPhase::Copying | MigrationPhase::Verifying => {
                        // Query BOTH shards during migration
                        shard_ids.insert(migration.source_shard);
                        shard_ids.insert(migration.dest_shard);
                    }
                    _ => {
                        shard_ids.insert(self.route_partition(*partition));
                    }
                }
            } else {
                shard_ids.insert(self.route_partition(*partition));
            }
        }

        // Fan out and deduplicate
        let results = self.fan_out_search(&query, shard_ids.into_iter().collect()).await?;
        self.deduplicate_results(results, query.k)
    }
}
```

### Background Copy Using Public APIs

```rust
impl Router {
    /// Background task to copy partition data
    async fn copy_partition(
        &self,
        migration: &Migration,
        batch_size: usize,
        rate_limit: RateLimiter,
    ) -> Result<()> {
        let source = &self.shard_clients[migration.source_shard];
        let dest = &self.shard_clients[migration.dest_shard];

        let mut cursor: Option<EntityId> = migration.last_copied_id;

        loop {
            // Rate limiting to avoid overwhelming shards
            rate_limit.acquire().await;

            // 1. Read batch from source using public API
            let batch = source.list_entities(ListEntitiesRequest {
                partition: migration.partition_id,
                after_id: cursor,
                limit: batch_size,
            }).await?;

            if batch.entities.is_empty() {
                break; // Done
            }

            // 2. For each entity, read full data and write to destination
            for entity_meta in &batch.entities {
                // Read vector
                let vector = source.get_vector(GetVectorRequest {
                    id: entity_meta.id,
                }).await?;

                // Read graph data (if applicable)
                let outgoing_edges = source.query_outgoing(OutgoingEdgesRequest {
                    node_id: entity_meta.id,
                }).await?;

                // Write to destination
                dest.insert_vector(InsertVectorRequest {
                    id: entity_meta.id,
                    embedding: vector.embedding,
                    vector: vector.data,
                }).await?;

                // Copy edges
                for edge in outgoing_edges.edges {
                    dest.add_edge(AddEdgeRequest {
                        source_id: edge.source_id,
                        target_id: edge.target_id,
                        name: edge.name,
                        weight: edge.weight,
                    }).await?;
                }

                cursor = Some(entity_meta.id);
            }

            // 3. Checkpoint progress
            self.update_migration_progress(migration.id, cursor, batch.entities.len()).await?;
        }

        Ok(())
    }
}
```

### Rebalancing Triggers

```rust
pub enum RebalanceStrategy {
    /// Manual: operator initiates
    Manual {
        source_shard: ShardId,
        dest_shard: ShardId,
        partitions: Vec<PartitionId>,
    },

    /// Automatic: based on shard size skew
    SizeBalance {
        max_skew_ratio: f64,  // e.g., 1.5 = largest shard 50% bigger than smallest
    },

    /// Automatic: based on query load
    LoadBalance {
        window_seconds: u64,
        max_qps_skew_ratio: f64,
    },

    /// Split: add new shard, redistribute
    Split {
        new_shard_endpoint: String,
    },
}

impl Router {
    /// Check if rebalancing is needed
    pub async fn check_rebalance_needed(&self) -> Option<RebalanceStrategy> {
        let stats = self.collect_shard_stats().await;

        // Check size skew
        let max_size = stats.iter().map(|s| s.total_bytes).max().unwrap_or(0);
        let min_size = stats.iter().map(|s| s.total_bytes).min().unwrap_or(1);

        if max_size as f64 / min_size as f64 > self.config.max_size_skew {
            return Some(RebalanceStrategy::SizeBalance {
                max_skew_ratio: self.config.max_size_skew,
            });
        }

        // Check QPS skew
        let max_qps = stats.iter().map(|s| s.qps).max().unwrap_or(0.0);
        let avg_qps = stats.iter().map(|s| s.qps).sum::<f64>() / stats.len() as f64;

        if max_qps > avg_qps * self.config.max_qps_skew {
            return Some(RebalanceStrategy::LoadBalance {
                window_seconds: 300,
                max_qps_skew_ratio: self.config.max_qps_skew,
            });
        }

        None
    }
}
```

### Failure Recovery

```rust
impl Router {
    /// Recover from router restart during migration
    pub async fn recover_migrations(&mut self) -> Result<()> {
        // Load active migrations from routing_db
        let migrations = self.routing_db.scan_prefix(b"migration/")?;

        for migration in migrations {
            match migration.phase {
                MigrationPhase::Preparing => {
                    // Restart preparation
                    self.start_migration(migration).await?;
                }
                MigrationPhase::DualWrite | MigrationPhase::Copying => {
                    // Resume copying from last checkpoint
                    self.resume_copy(migration).await?;
                }
                MigrationPhase::Verifying => {
                    // Re-verify and proceed
                    self.verify_and_cutover(migration).await?;
                }
                MigrationPhase::CuttingOver => {
                    // Complete the cutover
                    self.complete_cutover(migration).await?;
                }
                MigrationPhase::Cleaning => {
                    // Resume cleanup
                    self.cleanup_source(migration).await?;
                }
                MigrationPhase::Complete => {
                    // Remove from active
                }
                MigrationPhase::Failed(_) => {
                    // Log and alert, manual intervention needed
                }
            }
        }

        Ok(())
    }
}
```

### Consistency Guarantees

| Operation | During Migration | Guarantee |
|-----------|------------------|-----------|
| Insert | Dual-write to both shards | No data loss |
| Search | Query both, deduplicate | Correct results |
| Delete | Delete from both | Consistent removal |
| Update | Apply to both | Eventual consistency |

### Performance Impact During Migration

| Metric | Normal | During Migration |
|--------|--------|------------------|
| Insert latency | 1x | ~2x (dual-write) |
| Search latency | 1x | ~1.5x (extra shard) |
| Background bandwidth | 0 | Configurable (rate-limited) |

---

## IVF Routing Layer

### Centroid-Based Partitioning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         IVF Router                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Centroids: k-means cluster centers (stored, not derived)               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                      │
│  │ C0  │ │ C1  │ │ C2  │ │ C3  │ │ ... │ │ Ck  │                      │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └─────┘ └──┬──┘                      │
│     │       │       │       │               │                          │
│     ▼       ▼       ▼       ▼               ▼                          │
│  Shard 0  Shard 1  Shard 0  Shard 2      Shard N                       │
│                                                                         │
│  Partition Assignment: centroid_id → shard_id (many-to-one)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Centroid Training

```rust
/// Train IVF centroids for an index
pub fn train_centroids(
    sample_vectors: &[Vec<f32>],
    num_centroids: usize,      // Typically sqrt(N) to 4*sqrt(N)
    distance: Distance,
    iterations: usize,
) -> Vec<Vec<f32>> {
    // K-means clustering
    // Returns centroid vectors
}
```

**Guidelines**:
- Number of centroids: `sqrt(N)` to `4 * sqrt(N)` where N = total vectors
- 1M vectors → 1000-4000 centroids
- 100M vectors → 10K-40K centroids
- 1B vectors → 32K-128K centroids

### Centroid Storage

Centroids are **stored in RocksDB**, not derived at startup:

```rust
// Column family for routing metadata
CF "routing/centroids"
  Key:   [index_id: u64] + [centroid_id: u32]
  Value: Vec<f32>  // centroid vector

CF "routing/shard_map"
  Key:   [index_id: u64]
  Value: ShardMap {
      num_centroids: u32,
      num_shards: u32,
      centroid_to_shard: Vec<u16>,  // centroid_id → shard_id
      shard_stats: Vec<ShardStats>,
  }
```

### Why Stored, Not Derived?

1. **K-means is expensive** - cannot recompute on every startup
2. **Partition stability** - changing centroids requires reassigning all vectors
3. **Routing consistency** - inserts and queries must use same centroids

---

## Per-Shard Storage

### Shard Structure

Each shard is an independent instance of current subsystems:

```rust
pub struct UnifiedShard {
    shard_id: ShardId,

    // Vector subsystem (current Processor - unchanged)
    // Uses: vector/embedding_specs, vector/vectors, vector/edges,
    //       vector/binary_codes, vector/vec_meta, vector/graph_meta,
    //       vector/id_forward, vector/id_reverse, vector/id_alloc, vector/pending
    vectors: vector::Processor,

    // Graph subsystem (current Graph - unchanged)
    // Uses: graph/names, graph/nodes, graph/forward_edges, graph/reverse_edges,
    //       graph/node_summaries, graph/edge_summaries,
    //       graph/node_fragments, graph/edge_fragments
    graph: graph::Graph,

    // Fulltext subsystem (current Index - unchanged)
    // Uses: Tantivy index directory
    fulltext: fulltext::Index,

    // Shared entity storage (source of truth)
    entities: EntityStorage,
}
```

### Storage Layout Options

#### Option A: Shared RocksDB with Shard Prefix

```
Single RocksDB instance, keys prefixed by shard:
  [shard_id: u16] + [original_key]

Example:
  Key: [0x0001] + [vector/vectors] + [embedding: u64] + [vec_id: u32]
       ~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       shard 1            original vector key

Pro: Single process, simpler deployment
Con: Compaction contention between shards
```

#### Option B: Separate RocksDB per Shard

```
/data/shard_0/rocksdb/   → vector::, graph:: CFs
/data/shard_0/tantivy/   → fulltext index
/data/shard_1/rocksdb/
/data/shard_1/tantivy/
...

Pro: Full isolation, independent compaction
Con: More file handles, memory overhead
```

#### Option C: Hybrid (Recommended)

```
Small indexes: Shared RocksDB with prefix
Large indexes: Dedicated RocksDB per shard

Threshold: ~10M entities per index triggers dedicated storage
```

---

## Multi-Index Tenancy

### Per-Index Routing State

Each index maintains independent routing:

```rust
pub struct IndexRouter {
    index_id: IndexId,

    // IVF routing (per-index)
    centroids: Vec<Vec<f32>>,
    shard_assignments: Vec<ShardId>,

    // Term routing (per-index)
    term_shard_hints: BloomFilter<ShardId>,  // term → likely shards

    // Entity location cache (per-index)
    entity_shard_cache: LruCache<EntityId, ShardId>,
}

pub struct MultiTenantRouter {
    // Index-specific routers
    index_routers: HashMap<IndexId, IndexRouter>,

    // Shared shard connections
    shards: Vec<Arc<UnifiedShard>>,
}
```

### Multi-Index Storage Schema

```rust
// Routing metadata CF (shared across indexes)
CF "routing/index_meta"
  Key:   [index_id: u64]
  Value: IndexMeta {
      created_at: u64,
      num_centroids: u32,
      num_shards: u32,
      dimension: u32,
      distance: Distance,
      total_entities: u64,
  }

CF "routing/centroids"
  Key:   [index_id: u64] + [centroid_id: u32]
  Value: Vec<f32>

CF "routing/shard_map"
  Key:   [index_id: u64]
  Value: Vec<ShardId>  // centroid_id → shard_id
```

### Index Isolation Considerations

| Concern | Mitigation |
|---------|------------|
| Large index affects small index | Option B/C storage, separate compaction |
| Centroid count varies by index | Per-index centroid storage |
| Different dimensions per index | Per-index EmbeddingSpec (existing) |
| Rebalancing one index | Independent shard reassignment |

---

## Query Routing

### Unified Query Router

```rust
impl MultiTenantRouter {
    /// Route any query type to relevant shards
    pub fn route(&self, index_id: IndexId, query: &Query) -> Vec<ShardId> {
        let router = &self.index_routers[&index_id];

        match query {
            // Vector search: IVF routing via centroids
            Query::Vector { embedding, nprobe, .. } => {
                router.route_by_embedding(embedding, nprobe)
            }

            // Graph traversal: entity's home shard
            Query::GraphTraversal { node_id, .. } => {
                router.route_by_entity(node_id)
            }

            // Fulltext search: term-based routing
            Query::Fulltext { terms, .. } => {
                router.route_by_terms(terms)
            }

            // Hybrid: intersect routing decisions
            Query::Hybrid { embedding, terms, nprobe, .. } => {
                let vector_shards = router.route_by_embedding(embedding, nprobe);
                let text_shards = router.route_by_terms(terms);
                intersect_or_union(vector_shards, text_shards, query.strategy)
            }
        }
    }
}
```

### Vector Query Routing

```rust
impl IndexRouter {
    /// IVF routing: find nprobe nearest centroids
    pub fn route_by_embedding(&self, query: &[f32], nprobe: usize) -> Vec<ShardId> {
        // Compute distances to all centroids (cheap - only thousands)
        let distances: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(query, c, self.distance)))
            .collect();

        // Find top nprobe nearest centroids
        let mut heap = BinaryHeap::with_capacity(nprobe);
        for (centroid_id, dist) in distances {
            // ... top-k selection
        }

        // Map centroids to shards (dedup)
        heap.into_iter()
            .map(|(centroid_id, _)| self.shard_assignments[centroid_id])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
}
```

### Graph Query Routing

```rust
impl IndexRouter {
    /// Route graph query to entity's home shard
    pub fn route_by_entity(&self, entity_id: EntityId) -> Vec<ShardId> {
        // Check cache first
        if let Some(shard) = self.entity_shard_cache.get(&entity_id) {
            return vec![*shard];
        }

        // Entity shard determined by its embedding at insert time
        // If unknown, must fan out or lookup in routing table
        self.lookup_entity_shard(entity_id)
    }

    /// Multi-hop traversal may cross shards
    pub fn route_traversal(
        &self,
        start: EntityId,
        hops: usize,
    ) -> TraversalPlan {
        // First hop: start node's shard
        // Subsequent hops: may need to follow edges to other shards
        // Returns execution plan with shard ordering
    }
}
```

### Fulltext Query Routing

```rust
impl IndexRouter {
    /// Route fulltext query by terms
    pub fn route_by_terms(&self, terms: &[String]) -> Vec<ShardId> {
        // Option 1: Global term index (small, replicated)
        // Maps term → shards containing documents with that term

        // Option 2: Bloom filter hints
        // Probabilistic: may have false positives, never false negatives

        let mut candidate_shards = HashSet::new();
        for term in terms {
            if let Some(shards) = self.term_shard_hints.get(term) {
                candidate_shards.extend(shards);
            } else {
                // Unknown term: must check all shards
                return (0..self.num_shards).collect();
            }
        }
        candidate_shards.into_iter().collect()
    }
}
```

---

## Cross-Shard Operations

### Graph Edge Handling

Edges may connect nodes on different shards:

```
Node A (shard 0) ───edge───► Node B (shard 1)
```

#### Storage Strategy: Source-Based + Optional Reverse Index

```rust
// Edge stored on SOURCE node's shard (outgoing traversal is local)
Shard 0:
  graph/forward_edges: [src_id=A, dst_id=B, name_hash] → EdgeValue

// Reverse index options:

// Option 1: Reverse edge on destination shard (incoming traversal local)
Shard 1:
  graph/reverse_edges: [dst_id=B, src_id=A, name_hash] → ReverseEdgeValue

// Option 2: Global reverse edge index (smaller, query-only)
Routing DB:
  routing/reverse_edge_index: [dst_id] → [(src_id, src_shard)]
```

#### Trade-offs

| Strategy | Outgoing Traversal | Incoming Traversal | Storage |
|----------|-------------------|-------------------|---------|
| Source-only | Local | Fan-out all shards | 1x |
| Bidirectional | Local | Local | 2x edges |
| Source + global reverse | Local | Lookup + targeted | 1x + index |

**Recommendation**: Bidirectional replication for small-medium graphs,
source + global reverse index for billion-scale.

### Cross-Shard Query Execution

```rust
impl UnifiedCoordinator {
    /// Execute query across shards
    pub async fn execute(&self, query: Query) -> Result<QueryResult> {
        // 1. Route to relevant shards
        let shards = self.router.route(query.index_id, &query);

        // 2. Fan out query to shards (parallel)
        let futures: Vec<_> = shards.iter()
            .map(|shard_id| {
                let shard = &self.shards[*shard_id];
                self.execute_on_shard(shard, query.clone())
            })
            .collect();

        let shard_results = futures::future::join_all(futures).await;

        // 3. Merge results
        self.merge_results(shard_results, &query)
    }

    /// Merge results from multiple shards
    fn merge_results(
        &self,
        shard_results: Vec<ShardResult>,
        query: &Query,
    ) -> QueryResult {
        match query {
            Query::Vector { k, .. } => {
                // Merge top-k by distance
                let mut all_results: Vec<_> = shard_results
                    .into_iter()
                    .flat_map(|r| r.vector_results)
                    .collect();
                all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                all_results.truncate(*k);
                QueryResult::Vector(all_results)
            }
            Query::Fulltext { limit, .. } => {
                // Merge by BM25 score
                let mut all_results: Vec<_> = shard_results
                    .into_iter()
                    .flat_map(|r| r.fulltext_results)
                    .collect();
                all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                all_results.truncate(*limit);
                QueryResult::Fulltext(all_results)
            }
            // ... other query types
        }
    }
}
```

---

## Storage Schema

### Current Subsystem Column Families (Unchanged)

#### Vector Subsystem (10 CFs)
```
vector/embedding_specs   [embedding_code] → EmbeddingSpec
vector/vectors           [embedding_code, vec_id] → Vec<f32>
vector/edges             [embedding_code, vec_id, layer] → RoaringBitmap
vector/binary_codes      [embedding_code, vec_id] → BinaryCode + ADC
vector/vec_meta          [embedding_code, vec_id] → VecMetadata
vector/graph_meta        [embedding_code, field] → GraphMetaValue
vector/id_forward        [embedding_code, ulid] → vec_id
vector/id_reverse        [embedding_code, vec_id] → ulid
vector/id_alloc          [embedding_code, field] → AllocState
vector/pending           [embedding_code, timestamp, vec_id] → ()
```

#### Graph Subsystem (8 CFs)
```
graph/names              [name_hash] → String
graph/nodes              [node_id] → NodeCfValue
graph/forward_edges      [src_id, dst_id, name_hash] → ForwardEdgeCfValue
graph/reverse_edges      [dst_id, src_id, name_hash] → ReverseEdgeCfValue
graph/node_summaries     [summary_hash] → NodeSummary
graph/edge_summaries     [summary_hash] → EdgeSummary
graph/node_fragments     [node_id, timestamp] → FragmentValue
graph/edge_fragments     [src_id, dst_id, name_hash, timestamp] → FragmentValue
```

#### Fulltext Subsystem (Tantivy)
```
Tantivy index directory with fields:
  id, src_id, dst_id, node_name, edge_name, content,
  creation_timestamp, valid_since, valid_until, doc_type,
  weight, doc_type_facet, tags_facet, validity_facet
```

### New Routing Layer CFs

```rust
// Routing metadata (single shared RocksDB or dedicated routing DB)
CF "routing/index_meta"
    Key:   [index_id: u64]
    Value: IndexMeta

CF "routing/centroids"
    Key:   [index_id: u64] + [centroid_id: u32]
    Value: Vec<f32>

CF "routing/shard_map"
    Key:   [index_id: u64]
    Value: ShardMap { centroid_to_shard: Vec<ShardId>, ... }

CF "routing/entity_shard"
    Key:   [index_id: u64] + [entity_id: 16]
    Value: shard_id: u16

CF "routing/term_hints"
    Key:   [index_id: u64] + [term_hash: u64]
    Value: BloomFilter<ShardId>  // probabilistic shard hints

// Optional: Global reverse edge index (for incoming traversal)
CF "routing/reverse_edge_index"
    Key:   [index_id: u64] + [dst_id: 16]
    Value: Vec<(src_id, src_shard)>
```

### Sharded Key Format

When using shared RocksDB with shard prefix:

```
Original key: [cf_prefix] + [key_bytes]
Sharded key:  [shard_id: u16] + [cf_prefix] + [key_bytes]

Example:
  Original: vector/vectors + [embedding: u64] + [vec_id: u32]
  Sharded:  [0x0001] + vector/vectors + [embedding: u64] + [vec_id: u32]
```

---

## Implementation Phases

### Phase 1: Foundation (Current)

```
✓ Vector subsystem with HNSW + RaBitQ
✓ Graph subsystem with bidirectional edges
✓ Fulltext subsystem with Tantivy
✓ Single-node deployment up to ~10M entities
```

### Phase 2: Routing Layer

```
[ ] Define routing metadata schema
[ ] Implement IVF centroid training
[ ] Implement centroid storage/loading
[ ] Add entity→shard assignment on insert
[ ] Implement unified router interface
```

### Phase 3: Multi-Shard Coordination

```
[ ] Implement shard-prefixed key encoding
[ ] Add query fan-out and result merging
[ ] Implement cross-shard graph traversal
[ ] Add term→shard hints for fulltext
```

### Phase 4: Multi-Index Tenancy

```
[ ] Per-index routing state management
[ ] Index isolation (storage separation)
[ ] Independent rebalancing per index
[ ] Resource quotas and fairness
```

### Phase 5: Production Hardening

```
[ ] Shard health monitoring
[ ] Automatic rebalancing on skew
[ ] Graceful shard migration
[ ] Distributed deployment (optional)
```

---

## API Sketch

### Unified Shard Manager

```rust
pub struct ShardManager {
    router: MultiTenantRouter,
    shards: Vec<Arc<UnifiedShard>>,
    config: ShardConfig,
}

impl ShardManager {
    /// Create new index with initial sharding
    pub async fn create_index(
        &self,
        index_id: IndexId,
        spec: IndexSpec,
        initial_shards: usize,
    ) -> Result<()>;

    /// Insert entity (routes to appropriate shard)
    pub async fn insert(
        &self,
        index_id: IndexId,
        entity: Entity,
    ) -> Result<EntityId>;

    /// Batch insert with single centroid computation
    pub async fn insert_batch(
        &self,
        index_id: IndexId,
        entities: Vec<Entity>,
    ) -> Result<Vec<EntityId>>;

    /// Unified search across modalities
    pub async fn search(
        &self,
        index_id: IndexId,
        query: UnifiedQuery,
    ) -> Result<SearchResults>;

    /// Rebalance shards for an index
    pub async fn rebalance(
        &self,
        index_id: IndexId,
        strategy: RebalanceStrategy,
    ) -> Result<RebalanceReport>;
}
```

### Unified Query Types

```rust
pub enum UnifiedQuery {
    /// Pure vector similarity search
    Vector {
        embedding: Vec<f32>,
        k: usize,
        nprobe: usize,
        ef_search: usize,
    },

    /// Pure graph traversal
    Graph {
        start_nodes: Vec<EntityId>,
        edge_filter: Option<EdgeFilter>,
        max_hops: usize,
        limit: usize,
    },

    /// Pure fulltext search
    Fulltext {
        query: String,
        limit: usize,
        fuzzy: FuzzyLevel,
        filters: Vec<Filter>,
    },

    /// Hybrid: vector + fulltext
    HybridVectorText {
        embedding: Vec<f32>,
        text_query: String,
        k: usize,
        nprobe: usize,
        alpha: f32,  // vector vs text weight
    },

    /// Hybrid: vector + graph expansion
    HybridVectorGraph {
        embedding: Vec<f32>,
        k: usize,
        expand_hops: usize,  // expand results via graph
        edge_filter: Option<EdgeFilter>,
    },

    /// Multi-stage: vector → graph → fulltext
    Pipeline(Vec<PipelineStage>),
}
```

---

## References

- [PHASE8.md](../vector/docs/PHASE8.md) - Scale validation and 1B feasibility analysis
- [BASELINE.md](../vector/docs/BASELINE.md) - Current benchmark baselines
- [Vector Schema](../src/vector/schema.rs) - Vector subsystem storage schema
- [Graph Schema](../src/graph/schema.rs) - Graph subsystem storage schema
- [Fulltext Schema](../src/fulltext/schema.rs) - Fulltext subsystem field definitions

---

## Appendix: Subsystem Summary

### Vector Subsystem Key Characteristics

| Aspect | Detail |
|--------|--------|
| ID System | 64-bit embedding code + 32-bit vec_id |
| External ID | ULID (16 bytes) mapped via IdForward/IdReverse |
| Index Structure | HNSW with configurable M, ef_construction |
| Compression | RaBitQ binary codes with ADC correction |
| Lifecycle | 4 states: Indexed, Deleted, Pending, PendingDeleted |
| Async Support | Pending CF for deferred graph construction |

### Graph Subsystem Key Characteristics

| Aspect | Detail |
|--------|--------|
| ID System | UUID (16-byte Id) |
| Edge Storage | Bidirectional: ForwardEdges + ReverseEdges |
| Name Handling | xxHash64 interning via NameCache |
| Summary Storage | Content-addressed via SummaryHash |
| Temporal Support | TemporalRange on nodes, edges, fragments |
| Hot/Cold Separation | Metadata hot, content cold |

### Fulltext Subsystem Key Characteristics

| Aspect | Detail |
|--------|--------|
| Engine | Tantivy (Rust Lucene-like) |
| Scoring | BM25 relevance |
| Doc Types | nodes, edges, node_fragments, edge_fragments |
| Facets | doc_type, tags (extracted #hashtags), validity |
| Fuzzy Search | Levenshtein distance (0, 1, 2) |
| Storage | Minimal in Tantivy, RocksDB is source of truth |
