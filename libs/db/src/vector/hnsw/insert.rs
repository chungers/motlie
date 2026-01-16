//! HNSW insert algorithm implementation.
//!
//! This module provides two insert APIs:
//! - `insert()` - Transaction-aware API that accepts a transaction and returns
//!   a `CacheUpdate` to apply after commit (for production use)
//! - `insert_for_batch()` - Batch-aware variant that uses `BatchEdgeCache` for
//!   visibility of uncommitted edges during batch inserts
//!
//! # Transaction Safety
//!
//! For atomic inserts, use `insert()` within a transaction scope:
//! ```rust,ignore
//! let txn = txn_db.transaction();
//! let cache_update = insert(&index, &txn, &txn_db, vec_id, &vector)?;
//! txn.commit()?;
//! cache_update.apply(&nav_cache, config.m);  // Only after successful commit
//! ```

use anyhow::Result;

use super::graph::{
    connect_neighbors, greedy_search_layer, greedy_search_layer_with_batch_cache,
    search_layer, search_layer_with_batch_cache, BatchEdgeCache,
};
use super::Index;
use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};
use crate::vector::cache::{NavigationCache, NavigationLayerInfo};
use crate::vector::schema::{
    EmbeddingCode, GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField, HnswLayer, VecId,
    VecMeta, VecMetaCfKey, VecMetaCfValue, VecMetadata,
};
use crate::vector::Storage;

// ============================================================================
// CacheUpdate - Deferred cache update after transaction commit
// ============================================================================

/// Deferred cache update to apply after transaction commits successfully.
///
/// This struct holds the information needed to update the navigation cache
/// after a successful insert. It should only be applied after `txn.commit()`
/// succeeds to avoid caching uncommitted state.
#[derive(Debug, Clone)]
pub struct CacheUpdate {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// The inserted vector's ID
    pub vec_id: VecId,
    /// The layer assigned to this vector
    pub node_layer: HnswLayer,
    /// Whether this node became the new entry point
    pub is_new_entry_point: bool,
    /// M parameter for cache update
    pub m: usize,
}

impl CacheUpdate {
    /// Apply this cache update to the navigation cache.
    ///
    /// **IMPORTANT**: Only call this after `txn.commit()` succeeds.
    /// Calling before commit risks caching uncommitted state.
    ///
    /// In HNSW, every node exists at ALL layers from 0 up to its `node_layer`.
    /// This method increments layer_counts for all layers the node exists at,
    /// ensuring `total_nodes()` (which uses `layer_counts[0]`) is correct.
    pub fn apply(self, nav_cache: &NavigationCache) {
        nav_cache.update(self.embedding, self.m, |info| {
            info.maybe_update_entry(self.vec_id, self.node_layer);
            // Increment layer counts for all layers this node exists at (0 to node_layer)
            for layer in 0..=self.node_layer {
                info.increment_layer_count(layer);
            }
        });
    }
}

// ============================================================================
// Transaction-aware insert API
// ============================================================================

/// Insert a vector into the HNSW index within a transaction.
///
/// This is the recommended API for production use. It accepts a transaction
/// reference and returns a `CacheUpdate` that should be applied only after
/// the transaction commits successfully.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `txn` - RocksDB transaction to write within
/// * `txn_db` - TransactionDB for CF handle lookup
/// * `storage` - Storage for reads during graph traversal
/// * `vec_id` - Internal vector ID (already allocated)
/// * `vector` - The vector data (already stored in Vectors CF)
///
/// # Returns
/// A `CacheUpdate` to apply after `txn.commit()` succeeds.
///
/// # Example
/// ```rust,ignore
/// let txn = txn_db.transaction();
/// let cache_update = insert(&index, &txn, &txn_db, &storage, vec_id, &vector)?;
/// txn.commit()?;
/// cache_update.apply(index.nav_cache());
/// ```
pub fn insert(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    vec_id: VecId,
    vector: &[f32],
) -> Result<CacheUpdate> {
    // 1. Get or initialize navigation state FIRST (fixes cold cache bug)
    // This ensures we have correct max_layer for layer distribution even after restart
    let nav_info = get_or_init_navigation(index, storage)?;

    // 2. Assign random layer using proper exponential distribution
    let mut rng = rand::thread_rng();
    let node_layer = nav_info.random_layer(&mut rng);

    // 3. Store node metadata (using transaction)
    store_vec_meta(index, txn, txn_db, vec_id, node_layer)?;

    // 4. Handle empty graph case
    if nav_info.is_empty() {
        // Persist entry point within transaction
        update_entry_point(index, txn, txn_db, vec_id, node_layer)?;

        return Ok(CacheUpdate {
            embedding: index.embedding(),
            vec_id,
            node_layer,
            is_new_entry_point: true,
            m: index.config().m,
        });
    }

    // 5. Greedy descent to find entry point at target layer
    let entry_point = nav_info.entry_point().unwrap();
    let max_layer = nav_info.max_layer;

    let mut current = entry_point;
    let mut current_dist = super::graph::distance(index, storage, vector, current)?;

    // Descend from max_layer to node_layer + 1 (greedy, single candidate)
    // use_cache: false - index build should not use cache
    for layer in (node_layer + 1..=max_layer).rev() {
        (current, current_dist) =
            greedy_search_layer(index, storage, vector, current, current_dist, layer, false)?;
    }

    // 6. At each layer from node_layer down to 0: find neighbors and connect
    for layer in (0..=node_layer).rev() {
        // Search for neighbors at this layer (use_cache: false for build)
        let neighbors = search_layer(
            index,
            storage,
            vector,
            current,
            index.config().ef_construction,
            layer,
            false,
        )?;

        // Select M neighbors (simple heuristic: take closest)
        let m = if layer == 0 {
            index.config().m * 2
        } else {
            index.config().m
        };
        let selected: Vec<_> = neighbors.into_iter().take(m as usize).collect();

        // Connect bidirectionally (using transaction)
        connect_neighbors(index, txn, txn_db, vec_id, &selected, layer)?;

        // Update current for next layer
        if let Some(&(dist, id)) = selected.first() {
            current = id;
            current_dist = dist;
        }
    }

    // Suppress unused variable warning
    let _ = current_dist;

    // 7. Update entry point if needed (using transaction)
    let is_new_entry_point = node_layer > max_layer;
    if is_new_entry_point {
        update_entry_point(index, txn, txn_db, vec_id, node_layer)?;
    }

    // Return cache update to apply AFTER commit
    Ok(CacheUpdate {
        embedding: index.embedding(),
        vec_id,
        node_layer,
        is_new_entry_point,
        m: index.config().m,
    })
}

/// Insert a vector into the HNSW index during batch insert.
///
/// This variant uses transaction-aware reads AND batch edge cache to support
/// batch inserts where both vectors and edges may not be committed yet.
///
/// RocksDB's `Transaction::get()` does NOT include pending merge operands,
/// so edges written via `txn.merge_cf()` earlier in the batch are not visible
/// to later inserts. The `batch_cache` tracks these edges so that later inserts
/// can see them, ensuring proper graph connectivity.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `txn` - The active transaction (for reading uncommitted vectors)
/// * `txn_db` - The transaction DB (for CF handles)
/// * `storage` - Storage for committed data reads
/// * `vec_id` - The vector ID to insert
/// * `vector` - The vector data
/// * `batch_cache` - Cache of edges added during this batch (mutated by this call)
///
/// # Returns
/// A `CacheUpdate` that should be applied immediately for subsequent inserts
/// to see the updated navigation state.
pub fn insert_for_batch(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    vec_id: VecId,
    vector: &[f32],
    batch_cache: &mut BatchEdgeCache,
) -> Result<CacheUpdate> {
    // 1. Get or initialize navigation state FIRST
    let nav_info = get_or_init_navigation(index, storage)?;

    // 2. Assign random layer using proper exponential distribution
    let mut rng = rand::thread_rng();
    let node_layer = nav_info.random_layer(&mut rng);

    // 3. Store node metadata (using transaction)
    store_vec_meta(index, txn, txn_db, vec_id, node_layer)?;

    // 4. Handle empty graph case
    if nav_info.is_empty() {
        // Persist entry point within transaction
        update_entry_point(index, txn, txn_db, vec_id, node_layer)?;

        return Ok(CacheUpdate {
            embedding: index.embedding(),
            vec_id,
            node_layer,
            is_new_entry_point: true,
            m: index.config().m,
        });
    }

    // 5. Greedy descent to find entry point at target layer
    // Use transaction-aware distance for uncommitted vectors
    // Use batch cache for edge visibility
    let entry_point = nav_info.entry_point().unwrap();
    let max_layer = nav_info.max_layer;

    let mut current = entry_point;
    let mut current_dist = super::graph::distance_in_txn(index, txn, txn_db, vector, current)?;

    // Descend from max_layer to node_layer + 1 (greedy, single candidate)
    // Use batch-cache-aware search to see edges from earlier batch inserts
    for layer in (node_layer + 1..=max_layer).rev() {
        (current, current_dist) = greedy_search_layer_with_batch_cache(
            index, txn, txn_db, storage, vector, current, current_dist, layer, batch_cache,
        )?;
    }

    // 6. At each layer from node_layer down to 0: find neighbors and connect
    for layer in (0..=node_layer).rev() {
        // Search for neighbors using batch-cache-aware reads
        let neighbors = search_layer_with_batch_cache(
            index,
            txn,
            txn_db,
            storage,
            vector,
            current,
            index.config().ef_construction,
            layer,
            batch_cache,
        )?;

        // Select M neighbors (simple heuristic: take closest)
        let m = if layer == 0 {
            index.config().m * 2
        } else {
            index.config().m
        };
        let selected: Vec<_> = neighbors.into_iter().take(m as usize).collect();

        // Connect bidirectionally (using transaction)
        connect_neighbors(index, txn, txn_db, vec_id, &selected, layer)?;

        // Record edges in batch cache for later inserts to see
        let neighbor_ids: Vec<VecId> = selected.iter().map(|(_, id)| *id).collect();
        batch_cache.add_edges(vec_id, layer, &neighbor_ids);

        // Update current for next layer
        if let Some(&(dist, id)) = selected.first() {
            current = id;
            current_dist = dist;
        }
    }

    // Suppress unused variable warning
    let _ = current_dist;

    // 7. Update entry point if needed (using transaction)
    let is_new_entry_point = node_layer > max_layer;
    if is_new_entry_point {
        update_entry_point(index, txn, txn_db, vec_id, node_layer)?;
    }

    // Return cache update - caller should apply immediately for batch
    Ok(CacheUpdate {
        embedding: index.embedding(),
        vec_id,
        node_layer,
        is_new_entry_point,
        m: index.config().m,
    })
}

// ============================================================================
// Transaction-aware helper functions
// ============================================================================

/// Store vector metadata (layer assignment) within a transaction.
fn store_vec_meta(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
    max_layer: HnswLayer,
) -> Result<()> {
    let key = VecMetaCfKey(index.embedding(), vec_id);
    let value = VecMetaCfValue(VecMetadata {
        max_layer,
        flags: 0,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    });

    let cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;

    txn.put_cf(
        &cf,
        VecMeta::key_to_bytes(&key),
        VecMeta::value_to_bytes(&value)?,
    )?;

    Ok(())
}

/// Update the global entry point within a transaction.
fn update_entry_point(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
    layer: HnswLayer,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

    // Store entry point
    let ep_key = GraphMetaCfKey::entry_point(index.embedding());
    let ep_value = GraphMetaCfValue(GraphMetaField::EntryPoint(vec_id));
    txn.put_cf(
        &cf,
        GraphMeta::key_to_bytes(&ep_key),
        GraphMeta::value_to_bytes(&ep_value),
    )?;

    // Store max level
    let level_key = GraphMetaCfKey::max_level(index.embedding());
    let level_value = GraphMetaCfValue(GraphMetaField::MaxLevel(layer));
    txn.put_cf(
        &cf,
        GraphMeta::key_to_bytes(&level_key),
        GraphMeta::value_to_bytes(&level_value),
    )?;

    Ok(())
}

// ============================================================================
// Navigation info loading (read-only, no transaction needed)
// ============================================================================

/// Get or initialize navigation info from cache or storage.
pub(super) fn get_or_init_navigation(index: &Index, storage: &Storage) -> Result<NavigationLayerInfo> {
    if let Some(info) = index.nav_cache().get(index.embedding()) {
        return Ok(info);
    }

    // Try to load from storage
    if let Ok(info) = load_navigation(index, storage) {
        index.nav_cache().put(index.embedding(), info.clone());
        return Ok(info);
    }

    // Initialize empty
    let info = NavigationLayerInfo::new(index.config().m);
    index.nav_cache().put(index.embedding(), info.clone());
    Ok(info)
}

/// Load navigation info from GraphMeta CF.
pub(super) fn load_navigation(index: &Index, storage: &Storage) -> Result<NavigationLayerInfo> {
    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

    // Load entry point
    let ep_key = GraphMetaCfKey::entry_point(index.embedding());
    let ep_bytes = txn_db
        .get_cf(&cf, GraphMeta::key_to_bytes(&ep_key))?
        .ok_or_else(|| anyhow::anyhow!("No entry point for embedding {}", index.embedding()))?;
    let ep_value = GraphMeta::value_from_bytes(&ep_key, &ep_bytes)?;
    let entry_point = match ep_value.0 {
        GraphMetaField::EntryPoint(id) => id,
        _ => anyhow::bail!("Unexpected GraphMeta value type"),
    };

    // Load max level
    let level_key = GraphMetaCfKey::max_level(index.embedding());
    let level_bytes = txn_db
        .get_cf(&cf, GraphMeta::key_to_bytes(&level_key))?
        .ok_or_else(|| anyhow::anyhow!("No max level for embedding {}", index.embedding()))?;
    let level_value = GraphMeta::value_from_bytes(&level_key, &level_bytes)?;
    let max_layer = match level_value.0 {
        GraphMetaField::MaxLevel(l) => l,
        _ => anyhow::bail!("Unexpected GraphMeta value type"),
    };

    // Build navigation info
    let mut info = NavigationLayerInfo::new(index.config().m);
    for _ in 0..=max_layer {
        info.entry_points.push(entry_point);
        info.layer_counts.push(0); // Counts not persisted yet
    }
    info.max_layer = max_layer;

    Ok(info)
}
