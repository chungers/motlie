//! HNSW insert algorithm implementation.

use anyhow::Result;

use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};
use super::graph::{connect_neighbors, greedy_search_layer, search_layer};
use super::Index;
use crate::vector::cache::NavigationLayerInfo;
use crate::vector::schema::{
    GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField, HnswLayer, VecId, VecMeta,
    VecMetaCfKey, VecMetaCfValue, VecMetadata,
};
use crate::vector::Storage;

/// Insert a vector into the HNSW index.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `storage` - RocksDB storage handle
/// * `vec_id` - Internal vector ID (already allocated)
/// * `vector` - The vector data (already stored in Vectors CF)
///
/// # Algorithm
/// 1. Assign random layer using HNSW distribution
/// 2. Get or initialize navigation info
/// 3. If graph is empty, just set as entry point
/// 4. Otherwise: greedy descent from entry point to target layer
/// 5. At each layer from target down to 0: find and connect neighbors
/// 6. Update entry point if this node has higher layer
pub fn insert(index: &Index, storage: &Storage, vec_id: VecId, vector: &[f32]) -> Result<()> {
    let txn_db = storage.transaction_db()?;

    // 1. Assign random layer
    let mut rng = rand::thread_rng();
    let node_layer = index
        .nav_cache()
        .get(index.embedding())
        .map(|info| info.random_layer(&mut rng))
        .unwrap_or(0);

    // 2. Store node metadata
    store_vec_meta(index, txn_db, vec_id, node_layer)?;

    // 3. Get current navigation state
    let nav_info = get_or_init_navigation(index, storage)?;

    // 4. Handle empty graph case
    if nav_info.is_empty() {
        init_first_node(index, storage, vec_id, node_layer)?;
        return Ok(());
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
        let neighbors =
            search_layer(index, storage, vector, current, index.config().ef_construction, layer, false)?;

        // Select M neighbors (simple heuristic: take closest)
        let m = if layer == 0 {
            index.config().m * 2
        } else {
            index.config().m
        };
        let selected: Vec<_> = neighbors.into_iter().take(m as usize).collect();

        // Connect bidirectionally
        connect_neighbors(index, txn_db, vec_id, &selected, layer)?;

        // Update current for next layer
        if let Some(&(dist, id)) = selected.first() {
            current = id;
            current_dist = dist;
        }
    }

    // Suppress unused variable warning
    let _ = current_dist;

    // 7. Update entry point if needed
    if node_layer > max_layer {
        update_entry_point(index, storage, vec_id, node_layer)?;
    }

    // Update navigation cache
    index.nav_cache().update(index.embedding(), index.config().m, |info| {
        info.maybe_update_entry(vec_id, node_layer);
        info.increment_layer_count(node_layer);
    });

    Ok(())
}

/// Initialize the first node in an empty graph.
fn init_first_node(
    index: &Index,
    storage: &Storage,
    vec_id: VecId,
    node_layer: HnswLayer,
) -> Result<()> {
    // Update navigation cache
    index.nav_cache().update(index.embedding(), index.config().m, |info| {
        info.maybe_update_entry(vec_id, node_layer);
        info.increment_layer_count(0);
    });

    // Persist to GraphMeta
    update_entry_point(index, storage, vec_id, node_layer)?;

    Ok(())
}

/// Store vector metadata (layer assignment).
fn store_vec_meta(
    index: &Index,
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

    txn_db.put_cf(
        &cf,
        VecMeta::key_to_bytes(&key),
        VecMeta::value_to_bytes(&value)?,
    )?;

    Ok(())
}

/// Update the global entry point.
fn update_entry_point(
    index: &Index,
    storage: &Storage,
    vec_id: VecId,
    layer: HnswLayer,
) -> Result<()> {
    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

    // Store entry point
    let ep_key = GraphMetaCfKey::entry_point(index.embedding());
    let ep_value = GraphMetaCfValue(GraphMetaField::EntryPoint(vec_id));
    txn_db.put_cf(&cf, GraphMeta::key_to_bytes(&ep_key), GraphMeta::value_to_bytes(&ep_value))?;

    // Store max level
    let level_key = GraphMetaCfKey::max_level(index.embedding());
    let level_value = GraphMetaCfValue(GraphMetaField::MaxLevel(layer));
    txn_db.put_cf(
        &cf,
        GraphMeta::key_to_bytes(&level_key),
        GraphMeta::value_to_bytes(&level_value),
    )?;

    Ok(())
}

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
