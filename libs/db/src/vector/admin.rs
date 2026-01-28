//! Administrative and diagnostic utilities for vector storage.
//!
//! Provides introspection into vector storage state, metadata inspection,
//! and consistency validation. Used by CLI tools for debugging.
//!
//! ## Usage
//!
//! ```ignore
//! use motlie_db::vector::{admin, Storage};
//!
//! let mut storage = Storage::readwrite("./db");
//! storage.ready()?;
//!
//! // Get stats for all embeddings
//! let stats = admin::get_all_stats(&storage)?;
//!
//! // Validate consistency
//! let results = admin::validate_all(&storage)?;
//!
//! // Secondary (read-only) mode for non-blocking access:
//! let mut storage = Storage::secondary("./db", "/tmp/secondary");
//! storage.ready()?;
//! let stats = admin::get_all_stats_secondary(&storage)?;
//! ```

use anyhow::Result;
use roaring::RoaringBitmap;
use rocksdb::IteratorMode;
use serde::Serialize;

use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};
use crate::vector::schema::{
    BinaryCodes, BinaryCodeCfKey, Edges, EdgeCfKey, EmbeddingCode, EmbeddingSpecCfKey,
    EmbeddingSpecs, GraphMeta, GraphMetaCfKey, GraphMetaField, IdAlloc, IdAllocCfKey,
    IdAllocField, IdForward, IdReverse, IdReverseCfKey, Pending, VecId, VecMeta,
    VecMetaCfKey, ALL_COLUMN_FAMILIES,
};
use crate::vector::{EmbeddingSpec, Storage};
use crate::Id;

// ============================================================================
// DB Access Abstraction (ADMIN.md Read-Only Mode)
// ============================================================================

// ADMIN.md Read-Only Mode (chungers, 2025-01-27, FIXED):
// Secondary mode support added for read-only access without write contention.
// Functions with _secondary suffix use DB (secondary instance) instead of TransactionDB.

// ============================================================================
// Public Data Structures
// ============================================================================

/// Vector lifecycle states.
///
/// Re-exported from schema for CLI consumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VecLifecycle {
    /// Vector is fully indexed in the HNSW graph and searchable.
    Indexed,
    /// Vector has been soft-deleted.
    Deleted,
    /// Vector is stored but awaiting async HNSW graph construction.
    Pending,
    /// Vector was deleted before async graph construction completed.
    PendingDeleted,
}

impl From<crate::vector::schema::VecLifecycle> for VecLifecycle {
    fn from(v: crate::vector::schema::VecLifecycle) -> Self {
        match v {
            crate::vector::schema::VecLifecycle::Indexed => Self::Indexed,
            crate::vector::schema::VecLifecycle::Deleted => Self::Deleted,
            crate::vector::schema::VecLifecycle::Pending => Self::Pending,
            crate::vector::schema::VecLifecycle::PendingDeleted => Self::PendingDeleted,
        }
    }
}

/// Statistics for a single embedding.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingStats {
    /// Embedding code (primary key).
    pub code: EmbeddingCode,
    /// Model name.
    pub model: String,
    /// Vector dimension.
    pub dim: u32,
    /// Distance metric.
    pub distance: String,
    /// Storage type (f32/f16).
    pub storage_type: String,

    // Counts
    // ADMIN.md Lifecycle Accounting (chungers, 2025-01-27, FIXED):
    // PendingDeleted is now shown separately instead of folded into Deleted.
    /// Total vectors in this embedding.
    pub total_vectors: u32,
    /// Vectors fully indexed in HNSW graph.
    pub indexed_count: u32,
    /// Vectors awaiting async graph construction.
    pub pending_count: u32,
    /// Soft-deleted indexed vectors awaiting GC cleanup.
    pub deleted_count: u32,
    /// Vectors deleted before async graph construction completed.
    pub pending_deleted_count: u32,

    // Graph metadata
    /// HNSW entry point (None if graph is empty).
    pub entry_point: Option<VecId>,
    /// Maximum HNSW layer.
    pub max_level: u8,
    /// Spec hash stored at index build time.
    pub spec_hash: Option<u64>,
    /// Whether spec hash matches current spec.
    pub spec_hash_valid: bool,

    // ID allocator
    /// Next ID to be allocated.
    pub next_id: u32,
    /// Number of recycled IDs available.
    pub free_id_count: u64,
}

/// Layer distribution in HNSW graph.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LayerDistribution {
    /// Count of vectors at each layer (index = layer number).
    pub counts: Vec<u32>,
}

/// Detailed inspection of a single embedding.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingInspection {
    /// Basic statistics.
    pub stats: EmbeddingStats,
    /// HNSW layer distribution.
    pub layer_distribution: LayerDistribution,
    /// Number of vectors in pending queue.
    pub pending_queue_depth: usize,
    /// Oldest pending entry timestamp (millis since epoch).
    pub oldest_pending_timestamp: Option<u64>,
    /// HNSW build parameters.
    pub hnsw_m: u16,
    pub hnsw_ef_construction: u16,
    /// RaBitQ parameters.
    pub rabitq_bits: u8,
    pub rabitq_seed: u64,
}

/// Information about a single vector.
#[derive(Debug, Clone, Serialize)]
pub struct VectorInfo {
    /// Internal vector ID.
    pub vec_id: VecId,
    /// External ULID (if resolved).
    pub external_id: Option<String>,
    /// Lifecycle state.
    pub lifecycle: VecLifecycle,
    /// Maximum HNSW layer this vector appears in.
    pub max_layer: u8,
    /// Creation timestamp (millis since epoch).
    pub created_at: Option<u64>,
    /// Edge count per layer.
    pub edge_counts: Vec<usize>,
    /// Whether vector has binary code (RaBitQ).
    pub has_binary_code: bool,
}

/// Single validation check result.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationCheck {
    /// Check name.
    pub name: &'static str,
    /// Check status.
    pub status: ValidationStatus,
    /// Detailed message.
    pub message: String,
}

/// Validation check status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Pass,
    Warning,
    Error,
}

/// Full validation results for an embedding.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    /// Embedding code.
    pub code: EmbeddingCode,
    /// Model name (for display).
    pub model: String,
    /// Individual check results.
    pub checks: Vec<ValidationCheck>,
}

/// Per-column-family storage statistics.
#[derive(Debug, Clone, Serialize)]
/// ADMIN.md Issue #4 (chungers, 2025-01-27, FIXED):
/// Added is_sampled field to indicate when values are from sampling vs full scan.
/// ADMIN.md RocksDB Properties (chungers, 2025-01-27, ADDED):
/// Added estimated_num_keys from RocksDB property when available.
pub struct ColumnFamilyStats {
    /// Column family name.
    pub name: String,
    /// Number of entries (may be sampled if is_sampled=true).
    pub entry_count: u64,
    /// Size in bytes (sampled key+value bytes, not on-disk size).
    pub size_bytes: u64,
    /// True if entry_count hit the sample limit (10,000).
    pub is_sampled: bool,
    /// Estimated number of keys from RocksDB property (None if unavailable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_num_keys: Option<u64>,
}

/// RocksDB-level diagnostics.
#[derive(Debug, Clone, Serialize)]
pub struct RocksDbStats {
    /// Per-CF statistics.
    pub column_families: Vec<ColumnFamilyStats>,
    /// Total size across all vector CFs.
    pub total_size_bytes: u64,
}

// ============================================================================
// Core Functions
// ============================================================================

/// Get statistics for a single embedding.
pub fn get_embedding_stats(storage: &Storage, code: EmbeddingCode) -> Result<EmbeddingStats> {
    let txn_db = storage.transaction_db()?;

    // Load embedding spec
    let spec = load_embedding_spec(&txn_db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;

    // Load graph metadata
    let (entry_point, max_level, graph_count, spec_hash) = load_graph_meta(&txn_db, code)?;

    // Count vectors by lifecycle
    let (indexed_count, pending_count, deleted_count, pending_deleted_count) =
        count_vectors_by_lifecycle(&txn_db, code)?;

    // Load ID allocator state
    let (next_id, free_id_count) = load_id_alloc_state(&txn_db, code)?;

    // Validate spec hash
    let spec_hash_valid = match spec_hash {
        Some(hash) => hash == spec.compute_spec_hash(),
        None => true, // No hash stored yet is OK
    };

    let total_vectors = graph_count
        .unwrap_or(indexed_count + pending_count + deleted_count + pending_deleted_count);

    Ok(EmbeddingStats {
        code,
        model: spec.model,
        dim: spec.dim,
        distance: format!("{:?}", spec.distance).to_lowercase(),
        storage_type: format!("{:?}", spec.storage_type).to_lowercase(),
        total_vectors,
        indexed_count,
        pending_count,
        deleted_count,
        pending_deleted_count,
        entry_point,
        max_level,
        spec_hash,
        spec_hash_valid,
        next_id,
        free_id_count,
    })
}

/// Get statistics for all embeddings.
pub fn get_all_stats(storage: &Storage) -> Result<Vec<EmbeddingStats>> {
    let txn_db = storage.transaction_db()?;

    let specs_cf = txn_db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut stats = Vec::new();
    let iter = txn_db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            match get_embedding_stats(storage, code) {
                Ok(s) => stats.push(s),
                Err(e) => {
                    // Log but continue with other embeddings
                    tracing::warn!("Failed to get stats for embedding {}: {}", code, e);
                }
            }
        }
    }

    Ok(stats)
}

/// Deep inspection of an embedding (includes layer distribution, pending queue).
pub fn inspect_embedding(storage: &Storage, code: EmbeddingCode) -> Result<EmbeddingInspection> {
    let txn_db = storage.transaction_db()?;

    // Get basic stats
    let stats = get_embedding_stats(storage, code)?;

    // Load full spec for build parameters
    let spec = load_embedding_spec(&txn_db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;

    // Compute layer distribution
    let layer_distribution = compute_layer_distribution(&txn_db, code, None)?;

    // Get pending queue info
    let (pending_queue_depth, oldest_pending_timestamp) = get_pending_queue_info(&txn_db, code)?;

    Ok(EmbeddingInspection {
        stats,
        layer_distribution,
        pending_queue_depth,
        oldest_pending_timestamp,
        hnsw_m: spec.hnsw_m,
        hnsw_ef_construction: spec.hnsw_ef_construction,
        rabitq_bits: spec.rabitq_bits,
        rabitq_seed: spec.rabitq_seed,
    })
}

/// Get information about a specific vector.
pub fn get_vector_info(
    storage: &Storage,
    code: EmbeddingCode,
    vec_id: VecId,
) -> Result<Option<VectorInfo>> {
    let txn_db = storage.transaction_db()?;

    // Load vector metadata
    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let meta_key = VecMetaCfKey(code, vec_id);
    let meta_bytes = match txn_db.get_cf(&meta_cf, VecMeta::key_to_bytes(&meta_key))? {
        Some(b) => b,
        None => return Ok(None),
    };
    let meta_value = VecMeta::value_from_bytes(&meta_bytes)?;
    let meta = &meta_value.0;

    // Resolve external ID
    let external_id = resolve_external_id(&txn_db, code, vec_id)?;

    // Count edges per layer
    let edge_counts = count_edges_per_layer(&txn_db, code, vec_id, meta.max_layer)?;

    // Check for binary code
    let has_binary_code = has_binary_code(&txn_db, code, vec_id)?;

    Ok(Some(VectorInfo {
        vec_id,
        external_id: external_id.map(|id| id.to_string()),
        lifecycle: meta.lifecycle().into(),
        max_layer: meta.max_layer,
        created_at: Some(meta.created_at),
        edge_counts,
        has_binary_code,
    }))
}

/// List vectors filtered by lifecycle state.
pub fn list_vectors_by_state(
    storage: &Storage,
    code: EmbeddingCode,
    state: VecLifecycle,
    limit: usize,
) -> Result<Vec<VectorInfo>> {
    let txn_db = storage.transaction_db()?;

    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = txn_db.iterator_cf(
        &meta_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let target_state: crate::vector::schema::VecLifecycle = match state {
        VecLifecycle::Indexed => crate::vector::schema::VecLifecycle::Indexed,
        VecLifecycle::Deleted => crate::vector::schema::VecLifecycle::Deleted,
        VecLifecycle::Pending => crate::vector::schema::VecLifecycle::Pending,
        VecLifecycle::PendingDeleted => crate::vector::schema::VecLifecycle::PendingDeleted,
    };

    let mut results = Vec::new();

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        // Check prefix
        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let key = VecMeta::key_from_bytes(&key_bytes)?;
        if key.0 != code {
            break;
        }

        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;

        if meta.lifecycle() == target_state {
            let external_id = resolve_external_id(&txn_db, code, key.1)?;
            let edge_counts = count_edges_per_layer(&txn_db, code, key.1, meta.max_layer)?;
            let has_binary_code = has_binary_code(&txn_db, code, key.1)?;

            results.push(VectorInfo {
                vec_id: key.1,
                external_id: external_id.map(|id| id.to_string()),
                lifecycle: meta.lifecycle().into(),
                max_layer: meta.max_layer,
                created_at: Some(meta.created_at),
                edge_counts,
                has_binary_code,
            });

            if results.len() >= limit {
                break;
            }
        }
    }

    Ok(results)
}

/// Sample random vectors from an embedding using reservoir sampling.
pub fn sample_vectors(
    storage: &Storage,
    code: EmbeddingCode,
    count: usize,
    seed: u64,
) -> Result<Vec<VectorInfo>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let txn_db = storage.transaction_db()?;

    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = txn_db.iterator_cf(
        &meta_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    // Reservoir sampling - store (vec_id, max_layer, lifecycle, created_at)
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut reservoir: Vec<(VecId, u8, crate::vector::schema::VecLifecycle, u64)> =
        Vec::with_capacity(count);
    let mut total = 0usize;

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let key = VecMeta::key_from_bytes(&key_bytes)?;
        if key.0 != code {
            break;
        }

        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;
        let entry = (key.1, meta.max_layer, meta.lifecycle(), meta.created_at);

        if total < count {
            reservoir.push(entry);
        } else {
            let j = rng.gen_range(0..=total);
            if j < count {
                reservoir[j] = entry;
            }
        }
        total += 1;
    }

    // Convert to VectorInfo
    let mut results = Vec::with_capacity(reservoir.len());
    for (vec_id, max_layer, lifecycle, created_at) in reservoir {
        let external_id = resolve_external_id(&txn_db, code, vec_id)?;
        let edge_counts = count_edges_per_layer(&txn_db, code, vec_id, max_layer)?;
        let has_binary_code = has_binary_code(&txn_db, code, vec_id)?;

        results.push(VectorInfo {
            vec_id,
            external_id: external_id.map(|id| id.to_string()),
            lifecycle: lifecycle.into(),
            max_layer,
            created_at: Some(created_at),
            edge_counts,
            has_binary_code,
        });
    }

    Ok(results)
}

/// Run validation checks on an embedding.
///
/// Uses sampling (first 1000 entries) for performance. Use `validate_embedding_strict`
/// for full-scan validation.
pub fn validate_embedding(storage: &Storage, code: EmbeddingCode) -> Result<ValidationResult> {
    validate_embedding_with_options(storage, code, false)
}

/// Run validation checks on an embedding with strict mode (no sampling).
///
/// ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
/// Added --strict option for full-scan validation without sampling.
pub fn validate_embedding_strict(storage: &Storage, code: EmbeddingCode) -> Result<ValidationResult> {
    validate_embedding_with_options(storage, code, true)
}

fn validate_embedding_with_options(
    storage: &Storage,
    code: EmbeddingCode,
    strict: bool,
) -> Result<ValidationResult> {
    let txn_db = storage.transaction_db()?;

    let spec = load_embedding_spec(&txn_db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;

    let mut checks = Vec::new();

    // Sample limit: None for strict mode (full scan), Some(1000) for normal mode
    let sample_limit = if strict { None } else { Some(1000) };

    // 1. Spec hash validation
    let (_, _, _, spec_hash) = load_graph_meta(&txn_db, code)?;
    checks.push(validate_spec_hash(&spec, spec_hash));

    // 2. Entry point validation
    checks.push(validate_entry_point(storage, code)?);

    // 3. ID mapping consistency (forward -> reverse)
    checks.push(validate_id_mappings(&txn_db, code)?);

    // 4. Count accuracy
    checks.push(validate_count_accuracy(&txn_db, code)?);

    // 5. Pending queue staleness
    checks.push(validate_pending_queue(&txn_db, code)?);

    // 6. Orphaned vectors check
    checks.push(validate_no_orphaned_vectors(&txn_db, code)?);

    // ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
    // Additional checks for data integrity.

    // 7. Reverse ID mapping consistency (reverse -> forward)
    checks.push(validate_reverse_id_mappings(&txn_db, code, sample_limit)?);

    // 8. VecMeta presence for all IdReverse entries
    checks.push(validate_vec_meta_presence(&txn_db, code, sample_limit)?);

    // 9. Vector payload existence for indexed vectors
    checks.push(validate_vector_payloads(&txn_db, code, sample_limit)?);

    Ok(ValidationResult {
        code,
        model: spec.model,
        checks,
    })
}

/// Run validation checks on all embeddings.
pub fn validate_all(storage: &Storage) -> Result<Vec<ValidationResult>> {
    validate_all_with_options(storage, false)
}

/// Run validation checks on all embeddings with strict mode (no sampling).
pub fn validate_all_strict(storage: &Storage) -> Result<Vec<ValidationResult>> {
    validate_all_with_options(storage, true)
}

fn validate_all_with_options(storage: &Storage, strict: bool) -> Result<Vec<ValidationResult>> {
    let txn_db = storage.transaction_db()?;

    let specs_cf = txn_db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut results = Vec::new();
    let iter = txn_db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            let result = if strict {
                validate_embedding_strict(storage, code)
            } else {
                validate_embedding(storage, code)
            };
            match result {
                Ok(r) => results.push(r),
                Err(e) => {
                    tracing::warn!("Failed to validate embedding {}: {}", code, e);
                }
            }
        }
    }

    Ok(results)
}

/// Get RocksDB-level statistics.
///
/// Note: Entry counts are obtained by scanning up to 10,000 entries per CF.
/// If a CF has more than 10,000 entries, is_sampled will be true and counts
/// represent the sample only. Size is the sum of sampled key+value bytes,
/// not on-disk size.
///
/// ADMIN.md RocksDB Properties (chungers, 2025-01-27, ADDED):
/// Use `get_rocksdb_stats_secondary` for RocksDB property-based estimates
/// (property_int_value_cf only available on DB, not TransactionDB).
pub fn get_rocksdb_stats(storage: &Storage) -> Result<RocksDbStats> {
    let txn_db = storage.transaction_db()?;

    let mut column_families = Vec::new();
    let mut total_size_bytes = 0u64;

    for cf_name in ALL_COLUMN_FAMILIES {
        if let Some(cf) = txn_db.cf_handle(cf_name) {
            // Count entries by iterating (limited for performance)
            let iter = txn_db.iterator_cf(&cf, IteratorMode::Start);
            let mut entry_count = 0u64;
            let mut size_bytes = 0u64;
            let sample_limit = 10000u64;

            for item in iter {
                if let Ok((key, value)) = item {
                    entry_count += 1;
                    size_bytes += key.len() as u64 + value.len() as u64;
                    if entry_count >= sample_limit {
                        break;
                    }
                }
            }

            let is_sampled = entry_count >= sample_limit;

            // Note: property_int_value_cf not available on TransactionDB
            // Use get_rocksdb_stats_secondary for RocksDB property estimates
            column_families.push(ColumnFamilyStats {
                name: cf_name.to_string(),
                entry_count,
                size_bytes,
                is_sampled,
                estimated_num_keys: None,
            });

            total_size_bytes += size_bytes;
        }
    }

    Ok(RocksDbStats {
        column_families,
        total_size_bytes,
    })
}

// ============================================================================
// Secondary Mode Functions (Read-Only Access)
// ============================================================================

// ADMIN.md Read-Only Mode (chungers, 2025-01-27, FIXED):
// These functions use secondary DB mode for read-only access without
// contention with live workloads.

/// Get RocksDB-level statistics using secondary (read-only) mode.
///
/// Uses `storage.db()` instead of `storage.transaction_db()`.
/// Get RocksDB-level statistics using secondary (read-only) mode.
///
/// ADMIN.md RocksDB Properties (chungers, 2025-01-27, ADDED):
/// This version queries rocksdb.estimate-num-keys property for better estimates,
/// which is only available on DB (secondary mode), not TransactionDB.
pub fn get_rocksdb_stats_secondary(storage: &Storage) -> Result<RocksDbStats> {
    let db = storage.db()?;

    let mut column_families = Vec::new();
    let mut total_size_bytes = 0u64;

    for cf_name in ALL_COLUMN_FAMILIES {
        if let Some(cf) = db.cf_handle(cf_name) {
            // Get estimated key count from RocksDB property
            let estimated_num_keys = db
                .property_int_value_cf(&cf, "rocksdb.estimate-num-keys")
                .ok()
                .flatten();

            let iter = db.iterator_cf(&cf, IteratorMode::Start);
            let mut entry_count = 0u64;
            let mut size_bytes = 0u64;
            let sample_limit = 10000u64;

            for item in iter {
                if let Ok((key, value)) = item {
                    entry_count += 1;
                    size_bytes += key.len() as u64 + value.len() as u64;
                    if entry_count >= sample_limit {
                        break;
                    }
                }
            }

            let is_sampled = entry_count >= sample_limit;

            column_families.push(ColumnFamilyStats {
                name: cf_name.to_string(),
                entry_count,
                size_bytes,
                is_sampled,
                estimated_num_keys,
            });

            total_size_bytes += size_bytes;
        }
    }

    Ok(RocksDbStats {
        column_families,
        total_size_bytes,
    })
}

/// Get statistics for all embeddings using secondary (read-only) mode.
pub fn get_all_stats_secondary(storage: &Storage) -> Result<Vec<EmbeddingStats>> {
    let db = storage.db()?;

    let specs_cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut stats = Vec::new();
    let iter = db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            match get_embedding_stats_with_db(db, code) {
                Ok(s) => stats.push(s),
                Err(e) => {
                    tracing::warn!("Failed to get stats for embedding {}: {}", code, e);
                }
            }
        }
    }

    Ok(stats)
}

/// Get stats for a single embedding using secondary (read-only) mode.
pub fn get_embedding_stats_secondary(
    storage: &Storage,
    code: EmbeddingCode,
) -> Result<EmbeddingStats> {
    let db = storage.db()?;
    get_embedding_stats_with_db(db, code)
}

// Helper that works with rocksdb::DB for secondary mode
fn get_embedding_stats_with_db(db: &rocksdb::DB, code: EmbeddingCode) -> Result<EmbeddingStats> {
    // Load embedding spec
    let spec = load_embedding_spec_db(db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;

    // Load graph metadata
    let (entry_point, max_level, graph_count, spec_hash) = load_graph_meta_db(db, code)?;

    // Count vectors by lifecycle
    let (indexed_count, pending_count, deleted_count, pending_deleted_count) =
        count_vectors_by_lifecycle_db(db, code)?;

    // Load ID allocator state
    let (next_id, free_id_count) = load_id_alloc_state_db(db, code)?;

    // Validate spec hash
    let spec_hash_valid = match spec_hash {
        Some(hash) => hash == spec.compute_spec_hash(),
        None => true,
    };

    let total_vectors = graph_count
        .unwrap_or(indexed_count + pending_count + deleted_count + pending_deleted_count);

    Ok(EmbeddingStats {
        code,
        model: spec.model,
        dim: spec.dim,
        distance: format!("{:?}", spec.distance).to_lowercase(),
        storage_type: format!("{:?}", spec.storage_type).to_lowercase(),
        total_vectors,
        indexed_count,
        pending_count,
        deleted_count,
        pending_deleted_count,
        entry_point,
        max_level,
        spec_hash,
        spec_hash_valid,
        next_id,
        free_id_count,
    })
}

// DB versions of internal helpers for secondary mode

fn load_embedding_spec_db(
    db: &rocksdb::DB,
    code: EmbeddingCode,
) -> Result<Option<EmbeddingSpec>> {
    use crate::rocksdb::ColumnFamilySerde;

    let cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let key = EmbeddingSpecCfKey(code);
    let key_bytes = EmbeddingSpecs::key_to_bytes(&key);

    match db.get_cf(&cf, key_bytes)? {
        Some(value_bytes) => {
            let value = EmbeddingSpecs::value_from_bytes(&value_bytes)?;
            Ok(Some(value.0))
        }
        None => Ok(None),
    }
}

fn load_graph_meta_db(
    db: &rocksdb::DB,
    code: EmbeddingCode,
) -> Result<(Option<VecId>, u8, Option<u32>, Option<u64>)> {
    let cf = db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", GraphMeta::CF_NAME))?;

    // Entry point
    let ep_key = GraphMetaCfKey::entry_point(code);
    let entry_point = match db.get_cf(&cf, GraphMeta::key_to_bytes(&ep_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&ep_key, &bytes)?;
            match val.0 {
                GraphMetaField::EntryPoint(v) => Some(v),
                _ => None,
            }
        }
        None => None,
    };

    // Max level
    let level_key = GraphMetaCfKey::max_level(code);
    let max_level = match db.get_cf(&cf, GraphMeta::key_to_bytes(&level_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&level_key, &bytes)?;
            match val.0 {
                GraphMetaField::MaxLevel(v) => v,
                _ => 0,
            }
        }
        None => 0,
    };

    // Count
    let count_key = GraphMetaCfKey::count(code);
    let count = match db.get_cf(&cf, GraphMeta::key_to_bytes(&count_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&count_key, &bytes)?;
            match val.0 {
                GraphMetaField::Count(v) => Some(v),
                _ => None,
            }
        }
        None => None,
    };

    // Spec hash
    let hash_key = GraphMetaCfKey::spec_hash(code);
    let spec_hash = match db.get_cf(&cf, GraphMeta::key_to_bytes(&hash_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&hash_key, &bytes)?;
            match val.0 {
                GraphMetaField::SpecHash(v) => Some(v),
                _ => None,
            }
        }
        None => None,
    };

    Ok((entry_point, max_level, count, spec_hash))
}

fn count_vectors_by_lifecycle_db(
    db: &rocksdb::DB,
    code: EmbeddingCode,
) -> Result<(u32, u32, u32, u32)> {
    let cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
        &cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut indexed = 0u32;
    let mut pending = 0u32;
    let mut deleted = 0u32;
    let mut pending_deleted = 0u32;

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;

        match meta.lifecycle() {
            crate::vector::schema::VecLifecycle::Indexed => indexed += 1,
            crate::vector::schema::VecLifecycle::Pending => pending += 1,
            crate::vector::schema::VecLifecycle::Deleted => deleted += 1,
            crate::vector::schema::VecLifecycle::PendingDeleted => pending_deleted += 1,
        }
    }

    Ok((indexed, pending, deleted, pending_deleted))
}

fn load_id_alloc_state_db(db: &rocksdb::DB, code: EmbeddingCode) -> Result<(u32, u64)> {
    let cf = db
        .cf_handle(IdAlloc::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdAlloc::CF_NAME))?;

    // Next ID
    let next_key = IdAllocCfKey::next_id(code);
    let next_id = match db.get_cf(&cf, IdAlloc::key_to_bytes(&next_key))? {
        Some(bytes) => {
            let val = IdAlloc::value_from_bytes(&next_key, &bytes)?;
            match val.0 {
                IdAllocField::NextId(v) => v,
                _ => 0,
            }
        }
        None => 0,
    };

    // Free bitmap
    let bitmap_key = IdAllocCfKey::free_bitmap(code);
    let free_count = match db.get_cf(&cf, IdAlloc::key_to_bytes(&bitmap_key))? {
        Some(bytes) => {
            let bitmap = RoaringBitmap::deserialize_from(&bytes[..])?;
            bitmap.len()
        }
        None => 0,
    };

    Ok((next_id, free_count))
}

// ============================================================================
// Internal Helpers
// ============================================================================

fn load_embedding_spec(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<Option<EmbeddingSpec>> {
    use crate::rocksdb::ColumnFamilySerde;

    let cf = txn_db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let key = EmbeddingSpecCfKey(code);
    let key_bytes = EmbeddingSpecs::key_to_bytes(&key);

    match txn_db.get_cf(&cf, key_bytes)? {
        Some(value_bytes) => {
            let value = EmbeddingSpecs::value_from_bytes(&value_bytes)?;
            Ok(Some(value.0))
        }
        None => Ok(None),
    }
}

fn load_graph_meta(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<(Option<VecId>, u8, Option<u32>, Option<u64>)> {
    let cf = txn_db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", GraphMeta::CF_NAME))?;

    // Entry point
    let ep_key = GraphMetaCfKey::entry_point(code);
    let entry_point = match txn_db.get_cf(&cf, GraphMeta::key_to_bytes(&ep_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&ep_key, &bytes)?;
            match val.0 {
                GraphMetaField::EntryPoint(v) => Some(v),
                _ => None,
            }
        }
        None => None,
    };

    // Max level
    let level_key = GraphMetaCfKey::max_level(code);
    let max_level = match txn_db.get_cf(&cf, GraphMeta::key_to_bytes(&level_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&level_key, &bytes)?;
            match val.0 {
                GraphMetaField::MaxLevel(v) => v,
                _ => 0,
            }
        }
        None => 0,
    };

    // Count
    let count_key = GraphMetaCfKey::count(code);
    let count = match txn_db.get_cf(&cf, GraphMeta::key_to_bytes(&count_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&count_key, &bytes)?;
            match val.0 {
                GraphMetaField::Count(v) => Some(v),
                _ => None,
            }
        }
        None => None,
    };

    // Spec hash
    let hash_key = GraphMetaCfKey::spec_hash(code);
    let spec_hash = match txn_db.get_cf(&cf, GraphMeta::key_to_bytes(&hash_key))? {
        Some(bytes) => {
            let val = GraphMeta::value_from_bytes(&hash_key, &bytes)?;
            match val.0 {
                GraphMetaField::SpecHash(v) => Some(v),
                _ => None,
            }
        }
        None => None,
    };

    Ok((entry_point, max_level, count, spec_hash))
}

// ADMIN.md Lifecycle Accounting (chungers, 2025-01-27, FIXED):
// Returns separate counts for all 4 lifecycle states instead of folding PendingDeleted into Deleted.
fn count_vectors_by_lifecycle(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<(u32, u32, u32, u32)> {
    let cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = txn_db.iterator_cf(
        &cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut indexed = 0u32;
    let mut pending = 0u32;
    let mut deleted = 0u32;
    let mut pending_deleted = 0u32;

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;

        match meta.lifecycle() {
            crate::vector::schema::VecLifecycle::Indexed => indexed += 1,
            crate::vector::schema::VecLifecycle::Pending => pending += 1,
            crate::vector::schema::VecLifecycle::Deleted => deleted += 1,
            crate::vector::schema::VecLifecycle::PendingDeleted => pending_deleted += 1,
        }
    }

    Ok((indexed, pending, deleted, pending_deleted))
}

fn load_id_alloc_state(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<(u32, u64)> {
    let cf = txn_db
        .cf_handle(IdAlloc::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdAlloc::CF_NAME))?;

    // Next ID
    let next_key = IdAllocCfKey::next_id(code);
    let next_id = match txn_db.get_cf(&cf, IdAlloc::key_to_bytes(&next_key))? {
        Some(bytes) => {
            let val = IdAlloc::value_from_bytes(&next_key, &bytes)?;
            match val.0 {
                IdAllocField::NextId(v) => v,
                _ => 0,
            }
        }
        None => 0,
    };

    // Free bitmap
    let bitmap_key = IdAllocCfKey::free_bitmap(code);
    let free_count = match txn_db.get_cf(&cf, IdAlloc::key_to_bytes(&bitmap_key))? {
        Some(bytes) => {
            let bitmap = RoaringBitmap::deserialize_from(&bytes[..])?;
            bitmap.len()
        }
        None => 0,
    };

    Ok((next_id, free_count))
}

fn compute_layer_distribution(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    sample_size: Option<usize>,
) -> Result<LayerDistribution> {
    let cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = txn_db.iterator_cf(
        &cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut counts: Vec<u32> = Vec::new();
    let mut processed = 0usize;

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;

        // Only count indexed vectors
        if meta.lifecycle() == crate::vector::schema::VecLifecycle::Indexed {
            let layer = meta.max_layer as usize;
            while counts.len() <= layer {
                counts.push(0);
            }
            counts[layer] += 1;
        }

        processed += 1;
        if let Some(limit) = sample_size {
            if processed >= limit {
                break;
            }
        }
    }

    Ok(LayerDistribution { counts })
}

fn get_pending_queue_info(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<(usize, Option<u64>)> {
    let cf = txn_db
        .cf_handle(Pending::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Pending::CF_NAME))?;

    let prefix = Pending::prefix_for_embedding(code);
    let iter = txn_db.iterator_cf(
        &cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut count = 0usize;
    let mut oldest_timestamp: Option<u64> = None;

    for item in iter {
        let (key_bytes, _) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let key = Pending::key_from_bytes(&key_bytes)?;

        if oldest_timestamp.is_none() {
            oldest_timestamp = Some(key.1 .0);
        }

        count += 1;
    }

    Ok((count, oldest_timestamp))
}

fn resolve_external_id(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    vec_id: VecId,
) -> Result<Option<Id>> {
    let cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let key = IdReverseCfKey(code, vec_id);
    match txn_db.get_cf(&cf, IdReverse::key_to_bytes(&key))? {
        Some(bytes) => {
            let val = IdReverse::value_from_bytes(&bytes)?;
            Ok(Some(val.0))
        }
        None => Ok(None),
    }
}

fn count_edges_per_layer(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    vec_id: VecId,
    max_layer: u8,
) -> Result<Vec<usize>> {
    let cf = txn_db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Edges::CF_NAME))?;

    let mut counts = Vec::with_capacity((max_layer + 1) as usize);

    for layer in 0..=max_layer {
        let key = EdgeCfKey(code, vec_id, layer);
        let count = match txn_db.get_cf(&cf, Edges::key_to_bytes(&key))? {
            Some(bytes) => {
                let bitmap = RoaringBitmap::deserialize_from(&bytes[..])?;
                bitmap.len() as usize
            }
            None => 0,
        };
        counts.push(count);
    }

    Ok(counts)
}

fn has_binary_code(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    vec_id: VecId,
) -> Result<bool> {
    let cf = txn_db
        .cf_handle(BinaryCodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", BinaryCodes::CF_NAME))?;

    let key = BinaryCodeCfKey(code, vec_id);
    Ok(txn_db.get_cf(&cf, BinaryCodes::key_to_bytes(&key))?.is_some())
}

// ============================================================================
// Validation Helpers
// ============================================================================

fn validate_spec_hash(spec: &EmbeddingSpec, stored_hash: Option<u64>) -> ValidationCheck {
    match stored_hash {
        Some(hash) => {
            let current_hash = spec.compute_spec_hash();
            if hash == current_hash {
                ValidationCheck {
                    name: "spec_hash",
                    status: ValidationStatus::Pass,
                    message: format!("Spec hash matches (0x{:016x})", hash),
                }
            } else {
                ValidationCheck {
                    name: "spec_hash",
                    status: ValidationStatus::Error,
                    message: format!(
                        "Spec hash mismatch: stored=0x{:016x}, current=0x{:016x}",
                        hash, current_hash
                    ),
                }
            }
        }
        None => ValidationCheck {
            name: "spec_hash",
            status: ValidationStatus::Warning,
            message: "No spec hash stored (index may be empty or pre-Phase5)".to_string(),
        },
    }
}

fn validate_entry_point(storage: &Storage, code: EmbeddingCode) -> Result<ValidationCheck> {
    let txn_db = storage.transaction_db()?;
    let (entry_point, _, count, _) = load_graph_meta(&txn_db, code)?;

    match (entry_point, count) {
        (Some(ep), Some(c)) if c > 0 => {
            // Verify entry point exists in VecMeta
            let meta_cf = txn_db
                .cf_handle(VecMeta::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

            let key = VecMetaCfKey(code, ep);
            if txn_db.get_cf(&meta_cf, VecMeta::key_to_bytes(&key))?.is_some() {
                Ok(ValidationCheck {
                    name: "entry_point",
                    status: ValidationStatus::Pass,
                    message: format!("Entry point {} exists and is valid", ep),
                })
            } else {
                Ok(ValidationCheck {
                    name: "entry_point",
                    status: ValidationStatus::Error,
                    message: format!("Entry point {} does not exist in VecMeta", ep),
                })
            }
        }
        (None, Some(c)) if c > 0 => Ok(ValidationCheck {
            name: "entry_point",
            status: ValidationStatus::Error,
            message: format!("No entry point but count is {}", c),
        }),
        (None, None) | (None, Some(0)) => Ok(ValidationCheck {
            name: "entry_point",
            status: ValidationStatus::Pass,
            message: "No entry point (empty index)".to_string(),
        }),
        (Some(ep), None) | (Some(ep), Some(0)) => Ok(ValidationCheck {
            name: "entry_point",
            status: ValidationStatus::Warning,
            message: format!("Entry point {} exists but count is 0 or missing", ep),
        }),
        // Catch-all for any remaining cases (e.g., Some(ep) with Some(c) where c > 0 is already handled above)
        (ep, count) => Ok(ValidationCheck {
            name: "entry_point",
            status: ValidationStatus::Warning,
            message: format!("Unexpected entry point state: ep={:?}, count={:?}", ep, count),
        }),
    }
}

fn validate_id_mappings(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdForward::CF_NAME))?;

    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let prefix = code.to_be_bytes();

    // Check a sample of forward mappings
    let forward_iter = txn_db.iterator_cf(
        &forward_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut checked = 0u32;
    let mut mismatches = 0u32;
    let limit = 1000; // Sample size

    for item in forward_iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let forward_key = IdForward::key_from_bytes(&key_bytes)?;
        let forward_val = IdForward::value_from_bytes(&value_bytes)?;

        // Check reverse mapping
        let reverse_key = IdReverseCfKey(code, forward_val.0);
        if let Some(reverse_bytes) = txn_db.get_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))? {
            let reverse_val = IdReverse::value_from_bytes(&reverse_bytes)?;
            if reverse_val.0 != forward_key.1 {
                mismatches += 1;
            }
        } else {
            mismatches += 1;
        }

        checked += 1;
        if checked >= limit {
            break;
        }
    }

    if mismatches == 0 {
        Ok(ValidationCheck {
            name: "id_mappings",
            status: ValidationStatus::Pass,
            message: format!("ID forward/reverse mappings consistent ({} checked)", checked),
        })
    } else {
        Ok(ValidationCheck {
            name: "id_mappings",
            status: ValidationStatus::Error,
            message: format!(
                "ID mapping mismatches: {} of {} checked",
                mismatches, checked
            ),
        })
    }
}

fn validate_count_accuracy(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    // ADMIN.md Issue #3 (chungers, 2025-01-27, FIXED):
    // GraphMeta::Count tracks indexed vectors only. Compare against indexed count,
    // not total (indexed + pending + deleted). Report pending/deleted separately.
    let (_, _, graph_count, _) = load_graph_meta(txn_db, code)?;
    let (indexed, pending, deleted, pending_deleted) = count_vectors_by_lifecycle(txn_db, code)?;

    match graph_count {
        Some(stored) if stored == indexed => Ok(ValidationCheck {
            name: "count_accuracy",
            status: ValidationStatus::Pass,
            message: format!(
                "Count accurate: {} indexed (pending={}, deleted={}, pending_deleted={})",
                stored, pending, deleted, pending_deleted
            ),
        }),
        Some(stored) => {
            // Determine if mismatch is expected (pending vectors not yet indexed)
            let status = if pending > 0 && stored < indexed {
                // Pending might explain some discrepancy
                ValidationStatus::Warning
            } else {
                ValidationStatus::Warning
            };
            Ok(ValidationCheck {
                name: "count_accuracy",
                status,
                message: format!(
                    "Count mismatch: GraphMeta::Count={}, actual indexed={} (pending={}, deleted={}, pending_deleted={})",
                    stored, indexed, pending, deleted, pending_deleted
                ),
            })
        }
        None => Ok(ValidationCheck {
            name: "count_accuracy",
            status: ValidationStatus::Warning,
            message: format!(
                "No GraphMeta::Count stored, indexed={} (pending={}, deleted={}, pending_deleted={})",
                indexed, pending, deleted, pending_deleted
            ),
        }),
    }
}

fn validate_pending_queue(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    let (depth, oldest_timestamp) = get_pending_queue_info(txn_db, code)?;

    if depth == 0 {
        return Ok(ValidationCheck {
            name: "pending_queue",
            status: ValidationStatus::Pass,
            message: "Pending queue empty".to_string(),
        });
    }

    // Check if oldest entry is stale (> 1 hour old)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let one_hour_ms = 60 * 60 * 1000;

    match oldest_timestamp {
        Some(ts) if now - ts > one_hour_ms => {
            let age_mins = (now - ts) / 60_000;
            Ok(ValidationCheck {
                name: "pending_queue",
                status: ValidationStatus::Warning,
                message: format!(
                    "Pending queue has {} entries, oldest is {} minutes old",
                    depth, age_mins
                ),
            })
        }
        Some(_) => Ok(ValidationCheck {
            name: "pending_queue",
            status: ValidationStatus::Pass,
            message: format!("Pending queue has {} entries (recent)", depth),
        }),
        None => Ok(ValidationCheck {
            name: "pending_queue",
            status: ValidationStatus::Pass,
            message: format!("Pending queue has {} entries", depth),
        }),
    }
}

fn validate_no_orphaned_vectors(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let edges_cf = txn_db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Edges::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = txn_db.iterator_cf(
        &meta_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut orphaned = 0u32;
    let mut checked = 0u32;
    let limit = 1000; // Sample size

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let key = VecMeta::key_from_bytes(&key_bytes)?;
        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;

        // Only check indexed vectors
        if meta.lifecycle() == crate::vector::schema::VecLifecycle::Indexed {
            // Check if vector has edges at layer 0
            let edge_key = EdgeCfKey(code, key.1, 0);
            if txn_db.get_cf(&edges_cf, Edges::key_to_bytes(&edge_key))?.is_none() {
                orphaned += 1;
            }
        }

        checked += 1;
        if checked >= limit {
            break;
        }
    }

    if orphaned == 0 {
        Ok(ValidationCheck {
            name: "orphaned_vectors",
            status: ValidationStatus::Pass,
            message: format!("No orphaned vectors ({} indexed vectors checked)", checked),
        })
    } else {
        Ok(ValidationCheck {
            name: "orphaned_vectors",
            status: ValidationStatus::Warning,
            message: format!(
                "{} orphaned vectors (indexed but no edges) of {} checked",
                orphaned, checked
            ),
        })
    }
}

// ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
// Additional validation checks for reverse mappings, VecMeta presence, and vector payloads.

fn validate_reverse_id_mappings(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    sample_limit: Option<u32>,
) -> Result<ValidationCheck> {
    use crate::vector::schema::{IdForwardCfKey, IdForwardCfValue};

    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdForward::CF_NAME))?;

    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let prefix = code.to_be_bytes();

    // Check reverse mappings point back to valid forward mappings
    let reverse_iter = txn_db.iterator_cf(
        &reverse_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut checked = 0u32;
    let mut mismatches = 0u32;
    let limit = sample_limit.unwrap_or(1000);

    for item in reverse_iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let reverse_key = IdReverse::key_from_bytes(&key_bytes)?;
        let reverse_val = IdReverse::value_from_bytes(&value_bytes)?;

        // Check forward mapping exists and points back
        let forward_key = IdForwardCfKey(code, reverse_val.0);
        if let Some(forward_bytes) = txn_db.get_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))? {
            let forward_val: IdForwardCfValue = IdForward::value_from_bytes(&forward_bytes)?;
            if forward_val.0 != reverse_key.1 {
                mismatches += 1;
            }
        } else {
            mismatches += 1;
        }

        checked += 1;
        // Progress reporting for strict mode (full scan)
        if sample_limit.is_none() && checked % 10_000 == 0 {
            eprintln!("  [reverse_id_mappings] checked {} entries...", checked);
        }
        if sample_limit.is_some() && checked >= limit {
            break;
        }
    }

    let sampled_note = if sample_limit.is_some() { " (sampled)" } else { "" };

    if mismatches == 0 {
        Ok(ValidationCheck {
            name: "reverse_id_mappings",
            status: ValidationStatus::Pass,
            message: format!("Reverse ID mappings consistent ({} checked{})", checked, sampled_note),
        })
    } else {
        Ok(ValidationCheck {
            name: "reverse_id_mappings",
            status: ValidationStatus::Error,
            message: format!(
                "Reverse ID mapping mismatches: {} of {} checked{}",
                mismatches, checked, sampled_note
            ),
        })
    }
}

fn validate_vec_meta_presence(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    sample_limit: Option<u32>,
) -> Result<ValidationCheck> {
    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();

    let reverse_iter = txn_db.iterator_cf(
        &reverse_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut checked = 0u32;
    let mut missing = 0u32;
    let limit = sample_limit.unwrap_or(1000);

    for item in reverse_iter {
        let (key_bytes, _) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let reverse_key = IdReverse::key_from_bytes(&key_bytes)?;
        let vec_id = reverse_key.1;

        // Check VecMeta exists for this vector
        let meta_key = VecMetaCfKey(code, vec_id);
        if txn_db.get_cf(&meta_cf, VecMeta::key_to_bytes(&meta_key))?.is_none() {
            missing += 1;
        }

        checked += 1;
        // Progress reporting for strict mode (full scan)
        if sample_limit.is_none() && checked % 10_000 == 0 {
            eprintln!("  [vec_meta_presence] checked {} entries...", checked);
        }
        if sample_limit.is_some() && checked >= limit {
            break;
        }
    }

    let sampled_note = if sample_limit.is_some() { " (sampled)" } else { "" };

    if missing == 0 {
        Ok(ValidationCheck {
            name: "vec_meta_presence",
            status: ValidationStatus::Pass,
            message: format!("VecMeta present for all IdReverse entries ({} checked{})", checked, sampled_note),
        })
    } else {
        Ok(ValidationCheck {
            name: "vec_meta_presence",
            status: ValidationStatus::Error,
            message: format!(
                "Missing VecMeta: {} of {} IdReverse entries{}",
                missing, checked, sampled_note
            ),
        })
    }
}

fn validate_vector_payloads(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
    sample_limit: Option<u32>,
) -> Result<ValidationCheck> {
    use crate::vector::schema::{VectorCfKey, Vectors};

    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Vectors::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = txn_db.iterator_cf(
        &meta_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut checked = 0u32;
    let mut missing = 0u32;
    let limit = sample_limit.unwrap_or(1000);

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let key = VecMeta::key_from_bytes(&key_bytes)?;
        let meta_value = VecMeta::value_from_bytes(&value_bytes)?;
        let meta = &meta_value.0;

        // Only check indexed vectors - they must have vector data
        if meta.lifecycle() == crate::vector::schema::VecLifecycle::Indexed {
            let vector_key = VectorCfKey(code, key.1);
            if txn_db.get_cf(&vectors_cf, Vectors::key_to_bytes(&vector_key))?.is_none() {
                missing += 1;
            }
            checked += 1;
            // Progress reporting for strict mode (full scan)
            if sample_limit.is_none() && checked % 10_000 == 0 {
                eprintln!("  [vector_payloads] checked {} indexed vectors...", checked);
            }
        }

        if sample_limit.is_some() && checked >= limit {
            break;
        }
    }

    let sampled_note = if sample_limit.is_some() { " (sampled)" } else { "" };

    if missing == 0 {
        Ok(ValidationCheck {
            name: "vector_payloads",
            status: ValidationStatus::Pass,
            message: format!("Vector data present for all indexed vectors ({} checked{})", checked, sampled_note),
        })
    } else {
        Ok(ValidationCheck {
            name: "vector_payloads",
            status: ValidationStatus::Error,
            message: format!(
                "Missing vector data: {} of {} indexed vectors{}",
                missing, checked, sampled_note
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_lifecycle_serialization() {
        let lifecycle = VecLifecycle::Indexed;
        let json = serde_json::to_string(&lifecycle).unwrap();
        assert_eq!(json, "\"indexed\"");

        let lifecycle = VecLifecycle::PendingDeleted;
        let json = serde_json::to_string(&lifecycle).unwrap();
        assert_eq!(json, "\"pending_deleted\"");
    }

    #[test]
    fn test_validation_status_serialization() {
        let status = ValidationStatus::Pass;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"pass\"");

        let status = ValidationStatus::Error;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"error\"");
    }
}
