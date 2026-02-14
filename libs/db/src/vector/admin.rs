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
    EmbeddingSpecs, ExternalKey, GraphMeta, GraphMetaCfKey, GraphMetaField, IdAlloc, IdAllocCfKey,
    IdAllocField, IdForward, IdReverse, IdReverseCfKey, LifecycleCounts, LifecycleCountsCfKey,
    LifecycleCountsValue, Pending, VecId, VecMeta, VecMetaCfKey, ALL_COLUMN_FAMILIES,
};
use crate::vector::{EmbeddingSpec, Storage};

// ============================================================================
// DB Access Abstraction (ADMIN.md ADMIN-1 / Read-Only Mode)
// ============================================================================

// ADMIN.md ADMIN-1 (claude, 2025-01-27, FIXED):
// All admin commands now support --secondary via the AdminDb trait, which
// abstracts over rocksdb::DB (secondary/readonly) and TransactionDB (readwrite).
// This eliminates helper function duplication and satisfies ADMIN-1.

/// Read-only database access trait for admin operations.
///
/// Both `rocksdb::DB` (secondary/readonly mode) and `rocksdb::TransactionDB`
/// (readwrite mode) implement this trait, allowing all admin helpers to work
/// with either backend without code duplication.
pub(crate) trait AdminDb {
    fn cf_handle(&self, name: &str) -> Option<&rocksdb::ColumnFamily>;
    fn get_cf(&self, cf: &rocksdb::ColumnFamily, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, rocksdb::Error>;
    fn iterator_cf<'a>(
        &'a self,
        cf: &rocksdb::ColumnFamily,
        mode: IteratorMode<'_>,
    ) -> Box<dyn Iterator<Item = std::result::Result<(Box<[u8]>, Box<[u8]>), rocksdb::Error>> + 'a>;
    fn property_int_value_cf(
        &self,
        cf: &rocksdb::ColumnFamily,
        name: &str,
    ) -> std::result::Result<Option<u64>, rocksdb::Error>;
}

impl AdminDb for rocksdb::DB {
    fn cf_handle(&self, name: &str) -> Option<&rocksdb::ColumnFamily> {
        self.cf_handle(name)
    }
    fn get_cf(&self, cf: &rocksdb::ColumnFamily, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, rocksdb::Error> {
        self.get_cf(cf, key)
    }
    fn iterator_cf<'a>(
        &'a self,
        cf: &rocksdb::ColumnFamily,
        mode: IteratorMode<'_>,
    ) -> Box<dyn Iterator<Item = std::result::Result<(Box<[u8]>, Box<[u8]>), rocksdb::Error>> + 'a> {
        Box::new(rocksdb::DB::iterator_cf(self, cf, mode))
    }
    fn property_int_value_cf(
        &self,
        cf: &rocksdb::ColumnFamily,
        name: &str,
    ) -> std::result::Result<Option<u64>, rocksdb::Error> {
        self.property_int_value_cf(cf, name)
    }
}

impl AdminDb for rocksdb::TransactionDB {
    fn cf_handle(&self, name: &str) -> Option<&rocksdb::ColumnFamily> {
        self.cf_handle(name)
    }
    fn get_cf(&self, cf: &rocksdb::ColumnFamily, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, rocksdb::Error> {
        self.get_cf(cf, key)
    }
    fn iterator_cf<'a>(
        &'a self,
        cf: &rocksdb::ColumnFamily,
        mode: IteratorMode<'_>,
    ) -> Box<dyn Iterator<Item = std::result::Result<(Box<[u8]>, Box<[u8]>), rocksdb::Error>> + 'a> {
        Box::new(rocksdb::TransactionDB::iterator_cf(self, cf, mode))
    }
    fn property_int_value_cf(
        &self,
        _cf: &rocksdb::ColumnFamily,
        _name: &str,
    ) -> std::result::Result<Option<u64>, rocksdb::Error> {
        // property_int_value_cf is not available on TransactionDB
        Ok(None)
    }
}

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

/// Statistics grouped by external key type.
/// IDMAP T6.2: Counts of vectors by ExternalKey variant.
#[derive(Debug, Clone, Default, Serialize)]
pub struct KeyTypeStats {
    /// Embedding code (primary key).
    pub code: EmbeddingCode,
    /// Total vectors in this embedding.
    pub total_vectors: u64,
    /// Count of NodeId keys.
    pub node_id: u64,
    /// Count of NodeFragment keys.
    pub node_fragment: u64,
    /// Count of Edge keys.
    pub edge: u64,
    /// Count of EdgeFragment keys.
    pub edge_fragment: u64,
    /// Count of NodeSummary keys.
    pub node_summary: u64,
    /// Count of EdgeSummary keys.
    pub edge_summary: u64,
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
    /// External key (typed polymorphic ID).
    /// IDMAP T6.1: Changed from `external_id: Option<String>` to full ExternalKey.
    pub external_key: Option<ExternalKey>,
    /// External key type name (for display convenience).
    pub external_key_type: Option<String>,
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
    /// Whether validation was stopped early due to max_errors limit.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stopped_early: bool,
}

/// Options for validation operations.
///
/// ADMIN.md Validation UX (claude, 2025-01-29, ADDED):
/// Configurable sample size, max errors, and max entries for validation.
#[derive(Debug, Clone, Default)]
pub struct ValidationOptions {
    /// Sample size for non-strict validation (default 1000).
    /// Set to 0 for no limit (full scan).
    pub sample_size: u32,
    /// Maximum errors before stopping validation (0 = no limit).
    pub max_errors: u32,
    /// Maximum entries to check across all validations (0 = no limit).
    /// Only applies to strict mode.
    pub max_entries: u64,
}

impl ValidationOptions {
    /// Default options for non-strict validation.
    pub fn default_sampled() -> Self {
        Self {
            sample_size: 1000,
            max_errors: 0,
            max_entries: 0,
        }
    }

    /// Default options for strict validation (full scan).
    pub fn default_strict() -> Self {
        Self {
            sample_size: 0,
            max_errors: 0,
            max_entries: 0,
        }
    }

    /// Builder: set sample size.
    pub fn with_sample_size(mut self, size: u32) -> Self {
        self.sample_size = size;
        self
    }

    /// Builder: set max errors.
    pub fn with_max_errors(mut self, max: u32) -> Self {
        self.max_errors = max;
        self
    }

    /// Builder: set max entries for strict mode.
    pub fn with_max_entries(mut self, max: u64) -> Self {
        self.max_entries = max;
        self
    }
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
    get_embedding_stats_impl(storage.transaction_db()?, code)
}

/// Get statistics for a single embedding using secondary (read-only) mode.
pub fn get_embedding_stats_secondary(
    storage: &Storage,
    code: EmbeddingCode,
) -> Result<EmbeddingStats> {
    get_embedding_stats_impl(storage.db()?, code)
}

fn get_embedding_stats_impl(db: &dyn AdminDb, code: EmbeddingCode) -> Result<EmbeddingStats> {
    let spec = load_embedding_spec(db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;

    let (entry_point, max_level, graph_count, spec_hash) = load_graph_meta(db, code)?;
    let (indexed_count, pending_count, deleted_count, pending_deleted_count) =
        count_vectors_by_lifecycle(db, code)?;
    let (next_id, free_id_count) = load_id_alloc_state(db, code)?;

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

/// Get statistics for all embeddings.
pub fn get_all_stats(storage: &Storage) -> Result<Vec<EmbeddingStats>> {
    get_all_stats_impl(storage.transaction_db()?)
}

/// Get statistics for all embeddings using secondary (read-only) mode.
pub fn get_all_stats_secondary(storage: &Storage) -> Result<Vec<EmbeddingStats>> {
    get_all_stats_impl(storage.db()?)
}

fn get_all_stats_impl(db: &dyn AdminDb) -> Result<Vec<EmbeddingStats>> {
    let specs_cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut stats = Vec::new();
    let iter = db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            match get_embedding_stats_impl(db, code) {
                Ok(s) => stats.push(s),
                Err(e) => {
                    tracing::warn!("Failed to get stats for embedding {}: {}", code, e);
                }
            }
        }
    }

    Ok(stats)
}

/// Get statistics grouped by external key type for a single embedding.
/// IDMAP T6.2: Counts vectors by ExternalKey variant.
pub fn get_key_type_stats(storage: &Storage, code: EmbeddingCode) -> Result<KeyTypeStats> {
    get_key_type_stats_impl(storage.transaction_db()?, code)
}

/// Get key type statistics using secondary (read-only) mode.
pub fn get_key_type_stats_secondary(storage: &Storage, code: EmbeddingCode) -> Result<KeyTypeStats> {
    get_key_type_stats_impl(storage.db()?, code)
}

fn get_key_type_stats_impl(db: &dyn AdminDb, code: EmbeddingCode) -> Result<KeyTypeStats> {
    let reverse_cf = db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
        &reverse_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut stats = KeyTypeStats {
        code,
        ..Default::default()
    };

    for item in iter {
        let (key_bytes, value_bytes) = item?;

        if !key_bytes.starts_with(&prefix) {
            break;
        }

        let external_key = IdReverse::value_from_bytes(&value_bytes)?;
        stats.total_vectors += 1;

        match external_key.0.variant_name() {
            "NodeId" => stats.node_id += 1,
            "NodeFragment" => stats.node_fragment += 1,
            "Edge" => stats.edge += 1,
            "EdgeFragment" => stats.edge_fragment += 1,
            "NodeSummary" => stats.node_summary += 1,
            "EdgeSummary" => stats.edge_summary += 1,
            _ => {} // Unknown variant (should not happen)
        }
    }

    Ok(stats)
}

/// Get key type statistics for all embeddings.
pub fn get_all_key_type_stats(storage: &Storage) -> Result<Vec<KeyTypeStats>> {
    get_all_key_type_stats_impl(storage.transaction_db()?)
}

/// Get key type statistics for all embeddings using secondary (read-only) mode.
pub fn get_all_key_type_stats_secondary(storage: &Storage) -> Result<Vec<KeyTypeStats>> {
    get_all_key_type_stats_impl(storage.db()?)
}

fn get_all_key_type_stats_impl(db: &dyn AdminDb) -> Result<Vec<KeyTypeStats>> {
    let specs_cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut stats = Vec::new();
    let iter = db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            match get_key_type_stats_impl(db, code) {
                Ok(s) => stats.push(s),
                Err(e) => {
                    tracing::warn!("Failed to get key type stats for embedding {}: {}", code, e);
                }
            }
        }
    }

    Ok(stats)
}

/// Deep inspection of an embedding (includes layer distribution, pending queue).
pub fn inspect_embedding(storage: &Storage, code: EmbeddingCode) -> Result<EmbeddingInspection> {
    inspect_embedding_impl(storage.transaction_db()?, code)
}

/// Deep inspection using secondary (read-only) mode.
pub fn inspect_embedding_secondary(storage: &Storage, code: EmbeddingCode) -> Result<EmbeddingInspection> {
    inspect_embedding_impl(storage.db()?, code)
}

fn inspect_embedding_impl(db: &dyn AdminDb, code: EmbeddingCode) -> Result<EmbeddingInspection> {
    let stats = get_embedding_stats_impl(db, code)?;
    let spec = load_embedding_spec(db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;
    let layer_distribution = compute_layer_distribution(db, code, None)?;
    let (pending_queue_depth, oldest_pending_timestamp) = get_pending_queue_info(db, code)?;

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
    get_vector_info_impl(storage.transaction_db()?, code, vec_id)
}

/// Get vector info using secondary (read-only) mode.
pub fn get_vector_info_secondary(
    storage: &Storage,
    code: EmbeddingCode,
    vec_id: VecId,
) -> Result<Option<VectorInfo>> {
    get_vector_info_impl(storage.db()?, code, vec_id)
}

fn get_vector_info_impl(db: &dyn AdminDb, code: EmbeddingCode, vec_id: VecId) -> Result<Option<VectorInfo>> {
    let meta_cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let meta_key = VecMetaCfKey(code, vec_id);
    let meta_bytes = match db.get_cf(&meta_cf, &VecMeta::key_to_bytes(&meta_key))? {
        Some(b) => b,
        None => return Ok(None),
    };
    let meta_value = VecMeta::value_from_bytes(&meta_bytes)?;
    let meta = &meta_value.0;

    let external_key = resolve_external_key(db, code, vec_id)?;
    let external_key_type = external_key.as_ref().map(|k| k.variant_name().to_string());
    let edge_counts = count_edges_per_layer(db, code, vec_id, meta.max_layer)?;
    let has_binary_code = has_binary_code(db, code, vec_id)?;

    Ok(Some(VectorInfo {
        vec_id,
        external_key,
        external_key_type,
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
    list_vectors_by_state_impl(storage.transaction_db()?, code, state, limit)
}

/// List vectors by state using secondary (read-only) mode.
pub fn list_vectors_by_state_secondary(
    storage: &Storage,
    code: EmbeddingCode,
    state: VecLifecycle,
    limit: usize,
) -> Result<Vec<VectorInfo>> {
    list_vectors_by_state_impl(storage.db()?, code, state, limit)
}

fn list_vectors_by_state_impl(
    db: &dyn AdminDb,
    code: EmbeddingCode,
    state: VecLifecycle,
    limit: usize,
) -> Result<Vec<VectorInfo>> {
    let meta_cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
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
            let external_key = resolve_external_key(db, code, key.1)?;
            let external_key_type = external_key.as_ref().map(|k| k.variant_name().to_string());
            let edge_counts = count_edges_per_layer(db, code, key.1, meta.max_layer)?;
            let has_binary_code = has_binary_code(db, code, key.1)?;

            results.push(VectorInfo {
                vec_id: key.1,
                external_key,
                external_key_type,
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
    sample_vectors_impl(storage.transaction_db()?, code, count, seed)
}

/// Sample vectors using secondary (read-only) mode.
pub fn sample_vectors_secondary(
    storage: &Storage,
    code: EmbeddingCode,
    count: usize,
    seed: u64,
) -> Result<Vec<VectorInfo>> {
    sample_vectors_impl(storage.db()?, code, count, seed)
}

fn sample_vectors_impl(
    db: &dyn AdminDb,
    code: EmbeddingCode,
    count: usize,
    seed: u64,
) -> Result<Vec<VectorInfo>> {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let meta_cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
        &meta_cf,
        IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

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
            let j = rng.random_range(0..=total);
            if j < count {
                reservoir[j] = entry;
            }
        }
        total += 1;
    }

    let mut results = Vec::with_capacity(reservoir.len());
    for (vec_id, max_layer, lifecycle, created_at) in reservoir {
        let external_key = resolve_external_key(db, code, vec_id)?;
        let external_key_type = external_key.as_ref().map(|k| k.variant_name().to_string());
        let edge_counts = count_edges_per_layer(db, code, vec_id, max_layer)?;
        let has_binary_code = has_binary_code(db, code, vec_id)?;

        results.push(VectorInfo {
            vec_id,
            external_key,
            external_key_type,
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
    validate_embedding_with_opts(storage, code, ValidationOptions::default_sampled())
}

/// Run validation checks on an embedding with strict mode (no sampling).
///
/// ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
/// Added --strict option for full-scan validation without sampling.
pub fn validate_embedding_strict(storage: &Storage, code: EmbeddingCode) -> Result<ValidationResult> {
    validate_embedding_with_opts(storage, code, ValidationOptions::default_strict())
}

/// Run validation checks using secondary (read-only) mode.
pub fn validate_embedding_secondary(storage: &Storage, code: EmbeddingCode) -> Result<ValidationResult> {
    validate_embedding_secondary_with_opts(storage, code, ValidationOptions::default_sampled())
}

/// Run strict validation checks using secondary (read-only) mode.
pub fn validate_embedding_strict_secondary(storage: &Storage, code: EmbeddingCode) -> Result<ValidationResult> {
    validate_embedding_secondary_with_opts(storage, code, ValidationOptions::default_strict())
}

/// Run validation checks with custom options.
///
/// ADMIN.md Validation UX (claude, 2025-01-29, ADDED):
/// Configurable sample size, max errors, and max entries.
pub fn validate_embedding_with_opts(
    storage: &Storage,
    code: EmbeddingCode,
    opts: ValidationOptions,
) -> Result<ValidationResult> {
    validate_embedding_with_options_impl(storage.transaction_db()?, code, opts)
}

/// Run validation checks with custom options using secondary (read-only) mode.
pub fn validate_embedding_secondary_with_opts(
    storage: &Storage,
    code: EmbeddingCode,
    opts: ValidationOptions,
) -> Result<ValidationResult> {
    validate_embedding_with_options_impl(storage.db()?, code, opts)
}

fn validate_embedding_with_options_impl(
    db: &dyn AdminDb,
    code: EmbeddingCode,
    opts: ValidationOptions,
) -> Result<ValidationResult> {
    let spec = load_embedding_spec(db, code)?
        .ok_or_else(|| anyhow::anyhow!("Embedding {} not found", code))?;

    let mut checks = Vec::new();
    let mut stopped_early = false;
    let mut error_count = 0u32;

    // Sample limit: None for full scan (sample_size=0), Some(N) for sampling
    let sample_limit = if opts.sample_size == 0 { None } else { Some(opts.sample_size) };

    // Helper to check if we should stop early
    let should_stop = |checks: &[ValidationCheck], error_count: &mut u32| -> bool {
        if opts.max_errors > 0 {
            for check in checks.iter() {
                if check.status == ValidationStatus::Error {
                    *error_count += 1;
                }
            }
            *error_count >= opts.max_errors
        } else {
            false
        }
    };

    // 1. Spec hash validation
    let (_, _, _, spec_hash) = load_graph_meta(db, code)?;
    checks.push(validate_spec_hash(&spec, spec_hash));
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 2. Entry point validation
    checks.push(validate_entry_point(db, code)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 3. ID mapping consistency (forward -> reverse)
    checks.push(validate_id_mappings(db, code)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 4. Count accuracy
    checks.push(validate_count_accuracy(db, code)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 5. Pending queue staleness
    checks.push(validate_pending_queue(db, code)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 6. Orphaned vectors check
    checks.push(validate_no_orphaned_vectors(db, code)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
    // Additional checks for data integrity.

    // 7. Reverse ID mapping consistency (reverse -> forward)
    checks.push(validate_reverse_id_mappings(db, code, sample_limit, opts.max_entries)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 8. VecMeta presence for all IdReverse entries
    checks.push(validate_vec_meta_presence(db, code, sample_limit, opts.max_entries)?);
    if should_stop(&checks[checks.len()-1..], &mut error_count) {
        stopped_early = true;
        return Ok(ValidationResult { code, model: spec.model, checks, stopped_early });
    }

    // 9. Vector payload existence for indexed vectors
    checks.push(validate_vector_payloads(db, code, sample_limit, opts.max_entries)?);

    Ok(ValidationResult {
        code,
        model: spec.model,
        checks,
        stopped_early,
    })
}

/// Run validation checks on all embeddings.
pub fn validate_all(storage: &Storage) -> Result<Vec<ValidationResult>> {
    validate_all_with_opts(storage, ValidationOptions::default_sampled())
}

/// Run validation checks on all embeddings with strict mode (no sampling).
pub fn validate_all_strict(storage: &Storage) -> Result<Vec<ValidationResult>> {
    validate_all_with_opts(storage, ValidationOptions::default_strict())
}

/// Run validation checks on all embeddings using secondary (read-only) mode.
pub fn validate_all_secondary(storage: &Storage) -> Result<Vec<ValidationResult>> {
    validate_all_secondary_with_opts(storage, ValidationOptions::default_sampled())
}

/// Run strict validation checks on all embeddings using secondary (read-only) mode.
pub fn validate_all_strict_secondary(storage: &Storage) -> Result<Vec<ValidationResult>> {
    validate_all_secondary_with_opts(storage, ValidationOptions::default_strict())
}

/// Run validation checks on all embeddings with custom options.
pub fn validate_all_with_opts(storage: &Storage, opts: ValidationOptions) -> Result<Vec<ValidationResult>> {
    validate_all_with_options_impl(storage.transaction_db()?, opts)
}

/// Run validation checks on all embeddings with custom options using secondary mode.
pub fn validate_all_secondary_with_opts(storage: &Storage, opts: ValidationOptions) -> Result<Vec<ValidationResult>> {
    validate_all_with_options_impl(storage.db()?, opts)
}

fn validate_all_with_options_impl(db: &dyn AdminDb, opts: ValidationOptions) -> Result<Vec<ValidationResult>> {
    let specs_cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut results = Vec::new();
    let iter = db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            match validate_embedding_with_options_impl(db, code, opts.clone()) {
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
/// Uses property_int_value_cf for estimated key counts when available
/// (returns None on TransactionDB, populated values on secondary DB).
pub fn get_rocksdb_stats(storage: &Storage) -> Result<RocksDbStats> {
    get_rocksdb_stats_impl(storage.transaction_db()?)
}

/// Get RocksDB-level statistics using secondary (read-only) mode.
///
/// ADMIN.md RocksDB Properties (chungers, 2025-01-27, ADDED):
/// Secondary mode queries rocksdb.estimate-num-keys property for better estimates,
/// which is only available on DB (secondary mode), not TransactionDB.
pub fn get_rocksdb_stats_secondary(storage: &Storage) -> Result<RocksDbStats> {
    get_rocksdb_stats_impl(storage.db()?)
}

fn get_rocksdb_stats_impl(db: &dyn AdminDb) -> Result<RocksDbStats> {
    let mut column_families = Vec::new();
    let mut total_size_bytes = 0u64;

    for cf_name in ALL_COLUMN_FAMILIES {
        if let Some(cf) = db.cf_handle(cf_name) {
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

// ============================================================================
// Lifecycle Counter Migration
// ============================================================================

/// Result of migrating lifecycle counters for an embedding.
#[derive(Debug, Clone, Serialize)]
pub struct MigrateCountsResult {
    /// Embedding code
    pub code: EmbeddingCode,
    /// Counted indexed vectors
    pub indexed: u64,
    /// Counted pending vectors
    pub pending: u64,
    /// Counted deleted vectors
    pub deleted: u64,
    /// Counted pending_deleted vectors
    pub pending_deleted: u64,
}

/// Migrate lifecycle counters for a single embedding by scanning VecMeta and
/// writing the counts to LifecycleCounts CF.
///
/// This is a one-time migration operation for existing databases that don't
/// have lifecycle counters initialized. The counters enable O(1) stats queries.
///
/// Note: This must be run in readwrite mode (not secondary).
pub fn migrate_lifecycle_counts(
    storage: &Storage,
    code: EmbeddingCode,
) -> Result<MigrateCountsResult> {
    let txn_db = storage.transaction_db()?;
    migrate_lifecycle_counts_impl(txn_db, code)
}

/// Migrate lifecycle counters for all embeddings.
pub fn migrate_all_lifecycle_counts(storage: &Storage) -> Result<Vec<MigrateCountsResult>> {
    let txn_db = storage.transaction_db()?;
    let codes = list_embedding_codes(txn_db)?;

    let mut results = Vec::with_capacity(codes.len());
    for code in codes {
        results.push(migrate_lifecycle_counts_impl(txn_db, code)?);
    }
    Ok(results)
}

fn migrate_lifecycle_counts_impl(
    txn_db: &rocksdb::TransactionDB,
    code: EmbeddingCode,
) -> Result<MigrateCountsResult> {
    // Scan VecMeta to count vectors by lifecycle state
    let (indexed, pending, deleted, pending_deleted) = count_vectors_by_lifecycle_scan(txn_db, code)?;

    // Write counts to LifecycleCounts CF
    let lifecycle_cf = txn_db
        .cf_handle(LifecycleCounts::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("LifecycleCounts CF not found"))?;

    let key = LifecycleCountsCfKey(code);
    let value = LifecycleCountsValue {
        indexed: indexed as u64,
        pending: pending as u64,
        deleted: deleted as u64,
        pending_deleted: pending_deleted as u64,
    };

    txn_db.put_cf(&lifecycle_cf, LifecycleCounts::key_to_bytes(&key), value.to_bytes())?;

    Ok(MigrateCountsResult {
        code,
        indexed: indexed as u64,
        pending: pending as u64,
        deleted: deleted as u64,
        pending_deleted: pending_deleted as u64,
    })
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// List all embedding codes by scanning EmbeddingSpecs CF.
fn list_embedding_codes(db: &dyn AdminDb) -> Result<Vec<EmbeddingCode>> {
    let specs_cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let mut codes = Vec::new();
    let iter = db.iterator_cf(&specs_cf, IteratorMode::Start);

    for item in iter {
        let (key_bytes, _) = item?;
        if key_bytes.len() == 8 {
            let code = u64::from_be_bytes(key_bytes[..].try_into()?);
            codes.push(code);
        }
    }

    Ok(codes)
}

fn load_embedding_spec(
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<Option<EmbeddingSpec>> {
    use crate::rocksdb::ColumnFamilySerde;

    let cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

    let key = EmbeddingSpecCfKey(code);
    let key_bytes = EmbeddingSpecs::key_to_bytes(&key);

    match db.get_cf(&cf, &key_bytes)? {
        Some(value_bytes) => {
            let value = EmbeddingSpecs::value_from_bytes(&value_bytes)?;
            Ok(Some(value.0))
        }
        None => Ok(None),
    }
}

fn load_graph_meta(
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<(Option<VecId>, u8, Option<u32>, Option<u64>)> {
    let cf = db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", GraphMeta::CF_NAME))?;

    // Entry point
    let ep_key = GraphMetaCfKey::entry_point(code);
    let entry_point = match db.get_cf(&cf, &GraphMeta::key_to_bytes(&ep_key))? {
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
    let max_level = match db.get_cf(&cf, &GraphMeta::key_to_bytes(&level_key))? {
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
    let count = match db.get_cf(&cf, &GraphMeta::key_to_bytes(&count_key))? {
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
    let spec_hash = match db.get_cf(&cf, &GraphMeta::key_to_bytes(&hash_key))? {
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
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<(u32, u32, u32, u32)> {
    // First try reading from LifecycleCounts CF (O(1))
    if let Some(counts) = read_lifecycle_counts(db, code)? {
        return Ok((
            counts.indexed as u32,
            counts.pending as u32,
            counts.deleted as u32,
            counts.pending_deleted as u32,
        ));
    }

    // Fallback: Full VecMeta scan (O(N))
    eprintln!(
        "Warning: lifecycle counters not found for embedding {}. \
         Falling back to VecMeta scan (O(N)). \
         Run `admin migrate-lifecycle-counts` to initialize counters.",
        code
    );
    count_vectors_by_lifecycle_scan(db, code)
}

/// Read lifecycle counts from the LifecycleCounts CF.
/// Returns None if counters are not yet initialized for this embedding.
fn read_lifecycle_counts(
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<Option<LifecycleCountsValue>> {
    let cf = match db.cf_handle(LifecycleCounts::CF_NAME) {
        Some(cf) => cf,
        None => return Ok(None), // CF doesn't exist (old DB)
    };

    let key = LifecycleCountsCfKey(code);
    match db.get_cf(&cf, &LifecycleCounts::key_to_bytes(&key))? {
        Some(bytes) => Ok(Some(LifecycleCountsValue::from_bytes(&bytes)?)),
        None => Ok(None),
    }
}

/// Count vectors by scanning VecMeta (O(N) fallback).
fn count_vectors_by_lifecycle_scan(
    db: &dyn AdminDb,
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

fn load_id_alloc_state(
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<(u32, u64)> {
    let cf = db
        .cf_handle(IdAlloc::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdAlloc::CF_NAME))?;

    // Next ID
    let next_key = IdAllocCfKey::next_id(code);
    let next_id = match db.get_cf(&cf, &IdAlloc::key_to_bytes(&next_key))? {
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
    let free_count = match db.get_cf(&cf, &IdAlloc::key_to_bytes(&bitmap_key))? {
        Some(bytes) => {
            let bitmap = RoaringBitmap::deserialize_from(&bytes[..])?;
            bitmap.len()
        }
        None => 0,
    };

    Ok((next_id, free_count))
}

fn compute_layer_distribution(
    db: &dyn AdminDb,
    code: EmbeddingCode,
    sample_size: Option<usize>,
) -> Result<LayerDistribution> {
    let cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
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
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<(usize, Option<u64>)> {
    let cf = db
        .cf_handle(Pending::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Pending::CF_NAME))?;

    let prefix = Pending::prefix_for_embedding(code);
    let iter = db.iterator_cf(
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

/// Resolve the external key for a vector.
/// IDMAP T6.1: Returns full ExternalKey instead of just Id.
fn resolve_external_key(
    db: &dyn AdminDb,
    code: EmbeddingCode,
    vec_id: VecId,
) -> Result<Option<ExternalKey>> {
    let cf = db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let key = IdReverseCfKey(code, vec_id);
    match db.get_cf(&cf, &IdReverse::key_to_bytes(&key))? {
        Some(bytes) => {
            let val = IdReverse::value_from_bytes(&bytes)?;
            Ok(Some(val.0))
        }
        None => Ok(None),
    }
}

fn count_edges_per_layer(
    db: &dyn AdminDb,
    code: EmbeddingCode,
    vec_id: VecId,
    max_layer: u8,
) -> Result<Vec<usize>> {
    let cf = db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Edges::CF_NAME))?;

    let mut counts = Vec::with_capacity((max_layer + 1) as usize);

    for layer in 0..=max_layer {
        let key = EdgeCfKey(code, vec_id, layer);
        let count = match db.get_cf(&cf, &Edges::key_to_bytes(&key))? {
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
    db: &dyn AdminDb,
    code: EmbeddingCode,
    vec_id: VecId,
) -> Result<bool> {
    let cf = db
        .cf_handle(BinaryCodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", BinaryCodes::CF_NAME))?;

    let key = BinaryCodeCfKey(code, vec_id);
    Ok(db.get_cf(&cf, &BinaryCodes::key_to_bytes(&key))?.is_some())
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

fn validate_entry_point(db: &dyn AdminDb, code: EmbeddingCode) -> Result<ValidationCheck> {
    let (entry_point, _, count, _) = load_graph_meta(db, code)?;

    match (entry_point, count) {
        (Some(ep), Some(c)) if c > 0 => {
            // Verify entry point exists in VecMeta
            let meta_cf = db
                .cf_handle(VecMeta::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

            let key = VecMetaCfKey(code, ep);
            if db.get_cf(&meta_cf, &VecMeta::key_to_bytes(&key))?.is_some() {
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
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    let forward_cf = db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdForward::CF_NAME))?;

    let reverse_cf = db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let prefix = code.to_be_bytes();

    // Check a sample of forward mappings
    let forward_iter = db.iterator_cf(
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
        if let Some(reverse_bytes) = db.get_cf(&reverse_cf, &IdReverse::key_to_bytes(&reverse_key))? {
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
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    // ADMIN.md Issue #3 (chungers, 2025-01-27, FIXED):
    // GraphMeta::Count tracks indexed vectors only. Compare against indexed count,
    // not total (indexed + pending + deleted). Report pending/deleted separately.
    let (_, _, graph_count, _) = load_graph_meta(db, code)?;
    let (indexed, pending, deleted, pending_deleted) = count_vectors_by_lifecycle(db, code)?;

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
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    let (depth, oldest_timestamp) = get_pending_queue_info(db, code)?;

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
    db: &dyn AdminDb,
    code: EmbeddingCode,
) -> Result<ValidationCheck> {
    let meta_cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let edges_cf = db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Edges::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
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
            if db.get_cf(&edges_cf, &Edges::key_to_bytes(&edge_key))?.is_none() {
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
    db: &dyn AdminDb,
    code: EmbeddingCode,
    sample_limit: Option<u32>,
    max_entries: u64,
) -> Result<ValidationCheck> {
    use crate::vector::schema::{IdForwardCfKey, IdForwardCfValue};

    let forward_cf = db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdForward::CF_NAME))?;

    let reverse_cf = db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let prefix = code.to_be_bytes();

    // Check reverse mappings point back to valid forward mappings
    let reverse_iter = db.iterator_cf(
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
        if let Some(forward_bytes) = db.get_cf(&forward_cf, &IdForward::key_to_bytes(&forward_key))? {
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
        // Stop at sample limit or max entries
        if sample_limit.is_some() && checked >= limit {
            break;
        }
        if max_entries > 0 && checked as u64 >= max_entries {
            break;
        }
    }

    let sampled_note = if sample_limit.is_some() {
        format!(" (sample_size={})", limit)
    } else if max_entries > 0 {
        format!(" (max_entries={})", max_entries)
    } else {
        String::new()
    };

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
    db: &dyn AdminDb,
    code: EmbeddingCode,
    sample_limit: Option<u32>,
    max_entries: u64,
) -> Result<ValidationCheck> {
    let reverse_cf = db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdReverse::CF_NAME))?;

    let meta_cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let prefix = code.to_be_bytes();

    let reverse_iter = db.iterator_cf(
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
        if db.get_cf(&meta_cf, &VecMeta::key_to_bytes(&meta_key))?.is_none() {
            missing += 1;
        }

        checked += 1;
        // Progress reporting for strict mode (full scan)
        if sample_limit.is_none() && checked % 10_000 == 0 {
            eprintln!("  [vec_meta_presence] checked {} entries...", checked);
        }
        // Stop at sample limit or max entries
        if sample_limit.is_some() && checked >= limit {
            break;
        }
        if max_entries > 0 && checked as u64 >= max_entries {
            break;
        }
    }

    let sampled_note = if sample_limit.is_some() {
        format!(" (sample_size={})", limit)
    } else if max_entries > 0 {
        format!(" (max_entries={})", max_entries)
    } else {
        String::new()
    };

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
    db: &dyn AdminDb,
    code: EmbeddingCode,
    sample_limit: Option<u32>,
    max_entries: u64,
) -> Result<ValidationCheck> {
    use crate::vector::schema::{VectorCfKey, Vectors};

    let meta_cf = db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", VecMeta::CF_NAME))?;

    let vectors_cf = db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("CF {} not found", Vectors::CF_NAME))?;

    let prefix = code.to_be_bytes();
    let iter = db.iterator_cf(
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
            if db.get_cf(&vectors_cf, &Vectors::key_to_bytes(&vector_key))?.is_none() {
                missing += 1;
            }
            checked += 1;
            // Progress reporting for strict mode (full scan)
            if sample_limit.is_none() && checked % 10_000 == 0 {
                eprintln!("  [vector_payloads] checked {} indexed vectors...", checked);
            }
        }

        // Stop at sample limit or max entries
        if sample_limit.is_some() && checked >= limit {
            break;
        }
        if max_entries > 0 && checked as u64 >= max_entries {
            break;
        }
    }

    let sampled_note = if sample_limit.is_some() {
        format!(" (sample_size={})", limit)
    } else if max_entries > 0 {
        format!(" (max_entries={})", max_entries)
    } else {
        String::new()
    };

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
