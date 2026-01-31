//! Embedding registry for managing embedding spaces.
//!
//! Design rationale (ARCH-17):
//! - Registry allocates unique codes for embedding spaces
//! - Provides queryable discovery API
//! - Pre-warms from RocksDB on startup (following NameCache pattern)
//! - Stores storage reference to prevent DB misuse across calls

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use rocksdb::{IteratorMode, TransactionDB};
use std::sync::OnceLock;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, MutationCodec};
use super::distance::Distance;
use super::embedding::{Embedding, EmbeddingBuilder};
use super::mutation::AddEmbeddingSpec;
use super::schema::{EmbeddingSpec, EmbeddingSpecCfKey, EmbeddingSpecs};

// ============================================================================
// EmbeddingFilter
// ============================================================================

/// Filter for querying embedding spaces.
///
/// # Example
///
/// ```ignore
/// let filter = EmbeddingFilter::default()
///     .model("gemma")
///     .distance(Distance::Cosine);
///
/// let matches = registry.find(&filter);
/// ```
#[derive(Default, Debug, Clone)]
pub struct EmbeddingFilter {
    /// Filter by model name (exact match)
    pub model: Option<String>,
    /// Filter by dimensionality
    pub dim: Option<u32>,
    /// Filter by distance metric
    pub distance: Option<Distance>,
}

impl EmbeddingFilter {
    /// Filter by model name.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Filter by dimensionality.
    pub fn dim(mut self, dim: u32) -> Self {
        self.dim = Some(dim);
        self
    }

    /// Filter by distance metric.
    pub fn distance(mut self, distance: Distance) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Check if an embedding matches this filter.
    fn matches(&self, emb: &Embedding) -> bool {
        if let Some(ref model) = self.model {
            if emb.model() != model {
                return false;
            }
        }
        if let Some(dim) = self.dim {
            if emb.dim() != dim {
                return false;
            }
        }
        if let Some(distance) = self.distance {
            if emb.distance() != distance {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// EmbeddingRegistry
// ============================================================================

/// Registry key: (model, dim, distance)
type SpecKey = (String, u32, Distance);

/// Thread-safe registry for embedding spaces.
///
/// Stores a reference to storage to ensure consistent DB usage across all
/// operations (prevents footgun of using different DBs for prewarm vs register).
///
/// Pre-warms from RocksDB on startup following the NameCache pattern
/// from `graph/name_hash.rs`.
///
/// # Example
///
/// ```ignore
/// let registry = EmbeddingRegistry::new(storage.clone());
/// registry.prewarm()?;
///
/// // Register new embedding space
/// let gemma = registry.register(
///     EmbeddingBuilder::new("gemma", 768, Distance::Cosine),
/// )?;
///
/// // Query existing spaces
/// let all_gemma = registry.find_by_model("gemma");
/// ```
pub struct EmbeddingRegistry {
    /// (model, dim, distance) -> Embedding mapping
    by_spec: DashMap<SpecKey, Embedding>,
    /// code -> Embedding mapping (for lookup by code)
    by_code: DashMap<u64, Embedding>,
    /// Next code to allocate
    next_code: AtomicU64,
    /// Storage reference (set once via set_storage or new)
    storage: OnceLock<Arc<super::Storage>>,
}

impl EmbeddingRegistry {
    /// Create a new registry with storage reference.
    ///
    /// The storage is used for all DB operations (prewarm, register).
    pub fn new(storage: Arc<super::Storage>) -> Self {
        let cell = OnceLock::new();
        let _ = cell.set(storage);
        Self {
            by_spec: DashMap::new(),
            by_code: DashMap::new(),
            next_code: AtomicU64::new(1), // Start at 1 (0 is reserved)
            storage: cell,
        }
    }

    /// Create an empty registry without storage.
    ///
    /// Storage must be set via `set_storage()` before calling `prewarm()` or `register()`.
    /// This is primarily for use with Subsystem where storage isn't available at construction.
    pub fn new_without_storage() -> Self {
        Self {
            by_spec: DashMap::new(),
            by_code: DashMap::new(),
            next_code: AtomicU64::new(1), // Start at 1 (0 is reserved)
            storage: OnceLock::new(),
        }
    }

    /// Set the storage reference (can only be set once).
    ///
    /// Returns error if storage was already set.
    pub fn set_storage(&self, storage: Arc<super::Storage>) -> Result<()> {
        self.storage
            .set(storage)
            .map_err(|_| anyhow::anyhow!("Storage already set"))
    }

    /// Get the storage reference, or error if not set.
    fn storage(&self) -> Result<&Arc<super::Storage>> {
        self.storage
            .get()
            .ok_or_else(|| anyhow::anyhow!("Storage not set - call set_storage() first"))
    }

    /// Get a transaction from the stored storage.
    pub fn transaction(&self) -> Result<rocksdb::Transaction<'_, TransactionDB>> {
        let storage = self.storage()?;
        let db = storage.transaction_db()?;
        Ok(db.transaction())
    }

    /// Pre-warm the registry by loading all entries from RocksDB.
    ///
    /// Unlike NameCache (which limits to N entries), we load everything
    /// since we expect <1000 embeddings total.
    pub fn prewarm(&self) -> Result<usize> {
        let storage = self.storage()?;
        let db = storage.transaction_db()?;

        let cf = db
            .cf_handle(EmbeddingSpecs::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

        let iter = db.iterator_cf(&cf, IteratorMode::Start);

        let mut count = 0usize;
        let mut max_code = 0u64;

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            let key = EmbeddingSpecs::key_from_bytes(&key_bytes)?;
            let value = EmbeddingSpecs::value_from_bytes(&value_bytes)?;

            let code = key.0;
            let spec = &value.0;
            let embedding = Embedding::new(
                code,
                spec.model.clone(),
                spec.dim,
                spec.distance,
                spec.storage_type,
                None,
            );

            let spec_key = (spec.model.clone(), spec.dim, spec.distance);
            self.by_spec.insert(spec_key, embedding.clone());
            self.by_code.insert(code, embedding);

            max_code = max_code.max(code);
            count += 1;
        }

        // Set next_code to max + 1
        if count > 0 {
            self.next_code.store(max_code + 1, Ordering::SeqCst);
        }

        Ok(count)
    }

    // ─────────────────────────────────────────────────────────────
    // Registration
    // ─────────────────────────────────────────────────────────────

    /// Register a new embedding space (standalone transaction).
    ///
    /// Idempotent: returns existing embedding if spec already registered.
    /// Build parameters are only validated for NEW registrations.
    ///
    /// For atomic registration with other writes, use `register_in_txn()`.
    ///
    /// # Errors
    ///
    /// Returns an error if build parameters are invalid (new registration only):
    /// - `hnsw_m < 2`: Invalid for `m_l = 1/ln(m)` computation
    /// - `hnsw_ef_construction < 1`: Invalid HNSW config
    /// - `rabitq_bits ∉ {1, 2, 4}`: Unsupported quantization level
    pub fn register(&self, builder: EmbeddingBuilder) -> Result<Embedding> {
        let txn = self.transaction()?;
        let embedding = self.register_in_txn(builder, &txn)?;
        txn.commit()?;
        Ok(embedding)
    }

    /// Register a new embedding space within an existing transaction.
    ///
    /// Idempotent: returns existing embedding if spec already registered.
    /// Build parameters are only validated for NEW registrations.
    ///
    /// # Arguments
    /// * `builder` - Embedding specification builder
    /// * `txn` - Active transaction to write within
    ///
    /// # Returns
    /// The registered `Embedding`. Caller must call `txn.commit()` to persist.
    ///
    /// # Example
    /// ```rust,ignore
    /// let txn = registry.transaction()?;
    /// let embedding = registry.register_in_txn(builder, &txn)?;
    /// // ... other writes to txn ...
    /// txn.commit()?;
    /// ```
    pub fn register_in_txn(
        &self,
        builder: EmbeddingBuilder,
        txn: &rocksdb::Transaction<'_, TransactionDB>,
    ) -> Result<Embedding> {
        let storage = self.storage()?;
        let db = storage.transaction_db()?;
        let spec_key = (builder.model.clone(), builder.dim, builder.distance);

        // Fast path: already registered (idempotent - no validation needed)
        // This preserves idempotent behavior: caller can retrieve existing
        // embedding even with invalid build params in the builder.
        if let Some(existing) = self.by_spec.get(&spec_key) {
            let mut emb = existing.clone();
            // Attach embedder if provided
            if let Some(embedder) = builder.embedder {
                emb = emb.with_embedder(embedder);
                // Update cache with embedder
                self.by_spec.insert(spec_key.clone(), emb.clone());
                self.by_code.insert(emb.code(), emb.clone());
            }
            return Ok(emb);
        }

        // Slow path: new registration - validate build parameters
        builder.validate()?;

        // Allocate new code and persist
        let code = self.next_code.fetch_add(1, Ordering::SeqCst);

        // Get CF handle for embedding specs
        let cf = db
            .cf_handle(EmbeddingSpecs::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("CF {} not found", EmbeddingSpecs::CF_NAME))?;

        let add_op = AddEmbeddingSpec {
            code,
            model: builder.model.clone(),
            dim: builder.dim,
            distance: builder.distance,
            storage_type: super::schema::VectorElementType::default(), // F32 for backward compat
            // Build parameters from EmbeddingBuilder (Phase 5.7c)
            hnsw_m: builder.hnsw_m,
            hnsw_ef_construction: builder.hnsw_ef_construction,
            rabitq_bits: builder.rabitq_bits,
            rabitq_seed: builder.rabitq_seed,
        };
        let (key_bytes, value_bytes) = add_op.to_cf_bytes()?;

        // Write to transaction (caller commits)
        txn.put_cf(&cf, key_bytes, value_bytes)?;

        // Create embedding
        let embedding = Embedding::new(
            code,
            builder.model,
            builder.dim,
            builder.distance,
            super::schema::VectorElementType::default(), // F32 for backward compat
            builder.embedder,
        );

        // Update cache
        self.by_spec.insert(spec_key, embedding.clone());
        self.by_code.insert(code, embedding.clone());

        Ok(embedding)
    }

    // ─────────────────────────────────────────────────────────────
    // Lookup
    // ─────────────────────────────────────────────────────────────

    /// Get embedding by exact spec (None if not registered).
    pub fn get(&self, model: &str, dim: u32, distance: Distance) -> Option<Embedding> {
        let spec_key = (model.to_string(), dim, distance);
        self.by_spec.get(&spec_key).map(|e| e.clone())
    }

    /// Get embedding by code (for deserialization from storage).
    pub fn get_by_code(&self, code: u64) -> Option<Embedding> {
        self.by_code.get(&code).map(|e| e.clone())
    }

    /// Get the full EmbeddingSpec by code (reads from storage).
    ///
    /// Unlike `get_by_code()` which returns a lightweight `Embedding` handle,
    /// this method reads the full specification from RocksDB including HNSW
    /// and RaBitQ build parameters.
    ///
    /// # Returns
    /// - `Ok(Some(spec))` if found
    /// - `Ok(None)` if code not found in storage
    /// - `Err` if storage access fails
    ///
    /// # Fields in EmbeddingSpec
    /// - `model`, `dim`, `distance`, `storage_type` (basic fields)
    /// - `hnsw_m`, `hnsw_ef_construction` (HNSW build params)
    /// - `rabitq_bits`, `rabitq_seed` (RaBitQ build params)
    pub fn get_spec_by_code(&self, code: u64) -> Result<Option<EmbeddingSpec>> {
        let storage = self.storage()?;
        let db = storage.transaction_db()?;
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

    /// Attach/update embedder for existing space (runtime configuration).
    pub fn set_embedder(
        &self,
        code: u64,
        embedder: Arc<dyn super::embedding::Embedder>,
    ) -> Result<()> {
        let mut emb = self
            .by_code
            .get(&code)
            .map(|e| e.clone())
            .ok_or_else(|| anyhow::anyhow!("Unknown embedding code: {}", code))?;

        emb = emb.with_embedder(embedder);

        // Update both caches
        let spec_key = (emb.model().to_string(), emb.dim(), emb.distance());
        self.by_spec.insert(spec_key, emb.clone());
        self.by_code.insert(code, emb);

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────
    // Query / Discovery
    // ─────────────────────────────────────────────────────────────

    /// List all registered embedding spaces.
    pub fn list_all(&self) -> Vec<Embedding> {
        self.by_code.iter().map(|e| e.value().clone()).collect()
    }

    /// Find all spaces using a specific model.
    pub fn find_by_model(&self, model: &str) -> Vec<Embedding> {
        self.by_code
            .iter()
            .filter(|e| e.value().model() == model)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Find all spaces using a specific distance metric.
    pub fn find_by_distance(&self, distance: Distance) -> Vec<Embedding> {
        self.by_code
            .iter()
            .filter(|e| e.value().distance() == distance)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Find all spaces with a specific dimension.
    pub fn find_by_dim(&self, dim: u32) -> Vec<Embedding> {
        self.by_code
            .iter()
            .filter(|e| e.value().dim() == dim)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Find with multiple filters (AND logic).
    pub fn find(&self, filter: &EmbeddingFilter) -> Vec<Embedding> {
        self.by_code
            .iter()
            .filter(|e| filter.matches(e.value()))
            .map(|e| e.value().clone())
            .collect()
    }

    /// Count registered embedding spaces.
    pub fn len(&self) -> usize {
        self.by_code.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.by_code.is_empty()
    }

    // ─────────────────────────────────────────────────────────────
    // Internal: Pre-warming from read-only storage
    // ─────────────────────────────────────────────────────────────

    /// Register an embedding from database values (used during pre-warming).
    ///
    /// This method is used by `vector::Storage` when pre-warming from
    /// read-only DB (where `prewarm(&TransactionDB)` isn't available).
    pub(crate) fn register_from_db(
        &self,
        code: u64,
        model: &str,
        dim: u32,
        distance: Distance,
        storage_type: super::schema::VectorElementType,
    ) {
        let embedding = Embedding::new(code, model, dim, distance, storage_type, None);
        let spec_key = (model.to_string(), dim, distance);

        self.by_spec.insert(spec_key, embedding.clone());
        self.by_code.insert(code, embedding);

        // Update next_code if needed
        let current_next = self.next_code.load(Ordering::SeqCst);
        if code >= current_next {
            self.next_code.store(code + 1, Ordering::SeqCst);
        }
    }
}

impl Default for EmbeddingRegistry {
    fn default() -> Self {
        Self::new_without_storage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_filter() {
        let emb = Embedding::new(1, "gemma", 768, Distance::Cosine, crate::vector::schema::VectorElementType::default(), None);

        // Empty filter matches everything
        assert!(EmbeddingFilter::default().matches(&emb));

        // Model filter
        assert!(EmbeddingFilter::default().model("gemma").matches(&emb));
        assert!(!EmbeddingFilter::default().model("qwen3").matches(&emb));

        // Dim filter
        assert!(EmbeddingFilter::default().dim(768).matches(&emb));
        assert!(!EmbeddingFilter::default().dim(1024).matches(&emb));

        // Distance filter
        assert!(EmbeddingFilter::default().distance(Distance::Cosine).matches(&emb));
        assert!(!EmbeddingFilter::default().distance(Distance::L2).matches(&emb));

        // Combined filter
        let filter = EmbeddingFilter::default()
            .model("gemma")
            .dim(768)
            .distance(Distance::Cosine);
        assert!(filter.matches(&emb));

        let filter = EmbeddingFilter::default()
            .model("gemma")
            .dim(1024); // Wrong dim
        assert!(!filter.matches(&emb));
    }

    #[test]
    fn test_registry_new_without_storage() {
        let registry = EmbeddingRegistry::new_without_storage();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }
}
