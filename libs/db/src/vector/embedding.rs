//! Embedding type and Embedder trait definitions.
//!
//! Design rationale (ARCH-17, ARCH-18):
//! - `Embedding` is a rich struct with model, dim, distance, and optional embedder
//! - `Embedder` trait enables document-to-vector computation
//! - Direct field access without registry lookup

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use anyhow::Result;

use super::distance::Distance;
use super::error;

// ============================================================================
// Embedder Trait
// ============================================================================

/// Trait for computing embeddings from documents.
///
/// Implementations can wrap external services (Ollama, OpenAI) or local models.
///
/// # Example
///
/// ```ignore
/// struct OllamaEmbedder { client: OllamaClient }
///
/// impl Embedder for OllamaEmbedder {
///     fn embed(&self, document: &str) -> Result<Vec<f32>> {
///         self.client.embeddings("gemma", document)
///     }
///     fn dim(&self) -> u32 { 768 }
///     fn model(&self) -> &str { "gemma" }
/// }
/// ```
pub trait Embedder: Send + Sync {
    /// Embed a single document into a vector.
    fn embed(&self, document: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple documents for efficiency.
    ///
    /// Default implementation calls `embed` sequentially.
    fn embed_batch(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>> {
        documents.iter().map(|d| self.embed(d)).collect()
    }

    /// Output dimensionality of this embedder.
    fn dim(&self) -> u32;

    /// Model identifier (e.g., "gemma", "qwen3", "openai-ada-002").
    fn model(&self) -> &str;
}

// ============================================================================
// Embedding Struct
// ============================================================================

/// Complete embedding space specification with optional compute behavior.
///
/// Design rationale:
/// - `code`: u64 for fast key serialization (ARCH-14)
/// - `model`, `dim`, `distance`: Direct access without registry lookup (ARCH-17)
/// - `embedder`: Optional compute capability (ARCH-18)
///
/// # Example
///
/// ```ignore
/// let embedding = registry.register(
///     EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
/// )?;
///
/// // Direct field access
/// assert_eq!(embedding.model(), "gemma");
/// assert_eq!(embedding.dim(), 768);
/// assert_eq!(embedding.distance(), Distance::Cosine);
///
/// // If embedder is attached
/// let vec = embedding.embed("Hello world")?;
/// ```
#[derive(Clone)]
pub struct Embedding {
    /// Unique namespace code for storage keys (allocated by registry)
    code: u64,
    /// Model identifier (e.g., "gemma", "qwen3", "openai-ada-002")
    model: Arc<str>,
    /// Vector dimensionality
    dim: u32,
    /// Distance metric for similarity computation
    distance: Distance,
    /// Storage type for vector elements (F32 or F16)
    storage_type: super::schema::VectorElementType,
    /// Optional embedder for computing vectors from documents
    embedder: Option<Arc<dyn Embedder>>,
}

impl Embedding {
    /// Create a new Embedding (called by EmbeddingRegistry).
    pub(crate) fn new(
        code: u64,
        model: impl Into<Arc<str>>,
        dim: u32,
        distance: Distance,
        storage_type: super::schema::VectorElementType,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Self {
        Self {
            code,
            model: model.into(),
            dim,
            distance,
            storage_type,
            embedder,
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────

    /// Namespace code for storage keys.
    #[inline]
    pub fn code(&self) -> u64 {
        self.code
    }

    /// Code as big-endian bytes for key construction.
    #[inline]
    pub fn code_bytes(&self) -> [u8; 8] {
        self.code.to_be_bytes()
    }

    /// Model identifier.
    #[inline]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Vector dimensionality.
    #[inline]
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Distance metric.
    #[inline]
    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// Storage type for vector elements (F32 or F16).
    #[inline]
    pub fn storage_type(&self) -> super::schema::VectorElementType {
        self.storage_type
    }

    /// Check if this embedding has compute capability.
    #[inline]
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    // ─────────────────────────────────────────────────────────────
    // Behavior (delegated to Embedder trait)
    // ─────────────────────────────────────────────────────────────

    /// Compute embedding for a document (requires embedder).
    pub fn embed(&self, document: &str) -> Result<Vec<f32>> {
        self.embedder
            .as_ref()
            .ok_or_else(|| error::no_embedder(&self.model))?
            .embed(document)
    }

    /// Batch embed documents (requires embedder).
    pub fn embed_batch(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder
            .as_ref()
            .ok_or_else(|| error::no_embedder(&self.model))?
            .embed_batch(documents)
    }

    /// Compute distance between two vectors using this space's metric.
    #[inline]
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance.compute(a, b)
    }

    // ─────────────────────────────────────────────────────────────
    // Validation
    // ─────────────────────────────────────────────────────────────

    /// Validate vector dimension matches this embedding space.
    #[inline]
    pub fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        error::check_dimension(self.dim, vector)
    }

    // ─────────────────────────────────────────────────────────────
    // Internal
    // ─────────────────────────────────────────────────────────────

    /// Attach an embedder (used by registry).
    pub(crate) fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }
}

// Equality and Hash based on code only (for HashMap keys)
impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl Eq for Embedding {}

impl Hash for Embedding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.code.hash(state);
    }
}

impl fmt::Debug for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedding")
            .field("code", &self.code)
            .field("model", &self.model)
            .field("dim", &self.dim)
            .field("distance", &self.distance)
            .field("has_embedder", &self.embedder.is_some())
            .finish()
    }
}

impl fmt::Display for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.model,
            self.dim,
            self.distance.as_str()
        )
    }
}

// ============================================================================
// EmbeddingBuilder
// ============================================================================

/// Builder for registering embedding spaces.
///
/// # Example
///
/// ```ignore
/// let embedding = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
///     .with_hnsw_m(32)
///     .with_hnsw_ef_construction(400)
///     .with_rabitq_bits(2)
///     .with_embedder(Arc::new(my_embedder))
///     .register(&registry)?;
/// ```
pub struct EmbeddingBuilder {
    pub(crate) model: String,
    pub(crate) dim: u32,
    pub(crate) distance: Distance,
    pub(crate) embedder: Option<Arc<dyn Embedder>>,

    // Build Parameters (Phase 5.7c)
    /// HNSW M parameter: number of bidirectional links per node.
    /// Default: 16
    pub(crate) hnsw_m: u16,
    /// HNSW ef_construction: search beam width during index build.
    /// Default: 200
    pub(crate) hnsw_ef_construction: u16,
    /// RaBitQ bits per dimension (1, 2, or 4).
    /// Default: 1
    pub(crate) rabitq_bits: u8,
    /// RaBitQ rotation matrix seed.
    /// Default: 42
    pub(crate) rabitq_seed: u64,
}

impl EmbeddingBuilder {
    /// Create a new builder for an embedding space with default build params.
    ///
    /// Default build params:
    /// - hnsw_m: 16
    /// - hnsw_ef_construction: 200
    /// - rabitq_bits: 1
    /// - rabitq_seed: 42
    pub fn new(model: impl Into<String>, dim: u32, distance: Distance) -> Self {
        Self {
            model: model.into(),
            dim,
            distance,
            embedder: None,
            // Default build params matching schema.rs defaults
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            rabitq_bits: 1,
            rabitq_seed: 42,
        }
    }

    /// Attach an embedder for compute capability.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    // ─────────────────────────────────────────────────────────────
    // Build Parameters (Phase 5.7c)
    // ─────────────────────────────────────────────────────────────

    /// Set HNSW M parameter (number of bidirectional links per node).
    ///
    /// Higher M = better recall, more memory, slower insert.
    /// Recommended: 8-64, Default: 16
    ///
    /// **Constraint:** M must be >= 2 (validated at registration).
    pub fn with_hnsw_m(mut self, m: u16) -> Self {
        self.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction (search beam width during index build).
    ///
    /// Higher ef_construction = better graph quality, slower build.
    /// Recommended: >= 2*M, Default: 200
    pub fn with_hnsw_ef_construction(mut self, ef_construction: u16) -> Self {
        self.hnsw_ef_construction = ef_construction;
        self
    }

    /// Set RaBitQ bits per dimension (1, 2, or 4).
    ///
    /// - 1-bit: 32x compression, lowest recall without rerank
    /// - 2-bit: 16x compression, medium recall
    /// - 4-bit: 8x compression, highest recall
    ///
    /// Default: 1
    ///
    /// **Constraint:** Must be 1, 2, or 4 (validated at registration).
    pub fn with_rabitq_bits(mut self, bits: u8) -> Self {
        self.rabitq_bits = bits;
        self
    }

    /// Set RaBitQ rotation matrix seed for deterministic encoding.
    ///
    /// Same seed = same rotation matrix = reproducible binary codes.
    /// Default: 42
    pub fn with_rabitq_seed(mut self, seed: u64) -> Self {
        self.rabitq_seed = seed;
        self
    }

    // ─────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────

    /// Model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Distance metric.
    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// HNSW M parameter.
    pub fn hnsw_m(&self) -> u16 {
        self.hnsw_m
    }

    /// HNSW ef_construction.
    pub fn hnsw_ef_construction(&self) -> u16 {
        self.hnsw_ef_construction
    }

    /// RaBitQ bits per dimension.
    pub fn rabitq_bits(&self) -> u8 {
        self.rabitq_bits
    }

    /// RaBitQ rotation seed.
    pub fn rabitq_seed(&self) -> u64 {
        self.rabitq_seed
    }

    // ─────────────────────────────────────────────────────────────
    // Validation (called at registration)
    // ─────────────────────────────────────────────────────────────

    /// Validate build parameters before registration.
    ///
    /// Called by `EmbeddingRegistry::register()` to ensure parameters are valid
    /// before persisting to storage. Returns an error if validation fails.
    ///
    /// # Validation Rules
    ///
    /// - `hnsw_m >= 2`: Required for valid `m_l = 1/ln(m)` computation
    /// - `rabitq_bits ∈ {1, 2, 4}`: Only supported quantization levels
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
    ///     .with_hnsw_m(1); // Invalid!
    ///
    /// assert!(builder.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<()> {
        // HNSW M must be >= 2 for valid m_l computation
        // m_l = 1.0 / ln(m), so m=1 gives ln(1)=0 → division by zero
        if self.hnsw_m < 2 {
            return Err(anyhow::anyhow!(
                "hnsw_m must be >= 2 (got {}). Values <= 1 cause invalid layer assignment (m_l = 1/ln(m)).",
                self.hnsw_m
            ));
        }

        // RaBitQ only supports 1, 2, or 4 bits per dimension
        if self.rabitq_bits != 1 && self.rabitq_bits != 2 && self.rabitq_bits != 4 {
            return Err(anyhow::anyhow!(
                "rabitq_bits must be 1, 2, or 4 (got {})",
                self.rabitq_bits
            ));
        }

        Ok(())
    }
}

impl fmt::Debug for EmbeddingBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingBuilder")
            .field("model", &self.model)
            .field("dim", &self.dim)
            .field("distance", &self.distance)
            .field("has_embedder", &self.embedder.is_some())
            .field("hnsw_m", &self.hnsw_m)
            .field("hnsw_ef_construction", &self.hnsw_ef_construction)
            .field("rabitq_bits", &self.rabitq_bits)
            .field("rabitq_seed", &self.rabitq_seed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schema::VectorElementType;

    #[test]
    fn test_embedding_accessors() {
        let emb = Embedding::new(42, "gemma", 768, Distance::Cosine, VectorElementType::default(), None);

        assert_eq!(emb.code(), 42);
        assert_eq!(emb.model(), "gemma");
        assert_eq!(emb.dim(), 768);
        assert_eq!(emb.distance(), Distance::Cosine);
        assert!(!emb.has_embedder());
    }

    #[test]
    fn test_embedding_code_bytes() {
        let emb = Embedding::new(0x0102030405060708, "test", 128, Distance::L2, VectorElementType::default(), None);
        let bytes = emb.code_bytes();
        assert_eq!(bytes, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    #[test]
    fn test_embedding_equality() {
        let emb1 = Embedding::new(42, "gemma", 768, Distance::Cosine, VectorElementType::default(), None);
        let emb2 = Embedding::new(42, "different", 1024, Distance::L2, VectorElementType::default(), None);
        let emb3 = Embedding::new(43, "gemma", 768, Distance::Cosine, VectorElementType::default(), None);

        // Same code = equal (regardless of other fields)
        assert_eq!(emb1, emb2);
        // Different code = not equal
        assert_ne!(emb1, emb3);
    }

    #[test]
    fn test_embedding_display() {
        let emb = Embedding::new(1, "gemma", 768, Distance::Cosine, VectorElementType::default(), None);
        assert_eq!(format!("{}", emb), "gemma:768:cosine");
    }

    #[test]
    fn test_embedding_validate_vector() {
        let emb = Embedding::new(1, "test", 4, Distance::Cosine, VectorElementType::default(), None);

        // Correct dimension
        assert!(emb.validate_vector(&[1.0, 2.0, 3.0, 4.0]).is_ok());

        // Wrong dimension
        assert!(emb.validate_vector(&[1.0, 2.0, 3.0]).is_err());
        assert!(emb.validate_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]).is_err());
    }

    #[test]
    fn test_embedding_builder() {
        let builder = EmbeddingBuilder::new("qwen3", 1024, Distance::DotProduct);

        assert_eq!(builder.model(), "qwen3");
        assert_eq!(builder.dim(), 1024);
        assert_eq!(builder.distance(), Distance::DotProduct);

        // Default build params
        assert_eq!(builder.hnsw_m(), 16);
        assert_eq!(builder.hnsw_ef_construction(), 200);
        assert_eq!(builder.rabitq_bits(), 1);
        assert_eq!(builder.rabitq_seed(), 42);
    }

    #[test]
    fn test_embedding_builder_with_build_params() {
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
            .with_hnsw_m(32)
            .with_hnsw_ef_construction(400)
            .with_rabitq_bits(2)
            .with_rabitq_seed(12345);

        assert_eq!(builder.hnsw_m(), 32);
        assert_eq!(builder.hnsw_ef_construction(), 400);
        assert_eq!(builder.rabitq_bits(), 2);
        assert_eq!(builder.rabitq_seed(), 12345);
    }

    #[test]
    fn test_embedding_builder_validate_invalid_rabitq_bits() {
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
            .with_rabitq_bits(3); // Invalid - only 1, 2, 4 allowed

        let result = builder.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rabitq_bits must be 1, 2, or 4"));
    }

    #[test]
    fn test_embedding_builder_validate_invalid_hnsw_m() {
        // hnsw_m = 0 is invalid
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
            .with_hnsw_m(0);
        let result = builder.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hnsw_m must be >= 2"));

        // hnsw_m = 1 is also invalid (ln(1) = 0)
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
            .with_hnsw_m(1);
        let result = builder.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hnsw_m must be >= 2"));

        // hnsw_m = 2 is valid (minimum)
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
            .with_hnsw_m(2);
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn test_embedding_builder_validate_success() {
        // Default builder should pass validation
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine);
        assert!(builder.validate().is_ok());

        // Custom valid params should also pass
        let builder = EmbeddingBuilder::new("test", 128, Distance::Cosine)
            .with_hnsw_m(32)
            .with_rabitq_bits(4);
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn test_embedding_no_embedder_error() {
        let emb = Embedding::new(1, "test", 128, Distance::Cosine, VectorElementType::default(), None);
        let result = emb.embed("hello");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No embedder"));
    }
}
