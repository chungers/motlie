//! Vector mutation types.
//!
//! This module contains mutation type definitions for vector storage operations.
//! Following the pattern from `graph::mutation`, mutations are grouped in an enum
//! for type-safe dispatch.
//!
//! Mutations implement `MutationCodec` to marshal themselves to CF key-value pairs.
//! This keeps marshaling logic with the mutation definitions rather than in schema.

use crate::rocksdb::MutationCodec;

use super::distance::Distance;
use super::schema::{
    EmbeddingCode, EmbeddingSpec, EmbeddingSpecCfKey, EmbeddingSpecCfValue, EmbeddingSpecs,
};

// ============================================================================
// Mutation Enum
// ============================================================================

/// Mutation enum for vector storage operations.
///
/// All vector mutations are variants of this enum, enabling type-safe
/// dispatch in mutation consumers.
#[derive(Debug, Clone)]
pub enum Mutation {
    /// Register a new embedding space
    AddEmbeddingSpec(AddEmbeddingSpec),
    // Future variants:
    // AddVector(AddVector),
    // UpdateEdges(UpdateEdges),
    // DeleteVector(DeleteVector),
}

// ============================================================================
// AddEmbeddingSpec
// ============================================================================

/// Mutation for registering an embedding space.
///
/// Used by `EmbeddingRegistry::register()` to persist new embedding specs
/// to the `EmbeddingSpecs` column family.
///
/// # Example
///
/// ```rust,ignore
/// let add_op = AddEmbeddingSpec {
///     code: allocated_code,
///     model: "gemma".to_string(),
///     dim: 768,
///     distance: Distance::Cosine,
/// };
/// let (key_bytes, value_bytes) = EmbeddingSpecs::create_bytes(&add_op)?;
/// ```
#[derive(Debug, Clone)]
pub struct AddEmbeddingSpec {
    /// Allocated embedding code (primary key)
    pub code: EmbeddingCode,
    /// Model name (e.g., "gemma", "qwen3", "ada-002")
    pub model: String,
    /// Vector dimensionality (e.g., 128, 768, 1536)
    pub dim: u32,
    /// Distance metric for similarity computation
    pub distance: Distance,
}

impl From<AddEmbeddingSpec> for Mutation {
    fn from(op: AddEmbeddingSpec) -> Self {
        Mutation::AddEmbeddingSpec(op)
    }
}

// ============================================================================
// MutationCodec Implementations
// ============================================================================

impl MutationCodec for AddEmbeddingSpec {
    type Cf = EmbeddingSpecs;

    fn to_record(&self) -> (EmbeddingSpecCfKey, EmbeddingSpecCfValue) {
        let key = EmbeddingSpecCfKey(self.code);
        let value = EmbeddingSpecCfValue(EmbeddingSpec {
            model: self.model.clone(),
            dim: self.dim,
            distance: self.distance,
        });
        (key, value)
    }
}
