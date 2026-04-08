//! Bundle-level embedding metadata contracts.

use crate::{ContentKind, ModelBundle};

/// Preferred vector distance semantics for an embedding bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingDistance {
    Cosine,
    Dot,
    SquaredL2,
}

/// Whether the embedding vectors are normalized before they are returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingNormalization {
    L2,
    None,
}

/// Bundle-level metadata that describes how embedding vectors should be interpreted downstream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingSpec {
    pub dimensions: Option<usize>,
    pub distance: EmbeddingDistance,
    pub normalization: EmbeddingNormalization,
    pub input: ContentKind,
    pub output: ContentKind,
    pub summary: &'static str,
}

/// Bundle-level descriptive contract for curated embedding bundles.
pub trait Embedding: ModelBundle {
    fn embedding_spec(&self) -> &EmbeddingSpec;
}
