//! Generic Mistral backend implementations for `motlie-model`.

mod embeddings;
mod text;

pub use embeddings::{MistralEmbeddingArch, MistralEmbeddingBundle, MistralEmbeddingSpec};
pub use text::{MistralTextArch, MistralTextBundle, MistralTextSpec};
