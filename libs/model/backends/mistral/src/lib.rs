//! Generic Mistral backend implementations for `motlie-model`.

mod common;
mod embeddings;
mod multimodal;
mod text;

pub use embeddings::{MistralEmbeddingArch, MistralEmbeddingBundle, MistralEmbeddingSpec};
pub use multimodal::{MistralMultimodalArch, MistralMultimodalBundle, MistralMultimodalSpec};
pub use text::{MistralTextArch, MistralTextBundle, MistralTextSpec};
