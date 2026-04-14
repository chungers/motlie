//! Generic Mistral backend implementations for `motlie-model`.

mod common;
mod embeddings;
mod multimodal;
mod text;

pub use embeddings::{
    MistralEmbeddingAdapter, MistralEmbeddingArch, MistralEmbeddingBundle, MistralEmbeddingSpec,
};
pub use multimodal::{
    MistralMultimodalAdapter, MistralMultimodalArch, MistralMultimodalBundle, MistralMultimodalSpec,
};
pub use text::{MistralTextAdapter, MistralTextArch, MistralTextBundle, MistralTextSpec};
