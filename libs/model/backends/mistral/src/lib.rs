//! Generic Mistral backend implementations for `motlie-model`.

mod common;
mod embeddings;
mod multimodal;
mod text;

pub use embeddings::{
    MistralEmbeddingAdapter, MistralEmbeddingArch, MistralEmbeddingBundle, MistralEmbeddingHandle,
    MistralEmbeddingSpec,
};
pub use multimodal::{
    MistralMultimodalAdapter, MistralMultimodalArch, MistralMultimodalBundle,
    MistralMultimodalHandle, MistralMultimodalSpec,
};
pub use text::{
    MistralTextAdapter, MistralTextArch, MistralTextBundle, MistralTextHandle, MistralTextSpec,
};
