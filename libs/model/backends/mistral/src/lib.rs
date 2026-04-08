//! Generic Mistral backend implementations for `motlie-model`.
//!
//! This first vertical slice exercises the contract shape with an embedding-only
//! bundle. The actual `mistral.rs` runtime wiring should replace the placeholder
//! embedder implementation in a follow-up step.

use async_trait::async_trait;
use motlie_model::{
    BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind, CompletionModel,
    EmbeddingModel, EmbeddingRequest, EmbeddingResponse, LoadedBundleDescriptor, ModelBundle,
    ModelError, StartOptions, ChatModel,
};

/// Static bundle specification for a curated Mistral-backed embedding stack.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_id: &'static str,
    pub capabilities: Capabilities,
}

impl MistralEmbeddingSpec {
    pub fn embeddinggemma_300m() -> Self {
        Self {
            id: BundleId::new("embeddinggemma_300m"),
            display_name: "EmbeddingGemma 300M",
            model_id: "google/embeddinggemma-300m",
            capabilities: Capabilities::embeddings_only(),
        }
    }
}

/// Curated bundle implementation backed by the generic Mistral embedding path.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingBundle {
    metadata: BundleMetadata,
    model_id: &'static str,
}

impl MistralEmbeddingBundle {
    pub fn new(spec: MistralEmbeddingSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id,
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities,
            },
            model_id: spec.model_id,
        }
    }
}

#[async_trait]
impl ModelBundle for MistralEmbeddingBundle {
    fn id(&self) -> &BundleId {
        &self.metadata.id
    }

    fn metadata(&self) -> &BundleMetadata {
        &self.metadata
    }

    fn capabilities(&self) -> &Capabilities {
        &self.metadata.capabilities
    }

    async fn start(
        &self,
        _options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
        Ok(Box::new(MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
            },
            embedder: PlaceholderMistralEmbedder {
                model_id: self.model_id,
            },
        }))
    }
}

struct MistralEmbeddingHandle {
    descriptor: LoadedBundleDescriptor,
    embedder: PlaceholderMistralEmbedder,
}

#[async_trait]
impl BundleHandle for MistralEmbeddingHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn chat(&self) -> Result<&dyn ChatModel, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
    }

    fn completion(&self) -> Result<&dyn CompletionModel, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Completion))
    }

    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError> {
        Ok(&self.embedder)
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

struct PlaceholderMistralEmbedder {
    model_id: &'static str,
}

#[async_trait]
impl EmbeddingModel for PlaceholderMistralEmbedder {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        let vectors = request
            .inputs
            .into_iter()
            .map(|input| placeholder_embedding(self.model_id, &input))
            .collect();
        Ok(EmbeddingResponse { vectors })
    }
}

fn placeholder_embedding(model_id: &str, input: &str) -> Vec<f32> {
    // Placeholder deterministic embedding for the first contract vertical slice.
    // This will be replaced by real `mistral.rs` inference.
    let bytes = input.as_bytes();
    let checksum = bytes.iter().fold(0u32, |acc, byte| acc + u32::from(*byte));
    vec![
        model_id.len() as f32,
        input.len() as f32,
        bytes.first().copied().unwrap_or_default() as f32,
        checksum as f32,
    ]
}
