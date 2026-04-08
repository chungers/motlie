use std::path::PathBuf;

use async_trait::async_trait;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};
use motlie_model::{
    BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind, ChatModel,
    CompletionModel, EmbeddingModel, EmbeddingRequest as ModelEmbeddingRequest, EmbeddingResponse,
    LoadedBundleDescriptor, ModelBundle, ModelError, StartOptions,
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

/// Generic `ModelBundle` implementation backed by `mistralrs` embeddings.
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

    async fn start(&self, options: StartOptions) -> Result<Box<dyn BundleHandle>, ModelError> {
        let model = build_embedding_model(self.model_id, options).await?;

        Ok(Box::new(MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
            },
            runtime: Box::new(MistralRuntime { model }),
        }))
    }
}

#[async_trait]
trait EmbeddingRuntime: Send + Sync {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError>;
}

struct MistralRuntime {
    model: mistralrs::Model,
}

#[async_trait]
impl EmbeddingRuntime for MistralRuntime {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        let builder = request
            .inputs
            .into_iter()
            .fold(EmbeddingRequest::builder(), |builder, input| {
                builder.add_prompt(input)
            });

        let vectors = self
            .model
            .generate_embeddings(builder)
            .await
            .map_err(|err| ModelError::Internal(err.to_string()))?;

        Ok(EmbeddingResponse { vectors })
    }
}

struct MistralEmbeddingHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn EmbeddingRuntime>,
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
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }

    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError> {
        Ok(self)
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl EmbeddingModel for MistralEmbeddingHandle {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        self.runtime.embed(request).await
    }
}

async fn build_embedding_model(
    model_id: &str,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        cache_root,
        unpack_root: _,
        max_concurrency,
    } = options;

    let mut builder = EmbeddingModelBuilder::new(model_id.to_owned());

    if let Some(cache_root) = cache_root {
        builder = builder.from_hf_cache_path(resolve_hf_cache_path(cache_root));
    }
    if let Some(max_num_seqs) = max_concurrency {
        builder = builder.with_max_num_seqs(max_num_seqs);
    }

    builder
        .build()
        .await
        .map_err(|err| ModelError::Internal(err.to_string()))
}

fn resolve_hf_cache_path(root: PathBuf) -> PathBuf {
    root
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubRuntime;

    #[async_trait]
    impl EmbeddingRuntime for StubRuntime {
        async fn embed(
            &self,
            request: ModelEmbeddingRequest,
        ) -> Result<EmbeddingResponse, ModelError> {
            Ok(EmbeddingResponse {
                vectors: request
                    .inputs
                    .into_iter()
                    .map(|input| vec![input.len() as f32])
                    .collect(),
            })
        }
    }

    #[test]
    fn embeddinggemma_spec_has_expected_identity() {
        let spec = MistralEmbeddingSpec::embeddinggemma_300m();

        assert_eq!(spec.id.as_str(), "embeddinggemma_300m");
        assert_eq!(spec.display_name, "EmbeddingGemma 300M");
        assert_eq!(spec.model_id, "google/embeddinggemma-300m");
        assert!(spec.capabilities.supports(CapabilityKind::Embeddings));
    }

    #[tokio::test]
    async fn embedding_handle_rejects_unsupported_capabilities() {
        let handle = MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("embeddinggemma_300m"),
                display_name: "EmbeddingGemma 300M".into(),
                capabilities: Capabilities::embeddings_only(),
            },
            runtime: Box::new(StubRuntime),
        };

        assert!(matches!(
            handle.chat(),
            Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
        ));
        assert!(matches!(
            handle.completion(),
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion
            ))
        ));
        assert!(handle.embeddings().is_ok());
        let response = handle
            .embed(ModelEmbeddingRequest {
                inputs: vec!["abc".into(), "abcd".into()],
            })
            .await
            .expect("stub runtime should succeed");
        assert_eq!(response.vectors, vec![vec![3.0], vec![4.0]]);
    }
}
