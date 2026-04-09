use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use mistralrs::core::EmbeddingLoaderType;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest, ModelDType};
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind,
    ChatModel, CompletionModel, EmbeddingModel, EmbeddingRequest as ModelEmbeddingRequest,
    EmbeddingResponse, LoadedBundleDescriptor, ModelBundle, ModelError, ModelMetricSnapshot,
    StartOptions,
};
use crate::common::{
    observe_embedding_request, observe_memory, snapshot_embedding_metrics, EmbeddingMetricState,
    RuntimeMetricState,
};

/// Embedding architecture discriminant that selects the correct `mistralrs` loader path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MistralEmbeddingArch {
    EmbeddingGemma,
}

impl MistralEmbeddingArch {
    fn loader_type(self) -> EmbeddingLoaderType {
        match self {
            Self::EmbeddingGemma => EmbeddingLoaderType::EmbeddingGemma,
        }
    }
}

/// Static bundle specification for a curated Mistral-backed embedding stack.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_id: &'static str,
    pub arch: MistralEmbeddingArch,
    pub capabilities: Capabilities,
}

impl MistralEmbeddingSpec {
    pub fn embeddinggemma_300m() -> Self {
        Self {
            id: BundleId::new("embeddinggemma_300m"),
            display_name: "EmbeddingGemma 300M",
            model_id: "google/embeddinggemma-300m",
            arch: MistralEmbeddingArch::EmbeddingGemma,
            capabilities: Capabilities::embeddings_only(),
        }
    }
}

/// Generic `ModelBundle` implementation backed by `mistralrs` embeddings.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingBundle {
    metadata: BundleMetadata,
    arch: MistralEmbeddingArch,
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
            arch: spec.arch,
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
        let model = build_embedding_model(self.model_id, self.arch, options).await?;
        let metrics = Arc::new(Mutex::new(EmbeddingMetricsState::default()));
        if let Ok(mut metrics) = metrics.lock() {
            observe_memory(&mut metrics.runtime);
        }

        Ok(Box::new(MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
            },
            runtime: Box::new(MistralRuntime {
                model,
                metrics: Arc::clone(&metrics),
            }),
            metrics,
        }))
    }
}

#[async_trait]
trait EmbeddingRuntime: Send + Sync {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError>;
}

struct MistralRuntime {
    model: mistralrs::Model,
    metrics: Arc<Mutex<EmbeddingMetricsState>>,
}

#[async_trait]
impl EmbeddingRuntime for MistralRuntime {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        let input_count = request.inputs.len();
        let builder = request
            .inputs
            .into_iter()
            .fold(EmbeddingRequest::builder(), |builder, input| {
                builder.add_prompt(input)
            });
        let started_at = Instant::now();

        let vectors = self
            .model
            .generate_embeddings(builder)
            .await
            .map_err(|err| ModelError::BackendExecution {
                backend: "mistralrs",
                operation: "generate_embeddings",
                message: err.to_string(),
            })?;
        let elapsed = started_at.elapsed();

        if let Ok(mut metrics) = self.metrics.lock() {
            let EmbeddingMetricsState {
                runtime,
                embeddings,
            } = &mut *metrics;
            observe_embedding_request(runtime, embeddings, elapsed, input_count);
        }

        Ok(EmbeddingResponse { vectors })
    }
}

struct MistralEmbeddingHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn EmbeddingRuntime>,
    metrics: Arc<Mutex<EmbeddingMetricsState>>,
}

#[async_trait]
impl BundleHandle for MistralEmbeddingHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = self.metrics.lock().ok()?.clone();
        Some(snapshot_embedding_metrics(
            &metrics.runtime,
            &metrics.embeddings,
        ))
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

#[derive(Clone, Debug, Default)]
struct EmbeddingMetricsState {
    runtime: RuntimeMetricState,
    embeddings: EmbeddingMetricState,
}

async fn build_embedding_model(
    model_id: &str,
    arch: MistralEmbeddingArch,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        artifact_policy,
        quantization,
        unpack_root,
        max_concurrency,
    } = options;

    if quantization.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "`mistralrs` embedding models do not support quantization; omit `StartOptions.quantization`".into(),
        ));
    }

    if let Some(unpack_root) = unpack_root {
        return Err(ModelError::InvalidConfiguration(format!(
            "`mistralrs` embedding startup does not support `unpack_root` yet (got `{}`)",
            unpack_root.display()
        )));
    }

    let mut model_target = model_id.to_owned();
    let mut hf_cache_root = None;

    if let Some(artifact_policy) = artifact_policy {
        let configured = configure_artifact_policy(model_id, artifact_policy)?;
        model_target = configured.model_target;
        hf_cache_root = configured.hf_cache_root;
    }

    let mut builder = EmbeddingModelBuilder::new(model_target)
        .with_loader_type(arch.loader_type())
        .with_dtype(ModelDType::F32);

    if should_force_cpu() {
        builder = builder.with_force_cpu();
    }
    if let Some(hf_cache_root) = hf_cache_root {
        builder = builder.from_hf_cache_path(hf_cache_root);
    }
    if let Some(max_num_seqs) = max_concurrency {
        builder = builder.with_max_num_seqs(max_num_seqs);
    }

    builder
        .build()
        .await
        .map_err(|err| ModelError::BackendInitialization {
            backend: "mistralrs",
            message: err.to_string(),
        })
}

struct ConfiguredBuilder {
    model_target: String,
    hf_cache_root: Option<PathBuf>,
}

fn configure_artifact_policy(
    model_id: &str,
    policy: ArtifactPolicy,
) -> Result<ConfiguredBuilder, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => Ok(ConfiguredBuilder {
            model_target: model_id.to_owned(),
            hf_cache_root: root,
        }),
        ArtifactPolicy::LocalOnly { root } => Ok(ConfiguredBuilder {
            model_target: root.display().to_string(),
            hf_cache_root: None,
        }),
    }
}

fn should_force_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::StartOptions;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        assert_eq!(spec.arch, MistralEmbeddingArch::EmbeddingGemma);
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
            metrics: Arc::new(Mutex::new(EmbeddingMetricsState::default())),
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

    #[test]
    fn local_only_policy_uses_resolved_local_model_path() {
        let root = unique_temp_dir();
        let configured = configure_artifact_policy(
            "google/embeddinggemma-300m",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect("resolved local model path should be accepted");

        assert_eq!(configured.model_target, root.display().to_string());
        assert_eq!(configured.hf_cache_root, None);
    }

    #[test]
    fn unpack_root_is_rejected_explicitly() {
        let result = tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(build_embedding_model(
                "google/embeddinggemma-300m",
                MistralEmbeddingArch::EmbeddingGemma,
                StartOptions {
                    unpack_root: Some(PathBuf::from("/tmp/motlie-model-unpack")),
                    ..Default::default()
                },
            ));
        let error = result
            .err()
            .expect("unpack_root should be rejected for mistral embeddings");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message)
                if message.contains("unpack_root")
        ));
    }

    #[tokio::test]
    #[ignore = "requires pre-downloaded embeddinggemma artifacts under MOTLIE_EMBEDDINGGEMMA_ROOT"]
    async fn local_only_embeddinggemma_produces_finite_vectors_for_multi_input_requests() {
        let root = std::env::var("MOTLIE_EMBEDDINGGEMMA_ROOT")
            .expect("MOTLIE_EMBEDDINGGEMMA_ROOT must point at the curated HF cache root");
        let bundle = MistralEmbeddingBundle::new(MistralEmbeddingSpec::embeddinggemma_300m());
        let handle = bundle
            .start(StartOptions {
                artifact_policy: Some(ArtifactPolicy::LocalOnly {
                    root: PathBuf::from(root),
                }),
                ..Default::default()
            })
            .await
            .expect("bundle should start from local artifacts");

        let response = handle
            .embeddings()
            .expect("embeddings capability should exist")
            .embed(ModelEmbeddingRequest {
                inputs: vec![
                    "motlie curated model bundle".into(),
                    "motlie regulated local-only inference".into(),
                ],
            })
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.vectors.len(), 2, "expected one vector per input");
        for vector in response.vectors {
            assert!(!vector.is_empty(), "embedding vector should not be empty");
            assert!(
                vector.iter().all(|value| value.is_finite()),
                "embedding vector should not contain NaN or Inf values: {vector:?}"
            );
        }
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-model-mistral-test-{unique}"))
    }
}
