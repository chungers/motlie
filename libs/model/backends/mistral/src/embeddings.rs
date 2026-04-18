use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::common::{
    EmbeddingMetricState, RuntimeMetricState, configure_artifact_policy, lock_metrics,
    map_quantization_bits, observe_embedding_request, observe_memory, should_force_cpu,
    snapshot_embedding_metrics,
};
use async_trait::async_trait;
use mistralrs::core::EmbeddingLoaderType;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest, ModelDType};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    EmbeddingRequest as ModelEmbeddingRequest, EmbeddingResponse, LoadedBundleDescriptor,
    ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, QuantizationBits,
    QuantizationSupport, ResolvedCheckpoint, StartOptions,
};

const MISTRAL_EMBEDDING_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Safetensors];

/// Embedding architecture discriminant that selects the correct `mistralrs` loader path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MistralEmbeddingArch {
    EmbeddingGemma,
    Qwen3Embedding,
}

impl MistralEmbeddingArch {
    fn loader_type(self) -> EmbeddingLoaderType {
        match self {
            Self::EmbeddingGemma => EmbeddingLoaderType::EmbeddingGemma,
            Self::Qwen3Embedding => EmbeddingLoaderType::Qwen3Embedding,
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
    pub quantization: QuantizationSupport,
}

impl MistralEmbeddingSpec {
    pub fn embeddinggemma_300m() -> Self {
        Self {
            id: BundleId::new("embeddinggemma_300m"),
            display_name: "EmbeddingGemma 300M",
            model_id: "google/embeddinggemma-300m",
            arch: MistralEmbeddingArch::EmbeddingGemma,
            capabilities: Capabilities::embeddings_only(),
            quantization: QuantizationSupport::none(),
        }
    }

    pub fn qwen3_embedding_06b() -> Self {
        Self {
            id: BundleId::new("qwen3_embedding_06b"),
            display_name: "Qwen3 Embedding 0.6B",
            model_id: "Qwen/Qwen3-Embedding-0.6B",
            arch: MistralEmbeddingArch::Qwen3Embedding,
            capabilities: Capabilities::embeddings_only(),
            quantization: QuantizationSupport::without_recommended([QuantizationBits::Eight]),
        }
    }
}

/// Backend adapter for `mistralrs` embedding models over safetensors checkpoints.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingAdapter {
    arch: MistralEmbeddingArch,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
}

impl MistralEmbeddingAdapter {
    pub fn embedding_gemma() -> Self {
        let spec = MistralEmbeddingSpec::embeddinggemma_300m();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
        }
    }

    pub fn qwen3_embedding() -> Self {
        let spec = MistralEmbeddingSpec::qwen3_embedding_06b();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
        }
    }
}

#[async_trait]
impl BackendAdapter for MistralEmbeddingAdapter {
    type Handle = MistralEmbeddingHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &MISTRAL_EMBEDDING_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::MistralRs
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn quantization(&self) -> &QuantizationSupport {
        &self.quantization
    }

    async fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError> {
        let resolved_quantization = self
            .quantization
            .resolve(options.quantization, &identity.id)?;
        let (model_id, options) = crate::common::resolve_local_checkpoint(
            checkpoint,
            CheckpointFormat::Safetensors,
            options,
        )?;
        let model =
            build_embedding_model(model_id, self.arch, resolved_quantization, options).await?;

        Ok(new_embedding_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.capabilities.clone(),
            self.quantization.clone(),
            resolved_quantization,
            model,
        ))
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
                quantization: spec.quantization,
            },
            arch: spec.arch,
            model_id: spec.model_id,
        }
    }
}

#[async_trait]
impl ModelBundle for MistralEmbeddingBundle {
    type Handle = MistralEmbeddingHandle;

    fn id(&self) -> &BundleId {
        &self.metadata.id
    }

    fn metadata(&self) -> &BundleMetadata {
        &self.metadata
    }

    fn capabilities(&self) -> &Capabilities {
        &self.metadata.capabilities
    }

    async fn start(&self, options: StartOptions) -> Result<Self::Handle, ModelError> {
        let resolved_quantization = self
            .metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;
        let model =
            build_embedding_model(self.model_id, self.arch, resolved_quantization, options).await?;

        Ok(new_embedding_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            resolved_quantization,
            model,
        ))
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

        {
            let mut metrics = lock_metrics(&self.metrics, "mistral-embeddings-embed");
            let EmbeddingMetricsState {
                runtime,
                embeddings,
            } = &mut *metrics;
            observe_embedding_request(runtime, embeddings, elapsed, input_count);
        }

        Ok(EmbeddingResponse { vectors })
    }
}

pub struct MistralEmbeddingHandle {
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
        let metrics = lock_metrics(&self.metrics, "mistral-embeddings-metric-snapshot").clone();
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

    async fn shutdown(self) -> Result<(), ModelError> {
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

fn new_embedding_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    resolved_quantization: Option<QuantizationBits>,
    model: mistralrs::Model,
) -> MistralEmbeddingHandle {
    let metrics = Arc::new(Mutex::new(EmbeddingMetricsState::default()));
    {
        let mut metrics = lock_metrics(&metrics, "mistral-embeddings-start");
        observe_memory(&mut metrics.runtime);
    }

    MistralEmbeddingHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization,
        },
        runtime: Box::new(MistralRuntime {
            model,
            metrics: Arc::clone(&metrics),
        }),
        metrics,
    }
}

async fn build_embedding_model(
    model_id: &str,
    arch: MistralEmbeddingArch,
    resolved_quantization: Option<QuantizationBits>,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        artifact_policy,
        quantization: _, // already resolved by caller
        unpack_root,
        max_concurrency,
    } = options;

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

    if let Some(bits) = resolved_quantization {
        builder = builder.with_auto_isq(map_quantization_bits(bits));
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::{ArtifactPolicy, BackendAdapter, BackendKind, StartOptions};
    use std::path::PathBuf;
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

    #[test]
    fn qwen3_embedding_spec_has_expected_identity_and_quantization_support() {
        let spec = MistralEmbeddingSpec::qwen3_embedding_06b();

        assert_eq!(spec.id.as_str(), "qwen3_embedding_06b");
        assert_eq!(spec.display_name, "Qwen3 Embedding 0.6B");
        assert_eq!(spec.model_id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(spec.arch, MistralEmbeddingArch::Qwen3Embedding);
        assert!(spec.capabilities.supports(CapabilityKind::Embeddings));
        assert_eq!(spec.quantization.recommended(), None);
        assert!(spec.quantization.supports(QuantizationBits::Eight));
        assert!(!spec.quantization.supports(QuantizationBits::Four));
    }

    #[test]
    fn qwen3_embedding_adapter_reports_backend_metadata() {
        let adapter = MistralEmbeddingAdapter::qwen3_embedding();

        assert_eq!(
            adapter.supported_formats(),
            &[CheckpointFormat::Safetensors]
        );
        assert_eq!(adapter.backend_kind(), BackendKind::MistralRs);
        assert_eq!(adapter.capabilities(), &Capabilities::embeddings_only());
        assert_eq!(adapter.quantization().recommended(), None);
        assert!(adapter.quantization().supports(QuantizationBits::Eight));
    }

    #[tokio::test]
    async fn embedding_handle_rejects_unsupported_capabilities() {
        let handle = MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("embeddinggemma_300m"),
                display_name: "EmbeddingGemma 300M".into(),
                capabilities: Capabilities::embeddings_only(),
                quantization: QuantizationSupport::none(),
                resolved_quantization: None,
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
                None, // resolved quantization not relevant for unpack_root test
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
