use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, BatchTranscriber, Mono};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationSupport, ResolvedCheckpoint,
    RuntimeAcceleratorObservation, StartOptions, TranscriptSegment, TranscriptionParams,
    TranscriptionUpdate, UnsupportedChat, UnsupportedCompletion, UnsupportedEmbeddings,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_ggml_model_path, RuntimeMetricState,
};

const WHISPER_CPP_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Ggml];

/// Whisper expects mono 16 kHz f32 PCM.
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Static bundle specification for a curated whisper.cpp-backed ASR stack.
#[derive(Clone, Debug)]
pub struct WhisperCppTranscriptionSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_filename: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl WhisperCppTranscriptionSpec {
    pub fn whisper_base_en() -> Self {
        Self {
            id: BundleId::new("whisper_base_en"),
            display_name: "Whisper Base.en",
            model_filename: "ggml-base.en.bin",
            capabilities: Capabilities::transcription_batch_only(),
            quantization: QuantizationSupport::none(),
        }
    }
}

/// Backend adapter for `whisper.cpp` transcription over ggml checkpoints.
#[derive(Clone, Debug)]
pub struct WhisperCppTranscriptionAdapter {
    capabilities: Capabilities,
    quantization: QuantizationSupport,
}

impl WhisperCppTranscriptionAdapter {
    pub fn whisper_base_en() -> Self {
        let spec = WhisperCppTranscriptionSpec::whisper_base_en();
        Self {
            capabilities: spec.capabilities,
            quantization: spec.quantization,
        }
    }
}

#[async_trait]
impl BackendAdapter for WhisperCppTranscriptionAdapter {
    type Handle = WhisperCppHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &WHISPER_CPP_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::WhisperCpp
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
        // Reject unsupported quantization requests explicitly.
        self.quantization
            .resolve(options.quantization, &identity.id)?;

        let model_path = resolve_ggml_model_path(checkpoint)?;
        let ctx = load_whisper_model(&model_path)?;

        Ok(new_transcription_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.capabilities.clone(),
            self.quantization.clone(),
            ctx,
        ))
    }
}

/// Generic `ModelBundle` implementation backed by `whisper.cpp`.
#[derive(Clone, Debug)]
pub struct WhisperCppTranscriptionBundle {
    metadata: BundleMetadata,
    model_filename: &'static str,
}

impl WhisperCppTranscriptionBundle {
    pub fn new(spec: WhisperCppTranscriptionSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id,
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities,
                quantization: spec.quantization,
            },
            model_filename: spec.model_filename,
        }
    }
}

#[async_trait]
impl ModelBundle for WhisperCppTranscriptionBundle {
    type Handle = WhisperCppHandle;

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
        self.start_typed(options).await
    }
}

impl WhisperCppTranscriptionBundle {
    pub async fn start_typed(&self, options: StartOptions) -> Result<WhisperCppHandle, ModelError> {
        // Reject unsupported quantization requests explicitly.
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let model_path = if let Some(artifact_policy) = options.artifact_policy {
            configure_artifact_policy(self.model_filename, artifact_policy)?
        } else {
            PathBuf::from(self.model_filename)
        };

        let ctx = load_whisper_model(&model_path)?;

        Ok(new_transcription_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            ctx,
        ))
    }
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

fn load_whisper_model(
    model_path: &std::path::Path,
) -> Result<Arc<whisper_rs::WhisperContext>, ModelError> {
    let params = whisper_rs::WhisperContextParameters::default();
    let ctx = whisper_rs::WhisperContext::new_with_params(
        model_path.to_str().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "whisper model path `{}` contains non-UTF-8 characters",
                model_path.display()
            ))
        })?,
        params,
    )
    .map_err(|err| ModelError::BackendInitialization {
        backend: "whisper-cpp",
        message: format!(
            "failed to load whisper model from `{}`: {err}",
            model_path.display()
        ),
    })?;

    Ok(Arc::new(ctx))
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

pub struct WhisperCppHandle {
    descriptor: LoadedBundleDescriptor,
    ctx: Arc<whisper_rs::WhisperContext>,
    metrics: Arc<Mutex<AsrMetrics>>,
}

impl WhisperCppHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }
}

#[derive(Clone, Debug, Default)]
struct AsrMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for WhisperCppHandle {
    type Chat = UnsupportedChat;
    type Completion = UnsupportedCompletion;
    type Embeddings = UnsupportedEmbeddings;

    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "whisper-cpp-metric-snapshot").clone();
        Some(ModelMetricSnapshot {
            runtime: Some(motlie_model::RuntimeMetrics {
                resident_memory: None,
                peak_resident_memory: None,
                request_count: Some(metrics.runtime.request_count),
                last_latency: metrics
                    .runtime
                    .last_latency_msec
                    .map(motlie_model::Milliseconds),
                max_latency: metrics
                    .runtime
                    .max_latency_msec
                    .map(motlie_model::Milliseconds),
                avg_latency: None,
            }),
            text_generation: None,
            embeddings: None,
        })
    }

    fn accelerator_observation(&self) -> Option<RuntimeAcceleratorObservation> {
        if cfg!(feature = "cuda") {
            Some(RuntimeAcceleratorObservation {
                backend_mode: "whisper_cpp:cuda".to_owned(),
                offload: Some("cuda_execution_provider=on".to_owned()),
                selected_device: Some("0".to_owned()),
            })
        } else {
            Some(RuntimeAcceleratorObservation {
                backend_mode: "whisper_cpp:cpu".to_owned(),
                offload: Some("accelerator_feature=none".to_owned()),
                selected_device: None,
            })
        }
    }

    fn chat(&self) -> Result<&Self::Chat, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
    }

    fn completion(&self) -> Result<&Self::Completion, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }

    fn embeddings(&self) -> Result<&Self::Embeddings, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }

    async fn shutdown(self) -> Result<(), ModelError> {
        Ok(())
    }
}

fn new_transcription_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    ctx: Arc<whisper_rs::WhisperContext>,
) -> WhisperCppHandle {
    let metrics = Arc::new(Mutex::new(AsrMetrics::default()));
    {
        let mut m = lock_metrics(&metrics, "whisper-cpp-start");
        observe_memory(&mut m.runtime);
    }

    WhisperCppHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization: None,
        },
        ctx,
        metrics,
    }
}

fn decode_samples(
    ctx: &whisper_rs::WhisperContext,
    pcm_buffer: &[f32],
    params: &TranscriptionParams,
    metrics: Option<&Arc<Mutex<AsrMetrics>>>,
) -> Result<TranscriptionUpdate, ModelError> {
    let started_at = Instant::now();

    let mut state = ctx
        .create_state()
        .map_err(|err| ModelError::BackendExecution {
            backend: "whisper-cpp",
            operation: "create_state",
            message: err.to_string(),
        })?;

    let mut whisper_params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    if let Some(ref lang) = params.language {
        whisper_params.set_language(Some(lang));
    }
    whisper_params.set_print_special(false);
    whisper_params.set_print_progress(false);
    whisper_params.set_print_realtime(false);
    whisper_params.set_print_timestamps(false);
    whisper_params.set_debug_mode(false);
    whisper_params.set_single_segment(false);

    state
        .full(whisper_params, pcm_buffer)
        .map_err(|err| ModelError::BackendExecution {
            backend: "whisper-cpp",
            operation: "full",
            message: err.to_string(),
        })?;

    let num_segments = state.full_n_segments() as usize;

    let mut segments = Vec::with_capacity(num_segments);
    for i in 0..num_segments as i32 {
        let segment = state
            .get_segment(i)
            .ok_or_else(|| ModelError::BackendExecution {
                backend: "whisper-cpp",
                operation: "get_segment",
                message: format!("segment {i} out of bounds"),
            })?;
        let text = segment
            .to_str()
            .map_err(|err| ModelError::BackendExecution {
                backend: "whisper-cpp",
                operation: "segment.to_str",
                message: err.to_string(),
            })?
            .to_owned();
        let t0 = segment.start_timestamp();
        let t1 = segment.end_timestamp();

        segments.push(TranscriptSegment {
            start_ms: (t0 * 10) as u64,
            end_ms: (t1 * 10) as u64,
            text,
            final_segment: true,
        });
    }

    if let Some(metrics) = metrics {
        let elapsed = started_at.elapsed();
        let mut m = lock_metrics(metrics, "whisper-cpp-decode");
        observe_latency(&mut m.runtime, elapsed);
    }

    Ok(TranscriptionUpdate { segments })
}

impl BatchTranscriber for WhisperCppHandle {
    type Input = AudioBuf<f32, WHISPER_SAMPLE_RATE, Mono>;

    async fn transcribe(
        &self,
        audio: Self::Input,
        params: TranscriptionParams,
    ) -> Result<TranscriptionUpdate, ModelError> {
        let ctx = Arc::clone(&self.ctx);
        let metrics = Arc::clone(&self.metrics);
        let samples = audio.into_samples();

        decode_samples(&ctx, &samples, &params, Some(&metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::{BackendAdapter, BackendKind, QuantizationSupport};

    #[test]
    fn whisper_base_en_spec_has_expected_identity() {
        let spec = WhisperCppTranscriptionSpec::whisper_base_en();

        assert_eq!(spec.id.as_str(), "whisper_base_en");
        assert_eq!(spec.display_name, "Whisper Base.en");
        assert_eq!(spec.model_filename, "ggml-base.en.bin");
        assert!(spec.capabilities.supports(CapabilityKind::Transcription));
        assert!(!spec.capabilities.supports(CapabilityKind::Chat));
    }

    #[test]
    fn adapter_reports_backend_metadata() {
        let adapter = WhisperCppTranscriptionAdapter::whisper_base_en();

        assert_eq!(adapter.supported_formats(), &[CheckpointFormat::Ggml]);
        assert_eq!(adapter.backend_kind(), BackendKind::WhisperCpp);
        assert!(adapter
            .capabilities()
            .supports(CapabilityKind::Transcription));
        assert_eq!(adapter.quantization(), &QuantizationSupport::none());
    }

    #[test]
    fn bundle_metadata_matches_spec() {
        let bundle =
            WhisperCppTranscriptionBundle::new(WhisperCppTranscriptionSpec::whisper_base_en());

        assert_eq!(bundle.id().as_str(), "whisper_base_en");
        assert!(bundle
            .capabilities()
            .supports(CapabilityKind::Transcription));
        assert_eq!(bundle.metadata().quantization, QuantizationSupport::none());
    }
}
