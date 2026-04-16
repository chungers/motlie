use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot,
    SpeechModel, StartOptions, TranscriptSegment, TranscriptionModel, TranscriptionParams,
    TranscriptionStream, TranscriptionUpdate,
};
use transcribe_rs::onnx::moonshine::StreamingModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{set_ort_accelerator, OrtAccelerator, SpeechModel as _, TranscribeOptions};

use crate::common::{
    configure_artifact_policy, decode_pcm_to_f32, downmix_to_mono, lock_metrics, observe_latency,
    resample_mono, resolve_onnx_artifacts, MoonshineArtifactPaths, MoonshineArtifactSpec,
    RuntimeMetricState, StagedModelDir,
};

const MOONSHINE_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];

#[derive(Clone, Debug)]
pub struct MoonshineStreamingSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub frontend_filename: &'static str,
    pub encoder_filename: &'static str,
    pub adapter_filename: &'static str,
    pub cross_kv_filename: &'static str,
    pub decoder_kv_filename: &'static str,
    pub streaming_config_filename: &'static str,
    pub tokenizer_json_filename: &'static str,
    pub capabilities: Capabilities,
    pub quantization: motlie_model::QuantizationSupport,
}

impl MoonshineStreamingSpec {
    pub fn small_en() -> Self {
        Self {
            id: BundleId::new("moonshine_streaming_en"),
            display_name: "Moonshine Streaming EN",
            frontend_filename: "onnx/small/frontend.ort",
            encoder_filename: "onnx/small/encoder.ort",
            adapter_filename: "onnx/small/adapter.ort",
            cross_kv_filename: "onnx/small/cross_kv.ort",
            decoder_kv_filename: "onnx/small/decoder_kv.ort",
            streaming_config_filename: "onnx/small/streaming_config.json",
            tokenizer_json_filename: "onnx/small/tokenizer.json",
            capabilities: Capabilities::transcription_stream_only(),
            quantization: motlie_model::QuantizationSupport::none(),
        }
    }

    fn artifact_spec(&self) -> MoonshineArtifactSpec<'_> {
        MoonshineArtifactSpec {
            frontend: self.frontend_filename,
            encoder: self.encoder_filename,
            adapter: self.adapter_filename,
            cross_kv: self.cross_kv_filename,
            decoder_kv: self.decoder_kv_filename,
            streaming_config: self.streaming_config_filename,
            tokenizer_json: self.tokenizer_json_filename,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoonshineStreamingAdapter {
    spec: MoonshineStreamingSpec,
}

impl MoonshineStreamingAdapter {
    pub fn small_en() -> Self {
        Self {
            spec: MoonshineStreamingSpec::small_en(),
        }
    }
}

#[async_trait]
impl BackendAdapter for MoonshineStreamingAdapter {
    fn supported_formats(&self) -> &[CheckpointFormat] {
        &MOONSHINE_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Ort
    }

    fn capabilities(&self) -> &Capabilities {
        &self.spec.capabilities
    }

    fn quantization(&self) -> &motlie_model::QuantizationSupport {
        &self.spec.quantization
    }

    async fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &motlie_model::ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
        self.quantization()
            .resolve(options.quantization, &identity.id)?;

        let artifacts = resolve_onnx_artifacts(checkpoint, self.spec.artifact_spec())?;
        let runtime = Arc::new(load_runtime(&artifacts)?);

        Ok(new_transcription_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.spec.capabilities.clone(),
            self.spec.quantization.clone(),
            runtime,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct MoonshineStreamingBundle {
    metadata: BundleMetadata,
    spec: MoonshineStreamingSpec,
}

impl MoonshineStreamingBundle {
    pub fn new(spec: MoonshineStreamingSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id.clone(),
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities.clone(),
                quantization: motlie_model::QuantizationSupport::none(),
            },
            spec,
        }
    }
}

#[async_trait]
impl ModelBundle for MoonshineStreamingBundle {
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
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let artifacts = if let Some(policy) = options.artifact_policy {
            configure_artifact_policy(self.spec.artifact_spec(), policy)?
        } else {
            configure_artifact_policy(
                self.spec.artifact_spec(),
                motlie_model::ArtifactPolicy::LocalOnly {
                    root: PathBuf::from("."),
                },
            )?
        };

        let runtime = Arc::new(load_runtime(&artifacts)?);
        Ok(new_transcription_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            runtime,
        ))
    }
}

struct MoonshineHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<MoonshineRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct AsrMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for MoonshineHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "moonshine-metric-snapshot").clone();
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

    fn chat(&self) -> Result<&dyn ChatModel, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
    }

    fn completion(&self) -> Result<&dyn CompletionModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }

    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }

    fn speech(&self) -> Result<&dyn SpeechModel, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Speech))
    }

    fn transcription(&self) -> Result<&dyn TranscriptionModel, ModelError> {
        Ok(self)
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl TranscriptionModel for MoonshineHandle {
    async fn open_stream(
        &self,
        spec: AudioSpec,
        params: TranscriptionParams,
    ) -> Result<Box<dyn TranscriptionStream>, ModelError> {
        if spec.channels == 0 {
            return Err(ModelError::InvalidConfiguration(
                "audio stream must declare at least one channel".into(),
            ));
        }
        if spec.sample_rate_hz == 0 {
            return Err(ModelError::InvalidConfiguration(
                "audio stream must declare a non-zero sample rate".into(),
            ));
        }

        Ok(Box::new(MoonshineStream {
            spec,
            params,
            runtime: Arc::clone(&self.runtime),
            metrics: Arc::clone(&self.metrics),
            raw_pcm: Vec::new(),
            next_sequence: 0,
            saw_end_of_stream: false,
        }))
    }
}

fn new_transcription_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: motlie_model::QuantizationSupport,
    runtime: Arc<MoonshineRuntime>,
) -> Box<dyn BundleHandle> {
    Box::new(MoonshineHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization: None,
        },
        runtime,
        metrics: Arc::new(Mutex::new(AsrMetrics::default())),
    })
}

struct MoonshineRuntime {
    model: Mutex<StreamingModel>,
    _staged_root: StagedModelDir,
}

impl MoonshineRuntime {
    fn transcribe(
        &self,
        raw_pcm: &[u8],
        spec: &AudioSpec,
        params: &TranscriptionParams,
    ) -> Result<TranscriptionUpdate, ModelError> {
        let decoded = decode_pcm_to_f32(raw_pcm, spec.encoding)?;
        let mono = downmix_to_mono(&decoded, spec.channels);
        let resampled = resample_mono(&mono, spec.sample_rate_hz);
        if resampled.is_empty() {
            return Ok(TranscriptionUpdate::default());
        }

        let duration_ms = ((resampled.len() as f64 / 16_000.0) * 1000.0).round() as u64;
        let mut model = self
            .model
            .lock()
            .map_err(|_| ModelError::Internal("moonshine runtime mutex poisoned".into()))?;
        let result = model
            .transcribe(
                &resampled,
                &TranscribeOptions {
                    language: params.language.clone(),
                    ..Default::default()
                },
            )
            .map_err(transcribe_error)?;

        let text = result.text.trim().to_owned();
        if text.is_empty() {
            return Ok(TranscriptionUpdate::default());
        }

        Ok(TranscriptionUpdate {
            segments: vec![TranscriptSegment {
                start_ms: 0,
                end_ms: duration_ms,
                text,
                final_segment: true,
            }],
        })
    }
}

fn load_runtime(artifacts: &MoonshineArtifactPaths) -> Result<MoonshineRuntime, ModelError> {
    // transcribe-rs uses a global accelerator preference for ORT-backed models.
    // Force CPU because Moonshine incremental chunks are currently unstable on CUDA.
    set_ort_accelerator(OrtAccelerator::CpuOnly);
    tracing::info!(
        "Moonshine backend forcing CPU ORT execution; CUDA incremental chunking is not enabled"
    );

    let staged_root = StagedModelDir::prepare(artifacts)?;
    let model = StreamingModel::load(staged_root.path(), 4, &Quantization::default())
        .map_err(transcribe_error)?;

    Ok(MoonshineRuntime {
        model: Mutex::new(model),
        _staged_root: staged_root,
    })
}

fn transcribe_error(err: transcribe_rs::TranscribeError) -> ModelError {
    ModelError::BackendExecution {
        backend: "moonshine",
        operation: "transcribe",
        message: err.to_string(),
    }
}

struct MoonshineStream {
    spec: AudioSpec,
    params: TranscriptionParams,
    runtime: Arc<MoonshineRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
    raw_pcm: Vec<u8>,
    next_sequence: u64,
    saw_end_of_stream: bool,
}

#[async_trait]
impl TranscriptionStream for MoonshineStream {
    async fn push_chunk(
        &mut self,
        chunk: motlie_model::PcmChunk,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        if self.saw_end_of_stream {
            return Err(ModelError::InvalidConfiguration(
                "push_chunk called after end_of_stream".into(),
            ));
        }
        if chunk.sequence != self.next_sequence {
            return Err(ModelError::InvalidConfiguration(format!(
                "expected chunk sequence {}, got {}",
                self.next_sequence, chunk.sequence
            )));
        }

        self.next_sequence += 1;
        self.saw_end_of_stream = chunk.end_of_stream;

        if chunk.data.is_empty() && !chunk.end_of_stream {
            return Ok(None);
        }

        self.raw_pcm.extend_from_slice(&chunk.data);
        Ok(None)
    }

    async fn finish(self: Box<Self>) -> Result<TranscriptionUpdate, ModelError> {
        let started = Instant::now();
        let update = self
            .runtime
            .transcribe(&self.raw_pcm, &self.spec, &self.params)?;
        let elapsed = started.elapsed();

        {
            let mut metrics = lock_metrics(&self.metrics, "moonshine-finish");
            observe_latency(&mut metrics.runtime, elapsed);
        }

        Ok(update)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_defaults_to_secondary_small_streaming_bundle() {
        let spec = MoonshineStreamingSpec::small_en();

        assert_eq!(spec.id.as_str(), "moonshine_streaming_en");
        assert_eq!(spec.display_name, "Moonshine Streaming EN");
        assert_eq!(spec.frontend_filename, "onnx/small/frontend.ort");
    }

    #[test]
    fn adapter_reports_onnx_and_ort() {
        let adapter = MoonshineStreamingAdapter::small_en();

        assert_eq!(adapter.supported_formats(), &MOONSHINE_FORMATS);
        assert_eq!(adapter.backend_kind(), BackendKind::Ort);
    }
}
