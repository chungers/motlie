use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity,
    ModelMetricSnapshot, PcmChunk, PcmEncoding, QuantizationSupport, ResolvedCheckpoint,
    StartOptions, TranscriptSegment, TranscriptionModel, TranscriptionParams, TranscriptionStream,
    TranscriptionUpdate,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_ggml_model_path, RuntimeMetricState,
};

const WHISPER_CPP_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Ggml];

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
            capabilities: Capabilities::transcription_stream_only(),
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
        _options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
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
    let ctx =
        whisper_rs::WhisperContext::new_with_params(model_path.to_str().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "whisper model path `{}` contains non-UTF-8 characters",
                model_path.display()
            ))
        })?, params)
        .map_err(|err| ModelError::BackendInitialization {
            backend: "whisper-cpp",
            message: format!("failed to load whisper model from `{}`: {err}", model_path.display()),
        })?;

    Ok(Arc::new(ctx))
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

struct WhisperCppHandle {
    descriptor: LoadedBundleDescriptor,
    ctx: Arc<whisper_rs::WhisperContext>,
    metrics: Arc<Mutex<AsrMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct AsrMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for WhisperCppHandle {
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
                last_latency: metrics.runtime.last_latency_msec.map(motlie_model::Milliseconds),
                max_latency: metrics.runtime.max_latency_msec.map(motlie_model::Milliseconds),
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

    fn transcription(&self) -> Result<&dyn TranscriptionModel, ModelError> {
        Ok(self)
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl TranscriptionModel for WhisperCppHandle {
    async fn open_stream(
        &self,
        spec: AudioSpec,
        params: TranscriptionParams,
    ) -> Result<Box<dyn TranscriptionStream>, ModelError> {
        Ok(Box::new(WhisperCppStream {
            ctx: Arc::clone(&self.ctx),
            spec,
            params,
            pcm_buffer: Vec::new(),
            last_sequence: None,
            end_of_stream_received: false,
            metrics: Arc::clone(&self.metrics),
        }))
    }
}

fn new_transcription_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    ctx: Arc<whisper_rs::WhisperContext>,
) -> Box<dyn BundleHandle> {
    let metrics = Arc::new(Mutex::new(AsrMetrics::default()));
    {
        let mut m = lock_metrics(&metrics, "whisper-cpp-start");
        observe_memory(&mut m.runtime);
    }

    Box::new(WhisperCppHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization: None,
        },
        ctx,
        metrics,
    })
}

// ---------------------------------------------------------------------------
// Streaming runtime
// ---------------------------------------------------------------------------

/// Decode step interval in seconds worth of audio samples.
const DECODE_STEP_SAMPLES: usize = 16_000 / 2; // 500ms at 16kHz

struct WhisperCppStream {
    ctx: Arc<whisper_rs::WhisperContext>,
    spec: AudioSpec,
    params: TranscriptionParams,
    pcm_buffer: Vec<f32>,
    last_sequence: Option<u64>,
    end_of_stream_received: bool,
    metrics: Arc<Mutex<AsrMetrics>>,
}

impl WhisperCppStream {
    fn normalize_and_append(&mut self, chunk: &PcmChunk) -> Result<(), ModelError> {
        match self.spec.encoding {
            PcmEncoding::S16Le => {
                if chunk.data.len() % 2 != 0 {
                    return Err(ModelError::InvalidConfiguration(
                        "S16Le PCM data length must be even".into(),
                    ));
                }
                for sample_bytes in chunk.data.chunks_exact(2) {
                    let sample = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                    self.pcm_buffer.push(sample as f32 / 32768.0);
                }
            }
            PcmEncoding::F32Le => {
                if chunk.data.len() % 4 != 0 {
                    return Err(ModelError::InvalidConfiguration(
                        "F32Le PCM data length must be a multiple of 4".into(),
                    ));
                }
                for sample_bytes in chunk.data.chunks_exact(4) {
                    let sample =
                        f32::from_le_bytes([sample_bytes[0], sample_bytes[1], sample_bytes[2], sample_bytes[3]]);
                    self.pcm_buffer.push(sample);
                }
            }
        }
        Ok(())
    }

    fn run_decode(&self) -> Result<TranscriptionUpdate, ModelError> {
        let started_at = Instant::now();

        let mut state = self.ctx.create_state().map_err(|err| {
            ModelError::BackendExecution {
                backend: "whisper-cpp",
                operation: "create_state",
                message: err.to_string(),
            }
        })?;

        let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        if let Some(ref lang) = self.params.language {
            params.set_language(Some(lang));
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);

        state
            .full(params, &self.pcm_buffer)
            .map_err(|err| ModelError::BackendExecution {
                backend: "whisper-cpp",
                operation: "full",
                message: err.to_string(),
            })?;

        let num_segments = state.full_n_segments().map_err(|err| {
            ModelError::BackendExecution {
                backend: "whisper-cpp",
                operation: "full_n_segments",
                message: err.to_string(),
            }
        })?;

        let mut segments = Vec::with_capacity(num_segments as usize);
        for i in 0..num_segments {
            let text = state.full_get_segment_text(i).map_err(|err| {
                ModelError::BackendExecution {
                    backend: "whisper-cpp",
                    operation: "full_get_segment_text",
                    message: err.to_string(),
                }
            })?;
            let start = state.full_get_segment_t0(i).map_err(|err| {
                ModelError::BackendExecution {
                    backend: "whisper-cpp",
                    operation: "full_get_segment_t0",
                    message: err.to_string(),
                }
            })?;
            let end = state.full_get_segment_t1(i).map_err(|err| {
                ModelError::BackendExecution {
                    backend: "whisper-cpp",
                    operation: "full_get_segment_t1",
                    message: err.to_string(),
                }
            })?;

            segments.push(TranscriptSegment {
                start_ms: (start * 10) as u64,
                end_ms: (end * 10) as u64,
                text,
                final_segment: true,
            });
        }

        let elapsed = started_at.elapsed();
        {
            let mut m = lock_metrics(&self.metrics, "whisper-cpp-decode");
            observe_latency(&mut m.runtime, elapsed);
        }

        Ok(TranscriptionUpdate { segments })
    }
}

#[async_trait]
impl TranscriptionStream for WhisperCppStream {
    async fn push_chunk(
        &mut self,
        chunk: PcmChunk,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        // Edge case: push after EOS
        if self.end_of_stream_received {
            return Err(ModelError::InvalidConfiguration(
                "push_chunk called after end_of_stream was received".into(),
            ));
        }

        // Edge case: non-monotonic sequence
        if let Some(last) = self.last_sequence {
            if chunk.sequence <= last {
                return Err(ModelError::InvalidConfiguration(format!(
                    "non-monotonic chunk sequence: got {}, last was {last}",
                    chunk.sequence
                )));
            }
        }
        self.last_sequence = Some(chunk.sequence);

        // Edge case: empty data (non-EOS)
        if chunk.data.is_empty() && !chunk.end_of_stream {
            return Ok(None);
        }

        if chunk.end_of_stream {
            self.end_of_stream_received = true;
        }

        if !chunk.data.is_empty() {
            self.normalize_and_append(&chunk)?;
        }

        // Trigger decode when we have enough samples
        if self.pcm_buffer.len() >= DECODE_STEP_SAMPLES {
            let update = self.run_decode()?;
            if update.segments.is_empty() {
                return Ok(None);
            }
            return Ok(Some(update));
        }

        Ok(None)
    }

    async fn finish(self: Box<Self>) -> Result<TranscriptionUpdate, ModelError> {
        if self.pcm_buffer.is_empty() {
            return Ok(TranscriptionUpdate::default());
        }
        self.run_decode()
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
        assert!(adapter.capabilities().supports(CapabilityKind::Transcription));
        assert_eq!(adapter.quantization(), &QuantizationSupport::none());
    }

    #[test]
    fn bundle_metadata_matches_spec() {
        let bundle = WhisperCppTranscriptionBundle::new(
            WhisperCppTranscriptionSpec::whisper_base_en(),
        );

        assert_eq!(bundle.id().as_str(), "whisper_base_en");
        assert!(bundle.capabilities().supports(CapabilityKind::Transcription));
        assert_eq!(bundle.metadata().quantization, QuantizationSupport::none());
    }

    #[test]
    fn s16le_normalization_produces_correct_f32_range() {
        // Test the normalization math in isolation without requiring a WhisperContext.
        let max_s16: i16 = i16::MAX;
        let bytes = max_s16.to_le_bytes();
        // S16Le: divide by 32768.0 to normalize to [-1.0, 1.0)
        let normalized = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0;
        assert!((normalized - (32767.0 / 32768.0)).abs() < 1e-5);

        let min_s16: i16 = i16::MIN;
        let bytes = min_s16.to_le_bytes();
        let normalized = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0;
        assert!((normalized - (-1.0)).abs() < 1e-5);
    }
}
