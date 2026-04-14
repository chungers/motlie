use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, PcmChunk,
    PcmEncoding, QuantizationSupport, ResolvedCheckpoint, StartOptions, TranscriptSegment,
    TranscriptionModel, TranscriptionParams, TranscriptionStream, TranscriptionUpdate,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_ggml_model_path, RuntimeMetricState,
};

const WHISPER_CPP_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Ggml];

/// Whisper expects mono 16 kHz f32 PCM.
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decode step: trigger a decode every 500ms of new audio (8000 samples at 16kHz).
const DECODE_STEP_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize / 2;

/// Rolling window: decode the last 5 seconds of audio (80000 samples at 16kHz).
const WINDOW_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize * 5;

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
        options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
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
    let ctx =
        whisper_rs::WhisperContext::new_with_params(model_path.to_str().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "whisper model path `{}` contains non-UTF-8 characters",
                model_path.display()
            ))
        })?, params)
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
            samples_at_last_decode: 0,
            committed_segment_count: 0,
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
// Audio normalization: resample to 16kHz mono f32
// ---------------------------------------------------------------------------

/// Decode raw PCM bytes into f32 samples according to the stream's encoding.
fn decode_pcm_to_f32(data: &[u8], encoding: PcmEncoding) -> Result<Vec<f32>, ModelError> {
    match encoding {
        PcmEncoding::S16Le => {
            if !data.len().is_multiple_of(2) {
                return Err(ModelError::InvalidConfiguration(
                    "S16Le PCM data length must be even".into(),
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect())
        }
        PcmEncoding::F32Le => {
            if !data.len().is_multiple_of(4) {
                return Err(ModelError::InvalidConfiguration(
                    "F32Le PCM data length must be a multiple of 4".into(),
                ));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
    }
}

/// Downmix multi-channel audio to mono by averaging channels.
fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample audio from `src_rate` to `dst_rate` using linear interpolation.
///
/// This is a simple resampler suitable for the v1 ASR vertical slice. A
/// higher-quality resampler (e.g., sinc-based) can replace this if needed.
fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };
        output.push(sample as f32);
    }

    output
}

/// Full normalization pipeline: decode → downmix → resample → mono 16kHz f32.
fn normalize_chunk(
    data: &[u8],
    spec: &AudioSpec,
) -> Result<Vec<f32>, ModelError> {
    let decoded = decode_pcm_to_f32(data, spec.encoding)?;
    let mono = downmix_to_mono(&decoded, spec.channels);
    Ok(resample_linear(&mono, spec.sample_rate_hz, WHISPER_SAMPLE_RATE))
}

// ---------------------------------------------------------------------------
// Streaming runtime
// ---------------------------------------------------------------------------

struct WhisperCppStream {
    ctx: Arc<whisper_rs::WhisperContext>,
    spec: AudioSpec,
    params: TranscriptionParams,
    /// Accumulated mono 16kHz f32 PCM ready for whisper.
    pcm_buffer: Vec<f32>,
    /// Buffer length at the time of the last decode, used to detect step boundaries.
    samples_at_last_decode: usize,
    /// Number of segments from the previous decode that have been committed (emitted as final).
    committed_segment_count: usize,
    last_sequence: Option<u64>,
    end_of_stream_received: bool,
    metrics: Arc<Mutex<AsrMetrics>>,
}

impl WhisperCppStream {
    /// Decode the rolling window and return new segments, respecting `emit_partials`.
    fn run_decode(&mut self, is_final: bool) -> Result<TranscriptionUpdate, ModelError> {
        let started_at = Instant::now();

        // Use the rolling window: decode the last WINDOW_SAMPLES (or entire buffer if shorter).
        let window_start = self.pcm_buffer.len().saturating_sub(WINDOW_SAMPLES);
        let window = &self.pcm_buffer[window_start..];

        let mut state = self.ctx.create_state().map_err(|err| {
            ModelError::BackendExecution {
                backend: "whisper-cpp",
                operation: "create_state",
                message: err.to_string(),
            }
        })?;

        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        if let Some(ref lang) = self.params.language {
            params.set_language(Some(lang));
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);

        state
            .full(params, window)
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
        })? as usize;

        // Collect all decoded segments with timing relative to the window start.
        let window_offset_ms = (window_start as u64 * 1000) / WHISPER_SAMPLE_RATE as u64;
        let mut all_segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments as i32 {
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

            all_segments.push(TranscriptSegment {
                start_ms: window_offset_ms + (start * 10) as u64,
                end_ms: window_offset_ms + (end * 10) as u64,
                text,
                final_segment: false, // will be set below
            });
        }

        // Determine which segments are new (not yet committed).
        let new_start = self.committed_segment_count.min(all_segments.len());
        let mut output_segments = Vec::new();

        if is_final {
            // On finish: all segments from new_start onward are final.
            for seg in &mut all_segments[new_start..] {
                seg.final_segment = true;
                output_segments.push(seg.clone());
            }
            self.committed_segment_count = all_segments.len();
        } else if self.params.emit_partials {
            // Streaming with partials: segments except the last are final (stable),
            // the last is partial (may still be extended by incoming audio).
            let new_segments = &mut all_segments[new_start..];
            let new_len = new_segments.len();
            for (i, seg) in new_segments.iter_mut().enumerate() {
                seg.final_segment = i < new_len.saturating_sub(1);
                output_segments.push(seg.clone());
            }
            // Commit the final segments (all except the last partial).
            let finals_emitted = new_len.saturating_sub(1);
            self.committed_segment_count = new_start + finals_emitted;
        } else {
            // No partials: commit all except the last segment, emit only committed ones.
            let new_segments = &mut all_segments[new_start..];
            let new_len = new_segments.len();
            let finals_to_emit = new_len.saturating_sub(1);
            for seg in new_segments.iter_mut().take(finals_to_emit) {
                seg.final_segment = true;
                output_segments.push(seg.clone());
            }
            self.committed_segment_count = new_start + finals_to_emit;
        }

        self.samples_at_last_decode = self.pcm_buffer.len();

        let elapsed = started_at.elapsed();
        {
            let mut m = lock_metrics(&self.metrics, "whisper-cpp-decode");
            observe_latency(&mut m.runtime, elapsed);
        }

        Ok(TranscriptionUpdate {
            segments: output_segments,
        })
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
            let normalized = normalize_chunk(&chunk.data, &self.spec)?;
            self.pcm_buffer.extend(normalized);
        }

        // Trigger decode when enough new audio has accumulated since last decode.
        let new_samples = self.pcm_buffer.len() - self.samples_at_last_decode;
        if new_samples >= DECODE_STEP_SAMPLES {
            let update = self.run_decode(false)?;
            if update.segments.is_empty() {
                return Ok(None);
            }
            return Ok(Some(update));
        }

        Ok(None)
    }

    async fn finish(mut self: Box<Self>) -> Result<TranscriptionUpdate, ModelError> {
        if self.pcm_buffer.is_empty() {
            return Ok(TranscriptionUpdate::default());
        }
        self.run_decode(true)
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
        assert!(
            adapter
                .capabilities()
                .supports(CapabilityKind::Transcription)
        );
        assert_eq!(adapter.quantization(), &QuantizationSupport::none());
    }

    #[test]
    fn bundle_metadata_matches_spec() {
        let bundle =
            WhisperCppTranscriptionBundle::new(WhisperCppTranscriptionSpec::whisper_base_en());

        assert_eq!(bundle.id().as_str(), "whisper_base_en");
        assert!(
            bundle
                .capabilities()
                .supports(CapabilityKind::Transcription)
        );
        assert_eq!(bundle.metadata().quantization, QuantizationSupport::none());
    }

    #[test]
    fn s16le_normalization_produces_correct_f32_range() {
        let max_s16: i16 = i16::MAX;
        let bytes = max_s16.to_le_bytes();
        let normalized = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0;
        assert!((normalized - (32767.0 / 32768.0)).abs() < 1e-5);

        let min_s16: i16 = i16::MIN;
        let bytes = min_s16.to_le_bytes();
        let normalized = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0;
        assert!((normalized - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn downmix_stereo_to_mono_averages_channels() {
        let stereo = vec![1.0f32, -1.0, 0.5, 0.5, 0.0, 0.0];
        let mono = downmix_to_mono(&stereo, 2);

        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.0).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let mono = vec![1.0f32, 0.5, -0.5];
        let result = downmix_to_mono(&mono, 1);

        assert_eq!(result, mono);
    }

    #[test]
    fn resample_same_rate_is_passthrough() {
        let samples = vec![1.0f32, 2.0, 3.0];
        let result = resample_linear(&samples, 16000, 16000);

        assert_eq!(result, samples);
    }

    #[test]
    fn resample_halves_sample_count_for_2x_downsample() {
        // 32kHz → 16kHz should roughly halve the sample count.
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = resample_linear(&samples, 32000, 16000);

        // Output should be approximately half the input length.
        assert!(result.len() >= 49 && result.len() <= 51);
    }

    #[test]
    fn normalize_chunk_downmixes_and_resamples() {
        // 2-channel S16Le at 32kHz: 4 stereo frames → 2 mono samples at 16kHz (approx)
        let spec = AudioSpec {
            sample_rate_hz: 32_000,
            channels: 2,
            encoding: PcmEncoding::S16Le,
        };
        // 4 stereo frames = 8 i16 samples = 16 bytes
        let mut data = Vec::new();
        for _ in 0..8 {
            data.extend_from_slice(&1000i16.to_le_bytes());
        }

        let result = normalize_chunk(&data, &spec).expect("normalization should succeed");

        // After downmix: 4 mono samples; after resample 32k→16k: ~2 samples
        assert!(!result.is_empty());
        // All values should be valid finite floats
        assert!(result.iter().all(|v| v.is_finite()));
    }
}
