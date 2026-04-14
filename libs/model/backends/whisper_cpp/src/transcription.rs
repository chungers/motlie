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
            normalizer: NormalizerState::new(),
            pcm_buffer: Vec::new(),
            total_samples_trimmed: 0,
            samples_at_last_decode: 0,
            committed_end_ms: 0,
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
// Stateful audio normalization: decode → downmix → resample to 16kHz mono f32
// ---------------------------------------------------------------------------

/// Per-stream normalization state that carries incomplete data across chunk
/// boundaries. This is necessary because the stream contract allows arbitrary
/// chunking — a chunk may end mid-sample (raw bytes) or mid-frame (multichannel).
struct NormalizerState {
    /// Leftover raw bytes that didn't form a complete sample at the end of the
    /// previous chunk. Prepended to the next chunk's data before decoding.
    pending_bytes: Vec<u8>,
    /// Leftover decoded f32 samples that didn't form a complete multichannel
    /// frame at the end of the previous chunk (only relevant when channels > 1).
    pending_channel_samples: Vec<f32>,
    /// Fractional resampler position carried across chunks so that linear
    /// interpolation is phase-continuous at chunk boundaries.
    resample_cursor: f64,
}

impl NormalizerState {
    fn new() -> Self {
        Self {
            pending_bytes: Vec::new(),
            pending_channel_samples: Vec::new(),
            resample_cursor: 0.0,
        }
    }

    /// Normalize a chunk of raw PCM bytes into mono 16 kHz f32 samples, carrying
    /// state across chunk boundaries for byte alignment, channel framing, and
    /// resampler phase continuity.
    fn normalize(
        &mut self,
        data: &[u8],
        spec: &AudioSpec,
    ) -> Result<Vec<f32>, ModelError> {
        // 1. Decode raw bytes → f32 samples, carrying pending bytes.
        let mut raw = std::mem::take(&mut self.pending_bytes);
        raw.extend_from_slice(data);

        let sample_bytes: usize = match spec.encoding {
            PcmEncoding::S16Le => 2,
            PcmEncoding::F32Le => 4,
        };
        let aligned_len = raw.len() - (raw.len() % sample_bytes);
        self.pending_bytes = raw[aligned_len..].to_vec();
        let aligned = &raw[..aligned_len];

        let decoded: Vec<f32> = match spec.encoding {
            PcmEncoding::S16Le => aligned
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect(),
            PcmEncoding::F32Le => aligned
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        };

        // 2. Downmix to mono, carrying pending channel samples.
        let mono = if spec.channels <= 1 {
            decoded
        } else {
            let ch = spec.channels as usize;
            let mut all_samples = std::mem::take(&mut self.pending_channel_samples);
            all_samples.extend(decoded);

            let complete_frames = all_samples.len() / ch;
            let used = complete_frames * ch;
            self.pending_channel_samples = all_samples[used..].to_vec();

            all_samples[..used]
                .chunks_exact(ch)
                .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        };

        // 3. Resample to 16 kHz with phase-continuous cursor.
        if spec.sample_rate_hz == WHISPER_SAMPLE_RATE || mono.is_empty() {
            return Ok(mono);
        }

        let ratio = spec.sample_rate_hz as f64 / WHISPER_SAMPLE_RATE as f64;
        let mut output = Vec::new();
        let mut cursor = self.resample_cursor;

        while cursor < mono.len() as f64 {
            let idx = cursor as usize;
            let frac = cursor - idx as f64;

            let sample = if idx + 1 < mono.len() {
                mono[idx] as f64 * (1.0 - frac) + mono[idx + 1] as f64 * frac
            } else {
                mono[idx] as f64
            };
            output.push(sample as f32);
            cursor += ratio;
        }

        // Carry the fractional cursor for the next chunk, offset by the samples consumed.
        self.resample_cursor = cursor - mono.len() as f64;

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Streaming runtime
// ---------------------------------------------------------------------------

struct WhisperCppStream {
    ctx: Arc<whisper_rs::WhisperContext>,
    spec: AudioSpec,
    params: TranscriptionParams,
    normalizer: NormalizerState,
    /// Accumulated mono 16 kHz f32 PCM ready for whisper. Trimmed after each
    /// decode to keep at most `WINDOW_SAMPLES` samples.
    pcm_buffer: Vec<f32>,
    /// Total mono 16 kHz samples consumed before the current `pcm_buffer[0]`.
    /// Used to compute absolute timestamps from window-relative decode output.
    total_samples_trimmed: u64,
    /// Buffer length at the time of the last decode, used to detect step boundaries.
    /// Measured relative to the current (possibly trimmed) buffer, not absolute.
    samples_at_last_decode: usize,
    /// Absolute end timestamp (ms) of the last committed (final) segment. Segments
    /// with `end_ms <= committed_end_ms` are already-emitted and skipped on the
    /// next decode. This is stable across buffer trims because it's timestamp-based,
    /// not index-based.
    committed_end_ms: u64,
    last_sequence: Option<u64>,
    end_of_stream_received: bool,
    metrics: Arc<Mutex<AsrMetrics>>,
}

impl WhisperCppStream {
    /// Decode the rolling window and return new segments, respecting `emit_partials`.
    fn run_decode(&mut self, is_final: bool) -> Result<TranscriptionUpdate, ModelError> {
        let started_at = Instant::now();

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
        })? as usize;

        // Absolute time offset: samples trimmed before the current buffer start.
        let buffer_offset_ms =
            (self.total_samples_trimmed * 1000) / WHISPER_SAMPLE_RATE as u64;

        let mut all_segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments as i32 {
            let text = state.full_get_segment_text(i).map_err(|err| {
                ModelError::BackendExecution {
                    backend: "whisper-cpp",
                    operation: "full_get_segment_text",
                    message: err.to_string(),
                }
            })?;
            let t0 = state.full_get_segment_t0(i).map_err(|err| {
                ModelError::BackendExecution {
                    backend: "whisper-cpp",
                    operation: "full_get_segment_t0",
                    message: err.to_string(),
                }
            })?;
            let t1 = state.full_get_segment_t1(i).map_err(|err| {
                ModelError::BackendExecution {
                    backend: "whisper-cpp",
                    operation: "full_get_segment_t1",
                    message: err.to_string(),
                }
            })?;

            all_segments.push(TranscriptSegment {
                start_ms: buffer_offset_ms + (t0 * 10) as u64,
                end_ms: buffer_offset_ms + (t1 * 10) as u64,
                text,
                final_segment: false,
            });
        }

        // Filter out already-committed segments (timestamp-based, stable across trims).
        let new_segments: Vec<_> = all_segments
            .into_iter()
            .filter(|seg| seg.end_ms > self.committed_end_ms)
            .collect();

        let mut output_segments = Vec::new();

        if is_final {
            for mut seg in new_segments {
                seg.final_segment = true;
                if seg.end_ms > self.committed_end_ms {
                    self.committed_end_ms = seg.end_ms;
                }
                output_segments.push(seg);
            }
        } else if self.params.emit_partials {
            let n = new_segments.len();
            for (i, mut seg) in new_segments.into_iter().enumerate() {
                seg.final_segment = i < n.saturating_sub(1);
                if seg.final_segment && seg.end_ms > self.committed_end_ms {
                    self.committed_end_ms = seg.end_ms;
                }
                output_segments.push(seg);
            }
        } else {
            let n = new_segments.len();
            let finals_to_emit = n.saturating_sub(1);
            for mut seg in new_segments.into_iter().take(finals_to_emit) {
                seg.final_segment = true;
                if seg.end_ms > self.committed_end_ms {
                    self.committed_end_ms = seg.end_ms;
                }
                output_segments.push(seg);
            }
        }

        self.samples_at_last_decode = self.pcm_buffer.len();

        // Trim buffer to the rolling window to bound memory.
        if self.pcm_buffer.len() > WINDOW_SAMPLES {
            let trim = self.pcm_buffer.len() - WINDOW_SAMPLES;
            self.pcm_buffer.drain(..trim);
            self.total_samples_trimmed += trim as u64;
            self.samples_at_last_decode = self.pcm_buffer.len();
        }

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
            let normalized = self.normalizer.normalize(&chunk.data, &self.spec)?;
            self.pcm_buffer.extend(normalized);
        }

        // Trigger decode when enough new audio has accumulated since last decode.
        let new_samples = self.pcm_buffer.len().saturating_sub(self.samples_at_last_decode);
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
    fn normalizer_carries_pending_bytes_across_chunks() {
        let spec = AudioSpec {
            sample_rate_hz: 16_000,
            channels: 1,
            encoding: PcmEncoding::S16Le,
        };
        let mut norm = NormalizerState::new();

        // Send 3 bytes: 1 complete S16Le sample (2 bytes) + 1 pending byte.
        let data = vec![0x00, 0x40, 0xFF]; // 0x4000 = 16384 → ~0.5; 0xFF pending
        let result = norm.normalize(&data, &spec).expect("normalize should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 0.01);
        assert_eq!(norm.pending_bytes.len(), 1);

        // Send 1 more byte to complete the second sample: 0xFF00 = -256 → small negative
        let data2 = vec![0x00];
        let result2 = norm.normalize(&data2, &spec).expect("normalize should succeed");
        assert_eq!(result2.len(), 1);
        assert!(norm.pending_bytes.is_empty());
    }

    #[test]
    fn normalizer_carries_partial_multichannel_frames() {
        let spec = AudioSpec {
            sample_rate_hz: 16_000,
            channels: 2,
            encoding: PcmEncoding::S16Le,
        };
        let mut norm = NormalizerState::new();

        // 3 S16Le samples = 6 bytes. With 2 channels, that's 1 complete frame + 1 pending sample.
        let data: Vec<u8> = [1000i16, -1000, 500]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = norm.normalize(&data, &spec).expect("normalize should succeed");
        // 1 complete stereo frame → 1 mono sample (average of 1000 and -1000 → 0)
        assert_eq!(result.len(), 1);
        assert!((result[0]).abs() < 0.01);
        assert_eq!(norm.pending_channel_samples.len(), 1);

        // Send the second channel of the pending frame.
        let data2: Vec<u8> = 500i16.to_le_bytes().to_vec();
        let result2 = norm.normalize(&data2, &spec).expect("normalize should succeed");
        // Completes the second frame: average of 500 and 500 → 500
        assert_eq!(result2.len(), 1);
        assert!(norm.pending_channel_samples.is_empty());
    }

    #[test]
    fn normalizer_resamples_with_phase_continuity() {
        let spec = AudioSpec {
            sample_rate_hz: 32_000,
            channels: 1,
            encoding: PcmEncoding::F32Le,
        };
        let mut norm = NormalizerState::new();

        // Send 100 f32 samples at 32kHz in two chunks of 50.
        let samples1: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
        let data1: Vec<u8> = samples1.iter().flat_map(|s| s.to_le_bytes()).collect();
        let out1 = norm.normalize(&data1, &spec).expect("chunk 1");

        let samples2: Vec<f32> = (50..100).map(|i| i as f32 / 50.0).collect();
        let data2: Vec<u8> = samples2.iter().flat_map(|s| s.to_le_bytes()).collect();
        let out2 = norm.normalize(&data2, &spec).expect("chunk 2");

        // Combined output should be ~50 samples (32kHz → 16kHz = halved).
        let total = out1.len() + out2.len();
        assert!(total >= 48 && total <= 52, "expected ~50 resampled samples, got {total}");
        // All values should be finite and monotonically increasing (since input is a ramp).
        let combined: Vec<f32> = out1.into_iter().chain(out2).collect();
        assert!(combined.iter().all(|v| v.is_finite()));
        for w in combined.windows(2) {
            assert!(w[1] >= w[0], "resampled ramp should be monotonic");
        }
    }

    #[test]
    fn normalizer_16khz_mono_is_passthrough() {
        let spec = AudioSpec {
            sample_rate_hz: 16_000,
            channels: 1,
            encoding: PcmEncoding::F32Le,
        };
        let mut norm = NormalizerState::new();

        let samples = vec![0.1f32, 0.2, 0.3];
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let result = norm.normalize(&data, &spec).expect("passthrough");

        assert_eq!(result.len(), 3);
        for (a, b) in result.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
