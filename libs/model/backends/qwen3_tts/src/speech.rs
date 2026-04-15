use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, PcmChunk,
    PcmEncoding, QuantizationSupport, ResolvedCheckpoint, SpeechModel, SpeechRequest, SpeechStream,
    StartOptions, TranscriptionModel, VoiceConditioning,
};
use motlie_model_ort::build_session;
use ndarray::{Array2, Array3};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_onnx_artifacts, Qwen3TtsArtifactPaths, Qwen3TtsConfig, RuntimeMetricState,
};

const QWEN3_TTS_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];

/// Duration of each output PCM chunk in milliseconds.
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;

/// Static bundle specification for a curated Qwen3-TTS ONNX voice.
#[derive(Clone, Debug)]
pub struct Qwen3TtsSpeechSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl Qwen3TtsSpeechSpec {
    pub fn qwen3_tts_12hz_0_6b() -> Self {
        Self {
            id: BundleId::new("qwen3_tts_12hz_0_6b"),
            display_name: "Qwen3-TTS 12Hz 0.6B",
            capabilities: Capabilities::speech_stream_only(),
            quantization: QuantizationSupport::none(),
        }
    }
}

/// Backend adapter for Qwen3-TTS over ONNX-exported model components.
#[derive(Clone, Debug)]
pub struct Qwen3TtsSpeechAdapter {
    spec: Qwen3TtsSpeechSpec,
}

impl Qwen3TtsSpeechAdapter {
    pub fn qwen3_tts_12hz_0_6b() -> Self {
        Self {
            spec: Qwen3TtsSpeechSpec::qwen3_tts_12hz_0_6b(),
        }
    }
}

#[async_trait]
impl BackendAdapter for Qwen3TtsSpeechAdapter {
    fn supported_formats(&self) -> &[CheckpointFormat] {
        &QWEN3_TTS_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Ort
    }

    fn capabilities(&self) -> &Capabilities {
        &self.spec.capabilities
    }

    fn quantization(&self) -> &QuantizationSupport {
        &self.spec.quantization
    }

    async fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
        self.spec
            .quantization
            .resolve(options.quantization, &identity.id)?;

        let artifacts = resolve_onnx_artifacts(checkpoint)?;
        let runtime = Arc::new(load_runtime(&artifacts)?);

        Ok(new_speech_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.spec.capabilities.clone(),
            self.spec.quantization.clone(),
            runtime,
        ))
    }
}

/// Generic `ModelBundle` backed by Qwen3-TTS ONNX components.
#[derive(Clone, Debug)]
pub struct Qwen3TtsSpeechBundle {
    metadata: BundleMetadata,
}

impl Qwen3TtsSpeechBundle {
    pub fn new(spec: Qwen3TtsSpeechSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id,
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities,
                quantization: spec.quantization,
            },
        }
    }
}

#[async_trait]
impl ModelBundle for Qwen3TtsSpeechBundle {
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
            configure_artifact_policy(policy)?
        } else {
            configure_artifact_policy(motlie_model::ArtifactPolicy::LocalOnly {
                root: PathBuf::from("."),
            })?
        };
        let runtime = Arc::new(load_runtime(&artifacts)?);

        Ok(new_speech_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            runtime,
        ))
    }
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

struct Qwen3TtsHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<Qwen3TtsRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct SpeechMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for Qwen3TtsHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "qwen3-tts-metric-snapshot").clone();
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
        Ok(self)
    }

    fn transcription(&self) -> Result<&dyn TranscriptionModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Transcription,
        ))
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl SpeechModel for Qwen3TtsHandle {
    async fn open_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Box<dyn SpeechStream>, ModelError> {
        let started_at = Instant::now();
        let pcm = self.runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&self.metrics, "qwen3-tts-open-stream");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(Box::new(Qwen3TtsSpeechStream::new(
            self.runtime.config.audio_spec(),
            pcm,
        )?))
    }
}

fn new_speech_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<Qwen3TtsRuntime>,
) -> Box<dyn BundleHandle> {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "qwen3-tts-start");
        observe_memory(&mut state.runtime);
    }

    Box::new(Qwen3TtsHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization: None,
        },
        runtime,
        metrics,
    })
}

// ---------------------------------------------------------------------------
// Multi-model ONNX runtime
// ---------------------------------------------------------------------------

struct Qwen3TtsRuntime {
    // The current `ort` RC exposes `Session::run(&mut self, ...)`, so shared
    // bundle handles must serialize access around the loaded sessions.
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    vocoder: Mutex<Session>,
    config: Qwen3TtsConfig,
}

impl Qwen3TtsRuntime {
    fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<u8>, ModelError> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request requires non-empty text".into(),
            ));
        }

        // Encode reference audio conditioning if provided.
        let ref_mel = match &request.conditioning {
            Some(VoiceConditioning::ReferenceAudio { audio_spec, pcm }) => {
                Some(self.encode_reference_audio(audio_spec, pcm)?)
            }
            Some(VoiceConditioning::SpeakerId(_)) => {
                return Err(ModelError::InvalidConfiguration(
                    "qwen3-tts backend uses VoiceConditioning::ReferenceAudio for voice cloning; SpeakerId is not supported".into(),
                ));
            }
            None => None,
        };

        // Step 1: Tokenize text → token IDs.
        let token_ids = self.tokenize(text)?;
        if token_ids.is_empty() {
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "tokenize",
                message: "tokenizer produced no tokens for request text".into(),
            });
        }

        // Step 2: Encoder — token IDs → hidden states.
        let hidden = self.run_encoder(&token_ids)?;

        // Step 3: Decoder — hidden states (+ optional ref mel) → mel spectrogram.
        let mel = self.run_decoder(&hidden, ref_mel.as_deref())?;

        // Step 4: Vocoder — mel spectrogram → raw audio samples.
        let samples = self.run_vocoder(&mel)?;

        // Convert f32 samples → PCM bytes matching the configured encoding.
        let pcm = encode_pcm(&samples, self.config.audio_spec().encoding);

        if pcm.is_empty() {
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "synthesize",
                message: "qwen3-tts synthesis produced no PCM output".into(),
            });
        }

        Ok(pcm)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i64>, ModelError> {
        // Simple character-level tokenization for the ONNX-exported encoder.
        // The actual Qwen3-TTS model uses a SentencePiece/BPE tokenizer;
        // a proper tokenizer should replace this once the ONNX export pipeline
        // and vocabulary files are standardized.
        let ids: Vec<i64> = text.chars().map(|c| c as i64).collect();
        Ok(ids)
    }

    fn encode_reference_audio(
        &self,
        audio_spec: &AudioSpec,
        pcm: &[u8],
    ) -> Result<Vec<f32>, ModelError> {
        // Decode PCM to f32 samples, then compute a simple log-mel spectrogram
        // as conditioning input for the decoder.
        let samples = decode_pcm_to_f32(pcm, audio_spec.encoding)?;
        if samples.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "reference audio conditioning requires non-empty PCM data".into(),
            ));
        }

        // Compute simplified mel spectrogram: chunk samples into frames,
        // compute energy per mel bin via simple FFT-free approximation.
        // A production implementation should use a proper STFT + mel filterbank.
        let mel_channels = self.config.mel_channels as usize;
        let hop_length = self.config.hop_length as usize;
        let num_frames = samples.len() / hop_length.max(1);

        let mut mel = Vec::with_capacity(num_frames * mel_channels);
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            let end = (start + hop_length).min(samples.len());
            let frame = &samples[start..end];

            // Distribute frame energy across mel bins.
            let energy: f32 = frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32;
            let log_energy = (energy + 1e-10).ln();

            for bin in 0..mel_channels {
                // Spread energy with a frequency-dependent offset per bin.
                let bin_weight = (bin as f32 + 1.0) / mel_channels as f32;
                mel.push(log_energy * bin_weight);
            }
        }

        Ok(mel)
    }

    fn run_encoder(&self, token_ids: &[i64]) -> Result<Vec<f32>, ModelError> {
        let seq_len = token_ids.len();
        let input = Array2::<i64>::from_shape_vec((1, seq_len), token_ids.to_vec()).map_err(
            |err| ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "build_encoder_input",
                message: err.to_string(),
            },
        )?;

        let input_tensor = Tensor::<i64>::from_array(input).map_err(ort_error)?;
        let inputs = vec![SessionInputValue::from(input_tensor)];

        let mut session = self
            .encoder
            .lock()
            .map_err(|_| ModelError::Internal("qwen3-tts encoder mutex poisoned".into()))?;
        let outputs = session
            .run(inputs.as_slice())
            .map_err(|err| ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "run_encoder",
                message: err.to_string(),
            })?;

        let (_, hidden) =
            outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|err| ModelError::BackendExecution {
                    backend: "qwen3-tts",
                    operation: "extract_encoder_output",
                    message: err.to_string(),
                })?;

        Ok(hidden.to_vec())
    }

    fn run_decoder(
        &self,
        hidden: &[f32],
        ref_mel: Option<&[f32]>,
    ) -> Result<Vec<f32>, ModelError> {
        // Build decoder input: hidden states as a 1×T×D tensor.
        // For simplicity, infer dimensions from the flat hidden vector.
        let hidden_len = hidden.len();
        let hidden_input = Array2::<f32>::from_shape_vec((1, hidden_len), hidden.to_vec())
            .map_err(|err| ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "build_decoder_input",
                message: err.to_string(),
            })?;
        let hidden_tensor = Tensor::<f32>::from_array(hidden_input).map_err(ort_error)?;

        let mut inputs = vec![SessionInputValue::from(hidden_tensor)];

        // If reference mel is provided, add it as the second input for voice cloning.
        if let Some(mel) = ref_mel {
            let mel_channels = self.config.mel_channels as usize;
            let mel_frames = mel.len() / mel_channels.max(1);
            let mel_input =
                Array3::<f32>::from_shape_vec((1, mel_frames, mel_channels), mel.to_vec())
                    .map_err(|err| ModelError::BackendExecution {
                        backend: "qwen3-tts",
                        operation: "build_decoder_ref_mel",
                        message: err.to_string(),
                    })?;
            let mel_tensor = Tensor::<f32>::from_array(mel_input).map_err(ort_error)?;
            inputs.push(SessionInputValue::from(mel_tensor));
        }

        let mut session = self
            .decoder
            .lock()
            .map_err(|_| ModelError::Internal("qwen3-tts decoder mutex poisoned".into()))?;
        let outputs = session
            .run(inputs.as_slice())
            .map_err(|err| ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "run_decoder",
                message: err.to_string(),
            })?;

        let (_, mel_out) =
            outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|err| ModelError::BackendExecution {
                    backend: "qwen3-tts",
                    operation: "extract_decoder_output",
                    message: err.to_string(),
                })?;

        Ok(mel_out.to_vec())
    }

    fn run_vocoder(&self, mel: &[f32]) -> Result<Vec<f32>, ModelError> {
        let mel_channels = self.config.mel_channels as usize;
        let mel_frames = mel.len() / mel_channels.max(1);
        let mel_input =
            Array3::<f32>::from_shape_vec((1, mel_channels, mel_frames), mel.to_vec()).map_err(
                |err| ModelError::BackendExecution {
                    backend: "qwen3-tts",
                    operation: "build_vocoder_input",
                    message: err.to_string(),
                },
            )?;

        let mel_tensor = Tensor::<f32>::from_array(mel_input).map_err(ort_error)?;
        let inputs = vec![SessionInputValue::from(mel_tensor)];

        let mut session = self
            .vocoder
            .lock()
            .map_err(|_| ModelError::Internal("qwen3-tts vocoder mutex poisoned".into()))?;
        let outputs = session
            .run(inputs.as_slice())
            .map_err(|err| ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "run_vocoder",
                message: err.to_string(),
            })?;

        let (_, audio) =
            outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|err| ModelError::BackendExecution {
                    backend: "qwen3-tts",
                    operation: "extract_vocoder_output",
                    message: err.to_string(),
                })?;

        Ok(audio.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Qwen3-TTS streams expose already-synthesized PCM in monotonic chunks.
///
/// Like Piper in Phase 1, `open_stream()` runs the full multi-model pipeline
/// up front (encoder → decoder → vocoder) and `next_chunk()` yields buffered
/// PCM for sink adapters. A truly streaming implementation would require
/// autoregressive token generation, which is deferred to a future iteration.
struct Qwen3TtsSpeechStream {
    audio_spec: AudioSpec,
    pcm: Vec<u8>,
    offset: usize,
    next_sequence: u64,
    chunk_len_bytes: usize,
}

impl Qwen3TtsSpeechStream {
    fn new(audio_spec: AudioSpec, pcm: Vec<u8>) -> Result<Self, ModelError> {
        let bytes_per_sample = match audio_spec.encoding {
            PcmEncoding::S16Le => 2,
            PcmEncoding::F32Le => 4,
        };
        let frames_per_chunk =
            ((audio_spec.sample_rate_hz as u64 * OUTPUT_CHUNK_DURATION_MS as u64) / 1000) as usize;
        let chunk_len_bytes =
            frames_per_chunk.max(1) * audio_spec.channels as usize * bytes_per_sample;

        Ok(Self {
            audio_spec,
            pcm,
            offset: 0,
            next_sequence: 0,
            chunk_len_bytes,
        })
    }
}

#[async_trait]
impl SpeechStream for Qwen3TtsSpeechStream {
    fn audio_spec(&self) -> &AudioSpec {
        &self.audio_spec
    }

    async fn next_chunk(&mut self) -> Result<Option<PcmChunk>, ModelError> {
        if self.offset >= self.pcm.len() {
            return Ok(None);
        }

        let end = (self.offset + self.chunk_len_bytes).min(self.pcm.len());
        let chunk = PcmChunk {
            data: self.pcm[self.offset..end].to_vec(),
            sequence: self.next_sequence,
            end_of_stream: end == self.pcm.len(),
        };
        self.offset = end;
        self.next_sequence = self.next_sequence.saturating_add(1);

        Ok(Some(chunk))
    }

    async fn finish(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_runtime(artifacts: &Qwen3TtsArtifactPaths) -> Result<Qwen3TtsRuntime, ModelError> {
    Ok(Qwen3TtsRuntime {
        encoder: Mutex::new(build_session("qwen3-tts", &artifacts.encoder)?),
        decoder: Mutex::new(build_session("qwen3-tts", &artifacts.decoder)?),
        vocoder: Mutex::new(build_session("qwen3-tts", &artifacts.vocoder)?),
        config: Qwen3TtsConfig::from_path(&artifacts.config)?,
    })
}

fn decode_pcm_to_f32(pcm: &[u8], encoding: PcmEncoding) -> Result<Vec<f32>, ModelError> {
    match encoding {
        PcmEncoding::S16Le => {
            if !pcm.len().is_multiple_of(2) {
                return Err(ModelError::InvalidConfiguration(
                    "S16Le reference audio length must be even".into(),
                ));
            }
            Ok(pcm
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect())
        }
        PcmEncoding::F32Le => {
            if !pcm.len().is_multiple_of(4) {
                return Err(ModelError::InvalidConfiguration(
                    "F32Le reference audio length must be a multiple of 4".into(),
                ));
            }
            Ok(pcm
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
    }
}

fn encode_pcm(samples: &[f32], encoding: PcmEncoding) -> Vec<u8> {
    match encoding {
        PcmEncoding::S16Le => {
            let mut out = Vec::with_capacity(samples.len() * 2);
            for sample in samples {
                let clamped = sample.clamp(-1.0, 1.0);
                let as_i16 = (clamped * i16::MAX as f32) as i16;
                out.extend_from_slice(&as_i16.to_le_bytes());
            }
            out
        }
        PcmEncoding::F32Le => {
            let mut out = Vec::with_capacity(samples.len() * 4);
            for sample in samples {
                out.extend_from_slice(&sample.to_le_bytes());
            }
            out
        }
    }
}

fn ort_error(err: ort::Error) -> ModelError {
    ModelError::BackendExecution {
        backend: "qwen3-tts",
        operation: "build_tensor",
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::{BackendAdapter, BackendKind, QuantizationSupport};

    #[test]
    fn spec_has_expected_identity() {
        let spec = Qwen3TtsSpeechSpec::qwen3_tts_12hz_0_6b();

        assert_eq!(spec.id.as_str(), "qwen3_tts_12hz_0_6b");
        assert_eq!(spec.display_name, "Qwen3-TTS 12Hz 0.6B");
        assert!(spec.capabilities.supports(CapabilityKind::Speech));
        assert!(!spec.capabilities.supports(CapabilityKind::Chat));
    }

    #[test]
    fn adapter_reports_backend_metadata() {
        let adapter = Qwen3TtsSpeechAdapter::qwen3_tts_12hz_0_6b();

        assert_eq!(adapter.supported_formats(), &[CheckpointFormat::Onnx]);
        assert_eq!(adapter.backend_kind(), BackendKind::Ort);
        assert!(adapter.capabilities().supports(CapabilityKind::Speech));
        assert_eq!(adapter.quantization(), &QuantizationSupport::none());
    }

    #[test]
    fn bundle_metadata_matches_spec() {
        let bundle = Qwen3TtsSpeechBundle::new(Qwen3TtsSpeechSpec::qwen3_tts_12hz_0_6b());

        assert_eq!(bundle.id().as_str(), "qwen3_tts_12hz_0_6b");
        assert!(bundle.capabilities().supports(CapabilityKind::Speech));
        assert_eq!(bundle.metadata().quantization, QuantizationSupport::none());
    }

    #[tokio::test]
    async fn stream_emits_monotonic_chunks_and_finishes() {
        let audio_spec = AudioSpec {
            sample_rate_hz: 22_050,
            channels: 1,
            encoding: PcmEncoding::F32Le,
        };
        let pcm = vec![0_u8; 10_000];
        let mut stream =
            Qwen3TtsSpeechStream::new(audio_spec, pcm).expect("stream should build");

        let mut prev_seq = None;
        let mut saw_final = false;
        while let Some(chunk) = stream.next_chunk().await.expect("should succeed") {
            if let Some(prev) = prev_seq {
                assert!(chunk.sequence > prev);
            }
            prev_seq = Some(chunk.sequence);
            saw_final = chunk.end_of_stream;
        }

        assert!(saw_final);
        assert!(stream
            .next_chunk()
            .await
            .expect("should stay exhausted")
            .is_none());
    }

    #[test]
    fn encode_pcm_s16le_round_trips() {
        let samples = vec![0.5_f32, -0.5, 0.0, 1.0, -1.0];
        let encoded = encode_pcm(&samples, PcmEncoding::S16Le);
        assert_eq!(encoded.len(), samples.len() * 2);

        let decoded = decode_pcm_to_f32(&encoded, PcmEncoding::S16Le)
            .expect("round-trip should succeed");
        for (orig, dec) in samples.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.001, "orig={orig}, decoded={dec}");
        }
    }

    #[test]
    fn encode_pcm_f32le_round_trips() {
        let samples = vec![0.5_f32, -0.5, 0.0];
        let encoded = encode_pcm(&samples, PcmEncoding::F32Le);
        assert_eq!(encoded.len(), samples.len() * 4);

        let decoded = decode_pcm_to_f32(&encoded, PcmEncoding::F32Le)
            .expect("round-trip should succeed");
        assert_eq!(samples, decoded);
    }

    #[test]
    fn decode_pcm_rejects_misaligned_s16le() {
        let err = decode_pcm_to_f32(&[0x00, 0x01, 0x02], PcmEncoding::S16Le)
            .expect_err("odd-length S16Le should fail");

        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("even")));
    }

    #[test]
    fn decode_pcm_rejects_misaligned_f32le() {
        let err = decode_pcm_to_f32(&[0x00, 0x01], PcmEncoding::F32Le)
            .expect_err("non-4-aligned F32Le should fail");

        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("multiple of 4")));
    }
}
