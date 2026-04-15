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
use ndarray::Array2;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::common::{
    compute_log_mel_spectrogram, configure_artifact_policy, decode_pcm_to_f32, downmix_to_mono,
    encode_pcm, lock_metrics, observe_latency, observe_memory, resample_mono,
    resolve_onnx_artifacts, Qwen3TtsArtifactPaths, Qwen3TtsConfig, RuntimeMetricState, Vocabulary,
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
            tracing::warn!(
                "no artifact_policy provided for qwen3-tts bundle; \
                 defaulting to LocalOnly with current working directory"
            );
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
    // The full encoder → decoder → vocoder pipeline is serialized behind a
    // single mutex to prevent interleaved stage execution across concurrent
    // callers. Individual sessions also need `&mut self` for the current `ort` RC.
    pipeline: Mutex<Qwen3TtsPipeline>,
    config: Qwen3TtsConfig,
    vocab: Vocabulary,
}

struct Qwen3TtsPipeline {
    encoder: Session,
    decoder: Session,
    vocoder: Session,
}

/// Shape metadata carried between pipeline stages to avoid flattening tensor ranks.
struct TensorWithShape {
    data: Vec<f32>,
    /// Shape as reported by ORT (i64). Convert to usize when building arrays.
    shape: Vec<i64>,
}

/// Reference conditioning for voice cloning: mel spectrogram + optional transcript tokens.
struct ReferenceConditioning {
    mel: TensorWithShape,
    /// Tokenized reference transcript for prompted cloning. When present, passed
    /// as an additional decoder input alongside the mel conditioning. When absent,
    /// the decoder operates in audio-only (reduced-quality) mode.
    ref_token_ids: Option<Vec<i64>>,
}

impl TensorWithShape {
    fn shape_as_usize(&self) -> Vec<usize> {
        self.shape.iter().map(|&d| d as usize).collect()
    }
}

impl Qwen3TtsRuntime {
    fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<u8>, ModelError> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request requires non-empty text".into(),
            ));
        }

        // Encode reference audio + transcript conditioning if provided.
        let ref_conditioning = match &request.conditioning {
            Some(VoiceConditioning::ReferenceAudio {
                audio_spec,
                pcm,
                reference_text,
            }) => {
                if reference_text.is_none() {
                    tracing::warn!(
                        "qwen3-tts voice cloning without reference_text uses audio-only \
                         conditioning (reduced quality); the official API requires both \
                         ref_audio and ref_text for prompted cloning"
                    );
                }
                let mel = self.encode_reference_audio(audio_spec, pcm)?;
                let ref_token_ids = reference_text
                    .as_deref()
                    .map(|text| self.vocab.tokenize(text));
                Some(ReferenceConditioning {
                    mel,
                    ref_token_ids,
                })
            }
            Some(VoiceConditioning::SpeakerId(_)) => {
                return Err(ModelError::InvalidConfiguration(
                    "qwen3-tts backend uses VoiceConditioning::ReferenceAudio for voice cloning; \
                     SpeakerId is not supported"
                        .into(),
                ));
            }
            None => None,
        };

        // Tokenize text using the model's vocabulary.
        let token_ids = self.vocab.tokenize(text);
        if token_ids.len() <= 2 {
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "tokenize",
                message: "tokenizer produced no content tokens for request text".into(),
            });
        }

        // Lock the entire pipeline for the full encoder → decoder → vocoder flow.
        let mut pipeline = self
            .pipeline
            .lock()
            .map_err(|_| ModelError::Internal("qwen3-tts pipeline mutex poisoned".into()))?;

        let hidden = run_encoder(&mut pipeline.encoder, &token_ids)?;
        let mel = run_decoder(
            &mut pipeline.decoder,
            &hidden,
            ref_conditioning.as_ref(),
            &self.config,
        )?;
        let samples = run_vocoder(&mut pipeline.vocoder, &mel, &self.config)?;

        drop(pipeline); // release lock before PCM encoding

        let pcm = encode_pcm(&samples.data, self.config.audio_spec().encoding);

        if pcm.is_empty() {
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "synthesize",
                message: "qwen3-tts synthesis produced no PCM output".into(),
            });
        }

        Ok(pcm)
    }

    fn encode_reference_audio(
        &self,
        audio_spec: &AudioSpec,
        pcm: &[u8],
    ) -> Result<TensorWithShape, ModelError> {
        let samples = decode_pcm_to_f32(pcm, audio_spec.encoding)?;
        if samples.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "reference audio conditioning requires non-empty PCM data".into(),
            ));
        }

        let mono = downmix_to_mono(&samples, audio_spec.channels);
        let resampled = resample_mono(&mono, audio_spec.sample_rate_hz, self.config.sample_rate);

        let mel_channels = self.config.mel_channels as usize;
        let mel = compute_log_mel_spectrogram(
            &resampled,
            self.config.sample_rate,
            self.config.fft_size,
            self.config.hop_length as usize,
            mel_channels,
        );

        let mel_frames = mel.len() / mel_channels.max(1);
        Ok(TensorWithShape {
            data: mel,
            shape: vec![1, mel_frames as i64, mel_channels as i64],
        })
    }
}

// Pipeline stage functions take `&mut Session` directly (no per-session mutex).

fn run_encoder(encoder: &mut Session, token_ids: &[i64]) -> Result<TensorWithShape, ModelError> {
    let seq_len = token_ids.len();
    let input = Array2::<i64>::from_shape_vec((1, seq_len), token_ids.to_vec()).map_err(|err| {
        ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "build_encoder_input",
            message: err.to_string(),
        }
    })?;

    let input_tensor = Tensor::<i64>::from_array(input).map_err(ort_error)?;
    let inputs = vec![SessionInputValue::from(input_tensor)];

    let outputs = encoder
        .run(inputs.as_slice())
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "run_encoder",
            message: err.to_string(),
        })?;

    let (shape, hidden) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "extract_encoder_output",
            message: err.to_string(),
        })?;

    Ok(TensorWithShape {
        data: hidden.to_vec(),
        shape: shape.to_vec(),
    })
}

fn run_decoder(
    decoder: &mut Session,
    hidden: &TensorWithShape,
    ref_cond: Option<&ReferenceConditioning>,
    _config: &Qwen3TtsConfig,
) -> Result<TensorWithShape, ModelError> {
    let hidden_array = ndarray::ArrayD::<f32>::from_shape_vec(
        ndarray::IxDyn(&hidden.shape_as_usize()),
        hidden.data.clone(),
    )
    .map_err(|err| ModelError::BackendExecution {
        backend: "qwen3-tts",
        operation: "rebuild_encoder_tensor",
        message: format!(
            "failed to reconstruct encoder output with shape {:?}: {err}",
            hidden.shape
        ),
    })?;
    let hidden_tensor = Tensor::<f32>::from_array(hidden_array).map_err(ort_error)?;

    let mut inputs = vec![SessionInputValue::from(hidden_tensor)];

    if let Some(cond) = ref_cond {
        // Add mel conditioning.
        let mel_array = ndarray::ArrayD::<f32>::from_shape_vec(
            ndarray::IxDyn(&cond.mel.shape_as_usize()),
            cond.mel.data.clone(),
        )
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "rebuild_ref_mel_tensor",
            message: format!(
                "failed to reconstruct reference mel with shape {:?}: {err}",
                cond.mel.shape
            ),
        })?;
        let mel_tensor = Tensor::<f32>::from_array(mel_array).map_err(ort_error)?;
        inputs.push(SessionInputValue::from(mel_tensor));

        // Add tokenized reference transcript for prompted cloning.
        if let Some(ref_ids) = &cond.ref_token_ids {
            let ref_len = ref_ids.len();
            let ref_input =
                Array2::<i64>::from_shape_vec((1, ref_len), ref_ids.clone()).map_err(|err| {
                    ModelError::BackendExecution {
                        backend: "qwen3-tts",
                        operation: "build_ref_text_input",
                        message: err.to_string(),
                    }
                })?;
            let ref_tensor = Tensor::<i64>::from_array(ref_input).map_err(ort_error)?;
            inputs.push(SessionInputValue::from(ref_tensor));
        }
    }

    let outputs = decoder
        .run(inputs.as_slice())
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "run_decoder",
            message: err.to_string(),
        })?;

    let (shape, mel_out) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "extract_decoder_output",
            message: err.to_string(),
        })?;

    Ok(TensorWithShape {
        data: mel_out.to_vec(),
        shape: shape.to_vec(),
    })
}

fn run_vocoder(
    vocoder: &mut Session,
    mel: &TensorWithShape,
    _config: &Qwen3TtsConfig,
) -> Result<TensorWithShape, ModelError> {
    let mel_array = ndarray::ArrayD::<f32>::from_shape_vec(
        ndarray::IxDyn(&mel.shape_as_usize()),
        mel.data.clone(),
    )
    .map_err(|err| ModelError::BackendExecution {
        backend: "qwen3-tts",
        operation: "rebuild_decoder_tensor",
        message: format!(
            "failed to reconstruct decoder mel output with shape {:?}: {err}",
            mel.shape
        ),
    })?;

    let mel_tensor = Tensor::<f32>::from_array(mel_array).map_err(ort_error)?;
    let inputs = vec![SessionInputValue::from(mel_tensor)];

    let outputs = vocoder
        .run(inputs.as_slice())
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "run_vocoder",
            message: err.to_string(),
        })?;

    let (shape, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|err| ModelError::BackendExecution {
            backend: "qwen3-tts",
            operation: "extract_vocoder_output",
            message: err.to_string(),
        })?;

    Ok(TensorWithShape {
        data: audio.to_vec(),
        shape: shape.to_vec(),
    })
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
    let pipeline = Qwen3TtsPipeline {
        encoder: build_session("qwen3-tts", &artifacts.encoder)?,
        decoder: build_session("qwen3-tts", &artifacts.decoder)?,
        vocoder: build_session("qwen3-tts", &artifacts.vocoder)?,
    };
    Ok(Qwen3TtsRuntime {
        pipeline: Mutex::new(pipeline),
        config: Qwen3TtsConfig::from_path(&artifacts.config)?,
        vocab: Vocabulary::from_path(&artifacts.vocab)?,
    })
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
        use crate::common::{decode_pcm_to_f32, encode_pcm};

        let samples = vec![0.5_f32, -0.5, 0.0, 1.0, -1.0];
        let encoded = encode_pcm(&samples, PcmEncoding::S16Le);
        assert_eq!(encoded.len(), samples.len() * 2);

        let decoded = decode_pcm_to_f32(&encoded, PcmEncoding::S16Le)
            .expect("round-trip should succeed");
        for (orig, dec) in samples.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.001, "orig={orig}, decoded={dec}");
        }
    }
}
