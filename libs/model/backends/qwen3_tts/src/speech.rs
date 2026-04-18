use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{
    AudioBuf, CloneReference, Mono, SpeechStream as TypedSpeechStream, SpeechSynthesizer,
    SynthesisRequest, VoiceCloneSynthesizer,
};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationSupport, ResolvedCheckpoint, StartOptions,
    UnsupportedChat, UnsupportedCompletion, UnsupportedEmbeddings,
};
use motlie_model_ort::build_session;
use ndarray::Array2;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::common::{
    Qwen3TtsArtifactPaths, Qwen3TtsConfig, RuntimeMetricState, Vocabulary,
    compute_log_mel_spectrogram, configure_artifact_policy, lock_metrics, observe_latency,
    observe_memory, resample_mono, resolve_onnx_artifacts,
};

const QWEN3_TTS_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const OUTPUT_SAMPLE_RATE_HZ: u32 = 24_000;
const REFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;

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
    type Handle = Qwen3TtsHandle;

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
    ) -> Result<Self::Handle, ModelError> {
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
    type Handle = Qwen3TtsHandle;

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

impl Qwen3TtsSpeechBundle {
    pub async fn start_typed(&self, options: StartOptions) -> Result<Qwen3TtsHandle, ModelError> {
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

pub struct Qwen3TtsHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<Qwen3TtsRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

impl Qwen3TtsHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }
}

#[derive(Clone, Debug, Default)]
struct SpeechMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for Qwen3TtsHandle {
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

fn new_speech_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<Qwen3TtsRuntime>,
) -> Qwen3TtsHandle {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "qwen3-tts-start");
        observe_memory(&mut state.runtime);
    }

    Qwen3TtsHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization: None,
        },
        runtime,
        metrics,
    }
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
#[derive(Clone)]
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
    fn synthesize(&self, request: &SynthesisRequest) -> Result<Vec<f32>, ModelError> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request requires non-empty text".into(),
            ));
        }
        self.synthesize_inner(text, None)
    }

    fn synthesize_with_reference(
        &self,
        request: &SynthesisRequest,
        reference: &CloneReference<REFERENCE_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Vec<f32>, ModelError> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request requires non-empty text".into(),
            ));
        }

        if reference.audio.samples().is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "reference audio conditioning requires non-empty audio".into(),
            ));
        }

        if reference.transcript.is_none() {
            tracing::warn!(
                "qwen3-tts voice cloning without transcript uses audio-only conditioning (reduced quality)"
            );
        }

        let mel = self.encode_reference_audio(reference.audio.samples())?;
        let ref_token_ids = reference
            .transcript
            .as_deref()
            .map(|text| self.vocab.tokenize(text));
        let ref_conditioning = Some(ReferenceConditioning { mel, ref_token_ids });
        self.synthesize_inner(text, ref_conditioning)
    }

    fn synthesize_inner(
        &self,
        text: &str,
        ref_conditioning: Option<ReferenceConditioning>,
    ) -> Result<Vec<f32>, ModelError> {
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

        drop(pipeline);

        if samples.data.is_empty() {
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts",
                operation: "synthesize",
                message: "qwen3-tts synthesis produced no PCM output".into(),
            });
        }

        Ok(samples.data)
    }

    fn encode_reference_audio(&self, samples: &[f32]) -> Result<TensorWithShape, ModelError> {
        let resampled = resample_mono(samples, REFERENCE_SAMPLE_RATE_HZ, self.config.sample_rate);

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

    let (shape, hidden) =
        outputs[0]
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

    let (shape, mel_out) =
        outputs[0]
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

    let (shape, audio) =
        outputs[0]
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
pub struct Qwen3TtsSpeechStream {
    samples: Vec<f32>,
    offset: usize,
    chunk_len_samples: usize,
}

impl Qwen3TtsSpeechStream {
    fn new(samples: Vec<f32>) -> Self {
        let chunk_len_samples =
            ((OUTPUT_SAMPLE_RATE_HZ as u64 * OUTPUT_CHUNK_DURATION_MS as u64) / 1000) as usize;

        Self {
            samples,
            offset: 0,
            chunk_len_samples: chunk_len_samples.max(1),
        }
    }
}

impl Qwen3TtsSpeechStream {
    async fn next_audio_chunk(
        &mut self,
    ) -> Result<Option<AudioBuf<f32, OUTPUT_SAMPLE_RATE_HZ, Mono>>, ModelError> {
        if self.offset >= self.samples.len() {
            return Ok(None);
        }

        let end = (self.offset + self.chunk_len_samples).min(self.samples.len());
        let chunk = AudioBuf::new(self.samples[self.offset..end].to_vec());
        self.offset = end;

        Ok(Some(chunk))
    }

    async fn finish_stream(self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl SpeechSynthesizer for Qwen3TtsHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<f32, OUTPUT_SAMPLE_RATE_HZ, Mono>;
    type Stream = Qwen3TtsSpeechStream;

    async fn synthesize(&self, request: Self::Request) -> Result<Self::Stream, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        let started_at = Instant::now();
        let pcm = runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&metrics, "qwen3-tts-typed-synthesize");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(Qwen3TtsSpeechStream::new(pcm))
    }
}

impl VoiceCloneSynthesizer<REFERENCE_SAMPLE_RATE_HZ, Mono> for Qwen3TtsHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<f32, OUTPUT_SAMPLE_RATE_HZ, Mono>;
    type Stream = Qwen3TtsSpeechStream;

    async fn synthesize_with_reference(
        &self,
        request: Self::Request,
        reference: CloneReference<REFERENCE_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Self::Stream, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        let started_at = Instant::now();
        let pcm = runtime.synthesize_with_reference(&request, &reference)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&metrics, "qwen3-tts-typed-clone");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(Qwen3TtsSpeechStream::new(pcm))
    }
}

impl TypedSpeechStream for Qwen3TtsSpeechStream {
    type Chunk = AudioBuf<f32, OUTPUT_SAMPLE_RATE_HZ, Mono>;

    async fn next_chunk(&mut self) -> Result<Option<Self::Chunk>, ModelError> {
        self.next_audio_chunk().await
    }

    async fn finish(self) -> Result<(), ModelError> {
        self.finish_stream().await
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
    async fn stream_emits_chunks_and_finishes() {
        let pcm = vec![0.25_f32; 10_000];
        let mut stream = Qwen3TtsSpeechStream::new(pcm);

        let mut total = 0usize;
        while let Some(chunk) = TypedSpeechStream::next_chunk(&mut stream)
            .await
            .expect("should succeed")
        {
            total += chunk.samples().len();
        }

        assert_eq!(total, 10_000);
        assert!(
            TypedSpeechStream::next_chunk(&mut stream)
                .await
                .expect("should stay exhausted")
                .is_none()
        );
    }

    #[test]
    fn reference_conditioning_with_text_carries_token_ids() {
        use crate::common::Vocabulary;

        let vocab =
            Vocabulary::from_entries(&[("<unk>", 0), ("<bos>", 1), ("<eos>", 2), ("hi", 10)])
                .expect("test vocab");

        let mel = TensorWithShape {
            data: vec![0.1, 0.2, 0.3],
            shape: vec![1, 1, 3],
        };

        // With reference_text
        let ref_text = Some("hi");
        let ref_ids = ref_text.map(|t| vocab.tokenize(t));
        let cond = ReferenceConditioning {
            mel: mel.clone(),
            ref_token_ids: ref_ids.clone(),
        };
        assert!(cond.ref_token_ids.is_some());
        assert_eq!(cond.ref_token_ids.as_ref().unwrap(), &vec![1, 10, 2]);

        // Without reference_text
        let cond_no_text = ReferenceConditioning {
            mel,
            ref_token_ids: None,
        };
        assert!(cond_no_text.ref_token_ids.is_none());
    }
}
