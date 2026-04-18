use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use espeak_rs::text_to_phonemes;
use motlie_model::typed::{
    AudioBuf, Mono, SpeechStream as TypedSpeechStream, SpeechSynthesizer, SynthesisRequest,
};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationSupport, ResolvedCheckpoint, SpeechParams,
    StartOptions, UnsupportedChat, UnsupportedCompletion, UnsupportedEmbeddings,
};
use motlie_model_ort::build_session;
use ndarray::{Array1, Array2};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::common::{
    PiperArtifactPaths, PiperConfig, RuntimeMetricState, configure_artifact_policy, lock_metrics,
    observe_latency, observe_memory, resolve_onnx_artifacts,
};

const PIPER_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const PIPER_SAMPLE_RATE_HZ: u32 = 22_050;

#[derive(Clone, Debug)]
pub struct PiperSpeechSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_filename: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl PiperSpeechSpec {
    pub fn en_us_ljspeech_medium() -> Self {
        Self {
            id: BundleId::new("piper_en_us_ljspeech_medium"),
            display_name: "Piper en_US ljspeech medium",
            model_filename: "en_US-ljspeech-medium.onnx",
            capabilities: Capabilities::speech_stream_only(),
            quantization: QuantizationSupport::none(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PiperSpeechAdapter {
    spec: PiperSpeechSpec,
}

impl PiperSpeechAdapter {
    pub fn en_us_ljspeech_medium() -> Self {
        Self {
            spec: PiperSpeechSpec::en_us_ljspeech_medium(),
        }
    }
}

#[async_trait]
impl BackendAdapter for PiperSpeechAdapter {
    type Handle = PiperHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &PIPER_FORMATS
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

        let artifacts = resolve_onnx_artifacts(checkpoint, self.spec.model_filename)?;
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

#[derive(Clone, Debug)]
pub struct PiperSpeechBundle {
    metadata: BundleMetadata,
    spec: PiperSpeechSpec,
}

impl PiperSpeechBundle {
    pub fn new(spec: PiperSpeechSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id.clone(),
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities.clone(),
                quantization: spec.quantization.clone(),
            },
            spec,
        }
    }
}

#[async_trait]
impl ModelBundle for PiperSpeechBundle {
    type Handle = PiperHandle;

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

impl PiperSpeechBundle {
    pub async fn start_typed(&self, options: StartOptions) -> Result<PiperHandle, ModelError> {
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let artifacts = if let Some(policy) = options.artifact_policy {
            configure_artifact_policy(self.spec.model_filename, policy)?
        } else {
            configure_artifact_policy(
                self.spec.model_filename,
                motlie_model::ArtifactPolicy::LocalOnly {
                    root: PathBuf::from("."),
                },
            )?
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

pub struct PiperHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<PiperRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

impl PiperHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }
}

#[derive(Clone, Debug, Default)]
struct SpeechMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for PiperHandle {
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
        let metrics = lock_metrics(&self.metrics, "piper-metric-snapshot").clone();
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
    runtime: Arc<PiperRuntime>,
) -> PiperHandle {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "piper-start");
        observe_memory(&mut state.runtime);
    }

    PiperHandle {
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

struct PiperRuntime {
    // The current `ort` RC used by Motlie exposes `Session::run(&mut self, ...)`,
    // so shared bundle handles must serialize access around the loaded session.
    session: Mutex<Session>,
    config: PiperConfig,
}

impl PiperRuntime {
    fn synthesize(&self, request: &SynthesisRequest) -> Result<Vec<i16>, ModelError> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request requires non-empty text".into(),
            ));
        }
        if request.params.seed.is_some() {
            return Err(ModelError::InvalidConfiguration(
                "piper backend does not support `SpeechParams.seed`".into(),
            ));
        }

        let scales = synthesis_scales(&self.config, &request.params, None)?;
        let phoneme_batches = text_to_phonemes(text, &self.config.espeak_voice, None, true, false)
            .map_err(|err| ModelError::BackendExecution {
                backend: "piper",
                operation: "phonemize_text",
                message: err.to_string(),
            })?;
        if phoneme_batches.is_empty() {
            return Err(ModelError::BackendExecution {
                backend: "piper",
                operation: "phonemize_text",
                message: "phonemizer produced no output for request text".into(),
            });
        }

        let mut pcm = Vec::new();
        for phonemes in phoneme_batches {
            let input_ids = phonemes_to_input_ids(&self.config, &phonemes);
            if input_ids.len() <= 2 {
                continue;
            }
            let samples = run_inference(&self.session, &input_ids, &scales)?;
            append_i16_samples(&mut pcm, &samples);
        }

        if pcm.is_empty() {
            return Err(ModelError::BackendExecution {
                backend: "piper",
                operation: "synthesize",
                message: "piper synthesis produced no PCM output".into(),
            });
        }

        Ok(pcm)
    }
}

/// Piper streams expose already-synthesized PCM in monotonic chunks.
///
/// Piper is a non-autoregressive VITS-style model in this slice, so
/// `open_stream()` performs the full synthesis up front and `next_chunk()`
/// subsequently yields buffered PCM for sink adapters.
pub struct PiperSpeechStream {
    pcm: Vec<i16>,
    offset: usize,
    chunk_len_samples: usize,
}

impl PiperSpeechStream {
    fn new(pcm: Vec<i16>) -> Self {
        let frames_per_chunk =
            ((PIPER_SAMPLE_RATE_HZ as u64 * OUTPUT_CHUNK_DURATION_MS as u64) / 1000) as usize;

        Self {
            pcm,
            offset: 0,
            chunk_len_samples: frames_per_chunk.max(1),
        }
    }
}

impl PiperSpeechStream {
    async fn next_audio_chunk(
        &mut self,
    ) -> Result<Option<AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>>, ModelError> {
        if self.offset >= self.pcm.len() {
            return Ok(None);
        }

        let end = (self.offset + self.chunk_len_samples).min(self.pcm.len());
        let chunk = AudioBuf::new(self.pcm[self.offset..end].to_vec());
        self.offset = end;

        Ok(Some(chunk))
    }

    async fn finish_stream(self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl SpeechSynthesizer for PiperHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<i16, 22_050, Mono>;
    type Stream = PiperSpeechStream;

    async fn synthesize(&self, request: Self::Request) -> Result<Self::Stream, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        let started_at = Instant::now();
        let pcm = runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&metrics, "piper-typed-synthesize");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(PiperSpeechStream::new(pcm))
    }
}

impl TypedSpeechStream for PiperSpeechStream {
    type Chunk = AudioBuf<i16, 22_050, Mono>;

    async fn next_chunk(&mut self) -> Result<Option<Self::Chunk>, ModelError> {
        self.next_audio_chunk().await
    }

    async fn finish(self) -> Result<(), ModelError> {
        self.finish_stream().await
    }
}

#[derive(Clone, Copy)]
struct PiperSynthesisScales {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: Option<i64>,
}

fn load_runtime(artifacts: &PiperArtifactPaths) -> Result<PiperRuntime, ModelError> {
    let config = PiperConfig::from_path(&artifacts.config)?;
    if config.sample_rate_hz != PIPER_SAMPLE_RATE_HZ {
        return Err(ModelError::InvalidConfiguration(format!(
            "curated Piper bundle expects {} Hz output, config declares {} Hz",
            PIPER_SAMPLE_RATE_HZ, config.sample_rate_hz
        )));
    }

    Ok(PiperRuntime {
        session: Mutex::new(build_session("piper", &artifacts.model)?),
        config,
    })
}

fn synthesis_scales(
    config: &PiperConfig,
    params: &SpeechParams,
    speaker_id: Option<i64>,
) -> Result<PiperSynthesisScales, ModelError> {
    let length_scale = match params.speaking_rate {
        Some(rate) if !rate.is_finite() || rate <= 0.0 => {
            return Err(ModelError::InvalidConfiguration(format!(
                "speech speaking_rate must be positive and finite, got {rate}"
            )));
        }
        Some(rate) => 1.0 / rate,
        None => config.default_length_scale,
    };

    Ok(PiperSynthesisScales {
        noise_scale: config.default_noise_scale,
        length_scale,
        noise_w: config.default_noise_w,
        speaker_id,
    })
}

fn phonemes_to_input_ids(config: &PiperConfig, phonemes: &str) -> Vec<i64> {
    let mut ids = Vec::with_capacity((phonemes.chars().count() + 1) * 2);
    ids.push(config.bos_id);

    for phoneme in phonemes.chars() {
        if let Some(mapped_ids) = config.phoneme_id_map.get(&phoneme.to_string()) {
            for id in mapped_ids {
                ids.push(*id);
            }
            ids.push(config.pad_id);
        }
    }

    ids.push(config.eos_id);
    ids
}

fn run_inference(
    session: &Mutex<Session>,
    input_ids: &[i64],
    scales: &PiperSynthesisScales,
) -> Result<Vec<f32>, ModelError> {
    let input_len = input_ids.len();
    let phoneme_inputs = Array2::<i64>::from_shape_vec((1, input_len), input_ids.to_vec())
        .map_err(|err| ModelError::BackendExecution {
            backend: "piper",
            operation: "build_phoneme_tensor",
            message: err.to_string(),
        })?;
    let input_lengths = Array1::<i64>::from_vec(vec![input_len as i64]);
    let scale_values = Array1::<f32>::from_vec(vec![
        scales.noise_scale,
        scales.length_scale,
        scales.noise_w,
    ]);

    let phoneme_tensor = Tensor::<i64>::from_array(phoneme_inputs).map_err(ort_tensor_error)?;
    let length_tensor = Tensor::<i64>::from_array(input_lengths).map_err(ort_tensor_error)?;
    let scale_tensor = Tensor::<f32>::from_array(scale_values).map_err(ort_tensor_error)?;

    let mut inputs = vec![
        SessionInputValue::from(phoneme_tensor),
        SessionInputValue::from(length_tensor),
        SessionInputValue::from(scale_tensor),
    ];
    if let Some(speaker_id) = scales.speaker_id {
        let speaker_tensor = Tensor::<i64>::from_array(Array1::<i64>::from_vec(vec![speaker_id]))
            .map_err(ort_tensor_error)?;
        inputs.push(SessionInputValue::from(speaker_tensor));
    }

    let mut session = session
        .lock()
        .map_err(|_| ModelError::Internal("piper inference session mutex was poisoned".into()))?;
    let outputs = session
        .run(inputs.as_slice())
        .map_err(|err| ModelError::BackendExecution {
            backend: "piper",
            operation: "run_inference",
            message: err.to_string(),
        })?;

    let (_, samples) =
        outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|err| ModelError::BackendExecution {
                backend: "piper",
                operation: "extract_audio_tensor",
                message: err.to_string(),
            })?;

    Ok(samples.to_vec())
}

fn append_i16_samples(target: &mut Vec<i16>, samples: &[f32]) {
    target.reserve(samples.len());
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let as_i16 = (clamped * i16::MAX as f32) as i16;
        target.push(as_i16);
    }
}

fn ort_tensor_error(err: ort::Error) -> ModelError {
    ModelError::BackendExecution {
        backend: "piper",
        operation: "build_tensor",
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::SpeechParams;

    #[tokio::test]
    async fn stream_emits_chunks_and_finishes_once() {
        let pcm = vec![1_i16; 10_000];
        let mut stream = PiperSpeechStream::new(pcm);

        let mut total = 0usize;
        while let Some(chunk) = stream.next_chunk().await.expect("chunking should succeed") {
            total += chunk.samples().len();
        }

        assert_eq!(total, 10_000);
        assert!(
            stream
                .next_chunk()
                .await
                .expect("stream should stay exhausted")
                .is_none()
        );
    }

    #[test]
    fn speaking_rate_maps_to_inverse_length_scale() {
        let config = PiperConfig {
            sample_rate_hz: 22_050,
            espeak_voice: "en-us".into(),
            phoneme_id_map: Default::default(),
            default_noise_scale: 0.667,
            default_length_scale: 1.0,
            default_noise_w: 0.8,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        };

        let scales = synthesis_scales(
            &config,
            &SpeechParams {
                speaking_rate: Some(2.0),
                seed: None,
            },
            None,
        )
        .expect("valid speaking rate should map");

        assert_eq!(scales.length_scale, 0.5);
    }

    #[test]
    fn phoneme_ids_wrap_with_bos_pad_and_eos() {
        let mut phoneme_id_map = std::collections::HashMap::new();
        phoneme_id_map.insert("^".into(), vec![1]);
        phoneme_id_map.insert("_".into(), vec![0]);
        phoneme_id_map.insert("$".into(), vec![2]);
        phoneme_id_map.insert("a".into(), vec![10]);
        phoneme_id_map.insert("b".into(), vec![11]);

        let config = PiperConfig {
            sample_rate_hz: 22_050,
            espeak_voice: "en-us".into(),
            phoneme_id_map,
            default_noise_scale: 0.667,
            default_length_scale: 1.0,
            default_noise_w: 0.8,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        };

        let ids = phonemes_to_input_ids(&config, "ab");
        assert_eq!(ids, vec![1, 10, 0, 11, 0, 2]);
    }
}
