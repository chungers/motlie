use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use espeak_rs::text_to_phonemes;
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, PcmChunk,
    QuantizationSupport, ResolvedCheckpoint, SpeechModel, SpeechParams, SpeechRequest,
    SpeechStream, StartOptions, TranscriptionModel, VoiceConditioning,
};
use motlie_model_ort::build_session;
use ndarray::{Array1, Array2};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_onnx_artifacts, PiperArtifactPaths, PiperConfig, RuntimeMetricState,
};

const PIPER_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;

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
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
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

struct PiperHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<PiperRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct SpeechMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for PiperHandle {
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
impl SpeechModel for PiperHandle {
    async fn open_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Box<dyn SpeechStream>, ModelError> {
        let started_at = Instant::now();
        let pcm = self.runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&self.metrics, "piper-open-stream");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(Box::new(PiperSpeechStream::new(
            self.runtime.config.audio_spec.clone(),
            pcm,
        )?))
    }
}

fn new_speech_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<PiperRuntime>,
) -> Box<dyn BundleHandle> {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "piper-start");
        observe_memory(&mut state.runtime);
    }

    Box::new(PiperHandle {
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

struct PiperRuntime {
    // The current `ort` RC used by Motlie exposes `Session::run(&mut self, ...)`,
    // so shared bundle handles must serialize access around the loaded session.
    session: Mutex<Session>,
    config: PiperConfig,
}

impl PiperRuntime {
    fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<u8>, ModelError> {
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

        let speaker_id = match &request.conditioning {
            None => None,
            Some(VoiceConditioning::SpeakerId(speaker_id)) => {
                if self.config.num_speakers <= 1 {
                    return Err(ModelError::InvalidConfiguration(
                        "piper voice does not support speaker selection".into(),
                    ));
                }
                if !self.config.supports_speaker(*speaker_id) {
                    return Err(ModelError::InvalidConfiguration(format!(
                        "speaker id `{speaker_id}` is not defined by this piper voice"
                    )));
                }
                Some(*speaker_id as i64)
            }
            Some(VoiceConditioning::ReferenceAudio { .. }) => {
                return Err(ModelError::InvalidConfiguration(
                    "piper backend does not support reference-audio conditioning".into(),
                ));
            }
        };

        let scales = synthesis_scales(&self.config, &request.params, speaker_id)?;
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
            append_s16le_pcm(&mut pcm, &samples);
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
struct PiperSpeechStream {
    audio_spec: motlie_model::AudioSpec,
    pcm: Vec<u8>,
    offset: usize,
    next_sequence: u64,
    chunk_len_bytes: usize,
}

impl PiperSpeechStream {
    fn new(audio_spec: motlie_model::AudioSpec, pcm: Vec<u8>) -> Result<Self, ModelError> {
        let bytes_per_sample = match audio_spec.encoding {
            motlie_model::PcmEncoding::S16Le => 2,
            motlie_model::PcmEncoding::F32Le => 4,
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
impl SpeechStream for PiperSpeechStream {
    fn audio_spec(&self) -> &motlie_model::AudioSpec {
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

#[derive(Clone, Copy)]
struct PiperSynthesisScales {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: Option<i64>,
}

fn load_runtime(artifacts: &PiperArtifactPaths) -> Result<PiperRuntime, ModelError> {
    Ok(PiperRuntime {
        session: Mutex::new(build_session("piper", &artifacts.model)?),
        config: PiperConfig::from_path(&artifacts.config)?,
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

fn append_s16le_pcm(target: &mut Vec<u8>, samples: &[f32]) {
    target.reserve(samples.len() * 2);
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let as_i16 = (clamped * i16::MAX as f32) as i16;
        target.extend_from_slice(&as_i16.to_le_bytes());
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
    use motlie_model::{PcmEncoding, SpeechParams};

    #[tokio::test]
    async fn stream_emits_monotonic_chunks_and_finishes_once() {
        let audio_spec = motlie_model::AudioSpec {
            sample_rate_hz: 16_000,
            channels: 1,
            encoding: PcmEncoding::S16Le,
        };
        let pcm = vec![1_u8; 10_000];
        let mut stream = PiperSpeechStream::new(audio_spec, pcm).expect("stream should build");

        let mut previous_sequence = None;
        let mut saw_final = false;
        while let Some(chunk) = stream.next_chunk().await.expect("chunking should succeed") {
            if let Some(previous) = previous_sequence {
                assert!(chunk.sequence > previous);
            }
            previous_sequence = Some(chunk.sequence);
            saw_final = chunk.end_of_stream;
        }

        assert!(saw_final);
        assert!(stream
            .next_chunk()
            .await
            .expect("stream should stay exhausted")
            .is_none());
    }

    #[test]
    fn speaking_rate_maps_to_inverse_length_scale() {
        let config = PiperConfig {
            audio_spec: motlie_model::AudioSpec {
                sample_rate_hz: 22_050,
                channels: 1,
                encoding: PcmEncoding::S16Le,
            },
            espeak_voice: "en-us".into(),
            num_speakers: 1,
            speaker_id_map: Default::default(),
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
            audio_spec: motlie_model::AudioSpec {
                sample_rate_hz: 22_050,
                channels: 1,
                encoding: PcmEncoding::S16Le,
            },
            espeak_voice: "en-us".into(),
            num_speakers: 1,
            speaker_id_map: Default::default(),
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
