use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use fish_speech_core::config::{WhichFishVersion, WhichLM, WhichModel};
use fish_speech_core::lm::generate::generate_blocking_with_hidden;
use fish_speech_core::text::{clean::preprocess_text, prompt::PromptEncoder};
use fish_speech_server::state::AppState;
use fish_speech_server::utils::load::{load_codec, load_lm, Args};
use motlie_model::{
    ArtifactPolicy, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata,
    Capabilities, CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, PcmChunk,
    PcmEncoding, QuantizationSupport, ResolvedCheckpoint, SpeechModel, SpeechRequest, SpeechStream,
    StartOptions, TranscriptionModel, VoiceConditioning,
};
use tempfile::TempDir;
use tokio::sync::Mutex;

const FISH_SPEECH_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Safetensors];
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const DEFAULT_SAMPLE_TEMP: f64 = 0.7;
const DEFAULT_TOP_P: f64 = 0.8;
const EMBEDDED_DEFAULT_VOICE_NPY: &[u8] = include_bytes!("../assets/default.npy");
const EMBEDDED_DEFAULT_VOICE_INDEX: &str = include_str!("../assets/index.json");

const REQUIRED_ARTIFACTS: [&str; 6] = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors",
];

#[derive(Clone, Debug)]
pub struct FishSpeechSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl FishSpeechSpec {
    pub fn fish_speech_1_5() -> Self {
        Self {
            id: BundleId::new("fish_speech_1_5"),
            display_name: "Fish Speech 1.5",
            capabilities: Capabilities::speech_stream_only(),
            quantization: QuantizationSupport::none(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FishSpeechAdapter {
    spec: FishSpeechSpec,
}

impl FishSpeechAdapter {
    pub fn fish_speech_1_5() -> Self {
        Self {
            spec: FishSpeechSpec::fish_speech_1_5(),
        }
    }
}

#[async_trait]
impl BackendAdapter for FishSpeechAdapter {
    fn supported_formats(&self) -> &[CheckpointFormat] {
        &FISH_SPEECH_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::FishSpeech
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

        let checkpoint_dir = resolve_checkpoint_dir(checkpoint)?;
        let runtime = Arc::new(FishSpeechRuntime::load(checkpoint_dir)?);

        Ok(Box::new(FishSpeechHandle {
            descriptor: LoadedBundleDescriptor {
                id: identity.id.clone(),
                display_name: identity.display_name.clone(),
                capabilities: self.spec.capabilities.clone(),
                quantization: self.spec.quantization.clone(),
                resolved_quantization: None,
            },
            runtime,
            metrics: Arc::new(Mutex::new(FishSpeechMetrics::default())),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct FishSpeechBundle {
    metadata: BundleMetadata,
    spec: FishSpeechSpec,
}

impl FishSpeechBundle {
    pub fn new(spec: FishSpeechSpec) -> Self {
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
impl ModelBundle for FishSpeechBundle {
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
        self.spec
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let root = match options.artifact_policy {
            Some(ArtifactPolicy::LocalOnly { root }) => root,
            Some(ArtifactPolicy::AllowFetch { root }) => root.unwrap_or_else(|| PathBuf::from(".")),
            None => PathBuf::from("."),
        };
        let checkpoint = ResolvedCheckpoint {
            checkpoint: motlie_model::ModelCheckpoint {
                format: CheckpointFormat::Safetensors,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "jkeisling/fish-speech-1.5",
                },
                include: Vec::new(),
                quantization: None,
            },
            path: root,
        };

        let checkpoint_dir = resolve_checkpoint_dir(&checkpoint)?;
        let runtime = Arc::new(FishSpeechRuntime::load(checkpoint_dir)?);

        Ok(Box::new(FishSpeechHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
                quantization: self.metadata.quantization.clone(),
                resolved_quantization: None,
            },
            runtime,
            metrics: Arc::new(Mutex::new(FishSpeechMetrics::default())),
        }))
    }
}

struct FishSpeechHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<FishSpeechRuntime>,
    metrics: Arc<Mutex<FishSpeechMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct FishSpeechMetrics {
    request_count: u64,
    last_latency_msec: Option<u128>,
    max_latency_msec: Option<u128>,
}

#[async_trait]
impl BundleHandle for FishSpeechHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = self.metrics.blocking_lock().clone();
        Some(ModelMetricSnapshot {
            runtime: Some(motlie_model::RuntimeMetrics {
                resident_memory: None,
                peak_resident_memory: None,
                request_count: Some(metrics.request_count),
                last_latency: metrics
                    .last_latency_msec
                    .map(|value| motlie_model::Milliseconds(value as u64)),
                max_latency: metrics
                    .max_latency_msec
                    .map(|value| motlie_model::Milliseconds(value as u64)),
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
impl SpeechModel for FishSpeechHandle {
    async fn open_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Box<dyn SpeechStream>, ModelError> {
        validate_request(&request)?;

        let started_at = Instant::now();
        let pcm = self.runtime.synthesize(&request).await?;
        let elapsed = started_at.elapsed().as_millis();

        {
            let mut metrics = self.metrics.lock().await;
            metrics.request_count += 1;
            metrics.last_latency_msec = Some(elapsed);
            metrics.max_latency_msec = Some(
                metrics
                    .max_latency_msec
                    .map_or(elapsed, |current| current.max(elapsed)),
            );
        }

        Ok(Box::new(FishSpeechStream::new(
            motlie_model::AudioSpec {
                sample_rate_hz: self.runtime.sample_rate,
                channels: 1,
                encoding: PcmEncoding::S16Le,
            },
            pcm,
        )?))
    }
}

struct FishSpeechRuntime {
    state: Arc<AppState>,
    sample_rate: u32,
    _embedded_voice_dir: Option<TempDir>,
}

impl FishSpeechRuntime {
    fn load(checkpoint_dir: PathBuf) -> Result<Self, ModelError> {
        let (device, dtype) = select_device_and_dtype()?;
        let voice_dir = prepare_voice_dir()?;
        let args = Args {
            checkpoint: Some(checkpoint_dir.clone()),
            fish_version: WhichModel::Fish1_5,
            voice_dir: voice_dir
                .as_ref()
                .map(|dir| dir.path().to_path_buf())
                .unwrap_or_else(|| checkpoint_dir.join("voices")),
            port: 3000,
            temp: DEFAULT_SAMPLE_TEMP,
            top_p: DEFAULT_TOP_P,
        };

        let lm_state = load_lm(&args, Some(checkpoint_dir), dtype, &device).map_err(|err| {
            ModelError::BackendInitialization {
                backend: "fish-speech",
                message: format!("failed to load language model: {err}"),
            }
        })?;
        let (codec, sample_rate) =
            load_codec(&args, dtype, &device, lm_state.config.num_codebooks).map_err(|err| {
                ModelError::BackendInitialization {
                    backend: "fish-speech",
                    message: format!("failed to load codec: {err}"),
                }
            })?;

        Ok(Self {
            state: Arc::new(AppState {
                lm: Arc::new(lm_state),
                codec: Arc::new(codec),
                device,
                model_type: WhichModel::Fish1_5,
                sample_rate,
            }),
            sample_rate,
            _embedded_voice_dir: voice_dir,
        })
    }

    async fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<u8>, ModelError> {
        let mut voice_embedding = Some(self.state.lm.default_voice.as_ref().clone());
        match request.conditioning.as_ref() {
            None => {}
            Some(VoiceConditioning::SpeakerId(0)) => {}
            Some(VoiceConditioning::SpeakerId(other)) => {
                return Err(ModelError::InvalidConfiguration(format!(
                    "fish-speech backend only exposes the embedded default voice in this vertical slice; speaker_id `{other}` is unsupported"
                )));
            }
            Some(VoiceConditioning::ReferenceAudio { .. }) => {
                return Err(ModelError::InvalidConfiguration(
                    "fish-speech backend does not implement reference-audio cloning in the v0.3 vertical slice".into(),
                ));
            }
        }

        let chunks = preprocess_text(&request.text);
        if chunks.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request text must not be empty".into(),
            ));
        }

        let num_codebooks = self.state.lm.config.num_codebooks;
        let prompt_encoder = PromptEncoder::new(
            &self.state.lm.tokenizer,
            &self.state.device,
            num_codebooks,
            self.state.lm.model_type,
        );
        let prompts = prompt_encoder
            .encode_sequence(
                chunks,
                Some("Speak out the provided text.".to_string()),
                voice_embedding.take(),
                true,
            )
            .map_err(|err| backend_execution_error("encode_sequence", err))?;

        let audio = generate_pcm(self.state.clone(), prompts).await?;
        pcm_f32_to_s16le(&audio)
    }
}

fn resolve_checkpoint_dir(checkpoint: &ResolvedCheckpoint) -> Result<PathBuf, ModelError> {
    let path = checkpoint.path.clone();
    if !path.is_dir() {
        return Err(ModelError::InvalidConfiguration(format!(
            "fish-speech checkpoint path must be a directory containing safetensor artifacts, got `{}`",
            path.display()
        )));
    }

    for filename in REQUIRED_ARTIFACTS {
        let artifact = path.join(filename);
        if !artifact.exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "fish-speech checkpoint directory `{}` is missing required artifact `{filename}`",
                path.display()
            )));
        }
    }

    Ok(path)
}

fn validate_request(request: &SpeechRequest) -> Result<(), ModelError> {
    if request.text.trim().is_empty() {
        return Err(ModelError::InvalidConfiguration(
            "speech request text must not be empty".into(),
        ));
    }
    if request.params.speaking_rate.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "fish-speech backend does not support SpeechParams.speaking_rate in the v0.3 vertical slice".into(),
        ));
    }
    if request.params.seed.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "fish-speech backend does not support SpeechParams.seed in the v0.3 vertical slice".into(),
        ));
    }
    Ok(())
}

fn prepare_voice_dir() -> Result<Option<TempDir>, ModelError> {
    let dir = TempDir::new().map_err(|err| ModelError::BackendInitialization {
        backend: "fish-speech",
        message: format!("failed to create embedded voice tempdir: {err}"),
    })?;

    std::fs::write(dir.path().join("default.npy"), EMBEDDED_DEFAULT_VOICE_NPY).map_err(|err| {
        ModelError::BackendInitialization {
            backend: "fish-speech",
            message: format!("failed to write embedded default.npy: {err}"),
        }
    })?;
    std::fs::write(dir.path().join("index.json"), EMBEDDED_DEFAULT_VOICE_INDEX).map_err(|err| {
        ModelError::BackendInitialization {
            backend: "fish-speech",
            message: format!("failed to write embedded index.json: {err}"),
        }
    })?;

    Ok(Some(dir))
}

fn select_device_and_dtype() -> Result<(Device, DType), ModelError> {
    let force_cpu = std::env::var_os("MOTLIE_MODEL_FORCE_CPU").is_some();
    if force_cpu {
        return Ok((Device::Cpu, DType::F32));
    }

    #[cfg(feature = "cuda")]
    {
        let device = Device::cuda_if_available(0).map_err(|err| ModelError::BackendInitialization {
            backend: "fish-speech",
            message: format!("failed to initialize CUDA device: {err}"),
        })?;
        return Ok((device, DType::BF16));
    }

    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        let device = Device::new_metal(0).map_err(|err| ModelError::BackendInitialization {
            backend: "fish-speech",
            message: format!("failed to initialize Metal device: {err}"),
        })?;
        return Ok((device, DType::F32));
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        Ok((Device::Cpu, DType::F32))
    }
}

async fn generate_pcm(
    state: Arc<AppState>,
    prompts: (usize, Vec<Tensor>),
) -> Result<Vec<f32>, ModelError> {
    let (n_conditioning_tokens, prompts) = prompts;
    let mut all_pcm = Vec::new();

    for prompt in prompts {
        let semantic_tokens =
            generate_semantic_tokens(state.clone(), &prompt, n_conditioning_tokens).await?;
        let pcm = vocode_semantic_tokens(state.clone(), &semantic_tokens).await?;
        all_pcm.extend(pcm);
    }

    clear_caches(&state).await?;
    Ok(all_pcm)
}

async fn generate_semantic_tokens(
    state: Arc<AppState>,
    encoded_input: &Tensor,
    n_conditioning_tokens: usize,
) -> Result<Tensor, ModelError> {
    let mut model = state.lm.model.lock().await;
    let (tokens, _) = generate_blocking_with_hidden(
        &mut model,
        encoded_input,
        state.lm.max_new_tokens,
        &state.lm.default_sampling_args,
        false,
        true,
    )
    .map_err(|err| backend_execution_error("generate_blocking_with_hidden", err))?;

    model
        .clear_slow_caches_until(n_conditioning_tokens)
        .map_err(|err| backend_execution_error("clear_slow_caches_until", err))?;

    match state.lm.model_type {
        WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => Ok(tokens),
        _ => {
            let ones = Tensor::ones_like(&tokens)
                .map_err(|err| backend_execution_error("ones_like", err))?;
            tokens
                .broadcast_sub(&ones)
                .map_err(|err| backend_execution_error("broadcast_sub", err))
        }
    }
}

async fn vocode_semantic_tokens(
    state: Arc<AppState>,
    semantic_tokens: &Tensor,
) -> Result<Vec<f32>, ModelError> {
    let (_, sequence_len) = semantic_tokens
        .dims2()
        .map_err(|err| backend_execution_error("dims2", err))?;
    let tokens = match state.lm.model_type {
        WhichLM::DualAR => semantic_tokens
            .i((.., ..sequence_len - 1))
            .map_err(|err| backend_execution_error("slice_semantic_tokens", err))?,
        _ => semantic_tokens.clone(),
    };

    let out = state
        .codec
        .decode_batch(&tokens)
        .await
        .map_err(|err| backend_execution_error("decode_batch", err))?;
    out.squeeze(0)
        .and_then(|tensor| tensor.squeeze(0))
        .map_err(|err| backend_execution_error("squeeze_audio", err))?
        .to_vec1::<f32>()
        .map_err(|err| backend_execution_error("tensor_to_vec1", err))
}

async fn clear_caches(state: &Arc<AppState>) -> Result<(), ModelError> {
    let mut model = state.lm.model.lock().await;
    model.clear_slow_layer_caches();
    Ok(())
}

fn pcm_f32_to_s16le(samples: &[f32]) -> Result<Vec<u8>, ModelError> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let quantized = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    Ok(bytes)
}

fn backend_execution_error(
    operation: &'static str,
    error: impl std::fmt::Display,
) -> ModelError {
    ModelError::BackendExecution {
        backend: "fish-speech",
        operation,
        message: error.to_string(),
    }
}

struct FishSpeechStream {
    audio_spec: motlie_model::AudioSpec,
    pcm: Vec<u8>,
    chunk_bytes: usize,
    offset: usize,
    next_sequence: u64,
    finished: bool,
}

impl FishSpeechStream {
    fn new(audio_spec: motlie_model::AudioSpec, pcm: Vec<u8>) -> Result<Self, ModelError> {
        let bytes_per_sample = match audio_spec.encoding {
            PcmEncoding::S16Le => 2usize,
            PcmEncoding::F32Le => 4usize,
        };
        let chunk_bytes =
            ((audio_spec.sample_rate_hz as usize * bytes_per_sample * OUTPUT_CHUNK_DURATION_MS as usize)
                / 1000)
                .max(bytes_per_sample);
        Ok(Self {
            audio_spec,
            pcm,
            chunk_bytes,
            offset: 0,
            next_sequence: 0,
            finished: false,
        })
    }
}

#[async_trait]
impl SpeechStream for FishSpeechStream {
    fn audio_spec(&self) -> &motlie_model::AudioSpec {
        &self.audio_spec
    }

    async fn next_chunk(&mut self) -> Result<Option<PcmChunk>, ModelError> {
        if self.finished {
            return Ok(None);
        }

        let end = (self.offset + self.chunk_bytes).min(self.pcm.len());
        let data = self.pcm[self.offset..end].to_vec();
        self.offset = end;
        self.finished = self.offset >= self.pcm.len();
        let sequence = self.next_sequence;
        self.next_sequence += 1;

        Ok(Some(PcmChunk {
            data,
            sequence,
            end_of_stream: self.finished,
        }))
    }

    async fn finish(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_voice_assets_exist() {
        assert!(!EMBEDDED_DEFAULT_VOICE_NPY.is_empty());
        assert!(EMBEDDED_DEFAULT_VOICE_INDEX.contains("default"));
    }

    #[test]
    fn request_validation_rejects_unsupported_controls() {
        let error = validate_request(&SpeechRequest {
            text: "hello".into(),
            params: motlie_model::SpeechParams {
                speaking_rate: Some(1.1),
                seed: None,
            },
            conditioning: None,
        })
        .expect_err("speaking rate should be rejected");

        assert!(matches!(error, ModelError::InvalidConfiguration(_)));
    }

    #[test]
    fn checkpoint_dir_validation_requires_all_artifacts() {
        let root = tempfile::tempdir().expect("tempdir should be creatable");
        for filename in REQUIRED_ARTIFACTS.iter().take(3) {
            std::fs::write(root.path().join(filename), b"stub").expect("stub should write");
        }

        let error = resolve_checkpoint_dir(&ResolvedCheckpoint {
            checkpoint: motlie_model::ModelCheckpoint {
                format: CheckpointFormat::Safetensors,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "jkeisling/fish-speech-1.5",
                },
                include: Vec::new(),
                quantization: None,
            },
            path: root.path().to_path_buf(),
        })
        .expect_err("missing artifacts should be rejected");

        assert!(matches!(error, ModelError::InvalidConfiguration(_)));
    }
}
