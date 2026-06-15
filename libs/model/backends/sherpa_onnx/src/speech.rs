use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{
    AudioBuf, BufferedSpeechChunkStream, BufferedSpeechSynthesizer, IncrementalSpeechCancelToken,
    IncrementalSpeechChunk, IncrementalSpeechControls, IncrementalSpeechStream,
    IncrementalSpeechSummary, IncrementalSpeechSynthesizer, Mono, SpeechSynthesizer,
    SynthesisRequest,
};
use motlie_model::{
    ArtifactPolicy, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata,
    Capabilities, CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle,
    ModelError, ModelIdentity, ModelMetricSnapshot, QuantizationSupport,
    RuntimeAcceleratorObservation, SpeechParams, StartOptions, UnsupportedChat,
    UnsupportedCompletion, UnsupportedEmbeddings,
};
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig};
use tokio::sync::mpsc;

use crate::common::{RuntimeMetricState, lock_metrics, observe_latency, observe_memory};

const SHERPA_ONNX_TTS_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const NUM_THREADS: i32 = 2;
const DEFAULT_MAX_BUFFERED_AUDIO_MS: u32 = 80;

pub type SherpaOnnxKokoroBufferedSpeechStream =
    BufferedSpeechChunkStream<i16, KOKORO_SAMPLE_RATE_HZ, Mono>;

#[derive(Clone, Debug)]
pub struct SherpaOnnxKokoroTtsArtifactSpec<'a> {
    pub model: &'a str,
    pub voices: &'a str,
    pub tokens: &'a str,
    pub data_dir: Option<&'a str>,
    pub dict_dir: Option<&'a str>,
    pub lexicon: Option<&'a str>,
    pub lang: Option<&'a str>,
}

#[derive(Clone, Debug)]
struct SherpaOnnxKokoroTtsArtifactPaths {
    model: PathBuf,
    voices: PathBuf,
    tokens: PathBuf,
    data_dir: Option<PathBuf>,
    dict_dir: Option<PathBuf>,
    lexicon: Option<PathBuf>,
    lang: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxKokoroTtsSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub artifact: SherpaOnnxKokoroTtsArtifactSpec<'static>,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl SherpaOnnxKokoroTtsSpec {
    pub fn kokoro_en_v0_19() -> Self {
        Self {
            id: BundleId::new("sherpa_onnx_kokoro_en_v0_19"),
            display_name: "Sherpa ONNX Kokoro EN v0.19 incremental TTS",
            artifact: SherpaOnnxKokoroTtsArtifactSpec {
                model: "model.onnx",
                voices: "voices.bin",
                tokens: "tokens.txt",
                data_dir: Some("espeak-ng-data"),
                dict_dir: None,
                lexicon: None,
                lang: Some("en-us"),
            },
            capabilities: Capabilities::speech_buffered_and_streaming(),
            quantization: QuantizationSupport::none(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxKokoroTtsAdapter {
    spec: SherpaOnnxKokoroTtsSpec,
}

impl SherpaOnnxKokoroTtsAdapter {
    pub fn kokoro_en_v0_19() -> Self {
        Self {
            spec: SherpaOnnxKokoroTtsSpec::kokoro_en_v0_19(),
        }
    }
}

#[async_trait]
impl BackendAdapter for SherpaOnnxKokoroTtsAdapter {
    type Handle = SherpaOnnxKokoroTtsHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &SHERPA_ONNX_TTS_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::SherpaOnnx
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
        checkpoint: &motlie_model::ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError> {
        self.spec
            .quantization
            .resolve(options.quantization, &identity.id)?;

        let artifacts = resolve_kokoro_tts_artifacts(checkpoint, self.spec.artifact.clone())?;
        let runtime = Arc::new(load_kokoro_tts_runtime(&artifacts)?);

        Ok(new_kokoro_tts_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.spec.capabilities.clone(),
            self.spec.quantization.clone(),
            runtime,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxKokoroTtsBundle {
    metadata: BundleMetadata,
    spec: SherpaOnnxKokoroTtsSpec,
}

impl SherpaOnnxKokoroTtsBundle {
    pub fn new(spec: SherpaOnnxKokoroTtsSpec) -> Self {
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

    pub async fn start_typed(
        &self,
        options: StartOptions,
    ) -> Result<SherpaOnnxKokoroTtsHandle, ModelError> {
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let artifacts = if let Some(policy) = options.artifact_policy {
            configure_kokoro_tts_artifact_policy(self.spec.artifact.clone(), policy)?
        } else {
            configure_kokoro_tts_artifact_policy(
                self.spec.artifact.clone(),
                ArtifactPolicy::LocalOnly {
                    root: PathBuf::from("."),
                },
            )?
        };
        let runtime = Arc::new(load_kokoro_tts_runtime(&artifacts)?);

        Ok(new_kokoro_tts_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            runtime,
        ))
    }
}

#[async_trait]
impl ModelBundle for SherpaOnnxKokoroTtsBundle {
    type Handle = SherpaOnnxKokoroTtsHandle;

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

pub struct SherpaOnnxKokoroTtsHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<SherpaOnnxKokoroTtsRuntime>,
    metrics: Arc<Mutex<TtsMetrics>>,
}

impl SherpaOnnxKokoroTtsHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }
}

#[derive(Clone, Debug, Default)]
struct TtsMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for SherpaOnnxKokoroTtsHandle {
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
        let metrics = lock_metrics(&self.metrics, "sherpa-onnx-kokoro-tts-metric-snapshot").clone();
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

    fn accelerator_observation(&self) -> Option<RuntimeAcceleratorObservation> {
        Some(RuntimeAcceleratorObservation {
            backend_mode: "sherpa_onnx:cpu".to_owned(),
            offload: Some(sherpa_cpu_offload_reason()),
            selected_device: None,
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

fn new_kokoro_tts_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<SherpaOnnxKokoroTtsRuntime>,
) -> SherpaOnnxKokoroTtsHandle {
    let metrics = Arc::new(Mutex::new(TtsMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "sherpa-onnx-kokoro-tts-start");
        observe_memory(&mut state.runtime);
    }

    SherpaOnnxKokoroTtsHandle {
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

struct SherpaOnnxKokoroTtsRuntime {
    tts: Mutex<OfflineTts>,
    sample_rate_hz: u32,
}

impl SherpaOnnxKokoroTtsRuntime {
    fn synthesize_buffered(
        &self,
        request: SynthesisRequest,
    ) -> Result<AudioBuf<i16, KOKORO_SAMPLE_RATE_HZ, Mono>, ModelError> {
        let text = validated_text(&request)?;
        let config = generation_config(&request.params)?;
        let tts = self
            .tts
            .lock()
            .map_err(|_| ModelError::Internal("sherpa-onnx kokoro tts mutex poisoned".into()))?;
        let audio = tts
            .generate_with_config(text, &config, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| ModelError::BackendExecution {
                backend: "sherpa-onnx-kokoro-tts",
                operation: "generate_with_config",
                message: "upstream TTS generation returned no audio".into(),
            })?;

        Ok(AudioBuf::new(f32_to_i16_samples(audio.samples())))
    }

    fn synthesize_incremental(
        &self,
        request: SynthesisRequest,
        controls: IncrementalSpeechControls,
        sender: mpsc::Sender<Result<IncrementalSpeechChunk, ModelError>>,
    ) -> Result<IncrementalSpeechSummary, ModelError> {
        if controls.cancel.is_canceled() {
            return Ok(IncrementalSpeechSummary {
                canceled: true,
                ..IncrementalSpeechSummary::default()
            });
        }

        let text = validated_text(&request)?;
        let config = generation_config(&request.params)?;
        let progress = Arc::new(Mutex::new(IncrementalWorkerProgress::new(
            self.sample_rate_hz,
            1,
        )));
        let callback_progress = Arc::clone(&progress);
        let callback_cancel = controls.cancel.clone();
        let callback_sender = sender;

        let tts = self
            .tts
            .lock()
            .map_err(|_| ModelError::Internal("sherpa-onnx kokoro tts mutex poisoned".into()))?;
        let audio = tts.generate_with_config(
            text,
            &config,
            Some(move |samples: &[f32], progress_value: f32| -> bool {
                send_incremental_delta(
                    samples,
                    progress_value,
                    &callback_progress,
                    &callback_cancel,
                    &callback_sender,
                )
            }),
        );

        let worker_progress = progress
            .lock()
            .map_err(|_| {
                ModelError::Internal("sherpa-onnx kokoro tts progress mutex poisoned".into())
            })?
            .clone();
        let canceled = controls.cancel.is_canceled();
        let synthesis_completed = audio.is_some() && !canceled;
        if audio.is_none() && !canceled {
            return Err(ModelError::BackendExecution {
                backend: "sherpa-onnx-kokoro-tts",
                operation: "generate_with_config",
                message: "upstream TTS generation stopped before completion".into(),
            });
        }
        if synthesis_completed && worker_progress.chunks == 0 {
            return Err(ModelError::BackendExecution {
                backend: "sherpa-onnx-kokoro-tts",
                operation: "generate_with_config",
                message: "upstream TTS generation completed without callback audio chunks".into(),
            });
        }

        Ok(IncrementalSpeechSummary {
            chunks: worker_progress.chunks,
            audio_ms: worker_progress.audio_ms,
            canceled,
            synthesis_completed,
        })
    }
}

#[derive(Clone, Debug)]
struct IncrementalWorkerProgress {
    sample_rate_hz: u32,
    channels: u16,
    sent_samples: usize,
    chunks: u64,
    audio_ms: u64,
    last_progress: f32,
}

impl IncrementalWorkerProgress {
    fn new(sample_rate_hz: u32, channels: u16) -> Self {
        Self {
            sample_rate_hz,
            channels,
            sent_samples: 0,
            chunks: 0,
            audio_ms: 0,
            last_progress: 0.0,
        }
    }
}

fn send_incremental_delta(
    samples: &[f32],
    progress_value: f32,
    progress: &Arc<Mutex<IncrementalWorkerProgress>>,
    cancel: &IncrementalSpeechCancelToken,
    sender: &mpsc::Sender<Result<IncrementalSpeechChunk, ModelError>>,
) -> bool {
    if cancel.is_canceled() {
        return false;
    }

    let chunks = {
        let Ok(mut state) = progress.lock() else {
            let _ = sender.blocking_send(Err(ModelError::Internal(
                "sherpa-onnx kokoro tts progress mutex poisoned".into(),
            )));
            return false;
        };
        state.last_progress = progress_value;
        if samples.len() <= state.sent_samples {
            return true;
        }

        let delta = f32_to_i16_samples(&samples[state.sent_samples..]);
        state.sent_samples = samples.len();
        let chunk_len = samples_per_output_chunk(state.sample_rate_hz, state.channels);
        let mut chunks = delta
            .chunks(chunk_len)
            .filter(|samples| !samples.is_empty())
            .map(|samples| {
                let chunk = IncrementalSpeechChunk {
                    samples_i16: samples.to_vec(),
                    sample_rate_hz: state.sample_rate_hz,
                    channels: state.channels,
                    chunk_index: state.chunks,
                    is_final: false,
                };
                state.chunks = state.chunks.saturating_add(1);
                state.audio_ms = state.audio_ms.saturating_add(chunk.audio_ms());
                chunk
            })
            .collect::<Vec<_>>();
        if progress_value >= 1.0 {
            if let Some(last) = chunks.last_mut() {
                last.is_final = true;
            }
        }
        chunks
    };

    for chunk in chunks {
        if sender.blocking_send(Ok(chunk)).is_err() || cancel.is_canceled() {
            return false;
        }
    }

    true
}

fn samples_per_output_chunk(sample_rate_hz: u32, channels: u16) -> usize {
    let frames = (sample_rate_hz as usize)
        .saturating_mul(OUTPUT_CHUNK_DURATION_MS as usize)
        .div_ceil(1000)
        .max(1);
    frames.saturating_mul(channels.max(1) as usize)
}

impl BufferedSpeechSynthesizer for SherpaOnnxKokoroTtsHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<i16, KOKORO_SAMPLE_RATE_HZ, Mono>;

    async fn synthesize_buffered(
        &self,
        request: Self::Request,
    ) -> Result<Self::Output, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);
        join_sherpa_tts_blocking(tokio::task::spawn_blocking(move || {
            let started_at = Instant::now();
            let result = runtime.synthesize_buffered(request);
            observe_tts_latency(&metrics, started_at);
            result
        }))
        .await
    }
}

impl SpeechSynthesizer for SherpaOnnxKokoroTtsHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<i16, KOKORO_SAMPLE_RATE_HZ, Mono>;
    type Stream = SherpaOnnxKokoroBufferedSpeechStream;

    async fn synthesize(&self, request: Self::Request) -> Result<Self::Stream, ModelError> {
        Ok(SherpaOnnxKokoroBufferedSpeechStream::new(
            self.synthesize_buffered(request).await?,
            OUTPUT_CHUNK_DURATION_MS,
        ))
    }
}

impl IncrementalSpeechSynthesizer for SherpaOnnxKokoroTtsHandle {
    type Request = SynthesisRequest;
    type Stream = SherpaOnnxKokoroIncrementalSpeechStream;

    async fn synthesize_incremental(
        &self,
        request: Self::Request,
        mut controls: IncrementalSpeechControls,
    ) -> Result<Self::Stream, ModelError> {
        if controls.max_buffered_audio_ms == 0 {
            controls.max_buffered_audio_ms = DEFAULT_MAX_BUFFERED_AUDIO_MS;
        }
        let capacity = buffered_audio_capacity(controls.max_buffered_audio_ms);
        let (sender, receiver) = mpsc::channel(capacity);
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);
        let worker = tokio::task::spawn_blocking(move || {
            let started_at = Instant::now();
            let result = runtime.synthesize_incremental(request, controls, sender);
            observe_tts_latency(&metrics, started_at);
            result
        });

        Ok(SherpaOnnxKokoroIncrementalSpeechStream { receiver, worker })
    }
}

pub struct SherpaOnnxKokoroIncrementalSpeechStream {
    receiver: mpsc::Receiver<Result<IncrementalSpeechChunk, ModelError>>,
    worker: tokio::task::JoinHandle<Result<IncrementalSpeechSummary, ModelError>>,
}

impl IncrementalSpeechStream for SherpaOnnxKokoroIncrementalSpeechStream {
    async fn next_audio_chunk(&mut self) -> Result<Option<IncrementalSpeechChunk>, ModelError> {
        match self.receiver.recv().await {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }
    }

    async fn finish(mut self) -> Result<IncrementalSpeechSummary, ModelError> {
        while let Some(event) = self.receiver.recv().await {
            event?;
        }
        join_sherpa_tts_blocking(self.worker).await
    }
}

async fn join_sherpa_tts_blocking<T>(
    handle: tokio::task::JoinHandle<Result<T, ModelError>>,
) -> Result<T, ModelError> {
    match handle.await {
        Ok(result) => result,
        Err(err) => Err(ModelError::BackendExecution {
            backend: "sherpa-onnx-kokoro-tts",
            operation: "join_blocking_tts_task",
            message: err.to_string(),
        }),
    }
}

fn observe_tts_latency(metrics: &Arc<Mutex<TtsMetrics>>, started_at: Instant) {
    let mut state = lock_metrics(metrics, "sherpa-onnx-kokoro-tts-synthesize");
    observe_latency(&mut state.runtime, started_at.elapsed());
}

fn load_kokoro_tts_runtime(
    artifacts: &SherpaOnnxKokoroTtsArtifactPaths,
) -> Result<SherpaOnnxKokoroTtsRuntime, ModelError> {
    let mut config = OfflineTtsConfig::default();
    config.model.kokoro = OfflineTtsKokoroModelConfig {
        model: Some(path_to_string(&artifacts.model)?),
        voices: Some(path_to_string(&artifacts.voices)?),
        tokens: Some(path_to_string(&artifacts.tokens)?),
        data_dir: optional_path_to_string(artifacts.data_dir.as_deref())?,
        length_scale: 1.0,
        dict_dir: optional_path_to_string(artifacts.dict_dir.as_deref())?,
        lexicon: optional_path_to_string(artifacts.lexicon.as_deref())?,
        lang: artifacts.lang.clone(),
    };
    config.model.num_threads = NUM_THREADS;
    config.model.provider = Some("cpu".to_owned());

    let tts = OfflineTts::create(&config).ok_or_else(|| ModelError::BackendInitialization {
        backend: "sherpa-onnx-kokoro-tts",
        message: "failed to create upstream sherpa-onnx offline TTS".into(),
    })?;
    let sample_rate_hz =
        u32::try_from(tts.sample_rate()).map_err(|err| ModelError::BackendInitialization {
            backend: "sherpa-onnx-kokoro-tts",
            message: format!("invalid Kokoro sample rate reported by sherpa-onnx: {err}"),
        })?;
    if sample_rate_hz != KOKORO_SAMPLE_RATE_HZ {
        return Err(ModelError::InvalidConfiguration(format!(
            "sherpa-onnx Kokoro sample rate {sample_rate_hz} does not match expected {KOKORO_SAMPLE_RATE_HZ}"
        )));
    }

    Ok(SherpaOnnxKokoroTtsRuntime {
        tts: Mutex::new(tts),
        sample_rate_hz,
    })
}

fn resolve_kokoro_tts_artifacts(
    checkpoint: &motlie_model::ResolvedCheckpoint,
    spec: SherpaOnnxKokoroTtsArtifactSpec<'_>,
) -> Result<SherpaOnnxKokoroTtsArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "sherpa-onnx Kokoro TTS expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .ok_or_else(|| {
                ModelError::InvalidConfiguration(format!(
                    "onnx checkpoint path `{}` has no parent directory",
                    checkpoint.path.display()
                ))
            })?
            .to_path_buf()
    };

    resolve_kokoro_tts_paths(&root, spec)
}

fn configure_kokoro_tts_artifact_policy(
    spec: SherpaOnnxKokoroTtsArtifactSpec<'_>,
    policy: ArtifactPolicy,
) -> Result<SherpaOnnxKokoroTtsArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    resolve_kokoro_tts_paths(&root, spec)
}

fn resolve_kokoro_tts_paths(
    root: &Path,
    spec: SherpaOnnxKokoroTtsArtifactSpec<'_>,
) -> Result<SherpaOnnxKokoroTtsArtifactPaths, ModelError> {
    Ok(SherpaOnnxKokoroTtsArtifactPaths {
        model: require_file(root, spec.model, "model")?,
        voices: require_file(root, spec.voices, "voices")?,
        tokens: require_file(root, spec.tokens, "tokens")?,
        data_dir: optional_existing_path(root, spec.data_dir, "data_dir")?,
        dict_dir: optional_existing_path(root, spec.dict_dir, "dict_dir")?,
        lexicon: optional_existing_path(root, spec.lexicon, "lexicon")?,
        lang: spec.lang.map(str::to_owned),
    })
}

fn require_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf, ModelError> {
    let path = root.join(relative);
    if !path.is_file() {
        return Err(ModelError::InvalidConfiguration(format!(
            "required sherpa-onnx Kokoro TTS {label} artifact `{relative}` not found under `{}`",
            root.display()
        )));
    }
    Ok(path)
}

fn optional_existing_path(
    root: &Path,
    relative: Option<&str>,
    label: &str,
) -> Result<Option<PathBuf>, ModelError> {
    let Some(relative) = relative else {
        return Ok(None);
    };
    let path = root.join(relative);
    if !path.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "required sherpa-onnx Kokoro TTS {label} artifact `{relative}` not found under `{}`",
            root.display()
        )));
    }
    Ok(Some(path))
}

fn validated_text(request: &SynthesisRequest) -> Result<&str, ModelError> {
    let text = request.text.trim();
    if text.is_empty() {
        return Err(ModelError::InvalidConfiguration(
            "speech request requires non-empty text".into(),
        ));
    }
    Ok(text)
}

fn generation_config(params: &SpeechParams) -> Result<GenerationConfig, ModelError> {
    if params.seed.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "sherpa-onnx Kokoro backend does not support `SpeechParams.seed`".into(),
        ));
    }
    Ok(GenerationConfig {
        speed: synthesis_speed(params)?,
        ..GenerationConfig::default()
    })
}

fn synthesis_speed(params: &SpeechParams) -> Result<f32, ModelError> {
    match params.speaking_rate {
        Some(rate) if !rate.is_finite() || rate <= 0.0 => Err(ModelError::InvalidConfiguration(
            format!("speech speaking_rate must be positive and finite, got {rate}"),
        )),
        Some(rate) => Ok(rate),
        None => Ok(1.0),
    }
}

fn f32_to_i16_samples(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

fn buffered_audio_capacity(max_buffered_audio_ms: u32) -> usize {
    let chunk_ms = OUTPUT_CHUNK_DURATION_MS.max(1);
    let capacity = max_buffered_audio_ms.div_ceil(chunk_ms).max(1);
    capacity as usize
}

fn path_to_string(path: &Path) -> Result<String, ModelError> {
    path.to_str().map(ToOwned::to_owned).ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("non-UTF-8 path: {}", path.display()))
    })
}

fn optional_path_to_string(path: Option<&Path>) -> Result<Option<String>, ModelError> {
    path.map(path_to_string).transpose()
}

fn sherpa_cpu_offload_reason() -> String {
    if motlie_model::metrics_runtime::should_force_cpu() {
        "provider=cpu;force_cpu=true".to_owned()
    } else if cfg!(feature = "cuda") {
        "provider=cpu;cuda_feature_noop".to_owned()
    } else {
        "provider=cpu".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::SpeechGeneration;

    #[test]
    fn sherpa_kokoro_spec_advertises_buffered_and_streaming_speech() {
        let spec = SherpaOnnxKokoroTtsSpec::kokoro_en_v0_19();

        assert!(spec.capabilities.supports(CapabilityKind::Speech));
        assert!(
            spec.capabilities
                .supports_speech_generation(SpeechGeneration::Buffered)
        );
        assert!(
            spec.capabilities
                .supports_speech_generation(SpeechGeneration::Streaming)
        );
    }

    #[test]
    fn buffered_audio_capacity_is_bounded_to_at_least_one_chunk() {
        assert_eq!(buffered_audio_capacity(0), 1);
        assert_eq!(buffered_audio_capacity(1), 1);
        assert_eq!(buffered_audio_capacity(40), 1);
        assert_eq!(buffered_audio_capacity(41), 2);
        assert_eq!(buffered_audio_capacity(80), 2);
    }

    #[test]
    fn delta_callback_sends_only_new_samples() {
        let (sender, mut receiver) = mpsc::channel(4);
        let progress = Arc::new(Mutex::new(IncrementalWorkerProgress::new(24_000, 1)));
        let cancel = IncrementalSpeechCancelToken::new();

        assert!(send_incremental_delta(
            &[0.0, 0.25],
            0.2,
            &progress,
            &cancel,
            &sender,
        ));
        assert!(send_incremental_delta(
            &[0.0, 0.25, 0.5],
            1.0,
            &progress,
            &cancel,
            &sender,
        ));
        drop(sender);

        let first = receiver
            .blocking_recv()
            .expect("first event should exist")
            .expect("first event should be ok");
        let second = receiver
            .blocking_recv()
            .expect("second event should exist")
            .expect("second event should be ok");
        assert_eq!(first.samples_i16.len(), 2);
        assert!(!first.is_final);
        assert_eq!(second.samples_i16.len(), 1);
        assert!(second.is_final);
    }

    #[test]
    fn callback_delta_is_split_into_bounded_audio_chunks() {
        let (sender, mut receiver) = mpsc::channel(4);
        let progress = Arc::new(Mutex::new(IncrementalWorkerProgress::new(1_000, 1)));
        let cancel = IncrementalSpeechCancelToken::new();
        let samples = vec![0.0; 100];

        assert!(send_incremental_delta(
            &samples, 1.0, &progress, &cancel, &sender,
        ));
        drop(sender);

        let mut lengths = Vec::new();
        let mut finals = Vec::new();
        while let Some(event) = receiver.blocking_recv() {
            let chunk = event.expect("chunk event should be ok");
            lengths.push(chunk.samples_i16.len());
            finals.push(chunk.is_final);
        }

        assert_eq!(lengths, vec![40, 40, 20]);
        assert_eq!(finals, vec![false, false, true]);
    }

    #[test]
    fn invalid_speed_is_rejected() {
        let err = generation_config(&SpeechParams {
            speaking_rate: Some(0.0),
            seed: None,
        })
        .expect_err("zero speed should fail");

        assert!(matches!(
            err,
            ModelError::InvalidConfiguration(message) if message.contains("speaking_rate")
        ));
    }
}
