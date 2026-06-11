use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, Mono, StreamingTranscriber, TranscriptionSession};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationSupport, ResolvedCheckpoint,
    RuntimeAcceleratorObservation, StartOptions, TranscriptSegment, TranscriptionParams,
    TranscriptionUpdate, UnsupportedChat, UnsupportedCompletion, UnsupportedEmbeddings,
};
use sherpa_onnx::{
    OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, OnlineTransducerModelConfig,
    RecognizerResult,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_onnx_artifacts, RuntimeMetricState, SherpaArtifactPaths, SherpaArtifactSpec,
};

const SHERPA_ONNX_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
const NUM_THREADS: i32 = 2;
const MODIFIED_BEAM_SEARCH_PATHS: i32 = 4;
const RULE1_MIN_TRAILING_SILENCE_SECS: f32 = 2.4;
const RULE2_MIN_TRAILING_SILENCE_SECS: f32 = 1.2;
const RULE3_MIN_UTTERANCE_LENGTH_SECS: f32 = 20.0;

#[derive(Clone, Debug)]
pub struct SherpaOnnxStreamingSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub encoder_filename: &'static str,
    pub decoder_filename: &'static str,
    pub joiner_filename: &'static str,
    pub tokens_filename: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl SherpaOnnxStreamingSpec {
    pub fn zipformer_en_streaming() -> Self {
        Self {
            id: BundleId::new("sherpa_onnx_streaming_zipformer_en"),
            display_name: "Sherpa ONNX Streaming Zipformer EN",
            encoder_filename: "encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
            decoder_filename: "decoder-epoch-99-avg-1-chunk-16-left-64.onnx",
            joiner_filename: "joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
            tokens_filename: "tokens.txt",
            capabilities: Capabilities::transcription_stream_partial_only(),
            quantization: QuantizationSupport::none(),
        }
    }

    fn artifact_spec(&self) -> SherpaArtifactSpec<'_> {
        SherpaArtifactSpec {
            encoder: self.encoder_filename,
            decoder: self.decoder_filename,
            joiner: self.joiner_filename,
            tokens: self.tokens_filename,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxStreamingAdapter {
    spec: SherpaOnnxStreamingSpec,
}

impl SherpaOnnxStreamingAdapter {
    pub fn zipformer_en_streaming() -> Self {
        Self {
            spec: SherpaOnnxStreamingSpec::zipformer_en_streaming(),
        }
    }
}

#[async_trait]
impl BackendAdapter for SherpaOnnxStreamingAdapter {
    type Handle = SherpaOnnxHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &SHERPA_ONNX_FORMATS
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
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError> {
        self.spec
            .quantization
            .resolve(options.quantization, &identity.id)?;

        let artifacts = resolve_onnx_artifacts(checkpoint, self.spec.artifact_spec())?;
        let runtime = Arc::new(load_runtime(&artifacts)?);

        Ok(new_transcription_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.spec.capabilities.clone(),
            self.spec.quantization.clone(),
            runtime,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxStreamingBundle {
    metadata: BundleMetadata,
    artifacts: SherpaOnnxStreamingSpec,
}

impl SherpaOnnxStreamingBundle {
    pub fn new(spec: SherpaOnnxStreamingSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id.clone(),
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities.clone(),
                quantization: spec.quantization.clone(),
            },
            artifacts: spec,
        }
    }
}

#[async_trait]
impl ModelBundle for SherpaOnnxStreamingBundle {
    type Handle = SherpaOnnxHandle;

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

impl SherpaOnnxStreamingBundle {
    pub async fn start_typed(&self, options: StartOptions) -> Result<SherpaOnnxHandle, ModelError> {
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let artifacts = if let Some(policy) = options.artifact_policy {
            configure_artifact_policy(self.artifacts.artifact_spec(), policy)?
        } else {
            configure_artifact_policy(
                self.artifacts.artifact_spec(),
                motlie_model::ArtifactPolicy::LocalOnly {
                    root: PathBuf::from("."),
                },
            )?
        };

        let runtime = Arc::new(load_runtime(&artifacts)?);
        Ok(new_transcription_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            runtime,
        ))
    }
}

pub struct SherpaOnnxHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<SherpaOnnxRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
}

impl SherpaOnnxHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }
}

#[derive(Clone, Debug, Default)]
struct AsrMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for SherpaOnnxHandle {
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
        let metrics = lock_metrics(&self.metrics, "sherpa-onnx-metric-snapshot").clone();
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

fn new_transcription_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<SherpaOnnxRuntime>,
) -> SherpaOnnxHandle {
    let metrics = Arc::new(Mutex::new(AsrMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "sherpa-onnx-start");
        observe_memory(&mut state.runtime);
    }

    SherpaOnnxHandle {
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

struct SherpaOnnxRuntime {
    recognizer: Mutex<OnlineRecognizer>,
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

fn load_runtime(artifacts: &SherpaArtifactPaths) -> Result<SherpaOnnxRuntime, ModelError> {
    let mut config = OnlineRecognizerConfig::default();
    config.feat_config.sample_rate = TARGET_SAMPLE_RATE_HZ as i32;
    config.feat_config.feature_dim = 80;
    config.model_config.transducer = OnlineTransducerModelConfig {
        encoder: Some(path_to_string(&artifacts.encoder)?),
        decoder: Some(path_to_string(&artifacts.decoder)?),
        joiner: Some(path_to_string(&artifacts.joiner)?),
    };
    config.model_config.tokens = Some(path_to_string(&artifacts.tokens)?);
    config.model_config.num_threads = NUM_THREADS;
    config.model_config.provider = Some("cpu".to_string());
    config.decoding_method = Some("modified_beam_search".to_string());
    config.max_active_paths = MODIFIED_BEAM_SEARCH_PATHS;
    config.enable_endpoint = true;
    config.rule1_min_trailing_silence = RULE1_MIN_TRAILING_SILENCE_SECS;
    config.rule2_min_trailing_silence = RULE2_MIN_TRAILING_SILENCE_SECS;
    config.rule3_min_utterance_length = RULE3_MIN_UTTERANCE_LENGTH_SECS;

    let recognizer =
        OnlineRecognizer::create(&config).ok_or_else(|| ModelError::BackendInitialization {
            backend: "sherpa-onnx",
            message: "failed to create upstream sherpa-onnx online recognizer".into(),
        })?;

    Ok(SherpaOnnxRuntime {
        recognizer: Mutex::new(recognizer),
    })
}

fn path_to_string(path: &std::path::Path) -> Result<String, ModelError> {
    path.to_str().map(ToOwned::to_owned).ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("non-UTF-8 path: {}", path.display()))
    })
}

pub struct SherpaOnnxStream {
    runtime: Arc<SherpaOnnxRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
    params: TranscriptionParams,
    stream: OnlineStream,
    samples_seen: usize,
    last_partial: String,
}

impl SherpaOnnxStream {
    fn new(
        runtime: Arc<SherpaOnnxRuntime>,
        metrics: Arc<Mutex<AsrMetrics>>,
        params: TranscriptionParams,
    ) -> Result<Self, ModelError> {
        let stream = runtime
            .recognizer
            .lock()
            .map_err(|_| ModelError::Internal("sherpa recognizer mutex poisoned".into()))?
            .create_stream();

        Ok(Self {
            runtime,
            metrics,
            params,
            stream,
            samples_seen: 0,
            last_partial: String::new(),
        })
    }

    async fn ingest_chunk(
        &mut self,
        audio: AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        let started_at = Instant::now();
        let normalized: Vec<f32> = audio
            .into_samples()
            .into_iter()
            .map(|sample| sample as f32 / 32768.0)
            .collect();
        if normalized.is_empty() {
            return Ok(None);
        }

        self.samples_seen += normalized.len();
        self.stream
            .accept_waveform(TARGET_SAMPLE_RATE_HZ as i32, &normalized);
        let update = self.decode_available(false)?;

        {
            let mut state = lock_metrics(&self.metrics, "sherpa-onnx-decode");
            observe_latency(&mut state.runtime, started_at.elapsed());
        }

        Ok(non_empty_update(update))
    }

    async fn finish_stream(mut self) -> Result<TranscriptionUpdate, ModelError> {
        self.stream.input_finished();
        self.decode_available(true)
    }

    fn decode_available(&mut self, force_final: bool) -> Result<TranscriptionUpdate, ModelError> {
        let mut segments = Vec::new();

        let (endpoint, result) = {
            let recognizer = self
                .runtime
                .recognizer
                .lock()
                .map_err(|_| ModelError::Internal("sherpa recognizer mutex poisoned".into()))?;
            while recognizer.is_ready(&self.stream) {
                recognizer.decode(&self.stream);
            }

            let endpoint = recognizer.is_endpoint(&self.stream);
            let result = recognizer.get_result(&self.stream);

            if endpoint {
                recognizer.reset(&self.stream);
            }

            (endpoint, result)
        };

        if let Some(result) = result {
            if let Some(segment) = self.segment_from_result(&result, endpoint || force_final) {
                segments.push(segment);
            }
        }

        if endpoint {
            self.last_partial.clear();
        }

        Ok(TranscriptionUpdate { segments })
    }

    fn segment_from_result(
        &mut self,
        result: &RecognizerResult,
        final_segment: bool,
    ) -> Option<TranscriptSegment> {
        let text = result.text.trim();
        if text.is_empty() {
            return None;
        }

        if final_segment || result.is_final {
            self.last_partial.clear();
            return Some(TranscriptSegment {
                start_ms: result_start_ms(result),
                end_ms: self.audio_position_ms(),
                text: text.to_string(),
                final_segment: true,
            });
        }

        if !self.params.emit_partials || text == self.last_partial {
            return None;
        }
        self.last_partial = text.to_string();

        Some(TranscriptSegment {
            start_ms: result_start_ms(result),
            end_ms: self.audio_position_ms(),
            text: text.to_string(),
            final_segment: false,
        })
    }

    fn audio_position_ms(&self) -> u64 {
        (self.samples_seen as u64 * 1_000) / TARGET_SAMPLE_RATE_HZ as u64
    }
}

impl StreamingTranscriber for SherpaOnnxHandle {
    type Input = AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>;
    type Session = SherpaOnnxStream;

    async fn open_session(&self, params: TranscriptionParams) -> Result<Self::Session, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        SherpaOnnxStream::new(runtime, metrics, params)
    }
}

impl TranscriptionSession for SherpaOnnxStream {
    type Input = AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>;

    async fn ingest(
        &mut self,
        audio: Self::Input,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        self.ingest_chunk(audio).await
    }

    async fn finish(self) -> Result<TranscriptionUpdate, ModelError> {
        self.finish_stream().await
    }
}

fn result_start_ms(result: &RecognizerResult) -> u64 {
    result
        .start_time
        .map(|start| start.max(0.0) * 1_000.0)
        .unwrap_or_default() as u64
}

fn non_empty_update(update: TranscriptionUpdate) -> Option<TranscriptionUpdate> {
    if update.segments.is_empty() {
        None
    } else {
        Some(update)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_reports_backend_metadata() {
        let adapter = SherpaOnnxStreamingAdapter::zipformer_en_streaming();

        assert_eq!(adapter.supported_formats(), &[CheckpointFormat::Onnx]);
        assert_eq!(adapter.backend_kind(), BackendKind::SherpaOnnx);
        assert!(adapter
            .capabilities()
            .supports(CapabilityKind::Transcription));
        assert_eq!(adapter.quantization(), &QuantizationSupport::none());
    }

    #[test]
    fn empty_update_maps_to_none() {
        assert!(non_empty_update(TranscriptionUpdate::default()).is_none());
    }

    #[test]
    fn result_start_time_maps_to_milliseconds() {
        let result = RecognizerResult {
            text: "hello".into(),
            tokens: vec![],
            timestamps: None,
            segment: None,
            start_time: Some(1.25),
            is_final: false,
        };

        assert_eq!(result_start_ms(&result), 1_250);
    }

    #[test]
    fn negative_result_start_time_saturates_to_zero() {
        let result = RecognizerResult {
            text: "hello".into(),
            tokens: vec![],
            timestamps: None,
            segment: None,
            start_time: Some(-0.25),
            is_final: false,
        };

        assert_eq!(result_start_ms(&result), 0);
    }
}
