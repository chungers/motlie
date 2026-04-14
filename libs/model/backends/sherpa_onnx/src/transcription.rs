use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, PcmChunk,
    PcmEncoding, QuantizationSupport, ResolvedCheckpoint, StartOptions, TranscriptSegment,
    TranscriptionModel, TranscriptionParams, TranscriptionStream, TranscriptionUpdate,
};
use ndarray::{ArrayViewD, Ix3};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_onnx_artifacts, RuntimeMetricState, SherpaArtifactPaths, SherpaArtifactSpec,
};

const SHERPA_ONNX_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
const TARGET_FRAME_SHIFT_MS: u64 = 40;
const TAIL_PADDING_SAMPLES: usize = 4_800;
const BLANK_ID: i64 = 0;

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
            capabilities: Capabilities::transcription_stream_only(),
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
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
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
            configure_artifact_policy(self.artifacts.artifact_spec(), policy)?
        } else {
            let root = PathBuf::from(".");
            SherpaArtifactPaths {
                encoder: root.join(self.artifacts.encoder_filename),
                decoder: root.join(self.artifacts.decoder_filename),
                joiner: root.join(self.artifacts.joiner_filename),
                tokens: root.join(self.artifacts.tokens_filename),
            }
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

struct SherpaOnnxHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<SherpaOnnxRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct AsrMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for SherpaOnnxHandle {
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
impl TranscriptionModel for SherpaOnnxHandle {
    async fn open_stream(
        &self,
        spec: AudioSpec,
        params: TranscriptionParams,
    ) -> Result<Box<dyn TranscriptionStream>, ModelError> {
        if spec.channels == 0 {
            return Err(ModelError::InvalidConfiguration(
                "transcription stream requires at least one channel".into(),
            ));
        }
        if spec.sample_rate_hz == 0 {
            return Err(ModelError::InvalidConfiguration(
                "transcription stream requires a non-zero sample rate".into(),
            ));
        }

        Ok(Box::new(SherpaOnnxStream::new(
            Arc::clone(&self.runtime),
            Arc::clone(&self.metrics),
            spec,
            params,
        )?))
    }
}

fn new_transcription_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<SherpaOnnxRuntime>,
) -> Box<dyn BundleHandle> {
    let metrics = Arc::new(Mutex::new(AsrMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "sherpa-onnx-start");
        observe_memory(&mut state.runtime);
    }

    Box::new(SherpaOnnxHandle {
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

struct SherpaOnnxRuntime {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    joiner: Mutex<Session>,
    config: ZipformerConfig,
    tokens: TokenTable,
}

#[derive(Clone, Debug)]
struct ZipformerConfig {
    feature_dim: usize,
    context_size: usize,
    chunk_size: usize,
    chunk_shift: usize,
    frame_shift_ms: u64,
    unk_id: Option<i64>,
    state_layout: StateLayout,
}

#[derive(Clone, Debug)]
struct StateLayout {
    encoder_dims: Vec<usize>,
    attention_dims: Vec<usize>,
    num_encoder_layers: Vec<usize>,
    cnn_module_kernels: Vec<usize>,
    left_context_len: Vec<usize>,
}

#[derive(Clone, Debug)]
struct TokenTable {
    id_to_piece: HashMap<i64, String>,
    unk_id: Option<i64>,
}

impl TokenTable {
    fn from_path(path: &Path) -> Result<Self, ModelError> {
        let raw =
            std::fs::read_to_string(path).map_err(|err| ModelError::BackendInitialization {
                backend: "sherpa-onnx",
                message: format!("failed to read tokens from `{}`: {err}", path.display()),
            })?;

        let mut id_to_piece = HashMap::new();
        let mut unk_id = None;

        for (line_no, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            let piece = parts
                .next()
                .ok_or_else(|| ModelError::BackendInitialization {
                    backend: "sherpa-onnx",
                    message: format!(
                        "invalid tokens line {} in `{}`: missing token piece",
                        line_no + 1,
                        path.display()
                    ),
                })?;
            let id = parts
                .next()
                .ok_or_else(|| ModelError::BackendInitialization {
                    backend: "sherpa-onnx",
                    message: format!(
                        "invalid tokens line {} in `{}`: missing token id",
                        line_no + 1,
                        path.display()
                    ),
                })?
                .parse::<i64>()
                .map_err(|err| ModelError::BackendInitialization {
                    backend: "sherpa-onnx",
                    message: format!(
                        "invalid token id on line {} in `{}`: {err}",
                        line_no + 1,
                        path.display()
                    ),
                })?;

            if piece == "<unk>" {
                unk_id = Some(id);
            }

            id_to_piece.insert(id, piece.to_string());
        }

        Ok(Self {
            id_to_piece,
            unk_id,
        })
    }

    fn piece(&self, id: i64) -> Option<&str> {
        self.id_to_piece.get(&id).map(String::as_str)
    }
}

fn load_runtime(artifacts: &SherpaArtifactPaths) -> Result<SherpaOnnxRuntime, ModelError> {
    let encoder = build_session(&artifacts.encoder)?;
    let config = ZipformerConfig::from_sessions(&encoder, &build_session(&artifacts.decoder)?)?;
    let decoder = build_session(&artifacts.decoder)?;
    let joiner = build_session(&artifacts.joiner)?;
    let tokens = TokenTable::from_path(&artifacts.tokens)?;

    Ok(SherpaOnnxRuntime {
        encoder: Mutex::new(encoder),
        decoder: Mutex::new(decoder),
        joiner: Mutex::new(joiner),
        config: ZipformerConfig {
            unk_id: tokens.unk_id,
            ..config
        },
        tokens,
    })
}

fn build_session(model_path: &Path) -> Result<Session, ModelError> {
    let mut builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
        backend: "sherpa-onnx",
        message: format!("failed to create ONNX Runtime session builder: {err}"),
    })?;

    #[cfg(feature = "cuda")]
    let builder = builder
        .with_execution_providers([ort::ep::CUDA::default().build()])
        .map_err(|err| ModelError::BackendInitialization {
            backend: "sherpa-onnx",
            message: format!("failed to configure CUDA execution provider: {err}"),
        })?;

    builder
        .commit_from_file(model_path)
        .map_err(|err| ModelError::BackendInitialization {
            backend: "sherpa-onnx",
            message: format!(
                "failed to load ONNX model from `{}`: {err}",
                model_path.display()
            ),
        })
}

impl ZipformerConfig {
    fn from_sessions(encoder: &Session, decoder: &Session) -> Result<Self, ModelError> {
        let encoder_meta = encoder
            .metadata()
            .map_err(|err| ModelError::BackendInitialization {
                backend: "sherpa-onnx",
                message: format!("failed to inspect encoder metadata: {err}"),
            })?;
        let decoder_meta = decoder
            .metadata()
            .map_err(|err| ModelError::BackendInitialization {
                backend: "sherpa-onnx",
                message: format!("failed to inspect decoder metadata: {err}"),
            })?;

        let state_layout = StateLayout {
            encoder_dims: parse_i32_list(metadata_value(&encoder_meta, "encoder_dims")?)?,
            attention_dims: parse_i32_list(metadata_value(&encoder_meta, "attention_dims")?)?,
            num_encoder_layers: parse_i32_list(metadata_value(
                &encoder_meta,
                "num_encoder_layers",
            )?)?,
            cnn_module_kernels: parse_i32_list(metadata_value(
                &encoder_meta,
                "cnn_module_kernels",
            )?)?,
            left_context_len: parse_i32_list(metadata_value(&encoder_meta, "left_context_len")?)?,
        };

        Ok(Self {
            feature_dim: 80,
            context_size: parse_i32_scalar(metadata_value(&decoder_meta, "context_size")?)?,
            chunk_size: parse_i32_scalar(metadata_value(&encoder_meta, "T")?)?,
            chunk_shift: parse_i32_scalar(metadata_value(&encoder_meta, "decode_chunk_len")?)?,
            frame_shift_ms: TARGET_FRAME_SHIFT_MS,
            unk_id: None,
            state_layout,
        })
    }
}

fn metadata_value(
    metadata: &ort::session::ModelMetadata<'_>,
    key: &str,
) -> Result<String, ModelError> {
    metadata
        .custom(key)
        .ok_or_else(|| ModelError::BackendInitialization {
            backend: "sherpa-onnx",
            message: format!("missing required ONNX metadata key `{key}`"),
        })
}

fn parse_i32_scalar(value: String) -> Result<usize, ModelError> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|err| ModelError::BackendInitialization {
            backend: "sherpa-onnx",
            message: format!("invalid ONNX metadata scalar `{value}`: {err}"),
        })
}

fn parse_i32_list(value: String) -> Result<Vec<usize>, ModelError> {
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|err| ModelError::BackendInitialization {
                    backend: "sherpa-onnx",
                    message: format!("invalid ONNX metadata list `{value}`: {err}"),
                })
        })
        .collect()
}

struct SherpaOnnxStream {
    runtime: Arc<SherpaOnnxRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
    spec: AudioSpec,
    params: TranscriptionParams,
    normalizer: NormalizerState,
    features: OnlineFeature,
    decoder_state: DecoderState,
    assembler: SegmentAssembler,
    last_sequence: Option<u64>,
    end_of_stream_received: bool,
}

impl SherpaOnnxStream {
    fn new(
        runtime: Arc<SherpaOnnxRuntime>,
        metrics: Arc<Mutex<AsrMetrics>>,
        spec: AudioSpec,
        params: TranscriptionParams,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            features: new_feature_extractor()?,
            decoder_state: DecoderState::new(&runtime.config)?,
            runtime,
            metrics,
            spec,
            params,
            normalizer: NormalizerState::new(),
            assembler: SegmentAssembler::default(),
            last_sequence: None,
            end_of_stream_received: false,
        })
    }

    fn is_ready(&self) -> bool {
        self.decoder_state.processed_frames + self.runtime.config.chunk_size
            < self.features.num_frames_ready()
    }

    fn accept_samples(&mut self, samples: &[f32]) {
        if !samples.is_empty() {
            self.features
                .accept_waveform(TARGET_SAMPLE_RATE_HZ as f32, samples);
        }
    }

    fn decode_ready_chunks(&mut self) -> Result<Vec<TranscriptSegment>, ModelError> {
        let mut emitted = Vec::new();
        while self.is_ready() {
            emitted.extend(self.decode_next_chunk()?);
        }
        Ok(emitted)
    }

    fn decode_next_chunk(&mut self) -> Result<Vec<TranscriptSegment>, ModelError> {
        let started_at = Instant::now();
        let chunk_size = self.runtime.config.chunk_size;
        let chunk_shift = self.runtime.config.chunk_shift;
        let feature_dim = self.runtime.config.feature_dim;
        let context_size = self.runtime.config.context_size;
        let frame_shift_ms = self.runtime.config.frame_shift_ms;

        let features = collect_feature_chunk(
            &self.features,
            self.decoder_state.processed_frames,
            chunk_size,
            feature_dim,
        )?;

        let feature_tensor = Tensor::<f32>::from_array((
            vec![1_i64, chunk_size as i64, feature_dim as i64],
            features,
        ))
        .map_err(ort_tensor_error)?;

        let encoder_out = {
            let mut inputs = Vec::with_capacity(1 + self.decoder_state.encoder_state.len());
            inputs.push(SessionInputValue::from(feature_tensor));
            for state in &self.decoder_state.encoder_state {
                inputs.push(SessionInputValue::from(state));
            }

            let mut encoder = self
                .runtime
                .encoder
                .lock()
                .map_err(|_| ModelError::Internal("encoder session mutex poisoned".into()))?;
            let outputs = encoder
                .run(inputs.as_slice())
                .map_err(ort_run_error("encoder"))?;
            split_encoder_outputs(outputs, &mut self.decoder_state.encoder_state)?
        };

        self.run_greedy_decoder(encoder_out)?;
        self.decoder_state.processed_frames += chunk_shift;

        {
            let mut state = lock_metrics(&self.metrics, "sherpa-onnx-decode");
            observe_latency(&mut state.runtime, started_at.elapsed());
        }

        self.assembler.consume(
            &self.decoder_state.tokens[context_size..],
            &self.decoder_state.timestamps,
            &self.runtime.tokens,
            frame_shift_ms,
            self.params.emit_partials,
            false,
        )
    }

    fn run_greedy_decoder(&mut self, encoder_out: DynValue) -> Result<(), ModelError> {
        let chunk_size = self.runtime.config.chunk_size;
        let unk_id = self.runtime.config.unk_id;
        let (_, raw) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(ort_extract_error)?;
        let frame_view = ArrayViewD::from_shape(vec![1, chunk_size, raw.len() / chunk_size], raw)
            .map_err(|err| ModelError::BackendExecution {
                backend: "sherpa-onnx",
                operation: "encoder_out_shape",
                message: err.to_string(),
            })?
            .into_dimensionality::<Ix3>()
            .map_err(|err| ModelError::BackendExecution {
                backend: "sherpa-onnx",
                operation: "encoder_out_dimensionality",
                message: err.to_string(),
            })?;

        if self.decoder_state.decoder_out.is_none() {
            self.decoder_state.decoder_out = Some(self.run_decoder()?);
        }

        for frame_idx in 0..frame_view.shape()[1] {
            let frame = frame_view
                .index_axis(ndarray::Axis(1), frame_idx)
                .to_owned();
            let (frame_data, _offset) = frame.into_raw_vec_and_offset();
            let frame_tensor =
                Tensor::<f32>::from_array((vec![1_i64, frame_data.len() as i64], frame_data))
                    .map_err(ort_tensor_error)?;

            let decoder_out = self.decoder_state.decoder_out.as_ref().ok_or_else(|| {
                ModelError::Internal("decoder cache should be initialized".into())
            })?;
            let logits = {
                let mut joiner =
                    self.runtime.joiner.lock().map_err(|_| {
                        ModelError::Internal("joiner session mutex poisoned".into())
                    })?;
                let outputs = joiner
                    .run([
                        SessionInputValue::from(frame_tensor),
                        SessionInputValue::from(decoder_out),
                    ])
                    .map_err(ort_run_error("joiner"))?;
                first_output(outputs, "joiner")?
            };

            let (_, logits) = logits
                .try_extract_tensor::<f32>()
                .map_err(ort_extract_error)?;
            let best = argmax(logits).ok_or_else(|| ModelError::BackendExecution {
                backend: "sherpa-onnx",
                operation: "joiner_argmax",
                message: "joiner produced an empty logits tensor".into(),
            })? as i64;

            if best != BLANK_ID && Some(best) != unk_id {
                self.decoder_state.tokens.push(best);
                self.decoder_state
                    .timestamps
                    .push((self.decoder_state.frame_offset + frame_idx) as i32);
                self.decoder_state.num_trailing_blanks = 0;
                self.decoder_state.decoder_out = Some(self.run_decoder()?);
            } else {
                self.decoder_state.num_trailing_blanks += 1;
            }
        }

        self.decoder_state.frame_offset += frame_view.shape()[1];
        Ok(())
    }

    fn run_decoder(&mut self) -> Result<DynValue, ModelError> {
        let context_size = self.runtime.config.context_size;
        let start = self.decoder_state.tokens.len().saturating_sub(context_size);
        let input = self.decoder_state.tokens[start..].to_vec();
        let tensor = Tensor::<i64>::from_array((vec![1_i64, context_size as i64], input))
            .map_err(ort_tensor_error)?;

        let mut decoder = self
            .runtime
            .decoder
            .lock()
            .map_err(|_| ModelError::Internal("decoder session mutex poisoned".into()))?;
        let outputs = decoder
            .run([SessionInputValue::from(tensor)])
            .map_err(ort_run_error("decoder"))?;
        first_output(outputs, "decoder")
    }
}

#[async_trait]
impl TranscriptionStream for SherpaOnnxStream {
    async fn push_chunk(
        &mut self,
        chunk: PcmChunk,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        if self.end_of_stream_received {
            return Err(ModelError::InvalidConfiguration(
                "push_chunk called after end_of_stream was received".into(),
            ));
        }

        if let Some(last) = self.last_sequence {
            if chunk.sequence <= last {
                return Err(ModelError::InvalidConfiguration(format!(
                    "non-monotonic chunk sequence: got {}, last was {last}",
                    chunk.sequence
                )));
            }
        }
        self.last_sequence = Some(chunk.sequence);

        if chunk.data.is_empty() && !chunk.end_of_stream {
            return Ok(None);
        }

        if !chunk.data.is_empty() {
            let normalized = self.normalizer.normalize(&chunk.data, &self.spec)?;
            self.accept_samples(&normalized);
        }

        if chunk.end_of_stream {
            self.end_of_stream_received = true;
        }

        let segments = self.decode_ready_chunks()?;
        if segments.is_empty() {
            return Ok(None);
        }

        Ok(Some(TranscriptionUpdate { segments }))
    }

    async fn finish(mut self: Box<Self>) -> Result<TranscriptionUpdate, ModelError> {
        let tail = self.normalizer.flush();
        self.accept_samples(&tail);
        self.accept_samples(&vec![0.0; TAIL_PADDING_SAMPLES]);
        self.features.input_finished();

        let mut segments = self.decode_ready_chunks()?;
        segments.extend(self.assembler.consume(
            &self.decoder_state.tokens[self.runtime.config.context_size..],
            &self.decoder_state.timestamps,
            &self.runtime.tokens,
            self.runtime.config.frame_shift_ms,
            false,
            true,
        )?);

        Ok(TranscriptionUpdate { segments })
    }
}

fn new_feature_extractor() -> Result<OnlineFeature, ModelError> {
    let options = FbankOptions {
        use_energy: false,
        frame_opts: kaldi_native_fbank::FrameOptions {
            snip_edges: false,
            dither: 0.0,
            samp_freq: TARGET_SAMPLE_RATE_HZ as f32,
            ..Default::default()
        },
        mel_opts: MelOptions {
            num_bins: 80,
            ..Default::default()
        },
        ..Default::default()
    };

    let computer =
        FbankComputer::new(options).map_err(|err| ModelError::BackendInitialization {
            backend: "sherpa-onnx",
            message: format!("failed to initialize fbank frontend: {err}"),
        })?;

    Ok(OnlineFeature::new(FeatureComputer::Fbank(computer)))
}

fn collect_feature_chunk(
    features: &OnlineFeature,
    start: usize,
    len: usize,
    feature_dim: usize,
) -> Result<Vec<f32>, ModelError> {
    let mut out = Vec::with_capacity(len * feature_dim);
    for frame_index in start..start + len {
        let frame =
            features
                .get_frame(frame_index)
                .ok_or_else(|| ModelError::BackendExecution {
                    backend: "sherpa-onnx",
                    operation: "get_frame",
                    message: format!("requested unavailable feature frame {frame_index}"),
                })?;
        out.extend_from_slice(frame);
    }
    Ok(out)
}

fn split_encoder_outputs(
    outputs: ort::session::SessionOutputs<'_>,
    state_out: &mut Vec<DynValue>,
) -> Result<DynValue, ModelError> {
    let mut iter = outputs.into_iter();
    let (_, encoder_out) = iter.next().ok_or_else(|| ModelError::BackendExecution {
        backend: "sherpa-onnx",
        operation: "encoder_outputs",
        message: "encoder returned no outputs".into(),
    })?;

    *state_out = iter.map(|(_, value)| value).collect();
    Ok(encoder_out)
}

fn first_output(
    outputs: ort::session::SessionOutputs<'_>,
    operation: &'static str,
) -> Result<DynValue, ModelError> {
    outputs
        .into_iter()
        .next()
        .map(|(_, value)| value)
        .ok_or_else(|| ModelError::BackendExecution {
            backend: "sherpa-onnx",
            operation,
            message: "session returned no outputs".into(),
        })
}

fn ort_tensor_error(err: ort::Error) -> ModelError {
    ModelError::BackendExecution {
        backend: "sherpa-onnx",
        operation: "tensor_create",
        message: err.to_string(),
    }
}

fn ort_extract_error(err: ort::Error) -> ModelError {
    ModelError::BackendExecution {
        backend: "sherpa-onnx",
        operation: "tensor_extract",
        message: err.to_string(),
    }
}

fn ort_run_error(operation: &'static str) -> impl FnOnce(ort::Error) -> ModelError {
    move |err| ModelError::BackendExecution {
        backend: "sherpa-onnx",
        operation,
        message: err.to_string(),
    }
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}

struct DecoderState {
    processed_frames: usize,
    frame_offset: usize,
    num_trailing_blanks: usize,
    tokens: Vec<i64>,
    timestamps: Vec<i32>,
    encoder_state: Vec<DynValue>,
    decoder_out: Option<DynValue>,
}

impl DecoderState {
    fn new(config: &ZipformerConfig) -> Result<Self, ModelError> {
        let mut tokens = vec![-1; config.context_size];
        if let Some(last) = tokens.last_mut() {
            *last = BLANK_ID;
        }

        Ok(Self {
            processed_frames: 0,
            frame_offset: 0,
            num_trailing_blanks: 0,
            tokens,
            timestamps: Vec::new(),
            encoder_state: initial_encoder_state(&config.state_layout)?,
            decoder_out: None,
        })
    }
}

fn initial_encoder_state(layout: &StateLayout) -> Result<Vec<DynValue>, ModelError> {
    let mut state = Vec::new();

    for i in 0..layout.encoder_dims.len() {
        let layers = layout.num_encoder_layers[i];
        let encoder_dim = layout.encoder_dims[i];
        let attention_dim = layout.attention_dims[i];
        let left_context = layout.left_context_len[i];
        let cnn_kernel = layout.cnn_module_kernels[i];

        state.push(
            Tensor::<i64>::from_array((vec![layers as i64, 1], vec![0_i64; layers]))
                .map_err(ort_tensor_error)?
                .into_dyn(),
        );
        state.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, 1, encoder_dim as i64],
                vec![0.0_f32; layers * encoder_dim],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        state.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, left_context as i64, 1, attention_dim as i64],
                vec![0.0_f32; layers * left_context * attention_dim],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        let half_attention = attention_dim / 2;
        state.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, left_context as i64, 1, half_attention as i64],
                vec![0.0_f32; layers * left_context * half_attention],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        state.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, left_context as i64, 1, half_attention as i64],
                vec![0.0_f32; layers * left_context * half_attention],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        state.push(
            Tensor::<f32>::from_array((
                vec![
                    layers as i64,
                    1,
                    encoder_dim as i64,
                    (cnn_kernel.saturating_sub(1)) as i64,
                ],
                vec![0.0_f32; layers * encoder_dim * cnn_kernel.saturating_sub(1)],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        state.push(
            Tensor::<f32>::from_array((
                vec![
                    layers as i64,
                    1,
                    encoder_dim as i64,
                    (cnn_kernel.saturating_sub(1)) as i64,
                ],
                vec![0.0_f32; layers * encoder_dim * cnn_kernel.saturating_sub(1)],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
    }

    Ok(state)
}

#[derive(Clone, Debug)]
struct PendingPiece {
    text: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Default)]
struct SegmentAssembler {
    surfaced_tokens: usize,
    pending_word: Vec<PendingPiece>,
}

impl SegmentAssembler {
    fn consume(
        &mut self,
        tokens: &[i64],
        timestamps: &[i32],
        table: &TokenTable,
        frame_shift_ms: u64,
        emit_partials: bool,
        final_flush: bool,
    ) -> Result<Vec<TranscriptSegment>, ModelError> {
        if tokens.len() != timestamps.len() {
            return Err(ModelError::BackendExecution {
                backend: "sherpa-onnx",
                operation: "segment_assembly",
                message: format!(
                    "token/timestamp length mismatch: {} tokens vs {} timestamps",
                    tokens.len(),
                    timestamps.len()
                ),
            });
        }

        if self.surfaced_tokens > tokens.len() {
            return Err(ModelError::BackendExecution {
                backend: "sherpa-onnx",
                operation: "segment_assembly",
                message: "stream token count regressed".into(),
            });
        }

        let mut segments = Vec::new();
        for index in self.surfaced_tokens..tokens.len() {
            let piece = table
                .piece(tokens[index])
                .ok_or_else(|| ModelError::BackendExecution {
                    backend: "sherpa-onnx",
                    operation: "token_lookup",
                    message: format!("missing token id {}", tokens[index]),
                })?;

            if piece == "<unk>" || piece == "<blk>" || piece == "<sos/eos>" {
                continue;
            }

            if piece.starts_with('▁') && !self.pending_word.is_empty() {
                segments.push(finalize_word(&mut self.pending_word, true));
            }

            let start_ms = timestamps[index] as u64 * frame_shift_ms;
            let end_ms = start_ms + frame_shift_ms;
            self.pending_word.push(PendingPiece {
                text: piece.to_string(),
                start_ms,
                end_ms,
            });
        }

        self.surfaced_tokens = tokens.len();

        if final_flush && !self.pending_word.is_empty() {
            segments.push(finalize_word(&mut self.pending_word, true));
        } else if emit_partials && !self.pending_word.is_empty() {
            segments.push(render_word(&self.pending_word, false));
        }

        Ok(segments)
    }
}

fn finalize_word(pieces: &mut Vec<PendingPiece>, final_segment: bool) -> TranscriptSegment {
    let segment = render_word(pieces, final_segment);
    pieces.clear();
    segment
}

fn render_word(pieces: &[PendingPiece], final_segment: bool) -> TranscriptSegment {
    let start_ms = pieces.first().map(|p| p.start_ms).unwrap_or_default();
    let end_ms = pieces.last().map(|p| p.end_ms).unwrap_or(start_ms);
    let mut text = String::new();
    for piece in pieces {
        text.push_str(&piece.text);
    }
    let text = text.replace('▁', " ").trim().to_string();

    TranscriptSegment {
        start_ms,
        end_ms,
        text,
        final_segment,
    }
}

struct NormalizerState {
    pending_bytes: Vec<u8>,
    pending_channel_samples: Vec<f32>,
    resample_cursor: f64,
    resampling_active: bool,
    carry_sample: Option<f32>,
}

impl NormalizerState {
    fn new() -> Self {
        Self {
            pending_bytes: Vec::new(),
            pending_channel_samples: Vec::new(),
            resample_cursor: 0.0,
            resampling_active: false,
            carry_sample: None,
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        if self.resampling_active {
            self.carry_sample.take().into_iter().collect()
        } else {
            Vec::new()
        }
    }

    fn normalize(&mut self, data: &[u8], spec: &AudioSpec) -> Result<Vec<f32>, ModelError> {
        let mut raw = std::mem::take(&mut self.pending_bytes);
        raw.extend_from_slice(data);

        let sample_bytes = match spec.encoding {
            PcmEncoding::S16Le => 2,
            PcmEncoding::F32Le => 4,
        };
        let aligned_len = raw.len() - (raw.len() % sample_bytes);
        self.pending_bytes = raw[aligned_len..].to_vec();
        let aligned = &raw[..aligned_len];

        let decoded: Vec<f32> = match spec.encoding {
            PcmEncoding::S16Le => aligned
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
                .collect(),
            PcmEncoding::F32Le => aligned
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect(),
        };

        let mono = if spec.channels <= 1 {
            decoded
        } else {
            let channels = spec.channels as usize;
            let mut samples = std::mem::take(&mut self.pending_channel_samples);
            samples.extend(decoded);

            let complete_frames = samples.len() / channels;
            let used = complete_frames * channels;
            self.pending_channel_samples = samples[used..].to_vec();

            samples[..used]
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        };

        if spec.sample_rate_hz == TARGET_SAMPLE_RATE_HZ || mono.is_empty() {
            self.resampling_active = false;
            if let Some(&last) = mono.last() {
                self.carry_sample = Some(last);
            }
            return Ok(mono);
        }

        self.resampling_active = true;
        let samples: Vec<f32> = if let Some(carry) = self.carry_sample {
            let mut combined = Vec::with_capacity(mono.len() + 1);
            combined.push(carry);
            combined.extend_from_slice(&mono);
            combined
        } else {
            mono
        };

        let ratio = spec.sample_rate_hz as f64 / TARGET_SAMPLE_RATE_HZ as f64;
        let mut output = Vec::new();
        let mut cursor = self.resample_cursor;

        while (cursor as usize) + 1 < samples.len() {
            let index = cursor as usize;
            let frac = cursor - index as f64;
            let interpolated =
                samples[index] as f64 * (1.0 - frac) + samples[index + 1] as f64 * frac;
            output.push(interpolated as f32);
            cursor += ratio;
        }

        let last_index = samples.len().saturating_sub(1) as f64;
        self.resample_cursor = (cursor - last_index).max(0.0);
        self.carry_sample = samples.last().copied();

        Ok(output)
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
    fn initial_decoder_state_contains_context_padding() {
        let state = DecoderState::new(&ZipformerConfig {
            feature_dim: 80,
            context_size: 2,
            chunk_size: 16,
            chunk_shift: 8,
            frame_shift_ms: TARGET_FRAME_SHIFT_MS,
            unk_id: Some(1),
            state_layout: StateLayout {
                encoder_dims: vec![4],
                attention_dims: vec![4],
                num_encoder_layers: vec![2],
                cnn_module_kernels: vec![3],
                left_context_len: vec![2],
            },
        })
        .expect("decoder state should initialize");

        assert_eq!(state.tokens, vec![-1, BLANK_ID]);
        assert!(state.timestamps.is_empty());
        assert_eq!(state.encoder_state.len(), 7);
    }

    #[test]
    fn segment_assembler_finalizes_previous_word_on_new_boundary() {
        let mut table = HashMap::new();
        table.insert(10, "▁hello".to_string());
        table.insert(11, "world".to_string());
        table.insert(12, "▁again".to_string());

        let mut assembler = SegmentAssembler::default();
        let segments = assembler
            .consume(
                &[10, 11, 12],
                &[0, 1, 2],
                &TokenTable {
                    id_to_piece: table,
                    unk_id: None,
                },
                TARGET_FRAME_SHIFT_MS,
                true,
                false,
            )
            .expect("segment assembly should succeed");

        assert_eq!(segments.len(), 2);
        assert!(segments[0].final_segment);
        assert_eq!(segments[0].text, "helloworld");
        assert!(!segments[1].final_segment);
        assert_eq!(segments[1].text, "again");
    }
}
