use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use motlie_model::typed::{AudioBuf, Mono, StreamingTranscriber, TranscriptionSession};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationSupport, ResolvedCheckpoint, StartOptions,
    TranscriptSegment, TranscriptionParams, TranscriptionUpdate, UnsupportedChat,
    UnsupportedCompletion, UnsupportedEmbeddings,
};
use motlie_model_ort::build_session;
use ndarray::ArrayView3;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};

use crate::common::{
    RuntimeMetricState, SherpaArtifactPaths, SherpaArtifactSpec, configure_artifact_policy,
    lock_metrics, observe_latency, observe_memory, resolve_onnx_artifacts,
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
    encoder_input_frames: usize,
    decode_chunk_len: usize,
    frame_shift_ms: u64,
    unk_id: Option<i64>,
    state_layout: StateLayout,
}

#[derive(Clone, Debug)]
enum StateLayout {
    Zipformer { stacks: Vec<EncoderStackSpec> },
    Zipformer2 { stacks: Vec<Zipformer2StackSpec> },
}

#[derive(Clone, Debug)]
struct EncoderStackSpec {
    encoder_dim: usize,
    attention_dim: usize,
    num_layers: usize,
    cnn_module_kernel: usize,
    left_context_len: usize,
}

#[derive(Clone, Debug)]
struct Zipformer2StackSpec {
    encoder_dim: usize,
    query_head_dim: usize,
    value_head_dim: usize,
    num_heads: usize,
    num_layers: usize,
    cnn_module_kernel: usize,
    left_context_len: usize,
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
    let encoder = build_session("sherpa-onnx", &artifacts.encoder)?;
    let config = ZipformerConfig::from_sessions(
        &encoder,
        &build_session("sherpa-onnx", &artifacts.decoder)?,
    )?;
    let decoder = build_session("sherpa-onnx", &artifacts.decoder)?;
    let joiner = build_session("sherpa-onnx", &artifacts.joiner)?;
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

        let state_layout = StateLayout::from_metadata(&encoder_meta)?;

        Ok(Self {
            feature_dim: 80,
            context_size: parse_i32_scalar(metadata_value_required(
                &decoder_meta,
                "context_size",
            )?)?,
            encoder_input_frames: parse_i32_scalar(metadata_value_required(&encoder_meta, "T")?)?,
            decode_chunk_len: parse_i32_scalar(metadata_value_required(
                &encoder_meta,
                "decode_chunk_len",
            )?)?,
            frame_shift_ms: TARGET_FRAME_SHIFT_MS,
            unk_id: None,
            state_layout,
        })
    }
}

impl StateLayout {
    fn from_metadata(metadata: &ort::session::ModelMetadata<'_>) -> Result<Self, ModelError> {
        let model_type = metadata_value_optional(metadata, "model_type")?;
        let encoder_dims = parse_i32_list(metadata_value_required(metadata, "encoder_dims")?)?;
        let num_encoder_layers =
            parse_i32_list(metadata_value_required(metadata, "num_encoder_layers")?)?;
        let cnn_module_kernels =
            parse_i32_list(metadata_value_required(metadata, "cnn_module_kernels")?)?;
        let left_context_len =
            parse_i32_list(metadata_value_required(metadata, "left_context_len")?)?;
        let attention_dims = metadata_value_optional(metadata, "attention_dims")?
            .map(parse_i32_list)
            .transpose()?;
        let query_head_dims = metadata_value_optional(metadata, "query_head_dims")?
            .map(parse_i32_list)
            .transpose()?;
        let value_head_dims = metadata_value_optional(metadata, "value_head_dims")?
            .map(parse_i32_list)
            .transpose()?;
        let num_heads = metadata_value_optional(metadata, "num_heads")?
            .map(parse_i32_list)
            .transpose()?;

        let use_zipformer2 = matches!(model_type.as_deref(), Some("zipformer2" | "zipformer2r"))
            || (attention_dims.is_none()
                && query_head_dims.is_some()
                && value_head_dims.is_some()
                && num_heads.is_some());

        if use_zipformer2 {
            return Self::try_from_zipformer2_parts(
                encoder_dims,
                query_head_dims.ok_or_else(|| {
                    invalid_metadata(
                        "zipformer2 encoder metadata missing required key `query_head_dims`",
                    )
                })?,
                value_head_dims.ok_or_else(|| {
                    invalid_metadata(
                        "zipformer2 encoder metadata missing required key `value_head_dims`",
                    )
                })?,
                num_heads.ok_or_else(|| {
                    invalid_metadata("zipformer2 encoder metadata missing required key `num_heads`")
                })?,
                num_encoder_layers,
                cnn_module_kernels,
                left_context_len,
            );
        }

        Self::try_from_zipformer_parts(
            encoder_dims,
            attention_dims.ok_or_else(|| {
                invalid_metadata("zipformer encoder metadata missing required key `attention_dims`")
            })?,
            num_encoder_layers,
            cnn_module_kernels,
            left_context_len,
        )
    }

    fn try_from_zipformer_parts(
        encoder_dims: Vec<usize>,
        attention_dims: Vec<usize>,
        num_encoder_layers: Vec<usize>,
        cnn_module_kernels: Vec<usize>,
        left_context_len: Vec<usize>,
    ) -> Result<Self, ModelError> {
        let stack_count = encoder_dims.len();
        if stack_count == 0 {
            return Err(invalid_metadata(
                "zipformer encoder metadata declared zero encoder stacks",
            ));
        }

        for (name, values) in [
            ("attention_dims", &attention_dims),
            ("num_encoder_layers", &num_encoder_layers),
            ("cnn_module_kernels", &cnn_module_kernels),
            ("left_context_len", &left_context_len),
        ] {
            if values.len() != stack_count {
                return Err(invalid_metadata(format!(
                    "zipformer encoder metadata length mismatch: encoder_dims has {stack_count} entries but {name} has {}",
                    values.len()
                )));
            }
        }

        let mut stacks = Vec::with_capacity(stack_count);
        for (
            stack_index,
            ((((encoder_dim, attention_dim), num_layers), cnn_module_kernel), left_context_len),
        ) in encoder_dims
            .into_iter()
            .zip(attention_dims)
            .zip(num_encoder_layers)
            .zip(cnn_module_kernels)
            .zip(left_context_len)
            .enumerate()
        {
            if cnn_module_kernel == 0 {
                return Err(invalid_metadata(format!(
                    "zipformer encoder metadata stack {stack_index} has cnn_module_kernel=0"
                )));
            }

            stacks.push(EncoderStackSpec {
                encoder_dim,
                attention_dim,
                num_layers,
                cnn_module_kernel,
                left_context_len,
            });
        }

        Ok(Self::Zipformer { stacks })
    }

    fn try_from_zipformer2_parts(
        encoder_dims: Vec<usize>,
        query_head_dims: Vec<usize>,
        value_head_dims: Vec<usize>,
        num_heads: Vec<usize>,
        num_encoder_layers: Vec<usize>,
        cnn_module_kernels: Vec<usize>,
        left_context_len: Vec<usize>,
    ) -> Result<Self, ModelError> {
        let stack_count = encoder_dims.len();
        if stack_count == 0 {
            return Err(invalid_metadata(
                "zipformer2 encoder metadata declared zero encoder stacks",
            ));
        }

        for (name, values) in [
            ("query_head_dims", &query_head_dims),
            ("value_head_dims", &value_head_dims),
            ("num_heads", &num_heads),
            ("num_encoder_layers", &num_encoder_layers),
            ("cnn_module_kernels", &cnn_module_kernels),
            ("left_context_len", &left_context_len),
        ] {
            if values.len() != stack_count {
                return Err(invalid_metadata(format!(
                    "zipformer2 encoder metadata length mismatch: encoder_dims has {stack_count} entries but {name} has {}",
                    values.len()
                )));
            }
        }

        let mut stacks = Vec::with_capacity(stack_count);
        for (
            stack_index,
            (
                (
                    ((((encoder_dim, query_head_dim), value_head_dim), num_heads), num_layers),
                    cnn_module_kernel,
                ),
                left_context_len,
            ),
        ) in encoder_dims
            .into_iter()
            .zip(query_head_dims)
            .zip(value_head_dims)
            .zip(num_heads)
            .zip(num_encoder_layers)
            .zip(cnn_module_kernels)
            .zip(left_context_len)
            .enumerate()
        {
            if query_head_dim == 0 || value_head_dim == 0 || num_heads == 0 {
                return Err(invalid_metadata(format!(
                    "zipformer2 encoder metadata stack {stack_index} has zero-valued attention geometry"
                )));
            }
            if cnn_module_kernel < 2 {
                return Err(invalid_metadata(format!(
                    "zipformer2 encoder metadata stack {stack_index} has cnn_module_kernel={cnn_module_kernel}, expected >= 2"
                )));
            }

            stacks.push(Zipformer2StackSpec {
                encoder_dim,
                query_head_dim,
                value_head_dim,
                num_heads,
                num_layers,
                cnn_module_kernel,
                left_context_len,
            });
        }

        Ok(Self::Zipformer2 { stacks })
    }
}

fn metadata_value_optional(
    metadata: &ort::session::ModelMetadata<'_>,
    key: &str,
) -> Result<Option<String>, ModelError> {
    Ok(metadata.custom(key).and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        }
    }))
}

fn metadata_value_required(
    metadata: &ort::session::ModelMetadata<'_>,
    key: &str,
) -> Result<String, ModelError> {
    metadata_value_optional(metadata, key)?.ok_or_else(|| ModelError::BackendInitialization {
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

fn invalid_metadata(message: impl Into<String>) -> ModelError {
    ModelError::BackendInitialization {
        backend: "sherpa-onnx",
        message: message.into(),
    }
}

pub struct SherpaOnnxStream {
    runtime: Arc<SherpaOnnxRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
    params: TranscriptionParams,
    features: OnlineFeature,
    decoder_state: DecoderState,
    assembler: SegmentAssembler,
}

impl SherpaOnnxStream {
    fn new(
        runtime: Arc<SherpaOnnxRuntime>,
        metrics: Arc<Mutex<AsrMetrics>>,
        params: TranscriptionParams,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            features: new_feature_extractor()?,
            decoder_state: DecoderState::new(&runtime.config)?,
            runtime,
            metrics,
            params,
            assembler: SegmentAssembler::default(),
        })
    }

    fn is_ready(&self) -> bool {
        self.decoder_state.processed_frames + self.runtime.config.encoder_input_frames
            <= self.features.num_frames_ready()
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
        let encoder_input_frames = self.runtime.config.encoder_input_frames;
        let decode_chunk_len = self.runtime.config.decode_chunk_len;
        let feature_dim = self.runtime.config.feature_dim;
        let context_size = self.runtime.config.context_size;
        let frame_shift_ms = self.runtime.config.frame_shift_ms;

        let features = collect_feature_chunk(
            &self.features,
            self.decoder_state.processed_frames,
            encoder_input_frames,
            feature_dim,
        )?;

        let feature_tensor = Tensor::<f32>::from_array((
            vec![1_i64, encoder_input_frames as i64, feature_dim as i64],
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
        self.decoder_state.processed_frames += decode_chunk_len;

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
        let unk_id = self.runtime.config.unk_id;
        let frame_view = extract_encoder_frames(&encoder_out)?;

        if self.decoder_state.decoder_out.is_none() {
            self.decoder_state.decoder_out = Some(self.run_decoder()?);
        }

        let emitted_frames = frame_view.shape()[1];
        for frame_idx in 0..emitted_frames {
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

        self.decoder_state.frame_offset += emitted_frames;
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

impl SherpaOnnxStream {
    async fn ingest_chunk(
        &mut self,
        audio: AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        let normalized: Vec<f32> = audio
            .into_samples()
            .into_iter()
            .map(|sample| sample as f32 / 32768.0)
            .collect();
        if normalized.is_empty() {
            return Ok(None);
        }
        self.accept_samples(&normalized);

        let segments = self.decode_ready_chunks()?;
        if segments.is_empty() {
            return Ok(None);
        }

        Ok(Some(TranscriptionUpdate { segments }))
    }

    async fn finish_stream(mut self) -> Result<TranscriptionUpdate, ModelError> {
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

fn extract_encoder_frames<'a>(
    encoder_out: &'a DynValue,
) -> Result<ArrayView3<'a, f32>, ModelError> {
    let (shape, raw) = encoder_out
        .try_extract_tensor::<f32>()
        .map_err(ort_extract_error)?;

    if shape.len() != 3 || shape[0] != 1 || shape[1] < 0 || shape[2] < 0 {
        return Err(ModelError::BackendExecution {
            backend: "sherpa-onnx",
            operation: "encoder_out_shape",
            message: format!("expected encoder output shape [1, T, C], got {shape}"),
        });
    }

    ArrayView3::from_shape((1, shape[1] as usize, shape[2] as usize), raw).map_err(|err| {
        ModelError::BackendExecution {
            backend: "sherpa-onnx",
            operation: "encoder_out_shape",
            message: err.to_string(),
        }
    })
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
            encoder_state: initial_encoder_state(&config.state_layout, config.feature_dim)?,
            decoder_out: None,
        })
    }
}

fn initial_encoder_state(
    layout: &StateLayout,
    feature_dim: usize,
) -> Result<Vec<DynValue>, ModelError> {
    match layout {
        StateLayout::Zipformer { stacks } => initial_zipformer_state(stacks),
        StateLayout::Zipformer2 { stacks } => initial_zipformer2_state(stacks, feature_dim),
    }
}

fn initial_zipformer_state(stacks: &[EncoderStackSpec]) -> Result<Vec<DynValue>, ModelError> {
    let mut cached_len = Vec::with_capacity(stacks.len());
    let mut cached_avg = Vec::with_capacity(stacks.len());
    let mut cached_key = Vec::with_capacity(stacks.len());
    let mut cached_val = Vec::with_capacity(stacks.len());
    let mut cached_val2 = Vec::with_capacity(stacks.len());
    let mut cached_conv1 = Vec::with_capacity(stacks.len());
    let mut cached_conv2 = Vec::with_capacity(stacks.len());

    for stack in stacks {
        let layers = stack.num_layers;
        let encoder_dim = stack.encoder_dim;
        let attention_dim = stack.attention_dim;
        let left_context = stack.left_context_len;
        let cnn_kernel = stack.cnn_module_kernel;

        cached_len.push(
            Tensor::<i64>::from_array((vec![layers as i64, 1], vec![0_i64; layers]))
                .map_err(ort_tensor_error)?
                .into_dyn(),
        );
        cached_avg.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, 1, encoder_dim as i64],
                vec![0.0_f32; layers * encoder_dim],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        cached_key.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, left_context as i64, 1, attention_dim as i64],
                vec![0.0_f32; layers * left_context * attention_dim],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        let half_attention = attention_dim / 2;
        cached_val.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, left_context as i64, 1, half_attention as i64],
                vec![0.0_f32; layers * left_context * half_attention],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        cached_val2.push(
            Tensor::<f32>::from_array((
                vec![layers as i64, left_context as i64, 1, half_attention as i64],
                vec![0.0_f32; layers * left_context * half_attention],
            ))
            .map_err(ort_tensor_error)?
            .into_dyn(),
        );
        cached_conv1.push(
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
        cached_conv2.push(
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

    let mut state = Vec::with_capacity(stacks.len() * 7);
    state.extend(cached_len);
    state.extend(cached_avg);
    state.extend(cached_key);
    state.extend(cached_val);
    state.extend(cached_val2);
    state.extend(cached_conv1);
    state.extend(cached_conv2);

    Ok(state)
}

fn initial_zipformer2_state(
    stacks: &[Zipformer2StackSpec],
    feature_dim: usize,
) -> Result<Vec<DynValue>, ModelError> {
    let layer_count = stacks.iter().map(|stack| stack.num_layers).sum::<usize>();
    let mut state = Vec::with_capacity(layer_count * 6 + 2);

    for stack in stacks {
        let key_dim = stack.query_head_dim * stack.num_heads;
        let value_dim = stack.value_head_dim * stack.num_heads;
        let nonlin_attn_head_dim = 3 * stack.encoder_dim / 4;
        let conv_cache_len = stack.cnn_module_kernel / 2;

        for _ in 0..stack.num_layers {
            state.push(
                Tensor::<f32>::from_array((
                    vec![stack.left_context_len as i64, 1, key_dim as i64],
                    vec![0.0_f32; stack.left_context_len * key_dim],
                ))
                .map_err(ort_tensor_error)?
                .into_dyn(),
            );
            state.push(
                Tensor::<f32>::from_array((
                    vec![
                        1,
                        1,
                        stack.left_context_len as i64,
                        nonlin_attn_head_dim as i64,
                    ],
                    vec![0.0_f32; stack.left_context_len * nonlin_attn_head_dim],
                ))
                .map_err(ort_tensor_error)?
                .into_dyn(),
            );
            state.push(
                Tensor::<f32>::from_array((
                    vec![stack.left_context_len as i64, 1, value_dim as i64],
                    vec![0.0_f32; stack.left_context_len * value_dim],
                ))
                .map_err(ort_tensor_error)?
                .into_dyn(),
            );
            state.push(
                Tensor::<f32>::from_array((
                    vec![stack.left_context_len as i64, 1, value_dim as i64],
                    vec![0.0_f32; stack.left_context_len * value_dim],
                ))
                .map_err(ort_tensor_error)?
                .into_dyn(),
            );
            state.push(
                Tensor::<f32>::from_array((
                    vec![1, stack.encoder_dim as i64, conv_cache_len as i64],
                    vec![0.0_f32; stack.encoder_dim * conv_cache_len],
                ))
                .map_err(ort_tensor_error)?
                .into_dyn(),
            );
            state.push(
                Tensor::<f32>::from_array((
                    vec![1, stack.encoder_dim as i64, conv_cache_len as i64],
                    vec![0.0_f32; stack.encoder_dim * conv_cache_len],
                ))
                .map_err(ort_tensor_error)?
                .into_dyn(),
            );
        }
    }

    let embed_dim = (((feature_dim.saturating_sub(1)) / 2).saturating_sub(1)) / 2;
    state.push(
        Tensor::<f32>::from_array((
            vec![1, 128, 3, embed_dim as i64],
            vec![0.0_f32; 128 * 3 * embed_dim],
        ))
        .map_err(ort_tensor_error)?
        .into_dyn(),
    );
    state.push(
        Tensor::<i64>::from_array((vec![1], vec![0_i64]))
            .map_err(ort_tensor_error)?
            .into_dyn(),
    );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_reports_backend_metadata() {
        let adapter = SherpaOnnxStreamingAdapter::zipformer_en_streaming();

        assert_eq!(adapter.supported_formats(), &[CheckpointFormat::Onnx]);
        assert_eq!(adapter.backend_kind(), BackendKind::SherpaOnnx);
        assert!(
            adapter
                .capabilities()
                .supports(CapabilityKind::Transcription)
        );
        assert_eq!(adapter.quantization(), &QuantizationSupport::none());
    }

    #[test]
    fn initial_decoder_state_contains_context_padding() {
        let state = DecoderState::new(&ZipformerConfig {
            feature_dim: 80,
            context_size: 2,
            encoder_input_frames: 16,
            decode_chunk_len: 8,
            frame_shift_ms: TARGET_FRAME_SHIFT_MS,
            unk_id: Some(1),
            state_layout: StateLayout::Zipformer {
                stacks: vec![EncoderStackSpec {
                    encoder_dim: 4,
                    attention_dim: 4,
                    num_layers: 2,
                    cnn_module_kernel: 3,
                    left_context_len: 2,
                }],
            },
        })
        .expect("decoder state should initialize");

        assert_eq!(state.tokens, vec![-1, BLANK_ID]);
        assert!(state.timestamps.is_empty());
        assert_eq!(state.encoder_state.len(), 7);
    }

    #[test]
    fn zipformer_initial_state_matches_upstream_grouped_ordering() {
        let layout = StateLayout::Zipformer {
            stacks: vec![
                EncoderStackSpec {
                    encoder_dim: 4,
                    attention_dim: 8,
                    num_layers: 2,
                    cnn_module_kernel: 3,
                    left_context_len: 5,
                },
                EncoderStackSpec {
                    encoder_dim: 6,
                    attention_dim: 10,
                    num_layers: 3,
                    cnn_module_kernel: 7,
                    left_context_len: 4,
                },
            ],
        };

        let state = initial_encoder_state(&layout, 80).expect("zipformer state should initialize");
        assert_eq!(state.len(), 14);

        let (shape, _) = state[0]
            .try_extract_tensor::<i64>()
            .expect("first state should be i64 cached_len for stack 0");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![2, 1]);

        let (shape, _) = state[1]
            .try_extract_tensor::<i64>()
            .expect("second state should be i64 cached_len for stack 1");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![3, 1]);

        let (shape, _) = state[2]
            .try_extract_tensor::<f32>()
            .expect("third state should be f32 cached_avg for stack 0");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![2, 1, 4]);

        let (shape, _) = state[8]
            .try_extract_tensor::<f32>()
            .expect("ninth state should be f32 cached_val2 for stack 0");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![2, 5, 1, 4]);
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

    #[test]
    fn state_layout_rejects_mismatched_metadata_lengths() {
        let err = StateLayout::try_from_zipformer_parts(
            vec![4, 8],
            vec![4],
            vec![2, 2],
            vec![3, 3],
            vec![2, 2],
        )
        .expect_err("metadata length mismatch should fail");

        assert!(matches!(err, ModelError::BackendInitialization { .. }));
        assert!(
            err.to_string()
                .contains("zipformer encoder metadata length mismatch")
        );
    }

    #[test]
    fn zipformer2_initial_state_matches_upstream_layout() {
        let layout = StateLayout::try_from_zipformer2_parts(
            vec![192, 256, 384, 512, 384, 256],
            vec![32, 32, 32, 32, 32, 32],
            vec![12, 12, 12, 12, 12, 12],
            vec![4, 4, 4, 8, 4, 4],
            vec![2, 2, 3, 4, 3, 2],
            vec![31, 31, 15, 15, 15, 31],
            vec![64, 32, 16, 8, 16, 32],
        )
        .expect("zipformer2 metadata should be accepted");

        let state = initial_encoder_state(&layout, 80).expect("zipformer2 state should initialize");
        assert_eq!(state.len(), 16 * 6 + 2);

        let (shape, _) = state[0]
            .try_extract_tensor::<f32>()
            .expect("first state should be f32");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![64, 1, 128]);

        let (shape, _) = state[1]
            .try_extract_tensor::<f32>()
            .expect("second state should be f32");
        assert_eq!(
            shape.iter().copied().collect::<Vec<_>>(),
            vec![1, 1, 64, 144]
        );

        let (shape, _) = state[state.len() - 2]
            .try_extract_tensor::<f32>()
            .expect("penultimate state should be f32");
        assert_eq!(
            shape.iter().copied().collect::<Vec<_>>(),
            vec![1, 128, 3, 19]
        );

        let (shape, _) = state[state.len() - 1]
            .try_extract_tensor::<i64>()
            .expect("last state should be i64");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn encoder_frame_extraction_uses_returned_tensor_shape() {
        let encoder_out = Tensor::<f32>::from_array((vec![1_i64, 7, 4], vec![0.0_f32; 28]))
            .expect("tensor shape should be valid")
            .into_dyn();

        let frame_view = extract_encoder_frames(&encoder_out).expect("shape should be accepted");
        assert_eq!(frame_view.shape(), &[1, 7, 4]);
    }
}
