use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, Mono, StreamingTranscriber, TranscriptionSession};
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, PcmChunk, QuantizationSupport, StartOptions,
    TranscriptSegment, TranscriptionParams, TranscriptionUpdate, UnsupportedChat,
    UnsupportedCompletion, UnsupportedEmbeddings,
};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use serde_json::Value;

use crate::common::{
    MoonshineArtifactPaths, MoonshineArtifactSpec, NormalizerState, RuntimeMetricState,
    StagedModelDir, configure_artifact_policy, lock_metrics, observe_latency,
    resolve_onnx_artifacts,
};

const MOONSHINE_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
const PREFERRED_CHUNK_BYTES: usize = 6_400;
const CHUNK_SIZE: usize = 1_280;
const TOKENS_PER_SECOND: f32 = 6.5;
const NUM_THREADS: usize = 4;

#[derive(Clone, Debug)]
pub struct MoonshineStreamingSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub frontend_filename: &'static str,
    pub encoder_filename: &'static str,
    pub adapter_filename: &'static str,
    pub cross_kv_filename: &'static str,
    pub decoder_kv_filename: &'static str,
    pub streaming_config_filename: &'static str,
    pub tokenizer_json_filename: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl MoonshineStreamingSpec {
    pub fn small_en() -> Self {
        Self {
            id: BundleId::new("moonshine_streaming_en"),
            display_name: "Moonshine Streaming EN",
            frontend_filename: "onnx/small/frontend.ort",
            encoder_filename: "onnx/small/encoder.ort",
            adapter_filename: "onnx/small/adapter.ort",
            cross_kv_filename: "onnx/small/cross_kv.ort",
            decoder_kv_filename: "onnx/small/decoder_kv.ort",
            streaming_config_filename: "onnx/small/streaming_config.json",
            tokenizer_json_filename: "onnx/small/tokenizer.json",
            capabilities: Capabilities::transcription_stream_only(),
            quantization: QuantizationSupport::none(),
        }
    }

    fn artifact_spec(&self) -> MoonshineArtifactSpec<'_> {
        MoonshineArtifactSpec {
            frontend: self.frontend_filename,
            encoder: self.encoder_filename,
            adapter: self.adapter_filename,
            cross_kv: self.cross_kv_filename,
            decoder_kv: self.decoder_kv_filename,
            streaming_config: self.streaming_config_filename,
            tokenizer_json: self.tokenizer_json_filename,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoonshineStreamingAdapter {
    spec: MoonshineStreamingSpec,
}

impl MoonshineStreamingAdapter {
    pub fn small_en() -> Self {
        Self {
            spec: MoonshineStreamingSpec::small_en(),
        }
    }
}

#[async_trait]
impl BackendAdapter for MoonshineStreamingAdapter {
    type Handle = MoonshineHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &MOONSHINE_FORMATS
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
        checkpoint: &motlie_model::ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError> {
        self.quantization()
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
pub struct MoonshineStreamingBundle {
    metadata: BundleMetadata,
    spec: MoonshineStreamingSpec,
}

impl MoonshineStreamingBundle {
    pub fn new(spec: MoonshineStreamingSpec) -> Self {
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
impl ModelBundle for MoonshineStreamingBundle {
    type Handle = MoonshineHandle;

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

impl MoonshineStreamingBundle {
    pub async fn start_typed(&self, options: StartOptions) -> Result<MoonshineHandle, ModelError> {
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let artifacts = if let Some(policy) = options.artifact_policy {
            configure_artifact_policy(self.spec.artifact_spec(), policy)?
        } else {
            configure_artifact_policy(
                self.spec.artifact_spec(),
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

pub struct MoonshineHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<MoonshineRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
}

impl MoonshineHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }
}

#[derive(Clone, Debug, Default)]
struct AsrMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for MoonshineHandle {
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
        let metrics = lock_metrics(&self.metrics, "moonshine-metric-snapshot").clone();
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
    runtime: Arc<MoonshineRuntime>,
) -> MoonshineHandle {
    MoonshineHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization: None,
        },
        runtime,
        metrics: Arc::new(Mutex::new(AsrMetrics::default())),
    }
}

#[derive(Clone, Debug)]
struct StreamingConfig {
    encoder_dim: usize,
    decoder_dim: usize,
    depth: usize,
    nheads: usize,
    head_dim: usize,
    vocab_size: usize,
    bos_id: i64,
    eos_id: i64,
    total_lookahead: usize,
    d_model_frontend: usize,
    c1: usize,
    max_seq_len: usize,
}

impl StreamingConfig {
    fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let config_path = model_dir.join("streaming_config.json");
        let contents =
            fs::read_to_string(&config_path).map_err(|err| ModelError::BackendInitialization {
                backend: "moonshine",
                message: format!(
                    "failed to read Moonshine streaming config `{}`: {err}",
                    config_path.display()
                ),
            })?;
        let json: Value =
            serde_json::from_str(&contents).map_err(|err| ModelError::BackendInitialization {
                backend: "moonshine",
                message: format!(
                    "failed to parse Moonshine streaming config `{}`: {err}",
                    config_path.display()
                ),
            })?;

        let get_usize = |key: &str| -> usize {
            json.get(key).and_then(|value| value.as_i64()).unwrap_or(0) as usize
        };
        let get_i64 =
            |key: &str| -> i64 { json.get(key).and_then(|value| value.as_i64()).unwrap_or(0) };

        let max_seq_len = match get_usize("max_seq_len") {
            0 => 448,
            value => value,
        };

        let config = Self {
            encoder_dim: get_usize("encoder_dim"),
            decoder_dim: get_usize("decoder_dim"),
            depth: get_usize("depth"),
            nheads: get_usize("nheads"),
            head_dim: get_usize("head_dim"),
            vocab_size: get_usize("vocab_size"),
            bos_id: get_i64("bos_id"),
            eos_id: get_i64("eos_id"),
            total_lookahead: get_usize("total_lookahead"),
            d_model_frontend: get_usize("d_model_frontend"),
            c1: get_usize("c1"),
            max_seq_len,
        };

        if config.depth == 0 || config.decoder_dim == 0 || config.vocab_size == 0 {
            return Err(ModelError::BackendInitialization {
                backend: "moonshine",
                message: format!(
                    "invalid Moonshine streaming config `{}`: depth, decoder_dim, and vocab_size must be > 0",
                    config_path.display()
                ),
            });
        }

        Ok(config)
    }
}

struct BinTokenizer {
    tokens_to_bytes: Vec<Vec<u8>>,
}

impl BinTokenizer {
    fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let tokenizer_path = model_dir.join("tokenizer.bin");
        let data = fs::read(&tokenizer_path).map_err(|err| ModelError::BackendInitialization {
            backend: "moonshine",
            message: format!(
                "failed to read Moonshine tokenizer `{}`: {err}",
                tokenizer_path.display()
            ),
        })?;

        let mut tokens_to_bytes = Vec::new();
        let mut offset = 0_usize;
        while offset < data.len() {
            let first_byte = data[offset];
            offset += 1;

            if first_byte == 0 {
                tokens_to_bytes.push(Vec::new());
                continue;
            }

            let byte_count = if first_byte < 128 {
                first_byte as usize
            } else {
                if offset >= data.len() {
                    break;
                }
                let second_byte = data[offset];
                offset += 1;
                (second_byte as usize * 128) + first_byte as usize - 128
            };

            if offset + byte_count > data.len() {
                break;
            }

            tokens_to_bytes.push(data[offset..offset + byte_count].to_vec());
            offset += byte_count;
        }

        if tokens_to_bytes.is_empty() {
            return Err(ModelError::BackendInitialization {
                backend: "moonshine",
                message: format!(
                    "Moonshine tokenizer `{}` contained no token entries",
                    tokenizer_path.display()
                ),
            });
        }

        Ok(Self { tokens_to_bytes })
    }

    fn decode(&self, tokens: &[i64]) -> String {
        let mut result_bytes = Vec::new();
        for &token in tokens {
            let idx = token as usize;
            if idx >= self.tokens_to_bytes.len() {
                continue;
            }
            let bytes = &self.tokens_to_bytes[idx];
            if bytes.is_empty() {
                continue;
            }
            if bytes.len() > 2 && bytes.first() == Some(&b'<') && bytes.last() == Some(&b'>') {
                continue;
            }
            result_bytes.extend_from_slice(bytes);
        }

        let text = String::from_utf8_lossy(&result_bytes);
        text.replace('\u{2581}', " ").trim().to_owned()
    }
}

#[derive(Clone)]
struct StreamingState {
    sample_buffer: Vec<f32>,
    sample_len: i64,
    conv1_buffer: Vec<f32>,
    conv2_buffer: Vec<f32>,
    frame_count: i64,
    accumulated_features: Vec<f32>,
    accumulated_feature_count: i32,
    encoder_frames_emitted: i32,
    adapter_pos_offset: i64,
    memory: Vec<f32>,
    memory_len: i32,
    k_self: Vec<f32>,
    v_self: Vec<f32>,
    cache_seq_len: i32,
    k_cross: Vec<f32>,
    v_cross: Vec<f32>,
    cross_len: i32,
    cross_kv_valid: bool,
}

impl StreamingState {
    fn new(config: &StreamingConfig) -> Self {
        Self {
            sample_buffer: vec![0.0; 79],
            sample_len: 0,
            conv1_buffer: vec![0.0; config.d_model_frontend * 4],
            conv2_buffer: vec![0.0; config.c1 * 4],
            frame_count: 0,
            accumulated_features: Vec::new(),
            accumulated_feature_count: 0,
            encoder_frames_emitted: 0,
            adapter_pos_offset: 0,
            memory: Vec::new(),
            memory_len: 0,
            k_self: Vec::new(),
            v_self: Vec::new(),
            cache_seq_len: 0,
            k_cross: Vec::new(),
            v_cross: Vec::new(),
            cross_len: 0,
            cross_kv_valid: false,
        }
    }
}

struct GreedyDecoder {
    eos_id: i64,
    last_token: i64,
    consecutive_count: usize,
}

impl GreedyDecoder {
    fn new(eos_id: i64) -> Self {
        Self {
            eos_id,
            last_token: -1,
            consecutive_count: 0,
        }
    }

    fn next_token(&mut self, logits: &[f32]) -> Option<i64> {
        let token = argmax(logits) as i64;
        if token == self.eos_id {
            return None;
        }

        if token == self.last_token {
            self.consecutive_count += 1;
            if self.consecutive_count > 8 {
                return None;
            }
        } else {
            self.consecutive_count = 1;
        }

        self.last_token = token;
        Some(token)
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > max_val {
            max_val = value;
            max_idx = index;
        }
    }
    max_idx
}

struct MoonshineRuntime {
    frontend: Mutex<Session>,
    encoder: Mutex<Session>,
    adapter: Mutex<Session>,
    cross_kv: Mutex<Session>,
    decoder_kv: Mutex<Session>,
    tokenizer: BinTokenizer,
    config: StreamingConfig,
    _staged_root: StagedModelDir,
}

impl MoonshineRuntime {
    fn create_state(&self) -> StreamingState {
        StreamingState::new(&self.config)
    }

    fn infer_chunk(
        &self,
        state: &mut StreamingState,
        normalized_chunk: &[f32],
        is_final: bool,
        total_samples: usize,
    ) -> Result<Option<String>, ModelError> {
        if !normalized_chunk.is_empty() {
            for subchunk in normalized_chunk.chunks(CHUNK_SIZE) {
                self.process_audio_chunk(state, subchunk)?;
            }
        }

        let new_frames = self.encode_streaming(state, is_final)?;
        if !is_final && new_frames <= 0 {
            return Ok(None);
        }
        if state.memory_len <= 0 {
            return Ok(None);
        }

        let text = self.decode_transcript(state, total_samples)?;
        if text.is_empty() {
            Ok(None)
        } else {
            Ok(Some(text))
        }
    }

    fn process_audio_chunk(
        &self,
        state: &mut StreamingState,
        audio_chunk: &[f32],
    ) -> Result<(), ModelError> {
        if audio_chunk.is_empty() {
            return Ok(());
        }

        let audio_dyn =
            ArrayD::from_shape_vec(IxDyn(&[1, audio_chunk.len()]), audio_chunk.to_vec())
                .map_err(|err| backend_exec_error("process_audio_chunk", err.to_string()))?;
        let sample_buffer_dyn =
            ArrayD::from_shape_vec(IxDyn(&[1, 79]), state.sample_buffer.clone())
                .map_err(|err| backend_exec_error("process_audio_chunk", err.to_string()))?;
        let sample_len_dyn = ArrayD::from_shape_vec(IxDyn(&[1]), vec![state.sample_len])
            .map_err(|err| backend_exec_error("process_audio_chunk", err.to_string()))?;
        let conv1_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1, self.config.d_model_frontend, 4]),
            state.conv1_buffer.clone(),
        )
        .map_err(|err| backend_exec_error("process_audio_chunk", err.to_string()))?;
        let conv2_dyn =
            ArrayD::from_shape_vec(IxDyn(&[1, self.config.c1, 4]), state.conv2_buffer.clone())
                .map_err(|err| backend_exec_error("process_audio_chunk", err.to_string()))?;
        let frame_count_dyn = ArrayD::from_shape_vec(IxDyn(&[1]), vec![state.frame_count])
            .map_err(|err| backend_exec_error("process_audio_chunk", err.to_string()))?;

        let mut frontend = self.frontend.lock().map_err(|_| {
            backend_exec_error(
                "process_audio_chunk",
                "frontend session mutex poisoned".into(),
            )
        })?;
        let outputs = frontend
            .run(inputs![
                "audio_chunk" => TensorRef::from_array_view(audio_dyn.view()).map_err(ort_exec_error("process_audio_chunk"))?,
                "sample_buffer" => TensorRef::from_array_view(sample_buffer_dyn.view()).map_err(ort_exec_error("process_audio_chunk"))?,
                "sample_len" => TensorRef::from_array_view(sample_len_dyn.view()).map_err(ort_exec_error("process_audio_chunk"))?,
                "conv1_buffer" => TensorRef::from_array_view(conv1_dyn.view()).map_err(ort_exec_error("process_audio_chunk"))?,
                "conv2_buffer" => TensorRef::from_array_view(conv2_dyn.view()).map_err(ort_exec_error("process_audio_chunk"))?,
                "frame_count" => TensorRef::from_array_view(frame_count_dyn.view()).map_err(ort_exec_error("process_audio_chunk"))?,
            ])
            .map_err(ort_exec_error("process_audio_chunk"))?;

        let features = output_tensor_f32(&outputs, "features", "process_audio_chunk")?;
        let feat_shape = &features.shape;
        let num_features = feat_shape.get(1).copied().ok_or_else(|| {
            backend_exec_error("process_audio_chunk", "features missing dim 1".into())
        })? as i32;
        if num_features > 0 {
            let feat_size = feat_shape
                .get(1)
                .zip(feat_shape.get(2))
                .map(|(a, b)| a * b)
                .ok_or_else(|| {
                    backend_exec_error("process_audio_chunk", "features missing dims".into())
                })?;
            state
                .accumulated_features
                .extend_from_slice(&features.data[..feat_size]);
            state.accumulated_feature_count += num_features;
        }

        let sample_buffer_out =
            output_tensor_f32(&outputs, "sample_buffer_out", "process_audio_chunk")?;
        state.sample_buffer = sample_buffer_out
            .data
            .get(..79)
            .ok_or_else(|| {
                backend_exec_error(
                    "process_audio_chunk",
                    "sample_buffer_out shorter than 79 samples".into(),
                )
            })?
            .to_vec();

        let sample_len_out = output_tensor_i64(&outputs, "sample_len_out", "process_audio_chunk")?;
        state.sample_len = first_i64(&sample_len_out, "sample_len_out", "process_audio_chunk")?;

        let conv1_out = output_tensor_f32(&outputs, "conv1_buffer_out", "process_audio_chunk")?;
        state.conv1_buffer = pad_or_truncate(&conv1_out.data, self.config.d_model_frontend * 4);

        let conv2_out = output_tensor_f32(&outputs, "conv2_buffer_out", "process_audio_chunk")?;
        state.conv2_buffer = pad_or_truncate(&conv2_out.data, self.config.c1 * 4);

        let frame_count_out =
            output_tensor_i64(&outputs, "frame_count_out", "process_audio_chunk")?;
        state.frame_count = first_i64(&frame_count_out, "frame_count_out", "process_audio_chunk")?;

        Ok(())
    }

    fn encode_streaming(
        &self,
        state: &mut StreamingState,
        is_final: bool,
    ) -> Result<i32, ModelError> {
        let total_features = state.accumulated_feature_count;
        if total_features == 0 {
            return Ok(0);
        }

        let stable_count = if is_final {
            total_features
        } else {
            (total_features - self.config.total_lookahead as i32).max(0)
        };

        let new_frames = stable_count - state.encoder_frames_emitted;
        if new_frames <= 0 {
            return Ok(0);
        }

        let left_context_frames = (16 * self.config.depth) as i32;
        let window_start = (state.encoder_frames_emitted - left_context_frames).max(0);
        let window_size = total_features - window_start;
        let start_idx = (window_start as usize) * self.config.encoder_dim;
        let end_idx = start_idx + (window_size as usize) * self.config.encoder_dim;
        let window_features = &state.accumulated_features[start_idx..end_idx];

        let features_view = ArrayViewD::from_shape(
            IxDyn(&[1, window_size as usize, self.config.encoder_dim]),
            window_features,
        )
        .map_err(|err| backend_exec_error("encode_streaming", err.to_string()))?;

        let mut encoder = self.encoder.lock().map_err(|_| {
            backend_exec_error("encode_streaming", "encoder session mutex poisoned".into())
        })?;
        let enc_outputs = encoder
            .run(inputs!["features" => TensorRef::from_array_view(features_view).map_err(ort_exec_error("encode_streaming"))?])
            .map_err(ort_exec_error("encode_streaming"))?;

        let encoded = output_tensor_f32(&enc_outputs, "encoded", "encode_streaming")?;
        let enc_shape = &encoded.shape;
        let total_encoded = *enc_shape
            .get(1)
            .ok_or_else(|| backend_exec_error("encode_streaming", "encoded missing dim 1".into()))?
            as i32;
        let encoded_data = &encoded.data;

        let slice_start = (state.encoder_frames_emitted - window_start) as usize;
        if slice_start + new_frames as usize > total_encoded as usize {
            return Err(backend_exec_error(
                "encode_streaming",
                format!(
                    "encoder window misaligned: start={slice_start}, new_frames={new_frames}, total={total_encoded}"
                ),
            ));
        }

        let new_encoded: Vec<f32> = (0..new_frames as usize)
            .flat_map(|index| {
                let base = (slice_start + index) * self.config.encoder_dim;
                encoded_data[base..base + self.config.encoder_dim]
                    .iter()
                    .copied()
            })
            .collect();

        let enc_slice_view = ArrayViewD::from_shape(
            IxDyn(&[1, new_frames as usize, self.config.encoder_dim]),
            &new_encoded,
        )
        .map_err(|err| backend_exec_error("encode_streaming", err.to_string()))?;
        let pos_offset_val = [state.adapter_pos_offset];
        let pos_offset_view = ArrayViewD::from_shape(IxDyn(&[1]), &pos_offset_val)
            .map_err(|err| backend_exec_error("encode_streaming", err.to_string()))?;

        let mut adapter = self.adapter.lock().map_err(|_| {
            backend_exec_error("encode_streaming", "adapter session mutex poisoned".into())
        })?;
        let adapter_outputs = adapter
            .run(inputs![
                "encoded" => TensorRef::from_array_view(enc_slice_view).map_err(ort_exec_error("encode_streaming"))?,
                "pos_offset" => TensorRef::from_array_view(pos_offset_view).map_err(ort_exec_error("encode_streaming"))?,
            ])
            .map_err(ort_exec_error("encode_streaming"))?;

        let memory_out = output_tensor_f32(&adapter_outputs, "memory", "encode_streaming")?;
        let memory_data = &memory_out.data;
        let mem_size = new_frames as usize * self.config.decoder_dim;
        state.memory.extend_from_slice(&memory_data[..mem_size]);
        state.memory_len += new_frames;
        state.cross_kv_valid = false;
        state.encoder_frames_emitted = stable_count;
        state.adapter_pos_offset += new_frames as i64;

        Ok(new_frames)
    }

    fn compute_cross_kv(&self, state: &mut StreamingState) -> Result<(), ModelError> {
        if state.memory_len <= 0 {
            return Err(backend_exec_error(
                "compute_cross_kv",
                "memory is empty, cannot compute cross-attention cache".into(),
            ));
        }

        let memory_view = ArrayViewD::from_shape(
            IxDyn(&[1, state.memory_len as usize, self.config.decoder_dim]),
            &state.memory,
        )
        .map_err(|err| backend_exec_error("compute_cross_kv", err.to_string()))?;

        let mut cross_kv = self.cross_kv.lock().map_err(|_| {
            backend_exec_error("compute_cross_kv", "cross_kv session mutex poisoned".into())
        })?;
        let outputs = cross_kv
            .run(inputs!["memory" => TensorRef::from_array_view(memory_view).map_err(ort_exec_error("compute_cross_kv"))?])
            .map_err(ort_exec_error("compute_cross_kv"))?;

        let k_cross = output_tensor_f32(&outputs, "k_cross", "compute_cross_kv")?;
        let v_cross = output_tensor_f32(&outputs, "v_cross", "compute_cross_kv")?;
        let k_shape = &k_cross.shape;
        let cross_len = *k_shape
            .get(3)
            .ok_or_else(|| backend_exec_error("compute_cross_kv", "k_cross missing dim 3".into()))?
            as i32;
        let kv_size =
            self.config.depth * self.config.nheads * cross_len as usize * self.config.head_dim;

        state.k_cross = k_cross.data[..kv_size].to_vec();
        state.v_cross = v_cross.data[..kv_size].to_vec();
        state.cross_len = cross_len;
        state.cross_kv_valid = true;
        Ok(())
    }

    fn decode_step_logits(
        &self,
        state: &mut StreamingState,
        token: i64,
    ) -> Result<Vec<f32>, ModelError> {
        if !state.cross_kv_valid {
            self.compute_cross_kv(state)?;
        }

        let cache_len = state.cache_seq_len as usize;
        let kv_self_size =
            self.config.depth * self.config.nheads * cache_len * self.config.head_dim;
        if state.k_self.len() != kv_self_size {
            state.k_self.resize(kv_self_size, 0.0);
            state.v_self.resize(kv_self_size, 0.0);
        }

        let token_val = [token];
        let token_view = ArrayViewD::from_shape(IxDyn(&[1, 1]), &token_val)
            .map_err(|err| backend_exec_error("decode_step_logits", err.to_string()))?;
        let kv_shape = [
            self.config.depth,
            1,
            self.config.nheads,
            cache_len,
            self.config.head_dim,
        ];
        let k_self_view = ArrayViewD::from_shape(IxDyn(&kv_shape), &state.k_self)
            .map_err(|err| backend_exec_error("decode_step_logits", err.to_string()))?;
        let v_self_view = ArrayViewD::from_shape(IxDyn(&kv_shape), &state.v_self)
            .map_err(|err| backend_exec_error("decode_step_logits", err.to_string()))?;

        let cross_len = state.cross_len as usize;
        let cross_shape = [
            self.config.depth,
            1,
            self.config.nheads,
            cross_len,
            self.config.head_dim,
        ];
        let k_cross_view = ArrayViewD::from_shape(IxDyn(&cross_shape), &state.k_cross)
            .map_err(|err| backend_exec_error("decode_step_logits", err.to_string()))?;
        let v_cross_view = ArrayViewD::from_shape(IxDyn(&cross_shape), &state.v_cross)
            .map_err(|err| backend_exec_error("decode_step_logits", err.to_string()))?;

        let mut decoder_kv = self.decoder_kv.lock().map_err(|_| {
            backend_exec_error(
                "decode_step_logits",
                "decoder_kv session mutex poisoned".into(),
            )
        })?;
        let outputs = decoder_kv
            .run(inputs![
                "token" => TensorRef::from_array_view(token_view).map_err(ort_exec_error("decode_step_logits"))?,
                "k_self" => TensorRef::from_array_view(k_self_view).map_err(ort_exec_error("decode_step_logits"))?,
                "v_self" => TensorRef::from_array_view(v_self_view).map_err(ort_exec_error("decode_step_logits"))?,
                "out_k_cross" => TensorRef::from_array_view(k_cross_view).map_err(ort_exec_error("decode_step_logits"))?,
                "out_v_cross" => TensorRef::from_array_view(v_cross_view).map_err(ort_exec_error("decode_step_logits"))?,
            ])
            .map_err(ort_exec_error("decode_step_logits"))?;

        let k_self_out = output_tensor_f32(&outputs, "out_k_self", "decode_step_logits")?;
        let v_self_out = output_tensor_f32(&outputs, "out_v_self", "decode_step_logits")?;
        let new_cache_len = *k_self_out.shape.get(3).ok_or_else(|| {
            backend_exec_error("decode_step_logits", "out_k_self missing dim 3".into())
        })? as i32;
        let new_cache_size =
            self.config.depth * self.config.nheads * new_cache_len as usize * self.config.head_dim;
        state.k_self = k_self_out.data[..new_cache_size].to_vec();
        state.v_self = v_self_out.data[..new_cache_size].to_vec();
        state.cache_seq_len = new_cache_len;

        let logits = output_tensor_f32(&outputs, "logits", "decode_step_logits")?;
        Ok(logits.data[..self.config.vocab_size].to_vec())
    }

    fn decode_transcript(
        &self,
        state: &StreamingState,
        total_samples: usize,
    ) -> Result<String, ModelError> {
        let mut decode_state = state.clone();
        let mut greedy = GreedyDecoder::new(self.config.eos_id);
        let duration_sec = total_samples as f32 / TARGET_SAMPLE_RATE_HZ as f32;
        let max_tokens = ((duration_sec * TOKENS_PER_SECOND).ceil() as usize).max(1);
        let max_tokens = max_tokens.min(self.config.max_seq_len);

        let mut current_token = self.config.bos_id;
        let mut tokens = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.decode_step_logits(&mut decode_state, current_token)?;
            let next_token = match greedy.next_token(&logits) {
                Some(token) => token,
                None => break,
            };
            tokens.push(next_token);
            current_token = next_token;
        }

        Ok(self.tokenizer.decode(&tokens))
    }
}

fn load_runtime(artifacts: &MoonshineArtifactPaths) -> Result<MoonshineRuntime, ModelError> {
    let staged_root = StagedModelDir::prepare(artifacts)?;
    let config = StreamingConfig::load(staged_root.path())?;

    Ok(MoonshineRuntime {
        frontend: Mutex::new(load_component_session(staged_root.path(), "frontend")?),
        encoder: Mutex::new(load_component_session(staged_root.path(), "encoder")?),
        adapter: Mutex::new(load_component_session(staged_root.path(), "adapter")?),
        cross_kv: Mutex::new(load_component_session(staged_root.path(), "cross_kv")?),
        decoder_kv: Mutex::new(load_component_session(staged_root.path(), "decoder_kv")?),
        tokenizer: BinTokenizer::load(staged_root.path())?,
        config,
        _staged_root: staged_root,
    })
}

fn load_component_session(model_dir: &Path, name: &str) -> Result<Session, ModelError> {
    for extension in ["ort", "onnx"] {
        let path = model_dir.join(format!("{name}.{extension}"));
        if !path.exists() {
            continue;
        }

        let builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
            backend: "moonshine",
            message: format!("failed to create ORT session builder for `{name}`: {err}"),
        })?;
        let mut builder = builder.with_intra_threads(NUM_THREADS).map_err(|err| {
            ModelError::BackendInitialization {
                backend: "moonshine",
                message: format!("failed to configure ORT threads for `{name}`: {err}"),
            }
        })?;

        return builder
            .commit_from_file(&path)
            .map_err(|err| ModelError::BackendInitialization {
                backend: "moonshine",
                message: format!(
                    "failed to load Moonshine component `{}`: {err}",
                    path.display()
                ),
            });
    }

    Err(ModelError::BackendInitialization {
        backend: "moonshine",
        message: format!(
            "missing Moonshine model component `{name}` under `{}`",
            model_dir.display()
        ),
    })
}

struct OwnedTensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

fn output_tensor_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    operation: &'static str,
) -> Result<OwnedTensor<f32>, ModelError> {
    let array = outputs
        .get(name)
        .ok_or_else(|| backend_exec_error(operation, format!("missing output: {name}")))?
        .try_extract_array::<f32>()
        .map_err(ort_exec_error(operation))?;
    let shape = array.shape().to_vec();
    let data = array
        .as_slice()
        .ok_or_else(|| backend_exec_error(operation, format!("{name} tensor is not contiguous")))?
        .to_vec();
    Ok(OwnedTensor { shape, data })
}

fn output_tensor_i64(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    operation: &'static str,
) -> Result<OwnedTensor<i64>, ModelError> {
    let array = outputs
        .get(name)
        .ok_or_else(|| backend_exec_error(operation, format!("missing output: {name}")))?
        .try_extract_array::<i64>()
        .map_err(ort_exec_error(operation))?;
    let shape = array.shape().to_vec();
    let data = array
        .as_slice()
        .ok_or_else(|| backend_exec_error(operation, format!("{name} tensor is not contiguous")))?
        .to_vec();
    Ok(OwnedTensor { shape, data })
}

fn first_i64(
    array: &OwnedTensor<i64>,
    name: &str,
    operation: &'static str,
) -> Result<i64, ModelError> {
    array
        .data
        .first()
        .copied()
        .ok_or_else(|| backend_exec_error(operation, format!("{name} tensor is empty")))
}

fn pad_or_truncate(values: &[f32], expected_len: usize) -> Vec<f32> {
    if values.len() >= expected_len {
        values[..expected_len].to_vec()
    } else {
        let mut padded = vec![0.0; expected_len];
        padded[..values.len()].copy_from_slice(values);
        padded
    }
}

fn ort_exec_error(operation: &'static str) -> impl Fn(ort::Error) -> ModelError {
    move |err| backend_exec_error(operation, err.to_string())
}

fn backend_exec_error(operation: &'static str, message: String) -> ModelError {
    ModelError::BackendExecution {
        backend: "moonshine",
        operation,
        message,
    }
}

pub struct MoonshineStream {
    spec: AudioSpec,
    params: TranscriptionParams,
    runtime: Arc<MoonshineRuntime>,
    metrics: Arc<Mutex<AsrMetrics>>,
    normalizer: NormalizerState,
    state: StreamingState,
    total_samples: usize,
    next_sequence: u64,
    saw_end_of_stream: bool,
    last_partial_text: String,
}

impl MoonshineStream {
    async fn push_chunk(
        &mut self,
        chunk: PcmChunk,
    ) -> Result<Option<TranscriptionUpdate>, ModelError> {
        if self.saw_end_of_stream {
            return Err(ModelError::InvalidConfiguration(
                "push_chunk called after end_of_stream".into(),
            ));
        }
        if chunk.sequence != self.next_sequence {
            return Err(ModelError::InvalidConfiguration(format!(
                "expected chunk sequence {}, got {}",
                self.next_sequence, chunk.sequence
            )));
        }

        self.next_sequence += 1;
        if chunk.end_of_stream {
            self.saw_end_of_stream = true;
        }

        if chunk.data.is_empty() && !chunk.end_of_stream {
            return Ok(None);
        }

        let normalized = self.normalizer.normalize(&chunk.data, &self.spec)?;
        self.total_samples += normalized.len();

        let started = Instant::now();
        let maybe_text =
            self.runtime
                .infer_chunk(&mut self.state, &normalized, false, self.total_samples)?;
        {
            let mut metrics = lock_metrics(&self.metrics, "moonshine-push");
            observe_latency(&mut metrics.runtime, started.elapsed());
        }

        if !self.params.emit_partials {
            return Ok(None);
        }

        let Some(text) = maybe_text else {
            return Ok(None);
        };
        if text.is_empty() || text == self.last_partial_text {
            return Ok(None);
        }
        self.last_partial_text = text.clone();

        Ok(Some(TranscriptionUpdate {
            segments: vec![TranscriptSegment {
                start_ms: 0,
                end_ms: samples_to_ms(self.total_samples),
                text,
                final_segment: false,
            }],
        }))
    }

    async fn finish_stream(mut self) -> Result<TranscriptionUpdate, ModelError> {
        let mut tail = Vec::new();
        self.normalizer.flush(&mut tail);
        self.total_samples += tail.len();

        let started = Instant::now();
        let final_text =
            self.runtime
                .infer_chunk(&mut self.state, &tail, true, self.total_samples)?;
        {
            let mut metrics = lock_metrics(&self.metrics, "moonshine-finish");
            observe_latency(&mut metrics.runtime, started.elapsed());
        }

        match final_text {
            Some(text) if !text.is_empty() => Ok(TranscriptionUpdate {
                segments: vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: samples_to_ms(self.total_samples),
                    text,
                    final_segment: true,
                }],
            }),
            _ => Ok(TranscriptionUpdate::default()),
        }
    }
}

impl StreamingTranscriber for MoonshineHandle {
    type Input = AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>;
    type Session = MoonshineStream;

    fn open_session(
        &self,
        params: TranscriptionParams,
    ) -> impl std::future::Future<Output = Result<Self::Session, ModelError>> + Send {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        async move {
            Ok(MoonshineStream {
                spec: AudioSpec {
                    sample_rate_hz: TARGET_SAMPLE_RATE_HZ,
                    channels: 1,
                    encoding: motlie_model::PcmEncoding::S16Le,
                    preferred_chunk_bytes: PREFERRED_CHUNK_BYTES,
                },
                params,
                runtime: Arc::clone(&runtime),
                metrics,
                normalizer: NormalizerState::new(),
                state: runtime.create_state(),
                total_samples: 0,
                next_sequence: 0,
                saw_end_of_stream: false,
                last_partial_text: String::new(),
            })
        }
    }
}

impl TranscriptionSession for MoonshineStream {
    type Input = AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>;

    fn ingest(
        &mut self,
        audio: Self::Input,
    ) -> impl std::future::Future<Output = Result<Option<TranscriptionUpdate>, ModelError>> + Send
    {
        let data: Vec<u8> = audio
            .into_samples()
            .into_iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let sequence = self.next_sequence;

        async move {
            self.push_chunk(PcmChunk {
                data,
                sequence,
                end_of_stream: false,
            })
            .await
        }
    }

    async fn finish(self) -> Result<TranscriptionUpdate, ModelError> {
        self.finish_stream().await
    }
}

fn samples_to_ms(total_samples: usize) -> u64 {
    (total_samples as u64 * 1000) / TARGET_SAMPLE_RATE_HZ as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_defaults_to_secondary_small_streaming_bundle() {
        let spec = MoonshineStreamingSpec::small_en();

        assert_eq!(spec.id.as_str(), "moonshine_streaming_en");
        assert_eq!(spec.display_name, "Moonshine Streaming EN");
        assert_eq!(spec.frontend_filename, "onnx/small/frontend.ort");
    }

    #[test]
    fn adapter_reports_onnx_and_ort() {
        let adapter = MoonshineStreamingAdapter::small_en();

        assert_eq!(adapter.supported_formats(), &MOONSHINE_FORMATS);
        assert_eq!(adapter.backend_kind(), BackendKind::Ort);
    }

    #[test]
    fn greedy_decoder_stops_after_repeated_token() {
        let mut decoder = GreedyDecoder::new(99);
        let logits = [0.0, 10.0, 0.0];

        for _ in 0..8 {
            assert_eq!(decoder.next_token(&logits), Some(1));
        }
        assert_eq!(decoder.next_token(&logits), None);
    }
}
