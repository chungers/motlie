use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{
    AudioBuf, BufferedSpeechChunkStream, BufferedSpeechSynthesizer, Mono, SpeechSynthesizer,
    SynthesisRequest,
};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationSupport, ResolvedCheckpoint, SpeechParams,
    StartOptions, UnsupportedChat, UnsupportedCompletion, UnsupportedEmbeddings,
};
use motlie_model_espeak_ng::text_to_phonemes;
use motlie_model_ort::{build_session_with_target, OrtExecutionTarget};
use ndarray::{Array1, Array2};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    resolve_onnx_artifacts, KokoroArtifactPaths, KokoroArtifactSpec, RuntimeMetricState,
};

const KOKORO_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Onnx];
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;
const MAX_TOKENS_WITHOUT_PADS: usize = 510;
const STYLE_WIDTH: usize = 256;

pub type KokoroSpeechStream = BufferedSpeechChunkStream<i16, KOKORO_SAMPLE_RATE_HZ, Mono>;

#[derive(Clone, Debug)]
pub struct KokoroSpeechSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub(crate) artifact: KokoroArtifactSpec<'static>,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl KokoroSpeechSpec {
    pub fn kokoro_82m() -> Self {
        Self {
            id: BundleId::new("kokoro_82m"),
            display_name: "Kokoro-82M v1.0 ONNX af_bella",
            artifact: KokoroArtifactSpec {
                model: "onnx/model_quantized.onnx",
                tokenizer_json: "tokenizer.json",
                voice: "voices/af_bella.bin",
            },
            capabilities: Capabilities::speech_buffered_only(),
            quantization: QuantizationSupport::none(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KokoroSpeechAdapter {
    spec: KokoroSpeechSpec,
}

impl KokoroSpeechAdapter {
    pub fn kokoro_82m() -> Self {
        Self {
            spec: KokoroSpeechSpec::kokoro_82m(),
        }
    }
}

#[async_trait]
impl BackendAdapter for KokoroSpeechAdapter {
    type Handle = KokoroHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &KOKORO_FORMATS
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

        let artifacts = resolve_onnx_artifacts(checkpoint, self.spec.artifact)?;
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
pub struct KokoroSpeechBundle {
    metadata: BundleMetadata,
    spec: KokoroSpeechSpec,
}

impl KokoroSpeechBundle {
    pub fn new(spec: KokoroSpeechSpec) -> Self {
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

    pub async fn start_typed(&self, options: StartOptions) -> Result<KokoroHandle, ModelError> {
        self.metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;

        let artifacts = if let Some(policy) = options.artifact_policy {
            configure_artifact_policy(self.spec.artifact, policy)?
        } else {
            configure_artifact_policy(
                self.spec.artifact,
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

#[async_trait]
impl ModelBundle for KokoroSpeechBundle {
    type Handle = KokoroHandle;

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

pub struct KokoroHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<KokoroRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

impl KokoroHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }

    async fn synthesize_pcm(
        &self,
        request: SynthesisRequest,
    ) -> Result<AudioBuf<i16, KOKORO_SAMPLE_RATE_HZ, Mono>, ModelError> {
        let started_at = Instant::now();
        let pcm = self.runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&self.metrics, "kokoro-typed-synthesize");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(AudioBuf::new(pcm))
    }
}

#[derive(Clone, Debug, Default)]
struct SpeechMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for KokoroHandle {
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
        let metrics = lock_metrics(&self.metrics, "kokoro-metric-snapshot").clone();
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
    runtime: Arc<KokoroRuntime>,
) -> KokoroHandle {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "kokoro-start");
        observe_memory(&mut state.runtime);
    }

    KokoroHandle {
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

struct KokoroRuntime {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    voice: VoiceStyle,
}

impl KokoroRuntime {
    fn synthesize(&self, request: &SynthesisRequest) -> Result<Vec<i16>, ModelError> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "speech request requires non-empty text".into(),
            ));
        }
        if request.params.seed.is_some() {
            return Err(ModelError::InvalidConfiguration(
                "kokoro backend does not support `SpeechParams.seed`".into(),
            ));
        }

        let speed = synthesis_speed(&request.params)?;
        let phonemes = phonemize_text(text)?;
        let tokens = tokenize_phonemes(&self.tokenizer, &phonemes)?;
        let style = self.voice.style_for_token_count(tokens.len())?;
        let samples = run_inference(&self.session, &tokens, style, speed)?;
        Ok(f32_to_i16_samples(&samples))
    }
}

impl BufferedSpeechSynthesizer for KokoroHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<i16, KOKORO_SAMPLE_RATE_HZ, Mono>;

    async fn synthesize_buffered(
        &self,
        request: Self::Request,
    ) -> Result<Self::Output, ModelError> {
        self.synthesize_pcm(request).await
    }
}

impl SpeechSynthesizer for KokoroHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<i16, KOKORO_SAMPLE_RATE_HZ, Mono>;
    type Stream = KokoroSpeechStream;

    async fn synthesize(&self, request: Self::Request) -> Result<Self::Stream, ModelError> {
        Ok(KokoroSpeechStream::new(
            self.synthesize_pcm(request).await?,
            OUTPUT_CHUNK_DURATION_MS,
        ))
    }
}

#[derive(Clone, Debug)]
struct VoiceStyle {
    rows: Vec<f32>,
    row_count: usize,
}

impl VoiceStyle {
    fn from_path(path: &std::path::Path) -> Result<Self, ModelError> {
        let bytes = std::fs::read(path).map_err(|err| ModelError::BackendInitialization {
            backend: "kokoro",
            message: format!("failed to read voice style `{}`: {err}", path.display()),
        })?;
        let floats = bytes_to_f32_le(&bytes).map_err(|message| {
            ModelError::InvalidConfiguration(format!(
                "invalid kokoro voice style `{}`: {message}",
                path.display()
            ))
        })?;
        if floats.is_empty() || floats.len() % STYLE_WIDTH != 0 {
            return Err(ModelError::InvalidConfiguration(format!(
                "kokoro voice style `{}` must contain a non-empty multiple of {STYLE_WIDTH} f32 values, found {}",
                path.display(),
                floats.len()
            )));
        }
        let row_count = floats.len() / STYLE_WIDTH;
        Ok(Self {
            rows: floats,
            row_count,
        })
    }

    fn style_for_token_count(&self, token_count: usize) -> Result<&[f32], ModelError> {
        if self.row_count == 0 {
            return Err(ModelError::InvalidConfiguration(
                "kokoro voice style contains no rows".into(),
            ));
        }
        let row = token_count.min(self.row_count - 1);
        let start = row * STYLE_WIDTH;
        Ok(&self.rows[start..start + STYLE_WIDTH])
    }
}

fn load_runtime(artifacts: &KokoroArtifactPaths) -> Result<KokoroRuntime, ModelError> {
    let tokenizer = load_tokenizer(&artifacts.tokenizer_json)?;
    let voice = VoiceStyle::from_path(&artifacts.voice)?;

    Ok(KokoroRuntime {
        session: Mutex::new(build_session_with_target(
            "kokoro",
            &artifacts.model,
            kokoro_ort_target(),
        )?),
        tokenizer,
        voice,
    })
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, ModelError> {
    let content =
        std::fs::read_to_string(path).map_err(|err| ModelError::BackendInitialization {
            backend: "kokoro",
            message: format!("failed to read tokenizer json `{}`: {err}", path.display()),
        })?;

    match Tokenizer::from_str(&content) {
        Ok(tokenizer) => Ok(tokenizer),
        Err(original_err) => load_tokenizer_without_post_processor(path, &content).map_err(|fallback_err| {
            ModelError::BackendInitialization {
                backend: "kokoro",
                message: format!(
                    "failed to load tokenizer json `{}`: {original_err}; fallback without post_processor also failed: {fallback_err}",
                    path.display()
                ),
            }
        }),
    }
}

fn load_tokenizer_without_post_processor(
    path: &Path,
    content: &str,
) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let mut value = serde_json::from_str::<serde_json::Value>(content)?;
    let Some(object) = value.as_object_mut() else {
        return Err(format!("tokenizer json `{}` is not an object", path.display()).into());
    };
    if !object.contains_key("post_processor") {
        return Err("tokenizer json has no post_processor to sanitize".into());
    }
    object.insert("post_processor".into(), serde_json::Value::Null);
    if let Some(model) = object
        .get_mut("model")
        .and_then(serde_json::Value::as_object_mut)
    {
        model.entry("type").or_insert_with(|| "WordLevel".into());
        model.entry("unk_token").or_insert_with(|| "$".into());
    }
    let sanitized = serde_json::to_string(&value)?;
    Tokenizer::from_str(&sanitized)
}

fn kokoro_ort_target() -> OrtExecutionTarget {
    match std::env::var("MOTLIE_KOKORO_ALLOW_CUDA") {
        Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") => OrtExecutionTarget::Auto,
        _ => OrtExecutionTarget::CpuOnly,
    }
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

fn phonemize_text(text: &str) -> Result<String, ModelError> {
    let batches = text_to_phonemes(text, "en-us", None, true, false).map_err(|err| {
        ModelError::BackendExecution {
            backend: "kokoro",
            operation: "phonemize_text",
            message: err.to_string(),
        }
    })?;
    let phonemes = batches
        .into_iter()
        .map(|batch| normalize_espeak_ipa_for_kokoro(&batch))
        .filter(|batch| !batch.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if phonemes.trim().is_empty() {
        return Err(ModelError::BackendExecution {
            backend: "kokoro",
            operation: "phonemize_text",
            message: "phonemizer produced no output for request text".into(),
        });
    }
    Ok(phonemes)
}

fn normalize_espeak_ipa_for_kokoro(phonemes: &str) -> String {
    phonemes
        .replace(['ˌ', 'ˈ'], "")
        .replace("ɜɹ", "ɜː")
        .replace("ɚ", "ə")
}

fn tokenize_phonemes(tokenizer: &Tokenizer, phonemes: &str) -> Result<Vec<i64>, ModelError> {
    let encoding =
        tokenizer
            .encode(phonemes, false)
            .map_err(|err| ModelError::BackendExecution {
                backend: "kokoro",
                operation: "tokenize_phonemes",
                message: err.to_string(),
            })?;
    let ids = encoding
        .get_ids()
        .iter()
        .copied()
        .filter(|id| *id != 0)
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Err(ModelError::BackendExecution {
            backend: "kokoro",
            operation: "tokenize_phonemes",
            message: "tokenizer produced no tokens".into(),
        });
    }
    if ids.len() > MAX_TOKENS_WITHOUT_PADS {
        return Err(ModelError::InvalidConfiguration(format!(
            "kokoro tokenized input length {} exceeds maximum {MAX_TOKENS_WITHOUT_PADS}",
            ids.len()
        )));
    }

    let mut padded = Vec::with_capacity(ids.len() + 2);
    padded.push(0);
    padded.extend(ids.into_iter().map(i64::from));
    padded.push(0);
    Ok(padded)
}

fn run_inference(
    session: &Mutex<Session>,
    tokens: &[i64],
    style: &[f32],
    speed: f32,
) -> Result<Vec<f32>, ModelError> {
    let input_ids =
        Array2::<i64>::from_shape_vec((1, tokens.len()), tokens.to_vec()).map_err(|err| {
            ModelError::BackendExecution {
                backend: "kokoro",
                operation: "build_input_ids_tensor",
                message: err.to_string(),
            }
        })?;
    let style_tensor =
        Array2::<f32>::from_shape_vec((1, STYLE_WIDTH), style.to_vec()).map_err(|err| {
            ModelError::BackendExecution {
                backend: "kokoro",
                operation: "build_style_tensor",
                message: err.to_string(),
            }
        })?;
    let speed_tensor = Array1::<f32>::from_vec(vec![speed]);

    let input_ids = Tensor::<i64>::from_array(input_ids).map_err(ort_tensor_error)?;
    let style_tensor = Tensor::<f32>::from_array(style_tensor).map_err(ort_tensor_error)?;
    let speed_tensor = Tensor::<f32>::from_array(speed_tensor).map_err(ort_tensor_error)?;
    let inputs = vec![
        SessionInputValue::from(input_ids),
        SessionInputValue::from(style_tensor),
        SessionInputValue::from(speed_tensor),
    ];

    let mut session = session
        .lock()
        .map_err(|_| ModelError::Internal("kokoro inference session mutex was poisoned".into()))?;
    let outputs = session
        .run(inputs.as_slice())
        .map_err(|err| ModelError::BackendExecution {
            backend: "kokoro",
            operation: "run_inference",
            message: err.to_string(),
        })?;

    let (_, samples) =
        outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|err| ModelError::BackendExecution {
                backend: "kokoro",
                operation: "extract_audio_tensor",
                message: err.to_string(),
            })?;
    Ok(samples.to_vec())
}

fn f32_to_i16_samples(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

fn bytes_to_f32_le(bytes: &[u8]) -> Result<Vec<f32>, &'static str> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err("byte length is not a multiple of f32");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn ort_tensor_error(err: ort::Error) -> ModelError {
    ModelError::BackendExecution {
        backend: "kokoro",
        operation: "build_tensor",
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_style_selects_token_length_row() {
        let mut rows = Vec::new();
        for row in 0..3 {
            rows.extend(std::iter::repeat(row as f32).take(STYLE_WIDTH));
        }
        let style = VoiceStyle { rows, row_count: 3 };

        assert_eq!(style.style_for_token_count(0).expect("row")[0], 0.0);
        assert_eq!(style.style_for_token_count(2).expect("row")[0], 2.0);
        assert_eq!(style.style_for_token_count(99).expect("row")[0], 2.0);
    }

    #[test]
    fn voice_style_bytes_require_f32_alignment() {
        let error = bytes_to_f32_le(&[1, 2, 3]).expect_err("unaligned bytes should fail");
        assert!(error.contains("multiple of f32"));
    }

    #[test]
    fn speed_must_be_positive() {
        let error = synthesis_speed(&SpeechParams {
            speaking_rate: Some(0.0),
            ..Default::default()
        })
        .expect_err("zero speed should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("speaking_rate")
        ));
    }
}
