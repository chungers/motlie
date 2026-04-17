use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::slice;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::{
    AudioSpec, BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, CheckpointFormat, CompletionModel, EmbeddingModel,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, PcmChunk,
    PcmEncoding, QuantizationBits, QuantizationSupport, ResolvedCheckpoint, SpeechModel,
    SpeechParams, SpeechRequest, SpeechStream, StartOptions, TranscriptionModel, VoiceConditioning,
};

use crate::common::{
    DEFAULT_SAMPLE_RATE_HZ, Qwen3TtsCppArtifactPaths, RuntimeMetricState, audio_spec,
    configure_artifact_policy, decode_pcm_to_f32, downmix_to_mono, encode_pcm, lock_metrics,
    observe_latency, observe_memory, resample_mono, resolve_gguf_artifacts,
};

const QWEN3_TTS_CPP_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Gguf];
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const DEFAULT_LANGUAGE_ID_EN: i32 = 2050;

#[derive(Clone, Debug)]
pub struct Qwen3TtsCppSpeechSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub hf_repo: &'static str,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl Qwen3TtsCppSpeechSpec {
    pub fn qwen3_tts_cpp_0_6b() -> Self {
        Self {
            id: BundleId::new("qwen3_tts_cpp_0_6b"),
            display_name: "Qwen3-TTS CPP 0.6B",
            hf_repo: "koboldcpp/tts",
            capabilities: Capabilities::speech_stream_only(),
            quantization: QuantizationSupport::without_recommended([QuantizationBits::Eight]),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3TtsCppSpeechAdapter {
    spec: Qwen3TtsCppSpeechSpec,
}

impl Qwen3TtsCppSpeechAdapter {
    pub fn qwen3_tts_cpp_0_6b() -> Self {
        Self {
            spec: Qwen3TtsCppSpeechSpec::qwen3_tts_cpp_0_6b(),
        }
    }
}

#[async_trait]
impl BackendAdapter for Qwen3TtsCppSpeechAdapter {
    fn supported_formats(&self) -> &[CheckpointFormat] {
        &QWEN3_TTS_CPP_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Qwen3TtsCpp
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

        let artifacts = resolve_gguf_artifacts(checkpoint)?;
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
pub struct Qwen3TtsCppSpeechBundle {
    metadata: BundleMetadata,
    hf_repo: &'static str,
}

impl Qwen3TtsCppSpeechBundle {
    pub fn new(spec: Qwen3TtsCppSpeechSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id,
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities,
                quantization: spec.quantization,
            },
            hf_repo: spec.hf_repo,
        }
    }
}

#[async_trait]
impl ModelBundle for Qwen3TtsCppSpeechBundle {
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
            configure_artifact_policy(self.hf_repo, policy)?
        } else {
            tracing::warn!(
                "no artifact_policy provided for qwen3-tts.cpp bundle; defaulting to LocalOnly with current working directory"
            );
            configure_artifact_policy(
                self.hf_repo,
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

struct Qwen3TtsCppHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<Qwen3TtsCppRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

#[derive(Clone, Debug, Default)]
struct SpeechMetrics {
    runtime: RuntimeMetricState,
}

#[async_trait]
impl BundleHandle for Qwen3TtsCppHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "qwen3-tts-cpp-metric-snapshot").clone();
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
impl SpeechModel for Qwen3TtsCppHandle {
    async fn open_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Box<dyn SpeechStream>, ModelError> {
        let started_at = Instant::now();
        let pcm = self.runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&self.metrics, "qwen3-tts-cpp-open-stream");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(Box::new(Qwen3TtsCppSpeechStream::new(
            self.runtime.audio_spec.clone(),
            pcm,
        )))
    }
}

fn new_speech_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    runtime: Arc<Qwen3TtsCppRuntime>,
) -> Box<dyn BundleHandle> {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "qwen3-tts-cpp-start");
        observe_memory(&mut state.runtime);
    }

    Box::new(Qwen3TtsCppHandle {
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

struct Qwen3TtsCppRuntime {
    engine: Mutex<Qwen3TtsEngine>,
    audio_spec: AudioSpec,
}

impl Qwen3TtsCppRuntime {
    fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<u8>, ModelError> {
        validate_request(request)?;
        let params = GenerationParams::default();

        let samples = match &request.conditioning {
            None => self
                .engine
                .lock()
                .map_err(|_| poison_lock("qwen3-tts-cpp-engine"))?
                .synthesize(&request.text, &params)?,
            Some(VoiceConditioning::SpeakerId(_)) => {
                return Err(ModelError::InvalidConfiguration(
                    "qwen3-tts.cpp does not support VoiceConditioning::SpeakerId; use ReferenceAudio for cloning".into(),
                ));
            }
            Some(VoiceConditioning::ReferenceAudio {
                audio_spec,
                pcm,
                reference_text,
            }) => {
                if reference_text.is_some() {
                    tracing::debug!(
                        "qwen3-tts.cpp backend ignores reference_text; voice cloning uses reference audio only"
                    );
                }

                let decoded = decode_pcm_to_f32(pcm, audio_spec.encoding)?;
                let mono = downmix_to_mono(&decoded, audio_spec.channels);
                let resampled =
                    resample_mono(&mono, audio_spec.sample_rate_hz, DEFAULT_SAMPLE_RATE_HZ);

                if resampled.is_empty() {
                    return Err(ModelError::InvalidConfiguration(
                        "reference audio became empty after decode/downmix/resample".into(),
                    ));
                }

                self.engine
                    .lock()
                    .map_err(|_| poison_lock("qwen3-tts-cpp-engine"))?
                    .synthesize_with_voice_samples(&request.text, &resampled, &params)?
            }
        };

        Ok(encode_pcm(&samples, self.audio_spec.encoding))
    }
}

fn load_runtime(artifacts: &Qwen3TtsCppArtifactPaths) -> Result<Qwen3TtsCppRuntime, ModelError> {
    tracing::debug!(
        model = %artifacts.model.display(),
        tokenizer = %artifacts.tokenizer.display(),
        model_dir = %artifacts.model_dir.display(),
        "loading qwen3-tts.cpp runtime"
    );
    let engine = Qwen3TtsEngine::new(&artifacts.model_dir)?;
    Ok(Qwen3TtsCppRuntime {
        engine: Mutex::new(engine),
        audio_spec: audio_spec(),
    })
}

struct Qwen3TtsCppSpeechStream {
    audio_spec: AudioSpec,
    pcm: Vec<u8>,
    next_offset: usize,
    next_sequence: u64,
    finished: bool,
}

impl Qwen3TtsCppSpeechStream {
    fn new(audio_spec: AudioSpec, pcm: Vec<u8>) -> Self {
        Self {
            audio_spec,
            pcm,
            next_offset: 0,
            next_sequence: 0,
            finished: false,
        }
    }

    fn chunk_size_bytes(&self) -> usize {
        let samples_per_chunk =
            (self.audio_spec.sample_rate_hz as usize * OUTPUT_CHUNK_DURATION_MS as usize) / 1000;
        let bytes_per_sample = match self.audio_spec.encoding {
            PcmEncoding::S16Le => 2,
            PcmEncoding::F32Le => 4,
        };
        samples_per_chunk * bytes_per_sample * self.audio_spec.channels as usize
    }
}

#[async_trait]
impl SpeechStream for Qwen3TtsCppSpeechStream {
    fn audio_spec(&self) -> &AudioSpec {
        &self.audio_spec
    }

    async fn next_chunk(&mut self) -> Result<Option<PcmChunk>, ModelError> {
        if self.finished {
            return Ok(None);
        }

        if self.pcm.is_empty() {
            self.finished = true;
            return Ok(Some(PcmChunk {
                data: Vec::new(),
                sequence: self.next_sequence,
                end_of_stream: true,
            }));
        }

        let chunk_size = self.chunk_size_bytes().max(1);
        let end = (self.next_offset + chunk_size).min(self.pcm.len());
        let data = self.pcm[self.next_offset..end].to_vec();
        let end_of_stream = end >= self.pcm.len();
        let chunk = PcmChunk {
            data,
            sequence: self.next_sequence,
            end_of_stream,
        };

        self.next_offset = end;
        self.next_sequence += 1;
        if end_of_stream {
            self.finished = true;
        }

        Ok(Some(chunk))
    }

    async fn finish(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

fn validate_request(request: &SpeechRequest) -> Result<(), ModelError> {
    if request.text.trim().is_empty() {
        return Err(ModelError::InvalidConfiguration(
            "speech request text must not be empty".into(),
        ));
    }

    validate_params(&request.params)
}

fn validate_params(params: &SpeechParams) -> Result<(), ModelError> {
    if params.speaking_rate.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "qwen3-tts.cpp backend does not support SpeechParams.speaking_rate".into(),
        ));
    }
    if params.seed.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "qwen3-tts.cpp backend does not support SpeechParams.seed".into(),
        ));
    }
    Ok(())
}

fn poison_lock(name: &str) -> ModelError {
    ModelError::BackendExecution {
        backend: "qwen3-tts-cpp",
        operation: "lock-mutex",
        message: format!("mutex poisoned: {name}"),
    }
}

#[derive(Clone, Copy, Debug)]
struct GenerationParams {
    max_audio_tokens: i32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    n_threads: i32,
    repetition_penalty: f32,
    language_id: i32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .max(1);
        Self {
            max_audio_tokens: 4096,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            n_threads,
            repetition_penalty: 1.05,
            language_id: DEFAULT_LANGUAGE_ID_EN,
        }
    }
}

struct Qwen3TtsEngine {
    raw: *mut ffi::Qwen3Tts,
}

// SAFETY: `Qwen3TtsEngine` owns the native pointer and all access goes through
// `&mut self`, so the Rust side never aliases concurrent calls into the same
// engine. Destruction also stays tied to ownership of the wrapper.
unsafe impl Send for Qwen3TtsEngine {}

impl Qwen3TtsEngine {
    fn new(model_dir: &std::path::Path) -> Result<Self, ModelError> {
        let model_dir_utf8 = model_dir.to_str().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "qwen3-tts.cpp model dir `{}` contains non-UTF-8 characters",
                model_dir.display()
            ))
        })?;
        let model_dir_cstr = CString::new(model_dir_utf8).map_err(|_| {
            ModelError::InvalidConfiguration(format!(
                "qwen3-tts.cpp model dir `{}` contains interior NUL bytes",
                model_dir.display()
            ))
        })?;

        let raw = unsafe { ffi::qwen3_tts_create(model_dir_cstr.as_ptr(), 0) };
        if raw.is_null() {
            return Err(ModelError::BackendInitialization {
                backend: "qwen3-tts-cpp",
                message: format!(
                    "failed to create qwen3-tts.cpp engine from `{}`; verify GGUF artifacts are valid",
                    model_dir.display()
                ),
            });
        }

        let loaded = unsafe { ffi::qwen3_tts_is_loaded(raw) };
        if loaded == 0 {
            let engine = Self { raw };
            let message = engine.last_error_message();
            return Err(ModelError::BackendInitialization {
                backend: "qwen3-tts-cpp",
                message: if message.is_empty() {
                    "qwen3-tts.cpp reported unloaded state after create".into()
                } else {
                    message
                },
            });
        }

        Ok(Self { raw })
    }

    fn synthesize(
        &mut self,
        text: &str,
        params: &GenerationParams,
    ) -> Result<Vec<f32>, ModelError> {
        let text_cstr = c_string(text, "speech request text")?;
        let native_params = NativeParams::from_generation_params(params)?;

        let audio =
            unsafe { ffi::qwen3_tts_synthesize(self.raw, text_cstr.as_ptr(), &native_params.raw) };
        self.extract_audio(audio)
    }

    fn synthesize_with_voice_samples(
        &mut self,
        text: &str,
        ref_samples: &[f32],
        params: &GenerationParams,
    ) -> Result<Vec<f32>, ModelError> {
        let text_cstr = c_string(text, "speech request text")?;
        let native_params = NativeParams::from_generation_params(params)?;

        let audio = unsafe {
            ffi::qwen3_tts_synthesize_with_voice_samples(
                self.raw,
                text_cstr.as_ptr(),
                ref_samples.as_ptr(),
                ref_samples.len() as i32,
                &native_params.raw,
            )
        };
        self.extract_audio(audio)
    }

    fn extract_audio(&mut self, audio: *mut ffi::Qwen3TtsAudio) -> Result<Vec<f32>, ModelError> {
        if audio.is_null() {
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts-cpp",
                operation: "synthesize",
                message: self.last_error_message(),
            });
        }

        let audio_ref = unsafe { &*audio };
        if audio_ref.n_samples < 0 {
            unsafe { ffi::qwen3_tts_free_audio(audio) };
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts-cpp",
                operation: "decode-audio-buffer",
                message: "qwen3-tts.cpp returned negative sample count".into(),
            });
        }
        if audio_ref.samples.is_null() {
            unsafe { ffi::qwen3_tts_free_audio(audio) };
            return Err(ModelError::BackendExecution {
                backend: "qwen3-tts-cpp",
                operation: "decode-audio-buffer",
                message: "qwen3-tts.cpp returned a null audio sample buffer".into(),
            });
        }

        let sample_rate = audio_ref.sample_rate;
        if sample_rate != DEFAULT_SAMPLE_RATE_HZ as i32 {
            tracing::warn!(
                "qwen3-tts.cpp returned sample_rate={} instead of {}",
                sample_rate,
                DEFAULT_SAMPLE_RATE_HZ
            );
        }

        let samples =
            unsafe { slice::from_raw_parts(audio_ref.samples, audio_ref.n_samples as usize) }
                .to_vec();
        unsafe { ffi::qwen3_tts_free_audio(audio) };
        Ok(samples)
    }

    fn last_error_message(&self) -> String {
        let ptr = unsafe { ffi::qwen3_tts_get_error(self.raw) };
        if ptr.is_null() {
            return "qwen3-tts.cpp returned a null error pointer".into();
        }

        let c_str = unsafe { CStr::from_ptr(ptr) };
        c_str.to_string_lossy().to_string()
    }
}

impl Drop for Qwen3TtsEngine {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::qwen3_tts_destroy(self.raw) };
        }
    }
}

struct NativeParams {
    raw: ffi::Qwen3TtsParams,
}

impl NativeParams {
    fn from_generation_params(params: &GenerationParams) -> Result<Self, ModelError> {
        let mut raw = ffi::Qwen3TtsParams {
            max_audio_tokens: 0,
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            n_threads: 0,
            repetition_penalty: 0.0,
            language_id: 0,
        };

        unsafe { ffi::qwen3_tts_default_params(&mut raw) };
        raw.max_audio_tokens = params.max_audio_tokens;
        raw.temperature = params.temperature;
        raw.top_p = params.top_p;
        raw.top_k = params.top_k;
        raw.n_threads = params.n_threads;
        raw.repetition_penalty = params.repetition_penalty;
        raw.language_id = params.language_id;

        if raw.n_threads <= 0 {
            return Err(ModelError::InvalidConfiguration(
                "qwen3-tts.cpp n_threads must be positive".into(),
            ));
        }

        Ok(Self { raw })
    }
}

fn c_string(value: &str, label: &str) -> Result<CString, ModelError> {
    CString::new(value).map_err(|_| {
        ModelError::InvalidConfiguration(format!("{label} contains interior NUL bytes"))
    })
}

mod ffi {
    use super::c_char;

    #[repr(C)]
    pub struct Qwen3Tts {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Qwen3TtsParams {
        pub max_audio_tokens: i32,
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: i32,
        pub n_threads: i32,
        pub repetition_penalty: f32,
        pub language_id: i32,
    }

    #[repr(C)]
    pub struct Qwen3TtsAudio {
        pub samples: *const f32,
        pub n_samples: i32,
        pub sample_rate: i32,
    }

    #[link(name = "qwen3tts")]
    unsafe extern "C" {
        pub fn qwen3_tts_default_params(params: *mut Qwen3TtsParams);
        pub fn qwen3_tts_create(model_dir: *const c_char, n_threads: i32) -> *mut Qwen3Tts;
        pub fn qwen3_tts_is_loaded(tts: *const Qwen3Tts) -> i32;
        pub fn qwen3_tts_synthesize(
            tts: *mut Qwen3Tts,
            text: *const c_char,
            params: *const Qwen3TtsParams,
        ) -> *mut Qwen3TtsAudio;
        pub fn qwen3_tts_synthesize_with_voice_samples(
            tts: *mut Qwen3Tts,
            text: *const c_char,
            ref_samples: *const f32,
            n_ref_samples: i32,
            params: *const Qwen3TtsParams,
        ) -> *mut Qwen3TtsAudio;
        pub fn qwen3_tts_get_error(tts: *const Qwen3Tts) -> *const c_char;
        pub fn qwen3_tts_free_audio(audio: *mut Qwen3TtsAudio);
        pub fn qwen3_tts_destroy(tts: *mut Qwen3Tts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_uses_expected_bundle_identity() {
        let spec = Qwen3TtsCppSpeechSpec::qwen3_tts_cpp_0_6b();

        assert_eq!(spec.id.as_str(), "qwen3_tts_cpp_0_6b");
        assert_eq!(spec.display_name, "Qwen3-TTS CPP 0.6B");
        assert_eq!(spec.hf_repo, "koboldcpp/tts");
        assert!(spec.capabilities.supports(CapabilityKind::Speech));
    }

    #[test]
    fn validate_request_rejects_empty_text() {
        let error = validate_request(&SpeechRequest {
            text: "   ".into(),
            params: SpeechParams::default(),
            conditioning: None,
        })
        .expect_err("empty text should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("must not be empty")
        ));
    }

    #[test]
    fn validate_params_rejects_unimplemented_controls() {
        let error = validate_params(&SpeechParams {
            speaking_rate: Some(1.25),
            seed: None,
        })
        .expect_err("speaking rate should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("speaking_rate")
        ));
    }
}
