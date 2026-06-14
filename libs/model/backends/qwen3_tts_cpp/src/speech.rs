use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::slice;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::typed::{
    AudioBuf, BufferedSpeechChunkStream, BufferedSpeechSynthesizer, BufferedVoiceCloneSynthesizer,
    CloneReference, Mono, SpeechSynthesizer, SynthesisRequest, VoiceCloneSynthesizer,
};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, CheckpointFormat, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationScheme, QuantizationSupport,
    ResolvedCheckpoint, RuntimeAcceleratorObservation, SpeechParams, StartOptions, UnsupportedChat,
    UnsupportedCompletion, UnsupportedEmbeddings,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory, resample_mono,
    resolve_gguf_artifacts, Qwen3TtsCppArtifactPaths, RuntimeMetricState, DEFAULT_SAMPLE_RATE_HZ,
};

const QWEN3_TTS_CPP_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Gguf];
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const DEFAULT_LANGUAGE_ID_EN: i32 = 2050;
const REFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;

pub type Qwen3TtsCppSpeechStream = BufferedSpeechChunkStream<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>;

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
            capabilities: Capabilities::speech_buffered_with_voice_clone(),
            quantization: QuantizationSupport::with_recommended(
                [QuantizationScheme::GgufQ8_0, QuantizationScheme::Fp16],
                QuantizationScheme::GgufQ8_0,
            )
            .unwrap_or_else(|e| {
                tracing::error!("curated quantization construction failed (this is a bug): {e}");
                QuantizationSupport::without_recommended([
                    QuantizationScheme::GgufQ8_0,
                    QuantizationScheme::Fp16,
                ])
            }),
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
    type Handle = Qwen3TtsCppHandle;

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
    ) -> Result<Self::Handle, ModelError> {
        self.spec
            .quantization
            .resolve(options.quantization_scheme, &identity.id)?;

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
    type Handle = Qwen3TtsCppHandle;

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

impl Qwen3TtsCppSpeechBundle {
    pub async fn start_typed(
        &self,
        options: StartOptions,
    ) -> Result<Qwen3TtsCppHandle, ModelError> {
        self.metadata
            .quantization
            .resolve(options.quantization_scheme, &self.metadata.id)?;

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

pub struct Qwen3TtsCppHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Arc<Qwen3TtsCppRuntime>,
    metrics: Arc<Mutex<SpeechMetrics>>,
}

impl Qwen3TtsCppHandle {
    pub async fn shutdown(self) -> Result<(), ModelError> {
        <Self as BundleHandle>::shutdown(self).await
    }

    async fn synthesize_pcm(
        &self,
        request: SynthesisRequest,
    ) -> Result<AudioBuf<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        let started_at = Instant::now();
        let pcm = runtime.synthesize(&request)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&metrics, "qwen3-tts-cpp-typed-synthesize");
            observe_latency(&mut state.runtime, elapsed);
        }

        Ok(AudioBuf::new(pcm))
    }

    async fn synthesize_pcm_with_reference(
        &self,
        request: SynthesisRequest,
        reference: CloneReference<REFERENCE_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<AudioBuf<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>, ModelError> {
        let runtime = Arc::clone(&self.runtime);
        let metrics = Arc::clone(&self.metrics);

        let started_at = Instant::now();
        let pcm = runtime.synthesize_with_reference(&request, &reference)?;
        let elapsed = started_at.elapsed();

        {
            let mut state = lock_metrics(&metrics, "qwen3-tts-cpp-typed-clone");
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
impl BundleHandle for Qwen3TtsCppHandle {
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

    fn accelerator_observation(&self) -> Option<RuntimeAcceleratorObservation> {
        if cfg!(feature = "cuda") {
            Some(RuntimeAcceleratorObservation {
                backend_mode: "qwen3_tts_cpp:cuda".to_owned(),
                offload: Some("ggml_cuda=on".to_owned()),
                selected_device: Some("0".to_owned()),
            })
        } else if cfg!(target_os = "macos") {
            Some(RuntimeAcceleratorObservation {
                backend_mode: "qwen3_tts_cpp:metal".to_owned(),
                offload: Some("ggml_metal=on".to_owned()),
                selected_device: Some("0".to_owned()),
            })
        } else {
            Some(RuntimeAcceleratorObservation {
                backend_mode: "qwen3_tts_cpp:cpu".to_owned(),
                offload: Some("accelerator_feature=none".to_owned()),
                selected_device: None,
            })
        }
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
    runtime: Arc<Qwen3TtsCppRuntime>,
) -> Qwen3TtsCppHandle {
    let metrics = Arc::new(Mutex::new(SpeechMetrics::default()));
    {
        let mut state = lock_metrics(&metrics, "qwen3-tts-cpp-start");
        observe_memory(&mut state.runtime);
    }

    Qwen3TtsCppHandle {
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

struct Qwen3TtsCppRuntime {
    engine: Mutex<Qwen3TtsEngine>,
}

impl Qwen3TtsCppRuntime {
    fn synthesize(&self, request: &SynthesisRequest) -> Result<Vec<f32>, ModelError> {
        validate_request(request)?;
        let params = GenerationParams::default();

        self.engine
            .lock()
            .map_err(|_| poison_lock("qwen3-tts-cpp-engine"))?
            .synthesize(&request.text, &params)
    }

    fn synthesize_with_reference(
        &self,
        request: &SynthesisRequest,
        reference: &CloneReference<REFERENCE_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Vec<f32>, ModelError> {
        validate_request(request)?;
        let params = GenerationParams::default();

        if reference.transcript.is_some() {
            tracing::debug!(
                "qwen3-tts.cpp backend ignores clone transcript and uses reference audio only"
            );
        }

        if reference.audio.samples().is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "reference audio became empty after normalization".into(),
            ));
        }

        let resampled = resample_mono(
            reference.audio.samples(),
            REFERENCE_SAMPLE_RATE_HZ,
            DEFAULT_SAMPLE_RATE_HZ,
        );
        if resampled.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "reference audio became empty after resample".into(),
            ));
        }

        self.engine
            .lock()
            .map_err(|_| poison_lock("qwen3-tts-cpp-engine"))?
            .synthesize_with_voice_samples(&request.text, &resampled, &params)
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
    })
}

impl BufferedSpeechSynthesizer for Qwen3TtsCppHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>;

    async fn synthesize_buffered(
        &self,
        request: Self::Request,
    ) -> Result<Self::Output, ModelError> {
        self.synthesize_pcm(request).await
    }
}

impl SpeechSynthesizer for Qwen3TtsCppHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>;
    type Stream = Qwen3TtsCppSpeechStream;

    async fn synthesize(&self, request: Self::Request) -> Result<Self::Stream, ModelError> {
        Ok(Qwen3TtsCppSpeechStream::new(
            self.synthesize_pcm(request).await?,
            OUTPUT_CHUNK_DURATION_MS,
        ))
    }
}

impl BufferedVoiceCloneSynthesizer<REFERENCE_SAMPLE_RATE_HZ, Mono> for Qwen3TtsCppHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>;

    async fn synthesize_with_reference_buffered(
        &self,
        request: Self::Request,
        reference: CloneReference<REFERENCE_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Self::Output, ModelError> {
        self.synthesize_pcm_with_reference(request, reference).await
    }
}

impl VoiceCloneSynthesizer<REFERENCE_SAMPLE_RATE_HZ, Mono> for Qwen3TtsCppHandle {
    type Request = SynthesisRequest;
    type Output = AudioBuf<f32, DEFAULT_SAMPLE_RATE_HZ, Mono>;
    type Stream = Qwen3TtsCppSpeechStream;

    async fn synthesize_with_reference(
        &self,
        request: Self::Request,
        reference: CloneReference<REFERENCE_SAMPLE_RATE_HZ, Mono>,
    ) -> Result<Self::Stream, ModelError> {
        Ok(Qwen3TtsCppSpeechStream::new(
            self.synthesize_pcm_with_reference(request, reference)
                .await?,
            OUTPUT_CHUNK_DURATION_MS,
        ))
    }
}

fn validate_request(request: &SynthesisRequest) -> Result<(), ModelError> {
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
        let error = validate_request(&SynthesisRequest {
            text: "   ".into(),
            params: SpeechParams::default(),
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
