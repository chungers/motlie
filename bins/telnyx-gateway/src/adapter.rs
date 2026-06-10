#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fmt, str::FromStr};

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
use anyhow::{bail, Context};
use async_trait::async_trait;
use clap::ValueEnum;
use motlie_model::typed::{AudioBuf, Mono};
#[cfg(feature = "whisper")]
use motlie_model::typed::{AudioTransform, BatchTranscriber, I16ToF32};
#[cfg(any(feature = "sherpa", feature = "moonshine"))]
use motlie_model::typed::{StreamingTranscriber, TranscriptionSession};
#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
use motlie_model::{ArtifactPolicy, ModelError, StartOptions, TranscriptionParams};
use motlie_voice::app::TranscriptEvent;
#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
use tokio::sync::Mutex;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AsrTranscriptEvent {
    pub event: TranscriptEvent,
    pub decision: AsrTranscriptDecision,
}

impl AsrTranscriptEvent {
    pub fn emit(event: TranscriptEvent) -> Self {
        Self {
            event,
            decision: AsrTranscriptDecision::Emit,
        }
    }

    pub fn suppress(
        event: TranscriptEvent,
        reason: AsrTranscriptSuppressionReason,
        reset_session: bool,
    ) -> Self {
        Self {
            event,
            decision: AsrTranscriptDecision::Suppress {
                reason,
                reset_session,
            },
        }
    }

    pub fn is_suppressed(&self) -> bool {
        matches!(self.decision, AsrTranscriptDecision::Suppress { .. })
    }

    pub fn requires_session_reset(&self) -> bool {
        matches!(
            self.decision,
            AsrTranscriptDecision::Suppress {
                reset_session: true,
                ..
            }
        )
    }

    pub fn suppression_reason(&self) -> Option<AsrTranscriptSuppressionReason> {
        match self.decision {
            AsrTranscriptDecision::Emit => None,
            AsrTranscriptDecision::Suppress { reason, .. } => Some(reason),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsrTranscriptDecision {
    Emit,
    Suppress {
        reason: AsrTranscriptSuppressionReason,
        reset_session: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsrTranscriptSuppressionReason {
    RepeatedTokenHallucination,
}

impl AsrTranscriptSuppressionReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::RepeatedTokenHallucination => "repeated_token_hallucination",
        }
    }
}

#[async_trait]
pub trait InboundAsrSession: Send {
    async fn ingest(
        &mut self,
        audio: AudioBuf<i16, 16_000, Mono>,
    ) -> anyhow::Result<Vec<AsrTranscriptEvent>>;

    async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>>;
}

#[async_trait]
pub trait InboundAsrFactory: Send + Sync {
    // justification: the gateway chooses the concrete ASR backend at process startup while media handlers need a shared backend-independent session factory.
    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>>;

    async fn warm(&self) -> anyhow::Result<()> {
        self.open_session().await.map(|_| ())
    }
}

// justification: media WebSocket tasks share one process-selected ASR factory without coupling Telnyx wiring to a concrete model backend.
pub type SharedAsrFactory = Arc<dyn InboundAsrFactory>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum LiveAsrBackend {
    #[value(name = "sherpa-2023", alias = "sherpa-zipformer-2023")]
    Sherpa2023,
    #[default]
    #[value(name = "kroko-2025", alias = "sherpa-zipformer-kroko-2025")]
    Kroko2025,
}

impl LiveAsrBackend {
    pub const fn available() -> [Self; 2] {
        [Self::Kroko2025, Self::Sherpa2023]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sherpa2023 => "sherpa-2023",
            Self::Kroko2025 => "kroko-2025",
        }
    }

    pub fn model_label(self) -> &'static str {
        self.artifact().label()
    }

    pub fn artifact(self) -> SherpaAsrArtifact {
        match self {
            Self::Sherpa2023 => SherpaAsrArtifact::ZipformerEn20230626,
            Self::Kroko2025 => SherpaAsrArtifact::ZipformerEnKroko20250806,
        }
    }
}

impl fmt::Display for LiveAsrBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("unsupported live ASR backend `{value}`; expected sherpa-2023 or kroko-2025")]
pub struct LiveAsrBackendParseError {
    value: String,
}

impl FromStr for LiveAsrBackend {
    type Err = LiveAsrBackendParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "sherpa-2023" | "sherpa-zipformer-2023" | "sherpa-zipformer-en-2023-06-26" => {
                Ok(Self::Sherpa2023)
            }
            "kroko-2025"
            | "sherpa-zipformer-kroko-2025"
            | "sherpa-zipformer-en-kroko-2025-08-06" => Ok(Self::Kroko2025),
            other => Err(LiveAsrBackendParseError {
                value: other.to_string(),
            }),
        }
    }
}

#[derive(Clone)]
pub struct AsrRegistry {
    sherpa_2023: SharedAsrFactory,
    kroko_2025: SharedAsrFactory,
}

pub type SharedAsrRegistry = Arc<AsrRegistry>;

impl AsrRegistry {
    pub fn new(sherpa_2023: SharedAsrFactory, kroko_2025: SharedAsrFactory) -> Self {
        Self {
            sherpa_2023,
            kroko_2025,
        }
    }

    pub fn factory(&self, backend: LiveAsrBackend) -> SharedAsrFactory {
        match backend {
            LiveAsrBackend::Sherpa2023 => self.sherpa_2023.clone(),
            LiveAsrBackend::Kroko2025 => self.kroko_2025.clone(),
        }
    }

    pub async fn open_session(
        &self,
        backend: LiveAsrBackend,
    ) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        self.factory(backend).open_session().await
    }

    pub async fn warm(&self, backend: LiveAsrBackend) -> anyhow::Result<()> {
        self.factory(backend).warm().await
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SherpaAsrArtifact {
    #[default]
    ZipformerEn20230626,
    ZipformerEnKroko20250806,
}

impl SherpaAsrArtifact {
    pub fn label(self) -> &'static str {
        match self {
            Self::ZipformerEn20230626 => "sherpa-zipformer-en-2023-06-26",
            Self::ZipformerEnKroko20250806 => "sherpa-zipformer-en-kroko-2025-08-06",
        }
    }

    #[cfg(feature = "sherpa")]
    fn asr_model(self) -> motlie_models::AsrModels {
        match self {
            Self::ZipformerEn20230626 => motlie_models::AsrModels::SherpaOnnxStreamingEn,
            Self::ZipformerEnKroko20250806 => {
                motlie_models::AsrModels::SherpaOnnxStreamingEnKroko2025
            }
        }
    }
}

pub struct UnavailableAsrFactory {
    message: &'static str,
}

impl UnavailableAsrFactory {
    pub fn new(message: &'static str) -> Self {
        Self { message }
    }
}

#[async_trait]
impl InboundAsrFactory for UnavailableAsrFactory {
    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        Err(anyhow::anyhow!(self.message))
    }
}

#[cfg(feature = "sherpa")]
pub struct SherpaAsrFactory {
    artifact_root: PathBuf,
    allow_download: bool,
    artifact: SherpaAsrArtifact,
    handle: Mutex<Option<Arc<motlie_model_sherpa_onnx::SherpaOnnxHandle>>>,
}

#[cfg(feature = "sherpa")]
impl SherpaAsrFactory {
    pub fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        Self::with_artifact(artifact_root, allow_download, SherpaAsrArtifact::default())
    }

    pub fn with_artifact(
        artifact_root: PathBuf,
        allow_download: bool,
        artifact: SherpaAsrArtifact,
    ) -> Self {
        Self {
            artifact_root,
            allow_download,
            artifact,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_sherpa_onnx::SherpaOnnxHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle =
            Arc::new(start_sherpa(&self.artifact_root, self.allow_download, self.artifact).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "sherpa")]
#[async_trait]
impl InboundAsrFactory for SherpaAsrFactory {
    async fn warm(&self) -> anyhow::Result<()> {
        self.handle().await.map(|_| ())
    }

    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        let handle = self.handle().await?;
        let session = handle
            .open_session(transcription_params(true))
            .await
            .context("open Sherpa streaming ASR session")?;
        Ok(Box::new(SherpaAsrSession { session }))
    }
}

#[cfg(feature = "sherpa")]
struct SherpaAsrSession {
    session: motlie_model_sherpa_onnx::SherpaOnnxStream,
}

#[cfg(feature = "sherpa")]
#[async_trait]
impl InboundAsrSession for SherpaAsrSession {
    async fn ingest(
        &mut self,
        audio: AudioBuf<i16, 16_000, Mono>,
    ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        let update = self
            .session
            .ingest(audio)
            .await
            .context("ingest audio into Sherpa ASR")?;
        Ok(update
            .map(events_from_update)
            .unwrap_or_default()
            .into_iter()
            .map(apply_sherpa_transcript_policy)
            .collect())
    }

    async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        let update = self
            .session
            .finish()
            .await
            .context("finish Sherpa ASR session")?;
        Ok(events_from_update(update)
            .into_iter()
            .map(apply_sherpa_transcript_policy)
            .collect())
    }
}

#[cfg(feature = "moonshine")]
pub struct MoonshineAsrFactory {
    artifact_root: PathBuf,
    allow_download: bool,
    handle: Mutex<Option<Arc<motlie_model_moonshine::MoonshineHandle>>>,
}

#[cfg(feature = "moonshine")]
impl MoonshineAsrFactory {
    pub fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        Self {
            artifact_root,
            allow_download,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_moonshine::MoonshineHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_moonshine(&self.artifact_root, self.allow_download).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "moonshine")]
#[async_trait]
impl InboundAsrFactory for MoonshineAsrFactory {
    async fn warm(&self) -> anyhow::Result<()> {
        self.handle().await.map(|_| ())
    }

    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        let handle = self.handle().await?;
        let session = handle
            .open_session(transcription_params(true))
            .await
            .context("open Moonshine streaming ASR session")?;
        Ok(Box::new(MoonshineAsrSession { session }))
    }
}

#[cfg(feature = "moonshine")]
struct MoonshineAsrSession {
    session: motlie_model_moonshine::MoonshineStream,
}

#[cfg(feature = "moonshine")]
#[async_trait]
impl InboundAsrSession for MoonshineAsrSession {
    async fn ingest(
        &mut self,
        audio: AudioBuf<i16, 16_000, Mono>,
    ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        let update = self
            .session
            .ingest(audio)
            .await
            .context("ingest audio into Moonshine ASR")?;
        Ok(update
            .map(events_from_update)
            .unwrap_or_default()
            .into_iter()
            .map(AsrTranscriptEvent::emit)
            .collect())
    }

    async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        let update = self
            .session
            .finish()
            .await
            .context("finish Moonshine ASR session")?;
        Ok(events_from_update(update)
            .into_iter()
            .map(AsrTranscriptEvent::emit)
            .collect())
    }
}

#[cfg(feature = "whisper")]
pub struct WhisperAsrFactory {
    artifact_root: PathBuf,
    allow_download: bool,
    handle: Mutex<Option<Arc<motlie_model_whisper_cpp::WhisperCppHandle>>>,
}

#[cfg(feature = "whisper")]
impl WhisperAsrFactory {
    pub fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        Self {
            artifact_root,
            allow_download,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_whisper_cpp::WhisperCppHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_whisper(&self.artifact_root, self.allow_download).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "whisper")]
#[async_trait]
impl InboundAsrFactory for WhisperAsrFactory {
    async fn warm(&self) -> anyhow::Result<()> {
        self.handle().await.map(|_| ())
    }

    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        Ok(Box::new(WhisperAsrSession {
            handle: self.handle().await?,
            samples: Vec::new(),
        }))
    }
}

#[cfg(feature = "whisper")]
struct WhisperAsrSession {
    handle: Arc<motlie_model_whisper_cpp::WhisperCppHandle>,
    samples: Vec<i16>,
}

#[cfg(feature = "whisper")]
#[async_trait]
impl InboundAsrSession for WhisperAsrSession {
    async fn ingest(
        &mut self,
        audio: AudioBuf<i16, 16_000, Mono>,
    ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        self.samples.extend(audio.into_samples());
        Ok(Vec::new())
    }

    async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        let audio = I16ToF32::<16_000, Mono>::new()
            .transform(AudioBuf::<i16, 16_000, Mono>::new(self.samples))
            .context("convert Whisper ASR input to f32")?;
        let update = self
            .handle
            .transcribe(audio, transcription_params(false))
            .await
            .context("run Whisper final-pass ASR")?;
        Ok(events_from_update(update)
            .into_iter()
            .map(AsrTranscriptEvent::emit)
            .collect())
    }
}

#[cfg(feature = "sherpa")]
async fn start_sherpa(
    artifact_root: &Path,
    allow_download: bool,
    artifact: SherpaAsrArtifact,
) -> anyhow::Result<motlie_model_sherpa_onnx::SherpaOnnxHandle> {
    let options = local_only_options(artifact_root);
    match start_sherpa_artifact(artifact, options).await {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                artifact = artifact.label(),
                "downloading Sherpa ONNX artifacts"
            );
            download_sherpa_artifact(artifact, artifact_root)?;
            start_sherpa_artifact(artifact, local_only_options(artifact_root))
                .await
                .map_err(anyhow::Error::from)
                .with_context(|| format!("start {} after downloading artifacts", artifact.label()))
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail_missing_artifacts(artifact.label(), artifact_root)
        }
        Err(err) => Err(anyhow::Error::from(err))
            .with_context(|| format!("start {} Sherpa ONNX", artifact.label())),
    }
}

#[cfg(feature = "moonshine")]
async fn start_moonshine(
    artifact_root: &Path,
    allow_download: bool,
) -> anyhow::Result<motlie_model_moonshine::MoonshineHandle> {
    match motlie_models::asr::moonshine_streaming_en::start_typed(local_only_options(artifact_root))
        .await
    {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                artifact = "moonshine-streaming-en",
                "downloading Moonshine artifacts"
            );
            download_asr_artifact(
                motlie_models::AsrModels::MoonshineStreamingEn,
                artifact_root,
            )?;
            motlie_models::asr::moonshine_streaming_en::start_typed(local_only_options(
                artifact_root,
            ))
            .await
            .map_err(anyhow::Error::from)
            .context("start Moonshine after downloading artifacts")
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail_missing_artifacts("moonshine-streaming-en", artifact_root)
        }
        Err(err) => Err(anyhow::Error::from(err)).context("start Moonshine ASR"),
    }
}

#[cfg(feature = "whisper")]
async fn start_whisper(
    artifact_root: &Path,
    allow_download: bool,
) -> anyhow::Result<motlie_model_whisper_cpp::WhisperCppHandle> {
    match motlie_models::asr::whisper_base_en::start_typed(local_only_options(artifact_root)).await
    {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                artifact = "whisper-base-en",
                "downloading Whisper artifacts"
            );
            download_asr_artifact(motlie_models::AsrModels::WhisperBaseEn, artifact_root)?;
            motlie_models::asr::whisper_base_en::start_typed(local_only_options(artifact_root))
                .await
                .map_err(anyhow::Error::from)
                .context("start Whisper after downloading artifacts")
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail_missing_artifacts("whisper-base-en", artifact_root)
        }
        Err(err) => Err(anyhow::Error::from(err)).context("start Whisper ASR"),
    }
}

#[cfg(feature = "sherpa")]
async fn start_sherpa_artifact(
    artifact: SherpaAsrArtifact,
    options: StartOptions,
) -> Result<motlie_model_sherpa_onnx::SherpaOnnxHandle, ModelError> {
    match artifact {
        SherpaAsrArtifact::ZipformerEn20230626 => {
            motlie_models::asr::sherpa_onnx_streaming_en::start_typed(options).await
        }
        SherpaAsrArtifact::ZipformerEnKroko20250806 => {
            motlie_models::asr::sherpa_onnx_streaming_en_kroko_2025::start_typed(options).await
        }
    }
}

#[cfg(feature = "sherpa")]
fn download_sherpa_artifact(
    artifact: SherpaAsrArtifact,
    artifact_root: &Path,
) -> anyhow::Result<()> {
    download_asr_artifact(artifact.asr_model(), artifact_root)
        .with_context(|| format!("download {} Sherpa ONNX artifacts", artifact.label()))
}

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
fn download_asr_artifact(
    model: motlie_models::AsrModels,
    artifact_root: &Path,
) -> anyhow::Result<()> {
    let catalog = motlie_models::Catalog::with_defaults();
    motlie_models::download_bundle_artifacts(&catalog, &model.bundle_id(), artifact_root)
        .map(|_| ())
        .map_err(anyhow::Error::from)
}

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
fn local_only_options(artifact_root: &Path) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root.to_path_buf(),
        }),
        ..Default::default()
    }
}

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
fn missing_local_artifacts(error: &ModelError) -> bool {
    match error {
        ModelError::InvalidConfiguration(message) => {
            message.contains(motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX)
        }
        _ => false,
    }
}

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
fn bail_missing_artifacts<T>(label: &str, artifact_root: &Path) -> anyhow::Result<T> {
    bail!(
        "{} missing for {} under '{}'; rerun without --no-asr-download or preinstall artifacts",
        motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX,
        label,
        artifact_root.display()
    )
}

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
fn transcription_params(emit_partials: bool) -> TranscriptionParams {
    TranscriptionParams {
        language: Some("en".to_string()),
        emit_partials,
    }
}

#[cfg(any(feature = "sherpa", feature = "moonshine", feature = "whisper"))]
fn events_from_update(update: motlie_model::TranscriptionUpdate) -> Vec<TranscriptEvent> {
    update
        .segments
        .iter()
        .filter_map(|segment| {
            let text = segment.text.trim();
            if text.is_empty() {
                return None;
            }
            let one_segment_update = motlie_model::TranscriptionUpdate {
                segments: vec![segment.clone()],
            };
            if segment.final_segment {
                Some(TranscriptEvent::Final {
                    text: text.to_string(),
                    update: one_segment_update,
                })
            } else {
                Some(TranscriptEvent::Partial {
                    text: text.to_string(),
                    update: one_segment_update,
                })
            }
        })
        .collect()
}

#[cfg(feature = "sherpa")]
fn apply_sherpa_transcript_policy(event: TranscriptEvent) -> AsrTranscriptEvent {
    if looks_like_repeated_token_hallucination(event.text()) {
        AsrTranscriptEvent::suppress(
            event,
            AsrTranscriptSuppressionReason::RepeatedTokenHallucination,
            true,
        )
    } else {
        AsrTranscriptEvent::emit(event)
    }
}

#[cfg(feature = "sherpa")]
fn looks_like_repeated_token_hallucination(text: &str) -> bool {
    const REPEATED_TOKEN_RUN_THRESHOLD: usize = 16;
    const REPEATED_Q_RUN_THRESHOLD: usize = 8;

    let mut previous = None;
    let mut run = 0usize;
    let mut max_run = 0usize;
    let mut chars = 0usize;
    let mut q_count = 0usize;

    for ch in text.chars().filter(|ch| !ch.is_whitespace()) {
        chars += 1;
        if ch == 'Q' {
            q_count += 1;
        }
        if previous == Some(ch) {
            run += 1;
        } else {
            previous = Some(ch);
            run = 1;
        }
        max_run = max_run.max(run);
    }

    max_run >= REPEATED_TOKEN_RUN_THRESHOLD
        || (q_count >= REPEATED_Q_RUN_THRESHOLD
            && q_count.saturating_mul(3) >= chars.saturating_mul(2))
}

#[derive(Default)]
pub struct EchoAsrFactory;

#[async_trait]
impl InboundAsrFactory for EchoAsrFactory {
    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        Ok(Box::<EchoAsrSession>::default())
    }
}

#[derive(Default)]
struct EchoAsrSession {
    samples: usize,
}

#[async_trait]
impl InboundAsrSession for EchoAsrSession {
    async fn ingest(
        &mut self,
        audio: AudioBuf<i16, 16_000, Mono>,
    ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        self.samples += audio.samples().len();
        if self.samples >= 16_000 {
            Ok(vec![AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                text: format!("received {} samples", self.samples),
                update: motlie_model::TranscriptionUpdate::default(),
            })])
        } else {
            Ok(Vec::new())
        }
    }

    async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
        Ok(vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
            text: format!("received {} samples", self.samples),
            update: motlie_model::TranscriptionUpdate::default(),
        })])
    }
}

pub fn default_artifact_root(cli_root: Option<PathBuf>) -> PathBuf {
    if let Some(root) = cli_root {
        return root;
    }
    if let Some(root) = std::env::var_os("MOTLIE_VOICE_ARTIFACT_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(".agents/skills/voice/artifacts/hf-cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn final_event(text: &str) -> TranscriptEvent {
        TranscriptEvent::Final {
            text: text.to_string(),
            update: motlie_model::TranscriptionUpdate::default(),
        }
    }

    #[test]
    fn pass_through_event_has_no_suppression_or_reset() {
        let event = AsrTranscriptEvent::emit(final_event("MEQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"));

        assert!(!event.is_suppressed());
        assert!(!event.requires_session_reset());
        assert_eq!(event.suppression_reason(), None);
    }

    #[test]
    fn sherpa_artifact_labels_are_stable_for_replay_reports() {
        assert_eq!(
            SherpaAsrArtifact::ZipformerEn20230626.label(),
            "sherpa-zipformer-en-2023-06-26"
        );
        assert_eq!(
            SherpaAsrArtifact::ZipformerEnKroko20250806.label(),
            "sherpa-zipformer-en-kroko-2025-08-06"
        );
    }

    #[cfg(feature = "sherpa")]
    #[test]
    fn sherpa_policy_suppresses_repeated_token_hallucinations() {
        let event = apply_sherpa_transcript_policy(final_event("MEQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"));

        assert!(event.is_suppressed());
        assert!(event.requires_session_reset());
        assert_eq!(
            event.suppression_reason(),
            Some(AsrTranscriptSuppressionReason::RepeatedTokenHallucination)
        );
    }

    #[cfg(feature = "sherpa")]
    #[test]
    fn sherpa_policy_passes_normal_transcripts_through() {
        let event = apply_sherpa_transcript_policy(final_event("meet me at the front desk"));

        assert!(!event.is_suppressed());
        assert!(!event.requires_session_reset());
    }
}
