#[cfg(feature = "sherpa")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "sherpa")]
use anyhow::{bail, Context};
use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, Mono};
#[cfg(feature = "sherpa")]
use motlie_model::typed::{StreamingTranscriber, TranscriptionSession};
#[cfg(feature = "sherpa")]
use motlie_model::{ArtifactPolicy, ModelError, StartOptions, TranscriptionParams};
use motlie_voice::app::TranscriptEvent;
#[cfg(feature = "sherpa")]
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
}

// justification: media WebSocket tasks share one process-selected ASR factory without coupling Telnyx wiring to a concrete model backend.
pub type SharedAsrFactory = Arc<dyn InboundAsrFactory>;

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
    handle: Mutex<Option<Arc<motlie_model_sherpa_onnx::SherpaOnnxHandle>>>,
}

#[cfg(feature = "sherpa")]
impl SherpaAsrFactory {
    pub fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        Self {
            artifact_root,
            allow_download,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_sherpa_onnx::SherpaOnnxHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_sherpa(&self.artifact_root, self.allow_download).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "sherpa")]
#[async_trait]
impl InboundAsrFactory for SherpaAsrFactory {
    async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
        let handle = self.handle().await?;
        let session = handle
            .open_session(TranscriptionParams {
                language: Some("en".to_string()),
                emit_partials: true,
            })
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

#[cfg(feature = "sherpa")]
async fn start_sherpa(
    artifact_root: &Path,
    allow_download: bool,
) -> anyhow::Result<motlie_model_sherpa_onnx::SherpaOnnxHandle> {
    use motlie_models::asr::sherpa_onnx_streaming_en;
    use motlie_models::{AsrModels, Catalog, LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX};

    let options = local_only_options(artifact_root);
    match sherpa_onnx_streaming_en::start_typed(options).await {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                "downloading Sherpa ONNX artifacts"
            );
            let catalog = Catalog::with_defaults();
            motlie_models::download_bundle_artifacts(
                &catalog,
                &AsrModels::SherpaOnnxStreamingEn.bundle_id(),
                artifact_root,
            )
            .map_err(anyhow::Error::from)
            .context("download Sherpa ONNX artifacts")?;
            sherpa_onnx_streaming_en::start_typed(local_only_options(artifact_root))
                .await
                .map_err(anyhow::Error::from)
                .context("start Sherpa after downloading artifacts")
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail!(
                "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} missing under '{}'; rerun without --no-asr-download or preinstall artifacts",
                artifact_root.display()
            )
        }
        Err(err) => Err(anyhow::Error::from(err)).context("start Sherpa ONNX"),
    }
}

#[cfg(feature = "sherpa")]
fn local_only_options(artifact_root: &Path) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root.to_path_buf(),
        }),
        ..Default::default()
    }
}

#[cfg(feature = "sherpa")]
fn missing_local_artifacts(error: &ModelError) -> bool {
    match error {
        ModelError::InvalidConfiguration(message) => {
            message.contains(motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX)
        }
        _ => false,
    }
}

#[cfg(feature = "sherpa")]
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
