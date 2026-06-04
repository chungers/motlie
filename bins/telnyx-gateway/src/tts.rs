use std::sync::Arc;

use anyhow::bail;
use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, Mono};

#[cfg(feature = "piper")]
use anyhow::Context;
#[cfg(feature = "piper")]
use motlie_model::typed::{SpeechStream, SpeechSynthesizer, SynthesisRequest};
#[cfg(feature = "piper")]
use motlie_model::{ArtifactPolicy, ModelError, StartOptions};
#[cfg(feature = "piper")]
use std::path::{Path, PathBuf};
#[cfg(feature = "piper")]
use tokio::sync::Mutex;

pub const PIPER_SAMPLE_RATE_HZ: u32 = 22_050;

#[async_trait]
pub trait OutboundTtsFactory: Send + Sync {
    async fn synthesize_chunks(
        &self,
        text: String,
    ) -> anyhow::Result<Vec<AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>>>;

    fn label(&self) -> &'static str;
}

// justification: operator commands and media tests need one shared TTS handle without coupling the gateway command layer to a concrete compiled backend.
pub type SharedTtsFactory = Arc<dyn OutboundTtsFactory>;

#[derive(Clone)]
pub struct TtsRegistry {
    piper: SharedTtsFactory,
}

pub type SharedTtsRegistry = Arc<TtsRegistry>;

impl TtsRegistry {
    pub fn new(piper: SharedTtsFactory) -> Self {
        Self { piper }
    }

    pub fn piper(&self) -> SharedTtsFactory {
        self.piper.clone()
    }
}

pub struct UnavailableTtsFactory {
    message: &'static str,
}

impl UnavailableTtsFactory {
    pub fn new(message: &'static str) -> Self {
        Self { message }
    }
}

#[async_trait]
impl OutboundTtsFactory for UnavailableTtsFactory {
    async fn synthesize_chunks(
        &self,
        _text: String,
    ) -> anyhow::Result<Vec<AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>>> {
        bail!(self.message)
    }

    fn label(&self) -> &'static str {
        "unavailable"
    }
}

#[cfg(feature = "piper")]
pub struct PiperTtsFactory {
    artifact_root: PathBuf,
    allow_download: bool,
    handle: Mutex<Option<Arc<motlie_model_piper::PiperHandle>>>,
}

#[cfg(feature = "piper")]
impl PiperTtsFactory {
    pub fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        Self {
            artifact_root,
            allow_download,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_piper::PiperHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_piper(&self.artifact_root, self.allow_download).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "piper")]
#[async_trait]
impl OutboundTtsFactory for PiperTtsFactory {
    async fn synthesize_chunks(
        &self,
        text: String,
    ) -> anyhow::Result<Vec<AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>>> {
        let handle = self.handle().await?;
        let mut stream = handle
            .synthesize(SynthesisRequest {
                text,
                params: Default::default(),
            })
            .await
            .context("open Piper speech stream")?;
        let mut chunks = Vec::new();
        while let Some(chunk) = stream
            .next_chunk()
            .await
            .context("read Piper speech chunk")?
        {
            chunks.push(chunk);
        }
        stream
            .finish()
            .await
            .context("finish Piper speech stream")?;
        Ok(chunks)
    }

    fn label(&self) -> &'static str {
        "piper/en_us_ljspeech_medium"
    }
}

pub fn split_speech_text(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | ',' | ';' | ':' | '\n') {
            push_speech_chunk(&mut chunks, &mut current);
        }
    }
    push_speech_chunk(&mut chunks, &mut current);
    chunks
}

fn push_speech_chunk(chunks: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        chunks.push(trimmed.to_string());
    }
    current.clear();
}

pub fn unavailable_registry() -> SharedTtsRegistry {
    Arc::new(TtsRegistry::new(Arc::new(UnavailableTtsFactory::new(
        "Piper TTS is unavailable; rebuild with --features piper",
    ))))
}

#[cfg(feature = "piper")]
async fn start_piper(
    artifact_root: &Path,
    allow_download: bool,
) -> anyhow::Result<motlie_model_piper::PiperHandle> {
    match motlie_models::tts::piper_en_us_ljspeech_medium::start_typed(local_only_options(
        artifact_root,
    ))
    .await
    {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                artifact = "piper/en_us_ljspeech_medium",
                "downloading Piper artifacts"
            );
            download_piper_artifact(artifact_root)?;
            motlie_models::tts::piper_en_us_ljspeech_medium::start_typed(local_only_options(
                artifact_root,
            ))
            .await
            .map_err(anyhow::Error::from)
            .context("start Piper after downloading artifacts")
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail_missing_artifacts("piper/en_us_ljspeech_medium", artifact_root)
        }
        Err(err) => Err(anyhow::Error::from(err)).context("start Piper TTS"),
    }
}

#[cfg(feature = "piper")]
fn download_piper_artifact(artifact_root: &Path) -> anyhow::Result<()> {
    let catalog = motlie_models::Catalog::with_defaults();
    let model = motlie_models::TtsModels::PiperEnUsLjspeechMedium;
    motlie_models::download_bundle_artifacts(&catalog, &model.bundle_id(), artifact_root)
        .map(|_| ())
        .map_err(anyhow::Error::from)
        .context("download Piper artifacts")
}

#[cfg(feature = "piper")]
fn local_only_options(artifact_root: &Path) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root.to_path_buf(),
        }),
        ..Default::default()
    }
}

#[cfg(feature = "piper")]
fn missing_local_artifacts(error: &ModelError) -> bool {
    match error {
        ModelError::InvalidConfiguration(message) => {
            message.contains(motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX)
        }
        _ => false,
    }
}

#[cfg(feature = "piper")]
fn bail_missing_artifacts<T>(label: &str, artifact_root: &Path) -> anyhow::Result<T> {
    bail!(
        "{} missing for {} under '{}'; rerun without --no-asr-download or preinstall artifacts",
        motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX,
        label,
        artifact_root.display()
    )
}

#[cfg(test)]
pub struct StaticTtsFactory {
    chunks: Vec<AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>>,
}

#[cfg(test)]
impl StaticTtsFactory {
    pub fn new(samples: Vec<i16>) -> Self {
        Self {
            chunks: vec![AudioBuf::new(samples)],
        }
    }
}

#[cfg(test)]
#[async_trait]
impl OutboundTtsFactory for StaticTtsFactory {
    async fn synthesize_chunks(
        &self,
        _text: String,
    ) -> anyhow::Result<Vec<AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>>> {
        Ok(self.chunks.clone())
    }

    fn label(&self) -> &'static str {
        "static-test-tts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_speech_text_uses_sentence_boundaries() {
        assert_eq!(
            split_speech_text("Hello there. This is Motlie!"),
            vec!["Hello there.", "This is Motlie!"]
        );
    }

    #[test]
    fn split_speech_text_keeps_unsentenced_text() {
        assert_eq!(
            split_speech_text("Hello from Motlie"),
            vec!["Hello from Motlie"]
        );
    }

    #[test]
    fn split_speech_text_uses_clause_boundaries() {
        assert_eq!(
            split_speech_text("Hello, then continue: now stop;"),
            vec!["Hello,", "then continue:", "now stop;"]
        );
    }
}
