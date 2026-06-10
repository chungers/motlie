use std::sync::Arc;
use std::{fmt, str::FromStr};

#[cfg(any(feature = "kokoro", feature = "piper"))]
use anyhow::Context;
use anyhow::bail;
use async_trait::async_trait;
use clap::ValueEnum;
#[cfg(feature = "kokoro")]
use motlie_model::typed::BufferedSpeechSynthesizer;
#[cfg(any(feature = "kokoro", feature = "piper"))]
use motlie_model::typed::SynthesisRequest;
#[cfg(feature = "piper")]
use motlie_model::typed::{SpeechStream, SpeechSynthesizer};
#[cfg(any(feature = "kokoro", feature = "piper"))]
use motlie_model::{ArtifactPolicy, ModelError, StartOptions};
#[cfg(any(feature = "kokoro", feature = "piper"))]
use std::path::{Path, PathBuf};
#[cfg(any(feature = "kokoro", feature = "piper"))]
use tokio::sync::Mutex;

pub const KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;
pub const PIPER_SAMPLE_RATE_HZ: u32 = 22_050;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TtsAudio {
    samples_i16: Vec<i16>,
    sample_rate_hz: u32,
}

impl TtsAudio {
    pub fn new(samples_i16: Vec<i16>, sample_rate_hz: u32) -> anyhow::Result<Self> {
        if sample_rate_hz == 0 {
            bail!("TTS sample rate must be non-zero");
        }
        Ok(Self {
            samples_i16,
            sample_rate_hz,
        })
    }

    pub fn samples_i16(&self) -> &[i16] {
        &self.samples_i16
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn into_samples_i16(self) -> Vec<i16> {
        self.samples_i16
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum LiveTtsBackend {
    #[default]
    #[value(name = "kokoro-82m", alias = "kokoro", alias = "kokoro/kokoro_82m")]
    Kokoro82m,
    #[value(
        name = "piper",
        alias = "piper-en-us-ljspeech-medium",
        alias = "piper/en_us_ljspeech_medium"
    )]
    Piper,
}

impl LiveTtsBackend {
    pub const fn available() -> [Self; 2] {
        [Self::Kokoro82m, Self::Piper]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Kokoro82m => "kokoro-82m",
            Self::Piper => "piper",
        }
    }

    pub fn model_label(self) -> &'static str {
        match self {
            Self::Kokoro82m => "kokoro/kokoro_82m",
            Self::Piper => "piper/en_us_ljspeech_medium",
        }
    }

    pub fn fallback(self) -> Option<Self> {
        match self {
            Self::Kokoro82m => Some(Self::Piper),
            Self::Piper => None,
        }
    }
}

impl fmt::Display for LiveTtsBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("unsupported live TTS backend `{value}`; expected kokoro-82m or piper")]
pub struct LiveTtsBackendParseError {
    value: String,
}

impl FromStr for LiveTtsBackend {
    type Err = LiveTtsBackendParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "kokoro-82m" | "kokoro" | "kokoro/kokoro_82m" => Ok(Self::Kokoro82m),
            "piper" | "piper-en-us-ljspeech-medium" | "piper/en_us_ljspeech_medium" => {
                Ok(Self::Piper)
            }
            _ => Err(LiveTtsBackendParseError {
                value: value.to_string(),
            }),
        }
    }
}

#[async_trait]
pub trait OutboundTtsFactory: Send + Sync {
    async fn synthesize_chunks(&self, text: String) -> anyhow::Result<Vec<TtsAudio>>;

    async fn warm(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn label(&self) -> &'static str;

    fn is_available(&self) -> bool {
        true
    }

    fn unavailable_reason(&self) -> Option<&'static str> {
        None
    }
}

// justification: operator commands and media tests need one shared TTS handle without coupling the gateway command layer to a concrete compiled backend.
pub type SharedTtsFactory = Arc<dyn OutboundTtsFactory>;

#[derive(Clone)]
pub struct TtsRegistry {
    kokoro: SharedTtsFactory,
    piper: SharedTtsFactory,
}

pub type SharedTtsRegistry = Arc<TtsRegistry>;

impl TtsRegistry {
    pub fn new(kokoro: SharedTtsFactory, piper: SharedTtsFactory) -> Self {
        Self { kokoro, piper }
    }

    pub fn kokoro(&self) -> SharedTtsFactory {
        self.kokoro.clone()
    }

    pub fn piper(&self) -> SharedTtsFactory {
        self.piper.clone()
    }

    pub fn factory(&self, backend: LiveTtsBackend) -> SharedTtsFactory {
        match backend {
            LiveTtsBackend::Kokoro82m => self.kokoro(),
            LiveTtsBackend::Piper => self.piper(),
        }
    }

    pub async fn warm(&self, backend: LiveTtsBackend) -> anyhow::Result<()> {
        self.factory(backend).warm().await
    }
}

pub struct UnavailableTtsFactory {
    label: &'static str,
    message: &'static str,
}

impl UnavailableTtsFactory {
    pub fn new(label: &'static str, message: &'static str) -> Self {
        Self { label, message }
    }
}

#[async_trait]
impl OutboundTtsFactory for UnavailableTtsFactory {
    async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
        bail!(self.message)
    }

    async fn warm(&self) -> anyhow::Result<()> {
        bail!(self.message)
    }

    fn label(&self) -> &'static str {
        self.label
    }

    fn is_available(&self) -> bool {
        false
    }

    fn unavailable_reason(&self) -> Option<&'static str> {
        Some(self.message)
    }
}

#[cfg(feature = "kokoro")]
pub struct KokoroTtsFactory {
    artifact_root: PathBuf,
    allow_download: bool,
    handle: Mutex<Option<Arc<motlie_model_kokoro::KokoroHandle>>>,
}

#[cfg(feature = "kokoro")]
impl KokoroTtsFactory {
    pub fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        Self {
            artifact_root,
            allow_download,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_kokoro::KokoroHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_kokoro(&self.artifact_root, self.allow_download).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "kokoro")]
#[async_trait]
impl OutboundTtsFactory for KokoroTtsFactory {
    async fn warm(&self) -> anyhow::Result<()> {
        self.handle().await.map(|_| ())
    }

    async fn synthesize_chunks(&self, text: String) -> anyhow::Result<Vec<TtsAudio>> {
        let handle = self.handle().await?;
        let audio = handle
            .synthesize_buffered(SynthesisRequest {
                text,
                params: Default::default(),
            })
            .await
            .context("synthesize Kokoro-82M speech")?;
        Ok(vec![TtsAudio::new(
            audio.into_samples(),
            KOKORO_SAMPLE_RATE_HZ,
        )?])
    }

    fn label(&self) -> &'static str {
        "kokoro/kokoro_82m"
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

    pub(crate) async fn warm(&self) -> anyhow::Result<()> {
        self.handle().await.map(|_| ())
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
    async fn warm(&self) -> anyhow::Result<()> {
        self.handle().await.map(|_| ())
    }

    async fn synthesize_chunks(&self, text: String) -> anyhow::Result<Vec<TtsAudio>> {
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
            chunks.push(TtsAudio::new(chunk.into_samples(), PIPER_SAMPLE_RATE_HZ)?);
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
    split_speech_text_with_max_chars(text, usize::MAX)
}

pub fn split_speech_text_with_max_chars(text: &str, max_chars: usize) -> Vec<String> {
    split_speech_text_with_first_chunk_max_chars(text, max_chars, 0)
}

pub fn split_speech_text_with_first_chunk_max_chars(
    text: &str,
    max_chars: usize,
    first_chunk_max_chars: usize,
) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let segments = speech_segments(text);
    let mut chunks = Vec::new();
    let mut segment_index = 0;

    if first_chunk_max_chars > 0 {
        segment_index = push_first_speech_chunk(&mut chunks, &segments, first_chunk_max_chars);
    }

    let mut pending_chunk = String::new();
    for segment in segments.iter().skip(segment_index) {
        push_speech_segment(&mut chunks, &mut pending_chunk, segment, max_chars);
    }
    flush_speech_chunk(&mut chunks, &mut pending_chunk);
    chunks
}

fn speech_segments(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut segment = String::new();
    for ch in text.chars() {
        segment.push(ch);
        if is_speech_segment_boundary(ch) {
            push_trimmed_segment(&mut segments, &mut segment);
        }
    }
    push_trimmed_segment(&mut segments, &mut segment);
    segments
}

fn push_trimmed_segment(segments: &mut Vec<String>, segment: &mut String) {
    let trimmed = segment.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }
    segment.clear();
}

fn push_first_speech_chunk(
    chunks: &mut Vec<String>,
    segments: &[String],
    first_chunk_max_chars: usize,
) -> usize {
    let mut first_chunk = String::new();
    let mut consumed = 0;
    for segment in segments {
        let separator = usize::from(!first_chunk.is_empty());
        let next_chars = first_chunk
            .chars()
            .count()
            .saturating_add(separator)
            .saturating_add(segment.chars().count());
        if !first_chunk.is_empty() && next_chars > first_chunk_max_chars {
            break;
        }
        if !first_chunk.is_empty() {
            first_chunk.push(' ');
        }
        first_chunk.push_str(segment);
        consumed += 1;
        if first_chunk.chars().count() >= first_chunk_max_chars {
            break;
        }
    }
    if !first_chunk.is_empty() {
        chunks.push(first_chunk);
    }
    consumed
}

fn is_speech_segment_boundary(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '\n')
}

fn push_speech_segment(
    chunks: &mut Vec<String>,
    pending_chunk: &mut String,
    segment: &str,
    max_chars: usize,
) {
    if segment.is_empty() {
        return;
    }

    if segment.chars().count() > max_chars {
        flush_speech_chunk(chunks, pending_chunk);
        push_bounded_speech_chunk(chunks, segment, max_chars);
        return;
    }

    let separator = usize::from(!pending_chunk.is_empty());
    let next_chars = pending_chunk
        .chars()
        .count()
        .saturating_add(separator)
        .saturating_add(segment.chars().count());
    if !pending_chunk.is_empty() && next_chars > max_chars {
        flush_speech_chunk(chunks, pending_chunk);
    }
    if !pending_chunk.is_empty() {
        pending_chunk.push(' ');
    }
    pending_chunk.push_str(segment);
}

fn flush_speech_chunk(chunks: &mut Vec<String>, pending_chunk: &mut String) {
    let trimmed = pending_chunk.trim();
    if !trimmed.is_empty() {
        chunks.push(trimmed.to_string());
    }
    pending_chunk.clear();
}

fn push_bounded_speech_chunk(chunks: &mut Vec<String>, text: &str, max_chars: usize) {
    if text.chars().count() <= max_chars {
        chunks.push(text.to_string());
        return;
    }

    let mut current = String::new();
    for word in text.split_whitespace() {
        let word_chars = word.chars().count();
        if word_chars > max_chars {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            push_long_word_chunks(chunks, word, max_chars);
            continue;
        }

        let separator = usize::from(!current.is_empty());
        let next_chars = current
            .chars()
            .count()
            .saturating_add(separator)
            .saturating_add(word_chars);
        if !current.is_empty() && next_chars > max_chars {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
}

fn push_long_word_chunks(chunks: &mut Vec<String>, word: &str, max_chars: usize) {
    let mut current = String::new();
    for ch in word.chars() {
        if current.chars().count() >= max_chars {
            chunks.push(std::mem::take(&mut current));
        }
        current.push(ch);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
}

pub fn unavailable_registry() -> SharedTtsRegistry {
    Arc::new(TtsRegistry::new(
        Arc::new(UnavailableTtsFactory::new(
            LiveTtsBackend::Kokoro82m.model_label(),
            "Kokoro-82M TTS is unavailable; rebuild with --features kokoro",
        )),
        Arc::new(UnavailableTtsFactory::new(
            LiveTtsBackend::Piper.model_label(),
            "Piper TTS is unavailable; rebuild with --features piper",
        )),
    ))
}

#[cfg(feature = "kokoro")]
async fn start_kokoro(
    artifact_root: &Path,
    allow_download: bool,
) -> anyhow::Result<motlie_model_kokoro::KokoroHandle> {
    match motlie_models::tts::kokoro_82m::start_typed(local_only_options(artifact_root)).await {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                artifact = "kokoro/kokoro_82m",
                "downloading Kokoro-82M artifacts"
            );
            download_kokoro_artifact(artifact_root)?;
            motlie_models::tts::kokoro_82m::start_typed(local_only_options(artifact_root))
                .await
                .map_err(anyhow::Error::from)
                .context("start Kokoro-82M after downloading artifacts")
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail_missing_artifacts("kokoro/kokoro_82m", artifact_root)
        }
        Err(err) => Err(anyhow::Error::from(err)).context("start Kokoro-82M TTS"),
    }
}

#[cfg(feature = "kokoro")]
fn download_kokoro_artifact(artifact_root: &Path) -> anyhow::Result<()> {
    let catalog = motlie_models::Catalog::with_defaults();
    let bundle_id = motlie_models::tts::kokoro_82m::descriptor().id;
    motlie_models::download_bundle_artifacts(&catalog, &bundle_id, artifact_root)
        .map(|_| ())
        .map_err(anyhow::Error::from)
        .context("download Kokoro-82M artifacts")
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

#[cfg(any(feature = "kokoro", feature = "piper"))]
fn local_only_options(artifact_root: &Path) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root.to_path_buf(),
        }),
        ..Default::default()
    }
}

#[cfg(any(feature = "kokoro", feature = "piper"))]
fn missing_local_artifacts(error: &ModelError) -> bool {
    match error {
        ModelError::InvalidConfiguration(message) => {
            message.contains(motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX)
        }
        _ => false,
    }
}

#[cfg(any(feature = "kokoro", feature = "piper"))]
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
    chunks: Vec<TtsAudio>,
}

#[cfg(test)]
impl StaticTtsFactory {
    pub fn new(samples: Vec<i16>) -> Self {
        Self::with_sample_rate(samples, PIPER_SAMPLE_RATE_HZ)
    }

    pub fn with_sample_rate(samples: Vec<i16>, sample_rate_hz: u32) -> Self {
        Self {
            chunks: vec![
                TtsAudio::new(samples, sample_rate_hz)
                    .expect("test TTS sample rate should be non-zero"),
            ],
        }
    }
}

#[cfg(test)]
#[async_trait]
impl OutboundTtsFactory for StaticTtsFactory {
    async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
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
    fn live_tts_default_is_kokoro_with_piper_fallback() {
        assert_eq!(LiveTtsBackend::default(), LiveTtsBackend::Kokoro82m);
        assert_eq!(
            LiveTtsBackend::Kokoro82m.fallback(),
            Some(LiveTtsBackend::Piper)
        );
        assert_eq!(LiveTtsBackend::Piper.fallback(), None);
        assert_eq!(
            LiveTtsBackend::available(),
            [LiveTtsBackend::Kokoro82m, LiveTtsBackend::Piper]
        );
    }

    #[test]
    fn live_tts_backend_parses_curated_aliases() {
        assert_eq!(
            "kokoro".parse::<LiveTtsBackend>().unwrap(),
            LiveTtsBackend::Kokoro82m
        );
        assert_eq!(
            "kokoro/kokoro_82m".parse::<LiveTtsBackend>().unwrap(),
            LiveTtsBackend::Kokoro82m
        );
        assert_eq!(
            "piper".parse::<LiveTtsBackend>().unwrap(),
            LiveTtsBackend::Piper
        );
    }

    #[test]
    fn split_speech_text_packs_complete_sentences() {
        assert_eq!(
            split_speech_text_with_max_chars("Hello there. This is Motlie!", 90),
            vec!["Hello there. This is Motlie!"]
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
    fn split_speech_text_does_not_split_clause_fragments() {
        assert_eq!(
            split_speech_text_with_max_chars("Hello, then continue: now stop;", 90),
            vec!["Hello, then continue: now stop;"]
        );
    }

    #[test]
    fn split_speech_text_keeps_short_colon_prelude_with_following_text() {
        assert_eq!(
            split_speech_text_with_max_chars("I heard: hello there. Done.", 90),
            vec!["I heard: hello there. Done."]
        );
    }

    #[test]
    fn split_speech_text_starts_smoke_echo_as_complete_sentence() {
        assert_eq!(
            split_speech_text_with_max_chars("I heard: Okay, hello. Can you hear me?", 90),
            vec!["I heard: Okay, hello. Can you hear me?"]
        );
    }

    #[test]
    fn split_speech_text_first_chunk_ramp_flushes_complete_sentence() {
        assert_eq!(
            split_speech_text_with_first_chunk_max_chars(
                "A complete first sentence. The second sentence follows quickly.",
                90,
                12,
            ),
            vec![
                "A complete first sentence.",
                "The second sentence follows quickly."
            ]
        );
    }

    #[test]
    fn split_speech_text_first_chunk_zero_uses_normal_packing() {
        assert_eq!(
            split_speech_text_with_first_chunk_max_chars("First sentence. Second sentence.", 90, 0,),
            vec!["First sentence. Second sentence."]
        );
    }

    #[test]
    fn split_speech_text_with_max_chars_breaks_long_clauses_on_words() {
        assert_eq!(
            split_speech_text_with_max_chars("alpha beta gamma delta", 12),
            vec!["alpha beta", "gamma delta"]
        );
    }

    #[test]
    fn tts_audio_requires_non_zero_sample_rate() {
        let err = TtsAudio::new(vec![0], 0).expect_err("zero sample rate should fail");

        assert!(err.to_string().contains("sample rate"));
    }
}
