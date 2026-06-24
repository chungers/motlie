use std::collections::VecDeque;
use std::sync::Arc;
use std::{fmt, str::FromStr};

use anyhow::bail;
#[cfg(any(feature = "kokoro", feature = "piper"))]
use anyhow::Context;
use async_trait::async_trait;
use clap::ValueEnum;
use motlie_model::typed::IncrementalSpeechCancelToken;
#[cfg(any(feature = "kokoro", feature = "piper"))]
use motlie_model::typed::SynthesisRequest;
#[cfg(feature = "kokoro")]
use motlie_model::typed::{
    BufferedSpeechSynthesizer, IncrementalSpeechStream, IncrementalSpeechSynthesizer,
};
#[cfg(feature = "kokoro")]
use motlie_model::typed::{IncrementalSpeechControls, IncrementalSpeechRequestLabel};
#[cfg(feature = "piper")]
use motlie_model::typed::{SpeechStream, SpeechSynthesizer};
#[cfg(any(feature = "kokoro", feature = "piper"))]
use motlie_model::{ArtifactPolicy, StartOptions};
#[cfg(any(feature = "kokoro", feature = "piper"))]
use std::path::{Path, PathBuf};
#[cfg(any(feature = "kokoro", feature = "piper"))]
use tokio::sync::Mutex;
#[cfg(any(feature = "kokoro", feature = "piper"))]
use tokio::task;

pub const KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;
pub const PIPER_SAMPLE_RATE_HZ: u32 = 22_050;
#[cfg(any(feature = "kokoro", feature = "piper"))]
const TTS_WARMUP_TEXT: &str = "Ready.";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TtsAudio {
    samples_i16: Vec<i16>,
    sample_rate_hz: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IncrementalTtsSummary {
    pub chunks: u64,
    pub audio_ms: u64,
    pub canceled: bool,
    pub synthesis_completed: bool,
}

#[async_trait]
pub trait OutboundIncrementalTtsStream: Send {
    async fn next_audio_chunk(&mut self) -> anyhow::Result<Option<TtsAudio>>;

    async fn finish(self: Box<Self>) -> anyhow::Result<IncrementalTtsSummary>;
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

    async fn synthesize_incremental(
        &self,
        _text: String,
        _cancel: IncrementalSpeechCancelToken,
        _request_label: Option<String>,
        _max_buffered_audio_ms: u32,
    ) -> anyhow::Result<Box<dyn OutboundIncrementalTtsStream>> {
        bail!(
            "TTS backend {} does not support streaming generation",
            self.label()
        )
    }

    async fn warm(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn warm_streaming(&self) -> anyhow::Result<()> {
        let cancel = IncrementalSpeechCancelToken::new();
        let mut stream = self
            .synthesize_incremental("Ready.".to_string(), cancel, Some("warm".to_string()), 80)
            .await?;
        while stream.next_audio_chunk().await?.is_some() {}
        let _ = stream.finish().await?;
        Ok(())
    }

    fn supports_incremental(&self) -> bool {
        false
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

    pub async fn warm_streaming(&self, backend: LiveTtsBackend) -> anyhow::Result<()> {
        self.factory(backend).warm_streaming().await
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
    handle: Mutex<Option<Arc<motlie_model_kokoro::KokoroHandle>>>,
}

#[cfg(feature = "kokoro")]
impl KokoroTtsFactory {
    pub fn new(artifact_root: PathBuf) -> Self {
        Self {
            artifact_root,
            handle: Mutex::new(None),
        }
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_kokoro::KokoroHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_kokoro(&self.artifact_root).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "kokoro")]
#[async_trait]
impl OutboundTtsFactory for KokoroTtsFactory {
    async fn warm(&self) -> anyhow::Result<()> {
        self.synthesize_chunks(TTS_WARMUP_TEXT.to_string())
            .await
            .map(|_| ())
    }

    async fn synthesize_chunks(&self, text: String) -> anyhow::Result<Vec<TtsAudio>> {
        let handle = self.handle().await?;
        let runtime = tokio::runtime::Handle::current();
        let audio = task::spawn_blocking(move || {
            runtime.block_on(handle.synthesize_buffered(SynthesisRequest {
                text,
                params: Default::default(),
            }))
        })
        .await
        .context("join Kokoro-82M synthesis task")?
        .context("synthesize Kokoro-82M speech")?;
        Ok(vec![TtsAudio::new(
            audio.into_samples(),
            KOKORO_SAMPLE_RATE_HZ,
        )?])
    }

    async fn synthesize_incremental(
        &self,
        text: String,
        cancel: IncrementalSpeechCancelToken,
        request_label: Option<String>,
        max_buffered_audio_ms: u32,
    ) -> anyhow::Result<Box<dyn OutboundIncrementalTtsStream>> {
        let handle = self.handle().await?;
        let controls = IncrementalSpeechControls {
            cancel,
            request_label: request_label.map(IncrementalSpeechRequestLabel::new),
            max_buffered_audio_ms,
        };
        let stream = handle
            .synthesize_incremental(
                SynthesisRequest {
                    text,
                    params: Default::default(),
                },
                controls,
            )
            .await
            .context("open Kokoro-82M incremental speech stream")?;
        Ok(Box::new(KokoroOutboundIncrementalTtsStream { stream }))
    }

    fn supports_incremental(&self) -> bool {
        true
    }

    fn label(&self) -> &'static str {
        "kokoro/kokoro_82m"
    }
}

#[cfg(feature = "kokoro")]
struct KokoroOutboundIncrementalTtsStream {
    stream: motlie_model_kokoro::KokoroMeteredIncrementalSpeechStream,
}

#[cfg(feature = "kokoro")]
#[async_trait]
impl OutboundIncrementalTtsStream for KokoroOutboundIncrementalTtsStream {
    async fn next_audio_chunk(&mut self) -> anyhow::Result<Option<TtsAudio>> {
        let Some(chunk) = self
            .stream
            .next_audio_chunk()
            .await
            .context("read Kokoro-82M incremental speech chunk")?
        else {
            return Ok(None);
        };
        if chunk.channels != 1 {
            bail!(
                "Kokoro-82M incremental speech emitted {} channels; expected mono",
                chunk.channels
            );
        }
        Ok(Some(TtsAudio::new(
            chunk.samples_i16,
            chunk.sample_rate_hz,
        )?))
    }

    async fn finish(self: Box<Self>) -> anyhow::Result<IncrementalTtsSummary> {
        let summary = self
            .stream
            .finish()
            .await
            .context("finish Kokoro-82M incremental speech stream")?;
        Ok(IncrementalTtsSummary {
            chunks: summary.chunks,
            audio_ms: summary.audio_ms,
            canceled: summary.canceled,
            synthesis_completed: summary.synthesis_completed,
        })
    }
}

#[cfg(feature = "piper")]
pub struct PiperTtsFactory {
    artifact_root: PathBuf,
    handle: Mutex<Option<Arc<motlie_model_piper::PiperHandle>>>,
}

#[cfg(feature = "piper")]
impl PiperTtsFactory {
    pub fn new(artifact_root: PathBuf) -> Self {
        Self {
            artifact_root,
            handle: Mutex::new(None),
        }
    }

    pub(crate) async fn warm(&self) -> anyhow::Result<()> {
        self.synthesize_chunks(TTS_WARMUP_TEXT.to_string())
            .await
            .map(|_| ())
    }

    async fn handle(&self) -> anyhow::Result<Arc<motlie_model_piper::PiperHandle>> {
        let mut guard = self.handle.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(start_piper(&self.artifact_root).await?);
        *guard = Some(Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(feature = "piper")]
#[async_trait]
impl OutboundTtsFactory for PiperTtsFactory {
    async fn warm(&self) -> anyhow::Result<()> {
        self.synthesize_chunks(TTS_WARMUP_TEXT.to_string())
            .await
            .map(|_| ())
    }

    async fn synthesize_chunks(&self, text: String) -> anyhow::Result<Vec<TtsAudio>> {
        let handle = self.handle().await?;
        let runtime = tokio::runtime::Handle::current();
        task::spawn_blocking(move || -> anyhow::Result<Vec<TtsAudio>> {
            let mut stream = runtime
                .block_on(handle.synthesize(SynthesisRequest {
                    text,
                    params: Default::default(),
                }))
                .context("open Piper speech stream")?;
            let mut chunks = Vec::new();
            while let Some(chunk) = runtime
                .block_on(stream.next_chunk())
                .context("read Piper speech chunk")?
            {
                chunks.push(TtsAudio::new(chunk.into_samples(), PIPER_SAMPLE_RATE_HZ)?);
            }
            runtime
                .block_on(stream.finish())
                .context("finish Piper speech stream")?;
            Ok(chunks)
        })
        .await
        .context("join Piper synthesis task")?
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
    let mut segments = VecDeque::from(speech_segments(text));
    let mut chunks = Vec::new();

    if first_chunk_max_chars > 0 {
        push_first_speech_chunk(&mut chunks, &mut segments, first_chunk_max_chars);
    }

    let mut pending_chunk = String::new();
    for segment in segments {
        push_speech_segment(&mut chunks, &mut pending_chunk, &segment, max_chars);
    }
    flush_speech_chunk(&mut chunks, &mut pending_chunk);
    chunks
}

#[derive(Debug, Clone)]
pub struct StreamingSpeechTextPacker {
    chunking_enabled: bool,
    max_chars: usize,
    first_chunk_max_chars: usize,
    pending_text: String,
    emitted_first_chunk: bool,
}

impl StreamingSpeechTextPacker {
    pub fn new(chunking_enabled: bool, max_chars: usize, first_chunk_max_chars: usize) -> Self {
        Self {
            chunking_enabled,
            max_chars: max_chars.max(1),
            first_chunk_max_chars,
            pending_text: String::new(),
            emitted_first_chunk: false,
        }
    }

    pub fn push_fragment(&mut self, fragment: &str, final_fragment: bool) -> Vec<String> {
        self.pending_text.push_str(fragment);
        if !self.chunking_enabled {
            if final_fragment {
                return self.flush_all_pending();
            }
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let segments = if final_fragment {
            self.flush_all_pending_segments()
        } else {
            self.take_complete_segments()
        };
        self.push_streaming_segments(&mut chunks, &segments);
        chunks
    }

    fn push_streaming_segments(&mut self, chunks: &mut Vec<String>, segments: &[String]) {
        if segments.is_empty() {
            return;
        }
        let mut pending_segments = VecDeque::from(segments.to_vec());
        if !self.emitted_first_chunk
            && self.first_chunk_max_chars > 0
            && push_first_speech_chunk(chunks, &mut pending_segments, self.first_chunk_max_chars)
        {
            self.emitted_first_chunk = true;
        }
        let before = chunks.len();
        let mut pending = String::new();
        for segment in pending_segments {
            push_speech_segment(chunks, &mut pending, &segment, self.max_chars);
        }
        flush_speech_chunk(chunks, &mut pending);
        if chunks.len() > before {
            self.emitted_first_chunk = true;
        }
    }

    fn take_complete_segments(&mut self) -> Vec<String> {
        let Some(boundary_end) = last_speech_boundary_end(&self.pending_text) else {
            return Vec::new();
        };
        let complete = self.pending_text[..boundary_end].to_string();
        let remainder = self.pending_text[boundary_end..].to_string();
        self.pending_text = remainder;
        speech_segments(&complete)
    }

    fn flush_all_pending_segments(&mut self) -> Vec<String> {
        let text = std::mem::take(&mut self.pending_text);
        speech_segments(&text)
    }

    fn flush_all_pending(&mut self) -> Vec<String> {
        let text = std::mem::take(&mut self.pending_text);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed.to_string()]
        }
    }
}

fn last_speech_boundary_end(text: &str) -> Option<usize> {
    text.char_indices()
        .rev()
        .find_map(|(index, ch)| is_speech_segment_boundary(ch).then_some(index + ch.len_utf8()))
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
    segments: &mut VecDeque<String>,
    first_chunk_max_chars: usize,
) -> bool {
    let first_chunk_max_chars = first_chunk_max_chars.max(1);
    let mut first_chunk = String::new();
    while let Some(segment) = segments.front().cloned() {
        let separator = usize::from(!first_chunk.is_empty());
        let next_chars = first_chunk
            .chars()
            .count()
            .saturating_add(separator)
            .saturating_add(segment.chars().count());
        if !first_chunk.is_empty() && next_chars > first_chunk_max_chars {
            break;
        }
        segments.pop_front();
        if first_chunk.is_empty() && segment.chars().count() > first_chunk_max_chars {
            let (prefix, remainder) =
                split_speech_prefix_at_word_boundary(&segment, first_chunk_max_chars);
            if !remainder.is_empty() {
                segments.push_front(remainder);
            }
            first_chunk.push_str(&prefix);
            break;
        }
        if !first_chunk.is_empty() {
            first_chunk.push(' ');
        }
        first_chunk.push_str(&segment);
        if first_chunk.chars().count() >= first_chunk_max_chars {
            break;
        }
    }
    if first_chunk.is_empty() {
        return false;
    }
    chunks.push(first_chunk);
    true
}

fn split_speech_prefix_at_word_boundary(text: &str, max_chars: usize) -> (String, String) {
    let max_chars = max_chars.max(1);
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return (String::new(), String::new());
    }

    let mut prefix_words = Vec::new();
    let mut prefix_chars = 0usize;
    for (index, word) in words.iter().enumerate() {
        let word_chars = word.chars().count();
        if prefix_words.is_empty() && word_chars > max_chars {
            let prefix = word.chars().take(max_chars).collect::<String>();
            let suffix = word.chars().skip(max_chars).collect::<String>();
            let mut remainder_words = Vec::new();
            if !suffix.is_empty() {
                remainder_words.push(suffix);
            }
            remainder_words.extend(words.iter().skip(index + 1).map(|word| (*word).to_string()));
            return (prefix, remainder_words.join(" "));
        }

        let separator = usize::from(!prefix_words.is_empty());
        let next_chars = prefix_chars
            .saturating_add(separator)
            .saturating_add(word_chars);
        if !prefix_words.is_empty() && next_chars > max_chars {
            return (
                prefix_words.join(" "),
                words
                    .iter()
                    .skip(index)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        prefix_words.push(*word);
        prefix_chars = next_chars;
    }

    (prefix_words.join(" "), String::new())
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
async fn start_kokoro(artifact_root: &Path) -> anyhow::Result<motlie_model_kokoro::KokoroHandle> {
    motlie_models::tts::kokoro_82m::start_typed(local_only_options(artifact_root))
        .await
        .map_err(anyhow::Error::from)
        .context("start Kokoro-82M TTS")
}

#[cfg(feature = "piper")]
async fn start_piper(artifact_root: &Path) -> anyhow::Result<motlie_model_piper::PiperHandle> {
    motlie_models::tts::piper_en_us_ljspeech_medium::start_typed(local_only_options(artifact_root))
        .await
        .map_err(anyhow::Error::from)
        .context("start Piper TTS")
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
            chunks: vec![TtsAudio::new(samples, sample_rate_hz)
                .expect("test TTS sample rate should be non-zero")],
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
    fn streaming_speech_packer_holds_incomplete_fragment_until_sentence_boundary() {
        let mut packer = StreamingSpeechTextPacker::new(true, 90, 45);
        assert!(packer.push_fragment("Hello wor", false).is_empty());
        assert_eq!(
            packer.push_fragment("ld. Next", false),
            vec!["Hello world."]
        );
        assert!(packer.push_fragment(" bit", false).is_empty());
        assert_eq!(packer.push_fragment(" done", true), vec!["Next bit done"]);
    }

    #[test]
    fn streaming_speech_packer_flushes_final_unsentenced_text() {
        let mut packer = StreamingSpeechTextPacker::new(true, 90, 0);
        assert!(packer.push_fragment("No boundary yet", false).is_empty());
        assert_eq!(packer.push_fragment("", true), vec!["No boundary yet"]);
    }

    #[test]
    fn streaming_speech_packer_respects_chunking_disabled_until_final() {
        let mut packer = StreamingSpeechTextPacker::new(false, 10, 0);
        assert!(packer.push_fragment("First. ", false).is_empty());
        assert_eq!(
            packer.push_fragment("Second.", true),
            vec!["First. Second."]
        );
    }

    #[test]
    fn streaming_speech_packer_packs_short_sentences_from_one_fragment() {
        let mut packer = StreamingSpeechTextPacker::new(true, 90, 0);
        assert_eq!(
            packer.push_fragment("Hi. Yes. OK. Sure.", false),
            vec!["Hi. Yes. OK. Sure."]
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
    fn split_speech_text_first_chunk_ramp_honors_cap_before_sentence_boundary() {
        assert_eq!(
            split_speech_text_with_first_chunk_max_chars(
                "A complete first sentence. The second sentence follows quickly.",
                90,
                12,
            ),
            vec![
                "A complete",
                "first sentence. The second sentence follows quickly."
            ]
        );
    }

    #[test]
    fn split_speech_text_first_chunk_splits_long_unsentenced_text() {
        assert_eq!(
            split_speech_text_with_first_chunk_max_chars(
                "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
                30,
                18,
            ),
            vec![
                "alpha beta gamma",
                "delta epsilon zeta eta theta",
                "iota kappa lambda mu",
            ]
        );
    }

    #[test]
    fn streaming_speech_packer_first_chunk_splits_long_final_fragment() {
        let mut packer = StreamingSpeechTextPacker::new(true, 30, 18);
        assert_eq!(
            packer.push_fragment(
                "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
                true,
            ),
            vec![
                "alpha beta gamma",
                "delta epsilon zeta eta theta",
                "iota kappa lambda mu",
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
