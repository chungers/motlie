use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::early_response::{BoundaryRequirement, EarlyResponsePolicy, EarlyResponseStartTiming};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QualityProfile {
    Fast,
    #[default]
    Balanced,
    Complete,
    Noisy,
}

impl QualityProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Complete => "complete",
            Self::Noisy => "noisy",
        }
    }
}

impl FromStr for QualityProfile {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "fast" => Ok(Self::Fast),
            "balanced" => Ok(Self::Balanced),
            "complete" => Ok(Self::Complete),
            "noisy" => Ok(Self::Noisy),
            other => Err(format!("unknown quality profile {other}")),
        }
    }
}

impl fmt::Display for QualityProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RedactionMode {
    #[default]
    MetricsOnly,
    HashedText,
    RedactedText,
    SensitivePlaintext,
}

impl RedactionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MetricsOnly => "metrics-only",
            Self::HashedText => "hashed-text",
            Self::RedactedText => "redacted-text",
            Self::SensitivePlaintext => "sensitive-plaintext",
        }
    }
}

impl FromStr for RedactionMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "metrics-only" => Ok(Self::MetricsOnly),
            "hashed-text" => Ok(Self::HashedText),
            "redacted-text" => Ok(Self::RedactedText),
            "sensitive-plaintext" => Ok(Self::SensitivePlaintext),
            other => Err(format!("unknown redaction mode {other}")),
        }
    }
}

impl fmt::Display for RedactionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JudgeMode {
    #[default]
    Offline,
    LiveSample,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyBoundary {
    Immediate,
    NewCall,
    NextAsrSession,
    NewTextCallSession,
    NewTurn,
    NewPlaybackRequest,
    ReportOnly,
    NextJudgeJob,
    NextWriterStart,
}

impl ApplyBoundary {
    pub fn label(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::NewCall => "new_call",
            Self::NextAsrSession => "next_asr_session",
            Self::NewTextCallSession => "new_text_call_session",
            Self::NewTurn => "new_turn",
            Self::NewPlaybackRequest => "new_playback_request",
            Self::ReportOnly => "report_only",
            Self::NextJudgeJob => "next_judge_job",
            Self::NextWriterStart => "next_writer_start",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeechQualityConfig {
    pub rms_threshold: f32,
    pub peak_threshold: i32,
    pub onset_min_silence_ms: u64,
}

impl Default for SpeechQualityConfig {
    fn default() -> Self {
        Self {
            rms_threshold: 220.0,
            peak_threshold: 1_100,
            onset_min_silence_ms: 180,
        }
    }
}

fn default_final_settle_trailing_punctuation() -> Vec<String> {
    vec![",".to_string(), ":".to_string(), ";".to_string()]
}

fn default_final_settle_lead_words() -> Vec<String> {
    [
        "although", "because", "whereas", "while", "when", "if", "unless", "since", "before",
        "after", "though",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn default_final_settle_tail_words() -> Vec<String> {
    [
        "a", "an", "also", "the", "and", "or", "but", "if", "then", "than", "because", "so", "to",
        "of", "for", "with", "without", "in", "on", "at", "by", "from", "as", "is", "are", "was",
        "were", "be", "being", "been", "this", "that", "these", "those", "my", "your", "our",
        "their", "his", "her", "its",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn default_final_settle_dangling_suffixes() -> Vec<String> {
    vec!["'".to_string(), "-".to_string()]
}

fn default_conversation_tail_words() -> Vec<String> {
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
        "did", "do", "does", "for", "from", "had", "has", "have", "if", "in", "is", "may", "might",
        "must", "of", "on", "or", "should", "some", "that", "the", "this", "to", "was", "were",
        "where", "will", "with", "would",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn default_conversation_incomplete_tail_hold_ms() -> u64 {
    2_500
}

fn default_conversation_low_confidence_threshold_percent() -> u64 {
    45
}

fn default_conversation_playback_hold_poll_ms() -> u64 {
    100
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EndpointQualityConfig {
    pub trailing_silence_ms: u64,
    pub min_turn_words: usize,
    pub min_turn_chars: usize,
    pub merge_window_ms: u64,
    pub final_settle_ms: u64,
    #[serde(default = "default_final_settle_trailing_punctuation")]
    pub final_settle_trailing_punctuation: Vec<String>,
    #[serde(default = "default_final_settle_lead_words")]
    pub final_settle_lead_words: Vec<String>,
    #[serde(default = "default_final_settle_tail_words")]
    pub final_settle_tail_words: Vec<String>,
    #[serde(default = "default_final_settle_dangling_suffixes")]
    pub final_settle_dangling_suffixes: Vec<String>,
    #[serde(default = "default_conversation_tail_words")]
    pub conversation_tail_words: Vec<String>,
    #[serde(default = "default_conversation_incomplete_tail_hold_ms")]
    pub conversation_incomplete_tail_hold_ms: u64,
    #[serde(default = "default_conversation_low_confidence_threshold_percent")]
    pub conversation_low_confidence_threshold_percent: u64,
    #[serde(default = "default_conversation_playback_hold_poll_ms")]
    pub conversation_playback_hold_poll_ms: u64,
    pub max_turn_words: usize,
    pub max_turn_duration_ms: u64,
}

impl Default for EndpointQualityConfig {
    fn default() -> Self {
        Self {
            trailing_silence_ms: 900,
            min_turn_words: 2,
            min_turn_chars: 6,
            merge_window_ms: 350,
            final_settle_ms: 800,
            final_settle_trailing_punctuation: default_final_settle_trailing_punctuation(),
            final_settle_lead_words: default_final_settle_lead_words(),
            final_settle_tail_words: default_final_settle_tail_words(),
            final_settle_dangling_suffixes: default_final_settle_dangling_suffixes(),
            conversation_tail_words: default_conversation_tail_words(),
            conversation_incomplete_tail_hold_ms: default_conversation_incomplete_tail_hold_ms(),
            conversation_low_confidence_threshold_percent:
                default_conversation_low_confidence_threshold_percent(),
            conversation_playback_hold_poll_ms: default_conversation_playback_hold_poll_ms(),
            max_turn_words: 80,
            max_turn_duration_ms: 12_000,
        }
    }
}

impl EndpointQualityConfig {
    pub fn final_fragment_hold_reason(&self, text: &str) -> Option<&'static str> {
        let trimmed = text.trim();
        if trimmed.is_empty() || has_terminal_punctuation(trimmed) {
            return None;
        }
        if matches_config_suffix(&self.final_settle_trailing_punctuation, trimmed) {
            return Some("trailing_punctuation");
        }
        let first_word = trimmed
            .split_whitespace()
            .next()
            .map(normalize_fragment_word)
            .unwrap_or_default();
        if config_word_list_contains(&self.final_settle_lead_words, &first_word) {
            return Some("lead_word");
        }
        let last_word = trimmed
            .split_whitespace()
            .last()
            .map(normalize_fragment_word)
            .unwrap_or_default();
        if matches_config_suffix(&self.final_settle_dangling_suffixes, &last_word) {
            return Some("dangling_tail");
        }
        config_word_list_contains(&self.final_settle_tail_words, &last_word).then_some("tail_word")
    }

    pub fn conversation_incomplete_tail_reason(&self, text: &str) -> Option<&'static str> {
        let trimmed = text.trim();
        if trimmed.is_empty() || has_terminal_punctuation(trimmed) {
            return None;
        }
        if matches_config_suffix(&self.final_settle_dangling_suffixes, trimmed) {
            return Some("dangling_tail");
        }
        let tail = trimmed
            .split_whitespace()
            .last()
            .map(normalize_fragment_word)
            .unwrap_or_default();
        config_word_list_contains(&self.conversation_tail_words, &tail).then_some("tail_word")
    }

    pub fn conversation_low_confidence_threshold(&self) -> f32 {
        self.conversation_low_confidence_threshold_percent as f32 / 100.0
    }

    pub fn conversation_low_confidence_hold_allowed(&self, text: &str) -> bool {
        let trimmed = text.trim();
        !trimmed.is_empty() && !has_terminal_punctuation(trimmed)
    }
}

fn has_terminal_punctuation(text: &str) -> bool {
    text.chars()
        .rev()
        .find(|ch| !ch.is_whitespace())
        .is_some_and(|ch| matches!(ch, '.' | '?' | '!'))
}

fn normalize_fragment_word(word: &str) -> String {
    word.trim_matches(|ch: char| ch.is_ascii_punctuation() && ch != '\'' && ch != '-')
        .to_ascii_lowercase()
}

fn config_word_list_contains(words: &[String], word: &str) -> bool {
    words
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(word))
}

fn matches_config_suffix(suffixes: &[String], text: &str) -> bool {
    suffixes
        .iter()
        .any(|suffix| !suffix.is_empty() && text.ends_with(suffix))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AsrQualityConfig {
    pub repeated_token_run_threshold: usize,
    pub repeated_q_run_threshold: usize,
    pub finish_pad_ms: u64,
}

impl Default for AsrQualityConfig {
    fn default() -> Self {
        Self {
            repeated_token_run_threshold: 16,
            repeated_q_run_threshold: 8,
            finish_pad_ms: 320,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextCallQualityConfig {
    pub max_active_turns: usize,
    pub media_ready_timeout_ms: u64,
    pub playback_wait_timeout_ms: u64,
    pub latest_response_wins: bool,
    pub callback_timeout_ms: u64,
}

impl Default for TextCallQualityConfig {
    fn default() -> Self {
        Self {
            max_active_turns: 32,
            media_ready_timeout_ms: 20_000,
            playback_wait_timeout_ms: 180_000,
            latest_response_wins: true,
            callback_timeout_ms: 5_000,
        }
    }
}

impl TextCallQualityConfig {
    pub fn media_ready_timeout(&self) -> Duration {
        Duration::from_millis(self.media_ready_timeout_ms)
    }

    pub fn playback_wait_timeout(&self) -> Duration {
        Duration::from_millis(self.playback_wait_timeout_ms)
    }

    pub fn callback_timeout(&self) -> Duration {
        Duration::from_millis(self.callback_timeout_ms)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsGenerationMode {
    #[default]
    Buffered,
    Streaming,
}

impl TtsGenerationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Buffered => "buffered",
            Self::Streaming => "streaming",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TtsQualityConfig {
    #[serde(default)]
    pub generation_mode: TtsGenerationMode,
    pub chunking_enabled: bool,
    pub max_text_chunk_chars: usize,
    pub first_chunk_max_chars: usize,
    pub prebuffer_chunks: usize,
}

impl Default for TtsQualityConfig {
    fn default() -> Self {
        Self {
            generation_mode: TtsGenerationMode::Buffered,
            chunking_enabled: true,
            max_text_chunk_chars: 90,
            first_chunk_max_chars: 40,
            prebuffer_chunks: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnsetDuringPlaybackPolicy {
    #[default]
    DeferToPartial,
    Trust,
}

impl OnsetDuringPlaybackPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::DeferToPartial => "defer_to_partial",
            Self::Trust => "trust",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BargeInQualityConfig {
    pub enabled: bool,
    pub speech_onset_cancel_enabled: bool,
    #[serde(default)]
    pub onset_during_playback: OnsetDuringPlaybackPolicy,
    pub partial_asr_cancel_enabled: bool,
    pub final_asr_cancel_enabled: bool,
    pub clear_timeout_ms: u64,
}

impl Default for BargeInQualityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            speech_onset_cancel_enabled: true,
            onset_during_playback: OnsetDuringPlaybackPolicy::DeferToPartial,
            partial_asr_cancel_enabled: true,
            final_asr_cancel_enabled: true,
            clear_timeout_ms: 1_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EchoSuppressionQualityConfig {
    pub enabled: bool,
    pub min_text_chars: usize,
    pub tail_window_ms: u64,
    pub short_token_coverage_percent: u64,
    pub short_longest_token_run: usize,
    pub long_min_tokens: usize,
    pub long_token_coverage_percent: u64,
    pub long_longest_token_run: usize,
}

impl Default for EchoSuppressionQualityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_text_chars: 10,
            tail_window_ms: 2_000,
            short_token_coverage_percent: 66,
            short_longest_token_run: 2,
            long_min_tokens: 4,
            long_token_coverage_percent: 60,
            long_longest_token_run: 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoggingQualityConfig {
    pub enabled: bool,
    pub queue_capacity: usize,
    pub per_frame_sample_rate: f32,
    pub include_transcript_text: bool,
    pub redaction_mode: RedactionMode,
}

impl Default for LoggingQualityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            queue_capacity: 65_536,
            per_frame_sample_rate: 0.0,
            include_transcript_text: false,
            redaction_mode: RedactionMode::MetricsOnly,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityJudgeConfig {
    pub enabled: bool,
    pub mode: JudgeMode,
    pub sample_rate: f32,
    pub model: String,
    pub batch_size: usize,
    pub timeout_ms: u64,
}

impl Default for QualityJudgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: JudgeMode::Offline,
            sample_rate: 0.0,
            model: "default".to_string(),
            batch_size: 20,
            timeout_ms: 30_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityTargetsConfig {
    pub p50_endpoint_trailing_silence_ms: u64,
    pub p95_endpoint_trailing_silence_ms: u64,
    pub p50_turn_to_playback_started_ms: u64,
    pub p95_turn_to_playback_started_ms: u64,
    pub max_incomplete_turn_rate: f32,
    pub max_overmerged_turn_rate: f32,
    pub max_garbled_turn_rate: f32,
    pub max_inappropriate_cancel_rate: f32,
}

impl Default for QualityTargetsConfig {
    fn default() -> Self {
        Self {
            p50_endpoint_trailing_silence_ms: 900,
            p95_endpoint_trailing_silence_ms: 1_300,
            p50_turn_to_playback_started_ms: 1_200,
            p95_turn_to_playback_started_ms: 2_500,
            max_incomplete_turn_rate: 0.05,
            max_overmerged_turn_rate: 0.05,
            max_garbled_turn_rate: 0.03,
            max_inappropriate_cancel_rate: 0.03,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoiceQualityConfig {
    pub profile: QualityProfile,
    pub speech: SpeechQualityConfig,
    pub endpoint: EndpointQualityConfig,
    pub asr: AsrQualityConfig,
    pub text_call: TextCallQualityConfig,
    pub tts: TtsQualityConfig,
    #[serde(default)]
    pub early_response: EarlyResponsePolicy,
    pub barge_in: BargeInQualityConfig,
    #[serde(default)]
    pub echo_suppression: EchoSuppressionQualityConfig,
    pub logging: LoggingQualityConfig,
    pub quality_judge: QualityJudgeConfig,
    pub targets: QualityTargetsConfig,
}

impl Default for VoiceQualityConfig {
    fn default() -> Self {
        Self::for_profile(QualityProfile::Balanced)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualityMutationOutcome {
    pub key: &'static str,
    pub value: String,
    pub config_id: String,
    pub apply_boundary: ApplyBoundary,
    pub clamped: bool,
}

impl VoiceQualityConfig {
    pub fn for_profile(profile: QualityProfile) -> Self {
        let mut config = Self {
            profile,
            speech: SpeechQualityConfig::default(),
            endpoint: EndpointQualityConfig::default(),
            asr: AsrQualityConfig::default(),
            text_call: TextCallQualityConfig::default(),
            tts: TtsQualityConfig::default(),
            early_response: EarlyResponsePolicy::default(),
            barge_in: BargeInQualityConfig::default(),
            echo_suppression: EchoSuppressionQualityConfig::default(),
            logging: LoggingQualityConfig::default(),
            quality_judge: QualityJudgeConfig::default(),
            targets: QualityTargetsConfig::default(),
        };
        match profile {
            QualityProfile::Fast => {
                config.endpoint.trailing_silence_ms = 550;
                config.endpoint.final_settle_ms = 350;
                config.endpoint.merge_window_ms = 120;
                config.endpoint.conversation_incomplete_tail_hold_ms = 700;
                config.endpoint.conversation_playback_hold_poll_ms = 50;
                config.asr.finish_pad_ms = 80;
            }
            QualityProfile::Balanced => {}
            QualityProfile::Complete => {
                config.endpoint.trailing_silence_ms = 1_100;
                config.endpoint.final_settle_ms = 1_000;
                config.asr.finish_pad_ms = 320;
            }
            QualityProfile::Noisy => {
                config.endpoint.trailing_silence_ms = 950;
                config.endpoint.final_settle_ms = 900;
                config.speech.rms_threshold = 260.0;
                config.speech.peak_threshold = 1_200;
                config.asr.finish_pad_ms = 240;
            }
        }
        config
    }

    pub fn config_id(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("voice quality config serializes");
        let digest = Sha256::digest(encoded);
        format!("cfg_{}", hex::encode(digest))
    }

    pub fn validate_resolved(&self) -> Result<()> {
        ensure_f32(
            "speech.rms_threshold",
            self.speech.rms_threshold,
            0.0,
            20_000.0,
        )?;
        ensure_i32(
            "speech.peak_threshold",
            self.speech.peak_threshold,
            0,
            32_767,
        )?;
        ensure_u64(
            "speech.onset_min_silence_ms",
            self.speech.onset_min_silence_ms,
            0,
            2_000,
        )?;
        ensure_u64(
            "endpoint.trailing_silence_ms",
            self.endpoint.trailing_silence_ms,
            100,
            5_000,
        )?;
        ensure_usize(
            "endpoint.min_turn_words",
            self.endpoint.min_turn_words,
            0,
            50,
        )?;
        ensure_usize(
            "endpoint.min_turn_chars",
            self.endpoint.min_turn_chars,
            0,
            200,
        )?;
        ensure_u64(
            "endpoint.merge_window_ms",
            self.endpoint.merge_window_ms,
            0,
            5_000,
        )?;
        ensure_u64(
            "endpoint.final_settle_ms",
            self.endpoint.final_settle_ms,
            0,
            5_000,
        )?;
        ensure_string_list(
            "endpoint.final_settle_trailing_punctuation",
            &self.endpoint.final_settle_trailing_punctuation,
            8,
            32,
        )?;
        ensure_string_list(
            "endpoint.final_settle_lead_words",
            &self.endpoint.final_settle_lead_words,
            40,
            256,
        )?;
        ensure_string_list(
            "endpoint.final_settle_tail_words",
            &self.endpoint.final_settle_tail_words,
            64,
            256,
        )?;
        ensure_string_list(
            "endpoint.final_settle_dangling_suffixes",
            &self.endpoint.final_settle_dangling_suffixes,
            8,
            32,
        )?;
        ensure_string_list(
            "endpoint.conversation_tail_words",
            &self.endpoint.conversation_tail_words,
            64,
            256,
        )?;
        ensure_u64(
            "endpoint.conversation_incomplete_tail_hold_ms",
            self.endpoint.conversation_incomplete_tail_hold_ms,
            0,
            10_000,
        )?;
        ensure_u64(
            "endpoint.conversation_low_confidence_threshold_percent",
            self.endpoint.conversation_low_confidence_threshold_percent,
            0,
            100,
        )?;
        ensure_u64(
            "endpoint.conversation_playback_hold_poll_ms",
            self.endpoint.conversation_playback_hold_poll_ms,
            10,
            1_000,
        )?;
        ensure_usize(
            "endpoint.max_turn_words",
            self.endpoint.max_turn_words,
            1,
            500,
        )?;
        ensure_u64(
            "endpoint.max_turn_duration_ms",
            self.endpoint.max_turn_duration_ms,
            1_000,
            120_000,
        )?;
        ensure_usize(
            "asr.repeated_token_run_threshold",
            self.asr.repeated_token_run_threshold,
            2,
            128,
        )?;
        ensure_usize(
            "asr.repeated_q_run_threshold",
            self.asr.repeated_q_run_threshold,
            2,
            64,
        )?;
        ensure_u64("asr.finish_pad_ms", self.asr.finish_pad_ms, 0, 2_000)?;
        ensure_usize(
            "text_call.max_active_turns",
            self.text_call.max_active_turns,
            1,
            1_024,
        )?;
        ensure_u64(
            "text_call.media_ready_timeout_ms",
            self.text_call.media_ready_timeout_ms,
            1_000,
            120_000,
        )?;
        ensure_u64(
            "text_call.playback_wait_timeout_ms",
            self.text_call.playback_wait_timeout_ms,
            1_000,
            600_000,
        )?;
        ensure_u64(
            "text_call.callback_timeout_ms",
            self.text_call.callback_timeout_ms,
            100,
            60_000,
        )?;
        ensure_usize(
            "tts.max_text_chunk_chars",
            self.tts.max_text_chunk_chars,
            40,
            500,
        )?;
        if self.tts.first_chunk_max_chars != 0 {
            ensure_usize(
                "tts.first_chunk_max_chars",
                self.tts.first_chunk_max_chars,
                40,
                500,
            )?;
        }
        ensure_usize("tts.prebuffer_chunks", self.tts.prebuffer_chunks, 1, 64)?;
        ensure_usize(
            "early_response.min_text_chars",
            self.early_response.min_text_chars,
            1,
            500,
        )?;
        ensure_usize(
            "early_response.min_text_tokens",
            self.early_response.min_text_tokens,
            1,
            100,
        )?;
        if let Some(value) = self.early_response.min_confidence {
            ensure_f32("early_response.min_confidence", value, 0.0, 1.0)?;
        }
        if let Some(value) = self.early_response.min_stability {
            ensure_f32("early_response.min_stability", value, 0.0, 1.0)?;
        }
        ensure_u64(
            "early_response.debounce_ms",
            self.early_response.debounce_ms,
            0,
            5_000,
        )?;
        ensure_usize(
            "early_response.max_updates_per_utterance",
            self.early_response.max_updates_per_utterance,
            0,
            128,
        )?;
        ensure_usize(
            "early_response.provisional_max_prebuffer_frames",
            self.early_response.provisional_max_prebuffer_frames,
            1,
            1,
        )?;
        ensure_u64(
            "barge_in.clear_timeout_ms",
            self.barge_in.clear_timeout_ms,
            100,
            10_000,
        )?;
        ensure_usize(
            "echo_suppression.min_text_chars",
            self.echo_suppression.min_text_chars,
            1,
            500,
        )?;
        ensure_u64(
            "echo_suppression.tail_window_ms",
            self.echo_suppression.tail_window_ms,
            0,
            10_000,
        )?;
        ensure_u64(
            "echo_suppression.short_token_coverage_percent",
            self.echo_suppression.short_token_coverage_percent,
            0,
            100,
        )?;
        ensure_usize(
            "echo_suppression.short_longest_token_run",
            self.echo_suppression.short_longest_token_run,
            1,
            64,
        )?;
        ensure_usize(
            "echo_suppression.long_min_tokens",
            self.echo_suppression.long_min_tokens,
            2,
            64,
        )?;
        ensure_u64(
            "echo_suppression.long_token_coverage_percent",
            self.echo_suppression.long_token_coverage_percent,
            0,
            100,
        )?;
        ensure_usize(
            "echo_suppression.long_longest_token_run",
            self.echo_suppression.long_longest_token_run,
            1,
            64,
        )?;
        ensure_usize(
            "logging.queue_capacity",
            self.logging.queue_capacity,
            1_024,
            1_048_576,
        )?;
        ensure_f32(
            "logging.per_frame_sample_rate",
            self.logging.per_frame_sample_rate,
            0.0,
            1.0,
        )?;
        ensure_f32(
            "quality_judge.sample_rate",
            self.quality_judge.sample_rate,
            0.0,
            1.0,
        )?;
        ensure_usize(
            "quality_judge.batch_size",
            self.quality_judge.batch_size,
            1,
            1_000,
        )?;
        ensure_u64(
            "quality_judge.timeout_ms",
            self.quality_judge.timeout_ms,
            1_000,
            120_000,
        )?;
        if self.quality_judge.model.trim().is_empty() {
            anyhow::bail!("quality_judge.model must be non-empty");
        }
        ensure_f32(
            "targets.max_incomplete_turn_rate",
            self.targets.max_incomplete_turn_rate,
            0.0,
            1.0,
        )?;
        ensure_f32(
            "targets.max_overmerged_turn_rate",
            self.targets.max_overmerged_turn_rate,
            0.0,
            1.0,
        )?;
        ensure_f32(
            "targets.max_garbled_turn_rate",
            self.targets.max_garbled_turn_rate,
            0.0,
            1.0,
        )?;
        ensure_f32(
            "targets.max_inappropriate_cancel_rate",
            self.targets.max_inappropriate_cancel_rate,
            0.0,
            1.0,
        )?;
        Ok(())
    }

    pub fn load_toml(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read quality config {}", path.display()))?;
        Self::from_toml_str(&raw)
            .with_context(|| format!("parse quality config {}", path.display()))
    }

    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let mut config = Self::default();
        config.apply_toml_str(raw)?;
        Ok(config)
    }

    pub fn apply_toml_file(&mut self, path: &Path) -> Result<()> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read quality config {}", path.display()))?;
        self.apply_toml_str(&raw)
            .with_context(|| format!("parse quality config {}", path.display()))
    }

    pub fn apply_toml_str(&mut self, raw: &str) -> Result<()> {
        let document: QualityConfigDocument = toml::from_str(raw)?;
        if let Some(patch) = document.voice_quality {
            self.apply_patch(patch)?;
        }
        Ok(())
    }

    pub fn apply_patch(&mut self, patch: QualityConfigPatch) -> Result<()> {
        if let Some(profile) = patch.profile {
            *self = Self::for_profile(profile);
        }
        if let Some(speech) = patch.speech {
            if let Some(value) = speech.rms_threshold {
                self.set_speech_rms_threshold(value)?;
            }
            if let Some(value) = speech.peak_threshold {
                self.set_speech_peak_threshold(value);
            }
            if let Some(value) = speech.onset_min_silence_ms {
                self.set_speech_onset_min_silence_ms(value);
            }
        }
        if let Some(endpoint) = patch.endpoint {
            if let Some(value) = endpoint.trailing_silence_ms {
                self.set_endpoint_trailing_silence_ms(value);
            }
            if let Some(value) = endpoint.min_turn_words {
                self.set_endpoint_min_turn_words(value);
            }
            if let Some(value) = endpoint.min_turn_chars {
                self.set_endpoint_min_turn_chars(value);
            }
            if let Some(value) = endpoint.merge_window_ms {
                self.set_endpoint_merge_window_ms(value);
            }
            if let Some(value) = endpoint.final_settle_ms {
                self.set_endpoint_final_settle_ms(value);
            }
            if let Some(value) = endpoint.final_settle_trailing_punctuation {
                self.endpoint.final_settle_trailing_punctuation = value;
            }
            if let Some(value) = endpoint.final_settle_lead_words {
                self.endpoint.final_settle_lead_words = value;
            }
            if let Some(value) = endpoint.final_settle_tail_words {
                self.endpoint.final_settle_tail_words = value;
            }
            if let Some(value) = endpoint.final_settle_dangling_suffixes {
                self.endpoint.final_settle_dangling_suffixes = value;
            }
            if let Some(value) = endpoint.conversation_tail_words {
                self.endpoint.conversation_tail_words = value;
            }
            if let Some(value) = endpoint.conversation_incomplete_tail_hold_ms {
                self.set_endpoint_conversation_incomplete_tail_hold_ms(value);
            }
            if let Some(value) = endpoint.conversation_low_confidence_threshold_percent {
                self.set_endpoint_conversation_low_confidence_threshold_percent(value);
            }
            if let Some(value) = endpoint.conversation_playback_hold_poll_ms {
                self.set_endpoint_conversation_playback_hold_poll_ms(value);
            }
            if let Some(value) = endpoint.max_turn_words {
                self.set_endpoint_max_turn_words(value);
            }
            if let Some(value) = endpoint.max_turn_duration_ms {
                self.set_endpoint_max_turn_duration_ms(value);
            }
        }
        if let Some(asr) = patch.asr {
            if let Some(value) = asr.repeated_token_run_threshold {
                self.set_asr_repeated_token_run_threshold(value);
            }
            if let Some(value) = asr.repeated_q_run_threshold {
                self.set_asr_repeated_q_run_threshold(value);
            }
            if let Some(value) = asr.finish_pad_ms {
                self.set_asr_finish_pad_ms(value);
            }
        }
        if let Some(text_call) = patch.text_call {
            if let Some(value) = text_call.max_active_turns {
                self.set_text_call_max_active_turns(value);
            }
            if let Some(value) = text_call.media_ready_timeout_ms {
                self.set_text_call_media_ready_timeout_ms(value);
            }
            if let Some(value) = text_call.playback_wait_timeout_ms {
                self.set_text_call_playback_wait_timeout_ms(value);
            }
            if let Some(value) = text_call.latest_response_wins {
                self.set_text_call_latest_response_wins(value);
            }
            if let Some(value) = text_call.callback_timeout_ms {
                self.set_text_call_callback_timeout_ms(value);
            }
        }
        if let Some(tts) = patch.tts {
            if let Some(value) = tts.generation_mode {
                self.set_tts_generation_mode(value);
            }
            if let Some(value) = tts.chunking_enabled {
                self.set_tts_chunking_enabled(value);
            }
            if let Some(value) = tts.max_text_chunk_chars {
                self.set_tts_max_text_chunk_chars(value);
            }
            if let Some(value) = tts.first_chunk_max_chars {
                self.set_tts_first_chunk_max_chars(value);
            }
            if let Some(value) = tts.prebuffer_chunks {
                self.set_tts_prebuffer_chunks(value);
            }
        }
        if let Some(early_response) = patch.early_response {
            self.early_response = early_response;
        }
        if let Some(barge_in) = patch.barge_in {
            if let Some(value) = barge_in.enabled {
                self.set_barge_in_enabled(value);
            }
            if let Some(value) = barge_in.speech_onset_cancel_enabled {
                self.set_barge_in_speech_onset_cancel_enabled(value);
            }
            if let Some(value) = barge_in.onset_during_playback {
                self.set_barge_in_onset_during_playback(value);
            }
            if let Some(value) = barge_in.partial_asr_cancel_enabled {
                self.set_barge_in_partial_asr_cancel_enabled(value);
            }
            if let Some(value) = barge_in.final_asr_cancel_enabled {
                self.set_barge_in_final_asr_cancel_enabled(value);
            }
            if let Some(value) = barge_in.clear_timeout_ms {
                self.set_barge_in_clear_timeout_ms(value);
            }
        }
        if let Some(echo) = patch.echo_suppression {
            if let Some(value) = echo.enabled {
                self.set_echo_suppression_enabled(value);
            }
            if let Some(value) = echo.min_text_chars {
                self.set_echo_suppression_min_text_chars(value);
            }
            if let Some(value) = echo.tail_window_ms {
                self.set_echo_suppression_tail_window_ms(value);
            }
            if let Some(value) = echo.short_token_coverage_percent {
                self.set_echo_suppression_short_token_coverage_percent(value);
            }
            if let Some(value) = echo.short_longest_token_run {
                self.set_echo_suppression_short_longest_token_run(value);
            }
            if let Some(value) = echo.long_min_tokens {
                self.set_echo_suppression_long_min_tokens(value);
            }
            if let Some(value) = echo.long_token_coverage_percent {
                self.set_echo_suppression_long_token_coverage_percent(value);
            }
            if let Some(value) = echo.long_longest_token_run {
                self.set_echo_suppression_long_longest_token_run(value);
            }
        }
        if let Some(logging) = patch.logging {
            if let Some(value) = logging.enabled {
                self.logging.enabled = value;
            }
            if let Some(value) = logging.queue_capacity {
                self.set_logging_queue_capacity(value);
            }
            if let Some(value) = logging.per_frame_sample_rate {
                self.set_logging_per_frame_sample_rate(value)?;
            }
            if let Some(value) = logging.include_transcript_text {
                self.logging.include_transcript_text = value;
            }
            if let Some(value) = logging.redaction_mode {
                self.logging.redaction_mode = value;
            }
        }
        if let Some(judge) = patch.quality_judge.or(patch.llm_judge) {
            if let Some(value) = judge.enabled {
                self.quality_judge.enabled = value;
            }
            if let Some(value) = judge.mode {
                self.quality_judge.mode = value;
            }
            if let Some(value) = judge.sample_rate {
                self.set_quality_judge_sample_rate(value)?;
            }
            if let Some(value) = judge.model {
                if value.trim().is_empty() {
                    anyhow::bail!("quality_judge.model must be non-empty");
                }
                self.quality_judge.model = value;
            }
            if let Some(value) = judge.batch_size {
                self.set_quality_judge_batch_size(value);
            }
            if let Some(value) = judge.timeout_ms {
                self.set_quality_judge_timeout_ms(value);
            }
        }
        if let Some(targets) = patch.targets {
            if let Some(value) = targets.p50_endpoint_trailing_silence_ms {
                self.targets.p50_endpoint_trailing_silence_ms = value;
            }
            if let Some(value) = targets.p95_endpoint_trailing_silence_ms {
                self.targets.p95_endpoint_trailing_silence_ms = value;
            }
            if let Some(value) = targets.p50_turn_to_playback_started_ms {
                self.targets.p50_turn_to_playback_started_ms = value;
            }
            if let Some(value) = targets.p95_turn_to_playback_started_ms {
                self.targets.p95_turn_to_playback_started_ms = value;
            }
            if let Some(value) = targets.max_incomplete_turn_rate {
                self.targets.max_incomplete_turn_rate = clamp_ratio(value)?;
            }
            if let Some(value) = targets.max_overmerged_turn_rate {
                self.targets.max_overmerged_turn_rate = clamp_ratio(value)?;
            }
            if let Some(value) = targets.max_garbled_turn_rate {
                self.targets.max_garbled_turn_rate = clamp_ratio(value)?;
            }
            if let Some(value) = targets.max_inappropriate_cancel_rate {
                self.targets.max_inappropriate_cancel_rate = clamp_ratio(value)?;
            }
        }
        self.validate_resolved()?;
        Ok(())
    }

    pub fn set_profile(&mut self, profile: QualityProfile) -> QualityMutationOutcome {
        *self = Self::for_profile(profile);
        self.outcome(
            "profile",
            profile.label(),
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_speech_rms_threshold(&mut self, value: f32) -> Result<QualityMutationOutcome> {
        if !value.is_finite() {
            anyhow::bail!("speech.rms_threshold must be finite");
        }
        let clamped = clamp_f32(value, 0.0, 20_000.0);
        self.speech.rms_threshold = clamped.value;
        Ok(self.outcome(
            "speech.rms_threshold",
            format_float(clamped.value),
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        ))
    }

    pub fn set_speech_peak_threshold(&mut self, value: i32) -> QualityMutationOutcome {
        let clamped = clamp_i32(value, 0, 32_767);
        self.speech.peak_threshold = clamped.value;
        self.outcome(
            "speech.peak_threshold",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_speech_onset_min_silence_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 2_000);
        self.speech.onset_min_silence_ms = clamped.value;
        self.outcome(
            "speech.onset_min_silence_ms",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_trailing_silence_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 100, 5_000);
        self.endpoint.trailing_silence_ms = clamped.value;
        self.outcome(
            "endpoint.trailing_silence_ms",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_min_turn_words(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 0, 50);
        self.endpoint.min_turn_words = clamped.value;
        self.outcome(
            "endpoint.min_turn_words",
            clamped.value,
            ApplyBoundary::ReportOnly,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_min_turn_chars(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 0, 200);
        self.endpoint.min_turn_chars = clamped.value;
        self.outcome(
            "endpoint.min_turn_chars",
            clamped.value,
            ApplyBoundary::ReportOnly,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_merge_window_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 5_000);
        self.endpoint.merge_window_ms = clamped.value;
        self.outcome(
            "endpoint.merge_window_ms",
            clamped.value,
            ApplyBoundary::NewTurn,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_final_settle_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 5_000);
        self.endpoint.final_settle_ms = clamped.value;
        self.outcome(
            "endpoint.final_settle_ms",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_conversation_incomplete_tail_hold_ms(
        &mut self,
        value: u64,
    ) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 10_000);
        self.endpoint.conversation_incomplete_tail_hold_ms = clamped.value;
        self.outcome(
            "endpoint.conversation_incomplete_tail_hold_ms",
            clamped.value,
            ApplyBoundary::NewTurn,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_conversation_low_confidence_threshold_percent(
        &mut self,
        value: u64,
    ) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 100);
        self.endpoint.conversation_low_confidence_threshold_percent = clamped.value;
        self.outcome(
            "endpoint.conversation_low_confidence_threshold_percent",
            clamped.value,
            ApplyBoundary::NewTurn,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_conversation_playback_hold_poll_ms(
        &mut self,
        value: u64,
    ) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 10, 1_000);
        self.endpoint.conversation_playback_hold_poll_ms = clamped.value;
        self.outcome(
            "endpoint.conversation_playback_hold_poll_ms",
            clamped.value,
            ApplyBoundary::NewTurn,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_max_turn_words(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 500);
        self.endpoint.max_turn_words = clamped.value;
        self.outcome(
            "endpoint.max_turn_words",
            clamped.value,
            ApplyBoundary::ReportOnly,
            clamped.clamped,
        )
    }

    pub fn set_endpoint_max_turn_duration_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 1_000, 120_000);
        self.endpoint.max_turn_duration_ms = clamped.value;
        self.outcome(
            "endpoint.max_turn_duration_ms",
            clamped.value,
            ApplyBoundary::ReportOnly,
            clamped.clamped,
        )
    }

    pub fn set_asr_repeated_token_run_threshold(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 2, 128);
        self.asr.repeated_token_run_threshold = clamped.value;
        self.outcome(
            "asr.repeated_token_run_threshold",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_asr_repeated_q_run_threshold(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 2, 64);
        self.asr.repeated_q_run_threshold = clamped.value;
        self.outcome(
            "asr.repeated_q_run_threshold",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_asr_finish_pad_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 2_000);
        self.asr.finish_pad_ms = clamped.value;
        self.outcome(
            "asr.finish_pad_ms",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_text_call_max_active_turns(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 1_024);
        self.text_call.max_active_turns = clamped.value;
        self.outcome(
            "text_call.max_active_turns",
            clamped.value,
            ApplyBoundary::NewTextCallSession,
            clamped.clamped,
        )
    }

    pub fn set_text_call_media_ready_timeout_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 1_000, 120_000);
        self.text_call.media_ready_timeout_ms = clamped.value;
        self.outcome(
            "text_call.media_ready_timeout_ms",
            clamped.value,
            ApplyBoundary::NewTextCallSession,
            clamped.clamped,
        )
    }

    pub fn set_text_call_playback_wait_timeout_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 1_000, 600_000);
        self.text_call.playback_wait_timeout_ms = clamped.value;
        self.outcome(
            "text_call.playback_wait_timeout_ms",
            clamped.value,
            ApplyBoundary::NewTextCallSession,
            clamped.clamped,
        )
    }

    pub fn set_text_call_latest_response_wins(&mut self, value: bool) -> QualityMutationOutcome {
        self.text_call.latest_response_wins = value;
        self.outcome(
            "text_call.latest_response_wins",
            value,
            ApplyBoundary::NewTextCallSession,
            false,
        )
    }

    pub fn set_text_call_callback_timeout_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 100, 60_000);
        self.text_call.callback_timeout_ms = clamped.value;
        self.outcome(
            "text_call.callback_timeout_ms",
            clamped.value,
            ApplyBoundary::NewTurn,
            clamped.clamped,
        )
    }

    pub fn set_tts_chunking_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.tts.chunking_enabled = value;
        self.outcome(
            "tts.chunking_enabled",
            value,
            ApplyBoundary::NewPlaybackRequest,
            false,
        )
    }

    pub fn set_tts_generation_mode(&mut self, value: TtsGenerationMode) -> QualityMutationOutcome {
        self.tts.generation_mode = value;
        self.outcome(
            "tts.generation_mode",
            value.label(),
            ApplyBoundary::NewPlaybackRequest,
            false,
        )
    }

    pub fn set_tts_max_text_chunk_chars(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 40, 500);
        self.tts.max_text_chunk_chars = clamped.value;
        self.outcome(
            "tts.max_text_chunk_chars",
            clamped.value,
            ApplyBoundary::NewPlaybackRequest,
            clamped.clamped,
        )
    }

    pub fn set_tts_first_chunk_max_chars(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = if value == 0 {
            Clamped {
                value,
                clamped: false,
            }
        } else {
            clamp_usize(value, 40, 500)
        };
        self.tts.first_chunk_max_chars = clamped.value;
        self.outcome(
            "tts.first_chunk_max_chars",
            clamped.value,
            ApplyBoundary::NewPlaybackRequest,
            clamped.clamped,
        )
    }

    pub fn set_tts_prebuffer_chunks(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 64);
        self.tts.prebuffer_chunks = clamped.value;
        self.outcome(
            "tts.prebuffer_chunks",
            clamped.value,
            ApplyBoundary::NewPlaybackRequest,
            clamped.clamped,
        )
    }

    pub fn set_early_response_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.early_response.enabled = value;
        self.outcome(
            "early_response.enabled",
            value,
            ApplyBoundary::NewCall,
            false,
        )
    }

    pub fn set_early_response_start_timing(
        &mut self,
        value: EarlyResponseStartTiming,
    ) -> QualityMutationOutcome {
        self.early_response.set_start_timing(value);
        self.outcome(
            "early_response.start_timing",
            match value {
                EarlyResponseStartTiming::EndpointCandidateOnly => "endpoint_candidate_only",
                EarlyResponseStartTiming::WhileSpeaking => "while_speaking",
            },
            ApplyBoundary::NewCall,
            false,
        )
    }

    pub fn set_early_response_boundary(
        &mut self,
        value: BoundaryRequirement,
    ) -> QualityMutationOutcome {
        self.early_response.boundary = value;
        self.outcome(
            "early_response.boundary",
            match value {
                BoundaryRequirement::None => "none",
                BoundaryRequirement::Clause => "clause",
                BoundaryRequirement::Sentence => "sentence",
            },
            ApplyBoundary::NewCall,
            false,
        )
    }

    pub fn set_barge_in_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.barge_in.enabled = value;
        self.outcome(
            "barge_in.enabled",
            value,
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_barge_in_speech_onset_cancel_enabled(
        &mut self,
        value: bool,
    ) -> QualityMutationOutcome {
        self.barge_in.speech_onset_cancel_enabled = value;
        self.outcome(
            "barge_in.speech_onset_cancel_enabled",
            value,
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_barge_in_onset_during_playback(
        &mut self,
        value: OnsetDuringPlaybackPolicy,
    ) -> QualityMutationOutcome {
        self.barge_in.onset_during_playback = value;
        self.outcome(
            "barge_in.onset_during_playback",
            value.label(),
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_barge_in_partial_asr_cancel_enabled(
        &mut self,
        value: bool,
    ) -> QualityMutationOutcome {
        self.barge_in.partial_asr_cancel_enabled = value;
        self.outcome(
            "barge_in.partial_asr_cancel_enabled",
            value,
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_barge_in_final_asr_cancel_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.barge_in.final_asr_cancel_enabled = value;
        self.outcome(
            "barge_in.final_asr_cancel_enabled",
            value,
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_barge_in_clear_timeout_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 100, 10_000);
        self.barge_in.clear_timeout_ms = clamped.value;
        self.outcome(
            "barge_in.clear_timeout_ms",
            clamped.value,
            ApplyBoundary::NewTurn,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.echo_suppression.enabled = value;
        self.outcome(
            "echo_suppression.enabled",
            value,
            ApplyBoundary::NextAsrSession,
            false,
        )
    }

    pub fn set_echo_suppression_min_text_chars(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 500);
        self.echo_suppression.min_text_chars = clamped.value;
        self.outcome(
            "echo_suppression.min_text_chars",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_tail_window_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 10_000);
        self.echo_suppression.tail_window_ms = clamped.value;
        self.outcome(
            "echo_suppression.tail_window_ms",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_short_token_coverage_percent(
        &mut self,
        value: u64,
    ) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 100);
        self.echo_suppression.short_token_coverage_percent = clamped.value;
        self.outcome(
            "echo_suppression.short_token_coverage_percent",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_short_longest_token_run(
        &mut self,
        value: usize,
    ) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 64);
        self.echo_suppression.short_longest_token_run = clamped.value;
        self.outcome(
            "echo_suppression.short_longest_token_run",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_long_min_tokens(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 2, 64);
        self.echo_suppression.long_min_tokens = clamped.value;
        self.outcome(
            "echo_suppression.long_min_tokens",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_long_token_coverage_percent(
        &mut self,
        value: u64,
    ) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 0, 100);
        self.echo_suppression.long_token_coverage_percent = clamped.value;
        self.outcome(
            "echo_suppression.long_token_coverage_percent",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_echo_suppression_long_longest_token_run(
        &mut self,
        value: usize,
    ) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 64);
        self.echo_suppression.long_longest_token_run = clamped.value;
        self.outcome(
            "echo_suppression.long_longest_token_run",
            clamped.value,
            ApplyBoundary::NextAsrSession,
            clamped.clamped,
        )
    }

    pub fn set_logging_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.logging.enabled = value;
        self.outcome("logging.enabled", value, ApplyBoundary::Immediate, false)
    }

    pub fn set_logging_queue_capacity(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1_024, 1_048_576);
        self.logging.queue_capacity = clamped.value;
        self.outcome(
            "logging.queue_capacity",
            clamped.value,
            ApplyBoundary::NextWriterStart,
            clamped.clamped,
        )
    }

    pub fn set_logging_per_frame_sample_rate(
        &mut self,
        value: f32,
    ) -> Result<QualityMutationOutcome> {
        if !value.is_finite() {
            anyhow::bail!("logging.per_frame_sample_rate must be finite");
        }
        let clamped = clamp_f32(value, 0.0, 1.0);
        self.logging.per_frame_sample_rate = clamped.value;
        Ok(self.outcome(
            "logging.per_frame_sample_rate",
            format_float(clamped.value),
            ApplyBoundary::Immediate,
            clamped.clamped,
        ))
    }

    pub fn set_logging_include_transcript_text(&mut self, value: bool) -> QualityMutationOutcome {
        self.logging.include_transcript_text = value;
        self.outcome(
            "logging.include_transcript_text",
            value,
            ApplyBoundary::Immediate,
            false,
        )
    }

    pub fn set_logging_redaction_mode(&mut self, value: RedactionMode) -> QualityMutationOutcome {
        self.logging.redaction_mode = value;
        self.outcome(
            "logging.redaction_mode",
            value.label(),
            ApplyBoundary::Immediate,
            false,
        )
    }

    pub fn set_quality_judge_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.quality_judge.enabled = value;
        self.outcome(
            "quality_judge.enabled",
            value,
            ApplyBoundary::NextJudgeJob,
            false,
        )
    }

    pub fn set_quality_judge_sample_rate(&mut self, value: f32) -> Result<QualityMutationOutcome> {
        if !value.is_finite() {
            anyhow::bail!("quality_judge.sample_rate must be finite");
        }
        let clamped = clamp_f32(value, 0.0, 1.0);
        self.quality_judge.sample_rate = clamped.value;
        Ok(self.outcome(
            "quality_judge.sample_rate",
            format_float(clamped.value),
            ApplyBoundary::NextJudgeJob,
            clamped.clamped,
        ))
    }

    pub fn set_quality_judge_batch_size(&mut self, value: usize) -> QualityMutationOutcome {
        let clamped = clamp_usize(value, 1, 1_000);
        self.quality_judge.batch_size = clamped.value;
        self.outcome(
            "quality_judge.batch_size",
            clamped.value,
            ApplyBoundary::NextJudgeJob,
            clamped.clamped,
        )
    }

    pub fn set_quality_judge_timeout_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 1_000, 120_000);
        self.quality_judge.timeout_ms = clamped.value;
        self.outcome(
            "quality_judge.timeout_ms",
            clamped.value,
            ApplyBoundary::NextJudgeJob,
            clamped.clamped,
        )
    }

    fn outcome(
        &self,
        key: &'static str,
        value: impl ToString,
        apply_boundary: ApplyBoundary,
        clamped: bool,
    ) -> QualityMutationOutcome {
        QualityMutationOutcome {
            key,
            value: value.to_string(),
            config_id: self.config_id(),
            apply_boundary,
            clamped,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Clamped<T> {
    value: T,
    clamped: bool,
}

fn clamp_u64(value: u64, min: u64, max: u64) -> Clamped<u64> {
    let clamped_value = value.clamp(min, max);
    Clamped {
        value: clamped_value,
        clamped: clamped_value != value,
    }
}

fn clamp_usize(value: usize, min: usize, max: usize) -> Clamped<usize> {
    let clamped_value = value.clamp(min, max);
    Clamped {
        value: clamped_value,
        clamped: clamped_value != value,
    }
}

fn clamp_i32(value: i32, min: i32, max: i32) -> Clamped<i32> {
    let clamped_value = value.clamp(min, max);
    Clamped {
        value: clamped_value,
        clamped: clamped_value != value,
    }
}

fn clamp_f32(value: f32, min: f32, max: f32) -> Clamped<f32> {
    let clamped_value = value.clamp(min, max);
    Clamped {
        value: clamped_value,
        clamped: clamped_value != value,
    }
}

fn clamp_ratio(value: f32) -> Result<f32> {
    if !value.is_finite() {
        anyhow::bail!("ratio must be finite");
    }
    Ok(value.clamp(0.0, 1.0))
}

fn format_float(value: f32) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

fn ensure_u64(name: &'static str, value: u64, min: u64, max: u64) -> Result<()> {
    if !(min..=max).contains(&value) {
        anyhow::bail!("{name} must be in {min}..={max}");
    }
    Ok(())
}

fn ensure_usize(name: &'static str, value: usize, min: usize, max: usize) -> Result<()> {
    if !(min..=max).contains(&value) {
        anyhow::bail!("{name} must be in {min}..={max}");
    }
    Ok(())
}

fn ensure_i32(name: &'static str, value: i32, min: i32, max: i32) -> Result<()> {
    if !(min..=max).contains(&value) {
        anyhow::bail!("{name} must be in {min}..={max}");
    }
    Ok(())
}

fn ensure_f32(name: &'static str, value: f32, min: f32, max: f32) -> Result<()> {
    if !value.is_finite() {
        anyhow::bail!("{name} must be finite");
    }
    if value < min || value > max {
        anyhow::bail!("{name} must be in {min}..={max}");
    }
    Ok(())
}

fn ensure_string_list(
    name: &'static str,
    values: &[String],
    max_items: usize,
    max_chars: usize,
) -> Result<()> {
    if values.len() > max_items {
        anyhow::bail!("{name} must have at most {max_items} entries");
    }
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            anyhow::bail!("{name} entries must be non-empty when present");
        }
        if trimmed != value {
            anyhow::bail!("{name} entries must not have surrounding whitespace");
        }
        if value.chars().count() > max_chars {
            anyhow::bail!("{name} entries must be at most {max_chars} characters");
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct QualityConfigDocument {
    #[serde(rename = "version")]
    _version: Option<u32>,
    #[serde(rename = "generated_at")]
    _generated_at: Option<String>,
    #[serde(rename = "process")]
    _process: Option<toml::Value>,
    #[serde(rename = "telnyx")]
    _telnyx: Option<toml::Value>,
    #[serde(rename = "gateway")]
    _gateway: Option<toml::Value>,
    #[serde(rename = "inbound")]
    _inbound: Option<toml::Value>,
    #[serde(rename = "conversation")]
    _conversation: Option<toml::Value>,
    #[serde(rename = "startup")]
    _startup: Option<toml::Value>,
    #[serde(rename = "quality_logging")]
    _quality_logging: Option<toml::Value>,
    #[serde(default)]
    voice_quality: Option<QualityConfigPatch>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityConfigPatch {
    pub profile: Option<QualityProfile>,
    #[serde(default)]
    pub speech: Option<SpeechQualityConfigPatch>,
    #[serde(default)]
    pub endpoint: Option<EndpointQualityConfigPatch>,
    #[serde(default)]
    pub asr: Option<AsrQualityConfigPatch>,
    #[serde(default)]
    pub text_call: Option<TextCallQualityConfigPatch>,
    #[serde(default)]
    pub tts: Option<TtsQualityConfigPatch>,
    #[serde(default)]
    pub early_response: Option<EarlyResponsePolicy>,
    #[serde(default)]
    pub barge_in: Option<BargeInQualityConfigPatch>,
    #[serde(default)]
    pub echo_suppression: Option<EchoSuppressionQualityConfigPatch>,
    #[serde(default)]
    pub logging: Option<LoggingQualityConfigPatch>,
    #[serde(default)]
    pub quality_judge: Option<QualityJudgeConfigPatch>,
    #[serde(default)]
    pub llm_judge: Option<QualityJudgeConfigPatch>,
    #[serde(default)]
    pub targets: Option<QualityTargetsConfigPatch>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeechQualityConfigPatch {
    pub rms_threshold: Option<f32>,
    pub peak_threshold: Option<i32>,
    pub onset_min_silence_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EndpointQualityConfigPatch {
    pub trailing_silence_ms: Option<u64>,
    pub min_turn_words: Option<usize>,
    pub min_turn_chars: Option<usize>,
    pub merge_window_ms: Option<u64>,
    pub final_settle_ms: Option<u64>,
    pub final_settle_trailing_punctuation: Option<Vec<String>>,
    pub final_settle_lead_words: Option<Vec<String>>,
    pub final_settle_tail_words: Option<Vec<String>>,
    pub final_settle_dangling_suffixes: Option<Vec<String>>,
    pub conversation_tail_words: Option<Vec<String>>,
    pub conversation_incomplete_tail_hold_ms: Option<u64>,
    pub conversation_low_confidence_threshold_percent: Option<u64>,
    pub conversation_playback_hold_poll_ms: Option<u64>,
    pub max_turn_words: Option<usize>,
    pub max_turn_duration_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrQualityConfigPatch {
    pub repeated_token_run_threshold: Option<usize>,
    pub repeated_q_run_threshold: Option<usize>,
    pub finish_pad_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextCallQualityConfigPatch {
    pub max_active_turns: Option<usize>,
    pub media_ready_timeout_ms: Option<u64>,
    pub playback_wait_timeout_ms: Option<u64>,
    pub latest_response_wins: Option<bool>,
    pub callback_timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TtsQualityConfigPatch {
    pub generation_mode: Option<TtsGenerationMode>,
    pub chunking_enabled: Option<bool>,
    pub max_text_chunk_chars: Option<usize>,
    pub first_chunk_max_chars: Option<usize>,
    pub prebuffer_chunks: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BargeInQualityConfigPatch {
    pub enabled: Option<bool>,
    pub speech_onset_cancel_enabled: Option<bool>,
    pub onset_during_playback: Option<OnsetDuringPlaybackPolicy>,
    pub partial_asr_cancel_enabled: Option<bool>,
    pub final_asr_cancel_enabled: Option<bool>,
    pub clear_timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EchoSuppressionQualityConfigPatch {
    pub enabled: Option<bool>,
    pub min_text_chars: Option<usize>,
    pub tail_window_ms: Option<u64>,
    pub short_token_coverage_percent: Option<u64>,
    pub short_longest_token_run: Option<usize>,
    pub long_min_tokens: Option<usize>,
    pub long_token_coverage_percent: Option<u64>,
    pub long_longest_token_run: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingQualityConfigPatch {
    pub enabled: Option<bool>,
    pub queue_capacity: Option<usize>,
    pub per_frame_sample_rate: Option<f32>,
    pub include_transcript_text: Option<bool>,
    pub redaction_mode: Option<RedactionMode>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityJudgeConfigPatch {
    pub enabled: Option<bool>,
    pub mode: Option<JudgeMode>,
    pub sample_rate: Option<f32>,
    pub model: Option<String>,
    pub batch_size: Option<usize>,
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityTargetsConfigPatch {
    pub p50_endpoint_trailing_silence_ms: Option<u64>,
    pub p95_endpoint_trailing_silence_ms: Option<u64>,
    pub p50_turn_to_playback_started_ms: Option<u64>,
    pub p95_turn_to_playback_started_ms: Option<u64>,
    pub max_incomplete_turn_rate: Option<f32>,
    pub max_overmerged_turn_rate: Option<f32>,
    pub max_garbled_turn_rate: Option<f32>,
    pub max_inappropriate_cancel_rate: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_transcript_text_is_disabled() {
        let config = VoiceQualityConfig::default();
        assert!(!config.logging.include_transcript_text);
        assert_eq!(config.logging.redaction_mode, RedactionMode::MetricsOnly);
    }

    #[test]
    fn balanced_defaults_match_live_call_tuned_values() {
        let config = VoiceQualityConfig::default();
        assert_eq!(config.profile, QualityProfile::Balanced);
        assert_eq!(config.endpoint.trailing_silence_ms, 900);
        assert_eq!(config.speech.rms_threshold, 220.0);
        assert_eq!(config.speech.peak_threshold, 1_100);
        assert_eq!(config.speech.onset_min_silence_ms, 180);
        assert_eq!(config.endpoint.final_settle_ms, 800);
        assert_eq!(config.asr.finish_pad_ms, 320);
        assert!(config.tts.chunking_enabled);
        assert_eq!(config.tts.max_text_chunk_chars, 90);
        assert_eq!(config.tts.first_chunk_max_chars, 40);
        assert_eq!(config.tts.prebuffer_chunks, 1);
    }

    #[test]
    fn config_id_is_stable_for_same_resolved_config() {
        let left = VoiceQualityConfig::default();
        let right = VoiceQualityConfig::for_profile(QualityProfile::Balanced);
        assert_eq!(left.config_id(), right.config_id());
    }

    #[test]
    fn live_endpoint_knobs_clamp_and_apply_at_next_session() {
        let mut config = VoiceQualityConfig::default();
        let outcome = config.set_endpoint_trailing_silence_ms(10);
        assert_eq!(config.endpoint.trailing_silence_ms, 100);
        assert!(outcome.clamped);
        assert_eq!(outcome.apply_boundary, ApplyBoundary::NextAsrSession);
    }

    #[test]
    fn merge_window_applies_to_new_conversation_turns() {
        let mut config = VoiceQualityConfig::default();
        let outcome = config.set_endpoint_merge_window_ms(9_999);
        assert_eq!(config.endpoint.merge_window_ms, 5_000);
        assert_eq!(outcome.apply_boundary, ApplyBoundary::NewTurn);
    }

    #[test]
    fn final_settle_knob_clamps_and_applies_at_next_session() {
        let mut config = VoiceQualityConfig::default();
        let outcome = config.set_endpoint_final_settle_ms(9_999);
        assert_eq!(config.endpoint.final_settle_ms, 5_000);
        assert_eq!(outcome.apply_boundary, ApplyBoundary::NextAsrSession);
    }

    #[test]
    fn endpoint_fragment_classifiers_follow_configured_policy() {
        let mut endpoint = EndpointQualityConfig::default();
        assert_eq!(
            endpoint.final_fragment_hold_reason("we also"),
            Some("tail_word")
        );
        assert_eq!(
            endpoint.conversation_incomplete_tail_reason("the endpoints are"),
            Some("tail_word")
        );

        endpoint.final_settle_tail_words.clear();
        endpoint.conversation_tail_words.clear();
        assert_eq!(endpoint.final_fragment_hold_reason("we also"), None);
        assert_eq!(
            endpoint.conversation_incomplete_tail_reason("the endpoints are"),
            None
        );
    }

    #[test]
    fn endpoint_policy_lists_reject_ambiguous_entries() {
        let mut config = VoiceQualityConfig::default();
        config
            .endpoint
            .final_settle_tail_words
            .push(" spaced".to_string());

        let error = config
            .validate_resolved()
            .expect_err("policy list entries with surrounding whitespace should be rejected");
        assert!(error
            .to_string()
            .contains("endpoint.final_settle_tail_words"));
    }

    #[test]
    fn early_response_provisional_prebuffer_rejects_values_above_hard_cap() {
        let mut config = VoiceQualityConfig::default();
        config.early_response.provisional_max_prebuffer_frames = 2;

        let error = config
            .validate_resolved()
            .expect_err("provisional prebuffer must stay at the one-frame cap");

        assert!(error
            .to_string()
            .contains("early_response.provisional_max_prebuffer_frames"));
    }

    #[test]
    fn rejects_non_finite_float_knobs() {
        let mut config = VoiceQualityConfig::default();
        let error = config
            .set_speech_rms_threshold(f32::NAN)
            .expect_err("NaN should be rejected");
        assert!(error.to_string().contains("finite"));
    }

    #[test]
    fn parses_partial_toml_over_profile_defaults() {
        let config = VoiceQualityConfig::from_toml_str(
            r#"
            [voice_quality]
            profile = "noisy"

            [voice_quality.endpoint]
            trailing_silence_ms = 50

            [voice_quality.logging]
            include_transcript_text = false
            redaction_mode = "metrics-only"
            "#,
        )
        .expect("quality config parses");

        assert_eq!(config.profile, QualityProfile::Noisy);
        assert_eq!(config.speech.rms_threshold, 260.0);
        assert_eq!(config.endpoint.trailing_silence_ms, 100);
        assert!(!config.logging.include_transcript_text);
    }

    #[test]
    fn toml_rejects_unknown_voice_quality_keys() {
        for raw in [
            r#"
            [voice_quality]
            generation_mod = "streaming"
            "#,
            r#"
            [voice_quality.tts]
            prebuffer_chunk = 1
            "#,
            r#"
            [voice_quality.early_response]
            start_timng = "endpoint_candidate_only"
            "#,
        ] {
            let error = VoiceQualityConfig::from_toml_str(raw)
                .expect_err("unknown voice_quality keys should fail closed");
            assert!(
                error.to_string().contains("unknown field"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn toml_accepts_gateway_metadata_and_known_outer_tables() {
        let config = VoiceQualityConfig::from_toml_str(
            r#"
            version = 1
            generated_at = "2026-06-16T00:00:00Z"

            [process]
            bind = "127.0.0.1:8080"

            [quality_logging]
            path = "$HOME/telnyx-test/quality-events.jsonl"

            [voice_quality.tts]
            generation_mode = "streaming"
            "#,
        )
        .expect("quality parser should accept full gateway TOML metadata");

        assert_eq!(config.tts.generation_mode, TtsGenerationMode::Streaming);
    }

    #[test]
    fn toml_overlay_preserves_existing_profile_when_toml_omits_profile() {
        let mut config = VoiceQualityConfig::for_profile(QualityProfile::Noisy);
        config
            .apply_toml_str(
                r#"
            [voice_quality.endpoint]
            trailing_silence_ms = 700
            "#,
            )
            .expect("toml overlay parses");
        assert_eq!(config.profile, QualityProfile::Noisy);
        assert_eq!(config.speech.rms_threshold, 260.0);
        assert_eq!(config.endpoint.trailing_silence_ms, 700);
    }
}
