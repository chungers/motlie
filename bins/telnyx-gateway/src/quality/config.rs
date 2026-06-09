use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
            rms_threshold: 180.0,
            peak_threshold: 900,
            onset_min_silence_ms: 120,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EndpointQualityConfig {
    pub trailing_silence_ms: u64,
    pub min_turn_words: usize,
    pub min_turn_chars: usize,
    pub merge_window_ms: u64,
    pub max_turn_words: usize,
    pub max_turn_duration_ms: u64,
}

impl Default for EndpointQualityConfig {
    fn default() -> Self {
        Self {
            trailing_silence_ms: 800,
            min_turn_words: 2,
            min_turn_chars: 6,
            merge_window_ms: 350,
            max_turn_words: 80,
            max_turn_duration_ms: 12_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AsrQualityConfig {
    pub repeated_token_run_threshold: usize,
    pub repeated_q_run_threshold: usize,
}

impl Default for AsrQualityConfig {
    fn default() -> Self {
        Self {
            repeated_token_run_threshold: 16,
            repeated_q_run_threshold: 8,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BargeInQualityConfig {
    pub enabled: bool,
    pub speech_onset_cancel_enabled: bool,
    pub partial_asr_cancel_enabled: bool,
    pub final_asr_cancel_enabled: bool,
    pub clear_timeout_ms: u64,
}

impl Default for BargeInQualityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            speech_onset_cancel_enabled: true,
            partial_asr_cancel_enabled: true,
            final_asr_cancel_enabled: true,
            clear_timeout_ms: 1_000,
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
    pub barge_in: BargeInQualityConfig,
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
            barge_in: BargeInQualityConfig::default(),
            logging: LoggingQualityConfig::default(),
            quality_judge: QualityJudgeConfig::default(),
            targets: QualityTargetsConfig::default(),
        };
        match profile {
            QualityProfile::Fast => {
                config.endpoint.trailing_silence_ms = 550;
            }
            QualityProfile::Balanced => {}
            QualityProfile::Complete => {
                config.endpoint.trailing_silence_ms = 1_100;
            }
            QualityProfile::Noisy => {
                config.endpoint.trailing_silence_ms = 950;
                config.speech.rms_threshold = 220.0;
                config.speech.peak_threshold = 1_200;
            }
        }
        config
    }

    pub fn config_id(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("voice quality config serializes");
        let digest = Sha256::digest(encoded);
        format!("cfg_{}", hex::encode(digest))
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
                self.text_call.latest_response_wins = value;
            }
            if let Some(value) = text_call.callback_timeout_ms {
                self.set_text_call_callback_timeout_ms(value);
            }
        }
        if let Some(barge_in) = patch.barge_in {
            if let Some(value) = barge_in.enabled {
                self.barge_in.enabled = value;
            }
            if let Some(value) = barge_in.speech_onset_cancel_enabled {
                self.barge_in.speech_onset_cancel_enabled = value;
            }
            if let Some(value) = barge_in.partial_asr_cancel_enabled {
                self.barge_in.partial_asr_cancel_enabled = value;
            }
            if let Some(value) = barge_in.final_asr_cancel_enabled {
                self.barge_in.final_asr_cancel_enabled = value;
            }
            if let Some(value) = barge_in.clear_timeout_ms {
                self.set_barge_in_clear_timeout_ms(value);
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
            ApplyBoundary::ReportOnly,
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
            ApplyBoundary::NewPlaybackRequest,
            clamped.clamped,
        )
    }

    pub fn set_text_call_playback_wait_timeout_ms(&mut self, value: u64) -> QualityMutationOutcome {
        let clamped = clamp_u64(value, 1_000, 600_000);
        self.text_call.playback_wait_timeout_ms = clamped.value;
        self.outcome(
            "text_call.playback_wait_timeout_ms",
            clamped.value,
            ApplyBoundary::NewPlaybackRequest,
            clamped.clamped,
        )
    }

    pub fn set_text_call_latest_response_wins(&mut self, value: bool) -> QualityMutationOutcome {
        self.text_call.latest_response_wins = value;
        self.outcome(
            "text_call.latest_response_wins",
            value,
            ApplyBoundary::NewTurn,
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

    pub fn set_barge_in_enabled(&mut self, value: bool) -> QualityMutationOutcome {
        self.barge_in.enabled = value;
        self.outcome(
            "barge_in.enabled",
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

#[derive(Clone, Debug, Default, Deserialize)]
struct QualityConfigDocument {
    #[serde(default)]
    voice_quality: Option<QualityConfigPatch>,
}

#[derive(Clone, Debug, Default, Deserialize)]
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
    pub barge_in: Option<BargeInQualityConfigPatch>,
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
pub struct SpeechQualityConfigPatch {
    pub rms_threshold: Option<f32>,
    pub peak_threshold: Option<i32>,
    pub onset_min_silence_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct EndpointQualityConfigPatch {
    pub trailing_silence_ms: Option<u64>,
    pub min_turn_words: Option<usize>,
    pub min_turn_chars: Option<usize>,
    pub merge_window_ms: Option<u64>,
    pub max_turn_words: Option<usize>,
    pub max_turn_duration_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AsrQualityConfigPatch {
    pub repeated_token_run_threshold: Option<usize>,
    pub repeated_q_run_threshold: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TextCallQualityConfigPatch {
    pub max_active_turns: Option<usize>,
    pub media_ready_timeout_ms: Option<u64>,
    pub playback_wait_timeout_ms: Option<u64>,
    pub latest_response_wins: Option<bool>,
    pub callback_timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct BargeInQualityConfigPatch {
    pub enabled: Option<bool>,
    pub speech_onset_cancel_enabled: Option<bool>,
    pub partial_asr_cancel_enabled: Option<bool>,
    pub final_asr_cancel_enabled: Option<bool>,
    pub clear_timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct LoggingQualityConfigPatch {
    pub enabled: Option<bool>,
    pub queue_capacity: Option<usize>,
    pub per_frame_sample_rate: Option<f32>,
    pub include_transcript_text: Option<bool>,
    pub redaction_mode: Option<RedactionMode>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct QualityJudgeConfigPatch {
    pub enabled: Option<bool>,
    pub mode: Option<JudgeMode>,
    pub sample_rate: Option<f32>,
    pub model: Option<String>,
    pub batch_size: Option<usize>,
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
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
    fn report_only_knobs_do_not_get_live_apply_boundary() {
        let mut config = VoiceQualityConfig::default();
        let outcome = config.set_endpoint_merge_window_ms(9_999);
        assert_eq!(config.endpoint.merge_window_ms, 5_000);
        assert_eq!(outcome.apply_boundary, ApplyBoundary::ReportOnly);
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
        assert_eq!(config.speech.rms_threshold, 220.0);
        assert_eq!(config.endpoint.trailing_silence_ms, 100);
        assert!(!config.logging.include_transcript_text);
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
        assert_eq!(config.speech.rms_threshold, 220.0);
        assert_eq!(config.endpoint.trailing_silence_ms, 700);
    }
}
