use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::adapter::LiveAsrBackend;
use crate::call_control::TelnyxMediaConfig;
use crate::processors::ConversationProcessorKind;
use crate::quality::{
    ActiveAsrQualitySession, CallerTurnEventMetadata, QualityEvent, QualityEventContext,
    QualityEventSink, RedactionMode, TtsGenerationMode, TtsQualityConfig, VoiceQualityConfig,
};
use crate::tts::LiveTtsBackend;

pub type SharedState = Arc<RwLock<GatewayState>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InboundMode {
    #[default]
    Disabled,
    Manual,
    AutoTranscribe,
}

impl InboundMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Manual => "manual",
            Self::AutoTranscribe => "auto-transcribe",
        }
    }
}

#[derive(Clone, Debug)]
pub struct QualityRuntimeState {
    pub run_id: String,
    pub config: VoiceQualityConfig,
    pub config_id: String,
    pub event_sink: QualityEventSink,
    pub log_path: Option<PathBuf>,
    pub event_sequence: u64,
}

impl Default for QualityRuntimeState {
    fn default() -> Self {
        let config = VoiceQualityConfig::default();
        let config_id = config.config_id();
        Self {
            run_id: format!("run_{}", Uuid::new_v4().simple()),
            config,
            config_id,
            event_sink: QualityEventSink::disabled(),
            log_path: None,
            event_sequence: 0,
        }
    }
}

pub struct QualitySpanEmission {
    pub config_id: String,
    pub redaction_mode: RedactionMode,
    pub span_name: &'static str,
    pub category: &'static str,
    pub duration: Duration,
    pub critical_path: bool,
    pub concurrent: bool,
    pub payload: Map<String, Value>,
}

impl QualityRuntimeState {
    pub fn set_config(&mut self, config: VoiceQualityConfig) -> String {
        self.config = config;
        self.config_id = self.config.config_id();
        self.config_id.clone()
    }

    fn next_sequence(&mut self) -> u64 {
        self.event_sequence = self.event_sequence.saturating_add(1);
        self.event_sequence
    }
}

#[derive(Clone, Debug, Default)]
pub struct GatewayConfig {
    pub bind: Option<SocketAddr>,
    pub tui: bool,
    pub socket: Option<PathBuf>,
    pub artifact_root: Option<PathBuf>,
    pub log_file: Option<PathBuf>,
    pub public_webhook_url: Option<String>,
    pub public_media_url: Option<String>,
    pub telnyx_media: TelnyxMediaConfig,
    pub capture_dir: Option<PathBuf>,
    pub telnyx_api_base: String,
    pub telnyx_api_key_ref: Option<String>,
    pub dry_run_telnyx: bool,
    pub selected_connection_id: Option<String>,
    pub selected_application_name: Option<String>,
    pub selected_phone_number: Option<String>,
    pub default_from_number: Option<String>,
    pub state_path: Option<PathBuf>,
    pub conversation_enabled: bool,
    pub conversation_final_coalescing_enabled: bool,
    pub conversation_barge_in_enabled: bool,
    pub conversation_processor: ConversationProcessorKind,
    pub startup_warm_models: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InboundSubscription {
    pub id: String,
    pub phone_number: String,
    pub normalized_phone_number: String,
    pub callback_url: String,
    pub priority: i32,
    pub secret_ref: Option<String>,
    pub enabled: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpsertInboundSubscription {
    pub id: String,
    pub phone_number: String,
    pub callback_url: String,
    pub priority: i32,
    pub secret_ref: Option<String>,
    pub enabled: bool,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelnyxIds {
    pub call_control_id: String,
    pub call_session_id: Option<String>,
    pub call_leg_id: Option<String>,
    pub stream_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallDirection {
    Inbound,
    Outbound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallStatus {
    PendingInbound,
    IgnoredInbound,
    Dialing,
    Answering,
    Answered,
    MediaStarted,
    Transcribing,
    Speaking,
    Ended,
    Failed,
}

impl CallStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::PendingInbound => "waiting",
            Self::IgnoredInbound => "disabled",
            Self::Dialing => "dialing",
            Self::Answering => "answering",
            Self::Answered => "answered",
            Self::MediaStarted => "media",
            Self::Transcribing => "transcribing",
            Self::Speaking => "speaking",
            Self::Ended => "ended",
            Self::Failed => "failed",
        }
    }

    pub fn allows_media_start(self) -> bool {
        matches!(self, Self::Dialing | Self::Answering | Self::Answered)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TtsPlaybackStatus {
    Queued,
    Playing,
    MarkSent,
    Completed,
    Canceling,
    Canceled,
    Failed,
}

impl TtsPlaybackStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Playing => "playing",
            Self::MarkSent => "mark-sent",
            Self::Completed => "completed",
            Self::Canceling => "canceling",
            Self::Canceled => "canceled",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TtsPlaybackState {
    pub playback_id: String,
    pub status: TtsPlaybackStatus,
    pub backend: LiveTtsBackend,
    pub text_preview: String,
    pub echo_signature: String,
    pub source_asr_session_ids: Vec<String>,
    pub source_utterance_ids: Vec<String>,
    pub frames_queued: usize,
    pub frames_sent: usize,
    pub underrun_ticks: usize,
    pub pre_audio_wait_ticks: usize,
    pub first_audio_latency_ms: Option<u64>,
    pub mark_name: Option<String>,
    pub error: Option<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QualityPlaybackMetadata {
    pub source: Option<String>,
    pub source_channel: Option<String>,
    pub manual_injection: bool,
    pub related_turn_id: Option<String>,
}

impl QualityPlaybackMetadata {
    pub fn operator_speak(source_channel: impl Into<String>) -> Self {
        Self {
            source: Some("operator_speak".to_string()),
            source_channel: Some(source_channel.into()),
            manual_injection: true,
            related_turn_id: None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QualityPlaybackLinkage {
    pub turn_id: Option<String>,
    pub coalesced_turn_ids: Vec<String>,
    pub source_asr_session_ids: Vec<String>,
    pub source_utterance_ids: Vec<String>,
    pub source_label: String,
    pub metadata: QualityPlaybackMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QualityPlaybackRecord {
    playback_id: String,
    turn_id: Option<String>,
    source_turn_ids: Vec<String>,
    source_asr_session_ids: Vec<String>,
    source_utterance_ids: Vec<String>,
    tts_backend: LiveTtsBackend,
    source_label: String,
    metadata: QualityPlaybackMetadata,
    first_audio_sent: bool,
    terminal_status: Option<String>,
    terminal_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QualityTurnReportState {
    raw_asr_final_events: usize,
    caller_turn_ids: BTreeSet<String>,
    coalesced_selected_turn_ids: BTreeSet<String>,
    coalesced_source_turn_ids: BTreeSet<String>,
    response_attempted_turn_ids: BTreeSet<String>,
    response_attempted_playback_ids: BTreeSet<String>,
    played_turn_ids: BTreeSet<String>,
    played_playback_ids: BTreeSet<String>,
    terminal_completed_playback_ids: BTreeSet<String>,
    terminal_canceled_playback_ids: BTreeSet<String>,
    terminal_failed_playback_ids: BTreeSet<String>,
    canceled_after_call_end_turn_ids: BTreeSet<String>,
    canceled_after_call_end_playback_ids: BTreeSet<String>,
    playback_records: BTreeMap<String, QualityPlaybackRecord>,
}

impl QualityTurnReportState {
    fn record_raw_final(&mut self) {
        self.raw_asr_final_events = self.raw_asr_final_events.saturating_add(1);
    }

    fn record_caller_turn(&mut self, turn_id: &str, source_turn_ids: &[String]) {
        self.caller_turn_ids.insert(turn_id.to_string());
        if source_turn_ids.len() > 1 {
            self.coalesced_selected_turn_ids.insert(turn_id.to_string());
            self.coalesced_source_turn_ids
                .extend(source_turn_ids.iter().cloned());
        }
    }

    fn record_playback_queued(
        &mut self,
        playback_id: String,
        tts_backend: LiveTtsBackend,
        linkage: QualityPlaybackLinkage,
    ) -> QualityPlaybackRecord {
        let source_turn_ids =
            normalized_source_turn_ids(linkage.turn_id.as_deref(), linkage.coalesced_turn_ids);
        if let Some(turn_id) = linkage.turn_id.as_ref() {
            self.response_attempted_turn_ids.insert(turn_id.clone());
        }
        self.response_attempted_playback_ids
            .insert(playback_id.clone());
        let record = QualityPlaybackRecord {
            playback_id: playback_id.clone(),
            turn_id: linkage.turn_id,
            source_turn_ids,
            source_asr_session_ids: linkage.source_asr_session_ids,
            source_utterance_ids: linkage.source_utterance_ids,
            tts_backend,
            source_label: linkage.source_label,
            metadata: linkage.metadata,
            first_audio_sent: false,
            terminal_status: None,
            terminal_reason: None,
        };
        self.playback_records.insert(playback_id, record.clone());
        record
    }

    fn record_first_audio(&mut self, playback_id: &str) -> Option<QualityPlaybackRecord> {
        let record = self.playback_records.get_mut(playback_id)?;
        if record.first_audio_sent {
            return None;
        }
        record.first_audio_sent = true;
        if let Some(turn_id) = record.turn_id.as_ref() {
            self.played_turn_ids.insert(turn_id.clone());
        }
        self.played_playback_ids.insert(playback_id.to_string());
        Some(record.clone())
    }

    fn record_terminal(
        &mut self,
        playback_id: &str,
        status: &'static str,
        reason: Option<&str>,
    ) -> Option<QualityPlaybackRecord> {
        let record = self.playback_records.get_mut(playback_id)?;
        if record.terminal_status.is_some() {
            return None;
        }
        record.terminal_status = Some(status.to_string());
        record.terminal_reason = reason.map(str::to_string);
        match status {
            "completed" => {
                self.terminal_completed_playback_ids
                    .insert(playback_id.to_string());
            }
            "canceled" => {
                self.terminal_canceled_playback_ids
                    .insert(playback_id.to_string());
            }
            "failed" => {
                self.terminal_failed_playback_ids
                    .insert(playback_id.to_string());
            }
            _ => {}
        }
        if reason == Some("canceled_after_call_end") {
            self.canceled_after_call_end_playback_ids
                .insert(playback_id.to_string());
            if let Some(turn_id) = record.turn_id.as_ref() {
                self.canceled_after_call_end_turn_ids
                    .insert(turn_id.clone());
            }
        }
        Some(record.clone())
    }

    fn summary_payload(&self, summary_reason: &str, call: &CallSession) -> Map<String, Value> {
        let turns_without_playback = self
            .caller_turn_ids
            .difference(&self.response_attempted_turn_ids)
            .count();
        json_map(serde_json::json!({
            "summary_reason": summary_reason,
            "call_status": call.status.label(),
            "terminal_reason": call.terminal_reason,
            "raw_asr_final_events": self.raw_asr_final_events,
            "caller_turns": self.caller_turn_ids.len(),
            "attempted_turns": self.response_attempted_turn_ids.len(),
            "attempted_playbacks": self.response_attempted_playback_ids.len(),
            "coalesced_turns": self.coalesced_source_turn_ids.len(),
            "coalesced_response_turns": self.coalesced_selected_turn_ids.len(),
            "played_turns": self.played_turn_ids.len(),
            "played_playbacks": self.played_playback_ids.len(),
            "completed_playbacks": self.terminal_completed_playback_ids.len(),
            "canceled_playbacks": self.terminal_canceled_playback_ids.len(),
            "failed_playbacks": self.terminal_failed_playback_ids.len(),
            "canceled_after_call_end_turns": self.canceled_after_call_end_turn_ids.len(),
            "canceled_after_call_end_playbacks": self.canceled_after_call_end_playback_ids.len(),
            "excluded_turns_without_playback": turns_without_playback,
        }))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConversationMode {
    #[default]
    Manual,
    Auto,
}

impl ConversationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Auto => "auto",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConversationStatus {
    #[default]
    Idle,
    Thinking,
    Proposed,
    Speaking,
    Interrupted,
    Failed,
}

impl ConversationStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Thinking => "thinking",
            Self::Proposed => "proposed",
            Self::Speaking => "speaking",
            Self::Interrupted => "interrupted",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversationRole {
    User,
    Assistant,
    System,
}

impl ConversationRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConversationLine {
    pub at: DateTime<Utc>,
    pub role: ConversationRole,
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConversationState {
    pub attached: bool,
    pub mode: ConversationMode,
    pub processor: ConversationProcessorKind,
    pub status: ConversationStatus,
    pub lines: Vec<ConversationLine>,
    pub last_user_text: Option<String>,
    pub last_assistant_text: Option<String>,
    pub last_playback_id: Option<String>,
    pub last_error: Option<String>,
    pub updated_at: DateTime<Utc>,
}

impl Default for ConversationState {
    fn default() -> Self {
        Self {
            attached: false,
            mode: ConversationMode::Manual,
            processor: ConversationProcessorKind::default(),
            status: ConversationStatus::Idle,
            lines: Vec::new(),
            last_user_text: None,
            last_assistant_text: None,
            last_playback_id: None,
            last_error: None,
            updated_at: Utc::now(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeechOutputConfig {
    pub tts_backend: LiveTtsBackend,
    pub tts_generation_mode: TtsGenerationMode,
    pub tts_chunking_enabled: bool,
    pub tts_max_text_chunk_chars: usize,
    pub tts_first_chunk_max_chars: usize,
    pub tts_prebuffer_chunks: usize,
}

impl SpeechOutputConfig {
    pub fn from_quality(tts_backend: LiveTtsBackend, tts: &TtsQualityConfig) -> Self {
        Self {
            tts_backend,
            tts_generation_mode: tts.generation_mode,
            tts_chunking_enabled: tts.chunking_enabled,
            tts_max_text_chunk_chars: tts.max_text_chunk_chars,
            tts_first_chunk_max_chars: tts.first_chunk_max_chars,
            tts_prebuffer_chunks: tts.prebuffer_chunks,
        }
    }
}

impl Default for SpeechOutputConfig {
    fn default() -> Self {
        Self::from_quality(LiveTtsBackend::default(), &TtsQualityConfig::default())
    }
}

impl ConversationState {
    pub fn status_label(&self) -> &'static str {
        if self.attached {
            self.status.label()
        } else {
            "detached"
        }
    }

    fn push_line(&mut self, role: ConversationRole, text: String) {
        self.lines.push(ConversationLine {
            at: Utc::now(),
            role,
            text,
        });
        if self.lines.len() > 40 {
            let remove_count = self.lines.len().saturating_sub(40);
            self.lines.drain(0..remove_count);
        }
        self.updated_at = Utc::now();
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MediaMetadata {
    pub stream_id: Option<String>,
    pub encoding: Option<String>,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub track: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TranscriptKind {
    Partial,
    Final,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptLine {
    pub at: DateTime<Utc>,
    pub kind: TranscriptKind,
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimelineEntry {
    pub at: DateTime<Utc>,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallSession {
    pub gateway_call_id: String,
    pub direction: CallDirection,
    pub status: CallStatus,
    pub ids: TelnyxIds,
    pub from: Option<String>,
    pub to: Option<String>,
    pub media: MediaMetadata,
    pub transcripts: Vec<TranscriptLine>,
    pub final_transcript: String,
    pub current_partial: Option<String>,
    pub timeline: Vec<TimelineEntry>,
    pub unread_events: u32,
    pub last_error: Option<String>,
    pub terminal_reason: Option<String>,
    pub asr_backend: Option<LiveAsrBackend>,
    pub tts: Option<TtsPlaybackState>,
    pub echo_suppressed_transcripts: usize,
    pub last_echo_suppressed_preview: Option<String>,
    pub conversation: ConversationState,
    pub speech_output: SpeechOutputConfig,
    pub quality_turns: QualityTurnReportState,
}

impl CallSession {
    pub fn new(
        direction: CallDirection,
        ids: TelnyxIds,
        from: Option<String>,
        to: Option<String>,
        status: CallStatus,
    ) -> Self {
        let mut call = Self {
            gateway_call_id: format!("gwc_{}", Uuid::new_v4().simple()),
            direction,
            status,
            ids,
            from,
            to,
            media: MediaMetadata::default(),
            transcripts: Vec::new(),
            final_transcript: String::new(),
            current_partial: None,
            timeline: Vec::new(),
            unread_events: 0,
            last_error: None,
            terminal_reason: None,
            asr_backend: None,
            tts: None,
            echo_suppressed_transcripts: 0,
            last_echo_suppressed_preview: None,
            conversation: ConversationState::default(),
            speech_output: SpeechOutputConfig::default(),
            quality_turns: QualityTurnReportState::default(),
        };
        call.push_timeline("call created");
        call
    }

    pub fn pending_inbound(
        ids: TelnyxIds,
        from: Option<String>,
        to: Option<String>,
        status: CallStatus,
    ) -> Self {
        Self::new(CallDirection::Inbound, ids, from, to, status)
    }

    pub fn outbound(
        ids: TelnyxIds,
        from: Option<String>,
        to: Option<String>,
        status: CallStatus,
    ) -> Self {
        Self::new(CallDirection::Outbound, ids, from, to, status)
    }

    pub fn push_timeline(&mut self, message: impl Into<String>) {
        self.unread_events = self.unread_events.saturating_add(1);
        self.timeline.push(TimelineEntry {
            at: Utc::now(),
            message: message.into(),
        });
    }

    pub fn updated_at(&self) -> Option<DateTime<Utc>> {
        self.timeline.last().map(|entry| entry.at)
    }

    pub fn assembled_transcript_text(&self) -> String {
        match (
            self.final_transcript.trim(),
            self.current_partial.as_deref().map(str::trim),
        ) {
            ("", Some(partial)) if !partial.is_empty() => partial.to_string(),
            (final_text, Some(partial)) if !final_text.is_empty() && !partial.is_empty() => {
                format!("{final_text} {partial}")
            }
            (final_text, _) if !final_text.is_empty() => final_text.to_string(),
            _ => "<none>".to_string(),
        }
    }

    pub fn tts_status_label(&self) -> &'static str {
        self.tts
            .as_ref()
            .map(|tts| tts.status.label())
            .unwrap_or("idle")
    }

    pub fn conversation_status_label(&self) -> &'static str {
        self.conversation.status_label()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogEntry {
    pub at: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWarmStatus {
    pub label: String,
    pub model: String,
    pub warmed_at: DateTime<Utc>,
    pub elapsed_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GatewayState {
    pub config: GatewayConfig,
    pub inbound_mode: InboundMode,
    pub started_at: DateTime<Utc>,
    pub conversation_tts_backend: LiveTtsBackend,
    pub calls: BTreeMap<String, CallSession>,
    pub call_control_index: BTreeMap<String, String>,
    pub stream_index: BTreeMap<String, String>,
    pub inbound_subscriptions: BTreeMap<String, InboundSubscription>,
    pub logs: VecDeque<LogEntry>,
    pub model_warmups: BTreeMap<String, ModelWarmStatus>,
    pub quality: QualityRuntimeState,
    pub shutdown_requested: bool,
}

pub fn asr_warm_key(backend: LiveAsrBackend) -> String {
    format!("asr:{}", backend.label())
}

pub fn tts_warm_key(backend: LiveTtsBackend) -> String {
    format!("tts:{}", backend.label())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamAttachOutcome {
    Attached {
        gateway_call_id: String,
        asr_backend: LiveAsrBackend,
    },
    NotAnswered {
        gateway_call_id: String,
        status: CallStatus,
    },
    UnknownCallControl,
}

impl GatewayState {
    pub fn new(bind: SocketAddr) -> Self {
        Self {
            config: GatewayConfig {
                bind: Some(bind),
                telnyx_media: TelnyxMediaConfig::default(),
                telnyx_api_base: "https://api.telnyx.com/v2".to_string(),
                telnyx_api_key_ref: Some("env:TELNYX_API_KEY".to_string()),
                conversation_barge_in_enabled: true,
                conversation_processor: ConversationProcessorKind::Identity,
                ..GatewayConfig::default()
            },
            inbound_mode: InboundMode::Disabled,
            started_at: Utc::now(),
            conversation_tts_backend: LiveTtsBackend::default(),
            calls: BTreeMap::new(),
            call_control_index: BTreeMap::new(),
            stream_index: BTreeMap::new(),
            inbound_subscriptions: BTreeMap::new(),
            logs: VecDeque::new(),
            model_warmups: BTreeMap::new(),
            quality: QualityRuntimeState::default(),
            shutdown_requested: false,
        }
    }

    pub fn mark_model_warm(
        &mut self,
        key: String,
        label: impl Into<String>,
        model: impl Into<String>,
        elapsed_ms: u64,
    ) {
        self.model_warmups.insert(
            key,
            ModelWarmStatus {
                label: label.into(),
                model: model.into(),
                warmed_at: Utc::now(),
                elapsed_ms,
            },
        );
    }

    pub fn set_quality_config(&mut self, config: VoiceQualityConfig) -> String {
        self.quality.set_config(config)
    }

    pub fn set_quality_event_sink(&mut self, sink: QualityEventSink, log_path: Option<PathBuf>) {
        self.quality.event_sink = sink;
        self.quality.log_path = log_path;
    }

    fn quality_event_context(&mut self, gateway_call_id: Option<String>) -> QualityEventContext {
        self.quality_event_context_with_config(gateway_call_id, self.quality.config_id.clone())
    }

    fn quality_event_context_with_config(
        &mut self,
        gateway_call_id: Option<String>,
        config_id: String,
    ) -> QualityEventContext {
        self.quality_event_context_with_config_and_redaction(
            gateway_call_id,
            config_id,
            self.quality.config.logging.redaction_mode,
        )
    }

    fn quality_event_context_with_config_and_redaction(
        &mut self,
        gateway_call_id: Option<String>,
        config_id: String,
        redaction_mode: crate::quality::RedactionMode,
    ) -> QualityEventContext {
        QualityEventContext::new(
            self.quality.next_sequence(),
            self.quality.run_id.clone(),
            gateway_call_id,
            config_id,
            redaction_mode,
        )
    }

    pub fn emit_quality_config_snapshot(
        &mut self,
        gateway_call_id: &str,
        snapshot_reason: &'static str,
        effective_scope: &'static str,
        effective_after_asr_session_id: Option<String>,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::config_snapshot(
            self.quality_event_context(Some(gateway_call_id.to_string())),
            &self.quality.config,
            snapshot_reason,
            effective_scope,
            effective_after_asr_session_id,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn start_quality_asr_session(
        &mut self,
        gateway_call_id: &str,
        stream_id: Option<&str>,
        reason: &'static str,
    ) -> ActiveAsrQualitySession {
        let session = ActiveAsrQualitySession::new(&self.quality.config);
        if self.quality.event_sink.is_enabled() {
            let event = QualityEvent::asr_session_started(
                self.quality_event_context_with_config_and_redaction(
                    Some(gateway_call_id.to_string()),
                    session.config_id.clone(),
                    session.redaction_mode,
                ),
                &session,
                stream_id,
                reason,
            );
            self.quality.event_sink.emit(event);
        }
        session
    }

    pub fn emit_quality_asr_turn_mapped(
        &mut self,
        gateway_call_id: &str,
        session: &ActiveAsrQualitySession,
        turn_id: &str,
        final_transcript_event_id: &str,
        caller_turn_sent: bool,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::asr_turn_mapped(
            self.quality_event_context_with_config_and_redaction(
                Some(gateway_call_id.to_string()),
                session.config_id.clone(),
                session.redaction_mode,
            ),
            session,
            turn_id.to_string(),
            final_transcript_event_id.to_string(),
            caller_turn_sent,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_caller_turn_sent(
        &mut self,
        gateway_call_id: &str,
        turn_id: &str,
        text: &str,
        session: Option<&ActiveAsrQualitySession>,
        mut metadata: CallerTurnEventMetadata,
    ) {
        let source_turn_ids =
            normalized_source_turn_ids(Some(turn_id), metadata.coalesced_turn_ids.clone());
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.quality_turns
                .record_caller_turn(turn_id, &source_turn_ids);
        }
        metadata.coalesced_turn_ids = source_turn_ids;
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let (context, include_transcript_text) = if let Some(session) = session {
            if metadata.asr_session_id.is_none() {
                metadata.asr_session_id = Some(session.asr_session_id.clone());
            }
            if metadata.asr_session_ids.is_empty() {
                metadata
                    .asr_session_ids
                    .push(session.asr_session_id.clone());
            }
            if metadata.utterance_id.is_none() {
                metadata.utterance_id = Some(session.utterance_id.clone());
            }
            if metadata.utterance_ids.is_empty() {
                metadata.utterance_ids.push(session.utterance_id.clone());
            }
            (
                self.quality_event_context_with_config_and_redaction(
                    Some(gateway_call_id.to_string()),
                    session.config_id.clone(),
                    session.redaction_mode,
                ),
                session.include_transcript_text,
            )
        } else {
            (
                self.quality_event_context(Some(gateway_call_id.to_string())),
                self.quality.config.logging.include_transcript_text,
            )
        };
        let event = QualityEvent::caller_turn_sent(
            context,
            turn_id.to_string(),
            text,
            include_transcript_text,
            metadata,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_caller_partial_sent(
        &mut self,
        gateway_call_id: &str,
        session: &ActiveAsrQualitySession,
        text: &str,
        confidence: Option<f32>,
        stability: Option<f32>,
        speech_state: &'static str,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::caller_partial_sent(
            self.quality_event_context_with_config_and_redaction(
                Some(gateway_call_id.to_string()),
                session.config_id.clone(),
                session.redaction_mode,
            ),
            session,
            text,
            confidence,
            stability,
            speech_state,
            session.include_transcript_text,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_transcript_suppressed(
        &mut self,
        gateway_call_id: &str,
        session: Option<&ActiveAsrQualitySession>,
        transcript_kind: &'static str,
        suppression_reason: &'static str,
        text: &str,
        extra: Map<String, Value>,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let (context, include_transcript_text) = if let Some(session) = session {
            (
                self.quality_event_context_with_config_and_redaction(
                    Some(gateway_call_id.to_string()),
                    session.config_id.clone(),
                    session.redaction_mode,
                ),
                session.include_transcript_text,
            )
        } else {
            (
                self.quality_event_context(Some(gateway_call_id.to_string())),
                self.quality.config.logging.include_transcript_text,
            )
        };
        let event = QualityEvent::transcript_suppressed(
            context,
            session,
            transcript_kind,
            suppression_reason,
            text,
            include_transcript_text,
            extra,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_span_finished(&mut self, gateway_call_id: &str, span: QualitySpanEmission) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::span_finished(
            self.quality_event_context_with_config_and_redaction(
                Some(gateway_call_id.to_string()),
                span.config_id,
                span.redaction_mode,
            ),
            span.span_name,
            span.category,
            span.duration,
            span.critical_path,
            span.concurrent,
            span.payload,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_turn_batch_lifecycle(
        &mut self,
        gateway_call_id: &str,
        lifecycle_event: &'static str,
        payload: Map<String, Value>,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::turn_batch_lifecycle(
            self.quality_event_context(Some(gateway_call_id.to_string())),
            lifecycle_event,
            payload,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_inbound_transport_rollup(
        &mut self,
        gateway_call_id: &str,
        config_id: String,
        redaction_mode: RedactionMode,
        payload: Map<String, Value>,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::inbound_transport_rollup(
            self.quality_event_context_with_config_and_redaction(
                Some(gateway_call_id.to_string()),
                config_id,
                redaction_mode,
            ),
            payload,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn emit_quality_outbound_pacing_rollup(
        &mut self,
        gateway_call_id: &str,
        config_id: String,
        redaction_mode: RedactionMode,
        payload: Map<String, Value>,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::outbound_pacing_rollup(
            self.quality_event_context_with_config_and_redaction(
                Some(gateway_call_id.to_string()),
                config_id,
                redaction_mode,
            ),
            payload,
        );
        self.quality.event_sink.emit(event);
    }

    pub fn record_quality_playback_terminal(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        status: &'static str,
        reason: Option<&str>,
    ) {
        let record = self.calls.get_mut(gateway_call_id).and_then(|call| {
            call.quality_turns
                .record_terminal(playback_id, status, reason)
        });
        if let Some(record) = record {
            self.emit_quality_turn_playback_link(
                gateway_call_id,
                "terminal",
                &record,
                None,
                record.terminal_status.as_deref(),
                record.terminal_reason.as_deref(),
            );
            self.emit_quality_report_summary(gateway_call_id, "playback_terminal");
        }
    }

    pub fn emit_quality_report_summary(&mut self, gateway_call_id: &str, summary_reason: &str) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let mut payload = {
            let Some(call) = self.calls.get(gateway_call_id) else {
                return;
            };
            call.quality_turns.summary_payload(summary_reason, call)
        };
        payload.insert(
            "dropped_quality_events".to_string(),
            Value::Number(serde_json::Number::from(
                self.quality.event_sink.dropped_count(),
            )),
        );
        let event = QualityEvent::report_summary(
            self.quality_event_context(Some(gateway_call_id.to_string())),
            payload,
        );
        self.quality.event_sink.emit(event);
    }

    fn emit_quality_turn_playback_link(
        &mut self,
        gateway_call_id: &str,
        link_stage: &'static str,
        record: &QualityPlaybackRecord,
        replaced_playback_id: Option<&str>,
        terminal_status: Option<&str>,
        terminal_reason: Option<&str>,
    ) {
        if !self.quality.event_sink.is_enabled() {
            return;
        }
        let event = QualityEvent::turn_playback_linked(
            self.quality_event_context(Some(gateway_call_id.to_string())),
            playback_link_payload(
                link_stage,
                record,
                replaced_playback_id,
                terminal_status,
                terminal_reason,
            ),
        );
        self.quality.event_sink.emit(event);
    }

    pub fn log(&mut self, level: LogLevel, message: impl Into<String>) {
        self.logs.push_back(LogEntry {
            at: Utc::now(),
            level,
            message: message.into(),
        });
        while self.logs.len() > 500 {
            let _ = self.logs.pop_front();
        }
    }

    pub fn add_or_update_inbound_call(
        &mut self,
        ids: TelnyxIds,
        from: Option<String>,
        to: Option<String>,
        status: CallStatus,
    ) -> String {
        if let Some(existing) = self.call_control_index.get(&ids.call_control_id).cloned() {
            if let Some(call) = self.calls.get_mut(&existing) {
                call.status = status;
                call.ids.call_session_id = ids.call_session_id.or(call.ids.call_session_id.clone());
                call.ids.call_leg_id = ids.call_leg_id.or(call.ids.call_leg_id.clone());
                call.push_timeline(format!("call status -> {}", status.label()));
            }
            return existing;
        }

        let mut call = CallSession::pending_inbound(ids.clone(), from, to, status);
        call.conversation.processor = self.config.conversation_processor.clone();
        call.speech_output = SpeechOutputConfig::from_quality(
            self.conversation_tts_backend,
            &self.quality.config.tts,
        );
        let gateway_call_id = call.gateway_call_id.clone();
        self.call_control_index
            .insert(ids.call_control_id, gateway_call_id.clone());
        self.calls.insert(gateway_call_id.clone(), call);
        gateway_call_id
    }

    pub fn add_or_update_outbound_call(
        &mut self,
        ids: TelnyxIds,
        from: Option<String>,
        to: Option<String>,
        status: CallStatus,
    ) -> String {
        if let Some(existing) = self.call_control_index.get(&ids.call_control_id).cloned() {
            if let Some(call) = self.calls.get_mut(&existing) {
                call.status = status;
                call.ids.call_session_id = ids.call_session_id.or(call.ids.call_session_id.clone());
                call.ids.call_leg_id = ids.call_leg_id.or(call.ids.call_leg_id.clone());
                if from.is_some() {
                    call.from = from;
                }
                if to.is_some() {
                    call.to = to;
                }
                call.push_timeline(format!("call status -> {}", status.label()));
            }
            return existing;
        }

        let mut call = CallSession::outbound(ids.clone(), from, to, status);
        call.conversation.processor = self.config.conversation_processor.clone();
        call.speech_output = SpeechOutputConfig::from_quality(
            self.conversation_tts_backend,
            &self.quality.config.tts,
        );
        let gateway_call_id = call.gateway_call_id.clone();
        self.call_control_index
            .insert(ids.call_control_id, gateway_call_id.clone());
        self.calls.insert(gateway_call_id.clone(), call);
        gateway_call_id
    }

    pub fn upsert_inbound_subscription(
        &mut self,
        request: UpsertInboundSubscription,
    ) -> InboundSubscription {
        let now = Utc::now();
        let created_at = self
            .inbound_subscriptions
            .get(&request.id)
            .map(|subscription| subscription.created_at)
            .unwrap_or(now);
        let subscription = InboundSubscription {
            id: request.id,
            normalized_phone_number: normalize_phone_number(&request.phone_number),
            phone_number: request.phone_number,
            callback_url: request.callback_url,
            priority: request.priority,
            secret_ref: request.secret_ref,
            enabled: request.enabled,
            metadata: request.metadata,
            created_at,
            updated_at: now,
        };
        self.inbound_subscriptions
            .insert(subscription.id.clone(), subscription.clone());
        subscription
    }

    pub fn remove_inbound_subscription(&mut self, id: &str) -> Option<InboundSubscription> {
        self.inbound_subscriptions.remove(id)
    }

    pub fn inbound_subscription(&self, id: &str) -> Option<&InboundSubscription> {
        self.inbound_subscriptions.get(id)
    }

    pub fn inbound_subscriptions_for_phone(&self, phone_number: &str) -> Vec<InboundSubscription> {
        let normalized = normalize_phone_number(phone_number);
        let mut subscriptions: Vec<_> = self
            .inbound_subscriptions
            .values()
            .filter(|subscription| {
                subscription.enabled && subscription.normalized_phone_number == normalized
            })
            .cloned()
            .collect();
        subscriptions.sort_by(|left, right| {
            left.priority
                .cmp(&right.priority)
                .then(left.created_at.cmp(&right.created_at))
                .then(left.id.cmp(&right.id))
        });
        subscriptions
    }

    pub fn has_enabled_inbound_subscribers_for_phone(&self, phone_number: Option<&str>) -> bool {
        let Some(phone_number) = phone_number else {
            return false;
        };
        !self
            .inbound_subscriptions_for_phone(phone_number)
            .is_empty()
    }

    pub fn call_by_control_id_mut(&mut self, call_control_id: &str) -> Option<&mut CallSession> {
        let gateway_call_id = self.call_control_index.get(call_control_id)?.clone();
        self.calls.get_mut(&gateway_call_id)
    }

    pub fn gateway_call_id_for_stream(&self, stream_id: &str) -> Option<String> {
        self.stream_index.get(stream_id).cloned()
    }

    pub fn set_call_stream(
        &mut self,
        call_control_id: &str,
        stream_id: String,
        media: MediaMetadata,
    ) -> StreamAttachOutcome {
        let Some(gateway_call_id) = self.call_control_index.get(call_control_id).cloned() else {
            return StreamAttachOutcome::UnknownCallControl;
        };
        let Some(call) = self.calls.get_mut(&gateway_call_id) else {
            return StreamAttachOutcome::UnknownCallControl;
        };
        if !call.status.allows_media_start() {
            return StreamAttachOutcome::NotAnswered {
                gateway_call_id,
                status: call.status,
            };
        }
        let asr_backend = call.asr_backend.unwrap_or_default();
        call.asr_backend = Some(asr_backend);
        call.ids.stream_id = Some(stream_id.clone());
        call.media = media;
        call.status = CallStatus::MediaStarted;
        call.push_timeline("media stream started");
        self.stream_index.insert(stream_id, gateway_call_id.clone());
        StreamAttachOutcome::Attached {
            gateway_call_id,
            asr_backend,
        }
    }

    pub fn add_transcript(&mut self, gateway_call_id: &str, kind: TranscriptKind, text: String) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            if call.status != CallStatus::Speaking {
                call.status = CallStatus::Transcribing;
            }
            call.unread_events = call.unread_events.saturating_add(1);
            match kind {
                TranscriptKind::Partial => call.current_partial = Some(text.clone()),
                TranscriptKind::Final => {
                    append_transcript_fragment(&mut call.final_transcript, &text);
                    call.current_partial = None;
                    call.quality_turns.record_raw_final();
                }
            }
            call.transcripts.push(TranscriptLine {
                at: Utc::now(),
                kind,
                text,
            });
        }
    }

    pub fn attach_conversation(&mut self, gateway_call_id: &str, mode: ConversationMode) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.attached = true;
            call.conversation.mode = mode;
            call.conversation.status = ConversationStatus::Idle;
            call.conversation.last_error = None;
            call.conversation.updated_at = Utc::now();
            call.push_timeline(format!("conversation attached ({})", mode.label()));
        }
    }

    pub fn detach_conversation(&mut self, gateway_call_id: &str) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.attached = false;
            call.conversation.status = ConversationStatus::Idle;
            call.conversation.updated_at = Utc::now();
            call.push_timeline("conversation detached");
        }
    }

    pub fn set_conversation_mode(&mut self, gateway_call_id: &str, mode: ConversationMode) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.mode = mode;
            call.conversation.attached = true;
            call.conversation.updated_at = Utc::now();
            call.push_timeline(format!("conversation mode -> {}", mode.label()));
        }
    }

    pub fn set_conversation_processor(
        &mut self,
        gateway_call_id: &str,
        processor: ConversationProcessorKind,
    ) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            let label = processor.label();
            call.conversation.processor = processor;
            call.conversation.updated_at = Utc::now();
            call.push_timeline(format!("conversation processor -> {label}"));
        }
    }

    pub fn record_conversation_user_turn(&mut self, gateway_call_id: &str, text: String) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.last_user_text = Some(text.clone());
            call.conversation.status = ConversationStatus::Thinking;
            call.conversation.last_error = None;
            call.conversation.push_line(ConversationRole::User, text);
            call.push_timeline("conversation user turn");
        }
    }

    pub fn record_conversation_proposal(&mut self, gateway_call_id: &str, text: String) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.last_assistant_text = Some(text.clone());
            call.conversation.last_playback_id = None;
            call.conversation.status = ConversationStatus::Proposed;
            call.conversation.last_error = None;
            call.conversation
                .push_line(ConversationRole::Assistant, text);
            call.push_timeline("conversation assistant proposed");
        }
    }

    pub fn record_conversation_speaking(
        &mut self,
        gateway_call_id: &str,
        text: String,
        playback_id: String,
    ) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.last_assistant_text = Some(text.clone());
            call.conversation.last_playback_id = Some(playback_id.clone());
            call.conversation.status = ConversationStatus::Speaking;
            call.conversation.last_error = None;
            call.conversation
                .push_line(ConversationRole::Assistant, text);
            call.push_timeline(format!("conversation assistant speaking {playback_id}"));
        }
    }

    pub fn record_conversation_approved_speaking(
        &mut self,
        gateway_call_id: &str,
        playback_id: String,
    ) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.last_playback_id = Some(playback_id.clone());
            call.conversation.status = ConversationStatus::Speaking;
            call.conversation.last_error = None;
            call.conversation.updated_at = Utc::now();
            call.push_timeline(format!("conversation assistant approved {playback_id}"));
        }
    }

    pub fn record_conversation_interrupted(&mut self, gateway_call_id: &str, playback_id: &str) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.status = ConversationStatus::Interrupted;
            call.conversation.updated_at = Utc::now();
            call.push_timeline(format!("conversation interrupted {playback_id}"));
        }
    }

    pub fn record_conversation_idle(&mut self, gateway_call_id: &str) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.status = ConversationStatus::Idle;
            call.conversation.updated_at = Utc::now();
        }
    }

    pub fn record_conversation_failed(&mut self, gateway_call_id: &str, error: String) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.conversation.status = ConversationStatus::Failed;
            call.conversation.last_error = Some(error.clone());
            call.conversation.updated_at = Utc::now();
            call.push_timeline(format!("conversation failed: {error}"));
        }
    }

    pub fn start_tts_job(
        &mut self,
        gateway_call_id: &str,
        playback_id: String,
        backend: LiveTtsBackend,
        text: &str,
    ) {
        self.start_tts_job_with_linkage(
            gateway_call_id,
            playback_id,
            backend,
            text,
            QualityPlaybackLinkage::default(),
            None,
        );
    }

    pub fn start_tts_job_with_linkage(
        &mut self,
        gateway_call_id: &str,
        playback_id: String,
        backend: LiveTtsBackend,
        text: &str,
        linkage: QualityPlaybackLinkage,
        replaced_playback_id: Option<&str>,
    ) {
        let record = if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.status = CallStatus::Speaking;
            call.tts = Some(TtsPlaybackState {
                playback_id: playback_id.clone(),
                status: TtsPlaybackStatus::Queued,
                backend,
                text_preview: preview_text(text),
                echo_signature: speech_echo_signature(text),
                source_asr_session_ids: linkage.source_asr_session_ids.clone(),
                source_utterance_ids: linkage.source_utterance_ids.clone(),
                frames_queued: 0,
                frames_sent: 0,
                underrun_ticks: 0,
                pre_audio_wait_ticks: 0,
                first_audio_latency_ms: None,
                mark_name: None,
                error: None,
                updated_at: Utc::now(),
            });
            let record =
                call.quality_turns
                    .record_playback_queued(playback_id.clone(), backend, linkage);
            call.push_timeline(format!("tts {playback_id} queued"));
            Some(record)
        } else {
            None
        };
        if let Some(record) = record {
            self.emit_quality_turn_playback_link(
                gateway_call_id,
                "queued",
                &record,
                replaced_playback_id,
                None,
                None,
            );
        }
    }

    pub fn mark_tts_frames_queued(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        frames: usize,
    ) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.frames_queued = tts.frames_queued.saturating_add(frames);
            if tts.status == TtsPlaybackStatus::Queued && frames > 0 {
                tts.status = TtsPlaybackStatus::Playing;
            }
        });
    }

    pub fn mark_tts_frame_sent(&mut self, gateway_call_id: &str, playback_id: &str) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.frames_sent = tts.frames_sent.saturating_add(1);
            if matches!(
                tts.status,
                TtsPlaybackStatus::Queued | TtsPlaybackStatus::Playing
            ) {
                tts.status = TtsPlaybackStatus::Playing;
            }
        });
    }

    pub fn mark_tts_first_audio_latency_and_pacing(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        latency_ms: u64,
        pre_audio_wait_ticks: usize,
        underrun_ticks: usize,
    ) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            if tts.first_audio_latency_ms.is_none() {
                tts.first_audio_latency_ms = Some(latency_ms);
            }
            tts.pre_audio_wait_ticks = tts
                .pre_audio_wait_ticks
                .saturating_add(pre_audio_wait_ticks);
            tts.underrun_ticks = tts.underrun_ticks.saturating_add(underrun_ticks);
        });
        let record = self
            .calls
            .get_mut(gateway_call_id)
            .and_then(|call| call.quality_turns.record_first_audio(playback_id));
        if let Some(record) = record {
            self.emit_quality_turn_playback_link(
                gateway_call_id,
                "first_audio",
                &record,
                None,
                None,
                None,
            );
            self.emit_quality_report_summary(gateway_call_id, "first_audio");
        }
    }

    pub fn mark_tts_pacing_counts(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        pre_audio_wait_ticks: usize,
        underrun_ticks: usize,
    ) {
        if pre_audio_wait_ticks == 0 && underrun_ticks == 0 {
            return;
        }
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.pre_audio_wait_ticks = tts
                .pre_audio_wait_ticks
                .saturating_add(pre_audio_wait_ticks);
            tts.underrun_ticks = tts.underrun_ticks.saturating_add(underrun_ticks);
        });
    }

    pub fn mark_tts_mark_sent(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        mark_name: &str,
    ) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.status = TtsPlaybackStatus::MarkSent;
            tts.mark_name = Some(mark_name.to_string());
        });
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.push_timeline(format!("tts {playback_id} mark sent"));
        }
    }

    pub fn mark_tts_completed(&mut self, gateway_call_id: &str, mark_name: &str) {
        let completed_playback = if let Some(call) = self.calls.get_mut(gateway_call_id) {
            let completed_playback = if let Some(tts) = call
                .tts
                .as_mut()
                .filter(|tts| tts.mark_name.as_deref() == Some(mark_name))
            {
                tts.status = TtsPlaybackStatus::Completed;
                tts.updated_at = Utc::now();
                Some(tts.playback_id.clone())
            } else {
                None
            };
            if let Some(playback_id) = completed_playback.as_ref() {
                if call.status == CallStatus::Speaking {
                    call.status = status_after_tts(call);
                }
                if call.conversation.last_playback_id.as_deref() == Some(playback_id.as_str()) {
                    call.conversation.status = ConversationStatus::Idle;
                    call.conversation.updated_at = Utc::now();
                }
                call.push_timeline(format!("tts {playback_id} completed"));
            }
            completed_playback
        } else {
            None
        };
        if let Some(playback_id) = completed_playback {
            self.record_quality_playback_terminal(gateway_call_id, &playback_id, "completed", None);
        }
    }

    pub fn mark_tts_canceling(&mut self, gateway_call_id: &str, playback_id: &str) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.status = TtsPlaybackStatus::Canceling;
        });
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.push_timeline(format!("tts {playback_id} cancel requested"));
        }
    }

    pub fn mark_tts_canceled(&mut self, gateway_call_id: &str, playback_id: &str) {
        self.mark_tts_canceled_with_reason(gateway_call_id, playback_id, None);
    }

    pub fn mark_tts_canceled_with_reason(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        terminal_reason: Option<&'static str>,
    ) {
        let mut canceled = false;
        self.update_tts(gateway_call_id, playback_id, |tts| {
            if tts.status != TtsPlaybackStatus::Failed {
                tts.status = TtsPlaybackStatus::Canceled;
                canceled = true;
            }
        });
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            if canceled && call.status == CallStatus::Speaking {
                call.status = status_after_tts(call);
            }
            if canceled && call.conversation.last_playback_id.as_deref() == Some(playback_id) {
                call.conversation.status = ConversationStatus::Idle;
                call.conversation.updated_at = Utc::now();
            }
            if canceled {
                call.push_timeline(format!("tts {playback_id} canceled"));
            } else {
                call.push_timeline(format!(
                    "tts {playback_id} clear sent after inactive playback"
                ));
            }
        }
        self.record_quality_playback_terminal(
            gateway_call_id,
            playback_id,
            "canceled",
            terminal_reason,
        );
    }

    pub fn mark_tts_failed(&mut self, gateway_call_id: &str, playback_id: &str, error: String) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.status = TtsPlaybackStatus::Failed;
            tts.error = Some(error.clone());
        });
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.last_error = Some(error.clone());
            if call.status == CallStatus::Speaking {
                call.status = status_after_tts(call);
            }
            if call.conversation.last_playback_id.as_deref() == Some(playback_id) {
                call.conversation.status = ConversationStatus::Failed;
                call.conversation.last_error = Some(error.clone());
                call.conversation.updated_at = Utc::now();
            }
            call.push_timeline(format!("tts {playback_id} failed: {error}"));
        }
        self.record_quality_playback_terminal(
            gateway_call_id,
            playback_id,
            "failed",
            Some("tts_failed"),
        );
    }

    pub fn record_echo_suppressed_transcript(&mut self, gateway_call_id: &str, text: &str) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.echo_suppressed_transcripts = call.echo_suppressed_transcripts.saturating_add(1);
            call.last_echo_suppressed_preview = Some(preview_text(text));
            call.push_timeline("transcript suppressed assistant echo");
        }
    }

    fn update_tts(
        &mut self,
        gateway_call_id: &str,
        playback_id: &str,
        update: impl FnOnce(&mut TtsPlaybackState),
    ) {
        if let Some(tts) = self
            .calls
            .get_mut(gateway_call_id)
            .and_then(|call| call.tts.as_mut())
            .filter(|tts| tts.playback_id == playback_id)
        {
            update(tts);
            tts.updated_at = Utc::now();
        }
    }
}

pub fn speech_echo_signature(text: &str) -> String {
    let mut normalized = String::new();
    let mut previous_was_space = true;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            previous_was_space = false;
        } else if !previous_was_space {
            normalized.push(' ');
            previous_was_space = true;
        }
    }
    normalized.trim().to_string()
}

fn append_transcript_fragment(transcript: &mut String, fragment: &str) {
    let fragment = fragment.trim();
    if fragment.is_empty() {
        return;
    }
    if !transcript.is_empty() {
        transcript.push(' ');
    }
    transcript.push_str(fragment);
}

fn preview_text(text: &str) -> String {
    const MAX_CHARS: usize = 80;
    let mut preview = text.chars().take(MAX_CHARS).collect::<String>();
    if text.chars().count() > MAX_CHARS {
        preview.push_str("...");
    }
    preview
}

fn status_after_tts(call: &CallSession) -> CallStatus {
    if !call.transcripts.is_empty()
        || !call.final_transcript.is_empty()
        || call.current_partial.is_some()
    {
        CallStatus::Transcribing
    } else {
        CallStatus::MediaStarted
    }
}

fn normalized_source_turn_ids(
    turn_id: Option<&str>,
    mut coalesced_turn_ids: Vec<String>,
) -> Vec<String> {
    coalesced_turn_ids.retain(|turn_id| !turn_id.trim().is_empty());
    if coalesced_turn_ids.is_empty() {
        if let Some(turn_id) = turn_id {
            coalesced_turn_ids.push(turn_id.to_string());
        }
    } else if let Some(turn_id) = turn_id {
        if !coalesced_turn_ids
            .iter()
            .any(|source_id| source_id == turn_id)
        {
            coalesced_turn_ids.insert(0, turn_id.to_string());
        }
    }
    coalesced_turn_ids
}

fn playback_link_payload(
    link_stage: &'static str,
    record: &QualityPlaybackRecord,
    replaced_playback_id: Option<&str>,
    terminal_status: Option<&str>,
    terminal_reason: Option<&str>,
) -> Map<String, Value> {
    json_map(serde_json::json!({
        "link_stage": link_stage,
        "playback_id": record.playback_id,
        "turn_id": record.turn_id,
        "selected_turn_id": record.turn_id,
        "source_turn_ids": record.source_turn_ids,
        "coalesced_turn_ids": record.source_turn_ids,
        "coalesced_turn_count": record.source_turn_ids.len(),
        "source_asr_session_ids": record.source_asr_session_ids,
        "source_utterance_ids": record.source_utterance_ids,
        "tts_backend": record.tts_backend.label(),
        "source_label": record.source_label,
        "source": record.metadata.source,
        "source_channel": record.metadata.source_channel,
        "manual_injection": record.metadata.manual_injection,
        "related_turn_id": record.metadata.related_turn_id,
        "first_audio_sent": record.first_audio_sent,
        "terminal_status": terminal_status,
        "terminal_reason": terminal_reason,
        "replaced_playback_id": replaced_playback_id,
    }))
}

fn json_map(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

pub fn normalize_phone_number(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

pub fn redact_phone_for_log(value: Option<&str>) -> &'static str {
    match value {
        Some(value) if !value.trim().is_empty() => "<redacted-phone>",
        _ => "<unknown>",
    }
}

pub fn shared_state(bind: SocketAddr) -> SharedState {
    Arc::new(RwLock::new(GatewayState::new(bind)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_transcript_tracks_finals_and_current_partial() {
        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = state.add_or_update_inbound_call(
            TelnyxIds {
                call_control_id: "call-1".to_string(),
                call_session_id: None,
                call_leg_id: None,
                stream_id: None,
            },
            None,
            None,
            CallStatus::PendingInbound,
        );

        state.add_transcript(&call_id, TranscriptKind::Partial, "HEL".to_string());
        state.add_transcript(&call_id, TranscriptKind::Final, "HELLO".to_string());
        state.add_transcript(&call_id, TranscriptKind::Partial, "WOR".to_string());

        let call = state.calls.get(&call_id).expect("call exists");
        assert_eq!(call.final_transcript, "HELLO");
        assert_eq!(call.current_partial.as_deref(), Some("WOR"));
    }

    #[test]
    fn stale_tts_cancel_does_not_demote_newer_active_playback() {
        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = state.add_or_update_outbound_call(
            TelnyxIds {
                call_control_id: "call-1".to_string(),
                call_session_id: Some("session-1".to_string()),
                call_leg_id: Some("leg-1".to_string()),
                stream_id: Some("stream-1".to_string()),
            },
            None,
            None,
            CallStatus::MediaStarted,
        );
        state.start_tts_job(
            &call_id,
            "tts_old".to_string(),
            LiveTtsBackend::default(),
            "old reply",
        );
        state.start_tts_job(
            &call_id,
            "tts_new".to_string(),
            LiveTtsBackend::Piper,
            "new reply",
        );

        state.mark_tts_canceled(&call_id, "tts_old");

        let call = state.calls.get(&call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::Speaking);
        let tts = call.tts.as_ref().expect("new TTS should remain active");
        assert_eq!(tts.playback_id, "tts_new");
        assert_eq!(tts.status, TtsPlaybackStatus::Queued);
    }

    #[test]
    fn quality_turn_report_counts_denominators_and_playback_linkage() {
        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        state.set_quality_event_sink(QualityEventSink::with_sender(tx), None);
        let call_id = state.add_or_update_outbound_call(
            TelnyxIds {
                call_control_id: "call-1".to_string(),
                call_session_id: Some("session-1".to_string()),
                call_leg_id: Some("leg-1".to_string()),
                stream_id: Some("stream-1".to_string()),
            },
            None,
            None,
            CallStatus::MediaStarted,
        );

        state.add_transcript(&call_id, TranscriptKind::Final, "hello".to_string());
        state.emit_quality_caller_turn_sent(
            &call_id,
            "turn_selected",
            "hello",
            None,
            CallerTurnEventMetadata {
                coalesced_turn_ids: vec!["turn_selected".to_string(), "turn_next".to_string()],
                ..Default::default()
            },
        );
        state.start_tts_job_with_linkage(
            &call_id,
            "tts_test".to_string(),
            LiveTtsBackend::Piper,
            "reply",
            QualityPlaybackLinkage {
                turn_id: Some("turn_selected".to_string()),
                coalesced_turn_ids: vec!["turn_selected".to_string(), "turn_next".to_string()],
                source_asr_session_ids: Vec::new(),
                source_utterance_ids: Vec::new(),
                source_label: "test".to_string(),
                metadata: QualityPlaybackMetadata::default(),
            },
            Some("tts_replaced"),
        );
        state.mark_tts_first_audio_latency_and_pacing(&call_id, "tts_test", 42, 0, 0);
        state.mark_tts_canceled_with_reason(&call_id, "tts_test", Some("canceled_after_call_end"));
        state.emit_quality_report_summary(&call_id, "test");

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        let link_stages: Vec<_> = events
            .iter()
            .filter(|event| event.event == "quality.turn.playback_linked")
            .map(|event| event.payload["link_stage"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(link_stages, vec!["queued", "first_audio", "terminal"]);

        let queued = events
            .iter()
            .find(|event| {
                event.event == "quality.turn.playback_linked"
                    && event.payload["link_stage"] == "queued"
            })
            .expect("queued link event");
        assert_eq!(queued.payload["turn_id"], "turn_selected");
        assert_eq!(queued.payload["source_turn_ids"][0], "turn_selected");
        assert_eq!(queued.payload["source_turn_ids"][1], "turn_next");
        assert_eq!(queued.payload["replaced_playback_id"], "tts_replaced");

        let terminal = events
            .iter()
            .find(|event| {
                event.event == "quality.turn.playback_linked"
                    && event.payload["link_stage"] == "terminal"
            })
            .expect("terminal link event");
        assert_eq!(terminal.payload["terminal_status"], "canceled");
        assert_eq!(
            terminal.payload["terminal_reason"],
            "canceled_after_call_end"
        );

        let summary = events
            .iter()
            .rev()
            .find(|event| event.event == "quality.report.summary")
            .expect("summary event");
        assert_eq!(summary.payload["raw_asr_final_events"], 1);
        assert_eq!(summary.payload["caller_turns"], 1);
        assert_eq!(summary.payload["attempted_turns"], 1);
        assert_eq!(summary.payload["attempted_playbacks"], 1);
        assert_eq!(summary.payload["coalesced_turns"], 2);
        assert_eq!(summary.payload["coalesced_response_turns"], 1);
        assert_eq!(summary.payload["played_turns"], 1);
        assert_eq!(summary.payload["played_playbacks"], 1);
        assert_eq!(summary.payload["canceled_playbacks"], 1);
        assert_eq!(summary.payload["canceled_after_call_end_turns"], 1);
        assert_eq!(summary.payload["canceled_after_call_end_playbacks"], 1);
        assert_eq!(summary.payload["excluded_turns_without_playback"], 0);
    }

    #[test]
    fn inbound_subscriptions_are_ordered_by_priority_then_creation() {
        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        state.upsert_inbound_subscription(UpsertInboundSubscription {
            id: "sub-later".to_string(),
            phone_number: "<called-phone-number>".to_string(),
            callback_url: "https://agent.example.test/offers".to_string(),
            priority: 20,
            secret_ref: None,
            enabled: true,
            metadata: BTreeMap::new(),
        });
        state.upsert_inbound_subscription(UpsertInboundSubscription {
            id: "sub-first".to_string(),
            phone_number: "<called-phone-number>".to_string(),
            callback_url: "https://agent.example.test/offers".to_string(),
            priority: 10,
            secret_ref: None,
            enabled: true,
            metadata: BTreeMap::new(),
        });

        let ordered = state.inbound_subscriptions_for_phone("<called-phone-number>");
        let ids: Vec<_> = ordered
            .into_iter()
            .map(|subscription| subscription.id)
            .collect();
        assert_eq!(ids, vec!["sub-first", "sub-later"]);
    }
}
