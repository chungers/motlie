use std::collections::{BTreeMap, VecDeque};
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
use crate::quality::{
    ActiveAsrQualitySession, QualityEvent, QualityEventContext, QualityEventSink, RedactionMode,
    VoiceQualityConfig,
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
    pub public_webhook_url: Option<String>,
    pub public_media_url: Option<String>,
    pub telnyx_media: TelnyxMediaConfig,
    pub capture_dir: Option<PathBuf>,
    pub selected_connection_id: Option<String>,
    pub selected_application_name: Option<String>,
    pub selected_phone_number: Option<String>,
    pub default_from_number: Option<String>,
    pub state_path: Option<PathBuf>,
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
    pub text_preview: String,
    pub frames_queued: usize,
    pub frames_sent: usize,
    pub mark_name: Option<String>,
    pub error: Option<String>,
    pub updated_at: DateTime<Utc>,
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
    pub conversation: ConversationState,
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
            conversation: ConversationState::default(),
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
                ..GatewayConfig::default()
            },
            inbound_mode: InboundMode::Disabled,
            started_at: Utc::now(),
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
        let event = QualityEvent::caller_turn_sent(
            context,
            turn_id.to_string(),
            text,
            include_transcript_text,
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

        let call = CallSession::pending_inbound(ids.clone(), from, to, status);
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

        let call = CallSession::outbound(ids.clone(), from, to, status);
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

    pub fn start_tts_job(&mut self, gateway_call_id: &str, playback_id: String, text: &str) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.status = CallStatus::Speaking;
            call.tts = Some(TtsPlaybackState {
                playback_id: playback_id.clone(),
                status: TtsPlaybackStatus::Queued,
                text_preview: preview_text(text),
                frames_queued: 0,
                frames_sent: 0,
                mark_name: None,
                error: None,
                updated_at: Utc::now(),
            });
            call.push_timeline(format!("tts {playback_id} queued"));
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
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
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
            if let Some(playback_id) = completed_playback {
                if call.status == CallStatus::Speaking {
                    call.status = status_after_tts(call);
                }
                if call.conversation.last_playback_id.as_deref() == Some(playback_id.as_str()) {
                    call.conversation.status = ConversationStatus::Idle;
                    call.conversation.updated_at = Utc::now();
                }
                call.push_timeline(format!("tts {playback_id} completed"));
            }
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
        state.start_tts_job(&call_id, "tts_old".to_string(), "old reply");
        state.start_tts_job(&call_id, "tts_new".to_string(), "new reply");

        state.mark_tts_canceled(&call_id, "tts_old");

        let call = state.calls.get(&call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::Speaking);
        let tts = call.tts.as_ref().expect("new TTS should remain active");
        assert_eq!(tts.playback_id, "tts_new");
        assert_eq!(tts.status, TtsPlaybackStatus::Queued);
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
