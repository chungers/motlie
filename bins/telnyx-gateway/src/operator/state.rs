use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::adapter::LiveAsrBackend;
use crate::call_control::TelnyxMediaConfig;

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
    pub logs: VecDeque<LogEntry>,
    pub shutdown_requested: bool,
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
            logs: VecDeque::new(),
            shutdown_requested: false,
        }
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
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.status = TtsPlaybackStatus::Canceled;
        });
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            if call.status == CallStatus::Speaking {
                call.status = status_after_tts(call);
            }
            if call.conversation.last_playback_id.as_deref() == Some(playback_id) {
                call.conversation.status = ConversationStatus::Idle;
                call.conversation.updated_at = Utc::now();
            }
            call.push_timeline(format!("tts {playback_id} canceled"));
        }
    }

    pub fn mark_tts_failed(&mut self, gateway_call_id: &str, playback_id: &str, error: String) {
        self.update_tts(gateway_call_id, playback_id, |tts| {
            tts.status = TtsPlaybackStatus::Failed;
            tts.error = Some(error.clone());
        });
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.last_error = Some(error.clone());
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
}
