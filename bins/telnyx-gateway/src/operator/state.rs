use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

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
    Answering,
    Answered,
    MediaStarted,
    Transcribing,
    Ended,
    Failed,
}

impl CallStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::PendingInbound => "waiting",
            Self::IgnoredInbound => "disabled",
            Self::Answering => "answering",
            Self::Answered => "answered",
            Self::MediaStarted => "media",
            Self::Transcribing => "transcribing",
            Self::Ended => "ended",
            Self::Failed => "failed",
        }
    }

    pub fn allows_media_start(self) -> bool {
        matches!(self, Self::Answering | Self::Answered)
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
    pub timeline: Vec<TimelineEntry>,
    pub unread_events: u32,
    pub last_error: Option<String>,
    pub terminal_reason: Option<String>,
}

impl CallSession {
    pub fn pending_inbound(
        ids: TelnyxIds,
        from: Option<String>,
        to: Option<String>,
        status: CallStatus,
    ) -> Self {
        let mut call = Self {
            gateway_call_id: format!("gwc_{}", Uuid::new_v4().simple()),
            direction: CallDirection::Inbound,
            status,
            ids,
            from,
            to,
            media: MediaMetadata::default(),
            transcripts: Vec::new(),
            timeline: Vec::new(),
            unread_events: 0,
            last_error: None,
            terminal_reason: None,
        };
        call.push_timeline("call created");
        call
    }

    pub fn push_timeline(&mut self, message: impl Into<String>) {
        self.unread_events = self.unread_events.saturating_add(1);
        self.timeline.push(TimelineEntry {
            at: Utc::now(),
            message: message.into(),
        });
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
    pub calls: BTreeMap<String, CallSession>,
    pub call_control_index: BTreeMap<String, String>,
    pub stream_index: BTreeMap<String, String>,
    pub selected_call: Option<String>,
    pub logs: VecDeque<LogEntry>,
    pub shutdown_requested: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamAttachOutcome {
    Attached {
        gateway_call_id: String,
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
                ..GatewayConfig::default()
            },
            inbound_mode: InboundMode::Disabled,
            calls: BTreeMap::new(),
            call_control_index: BTreeMap::new(),
            stream_index: BTreeMap::new(),
            selected_call: None,
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
        if self.selected_call.is_none() && status == CallStatus::PendingInbound {
            self.selected_call = Some(gateway_call_id.clone());
        }
        self.calls.insert(gateway_call_id.clone(), call);
        gateway_call_id
    }

    pub fn call_by_target_mut(&mut self, target: Option<&str>) -> Option<&mut CallSession> {
        let id = match target {
            Some(value) => value.to_string(),
            None => self.selected_call.clone()?,
        };
        self.calls.get_mut(&id)
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
        call.ids.stream_id = Some(stream_id.clone());
        call.media = media;
        call.status = CallStatus::MediaStarted;
        call.push_timeline("media stream started");
        self.stream_index.insert(stream_id, gateway_call_id.clone());
        StreamAttachOutcome::Attached { gateway_call_id }
    }

    pub fn add_transcript(&mut self, gateway_call_id: &str, kind: TranscriptKind, text: String) {
        if let Some(call) = self.calls.get_mut(gateway_call_id) {
            call.status = CallStatus::Transcribing;
            call.unread_events = call.unread_events.saturating_add(1);
            call.transcripts.push(TranscriptLine {
                at: Utc::now(),
                kind,
                text,
            });
        }
    }
}

pub fn shared_state(bind: SocketAddr) -> SharedState {
    Arc::new(RwLock::new(GatewayState::new(bind)))
}
