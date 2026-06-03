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
    pub asr_backend: LiveAsrBackend,
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
            Self::Ended => "ended",
            Self::Failed => "failed",
        }
    }

    pub fn allows_media_start(self) -> bool {
        matches!(self, Self::Dialing | Self::Answering | Self::Answered)
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
        let default_asr_backend = self.config.asr_backend;
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
        let asr_backend = call.asr_backend.unwrap_or(default_asr_backend);
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
            call.status = CallStatus::Transcribing;
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
