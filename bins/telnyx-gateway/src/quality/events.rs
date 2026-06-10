use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::config::{RedactionMode, VoiceQualityConfig};

pub const QUALITY_SCHEMA_VERSION: u32 = 1;

pub type QualityEventPayload = Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityEvent {
    pub quality_schema_version: u32,
    pub event_sequence: u64,
    pub event: String,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_call_id: Option<String>,
    pub host_id: String,
    pub git_sha: String,
    pub config_id: String,
    pub redaction_mode: RedactionMode,
    pub created_at: DateTime<Utc>,
    #[serde(flatten)]
    pub payload: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct QualityEventContext {
    pub event_sequence: u64,
    pub run_id: String,
    pub gateway_call_id: Option<String>,
    pub config_id: String,
    pub redaction_mode: RedactionMode,
}

impl QualityEventContext {
    pub fn new(
        event_sequence: u64,
        run_id: impl Into<String>,
        gateway_call_id: Option<String>,
        config_id: impl Into<String>,
        redaction_mode: RedactionMode,
    ) -> Self {
        Self {
            event_sequence,
            run_id: run_id.into(),
            gateway_call_id,
            config_id: config_id.into(),
            redaction_mode,
        }
    }
}

impl QualityEvent {
    pub fn new(
        context: QualityEventContext,
        event: impl Into<String>,
        payload: Map<String, Value>,
    ) -> Self {
        Self {
            quality_schema_version: QUALITY_SCHEMA_VERSION,
            event_sequence: context.event_sequence,
            event: event.into(),
            run_id: context.run_id,
            gateway_call_id: context.gateway_call_id,
            host_id: host_id(),
            git_sha: git_sha(),
            config_id: context.config_id,
            redaction_mode: context.redaction_mode,
            created_at: Utc::now(),
            payload,
        }
    }

    pub fn config_snapshot(
        context: QualityEventContext,
        config: &VoiceQualityConfig,
        snapshot_reason: &'static str,
        effective_scope: &'static str,
        effective_after_asr_session_id: Option<String>,
    ) -> Self {
        let mut payload = Map::new();
        payload.insert(
            "config_hash_algorithm".to_string(),
            Value::String("sha256-canonical-json".to_string()),
        );
        payload.insert(
            "snapshot_reason".to_string(),
            Value::String(snapshot_reason.to_string()),
        );
        payload.insert(
            "effective_scope".to_string(),
            Value::String(effective_scope.to_string()),
        );
        payload.insert(
            "effective_after_asr_session_id".to_string(),
            effective_after_asr_session_id
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        payload.insert(
            "resolved_config".to_string(),
            serde_json::to_value(config).expect("quality config serializes"),
        );
        Self::new(context, "call.config.snapshot", payload)
    }

    pub fn asr_session_started(
        context: QualityEventContext,
        session: &ActiveAsrQualitySession,
        stream_id: Option<&str>,
        reason: &'static str,
    ) -> Self {
        let payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id,
            "utterance_id": session.utterance_id,
            "stream_id": stream_id,
            "reason": reason,
        }));
        Self::new(context, "asr.session.started", payload)
    }

    pub fn asr_turn_mapped(
        context: QualityEventContext,
        session: &ActiveAsrQualitySession,
        turn_id: impl Into<String>,
        final_transcript_event_id: impl Into<String>,
        caller_turn_sent: bool,
    ) -> Self {
        let payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id,
            "utterance_id": session.utterance_id,
            "turn_id": turn_id.into(),
            "final_transcript_event_id": final_transcript_event_id.into(),
            "caller_turn_sent": caller_turn_sent,
        }));
        Self::new(context, "asr.turn_mapped", payload)
    }

    pub fn caller_turn_sent(
        context: QualityEventContext,
        turn_id: impl Into<String>,
        text: &str,
        include_transcript_text: bool,
    ) -> Self {
        let mut payload = map_from_value(json!({
            "turn_id": turn_id.into(),
            "text_words": text.split_whitespace().count(),
            "text_chars": text.chars().count(),
            "transcript_text_included": include_transcript_text,
        }));
        if include_transcript_text {
            payload.insert("text".to_string(), Value::String(text.to_string()));
        }
        Self::new(context, "text_call.caller_turn.sent", payload)
    }

    pub fn span_finished(
        context: QualityEventContext,
        span_name: impl Into<String>,
        category: impl Into<String>,
        duration: Duration,
        critical_path: bool,
        concurrent: bool,
        mut payload: Map<String, Value>,
    ) -> Self {
        payload.insert("span".to_string(), Value::String(span_name.into()));
        payload.insert("category".to_string(), Value::String(category.into()));
        payload.insert(
            "duration_ms".to_string(),
            Value::Number(serde_json::Number::from(duration.as_millis() as u64)),
        );
        payload.insert("critical_path".to_string(), Value::Bool(critical_path));
        payload.insert("concurrent".to_string(), Value::Bool(concurrent));
        Self::new(context, "voice.span.finished", payload)
    }

    pub fn inbound_transport_rollup(
        context: QualityEventContext,
        payload: Map<String, Value>,
    ) -> Self {
        Self::new(context, "media.inbound_transport.rollup", payload)
    }

    pub fn outbound_pacing_rollup(
        context: QualityEventContext,
        payload: Map<String, Value>,
    ) -> Self {
        Self::new(context, "media.outbound_pacing.rollup", payload)
    }

    #[cfg(test)]
    fn test_event() -> Self {
        Self::new(
            QualityEventContext::new(
                1,
                "run_test",
                Some("gwc_test".to_string()),
                "cfg_test",
                RedactionMode::MetricsOnly,
            ),
            "quality.test",
            Map::new(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ActiveAsrQualitySession {
    pub asr_session_id: String,
    pub utterance_id: String,
    pub config_id: String,
    pub redaction_mode: RedactionMode,
    pub include_transcript_text: bool,
    pub opened_at: Instant,
}

impl ActiveAsrQualitySession {
    pub fn new(config: &VoiceQualityConfig) -> Self {
        Self {
            asr_session_id: format!("asr_{}", Uuid::new_v4().simple()),
            utterance_id: format!("utt_{}", Uuid::new_v4().simple()),
            config_id: config.config_id(),
            redaction_mode: config.logging.redaction_mode,
            include_transcript_text: config.logging.include_transcript_text,
            opened_at: Instant::now(),
        }
    }
}

#[derive(Clone)]
pub struct QualityEventSink {
    tx: Option<mpsc::Sender<QualityEvent>>,
    dropped: Arc<AtomicU64>,
}

impl fmt::Debug for QualityEventSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualityEventSink")
            .field("enabled", &self.tx.is_some())
            .field("dropped", &self.dropped_count())
            .finish()
    }
}

impl Default for QualityEventSink {
    fn default() -> Self {
        Self::disabled()
    }
}

impl QualityEventSink {
    pub fn disabled() -> Self {
        Self {
            tx: None,
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn with_sender(tx: mpsc::Sender<QualityEvent>) -> Self {
        Self {
            tx: Some(tx),
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn start_jsonl_writer(path: &Path, capacity: usize) -> Result<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create quality log directory {}", parent.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open quality log {}", path.display()))?;
        let (tx, mut rx) = mpsc::channel(capacity.max(1));
        let path = path.to_path_buf();
        tokio::spawn(async move {
            run_jsonl_writer(path, file, &mut rx).await;
        });
        Ok(Self::with_sender(tx))
    }

    pub fn emit(&self, event: QualityEvent) {
        let Some(tx) = &self.tx else {
            return;
        };
        if tx.try_send(event).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn is_enabled(&self) -> bool {
        self.tx.is_some()
    }
}

async fn run_jsonl_writer(
    path: PathBuf,
    mut file: std::fs::File,
    rx: &mut mpsc::Receiver<QualityEvent>,
) {
    while let Some(event) = rx.recv().await {
        match serde_json::to_string(&event) {
            Ok(encoded) => {
                if let Err(error) = writeln!(file, "{encoded}") {
                    tracing::warn!(path = %path.display(), error = %error, "quality_log.writer_error");
                    break;
                }
            }
            Err(error) => tracing::warn!(error = %error, "quality_log.serialize_error"),
        }
    }
}

fn map_from_value(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

fn host_id() -> String {
    static HOST_ID: OnceLock<String> = OnceLock::new();
    HOST_ID
        .get_or_init(|| std::env::var("HOSTNAME").unwrap_or_else(|_| "host_unknown".to_string()))
        .clone()
}

fn git_sha() -> String {
    static GIT_SHA: OnceLock<String> = OnceLock::new();
    GIT_SHA
        .get_or_init(|| {
            option_env!("GIT_SHA")
                .or(option_env!("VERGEN_GIT_SHA"))
                .unwrap_or("git_unknown")
                .to_string()
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn nonblocking_sink_drops_when_queue_is_full() {
        let (tx, mut rx) = mpsc::channel(1);
        let sink = QualityEventSink::with_sender(tx);

        sink.emit(QualityEvent::test_event());
        sink.emit(QualityEvent::test_event());

        assert_eq!(sink.dropped_count(), 1);
        let event = rx.recv().await.expect("first event should remain queued");
        assert_eq!(event.event, "quality.test");
    }

    #[test]
    fn quality_events_include_build_git_sha() {
        let event = QualityEvent::test_event();

        assert_ne!(event.git_sha, "git_unknown");
        assert!(!event.git_sha.trim().is_empty());
    }

    #[test]
    fn caller_turn_event_omits_transcript_text_by_default() {
        let event = QualityEvent::caller_turn_sent(
            QualityEventContext::new(
                1,
                "run_test",
                Some("gwc_test".to_string()),
                "cfg_test",
                RedactionMode::MetricsOnly,
            ),
            "turn_test",
            "reset account access",
            false,
        );
        assert_eq!(event.payload["text_words"], 3);
        assert_eq!(event.payload["transcript_text_included"], false);
        assert!(!event.payload.contains_key("text"));
    }

    #[test]
    fn span_and_rollup_events_use_normalized_payloads() {
        let context = QualityEventContext::new(
            2,
            "run_test",
            Some("gwc_test".to_string()),
            "cfg_test",
            RedactionMode::MetricsOnly,
        );
        let span = QualityEvent::span_finished(
            context.clone(),
            "asr.endpoint_wait",
            "endpointing",
            Duration::from_millis(123),
            true,
            false,
            map_from_value(json!({ "asr_session_id": "asr_test" })),
        );
        assert_eq!(span.event, "voice.span.finished");
        assert_eq!(span.payload["span"], "asr.endpoint_wait");
        assert_eq!(span.payload["category"], "endpointing");
        assert_eq!(span.payload["duration_ms"], 123);
        assert_eq!(span.payload["critical_path"], true);
        assert_eq!(span.payload["concurrent"], false);

        let inbound = QualityEvent::inbound_transport_rollup(
            context.clone(),
            map_from_value(json!({ "packets_total": 12 })),
        );
        assert_eq!(inbound.event, "media.inbound_transport.rollup");
        assert_eq!(inbound.payload["packets_total"], 12);

        let outbound = QualityEvent::outbound_pacing_rollup(
            context,
            map_from_value(json!({ "underrun_count": 1 })),
        );
        assert_eq!(outbound.event, "media.outbound_pacing.rollup");
        assert_eq!(outbound.payload["underrun_count"], 1);
    }

    #[test]
    fn config_snapshot_carries_resolved_config_once() {
        let config = VoiceQualityConfig::default();
        let event = QualityEvent::config_snapshot(
            QualityEventContext::new(
                1,
                "run_test",
                Some("gwc_test".to_string()),
                config.config_id(),
                config.logging.redaction_mode,
            ),
            &config,
            "stream_start",
            "new_asr_sessions",
            None,
        );
        assert_eq!(event.event, "call.config.snapshot");
        assert!(event.payload.contains_key("resolved_config"));
        assert_eq!(
            event.payload["resolved_config"]["logging"]["include_transcript_text"],
            false
        );
    }
}
