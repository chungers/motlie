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
use sha2::{Digest, Sha256};
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
        mut extra: Map<String, Value>,
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
        payload.append(&mut extra);
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
        metadata: CallerTurnEventMetadata,
    ) -> Self {
        let redaction_mode = context.redaction_mode;
        let turn_id = turn_id.into();
        let coalesced_turn_ids = if metadata.coalesced_turn_ids.is_empty() {
            vec![turn_id.clone()]
        } else {
            metadata.coalesced_turn_ids
        };
        let asr_session_id = metadata.asr_session_id;
        let asr_session_ids = if metadata.asr_session_ids.is_empty() {
            asr_session_id.iter().cloned().collect()
        } else {
            metadata.asr_session_ids
        };
        let utterance_id = metadata.utterance_id;
        let utterance_ids = if metadata.utterance_ids.is_empty() {
            utterance_id.iter().cloned().collect()
        } else {
            metadata.utterance_ids
        };
        let mut payload = map_from_value(json!({
            "turn_id": turn_id,
            "asr_session_id": asr_session_id,
            "asr_session_ids": asr_session_ids,
            "utterance_id": utterance_id,
            "utterance_ids": utterance_ids,
            "confidence": metadata.confidence,
            "transcript_event_count": metadata.transcript_event_count,
            "coalesced_turn_count": coalesced_turn_ids.len(),
            "coalesced_turn_ids": coalesced_turn_ids,
            "text_words": text.split_whitespace().count(),
            "text_chars": text.chars().count(),
            "transcript_text_included": transcript_plaintext_included(
                redaction_mode,
                include_transcript_text
            ),
        }));
        insert_transcript_text_fields(
            &mut payload,
            redaction_mode,
            include_transcript_text,
            "text",
            text,
        );
        Self::new(context, "text_call.caller_turn.sent", payload)
    }

    pub fn conversation_processor_visible_turn(
        context: QualityEventContext,
        turn_id: Option<&str>,
        text: &str,
        include_transcript_text: bool,
        metadata: ProcessorVisibleTurnEventMetadata<'_>,
    ) -> Self {
        let redaction_mode = context.redaction_mode;
        let mut payload = map_from_value(json!({
            "turn_id": turn_id,
            "processor": metadata.processor,
            "response_mode": metadata.response_mode,
            "confidence": metadata.confidence,
            "coalesced_turn_count": metadata.coalesced_turn_ids.len(),
            "coalesced_turn_ids": metadata.coalesced_turn_ids,
            "source_asr_session_ids": metadata.source_asr_session_ids,
            "source_utterance_ids": metadata.source_utterance_ids,
            "text_words": text.split_whitespace().count(),
            "text_chars": text.chars().count(),
            "transcript_text_included": transcript_plaintext_included(
                redaction_mode,
                include_transcript_text
            ),
        }));
        insert_transcript_text_fields(
            &mut payload,
            redaction_mode,
            include_transcript_text,
            "text",
            text,
        );
        Self::new(context, "conversation.processor_visible_turn", payload)
    }

    pub fn transcript_suppressed(
        context: QualityEventContext,
        session: Option<&ActiveAsrQualitySession>,
        transcript_kind: impl Into<String>,
        suppression_reason: impl Into<String>,
        text: &str,
        include_transcript_text: bool,
        mut extra: Map<String, Value>,
    ) -> Self {
        let redaction_mode = context.redaction_mode;
        let mut payload = map_from_value(json!({
            "asr_session_id": session.map(|session| session.asr_session_id.clone()),
            "utterance_id": session.map(|session| session.utterance_id.clone()),
            "transcript_kind": transcript_kind.into(),
            "suppression_reason": suppression_reason.into(),
            "text_words": text.split_whitespace().count(),
            "text_chars": text.chars().count(),
            "transcript_text_included": transcript_plaintext_included(
                redaction_mode,
                include_transcript_text
            ),
        }));
        payload.append(&mut extra);
        insert_transcript_text_fields(
            &mut payload,
            redaction_mode,
            include_transcript_text,
            "text",
            text,
        );
        Self::new(context, "transcript.suppressed", payload)
    }

    pub fn caller_partial_sent(
        context: QualityEventContext,
        session: &ActiveAsrQualitySession,
        text: &str,
        confidence: Option<f32>,
        stability: Option<f32>,
        speech_state: impl Into<String>,
        include_transcript_text: bool,
    ) -> Self {
        let redaction_mode = context.redaction_mode;
        let mut payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id,
            "utterance_id": session.utterance_id,
            "speech_state": speech_state.into(),
            "confidence": confidence,
            "stability": stability,
            "text_words": text.split_whitespace().count(),
            "text_chars": text.chars().count(),
            "transcript_text_included": transcript_plaintext_included(
                redaction_mode,
                include_transcript_text
            ),
        }));
        insert_transcript_text_fields(
            &mut payload,
            redaction_mode,
            include_transcript_text,
            "text",
            text,
        );
        Self::new(context, "text_call.caller_partial.sent", payload)
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

    pub fn turn_batch_lifecycle(
        context: QualityEventContext,
        lifecycle_event: &'static str,
        mut payload: Map<String, Value>,
    ) -> Self {
        payload.insert(
            "lifecycle_event".to_string(),
            Value::String(lifecycle_event.to_string()),
        );
        let event = match lifecycle_event {
            "accumulated" => "turn_batch_accumulated",
            "prompt_complete" => "turn_batch_prompt_complete",
            "reset" => "turn_batch_reset",
            "output_rejected" => "turn_batch_output_rejected",
            _ => "turn_batch_lifecycle",
        };
        Self::new(context, event, payload)
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

    pub fn turn_playback_linked(context: QualityEventContext, payload: Map<String, Value>) -> Self {
        Self::new(context, "quality.turn.playback_linked", payload)
    }

    pub fn report_summary(context: QualityEventContext, payload: Map<String, Value>) -> Self {
        Self::new(context, "quality.report.summary", payload)
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallerTurnEventMetadata {
    pub asr_session_id: Option<String>,
    pub asr_session_ids: Vec<String>,
    pub utterance_id: Option<String>,
    pub utterance_ids: Vec<String>,
    pub confidence: Option<f32>,
    pub transcript_event_count: usize,
    pub coalesced_turn_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProcessorVisibleTurnEventMetadata<'a> {
    pub coalesced_turn_ids: &'a [String],
    pub source_asr_session_ids: &'a [String],
    pub source_utterance_ids: &'a [String],
    pub processor: &'a str,
    pub response_mode: &'a str,
    pub confidence: Option<f32>,
}

pub fn transcript_plaintext_included(
    redaction_mode: RedactionMode,
    include_transcript_text: bool,
) -> bool {
    include_transcript_text && matches!(redaction_mode, RedactionMode::SensitivePlaintext)
}

pub fn insert_transcript_text_fields(
    payload: &mut Map<String, Value>,
    redaction_mode: RedactionMode,
    include_transcript_text: bool,
    field_name: &str,
    text: &str,
) {
    match redaction_mode {
        RedactionMode::SensitivePlaintext if include_transcript_text => {
            payload.insert(field_name.to_string(), Value::String(text.to_string()));
        }
        RedactionMode::HashedText => {
            payload.insert(
                format!("{field_name}_hash_algorithm"),
                Value::String("sha256".to_string()),
            );
            payload.insert(
                format!("{field_name}_sha256"),
                Value::String(format!("{:x}", Sha256::digest(text.as_bytes()))),
            );
        }
        RedactionMode::RedactedText => {
            payload.insert(format!("{field_name}_redacted"), Value::Bool(true));
        }
        RedactionMode::MetricsOnly | RedactionMode::SensitivePlaintext => {}
    }
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
            CallerTurnEventMetadata::default(),
        );
        assert_eq!(event.payload["text_words"], 3);
        assert_eq!(event.payload["coalesced_turn_count"], 1);
        assert_eq!(event.payload["coalesced_turn_ids"][0], "turn_test");
        assert_eq!(event.payload["transcript_text_included"], false);
        assert!(!event.payload.contains_key("text"));
    }

    #[test]
    fn caller_turn_event_hashes_text_when_hash_redaction_is_selected() {
        let event = QualityEvent::caller_turn_sent(
            QualityEventContext::new(
                1,
                "run_test",
                Some("gwc_test".to_string()),
                "cfg_test",
                RedactionMode::HashedText,
            ),
            "turn_test",
            "reset account access",
            true,
            CallerTurnEventMetadata {
                asr_session_id: Some("asr_test".to_string()),
                asr_session_ids: vec!["asr_test".to_string(), "asr_next".to_string()],
                utterance_id: Some("utt_test".to_string()),
                utterance_ids: vec!["utt_test".to_string(), "utt_next".to_string()],
                confidence: Some(0.81),
                transcript_event_count: 2,
                coalesced_turn_ids: vec!["turn_test".to_string(), "turn_next".to_string()],
            },
        );
        assert_eq!(event.payload["asr_session_id"], "asr_test");
        assert_eq!(event.payload["asr_session_ids"][0], "asr_test");
        assert_eq!(event.payload["asr_session_ids"][1], "asr_next");
        assert_eq!(event.payload["utterance_id"], "utt_test");
        assert_eq!(event.payload["utterance_ids"][0], "utt_test");
        assert_eq!(event.payload["utterance_ids"][1], "utt_next");
        assert_eq!(event.payload["transcript_event_count"], 2);
        assert_eq!(event.payload["coalesced_turn_count"], 2);
        assert_eq!(event.payload["transcript_text_included"], false);
        assert_eq!(event.payload["text_hash_algorithm"], "sha256");
        assert_eq!(
            event.payload["text_sha256"]
                .as_str()
                .expect("hash should be string")
                .len(),
            64
        );
        assert!(!event.payload.contains_key("text"));
    }

    #[test]
    fn caller_partial_event_carries_scores_and_redacts_text_by_default() {
        let config = VoiceQualityConfig::default();
        let session = ActiveAsrQualitySession::new(&config);
        let event = QualityEvent::caller_partial_sent(
            QualityEventContext::new(
                1,
                "run_test",
                Some("gwc_test".to_string()),
                config.config_id(),
                config.logging.redaction_mode,
            ),
            &session,
            "reset account access",
            Some(0.82),
            Some(0.67),
            "speaking",
            false,
        );
        assert_eq!(event.event, "text_call.caller_partial.sent");
        assert_eq!(event.payload["asr_session_id"], session.asr_session_id);
        assert_eq!(event.payload["utterance_id"], session.utterance_id);
        assert_eq!(event.payload["speech_state"], "speaking");
        let confidence = event.payload["confidence"]
            .as_f64()
            .expect("confidence should be numeric");
        let stability = event.payload["stability"]
            .as_f64()
            .expect("stability should be numeric");
        assert!((confidence - 0.82).abs() < 0.000_001);
        assert!((stability - 0.67).abs() < 0.000_001);
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
    fn turn_batch_lifecycle_events_use_greppable_names() {
        let context = QualityEventContext::new(
            4,
            "run_test",
            Some("gwc_test".to_string()),
            "cfg_test",
            RedactionMode::MetricsOnly,
        );
        let event = QualityEvent::turn_batch_lifecycle(
            context,
            "accumulated",
            map_from_value(json!({
                "batch_id": "turn-batch-0-0",
                "epoch": 0,
                "accumulated_turn_count": 1,
                "target_turn_count": 3
            })),
        );

        assert_eq!(event.event, "turn_batch_accumulated");
        assert_eq!(event.payload["lifecycle_event"], "accumulated");
        assert_eq!(event.payload["batch_id"], "turn-batch-0-0");
        assert_eq!(event.payload["accumulated_turn_count"], 1);
        assert_eq!(event.payload["target_turn_count"], 3);
    }

    #[test]
    fn turn_linkage_and_summary_events_use_normalized_names() {
        let context = QualityEventContext::new(
            3,
            "run_test",
            Some("gwc_test".to_string()),
            "cfg_test",
            RedactionMode::MetricsOnly,
        );
        let link = QualityEvent::turn_playback_linked(
            context.clone(),
            map_from_value(json!({
                "link_stage": "queued",
                "turn_id": "turn_selected",
                "playback_id": "tts_test"
            })),
        );
        assert_eq!(link.event, "quality.turn.playback_linked");
        assert_eq!(link.payload["link_stage"], "queued");
        assert_eq!(link.payload["turn_id"], "turn_selected");

        let summary = QualityEvent::report_summary(
            context,
            map_from_value(json!({
                "attempted_turns": 1,
                "played_turns": 1,
                "canceled_after_call_end_turns": 0
            })),
        );
        assert_eq!(summary.event, "quality.report.summary");
        assert_eq!(summary.payload["attempted_turns"], 1);
        assert_eq!(summary.payload["played_turns"], 1);
    }

    #[test]
    fn processor_visible_turn_event_uses_transcript_redaction_policy() {
        let context = QualityEventContext::new(
            5,
            "run_test",
            Some("gwc_test".to_string()),
            "cfg_test",
            RedactionMode::HashedText,
        );
        let coalesced_turn_ids = vec!["turn_1".to_string(), "turn_2".to_string()];
        let source_asr_session_ids = vec!["asr_1".to_string()];
        let source_utterance_ids = vec!["utt_1".to_string()];
        let event = QualityEvent::conversation_processor_visible_turn(
            context,
            Some("turn_1"),
            "visible caller text",
            true,
            ProcessorVisibleTurnEventMetadata {
                coalesced_turn_ids: &coalesced_turn_ids,
                source_asr_session_ids: &source_asr_session_ids,
                source_utterance_ids: &source_utterance_ids,
                processor: "identity",
                response_mode: "command",
                confidence: Some(0.91),
            },
        );

        assert_eq!(event.event, "conversation.processor_visible_turn");
        assert_eq!(event.payload["turn_id"], "turn_1");
        assert_eq!(event.payload["coalesced_turn_count"], 2);
        assert_eq!(event.payload["source_asr_session_ids"][0], "asr_1");
        assert_eq!(event.payload["processor"], "identity");
        assert_eq!(event.payload["transcript_text_included"], false);
        assert!(event.payload.get("text").is_none());
        assert!(event.payload.get("text_sha256").is_some());
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
            Map::new(),
        );
        assert_eq!(event.event, "call.config.snapshot");
        assert!(event.payload.contains_key("resolved_config"));
        assert_eq!(
            event.payload["resolved_config"]["logging"]["include_transcript_text"],
            false
        );
    }
}
