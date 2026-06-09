use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context};
use async_trait::async_trait;
use motlie_voice::app::{
    CallContext, CallIds, ConversationCommand, ConversationHandler, TranscriptEvent, VoiceAppError,
};
use motlie_voice::telephony::CallAction;

use crate::call_control::TelnyxClient;
use crate::media::{SharedMediaRegistry, SpeechClearReason};
use crate::operator::state::{ConversationMode, LogLevel, QualitySpanEmission, SharedState};
use crate::quality::{BargeInQualityConfig, RedactionMode, VoiceQualityConfig};
use crate::speech;
use crate::speech::{SpeechConflictPolicy, SpeechQueueRequest};
use crate::tts::{LiveTtsBackend, SharedTtsRegistry};

const PARTIAL_BARGE_IN_MIN_CHARS: usize = 3;

pub type SharedConversationHandler = Arc<dyn ConversationHandler>;

#[derive(Clone)]
pub struct ConversationRuntime {
    telnyx: TelnyxClient,
    tts: SharedTtsRegistry,
    handler: SharedConversationHandler,
    smoke_test_enabled: Arc<AtomicBool>,
    barge_in_enabled: Arc<AtomicBool>,
}

impl ConversationRuntime {
    pub fn new(
        telnyx: TelnyxClient,
        tts: SharedTtsRegistry,
        handler: SharedConversationHandler,
        smoke_test_enabled: bool,
    ) -> Self {
        Self {
            telnyx,
            tts,
            handler,
            smoke_test_enabled: Arc::new(AtomicBool::new(smoke_test_enabled)),
            barge_in_enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn smoke_test_enabled(&self) -> bool {
        self.smoke_test_enabled.load(Ordering::SeqCst)
    }

    pub fn set_smoke_test_enabled(&self, enabled: bool) {
        self.smoke_test_enabled.store(enabled, Ordering::SeqCst);
    }

    pub fn barge_in_enabled(&self) -> bool {
        self.barge_in_enabled.load(Ordering::SeqCst)
    }

    pub fn set_barge_in_enabled(&self, enabled: bool) {
        self.barge_in_enabled.store(enabled, Ordering::SeqCst);
    }

    pub fn barge_in_label(&self) -> &'static str {
        if self.barge_in_enabled() {
            "on"
        } else {
            "off"
        }
    }

    pub fn handler_label(&self) -> &'static str {
        if self.smoke_test_enabled() {
            "smoke-test"
        } else {
            "disabled"
        }
    }
}

pub fn default_conversation_handler() -> SharedConversationHandler {
    Arc::new(SmokeTestConversationHandler)
}

#[derive(Clone, Debug, Default)]
pub struct SmokeTestConversationHandler;

#[async_trait]
impl ConversationHandler for SmokeTestConversationHandler {
    async fn on_transcript(
        &self,
        event: TranscriptEvent,
        _context: &mut CallContext,
    ) -> Result<ConversationCommand, VoiceAppError> {
        if !event.is_final() {
            return Ok(ConversationCommand::Noop);
        }
        let text = event.text().trim();
        if text.is_empty() {
            return Ok(ConversationCommand::Noop);
        }
        Ok(ConversationCommand::Say {
            text: format!("I heard: {text}"),
        })
    }
}

pub async fn handle_transcript_event(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
    quality_config: Option<&VoiceQualityConfig>,
) -> anyhow::Result<()> {
    let transcript_text = event.text().trim().to_string();
    if transcript_text.is_empty() {
        return Ok(());
    }

    let Some(snapshot) = conversation_snapshot(state, gateway_call_id, quality_config).await else {
        return Ok(());
    };
    if !snapshot.attached {
        return Ok(());
    }

    if !event.is_final() {
        if barge_in_allows(&snapshot.barge_in, BargeInTrigger::Partial)
            && is_meaningful_partial_barge_in(&transcript_text)
        {
            cancel_active_speech_for_barge_in(
                state,
                media_registry,
                gateway_call_id,
                BargeInTrigger::Partial,
                snapshot.config_id.clone(),
                snapshot.redaction_mode,
            )
            .await?;
        }
        return Ok(());
    }

    state
        .write()
        .await
        .record_conversation_user_turn(gateway_call_id, transcript_text);

    if barge_in_allows(&snapshot.barge_in, BargeInTrigger::Final) {
        cancel_active_speech_for_barge_in(
            state,
            media_registry,
            gateway_call_id,
            BargeInTrigger::Final,
            snapshot.config_id.clone(),
            snapshot.redaction_mode,
        )
        .await?;
    }

    if !runtime.smoke_test_enabled() {
        state
            .write()
            .await
            .record_conversation_idle(gateway_call_id);
        tracing::debug!(
            gateway_call_id,
            "conversation.handler.disabled_for_final_transcript"
        );
        return Ok(());
    }

    let mut context = snapshot.context;
    let command = match runtime.handler.on_transcript(event, &mut context).await {
        Ok(command) => command,
        Err(error) => {
            let error = format!("{error:#}");
            state
                .write()
                .await
                .record_conversation_failed(gateway_call_id, error.clone());
            tracing::warn!(gateway_call_id, error, "conversation.handler.failed");
            return Ok(());
        }
    };
    apply_conversation_command(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        ConversationCommandTarget {
            mode: snapshot.mode,
            call_control_id: snapshot.call_control_id,
            barge_in: snapshot.barge_in,
        },
        command,
    )
    .await
}

pub async fn handle_speech_onset(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    _runtime: &ConversationRuntime,
    gateway_call_id: &str,
    quality_config: Option<&VoiceQualityConfig>,
) -> anyhow::Result<()> {
    let Some(snapshot) = conversation_snapshot(state, gateway_call_id, quality_config).await else {
        return Ok(());
    };
    if !snapshot.attached || !barge_in_allows(&snapshot.barge_in, BargeInTrigger::SpeechOnset) {
        return Ok(());
    }

    cancel_active_speech_for_barge_in(
        state,
        media_registry,
        gateway_call_id,
        BargeInTrigger::SpeechOnset,
        snapshot.config_id.clone(),
        snapshot.redaction_mode,
    )
    .await
}

#[derive(Clone, Copy, Debug)]
enum BargeInTrigger {
    Partial,
    Final,
    SpeechOnset,
}

impl BargeInTrigger {
    fn as_str(self) -> &'static str {
        match self {
            Self::Partial => "partial",
            Self::Final => "final",
            Self::SpeechOnset => "speech_onset",
        }
    }

    fn source_label(self) -> &'static str {
        match self {
            Self::Partial => "conversation partial barge-in",
            Self::Final => "conversation final barge-in",
            Self::SpeechOnset => "conversation speech-onset barge-in",
        }
    }

    fn cancel_span_name(self) -> &'static str {
        match self {
            Self::Partial => "barge_in.partial_to_cancel_request",
            Self::Final => "barge_in.final_to_cancel_request",
            Self::SpeechOnset => "barge_in.speech_onset_to_cancel_request",
        }
    }
}

fn barge_in_allows(config: &BargeInQualityConfig, trigger: BargeInTrigger) -> bool {
    config.enabled
        && match trigger {
            BargeInTrigger::Partial => config.partial_asr_cancel_enabled,
            BargeInTrigger::Final => config.final_asr_cancel_enabled,
            BargeInTrigger::SpeechOnset => config.speech_onset_cancel_enabled,
        }
}

fn is_meaningful_partial_barge_in(text: &str) -> bool {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .take(PARTIAL_BARGE_IN_MIN_CHARS)
        .count()
        >= PARTIAL_BARGE_IN_MIN_CHARS
}

async fn cancel_active_speech_for_barge_in(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    gateway_call_id: &str,
    trigger: BargeInTrigger,
    config_id: String,
    redaction_mode: RedactionMode,
) -> anyhow::Result<()> {
    if media_registry
        .active_speech_playback_id(gateway_call_id)
        .await
        .is_none()
    {
        return Ok(());
    }

    let cancel_started_at = Instant::now();
    let playback_id = match speech::cancel_speech_with_reason(
        state,
        media_registry,
        gateway_call_id,
        trigger.source_label(),
        SpeechClearReason::BargeIn,
    )
    .await
    {
        Ok(playback_id) => playback_id,
        Err(error) if format!("{error:#}").contains("no active speech job") => return Ok(()),
        Err(error) => return Err(error),
    };

    {
        let payload = serde_json::json!({
            "playback_id": playback_id,
            "trigger": trigger.as_str(),
        });
        let payload = match payload {
            serde_json::Value::Object(map) => map,
            _ => serde_json::Map::new(),
        };
        state.write().await.emit_quality_span_finished(
            gateway_call_id,
            QualitySpanEmission {
                config_id,
                redaction_mode,
                span_name: trigger.cancel_span_name(),
                category: "barge_in",
                duration: cancel_started_at.elapsed(),
                critical_path: false,
                concurrent: true,
                payload,
            },
        );
    }
    state
        .write()
        .await
        .record_conversation_interrupted(gateway_call_id, &playback_id);
    tracing::info!(
        gateway_call_id,
        playback_id,
        trigger = trigger.as_str(),
        partial_triggered = matches!(trigger, BargeInTrigger::Partial),
        speech_onset_triggered = matches!(trigger, BargeInTrigger::SpeechOnset),
        "conversation.barge_in.cancel_requested"
    );
    Ok(())
}

struct ConversationSnapshot {
    attached: bool,
    mode: ConversationMode,
    call_control_id: String,
    barge_in: BargeInQualityConfig,
    config_id: String,
    redaction_mode: RedactionMode,
    context: CallContext,
}

#[derive(Clone, Debug)]
struct ConversationCommandTarget {
    mode: ConversationMode,
    call_control_id: String,
    barge_in: BargeInQualityConfig,
}

async fn conversation_snapshot(
    state: &SharedState,
    gateway_call_id: &str,
    quality_config: Option<&VoiceQualityConfig>,
) -> Option<ConversationSnapshot> {
    let guard = state.read().await;
    let call = guard.calls.get(gateway_call_id)?;
    let mut custom_state = BTreeMap::new();
    custom_state.insert("gateway_call_id".to_string(), call.gateway_call_id.clone());
    custom_state.insert(
        "conversation_mode".to_string(),
        call.conversation.mode.label().to_string(),
    );
    if let Some(text) = &call.conversation.last_assistant_text {
        custom_state.insert("last_assistant_text".to_string(), text.clone());
    }
    let (barge_in, config_id, redaction_mode) = quality_config
        .map(|quality_config| {
            (
                quality_config.barge_in.clone(),
                quality_config.config_id(),
                quality_config.logging.redaction_mode,
            )
        })
        .unwrap_or_else(|| {
            (
                guard.quality.config.barge_in.clone(),
                guard.quality.config_id.clone(),
                guard.quality.config.logging.redaction_mode,
            )
        });
    Some(ConversationSnapshot {
        attached: call.conversation.attached,
        mode: call.conversation.mode,
        call_control_id: call.ids.call_control_id.clone(),
        barge_in,
        config_id,
        redaction_mode,
        context: CallContext {
            ids: Some(CallIds {
                provider_call_id: call.ids.call_control_id.clone(),
                provider_session_id: call.ids.call_session_id.clone(),
                media_stream_id: call.ids.stream_id.clone(),
            }),
            custom_state,
        },
    })
}

async fn apply_conversation_command(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    target: ConversationCommandTarget,
    command: ConversationCommand,
) -> anyhow::Result<()> {
    match command {
        ConversationCommand::Noop => {
            state
                .write()
                .await
                .record_conversation_idle(gateway_call_id);
            Ok(())
        }
        ConversationCommand::Say { text } => {
            let response_text = text.trim().to_string();
            if response_text.is_empty() {
                state
                    .write()
                    .await
                    .record_conversation_idle(gateway_call_id);
                return Ok(());
            }
            match target.mode {
                ConversationMode::Manual => {
                    state
                        .write()
                        .await
                        .record_conversation_proposal(gateway_call_id, response_text);
                    Ok(())
                }
                ConversationMode::Auto => {
                    if !target.barge_in.enabled
                        && media_registry
                            .active_speech_playback_id(gateway_call_id)
                            .await
                            .is_some()
                    {
                        state
                            .write()
                            .await
                            .record_conversation_proposal(gateway_call_id, response_text);
                        tracing::info!(
                            gateway_call_id,
                            "conversation.say.deferred_barge_in_disabled"
                        );
                        return Ok(());
                    }
                    let queued = speech::queue_speech_with_request(
                        state,
                        media_registry,
                        &runtime.tts,
                        SpeechQueueRequest {
                            tts_backend: LiveTtsBackend::default(),
                            gateway_call_id: gateway_call_id.to_string(),
                            text: response_text.clone(),
                            source_label: "conversation say".to_string(),
                            conflict_policy: SpeechConflictPolicy::CancelAndReplace,
                        },
                    )
                    .await
                    .with_context(|| format!("queue conversation response for {gateway_call_id}"));
                    match queued {
                        Ok(queued) => {
                            {
                                let mut guard = state.write().await;
                                if let Some(replaced_playback_id) = &queued.replaced_playback_id {
                                    guard.record_conversation_interrupted(
                                        gateway_call_id,
                                        replaced_playback_id,
                                    );
                                }
                                guard.record_conversation_speaking(
                                    gateway_call_id,
                                    response_text,
                                    queued.playback_id.clone(),
                                );
                            }
                            tracing::info!(
                                gateway_call_id,
                                playback_id = queued.playback_id,
                                replaced_playback_id = queued.replaced_playback_id.as_deref(),
                                "conversation.say.queued"
                            );
                            Ok(())
                        }
                        Err(error) => {
                            let error = format!("{error:#}");
                            state
                                .write()
                                .await
                                .record_conversation_failed(gateway_call_id, error.clone());
                            tracing::warn!(gateway_call_id, error, "conversation.say.failed");
                            Ok(())
                        }
                    }
                }
            }
        }
        ConversationCommand::Call(action) => match action {
            CallAction::Hangup => {
                runtime
                    .telnyx
                    .hangup_call(&target.call_control_id)
                    .await
                    .with_context(|| format!("hang up call {gateway_call_id}"))?;
                let mut guard = state.write().await;
                guard.record_conversation_idle(gateway_call_id);
                guard.log(
                    LogLevel::Info,
                    format!("conversation hangup requested for {gateway_call_id}"),
                );
                Ok(())
            }
            _ => {
                let error = "unsupported conversation call action".to_string();
                state
                    .write()
                    .await
                    .record_conversation_failed(gateway_call_id, error.clone());
                bail!(error)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::SpeechCancelToken;
    use crate::operator::state::{shared_state, CallStatus, ConversationStatus, TelnyxIds};
    use tokio::sync::mpsc;

    fn test_runtime() -> ConversationRuntime {
        ConversationRuntime::new(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            default_conversation_handler(),
            true,
        )
    }

    fn failing_runtime() -> ConversationRuntime {
        ConversationRuntime::new(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            Arc::new(FailingConversationHandler),
            true,
        )
    }

    #[derive(Clone, Debug)]
    struct FailingConversationHandler;

    #[async_trait]
    impl ConversationHandler for FailingConversationHandler {
        async fn on_transcript(
            &self,
            _event: TranscriptEvent,
            _context: &mut CallContext,
        ) -> Result<ConversationCommand, VoiceAppError> {
            Err(VoiceAppError::new("model unavailable"))
        }
    }

    async fn seed_conversation_call(state: &SharedState, mode: ConversationMode) -> String {
        let mut guard = state.write().await;
        let call_id = guard.add_or_update_outbound_call(
            TelnyxIds {
                call_control_id: "call-control-1".to_string(),
                call_session_id: Some("session-1".to_string()),
                call_leg_id: Some("leg-1".to_string()),
                stream_id: Some("stream-1".to_string()),
            },
            None,
            None,
            CallStatus::MediaStarted,
        );
        guard.attach_conversation(&call_id, mode);
        call_id
    }

    #[tokio::test]
    async fn smoke_test_handler_turns_final_transcript_into_say() {
        let handler = SmokeTestConversationHandler;
        let mut context = CallContext {
            ids: None,
            custom_state: BTreeMap::new(),
        };
        let command = handler
            .on_transcript(
                TranscriptEvent::Final {
                    text: "hello".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                &mut context,
            )
            .await
            .expect("smoke-test handler should accept final transcript");

        match command {
            ConversationCommand::Say { text } => assert_eq!(text, "I heard: hello"),
            _ => panic!("expected say command"),
        }
    }

    #[tokio::test]
    async fn smoke_test_handler_ignores_partial_transcript() {
        let handler = SmokeTestConversationHandler;
        let mut context = CallContext {
            ids: None,
            custom_state: BTreeMap::new(),
        };
        let command = handler
            .on_transcript(
                TranscriptEvent::Partial {
                    text: "hello".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                &mut context,
            )
            .await
            .expect("smoke-test handler should accept partial transcript");

        match command {
            ConversationCommand::Noop => {}
            _ => panic!("expected noop command"),
        }
    }

    #[tokio::test]
    async fn disabled_runtime_records_user_turn_without_invoking_handler() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        let runtime = ConversationRuntime::new(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            Arc::new(FailingConversationHandler),
            false,
        );

        handle_transcript_event(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("disabled handler should not fail media");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Idle);
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("hello"));
        assert!(call.conversation.last_error.is_none());
    }

    #[tokio::test]
    async fn apply_manual_say_records_proposal_without_speaking() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = test_runtime();

        apply_conversation_command(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            ConversationCommandTarget {
                mode: ConversationMode::Manual,
                call_control_id: "call-control-1".to_string(),
                barge_in: BargeInQualityConfig::default(),
            },
            ConversationCommand::Say {
                text: "  assistant response  ".to_string(),
            },
        )
        .await
        .expect("manual say should record a proposal");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("assistant response")
        );
        assert!(call.conversation.last_playback_id.is_none());
        assert!(call.tts.is_none());
    }

    #[tokio::test]
    async fn meaningful_partial_interrupts_active_playback_without_regenerating() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime();

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Partial {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("partial barge-in should cancel active speech");

        assert!(cancel.is_canceled());
        assert!(media_registry
            .active_speech_playback_id(&gateway_call_id)
            .await
            .is_none());
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Interrupted);
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("assistant is speaking")
        );
    }

    #[tokio::test]
    async fn speech_onset_interrupts_active_playback_without_regenerating() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime();

        handle_speech_onset(&state, &media_registry, &runtime, &gateway_call_id, None)
            .await
            .expect("speech onset barge-in should cancel active speech");

        assert!(cancel.is_canceled());
        assert!(media_registry
            .active_speech_playback_id(&gateway_call_id)
            .await
            .is_none());
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Interrupted);
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("assistant is speaking")
        );
        assert!(call.conversation.last_user_text.is_none());
    }

    #[tokio::test]
    async fn disabled_barge_in_does_not_interrupt_active_playback_on_partial() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime();
        {
            let mut guard = state.write().await;
            guard.quality.config.set_barge_in_enabled(false);
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Partial {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("disabled barge-in should not cancel active speech");

        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Speaking);
    }

    #[tokio::test]
    async fn disabled_barge_in_defers_auto_say_when_final_arrives_during_playback() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime();
        {
            let mut guard = state.write().await;
            guard.quality.config.set_barge_in_enabled(false);
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("disabled barge-in should not fail overlapping final turn");

        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("hello"));
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("I heard: hello")
        );
    }

    #[tokio::test]
    async fn empty_or_tiny_partial_does_not_interrupt_active_playback() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime();

        for text in ["   ", "uh"] {
            handle_transcript_event(
                &state,
                &media_registry,
                &runtime,
                &gateway_call_id,
                TranscriptEvent::Partial {
                    text: text.to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                None,
            )
            .await
            .expect("non-meaningful partial should not cut speech");
        }

        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Speaking);
    }

    #[tokio::test]
    async fn final_transcript_still_regenerates_after_partial_barge_in() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime();

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Partial {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("partial should interrupt active speech");
        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "hello there".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("final should still regenerate through handler");

        assert!(cancel.is_canceled());
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("hello there")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("I heard: hello there")
        );
    }

    #[tokio::test]
    async fn handler_error_marks_conversation_failed_without_failing_media() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        let runtime = failing_runtime();

        handle_transcript_event(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("handler error should remain conversation-scoped");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::MediaStarted);
        assert!(call.last_error.is_none());
        assert_eq!(call.conversation.status, ConversationStatus::Failed);
        assert_eq!(
            call.conversation.last_error.as_deref(),
            Some("model unavailable")
        );
    }

    #[tokio::test]
    async fn apply_noop_marks_conversation_idle() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        state
            .write()
            .await
            .record_conversation_user_turn(&gateway_call_id, "user turn".to_string());
        let runtime = test_runtime();

        apply_conversation_command(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            ConversationCommandTarget {
                mode: ConversationMode::Manual,
                call_control_id: "call-control-1".to_string(),
                barge_in: BargeInQualityConfig::default(),
            },
            ConversationCommand::Noop,
        )
        .await
        .expect("noop should mark idle");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Idle);
    }

    #[tokio::test]
    async fn apply_unsupported_call_action_fails_closed() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        let runtime = test_runtime();

        let error = apply_conversation_command(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            ConversationCommandTarget {
                mode: ConversationMode::Auto,
                call_control_id: "call-control-1".to_string(),
                barge_in: BargeInQualityConfig::default(),
            },
            ConversationCommand::Call(CallAction::Transfer {
                destination: "sip:agent@example.test".to_string(),
            }),
        )
        .await
        .expect_err("unsupported action should fail");

        assert!(format!("{error:#}").contains("unsupported conversation call action"));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Failed);
        assert_eq!(
            call.conversation.last_error.as_deref(),
            Some("unsupported conversation call action")
        );
    }
}
