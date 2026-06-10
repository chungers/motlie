use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, bail};
use async_trait::async_trait;
use motlie_voice::app::{
    CallContext, CallIds, ConversationCommand, ConversationHandler, TranscriptEvent, VoiceAppError,
};
use motlie_voice::telephony::CallAction;
use tokio::sync::Mutex;
use tokio::time::{Duration, sleep};

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
    handler_enabled: Arc<AtomicBool>,
    smoke_test_final_coalescing_enabled: Arc<AtomicBool>,
    barge_in_enabled: Arc<AtomicBool>,
    smoke_test_pending_finals: Arc<Mutex<HashMap<String, PendingSmokeTestFinal>>>,
    deferred_say_generations: Arc<Mutex<HashMap<String, u64>>>,
}

impl ConversationRuntime {
    pub fn new(
        telnyx: TelnyxClient,
        tts: SharedTtsRegistry,
        handler: SharedConversationHandler,
        smoke_test_enabled: bool,
    ) -> Self {
        Self::new_with_handler_options(telnyx, tts, handler, smoke_test_enabled, false)
    }

    pub fn new_with_handler_options(
        telnyx: TelnyxClient,
        tts: SharedTtsRegistry,
        handler: SharedConversationHandler,
        handler_enabled: bool,
        smoke_test_final_coalescing_enabled: bool,
    ) -> Self {
        Self {
            telnyx,
            tts,
            handler,
            handler_enabled: Arc::new(AtomicBool::new(handler_enabled)),
            smoke_test_final_coalescing_enabled: Arc::new(AtomicBool::new(
                smoke_test_final_coalescing_enabled,
            )),
            barge_in_enabled: Arc::new(AtomicBool::new(true)),
            smoke_test_pending_finals: Arc::new(Mutex::new(HashMap::new())),
            deferred_say_generations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn handler_enabled(&self) -> bool {
        self.handler_enabled.load(Ordering::SeqCst)
    }

    pub fn set_handler_enabled(&self, enabled: bool) {
        self.handler_enabled.store(enabled, Ordering::SeqCst);
    }

    pub fn smoke_test_enabled(&self) -> bool {
        self.handler_enabled()
    }

    pub fn set_smoke_test_enabled(&self, enabled: bool) {
        self.set_handler_enabled(enabled);
        self.smoke_test_final_coalescing_enabled
            .store(enabled, Ordering::SeqCst);
    }

    pub fn smoke_test_final_coalescing_enabled(&self) -> bool {
        self.smoke_test_final_coalescing_enabled
            .load(Ordering::SeqCst)
    }

    pub fn barge_in_enabled(&self) -> bool {
        self.barge_in_enabled.load(Ordering::SeqCst)
    }

    pub fn set_barge_in_enabled(&self, enabled: bool) {
        self.barge_in_enabled.store(enabled, Ordering::SeqCst);
    }

    pub fn barge_in_label(&self) -> &'static str {
        if self.barge_in_enabled() { "on" } else { "off" }
    }

    pub fn handler_label(&self) -> &'static str {
        if !self.handler_enabled() {
            "disabled"
        } else if self.smoke_test_final_coalescing_enabled() {
            "smoke-test"
        } else {
            "handler"
        }
    }
}

#[derive(Clone, Debug)]
struct SmokeTestFinalInput {
    event: TranscriptEvent,
    quality_config: Option<VoiceQualityConfig>,
    final_transcript_at: Instant,
    debounce: Duration,
}

#[derive(Clone, Debug)]
struct PendingSmokeTestFinal {
    event: TranscriptEvent,
    quality_config: Option<VoiceQualityConfig>,
    final_transcript_at: Instant,
    generation: u64,
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

    let final_transcript_at = Instant::now();

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

    if !runtime.handler_enabled() {
        state
            .write()
            .await
            .record_conversation_user_turn(gateway_call_id, transcript_text);
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

    let event = event_with_trimmed_text(event, transcript_text);
    let quality_config = quality_config.cloned();
    if snapshot.endpoint_merge_window_ms == 0 || !runtime.smoke_test_final_coalescing_enabled() {
        return dispatch_final_transcript_to_handler(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            event,
            quality_config.as_ref(),
            final_transcript_at,
        )
        .await;
    }

    schedule_smoke_test_final(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        SmokeTestFinalInput {
            event,
            quality_config,
            final_transcript_at,
            debounce: Duration::from_millis(snapshot.endpoint_merge_window_ms),
        },
    )
    .await;
    Ok(())
}

fn event_with_trimmed_text(event: TranscriptEvent, text: String) -> TranscriptEvent {
    match event {
        TranscriptEvent::Partial { update, .. } => TranscriptEvent::Partial { text, update },
        TranscriptEvent::Final { update, .. } => TranscriptEvent::Final { text, update },
    }
}

async fn schedule_smoke_test_final(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    input: SmokeTestFinalInput,
) {
    let generation = {
        let mut pending = runtime.smoke_test_pending_finals.lock().await;
        let entry = pending
            .entry(gateway_call_id.to_string())
            .or_insert_with(|| PendingSmokeTestFinal {
                event: input.event.clone(),
                quality_config: input.quality_config.clone(),
                final_transcript_at: input.final_transcript_at,
                generation: 0,
            });
        if entry.generation == 0 {
            entry.event = input.event;
        } else {
            entry.event = merge_smoke_test_final_events(&entry.event, input.event);
        }
        if entry.quality_config.is_none() {
            entry.quality_config = input.quality_config;
        }
        entry.generation = entry.generation.saturating_add(1);
        entry.generation
    };

    let state = state.clone();
    let media_registry = media_registry.clone();
    let runtime = runtime.clone();
    let gateway_call_id = gateway_call_id.to_string();
    let debounce = input.debounce;
    tokio::spawn(async move {
        sleep(debounce).await;
        let Some(pending) =
            take_ready_smoke_test_final(&runtime, &gateway_call_id, generation).await
        else {
            return;
        };
        if !runtime.handler_enabled() {
            return;
        }
        emit_smoke_final_debounce_span(&state, &gateway_call_id, &pending, debounce).await;
        if let Err(error) = dispatch_final_transcript_to_handler(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            pending.event,
            pending.quality_config.as_ref(),
            pending.final_transcript_at,
        )
        .await
        {
            let error = format!("{error:#}");
            state
                .write()
                .await
                .record_conversation_failed(&gateway_call_id, error.clone());
            tracing::warn!(
                gateway_call_id,
                error,
                "conversation.smoke_test.debounce_failed"
            );
        }
    });
}

async fn take_ready_smoke_test_final(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> Option<PendingSmokeTestFinal> {
    let mut pending = runtime.smoke_test_pending_finals.lock().await;
    match pending.get(gateway_call_id) {
        Some(entry) if entry.generation == generation => pending.remove(gateway_call_id),
        _ => None,
    }
}

fn merge_smoke_test_final_events(
    existing: &TranscriptEvent,
    next: TranscriptEvent,
) -> TranscriptEvent {
    let merged_text = merge_smoke_test_finals(existing.text(), next.text());
    event_with_trimmed_text(next, merged_text)
}

fn merge_smoke_test_finals(existing: &str, next: &str) -> String {
    let existing = existing.trim();
    let next = next.trim();
    match (existing.is_empty(), next.is_empty()) {
        (_, true) => existing.to_string(),
        (true, false) => next.to_string(),
        (false, false) => format!("{existing} {next}"),
    }
}

async fn emit_smoke_final_debounce_span(
    state: &SharedState,
    gateway_call_id: &str,
    pending: &PendingSmokeTestFinal,
    debounce: Duration,
) {
    let (config_id, redaction_mode) = if let Some(config) = pending.quality_config.as_ref() {
        (config.config_id(), config.logging.redaction_mode)
    } else {
        let guard = state.read().await;
        (
            guard.quality.config_id.clone(),
            guard.quality.config.logging.redaction_mode,
        )
    };
    let payload = serde_json::json!({
        "debounce_ms": debounce.as_millis() as u64,
        "text_chars": pending.event.text().chars().count(),
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
            span_name: "conversation.smoke_final_debounce",
            category: "intentional_delay",
            duration: pending.final_transcript_at.elapsed(),
            critical_path: true,
            concurrent: false,
            payload,
        },
    );
}

async fn dispatch_final_transcript_to_handler(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
    quality_config: Option<&VoiceQualityConfig>,
    final_transcript_at: Instant,
) -> anyhow::Result<()> {
    if !runtime.handler_enabled() {
        return Ok(());
    }
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
    state
        .write()
        .await
        .record_conversation_user_turn(gateway_call_id, transcript_text);

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
    apply_conversation_command_with_timing(
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
        Some(final_transcript_at),
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
    endpoint_merge_window_ms: u64,
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
    let (barge_in, config_id, redaction_mode, endpoint_merge_window_ms) = quality_config
        .map(|quality_config| {
            (
                quality_config.barge_in.clone(),
                quality_config.config_id(),
                quality_config.logging.redaction_mode,
                quality_config.endpoint.merge_window_ms,
            )
        })
        .unwrap_or_else(|| {
            (
                guard.quality.config.barge_in.clone(),
                guard.quality.config_id.clone(),
                guard.quality.config.logging.redaction_mode,
                guard.quality.config.endpoint.merge_window_ms,
            )
        });
    Some(ConversationSnapshot {
        attached: call.conversation.attached,
        mode: call.conversation.mode,
        call_control_id: call.ids.call_control_id.clone(),
        barge_in,
        config_id,
        redaction_mode,
        endpoint_merge_window_ms,
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

#[cfg(test)]
async fn apply_conversation_command(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    target: ConversationCommandTarget,
    command: ConversationCommand,
) -> anyhow::Result<()> {
    apply_conversation_command_with_timing(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        target,
        command,
        None,
    )
    .await
}

async fn apply_conversation_command_with_timing(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    target: ConversationCommandTarget,
    command: ConversationCommand,
    turn_finalized_at: Option<Instant>,
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
                            .record_conversation_proposal(gateway_call_id, response_text.clone());
                        tracing::info!(
                            gateway_call_id,
                            "conversation.say.deferred_barge_in_disabled"
                        );
                        spawn_deferred_conversation_say(
                            state,
                            media_registry,
                            runtime,
                            gateway_call_id,
                            response_text,
                            turn_finalized_at,
                        )
                        .await;
                        return Ok(());
                    }
                    queue_conversation_speech(
                        state,
                        media_registry,
                        runtime,
                        gateway_call_id,
                        response_text,
                        SpeechConflictPolicy::CancelAndReplace,
                        turn_finalized_at,
                    )
                    .await;
                    Ok(())
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

async fn spawn_deferred_conversation_say(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    response_text: String,
    turn_finalized_at: Option<Instant>,
) {
    let generation = next_deferred_say_generation(runtime, gateway_call_id).await;
    let state = state.clone();
    let media_registry = media_registry.clone();
    let runtime = runtime.clone();
    let gateway_call_id = gateway_call_id.to_string();
    tokio::spawn(async move {
        wait_and_queue_deferred_conversation_say(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            generation,
            response_text,
            turn_finalized_at,
        )
        .await;
    });
}

async fn next_deferred_say_generation(runtime: &ConversationRuntime, gateway_call_id: &str) -> u64 {
    let mut generations = runtime.deferred_say_generations.lock().await;
    let generation = generations.entry(gateway_call_id.to_string()).or_insert(0);
    *generation = generation.saturating_add(1);
    *generation
}

async fn is_latest_deferred_say_generation(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> bool {
    runtime
        .deferred_say_generations
        .lock()
        .await
        .get(gateway_call_id)
        .is_some_and(|current| *current == generation)
}

async fn take_latest_deferred_say_generation(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> bool {
    let mut generations = runtime.deferred_say_generations.lock().await;
    match generations.get(gateway_call_id) {
        Some(current) if *current == generation => {
            generations.remove(gateway_call_id);
            true
        }
        _ => false,
    }
}

async fn wait_and_queue_deferred_conversation_say(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
    response_text: String,
    turn_finalized_at: Option<Instant>,
) {
    let timeout_ms = state
        .read()
        .await
        .quality
        .config
        .text_call
        .playback_wait_timeout_ms;
    let started = Instant::now();
    while media_registry
        .active_speech_playback_id(gateway_call_id)
        .await
        .is_some()
    {
        if !is_latest_deferred_say_generation(runtime, gateway_call_id, generation).await {
            tracing::debug!(
                gateway_call_id,
                generation,
                "conversation.say.deferred_superseded"
            );
            return;
        }
        if started.elapsed() >= Duration::from_millis(timeout_ms) {
            let error = format!(
                "deferred conversation response timed out after {timeout_ms}ms waiting for active playback"
            );
            state
                .write()
                .await
                .record_conversation_failed(gateway_call_id, error.clone());
            tracing::warn!(gateway_call_id, error, "conversation.say.deferred_timeout");
            return;
        }
        sleep(Duration::from_millis(50)).await;
    }
    if !take_latest_deferred_say_generation(runtime, gateway_call_id, generation).await {
        tracing::debug!(
            gateway_call_id,
            generation,
            "conversation.say.deferred_superseded"
        );
        return;
    }

    queue_conversation_speech(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        response_text,
        SpeechConflictPolicy::Reject,
        turn_finalized_at,
    )
    .await;
}

async fn queue_conversation_speech(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    response_text: String,
    conflict_policy: SpeechConflictPolicy,
    turn_finalized_at: Option<Instant>,
) {
    let queued = speech::queue_speech_with_request(
        state,
        media_registry,
        &runtime.tts,
        SpeechQueueRequest {
            tts_backend: LiveTtsBackend::default(),
            gateway_call_id: gateway_call_id.to_string(),
            text: response_text.clone(),
            source_label: "conversation say".to_string(),
            conflict_policy,
            turn_finalized_at,
        },
    )
    .await
    .with_context(|| format!("queue conversation response for {gateway_call_id}"));
    match queued {
        Ok(queued) => {
            {
                let mut guard = state.write().await;
                if let Some(replaced_playback_id) = &queued.replaced_playback_id {
                    guard.record_conversation_interrupted(gateway_call_id, replaced_playback_id);
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
        }
        Err(error) => {
            let error = format!("{error:#}");
            state
                .write()
                .await
                .record_conversation_failed(gateway_call_id, error.clone());
            tracing::warn!(gateway_call_id, error, "conversation.say.failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::SpeechCancelToken;
    use crate::operator::state::{CallStatus, ConversationStatus, TelnyxIds, shared_state};
    use crate::tts::{OutboundTtsFactory, PIPER_SAMPLE_RATE_HZ, TtsAudio, TtsRegistry};
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    fn test_runtime() -> ConversationRuntime {
        ConversationRuntime::new_with_handler_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            default_conversation_handler(),
            true,
            true,
        )
    }

    fn test_runtime_with_tts() -> ConversationRuntime {
        let tts = Arc::new(TtsRegistry::new(
            Arc::new(StaticTtsFactory),
            Arc::new(StaticTtsFactory),
        ));
        ConversationRuntime::new_with_handler_options(
            TelnyxClient::new("https://api.example.test", None, true),
            tts,
            default_conversation_handler(),
            true,
            true,
        )
    }

    #[derive(Clone, Debug)]
    struct StaticTtsFactory;

    #[async_trait]
    impl OutboundTtsFactory for StaticTtsFactory {
        async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
            Ok(vec![TtsAudio::new(
                vec![1_000; 2_205],
                PIPER_SAMPLE_RATE_HZ,
            )?])
        }

        fn label(&self) -> &'static str {
            "static-test-tts"
        }
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
    struct PrefixConversationHandler;

    #[async_trait]
    impl ConversationHandler for PrefixConversationHandler {
        async fn on_transcript(
            &self,
            event: TranscriptEvent,
            _context: &mut CallContext,
        ) -> Result<ConversationCommand, VoiceAppError> {
            Ok(ConversationCommand::Say {
                text: format!("agent: {}", event.text()),
            })
        }
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

    async fn wait_for_smoke_test_final() {
        sleep(Duration::from_millis(
            VoiceQualityConfig::default().endpoint.merge_window_ms + 30,
        ))
        .await;
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
    async fn generic_handler_dispatches_without_smoke_final_coalescing() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = ConversationRuntime::new_with_handler_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            Arc::new(PrefixConversationHandler),
            true,
            false,
        );
        assert_eq!(runtime.handler_label(), "handler");

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
        .expect("generic handler should dispatch immediately");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("hello"));
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("agent: hello")
        );
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
        assert!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .is_none()
        );
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
        assert!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .is_none()
        );
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
        let (tx, mut rx) = mpsc::channel(16);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime_with_tts();
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
        wait_for_smoke_test_final().await;

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
        drop(guard);

        media_registry
            .finish_speech(&gateway_call_id, "tts_test")
            .await;
        let command = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("deferred speech should enqueue after prior playback finishes")
            .expect("deferred speech should emit a media command");
        match command {
            crate::media::OutboundMediaCommand::Frame(frame) => {
                assert_ne!(frame.playback_id, "tts_test");
            }
            other => panic!("expected deferred frame, got {other:?}"),
        }
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Speaking);
    }

    #[tokio::test]
    async fn disabled_barge_in_deferred_auto_say_uses_latest_final_turn() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(16);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime_with_tts();
        {
            let mut guard = state.write().await;
            guard.quality.config.set_barge_in_enabled(false);
            guard.quality.config.set_endpoint_merge_window_ms(0);
            guard.quality.config_id = guard.quality.config.config_id();
        }

        for text in ["first", "second"] {
            handle_transcript_event(
                &state,
                &media_registry,
                &runtime,
                &gateway_call_id,
                TranscriptEvent::Final {
                    text: text.to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                None,
            )
            .await
            .expect("disabled barge-in should defer overlapping final turn");
        }

        assert!(!cancel.is_canceled());
        media_registry
            .finish_speech(&gateway_call_id, "tts_test")
            .await;
        timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("latest deferred speech should enqueue")
            .expect("latest deferred speech should emit a media command");
        sleep(Duration::from_millis(100)).await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Speaking);
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("second"));
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("I heard: second")
        );
        assert!(call.conversation.last_error.is_none());
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
    async fn smoke_test_debounce_merges_adjacent_final_fragments() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = test_runtime();
        let media_registry = SharedMediaRegistry::default();

        for text in [
            "Yeah, this is not",
            "the last fragment or frame didn't come through still.",
        ] {
            handle_transcript_event(
                &state,
                &media_registry,
                &runtime,
                &gateway_call_id,
                TranscriptEvent::Final {
                    text: text.to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                None,
            )
            .await
            .expect("final fragment should be accepted");
        }
        wait_for_smoke_test_final().await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("Yeah, this is not the last fragment or frame didn't come through still.")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some(
                "I heard: Yeah, this is not the last fragment or frame didn't come through still."
            )
        );
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
        wait_for_smoke_test_final().await;

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
        wait_for_smoke_test_final().await;

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
