use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context};
use motlie_agent::voice::turn_batching::{Accumulating, Prompt, TurnBatchReset};
use motlie_voice::app::{ConversationCommand, TranscriptEvent};
use motlie_voice::telephony::CallAction;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

use crate::call_control::TelnyxClient;
use crate::early_response::{EarlyResponseCancelReason, EarlyResponseEvent};
use crate::media::{SharedMediaRegistry, SpeechClearReason};
use crate::operator::state::{
    CallStatus, ConversationMode, LogLevel, QualitySpanEmission, SharedState,
};
use crate::processors::ConversationProcessorHost;
pub use crate::processors::{
    ConversationCommittedTurn, ConversationProcessorInput, ConversationProcessorKind,
    ConversationProcessorOutput,
};
use crate::quality::{
    BargeInQualityConfig, EndpointQualityConfig, RedactionMode, VoiceQualityConfig,
};
use crate::speech;
use crate::speech::{SpeechConflictPolicy, SpeechQueueRequest};
use crate::tts::SharedTtsRegistry;

const PARTIAL_BARGE_IN_MIN_CHARS: usize = 3;

#[derive(Clone)]
pub struct ConversationRuntime {
    telnyx: TelnyxClient,
    tts: SharedTtsRegistry,
    processor_enabled: Arc<AtomicBool>,
    final_coalescing_enabled: Arc<AtomicBool>,
    barge_in_enabled: Arc<AtomicBool>,
    pending_conversation_finals: Arc<Mutex<HashMap<String, PendingConversationFinal>>>,
    deferred_say_generations: Arc<Mutex<HashMap<String, u64>>>,
    processor_hosts: Arc<Mutex<HashMap<String, ConversationProcessorHost>>>,
}

impl ConversationRuntime {
    pub fn new(telnyx: TelnyxClient, tts: SharedTtsRegistry, processor_enabled: bool) -> Self {
        Self::new_with_processor_options(telnyx, tts, processor_enabled, false)
    }

    pub fn new_with_processor_options(
        telnyx: TelnyxClient,
        tts: SharedTtsRegistry,
        processor_enabled: bool,
        final_coalescing_enabled: bool,
    ) -> Self {
        Self {
            telnyx,
            tts,
            processor_enabled: Arc::new(AtomicBool::new(processor_enabled)),
            final_coalescing_enabled: Arc::new(AtomicBool::new(final_coalescing_enabled)),
            barge_in_enabled: Arc::new(AtomicBool::new(true)),
            pending_conversation_finals: Arc::new(Mutex::new(HashMap::new())),
            deferred_say_generations: Arc::new(Mutex::new(HashMap::new())),
            processor_hosts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn processor_enabled(&self) -> bool {
        self.processor_enabled.load(Ordering::SeqCst)
    }

    pub fn set_processor_enabled(&self, enabled: bool) {
        self.processor_enabled.store(enabled, Ordering::SeqCst);
    }

    pub fn final_coalescing_enabled(&self) -> bool {
        self.final_coalescing_enabled.load(Ordering::SeqCst)
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

    pub fn processor_label(&self) -> &'static str {
        if !self.processor_enabled() {
            "disabled"
        } else if self.final_coalescing_enabled() {
            "smoke-test"
        } else {
            "processor"
        }
    }

    pub fn tts_registry(&self) -> SharedTtsRegistry {
        self.tts.clone()
    }

    async fn process_processor_input(
        &self,
        gateway_call_id: &str,
        kind: ConversationProcessorKind,
        input: ConversationProcessorInput,
    ) -> Option<ConversationProcessorOutput> {
        let mut hosts = self.processor_hosts.lock().await;
        let host = hosts
            .entry(gateway_call_id.to_string())
            .or_insert_with(|| ConversationProcessorHost::new(kind.clone()));
        if !host.is_kind(&kind) {
            *host = ConversationProcessorHost::new(kind);
        }
        host.process_input(input)
    }

    async fn complete_pending_processor_batch(
        &self,
        gateway_call_id: &str,
        batch_id: &str,
        epoch: u64,
    ) -> Option<ConversationProcessorOutput> {
        let mut hosts = self.processor_hosts.lock().await;
        hosts
            .get_mut(gateway_call_id)
            .and_then(|host| host.complete_pending(batch_id, epoch))
    }
}

#[derive(Clone, Debug)]
struct ConversationFinalInput {
    event: TranscriptEvent,
    quality_config: VoiceQualityConfig,
    final_transcript_at: Instant,
    debounce: Duration,
    turn_id: Option<String>,
}

#[derive(Clone, Debug)]
struct PendingConversationFinal {
    event: TranscriptEvent,
    quality_config: VoiceQualityConfig,
    final_transcript_at: Instant,
    latest_final_transcript_at: Instant,
    generation: u64,
    turn_ids: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct ConversationTurnContext {
    finalized_at: Option<Instant>,
    latest_finalized_at: Option<Instant>,
    turn_id: Option<String>,
    coalesced_turn_ids: Vec<String>,
}

pub async fn handle_transcript_event(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
    quality_config: Option<&VoiceQualityConfig>,
) -> anyhow::Result<()> {
    handle_transcript_event_with_turn(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        event,
        quality_config,
        None,
    )
    .await
}

pub async fn handle_transcript_event_with_turn(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
    quality_config: Option<&VoiceQualityConfig>,
    turn_id: Option<&str>,
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
                runtime,
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
            runtime,
            gateway_call_id,
            BargeInTrigger::Final,
            snapshot.config_id.clone(),
            snapshot.redaction_mode,
        )
        .await?;
    }

    if !runtime.processor_enabled() {
        state
            .write()
            .await
            .record_conversation_user_turn(gateway_call_id, transcript_text.clone());
        state
            .write()
            .await
            .record_conversation_idle(gateway_call_id);
        tracing::debug!(
            gateway_call_id,
            "conversation.processor.disabled_for_final_transcript"
        );
        return Ok(());
    }

    let event = event_with_trimmed_text(event, transcript_text);
    let quality_config = snapshot.quality_config.clone();
    if snapshot.endpoint_merge_window_ms == 0 || !runtime.final_coalescing_enabled() {
        return dispatch_final_transcript_to_processor(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            event,
            Some(&quality_config),
            ConversationTurnContext {
                finalized_at: Some(final_transcript_at),
                latest_finalized_at: Some(final_transcript_at),
                turn_id: turn_id.map(str::to_string),
                coalesced_turn_ids: turn_id.map(|id| vec![id.to_string()]).unwrap_or_default(),
            },
        )
        .await;
    }

    schedule_conversation_final(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        ConversationFinalInput {
            event,
            quality_config,
            final_transcript_at,
            debounce: conversation_final_debounce(snapshot.endpoint_merge_window_ms),
            turn_id: turn_id.map(str::to_string),
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

fn conversation_final_debounce(configured_ms: u64) -> Duration {
    Duration::from_millis(configured_ms)
}

async fn schedule_conversation_final(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    input: ConversationFinalInput,
) {
    let generation = {
        let mut pending = runtime.pending_conversation_finals.lock().await;
        let entry = pending
            .entry(gateway_call_id.to_string())
            .or_insert_with(|| PendingConversationFinal {
                event: input.event.clone(),
                quality_config: input.quality_config.clone(),
                final_transcript_at: input.final_transcript_at,
                latest_final_transcript_at: input.final_transcript_at,
                generation: 0,
                turn_ids: input.turn_id.iter().cloned().collect(),
            });
        if entry.generation == 0 {
            entry.event = input.event;
        } else {
            entry.event = merge_conversation_final_events(&entry.event, input.event);
        }
        if let Some(turn_id) = input.turn_id {
            if !entry.turn_ids.contains(&turn_id) {
                entry.turn_ids.push(turn_id);
            }
        }
        entry.latest_final_transcript_at = input.final_transcript_at;
        entry.generation = entry.generation.saturating_add(1);
        entry.generation
    };

    let state = state.clone();
    let media_registry = media_registry.clone();
    let runtime = runtime.clone();
    let gateway_call_id = gateway_call_id.to_string();
    let debounce = input.debounce;
    tokio::spawn(async move {
        let mut generation = generation;
        let mut delay = debounce;
        loop {
            sleep(delay).await;
            if !runtime.processor_enabled() {
                let _ = take_ready_conversation_final(&runtime, &gateway_call_id, generation).await;
                return;
            }
            if media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .is_some()
            {
                let Some((next_generation, next_delay)) =
                    defer_ready_conversation_final(&runtime, &gateway_call_id, generation).await
                else {
                    return;
                };
                generation = next_generation;
                delay = next_delay;
                tracing::debug!(
                    gateway_call_id,
                    generation,
                    "conversation.final_debounce.held_for_playback"
                );
                continue;
            }

            let hold_reason = {
                let pending = runtime.pending_conversation_finals.lock().await;
                let Some(entry) = pending.get(&gateway_call_id) else {
                    return;
                };
                if entry.generation != generation {
                    return;
                }
                conversation_final_hold_reason(entry)
            };
            if let Some(reason) = hold_reason {
                let Some((next_generation, next_delay)) =
                    defer_ready_conversation_final(&runtime, &gateway_call_id, generation).await
                else {
                    return;
                };
                generation = next_generation;
                delay = next_delay;
                tracing::debug!(
                    gateway_call_id,
                    generation,
                    reason,
                    "conversation.final_debounce.held_for_incomplete_tail"
                );
                continue;
            }

            let Some(pending) =
                take_ready_conversation_final(&runtime, &gateway_call_id, generation).await
            else {
                return;
            };
            emit_conversation_final_debounce_span(&state, &gateway_call_id, &pending, debounce)
                .await;
            if let Err(error) = dispatch_final_transcript_to_processor(
                &state,
                &media_registry,
                &runtime,
                &gateway_call_id,
                pending.event,
                Some(&pending.quality_config),
                ConversationTurnContext {
                    finalized_at: Some(pending.final_transcript_at),
                    latest_finalized_at: Some(pending.latest_final_transcript_at),
                    turn_id: pending.turn_ids.first().cloned(),
                    coalesced_turn_ids: pending.turn_ids,
                },
            )
            .await
            {
                let error = format!("{error:#}");
                state
                    .write()
                    .await
                    .record_conversation_failed(&gateway_call_id, error.clone());
                tracing::warn!(gateway_call_id, error, "conversation.final_debounce.failed");
            }
            return;
        }
    });
}

async fn defer_ready_conversation_final(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> Option<(u64, Duration)> {
    let mut pending = runtime.pending_conversation_finals.lock().await;
    match pending.get_mut(gateway_call_id) {
        Some(entry) if entry.generation == generation => {
            entry.generation = entry.generation.saturating_add(1);
            let delay = Duration::from_millis(
                entry
                    .quality_config
                    .endpoint
                    .conversation_playback_hold_poll_ms,
            );
            Some((entry.generation, delay))
        }
        _ => None,
    }
}

async fn take_ready_conversation_final(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> Option<PendingConversationFinal> {
    let mut pending = runtime.pending_conversation_finals.lock().await;
    match pending.get(gateway_call_id) {
        Some(entry) if entry.generation == generation => pending.remove(gateway_call_id),
        _ => None,
    }
}

fn merge_conversation_final_events(
    existing: &TranscriptEvent,
    next: TranscriptEvent,
) -> TranscriptEvent {
    let merged_text = merge_conversation_finals(existing.text(), next.text());
    event_with_trimmed_text(next, merged_text)
}

fn merge_conversation_finals(existing: &str, next: &str) -> String {
    let existing = existing.trim();
    let next = next.trim();
    match (existing.is_empty(), next.is_empty()) {
        (_, true) => existing.to_string(),
        (true, false) => next.to_string(),
        (false, false) => format!("{existing} {next}"),
    }
}

fn conversation_final_hold_reason(pending: &PendingConversationFinal) -> Option<&'static str> {
    let endpoint = &pending.quality_config.endpoint;
    if pending.final_transcript_at.elapsed()
        >= Duration::from_millis(endpoint.conversation_incomplete_tail_hold_ms)
    {
        return None;
    }
    endpoint
        .conversation_incomplete_tail_reason(pending.event.text())
        .or_else(|| conversation_low_confidence_hold_reason(&pending.event, endpoint))
}

fn conversation_low_confidence_hold_reason(
    event: &TranscriptEvent,
    endpoint: &EndpointQualityConfig,
) -> Option<&'static str> {
    let text = event.text().trim();
    if !endpoint.conversation_low_confidence_hold_allowed(text) {
        return None;
    }
    let confidence = latest_transcript_confidence(event)?;
    (confidence < endpoint.conversation_low_confidence_threshold()).then_some("low_confidence_tail")
}

fn latest_transcript_confidence(event: &TranscriptEvent) -> Option<f32> {
    let update = match event {
        TranscriptEvent::Partial { update, .. } | TranscriptEvent::Final { update, .. } => update,
    };
    update.segments.iter().rev().find_map(|segment| {
        segment
            .confidence
            .filter(|confidence| confidence.is_finite() && *confidence >= 0.0 && *confidence <= 1.0)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessorOutputResult {
    CommandApplied,
    NoCommand,
    Stop,
}

async fn handle_conversation_processor_output(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    target: &ConversationCommandTarget,
    turn_context: &ConversationTurnContext,
    output: ConversationProcessorOutput,
) -> anyhow::Result<ProcessorOutputResult> {
    match output {
        ConversationProcessorOutput::Command(command) => {
            apply_conversation_command_with_timing(
                state,
                media_registry,
                runtime,
                gateway_call_id,
                target.clone(),
                command,
                turn_context.clone(),
            )
            .await?;
            Ok(ProcessorOutputResult::CommandApplied)
        }
        ConversationProcessorOutput::PromptComplete(prompt) => {
            emit_turn_batch_prompt_complete_lifecycle(state, gateway_call_id, &prompt).await;
            apply_conversation_command_with_timing(
                state,
                media_registry,
                runtime,
                gateway_call_id,
                target.clone(),
                ConversationCommand::Say { text: prompt.text },
                turn_context.clone(),
            )
            .await?;
            Ok(ProcessorOutputResult::CommandApplied)
        }
        ConversationProcessorOutput::Accumulating(batch) => {
            emit_turn_batch_accumulated_lifecycle(state, gateway_call_id, &batch).await;
            schedule_turn_batch_timeout(
                state,
                media_registry,
                runtime,
                gateway_call_id,
                batch.batch_id,
                batch.epoch,
                batch.deadline_ms,
            );
            Ok(ProcessorOutputResult::NoCommand)
        }
        ConversationProcessorOutput::Reset(reset) => {
            emit_turn_batch_reset_lifecycle(state, gateway_call_id, &reset).await;
            Ok(ProcessorOutputResult::NoCommand)
        }
        ConversationProcessorOutput::EarlyResponse(_) => {
            tracing::warn!(
                gateway_call_id,
                "conversation.processor.early_response_output_ignored_for_committed_turn"
            );
            Ok(ProcessorOutputResult::NoCommand)
        }
        ConversationProcessorOutput::CommittedSpeech(_) => {
            tracing::warn!(
                gateway_call_id,
                "conversation.processor.committed_speech_output_ignored_for_committed_turn"
            );
            Ok(ProcessorOutputResult::NoCommand)
        }
        ConversationProcessorOutput::Error(error) => {
            state
                .write()
                .await
                .record_conversation_failed(gateway_call_id, error.clone());
            tracing::warn!(gateway_call_id, error, "conversation.processor.failed");
            Ok(ProcessorOutputResult::Stop)
        }
    }
}

fn schedule_turn_batch_timeout(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    batch_id: String,
    epoch: u64,
    deadline_ms: u64,
) {
    if deadline_ms == 0 {
        return;
    }
    let state = state.clone();
    let media_registry = media_registry.clone();
    let runtime = runtime.clone();
    let gateway_call_id = gateway_call_id.to_string();
    tokio::spawn(async move {
        sleep(Duration::from_millis(deadline_ms)).await;
        if !runtime.processor_enabled() {
            return;
        }
        let Some(output) = runtime
            .complete_pending_processor_batch(&gateway_call_id, &batch_id, epoch)
            .await
        else {
            return;
        };
        let Some(snapshot) = conversation_snapshot(&state, &gateway_call_id, None).await else {
            return;
        };
        if !snapshot.attached {
            return;
        }
        let target = ConversationCommandTarget {
            mode: snapshot.mode,
            call_control_id: snapshot.call_control_id,
            barge_in: snapshot.barge_in,
        };
        if let Err(error) = handle_conversation_processor_output(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            &target,
            &ConversationTurnContext::default(),
            output,
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
                batch_id,
                epoch,
                error,
                "conversation.processor.turn_batch_timeout.failed"
            );
        }
    });
}

async fn emit_conversation_final_debounce_span(
    state: &SharedState,
    gateway_call_id: &str,
    pending: &PendingConversationFinal,
    debounce: Duration,
) {
    let config_id = pending.quality_config.config_id();
    let redaction_mode = pending.quality_config.logging.redaction_mode;
    let payload = serde_json::json!({
        "debounce_ms": debounce.as_millis() as u64,
        "text_chars": pending.event.text().chars().count(),
        "turn_id": pending.turn_ids.first(),
        "coalesced_turn_count": pending.turn_ids.len(),
        "coalesced_turn_ids": pending.turn_ids.as_slice(),
        "latest_final_to_dispatch_ms": pending.latest_final_transcript_at.elapsed().as_millis() as u64,
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
            span_name: "conversation.final_debounce",
            category: "intentional_delay",
            duration: pending.final_transcript_at.elapsed(),
            critical_path: true,
            concurrent: false,
            payload,
        },
    );
}

async fn dispatch_final_transcript_to_processor(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
    quality_config: Option<&VoiceQualityConfig>,
    turn_context: ConversationTurnContext,
) -> anyhow::Result<()> {
    if !runtime.processor_enabled() {
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
        .record_conversation_user_turn(gateway_call_id, transcript_text.clone());

    let processor_input = ConversationProcessorInput::CommittedTurn(ConversationCommittedTurn {
        call_id: gateway_call_id.to_string(),
        turn_id: turn_context.turn_id.clone(),
        text: transcript_text,
        event,
    });
    let target = ConversationCommandTarget {
        mode: snapshot.mode,
        call_control_id: snapshot.call_control_id,
        barge_in: snapshot.barge_in,
    };
    let mut saw_command = false;
    if let Some(output) = runtime
        .process_processor_input(gateway_call_id, snapshot.processor, processor_input)
        .await
    {
        match handle_conversation_processor_output(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            &target,
            &turn_context,
            output,
        )
        .await?
        {
            ProcessorOutputResult::CommandApplied => saw_command = true,
            ProcessorOutputResult::NoCommand => {}
            ProcessorOutputResult::Stop => return Ok(()),
        }
    }
    if !saw_command {
        apply_conversation_command_with_timing(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            target,
            ConversationCommand::Noop,
            turn_context,
        )
        .await?;
    }
    Ok(())
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
        _runtime,
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
    runtime: &ConversationRuntime,
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
    reset_turn_batch_for_barge_in(state, runtime, gateway_call_id).await;
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

async fn reset_turn_batch_for_barge_in(
    state: &SharedState,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
) {
    let processor = {
        let guard = state.read().await;
        guard
            .calls
            .get(gateway_call_id)
            .map(|call| call.conversation.processor.clone())
            .unwrap_or_default()
    };
    if !matches!(
        processor,
        ConversationProcessorKind::TurnBatchedIdentity { .. }
    ) {
        return;
    }
    let input = ConversationProcessorInput::EarlyResponse(EarlyResponseEvent::Canceled {
        provisional_turn_id: "barge_in".to_string(),
        call_id: gateway_call_id.to_string(),
        utterance_id: "barge_in".to_string(),
        generation: 0,
        reason: EarlyResponseCancelReason::CallerBargeIn,
    });
    if let Some(ConversationProcessorOutput::Reset(reset)) = runtime
        .process_processor_input(gateway_call_id, processor, input)
        .await
    {
        emit_turn_batch_reset_lifecycle(state, gateway_call_id, &reset).await;
    }
}

async fn emit_turn_batch_accumulated_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    batch: &Accumulating,
) {
    let accumulated_turn_count = batch.source_turn_ids.len();
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "turn-accumulated",
        batch_id = %batch.batch_id,
        epoch = batch.epoch,
        accumulated_turn_count,
        target_turn_count = batch.target_turn_count,
        deadline_ms = batch.deadline_ms,
        "conversation.processor.turn_batch.turn_accumulated"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "batch_id": &batch.batch_id,
        "epoch": batch.epoch,
        "accumulated_turn_count": accumulated_turn_count,
        "target_turn_count": batch.target_turn_count,
        "source_turn_ids": &batch.source_turn_ids,
        "deadline_ms": batch.deadline_ms,
    }));
    state.write().await.emit_quality_turn_batch_lifecycle(
        gateway_call_id,
        "turn-accumulated",
        payload,
    );
}

async fn emit_turn_batch_prompt_complete_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    prompt: &Prompt,
) {
    let joined_source_turn_count = prompt.source_turn_ids.len();
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "prompt-complete",
        batch_id = %prompt.batch_id,
        epoch = prompt.epoch,
        joined_source_turn_count,
        response_turn_id = %prompt.response_turn_id,
        prompt_chars = prompt.text.chars().count(),
        "conversation.processor.turn_batch.prompt_complete"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "batch_id": &prompt.batch_id,
        "epoch": prompt.epoch,
        "joined_source_turn_count": joined_source_turn_count,
        "source_turn_ids": &prompt.source_turn_ids,
        "response_turn_id": &prompt.response_turn_id,
        "prompt_words": prompt.text.split_whitespace().count(),
        "prompt_chars": prompt.text.chars().count(),
    }));
    state.write().await.emit_quality_turn_batch_lifecycle(
        gateway_call_id,
        "prompt-complete",
        payload,
    );
}

async fn emit_turn_batch_reset_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    reset: &TurnBatchReset,
) {
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "batch-reset",
        reason = reset.reason.as_str(),
        epoch = reset.epoch,
        batch_id = ?reset.batch_id.as_deref(),
        "conversation.processor.turn_batch.batch_reset"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "reason": reset.reason.as_str(),
        "batch_id": &reset.batch_id,
        "epoch": reset.epoch,
    }));
    state
        .write()
        .await
        .emit_quality_turn_batch_lifecycle(gateway_call_id, "batch-reset", payload);
}

fn turn_batch_payload(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(payload) => payload,
        _ => serde_json::Map::new(),
    }
}

struct ConversationSnapshot {
    attached: bool,
    mode: ConversationMode,
    processor: ConversationProcessorKind,
    call_control_id: String,
    barge_in: BargeInQualityConfig,
    config_id: String,
    redaction_mode: RedactionMode,
    endpoint_merge_window_ms: u64,
    quality_config: VoiceQualityConfig,
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
    let effective_quality = quality_config
        .cloned()
        .unwrap_or_else(|| guard.quality.config.clone());
    let barge_in = effective_quality.barge_in.clone();
    let config_id = effective_quality.config_id();
    let redaction_mode = effective_quality.logging.redaction_mode;
    let endpoint_merge_window_ms = effective_quality.endpoint.merge_window_ms;
    Some(ConversationSnapshot {
        attached: call.conversation.attached,
        mode: call.conversation.mode,
        processor: call.conversation.processor.clone(),
        call_control_id: call.ids.call_control_id.clone(),
        barge_in,
        config_id,
        redaction_mode,
        endpoint_merge_window_ms,
        quality_config: effective_quality,
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
        ConversationTurnContext::default(),
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
    turn_context: ConversationTurnContext,
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
                            turn_context.clone(),
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
                        turn_context.clone(),
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
    turn_context: ConversationTurnContext,
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
            turn_context,
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
    turn_context: ConversationTurnContext,
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
            if record_conversation_failed_unless_terminal(
                state,
                gateway_call_id,
                error.clone(),
                "conversation.say.deferred_timeout_after_call_end",
            )
            .await
            {
                return;
            }
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
        turn_context,
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
    turn_context: ConversationTurnContext,
) {
    let speech_output = {
        let guard = state.read().await;
        guard
            .calls
            .get(gateway_call_id)
            .map(|call| call.speech_output)
            .unwrap_or_else(|| {
                crate::operator::state::SpeechOutputConfig::from_quality(
                    guard.conversation_tts_backend,
                    &guard.quality.config.tts,
                )
            })
    };
    let queued = speech::queue_speech_with_request(
        state,
        media_registry,
        &runtime.tts,
        SpeechQueueRequest {
            tts_backend: speech_output.tts_backend,
            gateway_call_id: gateway_call_id.to_string(),
            text: response_text.clone(),
            source_label: "conversation say".to_string(),
            conflict_policy,
            turn_finalized_at: turn_context.finalized_at,
            latest_turn_finalized_at: turn_context.latest_finalized_at,
            turn_id: turn_context.turn_id,
            coalesced_turn_ids: turn_context.coalesced_turn_ids,
            source_asr_session_ids: Vec::new(),
            source_utterance_ids: Vec::new(),
            prebuffer_chunks_override: None,
            speech_output: Some(speech_output),
            metadata: crate::operator::state::QualityPlaybackMetadata::default(),
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
                tts_backend = speech_output.tts_backend.label(),
                "conversation.say.queued"
            );
        }
        Err(error) => {
            let error = format!("{error:#}");
            if record_conversation_failed_unless_terminal(
                state,
                gateway_call_id,
                error.clone(),
                "conversation.say.canceled_after_call_end",
            )
            .await
            {
                return;
            }
            tracing::warn!(gateway_call_id, error, "conversation.say.failed");
        }
    }
}

async fn record_conversation_failed_unless_terminal(
    state: &SharedState,
    gateway_call_id: &str,
    error: String,
    terminal_event: &'static str,
) -> bool {
    let call_status = {
        let guard = state.read().await;
        guard.calls.get(gateway_call_id).map(|call| call.status)
    };
    if matches!(
        call_status,
        None | Some(CallStatus::Ended | CallStatus::Failed)
    ) {
        if matches!(call_status, Some(CallStatus::Ended)) {
            state
                .write()
                .await
                .record_conversation_idle(gateway_call_id);
        }
        tracing::info!(gateway_call_id, error, terminal_event);
        true
    } else {
        state
            .write()
            .await
            .record_conversation_failed(gateway_call_id, error);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::SpeechCancelToken;
    use crate::operator::state::{
        shared_state, CallStatus, ConversationRole, ConversationStatus, TelnyxIds,
    };
    use crate::tts::{OutboundTtsFactory, TtsAudio, TtsRegistry, PIPER_SAMPLE_RATE_HZ};
    use async_trait::async_trait;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    fn test_runtime() -> ConversationRuntime {
        ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            true,
        )
    }

    fn test_runtime_with_tts() -> ConversationRuntime {
        let tts = Arc::new(TtsRegistry::new(
            Arc::new(StaticTtsFactory),
            Arc::new(StaticTtsFactory),
        ));
        ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            tts,
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

    fn transcription_update_with_confidence(
        text: &str,
        confidence: Option<f32>,
        final_segment: bool,
    ) -> motlie_model::TranscriptionUpdate {
        motlie_model::TranscriptionUpdate {
            segments: vec![motlie_model::TranscriptSegment {
                start_ms: 0,
                end_ms: 100,
                text: text.to_string(),
                confidence,
                final_segment,
            }],
        }
    }

    async fn wait_for_conversation_final() {
        sleep(
            conversation_final_debounce(VoiceQualityConfig::default().endpoint.merge_window_ms)
                + Duration::from_millis(50),
        )
        .await;
    }

    #[tokio::test]
    async fn identity_processor_dispatches_without_final_coalescing() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            false,
        );
        assert_eq!(runtime.processor_label(), "processor");

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
        .expect("identity processor should dispatch immediately");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("hello"));
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("hello")
        );
    }

    #[tokio::test]
    async fn identity_processor_can_coalesce_adjacent_final_fragments() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            true,
        );
        let media_registry = SharedMediaRegistry::default();

        for text in ["this should merge", "into one processor turn."] {
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
            .expect("identity coalescing processor should accept final fragment");
        }
        wait_for_conversation_final().await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("this should merge into one processor turn.")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("this should merge into one processor turn.")
        );
    }

    #[tokio::test]
    async fn turn_batched_identity_batches_two_committed_turns_without_reset() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        {
            let mut guard = state.write().await;
            let call = guard.calls.get_mut(&gateway_call_id).expect("call exists");
            call.conversation.processor = ConversationProcessorKind::turn_batched_identity(
                motlie_agent::voice::turn_batching::IdentityTurnBatcherConfig::fixed_batch_size(2)
                    .with_max_batch_wait_ms(1_000),
            );
        }
        let runtime = ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            false,
        );
        let media_registry = SharedMediaRegistry::default();

        handle_transcript_event_with_turn(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "first".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            Some("turn-1"),
        )
        .await
        .expect("first turn should accumulate");
        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(call.conversation.last_user_text.as_deref(), Some("first"));
            assert!(call.conversation.last_assistant_text.is_none());
        }

        handle_transcript_event_with_turn(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "second".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            Some("turn-2"),
        )
        .await
        .expect("second turn should complete the batch");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("second"));
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some(
                "first
second"
            )
        );
        assert_eq!(
            call.conversation
                .lines
                .iter()
                .filter(|line| line.role == ConversationRole::Assistant)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn turn_batched_identity_completes_pending_batch_after_wait_ceiling() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        {
            let mut guard = state.write().await;
            let call = guard.calls.get_mut(&gateway_call_id).expect("call exists");
            call.conversation.processor = ConversationProcessorKind::turn_batched_identity(
                motlie_agent::voice::turn_batching::IdentityTurnBatcherConfig::fixed_batch_size(3)
                    .with_max_batch_wait_ms(25),
            );
        }
        let runtime = ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            false,
        );
        let media_registry = SharedMediaRegistry::default();

        handle_transcript_event_with_turn(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "only turn".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            Some("turn-1"),
        )
        .await
        .expect("turn should accumulate before timeout");
        assert!(state
            .read()
            .await
            .calls
            .get(&gateway_call_id)
            .expect("call exists")
            .conversation
            .last_assistant_text
            .is_none());

        timeout(Duration::from_secs(2), async {
            loop {
                if state
                    .read()
                    .await
                    .calls
                    .get(&gateway_call_id)
                    .and_then(|call| call.conversation.last_assistant_text.as_deref())
                    == Some("only turn")
                {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("wait ceiling should complete pending batch");
    }

    #[tokio::test]
    async fn disabled_runtime_records_user_turn_without_invoking_processor() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        let runtime = ConversationRuntime::new(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
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
        .expect("disabled processor should not fail media");

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
    async fn smoke_test_holds_and_merges_final_fragments_while_playback_is_active() {
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

        for text in [
            "You're still missing some",
            "end points always seems to be the last word that's missing",
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
            .expect("disabled barge-in should not fail overlapping final turn");
        }
        wait_for_conversation_final().await;

        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );
        assert!(
            timeout(Duration::from_millis(200), rx.recv())
                .await
                .is_err(),
            "smoke-test coalescer should hold replies until active playback finishes"
        );
        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(call.conversation.status, ConversationStatus::Speaking);
            assert_eq!(
                call.conversation.last_assistant_text.as_deref(),
                Some("assistant is speaking")
            );
        }

        media_registry
            .finish_speech(&gateway_call_id, "tts_test")
            .await;
        let command = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("merged deferred speech should enqueue after prior playback finishes")
            .expect("merged deferred speech should emit a media command");
        match command {
            crate::media::OutboundMediaCommand::Frame(frame) => {
                assert_ne!(frame.playback_id, "tts_test");
            }
            other => panic!("expected deferred frame, got {other:?}"),
        }
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Speaking);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("You're still missing some end points always seems to be the last word that's missing")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("You're still missing some end points always seems to be the last word that's missing")
        );
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
            Some("second")
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
    async fn conversation_debounce_merges_adjacent_final_fragments() {
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
        wait_for_conversation_final().await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("Yeah, this is not the last fragment or frame didn't come through still.")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("Yeah, this is not the last fragment or frame didn't come through still.")
        );
    }

    #[test]
    fn conversation_incomplete_tail_detector_is_conservative() {
        assert_eq!(
            EndpointQualityConfig::default()
                .conversation_incomplete_tail_reason("Endpointing is still a problem, isn'"),
            Some("dangling_tail")
        );
        assert_eq!(
            EndpointQualityConfig::default()
                .conversation_incomplete_tail_reason("the endpoints are"),
            Some("tail_word")
        );
        assert_eq!(
            EndpointQualityConfig::default().conversation_incomplete_tail_reason("Can you hear me"),
            None
        );
        assert_eq!(
            EndpointQualityConfig::default()
                .conversation_incomplete_tail_reason("Can you hear me?"),
            None
        );
    }

    #[tokio::test]
    async fn conversation_holds_incomplete_tail_until_continuation_final() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = test_runtime();
        let media_registry = SharedMediaRegistry::default();

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "Endpointing is still a problem, isn'".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("incomplete final fragment should be accepted");
        wait_for_conversation_final().await;

        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(call.conversation.status, ConversationStatus::Idle);
            assert!(call.conversation.last_user_text.is_none());
            assert!(call.conversation.last_assistant_text.is_none());
        }

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "it?".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("continuation final should be accepted");
        wait_for_conversation_final().await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("Endpointing is still a problem, isn' it?")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("Endpointing is still a problem, isn' it?")
        );
    }

    #[tokio::test]
    async fn conversation_low_confidence_nonterminal_final_waits_for_continuation() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = test_runtime();
        let media_registry = SharedMediaRegistry::default();

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "The endpoint sounded glitchy".to_string(),
                update: transcription_update_with_confidence(
                    "The endpoint sounded glitchy",
                    Some(0.2),
                    true,
                ),
            },
            None,
        )
        .await
        .expect("low-confidence final should be accepted");
        wait_for_conversation_final().await;

        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(call.conversation.status, ConversationStatus::Idle);
            assert!(call.conversation.last_user_text.is_none());
            assert!(call.conversation.last_assistant_text.is_none());
        }

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "today.".to_string(),
                update: transcription_update_with_confidence("today.", Some(0.9), true),
            },
            None,
        )
        .await
        .expect("continuation final should be accepted");
        wait_for_conversation_final().await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("The endpoint sounded glitchy today.")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("The endpoint sounded glitchy today.")
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
        .expect("final should still regenerate through processor");
        wait_for_conversation_final().await;

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
            Some("hello there")
        );
    }

    #[tokio::test]
    async fn queue_failure_after_call_end_does_not_mark_conversation_failed() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state
            .write()
            .await
            .calls
            .get_mut(&gateway_call_id)
            .unwrap()
            .status = CallStatus::Ended;
        let runtime = test_runtime_with_tts();

        queue_conversation_speech(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            "normal hangup race".to_string(),
            SpeechConflictPolicy::Reject,
            ConversationTurnContext {
                finalized_at: None,
                latest_finalized_at: None,
                turn_id: Some("turn_test".to_string()),
                coalesced_turn_ids: vec!["turn_test".to_string()],
            },
        )
        .await;

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::Ended);
        assert_ne!(call.conversation.status, ConversationStatus::Failed);
        assert!(call.conversation.last_error.is_none());
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
