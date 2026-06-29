use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context};
use motlie_agent::voice::turn_batching::{
    Accumulating, Prompt, TurnBatchCompletionReason, TurnBatchReset,
};
use motlie_voice::app::{ConversationCommand, TranscriptEvent};
use motlie_voice::telephony::CallAction;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};

use crate::call_control::TelnyxClient;
use crate::conversation_policy::{
    AssistantOutputPolicyAction, BargeInPolicyDecision, BargeInTranscriptEvidence, BargeInTrigger,
    ConversationPolicyQueue, FinalTranscriptDispatchDecision, FinalTranscriptDispatchEvidence,
    GenerationPolicyAction, PendingPolicyOutput,
};
use crate::early_response::{EarlyResponseCancelReason, EarlyResponseEvent};
use crate::media::{SharedMediaRegistry, SpeechClearReason};
use crate::operator::state::{
    CallStatus, ConversationMode, LogLevel, QualityConversationProcessorVisibleTurn,
    QualityPlaybackMetadata, QualitySpanEmission, SharedState, TurnBatchActiveState,
};
pub use crate::processors::{
    ConversationCommittedTurn, ConversationProcessorInput, ConversationProcessorKind,
    ConversationProcessorOutput,
};
use crate::processors::{ConversationProcessorHost, TurnBatchOutputRejectionReason};
use crate::quality::{
    BargeInQualityConfig, ConversationPolicyConfig, ConversationPolicyMode, EndpointQualityConfig,
    RedactionMode, VoiceQualityConfig,
};
use crate::speech;
use crate::speech::{SpeechConflictPolicy, SpeechQueueRequest};
use crate::tts::SharedTtsRegistry;

#[derive(Clone)]
pub struct ConversationRuntime {
    telnyx: TelnyxClient,
    tts: SharedTtsRegistry,
    processor_enabled: Arc<AtomicBool>,
    final_coalescing_enabled: Arc<AtomicBool>,
    barge_in_enabled: Arc<AtomicBool>,
    pending_conversation_finals: Arc<Mutex<HashMap<String, PendingConversationFinal>>>,
    deferred_say_generations: Arc<Mutex<HashMap<String, u64>>>,
    deferred_policy_outputs:
        Arc<Mutex<HashMap<String, ConversationPolicyQueue<PendingConversationSay>>>>,
    barge_in_coalesce_windows: Arc<Mutex<HashMap<String, BargeInCoalesceWindow>>>,
    post_barge_in_replacement_guard_intents: Arc<Mutex<HashMap<String, Instant>>>,
    post_barge_in_dispatch_guards: Arc<Mutex<HashMap<String, PostBargeInDispatchGuard>>>,
    processor_generations: Arc<Mutex<HashMap<String, u64>>>,
    processor_hosts: Arc<Mutex<HashMap<String, ConversationProcessorHost>>>,
    turn_batch_timers: Arc<Mutex<HashMap<String, ActiveTurnBatchTimers>>>,
}

#[derive(Debug)]
struct ActiveTurnBatchTimers {
    batch_id: String,
    epoch: u64,
    first_turn_at: Instant,
    last_turn_at: Instant,
    source_turn_ids: Vec<String>,
    pending_turn_count: usize,
    target_turn_count: usize,
    batch_wait_ms: u64,
    idle_wait_ms: u64,
    batch_wait_handle: Option<JoinHandle<()>>,
    idle_handle: Option<JoinHandle<()>>,
}

impl ActiveTurnBatchTimers {
    fn abort(&mut self) {
        if let Some(handle) = self.batch_wait_handle.take() {
            handle.abort();
        }
        if let Some(handle) = self.idle_handle.take() {
            handle.abort();
        }
    }
}

#[derive(Clone, Debug)]
struct TurnBatchTimingSnapshot {
    batch_id: String,
    epoch: u64,
    pending_turn_count: usize,
    target_turn_count: usize,
    source_turn_ids: Vec<String>,
    first_turn_age_ms: u64,
    idle_age_ms: u64,
    accumulation_ms: u64,
    batch_wait_remaining_ms: Option<u64>,
    idle_wait_remaining_ms: Option<u64>,
    effective_deadline_remaining_ms: Option<u64>,
    effective_deadline_source: Option<&'static str>,
}

impl TurnBatchTimingSnapshot {
    fn from_timers(timers: &ActiveTurnBatchTimers, now: Instant) -> Self {
        let first_turn_age_ms = millis_since(now, timers.first_turn_at);
        let idle_age_ms = millis_since(now, timers.last_turn_at);
        let batch_wait_remaining_ms = remaining_ms(timers.batch_wait_ms, first_turn_age_ms);
        let idle_wait_remaining_ms = remaining_ms(timers.idle_wait_ms, idle_age_ms);
        let (effective_deadline_remaining_ms, effective_deadline_source) =
            effective_deadline(batch_wait_remaining_ms, idle_wait_remaining_ms);
        Self {
            batch_id: timers.batch_id.clone(),
            epoch: timers.epoch,
            pending_turn_count: timers.pending_turn_count,
            target_turn_count: timers.target_turn_count,
            source_turn_ids: timers.source_turn_ids.clone(),
            first_turn_age_ms,
            idle_age_ms,
            accumulation_ms: first_turn_age_ms,
            batch_wait_remaining_ms,
            idle_wait_remaining_ms,
            effective_deadline_remaining_ms,
            effective_deadline_source,
        }
    }
}

#[derive(Clone, Debug)]
struct TurnBatchOutputRejection {
    batch_id: String,
    epoch: u64,
    reason: TurnBatchOutputRejectionReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BargeInCoalesceWindow {
    silence_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PostBargeInDispatchGuard {
    armed_at: Instant,
    playback_id: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessorGenerationCancel {
    generation: u64,
    dropped_policy_outputs: usize,
}

enum TurnBatchCompletionAttempt {
    Completed(ConversationProcessorOutput),
    Rejected(TurnBatchOutputRejection),
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
            deferred_policy_outputs: Arc::new(Mutex::new(HashMap::new())),
            barge_in_coalesce_windows: Arc::new(Mutex::new(HashMap::new())),
            post_barge_in_replacement_guard_intents: Arc::new(Mutex::new(HashMap::new())),
            post_barge_in_dispatch_guards: Arc::new(Mutex::new(HashMap::new())),
            processor_generations: Arc::new(Mutex::new(HashMap::new())),
            processor_hosts: Arc::new(Mutex::new(HashMap::new())),
            turn_batch_timers: Arc::new(Mutex::new(HashMap::new())),
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

    async fn current_processor_generation(&self, gateway_call_id: &str) -> u64 {
        self.processor_generations
            .lock()
            .await
            .get(gateway_call_id)
            .copied()
            .unwrap_or_default()
    }

    async fn is_processor_generation_current(
        &self,
        gateway_call_id: &str,
        generation: u64,
    ) -> bool {
        self.current_processor_generation(gateway_call_id).await == generation
    }

    async fn cancel_active_processor_generation(
        &self,
        gateway_call_id: &str,
    ) -> ProcessorGenerationCancel {
        let generation = {
            let mut generations = self.processor_generations.lock().await;
            let generation = generations.entry(gateway_call_id.to_string()).or_insert(0);
            *generation = generation.saturating_add(1);
            *generation
        };
        {
            let mut generations = self.deferred_say_generations.lock().await;
            let generation = generations.entry(gateway_call_id.to_string()).or_insert(0);
            *generation = generation.saturating_add(1);
        }
        let dropped_policy_outputs = self
            .deferred_policy_outputs
            .lock()
            .await
            .get_mut(gateway_call_id)
            .map(ConversationPolicyQueue::clear)
            .unwrap_or_default();
        ProcessorGenerationCancel {
            generation,
            dropped_policy_outputs,
        }
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
        reason: TurnBatchCompletionReason,
    ) -> TurnBatchCompletionAttempt {
        let mut hosts = self.processor_hosts.lock().await;
        match hosts.get_mut(gateway_call_id) {
            Some(host) => match host.complete_pending(batch_id, epoch, reason) {
                Ok(output) => TurnBatchCompletionAttempt::Completed(output),
                Err(reason) => TurnBatchCompletionAttempt::Rejected(TurnBatchOutputRejection {
                    batch_id: batch_id.to_string(),
                    epoch,
                    reason,
                }),
            },
            None => TurnBatchCompletionAttempt::Rejected(TurnBatchOutputRejection {
                batch_id: batch_id.to_string(),
                epoch,
                reason: TurnBatchOutputRejectionReason::InactiveBatch,
            }),
        }
    }

    async fn upsert_turn_batch_timers(
        &self,
        gateway_call_id: &str,
        batch: &Accumulating,
    ) -> (TurnBatchTimingSnapshot, bool, bool) {
        let now = Instant::now();
        let mut timers = self.turn_batch_timers.lock().await;
        let mut spawn_batch_wait = false;
        let mut spawn_idle = false;
        let entry = timers
            .entry(gateway_call_id.to_string())
            .or_insert_with(|| {
                spawn_batch_wait = batch.batch_wait_remaining_ms > 0;
                spawn_idle = batch.idle_wait_ms > 0;
                ActiveTurnBatchTimers {
                    batch_id: batch.batch_id.clone(),
                    epoch: batch.epoch,
                    first_turn_at: now,
                    last_turn_at: now,
                    source_turn_ids: batch.source_turn_ids.clone(),
                    pending_turn_count: batch.source_turn_ids.len(),
                    target_turn_count: batch.target_turn_count,
                    batch_wait_ms: batch.batch_wait_remaining_ms,
                    idle_wait_ms: batch.idle_wait_ms,
                    batch_wait_handle: None,
                    idle_handle: None,
                }
            });
        if entry.batch_id != batch.batch_id || entry.epoch != batch.epoch {
            entry.abort();
            *entry = ActiveTurnBatchTimers {
                batch_id: batch.batch_id.clone(),
                epoch: batch.epoch,
                first_turn_at: now,
                last_turn_at: now,
                source_turn_ids: batch.source_turn_ids.clone(),
                pending_turn_count: batch.source_turn_ids.len(),
                target_turn_count: batch.target_turn_count,
                batch_wait_ms: batch.batch_wait_remaining_ms,
                idle_wait_ms: batch.idle_wait_ms,
                batch_wait_handle: None,
                idle_handle: None,
            };
            spawn_batch_wait = batch.batch_wait_remaining_ms > 0;
            spawn_idle = batch.idle_wait_ms > 0;
        } else {
            entry.last_turn_at = now;
            entry.source_turn_ids = batch.source_turn_ids.clone();
            entry.pending_turn_count = batch.source_turn_ids.len();
            entry.target_turn_count = batch.target_turn_count;
            entry.batch_wait_ms = batch.batch_wait_remaining_ms;
            entry.idle_wait_ms = batch.idle_wait_ms;
            if let Some(handle) = entry.idle_handle.take() {
                handle.abort();
            }
            spawn_idle = batch.idle_wait_ms > 0;
        }
        let snapshot = TurnBatchTimingSnapshot::from_timers(entry, now);
        (snapshot, spawn_batch_wait, spawn_idle)
    }

    async fn store_batch_wait_handle(
        &self,
        gateway_call_id: &str,
        batch_id: &str,
        epoch: u64,
        handle: JoinHandle<()>,
    ) {
        let mut timers = self.turn_batch_timers.lock().await;
        if let Some(entry) = timers.get_mut(gateway_call_id) {
            if entry.batch_id == batch_id && entry.epoch == epoch {
                entry.batch_wait_handle = Some(handle);
                return;
            }
        }
        handle.abort();
    }

    async fn store_idle_handle(
        &self,
        gateway_call_id: &str,
        batch_id: &str,
        epoch: u64,
        handle: JoinHandle<()>,
    ) {
        let mut timers = self.turn_batch_timers.lock().await;
        if let Some(entry) = timers.get_mut(gateway_call_id) {
            if entry.batch_id == batch_id && entry.epoch == epoch {
                entry.idle_handle = Some(handle);
                return;
            }
        }
        handle.abort();
    }

    async fn finish_turn_batch_timers(
        &self,
        gateway_call_id: &str,
        batch_id: Option<&str>,
        epoch: u64,
        completed_by: Option<TurnBatchCompletionReason>,
    ) -> Option<TurnBatchTimingSnapshot> {
        let now = Instant::now();
        let mut timers = self.turn_batch_timers.lock().await;
        let should_remove = timers.get(gateway_call_id).is_some_and(|entry| {
            if let Some(batch_id) = batch_id {
                entry.batch_id == batch_id
            } else {
                entry.epoch == epoch
            }
        });
        if !should_remove {
            return None;
        }
        let mut entry = timers.remove(gateway_call_id)?;
        match completed_by {
            Some(TurnBatchCompletionReason::MaxBatchWaitTimeout) => {
                entry.batch_wait_handle.take();
            }
            Some(TurnBatchCompletionReason::MaxIdleTimeout) => {
                entry.idle_handle.take();
            }
            _ => {}
        }
        entry.abort();
        Some(TurnBatchTimingSnapshot::from_timers(&entry, now))
    }
}

fn millis_since(now: Instant, then: Instant) -> u64 {
    now.saturating_duration_since(then).as_millis() as u64
}

fn remaining_ms(limit_ms: u64, elapsed_ms: u64) -> Option<u64> {
    if limit_ms == 0 {
        None
    } else {
        Some(limit_ms.saturating_sub(elapsed_ms))
    }
}

fn effective_deadline(
    batch_wait_remaining_ms: Option<u64>,
    idle_wait_remaining_ms: Option<u64>,
) -> (Option<u64>, Option<&'static str>) {
    match (batch_wait_remaining_ms, idle_wait_remaining_ms) {
        (Some(batch), Some(idle)) if batch <= idle => (Some(batch), Some("max_batch_wait_timeout")),
        (Some(_batch), Some(idle)) => (Some(idle), Some("max_idle_timeout")),
        (Some(batch), None) => (Some(batch), Some("max_batch_wait_timeout")),
        (None, Some(idle)) => (Some(idle), Some("max_idle_timeout")),
        (None, None) => (None, None),
    }
}

#[derive(Clone, Debug)]
struct ConversationFinalInput {
    event: TranscriptEvent,
    quality_config: VoiceQualityConfig,
    final_transcript_at: Instant,
    debounce: Duration,
    turn_id: Option<String>,
    source_asr_session_ids: Vec<String>,
    source_utterance_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConversationFinalHoldKind {
    Playback,
    IncompleteTail,
}

#[derive(Clone, Debug)]
struct PendingConversationFinal {
    event: TranscriptEvent,
    quality_config: VoiceQualityConfig,
    final_transcript_at: Instant,
    latest_final_transcript_at: Instant,
    generation: u64,
    turn_ids: Vec<String>,
    source_asr_session_ids: Vec<String>,
    source_utterance_ids: Vec<String>,
    active_hold_kind: Option<ConversationFinalHoldKind>,
    active_hold_started_at: Option<Instant>,
    playback_hold_count: u64,
    playback_hold_ms: u64,
    incomplete_tail_hold_count: u64,
    incomplete_tail_hold_ms: u64,
    playback_hold_limit_reached: bool,
}

#[derive(Clone, Debug, Default)]
struct ConversationTurnContext {
    finalized_at: Option<Instant>,
    latest_finalized_at: Option<Instant>,
    processor_visible_turn_at: Option<Instant>,
    processor_generation: Option<u64>,
    turn_id: Option<String>,
    coalesced_turn_ids: Vec<String>,
    source_asr_session_ids: Vec<String>,
    source_utterance_ids: Vec<String>,
    metadata: QualityPlaybackMetadata,
    post_barge_in_dispatch_guard: bool,
}

#[derive(Clone, Debug)]
struct PendingConversationSay {
    response_text: String,
    turn_context: ConversationTurnContext,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConversationTranscriptMetadata<'a> {
    pub turn_id: Option<&'a str>,
    pub source_asr_session_ids: Option<&'a [String]>,
    pub source_utterance_ids: Option<&'a [String]>,
    pub confidence: Option<f32>,
    pub stability: Option<f32>,
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
    handle_transcript_event_with_metadata(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        event,
        quality_config,
        ConversationTranscriptMetadata {
            turn_id,
            ..ConversationTranscriptMetadata::default()
        },
    )
    .await
}

pub async fn handle_transcript_event_with_metadata(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
    quality_config: Option<&VoiceQualityConfig>,
    metadata: ConversationTranscriptMetadata<'_>,
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

    let active_playback_id = media_registry
        .active_speech_playback_id(gateway_call_id)
        .await;
    let active_playback = active_playback_id.is_some();
    let transcript_confidence = metadata
        .confidence
        .or_else(|| latest_transcript_confidence(&event));

    if !event.is_final() {
        let evidence = BargeInTranscriptEvidence {
            text: &transcript_text,
            active_playback,
            confidence: transcript_confidence,
            stability: metadata.stability,
        };
        let decision = snapshot
            .quality_config
            .conversation_policy
            .decide_transcript_barge_in(&snapshot.barge_in, BargeInTrigger::Partial, evidence);
        if decision.cancels_playback() {
            cancel_active_speech_for_barge_in(
                state,
                media_registry,
                runtime,
                gateway_call_id,
                decision,
                snapshot.config_id.clone(),
                snapshot.redaction_mode,
            )
            .await?;
        } else if !decision.forwards_caller_transcript() {
            log_barge_in_transcript_ignored(gateway_call_id, decision, evidence);
        }
        return Ok(());
    }

    let final_transcript_at = Instant::now();

    let post_barge_in_guard_active = post_barge_in_dispatch_guard_active(
        runtime,
        media_registry,
        gateway_call_id,
        &snapshot.quality_config.conversation_policy,
        active_playback_id.as_deref(),
        snapshot.tts_playback_id.as_deref(),
    )
    .await;
    let dispatch_evidence = FinalTranscriptDispatchEvidence {
        text: &transcript_text,
        post_barge_in_guard_active,
        active_or_recent_playback: active_playback || post_barge_in_guard_active,
        confidence: transcript_confidence,
        stability: metadata.stability,
        assistant_echo_signature: snapshot.assistant_echo_signature.as_deref(),
    };
    let dispatch_decision = snapshot
        .quality_config
        .conversation_policy
        .decide_final_transcript_dispatch(
            &snapshot.barge_in,
            &snapshot.quality_config.echo_suppression,
            dispatch_evidence,
        );
    if !dispatch_decision.forwards() {
        log_final_transcript_dispatch_suppressed(
            gateway_call_id,
            dispatch_decision,
            dispatch_evidence,
        );
        return Ok(());
    }
    if dispatch_evidence.post_barge_in_guard_active {
        clear_post_barge_in_dispatch_guard(runtime, gateway_call_id).await;
    }

    let evidence = BargeInTranscriptEvidence {
        text: &transcript_text,
        active_playback,
        confidence: transcript_confidence,
        stability: metadata.stability,
    };
    let final_barge_in_decision = snapshot
        .quality_config
        .conversation_policy
        .decide_transcript_barge_in(&snapshot.barge_in, BargeInTrigger::Final, evidence);
    if final_barge_in_decision.cancels_playback() {
        cancel_active_speech_for_barge_in(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            final_barge_in_decision,
            snapshot.config_id.clone(),
            snapshot.redaction_mode,
        )
        .await?;
    } else if !final_barge_in_decision.forwards_caller_transcript() {
        log_barge_in_transcript_ignored(gateway_call_id, final_barge_in_decision, evidence);
        return Ok(());
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
    if let Some(debounce) = post_barge_in_policy_debounce(
        runtime,
        gateway_call_id,
        &quality_config.conversation_policy,
    )
    .await
    {
        schedule_conversation_final(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            ConversationFinalInput {
                event,
                quality_config,
                final_transcript_at,
                debounce,
                turn_id: metadata.turn_id.map(str::to_string),
                source_asr_session_ids: metadata
                    .source_asr_session_ids
                    .map(<[String]>::to_vec)
                    .unwrap_or_default(),
                source_utterance_ids: metadata
                    .source_utterance_ids
                    .map(<[String]>::to_vec)
                    .unwrap_or_default(),
            },
        )
        .await;
        return Ok(());
    }
    if snapshot.endpoint_merge_window_ms == 0 || !runtime.final_coalescing_enabled() {
        let post_barge_in_dispatch_guard = take_post_barge_in_replacement_guard_intent(
            runtime,
            gateway_call_id,
            &quality_config.conversation_policy,
        )
        .await;
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
                processor_visible_turn_at: None,
                processor_generation: None,
                turn_id: metadata.turn_id.map(str::to_string),
                coalesced_turn_ids: metadata
                    .turn_id
                    .map(|id| vec![id.to_string()])
                    .unwrap_or_default(),
                source_asr_session_ids: metadata
                    .source_asr_session_ids
                    .map(<[String]>::to_vec)
                    .unwrap_or_default(),
                source_utterance_ids: metadata
                    .source_utterance_ids
                    .map(<[String]>::to_vec)
                    .unwrap_or_default(),
                metadata: QualityPlaybackMetadata::default(),
                post_barge_in_dispatch_guard,
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
            turn_id: metadata.turn_id.map(str::to_string),
            source_asr_session_ids: metadata
                .source_asr_session_ids
                .map(<[String]>::to_vec)
                .unwrap_or_default(),
            source_utterance_ids: metadata
                .source_utterance_ids
                .map(<[String]>::to_vec)
                .unwrap_or_default(),
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

async fn record_barge_in_policy_window(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    decision: BargeInPolicyDecision,
) {
    let Some(silence_ms) = decision.caller_turn.silence_ms() else {
        return;
    };
    runtime.barge_in_coalesce_windows.lock().await.insert(
        gateway_call_id.to_string(),
        BargeInCoalesceWindow { silence_ms },
    );
}

async fn post_barge_in_policy_debounce(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    policy: &ConversationPolicyConfig,
) -> Option<Duration> {
    if !matches!(
        policy.mode,
        ConversationPolicyMode::BargeInCoalesceAfterSilence
    ) {
        return None;
    }
    runtime
        .barge_in_coalesce_windows
        .lock()
        .await
        .get(gateway_call_id)
        .map(|window| Duration::from_millis(window.silence_ms))
}

async fn clear_barge_in_policy_window(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
) -> bool {
    runtime
        .barge_in_coalesce_windows
        .lock()
        .await
        .remove(gateway_call_id)
        .is_some()
}

const STALE_REPLACEMENT_GUARD_INTENT_MS: u64 = 30_000;

fn conversation_policy_uses_post_barge_in_dispatch_guard(
    policy: &ConversationPolicyConfig,
) -> bool {
    matches!(
        policy.mode,
        ConversationPolicyMode::BargeInCancelOnly
            | ConversationPolicyMode::BargeInCoalesceAfterSilence
    ) && policy.post_barge_in_echo_guard_ms > 0
}

fn conversation_policy_mode_can_arm_post_barge_in_dispatch_guard(
    mode: ConversationPolicyMode,
) -> bool {
    matches!(
        mode,
        ConversationPolicyMode::BargeInCancelOnly
            | ConversationPolicyMode::BargeInCoalesceAfterSilence
    )
}

async fn record_post_barge_in_replacement_guard_intent(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    mode: ConversationPolicyMode,
) {
    if !conversation_policy_mode_can_arm_post_barge_in_dispatch_guard(mode) {
        return;
    }
    runtime
        .post_barge_in_replacement_guard_intents
        .lock()
        .await
        .insert(gateway_call_id.to_string(), Instant::now());
}

async fn take_post_barge_in_replacement_guard_intent(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    policy: &ConversationPolicyConfig,
) -> bool {
    if !conversation_policy_uses_post_barge_in_dispatch_guard(policy) {
        return false;
    }
    let Some(armed_at) = runtime
        .post_barge_in_replacement_guard_intents
        .lock()
        .await
        .remove(gateway_call_id)
    else {
        return false;
    };
    armed_at.elapsed() <= Duration::from_millis(STALE_REPLACEMENT_GUARD_INTENT_MS)
}

async fn arm_post_barge_in_dispatch_guard(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    playback_id: &str,
) {
    runtime.post_barge_in_dispatch_guards.lock().await.insert(
        gateway_call_id.to_string(),
        PostBargeInDispatchGuard {
            armed_at: Instant::now(),
            playback_id: playback_id.to_string(),
        },
    );
}

async fn post_barge_in_dispatch_guard_active(
    runtime: &ConversationRuntime,
    media_registry: &SharedMediaRegistry,
    gateway_call_id: &str,
    policy: &ConversationPolicyConfig,
    active_playback_id: Option<&str>,
    tts_playback_id: Option<&str>,
) -> bool {
    // Backstop for leaked guards; normal lifetime is bounded by the configured echo window
    // and the guarded playback ID remaining active or recently terminal in the media registry.
    const STALE_GUARD_MAX_MS: u64 = 300_000;
    if !conversation_policy_uses_post_barge_in_dispatch_guard(policy) {
        return false;
    }

    let Some(guard) = ({
        let mut guards = runtime.post_barge_in_dispatch_guards.lock().await;
        let Some(guard) = guards.get(gateway_call_id).cloned() else {
            return false;
        };
        if guard.armed_at.elapsed() > Duration::from_millis(STALE_GUARD_MAX_MS) {
            guards.remove(gateway_call_id);
            return false;
        }
        Some(guard)
    }) else {
        return false;
    };

    let guarded_playback = guard.playback_id.as_str();
    let different_active_playback =
        active_playback_id.is_some_and(|playback_id| playback_id != guarded_playback);
    let different_tts_playback =
        tts_playback_id.is_some_and(|playback_id| playback_id != guarded_playback);
    if different_active_playback || different_tts_playback {
        clear_post_barge_in_dispatch_guard(runtime, gateway_call_id).await;
        return false;
    }

    if media_registry
        .speech_playback_active_or_recent(
            gateway_call_id,
            guarded_playback,
            Duration::from_millis(policy.post_barge_in_echo_guard_ms),
        )
        .await
    {
        return true;
    }

    clear_post_barge_in_dispatch_guard(runtime, gateway_call_id).await;
    false
}

async fn clear_post_barge_in_dispatch_guard(runtime: &ConversationRuntime, gateway_call_id: &str) {
    runtime
        .post_barge_in_dispatch_guards
        .lock()
        .await
        .remove(gateway_call_id);
}

fn extend_unique(target: &mut Vec<String>, values: Vec<String>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
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
                source_asr_session_ids: input.source_asr_session_ids.clone(),
                source_utterance_ids: input.source_utterance_ids.clone(),
                active_hold_kind: None,
                active_hold_started_at: None,
                playback_hold_count: 0,
                playback_hold_ms: 0,
                incomplete_tail_hold_count: 0,
                incomplete_tail_hold_ms: 0,
                playback_hold_limit_reached: false,
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
        extend_unique(
            &mut entry.source_asr_session_ids,
            input.source_asr_session_ids,
        );
        extend_unique(&mut entry.source_utterance_ids, input.source_utterance_ids);
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
                let Some(limit_reached) = conversation_playback_hold_limit_reached(
                    &runtime,
                    &gateway_call_id,
                    generation,
                )
                .await
                else {
                    return;
                };
                if !limit_reached {
                    let Some((next_generation, next_delay)) = defer_ready_conversation_final(
                        &runtime,
                        &gateway_call_id,
                        generation,
                        ConversationFinalHoldKind::Playback,
                    )
                    .await
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
                tracing::debug!(
                    gateway_call_id,
                    generation,
                    "conversation.final_debounce.playback_hold_limit_reached"
                );
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
                let Some((next_generation, next_delay)) = defer_ready_conversation_final(
                    &runtime,
                    &gateway_call_id,
                    generation,
                    ConversationFinalHoldKind::IncompleteTail,
                )
                .await
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
            let had_barge_in_policy_window =
                clear_barge_in_policy_window(&runtime, &gateway_call_id).await;
            let post_barge_in_dispatch_guard = take_post_barge_in_replacement_guard_intent(
                &runtime,
                &gateway_call_id,
                &pending.quality_config.conversation_policy,
            )
            .await
                || (had_barge_in_policy_window
                    && conversation_policy_uses_post_barge_in_dispatch_guard(
                        &pending.quality_config.conversation_policy,
                    ));
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
                    processor_visible_turn_at: None,
                    processor_generation: None,
                    turn_id: pending.turn_ids.first().cloned(),
                    coalesced_turn_ids: pending.turn_ids,
                    source_asr_session_ids: pending.source_asr_session_ids,
                    source_utterance_ids: pending.source_utterance_ids,
                    metadata: QualityPlaybackMetadata::default(),
                    post_barge_in_dispatch_guard,
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

async fn conversation_playback_hold_limit_reached(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> Option<bool> {
    let mut pending = runtime.pending_conversation_finals.lock().await;
    let entry = match pending.get_mut(gateway_call_id) {
        Some(entry) if entry.generation == generation => entry,
        _ => return None,
    };
    let max_hold_ms = entry
        .quality_config
        .endpoint
        .conversation_playback_max_hold_ms;
    if max_hold_ms == 0 {
        return Some(false);
    }
    let max_hold = Duration::from_millis(max_hold_ms);
    if conversation_final_hold_elapsed(entry, ConversationFinalHoldKind::Playback) < max_hold {
        return Some(false);
    }
    finish_conversation_final_hold(entry);
    entry.playback_hold_limit_reached = true;
    Some(true)
}

async fn defer_ready_conversation_final(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
    hold_kind: ConversationFinalHoldKind,
) -> Option<(u64, Duration)> {
    let mut pending = runtime.pending_conversation_finals.lock().await;
    match pending.get_mut(gateway_call_id) {
        Some(entry) if entry.generation == generation => {
            mark_conversation_final_hold(entry, hold_kind);
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

fn mark_conversation_final_hold(
    pending: &mut PendingConversationFinal,
    hold_kind: ConversationFinalHoldKind,
) {
    if pending.active_hold_kind == Some(hold_kind) {
        return;
    }
    finish_conversation_final_hold(pending);
    pending.active_hold_kind = Some(hold_kind);
    pending.active_hold_started_at = Some(Instant::now());
    match hold_kind {
        ConversationFinalHoldKind::Playback => {
            pending.playback_hold_count = pending.playback_hold_count.saturating_add(1);
        }
        ConversationFinalHoldKind::IncompleteTail => {
            pending.incomplete_tail_hold_count =
                pending.incomplete_tail_hold_count.saturating_add(1);
        }
    }
}

fn finish_conversation_final_hold(pending: &mut PendingConversationFinal) {
    let Some(hold_kind) = pending.active_hold_kind.take() else {
        pending.active_hold_started_at = None;
        return;
    };
    let Some(started_at) = pending.active_hold_started_at.take() else {
        return;
    };
    let elapsed_ms = started_at.elapsed().as_millis() as u64;
    match hold_kind {
        ConversationFinalHoldKind::Playback => {
            pending.playback_hold_ms = pending.playback_hold_ms.saturating_add(elapsed_ms);
        }
        ConversationFinalHoldKind::IncompleteTail => {
            pending.incomplete_tail_hold_ms =
                pending.incomplete_tail_hold_ms.saturating_add(elapsed_ms);
        }
    }
}

fn conversation_final_hold_elapsed(
    pending: &PendingConversationFinal,
    hold_kind: ConversationFinalHoldKind,
) -> Duration {
    let base_ms = match hold_kind {
        ConversationFinalHoldKind::Playback => pending.playback_hold_ms,
        ConversationFinalHoldKind::IncompleteTail => pending.incomplete_tail_hold_ms,
    };
    let mut elapsed = Duration::from_millis(base_ms);
    if pending.active_hold_kind == Some(hold_kind) {
        if let Some(started_at) = pending.active_hold_started_at {
            elapsed = elapsed.saturating_add(started_at.elapsed());
        }
    }
    elapsed
}

async fn take_ready_conversation_final(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    generation: u64,
) -> Option<PendingConversationFinal> {
    let mut pending = runtime.pending_conversation_finals.lock().await;
    match pending.get(gateway_call_id) {
        Some(entry) if entry.generation == generation => {
            let mut entry = pending.remove(gateway_call_id)?;
            finish_conversation_final_hold(&mut entry);
            Some(entry)
        }
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
            let timing = runtime
                .finish_turn_batch_timers(
                    gateway_call_id,
                    Some(&prompt.batch_id),
                    prompt.epoch,
                    Some(prompt.completion_reason),
                )
                .await;
            emit_turn_batch_prompt_complete_lifecycle(
                state,
                gateway_call_id,
                &prompt,
                timing.as_ref(),
            )
            .await;
            let mut prompt_turn_context = turn_context.clone();
            prompt_turn_context.turn_id = Some(prompt.response_turn_id.clone());
            prompt_turn_context.coalesced_turn_ids = prompt.source_turn_ids.clone();
            prompt_turn_context.metadata = QualityPlaybackMetadata::turn_batch(
                prompt.batch_id.clone(),
                prompt.epoch,
                prompt.response_turn_id.clone(),
                prompt.completion_reason.as_str(),
            );
            apply_conversation_command_with_timing(
                state,
                media_registry,
                runtime,
                gateway_call_id,
                target.clone(),
                ConversationCommand::Say { text: prompt.text },
                prompt_turn_context,
            )
            .await?;
            Ok(ProcessorOutputResult::CommandApplied)
        }
        ConversationProcessorOutput::Accumulating(batch) => {
            let timing = schedule_turn_batch_timeouts(
                state,
                media_registry,
                runtime,
                gateway_call_id,
                &batch,
            )
            .await;
            emit_turn_batch_accumulated_lifecycle(state, gateway_call_id, &batch, &timing).await;
            Ok(ProcessorOutputResult::NoCommand)
        }
        ConversationProcessorOutput::Reset(reset) => {
            let timing = runtime
                .finish_turn_batch_timers(
                    gateway_call_id,
                    reset.batch_id.as_deref(),
                    reset.epoch,
                    None,
                )
                .await;
            emit_turn_batch_reset_lifecycle(state, gateway_call_id, &reset, timing.as_ref()).await;
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

async fn schedule_turn_batch_timeouts(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    batch: &Accumulating,
) -> TurnBatchTimingSnapshot {
    let (timing, spawn_batch_wait, spawn_idle) = runtime
        .upsert_turn_batch_timers(gateway_call_id, batch)
        .await;
    if spawn_batch_wait {
        let handle = spawn_turn_batch_completion_timer(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            TurnBatchTimerRequest {
                batch_id: batch.batch_id.clone(),
                epoch: batch.epoch,
                deadline_ms: batch.batch_wait_remaining_ms,
                reason: TurnBatchCompletionReason::MaxBatchWaitTimeout,
            },
        );
        runtime
            .store_batch_wait_handle(gateway_call_id, &batch.batch_id, batch.epoch, handle)
            .await;
    }
    if spawn_idle {
        let handle = spawn_turn_batch_completion_timer(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            TurnBatchTimerRequest {
                batch_id: batch.batch_id.clone(),
                epoch: batch.epoch,
                deadline_ms: batch.idle_wait_ms,
                reason: TurnBatchCompletionReason::MaxIdleTimeout,
            },
        );
        runtime
            .store_idle_handle(gateway_call_id, &batch.batch_id, batch.epoch, handle)
            .await;
    }
    timing
}

struct TurnBatchTimerRequest {
    batch_id: String,
    epoch: u64,
    deadline_ms: u64,
    reason: TurnBatchCompletionReason,
}

fn spawn_turn_batch_completion_timer(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    request: TurnBatchTimerRequest,
) -> JoinHandle<()> {
    let TurnBatchTimerRequest {
        batch_id,
        epoch,
        deadline_ms,
        reason,
    } = request;
    let state = state.clone();
    let media_registry = media_registry.clone();
    let runtime = runtime.clone();
    let gateway_call_id = gateway_call_id.to_string();
    tokio::spawn(async move {
        sleep(Duration::from_millis(deadline_ms)).await;
        if !runtime.processor_enabled() {
            return;
        }
        let attempt = runtime
            .complete_pending_processor_batch(&gateway_call_id, &batch_id, epoch, reason)
            .await;
        let output = match attempt {
            TurnBatchCompletionAttempt::Completed(output) => output,
            TurnBatchCompletionAttempt::Rejected(rejection) => {
                emit_turn_batch_output_rejected_lifecycle(&state, &gateway_call_id, &rejection)
                    .await;
                return;
            }
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
            conversation_policy: snapshot.quality_config.conversation_policy.clone(),
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
    })
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
        "playback_hold_ms": pending.playback_hold_ms,
        "playback_hold_count": pending.playback_hold_count,
        "playback_hold_limit_reached": pending.playback_hold_limit_reached,
        "incomplete_tail_hold_ms": pending.incomplete_tail_hold_ms,
        "incomplete_tail_hold_count": pending.incomplete_tail_hold_count,
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
    mut turn_context: ConversationTurnContext,
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
    let processor_visible_at = Instant::now();
    turn_context.processor_visible_turn_at = Some(processor_visible_at);
    {
        let mut guard = state.write().await;
        guard.record_conversation_user_turn(gateway_call_id, transcript_text.clone());
        guard.emit_quality_conversation_processor_visible_turn(
            gateway_call_id,
            QualityConversationProcessorVisibleTurn {
                config_id: &snapshot.config_id,
                redaction_mode: snapshot.redaction_mode,
                include_transcript_text: snapshot.quality_config.logging.include_transcript_text,
                turn_id: turn_context.turn_id.as_deref(),
                text: &transcript_text,
                coalesced_turn_ids: &turn_context.coalesced_turn_ids,
                source_asr_session_ids: &turn_context.source_asr_session_ids,
                source_utterance_ids: &turn_context.source_utterance_ids,
                processor: snapshot.processor.label(),
                response_mode: snapshot.processor.response_mode_label(),
                confidence: latest_transcript_confidence(&event),
            },
        );
    }

    let processor_generation = runtime.current_processor_generation(gateway_call_id).await;
    turn_context.processor_generation = Some(processor_generation);
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
        conversation_policy: snapshot.quality_config.conversation_policy.clone(),
    };
    let mut saw_command = false;
    if let Some(output) = runtime
        .process_processor_input(gateway_call_id, snapshot.processor, processor_input)
        .await
    {
        if !runtime
            .is_processor_generation_current(gateway_call_id, processor_generation)
            .await
        {
            tracing::info!(
                gateway_call_id,
                processor_generation,
                "conversation.processor.output_dropped_stale_generation"
            );
            return Ok(());
        }
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
    if !snapshot.attached {
        return Ok(());
    }
    let decision = snapshot
        .quality_config
        .conversation_policy
        .decide_barge_in(&snapshot.barge_in, BargeInTrigger::SpeechOnset);
    if !decision.cancels_playback() {
        return Ok(());
    }

    cancel_active_speech_for_barge_in(
        state,
        media_registry,
        _runtime,
        gateway_call_id,
        decision,
        snapshot.config_id.clone(),
        snapshot.redaction_mode,
    )
    .await
}

fn log_barge_in_transcript_ignored(
    gateway_call_id: &str,
    decision: BargeInPolicyDecision,
    evidence: BargeInTranscriptEvidence<'_>,
) {
    tracing::info!(
        gateway_call_id,
        trigger = decision.trigger.as_str(),
        policy_mode = decision.mode.label(),
        caller_transcript_action = decision.caller_transcript.label(),
        transcript_chars = evidence.char_count(),
        transcript_words = evidence.word_count(),
        transcript_confidence = evidence.confidence,
        transcript_stability = evidence.stability,
        "conversation.barge_in.transcript_ignored"
    );
}

fn log_final_transcript_dispatch_suppressed(
    gateway_call_id: &str,
    decision: FinalTranscriptDispatchDecision,
    evidence: FinalTranscriptDispatchEvidence<'_>,
) {
    tracing::info!(
        gateway_call_id,
        dispatch_action = decision.action.label(),
        dispatch_reason = decision.reason.label(),
        transcript_chars = evidence
            .text
            .chars()
            .filter(|ch| ch.is_alphanumeric())
            .count(),
        transcript_words = evidence
            .text
            .split_whitespace()
            .filter(|word| word.chars().any(|ch| ch.is_alphanumeric()))
            .count(),
        transcript_confidence = evidence.confidence,
        transcript_stability = evidence.stability,
        active_or_recent_playback = evidence.active_or_recent_playback,
        post_barge_in_guard_active = evidence.post_barge_in_guard_active,
        assistant_echo_signature_present = evidence.assistant_echo_signature.is_some(),
        "conversation.final_transcript.dispatch_suppressed"
    );
}

async fn cancel_active_speech_for_barge_in(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    decision: BargeInPolicyDecision,
    config_id: String,
    redaction_mode: RedactionMode,
) -> anyhow::Result<()> {
    let trigger = decision.trigger;
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
            "policy_mode": decision.mode.label(),
            "playback_action": decision.playback.label(),
            "generation_action": decision.generation.label(),
            "caller_turn_action": decision.caller_turn.label(),
            "caller_transcript_action": decision.caller_transcript.label(),
            "post_barge_in_silence_ms": decision.caller_turn.silence_ms(),
            "reset_turn_batch": decision.reset_turn_batch,
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
    let generation_cancel = if decision.generation == GenerationPolicyAction::CancelActive {
        Some(
            runtime
                .cancel_active_processor_generation(gateway_call_id)
                .await,
        )
    } else {
        None
    };
    if decision.reset_turn_batch {
        reset_turn_batch_for_barge_in(state, runtime, gateway_call_id).await;
    }
    record_barge_in_policy_window(runtime, gateway_call_id, decision).await;
    record_post_barge_in_replacement_guard_intent(runtime, gateway_call_id, decision.mode).await;
    state
        .write()
        .await
        .record_conversation_interrupted(gateway_call_id, &playback_id);
    tracing::info!(
        gateway_call_id,
        playback_id,
        trigger = trigger.as_str(),
        policy_mode = decision.mode.label(),
        playback_action = decision.playback.label(),
        generation_action = decision.generation.label(),
        caller_turn_action = decision.caller_turn.label(),
        caller_transcript_action = decision.caller_transcript.label(),
        post_barge_in_silence_ms = decision.caller_turn.silence_ms(),
        processor_generation = generation_cancel.map(|cancel| cancel.generation),
        dropped_policy_outputs = generation_cancel.map(|cancel| cancel.dropped_policy_outputs),
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
        let timing = runtime
            .finish_turn_batch_timers(
                gateway_call_id,
                reset.batch_id.as_deref(),
                reset.epoch,
                None,
            )
            .await;
        emit_turn_batch_reset_lifecycle(state, gateway_call_id, &reset, timing.as_ref()).await;
    }
}

async fn emit_turn_batch_accumulated_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    batch: &Accumulating,
    timing: &TurnBatchTimingSnapshot,
) {
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "accumulated",
        batch_id = %batch.batch_id,
        epoch = batch.epoch,
        pending_turn_count = timing.pending_turn_count,
        target_turn_count = timing.target_turn_count,
        first_turn_age_ms = timing.first_turn_age_ms,
        idle_age_ms = timing.idle_age_ms,
        batch_wait_remaining_ms = ?timing.batch_wait_remaining_ms,
        idle_wait_remaining_ms = ?timing.idle_wait_remaining_ms,
        effective_deadline_remaining_ms = ?timing.effective_deadline_remaining_ms,
        effective_deadline_source = ?timing.effective_deadline_source,
        "conversation.processor.turn_batch.accumulated"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "batch_id": &batch.batch_id,
        "epoch": batch.epoch,
        "pending_turn_count": timing.pending_turn_count,
        "target_turn_count": timing.target_turn_count,
        "k_of_n": { "k": timing.pending_turn_count, "n": timing.target_turn_count },
        "turn_id": batch.source_turn_ids.last(),
        "source_turn_ids": &batch.source_turn_ids,
        "first_turn_age_ms": timing.first_turn_age_ms,
        "idle_age_ms": timing.idle_age_ms,
        "batch_wait_remaining_ms": timing.batch_wait_remaining_ms,
        "idle_wait_ms": batch.idle_wait_ms,
        "idle_wait_remaining_ms": timing.idle_wait_remaining_ms,
        "effective_deadline_remaining_ms": timing.effective_deadline_remaining_ms,
        "effective_deadline_source": timing.effective_deadline_source,
    }));
    let active = turn_batch_active_state(timing);
    let mut guard = state.write().await;
    guard.record_turn_batch_active(gateway_call_id, active);
    guard.emit_quality_turn_batch_lifecycle(gateway_call_id, "accumulated", payload);
}

async fn emit_turn_batch_prompt_complete_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    prompt: &Prompt,
    timing: Option<&TurnBatchTimingSnapshot>,
) {
    let source_turn_count = prompt.source_turn_ids.len();
    let accumulation_ms = timing
        .map(|timing| timing.accumulation_ms)
        .unwrap_or_default();
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "prompt_complete",
        batch_id = %prompt.batch_id,
        epoch = prompt.epoch,
        source_turn_count,
        completion_reason = prompt.completion_reason.as_str(),
        accumulation_ms,
        response_turn_id = %prompt.response_turn_id,
        prompt_chars = prompt.text.chars().count(),
        "conversation.processor.turn_batch.prompt_complete"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "batch_id": &prompt.batch_id,
        "epoch": prompt.epoch,
        "source_turn_count": source_turn_count,
        "joined_source_turn_count": source_turn_count,
        "source_turn_ids": &prompt.source_turn_ids,
        "response_turn_id": &prompt.response_turn_id,
        "completion_reason": prompt.completion_reason.as_str(),
        "accumulation_ms": accumulation_ms,
        "first_turn_age_ms": timing.map(|timing| timing.first_turn_age_ms),
        "idle_age_ms": timing.map(|timing| timing.idle_age_ms),
        "effective_deadline_remaining_ms": timing.and_then(|timing| timing.effective_deadline_remaining_ms),
        "effective_deadline_source": timing.and_then(|timing| timing.effective_deadline_source),
        "prompt_words": prompt.text.split_whitespace().count(),
        "prompt_chars": prompt.text.chars().count(),
    }));
    let mut guard = state.write().await;
    guard.clear_turn_batch_active(gateway_call_id);
    guard.record_quality_turn_batch_prompt_complete(
        gateway_call_id,
        source_turn_count,
        accumulation_ms,
        prompt.completion_reason.as_str(),
    );
    guard.emit_quality_turn_batch_lifecycle(gateway_call_id, "prompt_complete", payload);
}

async fn emit_turn_batch_reset_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    reset: &TurnBatchReset,
    timing: Option<&TurnBatchTimingSnapshot>,
) {
    let accumulation_ms = timing
        .map(|timing| timing.accumulation_ms)
        .unwrap_or_default();
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "reset",
        reason = reset.reason.as_str(),
        epoch = reset.epoch,
        batch_id = ?reset.batch_id.as_deref(),
        discarded_turn_count = reset.discarded_turn_count,
        accumulation_ms,
        "conversation.processor.turn_batch.reset"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "reason": reset.reason.as_str(),
        "batch_id": &reset.batch_id,
        "epoch": reset.epoch,
        "discarded_turn_count": reset.discarded_turn_count,
        "accumulation_ms": accumulation_ms,
        "first_turn_age_ms": timing.map(|timing| timing.first_turn_age_ms),
        "idle_age_ms": timing.map(|timing| timing.idle_age_ms),
        "pending_turn_count": timing.map(|timing| timing.pending_turn_count),
        "source_turn_ids": timing.map(|timing| timing.source_turn_ids.as_slice()),
        "effective_deadline_remaining_ms": timing.and_then(|timing| timing.effective_deadline_remaining_ms),
        "effective_deadline_source": timing.and_then(|timing| timing.effective_deadline_source),
    }));
    let mut guard = state.write().await;
    guard.clear_turn_batch_active(gateway_call_id);
    guard.record_quality_turn_batch_reset(
        gateway_call_id,
        reset.reason.as_str(),
        reset.discarded_turn_count,
    );
    guard.emit_quality_turn_batch_lifecycle(gateway_call_id, "reset", payload);
}

async fn emit_turn_batch_output_rejected_lifecycle(
    state: &SharedState,
    gateway_call_id: &str,
    rejection: &TurnBatchOutputRejection,
) {
    tracing::info!(
        gateway_call_id,
        lifecycle_event = "output_rejected",
        batch_id = %rejection.batch_id,
        epoch = rejection.epoch,
        reason = rejection.reason.as_str(),
        "conversation.processor.turn_batch.output_rejected"
    );
    let payload = turn_batch_payload(serde_json::json!({
        "batch_id": &rejection.batch_id,
        "epoch": rejection.epoch,
        "reason": rejection.reason.as_str(),
    }));
    let mut guard = state.write().await;
    guard.record_quality_turn_batch_output_rejected(gateway_call_id, rejection.reason.as_str());
    guard.emit_quality_turn_batch_lifecycle(gateway_call_id, "output_rejected", payload);
}

fn turn_batch_active_state(timing: &TurnBatchTimingSnapshot) -> TurnBatchActiveState {
    let now = chrono::Utc::now();
    let effective_deadline_at = timing
        .effective_deadline_remaining_ms
        .map(|remaining| now + chrono::Duration::milliseconds(remaining as i64));
    TurnBatchActiveState {
        batch_id: timing.batch_id.clone(),
        epoch: timing.epoch,
        pending_turn_count: timing.pending_turn_count,
        target_turn_count: timing.target_turn_count,
        source_turn_ids: timing.source_turn_ids.clone(),
        first_turn_at: now - chrono::Duration::milliseconds(timing.first_turn_age_ms as i64),
        last_turn_at: now - chrono::Duration::milliseconds(timing.idle_age_ms as i64),
        effective_deadline_at,
        effective_deadline_source: timing.effective_deadline_source.map(str::to_string),
    }
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
    assistant_echo_signature: Option<String>,
    tts_playback_id: Option<String>,
    quality_config: VoiceQualityConfig,
}

#[derive(Clone, Debug)]
struct ConversationCommandTarget {
    mode: ConversationMode,
    call_control_id: String,
    barge_in: BargeInQualityConfig,
    conversation_policy: ConversationPolicyConfig,
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
    let assistant_echo_signature = call
        .tts
        .as_ref()
        .map(|tts| tts.echo_signature.clone())
        .filter(|signature| !signature.is_empty());
    let tts_playback_id = call.tts.as_ref().map(|tts| tts.playback_id.clone());
    Some(ConversationSnapshot {
        attached: call.conversation.attached,
        mode: call.conversation.mode,
        processor: call.conversation.processor.clone(),
        call_control_id: call.ids.call_control_id.clone(),
        barge_in,
        config_id,
        redaction_mode,
        endpoint_merge_window_ms,
        assistant_echo_signature,
        tts_playback_id,
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
                    let active_playback = media_registry
                        .active_speech_playback_id(gateway_call_id)
                        .await
                        .is_some();
                    let conversation_policy = target.conversation_policy.clone();
                    let say_decision = conversation_policy
                        .decide_say_overlap(target.barge_in.enabled, active_playback);
                    match say_decision.assistant_output {
                        AssistantOutputPolicyAction::RetainBoundedPending => {
                            state.write().await.record_conversation_proposal(
                                gateway_call_id,
                                response_text.clone(),
                            );
                            tracing::info!(
                                gateway_call_id,
                                policy_mode = say_decision.mode.label(),
                                assistant_output_action = say_decision.assistant_output.label(),
                                "conversation.say.deferred_barge_in_disabled"
                            );
                            enqueue_policy_pending_conversation_say(
                                state,
                                media_registry,
                                runtime,
                                gateway_call_id,
                                conversation_policy,
                                response_text,
                                turn_context.clone(),
                            )
                            .await;
                            return Ok(());
                        }
                        AssistantOutputPolicyAction::LegacyDeferLatest => {
                            state.write().await.record_conversation_proposal(
                                gateway_call_id,
                                response_text.clone(),
                            );
                            tracing::info!(
                                gateway_call_id,
                                policy_mode = say_decision.mode.label(),
                                assistant_output_action = say_decision.assistant_output.label(),
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
                        AssistantOutputPolicyAction::QueueNow => {}
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

async fn enqueue_policy_pending_conversation_say(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    policy: ConversationPolicyConfig,
    response_text: String,
    turn_context: ConversationTurnContext,
) {
    let pending = PendingConversationSay {
        response_text,
        turn_context,
    };
    let (outcome, spawn_drain) = {
        let mut outputs = runtime.deferred_policy_outputs.lock().await;
        let queue = outputs.entry(gateway_call_id.to_string()).or_default();
        let outcome = queue.enqueue(&policy, pending);
        let spawn_drain = !queue.drain_running();
        if spawn_drain {
            queue.set_drain_running(true);
        }
        (outcome, spawn_drain)
    };

    tracing::info!(
        gateway_call_id,
        policy_mode = policy.mode.label(),
        generation = outcome.sequence,
        pending_len = outcome.pending_len,
        dropped_count = outcome.dropped_count,
        pending_output_order = outcome.order.label(),
        "conversation.say.policy_pending_enqueued"
    );

    if spawn_drain {
        let state = state.clone();
        let media_registry = media_registry.clone();
        let runtime = runtime.clone();
        let gateway_call_id = gateway_call_id.to_string();
        tokio::spawn(async move {
            drain_policy_pending_conversation_says(
                &state,
                &media_registry,
                &runtime,
                &gateway_call_id,
                policy,
            )
            .await;
        });
    }
}

async fn drain_policy_pending_conversation_says(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    policy: ConversationPolicyConfig,
) {
    let hold_config = deferred_policy_hold_config(state, &policy).await;
    'pending: loop {
        let Some(head) = front_deferred_policy_output(runtime, gateway_call_id).await else {
            if finish_deferred_policy_output_drain(runtime, gateway_call_id).await {
                continue;
            }
            return;
        };

        while media_registry
            .active_speech_playback_id(gateway_call_id)
            .await
            .is_some()
        {
            if conversation_call_terminal(state, gateway_call_id).await {
                clear_deferred_policy_outputs(runtime, gateway_call_id).await;
                emit_deferred_say_playback_hold_span(
                    state,
                    gateway_call_id,
                    DeferredSayPlaybackHoldSpan {
                        config: &hold_config,
                        started: head.enqueued_at,
                        generation: head.sequence,
                        result: "call_ended",
                        turn_context: &head.payload.turn_context,
                    },
                )
                .await;
                if finish_deferred_policy_output_drain(runtime, gateway_call_id).await {
                    continue 'pending;
                }
                return;
            }

            if !is_deferred_policy_head(runtime, gateway_call_id, head.sequence).await {
                emit_deferred_say_playback_hold_span(
                    state,
                    gateway_call_id,
                    DeferredSayPlaybackHoldSpan {
                        config: &hold_config,
                        started: head.enqueued_at,
                        generation: head.sequence,
                        result: "superseded",
                        turn_context: &head.payload.turn_context,
                    },
                )
                .await;
                continue 'pending;
            }

            if hold_config.max_hold_ms > 0
                && head.enqueued_at.elapsed() >= Duration::from_millis(hold_config.max_hold_ms)
            {
                let dropped =
                    drop_deferred_policy_head(runtime, gateway_call_id, head.sequence).await;
                emit_deferred_say_playback_hold_span(
                    state,
                    gateway_call_id,
                    DeferredSayPlaybackHoldSpan {
                        config: &hold_config,
                        started: head.enqueued_at,
                        generation: head.sequence,
                        result: if dropped {
                            "max_hold_dropped"
                        } else {
                            "superseded"
                        },
                        turn_context: &head.payload.turn_context,
                    },
                )
                .await;
                tracing::info!(
                    gateway_call_id,
                    generation = head.sequence,
                    max_hold_ms = hold_config.max_hold_ms,
                    dropped,
                    "conversation.say.deferred_playback_max_hold_dropped"
                );
                continue 'pending;
            }

            if head.enqueued_at.elapsed() >= Duration::from_millis(hold_config.timeout_ms) {
                let dropped_count = clear_deferred_policy_outputs(runtime, gateway_call_id).await;
                emit_deferred_say_playback_hold_span(
                    state,
                    gateway_call_id,
                    DeferredSayPlaybackHoldSpan {
                        config: &hold_config,
                        started: head.enqueued_at,
                        generation: head.sequence,
                        result: "timeout",
                        turn_context: &head.payload.turn_context,
                    },
                )
                .await;
                let timeout_ms = hold_config.timeout_ms;
                let error = format!(
                    "deferred conversation policy timed out after {timeout_ms}ms waiting for active playback; dropped {dropped_count} pending outputs"
                );
                if record_conversation_failed_unless_terminal(
                    state,
                    gateway_call_id,
                    error.clone(),
                    "conversation.say.policy_pending_timeout_after_call_end",
                )
                .await
                {
                    if finish_deferred_policy_output_drain(runtime, gateway_call_id).await {
                        continue 'pending;
                    }
                    return;
                }
                tracing::warn!(
                    gateway_call_id,
                    error,
                    "conversation.say.policy_pending_timeout"
                );
                if finish_deferred_policy_output_drain(runtime, gateway_call_id).await {
                    continue 'pending;
                }
                return;
            }

            sleep(Duration::from_millis(hold_config.poll_ms)).await;
        }

        let Some(pending) = take_next_deferred_policy_output(runtime, gateway_call_id).await else {
            continue;
        };
        if pending.sequence != head.sequence {
            emit_deferred_say_playback_hold_span(
                state,
                gateway_call_id,
                DeferredSayPlaybackHoldSpan {
                    config: &hold_config,
                    started: head.enqueued_at,
                    generation: head.sequence,
                    result: "superseded",
                    turn_context: &head.payload.turn_context,
                },
            )
            .await;
            continue;
        }

        emit_deferred_say_playback_hold_span(
            state,
            gateway_call_id,
            DeferredSayPlaybackHoldSpan {
                config: &hold_config,
                started: pending.enqueued_at,
                generation: pending.sequence,
                result: "playback_cleared",
                turn_context: &pending.payload.turn_context,
            },
        )
        .await;

        queue_conversation_speech(
            state,
            media_registry,
            runtime,
            gateway_call_id,
            pending.payload.response_text,
            SpeechConflictPolicy::Reject,
            pending.payload.turn_context,
        )
        .await;
    }
}

async fn deferred_policy_hold_config(
    state: &SharedState,
    policy: &ConversationPolicyConfig,
) -> DeferredSayPlaybackHoldConfig {
    let guard = state.read().await;
    let config = &guard.quality.config;
    DeferredSayPlaybackHoldConfig {
        config_id: config.config_id(),
        redaction_mode: config.logging.redaction_mode,
        timeout_ms: config.text_call.playback_wait_timeout_ms,
        max_hold_ms: policy.active_playback_hold_ms,
        poll_ms: config.endpoint.conversation_playback_hold_poll_ms,
        policy_mode: policy.mode.label(),
    }
}

async fn front_deferred_policy_output(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
) -> Option<PendingPolicyOutput<PendingConversationSay>> {
    runtime
        .deferred_policy_outputs
        .lock()
        .await
        .get(gateway_call_id)
        .and_then(|queue| queue.front().cloned())
}

async fn is_deferred_policy_head(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    sequence: u64,
) -> bool {
    runtime
        .deferred_policy_outputs
        .lock()
        .await
        .get(gateway_call_id)
        .and_then(|queue| queue.front())
        .is_some_and(|pending| pending.sequence == sequence)
}

async fn take_next_deferred_policy_output(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
) -> Option<PendingPolicyOutput<PendingConversationSay>> {
    runtime
        .deferred_policy_outputs
        .lock()
        .await
        .get_mut(gateway_call_id)
        .and_then(ConversationPolicyQueue::take_next)
}

async fn clear_deferred_policy_outputs(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
) -> usize {
    runtime
        .deferred_policy_outputs
        .lock()
        .await
        .get_mut(gateway_call_id)
        .map(ConversationPolicyQueue::clear)
        .unwrap_or_default()
}

async fn drop_deferred_policy_head(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    sequence: u64,
) -> bool {
    let mut outputs = runtime.deferred_policy_outputs.lock().await;
    let Some(queue) = outputs.get_mut(gateway_call_id) else {
        return false;
    };
    if queue
        .front()
        .is_some_and(|pending| pending.sequence == sequence)
    {
        queue.take_next();
        true
    } else {
        false
    }
}

async fn finish_deferred_policy_output_drain(
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
) -> bool {
    let mut outputs = runtime.deferred_policy_outputs.lock().await;
    match outputs.get_mut(gateway_call_id) {
        Some(queue) if queue.is_empty() => {
            queue.set_drain_running(false);
            outputs.remove(gateway_call_id);
            false
        }
        Some(queue) => {
            queue.set_drain_running(true);
            true
        }
        None => false,
    }
}

async fn conversation_call_terminal(state: &SharedState, gateway_call_id: &str) -> bool {
    let guard = state.read().await;
    matches!(
        guard.calls.get(gateway_call_id).map(|call| call.status),
        None | Some(CallStatus::Ended | CallStatus::Failed)
    )
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
    let hold_config = {
        let guard = state.read().await;
        let config = &guard.quality.config;
        DeferredSayPlaybackHoldConfig {
            config_id: config.config_id(),
            redaction_mode: config.logging.redaction_mode,
            timeout_ms: config.text_call.playback_wait_timeout_ms,
            max_hold_ms: config.endpoint.conversation_playback_max_hold_ms,
            poll_ms: config.endpoint.conversation_playback_hold_poll_ms,
            policy_mode: config.conversation_policy.mode.label(),
        }
    };
    let wait_limit_ms = if hold_config.max_hold_ms == 0 {
        hold_config.timeout_ms
    } else {
        hold_config.timeout_ms.min(hold_config.max_hold_ms)
    };
    let started = Instant::now();
    while media_registry
        .active_speech_playback_id(gateway_call_id)
        .await
        .is_some()
    {
        if !is_latest_deferred_say_generation(runtime, gateway_call_id, generation).await {
            emit_deferred_say_playback_hold_span(
                state,
                gateway_call_id,
                DeferredSayPlaybackHoldSpan {
                    config: &hold_config,
                    started,
                    generation,
                    result: "superseded",
                    turn_context: &turn_context,
                },
            )
            .await;
            tracing::debug!(
                gateway_call_id,
                generation,
                "conversation.say.deferred_superseded"
            );
            return;
        }
        if started.elapsed() >= Duration::from_millis(wait_limit_ms) {
            if hold_config.max_hold_ms > 0 && hold_config.max_hold_ms <= hold_config.timeout_ms {
                let _ =
                    take_latest_deferred_say_generation(runtime, gateway_call_id, generation).await;
                emit_deferred_say_playback_hold_span(
                    state,
                    gateway_call_id,
                    DeferredSayPlaybackHoldSpan {
                        config: &hold_config,
                        started,
                        generation,
                        result: "max_hold_reached",
                        turn_context: &turn_context,
                    },
                )
                .await;
                tracing::info!(
                    gateway_call_id,
                    generation,
                    max_hold_ms = hold_config.max_hold_ms,
                    "conversation.say.deferred_playback_max_hold_reached"
                );
                return;
            }
            emit_deferred_say_playback_hold_span(
                state,
                gateway_call_id,
                DeferredSayPlaybackHoldSpan {
                    config: &hold_config,
                    started,
                    generation,
                    result: "timeout",
                    turn_context: &turn_context,
                },
            )
            .await;
            let timeout_ms = hold_config.timeout_ms;
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
        sleep(Duration::from_millis(hold_config.poll_ms)).await;
    }
    if !take_latest_deferred_say_generation(runtime, gateway_call_id, generation).await {
        emit_deferred_say_playback_hold_span(
            state,
            gateway_call_id,
            DeferredSayPlaybackHoldSpan {
                config: &hold_config,
                started,
                generation,
                result: "superseded",
                turn_context: &turn_context,
            },
        )
        .await;
        tracing::debug!(
            gateway_call_id,
            generation,
            "conversation.say.deferred_superseded"
        );
        return;
    }
    emit_deferred_say_playback_hold_span(
        state,
        gateway_call_id,
        DeferredSayPlaybackHoldSpan {
            config: &hold_config,
            started,
            generation,
            result: "playback_cleared",
            turn_context: &turn_context,
        },
    )
    .await;

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

struct DeferredSayPlaybackHoldConfig {
    config_id: String,
    redaction_mode: RedactionMode,
    timeout_ms: u64,
    max_hold_ms: u64,
    poll_ms: u64,
    policy_mode: &'static str,
}

struct DeferredSayPlaybackHoldSpan<'a> {
    config: &'a DeferredSayPlaybackHoldConfig,
    started: Instant,
    generation: u64,
    result: &'static str,
    turn_context: &'a ConversationTurnContext,
}

async fn emit_deferred_say_playback_hold_span(
    state: &SharedState,
    gateway_call_id: &str,
    span: DeferredSayPlaybackHoldSpan<'_>,
) {
    let payload = serde_json::json!({
        "generation": span.generation,
        "result": span.result,
        "timeout_ms": span.config.timeout_ms,
        "max_hold_ms": span.config.max_hold_ms,
        "poll_ms": span.config.poll_ms,
        "policy_mode": span.config.policy_mode,
        "turn_id": span.turn_context.turn_id.as_deref(),
        "coalesced_turn_count": span.turn_context.coalesced_turn_ids.len(),
        "coalesced_turn_ids": span.turn_context.coalesced_turn_ids.as_slice(),
        "final_to_deferred_ms": span.turn_context.finalized_at.map(|instant| instant.elapsed().as_millis() as u64),
        "latest_final_to_deferred_ms": span.turn_context.latest_finalized_at.map(|instant| instant.elapsed().as_millis() as u64),
    });
    let payload = match payload {
        serde_json::Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    state.write().await.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id: span.config.config_id.clone(),
            redaction_mode: span.config.redaction_mode,
            span_name: "conversation.say.deferred_playback_hold",
            category: "intentional_delay",
            duration: span.started.elapsed(),
            critical_path: true,
            concurrent: false,
            payload,
        },
    );
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
    if let Some(processor_generation) = turn_context.processor_generation {
        if !runtime
            .is_processor_generation_current(gateway_call_id, processor_generation)
            .await
        {
            tracing::info!(
                gateway_call_id,
                processor_generation,
                "conversation.say.dropped_stale_generation"
            );
            return;
        }
    }

    let post_barge_in_dispatch_guard = turn_context.post_barge_in_dispatch_guard;
    let (speech_output, barge_in_cancel_terminal_at) = {
        let mut guard = state.write().await;
        let speech_output = guard
            .calls
            .get(gateway_call_id)
            .map(|call| call.speech_output)
            .unwrap_or_else(|| {
                crate::operator::state::SpeechOutputConfig::from_quality(
                    guard.conversation_tts_backend,
                    &guard.quality.config.tts,
                )
            });
        let barge_in_cancel_terminal_at =
            guard.take_last_barge_in_cancel_terminal_at(gateway_call_id);
        (speech_output, barge_in_cancel_terminal_at)
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
            processor_visible_turn_at: turn_context.processor_visible_turn_at,
            barge_in_cancel_terminal_at,
            turn_id: turn_context.turn_id,
            coalesced_turn_ids: turn_context.coalesced_turn_ids,
            source_asr_session_ids: turn_context.source_asr_session_ids,
            source_utterance_ids: turn_context.source_utterance_ids,
            prebuffer_chunks_override: None,
            speech_output: Some(speech_output),
            metadata: turn_context.metadata,
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
            if post_barge_in_dispatch_guard {
                arm_post_barge_in_dispatch_guard(runtime, gateway_call_id, &queued.playback_id)
                    .await;
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

    async fn wait_for_active_playback_change(
        media_registry: &SharedMediaRegistry,
        gateway_call_id: &str,
        previous_playback_id: &str,
    ) -> String {
        timeout(Duration::from_secs(2), async {
            loop {
                if let Some(playback_id) = media_registry
                    .active_speech_playback_id(gateway_call_id)
                    .await
                    .filter(|playback_id| playback_id != previous_playback_id)
                {
                    return playback_id;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("new active playback should start")
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
    async fn turn_batched_identity_max_batch_wait_is_anchored_to_first_turn() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        {
            let mut guard = state.write().await;
            let call = guard.calls.get_mut(&gateway_call_id).expect("call exists");
            call.conversation.processor = ConversationProcessorKind::turn_batched_identity(
                motlie_agent::voice::turn_batching::IdentityTurnBatcherConfig::fixed_batch_size(3)
                    .with_max_batch_wait_ms(80)
                    .with_max_idle_wait_ms(500),
            );
        }
        let runtime = ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            false,
        );
        let media_registry = SharedMediaRegistry::default();
        let started = Instant::now();

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
        sleep(Duration::from_millis(40)).await;
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
        .expect("second turn should not reset max batch wait");

        timeout(Duration::from_secs(2), async {
            loop {
                if state
                    .read()
                    .await
                    .calls
                    .get(&gateway_call_id)
                    .and_then(|call| call.conversation.last_assistant_text.as_deref())
                    == Some("first\nsecond")
                {
                    break;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("first-turn max batch wait should complete pending batch");

        assert!(
            started.elapsed() < Duration::from_millis(250),
            "max_batch_wait_ms must not restart on the second turn"
        );
    }

    #[tokio::test]
    async fn turn_batched_identity_idle_wait_resets_on_new_turn() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        {
            let mut guard = state.write().await;
            let call = guard.calls.get_mut(&gateway_call_id).expect("call exists");
            call.conversation.processor = ConversationProcessorKind::turn_batched_identity(
                motlie_agent::voice::turn_batching::IdentityTurnBatcherConfig::fixed_batch_size(3)
                    .with_max_batch_wait_ms(500)
                    .with_max_idle_wait_ms(50),
            );
        }
        let runtime = ConversationRuntime::new_with_processor_options(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            true,
            false,
        );
        let media_registry = SharedMediaRegistry::default();
        let started = Instant::now();

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
        sleep(Duration::from_millis(30)).await;
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
        .expect("second turn should reset idle wait");

        timeout(Duration::from_secs(2), async {
            loop {
                if state
                    .read()
                    .await
                    .calls
                    .get(&gateway_call_id)
                    .and_then(|call| call.conversation.last_assistant_text.as_deref())
                    == Some("first\nsecond")
                {
                    break;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("idle wait should complete pending batch before batch wait ceiling");

        assert!(
            started.elapsed() < Duration::from_millis(250),
            "max_idle_wait_ms should fire before the first-turn batch ceiling"
        );
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
                conversation_policy: ConversationPolicyConfig::default(),
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
                text: "hello there".to_string(),
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
    async fn current_compat_short_final_interrupts_active_playback_and_dispatches() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(&gateway_call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        let runtime = test_runtime_with_tts();

        handle_transcript_event_with_metadata(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "Stop".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            ConversationTranscriptMetadata {
                turn_id: Some("turn-stop"),
                confidence: Some(0.91),
                stability: None,
                source_asr_session_ids: None,
                source_utterance_ids: None,
            },
        )
        .await
        .expect("current_compat should preserve one-word final barge-in behavior");

        assert!(cancel.is_canceled());
        timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("replacement speech should be queued")
            .expect("replacement speech should emit a media command");
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.last_user_text.as_deref(), Some("Stop"));
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("Stop")
        );
        assert!(call.conversation.last_error.is_none());
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
    async fn barge_in_cancel_only_policy_preserves_current_cancel_behavior() {
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
            guard.quality.config.conversation_policy.mode =
                ConversationPolicyMode::BargeInCancelOnly;
            guard.quality.config.barge_in.partial_min_confidence = Some(0.50);
            guard.quality.config.barge_in.partial_min_stability = Some(0.50);
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event_with_metadata(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Partial {
                text: "interrupt now".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            ConversationTranscriptMetadata {
                confidence: Some(0.91),
                stability: Some(0.91),
                ..ConversationTranscriptMetadata::default()
            },
        )
        .await
        .expect("barge-in cancel policy should cancel active speech");

        assert!(cancel.is_canceled());
        assert!(media_registry
            .active_speech_playback_id(&gateway_call_id)
            .await
            .is_none());
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Interrupted);
        assert!(call.conversation.last_user_text.is_none());
    }

    #[tokio::test]
    async fn barge_in_coalesce_after_silence_merges_finals_before_dispatch() {
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
            guard.quality.config.set_endpoint_merge_window_ms(0);
            guard.quality.config.conversation_policy.mode =
                ConversationPolicyMode::BargeInCoalesceAfterSilence;
            guard.quality.config.barge_in.final_min_confidence = Some(0.70);
            guard
                .quality
                .config
                .conversation_policy
                .post_barge_in_silence_ms = 50;
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event_with_metadata(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "first interruption sentence.".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            ConversationTranscriptMetadata {
                confidence: Some(0.91),
                ..ConversationTranscriptMetadata::default()
            },
        )
        .await
        .expect("first final should cancel active playback and enter coalesce window");
        assert!(cancel.is_canceled());
        assert!(
            timeout(Duration::from_millis(20), rx.recv()).await.is_err(),
            "policy should hold the first post-barge-in final until the silence window closes"
        );

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "second interruption sentence.".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("second final should merge into post-barge-in coalesced turn");

        timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("coalesced response should queue after silence window")
            .expect("coalesced response should emit a media command");
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("first interruption sentence. second interruption sentence.")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("first interruption sentence. second interruption sentence.")
        );
        assert!(call.conversation.last_error.is_none());
    }

    #[tokio::test]
    async fn barge_in_coalesce_after_silence_rearms_after_late_final() {
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
            guard.quality.config.set_endpoint_merge_window_ms(0);
            guard.quality.config.conversation_policy.mode =
                ConversationPolicyMode::BargeInCoalesceAfterSilence;
            guard.quality.config.barge_in.partial_min_confidence = Some(0.50);
            guard.quality.config.barge_in.partial_min_stability = Some(0.50);
            guard
                .quality
                .config
                .conversation_policy
                .post_barge_in_silence_ms = 100;
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event_with_metadata(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Partial {
                text: "interrupt now".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
            ConversationTranscriptMetadata {
                confidence: Some(0.91),
                stability: Some(0.91),
                ..ConversationTranscriptMetadata::default()
            },
        )
        .await
        .expect("partial should cancel active playback and open coalesce window");
        assert!(cancel.is_canceled());

        sleep(Duration::from_millis(70)).await;
        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "first interruption sentence.".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("first final should schedule post-barge-in coalescing");

        sleep(Duration::from_millis(60)).await;
        assert!(
            timeout(Duration::from_millis(10), rx.recv()).await.is_err(),
            "first final should not dispatch before its silence window closes"
        );
        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "second interruption sentence.".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("late final should merge into the pending post-barge-in turn");
        assert!(
            timeout(Duration::from_millis(70), rx.recv()).await.is_err(),
            "late final should re-arm a sliding silence window instead of dispatching immediately"
        );

        timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("coalesced response should queue after silence from the late final")
            .expect("coalesced response should emit a media command");
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("first interruption sentence. second interruption sentence.")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("first interruption sentence. second interruption sentence.")
        );
        assert!(call.conversation.last_error.is_none());
    }

    #[tokio::test]
    async fn coalesce_post_barge_in_dispatch_guard_suppresses_playback_fragments() {
        assert_post_barge_in_dispatch_guard_suppresses_playback_fragments(
            ConversationPolicyMode::BargeInCoalesceAfterSilence,
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn cancel_only_post_barge_in_dispatch_guard_suppresses_playback_fragments() {
        assert_post_barge_in_dispatch_guard_suppresses_playback_fragments(
            ConversationPolicyMode::BargeInCancelOnly,
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn post_barge_in_dispatch_guard_suppresses_recently_finished_playback_fragments() {
        assert_post_barge_in_dispatch_guard_suppresses_playback_fragments(
            ConversationPolicyMode::BargeInCancelOnly,
            true,
        )
        .await;
    }

    async fn assert_post_barge_in_dispatch_guard_suppresses_playback_fragments(
        policy_mode: ConversationPolicyMode,
        finish_replacement_before_fragment: bool,
    ) {
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
            guard.quality.config.set_endpoint_merge_window_ms(0);
            guard.quality.config.conversation_policy.mode = policy_mode;
            guard.quality.config.barge_in.final_min_confidence = Some(0.70);
            guard
                .quality
                .config
                .conversation_policy
                .post_barge_in_silence_ms = 40;
            guard
                .quality
                .config
                .conversation_policy
                .post_barge_in_echo_guard_ms = 2_000;
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event_with_metadata(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "please repeat this replacement sentence clearly after the interruption"
                    .to_string(),
                update: transcription_update_with_confidence(
                    "please repeat this replacement sentence clearly after the interruption",
                    Some(0.91),
                    true,
                ),
            },
            None,
            ConversationTranscriptMetadata {
                confidence: Some(0.91),
                ..ConversationTranscriptMetadata::default()
            },
        )
        .await
        .expect("replacement final should cancel active playback and queue replacement");
        assert!(cancel.is_canceled());

        let replacement_playback_id = match timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("replacement response should queue")
            .expect("replacement response should emit a media command")
        {
            crate::media::OutboundMediaCommand::Frame(frame) => frame.playback_id,
            other => panic!("expected replacement frame, got {other:?}"),
        };
        assert_ne!(replacement_playback_id, "tts_test");
        if finish_replacement_before_fragment {
            media_registry
                .finish_speech(&gateway_call_id, &replacement_playback_id)
                .await;
        }

        handle_transcript_event_with_metadata(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "up now".to_string(),
                update: transcription_update_with_confidence("up now", Some(0.94), true),
            },
            None,
            ConversationTranscriptMetadata {
                confidence: Some(0.94),
                ..ConversationTranscriptMetadata::default()
            },
        )
        .await
        .expect("post-playback fragment should be suppressed");

        let active_playback_after_fragment = media_registry
            .active_speech_playback_id(&gateway_call_id)
            .await;
        if finish_replacement_before_fragment {
            assert!(active_playback_after_fragment.is_none());
        } else {
            assert_eq!(
                active_playback_after_fragment.as_deref(),
                Some(replacement_playback_id.as_str())
            );
        }
        let no_new_playback = timeout(Duration::from_millis(120), async {
            while let Some(command) = rx.recv().await {
                if let crate::media::OutboundMediaCommand::Frame(frame) = command {
                    if frame.playback_id != replacement_playback_id {
                        return false;
                    }
                }
            }
            true
        })
        .await
        .unwrap_or(true);
        assert!(
            no_new_playback,
            "suppressed fragment must not queue a second response"
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("please repeat this replacement sentence clearly after the interruption")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("please repeat this replacement sentence clearly after the interruption")
        );
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
            Some(
                "You're still missing some end points always seems to be the last word that's missing"
            )
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some(
                "You're still missing some end points always seems to be the last word that's missing"
            )
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
    async fn disabled_barge_in_deferred_auto_say_drops_after_playback_max_hold() {
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
            guard
                .quality
                .config
                .set_endpoint_conversation_playback_hold_poll_ms(10);
            guard
                .quality
                .config
                .set_endpoint_conversation_playback_max_hold_ms(30);
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "drop stale repeat".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("disabled barge-in should defer overlapping final turn");
        sleep(Duration::from_millis(250)).await;

        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );
        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "max hold should drop stale deferred speech instead of queueing behind active playback"
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("drop stale repeat")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("drop stale repeat")
        );
        assert!(call.conversation.last_error.is_none());
    }

    #[tokio::test]
    async fn disabled_barge_in_bounded_pending_fifo_drains_all_repeats() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        state.write().await.record_conversation_speaking(
            &gateway_call_id,
            "assistant is speaking".to_string(),
            "tts_test".to_string(),
        );
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(16);
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
            guard
                .quality
                .config
                .set_endpoint_conversation_playback_hold_poll_ms(10);
            guard.quality.config.conversation_policy.mode =
                ConversationPolicyMode::NoBargeInBoundedPending;
            guard
                .quality
                .config
                .conversation_policy
                .active_playback_hold_ms = 0;
            guard.quality.config.conversation_policy.max_pending_outputs = 3;
            guard
                .quality
                .config
                .conversation_policy
                .pending_output_order = crate::quality::PendingOutputOrder::Fifo;
            guard.quality.config_id = guard.quality.config.config_id();
        }

        for text in ["first repeat", "second repeat", "third repeat"] {
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
            .expect("disabled barge-in should retain overlapping final turns");
        }

        sleep(Duration::from_millis(120)).await;
        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );

        media_registry
            .finish_speech(&gateway_call_id, "tts_test")
            .await;
        let first_playback =
            wait_for_active_playback_change(&media_registry, &gateway_call_id, "tts_test").await;
        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(
                call.conversation.last_assistant_text.as_deref(),
                Some("first repeat")
            );
        }

        media_registry
            .finish_speech(&gateway_call_id, &first_playback)
            .await;
        let second_playback =
            wait_for_active_playback_change(&media_registry, &gateway_call_id, &first_playback)
                .await;
        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(
                call.conversation.last_assistant_text.as_deref(),
                Some("second repeat")
            );
        }

        media_registry
            .finish_speech(&gateway_call_id, &second_playback)
            .await;
        let third_playback =
            wait_for_active_playback_change(&media_registry, &gateway_call_id, &second_playback)
                .await;
        {
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(call.conversation.status, ConversationStatus::Speaking);
            assert_eq!(
                call.conversation.last_assistant_text.as_deref(),
                Some("third repeat")
            );
            assert!(call.conversation.last_error.is_none());
        }
        media_registry
            .finish_speech(&gateway_call_id, &third_playback)
            .await;
    }

    #[tokio::test]
    async fn disabled_barge_in_bounded_pending_drops_after_active_playback_hold() {
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
            guard
                .quality
                .config
                .set_endpoint_conversation_playback_hold_poll_ms(10);
            guard.quality.config.conversation_policy.mode =
                ConversationPolicyMode::NoBargeInBoundedPending;
            guard
                .quality
                .config
                .conversation_policy
                .active_playback_hold_ms = 30;
            guard.quality.config.conversation_policy.max_pending_outputs = 1;
            guard.quality.config_id = guard.quality.config.config_id();
        }

        handle_transcript_event(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            TranscriptEvent::Final {
                text: "stale bounded repeat".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("disabled barge-in should retain overlapping final turn briefly");
        sleep(Duration::from_millis(120)).await;

        assert!(!cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some("tts_test")
        );
        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "active_playback_hold_ms should drop stale bounded output while playback remains active"
        );

        media_registry
            .finish_speech(&gateway_call_id, "tts_test")
            .await;
        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "dropped bounded output should not queue after the original playback clears"
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_user_text.as_deref(),
            Some("stale bounded repeat")
        );
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("stale bounded repeat")
        );
        assert!(call.conversation.last_error.is_none());
    }

    #[tokio::test]
    async fn cancel_active_generation_drops_stale_processor_speech() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        let media_registry = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let runtime = test_runtime_with_tts();
        let stale_context = ConversationTurnContext {
            processor_generation: Some(
                runtime.current_processor_generation(&gateway_call_id).await,
            ),
            ..ConversationTurnContext::default()
        };

        let cancel = runtime
            .cancel_active_processor_generation(&gateway_call_id)
            .await;
        assert_eq!(cancel.generation, 1);

        queue_conversation_speech(
            &state,
            &media_registry,
            &runtime,
            &gateway_call_id,
            "stale processor output".to_string(),
            SpeechConflictPolicy::Reject,
            stale_context,
        )
        .await;

        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "processor output from a canceled generation must not reach TTS"
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert!(call.conversation.last_assistant_text.is_none());
        assert!(call.tts.is_none());
    }

    #[tokio::test]
    async fn finish_deferred_policy_drain_keeps_running_when_output_arrives_during_finish() {
        let runtime = test_runtime();
        let gateway_call_id = "gwc_lost_wakeup";
        let mut policy = ConversationPolicyConfig {
            max_pending_outputs: 2,
            ..ConversationPolicyConfig::default()
        };
        policy.pending_output_order = crate::quality::PendingOutputOrder::Fifo;
        let turn_context = ConversationTurnContext::default();
        {
            let mut outputs = runtime.deferred_policy_outputs.lock().await;
            let queue = outputs.entry(gateway_call_id.to_string()).or_default();
            queue.set_drain_running(true);
            queue.enqueue(
                &policy,
                PendingConversationSay {
                    response_text: "first".to_string(),
                    turn_context: turn_context.clone(),
                },
            );
            queue.take_next();
            queue.enqueue(
                &policy,
                PendingConversationSay {
                    response_text: "second".to_string(),
                    turn_context,
                },
            );
        }

        assert!(
            finish_deferred_policy_output_drain(&runtime, gateway_call_id).await,
            "finish should report that a new pending item must be drained"
        );
        let outputs = runtime.deferred_policy_outputs.lock().await;
        let queue = outputs
            .get(gateway_call_id)
            .expect("queue should remain present for the pending output");
        assert!(queue.drain_running());
        assert_eq!(
            queue.front().expect("pending output").payload.response_text,
            "second"
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
                text: "hello there".to_string(),
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
                processor_visible_turn_at: None,
                processor_generation: None,
                turn_id: Some("turn_test".to_string()),
                coalesced_turn_ids: vec!["turn_test".to_string()],
                source_asr_session_ids: Vec::new(),
                source_utterance_ids: Vec::new(),
                metadata: QualityPlaybackMetadata::default(),
                post_barge_in_dispatch_guard: false,
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
                conversation_policy: ConversationPolicyConfig::default(),
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
                conversation_policy: ConversationPolicyConfig::default(),
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
