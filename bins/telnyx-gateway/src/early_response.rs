use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use futures_util::{stream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::media::{SharedMediaRegistry, SpeechClearReason};
use crate::operator::state::{CallStatus, SharedState, SpeechOutputConfig, TtsPlaybackStatus};
use crate::processors::{
    CommittedSpeechIntent, ConversationProcessorInput, ConversationProcessorKind,
    ConversationProcessorOutput,
};
use crate::speech::{self, AppendSpeechHandle, SpeechConflictPolicy, SpeechQueueRequest};
use crate::text_calls::turns::{CallerSpeechState, PlaybackFinishedStatus};
use crate::text_calls::SharedTextCallRegistry;
use crate::tts::SharedTtsRegistry;
use tokio::sync::{mpsc, Notify};
use tokio::time;

const EARLY_RESPONSE_INPUT_CAPACITY: usize = 8;
const EARLY_RESPONSE_EVENT_CAPACITY: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppendOrReplace {
    Append,
    Replace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyResponseAudioMode {
    SpeakProvisionally,
    PrepareOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRequirement {
    None,
    Clause,
    Sentence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingSignalPolicy {
    Conservative,
}

impl MissingSignalPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyResponseStartTiming {
    EndpointCandidateOnly,
    WhileSpeaking,
}

fn start_states_for_timing(timing: EarlyResponseStartTiming) -> Vec<CallerSpeechState> {
    match timing {
        EarlyResponseStartTiming::EndpointCandidateOnly => {
            vec![CallerSpeechState::EndpointCandidate]
        }
        EarlyResponseStartTiming::WhileSpeaking => {
            vec![
                CallerSpeechState::Speaking,
                CallerSpeechState::EndpointCandidate,
            ]
        }
    }
}

fn update_states_for_timing(timing: EarlyResponseStartTiming) -> Vec<CallerSpeechState> {
    match timing {
        EarlyResponseStartTiming::EndpointCandidateOnly => {
            vec![
                CallerSpeechState::EndpointCandidate,
                CallerSpeechState::Finalizing,
            ]
        }
        EarlyResponseStartTiming::WhileSpeaking => vec![
            CallerSpeechState::Speaking,
            CallerSpeechState::EndpointCandidate,
            CallerSpeechState::Finalizing,
        ],
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyResponseAppendMode {
    ReplaceOnly,
    PrefixMonotonicBackend,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EarlyResponsePolicy {
    pub enabled: bool,
    pub audio_mode: EarlyResponseAudioMode,
    pub min_text_chars: usize,
    pub min_text_tokens: usize,
    pub boundary: BoundaryRequirement,
    pub min_confidence: Option<f32>,
    pub min_stability: Option<f32>,
    pub missing_signal_policy: MissingSignalPolicy,
    pub allowed_start_speech_states: Vec<CallerSpeechState>,
    pub allowed_update_speech_states: Vec<CallerSpeechState>,
    pub debounce_ms: u64,
    pub max_updates_per_utterance: usize,
    pub start_timing: EarlyResponseStartTiming,
    pub append_mode: EarlyResponseAppendMode,
    pub provisional_max_prebuffer_frames: usize,
}

impl Default for EarlyResponsePolicy {
    fn default() -> Self {
        let start_timing = EarlyResponseStartTiming::WhileSpeaking;
        Self {
            enabled: false,
            audio_mode: EarlyResponseAudioMode::SpeakProvisionally,
            min_text_chars: 12,
            min_text_tokens: 3,
            boundary: BoundaryRequirement::Clause,
            min_confidence: Some(0.70),
            min_stability: Some(0.80),
            missing_signal_policy: MissingSignalPolicy::Conservative,
            allowed_start_speech_states: start_states_for_timing(start_timing),
            allowed_update_speech_states: update_states_for_timing(start_timing),
            debounce_ms: 120,
            max_updates_per_utterance: 3,
            start_timing,
            append_mode: EarlyResponseAppendMode::ReplaceOnly,
            provisional_max_prebuffer_frames: 1,
        }
    }
}

impl EarlyResponsePolicy {
    pub fn smoke_identity_test_policy() -> Self {
        Self {
            enabled: true,
            min_text_chars: 1,
            min_text_tokens: 1,
            boundary: BoundaryRequirement::None,
            min_confidence: None,
            min_stability: None,
            debounce_ms: 0,
            max_updates_per_utterance: 8,
            ..Self::default()
        }
    }

    pub fn set_start_timing(&mut self, timing: EarlyResponseStartTiming) {
        self.start_timing = timing;
        self.allowed_start_speech_states = start_states_for_timing(timing);
        self.allowed_update_speech_states = update_states_for_timing(timing);
    }

    fn allows_start_speech_state(&self, state: CallerSpeechState) -> bool {
        self.allowed_start_speech_states.contains(&state)
    }

    fn allows_update_speech_state(&self, state: CallerSpeechState) -> bool {
        self.allowed_update_speech_states.contains(&state)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EarlyResponseInput {
    Partial(EarlyResponsePartial),
    CommitBoundary(EarlyResponseCommitBoundary),
    CancelProvisional {
        call_id: String,
        provisional_turn_id: String,
        generation: u64,
        reason: EarlyResponseCancelReason,
    },
    CancelUtterance {
        call_id: String,
        utterance_id: String,
        reason: EarlyResponseCancelReason,
    },
    CancelCall {
        call_id: String,
        reason: EarlyResponseCancelReason,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct EarlyResponsePartial {
    pub call_id: String,
    pub utterance_id: String,
    pub sequence: u64,
    pub received_at_ms: u64,
    pub text: String,
    pub confidence: Option<f32>,
    pub stability: Option<f32>,
    pub speech_state: CallerSpeechState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EarlyResponseCommitBoundary {
    pub call_id: String,
    pub sequence: u64,
    pub turn_id: String,
    pub coalesced_turn_ids: Vec<String>,
    pub final_text: String,
    pub members: Vec<EarlyResponseCommitMember>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EarlyResponseCommitMember {
    pub utterance_id: String,
    pub member_index: usize,
    pub member_final_text: String,
    pub transcript_event_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EarlyResponseEvent {
    Started {
        provisional_turn_id: String,
        call_id: String,
        utterance_id: String,
        generation: u64,
        text: String,
        confidence: Option<f32>,
        stability: Option<f32>,
        speech_state: CallerSpeechState,
    },
    Updated {
        provisional_turn_id: String,
        call_id: String,
        utterance_id: String,
        generation: u64,
        text: String,
        full_text: String,
        append_or_replace: AppendOrReplace,
    },
    Committed {
        provisional_turn_id: String,
        call_id: String,
        utterance_id: String,
        generation: u64,
        turn_id: String,
        coalesced_turn_ids: Vec<String>,
        coalesced_utterance_ids: Vec<String>,
        member_final_text: String,
        final_text: String,
        role: EarlyResponseCommitRole,
    },
    Canceled {
        provisional_turn_id: String,
        call_id: String,
        utterance_id: String,
        generation: u64,
        reason: EarlyResponseCancelReason,
    },
}

impl EarlyResponseEvent {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Started { text, .. } | Self::Updated { text, .. } => Some(text),
            Self::Committed { .. } | Self::Canceled { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyResponseCommitRole {
    PrimaryPlayback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyResponseCancelReason {
    AsrCorrection,
    FinalTranscriptMismatch,
    SupersededByNewGeneration,
    PolicyDisabled,
    PolicyNoLongerSatisfied,
    MaxUpdatesExceeded,
    CallerBargeIn,
    ProcessorRejected,
    TtsCanceled,
    StaleGeneration,
    CoalescedIntoFinalTurn,
    SessionEnded,
    Hangup,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EarlyResponseIntent {
    Speak {
        provisional_turn_id: String,
        call_id: String,
        utterance_id: String,
        generation: u64,
        text: String,
        append_or_replace: AppendOrReplace,
        final_fragment: bool,
    },
    Cancel {
        provisional_turn_id: String,
        call_id: String,
        utterance_id: String,
        generation: u64,
        reason: EarlyResponseCancelReason,
    },
    Commit {
        provisional_turn_id: String,
        call_id: String,
        generation: u64,
        turn_id: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvisionalPlaybackKey {
    pub call_id: String,
    pub provisional_turn_id: String,
    pub generation: u64,
    pub playback_id: Option<String>,
}

pub trait EarlyResponsePriorityCancelSink: Send + Sync {
    fn cancel_provisional(&self, key: ProvisionalPlaybackKey, reason: EarlyResponseCancelReason);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopEarlyResponseCancelSink;

impl EarlyResponsePriorityCancelSink for NoopEarlyResponseCancelSink {
    fn cancel_provisional(&self, _key: ProvisionalPlaybackKey, _reason: EarlyResponseCancelReason) {
    }
}

pub fn aggregate_early_resp_partials<S, C>(
    partial_stream: S,
    policy: EarlyResponsePolicy,
    cancel_sink: C,
) -> impl Stream<Item = EarlyResponseEvent>
where
    S: Stream<Item = EarlyResponseInput>,
    C: EarlyResponsePriorityCancelSink,
{
    stream::unfold(
        (
            Box::pin(partial_stream),
            EarlyResponseAggregator::new(policy, cancel_sink),
            VecDeque::new(),
        ),
        |(mut partial_stream, mut aggregator, mut pending)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (partial_stream, aggregator, pending)));
                }
                let input = partial_stream.as_mut().next().await?;
                pending.extend(aggregator.handle_input(input));
            }
        },
    )
}

#[derive(Clone)]
pub struct EarlyResponsePipelineHandle {
    call_id: String,
    partial_tx: EarlyResponsePartialSender,
    control_tx: mpsc::UnboundedSender<EarlyResponseInput>,
    _processor_tx: mpsc::Sender<ConversationProcessorInput>,
}

impl EarlyResponsePipelineHandle {
    pub fn try_send(&self, input: EarlyResponseInput) -> bool {
        match input {
            EarlyResponseInput::Partial(partial) => self.partial_tx.send(partial),
            control => match self.control_tx.send(control) {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(
                        gateway_call_id = self.call_id.as_str(),
                        error = %error,
                        "early_response.control.closed"
                    );
                    false
                }
            },
        }
    }

    pub fn cancel_call(&self, reason: EarlyResponseCancelReason) {
        let _ = self.try_send(EarlyResponseInput::CancelCall {
            call_id: self.call_id.clone(),
            reason,
        });
    }

    #[cfg(test)]
    pub(crate) fn processor_input_sender(&self) -> mpsc::Sender<ConversationProcessorInput> {
        self._processor_tx.clone()
    }
}

#[derive(Default)]
struct EarlyResponsePartialLaneState {
    queue: VecDeque<EarlyResponsePartial>,
    sender_count: usize,
    closed: bool,
}

struct EarlyResponsePartialLaneInner {
    state: Mutex<EarlyResponsePartialLaneState>,
    notify: Notify,
}

#[derive(Clone)]
struct EarlyResponsePartialReceiver {
    inner: Arc<EarlyResponsePartialLaneInner>,
}

struct EarlyResponsePartialSender {
    inner: Arc<EarlyResponsePartialLaneInner>,
}

impl Clone for EarlyResponsePartialSender {
    fn clone(&self) -> Self {
        let mut guard = self
            .inner
            .state
            .lock()
            .expect("early response partial lane lock");
        guard.sender_count += 1;
        drop(guard);
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Drop for EarlyResponsePartialSender {
    fn drop(&mut self) {
        let mut guard = self
            .inner
            .state
            .lock()
            .expect("early response partial lane lock");
        guard.sender_count = guard.sender_count.saturating_sub(1);
        if guard.sender_count == 0 {
            guard.closed = true;
            drop(guard);
            self.inner.notify.notify_waiters();
        }
    }
}

impl EarlyResponsePartialSender {
    fn send(&self, partial: EarlyResponsePartial) -> bool {
        let partial_call_id = partial.call_id.clone();
        let partial_utterance_id = partial.utterance_id.clone();
        let mut guard = self
            .inner
            .state
            .lock()
            .expect("early response partial lane lock");
        if guard.closed {
            tracing::warn!(
                gateway_call_id = partial_call_id.as_str(),
                utterance_id = partial_utterance_id.as_str(),
                "early_response.partial.closed"
            );
            return false;
        }
        if let Some(queued) = guard.queue.iter_mut().find(|queued| {
            queued.call_id == partial_call_id && queued.utterance_id == partial_utterance_id
        }) {
            *queued = partial;
            drop(guard);
            self.inner.notify.notify_one();
            tracing::debug!(
                gateway_call_id = partial_call_id.as_str(),
                utterance_id = partial_utterance_id.as_str(),
                "early_response.partial.coalesced"
            );
            return true;
        }
        if guard.queue.len() >= EARLY_RESPONSE_INPUT_CAPACITY {
            if let Some(dropped) = guard.queue.pop_front() {
                tracing::warn!(
                    gateway_call_id = dropped.call_id.as_str(),
                    utterance_id = dropped.utterance_id.as_str(),
                    "early_response.partial.drop_oldest"
                );
            }
        }
        guard.queue.push_back(partial);
        drop(guard);
        self.inner.notify.notify_one();
        true
    }
}

impl EarlyResponsePartialReceiver {
    async fn recv(&self) -> Option<EarlyResponsePartial> {
        loop {
            let notified = {
                let mut guard = self
                    .inner
                    .state
                    .lock()
                    .expect("early response partial lane lock");
                if let Some(partial) = guard.queue.pop_front() {
                    return Some(partial);
                }
                if guard.closed {
                    return None;
                }
                self.inner.notify.notified()
            };
            notified.await;
        }
    }
}

fn early_response_partial_lane() -> (EarlyResponsePartialSender, EarlyResponsePartialReceiver) {
    let inner = Arc::new(EarlyResponsePartialLaneInner {
        state: Mutex::new(EarlyResponsePartialLaneState {
            sender_count: 1,
            ..EarlyResponsePartialLaneState::default()
        }),
        notify: Notify::new(),
    });
    (
        EarlyResponsePartialSender {
            inner: inner.clone(),
        },
        EarlyResponsePartialReceiver { inner },
    )
}

fn early_response_input_stream(
    partial_rx: EarlyResponsePartialReceiver,
    control_rx: mpsc::UnboundedReceiver<EarlyResponseInput>,
) -> impl Stream<Item = EarlyResponseInput> {
    stream::unfold(
        (partial_rx, control_rx, false, false),
        |(partial_rx, mut control_rx, mut partial_closed, mut control_closed)| async move {
            loop {
                if !control_closed {
                    match control_rx.try_recv() {
                        Ok(input) => {
                            return Some((
                                input,
                                (partial_rx, control_rx, partial_closed, control_closed),
                            ));
                        }
                        Err(mpsc::error::TryRecvError::Empty) => {}
                        Err(mpsc::error::TryRecvError::Disconnected) => control_closed = true,
                    }
                }
                if partial_closed && control_closed {
                    return None;
                }
                tokio::select! {
                    biased;
                    control = control_rx.recv(), if !control_closed => {
                        match control {
                            Some(input) => {
                                return Some((
                                    input,
                                    (partial_rx, control_rx, partial_closed, control_closed),
                                ));
                            }
                            None => control_closed = true,
                        }
                    }
                    partial = partial_rx.recv(), if !partial_closed => {
                        match partial {
                            Some(partial) => {
                                return Some((
                                    EarlyResponseInput::Partial(partial),
                                    (partial_rx, control_rx, partial_closed, control_closed),
                                ));
                            }
                            None => partial_closed = true,
                        }
                    }
                }
            }
        },
    )
}

pub struct EarlyResponsePipelineServices {
    pub state: SharedState,
    pub media_registry: SharedMediaRegistry,
    pub tts: SharedTtsRegistry,
    pub text_calls: SharedTextCallRegistry,
    pub speech_output: SpeechOutputConfig,
    pub processor: ConversationProcessorKind,
}

pub fn spawn_early_response_pipeline(
    call_id: String,
    policy: EarlyResponsePolicy,
    services: EarlyResponsePipelineServices,
) -> EarlyResponsePipelineHandle {
    let (partial_tx, partial_rx) = early_response_partial_lane();
    let (control_tx, control_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::channel(EARLY_RESPONSE_EVENT_CAPACITY);
    let (processor_tx, processor_rx) = mpsc::channel(EARLY_RESPONSE_EVENT_CAPACITY);
    let registry = ProvisionalPlaybackRegistry::default();
    let committed_registry = CommittedPlaybackRegistry::default();
    let cancel_sink = PipelinePriorityCancelSink {
        registry: registry.clone(),
        media_registry: services.media_registry.clone(),
    };

    let audio_mode = policy.audio_mode;
    let provisional_max_prebuffer_frames = policy.provisional_max_prebuffer_frames;
    let processor = services.processor;
    let processor_inputs = processor_input_stream(event_rx, processor_rx);
    let outputs = processor.process_stream(processor_inputs);
    let register_text_calls = services.text_calls.clone();
    let register_call_id = call_id.clone();
    let register_processor_tx = processor_tx.clone();
    tokio::spawn(async move {
        register_text_calls
            .set_processor_input_sender(&register_call_id, register_processor_tx)
            .await;
    });

    let intent_services = EarlyResponseIntentServices {
        state: services.state.clone(),
        media_registry: services.media_registry.clone(),
        tts: services.tts.clone(),
        speech_output: services.speech_output,
        text_calls: services.text_calls.clone(),
        registry: registry.clone(),
        audio_mode,
        committed_registry: committed_registry.clone(),
        provisional_max_prebuffer_frames,
    };
    tokio::spawn(async move {
        futures_util::pin_mut!(outputs);
        while let Some(output) = outputs.next().await {
            match output {
                ConversationProcessorOutput::EarlyResponse(intent) => {
                    if let Err(error) = handle_early_response_intent(&intent_services, intent).await
                    {
                        tracing::warn!(error = %error, "early_response.intent.failed");
                    }
                }
                ConversationProcessorOutput::CommittedSpeech(intent) => {
                    if let Err(error) =
                        handle_committed_speech_intent(&intent_services, intent).await
                    {
                        tracing::warn!(error = %error, "conversation.committed_speech.intent.failed");
                    }
                }
                ConversationProcessorOutput::Command(_) => {
                    tracing::warn!(
                        "early_response.processor.command_output_ignored_for_provisional_turn"
                    );
                }
                ConversationProcessorOutput::Accumulating(_)
                | ConversationProcessorOutput::PromptComplete(_)
                | ConversationProcessorOutput::Reset(_) => {
                    tracing::debug!(
                        "early_response.processor.turn_batch_output_ignored_for_provisional_turn"
                    );
                }
                ConversationProcessorOutput::Error(error) => {
                    tracing::warn!(error, "early_response.processor.failed");
                }
            }
        }
    });

    let event_registry = registry.clone();
    let event_text_calls = services.text_calls.clone();
    let event_call_id = call_id.clone();
    tokio::spawn(async move {
        let input_stream = early_response_input_stream(partial_rx, control_rx);
        let mut events = Box::pin(aggregate_early_resp_partials(
            input_stream,
            policy,
            cancel_sink,
        ));
        while let Some(event) = events.as_mut().next().await {
            event_registry.observe_event(&event);
            match &event {
                EarlyResponseEvent::Canceled {
                    call_id,
                    provisional_turn_id,
                    generation,
                    reason,
                    ..
                } => {
                    cancel_provisional_playback(
                        &event_registry,
                        &services.media_registry,
                        call_id,
                        provisional_turn_id,
                        *generation,
                        *reason,
                    );
                }
                EarlyResponseEvent::Committed {
                    call_id,
                    provisional_turn_id,
                    generation,
                    ..
                } => {
                    if let Some(handle) =
                        event_registry.finish_generation(call_id, provisional_turn_id, *generation)
                    {
                        let _ = handle.finish().await;
                    }
                }
                EarlyResponseEvent::Started { .. } | EarlyResponseEvent::Updated { .. } => {}
            }
            spawn_early_response_event_forward(
                event_text_calls.clone(),
                event_call_id.clone(),
                event.clone(),
            );
            if event_tx.send(event).await.is_err() {
                break;
            }
        }
    });

    EarlyResponsePipelineHandle {
        call_id,
        partial_tx,
        control_tx,
        _processor_tx: processor_tx,
    }
}

fn spawn_early_response_event_forward(
    text_calls: SharedTextCallRegistry,
    call_id: String,
    event: EarlyResponseEvent,
) {
    tokio::spawn(async move {
        if let Err(error) = text_calls.send_early_response_event(&call_id, event).await {
            tracing::warn!(
                gateway_call_id = call_id.as_str(),
                error = %error,
                "text_call.early_response.forward_failed"
            );
        }
    });
}

fn receiver_stream<T>(rx: mpsc::Receiver<T>) -> impl Stream<Item = T> {
    stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    })
}

fn processor_input_stream(
    event_rx: mpsc::Receiver<EarlyResponseEvent>,
    processor_rx: mpsc::Receiver<ConversationProcessorInput>,
) -> impl Stream<Item = ConversationProcessorInput> {
    stream::select(
        receiver_stream(event_rx).map(ConversationProcessorInput::EarlyResponse),
        receiver_stream(processor_rx),
    )
}

struct EarlyResponseIntentServices {
    state: SharedState,
    media_registry: SharedMediaRegistry,
    tts: SharedTtsRegistry,
    speech_output: SpeechOutputConfig,
    text_calls: SharedTextCallRegistry,
    registry: ProvisionalPlaybackRegistry,
    committed_registry: CommittedPlaybackRegistry,
    audio_mode: EarlyResponseAudioMode,
    provisional_max_prebuffer_frames: usize,
}

async fn handle_early_response_intent(
    services: &EarlyResponseIntentServices,
    intent: EarlyResponseIntent,
) -> anyhow::Result<()> {
    match intent {
        EarlyResponseIntent::Speak {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            text,
            append_or_replace,
            final_fragment,
        } => {
            services
                .registry
                .observe_agent_generation(&call_id, &provisional_turn_id, generation);
            if text.trim().is_empty()
                || !services
                    .registry
                    .is_current(&call_id, &provisional_turn_id, generation)
            {
                return Ok(());
            }
            if services.audio_mode == EarlyResponseAudioMode::PrepareOnly {
                tracing::debug!(
                    gateway_call_id = call_id.as_str(),
                    utterance_id = utterance_id.as_str(),
                    provisional_turn_id = provisional_turn_id.as_str(),
                    generation,
                    "early_response.provisional.speech_suppressed_prepare_only"
                );
                return Ok(());
            }
            match append_or_replace {
                AppendOrReplace::Append => {
                    append_provisional_speech(
                        services,
                        call_id,
                        provisional_turn_id,
                        utterance_id,
                        generation,
                        text,
                        final_fragment,
                    )
                    .await
                }
                AppendOrReplace::Replace => {
                    start_provisional_speech(
                        services,
                        call_id,
                        provisional_turn_id,
                        utterance_id,
                        generation,
                        text,
                        final_fragment,
                    )
                    .await
                }
            }
        }
        EarlyResponseIntent::Cancel {
            provisional_turn_id,
            call_id,
            generation,
            reason,
            ..
        } => {
            cancel_provisional_playback(
                &services.registry,
                &services.media_registry,
                &call_id,
                &provisional_turn_id,
                generation,
                reason,
            );
            Ok(())
        }
        EarlyResponseIntent::Commit {
            provisional_turn_id,
            call_id,
            generation,
            turn_id,
        } => {
            if let Some(handle) =
                services
                    .registry
                    .finish_generation(&call_id, &provisional_turn_id, generation)
            {
                handle.finish().await.context("finish provisional speech")?;
                tracing::info!(
                    gateway_call_id = call_id.as_str(),
                    provisional_turn_id = provisional_turn_id.as_str(),
                    generation,
                    turn_id = turn_id.as_str(),
                    playback_id = handle.playback_id.as_str(),
                    "early_response.provisional.committed"
                );
            }
            Ok(())
        }
    }
}

async fn start_provisional_speech(
    services: &EarlyResponseIntentServices,
    call_id: String,
    provisional_turn_id: String,
    utterance_id: String,
    generation: u64,
    text: String,
    final_fragment: bool,
) -> anyhow::Result<()> {
    let (handle, queued) = speech::queue_append_speech_with_request(
        &services.state,
        &services.media_registry,
        &services.tts,
        SpeechQueueRequest {
            tts_backend: services.speech_output.tts_backend,
            gateway_call_id: call_id.clone(),
            text: text.clone(),
            source_label: "early response".to_string(),
            conflict_policy: SpeechConflictPolicy::CancelAndReplace,
            turn_finalized_at: None,
            latest_turn_finalized_at: None,
            processor_visible_turn_at: None,
            barge_in_cancel_terminal_at: None,
            turn_id: Some(provisional_turn_id.clone()),
            coalesced_turn_ids: Vec::new(),
            source_asr_session_ids: Vec::new(),
            source_utterance_ids: vec![utterance_id.clone()],
            prebuffer_chunks_override: Some(services.provisional_max_prebuffer_frames),
            speech_output: Some(services.speech_output),
            metadata: crate::operator::state::QualityPlaybackMetadata::default(),
        },
        vec![text],
    )
    .await?;
    if !services
        .registry
        .is_current(&call_id, &provisional_turn_id, generation)
    {
        handle.cancel_now();
        queue_provisional_media_clear(
            &services.media_registry,
            &call_id,
            &queued.playback_id,
            EarlyResponseCancelReason::StaleGeneration,
        );
        return Ok(());
    }
    services.registry.insert_playback(
        call_id.clone(),
        provisional_turn_id.clone(),
        generation,
        handle.clone(),
    );
    services
        .text_calls
        .send_early_response_playback_started(
            &call_id,
            provisional_turn_id.clone(),
            generation,
            queued.playback_id.clone(),
        )
        .await?;
    if final_fragment {
        handle.finish().await.context("finish provisional speech")?;
    }
    tracing::info!(
        gateway_call_id = call_id.as_str(),
        utterance_id = utterance_id.as_str(),
        provisional_turn_id = provisional_turn_id.as_str(),
        generation,
        playback_id = queued.playback_id.as_str(),
        "early_response.provisional.speech_started"
    );
    Ok(())
}

async fn append_provisional_speech(
    services: &EarlyResponseIntentServices,
    call_id: String,
    provisional_turn_id: String,
    utterance_id: String,
    generation: u64,
    text: String,
    final_fragment: bool,
) -> anyhow::Result<()> {
    let Some(handle) =
        services
            .registry
            .promote_for_append(&call_id, &provisional_turn_id, generation)
    else {
        return start_provisional_speech(
            services,
            call_id,
            provisional_turn_id,
            utterance_id,
            generation,
            text,
            final_fragment,
        )
        .await;
    };
    handle
        .append_chunks(vec![text], false)
        .await
        .context("append provisional speech")?;
    if final_fragment {
        handle.finish().await.context("finish provisional speech")?;
    }
    Ok(())
}

async fn handle_committed_speech_intent(
    services: &EarlyResponseIntentServices,
    intent: CommittedSpeechIntent,
) -> anyhow::Result<()> {
    let text = intent.text.trim().to_string();
    if text.is_empty() {
        if intent.final_fragment {
            if let Some(handle) = services
                .committed_registry
                .remove(&intent.call_id, &intent.turn_id)
            {
                handle.finish().await.context("finish committed speech")?;
            }
        }
        return Ok(());
    }

    if let Some(handle) = services
        .committed_registry
        .get(&intent.call_id, &intent.turn_id)
    {
        handle
            .append_chunks(vec![text], false)
            .await
            .context("append committed speech")?;
        if intent.final_fragment {
            handle.finish().await.context("finish committed speech")?;
            services
                .committed_registry
                .remove(&intent.call_id, &intent.turn_id);
        }
        return Ok(());
    }

    let (handle, queued) = queue_committed_speech_with_media_wait(services, &intent, text).await?;
    if let Some(replaced_playback_id) = queued.replaced_playback_id.as_deref() {
        services
            .text_calls
            .send_replaced_playback_canceled(&intent.call_id, replaced_playback_id)
            .await?;
    }
    if !services
        .text_calls
        .start_agent_playback_if_active(
            &intent.call_id,
            &intent.turn_id,
            queued.playback_id.clone(),
            intent.timing,
        )
        .await?
    {
        handle.cancel_now();
        queue_provisional_media_clear(
            &services.media_registry,
            &intent.call_id,
            &queued.playback_id,
            EarlyResponseCancelReason::StaleGeneration,
        );
        return Ok(());
    }
    spawn_committed_playback_terminal_waiter(
        services.state.clone(),
        services.text_calls.clone(),
        intent.call_id.clone(),
        queued.playback_id.clone(),
        intent.config.playback_wait_timeout,
    );
    if intent.final_fragment {
        handle.finish().await.context("finish committed speech")?;
    } else {
        services
            .committed_registry
            .insert(intent.call_id, intent.turn_id, handle);
    }
    Ok(())
}

async fn queue_committed_speech_with_media_wait(
    services: &EarlyResponseIntentServices,
    intent: &CommittedSpeechIntent,
    text: String,
) -> anyhow::Result<(AppendSpeechHandle, speech::QueuedSpeech)> {
    let media_ready_deadline = Instant::now() + intent.config.media_ready_timeout;
    let playback_ready_deadline = Instant::now() + intent.config.playback_wait_timeout;
    let conflict_policy = if intent.config.latest_response_wins {
        SpeechConflictPolicy::CancelAndReplace
    } else {
        SpeechConflictPolicy::Reject
    };
    loop {
        match speech::queue_append_speech_with_request(
            &services.state,
            &services.media_registry,
            &services.tts,
            SpeechQueueRequest {
                tts_backend: intent.speech_output.tts_backend,
                gateway_call_id: intent.call_id.clone(),
                text: text.clone(),
                source_label: "text-call agent.turn".to_string(),
                conflict_policy,
                turn_finalized_at: Some(intent.timing.finalized_at),
                latest_turn_finalized_at: Some(intent.timing.finalized_at),
                processor_visible_turn_at: None,
                barge_in_cancel_terminal_at: None,
                turn_id: Some(intent.turn_id.clone()),
                coalesced_turn_ids: vec![intent.turn_id.clone()],
                source_asr_session_ids: Vec::new(),
                source_utterance_ids: Vec::new(),
                prebuffer_chunks_override: None,
                speech_output: Some(intent.speech_output),
                metadata: crate::operator::state::QualityPlaybackMetadata::default(),
            },
            vec![text.clone()],
        )
        .await
        {
            Ok(queued) => return Ok(queued),
            Err(error) => {
                let detail = format!("{error:#}");
                if detail.contains("media stream is not ready")
                    && Instant::now() < media_ready_deadline
                {
                    time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                if !intent.config.latest_response_wins
                    && detail.contains("active speech job")
                    && Instant::now() < playback_ready_deadline
                {
                    time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

fn spawn_committed_playback_terminal_waiter(
    state: SharedState,
    text_calls: SharedTextCallRegistry,
    call_id: String,
    playback_id: String,
    playback_wait_timeout: Duration,
) {
    tokio::spawn(async move {
        let deadline = Instant::now() + playback_wait_timeout;
        loop {
            if !text_calls.is_playback_active(&call_id, &playback_id).await {
                return;
            }
            if let Some(status) = playback_terminal_status(&state, &call_id, &playback_id).await {
                let _ = text_calls
                    .finish_playback(&call_id, &playback_id, status)
                    .await;
                return;
            }
            if Instant::now() >= deadline {
                let _ = text_calls
                    .finish_playback(&call_id, &playback_id, PlaybackFinishedStatus::Failed)
                    .await;
                return;
            }
            time::sleep(Duration::from_millis(100)).await;
        }
    });
}

async fn playback_terminal_status(
    state: &SharedState,
    gateway_call_id: &str,
    playback_id: &str,
) -> Option<PlaybackFinishedStatus> {
    let guard = state.read().await;
    let Some(call) = guard.calls.get(gateway_call_id) else {
        return Some(PlaybackFinishedStatus::Failed);
    };
    if matches!(call.status, CallStatus::Ended | CallStatus::Failed) {
        return Some(PlaybackFinishedStatus::Failed);
    }
    call.tts.as_ref().and_then(|tts| {
        if tts.playback_id == playback_id {
            playback_finished_status(tts.status)
        } else {
            None
        }
    })
}

fn playback_finished_status(status: TtsPlaybackStatus) -> Option<PlaybackFinishedStatus> {
    match status {
        TtsPlaybackStatus::Completed => Some(PlaybackFinishedStatus::Completed),
        TtsPlaybackStatus::Canceled => Some(PlaybackFinishedStatus::Canceled),
        TtsPlaybackStatus::Failed => Some(PlaybackFinishedStatus::Failed),
        TtsPlaybackStatus::Queued
        | TtsPlaybackStatus::Playing
        | TtsPlaybackStatus::MarkSent
        | TtsPlaybackStatus::Canceling => None,
    }
}

fn cancel_provisional_playback(
    registry: &ProvisionalPlaybackRegistry,
    media_registry: &SharedMediaRegistry,
    call_id: &str,
    provisional_turn_id: &str,
    generation: u64,
    reason: EarlyResponseCancelReason,
) {
    let Some(handle) = registry.cancel_generation(call_id, provisional_turn_id, generation) else {
        return;
    };
    handle.cancel_now();
    queue_provisional_media_clear(media_registry, call_id, &handle.playback_id, reason);
}

fn queue_provisional_media_clear(
    media_registry: &SharedMediaRegistry,
    call_id: &str,
    playback_id: &str,
    reason: EarlyResponseCancelReason,
) {
    let media_registry = media_registry.clone();
    let call_id = call_id.to_string();
    let playback_id = playback_id.to_string();
    tokio::spawn(async move {
        let reason = speech_clear_reason_for(reason);
        match media_registry
            .cancel_speech_playback_for_reason(&call_id, &playback_id, reason)
            .await
        {
            Ok(true) => {}
            Ok(false) => {}
            Err(error) => tracing::warn!(
                gateway_call_id = call_id.as_str(),
                playback_id = playback_id.as_str(),
                error = %error,
                "early_response.priority_cancel.failed"
            ),
        }
    });
}

fn speech_clear_reason_for(reason: EarlyResponseCancelReason) -> SpeechClearReason {
    match reason {
        EarlyResponseCancelReason::CallerBargeIn => SpeechClearReason::BargeIn,
        _ => SpeechClearReason::CancelAndReplace,
    }
}

#[derive(Clone, Debug, Eq)]
struct ProvisionalBaseKey {
    call_id: String,
    provisional_turn_id: String,
}

impl PartialEq for ProvisionalBaseKey {
    fn eq(&self, other: &Self) -> bool {
        self.call_id == other.call_id && self.provisional_turn_id == other.provisional_turn_id
    }
}

impl Hash for ProvisionalBaseKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.call_id.hash(state);
        self.provisional_turn_id.hash(state);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ProvisionalGenerationKey {
    call_id: String,
    provisional_turn_id: String,
    generation: u64,
}

#[derive(Clone, Debug)]
struct ProvisionalPlayback {
    generation: u64,
    handle: AppendSpeechHandle,
}

#[derive(Default)]
struct ProvisionalPlaybackState {
    latest: HashMap<ProvisionalBaseKey, u64>,
    canceled: HashSet<ProvisionalGenerationKey>,
    playbacks: HashMap<ProvisionalBaseKey, ProvisionalPlayback>,
}

#[derive(Clone, Default)]
struct ProvisionalPlaybackRegistry {
    inner: Arc<Mutex<ProvisionalPlaybackState>>,
}

impl ProvisionalPlaybackRegistry {
    fn observe_event(&self, event: &EarlyResponseEvent) {
        let mut guard = self.inner.lock().expect("early response registry lock");
        match event {
            EarlyResponseEvent::Started {
                call_id,
                provisional_turn_id,
                generation,
                ..
            }
            | EarlyResponseEvent::Updated {
                call_id,
                provisional_turn_id,
                generation,
                ..
            }
            | EarlyResponseEvent::Committed {
                call_id,
                provisional_turn_id,
                generation,
                ..
            } => {
                guard.latest.insert(
                    ProvisionalBaseKey {
                        call_id: call_id.clone(),
                        provisional_turn_id: provisional_turn_id.clone(),
                    },
                    *generation,
                );
            }
            EarlyResponseEvent::Canceled {
                call_id,
                provisional_turn_id,
                generation,
                ..
            } => {
                guard.canceled.insert(ProvisionalGenerationKey {
                    call_id: call_id.clone(),
                    provisional_turn_id: provisional_turn_id.clone(),
                    generation: *generation,
                });
            }
        }
    }

    fn observe_agent_generation(&self, call_id: &str, provisional_turn_id: &str, generation: u64) {
        let mut guard = self.inner.lock().expect("early response registry lock");
        let base = ProvisionalBaseKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
        };
        let entry = guard.latest.entry(base).or_insert(generation);
        if generation > *entry {
            *entry = generation;
        }
    }

    fn is_current(&self, call_id: &str, provisional_turn_id: &str, generation: u64) -> bool {
        let guard = self.inner.lock().expect("early response registry lock");
        let base = ProvisionalBaseKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
        };
        let gen = ProvisionalGenerationKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
            generation,
        };
        guard.latest.get(&base).copied() == Some(generation) && !guard.canceled.contains(&gen)
    }

    fn insert_playback(
        &self,
        call_id: String,
        provisional_turn_id: String,
        generation: u64,
        handle: AppendSpeechHandle,
    ) {
        let mut guard = self.inner.lock().expect("early response registry lock");
        let base = ProvisionalBaseKey {
            call_id,
            provisional_turn_id,
        };
        if let Some(previous) = guard
            .playbacks
            .insert(base, ProvisionalPlayback { generation, handle })
        {
            previous.handle.cancel_now();
        }
    }

    fn promote_for_append(
        &self,
        call_id: &str,
        provisional_turn_id: &str,
        generation: u64,
    ) -> Option<AppendSpeechHandle> {
        let mut guard = self.inner.lock().expect("early response registry lock");
        let base = ProvisionalBaseKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
        };
        let playback = guard.playbacks.get_mut(&base)?;
        if generation < playback.generation {
            return None;
        }
        playback.generation = generation;
        Some(playback.handle.clone())
    }

    fn finish_generation(
        &self,
        call_id: &str,
        provisional_turn_id: &str,
        generation: u64,
    ) -> Option<AppendSpeechHandle> {
        let mut guard = self.inner.lock().expect("early response registry lock");
        let base = ProvisionalBaseKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
        };
        let playback = guard.playbacks.remove(&base)?;
        (playback.generation == generation).then_some(playback.handle)
    }

    fn cancel_generation(
        &self,
        call_id: &str,
        provisional_turn_id: &str,
        generation: u64,
    ) -> Option<AppendSpeechHandle> {
        let mut guard = self.inner.lock().expect("early response registry lock");
        guard.canceled.insert(ProvisionalGenerationKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
            generation,
        });
        let base = ProvisionalBaseKey {
            call_id: call_id.to_string(),
            provisional_turn_id: provisional_turn_id.to_string(),
        };
        let playback = guard.playbacks.remove(&base)?;
        (playback.generation == generation).then_some(playback.handle)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CommittedPlaybackKey {
    call_id: String,
    turn_id: String,
}

#[derive(Clone, Default)]
struct CommittedPlaybackRegistry {
    inner: Arc<Mutex<HashMap<CommittedPlaybackKey, AppendSpeechHandle>>>,
}

impl CommittedPlaybackRegistry {
    fn get(&self, call_id: &str, turn_id: &str) -> Option<AppendSpeechHandle> {
        self.inner
            .lock()
            .expect("committed playback registry lock")
            .get(&CommittedPlaybackKey {
                call_id: call_id.to_string(),
                turn_id: turn_id.to_string(),
            })
            .cloned()
    }

    fn insert(&self, call_id: String, turn_id: String, handle: AppendSpeechHandle) {
        self.inner
            .lock()
            .expect("committed playback registry lock")
            .insert(CommittedPlaybackKey { call_id, turn_id }, handle);
    }

    fn remove(&self, call_id: &str, turn_id: &str) -> Option<AppendSpeechHandle> {
        self.inner
            .lock()
            .expect("committed playback registry lock")
            .remove(&CommittedPlaybackKey {
                call_id: call_id.to_string(),
                turn_id: turn_id.to_string(),
            })
    }
}

#[derive(Clone)]
struct PipelinePriorityCancelSink {
    registry: ProvisionalPlaybackRegistry,
    media_registry: SharedMediaRegistry,
}

impl EarlyResponsePriorityCancelSink for PipelinePriorityCancelSink {
    fn cancel_provisional(&self, key: ProvisionalPlaybackKey, reason: EarlyResponseCancelReason) {
        cancel_provisional_playback(
            &self.registry,
            &self.media_registry,
            &key.call_id,
            &key.provisional_turn_id,
            key.generation,
            reason,
        );
    }
}

#[derive(Clone, Debug)]
pub struct EarlyResponseAggregator<C> {
    policy: EarlyResponsePolicy,
    cancel_sink: C,
    active: HashMap<UtteranceKey, ActiveProvisional>,
}

impl<C> EarlyResponseAggregator<C>
where
    C: EarlyResponsePriorityCancelSink,
{
    pub fn new(policy: EarlyResponsePolicy, cancel_sink: C) -> Self {
        Self {
            policy,
            cancel_sink,
            active: HashMap::new(),
        }
    }

    pub fn handle_input(&mut self, input: EarlyResponseInput) -> Vec<EarlyResponseEvent> {
        match input {
            EarlyResponseInput::Partial(partial) => self.handle_partial(partial),
            EarlyResponseInput::CommitBoundary(boundary) => self.handle_commit_boundary(boundary),
            EarlyResponseInput::CancelProvisional {
                call_id,
                provisional_turn_id,
                generation,
                reason,
            } => self.cancel_provisional(&call_id, &provisional_turn_id, generation, reason),
            EarlyResponseInput::CancelUtterance {
                call_id,
                utterance_id,
                reason,
            } => self.cancel_utterance(&call_id, &utterance_id, reason),
            EarlyResponseInput::CancelCall { call_id, reason } => {
                self.cancel_call(&call_id, reason)
            }
        }
    }

    fn handle_partial(&mut self, partial: EarlyResponsePartial) -> Vec<EarlyResponseEvent> {
        let key = UtteranceKey::new(&partial.call_id, &partial.utterance_id);
        let active = self.active.get(&key).cloned();
        if !self.policy.enabled {
            return active
                .map(|active| {
                    self.cancel_active(key, active, EarlyResponseCancelReason::PolicyDisabled)
                })
                .unwrap_or_default();
        }
        if active.is_none() && !self.policy.allows_start_speech_state(partial.speech_state) {
            return Vec::new();
        }
        if active.is_some() && !self.policy.allows_update_speech_state(partial.speech_state) {
            return active
                .map(|active| {
                    self.cancel_active(
                        key,
                        active,
                        EarlyResponseCancelReason::PolicyNoLongerSatisfied,
                    )
                })
                .unwrap_or_default();
        }
        if active.is_none() && matches!(partial.speech_state, CallerSpeechState::Finalizing) {
            return Vec::new();
        }
        let accepted_text = partial.text.trim().to_string();
        if !self.policy_accepts_text(&accepted_text)
            || !self.policy_accepts_score(partial.confidence, self.policy.min_confidence)
            || !self.policy_accepts_score(partial.stability, self.policy.min_stability)
        {
            return if let Some(active) = active {
                self.cancel_active(
                    key,
                    active,
                    EarlyResponseCancelReason::PolicyNoLongerSatisfied,
                )
            } else {
                Vec::new()
            };
        }
        match active {
            None => {
                let provisional_turn_id = provisional_turn_id_for(&partial);
                let active = ActiveProvisional {
                    provisional_turn_id: provisional_turn_id.clone(),
                    call_id: partial.call_id.clone(),
                    utterance_id: partial.utterance_id.clone(),
                    generation: 1,
                    text: accepted_text.clone(),
                    last_emitted_at_ms: partial.received_at_ms,
                    updates: 0,
                };
                self.active.insert(key, active);
                vec![EarlyResponseEvent::Started {
                    provisional_turn_id,
                    call_id: partial.call_id,
                    utterance_id: partial.utterance_id,
                    generation: 1,
                    text: accepted_text,
                    confidence: partial.confidence,
                    stability: partial.stability,
                    speech_state: partial.speech_state,
                }]
            }
            Some(mut active) => {
                if active.text == accepted_text {
                    return Vec::new();
                }
                if partial
                    .received_at_ms
                    .saturating_sub(active.last_emitted_at_ms)
                    < self.policy.debounce_ms
                {
                    return Vec::new();
                }
                if active.updates >= self.policy.max_updates_per_utterance {
                    // The update cap is a churn guard. Canceling beats freezing a stale
                    // provisional generation that may already have reached audio.
                    return self.cancel_active(
                        key,
                        active,
                        EarlyResponseCancelReason::MaxUpdatesExceeded,
                    );
                }
                let previous_text = active.text.clone();
                let append_or_replace =
                    append_or_replace_for(self.policy.append_mode, &previous_text, &accepted_text);
                if matches!(append_or_replace, AppendOrReplace::Replace) {
                    self.cancel_sink.cancel_provisional(
                        ProvisionalPlaybackKey {
                            call_id: active.call_id.clone(),
                            provisional_turn_id: active.provisional_turn_id.clone(),
                            generation: active.generation,
                            playback_id: None,
                        },
                        EarlyResponseCancelReason::SupersededByNewGeneration,
                    );
                }
                active.generation += 1;
                active.updates += 1;
                active.last_emitted_at_ms = partial.received_at_ms;
                let output_text =
                    updated_output_text(append_or_replace, &previous_text, &accepted_text);
                active.text = accepted_text.clone();
                self.active.insert(key, active.clone());
                vec![EarlyResponseEvent::Updated {
                    provisional_turn_id: active.provisional_turn_id,
                    call_id: active.call_id,
                    utterance_id: active.utterance_id,
                    generation: active.generation,
                    text: output_text,
                    full_text: accepted_text,
                    append_or_replace,
                }]
            }
        }
    }

    fn handle_commit_boundary(
        &mut self,
        boundary: EarlyResponseCommitBoundary,
    ) -> Vec<EarlyResponseEvent> {
        let mut candidates = Vec::new();
        let coalesced_utterance_ids = boundary
            .members
            .iter()
            .map(|member| member.utterance_id.clone())
            .collect::<Vec<_>>();

        for member in &boundary.members {
            let key = UtteranceKey::new(&boundary.call_id, &member.utterance_id);
            if let Some(active) = self.active.get(&key) {
                if compatible_with_final(&active.text, &member.member_final_text) {
                    candidates.push(CommitCandidate {
                        key,
                        active: active.clone(),
                        member: member.clone(),
                        prefix_coverage: normalized_prefix_coverage(
                            &active.text,
                            &boundary.final_text,
                        ),
                    });
                }
            }
        }

        candidates.retain(|candidate| candidate.prefix_coverage > 0);
        candidates.sort_by_key(|candidate| {
            (
                Reverse(candidate.prefix_coverage),
                candidate.member.member_index,
                Reverse(candidate.active.generation),
            )
        });
        let primary_key = candidates.first().map(|candidate| candidate.key.clone());
        let mut events = Vec::new();

        for member in &boundary.members {
            let key = UtteranceKey::new(&boundary.call_id, &member.utterance_id);
            let Some(active) = self.active.remove(&key) else {
                continue;
            };
            if Some(&key) == primary_key.as_ref() {
                events.push(EarlyResponseEvent::Committed {
                    provisional_turn_id: active.provisional_turn_id,
                    call_id: active.call_id,
                    utterance_id: active.utterance_id,
                    generation: active.generation,
                    turn_id: boundary.turn_id.clone(),
                    coalesced_turn_ids: boundary.coalesced_turn_ids.clone(),
                    coalesced_utterance_ids: coalesced_utterance_ids.clone(),
                    member_final_text: member.member_final_text.clone(),
                    final_text: boundary.final_text.clone(),
                    role: EarlyResponseCommitRole::PrimaryPlayback,
                });
            } else if compatible_with_final(&active.text, &member.member_final_text) {
                events.extend(self.cancel_removed_active(
                    active,
                    EarlyResponseCancelReason::CoalescedIntoFinalTurn,
                ));
            } else {
                events.extend(self.cancel_removed_active(
                    active,
                    EarlyResponseCancelReason::FinalTranscriptMismatch,
                ));
            }
        }
        events
    }

    fn cancel_provisional(
        &mut self,
        call_id: &str,
        provisional_turn_id: &str,
        generation: u64,
        reason: EarlyResponseCancelReason,
    ) -> Vec<EarlyResponseEvent> {
        let Some((key, active)) = self
            .active
            .iter()
            .find(|(_, active)| {
                active.call_id == call_id
                    && active.provisional_turn_id == provisional_turn_id
                    && active.generation == generation
            })
            .map(|(key, active)| (key.clone(), active.clone()))
        else {
            self.cancel_sink.cancel_provisional(
                ProvisionalPlaybackKey {
                    call_id: call_id.to_string(),
                    provisional_turn_id: provisional_turn_id.to_string(),
                    generation,
                    playback_id: None,
                },
                reason,
            );
            return vec![EarlyResponseEvent::Canceled {
                provisional_turn_id: provisional_turn_id.to_string(),
                call_id: call_id.to_string(),
                utterance_id: provisional_turn_id.to_string(),
                generation,
                reason,
            }];
        };
        self.cancel_active(key, active, reason)
    }

    fn cancel_utterance(
        &mut self,
        call_id: &str,
        utterance_id: &str,
        reason: EarlyResponseCancelReason,
    ) -> Vec<EarlyResponseEvent> {
        let key = UtteranceKey::new(call_id, utterance_id);
        self.active
            .get(&key)
            .cloned()
            .map(|active| self.cancel_active(key, active, reason))
            .unwrap_or_default()
    }

    fn cancel_call(
        &mut self,
        call_id: &str,
        reason: EarlyResponseCancelReason,
    ) -> Vec<EarlyResponseEvent> {
        let keys = self
            .active
            .keys()
            .filter(|key| key.call_id == call_id)
            .cloned()
            .collect::<Vec<_>>();
        let mut events = Vec::new();
        for key in keys {
            if let Some(active) = self.active.get(&key).cloned() {
                events.extend(self.cancel_active(key, active, reason));
            }
        }
        events
    }

    fn cancel_active(
        &mut self,
        key: UtteranceKey,
        active: ActiveProvisional,
        reason: EarlyResponseCancelReason,
    ) -> Vec<EarlyResponseEvent> {
        self.active.remove(&key);
        self.cancel_removed_active(active, reason)
    }

    fn cancel_removed_active(
        &self,
        active: ActiveProvisional,
        reason: EarlyResponseCancelReason,
    ) -> Vec<EarlyResponseEvent> {
        self.cancel_sink.cancel_provisional(
            ProvisionalPlaybackKey {
                call_id: active.call_id.clone(),
                provisional_turn_id: active.provisional_turn_id.clone(),
                generation: active.generation,
                playback_id: None,
            },
            reason,
        );
        vec![EarlyResponseEvent::Canceled {
            provisional_turn_id: active.provisional_turn_id,
            call_id: active.call_id,
            utterance_id: active.utterance_id,
            generation: active.generation,
            reason,
        }]
    }

    fn policy_accepts_text(&self, text: &str) -> bool {
        text.chars().count() >= self.policy.min_text_chars
            && text.split_whitespace().count() >= self.policy.min_text_tokens
            && boundary_matches(self.policy.boundary, text)
    }

    fn policy_accepts_score(&self, score: Option<f32>, threshold: Option<f32>) -> bool {
        match (score, threshold, self.policy.missing_signal_policy) {
            (_, None, _) => true,
            (Some(score), Some(threshold), _) => {
                score.is_finite()
                    && threshold.is_finite()
                    && (0.0..=1.0).contains(&score)
                    && (0.0..=1.0).contains(&threshold)
                    && score >= threshold
            }
            (None, Some(_), MissingSignalPolicy::Conservative) => false,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct UtteranceKey {
    call_id: String,
    utterance_id: String,
}

impl UtteranceKey {
    fn new(call_id: &str, utterance_id: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            utterance_id: utterance_id.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct ActiveProvisional {
    provisional_turn_id: String,
    call_id: String,
    utterance_id: String,
    generation: u64,
    text: String,
    last_emitted_at_ms: u64,
    updates: usize,
}

#[derive(Clone, Debug)]
struct CommitCandidate {
    key: UtteranceKey,
    active: ActiveProvisional,
    member: EarlyResponseCommitMember,
    prefix_coverage: usize,
}

fn provisional_turn_id_for(partial: &EarlyResponsePartial) -> String {
    let mut hasher = Sha256::new();
    hasher.update(partial.call_id.as_bytes());
    hasher.update([0]);
    hasher.update(partial.utterance_id.as_bytes());
    hasher.update([0]);
    hasher.update(partial.sequence.to_be_bytes());
    let digest = hasher.finalize();
    format!("pt_{}", hex::encode(&digest[..16]))
}

fn boundary_matches(requirement: BoundaryRequirement, text: &str) -> bool {
    match requirement {
        BoundaryRequirement::None => true,
        BoundaryRequirement::Clause => text
            .trim_end()
            .chars()
            .next_back()
            .is_some_and(|ch| matches!(ch, '.' | '!' | '?' | ',' | ';')),
        BoundaryRequirement::Sentence => text
            .trim_end()
            .chars()
            .next_back()
            .is_some_and(|ch| matches!(ch, '.' | '!' | '?')),
    }
}

fn append_or_replace_for(
    mode: EarlyResponseAppendMode,
    previous_text: &str,
    accepted_text: &str,
) -> AppendOrReplace {
    match mode {
        EarlyResponseAppendMode::ReplaceOnly => AppendOrReplace::Replace,
        EarlyResponseAppendMode::PrefixMonotonicBackend => {
            if accepted_text.starts_with(previous_text) && accepted_text.len() > previous_text.len()
            {
                AppendOrReplace::Append
            } else {
                AppendOrReplace::Replace
            }
        }
    }
}

fn updated_output_text(
    append_or_replace: AppendOrReplace,
    previous_text: &str,
    accepted_text: &str,
) -> String {
    match append_or_replace {
        AppendOrReplace::Replace => accepted_text.to_string(),
        AppendOrReplace::Append => accepted_text
            .strip_prefix(previous_text)
            .unwrap_or(accepted_text)
            .trim_start()
            .to_string(),
    }
}

fn compatible_with_final(provisional_text: &str, final_text: &str) -> bool {
    let provisional = normalize_for_reconciliation(provisional_text);
    let final_text = normalize_for_reconciliation(final_text);
    !provisional.is_empty() && final_text.starts_with(&provisional)
}

fn normalized_prefix_coverage(provisional_text: &str, final_text: &str) -> usize {
    if compatible_with_final(provisional_text, final_text) {
        normalize_for_reconciliation(provisional_text).len()
    } else {
        0
    }
}

fn normalize_for_reconciliation(text: &str) -> String {
    let mut normalized = String::new();
    let mut previous_space = true;
    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            normalized.push(ch);
            previous_space = false;
        } else if !previous_space {
            normalized.push(' ');
            previous_space = true;
        }
    }
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::StreamExt;

    use super::*;
    use crate::media::SharedMediaRegistry;
    use crate::operator::state::shared_state;
    use crate::text_calls::SharedTextCallRegistry;
    use crate::tts::{unavailable_registry, LiveTtsBackend};

    #[derive(Clone, Default)]
    struct RecordingCancelSink {
        calls: Arc<Mutex<Vec<(ProvisionalPlaybackKey, EarlyResponseCancelReason)>>>,
    }

    impl RecordingCancelSink {
        fn calls(&self) -> Vec<(ProvisionalPlaybackKey, EarlyResponseCancelReason)> {
            self.calls.lock().expect("recording sink lock").clone()
        }
    }

    impl EarlyResponsePriorityCancelSink for RecordingCancelSink {
        fn cancel_provisional(
            &self,
            key: ProvisionalPlaybackKey,
            reason: EarlyResponseCancelReason,
        ) {
            self.calls
                .lock()
                .expect("recording sink lock")
                .push((key, reason));
        }
    }

    fn partial(text: &str, sequence: u64, received_at_ms: u64) -> EarlyResponsePartial {
        EarlyResponsePartial {
            call_id: "call-1".to_string(),
            utterance_id: "utt-1".to_string(),
            sequence,
            received_at_ms,
            text: text.to_string(),
            confidence: Some(0.91),
            stability: Some(0.86),
            speech_state: CallerSpeechState::EndpointCandidate,
        }
    }

    fn enabled_policy() -> EarlyResponsePolicy {
        EarlyResponsePolicy {
            enabled: true,
            debounce_ms: 0,
            ..EarlyResponsePolicy::default()
        }
    }

    #[test]
    fn partial_starts_provisional_when_policy_gates_pass() {
        let mut aggregator =
            EarlyResponseAggregator::new(enabled_policy(), NoopEarlyResponseCancelSink);
        let first = partial("I need a tow truck.", 7, 100);
        let provisional_turn_id = provisional_turn_id_for(&first);
        let events = aggregator.handle_input(EarlyResponseInput::Partial(first));

        assert_eq!(
            events,
            vec![EarlyResponseEvent::Started {
                provisional_turn_id,
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 1,
                text: "I need a tow truck.".to_string(),
                confidence: Some(0.91),
                stability: Some(0.86),
                speech_state: CallerSpeechState::EndpointCandidate,
            }]
        );
    }

    #[test]
    fn clause_boundary_rejects_unpunctuated_partial() {
        let mut aggregator =
            EarlyResponseAggregator::new(enabled_policy(), NoopEarlyResponseCancelSink);

        assert!(aggregator
            .handle_input(EarlyResponseInput::Partial(partial(
                "I need a tow truck",
                7,
                100
            )))
            .is_empty());
    }

    #[test]
    fn none_boundary_can_start_from_unpunctuated_partial() {
        let mut policy = enabled_policy();
        policy.boundary = BoundaryRequirement::None;
        let mut aggregator = EarlyResponseAggregator::new(policy, NoopEarlyResponseCancelSink);

        assert!(matches!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(partial("I need a tow truck", 7, 100)))
                .as_slice(),
            [EarlyResponseEvent::Started { text, .. }] if text == "I need a tow truck"
        ));
    }

    #[tokio::test]
    async fn prepare_only_suppresses_provisional_gateway_tts() {
        let registry = ProvisionalPlaybackRegistry::default();
        registry.observe_event(&EarlyResponseEvent::Started {
            provisional_turn_id: "pt-1".to_string(),
            call_id: "call-1".to_string(),
            utterance_id: "utt-1".to_string(),
            generation: 1,
            text: "hello".to_string(),
            confidence: Some(0.9),
            stability: Some(0.9),
            speech_state: CallerSpeechState::EndpointCandidate,
        });
        let services = EarlyResponseIntentServices {
            state: shared_state("127.0.0.1:0".parse().expect("valid addr")),
            media_registry: SharedMediaRegistry::default(),
            tts: unavailable_registry(),
            speech_output: SpeechOutputConfig::from_quality(
                LiveTtsBackend::Piper,
                &crate::quality::TtsQualityConfig::default(),
            ),
            text_calls: SharedTextCallRegistry::default(),
            registry: registry.clone(),
            committed_registry: CommittedPlaybackRegistry::default(),
            audio_mode: EarlyResponseAudioMode::PrepareOnly,
            provisional_max_prebuffer_frames: 1,
        };

        handle_early_response_intent(
            &services,
            EarlyResponseIntent::Speak {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 1,
                text: "hello".to_string(),
                append_or_replace: AppendOrReplace::Replace,
                final_fragment: false,
            },
        )
        .await
        .expect("prepare-only suppresses provisional speech");

        assert!(registry.promote_for_append("call-1", "pt-1", 1).is_none());
    }

    #[test]
    fn default_policy_can_start_while_caller_is_speaking() {
        let mut speaking = partial("I need a tow truck.", 7, 100);
        speaking.speech_state = CallerSpeechState::Speaking;
        let mut aggregator =
            EarlyResponseAggregator::new(enabled_policy(), NoopEarlyResponseCancelSink);

        assert!(matches!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(speaking))
                .as_slice(),
            [EarlyResponseEvent::Started {
                speech_state: CallerSpeechState::Speaking,
                ..
            }]
        ));
    }

    #[test]
    fn endpoint_candidate_only_timing_does_not_start_on_speaking_partial() {
        let mut policy = enabled_policy();
        policy.set_start_timing(EarlyResponseStartTiming::EndpointCandidateOnly);
        let mut speaking = partial("I need a tow truck.", 7, 100);
        speaking.speech_state = CallerSpeechState::Speaking;
        let mut aggregator = EarlyResponseAggregator::new(policy, NoopEarlyResponseCancelSink);

        assert!(aggregator
            .handle_input(EarlyResponseInput::Partial(speaking))
            .is_empty());
    }

    #[test]
    fn missing_required_score_fails_closed() {
        let mut no_confidence = partial("I need a tow truck.", 7, 100);
        no_confidence.confidence = None;
        let mut aggregator =
            EarlyResponseAggregator::new(enabled_policy(), NoopEarlyResponseCancelSink);

        assert!(aggregator
            .handle_input(EarlyResponseInput::Partial(no_confidence))
            .is_empty());
    }

    #[test]
    fn finalizing_partial_cannot_start_new_provisional() {
        let mut finalizing = partial("I need a tow truck.", 7, 100);
        finalizing.speech_state = CallerSpeechState::Finalizing;
        let mut aggregator =
            EarlyResponseAggregator::new(enabled_policy(), NoopEarlyResponseCancelSink);

        assert!(aggregator
            .handle_input(EarlyResponseInput::Partial(finalizing))
            .is_empty());
    }

    #[test]
    fn later_partial_replaces_prior_generation_and_priority_cancels_old_audio() {
        let sink = RecordingCancelSink::default();
        let mut aggregator = EarlyResponseAggregator::new(enabled_policy(), sink.clone());
        let first = partial("I need a tow truck.", 7, 100);
        let provisional_turn_id = provisional_turn_id_for(&first);
        assert_eq!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(first))
                .len(),
            1
        );
        let events = aggregator.handle_input(EarlyResponseInput::Partial(partial(
            "I need a tow truck in Oakland.",
            8,
            250,
        )));

        assert_eq!(
            events,
            vec![EarlyResponseEvent::Updated {
                provisional_turn_id: provisional_turn_id.clone(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                text: "I need a tow truck in Oakland.".to_string(),
                full_text: "I need a tow truck in Oakland.".to_string(),
                append_or_replace: AppendOrReplace::Replace,
            }]
        );
        assert_eq!(
            sink.calls(),
            vec![(
                ProvisionalPlaybackKey {
                    call_id: "call-1".to_string(),
                    provisional_turn_id: provisional_turn_id.clone(),
                    generation: 1,
                    playback_id: None,
                },
                EarlyResponseCancelReason::SupersededByNewGeneration,
            )]
        );
    }

    #[test]
    fn debounce_suppresses_too_early_update() {
        let policy = EarlyResponsePolicy {
            enabled: true,
            debounce_ms: 120,
            ..EarlyResponsePolicy::default()
        };
        let mut aggregator = EarlyResponseAggregator::new(policy, NoopEarlyResponseCancelSink);
        assert_eq!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(partial(
                    "I need a tow truck.",
                    7,
                    100,
                )))
                .len(),
            1
        );

        assert!(aggregator
            .handle_input(EarlyResponseInput::Partial(partial(
                "I need a tow truck in Oakland.",
                8,
                180,
            )))
            .is_empty());
    }

    #[test]
    fn provisional_turn_ids_are_collision_resistant_for_punctuation_variants() {
        let mut dotted = partial("I need a tow truck.", 7, 100);
        dotted.call_id = "call.1".to_string();
        let mut slashed = partial("I need a tow truck.", 7, 100);
        slashed.call_id = "call/1".to_string();

        assert_ne!(
            provisional_turn_id_for(&dotted),
            provisional_turn_id_for(&slashed)
        );
    }

    #[test]
    fn prefix_append_update_emits_suffix_only() {
        let policy = EarlyResponsePolicy {
            enabled: true,
            debounce_ms: 0,
            append_mode: EarlyResponseAppendMode::PrefixMonotonicBackend,
            ..EarlyResponsePolicy::default()
        };
        let sink = RecordingCancelSink::default();
        let mut aggregator = EarlyResponseAggregator::new(policy, sink.clone());
        assert_eq!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(partial(
                    "I need a tow truck.",
                    7,
                    100,
                )))
                .len(),
            1
        );

        let events = aggregator.handle_input(EarlyResponseInput::Partial(partial(
            "I need a tow truck. now.",
            8,
            250,
        )));

        assert!(matches!(
            events.as_slice(),
            [EarlyResponseEvent::Updated {
                text,
                full_text,
                append_or_replace: AppendOrReplace::Append,
                ..
            }] if text == "now." && full_text == "I need a tow truck. now."
        ));
        assert!(sink.calls().is_empty());
    }

    #[test]
    fn max_updates_cancel_active_provisional() {
        let policy = EarlyResponsePolicy {
            enabled: true,
            debounce_ms: 0,
            max_updates_per_utterance: 1,
            ..EarlyResponsePolicy::default()
        };
        let sink = RecordingCancelSink::default();
        let mut aggregator = EarlyResponseAggregator::new(policy, sink.clone());
        assert_eq!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(partial(
                    "I need a tow truck.",
                    7,
                    100,
                )))
                .len(),
            1
        );
        assert_eq!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(partial(
                    "I need a tow truck now.",
                    8,
                    250,
                )))
                .len(),
            1
        );

        let events = aggregator.handle_input(EarlyResponseInput::Partial(partial(
            "I need a tow truck now please.",
            9,
            400,
        )));

        assert!(matches!(
            events.as_slice(),
            [EarlyResponseEvent::Canceled {
                reason: EarlyResponseCancelReason::MaxUpdatesExceeded,
                ..
            }]
        ));
        assert_eq!(
            sink.calls().last().map(|(_, reason)| *reason),
            Some(EarlyResponseCancelReason::MaxUpdatesExceeded)
        );
    }

    #[test]
    fn commit_boundary_commits_primary_and_cancels_mismatches() {
        let sink = RecordingCancelSink::default();
        let mut aggregator = EarlyResponseAggregator::new(enabled_policy(), sink.clone());
        let first = partial("I need a tow truck.", 7, 100);
        let first_provisional_id = provisional_turn_id_for(&first);
        aggregator.handle_input(EarlyResponseInput::Partial(first));
        let mut second = partial("Wrong city downtown.", 9, 100);
        second.utterance_id = "utt-2".to_string();
        let second_provisional_id = provisional_turn_id_for(&second);
        aggregator.handle_input(EarlyResponseInput::Partial(second));

        let events = aggregator.handle_input(EarlyResponseInput::CommitBoundary(
            EarlyResponseCommitBoundary {
                call_id: "call-1".to_string(),
                sequence: 10,
                turn_id: "turn-1".to_string(),
                coalesced_turn_ids: vec!["turn-frag-1".to_string()],
                final_text: "I need a tow truck in Oakland.".to_string(),
                members: vec![
                    EarlyResponseCommitMember {
                        utterance_id: "utt-1".to_string(),
                        member_index: 0,
                        member_final_text: "I need a tow truck in Oakland.".to_string(),
                        transcript_event_ids: vec!["event-1".to_string()],
                    },
                    EarlyResponseCommitMember {
                        utterance_id: "utt-2".to_string(),
                        member_index: 1,
                        member_final_text: "Correct city downtown.".to_string(),
                        transcript_event_ids: vec!["event-2".to_string()],
                    },
                ],
            },
        ));

        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            EarlyResponseEvent::Committed {
                provisional_turn_id,
                turn_id,
                role: EarlyResponseCommitRole::PrimaryPlayback,
                ..
            } if provisional_turn_id == &first_provisional_id && turn_id == "turn-1"
        ));
        assert!(matches!(
            &events[1],
            EarlyResponseEvent::Canceled {
                provisional_turn_id,
                reason: EarlyResponseCancelReason::FinalTranscriptMismatch,
                ..
            } if provisional_turn_id == &second_provisional_id
        ));
        assert_eq!(sink.calls().len(), 1);
    }

    #[test]
    fn commit_boundary_does_not_promote_suffix_only_coalesced_member() {
        let sink = RecordingCancelSink::default();
        let mut aggregator = EarlyResponseAggregator::new(enabled_policy(), sink.clone());
        let mut suffix_member = partial("Oakland downtown now.", 9, 100);
        suffix_member.utterance_id = "utt-2".to_string();
        let suffix_provisional_id = provisional_turn_id_for(&suffix_member);
        assert_eq!(
            aggregator
                .handle_input(EarlyResponseInput::Partial(suffix_member))
                .len(),
            1
        );

        let events = aggregator.handle_input(EarlyResponseInput::CommitBoundary(
            EarlyResponseCommitBoundary {
                call_id: "call-1".to_string(),
                sequence: 10,
                turn_id: "turn-1".to_string(),
                coalesced_turn_ids: vec!["turn-frag-1".to_string(), "turn-frag-2".to_string()],
                final_text: "I need a tow truck in Oakland downtown now.".to_string(),
                members: vec![
                    EarlyResponseCommitMember {
                        utterance_id: "utt-1".to_string(),
                        member_index: 0,
                        member_final_text: "I need a tow truck in".to_string(),
                        transcript_event_ids: vec!["event-1".to_string()],
                    },
                    EarlyResponseCommitMember {
                        utterance_id: "utt-2".to_string(),
                        member_index: 1,
                        member_final_text: "Oakland downtown now.".to_string(),
                        transcript_event_ids: vec!["event-2".to_string()],
                    },
                ],
            },
        ));

        assert!(matches!(
            events.as_slice(),
            [EarlyResponseEvent::Canceled {
                provisional_turn_id,
                reason: EarlyResponseCancelReason::CoalescedIntoFinalTurn,
                ..
            }] if provisional_turn_id == &suffix_provisional_id
        ));
        assert_eq!(sink.calls().len(), 1);
    }

    #[tokio::test]
    async fn input_stream_prioritizes_control_over_full_partial_lane() {
        let (partial_tx, partial_rx) = early_response_partial_lane();
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        for index in 0..(EARLY_RESPONSE_INPUT_CAPACITY + 2) {
            let mut item = partial("I need a tow truck.", index as u64, 100 + index as u64);
            item.utterance_id = format!("utt-{index}");
            assert!(partial_tx.send(item));
        }
        control_tx
            .send(EarlyResponseInput::CancelCall {
                call_id: "call-1".to_string(),
                reason: EarlyResponseCancelReason::CallerBargeIn,
            })
            .expect("control lane open");

        let mut inputs = Box::pin(early_response_input_stream(partial_rx, control_rx));

        assert!(matches!(
            inputs.next().await,
            Some(EarlyResponseInput::CancelCall {
                reason: EarlyResponseCancelReason::CallerBargeIn,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn partial_lane_coalesces_same_utterance_to_latest_partial() {
        let (partial_tx, partial_rx) = early_response_partial_lane();
        let mut first = partial("I need a tow truck.", 7, 100);
        first.utterance_id = "utt-shared".to_string();
        let mut second = partial("I need a tow truck in Oakland.", 8, 180);
        second.utterance_id = "utt-shared".to_string();
        assert!(partial_tx.send(first));
        assert!(partial_tx.send(second));

        let received = partial_rx.recv().await.expect("partial available");

        assert_eq!(received.utterance_id, "utt-shared");
        assert_eq!(received.sequence, 8);
        assert_eq!(received.text, "I need a tow truck in Oakland.");
    }

    #[tokio::test]
    async fn partial_lane_drops_oldest_when_capacity_is_exceeded() {
        let (partial_tx, partial_rx) = early_response_partial_lane();
        for index in 0..EARLY_RESPONSE_INPUT_CAPACITY {
            let mut item = partial("I need a tow truck.", index as u64, 100 + index as u64);
            item.utterance_id = format!("utt-{index}");
            assert!(partial_tx.send(item));
        }
        let mut extra = partial("I need a tow truck now.", 99, 250);
        extra.utterance_id = "utt-extra".to_string();
        assert!(partial_tx.send(extra));

        let received = partial_rx.recv().await.expect("partial available");

        assert_eq!(received.utterance_id, "utt-1");
    }

    #[tokio::test]
    async fn stream_helper_preserves_aggregated_event_order() {
        let inputs = futures_util::stream::iter(vec![
            EarlyResponseInput::Partial(partial("I need a tow truck.", 7, 100)),
            EarlyResponseInput::Partial(partial("I need a tow truck in Oakland.", 8, 250)),
        ]);
        let events =
            aggregate_early_resp_partials(inputs, enabled_policy(), NoopEarlyResponseCancelSink)
                .collect::<Vec<_>>()
                .await;

        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            EarlyResponseEvent::Started { generation: 1, .. }
        ));
        assert!(matches!(
            events[1],
            EarlyResponseEvent::Updated { generation: 2, .. }
        ));
    }
}
