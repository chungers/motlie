use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};

use futures_util::{stream, Stream, StreamExt};

use crate::text_calls::turns::CallerSpeechState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AppendOrReplace {
    Append,
    Replace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EarlyResponseAudioMode {
    SpeakProvisionally,
    PrepareOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundaryRequirement {
    None,
    Clause,
    Sentence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingSignalPolicy {
    Conservative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EarlyResponseStartTiming {
    EndpointCandidateOnly,
    WhileSpeaking,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EarlyResponseAppendMode {
    ReplaceOnly,
    PrefixMonotonicBackend,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EarlyResponsePolicy {
    pub enabled: bool,
    pub audio_mode: EarlyResponseAudioMode,
    pub min_text_chars: usize,
    pub min_text_tokens: usize,
    pub boundary: BoundaryRequirement,
    pub min_confidence: Option<ScoreThreshold>,
    pub min_stability: Option<ScoreThreshold>,
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
        Self {
            enabled: false,
            audio_mode: EarlyResponseAudioMode::SpeakProvisionally,
            min_text_chars: 12,
            min_text_tokens: 3,
            boundary: BoundaryRequirement::Clause,
            min_confidence: ScoreThreshold::new(0.70),
            min_stability: ScoreThreshold::new(0.80),
            missing_signal_policy: MissingSignalPolicy::Conservative,
            allowed_start_speech_states: vec![CallerSpeechState::EndpointCandidate],
            allowed_update_speech_states: vec![
                CallerSpeechState::EndpointCandidate,
                CallerSpeechState::Finalizing,
            ],
            debounce_ms: 120,
            max_updates_per_utterance: 3,
            start_timing: EarlyResponseStartTiming::EndpointCandidateOnly,
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreThreshold(f32);

impl ScoreThreshold {
    pub fn new(value: f32) -> Option<Self> {
        (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Eq for ScoreThreshold {}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EarlyResponseCommitRole {
    PrimaryPlayback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

pub trait EarlyResponseProcessor {
    fn process_event(&self, event: EarlyResponseEvent) -> Option<EarlyResponseIntent>;
}

pub fn identity_passthrough(event: EarlyResponseEvent) -> Option<EarlyResponseIntent> {
    match event {
        EarlyResponseEvent::Started {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            text,
            ..
        } => Some(EarlyResponseIntent::Speak {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            text,
            append_or_replace: AppendOrReplace::Replace,
        }),
        EarlyResponseEvent::Updated {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            text,
            append_or_replace,
        } => Some(EarlyResponseIntent::Speak {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            text,
            append_or_replace,
        }),
        EarlyResponseEvent::Canceled {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            reason,
        } => Some(EarlyResponseIntent::Cancel {
            provisional_turn_id,
            call_id,
            utterance_id,
            generation,
            reason,
        }),
        EarlyResponseEvent::Committed {
            provisional_turn_id,
            call_id,
            generation,
            turn_id,
            ..
        } => Some(EarlyResponseIntent::Commit {
            provisional_turn_id,
            call_id,
            generation,
            turn_id,
        }),
    }
}

pub fn aggregate_early_resp_partials<S, C>(
    partial_stream: S,
    policy: EarlyResponsePolicy,
    cancel_sink: C,
) -> impl Stream<Item = EarlyResponseEvent>
where
    S: Stream<Item = EarlyResponseInput> + Unpin,
    C: EarlyResponsePriorityCancelSink,
{
    stream::unfold(
        (
            partial_stream,
            EarlyResponseAggregator::new(policy, cancel_sink),
            VecDeque::new(),
        ),
        |(mut partial_stream, mut aggregator, mut pending)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (partial_stream, aggregator, pending)));
                }
                let input = partial_stream.next().await?;
                pending.extend(aggregator.handle_input(input));
            }
        },
    )
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
        if active.is_none()
            && !self
                .policy
                .allowed_start_speech_states
                .contains(&partial.speech_state)
        {
            return Vec::new();
        }
        if active.is_some()
            && !self
                .policy
                .allowed_update_speech_states
                .contains(&partial.speech_state)
        {
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
                    return self.cancel_active(
                        key,
                        active,
                        EarlyResponseCancelReason::MaxUpdatesExceeded,
                    );
                }
                self.cancel_sink.cancel_provisional(
                    ProvisionalPlaybackKey {
                        call_id: active.call_id.clone(),
                        provisional_turn_id: active.provisional_turn_id.clone(),
                        generation: active.generation,
                        playback_id: None,
                    },
                    EarlyResponseCancelReason::SupersededByNewGeneration,
                );
                active.generation += 1;
                active.updates += 1;
                active.last_emitted_at_ms = partial.received_at_ms;
                let append_or_replace =
                    append_or_replace_for(self.policy.append_mode, &active.text, &accepted_text);
                active.text = accepted_text.clone();
                self.active.insert(key, active.clone());
                vec![EarlyResponseEvent::Updated {
                    provisional_turn_id: active.provisional_turn_id,
                    call_id: active.call_id,
                    utterance_id: active.utterance_id,
                    generation: active.generation,
                    text: accepted_text,
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
            return Vec::new();
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

    fn policy_accepts_score(&self, score: Option<f32>, threshold: Option<ScoreThreshold>) -> bool {
        match (score, threshold, self.policy.missing_signal_policy) {
            (_, None, _) => true,
            (Some(score), Some(threshold), _) => {
                score.is_finite() && (0.0..=1.0).contains(&score) && score >= threshold.get()
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
    format!(
        "pt_{}_{}_{}",
        token_component(&partial.call_id),
        token_component(&partial.utterance_id),
        partial.sequence
    )
}

fn token_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
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
            if accepted_text.starts_with(previous_text) {
                AppendOrReplace::Append
            } else {
                AppendOrReplace::Replace
            }
        }
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
        let events = aggregator.handle_input(EarlyResponseInput::Partial(partial(
            "I need a tow truck.",
            7,
            100,
        )));

        assert_eq!(
            events,
            vec![EarlyResponseEvent::Started {
                provisional_turn_id: "pt_call-1_utt-1_7".to_string(),
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
            "I need a tow truck in Oakland.",
            8,
            250,
        )));

        assert_eq!(
            events,
            vec![EarlyResponseEvent::Updated {
                provisional_turn_id: "pt_call-1_utt-1_7".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                text: "I need a tow truck in Oakland.".to_string(),
                append_or_replace: AppendOrReplace::Replace,
            }]
        );
        assert_eq!(
            sink.calls(),
            vec![(
                ProvisionalPlaybackKey {
                    call_id: "call-1".to_string(),
                    provisional_turn_id: "pt_call-1_utt-1_7".to_string(),
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
    fn commit_boundary_commits_primary_and_cancels_mismatches() {
        let sink = RecordingCancelSink::default();
        let mut aggregator = EarlyResponseAggregator::new(enabled_policy(), sink.clone());
        aggregator.handle_input(EarlyResponseInput::Partial(partial(
            "I need a tow truck.",
            7,
            100,
        )));
        let mut second = partial("Wrong city downtown.", 9, 100);
        second.utterance_id = "utt-2".to_string();
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
            } if provisional_turn_id == "pt_call-1_utt-1_7" && turn_id == "turn-1"
        ));
        assert!(matches!(
            &events[1],
            EarlyResponseEvent::Canceled {
                provisional_turn_id,
                reason: EarlyResponseCancelReason::FinalTranscriptMismatch,
                ..
            } if provisional_turn_id == "pt_call-1_utt-2_9"
        ));
        assert_eq!(sink.calls().len(), 1);
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

    #[test]
    fn identity_passthrough_repeats_accepted_transcript_fragment_without_prefix() {
        let event = EarlyResponseEvent::Started {
            provisional_turn_id: "pt-1".to_string(),
            call_id: "call-1".to_string(),
            utterance_id: "utt-1".to_string(),
            generation: 1,
            text: "I need a tow truck.".to_string(),
            confidence: Some(0.9),
            stability: Some(0.8),
            speech_state: CallerSpeechState::EndpointCandidate,
        };

        assert_eq!(
            identity_passthrough(event),
            Some(EarlyResponseIntent::Speak {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 1,
                text: "I need a tow truck.".to_string(),
                append_or_replace: AppendOrReplace::Replace,
            })
        );
    }
}
