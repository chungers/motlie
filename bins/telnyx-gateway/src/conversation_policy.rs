use std::collections::VecDeque;
use std::time::Instant;

use crate::echo_match::match_assistant_echo_signature;

use crate::early_response::MissingSignalPolicy;
use crate::quality::{
    BargeInQualityConfig, ConversationPolicyConfig, ConversationPolicyMode,
    EchoSuppressionQualityConfig, PendingOutputOrder,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BargeInTrigger {
    Partial,
    Final,
    SpeechOnset,
}

impl BargeInTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Partial => "partial",
            Self::Final => "final",
            Self::SpeechOnset => "speech_onset",
        }
    }

    pub fn source_label(self) -> &'static str {
        match self {
            Self::Partial => "conversation partial barge-in",
            Self::Final => "conversation final barge-in",
            Self::SpeechOnset => "conversation speech-onset barge-in",
        }
    }

    pub fn cancel_span_name(self) -> &'static str {
        match self {
            Self::Partial => "barge_in.partial_to_cancel_request",
            Self::Final => "barge_in.final_to_cancel_request",
            Self::SpeechOnset => "barge_in.speech_onset_to_cancel_request",
        }
    }

    fn enabled_by(self, config: &BargeInQualityConfig) -> bool {
        config.enabled
            && match self {
                Self::Partial => config.partial_asr_cancel_enabled,
                Self::Final => config.final_asr_cancel_enabled,
                Self::SpeechOnset => config.speech_onset_cancel_enabled,
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaybackPolicyAction {
    Continue,
    CancelActive,
}

impl PlaybackPolicyAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::CancelActive => "cancel_active",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationPolicyAction {
    Continue,
    CancelActive,
}

impl GenerationPolicyAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::CancelActive => "cancel_active",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerTurnPolicyAction {
    Preserve,
    CoalesceAfterSilence { silence_ms: u64 },
}

impl CallerTurnPolicyAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Preserve => "preserve",
            Self::CoalesceAfterSilence { .. } => "coalesce_after_silence",
        }
    }

    pub fn silence_ms(self) -> Option<u64> {
        match self {
            Self::Preserve => None,
            Self::CoalesceAfterSilence { silence_ms } => Some(silence_ms),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerTranscriptPolicyAction {
    Forward,
    IgnoreDuringPlayback,
}

impl CallerTranscriptPolicyAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::IgnoreDuringPlayback => "ignore_during_playback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssistantOutputPolicyAction {
    QueueNow,
    LegacyDeferLatest,
    RetainBoundedPending,
}

impl AssistantOutputPolicyAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::QueueNow => "queue_now",
            Self::LegacyDeferLatest => "legacy_defer_latest",
            Self::RetainBoundedPending => "retain_bounded_pending",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BargeInPolicyDecision {
    pub allowed: bool,
    pub trigger: BargeInTrigger,
    pub mode: ConversationPolicyMode,
    pub playback: PlaybackPolicyAction,
    pub generation: GenerationPolicyAction,
    pub caller_turn: CallerTurnPolicyAction,
    pub caller_transcript: CallerTranscriptPolicyAction,
    pub reset_turn_batch: bool,
}

impl BargeInPolicyDecision {
    pub fn inactive(trigger: BargeInTrigger, mode: ConversationPolicyMode) -> Self {
        Self {
            allowed: false,
            trigger,
            mode,
            playback: PlaybackPolicyAction::Continue,
            generation: GenerationPolicyAction::Continue,
            caller_turn: CallerTurnPolicyAction::Preserve,
            caller_transcript: CallerTranscriptPolicyAction::Forward,
            reset_turn_batch: false,
        }
    }

    pub fn ignored_transcript(trigger: BargeInTrigger, mode: ConversationPolicyMode) -> Self {
        Self {
            caller_transcript: CallerTranscriptPolicyAction::IgnoreDuringPlayback,
            ..Self::inactive(trigger, mode)
        }
    }

    pub fn cancels_playback(self) -> bool {
        self.allowed && self.playback == PlaybackPolicyAction::CancelActive
    }

    pub fn forwards_caller_transcript(self) -> bool {
        self.caller_transcript == CallerTranscriptPolicyAction::Forward
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BargeInTranscriptEvidence<'a> {
    pub text: &'a str,
    pub active_playback: bool,
    pub confidence: Option<f32>,
    pub stability: Option<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinalTranscriptDispatchAction {
    Forward,
    SuppressPostPlaybackEcho,
}

impl FinalTranscriptDispatchAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::SuppressPostPlaybackEcho => "suppress_post_playback_echo",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinalTranscriptDispatchReason {
    Forward,
    EmptyTranscript,
    AssistantEcho,
    ShortFragment,
    WeakSignal,
}

impl FinalTranscriptDispatchReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::EmptyTranscript => "empty_transcript",
            Self::AssistantEcho => "assistant_echo",
            Self::ShortFragment => "short_fragment",
            Self::WeakSignal => "weak_signal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FinalTranscriptDispatchDecision {
    pub action: FinalTranscriptDispatchAction,
    pub reason: FinalTranscriptDispatchReason,
}

impl FinalTranscriptDispatchDecision {
    fn forward() -> Self {
        Self {
            action: FinalTranscriptDispatchAction::Forward,
            reason: FinalTranscriptDispatchReason::Forward,
        }
    }

    fn suppress(reason: FinalTranscriptDispatchReason) -> Self {
        Self {
            action: FinalTranscriptDispatchAction::SuppressPostPlaybackEcho,
            reason,
        }
    }

    pub fn forwards(self) -> bool {
        self.action == FinalTranscriptDispatchAction::Forward
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalTranscriptDispatchEvidence<'a> {
    pub text: &'a str,
    pub post_barge_in_guard_active: bool,
    pub active_or_recent_playback: bool,
    pub confidence: Option<f32>,
    pub stability: Option<f32>,
    pub assistant_echo_signature: Option<&'a str>,
}

impl BargeInTranscriptEvidence<'_> {
    pub fn char_count(self) -> usize {
        self.text.chars().filter(|ch| !ch.is_whitespace()).count()
    }

    pub fn word_count(self) -> usize {
        self.text.split_whitespace().count()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SayOverlapPolicyDecision {
    pub mode: ConversationPolicyMode,
    pub assistant_output: AssistantOutputPolicyAction,
}

impl ConversationPolicyConfig {
    pub fn decide_barge_in(
        &self,
        barge_in: &BargeInQualityConfig,
        trigger: BargeInTrigger,
    ) -> BargeInPolicyDecision {
        if !trigger.enabled_by(barge_in) {
            return BargeInPolicyDecision::inactive(trigger, self.mode);
        }

        self.allowed_barge_in_decision(trigger)
    }

    pub fn decide_transcript_barge_in(
        &self,
        barge_in: &BargeInQualityConfig,
        trigger: BargeInTrigger,
        evidence: BargeInTranscriptEvidence<'_>,
    ) -> BargeInPolicyDecision {
        if !trigger.enabled_by(barge_in) {
            return BargeInPolicyDecision::inactive(trigger, self.mode);
        }
        if evidence.active_playback
            && !transcript_evidence_allows_barge_in(self.mode, barge_in, trigger, evidence)
        {
            return BargeInPolicyDecision::ignored_transcript(trigger, self.mode);
        }

        self.allowed_barge_in_decision(trigger)
    }

    fn allowed_barge_in_decision(&self, trigger: BargeInTrigger) -> BargeInPolicyDecision {
        let caller_turn = match self.mode {
            ConversationPolicyMode::BargeInCoalesceAfterSilence => {
                CallerTurnPolicyAction::CoalesceAfterSilence {
                    silence_ms: self.post_barge_in_silence_ms,
                }
            }
            ConversationPolicyMode::CurrentCompat
            | ConversationPolicyMode::NoBargeInBoundedPending
            | ConversationPolicyMode::BargeInCancelOnly => CallerTurnPolicyAction::Preserve,
        };

        BargeInPolicyDecision {
            allowed: true,
            trigger,
            mode: self.mode,
            playback: PlaybackPolicyAction::CancelActive,
            generation: GenerationPolicyAction::CancelActive,
            caller_turn,
            caller_transcript: CallerTranscriptPolicyAction::Forward,
            reset_turn_batch: true,
        }
    }

    pub fn decide_say_overlap(
        &self,
        barge_in_enabled: bool,
        active_playback: bool,
    ) -> SayOverlapPolicyDecision {
        let assistant_output = if !active_playback || barge_in_enabled {
            AssistantOutputPolicyAction::QueueNow
        } else if self.mode == ConversationPolicyMode::NoBargeInBoundedPending {
            AssistantOutputPolicyAction::RetainBoundedPending
        } else {
            AssistantOutputPolicyAction::LegacyDeferLatest
        };
        SayOverlapPolicyDecision {
            mode: self.mode,
            assistant_output,
        }
    }

    pub fn decide_final_transcript_dispatch(
        &self,
        barge_in: &BargeInQualityConfig,
        echo_suppression: &EchoSuppressionQualityConfig,
        evidence: FinalTranscriptDispatchEvidence<'_>,
    ) -> FinalTranscriptDispatchDecision {
        if self.mode == ConversationPolicyMode::CurrentCompat
            || !barge_in.enabled
            || !evidence.post_barge_in_guard_active
            || !evidence.active_or_recent_playback
        {
            return FinalTranscriptDispatchDecision::forward();
        }
        if !has_normalized_transcript_content(evidence.text) {
            return FinalTranscriptDispatchDecision::suppress(
                FinalTranscriptDispatchReason::EmptyTranscript,
            );
        }
        if post_playback_echo_match(echo_suppression, evidence) {
            return FinalTranscriptDispatchDecision::suppress(
                FinalTranscriptDispatchReason::AssistantEcho,
            );
        }
        // ASR confidence can be high on post-playback echo fragments; length remains the
        // echo-biased guardrail during this narrow replacement-playback window.
        if is_post_playback_fragment(evidence.text, self) {
            return FinalTranscriptDispatchDecision::suppress(
                FinalTranscriptDispatchReason::ShortFragment,
            );
        }
        let barge_in_evidence = BargeInTranscriptEvidence {
            text: evidence.text,
            active_playback: true,
            confidence: evidence.confidence,
            stability: evidence.stability,
        };
        if !transcript_scores_allow_barge_in(barge_in, BargeInTrigger::Final, barge_in_evidence) {
            return FinalTranscriptDispatchDecision::suppress(
                FinalTranscriptDispatchReason::WeakSignal,
            );
        }
        FinalTranscriptDispatchDecision::forward()
    }
}

fn transcript_evidence_allows_barge_in(
    mode: ConversationPolicyMode,
    barge_in: &BargeInQualityConfig,
    trigger: BargeInTrigger,
    evidence: BargeInTranscriptEvidence<'_>,
) -> bool {
    let char_count = evidence.char_count();
    let word_count = evidence.word_count();
    if mode == ConversationPolicyMode::CurrentCompat {
        let legacy_length_allowed = match trigger {
            BargeInTrigger::Partial => char_count >= 3,
            BargeInTrigger::Final => word_count > 0,
            BargeInTrigger::SpeechOnset => true,
        };
        return legacy_length_allowed
            && legacy_transcript_scores_allow_barge_in(barge_in, trigger, evidence);
    }

    if !has_normalized_transcript_content(evidence.text) {
        return false;
    }
    transcript_scores_allow_barge_in(barge_in, trigger, evidence)
}

fn has_normalized_transcript_content(text: &str) -> bool {
    text.chars().any(|ch| ch.is_alphanumeric())
}

fn is_post_playback_fragment(text: &str, policy: &ConversationPolicyConfig) -> bool {
    let char_count = text.chars().filter(|ch| ch.is_alphanumeric()).count();
    let word_count = text
        .split_whitespace()
        .filter(|word| has_normalized_transcript_content(word))
        .count();
    char_count <= policy.post_barge_in_fragment_max_chars
        || word_count <= policy.post_barge_in_fragment_max_words
}

fn post_playback_echo_match(
    config: &EchoSuppressionQualityConfig,
    evidence: FinalTranscriptDispatchEvidence<'_>,
) -> bool {
    let Some(assistant_echo_signature) = evidence.assistant_echo_signature else {
        return false;
    };

    match_assistant_echo_signature(config, evidence.text, assistant_echo_signature).is_some()
}

fn legacy_transcript_scores_allow_barge_in(
    barge_in: &BargeInQualityConfig,
    trigger: BargeInTrigger,
    evidence: BargeInTranscriptEvidence<'_>,
) -> bool {
    match trigger {
        BargeInTrigger::Partial => {
            legacy_score_meets_threshold(evidence.confidence, barge_in.partial_min_confidence)
                && legacy_score_meets_threshold(evidence.stability, barge_in.partial_min_stability)
        }
        BargeInTrigger::Final => {
            legacy_score_meets_threshold(evidence.confidence, barge_in.final_min_confidence)
                && legacy_score_meets_threshold(evidence.stability, barge_in.final_min_stability)
        }
        BargeInTrigger::SpeechOnset => true,
    }
}

fn transcript_scores_allow_barge_in(
    barge_in: &BargeInQualityConfig,
    trigger: BargeInTrigger,
    evidence: BargeInTranscriptEvidence<'_>,
) -> bool {
    match trigger {
        BargeInTrigger::Partial => {
            required_score_meets_threshold(
                evidence.confidence,
                barge_in.partial_min_confidence,
                barge_in.missing_signal_policy,
            ) && required_score_meets_threshold(
                evidence.stability,
                barge_in.partial_min_stability,
                barge_in.missing_signal_policy,
            )
        }
        BargeInTrigger::Final => {
            required_score_meets_threshold(
                evidence.confidence,
                barge_in.final_min_confidence,
                barge_in.missing_signal_policy,
            ) && optional_score_meets_threshold(
                evidence.stability,
                barge_in.final_min_stability,
                barge_in.missing_signal_policy,
            )
        }
        BargeInTrigger::SpeechOnset => true,
    }
}

fn legacy_score_meets_threshold(score: Option<f32>, threshold: Option<f32>) -> bool {
    match threshold {
        Some(threshold) => score.is_some_and(|score| score >= threshold),
        None => true,
    }
}

fn required_score_meets_threshold(
    score: Option<f32>,
    threshold: Option<f32>,
    policy: MissingSignalPolicy,
) -> bool {
    match (score, threshold, policy) {
        (Some(score), Some(threshold), _) => {
            valid_score(score) && valid_score(threshold) && score >= threshold
        }
        (None, Some(_), MissingSignalPolicy::Conservative)
        | (_, None, MissingSignalPolicy::Conservative) => false,
    }
}

fn optional_score_meets_threshold(
    score: Option<f32>,
    threshold: Option<f32>,
    policy: MissingSignalPolicy,
) -> bool {
    match threshold {
        Some(threshold) => required_score_meets_threshold(score, Some(threshold), policy),
        None => true,
    }
}

fn valid_score(score: f32) -> bool {
    score.is_finite() && (0.0..=1.0).contains(&score)
}

#[derive(Clone, Debug)]
pub struct PendingPolicyOutput<T> {
    pub sequence: u64,
    pub enqueued_at: Instant,
    pub payload: T,
}

#[derive(Debug, Eq, PartialEq)]
pub struct PendingPolicyEnqueueOutcome {
    pub sequence: u64,
    pub pending_len: usize,
    pub dropped_count: usize,
    pub order: PendingOutputOrder,
}

#[derive(Debug)]
pub struct ConversationPolicyQueue<T> {
    next_sequence: u64,
    drain_running: bool,
    outputs: VecDeque<PendingPolicyOutput<T>>,
}

impl<T> Default for ConversationPolicyQueue<T> {
    fn default() -> Self {
        Self {
            next_sequence: 0,
            drain_running: false,
            outputs: VecDeque::new(),
        }
    }
}

impl<T> ConversationPolicyQueue<T> {
    pub fn enqueue(
        &mut self,
        config: &ConversationPolicyConfig,
        payload: T,
    ) -> PendingPolicyEnqueueOutcome {
        self.next_sequence = self.next_sequence.saturating_add(1);
        let sequence = self.next_sequence;
        let mut dropped_count = 0;
        let max_pending = config.max_pending_outputs.max(1);

        match config.pending_output_order {
            PendingOutputOrder::LatestOnly => {
                dropped_count = self.outputs.len();
                self.outputs.clear();
            }
            PendingOutputOrder::Fifo => {
                while self.outputs.len() >= max_pending {
                    self.outputs.pop_front();
                    dropped_count += 1;
                }
            }
        }

        self.outputs.push_back(PendingPolicyOutput {
            sequence,
            enqueued_at: Instant::now(),
            payload,
        });

        PendingPolicyEnqueueOutcome {
            sequence,
            pending_len: self.outputs.len(),
            dropped_count,
            order: config.pending_output_order,
        }
    }

    pub fn front(&self) -> Option<&PendingPolicyOutput<T>> {
        self.outputs.front()
    }

    pub fn take_next(&mut self) -> Option<PendingPolicyOutput<T>> {
        self.outputs.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    pub fn drain_running(&self) -> bool {
        self.drain_running
    }

    pub fn set_drain_running(&mut self, running: bool) {
        self.drain_running = running;
    }

    pub fn clear(&mut self) -> usize {
        let dropped = self.outputs.len();
        self.outputs.clear();
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::{ConversationPolicyMode, PendingOutputOrder};

    fn config(order: PendingOutputOrder, max_pending_outputs: usize) -> ConversationPolicyConfig {
        ConversationPolicyConfig {
            mode: ConversationPolicyMode::NoBargeInBoundedPending,
            active_playback_hold_ms: 1_000,
            max_pending_outputs,
            pending_output_order: order,
            post_barge_in_silence_ms: 1_200,
            post_barge_in_echo_guard_ms: 2_000,
            post_barge_in_fragment_max_chars: 12,
            post_barge_in_fragment_max_words: 2,
        }
    }

    #[test]
    fn latest_only_replaces_existing_pending_output() {
        let mut queue = ConversationPolicyQueue::default();
        let config = config(PendingOutputOrder::LatestOnly, 3);

        assert_eq!(queue.enqueue(&config, "first").dropped_count, 0);
        let outcome = queue.enqueue(&config, "second");

        assert_eq!(outcome.dropped_count, 1);
        assert_eq!(outcome.pending_len, 1);
        assert_eq!(queue.take_next().expect("pending output").payload, "second");
        assert!(queue.is_empty());
    }

    #[test]
    fn fifo_drops_oldest_when_bounded_queue_is_full() {
        let mut queue = ConversationPolicyQueue::default();
        let config = config(PendingOutputOrder::Fifo, 2);

        queue.enqueue(&config, "first");
        queue.enqueue(&config, "second");
        let outcome = queue.enqueue(&config, "third");

        assert_eq!(outcome.dropped_count, 1);
        assert_eq!(outcome.pending_len, 2);
        assert_eq!(queue.take_next().expect("second output").payload, "second");
        assert_eq!(queue.take_next().expect("third output").payload, "third");
        assert!(queue.is_empty());
    }

    #[test]
    fn current_compat_barge_in_preserves_current_cancel_semantics() {
        let policy = ConversationPolicyConfig::default();
        let decision =
            policy.decide_barge_in(&BargeInQualityConfig::default(), BargeInTrigger::Partial);

        assert!(decision.allowed);
        assert_eq!(decision.playback, PlaybackPolicyAction::CancelActive);
        assert_eq!(decision.generation, GenerationPolicyAction::CancelActive);
        assert_eq!(decision.caller_turn, CallerTurnPolicyAction::Preserve);
        assert!(decision.reset_turn_batch);
    }

    #[test]
    fn tiny_partial_during_active_playback_is_ignored() {
        let policy = ConversationPolicyConfig::default();
        let decision = policy.decide_transcript_barge_in(
            &BargeInQualityConfig::default(),
            BargeInTrigger::Partial,
            BargeInTranscriptEvidence {
                text: "uh",
                active_playback: true,
                confidence: Some(0.91),
                stability: Some(0.91),
            },
        );

        assert!(!decision.cancels_playback());
        assert!(!decision.forwards_caller_transcript());
    }

    #[test]
    fn current_compat_short_partial_during_active_playback_can_cancel() {
        let policy = ConversationPolicyConfig::default();
        let decision = policy.decide_transcript_barge_in(
            &BargeInQualityConfig::default(),
            BargeInTrigger::Partial,
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: Some(0.91),
                stability: Some(0.91),
            },
        );

        assert!(decision.cancels_playback());
        assert!(decision.forwards_caller_transcript());
    }

    #[test]
    fn current_compat_short_final_during_active_playback_can_cancel() {
        let policy = ConversationPolicyConfig::default();
        let decision = policy.decide_transcript_barge_in(
            &BargeInQualityConfig::default(),
            BargeInTrigger::Final,
            BargeInTranscriptEvidence {
                text: "no",
                active_playback: true,
                confidence: Some(0.91),
                stability: None,
            },
        );

        assert!(decision.cancels_playback());
        assert!(decision.forwards_caller_transcript());
    }

    fn post_barge_in_policy() -> ConversationPolicyConfig {
        ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCoalesceAfterSilence,
            post_barge_in_echo_guard_ms: 2_000,
            ..ConversationPolicyConfig::default()
        }
    }

    fn scored_final_barge_in() -> BargeInQualityConfig {
        BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        }
    }

    fn guarded_final_evidence<'a>(
        text: &'a str,
        assistant_echo_signature: Option<&'a str>,
    ) -> FinalTranscriptDispatchEvidence<'a> {
        FinalTranscriptDispatchEvidence {
            text,
            post_barge_in_guard_active: true,
            active_or_recent_playback: true,
            confidence: Some(0.91),
            stability: None,
            assistant_echo_signature,
        }
    }

    #[test]
    fn current_compat_dispatch_guard_forwards_short_finals() {
        let policy = ConversationPolicyConfig::default();
        let decision = policy.decide_final_transcript_dispatch(
            &scored_final_barge_in(),
            &EchoSuppressionQualityConfig::default(),
            guarded_final_evidence("up now", None),
        );

        assert!(decision.forwards());
    }

    #[test]
    fn post_barge_in_dispatch_guard_suppresses_short_final_fragments() {
        let policy = post_barge_in_policy();
        let decision = policy.decide_final_transcript_dispatch(
            &scored_final_barge_in(),
            &EchoSuppressionQualityConfig::default(),
            guarded_final_evidence("up now", None),
        );

        assert_eq!(
            decision.action,
            FinalTranscriptDispatchAction::SuppressPostPlaybackEcho
        );
        assert_eq!(
            decision.reason,
            FinalTranscriptDispatchReason::ShortFragment
        );
    }

    #[test]
    fn cancel_only_dispatch_guard_suppresses_short_final_fragments() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            post_barge_in_echo_guard_ms: 2_000,
            ..ConversationPolicyConfig::default()
        };
        let decision = policy.decide_final_transcript_dispatch(
            &scored_final_barge_in(),
            &EchoSuppressionQualityConfig::default(),
            guarded_final_evidence("up now", None),
        );

        assert_eq!(
            decision.reason,
            FinalTranscriptDispatchReason::ShortFragment
        );
    }

    #[test]
    fn post_barge_in_dispatch_guard_suppresses_score_absent_finals() {
        let policy = post_barge_in_policy();
        let decision = policy.decide_final_transcript_dispatch(
            &scored_final_barge_in(),
            &EchoSuppressionQualityConfig::default(),
            FinalTranscriptDispatchEvidence {
                text: "this sentence has enough non echo words",
                post_barge_in_guard_active: true,
                active_or_recent_playback: true,
                confidence: None,
                stability: None,
                assistant_echo_signature: None,
            },
        );

        assert_eq!(decision.reason, FinalTranscriptDispatchReason::WeakSignal);
    }

    #[test]
    fn post_barge_in_dispatch_guard_suppresses_garbled_assistant_echo() {
        let policy = post_barge_in_policy();
        let decision = policy.decide_final_transcript_dispatch(
            &scored_final_barge_in(),
            &EchoSuppressionQualityConfig::default(),
            guarded_final_evidence(
                "He's been paint this replacement sentence clearly after the interrup",
                Some("please repeat this replacement sentence clearly after the interruption"),
            ),
        );

        assert_eq!(
            decision.reason,
            FinalTranscriptDispatchReason::AssistantEcho
        );
    }

    #[test]
    fn post_barge_in_dispatch_guard_forwards_strong_non_echo_replacement() {
        let policy = post_barge_in_policy();
        let decision = policy.decide_final_transcript_dispatch(
            &scored_final_barge_in(),
            &EchoSuppressionQualityConfig::default(),
            guarded_final_evidence(
                "please repeat this replacement sentence clearly after the interruption",
                Some("the gateway will begin repeating this long sentence"),
            ),
        );

        assert!(decision.forwards());
    }

    #[test]
    fn non_compat_score_missing_partial_fails_closed() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let barge_in = BargeInQualityConfig {
            partial_min_confidence: Some(0.50),
            partial_min_stability: Some(0.50),
            ..BargeInQualityConfig::default()
        };

        for evidence in [
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: None,
                stability: Some(0.91),
            },
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: Some(0.91),
                stability: None,
            },
        ] {
            let decision =
                policy.decide_transcript_barge_in(&barge_in, BargeInTrigger::Partial, evidence);
            assert!(!decision.cancels_playback());
            assert!(!decision.forwards_caller_transcript());
        }
    }

    #[test]
    fn non_compat_score_missing_final_fails_closed() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let barge_in = BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };
        let decision = policy.decide_transcript_barge_in(
            &barge_in,
            BargeInTrigger::Final,
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: None,
                stability: None,
            },
        );

        assert!(!decision.cancels_playback());
        assert!(!decision.forwards_caller_transcript());
    }

    #[test]
    fn non_compat_unconfigured_scores_fail_closed() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };

        for (trigger, text) in [
            (BargeInTrigger::Partial, "stop"),
            (BargeInTrigger::Final, "stop"),
            (BargeInTrigger::Partial, "uh"),
            (BargeInTrigger::Final, "uh"),
        ] {
            let decision = policy.decide_transcript_barge_in(
                &BargeInQualityConfig::default(),
                trigger,
                BargeInTranscriptEvidence {
                    text,
                    active_playback: true,
                    confidence: Some(0.91),
                    stability: Some(0.91),
                },
            );
            assert!(!decision.cancels_playback());
            assert!(!decision.forwards_caller_transcript());
        }
    }

    #[test]
    fn non_compat_one_word_interrupts_cancel_with_required_scores() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let partial_barge_in = BargeInQualityConfig {
            partial_min_confidence: Some(0.50),
            partial_min_stability: Some(0.50),
            ..BargeInQualityConfig::default()
        };
        let partial = policy.decide_transcript_barge_in(
            &partial_barge_in,
            BargeInTrigger::Partial,
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: Some(0.91),
                stability: Some(0.91),
            },
        );
        assert!(partial.cancels_playback());
        assert!(partial.forwards_caller_transcript());

        let final_barge_in = BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };
        let final_decision = policy.decide_transcript_barge_in(
            &final_barge_in,
            BargeInTrigger::Final,
            BargeInTranscriptEvidence {
                text: "no",
                active_playback: true,
                confidence: Some(0.91),
                stability: None,
            },
        );
        assert!(final_decision.cancels_playback());
        assert!(final_decision.forwards_caller_transcript());
    }

    #[test]
    fn non_compat_rejects_empty_whitespace_and_punctuation_only_text() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let barge_in = BargeInQualityConfig {
            partial_min_confidence: Some(0.50),
            partial_min_stability: Some(0.50),
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };

        for (trigger, text) in [
            (BargeInTrigger::Partial, ""),
            (BargeInTrigger::Partial, "   "),
            (BargeInTrigger::Partial, "...?!"),
            (BargeInTrigger::Final, "...?!"),
        ] {
            let decision = policy.decide_transcript_barge_in(
                &barge_in,
                trigger,
                BargeInTranscriptEvidence {
                    text,
                    active_playback: true,
                    confidence: Some(0.91),
                    stability: Some(0.91),
                },
            );
            assert!(!decision.cancels_playback());
            assert!(!decision.forwards_caller_transcript());
        }
    }

    #[test]
    fn non_compat_revision_churn_waits_for_score_stable_partial() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let barge_in = BargeInQualityConfig {
            partial_min_confidence: Some(0.50),
            partial_min_stability: Some(0.80),
            ..BargeInQualityConfig::default()
        };

        for evidence in [
            BargeInTranscriptEvidence {
                text: "s",
                active_playback: true,
                confidence: Some(0.91),
                stability: None,
            },
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: Some(0.91),
                stability: Some(0.40),
            },
        ] {
            let decision =
                policy.decide_transcript_barge_in(&barge_in, BargeInTrigger::Partial, evidence);
            assert!(!decision.cancels_playback());
            assert!(!decision.forwards_caller_transcript());
        }

        let stable = policy.decide_transcript_barge_in(
            &barge_in,
            BargeInTrigger::Partial,
            BargeInTranscriptEvidence {
                text: "stop",
                active_playback: true,
                confidence: Some(0.91),
                stability: Some(0.91),
            },
        );
        assert!(stable.cancels_playback());
        assert!(stable.forwards_caller_transcript());
    }

    #[test]
    fn meaningful_partial_during_active_playback_can_cancel() {
        let policy = ConversationPolicyConfig::default();
        let decision = policy.decide_transcript_barge_in(
            &BargeInQualityConfig::default(),
            BargeInTrigger::Partial,
            BargeInTranscriptEvidence {
                text: "please stop",
                active_playback: true,
                confidence: Some(0.91),
                stability: None,
            },
        );

        assert!(decision.cancels_playback());
        assert!(decision.forwards_caller_transcript());
    }

    #[test]
    fn low_confidence_partial_during_active_playback_is_ignored_when_configured() {
        let policy = ConversationPolicyConfig::default();
        let barge_in = BargeInQualityConfig {
            partial_min_confidence: Some(0.50),
            ..BargeInQualityConfig::default()
        };
        let decision = policy.decide_transcript_barge_in(
            &barge_in,
            BargeInTrigger::Partial,
            BargeInTranscriptEvidence {
                text: "please stop",
                active_playback: true,
                confidence: Some(0.34),
                stability: Some(0.91),
            },
        );

        assert!(!decision.cancels_playback());
        assert!(!decision.forwards_caller_transcript());
    }

    #[test]
    fn low_confidence_final_during_active_playback_is_ignored_when_configured() {
        let policy = ConversationPolicyConfig::default();
        let barge_in = BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };
        let decision = policy.decide_transcript_barge_in(
            &barge_in,
            BargeInTrigger::Final,
            BargeInTranscriptEvidence {
                text: "at now please",
                active_playback: true,
                confidence: Some(0.62),
                stability: None,
            },
        );

        assert!(!decision.cancels_playback());
        assert!(!decision.forwards_caller_transcript());
    }

    #[test]
    fn high_confidence_final_during_active_playback_can_cancel_when_configured() {
        let policy = ConversationPolicyConfig::default();
        let barge_in = BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };
        let decision = policy.decide_transcript_barge_in(
            &barge_in,
            BargeInTrigger::Final,
            BargeInTranscriptEvidence {
                text: "stop now please",
                active_playback: true,
                confidence: Some(0.91),
                stability: None,
            },
        );

        assert!(decision.cancels_playback());
        assert!(decision.forwards_caller_transcript());
    }

    #[test]
    fn coalesce_policy_preserves_caller_audio_after_cancel() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCoalesceAfterSilence,
            active_playback_hold_ms: 1_000,
            max_pending_outputs: 1,
            pending_output_order: PendingOutputOrder::LatestOnly,
            post_barge_in_silence_ms: 1_500,
            post_barge_in_echo_guard_ms: 2_000,
            post_barge_in_fragment_max_chars: 12,
            post_barge_in_fragment_max_words: 2,
        };
        let decision =
            policy.decide_barge_in(&BargeInQualityConfig::default(), BargeInTrigger::Final);

        assert!(decision.cancels_playback());
        assert_eq!(
            decision.caller_turn,
            CallerTurnPolicyAction::CoalesceAfterSilence { silence_ms: 1_500 }
        );
    }

    #[test]
    fn say_overlap_decision_keeps_default_compat_behavior() {
        let policy = ConversationPolicyConfig::default();
        let decision = policy.decide_say_overlap(false, true);

        assert_eq!(
            decision.assistant_output,
            AssistantOutputPolicyAction::LegacyDeferLatest
        );
    }

    #[test]
    fn say_overlap_decision_can_retain_bounded_pending() {
        let policy = config(PendingOutputOrder::Fifo, 3);
        let decision = policy.decide_say_overlap(false, true);

        assert_eq!(
            decision.assistant_output,
            AssistantOutputPolicyAction::RetainBoundedPending
        );
    }
}
