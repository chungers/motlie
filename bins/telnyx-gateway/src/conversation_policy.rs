use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::echo_match::match_assistant_echo_signature;

use crate::early_response::MissingSignalPolicy;
use crate::quality::{
    AudioBargeInMode, AudioBargeInQualityConfig, AudioBargeInUncertainPolicy, BargeInQualityConfig,
    ConversationPolicyConfig, ConversationPolicyMode, EchoSuppressionQualityConfig,
    PendingOutputOrder,
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
pub enum AudioOnsetDecision {
    TrustedCallerOnset,
    LikelyAssistantEcho,
    Ambiguous,
    Unavailable,
}

impl AudioOnsetDecision {
    pub fn label(self) -> &'static str {
        match self {
            Self::TrustedCallerOnset => "trusted_caller_onset",
            Self::LikelyAssistantEcho => "likely_assistant_echo",
            Self::Ambiguous => "ambiguous",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaybackEchoState {
    Idle,
    ActivePlayback,
    RecentTail,
    InterSegmentGap,
}

impl PlaybackEchoState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::ActivePlayback => "active_playback",
            Self::RecentTail => "recent_tail",
            Self::InterSegmentGap => "inter_segment_gap",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioEvidenceInvalidation {
    TransportInvalid,
    ShortReference,
    StalePlaybackEpoch,
    CalibrationUnavailable,
    DelayOutOfRange,
    LowCorrelationNonSpeech,
}

impl AudioEvidenceInvalidation {
    pub fn label(self) -> &'static str {
        match self {
            Self::TransportInvalid => "transport_invalid",
            Self::ShortReference => "short_reference",
            Self::StalePlaybackEpoch => "stale_playback_epoch",
            Self::CalibrationUnavailable => "calibration_unavailable",
            Self::DelayOutOfRange => "delay_out_of_range",
            Self::LowCorrelationNonSpeech => "low_correlation_non_speech",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallerOnsetEvidence {
    pub decision: AudioOnsetDecision,
    pub confidence: f32,
    pub playback_state: PlaybackEchoState,
    pub playback_id: Option<String>,
    pub playback_epoch: Option<u64>,
    pub caller_active_since: Option<Instant>,
    pub evidence_at: Instant,
    pub evidence_age_ms: u64,
    pub window_ms: u32,
    pub input_codec: String,
    pub input_sample_rate_hz: u32,
    pub evidence_sample_rate_hz: u32,
    pub inbound_rms_dbfs: f32,
    pub outbound_rms_dbfs: Option<f32>,
    pub estimated_delay_ms: Option<u32>,
    pub correlation_peak: Option<f32>,
    pub echo_return_db: Option<f32>,
    pub echo_margin_db: Option<f32>,
    pub invalidation: Option<AudioEvidenceInvalidation>,
}

impl CallerOnsetEvidence {
    pub fn fresh_at(&self, now: Instant, max_age_ms: u64) -> bool {
        now.saturating_duration_since(self.evidence_at) <= Duration::from_millis(max_age_ms)
    }

    pub fn matches_active_playback(
        &self,
        active_playback_id: Option<&str>,
        active_playback_epoch: Option<u64>,
    ) -> bool {
        self.playback_id.as_deref() == active_playback_id
            && self.playback_epoch == active_playback_epoch
            && active_playback_id.is_some()
            && active_playback_epoch.is_some()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ActivePlaybackTarget<'a> {
    pub playback_id: Option<&'a str>,
    pub playback_epoch: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct AudioBargeInDecisionInput<'a> {
    pub barge_in: &'a BargeInQualityConfig,
    pub audio_barge_in: &'a AudioBargeInQualityConfig,
    pub trigger: BargeInTrigger,
    pub evidence: &'a CallerOnsetEvidence,
    pub active_playback: ActivePlaybackTarget<'a>,
    pub now: Instant,
}

#[derive(Clone, Copy, Debug)]
pub struct TranscriptBargeInDecisionInput<'a> {
    pub barge_in: &'a BargeInQualityConfig,
    pub audio_barge_in: &'a AudioBargeInQualityConfig,
    pub trigger: BargeInTrigger,
    pub transcript: BargeInTranscriptEvidence<'a>,
    pub audio: Option<&'a CallerOnsetEvidence>,
    pub active_playback: ActivePlaybackTarget<'a>,
    pub now: Instant,
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

    pub fn decide_audio_barge_in(
        &self,
        input: AudioBargeInDecisionInput<'_>,
    ) -> BargeInPolicyDecision {
        if !input.trigger.enabled_by(input.barge_in) {
            return BargeInPolicyDecision::inactive(input.trigger, self.mode);
        }
        if !self.consumes_layer_b_audio(input.audio_barge_in) {
            return self.decide_barge_in(input.barge_in, input.trigger);
        }
        if !input
            .evidence
            .fresh_at(input.now, input.audio_barge_in.media.max_evidence_age_ms)
            || !input.evidence.matches_active_playback(
                input.active_playback.playback_id,
                input.active_playback.playback_epoch,
            )
        {
            return BargeInPolicyDecision::inactive(input.trigger, self.mode);
        }

        match input.evidence.decision {
            AudioOnsetDecision::TrustedCallerOnset => self.allowed_barge_in_decision(input.trigger),
            AudioOnsetDecision::LikelyAssistantEcho
            | AudioOnsetDecision::Ambiguous
            | AudioOnsetDecision::Unavailable => {
                BargeInPolicyDecision::inactive(input.trigger, self.mode)
            }
        }
    }

    pub fn decide_transcript_barge_in_with_audio(
        &self,
        input: TranscriptBargeInDecisionInput<'_>,
    ) -> BargeInPolicyDecision {
        let layer_a =
            self.decide_transcript_barge_in(input.barge_in, input.trigger, input.transcript);
        if !input.transcript.active_playback || !self.consumes_layer_b_audio(input.audio_barge_in) {
            return layer_a;
        }
        let Some(audio) = input.audio else {
            return layer_a;
        };
        if !audio.fresh_at(input.now, input.audio_barge_in.media.max_evidence_age_ms)
            || !audio.matches_active_playback(
                input.active_playback.playback_id,
                input.active_playback.playback_epoch,
            )
        {
            return layer_a;
        }

        match audio.decision {
            AudioOnsetDecision::TrustedCallerOnset => self.allowed_barge_in_decision(input.trigger),
            AudioOnsetDecision::LikelyAssistantEcho => {
                if layer_a.cancels_playback() {
                    BargeInPolicyDecision::ignored_transcript(input.trigger, self.mode)
                } else {
                    layer_a
                }
            }
            AudioOnsetDecision::Ambiguous => match input.audio_barge_in.policy.uncertain_policy {
                AudioBargeInUncertainPolicy::DeferToLayerA => layer_a,
                AudioBargeInUncertainPolicy::ContinuePlayback => {
                    if layer_a.cancels_playback() {
                        BargeInPolicyDecision::ignored_transcript(input.trigger, self.mode)
                    } else {
                        layer_a
                    }
                }
            },
            AudioOnsetDecision::Unavailable => layer_a,
        }
    }

    fn consumes_layer_b_audio(&self, audio_barge_in: &AudioBargeInQualityConfig) -> bool {
        audio_barge_in.media.mode != AudioBargeInMode::MeasureOnly
            && matches!(
                self.mode,
                ConversationPolicyMode::BargeInCancelOnly
                    | ConversationPolicyMode::BargeInCoalesceAfterSilence
            )
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

    fn audio_barge_in_config(
        mode: AudioBargeInMode,
        uncertain_policy: AudioBargeInUncertainPolicy,
    ) -> AudioBargeInQualityConfig {
        AudioBargeInQualityConfig {
            media: crate::quality::AudioBargeInMediaQualityConfig {
                mode,
                max_evidence_age_ms: 120,
                ..crate::quality::AudioBargeInMediaQualityConfig::default()
            },
            policy: crate::quality::config::AudioBargeInPolicyQualityConfig { uncertain_policy },
        }
    }

    fn audio_evidence(
        decision: AudioOnsetDecision,
        playback_id: &str,
        playback_epoch: u64,
        evidence_age_ms: u64,
    ) -> CallerOnsetEvidence {
        let now = Instant::now();
        CallerOnsetEvidence {
            decision,
            confidence: 0.91,
            playback_state: PlaybackEchoState::ActivePlayback,
            playback_id: Some(playback_id.to_string()),
            playback_epoch: Some(playback_epoch),
            caller_active_since: Some(now),
            evidence_at: now - Duration::from_millis(evidence_age_ms),
            evidence_age_ms,
            window_ms: 20,
            input_codec: "PCMU".to_string(),
            input_sample_rate_hz: 8_000,
            evidence_sample_rate_hz: 8_000,
            inbound_rms_dbfs: -20.0,
            outbound_rms_dbfs: Some(-35.0),
            estimated_delay_ms: Some(40),
            correlation_peak: Some(0.30),
            echo_return_db: Some(-20.0),
            echo_margin_db: Some(10.0),
            invalidation: None,
        }
    }

    #[test]
    fn trusted_audio_onset_cancels_without_waiting_for_transcript() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let audio = audio_barge_in_config(
            AudioBargeInMode::EchoAwareOnset,
            AudioBargeInUncertainPolicy::DeferToLayerA,
        );
        let evidence = audio_evidence(AudioOnsetDecision::TrustedCallerOnset, "playback-1", 7, 20);

        let barge_in = BargeInQualityConfig::default();
        let decision = policy.decide_audio_barge_in(AudioBargeInDecisionInput {
            barge_in: &barge_in,
            audio_barge_in: &audio,
            trigger: BargeInTrigger::SpeechOnset,
            evidence: &evidence,
            active_playback: ActivePlaybackTarget {
                playback_id: Some("playback-1"),
                playback_epoch: Some(7),
            },
            now: Instant::now(),
        });

        assert!(decision.cancels_playback());
    }

    #[test]
    fn likely_assistant_echo_vetoes_layer_a_transcript_cancel_for_fresh_epoch() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let audio = audio_barge_in_config(
            AudioBargeInMode::EchoAwareOnset,
            AudioBargeInUncertainPolicy::DeferToLayerA,
        );
        let barge_in = BargeInQualityConfig {
            partial_min_confidence: Some(0.50),
            partial_min_stability: Some(0.50),
            ..BargeInQualityConfig::default()
        };
        let evidence = audio_evidence(AudioOnsetDecision::LikelyAssistantEcho, "playback-1", 7, 20);

        let decision =
            policy.decide_transcript_barge_in_with_audio(TranscriptBargeInDecisionInput {
                barge_in: &barge_in,
                audio_barge_in: &audio,
                trigger: BargeInTrigger::Partial,
                transcript: BargeInTranscriptEvidence {
                    text: "stop now",
                    active_playback: true,
                    confidence: Some(0.91),
                    stability: Some(0.91),
                },
                audio: Some(&evidence),
                active_playback: ActivePlaybackTarget {
                    playback_id: Some("playback-1"),
                    playback_epoch: Some(7),
                },
                now: Instant::now(),
            });

        assert!(!decision.cancels_playback());
        assert!(!decision.forwards_caller_transcript());
    }

    #[test]
    fn stale_audio_epoch_is_ignored_and_layer_a_may_act_independently() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let audio = audio_barge_in_config(
            AudioBargeInMode::EchoAwareOnset,
            AudioBargeInUncertainPolicy::DeferToLayerA,
        );
        let barge_in = BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };
        let evidence = audio_evidence(
            AudioOnsetDecision::LikelyAssistantEcho,
            "old-playback",
            1,
            20,
        );

        let decision =
            policy.decide_transcript_barge_in_with_audio(TranscriptBargeInDecisionInput {
                barge_in: &barge_in,
                audio_barge_in: &audio,
                trigger: BargeInTrigger::Final,
                transcript: BargeInTranscriptEvidence {
                    text: "stop now",
                    active_playback: true,
                    confidence: Some(0.91),
                    stability: None,
                },
                audio: Some(&evidence),
                active_playback: ActivePlaybackTarget {
                    playback_id: Some("new-playback"),
                    playback_epoch: Some(2),
                },
                now: Instant::now(),
            });

        assert!(decision.cancels_playback());
    }

    #[test]
    fn ambiguous_audio_can_continue_playback_when_policy_is_strict() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCancelOnly,
            ..ConversationPolicyConfig::default()
        };
        let audio = audio_barge_in_config(
            AudioBargeInMode::EchoAwareOnset,
            AudioBargeInUncertainPolicy::ContinuePlayback,
        );
        let barge_in = BargeInQualityConfig {
            final_min_confidence: Some(0.70),
            ..BargeInQualityConfig::default()
        };
        let evidence = audio_evidence(AudioOnsetDecision::Ambiguous, "playback-1", 7, 20);

        let decision =
            policy.decide_transcript_barge_in_with_audio(TranscriptBargeInDecisionInput {
                barge_in: &barge_in,
                audio_barge_in: &audio,
                trigger: BargeInTrigger::Final,
                transcript: BargeInTranscriptEvidence {
                    text: "stop now",
                    active_playback: true,
                    confidence: Some(0.91),
                    stability: None,
                },
                audio: Some(&evidence),
                active_playback: ActivePlaybackTarget {
                    playback_id: Some("playback-1"),
                    playback_epoch: Some(7),
                },
                now: Instant::now(),
            });

        assert!(!decision.cancels_playback());
        assert!(!decision.forwards_caller_transcript());
    }

    #[test]
    fn current_compat_does_not_consume_layer_b_audio_evidence() {
        let policy = ConversationPolicyConfig::default();
        let audio = audio_barge_in_config(
            AudioBargeInMode::EchoAwareOnset,
            AudioBargeInUncertainPolicy::ContinuePlayback,
        );
        let evidence = audio_evidence(AudioOnsetDecision::LikelyAssistantEcho, "playback-1", 7, 20);

        let barge_in = BargeInQualityConfig::default();
        let decision =
            policy.decide_transcript_barge_in_with_audio(TranscriptBargeInDecisionInput {
                barge_in: &barge_in,
                audio_barge_in: &audio,
                trigger: BargeInTrigger::Partial,
                transcript: BargeInTranscriptEvidence {
                    text: "stop",
                    active_playback: true,
                    confidence: Some(0.91),
                    stability: Some(0.91),
                },
                audio: Some(&evidence),
                active_playback: ActivePlaybackTarget {
                    playback_id: Some("playback-1"),
                    playback_epoch: Some(7),
                },
                now: Instant::now(),
            });

        assert!(decision.cancels_playback());
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
