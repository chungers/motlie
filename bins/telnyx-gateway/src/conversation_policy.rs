use std::collections::VecDeque;
use std::time::Instant;

use crate::quality::{
    BargeInQualityConfig, ConversationPolicyConfig, ConversationPolicyMode, PendingOutputOrder,
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
            reset_turn_batch: false,
        }
    }

    pub fn cancels_playback(self) -> bool {
        self.allowed && self.playback == PlaybackPolicyAction::CancelActive
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
    fn coalesce_policy_preserves_caller_audio_after_cancel() {
        let policy = ConversationPolicyConfig {
            mode: ConversationPolicyMode::BargeInCoalesceAfterSilence,
            active_playback_hold_ms: 1_000,
            max_pending_outputs: 1,
            pending_output_order: PendingOutputOrder::LatestOnly,
            post_barge_in_silence_ms: 1_500,
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
