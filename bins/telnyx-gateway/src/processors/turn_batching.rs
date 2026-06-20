use motlie_agent::voice::turn_batching::{
    BatchDecision, IdentityPromptHandler, Turn, TurnBatchResetReason, TurnBatcher,
};
use motlie_voice::app::ConversationCommand;

use super::{ConversationProcessorInput, ConversationProcessorOutput};
use crate::early_response::{EarlyResponseCancelReason, EarlyResponseEvent};

#[derive(Clone, Debug, Default)]
pub(crate) struct TurnBatchedIdentityConversationProcessor {
    batcher: IdentityPromptHandler,
}

impl TurnBatchedIdentityConversationProcessor {
    pub(crate) fn process_input(
        &mut self,
        input: ConversationProcessorInput,
    ) -> Option<ConversationProcessorOutput> {
        match input {
            ConversationProcessorInput::CommittedTurn(turn) => {
                let text = turn.text.trim().to_string();
                if text.is_empty() {
                    return Some(ConversationProcessorOutput::Command(
                        ConversationCommand::Noop,
                    ));
                }
                let batch_turn = Turn {
                    turn_id: turn.turn_id.unwrap_or_else(|| turn.call_id.clone()),
                    utterance_id: None,
                    text,
                    sequence: 0,
                    epoch: self.batcher.epoch(),
                };
                Some(batch_decision_output(self.batcher.observe(batch_turn)))
            }
            ConversationProcessorInput::EarlyResponse(event) => reset_reason_for_event(&event)
                .map(|reason| batch_decision_output(self.batcher.reset(reason))),
            ConversationProcessorInput::AgentTextStream(_) => None,
        }
    }
}

fn reset_reason_for_event(event: &EarlyResponseEvent) -> Option<TurnBatchResetReason> {
    match event {
        EarlyResponseEvent::Canceled { reason, .. } => Some(match reason {
            EarlyResponseCancelReason::CallerBargeIn => TurnBatchResetReason::BargeIn,
            EarlyResponseCancelReason::StaleGeneration => TurnBatchResetReason::StaleGeneration,
            EarlyResponseCancelReason::CoalescedIntoFinalTurn
            | EarlyResponseCancelReason::SupersededByNewGeneration => {
                TurnBatchResetReason::FinalTurnSuperseded
            }
            EarlyResponseCancelReason::SessionEnded | EarlyResponseCancelReason::Hangup => {
                TurnBatchResetReason::SessionEnd
            }
            EarlyResponseCancelReason::AsrCorrection
            | EarlyResponseCancelReason::FinalTranscriptMismatch
            | EarlyResponseCancelReason::PolicyDisabled
            | EarlyResponseCancelReason::PolicyNoLongerSatisfied
            | EarlyResponseCancelReason::MaxUpdatesExceeded
            | EarlyResponseCancelReason::ProcessorRejected
            | EarlyResponseCancelReason::TtsCanceled => TurnBatchResetReason::Manual,
        }),
        _ => None,
    }
}

fn batch_decision_output(decision: BatchDecision) -> ConversationProcessorOutput {
    match decision {
        BatchDecision::Accumulating(state) => ConversationProcessorOutput::Accumulating(state),
        BatchDecision::PromptComplete(prompt) => {
            ConversationProcessorOutput::PromptComplete(prompt)
        }
        BatchDecision::Reset(reset) => ConversationProcessorOutput::Reset(reset),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::TranscriptionUpdate;
    use motlie_voice::app::TranscriptEvent;

    fn committed_input(turn_id: &str, text: &str) -> ConversationProcessorInput {
        ConversationProcessorInput::CommittedTurn(super::super::ConversationCommittedTurn {
            call_id: "call-turn-batch".to_string(),
            turn_id: Some(turn_id.to_string()),
            text: text.to_string(),
            event: TranscriptEvent::Final {
                text: text.to_string(),
                update: TranscriptionUpdate::default(),
            },
        })
    }

    #[test]
    fn turn_batched_identity_emits_prompt_complete_for_batch_of_one() {
        let mut processor = TurnBatchedIdentityConversationProcessor::default();

        match processor.process_input(committed_input("turn-1", "hello")) {
            Some(ConversationProcessorOutput::PromptComplete(prompt)) => {
                assert_eq!(prompt.batch_id, "turn-batch-0-0");
                assert_eq!(prompt.response_turn_id, "turn-1");
                assert_eq!(prompt.source_turn_ids, vec!["turn-1".to_string()]);
                assert_eq!(prompt.text, "hello");
            }
            _ => panic!("unexpected output"),
        }
    }
}
