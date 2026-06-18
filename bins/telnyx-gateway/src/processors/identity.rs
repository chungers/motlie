use motlie_voice::app::ConversationCommand;

use super::{ConversationProcessorInput, ConversationProcessorOutput};
use crate::early_response::{AppendOrReplace, EarlyResponseEvent, EarlyResponseIntent};

/// Minimal processor used for smoke tests and identity/repeat live validation.
///
/// It repeats committed turns exactly and passes provisional early-response text through
/// without adding a response prefix.
#[derive(Clone, Debug, Default)]
pub struct IdentityRepeatConversationProcessor;

impl IdentityRepeatConversationProcessor {
    pub(crate) fn process_input(
        self,
        input: ConversationProcessorInput,
    ) -> Option<ConversationProcessorOutput> {
        match input {
            ConversationProcessorInput::EarlyResponse(event) => {
                repeat_early_response(event).map(ConversationProcessorOutput::EarlyResponse)
            }
            ConversationProcessorInput::CommittedTurn(turn) => {
                let text = turn.text.trim().to_string();
                if text.is_empty() {
                    Some(ConversationProcessorOutput::Command(
                        ConversationCommand::Noop,
                    ))
                } else {
                    Some(ConversationProcessorOutput::Command(
                        ConversationCommand::Say { text },
                    ))
                }
            }
        }
    }
}

fn repeat_early_response(event: EarlyResponseEvent) -> Option<EarlyResponseIntent> {
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

#[cfg(test)]
mod tests {
    use motlie_voice::app::{ConversationCommand, TranscriptEvent};

    use super::*;
    use crate::early_response::{
        EarlyResponseCancelReason, EarlyResponseCommitRole, EarlyResponseIntent,
    };
    use crate::processors::{ConversationCommittedTurn, ConversationProcessorKind};
    use crate::text_calls::turns::CallerSpeechState;

    fn process_committed_turn(text: &str) -> Option<ConversationCommand> {
        let input = ConversationProcessorInput::CommittedTurn(ConversationCommittedTurn {
            call_id: "call-1".to_string(),
            turn_id: Some("turn-1".to_string()),
            text: text.to_string(),
            event: TranscriptEvent::Final {
                text: text.to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
        });
        match ConversationProcessorKind::Identity.process_input(input) {
            Some(ConversationProcessorOutput::Command(command)) => Some(command),
            Some(_other) => panic!("unexpected processor output"),
            None => None,
        }
    }

    #[test]
    fn identity_repeat_processor_turns_committed_transcript_into_plain_say() {
        let command =
            process_committed_turn("hello").expect("identity processor should emit a command");

        match command {
            ConversationCommand::Say { text } => assert_eq!(text, "hello"),
            _ => panic!("expected say command"),
        }
    }

    fn process_early_response_event(event: EarlyResponseEvent) -> Option<EarlyResponseIntent> {
        let input = ConversationProcessorInput::EarlyResponse(event);
        match ConversationProcessorKind::Identity.process_input(input) {
            Some(ConversationProcessorOutput::EarlyResponse(intent)) => Some(intent),
            Some(_other) => panic!("unexpected processor output"),
            None => None,
        }
    }

    #[test]
    fn identity_repeat_processor_repeats_early_response_start_as_identity_fragment() {
        let intent = process_early_response_event(EarlyResponseEvent::Started {
            provisional_turn_id: "pt-1".to_string(),
            call_id: "call-1".to_string(),
            utterance_id: "utt-1".to_string(),
            generation: 1,
            text: "I need a tow truck.".to_string(),
            confidence: Some(0.91),
            stability: Some(0.86),
            speech_state: CallerSpeechState::EndpointCandidate,
        });

        assert_eq!(
            intent,
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

    #[test]
    fn identity_repeat_processor_preserves_early_response_update_cancel_and_commit() {
        assert_eq!(
            process_early_response_event(EarlyResponseEvent::Updated {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                text: "I need a tow truck in Oakland.".to_string(),
                append_or_replace: AppendOrReplace::Replace,
            }),
            Some(EarlyResponseIntent::Speak {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                text: "I need a tow truck in Oakland.".to_string(),
                append_or_replace: AppendOrReplace::Replace,
            })
        );
        assert_eq!(
            process_early_response_event(EarlyResponseEvent::Canceled {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                reason: EarlyResponseCancelReason::AsrCorrection,
            }),
            Some(EarlyResponseIntent::Cancel {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                reason: EarlyResponseCancelReason::AsrCorrection,
            })
        );
        assert_eq!(
            process_early_response_event(EarlyResponseEvent::Committed {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                utterance_id: "utt-1".to_string(),
                generation: 2,
                turn_id: "turn-1".to_string(),
                coalesced_turn_ids: Vec::new(),
                coalesced_utterance_ids: vec!["utt-1".to_string()],
                member_final_text: "I need a tow truck in Oakland.".to_string(),
                final_text: "I need a tow truck in Oakland.".to_string(),
                role: EarlyResponseCommitRole::PrimaryPlayback,
            }),
            Some(EarlyResponseIntent::Commit {
                provisional_turn_id: "pt-1".to_string(),
                call_id: "call-1".to_string(),
                generation: 2,
                turn_id: "turn-1".to_string(),
            })
        );
    }

    #[test]
    fn identity_repeat_processor_empty_committed_turn_is_noop() {
        let command =
            process_committed_turn("   ").expect("identity processor should emit a command");

        match command {
            ConversationCommand::Noop => {}
            _ => panic!("expected noop command"),
        }
    }
}
