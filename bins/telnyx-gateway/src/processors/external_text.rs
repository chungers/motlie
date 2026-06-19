use crate::early_response::{AppendOrReplace, EarlyResponseIntent};
use crate::processors::{
    CommittedSpeechIntent, ConversationProcessorInput, ConversationProcessorOutput,
};
use crate::text_calls::turns::AgentTextFrame;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExternalTextStreamConversationProcessor;

impl ExternalTextStreamConversationProcessor {
    pub(crate) fn process_input(
        self,
        input: ConversationProcessorInput,
    ) -> Option<ConversationProcessorOutput> {
        let ConversationProcessorInput::AgentTextStream(input) = input else {
            return None;
        };
        match input.frame {
            AgentTextFrame::AgentTurnPartial { turn_id, text, .. } => {
                let timing = match input.timing {
                    Some(timing) => timing,
                    None => {
                        return Some(ConversationProcessorOutput::Error(
                            "agent.turn.partial missing accepted turn timing".to_string(),
                        ))
                    }
                };
                Some(ConversationProcessorOutput::CommittedSpeech(
                    CommittedSpeechIntent {
                        call_id: input.call_id,
                        turn_id,
                        text,
                        final_fragment: false,
                        timing,
                        config: input.config,
                        speech_output: input.speech_output,
                    },
                ))
            }
            AgentTextFrame::AgentTurn { turn_id, text } => {
                let timing = match input.timing {
                    Some(timing) => timing,
                    None => {
                        return Some(ConversationProcessorOutput::Error(
                            "agent.turn missing accepted turn timing".to_string(),
                        ))
                    }
                };
                Some(ConversationProcessorOutput::CommittedSpeech(
                    CommittedSpeechIntent {
                        call_id: input.call_id,
                        turn_id,
                        text,
                        final_fragment: true,
                        timing,
                        config: input.config,
                        speech_output: input.speech_output,
                    },
                ))
            }
            AgentTextFrame::AgentTurnProvisionalPartial {
                provisional_turn_id,
                generation,
                text,
                ..
            } => Some(ConversationProcessorOutput::EarlyResponse(
                EarlyResponseIntent::Speak {
                    provisional_turn_id: provisional_turn_id.clone(),
                    call_id: input.call_id,
                    utterance_id: provisional_turn_id,
                    generation,
                    text,
                    append_or_replace: AppendOrReplace::Append,
                    final_fragment: false,
                },
            )),
            AgentTextFrame::AgentTurnProvisional {
                provisional_turn_id,
                generation,
                text,
            } => Some(ConversationProcessorOutput::EarlyResponse(
                EarlyResponseIntent::Speak {
                    provisional_turn_id: provisional_turn_id.clone(),
                    call_id: input.call_id,
                    utterance_id: provisional_turn_id,
                    generation,
                    text,
                    append_or_replace: AppendOrReplace::Append,
                    final_fragment: true,
                },
            )),
            AgentTextFrame::AgentClose { .. } => None,
        }
    }
}
