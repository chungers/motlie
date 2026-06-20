use futures_util::{Stream, StreamExt};
use motlie_voice::app::{ConversationCommand, TranscriptEvent};

use crate::early_response::{EarlyResponseEvent, EarlyResponseIntent};
use crate::operator::state::SpeechOutputConfig;
use crate::text_calls::turns::{AgentTextFrame, TextCallAggregationPolicy};
use crate::text_calls::websocket::{TextCallSessionConfig, TextCallTurnTiming};

pub(crate) mod external_text;
mod identity;

/// Events that may drive a conversation processor.
#[derive(Clone, Debug)]
pub enum ConversationProcessorInput {
    /// Provisional ASR-derived event before the final committed turn boundary.
    EarlyResponse(EarlyResponseEvent),
    /// Final, post-settle conversation turn.
    CommittedTurn(ConversationCommittedTurn),
    /// Decoded agent return frame from an attached text-call websocket.
    AgentTextStream(AgentTextStreamInput),
}

/// Final turn data sent to buffered processors after endpoint settle/coalescing.
#[derive(Clone, Debug)]
pub struct ConversationCommittedTurn {
    pub call_id: String,
    pub turn_id: Option<String>,
    pub text: String,
    pub event: TranscriptEvent,
}

#[derive(Clone, Debug)]
pub struct AgentTextStreamInput {
    pub call_id: String,
    pub frame: AgentTextFrame,
    pub(crate) timing: Option<TextCallTurnTiming>,
    pub(crate) config: TextCallSessionConfig,
    pub speech_output: SpeechOutputConfig,
    pub aggregation: TextCallAggregationPolicy,
}

#[derive(Clone, Debug)]
pub struct CommittedSpeechIntent {
    pub call_id: String,
    pub turn_id: String,
    pub text: String,
    pub final_fragment: bool,
    pub(crate) timing: TextCallTurnTiming,
    pub(crate) config: TextCallSessionConfig,
    pub speech_output: SpeechOutputConfig,
}

/// Processor output accepted by the gateway.
pub enum ConversationProcessorOutput {
    /// Provisional speech/cancel/commit intent for the early-response pipeline.
    EarlyResponse(EarlyResponseIntent),
    /// Committed agent speech intent for the shared speech-output stage.
    CommittedSpeech(CommittedSpeechIntent),
    /// Committed-turn command for normal conversation state.
    Command(ConversationCommand),
    /// Conversation-scoped failure. This records failed state without failing the media path.
    Error(String),
}

/// Static processor selection for a call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConversationProcessorKind {
    /// Repeat accepted caller text exactly for both committed and provisional paths.
    #[default]
    Identity,
    /// External app/telnyx-agent text stream attached through the text-call protocol.
    ///
    /// This is adapter-backed: inbound caller turns are delivered by the text-call
    /// websocket registry, and agent return frames enter the shared processor
    /// pipeline as `AgentTextStream` inputs.
    ExternalTextStream,
}

impl ConversationProcessorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::ExternalTextStream => "external_text_stream",
        }
    }

    pub fn process_input(
        self,
        input: ConversationProcessorInput,
    ) -> Option<ConversationProcessorOutput> {
        if matches!(input, ConversationProcessorInput::AgentTextStream(_)) {
            return external_text::ExternalTextStreamConversationProcessor.process_input(input);
        }
        match self {
            Self::Identity => identity::IdentityRepeatConversationProcessor.process_input(input),
            Self::ExternalTextStream => {
                external_text::ExternalTextStreamConversationProcessor.process_input(input)
            }
        }
    }

    pub fn process_stream<S>(
        self,
        inputs: S,
    ) -> impl Stream<Item = ConversationProcessorOutput> + Send
    where
        S: Stream<Item = ConversationProcessorInput> + Send,
    {
        inputs.filter_map(move |input| async move { self.process_input(input) })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use motlie_voice::app::TranscriptEvent;

    use super::*;
    use crate::early_response::AppendOrReplace;
    use crate::quality::config::TextCallQualityConfig;

    fn agent_stream_input(
        frame: AgentTextFrame,
        timing: Option<TextCallTurnTiming>,
    ) -> ConversationProcessorInput {
        ConversationProcessorInput::AgentTextStream(AgentTextStreamInput {
            call_id: "call-agent".to_string(),
            frame,
            timing,
            config: TextCallSessionConfig::from(&TextCallQualityConfig::default()),
            speech_output: SpeechOutputConfig::default(),
            aggregation: TextCallAggregationPolicy::GatewayOwned,
        })
    }

    fn turn_timing() -> TextCallTurnTiming {
        let now = Instant::now();
        TextCallTurnTiming {
            finalized_at: now,
            caller_turn_sent_at: now,
        }
    }

    #[test]
    fn external_text_stream_processor_maps_agent_committed_frames_to_speech_intents() {
        let partial =
            ConversationProcessorKind::ExternalTextStream.process_input(agent_stream_input(
                AgentTextFrame::AgentTurnPartial {
                    turn_id: "turn-1".to_string(),
                    text: "partial".to_string(),
                    append: true,
                },
                Some(turn_timing()),
            ));
        match partial {
            Some(ConversationProcessorOutput::CommittedSpeech(intent)) => {
                assert_eq!(intent.call_id, "call-agent");
                assert_eq!(intent.turn_id, "turn-1");
                assert_eq!(intent.text, "partial");
                assert!(!intent.final_fragment);
            }
            _ => panic!("agent.turn.partial should produce committed speech"),
        }

        let final_turn =
            ConversationProcessorKind::ExternalTextStream.process_input(agent_stream_input(
                AgentTextFrame::AgentTurn {
                    turn_id: "turn-1".to_string(),
                    text: "final".to_string(),
                },
                Some(turn_timing()),
            ));
        match final_turn {
            Some(ConversationProcessorOutput::CommittedSpeech(intent)) => {
                assert_eq!(intent.call_id, "call-agent");
                assert_eq!(intent.turn_id, "turn-1");
                assert_eq!(intent.text, "final");
                assert!(intent.final_fragment);
            }
            _ => panic!("agent.turn should produce committed speech"),
        }
    }

    #[test]
    fn external_text_stream_processor_maps_agent_provisional_frames_to_speech_intents() {
        let partial =
            ConversationProcessorKind::ExternalTextStream.process_input(agent_stream_input(
                AgentTextFrame::AgentTurnProvisionalPartial {
                    provisional_turn_id: "pt-1".to_string(),
                    generation: 4,
                    text: "partial".to_string(),
                    append: true,
                },
                None,
            ));
        match partial {
            Some(ConversationProcessorOutput::EarlyResponse(EarlyResponseIntent::Speak {
                provisional_turn_id,
                call_id,
                utterance_id,
                generation,
                text,
                append_or_replace,
                final_fragment,
            })) => {
                assert_eq!(provisional_turn_id, "pt-1");
                assert_eq!(call_id, "call-agent");
                assert_eq!(utterance_id, "pt-1");
                assert_eq!(generation, 4);
                assert_eq!(text, "partial");
                assert_eq!(append_or_replace, AppendOrReplace::Append);
                assert!(!final_fragment);
            }
            _ => panic!("agent.turn.provisional.partial should produce provisional speech"),
        }

        let final_turn =
            ConversationProcessorKind::ExternalTextStream.process_input(agent_stream_input(
                AgentTextFrame::AgentTurnProvisional {
                    provisional_turn_id: "pt-1".to_string(),
                    generation: 4,
                    text: "final".to_string(),
                },
                None,
            ));
        match final_turn {
            Some(ConversationProcessorOutput::EarlyResponse(EarlyResponseIntent::Speak {
                provisional_turn_id,
                call_id,
                utterance_id,
                generation,
                text,
                append_or_replace,
                final_fragment,
            })) => {
                assert_eq!(provisional_turn_id, "pt-1");
                assert_eq!(call_id, "call-agent");
                assert_eq!(utterance_id, "pt-1");
                assert_eq!(generation, 4);
                assert_eq!(text, "final");
                assert_eq!(append_or_replace, AppendOrReplace::Append);
                assert!(final_fragment);
            }
            _ => panic!("agent.turn.provisional should produce provisional speech"),
        }
    }

    #[test]
    fn external_text_stream_processor_is_local_noop() {
        let input = ConversationProcessorInput::CommittedTurn(ConversationCommittedTurn {
            call_id: "call-external".to_string(),
            turn_id: Some("turn-external".to_string()),
            text: "hello".to_string(),
            event: TranscriptEvent::Final {
                text: "hello".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            },
        });

        assert!(ConversationProcessorKind::ExternalTextStream
            .process_input(input)
            .is_none());
    }
}
