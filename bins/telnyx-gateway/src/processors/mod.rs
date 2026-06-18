use futures_util::{Stream, StreamExt};
use motlie_voice::app::{ConversationCommand, TranscriptEvent};

use crate::early_response::{EarlyResponseEvent, EarlyResponseIntent};

pub(crate) mod external_text;
mod identity;

/// Events that may drive a conversation processor.
#[derive(Clone, Debug)]
pub enum ConversationProcessorInput {
    /// Provisional ASR-derived event before the final committed turn boundary.
    EarlyResponse(EarlyResponseEvent),
    /// Final, post-settle conversation turn.
    CommittedTurn(ConversationCommittedTurn),
}

/// Final turn data sent to buffered processors after endpoint settle/coalescing.
#[derive(Clone, Debug)]
pub struct ConversationCommittedTurn {
    pub call_id: String,
    pub turn_id: Option<String>,
    pub text: String,
    pub event: TranscriptEvent,
}

/// Processor output accepted by the gateway.
pub enum ConversationProcessorOutput {
    /// Provisional speech/cancel/commit intent for the early-response pipeline.
    EarlyResponse(EarlyResponseIntent),
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
    /// websocket registry, and agent frames are handled by `processors::external_text`.
    /// Local `process_input` intentionally returns no output so the media/ASR
    /// processor dispatcher does not synthesize a competing response.
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
        match self {
            Self::Identity => identity::IdentityRepeatConversationProcessor.process_input(input),
            // Adapter-backed processor: websocket-owned external text streams
            // handle agent frames and queue speech in `processors::external_text`.
            Self::ExternalTextStream => None,
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
    use motlie_voice::app::TranscriptEvent;

    use super::*;

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
