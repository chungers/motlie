use futures_util::{Stream, StreamExt};
use motlie_voice::app::{ConversationCommand, TranscriptEvent};

use crate::early_response::{EarlyResponseEvent, EarlyResponseIntent};

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
}

impl ConversationProcessorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Identity => "identity",
        }
    }

    pub fn process_input(
        self,
        input: ConversationProcessorInput,
    ) -> Option<ConversationProcessorOutput> {
        match self {
            Self::Identity => identity::IdentityRepeatConversationProcessor.process_input(input),
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
