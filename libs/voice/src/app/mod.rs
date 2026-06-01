use async_trait::async_trait;
use motlie_model::TranscriptionUpdate;

use crate::telephony::CallAction;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallIds {
    pub provider_call_id: String,
    pub provider_session_id: Option<String>,
    pub media_stream_id: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CallContext {
    pub ids: Option<CallIds>,
    pub custom_state: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TranscriptEvent {
    Partial {
        text: String,
        update: TranscriptionUpdate,
    },
    Final {
        text: String,
        update: TranscriptionUpdate,
    },
}

impl TranscriptEvent {
    pub fn text(&self) -> &str {
        match self {
            Self::Partial { text, .. } | Self::Final { text, .. } => text,
        }
    }

    pub fn is_final(&self) -> bool {
        matches!(self, Self::Final { .. })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct VoiceAppError {
    pub message: String,
}

impl VoiceAppError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait TranscriptSink: Send + Sync {
    async fn on_transcript(
        &self,
        event: TranscriptEvent,
        context: &mut CallContext,
    ) -> Result<Vec<CallAction>, VoiceAppError>;
}

pub enum ConversationCommand {
    Say { text: String },
    Call(CallAction),
    Noop,
}

#[async_trait]
pub trait ConversationHandler: Send + Sync {
    async fn on_transcript(
        &self,
        event: TranscriptEvent,
        context: &mut CallContext,
    ) -> Result<ConversationCommand, VoiceAppError>;
}
