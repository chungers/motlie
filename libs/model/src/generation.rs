use crate::chat::ChatMessage;

/// Shared generation parameters used by text-producing capabilities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// Request for chat-style generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
}

/// Response from a chat-style generation call.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChatResponse {
    pub content: String,
}

/// Request for text completion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompletionRequest {
    pub prompt: String,
    pub params: GenerationParams,
}

/// Response from text completion.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompletionResponse {
    pub content: String,
}
