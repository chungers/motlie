use crate::chat::ChatMessage;
use crate::tool::{ToolCall, ToolChoice, ToolSpec};

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
    pub tools: Vec<ToolSpec>,
    pub tool_choice: Option<ToolChoice>,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }

    pub fn requires_tool_use(&self) -> bool {
        !self.tools.is_empty()
            || self.messages.iter().any(ChatMessage::requires_tool_use)
            || matches!(
                self.tool_choice,
                Some(ToolChoice::Auto | ToolChoice::Named(_) | ToolChoice::Required)
            )
    }
}

/// Response from a chat-style generation call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatFinishReason {
    ContentFilter,
    Length,
    Other(String),
    Stop,
    ToolCalls,
}

/// Request-local token usage reported by a backend, when available.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GenerationUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// Response from a chat-style generation call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<ChatFinishReason>,
    pub reasoning: Option<String>,
    pub usage: Option<GenerationUsage>,
    pub raw_message: Option<String>,
}

impl ChatResponse {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    pub fn tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls,
            finish_reason: Some(ChatFinishReason::ToolCalls),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatRole, ToolCall};

    #[allow(dead_code)]
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct WeatherArgs {
        city: String,
    }

    #[test]
    fn plain_chat_request_does_not_require_tool_use() {
        let request = ChatRequest::new(vec![ChatMessage::text(ChatRole::User, "hello")]);

        assert!(!request.requires_tool_use());
    }

    #[test]
    fn request_with_tools_requires_tool_use() {
        let request = ChatRequest {
            messages: vec![ChatMessage::text(ChatRole::User, "hello")],
            tools: vec![
                ToolSpec::from_args::<WeatherArgs>("get_weather", "Get current weather.")
                    .expect("schema should build"),
            ],
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };

        assert!(request.requires_tool_use());
    }

    #[test]
    fn request_with_replayed_tool_call_requires_tool_use() {
        let call =
            ToolCall::from_json_args("call-1", "get_weather", "{}").expect("args should validate");
        let request = ChatRequest::new(vec![ChatMessage::assistant_tool_calls(vec![call])]);

        assert!(request.requires_tool_use());
    }

    #[test]
    fn text_response_defaults_tool_metadata() {
        let response = ChatResponse::text("done");

        assert_eq!(response.content, "done");
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.finish_reason, None);
        assert_eq!(response.reasoning, None);
        assert_eq!(response.usage, None);
        assert_eq!(response.raw_message, None);
    }
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
