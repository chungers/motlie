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

impl GenerationParams {
    /// Fill unset fields from `defaults`.
    ///
    /// Scalar options keep the caller-provided value when present. Stop
    /// sequences use the caller list when non-empty and fall back to defaults
    /// only when the caller list is empty.
    pub fn with_defaults(mut self, defaults: &Self) -> Self {
        self.max_tokens = self.max_tokens.or(defaults.max_tokens);
        self.temperature = self.temperature.or(defaults.temperature);
        self.top_p = self.top_p.or(defaults.top_p);
        if self.stop_sequences.is_empty() {
            self.stop_sequences = defaults.stop_sequences.clone();
        }
        self
    }
}

/// Backend hint for models that support optional thinking/reasoning tokens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ThinkingMode {
    #[default]
    Disabled,
    Auto,
}

/// Request for chat-style generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<ThinkingMode>,
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
    /// Backend-specific finish reason not yet represented by this enum.
    ///
    /// Callers must treat this as diagnostic data and should not match on the
    /// contained string for portable behavior.
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
    fn generation_params_keep_explicit_values_over_defaults() {
        let params = GenerationParams {
            max_tokens: Some(64),
            temperature: Some(0.2),
            stop_sequences: vec!["caller-stop".to_string()],
            ..Default::default()
        };
        let defaults = GenerationParams {
            max_tokens: Some(128),
            temperature: Some(1.0),
            top_p: Some(0.95),
            stop_sequences: vec!["default-stop".to_string()],
        };

        let merged = params.with_defaults(&defaults);

        assert_eq!(merged.max_tokens, Some(64));
        assert_eq!(merged.temperature, Some(0.2));
        assert_eq!(merged.top_p, Some(0.95));
        assert_eq!(merged.stop_sequences, ["caller-stop"]);
    }

    #[test]
    fn generation_params_use_default_stop_sequences_when_caller_has_none() {
        let merged = GenerationParams {
            max_tokens: Some(64),
            ..Default::default()
        }
        .with_defaults(&GenerationParams {
            max_tokens: Some(128),
            stop_sequences: vec!["<end>".to_string()],
            ..Default::default()
        });

        assert_eq!(merged.max_tokens, Some(64));
        assert_eq!(merged.stop_sequences, ["<end>"]);
    }

    #[test]
    fn text_response_defaults_tool_metadata() {
        let response = ChatResponse::text("done");

        assert_eq!(response.content, "done");
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.finish_reason, None);
        assert_eq!(response.reasoning, None);
        assert_eq!(response.usage, None);
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
