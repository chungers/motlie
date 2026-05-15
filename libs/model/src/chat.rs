use thiserror::Error;

use crate::tool::{ToolCall, ToolCallId, ToolCallIdError, ToolName, ToolNameError};
use crate::ContentKind;

/// Role labels used in chat-style requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    Assistant,
    System,
    Tool,
    User,
}

/// One normalized content part in a chat message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentPart {
    Text(String),
    Image { data: Vec<u8>, media_type: String },
    ImageUrl { url: String },
}

impl ContentPart {
    pub fn kind(&self) -> ContentKind {
        match self {
            Self::Text(_) => ContentKind::Text,
            Self::Image { .. } | Self::ImageUrl { .. } => ContentKind::Image,
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn image(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::Image {
            data,
            media_type: media_type.into(),
        }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl { url: url.into() }
    }
}

/// Single message in a chat request.
///
/// Prefer the role-specific constructors on this type. Fields remain public so
/// backend adapters and tests can build exact transcripts, but callers should
/// run `validate_tool_metadata` before sending hand-built messages that carry
/// tool calls or tool results.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ContentPart>,
    pub name: Option<ToolName>,
    pub tool_call_id: Option<ToolCallId>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: Option<String>,
}

impl ChatMessage {
    /// Convenience constructor for the dominant text-only path.
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self::text(role, content)
    }

    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(content.into())],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }

    pub fn with_parts(role: ChatRole, content: Vec<ContentPart>) -> Self {
        Self {
            role,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }

    pub fn text_and_image(
        role: ChatRole,
        text: impl Into<String>,
        data: Vec<u8>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            role,
            content: vec![
                ContentPart::Text(text.into()),
                ContentPart::Image {
                    data,
                    media_type: media_type.into(),
                },
            ],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: Vec::new(),
            name: None,
            tool_call_id: None,
            tool_calls,
            reasoning: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<Self, ChatMessageError> {
        Ok(Self::tool_result_parts(
            ToolCallId::new(tool_call_id)?,
            ToolName::new(name)?,
            content,
        ))
    }

    pub fn tool_result_parts(
        tool_call_id: ToolCallId,
        name: ToolName,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: ChatRole::Tool,
            content: vec![ContentPart::Text(content.into())],
            name: Some(name),
            tool_call_id: Some(tool_call_id),
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }

    pub fn with_name(mut self, name: ToolName) -> Self {
        self.name = Some(name);
        self
    }

    pub fn try_with_name(mut self, name: impl Into<String>) -> Result<Self, ChatMessageError> {
        self.name = Some(ToolName::new(name)?);
        Ok(self)
    }

    pub fn with_tool_call_id(mut self, tool_call_id: ToolCallId) -> Self {
        self.tool_call_id = Some(tool_call_id);
        self
    }

    pub fn try_with_tool_call_id(
        mut self,
        tool_call_id: impl Into<String>,
    ) -> Result<Self, ChatMessageError> {
        self.tool_call_id = Some(ToolCallId::new(tool_call_id)?);
        Ok(self)
    }

    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    pub fn requires_tool_use(&self) -> bool {
        self.role == ChatRole::Tool || self.tool_call_id.is_some() || !self.tool_calls.is_empty()
    }

    pub fn validate_tool_metadata(&self) -> Result<(), ChatMessageError> {
        if self.role != ChatRole::Assistant && !self.tool_calls.is_empty() {
            return Err(ChatMessageError::InvalidToolMetadata(
                "only assistant messages may carry tool calls",
            ));
        }
        if self.role != ChatRole::Tool && self.tool_call_id.is_some() {
            return Err(ChatMessageError::InvalidToolMetadata(
                "only tool messages may carry a tool call id",
            ));
        }
        if self.role == ChatRole::Tool && self.tool_call_id.is_none() {
            return Err(ChatMessageError::InvalidToolMetadata(
                "tool messages require a tool call id",
            ));
        }
        if self.role == ChatRole::Tool && !self.tool_calls.is_empty() {
            return Err(ChatMessageError::InvalidToolMetadata(
                "tool result messages cannot carry new tool calls",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ChatMessageError {
    #[error(transparent)]
    InvalidToolCallId(#[from] ToolCallIdError),
    #[error(transparent)]
    InvalidToolName(#[from] ToolNameError),
    #[error("{0}")]
    InvalidToolMetadata(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_image_preserves_parameter_order() {
        let message = ChatMessage::text_and_image(
            ChatRole::User,
            "describe this",
            vec![1, 2, 3],
            "image/png",
        );

        assert!(matches!(
            message.content.as_slice(),
            [
                ContentPart::Text(text),
                ContentPart::Image { data, media_type }
            ] if text == "describe this" && data == &vec![1, 2, 3] && media_type == "image/png"
        ));
    }

    #[test]
    fn text_constructor_defaults_tool_metadata() {
        let message = ChatMessage::text(ChatRole::User, "hello");

        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.name, None);
        assert_eq!(message.tool_call_id, None);
        assert!(message.tool_calls.is_empty());
        assert_eq!(message.reasoning, None);
        assert!(!message.requires_tool_use());
    }

    #[test]
    fn assistant_tool_call_message_preserves_calls() {
        let call = ToolCall::from_serializable_args(
            "call-1",
            "get_weather",
            &serde_json::json!({ "city": "Seattle" }),
        )
        .expect("args should serialize");

        let message = ChatMessage::assistant_tool_calls(vec![call.clone()]);

        assert_eq!(message.role, ChatRole::Assistant);
        assert!(message.content.is_empty());
        assert_eq!(message.tool_calls, vec![call]);
        assert!(message.requires_tool_use());
    }

    #[test]
    fn tool_result_message_carries_correlation_fields() {
        let message = ChatMessage::tool_result("call-1", "get_weather", "{\"temp\":72}")
            .expect("tool metadata should validate");

        assert_eq!(message.role, ChatRole::Tool);
        assert_eq!(
            message.name.as_ref().map(ToolName::as_str),
            Some("get_weather")
        );
        assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
        assert!(message.requires_tool_use());
        assert_eq!(
            message.content,
            vec![ContentPart::Text("{\"temp\":72}".into())]
        );
        assert!(message.validate_tool_metadata().is_ok());
    }

    #[test]
    fn validate_tool_metadata_rejects_invalid_role_combinations() {
        let mut user_message = ChatMessage::text(ChatRole::User, "hello")
            .try_with_tool_call_id("call-1")
            .expect("id should validate");
        assert!(matches!(
            user_message.validate_tool_metadata(),
            Err(ChatMessageError::InvalidToolMetadata(
                "only tool messages may carry a tool call id"
            ))
        ));

        user_message.tool_call_id = None;
        user_message.tool_calls.push(
            ToolCall::from_json_args("call-1", "get_weather", r#"{"city":"Seattle"}"#)
                .expect("call should validate"),
        );
        assert!(matches!(
            user_message.validate_tool_metadata(),
            Err(ChatMessageError::InvalidToolMetadata(
                "only assistant messages may carry tool calls"
            ))
        ));

        let tool_message = ChatMessage::text(ChatRole::Tool, "{}");
        assert!(matches!(
            tool_message.validate_tool_metadata(),
            Err(ChatMessageError::InvalidToolMetadata(
                "tool messages require a tool call id"
            ))
        ));
    }
}
