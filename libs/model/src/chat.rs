use crate::ContentKind;

/// Role labels used in chat-style requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    Assistant,
    System,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ContentPart>,
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
        }
    }

    pub fn with_parts(role: ChatRole, content: Vec<ContentPart>) -> Self {
        Self { role, content }
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
                ContentPart::Image {
                    data,
                    media_type: media_type.into(),
                },
                ContentPart::Text(text.into()),
            ],
        }
    }
}
