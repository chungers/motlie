use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const TEXT_CALL_PROTOCOL: &str = "motlie.telnyx.text.v1";
pub const TEXT_CALL_CONTENT_TYPE: &str = "text/plain; charset=utf-8";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextCallDirection {
    Inbound,
    Outbound,
}

impl TextCallDirection {
    pub fn label(self) -> &'static str {
        match self {
            Self::Inbound => "inbound",
            Self::Outbound => "outbound",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextStreamDescriptor {
    pub transport: String,
    pub content: String,
    pub turn_based: bool,
}

impl Default for TextStreamDescriptor {
    fn default() -> Self {
        Self {
            transport: "websocket".to_string(),
            content: TEXT_CALL_CONTENT_TYPE.to_string(),
            turn_based: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextCallInfo {
    pub id: String,
    pub direction: TextCallDirection,
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CallOfferPayload {
    #[serde(rename = "type")]
    pub kind: String,
    pub protocol: String,
    pub offer_id: String,
    pub call: TextCallInfo,
    pub text_stream: TextStreamDescriptor,
}

impl CallOfferPayload {
    pub fn new(offer_id: String, call: TextCallInfo) -> Self {
        Self {
            kind: "call.offer".to_string(),
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            offer_id,
            call,
            text_stream: TextStreamDescriptor::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CallConnectedPayload {
    #[serde(rename = "type")]
    pub kind: String,
    pub protocol: String,
    pub call: TextCallInfo,
    pub text_stream: TextStreamDescriptor,
}

impl CallConnectedPayload {
    pub fn new(call: TextCallInfo) -> Self {
        Self {
            kind: "call.connected".to_string(),
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            call,
            text_stream: TextStreamDescriptor::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceptCallResponse {
    pub protocol: String,
    pub call_url: String,
    #[serde(default)]
    pub accept: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GatewayTextFrame {
    #[serde(rename = "session.start")]
    SessionStart {
        protocol: String,
        call_id: String,
        direction: TextCallDirection,
    },
    #[serde(rename = "caller.turn")]
    CallerTurn {
        turn_id: String,
        sequence: u64,
        text: String,
    },
    #[serde(rename = "playback.started")]
    PlaybackStarted { turn_id: String, sequence: u64 },
    #[serde(rename = "playback.finished")]
    PlaybackFinished { turn_id: String, sequence: u64 },
    #[serde(rename = "session.end")]
    SessionEnd { reason: String, sequence: u64 },
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
        sequence: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentTextFrame {
    #[serde(rename = "agent.turn")]
    AgentTurn { turn_id: String, text: String },
    #[serde(rename = "agent.close")]
    AgentClose { reason: Option<String> },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextCallMetadata {
    #[serde(default)]
    pub values: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_caller_turn_as_protocol_frame() {
        let frame = GatewayTextFrame::CallerTurn {
            turn_id: "turn-test".to_string(),
            sequence: 1,
            text: "hello".to_string(),
        };
        let encoded = serde_json::to_value(frame).expect("frame serializes");
        assert_eq!(encoded["type"], "caller.turn");
        assert_eq!(encoded["turn_id"], "turn-test");
        assert_eq!(encoded["sequence"], 1);
    }

    #[test]
    fn app_agent_turn_deserializes() {
        let frame: AgentTextFrame =
            serde_json::from_str(r#"{"type":"agent.turn","turn_id":"turn-test","text":"reply"}"#)
                .expect("agent frame deserializes");
        assert_eq!(
            frame,
            AgentTextFrame::AgentTurn {
                turn_id: "turn-test".to_string(),
                text: "reply".to_string(),
            }
        );
    }
}
