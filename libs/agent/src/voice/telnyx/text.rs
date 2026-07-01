use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const TEXT_CALL_PROTOCOL: &str = "motlie.telnyx.text.v1";
pub const TEXT_CALL_DEBUG_EXTENSION: &str = "motlie.telnyx.text.debug.v1";
pub const TEXT_CALL_PARTIALS_EXTENSION: &str = "motlie.telnyx.text.partials.v1";
pub const TEXT_CALL_EARLY_TURNS_EXTENSION: &str = "motlie.telnyx.text.early_turns.v1";
pub const TEXT_CALL_CONTENT_TYPE: &str = "text/plain; charset=utf-8";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseMode {
    #[default]
    PerTurn,
    TurnBatched,
}

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
    #[serde(default)]
    pub extensions: Vec<String>,
}

impl Default for TextStreamDescriptor {
    fn default() -> Self {
        Self {
            transport: "websocket".to_string(),
            content: TEXT_CALL_CONTENT_TYPE.to_string(),
            turn_based: true,
            extensions: vec![
                TEXT_CALL_PARTIALS_EXTENSION.to_string(),
                TEXT_CALL_EARLY_TURNS_EXTENSION.to_string(),
            ],
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
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub response_mode: ResponseMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaybackFinishedStatus {
    Completed,
    Canceled,
    Failed,
    Superseded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallerSpeechState {
    Speaking,
    EndpointCandidate,
    Finalizing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GatewayTextFrame {
    #[serde(rename = "session.start")]
    SessionStart {
        protocol: String,
        call_id: String,
        direction: TextCallDirection,
        #[serde(default)]
        response_mode: ResponseMode,
    },
    #[serde(rename = "caller.turn")]
    CallerTurn {
        turn_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        utterance_id: Option<String>,
        sequence: u64,
        text: String,
    },
    #[serde(rename = "caller.partial")]
    CallerPartial {
        utterance_id: String,
        sequence: u64,
        text: String,
        /// Optional backend/model confidence for the current hypothesis.
        /// It is omitted when the backend does not provide a native score.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
        /// Optional gateway-estimated stream-convergence/churn signal.
        /// Use only for preparation, routing, or debounce decisions. Never use
        /// as truth, final response input, model/ASR confidence, calibrated
        /// probability, or a value to average/combine with `confidence`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stability: Option<f32>,
        speech_state: CallerSpeechState,
        reply_allowed: bool,
    },
    #[serde(rename = "caller.turn.provisional")]
    CallerTurnProvisional {
        provisional_turn_id: String,
        utterance_id: String,
        generation: u64,
        sequence: u64,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stability: Option<f32>,
        speech_state: CallerSpeechState,
        reply_allowed: bool,
    },
    #[serde(rename = "caller.turn.provisional.update")]
    CallerTurnProvisionalUpdate {
        provisional_turn_id: String,
        utterance_id: String,
        generation: u64,
        sequence: u64,
        text: String,
    },
    #[serde(rename = "caller.turn.provisional.cancel")]
    CallerTurnProvisionalCancel {
        provisional_turn_id: String,
        utterance_id: String,
        generation: u64,
        sequence: u64,
        reason: String,
    },
    #[serde(rename = "caller.turn.provisional.commit")]
    CallerTurnProvisionalCommit {
        provisional_turn_id: String,
        turn_id: String,
        utterance_id: String,
        coalesced_utterance_ids: Vec<String>,
        generation: u64,
        sequence: u64,
        final_text: String,
    },
    #[serde(rename = "playback.provisional.started")]
    ProvisionalPlaybackStarted {
        provisional_turn_id: String,
        generation: u64,
        playback_id: String,
        sequence: u64,
    },
    #[serde(rename = "playback.started")]
    PlaybackStarted { turn_id: String, sequence: u64 },
    #[serde(rename = "playback.finished")]
    PlaybackFinished {
        turn_id: String,
        sequence: u64,
        status: PlaybackFinishedStatus,
    },
    #[serde(rename = "turn.superseded")]
    TurnSuperseded {
        turn_id: String,
        superseded_by_turn_id: String,
        reason: String,
        sequence: u64,
    },
    #[serde(rename = "turn_batch.reset")]
    TurnBatchReset {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch_id: Option<String>,
        epoch: u64,
        sequence: u64,
    },
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
    #[serde(rename = "agent.turn.partial")]
    AgentTurnPartial {
        turn_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        epoch: Option<u64>,
        #[serde(default = "default_true")]
        append: bool,
    },
    #[serde(rename = "agent.turn")]
    AgentTurn {
        turn_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        epoch: Option<u64>,
    },
    #[serde(rename = "agent.turn.provisional.partial")]
    AgentTurnProvisionalPartial {
        provisional_turn_id: String,
        generation: u64,
        text: String,
        #[serde(default = "default_true")]
        append: bool,
    },
    #[serde(rename = "agent.turn.provisional")]
    AgentTurnProvisional {
        provisional_turn_id: String,
        generation: u64,
        text: String,
    },
    #[serde(rename = "agent.close")]
    AgentClose { reason: Option<String> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DebugTextStreamFrame {
    #[serde(rename = "debug.attach")]
    Attach {
        protocol: String,
        extension: String,
        #[serde(default)]
        extensions: Vec<String>,
        call_id: String,
    },
    #[serde(rename = "debug.attached")]
    Attached {
        protocol: String,
        extension: String,
        #[serde(default)]
        extensions: Vec<String>,
        call_id: String,
    },
    #[serde(rename = "debug.detach")]
    Detach { reason: Option<String> },
    #[serde(rename = "debug.detached")]
    Detached { reason: String },
    #[serde(rename = "debug.error")]
    Error { code: String, message: String },
}

impl DebugTextStreamFrame {
    pub fn attach(call_id: impl Into<String>) -> Self {
        Self::Attach {
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            extension: TEXT_CALL_DEBUG_EXTENSION.to_string(),
            extensions: Vec::new(),
            call_id: call_id.into(),
        }
    }

    pub fn attached(call_id: impl Into<String>) -> Self {
        Self::attached_with_extensions(call_id, Vec::new())
    }

    pub fn attached_with_extensions(call_id: impl Into<String>, extensions: Vec<String>) -> Self {
        Self::Attached {
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            extension: TEXT_CALL_DEBUG_EXTENSION.to_string(),
            extensions,
            call_id: call_id.into(),
        }
    }

    pub fn detached(reason: impl Into<String>) -> Self {
        Self::Detached {
            reason: reason.into(),
        }
    }

    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            code: code.into(),
            message: message.into(),
        }
    }
}

fn default_true() -> bool {
    true
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
            utterance_id: Some("utt-test".to_string()),
            sequence: 1,
            text: "hello".to_string(),
        };
        let encoded = serde_json::to_value(frame).expect("frame serializes");
        assert_eq!(encoded["type"], "caller.turn");
        assert_eq!(encoded["turn_id"], "turn-test");
        assert_eq!(encoded["utterance_id"], "utt-test");
        assert_eq!(encoded["sequence"], 1);
    }

    #[test]
    fn serializes_advisory_caller_partial_as_protocol_frame() {
        let frame = GatewayTextFrame::CallerPartial {
            utterance_id: "utt-test".to_string(),
            sequence: 2,
            text: "hello wor".to_string(),
            confidence: None,
            stability: None,
            speech_state: CallerSpeechState::Speaking,
            reply_allowed: false,
        };
        let encoded = serde_json::to_value(frame).expect("frame serializes");
        assert_eq!(encoded["type"], "caller.partial");
        assert_eq!(encoded["utterance_id"], "utt-test");
        assert_eq!(encoded["sequence"], 2);
        assert_eq!(encoded["text"], "hello wor");
        assert!(encoded.get("confidence").is_none());
        assert!(encoded.get("stability").is_none());
        assert_eq!(encoded["speech_state"], "speaking");
        assert_eq!(encoded["reply_allowed"], false);
    }

    #[test]
    fn serializes_advisory_caller_partial_scoring_when_present() {
        let frame = GatewayTextFrame::CallerPartial {
            utterance_id: "utt-test".to_string(),
            sequence: 2,
            text: "hello wor".to_string(),
            confidence: Some(0.82),
            stability: Some(0.67),
            speech_state: CallerSpeechState::EndpointCandidate,
            reply_allowed: false,
        };

        let encoded = serde_json::to_value(&frame).expect("frame serializes");
        assert_eq!(encoded["type"], "caller.partial");
        let confidence = encoded["confidence"].as_f64().expect("numeric confidence");
        let stability = encoded["stability"].as_f64().expect("numeric stability");
        assert!((confidence - 0.82).abs() < 0.000_001);
        assert!((stability - 0.67).abs() < 0.000_001);

        let decoded: GatewayTextFrame =
            serde_json::from_value(encoded).expect("frame deserializes");
        assert_eq!(decoded, frame);
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
                batch_id: None,
                epoch: None,
            }
        );
    }

    #[test]
    fn app_agent_turn_partial_deserializes_with_append_default() {
        let frame: AgentTextFrame = serde_json::from_str(
            r#"{"type":"agent.turn.partial","turn_id":"turn-test","text":"hello"}"#,
        )
        .expect("partial frame deserializes");
        assert_eq!(
            frame,
            AgentTextFrame::AgentTurnPartial {
                turn_id: "turn-test".to_string(),
                text: "hello".to_string(),
                batch_id: None,
                epoch: None,
                append: true,
            }
        );
    }

    #[test]
    fn agent_turn_can_carry_turn_batch_epoch() {
        let frame = AgentTextFrame::AgentTurn {
            turn_id: "turn-test".to_string(),
            text: "reply".to_string(),
            batch_id: Some("turn-batch-0-0".to_string()),
            epoch: Some(0),
        };
        let encoded = serde_json::to_value(&frame).expect("frame serializes");
        assert_eq!(encoded["type"], "agent.turn");
        assert_eq!(encoded["batch_id"], "turn-batch-0-0");
        assert_eq!(encoded["epoch"], 0);
        let decoded: AgentTextFrame = serde_json::from_value(encoded).expect("frame deserializes");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn playback_finished_serializes_terminal_status() {
        let frame = GatewayTextFrame::PlaybackFinished {
            turn_id: "turn-test".to_string(),
            sequence: 3,
            status: PlaybackFinishedStatus::Canceled,
        };
        let encoded = serde_json::to_value(frame).expect("frame serializes");
        assert_eq!(encoded["type"], "playback.finished");
        assert_eq!(encoded["turn_id"], "turn-test");
        assert_eq!(encoded["sequence"], 3);
        assert_eq!(encoded["status"], "canceled");
    }

    #[test]
    fn turn_superseded_serializes_as_control_frame() {
        let frame = GatewayTextFrame::TurnSuperseded {
            turn_id: "turn-old".to_string(),
            superseded_by_turn_id: "turn-new".to_string(),
            reason: "new_caller_turn".to_string(),
            sequence: 7,
        };
        let encoded = serde_json::to_value(frame).expect("frame serializes");
        assert_eq!(encoded["type"], "turn.superseded");
        assert_eq!(encoded["turn_id"], "turn-old");
        assert_eq!(encoded["superseded_by_turn_id"], "turn-new");
        assert_eq!(encoded["reason"], "new_caller_turn");
        assert_eq!(encoded["sequence"], 7);
    }

    #[test]
    fn session_start_carries_response_mode_default() {
        let frame: GatewayTextFrame = serde_json::from_str(
            r#"{"type":"session.start","protocol":"motlie.telnyx.text.v1","call_id":"call-1","direction":"inbound"}"#,
        )
        .expect("session start deserializes");
        assert_eq!(
            frame,
            GatewayTextFrame::SessionStart {
                protocol: TEXT_CALL_PROTOCOL.to_string(),
                call_id: "call-1".to_string(),
                direction: TextCallDirection::Inbound,
                response_mode: ResponseMode::PerTurn,
            }
        );
    }

    #[test]
    fn accept_response_uses_response_mode_contract() {
        let response = AcceptCallResponse {
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            call_url: "ws://127.0.0.1/text".to_string(),
            accept: true,
            extensions: Vec::new(),
            response_mode: ResponseMode::TurnBatched,
        };
        let encoded = serde_json::to_value(response).expect("response serializes");
        assert_eq!(encoded["response_mode"], "turn_batched");
        assert!(encoded.get("aggregation").is_none());
    }

    #[test]
    fn turn_batch_reset_serializes_as_ordered_gateway_frame() {
        let frame = GatewayTextFrame::TurnBatchReset {
            reason: "barge_in".to_string(),
            batch_id: Some("turn-batch-0-0".to_string()),
            epoch: 1,
            sequence: 8,
        };
        let encoded = serde_json::to_value(frame).expect("frame serializes");
        assert_eq!(encoded["type"], "turn_batch.reset");
        assert_eq!(encoded["reason"], "barge_in");
        assert_eq!(encoded["batch_id"], "turn-batch-0-0");
        assert_eq!(encoded["epoch"], 1);
        assert_eq!(encoded["sequence"], 8);
    }

    #[test]
    fn debug_attach_frame_carries_public_protocol_and_extension() {
        let encoded = serde_json::to_value(DebugTextStreamFrame::attach("call-1"))
            .expect("debug frame serializes");
        assert_eq!(encoded["type"], "debug.attach");
        assert_eq!(encoded["protocol"], TEXT_CALL_PROTOCOL);
        assert_eq!(encoded["extension"], TEXT_CALL_DEBUG_EXTENSION);
        assert_eq!(encoded["extensions"].as_array().expect("array").len(), 0);
        assert_eq!(encoded["call_id"], "call-1");
    }
}
