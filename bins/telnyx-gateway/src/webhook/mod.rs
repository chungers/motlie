use std::collections::BTreeMap;

use serde::Deserialize;

use crate::operator::state::{
    CallDirection, CallStatus, InboundMode, LogLevel, SharedState, TelnyxIds,
};

#[derive(Debug, Deserialize)]
pub struct TelnyxWebhookEnvelope {
    pub data: TelnyxWebhookData,
}

#[derive(Debug, Deserialize)]
pub struct TelnyxWebhookData {
    pub event_type: String,
    pub id: Option<String>,
    pub occurred_at: Option<String>,
    #[serde(default)]
    pub payload: TelnyxWebhookPayload,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TelnyxWebhookPayload {
    pub call_control_id: Option<String>,
    pub call_session_id: Option<String>,
    pub call_leg_id: Option<String>,
    pub connection_id: Option<String>,
    pub direction: Option<String>,
    pub state: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub stream_url: Option<String>,
    pub hangup_cause: Option<String>,
    pub hangup_source: Option<String>,
    pub sip_hangup_cause: Option<String>,
    pub sip_reason: Option<String>,
    pub cause: Option<String>,
    pub reason: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

pub async fn handle_voice_webhook(
    state: SharedState,
    body: serde_json::Value,
) -> anyhow::Result<String> {
    let envelope: TelnyxWebhookEnvelope = serde_json::from_value(body)?;
    let event_type = envelope.data.event_type.as_str();
    match event_type {
        "call.initiated" => handle_call_initiated(state, envelope).await,
        "call.answered" => {
            update_call_status(
                state,
                event_type,
                &envelope.data.payload,
                CallStatus::Answered,
                "call answered",
            )
            .await;
            Ok("ok".to_string())
        }
        "call.hangup" | "call.ended" => {
            update_call_status(
                state,
                event_type,
                &envelope.data.payload,
                CallStatus::Ended,
                "call ended",
            )
            .await;
            Ok("ok".to_string())
        }
        "streaming.started" => {
            update_call_status(
                state,
                event_type,
                &envelope.data.payload,
                CallStatus::MediaStarted,
                "streaming started webhook",
            )
            .await;
            Ok("ok".to_string())
        }
        "streaming.stopped" => {
            update_call_status(
                state,
                event_type,
                &envelope.data.payload,
                CallStatus::Ended,
                "streaming stopped webhook",
            )
            .await;
            Ok("ok".to_string())
        }
        "streaming.failed" => {
            let message = envelope
                .data
                .payload
                .hangup_cause
                .clone()
                .unwrap_or_else(|| "streaming failed".to_string());
            update_call_status(
                state,
                event_type,
                &envelope.data.payload,
                CallStatus::Failed,
                &message,
            )
            .await;
            Ok("ok".to_string())
        }
        other => {
            let mut guard = state.write().await;
            guard.log(LogLevel::Info, format!("ignored Telnyx webhook {other}"));
            Ok("ignored".to_string())
        }
    }
}

async fn handle_call_initiated(
    state: SharedState,
    envelope: TelnyxWebhookEnvelope,
) -> anyhow::Result<String> {
    let payload = envelope.data.payload;
    let Some(call_control_id) = payload.call_control_id.clone() else {
        let mut guard = state.write().await;
        guard.log(LogLevel::Warn, "call.initiated missing call_control_id");
        return Ok("ignored".to_string());
    };
    let is_inbound = payload.direction.as_deref() == Some("incoming")
        || payload.state.as_deref() == Some("parked");

    let mut guard = state.write().await;
    let ids = TelnyxIds {
        call_control_id: call_control_id.clone(),
        call_session_id: payload.call_session_id.clone(),
        call_leg_id: payload.call_leg_id.clone(),
        stream_id: None,
    };
    let (gateway_call_id, status) = if is_inbound {
        let status = if guard.inbound_mode != InboundMode::Disabled {
            CallStatus::PendingInbound
        } else {
            CallStatus::IgnoredInbound
        };
        (
            guard.add_or_update_inbound_call(ids, payload.from.clone(), payload.to.clone(), status),
            status,
        )
    } else {
        (
            guard.add_or_update_outbound_call(
                ids,
                payload.from.clone(),
                payload.to.clone(),
                CallStatus::Dialing,
            ),
            CallStatus::Dialing,
        )
    };

    if status == CallStatus::PendingInbound {
        guard.log(
            LogLevel::Info,
            format!(
                "inbound call pending: {gateway_call_id} from {:?}",
                payload.from
            ),
        );
        tracing::info!(
            gateway_call_id,
            call_control_id,
            call_session_id = payload.call_session_id.as_deref(),
            call_leg_id = payload.call_leg_id.as_deref(),
            event_id = envelope.data.id.as_deref(),
            occurred_at = envelope.data.occurred_at.as_deref(),
            "call.inbound.pending"
        );
    } else if status == CallStatus::Dialing {
        guard.log(
            LogLevel::Info,
            format!(
                "outbound call dialing: {gateway_call_id} to {:?}",
                payload.to
            ),
        );
        tracing::info!(
            gateway_call_id,
            call_control_id,
            call_session_id = payload.call_session_id.as_deref(),
            call_leg_id = payload.call_leg_id.as_deref(),
            event_id = envelope.data.id.as_deref(),
            occurred_at = envelope.data.occurred_at.as_deref(),
            direction = ?CallDirection::Outbound,
            "call.outbound.dialing"
        );
    } else {
        guard.log(
            LogLevel::Info,
            format!("inbound disabled; not answering call {gateway_call_id}"),
        );
        tracing::info!(
            gateway_call_id,
            call_control_id,
            call_session_id = payload.call_session_id.as_deref(),
            call_leg_id = payload.call_leg_id.as_deref(),
            "call.inbound.disabled"
        );
    }

    Ok("ok".to_string())
}

async fn update_call_status(
    state: SharedState,
    event_type: &str,
    payload: &TelnyxWebhookPayload,
    status: CallStatus,
    message: &str,
) {
    let Some(call_control_id) = payload.call_control_id.as_deref() else {
        return;
    };
    let mut guard = state.write().await;
    if let Some(call) = guard.call_by_control_id_mut(call_control_id) {
        call.status = status;
        let terminal_reason = termination_reason(event_type, payload);
        if matches!(status, CallStatus::Ended | CallStatus::Failed) {
            call.terminal_reason = terminal_reason.clone();
        }
        let timeline_message = terminal_reason
            .as_ref()
            .map(|reason| format!("{message}: {reason}"))
            .unwrap_or_else(|| message.to_string());
        call.push_timeline(timeline_message);
        tracing::info!(
            gateway_call_id = call.gateway_call_id,
            call_control_id,
            call_session_id = call.ids.call_session_id.as_deref(),
            call_leg_id = call.ids.call_leg_id.as_deref(),
            status = status.label(),
            telnyx_event = event_type,
            hangup_cause = payload.hangup_cause.as_deref(),
            hangup_source = payload.hangup_source.as_deref(),
            sip_hangup_cause = payload.sip_hangup_cause.as_deref(),
            sip_reason = payload.sip_reason.as_deref(),
            cause = payload.cause.as_deref(),
            reason = payload.reason.as_deref(),
            payload_extra = ?payload.extra,
            terminal_reason = terminal_reason.as_deref(),
            "{message}"
        );
    }
}

fn termination_reason(event_type: &str, payload: &TelnyxWebhookPayload) -> Option<String> {
    let mut parts = vec![format!("event={event_type}")];
    push_detail(&mut parts, "hangup_cause", payload.hangup_cause.as_deref());
    push_detail(
        &mut parts,
        "hangup_source",
        payload.hangup_source.as_deref(),
    );
    push_detail(
        &mut parts,
        "sip_hangup_cause",
        payload.sip_hangup_cause.as_deref(),
    );
    push_detail(&mut parts, "sip_reason", payload.sip_reason.as_deref());
    push_detail(&mut parts, "cause", payload.cause.as_deref());
    push_detail(&mut parts, "reason", payload.reason.as_deref());

    for key in [
        "call_hangup_cause",
        "hangup_code",
        "sip_code",
        "disconnect_reason",
        "termination_reason",
    ] {
        if let Some(value) = payload.extra.get(key).and_then(payload_value_string) {
            parts.push(format!("{key}={value}"));
        }
    }

    if parts.len() == 1 && !matches!(event_type, "call.hangup" | "call.ended") {
        return None;
    }
    Some(parts.join(" "))
}

fn push_detail(parts: &mut Vec<String>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        parts.push(format!("{key}={value}"));
    }
}

fn payload_value_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) if !value.is_empty() => Some(value.clone()),
        serde_json::Value::Number(value) => Some(value.to_string()),
        serde_json::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::{InboundMode, shared_state};

    #[tokio::test]
    async fn inbound_disabled_does_not_create_pending_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let body = serde_json::json!({
            "data": {
                "event_type": "call.initiated",
                "payload": {
                    "direction": "incoming",
                    "state": "parked",
                    "call_control_id": "call-1",
                    "call_session_id": "sess-1",
                    "call_leg_id": "leg-1",
                    "from": "+15550000001",
                    "to": "+15550000002"
                }
            }
        });

        handle_voice_webhook(state.clone(), body)
            .await
            .expect("webhook should be accepted");

        let guard = state.read().await;
        let call = guard
            .calls
            .values()
            .next()
            .expect("call should be recorded");
        assert_eq!(call.status, CallStatus::IgnoredInbound);
        assert_eq!(guard.inbound_mode, InboundMode::Disabled);
    }

    #[tokio::test]
    async fn manual_inbound_creates_pending_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        state.write().await.inbound_mode = InboundMode::Manual;
        let body = serde_json::json!({
            "data": {
                "event_type": "call.initiated",
                "payload": {
                    "direction": "incoming",
                    "state": "parked",
                    "call_control_id": "call-1"
                }
            }
        });

        handle_voice_webhook(state.clone(), body)
            .await
            .expect("webhook should be accepted");

        let guard = state.read().await;
        let call = guard
            .calls
            .values()
            .next()
            .expect("call should be recorded");
        assert_eq!(call.status, CallStatus::PendingInbound);
        assert!(guard.selected_call.is_some());
    }

    #[tokio::test]
    async fn outbound_initiated_creates_dialing_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let body = serde_json::json!({
            "data": {
                "event_type": "call.initiated",
                "payload": {
                    "direction": "outgoing",
                    "call_control_id": "call-1",
                    "call_session_id": "sess-1",
                    "call_leg_id": "leg-1",
                    "from": "+15550000001",
                    "to": "+15550000002"
                }
            }
        });

        handle_voice_webhook(state.clone(), body)
            .await
            .expect("webhook should be accepted");

        let guard = state.read().await;
        let call = guard
            .calls
            .values()
            .next()
            .expect("call should be recorded");
        assert_eq!(call.direction, CallDirection::Outbound);
        assert_eq!(call.status, CallStatus::Dialing);
        assert_eq!(
            guard.selected_call.as_deref(),
            Some(call.gateway_call_id.as_str())
        );
    }

    #[tokio::test]
    async fn call_end_records_telnyx_termination_reason() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        state.write().await.inbound_mode = InboundMode::Manual;
        handle_voice_webhook(
            state.clone(),
            serde_json::json!({
                "data": {
                    "event_type": "call.initiated",
                    "payload": {
                        "direction": "incoming",
                        "state": "parked",
                        "call_control_id": "call-1",
                        "call_session_id": "sess-1",
                        "call_leg_id": "leg-1"
                    }
                }
            }),
        )
        .await
        .expect("call.initiated should be accepted");

        handle_voice_webhook(
            state.clone(),
            serde_json::json!({
                "data": {
                    "event_type": "call.hangup",
                    "payload": {
                        "call_control_id": "call-1",
                        "call_session_id": "sess-1",
                        "call_leg_id": "leg-1",
                        "hangup_cause": "normal_clearing",
                        "hangup_source": "caller",
                        "sip_hangup_cause": "200",
                        "sip_code": 200
                    }
                }
            }),
        )
        .await
        .expect("call.hangup should be accepted");

        let guard = state.read().await;
        let call = guard.calls.values().next().expect("call exists");
        let reason = call
            .terminal_reason
            .as_deref()
            .expect("termination reason should be recorded");
        assert_eq!(call.status, CallStatus::Ended);
        assert!(reason.contains("event=call.hangup"));
        assert!(reason.contains("hangup_cause=normal_clearing"));
        assert!(reason.contains("hangup_source=caller"));
        assert!(reason.contains("sip_hangup_cause=200"));
        assert!(reason.contains("sip_code=200"));
    }
}
