use serde::Deserialize;

use crate::operator::state::{CallStatus, InboundMode, LogLevel, SharedState, TelnyxIds};

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
            update_call_status(state, &envelope.data.payload, CallStatus::Failed, &message).await;
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
    let status = if is_inbound && guard.inbound_mode != InboundMode::Disabled {
        CallStatus::PendingInbound
    } else {
        CallStatus::IgnoredInbound
    };
    let gateway_call_id = guard.add_or_update_inbound_call(
        TelnyxIds {
            call_control_id: call_control_id.clone(),
            call_session_id: payload.call_session_id.clone(),
            call_leg_id: payload.call_leg_id.clone(),
            stream_id: None,
        },
        payload.from.clone(),
        payload.to.clone(),
        status,
    );

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
        call.push_timeline(message);
        tracing::info!(
            gateway_call_id = call.gateway_call_id,
            call_control_id,
            call_session_id = call.ids.call_session_id.as_deref(),
            call_leg_id = call.ids.call_leg_id.as_deref(),
            status = status.label(),
            "{message}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::{shared_state, InboundMode};

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
}
