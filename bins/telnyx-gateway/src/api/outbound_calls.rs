use std::collections::BTreeMap;
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::time::{self, Instant};

use crate::call_control::DialRequest;
use crate::operator::state::{CallStatus, TelnyxIds};
use crate::serve::AppServices;
use crate::text_calls::offers::{
    callback_http_client, send_outbound_connected_callback, validate_callback_url, CallbackDecision,
};
use crate::text_calls::turns::{TextCallDirection, TextCallInfo, TEXT_CALL_PROTOCOL};
use crate::text_calls::websocket::{connect_application_stream, TextCallSetup};

use super::auth::require_control_api_auth;
use super::{ApiError, ApiResult};

#[derive(Debug, Deserialize)]
pub struct OutboundCallRequest {
    pub to: String,
    pub from: Option<String>,
    pub callback_url: String,
    pub timeout_ms: Option<u64>,
    pub secret_ref: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct OutboundCallResponse {
    pub call_id: String,
    pub state: String,
    pub protocol: String,
    pub text_stream: TextStreamState,
}

#[derive(Debug, Serialize)]
pub struct TextStreamState {
    pub state: String,
}

pub async fn post_outbound_call(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Json(request): Json<OutboundCallRequest>,
) -> ApiResult<(StatusCode, Json<OutboundCallResponse>)> {
    require_control_api_auth(&headers)?;
    validate_request(&request)?;
    let timeout = Duration::from_millis(request.timeout_ms.unwrap_or(45_000));
    let deadline = Instant::now() + timeout;

    let (connection_id, from, stream_url, media) =
        {
            let guard = services.state.read().await;
            let connection_id =
                guard.config.selected_connection_id.clone().ok_or_else(|| {
                    ApiError::bad_request("selected Telnyx connection is required")
                })?;
            let from = request
                .from
                .clone()
                .or_else(|| guard.config.default_from_number.clone())
                .ok_or_else(|| ApiError::bad_request("from is required"))?;
            let stream_url =
                guard.config.public_media_url.clone().ok_or_else(|| {
                    ApiError::bad_request("public media WebSocket URL is required")
                })?;
            (connection_id, from, stream_url, guard.config.telnyx_media)
        };

    let dialed = services
        .telnyx
        .dial_call(&DialRequest {
            connection_id: &connection_id,
            to: &request.to,
            from: &from,
            stream_url: &stream_url,
            webhook_url: None,
            media,
        })
        .await
        .map_err(|error| ApiError::conflict("telnyx_dial_failed", format!("{error:#}")))?;

    let gateway_call_id = {
        let mut guard = services.state.write().await;
        let gateway_call_id = guard.add_or_update_outbound_call(
            TelnyxIds {
                call_control_id: dialed.call_control_id.clone(),
                call_session_id: dialed.call_session_id.clone(),
                call_leg_id: dialed.call_leg_id.clone(),
                stream_id: None,
            },
            Some(from.clone()),
            Some(request.to.clone()),
            if services.telnyx.dry_run() {
                CallStatus::Answered
            } else {
                CallStatus::Dialing
            },
        );
        if let Some(call) = guard.calls.get_mut(&gateway_call_id) {
            call.push_timeline("outbound text-call dial requested");
        }
        gateway_call_id
    };

    wait_for_outbound_answer(&services, &gateway_call_id, deadline).await?;

    let client = callback_http_client()
        .map_err(|error| ApiError::internal(format!("build callback client: {error:#}")))?;
    let call = TextCallInfo {
        id: gateway_call_id.clone(),
        direction: TextCallDirection::Outbound,
        from: Some(from),
        to: Some(request.to),
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    let decision = send_outbound_connected_callback(
        &client,
        &request.callback_url,
        request.secret_ref.as_deref(),
        call,
        remaining.max(Duration::from_secs(1)),
    )
    .await;

    let call_url = match decision {
        CallbackDecision::Accept { call_url } => call_url,
        CallbackDecision::Decline => {
            let _ = hangup_outbound(&services, &gateway_call_id).await;
            return Err(ApiError::conflict(
                "callback_declined",
                "outbound connected callback declined",
            ));
        }
        CallbackDecision::Failed { reason } => {
            let _ = hangup_outbound(&services, &gateway_call_id).await;
            return Err(ApiError::conflict("callback_failed", reason));
        }
    };

    connect_application_stream(
        services.text_call_services(),
        TextCallSetup {
            gateway_call_id: gateway_call_id.clone(),
            call_url,
            direction: TextCallDirection::Outbound,
        },
    )
    .await
    .map_err(|error| {
        ApiError::conflict(
            "text_stream_failed",
            format!("connect text stream: {error:#}"),
        )
    })?;

    Ok((
        StatusCode::CREATED,
        Json(OutboundCallResponse {
            call_id: gateway_call_id,
            state: "connected".to_string(),
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            text_stream: TextStreamState {
                state: "connected".to_string(),
            },
        }),
    ))
}

fn validate_request(request: &OutboundCallRequest) -> ApiResult<()> {
    if request.to.trim().is_empty() {
        return Err(ApiError::bad_request("to is required"));
    }
    validate_callback_url(&request.callback_url)
        .map_err(|error| ApiError::bad_request(format!("invalid callback_url: {error:#}")))?;
    Ok(())
}

async fn wait_for_outbound_answer(
    services: &AppServices,
    gateway_call_id: &str,
    deadline: Instant,
) -> ApiResult<()> {
    loop {
        let status = {
            let guard = services.state.read().await;
            guard.calls.get(gateway_call_id).map(|call| call.status)
        };
        match status {
            Some(
                CallStatus::Answered
                | CallStatus::MediaStarted
                | CallStatus::Transcribing
                | CallStatus::Speaking,
            ) => return Ok(()),
            Some(CallStatus::Ended) => {
                return Err(ApiError::conflict(
                    "call_rejected",
                    "outbound call ended before text stream setup",
                ))
            }
            Some(CallStatus::Failed) => {
                return Err(ApiError::conflict(
                    "call_failed",
                    "outbound call failed before text stream setup",
                ))
            }
            Some(_) if Instant::now() < deadline => time::sleep(Duration::from_millis(100)).await,
            Some(_) | None => {
                return Err(ApiError::gateway_timeout(
                    "outbound call did not answer before timeout",
                ))
            }
        }
    }
}

async fn hangup_outbound(services: &AppServices, gateway_call_id: &str) -> anyhow::Result<()> {
    let call_control_id = {
        let guard = services.state.read().await;
        guard
            .calls
            .get(gateway_call_id)
            .map(|call| call.ids.call_control_id.clone())
    };
    if let Some(call_control_id) = call_control_id {
        services.telnyx.hangup_call(&call_control_id).await?;
    }
    Ok(())
}
