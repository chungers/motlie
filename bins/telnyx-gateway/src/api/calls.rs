use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::operator::state::{CallDirection, CallSession, CallStatus};
use crate::serve::AppServices;

use super::auth::require_control_api_auth;
use super::{ApiError, ApiResult};

#[derive(Debug, Serialize)]
pub struct CallSummary {
    pub call_id: String,
    pub direction: &'static str,
    pub status: &'static str,
    pub from_present: bool,
    pub to_present: bool,
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize)]
pub struct CallListResponse {
    pub calls: Vec<CallSummary>,
}

#[derive(Debug, Serialize)]
pub struct CallDetailResponse {
    pub call: CallSummary,
    pub transcript_final_chars: usize,
    pub transcript_events: usize,
}

pub async fn list_calls(
    State(services): State<AppServices>,
    headers: HeaderMap,
) -> ApiResult<Json<CallListResponse>> {
    require_control_api_auth(&headers)?;
    let guard = services.state.read().await;
    let calls = guard.calls.values().map(call_summary).collect();
    Ok(Json(CallListResponse { calls }))
}

pub async fn get_call(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Path(call_id): Path<String>,
) -> ApiResult<Json<CallDetailResponse>> {
    require_control_api_auth(&headers)?;
    let guard = services.state.read().await;
    let Some(call) = guard.calls.get(&call_id) else {
        return Err(ApiError::not_found("call not found"));
    };
    Ok(Json(CallDetailResponse {
        call: call_summary(call),
        transcript_final_chars: call.final_transcript.chars().count(),
        transcript_events: call.transcripts.len(),
    }))
}

fn call_summary(call: &CallSession) -> CallSummary {
    CallSummary {
        call_id: call.gateway_call_id.clone(),
        direction: direction_label(call.direction),
        status: status_label(call.status),
        from_present: call
            .from
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty()),
        to_present: call
            .to
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty()),
        updated_at: call.updated_at(),
    }
}

fn direction_label(direction: CallDirection) -> &'static str {
    match direction {
        CallDirection::Inbound => "inbound",
        CallDirection::Outbound => "outbound",
    }
}

fn status_label(status: CallStatus) -> &'static str {
    status.label()
}
