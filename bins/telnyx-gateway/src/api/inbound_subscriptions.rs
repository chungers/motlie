use std::collections::BTreeMap;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::operator::state::{InboundSubscription, UpsertInboundSubscription};
use crate::serve::AppServices;
use crate::text_calls::offers::validate_callback_url;

use super::auth::require_control_api_auth;
use super::{api_ok, ApiError, ApiResult};

#[derive(Debug, Deserialize)]
pub struct UpsertInboundSubscriptionRequest {
    pub subscription_id: Option<String>,
    pub phone_number: String,
    pub callback_url: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    pub secret_ref: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct ListInboundSubscriptionsQuery {
    pub phone_number: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct InboundSubscriptionResponse {
    pub subscription: InboundSubscription,
}

#[derive(Debug, Serialize)]
pub struct InboundSubscriptionListResponse {
    pub subscriptions: Vec<InboundSubscription>,
}

#[derive(Debug, Serialize)]
pub struct TestSubscriptionResponse {
    pub subscription_id: String,
    pub callback_url: String,
    pub valid: bool,
}

pub async fn upsert_subscription(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Json(request): Json<UpsertInboundSubscriptionRequest>,
) -> ApiResult<(StatusCode, Json<InboundSubscriptionResponse>)> {
    require_control_api_auth(&headers)?;
    validate_subscription_request(&request)?;
    let id = request
        .subscription_id
        .clone()
        .unwrap_or_else(|| format!("sub_{}", Uuid::new_v4().simple()));
    let mut guard = services.state.write().await;
    let existed = guard.inbound_subscription(&id).is_some();
    let subscription = guard.upsert_inbound_subscription(UpsertInboundSubscription {
        id,
        phone_number: request.phone_number,
        callback_url: request.callback_url,
        priority: request.priority,
        secret_ref: request.secret_ref,
        enabled: request.enabled,
        metadata: request.metadata,
    });
    let status = if existed {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((status, Json(InboundSubscriptionResponse { subscription })))
}

pub async fn list_subscriptions(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Query(query): Query<ListInboundSubscriptionsQuery>,
) -> ApiResult<Json<InboundSubscriptionListResponse>> {
    require_control_api_auth(&headers)?;
    let guard = services.state.read().await;
    let subscriptions = if let Some(phone_number) = query.phone_number.as_deref() {
        guard.inbound_subscriptions_for_phone(phone_number)
    } else {
        guard.inbound_subscriptions.values().cloned().collect()
    };
    Ok(Json(InboundSubscriptionListResponse { subscriptions }))
}

pub async fn get_subscription(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Path(subscription_id): Path<String>,
) -> ApiResult<Json<InboundSubscriptionResponse>> {
    require_control_api_auth(&headers)?;
    let guard = services.state.read().await;
    let Some(subscription) = guard.inbound_subscription(&subscription_id).cloned() else {
        return Err(ApiError::not_found("inbound subscription not found"));
    };
    Ok(Json(InboundSubscriptionResponse { subscription }))
}

pub async fn delete_subscription(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Path(subscription_id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    require_control_api_auth(&headers)?;
    let mut guard = services.state.write().await;
    if guard
        .remove_inbound_subscription(&subscription_id)
        .is_none()
    {
        return Err(ApiError::not_found("inbound subscription not found"));
    }
    Ok(api_ok())
}

pub async fn test_subscription(
    State(services): State<AppServices>,
    headers: HeaderMap,
    Path(subscription_id): Path<String>,
) -> ApiResult<Json<TestSubscriptionResponse>> {
    require_control_api_auth(&headers)?;
    let guard = services.state.read().await;
    let Some(subscription) = guard.inbound_subscription(&subscription_id).cloned() else {
        return Err(ApiError::not_found("inbound subscription not found"));
    };
    validate_callback_url(&subscription.callback_url)
        .map_err(|error| ApiError::bad_request(format!("invalid callback_url: {error:#}")))?;
    Ok(Json(TestSubscriptionResponse {
        subscription_id,
        callback_url: subscription.callback_url,
        valid: true,
    }))
}

fn validate_subscription_request(request: &UpsertInboundSubscriptionRequest) -> ApiResult<()> {
    if request.phone_number.trim().is_empty() {
        return Err(ApiError::bad_request("phone_number is required"));
    }
    validate_callback_url(&request.callback_url)
        .map_err(|error| ApiError::bad_request(format!("invalid callback_url: {error:#}")))?;
    Ok(())
}

fn default_priority() -> i32 {
    100
}

fn default_enabled() -> bool {
    true
}
