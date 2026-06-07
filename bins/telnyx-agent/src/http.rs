use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::body::Bytes;
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use hmac::{Hmac, Mac};
use motlie_telnyx_gateway::text_calls::turns::{
    AcceptCallResponse, CallConnectedPayload, CallOfferPayload, TEXT_CALL_PROTOCOL,
};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::config::DaemonArgs;
use crate::gateway_client::GatewayClient;
use crate::socket;
use crate::text_ws;
use crate::tmux_bridge::{KeystrokeInjectionConfig, TmuxBridge};

type HmacSha256 = Hmac<Sha256>;

const CALLBACK_MAX_SKEW_SECS: i64 = 300;
const HEADER_CALLBACK_ID: &str = "X-Motlie-Callback-Id";
const HEADER_TIMESTAMP: &str = "X-Motlie-Timestamp";
const HEADER_SIGNATURE: &str = "X-Motlie-Signature";
const SUBSCRIPTION_ID_SALT: &str = "motlie-telnyx-agent-subscription-v1";

#[derive(Clone)]
pub struct AgentState {
    pub public_url: String,
    pub gateway: GatewayClient,
    pub bridge: TmuxBridge,
    pub outbound_timeout_ms: u64,
    pub callback_secret_ref: String,
    callback_security: CallbackSecurity,
}

#[derive(Clone)]
struct CallbackSecurity {
    secret: Arc<[u8]>,
    replay: CallbackReplayCache,
}

#[derive(Clone, Default)]
struct CallbackReplayCache {
    seen: Arc<Mutex<BTreeMap<String, DateTime<Utc>>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallbackAuthError {
    MissingHeader,
    MalformedHeader,
    InvalidTimestamp,
    StaleTimestamp,
    BadSignature,
    DuplicateCallbackId,
}

pub async fn run_daemon(args: DaemonArgs) -> anyhow::Result<()> {
    let gateway = GatewayClient::new(args.gateway_url.clone(), args.gateway_token.clone());
    let public_url = args.public_url.trim_end_matches('/').to_string();
    let callback_secret_ref = args.callback_secret_ref.clone();
    let callback_security = CallbackSecurity::from_secret_ref(&callback_secret_ref)?;
    let bridge = TmuxBridge::new_with_config(
        &args.tmux_target,
        Duration::from_millis(args.reply_timeout_ms),
        KeystrokeInjectionConfig::new(
            Duration::from_millis(args.input_quiet_for_ms),
            Duration::from_millis(args.input_backoff_initial_ms),
            Duration::from_millis(args.input_backoff_max_ms),
            !args.no_trailing_enter,
            Duration::from_millis(args.trailing_enter_delay_ms),
        ),
    )
    .await?;
    let state = AgentState {
        public_url: public_url.clone(),
        gateway: gateway.clone(),
        bridge,
        outbound_timeout_ms: args.outbound_timeout_ms,
        callback_secret_ref: callback_secret_ref.clone(),
        callback_security,
    };

    let inbound_callback = format!("{public_url}/motlie/inbound-offers");
    for routing_value in &args.subscribe_numbers {
        let subscription_id = subscription_id_for_routing_value(routing_value);
        gateway
            .register_subscription(
                subscription_id,
                routing_value,
                &inbound_callback,
                Some(&callback_secret_ref),
            )
            .await?;
    }

    let socket_state = state.clone();
    let socket_path = args.socket.clone();
    tokio::spawn(async move {
        if let Err(error) = socket::run_daemon_socket(socket_path, socket_state).await {
            tracing::warn!(error = %error, "telnyx_agent.socket.failed");
        }
    });

    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    tracing::info!(bind = %args.bind, "telnyx_agent.http.listening");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

fn router(state: AgentState) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/motlie/inbound-offers", post(inbound_offer))
        .route("/motlie/outbound-connected", post(outbound_connected))
        .route("/motlie/text-calls/{call_id}", get(text_call_ws))
        .with_state(Arc::new(state))
}

async fn inbound_offer(
    State(state): State<Arc<AgentState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(error) = state.callback_security.verify(&headers, &body).await {
        return callback_auth_error_response(error);
    }
    let payload = match serde_json::from_slice::<CallOfferPayload>(&body) {
        Ok(payload) => payload,
        Err(error) => return callback_payload_error_response(error),
    };
    accept_response(&state, &payload.call.id).into_response()
}

async fn outbound_connected(
    State(state): State<Arc<AgentState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(error) = state.callback_security.verify(&headers, &body).await {
        return callback_auth_error_response(error);
    }
    let payload = match serde_json::from_slice::<CallConnectedPayload>(&body) {
        Ok(payload) => payload,
        Err(error) => return callback_payload_error_response(error),
    };
    accept_response(&state, &payload.call.id).into_response()
}

fn accept_response(state: &AgentState, call_id: &str) -> (StatusCode, Json<AcceptCallResponse>) {
    let response = AcceptCallResponse {
        protocol: TEXT_CALL_PROTOCOL.to_string(),
        call_url: format!("{}/motlie/text-calls/{call_id}", state.public_url)
            .replace("https://", "wss://")
            .replace("http://", "ws://"),
        accept: true,
    };
    (StatusCode::OK, Json(response))
}

async fn text_call_ws(
    State(state): State<Arc<AgentState>>,
    Path(_call_id): Path<String>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let bridge = state.bridge.clone();
    ws.on_upgrade(move |socket| text_ws::handle_gateway_socket(socket, bridge))
}

impl CallbackSecurity {
    fn from_secret_ref(secret_ref: &str) -> anyhow::Result<Self> {
        let secret = resolve_callback_secret_ref(secret_ref).with_context(|| {
            format!(
                "resolve callback secret ref {}",
                describe_secret_ref(secret_ref)
            )
        })?;
        Ok(Self::new(secret))
    }

    fn new(secret: Vec<u8>) -> Self {
        Self {
            secret: Arc::from(secret.into_boxed_slice()),
            replay: CallbackReplayCache::default(),
        }
    }

    async fn verify(&self, headers: &HeaderMap, raw_body: &[u8]) -> Result<(), CallbackAuthError> {
        let callback_id = required_header(headers, HEADER_CALLBACK_ID)?;
        if callback_id.trim().is_empty() {
            return Err(CallbackAuthError::MalformedHeader);
        }
        let timestamp = required_header(headers, HEADER_TIMESTAMP)?;
        let parsed_timestamp = DateTime::parse_from_rfc3339(timestamp)
            .map_err(|_| CallbackAuthError::InvalidTimestamp)?
            .with_timezone(&Utc);
        reject_stale_timestamp(parsed_timestamp)?;
        verify_callback_signature(
            timestamp,
            raw_body,
            self.secret.as_ref(),
            required_header(headers, HEADER_SIGNATURE)?,
        )?;
        self.replay.insert_once(callback_id).await
    }
}

impl CallbackReplayCache {
    async fn insert_once(&self, callback_id: &str) -> Result<(), CallbackAuthError> {
        let now = Utc::now();
        let cutoff = now - ChronoDuration::seconds(CALLBACK_MAX_SKEW_SECS);
        let mut seen = self.seen.lock().await;
        seen.retain(|_, first_seen| *first_seen >= cutoff);
        if seen.contains_key(callback_id) {
            return Err(CallbackAuthError::DuplicateCallbackId);
        }
        seen.insert(callback_id.to_string(), now);
        Ok(())
    }
}

impl CallbackAuthError {
    fn status(self) -> StatusCode {
        match self {
            Self::DuplicateCallbackId => StatusCode::CONFLICT,
            Self::MissingHeader
            | Self::MalformedHeader
            | Self::InvalidTimestamp
            | Self::StaleTimestamp
            | Self::BadSignature => StatusCode::UNAUTHORIZED,
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::MissingHeader => "missing_header",
            Self::MalformedHeader => "malformed_header",
            Self::InvalidTimestamp => "invalid_timestamp",
            Self::StaleTimestamp => "stale_timestamp",
            Self::BadSignature => "bad_signature",
            Self::DuplicateCallbackId => "duplicate_callback_id",
        }
    }
}

fn required_header<'a>(
    headers: &'a HeaderMap,
    name: &'static str,
) -> Result<&'a str, CallbackAuthError> {
    headers
        .get(name)
        .ok_or(CallbackAuthError::MissingHeader)?
        .to_str()
        .map_err(|_| CallbackAuthError::MalformedHeader)
}

fn reject_stale_timestamp(timestamp: DateTime<Utc>) -> Result<(), CallbackAuthError> {
    let age = Utc::now()
        .signed_duration_since(timestamp)
        .num_seconds()
        .abs();
    if age > CALLBACK_MAX_SKEW_SECS {
        return Err(CallbackAuthError::StaleTimestamp);
    }
    Ok(())
}

fn verify_callback_signature(
    timestamp: &str,
    raw_body: &[u8],
    secret: &[u8],
    signature: &str,
) -> Result<(), CallbackAuthError> {
    let signature = signature
        .strip_prefix("v1=")
        .ok_or(CallbackAuthError::BadSignature)?;
    let signature = hex::decode(signature).map_err(|_| CallbackAuthError::BadSignature)?;
    let mut mac =
        HmacSha256::new_from_slice(secret).map_err(|_| CallbackAuthError::BadSignature)?;
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(raw_body);
    mac.verify_slice(&signature)
        .map_err(|_| CallbackAuthError::BadSignature)
}

fn resolve_callback_secret_ref(secret_ref: &str) -> anyhow::Result<Vec<u8>> {
    let trimmed = secret_ref.trim();
    if let Some(env_name) = trimmed.strip_prefix("env:") {
        let value = std::env::var(env_name)
            .with_context(|| format!("read environment variable {env_name}"))?;
        if value.is_empty() {
            anyhow::bail!("environment variable {env_name} is empty");
        }
        return Ok(value.into_bytes());
    }
    anyhow::bail!(
        "unsupported callback secret ref {}; use env:<name>",
        describe_secret_ref(secret_ref)
    )
}

fn describe_secret_ref(secret_ref: &str) -> String {
    let trimmed = secret_ref.trim();
    if let Some(env_name) = trimmed.strip_prefix("env:") {
        format!("env:{env_name}")
    } else {
        "<unsupported>".to_string()
    }
}

fn callback_auth_error_response(error: CallbackAuthError) -> Response {
    tracing::warn!(reason = error.code(), "telnyx_agent.callback.rejected");
    (
        error.status(),
        Json(serde_json::json!({
            "error": "callback_rejected",
            "reason": error.code(),
        })),
    )
        .into_response()
}

fn callback_payload_error_response(error: serde_json::Error) -> Response {
    tracing::warn!(error = %error, "telnyx_agent.callback.invalid_payload");
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": "invalid_callback_payload"})),
    )
        .into_response()
}

fn subscription_id_for_routing_value(routing_value: &str) -> String {
    let normalized = routing_value.trim().to_ascii_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(SUBSCRIPTION_ID_SALT.as_bytes());
    hasher.update(b"\0");
    hasher.update(normalized.as_bytes());
    let digest = hex::encode(hasher.finalize());
    format!("telnyx-agent-sub_{}", &digest[..24])
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use motlie_telnyx_gateway::text_calls::turns::{TextCallDirection, TextCallInfo};

    const TEST_SECRET: &[u8] = b"callback-test-secret";

    #[tokio::test]
    async fn callback_security_accepts_valid_signature() {
        let security = CallbackSecurity::new(TEST_SECRET.to_vec());
        let body = offer_body();
        let timestamp = Utc::now().to_rfc3339();
        let headers = signed_headers("cb-valid", &timestamp, &body, TEST_SECRET);

        security
            .verify(&headers, &body)
            .await
            .expect("valid callback should verify");
    }

    #[tokio::test]
    async fn callback_security_rejects_bad_or_missing_signature() {
        let security = CallbackSecurity::new(TEST_SECRET.to_vec());
        let body = offer_body();
        let timestamp = Utc::now().to_rfc3339();

        let bad_headers = signed_headers("cb-bad", &timestamp, &body, b"wrong-secret");
        let error = security
            .verify(&bad_headers, &body)
            .await
            .expect_err("bad signature should fail");
        assert_eq!(error, CallbackAuthError::BadSignature);

        let mut missing_headers = signed_headers("cb-missing", &timestamp, &body, TEST_SECRET);
        missing_headers.remove(HEADER_SIGNATURE);
        let error = security
            .verify(&missing_headers, &body)
            .await
            .expect_err("missing signature should fail");
        assert_eq!(error, CallbackAuthError::MissingHeader);
    }

    #[tokio::test]
    async fn callback_security_rejects_stale_timestamp() {
        let security = CallbackSecurity::new(TEST_SECRET.to_vec());
        let body = offer_body();
        let timestamp =
            (Utc::now() - ChronoDuration::seconds(CALLBACK_MAX_SKEW_SECS + 1)).to_rfc3339();
        let headers = signed_headers("cb-stale", &timestamp, &body, TEST_SECRET);

        let error = security
            .verify(&headers, &body)
            .await
            .expect_err("stale timestamp should fail");
        assert_eq!(error, CallbackAuthError::StaleTimestamp);
    }

    #[tokio::test]
    async fn callback_security_rejects_duplicate_callback_id() {
        let security = CallbackSecurity::new(TEST_SECRET.to_vec());
        let body = offer_body();
        let timestamp = Utc::now().to_rfc3339();
        let headers = signed_headers("cb-duplicate", &timestamp, &body, TEST_SECRET);

        security
            .verify(&headers, &body)
            .await
            .expect("first callback id use should verify");
        let error = security
            .verify(&headers, &body)
            .await
            .expect_err("duplicate callback id should fail");
        assert_eq!(error, CallbackAuthError::DuplicateCallbackId);
    }

    #[test]
    fn subscription_id_for_routing_value_is_opaque_for_ids_and_logs() {
        let routing_value = "<called-phone-number>";
        let id = subscription_id_for_routing_value(routing_value);

        assert_eq!(
            id,
            subscription_id_for_routing_value("  <CALLED-PHONE-NUMBER>  ")
        );
        assert!(id.starts_with("telnyx-agent-sub_"));
        assert!(!id.contains(routing_value));
        assert!(!id.contains("called"));
        assert!(!id.contains("phone"));
        assert_eq!(id.len(), "telnyx-agent-sub_".len() + 24);
    }

    fn offer_body() -> Vec<u8> {
        let payload = CallOfferPayload::new(
            "offer-test".to_string(),
            TextCallInfo {
                id: "call-test".to_string(),
                direction: TextCallDirection::Inbound,
                from: Some("<caller-phone-number>".to_string()),
                to: Some("<called-phone-number>".to_string()),
            },
        );
        serde_json::to_vec(&payload).expect("serialize callback payload")
    }

    fn signed_headers(callback_id: &str, timestamp: &str, body: &[u8], secret: &[u8]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_CALLBACK_ID,
            HeaderValue::from_str(callback_id).expect("callback id header"),
        );
        headers.insert(
            HEADER_TIMESTAMP,
            HeaderValue::from_str(timestamp).expect("timestamp header"),
        );
        headers.insert(
            HEADER_SIGNATURE,
            HeaderValue::from_str(&sign_callback(timestamp, body, secret))
                .expect("signature header"),
        );
        headers
    }

    fn sign_callback(timestamp: &str, body: &[u8], secret: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret).expect("test signer");
        mac.update(timestamp.as_bytes());
        mac.update(b".");
        mac.update(body);
        format!("v1={}", hex::encode(mac.finalize().into_bytes()))
    }
}
