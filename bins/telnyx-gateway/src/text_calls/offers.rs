use std::time::Duration;

use anyhow::{bail, Context};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::{Client, StatusCode};
use serde::Serialize;
use sha2::Sha256;
use uuid::Uuid;

use crate::operator::state::InboundSubscription;

use super::turns::{
    AcceptCallResponse, CallConnectedPayload, CallOfferPayload, TextCallInfo, TEXT_CALL_PROTOCOL,
};

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CallbackDecision {
    Accept { call_url: String },
    Decline,
    Failed { reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OfferAttempt {
    pub subscription_id: String,
    pub decision: CallbackDecision,
}

pub fn callback_http_client() -> anyhow::Result<Client> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build callback HTTP client")
}

pub async fn send_inbound_offer(
    client: &Client,
    subscription: &InboundSubscription,
    call: TextCallInfo,
    timeout: Duration,
) -> OfferAttempt {
    let payload = CallOfferPayload::new(format!("offer_{}", Uuid::new_v4().simple()), call);
    let decision = post_callback(
        client,
        &subscription.callback_url,
        subscription.secret_ref.as_deref(),
        &payload,
        timeout,
    )
    .await;
    OfferAttempt {
        subscription_id: subscription.id.clone(),
        decision,
    }
}

pub async fn send_outbound_connected_callback(
    client: &Client,
    callback_url: &str,
    secret_ref: Option<&str>,
    call: TextCallInfo,
    timeout: Duration,
) -> CallbackDecision {
    let payload = CallConnectedPayload::new(call);
    post_callback(client, callback_url, secret_ref, &payload, timeout).await
}

async fn post_callback<T: Serialize>(
    client: &Client,
    callback_url: &str,
    secret_ref: Option<&str>,
    payload: &T,
    timeout: Duration,
) -> CallbackDecision {
    let raw_body = match serde_json::to_vec(payload) {
        Ok(raw_body) => raw_body,
        Err(error) => {
            return CallbackDecision::Failed {
                reason: format!("encode callback payload: {error}"),
            }
        }
    };
    let mut headers = match callback_headers(secret_ref, &raw_body) {
        Ok(headers) => headers,
        Err(error) => {
            return CallbackDecision::Failed {
                reason: format!("sign callback: {error:#}"),
            }
        }
    };
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let response = client
        .post(callback_url)
        .headers(headers)
        .body(raw_body)
        .timeout(timeout)
        .send()
        .await;

    let response = match response {
        Ok(response) => response,
        Err(error) if error.is_timeout() => {
            return CallbackDecision::Failed {
                reason: "callback timeout".to_string(),
            }
        }
        Err(error) => {
            return CallbackDecision::Failed {
                reason: format!("callback request failed: {error}"),
            }
        }
    };

    classify_callback_response(response).await
}

async fn classify_callback_response(response: reqwest::Response) -> CallbackDecision {
    let status = response.status();
    if status.is_redirection() {
        return CallbackDecision::Decline;
    }
    if status != StatusCode::OK {
        return CallbackDecision::Failed {
            reason: format!("callback returned {status}"),
        };
    }

    let parsed = response.json::<AcceptCallResponse>().await;
    let accepted = match parsed {
        Ok(parsed) => parsed,
        Err(error) => {
            return CallbackDecision::Failed {
                reason: format!("decode accept response: {error}"),
            }
        }
    };
    if accepted.protocol != TEXT_CALL_PROTOCOL {
        return CallbackDecision::Failed {
            reason: "accept response protocol mismatch".to_string(),
        };
    }
    if !accepted.accept {
        return CallbackDecision::Failed {
            reason: "accept response missing accept=true".to_string(),
        };
    }
    match validate_call_url(&accepted.call_url) {
        Ok(()) => CallbackDecision::Accept {
            call_url: accepted.call_url,
        },
        Err(error) => CallbackDecision::Failed {
            reason: format!("invalid call_url: {error:#}"),
        },
    }
}

pub fn validate_callback_url(callback_url: &str) -> anyhow::Result<()> {
    validate_url_scheme(callback_url, &["http", "https"])
}

pub fn validate_call_url(call_url: &str) -> anyhow::Result<()> {
    validate_url_scheme(call_url, &["ws", "wss"])
}

fn validate_url_scheme(value: &str, allowed: &[&str]) -> anyhow::Result<()> {
    let url = reqwest::Url::parse(value).with_context(|| format!("parse URL {value}"))?;
    if allowed.iter().any(|scheme| *scheme == url.scheme()) {
        Ok(())
    } else {
        bail!("scheme must be one of {}", allowed.join(", "))
    }
}

fn callback_headers(secret_ref: Option<&str>, raw_body: &[u8]) -> anyhow::Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let callback_id = format!("cb_{}", Uuid::new_v4().simple());
    let timestamp = Utc::now().to_rfc3339();
    insert_header(&mut headers, "X-Motlie-Callback-Id", &callback_id);
    insert_header(&mut headers, "X-Motlie-Timestamp", &timestamp);
    let secret_ref = secret_ref
        .filter(|value| !value.trim().is_empty())
        .context("callback secret_ref is required")?;
    let secret = resolve_secret_ref(secret_ref).context("callback secret_ref did not resolve")?;
    let signature = sign_callback(&timestamp, raw_body, secret.as_bytes())?;
    insert_header(&mut headers, "X-Motlie-Signature", &signature);
    Ok(headers)
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

fn resolve_secret_ref(secret_ref: &str) -> Option<String> {
    let value = secret_ref.trim();
    if let Some(env_name) = value.strip_prefix("env:") {
        return std::env::var(env_name)
            .ok()
            .filter(|value| !value.is_empty());
    }
    None
}

fn sign_callback(timestamp: &str, raw_body: &[u8], secret: &[u8]) -> anyhow::Result<String> {
    let mut mac = HmacSha256::new_from_slice(secret).context("initialize callback signer")?;
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(raw_body);
    Ok(format!("v1={}", hex::encode(mac.finalize().into_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::{Json, Router};
    use serde_json::json;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn missing_secret_ref_fails_before_posting_callback() {
        let client = callback_http_client().expect("client");
        let decision = post_callback(
            &client,
            "http://127.0.0.1:1/offer",
            None,
            &json!({"type":"call.offer"}),
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(
            decision,
            CallbackDecision::Failed {
                reason: "sign callback: callback secret_ref is required".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn redirect_response_declines_offer_without_following() {
        async fn handler() -> axum::response::Response {
            (
                StatusCode::SEE_OTHER,
                [(
                    axum::http::header::LOCATION,
                    "https://example.test/declined",
                )],
            )
                .into_response()
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/offer", post(handler)))
                .await
                .expect("serve fake callback");
        });

        let secret_ref = install_test_callback_secret("MOTLIE_TEST_CALLBACK_SECRET_DECLINE");
        let client = callback_http_client().expect("client");
        let decision = post_callback(
            &client,
            &format!("http://{addr}/offer"),
            Some(&secret_ref),
            &json!({"type":"call.offer"}),
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(decision, CallbackDecision::Decline);
    }

    #[tokio::test]
    async fn ok_with_websocket_call_url_accepts() {
        async fn handler() -> Json<serde_json::Value> {
            Json(json!({
                "protocol": TEXT_CALL_PROTOCOL,
                "call_url": "ws://127.0.0.1/text-call",
                "accept": true
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/offer", post(handler)))
                .await
                .expect("serve fake callback");
        });

        let secret_ref = install_test_callback_secret("MOTLIE_TEST_CALLBACK_SECRET_ACCEPT");
        let client = callback_http_client().expect("client");
        let decision = post_callback(
            &client,
            &format!("http://{addr}/offer"),
            Some(&secret_ref),
            &json!({"type":"call.offer"}),
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(
            decision,
            CallbackDecision::Accept {
                call_url: "ws://127.0.0.1/text-call".to_string(),
            }
        );
    }

    fn install_test_callback_secret(env_name: &str) -> String {
        std::env::set_var(env_name, "callback-test-secret");
        format!("env:{env_name}")
    }
}
