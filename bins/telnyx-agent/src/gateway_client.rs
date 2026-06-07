use anyhow::Context;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct GatewayClient {
    http: reqwest::Client,
    gateway_url: String,
    token: Option<String>,
}

#[derive(Debug, Serialize)]
struct SubscriptionRequest<'a> {
    subscription_id: String,
    phone_number: &'a str,
    callback_url: &'a str,
    priority: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    secret_ref: Option<&'a str>,
    enabled: bool,
}

#[derive(Debug, Serialize)]
struct OutboundCallRequest<'a> {
    to: &'a str,
    callback_url: &'a str,
    timeout_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    secret_ref: Option<&'a str>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OutboundCallResponse {
    pub call_id: String,
    pub state: String,
}

impl GatewayClient {
    pub fn new(gateway_url: String, token: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            gateway_url: gateway_url.trim_end_matches('/').to_string(),
            token,
        }
    }

    pub async fn register_subscription(
        &self,
        subscription_id: String,
        phone_number: &str,
        callback_url: &str,
        secret_ref: Option<&str>,
    ) -> anyhow::Result<()> {
        let request = SubscriptionRequest {
            subscription_id,
            phone_number,
            callback_url,
            priority: 100,
            secret_ref,
            enabled: true,
        };
        let response = self
            .request(reqwest::Method::POST, "/api/v1/inbound-subscriptions")
            .json(&request)
            .send()
            .await
            .context("register inbound subscription")?;
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("gateway subscription registration failed with {status}: {body}")
        }
    }

    pub async fn place_outbound_call(
        &self,
        to: &str,
        callback_url: &str,
        timeout_ms: u64,
        secret_ref: Option<&str>,
    ) -> anyhow::Result<OutboundCallResponse> {
        let request = OutboundCallRequest {
            to,
            callback_url,
            timeout_ms,
            secret_ref,
        };
        let response = self
            .request(reqwest::Method::POST, "/api/v1/outbound-calls")
            .json(&request)
            .send()
            .await
            .context("place outbound call")?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("gateway outbound call failed with {status}: {body}");
        }
        serde_json::from_str(&body)
            .with_context(|| format!("decode outbound call response: {body}"))
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let builder = self
            .http
            .request(method, format!("{}{}", self.gateway_url, path));
        if let Some(token) = self.token.as_deref() {
            builder.bearer_auth(token)
        } else {
            builder
        }
    }
}
