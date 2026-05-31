use anyhow::{bail, Context};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[derive(Clone)]
pub struct TelnyxClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    dry_run: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CallControlApplication {
    pub id: String,
    pub application_name: Option<String>,
    pub webhook_event_url: Option<String>,
    pub active: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PhoneNumber {
    pub id: String,
    pub phone_number: Option<String>,
    pub connection_id: Option<String>,
    pub connection_name: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiOne<T> {
    data: T,
}

#[derive(Debug, Deserialize)]
struct ApiList<T> {
    data: Vec<T>,
}

impl TelnyxClient {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>, dry_run: bool) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            dry_run,
        }
    }

    pub fn dry_run(&self) -> bool {
        self.dry_run
    }

    pub async fn list_applications(&self) -> anyhow::Result<Vec<CallControlApplication>> {
        self.get_list("call_control_applications").await
    }

    pub async fn create_application(
        &self,
        name: &str,
        webhook_url: &str,
    ) -> anyhow::Result<CallControlApplication> {
        let body = json!({
            "application_name": name,
            "webhook_event_url": webhook_url,
            "webhook_api_version": "2",
            "active": true,
        });
        self.post_one("call_control_applications", &body).await
    }

    pub async fn retrieve_application(
        &self,
        connection_id: &str,
    ) -> anyhow::Result<CallControlApplication> {
        self.get_one(&format!(
            "call_control_applications/{}",
            urlencoding::encode(connection_id)
        ))
        .await
    }

    pub async fn update_application_webhook(
        &self,
        connection_id: &str,
        webhook_url: &str,
    ) -> anyhow::Result<CallControlApplication> {
        let current = self.retrieve_application(connection_id).await?;
        let name = current
            .application_name
            .as_deref()
            .unwrap_or("motlie-gateway");
        let body = json!({
            "application_name": name,
            "webhook_event_url": webhook_url,
            "webhook_api_version": "2",
            "active": current.active.unwrap_or(true),
        });
        self.patch_one(
            &format!(
                "call_control_applications/{}",
                urlencoding::encode(connection_id)
            ),
            &body,
        )
        .await
    }

    pub async fn list_phone_numbers(&self) -> anyhow::Result<Vec<PhoneNumber>> {
        self.get_list("phone_numbers").await
    }

    pub async fn bind_phone_number(
        &self,
        phone_number: &str,
        connection_id: &str,
    ) -> anyhow::Result<PhoneNumber> {
        if self.dry_run {
            return Ok(PhoneNumber {
                id: phone_number.to_string(),
                phone_number: Some(phone_number.to_string()),
                connection_id: Some(connection_id.to_string()),
                connection_name: None,
                status: Some("dry-run".to_string()),
            });
        }
        let phone_number_id = self.resolve_phone_number_id(phone_number).await?;
        let body = json!({ "connection_id": connection_id });
        self.patch_one(
            &format!("phone_numbers/{}", urlencoding::encode(&phone_number_id)),
            &body,
        )
        .await
    }

    pub async fn answer_call(&self, request: &AnswerRequest<'_>) -> anyhow::Result<()> {
        let body = json!({
            "stream_url": request.stream_url,
            "stream_track": "inbound_track",
            "stream_codec": "L16",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "L16",
            "stream_bidirectional_sampling_rate": 16000,
            "stream_bidirectional_target_legs": "self",
            "command_id": format!("motlie-answer-{}", Uuid::new_v4()),
        });
        self.post_command(
            &format!(
                "calls/{}/actions/answer",
                urlencoding::encode(request.call_control_id)
            ),
            &body,
        )
        .await
    }

    pub async fn reject_call(&self, call_control_id: &str) -> anyhow::Result<()> {
        self.post_command(
            &format!(
                "calls/{}/actions/reject",
                urlencoding::encode(call_control_id)
            ),
            &json!({ "command_id": format!("motlie-reject-{}", Uuid::new_v4()) }),
        )
        .await
    }

    pub async fn hangup_call(&self, call_control_id: &str) -> anyhow::Result<()> {
        self.post_command(
            &format!(
                "calls/{}/actions/hangup",
                urlencoding::encode(call_control_id)
            ),
            &json!({ "command_id": format!("motlie-hangup-{}", Uuid::new_v4()) }),
        )
        .await
    }

    async fn get_one<T>(&self, path: &str) -> anyhow::Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        if self.dry_run {
            bail!("dry-run Telnyx client cannot GET {path}");
        }
        let response = self.request(reqwest::Method::GET, path)?.send().await?;
        decode_response::<ApiOne<T>>(response)
            .await
            .map(|body| body.data)
    }

    async fn get_list<T>(&self, path: &str) -> anyhow::Result<Vec<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        if self.dry_run {
            return Ok(Vec::new());
        }
        let response = self.request(reqwest::Method::GET, path)?.send().await?;
        decode_response::<ApiList<T>>(response)
            .await
            .map(|body| body.data)
    }

    async fn post_one<T, B>(&self, path: &str, body: &B) -> anyhow::Result<T>
    where
        T: for<'de> Deserialize<'de>,
        B: Serialize + ?Sized,
    {
        if self.dry_run {
            bail!("dry-run Telnyx client cannot POST {path}");
        }
        let response = self
            .request(reqwest::Method::POST, path)?
            .json(body)
            .send()
            .await?;
        decode_response::<ApiOne<T>>(response)
            .await
            .map(|body| body.data)
    }

    async fn patch_one<T, B>(&self, path: &str, body: &B) -> anyhow::Result<T>
    where
        T: for<'de> Deserialize<'de>,
        B: Serialize + ?Sized,
    {
        if self.dry_run {
            bail!("dry-run Telnyx client cannot PATCH {path}");
        }
        let response = self
            .request(reqwest::Method::PATCH, path)?
            .json(body)
            .send()
            .await?;
        decode_response::<ApiOne<T>>(response)
            .await
            .map(|body| body.data)
    }

    async fn post_command<B>(&self, path: &str, body: &B) -> anyhow::Result<()>
    where
        B: Serialize + ?Sized,
    {
        if self.dry_run {
            tracing::info!(telnyx_path = path, "dry-run Telnyx command accepted");
            return Ok(());
        }
        let response = self
            .request(reqwest::Method::POST, path)?
            .json(body)
            .send()
            .await?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let text = response.text().await.unwrap_or_default();
        bail!("Telnyx command {path} failed with {status}: {text}");
    }

    async fn resolve_phone_number_id(&self, phone_number: &str) -> anyhow::Result<String> {
        if phone_number.starts_with("pn_") || phone_number.starts_with("phone_number_") {
            return Ok(phone_number.to_string());
        }
        let numbers = self.list_phone_numbers().await?;
        numbers
            .into_iter()
            .find(|number| {
                number.id == phone_number || number.phone_number.as_deref() == Some(phone_number)
            })
            .map(|number| number.id)
            .with_context(|| format!("find Telnyx phone number {phone_number}"))
    }

    fn request(
        &self,
        method: reqwest::Method,
        path: &str,
    ) -> anyhow::Result<reqwest::RequestBuilder> {
        let api_key = self
            .api_key
            .as_deref()
            .context("missing Telnyx API key; set TELNYX_API_KEY or pass --dry-run-telnyx")?;
        Ok(self
            .http
            .request(method, format!("{}/{}", self.base_url, path))
            .bearer_auth(api_key)
            .header(reqwest::header::ACCEPT, "application/json"))
    }
}

pub struct AnswerRequest<'a> {
    pub call_control_id: &'a str,
    pub stream_url: &'a str,
}

async fn decode_response<T>(response: reqwest::Response) -> anyhow::Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if status == StatusCode::NO_CONTENT {
        bail!("Telnyx returned empty response where JSON body was expected");
    }
    if !status.is_success() {
        bail!("Telnyx request failed with {status}: {text}");
    }
    serde_json::from_str(&text).with_context(|| format!("decode Telnyx response: {text}"))
}
