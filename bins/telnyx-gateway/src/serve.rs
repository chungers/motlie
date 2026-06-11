use std::net::SocketAddr;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::adapter::SharedAsrRegistry;
use crate::call_control::TelnyxClient;
use crate::conversation::ConversationRuntime;
use crate::media::{self, SharedMediaRegistry};
use crate::operator::state::{LogLevel, SharedState};
use crate::text_calls::{self, SharedTextCallRegistry, TextCallStreamServices};
use crate::tts::SharedTtsRegistry;
use crate::webhook;

#[derive(Clone)]
pub struct AppServices {
    pub state: SharedState,
    pub telnyx: TelnyxClient,
    pub asr: SharedAsrRegistry,
    pub media: SharedMediaRegistry,
    pub tts: SharedTtsRegistry,
    pub conversation: ConversationRuntime,
    pub text_calls: SharedTextCallRegistry,
}

impl AppServices {
    pub fn text_call_services(&self) -> TextCallStreamServices {
        TextCallStreamServices {
            registry: self.text_calls.clone(),
            state: self.state.clone(),
            media: self.media.clone(),
            tts: self.tts.clone(),
            telnyx: self.telnyx.clone(),
        }
    }
}

pub fn router(services: AppServices) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route(
            "/api/v1/inbound-subscriptions",
            post(crate::api::inbound_subscriptions::upsert_subscription)
                .get(crate::api::inbound_subscriptions::list_subscriptions),
        )
        .route(
            "/api/v1/inbound-subscriptions/{subscription_id}",
            get(crate::api::inbound_subscriptions::get_subscription)
                .delete(crate::api::inbound_subscriptions::delete_subscription),
        )
        .route(
            "/api/v1/inbound-subscriptions/{subscription_id}/test",
            post(crate::api::inbound_subscriptions::test_subscription),
        )
        .route(
            "/api/v1/outbound-calls",
            post(crate::api::outbound_calls::post_outbound_call),
        )
        .route("/api/v1/calls", get(crate::api::calls::list_calls))
        .route("/api/v1/calls/{call_id}", get(crate::api::calls::get_call))
        .route("/telnyx/webhooks", post(telnyx_webhook))
        .route("/telnyx/media", get(telnyx_media))
        .with_state(services)
}

pub async fn serve(bind: SocketAddr, services: AppServices) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "telnyx gateway listener started");
    axum::serve(listener, router(services)).await?;
    Ok(())
}

async fn healthz() -> &'static str {
    "ok"
}

async fn telnyx_webhook(
    State(services): State<AppServices>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    match webhook::handle_voice_webhook_with_outcome(services.state.clone(), body).await {
        Ok(outcome) => {
            if let Some(trigger) = outcome.inbound_text_call {
                let text_services = services.text_call_services();
                tokio::spawn(async move {
                    text_calls::run_inbound_text_call_flow(text_services, trigger).await;
                });
            }
            (StatusCode::OK, outcome.message)
        }
        Err(error) => {
            let mut guard = services.state.write().await;
            guard.log(LogLevel::Warn, format!("invalid Telnyx webhook: {error:#}"));
            tracing::warn!(error = %error, "invalid Telnyx webhook");
            (StatusCode::OK, "ignored".to_string())
        }
    }
}

async fn telnyx_media(
    State(services): State<AppServices>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| {
        media::handle_socket(
            socket,
            services.state.clone(),
            services.asr.clone(),
            services.media.clone(),
            services.conversation.clone(),
            services.text_calls.clone(),
        )
    })
}
