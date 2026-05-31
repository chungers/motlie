use std::net::SocketAddr;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::adapter::SharedAsrFactory;
use crate::call_control::TelnyxClient;
use crate::media;
use crate::operator::state::{LogLevel, SharedState};
use crate::webhook;

#[derive(Clone)]
pub struct AppServices {
    pub state: SharedState,
    pub telnyx: TelnyxClient,
    pub asr: SharedAsrFactory,
}

pub fn router(services: AppServices) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
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
    match webhook::handle_voice_webhook(services.state.clone(), body).await {
        Ok(message) => (StatusCode::OK, message),
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
        media::handle_socket(socket, services.state.clone(), services.asr.clone())
    })
}
