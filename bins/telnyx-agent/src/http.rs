use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use motlie_telnyx_gateway::text_calls::turns::{
    AcceptCallResponse, CallConnectedPayload, CallOfferPayload, TEXT_CALL_PROTOCOL,
};

use crate::config::DaemonArgs;
use crate::gateway_client::GatewayClient;
use crate::socket;
use crate::text_ws;
use crate::tmux_bridge::TmuxBridge;

#[derive(Clone)]
pub struct AgentState {
    pub public_url: String,
    pub gateway: GatewayClient,
    pub bridge: TmuxBridge,
    pub outbound_timeout_ms: u64,
}

pub async fn run_daemon(args: DaemonArgs) -> anyhow::Result<()> {
    let gateway = GatewayClient::new(args.gateway_url.clone(), args.gateway_token.clone());
    let public_url = args.public_url.trim_end_matches('/').to_string();
    let bridge = TmuxBridge::new(
        &args.tmux_target,
        Duration::from_millis(args.reply_timeout_ms),
    )
    .await?;
    let state = AgentState {
        public_url: public_url.clone(),
        gateway: gateway.clone(),
        bridge,
        outbound_timeout_ms: args.outbound_timeout_ms,
    };

    let inbound_callback = format!("{public_url}/motlie/inbound-offers");
    for number in &args.subscribe_numbers {
        let subscription_id = subscription_id_for_number(number);
        gateway
            .register_subscription(subscription_id, number, &inbound_callback)
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
    Json(payload): Json<CallOfferPayload>,
) -> impl IntoResponse {
    accept_response(&state, &payload.call.id)
}

async fn outbound_connected(
    State(state): State<Arc<AgentState>>,
    Json(payload): Json<CallConnectedPayload>,
) -> impl IntoResponse {
    accept_response(&state, &payload.call.id)
}

fn accept_response(state: &AgentState, call_id: &str) -> impl IntoResponse {
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

fn subscription_id_for_number(number: &str) -> String {
    let sanitized = number
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || *character == '-' || *character == '_'
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "telnyx-agent-subscription".to_string()
    } else {
        format!("telnyx-agent-{sanitized}")
    }
}
