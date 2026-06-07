use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use motlie_telnyx_gateway::text_calls::turns::{AgentTextFrame, GatewayTextFrame};

use crate::tmux_bridge::TmuxBridge;

pub async fn handle_gateway_socket(socket: WebSocket, bridge: TmuxBridge) {
    let (mut write, mut read) = socket.split();
    while let Some(message) = read.next().await {
        match message {
            Ok(Message::Text(text)) => {
                let frame = serde_json::from_str::<GatewayTextFrame>(&text);
                match frame {
                    Ok(GatewayTextFrame::CallerTurn { turn_id, text, .. }) => {
                        match bridge.send_turn(&turn_id, &text).await {
                            Ok(reply) => {
                                let response = AgentTextFrame::AgentTurn {
                                    turn_id,
                                    text: reply,
                                };
                                if send_agent_frame(&mut write, &response).await.is_err() {
                                    break;
                                }
                            }
                            Err(error) => {
                                let response = AgentTextFrame::AgentClose {
                                    reason: Some(format!("tmux bridge failed: {error:#}")),
                                };
                                let _ = send_agent_frame(&mut write, &response).await;
                                break;
                            }
                        }
                    }
                    Ok(GatewayTextFrame::SessionEnd { .. }) => break,
                    Ok(GatewayTextFrame::SessionStart { .. })
                    | Ok(GatewayTextFrame::PlaybackStarted { .. })
                    | Ok(GatewayTextFrame::PlaybackFinished { .. })
                    | Ok(GatewayTextFrame::Error { .. }) => {}
                    Err(error) => {
                        tracing::warn!(error = %error, "telnyx_agent.text_ws.invalid_gateway_frame");
                        break;
                    }
                }
            }
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Binary(_)) => break,
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
        }
    }
}

async fn send_agent_frame(
    write: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    frame: &AgentTextFrame,
) -> anyhow::Result<()> {
    let encoded = serde_json::to_string(frame)?;
    write.send(Message::Text(encoded.into())).await?;
    Ok(())
}
