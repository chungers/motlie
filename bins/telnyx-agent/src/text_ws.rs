use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use motlie_agent::voice::telnyx::text::{AgentTextFrame, GatewayTextFrame, PlaybackFinishedStatus};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::tmux_bridge::{BridgeAbortToken, BridgeTurnEvent, TmuxBridge};

struct ActiveBridgeTurn {
    turn_id: String,
    abort: BridgeAbortToken,
    task: JoinHandle<()>,
}

pub async fn handle_gateway_socket(socket: WebSocket, bridge: TmuxBridge) {
    let (mut write, mut read) = socket.split();
    let (agent_tx, mut agent_rx) = mpsc::channel::<AgentTextFrame>(32);
    let writer = tokio::spawn(async move {
        while let Some(frame) = agent_rx.recv().await {
            if send_agent_frame(&mut write, &frame).await.is_err() {
                break;
            }
        }
    });
    let mut active: Option<ActiveBridgeTurn> = None;

    while let Some(message) = read.next().await {
        match message {
            Ok(Message::Text(text)) => {
                let frame = serde_json::from_str::<GatewayTextFrame>(&text);
                match frame {
                    Ok(GatewayTextFrame::CallerTurn { turn_id, text, .. }) => {
                        cancel_active_turn(&mut active);
                        active = Some(spawn_bridge_turn(
                            bridge.clone(),
                            agent_tx.clone(),
                            turn_id,
                            text,
                        ));
                    }
                    Ok(GatewayTextFrame::PlaybackFinished {
                        turn_id,
                        status:
                            PlaybackFinishedStatus::Canceled | PlaybackFinishedStatus::Superseded,
                        ..
                    }) => {
                        cancel_matching_turn(&mut active, &turn_id);
                    }
                    Ok(GatewayTextFrame::TurnSuperseded { turn_id, .. }) => {
                        cancel_matching_turn(&mut active, &turn_id);
                    }
                    Ok(GatewayTextFrame::SessionEnd { .. }) => break,
                    Ok(GatewayTextFrame::CallerPartial { .. })
                    | Ok(GatewayTextFrame::SessionStart { .. })
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

    cancel_active_turn(&mut active);
    drop(agent_tx);
    let _ = writer.await;
}

fn spawn_bridge_turn(
    bridge: TmuxBridge,
    agent_tx: mpsc::Sender<AgentTextFrame>,
    turn_id: String,
    text: String,
) -> ActiveBridgeTurn {
    let abort = BridgeAbortToken::default();
    let task_abort = abort.clone();
    let task_turn_id = turn_id.clone();
    let task = tokio::spawn(async move {
        let (event_tx, mut event_rx) = mpsc::channel(32);
        let bridge_fut =
            bridge.send_turn_streaming(&task_turn_id, &text, event_tx, task_abort.clone());
        tokio::pin!(bridge_fut);
        loop {
            tokio::select! {
                result = &mut bridge_fut => {
                    if let Err(error) = result {
                        if !task_abort.is_canceled() {
                            let _ = agent_tx.send(AgentTextFrame::AgentClose {
                                reason: Some(format!("tmux bridge failed: {error:#}")),
                            }).await;
                        }
                    }
                    break;
                }
                event = event_rx.recv() => {
                    match event {
                        Some(BridgeTurnEvent::Partial(text)) => {
                            let _ = agent_tx.send(AgentTextFrame::AgentTurnPartial {
                                turn_id: task_turn_id.clone(),
                                text,
                                append: true,
                            }).await;
                        }
                        Some(BridgeTurnEvent::Final(text)) => {
                            let _ = agent_tx.send(AgentTextFrame::AgentTurn {
                                turn_id: task_turn_id.clone(),
                                text,
                            }).await;
                        }
                        None => {}
                    }
                }
            }
        }
    });
    ActiveBridgeTurn {
        turn_id,
        abort,
        task,
    }
}

fn cancel_matching_turn(active: &mut Option<ActiveBridgeTurn>, turn_id: &str) {
    if active.as_ref().is_some_and(|turn| turn.turn_id == turn_id) {
        cancel_active_turn(active);
    }
}

fn cancel_active_turn(active: &mut Option<ActiveBridgeTurn>) {
    if let Some(turn) = active.take() {
        turn.abort.cancel();
        tokio::spawn(async move {
            let mut task = turn.task;
            tokio::select! {
                _ = &mut task => {}
                _ = tokio::time::sleep(Duration::from_millis(500)) => {
                    task.abort();
                }
            }
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;
    use tokio::time::{self, Duration};

    #[tokio::test]
    async fn cancel_active_turn_signals_abort_before_hard_abort() {
        let abort = BridgeAbortToken::default();
        let task_abort = abort.clone();
        let (observed_tx, observed_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            task_abort.canceled().await;
            let _ = observed_tx.send(());
            time::sleep(Duration::from_secs(5)).await;
        });
        let mut active = Some(ActiveBridgeTurn {
            turn_id: "turn-test".to_string(),
            abort,
            task,
        });

        cancel_active_turn(&mut active);

        time::timeout(Duration::from_secs(1), observed_rx)
            .await
            .expect("turn task should observe abort before hard abort")
            .expect("observer should send");
    }
}
