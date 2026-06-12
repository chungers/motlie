use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use motlie_agent::voice::telnyx::text::{
    AgentTextFrame, CallerSpeechState, GatewayTextFrame, PlaybackFinishedStatus,
};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::tmux_bridge::{BridgeAbortToken, BridgeTurnEvent, TmuxBridge};

struct ActiveBridgeTurn {
    turn_id: String,
    abort: BridgeAbortToken,
    task: JoinHandle<()>,
}

#[derive(Clone, Debug, PartialEq)]
struct PartialAdvisorySummary {
    utterance_id: String,
    latest_text: String,
    partial_count: usize,
    latest_confidence: Option<f32>,
    max_confidence: Option<f32>,
    latest_stability: Option<f32>,
    max_stability: Option<f32>,
    latest_speech_state: CallerSpeechState,
    reply_allowed_seen: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct PartialAdvisoryContext {
    utterance_id: Option<String>,
    latest_text: String,
    partial_count: usize,
    latest_confidence: Option<f32>,
    max_confidence: Option<f32>,
    latest_stability: Option<f32>,
    max_stability: Option<f32>,
    latest_speech_state: Option<CallerSpeechState>,
    reply_allowed_seen: bool,
}

impl PartialAdvisoryContext {
    fn observe(
        &mut self,
        utterance_id: String,
        text: String,
        confidence: Option<f32>,
        stability: Option<f32>,
        speech_state: CallerSpeechState,
        reply_allowed: bool,
    ) {
        if self.utterance_id.as_deref() != Some(utterance_id.as_str()) {
            *self = Self {
                utterance_id: Some(utterance_id),
                ..Self::default()
            };
        }
        self.partial_count += 1;
        self.latest_text = text;
        self.latest_confidence = normalized_advisory_score(confidence);
        self.latest_stability = normalized_advisory_score(stability);
        self.max_confidence = max_score(self.max_confidence, self.latest_confidence);
        self.max_stability = max_score(self.max_stability, self.latest_stability);
        self.latest_speech_state = Some(speech_state);
        self.reply_allowed_seen |= reply_allowed;
    }

    fn take_matching(&mut self, utterance_id: Option<&str>) -> Option<PartialAdvisorySummary> {
        if self.partial_count == 0 {
            return None;
        }
        if let Some(utterance_id) = utterance_id {
            if self.utterance_id.as_deref() != Some(utterance_id) {
                return None;
            }
        }
        let summary = PartialAdvisorySummary {
            utterance_id: self.utterance_id.clone()?,
            latest_text: self.latest_text.clone(),
            partial_count: self.partial_count,
            latest_confidence: self.latest_confidence,
            max_confidence: self.max_confidence,
            latest_stability: self.latest_stability,
            max_stability: self.max_stability,
            latest_speech_state: self.latest_speech_state?,
            reply_allowed_seen: self.reply_allowed_seen,
        };
        *self = Self::default();
        Some(summary)
    }
}

fn normalized_advisory_score(score: Option<f32>) -> Option<f32> {
    score.filter(|score| score.is_finite() && *score >= 0.0 && *score <= 1.0)
}

fn max_score(left: Option<f32>, right: Option<f32>) -> Option<f32> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
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
    let mut partial_context = PartialAdvisoryContext::default();

    while let Some(message) = read.next().await {
        match message {
            Ok(Message::Text(text)) => {
                let frame = serde_json::from_str::<GatewayTextFrame>(&text);
                match frame {
                    Ok(GatewayTextFrame::CallerTurn {
                        turn_id,
                        utterance_id,
                        text,
                        ..
                    }) => {
                        if let Some(summary) =
                            partial_context.take_matching(utterance_id.as_deref())
                        {
                            tracing::debug!(
                                utterance_id = summary.utterance_id,
                                partial_count = summary.partial_count,
                                latest_text_chars = summary.latest_text.chars().count(),
                                latest_confidence = summary.latest_confidence,
                                max_confidence = summary.max_confidence,
                                latest_stability = summary.latest_stability,
                                max_stability = summary.max_stability,
                                latest_speech_state = ?summary.latest_speech_state,
                                reply_allowed_seen = summary.reply_allowed_seen,
                                "telnyx_agent.text_ws.caller_partial_summary"
                            );
                        }
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
                    Ok(GatewayTextFrame::CallerPartial {
                        utterance_id,
                        text,
                        confidence,
                        stability,
                        speech_state,
                        reply_allowed,
                        ..
                    }) => {
                        partial_context.observe(
                            utterance_id.clone(),
                            text,
                            confidence,
                            stability,
                            speech_state,
                            reply_allowed,
                        );
                        tracing::trace!(
                            utterance_id,
                            confidence,
                            stability,
                            speech_state = ?speech_state,
                            reply_allowed,
                            "telnyx_agent.text_ws.caller_partial_advisory"
                        );
                    }
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

    #[test]
    fn partial_advisory_context_summarizes_scores_per_utterance() {
        let mut context = PartialAdvisoryContext::default();

        context.observe(
            "utt-1".to_string(),
            "hello wor".to_string(),
            Some(0.42),
            None,
            CallerSpeechState::Speaking,
            false,
        );
        context.observe(
            "utt-1".to_string(),
            "hello world".to_string(),
            Some(0.84),
            Some(0.70),
            CallerSpeechState::EndpointCandidate,
            false,
        );

        let summary = context
            .take_matching(Some("utt-1"))
            .expect("matching final turn should consume advisory partials");
        assert_eq!(summary.utterance_id, "utt-1");
        assert_eq!(summary.latest_text, "hello world");
        assert_eq!(summary.partial_count, 2);
        assert_eq!(summary.latest_confidence, Some(0.84));
        assert_eq!(summary.max_confidence, Some(0.84));
        assert_eq!(summary.latest_stability, Some(0.70));
        assert_eq!(summary.max_stability, Some(0.70));
        assert_eq!(
            summary.latest_speech_state,
            CallerSpeechState::EndpointCandidate
        );
        assert!(!summary.reply_allowed_seen);
        assert_eq!(context.take_matching(Some("utt-1")), None);
    }

    #[test]
    fn partial_advisory_context_omits_invalid_scores() {
        let mut context = PartialAdvisoryContext::default();

        context.observe(
            "utt-1".to_string(),
            "hello".to_string(),
            Some(1.1),
            Some(f32::NAN),
            CallerSpeechState::Speaking,
            false,
        );

        let summary = context
            .take_matching(Some("utt-1"))
            .expect("matching final turn should consume advisory partials");
        assert_eq!(summary.latest_confidence, None);
        assert_eq!(summary.max_confidence, None);
        assert_eq!(summary.latest_stability, None);
        assert_eq!(summary.max_stability, None);
    }

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
