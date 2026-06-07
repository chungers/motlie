use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{self, Instant};
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use crate::call_control::TelnyxClient;
use crate::media::SharedMediaRegistry;
use crate::operator::state::{CallStatus, LogLevel, SharedState, TtsPlaybackStatus};
use crate::speech;
use crate::tts::{LiveTtsBackend, SharedTtsRegistry};

use super::offers::validate_call_url;
use super::turns::{AgentTextFrame, GatewayTextFrame, TextCallDirection, TEXT_CALL_PROTOCOL};

const OUTBOUND_TEXT_FRAME_CAPACITY: usize = 64;
const MAX_ACTIVE_TEXT_CALL_TURNS: usize = 32;
const MEDIA_READY_TIMEOUT: Duration = Duration::from_secs(20);
const PLAYBACK_WAIT_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Clone, Default)]
pub struct SharedTextCallRegistry {
    inner: Arc<Mutex<BTreeMap<String, TextCallSessionHandle>>>,
}

impl SharedTextCallRegistry {
    async fn insert(&self, gateway_call_id: String, handle: TextCallSessionHandle) {
        self.inner.lock().await.insert(gateway_call_id, handle);
    }

    pub async fn remove(&self, gateway_call_id: &str) {
        self.inner.lock().await.remove(gateway_call_id);
    }

    pub async fn send_caller_turn(
        &self,
        gateway_call_id: &str,
        text: String,
    ) -> anyhow::Result<Option<String>> {
        let handle = { self.inner.lock().await.get(gateway_call_id).cloned() };
        let Some(handle) = handle else {
            return Ok(None);
        };
        let turn_id = format!("turn_{}", Uuid::new_v4().simple());
        {
            let mut active_turns = handle.active_turns.lock().await;
            if active_turns.len() >= MAX_ACTIVE_TEXT_CALL_TURNS {
                anyhow::bail!("too many outstanding text-call turns");
            }
            active_turns.insert(turn_id.clone());
        }
        handle
            .send(GatewayTextFrame::CallerTurn {
                turn_id: turn_id.clone(),
                sequence: handle.next_sequence(),
                text,
            })
            .await?;
        Ok(Some(turn_id))
    }

    pub async fn send_session_end(&self, gateway_call_id: &str, reason: impl Into<String>) {
        let handle = { self.inner.lock().await.get(gateway_call_id).cloned() };
        if let Some(handle) = handle {
            let _ = handle
                .send(GatewayTextFrame::SessionEnd {
                    reason: reason.into(),
                    sequence: handle.next_sequence(),
                })
                .await;
        }
    }
}

#[derive(Clone)]
struct TextCallSessionHandle {
    tx: mpsc::Sender<GatewayTextFrame>,
    sequence: Arc<AtomicU64>,
    active_turns: Arc<Mutex<BTreeSet<String>>>,
}

impl TextCallSessionHandle {
    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    async fn send(&self, frame: GatewayTextFrame) -> anyhow::Result<()> {
        self.tx
            .send(frame)
            .await
            .context("send text-call frame to websocket task")
    }
}

#[derive(Clone)]
pub struct TextCallStreamServices {
    pub registry: SharedTextCallRegistry,
    pub state: SharedState,
    pub media: SharedMediaRegistry,
    pub tts: SharedTtsRegistry,
    pub telnyx: TelnyxClient,
}

#[derive(Clone, Debug)]
pub struct TextCallSetup {
    pub gateway_call_id: String,
    pub call_url: String,
    pub direction: TextCallDirection,
}

pub async fn connect_application_stream(
    services: TextCallStreamServices,
    setup: TextCallSetup,
) -> anyhow::Result<()> {
    validate_call_url(&setup.call_url)?;
    let (socket, _) = tokio_tungstenite::connect_async(setup.call_url.as_str())
        .await
        .with_context(|| format!("connect text call websocket for {}", setup.gateway_call_id))?;
    let (mut write, read) = socket.split();

    let start = GatewayTextFrame::SessionStart {
        protocol: TEXT_CALL_PROTOCOL.to_string(),
        call_id: setup.gateway_call_id.clone(),
        direction: setup.direction,
    };
    send_json_frame(&mut write, &start).await?;

    let (tx, rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
    let handle = TextCallSessionHandle {
        tx,
        sequence: Arc::new(AtomicU64::new(1)),
        active_turns: Arc::new(Mutex::new(BTreeSet::new())),
    };
    services
        .registry
        .insert(setup.gateway_call_id.clone(), handle.clone())
        .await;

    tokio::spawn(run_text_call_session(
        services,
        setup.gateway_call_id,
        handle,
        read,
        write,
        rx,
    ));
    Ok(())
}

async fn run_text_call_session<W, R>(
    services: TextCallStreamServices,
    gateway_call_id: String,
    handle: TextCallSessionHandle,
    mut read: R,
    mut write: W,
    mut rx: mpsc::Receiver<GatewayTextFrame>,
) where
    W: futures_util::Sink<Message> + Unpin,
    W::Error: std::error::Error + Send + Sync + 'static,
    R: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut gateway_closed = false;
    loop {
        tokio::select! {
            frame = rx.recv() => {
                let Some(frame) = frame else {
                    gateway_closed = true;
                    break;
                };
                if matches!(frame, GatewayTextFrame::SessionEnd { .. }) {
                    gateway_closed = true;
                }
                if let Err(error) = send_json_frame(&mut write, &frame).await {
                    log_text_call_error(&services.state, &gateway_call_id, error).await;
                    break;
                }
                if gateway_closed {
                    let _ = write.close().await;
                    break;
                }
            }
            message = read.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(error) = handle_agent_message(
                            &services,
                            &gateway_call_id,
                            &handle,
                            text.as_str(),
                        ).await {
                            log_text_call_error(&services.state, &gateway_call_id, error).await;
                            let _ = send_error_frame(&handle, "protocol_error", "invalid text-call message").await;
                            let _ = hangup_gateway_call(&services, &gateway_call_id, "text-call protocol error").await;
                            break;
                        }
                    }
                    Some(Ok(Message::Binary(_))) => {
                        let _ = send_error_frame(&handle, "binary_not_allowed", "text-call protocol accepts JSON text frames only").await;
                        let _ = hangup_gateway_call(&services, &gateway_call_id, "binary text-call frame").await;
                        break;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                    Some(Err(error)) => {
                        log_text_call_error(&services.state, &gateway_call_id, anyhow::Error::from(error)).await;
                        break;
                    }
                }
            }
        }
    }

    handle.active_turns.lock().await.clear();
    services.registry.remove(&gateway_call_id).await;
    if !gateway_closed {
        let _ =
            hangup_gateway_call(&services, &gateway_call_id, "text-call websocket closed").await;
    }
}

async fn handle_agent_message(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    handle: &TextCallSessionHandle,
    text: &str,
) -> anyhow::Result<()> {
    let frame: AgentTextFrame = serde_json::from_str(text).context("decode app text-call frame")?;
    match frame {
        AgentTextFrame::AgentTurn { turn_id, text } => {
            let mut active_turns = handle.active_turns.lock().await;
            if !active_turns.remove(&turn_id) {
                drop(active_turns);
                send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                return Ok(());
            }
            drop(active_turns);
            let queued =
                queue_agent_speech_with_media_wait(services, gateway_call_id.to_string(), text)
                    .await?;
            handle
                .send(GatewayTextFrame::PlaybackStarted {
                    turn_id: turn_id.clone(),
                    sequence: handle.next_sequence(),
                })
                .await?;
            let wait_services = services.clone();
            let wait_handle = handle.clone();
            let wait_call_id = gateway_call_id.to_string();
            tokio::spawn(async move {
                wait_for_playback_terminal(
                    &wait_services.state,
                    &wait_call_id,
                    &queued.playback_id,
                )
                .await;
                let _ = wait_handle
                    .send(GatewayTextFrame::PlaybackFinished {
                        turn_id,
                        sequence: wait_handle.next_sequence(),
                    })
                    .await;
            });
        }
        AgentTextFrame::AgentClose { reason } => {
            handle
                .send(GatewayTextFrame::SessionEnd {
                    reason: reason.unwrap_or_else(|| "agent.close".to_string()),
                    sequence: handle.next_sequence(),
                })
                .await?;
            hangup_gateway_call(services, gateway_call_id, "agent requested close").await?;
        }
    }
    Ok(())
}

async fn send_error_frame(
    handle: &TextCallSessionHandle,
    code: impl Into<String>,
    message: impl Into<String>,
) -> anyhow::Result<()> {
    handle
        .send(GatewayTextFrame::Error {
            code: code.into(),
            message: message.into(),
            sequence: handle.next_sequence(),
        })
        .await
}

async fn queue_agent_speech_with_media_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
) -> anyhow::Result<speech::QueuedSpeech> {
    let deadline = Instant::now() + MEDIA_READY_TIMEOUT;
    loop {
        match speech::queue_speech(
            &services.state,
            &services.media,
            &services.tts,
            LiveTtsBackend::default(),
            gateway_call_id.clone(),
            text.clone(),
            "text-call agent.turn",
        )
        .await
        {
            Ok(queued) => return Ok(queued),
            Err(error)
                if Instant::now() < deadline
                    && format!("{error:#}").contains("media stream is not ready") =>
            {
                time::sleep(Duration::from_millis(250)).await;
            }
            Err(error) => return Err(error),
        }
    }
}

pub async fn queue_fallback_and_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
) -> anyhow::Result<()> {
    let queued =
        queue_agent_speech_with_media_wait(services, gateway_call_id.clone(), text).await?;
    wait_for_playback_terminal(&services.state, &gateway_call_id, &queued.playback_id).await;
    Ok(())
}

async fn wait_for_playback_terminal(state: &SharedState, gateway_call_id: &str, playback_id: &str) {
    let deadline = Instant::now() + PLAYBACK_WAIT_TIMEOUT;
    loop {
        let terminal = {
            let guard = state.read().await;
            guard.calls.get(gateway_call_id).is_none_or(|call| {
                matches!(call.status, CallStatus::Ended | CallStatus::Failed)
                    || call.tts.as_ref().is_some_and(|tts| {
                        tts.playback_id == playback_id
                            && matches!(
                                tts.status,
                                TtsPlaybackStatus::Completed
                                    | TtsPlaybackStatus::Canceled
                                    | TtsPlaybackStatus::Failed
                            )
                    })
            })
        };
        if terminal || Instant::now() >= deadline {
            return;
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

pub async fn hangup_gateway_call(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    reason: &str,
) -> anyhow::Result<()> {
    let call_control_id = {
        let guard = services.state.read().await;
        guard
            .calls
            .get(gateway_call_id)
            .map(|call| call.ids.call_control_id.clone())
    };
    let Some(call_control_id) = call_control_id else {
        return Ok(());
    };
    services.telnyx.hangup_call(&call_control_id).await?;
    let mut guard = services.state.write().await;
    if let Some(call) = guard.calls.get_mut(gateway_call_id) {
        call.status = CallStatus::Ended;
        call.push_timeline(reason.to_string());
    }
    Ok(())
}

async fn send_json_frame<W>(write: &mut W, frame: &GatewayTextFrame) -> anyhow::Result<()>
where
    W: futures_util::Sink<Message> + Unpin,
    W::Error: std::error::Error + Send + Sync + 'static,
{
    let encoded = serde_json::to_string(frame).context("encode text-call frame")?;
    write
        .send(Message::Text(encoded.into()))
        .await
        .context("send text-call websocket frame")
}

async fn log_text_call_error(state: &SharedState, gateway_call_id: &str, error: anyhow::Error) {
    let message = format!("text-call error for {gateway_call_id}: {error:#}");
    let mut guard = state.write().await;
    guard.log(LogLevel::Warn, message.clone());
    if let Some(call) = guard.calls.get_mut(gateway_call_id) {
        call.last_error = Some(message.clone());
        call.push_timeline(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn caller_turn_without_session_is_ignored() {
        let registry = SharedTextCallRegistry::default();
        let turn = registry
            .send_caller_turn("missing-call", "hello".to_string())
            .await
            .expect("registry should not fail");
        assert_eq!(turn, None);
    }

    #[tokio::test]
    async fn caller_turn_allows_multiple_outstanding_turns() {
        let registry = SharedTextCallRegistry::default();
        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        registry
            .insert("call-test".to_string(), handle.clone())
            .await;

        let first = registry
            .send_caller_turn("call-test", "first".to_string())
            .await
            .expect("first turn should send")
            .expect("first turn id");
        let second = registry
            .send_caller_turn("call-test", "second".to_string())
            .await
            .expect("second turn should send")
            .expect("second turn id");

        assert_ne!(first, second);
        assert_eq!(handle.active_turns.lock().await.len(), 2);
        assert!(matches!(
            rx.recv().await,
            Some(GatewayTextFrame::CallerTurn { text, .. }) if text == "first"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(GatewayTextFrame::CallerTurn { text, .. }) if text == "second"
        ));
    }

    #[tokio::test]
    async fn caller_turn_rejects_when_outstanding_turn_cap_is_reached() {
        let registry = SharedTextCallRegistry::default();
        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        {
            let mut active_turns = handle.active_turns.lock().await;
            for index in 0..MAX_ACTIVE_TEXT_CALL_TURNS {
                active_turns.insert(format!("turn-preexisting-{index}"));
            }
        }
        registry
            .insert("call-test".to_string(), handle.clone())
            .await;

        let error = registry
            .send_caller_turn("call-test", "overflow".to_string())
            .await
            .expect_err("turn cap should reject new caller turns");

        assert!(format!("{error:#}").contains("too many outstanding text-call turns"));
        assert!(rx.try_recv().is_err());
        assert_eq!(
            handle.active_turns.lock().await.len(),
            MAX_ACTIVE_TEXT_CALL_TURNS
        );
    }

    fn test_handle(tx: mpsc::Sender<GatewayTextFrame>) -> TextCallSessionHandle {
        TextCallSessionHandle {
            tx,
            sequence: Arc::new(AtomicU64::new(1)),
            active_turns: Arc::new(Mutex::new(BTreeSet::new())),
        }
    }
}
