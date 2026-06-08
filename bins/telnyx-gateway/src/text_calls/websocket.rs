use std::collections::BTreeMap;
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
use crate::speech::{self, SpeechConflictPolicy, SpeechQueueRequest};
use crate::tts::{LiveTtsBackend, SharedTtsRegistry};

use super::offers::validate_call_url;
use super::turns::{
    AgentTextFrame, GatewayTextFrame, PlaybackFinishedStatus, TextCallDirection, TEXT_CALL_PROTOCOL,
};

const OUTBOUND_TEXT_FRAME_CAPACITY: usize = 64;
const MAX_ACTIVE_TEXT_CALL_TURNS: usize = 32;
const MEDIA_READY_TIMEOUT: Duration = Duration::from_secs(20);
const PLAYBACK_WAIT_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Clone, Debug, Eq, PartialEq)]
enum TextCallTurnState {
    Pending,
    Superseded,
    Playing { playback_id: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AgentTurnDisposition {
    Accepted,
    Superseded,
    Invalid,
}

#[derive(Debug, Default)]
struct TextCallTurnTracker {
    turns: BTreeMap<String, TextCallTurnState>,
    playback_turns: BTreeMap<String, String>,
}

impl TextCallTurnTracker {
    fn add_caller_turn(&mut self, turn_id: String) -> anyhow::Result<()> {
        if self.turns.len() >= MAX_ACTIVE_TEXT_CALL_TURNS {
            anyhow::bail!("too many outstanding text-call turns");
        }
        for state in self.turns.values_mut() {
            if matches!(state, TextCallTurnState::Pending) {
                *state = TextCallTurnState::Superseded;
            }
        }
        self.turns.insert(turn_id, TextCallTurnState::Pending);
        Ok(())
    }

    fn accept_agent_turn(&mut self, turn_id: &str) -> AgentTurnDisposition {
        match self.turns.get(turn_id) {
            Some(TextCallTurnState::Pending) => {
                self.turns.remove(turn_id);
                AgentTurnDisposition::Accepted
            }
            Some(TextCallTurnState::Superseded) => {
                self.turns.remove(turn_id);
                AgentTurnDisposition::Superseded
            }
            Some(TextCallTurnState::Playing { .. }) | None => AgentTurnDisposition::Invalid,
        }
    }

    fn start_playback(&mut self, turn_id: String, playback_id: String) {
        if let Some(TextCallTurnState::Playing {
            playback_id: previous_playback_id,
        }) = self.turns.insert(
            turn_id.clone(),
            TextCallTurnState::Playing {
                playback_id: playback_id.clone(),
            },
        ) {
            self.playback_turns.remove(&previous_playback_id);
        }
        self.playback_turns.insert(playback_id, turn_id);
    }

    fn close_playback(&mut self, playback_id: &str) -> Option<String> {
        let turn_id = self.playback_turns.remove(playback_id)?;
        match self.turns.get(&turn_id) {
            Some(TextCallTurnState::Playing {
                playback_id: active,
            }) if active == playback_id => {
                self.turns.remove(&turn_id);
                Some(turn_id)
            }
            _ => None,
        }
    }

    fn is_playback_active(&self, playback_id: &str) -> bool {
        self.playback_turns.contains_key(playback_id)
    }

    fn clear(&mut self) {
        self.turns.clear();
        self.playback_turns.clear();
    }

    #[cfg(test)]
    fn outstanding_len(&self) -> usize {
        self.turns.len()
    }
}

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
        handle.turns.lock().await.add_caller_turn(turn_id.clone())?;
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
    turns: Arc<Mutex<TextCallTurnTracker>>,
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
        turns: Arc::new(Mutex::new(TextCallTurnTracker::default())),
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

    handle.turns.lock().await.clear();
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
            let disposition = handle.turns.lock().await.accept_agent_turn(&turn_id);
            match disposition {
                AgentTurnDisposition::Accepted => {}
                AgentTurnDisposition::Superseded => {
                    send_playback_finished(handle, turn_id, PlaybackFinishedStatus::Superseded)
                        .await?;
                    return Ok(());
                }
                AgentTurnDisposition::Invalid => {
                    send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                    return Ok(());
                }
            }

            let queued =
                queue_agent_speech_with_media_wait(services, gateway_call_id.to_string(), text)
                    .await?;
            if let Some(replaced_playback_id) = queued.replaced_playback_id.as_deref() {
                send_replaced_playback_canceled(handle, replaced_playback_id).await?;
            }

            handle
                .turns
                .lock()
                .await
                .start_playback(turn_id.clone(), queued.playback_id.clone());
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
                if let Some(status) = wait_for_playback_terminal(
                    &wait_services.state,
                    &wait_handle,
                    &wait_call_id,
                    &queued.playback_id,
                )
                .await
                {
                    let turn_id = wait_handle
                        .turns
                        .lock()
                        .await
                        .close_playback(&queued.playback_id);
                    if let Some(turn_id) = turn_id {
                        let _ = send_playback_finished(&wait_handle, turn_id, status).await;
                    }
                }
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

async fn send_playback_finished(
    handle: &TextCallSessionHandle,
    turn_id: String,
    status: PlaybackFinishedStatus,
) -> anyhow::Result<()> {
    handle
        .send(GatewayTextFrame::PlaybackFinished {
            turn_id,
            sequence: handle.next_sequence(),
            status,
        })
        .await
}

async fn send_replaced_playback_canceled(
    handle: &TextCallSessionHandle,
    replaced_playback_id: &str,
) -> anyhow::Result<Option<String>> {
    let replaced_turn_id = handle
        .turns
        .lock()
        .await
        .close_playback(replaced_playback_id);
    if let Some(replaced_turn_id) = replaced_turn_id.as_ref() {
        send_playback_finished(
            handle,
            replaced_turn_id.clone(),
            PlaybackFinishedStatus::Canceled,
        )
        .await?;
    }
    Ok(replaced_turn_id)
}

async fn queue_agent_speech_with_media_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
) -> anyhow::Result<speech::QueuedSpeech> {
    let deadline = Instant::now() + MEDIA_READY_TIMEOUT;
    loop {
        match speech::queue_speech_with_request(
            &services.state,
            &services.media,
            &services.tts,
            SpeechQueueRequest {
                tts_backend: LiveTtsBackend::default(),
                gateway_call_id: gateway_call_id.clone(),
                text: text.clone(),
                source_label: "text-call agent.turn".to_string(),
                conflict_policy: SpeechConflictPolicy::CancelAndReplace,
            },
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
    wait_for_playback_terminal_without_turn(&services.state, &gateway_call_id, &queued.playback_id)
        .await;
    Ok(())
}

async fn wait_for_playback_terminal_without_turn(
    state: &SharedState,
    gateway_call_id: &str,
    playback_id: &str,
) {
    let deadline = Instant::now() + PLAYBACK_WAIT_TIMEOUT;
    loop {
        if playback_terminal_status(state, gateway_call_id, playback_id)
            .await
            .is_some()
            || Instant::now() >= deadline
        {
            return;
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn wait_for_playback_terminal(
    state: &SharedState,
    handle: &TextCallSessionHandle,
    gateway_call_id: &str,
    playback_id: &str,
) -> Option<PlaybackFinishedStatus> {
    let deadline = Instant::now() + PLAYBACK_WAIT_TIMEOUT;
    loop {
        if !handle.turns.lock().await.is_playback_active(playback_id) {
            return None;
        }
        if let Some(status) = playback_terminal_status(state, gateway_call_id, playback_id).await {
            return Some(status);
        }
        if Instant::now() >= deadline {
            return Some(PlaybackFinishedStatus::Failed);
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn playback_terminal_status(
    state: &SharedState,
    gateway_call_id: &str,
    playback_id: &str,
) -> Option<PlaybackFinishedStatus> {
    let guard = state.read().await;
    let Some(call) = guard.calls.get(gateway_call_id) else {
        return Some(PlaybackFinishedStatus::Failed);
    };
    if matches!(call.status, CallStatus::Ended | CallStatus::Failed) {
        return Some(PlaybackFinishedStatus::Failed);
    }
    call.tts.as_ref().and_then(|tts| {
        if tts.playback_id == playback_id {
            playback_finished_status(tts.status)
        } else {
            None
        }
    })
}

fn playback_finished_status(status: TtsPlaybackStatus) -> Option<PlaybackFinishedStatus> {
    match status {
        TtsPlaybackStatus::Completed => Some(PlaybackFinishedStatus::Completed),
        TtsPlaybackStatus::Canceled => Some(PlaybackFinishedStatus::Canceled),
        TtsPlaybackStatus::Failed => Some(PlaybackFinishedStatus::Failed),
        TtsPlaybackStatus::Queued
        | TtsPlaybackStatus::Playing
        | TtsPlaybackStatus::MarkSent
        | TtsPlaybackStatus::Canceling => None,
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
        assert_eq!(handle.turns.lock().await.outstanding_len(), 2);
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
            let mut turns = handle.turns.lock().await;
            for index in 0..MAX_ACTIVE_TEXT_CALL_TURNS {
                turns.turns.insert(
                    format!("turn-preexisting-{index}"),
                    TextCallTurnState::Pending,
                );
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
            handle.turns.lock().await.outstanding_len(),
            MAX_ACTIVE_TEXT_CALL_TURNS
        );
    }

    #[test]
    fn turn_tracker_reports_older_pending_turns_as_superseded() {
        let mut tracker = TextCallTurnTracker::default();
        tracker
            .add_caller_turn("turn-old".to_string())
            .expect("old turn accepted");
        tracker
            .add_caller_turn("turn-new".to_string())
            .expect("new turn accepted");

        assert_eq!(
            tracker.accept_agent_turn("turn-old"),
            AgentTurnDisposition::Superseded
        );
        assert_eq!(tracker.outstanding_len(), 1);
        assert_eq!(
            tracker.accept_agent_turn("turn-old"),
            AgentTurnDisposition::Invalid
        );
        assert_eq!(
            tracker.accept_agent_turn("turn-new"),
            AgentTurnDisposition::Accepted
        );
    }

    #[test]
    fn turn_tracker_closes_replaced_playback_once() {
        let mut tracker = TextCallTurnTracker::default();
        tracker
            .add_caller_turn("turn-old".to_string())
            .expect("turn accepted");
        assert_eq!(
            tracker.accept_agent_turn("turn-old"),
            AgentTurnDisposition::Accepted
        );
        tracker.start_playback("turn-old".to_string(), "tts-old".to_string());

        assert_eq!(
            tracker.close_playback("tts-old"),
            Some("turn-old".to_string())
        );
        assert!(!tracker.is_playback_active("tts-old"));
        assert_eq!(tracker.close_playback("tts-old"), None);
        assert_eq!(
            tracker.accept_agent_turn("turn-old"),
            AgentTurnDisposition::Invalid
        );
    }

    #[test]
    fn tts_terminal_status_maps_to_playback_finished_status() {
        assert_eq!(
            playback_finished_status(TtsPlaybackStatus::Completed),
            Some(PlaybackFinishedStatus::Completed)
        );
        assert_eq!(
            playback_finished_status(TtsPlaybackStatus::Canceled),
            Some(PlaybackFinishedStatus::Canceled)
        );
        assert_eq!(
            playback_finished_status(TtsPlaybackStatus::Failed),
            Some(PlaybackFinishedStatus::Failed)
        );
        assert_eq!(playback_finished_status(TtsPlaybackStatus::Queued), None);
    }

    #[tokio::test]
    async fn replaced_playback_emits_canceled_finished_frame() {
        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        handle
            .turns
            .lock()
            .await
            .start_playback("turn-old".to_string(), "tts-old".to_string());

        let closed_turn = send_replaced_playback_canceled(&handle, "tts-old")
            .await
            .expect("canceled frame should send");

        assert_eq!(closed_turn.as_deref(), Some("turn-old"));
        assert!(matches!(
            rx.recv().await,
            Some(GatewayTextFrame::PlaybackFinished {
                turn_id,
                status: PlaybackFinishedStatus::Canceled,
                ..
            }) if turn_id == "turn-old"
        ));
        assert!(!handle.turns.lock().await.is_playback_active("tts-old"));
    }

    fn test_handle(tx: mpsc::Sender<GatewayTextFrame>) -> TextCallSessionHandle {
        TextCallSessionHandle {
            tx,
            sequence: Arc::new(AtomicU64::new(1)),
            turns: Arc::new(Mutex::new(TextCallTurnTracker::default())),
        }
    }
}
