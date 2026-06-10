use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, Mutex};
use tokio::time;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use crate::call_control::TelnyxClient;
use crate::media::SharedMediaRegistry;
use crate::operator::state::{CallStatus, LogLevel, SharedState, TtsPlaybackStatus};
use crate::quality::TextCallQualityConfig;
use crate::speech::{self, SpeechConflictPolicy, SpeechQueueRequest};
use crate::tts::{LiveTtsBackend, SharedTtsRegistry, StreamingSpeechTextPacker};

use super::offers::validate_call_url;
use super::turns::{
    AgentTextFrame, GatewayTextFrame, PlaybackFinishedStatus, TextCallDirection, TEXT_CALL_PROTOCOL,
};

const OUTBOUND_TEXT_FRAME_CAPACITY: usize = 64;
const DEFAULT_MAX_ACTIVE_TEXT_CALL_TURNS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextCallTurnTiming {
    finalized_at: Instant,
    caller_turn_sent_at: Instant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TextCallTurnState {
    Pending { timing: TextCallTurnTiming },
    Superseded { timing: TextCallTurnTiming },
    Playing { playback_id: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AgentTurnDisposition {
    Accepted { timing: TextCallTurnTiming },
    Superseded,
    Invalid,
}

struct AgentAppendTurn {
    packer: StreamingSpeechTextPacker,
    speech: Option<speech::AppendSpeechHandle>,
    timing: TextCallTurnTiming,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextCallSessionConfig {
    max_active_turns: usize,
    media_ready_timeout: Duration,
    playback_wait_timeout: Duration,
    latest_response_wins: bool,
}

impl From<&TextCallQualityConfig> for TextCallSessionConfig {
    fn from(config: &TextCallQualityConfig) -> Self {
        Self {
            max_active_turns: config.max_active_turns,
            media_ready_timeout: config.media_ready_timeout(),
            playback_wait_timeout: config.playback_wait_timeout(),
            latest_response_wins: config.latest_response_wins,
        }
    }
}

#[derive(Debug)]
struct TextCallTurnTracker {
    turns: BTreeMap<String, TextCallTurnState>,
    playback_turns: BTreeMap<String, String>,
    max_active_turns: usize,
}

impl Default for TextCallTurnTracker {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ACTIVE_TEXT_CALL_TURNS)
    }
}

impl TextCallTurnTracker {
    fn new(max_active_turns: usize) -> Self {
        Self {
            turns: BTreeMap::new(),
            playback_turns: BTreeMap::new(),
            max_active_turns: max_active_turns.max(1),
        }
    }

    fn add_caller_turn(
        &mut self,
        turn_id: String,
        finalized_at: Instant,
    ) -> anyhow::Result<TextCallTurnTiming> {
        self.ensure_caller_turn_capacity()?;
        self.mark_pending_superseded();
        Ok(self.add_caller_turn_unchecked(turn_id, finalized_at))
    }

    fn ensure_caller_turn_capacity(&self) -> anyhow::Result<()> {
        if self.turns.len() >= self.max_active_turns {
            anyhow::bail!("too many outstanding text-call turns");
        }
        Ok(())
    }

    fn add_caller_turn_unchecked(
        &mut self,
        turn_id: String,
        finalized_at: Instant,
    ) -> TextCallTurnTiming {
        let timing = TextCallTurnTiming {
            finalized_at,
            caller_turn_sent_at: Instant::now(),
        };
        self.turns
            .insert(turn_id, TextCallTurnState::Pending { timing });
        timing
    }

    fn mark_pending_superseded(&mut self) -> Vec<String> {
        let mut superseded = Vec::new();
        for (turn_id, state) in &mut self.turns {
            if let TextCallTurnState::Pending { timing } = state {
                superseded.push(turn_id.clone());
                *state = TextCallTurnState::Superseded { timing: *timing };
            }
        }
        superseded
    }

    fn agent_turn_disposition(&self, turn_id: &str) -> AgentTurnDisposition {
        match self.turns.get(turn_id).cloned() {
            Some(TextCallTurnState::Pending { timing }) => {
                AgentTurnDisposition::Accepted { timing }
            }
            Some(TextCallTurnState::Superseded { .. }) => AgentTurnDisposition::Superseded,
            Some(TextCallTurnState::Playing { .. }) | None => AgentTurnDisposition::Invalid,
        }
    }

    fn accept_agent_turn(&mut self, turn_id: &str) -> AgentTurnDisposition {
        let disposition = self.agent_turn_disposition(turn_id);
        if matches!(
            disposition,
            AgentTurnDisposition::Accepted { .. } | AgentTurnDisposition::Superseded
        ) {
            self.turns.remove(turn_id);
        }
        disposition
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

    pub async fn contains(&self, gateway_call_id: &str) -> bool {
        self.inner.lock().await.contains_key(gateway_call_id)
    }

    pub async fn send_caller_turn(
        &self,
        gateway_call_id: &str,
        text: String,
        finalized_at: Instant,
    ) -> anyhow::Result<Option<String>> {
        let handle = { self.inner.lock().await.get(gateway_call_id).cloned() };
        let Some(handle) = handle else {
            return Ok(None);
        };
        let turn_id = format!("turn_{}", Uuid::new_v4().simple());
        let mut turns = handle.turns.lock().await;
        turns.ensure_caller_turn_capacity()?;
        let superseded = turns.mark_pending_superseded();
        for superseded_turn_id in superseded {
            handle.try_send(GatewayTextFrame::TurnSuperseded {
                turn_id: superseded_turn_id,
                superseded_by_turn_id: turn_id.clone(),
                reason: "new_caller_turn".to_string(),
                sequence: handle.next_sequence(),
            })?;
        }
        handle.try_send(GatewayTextFrame::CallerTurn {
            turn_id: turn_id.clone(),
            sequence: handle.next_sequence(),
            text,
        })?;
        turns.add_caller_turn(turn_id.clone(), finalized_at)?;
        Ok(Some(turn_id))
    }

    pub async fn finish_playback(
        &self,
        gateway_call_id: &str,
        playback_id: &str,
        status: PlaybackFinishedStatus,
    ) -> bool {
        let handle = { self.inner.lock().await.get(gateway_call_id).cloned() };
        let Some(handle) = handle else {
            return false;
        };
        let turn_id = handle.turns.lock().await.close_playback(playback_id);
        let Some(turn_id) = turn_id else {
            return false;
        };
        let _ = send_playback_finished(&handle, turn_id, status).await;
        true
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
    append_turns: Arc<Mutex<BTreeMap<String, AgentAppendTurn>>>,
    config: TextCallSessionConfig,
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

    fn try_send(&self, frame: GatewayTextFrame) -> anyhow::Result<()> {
        self.tx.try_send(frame).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                anyhow::anyhow!("text-call outbound queue full")
            }
            mpsc::error::TrySendError::Closed(_) => {
                anyhow::anyhow!("text-call websocket task closed")
            }
        })
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

    let session_config = {
        let guard = services.state.read().await;
        TextCallSessionConfig::from(&guard.quality.config.text_call)
    };
    let (tx, rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
    let handle = TextCallSessionHandle {
        tx,
        sequence: Arc::new(AtomicU64::new(1)),
        turns: Arc::new(Mutex::new(TextCallTurnTracker::new(
            session_config.max_active_turns,
        ))),
        append_turns: Arc::new(Mutex::new(BTreeMap::new())),
        config: session_config,
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

    handle.append_turns.lock().await.clear();
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
        AgentTextFrame::AgentTurnPartial {
            turn_id,
            text,
            append,
        } => {
            if !append {
                send_error_frame(handle, "invalid_partial", "agent.turn.partial must append")
                    .await?;
                return Ok(());
            }
            let disposition = handle.turns.lock().await.agent_turn_disposition(&turn_id);
            let timing = match disposition {
                AgentTurnDisposition::Accepted { timing } => timing,
                AgentTurnDisposition::Superseded => {
                    send_playback_finished(handle, turn_id, PlaybackFinishedStatus::Superseded)
                        .await?;
                    return Ok(());
                }
                AgentTurnDisposition::Invalid => {
                    send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                    return Ok(());
                }
            };
            process_agent_turn_fragment(
                services,
                gateway_call_id,
                handle,
                turn_id,
                text,
                false,
                timing,
            )
            .await?;
        }
        AgentTextFrame::AgentTurn { turn_id, text } => {
            if handle.append_turns.lock().await.contains_key(&turn_id) {
                let disposition = handle.turns.lock().await.agent_turn_disposition(&turn_id);
                let timing = match disposition {
                    AgentTurnDisposition::Accepted { timing } => timing,
                    AgentTurnDisposition::Superseded => {
                        send_playback_finished(handle, turn_id, PlaybackFinishedStatus::Superseded)
                            .await?;
                        return Ok(());
                    }
                    AgentTurnDisposition::Invalid => {
                        send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                        return Ok(());
                    }
                };
                process_agent_turn_fragment(
                    services,
                    gateway_call_id,
                    handle,
                    turn_id,
                    text,
                    true,
                    timing,
                )
                .await?;
                return Ok(());
            }
            let agent_turn_received_at = Instant::now();
            let disposition = handle.turns.lock().await.accept_agent_turn(&turn_id);
            let timing = match disposition {
                AgentTurnDisposition::Accepted { timing } => timing,
                AgentTurnDisposition::Superseded => {
                    send_playback_finished(handle, turn_id, PlaybackFinishedStatus::Superseded)
                        .await?;
                    return Ok(());
                }
                AgentTurnDisposition::Invalid => {
                    send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                    return Ok(());
                }
            };
            emit_agent_turn_round_trip_span(
                services,
                gateway_call_id,
                &turn_id,
                timing,
                agent_turn_received_at,
            )
            .await;

            let queued = queue_agent_speech_with_media_wait(
                services,
                gateway_call_id.to_string(),
                text,
                handle.config,
                Some(timing.finalized_at),
            )
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
            spawn_playback_terminal_waiter(
                services.clone(),
                gateway_call_id.to_string(),
                handle.clone(),
                queued.playback_id.clone(),
            );
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

async fn process_agent_turn_fragment(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    handle: &TextCallSessionHandle,
    turn_id: String,
    text: String,
    final_fragment: bool,
    timing: TextCallTurnTiming,
) -> anyhow::Result<()> {
    let existing = handle.append_turns.lock().await.remove(&turn_id);
    let mut append_turn = if let Some(append_turn) = existing {
        append_turn
    } else {
        emit_agent_turn_round_trip_span(
            services,
            gateway_call_id,
            &turn_id,
            timing,
            Instant::now(),
        )
        .await;
        let (chunking_enabled, max_chars, first_chunk_max_chars) = {
            let guard = services.state.read().await;
            (
                guard.quality.config.tts.chunking_enabled,
                guard.quality.config.tts.max_text_chunk_chars,
                guard.quality.config.tts.first_chunk_max_chars,
            )
        };
        AgentAppendTurn {
            packer: StreamingSpeechTextPacker::new(
                chunking_enabled,
                max_chars,
                first_chunk_max_chars,
            ),
            speech: None,
            timing,
        }
    };

    let chunks = append_turn.packer.push_fragment(&text, final_fragment);
    if append_turn.speech.is_none() && !chunks.is_empty() {
        let (speech_handle, queued) = queue_append_agent_speech_with_media_wait(
            services,
            gateway_call_id.to_string(),
            chunks,
            handle.config,
            Some(append_turn.timing.finalized_at),
        )
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
        spawn_playback_terminal_waiter(
            services.clone(),
            gateway_call_id.to_string(),
            handle.clone(),
            queued.playback_id.clone(),
        );
        if final_fragment {
            speech_handle.finish().await?;
        } else {
            append_turn.speech = Some(speech_handle);
        }
    } else if let Some(speech) = append_turn.speech.as_ref() {
        speech.append_chunks(chunks, final_fragment).await?;
    } else if final_fragment {
        handle.turns.lock().await.accept_agent_turn(&turn_id);
        send_playback_finished(handle, turn_id.clone(), PlaybackFinishedStatus::Completed).await?;
    }

    if !final_fragment {
        handle
            .append_turns
            .lock()
            .await
            .insert(turn_id, append_turn);
    }
    Ok(())
}

fn spawn_playback_terminal_waiter(
    services: TextCallStreamServices,
    gateway_call_id: String,
    handle: TextCallSessionHandle,
    playback_id: String,
) {
    tokio::spawn(async move {
        if let Some(status) = wait_for_playback_terminal(
            &services.state,
            &handle,
            &gateway_call_id,
            &playback_id,
            handle.config.playback_wait_timeout,
        )
        .await
        {
            let turn_id = handle.turns.lock().await.close_playback(&playback_id);
            if let Some(turn_id) = turn_id {
                let _ = send_playback_finished(&handle, turn_id, status).await;
            }
        }
    });
}

async fn emit_agent_turn_round_trip_span(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    turn_id: &str,
    timing: TextCallTurnTiming,
    agent_turn_received_at: Instant,
) {
    let payload = match serde_json::json!({
        "turn_id": turn_id,
        "finalize_to_caller_turn_sent_ms": timing
            .caller_turn_sent_at
            .saturating_duration_since(timing.finalized_at)
            .as_millis() as u64,
    }) {
        serde_json::Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    let mut guard = services.state.write().await;
    let config_id = guard.quality.config_id.clone();
    let redaction_mode = guard.quality.config.logging.redaction_mode;
    guard.emit_quality_span_finished(
        gateway_call_id,
        crate::operator::state::QualitySpanEmission {
            config_id,
            redaction_mode,
            span_name: "app.agent_turn_wait",
            category: "model_generation",
            duration: agent_turn_received_at.saturating_duration_since(timing.caller_turn_sent_at),
            critical_path: true,
            concurrent: false,
            payload,
        },
    );
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

async fn queue_append_agent_speech_with_media_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    initial_chunks: Vec<String>,
    config: TextCallSessionConfig,
    turn_finalized_at: Option<Instant>,
) -> anyhow::Result<(speech::AppendSpeechHandle, speech::QueuedSpeech)> {
    let media_ready_deadline = Instant::now() + config.media_ready_timeout;
    let playback_ready_deadline = Instant::now() + config.playback_wait_timeout;
    let conflict_policy = if config.latest_response_wins {
        SpeechConflictPolicy::CancelAndReplace
    } else {
        SpeechConflictPolicy::Reject
    };
    let text = initial_chunks.join(" ");
    loop {
        match speech::queue_append_speech_with_request(
            &services.state,
            &services.media,
            &services.tts,
            SpeechQueueRequest {
                tts_backend: LiveTtsBackend::default(),
                gateway_call_id: gateway_call_id.clone(),
                text: text.clone(),
                source_label: "text-call agent.turn.partial".to_string(),
                conflict_policy,
                turn_finalized_at,
            },
            initial_chunks.clone(),
        )
        .await
        {
            Ok(queued) => return Ok(queued),
            Err(error) => {
                let detail = format!("{error:#}");
                if detail.contains("media stream is not ready")
                    && Instant::now() < media_ready_deadline
                {
                    time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                if !config.latest_response_wins
                    && detail.contains("active speech job")
                    && Instant::now() < playback_ready_deadline
                {
                    time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

async fn queue_agent_speech_with_media_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
    config: TextCallSessionConfig,
    turn_finalized_at: Option<Instant>,
) -> anyhow::Result<speech::QueuedSpeech> {
    let media_ready_deadline = Instant::now() + config.media_ready_timeout;
    let playback_ready_deadline = Instant::now() + config.playback_wait_timeout;
    let conflict_policy = if config.latest_response_wins {
        SpeechConflictPolicy::CancelAndReplace
    } else {
        SpeechConflictPolicy::Reject
    };
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
                conflict_policy,
                turn_finalized_at,
            },
        )
        .await
        {
            Ok(queued) => return Ok(queued),
            Err(error) => {
                let detail = format!("{error:#}");
                if detail.contains("media stream is not ready")
                    && Instant::now() < media_ready_deadline
                {
                    time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                if !config.latest_response_wins
                    && detail.contains("active speech job")
                    && Instant::now() < playback_ready_deadline
                {
                    time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

pub async fn queue_fallback_and_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
) -> anyhow::Result<()> {
    let config = {
        let guard = services.state.read().await;
        TextCallSessionConfig::from(&guard.quality.config.text_call)
    };
    let queued =
        queue_agent_speech_with_media_wait(services, gateway_call_id.clone(), text, config, None)
            .await?;
    wait_for_playback_terminal_without_turn(
        &services.state,
        &gateway_call_id,
        &queued.playback_id,
        config.playback_wait_timeout,
    )
    .await;
    Ok(())
}

async fn wait_for_playback_terminal_without_turn(
    state: &SharedState,
    gateway_call_id: &str,
    playback_id: &str,
    playback_wait_timeout: Duration,
) {
    let deadline = Instant::now() + playback_wait_timeout;
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
    playback_wait_timeout: Duration,
) -> Option<PlaybackFinishedStatus> {
    let deadline = Instant::now() + playback_wait_timeout;
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
            .send_caller_turn("missing-call", "hello".to_string(), Instant::now())
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
            .send_caller_turn("call-test", "first".to_string(), Instant::now())
            .await
            .expect("first turn should send")
            .expect("first turn id");
        let second = registry
            .send_caller_turn("call-test", "second".to_string(), Instant::now())
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
            Some(GatewayTextFrame::TurnSuperseded { turn_id, superseded_by_turn_id, reason, .. })
                if turn_id == first && superseded_by_turn_id == second && reason == "new_caller_turn"
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
            for index in 0..DEFAULT_MAX_ACTIVE_TEXT_CALL_TURNS {
                turns.turns.insert(
                    format!("turn-preexisting-{index}"),
                    TextCallTurnState::Pending {
                        timing: test_turn_timing(),
                    },
                );
            }
        }
        registry
            .insert("call-test".to_string(), handle.clone())
            .await;

        let error = registry
            .send_caller_turn("call-test", "overflow".to_string(), Instant::now())
            .await
            .expect_err("turn cap should reject new caller turns");

        assert!(format!("{error:#}").contains("too many outstanding text-call turns"));
        assert!(rx.try_recv().is_err());
        assert_eq!(
            handle.turns.lock().await.outstanding_len(),
            DEFAULT_MAX_ACTIVE_TEXT_CALL_TURNS
        );
    }

    #[test]
    fn turn_tracker_reports_older_pending_turns_as_superseded() {
        let mut tracker = TextCallTurnTracker::default();
        tracker
            .add_caller_turn("turn-old".to_string(), Instant::now())
            .expect("old turn accepted");
        tracker
            .add_caller_turn("turn-new".to_string(), Instant::now())
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
        assert!(matches!(
            tracker.accept_agent_turn("turn-new"),
            AgentTurnDisposition::Accepted { .. }
        ));
    }

    #[test]
    fn turn_tracker_closes_replaced_playback_once() {
        let mut tracker = TextCallTurnTracker::default();
        tracker
            .add_caller_turn("turn-old".to_string(), Instant::now())
            .expect("turn accepted");
        assert!(matches!(
            tracker.accept_agent_turn("turn-old"),
            AgentTurnDisposition::Accepted { .. }
        ));
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

    #[tokio::test]
    async fn caller_turn_try_send_fails_fast_when_outbound_queue_is_full() {
        let registry = SharedTextCallRegistry::default();
        let (tx, _rx) = mpsc::channel(1);
        let handle = test_handle(tx);
        registry
            .insert("call-test".to_string(), handle.clone())
            .await;

        registry
            .send_caller_turn("call-test", "first".to_string(), Instant::now())
            .await
            .expect("first turn should queue")
            .expect("first turn id");
        let error = registry
            .send_caller_turn("call-test", "second".to_string(), Instant::now())
            .await
            .expect_err("full websocket queue should fail without awaiting");

        assert!(format!("{error:#}").contains("text-call outbound queue full"));
        assert_eq!(handle.turns.lock().await.outstanding_len(), 1);
    }

    fn test_turn_timing() -> TextCallTurnTiming {
        let now = Instant::now();
        TextCallTurnTiming {
            finalized_at: now,
            caller_turn_sent_at: now,
        }
    }

    fn test_handle(tx: mpsc::Sender<GatewayTextFrame>) -> TextCallSessionHandle {
        TextCallSessionHandle {
            tx,
            sequence: Arc::new(AtomicU64::new(1)),
            turns: Arc::new(Mutex::new(TextCallTurnTracker::default())),
            append_turns: Arc::new(Mutex::new(BTreeMap::new())),
            config: TextCallSessionConfig::from(&TextCallQualityConfig::default()),
        }
    }
}
