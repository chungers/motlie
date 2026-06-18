use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use crate::call_control::TelnyxClient;
use crate::early_response::{AppendOrReplace, EarlyResponseCancelReason, EarlyResponseEvent};
use crate::media::{SharedMediaRegistry, SpeechClearReason};
use crate::operator::state::{CallStatus, LogLevel, SharedState, SpeechOutputConfig};
use crate::processors::ConversationProcessorKind;
use crate::quality::TextCallQualityConfig;
use crate::tts::SharedTtsRegistry;

use super::offers::validate_call_url;
use super::turns::{
    CallerSpeechState, DebugTextStreamFrame, GatewayTextFrame, PlaybackFinishedStatus,
    TextCallDirection, TEXT_CALL_EARLY_TURNS_EXTENSION, TEXT_CALL_PARTIALS_EXTENSION,
    TEXT_CALL_PROTOCOL,
};

const OUTBOUND_TEXT_FRAME_CAPACITY: usize = 64;
const DEFAULT_MAX_ACTIVE_TEXT_CALL_TURNS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TextCallTurnTiming {
    pub(crate) finalized_at: Instant,
    pub(crate) caller_turn_sent_at: Instant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TextCallTurnState {
    Pending {
        timing: TextCallTurnTiming,
    },
    Superseded {
        timing: TextCallTurnTiming,
    },
    Playing {
        playback_id: String,
        timing: TextCallTurnTiming,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AgentTurnDisposition {
    Accepted { timing: TextCallTurnTiming },
    Superseded,
    Invalid,
}

pub(crate) struct AgentAppendTurn {
    pub(crate) packer: crate::tts::StreamingSpeechTextPacker,
    pub(crate) speech: Option<crate::speech::AppendSpeechHandle>,
    pub(crate) timing: TextCallTurnTiming,
}

pub(crate) struct AgentProvisionalTurn {
    pub(crate) generation: u64,
    pub(crate) packer: crate::tts::StreamingSpeechTextPacker,
    pub(crate) speech: Option<crate::speech::AppendSpeechHandle>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TextCallSessionConfig {
    pub(crate) max_active_turns: usize,
    pub(crate) media_ready_timeout: Duration,
    pub(crate) playback_wait_timeout: Duration,
    pub(crate) latest_response_wins: bool,
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
pub(crate) struct TextCallTurnTracker {
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

    pub(crate) fn append_turn_disposition(&self, turn_id: &str) -> AgentTurnDisposition {
        match self.turns.get(turn_id).cloned() {
            Some(TextCallTurnState::Pending { timing })
            | Some(TextCallTurnState::Playing { timing, .. }) => {
                AgentTurnDisposition::Accepted { timing }
            }
            Some(TextCallTurnState::Superseded { .. }) => AgentTurnDisposition::Superseded,
            None => AgentTurnDisposition::Invalid,
        }
    }

    pub(crate) fn accept_agent_turn(&mut self, turn_id: &str) -> AgentTurnDisposition {
        let disposition = self.agent_turn_disposition(turn_id);
        if matches!(
            disposition,
            AgentTurnDisposition::Accepted { .. } | AgentTurnDisposition::Superseded
        ) {
            self.turns.remove(turn_id);
        }
        disposition
    }

    pub(crate) fn start_playback(
        &mut self,
        turn_id: String,
        playback_id: String,
        timing: TextCallTurnTiming,
    ) {
        if let Some(TextCallTurnState::Playing {
            playback_id: previous_playback_id,
            ..
        }) = self.turns.insert(
            turn_id.clone(),
            TextCallTurnState::Playing {
                playback_id: playback_id.clone(),
                timing,
            },
        ) {
            self.playback_turns.remove(&previous_playback_id);
        }
        self.playback_turns.insert(playback_id, turn_id);
    }

    pub(crate) fn close_playback(&mut self, playback_id: &str) -> Option<String> {
        let turn_id = self.playback_turns.remove(playback_id)?;
        match self.turns.get(&turn_id) {
            Some(TextCallTurnState::Playing {
                playback_id: active,
                ..
            }) if active == playback_id => {
                self.turns.remove(&turn_id);
                Some(turn_id)
            }
            _ => None,
        }
    }

    pub(crate) fn close_superseded(&mut self, turn_id: &str) -> bool {
        if matches!(
            self.turns.get(turn_id),
            Some(TextCallTurnState::Superseded { .. })
        ) {
            self.turns.remove(turn_id);
            true
        } else {
            false
        }
    }

    pub(crate) fn remove_turn(&mut self, turn_id: &str) {
        if let Some(TextCallTurnState::Playing { playback_id, .. }) = self.turns.remove(turn_id) {
            self.playback_turns.remove(&playback_id);
        }
    }

    pub(crate) fn is_playback_active(&self, playback_id: &str) -> bool {
        self.playback_turns.contains_key(playback_id)
    }

    pub(crate) fn clear(&mut self) {
        self.turns.clear();
        self.playback_turns.clear();
    }

    #[cfg(test)]
    fn outstanding_len(&self) -> usize {
        self.turns.len()
    }
}

#[derive(Clone)]
pub struct SharedTextCallRegistry {
    inner: Arc<Mutex<BTreeMap<String, TextCallRegistryEntry>>>,
    next_owner: Arc<AtomicU64>,
}

impl Default for SharedTextCallRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(BTreeMap::new())),
            next_owner: Arc::new(AtomicU64::new(1)),
        }
    }
}

impl SharedTextCallRegistry {
    async fn claim(
        &self,
        gateway_call_id: String,
        handle: TextCallSessionHandle,
    ) -> anyhow::Result<TextCallSessionOwner> {
        let mut guard = self.inner.lock().await;
        if guard.contains_key(&gateway_call_id) {
            anyhow::bail!("text-call stream already attached for {gateway_call_id}");
        }
        let owner = TextCallSessionOwner(self.next_owner.fetch_add(1, Ordering::SeqCst));
        guard.insert(gateway_call_id, TextCallRegistryEntry { owner, handle });
        Ok(owner)
    }

    #[cfg(test)]
    pub(crate) async fn insert_test_session(
        &self,
        gateway_call_id: impl Into<String>,
    ) -> mpsc::Receiver<GatewayTextFrame> {
        self.insert_test_session_with_partials(gateway_call_id, false)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn insert_test_session_with_partials(
        &self,
        gateway_call_id: impl Into<String>,
        emit_partials: bool,
    ) -> mpsc::Receiver<GatewayTextFrame> {
        self.insert_test_session_with_options(gateway_call_id, emit_partials, false)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn insert_test_session_with_options(
        &self,
        gateway_call_id: impl Into<String>,
        emit_partials: bool,
        emit_early_turns: bool,
    ) -> mpsc::Receiver<GatewayTextFrame> {
        let (tx, rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = TextCallSessionHandle {
            tx,
            sequence: Arc::new(AtomicU64::new(1)),
            turns: Arc::new(Mutex::new(TextCallTurnTracker::default())),
            append_turns: Arc::new(Mutex::new(BTreeMap::new())),
            provisional_turns: Arc::new(Mutex::new(BTreeMap::new())),
            config: TextCallSessionConfig::from(&TextCallQualityConfig::default()),
            speech_output: SpeechOutputConfig::default(),
            emit_partials,
            emit_early_turns,
        };
        self.claim(gateway_call_id.into(), handle)
            .await
            .expect("test session should claim text-call registry slot");
        rx
    }

    async fn remove_owner(&self, gateway_call_id: &str, owner: TextCallSessionOwner) -> bool {
        let mut guard = self.inner.lock().await;
        let Some(entry) = guard.get(gateway_call_id) else {
            return false;
        };
        if entry.owner != owner {
            return false;
        }
        guard.remove(gateway_call_id);
        true
    }

    pub async fn contains(&self, gateway_call_id: &str) -> bool {
        self.inner.lock().await.contains_key(gateway_call_id)
    }

    pub async fn send_caller_partial(
        &self,
        gateway_call_id: &str,
        utterance_id: String,
        text: String,
        confidence: Option<f32>,
        stability: Option<f32>,
        speech_state: CallerSpeechState,
    ) -> anyhow::Result<bool> {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
        let Some(handle) = handle else {
            return Ok(false);
        };
        if !handle.emit_partials {
            return Ok(false);
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return Ok(false);
        }
        handle.try_send(GatewayTextFrame::CallerPartial {
            utterance_id,
            sequence: handle.next_sequence(),
            text,
            confidence: Self::normalized_score(confidence),
            stability: Self::normalized_score(stability),
            speech_state,
            reply_allowed: false,
        })?;
        Ok(true)
    }

    pub async fn send_early_response_event(
        &self,
        gateway_call_id: &str,
        event: EarlyResponseEvent,
    ) -> anyhow::Result<bool> {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
        let Some(handle) = handle else {
            return Ok(false);
        };
        if !handle.emit_early_turns {
            return Ok(false);
        }
        match event {
            EarlyResponseEvent::Started {
                provisional_turn_id,
                utterance_id,
                generation,
                text,
                confidence,
                stability,
                speech_state,
                ..
            } => handle.try_send(GatewayTextFrame::CallerTurnProvisional {
                provisional_turn_id,
                utterance_id,
                generation,
                sequence: handle.next_sequence(),
                text,
                confidence: Self::normalized_score(confidence),
                stability: Self::normalized_score(stability),
                speech_state,
                reply_allowed: true,
            })?,
            EarlyResponseEvent::Updated {
                provisional_turn_id,
                utterance_id,
                generation,
                text,
                append_or_replace,
                ..
            } => handle.try_send(GatewayTextFrame::CallerTurnProvisionalUpdate {
                provisional_turn_id,
                utterance_id,
                generation,
                sequence: handle.next_sequence(),
                text,
                append_or_replace: append_or_replace_label(append_or_replace).to_string(),
            })?,
            EarlyResponseEvent::Canceled {
                provisional_turn_id,
                utterance_id,
                generation,
                reason,
                ..
            } => handle.try_send(GatewayTextFrame::CallerTurnProvisionalCancel {
                provisional_turn_id,
                utterance_id,
                generation,
                sequence: handle.next_sequence(),
                reason: early_cancel_reason_label(reason).to_string(),
            })?,
            EarlyResponseEvent::Committed {
                provisional_turn_id,
                utterance_id,
                generation,
                turn_id,
                coalesced_utterance_ids,
                final_text,
                ..
            } => handle.try_send(GatewayTextFrame::CallerTurnProvisionalCommit {
                provisional_turn_id,
                turn_id,
                utterance_id,
                coalesced_utterance_ids,
                generation,
                sequence: handle.next_sequence(),
                final_text,
            })?,
        }
        Ok(true)
    }

    pub async fn cancel_agent_provisional_turn(
        &self,
        media: &SharedMediaRegistry,
        gateway_call_id: &str,
        provisional_turn_id: &str,
        generation: u64,
        reason: SpeechClearReason,
    ) -> bool {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
        let Some(handle) = handle else {
            return false;
        };
        crate::processors::external_text::cancel_agent_provisional_turn(
            media,
            &handle,
            gateway_call_id,
            provisional_turn_id,
            generation,
            reason,
        )
        .await
    }

    pub async fn finish_agent_provisional_turn(
        &self,
        gateway_call_id: &str,
        provisional_turn_id: &str,
        generation: u64,
    ) -> bool {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
        let Some(handle) = handle else {
            return false;
        };
        crate::processors::external_text::finish_agent_provisional_turn(
            &handle,
            provisional_turn_id,
            generation,
        )
        .await
    }

    pub async fn send_early_response_playback_started(
        &self,
        gateway_call_id: &str,
        provisional_turn_id: String,
        generation: u64,
        playback_id: String,
    ) -> anyhow::Result<bool> {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
        let Some(handle) = handle else {
            return Ok(false);
        };
        if !handle.emit_early_turns {
            return Ok(false);
        }
        handle.try_send(GatewayTextFrame::ProvisionalPlaybackStarted {
            provisional_turn_id,
            generation,
            playback_id,
            sequence: handle.next_sequence(),
        })?;
        Ok(true)
    }

    fn normalized_score(score: Option<f32>) -> Option<f32> {
        score.filter(|score| score.is_finite() && *score >= 0.0 && *score <= 1.0)
    }

    pub async fn send_caller_turn(
        &self,
        gateway_call_id: &str,
        text: String,
        finalized_at: Instant,
    ) -> anyhow::Result<Option<String>> {
        self.send_caller_turn_with_utterance(gateway_call_id, text, finalized_at, None)
            .await
    }

    pub async fn send_caller_turn_with_utterance(
        &self,
        gateway_call_id: &str,
        text: String,
        finalized_at: Instant,
        utterance_id: Option<String>,
    ) -> anyhow::Result<Option<String>> {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
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
            utterance_id,
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
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
        let Some(handle) = handle else {
            return false;
        };
        let turn_id = handle.turns.lock().await.close_playback(playback_id);
        let Some(turn_id) = turn_id else {
            return false;
        };
        crate::processors::external_text::cancel_append_turn(&handle, &turn_id).await;
        let _ = crate::processors::external_text::send_playback_finished(&handle, turn_id, status)
            .await;
        true
    }

    pub async fn send_session_end(&self, gateway_call_id: &str, reason: impl Into<String>) {
        let handle = {
            self.inner
                .lock()
                .await
                .get(gateway_call_id)
                .map(|entry| entry.handle.clone())
        };
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextCallSessionOwner(u64);

#[derive(Clone)]
struct TextCallRegistryEntry {
    owner: TextCallSessionOwner,
    handle: TextCallSessionHandle,
}

#[derive(Clone)]
pub(crate) struct TextCallSessionHandle {
    pub(crate) tx: mpsc::Sender<GatewayTextFrame>,
    pub(crate) sequence: Arc<AtomicU64>,
    pub(crate) turns: Arc<Mutex<TextCallTurnTracker>>,
    pub(crate) append_turns: Arc<Mutex<BTreeMap<String, AgentAppendTurn>>>,
    pub(crate) provisional_turns: Arc<Mutex<BTreeMap<String, AgentProvisionalTurn>>>,
    pub(crate) config: TextCallSessionConfig,
    pub(crate) speech_output: SpeechOutputConfig,
    pub(crate) emit_partials: bool,
    pub(crate) emit_early_turns: bool,
}

impl TextCallSessionHandle {
    pub(crate) fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    pub(crate) async fn send(&self, frame: GatewayTextFrame) -> anyhow::Result<()> {
        self.tx
            .send(frame)
            .await
            .context("send text-call frame to websocket task")
    }

    pub(crate) fn try_send(&self, frame: GatewayTextFrame) -> anyhow::Result<()> {
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
    pub emit_partials: bool,
    pub emit_early_turns: bool,
}

#[derive(Clone, Debug)]
pub struct DebugTextCallSetup {
    pub gateway_call_id: String,
    pub direction: TextCallDirection,
    pub emit_partials: bool,
    pub emit_early_turns: bool,
}

fn text_call_session_handle(
    session_config: TextCallSessionConfig,
    speech_output: SpeechOutputConfig,
    emit_partials: bool,
    emit_early_turns: bool,
) -> (TextCallSessionHandle, mpsc::Receiver<GatewayTextFrame>) {
    let (tx, rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
    let handle = TextCallSessionHandle {
        tx,
        sequence: Arc::new(AtomicU64::new(1)),
        turns: Arc::new(Mutex::new(TextCallTurnTracker::new(
            session_config.max_active_turns,
        ))),
        append_turns: Arc::new(Mutex::new(BTreeMap::new())),
        provisional_turns: Arc::new(Mutex::new(BTreeMap::new())),
        config: session_config,
        speech_output,
        emit_partials,
        emit_early_turns,
    };
    (handle, rx)
}

pub(crate) async fn text_call_session_config(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
) -> (TextCallSessionConfig, SpeechOutputConfig) {
    let guard = services.state.read().await;
    let session_config = TextCallSessionConfig::from(&guard.quality.config.text_call);
    let speech_output = guard
        .calls
        .get(gateway_call_id)
        .map(|call| call.speech_output)
        .unwrap_or_else(|| {
            SpeechOutputConfig::from_quality(
                guard.conversation_tts_backend,
                &guard.quality.config.tts,
            )
        });
    (session_config, speech_output)
}

async fn mark_external_text_stream_processor(state: &SharedState, gateway_call_id: &str) {
    state.write().await.set_conversation_processor(
        gateway_call_id,
        ConversationProcessorKind::ExternalTextStream,
    );
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
    let (session_config, speech_output) =
        text_call_session_config(&services, &setup.gateway_call_id).await;
    let (handle, rx) = text_call_session_handle(
        session_config,
        speech_output,
        setup.emit_partials,
        setup.emit_early_turns,
    );
    let owner = services
        .registry
        .claim(setup.gateway_call_id.clone(), handle.clone())
        .await?;
    mark_external_text_stream_processor(&services.state, &setup.gateway_call_id).await;

    if let Err(error) = send_json_frame(&mut write, &start).await {
        services
            .registry
            .remove_owner(&setup.gateway_call_id, owner)
            .await;
        return Err(error);
    }

    tokio::spawn(run_text_call_session(
        services,
        setup.gateway_call_id,
        owner,
        handle,
        read,
        write,
        rx,
    ));
    Ok(())
}

pub async fn run_debug_text_stream<R, W>(
    services: TextCallStreamServices,
    setup: DebugTextCallSetup,
    read: R,
    write: W,
) -> anyhow::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let (session_config, speech_output) =
        text_call_session_config(&services, &setup.gateway_call_id).await;
    let (handle, rx) = text_call_session_handle(
        session_config,
        speech_output,
        setup.emit_partials,
        setup.emit_early_turns,
    );
    let owner = services
        .registry
        .claim(setup.gateway_call_id.clone(), handle.clone())
        .await?;
    mark_external_text_stream_processor(&services.state, &setup.gateway_call_id).await;

    let result = run_debug_text_call_session(
        services.clone(),
        setup.gateway_call_id.clone(),
        setup.direction,
        handle.clone(),
        read,
        write,
        rx,
    )
    .await;

    handle.append_turns.lock().await.clear();
    handle.provisional_turns.lock().await.clear();
    handle.turns.lock().await.clear();
    services
        .registry
        .remove_owner(&setup.gateway_call_id, owner)
        .await;
    result
}

async fn run_debug_text_call_session<R, W>(
    services: TextCallStreamServices,
    gateway_call_id: String,
    direction: TextCallDirection,
    handle: TextCallSessionHandle,
    mut read: R,
    mut write: W,
    mut rx: mpsc::Receiver<GatewayTextFrame>,
) -> anyhow::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    write_json_line(
        &mut write,
        &DebugTextStreamFrame::attached_with_extensions(
            gateway_call_id.clone(),
            handle
                .emit_partials
                .then(|| TEXT_CALL_PARTIALS_EXTENSION.to_string())
                .into_iter()
                .chain(
                    handle
                        .emit_early_turns
                        .then(|| TEXT_CALL_EARLY_TURNS_EXTENSION.to_string()),
                )
                .collect(),
        ),
    )
    .await?;
    write_json_line(
        &mut write,
        &GatewayTextFrame::SessionStart {
            protocol: TEXT_CALL_PROTOCOL.to_string(),
            call_id: gateway_call_id.clone(),
            direction,
        },
    )
    .await?;

    let mut line = String::new();
    loop {
        tokio::select! {
            frame = rx.recv() => {
                let Some(frame) = frame else {
                    break;
                };
                let gateway_closed = matches!(frame, GatewayTextFrame::SessionEnd { .. });
                write_json_line(&mut write, &frame).await?;
                if gateway_closed {
                    break;
                }
            }
            read = read.read_line(&mut line) => {
                let read = read?;
                if read == 0 {
                    break;
                }
                let message = line.trim();
                if message.is_empty() {
                    line.clear();
                    continue;
                }
                if let Ok(debug) = serde_json::from_str::<DebugTextStreamFrame>(message) {
                    match debug {
                        DebugTextStreamFrame::Detach { reason } => {
                            write_json_line(
                                &mut write,
                                &DebugTextStreamFrame::detached(
                                    reason.unwrap_or_else(|| "debug.detach".to_string()),
                                ),
                            )
                            .await?;
                            break;
                        }
                        DebugTextStreamFrame::Attach { .. } => {
                            write_json_line(
                                &mut write,
                                &DebugTextStreamFrame::error(
                                    "already_attached",
                                    "debug stream is already attached",
                                ),
                            )
                            .await?;
                        }
                        DebugTextStreamFrame::Attached { .. }
                        | DebugTextStreamFrame::Detached { .. }
                        | DebugTextStreamFrame::Error { .. } => {
                            write_json_line(
                                &mut write,
                                &DebugTextStreamFrame::error(
                                    "invalid_debug_frame",
                                    "client debug stream frame is not valid in stream mode",
                                ),
                            )
                            .await?;
                        }
                    }
                    line.clear();
                    continue;
                }

                if let Err(error) =
                    crate::processors::external_text::handle_agent_message(&services, &gateway_call_id, &handle, message).await
                {
                    log_text_call_error(&services.state, &gateway_call_id, error).await;
                    write_json_line(
                        &mut write,
                        &DebugTextStreamFrame::error(
                            "protocol_error",
                            "invalid text-call agent frame",
                        ),
                    )
                    .await?;
                }
                line.clear();
            }
        }
    }
    Ok(())
}

async fn run_text_call_session<W, R>(
    services: TextCallStreamServices,
    gateway_call_id: String,
    owner: TextCallSessionOwner,
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
                        if let Err(error) = crate::processors::external_text::handle_agent_message(
                            &services,
                            &gateway_call_id,
                            &handle,
                            text.as_str(),
                        ).await {
                            log_text_call_error(&services.state, &gateway_call_id, error).await;
                            let _ = crate::processors::external_text::send_error_frame(&handle, "protocol_error", "invalid text-call message").await;
                            let _ = hangup_gateway_call(&services, &gateway_call_id, "text-call protocol error").await;
                            break;
                        }
                    }
                    Some(Ok(Message::Binary(_))) => {
                        let _ = crate::processors::external_text::send_error_frame(&handle, "binary_not_allowed", "text-call protocol accepts JSON text frames only").await;
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
    handle.provisional_turns.lock().await.clear();
    handle.turns.lock().await.clear();
    services
        .registry
        .remove_owner(&gateway_call_id, owner)
        .await;
    if !gateway_closed {
        let _ =
            hangup_gateway_call(&services, &gateway_call_id, "text-call websocket closed").await;
    }
}

pub async fn queue_fallback_and_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
) -> anyhow::Result<()> {
    crate::processors::external_text::queue_fallback_and_wait(services, gateway_call_id, text).await
}

pub fn append_or_replace_label(value: AppendOrReplace) -> &'static str {
    match value {
        AppendOrReplace::Append => "append",
        AppendOrReplace::Replace => "replace",
    }
}

fn early_cancel_reason_label(reason: EarlyResponseCancelReason) -> &'static str {
    match reason {
        EarlyResponseCancelReason::AsrCorrection => "asr_correction",
        EarlyResponseCancelReason::FinalTranscriptMismatch => "final_transcript_mismatch",
        EarlyResponseCancelReason::SupersededByNewGeneration => "superseded_by_new_generation",
        EarlyResponseCancelReason::PolicyDisabled => "policy_disabled",
        EarlyResponseCancelReason::PolicyNoLongerSatisfied => "policy_no_longer_satisfied",
        EarlyResponseCancelReason::MaxUpdatesExceeded => "max_updates_exceeded",
        EarlyResponseCancelReason::CallerBargeIn => "caller_barge_in",
        EarlyResponseCancelReason::ProcessorRejected => "processor_rejected",
        EarlyResponseCancelReason::TtsCanceled => "tts_canceled",
        EarlyResponseCancelReason::StaleGeneration => "stale_generation",
        EarlyResponseCancelReason::CoalescedIntoFinalTurn => "coalesced_into_final_turn",
        EarlyResponseCancelReason::SessionEnded => "session_ended",
        EarlyResponseCancelReason::Hangup => "hangup",
    }
}

pub(crate) async fn hangup_gateway_call(
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
    guard.emit_quality_report_summary(gateway_call_id, "call_terminal");
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

async fn write_json_line<W, T>(write: &mut W, frame: &T) -> anyhow::Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let encoded = serde_json::to_string(frame).context("encode text-call JSONL frame")?;
    write.write_all(encoded.as_bytes()).await?;
    write.write_all(b"\n").await?;
    write.flush().await?;
    Ok(())
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
    use crate::processors::external_text::playback_finished_status;
    use crate::text_calls::turns::AgentTextFrame;
    use std::sync::Arc;
    use tokio::time;

    use crate::media::{OutboundMediaCommand, SharedMediaRegistry};
    use crate::operator::state::{shared_state, CallStatus, TelnyxIds, TtsPlaybackStatus};
    use crate::tts::{LiveTtsBackend, StaticTtsFactory, TtsRegistry};

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
    async fn registry_duplicate_attach_and_stale_debug_detach_do_not_remove_app_session() {
        let registry = SharedTextCallRegistry::default();
        let (debug_tx, mut debug_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let debug_handle = test_handle(debug_tx);
        let (app_tx, mut app_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let app_handle = test_handle(app_tx);

        let debug_owner = registry
            .claim("call-test".to_string(), debug_handle)
            .await
            .expect("debug attach should claim first");
        let duplicate_error = registry
            .claim("call-test".to_string(), app_handle.clone())
            .await
            .expect_err("app attach must not replace active debug owner");
        assert!(format!("{duplicate_error:#}").contains("already attached"));
        assert!(registry.contains("call-test").await);

        assert!(registry.remove_owner("call-test", debug_owner).await);
        let app_owner = registry
            .claim("call-test".to_string(), app_handle)
            .await
            .expect("app attach should claim after debug detaches");
        assert!(!registry.remove_owner("call-test", debug_owner).await);
        assert!(registry.contains("call-test").await);

        let turn_id = registry
            .send_caller_turn("call-test", "hello app".to_string(), Instant::now())
            .await
            .expect("app-owned session should accept caller turn")
            .expect("app-owned session should receive caller turn");
        assert!(matches!(
            app_rx.recv().await,
            Some(GatewayTextFrame::CallerTurn { turn_id: id, text, .. })
                if id == turn_id && text == "hello app"
        ));
        assert!(debug_rx.try_recv().is_err());
        assert!(registry.remove_owner("call-test", app_owner).await);
        assert!(!registry.contains("call-test").await);
    }

    #[tokio::test]
    async fn caller_partial_requires_opted_in_session() {
        let registry = SharedTextCallRegistry::default();
        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        registry
            .claim("call-default".to_string(), handle)
            .await
            .expect("default session should attach");

        let emitted = registry
            .send_caller_partial(
                "call-default",
                "utt-default".to_string(),
                "hello wor".to_string(),
                None,
                None,
                CallerSpeechState::Speaking,
            )
            .await
            .expect("default partial send should not fail");
        assert!(!emitted);
        assert!(rx.try_recv().is_err());

        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let mut handle = test_handle(tx);
        handle.emit_partials = true;
        registry
            .claim("call-partials".to_string(), handle)
            .await
            .expect("partial session should attach");

        let emitted = registry
            .send_caller_partial(
                "call-partials",
                "utt-partial".to_string(),
                "hello wor".to_string(),
                Some(0.84),
                Some(0.61),
                CallerSpeechState::Speaking,
            )
            .await
            .expect("partial send should succeed");
        assert!(emitted);
        assert!(matches!(
            rx.recv().await,
            Some(GatewayTextFrame::CallerPartial {
                utterance_id,
                text,
                confidence: Some(confidence),
                stability: Some(stability),
                speech_state: CallerSpeechState::Speaking,
                reply_allowed: false,
                ..
            }) if utterance_id == "utt-partial"
                && text == "hello wor"
                && (confidence - 0.84).abs() < f32::EPSILON
                && (stability - 0.61).abs() < f32::EPSILON
        ));
    }

    #[tokio::test]
    async fn early_response_events_require_opted_in_session() {
        let registry = SharedTextCallRegistry::default();
        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        registry
            .claim("call-default".to_string(), handle)
            .await
            .expect("default session should attach");
        let event = EarlyResponseEvent::Started {
            provisional_turn_id: "pt-1".to_string(),
            call_id: "call-default".to_string(),
            utterance_id: "utt-1".to_string(),
            generation: 1,
            text: "I need a tow truck.".to_string(),
            confidence: Some(0.91),
            stability: Some(0.86),
            speech_state: CallerSpeechState::EndpointCandidate,
        };
        let emitted = registry
            .send_early_response_event("call-default", event.clone())
            .await
            .expect("default early response send should not fail");
        assert!(!emitted);
        assert!(rx.try_recv().is_err());

        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let mut handle = test_handle(tx);
        handle.emit_early_turns = true;
        registry
            .claim("call-early".to_string(), handle)
            .await
            .expect("early-turn session should attach");
        let emitted = registry
            .send_early_response_event("call-early", event)
            .await
            .expect("early response send should succeed");
        assert!(emitted);
        assert!(matches!(
            rx.recv().await,
            Some(GatewayTextFrame::CallerTurnProvisional {
                provisional_turn_id,
                utterance_id,
                generation: 1,
                text,
                speech_state: CallerSpeechState::EndpointCandidate,
                ..
            }) if provisional_turn_id == "pt-1"
                && utterance_id == "utt-1"
                && text == "I need a tow truck."
        ));
    }

    #[tokio::test]
    async fn caller_turn_allows_multiple_outstanding_turns() {
        let registry = SharedTextCallRegistry::default();
        let (tx, mut rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        registry
            .claim("call-test".to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");

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
            .claim("call-test".to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");

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
        tracker.start_playback(
            "turn-old".to_string(),
            "tts-old".to_string(),
            test_turn_timing(),
        );

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
        handle.turns.lock().await.start_playback(
            "turn-old".to_string(),
            "tts-old".to_string(),
            test_turn_timing(),
        );

        let closed_turn =
            crate::processors::external_text::send_replaced_playback_canceled(&handle, "tts-old")
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
    async fn agent_turn_uses_text_call_session_tts_backend_snapshot() {
        let call_id = "gwc-backend-snapshot";
        let turn_id = "turn-backend-snapshot";
        let (services, mut media_rx) = test_services(call_id).await;
        {
            let mut guard = services.state.write().await;
            guard.conversation_tts_backend = LiveTtsBackend::Kokoro82m;
        }
        let (tx, mut outbound_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let mut handle = test_handle(tx);
        handle.speech_output.tts_backend = LiveTtsBackend::Piper;
        services
            .registry
            .claim(call_id.to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");
        handle
            .turns
            .lock()
            .await
            .add_caller_turn(turn_id.to_string(), Instant::now())
            .expect("caller turn should be tracked");

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurn {
                turn_id: turn_id.to_string(),
                text: "Use the session backend.".to_string(),
            },
        )
        .await;

        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::PlaybackStarted { turn_id: id, .. }) if id == turn_id
        ));
        let (playback_id, _frame_count) = collect_media_until_mark(&mut media_rx).await;
        let guard = services.state.read().await;
        let call = guard.calls.get(call_id).expect("call exists");
        let tts = call.tts.as_ref().expect("agent turn should queue TTS");
        assert_eq!(tts.playback_id, playback_id);
        assert_eq!(tts.backend, LiveTtsBackend::Piper);
    }

    #[tokio::test]
    async fn agent_provisional_turn_streams_through_session_speech_output() {
        let call_id = "gwc-provisional-stream";
        let provisional_turn_id = "pt-provisional-stream";
        let (services, mut media_rx) = test_services(call_id).await;
        {
            let mut guard = services.state.write().await;
            guard.conversation_tts_backend = LiveTtsBackend::Kokoro82m;
        }
        let (tx, mut outbound_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let mut handle = test_handle(tx);
        handle.speech_output.tts_backend = LiveTtsBackend::Piper;
        services
            .registry
            .claim(call_id.to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnProvisionalPartial {
                provisional_turn_id: provisional_turn_id.to_string(),
                generation: 7,
                text: "Early sentence.".to_string(),
                append: true,
            },
        )
        .await;
        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::ProvisionalPlaybackStarted {
                provisional_turn_id: id,
                generation: 7,
                ..
            }) if id == provisional_turn_id
        ));

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnProvisional {
                provisional_turn_id: provisional_turn_id.to_string(),
                generation: 7,
                text: " Final sentence.".to_string(),
            },
        )
        .await;

        let (playback_id, frame_count) = collect_media_until_mark(&mut media_rx).await;
        assert_eq!(frame_count, 2);
        let guard = services.state.read().await;
        let call = guard.calls.get(call_id).expect("call exists");
        let tts = call
            .tts
            .as_ref()
            .expect("provisional turn should queue TTS");
        assert_eq!(tts.playback_id, playback_id);
        assert_eq!(tts.backend, LiveTtsBackend::Piper);
    }

    #[tokio::test]
    async fn agent_provisional_cancel_clears_matching_playback() {
        let call_id = "gwc-provisional-cancel";
        let provisional_turn_id = "pt-provisional-cancel";
        let (services, _media_rx) = test_services(call_id).await;
        let (tx, mut outbound_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        services
            .registry
            .claim(call_id.to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnProvisionalPartial {
                provisional_turn_id: provisional_turn_id.to_string(),
                generation: 3,
                text: "Cancel me.".to_string(),
                append: true,
            },
        )
        .await;
        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::ProvisionalPlaybackStarted { .. })
        ));
        let playback_id = services
            .media
            .active_speech_playback_id(call_id)
            .await
            .expect("provisional speech should be active");

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnProvisional {
                provisional_turn_id: provisional_turn_id.to_string(),
                generation: 3,
                text: " Final fragment.".to_string(),
            },
        )
        .await;

        assert!(
            crate::processors::external_text::cancel_agent_provisional_turn(
                &services.media,
                &handle,
                call_id,
                provisional_turn_id,
                3,
                crate::media::SpeechClearReason::CancelAndReplace,
            )
            .await
        );
        assert_eq!(
            services.media.active_speech_playback_id(call_id).await,
            None
        );
        assert!(
            !crate::processors::external_text::cancel_agent_provisional_turn(
                &services.media,
                &handle,
                call_id,
                provisional_turn_id,
                3,
                crate::media::SpeechClearReason::CancelAndReplace,
            )
            .await
        );
        let guard = services.state.read().await;
        let call = guard.calls.get(call_id).expect("call exists");
        let tts = call
            .tts
            .as_ref()
            .expect("provisional turn should queue TTS");
        assert_eq!(tts.playback_id, playback_id);
    }

    #[tokio::test]
    async fn append_turn_streams_partials_and_terminal_on_one_playback() {
        let call_id = "gwc-stream";
        let turn_id = "turn-stream";
        let (services, mut media_rx) = test_services(call_id).await;
        let (tx, mut outbound_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        services
            .registry
            .claim(call_id.to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");
        handle
            .turns
            .lock()
            .await
            .add_caller_turn(turn_id.to_string(), Instant::now())
            .expect("caller turn should be tracked");

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnPartial {
                turn_id: turn_id.to_string(),
                text: "One.".to_string(),
                append: true,
            },
        )
        .await;
        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::PlaybackStarted { turn_id: id, .. }) if id == turn_id
        ));

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnPartial {
                turn_id: turn_id.to_string(),
                text: " Two.".to_string(),
                append: true,
            },
        )
        .await;
        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurn {
                turn_id: turn_id.to_string(),
                text: " Three.".to_string(),
            },
        )
        .await;

        let (playback_id, frame_count) = collect_media_until_mark(&mut media_rx).await;
        assert_eq!(frame_count, 3);
        mark_playback_completed(&services, call_id, &playback_id).await;

        let finished = time::timeout(Duration::from_secs(2), outbound_rx.recv())
            .await
            .expect("playback.finished should arrive")
            .expect("frame should exist");
        assert!(matches!(
            finished,
            GatewayTextFrame::PlaybackFinished {
                turn_id: id,
                status: PlaybackFinishedStatus::Completed,
                ..
            } if id == turn_id
        ));
        assert_eq!(handle.turns.lock().await.outstanding_len(), 0);
    }

    #[tokio::test]
    async fn stale_append_after_canceled_playback_does_not_error_out_session() {
        let call_id = "gwc-cancel";
        let turn_id = "turn-cancel";
        let (services, _media_rx) = test_services(call_id).await;
        let (tx, mut outbound_rx) = mpsc::channel(OUTBOUND_TEXT_FRAME_CAPACITY);
        let handle = test_handle(tx);
        services
            .registry
            .claim(call_id.to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");
        handle
            .turns
            .lock()
            .await
            .add_caller_turn(turn_id.to_string(), Instant::now())
            .expect("caller turn should be tracked");

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnPartial {
                turn_id: turn_id.to_string(),
                text: "One.".to_string(),
                append: true,
            },
        )
        .await;
        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::PlaybackStarted { turn_id: id, .. }) if id == turn_id
        ));
        let playback_id = services
            .media
            .active_speech_playback_id(call_id)
            .await
            .expect("append speech should be active");

        assert!(
            services
                .registry
                .finish_playback(call_id, &playback_id, PlaybackFinishedStatus::Canceled)
                .await
        );
        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::PlaybackFinished {
                turn_id: id,
                status: PlaybackFinishedStatus::Canceled,
                ..
            }) if id == turn_id
        ));

        send_agent_frame(
            &services,
            call_id,
            &handle,
            AgentTextFrame::AgentTurnPartial {
                turn_id: turn_id.to_string(),
                text: " Two.".to_string(),
                append: true,
            },
        )
        .await;
        assert!(matches!(
            outbound_rx.recv().await,
            Some(GatewayTextFrame::Error { code, .. }) if code == "invalid_turn"
        ));
        assert_eq!(handle.turns.lock().await.outstanding_len(), 0);
        let guard = services.state.read().await;
        assert_ne!(
            guard.calls.get(call_id).expect("call exists").status,
            CallStatus::Ended
        );
    }

    #[tokio::test]
    async fn caller_turn_try_send_fails_fast_when_outbound_queue_is_full() {
        let registry = SharedTextCallRegistry::default();
        let (tx, _rx) = mpsc::channel(1);
        let handle = test_handle(tx);
        registry
            .claim("call-test".to_string(), handle.clone())
            .await
            .expect("test session should claim registry slot");

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

    async fn test_services(
        call_id: &str,
    ) -> (TextCallStreamServices, mpsc::Receiver<OutboundMediaCommand>) {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            let generated_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: format!("cc-{call_id}"),
                    call_session_id: Some("sess-test".to_string()),
                    call_leg_id: Some("leg-test".to_string()),
                    stream_id: Some("stream-test".to_string()),
                },
                Some("<from-phone-number>".to_string()),
                Some("<to-phone-number>".to_string()),
                CallStatus::Answering,
            );
            if generated_call_id != call_id {
                let call = guard
                    .calls
                    .remove(&generated_call_id)
                    .expect("generated test call exists");
                guard.calls.insert(call_id.to_string(), call);
            }
        }
        let media = SharedMediaRegistry::default();
        let (media_tx, media_rx) = mpsc::channel(128);
        media.register_call(call_id.to_string(), media_tx).await;
        let tts = Arc::new(TtsRegistry::new(
            Arc::new(StaticTtsFactory::with_sample_rate(vec![1_000; 160], 8_000)),
            Arc::new(StaticTtsFactory::with_sample_rate(vec![1_000; 160], 8_000)),
        ));
        (
            TextCallStreamServices {
                registry: SharedTextCallRegistry::default(),
                state,
                media,
                tts,
                telnyx: TelnyxClient::new("http://127.0.0.1:1", None, true),
            },
            media_rx,
        )
    }

    async fn send_agent_frame(
        services: &TextCallStreamServices,
        call_id: &str,
        handle: &TextCallSessionHandle,
        frame: AgentTextFrame,
    ) {
        let encoded = serde_json::to_string(&frame).expect("agent frame serializes");
        crate::processors::external_text::handle_agent_message(services, call_id, handle, &encoded)
            .await
            .expect("agent frame should be handled without protocol error");
    }

    async fn collect_media_until_mark(
        media_rx: &mut mpsc::Receiver<OutboundMediaCommand>,
    ) -> (String, usize) {
        let mut playback_id = None;
        let mut frames = 0usize;
        loop {
            let command = time::timeout(Duration::from_secs(2), media_rx.recv())
                .await
                .expect("media command should arrive")
                .expect("media channel should stay open");
            match command {
                OutboundMediaCommand::Frame(frame) => {
                    if let Some(existing) = playback_id.as_deref() {
                        assert_eq!(existing, frame.playback_id);
                    } else {
                        playback_id = Some(frame.playback_id.clone());
                    }
                    frames = frames.saturating_add(1);
                }
                OutboundMediaCommand::Mark { playback_id: mark } => {
                    let playback_id = playback_id.unwrap_or_else(|| mark.clone());
                    assert_eq!(playback_id, mark);
                    return (playback_id, frames);
                }
                OutboundMediaCommand::AppendState { .. } => {}
                OutboundMediaCommand::Clear { .. } => panic!("append flow should not clear media"),
            }
        }
    }

    async fn mark_playback_completed(
        services: &TextCallStreamServices,
        call_id: &str,
        playback_id: &str,
    ) {
        {
            let mut guard = services.state.write().await;
            guard.mark_tts_mark_sent(call_id, playback_id, playback_id);
            guard.mark_tts_completed(call_id, playback_id);
        }
        services.media.finish_speech(call_id, playback_id).await;
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
            provisional_turns: Arc::new(Mutex::new(BTreeMap::new())),
            config: TextCallSessionConfig::from(&TextCallQualityConfig::default()),
            speech_output: SpeechOutputConfig::default(),
            emit_partials: false,
            emit_early_turns: false,
        }
    }
}
