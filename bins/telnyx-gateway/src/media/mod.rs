use anyhow::{bail, Context};
use axum::extract::ws::{Message, WebSocket};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use motlie_model::typed::{AudioBuf, Mono};
use motlie_voice::app::TranscriptEvent;
use motlie_voice::codec::{g711, l16};
use motlie_voice::pipeline::reorder::{SequencedFrame, SequencedFrameReorder};
use motlie_voice::pipeline::resample::{resample_i16_mono, WindowedSincResampler};
use motlie_voice::VoiceError;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{self, MissedTickBehavior};

use crate::adapter::{
    AsrTranscriptEvent, AsrTranscriptSuppressionReason, InboundAsrSession, LiveAsrBackend,
    SharedAsrRegistry,
};
use crate::call_control::{TelnyxMediaConfig, TelnyxStreamCodec};
use crate::conversation::{self, ConversationRuntime};
use crate::operator::state::{
    CallStatus, LogLevel, MediaMetadata, SharedState, StreamAttachOutcome, TranscriptKind,
};
use crate::quality::{ActiveAsrQualitySession, RedactionMode, SpeechQualityConfig, VoiceQualityConfig};
use crate::text_calls::SharedTextCallRegistry;
use crate::tts::PIPER_SAMPLE_RATE_HZ;

mod capture;

use capture::MediaCapture;

const PCMU_SILENCE_BYTE: u8 = 0xff;
const PCMA_SILENCE_BYTE: u8 = 0xd5;
const SILENCE_KEEPALIVE_INTERVAL: Duration = Duration::from_millis(20);
const OUTBOUND_MEDIA_QUEUE_CAPACITY: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedMediaFrame {
    pub payload: Vec<u8>,
    pub track: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MediaFormat {
    encoding: String,
    sample_rate_hz: u32,
    channels: u16,
}

#[derive(Debug, Deserialize)]
struct EventDiscriminator {
    event: String,
}

#[derive(Debug, Deserialize)]
struct StartEvent {
    stream_id: String,
    start: StartPayload,
}

#[derive(Debug, Deserialize)]
struct StartPayload {
    call_control_id: String,
    call_session_id: Option<String>,
    media_format: Option<MediaFormatPayload>,
}

#[derive(Clone, Debug, Deserialize)]
struct MediaFormatPayload {
    encoding: Option<String>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct MediaEvent {
    stream_id: String,
    media: MediaPayload,
}

#[derive(Debug, Deserialize)]
struct MediaPayload {
    track: Option<String>,
    chunk: String,
    payload: String,
}

#[derive(Debug, Deserialize)]
struct StopEvent {
    stream_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MarkEvent {
    mark: MarkPayload,
}

#[derive(Debug, Deserialize)]
struct MarkPayload {
    name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct OutboundFrameQualityContext {
    pub config_id: String,
    pub redaction_mode: RedactionMode,
    pub queued_at: Instant,
    pub first_for_playback: bool,
}

#[derive(Clone, Debug)]
pub struct OutboundMediaFrame {
    pub playback_id: String,
    pub payload: Vec<u8>,
    pub quality: Option<OutboundFrameQualityContext>,
}

impl OutboundMediaFrame {
    #[cfg(test)]
    fn new(playback_id: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            playback_id: playback_id.into(),
            payload,
            quality: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechClearReason {
    Operator,
    BargeIn,
    CancelAndReplace,
    TtsFailed,
}

impl SpeechClearReason {
    fn label(self) -> &'static str {
        match self {
            Self::Operator => "operator",
            Self::BargeIn => "barge_in",
            Self::CancelAndReplace => "cancel_and_replace",
            Self::TtsFailed => "tts_failed",
        }
    }
}

#[derive(Clone, Debug)]
struct PendingClear {
    playback_id: String,
    requested_at: Instant,
    reason: SpeechClearReason,
}

#[derive(Clone, Debug)]
pub enum OutboundMediaCommand {
    Frame(OutboundMediaFrame),
    Clear {
        playback_id: String,
        requested_at: Instant,
        reason: SpeechClearReason,
    },
    Mark { playback_id: String },
}

impl OutboundMediaCommand {
    fn playback_id(&self) -> &str {
        match self {
            Self::Frame(frame) => &frame.playback_id,
            Self::Clear { playback_id, .. } | Self::Mark { playback_id } => playback_id,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpeechCancelToken {
    canceled: Arc<AtomicBool>,
}

impl SpeechCancelToken {
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::SeqCst)
    }
}

#[derive(Clone)]
pub struct CallMediaHandle {
    tx: mpsc::Sender<OutboundMediaCommand>,
}

impl CallMediaHandle {
    pub async fn send(&self, command: OutboundMediaCommand) -> anyhow::Result<()> {
        self.tx
            .send(command)
            .await
            .context("send outbound media command")
    }
}

#[derive(Clone, Debug)]
struct ActiveSpeechJob {
    playback_id: String,
    cancel: SpeechCancelToken,
}

#[derive(Clone)]
struct MediaRegistryEntry {
    tx: mpsc::Sender<OutboundMediaCommand>,
    active_speech: Option<ActiveSpeechJob>,
    pending_clears: VecDeque<PendingClear>,
}

#[derive(Clone, Default)]
pub struct SharedMediaRegistry {
    inner: Arc<Mutex<HashMap<String, MediaRegistryEntry>>>,
}

impl SharedMediaRegistry {
    pub async fn register_call(
        &self,
        gateway_call_id: String,
        tx: mpsc::Sender<OutboundMediaCommand>,
    ) {
        self.inner.lock().await.insert(
            gateway_call_id,
            MediaRegistryEntry {
                tx,
                active_speech: None,
                pending_clears: VecDeque::new(),
            },
        );
    }

    pub async fn unregister_call(&self, gateway_call_id: &str) {
        self.inner.lock().await.remove(gateway_call_id);
    }

    pub async fn start_speech(
        &self,
        gateway_call_id: &str,
        playback_id: String,
        cancel: SpeechCancelToken,
    ) -> anyhow::Result<CallMediaHandle> {
        let mut guard = self.inner.lock().await;
        let entry = guard
            .get_mut(gateway_call_id)
            .with_context(|| format!("media stream is not ready for call {gateway_call_id}"))?;
        if let Some(active) = &entry.active_speech {
            bail!(
                "active speech job {} already exists for call {}; run speak cancel first",
                active.playback_id,
                gateway_call_id
            );
        }
        entry.active_speech = Some(ActiveSpeechJob {
            playback_id,
            cancel,
        });
        Ok(CallMediaHandle {
            tx: entry.tx.clone(),
        })
    }

    pub async fn start_speech_replacing_active(
        &self,
        gateway_call_id: &str,
        playback_id: String,
        cancel: SpeechCancelToken,
    ) -> anyhow::Result<(CallMediaHandle, Option<String>)> {
        let mut guard = self.inner.lock().await;
        let entry = guard
            .get_mut(gateway_call_id)
            .with_context(|| format!("media stream is not ready for call {gateway_call_id}"))?;
        let replaced_playback_id = entry.active_speech.take().map(|active| {
            active.cancel.cancel();
            entry.pending_clears.push_back(PendingClear {
                playback_id: active.playback_id.clone(),
                requested_at: Instant::now(),
                reason: SpeechClearReason::CancelAndReplace,
            });
            active.playback_id
        });
        entry.active_speech = Some(ActiveSpeechJob {
            playback_id,
            cancel,
        });
        Ok((
            CallMediaHandle {
                tx: entry.tx.clone(),
            },
            replaced_playback_id,
        ))
    }

    pub async fn cancel_speech(&self, gateway_call_id: &str) -> anyhow::Result<String> {
        self.cancel_speech_for_reason(gateway_call_id, SpeechClearReason::Operator)
            .await
    }

    pub async fn cancel_speech_for_reason(
        &self,
        gateway_call_id: &str,
        reason: SpeechClearReason,
    ) -> anyhow::Result<String> {
        let mut guard = self.inner.lock().await;
        let entry = guard
            .get_mut(gateway_call_id)
            .with_context(|| format!("media stream is not ready for call {gateway_call_id}"))?;
        let active = entry
            .active_speech
            .take()
            .with_context(|| format!("no active speech job for call {gateway_call_id}"))?;
        active.cancel.cancel();
        entry.pending_clears.push_back(PendingClear {
            playback_id: active.playback_id.clone(),
            requested_at: Instant::now(),
            reason,
        });
        Ok(active.playback_id)
    }

    async fn take_pending_clear(&self, gateway_call_id: &str) -> Option<PendingClear> {
        self.inner
            .lock()
            .await
            .get_mut(gateway_call_id)?
            .pending_clears
            .pop_front()
    }

    pub async fn active_speech_playback_id(&self, gateway_call_id: &str) -> Option<String> {
        self.inner
            .lock()
            .await
            .get(gateway_call_id)?
            .active_speech
            .as_ref()
            .map(|active| active.playback_id.clone())
    }

    pub async fn finish_speech(&self, gateway_call_id: &str, playback_id: &str) {
        if let Some(entry) = self.inner.lock().await.get_mut(gateway_call_id) {
            if entry
                .active_speech
                .as_ref()
                .is_some_and(|active| active.playback_id == playback_id)
            {
                entry.active_speech = None;
            }
        }
    }
}

#[derive(Debug, Default)]
struct InboundTransportStats {
    packets_total: u64,
    lost_packets: u64,
    stale_frames: u64,
    reordered_frames: u64,
    max_sequence_seen: Option<u64>,
    last_arrival_at: Option<Instant>,
    jitter_samples_ms: Vec<u64>,
}

impl InboundTransportStats {
    fn observe_packet(&mut self, sequence: u64, expected_frame_ms: u64) {
        self.packets_total = self.packets_total.saturating_add(1);
        let now = Instant::now();
        if let Some(last_arrival_at) = self.last_arrival_at.replace(now) {
            let observed_ms = now.duration_since(last_arrival_at).as_millis() as u64;
            self.jitter_samples_ms
                .push(observed_ms.abs_diff(expected_frame_ms));
        }
        match self.max_sequence_seen {
            Some(max_seen) if sequence <= max_seen => {
                self.reordered_frames = self.reordered_frames.saturating_add(1);
            }
            Some(max_seen) => {
                if sequence > max_seen.saturating_add(1) {
                    self.lost_packets = self
                        .lost_packets
                        .saturating_add(sequence.saturating_sub(max_seen).saturating_sub(1));
                }
                self.max_sequence_seen = Some(sequence);
            }
            None => self.max_sequence_seen = Some(sequence),
        }
    }

    fn observe_stale(&mut self) {
        self.stale_frames = self.stale_frames.saturating_add(1);
    }

    fn rollup_payload(
        &self,
        session: Option<&ActiveAsrQualitySession>,
        format: Option<&MediaFormat>,
    ) -> Map<String, Value> {
        map_from_value(json!({
            "asr_session_id": session.map(|session| session.asr_session_id.as_str()),
            "utterance_id": session.map(|session| session.utterance_id.as_str()),
            "packets_total": self.packets_total,
            "lost_packets": self.lost_packets,
            "stale_frames": self.stale_frames,
            "reordered_frames": self.reordered_frames,
            "jitter_ms_p50": percentile_u64(&self.jitter_samples_ms, 50),
            "jitter_ms_p95": percentile_u64(&self.jitter_samples_ms, 95),
            "jitter_ms_max": self.jitter_samples_ms.iter().copied().max().unwrap_or(0),
            "codec": format.map(|format| format.encoding.as_str()),
            "sample_rate_hz": format.map(|format| format.sample_rate_hz),
            "frame_ms": 20,
        }))
    }
}

#[derive(Debug, Default)]
struct OutboundPacingStats {
    frames_sent: u64,
    underrun_count: u64,
    inter_frame_gap_ms: Vec<u64>,
    queue_depth_samples: Vec<u64>,
}

impl OutboundPacingStats {
    fn observe_frame(&mut self, interval_ms: Option<u64>, queue_depth: usize) {
        self.frames_sent = self.frames_sent.saturating_add(1);
        if let Some(interval_ms) = interval_ms {
            self.inter_frame_gap_ms.push(interval_ms);
        }
        self.queue_depth_samples.push(queue_depth as u64);
    }

    fn observe_underrun(&mut self) {
        self.underrun_count = self.underrun_count.saturating_add(1);
    }

    fn rollup_payload(&self, playback_id: Option<&str>) -> Map<String, Value> {
        map_from_value(json!({
            "playback_id": playback_id,
            "frames_sent": self.frames_sent,
            "underrun_count": self.underrun_count,
            "inter_frame_gap_ms_p50": percentile_u64(&self.inter_frame_gap_ms, 50),
            "inter_frame_gap_ms_p95": percentile_u64(&self.inter_frame_gap_ms, 95),
            "inter_frame_gap_ms_max": self.inter_frame_gap_ms.iter().copied().max().unwrap_or(0),
            "queue_depth_p50": percentile_u64(&self.queue_depth_samples, 50),
            "queue_depth_p95": percentile_u64(&self.queue_depth_samples, 95),
        }))
    }
}

struct MediaSocketState {
    session: Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<String>,
    asr_backend: Option<LiveAsrBackend>,
    media_format: Option<MediaFormat>,
    active_quality_asr: Option<ActiveAsrQualitySession>,
    last_quality_asr: Option<ActiveAsrQualitySession>,
    quality_config: VoiceQualityConfig,
    reorder: SequencedFrameReorder<EncodedMediaFrame>,
    inbound_transport: InboundTransportStats,
    decoded_frame_count: usize,
    asr_gate: AsrGate,
    silence_keepalive: bool,
    silence_keepalive_frames: usize,
    capture: Option<MediaCapture>,
    outbound_rx: Option<mpsc::Receiver<OutboundMediaCommand>>,
    outbound_pending: VecDeque<OutboundMediaCommand>,
    canceled_playbacks: HashSet<String>,
    media_registry: SharedMediaRegistry,
    conversation: Option<ConversationRuntime>,
    outbound_frame_count: usize,
    outbound_underrun_ticks: usize,
    outbound_pacing: OutboundPacingStats,
    last_outbound_frame_sent_at: Option<Instant>,
    first_frame_sent_playbacks: HashSet<String>,
    playback_started_at: HashMap<String, Instant>,
    playback_quality_contexts: HashMap<String, OutboundFrameQualityContext>,
    rollups_emitted: bool,
}

impl MediaSocketState {
    #[cfg(test)]
    fn new() -> Self {
        Self::with_media_registry(SharedMediaRegistry::default())
    }

    fn with_media_registry(media_registry: SharedMediaRegistry) -> Self {
        Self {
            session: None,
            gateway_call_id: None,
            asr_backend: None,
            media_format: None,
            active_quality_asr: None,
            last_quality_asr: None,
            quality_config: VoiceQualityConfig::default(),
            reorder: SequencedFrameReorder::new_lazily(32),
            inbound_transport: InboundTransportStats::default(),
            decoded_frame_count: 0,
            asr_gate: AsrGate::default(),
            silence_keepalive: false,
            silence_keepalive_frames: 0,
            capture: None,
            outbound_rx: None,
            outbound_pending: VecDeque::new(),
            canceled_playbacks: HashSet::new(),
            media_registry,
            conversation: None,
            outbound_frame_count: 0,
            outbound_underrun_ticks: 0,
            outbound_pacing: OutboundPacingStats::default(),
            last_outbound_frame_sent_at: None,
            first_frame_sent_playbacks: HashSet::new(),
            playback_started_at: HashMap::new(),
            playback_quality_contexts: HashMap::new(),
            rollups_emitted: false,
        }
    }

    fn with_conversation(
        media_registry: SharedMediaRegistry,
        conversation: ConversationRuntime,
    ) -> Self {
        let mut state = Self::with_media_registry(media_registry);
        state.conversation = Some(conversation);
        state
    }
}

pub async fn handle_socket(
    mut socket: WebSocket,
    state: SharedState,
    asr: SharedAsrRegistry,
    media_registry: SharedMediaRegistry,
    conversation: ConversationRuntime,
    text_calls: SharedTextCallRegistry,
) {
    let mut media_state = MediaSocketState::with_conversation(media_registry.clone(), conversation);
    let mut silence_keepalive = time::interval(SILENCE_KEEPALIVE_INTERVAL);
    silence_keepalive.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            message = socket.next() => {
                let Some(message) = message else {
                    break;
                };
                match message {
                    Ok(Message::Text(text)) => {
                        if let Err(error) = handle_text_with_text_calls(&text, &state, &asr, &mut media_state, Some(&text_calls)).await {
                            log_media_error(&state, media_state.gateway_call_id.as_deref(), error).await;
                        } else {
                            ensure_outbound_registered(&media_registry, &mut media_state).await;
                        }
                    }
                    Ok(Message::Close(_)) => break,
                    Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
                    Err(error) => {
                        log_media_error(
                            &state,
                            media_state.gateway_call_id.as_deref(),
                            anyhow::Error::from(error),
                        )
                        .await;
                        break;
                    }
                }
            }
            _ = silence_keepalive.tick(), if media_state.silence_keepalive => {
                if let Err(error) = send_outbound_or_silence(&mut socket, &state, &mut media_state).await {
                    log_media_error(&state, media_state.gateway_call_id.as_deref(), error).await;
                    break;
                }
            }
        }
    }

    if let Some(call_id) = media_state.gateway_call_id.clone() {
        if let Err(error) = finish_asr_session(
            &state,
            &mut media_state,
            Some(call_id.as_str()),
            None,
            Some(&text_calls),
        )
        .await
        {
            log_media_error(&state, Some(&call_id), error).await;
        }
        let mut guard = state.write().await;
        if let Some(call) = guard.calls.get_mut(&call_id) {
            call.status = CallStatus::Ended;
            call.push_timeline("media websocket closed");
        }
    }
    emit_quality_transport_rollups(&state, &mut media_state).await;
    finalize_capture(&mut media_state).await;
    if let Some(call_id) = media_state.gateway_call_id.as_deref() {
        media_registry.unregister_call(call_id).await;
    }
}

async fn ensure_outbound_registered(
    media_registry: &SharedMediaRegistry,
    media_state: &mut MediaSocketState,
) {
    if media_state.outbound_rx.is_some() {
        return;
    }
    let Some(call_id) = media_state.gateway_call_id.clone() else {
        return;
    };
    let (tx, rx) = mpsc::channel(OUTBOUND_MEDIA_QUEUE_CAPACITY);
    media_registry.register_call(call_id.clone(), tx).await;
    media_state.outbound_rx = Some(rx);
    tracing::info!(gateway_call_id = call_id, "media.outbound_queue.registered");
}

async fn send_outbound_or_silence(
    socket: &mut WebSocket,
    state: &SharedState,
    media_state: &mut MediaSocketState,
) -> anyhow::Result<()> {
    if let Some(command) = pending_clear_command(media_state).await {
        return send_outbound_command(socket, state, media_state, command).await;
    }

    if let Some(command) = next_outbound_command(media_state) {
        return send_outbound_command(socket, state, media_state, command).await;
    }

    let active_lookup = media_state
        .gateway_call_id
        .clone()
        .map(|call_id| (call_id, media_state.media_registry.clone()));
    if let Some((call_id, media_registry)) = active_lookup {
        let Some(playback_id) = media_registry.active_speech_playback_id(&call_id).await else {
            return send_silence_keepalive(socket, media_state).await;
        };
        media_state.outbound_underrun_ticks = media_state.outbound_underrun_ticks.saturating_add(1);
        media_state.outbound_pacing.observe_underrun();
        if media_state.outbound_underrun_ticks <= 5
            || media_state.outbound_underrun_ticks.is_multiple_of(50)
        {
            tracing::warn!(
                gateway_call_id = call_id,
                playback_id,
                underrun_ticks = media_state.outbound_underrun_ticks,
                queue_depth = outbound_queue_depth(media_state),
                "tts.outbound.underrun"
            );
        }
        return Ok(());
    }

    send_silence_keepalive(socket, media_state).await
}

async fn pending_clear_command(media_state: &mut MediaSocketState) -> Option<OutboundMediaCommand> {
    let call_id = media_state.gateway_call_id.as_deref()?;
    let pending = media_state
        .media_registry
        .take_pending_clear(call_id)
        .await?;
    drop_queued_playback(media_state, &pending.playback_id);
    Some(OutboundMediaCommand::Clear {
        playback_id: pending.playback_id,
        requested_at: pending.requested_at,
        reason: pending.reason,
    })
}

fn drop_queued_playback(media_state: &mut MediaSocketState, playback_id: &str) {
    media_state
        .outbound_pending
        .retain(|command| command.playback_id() != playback_id);
    if let Some(rx) = media_state.outbound_rx.as_mut() {
        while let Ok(command) = rx.try_recv() {
            if command.playback_id() != playback_id {
                media_state.outbound_pending.push_back(command);
            }
        }
    }
}

fn next_outbound_command(media_state: &mut MediaSocketState) -> Option<OutboundMediaCommand> {
    loop {
        let command = if let Some(command) = media_state.outbound_pending.pop_front() {
            command
        } else {
            media_state
                .outbound_rx
                .as_mut()
                .and_then(|rx| rx.try_recv().ok())?
        };
        if media_state
            .canceled_playbacks
            .contains(command.playback_id())
        {
            continue;
        }
        return Some(command);
    }
}

async fn send_outbound_command(
    socket: &mut WebSocket,
    state: &SharedState,
    media_state: &mut MediaSocketState,
    command: OutboundMediaCommand,
) -> anyhow::Result<()> {
    let Some(call_id) = media_state.gateway_call_id.clone() else {
        bail!("outbound media command arrived before gateway call was known");
    };
    match command {
        OutboundMediaCommand::Frame(frame) => {
            socket
                .send(Message::Text(media_message(&frame.payload).into()))
                .await
                .context("send outbound media frame to Telnyx")?;
            log_outbound_frame_sent(media_state, &call_id, &frame.playback_id);
            maybe_emit_first_frame_span(state, media_state, &call_id, &frame).await;
            state
                .write()
                .await
                .mark_tts_frame_sent(&call_id, &frame.playback_id);
            Ok(())
        }
        OutboundMediaCommand::Clear {
            playback_id,
            requested_at,
            reason,
        } => {
            socket
                .send(Message::Text(clear_message().into()))
                .await
                .context("send Telnyx clear")?;
            media_state.canceled_playbacks.insert(playback_id.clone());
            emit_playback_terminal_spans(
                state,
                media_state,
                &call_id,
                &playback_id,
                "canceled",
                Some((requested_at, reason)),
            )
            .await;
            state
                .write()
                .await
                .mark_tts_canceled(&call_id, &playback_id);
            tracing::info!(gateway_call_id = call_id, playback_id, reason = reason.label(), "tts.clear.sent");
            Ok(())
        }
        OutboundMediaCommand::Mark { playback_id } => {
            socket
                .send(Message::Text(mark_message(&playback_id).into()))
                .await
                .context("send Telnyx mark")?;
            state
                .write()
                .await
                .mark_tts_mark_sent(&call_id, &playback_id, &playback_id);
            tracing::info!(gateway_call_id = call_id, playback_id, "tts.mark.sent");
            Ok(())
        }
    }
}

async fn maybe_emit_first_frame_span(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    call_id: &str,
    frame: &OutboundMediaFrame,
) {
    let Some(quality) = frame.quality.as_ref() else {
        return;
    };
    if !quality.first_for_playback
        || !media_state
            .first_frame_sent_playbacks
            .insert(frame.playback_id.clone())
    {
        return;
    }
    media_state
        .playback_started_at
        .insert(frame.playback_id.clone(), Instant::now());
    media_state
        .playback_quality_contexts
        .insert(frame.playback_id.clone(), quality.clone());
    let payload = map_from_value(json!({
        "playback_id": frame.playback_id.as_str(),
        "queue_depth": outbound_queue_depth(media_state),
    }));
    state.write().await.emit_quality_span_finished(
        call_id,
        quality.config_id.clone(),
        quality.redaction_mode,
        "media.first_frame_send",
        "playback_transport",
        quality.queued_at.elapsed(),
        true,
        false,
        payload,
    );
}

async fn emit_playback_terminal_spans(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    call_id: &str,
    playback_id: &str,
    status: &'static str,
    clear: Option<(Instant, SpeechClearReason)>,
) {
    let quality = media_state.playback_quality_contexts.get(playback_id).cloned();
    let (config_id, redaction_mode) = quality
        .as_ref()
        .map(|quality| (quality.config_id.clone(), quality.redaction_mode))
        .unwrap_or_else(|| {
            (
                media_state.quality_config.config_id(),
                media_state.quality_config.logging.redaction_mode,
            )
        });
    if let Some(started_at) = media_state.playback_started_at.remove(playback_id) {
        let payload = map_from_value(json!({
            "playback_id": playback_id,
            "status": status,
        }));
        state.write().await.emit_quality_span_finished(
            call_id,
            config_id.clone(),
            redaction_mode,
            "media.playback_terminal",
            "playback_transport",
            started_at.elapsed(),
            false,
            true,
            payload,
        );
    }
    if let Some((requested_at, reason)) = clear {
        if reason == SpeechClearReason::BargeIn {
            let payload = map_from_value(json!({
                "playback_id": playback_id,
                "terminal_status": status,
                "clear_reason": reason.label(),
            }));
            state.write().await.emit_quality_span_finished(
                call_id,
                config_id,
                redaction_mode,
                "barge_in.cancel_request_to_terminal",
                "playback_transport",
                requested_at.elapsed(),
                false,
                true,
                payload,
            );
        }
    }
    media_state.playback_quality_contexts.remove(playback_id);
}

fn log_outbound_frame_sent(media_state: &mut MediaSocketState, call_id: &str, playback_id: &str) {
    let now = Instant::now();
    let interval_ms = media_state
        .last_outbound_frame_sent_at
        .map(|last| now.duration_since(last).as_millis() as u64);
    media_state.last_outbound_frame_sent_at = Some(now);
    media_state.outbound_frame_count = media_state.outbound_frame_count.saturating_add(1);
    media_state
        .outbound_pacing
        .observe_frame(interval_ms, outbound_queue_depth(media_state));
    media_state.outbound_underrun_ticks = 0;

    let is_pacing_anomaly = interval_ms.is_some_and(|ms| !(15..=35).contains(&ms));
    if media_state.outbound_frame_count <= 5
        || media_state.outbound_frame_count.is_multiple_of(50)
        || is_pacing_anomaly
    {
        tracing::info!(
            gateway_call_id = call_id,
            playback_id,
            frame_index = media_state.outbound_frame_count,
            interval_ms = interval_ms.unwrap_or_default(),
            interval_observed = interval_ms.is_some(),
            queue_depth = outbound_queue_depth(media_state),
            "tts.outbound.frame.sent"
        );
    }
}

fn outbound_queue_depth(media_state: &MediaSocketState) -> usize {
    media_state.outbound_pending.len()
        + media_state
            .outbound_rx
            .as_ref()
            .map_or(0, mpsc::Receiver::len)
}

async fn send_silence_keepalive(
    socket: &mut WebSocket,
    media_state: &mut MediaSocketState,
) -> anyhow::Result<()> {
    let format = media_state
        .media_format
        .as_ref()
        .context("send silence keepalive before media format was known")?;
    socket
        .send(Message::Text(silence_keepalive_message(format)?.into()))
        .await
        .context("send silence keepalive to Telnyx")?;

    media_state.silence_keepalive_frames = media_state.silence_keepalive_frames.saturating_add(1);
    if media_state.silence_keepalive_frames == 1
        || media_state.silence_keepalive_frames.is_multiple_of(500)
    {
        tracing::info!(
            gateway_call_id = media_state.gateway_call_id.as_deref(),
            frames = media_state.silence_keepalive_frames,
            "media.silence_keepalive.sent"
        );
    }
    Ok(())
}

fn silence_keepalive_message(format: &MediaFormat) -> anyhow::Result<String> {
    let payload = STANDARD.encode(silence_payload(format)?);
    Ok(media_message_from_payload(payload))
}

fn media_message(payload: &[u8]) -> String {
    media_message_from_payload(STANDARD.encode(payload))
}

fn media_message_from_payload(payload: String) -> String {
    serde_json::json!({
        "event": "media",
        "media": {
            "payload": payload
        }
    })
    .to_string()
}

fn clear_message() -> String {
    serde_json::json!({
        "event": "clear"
    })
    .to_string()
}

fn mark_message(name: &str) -> String {
    serde_json::json!({
        "event": "mark",
        "mark": {
            "name": name
        }
    })
    .to_string()
}

fn silence_payload(format: &MediaFormat) -> anyhow::Result<Vec<u8>> {
    let samples_per_frame = samples_per_20ms(format.sample_rate_hz)?;
    match format.encoding.as_str() {
        "PCMU" => Ok(vec![PCMU_SILENCE_BYTE; samples_per_frame]),
        "PCMA" => Ok(vec![PCMA_SILENCE_BYTE; samples_per_frame]),
        "L16" => Ok(vec![0; samples_per_frame * 2]),
        other => bail!("unsupported silence keepalive encoding {other}"),
    }
}

fn samples_per_20ms(sample_rate_hz: u32) -> anyhow::Result<usize> {
    if sample_rate_hz == 0 || !sample_rate_hz.is_multiple_of(50) {
        bail!("sample rate {sample_rate_hz} cannot be packetized into 20ms frames");
    }
    Ok((sample_rate_hz / 50) as usize)
}

pub fn packetize_tts_chunk(
    chunk: AudioBuf<i16, PIPER_SAMPLE_RATE_HZ, Mono>,
    media: TelnyxMediaConfig,
) -> anyhow::Result<Vec<Vec<u8>>> {
    packetize_tts_samples(chunk.into_samples(), PIPER_SAMPLE_RATE_HZ, media)
}

pub fn packetize_tts_samples(
    mut samples: Vec<i16>,
    input_sample_rate_hz: u32,
    media: TelnyxMediaConfig,
) -> anyhow::Result<Vec<Vec<u8>>> {
    if input_sample_rate_hz == 0 {
        bail!("TTS input sample rate must be non-zero");
    }
    if media.sample_rate_hz != input_sample_rate_hz {
        samples = resample_i16_mono(
            &WindowedSincResampler::default(),
            &samples,
            input_sample_rate_hz,
            media.sample_rate_hz,
        )?;
    }
    let samples_per_packet = samples_per_20ms(media.sample_rate_hz)?;
    let mut packets = Vec::new();
    for packet_samples in samples.chunks(samples_per_packet) {
        if packet_samples.is_empty() {
            continue;
        }
        let mut packet_samples = packet_samples.to_vec();
        if packet_samples.len() < samples_per_packet {
            packet_samples.resize(samples_per_packet, 0);
        }
        let payload = match media.codec {
            TelnyxStreamCodec::Pcmu => g711::encode_pcmu(&packet_samples),
            TelnyxStreamCodec::L16 => l16::encode_l16_le(&packet_samples),
        };
        packets.push(payload);
    }
    Ok(packets)
}

#[cfg(test)]
async fn handle_text(
    text: &str,
    state: &SharedState,
    asr: &SharedAsrRegistry,
    media_state: &mut MediaSocketState,
) -> anyhow::Result<()> {
    handle_text_with_text_calls(text, state, asr, media_state, None).await
}

async fn handle_text_with_text_calls(
    text: &str,
    state: &SharedState,
    asr: &SharedAsrRegistry,
    media_state: &mut MediaSocketState,
    text_calls: Option<&SharedTextCallRegistry>,
) -> anyhow::Result<()> {
    let discriminator: EventDiscriminator =
        serde_json::from_str(text).context("parse Telnyx media event discriminator")?;
    match discriminator.event.as_str() {
        "connected" => Ok(()),
        "start" => {
            let event: StartEvent = serde_json::from_str(text).context("parse start event")?;
            let format = map_media_format(event.start.media_format.as_ref());
            validate_media_format(&format)?;
            let Some(registered) = register_start(state, &event, &format).await else {
                return Ok(());
            };
            let call_id = registered.gateway_call_id;
            if let Some(capture_root) = state.read().await.config.capture_dir.clone() {
                match MediaCapture::start(&capture_root, &call_id, &event.stream_id, &format) {
                    Ok(mut capture) => {
                        record_raw_capture(Some(&mut capture), text);
                        tracing::info!(
                            gateway_call_id = call_id,
                            stream_id = event.stream_id,
                            capture_dir = %capture.dir().display(),
                            "media.capture.started"
                        );
                        media_state.capture = Some(capture);
                    }
                    Err(error) => {
                        tracing::warn!(
                            gateway_call_id = call_id,
                            stream_id = event.stream_id,
                            error = %error,
                            "media.capture.start_failed"
                        );
                    }
                }
            }
            media_state.media_format = Some(format);
            media_state.gateway_call_id = Some(call_id.clone());
            media_state.asr_backend = Some(registered.asr_backend);
            open_asr_session(
                state,
                asr,
                media_state,
                &call_id,
                &event.stream_id,
                "media_start",
            )
            .await?;
            media_state.silence_keepalive = true;
            tracing::info!(
                gateway_call_id = media_state.gateway_call_id.as_deref(),
                stream_id = event.stream_id,
                asr_backend = registered.asr_backend.label(),
                asr_model = registered.asr_backend.model_label(),
                "media.silence_keepalive.started"
            );
            Ok(())
        }
        "media" => {
            record_raw_capture(media_state.capture.as_mut(), text);
            let event: MediaEvent = serde_json::from_str(text).context("parse media event")?;
            let sequence = event
                .media
                .chunk
                .parse::<u64>()
                .context("parse Telnyx media.chunk as sequence")?;
            media_state.inbound_transport.observe_packet(sequence, 20);
            let payload = STANDARD
                .decode(event.media.payload.as_bytes())
                .context("decode Telnyx media payload base64")?;
            let ready = match media_state.reorder.push(SequencedFrame {
                sequence,
                payload: EncodedMediaFrame {
                    payload,
                    track: event.media.track,
                },
            }) {
                Ok(ready) => ready,
                Err(VoiceError::StaleFrameSequence { .. }) => {
                    media_state.inbound_transport.observe_stale();
                    tracing::warn!(stream_id = event.stream_id, sequence, "media.frame.stale");
                    return Ok(());
                }
                Err(error) => return Err(anyhow::Error::from(error)),
            };
            for frame in ready {
                ingest_frame(
                    state,
                    asr,
                    media_state,
                    event.stream_id.as_str(),
                    frame.payload,
                    text_calls,
                )
                .await?;
            }
            Ok(())
        }
        "stop" => {
            record_raw_capture(media_state.capture.as_mut(), text);
            let event: StopEvent = serde_json::from_str(text).context("parse stop event")?;
            media_state.silence_keepalive = false;
            let result = finish_stream(state, media_state, event.stream_id, text_calls).await;
            emit_quality_transport_rollups(state, media_state).await;
            finalize_capture(media_state).await;
            result
        }
        "mark" => {
            let event: MarkEvent = serde_json::from_str(text).context("parse mark event")?;
            if let (Some(call_id), Some(name)) = (
                media_state.gateway_call_id.clone(),
                event.mark.name.clone(),
            ) {
                emit_playback_terminal_spans(
                    state,
                    media_state,
                    &call_id,
                    &name,
                    "completed",
                    None,
                )
                .await;
                state.write().await.mark_tts_completed(&call_id, &name);
                media_state
                    .media_registry
                    .finish_speech(&call_id, &name)
                    .await;
                tracing::info!(
                    gateway_call_id = call_id,
                    mark_name = name,
                    "tts.mark.received"
                );
            }
            Ok(())
        }
        "clear" | "dtmf" => Ok(()),
        "error" => bail!("Telnyx media error event: {text}"),
        other => bail!("unsupported Telnyx media event {other}"),
    }
}

struct RegisteredStart {
    gateway_call_id: String,
    asr_backend: LiveAsrBackend,
}

async fn register_start(
    state: &SharedState,
    event: &StartEvent,
    format: &MediaFormat,
) -> Option<RegisteredStart> {
    let mut guard = state.write().await;
    let media = MediaMetadata {
        stream_id: Some(event.stream_id.clone()),
        encoding: Some(format.encoding.clone()),
        sample_rate_hz: Some(format.sample_rate_hz),
        channels: Some(format.channels),
        track: Some("inbound".to_string()),
    };
    let gateway_call_id =
        guard.set_call_stream(&event.start.call_control_id, event.stream_id.clone(), media);
    match gateway_call_id {
        StreamAttachOutcome::Attached {
            gateway_call_id,
            asr_backend,
        } => {
            guard.log(
                LogLevel::Info,
                format!(
                    "media started for {gateway_call_id}: {} {} Hz {}ch asr={}",
                    format.encoding,
                    format.sample_rate_hz,
                    format.channels,
                    asr_backend.label()
                ),
            );
            guard.emit_quality_config_snapshot(
                &gateway_call_id,
                "stream_start",
                "new_asr_sessions",
                None,
            );
            tracing::info!(
                gateway_call_id,
                call_control_id = event.start.call_control_id,
                call_session_id = event.start.call_session_id.as_deref(),
                stream_id = event.stream_id,
                codec = format.encoding,
                sample_rate_hz = format.sample_rate_hz,
                channels = format.channels,
                asr_backend = asr_backend.label(),
                asr_model = asr_backend.model_label(),
                "media.started"
            );
            Some(RegisteredStart {
                gateway_call_id,
                asr_backend,
            })
        }
        StreamAttachOutcome::NotAnswered {
            gateway_call_id,
            status,
        } => {
            guard.log(
                LogLevel::Warn,
                format!(
                    "ignored media start for {gateway_call_id}; call is {} and was not answered by operator",
                    status.label()
                ),
            );
            tracing::warn!(
                gateway_call_id,
                call_control_id = event.start.call_control_id,
                status = status.label(),
                "media.start.rejected_not_answered"
            );
            None
        }
        StreamAttachOutcome::UnknownCallControl => {
            guard.log(
                LogLevel::Warn,
                format!(
                    "media start for unknown call_control_id {}",
                    event.start.call_control_id
                ),
            );
            tracing::warn!(
                call_control_id = event.start.call_control_id,
                stream_id = event.stream_id,
                "media.start.rejected_unknown_call"
            );
            None
        }
    }
}

async fn ingest_frame(
    state: &SharedState,
    asr: &SharedAsrRegistry,
    media_state: &mut MediaSocketState,
    stream_id: &str,
    frame: EncodedMediaFrame,
    text_calls: Option<&SharedTextCallRegistry>,
) -> anyhow::Result<()> {
    let format = media_state
        .media_format
        .clone()
        .context("media frame arrived before media format was known")?;
    let gateway_call_id = media_state
        .gateway_call_id
        .clone()
        .context("media frame arrived before gateway call was known")?;

    let mut samples = decode_payload(&format, &frame.payload)?;
    media_state.decoded_frame_count += 1;
    record_decoded_capture(media_state.capture.as_mut(), &samples);
    let stats = sample_stats(&samples);
    let frame_duration_ms = frame_duration_ms(samples.len(), format.sample_rate_hz);
    log_decoded_frame_stats(
        media_state.decoded_frame_count,
        &format,
        stream_id,
        &frame,
        &stats,
        samples.len(),
    );
    match media_state.asr_gate.accept(
        media_state.decoded_frame_count,
        stream_id,
        frame_duration_ms,
        &stats,
        &media_state.quality_config,
    ) {
        AsrFrameDecision::Suppress => return Ok(()),
        AsrFrameDecision::Continue { speech_onset } => {
            if speech_onset {
                if let Some(runtime) = media_state.conversation.as_ref() {
                    conversation::handle_speech_onset(
                        state,
                        &media_state.media_registry,
                        runtime,
                        &gateway_call_id,
                        Some(&media_state.quality_config),
                    )
                    .await?;
                }
            }
        }
        AsrFrameDecision::Finalize {
            trailing_silence_ms,
            endpoint_wait_started_at,
            speech_to_low_energy,
        } => {
            tracing::info!(
                gateway_call_id,
                stream_id,
                trailing_silence_ms,
                "asr.local_endpoint.finalizing"
            );
            emit_asr_endpoint_spans(
                state,
                media_state.active_quality_asr.as_ref(),
                &gateway_call_id,
                Some(stream_id),
                trailing_silence_ms,
                endpoint_wait_started_at,
                speech_to_low_energy,
            )
            .await;
            finish_asr_session(
                state,
                media_state,
                Some(gateway_call_id.as_str()),
                Some(stream_id.to_string()),
                text_calls,
            )
            .await?;
            media_state.asr_gate.wait_for_next_speech();
            return Ok(());
        }
    }
    if media_state.session.is_none() {
        open_asr_session(
            state,
            asr,
            media_state,
            &gateway_call_id,
            stream_id,
            "missing_session",
        )
        .await?;
    }
    if format.sample_rate_hz != 16_000 {
        samples = resample_i16_mono(
            &WindowedSincResampler::default(),
            &samples,
            format.sample_rate_hz,
            16_000,
        )?;
    }
    record_asr_capture(media_state.capture.as_mut(), &samples);

    let Some(session) = media_state.session.as_mut() else {
        bail!("ASR session unavailable after reopen");
    };
    let events = session
        .ingest(AudioBuf::<i16, 16_000, Mono>::new(samples))
        .await?;
    let conversation_events = conversation_events_from_transcripts(&events);
    let needs_reset = record_transcript_events(
        state,
        &gateway_call_id,
        events,
        TranscriptRecordContext {
            stream_id: Some(stream_id),
            media_format: Some(&format),
            capture: media_state.capture.as_mut(),
            text_calls,
            quality_session: media_state.active_quality_asr.as_ref(),
        },
    )
    .await;
    forward_conversation_events(
        state,
        &media_state.media_registry,
        media_state.conversation.as_ref(),
        &gateway_call_id,
        conversation_events,
        Some(&media_state.quality_config),
    )
    .await;
    if needs_reset {
        media_state.session = None;
        media_state.asr_gate.wait_for_next_speech();
        open_asr_session(
            state,
            asr,
            media_state,
            &gateway_call_id,
            stream_id,
            "repeated_token",
        )
        .await?;
    }
    Ok(())
}

async fn open_asr_session(
    state: &SharedState,
    asr: &SharedAsrRegistry,
    media_state: &mut MediaSocketState,
    gateway_call_id: &str,
    stream_id: &str,
    reason: &'static str,
) -> anyhow::Result<()> {
    let asr_backend = media_state
        .asr_backend
        .context("ASR backend was not bound to media stream")?;
    let (quality_session, quality_config) = {
        let mut guard = state.write().await;
        let session = guard.start_quality_asr_session(gateway_call_id, Some(stream_id), reason);
        (session, guard.quality.config.clone())
    };
    media_state.quality_config = quality_config;
    media_state.active_quality_asr = Some(quality_session);
    media_state.session = Some(asr.open_session(asr_backend).await?);
    tracing::info!(
        gateway_call_id,
        stream_id,
        reason,
        asr_backend = asr_backend.label(),
        asr_model = asr_backend.model_label(),
        "asr.session.opened"
    );
    Ok(())
}

fn record_raw_capture(capture: Option<&mut MediaCapture>, raw: &str) {
    if let Some(capture) = capture {
        if let Err(error) = capture.record_raw_event(raw) {
            tracing::warn!(error = %error, "media.capture.raw_failed");
        }
    }
}

fn record_decoded_capture(capture: Option<&mut MediaCapture>, samples: &[i16]) {
    if let Some(capture) = capture {
        if let Err(error) = capture.record_decoded_samples(samples) {
            tracing::warn!(error = %error, "media.capture.decoded_failed");
        }
    }
}

fn record_asr_capture(capture: Option<&mut MediaCapture>, samples: &[i16]) {
    if let Some(capture) = capture {
        if let Err(error) = capture.record_asr_samples(samples) {
            tracing::warn!(error = %error, "media.capture.asr_failed");
        }
    }
}

fn record_transcript_capture(capture: &mut MediaCapture, kind: &str, text: &str, suppressed: bool) {
    if let Err(error) = capture.record_transcript(kind, text, suppressed) {
        tracing::warn!(error = %error, "media.capture.transcript_failed");
    }
}

async fn emit_quality_transport_rollups(state: &SharedState, media_state: &mut MediaSocketState) {
    if media_state.rollups_emitted {
        return;
    }
    media_state.rollups_emitted = true;
    let Some(call_id) = media_state.gateway_call_id.clone() else {
        return;
    };
    let session = media_state
        .active_quality_asr
        .as_ref()
        .or(media_state.last_quality_asr.as_ref());
    let (config_id, redaction_mode) = session
        .map(|session| (session.config_id.clone(), session.redaction_mode))
        .unwrap_or_else(|| {
            (
                media_state.quality_config.config_id(),
                media_state.quality_config.logging.redaction_mode,
            )
        });
    let inbound_payload = media_state
        .inbound_transport
        .rollup_payload(session, media_state.media_format.as_ref());
    let playback_id = media_state.playback_started_at.keys().next().map(String::as_str);
    let outbound_payload = media_state.outbound_pacing.rollup_payload(playback_id);
    let mut guard = state.write().await;
    guard.emit_quality_inbound_transport_rollup(
        &call_id,
        config_id.clone(),
        redaction_mode,
        inbound_payload,
    );
    guard.emit_quality_outbound_pacing_rollup(
        &call_id,
        config_id,
        redaction_mode,
        outbound_payload,
    );
}

async fn finalize_capture(media_state: &mut MediaSocketState) {
    if let Some(capture) = media_state.capture.take() {
        match capture.finalize() {
            Ok(dir) => tracing::info!(capture_dir = %dir.display(), "media.capture.finalized"),
            Err(error) => tracing::warn!(error = %error, "media.capture.finalize_failed"),
        }
    }
}

fn decode_payload(format: &MediaFormat, payload: &[u8]) -> anyhow::Result<Vec<i16>> {
    validate_media_format(format)?;
    match format.encoding.as_str() {
        "L16" => Ok(l16::decode_l16_le(payload)?),
        "PCMU" => Ok(g711::decode_pcmu(payload)),
        "PCMA" => Ok(g711::decode_pcma(payload)),
        other => bail!("unsupported inbound media encoding {other}"),
    }
}

fn log_decoded_frame_stats(
    frame_index: usize,
    format: &MediaFormat,
    stream_id: &str,
    frame: &EncodedMediaFrame,
    stats: &SampleStats,
    sample_count: usize,
) {
    if frame_index > 5 && !frame_index.is_multiple_of(50) {
        return;
    }
    tracing::debug!(
        stream_id,
        frame_index,
        track = frame.track.as_deref().unwrap_or("<unknown>"),
        codec = format.encoding,
        sample_rate_hz = format.sample_rate_hz,
        channels = format.channels,
        payload_len = frame.payload.len(),
        samples = sample_count,
        peak = stats.peak,
        rms = stats.rms,
        mean = stats.mean,
        "media.frame.decoded"
    );
}

#[derive(Clone, Copy, Debug)]
struct SampleStats {
    peak: i16,
    rms: f32,
    mean: f32,
}

impl SampleStats {
    fn has_speech_energy(&self, speech: &SpeechQualityConfig) -> bool {
        self.rms >= speech.rms_threshold || i32::from(self.peak) >= speech.peak_threshold
    }
}

fn frame_duration_ms(sample_count: usize, sample_rate_hz: u32) -> u64 {
    if sample_rate_hz == 0 {
        return 0;
    }
    (sample_count as u64 * 1_000) / u64::from(sample_rate_hz)
}

fn map_from_value(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

fn percentile_u64(values: &[u64], percentile: u8) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let percentile = percentile.min(100) as usize;
    let index = ((sorted.len() - 1) * percentile) / 100;
    sorted[index]
}

fn sample_stats(samples: &[i16]) -> SampleStats {
    if samples.is_empty() {
        return SampleStats {
            peak: 0,
            rms: 0.0,
            mean: 0.0,
        };
    }

    let mut peak = 0i16;
    let mut sum = 0f64;
    let mut sum_squares = 0f64;
    for &sample in samples {
        let abs = sample.saturating_abs();
        if abs > peak {
            peak = abs;
        }
        let value = f64::from(sample);
        sum += value;
        sum_squares += value * value;
    }
    let len = samples.len() as f64;
    SampleStats {
        peak,
        rms: (sum_squares / len).sqrt() as f32,
        mean: (sum / len) as f32,
    }
}

#[derive(Default)]
struct AsrGate {
    speech_started: bool,
    speech_started_at: Option<Instant>,
    low_energy_started_at: Option<Instant>,
    low_energy_run_ms: u64,
    suppressed_initial_frames: usize,
    suppressed_tail_frames: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AsrFrameDecision {
    Suppress,
    Continue { speech_onset: bool },
    Finalize {
        trailing_silence_ms: u64,
        endpoint_wait_started_at: Option<Instant>,
        speech_to_low_energy: Option<Duration>,
    },
}

impl AsrGate {
    fn accept(
        &mut self,
        frame_index: usize,
        stream_id: &str,
        frame_duration_ms: u64,
        stats: &SampleStats,
        quality: &VoiceQualityConfig,
    ) -> AsrFrameDecision {
        let now = Instant::now();
        if stats.has_speech_energy(&quality.speech) {
            let was_started = self.speech_started;
            let resumed_after_onset_pause =
                self.low_energy_run_ms >= quality.speech.onset_min_silence_ms;
            let speech_onset = !was_started || resumed_after_onset_pause;
            self.speech_started = true;
            if speech_onset {
                self.speech_started_at = Some(now);
            }
            self.low_energy_started_at = None;
            self.low_energy_run_ms = 0;
            if !was_started {
                tracing::info!(
                    stream_id,
                    frame_index,
                    suppressed_frames = self.suppressed_initial_frames,
                    peak = stats.peak,
                    rms = stats.rms,
                    "media.speech.detected"
                );
            }
            return AsrFrameDecision::Continue { speech_onset };
        }

        if self.speech_started {
            if self.low_energy_started_at.is_none() {
                self.low_energy_started_at = Some(now);
            }
            self.low_energy_run_ms = self.low_energy_run_ms.saturating_add(frame_duration_ms);
            if self.low_energy_run_ms <= quality.endpoint.trailing_silence_ms {
                return AsrFrameDecision::Continue {
                    speech_onset: false,
                };
            }
            self.suppressed_tail_frames = self.suppressed_tail_frames.saturating_add(1);
            if self.suppressed_tail_frames <= 5 || self.suppressed_tail_frames.is_multiple_of(50) {
                tracing::debug!(
                    stream_id,
                    frame_index,
                    suppressed_tail_frames = self.suppressed_tail_frames,
                    low_energy_run_ms = self.low_energy_run_ms,
                    peak = stats.peak,
                    rms = stats.rms,
                    "media.frame.local_endpoint"
                );
            }
            let endpoint_wait_started_at = self.low_energy_started_at;
            let speech_to_low_energy = self
                .speech_started_at
                .zip(endpoint_wait_started_at)
                .map(|(started_at, low_energy_at)| low_energy_at.duration_since(started_at));
            return AsrFrameDecision::Finalize {
                trailing_silence_ms: self.low_energy_run_ms,
                endpoint_wait_started_at,
                speech_to_low_energy,
            };
        }

        self.suppressed_initial_frames = self.suppressed_initial_frames.saturating_add(1);
        if self.suppressed_initial_frames <= 5 || self.suppressed_initial_frames.is_multiple_of(50)
        {
            tracing::debug!(
                stream_id,
                frame_index,
                suppressed_frames = self.suppressed_initial_frames,
                peak = stats.peak,
                rms = stats.rms,
                "media.frame.suppressed_low_energy"
            );
        }
        AsrFrameDecision::Suppress
    }

    fn wait_for_next_speech(&mut self) {
        self.speech_started = false;
        self.speech_started_at = None;
        self.low_energy_started_at = None;
        self.low_energy_run_ms = 0;
    }
}

fn transcript_preview(text: &str) -> String {
    const PREVIEW_CHARS: usize = 48;

    let mut preview = text.chars().take(PREVIEW_CHARS).collect::<String>();
    if text.chars().count() > PREVIEW_CHARS {
        preview.push_str("...");
    }
    preview
}

async fn finish_stream(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    stream_id: Option<String>,
    text_calls: Option<&SharedTextCallRegistry>,
) -> anyhow::Result<()> {
    let gateway_call_id = media_state.gateway_call_id.clone();
    finish_asr_session(
        state,
        media_state,
        gateway_call_id.as_deref(),
        stream_id,
        text_calls,
    )
    .await?;
    if let Some(call_id) = gateway_call_id {
        let mut guard = state.write().await;
        if let Some(call) = guard.calls.get_mut(&call_id) {
            call.status = CallStatus::Ended;
            call.push_timeline("media stream stopped");
        }
    }
    Ok(())
}

async fn emit_asr_endpoint_spans(
    state: &SharedState,
    session: Option<&ActiveAsrQualitySession>,
    gateway_call_id: &str,
    stream_id: Option<&str>,
    trailing_silence_ms: u64,
    endpoint_wait_started_at: Option<Instant>,
    speech_to_low_energy: Option<Duration>,
) {
    let Some(session) = session else {
        return;
    };
    if let Some(duration) = speech_to_low_energy {
        let payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id.as_str(),
            "utterance_id": session.utterance_id.as_str(),
            "stream_id": stream_id,
        }));
        state.write().await.emit_quality_span_finished(
            gateway_call_id,
            session.config_id.clone(),
            session.redaction_mode,
            "utterance.speech_to_low_energy",
            "caller_speech",
            duration,
            true,
            false,
            payload,
        );
    }
    if let Some(started_at) = endpoint_wait_started_at {
        let payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id.as_str(),
            "utterance_id": session.utterance_id.as_str(),
            "stream_id": stream_id,
            "trailing_silence_ms": trailing_silence_ms,
        }));
        state.write().await.emit_quality_span_finished(
            gateway_call_id,
            session.config_id.clone(),
            session.redaction_mode,
            "asr.endpoint_wait",
            "endpointing",
            started_at.elapsed(),
            true,
            false,
            payload,
        );
    }
}

async fn finish_asr_session(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    gateway_call_id: Option<&str>,
    stream_id: Option<String>,
    text_calls: Option<&SharedTextCallRegistry>,
) -> anyhow::Result<()> {
    if let (Some(call_id), Some(asr_session)) = (gateway_call_id, media_state.session.take()) {
        let quality_session = media_state.active_quality_asr.take();
        let finish_started_at = Instant::now();
        let events = asr_session.finish().await?;
        if let Some(session) = quality_session.as_ref() {
            let payload = map_from_value(json!({
                "asr_session_id": session.asr_session_id.as_str(),
                "utterance_id": session.utterance_id.as_str(),
                "stream_id": stream_id.as_deref(),
                "transcript_events": events.len(),
            }));
            state.write().await.emit_quality_span_finished(
                call_id,
                session.config_id.clone(),
                session.redaction_mode,
                "asr.local_finish",
                "asr_generation",
                finish_started_at.elapsed(),
                true,
                false,
                payload,
            );
            media_state.last_quality_asr = Some(session.clone());
        }
        let conversation_events = conversation_events_from_transcripts(&events);
        let _ = record_transcript_events(
            state,
            call_id,
            events,
            TranscriptRecordContext {
                stream_id: stream_id.as_deref(),
                media_format: media_state.media_format.as_ref(),
                capture: media_state.capture.as_mut(),
                text_calls,
                quality_session: quality_session.as_ref(),
            },
        )
        .await;
        forward_conversation_events(
            state,
            &media_state.media_registry,
            media_state.conversation.as_ref(),
            call_id,
            conversation_events,
            Some(&media_state.quality_config),
        )
        .await;
    }
    Ok(())
}

fn conversation_events_from_transcripts(events: &[AsrTranscriptEvent]) -> Vec<TranscriptEvent> {
    events
        .iter()
        .filter(|event| !event.is_suppressed())
        .map(|event| event.event.clone())
        .collect()
}

async fn forward_conversation_events(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    conversation: Option<&ConversationRuntime>,
    gateway_call_id: &str,
    events: Vec<TranscriptEvent>,
    quality_config: Option<&VoiceQualityConfig>,
) {
    let Some(conversation) = conversation else {
        return;
    };
    for event in events {
        if let Err(error) = conversation::handle_transcript_event(
            state,
            media_registry,
            conversation,
            gateway_call_id,
            event,
            quality_config,
        )
        .await
        {
            log_media_error(state, Some(gateway_call_id), error).await;
        }
    }
}

struct TranscriptRecordContext<'a> {
    stream_id: Option<&'a str>,
    media_format: Option<&'a MediaFormat>,
    capture: Option<&'a mut MediaCapture>,
    text_calls: Option<&'a SharedTextCallRegistry>,
    quality_session: Option<&'a ActiveAsrQualitySession>,
}

struct FinalTurnCandidate {
    text: String,
    transcript_event_id: Option<String>,
}

async fn record_transcript_events(
    state: &SharedState,
    gateway_call_id: &str,
    events: Vec<AsrTranscriptEvent>,
    mut context: TranscriptRecordContext<'_>,
) -> bool {
    let mut guard = state.write().await;
    let include_transcript_text = guard.quality.config.logging.include_transcript_text;
    let mut reset_requested = false;
    let mut final_turns = Vec::new();
    for event in events {
        let kind = if event.event.is_final() {
            TranscriptKind::Final
        } else {
            TranscriptKind::Partial
        };
        let kind_label = if event.event.is_final() {
            "transcript.final"
        } else {
            "transcript.partial"
        };
        let text = event.event.text().to_string();
        let suppressed = event.is_suppressed();
        reset_requested |= event.requires_session_reset();
        if let Some(capture) = context.capture.as_deref_mut() {
            record_transcript_capture(capture, kind_label, &text, suppressed);
        }
        let call = guard.calls.get(gateway_call_id);
        let call_control_id = call.map(|call| call.ids.call_control_id.clone());
        let call_session_id = call.and_then(|call| call.ids.call_session_id.clone());
        let call_leg_id = call.and_then(|call| call.ids.call_leg_id.clone());
        let effective_stream_id = context
            .stream_id
            .map(str::to_string)
            .or_else(|| call.and_then(|call| call.ids.stream_id.clone()));
        let codec = context
            .media_format
            .map(|format| format.encoding.clone())
            .or_else(|| call.and_then(|call| call.media.encoding.clone()));
        let sample_rate_hz = context
            .media_format
            .map(|format| format.sample_rate_hz)
            .or_else(|| call.and_then(|call| call.media.sample_rate_hz));
        let asr_backend = call.and_then(|call| call.asr_backend);

        if suppressed {
            let suppression_reason = event
                .suppression_reason()
                .map(AsrTranscriptSuppressionReason::label);
            let transcript_preview = include_transcript_text.then(|| transcript_preview(&text));
            tracing::warn!(
                gateway_call_id,
                call_control_id = call_control_id.as_deref(),
                call_session_id = call_session_id.as_deref(),
                call_leg_id = call_leg_id.as_deref(),
                stream_id = effective_stream_id.as_deref(),
                codec = codec.as_deref(),
                sample_rate_hz,
                asr_backend = asr_backend.map(LiveAsrBackend::label),
                asr_model = asr_backend.map(LiveAsrBackend::model_label),
                transcript_kind = kind_label,
                suppression_reason,
                transcript_chars = text.chars().count(),
                transcript_preview = transcript_preview.as_deref(),
                "transcript.suppressed_repeated_token"
            );
            continue;
        }

        guard.add_transcript(gateway_call_id, kind.clone(), text.clone());
        if matches!(kind, TranscriptKind::Final) {
            let transcript_event_id = guard
                .quality
                .event_sink
                .is_enabled()
                .then(|| format!("trn_{}", uuid::Uuid::new_v4().simple()));
            final_turns.push(FinalTurnCandidate {
                text: text.clone(),
                transcript_event_id,
            });
        }
        let transcript_text = include_transcript_text.then_some(text.as_str());
        tracing::info!(
            gateway_call_id,
            call_control_id = call_control_id.as_deref(),
            call_session_id = call_session_id.as_deref(),
            call_leg_id = call_leg_id.as_deref(),
            stream_id = effective_stream_id.as_deref(),
            codec = codec.as_deref(),
            sample_rate_hz,
            asr_backend = asr_backend.map(LiveAsrBackend::label),
            asr_model = asr_backend.map(LiveAsrBackend::model_label),
            transcript_kind = kind_label,
            transcript_chars = text.chars().count(),
            transcript_text,
            "{kind_label}"
        );
    }
    drop(guard);
    if let Some(text_calls) = context.text_calls {
        for final_turn in final_turns {
            match text_calls
                .send_caller_turn(gateway_call_id, final_turn.text.clone())
                .await
            {
                Ok(Some(turn_id)) => {
                    let mut guard = state.write().await;
                    guard.emit_quality_caller_turn_sent(
                        gateway_call_id,
                        &turn_id,
                        &final_turn.text,
                        context.quality_session,
                    );
                    if let (Some(session), Some(transcript_event_id)) = (
                        context.quality_session,
                        final_turn.transcript_event_id.as_deref(),
                    ) {
                        guard.emit_quality_asr_turn_mapped(
                            gateway_call_id,
                            session,
                            &turn_id,
                            transcript_event_id,
                            true,
                        );
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        gateway_call_id,
                        error = %error,
                        "text_call.caller_turn.forward_failed"
                    );
                }
            }
        }
    }
    reset_requested
}

async fn log_media_error(state: &SharedState, gateway_call_id: Option<&str>, error: anyhow::Error) {
    let mut guard = state.write().await;
    let message = format!("media error: {error:#}");
    guard.log(LogLevel::Error, message.clone());
    if let Some(call_id) = gateway_call_id {
        if let Some(call) = guard.calls.get_mut(call_id) {
            call.status = CallStatus::Failed;
            call.last_error = Some(message.clone());
            call.push_timeline(message.clone());
        }
    }
    tracing::error!(gateway_call_id, error = %error, "media.failed");
}

fn map_media_format(input: Option<&MediaFormatPayload>) -> MediaFormat {
    MediaFormat {
        encoding: input
            .and_then(|value| value.encoding.clone())
            .unwrap_or_else(|| "PCMU".to_string())
            .to_ascii_uppercase(),
        sample_rate_hz: input.and_then(|value| value.sample_rate).unwrap_or(8_000),
        channels: input.and_then(|value| value.channels).unwrap_or(1),
    }
}

fn validate_media_format(format: &MediaFormat) -> anyhow::Result<()> {
    match format.encoding.as_str() {
        "L16" | "PCMU" | "PCMA" => Ok(()),
        other => bail!("unsupported inbound media encoding {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use motlie_model::typed::{AudioBuf, Mono};
    use motlie_voice::app::TranscriptEvent;

    use crate::adapter::{
        AsrRegistry, EchoAsrFactory, InboundAsrFactory, SharedAsrFactory, SharedAsrRegistry,
    };
    use crate::operator::state::{shared_state, CallStatus, TelnyxIds, TtsPlaybackStatus};
    use crate::tts::{
        LiveTtsBackend, OutboundTtsFactory, TtsAudio, TtsRegistry, PIPER_SAMPLE_RATE_HZ,
    };

    #[test]
    fn pcma_and_pcmu_decode_to_i16_audio() {
        let pcmu = decode_payload(
            &MediaFormat {
                encoding: "PCMU".to_string(),
                sample_rate_hz: 8_000,
                channels: 1,
            },
            &[0xff, 0x7f],
        )
        .expect("pcmu should decode");
        assert_eq!(pcmu.len(), 2);

        let pcma = decode_payload(
            &MediaFormat {
                encoding: "PCMA".to_string(),
                sample_rate_hz: 8_000,
                channels: 1,
            },
            &[0xd5, 0x55],
        )
        .expect("pcma should decode");
        assert_eq!(pcma.len(), 2);
    }

    #[test]
    fn telnyx_l16_decodes_as_little_endian_pcm() {
        let decoded = decode_payload(
            &MediaFormat {
                encoding: "L16".to_string(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            &[0x26, 0x03, 0x10, 0x02, 0x07, 0x01, 0x34, 0xff],
        )
        .expect("l16 should decode");

        assert_eq!(decoded, vec![806, 528, 263, -204]);
    }

    #[test]
    fn pcmu_silence_keepalive_message_is_telnyx_media_json() {
        let format = MediaFormat {
            encoding: "PCMU".to_string(),
            sample_rate_hz: 8_000,
            channels: 1,
        };
        let message = silence_keepalive_message(&format).expect("keepalive should encode");
        let value: serde_json::Value =
            serde_json::from_str(&message).expect("keepalive should be JSON");
        let payload = value["media"]["payload"]
            .as_str()
            .expect("payload should be a string");
        let decoded = STANDARD
            .decode(payload.as_bytes())
            .expect("payload should be base64");

        assert_eq!(value["event"], "media");
        assert_eq!(decoded, vec![PCMU_SILENCE_BYTE; 160]);
    }

    #[test]
    fn l16_silence_keepalive_uses_20ms_16khz_pcm() {
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };
        let payload = silence_payload(&format).expect("L16 silence should encode");

        assert_eq!(payload.len(), 640);
        assert!(payload.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn outbound_control_messages_are_telnyx_json() {
        let clear: serde_json::Value =
            serde_json::from_str(&clear_message()).expect("clear should be JSON");
        let mark: serde_json::Value =
            serde_json::from_str(&mark_message("tts_123")).expect("mark should be JSON");

        assert_eq!(clear["event"], "clear");
        assert_eq!(mark["event"], "mark");
        assert_eq!(mark["mark"]["name"], "tts_123");
    }

    #[test]
    fn piper_chunk_packetizes_to_pcmu_20ms_frames() {
        let packets = packetize_tts_chunk(
            AudioBuf::<i16, PIPER_SAMPLE_RATE_HZ, Mono>::new(vec![1_000; 2_205]),
            TelnyxMediaConfig::default(),
        )
        .expect("Piper chunk should packetize");

        assert_eq!(packets.len(), 5);
        assert!(packets.iter().all(|packet| packet.len() == 160));
    }

    #[test]
    fn normalized_tts_samples_packetize_from_non_piper_rate() {
        let packets =
            packetize_tts_samples(vec![1_000; 2_400], 24_000, TelnyxMediaConfig::default())
                .expect("24kHz normalized TTS samples should packetize");

        assert_eq!(packets.len(), 5);
        assert!(packets.iter().all(|packet| packet.len() == 160));
    }

    struct FailingSecondChunkTtsFactory {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl OutboundTtsFactory for FailingSecondChunkTtsFactory {
        async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == 2 {
                bail!("second chunk failed");
            }
            Ok(vec![TtsAudio::new(
                vec![1_000; 2_205],
                PIPER_SAMPLE_RATE_HZ,
            )?])
        }

        fn label(&self) -> &'static str {
            "failing-second-chunk-test-tts"
        }
    }

    #[tokio::test]
    async fn chunked_tts_failure_requests_clear_and_drops_queued_audio() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = {
            let mut guard = state.write().await;
            guard.add_or_update_outbound_call(
                TelnyxIds {
                    call_control_id: "call-control-1".to_string(),
                    call_session_id: Some("session-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: Some("stream-1".to_string()),
                },
                None,
                None,
                CallStatus::MediaStarted,
            )
        };
        let media_registry = SharedMediaRegistry::default();
        let (tx, rx) = mpsc::channel(16);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let mut media_state = MediaSocketState::with_media_registry(media_registry.clone());
        media_state.gateway_call_id = Some(gateway_call_id.clone());
        media_state.outbound_rx = Some(rx);
        let failing_tts = Arc::new(FailingSecondChunkTtsFactory {
            calls: AtomicUsize::new(0),
        });
        let tts = Arc::new(TtsRegistry::new(failing_tts.clone(), failing_tts));

        let queued = crate::speech::queue_speech(
            &state,
            &media_registry,
            &tts,
            LiveTtsBackend::Piper,
            gateway_call_id.clone(),
            "Hello, world.".to_string(),
            "test say",
        )
        .await
        .expect("speech should start");

        let clear = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if let Some(command) = pending_clear_command(&mut media_state).await {
                    return command;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("failed chunk should request clear");

        match clear {
            OutboundMediaCommand::Clear { playback_id, .. } => {
                assert_eq!(playback_id, queued.playback_id)
            }
            other => panic!("expected clear command, got {other:?}"),
        }
        assert!(next_outbound_command(&mut media_state).is_none());
        assert!(media_registry
            .active_speech_playback_id(&gateway_call_id)
            .await
            .is_none());

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        let tts = call.tts.as_ref().expect("TTS state should exist");
        assert_eq!(tts.status, TtsPlaybackStatus::Failed);
        assert!(tts
            .error
            .as_deref()
            .is_some_and(|error| error.contains("second chunk failed")));
        assert_eq!(call.status, CallStatus::MediaStarted);
        drop(guard);

        state
            .write()
            .await
            .mark_tts_canceled(&gateway_call_id, &queued.playback_id);
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        let tts = call.tts.as_ref().expect("TTS state should exist");
        assert_eq!(tts.status, TtsPlaybackStatus::Failed);
    }

    #[tokio::test]
    async fn outbound_clear_preempts_and_drops_queued_frames() {
        let media_registry = SharedMediaRegistry::default();
        let (tx, rx) = mpsc::channel(8);
        media_registry
            .register_call("gwc_test".to_string(), tx.clone())
            .await;
        media_registry
            .start_speech(
                "gwc_test",
                "tts_test".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("register active speech");
        let mut media_state = MediaSocketState::with_media_registry(media_registry.clone());
        media_state.gateway_call_id = Some("gwc_test".to_string());
        media_state.outbound_rx = Some(rx);
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new("tts_test", vec![1; 160])))
        .await
        .expect("queue first frame");
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new("tts_test", vec![2; 160])))
        .await
        .expect("queue second frame");
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new("tts_test", vec![3; 160])))
        .await
        .expect("queue post-clear frame");
        media_registry
            .cancel_speech("gwc_test")
            .await
            .expect("cancel active speech");

        match pending_clear_command(&mut media_state)
            .await
            .expect("clear should be selected")
        {
            OutboundMediaCommand::Clear { playback_id, .. } => assert_eq!(playback_id, "tts_test"),
            other => panic!("expected clear to preempt queued frames, got {other:?}"),
        }
        assert!(next_outbound_command(&mut media_state).is_none());
    }

    #[tokio::test]
    async fn pending_clears_are_queued_when_replacing_after_prior_cancel() {
        let media_registry = SharedMediaRegistry::default();
        let (tx, rx) = mpsc::channel(8);
        media_registry
            .register_call("gwc_test".to_string(), tx)
            .await;
        media_registry
            .start_speech(
                "gwc_test",
                "tts_first".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("register first speech");
        media_registry
            .cancel_speech("gwc_test")
            .await
            .expect("cancel first speech");
        media_registry
            .start_speech(
                "gwc_test",
                "tts_second".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("register second speech");
        media_registry
            .start_speech_replacing_active(
                "gwc_test",
                "tts_third".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("replace second speech");
        let mut media_state = MediaSocketState::with_media_registry(media_registry.clone());
        media_state.gateway_call_id = Some("gwc_test".to_string());
        media_state.outbound_rx = Some(rx);

        match pending_clear_command(&mut media_state)
            .await
            .expect("first clear should be queued")
        {
            OutboundMediaCommand::Clear { playback_id, .. } => assert_eq!(playback_id, "tts_first"),
            other => panic!("expected first clear, got {other:?}"),
        }
        match pending_clear_command(&mut media_state)
            .await
            .expect("second clear should be queued")
        {
            OutboundMediaCommand::Clear { playback_id, .. } => assert_eq!(playback_id, "tts_second"),
            other => panic!("expected second clear, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn inbound_asr_ingests_while_outbound_tts_is_queued() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let asr = registry_with_factory(Arc::new(EchoAsrFactory));
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open ASR session");
        let (tx, rx) = mpsc::channel(4);
        media_state.outbound_rx = Some(rx);
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new("tts_test", vec![PCMU_SILENCE_BYTE; 160])))
        .await
        .expect("queue outbound TTS frame");

        let speech = STANDARD.encode(l16_samples(16_000, 4_000));
        handle_text(
            &media_event("stream-1", "1", &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("inbound media should still feed ASR");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.transcripts.len(), 1);
        assert_eq!(call.transcripts[0].text, "received 16000 samples");
        assert!(media_state.outbound_rx.is_some());
    }

    #[tokio::test]
    async fn mark_event_completes_tts_playback() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        media_registry
            .start_speech(
                &gateway_call_id,
                "tts_test".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("register active speech");
        {
            let mut guard = state.write().await;
            guard.start_tts_job(&gateway_call_id, "tts_test".to_string(), "hello");
            guard.mark_tts_mark_sent(&gateway_call_id, "tts_test", "tts_test");
        }
        let asr = registry_with_factory(Arc::new(EchoAsrFactory));
        let mut media_state = MediaSocketState::with_media_registry(media_registry.clone());
        media_state.gateway_call_id = Some(gateway_call_id.clone());

        handle_text(
            &serde_json::json!({
                "event": "mark",
                "mark": {
                    "name": "tts_test"
                }
            })
            .to_string(),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("mark should complete TTS");

        let guard = state.read().await;
        let status = guard
            .calls
            .get(&gateway_call_id)
            .and_then(|call| call.tts.as_ref())
            .map(|tts| tts.status)
            .expect("TTS state should exist");
        assert_eq!(status, crate::operator::state::TtsPlaybackStatus::Completed);
        media_registry
            .start_speech(
                &gateway_call_id,
                "tts_next".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("mark event should release active speech slot");
    }

    #[tokio::test]
    async fn telnyx_media_replay_feeds_echo_asr_transcripts() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = {
            let mut guard = state.write().await;
            guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: None,
                },
                Some("<from-phone-number>".to_string()),
                Some("<to-phone-number>".to_string()),
                CallStatus::Answering,
            )
        };
        let asr = registry_with_factory(Arc::new(EchoAsrFactory));
        let mut media_state = MediaSocketState::new();

        let start = serde_json::json!({
            "event": "start",
            "stream_id": "stream-1",
            "start": {
                "call_control_id": "call-1",
                "call_session_id": "sess-1",
                "media_format": {
                    "encoding": "L16",
                    "sample_rate": 16000,
                    "channels": 1
                }
            }
        })
        .to_string();
        handle_text(&start, &state, &asr, &mut media_state)
            .await
            .expect("start event should open ASR session");

        let chunk = STANDARD.encode(l16_samples(8_000, 4_000));
        let media_one = media_event("stream-1", "7", &chunk);
        handle_text(&media_one, &state, &asr, &mut media_state)
            .await
            .expect("first non-one media chunk should establish reorder base");
        assert!(state
            .read()
            .await
            .calls
            .get(&gateway_call_id)
            .expect("call exists")
            .transcripts
            .is_empty());

        let media_two = media_event("stream-1", "8", &chunk);
        handle_text(&media_two, &state, &asr, &mut media_state)
            .await
            .expect("contiguous media should feed ASR");

        let stop = serde_json::json!({
            "event": "stop",
            "stream_id": "stream-1"
        })
        .to_string();
        handle_text(&stop, &state, &asr, &mut media_state)
            .await
            .expect("stop should finish ASR session");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(
            media_state.gateway_call_id.as_deref(),
            Some(gateway_call_id.as_str())
        );
        assert_eq!(call.status, CallStatus::Ended);
        assert_eq!(call.ids.stream_id.as_deref(), Some("stream-1"));
        assert_eq!(call.media.encoding.as_deref(), Some("L16"));
        assert_eq!(call.media.sample_rate_hz, Some(16_000));
        assert_eq!(call.transcripts.len(), 2);
        assert_eq!(call.transcripts[0].text, "received 16000 samples");
        assert_eq!(call.transcripts[1].text, "received 16000 samples");
    }

    #[tokio::test]
    async fn media_capture_writes_replay_artifacts() {
        let capture_root = std::env::temp_dir().join(format!(
            "motlie-telnyx-capture-test-{}",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = {
            let mut guard = state.write().await;
            guard.config.capture_dir = Some(capture_root.clone());
            guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: None,
                },
                Some("<from-phone-number>".to_string()),
                Some("<to-phone-number>".to_string()),
                CallStatus::Answering,
            )
        };
        let asr = registry_with_factory(Arc::new(EchoAsrFactory));
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open ASR session");
        let speech = STANDARD.encode(l16_samples(16_000, 4_000));
        handle_text(
            &media_event("stream-1", "1", &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("media should be captured");
        handle_text(
            &serde_json::json!({
                "event": "stop",
                "stream_id": "stream-1"
            })
            .to_string(),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("stop should finish capture");

        let capture_dir = capture_root.join(&gateway_call_id).join("stream-1");
        assert!(capture_dir.join("manifest.json").is_file());
        assert!(capture_dir.join("telnyx-media.jsonl").is_file());
        assert!(capture_dir.join("decoded-inbound.wav").is_file());
        assert!(capture_dir.join("asr-input-16khz.wav").is_file());
        assert!(capture_dir.join("transcripts.jsonl").is_file());
        let decoded_wav = hound::WavReader::open(capture_dir.join("decoded-inbound.wav"))
            .expect("decoded capture should be a finalized WAV");
        assert_eq!(decoded_wav.duration(), 16_000);
        let asr_wav = hound::WavReader::open(capture_dir.join("asr-input-16khz.wav"))
            .expect("ASR capture should be a finalized WAV");
        assert_eq!(asr_wav.duration(), 16_000);
        let transcripts = std::fs::read_to_string(capture_dir.join("transcripts.jsonl"))
            .expect("transcript capture should be readable");
        assert!(transcripts.contains("received 16000 samples"));

        std::fs::remove_dir_all(&capture_root).expect("capture temp dir should be removed");
    }

    #[tokio::test]
    async fn low_energy_media_is_suppressed_until_speech() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let asr = registry_with_factory(Arc::new(EchoAsrFactory));
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open ASR session");

        let silence = STANDARD.encode(l16_samples(8_000, 0));
        handle_text(
            &media_event("stream-1", "1", &silence),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("silence should be accepted by transport");
        assert!(state
            .read()
            .await
            .calls
            .get(&gateway_call_id)
            .expect("call exists")
            .transcripts
            .is_empty());

        let speech = STANDARD.encode(l16_samples(16_000, 4_000));
        handle_text(
            &media_event("stream-1", "2", &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("speech should pass into ASR");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.transcripts.len(), 1);
        assert_eq!(call.transcripts[0].text, "received 16000 samples");
    }

    #[test]
    fn frame_duration_uses_observed_media_rate() {
        assert_eq!(frame_duration_ms(160, 8_000), 20);
        assert_eq!(frame_duration_ms(320, 16_000), 20);
        assert_eq!(frame_duration_ms(320, 0), 0);
    }

    #[test]
    fn media_rollup_stats_count_transport_confounders() {
        let mut inbound = InboundTransportStats::default();
        inbound.observe_packet(1, 20);
        inbound.observe_packet(3, 20);
        inbound.observe_packet(2, 20);
        inbound.observe_stale();
        let payload = inbound.rollup_payload(None, None);
        assert_eq!(payload["packets_total"], 3);
        assert_eq!(payload["lost_packets"], 1);
        assert_eq!(payload["reordered_frames"], 1);
        assert_eq!(payload["stale_frames"], 1);

        let mut outbound = OutboundPacingStats::default();
        outbound.observe_frame(Some(20), 4);
        outbound.observe_frame(Some(50), 2);
        outbound.observe_underrun();
        let payload = outbound.rollup_payload(Some("tts_test"));
        assert_eq!(payload["playback_id"], "tts_test");
        assert_eq!(payload["frames_sent"], 2);
        assert_eq!(payload["underrun_count"], 1);
        assert_eq!(payload["inter_frame_gap_ms_max"], 50);
    }

    #[test]
    fn asr_gate_finalize_carries_endpoint_timing_boundaries() {
        let mut gate = AsrGate::default();
        let mut quality = VoiceQualityConfig::default();
        quality.endpoint.trailing_silence_ms = 20;
        let speech = SampleStats {
            peak: 4_000,
            rms: 4_000.0,
            mean: 0.0,
        };
        let silence = SampleStats {
            peak: 0,
            rms: 0.0,
            mean: 0.0,
        };

        assert_eq!(
            gate.accept(1, "stream-1", 20, &speech, &quality),
            AsrFrameDecision::Continue { speech_onset: true }
        );
        assert_eq!(
            gate.accept(2, "stream-1", 20, &silence, &quality),
            AsrFrameDecision::Continue {
                speech_onset: false,
            }
        );
        match gate.accept(3, "stream-1", 20, &silence, &quality) {
            AsrFrameDecision::Finalize {
                trailing_silence_ms,
                endpoint_wait_started_at,
                speech_to_low_energy,
            } => {
                assert_eq!(trailing_silence_ms, 40);
                assert!(endpoint_wait_started_at.is_some());
                assert!(speech_to_low_energy.is_some());
            }
            other => panic!("expected finalize, got {other:?}"),
        }
    }

    #[test]
    fn asr_gate_marks_speech_onset_after_short_pause() {
        let mut gate = AsrGate::default();
        let quality = VoiceQualityConfig::default();
        let speech = SampleStats {
            peak: 4_000,
            rms: 4_000.0,
            mean: 0.0,
        };
        let silence = SampleStats {
            peak: 0,
            rms: 0.0,
            mean: 0.0,
        };

        assert_eq!(
            gate.accept(1, "stream-1", 20, &speech, &quality),
            AsrFrameDecision::Continue { speech_onset: true }
        );
        for frame_index in 2..=7 {
            assert_eq!(
                gate.accept(frame_index, "stream-1", 20, &silence, &quality),
                AsrFrameDecision::Continue {
                    speech_onset: false,
                }
            );
        }
        assert_eq!(
            gate.accept(8, "stream-1", 20, &speech, &quality),
            AsrFrameDecision::Continue { speech_onset: true }
        );
        assert_eq!(
            gate.accept(9, "stream-1", 20, &speech, &quality),
            AsrFrameDecision::Continue {
                speech_onset: false,
            }
        );
    }

    #[tokio::test]
    async fn low_energy_media_finishes_after_local_endpoint() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let _gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr = registry_with_factory(counting_asr.clone());
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open ASR session");

        let speech = STANDARD.encode(l16_samples(320, 4_000));
        handle_text(
            &media_event("stream-1", "1", &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("speech should pass into ASR");

        let endpoint_frames = local_endpoint_silence_frames();
        let silence = STANDARD.encode(l16_samples(320, 0));
        for sequence in 2..=(endpoint_frames + 3) {
            handle_text(
                &media_event("stream-1", &sequence.to_string(), &silence),
                &state,
                &asr,
                &mut media_state,
            )
            .await
            .expect("silence should be accepted by transport");
        }

        assert_eq!(counting_asr.ingests(), 1 + endpoint_frames);
        assert_eq!(counting_asr.finishes(), 1);
    }

    #[tokio::test]
    async fn speech_resume_before_local_endpoint_keeps_asr_session() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let _gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr = registry_with_factory(counting_asr.clone());
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open ASR session");

        let speech = STANDARD.encode(l16_samples(320, 4_000));
        handle_text(
            &media_event("stream-1", "1", &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("first speech frame should pass into ASR");

        let short_pause_frames = local_endpoint_silence_frames() - 1;
        let silence = STANDARD.encode(l16_samples(320, 0));
        for offset in 0..short_pause_frames {
            let sequence = 2 + offset;
            handle_text(
                &media_event("stream-1", &sequence.to_string(), &silence),
                &state,
                &asr,
                &mut media_state,
            )
            .await
            .expect("short silence should stay in the ASR session");
        }

        handle_text(
            &media_event("stream-1", &(2 + short_pause_frames).to_string(), &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("resumed speech should continue ASR ingestion");

        assert_eq!(counting_asr.opens(), 1);
        assert_eq!(counting_asr.finishes(), 0);
        assert_eq!(counting_asr.ingests(), 2 + short_pause_frames);
    }

    #[tokio::test]
    async fn speech_resume_after_local_endpoint_opens_fresh_asr_session() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let _gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr = registry_with_factory(counting_asr.clone());
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open ASR session");

        let speech = STANDARD.encode(l16_samples(320, 4_000));
        handle_text(
            &media_event("stream-1", "1", &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("first speech frame should pass into ASR");

        let endpoint_frames = local_endpoint_silence_frames();
        let silence = STANDARD.encode(l16_samples(320, 0));
        for sequence in 2..=(endpoint_frames + 3) {
            handle_text(
                &media_event("stream-1", &sequence.to_string(), &silence),
                &state,
                &asr,
                &mut media_state,
            )
            .await
            .expect("silence should be accepted by transport");
        }
        assert_eq!(counting_asr.opens(), 1);
        assert_eq!(counting_asr.finishes(), 1);

        handle_text(
            &media_event("stream-1", &(endpoint_frames + 4).to_string(), &speech),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("resumed speech should open a fresh ASR session");

        assert_eq!(counting_asr.opens(), 2);
        assert_eq!(counting_asr.ingests(), 2 + endpoint_frames);

        let guard = state.read().await;
        let call = guard.calls.get(&_gateway_call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::MediaStarted);
    }

    #[tokio::test]
    async fn adapter_suppressed_transcripts_are_suppressed_from_call_detail() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let format = MediaFormat {
            encoding: "PCMU".to_string(),
            sample_rate_hz: 8_000,
            channels: 1,
        };

        let needs_reset = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![
                AsrTranscriptEvent::suppress(
                    TranscriptEvent::Partial {
                        text: "MEQQQQQQQQQQQQQQQQQQQQQQQQQQQQ".to_string(),
                        update: motlie_model::TranscriptionUpdate::default(),
                    },
                    AsrTranscriptSuppressionReason::RepeatedTokenHallucination,
                    true,
                ),
                AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "hello there".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                }),
            ],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: None,
                quality_session: None,
            },
        )
        .await;
        assert!(needs_reset);

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.transcripts.len(), 1);
        assert_eq!(call.transcripts[0].text, "hello there");
    }

    #[test]
    fn conversation_events_from_transcripts_forwards_unsuppressed_partials_and_finals() {
        let events = vec![
            AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                text: "partial".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            }),
            AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "forward final".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            }),
        ];

        let forwarded = conversation_events_from_transcripts(&events);

        assert_eq!(forwarded.len(), 2);
        assert!(!forwarded[0].is_final());
        assert_eq!(forwarded[0].text(), "partial");
        assert!(forwarded[1].is_final());
        assert_eq!(forwarded[1].text(), "forward final");
    }

    #[tokio::test]
    async fn non_sherpa_pass_through_transcripts_are_not_suppressed_by_media_loop() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let format = MediaFormat {
            encoding: "PCMU".to_string(),
            sample_rate_hz: 8_000,
            channels: 1,
        };
        let repeated_text = "MEQQQQQQQQQQQQQQQQQQQQQQQQQQQQ";

        let needs_reset = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: repeated_text.to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: None,
                quality_session: None,
            },
        )
        .await;
        assert!(!needs_reset);

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.transcripts.len(), 1);
        assert_eq!(call.transcripts[0].text, repeated_text);
    }

    #[tokio::test]
    async fn media_start_requires_operator_answer_gate() {
        for status in [CallStatus::IgnoredInbound, CallStatus::PendingInbound] {
            let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
            let gateway_call_id = seed_call(&state, "call-1", status).await;
            let counting_asr = Arc::new(CountingAsrFactory::default());
            let asr = registry_with_factory(counting_asr.clone());
            let mut media_state = MediaSocketState::new();

            handle_text(
                &start_event("call-1", "stream-1", "L16"),
                &state,
                &asr,
                &mut media_state,
            )
            .await
            .expect("start should be ignored before operator answer");

            assert!(media_state.session.is_none());
            assert!(media_state.gateway_call_id.is_none());
            assert!(media_state.media_format.is_none());
            assert_eq!(counting_asr.opens(), 0);
            let guard = state.read().await;
            let call = guard.calls.get(&gateway_call_id).expect("call exists");
            assert_eq!(call.status, status);
            assert!(call.ids.stream_id.is_none());
            assert!(call.transcripts.is_empty());
        }
    }

    #[tokio::test]
    async fn media_start_for_unknown_call_does_not_allocate_asr() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr = registry_with_factory(counting_asr.clone());
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("missing-call", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("unknown start should be ignored");

        assert!(media_state.session.is_none());
        assert!(media_state.gateway_call_id.is_none());
        assert!(media_state.media_format.is_none());
        assert_eq!(counting_asr.opens(), 0);
        assert!(state.read().await.calls.is_empty());
    }

    #[tokio::test]
    async fn unsupported_codec_is_rejected_before_asr_allocates() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr = registry_with_factory(counting_asr.clone());
        let mut media_state = MediaSocketState::new();

        let error = handle_text(
            &start_event("call-1", "stream-1", "OPUS"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect_err("unsupported codec should fail at start");

        assert!(error
            .to_string()
            .contains("unsupported inbound media encoding"));
        assert!(media_state.session.is_none());
        assert!(media_state.gateway_call_id.is_none());
        assert!(media_state.media_format.is_none());
        assert_eq!(counting_asr.opens(), 0);
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::Answering);
        assert!(call.ids.stream_id.is_none());
        assert!(call.transcripts.is_empty());
    }

    #[tokio::test]
    async fn media_start_opens_call_bound_asr_backend() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            let gateway_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: None,
                },
                Some("<from-phone-number>".to_string()),
                Some("<to-phone-number>".to_string()),
                CallStatus::Answering,
            );
            let call = guard
                .calls
                .get_mut(&gateway_call_id)
                .expect("seeded call should exist");
            call.asr_backend = Some(LiveAsrBackend::Kroko2025);
        }
        let sherpa_2023 = Arc::new(CountingAsrFactory::default());
        let kroko_2025 = Arc::new(CountingAsrFactory::default());
        let asr = Arc::new(AsrRegistry::new(sherpa_2023.clone(), kroko_2025.clone()));
        let mut media_state = MediaSocketState::new();

        handle_text(
            &start_event("call-1", "stream-1", "L16"),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("start event should open call-bound ASR");

        assert_eq!(sherpa_2023.opens(), 0);
        assert_eq!(kroko_2025.opens(), 1);
        assert_eq!(media_state.asr_backend, Some(LiveAsrBackend::Kroko2025));
    }

    fn registry_with_factory(factory: SharedAsrFactory) -> SharedAsrRegistry {
        Arc::new(AsrRegistry::new(factory.clone(), factory))
    }

    fn l16_samples(count: usize, sample: i16) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn media_event(stream_id: &str, chunk: &str, payload: &str) -> String {
        serde_json::json!({
            "event": "media",
            "stream_id": stream_id,
            "media": {
                "track": "inbound",
                "chunk": chunk,
                "payload": payload
            }
        })
        .to_string()
    }

    fn start_event(call_control_id: &str, stream_id: &str, encoding: &str) -> String {
        serde_json::json!({
            "event": "start",
            "stream_id": stream_id,
            "start": {
                "call_control_id": call_control_id,
                "call_session_id": "sess-1",
                "media_format": {
                    "encoding": encoding,
                    "sample_rate": 16000,
                    "channels": 1
                }
            }
        })
        .to_string()
    }

    async fn seed_call(state: &SharedState, call_control_id: &str, status: CallStatus) -> String {
        let mut guard = state.write().await;
        guard.add_or_update_inbound_call(
            TelnyxIds {
                call_control_id: call_control_id.to_string(),
                call_session_id: Some("sess-1".to_string()),
                call_leg_id: Some("leg-1".to_string()),
                stream_id: None,
            },
            Some("<from-phone-number>".to_string()),
            Some("<to-phone-number>".to_string()),
            status,
        )
    }

    fn local_endpoint_silence_frames() -> usize {
        let frame_ms = SILENCE_KEEPALIVE_INTERVAL.as_millis() as u64;
        (VoiceQualityConfig::default().endpoint.trailing_silence_ms / frame_ms) as usize
    }

    #[derive(Default)]
    struct CountingAsrFactory {
        opens: AtomicUsize,
        ingests: Arc<AtomicUsize>,
        finishes: Arc<AtomicUsize>,
    }

    impl CountingAsrFactory {
        fn opens(&self) -> usize {
            self.opens.load(Ordering::SeqCst)
        }

        fn ingests(&self) -> usize {
            self.ingests.load(Ordering::SeqCst)
        }

        fn finishes(&self) -> usize {
            self.finishes.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl InboundAsrFactory for CountingAsrFactory {
        async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(CountingAsrSession {
                ingests: Arc::clone(&self.ingests),
                finishes: Arc::clone(&self.finishes),
            }))
        }
    }

    struct CountingAsrSession {
        ingests: Arc<AtomicUsize>,
        finishes: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InboundAsrSession for CountingAsrSession {
        async fn ingest(
            &mut self,
            _audio: AudioBuf<i16, 16_000, Mono>,
        ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            self.ingests.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }

        async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            self.finishes.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }
    }
}
