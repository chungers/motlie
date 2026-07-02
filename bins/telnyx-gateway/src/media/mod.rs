use anyhow::{bail, Context};
use axum::extract::ws::{Message, WebSocket};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chrono::Utc;
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
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::time::{self, MissedTickBehavior};

use crate::adapter::{
    AsrTranscriptEvent, AsrTranscriptSuppressionReason, InboundAsrSession, LiveAsrBackend,
    SharedAsrRegistry,
};
use crate::call_control::{TelnyxMediaConfig, TelnyxStreamCodec};
use crate::conversation::{self, ConversationRuntime};
use crate::conversation_policy::{
    ActivePlaybackTarget, AudioBargeInDecisionInput, AudioEvidenceInvalidation, AudioOnsetDecision,
    BargeInTrigger, CallerOnsetEvidence, PlaybackEchoState,
};
use crate::early_response::{
    spawn_early_response_pipeline, EarlyResponseCancelReason, EarlyResponseCommitBoundary,
    EarlyResponseCommitMember, EarlyResponseInput, EarlyResponsePartial,
    EarlyResponsePipelineHandle, EarlyResponsePipelineServices,
};
use crate::echo_match::{match_assistant_echo_signature, AssistantEchoMatch};
use crate::operator::state::{
    CallSession, CallStatus, GatewayState, LogLevel, MediaMetadata, QualitySpanEmission,
    SharedState, StreamAttachOutcome, TranscriptKind, TtsPlaybackState, TtsPlaybackStatus,
};
use crate::processors::ConversationProcessorKind;
use crate::quality::{
    insert_transcript_text_fields, transcript_plaintext_included, ActiveAsrQualitySession,
    AudioBargeInMediaQualityConfig, AudioBargeInMode, CallerTurnEventMetadata,
    ConversationPolicyMode, EchoCharacterizationQualityConfig, EchoSuppressionQualityConfig,
    OnsetDuringPlaybackPolicy, RedactionMode, SpeechQualityConfig, VoiceQualityConfig,
};
use crate::speech;
use crate::text_calls::turns::{CallerSpeechState, PlaybackFinishedStatus};
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
    pub request_started_at: Instant,
    pub turn_finalized_at: Option<Instant>,
    pub latest_turn_finalized_at: Option<Instant>,
    pub processor_visible_turn_at: Option<Instant>,
    pub barge_in_cancel_terminal_at: Option<Instant>,
    pub turn_id: Option<String>,
    pub coalesced_turn_ids: Vec<String>,
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
    Mark {
        playback_id: String,
    },
    AppendState {
        playback_id: String,
        open: bool,
        empty: bool,
    },
}

impl OutboundMediaCommand {
    fn playback_id(&self) -> &str {
        match self {
            Self::Frame(frame) => &frame.playback_id,
            Self::Clear { playback_id, .. }
            | Self::Mark { playback_id }
            | Self::AppendState { playback_id, .. } => playback_id,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpeechCancelToken {
    canceled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl SpeechCancelToken {
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::SeqCst)
    }

    pub async fn canceled(&self) {
        if self.is_canceled() {
            return;
        }
        self.notify.notified().await;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSpeechPlaybackRef {
    pub playback_id: String,
    pub playback_epoch: u64,
}

#[derive(Clone, Debug)]
struct ActiveSpeechJob {
    playback_id: String,
    playback_epoch: u64,
    cancel: SpeechCancelToken,
}

#[derive(Clone, Debug)]
struct RecentSpeechPlayback {
    playback_id: String,
    terminal_at: Instant,
}

#[derive(Clone)]
struct MediaRegistryEntry {
    tx: mpsc::Sender<OutboundMediaCommand>,
    next_playback_epoch: u64,
    active_speech: Option<ActiveSpeechJob>,
    pending_clears: VecDeque<PendingClear>,
    recent_speech: VecDeque<RecentSpeechPlayback>,
    latest_caller_onset_evidence: Option<CallerOnsetEvidence>,
}

#[derive(Clone, Default)]
pub struct SharedMediaRegistry {
    inner: Arc<Mutex<HashMap<String, MediaRegistryEntry>>>,
}

const RECENT_SPEECH_PLAYBACK_LIMIT: usize = 8;

fn record_recent_speech(entry: &mut MediaRegistryEntry, playback_id: String) {
    entry.recent_speech.push_back(RecentSpeechPlayback {
        playback_id,
        terminal_at: Instant::now(),
    });
    while entry.recent_speech.len() > RECENT_SPEECH_PLAYBACK_LIMIT {
        entry.recent_speech.pop_front();
    }
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
                next_playback_epoch: 1,
                active_speech: None,
                pending_clears: VecDeque::new(),
                recent_speech: VecDeque::new(),
                latest_caller_onset_evidence: None,
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
        let playback_epoch = entry.next_playback_epoch;
        entry.next_playback_epoch = entry.next_playback_epoch.saturating_add(1);
        entry.active_speech = Some(ActiveSpeechJob {
            playback_id,
            playback_epoch,
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
            record_recent_speech(entry, active.playback_id.clone());
            active.playback_id
        });
        let playback_epoch = entry.next_playback_epoch;
        entry.next_playback_epoch = entry.next_playback_epoch.saturating_add(1);
        entry.active_speech = Some(ActiveSpeechJob {
            playback_id,
            playback_epoch,
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
        record_recent_speech(entry, active.playback_id.clone());
        entry.pending_clears.push_back(PendingClear {
            playback_id: active.playback_id.clone(),
            requested_at: Instant::now(),
            reason,
        });
        Ok(active.playback_id)
    }

    pub async fn cancel_speech_playback_for_reason(
        &self,
        gateway_call_id: &str,
        playback_id: &str,
        reason: SpeechClearReason,
    ) -> anyhow::Result<bool> {
        let mut guard = self.inner.lock().await;
        let entry = guard
            .get_mut(gateway_call_id)
            .with_context(|| format!("media stream is not ready for call {gateway_call_id}"))?;
        let Some(active) = entry.active_speech.as_ref() else {
            return Ok(false);
        };
        if active.playback_id != playback_id {
            return Ok(false);
        }
        let Some(active) = entry.active_speech.take() else {
            return Ok(false);
        };
        active.cancel.cancel();
        record_recent_speech(entry, active.playback_id.clone());
        entry.pending_clears.push_back(PendingClear {
            playback_id: active.playback_id,
            requested_at: Instant::now(),
            reason,
        });
        Ok(true)
    }

    pub async fn cancel_speech_playback_ref_for_reason(
        &self,
        gateway_call_id: &str,
        playback: &ActiveSpeechPlaybackRef,
        reason: SpeechClearReason,
    ) -> anyhow::Result<bool> {
        let mut guard = self.inner.lock().await;
        let entry = guard
            .get_mut(gateway_call_id)
            .with_context(|| format!("media stream is not ready for call {gateway_call_id}"))?;
        let Some(active) = entry.active_speech.as_ref() else {
            return Ok(false);
        };
        if active.playback_id != playback.playback_id
            || active.playback_epoch != playback.playback_epoch
        {
            return Ok(false);
        }
        let Some(active) = entry.active_speech.take() else {
            return Ok(false);
        };
        active.cancel.cancel();
        record_recent_speech(entry, active.playback_id.clone());
        entry.pending_clears.push_back(PendingClear {
            playback_id: active.playback_id,
            requested_at: Instant::now(),
            reason,
        });
        Ok(true)
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
        self.active_speech_playback_ref(gateway_call_id)
            .await
            .map(|active| active.playback_id)
    }

    pub async fn active_speech_playback_ref(
        &self,
        gateway_call_id: &str,
    ) -> Option<ActiveSpeechPlaybackRef> {
        self.inner
            .lock()
            .await
            .get(gateway_call_id)?
            .active_speech
            .as_ref()
            .map(|active| ActiveSpeechPlaybackRef {
                playback_id: active.playback_id.clone(),
                playback_epoch: active.playback_epoch,
            })
    }

    pub async fn record_caller_onset_evidence(
        &self,
        gateway_call_id: &str,
        evidence: CallerOnsetEvidence,
    ) {
        if let Some(entry) = self.inner.lock().await.get_mut(gateway_call_id) {
            entry.latest_caller_onset_evidence = Some(evidence);
        }
    }

    pub async fn latest_caller_onset_evidence(
        &self,
        gateway_call_id: &str,
    ) -> Option<CallerOnsetEvidence> {
        self.inner
            .lock()
            .await
            .get(gateway_call_id)?
            .latest_caller_onset_evidence
            .clone()
    }

    async fn recent_speech_playback_age(
        &self,
        gateway_call_id: &str,
        recent_window: Duration,
    ) -> Option<Duration> {
        let guard = self.inner.lock().await;
        let entry = guard.get(gateway_call_id)?;
        entry.recent_speech.iter().rev().find_map(|recent| {
            let age = recent.terminal_at.elapsed();
            (age <= recent_window).then_some(age)
        })
    }

    pub async fn speech_playback_active_or_recent(
        &self,
        gateway_call_id: &str,
        playback_id: &str,
        recent_window: Duration,
    ) -> bool {
        let guard = self.inner.lock().await;
        let Some(entry) = guard.get(gateway_call_id) else {
            return false;
        };
        if entry
            .active_speech
            .as_ref()
            .is_some_and(|active| active.playback_id == playback_id)
        {
            return true;
        }
        entry.recent_speech.iter().rev().any(|recent| {
            recent.playback_id == playback_id && recent.terminal_at.elapsed() <= recent_window
        })
    }

    pub async fn finish_speech(&self, gateway_call_id: &str, playback_id: &str) {
        if let Some(entry) = self.inner.lock().await.get_mut(gateway_call_id) {
            if entry
                .active_speech
                .as_ref()
                .is_some_and(|active| active.playback_id == playback_id)
            {
                let Some(active) = entry.active_speech.take() else {
                    return;
                };
                record_recent_speech(entry, active.playback_id);
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct InboundTransportSnapshot {
    packets_total: u64,
    lost_packets: u64,
    stale_frames: u64,
    reordered_frames: u64,
    jitter_samples: usize,
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

    fn snapshot(&self) -> InboundTransportSnapshot {
        InboundTransportSnapshot {
            packets_total: self.packets_total,
            lost_packets: self.lost_packets,
            stale_frames: self.stale_frames,
            reordered_frames: self.reordered_frames,
            jitter_samples: self.jitter_samples_ms.len(),
        }
    }

    fn max_jitter_since(&self, sample_index: usize) -> u64 {
        let start = sample_index.min(self.jitter_samples_ms.len());
        self.jitter_samples_ms[start..]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
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
    append_starvation_ticks: u64,
    post_mark_wait_ticks: u64,
    pre_audio_wait_ticks: u64,
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

    fn observe_append_starvation(&mut self) {
        self.append_starvation_ticks = self.append_starvation_ticks.saturating_add(1);
    }

    fn observe_post_mark_wait(&mut self) {
        self.post_mark_wait_ticks = self.post_mark_wait_ticks.saturating_add(1);
    }

    fn observe_pre_audio_wait(&mut self) {
        self.pre_audio_wait_ticks = self.pre_audio_wait_ticks.saturating_add(1);
    }

    fn rollup_payload(&self, playback_id: Option<&str>) -> Map<String, Value> {
        map_from_value(json!({
            "playback_id": playback_id,
            "frames_sent": self.frames_sent,
            "underrun_count": self.underrun_count,
            "append_starvation_ticks": self.append_starvation_ticks,
            "append_starvation_ms_estimate": self.append_starvation_ticks.saturating_mul(20),
            "post_mark_wait_ticks": self.post_mark_wait_ticks,
            "post_mark_wait_ms_estimate": self.post_mark_wait_ticks.saturating_mul(20),
            "pre_audio_wait_ticks": self.pre_audio_wait_ticks,
            "pre_audio_wait_ms_estimate": self.pre_audio_wait_ticks.saturating_mul(20),
            "inter_frame_gap_ms_p50": percentile_u64(&self.inter_frame_gap_ms, 50),
            "inter_frame_gap_ms_p95": percentile_u64(&self.inter_frame_gap_ms, 95),
            "inter_frame_gap_ms_max": self.inter_frame_gap_ms.iter().copied().max().unwrap_or(0),
            "queue_depth_p50": percentile_u64(&self.queue_depth_samples, 50),
            "queue_depth_p95": percentile_u64(&self.queue_depth_samples, 95),
        }))
    }
}

#[derive(Clone, Debug)]
struct EchoCharacterizationReferenceFrame {
    playback_id: String,
    sent_at: Instant,
    sample_rate_hz: u32,
    samples: Vec<i16>,
}

#[derive(Debug, Default)]
struct EchoCharacterizationState {
    outbound_reference: VecDeque<EchoCharacterizationReferenceFrame>,
    last_emit_at: Option<Instant>,
}

impl EchoCharacterizationState {
    fn observe_outbound_frame_with_retention(
        &mut self,
        playback_id: &str,
        sample_rate_hz: u32,
        samples: Vec<i16>,
        now: Instant,
        retention: Duration,
    ) {
        if samples.is_empty() {
            return;
        }
        self.outbound_reference
            .push_back(EchoCharacterizationReferenceFrame {
                playback_id: playback_id.to_string(),
                sent_at: now,
                sample_rate_hz,
                samples,
            });
        self.prune_to_retention(now, retention);
    }

    fn prune_to_retention(&mut self, now: Instant, retention: Duration) {
        while self
            .outbound_reference
            .front()
            .is_some_and(|frame| now.saturating_duration_since(frame.sent_at) > retention)
        {
            self.outbound_reference.pop_front();
        }
    }

    fn should_emit(&self, now: Instant, config: &EchoCharacterizationQualityConfig) -> bool {
        self.last_emit_at.is_none_or(|last_emit_at| {
            now.duration_since(last_emit_at) >= Duration::from_millis(config.emit_interval_ms)
        })
    }

    fn mark_emitted(&mut self, now: Instant) {
        self.last_emit_at = Some(now);
    }

    fn latest_playback_id(&self) -> Option<&str> {
        self.outbound_reference
            .back()
            .map(|frame| frame.playback_id.as_str())
    }

    fn reference_samples(
        &self,
        sample_rate_hz: u32,
        config: &EchoCharacterizationQualityConfig,
    ) -> Vec<i16> {
        self.reference_samples_for(
            sample_rate_hz,
            samples_for_ms(
                config.window_ms.saturating_add(config.max_delay_ms),
                sample_rate_hz,
            ),
        )
    }

    fn reference_samples_for(&self, sample_rate_hz: u32, max_samples: usize) -> Vec<i16> {
        let mut frames = Vec::new();
        let mut sample_count = 0usize;
        for frame in self.outbound_reference.iter().rev() {
            if frame.sample_rate_hz != sample_rate_hz {
                continue;
            }
            frames.push(frame);
            sample_count = sample_count.saturating_add(frame.samples.len());
            if sample_count >= max_samples {
                break;
            }
        }
        let mut samples = Vec::with_capacity(sample_count.min(max_samples));
        for frame in frames.into_iter().rev() {
            samples.extend_from_slice(&frame.samples);
        }
        if samples.len() > max_samples {
            samples[samples.len().saturating_sub(max_samples)..].to_vec()
        } else {
            samples
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EchoCharacterizationMetrics {
    correlation_peak: f32,
    estimated_delay_ms: u64,
    inbound_rms_dbfs: f32,
    outbound_rms_dbfs: f32,
    echo_return_db: f32,
}

#[derive(Clone, Copy, Debug)]
struct AudioEchoCalibration {
    erl_baseline_db: f32,
    delay_ms: f32,
    playback_only_ms: u64,
    update_count: u64,
}

#[derive(Debug, Default)]
struct AudioBargeInEvidenceState {
    calibration: Option<AudioEchoCalibration>,
    consecutive_trusted_windows: usize,
    caller_active_since: Option<Instant>,
    last_trusted_onset: Option<(String, u64)>,
    last_transport_snapshot: InboundTransportSnapshot,
}

#[derive(Clone, Copy)]
struct ActivePlaybackWindow<'a> {
    config: &'a AudioBargeInMediaQualityConfig,
    playback_ref: &'a ActiveSpeechPlaybackRef,
    format: &'a MediaFormat,
    samples: &'a [i16],
    stats: &'a SampleStats,
    speech: &'a SpeechQualityConfig,
    outbound_reference: &'a [i16],
    transport_invalidation: Option<AudioEvidenceInvalidation>,
    frame_duration_ms: u64,
    now: Instant,
}

#[derive(Clone, Copy, Debug)]
struct ResidualEchoMetrics {
    residual_rms_dbfs: f32,
    residual_peak: i16,
    residual_rms: f32,
}

impl ResidualEchoMetrics {
    fn has_speech_energy(self, speech: &SpeechQualityConfig) -> bool {
        self.residual_rms >= speech.rms_threshold
            || i32::from(self.residual_peak) >= speech.peak_threshold
    }
}

struct CallerOnsetEvidenceSeed<'a> {
    decision: AudioOnsetDecision,
    confidence: f32,
    playback_state: PlaybackEchoState,
    playback_ref: Option<&'a ActiveSpeechPlaybackRef>,
    caller_active_since: Option<Instant>,
    now: Instant,
    window_ms: u32,
    format: &'a MediaFormat,
    inbound_rms_dbfs: f32,
    outbound_rms_dbfs: Option<f32>,
    invalidation: Option<AudioEvidenceInvalidation>,
}

struct AudioBargeInFrame<'a> {
    gateway_call_id: &'a str,
    stream_id: &'a str,
    format: &'a MediaFormat,
    samples: &'a [i16],
    stats: &'a SampleStats,
    frame_duration_ms: u64,
    caller_candidate_active: bool,
}

impl AudioBargeInEvidenceState {
    fn transport_invalidation(
        &mut self,
        transport: &InboundTransportStats,
        config: &AudioBargeInMediaQualityConfig,
    ) -> Option<AudioEvidenceInvalidation> {
        let snapshot = transport.snapshot();
        let packets = snapshot
            .packets_total
            .saturating_sub(self.last_transport_snapshot.packets_total)
            .max(1);
        let invalid_frames = snapshot
            .lost_packets
            .saturating_sub(self.last_transport_snapshot.lost_packets)
            .saturating_add(
                snapshot
                    .reordered_frames
                    .saturating_sub(self.last_transport_snapshot.reordered_frames),
            )
            .saturating_add(
                snapshot
                    .stale_frames
                    .saturating_sub(self.last_transport_snapshot.stale_frames),
            );
        let window_max_jitter_ms =
            transport.max_jitter_since(self.last_transport_snapshot.jitter_samples);
        self.last_transport_snapshot = snapshot;
        let invalid_ratio = invalid_frames as f32 / packets as f32;
        if invalid_ratio > config.max_invalid_frame_ratio
            || window_max_jitter_ms > config.max_jitter_ms
        {
            Some(AudioEvidenceInvalidation::TransportInvalid)
        } else {
            None
        }
    }

    fn observe_playback_only_calibration(
        &mut self,
        config: &AudioBargeInMediaQualityConfig,
        frame_duration_ms: u64,
        metrics: EchoCharacterizationMetrics,
    ) {
        if metrics.echo_return_db < config.erl_min_db || metrics.echo_return_db > config.erl_max_db
        {
            return;
        }
        if metrics.estimated_delay_ms < config.delay_search_min_ms
            || metrics.estimated_delay_ms > config.delay_search_max_ms
        {
            return;
        }
        let alpha = config.calibration_ema_alpha;
        let next = if let Some(current) = self.calibration {
            AudioEchoCalibration {
                erl_baseline_db: current
                    .erl_baseline_db
                    .mul_add(1.0 - alpha, metrics.echo_return_db * alpha),
                delay_ms: current
                    .delay_ms
                    .mul_add(1.0 - alpha, metrics.estimated_delay_ms as f32 * alpha),
                playback_only_ms: current.playback_only_ms.saturating_add(frame_duration_ms),
                update_count: current.update_count.saturating_add(1),
            }
        } else {
            AudioEchoCalibration {
                erl_baseline_db: metrics.echo_return_db,
                delay_ms: metrics.estimated_delay_ms as f32,
                playback_only_ms: frame_duration_ms,
                update_count: 1,
            }
        };
        self.calibration = Some(next);
    }

    fn classify_active_playback_window(
        &mut self,
        window: ActivePlaybackWindow<'_>,
    ) -> CallerOnsetEvidence {
        let window_ms = window.frame_duration_ms.min(u64::from(u32::MAX)) as u32;
        let mut evidence = base_caller_onset_evidence(CallerOnsetEvidenceSeed {
            decision: AudioOnsetDecision::Unavailable,
            confidence: 0.0,
            playback_state: PlaybackEchoState::ActivePlayback,
            playback_ref: Some(window.playback_ref),
            caller_active_since: None,
            now: window.now,
            window_ms,
            format: window.format,
            inbound_rms_dbfs: rms_dbfs(window.samples),
            outbound_rms_dbfs: None,
            invalidation: window.transport_invalidation,
        });
        if evidence.invalidation.is_some() {
            self.consecutive_trusted_windows = 0;
            evidence.decision = AudioOnsetDecision::Ambiguous;
            return evidence;
        }
        let min_reference = window.samples.len().saturating_add(samples_for_ms(
            window.config.delay_search_max_ms,
            window.format.sample_rate_hz,
        ));
        if window.outbound_reference.len() < min_reference {
            self.consecutive_trusted_windows = 0;
            evidence.invalidation = Some(AudioEvidenceInvalidation::ShortReference);
            return evidence;
        }
        let Some(metrics) = characterize_echo(
            window.samples,
            window.outbound_reference,
            window.format.sample_rate_hz,
            window.config.delay_search_max_ms,
        ) else {
            self.consecutive_trusted_windows = 0;
            evidence.invalidation = Some(AudioEvidenceInvalidation::ShortReference);
            return evidence;
        };
        evidence.estimated_delay_ms =
            Some(metrics.estimated_delay_ms.min(u64::from(u32::MAX)) as u32);
        evidence.correlation_peak = Some(metrics.correlation_peak);
        if metrics.estimated_delay_ms < window.config.delay_search_min_ms
            || metrics.estimated_delay_ms > window.config.delay_search_max_ms
        {
            self.consecutive_trusted_windows = 0;
            evidence.decision = AudioOnsetDecision::Ambiguous;
            evidence.invalidation = Some(AudioEvidenceInvalidation::DelayOutOfRange);
            return evidence;
        }
        let Some(calibration) = self.calibration else {
            self.consecutive_trusted_windows = 0;
            evidence.invalidation = Some(AudioEvidenceInvalidation::CalibrationUnavailable);
            return evidence;
        };
        if calibration.playback_only_ms < window.config.calibration_min_playback_only_ms {
            self.consecutive_trusted_windows = 0;
            evidence.invalidation = Some(AudioEvidenceInvalidation::CalibrationUnavailable);
            return evidence;
        }
        let Some(calibrated_segment) = reference_segment_at_delay(
            window.samples.len(),
            window.outbound_reference,
            window.format.sample_rate_hz,
            calibration.delay_ms,
        ) else {
            self.consecutive_trusted_windows = 0;
            evidence.invalidation = Some(AudioEvidenceInvalidation::DelayOutOfRange);
            return evidence;
        };
        let delay_drift_ms = (metrics.estimated_delay_ms as f32 - calibration.delay_ms).abs();
        if metrics.correlation_peak >= window.config.min_speechlike_correlation
            && delay_drift_ms > window.config.calibrated_delay_tolerance_ms as f32
        {
            self.consecutive_trusted_windows = 0;
            evidence.decision = AudioOnsetDecision::Ambiguous;
            evidence.invalidation = Some(AudioEvidenceInvalidation::DelayOutOfRange);
            return evidence;
        }
        let calibrated_outbound_rms_dbfs = rms_dbfs(calibrated_segment);
        let calibrated_correlation = normalized_correlation(window.samples, calibrated_segment)
            .unwrap_or(metrics.correlation_peak);
        let echo_return_db = metrics.inbound_rms_dbfs - calibrated_outbound_rms_dbfs;
        evidence.outbound_rms_dbfs = Some(calibrated_outbound_rms_dbfs);
        evidence.correlation_peak = Some(calibrated_correlation);
        evidence.echo_return_db = Some(echo_return_db);
        let predicted_echo_dbfs = calibrated_outbound_rms_dbfs + calibration.erl_baseline_db;
        let echo_margin_db = metrics.inbound_rms_dbfs - predicted_echo_dbfs;
        evidence.echo_margin_db = Some(echo_margin_db);
        if echo_margin_db < window.config.min_echo_margin_db_floor {
            self.consecutive_trusted_windows = 0;
            evidence.decision = AudioOnsetDecision::LikelyAssistantEcho;
            evidence.confidence = calibrated_correlation.clamp(0.0, 1.0);
            return evidence;
        }
        let Some(residual) = residual_echo_metrics(window.samples, calibrated_segment) else {
            self.consecutive_trusted_windows = 0;
            evidence.decision = AudioOnsetDecision::Ambiguous;
            evidence.invalidation = Some(AudioEvidenceInvalidation::LowCorrelationSpeech);
            return evidence;
        };
        if !window.stats.has_speech_energy(window.speech)
            || !residual.has_speech_energy(window.speech)
        {
            self.consecutive_trusted_windows = 0;
            if calibrated_correlation >= window.config.min_speechlike_correlation {
                evidence.decision = AudioOnsetDecision::LikelyAssistantEcho;
                evidence.confidence = calibrated_correlation.clamp(0.0, 1.0);
            } else {
                evidence.decision = AudioOnsetDecision::Ambiguous;
                evidence.invalidation = if window.stats.has_speech_energy(window.speech) {
                    Some(AudioEvidenceInvalidation::LowCorrelationSpeech)
                } else {
                    Some(AudioEvidenceInvalidation::LowCorrelationNonSpeech)
                };
            }
            return evidence;
        }
        self.consecutive_trusted_windows = self.consecutive_trusted_windows.saturating_add(1);
        self.caller_active_since.get_or_insert(window.now);
        evidence.caller_active_since = self.caller_active_since;
        let residual_margin_db = (residual.residual_rms_dbfs - predicted_echo_dbfs).max(0.0);
        evidence.confidence = (echo_margin_db.max(residual_margin_db)
            / window.config.min_echo_margin_db_ceiling)
            .clamp(0.0, 1.0)
            .max(0.01);
        if self.consecutive_trusted_windows >= window.config.trusted_onset_min_windows {
            evidence.decision = AudioOnsetDecision::TrustedCallerOnset;
        } else {
            evidence.decision = AudioOnsetDecision::Ambiguous;
        }
        evidence
    }

    fn take_trusted_onset_edge(&mut self, evidence: &CallerOnsetEvidence) -> bool {
        if evidence.decision != AudioOnsetDecision::TrustedCallerOnset
            || evidence.playback_state != PlaybackEchoState::ActivePlayback
        {
            return false;
        }
        let Some(playback_id) = evidence.playback_id.as_ref() else {
            return false;
        };
        let Some(playback_epoch) = evidence.playback_epoch else {
            return false;
        };
        let key = (playback_id.clone(), playback_epoch);
        if self.last_trusted_onset.as_ref() == Some(&key) {
            return false;
        }
        self.last_trusted_onset = Some(key);
        true
    }

    fn reset_caller_active(&mut self) {
        self.consecutive_trusted_windows = 0;
        self.caller_active_since = None;
    }
}

fn base_caller_onset_evidence(seed: CallerOnsetEvidenceSeed<'_>) -> CallerOnsetEvidence {
    CallerOnsetEvidence {
        decision: seed.decision,
        confidence: seed.confidence,
        playback_state: seed.playback_state,
        playback_id: seed
            .playback_ref
            .map(|playback| playback.playback_id.clone()),
        playback_epoch: seed.playback_ref.map(|playback| playback.playback_epoch),
        caller_active_since: seed.caller_active_since,
        evidence_at: seed.now,
        evidence_age_ms: 0,
        window_ms: seed.window_ms,
        input_codec: seed.format.encoding.clone(),
        input_sample_rate_hz: seed.format.sample_rate_hz,
        evidence_sample_rate_hz: seed.format.sample_rate_hz,
        inbound_rms_dbfs: seed.inbound_rms_dbfs,
        outbound_rms_dbfs: seed.outbound_rms_dbfs,
        estimated_delay_ms: None,
        correlation_peak: None,
        echo_return_db: None,
        echo_margin_db: None,
        invalidation: seed.invalidation,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PlaybackPacingCounters {
    pre_audio_wait_ticks: usize,
    underrun_ticks: usize,
}

impl PlaybackPacingCounters {
    fn observe_pre_audio_wait(&mut self) {
        self.pre_audio_wait_ticks = self.pre_audio_wait_ticks.saturating_add(1);
    }

    fn observe_underrun(&mut self) {
        self.underrun_ticks = self.underrun_ticks.saturating_add(1);
    }
}

#[derive(Clone, Debug)]
struct PendingFinalTranscript {
    event: AsrTranscriptEvent,
    stream_id: Option<String>,
    quality_session: Option<ActiveAsrQualitySession>,
    created_at: Instant,
    flush_at: Instant,
    max_flush_at: Instant,
    hold_reason: &'static str,
    continuation_speech_seen: bool,
}

impl PendingFinalTranscript {
    fn new(
        event: AsrTranscriptEvent,
        stream_id: Option<&str>,
        quality_session: Option<&ActiveAsrQualitySession>,
        quality: &VoiceQualityConfig,
        hold_reason: &'static str,
    ) -> Self {
        let created_at = Instant::now();
        let settle = Duration::from_millis(quality.endpoint.final_settle_ms);
        let max_extra_ms = quality
            .endpoint
            .trailing_silence_ms
            .saturating_add(quality.asr.finish_pad_ms)
            .saturating_add(1_000);
        Self {
            event,
            stream_id: stream_id.map(str::to_string),
            quality_session: quality_session.cloned(),
            created_at,
            flush_at: created_at + settle,
            max_flush_at: created_at + settle + Duration::from_millis(max_extra_ms),
            hold_reason,
            continuation_speech_seen: false,
        }
    }

    fn is_ready(&self, now: Instant) -> bool {
        now >= self.max_flush_at || (now >= self.flush_at && !self.continuation_speech_seen)
    }
}

struct MediaSocketState {
    session: Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<String>,
    asr_backend: Option<LiveAsrBackend>,
    media_format: Option<MediaFormat>,
    active_quality_asr: Option<ActiveAsrQualitySession>,
    last_quality_asr: Option<ActiveAsrQualitySession>,
    pending_final: Option<PendingFinalTranscript>,
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
    text_calls: Option<SharedTextCallRegistry>,
    early_response: Option<EarlyResponsePipelineHandle>,
    outbound_frame_count: usize,
    outbound_underrun_ticks: usize,
    outbound_pre_audio_wait_ticks: usize,
    outbound_post_mark_wait_ticks: usize,
    outbound_pacing: OutboundPacingStats,
    echo_characterization: EchoCharacterizationState,
    audio_barge_in: AudioBargeInEvidenceState,
    playback_pacing_counters: HashMap<String, PlaybackPacingCounters>,
    last_outbound_frame_sent_at: Option<Instant>,
    first_frame_sent_playbacks: HashSet<String>,
    playback_started_at: HashMap<String, Instant>,
    playback_quality_contexts: HashMap<String, OutboundFrameQualityContext>,
    append_open_empty_playbacks: HashSet<String>,
    mark_sent_playbacks: HashSet<String>,
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
            pending_final: None,
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
            text_calls: None,
            early_response: None,
            outbound_frame_count: 0,
            outbound_underrun_ticks: 0,
            outbound_pre_audio_wait_ticks: 0,
            outbound_post_mark_wait_ticks: 0,
            outbound_pacing: OutboundPacingStats::default(),
            echo_characterization: EchoCharacterizationState::default(),
            audio_barge_in: AudioBargeInEvidenceState::default(),
            playback_pacing_counters: HashMap::new(),
            last_outbound_frame_sent_at: None,
            first_frame_sent_playbacks: HashSet::new(),
            playback_started_at: HashMap::new(),
            playback_quality_contexts: HashMap::new(),
            append_open_empty_playbacks: HashSet::new(),
            mark_sent_playbacks: HashSet::new(),
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
    media_state.text_calls = Some(text_calls.clone());
    let mut silence_keepalive = time::interval(SILENCE_KEEPALIVE_INTERVAL);
    silence_keepalive.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;

            _ = silence_keepalive.tick(), if media_state.silence_keepalive => {
                if let Err(error) = send_outbound_or_silence(&mut socket, &state, &mut media_state).await {
                    log_media_error(&state, media_state.gateway_call_id.as_deref(), error).await;
                    break;
                }
            }
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
    if let Some(handle) = media_state.early_response.take() {
        handle.cancel_call(EarlyResponseCancelReason::Hangup);
    }
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
        if !media_state
            .first_frame_sent_playbacks
            .contains(playback_id.as_str())
        {
            media_state.outbound_pre_audio_wait_ticks =
                media_state.outbound_pre_audio_wait_ticks.saturating_add(1);
            media_state.outbound_pacing.observe_pre_audio_wait();
            media_state
                .playback_pacing_counters
                .entry(playback_id.clone())
                .or_default()
                .observe_pre_audio_wait();
            if media_state.outbound_pre_audio_wait_ticks <= 5
                || media_state.outbound_pre_audio_wait_ticks.is_multiple_of(50)
            {
                tracing::debug!(
                    gateway_call_id = call_id,
                    playback_id = playback_id.as_str(),
                    pre_audio_wait_ticks = media_state.outbound_pre_audio_wait_ticks,
                    pre_audio_wait_ms_estimate =
                        media_state.outbound_pre_audio_wait_ticks.saturating_mul(20),
                    queue_depth = outbound_queue_depth(media_state),
                    "tts.outbound.pre_audio_wait"
                );
            }
            return Ok(());
        }

        if media_state
            .mark_sent_playbacks
            .contains(playback_id.as_str())
        {
            media_state.outbound_post_mark_wait_ticks =
                media_state.outbound_post_mark_wait_ticks.saturating_add(1);
            media_state.outbound_pacing.observe_post_mark_wait();
            if media_state.outbound_post_mark_wait_ticks <= 5
                || media_state.outbound_post_mark_wait_ticks.is_multiple_of(50)
            {
                tracing::debug!(
                    gateway_call_id = call_id,
                    playback_id = playback_id.as_str(),
                    post_mark_wait_ticks = media_state.outbound_post_mark_wait_ticks,
                    post_mark_wait_ms_estimate =
                        media_state.outbound_post_mark_wait_ticks.saturating_mul(20),
                    "tts.outbound.post_mark_wait"
                );
            }
            return Ok(());
        }

        media_state.outbound_underrun_ticks = media_state.outbound_underrun_ticks.saturating_add(1);
        media_state
            .playback_pacing_counters
            .entry(playback_id.clone())
            .or_default()
            .observe_underrun();
        let append_starved = media_state
            .append_open_empty_playbacks
            .contains(playback_id.as_str());
        if append_starved {
            media_state.outbound_pacing.observe_append_starvation();
        } else {
            media_state.outbound_pacing.observe_underrun();
        }
        if media_state.outbound_underrun_ticks <= 5
            || media_state.outbound_underrun_ticks.is_multiple_of(50)
        {
            if append_starved {
                tracing::warn!(
                    gateway_call_id = call_id,
                    playback_id = playback_id.as_str(),
                    append_starvation_ticks = media_state.outbound_underrun_ticks,
                    queue_depth = outbound_queue_depth(media_state),
                    "tts.outbound.append_starvation"
                );
            } else {
                tracing::warn!(
                    gateway_call_id = call_id,
                    playback_id = playback_id.as_str(),
                    underrun_ticks = media_state.outbound_underrun_ticks,
                    queue_depth = outbound_queue_depth(media_state),
                    "tts.outbound.underrun"
                );
            }
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
            capture_outbound_echo_reference(media_state, &frame);
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
            tracing::info!(
                gateway_call_id = call_id,
                playback_id,
                reason = reason.label(),
                "tts.clear.sent"
            );
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
            media_state.append_open_empty_playbacks.remove(&playback_id);
            media_state.mark_sent_playbacks.insert(playback_id.clone());
            tracing::info!(gateway_call_id = call_id, playback_id, "tts.mark.sent");
            Ok(())
        }
        OutboundMediaCommand::AppendState {
            playback_id,
            open,
            empty,
        } => {
            if open && empty {
                media_state
                    .append_open_empty_playbacks
                    .insert(playback_id.clone());
            } else {
                media_state.append_open_empty_playbacks.remove(&playback_id);
            }
            tracing::debug!(
                gateway_call_id = call_id,
                playback_id,
                open,
                empty,
                "tts.append.state"
            );
            Ok(())
        }
    }
}

fn capture_outbound_echo_reference(media_state: &mut MediaSocketState, frame: &OutboundMediaFrame) {
    let echo_config = media_state.quality_config.echo_characterization.clone();
    let audio_config = media_state.quality_config.audio_barge_in.media.clone();
    if !echo_config.enabled && audio_config.mode == AudioBargeInMode::MeasureOnly {
        return;
    }
    let Some(format) = media_state.media_format.as_ref() else {
        return;
    };
    let retention = outbound_reference_retention(&media_state.quality_config);
    match decode_payload(format, &frame.payload) {
        Ok(samples) => media_state
            .echo_characterization
            .observe_outbound_frame_with_retention(
                &frame.playback_id,
                format.sample_rate_hz,
                samples,
                Instant::now(),
                retention,
            ),
        Err(error) => tracing::debug!(
            playback_id = frame.playback_id.as_str(),
            error = %error,
            "media.echo_characterization.outbound_decode_skipped"
        ),
    }
}

async fn maybe_emit_echo_characterization_span(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    gateway_call_id: &str,
    stream_id: &str,
    format: &MediaFormat,
    samples: &[i16],
    inbound_stats: &SampleStats,
) {
    let config = media_state.quality_config.echo_characterization.clone();
    if !config.enabled {
        return;
    }
    let now = Instant::now();
    media_state.echo_characterization.prune_to_retention(
        now,
        outbound_reference_retention(&media_state.quality_config),
    );
    if !media_state.echo_characterization.should_emit(now, &config) {
        return;
    }
    let Some(playback_id) = media_state
        .echo_characterization
        .latest_playback_id()
        .map(str::to_string)
    else {
        return;
    };

    let media_registry = media_state.media_registry.clone();
    let active_playback_id = media_registry
        .active_speech_playback_id(gateway_call_id)
        .await;
    let is_active = active_playback_id.as_deref() == Some(playback_id.as_str());
    let is_recent = !is_active
        && media_registry
            .speech_playback_active_or_recent(
                gateway_call_id,
                &playback_id,
                echo_characterization_retention(&config),
            )
            .await;
    if !is_active && !is_recent {
        media_state.echo_characterization.mark_emitted(now);
        return;
    }

    let outbound_reference = media_state
        .echo_characterization
        .reference_samples(format.sample_rate_hz, &config);
    let Some(metrics) = characterize_echo(
        samples,
        &outbound_reference,
        format.sample_rate_hz,
        config.max_delay_ms,
    ) else {
        return;
    };
    media_state.echo_characterization.mark_emitted(now);

    let (config_id, redaction_mode, asr_session_id, utterance_id) = media_state
        .active_quality_asr
        .as_ref()
        .map(|session| {
            (
                session.config_id.clone(),
                session.redaction_mode,
                Some(session.asr_session_id.clone()),
                Some(session.utterance_id.clone()),
            )
        })
        .unwrap_or_else(|| {
            (
                media_state.quality_config.config_id(),
                media_state.quality_config.logging.redaction_mode,
                None,
                None,
            )
        });
    let payload = map_from_value(json!({
        "stream_id": stream_id,
        "playback_id": playback_id,
        "playback_state": if is_active { "active" } else { "recent" },
        "active_playback_id": active_playback_id.as_deref(),
        "asr_session_id": asr_session_id.as_deref(),
        "utterance_id": utterance_id.as_deref(),
        "codec": format.encoding.as_str(),
        "sample_rate_hz": format.sample_rate_hz,
        "inbound_samples": samples.len(),
        "outbound_reference_samples": outbound_reference.len(),
        "window_ms": config.window_ms,
        "max_delay_ms": config.max_delay_ms,
        "emit_interval_ms": config.emit_interval_ms,
        "inbound_peak": inbound_stats.peak,
        "inbound_rms": inbound_stats.rms,
        "inbound_rms_dbfs": metrics.inbound_rms_dbfs,
        "outbound_rms_dbfs": metrics.outbound_rms_dbfs,
        "echo_return_db": metrics.echo_return_db,
        "correlation_peak": metrics.correlation_peak,
        "estimated_delay_ms": metrics.estimated_delay_ms,
        "speech_rms_threshold": media_state.quality_config.speech.rms_threshold,
        "speech_peak_threshold": media_state.quality_config.speech.peak_threshold,
    }));
    state.write().await.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id,
            redaction_mode,
            span_name: "media.echo_characterization",
            category: "echo_characterization",
            duration: Duration::ZERO,
            critical_path: false,
            concurrent: true,
            payload,
        },
    );
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
    let queue_depth = outbound_queue_depth(media_state);
    let first_audio_latency_ms = quality.request_started_at.elapsed().as_millis() as u64;
    let pacing = take_playback_pacing_counters(media_state, &frame.playback_id);
    let first_frame_payload = map_from_value(json!({
        "playback_id": frame.playback_id.as_str(),
        "turn_id": quality.turn_id.as_deref(),
        "queue_depth": queue_depth,
    }));
    let first_audio_payload = map_from_value(json!({
        "playback_id": frame.playback_id.as_str(),
        "turn_id": quality.turn_id.as_deref(),
        "queue_depth": queue_depth,
        "request_to_enqueue_ms": quality
            .queued_at
            .saturating_duration_since(quality.request_started_at)
            .as_millis() as u64,
    }));
    let mut guard = state.write().await;
    guard.mark_tts_first_audio_latency_and_pacing(
        call_id,
        &frame.playback_id,
        first_audio_latency_ms,
        pacing.pre_audio_wait_ticks,
        pacing.underrun_ticks,
    );
    guard.emit_quality_span_finished(
        call_id,
        QualitySpanEmission {
            config_id: quality.config_id.clone(),
            redaction_mode: quality.redaction_mode,
            span_name: "media.first_frame_send",
            category: "playback_transport",
            duration: quality.queued_at.elapsed(),
            critical_path: true,
            concurrent: false,
            payload: first_frame_payload,
        },
    );
    guard.emit_quality_span_finished(
        call_id,
        QualitySpanEmission {
            config_id: quality.config_id.clone(),
            redaction_mode: quality.redaction_mode,
            span_name: "tts.request_to_first_audio",
            category: "tts_generation",
            duration: quality.request_started_at.elapsed(),
            critical_path: true,
            concurrent: false,
            payload: first_audio_payload,
        },
    );
    if let Some(visible_at) = quality.processor_visible_turn_at {
        let payload = map_from_value(json!({
            "playback_id": frame.playback_id.as_str(),
            "turn_id": quality.turn_id.as_deref(),
            "queue_depth": queue_depth,
            "processor_visible_to_request_ms": quality
                .request_started_at
                .saturating_duration_since(visible_at)
                .as_millis() as u64,
            "coalesced_turn_count": quality.coalesced_turn_ids.len(),
            "coalesced_turn_ids": quality.coalesced_turn_ids.as_slice(),
        }));
        guard.emit_quality_span_finished(
            call_id,
            QualitySpanEmission {
                config_id: quality.config_id.clone(),
                redaction_mode: quality.redaction_mode,
                span_name: "conversation.visible_turn_to_first_audio",
                category: "turn_taking",
                duration: visible_at.elapsed(),
                critical_path: true,
                concurrent: false,
                payload,
            },
        );
    }
    if let Some(cancel_terminal_at) = quality.barge_in_cancel_terminal_at {
        let payload = map_from_value(json!({
            "playback_id": frame.playback_id.as_str(),
            "turn_id": quality.turn_id.as_deref(),
            "queue_depth": queue_depth,
            "cancel_terminal_to_request_ms": quality
                .request_started_at
                .saturating_duration_since(cancel_terminal_at)
                .as_millis() as u64,
            "coalesced_turn_count": quality.coalesced_turn_ids.len(),
            "coalesced_turn_ids": quality.coalesced_turn_ids.as_slice(),
        }));
        guard.emit_quality_span_finished(
            call_id,
            QualitySpanEmission {
                config_id: quality.config_id.clone(),
                redaction_mode: quality.redaction_mode,
                span_name: "barge_in.cancel_terminal_to_replacement_first_audio",
                category: "barge_in",
                duration: cancel_terminal_at.elapsed(),
                critical_path: true,
                concurrent: false,
                payload,
            },
        );
    }
    if let Some(finalized_at) = quality.turn_finalized_at {
        let mut payload = map_from_value(json!({
            "playback_id": frame.playback_id.as_str(),
            "turn_id": quality.turn_id.as_deref(),
            "queue_depth": queue_depth,
            "handler_to_request_ms": quality
                .request_started_at
                .saturating_duration_since(finalized_at)
                .as_millis() as u64,
            "coalesced_turn_count": quality.coalesced_turn_ids.len(),
            "coalesced_turn_ids": quality.coalesced_turn_ids.as_slice(),
        }));
        if let Some(latest_finalized_at) = quality.latest_turn_finalized_at {
            payload.insert(
                "latest_turn_handler_to_request_ms".to_string(),
                Value::Number(serde_json::Number::from(
                    quality
                        .request_started_at
                        .saturating_duration_since(latest_finalized_at)
                        .as_millis() as u64,
                )),
            );
            payload.insert(
                "latest_turn_finalize_to_first_audio_ms".to_string(),
                Value::Number(serde_json::Number::from(
                    latest_finalized_at.elapsed().as_millis() as u64,
                )),
            );
        }
        guard.emit_quality_span_finished(
            call_id,
            QualitySpanEmission {
                config_id: quality.config_id.clone(),
                redaction_mode: quality.redaction_mode,
                span_name: "turn.finalize_to_first_audio",
                category: "turn_taking",
                duration: finalized_at.elapsed(),
                critical_path: true,
                concurrent: false,
                payload,
            },
        );
    }
}

async fn emit_playback_terminal_spans(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    call_id: &str,
    playback_id: &str,
    status: &'static str,
    clear: Option<(Instant, SpeechClearReason)>,
) {
    let quality = media_state
        .playback_quality_contexts
        .get(playback_id)
        .cloned();
    let (config_id, redaction_mode) = quality
        .as_ref()
        .map(|quality| (quality.config_id.clone(), quality.redaction_mode))
        .unwrap_or_else(|| {
            (
                media_state.quality_config.config_id(),
                media_state.quality_config.logging.redaction_mode,
            )
        });
    let pacing = take_playback_pacing_counters(media_state, playback_id);
    let terminal_span = media_state
        .playback_started_at
        .remove(playback_id)
        .map(|started_at| {
            let payload = map_from_value(json!({
                "playback_id": playback_id,
                "status": status,
            }));
            QualitySpanEmission {
                config_id: config_id.clone(),
                redaction_mode,
                span_name: "media.playback_terminal",
                category: "playback_transport",
                duration: started_at.elapsed(),
                critical_path: false,
                concurrent: true,
                payload,
            }
        });
    let terminal_reason = clear.as_ref().map(|(_, reason)| reason.label());
    let barge_in_span = clear.and_then(|(requested_at, reason)| {
        (reason == SpeechClearReason::BargeIn).then(|| {
            let payload = map_from_value(json!({
                "playback_id": playback_id,
                "terminal_status": status,
                "clear_reason": reason.label(),
            }));
            QualitySpanEmission {
                config_id: config_id.clone(),
                redaction_mode,
                span_name: "barge_in.cancel_request_to_terminal",
                category: "playback_transport",
                duration: requested_at.elapsed(),
                critical_path: false,
                concurrent: true,
                payload,
            }
        })
    });
    {
        let mut guard = state.write().await;
        guard.mark_tts_pacing_counts(
            call_id,
            playback_id,
            pacing.pre_audio_wait_ticks,
            pacing.underrun_ticks,
        );
        guard.record_quality_playback_terminal(call_id, playback_id, status, terminal_reason);
        if let Some(span) = terminal_span {
            guard.emit_quality_span_finished(call_id, span);
        }
        if let Some(span) = barge_in_span {
            guard.emit_quality_span_finished(call_id, span);
        }
    }
    media_state.playback_quality_contexts.remove(playback_id);
    media_state.append_open_empty_playbacks.remove(playback_id);
    media_state.mark_sent_playbacks.remove(playback_id);
}

fn log_outbound_frame_sent(media_state: &mut MediaSocketState, call_id: &str, playback_id: &str) {
    let now = Instant::now();
    let interval_ms = media_state
        .last_outbound_frame_sent_at
        .map(|last| now.duration_since(last).as_millis() as u64);
    let first_frame_for_playback = !media_state.first_frame_sent_playbacks.contains(playback_id);
    media_state.last_outbound_frame_sent_at = Some(now);
    media_state.outbound_frame_count = media_state.outbound_frame_count.saturating_add(1);
    media_state.outbound_pacing.observe_frame(
        (!first_frame_for_playback).then_some(interval_ms).flatten(),
        outbound_queue_depth(media_state),
    );
    media_state.outbound_underrun_ticks = 0;
    media_state.outbound_pre_audio_wait_ticks = 0;
    media_state.outbound_post_mark_wait_ticks = 0;

    let is_pacing_anomaly =
        !first_frame_for_playback && interval_ms.is_some_and(|ms| !(15..=35).contains(&ms));
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

fn take_playback_pacing_counters(
    media_state: &mut MediaSocketState,
    playback_id: &str,
) -> PlaybackPacingCounters {
    media_state
        .playback_pacing_counters
        .remove(playback_id)
        .unwrap_or_default()
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

pub struct TtsFramePacketizer {
    media: TelnyxMediaConfig,
    samples_per_packet: usize,
    pending_samples: Vec<i16>,
}

impl TtsFramePacketizer {
    pub fn new(media: TelnyxMediaConfig) -> anyhow::Result<Self> {
        Ok(Self {
            media,
            samples_per_packet: samples_per_20ms(media.sample_rate_hz)?,
            pending_samples: Vec::new(),
        })
    }

    pub fn push_samples(
        &mut self,
        mut samples: Vec<i16>,
        input_sample_rate_hz: u32,
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        if input_sample_rate_hz == 0 {
            bail!("TTS input sample rate must be non-zero");
        }
        if self.media.sample_rate_hz != input_sample_rate_hz {
            samples = resample_i16_mono(
                &WindowedSincResampler::default(),
                &samples,
                input_sample_rate_hz,
                self.media.sample_rate_hz,
            )?;
        }
        self.pending_samples.extend(samples);
        self.drain_complete_packets(false)
    }

    pub fn finish(&mut self) -> anyhow::Result<Vec<Vec<u8>>> {
        self.drain_complete_packets(true)
    }

    fn drain_complete_packets(&mut self, pad_final_packet: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        let full_packets = self.pending_samples.len() / self.samples_per_packet;
        let mut packets = Vec::new();
        for packet_index in 0..full_packets {
            let start = packet_index * self.samples_per_packet;
            let end = start + self.samples_per_packet;
            packets.push(encode_tts_packet(
                self.media,
                &self.pending_samples[start..end],
            ));
        }
        let drained = full_packets * self.samples_per_packet;
        if drained > 0 {
            self.pending_samples.drain(..drained);
        }
        if pad_final_packet && !self.pending_samples.is_empty() {
            let mut packet_samples = std::mem::take(&mut self.pending_samples);
            packet_samples.resize(self.samples_per_packet, 0);
            packets.push(encode_tts_packet(self.media, &packet_samples));
        }
        Ok(packets)
    }
}

fn encode_tts_packet(media: TelnyxMediaConfig, packet_samples: &[i16]) -> Vec<u8> {
    match media.codec {
        TelnyxStreamCodec::Pcmu => g711::encode_pcmu(packet_samples),
        TelnyxStreamCodec::L16 => l16::encode_l16_le(packet_samples),
    }
}

pub fn packetize_tts_samples(
    samples: Vec<i16>,
    input_sample_rate_hz: u32,
    media: TelnyxMediaConfig,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut packetizer = TtsFramePacketizer::new(media)?;
    let mut packets = packetizer.push_samples(samples, input_sample_rate_hz)?;
    packets.extend(packetizer.finish()?);
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
            if let (Some(call_id), Some(name)) =
                (media_state.gateway_call_id.clone(), event.mark.name.clone())
            {
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

    let _ = flush_ready_pending_final_transcript(
        state,
        media_state,
        &gateway_call_id,
        text_calls,
        false,
    )
    .await;

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
    maybe_emit_echo_characterization_span(
        state,
        media_state,
        &gateway_call_id,
        stream_id,
        &format,
        &samples,
        &stats,
    )
    .await;
    let audio_onset_evidence = update_audio_barge_in_evidence(
        state,
        media_state,
        AudioBargeInFrame {
            gateway_call_id: &gateway_call_id,
            stream_id,
            format: &format,
            samples: &samples,
            stats: &stats,
            frame_duration_ms,
            caller_candidate_active: media_state.asr_gate.speech_started,
        },
    )
    .await;
    let audio_trusted_onset = audio_onset_evidence.as_ref().is_some_and(|evidence| {
        take_audio_trusted_onset_trigger(
            &media_state.quality_config,
            &mut media_state.audio_barge_in,
            evidence,
        )
    });
    let partial_speech_state = match media_state.asr_gate.accept(
        media_state.decoded_frame_count,
        stream_id,
        frame_duration_ms,
        &stats,
        &media_state.quality_config,
    ) {
        AsrFrameDecision::Suppress => return Ok(()),
        AsrFrameDecision::Continue {
            speech_onset,
            speech_state,
        } => {
            if speech_onset {
                if let Some(pending) = media_state.pending_final.as_mut() {
                    pending.continuation_speech_seen = true;
                    tracing::debug!(
                        gateway_call_id,
                        stream_id,
                        hold_reason = pending.hold_reason,
                        "asr.final_settle.continuation_speech_seen"
                    );
                }
            }
            if (speech_onset || audio_trusted_onset)
                && speech_onset_barge_in_enabled(&media_state.quality_config)
            {
                let echo_decision = speech_onset_echo_decision(
                    state,
                    media_state.media_registry.clone(),
                    media_state.first_frame_sent_playbacks.clone(),
                    media_state.quality_config.clone(),
                    audio_onset_evidence.as_ref(),
                    &gateway_call_id,
                )
                .await;
                if echo_decision.defer_to_partial {
                    emit_speech_onset_deferred_span(
                        state,
                        media_state.active_quality_asr.as_ref(),
                        &gateway_call_id,
                        stream_id,
                        &echo_decision,
                    )
                    .await;
                    tracing::debug!(
                        gateway_call_id,
                        stream_id,
                        playback_id = echo_decision.playback_id.as_deref(),
                        reason = echo_decision.reason,
                        "barge_in.speech_onset.deferred_for_echo_guard"
                    );
                } else {
                    cancel_early_response_for_barge_in(media_state);
                    if let Some(text_calls) = text_calls {
                        cancel_text_call_speech_for_barge_in(
                            state,
                            &media_state.media_registry,
                            text_calls,
                            &gateway_call_id,
                        )
                        .await?;
                    }
                    if let Some(runtime) = media_state.conversation.as_ref() {
                        conversation::handle_speech_onset(
                            state,
                            &media_state.media_registry,
                            runtime,
                            &gateway_call_id,
                            Some(&media_state.quality_config),
                            audio_onset_evidence.as_ref(),
                        )
                        .await?;
                    }
                }
            }
            speech_state
        }
        AsrFrameDecision::Finalize {
            trailing_silence_ms,
            endpoint_wait_started_at,
            speech_to_low_energy,
            endpoint_gate,
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
                AsrEndpointSpanTiming {
                    trailing_silence_ms,
                    endpoint_wait_started_at,
                    speech_to_low_energy,
                    endpoint_gate: Some(endpoint_gate),
                },
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
    };
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
    let quality_session = media_state.active_quality_asr.clone();
    let record_outcome = record_and_forward_asr_events(
        state,
        media_state,
        &gateway_call_id,
        Some(stream_id),
        text_calls,
        quality_session.as_ref(),
        ForwardAsrEvents::new(events, partial_speech_state),
    )
    .await;
    if record_outcome.reset_requested {
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
    ensure_early_response_pipeline(state, media_state, gateway_call_id).await;
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

async fn ensure_early_response_pipeline(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    gateway_call_id: &str,
) {
    if media_state.early_response.is_some() {
        return;
    }
    let Some(runtime) = media_state.conversation.as_ref() else {
        return;
    };
    let (speech_output, processor) = {
        let guard = state.read().await;
        let call = guard.calls.get(gateway_call_id);
        let processor = call
            .map(|call| call.conversation.processor.clone())
            .unwrap_or_default();
        let speech_output = call.map(|call| call.speech_output).unwrap_or_else(|| {
            crate::operator::state::SpeechOutputConfig::from_quality(
                guard.conversation_tts_backend,
                &guard.quality.config.tts,
            )
        });
        (speech_output, processor)
    };
    if !media_state.quality_config.early_response.enabled
        && processor != ConversationProcessorKind::ExternalTextStream
    {
        return;
    }
    let text_calls = media_state.text_calls.clone().unwrap_or_default();
    let handle = spawn_early_response_pipeline(
        gateway_call_id.to_string(),
        media_state.quality_config.early_response.clone(),
        EarlyResponsePipelineServices {
            state: state.clone(),
            media_registry: media_state.media_registry.clone(),
            tts: runtime.tts_registry(),
            text_calls,
            speech_output,
            processor,
        },
    );
    media_state.early_response = Some(handle);
    tracing::info!(gateway_call_id, "early_response.pipeline.started");
}

fn cancel_early_response_for_barge_in(media_state: &MediaSocketState) {
    if let Some(handle) = media_state.early_response.as_ref() {
        handle.cancel_call(EarlyResponseCancelReason::CallerBargeIn);
    }
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct SpeechOnsetEchoDecision {
    defer_to_partial: bool,
    playback_id: Option<String>,
    reason: &'static str,
}

fn speech_onset_barge_in_enabled(config: &VoiceQualityConfig) -> bool {
    config
        .conversation_policy
        .decide_barge_in(&config.barge_in, BargeInTrigger::SpeechOnset)
        .cancels_playback()
}

fn audio_barge_in_evidence_enabled(config: &VoiceQualityConfig) -> bool {
    config.echo_characterization.enabled
        || config.audio_barge_in.media.mode.consumes_audio_evidence()
}

fn take_audio_trusted_onset_trigger(
    config: &VoiceQualityConfig,
    audio_state: &mut AudioBargeInEvidenceState,
    evidence: &CallerOnsetEvidence,
) -> bool {
    config.audio_barge_in.media.mode.consumes_audio_evidence()
        && evidence.playback_state == PlaybackEchoState::ActivePlayback
        && audio_state.take_trusted_onset_edge(evidence)
}

fn playback_state_without_active(
    recent_playback_age: Option<Duration>,
    outbound_queue_depth: usize,
) -> PlaybackEchoState {
    if recent_playback_age.is_none() {
        PlaybackEchoState::Idle
    } else if outbound_queue_depth > 0 {
        PlaybackEchoState::InterSegmentGap
    } else {
        PlaybackEchoState::RecentTail
    }
}

async fn update_audio_barge_in_evidence(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    frame: AudioBargeInFrame<'_>,
) -> Option<CallerOnsetEvidence> {
    let gateway_call_id = frame.gateway_call_id;
    let stream_id = frame.stream_id;
    let format = frame.format;
    let samples = frame.samples;
    let stats = frame.stats;
    let frame_duration_ms = frame.frame_duration_ms;
    if !audio_barge_in_evidence_enabled(&media_state.quality_config) {
        return None;
    }
    let now = Instant::now();
    let media_config = media_state.quality_config.audio_barge_in.media.clone();
    let transport_invalidation = media_state
        .audio_barge_in
        .transport_invalidation(&media_state.inbound_transport, &media_config);
    let speech_active = stats.has_speech_energy(&media_state.quality_config.speech);
    let active_ref = media_state
        .media_registry
        .active_speech_playback_ref(gateway_call_id)
        .await;

    let Some(active_ref) = active_ref else {
        media_state.audio_barge_in.reset_caller_active();
        if !speech_active {
            return None;
        }
        let recent_window =
            Duration::from_millis(media_state.quality_config.echo_suppression.tail_window_ms);
        let recent_playback_age = media_state
            .media_registry
            .recent_speech_playback_age(gateway_call_id, recent_window)
            .await;
        let playback_state =
            playback_state_without_active(recent_playback_age, outbound_queue_depth(media_state));
        let decision = if playback_state == PlaybackEchoState::Idle {
            AudioOnsetDecision::TrustedCallerOnset
        } else {
            AudioOnsetDecision::Ambiguous
        };
        let evidence = base_caller_onset_evidence(CallerOnsetEvidenceSeed {
            decision,
            confidence: if decision == AudioOnsetDecision::TrustedCallerOnset {
                1.0
            } else {
                0.0
            },
            playback_state,
            playback_ref: None,
            caller_active_since: (decision == AudioOnsetDecision::TrustedCallerOnset)
                .then_some(now),
            now,
            window_ms: frame_duration_ms.min(u64::from(u32::MAX)) as u32,
            format,
            inbound_rms_dbfs: rms_dbfs(samples),
            outbound_rms_dbfs: None,
            invalidation: None,
        });
        media_state
            .media_registry
            .record_caller_onset_evidence(gateway_call_id, evidence.clone())
            .await;
        emit_audio_barge_in_evidence_span(
            state,
            media_state.active_quality_asr.as_ref(),
            gateway_call_id,
            stream_id,
            &evidence,
            None,
        )
        .await;
        return Some(evidence);
    };

    if !media_state
        .first_frame_sent_playbacks
        .contains(active_ref.playback_id.as_str())
    {
        media_state.audio_barge_in.reset_caller_active();
        if !speech_active {
            return None;
        }
        let evidence = base_caller_onset_evidence(CallerOnsetEvidenceSeed {
            decision: AudioOnsetDecision::Unavailable,
            confidence: 0.0,
            playback_state: PlaybackEchoState::ActivePlayback,
            playback_ref: Some(&active_ref),
            caller_active_since: None,
            now,
            window_ms: frame_duration_ms.min(u64::from(u32::MAX)) as u32,
            format,
            inbound_rms_dbfs: rms_dbfs(samples),
            outbound_rms_dbfs: None,
            invalidation: Some(AudioEvidenceInvalidation::PreAudioPlayback),
        });
        media_state
            .media_registry
            .record_caller_onset_evidence(gateway_call_id, evidence.clone())
            .await;
        emit_audio_barge_in_evidence_span(
            state,
            media_state.active_quality_asr.as_ref(),
            gateway_call_id,
            stream_id,
            &evidence,
            None,
        )
        .await;
        return Some(evidence);
    }

    let reference_len = samples.len().saturating_add(samples_for_ms(
        media_config.delay_search_max_ms,
        format.sample_rate_hz,
    ));
    let outbound_reference = media_state
        .echo_characterization
        .reference_samples_for(format.sample_rate_hz, reference_len);

    if transport_invalidation.is_none() {
        if let Some(metrics) = playback_only_calibration_metrics(
            &media_config,
            format,
            samples,
            &media_state.quality_config.speech,
            &outbound_reference,
            frame.caller_candidate_active,
        ) {
            media_state
                .audio_barge_in
                .observe_playback_only_calibration(&media_config, frame_duration_ms, metrics);
        }
    }

    if !speech_active {
        media_state.audio_barge_in.reset_caller_active();
        return None;
    }

    let evidence =
        media_state
            .audio_barge_in
            .classify_active_playback_window(ActivePlaybackWindow {
                config: &media_config,
                playback_ref: &active_ref,
                format,
                samples,
                stats,
                speech: &media_state.quality_config.speech,
                outbound_reference: &outbound_reference,
                transport_invalidation,
                frame_duration_ms,
                now,
            });
    media_state
        .media_registry
        .record_caller_onset_evidence(gateway_call_id, evidence.clone())
        .await;
    emit_audio_barge_in_evidence_span(
        state,
        media_state.active_quality_asr.as_ref(),
        gateway_call_id,
        stream_id,
        &evidence,
        media_state.audio_barge_in.calibration,
    )
    .await;
    Some(evidence)
}

async fn emit_audio_barge_in_evidence_span(
    state: &SharedState,
    quality_session: Option<&ActiveAsrQualitySession>,
    gateway_call_id: &str,
    stream_id: &str,
    evidence: &CallerOnsetEvidence,
    calibration: Option<AudioEchoCalibration>,
) {
    let Some(session) = quality_session else {
        return;
    };
    let payload = map_from_value(json!({
        "asr_session_id": session.asr_session_id.as_str(),
        "utterance_id": session.utterance_id.as_str(),
        "stream_id": stream_id,
        "decision": evidence.decision.label(),
        "confidence": evidence.confidence,
        "playback_state": evidence.playback_state.label(),
        "playback_id": evidence.playback_id.as_deref(),
        "playback_epoch": evidence.playback_epoch,
        "window_ms": evidence.window_ms,
        "input_codec": evidence.input_codec.as_str(),
        "input_sample_rate_hz": evidence.input_sample_rate_hz,
        "evidence_sample_rate_hz": evidence.evidence_sample_rate_hz,
        "inbound_rms_dbfs": evidence.inbound_rms_dbfs,
        "outbound_rms_dbfs": evidence.outbound_rms_dbfs,
        "estimated_delay_ms": evidence.estimated_delay_ms,
        "correlation_peak": evidence.correlation_peak,
        "echo_return_db": evidence.echo_return_db,
        "echo_margin_db": evidence.echo_margin_db,
        "invalidation": evidence.invalidation.map(AudioEvidenceInvalidation::label),
        "calibrated_erl_baseline_db": calibration.map(|value| value.erl_baseline_db),
        "calibrated_delay_ms": calibration.map(|value| value.delay_ms),
        "calibrated_playback_only_ms": calibration.map(|value| value.playback_only_ms),
        "calibration_update_count": calibration.map(|value| value.update_count),
    }));
    state.write().await.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id: session.config_id.clone(),
            redaction_mode: session.redaction_mode,
            span_name: "media.audio_barge_in.evidence",
            category: "barge_in",
            duration: Duration::ZERO,
            critical_path: false,
            concurrent: true,
            payload,
        },
    );
}

async fn speech_onset_echo_decision(
    state: &SharedState,
    media_registry: SharedMediaRegistry,
    first_frame_sent_playbacks: HashSet<String>,
    quality_config: VoiceQualityConfig,
    audio_evidence: Option<&CallerOnsetEvidence>,
    gateway_call_id: &str,
) -> SpeechOnsetEchoDecision {
    if quality_config
        .audio_barge_in
        .media
        .mode
        .consumes_audio_evidence()
        && quality_config.conversation_policy.mode != ConversationPolicyMode::CurrentCompat
    {
        let active_ref = media_registry
            .active_speech_playback_ref(gateway_call_id)
            .await;
        let Some(active_ref) = active_ref else {
            return SpeechOnsetEchoDecision {
                defer_to_partial: false,
                playback_id: None,
                reason: "no_active_playback",
            };
        };
        let Some(audio_evidence) = audio_evidence else {
            return SpeechOnsetEchoDecision {
                defer_to_partial: true,
                playback_id: Some(active_ref.playback_id),
                reason: "audio_evidence_unavailable",
            };
        };
        let policy_decision =
            quality_config
                .conversation_policy
                .decide_audio_barge_in(AudioBargeInDecisionInput {
                    barge_in: &quality_config.barge_in,
                    audio_barge_in: &quality_config.audio_barge_in,
                    trigger: BargeInTrigger::SpeechOnset,
                    evidence: audio_evidence,
                    active_playback: ActivePlaybackTarget {
                        playback_id: Some(active_ref.playback_id.as_str()),
                        playback_epoch: Some(active_ref.playback_epoch),
                    },
                    now: Instant::now(),
                });
        return SpeechOnsetEchoDecision {
            defer_to_partial: !policy_decision.cancels_playback(),
            playback_id: Some(active_ref.playback_id),
            reason: if policy_decision.cancels_playback() {
                "audio_trusted_caller_onset"
            } else {
                audio_evidence.decision.label()
            },
        };
    }

    let onset_policy = quality_config.barge_in.onset_during_playback;
    let echo_config = quality_config.echo_suppression.clone();
    if onset_policy == OnsetDuringPlaybackPolicy::Trust {
        return SpeechOnsetEchoDecision {
            defer_to_partial: false,
            playback_id: None,
            reason: "policy_trust",
        };
    }
    if !echo_config.enabled {
        return SpeechOnsetEchoDecision {
            defer_to_partial: false,
            playback_id: None,
            reason: "echo_suppression_disabled",
        };
    }
    let Some(playback_id) = media_registry
        .active_speech_playback_id(gateway_call_id)
        .await
    else {
        return SpeechOnsetEchoDecision {
            defer_to_partial: false,
            playback_id: None,
            reason: "no_active_playback",
        };
    };
    if !first_frame_sent_playbacks.contains(playback_id.as_str()) {
        return SpeechOnsetEchoDecision {
            defer_to_partial: false,
            playback_id: Some(playback_id),
            reason: "pre_audio_playback",
        };
    }
    let likely_echo = {
        let guard = state.read().await;
        guard
            .calls
            .get(gateway_call_id)
            .and_then(|call| call.tts.as_ref())
            .filter(|tts| tts.playback_id == playback_id)
            .is_some_and(|tts| {
                !tts.echo_signature.is_empty()
                    && tts_in_echo_window(tts, echo_config.tail_window_ms as i64)
            })
    };
    SpeechOnsetEchoDecision {
        defer_to_partial: likely_echo,
        playback_id: Some(playback_id),
        reason: if likely_echo {
            "likely_assistant_echo"
        } else {
            "no_echo_signature"
        },
    }
}

async fn emit_speech_onset_deferred_span(
    state: &SharedState,
    quality_session: Option<&ActiveAsrQualitySession>,
    gateway_call_id: &str,
    stream_id: &str,
    decision: &SpeechOnsetEchoDecision,
) {
    let Some(session) = quality_session else {
        return;
    };
    let payload = map_from_value(json!({
        "asr_session_id": session.asr_session_id.as_str(),
        "utterance_id": session.utterance_id.as_str(),
        "stream_id": stream_id,
        "playback_id": decision.playback_id.as_deref(),
        "reason": decision.reason,
        "onset_during_playback": "defer_to_partial",
    }));
    state.write().await.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id: session.config_id.clone(),
            redaction_mode: session.redaction_mode,
            span_name: "barge_in.speech_onset_deferred_echo_guard",
            category: "barge_in",
            duration: Duration::ZERO,
            critical_path: false,
            concurrent: true,
            payload,
        },
    );
}

async fn cancel_text_call_speech_for_barge_in(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    text_calls: &SharedTextCallRegistry,
    gateway_call_id: &str,
) -> anyhow::Result<()> {
    if !text_calls.contains(gateway_call_id).await {
        return Ok(());
    }
    let _ = text_calls
        .send_turn_batch_reset(gateway_call_id, "barge_in")
        .await;
    if media_registry
        .active_speech_playback_id(gateway_call_id)
        .await
        .is_none()
    {
        return Ok(());
    }
    let playback_id = match speech::cancel_speech_with_reason(
        state,
        media_registry,
        gateway_call_id,
        "text-call speech-onset barge-in",
        SpeechClearReason::BargeIn,
    )
    .await
    {
        Ok(playback_id) => playback_id,
        Err(error) if format!("{error:#}").contains("no active speech job") => return Ok(()),
        Err(error) => return Err(error),
    };
    text_calls
        .finish_playback(
            gateway_call_id,
            &playback_id,
            PlaybackFinishedStatus::Canceled,
        )
        .await;
    tracing::info!(
        gateway_call_id,
        playback_id,
        "text_call.barge_in.cancel_requested"
    );
    Ok(())
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
    let playback_id = media_state
        .playback_started_at
        .keys()
        .next()
        .map(String::as_str);
    let outbound_payload = media_state.outbound_pacing.rollup_payload(playback_id);
    let pacing_counters = std::mem::take(&mut media_state.playback_pacing_counters);
    let mut guard = state.write().await;
    for (playback_id, counters) in pacing_counters {
        guard.mark_tts_pacing_counts(
            &call_id,
            &playback_id,
            counters.pre_audio_wait_ticks,
            counters.underrun_ticks,
        );
    }
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
    guard.emit_quality_report_summary(&call_id, "media_stop");
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

pub(crate) fn percentile_u64(values: &[u64], percentile: u64) -> u64 {
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

fn echo_characterization_retention(config: &EchoCharacterizationQualityConfig) -> Duration {
    Duration::from_millis(
        config
            .window_ms
            .saturating_add(config.max_delay_ms)
            .saturating_add(config.emit_interval_ms),
    )
}

fn audio_barge_in_reference_retention(config: &AudioBargeInMediaQualityConfig) -> Duration {
    Duration::from_millis(
        config
            .calibration_min_playback_only_ms
            .saturating_add(config.delay_search_max_ms)
            .saturating_add(config.max_evidence_age_ms)
            .saturating_add(500),
    )
}

fn outbound_reference_retention(config: &VoiceQualityConfig) -> Duration {
    let echo_retention = if config.echo_characterization.enabled {
        echo_characterization_retention(&config.echo_characterization)
    } else {
        Duration::ZERO
    };
    let audio_retention = if config.audio_barge_in.media.mode.consumes_audio_evidence() {
        audio_barge_in_reference_retention(&config.audio_barge_in.media)
    } else {
        Duration::ZERO
    };
    echo_retention.max(audio_retention)
}

fn samples_for_ms(duration_ms: u64, sample_rate_hz: u32) -> usize {
    let samples = duration_ms
        .saturating_mul(u64::from(sample_rate_hz))
        .saturating_add(999)
        / 1_000;
    samples.min(usize::MAX as u64) as usize
}

fn rms_dbfs(samples: &[i16]) -> f32 {
    let rms = sample_stats(samples).rms / 32_768.0;
    if rms <= 0.000001 {
        return -120.0;
    }
    (20.0 * rms.log10()).max(-120.0)
}

fn normalized_correlation(a: &[i16], b: &[i16]) -> Option<f32> {
    if a.len() != b.len() || a.len() < 4 {
        return None;
    }
    let mean_a = a.iter().map(|sample| f64::from(*sample)).sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().map(|sample| f64::from(*sample)).sum::<f64>() / b.len() as f64;
    let mut dot = 0.0f64;
    let mut energy_a = 0.0f64;
    let mut energy_b = 0.0f64;
    for (&sample_a, &sample_b) in a.iter().zip(b) {
        let centered_a = f64::from(sample_a) - mean_a;
        let centered_b = f64::from(sample_b) - mean_b;
        dot += centered_a * centered_b;
        energy_a += centered_a * centered_a;
        energy_b += centered_b * centered_b;
    }
    if energy_a <= f64::EPSILON || energy_b <= f64::EPSILON {
        return None;
    }
    Some((dot / (energy_a.sqrt() * energy_b.sqrt())).abs() as f32)
}

fn samples_for_delay_ms(delay_ms: f32, sample_rate_hz: u32) -> Option<usize> {
    if !delay_ms.is_finite() || delay_ms < 0.0 || sample_rate_hz == 0 {
        return None;
    }
    Some(((delay_ms * sample_rate_hz as f32) / 1_000.0).round() as usize)
}

fn reference_segment_at_delay(
    inbound_len: usize,
    outbound_reference: &[i16],
    sample_rate_hz: u32,
    delay_ms: f32,
) -> Option<&[i16]> {
    if inbound_len == 0 {
        return None;
    }
    let delay_samples = samples_for_delay_ms(delay_ms, sample_rate_hz)?;
    let end = outbound_reference.len().checked_sub(delay_samples)?;
    let start = end.checked_sub(inbound_len)?;
    Some(&outbound_reference[start..end])
}

fn residual_echo_metrics(inbound: &[i16], outbound_segment: &[i16]) -> Option<ResidualEchoMetrics> {
    if inbound.len() != outbound_segment.len() || inbound.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut outbound_energy = 0.0f64;
    for (&in_sample, &out_sample) in inbound.iter().zip(outbound_segment) {
        let inbound_value = f64::from(in_sample);
        let outbound_value = f64::from(out_sample);
        dot += inbound_value * outbound_value;
        outbound_energy += outbound_value * outbound_value;
    }
    if outbound_energy <= f64::EPSILON {
        return None;
    }
    let scale = dot / outbound_energy;
    let mut peak = 0i16;
    let mut sum_squares = 0.0f64;
    for (&in_sample, &out_sample) in inbound.iter().zip(outbound_segment) {
        let residual = f64::from(in_sample) - scale * f64::from(out_sample);
        let bounded = residual.clamp(f64::from(i16::MIN), f64::from(i16::MAX));
        let abs = bounded.abs().min(f64::from(i16::MAX)) as i16;
        if abs > peak {
            peak = abs;
        }
        sum_squares += residual * residual;
    }
    let rms = (sum_squares / inbound.len() as f64).sqrt() as f32;
    let residual_rms_dbfs = if rms <= 0.000001 {
        -120.0
    } else {
        (20.0 * (rms / 32_768.0).log10()).max(-120.0)
    };
    Some(ResidualEchoMetrics {
        residual_rms_dbfs,
        residual_peak: peak,
        residual_rms: rms,
    })
}

fn playback_only_calibration_metrics(
    config: &AudioBargeInMediaQualityConfig,
    format: &MediaFormat,
    samples: &[i16],
    speech: &SpeechQualityConfig,
    outbound_reference: &[i16],
    caller_candidate_active: bool,
) -> Option<EchoCharacterizationMetrics> {
    if caller_candidate_active {
        return None;
    }
    let metrics = characterize_echo(
        samples,
        outbound_reference,
        format.sample_rate_hz,
        config.delay_search_max_ms,
    )?;
    if metrics.estimated_delay_ms < config.delay_search_min_ms
        || metrics.estimated_delay_ms > config.delay_search_max_ms
        || metrics.correlation_peak < config.calibration_min_correlation
    {
        return None;
    }
    let segment = reference_segment_at_delay(
        samples.len(),
        outbound_reference,
        format.sample_rate_hz,
        metrics.estimated_delay_ms as f32,
    )?;
    if residual_echo_metrics(samples, segment)?.has_speech_energy(speech) {
        return None;
    }
    Some(metrics)
}

fn characterize_echo(
    inbound: &[i16],
    outbound_reference: &[i16],
    sample_rate_hz: u32,
    max_delay_ms: u64,
) -> Option<EchoCharacterizationMetrics> {
    if inbound.is_empty() || outbound_reference.is_empty() || sample_rate_hz == 0 {
        return None;
    }
    let max_delay_samples = samples_for_ms(max_delay_ms, sample_rate_hz)
        .min(outbound_reference.len().saturating_sub(inbound.len()));
    let mut best: Option<(usize, f32, &[i16])> = None;
    for delay_samples in 0..=max_delay_samples {
        let end = outbound_reference.len().saturating_sub(delay_samples);
        if end < inbound.len() {
            continue;
        }
        let start = end - inbound.len();
        let candidate = &outbound_reference[start..end];
        let Some(correlation) = normalized_correlation(inbound, candidate) else {
            continue;
        };
        if best.is_none_or(|(_, best_correlation, _)| correlation > best_correlation) {
            best = Some((delay_samples, correlation, candidate));
        }
    }
    let (delay_samples, correlation_peak, outbound_segment) = best?;
    let estimated_delay_ms =
        (delay_samples as u64).saturating_mul(1_000) / u64::from(sample_rate_hz);
    let inbound_rms_dbfs = rms_dbfs(inbound);
    let outbound_rms_dbfs = rms_dbfs(outbound_segment);
    Some(EchoCharacterizationMetrics {
        correlation_peak,
        estimated_delay_ms,
        inbound_rms_dbfs,
        outbound_rms_dbfs,
        echo_return_db: inbound_rms_dbfs - outbound_rms_dbfs,
    })
}

#[derive(Default)]
struct AsrGate {
    speech_started: bool,
    speech_started_at: Option<Instant>,
    low_energy_started_at: Option<Instant>,
    low_energy_run_ms: u64,
    suppressed_initial_frames: usize,
    suppressed_tail_frames: usize,
    last_speech_peak: Option<i16>,
    last_speech_rms: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AsrEndpointGateSnapshot {
    suppressed_tail_frames: usize,
    low_energy_run_ms: u64,
    endpoint_frame_peak: i16,
    endpoint_frame_rms: f32,
    last_speech_peak: Option<i16>,
    last_speech_rms: Option<f32>,
    rms_threshold: f32,
    peak_threshold: i32,
}

#[derive(Clone, Copy, Debug)]
struct AsrEndpointSpanTiming {
    trailing_silence_ms: u64,
    endpoint_wait_started_at: Option<Instant>,
    speech_to_low_energy: Option<Duration>,
    endpoint_gate: Option<AsrEndpointGateSnapshot>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum AsrFrameDecision {
    Suppress,
    Continue {
        speech_onset: bool,
        speech_state: CallerSpeechState,
    },
    Finalize {
        trailing_silence_ms: u64,
        endpoint_wait_started_at: Option<Instant>,
        speech_to_low_energy: Option<Duration>,
        endpoint_gate: AsrEndpointGateSnapshot,
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
            self.last_speech_peak = Some(stats.peak);
            self.last_speech_rms = Some(stats.rms);
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
            return AsrFrameDecision::Continue {
                speech_onset,
                speech_state: CallerSpeechState::Speaking,
            };
        }

        if self.speech_started {
            if self.low_energy_started_at.is_none() {
                self.low_energy_started_at = Some(now);
            }
            self.low_energy_run_ms = self.low_energy_run_ms.saturating_add(frame_duration_ms);
            if self.low_energy_run_ms <= quality.endpoint.trailing_silence_ms {
                return AsrFrameDecision::Continue {
                    speech_onset: false,
                    speech_state: CallerSpeechState::EndpointCandidate,
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
                endpoint_gate: self.endpoint_gate_snapshot(stats, quality),
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
        self.suppressed_tail_frames = 0;
        self.last_speech_peak = None;
        self.last_speech_rms = None;
    }

    fn endpoint_gate_snapshot(
        &self,
        endpoint_frame: &SampleStats,
        quality: &VoiceQualityConfig,
    ) -> AsrEndpointGateSnapshot {
        AsrEndpointGateSnapshot {
            suppressed_tail_frames: self.suppressed_tail_frames,
            low_energy_run_ms: self.low_energy_run_ms,
            endpoint_frame_peak: endpoint_frame.peak,
            endpoint_frame_rms: endpoint_frame.rms,
            last_speech_peak: self.last_speech_peak,
            last_speech_rms: self.last_speech_rms,
            rms_threshold: quality.speech.rms_threshold,
            peak_threshold: quality.speech.peak_threshold,
        }
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
        flush_ready_pending_final_transcript(state, media_state, &call_id, text_calls, true).await;
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
    timing: AsrEndpointSpanTiming,
) {
    let Some(session) = session else {
        return;
    };
    if let Some(duration) = timing.speech_to_low_energy {
        let payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id.as_str(),
            "utterance_id": session.utterance_id.as_str(),
            "stream_id": stream_id,
        }));
        state.write().await.emit_quality_span_finished(
            gateway_call_id,
            QualitySpanEmission {
                config_id: session.config_id.clone(),
                redaction_mode: session.redaction_mode,
                span_name: "utterance.speech_to_low_energy",
                category: "caller_speech",
                duration,
                critical_path: true,
                concurrent: false,
                payload,
            },
        );
    }
    if let Some(started_at) = timing.endpoint_wait_started_at {
        let mut payload = map_from_value(json!({
            "asr_session_id": session.asr_session_id.as_str(),
            "utterance_id": session.utterance_id.as_str(),
            "stream_id": stream_id,
            "trailing_silence_ms": timing.trailing_silence_ms,
        }));
        if let Some(gate) = timing.endpoint_gate {
            payload.extend(map_from_value(json!({
                "suppressed_tail_frames": gate.suppressed_tail_frames,
                "low_energy_run_ms": gate.low_energy_run_ms,
                "endpoint_frame_peak": gate.endpoint_frame_peak,
                "endpoint_frame_rms": gate.endpoint_frame_rms,
                "last_speech_peak": gate.last_speech_peak,
                "last_speech_rms": gate.last_speech_rms,
                "rms_threshold": gate.rms_threshold,
                "peak_threshold": gate.peak_threshold,
            })));
        }
        state.write().await.emit_quality_span_finished(
            gateway_call_id,
            QualitySpanEmission {
                config_id: session.config_id.clone(),
                redaction_mode: session.redaction_mode,
                span_name: "asr.endpoint_wait",
                category: "endpointing",
                duration: started_at.elapsed(),
                critical_path: true,
                concurrent: false,
                payload,
            },
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
    if let (Some(call_id), Some(mut asr_session)) = (gateway_call_id, media_state.session.take()) {
        let quality_session = media_state.active_quality_asr.take();
        let finish_pad_ms = media_state.quality_config.asr.finish_pad_ms;
        let pad_started_at = Instant::now();
        let mut transcript_events =
            ingest_asr_finish_silence(asr_session.as_mut(), finish_pad_ms).await?;
        let pad_duration = pad_started_at.elapsed();
        let pad_event_count = transcript_events.len();
        if let Some(session) = quality_session.as_ref() {
            let payload = map_from_value(json!({
                "asr_session_id": session.asr_session_id.as_str(),
                "utterance_id": session.utterance_id.as_str(),
                "stream_id": stream_id.as_deref(),
                "finish_pad_ms": finish_pad_ms,
                "pad_transcript_events": pad_event_count,
            }));
            state.write().await.emit_quality_span_finished(
                call_id,
                QualitySpanEmission {
                    config_id: session.config_id.clone(),
                    redaction_mode: session.redaction_mode,
                    span_name: "asr.finish_pad",
                    category: "asr_generation",
                    duration: pad_duration,
                    critical_path: true,
                    concurrent: false,
                    payload,
                },
            );
        }
        let finish_started_at = Instant::now();
        let events = asr_session.finish().await?;
        let finish_event_count = events.len();
        transcript_events.extend(events);
        if let Some(session) = quality_session.as_ref() {
            let payload = map_from_value(json!({
                "asr_session_id": session.asr_session_id.as_str(),
                "utterance_id": session.utterance_id.as_str(),
                "stream_id": stream_id.as_deref(),
                "finish_pad_ms": finish_pad_ms,
                "pad_transcript_events": pad_event_count,
                "finish_transcript_events": finish_event_count,
                "transcript_events": pad_event_count.saturating_add(finish_event_count),
            }));
            state.write().await.emit_quality_span_finished(
                call_id,
                QualitySpanEmission {
                    config_id: session.config_id.clone(),
                    redaction_mode: session.redaction_mode,
                    span_name: "asr.local_finish",
                    category: "asr_generation",
                    duration: finish_started_at.elapsed(),
                    critical_path: true,
                    concurrent: false,
                    payload,
                },
            );
            media_state.last_quality_asr = Some(session.clone());
        }
        record_and_forward_asr_events(
            state,
            media_state,
            call_id,
            stream_id.as_deref(),
            text_calls,
            quality_session.as_ref(),
            ForwardAsrEvents::new(transcript_events, CallerSpeechState::Finalizing),
        )
        .await;
    }
    Ok(())
}

struct ForwardAsrEvents {
    events: Vec<AsrTranscriptEvent>,
    partial_speech_state: CallerSpeechState,
}

impl ForwardAsrEvents {
    fn new(events: Vec<AsrTranscriptEvent>, partial_speech_state: CallerSpeechState) -> Self {
        Self {
            events,
            partial_speech_state,
        }
    }
}

async fn ingest_asr_finish_silence(
    asr_session: &mut dyn InboundAsrSession,
    pad_ms: u64,
) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
    const ASR_SAMPLE_RATE_HZ: u64 = 16_000;

    let samples = ((ASR_SAMPLE_RATE_HZ * pad_ms) / 1_000) as usize;
    if samples == 0 {
        return Ok(Vec::new());
    }
    asr_session
        .ingest(AudioBuf::<i16, 16_000, Mono>::new(vec![0; samples]))
        .await
}

async fn record_and_forward_asr_events(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    call_id: &str,
    stream_id: Option<&str>,
    text_calls: Option<&SharedTextCallRegistry>,
    quality_session: Option<&ActiveAsrQualitySession>,
    input: ForwardAsrEvents,
) -> TranscriptRecordOutcome {
    let mut events =
        reconcile_asr_final_events(state, call_id, stream_id, quality_session, input.events).await;
    let mut outcome = TranscriptRecordOutcome::default();
    if !has_emitted_final(&events) {
        outcome.merge(
            flush_ready_pending_final_transcript(state, media_state, call_id, text_calls, false)
                .await,
        );
    }
    events = apply_pending_final_settle(media_state, call_id, stream_id, quality_session, events);
    outcome.merge(
        record_and_forward_reconciled_events(
            state,
            media_state,
            call_id,
            stream_id,
            text_calls,
            quality_session,
            ForwardAsrEvents::new(events, input.partial_speech_state),
        )
        .await,
    );
    outcome
}

async fn record_and_forward_reconciled_events(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    call_id: &str,
    stream_id: Option<&str>,
    text_calls: Option<&SharedTextCallRegistry>,
    quality_session: Option<&ActiveAsrQualitySession>,
    input: ForwardAsrEvents,
) -> TranscriptRecordOutcome {
    if input.events.is_empty() {
        return TranscriptRecordOutcome::default();
    }
    let record_outcome = record_transcript_events(
        state,
        call_id,
        input.events,
        TranscriptRecordContext {
            stream_id,
            media_format: media_state.media_format.as_ref(),
            capture: media_state.capture.as_mut(),
            text_calls,
            early_response: media_state.early_response.clone(),
            quality_session,
            echo_config: Some(&media_state.quality_config.echo_suppression),
            partial_speech_state: input.partial_speech_state,
        },
    )
    .await;
    forward_conversation_events(
        state,
        &media_state.media_registry,
        media_state.conversation.as_ref(),
        call_id,
        record_outcome.conversation_events.clone(),
        Some(&media_state.quality_config),
    )
    .await;
    record_outcome
}

async fn flush_ready_pending_final_transcript(
    state: &SharedState,
    media_state: &mut MediaSocketState,
    call_id: &str,
    text_calls: Option<&SharedTextCallRegistry>,
    force: bool,
) -> TranscriptRecordOutcome {
    let now = Instant::now();
    let should_flush = media_state
        .pending_final
        .as_ref()
        .is_some_and(|pending| force || pending.is_ready(now));
    if !should_flush {
        return TranscriptRecordOutcome::default();
    }
    let Some(pending) = media_state.pending_final.take() else {
        return TranscriptRecordOutcome::default();
    };
    tracing::debug!(
        gateway_call_id = call_id,
        stream_id = pending.stream_id.as_deref(),
        hold_reason = pending.hold_reason,
        force,
        continuation_speech_seen = pending.continuation_speech_seen,
        held_ms = pending.created_at.elapsed().as_millis() as u64,
        "asr.final_settle.flushed"
    );
    record_and_forward_reconciled_events(
        state,
        media_state,
        call_id,
        pending.stream_id.as_deref(),
        text_calls,
        pending.quality_session.as_ref(),
        ForwardAsrEvents::new(vec![pending.event], CallerSpeechState::Finalizing),
    )
    .await
}

fn apply_pending_final_settle(
    media_state: &mut MediaSocketState,
    call_id: &str,
    stream_id: Option<&str>,
    quality_session: Option<&ActiveAsrQualitySession>,
    mut events: Vec<AsrTranscriptEvent>,
) -> Vec<AsrTranscriptEvent> {
    if events.is_empty() {
        return events;
    }
    if let Some(pending) = media_state.pending_final.take() {
        if has_emitted_final(&events) {
            tracing::debug!(
                gateway_call_id = call_id,
                previous_stream_id = pending.stream_id.as_deref(),
                stream_id,
                hold_reason = pending.hold_reason,
                held_ms = pending.created_at.elapsed().as_millis() as u64,
                "asr.final_settle.merged"
            );
            return merge_pending_final_into_events(pending.event, events);
        }
        media_state.pending_final = Some(pending);
        return events;
    }

    if media_state.quality_config.endpoint.final_settle_ms == 0 {
        return events;
    }
    if let Some((index, hold_reason)) =
        single_holdable_final_index(&events, &media_state.quality_config)
    {
        let final_event = events.remove(index);
        media_state.pending_final = Some(PendingFinalTranscript::new(
            final_event,
            stream_id,
            quality_session,
            &media_state.quality_config,
            hold_reason,
        ));
        tracing::debug!(
            gateway_call_id = call_id,
            stream_id,
            hold_reason,
            final_settle_ms = media_state.quality_config.endpoint.final_settle_ms,
            "asr.final_settle.held"
        );
    }
    events
}

fn has_emitted_final(events: &[AsrTranscriptEvent]) -> bool {
    events
        .iter()
        .any(|event| !event.is_suppressed() && event.event.is_final())
}

fn single_holdable_final_index(
    events: &[AsrTranscriptEvent],
    quality: &VoiceQualityConfig,
) -> Option<(usize, &'static str)> {
    let mut final_indexes = events
        .iter()
        .enumerate()
        .filter(|(_, event)| !event.is_suppressed() && event.event.is_final());
    let (index, event) = final_indexes.next()?;
    if final_indexes.next().is_some() {
        return None;
    }
    final_fragment_hold_reason(event.event.text(), quality).map(|reason| (index, reason))
}

fn final_fragment_hold_reason(text: &str, quality: &VoiceQualityConfig) -> Option<&'static str> {
    quality.endpoint.final_fragment_hold_reason(text)
}

fn merge_pending_final_into_events(
    pending_final: AsrTranscriptEvent,
    mut events: Vec<AsrTranscriptEvent>,
) -> Vec<AsrTranscriptEvent> {
    if let Some(index) = events
        .iter()
        .position(|event| !event.is_suppressed() && event.event.is_final())
    {
        let next_final = events.remove(index);
        let merged = merge_final_transcript_events(pending_final, next_final);
        events.insert(index, merged);
        events
    } else {
        let mut merged = Vec::with_capacity(events.len() + 1);
        merged.push(pending_final);
        merged.extend(events);
        merged
    }
}

fn merge_final_transcript_events(
    left: AsrTranscriptEvent,
    right: AsrTranscriptEvent,
) -> AsrTranscriptEvent {
    let merged_text = merge_transcript_fragments(left.event.text(), right.event.text());
    match right.event {
        TranscriptEvent::Final { update, .. } => AsrTranscriptEvent {
            event: TranscriptEvent::Final {
                text: merged_text,
                update,
            },
            decision: right.decision,
        },
        TranscriptEvent::Partial { .. } => left,
    }
}

fn merge_transcript_fragments(left: &str, right: &str) -> String {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    let left_lower = left.to_ascii_lowercase();
    let right_lower = right.to_ascii_lowercase();
    if right_lower.starts_with(&left_lower) {
        return right.to_string();
    }
    if left_lower.ends_with(&right_lower) {
        return left.to_string();
    }

    let left_tokens = left.split_whitespace().collect::<Vec<_>>();
    let right_tokens = right.split_whitespace().collect::<Vec<_>>();
    let max_overlap = left_tokens.len().min(right_tokens.len());
    let mut overlap = 0;
    for candidate in 1..=max_overlap {
        let left_slice = &left_tokens[left_tokens.len() - candidate..];
        let right_slice = &right_tokens[..candidate];
        if left_slice
            .iter()
            .zip(right_slice.iter())
            .all(|(left, right)| left.eq_ignore_ascii_case(right))
        {
            overlap = candidate;
        }
    }
    let mut merged = left_tokens
        .iter()
        .map(|token| (*token).to_string())
        .collect::<Vec<_>>();
    merged.extend(
        right_tokens
            .iter()
            .skip(overlap)
            .map(|token| (*token).to_string()),
    );
    merged.join(" ")
}

async fn reconcile_asr_final_events(
    state: &SharedState,
    gateway_call_id: &str,
    stream_id: Option<&str>,
    quality_session: Option<&ActiveAsrQualitySession>,
    events: Vec<AsrTranscriptEvent>,
) -> Vec<AsrTranscriptEvent> {
    let (mut latest_partial, include_transcript_text) = {
        let guard = state.read().await;
        (
            None::<(String, motlie_model::TranscriptionUpdate)>,
            guard.quality.config.logging.include_transcript_text,
        )
    };
    let mut reconciled = Vec::with_capacity(events.len());
    for event in events {
        if event.is_suppressed() {
            reconciled.push(event);
            continue;
        }

        let AsrTranscriptEvent { event, decision } = event;
        match event {
            TranscriptEvent::Partial { text, update } => {
                latest_partial = Some((text.clone(), update.clone()));
                reconciled.push(AsrTranscriptEvent {
                    event: TranscriptEvent::Partial { text, update },
                    decision,
                });
            }
            TranscriptEvent::Final { text, update } => {
                let latest_partial_text = latest_partial
                    .as_ref()
                    .map(|(partial_text, _)| partial_text.as_str());
                let reconciliation = reconcile_final_text(&text, latest_partial_text);
                emit_asr_final_reconciliation(
                    state,
                    gateway_call_id,
                    stream_id,
                    quality_session,
                    include_transcript_text,
                    &reconciliation,
                )
                .await;
                let selected_update = match reconciliation.selected_source {
                    FinalTextSource::LatestPartialExtension => latest_partial
                        .as_ref()
                        .map(|(_, partial_update)| {
                            final_update_from_selected_partial(partial_update.clone())
                        })
                        .unwrap_or_else(|| update.clone()),
                    FinalTextSource::AsrFinal => update,
                };
                latest_partial = None;
                reconciled.push(AsrTranscriptEvent {
                    event: TranscriptEvent::Final {
                        text: reconciliation.selected_text,
                        update: selected_update,
                    },
                    decision,
                });
            }
        }
    }
    reconciled
}

fn final_update_from_selected_partial(
    mut update: motlie_model::TranscriptionUpdate,
) -> motlie_model::TranscriptionUpdate {
    for segment in &mut update.segments {
        segment.final_segment = true;
    }
    update
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FinalTextReconciliation {
    selected_text: String,
    selected_source: FinalTextSource,
    final_text: String,
    latest_partial_text: Option<String>,
    final_chars: usize,
    latest_partial_chars: usize,
    selected_chars: usize,
    final_words: usize,
    latest_partial_words: usize,
    selected_words: usize,
    common_prefix_chars: usize,
    partial_is_strict_extension: bool,
    final_tail_word_chars: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FinalTextSource {
    AsrFinal,
    LatestPartialExtension,
}

impl FinalTextSource {
    fn label(self) -> &'static str {
        match self {
            Self::AsrFinal => "asr_final",
            Self::LatestPartialExtension => "latest_partial_extension",
        }
    }
}

fn reconcile_final_text(final_text: &str, latest_partial: Option<&str>) -> FinalTextReconciliation {
    let final_text = final_text.trim().to_string();
    let latest_partial_text = latest_partial.map(str::trim).and_then(|text| {
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    });
    let final_normalized = normalize_transcript_for_prefix_match(&final_text);
    let partial_normalized = latest_partial_text
        .as_deref()
        .map(normalize_transcript_for_prefix_match)
        .unwrap_or_default();
    let partial_is_strict_extension = !final_normalized.is_empty()
        && partial_normalized.chars().count() > final_normalized.chars().count()
        && partial_normalized.starts_with(&final_normalized);
    let (selected_text, selected_source) = if partial_is_strict_extension {
        (
            latest_partial_text
                .clone()
                .unwrap_or_else(|| final_text.clone()),
            FinalTextSource::LatestPartialExtension,
        )
    } else {
        (final_text.clone(), FinalTextSource::AsrFinal)
    };

    FinalTextReconciliation {
        final_chars: final_text.chars().count(),
        latest_partial_chars: latest_partial_text
            .as_deref()
            .map(|text| text.chars().count())
            .unwrap_or(0),
        selected_chars: selected_text.chars().count(),
        final_words: final_text.split_whitespace().count(),
        latest_partial_words: latest_partial_text
            .as_deref()
            .map(|text| text.split_whitespace().count())
            .unwrap_or(0),
        selected_words: selected_text.split_whitespace().count(),
        common_prefix_chars: common_prefix_chars(&final_normalized, &partial_normalized),
        partial_is_strict_extension,
        final_tail_word_chars: trailing_word_chars(&final_text),
        selected_text,
        selected_source,
        final_text,
        latest_partial_text,
    }
}

fn normalize_transcript_for_prefix_match(text: &str) -> String {
    normalize_transcript_whitespace(text)
        .trim_end_matches(['.', ',', '!', '?', ':', ';'])
        .to_ascii_lowercase()
}

fn normalize_transcript_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn common_prefix_chars(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(left, right)| left == right)
        .count()
}

fn trailing_word_chars(text: &str) -> usize {
    text.trim_end()
        .chars()
        .rev()
        .take_while(|ch| ch.is_alphanumeric() || *ch == '\'')
        .count()
}

async fn emit_asr_final_reconciliation(
    state: &SharedState,
    gateway_call_id: &str,
    stream_id: Option<&str>,
    quality_session: Option<&ActiveAsrQualitySession>,
    include_transcript_text: bool,
    reconciliation: &FinalTextReconciliation,
) {
    let include_plaintext = quality_session
        .map(|session| {
            transcript_plaintext_included(session.redaction_mode, include_transcript_text)
        })
        .unwrap_or(false);
    tracing::info!(
        gateway_call_id,
        stream_id,
        selected_source = reconciliation.selected_source.label(),
        final_chars = reconciliation.final_chars,
        latest_partial_chars = reconciliation.latest_partial_chars,
        selected_chars = reconciliation.selected_chars,
        final_words = reconciliation.final_words,
        latest_partial_words = reconciliation.latest_partial_words,
        selected_words = reconciliation.selected_words,
        common_prefix_chars = reconciliation.common_prefix_chars,
        partial_is_strict_extension = reconciliation.partial_is_strict_extension,
        final_tail_word_chars = reconciliation.final_tail_word_chars,
        asr_final_text = include_plaintext.then_some(reconciliation.final_text.as_str()),
        latest_partial_text = include_plaintext
            .then_some(reconciliation.latest_partial_text.as_deref())
            .flatten(),
        selected_text = include_plaintext.then_some(reconciliation.selected_text.as_str()),
        "asr.final_text_reconciled"
    );
    let Some(session) = quality_session else {
        return;
    };
    let mut payload = map_from_value(json!({
        "asr_session_id": session.asr_session_id.as_str(),
        "utterance_id": session.utterance_id.as_str(),
        "stream_id": stream_id,
        "selected_source": reconciliation.selected_source.label(),
        "final_chars": reconciliation.final_chars,
        "latest_partial_chars": reconciliation.latest_partial_chars,
        "selected_chars": reconciliation.selected_chars,
        "final_words": reconciliation.final_words,
        "latest_partial_words": reconciliation.latest_partial_words,
        "selected_words": reconciliation.selected_words,
        "common_prefix_chars": reconciliation.common_prefix_chars,
        "partial_is_strict_extension": reconciliation.partial_is_strict_extension,
        "final_tail_word_chars": reconciliation.final_tail_word_chars,
    }));
    insert_transcript_text_fields(
        &mut payload,
        session.redaction_mode,
        include_transcript_text,
        "asr_final_text",
        &reconciliation.final_text,
    );
    if let Some(partial) = &reconciliation.latest_partial_text {
        insert_transcript_text_fields(
            &mut payload,
            session.redaction_mode,
            include_transcript_text,
            "latest_partial_text",
            partial,
        );
    }
    insert_transcript_text_fields(
        &mut payload,
        session.redaction_mode,
        include_transcript_text,
        "selected_text",
        &reconciliation.selected_text,
    );
    state.write().await.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id: session.config_id.clone(),
            redaction_mode: session.redaction_mode,
            span_name: "asr.final_text_reconciliation",
            category: "asr_generation",
            duration: Duration::ZERO,
            critical_path: false,
            concurrent: false,
            payload,
        },
    );
}

#[derive(Clone, Debug)]
struct ConversationTranscriptEvent {
    event: TranscriptEvent,
    turn_id: Option<String>,
    source_asr_session_ids: Vec<String>,
    source_utterance_ids: Vec<String>,
    confidence: Option<f32>,
    stability: Option<f32>,
}

async fn forward_conversation_events(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    conversation: Option<&ConversationRuntime>,
    gateway_call_id: &str,
    events: Vec<ConversationTranscriptEvent>,
    quality_config: Option<&VoiceQualityConfig>,
) {
    let Some(conversation) = conversation else {
        return;
    };
    for event in events {
        if let Err(error) = conversation::handle_transcript_event_with_metadata(
            state,
            media_registry,
            conversation,
            gateway_call_id,
            event.event,
            quality_config,
            conversation::ConversationTranscriptMetadata {
                turn_id: event.turn_id.as_deref(),
                source_asr_session_ids: Some(&event.source_asr_session_ids),
                source_utterance_ids: Some(&event.source_utterance_ids),
                confidence: event.confidence,
                stability: event.stability,
            },
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
    early_response: Option<EarlyResponsePipelineHandle>,
    quality_session: Option<&'a ActiveAsrQualitySession>,
    echo_config: Option<&'a EchoSuppressionQualityConfig>,
    partial_speech_state: CallerSpeechState,
}

#[derive(Default)]
struct TranscriptRecordOutcome {
    reset_requested: bool,
    conversation_events: Vec<ConversationTranscriptEvent>,
}

impl TranscriptRecordOutcome {
    fn merge(&mut self, other: Self) {
        self.reset_requested |= other.reset_requested;
        self.conversation_events.extend(other.conversation_events);
    }
}

struct FinalTurnCandidate {
    turn_id: String,
    text: String,
    finalized_at: Instant,
    transcript_event_ids: Vec<String>,
    asr_session_ids: Vec<String>,
    utterance_ids: Vec<String>,
    confidence: Option<f32>,
    coalesced_turn_ids: Vec<String>,
    members: Vec<FinalTurnMember>,
}

struct FinalTurnMember {
    utterance_id: String,
    text: String,
    transcript_event_ids: Vec<String>,
}

struct PartialTurnCandidate {
    sequence: u64,
    utterance_id: String,
    text: String,
    confidence: Option<f32>,
    stability: Option<f32>,
    speech_state: CallerSpeechState,
    quality_session: ActiveAsrQualitySession,
}

fn caller_speech_state_label(state: CallerSpeechState) -> &'static str {
    match state {
        CallerSpeechState::Speaking => "speaking",
        CallerSpeechState::EndpointCandidate => "endpoint_candidate",
        CallerSpeechState::Finalizing => "finalizing",
    }
}

fn transcript_event_confidence(event: &TranscriptEvent) -> Option<f32> {
    let update = match event {
        TranscriptEvent::Partial { update, .. } | TranscriptEvent::Final { update, .. } => update,
    };
    update
        .segments
        .iter()
        .rev()
        .find_map(|segment| normalized_transcript_score(segment.confidence))
}

fn normalized_transcript_score(score: Option<f32>) -> Option<f32> {
    score.filter(|score| score.is_finite() && *score >= 0.0 && *score <= 1.0)
}

fn partial_stability_score(
    previous_partial: Option<&str>,
    current_partial: &str,
    speech_state: CallerSpeechState,
) -> Option<f32> {
    let previous = normalize_partial_for_stability(previous_partial?)?;
    let current = normalize_partial_for_stability(current_partial)?;
    let common = common_prefix_chars(&previous, &current);
    let current_chars = current.chars().count();
    if current_chars == 0 {
        return None;
    }
    let prefix_ratio = common as f32 / current_chars as f32;
    let base = if previous == current {
        0.92
    } else if current.starts_with(&previous) {
        0.55 + (prefix_ratio * 0.35)
    } else {
        prefix_ratio * 0.70
    };
    let speech_state_bonus = match speech_state {
        CallerSpeechState::Speaking => 0.0,
        CallerSpeechState::EndpointCandidate => 0.04,
        CallerSpeechState::Finalizing => 0.08,
    };
    Some((base + speech_state_bonus).clamp(0.0, 1.0))
}

fn normalize_partial_for_stability(text: &str) -> Option<String> {
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    (!normalized.is_empty()).then_some(normalized)
}

fn coalesce_final_turns(final_turns: Vec<FinalTurnCandidate>) -> Vec<FinalTurnCandidate> {
    if final_turns.len() <= 1 {
        return final_turns;
    }
    let mut turn_id = None;
    let mut text = String::new();
    let mut finalized_at = None;
    let mut transcript_event_ids = Vec::new();
    let mut asr_session_ids = Vec::new();
    let mut utterance_ids = Vec::new();
    let mut confidence = None;
    let mut coalesced_turn_ids = Vec::new();
    let mut members = Vec::new();
    for turn in final_turns {
        if turn_id.is_none() {
            turn_id = Some(turn.turn_id);
        }
        for asr_session_id in turn.asr_session_ids {
            if !asr_session_ids.contains(&asr_session_id) {
                asr_session_ids.push(asr_session_id);
            }
        }
        for utterance_id in turn.utterance_ids {
            if !utterance_ids.contains(&utterance_id) {
                utterance_ids.push(utterance_id);
            }
        }
        if turn.confidence.is_some() {
            confidence = turn.confidence;
        }
        coalesced_turn_ids.extend(turn.coalesced_turn_ids);
        members.extend(turn.members);
        let trimmed = turn.text.trim();
        if !trimmed.is_empty() {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(trimmed);
        }
        finalized_at = Some(turn.finalized_at);
        transcript_event_ids.extend(turn.transcript_event_ids);
    }
    if text.is_empty() {
        Vec::new()
    } else {
        vec![FinalTurnCandidate {
            turn_id: turn_id.unwrap_or_else(new_local_turn_id),
            text,
            finalized_at: finalized_at.unwrap_or_else(Instant::now),
            transcript_event_ids,
            asr_session_ids,
            utterance_ids,
            confidence,
            coalesced_turn_ids,
            members,
        }]
    }
}

fn new_local_turn_id() -> String {
    format!("turn_{}", uuid::Uuid::new_v4().simple())
}

fn tts_matches_asr_source(
    tts: &TtsPlaybackState,
    quality_session: Option<&ActiveAsrQualitySession>,
) -> bool {
    let Some(session) = quality_session else {
        return false;
    };
    tts.source_utterance_ids
        .iter()
        .any(|utterance_id| utterance_id == &session.utterance_id)
        || tts
            .source_asr_session_ids
            .iter()
            .any(|asr_session_id| asr_session_id == &session.asr_session_id)
}

fn assistant_echo_match(
    call: &CallSession,
    config: &EchoSuppressionQualityConfig,
    transcript_text: &str,
    quality_session: Option<&ActiveAsrQualitySession>,
) -> Option<AssistantEchoMatch> {
    let tts = call.tts.as_ref()?;
    if tts_matches_asr_source(tts, quality_session) {
        return None;
    }
    if !tts_in_echo_window(tts, config.tail_window_ms as i64) {
        return None;
    }

    match_assistant_echo_signature(config, transcript_text, &tts.echo_signature)
}

fn trim_quarantined_echo_prefix(text: &str, prefix: &str) -> Option<String> {
    let end = token_prefix_end(text, prefix)?;
    let trimmed = text[end..]
        .trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '.' | ',' | ':' | ';' | '!' | '?' | '-')
        })
        .trim();
    Some(trimmed.to_string())
}

fn token_prefix_end(text: &str, prefix: &str) -> Option<usize> {
    let prefix_tokens = normalized_tokens(prefix);
    if prefix_tokens.is_empty() {
        return None;
    }
    let text_tokens = normalized_tokens_with_ends(text);
    if text_tokens.len() < prefix_tokens.len() {
        return None;
    }
    for (index, prefix_token) in prefix_tokens.iter().enumerate() {
        if text_tokens[index].0 != *prefix_token {
            return None;
        }
    }
    text_tokens
        .get(prefix_tokens.len().saturating_sub(1))
        .map(|(_, end)| *end)
}

fn normalized_tokens(text: &str) -> Vec<String> {
    normalized_tokens_with_ends(text)
        .into_iter()
        .map(|(token, _)| token)
        .collect()
}

fn normalized_tokens_with_ends(text: &str) -> Vec<(String, usize)> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut token_end = 0usize;
    for (index, ch) in text.char_indices() {
        if ch.is_alphanumeric() || ch == '\'' {
            token.extend(ch.to_lowercase());
            token_end = index + ch.len_utf8();
        } else if !token.is_empty() {
            tokens.push((std::mem::take(&mut token), token_end));
        }
    }
    if !token.is_empty() {
        tokens.push((token, token_end));
    }
    tokens
}

fn sanitize_committed_echo_prefix(
    guard: &mut GatewayState,
    gateway_call_id: &str,
    echo_config: &EchoSuppressionQualityConfig,
    quality_session: Option<&ActiveAsrQualitySession>,
    redaction_mode: RedactionMode,
    text: &mut String,
) -> bool {
    let Some((prefix, age_ms)) = take_echo_prefix_candidate(guard, gateway_call_id) else {
        return false;
    };
    if age_ms < 0 || age_ms as u64 > echo_config.tail_window_ms {
        return false;
    }
    let Some(sanitized) = trim_quarantined_echo_prefix(text, &prefix) else {
        emit_committed_echo_prefix_span(
            guard,
            gateway_call_id,
            quality_session,
            redaction_mode,
            CommittedEchoPrefixSpan {
                action: "preserved",
                removed_chars: prefix.chars().count(),
                removed_words: prefix.split_whitespace().count(),
                remaining_chars: text.chars().count(),
                remaining_words: text.split_whitespace().count(),
                prefix_age_ms: age_ms as u64,
            },
        );
        return false;
    };

    let removed_chars = text
        .chars()
        .count()
        .saturating_sub(sanitized.chars().count());
    let removed_words = prefix.split_whitespace().count();
    let remaining_chars = sanitized.chars().count();
    let remaining_words = sanitized.split_whitespace().count();
    *text = sanitized;
    let dropped = text.trim().is_empty();
    emit_committed_echo_prefix_span(
        guard,
        gateway_call_id,
        quality_session,
        redaction_mode,
        CommittedEchoPrefixSpan {
            action: if dropped {
                "dropped_empty"
            } else {
                "trimmed_prefix"
            },
            removed_chars,
            removed_words,
            remaining_chars,
            remaining_words,
            prefix_age_ms: age_ms as u64,
        },
    );
    dropped
}

fn take_echo_prefix_candidate(
    guard: &mut GatewayState,
    gateway_call_id: &str,
) -> Option<(String, i64)> {
    let call = guard.calls.get_mut(gateway_call_id)?;
    let prefix = call.last_echo_suppressed_text.take()?;
    let at = call.last_echo_suppressed_at.take()?;
    let age_ms = Utc::now().signed_duration_since(at).num_milliseconds();
    Some((prefix, age_ms))
}

struct CommittedEchoPrefixSpan {
    action: &'static str,
    removed_chars: usize,
    removed_words: usize,
    remaining_chars: usize,
    remaining_words: usize,
    prefix_age_ms: u64,
}

fn emit_committed_echo_prefix_span(
    guard: &mut GatewayState,
    gateway_call_id: &str,
    quality_session: Option<&ActiveAsrQualitySession>,
    redaction_mode: RedactionMode,
    span: CommittedEchoPrefixSpan,
) {
    let config_id = quality_session
        .map(|session| session.config_id.clone())
        .unwrap_or_else(|| guard.quality.config.config_id());
    let payload = json!({
        "action": span.action,
        "removed_chars": span.removed_chars,
        "removed_words": span.removed_words,
        "remaining_chars": span.remaining_chars,
        "remaining_words": span.remaining_words,
        "prefix_age_ms": span.prefix_age_ms,
    });
    let payload = match payload {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    guard.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id,
            redaction_mode,
            span_name: "conversation.committed_echo_prefix",
            category: "echo_suppression",
            duration: Duration::ZERO,
            critical_path: true,
            concurrent: false,
            payload,
        },
    );
}

fn transcript_event_with_text(event: &TranscriptEvent, text: String) -> TranscriptEvent {
    match event {
        TranscriptEvent::Partial { update, .. } => TranscriptEvent::Partial {
            text,
            update: update.clone(),
        },
        TranscriptEvent::Final { update, .. } => TranscriptEvent::Final {
            text,
            update: update.clone(),
        },
    }
}

fn tts_in_echo_window(tts: &TtsPlaybackState, tail_window_ms: i64) -> bool {
    match tts.status {
        TtsPlaybackStatus::Queued
        | TtsPlaybackStatus::Playing
        | TtsPlaybackStatus::MarkSent
        | TtsPlaybackStatus::Canceling => true,
        TtsPlaybackStatus::Completed | TtsPlaybackStatus::Canceled => {
            let age_ms = Utc::now()
                .signed_duration_since(tts.updated_at)
                .num_milliseconds();
            (0..=tail_window_ms).contains(&age_ms)
        }
        TtsPlaybackStatus::Failed => false,
    }
}

async fn record_transcript_events(
    state: &SharedState,
    gateway_call_id: &str,
    events: Vec<AsrTranscriptEvent>,
    mut context: TranscriptRecordContext<'_>,
) -> TranscriptRecordOutcome {
    let mut guard = state.write().await;
    let include_transcript_text = guard.quality.config.logging.include_transcript_text;
    let redaction_mode = guard.quality.config.logging.redaction_mode;
    let live_echo_config = guard.quality.config.echo_suppression.clone();
    let echo_config = context.echo_config.unwrap_or(&live_echo_config);
    let mut reset_requested = false;
    let mut final_turns = Vec::new();
    let mut partial_turns = Vec::new();
    let mut conversation_events = Vec::new();
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
        let mut text = event.event.text().to_string();
        let suppressed = event.is_suppressed();
        reset_requested |= event.requires_session_reset();
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
        let previous_partial = call.and_then(|call| call.current_partial.clone());
        let transcript_confidence = transcript_event_confidence(&event.event);
        // `caller.partial.stability` is a gateway stream-convergence/churn
        // heuristic for preparation/routing/debounce analysis only. It is never
        // truth, final response input, model/ASR confidence, calibrated
        // probability, or a value to average/combine with model confidence.
        let transcript_stability = if matches!(kind, TranscriptKind::Partial) {
            partial_stability_score(
                previous_partial.as_deref(),
                &text,
                context.partial_speech_state,
            )
        } else {
            None
        };
        let assistant_echo = if suppressed {
            None
        } else {
            call.and_then(|call| {
                assistant_echo_match(call, echo_config, &text, context.quality_session)
            })
        };
        if let Some(capture) = context.capture.as_deref_mut() {
            record_transcript_capture(
                capture,
                kind_label,
                &text,
                suppressed || assistant_echo.is_some(),
            );
        }

        if suppressed {
            let suppression_reason = event
                .suppression_reason()
                .map(AsrTranscriptSuppressionReason::label)
                .unwrap_or("adapter_suppressed");
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
            guard.emit_quality_transcript_suppressed(
                gateway_call_id,
                context.quality_session,
                kind_label,
                suppression_reason,
                &text,
                Map::new(),
            );
            continue;
        }

        if let Some(echo) = assistant_echo {
            let transcript_preview = include_transcript_text.then(|| transcript_preview(&text));
            guard.record_echo_suppressed_transcript(gateway_call_id, &text);
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
                transcript_chars = text.chars().count(),
                token_coverage_percent = echo.token_coverage_percent,
                longest_token_run = echo.longest_token_run,
                transcript_preview = transcript_preview.as_deref(),
                "transcript.suppressed_assistant_echo"
            );
            let mut extra = Map::new();
            extra.insert(
                "token_coverage_percent".to_string(),
                json!(echo.token_coverage_percent),
            );
            extra.insert(
                "longest_token_run".to_string(),
                json!(echo.longest_token_run),
            );
            guard.emit_quality_transcript_suppressed(
                gateway_call_id,
                context.quality_session,
                kind_label,
                "assistant_echo",
                &text,
                extra,
            );
            continue;
        }

        if matches!(kind, TranscriptKind::Final)
            && sanitize_committed_echo_prefix(
                &mut guard,
                gateway_call_id,
                echo_config,
                context.quality_session,
                redaction_mode,
                &mut text,
            )
        {
            continue;
        }

        let turn_id = matches!(kind, TranscriptKind::Final).then(new_local_turn_id);
        guard.add_transcript(gateway_call_id, kind.clone(), text.clone());
        if matches!(kind, TranscriptKind::Final) {
            let transcript_event_id = guard
                .quality
                .event_sink
                .is_enabled()
                .then(|| format!("trn_{}", uuid::Uuid::new_v4().simple()));
            let turn_id = turn_id.clone().unwrap_or_else(new_local_turn_id);
            let transcript_event_ids = transcript_event_id.into_iter().collect::<Vec<_>>();
            let utterance_ids = context
                .quality_session
                .map(|session| vec![session.utterance_id.clone()])
                .unwrap_or_default();
            let members = context
                .quality_session
                .map(|session| {
                    vec![FinalTurnMember {
                        utterance_id: session.utterance_id.clone(),
                        text: text.clone(),
                        transcript_event_ids: transcript_event_ids.clone(),
                    }]
                })
                .unwrap_or_default();
            final_turns.push(FinalTurnCandidate {
                turn_id: turn_id.clone(),
                text: text.clone(),
                finalized_at: Instant::now(),
                transcript_event_ids,
                asr_session_ids: context
                    .quality_session
                    .map(|session| vec![session.asr_session_id.clone()])
                    .unwrap_or_default(),
                utterance_ids,
                confidence: transcript_confidence,
                coalesced_turn_ids: vec![turn_id],
                members,
            });
        } else if let Some(session) = context.quality_session {
            partial_turns.push(PartialTurnCandidate {
                sequence: partial_turns.len() as u64,
                utterance_id: session.utterance_id.clone(),
                text: text.clone(),
                confidence: transcript_confidence,
                stability: transcript_stability,
                speech_state: context.partial_speech_state,
                quality_session: session.clone(),
            });
        }
        let source_asr_session_ids = context
            .quality_session
            .map(|session| vec![session.asr_session_id.clone()])
            .unwrap_or_default();
        let source_utterance_ids = context
            .quality_session
            .map(|session| vec![session.utterance_id.clone()])
            .unwrap_or_default();
        conversation_events.push(ConversationTranscriptEvent {
            event: transcript_event_with_text(&event.event, text.clone()),
            turn_id,
            source_asr_session_ids,
            source_utterance_ids,
            confidence: transcript_confidence,
            stability: transcript_stability,
        });
        let transcript_text =
            transcript_plaintext_included(redaction_mode, include_transcript_text)
                .then_some(text.as_str());
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
            transcript_confidence,
            transcript_stability,
            turn_id = conversation_events
                .last()
                .and_then(|event| event.turn_id.as_deref()),
            transcript_text,
            "{kind_label}"
        );
    }
    drop(guard);
    if let Some(early_response) = context.early_response.as_ref() {
        for partial_turn in &partial_turns {
            early_response.try_send(EarlyResponseInput::Partial(EarlyResponsePartial {
                call_id: gateway_call_id.to_string(),
                utterance_id: partial_turn.utterance_id.clone(),
                sequence: partial_turn.sequence,
                received_at_ms: partial_turn.quality_session.opened_at.elapsed().as_millis() as u64,
                text: partial_turn.text.clone(),
                confidence: partial_turn.confidence,
                stability: partial_turn.stability,
                speech_state: partial_turn.speech_state,
            }));
        }
    }
    if let Some(text_calls) = context.text_calls {
        for partial_turn in &partial_turns {
            match text_calls
                .send_caller_partial(
                    gateway_call_id,
                    partial_turn.utterance_id.clone(),
                    partial_turn.text.clone(),
                    partial_turn.confidence,
                    partial_turn.stability,
                    partial_turn.speech_state,
                )
                .await
            {
                Ok(true) => state.write().await.emit_quality_caller_partial_sent(
                    gateway_call_id,
                    &partial_turn.quality_session,
                    &partial_turn.text,
                    partial_turn.confidence,
                    partial_turn.stability,
                    caller_speech_state_label(partial_turn.speech_state),
                ),
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(
                        gateway_call_id,
                        error = %error,
                        "text_call.caller_partial.forward_failed"
                    );
                }
            }
        }
    }
    let final_turns = coalesce_final_turns(final_turns);
    if let Some(early_response) = context.early_response.as_ref() {
        for final_turn in &final_turns {
            early_response.try_send(EarlyResponseInput::CommitBoundary(
                EarlyResponseCommitBoundary {
                    call_id: gateway_call_id.to_string(),
                    sequence: final_turn.finalized_at.elapsed().as_millis() as u64,
                    turn_id: final_turn.turn_id.clone(),
                    coalesced_turn_ids: final_turn.coalesced_turn_ids.clone(),
                    final_text: final_turn.text.clone(),
                    members: final_turn
                        .members
                        .iter()
                        .enumerate()
                        .map(|(member_index, member)| EarlyResponseCommitMember {
                            utterance_id: member.utterance_id.clone(),
                            member_index,
                            member_final_text: member.text.clone(),
                            transcript_event_ids: member.transcript_event_ids.clone(),
                        })
                        .collect(),
                },
            ));
        }
    }
    for final_turn in final_turns {
        let mut emitted_join = false;
        if let Some(text_calls) = context.text_calls {
            match text_calls
                .send_caller_turn_with_utterance(
                    gateway_call_id,
                    final_turn.text.clone(),
                    final_turn.finalized_at,
                    final_turn.utterance_ids.first().cloned(),
                )
                .await
            {
                Ok(Some(turn_id)) => {
                    let mut guard = state.write().await;
                    emit_quality_turn_join(
                        &mut guard,
                        gateway_call_id,
                        &turn_id,
                        &final_turn,
                        context.quality_session,
                        true,
                    );
                    emitted_join = true;
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
        if !emitted_join {
            let mut guard = state.write().await;
            emit_quality_turn_join(
                &mut guard,
                gateway_call_id,
                &final_turn.turn_id,
                &final_turn,
                context.quality_session,
                true,
            );
            tracing::debug!(
                gateway_call_id,
                turn_id = final_turn.turn_id.as_str(),
                "conversation.local_caller_turn.join_recorded"
            );
        }
    }
    TranscriptRecordOutcome {
        reset_requested,
        conversation_events,
    }
}

fn emit_quality_turn_join(
    state: &mut crate::operator::state::GatewayState,
    gateway_call_id: &str,
    turn_id: &str,
    final_turn: &FinalTurnCandidate,
    quality_session: Option<&ActiveAsrQualitySession>,
    caller_turn_sent: bool,
) {
    state.emit_quality_caller_turn_sent(
        gateway_call_id,
        turn_id,
        &final_turn.text,
        quality_session,
        CallerTurnEventMetadata {
            asr_session_id: final_turn
                .asr_session_ids
                .first()
                .cloned()
                .or_else(|| quality_session.map(|session| session.asr_session_id.clone())),
            asr_session_ids: final_turn.asr_session_ids.clone(),
            utterance_id: final_turn.utterance_ids.first().cloned(),
            utterance_ids: final_turn.utterance_ids.clone(),
            confidence: final_turn.confidence,
            transcript_event_count: final_turn.transcript_event_ids.len(),
            coalesced_turn_ids: final_turn.coalesced_turn_ids.clone(),
        },
    );
    if let Some(session) = quality_session {
        for transcript_event_id in &final_turn.transcript_event_ids {
            state.emit_quality_asr_turn_mapped(
                gateway_call_id,
                session,
                turn_id,
                transcript_event_id,
                caller_turn_sent,
            );
        }
    }
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
    use crate::text_calls::turns::{CallerSpeechState, GatewayTextFrame};
    use crate::text_calls::SharedTextCallRegistry;
    use crate::tts::{
        LiveTtsBackend, OutboundTtsFactory, TtsAudio, TtsRegistry, PIPER_SAMPLE_RATE_HZ,
    };

    fn continue_decision(speech_onset: bool, speech_state: CallerSpeechState) -> AsrFrameDecision {
        AsrFrameDecision::Continue {
            speech_onset,
            speech_state,
        }
    }

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
    fn echo_characterization_estimates_reference_delay() {
        let outbound: Vec<i16> = (0..320)
            .map(|index| ((((index * 37) % 211) as i16) - 105) * 120)
            .collect();
        let delay_samples = 64usize;
        let inbound_len = 120usize;
        let end = outbound.len() - delay_samples;
        let inbound = outbound[end - inbound_len..end].to_vec();

        let metrics = characterize_echo(&inbound, &outbound, 8_000, 20)
            .expect("delayed reference should characterize");

        assert!(metrics.correlation_peak > 0.99);
        assert_eq!(metrics.estimated_delay_ms, 8);
    }

    #[test]
    fn echo_characterization_ignores_zero_energy_reference() {
        let inbound = vec![0i16; 160];
        let outbound = vec![0i16; 320];

        assert!(characterize_echo(&inbound, &outbound, 8_000, 20).is_none());
    }

    fn audio_classifier_test_config() -> AudioBargeInMediaQualityConfig {
        AudioBargeInMediaQualityConfig {
            mode: AudioBargeInMode::EchoAwareOnset,
            trusted_onset_min_windows: 1,
            calibration_min_playback_only_ms: 20,
            delay_search_min_ms: 0,
            delay_search_max_ms: 20,
            erl_min_db: -80.0,
            erl_max_db: 10.0,
            min_echo_margin_db_floor: 3.0,
            min_echo_margin_db_ceiling: 18.0,
            ..AudioBargeInMediaQualityConfig::default()
        }
    }

    fn audio_classifier_test_format() -> MediaFormat {
        MediaFormat {
            encoding: "PCMU".to_string(),
            sample_rate_hz: 8_000,
            channels: 1,
        }
    }

    fn audio_classifier_reference() -> Vec<i16> {
        (0..360)
            .map(|index| ((((index * 37) % 211) as i16) - 105) * 120)
            .collect()
    }

    fn active_playback_evidence_for_trigger() -> CallerOnsetEvidence {
        let format = audio_classifier_test_format();
        base_caller_onset_evidence(CallerOnsetEvidenceSeed {
            decision: AudioOnsetDecision::TrustedCallerOnset,
            confidence: 1.0,
            playback_state: PlaybackEchoState::ActivePlayback,
            playback_ref: Some(&ActiveSpeechPlaybackRef {
                playback_id: "playback-1".to_string(),
                playback_epoch: 1,
            }),
            caller_active_since: Some(Instant::now()),
            now: Instant::now(),
            window_ms: 20,
            format: &format,
            inbound_rms_dbfs: -20.0,
            outbound_rms_dbfs: Some(-30.0),
            invalidation: None,
        })
    }

    #[test]
    fn audio_trusted_onset_trigger_is_opt_in_and_edge_triggered() {
        let mut config = VoiceQualityConfig::default();
        let evidence = active_playback_evidence_for_trigger();
        let mut state = AudioBargeInEvidenceState::default();

        assert!(!take_audio_trusted_onset_trigger(
            &config, &mut state, &evidence
        ));

        config.barge_in.enabled = true;
        config.conversation_policy.mode = ConversationPolicyMode::BargeInCancelOnly;
        config.audio_barge_in.media.mode = AudioBargeInMode::EchoAwareOnset;
        assert!(take_audio_trusted_onset_trigger(
            &config, &mut state, &evidence
        ));
        assert!(!take_audio_trusted_onset_trigger(
            &config, &mut state, &evidence
        ));
    }

    #[test]
    fn audio_trusted_onset_trigger_ignores_idle_evidence() {
        let mut config = VoiceQualityConfig::default();
        config.barge_in.enabled = true;
        config.conversation_policy.mode = ConversationPolicyMode::BargeInCancelOnly;
        config.audio_barge_in.media.mode = AudioBargeInMode::EchoAwareOnset;
        let format = audio_classifier_test_format();
        let evidence = base_caller_onset_evidence(CallerOnsetEvidenceSeed {
            decision: AudioOnsetDecision::TrustedCallerOnset,
            confidence: 1.0,
            playback_state: PlaybackEchoState::Idle,
            playback_ref: None,
            caller_active_since: Some(Instant::now()),
            now: Instant::now(),
            window_ms: 20,
            format: &format,
            inbound_rms_dbfs: -20.0,
            outbound_rms_dbfs: None,
            invalidation: None,
        });

        assert!(!take_audio_trusted_onset_trigger(
            &config,
            &mut AudioBargeInEvidenceState::default(),
            &evidence
        ));
    }

    #[test]
    fn playback_state_without_active_labels_tail_and_inter_segment_gap() {
        assert_eq!(
            playback_state_without_active(None, 0),
            PlaybackEchoState::Idle
        );
        assert_eq!(
            playback_state_without_active(Some(Duration::from_millis(20)), 0),
            PlaybackEchoState::RecentTail
        );
        assert_eq!(
            playback_state_without_active(Some(Duration::from_millis(20)), 1),
            PlaybackEchoState::InterSegmentGap
        );
    }

    #[test]
    fn playback_only_calibration_accepts_audible_echo_without_residual_caller() {
        let config = audio_classifier_test_config();
        let format = audio_classifier_test_format();
        let outbound = audio_classifier_reference();
        let echo = outbound[outbound.len() - 160..].to_vec();
        assert!(sample_stats(&echo).has_speech_energy(&SpeechQualityConfig::default()));

        let metrics = playback_only_calibration_metrics(
            &config,
            &format,
            &echo,
            &SpeechQualityConfig::default(),
            &outbound,
            false,
        );

        assert!(metrics.is_some());
    }

    #[test]
    fn playback_only_calibration_rejects_residual_caller_candidate() {
        let config = audio_classifier_test_config();
        let format = audio_classifier_test_format();
        let outbound = audio_classifier_reference();
        let echo = outbound[outbound.len() - 160..].to_vec();
        let caller_mix: Vec<i16> = echo
            .iter()
            .enumerate()
            .map(|(index, echo_sample)| {
                let caller_sample = if index % 17 < 8 {
                    14_000i16
                } else {
                    -14_000i16
                };
                echo_sample.saturating_add(caller_sample)
            })
            .collect();

        let metrics = playback_only_calibration_metrics(
            &config,
            &format,
            &caller_mix,
            &SpeechQualityConfig::default(),
            &outbound,
            false,
        );

        assert!(metrics.is_none());
        assert!(playback_only_calibration_metrics(
            &config,
            &format,
            &echo,
            &SpeechQualityConfig::default(),
            &outbound,
            true,
        )
        .is_none());
    }

    #[test]
    fn audio_transport_invalidation_uses_configured_window_jitter() {
        let mut config = audio_classifier_test_config();
        config.max_jitter_ms = 30;
        let mut transport = InboundTransportStats {
            packets_total: 1,
            ..InboundTransportStats::default()
        };
        transport.jitter_samples_ms.push(45);
        let mut state = AudioBargeInEvidenceState::default();

        assert_eq!(
            state.transport_invalidation(&transport, &config),
            Some(AudioEvidenceInvalidation::TransportInvalid)
        );
    }

    #[test]
    fn audio_barge_in_classifier_vetoes_calibrated_playback_echo() {
        let config = audio_classifier_test_config();
        let format = audio_classifier_test_format();
        let outbound = audio_classifier_reference();
        let inbound = outbound[outbound.len() - 160..].to_vec();
        let metrics = characterize_echo(
            &inbound,
            &outbound,
            format.sample_rate_hz,
            config.delay_search_max_ms,
        )
        .expect("echo should characterize");
        let mut state = AudioBargeInEvidenceState::default();
        state.observe_playback_only_calibration(&config, 20, metrics);

        let evidence = state.classify_active_playback_window(ActivePlaybackWindow {
            config: &config,
            playback_ref: &ActiveSpeechPlaybackRef {
                playback_id: "playback-1".to_string(),
                playback_epoch: 3,
            },
            format: &format,
            samples: &inbound,
            stats: &sample_stats(&inbound),
            speech: &SpeechQualityConfig::default(),
            outbound_reference: &outbound,
            transport_invalidation: None,
            frame_duration_ms: 20,
            now: Instant::now(),
        });

        assert_eq!(evidence.decision, AudioOnsetDecision::LikelyAssistantEcho);
        assert_eq!(evidence.playback_epoch, Some(3));
    }

    #[test]
    fn audio_barge_in_classifier_rejects_amplified_echo_without_residual_speech() {
        let config = audio_classifier_test_config();
        let format = audio_classifier_test_format();
        let outbound = audio_classifier_reference();
        let echo = outbound[outbound.len() - 160..].to_vec();
        let metrics = characterize_echo(
            &echo,
            &outbound,
            format.sample_rate_hz,
            config.delay_search_max_ms,
        )
        .expect("echo should characterize");
        let mut state = AudioBargeInEvidenceState::default();
        state.observe_playback_only_calibration(&config, 20, metrics);
        let amplified_echo: Vec<i16> = echo.iter().map(|sample| sample.saturating_mul(2)).collect();

        let evidence = state.classify_active_playback_window(ActivePlaybackWindow {
            config: &config,
            playback_ref: &ActiveSpeechPlaybackRef {
                playback_id: "playback-1".to_string(),
                playback_epoch: 3,
            },
            format: &format,
            samples: &amplified_echo,
            stats: &sample_stats(&amplified_echo),
            speech: &SpeechQualityConfig::default(),
            outbound_reference: &outbound,
            transport_invalidation: None,
            frame_duration_ms: 20,
            now: Instant::now(),
        });

        assert_eq!(evidence.decision, AudioOnsetDecision::LikelyAssistantEcho);
        assert!(evidence.echo_margin_db.is_some_and(|margin| margin >= 3.0));
    }

    #[test]
    fn audio_barge_in_classifier_trusts_calibrated_residual_caller_speech() {
        let config = audio_classifier_test_config();
        let format = audio_classifier_test_format();
        let outbound = audio_classifier_reference();
        let echo = outbound[outbound.len() - 160..].to_vec();
        let metrics = characterize_echo(
            &echo,
            &outbound,
            format.sample_rate_hz,
            config.delay_search_max_ms,
        )
        .expect("echo should characterize");
        let mut state = AudioBargeInEvidenceState::default();
        state.observe_playback_only_calibration(&config, 20, metrics);
        let caller_onset: Vec<i16> = echo
            .iter()
            .enumerate()
            .map(|(index, echo_sample)| {
                let caller_sample = if index % 17 < 8 {
                    14_000i16
                } else {
                    -14_000i16
                };
                echo_sample.saturating_add(caller_sample)
            })
            .collect();

        let evidence = state.classify_active_playback_window(ActivePlaybackWindow {
            config: &config,
            playback_ref: &ActiveSpeechPlaybackRef {
                playback_id: "playback-1".to_string(),
                playback_epoch: 3,
            },
            format: &format,
            samples: &caller_onset,
            stats: &sample_stats(&caller_onset),
            speech: &SpeechQualityConfig::default(),
            outbound_reference: &outbound,
            transport_invalidation: None,
            frame_duration_ms: 20,
            now: Instant::now(),
        });

        assert_eq!(evidence.decision, AudioOnsetDecision::TrustedCallerOnset);
        assert!(evidence.echo_margin_db.is_some_and(|margin| margin >= 3.0));
    }

    #[test]
    fn audio_barge_in_classifier_fails_safe_on_short_reference() {
        let config = audio_classifier_test_config();
        let format = audio_classifier_test_format();
        let inbound = vec![1_500i16; 160];
        let mut state = AudioBargeInEvidenceState::default();

        let evidence = state.classify_active_playback_window(ActivePlaybackWindow {
            config: &config,
            playback_ref: &ActiveSpeechPlaybackRef {
                playback_id: "short-playback".to_string(),
                playback_epoch: 1,
            },
            format: &format,
            samples: &inbound,
            stats: &sample_stats(&inbound),
            speech: &SpeechQualityConfig::default(),
            outbound_reference: &[1_500i16; 10],
            transport_invalidation: None,
            frame_duration_ms: 20,
            now: Instant::now(),
        });

        assert_eq!(evidence.decision, AudioOnsetDecision::Unavailable);
        assert_eq!(
            evidence.invalidation,
            Some(AudioEvidenceInvalidation::ShortReference)
        );
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
        {
            let mut guard = state.write().await;
            guard.quality.config.set_tts_max_text_chunk_chars(40);
            guard.quality.config.set_tts_prebuffer_chunks(2);
            let config_id = guard.quality.config.config_id();
            guard.quality.config_id = config_id;
        }
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
            "Hello world. Second sentence blocks here.".to_string(),
            "test say",
        )
        .await
        .expect("speech should start");

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if media_registry
                    .active_speech_playback_id(&gateway_call_id)
                    .await
                    .is_none()
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("failed prebuffered chunk should release active playback");

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
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new(
            "tts_test",
            vec![1; 160],
        )))
        .await
        .expect("queue first frame");
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new(
            "tts_test",
            vec![2; 160],
        )))
        .await
        .expect("queue second frame");
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new(
            "tts_test",
            vec![3; 160],
        )))
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
            OutboundMediaCommand::Clear { playback_id, .. } => {
                assert_eq!(playback_id, "tts_second")
            }
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
        tx.send(OutboundMediaCommand::Frame(OutboundMediaFrame::new(
            "tts_test",
            vec![PCMU_SILENCE_BYTE; 160],
        )))
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
            guard.start_tts_job(
                &gateway_call_id,
                "tts_test".to_string(),
                LiveTtsBackend::default(),
                "hello",
            );
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
        assert_eq!(call.transcripts.len(), 2 + finish_pad_silence_frames());
        assert_eq!(call.transcripts[0].text, "received 16000 samples");
        let expected_final_samples =
            16_000 + (16_000 * VoiceQualityConfig::default().asr.finish_pad_ms / 1_000);
        let expected_final_text = format!("received {expected_final_samples} samples");
        assert_eq!(
            call.transcripts.last().map(|event| event.text.as_str()),
            Some(expected_final_text.as_str())
        );
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
        outbound.observe_frame(None, 4);
        outbound.observe_frame(Some(20), 4);
        outbound.observe_frame(Some(50), 2);
        outbound.observe_underrun();
        outbound.observe_append_starvation();
        outbound.observe_post_mark_wait();
        outbound.observe_pre_audio_wait();
        let payload = outbound.rollup_payload(Some("tts_test"));
        assert_eq!(payload["playback_id"], "tts_test");
        assert_eq!(payload["frames_sent"], 3);
        assert_eq!(payload["underrun_count"], 1);
        assert_eq!(payload["append_starvation_ticks"], 1);
        assert_eq!(payload["append_starvation_ms_estimate"], 20);
        assert_eq!(payload["post_mark_wait_ticks"], 1);
        assert_eq!(payload["post_mark_wait_ms_estimate"], 20);
        assert_eq!(payload["pre_audio_wait_ticks"], 1);
        assert_eq!(payload["pre_audio_wait_ms_estimate"], 20);
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
            continue_decision(true, CallerSpeechState::Speaking)
        );
        assert_eq!(
            gate.accept(2, "stream-1", 20, &silence, &quality),
            continue_decision(false, CallerSpeechState::EndpointCandidate)
        );
        match gate.accept(3, "stream-1", 20, &silence, &quality) {
            AsrFrameDecision::Finalize {
                trailing_silence_ms,
                endpoint_wait_started_at,
                speech_to_low_energy,
                endpoint_gate,
            } => {
                assert_eq!(trailing_silence_ms, 40);
                assert!(endpoint_wait_started_at.is_some());
                assert!(speech_to_low_energy.is_some());
                assert_eq!(endpoint_gate.suppressed_tail_frames, 1);
                assert_eq!(endpoint_gate.low_energy_run_ms, 40);
                assert_eq!(endpoint_gate.endpoint_frame_peak, 0);
                assert_eq!(endpoint_gate.last_speech_peak, Some(4_000));
            }
            other => panic!("expected finalize, got {other:?}"),
        }
    }

    #[test]
    fn asr_gate_resets_tail_suppression_at_utterance_boundary() {
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
            continue_decision(true, CallerSpeechState::Speaking)
        );
        assert_eq!(
            gate.accept(2, "stream-1", 20, &silence, &quality),
            continue_decision(false, CallerSpeechState::EndpointCandidate)
        );
        match gate.accept(3, "stream-1", 20, &silence, &quality) {
            AsrFrameDecision::Finalize { endpoint_gate, .. } => {
                assert_eq!(endpoint_gate.suppressed_tail_frames, 1);
            }
            other => panic!("expected first finalize, got {other:?}"),
        }

        gate.wait_for_next_speech();

        assert_eq!(
            gate.accept(4, "stream-1", 20, &speech, &quality),
            continue_decision(true, CallerSpeechState::Speaking)
        );
        assert_eq!(
            gate.accept(5, "stream-1", 20, &silence, &quality),
            continue_decision(false, CallerSpeechState::EndpointCandidate)
        );
        match gate.accept(6, "stream-1", 20, &silence, &quality) {
            AsrFrameDecision::Finalize { endpoint_gate, .. } => {
                assert_eq!(endpoint_gate.suppressed_tail_frames, 1);
            }
            other => panic!("expected second finalize, got {other:?}"),
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
            continue_decision(true, CallerSpeechState::Speaking)
        );
        for frame_index in 2..=10 {
            assert_eq!(
                gate.accept(frame_index, "stream-1", 20, &silence, &quality),
                continue_decision(false, CallerSpeechState::EndpointCandidate)
            );
        }
        assert_eq!(
            gate.accept(11, "stream-1", 20, &speech, &quality),
            continue_decision(true, CallerSpeechState::Speaking)
        );
        assert_eq!(
            gate.accept(12, "stream-1", 20, &speech, &quality),
            continue_decision(false, CallerSpeechState::Speaking)
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

        assert_eq!(
            counting_asr.ingests(),
            1 + endpoint_frames + finish_pad_silence_frames()
        );
        assert_eq!(counting_asr.finishes(), 1);
    }

    #[tokio::test]
    async fn finish_asr_session_coalesces_pad_and_finish_finals_into_one_caller_turn() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.session = Some(Box::new(PadFinishFinalAsrSession));

        finish_asr_session(
            &state,
            &mut media_state,
            Some(gateway_call_id.as_str()),
            Some("stream-1".to_string()),
            Some(&text_calls),
        )
        .await
        .expect("finish should record and forward transcripts");

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "pad final finish final"
        ));
        assert!(text_rx.try_recv().is_err());
    }

    #[test]
    fn final_reconciliation_prefix_matching_ignores_case_and_final_punctuation() {
        let reconciliation = reconcile_final_text("Hello wor.", Some("hello world"));

        assert_eq!(
            reconciliation.selected_source,
            FinalTextSource::LatestPartialExtension
        );
        assert_eq!(reconciliation.selected_text, "hello world");
    }

    #[tokio::test]
    async fn final_reconciliation_keeps_selected_partial_update_metadata() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let update_with_confidence = |text: &str,
                                      confidence: Option<f32>,
                                      final_segment: bool|
         -> motlie_model::TranscriptionUpdate {
            motlie_model::TranscriptionUpdate {
                segments: vec![motlie_model::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: text.to_string(),
                    confidence,
                    final_segment,
                }],
            }
        };
        let events = vec![
            AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                text: "hello world".to_string(),
                update: update_with_confidence("hello world", Some(0.84), false),
            }),
            AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "Hello wor.".to_string(),
                update: update_with_confidence("Hello wor.", Some(0.12), true),
            }),
        ];

        let reconciled = reconcile_asr_final_events(&state, "call-1", None, None, events).await;

        match &reconciled[1].event {
            TranscriptEvent::Final { text, update } => {
                assert_eq!(text, "hello world");
                let segment = update.segments.first().expect("segment exists");
                assert_eq!(segment.confidence, Some(0.84));
                assert!(segment.final_segment);
            }
            TranscriptEvent::Partial { .. } => panic!("expected final event"),
        }
    }

    #[tokio::test]
    async fn finish_asr_session_uses_latest_partial_when_final_is_strict_prefix() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.session = Some(Box::new(PartialThenFinalAsrSession::new(
            "hello world",
            "Hello wor.",
        )));

        finish_asr_session(
            &state,
            &mut media_state,
            Some(gateway_call_id.as_str()),
            Some("stream-1".to_string()),
            Some(&text_calls),
        )
        .await
        .expect("finish should record and forward transcripts");

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "hello world"
        ));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.final_transcript, "hello world");
        assert_eq!(call.current_partial, None);
    }

    #[tokio::test]
    async fn finish_asr_session_ignores_stale_call_partial_for_final_only_next_utterance() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        state.write().await.add_transcript(
            &gateway_call_id,
            TranscriptKind::Partial,
            "hello world".to_string(),
        );
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.session = Some(Box::new(FinalOnlyAsrSession::new("hello")));

        finish_asr_session(
            &state,
            &mut media_state,
            Some(gateway_call_id.as_str()),
            Some("stream-1".to_string()),
            Some(&text_calls),
        )
        .await
        .expect("finish should record and forward transcripts");

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "hello"
        ));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.final_transcript, "hello");
        assert_eq!(call.current_partial, None);
    }

    #[tokio::test]
    async fn finish_asr_session_keeps_final_when_latest_partial_diverges() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.session = Some(Box::new(PartialThenFinalAsrSession::new(
            "endpointing seems to be a real trouble",
            "endpointing seems to be a real cha",
        )));

        finish_asr_session(
            &state,
            &mut media_state,
            Some(gateway_call_id.as_str()),
            Some("stream-1".to_string()),
            Some(&text_calls),
        )
        .await
        .expect("finish should record and forward transcripts");

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "endpointing seems to be a real cha"
        ));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.final_transcript, "endpointing seems to be a real cha");
        assert_eq!(call.current_partial, None);
    }

    #[tokio::test]
    async fn final_settle_holds_and_merges_fragment_final() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.quality_config.endpoint.final_settle_ms = 5_000;

        let first_outcome = record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "there is also a".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        assert!(!first_outcome.reset_requested);
        assert!(first_outcome.conversation_events.is_empty());
        assert!(media_state.pending_final.is_some());
        assert!(text_rx.try_recv().is_err());

        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "behavior".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "there is also a behavior"
        ));
        assert!(media_state.pending_final.is_none());
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.final_transcript, "there is also a behavior");
    }

    #[tokio::test]
    async fn final_settle_holds_and_merges_leading_fragment_final() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.quality_config.endpoint.final_settle_ms = 5_000;

        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "Whereas I said".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        assert!(media_state.pending_final.is_some());
        assert!(text_rx.try_recv().is_err());

        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "that the delay felt too long".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "Whereas I said that the delay felt too long"
        ));
        assert!(media_state.pending_final.is_none());
    }

    #[tokio::test]
    async fn final_settle_holds_and_merges_also_tail_final() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.quality_config.endpoint.final_settle_ms = 5_000;

        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "I also".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        assert!(media_state.pending_final.is_some());
        assert!(text_rx.try_recv().is_err());

        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "noticed the missing section".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "I also noticed the missing section"
        ));
        assert!(media_state.pending_final.is_none());
    }

    #[tokio::test]
    async fn final_settle_flushes_fragment_without_continuation() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.quality_config.endpoint.final_settle_ms = 1;

        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "there is also a".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                CallerSpeechState::Finalizing,
            ),
        )
        .await;

        assert!(media_state.pending_final.is_some());
        time::sleep(Duration::from_millis(5)).await;
        record_and_forward_asr_events(
            &state,
            &mut media_state,
            &gateway_call_id,
            Some("stream-1"),
            Some(&text_calls),
            None,
            ForwardAsrEvents::new(Vec::new(), CallerSpeechState::Finalizing),
        )
        .await;

        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted after settle")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "there is also a"
        ));
        assert!(media_state.pending_final.is_none());
    }

    #[tokio::test]
    async fn opted_in_text_call_receives_partial_before_final_turn() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session_with_partials(gateway_call_id.clone(), true)
            .await;
        let (quality_tx, mut quality_rx) = tokio::sync::mpsc::channel(16);
        let quality_session = {
            let mut guard = state.write().await;
            guard.set_quality_event_sink(
                crate::quality::QualityEventSink::with_sender(quality_tx),
                None,
            );
            guard.start_quality_asr_session(&gateway_call_id, Some("stream-1"), "test")
        };
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };
        let outcome = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![
                AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                    text: "hello wor".to_string(),
                    update: motlie_model::TranscriptionUpdate {
                        segments: vec![motlie_model::TranscriptSegment {
                            start_ms: 0,
                            end_ms: 420,
                            text: "hello wor".to_string(),
                            confidence: Some(0.81),
                            final_segment: false,
                        }],
                    },
                }),
                AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                    text: "hello world".to_string(),
                    update: motlie_model::TranscriptionUpdate {
                        segments: vec![motlie_model::TranscriptSegment {
                            start_ms: 0,
                            end_ms: 520,
                            text: "hello world".to_string(),
                            confidence: Some(0.84),
                            final_segment: false,
                        }],
                    },
                }),
                AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: "hello world".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                }),
            ],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: Some(&quality_session),
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;

        assert!(!outcome.reset_requested);
        assert_eq!(outcome.conversation_events.len(), 3);
        let first_partial = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("first caller.partial should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            first_partial,
            GatewayTextFrame::CallerPartial {
                utterance_id,
                text,
                confidence: Some(confidence),
                stability: None,
                speech_state: CallerSpeechState::Speaking,
                reply_allowed: false,
                ..
            } if utterance_id == quality_session.utterance_id
                && text == "hello wor"
                && (confidence - 0.81).abs() < f32::EPSILON
        ));
        let second_partial = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("second caller.partial should be emitted")
            .expect("text-call session should stay open");
        let second_stability = match second_partial {
            GatewayTextFrame::CallerPartial {
                utterance_id,
                text,
                confidence: Some(confidence),
                stability: Some(stability),
                speech_state: CallerSpeechState::Speaking,
                reply_allowed: false,
                ..
            } if utterance_id == quality_session.utterance_id
                && text == "hello world"
                && (confidence - 0.84).abs() < f32::EPSILON =>
            {
                stability
            }
            other => panic!("unexpected second partial frame: {other:?}"),
        };
        assert!(second_stability > 0.0);
        let final_turn = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            final_turn,
            GatewayTextFrame::CallerTurn {
                utterance_id: Some(utterance_id),
                text,
                ..
            } if utterance_id == quality_session.utterance_id && text == "hello world"
        ));

        let mut quality_events = Vec::new();
        while quality_events.len() < 5 {
            match time::timeout(Duration::from_millis(100), quality_rx.recv()).await {
                Ok(Some(event)) => quality_events.push(event),
                _ => break,
            }
        }
        let partial_events = quality_events
            .iter()
            .filter(|event| event.event == "text_call.caller_partial.sent")
            .collect::<Vec<_>>();
        assert_eq!(partial_events.len(), 2);
        let scored_partial = partial_events[1];
        assert_eq!(
            scored_partial.payload["asr_session_id"],
            quality_session.asr_session_id
        );
        assert_eq!(
            scored_partial.payload["utterance_id"],
            quality_session.utterance_id
        );
        assert_eq!(scored_partial.payload["speech_state"], "speaking");
        let logged_confidence = scored_partial.payload["confidence"]
            .as_f64()
            .expect("confidence should be numeric");
        assert!((logged_confidence - 0.84).abs() < 0.000_001);
        let logged_stability = scored_partial.payload["stability"]
            .as_f64()
            .expect("stability should be numeric");
        assert!((logged_stability - second_stability as f64).abs() < 0.000_001);
        assert_eq!(scored_partial.payload["text_chars"], 11);
        assert_eq!(scored_partial.payload["transcript_text_included"], false);
        assert!(!scored_partial.payload.contains_key("text"));
    }

    #[tokio::test]
    async fn opted_in_text_call_receives_endpoint_and_finalizing_partial_states() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session_with_partials(gateway_call_id.clone(), true)
            .await;
        let quality_session = {
            let mut guard = state.write().await;
            guard.start_quality_asr_session(&gateway_call_id, Some("stream-1"), "test")
        };
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };
        for (text, partial_speech_state) in [
            ("endpoint soon", CallerSpeechState::EndpointCandidate),
            ("finalizing now", CallerSpeechState::Finalizing),
        ] {
            let outcome = record_transcript_events(
                &state,
                &gateway_call_id,
                vec![AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                    text: text.to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                })],
                TranscriptRecordContext {
                    stream_id: Some("stream-1"),
                    media_format: Some(&format),
                    capture: None,
                    text_calls: Some(&text_calls),
                    early_response: None,
                    quality_session: Some(&quality_session),
                    echo_config: None,
                    partial_speech_state,
                },
            )
            .await;
            assert!(!outcome.reset_requested);
            let partial = time::timeout(Duration::from_secs(1), text_rx.recv())
                .await
                .expect("caller.partial should be emitted")
                .expect("text-call session should stay open");
            assert!(matches!(
                partial,
                GatewayTextFrame::CallerPartial {
                    utterance_id,
                    text: emitted_text,
                    confidence: None,
                    speech_state,
                    reply_allowed: false,
                    ..
                } if utterance_id == quality_session.utterance_id
                    && emitted_text == text
                    && speech_state == partial_speech_state
            ));
        }
    }

    #[tokio::test]
    async fn assistant_echo_transcript_is_suppressed_before_text_call_forwarding() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let (quality_tx, mut quality_rx) = mpsc::channel(8);
        {
            let mut guard = state.write().await;
            guard.set_quality_event_sink(
                crate::quality::QualityEventSink::with_sender(quality_tx),
                None,
            );
            guard.start_tts_job(
                &gateway_call_id,
                "tts_echo".to_string(),
                LiveTtsBackend::Kokoro82m,
                "Why did the database administrator leave the party early? Too many tables.",
            );
            guard.mark_tts_frames_queued(&gateway_call_id, "tts_echo", 20);
        }
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };

        let outcome = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "in many tables".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: None,
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;

        assert!(!outcome.reset_requested);
        assert!(outcome.conversation_events.is_empty());
        assert!(
            time::timeout(Duration::from_millis(100), text_rx.recv())
                .await
                .is_err(),
            "assistant echo should not become caller.turn"
        );
        let mut suppression_event = None;
        while let Ok(Some(event)) = time::timeout(Duration::from_secs(1), quality_rx.recv()).await {
            if event.event == "transcript.suppressed" {
                suppression_event = Some(event);
                break;
            }
        }
        let suppression_event =
            suppression_event.expect("quality suppression event should be emitted");
        assert_eq!(suppression_event.event, "transcript.suppressed");
        assert_eq!(
            suppression_event.payload["suppression_reason"],
            json!("assistant_echo")
        );
        assert_eq!(
            suppression_event.payload["transcript_kind"],
            json!("transcript.final")
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert!(call.transcripts.is_empty());
        assert_eq!(call.final_transcript, "");
        assert_eq!(call.echo_suppressed_transcripts, 1);
    }

    #[test]
    fn quarantined_echo_prefix_trims_only_leading_tokens() {
        assert_eq!(
            trim_quarantined_echo_prefix(
                "Gateway will begin. Stop now, please repeat this replacement sentence clearly",
                "gateway will begin",
            ),
            Some("Stop now, please repeat this replacement sentence clearly".to_string())
        );
        assert_eq!(
            trim_quarantined_echo_prefix(
                "Stop now, gateway will begin should remain when it is not a prefix",
                "gateway will begin",
            ),
            None
        );
    }

    #[tokio::test]
    async fn mixed_echo_prefix_final_is_trimmed_before_forwarding() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let (quality_tx, mut quality_rx) = mpsc::channel(16);
        {
            let mut guard = state.write().await;
            guard.set_quality_event_sink(
                crate::quality::QualityEventSink::with_sender(quality_tx),
                None,
            );
            guard.start_tts_job(
                &gateway_call_id,
                "tts_echo".to_string(),
                LiveTtsBackend::Kokoro82m,
                "The gateway will begin repeating this long sentence so I can interrupt it during playback.",
            );
            guard.mark_tts_frames_queued(&gateway_call_id, "tts_echo", 20);
        }
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };

        let suppressed = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                text: "Gateway will begin".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: None,
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;
        assert!(suppressed.conversation_events.is_empty());

        let outcome = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text:
                    "Gateway will begin. Stop now, please repeat this replacement sentence clearly"
                        .to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: None,
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;

        assert!(!outcome.reset_requested);
        assert_eq!(outcome.conversation_events.len(), 1);
        assert_eq!(
            outcome.conversation_events[0].event.text(),
            "Stop now, please repeat this replacement sentence clearly"
        );
        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. }
                if text == "Stop now, please repeat this replacement sentence clearly"
        ));

        let mut saw_suppressed = false;
        let mut saw_trimmed = false;
        while let Ok(Some(event)) =
            time::timeout(Duration::from_millis(100), quality_rx.recv()).await
        {
            if event.event == "transcript.suppressed" {
                saw_suppressed = true;
            }
            if event.event == "voice.span.finished"
                && event.payload["span"] == json!("conversation.committed_echo_prefix")
                && event.payload["action"] == json!("trimmed_prefix")
            {
                saw_trimmed = true;
            }
            if saw_suppressed && saw_trimmed {
                break;
            }
        }
        assert!(
            saw_suppressed,
            "assistant echo suppression event should be emitted"
        );
        assert!(saw_trimmed, "committed prefix trim span should be emitted");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(
            call.final_transcript,
            "Stop now, please repeat this replacement sentence clearly"
        );
        assert_eq!(call.echo_suppressed_transcripts, 1);
        assert!(call.last_echo_suppressed_text.is_none());
    }

    #[tokio::test]
    async fn same_source_provisional_tts_does_not_suppress_final_transcript() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let quality_session = {
            let mut guard = state.write().await;
            guard.start_quality_asr_session(&gateway_call_id, Some("stream-1"), "test")
        };
        let text = "The operator should ask for the patient age location and callback number";
        {
            let mut guard = state.write().await;
            guard.start_tts_job_with_linkage(
                &gateway_call_id,
                "tts_provisional".to_string(),
                LiveTtsBackend::Kokoro82m,
                text,
                crate::operator::state::QualityPlaybackLinkage {
                    turn_id: Some("pt_same".to_string()),
                    coalesced_turn_ids: Vec::new(),
                    source_asr_session_ids: vec![quality_session.asr_session_id.clone()],
                    source_utterance_ids: vec![quality_session.utterance_id.clone()],
                    source_label: "early response".to_string(),
                    metadata: crate::operator::state::QualityPlaybackMetadata::default(),
                },
                None,
            );
            guard.mark_tts_frames_queued(&gateway_call_id, "tts_provisional", 20);
        }
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };

        let outcome = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: text.to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: Some(&quality_session),
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;

        assert!(!outcome.reset_requested);
        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text: emitted, .. } if emitted == text
        ));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.final_transcript, text);
        assert_eq!(call.echo_suppressed_transcripts, 0);
    }

    #[tokio::test]
    async fn echo_suppression_uses_asr_session_config_snapshot() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        let echo_snapshot = {
            let mut guard = state.write().await;
            guard.start_tts_job(
                &gateway_call_id,
                "tts_echo".to_string(),
                LiveTtsBackend::Kokoro82m,
                "Why did the database administrator leave the party early? Too many tables.",
            );
            guard.mark_tts_frames_queued(&gateway_call_id, "tts_echo", 20);
            let snapshot = guard.quality.config.echo_suppression.clone();
            guard.quality.config.set_echo_suppression_enabled(false);
            snapshot
        };
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };

        let outcome = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "in many tables".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: None,
                echo_config: Some(&echo_snapshot),
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;

        assert!(!outcome.reset_requested);
        assert!(
            time::timeout(Duration::from_millis(100), text_rx.recv())
                .await
                .is_err(),
            "in-flight ASR session snapshot should still suppress echo"
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.echo_suppressed_transcripts, 1);
        assert!(!guard.quality.config.echo_suppression.enabled);
    }

    #[tokio::test]
    async fn echo_suppression_off_allows_text_call_forwarding() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let text_calls = SharedTextCallRegistry::default();
        let mut text_rx = text_calls
            .insert_test_session(gateway_call_id.clone())
            .await;
        {
            let mut guard = state.write().await;
            guard.quality.config.set_echo_suppression_enabled(false);
            guard.start_tts_job(
                &gateway_call_id,
                "tts_echo".to_string(),
                LiveTtsBackend::Kokoro82m,
                "Why did the database administrator leave the party early? Too many tables.",
            );
            guard.mark_tts_frames_queued(&gateway_call_id, "tts_echo", 20);
        }
        let format = MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        };

        let outcome = record_transcript_events(
            &state,
            &gateway_call_id,
            vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "in many tables".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })],
            TranscriptRecordContext {
                stream_id: Some("stream-1"),
                media_format: Some(&format),
                capture: None,
                text_calls: Some(&text_calls),
                early_response: None,
                quality_session: None,
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;

        assert!(!outcome.reset_requested);
        let frame = time::timeout(Duration::from_secs(1), text_rx.recv())
            .await
            .expect("caller.turn should be emitted")
            .expect("text-call session should stay open");
        assert!(matches!(
            frame,
            GatewayTextFrame::CallerTurn { text, .. } if text == "in many tables"
        ));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.echo_suppressed_transcripts, 0);
    }

    #[tokio::test]
    async fn onset_echo_guard_only_defers_likely_audible_assistant_echo() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let media_registry = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(8);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        media_registry
            .start_speech(
                &gateway_call_id,
                "tts_echo".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("active playback should register");
        {
            let mut guard = state.write().await;
            guard.start_tts_job(
                &gateway_call_id,
                "tts_echo".to_string(),
                LiveTtsBackend::Piper,
                "This assistant line may echo through the handset.",
            );
            guard.mark_tts_frames_queued(&gateway_call_id, "tts_echo", 20);
        }
        let mut media_state = MediaSocketState::with_media_registry(media_registry.clone());

        let pre_audio = speech_onset_echo_decision(
            &state,
            media_state.media_registry.clone(),
            media_state.first_frame_sent_playbacks.clone(),
            media_state.quality_config.clone(),
            None,
            &gateway_call_id,
        )
        .await;
        assert!(!pre_audio.defer_to_partial);
        assert_eq!(pre_audio.playback_id.as_deref(), Some("tts_echo"));
        assert_eq!(pre_audio.reason, "pre_audio_playback");

        media_state
            .first_frame_sent_playbacks
            .insert("tts_echo".to_string());
        let audible_echo = speech_onset_echo_decision(
            &state,
            media_state.media_registry.clone(),
            media_state.first_frame_sent_playbacks.clone(),
            media_state.quality_config.clone(),
            None,
            &gateway_call_id,
        )
        .await;
        assert!(audible_echo.defer_to_partial);
        assert_eq!(audible_echo.playback_id.as_deref(), Some("tts_echo"));
        assert_eq!(audible_echo.reason, "likely_assistant_echo");

        media_state.quality_config.barge_in.onset_during_playback =
            OnsetDuringPlaybackPolicy::Trust;
        let trusted = speech_onset_echo_decision(
            &state,
            media_state.media_registry.clone(),
            media_state.first_frame_sent_playbacks.clone(),
            media_state.quality_config.clone(),
            None,
            &gateway_call_id,
        )
        .await;
        assert!(!trusted.defer_to_partial);
        assert_eq!(trusted.reason, "policy_trust");
    }

    #[tokio::test]
    async fn finish_asr_session_emits_local_turn_join_when_text_call_session_absent() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let quality_session = {
            let mut guard = state.write().await;
            guard.set_quality_event_sink(crate::quality::QualityEventSink::with_sender(tx), None);
            guard.start_quality_asr_session(&gateway_call_id, Some("stream-1"), "test")
        };
        let text_calls = SharedTextCallRegistry::default();
        let mut media_state = MediaSocketState::new();
        media_state.media_format = Some(MediaFormat {
            encoding: "L16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
        });
        media_state.active_quality_asr = Some(quality_session.clone());
        media_state.session = Some(Box::new(FinalOnlyAsrSession::new("hello local join")));

        finish_asr_session(
            &state,
            &mut media_state,
            Some(gateway_call_id.as_str()),
            Some("stream-1".to_string()),
            Some(&text_calls),
        )
        .await
        .expect("finish should emit local quality joins");

        let mut events = Vec::new();
        while events.len() < 8 {
            match time::timeout(Duration::from_millis(100), rx.recv()).await {
                Ok(Some(event)) => events.push(event),
                _ => break,
            }
        }
        let caller_turn = events
            .iter()
            .find(|event| event.event == "text_call.caller_turn.sent")
            .expect("local caller turn event should be emitted");
        let mapped = events
            .iter()
            .find(|event| event.event == "asr.turn_mapped")
            .expect("ASR turn mapping event should be emitted");
        assert_eq!(caller_turn.payload["turn_id"], mapped.payload["turn_id"]);
        assert_eq!(
            mapped.payload["asr_session_id"],
            quality_session.asr_session_id
        );
        assert_eq!(mapped.payload["utterance_id"], quality_session.utterance_id);
        assert_eq!(mapped.payload["caller_turn_sent"], true);
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
        assert_eq!(
            counting_asr.ingests(),
            2 + endpoint_frames + finish_pad_silence_frames()
        );

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

        let outcome = record_transcript_events(
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
                early_response: None,
                quality_session: None,
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;
        assert!(outcome.reset_requested);
        assert_eq!(outcome.conversation_events.len(), 1);
        assert!(outcome.conversation_events[0].turn_id.is_some());

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.transcripts.len(), 1);
        assert_eq!(call.transcripts[0].text, "hello there");
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

        let outcome = record_transcript_events(
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
                early_response: None,
                quality_session: None,
                echo_config: None,
                partial_speech_state: CallerSpeechState::Speaking,
            },
        )
        .await;
        assert!(!outcome.reset_requested);
        assert_eq!(outcome.conversation_events.len(), 1);
        assert!(outcome.conversation_events[0].turn_id.is_some());

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

    fn finish_pad_silence_frames() -> usize {
        usize::from(VoiceQualityConfig::default().asr.finish_pad_ms > 0)
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

    struct FinalOnlyAsrSession {
        final_text: String,
    }

    impl FinalOnlyAsrSession {
        fn new(final_text: &str) -> Self {
            Self {
                final_text: final_text.to_string(),
            }
        }
    }

    #[async_trait]
    impl InboundAsrSession for FinalOnlyAsrSession {
        async fn ingest(
            &mut self,
            _audio: AudioBuf<i16, 16_000, Mono>,
        ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            Ok(Vec::new())
        }

        async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            Ok(vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: self.final_text,
                update: motlie_model::TranscriptionUpdate::default(),
            })])
        }
    }

    struct PartialThenFinalAsrSession {
        partial_text: String,
        final_text: String,
    }

    impl PartialThenFinalAsrSession {
        fn new(partial_text: &str, final_text: &str) -> Self {
            Self {
                partial_text: partial_text.to_string(),
                final_text: final_text.to_string(),
            }
        }
    }

    #[async_trait]
    impl InboundAsrSession for PartialThenFinalAsrSession {
        async fn ingest(
            &mut self,
            _audio: AudioBuf<i16, 16_000, Mono>,
        ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            Ok(Vec::new())
        }

        async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            Ok(vec![
                AsrTranscriptEvent::emit(TranscriptEvent::Partial {
                    text: self.partial_text,
                    update: motlie_model::TranscriptionUpdate::default(),
                }),
                AsrTranscriptEvent::emit(TranscriptEvent::Final {
                    text: self.final_text,
                    update: motlie_model::TranscriptionUpdate::default(),
                }),
            ])
        }
    }

    struct PadFinishFinalAsrSession;

    #[async_trait]
    impl InboundAsrSession for PadFinishFinalAsrSession {
        async fn ingest(
            &mut self,
            _audio: AudioBuf<i16, 16_000, Mono>,
        ) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            Ok(vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "pad final".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })])
        }

        async fn finish(self: Box<Self>) -> anyhow::Result<Vec<AsrTranscriptEvent>> {
            Ok(vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
                text: "finish final".to_string(),
                update: motlie_model::TranscriptionUpdate::default(),
            })])
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
