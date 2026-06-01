use anyhow::{bail, Context};
use axum::extract::ws::{Message, WebSocket};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::time::Duration;

use futures_util::StreamExt;
use motlie_model::typed::{AudioBuf, Mono};
use motlie_voice::app::TranscriptEvent;
use motlie_voice::codec::{g711, l16};
use motlie_voice::pipeline::reorder::{SequencedFrame, SequencedFrameReorder};
use motlie_voice::pipeline::resample::{resample_i16_mono, WindowedSincResampler};
use motlie_voice::VoiceError;
use serde::Deserialize;
use tokio::time::{self, MissedTickBehavior};

use crate::adapter::{InboundAsrSession, SharedAsrFactory};
use crate::operator::state::{
    CallStatus, LogLevel, MediaMetadata, SharedState, StreamAttachOutcome, TranscriptKind,
};

const SPEECH_RMS_THRESHOLD: f32 = 180.0;
const SPEECH_PEAK_THRESHOLD: i16 = 900;
const LOW_ENERGY_HANGOVER_FRAMES: usize = 75;
const REPEATED_TOKEN_RUN_THRESHOLD: usize = 16;
const REPEATED_Q_RUN_THRESHOLD: usize = 8;
const PCMU_SILENCE_FRAME: [u8; 160] = [0xff; 160];
const SILENCE_KEEPALIVE_INTERVAL: Duration = Duration::from_millis(20);

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

struct MediaSocketState {
    session: Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<String>,
    media_format: Option<MediaFormat>,
    reorder: SequencedFrameReorder<EncodedMediaFrame>,
    decoded_frame_count: usize,
    asr_gate: AsrGate,
    silence_keepalive: bool,
    silence_keepalive_frames: usize,
}

impl MediaSocketState {
    fn new() -> Self {
        Self {
            session: None,
            gateway_call_id: None,
            media_format: None,
            reorder: SequencedFrameReorder::new_lazily(32),
            decoded_frame_count: 0,
            asr_gate: AsrGate::default(),
            silence_keepalive: false,
            silence_keepalive_frames: 0,
        }
    }
}

pub async fn handle_socket(mut socket: WebSocket, state: SharedState, asr: SharedAsrFactory) {
    let mut media_state = MediaSocketState::new();
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
                        if let Err(error) = handle_text(&text, &state, &asr, &mut media_state).await {
                            log_media_error(&state, media_state.gateway_call_id.as_deref(), error).await;
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
                if let Err(error) = send_silence_keepalive(&mut socket, &mut media_state).await {
                    log_media_error(&state, media_state.gateway_call_id.as_deref(), error).await;
                    break;
                }
            }
        }
    }

    if let (Some(call_id), Some(asr_session)) = (
        media_state.gateway_call_id.as_deref(),
        media_state.session.take(),
    ) {
        match asr_session.finish().await {
            Ok(events) => {
                let _ = record_transcript_events(
                    &state,
                    call_id,
                    None,
                    media_state.media_format.as_ref(),
                    events,
                )
                .await;
            }
            Err(error) => log_media_error(&state, Some(call_id), error).await,
        }
        let mut guard = state.write().await;
        if let Some(call) = guard.calls.get_mut(call_id) {
            call.status = CallStatus::Ended;
            call.push_timeline("media websocket closed");
        }
    }
}

async fn send_silence_keepalive(
    socket: &mut WebSocket,
    media_state: &mut MediaSocketState,
) -> anyhow::Result<()> {
    socket
        .send(Message::Text(pcmu_silence_keepalive_message().into()))
        .await
        .context("send PCMU silence keepalive to Telnyx")?;

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

fn pcmu_silence_keepalive_message() -> String {
    let payload = STANDARD.encode(PCMU_SILENCE_FRAME);
    serde_json::json!({
        "event": "media",
        "media": {
            "payload": payload
        }
    })
    .to_string()
}

async fn handle_text(
    text: &str,
    state: &SharedState,
    asr: &SharedAsrFactory,
    media_state: &mut MediaSocketState,
) -> anyhow::Result<()> {
    let discriminator: EventDiscriminator =
        serde_json::from_str(text).context("parse Telnyx media event discriminator")?;
    match discriminator.event.as_str() {
        "connected" => Ok(()),
        "start" => {
            let event: StartEvent = serde_json::from_str(text).context("parse start event")?;
            let format = map_media_format(event.start.media_format.as_ref());
            validate_media_format(&format)?;
            let Some(call_id) = register_start(state, &event, &format).await else {
                return Ok(());
            };
            media_state.media_format = Some(format);
            media_state.gateway_call_id = Some(call_id);
            media_state.session = Some(asr.open_session().await?);
            media_state.silence_keepalive = true;
            tracing::info!(
                gateway_call_id = media_state.gateway_call_id.as_deref(),
                stream_id = event.stream_id,
                "media.silence_keepalive.started"
            );
            Ok(())
        }
        "media" => {
            let event: MediaEvent = serde_json::from_str(text).context("parse media event")?;
            let sequence = event
                .media
                .chunk
                .parse::<u64>()
                .context("parse Telnyx media.chunk as sequence")?;
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
                )
                .await?;
            }
            Ok(())
        }
        "stop" => {
            let event: StopEvent = serde_json::from_str(text).context("parse stop event")?;
            media_state.silence_keepalive = false;
            finish_stream(
                state,
                &mut media_state.session,
                media_state.gateway_call_id.as_deref(),
                event.stream_id,
                media_state.media_format.as_ref(),
            )
            .await
        }
        "mark" | "clear" | "dtmf" => Ok(()),
        "error" => bail!("Telnyx media error event: {text}"),
        other => bail!("unsupported Telnyx media event {other}"),
    }
}

async fn register_start(
    state: &SharedState,
    event: &StartEvent,
    format: &MediaFormat,
) -> Option<String> {
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
        StreamAttachOutcome::Attached { gateway_call_id } => {
            guard.log(
                LogLevel::Info,
                format!(
                    "media started for {gateway_call_id}: {} {} Hz {}ch",
                    format.encoding, format.sample_rate_hz, format.channels
                ),
            );
            tracing::info!(
                gateway_call_id,
                call_control_id = event.start.call_control_id,
                call_session_id = event.start.call_session_id.as_deref(),
                stream_id = event.stream_id,
                codec = format.encoding,
                sample_rate_hz = format.sample_rate_hz,
                channels = format.channels,
                "media.started"
            );
            Some(gateway_call_id)
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
    asr: &SharedAsrFactory,
    media_state: &mut MediaSocketState,
    stream_id: &str,
    frame: EncodedMediaFrame,
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
    let stats = sample_stats(&samples);
    log_decoded_frame_stats(
        media_state.decoded_frame_count,
        &format,
        stream_id,
        &frame,
        &stats,
        samples.len(),
    );
    match media_state
        .asr_gate
        .accept(media_state.decoded_frame_count, stream_id, &stats)
    {
        AsrFrameDecision::Suppress => return Ok(()),
        AsrFrameDecision::Continue => {}
    }
    if media_state.session.is_none() {
        open_asr_session(
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

    let Some(session) = media_state.session.as_mut() else {
        bail!("ASR session unavailable after reopen");
    };
    let events = session
        .ingest(AudioBuf::<i16, 16_000, Mono>::new(samples))
        .await?;
    if record_transcript_events(
        state,
        &gateway_call_id,
        Some(stream_id),
        Some(&format),
        events,
    )
    .await
    {
        media_state.session = None;
        media_state.asr_gate.wait_for_next_speech();
        open_asr_session(
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
    asr: &SharedAsrFactory,
    media_state: &mut MediaSocketState,
    gateway_call_id: &str,
    stream_id: &str,
    reason: &'static str,
) -> anyhow::Result<()> {
    media_state.session = Some(asr.open_session().await?);
    tracing::info!(gateway_call_id, stream_id, reason, "asr.session.opened");
    Ok(())
}

fn decode_payload(format: &MediaFormat, payload: &[u8]) -> anyhow::Result<Vec<i16>> {
    validate_media_format(format)?;
    match format.encoding.as_str() {
        "L16" => Ok(l16::decode_l16_be(payload)?),
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
    fn has_speech_energy(&self) -> bool {
        self.rms >= SPEECH_RMS_THRESHOLD || self.peak >= SPEECH_PEAK_THRESHOLD
    }
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
    low_energy_run: usize,
    suppressed_initial_frames: usize,
    suppressed_tail_frames: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AsrFrameDecision {
    Suppress,
    Continue,
}

impl AsrGate {
    fn accept(
        &mut self,
        frame_index: usize,
        stream_id: &str,
        stats: &SampleStats,
    ) -> AsrFrameDecision {
        if stats.has_speech_energy() {
            let was_started = self.speech_started;
            let resumed_after_tail = self.low_energy_run > LOW_ENERGY_HANGOVER_FRAMES;
            self.speech_started = true;
            self.low_energy_run = 0;
            if !was_started {
                tracing::info!(
                    stream_id,
                    frame_index,
                    suppressed_frames = self.suppressed_initial_frames,
                    peak = stats.peak,
                    rms = stats.rms,
                    "media.speech.detected"
                );
            } else if resumed_after_tail {
                tracing::info!(
                    stream_id,
                    frame_index,
                    suppressed_tail_frames = self.suppressed_tail_frames,
                    peak = stats.peak,
                    rms = stats.rms,
                    "media.speech.resumed"
                );
            }
            return AsrFrameDecision::Continue;
        }

        if self.speech_started {
            self.low_energy_run = self.low_energy_run.saturating_add(1);
            if self.low_energy_run <= LOW_ENERGY_HANGOVER_FRAMES {
                return AsrFrameDecision::Continue;
            }
            self.suppressed_tail_frames = self.suppressed_tail_frames.saturating_add(1);
            if self.suppressed_tail_frames <= 5 || self.suppressed_tail_frames.is_multiple_of(50) {
                tracing::debug!(
                    stream_id,
                    frame_index,
                    suppressed_tail_frames = self.suppressed_tail_frames,
                    low_energy_run = self.low_energy_run,
                    peak = stats.peak,
                    rms = stats.rms,
                    "media.frame.suppressed_low_energy_tail"
                );
            }
            return AsrFrameDecision::Suppress;
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
        self.low_energy_run = 0;
    }
}

fn looks_like_repeated_token_hallucination(text: &str) -> bool {
    let mut previous = None;
    let mut run = 0usize;
    let mut max_run = 0usize;
    let mut chars = 0usize;
    let mut q_count = 0usize;

    for ch in text.chars().filter(|ch| !ch.is_whitespace()) {
        chars += 1;
        if ch == 'Q' {
            q_count += 1;
        }
        if previous == Some(ch) {
            run += 1;
        } else {
            previous = Some(ch);
            run = 1;
        }
        max_run = max_run.max(run);
    }

    max_run >= REPEATED_TOKEN_RUN_THRESHOLD
        || (q_count >= REPEATED_Q_RUN_THRESHOLD
            && q_count.saturating_mul(3) >= chars.saturating_mul(2))
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
    session: &mut Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<&str>,
    stream_id: Option<String>,
    media_format: Option<&MediaFormat>,
) -> anyhow::Result<()> {
    finish_asr_session(state, session, gateway_call_id, stream_id, media_format).await?;
    if let Some(call_id) = gateway_call_id {
        let mut guard = state.write().await;
        if let Some(call) = guard.calls.get_mut(call_id) {
            call.status = CallStatus::Ended;
            call.push_timeline("media stream stopped");
        }
    }
    Ok(())
}

async fn finish_asr_session(
    state: &SharedState,
    session: &mut Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<&str>,
    stream_id: Option<String>,
    media_format: Option<&MediaFormat>,
) -> anyhow::Result<()> {
    if let (Some(call_id), Some(asr_session)) = (gateway_call_id, session.take()) {
        let events = asr_session.finish().await?;
        let _ =
            record_transcript_events(state, call_id, stream_id.as_deref(), media_format, events)
                .await;
    }
    Ok(())
}

async fn record_transcript_events(
    state: &SharedState,
    gateway_call_id: &str,
    stream_id: Option<&str>,
    media_format: Option<&MediaFormat>,
    events: Vec<TranscriptEvent>,
) -> bool {
    let mut guard = state.write().await;
    let mut suppressed_repeated_token = false;
    for event in events {
        let kind = if event.is_final() {
            TranscriptKind::Final
        } else {
            TranscriptKind::Partial
        };
        let kind_label = if event.is_final() {
            "transcript.final"
        } else {
            "transcript.partial"
        };
        let text = event.text().to_string();
        let call = guard.calls.get(gateway_call_id);
        let call_control_id = call.map(|call| call.ids.call_control_id.clone());
        let call_session_id = call.and_then(|call| call.ids.call_session_id.clone());
        let call_leg_id = call.and_then(|call| call.ids.call_leg_id.clone());
        let effective_stream_id = stream_id
            .map(str::to_string)
            .or_else(|| call.and_then(|call| call.ids.stream_id.clone()));
        let codec = media_format
            .map(|format| format.encoding.clone())
            .or_else(|| call.and_then(|call| call.media.encoding.clone()));
        let sample_rate_hz = media_format
            .map(|format| format.sample_rate_hz)
            .or_else(|| call.and_then(|call| call.media.sample_rate_hz));

        if looks_like_repeated_token_hallucination(&text) {
            suppressed_repeated_token = true;
            tracing::warn!(
                gateway_call_id,
                call_control_id = call_control_id.as_deref(),
                call_session_id = call_session_id.as_deref(),
                call_leg_id = call_leg_id.as_deref(),
                stream_id = effective_stream_id.as_deref(),
                codec = codec.as_deref(),
                sample_rate_hz,
                transcript_kind = kind_label,
                transcript_chars = text.chars().count(),
                transcript_preview = transcript_preview(&text),
                "transcript.suppressed_repeated_token"
            );
            continue;
        }

        guard.add_transcript(gateway_call_id, kind, text.clone());
        tracing::info!(
            gateway_call_id,
            call_control_id = call_control_id.as_deref(),
            call_session_id = call_session_id.as_deref(),
            call_leg_id = call_leg_id.as_deref(),
            stream_id = effective_stream_id.as_deref(),
            codec = codec.as_deref(),
            sample_rate_hz,
            transcript_kind = kind_label,
            transcript_text = text,
            "{kind_label}"
        );
    }
    suppressed_repeated_token
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

    use crate::adapter::{EchoAsrFactory, InboundAsrFactory};
    use crate::operator::state::{shared_state, CallStatus, TelnyxIds};

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
    fn pcmu_silence_keepalive_message_is_telnyx_media_json() {
        let message = pcmu_silence_keepalive_message();
        let value: serde_json::Value =
            serde_json::from_str(&message).expect("keepalive should be JSON");
        let payload = value["media"]["payload"]
            .as_str()
            .expect("payload should be a string");
        let decoded = STANDARD
            .decode(payload.as_bytes())
            .expect("payload should be base64");

        assert_eq!(value["event"], "media");
        assert_eq!(decoded, PCMU_SILENCE_FRAME);
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
                Some("+15550000001".to_string()),
                Some("+15550000002".to_string()),
                CallStatus::Answering,
            )
        };
        let asr: SharedAsrFactory = Arc::new(EchoAsrFactory);
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
    async fn low_energy_media_is_suppressed_until_speech() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let asr: SharedAsrFactory = Arc::new(EchoAsrFactory);
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

    #[tokio::test]
    async fn low_energy_media_is_suppressed_after_hangover() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let _gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr: SharedAsrFactory = counting_asr.clone();
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

        let silence = STANDARD.encode(l16_samples(320, 0));
        for sequence in 2..=(LOW_ENERGY_HANGOVER_FRAMES + 3) {
            handle_text(
                &media_event("stream-1", &sequence.to_string(), &silence),
                &state,
                &asr,
                &mut media_state,
            )
            .await
            .expect("silence should be accepted by transport");
        }

        assert_eq!(counting_asr.ingests(), 1 + LOW_ENERGY_HANGOVER_FRAMES);
    }

    #[tokio::test]
    async fn speech_resume_after_silence_keeps_asr_session() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let _gateway_call_id = seed_call(&state, "call-1", CallStatus::Answering).await;
        let counting_asr = Arc::new(CountingAsrFactory::default());
        let asr: SharedAsrFactory = counting_asr.clone();
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

        let silence = STANDARD.encode(l16_samples(320, 0));
        for sequence in 2..=(LOW_ENERGY_HANGOVER_FRAMES + 3) {
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

        handle_text(
            &media_event(
                "stream-1",
                &(LOW_ENERGY_HANGOVER_FRAMES + 4).to_string(),
                &speech,
            ),
            &state,
            &asr,
            &mut media_state,
        )
        .await
        .expect("resumed speech should continue ASR ingestion");

        assert_eq!(counting_asr.opens(), 1);
        assert_eq!(counting_asr.ingests(), 2 + LOW_ENERGY_HANGOVER_FRAMES);

        let guard = state.read().await;
        let call = guard.calls.get(&_gateway_call_id).expect("call exists");
        assert_eq!(call.status, CallStatus::MediaStarted);
    }

    #[tokio::test]
    async fn repeated_token_transcripts_are_suppressed_from_call_detail() {
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
            Some("stream-1"),
            Some(&format),
            vec![
                TranscriptEvent::Partial {
                    text: "MEQQQQQQQQQQQQQQQQQQQQQQQQQQQQ".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                TranscriptEvent::Final {
                    text: "hello there".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
            ],
        )
        .await;
        assert!(needs_reset);

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.transcripts.len(), 1);
        assert_eq!(call.transcripts[0].text, "hello there");
    }

    #[test]
    fn repeated_token_detector_ignores_normal_transcripts() {
        assert!(looks_like_repeated_token_hallucination(
            "MEQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"
        ));
        assert!(looks_like_repeated_token_hallucination("GOODQQQQQQQQ"));
        assert!(!looks_like_repeated_token_hallucination(
            "meet me at the front desk"
        ));
        assert!(!looks_like_repeated_token_hallucination("queue"));
    }

    #[tokio::test]
    async fn media_start_requires_operator_answer_gate() {
        for status in [CallStatus::IgnoredInbound, CallStatus::PendingInbound] {
            let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
            let gateway_call_id = seed_call(&state, "call-1", status).await;
            let counting_asr = Arc::new(CountingAsrFactory::default());
            let asr: SharedAsrFactory = counting_asr.clone();
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
        let asr: SharedAsrFactory = counting_asr.clone();
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
        let asr: SharedAsrFactory = counting_asr.clone();
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

    fn l16_samples(count: usize, sample: i16) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            bytes.extend_from_slice(&sample.to_be_bytes());
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
            Some("+15550000001".to_string()),
            Some("+15550000002".to_string()),
            status,
        )
    }

    #[derive(Default)]
    struct CountingAsrFactory {
        opens: AtomicUsize,
        ingests: Arc<AtomicUsize>,
    }

    impl CountingAsrFactory {
        fn opens(&self) -> usize {
            self.opens.load(Ordering::SeqCst)
        }

        fn ingests(&self) -> usize {
            self.ingests.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl InboundAsrFactory for CountingAsrFactory {
        async fn open_session(&self) -> anyhow::Result<Box<dyn InboundAsrSession>> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(CountingAsrSession {
                ingests: Arc::clone(&self.ingests),
            }))
        }
    }

    struct CountingAsrSession {
        ingests: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InboundAsrSession for CountingAsrSession {
        async fn ingest(
            &mut self,
            _audio: AudioBuf<i16, 16_000, Mono>,
        ) -> anyhow::Result<Vec<TranscriptEvent>> {
            self.ingests.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }

        async fn finish(self: Box<Self>) -> anyhow::Result<Vec<TranscriptEvent>> {
            Ok(Vec::new())
        }
    }
}
