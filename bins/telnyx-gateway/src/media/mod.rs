use anyhow::{bail, Context};
use axum::extract::ws::{Message, WebSocket};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures_util::StreamExt;
use motlie_model::typed::{AudioBuf, Mono};
use motlie_voice::app::TranscriptEvent;
use motlie_voice::codec::{g711, l16};
use motlie_voice::pipeline::reorder::{SequencedFrame, SequencedFrameReorder};
use motlie_voice::pipeline::resample::{resample_i16_mono, WindowedSincResampler};
use serde::Deserialize;

use crate::adapter::{InboundAsrSession, SharedAsrFactory};
use crate::operator::state::{CallStatus, LogLevel, MediaMetadata, SharedState, TranscriptKind};

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

pub async fn handle_socket(mut socket: WebSocket, state: SharedState, asr: SharedAsrFactory) {
    let mut session: Option<Box<dyn InboundAsrSession>> = None;
    let mut gateway_call_id: Option<String> = None;
    let mut media_format: Option<MediaFormat> = None;
    let mut reorder = SequencedFrameReorder::new(1, 32);

    while let Some(message) = socket.next().await {
        match message {
            Ok(Message::Text(text)) => {
                if let Err(error) = handle_text(
                    &text,
                    &state,
                    &asr,
                    &mut session,
                    &mut gateway_call_id,
                    &mut media_format,
                    &mut reorder,
                )
                .await
                {
                    log_media_error(&state, gateway_call_id.as_deref(), error).await;
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
            Err(error) => {
                log_media_error(
                    &state,
                    gateway_call_id.as_deref(),
                    anyhow::Error::from(error),
                )
                .await;
                break;
            }
        }
    }

    if let (Some(call_id), Some(asr_session)) = (gateway_call_id.as_deref(), session.take()) {
        match asr_session.finish().await {
            Ok(events) => record_transcript_events(&state, call_id, None, None, events).await,
            Err(error) => log_media_error(&state, Some(call_id), error).await,
        }
        let mut guard = state.write().await;
        if let Some(call) = guard.calls.get_mut(call_id) {
            call.status = CallStatus::Ended;
            call.push_timeline("media websocket closed");
        }
    }
}

async fn handle_text(
    text: &str,
    state: &SharedState,
    asr: &SharedAsrFactory,
    session: &mut Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: &mut Option<String>,
    media_format: &mut Option<MediaFormat>,
    reorder: &mut SequencedFrameReorder<EncodedMediaFrame>,
) -> anyhow::Result<()> {
    let discriminator: EventDiscriminator =
        serde_json::from_str(text).context("parse Telnyx media event discriminator")?;
    match discriminator.event.as_str() {
        "connected" => Ok(()),
        "start" => {
            let event: StartEvent = serde_json::from_str(text).context("parse start event")?;
            let format = map_media_format(event.start.media_format.as_ref());
            *media_format = Some(format.clone());
            *gateway_call_id = register_start(state, &event, &format).await;
            *session = Some(asr.open_session().await?);
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
            let ready = reorder.push(SequencedFrame {
                sequence,
                payload: EncodedMediaFrame {
                    payload,
                    track: event.media.track,
                },
            })?;
            for frame in ready {
                ingest_frame(
                    state,
                    session,
                    gateway_call_id.as_deref(),
                    media_format.as_ref(),
                    event.stream_id.as_str(),
                    frame.payload,
                )
                .await?;
            }
            Ok(())
        }
        "stop" => {
            let event: StopEvent = serde_json::from_str(text).context("parse stop event")?;
            finish_stream(state, session, gateway_call_id.as_deref(), event.stream_id).await
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
    if let Some(gateway_call_id) = gateway_call_id.as_deref() {
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
    } else {
        guard.log(
            LogLevel::Warn,
            format!(
                "media start for unknown call_control_id {}",
                event.start.call_control_id
            ),
        );
    }
    gateway_call_id
}

async fn ingest_frame(
    state: &SharedState,
    session: &mut Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<&str>,
    media_format: Option<&MediaFormat>,
    stream_id: &str,
    frame: EncodedMediaFrame,
) -> anyhow::Result<()> {
    let Some(session) = session.as_mut() else {
        bail!("media frame arrived before ASR session opened");
    };
    let Some(format) = media_format else {
        bail!("media frame arrived before media format was known");
    };
    let Some(gateway_call_id) = gateway_call_id else {
        bail!("media frame arrived before gateway call was known");
    };

    let mut samples = decode_payload(format, &frame.payload)?;
    if format.sample_rate_hz != 16_000 {
        samples = resample_i16_mono(
            &WindowedSincResampler::default(),
            &samples,
            format.sample_rate_hz,
            16_000,
        )?;
    }

    let events = session
        .ingest(AudioBuf::<i16, 16_000, Mono>::new(samples))
        .await?;
    record_transcript_events(
        state,
        gateway_call_id,
        Some(stream_id),
        Some(format),
        events,
    )
    .await;
    Ok(())
}

fn decode_payload(format: &MediaFormat, payload: &[u8]) -> anyhow::Result<Vec<i16>> {
    match format.encoding.as_str() {
        "L16" => Ok(l16::decode_l16_be(payload)?),
        "PCMU" => Ok(g711::decode_pcmu(payload)),
        "PCMA" => Ok(g711::decode_pcma(payload)),
        other => bail!("unsupported inbound media encoding {other}"),
    }
}

async fn finish_stream(
    state: &SharedState,
    session: &mut Option<Box<dyn InboundAsrSession>>,
    gateway_call_id: Option<&str>,
    stream_id: Option<String>,
) -> anyhow::Result<()> {
    if let (Some(call_id), Some(asr_session)) = (gateway_call_id, session.take()) {
        let events = asr_session.finish().await?;
        record_transcript_events(state, call_id, stream_id.as_deref(), None, events).await;
        let mut guard = state.write().await;
        if let Some(call) = guard.calls.get_mut(call_id) {
            call.status = CallStatus::Ended;
            call.push_timeline("media stream stopped");
        }
    }
    Ok(())
}

async fn record_transcript_events(
    state: &SharedState,
    gateway_call_id: &str,
    stream_id: Option<&str>,
    media_format: Option<&MediaFormat>,
    events: Vec<TranscriptEvent>,
) {
    let mut guard = state.write().await;
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
        guard.add_transcript(gateway_call_id, kind, text.clone());
        let call = guard.calls.get(gateway_call_id);
        tracing::info!(
            gateway_call_id,
            call_control_id = call.map(|call| call.ids.call_control_id.as_str()),
            call_session_id = call.and_then(|call| call.ids.call_session_id.as_deref()),
            call_leg_id = call.and_then(|call| call.ids.call_leg_id.as_deref()),
            stream_id,
            codec = media_format.map(|format| format.encoding.as_str()),
            sample_rate_hz = media_format.map(|format| format.sample_rate_hz),
            transcript_kind = kind_label,
            transcript_text = text,
            "{kind_label}"
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::adapter::EchoAsrFactory;
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
                CallStatus::PendingInbound,
            )
        };
        let asr: SharedAsrFactory = Arc::new(EchoAsrFactory);
        let mut session = None;
        let mut selected_call = None;
        let mut media_format = None;
        let mut reorder = SequencedFrameReorder::new(1, 32);

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
        handle_text(
            &start,
            &state,
            &asr,
            &mut session,
            &mut selected_call,
            &mut media_format,
            &mut reorder,
        )
        .await
        .expect("start event should open ASR session");

        let chunk = STANDARD.encode(l16_samples(8_000));
        let media_two = media_event("stream-1", "2", &chunk);
        handle_text(
            &media_two,
            &state,
            &asr,
            &mut session,
            &mut selected_call,
            &mut media_format,
            &mut reorder,
        )
        .await
        .expect("out-of-order media should buffer");
        assert!(state
            .read()
            .await
            .calls
            .get(&gateway_call_id)
            .expect("call exists")
            .transcripts
            .is_empty());

        let media_one = media_event("stream-1", "1", &chunk);
        handle_text(
            &media_one,
            &state,
            &asr,
            &mut session,
            &mut selected_call,
            &mut media_format,
            &mut reorder,
        )
        .await
        .expect("contiguous media should feed ASR");

        let stop = serde_json::json!({
            "event": "stop",
            "stream_id": "stream-1"
        })
        .to_string();
        handle_text(
            &stop,
            &state,
            &asr,
            &mut session,
            &mut selected_call,
            &mut media_format,
            &mut reorder,
        )
        .await
        .expect("stop should finish ASR session");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(selected_call.as_deref(), Some(gateway_call_id.as_str()));
        assert_eq!(call.status, CallStatus::Ended);
        assert_eq!(call.ids.stream_id.as_deref(), Some("stream-1"));
        assert_eq!(call.media.encoding.as_deref(), Some("L16"));
        assert_eq!(call.media.sample_rate_hz, Some(16_000));
        assert_eq!(call.transcripts.len(), 2);
        assert_eq!(call.transcripts[0].text, "received 16000 samples");
        assert_eq!(call.transcripts[1].text, "received 16000 samples");
    }

    fn l16_samples(count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            bytes.extend_from_slice(&0_i16.to_be_bytes());
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
}
