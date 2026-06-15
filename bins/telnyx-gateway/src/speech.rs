use anyhow::{bail, Context};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::call_control::TelnyxMediaConfig;
use crate::media::{
    packetize_tts_samples, percentile_u64, CallMediaHandle, OutboundFrameQualityContext,
    OutboundMediaCommand, OutboundMediaFrame, SharedMediaRegistry, SpeechCancelToken,
    SpeechClearReason,
};
use crate::operator::state::{CallStatus, LogLevel, QualitySpanEmission, SharedState};
use crate::quality::RedactionMode;
use crate::tts::{
    split_speech_text_with_first_chunk_max_chars, LiveTtsBackend, SharedTtsRegistry, TtsAudio,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedSpeech {
    pub playback_id: String,
    pub replaced_playback_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechConflictPolicy {
    Reject,
    CancelAndReplace,
    Append,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechQueueRequest {
    pub tts_backend: LiveTtsBackend,
    pub gateway_call_id: String,
    pub text: String,
    pub source_label: String,
    pub conflict_policy: SpeechConflictPolicy,
    pub turn_finalized_at: Option<Instant>,
    pub latest_turn_finalized_at: Option<Instant>,
    pub turn_id: Option<String>,
    pub coalesced_turn_ids: Vec<String>,
    pub prebuffer_chunks_override: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct AppendSpeechHandle {
    pub playback_id: String,
    tx: mpsc::Sender<AppendSpeechCommand>,
    cancel: SpeechCancelToken,
}

impl AppendSpeechHandle {
    pub async fn append_chunks(
        &self,
        chunks: Vec<String>,
        final_fragment: bool,
    ) -> anyhow::Result<()> {
        self.tx
            .send(AppendSpeechCommand::Chunks {
                chunks,
                final_fragment,
                received_at: Instant::now(),
            })
            .await
            .context("append speech chunk")
    }

    pub async fn finish(&self) -> anyhow::Result<()> {
        self.append_chunks(Vec::new(), true).await
    }

    pub async fn cancel(&self) {
        self.cancel.cancel();
        let _ = self.tx.send(AppendSpeechCommand::Cancel).await;
    }

    pub fn cancel_now(&self) {
        self.cancel.cancel();
        let _ = self.tx.try_send(AppendSpeechCommand::Cancel);
    }
}

#[derive(Debug)]
enum AppendSpeechCommand {
    Chunks {
        chunks: Vec<String>,
        final_fragment: bool,
        received_at: Instant,
    },
    Cancel,
}

pub async fn queue_speech(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    tts: &SharedTtsRegistry,
    tts_backend: LiveTtsBackend,
    gateway_call_id: String,
    text: String,
    source_label: &str,
) -> anyhow::Result<QueuedSpeech> {
    queue_speech_with_request(
        state,
        media_registry,
        tts,
        SpeechQueueRequest {
            tts_backend,
            gateway_call_id,
            text,
            source_label: source_label.to_string(),
            conflict_policy: SpeechConflictPolicy::Reject,
            turn_finalized_at: None,
            latest_turn_finalized_at: None,
            turn_id: None,
            coalesced_turn_ids: Vec::new(),
            prebuffer_chunks_override: None,
        },
    )
    .await
}

pub async fn queue_speech_with_request(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    tts: &SharedTtsRegistry,
    request: SpeechQueueRequest,
) -> anyhow::Result<QueuedSpeech> {
    let SpeechQueueRequest {
        tts_backend,
        gateway_call_id,
        text,
        source_label,
        conflict_policy,
        turn_finalized_at,
        latest_turn_finalized_at,
        turn_id,
        coalesced_turn_ids,
        prebuffer_chunks_override,
    } = request;
    let request_started_at = Instant::now();
    let playback_id = format!("tts_{}", Uuid::new_v4().simple());
    let cancel = SpeechCancelToken::default();
    let (
        media,
        quality_config_id,
        quality_redaction_mode,
        tts_chunking_enabled,
        tts_max_text_chunk_chars,
        tts_first_chunk_max_chars,
        tts_prebuffer_chunks,
    ) = {
        let guard = state.read().await;
        let media = guard.config.telnyx_media;
        let call = guard
            .calls
            .get(&gateway_call_id)
            .with_context(|| format!("call not found: {gateway_call_id}"))?;
        if call.ids.stream_id.is_none() {
            bail!("media stream is not ready for call {gateway_call_id}");
        }
        (
            media,
            guard.quality.config_id.clone(),
            guard.quality.config.logging.redaction_mode,
            guard.quality.config.tts.chunking_enabled,
            guard.quality.config.tts.max_text_chunk_chars,
            guard.quality.config.tts.first_chunk_max_chars,
            prebuffer_chunks_override.unwrap_or(guard.quality.config.tts.prebuffer_chunks),
        )
    };
    let (media_handle, replaced_playback_id) = match conflict_policy {
        SpeechConflictPolicy::Reject => (
            media_registry
                .start_speech(&gateway_call_id, playback_id.clone(), cancel.clone())
                .await?,
            None,
        ),
        SpeechConflictPolicy::CancelAndReplace => {
            media_registry
                .start_speech_replacing_active(
                    &gateway_call_id,
                    playback_id.clone(),
                    cancel.clone(),
                )
                .await?
        }
        SpeechConflictPolicy::Append => {
            bail!("append speech policy requires an append speech session")
        }
    };
    {
        let mut guard = state.write().await;
        if let Some(replaced_playback_id) = &replaced_playback_id {
            guard.mark_tts_canceling(&gateway_call_id, replaced_playback_id);
            guard.log(
                LogLevel::Info,
                format!(
                    "{source_label} replaced active speech for {gateway_call_id}: {replaced_playback_id}"
                ),
            );
        }
        guard.start_tts_job(&gateway_call_id, playback_id.clone(), tts_backend, &text);
        guard.log(
            LogLevel::Info,
            format!("{source_label} requested for {gateway_call_id}: {playback_id}"),
        );
    }

    let job = SpeechJob {
        state: state.clone(),
        tts: tts.clone(),
        media_registry: media_registry.clone(),
        media_handle,
        gateway_call_id: gateway_call_id.clone(),
        playback_id: playback_id.clone(),
        tts_backend,
        text,
        media,
        quality_config_id,
        quality_redaction_mode,
        tts_chunking_enabled,
        tts_max_text_chunk_chars,
        tts_first_chunk_max_chars,
        tts_prebuffer_chunks,
        request_started_at,
        turn_finalized_at,
        latest_turn_finalized_at,
        turn_id,
        coalesced_turn_ids,
        cancel,
    };
    tokio::spawn(async move {
        run_speech_job(job).await;
    });

    Ok(QueuedSpeech {
        playback_id,
        replaced_playback_id,
    })
}

pub async fn queue_append_speech_with_request(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    tts: &SharedTtsRegistry,
    request: SpeechQueueRequest,
    initial_chunks: Vec<String>,
) -> anyhow::Result<(AppendSpeechHandle, QueuedSpeech)> {
    if initial_chunks.is_empty() {
        bail!("append speech session requires at least one initial chunk");
    }
    let SpeechQueueRequest {
        tts_backend,
        gateway_call_id,
        text,
        source_label,
        conflict_policy,
        turn_finalized_at,
        latest_turn_finalized_at,
        turn_id,
        coalesced_turn_ids,
        prebuffer_chunks_override,
    } = request;
    let request_started_at = Instant::now();
    let playback_id = format!("tts_{}", Uuid::new_v4().simple());
    let cancel = SpeechCancelToken::default();
    let (
        media,
        quality_config_id,
        quality_redaction_mode,
        tts_chunking_enabled,
        tts_max_text_chunk_chars,
        tts_first_chunk_max_chars,
        tts_prebuffer_chunks,
    ) = {
        let guard = state.read().await;
        let media = guard.config.telnyx_media;
        let call = guard
            .calls
            .get(&gateway_call_id)
            .with_context(|| format!("call not found: {gateway_call_id}"))?;
        if call.ids.stream_id.is_none() {
            bail!("media stream is not ready for call {gateway_call_id}");
        }
        (
            media,
            guard.quality.config_id.clone(),
            guard.quality.config.logging.redaction_mode,
            guard.quality.config.tts.chunking_enabled,
            guard.quality.config.tts.max_text_chunk_chars,
            guard.quality.config.tts.first_chunk_max_chars,
            prebuffer_chunks_override.unwrap_or(guard.quality.config.tts.prebuffer_chunks),
        )
    };
    let (media_handle, replaced_playback_id) = match conflict_policy {
        SpeechConflictPolicy::Reject | SpeechConflictPolicy::Append => (
            media_registry
                .start_speech(&gateway_call_id, playback_id.clone(), cancel.clone())
                .await?,
            None,
        ),
        SpeechConflictPolicy::CancelAndReplace => {
            media_registry
                .start_speech_replacing_active(
                    &gateway_call_id,
                    playback_id.clone(),
                    cancel.clone(),
                )
                .await?
        }
    };
    {
        let mut guard = state.write().await;
        if let Some(replaced_playback_id) = &replaced_playback_id {
            guard.mark_tts_canceling(&gateway_call_id, replaced_playback_id);
            guard.log(
                LogLevel::Info,
                format!(
                    "{source_label} replaced active speech for {gateway_call_id}: {replaced_playback_id}"
                ),
            );
        }
        guard.start_tts_job(&gateway_call_id, playback_id.clone(), tts_backend, &text);
        guard.log(
            LogLevel::Info,
            format!("{source_label} append speech started for {gateway_call_id}: {playback_id}"),
        );
    }

    let (tx, rx) = mpsc::channel(32);
    let handle = AppendSpeechHandle {
        playback_id: playback_id.clone(),
        tx,
        cancel: cancel.clone(),
    };
    let job = AppendSpeechJob {
        speech: SpeechJob {
            state: state.clone(),
            tts: tts.clone(),
            media_registry: media_registry.clone(),
            media_handle,
            gateway_call_id: gateway_call_id.clone(),
            playback_id: playback_id.clone(),
            tts_backend,
            text,
            media,
            quality_config_id,
            quality_redaction_mode,
            tts_chunking_enabled,
            tts_max_text_chunk_chars,
            tts_first_chunk_max_chars,
            tts_prebuffer_chunks,
            request_started_at,
            turn_finalized_at,
            latest_turn_finalized_at,
            turn_id,
            coalesced_turn_ids,
            cancel,
        },
        rx,
        initial_chunks,
    };
    tokio::spawn(async move {
        run_append_speech_job(job).await;
    });

    Ok((
        handle,
        QueuedSpeech {
            playback_id,
            replaced_playback_id,
        },
    ))
}

pub async fn cancel_speech(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    gateway_call_id: &str,
    source_label: &str,
) -> anyhow::Result<String> {
    cancel_speech_with_reason(
        state,
        media_registry,
        gateway_call_id,
        source_label,
        SpeechClearReason::Operator,
    )
    .await
}

pub async fn cancel_speech_with_reason(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    gateway_call_id: &str,
    source_label: &str,
    reason: SpeechClearReason,
) -> anyhow::Result<String> {
    let playback_id = media_registry
        .cancel_speech_for_reason(gateway_call_id, reason)
        .await?;
    {
        let mut guard = state.write().await;
        guard.mark_tts_canceling(gateway_call_id, &playback_id);
        guard.log(
            LogLevel::Info,
            format!("{source_label} cancel requested for {gateway_call_id}: {playback_id}"),
        );
    }
    Ok(playback_id)
}

struct SpeechJob {
    state: SharedState,
    tts: SharedTtsRegistry,
    media_registry: SharedMediaRegistry,
    media_handle: CallMediaHandle,
    gateway_call_id: String,
    playback_id: String,
    tts_backend: LiveTtsBackend,
    text: String,
    media: TelnyxMediaConfig,
    quality_config_id: String,
    quality_redaction_mode: RedactionMode,
    tts_chunking_enabled: bool,
    tts_max_text_chunk_chars: usize,
    tts_first_chunk_max_chars: usize,
    tts_prebuffer_chunks: usize,
    request_started_at: Instant,
    turn_finalized_at: Option<Instant>,
    latest_turn_finalized_at: Option<Instant>,
    turn_id: Option<String>,
    coalesced_turn_ids: Vec<String>,
    cancel: SpeechCancelToken,
}

struct AppendSpeechJob {
    speech: SpeechJob,
    rx: mpsc::Receiver<AppendSpeechCommand>,
    initial_chunks: Vec<String>,
}

#[derive(Default)]
struct AppendSpeechStats {
    append_wait: Duration,
    inter_fragment_gap_ms: Vec<u64>,
    last_fragment_at: Option<Instant>,
}

async fn run_speech_job(job: SpeechJob) {
    let result = run_speech_job_inner(&job).await;
    match result {
        Ok(SpeechJobOutcome::MarkQueued) => {}
        Ok(SpeechJobOutcome::NoAudioOrCanceled) => {
            job.media_registry
                .finish_speech(&job.gateway_call_id, &job.playback_id)
                .await;
        }
        Err(failure) => {
            if mark_media_closed_after_call_end(&job, &failure, "speak").await {
                return;
            }
            if failure.queued_frames > 0 {
                request_failed_speech_clear(&job, failure.queued_frames).await;
            } else {
                job.media_registry
                    .finish_speech(&job.gateway_call_id, &job.playback_id)
                    .await;
            }
            let mut guard = job.state.write().await;
            guard.mark_tts_failed(
                &job.gateway_call_id,
                &job.playback_id,
                format!("{:#}", failure.error),
            );
            guard.log(
                LogLevel::Error,
                format!(
                    "speak failed for {}: {:#}",
                    job.gateway_call_id, failure.error
                ),
            );
            tracing::error!(
                gateway_call_id = job.gateway_call_id.as_str(),
                playback_id = job.playback_id.as_str(),
                queued_frames = failure.queued_frames,
                error = %failure.error,
                "tts.speak.failed"
            );
        }
    }
}

async fn run_append_speech_job(job: AppendSpeechJob) {
    let result = run_append_speech_job_inner(job).await;
    match result {
        Ok((speech, SpeechJobOutcome::MarkQueued)) => {
            let _ = speech
                .media_handle
                .send(OutboundMediaCommand::AppendState {
                    playback_id: speech.playback_id.clone(),
                    open: false,
                    empty: true,
                })
                .await;
        }
        Ok((speech, SpeechJobOutcome::NoAudioOrCanceled)) => {
            let _ = speech
                .media_handle
                .send(OutboundMediaCommand::AppendState {
                    playback_id: speech.playback_id.clone(),
                    open: false,
                    empty: true,
                })
                .await;
            speech
                .media_registry
                .finish_speech(&speech.gateway_call_id, &speech.playback_id)
                .await;
        }
        Err((speech, failure)) => {
            let _ = speech
                .media_handle
                .send(OutboundMediaCommand::AppendState {
                    playback_id: speech.playback_id.clone(),
                    open: false,
                    empty: true,
                })
                .await;
            if mark_media_closed_after_call_end(&speech, &failure, "append speak").await {
                return;
            }
            if failure.queued_frames > 0 {
                request_failed_speech_clear(&speech, failure.queued_frames).await;
            } else {
                speech
                    .media_registry
                    .finish_speech(&speech.gateway_call_id, &speech.playback_id)
                    .await;
            }
            let mut guard = speech.state.write().await;
            guard.mark_tts_failed(
                &speech.gateway_call_id,
                &speech.playback_id,
                format!("{:#}", failure.error),
            );
            guard.log(
                LogLevel::Error,
                format!(
                    "append speak failed for {}: {:#}",
                    speech.gateway_call_id, failure.error
                ),
            );
            tracing::error!(
                gateway_call_id = speech.gateway_call_id.as_str(),
                playback_id = speech.playback_id.as_str(),
                queued_frames = failure.queued_frames,
                error = %failure.error,
                "tts.append.failed"
            );
        }
    }
}

async fn run_append_speech_job_inner(
    mut job: AppendSpeechJob,
) -> Result<(SpeechJob, SpeechJobOutcome), (SpeechJob, SpeechJobFailure)> {
    let mut queued_frames = 0usize;
    let mut total_frames = 0usize;
    let mut model_chunks = 0usize;
    let mut text_chunks = 0usize;
    let mut enqueue_duration = Duration::ZERO;
    let mut emitted_first_synthesis = false;
    let mut emitted_first_packetize = false;
    let mut emitted_prebuffer_ready = false;
    let mut first_packet_for_playback = true;
    let mut stats = AppendSpeechStats::default();
    let synthesis_started_at = Instant::now();
    let mut pending_chunks = std::mem::take(&mut job.initial_chunks);
    let mut final_received = false;

    loop {
        for text_chunk in pending_chunks.drain(..) {
            if job.speech.cancel.is_canceled() {
                return Ok((job.speech, SpeechJobOutcome::NoAudioOrCanceled));
            }
            let chunk_index = text_chunks;
            text_chunks = text_chunks.saturating_add(1);
            let chunk_synthesis_started_at = Instant::now();
            let synthesis_context = SpeechSynthesisContext {
                state: &job.speech.state,
                tts: &job.speech.tts,
                backend: job.speech.tts_backend,
                gateway_call_id: &job.speech.gateway_call_id,
                playback_id: &job.speech.playback_id,
                cancel: &job.speech.cancel,
            };
            let audio_chunks = match synthesize_text_chunk_with_fallback(
                &synthesis_context,
                chunk_index,
                &text_chunk,
            )
            .await
            {
                Ok(Some(audio_chunks)) => audio_chunks,
                Ok(None) => return Ok((job.speech, SpeechJobOutcome::NoAudioOrCanceled)),
                Err(error) => {
                    return Err((job.speech, SpeechJobFailure::new(error, queued_frames)));
                }
            };
            if !emitted_first_synthesis {
                emitted_first_synthesis = true;
                emit_speech_span(
                    &job.speech,
                    "tts.synthesis_first_chunk",
                    "tts_generation",
                    chunk_synthesis_started_at.elapsed(),
                    true,
                    false,
                    serde_json::json!({
                        "playback_id": job.speech.playback_id.as_str(),
                        "tts_backend": job.speech.tts_backend.label(),
                        "text_chunk_index": chunk_index,
                        "text_chunking_enabled": job.speech.tts_chunking_enabled,
                        "max_text_chunk_chars": job.speech.tts_max_text_chunk_chars,
                        "first_chunk_max_chars": job.speech.tts_first_chunk_max_chars,
                        "append_stream": true,
                        "text_chars": text_chunk.chars().count(),
                        "audio_chunks": audio_chunks.len(),
                    }),
                )
                .await;
            }
            let synthesis = match concatenate_audio_chunks(job.speech.tts_backend, audio_chunks) {
                Ok(Some(synthesis)) => synthesis,
                Ok(None) => continue,
                Err(error) => {
                    return Err((job.speech, SpeechJobFailure::new(error, queued_frames)));
                }
            };
            let packetize_started_at = Instant::now();
            let frames = match packetize_tts_samples(
                synthesis.samples_i16,
                synthesis.sample_rate_hz,
                job.speech.media,
            ) {
                Ok(frames) => frames,
                Err(error) => {
                    return Err((job.speech, SpeechJobFailure::new(error, queued_frames)));
                }
            };
            if frames.is_empty() {
                continue;
            }
            if !emitted_first_packetize {
                emitted_first_packetize = true;
                emit_speech_span(
                    &job.speech,
                    "tts.packetize_first_chunk",
                    "media_packetization",
                    packetize_started_at.elapsed(),
                    true,
                    false,
                    serde_json::json!({
                        "playback_id": job.speech.playback_id.as_str(),
                        "tts_backend": job.speech.tts_backend.label(),
                        "text_chunk_index": chunk_index,
                        "append_stream": true,
                        "frames": frames.len(),
                    }),
                )
                .await;
            }
            total_frames = total_frames.saturating_add(frames.len());
            model_chunks = model_chunks.saturating_add(synthesis.audio_chunk_count);
            let mut prepared = vec![PreparedSpeechChunk {
                text_chunk_index: chunk_index,
                audio_chunk_count: synthesis.audio_chunk_count,
                sample_rate_hz: synthesis.sample_rate_hz,
                frames,
            }];
            if !emitted_prebuffer_ready {
                emitted_prebuffer_ready = true;
                let buffered_frames = prepared_frame_count(&prepared);
                emit_speech_span(
                    &job.speech,
                    "tts.prebuffer_ready",
                    "tts_generation",
                    synthesis_started_at.elapsed(),
                    true,
                    false,
                    serde_json::json!({
                        "playback_id": job.speech.playback_id.as_str(),
                        "tts_backend": job.speech.tts_backend.label(),
                        "prepared_text_chunks": 1,
                        "append_stream": true,
                        "frames": buffered_frames,
                        "text_chunking_enabled": job.speech.tts_chunking_enabled,
                        "max_text_chunk_chars": job.speech.tts_max_text_chunk_chars,
                        "first_chunk_max_chars": job.speech.tts_first_chunk_max_chars,
                    }),
                )
                .await;
            }
            match enqueue_prepared_chunks(
                &job.speech,
                &mut prepared,
                &mut queued_frames,
                &mut first_packet_for_playback,
            )
            .await
            {
                Ok(duration) => enqueue_duration += duration,
                Err(failure) => return Err((job.speech, failure)),
            }
        }

        if final_received || job.speech.cancel.is_canceled() {
            break;
        }

        let wait_started_at = Instant::now();
        let _ = job
            .speech
            .media_handle
            .send(OutboundMediaCommand::AppendState {
                playback_id: job.speech.playback_id.clone(),
                open: true,
                empty: true,
            })
            .await;
        let command = tokio::select! {
            command = job.rx.recv() => command,
            _ = job.speech.cancel.canceled() => None,
        };
        stats.append_wait += wait_started_at.elapsed();
        let _ = job
            .speech
            .media_handle
            .send(OutboundMediaCommand::AppendState {
                playback_id: job.speech.playback_id.clone(),
                open: true,
                empty: false,
            })
            .await;
        match command {
            Some(AppendSpeechCommand::Chunks {
                chunks,
                final_fragment,
                received_at,
            }) => {
                if let Some(last) = stats.last_fragment_at.replace(received_at) {
                    stats
                        .inter_fragment_gap_ms
                        .push(received_at.saturating_duration_since(last).as_millis() as u64);
                }
                pending_chunks = chunks;
                final_received = final_fragment;
            }
            Some(AppendSpeechCommand::Cancel) | None => {
                job.speech.cancel.cancel();
                break;
            }
        }
    }

    if queued_frames == 0 || job.speech.cancel.is_canceled() {
        return Ok((job.speech, SpeechJobOutcome::NoAudioOrCanceled));
    }

    emit_speech_span(
        &job.speech,
        "tts.synthesis_full",
        "tts_generation",
        synthesis_started_at.elapsed(),
        false,
        true,
        serde_json::json!({
            "playback_id": job.speech.playback_id.as_str(),
            "tts_backend": job.speech.tts_backend.label(),
            "text_chunks": text_chunks,
            "audio_chunks": model_chunks,
            "frames": total_frames,
            "append_stream": true,
            "append_wait_ms": stats.append_wait.as_millis() as u64,
            "inter_fragment_gap_p50_ms": percentile_u64(&stats.inter_fragment_gap_ms, 50),
            "inter_fragment_gap_p95_ms": percentile_u64(&stats.inter_fragment_gap_ms, 95),
        }),
    )
    .await;
    emit_speech_span(
        &job.speech,
        "tts.frames_enqueue",
        "media_packetization",
        enqueue_duration,
        false,
        true,
        serde_json::json!({
            "playback_id": job.speech.playback_id.as_str(),
            "tts_backend": job.speech.tts_backend.label(),
            "text_chunks": text_chunks,
            "frames": queued_frames,
            "append_stream": true,
        }),
    )
    .await;
    if let Err(error) = job
        .speech
        .media_handle
        .send(OutboundMediaCommand::Mark {
            playback_id: job.speech.playback_id.clone(),
        })
        .await
    {
        return Err((job.speech, SpeechJobFailure::new(error, queued_frames)));
    }
    tracing::info!(
        gateway_call_id = job.speech.gateway_call_id.as_str(),
        playback_id = job.speech.playback_id.as_str(),
        text_chunks,
        model_chunks,
        frames = queued_frames,
        append_wait_ms = stats.append_wait.as_millis() as u64,
        "tts.append.queued"
    );
    Ok((job.speech, SpeechJobOutcome::MarkQueued))
}

async fn request_failed_speech_clear(job: &SpeechJob, queued_frames: usize) {
    match job
        .media_registry
        .cancel_speech_for_reason(&job.gateway_call_id, SpeechClearReason::TtsFailed)
        .await
    {
        Ok(playback_id) => {
            tracing::warn!(
                gateway_call_id = job.gateway_call_id.as_str(),
                playback_id,
                queued_frames,
                "tts.speak.failed_clear_requested"
            );
        }
        Err(error) => {
            job.media_registry
                .finish_speech(&job.gateway_call_id, &job.playback_id)
                .await;
            tracing::warn!(
                gateway_call_id = job.gateway_call_id.as_str(),
                playback_id = job.playback_id.as_str(),
                queued_frames,
                error = %error,
                "tts.speak.failed_clear_unavailable"
            );
        }
    }
}

enum SpeechJobOutcome {
    MarkQueued,
    NoAudioOrCanceled,
}

struct SpeechJobFailure {
    error: anyhow::Error,
    queued_frames: usize,
}

impl SpeechJobFailure {
    fn new(error: anyhow::Error, queued_frames: usize) -> Self {
        Self {
            error,
            queued_frames,
        }
    }
}

async fn mark_media_closed_after_call_end(
    job: &SpeechJob,
    failure: &SpeechJobFailure,
    source_label: &'static str,
) -> bool {
    if !is_outbound_media_channel_closed(&failure.error) {
        return false;
    }
    let terminal_call = {
        let guard = job.state.read().await;
        guard
            .calls
            .get(&job.gateway_call_id)
            .map(|call| matches!(call.status, CallStatus::Ended | CallStatus::Failed))
            .unwrap_or(true)
    };
    if !terminal_call {
        return false;
    }

    job.media_registry
        .finish_speech(&job.gateway_call_id, &job.playback_id)
        .await;
    let mut guard = job.state.write().await;
    guard.mark_tts_canceled(&job.gateway_call_id, &job.playback_id);
    guard.log(
        LogLevel::Info,
        format!(
            "{source_label} stopped for {} after media closed: {}",
            job.gateway_call_id, job.playback_id
        ),
    );
    tracing::info!(
        gateway_call_id = job.gateway_call_id.as_str(),
        playback_id = job.playback_id.as_str(),
        queued_frames = failure.queued_frames,
        error = %failure.error,
        "tts.media_closed_after_call_end"
    );
    true
}

fn is_outbound_media_channel_closed(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<mpsc::error::SendError<OutboundMediaCommand>>()
            .is_some()
    })
}

async fn run_speech_job_inner(job: &SpeechJob) -> Result<SpeechJobOutcome, SpeechJobFailure> {
    let text_chunks = if job.tts_chunking_enabled {
        split_speech_text_with_first_chunk_max_chars(
            &job.text,
            job.tts_max_text_chunk_chars,
            job.tts_first_chunk_max_chars,
        )
    } else {
        let text = job.text.trim();
        if text.is_empty() {
            Vec::new()
        } else {
            vec![text.to_string()]
        }
    };
    let effective_prebuffer_chunks =
        effective_prebuffer_chunks(job.tts_prebuffer_chunks, text_chunks.len());
    let synthesis_context = SpeechSynthesisContext {
        state: &job.state,
        tts: &job.tts,
        backend: job.tts_backend,
        gateway_call_id: &job.gateway_call_id,
        playback_id: &job.playback_id,
        cancel: &job.cancel,
    };
    let mut prepared_chunks = Vec::new();
    let mut total_frames = 0usize;
    let mut queued_frames = 0usize;
    let mut model_chunks = 0usize;
    let synthesis_started_at = Instant::now();
    let mut enqueue_duration = Duration::ZERO;
    let mut emitted_first_synthesis = false;
    let mut emitted_first_packetize = false;
    let mut emitted_prebuffer_ready = false;
    let mut emitted_full_synthesis = false;
    let mut first_packet_for_playback = true;

    for (text_chunk_index, text_chunk) in text_chunks.iter().enumerate() {
        let chunk_synthesis_started_at = Instant::now();
        let Some(audio_chunks) =
            synthesize_text_chunk_with_fallback(&synthesis_context, text_chunk_index, text_chunk)
                .await
                .map_err(|error| SpeechJobFailure::new(error, queued_frames))?
        else {
            return Ok(SpeechJobOutcome::NoAudioOrCanceled);
        };
        if !emitted_first_synthesis {
            emitted_first_synthesis = true;
            emit_speech_span(
                job,
                "tts.synthesis_first_chunk",
                "tts_generation",
                chunk_synthesis_started_at.elapsed(),
                true,
                false,
                serde_json::json!({
                    "playback_id": job.playback_id.as_str(),
                    "tts_backend": job.tts_backend.label(),
                    "text_chunk_index": text_chunk_index,
                    "text_chunks": text_chunks.len(),
                    "text_chunking_enabled": job.tts_chunking_enabled,
                    "max_text_chunk_chars": job.tts_max_text_chunk_chars,
                    "first_chunk_max_chars": job.tts_first_chunk_max_chars,
                    "prebuffer_chunks": job.tts_prebuffer_chunks,
                    "effective_prebuffer_chunks": effective_prebuffer_chunks,
                    "text_chars": text_chunk.chars().count(),
                    "audio_chunks": audio_chunks.len(),
                }),
            )
            .await;
        }
        if job.cancel.is_canceled() {
            return Ok(SpeechJobOutcome::NoAudioOrCanceled);
        }
        let Some(synthesis) = concatenate_audio_chunks(job.tts_backend, audio_chunks)
            .map_err(|error| SpeechJobFailure::new(error, queued_frames))?
        else {
            continue;
        };

        let packetize_started_at = Instant::now();
        let frames =
            packetize_tts_samples(synthesis.samples_i16, synthesis.sample_rate_hz, job.media)
                .map_err(|error| SpeechJobFailure::new(error, queued_frames))?;
        if frames.is_empty() {
            continue;
        }
        if !emitted_first_packetize {
            emitted_first_packetize = true;
            emit_speech_span(
                job,
                "tts.packetize_first_chunk",
                "media_packetization",
                packetize_started_at.elapsed(),
                true,
                false,
                serde_json::json!({
                    "playback_id": job.playback_id.as_str(),
                    "tts_backend": job.tts_backend.label(),
                    "text_chunk_index": text_chunk_index,
                    "frames": frames.len(),
                }),
            )
            .await;
        }
        total_frames = total_frames.saturating_add(frames.len());
        model_chunks = model_chunks.saturating_add(synthesis.audio_chunk_count);
        prepared_chunks.push(PreparedSpeechChunk {
            text_chunk_index,
            audio_chunk_count: synthesis.audio_chunk_count,
            sample_rate_hz: synthesis.sample_rate_hz,
            frames,
        });

        let is_last_chunk = text_chunk_index + 1 == text_chunks.len();
        if is_last_chunk && total_frames > 0 && !emitted_full_synthesis {
            emitted_full_synthesis = true;
            emit_speech_span(
                job,
                "tts.synthesis_full",
                "tts_generation",
                synthesis_started_at.elapsed(),
                false,
                true,
                serde_json::json!({
                    "playback_id": job.playback_id.as_str(),
                    "tts_backend": job.tts_backend.label(),
                    "text_chunks": text_chunks.len(),
                    "audio_chunks": model_chunks,
                    "frames": total_frames,
                    "text_chunking_enabled": job.tts_chunking_enabled,
                    "max_text_chunk_chars": job.tts_max_text_chunk_chars,
                    "first_chunk_max_chars": job.tts_first_chunk_max_chars,
                    "prebuffer_chunks": job.tts_prebuffer_chunks,
                    "effective_prebuffer_chunks": effective_prebuffer_chunks,
                }),
            )
            .await;
        }

        let playback_started = !first_packet_for_playback;
        if playback_started || prepared_chunks.len() >= effective_prebuffer_chunks || is_last_chunk
        {
            if !emitted_prebuffer_ready {
                emitted_prebuffer_ready = true;
                let buffered_frames = prepared_frame_count(&prepared_chunks);
                emit_speech_span(
                    job,
                    "tts.prebuffer_ready",
                    "tts_generation",
                    synthesis_started_at.elapsed(),
                    true,
                    false,
                    serde_json::json!({
                        "playback_id": job.playback_id.as_str(),
                        "tts_backend": job.tts_backend.label(),
                        "prepared_text_chunks": prepared_chunks.len(),
                        "text_chunks": text_chunks.len(),
                        "frames": buffered_frames,
                        "text_chunking_enabled": job.tts_chunking_enabled,
                        "max_text_chunk_chars": job.tts_max_text_chunk_chars,
                        "first_chunk_max_chars": job.tts_first_chunk_max_chars,
                        "prebuffer_chunks": job.tts_prebuffer_chunks,
                    "effective_prebuffer_chunks": effective_prebuffer_chunks,
                    }),
                )
                .await;
                tracing::info!(
                    gateway_call_id = job.gateway_call_id.as_str(),
                    playback_id = job.playback_id.as_str(),
                    prepared_text_chunks = prepared_chunks.len(),
                    text_chunks = text_chunks.len(),
                    frames = buffered_frames,
                    elapsed_ms = synthesis_started_at.elapsed().as_millis(),
                    "tts.speak.prebuffer_ready"
                );
            }
            enqueue_duration += enqueue_prepared_chunks(
                job,
                &mut prepared_chunks,
                &mut queued_frames,
                &mut first_packet_for_playback,
            )
            .await?;
        }
    }

    if !prepared_chunks.is_empty() && !job.cancel.is_canceled() {
        if !emitted_prebuffer_ready {
            let buffered_frames = prepared_frame_count(&prepared_chunks);
            emit_speech_span(
                job,
                "tts.prebuffer_ready",
                "tts_generation",
                synthesis_started_at.elapsed(),
                true,
                false,
                serde_json::json!({
                    "playback_id": job.playback_id.as_str(),
                    "tts_backend": job.tts_backend.label(),
                    "prepared_text_chunks": prepared_chunks.len(),
                    "text_chunks": text_chunks.len(),
                    "frames": buffered_frames,
                    "text_chunking_enabled": job.tts_chunking_enabled,
                    "max_text_chunk_chars": job.tts_max_text_chunk_chars,
                    "first_chunk_max_chars": job.tts_first_chunk_max_chars,
                    "prebuffer_chunks": job.tts_prebuffer_chunks,
                    "effective_prebuffer_chunks": effective_prebuffer_chunks,
                }),
            )
            .await;
        }
        enqueue_duration += enqueue_prepared_chunks(
            job,
            &mut prepared_chunks,
            &mut queued_frames,
            &mut first_packet_for_playback,
        )
        .await?;
    }

    if queued_frames == 0 || job.cancel.is_canceled() {
        return Ok(SpeechJobOutcome::NoAudioOrCanceled);
    }

    if !emitted_full_synthesis {
        emit_speech_span(
            job,
            "tts.synthesis_full",
            "tts_generation",
            synthesis_started_at.elapsed(),
            false,
            true,
            serde_json::json!({
                "playback_id": job.playback_id.as_str(),
                "tts_backend": job.tts_backend.label(),
                "text_chunks": text_chunks.len(),
                "audio_chunks": model_chunks,
                "frames": queued_frames,
                "text_chunking_enabled": job.tts_chunking_enabled,
                "max_text_chunk_chars": job.tts_max_text_chunk_chars,
                "first_chunk_max_chars": job.tts_first_chunk_max_chars,
                "prebuffer_chunks": job.tts_prebuffer_chunks,
                    "effective_prebuffer_chunks": effective_prebuffer_chunks,
            }),
        )
        .await;
    }

    if !job.cancel.is_canceled() {
        emit_speech_span(
            job,
            "tts.frames_enqueue",
            "media_packetization",
            enqueue_duration,
            false,
            true,
            serde_json::json!({
                "playback_id": job.playback_id.as_str(),
                "tts_backend": job.tts_backend.label(),
                "text_chunks": text_chunks.len(),
                "frames": queued_frames,
            }),
        )
        .await;
        job.media_handle
            .send(OutboundMediaCommand::Mark {
                playback_id: job.playback_id.clone(),
            })
            .await
            .map_err(|error| SpeechJobFailure::new(error, queued_frames))?;
        tracing::info!(
            gateway_call_id = job.gateway_call_id.as_str(),
            playback_id = job.playback_id.as_str(),
            model_chunks,
            frames = queued_frames,
            "tts.speak.queued"
        );
        return Ok(SpeechJobOutcome::MarkQueued);
    }
    Ok(SpeechJobOutcome::NoAudioOrCanceled)
}

fn effective_prebuffer_chunks(configured_chunks: usize, text_chunk_count: usize) -> usize {
    configured_chunks.max(1).min(text_chunk_count.max(1))
}

fn prepared_frame_count(chunks: &[PreparedSpeechChunk]) -> usize {
    chunks.iter().map(|chunk| chunk.frames.len()).sum()
}

async fn enqueue_prepared_chunks(
    job: &SpeechJob,
    prepared_chunks: &mut Vec<PreparedSpeechChunk>,
    queued_frames: &mut usize,
    first_packet_for_playback: &mut bool,
) -> Result<Duration, SpeechJobFailure> {
    if prepared_chunks.is_empty() {
        return Ok(Duration::ZERO);
    }

    let enqueue_started_at = Instant::now();
    for chunk in prepared_chunks.drain(..) {
        let chunk_frames = chunk.frames.len();
        let starts_playback = *first_packet_for_playback;
        job.state.write().await.mark_tts_frames_queued(
            &job.gateway_call_id,
            &job.playback_id,
            chunk_frames,
        );
        *queued_frames = queued_frames.saturating_add(chunk_frames);
        tracing::info!(
            gateway_call_id = job.gateway_call_id.as_str(),
            playback_id = job.playback_id.as_str(),
            text_chunk_index = chunk.text_chunk_index,
            audio_chunks = chunk.audio_chunk_count,
            model_sample_rate_hz = chunk.sample_rate_hz,
            frames = chunk_frames,
            total_frames = *queued_frames,
            starts_playback,
            "tts.speak.chunk_queued"
        );

        for payload in chunk.frames {
            if job.cancel.is_canceled() {
                return Ok(enqueue_started_at.elapsed());
            }
            let quality = OutboundFrameQualityContext {
                config_id: job.quality_config_id.clone(),
                redaction_mode: job.quality_redaction_mode,
                request_started_at: job.request_started_at,
                turn_finalized_at: job.turn_finalized_at,
                latest_turn_finalized_at: job.latest_turn_finalized_at,
                turn_id: job.turn_id.clone(),
                coalesced_turn_ids: job.coalesced_turn_ids.clone(),
                queued_at: Instant::now(),
                first_for_playback: *first_packet_for_playback,
            };
            *first_packet_for_playback = false;
            job.media_handle
                .send(OutboundMediaCommand::Frame(OutboundMediaFrame {
                    playback_id: job.playback_id.clone(),
                    payload,
                    quality: Some(quality),
                }))
                .await
                .map_err(|error| SpeechJobFailure::new(error, *queued_frames))?;
        }
    }
    Ok(enqueue_started_at.elapsed())
}

struct PreparedSpeechChunk {
    text_chunk_index: usize,
    audio_chunk_count: usize,
    sample_rate_hz: u32,
    frames: Vec<Vec<u8>>,
}

async fn emit_speech_span(
    job: &SpeechJob,
    span_name: &'static str,
    category: &'static str,
    duration: Duration,
    critical_path: bool,
    concurrent: bool,
    payload: serde_json::Value,
) {
    let payload = match payload {
        serde_json::Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    job.state.write().await.emit_quality_span_finished(
        &job.gateway_call_id,
        QualitySpanEmission {
            config_id: job.quality_config_id.clone(),
            redaction_mode: job.quality_redaction_mode,
            span_name,
            category,
            duration,
            critical_path,
            concurrent,
            payload,
        },
    );
}

struct SynthesizedTextChunk {
    samples_i16: Vec<i16>,
    sample_rate_hz: u32,
    audio_chunk_count: usize,
}

fn concatenate_audio_chunks(
    backend: LiveTtsBackend,
    audio_chunks: Vec<TtsAudio>,
) -> anyhow::Result<Option<SynthesizedTextChunk>> {
    let mut samples_i16 = Vec::new();
    let mut sample_rate_hz = None;
    let mut audio_chunk_count = 0usize;

    for audio_chunk in audio_chunks {
        let chunk_rate_hz = audio_chunk.sample_rate_hz();
        match sample_rate_hz {
            Some(existing) if existing != chunk_rate_hz => {
                bail!(
                    "TTS backend {} emitted mixed sample rates: {existing}Hz and {chunk_rate_hz}Hz",
                    backend.label()
                );
            }
            Some(_) => {}
            None => sample_rate_hz = Some(chunk_rate_hz),
        }
        samples_i16.extend(audio_chunk.into_samples_i16());
        audio_chunk_count = audio_chunk_count.saturating_add(1);
    }

    if samples_i16.is_empty() {
        return Ok(None);
    }
    let Some(sample_rate_hz) = sample_rate_hz else {
        return Ok(None);
    };
    Ok(Some(SynthesizedTextChunk {
        samples_i16,
        sample_rate_hz,
        audio_chunk_count,
    }))
}

struct SpeechSynthesisContext<'a> {
    state: &'a SharedState,
    tts: &'a SharedTtsRegistry,
    backend: LiveTtsBackend,
    gateway_call_id: &'a str,
    playback_id: &'a str,
    cancel: &'a SpeechCancelToken,
}

async fn synthesize_text_chunk_with_fallback(
    context: &SpeechSynthesisContext<'_>,
    text_chunk_index: usize,
    text_chunk: &str,
) -> anyhow::Result<Option<Vec<TtsAudio>>> {
    match synthesize_text_chunk(context.tts, context.backend, text_chunk, context.cancel).await {
        Ok(audio_chunks) => Ok(audio_chunks),
        Err(error) => {
            let Some(fallback) = context.backend.fallback() else {
                return Err(error);
            };
            let error_message = format!("{error:#}");
            {
                let mut guard = context.state.write().await;
                guard.log(
                    LogLevel::Warn,
                    format!(
                        "TTS backend {} failed for {} playback={} chunk={text_chunk_index}; synthesizing chunk with {}: {error_message}",
                        context.backend.label(),
                        context.gateway_call_id,
                        context.playback_id,
                        fallback.label()
                    ),
                );
            }
            tracing::warn!(
                gateway_call_id = context.gateway_call_id,
                playback_id = context.playback_id,
                text_chunk_index,
                backend = context.backend.label(),
                backend_model = context.backend.model_label(),
                fallback = fallback.label(),
                fallback_model = fallback.model_label(),
                error = %error_message,
                "tts.speak.chunk_fallback"
            );
            synthesize_text_chunk(context.tts, fallback, text_chunk, context.cancel)
                .await
                .with_context(|| {
                    format!(
                        "fallback TTS backend {} after {} failed for chunk {text_chunk_index}: {error_message}",
                        fallback.label(),
                        context.backend.label()
                    )
                })
        }
    }
}

async fn synthesize_text_chunk(
    tts: &SharedTtsRegistry,
    backend: LiveTtsBackend,
    text_chunk: &str,
    cancel: &SpeechCancelToken,
) -> anyhow::Result<Option<Vec<TtsAudio>>> {
    if cancel.is_canceled() {
        return Ok(None);
    }
    let audio_chunks = tts
        .factory(backend)
        .synthesize_chunks(text_chunk.to_string())
        .await?;
    if cancel.is_canceled() {
        return Ok(None);
    }
    Ok(Some(audio_chunks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::{shared_state, CallStatus, TelnyxIds, TtsPlaybackStatus};
    use crate::tts::{OutboundTtsFactory, TtsAudio, TtsRegistry, PIPER_SAMPLE_RATE_HZ};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::{mpsc, Notify};
    use tokio::time::{timeout, Duration};

    struct SequencedTtsFactory {
        sample_rate_hz: u32,
        samples_per_chunk: usize,
        fail_on_call: Option<usize>,
        fail_message: &'static str,
        calls: AtomicUsize,
    }

    impl SequencedTtsFactory {
        fn new(
            sample_rate_hz: u32,
            samples_per_chunk: usize,
            fail_on_call: Option<usize>,
            fail_message: &'static str,
        ) -> Self {
            Self {
                sample_rate_hz,
                samples_per_chunk,
                fail_on_call,
                fail_message,
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl OutboundTtsFactory for SequencedTtsFactory {
        async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_call == Some(call) {
                bail!(self.fail_message);
            }
            Ok(vec![TtsAudio::new(
                vec![call as i16; self.samples_per_chunk],
                self.sample_rate_hz,
            )?])
        }

        fn label(&self) -> &'static str {
            "sequenced-test-tts"
        }
    }

    struct BlockingTtsFactory {
        sample_rate_hz: u32,
        samples_per_chunk: usize,
        started: Notify,
        release: Notify,
    }

    impl BlockingTtsFactory {
        fn new(sample_rate_hz: u32, samples_per_chunk: usize) -> Self {
            Self {
                sample_rate_hz,
                samples_per_chunk,
                started: Notify::new(),
                release: Notify::new(),
            }
        }

        async fn wait_started(&self) {
            self.started.notified().await;
        }

        fn release(&self) {
            self.release.notify_waiters();
        }
    }

    #[async_trait]
    impl OutboundTtsFactory for BlockingTtsFactory {
        async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
            self.started.notify_waiters();
            self.release.notified().await;
            Ok(vec![TtsAudio::new(
                vec![1_000; self.samples_per_chunk],
                self.sample_rate_hz,
            )?])
        }

        fn label(&self) -> &'static str {
            "blocking-test-tts"
        }
    }

    struct BlockingSecondChunkTtsFactory {
        sample_rate_hz: u32,
        samples_per_chunk: usize,
        calls: AtomicUsize,
        second_started: Notify,
        release_second: Notify,
    }

    impl BlockingSecondChunkTtsFactory {
        fn new(sample_rate_hz: u32, samples_per_chunk: usize) -> Self {
            Self {
                sample_rate_hz,
                samples_per_chunk,
                calls: AtomicUsize::new(0),
                second_started: Notify::new(),
                release_second: Notify::new(),
            }
        }

        async fn wait_for_second_call(&self) {
            loop {
                if self.calls.load(Ordering::SeqCst) >= 2 {
                    return;
                }
                self.second_started.notified().await;
            }
        }

        fn release_second_call(&self) {
            self.release_second.notify_waiters();
        }
    }

    #[async_trait]
    impl OutboundTtsFactory for BlockingSecondChunkTtsFactory {
        async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == 2 {
                self.second_started.notify_one();
                self.release_second.notified().await;
            }
            Ok(vec![TtsAudio::new(
                vec![call as i16; self.samples_per_chunk],
                self.sample_rate_hz,
            )?])
        }

        fn label(&self) -> &'static str {
            "blocking-test-tts"
        }
    }

    #[tokio::test]
    async fn kokoro_chunk_failure_falls_back_to_piper_for_that_chunk() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let kokoro = Arc::new(SequencedTtsFactory::new(
            24_000,
            2_400,
            Some(2),
            "kokoro chunk failure",
        ));
        let piper = Arc::new(SequencedTtsFactory::new(
            PIPER_SAMPLE_RATE_HZ,
            2_205,
            None,
            "unused",
        ));
        let tts = Arc::new(TtsRegistry::new(kokoro.clone(), piper.clone()));
        let cancel = SpeechCancelToken::default();

        let context = SpeechSynthesisContext {
            state: &state,
            tts: &tts,
            backend: LiveTtsBackend::Kokoro82m,
            gateway_call_id: "gwc_test",
            playback_id: "tts_test",
            cancel: &cancel,
        };

        let first = synthesize_text_chunk_with_fallback(&context, 0, "hello,")
            .await
            .expect("first chunk should synthesize")
            .expect("first chunk should return audio");
        let second = synthesize_text_chunk_with_fallback(&context, 1, "world.")
            .await
            .expect("second chunk should fall back to Piper")
            .expect("fallback chunk should return audio");

        assert_eq!(kokoro.calls(), 2);
        assert_eq!(piper.calls(), 1);
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(first[0].sample_rate_hz(), 24_000);
        assert_eq!(second[0].sample_rate_hz(), PIPER_SAMPLE_RATE_HZ);
        let guard = state.read().await;
        assert!(guard.logs.iter().any(|entry| {
            entry.level == LogLevel::Warn
                && entry.message.contains("synthesizing chunk with piper")
                && entry.message.contains("kokoro chunk failure")
        }));
    }

    #[test]
    fn effective_prebuffer_chunks_honors_configured_value() {
        assert_eq!(effective_prebuffer_chunks(1, 2), 1);
        assert_eq!(effective_prebuffer_chunks(2, 3), 2);
        assert_eq!(effective_prebuffer_chunks(64, 3), 3);
    }

    #[tokio::test]
    async fn queue_speech_waits_for_all_chunks_before_enqueuing_frames() {
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
        let (tx, mut rx) = mpsc::channel(16);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let kokoro = Arc::new(BlockingSecondChunkTtsFactory::new(24_000, 2_400));
        let piper = Arc::new(SequencedTtsFactory::new(
            PIPER_SAMPLE_RATE_HZ,
            2_205,
            None,
            "unused",
        ));
        let tts = Arc::new(TtsRegistry::new(kokoro.clone(), piper));

        let queued = queue_speech(
            &state,
            &media_registry,
            &tts,
            LiveTtsBackend::Kokoro82m,
            gateway_call_id.clone(),
            "Hello world. Second sentence blocks here. Third sentence confirms chunking."
                .to_string(),
            "test say",
        )
        .await
        .expect("speech should be queued");

        timeout(Duration::from_secs(1), kokoro.wait_for_second_call())
            .await
            .expect("second text chunk synthesis should start");
        assert!(
            timeout(Duration::from_millis(200), rx.recv())
                .await
                .is_err(),
            "no frame should be queued until all chunks are synthesized"
        );

        kokoro.release_second_call();
        let mut frame_count = 0usize;
        let mut saw_mark = false;
        for _ in 0..16 {
            let Some(command) = timeout(Duration::from_secs(1), rx.recv())
                .await
                .expect("speech job should finish")
            else {
                break;
            };
            match command {
                OutboundMediaCommand::Frame(frame) => {
                    assert_eq!(frame.playback_id, queued.playback_id);
                    frame_count += 1;
                }
                OutboundMediaCommand::Mark { playback_id } => {
                    assert_eq!(playback_id, queued.playback_id);
                    saw_mark = true;
                    break;
                }
                other => panic!("unexpected command: {other:?}"),
            }
        }
        assert_eq!(frame_count, 15);
        assert!(
            saw_mark,
            "speech job should enqueue a mark after all chunks"
        );
    }

    #[tokio::test]
    async fn default_prebuffer_starts_after_first_chunk_when_speech_is_chunked() {
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
            assert_eq!(guard.quality.config.tts.prebuffer_chunks, 1);
            let config_id = guard.quality.config.config_id();
            guard.quality.config_id = config_id;
        }
        let media_registry = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(16);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let kokoro = Arc::new(BlockingSecondChunkTtsFactory::new(24_000, 2_400));
        let piper = Arc::new(SequencedTtsFactory::new(
            PIPER_SAMPLE_RATE_HZ,
            2_205,
            None,
            "unused",
        ));
        let tts = Arc::new(TtsRegistry::new(kokoro.clone(), piper));

        let queued = queue_speech(
            &state,
            &media_registry,
            &tts,
            LiveTtsBackend::Kokoro82m,
            gateway_call_id.clone(),
            "Hello world. Second sentence blocks here. Third sentence confirms chunking."
                .to_string(),
            "test say",
        )
        .await
        .expect("speech should be queued");

        let first_command = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("default prebuffer should queue first audio promptly")
            .expect("media command should be present");
        let mut frame_count = match first_command {
            OutboundMediaCommand::Frame(frame) => {
                assert_eq!(frame.playback_id, queued.playback_id);
                1
            }
            other => panic!("expected first queued audio frame, got {other:?}"),
        };

        timeout(Duration::from_secs(1), async {
            loop {
                tokio::select! {
                    _ = kokoro.wait_for_second_call() => break,
                    command = rx.recv() => {
                        match command.expect("media command before second chunk") {
                            OutboundMediaCommand::Frame(frame) => {
                                assert_eq!(frame.playback_id, queued.playback_id);
                                frame_count += 1;
                            }
                            other => panic!("expected frame before second chunk, got {other:?}"),
                        }
                    }
                }
            }
        })
        .await
        .expect("second text chunk synthesis should start after first chunk frames drain");
        kokoro.release_second_call();
        for _ in 0..16 {
            let Some(command) = timeout(Duration::from_secs(1), rx.recv())
                .await
                .expect("speech job should finish")
            else {
                break;
            };
            match command {
                OutboundMediaCommand::Frame(frame) => {
                    assert_eq!(frame.playback_id, queued.playback_id);
                    frame_count += 1;
                }
                OutboundMediaCommand::Mark { playback_id } => {
                    assert_eq!(playback_id, queued.playback_id);
                    break;
                }
                other => panic!("unexpected command: {other:?}"),
            }
        }
        assert_eq!(frame_count, 15);
    }

    #[tokio::test]
    async fn media_closed_after_call_end_marks_tts_canceled() {
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
        let (tx, rx) = mpsc::channel(8);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let piper = Arc::new(BlockingTtsFactory::new(PIPER_SAMPLE_RATE_HZ, 2_205));
        let tts = Arc::new(TtsRegistry::new(piper.clone(), piper.clone()));

        let queued = queue_speech_with_request(
            &state,
            &media_registry,
            &tts,
            SpeechQueueRequest {
                tts_backend: LiveTtsBackend::Piper,
                gateway_call_id: gateway_call_id.clone(),
                text: "ending call".to_string(),
                source_label: "test end".to_string(),
                conflict_policy: SpeechConflictPolicy::Reject,
                turn_finalized_at: None,
                latest_turn_finalized_at: None,
                turn_id: None,
                coalesced_turn_ids: Vec::new(),
                prebuffer_chunks_override: None,
            },
        )
        .await
        .expect("speech should queue");
        timeout(Duration::from_secs(1), piper.wait_started())
            .await
            .expect("synthesis should start");
        drop(rx);
        state
            .write()
            .await
            .calls
            .get_mut(&gateway_call_id)
            .expect("call exists")
            .status = CallStatus::Ended;
        piper.release();

        timeout(Duration::from_secs(1), async {
            loop {
                let status = state
                    .read()
                    .await
                    .calls
                    .get(&gateway_call_id)
                    .and_then(|call| call.tts.as_ref().map(|tts| tts.status));
                if status == Some(TtsPlaybackStatus::Canceled) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("closed media should be treated as cancellation");
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        let tts = call.tts.as_ref().expect("tts state exists");
        assert_eq!(tts.playback_id, queued.playback_id);
        assert_eq!(tts.status, TtsPlaybackStatus::Canceled);
        assert!(tts.error.is_none());
        assert!(guard
            .logs
            .iter()
            .any(|entry| { entry.level == LogLevel::Info && entry.message.contains("stopped") }));
        assert!(!guard.logs.iter().any(|entry| {
            entry.level == LogLevel::Error && entry.message.contains("speak failed")
        }));
    }

    #[tokio::test]
    async fn queue_speech_cancel_and_replace_interrupts_active_slot() {
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
        let (tx, _rx) = mpsc::channel(8);
        media_registry
            .register_call(gateway_call_id.clone(), tx)
            .await;
        let existing_cancel = SpeechCancelToken::default();
        media_registry
            .start_speech(
                &gateway_call_id,
                "tts_existing".to_string(),
                existing_cancel.clone(),
            )
            .await
            .expect("register existing speech");
        state.write().await.start_tts_job(
            &gateway_call_id,
            "tts_existing".to_string(),
            LiveTtsBackend::Piper,
            "old reply",
        );
        let piper = Arc::new(SequencedTtsFactory::new(
            PIPER_SAMPLE_RATE_HZ,
            2_205,
            None,
            "unused",
        ));
        let tts = Arc::new(TtsRegistry::new(piper.clone(), piper));

        let queued = queue_speech_with_request(
            &state,
            &media_registry,
            &tts,
            SpeechQueueRequest {
                tts_backend: LiveTtsBackend::Piper,
                gateway_call_id: gateway_call_id.clone(),
                text: "new reply".to_string(),
                source_label: "test replace".to_string(),
                conflict_policy: SpeechConflictPolicy::CancelAndReplace,
                turn_finalized_at: None,
                latest_turn_finalized_at: None,
                turn_id: None,
                coalesced_turn_ids: Vec::new(),
                prebuffer_chunks_override: None,
            },
        )
        .await
        .expect("replacement speech should queue");

        assert_eq!(queued.replaced_playback_id.as_deref(), Some("tts_existing"));
        assert!(existing_cancel.is_canceled());
        assert_eq!(
            media_registry
                .active_speech_playback_id(&gateway_call_id)
                .await
                .as_deref(),
            Some(queued.playback_id.as_str())
        );
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(
            call.tts.as_ref().map(|tts| tts.playback_id.as_str()),
            Some(queued.playback_id.as_str())
        );
        assert!(guard.logs.iter().any(|entry| {
            entry.level == LogLevel::Info
                && entry
                    .message
                    .contains("test replace replaced active speech")
                && entry.message.contains("tts_existing")
        }));
    }
}
