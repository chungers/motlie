use anyhow::{bail, Context};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::call_control::TelnyxMediaConfig;
use crate::media::{
    packetize_tts_samples, CallMediaHandle, OutboundFrameQualityContext, OutboundMediaCommand,
    OutboundMediaFrame, SharedMediaRegistry, SpeechCancelToken, SpeechClearReason,
};
use crate::operator::state::{LogLevel, QualitySpanEmission, SharedState};
use crate::quality::RedactionMode;
use crate::tts::{split_speech_text, LiveTtsBackend, SharedTtsRegistry, TtsAudio};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedSpeech {
    pub playback_id: String,
    pub replaced_playback_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechConflictPolicy {
    Reject,
    CancelAndReplace,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechQueueRequest {
    pub tts_backend: LiveTtsBackend,
    pub gateway_call_id: String,
    pub text: String,
    pub source_label: String,
    pub conflict_policy: SpeechConflictPolicy,
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
    } = request;
    let playback_id = format!("tts_{}", Uuid::new_v4().simple());
    let cancel = SpeechCancelToken::default();
    let (media, quality_config_id, quality_redaction_mode, tts_chunking_enabled) = {
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
        guard.start_tts_job(&gateway_call_id, playback_id.clone(), &text);
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
    cancel: SpeechCancelToken,
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

async fn run_speech_job_inner(job: &SpeechJob) -> Result<SpeechJobOutcome, SpeechJobFailure> {
    let text_chunks = if job.tts_chunking_enabled {
        split_speech_text(&job.text)
    } else {
        let text = job.text.trim();
        if text.is_empty() {
            Vec::new()
        } else {
            vec![text.to_string()]
        }
    };
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
    let mut model_chunks = 0usize;
    let synthesis_started_at = Instant::now();
    let mut emitted_first_synthesis = false;
    let mut emitted_first_packetize = false;

    for (text_chunk_index, text_chunk) in text_chunks.iter().enumerate() {
        let chunk_synthesis_started_at = Instant::now();
        let Some(audio_chunks) =
            synthesize_text_chunk_with_fallback(&synthesis_context, text_chunk_index, text_chunk)
                .await
                .map_err(|error| SpeechJobFailure::new(error, 0))?
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
                    "text_chunking_enabled": job.tts_chunking_enabled,
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
            .map_err(|error| SpeechJobFailure::new(error, 0))?
        else {
            continue;
        };

        let packetize_started_at = Instant::now();
        let frames =
            packetize_tts_samples(synthesis.samples_i16, synthesis.sample_rate_hz, job.media)
                .map_err(|error| SpeechJobFailure::new(error, 0))?;
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
    }

    if total_frames == 0 || job.cancel.is_canceled() {
        return Ok(SpeechJobOutcome::NoAudioOrCanceled);
    }

    let mut queued_frames = 0usize;
    for chunk in prepared_chunks {
        let chunk_frames = chunk.frames.len();
        let mut first_packet_for_playback = queued_frames == 0;
        job.state.write().await.mark_tts_frames_queued(
            &job.gateway_call_id,
            &job.playback_id,
            chunk_frames,
        );
        queued_frames = queued_frames.saturating_add(chunk_frames);
        tracing::info!(
            gateway_call_id = job.gateway_call_id.as_str(),
            playback_id = job.playback_id.as_str(),
            text_chunk_index = chunk.text_chunk_index,
            audio_chunks = chunk.audio_chunk_count,
            model_sample_rate_hz = chunk.sample_rate_hz,
            frames = chunk_frames,
            total_frames = queued_frames,
            "tts.speak.chunk_queued"
        );

        for payload in chunk.frames {
            if job.cancel.is_canceled() {
                return Ok(SpeechJobOutcome::NoAudioOrCanceled);
            }
            let quality = OutboundFrameQualityContext {
                config_id: job.quality_config_id.clone(),
                redaction_mode: job.quality_redaction_mode,
                queued_at: Instant::now(),
                first_for_playback: first_packet_for_playback,
            };
            first_packet_for_playback = false;
            job.media_handle
                .send(OutboundMediaCommand::Frame(OutboundMediaFrame {
                    playback_id: job.playback_id.clone(),
                    payload,
                    quality: Some(quality),
                }))
                .await
                .map_err(|error| SpeechJobFailure::new(error, queued_frames))?;
        }
    }

    if !job.cancel.is_canceled() {
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
    use crate::operator::state::{shared_state, CallStatus, TelnyxIds};
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
            self.second_started.notified().await;
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
                self.second_started.notify_waiters();
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
            "Hello, world.".to_string(),
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
        assert_eq!(frame_count, 10);
        assert!(
            saw_mark,
            "speech job should enqueue a mark after all chunks"
        );
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
