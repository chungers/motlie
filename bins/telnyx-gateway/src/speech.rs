use anyhow::{bail, Context};
use uuid::Uuid;

use crate::call_control::TelnyxMediaConfig;
use crate::media::{
    packetize_tts_samples, CallMediaHandle, OutboundMediaCommand, OutboundMediaFrame,
    SharedMediaRegistry, SpeechCancelToken,
};
use crate::operator::state::{LogLevel, SharedState};
use crate::tts::{
    split_speech_text, LiveTtsBackend, SharedTtsRegistry, TtsAudio, PIPER_SAMPLE_RATE_HZ,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedSpeech {
    pub playback_id: String,
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
    let playback_id = format!("tts_{}", Uuid::new_v4().simple());
    let cancel = SpeechCancelToken::default();
    let media = {
        let guard = state.read().await;
        let media = guard.config.telnyx_media;
        let call = guard
            .calls
            .get(&gateway_call_id)
            .with_context(|| format!("call not found: {gateway_call_id}"))?;
        if call.ids.stream_id.is_none() {
            bail!("media stream is not ready for call {gateway_call_id}");
        }
        media
    };
    let media_handle = media_registry
        .start_speech(&gateway_call_id, playback_id.clone(), cancel.clone())
        .await?;
    {
        let mut guard = state.write().await;
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
        cancel,
    };
    tokio::spawn(async move {
        run_speech_job(job).await;
    });

    Ok(QueuedSpeech { playback_id })
}

pub async fn cancel_speech(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    gateway_call_id: &str,
    source_label: &str,
) -> anyhow::Result<String> {
    let playback_id = media_registry.cancel_speech(gateway_call_id).await?;
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
        Err(error) => {
            job.media_registry
                .finish_speech(&job.gateway_call_id, &job.playback_id)
                .await;
            let mut guard = job.state.write().await;
            guard.mark_tts_failed(&job.gateway_call_id, &job.playback_id, format!("{error:#}"));
            guard.log(
                LogLevel::Error,
                format!("speak failed for {}: {error:#}", job.gateway_call_id),
            );
            tracing::error!(
                gateway_call_id = job.gateway_call_id.as_str(),
                playback_id = job.playback_id.as_str(),
                error = %error,
                "tts.speak.failed"
            );
        }
    }
}

enum SpeechJobOutcome {
    MarkQueued,
    NoAudioOrCanceled,
}

async fn run_speech_job_inner(job: &SpeechJob) -> anyhow::Result<SpeechJobOutcome> {
    let text_chunks = split_speech_text(&job.text);
    let mut model_samples = Vec::new();
    let mut model_sample_rate_hz = None;
    let mut model_chunks = 0usize;
    for text_chunk in text_chunks {
        if job.cancel.is_canceled() {
            return Ok(SpeechJobOutcome::NoAudioOrCanceled);
        }
        let audio_chunks = synthesize_chunks_with_fallback(
            &job.state,
            &job.tts,
            job.tts_backend,
            &job.gateway_call_id,
            &job.playback_id,
            text_chunk,
        )
        .await?;
        for audio_chunk in audio_chunks {
            if job.cancel.is_canceled() {
                return Ok(SpeechJobOutcome::NoAudioOrCanceled);
            }
            let sample_rate_hz = audio_chunk.sample_rate_hz();
            match model_sample_rate_hz {
                Some(existing) if existing != sample_rate_hz => {
                    bail!(
                        "TTS backend emitted mixed sample rates: {existing}Hz and {sample_rate_hz}Hz"
                    );
                }
                Some(_) => {}
                None => model_sample_rate_hz = Some(sample_rate_hz),
            }
            model_samples.extend(audio_chunk.into_samples_i16());
            model_chunks = model_chunks.saturating_add(1);
        }
    }
    if model_samples.is_empty() || job.cancel.is_canceled() {
        return Ok(SpeechJobOutcome::NoAudioOrCanceled);
    }

    let model_sample_rate_hz = model_sample_rate_hz.unwrap_or(PIPER_SAMPLE_RATE_HZ);
    let packets = packetize_tts_samples(model_samples, model_sample_rate_hz, job.media)?;
    let queued_frames = packets.len();
    if queued_frames == 0 || job.cancel.is_canceled() {
        return Ok(SpeechJobOutcome::NoAudioOrCanceled);
    }
    job.state.write().await.mark_tts_frames_queued(
        &job.gateway_call_id,
        &job.playback_id,
        queued_frames,
    );
    tracing::info!(
        gateway_call_id = job.gateway_call_id.as_str(),
        playback_id = job.playback_id.as_str(),
        model_chunks,
        model_sample_rate_hz,
        frames = queued_frames,
        "tts.speak.prebuffered"
    );
    for payload in packets {
        if job.cancel.is_canceled() {
            return Ok(SpeechJobOutcome::NoAudioOrCanceled);
        }
        job.media_handle
            .send(OutboundMediaCommand::Frame(OutboundMediaFrame {
                playback_id: job.playback_id.clone(),
                payload,
            }))
            .await?;
    }
    if queued_frames > 0 && !job.cancel.is_canceled() {
        job.media_handle
            .send(OutboundMediaCommand::Mark {
                playback_id: job.playback_id.clone(),
            })
            .await?;
        tracing::info!(
            gateway_call_id = job.gateway_call_id.as_str(),
            playback_id = job.playback_id.as_str(),
            frames = queued_frames,
            "tts.speak.queued"
        );
        return Ok(SpeechJobOutcome::MarkQueued);
    }
    Ok(SpeechJobOutcome::NoAudioOrCanceled)
}

async fn synthesize_chunks_with_fallback(
    state: &SharedState,
    tts: &SharedTtsRegistry,
    backend: LiveTtsBackend,
    gateway_call_id: &str,
    playback_id: &str,
    text_chunk: String,
) -> anyhow::Result<Vec<TtsAudio>> {
    let primary = tts.factory(backend);
    match primary.synthesize_chunks(text_chunk.clone()).await {
        Ok(chunks) => Ok(chunks),
        Err(error) => {
            let Some(fallback) = backend.fallback() else {
                return Err(error);
            };
            let error_message = format!("{error:#}");
            {
                let mut guard = state.write().await;
                guard.log(
                    LogLevel::Warn,
                    format!(
                        "TTS backend {} failed for {gateway_call_id} playback={playback_id}; falling back to {}: {error_message}",
                        backend.label(),
                        fallback.label()
                    ),
                );
            }
            tracing::warn!(
                gateway_call_id,
                playback_id,
                backend = backend.label(),
                backend_model = backend.model_label(),
                fallback = fallback.label(),
                fallback_model = fallback.model_label(),
                error = %error_message,
                "tts.speak.fallback"
            );
            tts.factory(fallback)
                .synthesize_chunks(text_chunk)
                .await
                .with_context(|| {
                    format!(
                        "fallback TTS backend {} after {} failed: {error_message}",
                        fallback.label(),
                        backend.label()
                    )
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::shared_state;
    use crate::tts::{OutboundTtsFactory, StaticTtsFactory, TtsRegistry};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct FailingTtsFactory;

    #[async_trait]
    impl OutboundTtsFactory for FailingTtsFactory {
        async fn synthesize_chunks(&self, _text: String) -> anyhow::Result<Vec<TtsAudio>> {
            bail!("kokoro test failure")
        }

        fn label(&self) -> &'static str {
            "failing-test-tts"
        }
    }

    #[tokio::test]
    async fn kokoro_synthesis_failure_falls_back_to_piper() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let tts = Arc::new(TtsRegistry::new(
            Arc::new(FailingTtsFactory),
            Arc::new(StaticTtsFactory::new(vec![1_000; 2_205])),
        ));

        let chunks = synthesize_chunks_with_fallback(
            &state,
            &tts,
            LiveTtsBackend::Kokoro82m,
            "gwc_test",
            "tts_test",
            "hello".to_string(),
        )
        .await
        .expect("fallback should synthesize with Piper");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].sample_rate_hz(), PIPER_SAMPLE_RATE_HZ);
        let guard = state.read().await;
        assert!(guard.logs.iter().any(|entry| {
            entry.level == LogLevel::Warn
                && entry.message.contains("falling back to piper")
                && entry.message.contains("kokoro test failure")
        }));
    }
}
