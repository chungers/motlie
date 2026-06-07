use anyhow::{bail, Context};
use uuid::Uuid;

use crate::call_control::TelnyxMediaConfig;
use crate::media::{
    packetize_tts_samples, CallMediaHandle, OutboundMediaCommand, OutboundMediaFrame,
    SharedMediaRegistry, SpeechCancelToken,
};
use crate::operator::state::{LogLevel, SharedState};
use crate::tts::{split_speech_text, LiveTtsBackend, SharedTtsRegistry, PIPER_SAMPLE_RATE_HZ};

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
    let Some(synthesis) = synthesize_utterance_with_fallback(
        &job.state,
        &job.tts,
        job.tts_backend,
        &job.gateway_call_id,
        &job.playback_id,
        &text_chunks,
        &job.cancel,
    )
    .await?
    else {
        return Ok(SpeechJobOutcome::NoAudioOrCanceled);
    };
    if job.cancel.is_canceled() {
        return Ok(SpeechJobOutcome::NoAudioOrCanceled);
    }

    let packets =
        packetize_tts_samples(synthesis.samples_i16, synthesis.sample_rate_hz, job.media)?;
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
        model_chunks = synthesis.chunk_count,
        model_sample_rate_hz = synthesis.sample_rate_hz,
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

struct SynthesizedUtterance {
    samples_i16: Vec<i16>,
    sample_rate_hz: u32,
    chunk_count: usize,
}

async fn synthesize_utterance_with_fallback(
    state: &SharedState,
    tts: &SharedTtsRegistry,
    backend: LiveTtsBackend,
    gateway_call_id: &str,
    playback_id: &str,
    text_chunks: &[String],
    cancel: &SpeechCancelToken,
) -> anyhow::Result<Option<SynthesizedUtterance>> {
    match synthesize_utterance(tts, backend, text_chunks, cancel).await {
        Ok(synthesis) => Ok(synthesis),
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
                        "TTS backend {} failed for {gateway_call_id} playback={playback_id}; re-synthesizing utterance with {}: {error_message}",
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
            synthesize_utterance(tts, fallback, text_chunks, cancel)
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

async fn synthesize_utterance(
    tts: &SharedTtsRegistry,
    backend: LiveTtsBackend,
    text_chunks: &[String],
    cancel: &SpeechCancelToken,
) -> anyhow::Result<Option<SynthesizedUtterance>> {
    let factory = tts.factory(backend);
    let mut samples_i16 = Vec::new();
    let mut sample_rate_hz = None;
    let mut chunk_count = 0usize;
    for text_chunk in text_chunks {
        if cancel.is_canceled() {
            return Ok(None);
        }
        let audio_chunks = factory.synthesize_chunks(text_chunk.clone()).await?;
        for audio_chunk in audio_chunks {
            if cancel.is_canceled() {
                return Ok(None);
            }
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
            chunk_count = chunk_count.saturating_add(1);
        }
    }
    if samples_i16.is_empty() || cancel.is_canceled() {
        return Ok(None);
    }
    Ok(Some(SynthesizedUtterance {
        samples_i16,
        sample_rate_hz: sample_rate_hz.unwrap_or(PIPER_SAMPLE_RATE_HZ),
        chunk_count,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::shared_state;
    use crate::tts::{OutboundTtsFactory, TtsAudio, TtsRegistry};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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

    #[tokio::test]
    async fn kokoro_mid_utterance_failure_restarts_whole_utterance_with_piper() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let kokoro = Arc::new(SequencedTtsFactory::new(
            24_000,
            2_400,
            Some(2),
            "kokoro mid utterance failure",
        ));
        let piper = Arc::new(SequencedTtsFactory::new(
            PIPER_SAMPLE_RATE_HZ,
            2_205,
            None,
            "unused",
        ));
        let tts = Arc::new(TtsRegistry::new(kokoro.clone(), piper.clone()));
        let cancel = SpeechCancelToken::default();
        let text_chunks = vec!["hello,".to_string(), "world.".to_string()];

        let synthesis = synthesize_utterance_with_fallback(
            &state,
            &tts,
            LiveTtsBackend::Kokoro82m,
            "gwc_test",
            "tts_test",
            &text_chunks,
            &cancel,
        )
        .await
        .expect("fallback should synthesize with Piper")
        .expect("fallback should return audio");

        assert_eq!(kokoro.calls(), 2);
        assert_eq!(piper.calls(), 2);
        assert_eq!(synthesis.sample_rate_hz, PIPER_SAMPLE_RATE_HZ);
        assert_eq!(synthesis.chunk_count, 2);
        assert_eq!(synthesis.samples_i16.len(), 4_410);
        let guard = state.read().await;
        assert!(guard.logs.iter().any(|entry| {
            entry.level == LogLevel::Warn
                && entry
                    .message
                    .contains("re-synthesizing utterance with piper")
                && entry.message.contains("kokoro mid utterance failure")
        }));
    }
}
