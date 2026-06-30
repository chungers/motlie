use std::future::Future;
use std::path::PathBuf;

use anyhow::{Context, Result};
use motlie_model::typed::{AudioBuf, Mono, StreamingTranscriber, TranscriptionSession};
use motlie_model::{
    ArtifactPolicy, BundleHandle, ModelError, StartOptions, TranscriptSegment, TranscriptionParams,
};
use motlie_voice::pipeline::convert::f32_to_i16_clamped;

use crate::{asr_support, bundle_support, quiet_support};

/// @codex-tts 2026-04-21 -- Feed streaming ASR sessions in 200 ms windows at 16 kHz.
pub const STREAMING_ASR_CHUNK_SAMPLES: usize = 3_200;

pub struct StreamingAsrArgs {
    pub wav_path: Option<PathBuf>,
    pub artifact_root: Option<PathBuf>,
    pub quiet: bool,
    pub partials: bool,
}

pub fn decode_f32_to_i16_mono16k(
    spec: hound::WavSpec,
    samples: Vec<f32>,
) -> Result<AudioBuf<i16, { asr_support::TARGET_SAMPLE_RATE_HZ }, Mono>> {
    let audio = asr_support::decode_f32_to_f32_mono16k(spec, samples)?;
    Ok(AudioBuf::new(f32_to_i16_clamped(audio.samples())))
}

pub async fn run_streaming_asr<Handle, Start, StartFuture>(
    args: StreamingAsrArgs,
    banner: &'static str,
    start_context: &'static str,
    open_session_context: &'static str,
    ingest_context: &'static str,
    start: Start,
) -> Result<()>
where
    Handle: BundleHandle
        + StreamingTranscriber<Input = AudioBuf<i16, { asr_support::TARGET_SAMPLE_RATE_HZ }, Mono>>,
    Start: FnOnce(StartOptions) -> StartFuture,
    StartFuture: Future<Output = std::result::Result<Handle, ModelError>>,
{
    asr_support::log_status(args.quiet, banner);
    let input = asr_support::open_asr_input(args.wav_path)?;
    asr_support::log_status(
        args.quiet,
        &format!("wav: {}", asr_support::describe_input(&input.source)),
    );
    let audio = decode_f32_to_i16_mono16k(input.spec, input.samples)?;
    let _quiet_stderr = quiet_support::QuietStderrGuard::maybe_enable(args.quiet)
        .context("failed to enable quiet stderr mode")?;

    let handle = start(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .with_context(|| start_context.to_string())?;

    let final_segments = bundle_support::run_with_shutdown(handle, |handle| {
        Box::pin(async move {
            let mut session = handle
                .open_session(TranscriptionParams {
                    language: Some("en".into()),
                    emit_partials: args.partials,
                })
                .await
                .with_context(|| open_session_context.to_string())?;

            let mut final_segments = Vec::new();
            for chunk in audio.into_samples().chunks(STREAMING_ASR_CHUNK_SAMPLES) {
                if let Some(update) = session
                    .ingest(
                        AudioBuf::<i16, { asr_support::TARGET_SAMPLE_RATE_HZ }, Mono>::new(
                            chunk.to_vec(),
                        ),
                    )
                    .await
                    .with_context(|| ingest_context.to_string())?
                {
                    if args.partials {
                        print_segment_events(&update.segments);
                    } else {
                        final_segments.extend(
                            update
                                .segments
                                .into_iter()
                                .filter(|segment| segment.final_segment),
                        );
                    }
                }
            }

            let final_update = session.finish().await.context("finish failed")?;
            if args.partials {
                print_segment_events(&final_update.segments);
            } else {
                final_segments.extend(final_update.segments);
            }

            Ok(final_segments)
        })
    })
    .await?;

    if !args.partials {
        asr_support::print_plain_transcript(&final_segments);
    }

    Ok(())
}

pub fn print_segment_events(segments: &[TranscriptSegment]) {
    for segment in segments {
        let marker = if segment.final_segment {
            "[final]"
        } else {
            "[partial]"
        };
        let confidence = segment
            .confidence
            .map(|confidence| format!(" confidence={confidence:.3}"))
            .unwrap_or_default();
        println!(
            "{marker} [{:.2}s - {:.2}s] {}{}",
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.text,
            confidence
        );
    }
}
