//! asr_sherpa_onnx — streaming Zipformer ASR via the typed streaming API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_sherpa_onnx \
//!     --no-default-features --features model-sherpa-onnx-streaming \
//!     -- --wav path/to/audio.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{AudioBuf, Mono, StreamingTranscriber, TranscriptionSession};
use motlie_model::{ArtifactPolicy, StartOptions, TranscriptionParams};
use motlie_models::asr::sherpa_onnx_streaming_en;

#[path = "../asr_support.rs"]
mod asr_support;
#[path = "../audio_support.rs"]
mod audio_support;
#[path = "../quiet_support.rs"]
mod quiet_support;

const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
const DEMO_CHUNK_SAMPLES: usize = 3_200;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    wav_path: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    quiet: bool,
    partials: bool,
}

fn parse_args() -> Result<Args> {
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut quiet = false;
    let mut partials = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wav" => {
                wav_path = Some(PathBuf::from(
                    args.next().context("--wav requires a path argument")?,
                ));
            }
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(
                    args.next()
                        .context("--artifact-root requires a path argument")?,
                ));
            }
            "--quiet" => quiet = true,
            "--partials" => partials = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path,
        artifact_root,
        quiet,
        partials,
    })
}

async fn run(args: Args) -> Result<()> {
    asr_support::log_status(
        args.quiet,
        "=== motlie asr_sherpa_onnx — typed sherpa-onnx streaming ASR ===",
    );
    let input = asr_support::open_asr_input(args.wav_path)?;
    asr_support::log_status(
        args.quiet,
        &format!("wav: {}", asr_support::describe_input(&input.source)),
    );

    let audio = decode_wav_to_i16_mono16k(input.reader)?;
    let _quiet_stderr = quiet_support::QuietStderrGuard::maybe_enable(args.quiet)
        .context("failed to enable quiet stderr mode")?;

    let handle = sherpa_onnx_streaming_en::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed sherpa-onnx bundle")?;

    let mut session = handle
        .open_session(TranscriptionParams {
            language: Some("en".into()),
            emit_partials: args.partials,
        })
        .await
        .context("failed to open typed sherpa-onnx session")?;

    let mut final_segments = Vec::new();
    for chunk in audio.into_samples().chunks(DEMO_CHUNK_SAMPLES) {
        if let Some(update) = session
            .ingest(AudioBuf::<i16, TARGET_SAMPLE_RATE_HZ, Mono>::new(
                chunk.to_vec(),
            ))
            .await
            .context("typed sherpa ingest failed")?
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
        asr_support::print_plain_transcript(&final_segments);
    }

    handle.shutdown().await.context("shutdown failed")?;
    Ok(())
}

fn decode_wav_to_i16_mono16k<R: std::io::Read>(
    reader: hound::WavReader<R>,
) -> Result<AudioBuf<i16, TARGET_SAMPLE_RATE_HZ, Mono>> {
    let (spec, samples) = audio_support::decode_wav_to_f32(reader)?;
    let mono = audio_support::downmix_to_mono(&samples, spec.channels);
    let resampled =
        audio_support::resample_linear_f32(&mono, spec.sample_rate, TARGET_SAMPLE_RATE_HZ);
    Ok(AudioBuf::new(
        resampled
            .into_iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .collect(),
    ))
}

fn print_segment_events(segments: &[motlie_model::TranscriptSegment]) {
    for segment in segments {
        let marker = if segment.final_segment {
            "[final]"
        } else {
            "[partial]"
        };
        println!(
            "{marker} [{:.2}s - {:.2}s] {}",
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.text
        );
    }
}
