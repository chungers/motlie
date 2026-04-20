//! asr_moonshine — Moonshine ASR slice through the typed streaming API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_moonshine \
//!     --no-default-features --features model-moonshine-streaming \
//!     -- --wav path/to/audio.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{AudioBuf, Mono, StreamingTranscriber, TranscriptionSession};
use motlie_model::{ArtifactPolicy, StartOptions, TranscriptionParams};
use motlie_models::asr::moonshine_streaming_en;

#[path = "../asr_support.rs"]
mod asr_support;
#[path = "../audio_support.rs"]
mod audio_support;

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
}

fn parse_args() -> Result<Args> {
    let mut wav_path = None;
    let mut artifact_root = None;

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
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path,
        artifact_root,
    })
}

async fn run(args: Args) -> Result<()> {
    asr_support::log_status("=== motlie asr_moonshine — typed Moonshine streaming ASR ===");
    let input = asr_support::open_asr_input(args.wav_path)?;
    asr_support::log_status(&format!(
        "wav: {}",
        asr_support::describe_input(&input.source)
    ));

    let audio = decode_wav_to_i16_mono16k(input.reader)?;

    let handle = moonshine_streaming_en::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed Moonshine bundle")?;

    let mut session = handle
        .open_session(TranscriptionParams {
            language: Some("en".into()),
            emit_partials: false,
        })
        .await
        .context("failed to open typed Moonshine session")?;

    for chunk in audio.into_samples().chunks(DEMO_CHUNK_SAMPLES) {
        if let Some(update) = session
            .ingest(AudioBuf::<i16, TARGET_SAMPLE_RATE_HZ, Mono>::new(
                chunk.to_vec(),
            ))
            .await
            .context("typed Moonshine ingest failed")?
        {
            for segment in update.segments {
                println!(
                    "[partial] [{:.2}s - {:.2}s] {}",
                    segment.start_ms as f64 / 1000.0,
                    segment.end_ms as f64 / 1000.0,
                    segment.text
                );
            }
        }
    }

    let final_update = session.finish().await.context("finish failed")?;
    if final_update.segments.is_empty() {
        println!("transcript: <empty>");
    } else {
        for segment in final_update.segments {
            println!(
                "[final] [{:.2}s - {:.2}s] {}",
                segment.start_ms as f64 / 1000.0,
                segment.end_ms as f64 / 1000.0,
                segment.text
            );
        }
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
