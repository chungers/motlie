//! asr_whisper — Whisper `.wav` transcription through the typed batch API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_whisper \
//!     --no-default-features --features model-whisper-base-en \
//!     -- --wav path/to/audio.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{AudioBuf, BatchTranscriber, Mono};
use motlie_model::{ArtifactPolicy, StartOptions, TranscriptionParams};
use motlie_models::asr::whisper_base_en;

const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;

fn main() -> Result<()> {
    let args = parse_args()?;

    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    wav_path: PathBuf,
    artifact_root: Option<PathBuf>,
    language: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut wav_path: Option<PathBuf> = None;
    let mut artifact_root: Option<PathBuf> = None;
    let mut language: Option<String> = None;

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
            "--language" => {
                language = Some(args.next().context("--language requires a language code")?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let wav_path = wav_path.context("--wav <path> is required")?;
    Ok(Args {
        wav_path,
        artifact_root,
        language,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie asr_whisper — typed Whisper batch transcription ===");
    println!("wav:   {}", args.wav_path.display());

    let reader = hound::WavReader::open(&args.wav_path)
        .with_context(|| format!("failed to open wav file: {}", args.wav_path.display()))?;
    let wav_spec = reader.spec();
    println!(
        "format: {} Hz, {} ch, {:?}, {} bits",
        wav_spec.sample_rate, wav_spec.channels, wav_spec.sample_format, wav_spec.bits_per_sample,
    );

    let audio = decode_wav_to_whisper_input(reader)?;

    let artifact_root = args
        .artifact_root
        .unwrap_or_else(motlie_models::default_artifact_root);
    println!("artifacts: {}", artifact_root.display());

    let handle = whisper_base_en::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root,
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed whisper bundle")?;

    let update = handle
        .transcribe(
            audio,
            TranscriptionParams {
                language: args.language,
                emit_partials: false,
            },
        )
        .await
        .context("typed whisper transcription failed")?;

    for segment in &update.segments {
        println!(
            "[final] [{:.1}s - {:.1}s] {}",
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.text.trim()
        );
    }

    handle.shutdown().await.context("shutdown failed")?;

    Ok(())
}

fn decode_wav_to_whisper_input(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<AudioBuf<f32, TARGET_SAMPLE_RATE_HZ, Mono>> {
    let spec = reader.spec();
    let samples = decode_wav_to_f32(reader)?;
    let mono = downmix_to_mono(&samples, spec.channels);
    let resampled = resample_linear_f32(&mono, spec.sample_rate, TARGET_SAMPLE_RATE_HZ);
    Ok(AudioBuf::new(resampled))
}

fn decode_wav_to_f32(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<Vec<f32>> {
    match reader.spec().sample_format {
        hound::SampleFormat::Int => Ok(reader
            .into_samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to decode wav samples")?
            .into_iter()
            .map(|sample| sample as f32 / 32768.0)
            .collect()),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to decode wav samples"),
    }
}

fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    let channels = channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear_f32(samples: &[f32], input_rate_hz: u32, output_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate_hz == output_rate_hz {
        return samples.to_vec();
    }

    let ratio = input_rate_hz as f64 / output_rate_hz as f64;
    let out_len =
        ((samples.len() as f64) * output_rate_hz as f64 / input_rate_hz as f64).ceil() as usize;
    let max_index = samples.len().saturating_sub(1);
    let mut output = Vec::with_capacity(out_len.max(1));

    for out_idx in 0..out_len {
        let src_pos = out_idx as f64 * ratio;
        let left_idx = src_pos.floor() as usize;
        let right_idx = (left_idx + 1).min(max_index);
        let frac = (src_pos - left_idx as f64) as f32;
        let left = samples[left_idx.min(max_index)];
        let right = samples[right_idx];
        output.push(left + (right - left) * frac);
    }

    output
}
