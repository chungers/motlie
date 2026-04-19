//! tts_qwen3_tts_cpp — qwen3-tts.cpp synthesis to `.wav`.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_qwen3_tts_cpp \
//!     --no-default-features --features model-qwen3-tts-cpp \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{
    AudioBuf, CloneReference, Mono, SpeechStream, SpeechSynthesizer, SynthesisRequest,
    VoiceCloneSynthesizer,
};
use motlie_model::{ArtifactPolicy, SpeechParams, StartOptions};
use motlie_models::tts::qwen3_tts_cpp;

const TARGET_SAMPLE_RATE_HZ: u32 = 24_000;
const REFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: String,
    wav_path: PathBuf,
    artifact_root: Option<PathBuf>,
    reference_audio: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut reference_audio = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--text" => {
                text = Some(args.next().context("--text requires a value")?);
            }
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
            "--reference-audio" => {
                reference_audio = Some(PathBuf::from(
                    args.next()
                        .context("--reference-audio requires a path argument")?,
                ));
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text: text.context("--text <value> is required")?,
        wav_path: wav_path.context("--wav <path> is required")?,
        artifact_root,
        reference_audio,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie tts_qwen3_tts_cpp — typed qwen3-tts.cpp synthesis ===");
    println!("wav: {}", args.wav_path.display());

    let handle = qwen3_tts_cpp::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed qwen3-tts.cpp bundle")?;

    let request = SynthesisRequest {
        text: args.text,
        params: SpeechParams::default(),
    };

    let mut stream = if let Some(reference_audio) = &args.reference_audio {
        println!("reference: {}", reference_audio.display());
        let reader = hound::WavReader::open(reference_audio).with_context(|| {
            format!(
                "failed to open reference audio `{}`",
                reference_audio.display()
            )
        })?;
        let reference = decode_wav_to_reference(reader)?;
        handle
            .synthesize_with_reference(
                request,
                CloneReference::<REFERENCE_SAMPLE_RATE_HZ, Mono> {
                    audio: reference,
                    transcript: None,
                },
            )
            .await
            .context("failed to open typed cloned speech stream")?
    } else {
        handle
            .synthesize(request)
            .await
            .context("failed to open typed speech stream")?
    };

    let mut writer = hound::WavWriter::create(
        &args.wav_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE_HZ,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .with_context(|| format!("failed to create wav file `{}`", args.wav_path.display()))?;

    let mut total_samples = 0usize;
    while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
        total_samples += chunk.samples().len();
        for sample in chunk.into_samples() {
            writer
                .write_sample(sample)
                .context("failed to write wav sample")?;
        }
    }

    writer.finalize().context("failed to finalize wav file")?;
    stream.finish().await.context("finish failed")?;
    handle.shutdown().await.context("shutdown failed")?;

    println!(
        "wrote {} mono f32 samples at {} Hz to {}",
        total_samples,
        TARGET_SAMPLE_RATE_HZ,
        args.wav_path.display()
    );
    Ok(())
}

fn decode_wav_to_reference(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<AudioBuf<f32, REFERENCE_SAMPLE_RATE_HZ, Mono>> {
    let spec = reader.spec();
    let samples = decode_wav_to_f32(reader)?;
    let mono = downmix_to_mono(&samples, spec.channels);
    let resampled = resample_linear_f32(&mono, spec.sample_rate, REFERENCE_SAMPLE_RATE_HZ);
    Ok(AudioBuf::new(resampled))
}

fn decode_wav_to_f32(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<Vec<f32>> {
    match reader.spec().sample_format {
        hound::SampleFormat::Int => Ok(reader
            .into_samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to decode integer wav samples")?
            .into_iter()
            .map(|sample| sample as f32 / 32768.0)
            .collect()),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to decode float wav samples"),
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
