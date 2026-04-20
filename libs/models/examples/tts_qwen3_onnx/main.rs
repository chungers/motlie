//! tts_qwen3_onnx — Qwen3-TTS synthesis with optional voice cloning.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_qwen3_onnx \
//!     --no-default-features --features model-qwen3-tts-0_6b \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{
    AudioBuf, CloneReference, Mono, SpeechStream, SpeechSynthesizer, SynthesisRequest,
    VoiceCloneSynthesizer,
};
use motlie_model::{ArtifactPolicy, SpeechParams, StartOptions};
use motlie_models::tts::qwen3_tts_12hz_0_6b;

#[path = "../audio_support.rs"]
mod audio_support;
#[path = "../tts_support.rs"]
mod tts_support;

const TARGET_SAMPLE_RATE_HZ: u32 = 24_000;
const REFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: Option<String>,
    wav_path: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    reference_audio: Option<PathBuf>,
    reference_text: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut reference_audio = None;
    let mut reference_text = None;

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
            "--reference-text" => {
                reference_text = Some(args.next().context("--reference-text requires a value")?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text,
        wav_path,
        artifact_root,
        reference_audio,
        reference_text,
    })
}

async fn run(args: Args) -> Result<()> {
    let io = tts_support::resolve_text_and_output(args.text, args.wav_path)?;
    tts_support::log_status("=== motlie tts_qwen3_onnx — typed Qwen3-TTS speech synthesis ===");
    match &io.output {
        tts_support::TtsOutput::WavFile(path) => {
            tts_support::log_status(&format!("wav:  {}", path.display()));
        }
        tts_support::TtsOutput::Stdout => {
            tts_support::log_status("wav:  <stdout>");
        }
    }

    let handle = qwen3_tts_12hz_0_6b::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed qwen3-tts bundle")?;

    let request = SynthesisRequest {
        text: io.text,
        params: SpeechParams::default(),
    };

    let mut stream = if let Some(ref_path) = &args.reference_audio {
        tts_support::log_status(&format!("reference: {}", ref_path.display()));
        let (_, reader) = audio_support::open_wav_reader(Some(ref_path.as_path()))?;
        let reference = decode_wav_to_reference(reader)?;
        handle
            .synthesize_with_reference(
                request,
                CloneReference::<REFERENCE_SAMPLE_RATE_HZ, Mono> {
                    audio: reference,
                    transcript: args.reference_text.clone(),
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

    let mut samples = Vec::new();
    while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
        samples.extend(chunk.into_samples());
    }

    tts_support::write_wav(&io.output, TARGET_SAMPLE_RATE_HZ, &samples)?;
    stream.finish().await.context("finish failed")?;
    handle.shutdown().await.context("shutdown failed")?;

    tts_support::log_status(&format!(
        "wrote {} mono f32 samples at {} Hz to {}",
        samples.len(),
        TARGET_SAMPLE_RATE_HZ,
        match &io.output {
            tts_support::TtsOutput::WavFile(path) => path.display().to_string(),
            tts_support::TtsOutput::Stdout => "<stdout>".into(),
        }
    ));
    Ok(())
}

fn decode_wav_to_reference<R: std::io::Read>(
    reader: hound::WavReader<R>,
) -> Result<AudioBuf<f32, REFERENCE_SAMPLE_RATE_HZ, Mono>> {
    let (spec, samples) = audio_support::decode_wav_to_f32(reader)?;
    let mono = audio_support::downmix_to_mono(&samples, spec.channels);
    let resampled =
        audio_support::resample_linear_f32(&mono, spec.sample_rate, REFERENCE_SAMPLE_RATE_HZ);
    Ok(AudioBuf::new(resampled))
}
