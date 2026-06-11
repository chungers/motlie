//! tts_qwen3_tts_cpp — qwen3-tts.cpp synthesis to `.wav`.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_qwen3_tts_cpp \
//!     --no-default-features --features model-qwen3-tts-cpp \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_model::typed::{
    AudioBuf, CloneReference, Mono, SpeechStream, SpeechSynthesizer, SynthesisRequest,
    VoiceCloneSynthesizer,
};
use motlie_model::{ArtifactPolicy, SpeechParams, StartOptions};
use motlie_models::tts::qwen3_tts_cpp;
use motlie_voice::pipeline::convert::{decode_samples_to_f32, downmix_to_mono};
use motlie_voice::pipeline::resample::{LinearInterpolator, Resampler};

#[path = "../support/audio.rs"]
mod audio_support;
#[path = "../support/bundle.rs"]
mod bundle_support;
#[path = "../support/quiet.rs"]
mod quiet_support;
#[path = "../support/tts.rs"]
mod tts_support;

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
    quiet: bool,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut reference_audio = None;
    let mut quiet = false;

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
            "--quiet" => quiet = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text,
        wav_path,
        artifact_root,
        reference_audio,
        quiet,
    })
}

async fn run(args: Args) -> Result<()> {
    let io = tts_support::resolve_text_and_output(args.text, args.wav_path)?;
    tts_support::log_status(
        args.quiet,
        "=== motlie tts_qwen3_tts_cpp — typed qwen3-tts.cpp synthesis ===",
    );
    match &io.output {
        tts_support::TtsOutput::WavFile(path) => {
            tts_support::log_status(args.quiet, &format!("wav: {}", path.display()));
        }
        tts_support::TtsOutput::Stdout => {
            tts_support::log_status(args.quiet, "wav: <stdout>");
        }
    }
    let _quiet_stderr = quiet_support::QuietStderrGuard::maybe_enable(args.quiet)
        .context("failed to enable quiet stderr mode")?;

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
        text: io.text,
        params: SpeechParams::default(),
    };

    let output_label = match &io.output {
        tts_support::TtsOutput::WavFile(path) => path.display().to_string(),
        tts_support::TtsOutput::Stdout => "<stdout>".into(),
    };
    let (sample_count, sample_rate_hz) = bundle_support::run_with_shutdown(handle, |handle| {
        Box::pin(async move {
            let mut stream = if let Some(reference_audio) = &args.reference_audio {
                tts_support::log_status(
                    args.quiet,
                    &format!("reference: {}", reference_audio.display()),
                );
                let (_, reader) = audio_support::open_wav_reader(Some(reference_audio.as_path()))?;
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

            let first_chunk = stream
                .next_chunk()
                .await
                .context("next_chunk failed")?
                .context("typed qwen3-tts.cpp stream produced no audio chunks")?;
            let sample_rate_hz = first_chunk.sample_rate_hz();
            let mut sample_count = 0usize;
            let mut sink = tts_support::WavSink::<f32>::new(&io.output, sample_rate_hz)?;

            let first_samples = first_chunk.into_samples();
            sample_count += first_samples.len();
            sink.write_chunk(&first_samples)?;

            while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
                let samples = chunk.into_samples();
                sample_count += samples.len();
                sink.write_chunk(&samples)?;
            }

            stream.finish().await.context("finish failed")?;
            sink.finalize()?;
            Ok((sample_count, sample_rate_hz))
        })
    })
    .await?;

    tts_support::log_status(
        args.quiet,
        &format!(
            "wrote {} mono f32 samples at {} Hz to {}",
            sample_count, sample_rate_hz, output_label
        ),
    );
    Ok(())
}

fn decode_wav_to_reference<R: std::io::Read>(
    reader: hound::WavReader<R>,
) -> Result<AudioBuf<f32, REFERENCE_SAMPLE_RATE_HZ, Mono>> {
    let (spec, samples) =
        decode_samples_to_f32(reader).context("failed to decode reference wav samples")?;
    let mono =
        downmix_to_mono(&samples, spec.channels).context("failed to downmix reference wav")?;
    let resampled = LinearInterpolator
        .resample_f32(&mono, spec.sample_rate, REFERENCE_SAMPLE_RATE_HZ)
        .context("failed to resample reference wav to 16 kHz")?;
    Ok(AudioBuf::new(resampled))
}
