//! tts_v0.3 — Fish Speech vertical slice: synthesize text and write a `.wav` file.
//!
//! Usage:
//!   cargo run -p motlie-models --example models_tts_v0_3 \
//!     --no-default-features --features model-fish-speech-1_5 \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_model::{ArtifactPolicy, PcmEncoding, SpeechParams, SpeechRequest, StartOptions};
use motlie_models::tts::TtsModels;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: String,
    wav_path: PathBuf,
    artifact_root: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;

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
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text: text.context("--text <value> is required")?,
        wav_path: wav_path.context("--wav <path> is required")?,
        artifact_root,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie tts_v0.3 — Fish Speech synthesis ===");
    println!("wav:  {}", args.wav_path.display());

    let bundle = TtsModels::FishSpeech1_5.bundle();
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: args
                    .artifact_root
                    .unwrap_or_else(motlie_models::default_artifact_root),
            }),
            ..Default::default()
        })
        .await
        .context("failed to start fish speech bundle")?;

    let speech = handle
        .speech()
        .context("bundle should expose speech capability")?;
    let mut stream = speech
        .open_stream(SpeechRequest {
            text: args.text,
            params: SpeechParams::default(),
            conditioning: None,
        })
        .await
        .context("failed to open speech stream")?;

    let audio_spec = stream.audio_spec().clone();
    let mut writer = open_wav_writer(&args.wav_path, &audio_spec)?;
    let mut total_bytes = 0usize;

    while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
        writer
            .write_sample_iter(bytes_to_samples(&chunk.data, audio_spec.encoding)?)
            .context("failed to write wav samples")?;
        total_bytes += chunk.data.len();
        if chunk.end_of_stream {
            break;
        }
    }

    writer.finalize().context("failed to finalize wav file")?;
    stream.finish().await.context("finish failed")?;
    handle.shutdown().await.context("shutdown failed")?;

    println!(
        "wrote {} bytes of {:?} PCM at {} Hz to {}",
        total_bytes,
        audio_spec.encoding,
        audio_spec.sample_rate_hz,
        args.wav_path.display()
    );
    Ok(())
}

fn open_wav_writer(
    path: &PathBuf,
    spec: &motlie_model::AudioSpec,
) -> Result<hound::WavWriter<std::io::BufWriter<std::fs::File>>> {
    let sample_format = match spec.encoding {
        PcmEncoding::S16Le => hound::SampleFormat::Int,
        PcmEncoding::F32Le => hound::SampleFormat::Float,
    };
    let bits_per_sample = match spec.encoding {
        PcmEncoding::S16Le => 16,
        PcmEncoding::F32Le => 32,
    };

    hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: spec.channels,
            sample_rate: spec.sample_rate_hz,
            bits_per_sample,
            sample_format,
        },
    )
    .with_context(|| format!("failed to create wav file `{}`", path.display()))
}

fn bytes_to_samples(data: &[u8], encoding: PcmEncoding) -> Result<SampleIter<'_>> {
    match encoding {
        PcmEncoding::S16Le => {
            if !data.len().is_multiple_of(2) {
                bail!("S16Le chunk length {} is not divisible by 2", data.len());
            }
            Ok(SampleIter::I16(data.chunks_exact(2)))
        }
        PcmEncoding::F32Le => {
            if !data.len().is_multiple_of(4) {
                bail!("F32Le chunk length {} is not divisible by 4", data.len());
            }
            Ok(SampleIter::F32(data.chunks_exact(4)))
        }
    }
}

enum SampleIter<'a> {
    I16(std::slice::ChunksExact<'a, u8>),
    F32(std::slice::ChunksExact<'a, u8>),
}

impl<'a> SampleIter<'a> {
    fn write_into(
        self,
        writer: &mut hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    ) -> Result<()> {
        match self {
            Self::I16(chunks) => {
                for chunk in chunks {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    writer.write_sample(sample)?;
                }
            }
            Self::F32(chunks) => {
                for chunk in chunks {
                    let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    writer.write_sample(sample)?;
                }
            }
        }
        Ok(())
    }
}

trait WavWriterExt {
    fn write_sample_iter(&mut self, iter: SampleIter<'_>) -> Result<()>;
}

impl WavWriterExt for hound::WavWriter<std::io::BufWriter<std::fs::File>> {
    fn write_sample_iter(&mut self, iter: SampleIter<'_>) -> Result<()> {
        iter.write_into(self)
    }
}
