//! tts_v0.2 — Qwen3-TTS vertical slice: synthesize text with optional voice cloning.
//!
//! Usage:
//!   cargo run -p motlie-models --example models_tts_v0_2 \
//!     --no-default-features --features model-qwen3-tts-0_6b \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav
//!
//! With voice cloning (3-second reference audio):
//!   cargo run -p motlie-models --example models_tts_v0_2 \
//!     --no-default-features --features model-qwen3-tts-0_6b \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav \
//!        --reference-audio /path/to/reference.wav
//!
//! Preconditions:
//!   - The Qwen3-TTS ONNX-exported model components (encoder.onnx, decoder.onnx,
//!     vocoder.onnx, config.json) must be pre-exported and cached under the
//!     artifact root. See DESIGN_TTS.md Phase 2 for the export procedure.
//!   - A compatible ONNX Runtime installation must be discoverable.
//!     Set `ORT_LIB_PATH` if not installed system-wide.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::{
    ArtifactPolicy, AudioSpec, PcmEncoding, SpeechParams, SpeechRequest, StartOptions,
    VoiceConditioning,
};
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
        text: text.context("--text <value> is required")?,
        wav_path: wav_path.context("--wav <path> is required")?,
        artifact_root,
        reference_audio,
        reference_text,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie tts_v0.2 — Qwen3-TTS speech synthesis ===");
    println!("wav:  {}", args.wav_path.display());

    let bundle = TtsModels::Qwen3Tts12Hz0_6B.bundle();
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
        .context("failed to start qwen3-tts bundle")?;

    // Build conditioning from reference audio if provided.
    let conditioning = if let Some(ref_path) = &args.reference_audio {
        println!("reference: {}", ref_path.display());
        let reader = hound::WavReader::open(ref_path)
            .with_context(|| format!("failed to open reference audio: {}", ref_path.display()))?;
        let ref_spec = reader.spec();
        let (pcm, encoding) = decode_wav(reader)?;
        Some(VoiceConditioning::ReferenceAudio {
            audio_spec: AudioSpec {
                sample_rate_hz: ref_spec.sample_rate,
                channels: ref_spec.channels,
                encoding,
                preferred_chunk_bytes: 0,
            },
            pcm,
            reference_text: args.reference_text.clone(),
        })
    } else {
        None
    };

    let speech = handle
        .speech()
        .context("bundle should expose speech capability")?;
    let mut stream = speech
        .open_stream(SpeechRequest {
            text: args.text,
            params: SpeechParams::default(),
            conditioning,
        })
        .await
        .context("failed to open speech stream")?;

    let audio_spec = stream.audio_spec().clone();
    let mut writer = open_wav_writer(&args.wav_path, &audio_spec)?;
    let mut total_bytes = 0usize;

    while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
        write_pcm_samples(&mut writer, &chunk.data, audio_spec.encoding)?;
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
    spec: &AudioSpec,
) -> Result<hound::WavWriter<std::io::BufWriter<std::fs::File>>> {
    let (sample_format, bits_per_sample) = match spec.encoding {
        PcmEncoding::S16Le => (hound::SampleFormat::Int, 16),
        PcmEncoding::F32Le => (hound::SampleFormat::Float, 32),
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

fn write_pcm_samples(
    writer: &mut hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    data: &[u8],
    encoding: PcmEncoding,
) -> Result<()> {
    match encoding {
        PcmEncoding::S16Le => {
            for chunk in data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                writer.write_sample(sample)?;
            }
        }
        PcmEncoding::F32Le => {
            for chunk in data.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                writer.write_sample(sample)?;
            }
        }
    }
    Ok(())
}

fn decode_wav(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<(Vec<u8>, PcmEncoding)> {
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode wav samples")?;
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            Ok((bytes, PcmEncoding::S16Le))
        }
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode wav samples")?;
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            Ok((bytes, PcmEncoding::F32Le))
        }
    }
}
