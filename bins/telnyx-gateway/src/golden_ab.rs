use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;

use anyhow::{bail, Context};
use motlie_voice::pipeline::convert::{downmix_to_mono, f32_to_i16_clamped};
use motlie_voice::pipeline::resample::{resample_i16_mono, WindowedSincResampler};
use motlie_voice::telephony::{round_trip_telnyx_asr_samples, TelnyxAsrAudioSpec};
use motlie_voice::wav::decode_streaming_wav_to_f32;
use serde::{Deserialize, Serialize};

use crate::cli::{
    AsrGoldenAbArgs, GoldenCodecArg, GoldenTtsArgs, TtsGoldenAbArgs, TtsGoldenEngineArg,
};
use crate::replay::{compute_wer, replay_samples, ReplayBackend, ReplayLatencyReport, WerReport};
#[cfg(feature = "piper")]
use crate::tts::OutboundTtsFactory;

#[cfg(feature = "qwen3-tts-cpp")]
use motlie_model::typed::{BufferedSpeechSynthesizer, SynthesisRequest};
#[cfg(feature = "qwen3-tts-cpp")]
use motlie_model::{ArtifactPolicy, ModelError, SpeechParams, StartOptions};

const SOURCE_SAMPLE_RATE_HZ: u32 = 16_000;
const SOURCE_CHANNELS: u16 = 1;
const ALL_CATEGORIES: &str = "ALL";

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GoldenAbManifest {
    schema_version: u32,
    samples: Vec<GoldenAbSample>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GoldenAbSample {
    id: String,
    category: String,
    text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GoldenTtsReport {
    pub manifest_path: String,
    pub output_dir: String,
    pub generated: usize,
    pub skipped: usize,
    pub samples: Vec<GoldenTtsSampleReport>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GoldenTtsSampleReport {
    pub id: String,
    pub category: String,
    pub text: String,
    pub wav_path: String,
    pub status: String,
    pub sample_rate_hz: u32,
    pub sample_count: usize,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GoldenAbReport {
    pub manifest_path: String,
    pub audio_dir: String,
    pub chunk_ms: u32,
    pub trailing_silence_pad_ms: u32,
    pub entries: Vec<GoldenAbEntryReport>,
    pub summaries: Vec<GoldenAbSummaryReport>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GoldenAbEntryReport {
    pub id: String,
    pub category: String,
    pub text: String,
    pub codec: String,
    pub media_sample_rate_hz: u32,
    pub backend: String,
    pub source_wav: String,
    pub asr_wav: String,
    pub sample_count: usize,
    pub transcript: String,
    pub wer: WerReport,
    pub latency: ReplayLatencyReport,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GoldenAbSummaryReport {
    pub backend: String,
    pub codec: String,
    pub category: String,
    pub sample_count: usize,
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub errors: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub wer_percent: f64,
    pub audio_ms: u64,
    pub trailing_silence_pad_avg_ms: f64,
    pub trailing_silence_pad_chunks: usize,
    pub ingest_avg_ms: f64,
    pub finish_avg_ms: f64,
    pub wall_avg_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TtsGoldenAbReport {
    pub manifest_path: String,
    pub output_dir: String,
    pub asr_backend: String,
    pub chunk_ms: u32,
    pub trailing_silence_pad_ms: u32,
    pub entries: Vec<TtsGoldenAbEntryReport>,
    pub failures: Vec<TtsGoldenAbFailureReport>,
    pub summaries: Vec<TtsGoldenAbSummaryReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TtsGoldenAbEntryReport {
    pub id: String,
    pub category: String,
    pub text: String,
    pub engine: String,
    pub codec: String,
    pub media_sample_rate_hz: u32,
    pub asr_backend: String,
    pub source_wav: String,
    pub asr_wav: String,
    pub tts_elapsed_ms: u128,
    pub tts_realtime_factor: f64,
    pub objective: TtsObjectiveAudioReport,
    pub transcript: String,
    pub wer: WerReport,
    pub latency: ReplayLatencyReport,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TtsObjectiveAudioReport {
    pub sample_rate_hz: u32,
    pub sample_count: usize,
    pub audio_ms: u64,
    pub reference_words: usize,
    pub speaking_rate_wpm: f64,
    pub clipped_samples: usize,
    pub clipping_percent: f64,
    pub peak_abs: i32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TtsGoldenAbFailureReport {
    pub id: String,
    pub category: String,
    pub text: String,
    pub engine: String,
    pub codec: Option<String>,
    pub status: String,
    pub error: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TtsGoldenAbSummaryReport {
    pub engine: String,
    pub codec: String,
    pub category: String,
    pub sample_count: usize,
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub errors: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub wer_percent: f64,
    pub audio_ms: u64,
    pub tts_elapsed_avg_ms: f64,
    pub tts_realtime_factor_avg: f64,
    pub speaking_rate_wpm_avg: f64,
    pub clipping_percent_avg: f64,
    pub asr_wall_avg_ms: f64,
}

#[cfg(feature = "qwen3-tts-cpp")]
pub async fn generate_tts_wavs(
    args: &GoldenTtsArgs,
    artifact_root: PathBuf,
    allow_download: bool,
) -> anyhow::Result<GoldenTtsReport> {
    let manifest = load_manifest(&args.manifest)?;
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create golden TTS output dir {}", args.output_dir.display()))?;

    let handle = start_qwen3_tts(&artifact_root, allow_download).await?;
    let mut reports = Vec::new();
    let mut generated = 0usize;
    let mut skipped = 0usize;

    for sample in selected_samples(&manifest.samples, args.limit) {
        let wav_path = sample_wav_path(&args.output_dir, &sample.id);
        if wav_path.exists() && !args.force {
            let (sample_rate_hz, samples) = read_mono_i16_wav(&wav_path)?;
            skipped += 1;
            reports.push(GoldenTtsSampleReport {
                id: sample.id.clone(),
                category: sample.category.clone(),
                text: sample.text.clone(),
                wav_path: wav_path.display().to_string(),
                status: "skipped".to_string(),
                sample_rate_hz,
                sample_count: samples.len(),
                elapsed_ms: 0,
            });
            continue;
        }

        let started_at = Instant::now();
        let request = SynthesisRequest {
            text: sample.text.clone(),
            params: SpeechParams::default(),
        };
        let audio = handle
            .synthesize_buffered(request)
            .await
            .with_context(|| format!("synthesize Qwen3-TTS sample {}", sample.id))?;
        let resampled = resample_i16_mono(
            &WindowedSincResampler::default(),
            &f32_to_i16_clamped(audio.samples()),
            audio.sample_rate_hz(),
            SOURCE_SAMPLE_RATE_HZ,
        )?;
        write_i16_wav(&wav_path, SOURCE_SAMPLE_RATE_HZ, &resampled)
            .with_context(|| format!("write golden TTS WAV {}", wav_path.display()))?;
        generated += 1;
        reports.push(GoldenTtsSampleReport {
            id: sample.id.clone(),
            category: sample.category.clone(),
            text: sample.text.clone(),
            wav_path: wav_path.display().to_string(),
            status: "generated".to_string(),
            sample_rate_hz: SOURCE_SAMPLE_RATE_HZ,
            sample_count: resampled.len(),
            elapsed_ms: started_at.elapsed().as_millis(),
        });
    }

    Ok(GoldenTtsReport {
        manifest_path: args.manifest.display().to_string(),
        output_dir: args.output_dir.display().to_string(),
        generated,
        skipped,
        samples: reports,
    })
}

#[cfg(not(feature = "qwen3-tts-cpp"))]
pub async fn generate_tts_wavs(
    _args: &GoldenTtsArgs,
    _artifact_root: PathBuf,
    _allow_download: bool,
) -> anyhow::Result<GoldenTtsReport> {
    bail!("golden-tts requires rebuilding telnyx-gateway with --features qwen3-tts-cpp or --features golden-ab")
}

pub async fn run_golden_ab(
    args: &AsrGoldenAbArgs,
    backends: Vec<ReplayBackend>,
) -> anyhow::Result<GoldenAbReport> {
    if args.chunk_ms == 0 {
        bail!("--chunk-ms must be greater than zero");
    }
    if backends.is_empty() {
        bail!("at least one ASR backend is required");
    }

    let manifest = load_manifest(&args.manifest)?;
    let codecs = args.selected_codecs();
    let mut entries = Vec::new();

    for sample in selected_samples(&manifest.samples, args.limit) {
        let source_wav = sample_wav_path(&args.audio_dir, &sample.id);
        let (source_rate_hz, source_samples) = read_mono_i16_wav(&source_wav)
            .with_context(|| format!("read source golden WAV for sample {}", sample.id))?;

        for codec in &codecs {
            let spec = telnyx_spec(*codec);
            let asr_samples = round_trip_telnyx_asr_samples(&source_samples, source_rate_hz, spec)
                .with_context(|| {
                    format!("round trip sample {} through {}", sample.id, codec.label())
                })?;
            let asr_wav = asr_wav_path(&args.audio_dir, codec.label(), &sample.id);
            write_i16_wav(&asr_wav, SOURCE_SAMPLE_RATE_HZ, &asr_samples)
                .with_context(|| format!("write ASR input WAV {}", asr_wav.display()))?;

            for backend in &backends {
                let run = replay_samples(
                    &asr_samples,
                    args.chunk_ms,
                    args.trailing_silence_pad_ms,
                    backend.asr(),
                )
                .await
                .with_context(|| {
                    format!(
                        "run sample {} codec {} backend {}",
                        sample.id,
                        codec.label(),
                        backend.label()
                    )
                })?;
                let wer = compute_wer(&sample.text, &run.transcript);
                entries.push(GoldenAbEntryReport {
                    id: sample.id.clone(),
                    category: sample.category.clone(),
                    text: sample.text.clone(),
                    codec: codec.label().to_string(),
                    media_sample_rate_hz: spec.media_sample_rate_hz(),
                    backend: backend.label().to_string(),
                    source_wav: source_wav.display().to_string(),
                    asr_wav: asr_wav.display().to_string(),
                    sample_count: asr_samples.len(),
                    transcript: run.transcript,
                    wer,
                    latency: run.latency,
                });
            }
        }
    }

    let summaries = summarize_entries(&entries);
    let report = GoldenAbReport {
        manifest_path: args.manifest.display().to_string(),
        audio_dir: args.audio_dir.display().to_string(),
        chunk_ms: args.chunk_ms,
        trailing_silence_pad_ms: args.trailing_silence_pad_ms,
        entries,
        summaries,
    };

    if let Some(path) = &args.output_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create report dir {}", parent.display()))?;
            }
        }
        let file =
            File::create(path).with_context(|| format!("create report {}", path.display()))?;
        serde_json::to_writer_pretty(file, &report)
            .with_context(|| format!("write report {}", path.display()))?;
    }

    Ok(report)
}

pub async fn run_tts_golden_ab(
    args: &TtsGoldenAbArgs,
    asr_backend: ReplayBackend,
    artifact_root: PathBuf,
    allow_download: bool,
) -> anyhow::Result<TtsGoldenAbReport> {
    if args.chunk_ms == 0 {
        bail!("--chunk-ms must be greater than zero");
    }

    let manifest = load_manifest(&args.manifest)?;
    std::fs::create_dir_all(&args.output_dir).with_context(|| {
        format!(
            "create TTS golden A/B output dir {}",
            args.output_dir.display()
        )
    })?;

    let engines = args.selected_engines();
    let codecs = args.selected_codecs();
    let mut runner = TtsGoldenRunner::new(artifact_root, allow_download);
    let mut entries = Vec::new();
    let mut failures = Vec::new();

    for engine in engines {
        for sample in selected_samples(&manifest.samples, args.limit) {
            let synthesis = match runner.synthesize(engine, args, sample).await {
                Ok(synthesis) => synthesis,
                Err(error) => {
                    failures.push(TtsGoldenAbFailureReport {
                        id: sample.id.clone(),
                        category: sample.category.clone(),
                        text: sample.text.clone(),
                        engine: engine.label().to_string(),
                        codec: None,
                        status: "synthesis_failed".to_string(),
                        error: error.to_string(),
                    });
                    continue;
                }
            };

            let objective = objective_audio_report(&sample.text, &synthesis.samples_16k);
            for codec in &codecs {
                match run_tts_golden_codec(
                    args,
                    sample,
                    engine,
                    *codec,
                    &synthesis,
                    &objective,
                    &asr_backend,
                )
                .await
                {
                    Ok(entry) => entries.push(entry),
                    Err(error) => failures.push(TtsGoldenAbFailureReport {
                        id: sample.id.clone(),
                        category: sample.category.clone(),
                        text: sample.text.clone(),
                        engine: engine.label().to_string(),
                        codec: Some(codec.label().to_string()),
                        status: "asr_replay_failed".to_string(),
                        error: error.to_string(),
                    }),
                }
            }
        }
    }

    let summaries = summarize_tts_entries(&entries);
    let report = TtsGoldenAbReport {
        manifest_path: args.manifest.display().to_string(),
        output_dir: args.output_dir.display().to_string(),
        asr_backend: asr_backend.label().to_string(),
        chunk_ms: args.chunk_ms,
        trailing_silence_pad_ms: args.trailing_silence_pad_ms,
        entries,
        failures,
        summaries,
    };

    if let Some(path) = &args.output_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("create TTS golden A/B report dir {}", parent.display())
                })?;
            }
        }
        let file =
            File::create(path).with_context(|| format!("create report {}", path.display()))?;
        serde_json::to_writer_pretty(file, &report)
            .with_context(|| format!("write report {}", path.display()))?;
    }

    Ok(report)
}

async fn run_tts_golden_codec(
    args: &TtsGoldenAbArgs,
    sample: &GoldenAbSample,
    engine: TtsGoldenEngineArg,
    codec: GoldenCodecArg,
    synthesis: &TtsGoldenSynthesis,
    objective: &TtsObjectiveAudioReport,
    asr_backend: &ReplayBackend,
) -> anyhow::Result<TtsGoldenAbEntryReport> {
    let spec = telnyx_spec(codec);
    let asr_samples =
        round_trip_telnyx_asr_samples(&synthesis.samples_16k, SOURCE_SAMPLE_RATE_HZ, spec)
            .with_context(|| {
                format!(
                    "round trip TTS sample {} engine {} through {}",
                    sample.id,
                    engine.label(),
                    codec.label()
                )
            })?;
    let asr_wav = tts_asr_wav_path(&args.output_dir, engine, codec, &sample.id);
    write_i16_wav(&asr_wav, SOURCE_SAMPLE_RATE_HZ, &asr_samples)
        .with_context(|| format!("write TTS golden ASR input WAV {}", asr_wav.display()))?;

    let run = replay_samples(
        &asr_samples,
        args.chunk_ms,
        args.trailing_silence_pad_ms,
        asr_backend.asr(),
    )
    .await
    .with_context(|| {
        format!(
            "run fixed ASR for sample {} engine {} codec {} backend {}",
            sample.id,
            engine.label(),
            codec.label(),
            asr_backend.label()
        )
    })?;
    let wer = compute_wer(&sample.text, &run.transcript);
    Ok(TtsGoldenAbEntryReport {
        id: sample.id.clone(),
        category: sample.category.clone(),
        text: sample.text.clone(),
        engine: engine.label().to_string(),
        codec: codec.label().to_string(),
        media_sample_rate_hz: spec.media_sample_rate_hz(),
        asr_backend: asr_backend.label().to_string(),
        source_wav: synthesis.source_wav.display().to_string(),
        asr_wav: asr_wav.display().to_string(),
        tts_elapsed_ms: synthesis.elapsed_ms,
        tts_realtime_factor: realtime_factor(synthesis.elapsed_ms, objective.audio_ms),
        objective: objective.clone(),
        transcript: run.transcript,
        wer,
        latency: run.latency,
    })
}

struct TtsGoldenSynthesis {
    source_wav: PathBuf,
    samples_16k: Vec<i16>,
    elapsed_ms: u128,
}

struct TtsGoldenRunner {
    #[cfg(any(feature = "piper", feature = "qwen3-tts-cpp"))]
    artifact_root: PathBuf,
    #[cfg(any(feature = "piper", feature = "qwen3-tts-cpp"))]
    allow_download: bool,
    #[cfg(feature = "piper")]
    piper: Option<std::sync::Arc<crate::tts::PiperTtsFactory>>,
    #[cfg(feature = "qwen3-tts-cpp")]
    qwen3: Option<motlie_model_qwen3_tts_cpp::Qwen3TtsCppHandle>,
}

impl TtsGoldenRunner {
    fn new(artifact_root: PathBuf, allow_download: bool) -> Self {
        #[cfg(not(any(feature = "piper", feature = "qwen3-tts-cpp")))]
        let _ = (artifact_root, allow_download);
        Self {
            #[cfg(any(feature = "piper", feature = "qwen3-tts-cpp"))]
            artifact_root,
            #[cfg(any(feature = "piper", feature = "qwen3-tts-cpp"))]
            allow_download,
            #[cfg(feature = "piper")]
            piper: None,
            #[cfg(feature = "qwen3-tts-cpp")]
            qwen3: None,
        }
    }

    async fn synthesize(
        &mut self,
        engine: TtsGoldenEngineArg,
        args: &TtsGoldenAbArgs,
        sample: &GoldenAbSample,
    ) -> anyhow::Result<TtsGoldenSynthesis> {
        let source_wav = tts_source_wav_path(&args.output_dir, engine, &sample.id);
        match engine {
            TtsGoldenEngineArg::Piper => self.synthesize_piper(sample, source_wav).await,
            TtsGoldenEngineArg::Kokoro82m => synthesize_kokoro(args, sample, source_wav).await,
            TtsGoldenEngineArg::Qwen3TtsCpp => self.synthesize_qwen3(sample, source_wav).await,
        }
    }

    #[cfg(feature = "piper")]
    async fn synthesize_piper(
        &mut self,
        sample: &GoldenAbSample,
        source_wav: PathBuf,
    ) -> anyhow::Result<TtsGoldenSynthesis> {
        if self.piper.is_none() {
            self.piper = Some(std::sync::Arc::new(crate::tts::PiperTtsFactory::new(
                self.artifact_root.clone(),
                self.allow_download,
            )));
        }
        let piper = self
            .piper
            .as_ref()
            .context("Piper handle was not initialized")?;
        piper
            .warm()
            .await
            .context("warm Piper TTS model before timing sample")?;
        let started_at = Instant::now();
        let chunks = piper
            .synthesize_chunks(sample.text.clone())
            .await
            .with_context(|| format!("synthesize Piper sample {}", sample.id))?;
        let (sample_rate_hz, samples) = concatenate_tts_chunks(chunks)?;
        let samples_16k = normalize_to_source_rate(sample_rate_hz, samples)?;
        write_i16_wav(&source_wav, SOURCE_SAMPLE_RATE_HZ, &samples_16k)
            .with_context(|| format!("write Piper source WAV {}", source_wav.display()))?;
        Ok(TtsGoldenSynthesis {
            source_wav,
            samples_16k,
            elapsed_ms: started_at.elapsed().as_millis(),
        })
    }

    #[cfg(not(feature = "piper"))]
    async fn synthesize_piper(
        &mut self,
        _sample: &GoldenAbSample,
        _source_wav: PathBuf,
    ) -> anyhow::Result<TtsGoldenSynthesis> {
        bail!("Piper TTS is unavailable; rebuild with --features piper or --features golden-ab")
    }

    #[cfg(feature = "qwen3-tts-cpp")]
    async fn synthesize_qwen3(
        &mut self,
        sample: &GoldenAbSample,
        source_wav: PathBuf,
    ) -> anyhow::Result<TtsGoldenSynthesis> {
        if self.qwen3.is_none() {
            self.qwen3 = Some(start_qwen3_tts(&self.artifact_root, self.allow_download).await?);
        }
        let handle = self
            .qwen3
            .as_ref()
            .context("Qwen3-TTS handle was not initialized")?;
        let started_at = Instant::now();
        let request = SynthesisRequest {
            text: sample.text.clone(),
            params: SpeechParams::default(),
        };
        let audio = handle
            .synthesize_buffered(request)
            .await
            .with_context(|| format!("synthesize Qwen3-TTS sample {}", sample.id))?;
        let samples = f32_to_i16_clamped(audio.samples());
        let samples_16k = normalize_to_source_rate(audio.sample_rate_hz(), samples)?;
        write_i16_wav(&source_wav, SOURCE_SAMPLE_RATE_HZ, &samples_16k)
            .with_context(|| format!("write Qwen3-TTS source WAV {}", source_wav.display()))?;
        Ok(TtsGoldenSynthesis {
            source_wav,
            samples_16k,
            elapsed_ms: started_at.elapsed().as_millis(),
        })
    }

    #[cfg(not(feature = "qwen3-tts-cpp"))]
    async fn synthesize_qwen3(
        &mut self,
        _sample: &GoldenAbSample,
        _source_wav: PathBuf,
    ) -> anyhow::Result<TtsGoldenSynthesis> {
        bail!("Qwen3-TTS is unavailable; rebuild with --features qwen3-tts-cpp or --features golden-ab")
    }
}

#[cfg(feature = "piper")]
fn concatenate_tts_chunks(chunks: Vec<crate::tts::TtsAudio>) -> anyhow::Result<(u32, Vec<i16>)> {
    let mut chunks = chunks.into_iter();
    let first = chunks
        .next()
        .context("TTS engine returned no audio chunks")?;
    let sample_rate_hz = first.sample_rate_hz();
    let mut samples = first.into_samples_i16();
    for chunk in chunks {
        if chunk.sample_rate_hz() != sample_rate_hz {
            bail!(
                "TTS chunks used mixed sample rates: {} and {}",
                sample_rate_hz,
                chunk.sample_rate_hz()
            );
        }
        samples.extend_from_slice(chunk.samples_i16());
    }
    Ok((sample_rate_hz, samples))
}

async fn synthesize_kokoro(
    args: &TtsGoldenAbArgs,
    sample: &GoldenAbSample,
    source_wav: PathBuf,
) -> anyhow::Result<TtsGoldenSynthesis> {
    let template = args.kokoro_command.as_deref().context(
        "Kokoro-82M command not configured; pass --kokoro-command with {text} and {output} placeholders",
    )?;
    if let Some(parent) = source_wav.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create Kokoro output dir {}", parent.display()))?;
    }
    let command = render_kokoro_command(template, &sample.text, &source_wav)?;
    let program = command
        .first()
        .context("Kokoro command template produced no program")?;
    let started_at = Instant::now();
    let status = tokio::process::Command::new(program)
        .args(&command[1..])
        .stdin(Stdio::null())
        .status()
        .await
        .with_context(|| format!("run Kokoro command for sample {}", sample.id))?;
    if !status.success() {
        bail!(
            "Kokoro command failed for sample {} with status {}",
            sample.id,
            status
        );
    }
    let (sample_rate_hz, samples) = read_mono_i16_wav(&source_wav)
        .with_context(|| format!("read Kokoro output WAV {}", source_wav.display()))?;
    let samples_16k = normalize_to_source_rate(sample_rate_hz, samples)?;
    write_i16_wav(&source_wav, SOURCE_SAMPLE_RATE_HZ, &samples_16k).with_context(|| {
        format!(
            "write normalized Kokoro source WAV {}",
            source_wav.display()
        )
    })?;
    Ok(TtsGoldenSynthesis {
        source_wav,
        samples_16k,
        elapsed_ms: started_at.elapsed().as_millis(),
    })
}

fn render_kokoro_command(template: &str, text: &str, output: &Path) -> anyhow::Result<Vec<String>> {
    let parts = shlex::split(template).context("parse --kokoro-command template")?;
    if parts.is_empty() {
        bail!("--kokoro-command must include a program");
    }
    let output = output.display().to_string();
    let sample_rate = SOURCE_SAMPLE_RATE_HZ.to_string();
    let mut saw_text = false;
    let mut saw_output = false;
    let rendered = parts
        .into_iter()
        .map(|part| {
            if part.contains("{text}") {
                saw_text = true;
            }
            if part.contains("{output}") {
                saw_output = true;
            }
            part.replace("{text}", text)
                .replace("{output}", &output)
                .replace("{sample_rate}", &sample_rate)
        })
        .collect::<Vec<_>>();
    if !saw_text || !saw_output {
        bail!("--kokoro-command must include both {text} and {output} placeholders");
    }
    Ok(rendered)
}

fn normalize_to_source_rate(sample_rate_hz: u32, samples: Vec<i16>) -> anyhow::Result<Vec<i16>> {
    if samples.is_empty() {
        bail!("TTS engine returned empty audio");
    }
    if sample_rate_hz == SOURCE_SAMPLE_RATE_HZ {
        Ok(samples)
    } else {
        resample_i16_mono(
            &WindowedSincResampler::default(),
            &samples,
            sample_rate_hz,
            SOURCE_SAMPLE_RATE_HZ,
        )
        .context("resample TTS output to 16 kHz mono")
    }
}

fn objective_audio_report(text: &str, samples: &[i16]) -> TtsObjectiveAudioReport {
    let audio_ms = audio_duration_ms(samples.len(), SOURCE_SAMPLE_RATE_HZ);
    let reference_words = compute_wer(text, "").reference_words;
    let clipped_samples = samples
        .iter()
        .filter(|sample| **sample == i16::MAX || **sample == i16::MIN)
        .count();
    let peak_abs = samples
        .iter()
        .map(|sample| i32::from(*sample).abs())
        .max()
        .unwrap_or(0);
    TtsObjectiveAudioReport {
        sample_rate_hz: SOURCE_SAMPLE_RATE_HZ,
        sample_count: samples.len(),
        audio_ms,
        reference_words,
        speaking_rate_wpm: speaking_rate_wpm(reference_words, audio_ms),
        clipped_samples,
        clipping_percent: percent(clipped_samples, samples.len()),
        peak_abs,
    }
}

fn speaking_rate_wpm(words: usize, audio_ms: u64) -> f64 {
    if audio_ms == 0 {
        0.0
    } else {
        words as f64 * 60_000.0 / audio_ms as f64
    }
}

fn realtime_factor(elapsed_ms: u128, audio_ms: u64) -> f64 {
    if audio_ms == 0 {
        0.0
    } else {
        elapsed_ms as f64 / audio_ms as f64
    }
}

fn audio_duration_ms(sample_count: usize, sample_rate_hz: u32) -> u64 {
    if sample_rate_hz == 0 {
        0
    } else {
        (sample_count as u64 * 1000) / u64::from(sample_rate_hz)
    }
}

fn tts_source_wav_path(output_dir: &Path, engine: TtsGoldenEngineArg, id: &str) -> PathBuf {
    output_dir
        .join(engine_path_slug(engine))
        .join("source")
        .join(format!("{id}.wav"))
}

fn tts_asr_wav_path(
    output_dir: &Path,
    engine: TtsGoldenEngineArg,
    codec: GoldenCodecArg,
    id: &str,
) -> PathBuf {
    output_dir
        .join(engine_path_slug(engine))
        .join("asr-inputs")
        .join(codec.label().to_ascii_lowercase())
        .join(format!("{id}.wav"))
}

fn engine_path_slug(engine: TtsGoldenEngineArg) -> String {
    engine
        .label()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn summarize_tts_entries(entries: &[TtsGoldenAbEntryReport]) -> Vec<TtsGoldenAbSummaryReport> {
    let mut aggregates: BTreeMap<(String, String, String), TtsSummaryAccumulator> = BTreeMap::new();
    for entry in entries {
        for category in [ALL_CATEGORIES, entry.category.as_str()] {
            aggregates
                .entry((
                    entry.engine.clone(),
                    entry.codec.clone(),
                    category.to_string(),
                ))
                .or_default()
                .record(entry);
        }
    }

    aggregates
        .into_iter()
        .map(|((engine, codec, category), aggregate)| {
            aggregate.into_report(engine, codec, category)
        })
        .collect()
}

#[derive(Default)]
struct TtsSummaryAccumulator {
    sample_count: usize,
    reference_words: usize,
    hypothesis_words: usize,
    errors: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    audio_ms: u64,
    tts_elapsed_ms: u128,
    tts_realtime_factor: f64,
    speaking_rate_wpm: f64,
    clipping_percent: f64,
    asr_wall_ms: u128,
}

impl TtsSummaryAccumulator {
    fn record(&mut self, entry: &TtsGoldenAbEntryReport) {
        self.sample_count += 1;
        self.reference_words += entry.wer.reference_words;
        self.hypothesis_words += entry.wer.hypothesis_words;
        self.errors += entry.wer.errors;
        self.substitutions += entry.wer.substitutions;
        self.deletions += entry.wer.deletions;
        self.insertions += entry.wer.insertions;
        self.audio_ms += entry.objective.audio_ms;
        self.tts_elapsed_ms += entry.tts_elapsed_ms;
        self.tts_realtime_factor += entry.tts_realtime_factor;
        self.speaking_rate_wpm += entry.objective.speaking_rate_wpm;
        self.clipping_percent += entry.objective.clipping_percent;
        self.asr_wall_ms += entry.latency.wall_ms;
    }

    fn into_report(
        self,
        engine: String,
        codec: String,
        category: String,
    ) -> TtsGoldenAbSummaryReport {
        TtsGoldenAbSummaryReport {
            engine,
            codec,
            category,
            sample_count: self.sample_count,
            reference_words: self.reference_words,
            hypothesis_words: self.hypothesis_words,
            errors: self.errors,
            substitutions: self.substitutions,
            deletions: self.deletions,
            insertions: self.insertions,
            wer_percent: percent(self.errors, self.reference_words),
            audio_ms: self.audio_ms,
            tts_elapsed_avg_ms: average_u128(self.tts_elapsed_ms, self.sample_count),
            tts_realtime_factor_avg: average_f64(self.tts_realtime_factor, self.sample_count),
            speaking_rate_wpm_avg: average_f64(self.speaking_rate_wpm, self.sample_count),
            clipping_percent_avg: average_f64(self.clipping_percent, self.sample_count),
            asr_wall_avg_ms: average_u128(self.asr_wall_ms, self.sample_count),
        }
    }
}

fn average_f64(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn load_manifest(path: &Path) -> anyhow::Result<GoldenAbManifest> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read golden A/B manifest {}", path.display()))?;
    let manifest: GoldenAbManifest = serde_json::from_str(&raw)
        .with_context(|| format!("parse golden A/B manifest {}", path.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported golden A/B manifest schema_version {}; expected 1",
            manifest.schema_version
        );
    }
    if manifest.samples.is_empty() {
        bail!("golden A/B manifest must contain at least one sample");
    }
    Ok(manifest)
}

fn selected_samples(
    samples: &[GoldenAbSample],
    limit: Option<usize>,
) -> impl Iterator<Item = &GoldenAbSample> {
    samples.iter().take(limit.unwrap_or(usize::MAX))
}

fn sample_wav_path(audio_dir: &Path, id: &str) -> PathBuf {
    audio_dir.join(format!("{id}.wav"))
}

fn asr_wav_path(audio_dir: &Path, codec: &str, id: &str) -> PathBuf {
    audio_dir
        .join("asr-inputs")
        .join(codec.to_ascii_lowercase())
        .join(format!("{id}.wav"))
}

fn telnyx_spec(codec: GoldenCodecArg) -> TelnyxAsrAudioSpec {
    match codec {
        GoldenCodecArg::L16_16k => TelnyxAsrAudioSpec::L16_16k,
        GoldenCodecArg::Pcmu8k => TelnyxAsrAudioSpec::Pcmu8k,
    }
}

fn read_mono_i16_wav(path: &Path) -> anyhow::Result<(u32, Vec<i16>)> {
    let file = File::open(path).with_context(|| format!("open WAV {}", path.display()))?;
    let (spec, samples) = decode_streaming_wav_to_f32(file)
        .with_context(|| format!("decode WAV {}", path.display()))?;
    let mono = downmix_to_mono(&samples, spec.channels)?;
    Ok((spec.sample_rate, f32_to_i16_clamped(&mono)))
}

fn write_i16_wav(path: &Path, sample_rate_hz: u32, samples: &[i16]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create WAV dir {}", parent.display()))?;
        }
    }
    let spec = hound::WavSpec {
        channels: SOURCE_CHANNELS,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create WAV {}", path.display()))?;
    for sample in samples {
        writer
            .write_sample(*sample)
            .with_context(|| format!("write WAV sample to {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("finalize WAV {}", path.display()))?;
    Ok(())
}

fn summarize_entries(entries: &[GoldenAbEntryReport]) -> Vec<GoldenAbSummaryReport> {
    let mut aggregates: BTreeMap<(String, String, String), SummaryAccumulator> = BTreeMap::new();
    for entry in entries {
        for category in [ALL_CATEGORIES, entry.category.as_str()] {
            aggregates
                .entry((
                    entry.backend.clone(),
                    entry.codec.clone(),
                    category.to_string(),
                ))
                .or_default()
                .record(entry);
        }
    }

    aggregates
        .into_iter()
        .map(|((backend, codec, category), aggregate)| {
            aggregate.into_report(backend, codec, category)
        })
        .collect()
}

#[derive(Default)]
struct SummaryAccumulator {
    sample_count: usize,
    reference_words: usize,
    hypothesis_words: usize,
    errors: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    audio_ms: u64,
    trailing_silence_pad_ms: u64,
    trailing_silence_pad_chunks: usize,
    ingest_total_ms: u128,
    chunk_count: usize,
    finish_ms: u128,
    wall_ms: u128,
}

impl SummaryAccumulator {
    fn record(&mut self, entry: &GoldenAbEntryReport) {
        self.sample_count += 1;
        self.reference_words += entry.wer.reference_words;
        self.hypothesis_words += entry.wer.hypothesis_words;
        self.errors += entry.wer.errors;
        self.substitutions += entry.wer.substitutions;
        self.deletions += entry.wer.deletions;
        self.insertions += entry.wer.insertions;
        self.audio_ms += entry.latency.audio_ms;
        self.trailing_silence_pad_ms += u64::from(entry.latency.trailing_silence_pad_ms);
        self.trailing_silence_pad_chunks += entry.latency.trailing_silence_pad_chunks;
        self.ingest_total_ms += entry.latency.ingest_total_ms;
        self.chunk_count += entry.latency.ingest_chunk_count();
        self.finish_ms += entry.latency.finish_ms;
        self.wall_ms += entry.latency.wall_ms;
    }

    fn into_report(
        self,
        backend: String,
        codec: String,
        category: String,
    ) -> GoldenAbSummaryReport {
        GoldenAbSummaryReport {
            backend,
            codec,
            category,
            sample_count: self.sample_count,
            reference_words: self.reference_words,
            hypothesis_words: self.hypothesis_words,
            errors: self.errors,
            substitutions: self.substitutions,
            deletions: self.deletions,
            insertions: self.insertions,
            wer_percent: percent(self.errors, self.reference_words),
            audio_ms: self.audio_ms,
            trailing_silence_pad_avg_ms: average_u64(
                self.trailing_silence_pad_ms,
                self.sample_count,
            ),
            trailing_silence_pad_chunks: self.trailing_silence_pad_chunks,
            ingest_avg_ms: average_u128(self.ingest_total_ms, self.chunk_count),
            finish_avg_ms: average_u128(self.finish_ms, self.sample_count),
            wall_avg_ms: average_u128(self.wall_ms, self.sample_count),
        }
    }
}

fn percent(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 * 100.0 / denominator as f64
    }
}

fn average_u128(total: u128, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn average_u64(total: u64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

#[cfg(feature = "qwen3-tts-cpp")]
async fn start_qwen3_tts(
    artifact_root: &Path,
    allow_download: bool,
) -> anyhow::Result<motlie_model_qwen3_tts_cpp::Qwen3TtsCppHandle> {
    match motlie_models::tts::qwen3_tts_cpp::start_typed(local_only_options(artifact_root)).await {
        Ok(handle) => Ok(handle),
        Err(err) if allow_download && missing_local_artifacts(&err) => {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                artifact = "qwen3-tts-cpp-0.6b",
                "downloading Qwen3-TTS artifacts"
            );
            download_qwen3_tts_artifacts(artifact_root)?;
            motlie_models::tts::qwen3_tts_cpp::start_typed(local_only_options(artifact_root))
                .await
                .map_err(anyhow::Error::from)
                .context("start Qwen3-TTS after downloading artifacts")
        }
        Err(err) if !allow_download && missing_local_artifacts(&err) => {
            bail!(
                "{} missing for qwen3-tts-cpp-0.6b under '{}'; rerun without --no-asr-download or preinstall artifacts",
                motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX,
                artifact_root.display()
            )
        }
        Err(err) => Err(anyhow::Error::from(err)).context("start Qwen3-TTS"),
    }
}

#[cfg(feature = "qwen3-tts-cpp")]
fn local_only_options(artifact_root: &Path) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root.to_path_buf(),
        }),
        ..Default::default()
    }
}

#[cfg(feature = "qwen3-tts-cpp")]
fn missing_local_artifacts(error: &ModelError) -> bool {
    match error {
        ModelError::InvalidConfiguration(message) => {
            message.contains(motlie_models::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX)
        }
        _ => false,
    }
}

#[cfg(feature = "qwen3-tts-cpp")]
fn download_qwen3_tts_artifacts(artifact_root: &Path) -> anyhow::Result<()> {
    let catalog = motlie_models::Catalog::with_defaults();
    let bundle_id = motlie_models::tts::qwen3_tts_cpp::descriptor().id;
    motlie_models::download_bundle_artifacts(&catalog, &bundle_id, artifact_root)
        .map(|_| ())
        .map_err(anyhow::Error::from)
        .context("download Qwen3-TTS artifacts")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::adapter::EchoAsrFactory;
    use crate::cli::{AsrGoldenAbArgs, ReplayBackendArg, TtsGoldenAbArgs, TtsGoldenEngineArg};

    #[test]
    fn bundled_manifest_is_short_call_center_corpus() {
        let manifest_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/qwen3-call-center-golden.json");
        let manifest = load_manifest(&manifest_path).expect("manifest should parse");

        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.samples.len(), 72);
        assert!(manifest.samples.iter().all(|sample| {
            let words = sample.text.split_whitespace().count();
            (5..=20).contains(&words)
        }));
    }

    #[test]
    fn tts_golden_ab_defaults_include_baseline_and_candidates() {
        let args = TtsGoldenAbArgs {
            manifest: PathBuf::from("manifest.json"),
            output_dir: PathBuf::from("/tmp/motlie-tts-golden-ab-test"),
            engine: Vec::new(),
            codec: Vec::new(),
            asr_backend: ReplayBackendArg::SherpaZipformer2023,
            chunk_ms: 20,
            trailing_silence_pad_ms: crate::replay::DEFAULT_TRAILING_SILENCE_PAD_MS,
            limit: None,
            output_json: None,
            kokoro_command: None,
        };

        assert_eq!(
            args.selected_engines(),
            vec![
                TtsGoldenEngineArg::Piper,
                TtsGoldenEngineArg::Kokoro82m,
                TtsGoldenEngineArg::Qwen3TtsCpp,
            ]
        );
        assert_eq!(
            args.selected_codecs(),
            vec![GoldenCodecArg::L16_16k, GoldenCodecArg::Pcmu8k]
        );
    }

    #[test]
    fn kokoro_command_template_replaces_placeholders_without_shell() {
        let command = render_kokoro_command(
            "kokoro-cli --text {text} --output {output} --sample-rate {sample_rate}",
            "Hello from Motlie",
            Path::new("/tmp/out.wav"),
        )
        .expect("template should render");

        assert_eq!(
            command,
            vec![
                "kokoro-cli",
                "--text",
                "Hello from Motlie",
                "--output",
                "/tmp/out.wav",
                "--sample-rate",
                "16000",
            ]
        );
    }

    #[test]
    fn objective_audio_report_counts_rate_and_clipping() {
        let samples = vec![0; 16_000]
            .into_iter()
            .chain([i16::MAX, i16::MIN])
            .collect::<Vec<_>>();
        let report = objective_audio_report("hello world", &samples);

        assert_eq!(report.sample_rate_hz, SOURCE_SAMPLE_RATE_HZ);
        assert_eq!(report.reference_words, 2);
        assert_eq!(report.audio_ms, 1000);
        assert_eq!(report.clipped_samples, 2);
        assert_eq!(report.peak_abs, 32768);
        assert!((report.speaking_rate_wpm - 120.0).abs() < f64::EPSILON);
        assert!(report.clipping_percent > 0.0);
    }

    #[test]
    fn tts_summary_aggregates_by_engine_codec_and_category() {
        let entry = TtsGoldenAbEntryReport {
            id: "sample-1".to_string(),
            category: "digits".to_string(),
            text: "hello world".to_string(),
            engine: "kokoro-82m".to_string(),
            codec: "PCMU-8k".to_string(),
            media_sample_rate_hz: 8_000,
            asr_backend: "sherpa-zipformer-en-2023-06-26".to_string(),
            source_wav: "/tmp/source.wav".to_string(),
            asr_wav: "/tmp/asr.wav".to_string(),
            tts_elapsed_ms: 500,
            tts_realtime_factor: 0.5,
            objective: TtsObjectiveAudioReport {
                sample_rate_hz: SOURCE_SAMPLE_RATE_HZ,
                sample_count: 16_000,
                audio_ms: 1000,
                reference_words: 2,
                speaking_rate_wpm: 120.0,
                clipped_samples: 0,
                clipping_percent: 0.0,
                peak_abs: 1000,
            },
            transcript: "hello world".to_string(),
            wer: compute_wer("hello world", "hello world"),
            latency: ReplayLatencyReport {
                audio_ms: 1000,
                chunk_count: 50,
                trailing_silence_pad_ms: 800,
                trailing_silence_pad_chunks: 40,
                ingest_total_ms: 90,
                ingest_max_ms: 3,
                finish_ms: 10,
                wall_ms: 150,
            },
        };

        let summaries = summarize_tts_entries(&[entry]);

        assert!(summaries.iter().any(|summary| {
            summary.category == ALL_CATEGORIES
                && summary.sample_count == 1
                && summary.wer_percent == 0.0
                && summary.tts_elapsed_avg_ms == 500.0
        }));
        assert!(summaries.iter().any(|summary| summary.category == "digits"));
    }

    #[tokio::test]
    async fn golden_ab_runs_echo_backend_over_l16_fixture() {
        let root = std::env::temp_dir().join(format!("motlie-golden-ab-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp root");
        let manifest_path = root.join("manifest.json");
        std::fs::write(
            &manifest_path,
            r#"{
              "schema_version": 1,
              "samples": [
                {
                  "id": "echo-fixture",
                  "category": "fixture",
                  "text": "received 28800 samples"
                }
              ]
            }"#,
        )
        .expect("write manifest");
        let fixture_samples = vec![0; 16_000];
        write_i16_wav(
            &root.join("echo-fixture.wav"),
            SOURCE_SAMPLE_RATE_HZ,
            &fixture_samples,
        )
        .expect("write fixture wav");

        let args = AsrGoldenAbArgs {
            manifest: manifest_path,
            audio_dir: root.clone(),
            backend: vec![ReplayBackendArg::Echo],
            codec: vec![GoldenCodecArg::L16_16k],
            chunk_ms: 20,
            trailing_silence_pad_ms: crate::replay::DEFAULT_TRAILING_SILENCE_PAD_MS,
            limit: None,
            output_json: None,
        };
        let report = run_golden_ab(
            &args,
            vec![ReplayBackend::new("echo", Arc::new(EchoAsrFactory))],
        )
        .await
        .expect("golden A/B should run");

        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].wer.errors, 0);
        assert!(report
            .summaries
            .iter()
            .any(|summary| summary.category == ALL_CATEGORIES && summary.wer_percent == 0.0));
        std::fs::remove_dir_all(root).expect("remove temp root");
    }
}
