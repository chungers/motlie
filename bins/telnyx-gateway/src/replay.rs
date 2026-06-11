use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context};
use motlie_model::typed::{AudioBuf, Mono};
use motlie_voice::app::TranscriptEvent;
use motlie_voice::pipeline::convert::{downmix_to_mono, f32_to_i16_clamped};
use motlie_voice::wav::decode_streaming_wav_to_f32;
use serde::{Deserialize, Serialize};

use crate::adapter::{AsrTranscriptEvent, InboundAsrSession, SharedAsrFactory};
use crate::cli::{ReplayCaptureArgs, ReplayCorpusArgs};

const ASR_INPUT_WAV: &str = "asr-input-16khz.wav";
const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_TRAILING_SILENCE_PAD_MS: u32 = 800;

#[derive(Clone)]
pub struct ReplayBackend {
    label: String,
    asr: SharedAsrFactory,
}

impl ReplayBackend {
    pub fn new(label: impl Into<String>, asr: SharedAsrFactory) -> Self {
        Self {
            label: label.into(),
            asr,
        }
    }

    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn asr(&self) -> SharedAsrFactory {
        self.asr.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CorpusReplayReport {
    pub manifest_path: String,
    pub entries: Vec<CorpusEntryReplayReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CorpusEntryReplayReport {
    pub id: String,
    pub description: Option<String>,
    pub codec: String,
    pub media_sample_rate_hz: u32,
    pub direction: String,
    pub capture_dir: String,
    pub wav_path: String,
    pub baseline: Option<CorpusBaselineReport>,
    pub notes: Option<String>,
    pub reports: Vec<ReplayReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CorpusBaselineReport {
    pub backend: String,
    pub wer_percent: f64,
    pub errors: usize,
    pub reference_words: usize,
    pub hypothesis_words: Option<usize>,
    pub substitutions: Option<usize>,
    pub deletions: Option<usize>,
    pub insertions: Option<usize>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReplayReport {
    pub backend: String,
    pub capture_dir: String,
    pub wav_path: String,
    pub wav_sample_rate_hz: u32,
    pub wav_channels: u16,
    pub sample_count: usize,
    pub chunk_ms: u32,
    pub transcript: String,
    pub wer: Option<WerReport>,
    pub latency: ReplayLatencyReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReplayLatencyReport {
    pub audio_ms: u64,
    pub chunk_count: usize,
    pub trailing_silence_pad_ms: u32,
    pub trailing_silence_pad_chunks: usize,
    pub ingest_total_ms: u128,
    pub ingest_max_ms: u128,
    pub finish_ms: u128,
    pub wall_ms: u128,
}

impl ReplayLatencyReport {
    pub fn ingest_chunk_count(&self) -> usize {
        self.chunk_count + self.trailing_silence_pad_chunks
    }

    pub fn ingest_avg_ms(&self) -> f64 {
        let ingest_chunk_count = self.ingest_chunk_count();
        if ingest_chunk_count == 0 {
            return 0.0;
        }
        self.ingest_total_ms as f64 / ingest_chunk_count as f64
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WerReport {
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub errors: usize,
    pub errors_by_token: Vec<WerTokenError>,
}

impl WerReport {
    pub fn rate(&self) -> f64 {
        if self.reference_words == 0 {
            return 0.0;
        }
        self.errors as f64 / self.reference_words as f64
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum WerTokenError {
    Substitution {
        reference: String,
        hypothesis: String,
    },
    Deletion {
        reference: String,
    },
    Insertion {
        hypothesis: String,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GoldenCorpusManifest {
    schema_version: u32,
    entries: Vec<GoldenCorpusEntry>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GoldenCorpusEntry {
    id: String,
    description: Option<String>,
    capture_dir: PathBuf,
    asr_input_wav: Option<PathBuf>,
    reference_text: Option<String>,
    reference_file: Option<PathBuf>,
    codec: String,
    media_sample_rate_hz: u32,
    direction: String,
    baseline: Option<GoldenCorpusBaseline>,
    notes: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GoldenCorpusBaseline {
    backend: String,
    wer_percent: f64,
    errors: usize,
    reference_words: usize,
    hypothesis_words: Option<usize>,
    substitutions: Option<usize>,
    deletions: Option<usize>,
    insertions: Option<usize>,
    source: Option<String>,
}

impl From<&GoldenCorpusBaseline> for CorpusBaselineReport {
    fn from(baseline: &GoldenCorpusBaseline) -> Self {
        Self {
            backend: baseline.backend.clone(),
            wer_percent: baseline.wer_percent,
            errors: baseline.errors,
            reference_words: baseline.reference_words,
            hypothesis_words: baseline.hypothesis_words,
            substitutions: baseline.substitutions,
            deletions: baseline.deletions,
            insertions: baseline.insertions,
            source: baseline.source.clone(),
        }
    }
}

pub async fn replay_capture(
    args: &ReplayCaptureArgs,
    backend_label: &str,
    asr: SharedAsrFactory,
) -> anyhow::Result<ReplayReport> {
    if args.chunk_ms == 0 {
        bail!("--chunk-ms must be greater than zero");
    }

    let wav_path = args.capture_dir.join(ASR_INPUT_WAV);
    let reference = reference_text(args)?;
    replay_capture_wav(
        &args.capture_dir,
        &wav_path,
        reference.as_deref(),
        args.chunk_ms,
        args.trailing_silence_pad_ms,
        backend_label,
        asr,
    )
    .await
}

pub async fn replay_corpus(
    args: &ReplayCorpusArgs,
    backends: Vec<ReplayBackend>,
) -> anyhow::Result<CorpusReplayReport> {
    if args.chunk_ms == 0 {
        bail!("--chunk-ms must be greater than zero");
    }
    if backends.is_empty() {
        bail!("at least one replay backend is required");
    }

    let manifest = load_manifest(&args.manifest)?;
    let manifest_dir = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let mut entries = Vec::new();

    for entry in &manifest.entries {
        let capture_dir = resolve_manifest_path(manifest_dir, &entry.capture_dir);
        let wav_path = resolve_entry_wav_path(&capture_dir, entry);
        let reference = entry_reference_text(entry, manifest_dir)?;
        let mut reports = Vec::new();

        for backend in &backends {
            let report = replay_capture_wav(
                &capture_dir,
                &wav_path,
                Some(&reference),
                args.chunk_ms,
                args.trailing_silence_pad_ms,
                &backend.label,
                backend.asr.clone(),
            )
            .await
            .with_context(|| {
                format!(
                    "replay corpus entry '{}' with backend '{}'",
                    entry.id, backend.label
                )
            })?;
            reports.push(report);
        }

        entries.push(CorpusEntryReplayReport {
            id: entry.id.clone(),
            description: entry.description.clone(),
            codec: entry.codec.clone(),
            media_sample_rate_hz: entry.media_sample_rate_hz,
            direction: entry.direction.clone(),
            capture_dir: capture_dir.display().to_string(),
            wav_path: wav_path.display().to_string(),
            baseline: entry.baseline.as_ref().map(CorpusBaselineReport::from),
            notes: entry.notes.clone(),
            reports,
        });
    }

    Ok(CorpusReplayReport {
        manifest_path: args.manifest.display().to_string(),
        entries,
    })
}

fn load_manifest(path: &Path) -> anyhow::Result<GoldenCorpusManifest> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read golden corpus manifest {}", path.display()))?;
    let manifest: GoldenCorpusManifest = serde_json::from_str(&raw)
        .with_context(|| format!("parse golden corpus manifest {}", path.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported golden corpus manifest schema_version {}; expected 1",
            manifest.schema_version
        );
    }
    if manifest.entries.is_empty() {
        bail!("golden corpus manifest must contain at least one entry");
    }
    Ok(manifest)
}

fn entry_reference_text(entry: &GoldenCorpusEntry, manifest_dir: &Path) -> anyhow::Result<String> {
    if let Some(reference) = &entry.reference_text {
        return Ok(reference.clone());
    }
    if let Some(path) = &entry.reference_file {
        let path = resolve_manifest_path(manifest_dir, path);
        return std::fs::read_to_string(&path)
            .with_context(|| format!("read reference transcript {}", path.display()));
    }
    bail!(
        "golden corpus entry '{}' must set reference_text or reference_file",
        entry.id
    )
}

fn resolve_manifest_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn resolve_entry_wav_path(capture_dir: &Path, entry: &GoldenCorpusEntry) -> PathBuf {
    match &entry.asr_input_wav {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => capture_dir.join(path),
        None => capture_dir.join(ASR_INPUT_WAV),
    }
}

async fn replay_capture_wav(
    capture_dir: &Path,
    wav_path: &Path,
    reference: Option<&str>,
    chunk_ms: u32,
    trailing_silence_pad_ms: u32,
    backend_label: &str,
    asr: SharedAsrFactory,
) -> anyhow::Result<ReplayReport> {
    let file =
        File::open(wav_path).with_context(|| format!("open capture WAV {}", wav_path.display()))?;
    let (spec, samples) = decode_streaming_wav_to_f32(file)
        .with_context(|| format!("decode capture WAV {}", wav_path.display()))?;
    if spec.sample_rate != ASR_SAMPLE_RATE_HZ {
        bail!(
            "expected {ASR_SAMPLE_RATE_HZ} Hz ASR input WAV, got {} Hz",
            spec.sample_rate
        );
    }
    let mono = downmix_to_mono(&samples, spec.channels)?;
    let i16_samples = f32_to_i16_clamped(&mono);
    let run = replay_samples(&i16_samples, chunk_ms, trailing_silence_pad_ms, asr).await?;
    let wer = reference.map(|reference| compute_wer(reference, &run.transcript));

    Ok(ReplayReport {
        backend: backend_label.to_string(),
        capture_dir: capture_dir.display().to_string(),
        wav_path: wav_path.display().to_string(),
        wav_sample_rate_hz: spec.sample_rate,
        wav_channels: spec.channels,
        sample_count: i16_samples.len(),
        chunk_ms,
        transcript: run.transcript,
        wer,
        latency: run.latency,
    })
}

pub(crate) struct ReplayRun {
    pub(crate) transcript: String,
    pub(crate) latency: ReplayLatencyReport,
}

pub(crate) async fn replay_samples(
    samples: &[i16],
    chunk_ms: u32,
    trailing_silence_pad_ms: u32,
    asr: SharedAsrFactory,
) -> anyhow::Result<ReplayRun> {
    let chunk_samples = ((u64::from(ASR_SAMPLE_RATE_HZ) * u64::from(chunk_ms)) / 1_000) as usize;
    if chunk_samples == 0 {
        bail!("--chunk-ms is too small for {ASR_SAMPLE_RATE_HZ} Hz audio");
    }

    let mut session = asr.open_session().await?;
    let mut transcript = TranscriptAssembler::default();
    let wall_start = Instant::now();
    let mut ingest_stats = ReplayIngestStats::default();

    for chunk in samples.chunks(chunk_samples) {
        ingest_replay_chunk(&mut session, chunk, &mut transcript, &mut ingest_stats).await?;
        ingest_stats.chunk_count += 1;
    }

    let trailing_silence_pad_samples = trailing_silence_pad_sample_count(trailing_silence_pad_ms);
    if trailing_silence_pad_samples > 0 {
        let trailing_silence = vec![0_i16; trailing_silence_pad_samples];
        for chunk in trailing_silence.chunks(chunk_samples) {
            ingest_replay_chunk(&mut session, chunk, &mut transcript, &mut ingest_stats).await?;
            ingest_stats.trailing_silence_pad_chunks += 1;
        }
    }

    let finish_start = Instant::now();
    let events = session.finish().await?;
    let finish_ms = finish_start.elapsed().as_millis();
    transcript.record_events(events);
    let wall_ms = wall_start.elapsed().as_millis();
    let audio_ms = (samples.len() as u64).saturating_mul(1_000) / u64::from(ASR_SAMPLE_RATE_HZ);

    Ok(ReplayRun {
        transcript: transcript.assembled(),
        latency: ReplayLatencyReport {
            audio_ms,
            chunk_count: ingest_stats.chunk_count,
            trailing_silence_pad_ms,
            trailing_silence_pad_chunks: ingest_stats.trailing_silence_pad_chunks,
            ingest_total_ms: ingest_stats.ingest_total_ms,
            ingest_max_ms: ingest_stats.ingest_max_ms,
            finish_ms,
            wall_ms,
        },
    })
}

#[derive(Default)]
struct ReplayIngestStats {
    chunk_count: usize,
    trailing_silence_pad_chunks: usize,
    ingest_total_ms: u128,
    ingest_max_ms: u128,
}

async fn ingest_replay_chunk(
    session: &mut Box<dyn InboundAsrSession>,
    chunk: &[i16],
    transcript: &mut TranscriptAssembler,
    ingest_stats: &mut ReplayIngestStats,
) -> anyhow::Result<()> {
    let ingest_start = Instant::now();
    let events = session
        .ingest(AudioBuf::<i16, ASR_SAMPLE_RATE_HZ, Mono>::new(
            chunk.to_vec(),
        ))
        .await?;
    let ingest_ms = ingest_start.elapsed().as_millis();
    ingest_stats.ingest_total_ms += ingest_ms;
    ingest_stats.ingest_max_ms = ingest_stats.ingest_max_ms.max(ingest_ms);
    transcript.record_events(events);
    Ok(())
}

fn trailing_silence_pad_sample_count(trailing_silence_pad_ms: u32) -> usize {
    ((u64::from(ASR_SAMPLE_RATE_HZ) * u64::from(trailing_silence_pad_ms)) / 1_000) as usize
}

fn reference_text(args: &ReplayCaptureArgs) -> anyhow::Result<Option<String>> {
    if let Some(reference) = &args.reference {
        return Ok(Some(reference.clone()));
    }
    if let Some(path) = &args.reference_file {
        return Ok(Some(std::fs::read_to_string(path).with_context(|| {
            format!("read reference transcript {}", path.display())
        })?));
    }
    Ok(None)
}

#[derive(Default)]
struct TranscriptAssembler {
    final_text: String,
    current_partial: Option<String>,
}

impl TranscriptAssembler {
    fn record_events(&mut self, events: Vec<AsrTranscriptEvent>) {
        for event in events {
            match event.event {
                TranscriptEvent::Partial { text, .. } => self.current_partial = Some(text),
                TranscriptEvent::Final { text, .. } => {
                    append_fragment(&mut self.final_text, &text);
                    self.current_partial = None;
                }
            }
        }
    }

    fn assembled(&self) -> String {
        match (
            self.final_text.trim(),
            self.current_partial.as_deref().map(str::trim),
        ) {
            ("", Some(partial)) if !partial.is_empty() => partial.to_string(),
            (final_text, Some(partial)) if !final_text.is_empty() && !partial.is_empty() => {
                format!("{final_text} {partial}")
            }
            (final_text, _) if !final_text.is_empty() => final_text.to_string(),
            _ => String::new(),
        }
    }
}

fn append_fragment(buffer: &mut String, fragment: &str) {
    let fragment = fragment.trim();
    if fragment.is_empty() {
        return;
    }
    if !buffer.is_empty() {
        buffer.push(' ');
    }
    buffer.push_str(fragment);
}

pub fn compute_wer(reference: &str, hypothesis: &str) -> WerReport {
    let reference_tokens = normalize_tokens(reference);
    let hypothesis_tokens = normalize_tokens(hypothesis);
    let n = reference_tokens.len();
    let m = hypothesis_tokens.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    let mut backtrace = vec![vec![EditOp::Match; m + 1]; n + 1];

    for i in 1..=n {
        dp[i][0] = i;
        backtrace[i][0] = EditOp::Delete;
    }
    for j in 1..=m {
        dp[0][j] = j;
        backtrace[0][j] = EditOp::Insert;
    }

    for i in 1..=n {
        for j in 1..=m {
            let diagonal_op = if reference_tokens[i - 1] == hypothesis_tokens[j - 1] {
                EditOp::Match
            } else {
                EditOp::Substitute
            };
            let diagonal_cost = dp[i - 1][j - 1] + usize::from(diagonal_op == EditOp::Substitute);
            let delete_cost = dp[i - 1][j] + 1;
            let insert_cost = dp[i][j - 1] + 1;

            let (cost, op) = [
                (diagonal_cost, diagonal_op),
                (delete_cost, EditOp::Delete),
                (insert_cost, EditOp::Insert),
            ]
            .into_iter()
            .min_by_key(|(cost, _)| *cost)
            .unwrap_or((diagonal_cost, diagonal_op));
            dp[i][j] = cost;
            backtrace[i][j] = op;
        }
    }

    let mut i = n;
    let mut j = m;
    let mut errors_by_token = Vec::new();
    while i > 0 || j > 0 {
        match backtrace[i][j] {
            EditOp::Match => {
                i -= 1;
                j -= 1;
            }
            EditOp::Substitute => {
                errors_by_token.push(WerTokenError::Substitution {
                    reference: reference_tokens[i - 1].clone(),
                    hypothesis: hypothesis_tokens[j - 1].clone(),
                });
                i -= 1;
                j -= 1;
            }
            EditOp::Delete => {
                errors_by_token.push(WerTokenError::Deletion {
                    reference: reference_tokens[i - 1].clone(),
                });
                i -= 1;
            }
            EditOp::Insert => {
                errors_by_token.push(WerTokenError::Insertion {
                    hypothesis: hypothesis_tokens[j - 1].clone(),
                });
                j -= 1;
            }
        }
    }
    errors_by_token.reverse();

    let substitutions = errors_by_token
        .iter()
        .filter(|error| matches!(error, WerTokenError::Substitution { .. }))
        .count();
    let deletions = errors_by_token
        .iter()
        .filter(|error| matches!(error, WerTokenError::Deletion { .. }))
        .count();
    let insertions = errors_by_token
        .iter()
        .filter(|error| matches!(error, WerTokenError::Insertion { .. }))
        .count();

    WerReport {
        reference_words: n,
        hypothesis_words: m,
        substitutions,
        deletions,
        insertions,
        errors: substitutions + deletions + insertions,
        errors_by_token,
    }
}

fn normalize_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_uppercase());
        } else if matches!(ch, '\'' | '`') {
            continue;
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EditOp {
    Match,
    Substitute,
    Delete,
    Insert,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::adapter::EchoAsrFactory;
    use crate::cli::ReplayBackendArg;

    #[test]
    fn wer_counts_substitution_deletion_and_insertion() {
        let report = compute_wer("the quick brown fox", "the great brown fox now");

        assert_eq!(report.reference_words, 4);
        assert_eq!(report.hypothesis_words, 5);
        assert_eq!(report.substitutions, 1);
        assert_eq!(report.deletions, 0);
        assert_eq!(report.insertions, 1);
        assert_eq!(report.errors, 2);
        assert!((report.rate() - 0.5).abs() < f64::EPSILON);
        assert_eq!(
            report.errors_by_token,
            vec![
                WerTokenError::Substitution {
                    reference: "QUICK".to_string(),
                    hypothesis: "GREAT".to_string(),
                },
                WerTokenError::Insertion {
                    hypothesis: "NOW".to_string(),
                },
            ]
        );
    }

    #[test]
    fn transcript_assembler_uses_finals_plus_current_partial() {
        let mut assembler = TranscriptAssembler::default();
        assembler.record_events(vec![AsrTranscriptEvent::emit(TranscriptEvent::Partial {
            text: "HEL".to_string(),
            update: Default::default(),
        })]);
        assert_eq!(assembler.assembled(), "HEL");

        assembler.record_events(vec![AsrTranscriptEvent::emit(TranscriptEvent::Final {
            text: "HELLO".to_string(),
            update: Default::default(),
        })]);
        assembler.record_events(vec![AsrTranscriptEvent::emit(TranscriptEvent::Partial {
            text: "WOR".to_string(),
            update: Default::default(),
        })]);
        assert_eq!(assembler.assembled(), "HELLO WOR");
    }

    #[test]
    fn wer_normalization_keeps_apostrophe_words_together() {
        assert_eq!(
            normalize_tokens("whate'er John's"),
            vec!["WHATEER".to_string(), "JOHNS".to_string()]
        );
    }

    #[test]
    fn trailing_silence_pad_sample_count_uses_16khz_audio() {
        assert_eq!(trailing_silence_pad_sample_count(0), 0);
        assert_eq!(trailing_silence_pad_sample_count(20), 320);
        assert_eq!(
            trailing_silence_pad_sample_count(DEFAULT_TRAILING_SILENCE_PAD_MS),
            12_800
        );
    }

    #[tokio::test]
    async fn replay_capture_reports_backend_wer_and_latency() {
        let capture_dir = test_capture_dir("single");
        write_test_wav(&capture_dir.join(ASR_INPUT_WAV), 16_000);
        let args = ReplayCaptureArgs {
            capture_dir: capture_dir.clone(),
            reference: Some("received 28800 samples".to_string()),
            reference_file: None,
            chunk_ms: 20,
            trailing_silence_pad_ms: DEFAULT_TRAILING_SILENCE_PAD_MS,
            backend: ReplayBackendArg::Echo,
        };

        let report = replay_capture(&args, "echo", Arc::new(EchoAsrFactory))
            .await
            .expect("replay capture");

        assert_eq!(report.backend, "echo");
        assert_eq!(report.wav_sample_rate_hz, ASR_SAMPLE_RATE_HZ);
        assert_eq!(report.wav_channels, 1);
        assert_eq!(report.sample_count, 16_000);
        assert_eq!(report.chunk_ms, 20);
        assert_eq!(report.latency.audio_ms, 1_000);
        assert_eq!(report.latency.chunk_count, 50);
        assert_eq!(
            report.latency.trailing_silence_pad_ms,
            DEFAULT_TRAILING_SILENCE_PAD_MS
        );
        assert_eq!(report.latency.trailing_silence_pad_chunks, 40);
        let wer = report.wer.expect("WER report");
        assert_eq!(wer.errors, 0);
        std::fs::remove_dir_all(capture_dir).expect("remove test capture dir");
    }

    #[tokio::test]
    async fn replay_corpus_runs_manifest_entries_against_backends() {
        let capture_dir = test_capture_dir("corpus");
        write_test_wav(&capture_dir.join(ASR_INPUT_WAV), 16_000);
        let manifest_path = capture_dir.join("asr-golden.json");
        std::fs::write(
            &manifest_path,
            r#"{
              "schema_version": 1,
              "entries": [
                {
                  "id": "echo-fixture",
                  "description": "synthetic echo replay fixture",
                  "capture_dir": ".",
                  "asr_input_wav": "asr-input-16khz.wav",
                  "reference_text": "received 28800 samples",
                  "codec": "PCMU",
                  "media_sample_rate_hz": 8000,
                  "direction": "inbound"
                }
              ]
            }"#,
        )
        .expect("write manifest");
        let args = ReplayCorpusArgs {
            manifest: manifest_path.clone(),
            backend: vec![ReplayBackendArg::Echo],
            chunk_ms: 20,
            trailing_silence_pad_ms: DEFAULT_TRAILING_SILENCE_PAD_MS,
        };

        let report = replay_corpus(
            &args,
            vec![ReplayBackend::new("echo", Arc::new(EchoAsrFactory))],
        )
        .await
        .expect("replay corpus");

        assert_eq!(report.entries.len(), 1);
        let entry = &report.entries[0];
        assert_eq!(entry.id, "echo-fixture");
        assert_eq!(entry.codec, "PCMU");
        assert_eq!(entry.media_sample_rate_hz, 8000);
        assert_eq!(entry.direction, "inbound");
        assert_eq!(entry.reports.len(), 1);
        assert_eq!(entry.reports[0].backend, "echo");
        assert_eq!(entry.reports[0].wer.as_ref().expect("WER").errors, 0);
        std::fs::remove_dir_all(capture_dir).expect("remove test capture dir");
    }

    fn test_capture_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "motlie-telnyx-replay-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test capture dir");
        dir
    }

    fn write_test_wav(path: &Path, sample_count: usize) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: ASR_SAMPLE_RATE_HZ,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create test wav");
        for _ in 0..sample_count {
            writer.write_sample::<i16>(0).expect("write test sample");
        }
        writer.finalize().expect("finalize test wav");
    }
}
