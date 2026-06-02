use std::fs::File;

use anyhow::{bail, Context};
use motlie_model::typed::{AudioBuf, Mono};
use motlie_voice::app::TranscriptEvent;
use motlie_voice::pipeline::convert::{downmix_to_mono, f32_to_i16_clamped};
use motlie_voice::wav::decode_streaming_wav_to_f32;

use crate::adapter::SharedAsrFactory;
use crate::cli::ReplayCaptureArgs;

const ASR_INPUT_WAV: &str = "asr-input-16khz.wav";
const ASR_SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayReport {
    pub capture_dir: String,
    pub wav_path: String,
    pub sample_count: usize,
    pub transcript: String,
    pub wer: Option<WerReport>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
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

pub async fn replay_capture(
    args: &ReplayCaptureArgs,
    asr: SharedAsrFactory,
) -> anyhow::Result<ReplayReport> {
    if args.chunk_ms == 0 {
        bail!("--chunk-ms must be greater than zero");
    }

    let wav_path = args.capture_dir.join(ASR_INPUT_WAV);
    let file = File::open(&wav_path)
        .with_context(|| format!("open capture WAV {}", wav_path.display()))?;
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
    let transcript = replay_samples(&i16_samples, args.chunk_ms, asr).await?;
    let reference = reference_text(args)?;
    let wer = reference
        .as_deref()
        .map(|reference| compute_wer(reference, &transcript));

    Ok(ReplayReport {
        capture_dir: args.capture_dir.display().to_string(),
        wav_path: wav_path.display().to_string(),
        sample_count: i16_samples.len(),
        transcript,
        wer,
    })
}

async fn replay_samples(
    samples: &[i16],
    chunk_ms: u32,
    asr: SharedAsrFactory,
) -> anyhow::Result<String> {
    let chunk_samples = ((u64::from(ASR_SAMPLE_RATE_HZ) * u64::from(chunk_ms)) / 1_000) as usize;
    if chunk_samples == 0 {
        bail!("--chunk-ms is too small for {ASR_SAMPLE_RATE_HZ} Hz audio");
    }

    let mut session = asr.open_session().await?;
    let mut transcript = TranscriptAssembler::default();
    for chunk in samples.chunks(chunk_samples) {
        let events = session
            .ingest(AudioBuf::<i16, ASR_SAMPLE_RATE_HZ, Mono>::new(
                chunk.to_vec(),
            ))
            .await?;
        transcript.record_events(events);
    }
    let events = session.finish().await?;
    transcript.record_events(events);
    Ok(transcript.assembled())
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
    fn record_events(&mut self, events: Vec<TranscriptEvent>) {
        for event in events {
            match event {
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
    use super::*;

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
        assembler.record_events(vec![TranscriptEvent::Partial {
            text: "HEL".to_string(),
            update: Default::default(),
        }]);
        assert_eq!(assembler.assembled(), "HEL");

        assembler.record_events(vec![TranscriptEvent::Final {
            text: "HELLO".to_string(),
            update: Default::default(),
        }]);
        assembler.record_events(vec![TranscriptEvent::Partial {
            text: "WOR".to_string(),
            update: Default::default(),
        }]);
        assert_eq!(assembler.assembled(), "HELLO WOR");
    }

    #[test]
    fn wer_normalization_keeps_apostrophe_words_together() {
        assert_eq!(
            normalize_tokens("whate'er John's"),
            vec!["WHATEER".to_string(), "JOHNS".to_string()]
        );
    }
}
