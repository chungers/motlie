//! Shared types and utilities for TTS↔ASR end-to-end validation pipelines.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use motlie_model::{
    metrics_runtime::current_resident_memory_bytes, AudioSpec, PcmChunk, PcmEncoding,
    SpeechRequest, TranscriptionParams,
};
use serde::{Deserialize, Serialize};

/// One sample from the test dataset.
#[derive(Clone, Debug, Deserialize)]
pub struct TestSample {
    pub sample_id: String,
    pub category: String,
    pub word_count: usize,
    pub text: String,
}

/// Per-sample result with metrics.
#[derive(Clone, Debug, Serialize)]
pub struct PipelineResult {
    pub pipeline: String,
    pub sample_id: String,
    pub category: String,
    pub word_count: usize,
    pub original_text: String,
    pub transcribed_text: String,
    pub wer: f64,
    pub tts_latency_ms: u64,
    pub asr_latency_ms: u64,
    pub total_latency_ms: u64,
    pub pcm_bytes: usize,
    pub pcm_duration_ms: u64,
    pub resident_memory_bytes: Option<u64>,
    pub peak_resident_memory_bytes: Option<u64>,
}

/// Load the test dataset from a JSON file.
pub fn load_dataset(path: &Path) -> Result<Vec<TestSample>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open dataset: {}", path.display()))?;
    let mut samples: Vec<TestSample> =
        serde_json::from_reader(file).context("failed to parse dataset JSON")?;
    for sample in &mut samples {
        sample.word_count = sample.text.split_whitespace().count();
    }
    Ok(samples)
}

/// Run the TTS→ASR pipeline for a single sample.
///
/// 1. Synthesizes `sample.text` via `speech_model.open_stream()`
/// 2. Collects all PCM chunks from the speech stream
/// 3. Feeds the PCM into `transcription_model.open_stream()` + `push_chunk()`
/// 4. Collects final transcript text
/// 5. Computes WER between original and transcribed text
pub async fn run_pipeline(
    pipeline_name: &str,
    sample: &TestSample,
    speech: &dyn motlie_model::SpeechModel,
    transcription: &dyn motlie_model::TranscriptionModel,
) -> Result<PipelineResult> {
    let mut peak_resident_memory_bytes = current_resident_memory_bytes();

    // Step 1: TTS
    let tts_start = Instant::now();
    let mut speech_stream = speech
        .open_stream(SpeechRequest {
            text: sample.text.clone(),
            params: Default::default(),
            conditioning: None,
        })
        .await
        .context("TTS open_stream failed")?;
    observe_peak_resident_memory(&mut peak_resident_memory_bytes);

    let audio_spec = speech_stream.audio_spec().clone();
    let mut pcm_data = Vec::new();

    while let Some(chunk) = speech_stream
        .next_chunk()
        .await
        .context("TTS next_chunk failed")?
    {
        pcm_data.extend_from_slice(&chunk.data);
        observe_peak_resident_memory(&mut peak_resident_memory_bytes);
        if chunk.end_of_stream {
            break;
        }
    }
    speech_stream.finish().await.context("TTS finish failed")?;
    observe_peak_resident_memory(&mut peak_resident_memory_bytes);
    let tts_latency = tts_start.elapsed();

    if pcm_data.is_empty() {
        anyhow::bail!("TTS produced no PCM output for sample {}", sample.sample_id);
    }

    let pcm_duration_ms = compute_pcm_duration_ms(&audio_spec, pcm_data.len());

    // Step 2: ASR — feed TTS output into transcription stream
    let asr_start = Instant::now();

    // ASR expects 16kHz mono S16Le for whisper, or the native format for sherpa.
    // We pass the TTS audio spec directly and let the ASR backend normalize.
    let asr_spec = AudioSpec {
        sample_rate_hz: audio_spec.sample_rate_hz,
        channels: audio_spec.channels,
        encoding: audio_spec.encoding,
    };

    let mut asr_stream = transcription
        .open_stream(
            asr_spec,
            TranscriptionParams {
                language: Some("en".into()),
                emit_partials: false,
            },
        )
        .await
        .context("ASR open_stream failed")?;
    observe_peak_resident_memory(&mut peak_resident_memory_bytes);

    // Feed PCM in chunks
    let chunk_size = 16_000; // ~0.5s at 16kHz mono S16Le
    let mut offset = 0;
    let mut sequence = 0u64;
    let mut transcript_segments = Vec::new();

    while offset < pcm_data.len() {
        let end = (offset + chunk_size).min(pcm_data.len());
        let is_last = end >= pcm_data.len();

        let chunk = PcmChunk {
            data: pcm_data[offset..end].to_vec(),
            sequence,
            end_of_stream: is_last,
        };

        if let Some(update) = asr_stream
            .push_chunk(chunk)
            .await
            .context("ASR push_chunk failed")?
        {
            for seg in update.segments {
                transcript_segments.push(seg.text);
            }
        }
        observe_peak_resident_memory(&mut peak_resident_memory_bytes);

        offset = end;
        sequence += 1;
    }

    let final_update = asr_stream.finish().await.context("ASR finish failed")?;
    observe_peak_resident_memory(&mut peak_resident_memory_bytes);
    for seg in final_update.segments {
        transcript_segments.push(seg.text);
    }
    let asr_latency = asr_start.elapsed();

    let transcribed_text = transcript_segments.join(" ").trim().to_string();
    let wer = compute_wer(&sample.text, &transcribed_text);
    let resident_memory_bytes = current_resident_memory_bytes();

    Ok(PipelineResult {
        pipeline: pipeline_name.to_string(),
        sample_id: sample.sample_id.clone(),
        category: sample.category.clone(),
        word_count: sample.word_count,
        original_text: sample.text.clone(),
        transcribed_text,
        wer,
        tts_latency_ms: tts_latency.as_millis() as u64,
        asr_latency_ms: asr_latency.as_millis() as u64,
        total_latency_ms: (tts_latency + asr_latency).as_millis() as u64,
        pcm_bytes: pcm_data.len(),
        pcm_duration_ms,
        resident_memory_bytes,
        peak_resident_memory_bytes,
    })
}

/// Compute word error rate between reference and hypothesis text.
///
/// Uses the standard WER formula: (substitutions + insertions + deletions) / reference_length.
/// Implemented via minimum edit distance on word sequences.
pub fn compute_wer(reference: &str, hypothesis: &str) -> f64 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    let n = ref_words.len();
    let m = hyp_words.len();

    // Levenshtein distance on word sequences
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if ref_words[i - 1].to_lowercase() == hyp_words[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1) // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution
        }
    }

    dp[n][m] as f64 / n as f64
}

fn compute_pcm_duration_ms(spec: &AudioSpec, byte_len: usize) -> u64 {
    let bytes_per_sample = match spec.encoding {
        PcmEncoding::S16Le => 2,
        PcmEncoding::F32Le => 4,
    };
    let total_samples = byte_len / (bytes_per_sample * spec.channels as usize);
    if spec.sample_rate_hz == 0 {
        return 0;
    }
    (total_samples as u64 * 1000) / spec.sample_rate_hz as u64
}

fn observe_peak_resident_memory(peak: &mut Option<u64>) {
    if let Some(current) = current_resident_memory_bytes() {
        *peak = Some(match *peak {
            Some(previous) => previous.max(current),
            None => current,
        });
    }
}

/// Print a result as a JSON line to stdout.
pub fn emit_jsonl(result: &PipelineResult) {
    if let Ok(json) = serde_json::to_string(result) {
        println!("{json}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_identical_is_zero() {
        assert_eq!(compute_wer("hello world", "hello world"), 0.0);
    }

    #[test]
    fn wer_completely_wrong_is_one() {
        assert_eq!(compute_wer("hello world", "foo bar"), 1.0);
    }

    #[test]
    fn wer_partial_match() {
        // "the cat sat" vs "the dog sat" → 1 substitution / 3 words = 0.333...
        let wer = compute_wer("the cat sat", "the dog sat");
        assert!((wer - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn wer_empty_reference_empty_hypothesis() {
        assert_eq!(compute_wer("", ""), 0.0);
    }

    #[test]
    fn wer_empty_reference_nonempty_hypothesis() {
        assert_eq!(compute_wer("", "some words"), 1.0);
    }

    #[test]
    fn wer_case_insensitive() {
        assert_eq!(compute_wer("Hello World", "hello world"), 0.0);
    }

    #[test]
    fn wer_insertion() {
        // "a b" vs "a c b" → 1 insertion / 2 words = 0.5
        let wer = compute_wer("a b", "a c b");
        assert!((wer - 0.5).abs() < 0.01);
    }

    #[test]
    fn wer_deletion() {
        // "a b c" vs "a c" → 1 deletion / 3 words = 0.333...
        let wer = compute_wer("a b c", "a c");
        assert!((wer - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn load_dataset_recomputes_word_count_from_text() {
        let path = std::env::temp_dir().join(format!(
            "tts_asr_dataset_{}_common.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"[{"sample_id":"sample","category":"short","word_count":999,"text":"one two three"}]"#,
        )
        .unwrap();

        let samples = load_dataset(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].word_count, 3);
    }
}
