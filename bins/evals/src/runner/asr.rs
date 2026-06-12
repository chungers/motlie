use std::future::Future;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::typed::{
    AudioBuf, BatchTranscriber, Mono, StreamingTranscriber, TranscriptionSession,
};
use motlie_model::{
    BundleHandle, ModelError, StartOptions, TranscriptSegment, TranscriptionParams,
    TranscriptionUpdate,
};
use motlie_voice::pipeline::convert::{decode_samples_to_f32, downmix_to_mono, f32_to_i16_clamped};
use motlie_voice::pipeline::resample::{LinearInterpolator, Resampler};

use crate::metrics::{
    AsrPerformanceMetrics, CapabilityPerformanceMetrics, MetricUnavailable, PerformanceMetrics,
};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, bundle_filter_capability_kind, elapsed_ms,
    evaluate_performance_measured, evaluate_resource_status, mean, observe_backend_accelerator,
    percentile, prepare_bundle,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{AsrAssertions, CapabilityName};

const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
const DEFAULT_STREAMING_CHUNK_MS: u64 = 200;

pub struct AsrRunner;

#[async_trait]
impl ScenarioRunner for AsrRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::Asr,
            "scenario `{}` is not an ASR scenario",
            context.scenario.id
        );
        let asr_scenario = context
            .scenario
            .asr()
            .context("ASR scenario should carry ASR input/assertions")?
            .clone();
        let prepared = prepare_bundle(
            &context,
            bundle_filter_capability_kind(context.scenario.bundle_filter.capability),
            &[],
        )?;
        let audio_path = resolve_repo_path(&asr_scenario.input.audio);
        let decoded = decode_wav_mono16k(&audio_path)?;
        let params = TranscriptionParams {
            language: asr_scenario.input.language.clone(),
            emit_partials: false,
        };
        let iteration_config = IterationConfig {
            iterations: asr_scenario.input.iterations,
            warmup_iterations: asr_scenario.input.warmup_iterations,
        };

        let eval = run_selected_asr(
            &mut context,
            &prepared,
            decoded.f32_audio,
            decoded.i16_audio,
            params,
            asr_scenario
                .input
                .streaming_chunk_ms
                .unwrap_or(DEFAULT_STREAMING_CHUNK_MS),
            iteration_config,
        )
        .await?;

        let transcript = render_plain_transcript(&eval.segments).unwrap_or_default();
        let word_error_rate = asr_scenario
            .input
            .reference_transcript
            .as_ref()
            .map(|reference| word_error_rate(reference, &transcript));
        let asr_metrics = AsrPerformanceMetrics {
            iterations: Some(eval.iterations),
            successful_iterations: Some(eval.successful_iterations),
            failed_iterations: Some(eval.failed_iterations),
            audio_duration_ms: Some(decoded.duration_ms),
            transcription_latency_ms: None,
            mean_transcription_latency_ms: eval.mean_transcription_latency_ms,
            p95_transcription_latency_ms: eval.p95_transcription_latency_ms,
            ttfp_first_partial_ms: None,
            mean_ttfp_first_partial_ms: eval.mean_ttfp_first_partial_ms,
            p95_ttfp_first_partial_ms: eval.p95_ttfp_first_partial_ms,
            real_time_factor: eval.mean_transcription_latency_ms.and_then(|value| {
                (decoded.duration_ms > 0).then_some(value / decoded.duration_ms as f64)
            }),
            transcript_chars: Some(transcript.chars().count() as u64),
            segment_count: Some(eval.segments.len() as u64),
            word_error_rate,
        };
        let unavailable = required_asr_metric_gaps(&asr_metrics, eval.mode);
        let performance = PerformanceMetrics {
            startup_ms: Some(eval.startup_ms),
            warmup_ms: Some(eval.warmup_ms),
            request_latencies_ms: eval.request_latencies_ms,
            unavailable,
            capability_metrics: CapabilityPerformanceMetrics::Asr(asr_metrics),
        };
        let resources = context.metrics_sampler.finish();
        let assertions =
            evaluate_assertions(&transcript, word_error_rate, &asr_scenario.assertions);
        let performance_evaluation = evaluate_performance_measured(
            performance.startup_ms.is_some() && !performance.request_latencies_ms.is_empty(),
            "performance metrics missing startup or transcription latency",
        );
        let resource_evaluation = evaluate_resource_status(&resources, &context);
        let record = build_record(
            &context,
            &prepared,
            performance,
            resources,
            assertions,
            performance_evaluation,
            resource_evaluation,
        );

        context.output_sink.emit(&record)?;
        Ok(record)
    }
}

struct DecodedAsrAudio {
    f32_audio: AudioBuf<f32, ASR_SAMPLE_RATE_HZ, Mono>,
    i16_audio: AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>,
    duration_ms: u64,
}

struct AsrEvalOutput {
    startup_ms: u64,
    warmup_ms: u64,
    iterations: u64,
    successful_iterations: u64,
    failed_iterations: u64,
    request_latencies_ms: Vec<u64>,
    mean_transcription_latency_ms: Option<f64>,
    p95_transcription_latency_ms: Option<f64>,
    mean_ttfp_first_partial_ms: Option<f64>,
    p95_ttfp_first_partial_ms: Option<f64>,
    mode: AsrMode,
    segments: Vec<TranscriptSegment>,
}

struct AsrIterationMetrics {
    transcription_latency_ms: u64,
    ttfp_first_partial_ms: Option<u64>,
    segments: Vec<TranscriptSegment>,
}

#[derive(Clone, Copy)]
struct IterationConfig {
    iterations: u64,
    warmup_iterations: u64,
}

struct AsrIterationSummary {
    iterations: u64,
    successful_iterations: u64,
    failed_iterations: u64,
    request_latencies_ms: Vec<u64>,
    ttfp_first_partial_ms: Vec<u64>,
    segments: Vec<TranscriptSegment>,
}

#[derive(Clone, Copy)]
enum AsrMode {
    Batch,
    Streaming,
}

#[allow(unused_variables)]
async fn run_selected_asr(
    context: &mut RunContext,
    prepared: &crate::runner::support::PreparedBundle,
    f32_audio: AudioBuf<f32, ASR_SAMPLE_RATE_HZ, Mono>,
    i16_audio: AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>,
    params: TranscriptionParams,
    streaming_chunk_ms: u64,
    iteration_config: IterationConfig,
) -> Result<AsrEvalOutput> {
    let bundle_id = prepared.bundle_id.as_str();

    #[cfg(feature = "model-whisper-base-en")]
    if bundle_id == "whisper_base_en" {
        return run_batch_asr(
            context,
            crate::runner::support::start_options(context, prepared),
            f32_audio,
            params,
            iteration_config,
            motlie_models::asr::whisper_base_en::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-sherpa-onnx-streaming")]
    if bundle_id == "sherpa_onnx_streaming_zipformer_en" {
        return run_streaming_asr(
            context,
            crate::runner::support::start_options(context, prepared),
            i16_audio,
            streaming_transcription_params(params),
            streaming_chunk_ms,
            iteration_config,
            motlie_models::asr::sherpa_onnx_streaming_en::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-moonshine-streaming")]
    if bundle_id == "moonshine_streaming_en" {
        return run_streaming_asr(
            context,
            crate::runner::support::start_options(context, prepared),
            i16_audio,
            streaming_transcription_params(params),
            streaming_chunk_ms,
            iteration_config,
            motlie_models::asr::moonshine_streaming_en::start_typed,
        )
        .await;
    }

    bail!("ASR bundle `{bundle_id}` is not enabled or not supported by the eval runner")
}

#[allow(dead_code)]
fn streaming_transcription_params(mut params: TranscriptionParams) -> TranscriptionParams {
    params.emit_partials = true;
    params
}

#[allow(dead_code)]
async fn run_batch_asr<Handle, Start, StartFuture>(
    context: &mut RunContext,
    options: StartOptions,
    audio: AudioBuf<f32, ASR_SAMPLE_RATE_HZ, Mono>,
    params: TranscriptionParams,
    iteration_config: IterationConfig,
    start: Start,
) -> Result<AsrEvalOutput>
where
    Handle: BundleHandle + BatchTranscriber<Input = AudioBuf<f32, ASR_SAMPLE_RATE_HZ, Mono>>,
    Start: FnOnce(StartOptions) -> StartFuture,
    StartFuture: Future<Output = std::result::Result<Handle, ModelError>> + Send,
{
    context.metrics_sampler.sample();
    let startup_started_at = std::time::Instant::now();
    let handle = start(options).await.context("failed to start ASR bundle")?;
    let startup_ms = elapsed_ms(startup_started_at.elapsed());
    observe_backend_accelerator(context, &handle);
    context.metrics_sampler.sample();

    let warmup_started_at = std::time::Instant::now();
    for _ in 0..iteration_config.warmup_iterations {
        let _ = run_one_batch_asr(&handle, audio.clone(), params.clone()).await?;
        context.metrics_sampler.sample();
    }
    let warmup_ms = elapsed_ms(warmup_started_at.elapsed());

    let mut request_latencies_ms = Vec::new();
    let mut ttfp_first_partial_ms = Vec::new();
    let mut successful_iterations = 0_u64;
    let mut failed_iterations = 0_u64;
    let mut segments = Vec::new();
    for _ in 0..iteration_config.iterations {
        match run_one_batch_asr(&handle, audio.clone(), params.clone()).await {
            Ok(iteration) => {
                request_latencies_ms.push(iteration.transcription_latency_ms);
                push_if_some(&mut ttfp_first_partial_ms, iteration.ttfp_first_partial_ms);
                segments = iteration.segments;
                successful_iterations += 1;
            }
            Err(_) => failed_iterations += 1,
        }
        context.metrics_sampler.sample();
    }

    handle.shutdown().await.context("ASR shutdown failed")?;

    Ok(asr_eval_output(
        startup_ms,
        warmup_ms,
        AsrMode::Batch,
        AsrIterationSummary {
            iterations: iteration_config.iterations,
            successful_iterations,
            failed_iterations,
            request_latencies_ms,
            ttfp_first_partial_ms,
            segments,
        },
    ))
}

async fn run_one_batch_asr<Handle>(
    handle: &Handle,
    audio: AudioBuf<f32, ASR_SAMPLE_RATE_HZ, Mono>,
    params: TranscriptionParams,
) -> Result<AsrIterationMetrics>
where
    Handle: BatchTranscriber<Input = AudioBuf<f32, ASR_SAMPLE_RATE_HZ, Mono>>,
{
    let transcription_started_at = std::time::Instant::now();
    let update = handle
        .transcribe(audio, params)
        .await
        .context("batch transcription failed")?;
    let transcription_latency_ms = elapsed_ms(transcription_started_at.elapsed());

    Ok(AsrIterationMetrics {
        transcription_latency_ms,
        ttfp_first_partial_ms: None,
        segments: update.segments,
    })
}

#[allow(dead_code)]
async fn run_streaming_asr<Handle, Start, StartFuture>(
    context: &mut RunContext,
    options: StartOptions,
    audio: AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>,
    params: TranscriptionParams,
    chunk_ms: u64,
    iteration_config: IterationConfig,
    start: Start,
) -> Result<AsrEvalOutput>
where
    Handle: BundleHandle + StreamingTranscriber<Input = AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>>,
    Start: FnOnce(StartOptions) -> StartFuture,
    StartFuture: Future<Output = std::result::Result<Handle, ModelError>> + Send,
{
    context.metrics_sampler.sample();
    let startup_started_at = std::time::Instant::now();
    let handle = start(options).await.context("failed to start ASR bundle")?;
    let startup_ms = elapsed_ms(startup_started_at.elapsed());
    observe_backend_accelerator(context, &handle);
    context.metrics_sampler.sample();

    let warmup_started_at = std::time::Instant::now();
    for _ in 0..iteration_config.warmup_iterations {
        let _ = run_one_streaming_asr(&handle, audio.clone(), params.clone(), chunk_ms).await?;
        context.metrics_sampler.sample();
    }
    let warmup_ms = elapsed_ms(warmup_started_at.elapsed());

    let mut request_latencies_ms = Vec::new();
    let mut ttfp_first_partial_ms = Vec::new();
    let mut successful_iterations = 0_u64;
    let mut failed_iterations = 0_u64;
    let mut segments = Vec::new();
    for _ in 0..iteration_config.iterations {
        match run_one_streaming_asr(&handle, audio.clone(), params.clone(), chunk_ms).await {
            Ok(iteration) => {
                request_latencies_ms.push(iteration.transcription_latency_ms);
                push_if_some(&mut ttfp_first_partial_ms, iteration.ttfp_first_partial_ms);
                segments = iteration.segments;
                successful_iterations += 1;
            }
            Err(_) => failed_iterations += 1,
        }
        context.metrics_sampler.sample();
    }

    handle.shutdown().await.context("ASR shutdown failed")?;

    Ok(asr_eval_output(
        startup_ms,
        warmup_ms,
        AsrMode::Streaming,
        AsrIterationSummary {
            iterations: iteration_config.iterations,
            successful_iterations,
            failed_iterations,
            request_latencies_ms,
            ttfp_first_partial_ms,
            segments,
        },
    ))
}

async fn run_one_streaming_asr<Handle>(
    handle: &Handle,
    audio: AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>,
    params: TranscriptionParams,
    chunk_ms: u64,
) -> Result<AsrIterationMetrics>
where
    Handle: StreamingTranscriber<Input = AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>>,
{
    let transcription_started_at = std::time::Instant::now();
    let mut session = handle
        .open_session(params)
        .await
        .context("failed to open streaming ASR session")?;
    let chunk_samples = streaming_chunk_samples(chunk_ms);
    let mut segments = Vec::new();
    let mut first_audio_submitted_at = None;
    let mut ttfp_first_partial_ms = None;
    for chunk in audio.into_samples().chunks(chunk_samples) {
        let audio_chunk = AudioBuf::<i16, ASR_SAMPLE_RATE_HZ, Mono>::new(chunk.to_vec());
        if first_audio_submitted_at.is_none() {
            first_audio_submitted_at = Some(std::time::Instant::now());
        }
        if let Some(update) = session
            .ingest(audio_chunk)
            .await
            .context("streaming ASR ingest failed")?
        {
            if ttfp_first_partial_ms.is_none() && has_partial_transcript(&update) {
                if let Some(submitted_at) = first_audio_submitted_at.as_ref() {
                    ttfp_first_partial_ms = Some(elapsed_ms(submitted_at.elapsed()));
                }
            }
            segments.extend(final_segments(update));
        }
    }
    let final_update = session
        .finish()
        .await
        .context("streaming ASR finish failed")?;
    segments.extend(final_segments(final_update));
    let transcription_latency_ms = elapsed_ms(transcription_started_at.elapsed());

    Ok(AsrIterationMetrics {
        transcription_latency_ms,
        ttfp_first_partial_ms,
        segments,
    })
}

fn asr_eval_output(
    startup_ms: u64,
    warmup_ms: u64,
    mode: AsrMode,
    summary: AsrIterationSummary,
) -> AsrEvalOutput {
    let mean_transcription_latency_ms = mean(&summary.request_latencies_ms);
    let p95_transcription_latency_ms = percentile(&summary.request_latencies_ms, 0.95);
    let mean_ttfp_first_partial_ms = mean(&summary.ttfp_first_partial_ms);
    let p95_ttfp_first_partial_ms = percentile(&summary.ttfp_first_partial_ms, 0.95);

    AsrEvalOutput {
        startup_ms,
        warmup_ms,
        iterations: summary.iterations,
        successful_iterations: summary.successful_iterations,
        failed_iterations: summary.failed_iterations,
        mean_transcription_latency_ms,
        p95_transcription_latency_ms,
        mean_ttfp_first_partial_ms,
        p95_ttfp_first_partial_ms,
        request_latencies_ms: summary.request_latencies_ms,
        mode,
        segments: summary.segments,
    }
}

fn required_asr_metric_gaps(
    metrics: &AsrPerformanceMetrics,
    mode: AsrMode,
) -> Vec<MetricUnavailable> {
    if metrics.mean_ttfp_first_partial_ms.is_some() {
        return Vec::new();
    }

    let reason = match mode {
        AsrMode::Batch => "metric_not_applicable_for_batch_engine",
        AsrMode::Streaming => "metric_not_reported_by_backend",
    };
    vec![MetricUnavailable::new(
        "ttfp_first_partial_ms",
        reason,
        "asr_runner",
    )]
}

fn push_if_some(values: &mut Vec<u64>, value: Option<u64>) {
    if let Some(value) = value {
        values.push(value);
    }
}

fn decode_wav_mono16k(path: &Path) -> Result<DecodedAsrAudio> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open ASR wav `{}`", path.display()))?;
    let (spec, samples) = decode_samples_to_f32(reader).context("failed to decode wav samples")?;
    let source_frames = samples.len() as u64 / u64::from(spec.channels.max(1));
    let duration_ms = source_frames.saturating_mul(1000) / u64::from(spec.sample_rate.max(1));
    let mono = downmix_to_mono(&samples, spec.channels).context("failed to downmix wav to mono")?;
    let resampled = LinearInterpolator
        .resample_f32(&mono, spec.sample_rate, ASR_SAMPLE_RATE_HZ)
        .context("failed to resample wav to 16 kHz")?;
    let f32_audio = AudioBuf::<f32, ASR_SAMPLE_RATE_HZ, Mono>::new(resampled);
    let i16_audio =
        AudioBuf::<i16, ASR_SAMPLE_RATE_HZ, Mono>::new(f32_to_i16_clamped(f32_audio.samples()));

    Ok(DecodedAsrAudio {
        f32_audio,
        i16_audio,
        duration_ms,
    })
}

fn evaluate_assertions(
    transcript: &str,
    word_error_rate: Option<f64>,
    assertions: &AsrAssertions,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_transcript_chars) = assertions.min_transcript_chars {
        outcomes.push(assertion(
            "min_transcript_chars",
            transcript.chars().count() >= min_transcript_chars,
            Some(format!(
                "transcript_chars={} min={min_transcript_chars}",
                transcript.chars().count()
            )),
        ));
    }
    if let Some(max_word_error_rate) = assertions.max_word_error_rate {
        outcomes.push(assertion(
            "max_word_error_rate",
            word_error_rate.is_some_and(|value| value <= max_word_error_rate),
            Some(format!(
                "word_error_rate={} max={max_word_error_rate}",
                word_error_rate
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "unavailable".to_owned())
            )),
        ));
    }
    for required in &assertions.required_substrings {
        outcomes.push(assertion(
            format!("required_substring:{required}"),
            contains_case_insensitive(transcript, required),
            Some(format!("transcript_contains={required}")),
        ));
    }
    if outcomes.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "transcript_non_empty".to_owned(),
            status: if transcript.trim().is_empty() {
                AcceptanceStatus::Fail
            } else {
                AcceptanceStatus::Pass
            },
            message: Some(format!("transcript_chars={}", transcript.chars().count())),
        });
    }
    outcomes
}

fn render_plain_transcript(segments: &[TranscriptSegment]) -> Option<String> {
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[allow(dead_code)]
fn final_segments(update: TranscriptionUpdate) -> Vec<TranscriptSegment> {
    update
        .segments
        .into_iter()
        .filter(|segment| segment.final_segment)
        .collect()
}

#[allow(dead_code)]
fn has_partial_transcript(update: &TranscriptionUpdate) -> bool {
    update
        .segments
        .iter()
        .any(|segment| !segment.final_segment && !segment.text.trim().is_empty())
}

#[allow(dead_code)]
fn streaming_chunk_samples(chunk_ms: u64) -> usize {
    let samples = u64::from(ASR_SAMPLE_RATE_HZ)
        .saturating_mul(chunk_ms)
        .saturating_div(1000)
        .max(1);
    usize::try_from(samples).unwrap_or(usize::MAX)
}

fn resolve_repo_path(path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    repo_root().join(path)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("bins/evals should live two levels below the repo root")
        .to_path_buf()
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference_words = normalize_words(reference);
    let hypothesis_words = normalize_words(hypothesis);
    if reference_words.is_empty() {
        return if hypothesis_words.is_empty() {
            0.0
        } else {
            1.0
        };
    }
    edit_distance(&reference_words, &hypothesis_words) as f64 / reference_words.len() as f64
}

fn normalize_words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn edit_distance(left: &[String], right: &[String]) -> usize {
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_word) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_word) in right.iter().enumerate() {
            let substitution = previous[right_index] + usize::from(left_word != right_word);
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            current[right_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_normalizes_punctuation_and_case() {
        assert_eq!(word_error_rate("Hello, Motlie!", "hello motlie"), 0.0);
    }

    #[test]
    fn streaming_chunk_samples_clamps_to_one() {
        assert_eq!(streaming_chunk_samples(0), 1);
    }

    #[test]
    fn final_segments_filters_partial_finish_updates() {
        let update = TranscriptionUpdate {
            segments: vec![
                TranscriptSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "hel".to_owned(),
                    final_segment: false,
                },
                TranscriptSegment {
                    start_ms: 0,
                    end_ms: 200,
                    text: "hello".to_owned(),
                    final_segment: true,
                },
            ],
        };

        let segments = final_segments(update);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello");
    }

    #[test]
    fn asr_metric_gaps_distinguish_batch_and_streaming_null_ttfp() {
        let metrics = AsrPerformanceMetrics::default();

        let batch = required_asr_metric_gaps(&metrics, AsrMode::Batch);
        let streaming = required_asr_metric_gaps(&metrics, AsrMode::Streaming);

        assert_eq!(batch[0].metric, "ttfp_first_partial_ms");
        assert_eq!(batch[0].reason, "metric_not_applicable_for_batch_engine");
        assert_eq!(streaming[0].metric, "ttfp_first_partial_ms");
        assert_eq!(streaming[0].reason, "metric_not_reported_by_backend");
    }

    #[test]
    fn asr_eval_output_reports_mean_and_p95_over_iterations() {
        let output = asr_eval_output(
            10,
            20,
            AsrMode::Streaming,
            AsrIterationSummary {
                iterations: 3,
                successful_iterations: 3,
                failed_iterations: 0,
                request_latencies_ms: vec![100, 200, 300],
                ttfp_first_partial_ms: vec![10, 20, 30],
                segments: Vec::new(),
            },
        );

        assert_eq!(output.warmup_ms, 20);
        assert_eq!(output.request_latencies_ms, vec![100, 200, 300]);
        assert_eq!(output.mean_transcription_latency_ms, Some(200.0));
        assert_eq!(output.p95_transcription_latency_ms, Some(300.0));
        assert_eq!(output.mean_ttfp_first_partial_ms, Some(20.0));
        assert_eq!(output.p95_ttfp_first_partial_ms, Some(30.0));
    }
}
