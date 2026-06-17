use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::typed::{
    AudioBuf, IncrementalSpeechCancelToken, IncrementalSpeechChunk, IncrementalSpeechControls,
    IncrementalSpeechRequestLabel, IncrementalSpeechStream, IncrementalSpeechSynthesizer, Mono,
    SpeechStream, SpeechSynthesizer, SynthesisRequest,
};
use motlie_model::{BundleHandle, ModelError, SpeechParams, StartOptions};

use crate::metrics::{
    CapabilityPerformanceMetrics, MetricUnavailable, PerformanceMetrics,
    TtsLengthPerformanceMetrics, TtsPerformanceMetrics,
};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, bundle_filter_capability_kind, elapsed_ms,
    evaluate_performance_measured, evaluate_resource_status, mean, observe_backend_accelerator,
    percentile, prepare_bundle,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, TtsAssertions, TtsCorpusItem, TtsInput, TtsSynthesisMode};

pub struct TtsRunner;

const STREAMING_EVAL_FRAME_MS: u64 = 20;
#[cfg(any(feature = "model-kokoro-82m", test))]
const STREAMING_EVAL_MAX_BUFFERED_AUDIO_MS: u32 = 80;
const STREAMING_EVAL_REQUEST_LABEL: &str = "evals-tts-streaming-synthesis";

#[async_trait]
impl ScenarioRunner for TtsRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::Tts,
            "scenario `{}` is not a TTS scenario",
            context.scenario.id
        );
        let tts_scenario = context
            .scenario
            .tts()
            .context("TTS scenario should carry TTS input/assertions")?
            .clone();
        let prepared = prepare_bundle(
            &context,
            bundle_filter_capability_kind(context.scenario.bundle_filter.capability),
            &[],
        )?;
        let cases = tts_eval_cases(&tts_scenario.input);

        let eval = run_selected_tts(&mut context, &prepared, &cases, &tts_scenario.input).await?;
        let assertions = evaluate_assertions(&eval, &tts_scenario.assertions);

        let tts_metrics = TtsPerformanceMetrics {
            iterations: Some(eval.iterations),
            successful_iterations: Some(eval.successful_iterations),
            failed_iterations: Some(eval.failed_iterations),
            last_iteration_error: eval.last_iteration_error,
            text_chars: Some(eval.text_chars),
            synthesis_latency_ms: None,
            mean_synthesis_latency_ms: eval.mean_synthesis_latency_ms,
            p95_synthesis_latency_ms: eval.p95_synthesis_latency_ms,
            ttfa_first_chunk_ms: None,
            ttfa_first_chunk_samples_ms: eval.ttfa_first_chunk_samples_ms,
            mean_ttfa_first_chunk_ms: eval.mean_ttfa_first_chunk_ms,
            p95_ttfa_first_chunk_ms: eval.p95_ttfa_first_chunk_ms,
            streaming_proof_first_pcm_before_synth_complete: eval
                .streaming_proof_first_pcm_before_synth_complete,
            synth_complete_samples_ms: eval.synth_complete_samples_ms.clone(),
            mean_synth_complete_ms: eval.mean_synth_complete_ms,
            p95_synth_complete_ms: eval.p95_synth_complete_ms,
            inter_chunk_gap_samples_ms: eval.inter_chunk_gap_samples_ms.clone(),
            mean_inter_chunk_gap_ms: eval.mean_inter_chunk_gap_ms,
            p95_inter_chunk_gap_ms: eval.p95_inter_chunk_gap_ms,
            max_inter_chunk_gap_ms: eval.max_inter_chunk_gap_ms,
            underrun_count: eval.underrun_count,
            streaming_frame_ms: eval.streaming_frame_ms,
            packetized_frame_count: eval.packetized_frame_count,
            max_buffered_audio_ms: eval.max_buffered_audio_ms,
            per_length: eval.per_length.clone(),
            audio_duration_ms: Some(eval.audio_duration_ms),
            real_time_factor: eval.mean_synthesis_latency_ms.and_then(|value| {
                (eval.audio_duration_ms > 0).then_some(value / eval.audio_duration_ms as f64)
            }),
            sample_count: Some(eval.sample_count),
            sample_rate_hz: Some(eval.sample_rate_hz),
            chunk_count: Some(eval.chunk_count),
        };
        let unavailable = required_tts_metric_gaps(&tts_metrics);
        let performance = PerformanceMetrics {
            startup_ms: Some(eval.startup_ms),
            warmup_ms: Some(eval.warmup_ms),
            request_latencies_ms: eval.request_latencies_ms,
            unavailable,
            capability_metrics: CapabilityPerformanceMetrics::Tts(tts_metrics),
        };
        let resources = context.metrics_sampler.finish();
        let performance_evaluation = evaluate_performance_measured(
            performance.startup_ms.is_some() && !performance.request_latencies_ms.is_empty(),
            "performance metrics missing startup or synthesis latency",
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

#[derive(Default)]
struct TtsEvalOutput {
    text_chars: u64,
    per_length: Vec<TtsLengthPerformanceMetrics>,
    startup_ms: u64,
    warmup_ms: u64,
    iterations: u64,
    successful_iterations: u64,
    failed_iterations: u64,
    last_iteration_error: Option<String>,
    request_latencies_ms: Vec<u64>,
    ttfa_first_chunk_samples_ms: Vec<u64>,
    mean_synthesis_latency_ms: Option<f64>,
    p95_synthesis_latency_ms: Option<f64>,
    mean_ttfa_first_chunk_ms: Option<f64>,
    p95_ttfa_first_chunk_ms: Option<f64>,
    streaming_proof_first_pcm_before_synth_complete: Option<bool>,
    synth_complete_samples_ms: Vec<u64>,
    mean_synth_complete_ms: Option<f64>,
    p95_synth_complete_ms: Option<f64>,
    inter_chunk_gap_samples_ms: Vec<u64>,
    mean_inter_chunk_gap_ms: Option<f64>,
    p95_inter_chunk_gap_ms: Option<f64>,
    max_inter_chunk_gap_ms: Option<u64>,
    underrun_count: Option<u64>,
    streaming_frame_ms: Option<u64>,
    packetized_frame_count: Option<u64>,
    max_buffered_audio_ms: Option<u64>,
    audio_duration_ms: u64,
    sample_count: u64,
    sample_rate_hz: u64,
    chunk_count: u64,
}

#[derive(Clone, Default)]
struct TtsIterationMetrics {
    synthesis_latency_ms: u64,
    ttfa_first_chunk_ms: Option<u64>,
    synth_complete_ms: Option<u64>,
    streaming_proof_first_pcm_before_synth_complete: Option<bool>,
    inter_chunk_gap_ms: Vec<u64>,
    underrun_count: Option<u64>,
    streaming_frame_ms: Option<u64>,
    packetized_frame_count: Option<u64>,
    max_buffered_audio_ms: Option<u64>,
    audio_duration_ms: u64,
    sample_count: u64,
    sample_rate_hz: u64,
    chunk_count: u64,
}

struct TtsEvalCase {
    id: String,
    label: Option<String>,
    text_chars: u64,
    request: SynthesisRequest,
}

#[derive(Default)]
struct TtsIterationSummary {
    iterations: u64,
    successful_iterations: u64,
    failed_iterations: u64,
    last_iteration_error: Option<String>,
    request_latencies_ms: Vec<u64>,
    ttfa_first_chunk_ms: Vec<u64>,
    synth_complete_ms: Vec<u64>,
    streaming_proofs: Vec<bool>,
    inter_chunk_gap_ms: Vec<u64>,
    underrun_count: Option<u64>,
    streaming_frame_ms: Option<u64>,
    packetized_frame_count: Option<u64>,
    max_buffered_audio_ms: Option<u64>,
    last_iteration: Option<TtsIterationMetrics>,
}

fn tts_eval_cases(input: &TtsInput) -> Vec<TtsEvalCase> {
    let corpus = if input.corpus.is_empty() {
        vec![TtsCorpusItem {
            id: "default".to_owned(),
            label: None,
            text: input.text.clone(),
        }]
    } else {
        input.corpus.clone()
    };

    corpus
        .into_iter()
        .map(|item| {
            let text_chars = item.text.chars().count() as u64;
            TtsEvalCase {
                id: item.id,
                label: item.label,
                text_chars,
                request: SynthesisRequest {
                    text: item.text,
                    params: SpeechParams {
                        speaking_rate: input.speaking_rate,
                        ..Default::default()
                    },
                },
            }
        })
        .collect()
}

#[allow(unused_variables)]
async fn run_selected_tts(
    context: &mut RunContext,
    prepared: &crate::runner::support::PreparedBundle,
    cases: &[TtsEvalCase],
    input: &TtsInput,
) -> Result<TtsEvalOutput> {
    let bundle_id = prepared.bundle_id.as_str();

    if input.synthesis_mode == TtsSynthesisMode::Streaming {
        #[cfg(feature = "model-kokoro-82m")]
        if bundle_id == "kokoro_82m" {
            return run_incremental_tts(
                context,
                crate::runner::support::start_options(context, prepared),
                cases,
                input.iterations,
                input.warmup_iterations,
                input
                    .max_buffered_audio_ms
                    .unwrap_or(STREAMING_EVAL_MAX_BUFFERED_AUDIO_MS),
                motlie_models::tts::kokoro_82m::start_typed,
            )
            .await;
        }

        bail!(
            "streaming TTS scenario requires the kokoro_82m incremental bundle; `{bundle_id}` is not supported"
        );
    }

    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    if bundle_id == "piper_en_us_ljspeech_medium" {
        return run_typed_tts::<_, _, _, i16, 22_050>(
            context,
            crate::runner::support::start_options(context, prepared),
            cases,
            input.iterations,
            input.warmup_iterations,
            motlie_models::tts::piper_en_us_ljspeech_medium::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-kokoro-82m")]
    if bundle_id == "kokoro_82m" {
        return run_typed_tts::<_, _, _, i16, 24_000>(
            context,
            crate::runner::support::start_options(context, prepared),
            cases,
            input.iterations,
            input.warmup_iterations,
            motlie_models::tts::kokoro_82m::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-qwen3-tts-cpp")]
    if bundle_id == "qwen3_tts_cpp_0_6b" {
        return run_typed_tts::<_, _, _, f32, 24_000>(
            context,
            crate::runner::support::start_options(context, prepared),
            cases,
            input.iterations,
            input.warmup_iterations,
            motlie_models::tts::qwen3_tts_cpp_0_6b::start_typed,
        )
        .await;
    }

    bail!("TTS bundle `{bundle_id}` is not enabled or not supported by the eval runner")
}

#[allow(dead_code)]
async fn run_typed_tts<Handle, Start, StartFuture, Sample, const RATE_HZ: u32>(
    context: &mut RunContext,
    options: StartOptions,
    cases: &[TtsEvalCase],
    iterations: u64,
    warmup_iterations: u64,
    start: Start,
) -> Result<TtsEvalOutput>
where
    Handle: BundleHandle
        + SpeechSynthesizer<Request = SynthesisRequest, Output = AudioBuf<Sample, RATE_HZ, Mono>>,
    Start: FnOnce(StartOptions) -> StartFuture,
    StartFuture: Future<Output = std::result::Result<Handle, ModelError>> + Send,
    Sample: Clone + Send + Sync + 'static,
{
    context.metrics_sampler.sample();
    let startup_started_at = Instant::now();
    let handle = start(options).await.context("failed to start TTS bundle")?;
    let startup_ms = elapsed_ms(startup_started_at.elapsed());
    observe_backend_accelerator(context, &handle);
    context.metrics_sampler.sample();

    let warmup_request = cases
        .first()
        .context("TTS scenario has no input cases")?
        .request
        .clone();
    let warmup_started_at = Instant::now();
    for _ in 0..warmup_iterations {
        let _ = run_one_typed_tts::<_, Sample, RATE_HZ>(&handle, warmup_request.clone()).await?;
        context.metrics_sampler.sample();
    }
    let warmup_ms = elapsed_ms(warmup_started_at.elapsed());

    let mut summary = TtsIterationSummary {
        iterations: iterations.saturating_mul(cases.len() as u64),
        ..Default::default()
    };
    let mut per_length = Vec::new();
    for case in cases {
        let mut case_summary = TtsIterationSummary {
            iterations,
            ..Default::default()
        };
        for _ in 0..iterations {
            match run_one_typed_tts::<_, Sample, RATE_HZ>(&handle, case.request.clone()).await {
                Ok(iteration) => {
                    record_tts_iteration(&mut case_summary, iteration.clone());
                    record_tts_iteration(&mut summary, iteration);
                }
                Err(error) => {
                    let message = error.to_string();
                    case_summary.failed_iterations += 1;
                    case_summary.last_iteration_error = Some(message.clone());
                    summary.failed_iterations += 1;
                    summary.last_iteration_error = Some(message);
                }
            }
            context.metrics_sampler.sample();
        }
        per_length.push(tts_length_performance(case, case_summary));
    }

    handle.shutdown().await.context("TTS shutdown failed")?;

    Ok(tts_eval_output(startup_ms, warmup_ms, summary, per_length))
}

#[allow(dead_code)]
async fn run_incremental_tts<Handle, Start, StartFuture>(
    context: &mut RunContext,
    options: StartOptions,
    cases: &[TtsEvalCase],
    iterations: u64,
    warmup_iterations: u64,
    max_buffered_audio_ms: u32,
    start: Start,
) -> Result<TtsEvalOutput>
where
    Handle: BundleHandle + IncrementalSpeechSynthesizer<Request = SynthesisRequest>,
    Start: FnOnce(StartOptions) -> StartFuture,
    StartFuture: Future<Output = std::result::Result<Handle, ModelError>> + Send,
{
    context.metrics_sampler.sample();
    let startup_started_at = Instant::now();
    let handle = start(options).await.context("failed to start TTS bundle")?;
    let startup_ms = elapsed_ms(startup_started_at.elapsed());
    observe_backend_accelerator(context, &handle);
    context.metrics_sampler.sample();

    let warmup_request = cases
        .first()
        .context("TTS scenario has no input cases")?
        .request
        .clone();
    let warmup_started_at = Instant::now();
    for index in 0..warmup_iterations {
        let label = format!("{STREAMING_EVAL_REQUEST_LABEL}-warmup-{index}");
        let _ = run_one_incremental_tts(
            &handle,
            warmup_request.clone(),
            max_buffered_audio_ms,
            label,
        )
        .await?;
        context.metrics_sampler.sample();
    }
    let warmup_ms = elapsed_ms(warmup_started_at.elapsed());

    let mut summary = TtsIterationSummary {
        iterations: iterations.saturating_mul(cases.len() as u64),
        streaming_frame_ms: Some(STREAMING_EVAL_FRAME_MS),
        max_buffered_audio_ms: Some(u64::from(max_buffered_audio_ms)),
        underrun_count: Some(0),
        packetized_frame_count: Some(0),
        ..Default::default()
    };
    let mut per_length = Vec::new();
    for case in cases {
        let mut case_summary = TtsIterationSummary {
            iterations,
            streaming_frame_ms: Some(STREAMING_EVAL_FRAME_MS),
            max_buffered_audio_ms: Some(u64::from(max_buffered_audio_ms)),
            underrun_count: Some(0),
            packetized_frame_count: Some(0),
            ..Default::default()
        };
        for index in 0..iterations {
            let label = format!("{STREAMING_EVAL_REQUEST_LABEL}-{}-{index}", case.id);
            match run_one_incremental_tts(
                &handle,
                case.request.clone(),
                max_buffered_audio_ms,
                label,
            )
            .await
            {
                Ok(iteration) => {
                    record_tts_iteration(&mut case_summary, iteration.clone());
                    record_tts_iteration(&mut summary, iteration);
                }
                Err(error) => {
                    let message = error.to_string();
                    case_summary.failed_iterations += 1;
                    case_summary.last_iteration_error = Some(message.clone());
                    summary.failed_iterations += 1;
                    summary.last_iteration_error = Some(message);
                }
            }
            context.metrics_sampler.sample();
        }
        per_length.push(tts_length_performance(case, case_summary));
    }

    handle.shutdown().await.context("TTS shutdown failed")?;

    Ok(tts_eval_output(startup_ms, warmup_ms, summary, per_length))
}

async fn run_one_typed_tts<Handle, Sample, const RATE_HZ: u32>(
    handle: &Handle,
    request: SynthesisRequest,
) -> Result<TtsIterationMetrics>
where
    Handle: SpeechSynthesizer<Request = SynthesisRequest, Output = AudioBuf<Sample, RATE_HZ, Mono>>,
    Sample: Clone + Send + Sync + 'static,
{
    let synthesis_started_at = Instant::now();
    let mut stream = handle
        .synthesize(request)
        .await
        .context("failed to open typed speech stream")?;
    let mut sample_count = 0_u64;
    let mut chunk_count = 0_u64;
    let mut ttfa_first_chunk_ms = None;
    while let Some(chunk) = stream
        .next_chunk()
        .await
        .context("speech next_chunk failed")?
    {
        sample_count = sample_count.saturating_add(chunk.samples().len() as u64);
        chunk_count = chunk_count.saturating_add(1);
        if ttfa_first_chunk_ms.is_none() {
            ttfa_first_chunk_ms = Some(elapsed_ms(synthesis_started_at.elapsed()));
        }
    }
    stream
        .finish()
        .await
        .context("speech stream finish failed")?;
    let synthesis_latency_ms = elapsed_ms(synthesis_started_at.elapsed());

    let audio_duration_ms = sample_count.saturating_mul(1000) / u64::from(RATE_HZ);
    Ok(TtsIterationMetrics {
        synthesis_latency_ms,
        ttfa_first_chunk_ms,
        audio_duration_ms,
        sample_count,
        sample_rate_hz: u64::from(RATE_HZ),
        chunk_count,
        ..Default::default()
    })
}

async fn run_one_incremental_tts<Handle>(
    handle: &Handle,
    request: SynthesisRequest,
    max_buffered_audio_ms: u32,
    request_label: String,
) -> Result<TtsIterationMetrics>
where
    Handle: IncrementalSpeechSynthesizer<Request = SynthesisRequest>,
{
    let synthesis_started_at = Instant::now();
    let controls = IncrementalSpeechControls {
        cancel: IncrementalSpeechCancelToken::new(),
        request_label: Some(IncrementalSpeechRequestLabel::new(request_label)),
        max_buffered_audio_ms,
    };
    let mut stream = handle
        .synthesize_incremental(request, controls)
        .await
        .context("failed to open incremental speech stream")?;

    let mut sample_count = 0_u64;
    let mut audio_duration_ms = 0_u64;
    let mut sample_rate_hz = 0_u64;
    let mut chunk_count = 0_u64;
    let mut ttfa_first_chunk_ms = None;
    let mut ttfa_first_chunk_elapsed = None;
    let mut last_chunk_at = None;
    let mut buffered_audio_budget = Duration::ZERO;
    let mut inter_chunk_gap_ms = Vec::new();
    let mut underrun_count = 0_u64;
    let mut packetized_frame_count = 0_u64;

    while let Some(chunk) = stream
        .next_audio_chunk()
        .await
        .context("incremental speech next_audio_chunk failed")?
    {
        if chunk.samples_i16.is_empty() {
            continue;
        }
        let now = Instant::now();
        if ttfa_first_chunk_elapsed.is_none() {
            let elapsed = now.saturating_duration_since(synthesis_started_at);
            ttfa_first_chunk_ms = Some(elapsed_ms(elapsed));
            ttfa_first_chunk_elapsed = Some(elapsed);
        }
        if let Some(last) = last_chunk_at {
            let gap = now.saturating_duration_since(last);
            let gap_ms = elapsed_ms(gap);
            inter_chunk_gap_ms.push(gap_ms);
            if gap >= buffered_audio_budget {
                if gap > buffered_audio_budget && !buffered_audio_budget.is_zero() {
                    underrun_count = underrun_count.saturating_add(1);
                }
                buffered_audio_budget = Duration::ZERO;
            } else {
                buffered_audio_budget -= gap;
            }
        }

        let frames = streaming_frame_count(&chunk, STREAMING_EVAL_FRAME_MS);
        packetized_frame_count = packetized_frame_count.saturating_add(frames);
        buffered_audio_budget +=
            Duration::from_millis(frames.saturating_mul(STREAMING_EVAL_FRAME_MS));
        sample_count = sample_count.saturating_add(chunk.samples_i16.len() as u64);
        audio_duration_ms = audio_duration_ms.saturating_add(chunk.audio_ms());
        sample_rate_hz = u64::from(chunk.sample_rate_hz);
        chunk_count = chunk_count.saturating_add(1);
        last_chunk_at = Some(now);
    }

    let stream_summary = stream
        .finish()
        .await
        .context("incremental speech stream finish failed")?;
    let synth_complete_elapsed = synthesis_started_at.elapsed();
    let synth_complete_ms = elapsed_ms(synth_complete_elapsed);
    let streaming_proof = ttfa_first_chunk_elapsed
        .map(|first_pcm| first_pcm < synth_complete_elapsed)
        .unwrap_or(false)
        && stream_summary.synthesis_completed;

    if audio_duration_ms == 0 {
        audio_duration_ms = stream_summary.audio_ms;
    }
    if chunk_count == 0 {
        chunk_count = stream_summary.chunks;
    }

    Ok(TtsIterationMetrics {
        synthesis_latency_ms: synth_complete_ms,
        ttfa_first_chunk_ms,
        synth_complete_ms: Some(synth_complete_ms),
        streaming_proof_first_pcm_before_synth_complete: Some(streaming_proof),
        inter_chunk_gap_ms,
        underrun_count: Some(underrun_count),
        streaming_frame_ms: Some(STREAMING_EVAL_FRAME_MS),
        packetized_frame_count: Some(packetized_frame_count),
        max_buffered_audio_ms: Some(u64::from(max_buffered_audio_ms)),
        audio_duration_ms,
        sample_count,
        sample_rate_hz,
        chunk_count,
    })
}

fn record_tts_iteration(summary: &mut TtsIterationSummary, iteration: TtsIterationMetrics) {
    summary
        .request_latencies_ms
        .push(iteration.synthesis_latency_ms);
    push_if_some(
        &mut summary.ttfa_first_chunk_ms,
        iteration.ttfa_first_chunk_ms,
    );
    push_if_some(&mut summary.synth_complete_ms, iteration.synth_complete_ms);
    if let Some(proof) = iteration.streaming_proof_first_pcm_before_synth_complete {
        summary.streaming_proofs.push(proof);
    }
    summary
        .inter_chunk_gap_ms
        .extend(iteration.inter_chunk_gap_ms.iter().copied());
    if let Some(count) = iteration.underrun_count {
        summary.underrun_count = Some(summary.underrun_count.unwrap_or(0).saturating_add(count));
    }
    if let Some(frames) = iteration.packetized_frame_count {
        summary.packetized_frame_count = Some(
            summary
                .packetized_frame_count
                .unwrap_or(0)
                .saturating_add(frames),
        );
    }
    if let Some(frame_ms) = iteration.streaming_frame_ms {
        summary.streaming_frame_ms = Some(frame_ms);
    }
    if let Some(max_buffered_audio_ms) = iteration.max_buffered_audio_ms {
        summary.max_buffered_audio_ms = Some(max_buffered_audio_ms);
    }
    summary.successful_iterations += 1;
    summary.last_iteration = Some(iteration);
}

fn streaming_frame_count(chunk: &IncrementalSpeechChunk, frame_ms: u64) -> u64 {
    if frame_ms == 0
        || chunk.samples_i16.is_empty()
        || chunk.sample_rate_hz == 0
        || chunk.channels == 0
    {
        return 0;
    }
    let samples_per_frame = (u64::from(chunk.sample_rate_hz)
        .saturating_mul(frame_ms)
        .div_ceil(1000))
    .max(1)
    .saturating_mul(u64::from(chunk.channels));
    (chunk.samples_i16.len() as u64).div_ceil(samples_per_frame)
}

fn tts_length_performance(
    case: &TtsEvalCase,
    summary: TtsIterationSummary,
) -> TtsLengthPerformanceMetrics {
    let mean_ttfa_first_chunk_ms = mean(&summary.ttfa_first_chunk_ms);
    let p95_ttfa_first_chunk_ms = percentile(&summary.ttfa_first_chunk_ms, 0.95);
    let mean_synth_complete_ms = mean(&summary.synth_complete_ms);
    let p95_synth_complete_ms = percentile(&summary.synth_complete_ms, 0.95);
    let mean_inter_chunk_gap_ms = mean(&summary.inter_chunk_gap_ms);
    let p95_inter_chunk_gap_ms = percentile(&summary.inter_chunk_gap_ms, 0.95);
    let streaming_proof_first_pcm_before_synth_complete = (!summary.streaming_proofs.is_empty())
        .then(|| summary.streaming_proofs.iter().all(|proof| *proof));
    let max_inter_chunk_gap_ms = summary.inter_chunk_gap_ms.iter().copied().max();
    let (audio_duration_ms, sample_count, sample_rate_hz, chunk_count) = summary
        .last_iteration
        .as_ref()
        .map(|iteration| {
            (
                iteration.audio_duration_ms,
                iteration.sample_count,
                iteration.sample_rate_hz,
                iteration.chunk_count,
            )
        })
        .unwrap_or((0, 0, 0, 0));
    let real_time_factor = mean_synth_complete_ms
        .and_then(|value| (audio_duration_ms > 0).then_some(value / audio_duration_ms as f64));

    TtsLengthPerformanceMetrics {
        id: case.id.clone(),
        label: case.label.clone(),
        text_chars: case.text_chars,
        iterations: summary.iterations,
        successful_iterations: summary.successful_iterations,
        failed_iterations: summary.failed_iterations,
        ttfa_first_chunk_samples_ms: summary.ttfa_first_chunk_ms,
        mean_ttfa_first_chunk_ms,
        p95_ttfa_first_chunk_ms,
        streaming_proof_first_pcm_before_synth_complete,
        synth_complete_samples_ms: summary.synth_complete_ms,
        mean_synth_complete_ms,
        p95_synth_complete_ms,
        inter_chunk_gap_samples_ms: summary.inter_chunk_gap_ms,
        mean_inter_chunk_gap_ms,
        p95_inter_chunk_gap_ms,
        max_inter_chunk_gap_ms,
        underrun_count: summary.underrun_count,
        streaming_frame_ms: summary.streaming_frame_ms,
        packetized_frame_count: summary.packetized_frame_count,
        max_buffered_audio_ms: summary.max_buffered_audio_ms,
        audio_duration_ms: Some(audio_duration_ms),
        real_time_factor,
        sample_count: Some(sample_count),
        sample_rate_hz: Some(sample_rate_hz),
        chunk_count: Some(chunk_count),
    }
}

fn tts_eval_output(
    startup_ms: u64,
    warmup_ms: u64,
    summary: TtsIterationSummary,
    per_length: Vec<TtsLengthPerformanceMetrics>,
) -> TtsEvalOutput {
    let text_chars = per_length.iter().map(|metric| metric.text_chars).sum();
    let mean_synthesis_latency_ms = mean(&summary.request_latencies_ms);
    let p95_synthesis_latency_ms = percentile(&summary.request_latencies_ms, 0.95);
    let mean_ttfa_first_chunk_ms = mean(&summary.ttfa_first_chunk_ms);
    let p95_ttfa_first_chunk_ms = percentile(&summary.ttfa_first_chunk_ms, 0.95);
    let mean_synth_complete_ms = mean(&summary.synth_complete_ms);
    let p95_synth_complete_ms = percentile(&summary.synth_complete_ms, 0.95);
    let mean_inter_chunk_gap_ms = mean(&summary.inter_chunk_gap_ms);
    let p95_inter_chunk_gap_ms = percentile(&summary.inter_chunk_gap_ms, 0.95);
    let streaming_proof_first_pcm_before_synth_complete = (!summary.streaming_proofs.is_empty())
        .then(|| summary.streaming_proofs.iter().all(|proof| *proof));
    let max_inter_chunk_gap_ms = summary.inter_chunk_gap_ms.iter().copied().max();
    let (audio_duration_ms, sample_count, sample_rate_hz, chunk_count) = summary
        .last_iteration
        .as_ref()
        .map(|iteration| {
            (
                iteration.audio_duration_ms,
                iteration.sample_count,
                iteration.sample_rate_hz,
                iteration.chunk_count,
            )
        })
        .unwrap_or((0, 0, 0, 0));

    TtsEvalOutput {
        text_chars,
        per_length,
        startup_ms,
        warmup_ms,
        iterations: summary.iterations,
        successful_iterations: summary.successful_iterations,
        failed_iterations: summary.failed_iterations,
        last_iteration_error: summary.last_iteration_error,
        mean_synthesis_latency_ms,
        p95_synthesis_latency_ms,
        mean_ttfa_first_chunk_ms,
        p95_ttfa_first_chunk_ms,
        streaming_proof_first_pcm_before_synth_complete,
        synth_complete_samples_ms: summary.synth_complete_ms,
        mean_synth_complete_ms,
        p95_synth_complete_ms,
        inter_chunk_gap_samples_ms: summary.inter_chunk_gap_ms,
        mean_inter_chunk_gap_ms,
        p95_inter_chunk_gap_ms,
        max_inter_chunk_gap_ms,
        underrun_count: summary.underrun_count,
        streaming_frame_ms: summary.streaming_frame_ms,
        packetized_frame_count: summary.packetized_frame_count,
        max_buffered_audio_ms: summary.max_buffered_audio_ms,
        request_latencies_ms: summary.request_latencies_ms,
        ttfa_first_chunk_samples_ms: summary.ttfa_first_chunk_ms,
        audio_duration_ms,
        sample_count,
        sample_rate_hz,
        chunk_count,
    }
}

fn required_tts_metric_gaps(metrics: &TtsPerformanceMetrics) -> Vec<MetricUnavailable> {
    if metrics.mean_ttfa_first_chunk_ms.is_some() {
        return Vec::new();
    }

    vec![MetricUnavailable::new(
        "ttfa_first_chunk_ms",
        "metric_not_reported_by_backend",
        "tts_runner",
    )]
}

fn push_if_some(values: &mut Vec<u64>, value: Option<u64>) {
    if let Some(value) = value {
        values.push(value);
    }
}

fn tts_min_audio_duration_pass(eval: &TtsEvalOutput, min_audio_duration_ms: u64) -> bool {
    if eval.per_length.is_empty() {
        return eval.audio_duration_ms >= min_audio_duration_ms;
    }
    eval.per_length
        .iter()
        .all(|metric| metric.audio_duration_ms.unwrap_or(0) >= min_audio_duration_ms)
}

fn tts_min_sample_count_pass(eval: &TtsEvalOutput, min_sample_count: u64) -> bool {
    if eval.per_length.is_empty() {
        return eval.sample_count >= min_sample_count;
    }
    eval.per_length
        .iter()
        .all(|metric| metric.sample_count.unwrap_or(0) >= min_sample_count)
}

fn tts_streaming_proof_pass(eval: &TtsEvalOutput) -> bool {
    if eval.per_length.is_empty() {
        return eval.streaming_proof_first_pcm_before_synth_complete == Some(true);
    }
    eval.per_length
        .iter()
        .all(|metric| metric.streaming_proof_first_pcm_before_synth_complete == Some(true))
}

fn evaluate_assertions(eval: &TtsEvalOutput, assertions: &TtsAssertions) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_audio_duration_ms) = assertions.min_audio_duration_ms {
        outcomes.push(assertion(
            "min_audio_duration_ms",
            tts_min_audio_duration_pass(eval, min_audio_duration_ms),
            Some(format!(
                "audio_duration_ms={} min={min_audio_duration_ms} length_count={}",
                eval.audio_duration_ms,
                eval.per_length.len()
            )),
        ));
    }
    if let Some(min_sample_count) = assertions.min_sample_count {
        outcomes.push(assertion(
            "min_sample_count",
            tts_min_sample_count_pass(eval, min_sample_count),
            Some(format!(
                "sample_count={} min={min_sample_count} length_count={}",
                eval.sample_count,
                eval.per_length.len()
            )),
        ));
    }
    if assertions.require_streaming_proof {
        outcomes.push(assertion(
            "streaming_first_pcm_before_synth_complete",
            tts_streaming_proof_pass(eval),
            Some(format!(
                "proof={:?} mean_ttfa_first_chunk_ms={:?} mean_synth_complete_ms={:?} length_count={}",
                eval.streaming_proof_first_pcm_before_synth_complete,
                eval.mean_ttfa_first_chunk_ms,
                eval.mean_synth_complete_ms,
                eval.per_length.len()
            )),
        ));
    }
    if outcomes.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "audio_non_empty".to_owned(),
            status: if eval.sample_count > 0 {
                AcceptanceStatus::Pass
            } else {
                AcceptanceStatus::Fail
            },
            message: Some(format!("sample_count={}", eval.sample_count)),
        });
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::typed::IncrementalSpeechSummary;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[test]
    fn min_duration_assertion_names_gate() {
        let eval = TtsEvalOutput {
            audio_duration_ms: 100,
            sample_count: 1,
            ..Default::default()
        };
        let outcomes = evaluate_assertions(
            &eval,
            &TtsAssertions {
                min_audio_duration_ms: Some(200),
                ..Default::default()
            },
        );

        assert_eq!(outcomes[0].name, "min_audio_duration_ms");
        assert_eq!(outcomes[0].status, AcceptanceStatus::Fail);
    }

    #[test]
    fn tts_metric_gap_records_null_ttfa() {
        let gaps = required_tts_metric_gaps(&TtsPerformanceMetrics::default());

        assert_eq!(gaps[0].metric, "ttfa_first_chunk_ms");
        assert_eq!(gaps[0].reason, "metric_not_reported_by_backend");
        assert_eq!(gaps[0].source.as_deref(), Some("tts_runner"));
    }

    #[test]
    fn tts_eval_output_reports_mean_and_p95_over_iterations() {
        let output = tts_eval_output(
            10,
            20,
            TtsIterationSummary {
                iterations: 4,
                successful_iterations: 3,
                failed_iterations: 1,
                last_iteration_error: Some("last failure".to_owned()),
                request_latencies_ms: vec![30, 40, 50],
                ttfa_first_chunk_ms: vec![10, 20, 30],
                synth_complete_ms: vec![35, 45, 55],
                streaming_proofs: vec![true, true, true],
                inter_chunk_gap_ms: vec![3, 4, 5],
                underrun_count: Some(0),
                streaming_frame_ms: Some(20),
                packetized_frame_count: Some(12),
                max_buffered_audio_ms: Some(80),
                last_iteration: Some(TtsIterationMetrics {
                    synthesis_latency_ms: 50,
                    ttfa_first_chunk_ms: Some(30),
                    synth_complete_ms: Some(55),
                    streaming_proof_first_pcm_before_synth_complete: Some(true),
                    inter_chunk_gap_ms: vec![5],
                    underrun_count: Some(0),
                    streaming_frame_ms: Some(20),
                    packetized_frame_count: Some(4),
                    max_buffered_audio_ms: Some(80),
                    audio_duration_ms: 1000,
                    sample_count: 16_000,
                    sample_rate_hz: 16_000,
                    chunk_count: 4,
                }),
            },
            vec![TtsLengthPerformanceMetrics {
                id: "case".to_owned(),
                text_chars: 12,
                ..Default::default()
            }],
        );

        assert_eq!(output.warmup_ms, 20);
        assert_eq!(output.text_chars, 12);
        assert_eq!(output.per_length.len(), 1);
        assert_eq!(output.request_latencies_ms, vec![30, 40, 50]);
        assert_eq!(output.mean_synthesis_latency_ms, Some(40.0));
        assert_eq!(output.p95_synthesis_latency_ms, Some(50.0));
        assert_eq!(output.last_iteration_error.as_deref(), Some("last failure"));
        assert_eq!(output.ttfa_first_chunk_samples_ms, vec![10, 20, 30]);
        assert_eq!(output.mean_ttfa_first_chunk_ms, Some(20.0));
        assert_eq!(output.p95_ttfa_first_chunk_ms, Some(30.0));
        assert_eq!(output.synth_complete_samples_ms, vec![35, 45, 55]);
        assert_eq!(output.mean_synth_complete_ms, Some(45.0));
        assert_eq!(output.p95_synth_complete_ms, Some(55.0));
        assert_eq!(
            output.streaming_proof_first_pcm_before_synth_complete,
            Some(true)
        );
        assert_eq!(output.inter_chunk_gap_samples_ms, vec![3, 4, 5]);
        assert_eq!(output.mean_inter_chunk_gap_ms, Some(4.0));
        assert_eq!(output.p95_inter_chunk_gap_ms, Some(5.0));
        assert_eq!(output.max_inter_chunk_gap_ms, Some(5));
        assert_eq!(output.underrun_count, Some(0));
        assert_eq!(output.streaming_frame_ms, Some(20));
        assert_eq!(output.packetized_frame_count, Some(12));
        assert_eq!(output.max_buffered_audio_ms, Some(80));
        assert_eq!(output.audio_duration_ms, 1000);
        assert_eq!(output.sample_rate_hz, 16_000);
    }

    #[tokio::test]
    async fn tts_warmup_iterations_control_measured_call() {
        let cold = run_counting_tts(0, 1).await;
        let warm = run_counting_tts(2, 1).await;

        assert_eq!(cold.calls, vec![1]);
        assert_eq!(cold.output.sample_count, 1);
        assert_eq!(cold.output.ttfa_first_chunk_samples_ms.len(), 1);
        assert_eq!(warm.calls, vec![1, 2, 3]);
        assert_eq!(warm.output.sample_count, 3);
        assert_eq!(warm.output.ttfa_first_chunk_samples_ms.len(), 1);
    }

    #[tokio::test]
    async fn streaming_tts_records_first_pcm_before_synth_complete() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let start_calls = Arc::clone(&calls);
        let mut context = test_context();
        let cases = test_cases("hello");
        let output = run_incremental_tts(
            &mut context,
            StartOptions::default(),
            &cases,
            1,
            0,
            STREAMING_EVAL_MAX_BUFFERED_AUDIO_MS,
            move |_| async move { Ok(CountingTts { calls: start_calls }) },
        )
        .await
        .unwrap();

        assert_eq!(calls.lock().unwrap().as_slice(), &[1]);
        assert_eq!(
            output.streaming_proof_first_pcm_before_synth_complete,
            Some(true)
        );
        assert_eq!(output.successful_iterations, 1);
        assert_eq!(output.failed_iterations, 0);
        assert_eq!(output.sample_count, 40);
        assert_eq!(output.sample_rate_hz, 1_000);
        assert_eq!(output.chunk_count, 2);
        assert_eq!(output.streaming_frame_ms, Some(STREAMING_EVAL_FRAME_MS));
        assert_eq!(output.packetized_frame_count, Some(2));
        assert_eq!(output.max_buffered_audio_ms, Some(80));
        assert_eq!(output.underrun_count, Some(0));
        assert_eq!(output.inter_chunk_gap_samples_ms.len(), 1);
        assert!(output.synth_complete_samples_ms[0] > output.ttfa_first_chunk_samples_ms[0]);
    }

    #[tokio::test]
    async fn streaming_tts_records_per_length_metrics() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let start_calls = Arc::clone(&calls);
        let mut context = test_context();
        let cases = vec![
            TtsEvalCase {
                id: "short".to_owned(),
                label: Some("short".to_owned()),
                text_chars: 5,
                request: SynthesisRequest {
                    text: "hello".to_owned(),
                    params: SpeechParams::default(),
                },
            },
            TtsEvalCase {
                id: "medium".to_owned(),
                label: Some("medium".to_owned()),
                text_chars: 24,
                request: SynthesisRequest {
                    text: "hello from a longer case".to_owned(),
                    params: SpeechParams::default(),
                },
            },
        ];

        let output = run_incremental_tts(
            &mut context,
            StartOptions::default(),
            &cases,
            1,
            1,
            STREAMING_EVAL_MAX_BUFFERED_AUDIO_MS,
            move |_| async move { Ok(CountingTts { calls: start_calls }) },
        )
        .await
        .unwrap();

        assert_eq!(calls.lock().unwrap().as_slice(), &[1, 2, 3]);
        assert_eq!(output.iterations, 2);
        assert_eq!(output.successful_iterations, 2);
        assert_eq!(output.per_length.len(), 2);
        assert_eq!(output.per_length[0].id, "short");
        assert_eq!(output.per_length[1].id, "medium");
        assert!(output.per_length.iter().all(|metric| {
            metric.streaming_proof_first_pcm_before_synth_complete == Some(true)
        }));
        assert!(tts_streaming_proof_pass(&output));
    }

    struct CountingTtsRun {
        output: TtsEvalOutput,
        calls: Vec<u64>,
    }

    async fn run_counting_tts(warmup_iterations: u64, iterations: u64) -> CountingTtsRun {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let start_calls = Arc::clone(&calls);
        let mut context = test_context();
        let cases = test_cases("hello");
        let output = run_typed_tts::<_, _, _, i16, 22_050>(
            &mut context,
            StartOptions::default(),
            &cases,
            iterations,
            warmup_iterations,
            move |_| async move { Ok(CountingTts { calls: start_calls }) },
        )
        .await
        .unwrap();
        let calls = calls.lock().unwrap().clone();

        CountingTtsRun { output, calls }
    }

    struct CountingTts {
        calls: Arc<Mutex<Vec<u64>>>,
    }

    #[async_trait]
    impl BundleHandle for CountingTts {
        type Chat = motlie_model::UnsupportedChat;
        type Completion = motlie_model::UnsupportedCompletion;
        type Embeddings = motlie_model::UnsupportedEmbeddings;

        fn descriptor(&self) -> &motlie_model::LoadedBundleDescriptor {
            unreachable!("test TTS handle descriptor is not used")
        }

        fn capabilities(&self) -> &motlie_model::Capabilities {
            unreachable!("test TTS handle capabilities are not used")
        }

        fn chat(&self) -> std::result::Result<&Self::Chat, ModelError> {
            Err(ModelError::UnsupportedCapability(
                motlie_model::CapabilityKind::Chat,
            ))
        }

        fn completion(&self) -> std::result::Result<&Self::Completion, ModelError> {
            Err(ModelError::UnsupportedCapability(
                motlie_model::CapabilityKind::Completion,
            ))
        }

        fn embeddings(&self) -> std::result::Result<&Self::Embeddings, ModelError> {
            Err(ModelError::UnsupportedCapability(
                motlie_model::CapabilityKind::Embeddings,
            ))
        }

        async fn shutdown(self) -> std::result::Result<(), ModelError> {
            Ok(())
        }
    }

    impl SpeechSynthesizer for CountingTts {
        type Request = SynthesisRequest;
        type Output = AudioBuf<i16, 22_050, Mono>;
        type Stream = CountingSpeechStream;

        async fn synthesize(
            &self,
            _request: Self::Request,
        ) -> std::result::Result<Self::Stream, ModelError> {
            let mut calls = self.calls.lock().unwrap();
            let call = calls.len() as u64 + 1;
            calls.push(call);
            Ok(CountingSpeechStream {
                chunk: Some(AudioBuf::new(vec![call as i16; call as usize])),
            })
        }
    }

    impl IncrementalSpeechSynthesizer for CountingTts {
        type Request = SynthesisRequest;
        type Stream = CountingIncrementalSpeechStream;

        async fn synthesize_incremental(
            &self,
            _request: Self::Request,
            _controls: IncrementalSpeechControls,
        ) -> std::result::Result<Self::Stream, ModelError> {
            let mut calls = self.calls.lock().unwrap();
            let call = calls.len() as u64 + 1;
            calls.push(call);
            Ok(CountingIncrementalSpeechStream {
                chunks: VecDeque::from([
                    IncrementalSpeechChunk {
                        samples_i16: vec![call as i16; 20],
                        sample_rate_hz: 1_000,
                        channels: 1,
                        chunk_index: 0,
                        is_final: false,
                    },
                    IncrementalSpeechChunk {
                        samples_i16: vec![call as i16; 20],
                        sample_rate_hz: 1_000,
                        channels: 1,
                        chunk_index: 1,
                        is_final: true,
                    },
                ]),
            })
        }
    }

    struct CountingIncrementalSpeechStream {
        chunks: VecDeque<IncrementalSpeechChunk>,
    }

    impl IncrementalSpeechStream for CountingIncrementalSpeechStream {
        async fn next_audio_chunk(
            &mut self,
        ) -> std::result::Result<Option<IncrementalSpeechChunk>, ModelError> {
            if self.chunks.is_empty() {
                return Ok(None);
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
            Ok(self.chunks.pop_front())
        }

        async fn finish(self) -> std::result::Result<IncrementalSpeechSummary, ModelError> {
            tokio::time::sleep(Duration::from_millis(25)).await;
            Ok(IncrementalSpeechSummary {
                chunks: 2,
                audio_ms: 40,
                canceled: false,
                synthesis_completed: true,
            })
        }
    }

    struct CountingSpeechStream {
        chunk: Option<AudioBuf<i16, 22_050, Mono>>,
    }

    impl SpeechStream for CountingSpeechStream {
        type Chunk = AudioBuf<i16, 22_050, Mono>;

        async fn next_chunk(&mut self) -> std::result::Result<Option<Self::Chunk>, ModelError> {
            Ok(self.chunk.take())
        }

        async fn finish(self) -> std::result::Result<(), ModelError> {
            Ok(())
        }
    }

    fn test_cases(text: &str) -> Vec<TtsEvalCase> {
        vec![TtsEvalCase {
            id: "test".to_owned(),
            label: None,
            text_chars: text.chars().count() as u64,
            request: SynthesisRequest {
                text: text.to_owned(),
                params: SpeechParams::default(),
            },
        }]
    }

    fn test_context() -> RunContext {
        let scenario = toml::from_str(
            r#"
schema_version = 1
id = "tts_test"
capability = "tts"
summary = "TTS test scenario."

[bundle_filter]
capability = "tts"

[input]
text = "hello"

[assertions]
min_sample_count = 1

[metrics]
capture_request_latency = true
"#,
        )
        .unwrap();

        RunContext {
            scenario,
            bundle_selection: crate::runner::BundleSelection {
                bundle_id: "test_tts".to_owned(),
                selector: None,
            },
            profile: crate::runner::ProfileSelection {
                name: "local-cpu-x86_64".to_owned(),
            },
            artifact_root: std::path::PathBuf::from("/tmp/motlie-test-artifacts"),
            runtime_flags: crate::runner::RuntimeFlags {
                command_line: Vec::new(),
                download_artifacts: false,
                precision: None,
                artifact_quantization: None,
                quiet_backend_logs: false,
                run_id: None,
            },
            coverage: None,
            accelerator: None,
            child_build: None,
            platform_collector: crate::platform::PlatformCollector::new("local-cpu-x86_64"),
            metrics_sampler: crate::metrics::MetricsSampler::new(),
            output_sink: crate::report::OutputSink::Stdout,
        }
    }
}
