use std::future::Future;

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, Mono, SpeechStream, SpeechSynthesizer, SynthesisRequest};
use motlie_model::{BundleHandle, ModelError, SpeechParams, StartOptions};

use crate::metrics::{
    CapabilityPerformanceMetrics, MetricUnavailable, PerformanceMetrics, TtsPerformanceMetrics,
};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, bundle_filter_capability_kind, elapsed_ms,
    evaluate_performance_measured, evaluate_resource_status, mean, observe_backend_accelerator,
    percentile, prepare_bundle,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, TtsAssertions};

pub struct TtsRunner;

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
        let request = SynthesisRequest {
            text: tts_scenario.input.text.clone(),
            params: SpeechParams {
                speaking_rate: tts_scenario.input.speaking_rate,
                ..Default::default()
            },
        };

        let eval = run_selected_tts(
            &mut context,
            &prepared,
            request,
            tts_scenario.input.iterations,
            tts_scenario.input.warmup_iterations,
        )
        .await?;

        let tts_metrics = TtsPerformanceMetrics {
            iterations: Some(eval.iterations),
            successful_iterations: Some(eval.successful_iterations),
            failed_iterations: Some(eval.failed_iterations),
            last_iteration_error: eval.last_iteration_error,
            text_chars: Some(tts_scenario.input.text.chars().count() as u64),
            synthesis_latency_ms: None,
            mean_synthesis_latency_ms: eval.mean_synthesis_latency_ms,
            p95_synthesis_latency_ms: eval.p95_synthesis_latency_ms,
            ttfa_first_chunk_ms: None,
            ttfa_first_chunk_samples_ms: eval.ttfa_first_chunk_samples_ms,
            mean_ttfa_first_chunk_ms: eval.mean_ttfa_first_chunk_ms,
            p95_ttfa_first_chunk_ms: eval.p95_ttfa_first_chunk_ms,
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
        let assertions = evaluate_assertions(
            eval.audio_duration_ms,
            eval.sample_count,
            &tts_scenario.assertions,
        );
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

struct TtsEvalOutput {
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
    audio_duration_ms: u64,
    sample_count: u64,
    sample_rate_hz: u64,
    chunk_count: u64,
}

struct TtsIterationMetrics {
    synthesis_latency_ms: u64,
    ttfa_first_chunk_ms: Option<u64>,
    audio_duration_ms: u64,
    sample_count: u64,
    chunk_count: u64,
}

struct TtsIterationSummary {
    iterations: u64,
    successful_iterations: u64,
    failed_iterations: u64,
    last_iteration_error: Option<String>,
    request_latencies_ms: Vec<u64>,
    ttfa_first_chunk_ms: Vec<u64>,
    last_iteration: Option<TtsIterationMetrics>,
}

#[allow(unused_variables)]
async fn run_selected_tts(
    context: &mut RunContext,
    prepared: &crate::runner::support::PreparedBundle,
    request: SynthesisRequest,
    iterations: u64,
    warmup_iterations: u64,
) -> Result<TtsEvalOutput> {
    let bundle_id = prepared.bundle_id.as_str();

    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    if bundle_id == "piper_en_us_ljspeech_medium" {
        return run_typed_tts::<_, _, _, i16, 22_050>(
            context,
            crate::runner::support::start_options(context, prepared),
            request,
            iterations,
            warmup_iterations,
            motlie_models::tts::piper_en_us_ljspeech_medium::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-kokoro-82m")]
    if bundle_id == "kokoro_82m" {
        return run_typed_tts::<_, _, _, i16, 24_000>(
            context,
            crate::runner::support::start_options(context, prepared),
            request,
            iterations,
            warmup_iterations,
            motlie_models::tts::kokoro_82m::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-qwen3-tts-cpp")]
    if bundle_id == "qwen3_tts_cpp_0_6b" {
        return run_typed_tts::<_, _, _, f32, 24_000>(
            context,
            crate::runner::support::start_options(context, prepared),
            request,
            iterations,
            warmup_iterations,
            motlie_models::tts::qwen3_tts_cpp::start_typed,
        )
        .await;
    }

    bail!("TTS bundle `{bundle_id}` is not enabled or not supported by the eval runner")
}

#[allow(dead_code)]
async fn run_typed_tts<Handle, Start, StartFuture, Sample, const RATE_HZ: u32>(
    context: &mut RunContext,
    options: StartOptions,
    request: SynthesisRequest,
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
    let startup_started_at = std::time::Instant::now();
    let handle = start(options).await.context("failed to start TTS bundle")?;
    let startup_ms = elapsed_ms(startup_started_at.elapsed());
    observe_backend_accelerator(context, &handle);
    context.metrics_sampler.sample();

    let warmup_started_at = std::time::Instant::now();
    for _ in 0..warmup_iterations {
        let _ = run_one_typed_tts::<_, Sample, RATE_HZ>(&handle, request.clone()).await?;
        context.metrics_sampler.sample();
    }
    let warmup_ms = elapsed_ms(warmup_started_at.elapsed());

    let mut request_latencies_ms = Vec::new();
    let mut ttfa_first_chunk_ms = Vec::new();
    let mut successful_iterations = 0_u64;
    let mut failed_iterations = 0_u64;
    let mut last_iteration_error = None;
    let mut last_iteration = None;
    for _ in 0..iterations {
        match run_one_typed_tts::<_, Sample, RATE_HZ>(&handle, request.clone()).await {
            Ok(iteration) => {
                request_latencies_ms.push(iteration.synthesis_latency_ms);
                push_if_some(&mut ttfa_first_chunk_ms, iteration.ttfa_first_chunk_ms);
                last_iteration = Some(iteration);
                successful_iterations += 1;
            }
            Err(error) => {
                failed_iterations += 1;
                last_iteration_error = Some(error.to_string());
            }
        }
        context.metrics_sampler.sample();
    }

    handle.shutdown().await.context("TTS shutdown failed")?;

    Ok(tts_eval_output(
        startup_ms,
        warmup_ms,
        TtsIterationSummary {
            iterations,
            successful_iterations,
            failed_iterations,
            last_iteration_error,
            request_latencies_ms,
            ttfa_first_chunk_ms,
            last_iteration,
        },
        u64::from(RATE_HZ),
    ))
}

async fn run_one_typed_tts<Handle, Sample, const RATE_HZ: u32>(
    handle: &Handle,
    request: SynthesisRequest,
) -> Result<TtsIterationMetrics>
where
    Handle: SpeechSynthesizer<Request = SynthesisRequest, Output = AudioBuf<Sample, RATE_HZ, Mono>>,
    Sample: Clone + Send + Sync + 'static,
{
    let synthesis_started_at = std::time::Instant::now();
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
        chunk_count,
    })
}

fn tts_eval_output(
    startup_ms: u64,
    warmup_ms: u64,
    summary: TtsIterationSummary,
    sample_rate_hz: u64,
) -> TtsEvalOutput {
    let mean_synthesis_latency_ms = mean(&summary.request_latencies_ms);
    let p95_synthesis_latency_ms = percentile(&summary.request_latencies_ms, 0.95);
    let mean_ttfa_first_chunk_ms = mean(&summary.ttfa_first_chunk_ms);
    let p95_ttfa_first_chunk_ms = percentile(&summary.ttfa_first_chunk_ms, 0.95);
    let (audio_duration_ms, sample_count, chunk_count) = summary
        .last_iteration
        .map(|iteration| {
            (
                iteration.audio_duration_ms,
                iteration.sample_count,
                iteration.chunk_count,
            )
        })
        .unwrap_or((0, 0, 0));

    TtsEvalOutput {
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

fn evaluate_assertions(
    audio_duration_ms: u64,
    sample_count: u64,
    assertions: &TtsAssertions,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_audio_duration_ms) = assertions.min_audio_duration_ms {
        outcomes.push(assertion(
            "min_audio_duration_ms",
            audio_duration_ms >= min_audio_duration_ms,
            Some(format!(
                "audio_duration_ms={audio_duration_ms} min={min_audio_duration_ms}"
            )),
        ));
    }
    if let Some(min_sample_count) = assertions.min_sample_count {
        outcomes.push(assertion(
            "min_sample_count",
            sample_count >= min_sample_count,
            Some(format!(
                "sample_count={sample_count} min={min_sample_count}"
            )),
        ));
    }
    if outcomes.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "audio_non_empty".to_owned(),
            status: if sample_count > 0 {
                AcceptanceStatus::Pass
            } else {
                AcceptanceStatus::Fail
            },
            message: Some(format!("sample_count={sample_count}")),
        });
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn min_duration_assertion_names_gate() {
        let outcomes = evaluate_assertions(
            100,
            1,
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
                last_iteration: Some(TtsIterationMetrics {
                    synthesis_latency_ms: 50,
                    ttfa_first_chunk_ms: Some(30),
                    audio_duration_ms: 1000,
                    sample_count: 16_000,
                    chunk_count: 4,
                }),
            },
            16_000,
        );

        assert_eq!(output.warmup_ms, 20);
        assert_eq!(output.request_latencies_ms, vec![30, 40, 50]);
        assert_eq!(output.mean_synthesis_latency_ms, Some(40.0));
        assert_eq!(output.p95_synthesis_latency_ms, Some(50.0));
        assert_eq!(output.last_iteration_error.as_deref(), Some("last failure"));
        assert_eq!(output.ttfa_first_chunk_samples_ms, vec![10, 20, 30]);
        assert_eq!(output.mean_ttfa_first_chunk_ms, Some(20.0));
        assert_eq!(output.p95_ttfa_first_chunk_ms, Some(30.0));
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

    struct CountingTtsRun {
        output: TtsEvalOutput,
        calls: Vec<u64>,
    }

    async fn run_counting_tts(warmup_iterations: u64, iterations: u64) -> CountingTtsRun {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let start_calls = Arc::clone(&calls);
        let mut context = test_context();
        let output = run_typed_tts::<_, _, _, i16, 22_050>(
            &mut context,
            StartOptions::default(),
            SynthesisRequest {
                text: "hello".to_owned(),
                params: SpeechParams::default(),
            },
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
