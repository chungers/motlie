use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::{BundleHandle, ChatMessage, ChatModel, ChatRequest, ChatRole};

use crate::metrics::{
    CapabilityPerformanceMetrics, MetricUnavailable, PerfPerformanceMetrics, PerformanceMetrics,
};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, bundle_filter_capability_kind, elapsed_ms, evaluate_resource_status,
    mean, observe_backend_accelerator, percentile, prepare_bundle, start_options,
    SectionEvaluation,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, PerfAssertions};

pub struct PerfRunner;

#[async_trait]
impl ScenarioRunner for PerfRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::Perf,
            "scenario `{}` is not a perf scenario",
            context.scenario.id
        );
        let perf_scenario = context
            .scenario
            .perf()
            .context("perf scenario should carry perf input/assertions")?
            .clone();
        ensure!(
            perf_scenario.input.workload == "chat_generation",
            "perf workload `{}` is not supported yet",
            perf_scenario.input.workload
        );

        let prepared = prepare_bundle(
            &context,
            bundle_filter_capability_kind(context.scenario.bundle_filter.capability),
            &[],
        )?;
        let prompt = resolve_prompt(
            perf_scenario.input.prompt.as_deref(),
            perf_scenario.input.dataset.as_deref(),
        )?;

        context.metrics_sampler.sample();
        let startup_started_at = std::time::Instant::now();
        let handle = prepared
            .bundle
            .start(start_options(&context, &prepared))
            .await
            .with_context(|| format!("failed to start bundle `{}`", prepared.bundle_id))?;
        let startup_ms = elapsed_ms(startup_started_at.elapsed());
        observe_backend_accelerator(&mut context, &handle);
        context.metrics_sampler.sample();

        let chat = handle
            .chat()
            .context("selected bundle should expose chat")?;
        let warmup_started_at = std::time::Instant::now();
        for _ in 0..perf_scenario.input.warmup_iterations {
            let _ = run_one(chat, &prompt).await?;
            context.metrics_sampler.sample();
        }
        let warmup_ms = elapsed_ms(warmup_started_at.elapsed());

        let mut request_latencies_ms = Vec::new();
        let mut successful_iterations = 0_u64;
        let mut failed_iterations = 0_u64;
        let mut total_output_words = 0_u64;
        for _ in 0..perf_scenario.input.iterations {
            match run_one(chat, &prompt).await {
                Ok((latency_ms, words)) => {
                    request_latencies_ms.push(latency_ms);
                    successful_iterations += 1;
                    total_output_words += words;
                }
                Err(_) => {
                    failed_iterations += 1;
                }
            }
            context.metrics_sampler.sample();
        }

        handle
            .shutdown()
            .await
            .with_context(|| format!("failed to shut down bundle `{}`", prepared.bundle_id))?;

        let mean_latency_ms = mean(&request_latencies_ms);
        let p95_latency_ms = percentile(&request_latencies_ms, 0.95);
        let perf_metrics = PerfPerformanceMetrics {
            iterations: Some(perf_scenario.input.iterations),
            successful_iterations: Some(successful_iterations),
            failed_iterations: Some(failed_iterations),
            mean_latency_ms,
            p95_latency_ms,
            total_output_words: Some(total_output_words),
        };
        let performance = PerformanceMetrics {
            startup_ms: Some(startup_ms),
            warmup_ms: Some(warmup_ms),
            request_latencies_ms,
            unavailable: required_perf_metric_gaps(),
            capability_metrics: CapabilityPerformanceMetrics::Perf(perf_metrics),
        };
        let resources = context.metrics_sampler.finish();
        let assertions = evaluate_assertions(
            &perf_scenario.assertions,
            successful_iterations,
            mean_latency_ms,
            p95_latency_ms,
        );
        let performance_evaluation = evaluate_performance_status(
            &perf_scenario.assertions,
            mean_latency_ms,
            p95_latency_ms,
            &performance.unavailable,
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

async fn run_one<C: ChatModel + ?Sized>(chat: &C, prompt: &str) -> Result<(u64, u64)> {
    let started_at = std::time::Instant::now();
    let response = chat
        .generate(ChatRequest {
            messages: vec![
                ChatMessage::text(ChatRole::System, "Be concise."),
                ChatMessage::text(ChatRole::User, prompt),
            ],
            ..Default::default()
        })
        .await
        .context("chat generation failed")?;
    let latency_ms = elapsed_ms(started_at.elapsed());
    let words = response.content.split_whitespace().count() as u64;
    Ok((latency_ms, words))
}

fn resolve_prompt(prompt: Option<&str>, dataset: Option<&str>) -> Result<String> {
    if let Some(prompt) = prompt {
        return Ok(prompt.to_owned());
    }
    let dataset = dataset.context("perf input requires prompt or dataset")?;
    let path = dataset_path(dataset);
    std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read perf dataset `{}`", path.display()))
}

fn dataset_path(dataset: &str) -> PathBuf {
    let path = Path::new(dataset);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    repo_eval_root().join(path)
}

fn repo_eval_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("bins/evals should live two levels below the repo root")
        .join("evals")
}

fn evaluate_assertions(
    assertions: &PerfAssertions,
    successful_iterations: u64,
    mean_latency_ms: Option<f64>,
    p95_latency_ms: Option<f64>,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_successful_iterations) = assertions.min_successful_iterations {
        outcomes.push(assertion(
            "min_successful_iterations",
            successful_iterations >= min_successful_iterations,
            Some(format!(
                "successful_iterations={successful_iterations} min={min_successful_iterations}"
            )),
        ));
    }
    if let Some(max_mean_latency_ms) = assertions.max_mean_latency_ms {
        outcomes.push(assertion(
            "max_mean_latency_ms",
            mean_latency_ms.is_some_and(|value| value <= max_mean_latency_ms),
            Some(format!(
                "mean_latency_ms={} max={max_mean_latency_ms}",
                format_optional(mean_latency_ms)
            )),
        ));
    }
    if let Some(max_p95_latency_ms) = assertions.max_p95_latency_ms {
        outcomes.push(assertion(
            "max_p95_latency_ms",
            p95_latency_ms.is_some_and(|value| value <= max_p95_latency_ms),
            Some(format!(
                "p95_latency_ms={} max={max_p95_latency_ms}",
                format_optional(p95_latency_ms)
            )),
        ));
    }
    if outcomes.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "perf_iterations_completed".to_owned(),
            status: if successful_iterations > 0 {
                AcceptanceStatus::Pass
            } else {
                AcceptanceStatus::Fail
            },
            message: Some(format!("successful_iterations={successful_iterations}")),
        });
    }
    outcomes
}

fn required_perf_metric_gaps() -> Vec<MetricUnavailable> {
    vec![
        MetricUnavailable::new(
            "time_to_first_token_ms",
            "metric_not_instrumented",
            "perf_runner_streaming_callback",
        ),
        MetricUnavailable::new(
            "decode_ms",
            "metric_not_instrumented",
            "perf_runner_streaming_callback",
        ),
        MetricUnavailable::new(
            "output_tokens",
            "metric_not_instrumented",
            "perf_runner_tokenizer",
        ),
        MetricUnavailable::new(
            "tokens_per_second",
            "metric_not_instrumented",
            "perf_runner_decode_interval",
        ),
    ]
}

fn evaluate_performance_status(
    assertions: &PerfAssertions,
    mean_latency_ms: Option<f64>,
    p95_latency_ms: Option<f64>,
    _unavailable: &[MetricUnavailable],
) -> SectionEvaluation {
    if let Some(max_mean_latency_ms) = assertions.max_mean_latency_ms {
        match mean_latency_ms {
            Some(value) if value <= max_mean_latency_ms => {}
            Some(value) => {
                return SectionEvaluation {
                    status: AcceptanceStatus::Fail,
                    failure_reason: Some(format!(
                        "performance gate max_mean_latency_ms={max_mean_latency_ms} exceeded: mean_latency_ms={value:.2}"
                    )),
                };
            }
            None => {
                return SectionEvaluation {
                    status: AcceptanceStatus::NotMeasured,
                    failure_reason: Some(
                        "performance gate max_mean_latency_ms could not be evaluated".to_owned(),
                    ),
                };
            }
        }
    }
    if let Some(max_p95_latency_ms) = assertions.max_p95_latency_ms {
        match p95_latency_ms {
            Some(value) if value <= max_p95_latency_ms => {}
            Some(value) => {
                return SectionEvaluation {
                    status: AcceptanceStatus::Fail,
                    failure_reason: Some(format!(
                        "performance gate max_p95_latency_ms={max_p95_latency_ms} exceeded: p95_latency_ms={value:.2}"
                    )),
                };
            }
            None => {
                return SectionEvaluation {
                    status: AcceptanceStatus::NotMeasured,
                    failure_reason: Some(
                        "performance gate max_p95_latency_ms could not be evaluated".to_owned(),
                    ),
                };
            }
        }
    }

    SectionEvaluation {
        status: if mean_latency_ms.is_some() {
            AcceptanceStatus::Pass
        } else {
            AcceptanceStatus::NotMeasured
        },
        failure_reason: None,
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "unavailable".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn performance_failure_names_mean_gate() {
        let evaluation = evaluate_performance_status(
            &PerfAssertions {
                max_mean_latency_ms: Some(10.0),
                ..Default::default()
            },
            Some(12.0),
            None,
            &[],
        );

        assert_eq!(evaluation.status, AcceptanceStatus::Fail);
        assert!(evaluation
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("max_mean_latency_ms=10"));
    }
}
