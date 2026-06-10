use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::{BundleHandle, CapabilityKind, EmbeddingModel, EmbeddingRequest};

use crate::metrics::{
    CapabilityPerformanceMetrics, EmbeddingPerformanceMetrics, PerformanceMetrics,
};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    build_record, elapsed_ms, evaluate_resource_status, prepare_bundle, start_options,
    SectionEvaluation,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, EmbeddingsAssertions, SimilarityOrder};

pub struct EmbeddingSimilarityRunner;

#[async_trait]
impl ScenarioRunner for EmbeddingSimilarityRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::Embeddings,
            "scenario `{}` is not an embeddings scenario",
            context.scenario.id
        );
        ensure!(
            context.scenario.bundle_filter.capability == CapabilityName::Embeddings,
            "scenario `{}` bundle filter must use capability=embeddings",
            context.scenario.id
        );
        let embeddings_scenario = context
            .scenario
            .embeddings()
            .context("embeddings scenario should carry embeddings input/assertions")?
            .clone();
        let input = embeddings_scenario.input;
        let assertions = embeddings_scenario.assertions;

        let prepared = prepare_bundle(&context, CapabilityKind::Embeddings, &[])?;

        context.metrics_sampler.sample();
        let startup_started_at = std::time::Instant::now();
        let handle = prepared
            .bundle
            .start(start_options(&context, &prepared))
            .await
            .with_context(|| format!("failed to start bundle `{}`", prepared.bundle_id))?;
        let startup_ms = elapsed_ms(startup_started_at.elapsed());
        context.metrics_sampler.sample();

        let embeddings = handle
            .embeddings()
            .context("selected bundle should expose embeddings")?;

        let custom_started_at = std::time::Instant::now();
        let custom_response = embeddings
            .embed(EmbeddingRequest {
                inputs: vec![input.custom_text.clone()],
            })
            .await
            .context("custom embedding request failed")?;
        let custom_latency_ms = elapsed_ms(custom_started_at.elapsed());
        context.metrics_sampler.sample();

        let custom_vector = custom_response
            .vectors
            .first()
            .context("custom embedding response did not include a vector")?;
        let embedding_dimensions = custom_vector.len();

        let similar_started_at = std::time::Instant::now();
        let similar_cosine = embed_pair(embeddings, &input.similar_a, &input.similar_b)
            .await
            .context("similar embedding pair failed")?;
        let similar_latency_ms = elapsed_ms(similar_started_at.elapsed());
        context.metrics_sampler.sample();

        let dissimilar_started_at = std::time::Instant::now();
        let dissimilar_cosine = embed_pair(embeddings, &input.dissimilar_a, &input.dissimilar_b)
            .await
            .context("dissimilar embedding pair failed")?;
        let dissimilar_latency_ms = elapsed_ms(dissimilar_started_at.elapsed());
        context.metrics_sampler.sample();

        let model_metrics = handle.metric_snapshot();
        handle
            .shutdown()
            .await
            .with_context(|| format!("failed to shut down bundle `{}`", prepared.bundle_id))?;

        let request_latencies_ms =
            vec![custom_latency_ms, similar_latency_ms, dissimilar_latency_ms];
        let request_latency_sum_ms = request_latencies_ms.iter().sum::<u64>();
        let vectors_per_second = if request_latency_sum_ms == 0 {
            None
        } else {
            Some(5.0 / (request_latency_sum_ms as f64 / 1000.0))
        };
        let similarity_gap = similar_cosine - dissimilar_cosine;

        let model_embedding_metrics = model_metrics.and_then(|snapshot| snapshot.embeddings);
        let embedding_metrics = EmbeddingPerformanceMetrics {
            custom_embedding_latency_ms: Some(custom_latency_ms),
            similar_pair_latency_ms: Some(similar_latency_ms),
            dissimilar_pair_latency_ms: Some(dissimilar_latency_ms),
            embedding_dimensions: Some(embedding_dimensions),
            vectors_per_second,
            similar_cosine: Some(similar_cosine),
            dissimilar_cosine: Some(dissimilar_cosine),
            similarity_gap: Some(similarity_gap),
            model_embedding_request_count: model_embedding_metrics
                .as_ref()
                .and_then(|metrics| metrics.request_count),
            model_embedding_input_count: model_embedding_metrics
                .as_ref()
                .and_then(|metrics| metrics.input_count),
        };
        let performance = PerformanceMetrics {
            startup_ms: Some(startup_ms),
            request_latencies_ms,
            capability_metrics: CapabilityPerformanceMetrics::Embeddings(embedding_metrics),
            ..Default::default()
        };
        let resources = context.metrics_sampler.finish();

        let assertion = evaluate_assertion(
            embedding_dimensions,
            similar_cosine,
            dissimilar_cosine,
            &assertions,
        );
        let performance_evaluation = evaluate_performance_status(&performance);
        let resource_evaluation = evaluate_resource_status(&resources, &context);
        let record = build_record(
            &context,
            &prepared,
            performance,
            resources,
            vec![assertion],
            performance_evaluation,
            resource_evaluation,
        );

        context.output_sink.emit(&record)?;
        Ok(record)
    }
}

async fn embed_pair<E: EmbeddingModel + ?Sized>(
    embeddings: &E,
    first: &str,
    second: &str,
) -> Result<f64> {
    let response = embeddings
        .embed(EmbeddingRequest {
            inputs: vec![first.to_owned(), second.to_owned()],
        })
        .await?;
    let mut vectors = response.vectors.into_iter();
    let first_vector = vectors
        .next()
        .context("expected first vector for embedding pair")?;
    let second_vector = vectors
        .next()
        .context("expected second vector for embedding pair")?;
    cosine_similarity(&first_vector, &second_vector)
}

fn evaluate_assertion(
    embedding_dimensions: usize,
    similar_cosine: f64,
    dissimilar_cosine: f64,
    assertions: &EmbeddingsAssertions,
) -> AssertionOutcome {
    let dimensions_ok = embedding_dimensions >= assertions.min_embedding_dimensions;
    let similarity_gap = similar_cosine - dissimilar_cosine;
    let similarity_ok = match assertions.similarity_order {
        SimilarityOrder::SimilarGtDissimilar => {
            similar_cosine > dissimilar_cosine && similarity_gap >= assertions.min_similarity_gap
        }
    };

    if dimensions_ok && similarity_ok {
        AssertionOutcome {
            name: "similar_gt_dissimilar".to_owned(),
            status: AcceptanceStatus::Pass,
            message: Some(format!(
                "similar={similar_cosine:.6} dissimilar={dissimilar_cosine:.6} gap={similarity_gap:.6}"
            )),
        }
    } else {
        AssertionOutcome {
            name: "similar_gt_dissimilar".to_owned(),
            status: AcceptanceStatus::Fail,
            message: Some(format!(
                "dimensions_ok={dimensions_ok} similar={similar_cosine:.6} dissimilar={dissimilar_cosine:.6} gap={similarity_gap:.6}"
            )),
        }
    }
}

fn evaluate_performance_status(performance: &PerformanceMetrics) -> SectionEvaluation {
    let CapabilityPerformanceMetrics::Embeddings(metrics) = &performance.capability_metrics else {
        return SectionEvaluation {
            status: AcceptanceStatus::NotMeasured,
            failure_reason: Some("performance capability metrics were not measured".to_owned()),
        };
    };

    if performance.startup_ms.is_some()
        && !performance.request_latencies_ms.is_empty()
        && metrics.embedding_dimensions.is_some()
    {
        SectionEvaluation {
            status: AcceptanceStatus::Pass,
            failure_reason: None,
        }
    } else {
        SectionEvaluation {
            status: AcceptanceStatus::NotMeasured,
            failure_reason: Some(
                "performance metrics missing startup, request latency, or embedding dimensions"
                    .to_owned(),
            ),
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f64> {
    ensure!(
        left.len() == right.len(),
        "cosine similarity requires same-dimension vectors"
    );
    ensure!(
        !left.is_empty(),
        "cosine similarity requires non-empty vectors"
    );

    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| f64::from(*lhs) * f64::from(*rhs))
        .sum::<f64>();
    let left_norm = left
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();

    ensure!(
        left_norm != 0.0 && right_norm != 0.0,
        "cosine similarity requires non-zero vector norms"
    );
    Ok(dot / (left_norm * right_norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MemoryPeak, MemoryPeakKind, MetricsSampler, ResourceMetrics};
    use crate::platform::PlatformCollector;
    use crate::report::OutputSink;
    use crate::runner::{BundleSelection, ProfileSelection, RuntimeFlags};

    #[test]
    fn apple_profile_without_swap_gate_passes_when_rss_is_measured() {
        let context = test_context("apple-metal");
        let resources = ResourceMetrics {
            rss_peak_bytes: Some(1),
            process_swap_delta_peak_bytes: None,
            memory_peaks: vec![MemoryPeak {
                kind: MemoryPeakKind::AppleFootprint,
                bytes: Some(1),
                device_id: None,
                source: "sysinfo_process_rss_proxy".to_owned(),
                unavailable_reason: None,
            }],
            ..Default::default()
        };

        let evaluation = evaluate_resource_status(&resources, &context);

        assert_eq!(evaluation.status, AcceptanceStatus::Pass);
        assert_eq!(evaluation.failure_reason, None);
    }

    fn test_context(profile_name: &str) -> RunContext {
        let scenario = toml::from_str(
            r#"
schema_version = 1
id = "embeddings_similarity"
capability = "embeddings"
summary = "Embed one input and compare a similar pair against a dissimilar pair."

[bundle_filter]
capability = "embeddings"
backend = ["MistralRs"]

[input]
custom_text = "motlie curated model bundle"
similar_a = "A small orange cat is sleeping on the couch."
similar_b = "An orange kitten is napping on a sofa."
dissimilar_a = "A small orange cat is sleeping on the couch."
dissimilar_b = "Quarterly revenue increased by twelve percent year over year."

[assertions]
min_embedding_dimensions = 1
similarity_order = "similar_gt_dissimilar"
min_similarity_gap = 0.05

[metrics]
capture_startup_ms = true
capture_request_latency = true

[profiles.local-cpu-x86_64.gates]
max_process_swap_delta_bytes = 0

[profiles.apple-metal]
"#,
        )
        .unwrap();

        RunContext {
            scenario,
            bundle_selection: BundleSelection {
                bundle_id: "embeddinggemma_300m".to_owned(),
                selector: None,
            },
            profile: ProfileSelection {
                name: profile_name.to_owned(),
            },
            artifact_root: std::path::PathBuf::from("/tmp/motlie-test-artifacts"),
            runtime_flags: RuntimeFlags {
                command_line: Vec::new(),
                download_artifacts: false,
                precision: None,
                quiet_backend_logs: false,
                run_id: None,
            },
            coverage: None,
            accelerator: None,
            child_build: None,
            platform_collector: PlatformCollector::new(profile_name),
            metrics_sampler: MetricsSampler::new(),
            output_sink: OutputSink::Stdout,
        }
    }

    #[test]
    fn resource_failure_reason_names_failing_gate() {
        let context = test_context("local-cpu-x86_64");
        let resources = ResourceMetrics {
            rss_peak_bytes: Some(1),
            process_swap_delta_peak_bytes: Some(1_677_721_600),
            ..Default::default()
        };

        let evaluation = evaluate_resource_status(&resources, &context);

        assert_eq!(evaluation.status, AcceptanceStatus::Fail);
        assert!(evaluation
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("max_process_swap_delta_bytes=0 exceeded"));
        assert!(evaluation
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("process_swap_delta_peak=1.56GiB"));
    }

    #[test]
    fn overall_failure_reason_prefers_resource_failure_over_passing_behavior_message() {
        let assertion = AssertionOutcome {
            name: "similar_gt_dissimilar".to_owned(),
            status: AcceptanceStatus::Pass,
            message: Some("similar=0.9 dissimilar=0.1 gap=0.8".to_owned()),
        };
        let performance = SectionEvaluation {
            status: AcceptanceStatus::Pass,
            failure_reason: None,
        };
        let resources = SectionEvaluation {
            status: AcceptanceStatus::Fail,
            failure_reason: Some("resource gate exceeded".to_owned()),
        };

        let reason = crate::runner::support::failure_reason(
            &AcceptanceStatus::Fail,
            &[assertion],
            &performance,
            &resources,
        )
        .unwrap();

        assert_eq!(
            reason,
            "resources section not accepted: resource gate exceeded"
        );
    }
}
