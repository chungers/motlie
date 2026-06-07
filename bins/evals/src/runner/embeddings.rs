use std::collections::BTreeMap;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, CapabilityKind, EmbeddingModel, EmbeddingRequest,
    QuantizationBits, StartOptions,
};
use motlie_models::{download_bundle_artifacts, Catalog};

use crate::metrics::{
    CapabilityPerformanceMetrics, EmbeddingPerformanceMetrics, PerformanceMetrics,
};
use crate::result::{
    overall_status, AcceptanceSection, AcceptanceStatus, AssertionOutcome, IdentitySection,
    ProfileSection, ResultRecord, RuntimeSection, SelectionSection,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, EmbeddingsAssertions, SimilarityOrder};

pub struct EmbeddingSimilarityRunner;

#[async_trait]
impl ScenarioRunner for EmbeddingSimilarityRunner {
    async fn run(&self, mut context: RunContext) -> Result<ResultRecord> {
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

        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new(context.bundle_selection.bundle_id.clone());
        let descriptor = catalog
            .bundle(&bundle_id)
            .with_context(|| format!("unknown compiled bundle `{bundle_id}`"))?
            .clone();
        ensure!(
            descriptor.capabilities.supports(CapabilityKind::Embeddings),
            "bundle `{bundle_id}` does not advertise embeddings"
        );

        if !context.scenario.bundle_filter.backend.is_empty() {
            let backend = format!("{:?}", descriptor.backend);
            ensure!(
                context
                    .scenario
                    .bundle_filter
                    .backend
                    .iter()
                    .any(|candidate| candidate == &backend),
                "bundle `{bundle_id}` backend `{backend}` is not accepted by scenario `{}`",
                context.scenario.id
            );
        }

        let downloaded_artifacts = if context.runtime_flags.download_artifacts {
            download_bundle_artifacts(&catalog, &bundle_id, &context.artifact_root)
                .with_context(|| format!("failed to download artifacts for `{bundle_id}`"))?
                .downloaded
                .into_iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let bundle = catalog
            .instantiate(&bundle_id)
            .with_context(|| format!("bundle `{bundle_id}` is not available in this build"))?;
        let quantization = parse_quantization(context.runtime_flags.precision.as_deref())?;
        let start_options = StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: context.artifact_root.clone(),
            }),
            quantization,
            ..Default::default()
        };

        context.metrics_sampler.sample();
        let startup_started_at = std::time::Instant::now();
        let handle = bundle
            .start(start_options)
            .await
            .with_context(|| format!("failed to start bundle `{bundle_id}`"))?;
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
            .with_context(|| format!("failed to shut down bundle `{bundle_id}`"))?;

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
        };
        let resources = context.metrics_sampler.finish();

        let assertion = evaluate_assertion(
            embedding_dimensions,
            similar_cosine,
            dissimilar_cosine,
            &assertions,
        );
        let behavior_status = assertion.status.clone();
        let performance_evaluation = evaluate_performance_status(&performance);
        let resource_evaluation = evaluate_resource_status(&resources, &context);
        let performance_status = performance_evaluation.status.clone();
        let resource_status = resource_evaluation.status.clone();
        let overall_status =
            overall_status(&behavior_status, &performance_status, &resource_status);
        let failure_reason = failure_reason(
            &overall_status,
            &behavior_status,
            &assertion,
            &performance_evaluation,
            &resource_evaluation,
        );

        let checkpoint = descriptor.checkpoint();
        let record = ResultRecord {
            schema_version: 1,
            identity: IdentitySection {
                run_id: run_id(),
                git_sha: git_output(["rev-parse", "HEAD"]),
                git_branch: git_output(["branch", "--show-current"]),
                command_line: context.runtime_flags.command_line.clone(),
            },
            selection: SelectionSection {
                bundle_id: bundle_id.as_str().to_owned(),
                selector: context.bundle_selection.selector.clone(),
                backend: Some(format!("{:?}", descriptor.backend)),
                checkpoint_format: checkpoint
                    .as_ref()
                    .map(|checkpoint| format!("{:?}", checkpoint.format)),
                artifact_snapshot: None,
                artifact_patterns: checkpoint
                    .as_ref()
                    .map(|checkpoint| {
                        checkpoint
                            .include
                            .iter()
                            .map(|rule| format!("{rule:?}"))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                artifact_files: downloaded_artifacts,
                scenario: context.scenario.id.clone(),
                capability: context.scenario.capability().as_str().to_owned(),
            },
            profile: ProfileSection {
                name: context.profile.name.clone(),
            },
            platform: context.platform_collector.collect(),
            runtime: RuntimeSection {
                cargo_features: active_features(),
                build_profile: option_env!("PROFILE").map(str::to_owned),
                quantization: context
                    .runtime_flags
                    .precision
                    .clone()
                    .or_else(|| Some("f32".to_owned())),
                artifact_root: context.artifact_root.display().to_string(),
                download_artifacts: context.runtime_flags.download_artifacts,
                context_length: None,
                gpu_layers: None,
                env: runtime_env(),
            },
            performance,
            resources,
            acceptance: AcceptanceSection {
                behavior_status,
                performance_status,
                resource_status,
                overall_status,
                failure_reason,
                assertions: vec![assertion],
            },
        };

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

#[derive(Clone, Debug, Eq, PartialEq)]
struct SectionEvaluation {
    status: AcceptanceStatus,
    failure_reason: Option<String>,
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

fn evaluate_resource_status(
    resources: &crate::metrics::ResourceMetrics,
    context: &RunContext,
) -> SectionEvaluation {
    if let Some(gates) = context.scenario.gates_for_profile(&context.profile.name) {
        if let Some(max_process_swap_delta_bytes) = gates.max_process_swap_delta_bytes {
            match resources.process_swap_delta_peak_bytes {
                Some(value) if value <= max_process_swap_delta_bytes => {}
                Some(value) => {
                    return SectionEvaluation {
                        status: AcceptanceStatus::Fail,
                        failure_reason: Some(format!(
                            "resource gate max_process_swap_delta_bytes={} exceeded: process_swap_delta_peak={} ({} bytes)",
                            max_process_swap_delta_bytes,
                            format_bytes(value),
                            value
                        )),
                    };
                }
                None => {
                    return SectionEvaluation {
                        status: AcceptanceStatus::NotMeasured,
                        failure_reason: Some(
                            "resource gate max_process_swap_delta_bytes could not be evaluated: process swap delta unavailable"
                                .to_owned(),
                        ),
                    };
                }
            }
        }
    }

    if resources.rss_peak_bytes.is_some() {
        SectionEvaluation {
            status: AcceptanceStatus::Pass,
            failure_reason: None,
        }
    } else {
        SectionEvaluation {
            status: AcceptanceStatus::NotMeasured,
            failure_reason: Some("resource metrics missing rss_peak_bytes".to_owned()),
        }
    }
}

fn failure_reason(
    overall_status: &AcceptanceStatus,
    behavior_status: &AcceptanceStatus,
    assertion: &AssertionOutcome,
    performance_evaluation: &SectionEvaluation,
    resource_evaluation: &SectionEvaluation,
) -> Option<String> {
    if matches!(overall_status, AcceptanceStatus::Pass) {
        return None;
    }

    if matches!(
        behavior_status,
        AcceptanceStatus::Fail | AcceptanceStatus::Blocked
    ) {
        return Some(format!(
            "behavior assertion `{}` failed: {}",
            assertion.name,
            assertion
                .message
                .as_deref()
                .unwrap_or("no assertion detail")
        ));
    }

    if matches!(
        performance_evaluation.status,
        AcceptanceStatus::Fail | AcceptanceStatus::Blocked | AcceptanceStatus::NotMeasured
    ) {
        if let Some(reason) = &performance_evaluation.failure_reason {
            return Some(format!("performance section not accepted: {reason}"));
        }
    }

    if matches!(
        resource_evaluation.status,
        AcceptanceStatus::Fail | AcceptanceStatus::Blocked | AcceptanceStatus::NotMeasured
    ) {
        if let Some(reason) = &resource_evaluation.failure_reason {
            return Some(format!("resources section not accepted: {reason}"));
        }
    }

    None
}

fn parse_quantization(precision: Option<&str>) -> Result<Option<QuantizationBits>> {
    match precision {
        Some("q4") => Ok(Some(QuantizationBits::Four)),
        Some("q8") => Ok(Some(QuantizationBits::Eight)),
        Some("f32") | None => Ok(None),
        Some(other) => bail!("unknown precision `{other}` - use q4, q8, or f32"),
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

fn elapsed_ms(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("run-{millis}")
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn active_features() -> Vec<String> {
    let mut features = Vec::new();
    if cfg!(feature = "model-google-gemma-300m") {
        features.push("model-google-gemma-300m".to_owned());
    }
    if cfg!(feature = "model-qwen3-embedding-06b") {
        features.push("model-qwen3-embedding-06b".to_owned());
    }
    features
}

fn runtime_env() -> BTreeMap<String, Option<String>> {
    [
        "MOTLIE_MODEL_FORCE_CPU",
        "MOTLIE_MODEL_GPU_LAYERS",
        "MOTLIE_PAGED_ATTN_CONTEXT",
        "CUDA_VISIBLE_DEVICES",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), std::env::var(key).ok()))
    .collect()
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * KIB;
    const GIB: f64 = 1024.0 * MIB;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2}GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.2}MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.2}KiB", bytes / KIB)
    } else {
        format!("{bytes:.0}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ResourceMetrics;
    use crate::runner::{BundleSelection, ProfileSelection, RunContext, RuntimeFlags};
    use crate::{metrics::MetricsSampler, platform::PlatformCollector, report::OutputSink};

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
    fn apple_profile_without_swap_gate_passes_when_rss_is_measured() {
        let context = test_context("apple-metal");
        let resources = ResourceMetrics {
            rss_peak_bytes: Some(1),
            process_swap_delta_peak_bytes: None,
            ..Default::default()
        };

        let evaluation = evaluate_resource_status(&resources, &context);

        assert_eq!(evaluation.status, AcceptanceStatus::Pass);
        assert_eq!(evaluation.failure_reason, None);
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

        let reason = failure_reason(
            &AcceptanceStatus::Fail,
            &assertion.status,
            &assertion,
            &performance,
            &resources,
        )
        .unwrap();

        assert_eq!(
            reason,
            "resources section not accepted: resource gate exceeded"
        );
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
            },
            platform_collector: PlatformCollector::new(profile_name),
            metrics_sampler: MetricsSampler::new(),
            output_sink: OutputSink::Stdout,
        }
    }
}
