use std::collections::BTreeMap;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use motlie_model::{ArtifactPolicy, BundleId, CapabilityKind, QuantizationBits, StartOptions};
use motlie_models::{download_bundle_artifacts, BundleDescriptor, Catalog, CuratedBundle};

use crate::accelerator;
use crate::metrics::{MemoryPeakKind, PerformanceMetrics, ResourceMetrics};
use crate::result::{
    overall_status_with_accelerator, reason_for_status, terminal_outcome, AcceleratorClass,
    AcceptanceSection, AcceptanceStatus, AssertionOutcome, CoverageSection, GateOutcome,
    IdentitySection, OutcomeReason, ProfileSection, ResultRecord, RuntimeSection, SelectionSection,
    RESULT_SCHEMA_VERSION,
};
use crate::runner::RunContext;
use crate::scenario::{CapabilityName, ModelCapabilityName};

pub struct PreparedBundle {
    pub bundle_id: BundleId,
    pub descriptor: BundleDescriptor,
    pub bundle: CuratedBundle,
    pub downloaded_artifacts: Vec<String>,
    pub quantization: Option<QuantizationBits>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionEvaluation {
    pub status: AcceptanceStatus,
    pub failure_reason: Option<String>,
}

pub fn prepare_bundle(
    context: &RunContext,
    default_capability: CapabilityKind,
    extra_capabilities: &[CapabilityKind],
) -> Result<PreparedBundle> {
    let catalog = Catalog::with_defaults();
    let bundle_id = BundleId::new(context.bundle_selection.bundle_id.clone());
    let descriptor = catalog
        .bundle(&bundle_id)
        .with_context(|| format!("unknown compiled bundle `{bundle_id}`"))?
        .clone();

    ensure_capability(&descriptor, default_capability)?;
    for capability in context
        .scenario
        .bundle_filter
        .required_capabilities
        .iter()
        .map(model_capability_kind)
        .chain(extra_capabilities.iter().copied())
    {
        ensure_capability(&descriptor, capability)?;
    }

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

    Ok(PreparedBundle {
        bundle_id,
        descriptor,
        bundle,
        downloaded_artifacts,
        quantization,
    })
}

pub fn start_options(context: &RunContext, prepared: &PreparedBundle) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: context.artifact_root.clone(),
        }),
        quantization: prepared.quantization,
        ..Default::default()
    }
}

pub fn model_capability_kind(capability: &ModelCapabilityName) -> CapabilityKind {
    match capability {
        ModelCapabilityName::Chat => CapabilityKind::Chat,
        ModelCapabilityName::Completion => CapabilityKind::Completion,
        ModelCapabilityName::Embeddings => CapabilityKind::Embeddings,
        ModelCapabilityName::Speech => CapabilityKind::Speech,
        ModelCapabilityName::ToolUse => CapabilityKind::ToolUse,
        ModelCapabilityName::Transcription => CapabilityKind::Transcription,
        ModelCapabilityName::Vision => CapabilityKind::Vision,
        ModelCapabilityName::VoiceClone => CapabilityKind::VoiceClone,
    }
}

pub fn bundle_filter_capability_kind(capability: CapabilityName) -> CapabilityKind {
    match capability {
        CapabilityName::Embeddings => CapabilityKind::Embeddings,
        CapabilityName::Chat => CapabilityKind::Chat,
        CapabilityName::ToolUse => CapabilityKind::ToolUse,
        CapabilityName::Asr => CapabilityKind::Transcription,
        CapabilityName::Tts => CapabilityKind::Speech,
        CapabilityName::Perf => CapabilityKind::Chat,
    }
}

pub fn behavior_status(assertions: &[AssertionOutcome]) -> AcceptanceStatus {
    if assertions.is_empty() {
        return AcceptanceStatus::NotMeasured;
    }
    if assertions
        .iter()
        .any(|assertion| assertion.status == AcceptanceStatus::Fail)
    {
        return AcceptanceStatus::Fail;
    }
    if assertions
        .iter()
        .any(|assertion| assertion.status == AcceptanceStatus::Blocked)
    {
        return AcceptanceStatus::Blocked;
    }
    if assertions
        .iter()
        .all(|assertion| assertion.status == AcceptanceStatus::Pass)
    {
        AcceptanceStatus::Pass
    } else {
        AcceptanceStatus::NotMeasured
    }
}

pub fn evaluate_performance_measured(
    measured: bool,
    missing_reason: impl Into<String>,
) -> SectionEvaluation {
    if measured {
        SectionEvaluation {
            status: AcceptanceStatus::Pass,
            failure_reason: None,
        }
    } else {
        SectionEvaluation {
            status: AcceptanceStatus::NotMeasured,
            failure_reason: Some(missing_reason.into()),
        }
    }
}

pub fn evaluate_resource_status(
    resources: &ResourceMetrics,
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
                        status: AcceptanceStatus::Blocked,
                        failure_reason: Some(
                            "resource gate max_process_swap_delta_bytes blocked: process swap delta unavailable"
                                .to_owned(),
                        ),
                    };
                }
            }
        }
    }

    if let Some(accelerator_memory_evaluation) =
        evaluate_accelerator_memory_gate(resources, context)
    {
        return accelerator_memory_evaluation;
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

fn evaluate_accelerator_memory_gate(
    resources: &ResourceMetrics,
    context: &RunContext,
) -> Option<SectionEvaluation> {
    let requested = context
        .accelerator
        .as_ref()
        .map(|accelerator| accelerator.requested_class)
        .unwrap_or_else(|| accelerator::requested_for_profile(&context.profile.name));

    match requested {
        AcceleratorClass::Cuda if resources.gpu_memory_peak_bytes.is_none() => {
            Some(SectionEvaluation {
                status: AcceptanceStatus::Blocked,
                failure_reason: Some(
                    "resource metric gpu_memory_peak_bytes blocked: CUDA peak VRAM sampler not instrumented"
                        .to_owned(),
                ),
            })
        }
        AcceleratorClass::Metal if !has_metal_unified_memory_peak(resources) => {
            Some(SectionEvaluation {
                status: AcceptanceStatus::Blocked,
                failure_reason: Some(
                    "resource metric apple unified-memory peak blocked: Metal memory sampler unavailable"
                        .to_owned(),
                ),
            })
        }
        _ => None,
    }
}

fn has_metal_unified_memory_peak(resources: &ResourceMetrics) -> bool {
    resources.memory_peaks.iter().any(|peak| {
        matches!(
            peak.kind,
            MemoryPeakKind::MetalCurrentAllocated
                | MemoryPeakKind::AppleFootprint
                | MemoryPeakKind::SystemUnifiedMemory
        ) && peak.bytes.is_some()
    })
}

pub fn failure_reason(
    overall_status: &AcceptanceStatus,
    assertions: &[AssertionOutcome],
    performance_evaluation: &SectionEvaluation,
    resource_evaluation: &SectionEvaluation,
) -> Option<String> {
    if matches!(overall_status, AcceptanceStatus::Pass) {
        return None;
    }

    if let Some(assertion) = assertions.iter().find(|assertion| {
        matches!(
            assertion.status,
            AcceptanceStatus::Fail | AcceptanceStatus::Blocked
        )
    }) {
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

pub fn build_record(
    context: &RunContext,
    prepared: &PreparedBundle,
    performance: PerformanceMetrics,
    resources: ResourceMetrics,
    assertions: Vec<AssertionOutcome>,
    performance_evaluation: SectionEvaluation,
    resource_evaluation: SectionEvaluation,
) -> ResultRecord {
    let behavior_status = behavior_status(&assertions);
    let performance_status = performance_evaluation.status.clone();
    let resource_status = resource_evaluation.status.clone();
    let platform = context.platform_collector.collect();
    let accelerator = context
        .accelerator
        .clone()
        .unwrap_or_else(|| accelerator::resolve_for_profile(&context.profile.name, &platform));
    let accelerator_status = accelerator::evaluate_use(&accelerator);
    let overall_status = overall_status_with_accelerator(
        &behavior_status,
        &performance_status,
        &resource_status,
        &accelerator_status,
    );
    let failure_reason = failure_reason(
        &overall_status,
        &assertions,
        &performance_evaluation,
        &resource_evaluation,
    )
    .or_else(|| {
        matches!(
            accelerator_status,
            AcceptanceStatus::Fail | AcceptanceStatus::Blocked
        )
        .then(|| {
            format!(
                "accelerator section not accepted: requested={} resolved={} reason={}",
                accelerator.requested_class.as_str(),
                accelerator.resolved_class.as_str(),
                accelerator
                    .fallback_reason
                    .as_ref()
                    .map(OutcomeReason::as_str)
                    .unwrap_or("none")
            )
        })
    });
    let checkpoint = prepared.descriptor.checkpoint();
    let checkpoint_format = checkpoint
        .as_ref()
        .map(|checkpoint| format!("{:?}", checkpoint.format));
    let artifact_quantization = artifact_quantization_label(
        checkpoint_format.as_deref(),
        context.runtime_flags.precision.as_deref(),
    );
    let host_id = platform
        .host_id
        .clone()
        .or_else(|| platform.hostname.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    let host_slug = platform
        .host_slug
        .clone()
        .unwrap_or_else(|| crate::platform::sanitize_slug(&host_id));
    let backend = format!("{:?}", prepared.descriptor.backend);
    let mut coverage = context.coverage.clone().unwrap_or_else(|| {
        default_coverage(
            context,
            prepared,
            &host_id,
            &host_slug,
            &backend,
            checkpoint_format.as_deref().unwrap_or("unknown"),
            &artifact_quantization,
            accelerator.requested_class,
            accelerator.resolved_class,
        )
    });
    coverage.resolved_accelerator = accelerator.resolved_class;
    coverage.terminal_outcome = terminal_outcome(&overall_status);
    if coverage.reason.is_none() {
        coverage.reason = reason_for_status(&overall_status);
    }

    ResultRecord {
        schema_version: RESULT_SCHEMA_VERSION,
        identity: IdentitySection {
            run_id: context.runtime_flags.run_id.clone().unwrap_or_else(run_id),
            git_sha: git_output(["rev-parse", "HEAD"]),
            git_branch: git_output(["branch", "--show-current"]),
            command_line: context.runtime_flags.command_line.clone(),
            host_id: Some(host_id),
            host_slug: Some(host_slug),
        },
        coverage,
        selection: SelectionSection {
            bundle_id: prepared.bundle_id.as_str().to_owned(),
            selector: context.bundle_selection.selector.clone(),
            backend: Some(backend),
            checkpoint_format,
            artifact_quantization: Some(artifact_quantization),
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
            artifact_files: prepared.downloaded_artifacts.clone(),
            scenario: context.scenario.id.clone(),
            capability: context.scenario.capability().as_str().to_owned(),
        },
        profile: ProfileSection {
            name: context.profile.name.clone(),
        },
        platform,
        accelerator,
        runtime: RuntimeSection {
            cargo_features: active_features(),
            build_profile: Some(build_profile().to_owned()),
            quantization: context
                .runtime_flags
                .precision
                .clone()
                .or_else(|| Some("backend_default".to_owned())),
            runtime_precision: context.runtime_flags.precision.clone(),
            artifact_root: context.artifact_root.display().to_string(),
            download_artifacts: context.runtime_flags.download_artifacts,
            context_length: None,
            gpu_layers: accelerator::runtime_gpu_layers(),
            child_build: context.child_build.clone(),
            budgets: Default::default(),
            env: runtime_env(),
        },
        performance,
        resources,
        acceptance: AcceptanceSection {
            behavior_status,
            performance_status,
            resource_status,
            accelerator_status,
            overall_status,
            failure_reason,
            assertions,
            gates: gate_outcomes(&performance_evaluation, &resource_evaluation),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn default_coverage(
    context: &RunContext,
    prepared: &PreparedBundle,
    host_id: &str,
    host_slug: &str,
    backend: &str,
    checkpoint_format: &str,
    artifact_quantization: &str,
    requested_accelerator: crate::result::AcceleratorClass,
    resolved_accelerator: crate::result::AcceleratorClass,
) -> CoverageSection {
    let mut grouping_keys = BTreeMap::new();
    grouping_keys.insert("bundle".to_owned(), prepared.bundle_id.as_str().to_owned());
    grouping_keys.insert(
        "capability".to_owned(),
        context.scenario.capability().as_str().to_owned(),
    );
    grouping_keys.insert(
        "depth".to_owned(),
        context.scenario.depth.as_str().to_owned(),
    );
    grouping_keys.insert("backend".to_owned(), backend.to_owned());
    grouping_keys.insert("checkpoint_format".to_owned(), checkpoint_format.to_owned());
    grouping_keys.insert("quantization".to_owned(), artifact_quantization.to_owned());
    grouping_keys.insert("profile".to_owned(), context.profile.name.clone());

    CoverageSection {
        snapshot_id: "ad_hoc".to_owned(),
        cell_id: format!("{}__{}", prepared.bundle_id, context.scenario.id),
        depth: context.scenario.depth,
        capability: context.scenario.capability().as_str().to_owned(),
        scenario_id: context.scenario.id.clone(),
        bundle_id: prepared.bundle_id.as_str().to_owned(),
        model_family: format!("{:?}", prepared.descriptor.family),
        checkpoint_format: checkpoint_format.to_owned(),
        quantization: artifact_quantization.to_owned(),
        backend: backend.to_owned(),
        profile: context.profile.name.clone(),
        host_id: host_id.to_owned(),
        host_slug: host_slug.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        requested_accelerator,
        resolved_accelerator,
        applicability: crate::result::ApplicabilityDecision::Applicable,
        terminal_outcome: crate::result::TerminalOutcome::Blocked,
        reason: None,
        grouping_keys,
    }
}

fn artifact_quantization_label(checkpoint_format: Option<&str>, precision: Option<&str>) -> String {
    if checkpoint_format
        .unwrap_or_default()
        .to_ascii_lowercase()
        .contains("gguf")
    {
        precision.unwrap_or("gguf_default").to_owned()
    } else {
        precision.unwrap_or("default").to_owned()
    }
}

fn gate_outcomes(
    performance_evaluation: &SectionEvaluation,
    resource_evaluation: &SectionEvaluation,
) -> Vec<GateOutcome> {
    vec![
        GateOutcome {
            section: "performance".to_owned(),
            name: "performance_measured".to_owned(),
            status: performance_evaluation.status.clone(),
            observed: None,
            threshold: None,
            source: Some("runner".to_owned()),
            reason: gate_reason(
                &performance_evaluation.status,
                OutcomeReason::PerformanceGateFailed,
            ),
            message: performance_evaluation.failure_reason.clone(),
        },
        GateOutcome {
            section: "resources".to_owned(),
            name: "resource_gates".to_owned(),
            status: resource_evaluation.status.clone(),
            observed: None,
            threshold: None,
            source: Some("metrics_sampler".to_owned()),
            reason: gate_reason(
                &resource_evaluation.status,
                OutcomeReason::ResourceGateFailed,
            ),
            message: resource_evaluation.failure_reason.clone(),
        },
    ]
}

fn gate_reason(status: &AcceptanceStatus, fail_reason: OutcomeReason) -> Option<OutcomeReason> {
    match status {
        AcceptanceStatus::Pass => None,
        AcceptanceStatus::Fail => Some(fail_reason),
        AcceptanceStatus::Blocked | AcceptanceStatus::NotMeasured => {
            Some(OutcomeReason::MetricUnavailableRequired)
        }
        AcceptanceStatus::Skipped | AcceptanceStatus::NotApplicable => {
            Some(OutcomeReason::MetricUnavailableOnPlatform)
        }
    }
}

pub fn assertion(
    name: impl Into<String>,
    passed: bool,
    message: impl Into<Option<String>>,
) -> AssertionOutcome {
    AssertionOutcome {
        name: name.into(),
        status: if passed {
            AcceptanceStatus::Pass
        } else {
            AcceptanceStatus::Fail
        },
        message: message.into(),
    }
}

pub fn elapsed_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

pub fn percentile(values: &[u64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((sorted.len() as f64 - 1.0) * percentile).round() as usize;
    sorted.get(rank).map(|value| *value as f64)
}

pub fn mean(values: &[u64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<u64>() as f64 / values.len() as f64)
}

pub fn format_bytes(bytes: u64) -> String {
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

fn ensure_capability(descriptor: &BundleDescriptor, capability: CapabilityKind) -> Result<()> {
    ensure!(
        descriptor.capabilities.supports(capability),
        "bundle `{}` does not advertise {:?}",
        descriptor.id,
        capability
    );
    Ok(())
}

fn parse_quantization(precision: Option<&str>) -> Result<Option<QuantizationBits>> {
    match precision {
        Some("q4") => Ok(Some(QuantizationBits::Four)),
        Some("q5") => Ok(Some(QuantizationBits::Five)),
        Some("q8") => Ok(Some(QuantizationBits::Eight)),
        Some("fp8") => Ok(Some(QuantizationBits::FloatEight)),
        Some("f16") | Some("f32") | None => Ok(None),
        Some(other) => bail!("unknown precision `{other}` - use q4, q5, q8, fp8, f16, or f32"),
    }
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

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn active_features() -> Vec<String> {
    let candidates = [
        (
            "model-google-gemma-300m",
            cfg!(feature = "model-google-gemma-300m"),
        ),
        (
            "model-qwen3-embedding-06b",
            cfg!(feature = "model-qwen3-embedding-06b"),
        ),
        ("model-qwen3-4b", cfg!(feature = "model-qwen3-4b")),
        ("model-gemma4-e2b", cfg!(feature = "model-gemma4-e2b")),
        ("model-gemma4-e4b", cfg!(feature = "model-gemma4-e4b")),
        (
            "model-gemma4-12b-gguf",
            cfg!(feature = "model-gemma4-12b-gguf"),
        ),
        (
            "model-gemma4-12b-qat-q4-0-gguf",
            cfg!(feature = "model-gemma4-12b-qat-q4-0-gguf"),
        ),
        ("model-qwen3-4b-gguf", cfg!(feature = "model-qwen3-4b-gguf")),
        (
            "model-gemma4-e2b-gguf",
            cfg!(feature = "model-gemma4-e2b-gguf"),
        ),
        (
            "model-gemma4-e4b-gguf",
            cfg!(feature = "model-gemma4-e4b-gguf"),
        ),
        (
            "model-qwen3-6-27b-gguf",
            cfg!(feature = "model-qwen3-6-27b-gguf"),
        ),
        (
            "model-piper-en-us-ljspeech-medium",
            cfg!(feature = "model-piper-en-us-ljspeech-medium"),
        ),
        ("model-qwen3-tts-cpp", cfg!(feature = "model-qwen3-tts-cpp")),
        (
            "model-moonshine-streaming",
            cfg!(feature = "model-moonshine-streaming"),
        ),
        (
            "model-sherpa-onnx-streaming",
            cfg!(feature = "model-sherpa-onnx-streaming"),
        ),
        (
            "model-whisper-base-en",
            cfg!(feature = "model-whisper-base-en"),
        ),
        ("llama-cpp-cuda", cfg!(feature = "llama-cpp-cuda")),
        ("piper-cuda", cfg!(feature = "piper-cuda")),
        ("qwen3-tts-cpp-cuda", cfg!(feature = "qwen3-tts-cpp-cuda")),
        ("sherpa-onnx-cuda", cfg!(feature = "sherpa-onnx-cuda")),
        ("whisper-cpp-cuda", cfg!(feature = "whisper-cpp-cuda")),
        ("cuda", cfg!(feature = "cuda")),
        ("cudnn", cfg!(feature = "cudnn")),
        ("flash-attn", cfg!(feature = "flash-attn")),
        ("metal", cfg!(feature = "metal")),
        ("accelerate", cfg!(feature = "accelerate")),
    ];

    candidates
        .into_iter()
        .filter(|(_, active)| *active)
        .map(|(name, _)| name.to_owned())
        .collect()
}

fn runtime_env() -> BTreeMap<String, Option<String>> {
    [
        "MOTLIE_MODEL_FORCE_CPU",
        "MOTLIE_MODEL_GPU_LAYERS",
        "MOTLIE_PAGED_ATTN_CONTEXT",
        "CUDA_VISIBLE_DEVICES",
        "MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), std::env::var(key).ok()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ResourceMetrics;
    use crate::runner::{BundleSelection, ProfileSelection, RuntimeFlags};
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
    fn overall_failure_reason_prefers_resource_failure_over_passing_behavior_message() {
        let assertions = vec![AssertionOutcome {
            name: "similar_gt_dissimilar".to_owned(),
            status: AcceptanceStatus::Pass,
            message: Some("similar=0.9 dissimilar=0.1 gap=0.8".to_owned()),
        }];
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
            &assertions,
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
id = "chat_smoke"
capability = "chat"
summary = "Minimal chat smoke scenario."

[bundle_filter]
capability = "chat"

[input]
prompt = "Say hello."

[assertions]
min_response_chars = 1

[metrics]
capture_startup_ms = true
capture_request_latency = true

[profiles.local-cpu-x86_64.gates]
max_process_swap_delta_bytes = 0
"#,
        )
        .unwrap();

        RunContext {
            scenario,
            bundle_selection: BundleSelection {
                bundle_id: "qwen3_4b".to_owned(),
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
}
