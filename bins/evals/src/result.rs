use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::metrics::{PerformanceMetrics, ResourceMetrics};
use crate::platform::PlatformSnapshot;

pub const RESULT_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultRecord {
    pub schema_version: u32,
    pub identity: IdentitySection,
    #[serde(default)]
    pub coverage: CoverageSection,
    pub selection: SelectionSection,
    pub profile: ProfileSection,
    pub platform: PlatformSnapshot,
    #[serde(default)]
    pub accelerator: AcceleratorSection,
    pub runtime: RuntimeSection,
    pub performance: PerformanceMetrics,
    pub resources: ResourceMetrics,
    pub acceptance: AcceptanceSection,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IdentitySection {
    pub run_id: String,
    pub git_sha: Option<String>,
    pub git_branch: Option<String>,
    pub command_line: Vec<String>,
    #[serde(default)]
    pub host_id: Option<String>,
    #[serde(default)]
    pub host_slug: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageSection {
    pub snapshot_id: String,
    pub cell_id: String,
    pub depth: EvalDepth,
    pub capability: String,
    pub scenario_id: String,
    pub bundle_id: String,
    pub model_family: String,
    pub checkpoint_format: String,
    pub quantization: String,
    pub backend: String,
    pub profile: String,
    pub host_id: String,
    pub host_slug: String,
    pub arch: String,
    pub requested_accelerator: AcceleratorClass,
    pub resolved_accelerator: AcceleratorClass,
    pub applicability: ApplicabilityDecision,
    pub terminal_outcome: TerminalOutcome,
    pub reason: Option<OutcomeReason>,
    #[serde(default)]
    pub grouping_keys: BTreeMap<String, String>,
}

impl Default for CoverageSection {
    fn default() -> Self {
        Self {
            snapshot_id: "ad_hoc".to_owned(),
            cell_id: "ad_hoc".to_owned(),
            depth: EvalDepth::Smoke,
            capability: "unknown".to_owned(),
            scenario_id: "unknown".to_owned(),
            bundle_id: "unknown".to_owned(),
            model_family: "unknown".to_owned(),
            checkpoint_format: "unknown".to_owned(),
            quantization: "unknown".to_owned(),
            backend: "unknown".to_owned(),
            profile: "unknown".to_owned(),
            host_id: "unknown".to_owned(),
            host_slug: "unknown".to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            requested_accelerator: AcceleratorClass::Any,
            resolved_accelerator: AcceleratorClass::Unavailable,
            applicability: ApplicabilityDecision::Applicable,
            terminal_outcome: TerminalOutcome::Blocked,
            reason: Some(OutcomeReason::MetricUnavailableRequired),
            grouping_keys: BTreeMap::new(),
        }
    }
}

impl CoverageSection {
    pub fn with_outcome(mut self, outcome: TerminalOutcome, reason: Option<OutcomeReason>) -> Self {
        self.terminal_outcome = outcome;
        self.reason = reason;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SelectionSection {
    pub bundle_id: String,
    pub selector: Option<String>,
    pub backend: Option<String>,
    pub checkpoint_format: Option<String>,
    #[serde(default)]
    pub artifact_quantization: Option<String>,
    pub artifact_snapshot: Option<String>,
    pub artifact_patterns: Vec<String>,
    pub artifact_files: Vec<String>,
    pub scenario: String,
    pub capability: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProfileSection {
    pub name: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorSection {
    pub requested_class: AcceleratorClass,
    pub resolved_class: AcceleratorClass,
    pub selected_devices: Vec<AcceleratorDevice>,
    pub backend_mode: Option<String>,
    pub offload: Option<String>,
    pub driver_versions: BTreeMap<String, String>,
    pub fallback_reason: Option<OutcomeReason>,
    pub use_proof_source: Option<String>,
}

impl Default for AcceleratorSection {
    fn default() -> Self {
        Self {
            requested_class: AcceleratorClass::Any,
            resolved_class: AcceleratorClass::Unavailable,
            selected_devices: Vec::new(),
            backend_mode: None,
            offload: None,
            driver_versions: BTreeMap::new(),
            fallback_reason: Some(OutcomeReason::AcceleratorUnavailable),
            use_proof_source: Some("unavailable".to_owned()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorDevice {
    pub id: Option<String>,
    pub name: Option<String>,
    pub backend: AcceleratorClass,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSection {
    pub cargo_features: Vec<String>,
    pub build_profile: Option<String>,
    pub quantization: Option<String>,
    #[serde(default)]
    pub runtime_precision: Option<String>,
    pub artifact_root: String,
    pub download_artifacts: bool,
    pub context_length: Option<u64>,
    pub gpu_layers: Option<u32>,
    #[serde(default)]
    pub child_build: Option<ChildBuildSection>,
    #[serde(default)]
    pub budgets: BTreeMap<String, String>,
    pub env: BTreeMap<String, Option<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChildBuildSection {
    pub command: Vec<String>,
    pub status: Option<i32>,
    pub duration_ms: Option<u64>,
    pub log_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceSection {
    pub behavior_status: AcceptanceStatus,
    pub performance_status: AcceptanceStatus,
    pub resource_status: AcceptanceStatus,
    #[serde(default = "default_acceptance_pass")]
    pub accelerator_status: AcceptanceStatus,
    pub overall_status: AcceptanceStatus,
    pub failure_reason: Option<String>,
    pub assertions: Vec<AssertionOutcome>,
    #[serde(default)]
    pub gates: Vec<GateOutcome>,
}

fn default_acceptance_pass() -> AcceptanceStatus {
    AcceptanceStatus::Pass
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssertionOutcome {
    pub name: String,
    pub status: AcceptanceStatus,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GateOutcome {
    pub section: String,
    pub name: String,
    pub status: AcceptanceStatus,
    pub observed: Option<String>,
    pub threshold: Option<String>,
    pub source: Option<String>,
    pub reason: Option<OutcomeReason>,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceStatus {
    Pass,
    Fail,
    Blocked,
    Skipped,
    NotMeasured,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalDepth {
    Smoke,
    Enriched,
}

impl EvalDepth {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Enriched => "enriched",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorClass {
    Cpu,
    Cuda,
    Metal,
    Any,
    Unavailable,
}

impl AcceleratorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Any => "any",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicabilityDecision {
    Applicable,
    NotApplicable,
    BlockedPreRun,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    Passed,
    Failed,
    Blocked,
    Skipped,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeReason {
    ProfileNotApplicable,
    FeatureGroupNotSupported,
    FeatureBuildFailed,
    ArtifactMissing,
    ArtifactUnauthorized,
    HfTokenMissing,
    NativeToolchainMissing,
    GgufToolchainFailed,
    GgufMetalUnverified,
    SubmoduleMissing,
    CpuProfileNotPractical,
    RuntimeBudgetExceeded,
    ResourceBudgetExceeded,
    AcceleratorUnavailable,
    AcceleratorMismatch,
    BackendOffloadUnverified,
    MetricUnavailableRequired,
    MetricUnsupportedByBackend,
    MetricNotInstrumented,
    MetricUnavailableOnPlatform,
    MetricCollectionFailed,
    BehaviorAssertionFailed,
    PerformanceGateFailed,
    ResourceGateFailed,
    ChildRunFailed,
    DryRun,
    #[serde(other)]
    Unknown,
}

impl OutcomeReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProfileNotApplicable => "profile_not_applicable",
            Self::FeatureGroupNotSupported => "feature_group_not_supported",
            Self::FeatureBuildFailed => "feature_build_failed",
            Self::ArtifactMissing => "artifact_missing",
            Self::ArtifactUnauthorized => "artifact_unauthorized",
            Self::HfTokenMissing => "hf_token_missing",
            Self::NativeToolchainMissing => "native_toolchain_missing",
            Self::GgufToolchainFailed => "gguf_toolchain_failed",
            Self::GgufMetalUnverified => "gguf_metal_unverified",
            Self::SubmoduleMissing => "submodule_missing",
            Self::CpuProfileNotPractical => "cpu_profile_not_practical",
            Self::RuntimeBudgetExceeded => "runtime_budget_exceeded",
            Self::ResourceBudgetExceeded => "resource_budget_exceeded",
            Self::AcceleratorUnavailable => "accelerator_unavailable",
            Self::AcceleratorMismatch => "accelerator_mismatch",
            Self::BackendOffloadUnverified => "backend_offload_unverified",
            Self::MetricUnavailableRequired => "metric_unavailable_required",
            Self::MetricUnsupportedByBackend => "metric_unsupported_by_backend",
            Self::MetricNotInstrumented => "metric_not_instrumented",
            Self::MetricUnavailableOnPlatform => "metric_unavailable_on_platform",
            Self::MetricCollectionFailed => "metric_collection_failed",
            Self::BehaviorAssertionFailed => "behavior_assertion_failed",
            Self::PerformanceGateFailed => "performance_gate_failed",
            Self::ResourceGateFailed => "resource_gate_failed",
            Self::ChildRunFailed => "child_run_failed",
            Self::DryRun => "dry_run",
            Self::Unknown => "unknown",
        }
    }
}

pub fn terminal_outcome(status: &AcceptanceStatus) -> TerminalOutcome {
    match status {
        AcceptanceStatus::Pass => TerminalOutcome::Passed,
        AcceptanceStatus::Fail => TerminalOutcome::Failed,
        AcceptanceStatus::Blocked | AcceptanceStatus::NotMeasured => TerminalOutcome::Blocked,
        AcceptanceStatus::Skipped | AcceptanceStatus::NotApplicable => TerminalOutcome::Skipped,
    }
}

pub fn reason_for_status(status: &AcceptanceStatus) -> Option<OutcomeReason> {
    match status {
        AcceptanceStatus::Pass => None,
        AcceptanceStatus::Fail => Some(OutcomeReason::BehaviorAssertionFailed),
        AcceptanceStatus::Blocked | AcceptanceStatus::NotMeasured => {
            Some(OutcomeReason::MetricUnavailableRequired)
        }
        AcceptanceStatus::Skipped | AcceptanceStatus::NotApplicable => {
            Some(OutcomeReason::ProfileNotApplicable)
        }
    }
}

pub fn overall_status(
    behavior: &AcceptanceStatus,
    performance: &AcceptanceStatus,
    resources: &AcceptanceStatus,
) -> AcceptanceStatus {
    overall_status_with_accelerator(behavior, performance, resources, &AcceptanceStatus::Pass)
}

pub fn overall_status_with_accelerator(
    behavior: &AcceptanceStatus,
    performance: &AcceptanceStatus,
    resources: &AcceptanceStatus,
    accelerator: &AcceptanceStatus,
) -> AcceptanceStatus {
    if [behavior, performance, resources, accelerator]
        .iter()
        .any(|status| matches!(status, AcceptanceStatus::Fail))
    {
        return AcceptanceStatus::Fail;
    }

    if [behavior, performance, resources, accelerator]
        .iter()
        .any(|status| matches!(status, AcceptanceStatus::Blocked))
    {
        return AcceptanceStatus::Blocked;
    }

    if [behavior, performance, resources, accelerator]
        .iter()
        .any(|status| matches!(status, AcceptanceStatus::Skipped))
    {
        return AcceptanceStatus::Skipped;
    }

    if matches!(behavior, AcceptanceStatus::Pass)
        && matches!(performance, AcceptanceStatus::Pass)
        && matches!(
            resources,
            AcceptanceStatus::Pass | AcceptanceStatus::NotApplicable
        )
        && matches!(
            accelerator,
            AcceptanceStatus::Pass | AcceptanceStatus::NotApplicable
        )
    {
        AcceptanceStatus::Pass
    } else {
        AcceptanceStatus::NotMeasured
    }
}
