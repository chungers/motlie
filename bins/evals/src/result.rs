use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::metrics::{PerformanceMetrics, ResourceMetrics};
use crate::platform::PlatformSnapshot;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultRecord {
    pub schema_version: u32,
    pub identity: IdentitySection,
    pub selection: SelectionSection,
    pub profile: ProfileSection,
    pub platform: PlatformSnapshot,
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
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SelectionSection {
    pub bundle_id: String,
    pub selector: Option<String>,
    pub backend: Option<String>,
    pub checkpoint_format: Option<String>,
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
pub struct RuntimeSection {
    pub cargo_features: Vec<String>,
    pub build_profile: Option<String>,
    pub quantization: Option<String>,
    pub artifact_root: String,
    pub download_artifacts: bool,
    pub context_length: Option<u64>,
    pub gpu_layers: Option<u32>,
    pub env: BTreeMap<String, Option<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceSection {
    pub behavior_status: AcceptanceStatus,
    pub performance_status: AcceptanceStatus,
    pub resource_status: AcceptanceStatus,
    pub overall_status: AcceptanceStatus,
    pub failure_reason: Option<String>,
    pub assertions: Vec<AssertionOutcome>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssertionOutcome {
    pub name: String,
    pub status: AcceptanceStatus,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceStatus {
    Pass,
    Fail,
    Blocked,
    NotMeasured,
    NotApplicable,
}

pub fn overall_status(
    behavior: &AcceptanceStatus,
    performance: &AcceptanceStatus,
    resources: &AcceptanceStatus,
) -> AcceptanceStatus {
    if matches!(behavior, AcceptanceStatus::Fail)
        || matches!(performance, AcceptanceStatus::Fail)
        || matches!(resources, AcceptanceStatus::Fail)
    {
        return AcceptanceStatus::Fail;
    }

    if matches!(behavior, AcceptanceStatus::Blocked)
        || matches!(performance, AcceptanceStatus::Blocked)
        || matches!(resources, AcceptanceStatus::Blocked)
    {
        return AcceptanceStatus::Blocked;
    }

    if matches!(behavior, AcceptanceStatus::Pass)
        && matches!(performance, AcceptanceStatus::Pass)
        && matches!(
            resources,
            AcceptanceStatus::Pass | AcceptanceStatus::NotApplicable
        )
    {
        AcceptanceStatus::Pass
    } else {
        AcceptanceStatus::NotMeasured
    }
}
