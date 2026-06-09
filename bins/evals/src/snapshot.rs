use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::result::{AcceleratorClass, EvalDepth};
use crate::scenario::CapabilityName;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalSnapshot {
    pub schema_version: u32,
    pub id: String,
    pub git_sha: Option<String>,
    #[serde(default)]
    pub cells: Vec<SnapshotCell>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotCell {
    pub id: String,
    pub bundle_id: String,
    pub scenario: String,
    pub capability: CapabilityName,
    #[serde(default = "default_depth")]
    pub depth: EvalDepth,
    #[serde(default = "default_model_family")]
    pub model_family: String,
    #[serde(default = "default_checkpoint_format")]
    pub checkpoint_format: String,
    #[serde(default = "default_quantization")]
    pub quantization: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub selector: Option<String>,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub profile_features: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub profiles: Vec<String>,
    #[serde(default)]
    pub requested_accelerator: Option<AcceleratorClass>,
    #[serde(default)]
    pub artifact: ArtifactRequirement,
    #[serde(default)]
    pub budgets: CellBudgets,
}

impl SnapshotCell {
    pub fn applies_to_profile(&self, profile: &str) -> bool {
        self.profiles.is_empty() || self.profiles.iter().any(|candidate| candidate == profile)
    }

    pub fn features_for_profile(&self, profile: &str) -> Vec<String> {
        let mut features = self.features.clone();
        if let Some(extra) = self.profile_features.get(profile) {
            for feature in extra {
                if !features.contains(feature) {
                    features.push(feature.clone());
                }
            }
        }
        features
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRequirement {
    #[serde(default)]
    pub requires_hf_token: bool,
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub allow_missing: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CellBudgets {
    pub max_wall_time_secs: Option<u64>,
    pub max_artifact_bytes: Option<u64>,
    pub max_rss_bytes: Option<u64>,
    #[serde(default)]
    pub cpu_depth_allowed: Vec<EvalDepth>,
}

impl CellBudgets {
    pub fn wall_time_secs(&self) -> u64 {
        self.max_wall_time_secs.unwrap_or(900)
    }
}

pub fn load_snapshot(path: &Path) -> Result<EvalSnapshot> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read snapshot `{}`", path.display()))?;
    let snapshot = toml::from_str::<EvalSnapshot>(&raw)
        .with_context(|| format!("failed to parse snapshot `{}`", path.display()))?;
    Ok(snapshot)
}

fn default_depth() -> EvalDepth {
    EvalDepth::Smoke
}

fn default_model_family() -> String {
    "unknown".to_owned()
}

fn default_checkpoint_format() -> String {
    "unknown".to_owned()
}

fn default_quantization() -> String {
    "default".to_owned()
}

fn default_backend() -> String {
    "unknown".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_snapshot_cell_with_quant_grouping() {
        let raw = r#"
schema_version = 1
id = "curated-v2-smoke"
git_sha = "abc123"

[[cells]]
id = "qwen3_4b__chat_smoke__smoke__hf_default"
bundle_id = "qwen3_4b"
scenario = "chat_smoke"
capability = "chat"
depth = "smoke"
model_family = "qwen3"
checkpoint_format = "hf_safetensors"
quantization = "default"
backend = "mistralrs"
features = ["model-qwen3-4b"]
profiles = ["local-cpu-x86_64"]
requested_accelerator = "cpu"

[cells.profile_features]
"local-cpu-x86_64" = ["accelerate"]

[cells.artifact]
requires_hf_token = true
patterns = ["*.safetensors"]

[cells.budgets]
max_wall_time_secs = 60
cpu_depth_allowed = ["smoke"]
"#;

        let snapshot = toml::from_str::<EvalSnapshot>(raw).unwrap();

        assert_eq!(snapshot.id, "curated-v2-smoke");
        assert_eq!(snapshot.cells.len(), 1);
        assert_eq!(snapshot.cells[0].quantization, "default");
        assert_eq!(
            snapshot.cells[0].requested_accelerator,
            Some(AcceleratorClass::Cpu)
        );
        assert_eq!(
            snapshot.cells[0].features_for_profile("local-cpu-x86_64"),
            vec!["model-qwen3-4b".to_owned(), "accelerate".to_owned()]
        );
        assert!(snapshot.cells[0].applies_to_profile("local-cpu-x86_64"));
        assert!(!snapshot.cells[0].applies_to_profile("apple-metal"));
    }
}
