use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScenarioSummary {
    pub id: String,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    pub schema_version: u32,
    pub id: String,
    pub capability: CapabilityName,
    pub summary: String,
    pub bundle_filter: BundleFilter,
    pub input: EmbeddingsInput,
    pub assertions: EmbeddingsAssertions,
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileConfig>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityName {
    Embeddings,
}

impl CapabilityName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BundleFilter {
    pub capability: CapabilityName,
    #[serde(default)]
    pub backend: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsInput {
    pub custom_text: String,
    pub similar_a: String,
    pub similar_b: String,
    pub dissimilar_a: String,
    pub dissimilar_b: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsAssertions {
    pub min_embedding_dimensions: usize,
    pub similarity_order: SimilarityOrder,
    pub min_similarity_gap: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityOrder {
    SimilarGtDissimilar,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetricsConfig {
    #[serde(default)]
    pub capture_startup_ms: bool,
    #[serde(default)]
    pub capture_request_latency: bool,
    #[serde(default)]
    pub capture_embedding_dimensions: bool,
    #[serde(default)]
    pub capture_vectors_per_second: bool,
    #[serde(default)]
    pub capture_peak_rss: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProfileConfig {
    pub gates: Option<ProfileGates>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProfileGates {
    pub max_swap_used_bytes: Option<u64>,
}

impl Scenario {
    pub fn gates_for_profile(&self, profile_name: &str) -> Option<&ProfileGates> {
        self.profiles
            .get(profile_name)
            .and_then(|profile| profile.gates.as_ref())
    }
}

pub fn list_scenarios(eval_root: &Path) -> Result<Vec<ScenarioSummary>> {
    let scenario_dir = eval_root.join("scenarios");
    let mut scenarios = Vec::new();

    for entry in fs::read_dir(&scenario_dir)
        .with_context(|| format!("failed to read `{}`", scenario_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        scenarios.push(ScenarioSummary {
            id: id.to_owned(),
            path: path.display().to_string(),
        });
    }

    scenarios.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(scenarios)
}

pub fn load_scenario(eval_root: &Path, scenario_id: &str) -> Result<Scenario> {
    let path = scenario_path(eval_root, scenario_id);
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read `{}`", path.display()))?;
    let scenario = toml::from_str::<Scenario>(&raw)
        .with_context(|| format!("failed to parse `{}`", path.display()))?;
    Ok(scenario)
}

fn scenario_path(eval_root: &Path, scenario_id: &str) -> PathBuf {
    eval_root
        .join("scenarios")
        .join(format!("{scenario_id}.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_repo_scenarios() {
        let eval_root = repo_eval_root();

        let scenarios = list_scenarios(&eval_root).unwrap();

        assert!(scenarios
            .iter()
            .any(|scenario| scenario.id == "embeddings_similarity"));
    }

    #[test]
    fn parses_embeddings_similarity() {
        let scenario = load_scenario(&repo_eval_root(), "embeddings_similarity").unwrap();

        assert_eq!(scenario.capability, CapabilityName::Embeddings);
        assert_eq!(
            scenario.bundle_filter.capability,
            CapabilityName::Embeddings
        );
        assert_eq!(
            scenario.assertions.similarity_order,
            SimilarityOrder::SimilarGtDissimilar
        );
        assert!(scenario.profiles.contains_key("dgx-spark"));
        assert!(scenario.profiles.contains_key("cuda-workstation"));
    }

    fn repo_eval_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("bins/evals should live two levels below the repo root")
            .join("evals")
    }
}
