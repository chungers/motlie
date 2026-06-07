use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScenarioSummary {
    pub id: String,
    pub path: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_repo_scenarios() {
        let eval_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("bins/evals should live two levels below the repo root")
            .join("evals");

        let scenarios = list_scenarios(&eval_root).unwrap();

        assert!(scenarios
            .iter()
            .any(|scenario| scenario.id == "embeddings_similarity"));
    }
}
