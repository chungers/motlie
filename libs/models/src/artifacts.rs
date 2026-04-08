use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use motlie_model::BundleId;

use crate::catalog::Catalog;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactSource {
    HuggingFace { repo: &'static str },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactRule {
    Exact(&'static str),
    Suffix(&'static str),
}

impl ArtifactRule {
    fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(expected) => filename == *expected,
            Self::Suffix(suffix) => filename.ends_with(suffix),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleArtifacts {
    pub control_name: &'static str,
    pub source: ArtifactSource,
    pub include: Vec<ArtifactRule>,
}

impl BundleArtifacts {
    pub fn includes(&self, filename: &str) -> bool {
        self.include.iter().any(|rule| rule.matches(filename))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactDownloadSummary {
    pub bundle_id: BundleId,
    pub downloaded: Vec<PathBuf>,
}

pub fn default_artifact_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/models/hf-cache")
}

pub fn download_bundle_artifacts(
    catalog: &Catalog,
    bundle_id: &BundleId,
    artifact_root: &Path,
) -> Result<ArtifactDownloadSummary> {
    let descriptor = catalog
        .bundle(bundle_id)
        .with_context(|| format!("unknown bundle `{bundle_id}`"))?;
    let artifacts = descriptor
        .artifacts
        .as_ref()
        .with_context(|| format!("bundle `{bundle_id}` does not define artifacts"))?;

    match &artifacts.source {
        ArtifactSource::HuggingFace { repo } => {
            std::fs::create_dir_all(artifact_root).with_context(|| {
                format!(
                    "failed to create artifact root `{}`",
                    artifact_root.display()
                )
            })?;

            let api = ApiBuilder::new()
                .with_cache_dir(artifact_root.to_path_buf())
                .build()
                .context("failed to create Hugging Face API client")?;
            let repo_api = api.model((*repo).to_string());
            let info = repo_api
                .info()
                .with_context(|| format!("failed to inspect model repo `{repo}`"))?;

            let mut downloaded = Vec::new();
            for sibling in info.siblings {
                if artifacts.includes(&sibling.rfilename) {
                    let path = repo_api.get(&sibling.rfilename).with_context(|| {
                        format!(
                            "failed to download `{}` from repo `{repo}`",
                            sibling.rfilename
                        )
                    })?;
                    downloaded.push(path);
                }
            }

            downloaded.sort();

            Ok(ArtifactDownloadSummary {
                bundle_id: bundle_id.clone(),
                downloaded,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_rules_match_expected_files() {
        let artifacts = BundleArtifacts {
            control_name: "embeddinggemma_300m",
            source: ArtifactSource::HuggingFace {
                repo: "google/embeddinggemma-300m",
            },
            include: vec![
                ArtifactRule::Exact("config.json"),
                ArtifactRule::Suffix(".safetensors"),
            ],
        };

        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("weights-00001.safetensors"));
        assert!(!artifacts.includes("README.md"));
    }
}
