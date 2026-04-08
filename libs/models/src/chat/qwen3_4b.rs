use std::path::{Path, PathBuf};

use hf_hub::{Cache, Repo, RepoType};
use motlie_model::eval::EvalTrack;
use motlie_model::{BundleId, ModelBundle, ModelError, StartOptions};
use motlie_model_mistral::{MistralTextBundle, MistralTextSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleArtifacts, BundleDescriptor,
    BundleFamily, BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "qwen/qwen3_4b";

#[derive(Clone, Debug)]
pub struct Qwen3_4B {
    inner: MistralTextBundle,
}

impl Qwen3_4B {
    pub fn new() -> Self {
        Self {
            inner: MistralTextBundle::new(MistralTextSpec::qwen3_4b()),
        }
    }
}

impl Default for Qwen3_4B {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ModelBundle for Qwen3_4B {
    fn id(&self) -> &BundleId {
        self.inner.id()
    }

    fn metadata(&self) -> &motlie_model::BundleMetadata {
        self.inner.metadata()
    }

    fn capabilities(&self) -> &motlie_model::Capabilities {
        self.inner.capabilities()
    }

    async fn start(
        &self,
        options: StartOptions,
    ) -> Result<Box<dyn motlie_model::BundleHandle>, ModelError> {
        let StartOptions {
            artifact_policy,
            quantization,
            unpack_root,
            max_concurrency,
        } = options;
        let artifact_policy = match artifact_policy {
            Some(motlie_model::ArtifactPolicy::LocalOnly { root }) => {
                Some(motlie_model::ArtifactPolicy::LocalOnly {
                    root: resolve_local_snapshot_root(&root)?,
                })
            }
            other => other,
        };
        let options = StartOptions {
            artifact_policy,
            quantization,
            unpack_root,
            max_concurrency,
        };
        self.inner.start(options).await
    }
}

pub fn descriptor() -> BundleDescriptor {
    BundleDescriptor {
        id: BundleId::new("qwen3_4b"),
        display_name: "Qwen3 4B".into(),
        family: BundleFamily::Qwen,
        capabilities: motlie_model::Capabilities::chat_and_completion(),
        backend: BackendKind::MistralRs,
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![BuildConstraint::Feature("backend-mistral".into())],
        },
        eval_tracks: vec![
            EvalTrack::Chat,
            EvalTrack::Reasoning,
            EvalTrack::Summarization,
            EvalTrack::Classification,
        ],
        artifacts: Some(BundleArtifacts {
            control_name: "qwen3_4b",
            source: ArtifactSource::HuggingFace {
                repo: "Qwen/Qwen3-4B",
            },
            include: vec![
                ArtifactRule::Exact("config.json"),
                ArtifactRule::Exact("tokenizer.json"),
                ArtifactRule::Exact("tokenizer_config.json"),
                ArtifactRule::Exact("generation_config.json"),
                ArtifactRule::Exact("special_tokens_map.json"),
                ArtifactRule::Suffix(".safetensors"),
                ArtifactRule::Suffix(".safetensors.index.json"),
            ],
        }),
    }
}

pub fn bundle() -> Box<dyn ModelBundle> {
    Box::new(Qwen3_4B::new())
}

fn resolve_local_snapshot_root(root: &Path) -> Result<PathBuf, ModelError> {
    let repo = Cache::new(root.to_path_buf()).repo(Repo::new(
        "Qwen/Qwen3-4B".to_owned(),
        RepoType::Model,
    ));

    let config = repo.get("config.json").ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached `config.json` for `Qwen/Qwen3-4B` under `{}`",
            root.display()
        ))
    })?;

    if repo.get("tokenizer.json").is_none() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached `tokenizer.json` for `Qwen/Qwen3-4B` under `{}`",
            root.display()
        )));
    }

    let snapshot_dir = config.parent().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` found invalid cache layout for `Qwen/Qwen3-4B` under `{}`",
            root.display()
        ))
    })?;

    Ok(snapshot_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::CapabilityDescriptor;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "qwen3_4b");
        assert_eq!(descriptor.display_name, "Qwen3 4B");
        assert_eq!(descriptor.family, BundleFamily::Qwen);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert!(descriptor.eval_tracks.contains(&EvalTrack::Chat));
        assert!(descriptor.eval_tracks.contains(&EvalTrack::Reasoning));
        assert!(
            descriptor
                .capabilities
                .supports(motlie_model::CapabilityKind::Chat)
        );
        assert!(
            descriptor
                .capabilities
                .supports(motlie_model::CapabilityKind::Completion)
        );
        assert_eq!(
            descriptor.capability_descriptors(),
            &[
                CapabilityDescriptor::chat(),
                CapabilityDescriptor::completion(),
            ]
        );

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "qwen3_4b");
        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("model-00001-of-00002.safetensors"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn default_catalog_includes_qwen3_bundle() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("qwen3_4b");

        #[cfg(feature = "model-qwen3-4b")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Chat)
                .any(|b| b.id == bundle_id));
        }
    }

    #[test]
    fn local_snapshot_resolution_rejects_missing_cache() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should be creatable");

        let error =
            resolve_local_snapshot_root(&root).expect_err("missing local cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("config.json")
        ));
    }

    #[test]
    fn local_snapshot_resolution_accepts_complete_hf_cache_layout() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_cache(&root, "Qwen/Qwen3-4B", "main");

        std::fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        std::fs::write(snapshot.join("tokenizer.json"), "{}").expect("tokenizer should be writable");
        std::fs::write(snapshot.join("model-00001-of-00002.safetensors"), "stub")
            .expect("weights should be writable");

        let resolved = resolve_local_snapshot_root(&root)
            .expect("complete cache layout should resolve to snapshot path");

        assert_eq!(resolved, snapshot);
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-qwen3-test-{unique}"))
    }

    fn create_fake_hf_cache(root: &Path, model_id: &str, revision: &str) -> PathBuf {
        let repo_folder = format!("models--{}", model_id.replace('/', "--"));
        let repo_root = root.join(repo_folder);
        let refs_dir = repo_root.join("refs");
        let snapshots_dir = repo_root.join("snapshots");
        let commit = "test-commit";
        let snapshot = snapshots_dir.join(commit);

        std::fs::create_dir_all(&snapshot).expect("snapshot dir should be creatable");
        std::fs::create_dir_all(&refs_dir).expect("refs dir should be creatable");
        std::fs::write(refs_dir.join(revision), commit).expect("ref file should be writable");

        snapshot
    }
}
