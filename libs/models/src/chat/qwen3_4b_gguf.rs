use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{BundleId, ModelBundle, ModelError, StartOptions};
use motlie_model_llama_cpp::{LlamaCppTextBundle, LlamaCppTextSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleArtifacts, BundleDescriptor,
    BundleFamily, BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "qwen/qwen3_4b_gguf";

#[derive(Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct Qwen3_4B_Gguf {
    inner: LlamaCppTextBundle,
}

impl Qwen3_4B_Gguf {
    pub fn new() -> Self {
        Self {
            inner: LlamaCppTextBundle::new(LlamaCppTextSpec::qwen3_4b()),
        }
    }
}

impl Default for Qwen3_4B_Gguf {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ModelBundle for Qwen3_4B_Gguf {
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
                    root: resolve_local_gguf_root(&root)?,
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

/// Curated bundle descriptor for Qwen3 4B running on the llama.cpp backend
/// with GGUF-quantized weights.
///
/// ## Weight compatibility with mistral.rs
///
/// The mistral.rs `qwen3_4b` bundle uses **safetensors** weights from
/// `Qwen/Qwen3-4B`. This bundle uses **GGUF** weights from
/// `Qwen/Qwen3-4B-GGUF`. The two artifact sets are **not interchangeable** —
/// each backend requires its own format — but they target the identical
/// upstream Qwen3-4B architecture. Both backends produce equivalent inference
/// results at the same quantization level.
pub fn descriptor() -> BundleDescriptor {
    BundleDescriptor {
        id: BundleId::new("qwen3_4b_gguf"),
        display_name: "Qwen3 4B (GGUF/llama.cpp)".into(),
        family: BundleFamily::Qwen,
        capabilities: motlie_model::Capabilities::chat_and_completion(),
        backend: BackendKind::LlamaCpp,
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![BuildConstraint::Feature("backend-llama-cpp".into())],
        },
        eval_tracks: vec![
            EvalTrack::Chat,
            EvalTrack::Reasoning,
            EvalTrack::Summarization,
            EvalTrack::Classification,
        ],
        artifacts: Some(BundleArtifacts {
            control_name: "qwen3_4b_gguf",
            source: ArtifactSource::HuggingFace {
                repo: "Qwen/Qwen3-4B-GGUF",
            },
            include: vec![
                ArtifactRule::Suffix("-Q4_K_M.gguf"),
                ArtifactRule::Suffix("-Q8_0.gguf"),
                ArtifactRule::Suffix("-f16.gguf"),
            ],
        }),
    }
}

pub fn bundle() -> Box<dyn ModelBundle> {
    Box::new(Qwen3_4B_Gguf::new())
}

fn resolve_local_gguf_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_gguf_snapshot("Qwen/Qwen3-4B-GGUF", root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::CapabilityDescriptor;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "qwen3_4b_gguf");
        assert_eq!(descriptor.display_name, "Qwen3 4B (GGUF/llama.cpp)");
        assert_eq!(descriptor.family, BundleFamily::Qwen);
        assert_eq!(descriptor.backend, BackendKind::LlamaCpp);
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
        assert_eq!(artifacts.control_name, "qwen3_4b_gguf");
        assert!(artifacts.includes("qwen3-4b-Q4_K_M.gguf"));
        assert!(artifacts.includes("qwen3-4b-Q8_0.gguf"));
        assert!(!artifacts.includes("README.md"));
        assert!(!artifacts.includes("config.json"));
    }

    #[test]
    fn local_gguf_resolution_rejects_missing_cache() {
        let root = unique_temp_dir();

        let error =
            resolve_local_gguf_root(&root).expect_err("missing cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("refs/main")
        ));
    }

    #[test]
    fn local_gguf_resolution_rejects_cache_without_gguf_files() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_gguf_cache(&root, "Qwen/Qwen3-4B-GGUF");
        // snapshot exists but has no .gguf files

        let error = resolve_local_gguf_root(&root)
            .expect_err("cache without .gguf should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains(".gguf")
        ));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn local_gguf_resolution_accepts_cache_with_gguf_file() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_gguf_cache(&root, "Qwen/Qwen3-4B-GGUF");
        std::fs::write(snapshot.join("Qwen3-4B-Q4_K_M.gguf"), "stub")
            .expect("gguf stub should be writable");

        let resolved =
            resolve_local_gguf_root(&root).expect("cache with .gguf should resolve");

        assert_eq!(resolved, snapshot);
        std::fs::remove_dir_all(&root).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-qwen3-gguf-test-{unique}"))
    }

    fn create_fake_hf_gguf_cache(root: &Path, model_id: &str) -> PathBuf {
        let repo_folder = format!("models--{}", model_id.replace('/', "--"));
        let repo_root = root.join(repo_folder);
        let refs_dir = repo_root.join("refs");
        let commit = "test-commit";
        let snapshot = repo_root.join("snapshots").join(commit);

        std::fs::create_dir_all(&snapshot).expect("snapshot dir should be creatable");
        std::fs::create_dir_all(&refs_dir).expect("refs dir should be creatable");
        std::fs::write(refs_dir.join("main"), commit).expect("ref file should be writable");

        snapshot
    }
}
