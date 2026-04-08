use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use hf_hub::{Cache, Repo, RepoType};
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind,
    ChatModel, CompletionModel, EmbeddingModel, EmbeddingRequest as ModelEmbeddingRequest,
    EmbeddingResponse, LoadedBundleDescriptor, ModelBundle, ModelError, StartOptions,
};

/// Static bundle specification for a curated Mistral-backed embedding stack.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_id: &'static str,
    pub capabilities: Capabilities,
}

impl MistralEmbeddingSpec {
    pub fn embeddinggemma_300m() -> Self {
        Self {
            id: BundleId::new("embeddinggemma_300m"),
            display_name: "EmbeddingGemma 300M",
            model_id: "google/embeddinggemma-300m",
            capabilities: Capabilities::embeddings_only(),
        }
    }
}

/// Generic `ModelBundle` implementation backed by `mistralrs` embeddings.
#[derive(Clone, Debug)]
pub struct MistralEmbeddingBundle {
    metadata: BundleMetadata,
    model_id: &'static str,
}

impl MistralEmbeddingBundle {
    pub fn new(spec: MistralEmbeddingSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id,
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities,
            },
            model_id: spec.model_id,
        }
    }
}

#[async_trait]
impl ModelBundle for MistralEmbeddingBundle {
    fn id(&self) -> &BundleId {
        &self.metadata.id
    }

    fn metadata(&self) -> &BundleMetadata {
        &self.metadata
    }

    fn capabilities(&self) -> &Capabilities {
        &self.metadata.capabilities
    }

    async fn start(&self, options: StartOptions) -> Result<Box<dyn BundleHandle>, ModelError> {
        let model = build_embedding_model(self.model_id, options).await?;

        Ok(Box::new(MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
            },
            runtime: Box::new(MistralRuntime { model }),
        }))
    }
}

#[async_trait]
trait EmbeddingRuntime: Send + Sync {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError>;
}

struct MistralRuntime {
    model: mistralrs::Model,
}

#[async_trait]
impl EmbeddingRuntime for MistralRuntime {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        let builder = request
            .inputs
            .into_iter()
            .fold(EmbeddingRequest::builder(), |builder, input| {
                builder.add_prompt(input)
            });

        let vectors = self
            .model
            .generate_embeddings(builder)
            .await
            .map_err(|err| ModelError::Internal(err.to_string()))?;

        Ok(EmbeddingResponse { vectors })
    }
}

struct MistralEmbeddingHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn EmbeddingRuntime>,
}

#[async_trait]
impl BundleHandle for MistralEmbeddingHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn chat(&self) -> Result<&dyn ChatModel, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
    }

    fn completion(&self) -> Result<&dyn CompletionModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }

    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError> {
        Ok(self)
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl EmbeddingModel for MistralEmbeddingHandle {
    async fn embed(&self, request: ModelEmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        self.runtime.embed(request).await
    }
}

async fn build_embedding_model(
    model_id: &str,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        artifact_policy,
        unpack_root: _,
        max_concurrency,
    } = options;

    let mut builder = EmbeddingModelBuilder::new(model_id.to_owned());

    if let Some(artifact_policy) = artifact_policy {
        builder = configure_artifact_policy(builder, model_id, artifact_policy)?;
    }
    if let Some(max_num_seqs) = max_concurrency {
        builder = builder.with_max_num_seqs(max_num_seqs);
    }

    builder
        .build()
        .await
        .map_err(|err| ModelError::Internal(err.to_string()))
}

fn configure_artifact_policy(
    builder: EmbeddingModelBuilder,
    model_id: &str,
    policy: ArtifactPolicy,
) -> Result<EmbeddingModelBuilder, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => Ok(match root {
            Some(root) => builder.from_hf_cache_path(resolve_hf_cache_path(root)),
            None => builder,
        }),
        ArtifactPolicy::LocalOnly { root } => {
            validate_local_artifacts(model_id, &root)?;
            Ok(builder.from_hf_cache_path(resolve_hf_cache_path(root)))
        }
    }
}

fn validate_local_artifacts(model_id: &str, root: &Path) -> Result<(), ModelError> {
    let repo = Cache::new(root.to_path_buf()).repo(Repo::new(model_id.to_owned(), RepoType::Model));

    let config = repo.get("config.json").ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached `config.json` for `{model_id}` under `{}`",
            root.display()
        ))
    })?;

    if repo.get("tokenizer.json").is_none() && repo.get("tokenizer.model").is_none() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached tokenizer files for `{model_id}` under `{}`",
            root.display()
        )));
    }

    let snapshot_dir = config.parent().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` found invalid cache layout for `{model_id}` under `{}`",
            root.display()
        ))
    })?;

    let has_weights = fs::read_dir(snapshot_dir)
        .map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to inspect cached artifacts for `{model_id}` in `{}`: {err}",
                snapshot_dir.display()
            ))
        })?
        .filter_map(Result::ok)
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| {
                    name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
                })
                .unwrap_or(false)
        });

    if !has_weights {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached weight files for `{model_id}` under `{}`",
            root.display()
        )));
    }

    Ok(())
}

fn resolve_hf_cache_path(root: PathBuf) -> PathBuf {
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct StubRuntime;

    #[async_trait]
    impl EmbeddingRuntime for StubRuntime {
        async fn embed(
            &self,
            request: ModelEmbeddingRequest,
        ) -> Result<EmbeddingResponse, ModelError> {
            Ok(EmbeddingResponse {
                vectors: request
                    .inputs
                    .into_iter()
                    .map(|input| vec![input.len() as f32])
                    .collect(),
            })
        }
    }

    #[test]
    fn embeddinggemma_spec_has_expected_identity() {
        let spec = MistralEmbeddingSpec::embeddinggemma_300m();

        assert_eq!(spec.id.as_str(), "embeddinggemma_300m");
        assert_eq!(spec.display_name, "EmbeddingGemma 300M");
        assert_eq!(spec.model_id, "google/embeddinggemma-300m");
        assert!(spec.capabilities.supports(CapabilityKind::Embeddings));
    }

    #[tokio::test]
    async fn embedding_handle_rejects_unsupported_capabilities() {
        let handle = MistralEmbeddingHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("embeddinggemma_300m"),
                display_name: "EmbeddingGemma 300M".into(),
                capabilities: Capabilities::embeddings_only(),
            },
            runtime: Box::new(StubRuntime),
        };

        assert!(matches!(
            handle.chat(),
            Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
        ));
        assert!(matches!(
            handle.completion(),
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion
            ))
        ));
        assert!(handle.embeddings().is_ok());
        let response = handle
            .embed(ModelEmbeddingRequest {
                inputs: vec!["abc".into(), "abcd".into()],
            })
            .await
            .expect("stub runtime should succeed");
        assert_eq!(response.vectors, vec![vec![3.0], vec![4.0]]);
    }

    #[test]
    fn local_only_policy_rejects_missing_cache() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("temp root should be creatable");

        let err = configure_artifact_policy(
            EmbeddingModelBuilder::new("google/embeddinggemma-300m"),
            "google/embeddinggemma-300m",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .err()
        .expect("missing artifacts should fail closed");

        assert!(
            matches!(err, ModelError::InvalidConfiguration(message) if message.contains("config.json"))
        );
    }

    #[test]
    fn local_only_policy_accepts_valid_hf_cache_layout() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_cache(&root, "google/embeddinggemma-300m", "main");

        fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        fs::write(snapshot.join("tokenizer.json"), "{}").expect("tokenizer should be writable");
        fs::write(snapshot.join("model-00001-of-00001.safetensors"), "stub")
            .expect("weights should be writable");

        configure_artifact_policy(
            EmbeddingModelBuilder::new("google/embeddinggemma-300m"),
            "google/embeddinggemma-300m",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect("complete cache layout should be accepted");
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-model-mistral-test-{unique}"))
    }

    fn create_fake_hf_cache(root: &Path, model_id: &str, revision: &str) -> PathBuf {
        let repo_folder = format!("models--{}", model_id.replace('/', "--"));
        let repo_root = root.join(repo_folder);
        let refs_dir = repo_root.join("refs");
        let snapshots_dir = repo_root.join("snapshots");
        let commit = "test-commit";
        let snapshot = snapshots_dir.join(commit);

        fs::create_dir_all(&snapshot).expect("snapshot dir should be creatable");
        fs::create_dir_all(&refs_dir).expect("refs dir should be creatable");
        fs::write(refs_dir.join(revision), commit).expect("ref file should be writable");

        snapshot
    }
}
