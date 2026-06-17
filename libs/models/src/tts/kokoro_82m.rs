use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, CheckpointQuantization, ModelCheckpoint, ModelError, ModelIdentity,
    StartOptions,
};
use motlie_model_kokoro::{KokoroHandle, KokoroSpeechBundle, KokoroSpeechSpec};

use crate::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX;
use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "kokoro/kokoro_82m";

const HF_REPO: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";
const MODEL_FILE: &str = "onnx/model_quantized.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";
const VOICE_FILE: &str = "voices/af_bella.bin";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("kokoro_82m"),
        display_name: "Kokoro-82M v1.0 ONNX af_bella".into(),
        family: BundleFamily::Kokoro,
        capabilities: motlie_model::Capabilities::speech_buffered_and_streaming(),
        eval_tracks: vec![EvalTrack::Speech],
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![],
        },
    }
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Onnx,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact(MODEL_FILE),
            ArtifactRule::Exact(TOKENIZER_FILE),
            ArtifactRule::Exact(VOICE_FILE),
            ArtifactRule::Exact("model.onnx"),
            ArtifactRule::Exact("voices.bin"),
            ArtifactRule::Exact("tokens.txt"),
        ],
        quantization: Some(CheckpointQuantization::Onnx { bits: 8 }),
    }
}

pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();

    BundleDescriptor {
        id: identity.id.clone(),
        model_id: identity.id,
        display_name: identity.display_name.clone(),
        family: identity.family,
        capabilities: identity.capabilities,
        backend: BackendKind::Ort,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-kokoro".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "kokoro_82m",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = KokoroSpeechSpec::kokoro_82m();
    crate::ModelVariantDescriptor {
        backend: BackendKind::Ort,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

const INCREMENTAL_TOKENS_FILE: &str = "tokens.txt";

pub(crate) fn prepare_downloaded_artifacts(downloaded: &mut Vec<PathBuf>) -> Result<(), String> {
    let tokenizer_json = downloaded
        .iter()
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some(TOKENIZER_FILE))
        .cloned()
        .ok_or_else(|| format!("downloaded Kokoro bundle did not include `{TOKENIZER_FILE}`"))?;
    let parent = tokenizer_json.parent().ok_or_else(|| {
        format!(
            "downloaded Kokoro tokenizer path `{}` has no parent",
            tokenizer_json.display()
        )
    })?;
    let tokens_path = parent.join(INCREMENTAL_TOKENS_FILE);
    let tokenizer = std::fs::read_to_string(&tokenizer_json).map_err(|err| {
        format!(
            "failed to read Kokoro tokenizer `{}`: {err}",
            tokenizer_json.display()
        )
    })?;
    let tokens = tokens_txt_from_tokenizer_json(&tokenizer)?;
    let needs_write = std::fs::read_to_string(&tokens_path)
        .map(|existing| existing != tokens)
        .unwrap_or(true);
    if needs_write {
        std::fs::write(&tokens_path, tokens).map_err(|err| {
            format!(
                "failed to write Kokoro incremental `{}`: {err}",
                tokens_path.display()
            )
        })?;
    }
    if !downloaded.iter().any(|path| path == &tokens_path) {
        downloaded.push(tokens_path);
    }
    Ok(())
}

fn tokens_txt_from_tokenizer_json(tokenizer_json: &str) -> Result<String, String> {
    let root: serde_json::Value = serde_json::from_str(tokenizer_json)
        .map_err(|err| format!("failed to parse Kokoro tokenizer JSON: {err}"))?;
    let vocab = root
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "Kokoro tokenizer JSON missing object at model.vocab".to_string())?;
    let mut entries = Vec::with_capacity(vocab.len());
    for (token, id) in vocab {
        if token.contains('\n') || token.contains('\r') {
            return Err("Kokoro tokenizer token contains a newline".to_string());
        }
        let id = id
            .as_u64()
            .ok_or_else(|| format!("Kokoro tokenizer id for token `{token}` is not an integer"))?;
        entries.push((id, token.as_str()));
    }
    entries.sort_by_key(|(id, _)| *id);
    if entries.is_empty() {
        return Err("Kokoro tokenizer model.vocab is empty".to_string());
    }
    let mut output = String::new();
    for (id, token) in entries {
        output.push_str(token);
        output.push(' ');
        output.push_str(&id.to_string());
        output.push('\n');
    }
    Ok(output)
}

pub fn typed_bundle() -> KokoroSpeechBundle {
    KokoroSpeechBundle::new(KokoroSpeechSpec::kokoro_82m())
}

pub async fn start_typed(options: StartOptions) -> Result<KokoroHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_model_path,
        )?)
        .await
}

fn resolve_local_model_path(root: &Path) -> Result<PathBuf, ModelError> {
    if [MODEL_FILE, TOKENIZER_FILE, VOICE_FILE]
        .into_iter()
        .all(|filename| root.join(filename).is_file())
    {
        return Ok(root.to_path_buf());
    }

    let direct_model = root.join(Path::new(MODEL_FILE).file_name().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "invalid curated Kokoro model path `{MODEL_FILE}`"
        ))
    })?);
    let direct_tokenizer = root.join(TOKENIZER_FILE);
    let direct_voice = root.join(Path::new(VOICE_FILE).file_name().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "invalid curated Kokoro voice path `{VOICE_FILE}`"
        ))
    })?);
    if direct_model.is_file() && direct_tokenizer.is_file() && direct_voice.is_file() {
        return Ok(root.to_path_buf());
    }

    let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
    let repo_root = root.join(&repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached ONNX artifacts for `{HF_REPO}` under `{}` or a direct Kokoro artifact directory; no refs/main found",
            root.display()
        )));
    }

    let commit = std::fs::read_to_string(&main_ref).map_err(|err| {
        ModelError::InvalidConfiguration(format!(
            "failed to read HF cache ref for `{HF_REPO}`: {err}"
        ))
    })?;
    let snapshot_dir = repo_root.join("snapshots").join(commit.trim());

    if !snapshot_dir.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "HF cache snapshot for `{HF_REPO}` not found under `{}`",
            root.display()
        )));
    }

    for filename in [MODEL_FILE, TOKENIZER_FILE, VOICE_FILE] {
        if !snapshot_dir.join(filename).is_file() {
            return Err(ModelError::InvalidConfiguration(format!(
                "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires `{filename}` in cached snapshot for `{HF_REPO}` under `{}`",
                root.display()
            )));
        }
    }

    Ok(snapshot_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::{CapabilityKind, SpeechGeneration};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "kokoro_82m");
        assert_eq!(descriptor.display_name, "Kokoro-82M v1.0 ONNX af_bella");
        assert_eq!(descriptor.family, BundleFamily::Kokoro);
        assert_eq!(descriptor.backend, BackendKind::Ort);
        assert!(descriptor.capabilities.supports(CapabilityKind::Speech));
        assert!(
            descriptor
                .capabilities
                .supports_speech_generation(SpeechGeneration::Buffered)
        );
        assert!(
            descriptor
                .capabilities
                .supports_speech_generation(SpeechGeneration::Streaming)
        );
        assert_eq!(descriptor.eval_tracks, vec![EvalTrack::Speech]);

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "kokoro_82m");
        assert!(artifacts.includes(MODEL_FILE));
        assert!(artifacts.includes(TOKENIZER_FILE));
        assert!(artifacts.includes(VOICE_FILE));
        assert!(artifacts.includes("model.onnx"));
        assert!(artifacts.includes("voices.bin"));
        assert!(artifacts.includes("tokens.txt"));
    }

    #[test]
    fn tokenizer_json_generates_sherpa_tokens_txt() {
        let raw = r#"{
            "model": {
                "vocab": {
                    "b": 2,
                    " ": 1,
                    "\u0283": 3,
                    "a": 0
                }
            }
        }"#;

        let tokens = tokens_txt_from_tokenizer_json(raw).expect("tokens should generate");
        let esh = char::from_u32(0x0283).expect("esh char should exist");

        assert_eq!(tokens, format!("a 0\n  1\nb 2\n{} 3\n", esh));
    }

    #[test]
    fn prepare_downloaded_artifacts_rewrites_kokoro_tokens_file() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should exist");
        let tokenizer = root.join(TOKENIZER_FILE);
        let tokens = root.join(INCREMENTAL_TOKENS_FILE);
        std::fs::write(&tokenizer, r#"{"model":{"vocab":{"z":2," ":1,"a":0}}}"#)
            .expect("tokenizer should be writable");
        std::fs::write(&tokens, "<pad> 0\n").expect("old tokens should be writable");
        let mut downloaded = vec![tokenizer.clone(), tokens.clone()];

        prepare_downloaded_artifacts(&mut downloaded).expect("tokens should be prepared");

        assert_eq!(
            std::fs::read_to_string(&tokens).expect("tokens should be readable"),
            "a 0\n  1\nz 2\n"
        );
        assert!(downloaded.iter().any(|path| path == &tokens));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("kokoro_82m");

        #[cfg(feature = "model-kokoro-82m")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Speech)
                .any(|bundle| bundle.id == bundle_id));
        }
    }

    #[test]
    fn local_resolution_accepts_direct_artifact_dir() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should exist");
        std::fs::write(root.join("model_quantized.onnx"), "stub").expect("model");
        std::fs::write(root.join(TOKENIZER_FILE), "stub").expect("tokenizer");
        std::fs::write(root.join("af_bella.bin"), "stub").expect("voice");

        let resolved = resolve_local_model_path(&root).expect("direct dir should resolve");

        assert_eq!(resolved, root);
        std::fs::remove_dir_all(resolved).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-kokoro-82m-{unique}"))
    }
}
