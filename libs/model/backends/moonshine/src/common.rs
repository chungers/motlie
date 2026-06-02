use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use motlie_model::{ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint};
use tokenizers::Tokenizer;

pub(crate) use motlie_model::metrics_runtime::{
    RuntimeMetricState, lock_metrics, observe_latency, observe_memory,
};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub(crate) struct MoonshineArtifactPaths {
    pub frontend: PathBuf,
    pub encoder: PathBuf,
    pub adapter: PathBuf,
    pub cross_kv: PathBuf,
    pub decoder_kv: PathBuf,
    pub streaming_config: PathBuf,
    pub tokenizer_json: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoonshineArtifactSpec<'a> {
    pub frontend: &'a str,
    pub encoder: &'a str,
    pub adapter: &'a str,
    pub cross_kv: &'a str,
    pub decoder_kv: &'a str,
    pub streaming_config: &'a str,
    pub tokenizer_json: &'a str,
}

pub(crate) fn resolve_onnx_artifacts(
    checkpoint: &ResolvedCheckpoint,
    spec: MoonshineArtifactSpec<'_>,
) -> Result<MoonshineArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "moonshine expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .ok_or_else(|| {
                ModelError::InvalidConfiguration(format!(
                    "onnx checkpoint path `{}` has no parent directory",
                    checkpoint.path.display()
                ))
            })?
            .to_path_buf()
    };

    build_artifacts(&root, spec)
}

pub(crate) fn configure_artifact_policy(
    spec: MoonshineArtifactSpec<'_>,
    policy: ArtifactPolicy,
) -> Result<MoonshineArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    build_artifacts(&root, spec)
}

fn build_artifacts(
    root: &Path,
    spec: MoonshineArtifactSpec<'_>,
) -> Result<MoonshineArtifactPaths, ModelError> {
    Ok(MoonshineArtifactPaths {
        frontend: require_file(root, spec.frontend)?,
        encoder: require_file(root, spec.encoder)?,
        adapter: require_file(root, spec.adapter)?,
        cross_kv: require_file(root, spec.cross_kv)?,
        decoder_kv: require_file(root, spec.decoder_kv)?,
        streaming_config: require_file(root, spec.streaming_config)?,
        tokenizer_json: require_file(root, spec.tokenizer_json)?,
    })
}

fn require_file(root: &Path, relative: &str) -> Result<PathBuf, ModelError> {
    let path = root.join(relative);
    if !path.is_file() {
        return Err(ModelError::InvalidConfiguration(format!(
            "required moonshine artifact `{relative}` not found under `{}`",
            root.display()
        )));
    }
    Ok(path)
}

pub(crate) struct StagedModelDir {
    path: PathBuf,
}

impl StagedModelDir {
    pub(crate) fn prepare(artifacts: &MoonshineArtifactPaths) -> Result<Self, ModelError> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| ModelError::Internal(format!("system clock error: {err}")))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "motlie-moonshine-{}-{}-{}",
            std::process::id(),
            unique,
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).map_err(|err| ModelError::BackendInitialization {
            backend: "moonshine",
            message: format!(
                "failed to create staged Moonshine runtime dir `{}`: {err}",
                path.display()
            ),
        })?;

        for (src, dest_name) in [
            (&artifacts.frontend, "frontend.ort"),
            (&artifacts.encoder, "encoder.ort"),
            (&artifacts.adapter, "adapter.ort"),
            (&artifacts.cross_kv, "cross_kv.ort"),
            (&artifacts.decoder_kv, "decoder_kv.ort"),
            (&artifacts.streaming_config, "streaming_config.json"),
            (&artifacts.tokenizer_json, "tokenizer.json"),
        ] {
            link_or_copy(src, &path.join(dest_name))?;
        }

        write_tokenizer_bin(&artifacts.tokenizer_json, &path.join("tokenizer.bin"))?;
        Ok(Self { path })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagedModelDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(unix)]
fn link_or_copy(src: &Path, dest: &Path) -> Result<(), ModelError> {
    use std::os::unix::fs::symlink;

    let symlink_src = src
        .canonicalize()
        .map_err(|err| ModelError::BackendInitialization {
            backend: "moonshine",
            message: format!(
                "failed to resolve `{}` before staging: {err}",
                src.display()
            ),
        })?;

    match symlink(&symlink_src, dest) {
        Ok(()) => Ok(()),
        Err(_) => match std::fs::hard_link(src, dest) {
            Ok(()) => Ok(()),
            Err(_) => {
                std::fs::copy(src, dest).map_err(|err| ModelError::BackendInitialization {
                    backend: "moonshine",
                    message: format!(
                        "failed to stage `{}` as `{}`: {err}",
                        src.display(),
                        dest.display()
                    ),
                })?;
                Ok(())
            }
        },
    }
}

#[cfg(not(unix))]
fn link_or_copy(src: &Path, dest: &Path) -> Result<(), ModelError> {
    std::fs::copy(src, dest).map_err(|err| ModelError::BackendInitialization {
        backend: "moonshine",
        message: format!(
            "failed to stage `{}` as `{}`: {err}",
            src.display(),
            dest.display()
        ),
    })?;
    Ok(())
}

fn write_tokenizer_bin(tokenizer_json: &Path, output: &Path) -> Result<(), ModelError> {
    let tokenizer =
        Tokenizer::from_file(tokenizer_json).map_err(|err| ModelError::BackendInitialization {
            backend: "moonshine",
            message: format!(
                "failed to load tokenizer json `{}`: {err}",
                tokenizer_json.display()
            ),
        })?;

    let mut by_id: BTreeMap<u32, String> = BTreeMap::new();
    for (token, id) in tokenizer.get_vocab(true) {
        by_id.insert(id, token);
    }

    let max_id = by_id.keys().copied().max().unwrap_or(0);
    let mut buffer = Vec::new();
    for id in 0..=max_id {
        let token = by_id.get(&id).map(String::as_str).unwrap_or("");
        write_token_bytes(&mut buffer, token_bytes(token))?;
    }

    std::fs::write(output, buffer).map_err(|err| ModelError::BackendInitialization {
        backend: "moonshine",
        message: format!(
            "failed to write generated tokenizer bin `{}`: {err}",
            output.display()
        ),
    })?;

    Ok(())
}

fn token_bytes(token: &str) -> Vec<u8> {
    if let Some(byte) = parse_hex_token(token) {
        vec![byte]
    } else {
        token.as_bytes().to_vec()
    }
}

fn parse_hex_token(token: &str) -> Option<u8> {
    if !(token.starts_with("<0x") && token.ends_with('>') && token.len() == 6) {
        return None;
    }
    u8::from_str_radix(&token[3..5], 16).ok()
}

fn write_token_bytes(out: &mut Vec<u8>, bytes: Vec<u8>) -> Result<(), ModelError> {
    let len = bytes.len();
    if len == 0 {
        out.push(0);
        return Ok(());
    }

    if len < 128 {
        let encoded = u8::try_from(len).map_err(|err| ModelError::Internal(err.to_string()))?;
        out.push(encoded);
    } else {
        if len >= 16_384 {
            return Err(ModelError::InvalidConfiguration(format!(
                "moonshine tokenizer token length {len} exceeds 16,383-byte binary encoding limit"
            )));
        }
        let low =
            u8::try_from((len % 128) + 128).map_err(|err| ModelError::Internal(err.to_string()))?;
        let high = u8::try_from(len / 128).map_err(|err| ModelError::Internal(err.to_string()))?;
        out.push(low);
        out.push(high);
    }
    out.extend_from_slice(&bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_tokens_are_converted_to_bytes() {
        assert_eq!(token_bytes("<0x20>"), vec![0x20]);
        assert_eq!(token_bytes("abc"), b"abc".to_vec());
    }

    #[test]
    fn oversized_token_lengths_fail_with_explicit_error() {
        let err = write_token_bytes(&mut Vec::new(), vec![0_u8; 16_384])
            .expect_err("oversized token must fail");
        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("16,383")));
    }
    #[cfg(unix)]
    #[test]
    fn link_or_copy_keeps_relative_sources_readable_from_temp_dest() {
        let original_cwd = std::env::current_dir().expect("current dir should be readable");
        let root =
            std::env::temp_dir().join(format!("moonshine-link-or-copy-{}", std::process::id()));
        let source_dir = root.join("source");
        let dest_dir = root.join("dest");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&source_dir).expect("source dir should be creatable");
        std::fs::create_dir_all(&dest_dir).expect("dest dir should be creatable");
        std::fs::write(source_dir.join("streaming_config.json"), b"{}")
            .expect("source file should be writable");

        std::env::set_current_dir(&source_dir).expect("test cwd should switch");
        link_or_copy(
            Path::new("streaming_config.json"),
            &dest_dir.join("streaming_config.json"),
        )
        .expect("relative source should stage");
        std::env::set_current_dir(original_cwd).expect("cwd should restore");

        let staged = std::fs::read_to_string(dest_dir.join("streaming_config.json"))
            .expect("staged file should be readable");
        assert_eq!(staged, "{}");
        std::fs::remove_dir_all(root).expect("temp root should be removable");
    }
}
