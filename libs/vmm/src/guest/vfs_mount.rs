//! VMM-owned v1.5 guest-side VFS mounter.
//!
//! This module intentionally reuses `motlie_vfs::client::guest` instead of
//! copying mount-loop logic. The ownership boundary is the important point:
//! v1.5 product guest binaries live in VMM, while VFS remains the filesystem
//! library.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use motlie_vfs::client::guest::{GuestMountRunner, GuestMountSpec};

pub const CONTRACT_MARKER: &str = "MOTLIE_VMM_GUEST_MOUNTER_V1_5";

const DEFAULT_MOUNTS_PATH: &str = "/etc/motlie-vfs/mounts.yaml";
const DEFAULT_BACKEND_ENV_PATH: &str = "/etc/motlie/v1.5/backend.env";
const DEFAULT_HOST_CID: u32 = 2;
const DEFAULT_VFS_PORT: u32 = 5000;
const DEFAULT_CONNECT_RETRY_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_CONNECT_RETRY_DELAY: Duration = Duration::from_millis(250);

#[derive(Debug, Clone)]
struct Args {
    mounts_path: PathBuf,
    backend_env_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct RuntimeConfig {
    host_cid: u32,
    vfs_port: u32,
    connect_retry_timeout: Duration,
    connect_retry_delay: Duration,
}

#[derive(serde::Deserialize)]
struct MountConfig {
    mounts: Vec<MountEntry>,
}

#[derive(serde::Deserialize)]
struct MountEntry {
    tag: String,
    guest_path: String,
    #[serde(default)]
    read_only: bool,
}

pub fn main() -> Result<()> {
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--contract" || arg == "--contract-marker")
    {
        println!("{CONTRACT_MARKER}");
        return Ok(());
    }

    let args = parse_args(std::env::args().skip(1))?;
    let runtime = RuntimeConfig::from_backend_env(args.backend_env_path.as_deref())?;
    let specs = load_mount_specs(&args.mounts_path)?;

    eprintln!(
        "motlie-vfs-guest-v1_5: mounting {} filesystem(s) via cid={} port={}",
        specs.len(),
        runtime.host_cid,
        runtime.vfs_port
    );
    for spec in &specs {
        eprintln!(
            "  tag={} path={} ro={}",
            spec.tag, spec.guest_path, spec.read_only
        );
    }

    let runner = GuestMountRunner::new(specs);
    let handles = {
        use motlie_vfs::vsock::client::VsockClientTransport;

        runner.mount_all(move |tag: &str, rt: &tokio::runtime::Runtime| {
            let tag = tag.to_string();
            let connect_tag = tag.clone();
            let runtime = runtime.clone();
            let stream =
                rt.block_on(async move { connect_with_retry(&connect_tag, &runtime).await })?;

            Ok(VsockClientTransport::new(stream, &tag))
        })?
    };

    eprintln!("motlie-vfs-guest-v1_5: all mounts started, waiting...");
    let results = handles.join_all();
    let mut mount_failed = false;
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(()) => {}
            Err(e) => {
                mount_failed = true;
                eprintln!("mount {i} failed: {e}");
            }
        }
    }
    if mount_failed {
        anyhow::bail!("one or more guest mounts failed");
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args> {
    let mut mounts_path = None;
    let mut backend_env_path = Some(PathBuf::from(DEFAULT_BACKEND_ENV_PATH));
    let mut positional_mounts_path = None;
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--mounts" => {
                let path = iter.next().context("--mounts requires a path argument")?;
                mounts_path = Some(PathBuf::from(path));
            }
            "--backend-env" => {
                let path = iter
                    .next()
                    .context("--backend-env requires a path argument")?;
                backend_env_path = Some(PathBuf::from(path));
            }
            "--no-backend-env" => {
                backend_env_path = None;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ if arg.starts_with('-') => {
                anyhow::bail!("unknown argument: {arg}");
            }
            _ => {
                if positional_mounts_path.is_some() {
                    anyhow::bail!("only one positional mounts.yaml path is supported");
                }
                positional_mounts_path = Some(PathBuf::from(arg));
            }
        }
    }

    Ok(Args {
        mounts_path: mounts_path
            .or(positional_mounts_path)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MOUNTS_PATH)),
        backend_env_path,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: motlie-vfs-guest-v1_5 [--contract] [--mounts PATH] [--backend-env PATH|--no-backend-env] [PATH]"
    );
    eprintln!("Contract marker: {CONTRACT_MARKER}");
    eprintln!("Default mounts path: {DEFAULT_MOUNTS_PATH}");
    eprintln!("Default backend env path: {DEFAULT_BACKEND_ENV_PATH}");
}

impl RuntimeConfig {
    fn from_backend_env(path: Option<&Path>) -> Result<Self> {
        let file_values = match path {
            Some(path) => parse_env_file(path)?,
            None => HashMap::new(),
        };

        Ok(Self {
            host_cid: config_u32(&file_values, "MOTLIE_VFS_HOST_CID", DEFAULT_HOST_CID)?,
            vfs_port: config_u32(&file_values, "MOTLIE_VFS_PORT", DEFAULT_VFS_PORT)?,
            connect_retry_timeout: config_duration_ms(
                &file_values,
                "MOTLIE_VFS_CONNECT_TIMEOUT_MS",
                DEFAULT_CONNECT_RETRY_TIMEOUT,
            )?,
            connect_retry_delay: config_duration_ms(
                &file_values,
                "MOTLIE_VFS_CONNECT_RETRY_MS",
                DEFAULT_CONNECT_RETRY_DELAY,
            )?,
        })
    }
}

fn parse_env_file(path: &Path) -> Result<HashMap<String, String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read backend env {}", path.display()))?;
    let mut values = HashMap::new();

    for (line_no, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            anyhow::bail!(
                "invalid backend env line {} in {}: expected KEY=VALUE",
                line_no + 1,
                path.display()
            );
        };
        let key = key.trim();
        if key.is_empty() {
            anyhow::bail!(
                "invalid backend env line {} in {}: empty key",
                line_no + 1,
                path.display()
            );
        }
        values.insert(key.to_string(), unquote_env_value(value.trim()));
    }

    Ok(values)
}

fn unquote_env_value(value: &str) -> String {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let quote = bytes[0];
        if (quote == b'\'' || quote == b'"') && bytes[value.len() - 1] == quote {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

fn config_u32(values: &HashMap<String, String>, key: &str, default: u32) -> Result<u32> {
    match std::env::var(key).ok().or_else(|| values.get(key).cloned()) {
        Some(value) => value
            .parse::<u32>()
            .with_context(|| format!("invalid {key} value {value:?}")),
        None => Ok(default),
    }
}

fn config_duration_ms(
    values: &HashMap<String, String>,
    key: &str,
    default: Duration,
) -> Result<Duration> {
    let Some(value) = std::env::var(key).ok().or_else(|| values.get(key).cloned()) else {
        return Ok(default);
    };
    let millis = value
        .parse::<u64>()
        .with_context(|| format!("invalid {key} value {value:?}"))?;
    Ok(Duration::from_millis(millis))
}

fn load_mount_specs(path: &Path) -> Result<Vec<GuestMountSpec>> {
    let config_str = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read mount config {}", path.display()))?;
    let config: MountConfig = serde_yaml::from_str(&config_str)
        .with_context(|| format!("failed to parse mount config {}", path.display()))?;

    Ok(config
        .mounts
        .into_iter()
        .map(|m| GuestMountSpec::new(m.tag, m.guest_path).read_only(m.read_only))
        .collect())
}

async fn connect_with_retry(
    tag: &str,
    runtime: &RuntimeConfig,
) -> Result<tokio_vsock::VsockStream> {
    let deadline = tokio::time::Instant::now() + runtime.connect_retry_timeout;

    loop {
        let addr = tokio_vsock::VsockAddr::new(runtime.host_cid, runtime.vfs_port);
        let err = match tokio_vsock::VsockStream::connect(addr).await {
            Ok(mut stream) => {
                match motlie_vfs::vsock::write_tag_handshake(&mut stream, tag).await {
                    Ok(()) => break Ok(stream),
                    Err(e) => format!("handshake failed: {e}"),
                }
            }
            Err(e) => e.to_string(),
        };

        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "failed to connect guest mount tag '{}' to host CID {} port {} within {:?}: {}",
                tag,
                runtime.host_cid,
                runtime.vfs_port,
                runtime.connect_retry_timeout,
                err
            );
        }

        tokio::time::sleep(runtime.connect_retry_delay).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_legacy_positional_mounts_path() {
        let args = parse_args(["/tmp/mounts.yaml".to_string()]).unwrap();
        assert_eq!(args.mounts_path, PathBuf::from("/tmp/mounts.yaml"));
        assert_eq!(
            args.backend_env_path,
            Some(PathBuf::from(DEFAULT_BACKEND_ENV_PATH))
        );
    }

    #[test]
    fn parse_env_file_reads_simple_key_values() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            temp.path(),
            "\n# comment\nMOTLIE_VFS_HOST_CID=2\nMOTLIE_VFS_PORT=\"5000\"\n",
        )
        .unwrap();

        let values = parse_env_file(temp.path()).unwrap();
        assert_eq!(values["MOTLIE_VFS_HOST_CID"], "2");
        assert_eq!(values["MOTLIE_VFS_PORT"], "5000");
    }
}
