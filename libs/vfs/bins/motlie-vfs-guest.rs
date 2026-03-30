//! motlie-vfs-guest: v1 guest-side mounter binary.
//!
//! Reads a mount config from a JSON file, uses GuestMountRunner to mount
//! all specified guest paths. VMM handshake and stream acquisition are
//! caller-provided — this binary demonstrates the library guest API.
//!
//! Usage: motlie-vfs-guest [/path/to/mounts.toml]
//! Default config path: /etc/motlie-vfs/mounts.toml

use anyhow::Result;
use motlie_vfs::client::guest::{GuestMountRunner, GuestMountSpec};

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

fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/etc/motlie-vfs/mounts.toml".to_string());

    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read config {config_path}: {e}"))?;
    let config: MountConfig = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("failed to parse config {config_path}: {e}"))?;

    let specs: Vec<GuestMountSpec> = config
        .mounts
        .into_iter()
        .map(|m| GuestMountSpec::new(m.tag, m.guest_path).read_only(m.read_only))
        .collect();

    eprintln!("motlie-vfs-guest: mounting {} filesystem(s)", specs.len());
    for spec in &specs {
        eprintln!("  tag={} path={} ro={}", spec.tag, spec.guest_path, spec.read_only);
    }

    // In production, mount_all() with a real vsock connector would be used here.
    // For now, use the stub which creates mount point directories but doesn't
    // actually mount FUSE filesystems.
    let runner = GuestMountRunner::new(specs);
    let handles = runner.mount_all_stub()?;

    let results = handles.join_all();
    for (i, result) in results.iter().enumerate() {
        if let Err(e) = result {
            eprintln!("mount {} failed: {e}", i);
        }
    }

    Ok(())
}
