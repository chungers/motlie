//! motlie-vfs-guest-v1_1: v1.1 guest-side mounter binary.
//!
//! Reads a mount config from a YAML file, connects to the host FsServer
//! over vsock, and mounts FUSE filesystems at the configured guest paths.
//!
//! Usage: motlie-vfs-guest-v1_1 [/path/to/mounts.yaml]
//! Default config path: /etc/motlie-vfs/mounts.yaml
//!
//! Build for guest:
//!   cross build --release --target x86_64-unknown-linux-musl \
//!       -p motlie-vfs --bin motlie-vfs-guest-v1_1 --features vsock,client

use anyhow::Result;
use motlie_vfs::client::guest::{GuestMountRunner, GuestMountSpec};

/// Host CID for vsock (always 2 in the guest→host direction).
#[cfg(all(feature = "vsock", feature = "client"))]
const HOST_CID: u32 = 2;

/// Port the host FsServer listens on via the vsock socket.
#[cfg(all(feature = "vsock", feature = "client"))]
const VMM_PORT: u32 = 5000;

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
        .unwrap_or_else(|| "/etc/motlie-vfs/mounts.yaml".to_string());

    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read config {config_path}: {e}"))?;
    let config: MountConfig = serde_yaml::from_str(&config_str)
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

    let runner = GuestMountRunner::new(specs);

    // On Linux with the vsock + client features, use the real FUSE mount path.
    // Each mount connects to the host listener on port 5000, sends a tiny
    // `TAG <name>\n` prelude, and then switches to the framed FsOp/FsResult
    // request/response stream.
    #[cfg(all(feature = "vsock", feature = "client"))]
    let handles = {
        use motlie_vfs::vsock::client::VsockClientTransport;

        runner.mount_all(|tag: &str, rt: &tokio::runtime::Runtime| {
            let tag = tag.to_string();
            let stream = rt.block_on(async {
                let addr = tokio_vsock::VsockAddr::new(HOST_CID, VMM_PORT);
                let mut stream = tokio_vsock::VsockStream::connect(addr).await?;
                motlie_vfs::vsock::write_tag_handshake(&mut stream, &tag).await?;
                Ok::<_, anyhow::Error>(stream)
            })?;

            Ok(VsockClientTransport::new(stream, &tag))
        })?
    };

    #[cfg(not(all(feature = "vsock", feature = "client")))]
    let handles = {
        eprintln!("motlie-vfs-guest: vsock+client features not enabled, using stub mounts");
        runner.mount_all_stub()?
    };

    eprintln!("motlie-vfs-guest: all mounts started, waiting...");
    let results = handles.join_all();
    let mut mount_failed = false;
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(()) => {}
            Err(e) => {
                mount_failed = true;
                eprintln!("mount {} failed: {e}", i);
            }
        }
    }
    if mount_failed {
        anyhow::bail!("one or more guest mounts failed");
    }

    Ok(())
}
