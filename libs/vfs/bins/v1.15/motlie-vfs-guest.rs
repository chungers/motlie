//! motlie-vfs-guest-v1_15: Apple Vz guest-side mounter binary.
//!
//! Reads a mount config from YAML, connects to the host FsServer over
//! virtio-vsock, sends the same `TAG <name>\n` handshake used by the CH demo,
//! and mounts FUSE filesystems at the configured guest paths.

use std::time::Duration;

use anyhow::Result;
use motlie_vfs::client::guest::{GuestMountRunner, GuestMountSpec};

const CONNECT_RETRY_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_RETRY_DELAY: Duration = Duration::from_millis(250);
#[cfg(all(feature = "vsock", feature = "client"))]
const HOST_CID: u32 = 2;
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

    eprintln!("motlie-vfs-guest-v1_15: mounting {} filesystem(s)", specs.len());
    for spec in &specs {
        eprintln!(
            "  tag={} path={} ro={}",
            spec.tag, spec.guest_path, spec.read_only
        );
    }

    let runner = GuestMountRunner::new(specs);

    #[cfg(all(feature = "vsock", feature = "client"))]
    let handles = {
        use motlie_vfs::vsock::client::VsockClientTransport;

        runner.mount_all(|tag: &str, rt: &tokio::runtime::Runtime| {
            let tag = tag.to_string();
            let stream = rt.block_on(async {
                let deadline = tokio::time::Instant::now() + CONNECT_RETRY_TIMEOUT;

                loop {
                    let addr = tokio_vsock::VsockAddr::new(HOST_CID, VMM_PORT);
                    let err = match tokio_vsock::VsockStream::connect(addr).await {
                        Ok(mut stream) => {
                            match motlie_vfs::vsock::write_tag_handshake(&mut stream, &tag).await {
                                Ok(()) => break Ok::<_, anyhow::Error>(stream),
                                Err(e) => format!("handshake failed: {e}"),
                            }
                        }
                        Err(e) => e.to_string(),
                    };

                    if tokio::time::Instant::now() >= deadline {
                        break Err(anyhow::anyhow!(
                            "failed to connect guest mount tag '{}' to host CID {} port {} within {:?}: {}",
                            tag,
                            HOST_CID,
                            VMM_PORT,
                            CONNECT_RETRY_TIMEOUT,
                            err
                        ));
                    }

                    tokio::time::sleep(CONNECT_RETRY_DELAY).await;
                }
            })?;

            Ok(VsockClientTransport::new(stream, &tag))
        })?
    };

    #[cfg(not(all(feature = "vsock", feature = "client")))]
    let handles = {
        eprintln!("motlie-vfs-guest-v1_15: vsock+client features not enabled, using stub mounts");
        runner.mount_all_stub()?
    };

    eprintln!("motlie-vfs-guest-v1_15: all mounts started, waiting...");
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
