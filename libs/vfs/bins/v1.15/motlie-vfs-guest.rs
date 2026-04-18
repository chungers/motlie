//! motlie-vfs-guest-v1_15: Apple Vz guest-side mounter binary.
//!
//! Reads a mount config from YAML, connects to the host FsServer over a TCP
//! stream, sends the same `TAG <name>\n` handshake used by the CH demo, and
//! mounts FUSE filesystems at the configured guest paths.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use motlie_vfs::client::fuse::{v1_mount_options, FuseClient};
use motlie_vfs::vz::client::TcpClientTransport;

const CONNECT_RETRY_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_RETRY_DELAY: Duration = Duration::from_millis(250);

#[derive(serde::Deserialize)]
struct MountConfig {
    server: ServerConfig,
    mounts: Vec<MountEntry>,
}

#[derive(serde::Deserialize)]
struct ServerConfig {
    host: String,
    port: u16,
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

    eprintln!(
        "motlie-vfs-guest-v1_15: mounting {} filesystem(s) via {}:{}",
        config.mounts.len(),
        config.server.host,
        config.server.port
    );
    for mount in &config.mounts {
        eprintln!(
            "  tag={} path={} ro={}",
            mount.tag, mount.guest_path, mount.read_only
        );
    }

    let mut handles = Vec::new();
    for mount in config.mounts {
        let server_host = config.server.host.clone();
        let server_port = config.server.port;
        let tag = mount.tag.clone();
        let guest_path = mount.guest_path.clone();
        let read_only = mount.read_only;

        let handle = std::thread::Builder::new()
            .name(format!("fuse-{tag}"))
            .spawn(move || -> Result<()> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;

                let stream = rt.block_on(async {
                    let deadline = tokio::time::Instant::now() + CONNECT_RETRY_TIMEOUT;
                    loop {
                        let addr = format!("{server_host}:{server_port}");
                        let err = match tokio::net::TcpStream::connect(&addr).await {
                            Ok(mut stream) => {
                                match motlie_vfs::vz::write_tag_handshake(&mut stream, &tag).await {
                                    Ok(()) => break Ok::<_, anyhow::Error>(stream),
                                    Err(e) => format!("handshake failed: {e}"),
                                }
                            }
                            Err(e) => e.to_string(),
                        };

                        if tokio::time::Instant::now() >= deadline {
                            break Err(anyhow::anyhow!(
                                "failed to connect guest mount tag '{}' to host {}:{} within {:?}: {}",
                                tag,
                                server_host,
                                server_port,
                                CONNECT_RETRY_TIMEOUT,
                                err
                            ));
                        }

                        tokio::time::sleep(CONNECT_RETRY_DELAY).await;
                    }
                })?;

                let transport = Arc::new(TcpClientTransport::new(stream, &tag));
                std::fs::create_dir_all(&guest_path)?;

                let transport_for_fuse = Arc::clone(&transport);
                let request_fn = move |op: motlie_vfs::core::op::FsOp| -> motlie_vfs::core::op::FsResult {
                    let transport = Arc::clone(&transport_for_fuse);
                    match rt.block_on(transport.request(&op)) {
                        Ok(result) => result,
                        Err(_) => motlie_vfs::core::op::FsResult::Error { errno: libc::EIO },
                    }
                };

                let fuse_client = FuseClient::new(request_fn);
                let opts = v1_mount_options(read_only);
                fuser::mount2(fuse_client, &guest_path, &opts)?;

                Ok(())
            })?;
        handles.push(handle);
    }

    eprintln!("motlie-vfs-guest-v1_15: all mounts started, waiting...");
    let mut mount_failed = false;
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                mount_failed = true;
                eprintln!("mount {i} failed: {e}");
            }
            Err(_) => {
                mount_failed = true;
                eprintln!("mount {i} panicked");
            }
        }
    }

    if mount_failed {
        anyhow::bail!("one or more guest mounts failed");
    }

    Ok(())
}
