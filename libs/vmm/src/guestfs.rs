use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use motlie_vfs::core::server::FsServer;
use motlie_vfs::vsock::handler::VsockConnectionHandler;
use thiserror::Error;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Instant};

use crate::spec::{GuestMountSpec, GuestSpec};

#[derive(Debug, Error)]
pub enum GuestFsError {
    #[error("guest id cannot be empty")]
    EmptyGuestId,
    #[error("guest socket path cannot be empty")]
    EmptySocketPath,
    #[error("failed to remove stale guestfs socket {path}: {source}")]
    RemoveSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to bind guestfs socket {path}: {source}")]
    BindSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create host mount path {path}: {source}")]
    CreateMountPath {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to add guestfs mount '{tag}': {reason}")]
    AddMount { tag: String, reason: String },
    #[error("timed out waiting for guestfs mount readiness for guest '{guest_id}'")]
    WaitForMounts { guest_id: String },
    #[error("failed to remove guestfs socket {path}: {source}")]
    CleanupSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("guestfs task state poisoned")]
    TaskStatePoisoned,
}

pub struct GuestFsHandle {
    guest_id: String,
    socket_path: PathBuf,
    server: Arc<FsServer>,
    required_mount_tags: Vec<String>,
    connected_mount_tags: Arc<tokio::sync::Mutex<HashSet<String>>>,
    task: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl GuestFsHandle {
    pub async fn provision(guest: &GuestSpec) -> Result<Self, GuestFsError> {
        if guest.guest_id.trim().is_empty() {
            return Err(GuestFsError::EmptyGuestId);
        }
        if guest.socket_path.as_os_str().is_empty() {
            return Err(GuestFsError::EmptySocketPath);
        }

        let server = Arc::new(
            FsServer::builder()
                .overlay(true)
                .events(256)
                .build()
                .map_err(|e| GuestFsError::AddMount {
                    tag: "<server>".to_string(),
                    reason: e.to_string(),
                })?,
        );
        let socket_path = guest.socket_path.clone();
        if let Err(source) = std::fs::remove_file(&socket_path) {
            if source.kind() != std::io::ErrorKind::NotFound {
                return Err(GuestFsError::RemoveSocket {
                    path: socket_path,
                    source,
                });
            }
        }
        let listener =
            UnixListener::bind(&socket_path).map_err(|source| GuestFsError::BindSocket {
                path: socket_path.clone(),
                source,
            })?;

        let mut required_mount_tags = Vec::new();
        for mount in &guest.mounts {
            add_mount_to_server(&server, mount, guest.user.uid, guest.user.gid)?;
            required_mount_tags.push(mount.tag.clone());
        }

        let connected_mount_tags = Arc::new(tokio::sync::Mutex::new(HashSet::new()));
        let task = spawn_guest_listener(
            guest.guest_id.clone(),
            listener,
            Arc::clone(&server),
            Arc::clone(&connected_mount_tags),
        );

        Ok(Self {
            guest_id: guest.guest_id.clone(),
            socket_path,
            server,
            required_mount_tags,
            connected_mount_tags,
            task: std::sync::Mutex::new(Some(task)),
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn required_mount_tags(&self) -> &[String] {
        &self.required_mount_tags
    }

    pub async fn wait_until_ready(&self, timeout: Duration) -> Result<(), GuestFsError> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let connected = self.connected_mount_tags.lock().await;
                let all_ready = self
                    .required_mount_tags
                    .iter()
                    .all(|tag| connected.contains(tag));
                if all_ready {
                    return Ok(());
                }
            }

            if Instant::now() >= deadline {
                return Err(GuestFsError::WaitForMounts {
                    guest_id: self.guest_id.clone(),
                });
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    pub fn has_mount(&self, tag: &str) -> bool {
        self.server.has_mount(tag)
    }

    pub fn shutdown(&self) -> Result<(), GuestFsError> {
        let mut task = self
            .task
            .lock()
            .map_err(|_| GuestFsError::TaskStatePoisoned)?;
        if let Some(task) = task.take() {
            task.abort();
        }
        if let Err(source) = std::fs::remove_file(&self.socket_path) {
            if source.kind() != std::io::ErrorKind::NotFound {
                return Err(GuestFsError::CleanupSocket {
                    path: self.socket_path.clone(),
                    source,
                });
            }
        }
        Ok(())
    }
}

fn add_mount_to_server(
    server: &Arc<FsServer>,
    mount: &GuestMountSpec,
    uid: u32,
    gid: u32,
) -> Result<(), GuestFsError> {
    std::fs::create_dir_all(&mount.host_path).map_err(|source| GuestFsError::CreateMountPath {
        path: mount.host_path.clone(),
        source,
    })?;
    server
        .add_mount_as(&mount.tag, mount.host_path.clone(), false, Some((uid, gid)))
        .map_err(|e| GuestFsError::AddMount {
            tag: mount.tag.clone(),
            reason: e.to_string(),
        })?;
    Ok(())
}

fn spawn_guest_listener(
    guest_id: String,
    listener: UnixListener,
    server: Arc<FsServer>,
    connected_mount_tags: Arc<tokio::sync::Mutex<HashSet<String>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((mut stream, _addr)) => {
                    let tag = match motlie_vfs::vsock::read_tag_handshake(&mut stream).await {
                        Ok(tag) => tag,
                        Err(e) => {
                            tracing::warn!("guestfs handshake error for {guest_id}: {e}");
                            continue;
                        }
                    };
                    if !server.has_mount(&tag) {
                        tracing::warn!("guestfs unknown tag for {guest_id}: {tag}");
                        continue;
                    }
                    {
                        let mut connected = connected_mount_tags.lock().await;
                        connected.insert(tag.clone());
                    }
                    let handler = VsockConnectionHandler::new(Arc::clone(&server), &tag);
                    let guest_id_for_conn = guest_id.clone();
                    let tag_for_conn = tag.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handler.serve(stream).await {
                            tracing::warn!(
                                "guestfs handler error for guest={guest_id_for_conn} tag={tag_for_conn}: {e}"
                            );
                        }
                    });
                    tracing::info!("accepted guestfs connection guest={guest_id} tag={tag}");
                }
                Err(e) => {
                    tracing::warn!("guestfs accept error for {guest_id}: {e}");
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{
        BootArtifacts, GuestResources, GuestSshAccess, GuestStorage, GuestUser, SoftwareProfile,
    };

    fn sample_guest(socket_path: PathBuf) -> GuestSpec {
        GuestSpec {
            guest_id: "alice".to_string(),
            hostname: "motlie-alice".to_string(),
            socket_path,
            user: GuestUser {
                name: "alice".to_string(),
                uid: 1000,
                gid: 1000,
                home: PathBuf::from("/home/alice"),
            },
            ssh: GuestSshAccess {
                principal: "alice".to_string(),
                login_user: "alice".to_string(),
            },
            mounts: vec![GuestMountSpec {
                tag: "alice-home".to_string(),
                guest_path: Some(PathBuf::from("/home/alice")),
                host_path: tempfile::tempdir().unwrap().keep(),
            }],
            software: SoftwareProfile::default(),
            resources: GuestResources::default(),
            storage: GuestStorage::default(),
            boot: BootArtifacts {
                kernel: PathBuf::from("/tmp/Image"),
                initramfs: None,
                firmware: None,
                cmdline: None,
            },
        }
    }

    #[tokio::test]
    async fn provision_registers_mounts() {
        let tempdir = tempfile::tempdir().unwrap();
        let guest = sample_guest(tempdir.path().join("alice.vsock_5000"));
        let handle = GuestFsHandle::provision(&guest).await.unwrap();

        assert!(handle.socket_path().exists());
        assert!(handle.has_mount("alice-home"));
        assert_eq!(handle.required_mount_tags(), &["alice-home".to_string()]);
    }
}
