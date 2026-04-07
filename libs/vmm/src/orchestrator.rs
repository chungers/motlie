use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::time::Duration;

use motlie_vnet::{VnetBackend, VnetConfig, VnetError, VnetHandle};
use thiserror::Error;
use tokio::time::{Instant, sleep};

use crate::artifacts::{
    ArtifactError, CloudInitArtifacts, LaunchArtifactRenderConfig, render_cloud_init_artifacts,
    render_launch_script,
};
use crate::ca::SshCa;
use crate::backend::{
    BackendError, BackendHandle, BackendKind, ChShellBackend, VmBackend,
};
use crate::guestfs::{GuestFsError, GuestFsHandle};
use crate::network::{NetworkModeError, NetworkModes, validate_network_modes};
use crate::network_alloc::{GuestNetAllocator, GuestNetAllocatorError, GuestNetAssignment};
use crate::ssh::{GuestBridgeHandle, GuestRegistry, SshProxyError, spawn_guest_ssh_bridge};
use crate::spec::{GuestRuntimePaths, GuestSpec, RuntimeNamespace, SpecError};

pub struct PrepareRequest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub network_modes: NetworkModes,
    pub backend_kind: BackendKind,
    pub base_dir: PathBuf,
    pub ssh_ca_pubkey: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PreparedGuest {
    pub guest: GuestSpec,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
    pub cloud_init: CloudInitArtifacts,
    pub launch_script: String,
    pub network_modes: NetworkModes,
    pub backend_kind: BackendKind,
    pub base_dir: PathBuf,
}

pub struct VmHandle {
    pub guest_id: String,
    pub pid: Option<u32>,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
    pub backend_kind: BackendKind,
    backend_handle: BackendHandle,
    guestfs: Option<GuestFsHandle>,
    vnet: std::sync::Mutex<Option<VnetHandle>>,
    ssh_bridge: Option<GuestBridgeHandle>,
}

#[derive(Clone)]
pub struct SshBridgeServices {
    pub ca: Arc<SshCa>,
    pub registry: GuestRegistry,
}

impl SshBridgeServices {
    pub fn new(ca: Arc<SshCa>, registry: GuestRegistry) -> Self {
        Self { ca, registry }
    }
}

#[derive(Clone, Default)]
pub struct LifecycleServices {
    pub ssh_bridge: Option<SshBridgeServices>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessStage {
    LaunchSpawned,
    ApiSocketReady,
    GuestFsConnected,
    SshBridgeReady,
    ExecReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadinessPolicy {
    pub api_socket_timeout: Duration,
    pub guestfs_timeout: Duration,
    pub ssh_bridge_timeout: Duration,
    pub exec_ready_timeout: Duration,
}

impl Default for ReadinessPolicy {
    fn default() -> Self {
        Self {
            api_socket_timeout: Duration::from_secs(10),
            guestfs_timeout: Duration::from_secs(15),
            ssh_bridge_timeout: Duration::from_secs(15),
            exec_ready_timeout: Duration::from_secs(20),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutdownReport {
    pub pid: Option<u32>,
    pub api_attempted: bool,
    pub forced: Option<&'static str>,
}

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error(transparent)]
    Spec(#[from] SpecError),
    #[error(transparent)]
    NetworkMode(#[from] NetworkModeError),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    GuestFs(#[from] GuestFsError),
    #[error(transparent)]
    Ssh(#[from] SshProxyError),
    #[error(transparent)]
    Vnet(#[from] VnetError),
    #[error(transparent)]
    NetworkAllocation(#[from] GuestNetAllocatorError),
    #[error("backend {0:?} is not implemented yet")]
    UnsupportedBackend(BackendKind),
    #[error("guest {guest_id} does not have SSH bridge services configured")]
    MissingSshBridge { guest_id: String },
    #[error("internal lifecycle state poisoned: {0}")]
    StatePoisoned(&'static str),
    #[error("guest {guest_id} exited before reaching readiness stage {stage:?}")]
    GuestExitedEarly { guest_id: String, stage: ReadinessStage },
    #[error("timed out waiting for readiness stage {stage:?} for guest {guest_id}")]
    ReadinessTimeout { guest_id: String, stage: ReadinessStage },
}

pub fn prepare(
    req: PrepareRequest,
    allocator: &mut GuestNetAllocator,
) -> Result<PreparedGuest, OrchestratorError> {
    req.guest.validate()?;
    validate_network_modes(&req.network_modes)?;

    let runtime_paths = GuestRuntimePaths::for_guest(&req.namespace, &req.guest.guest_id)?;
    let net_assignment = allocator.ensure(&req.guest.guest_id)?.clone();
    let cloud_init = render_cloud_init_artifacts(&req.guest)?;
    let launch_script = render_launch_script(&LaunchArtifactRenderConfig {
        guest: &req.guest,
        runtime_paths: &runtime_paths,
        network_modes: req.network_modes,
        net_assignment: &net_assignment,
        base_dir: &req.base_dir,
        ssh_ca_pubkey: req.ssh_ca_pubkey.as_deref(),
    })?;

    Ok(PreparedGuest {
        guest: req.guest,
        runtime_paths,
        net_assignment,
        cloud_init,
        launch_script,
        network_modes: req.network_modes,
        backend_kind: req.backend_kind,
        base_dir: req.base_dir,
    })
}

pub async fn boot(
    prepared: PreparedGuest,
    services: LifecycleServices,
) -> Result<VmHandle, OrchestratorError> {
    let guestfs = if prepared.guest.mounts.is_empty() {
        None
    } else {
        Some(GuestFsHandle::provision(&prepared.guest).await?)
    };

    let mut vnet = if matches!(prepared.network_modes.egress, crate::network::EgressNetMode::VhostUser) {
        Some(start_vnet_backend(&prepared)?)
    } else {
        None
    };

    let ssh_bridge = if let Some(ssh) = services.ssh_bridge {
        Some(spawn_guest_ssh_bridge(
            prepared.runtime_paths.vsock_socket.to_string_lossy().as_ref(),
            Arc::clone(&ssh.ca),
            prepared.guest.guest_id.clone(),
            Arc::clone(&ssh.registry),
        )?)
    } else {
        None
    };

    let backend_handle = match prepared.backend_kind {
        BackendKind::ChShell => match ChShellBackend::new().boot(&prepared) {
            Ok(handle) => handle,
            Err(err) => {
                if let Some(bridge) = ssh_bridge.as_ref() {
                    let _ = bridge.shutdown();
                }
                if let Some(vnet) = vnet.as_mut() {
                    let _ = vnet.shutdown();
                }
                if let Some(guestfs) = guestfs.as_ref() {
                    let _ = guestfs.shutdown();
                }
                return Err(err.into());
            }
        },
        kind => return Err(OrchestratorError::UnsupportedBackend(kind)),
    };

    Ok(VmHandle {
        guest_id: prepared.guest.guest_id.clone(),
        pid: backend_handle.pid(),
        runtime_paths: prepared.runtime_paths,
        net_assignment: prepared.net_assignment,
        backend_kind: prepared.backend_kind,
        backend_handle,
        guestfs,
        vnet: std::sync::Mutex::new(vnet),
        ssh_bridge,
    })
}

impl VmHandle {
    pub async fn ready(&self, policy: &ReadinessPolicy) -> Result<(), OrchestratorError> {
        self.wait_for_path(
            &self.runtime_paths.api_socket,
            ReadinessStage::ApiSocketReady,
            policy.api_socket_timeout,
        )
        .await?;

        if let Some(guestfs) = self.guestfs.as_ref() {
            guestfs.wait_until_ready(policy.guestfs_timeout).await?;
        }

        if let Some(bridge) = self.ssh_bridge.as_ref() {
            bridge.wait_ready(policy.ssh_bridge_timeout).await?;
            self.exec("/bin/true", policy.exec_ready_timeout).await?;
        }

        Ok(())
    }

    pub async fn exec(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<crate::ssh::ExecOutput, OrchestratorError> {
        let bridge = self
            .ssh_bridge
            .as_ref()
            .ok_or_else(|| OrchestratorError::MissingSshBridge {
                guest_id: self.guest_id.clone(),
            })?;
        Ok(bridge.exec(command, timeout).await?)
    }

    pub async fn shutdown(&self) -> Result<ShutdownReport, OrchestratorError> {
        let backend_result = match self.backend_kind {
            BackendKind::ChShell => ChShellBackend::new().shutdown(&self.backend_handle)?,
            kind => return Err(OrchestratorError::UnsupportedBackend(kind)),
        };

        if let Some(bridge) = self.ssh_bridge.as_ref() {
            bridge.shutdown()?;
        }
        if let Some(mut vnet) = self
            .vnet
            .lock()
            .map_err(|_| OrchestratorError::StatePoisoned("vnet handle"))?
            .take()
        {
            vnet.shutdown()?;
        }
        if let Some(guestfs) = self.guestfs.as_ref() {
            guestfs.shutdown()?;
        }

        Ok(ShutdownReport {
            pid: self.pid,
            api_attempted: backend_result.api_attempted,
            forced: backend_result.forced,
        })
    }

    async fn wait_for_path(
        &self,
        path: &Path,
        stage: ReadinessStage,
        timeout: Duration,
    ) -> Result<(), OrchestratorError> {
        let deadline = Instant::now() + timeout;
        loop {
            if path.exists() {
                return Ok(());
            }
            if let Some(pid) = self.pid {
                if !PathBuf::from(format!("/proc/{pid}")).exists() {
                    return Err(OrchestratorError::GuestExitedEarly {
                        guest_id: self.guest_id.clone(),
                        stage,
                    });
                }
            }
            if Instant::now() >= deadline {
                return Err(OrchestratorError::ReadinessTimeout {
                    guest_id: self.guest_id.clone(),
                    stage,
                });
            }
            sleep(Duration::from_millis(100)).await;
        }
    }
}

fn start_vnet_backend(prepared: &PreparedGuest) -> Result<VnetHandle, OrchestratorError> {
    if let Some(parent) = prepared.runtime_paths.vnet_socket.parent() {
        std::fs::create_dir_all(parent).map_err(|source| {
            OrchestratorError::Backend(BackendError::CreateRuntimeDir {
                path: parent.to_path_buf(),
                source,
            })
        })?;
    }

    let config = VnetConfig::builder()
        .socket_path(&prepared.runtime_paths.vnet_socket)
        .guest_ipv4(prepared.net_assignment.egress_ipv4.guest)
        .host_ipv4(prepared.net_assignment.egress_ipv4.host)
        .netmask(prepared.net_assignment.egress_ipv4.netmask)
        .dns_ipv4(prepared.net_assignment.egress_ipv4.dns)
        .mac(prepared.net_assignment.egress_mac)
        .build()?;
    Ok(VnetBackend::new(config).start()?)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::network::{AdminNetMode, EgressNetMode};
    use crate::network_alloc::GuestNetAllocatorConfig;
    use crate::spec::{
        BootArtifacts, GuestMountSpec, GuestResources, GuestSshAccess, GuestStorage, GuestUser,
        SoftwareProfile,
    };

    use super::*;

    fn sample_guest() -> GuestSpec {
        GuestSpec {
            guest_id: "alice".to_string(),
            hostname: "motlie-alice".to_string(),
            socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".to_string(),
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
                host_path: PathBuf::from("/tmp/demo/alice-home"),
            }],
            software: SoftwareProfile {
                packages: vec!["vim".to_string()],
            },
            resources: GuestResources::default(),
            storage: GuestStorage::default(),
            boot: BootArtifacts {
                kernel: PathBuf::from("/tmp/Image"),
                initramfs: None,
                firmware: None,
                cmdline: Some("console=ttyS0".to_string()),
            },
        }
    }

    #[test]
    fn prepare_materializes_reviewed_guest_inputs() {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp").unwrap();
        let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
        let prepared = prepare(
            PrepareRequest {
                guest: sample_guest(),
                namespace,
                network_modes: NetworkModes {
                    admin: AdminNetMode::None,
                    egress: EgressNetMode::VhostUser,
                },
                backend_kind: BackendKind::ChShell,
                base_dir: PathBuf::from("/tmp/vmm-v1.4/libs/vmm/examples/v1.4"),
                ssh_ca_pubkey: Some("ssh-ed25519 AAAA-test".to_string()),
            },
            &mut allocator,
        )
        .unwrap();

        assert_eq!(prepared.guest.guest_id, "alice");
        assert_eq!(prepared.net_assignment.cid, 3);
        assert!(prepared.cloud_init.user_data.contains("packages:"));
        assert!(prepared.launch_script.contains("OVERLAY_SIZE='2G'"));
        assert!(prepared.launch_script.contains("BOOT_KERNEL='/tmp/Image'"));
    }

    #[tokio::test]
    async fn ready_waits_for_api_socket() {
        let tempdir = tempfile::tempdir().unwrap();
        let api_socket = tempdir.path().join("guest-api.sock");
        let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
        let handle = VmHandle {
            guest_id: "alice".to_string(),
            pid: None,
            runtime_paths: GuestRuntimePaths {
                runtime_dir: tempdir.path().join("runtime"),
                launch_dir: tempdir.path().join("launch"),
                cloud_init_dir: tempdir.path().join("cloud-init"),
                api_socket: api_socket.clone(),
                vnet_socket: tempdir.path().join("vnet.sock"),
                vsock_socket: tempdir.path().join("vsock"),
                serial_log: tempdir.path().join("serial.log"),
                launch_log: tempdir.path().join("launch.log"),
            },
            net_assignment: allocator.ensure("alice").unwrap().clone(),
            backend_kind: BackendKind::ChShell,
            backend_handle: BackendHandle::ChShell(crate::backend::ChShellHandle {
                pid: None,
                launch_script_path: tempdir.path().join("launch.sh"),
                api_socket: api_socket.clone(),
            }),
            guestfs: None,
            vnet: std::sync::Mutex::new(None),
            ssh_bridge: None,
        };

        tokio::spawn({
            let api_socket = api_socket.clone();
            async move {
                sleep(Duration::from_millis(50)).await;
                std::fs::write(api_socket, []).unwrap();
            }
        });

        handle
            .ready(&ReadinessPolicy {
                api_socket_timeout: Duration::from_secs(1),
                guestfs_timeout: Duration::from_secs(1),
                ssh_bridge_timeout: Duration::from_secs(1),
                exec_ready_timeout: Duration::from_secs(1),
            })
            .await
            .unwrap();
    }
}
