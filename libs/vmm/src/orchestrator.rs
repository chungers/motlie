use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use thiserror::Error;
use tokio::time::{Instant, sleep};

use crate::artifacts::{
    ArtifactError, CloudInitArtifacts, LaunchArtifactRenderConfig, render_cloud_init_artifacts,
    render_launch_script,
};
use crate::backend::BackendHandle;
use crate::network::{NetworkModeError, NetworkModes, validate_network_modes};
use crate::network_alloc::{GuestNetAllocator, GuestNetAllocatorError, GuestNetAssignment};
use crate::observability::{
    ControlPlaneObservability, FilesystemObservability, NetworkObservability, VmArtifactKind,
    VmCapturePaths, VmHostMount, VmObservability, VmRunArtifact, VmRunBundle,
    VmRuntimePaths as VmRuntimePathsView,
};
use crate::runtime::{
    ControlPlaneHandle, FilesystemHandle, NetworkHandle, Runtime, RuntimeError, SharedRuntime,
};
use crate::spec::{GuestMountSpec, GuestRuntimePaths, GuestSpec, RuntimeNamespace, SpecError};

pub struct PrepareRequest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub network_modes: NetworkModes,
    pub base_dir: PathBuf,
    pub ssh_ca_pubkey: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PreparedGuest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub runtime_paths: GuestRuntimePaths,
    pub guest_socket_path: PathBuf,
    pub net_assignment: GuestNetAssignment,
    pub cloud_init: CloudInitArtifacts,
    pub launch_script: String,
    pub network_modes: NetworkModes,
    pub base_dir: PathBuf,
}

pub struct VmHandle {
    pub guest_id: String,
    pub pid: Option<u32>,
    namespace: RuntimeNamespace,
    pub runtime_paths: GuestRuntimePaths,
    guest_socket_path: PathBuf,
    guest_mounts: Vec<GuestMountSpec>,
    pub net_assignment: GuestNetAssignment,
    runtime: SharedRuntime,
    backend_handle: BackendHandle,
    filesystem: Option<FilesystemHandle>,
    network: std::sync::Mutex<Option<NetworkHandle>>,
    control_plane: Option<ControlPlaneHandle>,
}

#[derive(Clone)]
pub struct LifecycleServices {
    pub runtime: SharedRuntime,
}

impl Default for LifecycleServices {
    fn default() -> Self {
        Self {
            runtime: Arc::new(Runtime::simple_cloud_hypervisor_shell()),
        }
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
    Backend(#[from] crate::backend::BackendError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    NetworkAllocation(#[from] GuestNetAllocatorError),
    #[error("guest {guest_id} does not have SSH bridge services configured")]
    MissingSshBridge { guest_id: String },
    #[error("internal lifecycle state poisoned: {0}")]
    StatePoisoned(&'static str),
    #[error("guest {guest_id} exited before reaching readiness stage {stage:?}")]
    GuestExitedEarly {
        guest_id: String,
        stage: ReadinessStage,
    },
    #[error("timed out waiting for readiness stage {stage:?} for guest {guest_id}")]
    ReadinessTimeout {
        guest_id: String,
        stage: ReadinessStage,
    },
}

pub fn prepare(
    req: PrepareRequest,
    allocator: &mut GuestNetAllocator,
) -> Result<PreparedGuest, OrchestratorError> {
    req.guest.validate()?;
    validate_network_modes(&req.network_modes)?;

    let runtime_paths = GuestRuntimePaths::for_guest(&req.namespace, &req.guest.guest_id)?;
    let guest_socket_path = PathBuf::from(req.guest.socket_path.clone());
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
        namespace: req.namespace,
        runtime_paths,
        guest_socket_path,
        net_assignment,
        cloud_init,
        launch_script,
        network_modes: req.network_modes,
        base_dir: req.base_dir,
    })
}

pub async fn boot(
    prepared: PreparedGuest,
    services: LifecycleServices,
) -> Result<VmHandle, OrchestratorError> {
    let filesystem = services
        .runtime
        .filesystem
        .provision(&prepared.guest)
        .await?;
    let mut network = services.runtime.network.provision(&prepared)?;
    let control_plane = services.runtime.control_plane.provision(&prepared)?;

    let backend_handle = match services.runtime.hypervisor.boot(&prepared) {
        Ok(handle) => handle,
        Err(err) => {
            if let Some(control_plane) = control_plane.as_ref() {
                let _ = control_plane.shutdown();
            }
            if let Some(network) = network.as_mut() {
                let _ = network.shutdown();
            }
            if let Some(filesystem) = filesystem.as_ref() {
                let _ = filesystem.shutdown();
            }
            return Err(err.into());
        }
    };

    Ok(VmHandle {
        guest_id: prepared.guest.guest_id.clone(),
        pid: backend_handle.pid(),
        namespace: prepared.namespace,
        runtime_paths: prepared.runtime_paths,
        guest_socket_path: prepared.guest_socket_path,
        guest_mounts: prepared.guest.mounts,
        net_assignment: prepared.net_assignment,
        runtime: Arc::clone(&services.runtime),
        backend_handle,
        filesystem,
        network: std::sync::Mutex::new(network),
        control_plane,
    })
}

impl VmHandle {
    pub fn observability(&self) -> VmObservability {
        let filesystem_observability = match self.filesystem.as_ref() {
            Some(filesystem) => FilesystemObservability {
                backing: filesystem.backing_name(),
                socket_path: filesystem.socket_path().map(Path::to_path_buf),
                mount_tags: filesystem.mount_tags(),
            },
            None => FilesystemObservability {
                backing: "none",
                socket_path: None,
                mount_tags: Vec::new(),
            },
        };
        let network_observability = match self
            .network
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|n| n.backing_name()))
        {
            Some(backing) => NetworkObservability {
                backing,
                socket_path: Some(self.runtime_paths.vnet_socket.clone()),
            },
            None => NetworkObservability {
                backing: "none",
                socket_path: None,
            },
        };
        let control_plane_observability = match self.control_plane.as_ref() {
            Some(control_plane) => ControlPlaneObservability {
                backing: control_plane.backing_name(),
                ssh_bridge_socket_path: control_plane.bridge_socket_path(),
            },
            None => ControlPlaneObservability {
                backing: "none",
                ssh_bridge_socket_path: None,
            },
        };
        let bundle_root = self.runtime_paths.runtime_dir.join("bundle");
        let host_mounts: Vec<VmHostMount> = self
            .guest_mounts
            .iter()
            .map(|mount| VmHostMount {
                tag: mount.tag.clone(),
                host_path: mount.host_path.clone(),
                guest_path: mount.guest_path.clone(),
                exists: mount.host_path.exists(),
            })
            .collect();
        let mut artifacts = vec![
            VmRunArtifact {
                kind: VmArtifactKind::RuntimeDir,
                label: "runtime_dir".to_string(),
                path: self.runtime_paths.runtime_dir.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.runtime_dir.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::LaunchDir,
                label: "launch_dir".to_string(),
                path: self.runtime_paths.launch_dir.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.launch_dir.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::CloudInitDir,
                label: "cloud_init_dir".to_string(),
                path: self.runtime_paths.cloud_init_dir.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.cloud_init_dir.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::ApiSocket,
                label: "api_socket".to_string(),
                path: self.runtime_paths.api_socket.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.api_socket.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::VnetSocket,
                label: "vnet_socket".to_string(),
                path: self.runtime_paths.vnet_socket.clone(),
                guest_path: None,
                required: network_observability.socket_path.is_some(),
                exists: self.runtime_paths.vnet_socket.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::VsockSocket,
                label: "vsock_socket".to_string(),
                path: self.runtime_paths.vsock_socket.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.vsock_socket.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::GuestSocket,
                label: "guest_socket".to_string(),
                path: self.guest_socket_path.clone(),
                guest_path: None,
                required: true,
                exists: self.guest_socket_path.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::SerialLog,
                label: "serial_log".to_string(),
                path: self.runtime_paths.serial_log.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.serial_log.exists(),
            },
            VmRunArtifact {
                kind: VmArtifactKind::LaunchLog,
                label: "launch_log".to_string(),
                path: self.runtime_paths.launch_log.clone(),
                guest_path: None,
                required: true,
                exists: self.runtime_paths.launch_log.exists(),
            },
        ];
        if let Some(socket_path) = filesystem_observability.socket_path.clone() {
            artifacts.push(VmRunArtifact {
                kind: VmArtifactKind::FilesystemSocket,
                label: "filesystem_socket".to_string(),
                exists: socket_path.exists(),
                path: socket_path,
                guest_path: None,
                required: true,
            });
        }
        if let Some(socket_path) = control_plane_observability.ssh_bridge_socket_path.clone() {
            artifacts.push(VmRunArtifact {
                kind: VmArtifactKind::SshBridgeSocket,
                label: "ssh_bridge_socket".to_string(),
                exists: socket_path.exists(),
                path: socket_path,
                guest_path: None,
                required: true,
            });
        }
        artifacts.extend(host_mounts.iter().map(|mount| VmRunArtifact {
            kind: VmArtifactKind::HostMount,
            label: mount.tag.clone(),
            path: mount.host_path.clone(),
            guest_path: mount.guest_path.clone(),
            required: true,
            exists: mount.exists,
        }));

        VmObservability {
            guest_id: self.guest_id.clone(),
            pid: self.pid,
            namespace_prefix: self.namespace.prefix.clone(),
            temp_root: self.namespace.temp_root.clone(),
            guest_socket_path: self.guest_socket_path.clone(),
            runtime_paths: VmRuntimePathsView {
                runtime_dir: self.runtime_paths.runtime_dir.clone(),
                launch_dir: self.runtime_paths.launch_dir.clone(),
                cloud_init_dir: self.runtime_paths.cloud_init_dir.clone(),
                api_socket: self.runtime_paths.api_socket.clone(),
                vnet_socket: self.runtime_paths.vnet_socket.clone(),
                vsock_socket: self.runtime_paths.vsock_socket.clone(),
                serial_log: self.runtime_paths.serial_log.clone(),
                launch_log: self.runtime_paths.launch_log.clone(),
            },
            filesystem: filesystem_observability,
            network: network_observability,
            control_plane: control_plane_observability,
            run_bundle: VmRunBundle {
                capture_paths: VmCapturePaths {
                    scenario_result_json: bundle_root.join("scenario-result.json"),
                    pty_transcript_ndjson: bundle_root.join("pty-transcript.ndjson"),
                    pty_screen_json: bundle_root.join("pty-screen.json"),
                    pty_screen_svg: bundle_root.join("pty-screen.svg"),
                    pty_asciicast: bundle_root.join("pty.cast"),
                },
                bundle_root,
                host_mounts,
                artifacts,
            },
        }
    }

    pub async fn ready(&self, policy: &ReadinessPolicy) -> Result<(), OrchestratorError> {
        self.wait_for_path(
            &self.runtime_paths.api_socket,
            ReadinessStage::ApiSocketReady,
            policy.api_socket_timeout,
        )
        .await?;

        if let Some(filesystem) = self.filesystem.as_ref() {
            filesystem.wait_ready(policy.guestfs_timeout).await?;
        }

        if let Some(control_plane) = self.control_plane.as_ref() {
            control_plane.wait_ready(policy.ssh_bridge_timeout).await?;
            self.exec("/bin/true", policy.exec_ready_timeout).await?;
        }

        Ok(())
    }

    pub async fn exec(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<crate::ssh::ExecOutput, OrchestratorError> {
        let control_plane =
            self.control_plane
                .as_ref()
                .ok_or_else(|| OrchestratorError::MissingSshBridge {
                    guest_id: self.guest_id.clone(),
                })?;
        Ok(control_plane.exec(command, timeout).await?)
    }

    pub async fn open_pty(
        &self,
        request: crate::ssh::PtyRequest,
        timeout: Duration,
    ) -> Result<crate::ssh::GuestPtySession, OrchestratorError> {
        let control_plane =
            self.control_plane
                .as_ref()
                .ok_or_else(|| OrchestratorError::MissingSshBridge {
                    guest_id: self.guest_id.clone(),
                })?;
        Ok(control_plane.open_pty(request, timeout).await?)
    }

    pub async fn shutdown(&self) -> Result<ShutdownReport, OrchestratorError> {
        let backend_result = self.runtime.hypervisor.shutdown(&self.backend_handle)?;

        if let Some(control_plane) = self.control_plane.as_ref() {
            control_plane.shutdown()?;
        }
        if let Some(mut network) = self
            .network
            .lock()
            .map_err(|_| OrchestratorError::StatePoisoned("network handle"))?
            .take()
        {
            network.shutdown()?;
        }
        if let Some(filesystem) = self.filesystem.as_ref() {
            filesystem.shutdown()?;
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
            if self.backend_handle.has_exited()? {
                return Err(OrchestratorError::GuestExitedEarly {
                    guest_id: self.guest_id.clone(),
                    stage,
                });
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::backend::ch::shell::ChShellHandle;
    use crate::network::{AdminNetMode, EgressNetMode};
    use crate::network_alloc::GuestNetAllocatorConfig;
    use crate::runtime::{HypervisorBacking, Runtime};
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
        let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig::default()).unwrap();
        let prepared = prepare(
            PrepareRequest {
                guest: sample_guest(),
                namespace,
                network_modes: NetworkModes {
                    admin: AdminNetMode::None,
                    egress: EgressNetMode::VhostUser,
                },
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
        let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig::default()).unwrap();
        let handle = VmHandle {
            guest_id: "alice".to_string(),
            pid: None,
            namespace: RuntimeNamespace::new("motlie-vmm-v14-test", tempdir.path()).unwrap(),
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
            guest_socket_path: tempdir.path().join("guest.vsock_5000"),
            guest_mounts: vec![],
            net_assignment: allocator.ensure("alice").unwrap().clone(),
            runtime: Arc::new(Runtime {
                hypervisor: HypervisorBacking::CloudHypervisorShell(
                    crate::backend::ch::shell::ChShellBackend::new(),
                ),
                ..Runtime::simple_cloud_hypervisor_shell()
            }),
            backend_handle: BackendHandle::ChShell(ChShellHandle {
                pid: None,
                launch_script_path: tempdir.path().join("launch.sh"),
                api_socket: api_socket.clone(),
                child: std::sync::Mutex::new(None),
            }),
            filesystem: None,
            network: std::sync::Mutex::new(None),
            control_plane: None,
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
