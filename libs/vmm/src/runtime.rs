use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::backend::ch::shell::ChShellBackend;
use crate::backend::motlie::ssh_proxy::{MotlieSshProxyBacking, MotlieSshProxyHandle};
use crate::backend::motlie::vfs::{MotlieVfsBacking, MotlieVfsHandle};
#[cfg(target_os = "linux")]
use crate::backend::motlie::vnet::{MotlieVnetBacking, MotlieVnetHandle, MotlieVnetProvisionError};
use crate::backend::vz::shell::VzShellBackend;
use crate::backend::{BackendError, BackendHandle, BackendShutdownOutcome};
use crate::guestfs::GuestFsError;
use crate::observability::{
    ControlPlaneObservability, FilesystemObservability, NetworkObservability,
};
use crate::orchestrator::PreparedGuest;
use crate::spec::GuestRuntimePaths;
use crate::spec::GuestSpec;
use crate::ssh::{ExecOutput, GuestPtySession, PtyRequest, SshProxyError};
#[cfg(target_os = "linux")]
use motlie_vnet::VnetError;

#[derive(Debug, Clone)]
pub struct Runtime {
    pub hypervisor: HypervisorBacking,
    pub filesystem: FilesystemBacking,
    pub network: NetworkBacking,
    pub control_plane: ControlPlaneBacking,
}

impl Runtime {
    pub fn simple_cloud_hypervisor_shell() -> Self {
        Self {
            hypervisor: HypervisorBacking::CloudHypervisorShell(ChShellBackend::new()),
            filesystem: FilesystemBacking::HypervisorManaged,
            network: NetworkBacking::HypervisorManaged,
            control_plane: ControlPlaneBacking::None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum HypervisorBacking {
    CloudHypervisorShell(ChShellBackend),
    CloudHypervisorForkExec,
    CloudHypervisorVmmThread,
    AppleVirtualizationShell(VzShellBackend),
    AppleVirtualization,
}

#[derive(Debug, Clone)]
pub enum FilesystemBacking {
    HypervisorManaged,
    MotlieVfs(MotlieVfsBacking),
}

#[derive(Debug, Clone)]
pub enum NetworkBacking {
    None,
    HypervisorManaged,
    #[cfg(target_os = "linux")]
    MotlieVnet(MotlieVnetBacking),
    #[cfg(target_os = "linux")]
    HypervisorManagedPlusMotlieVnet(MotlieVnetBacking),
}

#[derive(Debug, Clone)]
pub enum ControlPlaneBacking {
    None,
    MotlieSshProxy(MotlieSshProxyBacking),
}

#[derive(Debug)]
pub enum FilesystemHandle {
    MotlieVfs(MotlieVfsHandle),
}

#[derive(Debug)]
pub enum NetworkHandle {
    #[cfg(target_os = "linux")]
    MotlieVnet(MotlieVnetHandle),
}

#[derive(Debug)]
pub enum ControlPlaneHandle {
    MotlieSshProxy(MotlieSshProxyHandle),
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    GuestFs(#[from] GuestFsError),
    #[error(transparent)]
    Ssh(#[from] SshProxyError),
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Vnet(#[from] MotlieVnetProvisionError),
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    VnetShutdown(#[from] VnetError),
    #[error("hypervisor backing is not implemented yet")]
    UnsupportedHypervisor,
}

impl HypervisorBacking {
    pub fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, RuntimeError> {
        match self {
            Self::CloudHypervisorShell(backend) => Ok(backend.boot(prepared)?),
            Self::AppleVirtualizationShell(backend) => Ok(backend.boot(prepared)?),
            Self::CloudHypervisorForkExec
            | Self::CloudHypervisorVmmThread
            | Self::AppleVirtualization => Err(RuntimeError::UnsupportedHypervisor),
        }
    }

    pub fn shutdown(&self, handle: &BackendHandle) -> Result<BackendShutdownOutcome, RuntimeError> {
        match self {
            Self::CloudHypervisorShell(backend) => Ok(backend.shutdown(handle)?),
            Self::AppleVirtualizationShell(backend) => Ok(backend.shutdown(handle)?),
            Self::CloudHypervisorForkExec
            | Self::CloudHypervisorVmmThread
            | Self::AppleVirtualization => Err(RuntimeError::UnsupportedHypervisor),
        }
    }
}

impl FilesystemBacking {
    pub async fn provision(
        &self,
        guest: &GuestSpec,
    ) -> Result<Option<FilesystemHandle>, RuntimeError> {
        match self {
            Self::HypervisorManaged => Ok(None),
            Self::MotlieVfs(backing) => Ok(backing
                .provision(guest)
                .await?
                .map(FilesystemHandle::MotlieVfs)),
        }
    }
}

impl FilesystemHandle {
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), RuntimeError> {
        match self {
            Self::MotlieVfs(handle) => Ok(handle.wait_ready(timeout).await?),
        }
    }

    pub fn shutdown(&self) -> Result<(), RuntimeError> {
        match self {
            Self::MotlieVfs(handle) => Ok(handle.shutdown()?),
        }
    }

    pub fn backing_name(&self) -> &'static str {
        match self {
            Self::MotlieVfs(_) => "motlie-vfs",
        }
    }

    pub fn socket_path(&self) -> Option<&Path> {
        match self {
            Self::MotlieVfs(handle) => Some(handle.socket_path()),
        }
    }

    pub fn mount_tags(&self) -> Vec<String> {
        match self {
            Self::MotlieVfs(handle) => handle.required_mount_tags().to_vec(),
        }
    }

    pub fn observability(&self) -> FilesystemObservability {
        FilesystemObservability {
            backing: self.backing_name(),
            socket_path: self.socket_path().map(Path::to_path_buf),
            mount_tags: self.mount_tags(),
        }
    }
}

impl NetworkBacking {
    pub fn provision(
        &self,
        _prepared: &PreparedGuest,
    ) -> Result<Option<NetworkHandle>, RuntimeError> {
        match self {
            Self::None | Self::HypervisorManaged => Ok(None),
            #[cfg(target_os = "linux")]
            Self::MotlieVnet(backing) | Self::HypervisorManagedPlusMotlieVnet(backing) => {
                Ok(backing.provision(_prepared)?.map(NetworkHandle::MotlieVnet))
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl NetworkHandle {
    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        match self {
            Self::MotlieVnet(handle) => Ok(handle.shutdown()?),
        }
    }

    pub fn backing_name(&self) -> &'static str {
        match self {
            Self::MotlieVnet(_) => "motlie-vnet",
        }
    }

    pub fn observability(&self, runtime_paths: &GuestRuntimePaths) -> NetworkObservability {
        NetworkObservability {
            backing: self.backing_name(),
            socket_path: Some(runtime_paths.vnet_socket.clone()),
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl NetworkHandle {
    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        match *self {}
    }

    pub fn backing_name(&self) -> &'static str {
        match *self {}
    }

    pub fn observability(&self, _runtime_paths: &GuestRuntimePaths) -> NetworkObservability {
        match *self {}
    }
}

impl ControlPlaneBacking {
    pub fn provision(
        &self,
        prepared: &PreparedGuest,
    ) -> Result<Option<ControlPlaneHandle>, RuntimeError> {
        match self {
            Self::None => Ok(None),
            Self::MotlieSshProxy(backing) => Ok(backing
                .provision(prepared)?
                .map(ControlPlaneHandle::MotlieSshProxy)),
        }
    }
}

impl ControlPlaneHandle {
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), RuntimeError> {
        match self {
            Self::MotlieSshProxy(handle) => Ok(handle.wait_ready(timeout).await?),
        }
    }

    pub async fn exec(&self, command: &str, timeout: Duration) -> Result<ExecOutput, RuntimeError> {
        match self {
            Self::MotlieSshProxy(handle) => Ok(handle.exec(command, timeout).await?),
        }
    }

    pub async fn open_pty(
        &self,
        request: PtyRequest,
        timeout: Duration,
    ) -> Result<GuestPtySession, RuntimeError> {
        match self {
            Self::MotlieSshProxy(handle) => Ok(handle.open_pty(request, timeout).await?),
        }
    }

    pub fn shutdown(&self) -> Result<(), RuntimeError> {
        match self {
            Self::MotlieSshProxy(handle) => Ok(handle.shutdown()?),
        }
    }

    pub fn backing_name(&self) -> &'static str {
        match self {
            Self::MotlieSshProxy(_) => "motlie-ssh-proxy",
        }
    }

    pub fn bridge_socket_path(&self) -> Option<PathBuf> {
        match self {
            Self::MotlieSshProxy(handle) => Some(handle.bridge_socket_path().to_path_buf()),
        }
    }

    pub fn observability(&self) -> ControlPlaneObservability {
        ControlPlaneObservability {
            backing: self.backing_name(),
            ssh_bridge_socket_path: self.bridge_socket_path(),
        }
    }
}

pub type SharedRuntime = Arc<Runtime>;
