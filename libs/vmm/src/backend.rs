use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

use crate::orchestrator::PreparedGuest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    ChShell,
    ChForkExec,
    ChVmmThread,
    Vz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmBackendCapabilities {
    pub same_process_vmm: bool,
    pub supports_api_socket: bool,
    pub supports_event_monitor: bool,
    pub supports_fd_handoff: bool,
    pub supports_memfd_boot_artifacts: bool,
    pub supports_guest_metrics: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChShellHandle {
    pub pid: Option<u32>,
    pub launch_script_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendHandle {
    ChShell(ChShellHandle),
}

impl BackendHandle {
    pub fn pid(&self) -> Option<u32> {
        match self {
            Self::ChShell(handle) => handle.pid,
        }
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::ChShell(_) => BackendKind::ChShell,
        }
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("failed to create launch directory {path}: {source}")]
    CreateLaunchDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write launch script {path}: {source}")]
    WriteLaunchScript {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to set executable permissions on {path}: {source}")]
    SetLaunchScriptPermissions {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to spawn shell backend from {path}: {source}")]
    SpawnShellBackend {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to invoke kill for pid {pid}: {source}")]
    KillProcess {
        pid: u32,
        #[source]
        source: std::io::Error,
    },
    #[error("kill returned non-zero for pid {pid}: {stderr}")]
    KillProcessFailed { pid: u32, stderr: String },
    #[error("backend handle {actual:?} does not match expected backend {expected:?}")]
    HandleKindMismatch {
        expected: BackendKind,
        actual: BackendKind,
    },
}

pub trait VmBackend {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> VmBackendCapabilities;
    fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError>;
    fn shutdown(&self, handle: &BackendHandle) -> Result<(), BackendError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ChShellBackend;

impl ChShellBackend {
    pub fn new() -> Self {
        Self
    }

    fn launch_script_path(prepared: &PreparedGuest) -> PathBuf {
        prepared.runtime_paths.launch_dir.join("launch.sh")
    }

    fn materialize_launch_script(
        &self,
        prepared: &PreparedGuest,
        launch_script_path: &Path,
    ) -> Result<(), BackendError> {
        let launch_dir = prepared.runtime_paths.launch_dir.clone();
        fs::create_dir_all(&launch_dir).map_err(|source| BackendError::CreateLaunchDir {
            path: launch_dir,
            source,
        })?;
        fs::write(launch_script_path, &prepared.launch_script).map_err(|source| {
            BackendError::WriteLaunchScript {
                path: launch_script_path.to_path_buf(),
                source,
            }
        })?;
        let mut perms = fs::metadata(launch_script_path)
            .map_err(|source| BackendError::SetLaunchScriptPermissions {
                path: launch_script_path.to_path_buf(),
                source,
            })?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(launch_script_path, perms).map_err(|source| {
            BackendError::SetLaunchScriptPermissions {
                path: launch_script_path.to_path_buf(),
                source,
            }
        })?;
        Ok(())
    }
}

impl VmBackend for ChShellBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::ChShell
    }

    fn capabilities(&self) -> VmBackendCapabilities {
        VmBackendCapabilities {
            same_process_vmm: false,
            supports_api_socket: true,
            supports_event_monitor: false,
            supports_fd_handoff: false,
            supports_memfd_boot_artifacts: false,
            supports_guest_metrics: false,
        }
    }

    fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError> {
        let launch_script_path = Self::launch_script_path(prepared);
        self.materialize_launch_script(prepared, &launch_script_path)?;

        let child = Command::new(&launch_script_path)
            .spawn()
            .map_err(|source| BackendError::SpawnShellBackend {
                path: launch_script_path.clone(),
                source,
            })?;

        Ok(BackendHandle::ChShell(ChShellHandle {
            pid: Some(child.id()),
            launch_script_path,
        }))
    }

    fn shutdown(&self, handle: &BackendHandle) -> Result<(), BackendError> {
        let BackendHandle::ChShell(shell_handle) = handle;
        let Some(pid) = shell_handle.pid else {
            return Ok(());
        };

        let output = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .output()
            .map_err(|source| BackendError::KillProcess { pid, source })?;
        if output.status.success() {
            return Ok(());
        }

        Err(BackendError::KillProcessFailed {
            pid,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ch_shell_backend_reports_expected_capabilities() {
        let backend = ChShellBackend::new();
        assert_eq!(backend.kind(), BackendKind::ChShell);
        assert_eq!(
            backend.capabilities(),
            VmBackendCapabilities {
                same_process_vmm: false,
                supports_api_socket: true,
                supports_event_monitor: false,
                supports_fd_handoff: false,
                supports_memfd_boot_artifacts: false,
                supports_guest_metrics: false,
            }
        );
    }
}
