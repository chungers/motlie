use std::fs;
use std::fs::File;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

use thiserror::Error;

use crate::backend::{
    BackendError, BackendHandle, BackendKind, BackendShutdownOutcome, VmBackendCapabilities,
};
use crate::orchestrator::PreparedGuest;

#[derive(Debug)]
pub struct VzShellHandle {
    pub pid: Option<u32>,
    pub guest_id: String,
    pub base_dir: PathBuf,
    pub launch_script_path: PathBuf,
    pub artifacts_dir: PathBuf,
    pub vm_name: String,
    pub runner_pid_file: PathBuf,
    pub egress_helper_pid_file: PathBuf,
    pub egress_socket_path: PathBuf,
    pub(crate) child: Mutex<Option<Child>>,
}

#[derive(Debug, Error)]
pub enum VzShellError {
    #[error("failed to create launch directory {path}: {source}")]
    CreateLaunchDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create runtime directory {path}: {source}")]
    CreateRuntimeDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create cloud-init directory {path}: {source}")]
    CreateCloudInitDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write cloud-init asset {path}: {source}")]
    WriteCloudInitAsset {
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
    #[error("failed to open launch log {path}: {source}")]
    OpenLaunchLog {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to spawn Vz shell backend from {path}: {source}")]
    SpawnShellBackend {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to run vz shutdown helper for guest {guest_id}: {source}")]
    ShutdownHelper {
        guest_id: String,
        #[source]
        source: std::io::Error,
    },
    #[error("vz shutdown helper failed for guest {guest_id}: {stderr}")]
    ShutdownHelperFailed { guest_id: String, stderr: String },
    #[error("backend child state poisoned")]
    ChildStatePoisoned,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct VzShellBackend;

impl VzShellBackend {
    pub fn new() -> Self {
        Self
    }

    fn launch_script_path(prepared: &PreparedGuest) -> PathBuf {
        prepared.runtime_paths.launch_dir.join("launch.sh")
    }

    fn shutdown_script_path(base_dir: &Path, launch_script_path: &Path) -> PathBuf {
        let configured = base_dir.join("shutdown-vz.sh");
        if configured.exists() {
            return configured;
        }
        launch_script_path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .map(|base| base.join("shutdown-vz.sh"))
            .unwrap_or_else(|| PathBuf::from("shutdown-vz.sh"))
    }

    fn materialize_launch_script(
        &self,
        prepared: &PreparedGuest,
        launch_script_path: &Path,
    ) -> Result<(), VzShellError> {
        let runtime_dir = prepared.runtime_paths.runtime_dir.clone();
        fs::create_dir_all(&runtime_dir).map_err(|source| VzShellError::CreateRuntimeDir {
            path: runtime_dir,
            source,
        })?;
        let launch_dir = prepared.runtime_paths.launch_dir.clone();
        fs::create_dir_all(&launch_dir).map_err(|source| VzShellError::CreateLaunchDir {
            path: launch_dir,
            source,
        })?;
        let cloud_init_dir = prepared.runtime_paths.cloud_init_dir.clone();
        fs::create_dir_all(&cloud_init_dir).map_err(|source| VzShellError::CreateCloudInitDir {
            path: cloud_init_dir.clone(),
            source,
        })?;
        for (name, contents) in [
            ("meta-data", prepared.cloud_init.meta_data.as_str()),
            ("user-data", prepared.cloud_init.user_data.as_str()),
            ("mounts.yaml", prepared.cloud_init.mounts_yaml.as_str()),
        ] {
            let path = cloud_init_dir.join(name);
            fs::write(&path, contents)
                .map_err(|source| VzShellError::WriteCloudInitAsset { path, source })?;
        }
        fs::write(launch_script_path, &prepared.launch_script).map_err(|source| {
            VzShellError::WriteLaunchScript {
                path: launch_script_path.to_path_buf(),
                source,
            }
        })?;
        let mut perms = fs::metadata(launch_script_path)
            .map_err(|source| VzShellError::SetLaunchScriptPermissions {
                path: launch_script_path.to_path_buf(),
                source,
            })?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(launch_script_path, perms).map_err(|source| {
            VzShellError::SetLaunchScriptPermissions {
                path: launch_script_path.to_path_buf(),
                source,
            }
        })?;
        Ok(())
    }

    fn child_lock<'a>(
        child: &'a Mutex<Option<Child>>,
    ) -> Result<std::sync::MutexGuard<'a, Option<Child>>, VzShellError> {
        child.lock().map_err(|_| VzShellError::ChildStatePoisoned)
    }

    fn pid_file_process_alive(path: &Path) -> bool {
        let Ok(pid) = fs::read_to_string(path) else {
            return false;
        };
        let pid = pid.trim();
        if pid.is_empty() {
            return false;
        }
        Command::new("kill")
            .arg("-0")
            .arg(pid)
            .status()
            .is_ok_and(|status| status.success())
    }
}

impl VzShellHandle {
    pub fn has_exited(&self) -> Result<bool, VzShellError> {
        let mut child = VzShellBackend::child_lock(&self.child)?;
        let Some(running_child) = child.as_mut() else {
            return Ok(!VzShellBackend::pid_file_process_alive(
                &self.runner_pid_file,
            ));
        };
        match running_child.try_wait() {
            Ok(Some(status)) => {
                let _ = child.take();
                if status.success() && VzShellBackend::pid_file_process_alive(&self.runner_pid_file)
                {
                    return Ok(false);
                }
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(source) => Err(VzShellError::ShutdownHelper {
                guest_id: self.guest_id.clone(),
                source,
            }),
        }
    }
}

impl VzShellBackend {
    pub fn kind(&self) -> BackendKind {
        BackendKind::Vz
    }

    pub fn capabilities(&self) -> VmBackendCapabilities {
        VmBackendCapabilities {
            same_process_vmm: false,
            supports_api_socket: false,
            supports_event_monitor: false,
            supports_fd_handoff: false,
            supports_memfd_boot_artifacts: false,
            supports_guest_metrics: false,
        }
    }

    pub fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError> {
        let launch_script_path = Self::launch_script_path(prepared);
        self.materialize_launch_script(prepared, &launch_script_path)?;
        let launch_log_path = prepared.runtime_paths.launch_log.clone();
        let launch_log =
            File::create(&launch_log_path).map_err(|source| VzShellError::OpenLaunchLog {
                path: launch_log_path,
                source,
            })?;
        let launch_log_err =
            launch_log
                .try_clone()
                .map_err(|source| VzShellError::OpenLaunchLog {
                    path: prepared.runtime_paths.launch_log.clone(),
                    source,
                })?;

        let child = Command::new("bash")
            .arg(&launch_script_path)
            .stdout(launch_log)
            .stderr(launch_log_err)
            .spawn()
            .map_err(|source| VzShellError::SpawnShellBackend {
                path: launch_script_path.clone(),
                source,
            })?;

        Ok(BackendHandle::VzShell(VzShellHandle {
            pid: Some(child.id()),
            guest_id: prepared.guest.guest_id.clone(),
            base_dir: prepared.base_dir.clone(),
            launch_script_path,
            artifacts_dir: super::artifacts_dir(&prepared.runtime_paths),
            vm_name: super::vm_name(&prepared.runtime_paths, &prepared.guest.guest_id),
            runner_pid_file: prepared.runtime_paths.runtime_dir.join("vz-runner.pid"),
            egress_helper_pid_file: prepared
                .runtime_paths
                .runtime_dir
                .join("vz-egress-helper.pid"),
            egress_socket_path: prepared.runtime_paths.runtime_dir.join("egress.sock"),
            child: Mutex::new(Some(child)),
        }))
    }

    pub fn shutdown(&self, handle: &BackendHandle) -> Result<BackendShutdownOutcome, BackendError> {
        let shell_handle = match handle {
            BackendHandle::VzShell(shell_handle) => shell_handle,
            other => {
                return Err(BackendError::HandleKindMismatch {
                    expected: BackendKind::Vz,
                    actual: other.kind(),
                });
            }
        };

        let shutdown_script =
            Self::shutdown_script_path(&shell_handle.base_dir, &shell_handle.launch_script_path);
        let output = Command::new(&shutdown_script)
            .env("MOTLIE_VZ_ARTIFACTS_DIR", &shell_handle.artifacts_dir)
            .env("MOTLIE_VZ_RUNNER_PID_FILE", &shell_handle.runner_pid_file)
            .env(
                "MOTLIE_VZ_EGRESS_HELPER_PID_FILE",
                &shell_handle.egress_helper_pid_file,
            )
            .env(
                "MOTLIE_VZ_EGRESS_SOCKET_PATH",
                &shell_handle.egress_socket_path,
            )
            .arg("--guest")
            .arg(&shell_handle.guest_id)
            .arg("--vm-name")
            .arg(&shell_handle.vm_name)
            .output()
            .map_err(|source| VzShellError::ShutdownHelper {
                guest_id: shell_handle.guest_id.clone(),
                source,
            })?;
        if !output.status.success() {
            return Err(VzShellError::ShutdownHelperFailed {
                guest_id: shell_handle.guest_id.clone(),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            }
            .into());
        }

        Ok(BackendShutdownOutcome {
            api_attempted: false,
            forced: None,
        })
    }
}
