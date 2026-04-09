use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::backend::{
    BackendError, BackendHandle, BackendKind, BackendShutdownOutcome, VmBackendCapabilities,
};
use crate::orchestrator::PreparedGuest;

const CH_SHELL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const CH_SHELL_SHUTDOWN_POLL: Duration = Duration::from_millis(100);
const CH_VM_SHUTDOWN_HTTP_PATH: &str = "/api/v1/vm.shutdown";

#[derive(Debug)]
pub struct ChShellHandle {
    pub pid: Option<u32>,
    pub launch_script_path: PathBuf,
    pub api_socket: PathBuf,
    pub(crate) child: Mutex<Option<Child>>,
}

#[derive(Debug, Error)]
pub enum ChShellError {
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
    #[error("failed to spawn shell backend from {path}: {source}")]
    SpawnShellBackend {
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
    #[error("failed to invoke kill for pid {pid}: {source}")]
    KillProcess {
        pid: u32,
        #[source]
        source: std::io::Error,
    },
    #[error("kill returned non-zero for pid {pid}: {stderr}")]
    KillProcessFailed { pid: u32, stderr: String },
    #[error("failed to issue API shutdown over {path}: {reason}")]
    ShutdownApi { path: PathBuf, reason: String },
    #[error("backend child state poisoned")]
    ChildStatePoisoned,
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
    ) -> Result<(), ChShellError> {
        let runtime_dir = prepared.runtime_paths.runtime_dir.clone();
        fs::create_dir_all(&runtime_dir).map_err(|source| ChShellError::CreateRuntimeDir {
            path: runtime_dir,
            source,
        })?;
        let launch_dir = prepared.runtime_paths.launch_dir.clone();
        fs::create_dir_all(&launch_dir).map_err(|source| ChShellError::CreateLaunchDir {
            path: launch_dir,
            source,
        })?;
        let cloud_init_dir = prepared.runtime_paths.cloud_init_dir.clone();
        fs::create_dir_all(&cloud_init_dir).map_err(|source| ChShellError::CreateCloudInitDir {
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
                .map_err(|source| ChShellError::WriteCloudInitAsset { path, source })?;
        }
        fs::write(launch_script_path, &prepared.launch_script).map_err(|source| {
            ChShellError::WriteLaunchScript {
                path: launch_script_path.to_path_buf(),
                source,
            }
        })?;
        let mut perms = fs::metadata(launch_script_path)
            .map_err(|source| ChShellError::SetLaunchScriptPermissions {
                path: launch_script_path.to_path_buf(),
                source,
            })?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(launch_script_path, perms).map_err(|source| {
            ChShellError::SetLaunchScriptPermissions {
                path: launch_script_path.to_path_buf(),
                source,
            }
        })?;
        Ok(())
    }

    fn signal_process(pid: u32, signal: &str) -> Result<(), ChShellError> {
        let output = Command::new("kill")
            .args([signal, &pid.to_string()])
            .output()
            .map_err(|source| ChShellError::KillProcess { pid, source })?;
        if output.status.success() {
            return Ok(());
        }

        if !process_exists(pid) {
            return Ok(());
        }

        Err(ChShellError::KillProcessFailed {
            pid,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }

    fn request_api_shutdown(api_socket: &Path) -> Result<bool, ChShellError> {
        if !api_socket.exists() {
            return Ok(false);
        }

        let mut stream =
            UnixStream::connect(api_socket).map_err(|source| ChShellError::ShutdownApi {
                path: api_socket.to_path_buf(),
                reason: source.to_string(),
            })?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|source| ChShellError::ShutdownApi {
                path: api_socket.to_path_buf(),
                reason: source.to_string(),
            })?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|source| ChShellError::ShutdownApi {
                path: api_socket.to_path_buf(),
                reason: source.to_string(),
            })?;
        stream
            .write_all(
                format!(
                    "PUT {CH_VM_SHUTDOWN_HTTP_PATH} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                )
                .as_bytes(),
            )
            .map_err(|source| ChShellError::ShutdownApi {
                path: api_socket.to_path_buf(),
                reason: source.to_string(),
            })?;

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader
            .read_line(&mut status_line)
            .map_err(|source| ChShellError::ShutdownApi {
                path: api_socket.to_path_buf(),
                reason: source.to_string(),
            })?;

        let ok = status_line.starts_with("HTTP/1.1 200")
            || status_line.starts_with("HTTP/1.1 202")
            || status_line.starts_with("HTTP/1.1 204");
        if ok {
            Ok(true)
        } else {
            Err(ChShellError::ShutdownApi {
                path: api_socket.to_path_buf(),
                reason: status_line.trim().to_string(),
            })
        }
    }

    fn child_lock<'a>(
        child: &'a Mutex<Option<Child>>,
    ) -> Result<std::sync::MutexGuard<'a, Option<Child>>, ChShellError> {
        child.lock().map_err(|_| ChShellError::ChildStatePoisoned)
    }
}

impl ChShellHandle {
    pub fn has_exited(&self) -> Result<bool, ChShellError> {
        let mut child = ChShellBackend::child_lock(&self.child)?;
        let Some(running_child) = child.as_mut() else {
            return Ok(self.pid.is_some());
        };
        match running_child
            .try_wait()
            .map_err(|source| ChShellError::KillProcess {
                pid: self.pid.unwrap_or_default(),
                source,
            })? {
            Some(_) => {
                let _ = child.take();
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

impl ChShellBackend {
    pub fn kind(&self) -> BackendKind {
        BackendKind::ChShell
    }

    pub fn capabilities(&self) -> VmBackendCapabilities {
        VmBackendCapabilities {
            same_process_vmm: false,
            supports_api_socket: true,
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
            File::create(&launch_log_path).map_err(|source| ChShellError::OpenLaunchLog {
                path: launch_log_path,
                source,
            })?;
        let launch_log_err =
            launch_log
                .try_clone()
                .map_err(|source| ChShellError::OpenLaunchLog {
                    path: prepared.runtime_paths.launch_log.clone(),
                    source,
                })?;

        let child = Command::new("bash")
            .arg(&launch_script_path)
            .stdout(launch_log)
            .stderr(launch_log_err)
            .spawn()
            .map_err(|source| ChShellError::SpawnShellBackend {
                path: launch_script_path.clone(),
                source,
            })?;

        Ok(BackendHandle::ChShell(ChShellHandle {
            pid: Some(child.id()),
            launch_script_path,
            api_socket: prepared.runtime_paths.api_socket.clone(),
            child: Mutex::new(Some(child)),
        }))
    }

    pub fn shutdown(&self, handle: &BackendHandle) -> Result<BackendShutdownOutcome, BackendError> {
        let shell_handle = match handle {
            BackendHandle::ChShell(shell_handle) => shell_handle,
        };

        let Some(pid) = shell_handle.pid else {
            return Ok(BackendShutdownOutcome {
                api_attempted: false,
                forced: None,
            });
        };

        let mut api_attempted = false;
        if shell_handle.api_socket.exists() {
            api_attempted = true;
            if Self::request_api_shutdown(&shell_handle.api_socket).is_ok() {
                let deadline = Instant::now() + CH_SHELL_SHUTDOWN_TIMEOUT;
                while Instant::now() < deadline {
                    if shell_handle.has_exited()? {
                        return Ok(BackendShutdownOutcome {
                            api_attempted,
                            forced: None,
                        });
                    }
                    thread::sleep(CH_SHELL_SHUTDOWN_POLL);
                }
            }
        }

        Self::signal_process(pid, "-TERM")?;
        let deadline = Instant::now() + CH_SHELL_SHUTDOWN_TIMEOUT;
        while Instant::now() < deadline {
            if shell_handle.has_exited()? {
                return Ok(BackendShutdownOutcome {
                    api_attempted,
                    forced: Some("term"),
                });
            }
            thread::sleep(CH_SHELL_SHUTDOWN_POLL);
        }

        {
            let mut child = Self::child_lock(&shell_handle.child)?;
            if let Some(child) = child.as_mut() {
                child
                    .kill()
                    .map_err(|source| ChShellError::KillProcess { pid, source })?;
                let _ = child
                    .wait()
                    .map_err(|source| ChShellError::KillProcess { pid, source })?;
            } else if process_exists(pid) {
                Self::signal_process(pid, "-KILL")?;
            }
            *child = None;
        }
        Ok(BackendShutdownOutcome {
            api_attempted,
            forced: Some("kill"),
        })
    }
}

fn process_exists(pid: u32) -> bool {
    PathBuf::from(format!("/proc/{pid}")).exists()
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use crate::artifacts::CloudInitArtifacts;
    use crate::network::{AdminNetMode, EgressNetMode, NetworkModes};
    use crate::network_alloc::{
        AdminIpv4Pair, EgressIpv4Layout, GuestNetAllocator, GuestNetAllocatorConfig,
    };
    use crate::spec::{
        BootArtifacts, GuestMountSpec, GuestResources, GuestRuntimePaths, GuestSpec,
        GuestSshAccess, GuestStorage, GuestUser, RuntimeNamespace, SoftwareProfile,
    };

    use super::*;

    fn sample_prepared_guest(script: &str) -> PreparedGuest {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp").unwrap();
        let runtime_paths = GuestRuntimePaths::for_guest(&namespace, "alice").unwrap();
        let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig::default()).unwrap();
        let net_assignment = allocator.ensure("alice").unwrap().clone();

        PreparedGuest {
            guest: GuestSpec {
                guest_id: "alice".to_string(),
                hostname: "motlie-alice".to_string(),
                socket_path: PathBuf::from("/tmp/motlie-vmm-v14-alice.vsock_5000"),
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
            },
            namespace,
            runtime_paths,
            guest_socket_path: PathBuf::from("/tmp/motlie-vmm-v14-alice.vsock_5000"),
            net_assignment,
            cloud_init: CloudInitArtifacts {
                meta_data: "instance-id: alice\nlocal-hostname: motlie-alice\n".to_string(),
                user_data: "#cloud-config\n".to_string(),
                mounts_yaml: "mounts:\n".to_string(),
            },
            launch_script: script.to_string(),
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
            base_dir: PathBuf::from("/tmp/vmm-v1.4/libs/vmm/examples/v1.4"),
        }
    }

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

    #[test]
    fn ch_shell_backend_materializes_assets_and_spawns_script() {
        let tempdir = tempfile::tempdir().unwrap();
        let namespace = RuntimeNamespace::new("motlie-vmm-v14-test", tempdir.path()).unwrap();
        let mut prepared = sample_prepared_guest("#!/usr/bin/env bash\nexit 0\n");
        prepared.runtime_paths = GuestRuntimePaths::for_guest(&namespace, "alice").unwrap();
        prepared.net_assignment = crate::network_alloc::GuestNetAssignment {
            guest_name: "alice".to_string(),
            slot: 0,
            cid: 3,
            admin_subnet: "172.20.0.0/30".parse().unwrap(),
            admin_ipv4: AdminIpv4Pair {
                host: Ipv4Addr::new(172, 20, 0, 1),
                guest: Ipv4Addr::new(172, 20, 0, 2),
            },
            admin_mac: [0x52, 0x54, 0x00, 0xa0, 0x00, 0x00],
            egress_subnet: "10.0.0.0/24".parse().unwrap(),
            egress_ipv4: EgressIpv4Layout {
                guest: Ipv4Addr::new(10, 0, 0, 15),
                host: Ipv4Addr::new(10, 0, 0, 2),
                dns: Ipv4Addr::new(10, 0, 0, 3),
                netmask: Ipv4Addr::new(255, 255, 255, 0),
            },
            egress_mac: [0x52, 0x54, 0x00, 0xe0, 0x00, 0x00],
            vnet_socket_path: tempdir.path().join("alice.sock"),
        };

        let backend = ChShellBackend::new();
        let handle = backend.boot(&prepared).unwrap();

        assert!(
            prepared
                .runtime_paths
                .cloud_init_dir
                .join("meta-data")
                .exists()
        );
        assert!(
            prepared
                .runtime_paths
                .cloud_init_dir
                .join("user-data")
                .exists()
        );
        assert!(
            prepared
                .runtime_paths
                .cloud_init_dir
                .join("mounts.yaml")
                .exists()
        );
        assert!(prepared.runtime_paths.launch_dir.join("launch.sh").exists());
        assert!(prepared.runtime_paths.launch_log.exists());
        assert_eq!(handle.kind(), BackendKind::ChShell);
    }

    #[test]
    fn ch_shell_backend_shutdown_terminates_process() {
        let tempdir = tempfile::tempdir().unwrap();
        let namespace = RuntimeNamespace::new("motlie-vmm-v14-test", tempdir.path()).unwrap();
        let mut prepared = sample_prepared_guest("#!/usr/bin/env bash\nsleep 30\n");
        prepared.runtime_paths = GuestRuntimePaths::for_guest(&namespace, "alice").unwrap();

        let backend = ChShellBackend::new();
        let handle = backend.boot(&prepared).unwrap();
        assert!(!handle.has_exited().unwrap());

        let outcome = backend.shutdown(&handle).unwrap();
        assert!(matches!(outcome.forced, None | Some("term") | Some("kill")));
        assert!(handle.has_exited().unwrap());
    }
}
