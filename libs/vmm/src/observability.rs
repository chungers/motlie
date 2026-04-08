use std::path::PathBuf;

use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VmObservability {
    pub guest_id: String,
    pub pid: Option<u32>,
    pub namespace_prefix: String,
    pub temp_root: PathBuf,
    pub guest_socket_path: PathBuf,
    pub runtime_paths: VmRuntimePaths,
    pub filesystem: FilesystemObservability,
    pub network: NetworkObservability,
    pub control_plane: ControlPlaneObservability,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VmRuntimePaths {
    pub runtime_dir: PathBuf,
    pub launch_dir: PathBuf,
    pub cloud_init_dir: PathBuf,
    pub api_socket: PathBuf,
    pub vnet_socket: PathBuf,
    pub vsock_socket: PathBuf,
    pub serial_log: PathBuf,
    pub launch_log: PathBuf,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FilesystemObservability {
    pub backing: &'static str,
    pub socket_path: Option<PathBuf>,
    pub mount_tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NetworkObservability {
    pub backing: &'static str,
    pub socket_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ControlPlaneObservability {
    pub backing: &'static str,
    pub ssh_bridge_socket_path: Option<PathBuf>,
}
