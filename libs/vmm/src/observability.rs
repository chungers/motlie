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
    pub run_bundle: VmRunBundle,
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

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VmArtifactKind {
    RuntimeDir,
    LaunchDir,
    CloudInitDir,
    ApiSocket,
    VnetSocket,
    VsockSocket,
    GuestSocket,
    SerialLog,
    LaunchLog,
    FilesystemSocket,
    SshBridgeSocket,
    HostMount,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VmRunArtifact {
    pub kind: VmArtifactKind,
    pub label: String,
    pub path: PathBuf,
    pub guest_path: Option<PathBuf>,
    pub required: bool,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VmHostMount {
    pub tag: String,
    pub host_path: PathBuf,
    pub guest_path: Option<PathBuf>,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VmCapturePaths {
    pub scenario_result_json: PathBuf,
    pub pty_transcript_ndjson: PathBuf,
    pub pty_screen_json: PathBuf,
    pub pty_screen_svg: PathBuf,
    pub pty_asciicast: PathBuf,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VmRunBundle {
    pub bundle_root: PathBuf,
    pub capture_paths: VmCapturePaths,
    pub host_mounts: Vec<VmHostMount>,
    pub artifacts: Vec<VmRunArtifact>,
}
