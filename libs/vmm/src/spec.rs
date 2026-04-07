use std::path::{Path, PathBuf};

use thiserror::Error;

/// Guest identity override for mounted content and generated runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestIdentity {
    pub uid: u32,
    pub gid: u32,
}

/// One host-backed mount made visible inside a guest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: Option<PathBuf>,
    pub host_path: PathBuf,
}

/// Pure guest input configuration.
///
/// This is the library-owned replacement target for the pure data currently
/// carried by the example harness's guest configuration structs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestSpec {
    pub name: String,
    pub socket_path: String,
    pub mounts: Vec<GuestMountSpec>,
    pub identity: Option<GuestIdentity>,
}

/// Namespace-sensitive runtime path configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeNamespace {
    pub prefix: String,
    pub temp_root: PathBuf,
}

impl RuntimeNamespace {
    pub fn new(prefix: impl Into<String>, temp_root: impl Into<PathBuf>) -> Result<Self, SpecError> {
        let prefix = prefix.into();
        if prefix.trim().is_empty() {
            return Err(SpecError::EmptyNamespacePrefix);
        }

        Ok(Self {
            prefix,
            temp_root: temp_root.into(),
        })
    }
}

/// Deterministic runtime layout for one guest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestRuntimePaths {
    pub runtime_dir: PathBuf,
    pub launch_dir: PathBuf,
    pub cloud_init_dir: PathBuf,
    pub api_socket: PathBuf,
    pub vnet_socket: PathBuf,
    pub vsock_socket: PathBuf,
    pub serial_log: PathBuf,
    pub launch_log: PathBuf,
}

impl GuestRuntimePaths {
    pub fn for_guest(namespace: &RuntimeNamespace, guest_name: &str) -> Result<Self, SpecError> {
        if guest_name.trim().is_empty() {
            return Err(SpecError::EmptyGuestName);
        }

        let runtime_dir = namespace
            .temp_root
            .join(format!("{}-runtime", namespace.prefix))
            .join(guest_name);
        let launch_dir = namespace
            .temp_root
            .join(format!("{}-launch", namespace.prefix))
            .join(guest_name);
        let cloud_init_dir = namespace
            .temp_root
            .join(format!("{}-cloud-init-{guest_name}", namespace.prefix));
        let api_socket = namespace
            .temp_root
            .join(format!("{}-{guest_name}-api.sock", namespace.prefix));
        let vnet_socket = namespace
            .temp_root
            .join(format!("{}-{guest_name}.sock", namespace.prefix));
        let vsock_socket = namespace
            .temp_root
            .join(format!("{}-{guest_name}.vsock", namespace.prefix));
        let serial_log = launch_dir.join("serial.log");
        let launch_log = launch_dir.join("launch.log");

        Ok(Self {
            runtime_dir,
            launch_dir,
            cloud_init_dir,
            api_socket,
            vnet_socket,
            vsock_socket,
            serial_log,
            launch_log,
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SpecError {
    #[error("namespace prefix cannot be empty")]
    EmptyNamespacePrefix,
    #[error("guest name cannot be empty")]
    EmptyGuestName,
}

pub fn pathbuf_from_optional_str(value: Option<&str>) -> Option<PathBuf> {
    value.map(Path::new).map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_paths_follow_namespace() {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp").unwrap();
        let paths = GuestRuntimePaths::for_guest(&namespace, "alice").unwrap();

        assert_eq!(paths.runtime_dir, PathBuf::from("/tmp/motlie-vmm-v14-runtime/alice"));
        assert_eq!(paths.launch_dir, PathBuf::from("/tmp/motlie-vmm-v14-launch/alice"));
        assert_eq!(
            paths.cloud_init_dir,
            PathBuf::from("/tmp/motlie-vmm-v14-cloud-init-alice")
        );
        assert_eq!(
            paths.api_socket,
            PathBuf::from("/tmp/motlie-vmm-v14-alice-api.sock")
        );
        assert_eq!(paths.vnet_socket, PathBuf::from("/tmp/motlie-vmm-v14-alice.sock"));
        assert_eq!(paths.vsock_socket, PathBuf::from("/tmp/motlie-vmm-v14-alice.vsock"));
        assert_eq!(
            paths.serial_log,
            PathBuf::from("/tmp/motlie-vmm-v14-launch/alice/serial.log")
        );
        assert_eq!(
            paths.launch_log,
            PathBuf::from("/tmp/motlie-vmm-v14-launch/alice/launch.log")
        );
    }

    #[test]
    fn runtime_paths_reject_empty_guest_name() {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp").unwrap();
        let err = GuestRuntimePaths::for_guest(&namespace, "").unwrap_err();
        assert_eq!(err, SpecError::EmptyGuestName);
    }

    #[test]
    fn namespace_rejects_empty_prefix() {
        let err = RuntimeNamespace::new("", "/tmp").unwrap_err();
        assert_eq!(err, SpecError::EmptyNamespacePrefix);
    }

    #[test]
    fn optional_string_to_pathbuf_preserves_absence() {
        assert_eq!(pathbuf_from_optional_str(None), None);
        assert_eq!(
            pathbuf_from_optional_str(Some("/home/alice")),
            Some(PathBuf::from("/home/alice"))
        );
    }
}
