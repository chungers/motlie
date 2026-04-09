use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

/// Guest OS login user modeled above the hypervisor/backend layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestUser {
    pub name: String,
    pub uid: u32,
    pub gid: u32,
    pub home: PathBuf,
}

/// SSH principal/login-user policy for a guest OS account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestSshAccess {
    pub principal: String,
    pub login_user: String,
}

/// One host-backed mount made visible inside a guest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: Option<PathBuf>,
    pub host_path: PathBuf,
}

/// Declarative software customization for the guest image / cloud-init flow.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SoftwareProfile {
    pub packages: Vec<String>,
}

/// CH-shaped compute and memory resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestResources {
    pub boot_vcpus: u8,
    pub memory_mib: u32,
    pub max_vcpus: Option<u8>,
}

impl Default for GuestResources {
    fn default() -> Self {
        Self {
            boot_vcpus: 2,
            memory_mib: 512,
            max_vcpus: None,
        }
    }
}

/// Storage/image policy that does not directly map to CPU/RAM sizing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestStorage {
    pub overlay_size: OverlaySize,
}

impl Default for GuestStorage {
    fn default() -> Self {
        Self {
            overlay_size: OverlaySize::new("2G").expect("default overlay size is valid"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlaySize(String);

impl OverlaySize {
    pub fn new(value: impl Into<String>) -> Result<Self, SpecError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(SpecError::InvalidOverlaySize {
                value,
                reason: "overlay size cannot be empty".to_string(),
            });
        }

        let digit_len = value
            .bytes()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        let (digits, suffix) = value.split_at(digit_len);
        if digits.is_empty() {
            return Err(SpecError::InvalidOverlaySize {
                value,
                reason: "overlay size must start with digits".to_string(),
            });
        }
        if !suffix.is_empty()
            && !matches!(suffix, "K" | "M" | "G" | "T" | "P" | "KiB" | "MiB" | "GiB")
        {
            return Err(SpecError::InvalidOverlaySize {
                value,
                reason: "overlay size suffix must be one of K/M/G/T/P or KiB/MiB/GiB".to_string(),
            });
        }

        Ok(Self(format!("{digits}{suffix}")))
    }
}

impl std::fmt::Display for OverlaySize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl AsRef<str> for OverlaySize {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Declarative boot artifact inputs above the renderer/backend layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootArtifacts {
    pub kernel: PathBuf,
    pub initramfs: Option<PathBuf>,
    pub firmware: Option<PathBuf>,
    pub cmdline: Option<String>,
}

/// Pure guest input configuration.
///
/// This is the library-owned replacement target for the pure data currently
/// carried by the example harness's guest configuration structs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestSpec {
    pub guest_id: String,
    pub hostname: String,
    pub socket_path: PathBuf,
    pub user: GuestUser,
    pub ssh: GuestSshAccess,
    pub mounts: Vec<GuestMountSpec>,
    pub software: SoftwareProfile,
    pub resources: GuestResources,
    pub storage: GuestStorage,
    pub boot: BootArtifacts,
}

/// Namespace-sensitive runtime path configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeNamespace {
    pub prefix: String,
    pub temp_root: PathBuf,
}

impl RuntimeNamespace {
    pub fn new(
        prefix: impl Into<String>,
        temp_root: impl Into<PathBuf>,
    ) -> Result<Self, SpecError> {
        let prefix = prefix.into();
        if prefix.trim().is_empty() {
            return Err(SpecError::EmptyNamespacePrefix);
        }

        Ok(Self {
            prefix,
            temp_root: temp_root.into(),
        })
    }

    pub fn root_from_env_or_temp() -> PathBuf {
        std::env::var_os("MOTLIE_VMM_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
    }

    pub fn for_process(
        prefix_base: &str,
        instance_tag: &str,
        temp_root: impl Into<PathBuf>,
    ) -> Result<Self, SpecError> {
        if prefix_base.trim().is_empty() {
            return Err(SpecError::EmptyNamespacePrefix);
        }
        let pid = std::process::id();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| SpecError::InvalidClock)?
            .as_millis()
            % 10_000;
        let prefix = format!("{prefix_base}-{instance_tag}{pid}-{nonce}");
        Self::new(prefix, temp_root)
    }

    pub fn guest_vsock_port_socket(
        &self,
        guest_name: &str,
        port: u32,
    ) -> Result<PathBuf, SpecError> {
        if guest_name.trim().is_empty() {
            return Err(SpecError::EmptyGuestId);
        }
        Ok(self
            .temp_root
            .join(format!("{}-{guest_name}.vsock_{port}", self.prefix)))
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
            return Err(SpecError::EmptyGuestId);
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
    #[error("guest id cannot be empty")]
    EmptyGuestId,
    #[error("guest hostname cannot be empty")]
    EmptyHostname,
    #[error("guest username cannot be empty")]
    EmptyGuestUserName,
    #[error("guest ssh principal cannot be empty")]
    EmptyGuestPrincipal,
    #[error("guest socket path cannot be empty")]
    EmptySocketPath,
    #[error("invalid overlay size '{value}': {reason}")]
    InvalidOverlaySize { value: String, reason: String },
    #[error("system clock is invalid for instance namespace generation")]
    InvalidClock,
}

pub fn pathbuf_from_optional_str(value: Option<&str>) -> Option<PathBuf> {
    value.map(Path::new).map(Path::to_path_buf)
}

impl GuestSpec {
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.guest_id.trim().is_empty() {
            return Err(SpecError::EmptyGuestId);
        }
        if self.hostname.trim().is_empty() {
            return Err(SpecError::EmptyHostname);
        }
        if self.user.name.trim().is_empty() {
            return Err(SpecError::EmptyGuestUserName);
        }
        if self.ssh.principal.trim().is_empty() {
            return Err(SpecError::EmptyGuestPrincipal);
        }
        if self.socket_path.as_os_str().is_empty() {
            return Err(SpecError::EmptySocketPath);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_paths_follow_namespace() {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp").unwrap();
        let paths = GuestRuntimePaths::for_guest(&namespace, "alice").unwrap();

        assert_eq!(
            paths.runtime_dir,
            PathBuf::from("/tmp/motlie-vmm-v14-runtime/alice")
        );
        assert_eq!(
            paths.launch_dir,
            PathBuf::from("/tmp/motlie-vmm-v14-launch/alice")
        );
        assert_eq!(
            paths.cloud_init_dir,
            PathBuf::from("/tmp/motlie-vmm-v14-cloud-init-alice")
        );
        assert_eq!(
            paths.api_socket,
            PathBuf::from("/tmp/motlie-vmm-v14-alice-api.sock")
        );
        assert_eq!(
            paths.vnet_socket,
            PathBuf::from("/tmp/motlie-vmm-v14-alice.sock")
        );
        assert_eq!(
            paths.vsock_socket,
            PathBuf::from("/tmp/motlie-vmm-v14-alice.vsock")
        );
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
        assert_eq!(err, SpecError::EmptyGuestId);
    }

    #[test]
    fn namespace_rejects_empty_prefix() {
        let err = RuntimeNamespace::new("", "/tmp").unwrap_err();
        assert_eq!(err, SpecError::EmptyNamespacePrefix);
    }

    #[test]
    fn runtime_namespace_guest_vsock_port_socket_follows_root_and_prefix() {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14-test", "/var/tmp").unwrap();
        let path = namespace.guest_vsock_port_socket("alice", 5000).unwrap();
        assert_eq!(
            path,
            PathBuf::from("/var/tmp/motlie-vmm-v14-test-alice.vsock_5000")
        );
    }

    #[test]
    fn optional_string_to_pathbuf_preserves_absence() {
        assert_eq!(pathbuf_from_optional_str(None), None);
        assert_eq!(
            pathbuf_from_optional_str(Some("/home/alice")),
            Some(PathBuf::from("/home/alice"))
        );
    }

    #[test]
    fn guest_spec_validates_required_fields() {
        let spec = GuestSpec {
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
            mounts: vec![],
            software: SoftwareProfile::default(),
            resources: GuestResources::default(),
            storage: GuestStorage::default(),
            boot: BootArtifacts {
                kernel: PathBuf::from("/tmp/Image"),
                initramfs: None,
                firmware: None,
                cmdline: None,
            },
        };

        assert!(spec.validate().is_ok());
    }
}
