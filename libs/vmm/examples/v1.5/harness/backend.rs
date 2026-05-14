use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use motlie_vmm::ca::SshCa;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::orchestrator::ReadinessPolicy;
use motlie_vmm::runtime::{
    ControlPlaneBacking, FilesystemBacking, HypervisorBacking, NetworkBacking, Runtime,
};
use motlie_vmm::spec::BootArtifacts;
use motlie_vmm::ssh::GuestRegistry;
use serde::{Deserialize, Serialize};

use crate::{ensure_file_exists, resolved_native_source_dir, DynError};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HarnessBackend {
    #[default]
    Vz,
    Ch,
}

impl fmt::Display for HarnessBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vz => f.write_str("vz"),
            Self::Ch => f.write_str("ch"),
        }
    }
}

impl std::str::FromStr for HarnessBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "vz" | "apple-vz" | "apple_virtualization" => Ok(Self::Vz),
            "ch" | "cloud-hypervisor" | "cloud_hypervisor" => Ok(Self::Ch),
            other => Err(format!(
                "unsupported harness backend '{other}' (expected 'vz' or 'ch')"
            )),
        }
    }
}

impl HarnessBackend {
    pub(crate) fn artifacts_dir(self, base_dir: &Path) -> PathBuf {
        match self {
            Self::Vz => resolved_native_source_dir(base_dir),
            Self::Ch => std::env::var_os("MOTLIE_V15_CH_BASE_ARTIFACTS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| base_dir.join("artifacts/base")),
        }
    }

    pub(crate) fn ensure_artifacts(self, artifacts_dir: &Path) -> Result<(), DynError> {
        match self {
            Self::Vz => {
                ensure_file_exists(&artifacts_dir.join("disk.img"))?;
                ensure_file_exists(&artifacts_dir.join("nvram.bin"))?;
            }
            Self::Ch => {
                ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))?;
                if !artifacts_dir.join("Image").exists()
                    && !artifacts_dir.join("vmlinux.bin").exists()
                {
                    return Err(format!(
                        "required CH kernel artifact missing: {} or {}",
                        artifacts_dir.join("Image").display(),
                        artifacts_dir.join("vmlinux.bin").display()
                    )
                    .into());
                }
                self.ensure_guest_contract(&artifacts_dir.join("guest-contract.json"))?;
            }
        }
        Ok(())
    }

    pub(crate) fn boot_artifacts(self, artifacts_dir: &Path) -> BootArtifacts {
        match self {
            Self::Vz => BootArtifacts {
                kernel: artifacts_dir.join("disk.img"),
                initramfs: None,
                firmware: Some(artifacts_dir.join("nvram.bin")),
                cmdline: None,
            },
            Self::Ch => {
                let kernel = if artifacts_dir.join("Image").exists() {
                    artifacts_dir.join("Image")
                } else {
                    artifacts_dir.join("vmlinux.bin")
                };
                BootArtifacts {
                    kernel,
                    initramfs: None,
                    firmware: None,
                    cmdline: None,
                }
            }
        }
    }

    pub(crate) fn network_modes(self) -> NetworkModes {
        match self {
            Self::Vz => NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VzUserspace,
            },
            Self::Ch => NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
        }
    }

    pub(crate) fn readiness_policy(self) -> ReadinessPolicy {
        match self {
            Self::Vz => ReadinessPolicy {
                api_socket_timeout: Duration::from_secs(10),
                guestfs_timeout: Duration::from_secs(90),
                ssh_bridge_timeout: Duration::from_secs(90),
                exec_ready_timeout: Duration::from_secs(90),
            },
            Self::Ch => ReadinessPolicy {
                api_socket_timeout: Duration::from_secs(10),
                guestfs_timeout: Duration::from_secs(90),
                ssh_bridge_timeout: Duration::from_secs(120),
                exec_ready_timeout: Duration::from_secs(90),
            },
        }
    }

    pub(crate) fn runtime(
        self,
        ca: &Arc<SshCa>,
        guest_registry: &GuestRegistry,
    ) -> Result<Runtime, DynError> {
        let hypervisor = match self {
            Self::Vz => HypervisorBacking::AppleVirtualizationShell(
                motlie_vmm::backend::vz::shell::VzShellBackend::new(),
            ),
            Self::Ch => HypervisorBacking::CloudHypervisorShell(
                motlie_vmm::backend::ch::shell::ChShellBackend::new(),
            ),
        };
        let network = self.network_backing()?;
        Ok(Runtime {
            hypervisor,
            filesystem: FilesystemBacking::MotlieVfs(
                motlie_vmm::backend::motlie::vfs::MotlieVfsBacking::new(),
            ),
            network,
            control_plane: ControlPlaneBacking::MotlieSshProxy(
                motlie_vmm::backend::motlie::ssh_proxy::MotlieSshProxyBacking::new(
                    Arc::clone(ca),
                    Arc::clone(guest_registry),
                ),
            ),
        })
    }

    pub(crate) fn cleanup_vz_disks(self) -> bool {
        matches!(self, Self::Vz)
    }

    fn network_backing(self) -> Result<NetworkBacking, DynError> {
        match self {
            Self::Vz => Ok(NetworkBacking::VzUserspaceEgress(
                motlie_vmm::backend::vz::egress::VzUserspaceEgressBacking::new(),
            )),
            Self::Ch => ch_network_backing(),
        }
    }

    fn ensure_guest_contract(self, path: &Path) -> Result<(), DynError> {
        ensure_file_exists(path)?;
        let contents = std::fs::read_to_string(path)?;
        for required in [
            "MOTLIE_VMM_GUEST_MOUNTER_V1_5",
            "--no-default-features --features guest-vfs",
        ] {
            if !contents.contains(required) {
                return Err(format!(
                    "{} does not record required v1.5 contract marker: {}",
                    path.display(),
                    required
                )
                .into());
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn ch_network_backing() -> Result<NetworkBacking, DynError> {
    Ok(NetworkBacking::MotlieVnet(
        motlie_vmm::backend::motlie::vnet::MotlieVnetBacking::new(),
    ))
}

#[cfg(not(target_os = "linux"))]
fn ch_network_backing() -> Result<NetworkBacking, DynError> {
    Err("backend 'ch' requires a Linux host build with Motlie VNET support".into())
}
