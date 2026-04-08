use std::fmt::Write as _;
use std::path::Path;

use thiserror::Error;

use crate::network::NetworkModes;
use crate::network_alloc::GuestNetAssignment;
use crate::spec::{GuestRuntimePaths, GuestSpec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudInitArtifacts {
    pub meta_data: String,
    pub user_data: String,
    pub mounts_yaml: String,
}

#[derive(Debug, Clone, Copy)]
pub struct LaunchArtifactRenderConfig<'a> {
    pub guest: &'a GuestSpec,
    pub runtime_paths: &'a GuestRuntimePaths,
    pub network_modes: NetworkModes,
    pub net_assignment: &'a GuestNetAssignment,
    pub base_dir: &'a Path,
    pub ssh_ca_pubkey: Option<&'a str>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    #[error("mount '{tag}' is missing guest_path; cannot render mounts.yaml")]
    MissingGuestPath { tag: String },
}

pub fn render_mounts_yaml(guest: &GuestSpec) -> Result<String, ArtifactError> {
    let mut out = String::from("mounts:\n");
    for mount in &guest.mounts {
        let Some(guest_path) = &mount.guest_path else {
            return Err(ArtifactError::MissingGuestPath {
                tag: mount.tag.clone(),
            });
        };
        writeln!(&mut out, "  - tag: {}", mount.tag).expect("writing to String cannot fail");
        writeln!(&mut out, "    guest_path: {}", guest_path.display())
            .expect("writing to String cannot fail");
        writeln!(&mut out, "    read_only: false").expect("writing to String cannot fail");
    }
    Ok(out)
}

pub fn render_cloud_init(guest: &GuestSpec) -> Result<String, ArtifactError> {
    let mut out = String::from("#cloud-config\n");
    writeln!(&mut out, "users:").expect("writing to String cannot fail");
    writeln!(&mut out, "  - name: {}", guest.user.name).expect("writing to String cannot fail");
    writeln!(&mut out, "    uid: {}", guest.user.uid).expect("writing to String cannot fail");
    writeln!(&mut out, "    gid: {}", guest.user.gid).expect("writing to String cannot fail");
    writeln!(&mut out, "    home: {}", guest.user.home.display())
        .expect("writing to String cannot fail");

    if !guest.software.packages.is_empty() {
        writeln!(&mut out, "packages:").expect("writing to String cannot fail");
        for package in &guest.software.packages {
            writeln!(&mut out, "  - {}", package).expect("writing to String cannot fail");
        }
    }

    Ok(out)
}

pub fn render_meta_data(guest_id: &str, hostname: &str) -> String {
    format!("instance-id: {guest_id}\nlocal-hostname: {hostname}\n")
}

pub fn render_cloud_init_artifacts(guest: &GuestSpec) -> Result<CloudInitArtifacts, ArtifactError> {
    Ok(CloudInitArtifacts {
        meta_data: render_meta_data(&guest.guest_id, &guest.hostname),
        user_data: render_cloud_init(guest)?,
        mounts_yaml: render_mounts_yaml(guest)?,
    })
}

pub fn render_launch_script(cfg: &LaunchArtifactRenderConfig<'_>) -> Result<String, ArtifactError> {
    let cloud_init = render_cloud_init(cfg.guest)?;
    let cloud_meta = render_meta_data(&cfg.guest.guest_id, &cfg.guest.hostname);
    let mounts_yaml = render_mounts_yaml(cfg.guest)?;
    let mut out = String::new();
    let login_home = cfg.guest.user.home.display().to_string();

    out.push_str("#!/usr/bin/env bash\n");
    out.push_str("set -euo pipefail\n\n");
    out.push_str("# Generated from library-owned guest/runtime state.\n\n");
    writeln!(
        &mut out,
        "GUEST_ID={}",
        shell_single_quote(&cfg.guest.guest_id)
    )
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "BASE_DIR=\"${{BASE_DIR:-{}}}\"",
        cfg.base_dir.display()
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "SEED_DIR=\"${{SEED_DIR:-{}}}\"",
        cfg.runtime_paths.cloud_init_dir.display()
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "ADMIN_NET={}",
        shell_single_quote(cfg.network_modes.admin.as_str())
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "EGRESS_NET={}",
        shell_single_quote(cfg.network_modes.egress.as_str())
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "VNET_SOCKET={}",
        shell_single_quote(cfg.runtime_paths.vnet_socket.to_string_lossy().as_ref())
    )
    .expect("writing to String cannot fail");
    if let Some(ca_pubkey) = cfg.ssh_ca_pubkey {
        writeln!(&mut out, "SSH_CA_PUBKEY={}", shell_single_quote(ca_pubkey))
            .expect("writing to String cannot fail");
    }
    out.push_str("INSTANCE_ID=\"${INSTANCE_ID:-${GUEST_ID}}\"\n");
    out.push_str("LOCAL_HOSTNAME=\"${LOCAL_HOSTNAME:-motlie-${GUEST_ID}}\"\n");
    out.push_str("mkdir -p \"$SEED_DIR\"\n");
    out.push_str("cat > \"$SEED_DIR/meta-data\" <<EOF\n");
    out.push_str(&cloud_meta);
    if !cloud_meta.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("EOF\n\n");
    out.push_str("cat > \"$SEED_DIR/mounts.yaml\" <<'EOF'\n");
    out.push_str(&mounts_yaml);
    out.push_str("EOF\n\n");
    out.push_str("cat > \"$SEED_DIR/user-data\" <<'EOF'\n");
    out.push_str(&cloud_init);
    if !cloud_init.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("EOF\n\n");
    out.push_str("echo \"Generated cloud-init assets in $SEED_DIR\"\n");
    out.push_str("echo \"Launching guest ${GUEST_ID} with seeded NoCloud dir ${SEED_DIR}\"\n");
    if matches!(cfg.network_modes.egress, crate::network::EgressNetMode::VhostUser) {
        out.push_str("echo \"Using motlie-vmm egress socket ${VNET_SOCKET}\"\n");
    }

    writeln!(&mut out, "GUEST_CID={}", cfg.net_assignment.cid)
        .expect("writing to String cannot fail");
    writeln!(&mut out, "HOST_IP={}", cfg.net_assignment.admin_ipv4.host)
        .expect("writing to String cannot fail");
    writeln!(&mut out, "GUEST_IP={}", cfg.net_assignment.admin_ipv4.guest)
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "ADMIN_MAC={}",
        shell_single_quote(&mac_fmt(&cfg.net_assignment.admin_mac))
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "EGRESS_MAC={}",
        shell_single_quote(&mac_fmt(&cfg.net_assignment.egress_mac))
    )
    .expect("writing to String cannot fail");
    writeln!(&mut out, "EGRESS_HOST_IP={}", cfg.net_assignment.egress_ipv4.host)
        .expect("writing to String cannot fail");
    writeln!(&mut out, "EGRESS_GUEST_IP={}", cfg.net_assignment.egress_ipv4.guest)
        .expect("writing to String cannot fail");
    writeln!(&mut out, "EGRESS_DNS_IP={}", cfg.net_assignment.egress_ipv4.dns)
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "SSH_USER={}",
        shell_single_quote(&cfg.guest.user.name)
    )
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "GUEST_HOSTNAME={}",
        shell_single_quote(&cfg.guest.hostname)
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "LOGIN_HOME={}",
        shell_single_quote(&login_home)
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "OVERLAY_SIZE={}",
        shell_single_quote(&cfg.guest.storage.overlay_size)
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "BOOT_KERNEL={}",
        shell_single_quote(cfg.guest.boot.kernel.to_string_lossy().as_ref())
    )
    .expect("writing to String cannot fail");
    if let Some(initramfs) = &cfg.guest.boot.initramfs {
        writeln!(
            &mut out,
            "BOOT_INITRAMFS={}",
            shell_single_quote(initramfs.to_string_lossy().as_ref())
        )
        .expect("writing to String cannot fail");
    }
    if let Some(firmware) = &cfg.guest.boot.firmware {
        writeln!(
            &mut out,
            "BOOT_FIRMWARE={}",
            shell_single_quote(firmware.to_string_lossy().as_ref())
        )
        .expect("writing to String cannot fail");
    }
    if let Some(cmdline) = &cfg.guest.boot.cmdline {
        writeln!(&mut out, "BOOT_CMDLINE_APPEND={}", shell_single_quote(cmdline))
            .expect("writing to String cannot fail");
    }
    out.push_str("LAUNCH_ARGS=(--guest \"$GUEST_ID\" --cloud-init-dir \"$SEED_DIR\" --admin-net \"$ADMIN_NET\" --egress-net \"$EGRESS_NET\")\n");
    out.push_str(
        "LAUNCH_ARGS+=(--cid \"$GUEST_CID\" --host-ip \"$HOST_IP\" --guest-ip \"$GUEST_IP\")\n",
    );
    out.push_str("LAUNCH_ARGS+=(--admin-mac \"$ADMIN_MAC\" --egress-mac \"$EGRESS_MAC\")\n");
    out.push_str(
        "LAUNCH_ARGS+=(--egress-host-ip \"$EGRESS_HOST_IP\" --egress-guest-ip \"$EGRESS_GUEST_IP\" --egress-dns-ip \"$EGRESS_DNS_IP\")\n",
    );
    out.push_str("LAUNCH_ARGS+=(--ssh-user \"$SSH_USER\" --hostname \"$GUEST_HOSTNAME\" --login-home \"$LOGIN_HOME\")\n");
    out.push_str("LAUNCH_ARGS+=(--overlay-size \"$OVERLAY_SIZE\")\n");
    out.push_str("LAUNCH_ARGS+=(--vnet-socket \"$VNET_SOCKET\")\n");
    if cfg.ssh_ca_pubkey.is_some() {
        out.push_str("LAUNCH_ARGS+=(--ssh-ca-pubkey \"$SSH_CA_PUBKEY\")\n");
    }
    out.push_str("# Boot artifact variables are rendered here for backend consumption.\n");
    out.push_str("\"$BASE_DIR/launch-ch.sh\" \"${LAUNCH_ARGS[@]}\" \"$@\"\n");

    Ok(out)
}

fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn mac_fmt(m: &[u8; 6]) -> String {
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        m[0], m[1], m[2], m[3], m[4], m[5]
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::network::{AdminNetMode, EgressNetMode, NetworkModes};
    use crate::network_alloc::{AdminIpv4Pair, EgressIpv4Layout, GuestNetAssignment};
    use crate::spec::{
        BootArtifacts, GuestMountSpec, GuestResources, GuestRuntimePaths, GuestSpec,
        GuestSshAccess, GuestStorage, GuestUser, RuntimeNamespace, SoftwareProfile,
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
                packages: vec!["vim".to_string(), "gh".to_string()],
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

    fn sample_paths() -> GuestRuntimePaths {
        let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp").unwrap();
        GuestRuntimePaths::for_guest(&namespace, "alice").unwrap()
    }

    fn sample_net() -> GuestNetAssignment {
        GuestNetAssignment {
            guest_name: "alice".to_string(),
            slot: 0,
            cid: 3,
            admin_ipv4: AdminIpv4Pair {
                host: "192.168.249.1".parse().unwrap(),
                guest: "192.168.249.2".parse().unwrap(),
            },
            admin_mac: [0x52, 0x54, 0x00, 0xad, 0x00, 0x01],
            egress_ipv4: EgressIpv4Layout {
                guest: "10.0.2.15".parse().unwrap(),
                host: "10.0.2.2".parse().unwrap(),
                dns: "10.0.2.3".parse().unwrap(),
                netmask: "255.255.255.0".parse().unwrap(),
            },
            egress_mac: [0x52, 0x54, 0x00, 0xe9, 0x00, 0x01],
            vnet_socket_path: PathBuf::from("/tmp/motlie-vmm-v14-alice.sock"),
        }
    }

    #[test]
    fn mounts_yaml_renders_guest_paths() {
        let yaml = render_mounts_yaml(&sample_guest()).unwrap();
        assert!(yaml.contains("tag: alice-home"));
        assert!(yaml.contains("guest_path: /home/alice"));
    }

    #[test]
    fn mounts_yaml_rejects_missing_guest_path() {
        let mut guest = sample_guest();
        guest.mounts[0].guest_path = None;
        let err = render_mounts_yaml(&guest).unwrap_err();
        assert_eq!(
            err,
            ArtifactError::MissingGuestPath {
                tag: "alice-home".to_string()
            }
        );
    }

    #[test]
    fn cloud_init_uses_reviewed_guest_shape() {
        let guest = sample_guest();
        let rendered = render_cloud_init(&guest).unwrap();
        assert!(rendered.contains("name: alice"));
        assert!(rendered.contains("packages:"));
        assert!(rendered.contains("  - vim"));
        assert!(rendered.contains("  - gh"));
    }

    #[test]
    fn launch_script_renders_expected_paths_and_network() {
        let guest = sample_guest();
        let paths = sample_paths();
        let net = sample_net();
        let script = render_launch_script(&LaunchArtifactRenderConfig {
            guest: &guest,
            runtime_paths: &paths,
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
            net_assignment: &net,
            base_dir: Path::new("/tmp/vmm-v1.4/libs/vmm/examples/v1.4"),
            ssh_ca_pubkey: Some("ssh-ed25519 AAAA-test"),
        })
        .unwrap();

        assert!(script.contains("GUEST_ID='alice'"));
        assert!(script.contains("SEED_DIR=\"${SEED_DIR:-/tmp/motlie-vmm-v14-cloud-init-alice}\""));
        assert!(script.contains("GUEST_CID=3"));
        assert!(script.contains("ADMIN_NET='none'"));
        assert!(script.contains("EGRESS_NET='vhost-user'"));
        assert!(script.contains("VNET_SOCKET='/tmp/motlie-vmm-v14-alice.sock'"));
        assert!(script.contains("OVERLAY_SIZE='2G'"));
        assert!(script.contains("BOOT_KERNEL='/tmp/Image'"));
        assert!(script.contains("BOOT_CMDLINE_APPEND='console=ttyS0'"));
        assert!(script.contains("LAUNCH_ARGS+=(--overlay-size \"$OVERLAY_SIZE\")"));
        assert!(script.contains("launch-ch.sh"));
    }
}
