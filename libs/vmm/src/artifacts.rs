use std::fmt::Write as _;
use std::path::Path;

use thiserror::Error;

use crate::backend::vz;
use crate::backend::BackendKind;
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
    pub backend_kind: BackendKind,
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
    writeln!(&mut out, "apt:").expect("writing to String cannot fail");
    writeln!(&mut out, "  preserve_sources_list: true").expect("writing to String cannot fail");
    writeln!(&mut out, "users:").expect("writing to String cannot fail");
    writeln!(&mut out, "  - name: {}", guest.user.name).expect("writing to String cannot fail");
    writeln!(&mut out, "    uid: {}", guest.user.uid).expect("writing to String cannot fail");
    writeln!(&mut out, "    gid: {}", guest.user.gid).expect("writing to String cannot fail");
    writeln!(&mut out, "    home: {}", guest.user.home.display())
        .expect("writing to String cannot fail");
    writeln!(&mut out, "    shell: /bin/bash").expect("writing to String cannot fail");
    writeln!(&mut out, "    groups: [sudo]").expect("writing to String cannot fail");
    writeln!(&mut out, "    sudo: ALL=(ALL) NOPASSWD:ALL").expect("writing to String cannot fail");
    writeln!(&mut out, "    lock_passwd: true").expect("writing to String cannot fail");

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
    if let Some(runtime_root) = cfg.runtime_paths.runtime_dir.parent() {
        writeln!(
            &mut out,
            "RUNTIME_ROOT=\"${{RUNTIME_ROOT:-{}}}\"",
            runtime_root.display()
        )
        .expect("writing to String cannot fail");
    }
    writeln!(
        &mut out,
        "API_SOCKET=\"${{API_SOCKET:-{}}}\"",
        cfg.runtime_paths.api_socket.display()
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "VSOCK_SOCKET=\"${{VSOCK_SOCKET:-{}}}\"",
        cfg.runtime_paths.vsock_socket.display()
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
    out.push_str("export RUNTIME_ROOT API_SOCKET VSOCK_SOCKET\n");
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
    if matches!(
        cfg.network_modes.egress,
        crate::network::EgressNetMode::VhostUser
    ) {
        out.push_str("echo \"Using motlie-vmm egress socket ${VNET_SOCKET}\"\n");
    }
    if matches!(
        cfg.network_modes.egress,
        crate::network::EgressNetMode::VzUserspace
    ) {
        out.push_str("echo \"Using Apple Vz userspace egress\"\n");
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
    writeln!(
        &mut out,
        "EGRESS_HOST_IP={}",
        cfg.net_assignment.egress_ipv4.host
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "EGRESS_GUEST_IP={}",
        cfg.net_assignment.egress_ipv4.guest
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "EGRESS_DNS_IP={}",
        cfg.net_assignment.egress_ipv4.dns
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "SSH_LOGIN_USER={}",
        shell_single_quote(&cfg.guest.ssh.login_user)
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "SSH_PRINCIPAL={}",
        shell_single_quote(&cfg.guest.ssh.principal)
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "GUEST_HOSTNAME={}",
        shell_single_quote(&cfg.guest.hostname)
    )
    .expect("writing to String cannot fail");
    writeln!(&mut out, "LOGIN_HOME={}", shell_single_quote(&login_home))
        .expect("writing to String cannot fail");
    writeln!(&mut out, "LOGIN_UID={}", cfg.guest.user.uid).expect("writing to String cannot fail");
    writeln!(&mut out, "LOGIN_GID={}", cfg.guest.user.gid).expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "OVERLAY_SIZE={}",
        shell_single_quote(cfg.guest.storage.overlay_size.as_ref())
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
        writeln!(
            &mut out,
            "BOOT_CMDLINE_APPEND={}",
            shell_single_quote(cmdline)
        )
        .expect("writing to String cannot fail");
    }
    if cfg.backend_kind == BackendKind::Vz {
        let vz_artifacts_dir = vz::artifacts_dir(cfg.runtime_paths);
        let vz_vm_name = vz::vm_name(cfg.runtime_paths, &cfg.guest.guest_id);
        if cfg.ssh_ca_pubkey.is_some() {
            out.push_str("export MOTLIE_VZ_SSH_CA_PUBKEY=\"$SSH_CA_PUBKEY\"\n");
        }
        out.push_str(
            "# Vz convergence contract: first-contact SSH waits for interactive-ready only.\n",
        );
        out.push_str(
            "# Full VFS/VNET/egress certification stays in harness validate/scenario steps.\n",
        );
        out.push_str("export MOTLIE_VZ_INLINE_VALIDATION=0\n");
        out.push_str(
            "# Vz egress is VMM-owned runtime state; launch-vz.sh must only consume the socket.\n",
        );
        out.push_str("export MOTLIE_VZ_EMBEDDED_EGRESS=1\n");
        export_path(&mut out, "MOTLIE_VZ_ARTIFACTS_DIR", &vz_artifacts_dir);
        export_path(
            &mut out,
            "MOTLIE_VZ_EGRESS_SOCKET_PATH",
            &cfg.runtime_paths.vnet_socket,
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_RUNNER_PID_FILE",
            &cfg.runtime_paths.runtime_dir.join("vz-runner.pid"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_RESULT_JSON",
            &cfg.runtime_paths.runtime_dir.join("vz-launch-result.json"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_SERIAL_LOG",
            &cfg.runtime_paths.serial_log,
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_GUEST_IP_FILE",
            &cfg.runtime_paths.runtime_dir.join("guest-ip.txt"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_SEED_DIR",
            &cfg.runtime_paths.runtime_dir.join("seed"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_SEED_IMAGE",
            &cfg.runtime_paths.runtime_dir.join("seed.dmg"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_CONTROL_READY_FILE",
            &cfg.runtime_paths.runtime_dir.join("control-plane-ready"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_INTERACTIVE_READY_FILE",
            &cfg.runtime_paths.runtime_dir.join("interactive-ready"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_VALIDATION_COMPLETE_FILE",
            &cfg.runtime_paths.runtime_dir.join("validation-complete"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_PHASES_LOG",
            &cfg.runtime_paths.runtime_dir.join("vz-phases.log"),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_CONTROL_PORT_FILE",
            &vz::egress::control_port_file(cfg.runtime_paths),
        );
        export_path(
            &mut out,
            "MOTLIE_VZ_VFS_VSOCK_SOCKET",
            &cfg.guest.socket_path,
        );
        export_value(
            &mut out,
            "MOTLIE_VZ_SSH_VSOCK_SOCKET",
            &format!("{}_2222", cfg.runtime_paths.vsock_socket.to_string_lossy()),
        );
        if let Some(home_mount) = cfg
            .guest
            .mounts
            .iter()
            .find(|mount| mount.guest_path.as_deref() == Some(cfg.guest.user.home.as_path()))
        {
            export_path(&mut out, "MOTLIE_VZ_HOST_HOME_DIR", &home_mount.host_path);
        }
        if let Some(workspace_mount) = cfg
            .guest
            .mounts
            .iter()
            .find(|mount| mount.guest_path.as_deref() == Some(Path::new("/workspace")))
        {
            export_path(
                &mut out,
                "MOTLIE_VZ_HOST_WORKSPACE_DIR",
                &workspace_mount.host_path,
            );
        }
        if let Some(agent_state_mount) = cfg
            .guest
            .mounts
            .iter()
            .find(|mount| mount.guest_path.as_deref() == Some(Path::new("/agent-state")))
        {
            export_path(
                &mut out,
                "MOTLIE_VZ_HOST_AGENT_STATE_DIR",
                &agent_state_mount.host_path,
            );
        }
        out.push_str("export MOTLIE_VZ_LOGIN_USER=\"$SSH_LOGIN_USER\"\n");
        out.push_str("export MOTLIE_VZ_UID_NUM=\"$LOGIN_UID\"\n");
        out.push_str("export MOTLIE_VZ_GID_NUM=\"$LOGIN_GID\"\n");
        out.push_str("export MOTLIE_VZ_GUEST_HOSTNAME=\"$GUEST_HOSTNAME\"\n");
        out.push_str("export MOTLIE_VZ_MOUNTS_FILE=\"$SEED_DIR/mounts.yaml\"\n");
        out.push_str("export MOTLIE_VZ_NET_MAC=\"$EGRESS_MAC\"\n");
        out.push_str("export MOTLIE_VZ_EGRESS_GUEST_IP=\"$EGRESS_GUEST_IP\"\n");
        writeln!(&mut out, "VZ_VM_NAME={}", shell_single_quote(&vz_vm_name))
            .expect("writing to String cannot fail");
        out.push_str("export MOTLIE_VZ_KEEP_RUNNING=1\n");
        out.push_str(
            "\"$BASE_DIR/launch-vz.sh\" --guest \"$GUEST_ID\" --vm-name \"$VZ_VM_NAME\" \"$@\"\n",
        );
    } else {
        if let Some(base_artifacts) = cfg.guest.boot.kernel.parent() {
            export_path(&mut out, "BASE_ARTIFACTS", base_artifacts);
        }
        out.push_str("LAUNCH_ARGS=(--guest \"$GUEST_ID\" --cloud-init-dir \"$SEED_DIR\" --admin-net \"$ADMIN_NET\" --egress-net \"$EGRESS_NET\")\n");
        out.push_str(
            "LAUNCH_ARGS+=(--cid \"$GUEST_CID\" --host-ip \"$HOST_IP\" --guest-ip \"$GUEST_IP\")\n",
        );
        out.push_str("LAUNCH_ARGS+=(--admin-mac \"$ADMIN_MAC\" --egress-mac \"$EGRESS_MAC\")\n");
        out.push_str(
            "LAUNCH_ARGS+=(--egress-host-ip \"$EGRESS_HOST_IP\" --egress-guest-ip \"$EGRESS_GUEST_IP\" --egress-dns-ip \"$EGRESS_DNS_IP\")\n",
        );
        out.push_str(
            "LAUNCH_ARGS+=(--ssh-user \"$SSH_LOGIN_USER\" --ssh-principal \"$SSH_PRINCIPAL\")\n",
        );
        out.push_str(
            "LAUNCH_ARGS+=(--hostname \"$GUEST_HOSTNAME\" --login-home \"$LOGIN_HOME\")\n",
        );
        out.push_str("LAUNCH_ARGS+=(--overlay-size \"$OVERLAY_SIZE\")\n");
        out.push_str("LAUNCH_ARGS+=(--vnet-socket \"$VNET_SOCKET\")\n");
        if cfg.ssh_ca_pubkey.is_some() {
            out.push_str("LAUNCH_ARGS+=(--ssh-ca-pubkey \"$SSH_CA_PUBKEY\")\n");
        }
        out.push_str("# Boot artifact variables are rendered here for backend consumption.\n");
        out.push_str("\"$BASE_DIR/launch-ch.sh\" \"${LAUNCH_ARGS[@]}\" \"$@\"\n");
    }

    Ok(out)
}

fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn export_value(out: &mut String, name: &str, value: &str) {
    writeln!(out, "export {name}={}", shell_single_quote(value))
        .expect("writing to String cannot fail");
}

fn export_path(out: &mut String, name: &str, path: &Path) {
    export_value(out, name, path.to_string_lossy().as_ref());
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
            socket_path: std::path::PathBuf::from("/tmp/motlie-vmm-v14-alice.vsock_5000"),
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
            admin_subnet: "172.20.0.0/30".parse().unwrap(),
            admin_ipv4: AdminIpv4Pair {
                host: "172.20.0.1".parse().unwrap(),
                guest: "172.20.0.2".parse().unwrap(),
            },
            admin_mac: [0x52, 0x54, 0x00, 0xa0, 0x00, 0x00],
            egress_subnet: "10.0.0.0/24".parse().unwrap(),
            egress_ipv4: EgressIpv4Layout {
                guest: "10.0.0.15".parse().unwrap(),
                host: "10.0.0.2".parse().unwrap(),
                dns: "10.0.0.3".parse().unwrap(),
                netmask: "255.255.255.0".parse().unwrap(),
            },
            egress_mac: [0x52, 0x54, 0x00, 0xe0, 0x00, 0x00],
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
        assert!(rendered.contains("preserve_sources_list: true"));
        assert!(rendered.contains("name: alice"));
        assert!(rendered.contains("sudo: ALL=(ALL) NOPASSWD:ALL"));
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
            backend_kind: BackendKind::ChShell,
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
        assert!(script.contains("RUNTIME_ROOT=\"${RUNTIME_ROOT:-/tmp/motlie-vmm-v14-runtime}\""));
        assert!(script.contains("API_SOCKET=\"${API_SOCKET:-/tmp/motlie-vmm-v14-alice-api.sock}\""));
        assert!(
            script.contains("VSOCK_SOCKET=\"${VSOCK_SOCKET:-/tmp/motlie-vmm-v14-alice.vsock}\"")
        );
        assert!(script.contains("GUEST_CID=3"));
        assert!(script.contains("ADMIN_NET='none'"));
        assert!(script.contains("EGRESS_NET='vhost-user'"));
        assert!(script.contains("VNET_SOCKET='/tmp/motlie-vmm-v14-alice.sock'"));
        assert!(script.contains("OVERLAY_SIZE='2G'"));
        assert!(script.contains("BOOT_KERNEL='/tmp/Image'"));
        assert!(script.contains("export BASE_ARTIFACTS='/tmp'"));
        assert!(script.contains("BOOT_CMDLINE_APPEND='console=ttyS0'"));
        assert!(script.contains("LAUNCH_ARGS+=(--overlay-size \"$OVERLAY_SIZE\")"));
        assert!(script.contains("launch-ch.sh"));
    }

    #[test]
    fn launch_script_uses_vz_launcher_for_vz_backend() {
        let guest = sample_guest();
        let paths = sample_paths();
        let net = sample_net();
        let script = render_launch_script(&LaunchArtifactRenderConfig {
            guest: &guest,
            runtime_paths: &paths,
            backend_kind: BackendKind::Vz,
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VzUserspace,
            },
            net_assignment: &net,
            base_dir: Path::new("/tmp/vmm-v1.45/libs/vmm/examples/v1.45"),
            ssh_ca_pubkey: Some("ssh-ed25519 AAAA-test"),
        })
        .unwrap();

        assert!(script.contains("ADMIN_NET='none'"));
        assert!(script.contains("EGRESS_NET='vz-userspace'"));
        assert!(script.contains("export MOTLIE_VZ_SSH_CA_PUBKEY=\"$SSH_CA_PUBKEY\""));
        assert!(script.contains("export MOTLIE_VZ_INLINE_VALIDATION=0"));
        assert!(script.contains("export MOTLIE_VZ_EMBEDDED_EGRESS=1"));
        assert!(script.contains(
            "export MOTLIE_VZ_ARTIFACTS_DIR='/tmp/motlie-vmm-v14-runtime/alice/vz-artifacts'"
        ));
        assert!(
            script.contains("export MOTLIE_VZ_EGRESS_SOCKET_PATH='/tmp/motlie-vmm-v14-alice.sock'")
        );
        assert!(script.contains(
            "export MOTLIE_VZ_RUNNER_PID_FILE='/tmp/motlie-vmm-v14-runtime/alice/vz-runner.pid'"
        ));
        assert!(script.contains(
            "export MOTLIE_VZ_RESULT_JSON='/tmp/motlie-vmm-v14-runtime/alice/vz-launch-result.json'"
        ));
        assert!(script.contains(
            "export MOTLIE_VZ_INTERACTIVE_READY_FILE='/tmp/motlie-vmm-v14-runtime/alice/interactive-ready'"
        ));
        assert!(script.contains(
            "export MOTLIE_VZ_VALIDATION_COMPLETE_FILE='/tmp/motlie-vmm-v14-runtime/alice/validation-complete'"
        ));
        assert!(script.contains(
            "export MOTLIE_VZ_PHASES_LOG='/tmp/motlie-vmm-v14-runtime/alice/vz-phases.log'"
        ));
        assert!(script.contains(
            "export MOTLIE_VZ_CONTROL_PORT_FILE='/tmp/motlie-vmm-v14-runtime/alice/control-port'"
        ));
        assert!(script
            .contains("export MOTLIE_VZ_VFS_VSOCK_SOCKET='/tmp/motlie-vmm-v14-alice.vsock_5000'"));
        assert!(script
            .contains("export MOTLIE_VZ_SSH_VSOCK_SOCKET='/tmp/motlie-vmm-v14-alice.vsock_2222'"));
        assert!(script.contains("export MOTLIE_VZ_EGRESS_GUEST_IP=\"$EGRESS_GUEST_IP\""));
        assert!(script.contains("VZ_VM_NAME='motlie-v1-45-motlie-vmm-v14-runtime-alice'"));
        assert!(script.contains("launch-vz.sh"));
        assert!(script.contains("--vm-name \"$VZ_VM_NAME\""));
        assert!(!script.contains("launch-ch.sh"));
    }

    #[test]
    fn launch_script_backend_not_egress_mode_selects_launcher() {
        let guest = sample_guest();
        let paths = sample_paths();
        let net = sample_net();
        let script = render_launch_script(&LaunchArtifactRenderConfig {
            guest: &guest,
            runtime_paths: &paths,
            backend_kind: BackendKind::ChShell,
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VzUserspace,
            },
            net_assignment: &net,
            base_dir: Path::new("/tmp/vmm-v1.45/libs/vmm/examples/v1.45"),
            ssh_ca_pubkey: Some("ssh-ed25519 AAAA-test"),
        })
        .unwrap();

        assert!(script.contains("launch-ch.sh"));
        assert!(!script.contains("launch-vz.sh"));
    }
}
