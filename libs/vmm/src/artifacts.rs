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
    #[error("guest '{guest_name}' is missing uid/gid; provision with explicit uid/gid")]
    MissingIdentity { guest_name: String },
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
    let _identity = guest.identity.ok_or_else(|| ArtifactError::MissingIdentity {
        guest_name: guest.name.clone(),
    })?;

    Ok("#cloud-config\n".to_string())
}

pub fn render_meta_data(guest_name: &str) -> String {
    let hostname = guest_hostname(guest_name);
    format!("instance-id: {guest_name}\nlocal-hostname: {hostname}\n")
}

pub fn render_cloud_init_artifacts(guest: &GuestSpec) -> Result<CloudInitArtifacts, ArtifactError> {
    Ok(CloudInitArtifacts {
        meta_data: render_meta_data(&guest.name),
        user_data: render_cloud_init(guest)?,
        mounts_yaml: render_mounts_yaml(guest)?,
    })
}

pub fn render_launch_script(cfg: &LaunchArtifactRenderConfig<'_>) -> Result<String, ArtifactError> {
    let cloud_init = render_cloud_init(cfg.guest)?;
    let cloud_meta = render_meta_data(&cfg.guest.name);
    let mounts_yaml = render_mounts_yaml(cfg.guest)?;
    let mut out = String::new();
    let login_home = format!("/home/{}", cfg.guest.name);

    out.push_str("#!/usr/bin/env bash\n");
    out.push_str("set -euo pipefail\n\n");
    out.push_str("# Generated from library-owned guest/runtime state.\n\n");
    writeln!(&mut out, "GUEST_ID={}", shell_single_quote(&cfg.guest.name))
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
    writeln!(&mut out, "SSH_USER={}", shell_single_quote(&cfg.guest.name))
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "GUEST_HOSTNAME={}",
        shell_single_quote(&guest_hostname(&cfg.guest.name))
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "LOGIN_HOME={}",
        shell_single_quote(&login_home)
    )
    .expect("writing to String cannot fail");
    out.push_str("LAUNCH_ARGS=(--guest \"$GUEST_ID\" --cloud-init-dir \"$SEED_DIR\" --admin-net \"$ADMIN_NET\" --egress-net \"$EGRESS_NET\")\n");
    out.push_str(
        "LAUNCH_ARGS+=(--cid \"$GUEST_CID\" --host-ip \"$HOST_IP\" --guest-ip \"$GUEST_IP\")\n",
    );
    out.push_str("LAUNCH_ARGS+=(--admin-mac \"$ADMIN_MAC\" --egress-mac \"$EGRESS_MAC\")\n");
    out.push_str("LAUNCH_ARGS+=(--ssh-user \"$SSH_USER\" --hostname \"$GUEST_HOSTNAME\" --login-home \"$LOGIN_HOME\")\n");
    out.push_str("LAUNCH_ARGS+=(--vnet-socket \"$VNET_SOCKET\")\n");
    if cfg.ssh_ca_pubkey.is_some() {
        out.push_str("LAUNCH_ARGS+=(--ssh-ca-pubkey \"$SSH_CA_PUBKEY\")\n");
    }
    out.push_str("\"$BASE_DIR/launch-ch.sh\" \"${LAUNCH_ARGS[@]}\" \"$@\"\n");

    Ok(out)
}

fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn guest_hostname(guest_name: &str) -> String {
    format!("motlie-{guest_name}")
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
    use crate::spec::{GuestIdentity, GuestMountSpec, GuestRuntimePaths, GuestSpec, RuntimeNamespace};

    use super::*;

    fn sample_guest() -> GuestSpec {
        GuestSpec {
            name: "alice".to_string(),
            socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".to_string(),
            mounts: vec![GuestMountSpec {
                tag: "alice-home".to_string(),
                guest_path: Some(PathBuf::from("/home/alice")),
                host_path: PathBuf::from("/tmp/demo/alice-home"),
            }],
            identity: Some(GuestIdentity { uid: 1000, gid: 1000 }),
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
    fn cloud_init_requires_identity() {
        let mut guest = sample_guest();
        guest.identity = None;
        let err = render_cloud_init(&guest).unwrap_err();
        assert_eq!(
            err,
            ArtifactError::MissingIdentity {
                guest_name: "alice".to_string()
            }
        );
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
        assert!(script.contains("launch-ch.sh"));
    }
}
