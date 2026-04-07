use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use motlie_vnet::{VnetBackend, VnetConfig, VnetHandle};
use motlie_vmm::backend::BackendKind;
use motlie_vmm::ca::SshCa;
use motlie_vmm::guestfs::GuestFsHandle;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::orchestrator::{PrepareRequest, PreparedGuest, ReadinessPolicy, boot, prepare};
use motlie_vmm::spec::{
    BootArtifacts, GuestMountSpec, GuestResources, GuestSpec, GuestSshAccess, GuestStorage,
    GuestUser, RuntimeNamespace, SoftwareProfile,
};
use motlie_vmm::ssh::{
    self, ExecOutput, GuestRegistry, SshProxyConfig, exec_on_guest, new_guest_registry,
    wait_for_guest_bridge_ready,
};
use tokio::time::sleep;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), DynError> {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.4");
    let artifacts_dir = base_dir.join("artifacts/base");
    ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))?;
    ensure_file_exists(&artifacts_dir.join("Image"))?;

    let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp")?;
    let demo_root = PathBuf::from("/tmp/motlie-vmm-v14-demo");
    let guest = demo_guest("alice", &artifacts_dir, &demo_root);
    let socket_root = namespace.temp_root.join(format!("{}-sockets", namespace.prefix));
    std::fs::create_dir_all(&socket_root)?;

    seed_host_mounts(&guest)?;

    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();
    let proxy_config = SshProxyConfig {
        listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 2224),
    };
    tokio::spawn(ssh::run_proxy(proxy_config.clone(), Arc::clone(&guest_registry)));

    let guestfs = GuestFsHandle::provision(&guest).await?;
    let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig {
        socket_dir: socket_root,
        ..GuestNetAllocatorConfig::default()
    });
    let prepared = prepare(
        PrepareRequest {
            guest,
            namespace,
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
            backend_kind: BackendKind::ChShell,
            base_dir: base_dir.clone(),
            ssh_ca_pubkey: Some(ca.public_key_openssh()?),
        },
        &mut allocator,
    )?;
    let mut vnet = Some(start_vnet_backend(&prepared)?);

    let (ssh_listener, _uds_path) = ssh::bind_vsock_ssh_listener(
        prepared.runtime_paths.vsock_socket.to_string_lossy().as_ref(),
    )?;
    tokio::spawn(ssh::run_guest_ssh_bridge(
        ssh_listener,
        ssh::vsock_ssh_uds_path(prepared.runtime_paths.vsock_socket.to_string_lossy().as_ref()),
        Arc::clone(&ca),
        prepared.guest.guest_id.clone(),
        Arc::clone(&guest_registry),
    ));

    let handle = boot(prepared)?;
    handle.ready(&ReadinessPolicy::default()).await?;
    guestfs.wait_until_ready(Duration::from_secs(20)).await?;
    wait_for_guest_bridge_ready(&guest_registry, &handle.guest_id, Duration::from_secs(20)).await?;

    let hello = exec_on_guest(
        &guest_registry,
        &handle.guest_id,
        "/bin/echo hello",
        Duration::from_secs(10),
    )
    .await?;
    ensure_success(&hello.stdout, "hello")?;

    let vfs = exec_on_guest(
        &guest_registry,
        &handle.guest_id,
        "/bin/sh -lc 'pwd && test -d /home/alice && test -d /workspace && test -d /agent-state && grep -q \"Alice workspace mounted from the host.\" /workspace/README.md && echo VFS_OK'",
        Duration::from_secs(10),
    )
    .await?;
    ensure_success(&vfs.stdout, "VFS_OK")?;

    let route = exec_until_success(
        &guest_registry,
        &handle.guest_id,
        r#"/bin/sh -lc 'ip route | grep -q "^default via 10.0.2.2 " && echo ROUTE_OK'"#,
        "ROUTE_OK",
        Duration::from_secs(10),
    )
    .await?;
    ensure_success(&route.stdout, "ROUTE_OK")?;

    let outbound = exec_until_success(
        &guest_registry,
        &handle.guest_id,
        r#"/bin/sh -lc 'code=$(curl -s -o /dev/null -w "%{http_code}" https://example.com); test "$code" = 200 && echo HTTPS_OK'"#,
        "HTTPS_OK",
        Duration::from_secs(20),
    )
    .await?;
    ensure_success(&outbound.stdout, "HTTPS_OK")?;

    let report = handle.shutdown().await?;
    if let Some(vnet) = vnet.as_mut() {
        vnet.shutdown()?;
    }
    println!(
        "v1.4 harness smoke passed: guest={} pid={:?} forced={:?} proxy=127.0.0.1:{}",
        handle.guest_id,
        report.pid,
        report.forced,
        proxy_config.listen.port()
    );
    Ok(())
}

fn ensure_file_exists(path: &Path) -> Result<(), DynError> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("required artifact missing: {}", path.display()).into())
    }
}

fn ensure_success(stdout: &str, needle: &str) -> Result<(), DynError> {
    if stdout.contains(needle) {
        Ok(())
    } else {
        Err(format!("expected output to contain '{needle}', got: {stdout}").into())
    }
}

fn demo_guest(guest_id: &str, artifacts_dir: &Path, demo_root: &Path) -> GuestSpec {
    GuestSpec {
        guest_id: guest_id.to_string(),
        hostname: format!("motlie-{guest_id}"),
        socket_path: format!("/tmp/motlie-vmm-v14-{guest_id}.vsock_5000"),
        user: GuestUser {
            name: guest_id.to_string(),
            uid: 1000,
            gid: 1000,
            home: PathBuf::from(format!("/home/{guest_id}")),
        },
        ssh: GuestSshAccess {
            principal: guest_id.to_string(),
            login_user: guest_id.to_string(),
        },
        mounts: vec![
            GuestMountSpec {
                tag: format!("{guest_id}-home"),
                guest_path: Some(PathBuf::from(format!("/home/{guest_id}"))),
                host_path: demo_root.join(format!("{guest_id}-home")),
            },
            GuestMountSpec {
                tag: format!("{guest_id}-workspace"),
                guest_path: Some(PathBuf::from("/workspace")),
                host_path: demo_root.join(format!("{guest_id}-workspace")),
            },
            GuestMountSpec {
                tag: format!("{guest_id}-agent-state"),
                guest_path: Some(PathBuf::from("/agent-state")),
                host_path: demo_root.join(format!("{guest_id}-agent-state")),
            },
        ],
        software: SoftwareProfile {
            packages: vec!["vim".to_string()],
        },
        resources: GuestResources::default(),
        storage: GuestStorage::default(),
        boot: BootArtifacts {
            kernel: artifacts_dir.join("Image"),
            initramfs: None,
            firmware: None,
            cmdline: None,
        },
    }
}

fn seed_host_mounts(guest: &GuestSpec) -> Result<(), DynError> {
    for mount in &guest.mounts {
        std::fs::create_dir_all(&mount.host_path)?;
    }
    std::fs::write(
        guest.mounts[1].host_path.join("README.md"),
        "Alice workspace mounted from the host.\n",
    )?;
    Ok(())
}

fn start_vnet_backend(prepared: &PreparedGuest) -> Result<VnetHandle, DynError> {
    if let Some(parent) = prepared.runtime_paths.vnet_socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let config = VnetConfig::builder()
        .socket_path(&prepared.runtime_paths.vnet_socket)
        .guest_ipv4(prepared.net_assignment.egress_ipv4.guest)
        .host_ipv4(prepared.net_assignment.egress_ipv4.host)
        .netmask(prepared.net_assignment.egress_ipv4.netmask)
        .dns_ipv4(prepared.net_assignment.egress_ipv4.dns)
        .mac(prepared.net_assignment.egress_mac)
        .build()?;
    Ok(VnetBackend::new(config).start()?)
}

async fn exec_until_success(
    guest_registry: &GuestRegistry,
    guest_id: &str,
    command: &str,
    needle: &str,
    timeout: Duration,
) -> Result<ExecOutput, DynError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_output: Option<ExecOutput> = None;
    loop {
        match exec_on_guest(guest_registry, guest_id, command, Duration::from_secs(10)).await {
            Ok(output) if output.exit_code == 0 && output.stdout.contains(needle) => return Ok(output),
            Ok(output) => last_output = Some(output),
            Err(_) => {}
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(match last_output {
                Some(output) => format!(
                    "timed out waiting for command success: cmd={command} exit={} stdout={} stderr={}",
                    output.exit_code, output.stdout, output.stderr
                )
                .into(),
                None => format!("timed out waiting for command success: cmd={command}").into(),
            });
        }

        sleep(Duration::from_secs(1)).await;
    }
}
