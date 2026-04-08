mod pty;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::time::Duration;

use motlie_vmm::ca::SshCa;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::orchestrator::{
    LifecycleServices, PrepareRequest, ReadinessPolicy, boot, prepare,
};
use motlie_vmm::runtime::{
    ControlPlaneBacking, FilesystemBacking, HypervisorBacking, NetworkBacking, Runtime,
};
use motlie_vmm::spec::{
    BootArtifacts, GuestMountSpec, GuestResources, GuestSpec, GuestSshAccess, GuestStorage,
    GuestUser, RuntimeNamespace, SoftwareProfile,
};
use motlie_vmm::ssh::{self, ExecOutput, SshProxyConfig, new_guest_registry};
use tokio::time::sleep;

type DynError = Box<dyn std::error::Error + Send + Sync>;

struct HarnessInstance {
    namespace: RuntimeNamespace,
    demo_root: PathBuf,
    socket_root: PathBuf,
    proxy_port: u16,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), DynError> {
    let scenario = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "smoke".to_string());

    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.4");
    let artifacts_dir = base_dir.join("artifacts/base");
    ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))?;
    ensure_file_exists(&artifacts_dir.join("Image"))?;

    let instance = new_harness_instance()?;
    let guest = demo_guest("alice", &artifacts_dir, &instance.demo_root, &instance.namespace);
    std::fs::create_dir_all(&instance.socket_root)?;

    seed_host_mounts(&guest)?;

    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();
    let proxy_config = SshProxyConfig {
        listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), instance.proxy_port),
    };
    tokio::spawn(ssh::run_proxy(proxy_config.clone(), Arc::clone(&guest_registry)));
    print_instance_details(&instance, &proxy_config);
    let runtime = Arc::new(Runtime {
        hypervisor: HypervisorBacking::CloudHypervisorShell(
            motlie_vmm::backend::ch::shell::ChShellBackend::new(),
        ),
        filesystem: FilesystemBacking::MotlieVfs(
            motlie_vmm::backend::motlie::vfs::MotlieVfsBacking::new(),
        ),
        network: NetworkBacking::MotlieVnet(
            motlie_vmm::backend::motlie::vnet::MotlieVnetBacking::new(),
        ),
        control_plane: ControlPlaneBacking::MotlieSshProxy(
            motlie_vmm::backend::motlie::ssh_proxy::MotlieSshProxyBacking::new(
                Arc::clone(&ca),
                Arc::clone(&guest_registry),
            ),
        ),
    });

    let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig {
        socket_dir: instance.socket_root.clone(),
        ..GuestNetAllocatorConfig::default()
    });
    let prepared = prepare(
        PrepareRequest {
            guest,
            namespace: instance.namespace.clone(),
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
            base_dir: base_dir.clone(),
            ssh_ca_pubkey: Some(ca.public_key_openssh()?),
        },
        &mut allocator,
    )?;
    let handle = boot(
        prepared,
        LifecycleServices {
            runtime,
        },
    )
    .await?;
    handle.ready(&ReadinessPolicy::default()).await?;
    match scenario.as_str() {
        "smoke" => run_smoke(&handle).await?,
        "pty" => pty::run_pty_smoke(&handle).await?,
        other => return Err(format!("unsupported scenario: {other}").into()),
    }

    let report = handle.shutdown().await?;
    println!(
        "v1.4 harness {} passed: guest={} pid={:?} forced={:?} proxy=127.0.0.1:{}",
        scenario,
        handle.guest_id,
        report.pid,
        report.forced,
        proxy_config.listen.port()
    );
    Ok(())
}

async fn run_smoke(handle: &motlie_vmm::orchestrator::VmHandle) -> Result<(), DynError> {
    let hello = handle.exec("/bin/echo hello", Duration::from_secs(10)).await?;
    ensure_success(&hello.stdout, "hello")?;

    let vfs = handle
        .exec(
            "/bin/sh -lc 'pwd && test -d /home/alice && test -d /workspace && test -d /agent-state && grep -q \"Alice workspace mounted from the host.\" /workspace/README.md && echo VFS_OK'",
            Duration::from_secs(10),
        )
        .await?;
    ensure_success(&vfs.stdout, "VFS_OK")?;

    let route = exec_until_success(
        handle,
        r#"/bin/sh -lc 'ip route | grep -q "^default via 10.0.2.2 " && echo ROUTE_OK'"#,
        "ROUTE_OK",
        Duration::from_secs(10),
    )
    .await?;
    ensure_success(&route.stdout, "ROUTE_OK")?;

    let outbound = exec_until_success(
        handle,
        r#"/bin/sh -lc 'code=$(curl -s -o /dev/null -w "%{http_code}" https://example.com); test "$code" = 200 && echo HTTPS_OK'"#,
        "HTTPS_OK",
        Duration::from_secs(20),
    )
    .await?;
    ensure_success(&outbound.stdout, "HTTPS_OK")?;
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

fn new_harness_instance() -> Result<HarnessInstance, DynError> {
    let pid = std::process::id();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        % 10_000;
    let prefix = format!("motlie-vmm-v14-h{pid}-{nonce}");
    let namespace = RuntimeNamespace::new(prefix, "/tmp")?;
    let demo_root = namespace.temp_root.join(format!("{}-demo", namespace.prefix));
    let socket_root = namespace
        .temp_root
        .join(format!("{}-sockets", namespace.prefix));
    let proxy_port = 32000 + u16::try_from(((u64::from(pid)) + (nonce as u64)) % 10_000)?;
    Ok(HarnessInstance {
        namespace,
        demo_root,
        socket_root,
        proxy_port,
    })
}

fn print_instance_details(instance: &HarnessInstance, proxy_config: &SshProxyConfig) {
    println!("v1.4 harness instance: {}", instance.namespace.prefix);
    println!("  demo_root={}", instance.demo_root.display());
    println!("  socket_root={}", instance.socket_root.display());
    println!("  proxy=ssh://localhost:{}", proxy_config.listen.port());
}

fn demo_guest(
    guest_id: &str,
    artifacts_dir: &Path,
    demo_root: &Path,
    namespace: &RuntimeNamespace,
) -> GuestSpec {
    let (uid, gid) = demo_guest_ids(guest_id);
    GuestSpec {
        guest_id: guest_id.to_string(),
        hostname: format!("motlie-{guest_id}"),
        socket_path: format!("/tmp/{}-{guest_id}.vsock_5000", namespace.prefix),
        user: GuestUser {
            name: guest_id.to_string(),
            uid,
            gid,
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
    let home = &guest.mounts[0].host_path;
    let ssh_dir = home.join(".ssh");
    std::fs::create_dir_all(home.join(".config"))?;
    std::fs::create_dir_all(&ssh_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&ssh_dir, std::fs::Permissions::from_mode(0o700))?;
    }
    write_host_file_if_missing(
        &ssh_dir.join("authorized_keys"),
        &format!(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample {}@dev\n",
            guest.guest_id
        ),
        0o600,
    )?;
    write_host_file_if_missing(
        &ssh_dir.join("config"),
        "Host github.com\n  User git\n",
        0o644,
    )?;
    write_host_file_if_missing(
        &home.join(".env"),
        &format!("{}_API_KEY=demo-{}\n", guest.guest_id.to_uppercase(), guest.guest_id),
        0o644,
    )?;
    write_host_file_if_missing(&home.join(".bashrc"), "# motlie v1.4 demo bashrc\n", 0o644)?;
    write_host_file_if_missing(
        &home.join(".profile"),
        "if [ -f \"$HOME/.bashrc\" ]; then\n  . \"$HOME/.bashrc\"\nfi\n",
        0o644,
    )?;
    write_host_file_if_missing(
        &guest.mounts[2].host_path.join("README.md"),
        "Dedicated read-write agent-state layer for Codex and Claude lives here.\n",
        0o644,
    )?;
    std::fs::write(
        guest.mounts[1].host_path.join("README.md"),
        format!(
            "{} workspace mounted from the host.\n",
            guest_display_name(&guest.guest_id)
        ),
    )?;
    Ok(())
}

fn write_host_file_if_missing(path: &Path, content: &str, mode: u32) -> Result<(), DynError> {
    if !path.exists() {
        std::fs::write(path, content)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
        }
    }
    Ok(())
}

fn demo_guest_ids(guest_id: &str) -> (u32, u32) {
    match guest_id {
        "alice" => (1000, 1000),
        "bob" => (1001, 1001),
        _ => (1000, 1000),
    }
}

fn guest_display_name(guest_id: &str) -> String {
    let mut chars = guest_id.chars();
    match chars.next() {
        Some(first) => {
            let mut out = first.to_uppercase().collect::<String>();
            out.push_str(chars.as_str());
            out
        }
        None => String::new(),
    }
}

async fn exec_until_success(
    handle: &motlie_vmm::orchestrator::VmHandle,
    command: &str,
    needle: &str,
    timeout: Duration,
) -> Result<ExecOutput, DynError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_output: Option<ExecOutput> = None;
    loop {
        match handle.exec(command, Duration::from_secs(10)).await {
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
