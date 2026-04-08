use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use motlie_vmm::ca::SshCa;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::orchestrator::{
    LifecycleServices, PrepareRequest, ReadinessPolicy, VmHandle, boot, prepare,
};
use motlie_vmm::runtime::{
    ControlPlaneBacking, FilesystemBacking, HypervisorBacking, NetworkBacking, Runtime,
};
use motlie_vmm::spec::{
    BootArtifacts, GuestMountSpec, GuestResources, GuestSpec, GuestSshAccess, GuestStorage,
    GuestUser, RuntimeNamespace, SoftwareProfile,
};
use motlie_vmm::ssh::{self, ExecOutput, SshProxyConfig, new_guest_registry};
use tokio::sync::mpsc;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), DynError> {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.4");
    let artifacts_dir = base_dir.join("artifacts/base");
    ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))?;
    ensure_file_exists(&artifacts_dir.join("Image"))?;

    let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp")?;
    let demo_root = PathBuf::from("/tmp/motlie-vmm-v14-demo");
    let socket_root = namespace.temp_root.join(format!("{}-sockets", namespace.prefix));
    std::fs::create_dir_all(&socket_root)?;

    let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig {
        socket_dir: socket_root.clone(),
        ..GuestNetAllocatorConfig::default()
    });
    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();
    let proxy_config = SshProxyConfig {
        listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 2224),
    };
    tokio::spawn(ssh::run_proxy(proxy_config.clone(), Arc::clone(&guest_registry)));

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

    let mut handles: HashMap<String, VmHandle> = HashMap::new();
    let mut stdout = io::stdout();

    println!("=== motlie-vmm repl_host_v1_4 ===");
    println!("Thin admin REPL over the extracted vmm lifecycle API");
    println!("SSH proxy: listening on 127.0.0.1:{}", proxy_config.listen.port());
    println!("Commands: help | boot <guest> | ready <guest> | exec <guest> <cmd> | validate <guest> | shutdown <guest> | status | guests | where [guest] | quit");

    let mut lines = spawn_stdin_reader();

    print_prompt(&mut stdout)?;
    while let Some(line_result) = lines.recv().await {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            print_prompt(&mut stdout)?;
            continue;
        }

        let result = if trimmed == "help" {
            print_help();
            Ok(())
        } else if trimmed == "status" || trimmed == "guests" {
            print_status(&handles);
            Ok(())
        } else if trimmed == "where" {
            print_where(&namespace, &demo_root, &socket_root, &proxy_config, None, &handles);
            Ok(())
        } else if let Some(rest) = trimmed.strip_prefix("where ") {
            let guest_id = rest.trim();
            let guest = (!guest_id.is_empty()).then_some(guest_id);
            print_where(
                &namespace,
                &demo_root,
                &socket_root,
                &proxy_config,
                guest,
                &handles,
            );
            Ok(())
        } else if trimmed == "quit" || trimmed == "exit" {
            break;
        } else if let Some(rest) = trimmed.strip_prefix("boot ") {
            let guest_id = rest.trim();
            if guest_id.is_empty() {
                Err::<(), DynError>("boot <guest>".into())
            } else {
                boot_guest(
                    guest_id,
                    &artifacts_dir,
                    &demo_root,
                    &base_dir,
                    &namespace,
                    &ca,
                    &runtime,
                    &mut allocator,
                    &mut handles,
                )
                .await
            }
        } else if let Some(rest) = trimmed.strip_prefix("ready ") {
            ready_guest(rest.trim(), &handles).await
        } else if let Some(rest) = trimmed.strip_prefix("shutdown ") {
            shutdown_guest(rest.trim(), &mut handles).await
        } else if let Some(rest) = trimmed.strip_prefix("validate ") {
            validate_guest(rest.trim(), &handles).await
        } else if let Some(rest) = trimmed.strip_prefix("exec ") {
            exec_guest(rest, &handles).await
        } else if let Some(rest) = trimmed.strip_prefix("launch ") {
            Err::<(), DynError>(format!("use 'boot {}' instead", rest.trim()).into())
        } else {
            Err::<(), DynError>(format!("unknown command: {trimmed}").into())
        };

        if let Err(err) = result {
            println!("error: {err}");
        }

        print_prompt(&mut stdout)?;
    }

    for (_guest_id, handle) in handles.drain() {
        let _ = handle.shutdown().await;
    }

    Ok(())
}

fn spawn_stdin_reader() -> mpsc::UnboundedReceiver<Result<String, io::Error>> {
    let (tx, rx) = mpsc::unbounded_channel();
    thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

fn print_help() {
    println!("boot <guest>             boot a Motlie-backed guest and wait until ready");
    println!("ready <guest>            wait until an already-booted guest is ready");
    println!("exec <guest> <command>   run a command inside the guest over the SSH control plane");
    println!("validate <guest>         run a smoke validation inside the guest");
    println!("shutdown <guest>         stop a guest");
    println!("status | guests          show active guests");
    println!("where [guest]            show current runtime roots and guest artifact paths");
    println!("quit                     exit the REPL");
}

fn print_status(handles: &HashMap<String, VmHandle>) {
    if handles.is_empty() {
        println!("(no guests)");
        return;
    }

    for (guest_id, handle) in handles {
        println!(
            "{} pid={:?} api={} vnet={} vsock={}",
            guest_id,
            handle.pid,
            handle.runtime_paths.api_socket.display(),
            handle.runtime_paths.vnet_socket.display(),
            handle.runtime_paths.vsock_socket.display(),
        );
    }
}

fn print_where(
    namespace: &RuntimeNamespace,
    demo_root: &Path,
    socket_root: &Path,
    proxy_config: &SshProxyConfig,
    guest_id: Option<&str>,
    handles: &HashMap<String, VmHandle>,
) {
    println!("namespace={}", namespace.prefix);
    println!("temp_root={}", namespace.temp_root.display());
    println!("demo_root={}", demo_root.display());
    println!("socket_root={}", socket_root.display());
    println!("proxy=ssh://localhost:{}", proxy_config.listen.port());

    match guest_id {
        Some(guest_id) => {
            let Some(handle) = handles.get(guest_id) else {
                println!("guest '{guest_id}' is not currently booted");
                return;
            };
            print_guest_where(guest_id, demo_root, handle);
        }
        None => {
            for (guest_id, handle) in handles {
                print_guest_where(guest_id, demo_root, handle);
            }
        }
    }
}

fn print_guest_where(guest_id: &str, demo_root: &Path, handle: &VmHandle) {
    println!("[{guest_id}]");
    println!("  home_host={}", demo_root.join(format!("{guest_id}-home")).display());
    println!(
        "  workspace_host={}",
        demo_root.join(format!("{guest_id}-workspace")).display()
    );
    println!(
        "  agent_state_host={}",
        demo_root.join(format!("{guest_id}-agent-state")).display()
    );
    println!("  runtime_dir={}", handle.runtime_paths.runtime_dir.display());
    println!("  launch_dir={}", handle.runtime_paths.launch_dir.display());
    println!(
        "  cloud_init_dir={}",
        handle.runtime_paths.cloud_init_dir.display()
    );
    println!("  api_socket={}", handle.runtime_paths.api_socket.display());
    println!("  vnet_socket={}", handle.runtime_paths.vnet_socket.display());
    println!("  vsock_socket={}", handle.runtime_paths.vsock_socket.display());
    println!("  launch_log={}", handle.runtime_paths.launch_log.display());
    println!("  serial_log={}", handle.runtime_paths.serial_log.display());
}

fn print_prompt(stdout: &mut io::Stdout) -> Result<(), DynError> {
    print!("v14> ");
    stdout.flush()?;
    Ok(())
}

async fn boot_guest(
    guest_id: &str,
    artifacts_dir: &Path,
    demo_root: &Path,
    base_dir: &Path,
    namespace: &RuntimeNamespace,
    ca: &Arc<SshCa>,
    runtime: &Arc<Runtime>,
    allocator: &mut GuestNetAllocator,
    handles: &mut HashMap<String, VmHandle>,
) -> Result<(), DynError> {
    if handles.contains_key(guest_id) {
        return Err(format!("guest '{guest_id}' already booted").into());
    }

    let guest = demo_guest(guest_id, artifacts_dir, demo_root, namespace);
    seed_host_mounts(&guest)?;

    let prepared = prepare(
        PrepareRequest {
            guest,
            namespace: namespace.clone(),
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
            base_dir: base_dir.to_path_buf(),
            ssh_ca_pubkey: Some(ca.public_key_openssh()?),
        },
        allocator,
    )?;

    let handle = boot(
        prepared,
        LifecycleServices {
            runtime: Arc::clone(runtime),
        },
    )
    .await?;
    println!("waiting for {}: api socket + guestfs + ssh bridge + exec-ready", guest_id);
    handle.ready(&ReadinessPolicy::default()).await?;
    println!(
        "ok: booted {} pid={:?} api={} proxy=127.0.0.1:2224",
        guest_id,
        handle.pid,
        handle.runtime_paths.api_socket.display()
    );
    handles.insert(guest_id.to_string(), handle);
    Ok(())
}

async fn ready_guest(guest_id: &str, handles: &HashMap<String, VmHandle>) -> Result<(), DynError> {
    let handle = handles
        .get(guest_id)
        .ok_or_else(|| format!("unknown guest '{guest_id}'"))?;
    handle.ready(&ReadinessPolicy::default()).await?;
    println!("ok: {guest_id} ready");
    Ok(())
}

async fn shutdown_guest(
    guest_id: &str,
    handles: &mut HashMap<String, VmHandle>,
) -> Result<(), DynError> {
    let handle = handles
        .remove(guest_id)
        .ok_or_else(|| format!("unknown guest '{guest_id}'"))?;
    let report = handle.shutdown().await?;
    println!(
        "ok: shutdown {} pid={:?} forced={:?}",
        guest_id, report.pid, report.forced
    );
    Ok(())
}

async fn exec_guest(rest: &str, handles: &HashMap<String, VmHandle>) -> Result<(), DynError> {
    let mut parts = rest.trim().splitn(2, char::is_whitespace);
    let guest_id = parts.next().unwrap_or("").trim();
    let command = parts.next().unwrap_or("").trim();
    if guest_id.is_empty() || command.is_empty() {
        return Err("exec <guest> <command>".into());
    }

    let handle = handles
        .get(guest_id)
        .ok_or_else(|| format!("unknown guest '{guest_id}'"))?;
    let output = handle.exec(command, Duration::from_secs(20)).await?;
    print_exec_output(&output);
    Ok(())
}

async fn validate_guest(
    guest_id: &str,
    handles: &HashMap<String, VmHandle>,
) -> Result<(), DynError> {
    let handle = handles
        .get(guest_id)
        .ok_or_else(|| format!("unknown guest '{guest_id}'"))?;
    let expected_gateway = handle.net_assignment.egress_ipv4.host;

    let checks = vec![
        (
            "vsock-ssh: uname",
            "/bin/uname -s".to_string(),
            "Linux".to_string(),
            Duration::from_secs(10),
        ),
        (
            "workspace: README visible through VFS",
            "/bin/sh -lc 'test -f /workspace/README.md && echo WORKSPACE_OK'".to_string(),
            "WORKSPACE_OK".to_string(),
            Duration::from_secs(10),
        ),
        (
            "agent-state: ~/.codex writable directory",
            "/bin/sh -lc 'test -d ~/.codex && test -w ~/.codex && echo CODEX_OK'".to_string(),
            "CODEX_OK".to_string(),
            Duration::from_secs(10),
        ),
        (
            "ip route: default via motlie-vmm",
            format!(
                "/bin/sh -lc 'ip route | grep -q \"^default via {} \" && echo ROUTE_OK'",
                expected_gateway
            ),
            "ROUTE_OK".to_string(),
            Duration::from_secs(10),
        ),
        (
            "egress: curl https://example.com",
            "/bin/sh -lc 'code=$(curl -s -o /dev/null -w \"%{http_code}\" https://example.com); test \"$code\" = 200 && echo HTTPS_OK'".to_string(),
            "HTTPS_OK".to_string(),
            Duration::from_secs(20),
        ),
    ];

    println!("=== validate {guest_id} ===");
    let mut passed = 0usize;
    let mut failed = 0usize;

    for (label, command, needle, timeout) in checks {
        match handle.exec(&command, timeout).await {
            Ok(output) if output.exit_code == 0 && output.stdout.contains(&needle) => {
                println!("  PASS: {label}");
                passed += 1;
            }
            Ok(output) => {
                println!(
                    "  FAIL: {} (exit={} stdout={} stderr={})",
                    label,
                    output.exit_code,
                    output.stdout.trim(),
                    output.stderr.trim()
                );
                failed += 1;
            }
            Err(err) => {
                println!("  FAIL: {} ({})", label, err);
                failed += 1;
            }
        }
    }

    println!("=== {} passed, {} failed ===", passed, failed);
    if failed == 0 {
        Ok(())
    } else {
        Err(format!("validation failed for guest '{guest_id}'").into())
    }
}

fn print_exec_output(output: &ExecOutput) {
    if !output.stdout.is_empty() {
        print!("{}", output.stdout);
        if !output.stdout.ends_with('\n') {
            println!();
        }
    }
    if !output.stderr.is_empty() {
        eprint!("{}", output.stderr);
        if !output.stderr.ends_with('\n') {
            eprintln!();
        }
    }
    println!("exit={}", output.exit_code);
}

fn ensure_file_exists(path: &Path) -> Result<(), DynError> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("required artifact missing: {}", path.display()).into())
    }
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
