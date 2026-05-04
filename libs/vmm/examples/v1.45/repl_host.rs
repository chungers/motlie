mod demo_support;

use std::io::{self};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use motlie_vmm::ca::SshCa;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::orchestrator::{LifecycleServices, ReadinessPolicy};
use motlie_vmm::provisioning::{
    GuestProvisioner, GuestProvisionerConfig, ProvisionedGuestSnapshot, ProvisioningGuestRequest,
};
use motlie_vmm::runtime::{
    ControlPlaneBacking, FilesystemBacking, HypervisorBacking, NetworkBacking, Runtime,
};
use motlie_vmm::spec::{
    BootArtifacts, GuestMountSpec, GuestResources, GuestSpec, GuestSshAccess, GuestStorage,
    GuestUser, RuntimeNamespace, SoftwareProfile,
};
use motlie_vmm::ssh::{
    new_guest_registry, ExecOutput, PrincipalResolver, SshProxyConfig, SshProxyError,
};

use demo_support::{
    cleanup_development_guest_disks, demo_guest_ids, demo_guest_socket_path, guest_runtime_paths,
    install_signal_watchers, prompt, shutdown_active_guests, spawn_host_events, spawn_proxy_task,
    stdin_line_or_detach, HostEvent, ProxyRestartState,
};

type DynError = Box<dyn std::error::Error + Send + Sync>;

const REPL_PROXY_BASE_PORT: u16 = 42_000;

fn resolved_native_source_dir(base_dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("MOTLIE_VZ_BASE_VM_DIR").map(PathBuf::from) {
        if path.join("disk.img").exists() && path.join("nvram.bin").exists() {
            return path;
        }
    }
    let local = base_dir.join("artifacts/source-base.vm");
    if local.join("disk.img").exists() && local.join("nvram.bin").exists() {
        local
    } else {
        base_dir.join("../v1.35/artifacts/source-base.vm")
    }
}

struct ReplInstance {
    namespace: RuntimeNamespace,
    demo_root: PathBuf,
    socket_root: PathBuf,
    proxy_port: u16,
}

fn build_guest_provisioner(
    base_dir: &Path,
    artifacts_dir: &Path,
    instance: &ReplInstance,
    allocator_config: GuestNetAllocatorConfig,
    ca: &Arc<SshCa>,
    runtime: &Arc<Runtime>,
) -> Result<GuestProvisioner, DynError> {
    Ok(GuestProvisioner::new(GuestProvisionerConfig {
        namespace: instance.namespace.clone(),
        base_dir: base_dir.to_path_buf(),
        network_modes: NetworkModes {
            admin: AdminNetMode::None,
            egress: EgressNetMode::VzUserspace,
        },
        readiness_policy: ReadinessPolicy {
            api_socket_timeout: Duration::from_secs(10),
            guestfs_timeout: Duration::from_secs(90),
            ssh_bridge_timeout: Duration::from_secs(90),
            exec_ready_timeout: Duration::from_secs(90),
        },
        services: LifecycleServices {
            runtime: Arc::clone(runtime),
        },
        allocator: GuestNetAllocator::new(allocator_config)?,
        ssh_ca_pubkey: ca.public_key_openssh()?,
        guest_spec_factory: Arc::new({
            let artifacts_dir = artifacts_dir.to_path_buf();
            let demo_root = instance.demo_root.clone();
            move |request: &ProvisioningGuestRequest| {
                demo_guest(
                    &request.principal,
                    request.net_assignment.slot,
                    &artifacts_dir,
                    &demo_root,
                    &request.namespace,
                )
                .map_err(|err| err.to_string())
            }
        }),
        host_seed_hook: Some(Arc::new(|guest| {
            seed_host_mounts(guest).map_err(|err| err.to_string())
        })),
    }))
}

fn build_principal_resolver(
    provisioner: &GuestProvisioner,
    auto_provision_enabled: &Arc<AtomicBool>,
) -> PrincipalResolver {
    let provisioner = provisioner.clone();
    let auto_provision_enabled = Arc::clone(auto_provision_enabled);
    Arc::new(move |principal: String| {
        let provisioner = provisioner.clone();
        let auto_provision_enabled = Arc::clone(&auto_provision_enabled);
        Box::pin(async move {
            if !auto_provision_enabled.load(Ordering::SeqCst) {
                return Ok(principal);
            }

            provisioner
                .ensure_guest_for_principal(&principal)
                .await
                .map(|guest| guest.spec.guest_id.clone())
                .map_err(|err| SshProxyError::ResolveGuest {
                    principal,
                    reason: err.to_string(),
                })
        })
    })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), DynError> {
    let mut args = std::env::args().skip(1);
    let mut root_override: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--root requires a path".to_string())?;
                root_override = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other if other.starts_with("--root=") => {
                root_override = Some(PathBuf::from(other.trim_start_matches("--root=")));
            }
            other => return Err(format!("unknown option: {other}").into()),
        }
    }

    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.45");
    let artifacts_dir = resolved_native_source_dir(&base_dir);
    ensure_file_exists(&artifacts_dir.join("disk.img"))?;
    ensure_file_exists(&artifacts_dir.join("nvram.bin"))?;

    let root_dir = root_override.unwrap_or_else(RuntimeNamespace::root_from_env_or_temp);
    let instance = new_repl_instance(&root_dir)?;
    std::fs::create_dir_all(&instance.socket_root)?;

    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();

    let runtime = Arc::new(Runtime {
        hypervisor: HypervisorBacking::AppleVirtualizationShell(
            motlie_vmm::backend::vz::shell::VzShellBackend::new(),
        ),
        filesystem: FilesystemBacking::MotlieVfs(
            motlie_vmm::backend::motlie::vfs::MotlieVfsBacking::new(),
        ),
        network: NetworkBacking::HypervisorManaged,
        control_plane: ControlPlaneBacking::MotlieSshProxy(
            motlie_vmm::backend::motlie::ssh_proxy::MotlieSshProxyBacking::new(
                Arc::clone(&ca),
                Arc::clone(&guest_registry),
            ),
        ),
    });
    let provisioner = build_guest_provisioner(
        &base_dir,
        &artifacts_dir,
        &instance,
        GuestNetAllocatorConfig {
            socket_dir: instance.socket_root.clone(),
            socket_name_prefix: instance.namespace.prefix.clone(),
            ..GuestNetAllocatorConfig::default()
        },
        &ca,
        &runtime,
    )?;
    let auto_provision_enabled = Arc::new(AtomicBool::new(false));
    let proxy_config = SshProxyConfig {
        listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), instance.proxy_port),
        principal_resolver: Some(build_principal_resolver(
            &provisioner,
            &auto_provision_enabled,
        )),
    };
    let mut proxy_task = spawn_proxy_task(proxy_config.clone(), Arc::clone(&guest_registry));
    let mut proxy_restart_state = ProxyRestartState::new();

    let mut stdout = io::stdout();

    println!("=== motlie-vmm repl_host_v1_45 ===");
    println!("Thin admin REPL over the extracted vmm lifecycle API on Apple Vz");
    println!("Instance: {}", instance.namespace.prefix);
    println!(
        "SSH proxy: listening on 127.0.0.1:{}",
        proxy_config.listen.port()
    );
    println!("Auto-provision: disabled");
    println!(
        "Commands: help | auto-provision <on|off|status> | boot <guest> | ready <guest> | exec <guest> <cmd> | validate <guest> | shutdown <guest> | status | guests | where [guest] | quit"
    );

    let (mut events, event_tx) = spawn_host_events();
    install_signal_watchers(event_tx)?;
    let mut events_open = true;
    let mut headless = false;

    prompt(&mut stdout, "v145> ", &mut headless);
    loop {
        tokio::select! {
            event = events.recv(), if events_open => {
                let Some(event) = event else {
                    events_open = false;
                    if !headless {
                        headless = true;
                        eprintln!("notice: operator event stream closed; continuing headless");
                    }
                    continue;
                };

                match event {
                    HostEvent::StdinLine(line_result) => {
                        let Some(line) = stdin_line_or_detach(line_result, &mut headless) else {
                            continue;
                        };
                        let trimmed = line.trim();

                        if trimmed.is_empty() || trimmed.starts_with('#') {
                            prompt(&mut stdout, "v145> ", &mut headless);
                            continue;
                        }

                        let mut should_exit = false;
                        let result = if trimmed == "help" {
                            print_help();
                            Ok(())
                        } else if trimmed == "auto-provision" || trimmed == "auto-provision status" {
                            println!(
                                "auto-provision={}",
                                auto_provision_status(auto_provision_enabled.load(Ordering::SeqCst))
                            );
                            Ok(())
                        } else if trimmed == "auto-provision on" {
                            auto_provision_enabled.store(true, Ordering::SeqCst);
                            println!("ok: auto-provision enabled");
                            Ok(())
                        } else if trimmed == "auto-provision off" {
                            auto_provision_enabled.store(false, Ordering::SeqCst);
                            println!("ok: auto-provision disabled");
                            Ok(())
                        } else if trimmed == "status" || trimmed == "guests" {
                            print_status(&provisioner, auto_provision_enabled.load(Ordering::SeqCst));
                            Ok(())
                        } else if trimmed == "where" {
                            print_where(
                                &instance.namespace,
                                &instance.demo_root,
                                &instance.socket_root,
                                &proxy_config,
                                None,
                                &provisioner,
                            );
                            Ok(())
                        } else if let Some(rest) = trimmed.strip_prefix("where ") {
                            let guest_id = rest.trim();
                            let guest = (!guest_id.is_empty()).then_some(guest_id);
                            print_where(
                                &instance.namespace,
                                &instance.demo_root,
                                &instance.socket_root,
                                &proxy_config,
                                guest,
                                &provisioner,
                            );
                            Ok(())
                        } else if trimmed == "quit" || trimmed == "exit" {
                            should_exit = true;
                            Ok(())
                        } else if let Some(rest) = trimmed.strip_prefix("boot ") {
                            let guest_id = rest.trim();
                            if guest_id.is_empty() {
                                Err::<(), DynError>("boot <guest>".into())
                            } else {
                                boot_guest(guest_id, &proxy_config, &provisioner).await
                            }
                        } else if let Some(rest) = trimmed.strip_prefix("ready ") {
                            ready_guest(rest.trim(), &provisioner).await
                        } else if let Some(rest) = trimmed.strip_prefix("shutdown ") {
                            shutdown_guest(rest.trim(), &provisioner).await
                        } else if let Some(rest) = trimmed.strip_prefix("validate ") {
                            validate_guest(rest.trim(), &provisioner).await
                        } else if let Some(rest) = trimmed.strip_prefix("exec ") {
                            exec_guest(rest, &provisioner).await
                        } else if let Some(rest) = trimmed.strip_prefix("launch ") {
                            Err::<(), DynError>(format!("use 'boot {}' instead", rest.trim()).into())
                        } else {
                            Err::<(), DynError>(format!("unknown command: {trimmed}").into())
                        };

                        if let Err(err) = result {
                            println!("error: {err}");
                        }
                        if should_exit {
                            break;
                        }
                        prompt(&mut stdout, "v145> ", &mut headless);
                    }
                    HostEvent::StdinClosed => {
                        if !headless {
                            headless = true;
                            eprintln!("notice: stdin closed; continuing headless. Use SIGINT or SIGTERM to stop the host.");
                        }
                    }
                    HostEvent::Terminate(signal_name) => {
                        eprintln!("notice: received {signal_name}; shutting down");
                        break;
                    }
                    HostEvent::Hangup => {
                        if !headless {
                            headless = true;
                        }
                        eprintln!("notice: received SIGHUP; keeping proxy and guests alive");
                    }
                }
            }
            proxy_result = &mut proxy_task => {
                match proxy_result {
                    Ok(Ok(())) => eprintln!("warning: SSH proxy exited unexpectedly"),
                    Ok(Err(err)) => eprintln!("warning: SSH proxy failed: {err}"),
                    Err(err) => eprintln!("warning: SSH proxy task aborted: {err}"),
                }
                let delay = proxy_restart_state
                    .next_delay()
                    .map_err(|err| -> DynError { err.into() })?;
                eprintln!("warning: restarting SSH proxy in {}s", delay.as_secs());
                tokio::time::sleep(delay).await;
                proxy_task = spawn_proxy_task(proxy_config.clone(), Arc::clone(&guest_registry));
                proxy_restart_state.mark_started();
                prompt(&mut stdout, "v145> ", &mut headless);
            }
        }
    }

    proxy_task.abort();
    shutdown_active_guests(&provisioner, "repl host").await;
    cleanup_development_guest_disks(&instance.namespace, "repl host");

    Ok(())
}

fn print_help() {
    println!("auto-provision <mode>    toggle SSH principal auto-provisioning on, off, or status");
    println!("boot <guest>             boot a Motlie-backed guest and wait until ready");
    println!("ready <guest>            wait until an already-booted guest is ready");
    println!("exec <guest> <command>   run a command inside the guest over the SSH control plane");
    println!("validate <guest>         run a smoke validation inside the guest");
    println!("shutdown <guest>         stop a guest");
    println!("status | guests          show active guests");
    println!("where [guest]            show current runtime roots and guest artifact paths");
    println!("quit                     exit the REPL");
}

fn auto_provision_status(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

fn print_status(provisioner: &GuestProvisioner, auto_provision_enabled: bool) {
    println!(
        "auto-provision={}",
        auto_provision_status(auto_provision_enabled)
    );
    let active: Vec<_> = provisioner
        .guests()
        .unwrap_or_default()
        .into_iter()
        .filter(|guest| guest.active)
        .collect();
    if active.is_empty() {
        println!("(no guests)");
        return;
    }

    for guest in active {
        let Some(runtime_paths) = guest_runtime_paths(&guest) else {
            println!(
                "{} pid={:?} runtime_paths=(missing)",
                guest.principal, guest.pid
            );
            continue;
        };
        println!(
            "{} pid={:?} api={} vnet={} vsock={}",
            guest.principal,
            guest.pid,
            runtime_paths.api_socket.display(),
            runtime_paths.vnet_socket.display(),
            runtime_paths.vsock_socket.display(),
        );
    }
}

fn print_where(
    namespace: &RuntimeNamespace,
    demo_root: &Path,
    socket_root: &Path,
    proxy_config: &SshProxyConfig,
    guest_id: Option<&str>,
    provisioner: &GuestProvisioner,
) {
    println!("namespace={}", namespace.prefix);
    println!("temp_root={}", namespace.temp_root.display());
    println!("demo_root={}", demo_root.display());
    println!("socket_root={}", socket_root.display());
    println!("proxy=ssh://localhost:{}", proxy_config.listen.port());

    match guest_id {
        Some(guest_id) => {
            let Some(guest) = provisioner
                .snapshot(guest_id)
                .unwrap_or(None)
                .filter(|guest| guest.active)
            else {
                println!("guest '{guest_id}' is not currently booted");
                return;
            };
            print_guest_where(demo_root, &guest);
        }
        None => {
            for guest in provisioner.guests().unwrap_or_default() {
                if guest.active {
                    print_guest_where(demo_root, &guest);
                }
            }
        }
    }
}

fn print_guest_where(demo_root: &Path, guest: &ProvisionedGuestSnapshot) {
    println!("[{}]", guest.principal);
    let Some(runtime_paths) = guest_runtime_paths(guest) else {
        println!("  runtime_paths=(missing)");
        return;
    };
    println!(
        "  home_host={}",
        demo_root
            .join(format!("{}-home", guest.principal))
            .display()
    );
    println!(
        "  workspace_host={}",
        demo_root
            .join(format!("{}-workspace", guest.principal))
            .display()
    );
    println!(
        "  agent_state_host={}",
        demo_root
            .join(format!("{}-agent-state", guest.principal))
            .display()
    );
    println!("  runtime_dir={}", runtime_paths.runtime_dir.display());
    println!("  launch_dir={}", runtime_paths.launch_dir.display());
    println!(
        "  cloud_init_dir={}",
        runtime_paths.cloud_init_dir.display()
    );
    println!("  api_socket={}", runtime_paths.api_socket.display());
    println!("  vnet_socket={}", runtime_paths.vnet_socket.display());
    println!("  vsock_socket={}", runtime_paths.vsock_socket.display());
    println!("  launch_log={}", runtime_paths.launch_log.display());
    println!("  serial_log={}", runtime_paths.serial_log.display());
}

async fn boot_guest(
    guest_id: &str,
    proxy_config: &SshProxyConfig,
    provisioner: &GuestProvisioner,
) -> Result<(), DynError> {
    if provisioner
        .snapshot(guest_id)?
        .is_some_and(|guest| guest.active)
    {
        return Err(format!("guest '{guest_id}' already booted").into());
    }

    let handle = provisioner.ensure_guest_for_principal(guest_id).await?;
    println!(
        "ok: booted {} pid={:?} api={} proxy=127.0.0.1:{}",
        guest_id,
        handle.handle.pid,
        handle.handle.runtime_paths.api_socket.display(),
        proxy_config.listen.port(),
    );
    Ok(())
}

async fn ready_guest(guest_id: &str, provisioner: &GuestProvisioner) -> Result<(), DynError> {
    provisioner.ready(guest_id).await?;
    println!("ok: {guest_id} ready");
    Ok(())
}

async fn shutdown_guest(guest_id: &str, provisioner: &GuestProvisioner) -> Result<(), DynError> {
    let report = provisioner
        .shutdown_guest(guest_id)
        .await?
        .ok_or_else(|| format!("unknown guest '{guest_id}'"))?;
    println!(
        "ok: shutdown {} pid={:?} forced={:?}",
        guest_id, report.pid, report.forced
    );
    Ok(())
}

async fn exec_guest(rest: &str, provisioner: &GuestProvisioner) -> Result<(), DynError> {
    let mut parts = rest.trim().splitn(2, char::is_whitespace);
    let guest_id = parts.next().unwrap_or("").trim();
    let command = parts.next().unwrap_or("").trim();
    if guest_id.is_empty() || command.is_empty() {
        return Err("exec <guest> <command>".into());
    }

    let output = provisioner
        .exec(guest_id, command, Duration::from_secs(20))
        .await?;
    print_exec_output(&output);
    Ok(())
}

async fn validate_guest(guest_id: &str, provisioner: &GuestProvisioner) -> Result<(), DynError> {
    let handle = provisioner.active_vm_handle(guest_id)?;
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

fn print_usage() {
    println!("usage: repl_host_v1_45 [--root <dir>]");
}

fn new_repl_instance(root_dir: &Path) -> Result<ReplInstance, DynError> {
    // Auto-provision can defer the first guest boot until after the operator
    // has detached. In that headless path we still allocate guestfs/vsock/vnet
    // AF_UNIX sockets under the explicit --root, so the per-instance prefix and
    // socket directory must stay compact enough to fit sun_path.
    let namespace = RuntimeNamespace::for_process("v145", "r", root_dir)?;
    let demo_root = namespace
        .temp_root
        .join(format!("{}-demo", namespace.prefix));
    let socket_root = namespace.temp_root.join("s");
    let proxy_port = REPL_PROXY_BASE_PORT + port_offset(&namespace.prefix);
    Ok(ReplInstance {
        namespace,
        demo_root,
        socket_root,
        proxy_port,
    })
}

fn demo_guest(
    guest_id: &str,
    slot: u32,
    artifacts_dir: &Path,
    demo_root: &Path,
    namespace: &RuntimeNamespace,
) -> Result<GuestSpec, DynError> {
    let (uid, gid) = demo_guest_ids(guest_id, slot)?;
    Ok(GuestSpec {
        guest_id: guest_id.to_string(),
        hostname: format!("motlie-{guest_id}"),
        socket_path: demo_guest_socket_path(namespace, guest_id)?,
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
        software: SoftwareProfile { packages: vec![] },
        resources: GuestResources::default(),
        storage: GuestStorage::default(),
        boot: BootArtifacts {
            kernel: artifacts_dir.join("disk.img"),
            initramfs: None,
            firmware: Some(artifacts_dir.join("nvram.bin")),
            cmdline: None,
        },
    })
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
        &format!(
            "{}_API_KEY=demo-{}\n",
            guest.guest_id.to_uppercase(),
            guest.guest_id
        ),
        0o644,
    )?;
    write_host_file_if_missing(&home.join(".bashrc"), "# motlie v1.45 demo bashrc\n", 0o644)?;
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

fn port_offset(seed: &str) -> u16 {
    seed.bytes().fold(0u16, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u16::from(byte))
    }) % 10_000
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
