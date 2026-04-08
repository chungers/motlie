use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::Path;
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
use motlie_vmm::ssh::{self, ExecOutput, SshProxyConfig, new_guest_registry};
use tokio::sync::mpsc;

use crate::{DynError, HarnessInstance, demo_guest, ensure_file_exists, print_instance_details, seed_host_mounts};

pub async fn run_shell(
    base_dir: &Path,
    artifacts_dir: &Path,
    instance: &HarnessInstance,
) -> Result<(), DynError> {
    ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))?;
    ensure_file_exists(&artifacts_dir.join("Image"))?;
    std::fs::create_dir_all(&instance.socket_root)?;

    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();
    let proxy_config = SshProxyConfig {
        listen: std::net::SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            instance.proxy_port,
        ),
    };
    tokio::spawn(ssh::run_proxy(proxy_config.clone(), Arc::clone(&guest_registry)));
    print_instance_details(instance, &proxy_config);

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
    let mut handles: HashMap<String, VmHandle> = HashMap::new();
    let mut stdout = io::stdout();

    println!("=== motlie-vmm harness shell ===");
    println!("Harness interactive/manual mode over the extracted vmm lifecycle API");
    println!(
        "Commands: help | boot <guest> | ready <guest> | exec <guest> <cmd> | validate <guest> | shutdown <guest> | status | guests | where [guest] | quit"
    );

    let mut lines = spawn_stdin_reader();
    print_prompt(&mut stdout)?;
    while let Some(line_result) = lines.recv().await {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
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
            print_where(instance, &proxy_config, None, &handles);
            Ok(())
        } else if let Some(rest) = trimmed.strip_prefix("where ") {
            let guest_id = rest.trim();
            let guest = (!guest_id.is_empty()).then_some(guest_id);
            print_where(instance, &proxy_config, guest, &handles);
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
                    artifacts_dir,
                    instance,
                    base_dir,
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
    println!("quit                     exit the harness shell");
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
    instance: &HarnessInstance,
    proxy_config: &SshProxyConfig,
    guest_id: Option<&str>,
    handles: &HashMap<String, VmHandle>,
) {
    println!("namespace={}", instance.namespace.prefix);
    println!("temp_root={}", instance.namespace.temp_root.display());
    println!("demo_root={}", instance.demo_root.display());
    println!("socket_root={}", instance.socket_root.display());
    println!("proxy=ssh://localhost:{}", proxy_config.listen.port());

    match guest_id {
        Some(guest_id) => {
            let Some(handle) = handles.get(guest_id) else {
                println!("guest '{guest_id}' is not currently booted");
                return;
            };
            print_guest_where(guest_id, &instance.demo_root, handle);
        }
        None => {
            for (guest_id, handle) in handles {
                print_guest_where(guest_id, &instance.demo_root, handle);
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
    print!("v14-harness> ");
    stdout.flush()?;
    Ok(())
}

async fn boot_guest(
    guest_id: &str,
    artifacts_dir: &Path,
    instance: &HarnessInstance,
    base_dir: &Path,
    ca: &Arc<SshCa>,
    runtime: &Arc<Runtime>,
    allocator: &mut GuestNetAllocator,
    handles: &mut HashMap<String, VmHandle>,
) -> Result<(), DynError> {
    if handles.contains_key(guest_id) {
        return Err(format!("guest '{guest_id}' already booted").into());
    }

    let guest = demo_guest(guest_id, artifacts_dir, &instance.demo_root, &instance.namespace);
    seed_host_mounts(&guest)?;

    let prepared = prepare(
        PrepareRequest {
            guest,
            namespace: instance.namespace.clone(),
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
        "ok: booted {} pid={:?} api={} proxy=127.0.0.1:{}",
        guest_id,
        handle.pid,
        handle.runtime_paths.api_socket.display(),
        instance.proxy_port,
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

    let mut passed = 0usize;
    let mut failed = 0usize;
    for (label, cmd, needle, timeout) in checks {
        let output = handle.exec(&cmd, timeout).await?;
        if output.exit_code == 0 && output.stdout.contains(&needle) {
            println!("ok: {label}");
            passed += 1;
        } else {
            println!(
                "fail: {} exit={} stdout={} stderr={}",
                label,
                output.exit_code,
                output.stdout.trim(),
                output.stderr.trim()
            );
            failed += 1;
        }
    }

    println!("validation: {} passed, {} failed", passed, failed);
    if failed == 0 {
        Ok(())
    } else {
        Err(format!("guest '{guest_id}' validation failed").into())
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
