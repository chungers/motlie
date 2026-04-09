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
use motlie_vmm::ssh::{self, ExecOutput, PtyRequest, SshProxyConfig, new_guest_registry};
use tokio::sync::mpsc;

use crate::terminal::{HarnessTerminalSession, TerminalBackendKind};
use crate::{
    APT_UPDATE_COMMAND, DynError, HarnessInstance, PACKAGE_MANAGER_QUIESCENT_COMMAND, demo_guest,
    ensure_file_exists, print_instance_details, seed_host_mounts,
};

pub async fn run_shell(
    base_dir: &Path,
    artifacts_dir: &Path,
    instance: &HarnessInstance,
    allocator_config: GuestNetAllocatorConfig,
    terminal_backend: TerminalBackendKind,
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
    tokio::spawn(ssh::run_proxy(
        proxy_config.clone(),
        Arc::clone(&guest_registry),
    ));
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

    let mut allocator = GuestNetAllocator::new(allocator_config)?;
    let mut handles: HashMap<String, VmHandle> = HashMap::new();
    let mut terminals: HashMap<String, HarnessTerminalSession> = HashMap::new();
    let mut stdout = io::stdout();

    println!("=== motlie-vmm harness shell ===");
    println!("Harness interactive/manual mode over the extracted vmm lifecycle API");
    println!("Terminal backend: {terminal_backend}");
    println!(
        "Commands: help | boot <guest> | ready <guest> | exec <guest> <cmd> | validate <guest> | pty-open <guest> <session> | pty-send <session> <text> | pty-send-line <session> <text> | pty-read <session> [timeout_ms] | pty-expect <session> <text> | pty-expect-screen <session> <text> | pty-resize <session> <cols> <rows> | pty-screen <session> | shutdown <guest> | status | guests | capacity | where [guest] | quit"
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
        } else if trimmed == "capacity" {
            print_capacity(&allocator);
            Ok(())
        } else if trimmed == "where" {
            print_where(
                instance,
                &proxy_config,
                &allocator,
                terminal_backend,
                None,
                &handles,
            );
            Ok(())
        } else if let Some(rest) = trimmed.strip_prefix("where ") {
            let guest_id = rest.trim();
            let guest = (!guest_id.is_empty()).then_some(guest_id);
            print_where(
                instance,
                &proxy_config,
                &allocator,
                terminal_backend,
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
        } else if let Some(rest) = trimmed.strip_prefix("pty-open ") {
            pty_open(rest, instance, &handles, &mut terminals, terminal_backend).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-send-line ") {
            pty_send_line(rest, &terminals).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-send ") {
            pty_send(rest, &terminals).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-read ") {
            pty_read(rest, &terminals).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-expect ") {
            pty_expect(rest, &terminals).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-expect-screen ") {
            pty_expect_screen(rest, &terminals).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-resize ") {
            pty_resize(rest, &terminals).await
        } else if let Some(rest) = trimmed.strip_prefix("pty-screen ") {
            pty_screen(rest, &terminals)
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

    for (_session, terminal) in terminals.drain() {
        let _ = terminal.persist_artifacts();
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
    println!("pty-open <guest> <name>  open a PTY session and start VTE/transcript capture");
    println!("pty-send <name> <text>   send raw PTY text");
    println!("pty-send-line <name> <text>");
    println!("pty-read <name> [ms]     read PTY output for up to the timeout");
    println!("pty-expect <name> <text> read until output contains text");
    println!("pty-expect-screen <name> <text> wait until the rendered VTE screen contains text");
    println!("pty-resize <name> <cols> <rows>");
    println!("pty-screen <name>        print the rendered VTE screen snapshot");
    println!("shutdown <guest>         stop a guest");
    println!("status | guests          show active guests");
    println!("capacity                 show allocator capacity and address plan");
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
    allocator: &GuestNetAllocator,
    terminal_backend: TerminalBackendKind,
    guest_id: Option<&str>,
    handles: &HashMap<String, VmHandle>,
) {
    println!("namespace={}", instance.namespace.prefix);
    println!("temp_root={}", instance.namespace.temp_root.display());
    println!("demo_root={}", instance.demo_root.display());
    println!("socket_root={}", instance.socket_root.display());
    println!("proxy=ssh://localhost:{}", proxy_config.listen.port());
    println!("terminal_backend={terminal_backend}");
    print_capacity(allocator);

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

fn print_capacity(allocator: &GuestNetAllocator) {
    let config = allocator.config();
    println!("allocator.first_cid={}", config.first_cid);
    println!(
        "allocator.capacity={} next_slot={} remaining={}",
        allocator.capacity().unwrap_or_default(),
        allocator.next_slot(),
        allocator.remaining_capacity().unwrap_or_default(),
    );
    println!(
        "allocator.admin_pool={} -> /{} first_subnet_slot={} host_offset={} guest_offset={}",
        config.admin_pool.base,
        config.admin_pool.guest_prefix_len,
        config.admin_pool.first_subnet_slot,
        config.admin_pool.host_offset,
        config.admin_pool.guest_offset,
    );
    println!(
        "allocator.egress_pool={} -> /{} first_subnet_slot={} host_offset={} guest_offset={} dns_offset={}",
        config.egress_pool.base,
        config.egress_pool.guest_prefix_len,
        config.egress_pool.first_subnet_slot,
        config.egress_pool.host_offset,
        config.egress_pool.guest_offset,
        config.egress_pool.dns_offset.unwrap_or_default(),
    );
}

async fn pty_open(
    rest: &str,
    instance: &HarnessInstance,
    handles: &HashMap<String, VmHandle>,
    terminals: &mut HashMap<String, HarnessTerminalSession>,
    terminal_backend: TerminalBackendKind,
) -> Result<(), DynError> {
    let mut parts = rest.split_whitespace();
    let guest_id = parts.next().unwrap_or("").trim();
    let session_name = parts.next().unwrap_or("").trim();
    if guest_id.is_empty() || session_name.is_empty() {
        return Err("pty-open <guest> <session>".into());
    }
    if terminals.contains_key(session_name) {
        return Err(format!("PTY session '{session_name}' already exists").into());
    }
    let handle = handles
        .get(guest_id)
        .ok_or_else(|| format!("unknown guest '{guest_id}'"))?;
    let request = PtyRequest::default();
    let guest_session = handle
        .open_pty(request.clone(), Duration::from_secs(10))
        .await?;
    let session_root = instance
        .namespace
        .temp_root
        .join(format!("{}-shell-pty", instance.namespace.prefix))
        .join(session_name);
    let terminal = HarnessTerminalSession::new(
        format!("{guest_id}:{session_name}"),
        guest_session,
        &request,
        terminal_backend,
        session_root.join("pty-transcript.ndjson"),
        session_root.join("pty-screen.json"),
        session_root.join("pty-screen.svg"),
        session_root.join("pty.cast"),
    );
    println!(
        "ok: opened PTY {} for {} backend={} transcript={} screen={} screen_svg={} cast={}",
        session_name,
        guest_id,
        terminal.backend(),
        terminal.transcript_path().display(),
        terminal.screen_path().display(),
        terminal.screen_svg_path().display(),
        terminal.asciicast_path().display()
    );
    terminals.insert(session_name.to_string(), terminal);
    Ok(())
}

async fn pty_send(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let (session_name, text) = split_session_and_text(rest, "pty-send <session> <text>")?;
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    terminal.send(text.as_bytes()).await?;
    println!("ok: sent {} bytes", text.len());
    Ok(())
}

async fn pty_send_line(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let (session_name, text) = split_session_and_text(rest, "pty-send-line <session> <text>")?;
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    terminal.send_line(text).await?;
    println!("ok: sent line '{}'", text);
    Ok(())
}

async fn pty_read(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let mut parts = rest.split_whitespace();
    let session_name = parts.next().unwrap_or("").trim();
    if session_name.is_empty() {
        return Err("pty-read <session> [timeout_ms]".into());
    }
    let timeout_ms = parts
        .next()
        .map(str::parse::<u64>)
        .transpose()?
        .unwrap_or(1000);
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    let read = terminal.read_for(Duration::from_millis(timeout_ms)).await?;
    print_pty_read(&read.output, read.exit_status, read.eof, read.closed);
    Ok(())
}

async fn pty_expect(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let (session_name, text) = split_session_and_text(rest, "pty-expect <session> <text>")?;
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    let read = terminal
        .read_until_contains("pty_expect", text, Duration::from_secs(10))
        .await?;
    print_pty_read(&read.output, read.exit_status, read.eof, read.closed);
    Ok(())
}

async fn pty_expect_screen(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let (session_name, text) = split_session_and_text(rest, "pty-expect-screen <session> <text>")?;
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    let read = terminal
        .read_until_screen_contains("pty_expect_screen", text, Duration::from_secs(10))
        .await?;
    print_pty_read(&read.output, read.exit_status, read.eof, read.closed);
    println!("ok: PTY screen for {} contains '{}'", session_name, text);
    Ok(())
}

async fn pty_resize(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let mut parts = rest.split_whitespace();
    let session_name = parts.next().unwrap_or("").trim();
    let cols = parts
        .next()
        .ok_or_else(|| "pty-resize <session> <cols> <rows>".to_string())?
        .parse::<u32>()?;
    let rows = parts
        .next()
        .ok_or_else(|| "pty-resize <session> <cols> <rows>".to_string())?
        .parse::<u32>()?;
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    terminal.resize(cols, rows, 0, 0).await?;
    println!("ok: resized {} to {}x{}", session_name, cols, rows);
    Ok(())
}

fn pty_screen(
    rest: &str,
    terminals: &HashMap<String, HarnessTerminalSession>,
) -> Result<(), DynError> {
    let session_name = rest.trim();
    if session_name.is_empty() {
        return Err("pty-screen <session>".into());
    }
    let terminal = terminals
        .get(session_name)
        .ok_or_else(|| format!("unknown PTY session '{session_name}'"))?;
    let screen = terminal.snapshot()?;
    println!(
        "backend={} mode={:?} rows={} cols={} cursor=({}, {})",
        screen.backend,
        screen.screen_mode,
        screen.rows,
        screen.cols,
        screen.cursor_row,
        screen.cursor_col
    );
    println!("{}", screen.visible_text);
    Ok(())
}

fn split_session_and_text<'a>(
    rest: &'a str,
    usage: &'static str,
) -> Result<(&'a str, &'a str), DynError> {
    let mut parts = rest.trim().splitn(2, char::is_whitespace);
    let session_name = parts.next().unwrap_or("").trim();
    let text = parts.next().unwrap_or("").trim();
    if session_name.is_empty() || text.is_empty() {
        return Err(usage.into());
    }
    Ok((session_name, text))
}

fn print_pty_read(output: &str, exit_status: Option<u32>, eof: bool, closed: bool) {
    if !output.is_empty() {
        print!("{output}");
        if !output.ends_with('\n') {
            println!();
        }
    }
    println!("pty: exit_status={exit_status:?} eof={eof} closed={closed}");
}

fn print_guest_where(guest_id: &str, demo_root: &Path, handle: &VmHandle) {
    println!("[{guest_id}]");
    println!(
        "  home_host={}",
        demo_root.join(format!("{guest_id}-home")).display()
    );
    println!(
        "  workspace_host={}",
        demo_root.join(format!("{guest_id}-workspace")).display()
    );
    println!(
        "  agent_state_host={}",
        demo_root.join(format!("{guest_id}-agent-state")).display()
    );
    println!(
        "  runtime_dir={}",
        handle.runtime_paths.runtime_dir.display()
    );
    println!("  launch_dir={}", handle.runtime_paths.launch_dir.display());
    println!(
        "  cloud_init_dir={}",
        handle.runtime_paths.cloud_init_dir.display()
    );
    println!("  api_socket={}", handle.runtime_paths.api_socket.display());
    println!(
        "  vnet_socket={}",
        handle.runtime_paths.vnet_socket.display()
    );
    println!(
        "  vsock_socket={}",
        handle.runtime_paths.vsock_socket.display()
    );
    println!("  cid={}", handle.net_assignment.cid);
    println!("  slot={}", handle.net_assignment.slot);
    println!("  admin_subnet={}", handle.net_assignment.admin_subnet);
    println!("  admin_host={}", handle.net_assignment.admin_ipv4.host);
    println!("  admin_guest={}", handle.net_assignment.admin_ipv4.guest);
    println!("  admin_mac={:02x?}", handle.net_assignment.admin_mac);
    println!("  egress_subnet={}", handle.net_assignment.egress_subnet);
    println!("  egress_host={}", handle.net_assignment.egress_ipv4.host);
    println!("  egress_guest={}", handle.net_assignment.egress_ipv4.guest);
    println!("  egress_dns={}", handle.net_assignment.egress_ipv4.dns);
    println!("  egress_mac={:02x?}", handle.net_assignment.egress_mac);
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

    let guest = demo_guest(
        guest_id,
        artifacts_dir,
        &instance.demo_root,
        &instance.namespace,
    )?;
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
    println!(
        "waiting for {}: api socket + guestfs + ssh bridge + exec-ready",
        guest_id
    );
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
            "sudo: passwordless sudo available",
            "/bin/sh -lc 'sudo -n true && echo SUDO_OK'".to_string(),
            "SUDO_OK".to_string(),
            Duration::from_secs(10),
        ),
        (
            "tooling: git preinstalled",
            "/bin/sh -lc 'git --version | grep -q \"^git version \" && echo GIT_OK'".to_string(),
            "GIT_OK".to_string(),
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
        (
            "apt: package manager quiescent",
            PACKAGE_MANAGER_QUIESCENT_COMMAND.to_string(),
            "PKG_IDLE_OK".to_string(),
            Duration::from_secs(65),
        ),
        (
            "apt: package index refresh",
            APT_UPDATE_COMMAND.to_string(),
            "APT_OK".to_string(),
            Duration::from_secs(60),
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
