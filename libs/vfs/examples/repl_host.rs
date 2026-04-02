//! repl_host: v1 host-side filesystem server with admin REPL.
//!
//! Starts one or more guest-scoped FsServer instances with MemOverlay,
//! listens on one Unix socket per guest, and serves filesystem
//! operations. Exposes every MemOverlay API operation through a command
//! interface.
//!
//! # Input modes
//!
//! The REPL detects how stdin is connected and adapts:
//!
//! **Interactive (stdin is a TTY):**
//!   Rustyline REPL with line editing, history, and tab completion.
//!   Server runs until `quit` command or Ctrl-D.
//!
//! **Pipe then interactive (`cat script.vfs - | repl_host ...`):**
//!   Reads and executes piped commands line by line.
//!   After pipe EOF, reopens `/dev/tty` for interactive rustyline REPL.
//!   Server keeps running throughout.
//!
//! **Pure pipe (`cat script.vfs | repl_host ...`):**
//!   Reads and executes piped commands line by line.
//!   After pipe EOF, server keeps running and serves guest filesystem
//!   connections until SIGTERM/SIGINT/SIGHUP is received.
//!   Use this mode for automated/agent-driven setups.
//!
//! In all modes, the `quit` command in the input stream shuts down
//! the server immediately.
//!
//! # Usage
//!
//! ```bash
//! # Interactive
//! cargo run --example repl_host --features vsock -- --tag alice-home --dir ~/alice
//!
//! # Multi-mount (repeat --mount)
//! cargo run --example repl_host --features vsock -- \
//!     --socket /tmp/motlie-vfs.vsock_5000 \
//!     --mount alice-home=~/alice \
//!     --mount workspace=~/workspace
//!
//! # Multi-guest (repeat --guest and guest-qualified --mount)
//! cargo run --example repl_host --features vsock -- \
//!     --guest alice=/tmp/motlie-vfs-alice.vsock_5000 \
//!     --mount alice:alice-home=~/alice \
//!     --mount alice:alice-workspace=~/workspace \
//!     --guest bob=/tmp/motlie-vfs-bob.vsock_5000 \
//!     --mount bob:bob-home=~/bob \
//!     --mount bob:bob-workspace=~/workspace-bob
//!
//! # Empty admin mode, provision from REPL script
//! cat setup-multiguest.sh.vfs | cargo run --example repl_host --features vsock -- --empty
//!
//! # Script then interactive
//! cat setup-alice.sh.vfs - | cargo run --example repl_host --features vsock -- --tag alice-home
//!
//! # Script only (server stays alive until signaled)
//! cat setup-alice.sh.vfs | cargo run --example repl_host --features vsock -- --tag alice-home
//!
//! # Agent-driven (write commands to stdin, server stays alive)
//! echo "layer creds 0" | cargo run --example repl_host --features vsock -- --tag alice-home
//! ```
//!
//! # Options
//!
//!   --empty                 start with no guest and provision from REPL
//!   --socket <path>         vsock socket path (single-guest mode)
//!   --guest <id=socket>     add a guest-scoped FsServer and listener
//!   --mount <tag=dir>       add a mount in single-guest mode
//!   --mount <id:tag=dir>    add a mount to one guest in multi-guest mode
//!   --tag <name>            mount tag (single-guest mode)
//!   --dir <path>            host backing directory (single-guest mode)

use std::io::{self, BufRead};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex as StdMutex;
use std::sync::Arc;
use std::collections::HashMap;

use anyhow::Result;
use bytes::Bytes;
use tokio::net::UnixListener;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use motlie_vfs::core::overlay::OverlayAttrs;
use motlie_vfs::core::server::FsServer;
use motlie_vfs::vsock::handler::VsockConnectionHandler;

#[derive(Clone)]
struct GuestConfig {
    name: String,
    socket_path: String,
    mounts: Vec<ConfiguredMount>,
    identity: Option<GuestIdentity>,
}

#[derive(Clone)]
struct ConfiguredMount {
    tag: String,
    guest_path: Option<String>,
    host_path: PathBuf,
}

#[derive(Clone, Copy)]
struct GuestIdentity {
    uid: u32,
    gid: u32,
}

struct GuestRuntime {
    server: Arc<FsServer>,
    socket_path: String,
    mounts: Vec<ConfiguredMount>,
    identity: Option<GuestIdentity>,
}

struct AdminState {
    guests: HashMap<String, GuestRuntime>,
    guest_order: Vec<String>,
    current_guest: String,
    multi_guest: bool,
    comment_stdout: bool,
    runtime: Handle,
    sockets_for_cleanup: Arc<StdMutex<Vec<String>>>,
}

enum MountSpec {
    Single { tag: String, dir: PathBuf },
    Guest { guest: String, tag: String, dir: PathBuf },
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut socket_path = "/tmp/motlie-vfs.vsock_5000".to_string();
    let mut tag = "alice-home".to_string();
    let mut host_dir: Option<PathBuf> = None;
    let mut single_mounts: Vec<(String, PathBuf)> = Vec::new();
    let mut guest_configs: Vec<GuestConfig> = Vec::new();
    let mut empty_mode = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--empty" => { empty_mode = true; i += 1; }
            "--socket" if i + 1 < args.len() => { socket_path = args[i + 1].clone(); i += 2; }
            "--guest" if i + 1 < args.len() => {
                let (guest_name, guest_socket) = parse_guest_spec(&args[i + 1])?;
                if guest_configs.iter().any(|cfg| cfg.name == guest_name) {
                    anyhow::bail!("duplicate --guest '{}'", guest_name);
                }
                guest_configs.push(GuestConfig {
                    name: guest_name,
                    socket_path: guest_socket,
                    mounts: Vec::new(),
                    identity: None,
                });
                i += 2;
            }
            "--mount" if i + 1 < args.len() => {
                match parse_mount_spec(&args[i + 1])? {
                    MountSpec::Single { tag, dir } => single_mounts.push((tag, dir)),
                    MountSpec::Guest { guest, tag, dir } => {
                        let Some(config) = guest_configs.iter_mut().find(|cfg| cfg.name == guest) else {
                            anyhow::bail!("guest '{guest}' must be declared with --guest before its --mount entries");
                        };
                        config.mounts.push(ConfiguredMount {
                            tag,
                            guest_path: None,
                            host_path: dir,
                        });
                    }
                }
                i += 2;
            }
            "--tag" if i + 1 < args.len() => { tag = args[i + 1].clone(); i += 2; }
            "--dir" if i + 1 < args.len() => { host_dir = Some(PathBuf::from(&args[i + 1])); i += 2; }
            other => { host_dir = Some(PathBuf::from(other)); i += 1; }
        }
    }

    let multi_guest = empty_mode || !guest_configs.is_empty();
    if multi_guest {
        if host_dir.is_some() {
            anyhow::bail!("cannot use --dir/positional path in multi-guest mode");
        }
        if tag != "alice-home" {
            anyhow::bail!("cannot use --tag in multi-guest mode");
        }
        if !single_mounts.is_empty() {
            anyhow::bail!("multi-guest mode requires --mount <guest>:<tag>=<dir>");
        }
        for config in &guest_configs {
            if config.mounts.is_empty() {
                anyhow::bail!("guest '{}' has no mounts; add at least one --mount {}:<tag>=<dir>", config.name, config.name);
            }
        }
    } else if !empty_mode {
        if !single_mounts.is_empty() && host_dir.is_some() {
            anyhow::bail!("cannot mix --mount with legacy --dir/positional path");
        }
        if !single_mounts.is_empty() && tag != "alice-home" {
            anyhow::bail!("cannot mix --mount with legacy --tag");
        }

        let tempdir = if single_mounts.is_empty() && host_dir.is_none() {
            Some(tempfile::tempdir()?)
        } else {
            None
        };
        if single_mounts.is_empty() {
            let host_path = match host_dir {
                Some(p) => {
                    std::fs::create_dir_all(&p)?;
                    p
                }
                None => {
                    let p = tempdir
                        .as_ref()
                        .map(|d| d.path().to_path_buf())
                        .ok_or_else(|| anyhow::anyhow!("internal tempdir setup failed"))?;
                    std::fs::create_dir_all(p.join("projects"))?;
                    std::fs::write(p.join("projects/README.md"), b"hello from host disk\n")?;
                    std::fs::write(p.join(".bashrc"), b"# bashrc\n")?;
                    eprintln!("Created temp host dir: {}", p.display());
                    p
                }
            };
            single_mounts.push((tag.clone(), host_path));
        }
        guest_configs.push(GuestConfig {
            name: "default".to_string(),
            socket_path,
            mounts: single_mounts.into_iter().map(|(tag, host_path)| ConfiguredMount {
                tag,
                guest_path: None,
                host_path,
            }).collect(),
            identity: None,
        });
        std::mem::forget(tempdir);
    }

    eprintln!("=== motlie-vfs repl_host ===");
    for config in &guest_configs {
        if multi_guest {
            eprintln!("Guest: {}", config.name);
        }
        eprintln!("  Socket: {}", config.socket_path);
        if let Some(identity) = config.identity {
            eprintln!("  Identity: uid={} gid={}", identity.uid, identity.gid);
        }
        eprintln!("  Mounts:");
        for mount in &config.mounts {
            if let Some(guest_path) = &mount.guest_path {
                eprintln!("    {}: {} -> {}", mount.tag, guest_path, mount.host_path.display());
            } else {
                eprintln!("    {} -> {}", mount.tag, mount.host_path.display());
            }
        }
    }
    eprintln!("");

    let mut admin_guests: HashMap<String, GuestRuntime> = HashMap::new();
    let mut guest_order = Vec::new();
    let sockets_for_cleanup = Arc::new(StdMutex::new(Vec::new()));
    let runtime = Handle::current();

    for config in guest_configs {
        let mut builder = FsServer::builder()
            .overlay(true)
            .events(256);
        for mount in &config.mounts {
            builder = builder.mount(&mount.tag, mount.host_path.clone(), false);
        }
        let server = Arc::new(builder.build()?);
        guest_order.push(config.name.clone());
        admin_guests.insert(config.name.clone(), GuestRuntime {
            server: Arc::clone(&server),
            socket_path: config.socket_path.clone(),
            mounts: config.mounts.clone(),
            identity: config.identity,
        });

        let _ = std::fs::remove_file(&config.socket_path);
        let listener = UnixListener::bind(&config.socket_path)?;
        sockets_for_cleanup.lock().expect("cleanup socket lock poisoned").push(config.socket_path.clone());
        spawn_guest_listener(&runtime, config.name.clone(), config.socket_path.clone(), listener, server);
    }

    eprintln!("Type 'help' for commands.");
    eprintln!("");

    let current_guest = guest_order.first().cloned().unwrap_or_default();
    let admin = AdminState {
        guests: admin_guests,
        guest_order,
        current_guest,
        multi_guest,
        comment_stdout: false,
        runtime,
        sockets_for_cleanup: Arc::clone(&sockets_for_cleanup),
    };

    // Spawn the admin input handler on a blocking thread
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    tokio::task::spawn_blocking(move || {
        run_input(admin);
        for socket_path in sockets_for_cleanup.lock().expect("cleanup socket lock poisoned").iter() {
            let _ = std::fs::remove_file(socket_path);
        }
        let _ = shutdown_tx.send(());
    });

    let _ = shutdown_rx.await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Input handling — adapts to TTY, pipe+TTY, or pure pipe
// ---------------------------------------------------------------------------

fn run_input(mut admin: AdminState) {
    let stdin_is_tty = atty::is(atty::Stream::Stdin);

    if stdin_is_tty {
        admin.comment_stdout = false;
        // Mode 1: Interactive TTY — rustyline REPL
        run_interactive_repl(&mut admin);
    } else {
        admin.comment_stdout = true;
        // Modes 2 & 3: Piped input — read stdin line by line
        eprintln!("--- reading commands from stdin ---");
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => { eprintln!("stdin read error: {e}"); break; }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
            eprintln!("vfs> {trimmed}");
            if dispatch_command(&mut admin, trimmed) == ControlFlow::Quit {
                eprintln!("shutting down (quit command)");
                return;
            }
        }
        eprintln!("--- stdin EOF ---");

        // Try to reopen /dev/tty for interactive mode (Mode 2: pipe then TTY)
        #[cfg(unix)]
        {
            if let Ok(_tty) = std::fs::File::open("/dev/tty") {
                // /dev/tty is available — the user piped a script but still has a terminal.
                // Switch to interactive rustyline REPL.
                eprintln!("--- entering interactive REPL (Ctrl-D to stop, server keeps running) ---");
                eprintln!("");
                admin.comment_stdout = false;
                run_interactive_repl(&mut admin);
                return;
            }
        }

        // Mode 3: Pure pipe, no TTY available.
        // Server keeps running for guest filesystem connections.
        // Wait for SIGTERM/SIGINT/SIGHUP.
        eprintln!("--- no TTY available, server running until signaled ---");
        eprintln!("--- send SIGTERM or SIGINT to stop ---");
        wait_for_signal();
        eprintln!("shutting down (signal received)");
    }
}

fn run_interactive_repl(admin: &mut AdminState) {
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to create REPL: {e}");
            return;
        }
    };

    loop {
        let prompt = if admin.multi_guest {
            format!("vfs[{}]> ", admin.current_guest)
        } else {
            "vfs> ".to_string()
        };
        let line = match rl.readline(&prompt) {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => { eprintln!("^C"); continue; }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => { eprintln!("REPL error: {e}"); break; }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let _ = rl.add_history_entry(trimmed);

        if dispatch_command(admin, trimmed) == ControlFlow::Quit {
            break;
        }
    }

    eprintln!("shutting down");
}

/// Block until SIGTERM, SIGINT, or SIGHUP is received.
fn wait_for_signal() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let running = Arc::new(AtomicBool::new(true));
    let r = Arc::clone(&running);
    ctrlc::set_handler(move || { r.store(false, Ordering::SeqCst); })
        .unwrap_or_else(|e| eprintln!("warning: failed to set signal handler: {e}"));

    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn spawn_guest_listener(
    runtime: &Handle,
    guest_name: String,
    socket_path: String,
    listener: UnixListener,
    server: Arc<FsServer>,
) {
    eprintln!("Listening on {socket_path} for guest '{guest_name}' (guest filesystem connections)");

    runtime.spawn(async move {
        loop {
            match listener.accept().await {
                Ok((mut stream, _addr)) => {
                    let tag = match motlie_vfs::vsock::read_tag_handshake(&mut stream).await {
                        Ok(tag) => tag,
                        Err(e) => {
                            eprintln!("[{guest_name}] connection handshake error: {e}");
                            continue;
                        }
                    };
                    if !server.has_mount(&tag) {
                        eprintln!("[{guest_name}] connection requested unknown tag: {tag}");
                        continue;
                    }
                    let handler = VsockConnectionHandler::new(Arc::clone(&server), &tag);
                    let guest_name_for_conn = guest_name.clone();
                    let tag_for_conn = tag.clone();
                    tokio::spawn(async move {
                        match handler.serve(stream).await {
                            Ok(()) => eprintln!("[{guest_name_for_conn}] connection handler closed cleanly for tag={tag_for_conn}"),
                            Err(e) => eprintln!("[{guest_name_for_conn}] connection handler error for tag={tag_for_conn}: {e}"),
                        }
                    });
                    eprintln!("[accepted guest connection guest={guest_name} tag={tag}]");
                }
                Err(e) => eprintln!("[{guest_name}] accept error: {e}"),
            }
        }
    });
}

fn parse_guest_spec(spec: &str) -> Result<(String, String)> {
    let Some((guest, socket_path)) = spec.split_once('=') else {
        anyhow::bail!("invalid --guest '{spec}'; expected <guest>=<socket>");
    };
    if guest.is_empty() || socket_path.is_empty() {
        anyhow::bail!("invalid --guest '{spec}'; guest and socket must be non-empty");
    }
    Ok((guest.to_string(), socket_path.to_string()))
}

fn parse_mount_spec(spec: &str) -> Result<MountSpec> {
    let Some((lhs, dir)) = spec.split_once('=') else {
        anyhow::bail!("invalid --mount '{spec}'; expected <tag>=<dir> or <guest>:<tag>=<dir>");
    };
    if lhs.is_empty() || dir.is_empty() {
        anyhow::bail!("invalid --mount '{spec}'; left-hand side and dir must be non-empty");
    }
    let path = PathBuf::from(dir);
    std::fs::create_dir_all(&path)?;

    if let Some((guest, tag)) = lhs.split_once(':') {
        if guest.is_empty() || tag.is_empty() {
            anyhow::bail!("invalid --mount '{spec}'; guest and tag must be non-empty");
        }
        return Ok(MountSpec::Guest {
            guest: guest.to_string(),
            tag: tag.to_string(),
            dir: path,
        });
    }

    Ok(MountSpec::Single {
        tag: lhs.to_string(),
        dir: path,
    })
}

fn parse_repl_mount_target(spec: &str) -> Result<ConfiguredMount> {
    let Some((tag, rhs)) = spec.split_once('=') else {
        anyhow::bail!("invalid mount spec '{spec}'; expected <tag>=<guest_path>,<host_path>");
    };
    let (guest_path, host_path) = match rhs.split_once(',') {
        Some((guest_path, host_path)) if !guest_path.is_empty() && !host_path.is_empty() => {
            (Some(guest_path.to_string()), PathBuf::from(host_path))
        }
        None if !rhs.is_empty() => (None, PathBuf::from(rhs)),
        _ => anyhow::bail!("invalid mount spec '{spec}'; expected <tag>=<guest_path>,<host_path>"),
    };
    std::fs::create_dir_all(&host_path)?;
    Ok(ConfiguredMount {
        tag: tag.to_string(),
        guest_path,
        host_path,
    })
}

fn provision_guest(
    admin: &mut AdminState,
    guest_name: &str,
    socket_path: &str,
    identity: Option<GuestIdentity>,
) -> Result<()> {
    if guest_name.is_empty() || socket_path.is_empty() {
        anyhow::bail!("provision requires non-empty guest and socket");
    }
    if admin.guests.contains_key(guest_name) {
        anyhow::bail!("guest '{guest_name}' already provisioned");
    }

    let server = Arc::new(
        FsServer::builder()
            .overlay(true)
            .events(256)
            .build()?,
    );
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)?;
    admin
        .sockets_for_cleanup
        .lock()
        .expect("cleanup socket lock poisoned")
        .push(socket_path.to_string());
    spawn_guest_listener(
        &admin.runtime,
        guest_name.to_string(),
        socket_path.to_string(),
        listener,
        Arc::clone(&server),
    );

    admin.guests.insert(
        guest_name.to_string(),
        GuestRuntime {
            server,
            socket_path: socket_path.to_string(),
            mounts: Vec::new(),
            identity,
        },
    );
    admin.guest_order.push(guest_name.to_string());
    if admin.current_guest.is_empty() {
        admin.current_guest = guest_name.to_string();
    }
    admin.multi_guest = true;
    Ok(())
}

fn add_guest_mount(admin: &mut AdminState, guest_name: &str, mount: ConfiguredMount) -> Result<()> {
    let Some(runtime) = admin.guests.get_mut(guest_name) else {
        anyhow::bail!("unknown guest '{guest_name}'");
    };
    if runtime.server.has_mount(&mount.tag) {
        anyhow::bail!("guest '{guest_name}' already has mount '{}'", mount.tag);
    }
    runtime
        .server
        .add_mount(&mount.tag, mount.host_path.clone(), false)?;
    runtime.mounts.push(mount);
    Ok(())
}

fn guest_login_name(guest_name: &str) -> &str {
    guest_name
}

fn guest_home(runtime: &GuestRuntime, guest_name: &str) -> String {
    runtime
        .mounts
        .iter()
        .find_map(|mount| {
            let guest_path = mount.guest_path.as_deref()?;
            if guest_path == format!("/home/{guest_name}") {
                Some(guest_path.to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| format!("/home/{guest_name}"))
}

fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn render_mounts_yaml(runtime: &GuestRuntime) -> Result<String> {
    let mut out = String::from("mounts:\n");
    for mount in &runtime.mounts {
        let Some(guest_path) = &mount.guest_path else {
            anyhow::bail!("mount '{}' is missing guest_path; cannot render mounts.yaml", mount.tag);
        };
        writeln!(&mut out, "  - tag: {}", mount.tag)?;
        writeln!(&mut out, "    guest_path: {}", guest_path)?;
        writeln!(&mut out, "    read_only: false")?;
    }
    Ok(out)
}

fn render_cloud_init(guest_name: &str, runtime: &GuestRuntime) -> Result<String> {
    let identity = runtime
        .identity
        .ok_or_else(|| anyhow::anyhow!("guest '{guest_name}' is missing uid/gid; provision with explicit uid/gid"))?;
    let mounts_yaml = render_mounts_yaml(runtime)?;
    let home_dir = guest_home(runtime, guest_name);

    let mut out = String::new();
    out.push_str("#cloud-config\n");
    out.push_str("write_files:\n");
    out.push_str("  - path: /etc/motlie-vfs/mounts.yaml\n");
    out.push_str("    owner: root:root\n");
    out.push_str("    permissions: '0644'\n");
    out.push_str("    content: |\n");
    for line in mounts_yaml.lines() {
        writeln!(&mut out, "      {line}")?;
    }
    out.push_str("runcmd:\n");
    out.push_str("  - |\n");
    writeln!(
        &mut out,
        "      if getent group {0} >/dev/null; then current_gid=\"$(getent group {0} | cut -d: -f3)\"; [ \"$current_gid\" = \"{1}\" ] || {{ echo \"gid mismatch for {0}: $current_gid != {1}\" >&2; exit 1; }}; else groupadd -g {1} {0}; fi",
        guest_login_name(guest_name),
        identity.gid
    )?;
    out.push_str("  - |\n");
    writeln!(
        &mut out,
        "      if id -u {0} >/dev/null 2>&1; then current_uid=\"$(id -u {0})\"; current_gid=\"$(id -g {0})\"; [ \"$current_uid\" = \"{1}\" ] && [ \"$current_gid\" = \"{2}\" ] || {{ echo \"uid/gid mismatch for {0}: $current_uid:$current_gid != {1}:{2}\" >&2; exit 1; }}; else useradd -m -u {1} -g {2} -s /bin/bash {0}; fi",
        guest_login_name(guest_name),
        identity.uid,
        identity.gid
    )?;
    out.push_str("  - |\n");
    writeln!(
        &mut out,
        "      install -d -m 0755 -o {0} -g {0} {1}",
        guest_login_name(guest_name),
        home_dir
    )?;
    out.push_str("  - |\n");
    writeln!(
        &mut out,
        "      install -d -m 0700 -o {0} -g {0} {1}/.ssh",
        guest_login_name(guest_name),
        home_dir
    )?;
    for mount in &runtime.mounts {
        if let Some(guest_path) = &mount.guest_path {
            if guest_path != &home_dir {
                out.push_str("  - |\n");
                writeln!(&mut out, "      install -d -m 0755 {}", guest_path)?;
            }
        }
    }
    out.push_str("  - |\n");
    out.push_str("      systemctl --no-block start motlie-vfs-guest.service\n");
    Ok(out)
}

fn render_launch_script(guest_name: &str, runtime: &GuestRuntime) -> Result<String> {
    let cloud_init = render_cloud_init(guest_name, runtime)?;
    if guest_name != "alice" && guest_name != "bob" {
        anyhow::bail!("launch prototype currently targets v1.1 demo guests alice/bob because launch-ch.sh still carries guest-specific runtime defaults");
    }
    let base_dir = "/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1";
    let mut out = String::new();
    out.push_str("#!/usr/bin/env bash\n");
    out.push_str("set -euo pipefail\n\n");
    out.push_str("# Generated by repl_host from the provisioned guest state.\n");
    out.push_str("# Rebuild the shared v1.1 base image with the current build-guest.sh so the\n");
    out.push_str("# guest includes cloud-init and consumes the seeded NoCloud directory at boot.\n\n");
    writeln!(&mut out, "GUEST_ID={}", shell_single_quote(guest_name))?;
    writeln!(&mut out, "BASE_DIR=\"${{BASE_DIR:-{}}}\"", base_dir)?;
    writeln!(&mut out, "SEED_DIR=\"${{SEED_DIR:-/tmp/motlie-vfs-cloud-init-${{GUEST_ID}}}}\"")?;
    out.push_str("INSTANCE_ID=\"${INSTANCE_ID:-${GUEST_ID}}\"\n");
    out.push_str("LOCAL_HOSTNAME=\"${LOCAL_HOSTNAME:-motlie-${GUEST_ID}}\"\n");
    out.push_str("mkdir -p \"$SEED_DIR\"\n");
    out.push_str("cat > \"$SEED_DIR/meta-data\" <<EOF\n");
    out.push_str("instance-id: ${INSTANCE_ID}\n");
    out.push_str("local-hostname: ${LOCAL_HOSTNAME}\n");
    out.push_str("EOF\n\n");
    out.push_str("cat > \"$SEED_DIR/user-data\" <<'EOF'\n");
    out.push_str(&cloud_init);
    if !cloud_init.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("EOF\n\n");
    out.push_str("echo \"Generated cloud-init assets in $SEED_DIR\"\n");
    out.push_str("echo \"Launching guest ${GUEST_ID} with seeded NoCloud dir ${SEED_DIR}\"\n");
    out.push_str("\"$BASE_DIR/launch-ch.sh\" --guest \"$GUEST_ID\" --cloud-init-dir \"$SEED_DIR\" \"$@\"\n");
    Ok(out)
}

fn execute_launch_script(script: &str) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new()?;
    use std::io::Write as _;
    temp.write_all(script.as_bytes())?;
    temp.flush()?;

    let status = Command::new("/bin/bash")
        .arg(temp.path())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("launch helper exited with status {status}");
    }
}

// ---------------------------------------------------------------------------
// Command dispatch — shared between all input modes
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum ControlFlow {
    Continue,
    Quit,
}

fn emit_line(comment_stdout: bool, line: impl AsRef<str>) {
    let line = line.as_ref();
    if comment_stdout {
        println!("# {line}");
    } else {
        println!("{line}");
    }
}

fn emit_status(admin: &AdminState, line: impl AsRef<str>) {
    emit_line(admin.comment_stdout, line);
}

fn emit_raw(text: impl AsRef<str>) {
    print!("{}", text.as_ref());
}

fn print_help(topic: Option<&str>, multi_guest: bool, comment_stdout: bool) {
    let out = |line: &str| emit_line(comment_stdout, line);

    match topic {
        Some("provision") => {
            out("provision <guest> <socket> <uid> <gid>");
            out("  Create one guest-scoped FsServer, record the guest uid/gid contract, and bind its Unix socket listener.");
            out("  Example:");
            out("    provision bob /tmp/motlie-vfs-bob.vsock_5000 1001 1001");
        }
        Some("mount") => {
            out("mount <guest> <tag>=<guest_path>,<host_path> [more...]");
            out("  Add one or more host-backed mount tags to an already-provisioned guest.");
            out("  guest_path is recorded for operator clarity and should match mounts.<guest>.yaml.");
            out("  FsServer routing still uses tag -> host_path on the host side.");
            out("  Example:");
            out("    mount bob bob-home=/home/bob,/tmp/motlie-vfs-demo/bob-home bob-workspace=/workspace,/tmp/motlie-vfs-demo/bob-workspace");
        }
        Some("launch") => {
            out("launch <guest>");
            out("  Generate a guest helper script and execute it via /bin/bash.");
            out("  The helper writes guest-specific cloud-init user-data and meta-data");
            out("  generated from the provisioned uid/gid and mount topology.");
            out("  launch-ch.sh then seeds those files into /var/lib/cloud/seed/nocloud/.");
            out("launch -script <guest>");
            out("  Render the helper shell script to stdout without executing it.");
        }
        Some("use") => {
            out("use <guest>");
            out("  Set the default target guest for subsequent admin commands.");
        }
        Some("guests") => {
            out("guests");
            out("  List provisioned guests, sockets, and configured mount counts.");
        }
        Some("layer") => {
            out("layer <name> <priority>");
            out("  Create or update a named overlay layer.");
        }
        Some("put") => {
            out("put <layer> <tag> <path> <content>");
            out("  Inject a file with default attrs into one tag within the current guest.");
        }
        Some("putattr") => {
            out("putattr <layer> <tag> <path> <uid> <gid> <mode> <content>");
            out("  Inject a file with explicit ownership and mode.");
        }
        Some("mkdir") => {
            out("mkdir <layer> <tag> <path> [mode]");
            out("  Create a synthetic directory in one overlay layer.");
            out("  Defaults ownership to the provisioned guest uid/gid when available.");
        }
        Some("whiteout") => {
            out("whiteout <layer> <tag> <path>");
            out("  Hide a lower-layer entry at the effective view.");
        }
        Some("rm") => {
            out("rm <layer> <tag> <path>");
            out("  Remove an overlay entry from one layer.");
        }
        Some("get") => {
            out("get <layer> <tag> <path>");
            out("  Read content from one overlay layer.");
        }
        Some("ls") => {
            out("ls <tag>");
            out("  List effective overlay entries for one tag.");
        }
        Some("lslayer") => {
            out("lslayer <layer> <tag>");
            out("  List entries from one layer only.");
        }
        Some("tree") => {
            out("tree [tag]");
            out("  Show the layered tree and effective winners. Without tag, show all tags.");
        }
        Some("quit") | Some("exit") => {
            out("quit");
            out("  Shut down repl_host.");
        }
        Some(other) => {
            emit_line(comment_stdout, format!("unknown help topic: {other}"));
            out("type 'help' for commands");
        }
        None => {
            if multi_guest {
                out("Multi-guest targeting:");
                out("  guests                                          — list configured guests");
                out("  use <guest>                                     — set default guest for subsequent commands");
                out("  <guest> <command ...>                           — run one command against a specific guest");
                out("  provision <guest> <socket> <uid> <gid>          — create one guest-scoped FsServer and listener");
                out("  mount <guest> <tag>=<guest_path>,<host_path>... — add one or more mounts to a guest");
                out("  launch <guest>                                  — generate and execute a guest launch helper");
                out("  launch -script <guest>                          — print the guest launch helper script");
                out("");
            }
            out("Layer management:");
            out("  layer <name> <priority>                         — create/update layer");
            out("  rmlayer <name>                                  — remove layer and all entries");
            out("  layers                                          — list all layers");
            out("");
            out("Content injection:");
            out("  put <layer> <tag> <path> <content>              — inject file (default attrs)");
            out("  putattr <layer> <tag> <path> <uid> <gid> <mode> <content>");
            out("                                                  — inject file with explicit attrs");
            out("  mkdir <layer> <tag> <path> [mode]               — create synthetic directory");
            out("");
            out("Suppression / removal:");
            out("  whiteout <layer> <tag> <path>                   — hide a lower-layer entry");
            out("  rm <layer> <tag> <path>                         — remove an overlay entry");
            out("");
            out("Inspection:");
            out("  get <layer> <tag> <path>                        — read content from a layer");
            out("  ls <tag>                                        — list effective overlay entries");
            out("  lslayer <layer> <tag>                           — list entries in a layer");
            out("  tree [tag]                                      — show layered tree (* = winner)");
            out("                                                    no tag = show all tags");
            out("");
            out("Other:");
            out("  help [command]                                  — show all commands or one command");
            out("  quit                                            — shut down server");
            out("");
            out("Input modes:");
            out("  Interactive:  stdin is a TTY → rustyline REPL");
            out("  Pipe + TTY:   cat script.vfs - | repl_host → script then REPL");
            out("  Pure pipe:    cat script.vfs | repl_host → script then serve until signaled");
        }
    }
}

fn dispatch_command(admin: &mut AdminState, line: &str) -> ControlFlow {
    let mut parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() { return ControlFlow::Continue; }

    match parts[0] {
        "guests" => {
            for guest in &admin.guest_order {
                let marker = if guest == &admin.current_guest { "*" } else { " " };
                if let Some(runtime) = admin.guests.get(guest) {
                    if let Some(identity) = runtime.identity {
                        emit_status(admin, format!("{marker} {guest} socket={} uid={} gid={} mounts={}", runtime.socket_path, identity.uid, identity.gid, runtime.mounts.len()));
                    } else {
                        emit_status(admin, format!("{marker} {guest} socket={} mounts={}", runtime.socket_path, runtime.mounts.len()));
                    }
                } else {
                    emit_status(admin, format!("{marker} {guest}"));
                }
            }
            return ControlFlow::Continue;
        }
        "use" if parts.len() == 2 => {
            if admin.guests.contains_key(parts[1]) {
                admin.current_guest = parts[1].to_string();
                emit_status(admin, format!("ok: using guest {}", admin.current_guest));
            } else {
                emit_status(admin, format!("error: unknown guest {}", parts[1]));
            }
            return ControlFlow::Continue;
        }
        "use" => {
            emit_status(admin, "error: use <guest>");
            return ControlFlow::Continue;
        }
        "provision" if parts.len() == 5 => {
            let uid = match parts[3].parse::<u32>() {
                Ok(uid) => uid,
                Err(_) => {
                    emit_status(admin, "error: uid must be a number");
                    return ControlFlow::Continue;
                }
            };
            let gid = match parts[4].parse::<u32>() {
                Ok(gid) => gid,
                Err(_) => {
                    emit_status(admin, "error: gid must be a number");
                    return ControlFlow::Continue;
                }
            };
            match provision_guest(admin, parts[1], parts[2], Some(GuestIdentity { uid, gid })) {
                Ok(()) => emit_status(admin, format!("ok: provision {} {} uid={} gid={}", parts[1], parts[2], uid, gid)),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
            return ControlFlow::Continue;
        }
        "provision" => {
            emit_status(admin, "error: provision <guest> <socket> <uid> <gid>");
            return ControlFlow::Continue;
        }
        "mount" if parts.len() >= 3 => {
            let guest_name = parts[1];
            for spec in &parts[2..] {
                match parse_repl_mount_target(spec)
                    .and_then(|mount| {
                        let mount_desc = if let Some(guest_path) = &mount.guest_path {
                            format!("{}:{} -> {}", mount.tag, guest_path, mount.host_path.display())
                        } else {
                            format!("{} -> {}", mount.tag, mount.host_path.display())
                        };
                        add_guest_mount(admin, guest_name, mount)?;
                        Ok(mount_desc)
                    }) {
                    Ok(mount_desc) => emit_status(admin, format!("ok: mount {guest_name} {mount_desc}")),
                    Err(e) => emit_status(admin, format!("error: {e}")),
                }
            }
            return ControlFlow::Continue;
        }
        "mount" => {
            emit_status(admin, "error: mount <guest> <tag>=<guest_path>,<host_path> [more...]");
            return ControlFlow::Continue;
        }
        "launch" if parts.len() == 2 || (parts.len() == 3 && parts[1] == "-script") => {
            let render_only = parts.len() == 3;
            let guest_name = if render_only { parts[2] } else { parts[1] };
            match admin.guests.get(guest_name).ok_or_else(|| anyhow::anyhow!("unknown guest '{guest_name}'"))
                .and_then(|runtime| render_launch_script(guest_name, runtime)) {
                Ok(script) => {
                    if render_only {
                        emit_raw(script);
                    } else {
                        emit_status(admin, format!("ok: launch {guest_name} (executing helper script)"));
                        if let Err(e) = execute_launch_script(&script) {
                            emit_status(admin, format!("error: {e}"));
                        }
                    }
                }
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
            return ControlFlow::Continue;
        }
        "launch" => {
            emit_status(admin, "error: launch <guest> | launch -script <guest>");
            return ControlFlow::Continue;
        }
        "help" => {
            print_help(parts.get(1).copied(), admin.multi_guest, admin.comment_stdout);
            return ControlFlow::Continue;
        }
        "quit" | "exit" => return ControlFlow::Quit,
        _ => {}
    }

    let target_guest = if admin.guests.contains_key(parts[0]) && parts.len() >= 2 {
        let guest = parts[0].to_string();
        parts.remove(0);
        guest
    } else {
        if admin.current_guest.is_empty() {
            emit_status(admin, "error: no guest selected; provision/use a guest or prefix commands with <guest>");
            return ControlFlow::Continue;
        }
        admin.current_guest.clone()
    };

    let Some(runtime) = admin.guests.get(&target_guest) else {
        emit_status(admin, format!("error: unknown guest {target_guest}"));
        return ControlFlow::Continue;
    };
    let overlay = match runtime.server.overlay() {
        Some(o) => o,
        None => { emit_status(admin, "error: overlay not enabled"); return ControlFlow::Continue; }
    };

    match parts[0] {
        // --- Layer management ---
        "layer" if parts.len() >= 3 => {
            let name = parts[1];
            match parts[2].parse::<u32>() {
                Ok(priority) => match overlay.put_layer(name, priority) {
                    Ok(()) => emit_status(admin, format!("ok: layer {name} priority={priority}")),
                    Err(e) => emit_status(admin, format!("error: {e}")),
                },
                Err(_) => emit_status(admin, "error: priority must be a number"),
            }
        }
        "rmlayer" if parts.len() >= 2 => {
            match overlay.remove_layer(parts[1]) {
                Ok(()) => emit_status(admin, format!("ok: rmlayer {}", parts[1])),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }
        "layers" => {
            let layers = overlay.layers();
            if layers.is_empty() { emit_status(admin, "(no layers)"); }
            for l in &layers {
                emit_status(admin, format!("  {} priority={} entries={}", l.name, l.priority, l.entry_count));
            }
        }

        // --- Content injection ---
        "put" if parts.len() >= 5 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let content = parts[4..].join(" ");
            match overlay.put(layer, tag, path, Bytes::from(content.clone())) {
                Ok(()) => emit_status(admin, format!("ok: put {layer} {tag} {path} ({} bytes)", content.len())),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }
        "putattr" if parts.len() >= 8 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let uid = match parts[4].parse::<u32>() { Ok(v) => v, Err(_) => { emit_status(admin, "error: uid must be a number"); return ControlFlow::Continue; } };
            let gid = match parts[5].parse::<u32>() { Ok(v) => v, Err(_) => { emit_status(admin, "error: gid must be a number"); return ControlFlow::Continue; } };
            let mode = match u32::from_str_radix(parts[6], 8) { Ok(v) => v, Err(_) => { emit_status(admin, "error: mode must be octal"); return ControlFlow::Continue; } };
            let content = parts[7..].join(" ");
            let attrs = OverlayAttrs { mode, uid, gid };
            match overlay.put_with_attrs(layer, tag, path, attrs, Bytes::from(content.clone())) {
                Ok(()) => emit_status(admin, format!("ok: putattr {layer} {tag} {path} uid={uid} gid={gid} mode={mode:o} ({} bytes)", content.len())),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }
        "mkdir" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let mode = if parts.len() >= 5 {
                u32::from_str_radix(parts[4], 8).unwrap_or(0o755)
            } else { 0o755 };
            let (uid, gid) = runtime
                .identity
                .map(|identity| (identity.uid, identity.gid))
                .unwrap_or((0, 0));
            let attrs = OverlayAttrs { mode, uid, gid };
            match overlay.create_dir(layer, tag, path, attrs) {
                Ok(()) => emit_status(admin, format!("ok: mkdir {layer} {tag} {path} mode={mode:o} uid={uid} gid={gid}")),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }

        // --- Suppression / removal ---
        "whiteout" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.whiteout(layer, tag, path) {
                Ok(()) => emit_status(admin, format!("ok: whiteout {layer} {tag} {path}")),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }
        "rm" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.remove(layer, tag, path) {
                Ok(()) => emit_status(admin, format!("ok: rm {layer} {tag} {path}")),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }

        // --- Inspection ---
        "get" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.get(layer, tag, path) {
                Some(data) => {
                    match std::str::from_utf8(&data) {
                        Ok(s) => emit_raw(format!("{s}\n")),
                        Err(_) => emit_status(admin, format!("({} bytes, binary)", data.len())),
                    }
                }
                None => emit_status(admin, "(not found)"),
            }
        }
        "ls" if parts.len() >= 2 => {
            let tag = parts[1];
            let entries = overlay.list_effective(tag);
            if entries.is_empty() { emit_status(admin, format!("(no overlay entries for tag '{tag}')")); }
            for entry in &entries {
                emit_status(admin, format!("  {:?} {} uid={} gid={} mode={:o}", entry.kind, entry.path, entry.uid, entry.gid, entry.mode));
            }
            emit_status(admin, format!("({} entries)", entries.len()));
        }
        "lslayer" if parts.len() >= 3 => {
            let (layer, tag) = (parts[1], parts[2]);
            let entries = overlay.list_layer(layer, tag);
            if entries.is_empty() { emit_status(admin, format!("(no entries in layer '{layer}' for tag '{tag}')")); }
            for entry in &entries {
                emit_status(admin, format!("  {:?} {} uid={} gid={} mode={:o}", entry.kind, entry.path, entry.uid, entry.gid, entry.mode));
            }
            emit_status(admin, format!("({} entries)", entries.len()));
        }

        // --- Tree view ---
        "tree" if parts.len() >= 2 => {
            let tag = parts[1];
            let layers = overlay.layers();
            if layers.is_empty() {
                emit_status(admin, "(no layers)");
            } else {
                // Collect effective entries to show which layer wins
                let effective = overlay.list_effective(tag);
                let mut winner: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                for e in &effective {
                    winner.insert(e.path.clone(), e.layer.clone());
                }

                emit_status(admin, format!("tag: {tag}"));
                emit_status(admin, "");
                for l in &layers {
                    let mut entries = overlay.list_layer(&l.name, tag);
                    entries.sort_by(|a, b| a.path.cmp(&b.path));
                    if entries.is_empty() { continue; }
                    emit_status(admin, format!("  layer: {} (priority={})", l.name, l.priority));
                    for entry in &entries {
                        let eff = if winner.get(&entry.path).map(|w| w == &l.name).unwrap_or(false) {
                            "*"
                        } else {
                            " "  // shadowed by higher-priority layer
                        };
                        emit_status(admin, format!("   {eff} {:?} {} uid={} gid={} mode={:o}",
                            entry.kind, entry.path, entry.uid, entry.gid, entry.mode));
                    }
                    emit_status(admin, "");
                }
                emit_status(admin, "  (* = effective winner)");
            }
        }
        "tree" => {
            let layers = overlay.layers();
            let tags = overlay.tags();
            if layers.is_empty() {
                emit_status(admin, "(no layers)");
            } else if tags.is_empty() {
                emit_status(admin, "(no entries)");
            } else {
                for tag in &tags {
                    emit_status(admin, format!("tag: {tag}"));
                    let effective = overlay.list_effective(tag);
                    let mut winner: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                    for e in &effective {
                        winner.insert(e.path.clone(), e.layer.clone());
                    }
                    for l in &layers {
                        let mut entries = overlay.list_layer(&l.name, tag);
                        entries.sort_by(|a, b| a.path.cmp(&b.path));
                        if entries.is_empty() { continue; }
                        emit_status(admin, format!("  layer: {} (priority={})", l.name, l.priority));
                        for entry in &entries {
                            let eff = if winner.get(&entry.path).map(|w| w == &l.name).unwrap_or(false) {
                                "*"
                            } else {
                                " "
                            };
                            emit_status(admin, format!("   {eff} {:?} {} uid={} gid={} mode={:o}",
                                entry.kind, entry.path, entry.uid, entry.gid, entry.mode));
                        }
                    }
                    emit_status(admin, "");
                }
                emit_status(admin, "(* = effective winner)");
            }
        }

        _ => {
            emit_status(admin, format!("unknown command: {line}"));
            emit_status(admin, "type 'help' for commands");
        }
    }

    ControlFlow::Continue
}
