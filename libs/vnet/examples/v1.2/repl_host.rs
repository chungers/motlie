//! repl_host_v1_2 scaffold: copied host-side filesystem server with admin REPL.
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
//! cargo run --example repl_host_v1_2 --features vsock -- --tag alice-home --dir ~/alice
//!
//! # Multi-mount (repeat --mount)
//! cargo run --example repl_host_v1_2 --features vsock -- \
//!     --socket /tmp/motlie-vfs.vsock_5000 \
//!     --mount alice-home=~/alice \
//!     --mount workspace=~/workspace
//!
//! # Multi-guest (repeat --guest and guest-qualified --mount)
//! cargo run --example repl_host_v1_2 --features vsock -- \
//!     --guest alice=/tmp/motlie-vfs-alice.vsock_5000 \
//!     --mount alice:alice-home=~/alice \
//!     --mount alice:alice-workspace=~/workspace \
//!     --guest bob=/tmp/motlie-vfs-bob.vsock_5000 \
//!     --mount bob:bob-home=~/bob \
//!     --mount bob:bob-workspace=~/workspace-bob
//!
//! # Empty admin mode, provision from REPL script
//! cat setup-multiguest.sh.vfs | cargo run --example repl_host_v1_2 --features vsock -- --empty
//!
//! # Preferred interactive setup-file flow (keeps rustyline on a real TTY)
//! cargo run --example repl_host_v1_2 --features vsock -- --empty --script setup-multiguest.sh.vfs --admin-net=tap --egress-net=vhost-user
//!
//! # Script then interactive
//! cat setup-alice.sh.vfs - | cargo run --example repl_host_v1_2 --features vsock -- --tag alice-home
//!
//! # Script only (server stays alive until signaled)
//! cat setup-alice.sh.vfs | cargo run --example repl_host_v1_2 --features vsock -- --tag alice-home
//!
//! # Agent-driven (write commands to stdin, server stays alive)
//! echo "layer creds 0" | cargo run --example repl_host_v1_2 --features vsock -- --tag alice-home
//! ```
//!
//! # Options
//!
//!   --empty                 start with no guest and provision from REPL
//!   --script <path>         execute a setup file before entering the REPL
//!   --admin-net <mode>      none | tap (default: tap)
//!   --egress-net <mode>     none | tap | vhost-user (default: vhost-user)
//!   --socket <path>         vsock socket path (single-guest mode)
//!   --guest <id=socket>     add a guest-scoped FsServer and listener
//!   --mount <tag=dir>       add a mount in single-guest mode
//!   --mount <id:tag=dir>    add a mount to one guest in multi-guest mode
//!   --tag <name>            mount tag (single-guest mode)
//!   --dir <path>            host backing directory (single-guest mode)

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::net::Ipv4Addr;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use bytes::Bytes;
use tempfile::TempDir;
use tokio::net::UnixListener;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use motlie_vnet::{VnetBackend, VnetConfig, VnetHandle};
use motlie_vfs::core::overlay::OverlayAttrs;
use motlie_vfs::core::server::FsServer;
use motlie_vfs::vsock::handler::VsockConnectionHandler;

fn build_git_sha() -> &'static str {
    option_env!("MOTLIE_VNET_BUILD_GIT_SHA").unwrap_or("unknown")
}

fn build_time_utc() -> &'static str {
    option_env!("MOTLIE_VNET_BUILD_TIME_UTC").unwrap_or("unknown")
}

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdminNet {
    None,
    Tap,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EgressNet {
    None,
    Tap,
    VhostUser,
}

impl AdminNet {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "tap" => Ok(Self::Tap),
            _ => anyhow::bail!("admin-net must be one of: none, tap"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tap => "tap",
        }
    }
}

impl EgressNet {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "tap" => Ok(Self::Tap),
            "vhost-user" => Ok(Self::VhostUser),
            _ => anyhow::bail!("egress-net must be one of: none, tap, vhost-user"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tap => "tap",
            Self::VhostUser => "vhost-user",
        }
    }
}

fn validate_network_modes(admin_net: AdminNet, egress_net: EgressNet) -> Result<()> {
    match (admin_net, egress_net) {
        (AdminNet::None, EgressNet::None)
        | (AdminNet::Tap, EgressNet::Tap)
        | (AdminNet::Tap, EgressNet::VhostUser) => Ok(()),
        _ => anyhow::bail!(
            "supported launch modes are --admin-net=none --egress-net=none, --admin-net=tap --egress-net=tap, and --admin-net=tap --egress-net=vhost-user"
        ),
    }
}

struct AdminState {
    guests: HashMap<String, GuestRuntime>,
    guest_order: Vec<String>,
    current_guest: String,
    multi_guest: bool,
    comment_stdout: bool,
    prompt_state: Arc<StdMutex<String>>,
    startup_scripts: Vec<PathBuf>,
    _retained_tempdirs: Vec<TempDir>,
    runtime: Handle,
    sockets_for_cleanup: Arc<StdMutex<Vec<String>>>,
    admin_net: AdminNet,
    egress_net: EgressNet,
    vnet_handles: HashMap<String, VnetHandle>,
    launch_pids: HashMap<String, u32>,
}

enum MountSpec {
    Single {
        tag: String,
        dir: PathBuf,
    },
    Guest {
        guest: String,
        tag: String,
        dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut socket_path = "/tmp/motlie-vfs.vsock_5000".to_string();
    let mut tag = "alice-home".to_string();
    let mut host_dir: Option<PathBuf> = None;
    let mut single_mounts: Vec<(String, PathBuf)> = Vec::new();
    let mut guest_configs: Vec<GuestConfig> = Vec::new();
    let mut empty_mode = false;
    let mut startup_scripts: Vec<PathBuf> = Vec::new();
    let mut retained_tempdirs = Vec::new();
    let mut admin_net = AdminNet::Tap;
    let mut egress_net = EgressNet::VhostUser;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--empty" => {
                empty_mode = true;
                i += 1;
            }
            "--socket" if i + 1 < args.len() => {
                socket_path = args[i + 1].clone();
                i += 2;
            }
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
                        let Some(config) = guest_configs.iter_mut().find(|cfg| cfg.name == guest)
                        else {
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
            "--script" if i + 1 < args.len() => {
                startup_scripts.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            arg if arg.starts_with("--script=") => {
                startup_scripts.push(PathBuf::from(&arg["--script=".len()..]));
                i += 1;
            }
            "--admin-net" if i + 1 < args.len() => {
                admin_net = AdminNet::parse(&args[i + 1])?;
                i += 2;
            }
            arg if arg.starts_with("--admin-net=") => {
                admin_net = AdminNet::parse(&arg["--admin-net=".len()..])?;
                i += 1;
            }
            "--egress-net" if i + 1 < args.len() => {
                egress_net = EgressNet::parse(&args[i + 1])?;
                i += 2;
            }
            arg if arg.starts_with("--egress-net=") => {
                egress_net = EgressNet::parse(&arg["--egress-net=".len()..])?;
                i += 1;
            }
            "--no-net" => {
                admin_net = AdminNet::None;
                egress_net = EgressNet::None;
                i += 1;
            }
            "--tag" if i + 1 < args.len() => {
                tag = args[i + 1].clone();
                i += 2;
            }
            "--dir" if i + 1 < args.len() => {
                host_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                host_dir = Some(PathBuf::from(other));
                i += 1;
            }
        }
    }

    validate_network_modes(admin_net, egress_net)?;

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
                anyhow::bail!(
                    "guest '{}' has no mounts; add at least one --mount {}:<tag>=<dir>",
                    config.name,
                    config.name
                );
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
            mounts: single_mounts
                .into_iter()
                .map(|(tag, host_path)| ConfiguredMount {
                    tag,
                    guest_path: None,
                    host_path,
                })
                .collect(),
            identity: None,
        });
        if let Some(tempdir) = tempdir {
            retained_tempdirs.push(tempdir);
        }
    }

    eprintln!("=== motlie-vfs repl_host_v1_2 ===");
    eprintln!("Build: {}", build_git_sha());
    eprintln!("Built At: {}", build_time_utc());
    eprintln!(
        "Network: admin={} egress={}",
        admin_net.as_str(),
        egress_net.as_str()
    );
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
                eprintln!(
                    "    {}: {} -> {}",
                    mount.tag,
                    guest_path,
                    mount.host_path.display()
                );
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
    let prompt_state = Arc::new(StdMutex::new("vfs> ".to_string()));

    for config in guest_configs {
        let mut builder = FsServer::builder().overlay(true).events(256);
        for mount in &config.mounts {
            let owner_override = config.identity.map(|identity| (identity.uid, identity.gid));
            seed_demo_host_mount(&config.name, mount)?;
            builder = builder.mount_as(&mount.tag, mount.host_path.clone(), false, owner_override);
        }
        let server = Arc::new(builder.build()?);
        guest_order.push(config.name.clone());
        admin_guests.insert(
            config.name.clone(),
            GuestRuntime {
                server: Arc::clone(&server),
                socket_path: config.socket_path.clone(),
                mounts: config.mounts.clone(),
                identity: config.identity,
            },
        );

        let _ = std::fs::remove_file(&config.socket_path);
        let listener = UnixListener::bind(&config.socket_path)?;
        push_cleanup_socket(&sockets_for_cleanup, config.socket_path.clone());
        spawn_guest_listener(
            &runtime,
            config.name.clone(),
            config.socket_path.clone(),
            listener,
            server,
            Arc::clone(&prompt_state),
        );
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
        prompt_state,
        startup_scripts,
        _retained_tempdirs: retained_tempdirs,
        runtime,
        sockets_for_cleanup: Arc::clone(&sockets_for_cleanup),
        admin_net,
        egress_net,
        vnet_handles: HashMap::new(),
        launch_pids: HashMap::new(),
    };

    // Spawn the admin input handler on a blocking thread
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    tokio::task::spawn_blocking(move || {
        run_input(admin);
        for socket_path in cleanup_sockets_snapshot(&sockets_for_cleanup) {
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
    if !admin.startup_scripts.is_empty() {
        let scripts = std::mem::take(&mut admin.startup_scripts);
        for script in scripts {
            if let Err(e) = run_script_file(&mut admin, &script) {
                eprintln!("script {} failed: {e}", script.display());
                return;
            }
        }
    }

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
                Err(e) => {
                    eprintln!("stdin read error: {e}");
                    break;
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
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
                eprintln!(
                    "--- entering interactive REPL (Ctrl-D to stop, server keeps running) ---"
                );
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

fn run_script_file(admin: &mut AdminState, path: &PathBuf) -> Result<()> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    eprintln!("--- reading commands from script {} ---", path.display());
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        eprintln!("vfs> {trimmed}");
        if dispatch_command(admin, trimmed) == ControlFlow::Quit {
            anyhow::bail!("quit command encountered in {}", path.display());
        }
    }
    Ok(())
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
        store_prompt(&admin.prompt_state, prompt.clone());
        let line = match rl.readline(&prompt) {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                eprintln!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("REPL error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
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
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .unwrap_or_else(|e| eprintln!("warning: failed to set signal handler: {e}"));

    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn store_prompt(prompt_state: &Arc<StdMutex<String>>, prompt: String) {
    if let Ok(mut current) = prompt_state.lock() {
        *current = prompt;
    }
}

fn redraw_prompt(prompt_state: &Arc<StdMutex<String>>) {
    let prompt = match prompt_state.lock() {
        Ok(current) => current.clone(),
        Err(_) => "vfs> ".to_string(),
    };
    let _ = writeln!(io::stderr());
    let _ = write!(io::stderr(), "{prompt}");
    let _ = io::stderr().flush();
}

fn push_cleanup_socket(sockets_for_cleanup: &Arc<StdMutex<Vec<String>>>, socket_path: String) {
    match sockets_for_cleanup.lock() {
        Ok(mut sockets) => sockets.push(socket_path),
        Err(_) => eprintln!("warning: cleanup socket list poisoned; skipping socket retention"),
    }
}

fn cleanup_sockets_snapshot(sockets_for_cleanup: &Arc<StdMutex<Vec<String>>>) -> Vec<String> {
    match sockets_for_cleanup.lock() {
        Ok(sockets) => sockets.clone(),
        Err(_) => {
            eprintln!("warning: cleanup socket list poisoned; skipping socket cleanup");
            Vec::new()
        }
    }
}

fn spawn_guest_listener(
    runtime: &Handle,
    guest_name: String,
    socket_path: String,
    listener: UnixListener,
    server: Arc<FsServer>,
    prompt_state: Arc<StdMutex<String>>,
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
                            redraw_prompt(&prompt_state);
                            continue;
                        }
                    };
                    if !server.has_mount(&tag) {
                        eprintln!("[{guest_name}] connection requested unknown tag: {tag}");
                        redraw_prompt(&prompt_state);
                        continue;
                    }
                    let handler = VsockConnectionHandler::new(Arc::clone(&server), &tag);
                    let guest_name_for_conn = guest_name.clone();
                    let tag_for_conn = tag.clone();
                    let prompt_state_for_conn = Arc::clone(&prompt_state);
                    tokio::spawn(async move {
                        match handler.serve(stream).await {
                            Ok(()) => eprintln!("[{guest_name_for_conn}] connection handler closed cleanly for tag={tag_for_conn}"),
                            Err(e) => eprintln!("[{guest_name_for_conn}] connection handler error for tag={tag_for_conn}: {e}"),
                        }
                        redraw_prompt(&prompt_state_for_conn);
                    });
                    eprintln!("[accepted guest connection guest={guest_name} tag={tag}]");
                    redraw_prompt(&prompt_state);
                }
                Err(e) => {
                    eprintln!("[{guest_name}] accept error: {e}");
                    redraw_prompt(&prompt_state);
                }
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

    let server = Arc::new(FsServer::builder().overlay(true).events(256).build()?);
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)?;
    push_cleanup_socket(&admin.sockets_for_cleanup, socket_path.to_string());
    spawn_guest_listener(
        &admin.runtime,
        guest_name.to_string(),
        socket_path.to_string(),
        listener,
        Arc::clone(&server),
        Arc::clone(&admin.prompt_state),
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
        .add_mount_as(
            &mount.tag,
            mount.host_path.clone(),
            false,
            runtime.identity.map(|identity| (identity.uid, identity.gid)),
        )?;
    seed_demo_host_mount(guest_name, &mount)?;
    runtime.mounts.push(mount);
    Ok(())
}

fn write_host_file_if_missing(path: &std::path::Path, content: &str, mode: u32) -> Result<()> {
    if !path.exists() {
        fs::write(path, content)?;
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    }
    Ok(())
}

fn seed_demo_host_mount(guest_name: &str, mount: &ConfiguredMount) -> Result<()> {
    let Some(guest_path) = &mount.guest_path else {
        return Ok(());
    };

    if guest_path != &format!("/home/{guest_name}") {
        return Ok(());
    }

    fs::create_dir_all(&mount.host_path)?;
    fs::create_dir_all(mount.host_path.join(".config"))?;
    fs::create_dir_all(mount.host_path.join(".ssh"))?;
    #[cfg(unix)]
    fs::set_permissions(mount.host_path.join(".ssh"), fs::Permissions::from_mode(0o700))?;

    let (authorized_key, ssh_config, env_content) = match guest_name {
        "alice" => (
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample alice@dev\n",
            "Host github.com\n  User git\n",
            "ALICE_API_KEY=demo-alice\n",
        ),
        "bob" => (
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample bob@dev\n",
            "Host gitlab.com\n  User git\n",
            "BOB_API_KEY=demo-bob\n",
        ),
        _ => return Ok(()),
    };

    write_host_file_if_missing(&mount.host_path.join(".ssh/authorized_keys"), authorized_key, 0o600)?;
    write_host_file_if_missing(&mount.host_path.join(".ssh/config"), ssh_config, 0o644)?;
    write_host_file_if_missing(&mount.host_path.join(".env"), env_content, 0o644)?;
    write_host_file_if_missing(&mount.host_path.join(".bashrc"), "# motlie v1.2 demo bashrc\n", 0o644)?;
    write_host_file_if_missing(
        &mount.host_path.join(".profile"),
        "if [ -f \"$HOME/.bashrc\" ]; then\n  . \"$HOME/.bashrc\"\nfi\n",
        0o644,
    )?;

    Ok(())
}

fn guest_api_socket_path(guest_name: &str) -> PathBuf {
    PathBuf::from(format!("/tmp/motlie-vfs-{guest_name}-api.sock"))
}

fn guest_vnet_socket_path(guest_name: &str) -> PathBuf {
    PathBuf::from(format!("/tmp/motlie-vnet-{guest_name}.sock"))
}

fn guest_egress_mac(_guest_name: &str) -> [u8; 6] {
    [0x12, 0x34, 0x56, 0x78, 0x90, 0xab]
}

fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn render_mounts_yaml(runtime: &GuestRuntime) -> Result<String> {
    let mut out = String::from("mounts:\n");
    for mount in &runtime.mounts {
        let Some(guest_path) = &mount.guest_path else {
            anyhow::bail!(
                "mount '{}' is missing guest_path; cannot render mounts.yaml",
                mount.tag
            );
        };
        writeln!(&mut out, "  - tag: {}", mount.tag)?;
        writeln!(&mut out, "    guest_path: {}", guest_path)?;
        writeln!(&mut out, "    read_only: false")?;
    }
    Ok(out)
}

fn render_cloud_init(guest_name: &str, runtime: &GuestRuntime) -> Result<String> {
    let _identity = runtime.identity.ok_or_else(|| {
        anyhow::anyhow!("guest '{guest_name}' is missing uid/gid; provision with explicit uid/gid")
    })?;
    let mut out = String::new();
    out.push_str("#cloud-config\n");
    Ok(out)
}

fn render_launch_script(
    guest_name: &str,
    runtime: &GuestRuntime,
    admin_net: AdminNet,
    egress_net: EgressNet,
    vnet_socket: Option<&PathBuf>,
) -> Result<String> {
    let cloud_init = render_cloud_init(guest_name, runtime)?;
    if guest_name != "alice" && guest_name != "bob" {
        anyhow::bail!("launch prototype currently targets v1.2 demo guests alice/bob because launch-ch.sh still carries guest-specific runtime defaults");
    }
    let base_dir = format!("{}/examples/v1.2", env!("CARGO_MANIFEST_DIR"));
    let mut out = String::new();
    out.push_str("#!/usr/bin/env bash\n");
    out.push_str("set -euo pipefail\n\n");
    out.push_str("# Generated by repl_host from the provisioned guest state.\n");
    out.push_str("# Rebuild the shared v1.2 base image with the current build-guest.sh so the\n");
    out.push_str(
        "# guest includes cloud-init and consumes the seeded NoCloud directory at boot.\n\n",
    );
    writeln!(&mut out, "GUEST_ID={}", shell_single_quote(guest_name))?;
    writeln!(&mut out, "BASE_DIR=\"${{BASE_DIR:-{}}}\"", base_dir)?;
    writeln!(
        &mut out,
        "SEED_DIR=\"${{SEED_DIR:-/tmp/motlie-vfs-cloud-init-${{GUEST_ID}}}}\""
    )?;
    writeln!(
        &mut out,
        "ADMIN_NET={}",
        shell_single_quote(admin_net.as_str())
    )?;
    writeln!(
        &mut out,
        "EGRESS_NET={}",
        shell_single_quote(egress_net.as_str())
    )?;
    if let Some(socket) = vnet_socket {
        writeln!(
            &mut out,
            "VNET_SOCKET={}",
            shell_single_quote(socket.to_string_lossy().as_ref())
        )?;
    }
    out.push_str("INSTANCE_ID=\"${INSTANCE_ID:-${GUEST_ID}}\"\n");
    out.push_str("LOCAL_HOSTNAME=\"${LOCAL_HOSTNAME:-motlie-${GUEST_ID}}\"\n");
    out.push_str("mkdir -p \"$SEED_DIR\"\n");
    out.push_str("cat > \"$SEED_DIR/meta-data\" <<EOF\n");
    out.push_str("instance-id: ${INSTANCE_ID}\n");
    out.push_str("local-hostname: ${LOCAL_HOSTNAME}\n");
    out.push_str("EOF\n\n");
    out.push_str("cat > \"$SEED_DIR/mounts.yaml\" <<'EOF'\n");
    out.push_str(&render_mounts_yaml(runtime)?);
    out.push_str("EOF\n\n");
    out.push_str("cat > \"$SEED_DIR/user-data\" <<'EOF'\n");
    out.push_str(&cloud_init);
    if !cloud_init.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("EOF\n\n");
    out.push_str("echo \"Generated cloud-init assets in $SEED_DIR\"\n");
    out.push_str("echo \"Launching guest ${GUEST_ID} with seeded NoCloud dir ${SEED_DIR}\"\n");
    if egress_net == EgressNet::VhostUser {
        out.push_str("echo \"Using motlie-vnet egress socket ${VNET_SOCKET}\"\n");
    }
    out.push_str("LAUNCH_ARGS=(--guest \"$GUEST_ID\" --cloud-init-dir \"$SEED_DIR\" --admin-net \"$ADMIN_NET\" --egress-net \"$EGRESS_NET\")\n");
    if egress_net == EgressNet::VhostUser {
        out.push_str("LAUNCH_ARGS+=(--vnet-socket \"$VNET_SOCKET\")\n");
    }
    out.push_str("\"$BASE_DIR/launch-ch.sh\" \"${LAUNCH_ARGS[@]}\" \"$@\"\n");
    Ok(out)
}

struct LaunchExecution {
    pid: u32,
    script_path: PathBuf,
    launch_log_path: PathBuf,
    serial_log_path: PathBuf,
}

fn execute_launch_script(guest_name: &str, script: &str) -> Result<LaunchExecution> {
    let log_dir = PathBuf::from(format!("/tmp/motlie-vfs-launch/{guest_name}"));
    std::fs::create_dir_all(&log_dir)?;

    let script_path = log_dir.join("launch.sh");
    let launch_log_path = log_dir.join("launch.log");
    let serial_log_path = log_dir.join("serial.log");

    std::fs::write(&script_path, script)?;

    let launch_log = File::create(&launch_log_path)?;
    let launch_log_err = launch_log.try_clone()?;

    let child = Command::new("/bin/bash")
        .arg(&script_path)
        .env(
            "CH_SERIAL_BACKEND",
            format!("file={}", serial_log_path.display()),
        )
        .env("CH_CONSOLE_BACKEND", "off")
        .stdin(Stdio::null())
        .stdout(Stdio::from(launch_log))
        .stderr(Stdio::from(launch_log_err))
        .spawn()?;

    Ok(LaunchExecution {
        pid: child.id(),
        script_path,
        launch_log_path,
        serial_log_path,
    })
}

fn process_exists(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

fn find_cloud_hypervisor_pid(api_socket: &std::path::Path) -> Option<u32> {
    let api_socket = api_socket.to_string_lossy();
    let proc_entries = fs::read_dir("/proc").ok()?;

    for entry in proc_entries.flatten() {
        let file_name = entry.file_name();
        let Some(pid) = file_name.to_string_lossy().parse::<u32>().ok() else {
            continue;
        };
        let cmdline_path = entry.path().join("cmdline");
        let Some(cmdline) = fs::read(&cmdline_path).ok() else {
            continue;
        };
        if cmdline.is_empty() {
            continue;
        }

        let argv: Vec<String> = cmdline
            .split(|b| *b == 0)
            .filter(|part| !part.is_empty())
            .map(|part| String::from_utf8_lossy(part).into_owned())
            .collect();

        if argv.is_empty() || !argv[0].contains("cloud-hypervisor") {
            continue;
        }

        let mut i = 0;
        while i < argv.len() {
            if argv[i] == "--api-socket" && i + 1 < argv.len() && argv[i + 1] == api_socket {
                return Some(pid);
            }
            i += 1;
        }
    }

    None
}

fn signal_pid(pid: u32, signal: &str) -> Result<()> {
    let status = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("kill -{signal} {pid} exited with status {status}");
    }
}

fn wait_for_guest_exit(guest_name: &str, pid: Option<u32>, timeout: Duration) -> Result<()> {
    let api_socket = guest_api_socket_path(guest_name);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let pid_gone = pid.map(|p| !process_exists(p));
        if pid_gone == Some(true) {
            if api_socket.exists() {
                let _ = std::fs::remove_file(&api_socket);
            }
            return Ok(());
        }
        let socket_gone = !api_socket.exists();
        if pid.is_none() && socket_gone {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }
    anyhow::bail!("guest did not exit within {}", timeout.as_secs());
}

struct ShutdownOutcome {
    pid: Option<u32>,
    api_failure: Option<String>,
    forced: Option<&'static str>,
}

fn shutdown_guest(guest_name: &str, pid: Option<u32>) -> Result<ShutdownOutcome> {
    let api_socket = guest_api_socket_path(guest_name);
    if !api_socket.exists() {
        anyhow::bail!("guest API socket not found at {}", api_socket.display());
    }
    let pid = find_cloud_hypervisor_pid(&api_socket).or(pid);

    let output = Command::new("curl")
        .arg("--silent")
        .arg("--show-error")
        .arg("--max-time")
        .arg("5")
        .arg("--write-out")
        .arg("\n%{http_code}")
        .arg("--unix-socket")
        .arg(&api_socket)
        .arg("-X")
        .arg("PUT")
        .arg("http://localhost/api/v1/vm.shutdown")
        .stdin(Stdio::null())
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let (body, http_code) = match stdout.rsplit_once('\n') {
        Some((body, code)) => (body.trim().to_string(), code.trim().to_string()),
        None => (stdout.trim().to_string(), String::new()),
    };
    let api_request_ok = output.status.success();
    let api_http_ok = matches!(http_code.as_str(), "200" | "202" | "204");
    let api_failure = if api_request_ok && api_http_ok {
        None
    } else if !api_request_ok {
        Some(if stderr.is_empty() {
            format!("curl exited with status {}", output.status)
        } else {
            format!("curl exited with status {}: {}", output.status, stderr)
        })
    } else if body.is_empty() {
        Some(format!("shutdown API returned HTTP {}", http_code))
    } else {
        Some(format!("shutdown API returned HTTP {}: {}", http_code, body))
    };

    if api_failure.is_none() && wait_for_guest_exit(guest_name, pid, Duration::from_secs(15)).is_ok() {
        return Ok(ShutdownOutcome {
            pid,
            api_failure,
            forced: None,
        });
    }

    if let Some(pid) = pid {
        if process_exists(pid) {
            signal_pid(pid, "TERM")?;
            if wait_for_guest_exit(guest_name, Some(pid), Duration::from_secs(5)).is_ok() {
                return Ok(ShutdownOutcome {
                    pid: Some(pid),
                    api_failure,
                    forced: Some("TERM"),
                });
            }
            if process_exists(pid) {
                signal_pid(pid, "KILL")?;
                wait_for_guest_exit(guest_name, Some(pid), Duration::from_secs(2))?;
                return Ok(ShutdownOutcome {
                    pid: Some(pid),
                    api_failure,
                    forced: Some("KILL"),
                });
            }
        }
    }

    match wait_for_guest_exit(guest_name, pid, Duration::from_secs(2)) {
        Ok(()) => Ok(ShutdownOutcome {
            pid,
            api_failure,
            forced: None,
        }),
        Err(wait_err) => {
            if let Some(api_failure) = api_failure {
                anyhow::bail!("{api_failure}; {wait_err}");
            } else {
                Err(wait_err)
            }
        }
    }
}

fn ensure_vnet_backend(admin: &mut AdminState, guest_name: &str) -> Result<Option<PathBuf>> {
    if admin.egress_net != EgressNet::VhostUser {
        return Ok(None);
    }

    let socket_path = guest_vnet_socket_path(guest_name);

    if let Some(handle) = admin.vnet_handles.get_mut(guest_name) {
        if handle.is_alive() {
            return Ok(Some(socket_path));
        }

        let _ = handle.shutdown();
        admin.vnet_handles.remove(guest_name);
        emit_status(
            admin,
            format!(
                "warn: restarted stale vnet backend for {guest_name} socket={}",
                socket_path.display()
            ),
        );
    }

    let config = VnetConfig::builder()
        .socket_path(&socket_path)
        .guest_ipv4(Ipv4Addr::new(10, 0, 2, 15))
        .host_ipv4(Ipv4Addr::new(10, 0, 2, 2))
        .netmask(Ipv4Addr::new(255, 255, 255, 0))
        .dns_ipv4(Ipv4Addr::new(10, 0, 2, 3))
        .mac(guest_egress_mac(guest_name))
        .build()?;

    let handle = VnetBackend::new(config).start()?;
    admin.vnet_handles.insert(guest_name.to_string(), handle);
    emit_status(
        admin,
        format!(
            "ok: vnet {guest_name} socket={} mode={}",
            socket_path.display(),
            admin.egress_net.as_str()
        ),
    );
    Ok(Some(socket_path))
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
            out("  Generate a guest helper script and execute it asynchronously via /bin/bash.");
            out("  The helper writes guest-specific cloud-init user-data and meta-data");
            out("  generated from the provisioned uid/gid and mount topology.");
            out("  launch-ch.sh then seeds those files into /var/lib/cloud/seed/nocloud/.");
            out("  In --egress-net=vhost-user mode, repl_host_v1_2 starts one motlie-vnet");
            out("  backend per guest before launching and reuses it across later launches.");
            out("  Logs land under /tmp/motlie-vfs-launch/<guest>/.");
            out("launch -script <guest>");
            out("  Render the helper shell script to stdout without executing it.");
            out("  In vhost-user mode, the helper assumes a motlie-vnet backend is");
            out("  already listening on the derived guest socket.");
        }
        Some("shutdown") => {
            out("shutdown <guest>");
            out("  Request VM shutdown through the guest's Cloud Hypervisor API socket.");
            out("  This shells out to curl --unix-socket /tmp/motlie-vfs-<guest>-api.sock ...");
        }
        Some("use") => {
            out("use <guest>");
            out("  Set the default target guest for subsequent admin commands.");
        }
        Some("guests") => {
            out("guests");
            out("  List provisioned guests, sockets, and configured mount counts.");
        }
        Some("build") => {
            out("build");
            out("  Print the git commit SHA and build timestamp baked into this repl_host_v1_2 binary at compile time.");
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
                out("  launch <guest>                                  — generate and start a guest launch helper asynchronously");
                out("  launch -script <guest>                          — print the guest launch helper script");
                out("  shutdown <guest>                                — request guest shutdown via CH API socket");
                out("  build                                           — print the build commit SHA");
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
            out("  Preferred scripted startup: repl_host_v1_2 --script setup.vfs → script then REPL");
            out("  Pipe + TTY:   cat script.vfs - | repl_host_v1_2 → limited terminal semantics");
            out("  Pure pipe:    cat script.vfs | repl_host_v1_2 → script then serve until signaled");
        }
    }
}

fn dispatch_command(admin: &mut AdminState, line: &str) -> ControlFlow {
    let mut parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return ControlFlow::Continue;
    }

    match parts[0] {
        "guests" => {
            for guest in &admin.guest_order {
                let marker = if guest == &admin.current_guest {
                    "*"
                } else {
                    " "
                };
                if let Some(runtime) = admin.guests.get(guest) {
                    let egress_state = if admin.vnet_handles.contains_key(guest) {
                        " egress=vhost-user(up)"
                    } else {
                        ""
                    };
                    if let Some(identity) = runtime.identity {
                        emit_status(
                            admin,
                            format!(
                                "{marker} {guest} socket={} uid={} gid={} mounts={}{}",
                                runtime.socket_path,
                                identity.uid,
                                identity.gid,
                                runtime.mounts.len(),
                                egress_state
                            ),
                        );
                    } else {
                        emit_status(
                            admin,
                            format!(
                                "{marker} {guest} socket={} mounts={}{}",
                                runtime.socket_path,
                                runtime.mounts.len(),
                                egress_state
                            ),
                        );
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
                if admin.multi_guest {
                    store_prompt(
                        &admin.prompt_state,
                        format!("vfs[{}]> ", admin.current_guest),
                    );
                } else {
                    store_prompt(&admin.prompt_state, "vfs> ".to_string());
                }
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
                Ok(()) => emit_status(
                    admin,
                    format!(
                        "ok: provision {} {} uid={} gid={}",
                        parts[1], parts[2], uid, gid
                    ),
                ),
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
                match parse_repl_mount_target(spec).and_then(|mount| {
                    let mount_desc = if let Some(guest_path) = &mount.guest_path {
                        format!(
                            "{}:{} -> {}",
                            mount.tag,
                            guest_path,
                            mount.host_path.display()
                        )
                    } else {
                        format!("{} -> {}", mount.tag, mount.host_path.display())
                    };
                    add_guest_mount(admin, guest_name, mount)?;
                    Ok(mount_desc)
                }) {
                    Ok(mount_desc) => {
                        emit_status(admin, format!("ok: mount {guest_name} {mount_desc}"))
                    }
                    Err(e) => emit_status(admin, format!("error: {e}")),
                }
            }
            return ControlFlow::Continue;
        }
        "mount" => {
            emit_status(
                admin,
                "error: mount <guest> <tag>=<guest_path>,<host_path> [more...]",
            );
            return ControlFlow::Continue;
        }
        "launch" if parts.len() == 2 || (parts.len() == 3 && parts[1] == "-script") => {
            let render_only = parts.len() == 3;
            let guest_name = if render_only { parts[2] } else { parts[1] };
            let vnet_socket = if render_only && admin.egress_net == EgressNet::VhostUser {
                Some(guest_vnet_socket_path(guest_name))
            } else {
                match ensure_vnet_backend(admin, guest_name) {
                    Ok(socket) => socket,
                    Err(e) => {
                        emit_status(admin, format!("error: {e}"));
                        return ControlFlow::Continue;
                    }
                }
            };
            match admin
                .guests
                .get(guest_name)
                .ok_or_else(|| anyhow::anyhow!("unknown guest '{guest_name}'"))
                .and_then(|runtime| {
                    render_launch_script(
                        guest_name,
                        runtime,
                        admin.admin_net,
                        admin.egress_net,
                        vnet_socket.as_ref(),
                    )
                })
            {
                Ok(script) => {
                    if render_only {
                        emit_raw(script);
                    } else {
                        match execute_launch_script(guest_name, &script) {
                            Ok(exec) => {
                                admin.launch_pids.insert(guest_name.to_string(), exec.pid);
                                emit_status(
                                    admin,
                                    format!(
                                        "ok: launch {guest_name} pid={} script={} log={} serial={}",
                                        exec.pid,
                                        exec.script_path.display(),
                                        exec.launch_log_path.display(),
                                        exec.serial_log_path.display(),
                                    ),
                                );
                            }
                            Err(e) => emit_status(admin, format!("error: {e}")),
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
        "shutdown" if parts.len() == 2 => {
            let guest_name = parts[1];
            let launch_pid = admin.launch_pids.get(guest_name).copied();
            match shutdown_guest(guest_name, launch_pid) {
                Ok(outcome) => {
                    admin.launch_pids.remove(guest_name);
                    let mut detail = String::new();
                    if let Some(pid) = outcome.pid {
                        detail.push_str(&format!(" pid={pid}"));
                    }
                    if let Some(forced) = outcome.forced {
                        detail.push_str(&format!(" forced={forced}"));
                    }
                    if let Some(api_failure) = outcome.api_failure {
                        detail.push_str(&format!(" api_fallback={}", api_failure));
                    }
                    emit_status(
                        admin,
                        format!(
                            "ok: shutdown {guest_name} api_socket={}{}",
                            guest_api_socket_path(guest_name).display(),
                            detail
                        ),
                    )
                }
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
            return ControlFlow::Continue;
        }
        "shutdown" => {
            emit_status(admin, "error: shutdown <guest>");
            return ControlFlow::Continue;
        }
        "build" if parts.len() == 1 => {
            emit_status(
                admin,
                format!("build: sha={} built_at={}", build_git_sha(), build_time_utc()),
            );
            return ControlFlow::Continue;
        }
        "help" => {
            print_help(
                parts.get(1).copied(),
                admin.multi_guest,
                admin.comment_stdout,
            );
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
            emit_status(
                admin,
                "error: no guest selected; provision/use a guest or prefix commands with <guest>",
            );
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
        None => {
            emit_status(admin, "error: overlay not enabled");
            return ControlFlow::Continue;
        }
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
        "rmlayer" if parts.len() >= 2 => match overlay.remove_layer(parts[1]) {
            Ok(()) => emit_status(admin, format!("ok: rmlayer {}", parts[1])),
            Err(e) => emit_status(admin, format!("error: {e}")),
        },
        "layers" => {
            let layers = overlay.layers();
            if layers.is_empty() {
                emit_status(admin, "(no layers)");
            }
            for l in &layers {
                emit_status(
                    admin,
                    format!(
                        "  {} priority={} entries={}",
                        l.name, l.priority, l.entry_count
                    ),
                );
            }
        }

        // --- Content injection ---
        "put" if parts.len() >= 5 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let content = parts[4..].join(" ");
            match overlay.put(layer, tag, path, Bytes::from(content.clone())) {
                Ok(()) => emit_status(
                    admin,
                    format!("ok: put {layer} {tag} {path} ({} bytes)", content.len()),
                ),
                Err(e) => emit_status(admin, format!("error: {e}")),
            }
        }
        "putattr" if parts.len() >= 8 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let uid = match parts[4].parse::<u32>() {
                Ok(v) => v,
                Err(_) => {
                    emit_status(admin, "error: uid must be a number");
                    return ControlFlow::Continue;
                }
            };
            let gid = match parts[5].parse::<u32>() {
                Ok(v) => v,
                Err(_) => {
                    emit_status(admin, "error: gid must be a number");
                    return ControlFlow::Continue;
                }
            };
            let mode = match u32::from_str_radix(parts[6], 8) {
                Ok(v) => v,
                Err(_) => {
                    emit_status(admin, "error: mode must be octal");
                    return ControlFlow::Continue;
                }
            };
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
            } else {
                0o755
            };
            let (uid, gid) = runtime
                .identity
                .map(|identity| (identity.uid, identity.gid))
                .unwrap_or((0, 0));
            let attrs = OverlayAttrs { mode, uid, gid };
            match overlay.create_dir(layer, tag, path, attrs) {
                Ok(()) => emit_status(
                    admin,
                    format!("ok: mkdir {layer} {tag} {path} mode={mode:o} uid={uid} gid={gid}"),
                ),
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
                Some(data) => match std::str::from_utf8(&data) {
                    Ok(s) => emit_raw(format!("{s}\n")),
                    Err(_) => emit_status(admin, format!("({} bytes, binary)", data.len())),
                },
                None => emit_status(admin, "(not found)"),
            }
        }
        "ls" if parts.len() >= 2 => {
            let tag = parts[1];
            let entries = overlay.list_effective(tag);
            if entries.is_empty() {
                emit_status(admin, format!("(no overlay entries for tag '{tag}')"));
            }
            for entry in &entries {
                emit_status(
                    admin,
                    format!(
                        "  {:?} {} uid={} gid={} mode={:o}",
                        entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                    ),
                );
            }
            emit_status(admin, format!("({} entries)", entries.len()));
        }
        "lslayer" if parts.len() >= 3 => {
            let (layer, tag) = (parts[1], parts[2]);
            let entries = overlay.list_layer(layer, tag);
            if entries.is_empty() {
                emit_status(
                    admin,
                    format!("(no entries in layer '{layer}' for tag '{tag}')"),
                );
            }
            for entry in &entries {
                emit_status(
                    admin,
                    format!(
                        "  {:?} {} uid={} gid={} mode={:o}",
                        entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                    ),
                );
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
                let mut winner: std::collections::HashMap<String, String> =
                    std::collections::HashMap::new();
                for e in &effective {
                    winner.insert(e.path.clone(), e.layer.clone());
                }

                emit_status(admin, format!("tag: {tag}"));
                emit_status(admin, "");
                for l in &layers {
                    let mut entries = overlay.list_layer(&l.name, tag);
                    entries.sort_by(|a, b| a.path.cmp(&b.path));
                    if entries.is_empty() {
                        continue;
                    }
                    emit_status(
                        admin,
                        format!("  layer: {} (priority={})", l.name, l.priority),
                    );
                    for entry in &entries {
                        let eff = if winner
                            .get(&entry.path)
                            .map(|w| w == &l.name)
                            .unwrap_or(false)
                        {
                            "*"
                        } else {
                            " " // shadowed by higher-priority layer
                        };
                        emit_status(
                            admin,
                            format!(
                                "   {eff} {:?} {} uid={} gid={} mode={:o}",
                                entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                            ),
                        );
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
                    let mut winner: std::collections::HashMap<String, String> =
                        std::collections::HashMap::new();
                    for e in &effective {
                        winner.insert(e.path.clone(), e.layer.clone());
                    }
                    for l in &layers {
                        let mut entries = overlay.list_layer(&l.name, tag);
                        entries.sort_by(|a, b| a.path.cmp(&b.path));
                        if entries.is_empty() {
                            continue;
                        }
                        emit_status(
                            admin,
                            format!("  layer: {} (priority={})", l.name, l.priority),
                        );
                        for entry in &entries {
                            let eff = if winner
                                .get(&entry.path)
                                .map(|w| w == &l.name)
                                .unwrap_or(false)
                            {
                                "*"
                            } else {
                                " "
                            };
                            emit_status(
                                admin,
                                format!(
                                    "   {eff} {:?} {} uid={} gid={} mode={:o}",
                                    entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                                ),
                            );
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
