//! repl_host: v1 host-side filesystem server with admin REPL.
//!
//! Starts one FsServer per guest VM with MemOverlay, listens on a
//! parameterized Unix socket (the vsock host-side path), and serves
//! filesystem operations. Exposes every MemOverlay API operation
//! through a command interface.
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
//!   --socket <path>   vsock socket path (default: /tmp/motlie-vfs.vsock_5000)
//!   --tag <name>      mount tag (default: alice-home)
//!   --dir <path>      host backing directory (default: temp dir with sample data)

use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use tokio::net::UnixListener;
use tokio::sync::oneshot;

use motlie_vfs::core::overlay::OverlayAttrs;
use motlie_vfs::core::server::FsServer;
use motlie_vfs::vsock::handler::VsockConnectionHandler;

#[tokio::main]
async fn main() -> Result<()> {
    let mut socket_path = "/tmp/motlie-vfs.vsock_5000".to_string();
    let mut tag = "alice-home".to_string();
    let mut host_dir: Option<PathBuf> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" if i + 1 < args.len() => {
                socket_path = args[i + 1].clone();
                i += 2;
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

    let _tempdir;
    let host_path = match host_dir {
        Some(p) => {
            std::fs::create_dir_all(&p)?;
            p
        }
        None => {
            _tempdir = tempfile::tempdir()?;
            let p = _tempdir.path().to_path_buf();
            std::fs::create_dir_all(p.join("projects"))?;
            std::fs::write(p.join("projects/README.md"), b"hello from host disk\n")?;
            std::fs::write(p.join(".bashrc"), b"# bashrc\n")?;
            eprintln!("Created temp host dir: {}", p.display());
            p
        }
    };

    eprintln!("=== motlie-vfs repl_host ===");
    eprintln!("Host dir: {}", host_path.display());
    eprintln!("Tag: {tag}");
    eprintln!("Socket: {socket_path}");
    eprintln!("");

    let server = Arc::new(
        FsServer::builder()
            .mount(&tag, host_path, false)
            .overlay(true)
            .events(256)
            .build()?,
    );

    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)?;
    eprintln!("Listening on {socket_path} (guest filesystem connections)");
    eprintln!("Type 'help' for commands.");
    eprintln!("");

    let server_for_accept = Arc::clone(&server);
    let tag_for_accept = tag.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let handler = VsockConnectionHandler::new(
                        Arc::clone(&server_for_accept),
                        &tag_for_accept,
                    );
                    tokio::spawn(async move {
                        if let Err(e) = handler.serve(stream).await {
                            eprintln!("Connection handler error: {e}");
                        }
                    });
                    eprintln!("[accepted guest connection]");
                }
                Err(e) => eprintln!("Accept error: {e}"),
            }
        }
    });

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_for_input = Arc::clone(&server);
    let socket_for_cleanup = socket_path.clone();

    tokio::task::spawn_blocking(move || {
        run_input(server_for_input);
        let _ = std::fs::remove_file(&socket_for_cleanup);
        let _ = shutdown_tx.send(());
    });

    let _ = shutdown_rx.await;
    Ok(())
}

fn run_input(server: Arc<FsServer>) {
    let stdin_is_tty = atty::is(atty::Stream::Stdin);

    if stdin_is_tty {
        run_interactive_repl(&server);
    } else {
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
            if dispatch_command(&server, trimmed) == ControlFlow::Quit {
                eprintln!("shutting down (quit command)");
                return;
            }
        }
        eprintln!("--- stdin EOF ---");

        #[cfg(unix)]
        {
            if let Ok(_tty) = std::fs::File::open("/dev/tty") {
                eprintln!(
                    "--- entering interactive REPL (Ctrl-D to stop, server keeps running) ---"
                );
                eprintln!("");
                run_interactive_repl(&server);
                return;
            }
        }

        eprintln!("--- no TTY available, server running until signaled ---");
        eprintln!("--- send SIGTERM or SIGINT to stop ---");
        wait_for_signal();
        eprintln!("shutting down (signal received)");
    }
}

fn run_interactive_repl(server: &FsServer) {
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to create REPL: {e}");
            return;
        }
    };

    loop {
        let line = match rl.readline("vfs> ") {
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

        if dispatch_command(server, trimmed) == ControlFlow::Quit {
            break;
        }
    }

    eprintln!("shutting down");
}

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

#[derive(PartialEq)]
enum ControlFlow {
    Continue,
    Quit,
}

fn dispatch_command(server: &FsServer, line: &str) -> ControlFlow {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return ControlFlow::Continue;
    }

    let overlay = match server.overlay() {
        Some(o) => o,
        None => {
            println!("error: overlay not enabled");
            return ControlFlow::Continue;
        }
    };

    match parts[0] {
        "layer" if parts.len() >= 3 => {
            let name = parts[1];
            match parts[2].parse::<u32>() {
                Ok(priority) => match overlay.put_layer(name, priority) {
                    Ok(()) => println!("ok: layer {name} priority={priority}"),
                    Err(e) => println!("error: {e}"),
                },
                Err(_) => println!("error: priority must be a number"),
            }
        }
        "rmlayer" if parts.len() >= 2 => match overlay.remove_layer(parts[1]) {
            Ok(()) => println!("ok: rmlayer {}", parts[1]),
            Err(e) => println!("error: {e}"),
        },
        "layers" => {
            let layers = overlay.layers();
            if layers.is_empty() {
                println!("(no layers)");
            }
            for l in &layers {
                println!(
                    "  {} priority={} entries={}",
                    l.name, l.priority, l.entry_count
                );
            }
        }
        "put" if parts.len() >= 5 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let content = parts[4..].join(" ");
            match overlay.put(layer, tag, path, Bytes::from(content.clone())) {
                Ok(()) => println!("ok: put {layer} {tag} {path} ({} bytes)", content.len()),
                Err(e) => println!("error: {e}"),
            }
        }
        "putattr" if parts.len() >= 8 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let uid = match parts[4].parse::<u32>() {
                Ok(v) => v,
                Err(_) => {
                    println!("error: uid must be a number");
                    return ControlFlow::Continue;
                }
            };
            let gid = match parts[5].parse::<u32>() {
                Ok(v) => v,
                Err(_) => {
                    println!("error: gid must be a number");
                    return ControlFlow::Continue;
                }
            };
            let mode = match u32::from_str_radix(parts[6], 8) {
                Ok(v) => v,
                Err(_) => {
                    println!("error: mode must be octal");
                    return ControlFlow::Continue;
                }
            };
            let content = parts[7..].join(" ");
            let attrs = OverlayAttrs { mode, uid, gid };
            match overlay.put_with_attrs(layer, tag, path, attrs, Bytes::from(content.clone())) {
                Ok(()) => println!(
                    "ok: putattr {layer} {tag} {path} uid={uid} gid={gid} mode={mode:o} ({} bytes)",
                    content.len()
                ),
                Err(e) => println!("error: {e}"),
            }
        }
        "mkdir" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let mode = if parts.len() >= 5 {
                u32::from_str_radix(parts[4], 8).unwrap_or(0o755)
            } else {
                0o755
            };
            let attrs = OverlayAttrs {
                mode,
                uid: 0,
                gid: 0,
            };
            match overlay.create_dir(layer, tag, path, attrs) {
                Ok(()) => println!("ok: mkdir {layer} {tag} {path} mode={mode:o}"),
                Err(e) => println!("error: {e}"),
            }
        }
        "whiteout" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.whiteout(layer, tag, path) {
                Ok(()) => println!("ok: whiteout {layer} {tag} {path}"),
                Err(e) => println!("error: {e}"),
            }
        }
        "rm" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.remove(layer, tag, path) {
                Ok(()) => println!("ok: rm {layer} {tag} {path}"),
                Err(e) => println!("error: {e}"),
            }
        }
        "get" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.get(layer, tag, path) {
                Some(data) => match std::str::from_utf8(&data) {
                    Ok(s) => println!("{s}"),
                    Err(_) => println!("({} bytes, binary)", data.len()),
                },
                None => println!("(not found)"),
            }
        }
        "ls" if parts.len() >= 2 => {
            let tag = parts[1];
            let entries = overlay.list_effective(tag);
            if entries.is_empty() {
                println!("(no overlay entries for tag '{tag}')");
            }
            for entry in &entries {
                println!(
                    "  {:?} {} uid={} gid={} mode={:o}",
                    entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                );
            }
            println!("({} entries)", entries.len());
        }
        "lslayer" if parts.len() >= 3 => {
            let (layer, tag) = (parts[1], parts[2]);
            let entries = overlay.list_layer(layer, tag);
            if entries.is_empty() {
                println!("(no entries in layer '{layer}' for tag '{tag}')");
            }
            for entry in &entries {
                println!(
                    "  {:?} {} uid={} gid={} mode={:o}",
                    entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                );
            }
            println!("({} entries)", entries.len());
        }
        "tree" if parts.len() >= 2 => {
            let tag = parts[1];
            let layers = overlay.layers();
            if layers.is_empty() {
                println!("(no layers)");
            } else {
                let effective = overlay.list_effective(tag);
                let mut winner: std::collections::HashMap<String, String> =
                    std::collections::HashMap::new();
                for e in &effective {
                    winner.insert(e.path.clone(), e.layer.clone());
                }

                println!("tag: {tag}");
                println!();
                for l in &layers {
                    let mut entries = overlay.list_layer(&l.name, tag);
                    entries.sort_by(|a, b| a.path.cmp(&b.path));
                    if entries.is_empty() {
                        continue;
                    }
                    println!("  layer: {} (priority={})", l.name, l.priority);
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
                        println!(
                            "   {eff} {:?} {} uid={} gid={} mode={:o}",
                            entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                        );
                    }
                    println!();
                }
                println!("  (* = effective winner)");
            }
        }
        "tree" => {
            let layers = overlay.layers();
            let tags = overlay.tags();
            if layers.is_empty() {
                println!("(no layers)");
            } else if tags.is_empty() {
                println!("(no entries)");
            } else {
                for tag in &tags {
                    println!("tag: {tag}");
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
                        println!("  layer: {} (priority={})", l.name, l.priority);
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
                            println!(
                                "   {eff} {:?} {} uid={} gid={} mode={:o}",
                                entry.kind, entry.path, entry.uid, entry.gid, entry.mode
                            );
                        }
                    }
                    println!();
                }
                println!("(* = effective winner)");
            }
        }
        "help" => {
            println!("Layer management:");
            println!("  layer <name> <priority>                         — create/update layer");
            println!(
                "  rmlayer <name>                                  — remove layer and all entries"
            );
            println!("  layers                                          — list all layers");
            println!("");
            println!("Content injection:");
            println!(
                "  put <layer> <tag> <path> <content>              — inject file (default attrs)"
            );
            println!("  putattr <layer> <tag> <path> <uid> <gid> <mode> <content>");
            println!("                                                  — inject file with explicit attrs");
            println!(
                "  mkdir <layer> <tag> <path> [mode]               — create synthetic directory"
            );
            println!("");
            println!("Suppression / removal:");
            println!(
                "  whiteout <layer> <tag> <path>                   — hide a lower-layer entry"
            );
            println!("  rm <layer> <tag> <path>                         — remove an overlay entry");
            println!("");
            println!("Inspection:");
            println!(
                "  get <layer> <tag> <path>                        — read content from a layer"
            );
            println!("  ls <tag>                                        — list effective overlay entries");
            println!("  lslayer <layer> <tag>                           — list entries in a layer");
            println!("  tree [tag]                                      — show layered tree (* = winner)");
            println!("                                                    no tag = show all tags");
            println!("");
            println!("Other:");
            println!("  help                                            — show this help");
            println!("  quit                                            — shut down server");
            println!("");
            println!("Input modes:");
            println!("  Interactive:  stdin is a TTY → rustyline REPL");
            println!("  Pipe + TTY:   cat script.vfs - | repl_host → script then REPL");
            println!(
                "  Pure pipe:    cat script.vfs | repl_host → script then serve until signaled"
            );
        }
        "quit" | "exit" => return ControlFlow::Quit,
        _ => {
            println!("unknown command: {line}");
            println!("type 'help' for commands");
        }
    }

    ControlFlow::Continue
}
