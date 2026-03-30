//! repl_host: v1 host-side filesystem server with rustyline admin REPL.
//!
//! Starts one FsServer per guest VM with MemOverlay, listens on a
//! parameterized Unix socket (the vsock host-side path), and serves
//! filesystem operations. Includes a rustyline REPL exposing every
//! MemOverlay API operation for interactive overlay mutation.
//!
//! Supports loading a script file at startup via `--script <file>`.
//! The script is a plain text file of REPL commands, one per line.
//! After the script finishes, the interactive REPL takes over.
//! Lines starting with `#` are comments. Empty lines are skipped.
//!
//! Usage:
//!   cargo run -p motlie-vfs --example repl_host --features vsock -- [options]
//!
//! Options:
//!   --socket <path>   vsock socket path (default: /tmp/motlie-vfs.vsock_5000)
//!   --tag <name>      mount tag (default: alice-home)
//!   --dir <path>      host backing directory (default: temp dir with sample data)
//!   --script <file>   play commands from file at startup, then enter interactive REPL

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
    let mut script_path: Option<String> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" if i + 1 < args.len() => { socket_path = args[i + 1].clone(); i += 2; }
            "--tag" if i + 1 < args.len() => { tag = args[i + 1].clone(); i += 2; }
            "--dir" if i + 1 < args.len() => { host_dir = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--script" if i + 1 < args.len() => { script_path = Some(args[i + 1].clone()); i += 2; }
            other => { host_dir = Some(PathBuf::from(other)); i += 1; }
        }
    }

    let _tempdir;
    let host_path = match host_dir {
        Some(p) => { std::fs::create_dir_all(&p)?; p }
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
    if let Some(ref s) = script_path {
        eprintln!("Script: {s}");
    }
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
    let server_for_repl = Arc::clone(&server);
    let socket_for_cleanup = socket_path.clone();

    tokio::task::spawn_blocking(move || {
        run_repl(server_for_repl, script_path);
        let _ = std::fs::remove_file(&socket_for_cleanup);
        let _ = shutdown_tx.send(());
    });

    let _ = shutdown_rx.await;
    Ok(())
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

fn run_repl(server: Arc<FsServer>, script_path: Option<String>) {
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to create REPL: {e}");
            return;
        }
    };

    // Play script file if provided
    if let Some(path) = script_path {
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                eprintln!("--- playing script: {path} ---");
                for (lineno, line) in content.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    eprintln!("vfs> {trimmed}");
                    if dispatch_command(&server, trimmed) == ControlFlow::Quit {
                        eprintln!("--- script quit at line {} ---", lineno + 1);
                        return;
                    }
                }
                eprintln!("--- script complete, entering interactive REPL ---");
                eprintln!("");
            }
            Err(e) => {
                eprintln!("error: failed to read script {path}: {e}");
                eprintln!("continuing with interactive REPL");
            }
        }
    }

    // Interactive REPL
    loop {
        let line = match rl.readline("vfs> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => { eprintln!("^C"); continue; }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => { eprintln!("REPL error: {e}"); break; }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let _ = rl.add_history_entry(trimmed);

        if dispatch_command(&server, trimmed) == ControlFlow::Quit {
            break;
        }
    }

    eprintln!("shutting down");
}

// ---------------------------------------------------------------------------
// Command dispatch — shared between script playback and interactive REPL
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum ControlFlow {
    Continue,
    Quit,
}

fn dispatch_command(server: &FsServer, line: &str) -> ControlFlow {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() { return ControlFlow::Continue; }

    let overlay = match server.overlay() {
        Some(o) => o,
        None => { println!("error: overlay not enabled"); return ControlFlow::Continue; }
    };

    match parts[0] {
        // --- Layer management ---
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
        "rmlayer" if parts.len() >= 2 => {
            match overlay.remove_layer(parts[1]) {
                Ok(()) => println!("ok: rmlayer {}", parts[1]),
                Err(e) => println!("error: {e}"),
            }
        }
        "layers" => {
            let layers = overlay.layers();
            if layers.is_empty() { println!("(no layers)"); }
            for l in &layers {
                println!("  {} priority={} entries={}", l.name, l.priority, l.entry_count);
            }
        }

        // --- Content injection ---
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
            let uid = match parts[4].parse::<u32>() { Ok(v) => v, Err(_) => { println!("error: uid must be a number"); return ControlFlow::Continue; } };
            let gid = match parts[5].parse::<u32>() { Ok(v) => v, Err(_) => { println!("error: gid must be a number"); return ControlFlow::Continue; } };
            let mode = match u32::from_str_radix(parts[6], 8) { Ok(v) => v, Err(_) => { println!("error: mode must be octal"); return ControlFlow::Continue; } };
            let content = parts[7..].join(" ");
            let attrs = OverlayAttrs { mode, uid, gid };
            match overlay.put_with_attrs(layer, tag, path, attrs, Bytes::from(content.clone())) {
                Ok(()) => println!("ok: putattr {layer} {tag} {path} uid={uid} gid={gid} mode={mode:o} ({} bytes)", content.len()),
                Err(e) => println!("error: {e}"),
            }
        }
        "mkdir" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            let mode = if parts.len() >= 5 {
                u32::from_str_radix(parts[4], 8).unwrap_or(0o755)
            } else { 0o755 };
            let attrs = OverlayAttrs { mode, uid: 0, gid: 0 };
            match overlay.create_dir(layer, tag, path, attrs) {
                Ok(()) => println!("ok: mkdir {layer} {tag} {path} mode={mode:o}"),
                Err(e) => println!("error: {e}"),
            }
        }

        // --- Suppression / removal ---
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

        // --- Inspection ---
        "get" if parts.len() >= 4 => {
            let (layer, tag, path) = (parts[1], parts[2], parts[3]);
            match overlay.get(layer, tag, path) {
                Some(data) => {
                    match std::str::from_utf8(&data) {
                        Ok(s) => println!("{s}"),
                        Err(_) => println!("({} bytes, binary)", data.len()),
                    }
                }
                None => println!("(not found)"),
            }
        }
        "ls" if parts.len() >= 2 => {
            let tag = parts[1];
            let entries = overlay.list_effective(tag);
            if entries.is_empty() { println!("(no overlay entries for tag '{tag}')"); }
            for entry in &entries {
                println!("  {:?} {} uid={} gid={} mode={:o}", entry.kind, entry.path, entry.uid, entry.gid, entry.mode);
            }
            println!("({} entries)", entries.len());
        }
        "lslayer" if parts.len() >= 3 => {
            let (layer, tag) = (parts[1], parts[2]);
            let entries = overlay.list_layer(layer, tag);
            if entries.is_empty() { println!("(no entries in layer '{layer}' for tag '{tag}')"); }
            for entry in &entries {
                println!("  {:?} {} uid={} gid={} mode={:o}", entry.kind, entry.path, entry.uid, entry.gid, entry.mode);
            }
            println!("({} entries)", entries.len());
        }

        // --- Help ---
        "help" => {
            println!("Layer management:");
            println!("  layer <name> <priority>                         — create/update layer");
            println!("  rmlayer <name>                                  — remove layer and all entries");
            println!("  layers                                          — list all layers");
            println!("");
            println!("Content injection:");
            println!("  put <layer> <tag> <path> <content>              — inject file (default attrs)");
            println!("  putattr <layer> <tag> <path> <uid> <gid> <mode> <content>");
            println!("                                                  — inject file with explicit attrs");
            println!("  mkdir <layer> <tag> <path> [mode]               — create synthetic directory");
            println!("");
            println!("Suppression / removal:");
            println!("  whiteout <layer> <tag> <path>                   — hide a lower-layer entry");
            println!("  rm <layer> <tag> <path>                         — remove an overlay entry");
            println!("");
            println!("Inspection:");
            println!("  get <layer> <tag> <path>                        — read content from a layer");
            println!("  ls <tag>                                        — list effective overlay entries");
            println!("  lslayer <layer> <tag>                           — list entries in a layer");
            println!("");
            println!("Other:");
            println!("  help                                            — show this help");
            println!("  quit                                            — shut down");
        }

        "quit" | "exit" => return ControlFlow::Quit,

        _ => {
            println!("unknown command: {line}");
            println!("type 'help' for commands");
        }
    }

    ControlFlow::Continue
}
