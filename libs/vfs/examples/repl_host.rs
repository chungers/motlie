//! repl_host: v1 host-side filesystem server with rustyline admin REPL.
//!
//! Starts one FsServer per guest VM with MemOverlay, listens on a
//! parameterized Unix socket (the vsock host-side path), and serves
//! filesystem operations. Includes a rustyline REPL exposing every
//! MemOverlay API operation for interactive overlay mutation.
//!
//! Each guest VM gets its own FsServer instance and its own vsock socket.
//! Tags identify mounted subtrees within that VM's server.
//! Admin is in-process REPL only — no network admin connections.
//!
//! Usage:
//!   cargo run -p motlie-vfs --example repl_host --features vsock -- [options]
//!
//! Options:
//!   --socket <path>   vsock socket path (default: /tmp/motlie-vfs.vsock_5000)
//!   --tag <name>      mount tag (default: alice-home)
//!   --dir <path>      host backing directory (default: temp dir with sample data)

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
            "--socket" if i + 1 < args.len() => { socket_path = args[i + 1].clone(); i += 2; }
            "--tag" if i + 1 < args.len() => { tag = args[i + 1].clone(); i += 2; }
            "--dir" if i + 1 < args.len() => { host_dir = Some(PathBuf::from(&args[i + 1])); i += 2; }
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

    // Spawn vsock connection acceptor (guest filesystem traffic only)
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

    // Spawn rustyline REPL on a blocking thread (in-process admin, no network)
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_for_repl = Arc::clone(&server);
    let socket_for_cleanup = socket_path.clone();

    tokio::task::spawn_blocking(move || {
        run_repl(server_for_repl);
        let _ = std::fs::remove_file(&socket_for_cleanup);
        let _ = shutdown_tx.send(());
    });

    let _ = shutdown_rx.await;
    Ok(())
}

fn run_repl(server: Arc<FsServer>) {
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
            Err(rustyline::error::ReadlineError::Interrupted) => { eprintln!("^C"); continue; }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => { eprintln!("REPL error: {e}"); break; }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let _ = rl.add_history_entry(trimmed);

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let overlay = match server.overlay() {
            Some(o) => o,
            None => { println!("error: overlay not enabled"); continue; }
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
                let uid = match parts[4].parse::<u32>() { Ok(v) => v, Err(_) => { println!("error: uid must be a number"); continue; } };
                let gid = match parts[5].parse::<u32>() { Ok(v) => v, Err(_) => { println!("error: gid must be a number"); continue; } };
                let mode = match u32::from_str_radix(parts[6], 8) { Ok(v) => v, Err(_) => { println!("error: mode must be octal"); continue; } };
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
                    match u32::from_str_radix(parts[4], 8) { Ok(v) => v, Err(_) => 0o755 }
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

            "quit" | "exit" => break,

            _ => {
                println!("unknown command: {trimmed}");
                println!("type 'help' for commands");
            }
        }
    }

    eprintln!("shutting down");
}
