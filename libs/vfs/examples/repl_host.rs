//! repl_host: proof-of-concept host server for the v1 CH harness.
//!
//! Starts one FsServer per guest VM with MemOverlay, listens on a
//! parameterized Unix socket (the vsock host-side path), and serves
//! filesystem operations. Includes a rustyline REPL for overlay mutation.
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
//!
//! REPL commands:
//!   put <layer> <tag> <path> <content>   — inject a file
//!   whiteout <layer> <tag> <path>        — hide a lower-layer file
//!   rm <layer> <tag> <path>              — remove an overlay entry
//!   ls <tag>                             — list effective overlay entries
//!   layers                               — list all overlay layers
//!   help                                 — show commands
//!   quit                                 — shut down

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use tokio::net::UnixListener;
use tokio::sync::oneshot;

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

    // One FsServer per guest VM
    let server = Arc::new(
        FsServer::builder()
            .mount(&tag, host_path, false)
            .overlay(true)
            .events(256)
            .build()?,
    );

    if let Some(overlay) = server.overlay() {
        overlay.put_layer("credentials", 0)?;
        eprintln!("Overlay layer 'credentials' created (priority 0)");
    }

    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)?;
    eprintln!("Listening on {socket_path} (guest filesystem connections)");
    eprintln!("Type 'help' for REPL commands.");
    eprintln!("");

    // Spawn vsock connection acceptor (guest filesystem traffic)
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

    // Spawn rustyline REPL on a blocking thread (admin is in-process only)
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_for_repl = Arc::clone(&server);
    let socket_for_cleanup = socket_path.clone();

    tokio::task::spawn_blocking(move || {
        run_repl(server_for_repl, shutdown_tx);
        let _ = std::fs::remove_file(&socket_for_cleanup);
    });

    // Wait for REPL to signal shutdown
    let _ = shutdown_rx.await;
    Ok(())
}

fn run_repl(server: Arc<FsServer>, shutdown: oneshot::Sender<()>) {
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to create REPL: {e}");
            let _ = shutdown.send(());
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
        if trimmed.is_empty() { continue; }
        let _ = rl.add_history_entry(trimmed);

        let parts: Vec<&str> = trimmed.split_whitespace().collect();

        match parts[0] {
            "put" if parts.len() >= 5 => {
                let (layer, tag, path) = (parts[1], parts[2], parts[3]);
                let content = parts[4..].join(" ");
                if let Some(overlay) = server.overlay() {
                    match overlay.put(layer, tag, path, Bytes::from(content.clone())) {
                        Ok(()) => println!("ok: put {layer} {tag} {path} ({} bytes)", content.len()),
                        Err(e) => println!("error: {e}"),
                    }
                }
            }
            "whiteout" if parts.len() >= 4 => {
                let (layer, tag, path) = (parts[1], parts[2], parts[3]);
                if let Some(overlay) = server.overlay() {
                    match overlay.whiteout(layer, tag, path) {
                        Ok(()) => println!("ok: whiteout {layer} {tag} {path}"),
                        Err(e) => println!("error: {e}"),
                    }
                }
            }
            "rm" if parts.len() >= 4 => {
                let (layer, tag, path) = (parts[1], parts[2], parts[3]);
                if let Some(overlay) = server.overlay() {
                    match overlay.remove(layer, tag, path) {
                        Ok(()) => println!("ok: rm {layer} {tag} {path}"),
                        Err(e) => println!("error: {e}"),
                    }
                }
            }
            "ls" if parts.len() >= 2 => {
                let tag = parts[1];
                if let Some(overlay) = server.overlay() {
                    let entries = overlay.list_effective(tag);
                    if entries.is_empty() {
                        println!("(no overlay entries for tag '{tag}')");
                    }
                    for entry in &entries {
                        println!("  {:?} {} uid={} gid={} mode={:o}", entry.kind, entry.path, entry.uid, entry.gid, entry.mode);
                    }
                    println!("({} entries)", entries.len());
                }
            }
            "layers" => {
                if let Some(overlay) = server.overlay() {
                    let layers = overlay.layers();
                    if layers.is_empty() {
                        println!("(no layers)");
                    }
                    for l in &layers {
                        println!("  {} priority={} entries={}", l.name, l.priority, l.entry_count);
                    }
                }
            }
            "help" => {
                println!("Commands:");
                println!("  put <layer> <tag> <path> <content>  — inject a file");
                println!("  whiteout <layer> <tag> <path>       — hide a lower-layer file");
                println!("  rm <layer> <tag> <path>             — remove an overlay entry");
                println!("  ls <tag>                            — list effective overlay entries");
                println!("  layers                              — list all overlay layers");
                println!("  help                                — show this help");
                println!("  quit                                — shut down");
            }
            "quit" | "exit" => break,
            _ => {
                println!("unknown command: {trimmed}");
                println!("type 'help' for commands");
            }
        }
    }

    eprintln!("shutting down");
    let _ = shutdown.send(());
}
