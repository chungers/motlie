//! simple_host: proof-of-concept host server for the v1 CH harness.
//!
//! Starts one FsServer per guest VM with MemOverlay, listens on a
//! parameterized Unix socket (the vsock host-side path), and serves
//! filesystem operations. Includes a stdin command loop for overlay mutation.
//!
//! Each guest VM gets its own FsServer instance and its own vsock socket.
//! Tags identify mounted subtrees within that VM's server.
//!
//! Usage:
//!   cargo run -p motlie-vfs --example simple_host --features vsock -- [options]
//!
//! Options:
//!   --socket <path>   vsock socket path (default: /tmp/motlie-vfs.vsock_5000)
//!   --tag <name>      mount tag (default: alice-home)
//!   --dir <path>      host backing directory (default: temp dir with sample data)
//!
//! Commands (stdin):
//!   put <layer> <tag> <path> <content>   — inject a file
//!   whiteout <layer> <tag> <path>        — hide a lower-layer file
//!   rm <layer> <tag> <path>              — remove an overlay entry
//!   ls <tag>                             — list effective overlay entries
//!   quit                                 — shut down

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use tokio::io::BufReader;
use tokio::net::UnixListener;

use motlie_vfs::core::server::FsServer;
use motlie_vfs::vsock::handler::VsockConnectionHandler;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments
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
            other => {
                // Backwards compat: bare arg is host dir
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

    eprintln!("=== motlie-vfs simple_host ===");
    eprintln!("Host dir: {}", host_path.display());
    eprintln!("Tag: {tag}");
    eprintln!("Socket: {socket_path}");
    eprintln!("");
    eprintln!("One FsServer per guest VM. Each VM gets its own socket.");
    eprintln!("For multiple guests, run separate instances with different");
    eprintln!("--socket and --tag values.");
    eprintln!("");

    // Build FsServer with overlay — one per guest VM
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

    // Remove stale socket
    let _ = std::fs::remove_file(&socket_path);

    // Listen for guest connections
    let listener = UnixListener::bind(&socket_path)?;
    eprintln!("Listening on {socket_path}");
    eprintln!("");
    eprintln!("Commands: put <layer> <tag> <path> <content>");
    eprintln!("          whiteout <layer> <tag> <path>");
    eprintln!("          rm <layer> <tag> <path>");
    eprintln!("          ls <tag>");
    eprintln!("          quit");
    eprintln!("");

    // Spawn the connection acceptor
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
                Err(e) => {
                    eprintln!("Accept error: {e}");
                }
            }
        }
    });

    // Stdin command loop for overlay mutation (in-process admin, no network)
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        use tokio::io::AsyncBufReadExt;
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }

        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "put" if parts.len() >= 5 => {
                let (layer, tag, path) = (parts[1], parts[2], parts[3]);
                let content = parts[4..].join(" ");
                if let Some(overlay) = server.overlay() {
                    match overlay.put(layer, tag, path, Bytes::from(content.clone())) {
                        Ok(()) => eprintln!("put {layer} {tag} {path} ({} bytes)", content.len()),
                        Err(e) => eprintln!("error: {e}"),
                    }
                }
            }
            "whiteout" if parts.len() >= 4 => {
                let (layer, tag, path) = (parts[1], parts[2], parts[3]);
                if let Some(overlay) = server.overlay() {
                    match overlay.whiteout(layer, tag, path) {
                        Ok(()) => eprintln!("whiteout {layer} {tag} {path}"),
                        Err(e) => eprintln!("error: {e}"),
                    }
                }
            }
            "rm" if parts.len() >= 4 => {
                let (layer, tag, path) = (parts[1], parts[2], parts[3]);
                if let Some(overlay) = server.overlay() {
                    match overlay.remove(layer, tag, path) {
                        Ok(()) => eprintln!("rm {layer} {tag} {path}"),
                        Err(e) => eprintln!("error: {e}"),
                    }
                }
            }
            "ls" if parts.len() >= 2 => {
                let tag = parts[1];
                if let Some(overlay) = server.overlay() {
                    let entries = overlay.list_effective(tag);
                    if entries.is_empty() {
                        eprintln!("(no overlay entries for tag '{tag}')");
                    }
                    for entry in entries {
                        eprintln!("  {:?} {} uid={} gid={} mode={:o}", entry.kind, entry.path, entry.uid, entry.gid, entry.mode);
                    }
                }
            }
            "quit" | "exit" => {
                eprintln!("shutting down");
                break;
            }
            _ => {
                eprintln!("unknown command: {}", line.trim());
                eprintln!("commands: put whiteout rm ls quit");
            }
        }
    }

    let _ = std::fs::remove_file(&socket_path);
    Ok(())
}
