//! simple_host: proof-of-concept host server for the v1 CH harness.
//!
//! Starts an FsServer with MemOverlay, listens on a Unix socket (simulating
//! the vsock host-side path), and serves filesystem operations to connecting
//! guests. Includes a simple stdin command loop for overlay mutation.
//!
//! Usage:
//!   cargo run -p motlie-vfs --example simple_host --features vsock
//!
//! By default:
//!   - Mounts a host directory at tag "alice-home"
//!   - Listens on /tmp/motlie-vfs.vsock_5000 (CH vsock convention)
//!   - Enables overlay with a "credentials" layer
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

/// Default vsock socket path (CH convention: <base>_<port>).
const VSOCK_SOCKET: &str = "/tmp/motlie-vfs.vsock_5000";

/// Default tag for the test mount.
const DEFAULT_TAG: &str = "alice-home";

#[tokio::main]
async fn main() -> Result<()> {
    // Parse optional host directory from args, default to a tempdir
    let host_dir = std::env::args().nth(1).map(PathBuf::from);
    let _tempdir; // keep alive if we create one
    let host_path = match host_dir {
        Some(p) => {
            std::fs::create_dir_all(&p)?;
            p
        }
        None => {
            _tempdir = tempfile::tempdir()?;
            let p = _tempdir.path().to_path_buf();
            // Seed some test data
            std::fs::create_dir_all(p.join("projects"))?;
            std::fs::write(p.join("projects/README.md"), b"hello from host disk\n")?;
            std::fs::write(p.join(".bashrc"), b"# bashrc\n")?;
            eprintln!("Created temp host dir: {}", p.display());
            p
        }
    };

    eprintln!("=== motlie-vfs simple_host ===");
    eprintln!("Host dir: {}", host_path.display());
    eprintln!("Tag: {DEFAULT_TAG}");
    eprintln!("Socket: {VSOCK_SOCKET}");

    // Build FsServer with overlay
    let server = Arc::new(
        FsServer::builder()
            .mount(DEFAULT_TAG, host_path, false)
            .overlay(true)
            .events(256)
            .build()?,
    );

    // Set up a default credentials layer
    if let Some(overlay) = server.overlay() {
        overlay.put_layer("credentials", 0)?;
        eprintln!("Overlay layer 'credentials' created (priority 0)");
    }

    // Remove stale socket
    let _ = std::fs::remove_file(VSOCK_SOCKET);

    // Listen for guest connections on the vsock Unix socket
    let listener = UnixListener::bind(VSOCK_SOCKET)?;
    eprintln!("Listening on {VSOCK_SOCKET}");
    eprintln!("");
    eprintln!("Commands: put <layer> <tag> <path> <content>");
    eprintln!("          whiteout <layer> <tag> <path>");
    eprintln!("          rm <layer> <tag> <path>");
    eprintln!("          ls <tag>");
    eprintln!("          quit");
    eprintln!("");

    // Spawn the connection acceptor
    let server_for_accept = Arc::clone(&server);
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let handler = VsockConnectionHandler::new(
                        Arc::clone(&server_for_accept),
                        DEFAULT_TAG,
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

    // Simple stdin command loop for overlay mutation
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        use tokio::io::AsyncBufReadExt;
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
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
                } else {
                    eprintln!("overlay not enabled");
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

    // Clean up socket
    let _ = std::fs::remove_file(VSOCK_SOCKET);
    Ok(())
}
