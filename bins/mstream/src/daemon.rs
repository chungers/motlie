use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{watch, Mutex};
use tokio::time::sleep;

use crate::jsonl;
use crate::protocol::ClientRequest;
use crate::state::DaemonState;

pub async fn start_background(socket: &Path) -> anyhow::Result<Vec<Value>> {
    if daemon_reachable(socket).await {
        return Ok(vec![jsonl::error(
            "daemon_already_running",
            format!("daemon is already reachable at {}", socket.display()),
        )]);
    }

    let exe = std::env::current_exe().context("failed to resolve current executable")?;
    let child = std::process::Command::new(exe)
        .arg("--socket")
        .arg(socket)
        .arg("daemon")
        .arg("start")
        .arg("--foreground")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn mstream daemon")?;

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if daemon_reachable(socket).await {
            return Ok(vec![serde_json::json!({
                "type": "ok",
                "op": "daemon_start",
                "socket": socket,
                "pid": child.id(),
            })]);
        }
        sleep(Duration::from_millis(50)).await;
    }

    bail!(
        "daemon process {} did not become reachable at {}",
        child.id(),
        socket.display()
    )
}

pub async fn run_foreground(socket: PathBuf) -> anyhow::Result<()> {
    prepare_socket(&socket).await?;
    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("failed to bind daemon socket {}", socket.display()))?;
    let event_store_path = DaemonState::event_store_path_for_socket(&socket);
    let state = Arc::new(Mutex::new(DaemonState::with_event_store(event_store_path)?));
    let output_audit_task = DaemonState::spawn_output_audit_task(Arc::clone(&state)).await?;
    let (stop_tx, mut stop_rx) = watch::channel(false);

    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let state = Arc::clone(&state);
                let stop_tx = stop_tx.clone();
                tokio::spawn(async move {
                    if let Err(err) = handle_connection(stream, state, stop_tx).await {
                        eprintln!("mstream daemon connection error: {err}");
                    }
                });
            }
            changed = stop_rx.changed() => {
                if changed.is_err() || *stop_rx.borrow() {
                    break;
                }
            }
        }
    }

    output_audit_task.abort();
    let _ = fs::remove_file(&socket);
    Ok(())
}

async fn handle_connection(
    stream: UnixStream,
    state: Arc<Mutex<DaemonState>>,
    stop_tx: watch::Sender<bool>,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let read = reader.read_line(&mut line).await?;
    if read == 0 {
        return Ok(());
    }

    let parsed: Result<ClientRequest, _> = serde_json::from_str(line.trim_end());
    let (records, stop) = match parsed {
        Ok(request) => {
            let stop = DaemonState::should_stop(&request);
            let records = match DaemonState::handle_shared(Arc::clone(&state), request).await {
                Ok(records) => records,
                Err(err) => vec![jsonl::error("request_failed", err.to_string())],
            };
            (records, stop)
        }
        Err(err) => (vec![jsonl::error("bad_request", err.to_string())], false),
    };

    let mut stream = reader.into_inner();
    for record in records {
        stream
            .write_all(serde_json::to_string(&record)?.as_bytes())
            .await?;
        stream.write_all(b"\n").await?;
    }
    stream.shutdown().await?;
    if stop {
        let _ = stop_tx.send(true);
    }
    Ok(())
}

async fn prepare_socket(socket: &Path) -> anyhow::Result<()> {
    if daemon_reachable(socket).await {
        bail!("daemon is already reachable at {}", socket.display());
    }
    match fs::remove_file(socket) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(err)
                .with_context(|| format!("failed to remove stale socket {}", socket.display()))
        }
    }
    Ok(())
}

async fn daemon_reachable(socket: &Path) -> bool {
    UnixStream::connect(socket).await.is_ok()
}
