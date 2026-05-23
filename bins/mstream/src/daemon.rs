use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
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
    let mut state = DaemonState::default();

    loop {
        let (stream, _) = listener.accept().await?;
        let stop = handle_connection(stream, &mut state).await?;
        if stop {
            break;
        }
    }

    let _ = fs::remove_file(&socket);
    Ok(())
}

pub async fn send_request(socket: &Path, request: &ClientRequest) -> anyhow::Result<Vec<Value>> {
    let mut stream = UnixStream::connect(socket)
        .await
        .with_context(|| format!("daemon unreachable at {}", socket.display()))?;
    let request = serde_json::to_vec(request)?;
    stream.write_all(&request).await?;
    stream.write_all(b"\n").await?;
    stream.shutdown().await?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let mut records = Vec::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line).await?;
        if read == 0 {
            break;
        }
        let record: Value = serde_json::from_str(line.trim_end())?;
        records.push(record);
    }
    Ok(records)
}

async fn handle_connection(stream: UnixStream, state: &mut DaemonState) -> anyhow::Result<bool> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let read = reader.read_line(&mut line).await?;
    if read == 0 {
        return Ok(false);
    }

    let parsed: Result<ClientRequest, _> = serde_json::from_str(line.trim_end());
    let (records, stop) = match parsed {
        Ok(request) => {
            let stop = DaemonState::should_stop(&request);
            let records = match state.handle(request).await {
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
    Ok(stop)
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
