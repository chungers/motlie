use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use uuid::Uuid;

use crate::config::CallArgs;
use crate::http::AgentState;

#[derive(Debug, Deserialize, Serialize)]
struct DialRequest {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    to: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DialResponse {
    id: String,
    ok: bool,
    call_id: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

pub async fn run_daemon_socket(path: PathBuf, state: AgentState) -> anyhow::Result<()> {
    prepare_socket_path(&path)?;
    let listener = UnixListener::bind(&path)?;
    tracing::info!(socket = %path.display(), "telnyx_agent.socket.listening");
    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(error) = handle_connection(stream, state).await {
                tracing::warn!(error = %error, "telnyx_agent.socket.connection_failed");
            }
        });
    }
}

async fn handle_connection(stream: UnixStream, state: AgentState) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line).await?;
        if read == 0 {
            return Ok(());
        }
        let response = match serde_json::from_str::<DialRequest>(line.trim()) {
            Ok(request) if request.kind == "dial" => {
                let callback_url = format!("{}/motlie/outbound-connected", state.public_url);
                match state
                    .gateway
                    .place_outbound_call(&request.to, &callback_url, state.outbound_timeout_ms)
                    .await
                {
                    Ok(call) => DialResponse {
                        id: request.id,
                        ok: true,
                        call_id: Some(call.call_id),
                        state: Some(call.state),
                        error: None,
                    },
                    Err(error) => DialResponse {
                        id: request.id,
                        ok: false,
                        call_id: None,
                        state: None,
                        error: Some(format!("{error:#}")),
                    },
                }
            }
            Ok(request) => DialResponse {
                id: request.id,
                ok: false,
                call_id: None,
                state: None,
                error: Some("unsupported request type".to_string()),
            },
            Err(error) => DialResponse {
                id: "invalid".to_string(),
                ok: false,
                call_id: None,
                state: None,
                error: Some(format!("invalid request: {error}")),
            },
        };
        let encoded = serde_json::to_string(&response)?;
        writer.write_all(encoded.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
}

pub async fn run_call_client(args: CallArgs) -> anyhow::Result<()> {
    let mut stream = UnixStream::connect(&args.socket)
        .await
        .with_context(|| format!("connect {}", args.socket.display()))?;
    let request = DialRequest {
        id: format!("req_{}", Uuid::new_v4().simple()),
        kind: "dial".to_string(),
        to: args.to,
    };
    let encoded = serde_json::to_string(&request)?;
    stream.write_all(encoded.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader.read_line(&mut response).await?;
    println!("{}", response.trim_end());
    Ok(())
}

fn prepare_socket_path(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => std::fs::remove_file(path)?,
        Ok(_) => anyhow::bail!("refusing to replace non-socket path {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
