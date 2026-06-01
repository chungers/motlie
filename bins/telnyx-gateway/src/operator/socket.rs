use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_driver::{CommandEffect, CommandEngine};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

use crate::operator::commands::{GatewayCommand, GatewayContext};

pub type SharedCommandEngine = Arc<Mutex<CommandEngine<GatewayContext, GatewayCommand>>>;

#[derive(Debug, Serialize)]
struct SocketCommandResponse {
    ok: bool,
    lines: Vec<String>,
    effects: Vec<&'static str>,
    error: Option<String>,
}

impl SocketCommandResponse {
    fn ok(lines: Vec<String>, effects: Vec<CommandEffect>) -> Self {
        Self {
            ok: true,
            lines,
            effects: effects.into_iter().map(effect_label).collect(),
            error: None,
        }
    }

    fn error(error: impl Into<String>) -> Self {
        Self {
            ok: false,
            lines: Vec::new(),
            effects: Vec::new(),
            error: Some(error.into()),
        }
    }
}

fn effect_label(effect: CommandEffect) -> &'static str {
    match effect {
        CommandEffect::ExitShell => "exit-shell",
        CommandEffect::EnterTui => "enter-tui",
        CommandEffect::ExitTui => "exit-tui",
    }
}

pub async fn run_command_socket(path: PathBuf, engine: SharedCommandEngine) -> anyhow::Result<()> {
    prepare_socket_path(&path)?;
    let listener = UnixListener::bind(&path)?;
    tracing::info!(socket = %path.display(), "operator.socket.listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let engine = Arc::clone(&engine);
        tokio::spawn(async move {
            if let Err(error) = handle_connection(stream, engine).await {
                tracing::warn!(error = %error, "operator.socket.connection_failed");
            }
        });
    }
}

fn prepare_socket_path(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

async fn handle_connection(stream: UnixStream, engine: SharedCommandEngine) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let read = reader.read_line(&mut line).await?;
        if read == 0 {
            return Ok(());
        }
        let command = line.trim();
        if command.is_empty() {
            continue;
        }

        let response = {
            let mut guard = engine.lock().await;
            match guard.run_line(command).await {
                Ok(output) => SocketCommandResponse::ok(output.lines, output.effects),
                Err(error) => SocketCommandResponse::error(error.to_string()),
            }
        };
        let encoded = serde_json::to_string(&response)?;
        writer.write_all(encoded.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call_control::TelnyxClient;
    use crate::operator::state::shared_state;

    #[tokio::test]
    async fn command_socket_runs_driver_commands() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let engine = Arc::new(Mutex::new(
            CommandEngine::<GatewayContext, GatewayCommand>::new(GatewayContext::new(
                state, telnyx,
            )),
        ));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), engine));

        let mut stream = connect_with_retry(&path).await;
        stream
            .write_all(b"status\n")
            .await
            .expect("write command to socket");
        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .await
            .expect("read socket response");
        let response: serde_json::Value =
            serde_json::from_str(&response).expect("response should be JSON");

        assert_eq!(response["ok"], true);
        assert!(response["lines"][0]
            .as_str()
            .expect("line should be a string")
            .starts_with("listener:"));

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    async fn connect_with_retry(path: &Path) -> UnixStream {
        for _ in 0..20 {
            match UnixStream::connect(path).await {
                Ok(stream) => return stream,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(10)).await,
            }
        }
        UnixStream::connect(path)
            .await
            .expect("socket should become available")
    }
}
