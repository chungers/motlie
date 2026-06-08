use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_driver::{CommandEffect, CommandEngine};
use serde::Serialize;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::operator::commands::{GatewayCommand, GatewayContext};
use crate::operator::script::run_operator_line;

pub type SharedCommandContext = Arc<GatewayContext>;

#[derive(Debug, Serialize)]
struct SocketCommandResponse {
    ok: bool,
    lines: Vec<String>,
    effects: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    error: Option<String>,
}

impl SocketCommandResponse {
    fn ok(command: &str, lines: Vec<String>, effects: Vec<CommandEffect>) -> Self {
        let data = structured_data(command, &lines);
        Self {
            ok: true,
            lines,
            effects: effects.into_iter().map(effect_label).collect(),
            data,
            error: None,
        }
    }

    fn error(error: impl Into<String>) -> Self {
        Self {
            ok: false,
            lines: Vec::new(),
            effects: Vec::new(),
            data: None,
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

pub async fn run_command_socket(
    path: PathBuf,
    context: SharedCommandContext,
) -> anyhow::Result<()> {
    prepare_socket_path(&path)?;
    let listener = UnixListener::bind(&path)?;
    tracing::info!(socket = %path.display(), "operator.socket.listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let context = Arc::clone(&context);
        tokio::spawn(async move {
            if let Err(error) = handle_connection(stream, context).await {
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

async fn handle_connection(
    stream: UnixStream,
    context: SharedCommandContext,
) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context.for_new_source());
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

        let response = match run_operator_line(&mut engine, command).await {
            Ok(output) => SocketCommandResponse::ok(command, output.lines, output.effects),
            Err(error) => SocketCommandResponse::error(error.to_string()),
        };
        let encoded = serde_json::to_string(&response)?;
        writer.write_all(encoded.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
}

fn structured_data(command: &str, lines: &[String]) -> Option<Value> {
    let argv = shlex::split(command)?;
    let root = argv.first()?.as_str();
    match root {
        "status" => Some(json!({
            "kind": "status",
            "fields": fields_from_lines(lines),
            "lines": lines,
        })),
        "calls" => Some(json!({
            "kind": "calls",
            "calls": call_rows(lines),
            "lines": lines,
        })),
        "call" if argv.get(1).is_some_and(|part| part == "show") => Some(json!({
            "kind": "call",
            "fields": fields_from_lines(lines),
            "lines": lines,
        })),
        "tts" if argv.get(1).is_some_and(|part| part == "list") => Some(json!({
            "kind": "tts_list",
            "backends": tts_list_rows(lines),
            "lines": lines,
        })),
        "tts" if argv.get(1).is_some_and(|part| part == "status") => {
            let active_start = lines
                .iter()
                .position(|line| line.starts_with("active-call "))
                .unwrap_or(lines.len());
            Some(json!({
                "kind": "tts",
                "fields": fields_from_lines(&lines[..active_start]),
                "active": tts_rows(&lines[active_start..]),
                "lines": lines,
            }))
        }
        _ => None,
    }
}

fn fields_from_lines(lines: &[String]) -> Value {
    let mut fields = Map::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(": ") {
            fields.insert(normalize_key(key), Value::String(value.to_string()));
        } else {
            fields.extend(kv_fields(line));
        }
    }
    Value::Object(fields)
}

fn call_rows(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .filter(|line| line.as_str() != "no calls")
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let call_id = parts.next()?;
            let status = parts.next()?;
            let mut fields = Map::new();
            fields.insert("call".to_string(), Value::String(call_id.to_string()));
            fields.insert("status".to_string(), Value::String(status.to_string()));
            for part in parts {
                if let Some((key, value)) = part.split_once('=') {
                    fields.insert(normalize_key(key), Value::String(value.to_string()));
                }
            }
            Some(Value::Object(fields))
        })
        .collect()
}

fn tts_rows(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .filter_map(|line| {
            let line = line.strip_prefix("active-call ").unwrap_or(line);
            let mut parts = line.split_whitespace();
            let call_id = parts.next()?;
            let status = parts.next()?;
            let mut fields = kv_fields_from_parts(parts);
            if let Value::Object(map) = &mut fields {
                map.insert("call".to_string(), Value::String(call_id.to_string()));
                map.insert("status".to_string(), Value::String(status.to_string()));
            }
            Some(fields)
        })
        .collect()
}

fn tts_list_rows(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let backend = parts.next()?;
            let model = parts.next()?;
            let status = parts.next()?;
            Some(json!({
                "backend": backend,
                "model": model,
                "status": status.trim_end_matches(':'),
                "line": line,
            }))
        })
        .collect()
}

fn kv_fields_from_line(line: &str) -> Value {
    kv_fields_from_parts(line.split_whitespace())
}

fn kv_fields_from_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> Value {
    let mut fields = Map::new();
    for part in parts {
        if let Some((key, value)) = part.split_once('=') {
            fields.insert(normalize_key(key), Value::String(value.to_string()));
        }
    }
    Value::Object(fields)
}

fn kv_fields(line: &str) -> Map<String, Value> {
    let Value::Object(fields) = kv_fields_from_line(line) else {
        return Map::new();
    };
    fields
}

fn normalize_key(key: &str) -> String {
    key.trim().replace(['-', ' '], "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call_control::TelnyxClient;
    use crate::operator::state::{shared_state, CallStatus, TelnyxIds};

    #[tokio::test]
    async fn command_socket_runs_driver_commands() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

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
        assert_eq!(response["data"]["kind"], "status");
        let listener = response["data"]["fields"]["listener"]
            .as_str()
            .expect("listener should be structured");
        assert!(listener.starts_with("127.0.0.1:"));
        assert!(!listener.contains("Some("));
        assert_eq!(
            response["data"]["fields"]["conversation_handler"],
            "disabled"
        );
        assert_eq!(response["data"]["fields"]["conversation_barge_in"], "on");

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn command_socket_returns_structured_calls_for_agent_polling() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let call_id = {
            let mut guard = state.write().await;
            add_test_call(&mut guard, "call-1")
        };
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

        let mut client = SocketTestClient::connect(&path).await;
        let response = client.command("calls").await;

        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["kind"], "calls");
        assert_eq!(response["data"]["calls"][0]["call"], call_id);
        assert_eq!(response["data"]["calls"][0]["status"], "waiting");

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn command_socket_exposes_help_to_agents() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

        let mut client = SocketTestClient::connect(&path).await;
        let response = client.command("help socket").await;
        let lines = response_lines(&response).join("\n");

        assert!(lines.contains("Agent socket interface"));
        assert!(lines.contains("Receive one JSON object"));
        assert!(lines.contains("Operational parity"));

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn command_socket_returns_structured_tts_discovery() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

        let mut client = SocketTestClient::connect(&path).await;
        let list = client.command("tts list").await;
        let status = client.command("tts status").await;

        assert_eq!(list["ok"], true);
        assert_eq!(list["data"]["kind"], "tts_list");
        assert_eq!(list["data"]["backends"][0]["backend"], "kokoro-82m");
        assert_eq!(list["data"]["backends"][0]["status"], "unavailable");
        assert_eq!(status["ok"], true);
        assert_eq!(status["data"]["kind"], "tts");
        assert_eq!(status["data"]["fields"]["next"], "kokoro-82m");
        assert_eq!(status["data"]["fields"]["status"], "unavailable");

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn socket_connections_keep_source_local_selection() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let (call_one, call_two) = {
            let mut guard = state.write().await;
            let call_one = add_test_call(&mut guard, "call-1");
            let call_two = add_test_call(&mut guard, "call-2");
            (call_one, call_two)
        };
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

        let mut client_one = SocketTestClient::connect(&path).await;
        let mut client_two = SocketTestClient::connect(&path).await;

        let first_use = client_one.command(&format!("call use {call_one}")).await;
        let second_use = client_two.command(&format!("call use {call_two}")).await;
        assert_eq!(first_use["ok"], true);
        assert_eq!(second_use["ok"], true);

        let first_show = client_one.command("call show").await;
        let second_show = client_two.command("call show").await;

        assert!(response_lines(&first_show)
            .iter()
            .any(|line| line == &format!("call: {call_one}")));
        assert!(response_lines(&second_show)
            .iter()
            .any(|line| line == &format!("call: {call_two}")));

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn socket_connections_keep_source_local_asr_backend() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

        let mut client_one = SocketTestClient::connect(&path).await;
        let mut client_two = SocketTestClient::connect(&path).await;

        let first_use = client_one.command("asr use sherpa-2023").await;
        assert_eq!(first_use["ok"], true);
        let first_status = client_one.command("asr status").await;
        let second_status = client_two.command("asr status").await;

        assert!(response_lines(&first_status)
            .iter()
            .any(|line| line == "next=sherpa-2023"));
        assert!(response_lines(&second_status)
            .iter()
            .any(|line| line == "next=kroko-2025"));

        socket_task.abort();
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn socket_opened_after_asr_use_keeps_startup_default_backend() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let state = shared_state("127.0.0.1:0".parse().expect("valid address"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = Arc::new(GatewayContext::new(state, telnyx));
        let socket_task = tokio::spawn(run_command_socket(path.clone(), context));

        let mut client_one = SocketTestClient::connect(&path).await;
        let first_use = client_one.command("asr use sherpa-2023").await;
        assert_eq!(first_use["ok"], true);

        let mut client_two = SocketTestClient::connect(&path).await;
        let first_status = client_one.command("asr status").await;
        let second_status = client_two.command("asr status").await;

        assert!(response_lines(&first_status)
            .iter()
            .any(|line| line == "next=sherpa-2023"));
        assert!(response_lines(&second_status)
            .iter()
            .any(|line| line == "next=kroko-2025"));

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

    fn add_test_call(state: &mut crate::operator::state::GatewayState, control_id: &str) -> String {
        state.add_or_update_inbound_call(
            TelnyxIds {
                call_control_id: control_id.to_string(),
                call_session_id: None,
                call_leg_id: None,
                stream_id: None,
            },
            None,
            None,
            CallStatus::PendingInbound,
        )
    }

    struct SocketTestClient {
        reader: BufReader<tokio::net::unix::OwnedReadHalf>,
        writer: tokio::net::unix::OwnedWriteHalf,
    }

    impl SocketTestClient {
        async fn connect(path: &Path) -> Self {
            let stream = connect_with_retry(path).await;
            let (reader, writer) = stream.into_split();
            Self {
                reader: BufReader::new(reader),
                writer,
            }
        }

        async fn command(&mut self, command: &str) -> serde_json::Value {
            self.writer
                .write_all(format!("{command}\n").as_bytes())
                .await
                .expect("write socket command");
            let mut response = String::new();
            self.reader
                .read_line(&mut response)
                .await
                .expect("read socket response");
            serde_json::from_str(&response).expect("socket response should be JSON")
        }
    }

    fn response_lines(response: &serde_json::Value) -> Vec<String> {
        assert_eq!(response["ok"], true);
        response["lines"]
            .as_array()
            .expect("lines should be an array")
            .iter()
            .map(|line| line.as_str().expect("line should be a string").to_string())
            .collect()
    }
}
