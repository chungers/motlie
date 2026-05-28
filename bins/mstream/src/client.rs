use std::path::Path;

use anyhow::Context;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::protocol::ClientRequest;

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
