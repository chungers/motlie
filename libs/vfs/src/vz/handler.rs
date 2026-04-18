use std::sync::Arc;

use anyhow::Result;
use tokio::io::{AsyncRead, AsyncWrite};

use super::{read_msg, write_msg};
use crate::core::op::FsOp;
use crate::core::server::FsServer;

/// Host-side Vz PoC handler for one known tag over a TCP stream.
pub struct TcpConnectionHandler {
    server: Arc<FsServer>,
    tag: String,
}

impl TcpConnectionHandler {
    pub fn new(server: Arc<FsServer>, tag: &str) -> Self {
        Self {
            server,
            tag: tag.to_string(),
        }
    }

    pub async fn serve<S>(&self, mut stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send,
    {
        let (mut reader, mut writer) = tokio::io::split(&mut stream);
        loop {
            let msg = match read_msg(&mut reader).await {
                Ok(m) => m,
                Err(e) => {
                    if e.downcast_ref::<std::io::Error>()
                        .map_or(false, |io| io.kind() == std::io::ErrorKind::UnexpectedEof)
                    {
                        return Ok(());
                    }
                    return Err(e);
                }
            };

            let op: FsOp = bincode::serde::decode_from_slice(&msg, bincode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| anyhow::anyhow!("bincode decode error: {e}"))?;

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.server.handle_op(&self.tag, op)
            }))
            .map_err(|payload| {
                let reason = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "non-string panic payload".to_string()
                };
                anyhow::anyhow!("server panic while handling tag {}: {reason}", self.tag)
            })?;

            let encoded = bincode::serde::encode_to_vec(&result, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("bincode encode error: {e}"))?;

            write_msg(&mut writer, &encoded).await?;
        }
    }
}
