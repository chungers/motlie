//! VsockConnectionHandler: host-side bincode FsOp/FsResult serving over a stream.

use std::sync::Arc;

use anyhow::Result;
use tokio::io::{AsyncRead, AsyncWrite};

use super::{read_msg, write_msg};
use crate::core::op::FsOp;
use crate::core::server::FsServer;

/// Host side: serves a single connection for a known tag.
/// The tag is established by the VMM multiplexer handshake before this handler sees the stream.
pub struct VsockConnectionHandler {
    server: Arc<FsServer>,
    tag: String,
}

impl VsockConnectionHandler {
    pub fn new(server: Arc<FsServer>, tag: &str) -> Self {
        Self {
            server,
            tag: tag.to_string(),
        }
    }

    /// Serve the connection: read FsOp, call handle_op(), write FsResult.
    /// Loops until the stream closes or an I/O error occurs.
    pub async fn serve<S>(&self, mut stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send,
    {
        let (mut reader, mut writer) = tokio::io::split(&mut stream);
        loop {
            let msg = match read_msg(&mut reader).await {
                Ok(m) => m,
                Err(e) => {
                    // EOF or connection closed — normal shutdown
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

            let result = self.server.handle_op(&self.tag, op);

            let encoded = bincode::serde::encode_to_vec(&result, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("bincode encode error: {e}"))?;

            write_msg(&mut writer, &encoded).await?;
        }
    }
}
