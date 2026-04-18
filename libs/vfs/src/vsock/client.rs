//! VsockClientTransport: guest-side request/response transport over an established stream.

use anyhow::Result;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::Mutex;

use super::{read_msg, write_msg};
use crate::core::op::{FsOp, FsResult};

/// Guest side: request/response transport over an established stream.
/// The tag is already established by the VMM handshake before this transport is created.
pub struct VsockClientTransport<S> {
    stream: Mutex<S>,
    tag: String,
}

impl<S> VsockClientTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    pub fn new(stream: S, tag: &str) -> Self {
        Self {
            stream: Mutex::new(stream),
            tag: tag.to_string(),
        }
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Send an FsOp and receive the FsResult.
    pub async fn request(&self, op: &FsOp) -> Result<FsResult> {
        let encoded = bincode::serde::encode_to_vec(op, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("bincode encode error: {e}"))?;

        let mut stream = self.stream.lock().await;
        write_msg(&mut *stream, &encoded).await?;
        let resp_buf = read_msg(&mut *stream).await?;
        drop(stream);

        let result: FsResult = bincode::serde::decode_from_slice(&resp_buf, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| anyhow::anyhow!("bincode decode error: {e}"))?;

        Ok(result)
    }
}
