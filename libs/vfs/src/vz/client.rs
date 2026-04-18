use anyhow::Result;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::Mutex;

use super::{read_msg, write_msg};
use crate::core::op::{FsOp, FsResult};

/// Guest-side Vz PoC transport over an established TCP stream.
pub struct TcpClientTransport<S> {
    stream: Mutex<S>,
    tag: String,
}

impl<S> TcpClientTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    pub fn new(stream: S, tag: &str) -> Self {
        Self {
            stream: Mutex::new(stream),
            tag: tag.to_string(),
        }
    }

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

    pub fn tag(&self) -> &str {
        &self.tag
    }
}
