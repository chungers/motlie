//! vsock composite: bincode-over-stream transport for VM guest ↔ host.
//!
//! Wire format: `[u32 big-endian length][bincode(FsOp or FsResult)]`.
//! No Frame wrapper, no Codec trait, no handshake — the VMM multiplexer
//! handles tag binding before handing off the stream.

pub mod handler;
pub mod client;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const MAX_MSG_SIZE: u32 = 64 * 1024 * 1024; // 64 MiB

/// Write a length-prefixed bincode message.
pub(crate) async fn write_msg<W: AsyncWrite + Unpin>(
    writer: &mut W,
    payload: &[u8],
) -> anyhow::Result<()> {
    let len = payload.len() as u32;
    writer.write_all(&len.to_be_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Read a length-prefixed bincode message.
pub(crate) async fn read_msg<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> anyhow::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_MSG_SIZE {
        anyhow::bail!("message too large: {len} bytes (max {MAX_MSG_SIZE})");
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).await?;
    Ok(buf)
}
