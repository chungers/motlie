//! vsock composite: bincode-over-stream transport for VM guest ↔ host.
//!
//! Wire format: `[u32 big-endian length][bincode(FsOp or FsResult)]`.
//! The request/response protocol itself is handshake-free; higher-level
//! bootstraps may optionally exchange a small tag-binding prelude before the
//! framed FsOp/FsResult traffic starts.

pub mod handler;
pub mod client;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const MAX_MSG_SIZE: u32 = 64 * 1024 * 1024; // 64 MiB
const MAX_TAG_HANDSHAKE_SIZE: usize = 4096;

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

/// Write a simple text handshake that binds the stream to one mount tag.
pub async fn write_tag_handshake<W: AsyncWrite + Unpin>(
    writer: &mut W,
    tag: &str,
) -> anyhow::Result<()> {
    if tag.is_empty() {
        anyhow::bail!("tag handshake requires a non-empty tag");
    }
    if tag.contains('\n') || tag.contains('\r') {
        anyhow::bail!("tag handshake may not contain newlines");
    }
    writer.write_all(format!("TAG {tag}\n").as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

/// Read a simple text handshake of the form `TAG <name>\n`.
pub async fn read_tag_handshake<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> anyhow::Result<String> {
    let mut buf = Vec::new();
    loop {
        if buf.len() >= MAX_TAG_HANDSHAKE_SIZE {
            anyhow::bail!("tag handshake too large");
        }
        let byte = reader.read_u8().await?;
        if byte == b'\n' {
            break;
        }
        buf.push(byte);
    }

    if let Some(b'\r') = buf.last().copied() {
        buf.pop();
    }

    let line = String::from_utf8(buf)
        .map_err(|e| anyhow::anyhow!("invalid utf-8 in tag handshake: {e}"))?;
    let Some(tag) = line.strip_prefix("TAG ") else {
        anyhow::bail!("invalid tag handshake prefix");
    };
    if tag.is_empty() {
        anyhow::bail!("tag handshake requires a non-empty tag");
    }
    Ok(tag.to_string())
}

#[cfg(test)]
mod tests {
    use super::{read_tag_handshake, write_tag_handshake};

    #[tokio::test]
    async fn tag_handshake_round_trip() {
        let (mut client, mut server) = tokio::io::duplex(128);

        let writer = tokio::spawn(async move {
            write_tag_handshake(&mut client, "alice-home").await.unwrap();
        });

        let tag = read_tag_handshake(&mut server).await.unwrap();
        writer.await.unwrap();
        assert_eq!(tag, "alice-home");
    }
}
