use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use motlie_vmm::ssh::VSOCK_SSH_PORT;
use russh::client;
use russh::{ChannelMsg, Pty};
use tokio::time::timeout;

fn default_pty_modes() -> Vec<(Pty, u32)> {
    vec![
        (Pty::VINTR, 3),
        (Pty::VQUIT, 28),
        (Pty::VERASE, 127),
        (Pty::VKILL, 21),
        (Pty::VEOF, 4),
        (Pty::VWERASE, 23),
        (Pty::VLNEXT, 22),
        (Pty::VREPRINT, 18),
        (Pty::VSUSP, 26),
        (Pty::ICRNL, 1),
        (Pty::IXON, 1),
        (Pty::ISIG, 1),
        (Pty::ICANON, 1),
        (Pty::IEXTEN, 1),
        (Pty::ECHO, 1),
        (Pty::ECHOE, 1),
        (Pty::ECHOK, 1),
        (Pty::ECHOCTL, 1),
        (Pty::ECHOKE, 1),
        (Pty::OPOST, 1),
        (Pty::ONLCR, 1),
        (Pty::TTY_OP_ISPEED, 38_400),
        (Pty::TTY_OP_OSPEED, 38_400),
    ]
}

struct ProbeClient;

impl client::Handler for ProbeClient {
    type Error = anyhow::Error;

    fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::PublicKey,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        async { Ok(true) }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let host = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "192.168.249.2:22".to_string());
    let command = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/bin/cat -v".to_string());
    let want_reply = std::env::args()
        .nth(3)
        .map(|s| s == "true" || s == "1" || s == "yes")
        .unwrap_or(false);

    let config = Arc::new(client::Config::default());
    let mut handle = client::connect(config, host, ProbeClient)
        .await
        .context("connect")?;

    let auth = handle
        .authenticate_none("alice")
        .await
        .context("authenticate_none")?;
    let auth = if auth.success() {
        auth
    } else {
        handle
            .authenticate_password("alice", "testpass")
            .await
            .context("authenticate_password")?
    };
    if !auth.success() {
        return Err(anyhow!("password auth failed"));
    }

    let mut channel = handle.channel_open_session().await.context("open session")?;
    let modes = default_pty_modes();
    channel
        .request_pty(want_reply, "xterm-256color", 80, 24, 0, 0, &modes)
        .await
        .context("request_pty")?;
    channel.exec(want_reply, command).await.context("exec")?;
    channel
        .data(&b"abc\r\x04"[..])
        .await
        .context("data")?;

    let mut stdout = Vec::new();
    loop {
        let msg = timeout(Duration::from_secs(3), channel.wait()).await;
        match msg {
            Ok(Some(ChannelMsg::Data { ref data })) => stdout.extend_from_slice(data),
            Ok(Some(ChannelMsg::ExtendedData { ref data, ext })) if ext == 1 => {
                eprintln!("stderr={:?}", String::from_utf8_lossy(data))
            }
            Ok(Some(ChannelMsg::ExitStatus { exit_status })) => {
                eprintln!("exit_status={exit_status}");
                break;
            }
            Ok(Some(ChannelMsg::Eof | ChannelMsg::Close)) => break,
            Ok(Some(other)) => eprintln!("other={other:?}"),
            Ok(None) => break,
            Err(_) => {
                eprintln!("timeout waiting for channel output");
                break;
            }
        }
    }

    println!("{}", String::from_utf8_lossy(&stdout));
    let _ = handle
        .disconnect(russh::Disconnect::ByApplication, "", "English")
        .await;
    let _ = VSOCK_SSH_PORT;
    Ok(())
}
