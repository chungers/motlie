use std::time::Duration;

use motlie_vmm::orchestrator::VmHandle;
use motlie_vmm::ssh::PtyRequest;

type DynError = Box<dyn std::error::Error + Send + Sync>;

pub async fn run_pty_smoke(handle: &VmHandle) -> Result<(), DynError> {
    let pty = handle
        .open_pty(PtyRequest::default(), Duration::from_secs(10))
        .await?;

    let login = pty
        .read_until_contains("Start tmux session?", Duration::from_secs(20))
        .await?;
    if !login.output.contains("v1.4 extraction / agent-state demo") {
        return Err(format!("expected MOTD in PTY login output, got: {}", login.output).into());
    }

    pty.send_line("n").await?;
    let shell = pty
        .read_until_contains("alice@motlie-alice", Duration::from_secs(10))
        .await?;
    if !shell.output.contains("alice@motlie-alice") {
        return Err(format!("expected login prompt, got: {}", shell.output).into());
    }

    pty.send_line("pwd").await?;
    let pwd = pty
        .read_until_contains("/home/alice", Duration::from_secs(10))
        .await?;
    if !pwd.output.contains("/home/alice") {
        return Err(format!("expected /home/alice in PTY output, got: {}", pwd.output).into());
    }

    pty.resize(120, 40, 0, 0).await?;
    pty.send_line("stty size").await?;
    let size = pty
        .read_until_contains("40 120", Duration::from_secs(10))
        .await?;
    if !size.output.contains("40 120") {
        return Err(format!("expected resized PTY dimensions, got: {}", size.output).into());
    }

    pty.send_line("exit").await?;
    let _ = pty.read_for(Duration::from_secs(5)).await?;
    let transcript = pty.transcript()?;
    if transcript.is_empty() {
        return Err("expected non-empty PTY transcript".into());
    }
    Ok(())
}
