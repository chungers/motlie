//! Example: Structured command execution via `Target::exec()`.
//!
//! Creates a session and runs a command inside the pane, getting back
//! structured `ExecOutput` with stdout and exit code.
//!
//! Usage:
//!   cargo run -p motlie-tmux --example exec_command -- ssh://localhost
//!   cargo run -p motlie-tmux --example exec_command -- ssh://localhost "uname -a"

use motlie_tmux::SshConfig;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());
    let command = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "echo hello_from_exec".to_string());

    let host = SshConfig::parse(&uri)?.connect().await?;
    let session_name = "motlie_example_exec";

    // Clean up leftover
    if let Ok(Some(t)) = host.session(session_name).await {
        let _ = t.kill().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Create session and let shell initialize
    let target = host.create_session(session_name, None, None).await?;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Execute command with 10s timeout
    println!("Executing: {}", command);
    let result = target.exec(&command, Duration::from_secs(10)).await?;

    println!("Exit code: {}", result.exit_code);
    println!("Success:   {}", result.success());
    println!("Stdout:");
    println!("{}", result.stdout);

    // Cleanup
    target.kill().await?;
    Ok(())
}
