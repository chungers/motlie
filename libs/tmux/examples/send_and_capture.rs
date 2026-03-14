//! Example: Send input to a pane and capture output.
//!
//! Creates a session, sends a command via `send_text()` + `send_keys()`,
//! waits for output, then captures the pane content.
//!
//! Usage:
//!   cargo run -p motlie-tmux --example send_and_capture -- ssh://localhost

use motlie_tmux::{KeySequence, SshConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let host = SshConfig::parse(&uri)?.connect().await?;
    let session_name = "motlie_example_capture";

    // Clean up leftover
    if let Ok(Some(t)) = host.session(session_name).await {
        let _ = t.kill().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Create session
    let target = host.create_session(session_name, &Default::default()).await?;
    tokio::time::sleep(Duration::from_millis(500)).await; // let shell start

    // Send text (does NOT press Enter)
    println!("Sending command...");
    target.send_text("echo HELLO_FROM_MOTLIE").await?;

    // Press Enter via KeySequence
    let enter = KeySequence::parse("{Enter}")?;
    target.send_keys(&enter).await?;

    // Wait for output
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Capture visible pane content
    let content = target.capture().await?;
    println!("--- Captured pane content ---");
    println!("{}", content);
    println!("--- End ---");

    // Verify
    if content.contains("HELLO_FROM_MOTLIE") {
        println!("Output verified.");
    } else {
        println!("WARNING: expected output not found in capture.");
    }

    // Cleanup
    target.kill().await?;
    println!("Done.");
    Ok(())
}
