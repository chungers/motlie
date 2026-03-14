//! Example: Session create → rename → kill lifecycle.
//!
//! Demonstrates the full session lifecycle including `rename()` returning
//! a new Target handle with the updated address.
//!
//! Usage:
//!   cargo run -p motlie-tmux --example session_lifecycle -- ssh://localhost

use motlie_tmux::SshConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let host = SshConfig::parse(&uri)?.connect().await?;
    let session_name = "motlie_example_lifecycle";

    // Clean up leftover from previous run
    if let Ok(Some(t)) = host.session(session_name).await {
        let _ = t.kill().await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // 1. Create
    println!("Creating session '{}'...", session_name);
    let target = host
        .create_session(session_name, &motlie_tmux::CreateSessionOptions {
            window_name: Some("main".to_string()),
            ..Default::default()
        })
        .await?;
    println!(
        "  Created: target={}, level={:?}",
        target.target_string(),
        target.level()
    );

    // Confirm it exists
    let sessions = host.list_sessions().await?;
    assert!(sessions.iter().any(|s| s.name == session_name));
    println!("  Confirmed in session list.");

    // 2. Rename — rename() returns a NEW Target with updated address
    let new_name = "motlie_example_renamed";
    println!("Renaming to '{}'...", new_name);
    let renamed = target.rename(new_name).await?;
    println!(
        "  Renamed: target={}, session_name={}",
        renamed.target_string(),
        renamed.session_name()
    );

    // Verify old name is gone, new name exists
    let sessions = host.list_sessions().await?;
    assert!(!sessions.iter().any(|s| s.name == session_name));
    assert!(sessions.iter().any(|s| s.name == new_name));
    println!("  Confirmed rename in session list.");

    // 3. Kill — use the returned handle (old handle has stale name)
    println!("Killing session...");
    renamed.kill().await?;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let sessions = host.list_sessions().await?;
    assert!(!sessions.iter().any(|s| s.name == new_name));
    println!("  Confirmed session is gone.");

    println!("Done.");
    Ok(())
}
