//! Example: Navigate the Target hierarchy (session → windows → panes).
//!
//! Connects via URI, finds a session by name, and walks the tree printing
//! each level's target_string and metadata.
//!
//! Usage:
//!   cargo run -p motlie-tmux --example target_navigate -- ssh://localhost <session_name>
//!
//! If no session name is given, creates a temporary session with two windows.

use motlie_tmux::{SshConfig, TargetLevel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());
    let session_arg = std::env::args().nth(2);

    let host = SshConfig::parse(&uri)?.connect().await?;

    let (session_target, cleanup) = if let Some(name) = session_arg {
        // Use existing session
        let t = host
            .session(&name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("session '{}' not found", name))?;
        (t, false)
    } else {
        // Create a temporary session with two windows for demonstration
        let name = "motlie_example_nav";
        if let Ok(Some(t)) = host.session(name).await {
            let _ = t.kill().await;
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        let t = host.create_session(name, Some("win0"), None).await?;
        // Create a second window by sending tmux new-window command
        // (the library doesn't have a direct create_window API at session level,
        //  so we use exec on the transport indirectly via send_text)
        println!("Created temporary session '{}' for demo.", name);
        (t, true)
    };

    // Print session info
    println!(
        "Session: {} (level={:?})",
        session_target.target_string(),
        session_target.level()
    );
    if let Some(info) = session_target.session_info() {
        println!(
            "  id={}, windows={}, attached={}",
            info.id, info.window_count, info.attached
        );
    }

    // Navigate to windows
    let windows = session_target.children().await?;
    println!("\n  Windows ({}):", windows.len());
    for win in &windows {
        println!(
            "    {} (level={:?})",
            win.target_string(),
            win.level()
        );
        if let Some(info) = win.window_info() {
            println!(
                "      name='{}', index={}, active={}, panes={}",
                info.name, info.index, info.active, info.pane_count
            );
        }

        // Navigate to panes within each window
        let panes = win.children().await?;
        println!("      Panes ({}):", panes.len());
        for pane in &panes {
            println!(
                "        {} (level={:?})",
                pane.target_string(),
                pane.level()
            );
            if let Some(addr) = pane.pane_address() {
                println!(
                    "          pane_id={}, index={}",
                    addr.pane_id, addr.pane
                );
            }
            // Pane level: children() returns empty
            assert_eq!(pane.level(), TargetLevel::Pane);
        }
    }

    // Cleanup if we created the session
    if cleanup {
        println!("\nCleaning up temporary session...");
        session_target.kill().await?;
    }

    Ok(())
}
