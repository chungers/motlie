//! Example: List tmux sessions on a host.
//!
//! Connects via SSH URI, then prints a table of all active sessions
//! with name, id, window count, and attached status.
//!
//! Usage:
//!   cargo run -p motlie-tmux --example list_sessions -- ssh://localhost
//!   cargo run -p motlie-tmux --example list_sessions -- ssh://deploy@prod
//!   cargo run -p motlie-tmux --example list_sessions -- 'ssh://deploy@prod?identity-file=/path/to/key'

use motlie_tmux::SshConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let host = SshConfig::parse(&uri)?.connect().await?;
    let sessions = host.list_sessions().await?;

    if sessions.is_empty() {
        println!("No active tmux sessions.");
        return Ok(());
    }

    println!(
        "{:<20} {:<8} {:<8} {:<8} {}",
        "NAME", "ID", "WINDOWS", "CLIENTS", "ATTACHED"
    );
    println!("{}", "-".repeat(50));
    for s in &sessions {
        println!(
            "{:<20} {:<8} {:<8} {:<8} {}",
            s.name,
            s.id,
            s.window_count,
            s.attached_count,
            if s.is_attached() { "yes" } else { "no" }
        );
    }

    Ok(())
}
