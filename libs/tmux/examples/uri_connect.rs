//! Example: Connect to a tmux host via SSH URI.
//!
//! Demonstrates `SshConfig::parse()` and `connect()` with automatic
//! transport selection (localhost → Local, remote → SSH).
//!
//! Usage:
//!   cargo run -p motlie-tmux --example uri_connect -- ssh://localhost
//!   cargo run -p motlie-tmux --example uri_connect -- ssh://deploy@prod:2222

use motlie_tmux::SshConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    println!("Parsing URI: {}", uri);
    let config = SshConfig::parse(&uri)?;
    println!(
        "  host={}, user={}, port={}",
        config.host(),
        config.user(),
        config.port()
    );

    println!("Connecting...");
    let host = config.connect().await?;

    // Verify connection by listing sessions
    let sessions = host.list_sessions().await?;
    println!("Connected. {} active session(s).", sessions.len());

    Ok(())
}
