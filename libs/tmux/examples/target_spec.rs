//! Example: Resolve a tmux target via `TargetSpec` string.
//!
//! Parses a target string (e.g. "build", "build:0", "build:0.1") and
//! resolves it against a live tmux server, printing the result.
//!
//! Usage:
//!   cargo run -p motlie-tmux --example target_spec -- ssh://localhost "session_name"
//!   cargo run -p motlie-tmux --example target_spec -- ssh://localhost "session:0"
//!   cargo run -p motlie-tmux --example target_spec -- ssh://localhost "session:0.0"

use motlie_tmux::{SshConfig, TargetLevel, TargetSpec};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());
    let target_str = std::env::args()
        .nth(2)
        .ok_or_else(|| anyhow::anyhow!("usage: target_spec <uri> <target_string>"))?;

    let host = SshConfig::parse(&uri)?.connect().await?;

    // Parse the target string
    let spec = TargetSpec::parse(&target_str)?;
    println!("Parsed TargetSpec: {}", spec);
    println!(
        "  session={}, window={:?}, pane={:?}",
        spec.session_name(),
        spec.window_selector(),
        spec.pane_index()
    );

    // Resolve against the live server
    match host.target(&spec).await? {
        Some(target) => {
            println!("\nResolved target: {}", target.target_string());
            println!("  level: {:?}", target.level());

            match target.level() {
                TargetLevel::Session => {
                    if let Some(info) = target.session_info() {
                        println!(
                            "  Session: name={}, id={}, windows={}, attached={}",
                            info.name, info.id, info.window_count, info.attached
                        );
                    }
                }
                TargetLevel::Window => {
                    if let Some(info) = target.window_info() {
                        println!(
                            "  Window: name={}, index={}, active={}, panes={}",
                            info.name, info.index, info.active, info.pane_count
                        );
                    }
                }
                TargetLevel::Pane => {
                    if let Some(addr) = target.pane_address() {
                        println!(
                            "  Pane: pane_id={}, address={}",
                            addr.pane_id, addr
                        );
                    }
                }
            }
        }
        None => {
            println!("Target '{}' not found.", target_str);
        }
    }

    Ok(())
}
