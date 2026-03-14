//! Example: Continuously stream pane content to stdout.
//!
//! Connects via SSH URI, finds a target, and streams new scrollback lines
//! to stdout using overlap-aware deduplication. Like `tail -f` for a tmux pane.
//!
//! Ctrl-C exits cleanly.
//!
//! Usage:
//!   # Stream an existing session's active pane (default: last 50 lines, 200ms poll)
//!   cargo run -p motlie-tmux --example stream_pane -- ssh://localhost my_session
//!
//!   # Stream with custom settings
//!   cargo run -p motlie-tmux --example stream_pane -- ssh://localhost my_session --lines 100 --interval 500
//!
//!   # Stream a specific pane
//!   cargo run -p motlie-tmux --example stream_pane -- ssh://localhost "my_session:0.1" --lines 30

use motlie_tmux::{overlap_deduplicate, ScrollbackQuery, SshConfig, TargetSpec};
use std::io::Write;
use std::time::Duration;

struct Args {
    uri: String,
    target: String,
    lines: usize,
    interval_ms: u64,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!(
            "usage: {} <uri> <target> [--lines N] [--interval MS]",
            args[0]
        ));
    }

    let mut lines = 50usize;
    let mut interval_ms = 200u64;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--lines" => {
                i += 1;
                lines = args
                    .get(i)
                    .ok_or("--lines requires a value")?
                    .parse()
                    .map_err(|_| "--lines must be a positive integer")?;
            }
            "--interval" => {
                i += 1;
                interval_ms = args
                    .get(i)
                    .ok_or("--interval requires a value")?
                    .parse()
                    .map_err(|_| "--interval must be a positive integer")?;
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }

    Ok(Args {
        uri: args[1].clone(),
        target: args[2].clone(),
        lines,
        interval_ms,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // Connect
    let host = SshConfig::parse(&args.uri)?.connect().await?;

    // Resolve target
    let spec = TargetSpec::parse(&args.target)?;
    let target = host
        .target(&spec)
        .await?
        .ok_or_else(|| anyhow::anyhow!("target '{}' not found", args.target))?;

    eprintln!(
        "Streaming {} (last {} lines, {}ms interval). Ctrl-C to stop.",
        target.target_string(),
        args.lines,
        args.interval_ms
    );

    let query = ScrollbackQuery::LastLines(args.lines);
    let interval = Duration::from_millis(args.interval_ms);
    let overlap = 5usize;
    let mut previous = String::new();
    let mut stdout = std::io::stdout().lock();

    // Initial capture — print everything
    let initial = target.sample_text(&query).await?;
    if !initial.is_empty() {
        stdout.write_all(initial.as_bytes())?;
        if !initial.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        stdout.flush()?;
        previous = initial;
    }

    // Poll loop with Ctrl-C handling
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                break;
            }
            _ = tokio::time::sleep(interval) => {
                let current = target.sample_text(&query).await?;
                if current == previous || current.is_empty() {
                    continue;
                }

                // Deduplicate: find new lines not in previous capture
                let (merged, _issues) = overlap_deduplicate(&previous, &current, overlap);

                // Print only the new portion (everything after the previous content)
                if merged.len() > previous.len() {
                    let new_content = &merged[previous.len()..];
                    stdout.write_all(new_content.as_bytes())?;
                    stdout.flush()?;
                } else if merged != previous {
                    // Dedup couldn't merge (resync) — print full current capture
                    stdout.write_all(current.as_bytes())?;
                    if !current.ends_with('\n') {
                        stdout.write_all(b"\n")?;
                    }
                    stdout.flush()?;
                }

                previous = merged;
            }
        }
    }

    eprintln!("\nStopped.");
    Ok(())
}
