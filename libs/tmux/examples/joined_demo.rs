//! Demo: JoinedStream merging output from two panes.
//!
//! Creates a 2-pane tmux session, sends different commands to each pane,
//! and shows the interleaved JoinedStream output with source labels.
//!
//! Usage:
//!   joined_demo [--format bracketed|prompt|separator]

use anyhow::{anyhow, Result};
use motlie_tmux::{strip_ansi, LabelFormat, SinkFilter, SshConfig};
use std::time::Duration;

#[derive(Clone, Copy)]
enum Mode {
    Bracketed,
    Prompt,
    Separator,
}

fn parse_args() -> Result<Mode> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("joined_demo [--format bracketed|prompt|separator]");
        std::process::exit(0);
    }
    let mut mode = Mode::Bracketed;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--format" {
            i += 1;
            mode = match args.get(i).map(|s| s.as_str()) {
                Some("bracketed") => Mode::Bracketed,
                Some("prompt") => Mode::Prompt,
                Some("separator") => Mode::Separator,
                other => return Err(anyhow!("unknown format: {:?}", other)),
            };
        }
        i += 1;
    }
    Ok(mode)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = parse_args()?;
    let session = format!("joined_demo_{}", std::process::id());

    let host = SshConfig::parse("ssh://localhost")?.connect().await?;

    // Clean up any leftover from a previous failed run
    if let Ok(Some(t)) = host.session(&session).await {
        let _ = t.kill().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Create a session (gives us pane 0)
    let _target = host
        .create_session(&session, &Default::default())
        .await?;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Split the window to create a second pane.
    // TODO: Replace with HostHandle::split_window() API when available.
    // This shells out to local tmux, so this example only works for localhost.
    let status = std::process::Command::new("tmux")
        .args(["split-window", "-t", &session, "-v"])
        .status()?;
    if !status.success() {
        return Err(anyhow!("tmux split-window failed"));
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Start monitoring the session
    let monitor = host.start_monitoring_session(&session).await?;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Subscribe and create JoinedStream
    let bus = host.output_bus();
    let sub = bus.subscribe(vec![SinkFilter::for_session(&session)], 64)?;

    let label_format = match mode {
        Mode::Bracketed => LabelFormat::Bracketed,
        Mode::Prompt => LabelFormat::Prompt,
        Mode::Separator => LabelFormat::Custom(|_source, _content| String::new()),
    };
    let mut stream = sub.joined(label_format);

    // Pane targets after create_session + split-window
    let pane0 = format!("{}:0.0", session);
    let pane1 = format!("{}:0.1", session);

    eprintln!("Session '{}' — panes: {}, {}", session, pane0, pane1);
    eprintln!("Monitor active: {}, subscribers: {}", monitor.is_active(), bus.subscriber_count());

    // Send commands to both panes with a stagger
    let tmux_send = |target: &str, cmd: &str| {
        std::process::Command::new("tmux")
            .args(["send-keys", "-t", target, cmd, "Enter"])
            .output()
    };

    tmux_send(&pane0, "ps aux | head -5")?;
    tokio::time::sleep(Duration::from_millis(500)).await;
    tmux_send(&pane1, "ls -la /tmp | head -5")?;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Collect output
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let use_separator = matches!(mode, Mode::Separator);
    let mut last_source = String::new();

    println!("--- JoinedStream output ---");
    while tokio::time::Instant::now() < deadline {
        tokio::select! {
            chunk = stream.next() => {
                match chunk {
                    Some(c) => {
                        let clean = strip_ansi(&c.output.content);
                        if clean.trim().is_empty() {
                            continue;
                        }
                        if use_separator {
                            let src = c.source.short();
                            if c.source_changed && !last_source.is_empty() {
                                println!("--- {} ---", src);
                            } else if last_source.is_empty() {
                                println!("--- {} ---", src);
                            }
                            last_source = src;
                            print!("{}", clean);
                        } else {
                            let label = c.source.short();
                            print!("[{}] {}", label, clean);
                        }
                    }
                    None => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => continue,
        }
    }
    println!("--- end ---");

    // Cleanup
    monitor.shutdown().await?;
    if let Ok(Some(t)) = host.session(&session).await {
        let _ = t.kill().await;
    }
    eprintln!("Done.");
    Ok(())
}
