//! Demo: rolling transcript/history for an external LLM-style agent loop.
//!
//! Creates a 2-pane tmux session where each pane simulates a different agent
//! chat trace. The example subscribes to the session output, builds a
//! `HistoryHandle`, and prints the exact rolling context window an external
//! reasoning agent would consume after each turn.
//!
//! Usage:
//!   history_demo [ssh://host] [--chars N] [--entries N]
//!   history_demo 'ssh://host?identity-file=/path/to/key'

use anyhow::{anyhow, Result};
use motlie_tmux::{
    CreateSessionOptions, HistoryOptions, KeySequence, LabelFormat, SinkFilter, SplitPaneOptions,
    SshConfig,
};
use std::time::Duration;

const CHAT_CMD: &str = "sh -c 'stty -echo 2>/dev/null || true; cat'";
const HELP: &str = "\
history_demo — rolling transcript for an external LLM/classifier loop

USAGE:
    history_demo [ssh://host] [--chars N] [--entries N]

ARGS:
    [ssh://host]   Optional SSH URI [default: ssh://localhost]
                   Supports ?identity-file=/path/to/key

OPTIONS:
    --chars N      Max rendered characters kept in rolling context [default: 420]
    --entries N    Max logical history entries kept [default: 8]
    -h, --help     Print this help

WHAT IT SHOWS:
    - two panes simulating two other agent chat traces
    - one OutputBus subscription filtered to the session
    - one HistoryHandle building rolling context
    - render_text() snapshots after each turn, exactly what an external
      reasoning agent would send to its model

EXAMPLES:
    history_demo
    history_demo ssh://localhost --chars 520
    history_demo 'ssh://deploy@prod?identity-file=/path/to/key'";

struct Args {
    uri: String,
    max_chars: usize,
    max_entries: usize,
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", HELP);
        std::process::exit(0);
    }

    let mut uri = "ssh://localhost".to_string();
    let mut max_chars = 420usize;
    let mut max_entries = 8usize;
    let mut i = 1usize;

    while i < argv.len() {
        match argv[i].as_str() {
            "--chars" => {
                i += 1;
                max_chars = argv
                    .get(i)
                    .ok_or_else(|| anyhow!("--chars requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--chars must be a positive integer"))?;
                if max_chars == 0 {
                    return Err(anyhow!("--chars must be > 0"));
                }
            }
            "--entries" => {
                i += 1;
                max_entries = argv
                    .get(i)
                    .ok_or_else(|| anyhow!("--entries requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--entries must be a positive integer"))?;
                if max_entries == 0 {
                    return Err(anyhow!("--entries must be > 0"));
                }
            }
            value if value.starts_with("ssh://") => {
                uri = value.to_string();
            }
            other => {
                return Err(anyhow!(
                    "unknown argument '{}'\n\nTry -h for detailed help.",
                    other
                ));
            }
        }
        i += 1;
    }

    Ok(Args {
        uri,
        max_chars,
        max_entries,
    })
}

async fn send_line(target: &motlie_tmux::Target, text: &str) -> Result<()> {
    target.send_text(text).await?;
    target.send_keys(&KeySequence::parse("{Enter}")?).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let session = format!("history_demo_{}", std::process::id());
    let host = SshConfig::parse(&args.uri)?.connect().await?;

    if let Ok(Some(existing)) = host.session(&session).await {
        let _ = existing.kill().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let create_opts = CreateSessionOptions {
        command: Some(CHAT_CMD.to_string()),
        ..Default::default()
    };
    let session_target = host.create_session(&session, &create_opts).await?;
    tokio::time::sleep(Duration::from_millis(300)).await;

    let windows = session_target.children().await?;
    let window = &windows[0];
    window
        .split_pane(&SplitPaneOptions {
            command: Some(CHAT_CMD.to_string()),
            ..Default::default()
        })
        .await?;
    tokio::time::sleep(Duration::from_millis(300)).await;

    let panes = window.children().await?;
    let pane_a = &panes[0];
    let pane_b = &panes[1];

    let monitor = host.start_monitoring_session(&session).await?;
    tokio::time::sleep(Duration::from_millis(400)).await;

    let bus = host.output_bus();
    let sub = bus.subscribe(vec![SinkFilter::for_session(&session)], 64)?;
    let history = sub.history(HistoryOptions {
        max_entries: args.max_entries,
        max_render_chars: args.max_chars,
        label_format: LabelFormat::Prompt,
        include_omission_marker: true,
    });

    println!("Session: {}", session);
    println!(
        "Simulating two chat traces: {} and {}",
        pane_a.target_string(),
        pane_b.target_string()
    );
    println!(
        "History window: max_entries={}, max_render_chars={}",
        args.max_entries, args.max_chars
    );
    println!();

    let turns = [
        (pane_a, "agent-a> I found the failing assertion in monitor.rs."),
        (
            pane_b,
            "agent-b> Verify the shared OutputBus is injected before monitoring starts.",
        ),
        (
            pane_a,
            "agent-a> Fleet::register now injects the bus and rejects alias mismatch.",
        ),
        (
            pane_b,
            "agent-b> Good. Check custom label budgeting in HistoryHandle.",
        ),
        (
            pane_a,
            "agent-a> rendered_chars now measures the fully rendered line.",
        ),
        (
            pane_b,
            "agent-b> Great. Update DESIGN and API to match the shipped contract.",
        ),
    ];

    for (idx, (pane, line)) in turns.iter().enumerate() {
        send_line(pane, line).await?;
        tokio::time::sleep(Duration::from_millis(250)).await;

        println!("=== rolling context after turn {} ===", idx + 1);
        // Normalize CRLF-ish terminal echo artifacts so the tutorial output stays
        // readable across platforms while still using the real HistoryHandle API.
        println!("{}", history.render_text().await.replace('\r', ""));
    }

    let snapshot = history.snapshot().await;
    println!(
        "Final snapshot: entries={}, omitted_entries={}, rendered_chars={}",
        snapshot.entries.len(),
        snapshot.omitted_entries,
        snapshot.rendered_chars
    );

    monitor.shutdown().await?;
    bus.unsubscribe(history.id())?;
    history.join().await?;

    if let Ok(Some(t)) = host.session(&session).await {
        let _ = t.kill().await;
    }

    Ok(())
}
