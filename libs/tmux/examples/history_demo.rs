//! Demo: rolling transcript/history for an external LLM-style agent loop.
//!
//! In **simulated** mode (default), creates a 2-pane tmux session where each
//! pane simulates a different agent chat trace.
//!
//! In **live** mode (two session names given), monitors two existing tmux
//! sessions and builds a combined rolling history from their real output.
//!
//! Usage:
//!   history_demo [ssh://host] [--chars N] [--entries N]
//!   history_demo [ssh://host] SESSION_A SESSION_B [--chars N] [--entries N]
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
    history_demo [ssh://host] SESSION_A SESSION_B [--chars N] [--entries N]

ARGS:
    [ssh://host]       Optional SSH URI [default: ssh://localhost]
                       Supports ?identity-file=/path/to/key
    SESSION_A SESSION_B  Two existing tmux session names to monitor live
                         (omit to run in simulated mode)

OPTIONS:
    --chars N      Max rendered characters kept in rolling context [default: 420]
    --entries N    Max logical history entries kept [default: 8]
    -h, --help     Print this help

WHAT IT SHOWS:
    Simulated mode (no session args):
    - two panes simulating two other agent chat traces
    - one OutputBus subscription filtered to the session
    - one HistoryHandle building rolling context
    - render_text() snapshots after each turn

    Live mode (two session args):
    - monitors two existing tmux sessions in real time
    - combined rolling history from both sessions
    - Ctrl-C to stop and print final snapshot

EXAMPLES:
    history_demo
    history_demo ssh://localhost --chars 520
    history_demo ssh://localhost agent_a agent_b
    history_demo 'ssh://deploy@prod?identity-file=/path/to/key' sess1 sess2";

struct Args {
    uri: String,
    max_chars: usize,
    max_entries: usize,
    live_sessions: Option<(String, String)>,
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
    let mut positional: Vec<String> = Vec::new();
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
            value if value.starts_with('-') => {
                return Err(anyhow!(
                    "unknown option '{}'\n\nTry -h for detailed help.",
                    value
                ));
            }
            _ => {
                positional.push(argv[i].clone());
            }
        }
        i += 1;
    }

    let live_sessions = match positional.len() {
        0 => None,
        2 => Some((positional[0].clone(), positional[1].clone())),
        _ => {
            return Err(anyhow!(
                "expected 0 or 2 session names, got {}\n\nTry -h for detailed help.",
                positional.len()
            ));
        }
    };

    Ok(Args {
        uri,
        max_chars,
        max_entries,
        live_sessions,
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
    let host = SshConfig::parse(&args.uri)?.connect().await?;

    match args.live_sessions {
        Some((ref a, ref b)) => run_live(&host, a, b, &args).await,
        None => run_simulated(&host, &args).await,
    }
}

/// Live mode: monitor two existing tmux sessions and build combined history.
async fn run_live(
    host: &motlie_tmux::HostHandle,
    session_a: &str,
    session_b: &str,
    args: &Args,
) -> Result<()> {
    // Subscribe BEFORE starting monitors to avoid the initial-output race
    let bus = host.output_bus();
    let filters = vec![
        SinkFilter::for_session(session_a),
        SinkFilter::for_session(session_b),
    ];
    let sub = bus.subscribe(filters, 64)?;
    let history = sub.history(HistoryOptions {
        max_entries: args.max_entries,
        max_render_chars: args.max_chars,
        label_format: LabelFormat::Prompt,
        include_omission_marker: true,
    });

    let monitor_a = host.start_monitoring_session(session_a).await?;
    let monitor_b = host.start_monitoring_session(session_b).await?;

    println!(
        "Monitoring live sessions: {} and {}",
        session_a, session_b
    );
    println!(
        "History window: max_entries={}, max_render_chars={}",
        args.max_entries, args.max_chars
    );
    println!("Ctrl-C to stop.\n");

    let mut interval = tokio::time::interval(Duration::from_secs(1));
    let mut tick = 0u64;
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = interval.tick() => {
                tick += 1;
                let rendered = history.render_text().await;
                if !rendered.is_empty() {
                    println!("=== rolling context (t={}s) ===", tick);
                    println!("{}", rendered.replace('\r', ""));
                }
            }
        }
    }

    // Shut down monitors before taking the final snapshot so no trailing
    // output arrives after the snapshot is captured.
    monitor_a.shutdown().await?;
    monitor_b.shutdown().await?;

    let snapshot = history.snapshot().await;
    println!(
        "\nFinal snapshot: entries={}, omitted_entries={}, rendered_chars={}",
        snapshot.entries.len(),
        snapshot.omitted_entries,
        snapshot.rendered_chars
    );

    bus.unsubscribe(history.id())?;
    history.join().await?;

    Ok(())
}

/// Simulated mode: create a temporary 2-pane session and replay scripted turns.
async fn run_simulated(host: &motlie_tmux::HostHandle, args: &Args) -> Result<()> {
    let session = format!("history_demo_{}", std::process::id());

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

    // Subscribe BEFORE starting the monitor to avoid the initial-output race
    let bus = host.output_bus();
    let sub = bus.subscribe(vec![SinkFilter::for_session(&session)], 64)?;
    let history = sub.history(HistoryOptions {
        max_entries: args.max_entries,
        max_render_chars: args.max_chars,
        label_format: LabelFormat::Prompt,
        include_omission_marker: true,
    });

    let monitor = host.start_monitoring_session(&session).await?;
    // Give the monitor task time to attach control mode before sending turns
    tokio::time::sleep(Duration::from_millis(400)).await;

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
