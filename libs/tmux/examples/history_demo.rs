//! Demo: rolling transcript/history for an external LLM-style agent loop.
//!
//! In **simulated** mode (default), creates a 2-pane tmux session where each
//! pane simulates a different agent chat trace.
//!
//! In **live** mode (two session names given), watches two existing tmux
//! sessions and builds a combined rolling history from their real output.
//!
//! Usage:
//!   history_demo [ssh://host] [--chars N] [--entries N]
//!   history_demo [ssh://host] SESSION_A SESSION_B [--chars N] [--entries N]
//!   history_demo 'ssh://host?identity-file=/path/to/key'

use anyhow::{anyhow, Result};
use motlie_tmux::{
    has_visible_text, pane_tail_excerpt, CreateSessionOptions, HistoryOptions, KeySequence,
    LabelFormat, PollHistory, SinkFilter, SplitPaneOptions, SshConfig,
};
use std::collections::HashMap;
use std::time::Duration;

const CHAT_CMD: &str = "sh -c 'stty -echo 2>/dev/null || true; cat'";
const HELP: &str = "\
history_demo — rolling transcript for an external LLM/classifier loop

USAGE:
    history_demo [ssh://host] [--chars N] [--entries N]
    history_demo [ssh://host] SESSION_A SESSION_B [--mode monitor|render|tail] [--chars N] [--entries N]

ARGS:
    [ssh://host]       Optional SSH URI [default: ssh://localhost]
                       Supports ?identity-file=/path/to/key
    SESSION_A SESSION_B  Two existing tmux session names to monitor live
                         (omit to run in simulated mode)

OPTIONS:
    --chars N      Max rendered characters kept in rolling context [default: 420]
    --entries N    Max logical history entries kept [default: 8]
    --mode MODE    Live capture mode for existing sessions [default: tail]
    -h, --help     Print this help

WHAT IT SHOWS:
    Simulated mode (no session args):
    - two panes simulating two other agent chat traces
    - one OutputBus subscription filtered to the session
    - one HistoryHandle building rolling context
    - render_text() snapshots after each turn

    Live mode (two session args):
    - monitor mode uses OutputBus + HistoryHandle on live sessions
    - polling modes use capture-based snapshots for live sessions
    - combined rolling history from both sessions
    - Ctrl-C to stop and print final snapshot

EXAMPLES:
    history_demo
    history_demo ssh://localhost --chars 520
    history_demo ssh://localhost agent_a agent_b --mode tail
    history_demo 'ssh://deploy@prod?identity-file=/path/to/key' sess1 sess2";

#[derive(Clone, Copy)]
enum LiveMode {
    Monitor,
    Render,
    Tail,
}

struct Args {
    uri: String,
    max_chars: usize,
    max_entries: usize,
    live_sessions: Option<(String, String)>,
    live_mode: LiveMode,
}

fn parse_live_mode(value: &str) -> Result<LiveMode> {
    match value {
        "monitor" => Ok(LiveMode::Monitor),
        "render" => Ok(LiveMode::Render),
        "tail" => Ok(LiveMode::Tail),
        other => Err(anyhow!("unknown mode '{}' (monitor|render|tail)", other)),
    }
}

fn live_mode_name(mode: LiveMode) -> &'static str {
    match mode {
        LiveMode::Monitor => "monitor",
        LiveMode::Render => "render",
        LiveMode::Tail => "tail",
    }
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
    let mut live_mode = LiveMode::Tail;
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
            "--mode" => {
                i += 1;
                live_mode = parse_live_mode(
                    argv.get(i)
                        .ok_or_else(|| anyhow!("--mode requires a value"))?,
                )?;
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
        live_mode,
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
    match args.live_mode {
        LiveMode::Monitor => run_live_monitor(host, session_a, session_b, args).await,
        LiveMode::Tail => {
            let mut history = PollHistory::new(args.max_entries, args.max_chars);
            let target_a = host
                .session(session_a)
                .await?
                .ok_or_else(|| anyhow!("session '{}' not found", session_a))?;
            let target_b = host
                .session(session_b)
                .await?
                .ok_or_else(|| anyhow!("session '{}' not found", session_b))?;
            println!(
                "Polling live sessions: {} and {} [mode={}]",
                session_a,
                session_b,
                live_mode_name(args.live_mode)
            );
            println!(
                "History window: max_entries={}, max_render_chars={}",
                args.max_entries, args.max_chars
            );
            println!("Baseline captured at startup; only new changes are appended.");
            println!("Ctrl-C to stop.\n");
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            let mut tick = 0u64;
            let mut last_rendered = String::new();
            let mut previous_a = target_a.capture_all().await?;
            let mut previous_b = target_b.capture_all().await?;

            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => break,
                    _ = interval.tick() => {
                        tick += 1;

                        let current_a = target_a.capture_all().await?;
                        if let Some(rendered) = tail_history_entry(session_a, &mut previous_a, &current_a) {
                            history.push_text(rendered);
                        }
                        let current_b = target_b.capture_all().await?;
                        if let Some(rendered) = tail_history_entry(session_b, &mut previous_b, &current_b) {
                            history.push_text(rendered);
                        }

                        let rendered = history.render_text();
                        if !rendered.is_empty() && rendered != last_rendered {
                            println!("=== rolling context (t={}s) ===", tick);
                            println!("{}", rendered);
                            last_rendered = rendered;
                        }
                    }
                }
            }
            println!(
                "\nFinal snapshot: entries={}, omitted_entries={}, rendered_chars={}",
                history.len(),
                history.omitted_entries(),
                history.rendered_chars()
            );
            Ok(())
        }
        LiveMode::Render => {
            let mut history = PollHistory::new(args.max_entries, args.max_chars);
            let target_a = host
                .session(session_a)
                .await?
                .ok_or_else(|| anyhow!("session '{}' not found", session_a))?;
            let target_b = host
                .session(session_b)
                .await?
                .ok_or_else(|| anyhow!("session '{}' not found", session_b))?;
            println!(
                "Polling live sessions: {} and {} [mode={}]",
                session_a,
                session_b,
                live_mode_name(args.live_mode)
            );
            println!(
                "History window: max_entries={}, max_render_chars={}",
                args.max_entries, args.max_chars
            );
            println!("Baseline captured at startup; only new changes are appended.");
            println!("Ctrl-C to stop.\n");
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            let mut tick = 0u64;
            let mut last_rendered = String::new();
            let mut previous_a = target_a.capture_all().await?;
            let mut previous_b = target_b.capture_all().await?;

            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => break,
                    _ = interval.tick() => {
                        tick += 1;

                        let current_a = target_a.capture_all().await?;
                        if current_a != previous_a {
                            if let Some(rendered) = render_history_entry(session_a, &current_a) {
                                history.push_text(rendered);
                            }
                            previous_a = current_a;
                        }

                        let current_b = target_b.capture_all().await?;
                        if current_b != previous_b {
                            if let Some(rendered) = render_history_entry(session_b, &current_b) {
                                history.push_text(rendered);
                            }
                            previous_b = current_b;
                        }

                        let rendered = history.render_text();
                        if !rendered.is_empty() && rendered != last_rendered {
                            println!("=== rolling context (t={}s) ===", tick);
                            println!("{}", rendered);
                            last_rendered = rendered;
                        }
                    }
                }
            }
            println!(
                "\nFinal snapshot: entries={}, omitted_entries={}, rendered_chars={}",
                history.len(),
                history.omitted_entries(),
                history.rendered_chars()
            );
            Ok(())
        }
    }
}

async fn run_live_monitor(
    host: &motlie_tmux::HostHandle,
    session_a: &str,
    session_b: &str,
    args: &Args,
) -> Result<()> {
    let bus = host.output_bus();
    let sub = bus.subscribe(
        vec![
            SinkFilter::for_session(session_a),
            SinkFilter::for_session(session_b),
        ],
        64,
    )?;
    let history = sub.history(HistoryOptions {
        max_entries: args.max_entries,
        max_render_chars: args.max_chars,
        label_format: LabelFormat::Prompt,
        include_omission_marker: true,
        ..Default::default()
    });
    let monitor_a = host.start_monitoring_session(session_a).await?;
    let monitor_b = host.start_monitoring_session(session_b).await?;

    println!(
        "Monitoring live sessions: {} and {} [mode=monitor]",
        session_a, session_b
    );
    println!(
        "History window: max_entries={}, max_render_chars={}",
        args.max_entries, args.max_chars
    );
    println!("Ctrl-C to stop.\n");

    let mut interval = tokio::time::interval(Duration::from_secs(1));
    let mut tick = 0u64;
    let mut last_rendered = String::new();
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = interval.tick() => {
                tick += 1;
                let rendered = history.render_text().await.replace('\r', "");
                if !rendered.is_empty() && rendered != last_rendered {
                    println!("=== rolling context (t={}s) ===", tick);
                    println!("{}", rendered);
                    last_rendered = rendered;
                }
            }
        }
    }

    monitor_a.shutdown().await?;
    monitor_b.shutdown().await?;
    bus.unsubscribe(history.id())?;
    let snapshot = history.join().await?;
    println!(
        "\nFinal snapshot: entries={}, omitted_entries={}, rendered_chars={}",
        snapshot.entries.len(),
        snapshot.omitted_entries,
        snapshot.rendered_chars
    );
    Ok(())
}

fn tail_history_entry(
    session_name: &str,
    previous: &mut HashMap<motlie_tmux::PaneAddress, String>,
    current: &HashMap<motlie_tmux::PaneAddress, String>,
) -> Option<String> {
    if current.is_empty() || current == previous {
        return None;
    }

    let mut pane_list: Vec<_> = current.iter().collect();
    pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

    let mut rendered = String::new();
    for (addr, content) in pane_list {
        let previous_content = previous.get(addr).map(String::as_str).unwrap_or_default();
        let previous_excerpt = pane_tail_excerpt(previous_content, 6);
        let current_excerpt = pane_tail_excerpt(content, 6);
        if current_excerpt.is_empty() || current_excerpt == previous_excerpt {
            continue;
        }
        rendered.push_str(&format!(
            "{}({})> {}\n",
            session_name,
            addr.pane_id,
            current_excerpt.trim_end()
        ));
    }

    *previous = current.clone();
    if rendered.is_empty() {
        None
    } else {
        Some(rendered)
    }
}

fn render_history_entry(
    session_name: &str,
    panes: &HashMap<motlie_tmux::PaneAddress, String>,
) -> Option<String> {
    let mut pane_list: Vec<_> = panes.iter().collect();
    pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

    let mut rendered = String::new();
    for (addr, content) in pane_list {
        if !has_visible_text(content) {
            continue;
        }
        rendered.push_str(&format!("--- {}({}) ---\n", session_name, addr.pane_id));
        rendered.push_str(content);
        if !content.ends_with('\n') {
            rendered.push('\n');
        }
    }

    if rendered.is_empty() {
        None
    } else {
        Some(rendered)
    }
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
        ..Default::default()
    });

    let monitor = host.start_monitoring_session(&session).await?;

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
        (
            pane_a,
            "agent-a> I found the failing assertion in monitor.rs.",
        ),
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

    // Shutdown order: stop monitor → unsubscribe → join (drains and snapshots).
    monitor.shutdown().await?;
    bus.unsubscribe(history.id())?;

    let snapshot = history.join().await?;
    println!(
        "Final snapshot: entries={}, omitted_entries={}, rendered_chars={}",
        snapshot.entries.len(),
        snapshot.omitted_entries,
        snapshot.rendered_chars
    );

    if let Ok(Some(t)) = host.session(&session).await {
        let _ = t.kill().await;
    }

    Ok(())
}
