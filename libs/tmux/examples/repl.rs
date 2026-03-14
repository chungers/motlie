//! Example: Interactive REPL for managing tmux sessions over SSH.
//!
//! Connects to a host via SSH URI, then enters a command loop for
//! creating/killing sessions, listing targets, sending input, and
//! capturing scrollback.
//!
//! Commands:
//!   create <name>            Create a new tmux session
//!   kill <target>            Kill a session/window/pane
//!   targets                  List all sessions with full target spec tree
//!   send <target> <text...>  Send text + Enter to a target
//!   capture <target> <n>     Print last N scrollback lines
//!   quit                     Disconnect and exit
//!
//! Usage:
//!   cargo run -p motlie-tmux --example repl -- ssh://localhost

use motlie_tmux::{KeySequence, ScrollbackQuery, SshConfig, TargetSpec};
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let host = SshConfig::parse(&uri)?.connect().await?;
    println!("Connected to {}", uri);

    let stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let enter = KeySequence::parse("{Enter}")?;

    write!(stdout, "repl> ")?;
    stdout.flush()?;

    for line in stdin.lines() {
        let line = line?;
        let parts: Vec<&str> = line.trim().splitn(3, ' ').collect();
        if parts.is_empty() || parts[0].is_empty() {
            write!(stdout, "repl> ")?;
            stdout.flush()?;
            continue;
        }

        match parts[0] {
            "create" => {
                let name = match parts.get(1) {
                    Some(n) => *n,
                    None => {
                        println!("usage: create <name>");
                        write!(stdout, "repl> ")?;
                        stdout.flush()?;
                        continue;
                    }
                };
                match host.create_session(name, None, None).await {
                    Ok(_) => println!("Created: {}", name),
                    Err(e) => println!("Error: {}", e),
                }
            }

            "kill" => {
                let target_str = match parts.get(1) {
                    Some(t) => *t,
                    None => {
                        println!("usage: kill <target>");
                        write!(stdout, "repl> ")?;
                        stdout.flush()?;
                        continue;
                    }
                };
                match resolve_target(&host, target_str).await {
                    Ok(target) => match target.kill().await {
                        Ok(()) => println!("Killed: {}", target_str),
                        Err(e) => println!("Error: {}", e),
                    },
                    Err(e) => println!("{}", e),
                }
            }

            "targets" => {
                match host.list_sessions().await {
                    Ok(sessions) => {
                        if sessions.is_empty() {
                            println!("  (no sessions)");
                        }
                        for s in &sessions {
                            let spec = TargetSpec::parse(&s.name)?;
                            let target = host.target(&spec).await?;
                            if let Some(t) = target {
                                let windows = t.children().await?;
                                println!(
                                    "  {:<20} (Session, {} window{})",
                                    s.name,
                                    windows.len(),
                                    if windows.len() == 1 { "" } else { "s" }
                                );
                                for w in &windows {
                                    let panes = w.children().await?;
                                    let winfo = w.window_info();
                                    let wname = winfo
                                        .map(|i| i.name.as_str())
                                        .unwrap_or("?");
                                    println!(
                                        "    {:<18} (Window, '{}', {} pane{})",
                                        w.target_string(),
                                        wname,
                                        panes.len(),
                                        if panes.len() == 1 { "" } else { "s" }
                                    );
                                    for p in &panes {
                                        let pid = p.pane_address()
                                            .map(|a| a.pane_id.as_str())
                                            .unwrap_or("?");
                                        println!(
                                            "      {:<16} (Pane, {})",
                                            p.target_string(),
                                            pid
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            "send" => {
                if parts.len() < 3 {
                    println!("usage: send <target> <text...>");
                    write!(stdout, "repl> ")?;
                    stdout.flush()?;
                    continue;
                }
                let target_str = parts[1];
                let text = parts[2];
                match resolve_target(&host, target_str).await {
                    Ok(target) => {
                        if let Err(e) = target.send_text(text).await {
                            println!("Error sending text: {}", e);
                        } else if let Err(e) = target.send_keys(&enter).await {
                            println!("Error sending Enter: {}", e);
                        } else {
                            println!("Sent to {}", target_str);
                        }
                    }
                    Err(e) => println!("{}", e),
                }
            }

            "capture" => {
                if parts.len() < 3 {
                    println!("usage: capture <target> <n>");
                    write!(stdout, "repl> ")?;
                    stdout.flush()?;
                    continue;
                }
                let target_str = parts[1];
                let n: usize = match parts[2].parse() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        println!("Error: <n> must be a positive integer");
                        write!(stdout, "repl> ")?;
                        stdout.flush()?;
                        continue;
                    }
                };
                match resolve_target(&host, target_str).await {
                    Ok(target) => {
                        let query = ScrollbackQuery::LastLines(n);
                        match target.sample_text(&query).await {
                            Ok(text) => {
                                if text.is_empty() {
                                    println!("(empty)");
                                } else {
                                    print!("{}", text);
                                    if !text.ends_with('\n') {
                                        println!();
                                    }
                                }
                            }
                            Err(e) => println!("Error: {}", e),
                        }
                    }
                    Err(e) => println!("{}", e),
                }
            }

            "quit" => {
                println!("Disconnected.");
                return Ok(());
            }

            other => {
                println!("Unknown command: {}", other);
                println!("Commands: create, kill, targets, send, capture, quit");
            }
        }

        write!(stdout, "repl> ")?;
        stdout.flush()?;
    }

    Ok(())
}

async fn resolve_target(
    host: &motlie_tmux::HostHandle,
    target_str: &str,
) -> Result<motlie_tmux::Target, String> {
    let spec = TargetSpec::parse(target_str).map_err(|e| format!("Invalid target '{}': {}", target_str, e))?;
    host.target(&spec)
        .await
        .map_err(|e| format!("Error resolving '{}': {}", target_str, e))?
        .ok_or_else(|| format!("Target '{}' not found", target_str))
}
