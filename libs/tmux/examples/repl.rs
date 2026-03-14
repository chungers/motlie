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

use motlie_tmux::{CreateSessionOptions, KeySequence, ScrollbackQuery, SshConfig, TargetSpec};
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
                let words: Vec<&str> = line.trim().split_whitespace().collect();
                if words.len() < 2 {
                    println!("usage: create <name> [--size WxH] [--history N]");
                    write!(stdout, "repl> ")?;
                    stdout.flush()?;
                    continue;
                }
                let name = words[1];
                let mut opts = CreateSessionOptions::default();
                let mut i = 2;
                let mut parse_err = false;
                while i < words.len() {
                    match words[i] {
                        "--size" => {
                            i += 1;
                            if let Some(val) = words.get(i) {
                                if let Some((w, h)) = val.split_once('x') {
                                    match (w.parse::<u16>(), h.parse::<u16>()) {
                                        (Ok(w), Ok(h)) if w > 0 && h > 0 => {
                                            opts.width = Some(w);
                                            opts.height = Some(h);
                                        }
                                        _ => {
                                            println!("Error: --size must be WxH (e.g. 200x50)");
                                            parse_err = true;
                                            break;
                                        }
                                    }
                                } else {
                                    println!("Error: --size must be WxH (e.g. 200x50)");
                                    parse_err = true;
                                    break;
                                }
                            } else {
                                println!("Error: --size requires a value");
                                parse_err = true;
                                break;
                            }
                        }
                        "--history" => {
                            i += 1;
                            match words.get(i).and_then(|v| v.parse::<u32>().ok()) {
                                Some(n) if n > 0 => opts.history_limit = Some(n),
                                _ => {
                                    println!("Error: --history must be a positive integer");
                                    parse_err = true;
                                    break;
                                }
                            }
                        }
                        other => {
                            println!("Error: unknown flag '{}'", other);
                            parse_err = true;
                            break;
                        }
                    }
                    i += 1;
                }
                if parse_err {
                    write!(stdout, "repl> ")?;
                    stdout.flush()?;
                    continue;
                }
                match host.create_session(name, &opts).await {
                    Ok(_) => {
                        let mut detail = String::new();
                        if let (Some(w), Some(h)) = (opts.width, opts.height) {
                            detail.push_str(&format!(" ({}x{})", w, h));
                        }
                        if let Some(n) = opts.history_limit {
                            detail.push_str(&format!(" history={}", n));
                        }
                        println!("Created: {}{}", name, detail);
                    }
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
                            let spec = match TargetSpec::parse(&s.name) {
                                Ok(sp) => sp,
                                Err(e) => {
                                    println!("  {} (parse error: {})", s.name, e);
                                    continue;
                                }
                            };
                            let target = match host.target(&spec).await {
                                Ok(Some(t)) => t,
                                Ok(None) => {
                                    println!("  {} (not found)", s.name);
                                    continue;
                                }
                                Err(e) => {
                                    println!("  {} (error: {})", s.name, e);
                                    continue;
                                }
                            };
                            let windows = match target.children().await {
                                Ok(w) => w,
                                Err(e) => {
                                    println!("  {:<20} (error listing windows: {})", s.name, e);
                                    continue;
                                }
                            };
                            println!(
                                "  {:<20} (Session, {} window{})",
                                s.name,
                                windows.len(),
                                if windows.len() == 1 { "" } else { "s" }
                            );
                            for w in &windows {
                                let panes = match w.children().await {
                                    Ok(p) => p,
                                    Err(e) => {
                                        println!("    {:<18} (error listing panes: {})", w.target_string(), e);
                                        continue;
                                    }
                                };
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
