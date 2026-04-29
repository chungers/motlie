//! Example: Interactive REPL for managing tmux sessions over SSH.
//!
//! Connects to a host via SSH URI, then enters a command loop for
//! creating/killing sessions, listing targets, sending input, and
//! capturing scrollback.
//!
//! Commands:
//!   help                     Show available commands
//!   create <name>            Create a new tmux session
//!   new-window <session> <name> [--size WxH]
//!                            Create a child window under a session
//!   split-pane <target> [--horizontal|--vertical] [--percent N|--cells N]
//!                            Split a pane from a window or pane target
//!   kill <target>            Kill a session/window/pane
//!   targets                  List all sessions with full target spec tree
//!   send <target> <text...>  Send text + Enter to a target
//!   keys <target> <keys...>  Send key sequence ({Escape}, {C-c}, etc.)
//!   capture <target> <n>     Print last N scrollback lines
//!   monitor <session> [secs] Stream live output for N seconds (default 3)
//!   tui on                   Enter split-screen TUI mirror mode (DC32)
//!   tui off                  Return to plain REPL mode (inside TUI)
//!   upload <local> <remote>  Upload a file or directory to the host
//!   download <remote> <local> Download a file or directory from the host
//!   quit                     Disconnect and exit
//!
//! Usage:
//!   cargo run -p motlie-tmux --example repl -- ssh://localhost
//!   cargo run -p motlie-tmux --example repl -- 'ssh://deploy@prod?identity-file=/path/to/key'

mod tui_mirror;

use motlie_tmux::{
    strip_ansi, CreateSessionOptions, CreateWindowOptions, KeySequence, LabelFormat,
    ScrollbackQuery, SinkFilter, SplitDirection, SplitPaneOptions, SplitSize, SshConfig,
    TargetSpec, TransferOptions,
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let host = SshConfig::parse(&uri)?.connect().await?;
    println!("Connected to {}", uri);

    let mut rl = DefaultEditor::new()?;
    let enter = KeySequence::parse("{Enter}")?;

    loop {
        let line = match tokio::task::block_in_place(|| rl.readline("repl> ")) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);

        let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();

        match parts[0] {
            "help" => {
                println!("Commands:");
                println!("  help                                    Show this help message");
                println!("  create <name> [--size WxH] [--history N]  Create a tmux session");
                println!(
                    "  new-window <session> <name> [--size WxH]   Create a window under a session"
                );
                println!("  split-pane <target> [--horizontal|--vertical] [--percent N|--cells N]");
                println!("                                          Split a pane from a window or pane target");
                println!("  kill <target>                           Kill a session/window/pane");
                println!(
                    "  targets                                 List all sessions with target tree"
                );
                println!("  send <target> <text...>                 Send text + Enter to a target");
                println!("  keys <target> <keys...>                 Send key sequence (e.g. {{Escape}}, {{C-c}})");
                println!("  capture <target> <n>                    Print last N scrollback lines");
                println!(
                    "  monitor <session> [secs]                Stream live output (default 3s)"
                );
                println!("  tui on                                  Enter split-screen TUI mode");
                println!("  upload <local> <remote> [-r]            Upload file/dir to the host");
                println!(
                    "  download <remote> <local> [-r]          Download file/dir from the host"
                );
                println!("  quit                                    Disconnect and exit");
                println!();
                println!("Targets: session, session:window, session:window.pane");
                println!("Transfer: use -r or --recursive for directories");
            }

            "create" => {
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                if words.len() < 2 {
                    println!("usage: create <name> [--size WxH] [--history N]");
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

            "new-window" => {
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                if words.len() < 3 {
                    println!("usage: new-window <session> <name> [--size WxH]");
                    continue;
                }
                let session_name = words[1];
                let window_name = words[2];
                let mut opts = CreateWindowOptions {
                    name: Some(window_name.to_string()),
                    ..Default::default()
                };
                let mut i = 3;
                let mut parse_err = false;
                while i < words.len() {
                    match words[i] {
                        "--size" => {
                            i += 1;
                            match words.get(i).and_then(|val| parse_size_arg(val)) {
                                Some((w, h)) => {
                                    opts.width = Some(w);
                                    opts.height = Some(h);
                                }
                                None => {
                                    println!("Error: --size must be WxH (e.g. 200x50)");
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
                    continue;
                }
                match resolve_target(&host, session_name).await {
                    Ok(target) => match target.new_window(&opts).await {
                        Ok(window) => println!("Created window: {}", window.target_string()),
                        Err(e) => println!("Error: {}", e),
                    },
                    Err(e) => println!("{}", e),
                }
            }

            "split-pane" => {
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                if words.len() < 2 {
                    println!("usage: split-pane <target> [--horizontal|--vertical] [--percent N|--cells N]");
                    continue;
                }
                let target_str = words[1];
                let mut opts = SplitPaneOptions::default();
                let mut i = 2;
                let mut parse_err = false;
                while i < words.len() {
                    match words[i] {
                        "--horizontal" => opts.direction = SplitDirection::Horizontal,
                        "--vertical" => opts.direction = SplitDirection::Vertical,
                        "--percent" => {
                            i += 1;
                            match words.get(i).and_then(|v| v.parse::<u8>().ok()) {
                                Some(v) => match SplitSize::percent(v) {
                                    Ok(size) => opts.size = Some(size),
                                    Err(e) => {
                                        println!("Error: {}", e);
                                        parse_err = true;
                                        break;
                                    }
                                },
                                None => {
                                    println!("Error: --percent must be an integer in 1..=100");
                                    parse_err = true;
                                    break;
                                }
                            }
                        }
                        "--cells" => {
                            i += 1;
                            match words.get(i).and_then(|v| v.parse::<u16>().ok()) {
                                Some(v) if v > 0 => opts.size = Some(SplitSize::Cells(v)),
                                _ => {
                                    println!("Error: --cells must be a positive integer");
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
                    continue;
                }
                match resolve_target(&host, target_str).await {
                    Ok(target) => match target.split_pane(&opts).await {
                        Ok(pane) => println!("Created pane: {}", pane.target_string()),
                        Err(e) => println!("Error: {}", e),
                    },
                    Err(e) => println!("{}", e),
                }
            }

            "kill" => {
                let target_str = match parts.get(1) {
                    Some(t) => *t,
                    None => {
                        println!("usage: kill <target>");
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

            "targets" => match host.list_sessions().await {
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
                                    println!(
                                        "    {:<18} (error listing panes: {})",
                                        w.target_string(),
                                        e
                                    );
                                    continue;
                                }
                            };
                            let winfo = w.window_info();
                            let wname = winfo.map(|i| i.name.as_str()).unwrap_or("?");
                            println!(
                                "    {:<18} (Window, '{}', {} pane{})",
                                w.target_string(),
                                wname,
                                panes.len(),
                                if panes.len() == 1 { "" } else { "s" }
                            );
                            for p in &panes {
                                let pid =
                                    p.pane_address().map(|a| a.pane_id.as_str()).unwrap_or("?");
                                println!("      {:<16} (Pane, {})", p.target_string(), pid);
                            }
                        }
                    }
                }
                Err(e) => println!("Error: {}", e),
            },

            "send" => {
                if parts.len() < 3 {
                    println!("usage: send <target> <text...>");
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

            "keys" => {
                if parts.len() < 3 {
                    println!("usage: keys <target> <keys...>");
                    println!("  e.g. keys mysession {{Escape}}");
                    println!("  e.g. keys mysession {{C-c}}");
                    println!("  e.g. keys mysession echo hello{{Enter}}");
                    continue;
                }
                let target_str = parts[1];
                let keys_str = parts[2];
                match KeySequence::parse(keys_str) {
                    Ok(keys) => match resolve_target(&host, target_str).await {
                        Ok(target) => match target.send_keys(&keys).await {
                            Ok(()) => println!("Sent keys to {}", target_str),
                            Err(e) => println!("Error: {}", e),
                        },
                        Err(e) => println!("{}", e),
                    },
                    Err(e) => println!("Parse error: {}", e),
                }
            }

            "capture" => {
                if parts.len() < 3 {
                    println!("usage: capture <target> <n>");
                    continue;
                }
                let target_str = parts[1];
                let n: usize = match parts[2].parse() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        println!("Error: <n> must be a positive integer");
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

            "monitor" => {
                if parts.len() < 2 {
                    println!("usage: monitor <session> [seconds]");
                    continue;
                }
                let session_name = parts[1];
                let seconds: u64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

                let bus = host.output_bus();
                let filter = SinkFilter::for_session(session_name);
                match bus.subscribe(vec![filter], 64) {
                    Ok(sub) => match host.start_monitoring_session(session_name).await {
                        Ok(monitor_handle) => {
                            println!("Monitoring {} for {}s...", session_name, seconds);

                            let mut stream = sub.joined(LabelFormat::Bracketed);
                            let deadline = tokio::time::Instant::now()
                                + std::time::Duration::from_secs(seconds);

                            loop {
                                tokio::select! {
                                    _ = tokio::time::sleep_until(deadline) => break,
                                    chunk = stream.next() => {
                                        match chunk {
                                            Some(chunk) => {
                                                let clean = strip_ansi(&chunk.output.content);
                                                if clean.trim().is_empty() {
                                                    continue;
                                                }
                                                if chunk.source_changed {
                                                    println!("\x1b[2m--- {} ---\x1b[0m",
                                                        chunk.source.minimal());
                                                }
                                                print!("{}", clean);
                                                if !clean.ends_with('\n') {
                                                    println!();
                                                }
                                            }
                                            None => break,
                                        }
                                    }
                                }
                            }

                            let _ = monitor_handle.shutdown().await;
                            println!("Monitor stopped.");
                        }
                        Err(e) => println!("Error: {}", e),
                    },
                    Err(e) => println!("Subscribe error: {}", e),
                }
            }

            "tui" => {
                if parts.get(1) == Some(&"on") {
                    match tui_mirror::run(&host, &uri).await? {
                        tui_mirror::TuiAction::TuiOff => {
                            println!("Returned to plain REPL mode.");
                        }
                        tui_mirror::TuiAction::Quit => {
                            println!("Disconnected.");
                            return Ok(());
                        }
                    }
                } else {
                    println!("usage: tui on | tui off (off only works inside TUI mode)");
                }
            }

            "upload" => {
                if parts.len() < 3 {
                    println!("usage: upload <local_path> <remote_path> [--recursive]");
                    continue;
                }
                let local_path = std::path::Path::new(parts[1]);
                // parts[2] may contain "<remote_path> --recursive"
                let rest: Vec<&str> = parts[2].split_whitespace().collect();
                let remote_path = std::path::Path::new(rest[0]);
                let recursive = rest.iter().any(|f| *f == "--recursive" || *f == "-r");
                let opts = TransferOptions {
                    overwrite: true,
                    recursive,
                };
                match host.upload(local_path, remote_path, &opts).await {
                    Ok(()) => println!(
                        "Uploaded {} → {}",
                        local_path.display(),
                        remote_path.display()
                    ),
                    Err(e) => println!("Error: {}", e),
                }
            }

            "download" => {
                if parts.len() < 3 {
                    println!("usage: download <remote_path> <local_path> [--recursive]");
                    continue;
                }
                let remote_path = std::path::Path::new(parts[1]);
                let rest: Vec<&str> = parts[2].split_whitespace().collect();
                let local_path = std::path::Path::new(rest[0]);
                let recursive = rest.iter().any(|f| *f == "--recursive" || *f == "-r");
                let opts = TransferOptions {
                    overwrite: true,
                    recursive,
                };
                match host.download(remote_path, local_path, &opts).await {
                    Ok(()) => println!(
                        "Downloaded {} → {}",
                        remote_path.display(),
                        local_path.display()
                    ),
                    Err(e) => println!("Error: {}", e),
                }
            }

            "quit" => {
                println!("Disconnected.");
                return Ok(());
            }

            other => {
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    other
                );
            }
        }
    }

    println!("Disconnected.");
    Ok(())
}

async fn resolve_target(
    host: &motlie_tmux::HostHandle,
    target_str: &str,
) -> Result<motlie_tmux::Target, String> {
    let spec = TargetSpec::parse(target_str)
        .map_err(|e| format!("Invalid target '{}': {}", target_str, e))?;
    host.target(&spec)
        .await
        .map_err(|e| format!("Error resolving '{}': {}", target_str, e))?
        .ok_or_else(|| format!("Target '{}' not found", target_str))
}

fn parse_size_arg(value: &str) -> Option<(u16, u16)> {
    let (w, h) = value.split_once('x')?;
    match (w.parse::<u16>(), h.parse::<u16>()) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => Some((w, h)),
        _ => None,
    }
}
