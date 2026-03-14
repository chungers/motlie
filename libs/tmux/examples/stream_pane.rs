//! Example: Continuously stream pane content to stdout.
//!
//! Demonstrates the distinct capture and streaming techniques in motlie-tmux.
//! Use `--mode` to select a strategy. Ctrl-C exits cleanly in all modes.
//!
//! Modes:
//!
//!   visible   Poll visible pane content via `capture()`. Prints full pane
//!             whenever it changes. Good for watching TUI programs.
//!
//!   tail      Poll scrollback via `sample_text(LastLines(n))` with
//!             `overlap_deduplicate()`. Prints only new lines. Like `tail -f`.
//!             This is the default.
//!
//!   until     Poll scrollback via `sample_text(Until { pattern, max_lines })`.
//!             Scans backwards from current position until a regex matches,
//!             prints from that match to the end. Good for watching output
//!             since the last shell prompt.
//!
//!   fidelity  Poll via `capture_with_options()` with `detect_reflow: true`.
//!             Prints content + fidelity metadata each tick. Demonstrates
//!             resize/reflow detection.
//!
//! Usage:
//!   stream_pane <uri> <target> [--mode MODE] [--lines N] [--interval MS] [--pattern REGEX]
//!
//! Examples:
//!   # Default tail mode
//!   ./target/debug/examples/stream_pane ssh://localhost my_session
//!
//!   # Watch visible pane (TUI programs)
//!   ./target/debug/examples/stream_pane ssh://localhost my_session --mode visible
//!
//!   # Tail with custom line count and interval
//!   ./target/debug/examples/stream_pane ssh://localhost my_session --mode tail --lines 100 --interval 500
//!
//!   # Stream from last shell prompt
//!   ./target/debug/examples/stream_pane ssh://localhost my_session --mode until --pattern '^\$ '
//!
//!   # Watch with fidelity detection (try resizing the target terminal)
//!   ./target/debug/examples/stream_pane ssh://localhost my_session --mode fidelity

use motlie_tmux::{
    overlap_deduplicate, CaptureNormalizeMode, CaptureOptions, ScrollbackQuery, SshConfig,
    TargetSpec,
};
use std::io::Write;
use std::time::Duration;

#[derive(Clone)]
enum Mode {
    Visible,
    Tail,
    Until,
    Fidelity,
}

struct Args {
    uri: String,
    target: String,
    mode: Mode,
    lines: usize,
    interval_ms: u64,
    pattern: String,
}

const HELP: &str = "\
stream_pane — continuously stream tmux pane content to stdout

USAGE:
    stream_pane <uri> <target> [OPTIONS]

ARGS:
    <uri>       SSH URI to connect (e.g. ssh://localhost, ssh://deploy@prod)
    <target>    tmux target string (session, session:window, session:window.pane)

OPTIONS:
    --mode MODE       Capture strategy [default: tail]
    --lines N         Scrollback line count for tail/until modes [default: 50]
    --interval MS     Poll interval in milliseconds [default: 200]
    --pattern REGEX   Regex for until mode [default: ^\\$ ]
    -h, --help        Print this help

MODES:
    tail       Poll scrollback via sample_text(LastLines(n)) with overlap_deduplicate().
               Prints only new lines as they appear. Like `tail -f`.
               Uses: ScrollbackQuery::LastLines, overlap_deduplicate()

    visible    Poll visible pane via capture(). Reprints full pane on change.
               No scrollback — only the current screen. Best for TUI programs
               (htop, vim, top) where the whole screen repaints.
               Uses: Target::capture()

    until      Poll scrollback via sample_text(Until { pattern, max_lines }).
               Scans backwards until regex matches, shows from match to end.
               Good for watching output since the last shell prompt.
               Uses: ScrollbackQuery::Until { pattern, max_lines }

    fidelity   Poll via capture_with_options() with detect_reflow enabled.
               Prints content + fidelity status line (CLEAN or DEGRADED with
               issue names). Resize the target terminal while running to see
               ClientResize / PaneResize issues appear.
               Uses: CaptureOptions { detect_reflow: true }, OutputFidelity

EXAMPLES:
    stream_pane ssh://localhost my_session
    stream_pane ssh://localhost my_session --mode visible
    stream_pane ssh://localhost my_session --mode tail --lines 100 --interval 500
    stream_pane ssh://localhost my_session --mode until --pattern '^\\$ '
    stream_pane ssh://localhost my_session --mode fidelity
    stream_pane ssh://localhost \"my_session:0.1\" --lines 30";

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();

    // Check for -h / --help anywhere in args
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", HELP);
        std::process::exit(0);
    }

    if args.len() < 3 {
        return Err(format!(
            "usage: {} <uri> <target> [--mode visible|tail|until|fidelity] \
             [--lines N] [--interval MS] [--pattern REGEX]\n\n\
             Try -h for detailed help.",
            args[0]
        ));
    }

    let mut mode = Mode::Tail;
    let mut lines = 50usize;
    let mut interval_ms = 200u64;
    let mut pattern = r"^\$ ".to_string();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                let val = args.get(i).ok_or("--mode requires a value")?;
                mode = match val.as_str() {
                    "visible" => Mode::Visible,
                    "tail" => Mode::Tail,
                    "until" => Mode::Until,
                    "fidelity" => Mode::Fidelity,
                    other => return Err(format!("unknown mode: '{}' (visible|tail|until|fidelity)", other)),
                };
            }
            "--lines" => {
                i += 1;
                lines = args
                    .get(i)
                    .ok_or("--lines requires a value")?
                    .parse()
                    .map_err(|_| "--lines must be a positive integer".to_string())?;
                if lines == 0 {
                    return Err("--lines must be > 0".to_string());
                }
            }
            "--interval" => {
                i += 1;
                interval_ms = args
                    .get(i)
                    .ok_or("--interval requires a value")?
                    .parse()
                    .map_err(|_| "--interval must be a positive integer".to_string())?;
                if interval_ms == 0 {
                    return Err("--interval must be > 0".to_string());
                }
            }
            "--pattern" => {
                i += 1;
                pattern = args
                    .get(i)
                    .ok_or("--pattern requires a value")?
                    .clone();
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }

    Ok(Args {
        uri: args[1].clone(),
        target: args[2].clone(),
        mode,
        lines,
        interval_ms,
        pattern,
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

    let host = SshConfig::parse(&args.uri)?.connect().await?;

    let spec = TargetSpec::parse(&args.target)?;
    let target = host
        .target(&spec)
        .await?
        .ok_or_else(|| anyhow::anyhow!("target '{}' not found", args.target))?;

    let mode_name = match args.mode {
        Mode::Visible => "visible",
        Mode::Tail => "tail",
        Mode::Until => "until",
        Mode::Fidelity => "fidelity",
    };
    eprintln!(
        "Streaming {} [mode={}, lines={}, interval={}ms]. Ctrl-C to stop.",
        target.target_string(),
        mode_name,
        args.lines,
        args.interval_ms
    );

    let interval = Duration::from_millis(args.interval_ms);

    match args.mode {
        Mode::Visible => stream_visible(&target, interval).await,
        Mode::Tail => stream_tail(&target, interval, args.lines).await,
        Mode::Until => stream_until(&target, interval, args.lines, &args.pattern).await,
        Mode::Fidelity => stream_fidelity(&target, interval).await,
    }
}

/// Mode: visible — poll `capture()`, print full pane when content changes.
///
/// Uses `capture()` which returns the visible pane area only (no scrollback).
/// Simple string comparison detects changes. Best for watching TUI programs
/// where the entire screen may repaint.
async fn stream_visible(
    target: &motlie_tmux::Target,
    interval: Duration,
) -> anyhow::Result<()> {
    let mut previous = String::new();
    let mut stdout = std::io::stdout().lock();
    let mut tick = 0u64;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = tokio::time::sleep(interval) => {
                let content = target.capture().await?;
                if content != previous {
                    // Clear screen and reprint for TUI-style display
                    if tick > 0 {
                        write!(stdout, "\x1b[2J\x1b[H")?; // clear + home
                    }
                    stdout.write_all(content.as_bytes())?;
                    stdout.flush()?;
                    previous = content;
                }
                tick += 1;
            }
        }
    }

    eprintln!("\nStopped.");
    Ok(())
}

/// Mode: tail — poll `sample_text(LastLines(n))` with `overlap_deduplicate()`.
///
/// Captures the last N lines of scrollback each tick. Uses 5-line overlap
/// deduplication to merge with the previous capture and print only new lines.
/// Like `tail -f` but for a tmux pane's scrollback buffer.
async fn stream_tail(
    target: &motlie_tmux::Target,
    interval: Duration,
    lines: usize,
) -> anyhow::Result<()> {
    let query = ScrollbackQuery::LastLines(lines);
    let overlap = 5usize;
    let mut previous = String::new();
    let mut stdout = std::io::stdout().lock();

    // Initial capture
    let initial = target.sample_text(&query).await?;
    if !initial.is_empty() {
        stdout.write_all(initial.as_bytes())?;
        if !initial.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        stdout.flush()?;
        previous = initial;
    }

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = tokio::time::sleep(interval) => {
                let current = target.sample_text(&query).await?;
                if current == previous || current.is_empty() {
                    continue;
                }

                let (merged, _issues) = overlap_deduplicate(&previous, &current, overlap);

                if merged.len() > previous.len() {
                    let new_content = &merged[previous.len()..];
                    stdout.write_all(new_content.as_bytes())?;
                    stdout.flush()?;
                } else if merged != previous {
                    // Dedup resync — print full current
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

/// Mode: until — poll `sample_text(Until { pattern, max_lines })`.
///
/// Scans backwards from the current scrollback position until the regex
/// pattern matches. Returns everything from that match to the end. Useful
/// for "show me everything since the last shell prompt." On each tick,
/// prints the full matched region (not incremental).
async fn stream_until(
    target: &motlie_tmux::Target,
    interval: Duration,
    max_lines: usize,
    pattern_str: &str,
) -> anyhow::Result<()> {
    let pattern = regex::Regex::new(pattern_str)
        .map_err(|e| anyhow::anyhow!("invalid regex '{}': {}", pattern_str, e))?;
    let query = ScrollbackQuery::Until {
        pattern,
        max_lines,
    };
    let mut previous = String::new();
    let mut stdout = std::io::stdout().lock();

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = tokio::time::sleep(interval) => {
                let content = target.sample_text(&query).await?;
                if content != previous {
                    // Clear and reprint the matched region
                    write!(stdout, "\x1b[2J\x1b[H")?;
                    stdout.write_all(content.as_bytes())?;
                    stdout.flush()?;
                    previous = content;
                }
            }
        }
    }

    eprintln!("\nStopped.");
    Ok(())
}

/// Mode: fidelity — poll `capture_with_options()` with reflow detection.
///
/// Uses `CaptureOptions { detect_reflow: true, normalize: PlainText }` to
/// capture the visible pane with geometry snapshot comparison. Prints content
/// plus a fidelity status line. Try resizing the target terminal while this
/// runs to see `ClientResize` / `PaneResize` fidelity issues appear.
async fn stream_fidelity(
    target: &motlie_tmux::Target,
    interval: Duration,
) -> anyhow::Result<()> {
    let opts = CaptureOptions {
        detect_reflow: true,
        normalize: CaptureNormalizeMode::PlainText,
        ..Default::default()
    };
    let mut previous_text = String::new();
    let mut previous_issues: Option<Vec<motlie_tmux::FidelityIssue>> = None;
    let mut stdout = std::io::stdout().lock();

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = tokio::time::sleep(interval) => {
                let result = target.capture_with_options(&opts).await?;
                let text_changed = result.text != previous_text;
                let fidelity_changed = result.fidelity.issues != previous_issues;

                if text_changed || fidelity_changed {
                    write!(stdout, "\x1b[2J\x1b[H")?;
                    stdout.write_all(result.text.as_bytes())?;

                    // Print fidelity status
                    write!(stdout, "\n\x1b[7m")?; // reverse video
                    if result.fidelity.degraded {
                        let issues = result.fidelity.issues.as_ref().unwrap();
                        let names: Vec<&str> = issues.iter().map(|i| match i {
                            motlie_tmux::FidelityIssue::ClientResize => "ClientResize",
                            motlie_tmux::FidelityIssue::PaneResize => "PaneResize",
                            motlie_tmux::FidelityIssue::HistoryTruncated => "HistoryTruncated",
                            motlie_tmux::FidelityIssue::OverlapResync => "OverlapResync",
                        }).collect();
                        write!(stdout, " DEGRADED: {} ", names.join(", "))?;
                    } else {
                        write!(stdout, " FIDELITY: CLEAN ")?;
                    }
                    write!(stdout, "\x1b[0m\n")?; // reset
                    stdout.flush()?;

                    previous_text = result.text;
                    previous_issues = result.fidelity.issues;
                }
            }
        }
    }

    eprintln!("\nStopped.");
    Ok(())
}
