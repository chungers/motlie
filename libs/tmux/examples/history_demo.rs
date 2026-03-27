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
    has_visible_text, pane_tail_excerpt, strip_ansi, CreateSessionOptions, HistoryOptions,
    KeySequence, LabelFormat, PollHistory, RenderMode, SinkFilter, SplitPaneOptions, SshConfig,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ContentFilter — pluggable per-source content filtering (prototype)
//
// Each agent TUI has its own chrome patterns (spinners, status bars, etc.)
// that change across versions. Filters are agent-specific and expected to
// evolve. The trait provides a stable interface; implementations are the
// volatile part.
// ---------------------------------------------------------------------------

trait ContentFilter {
    /// Filter a single line. Returns Some(cleaned) to keep, None to discard.
    fn filter_line(&self, line: &str) -> Option<String>;

    /// After collecting filtered lines, decide if the batch is worth recording.
    fn is_meaningful_batch(&self, lines: &[String]) -> bool;

    /// Semantic hint: does this line indicate the agent is waiting for user input?
    /// Used by PromptBoundary flush policy to detect turn boundaries.
    #[allow(unused_variables)]
    fn is_prompt(&self, line: &str) -> bool { false }
}

// --- Shared chrome detection helpers ---

/// Lines composed entirely of box-drawing / separator characters.
fn is_box_drawing_line(trimmed: &str) -> bool {
    trimmed.len() > 3
        && trimmed
            .chars()
            .all(|c| "─━═│┃┌┐└┘├┤┬┴┼╭╮╰╯-=+|".contains(c))
}

/// Lines that are just a bare prompt character with no content.
fn is_bare_prompt(trimmed: &str) -> bool {
    trimmed == "❯" || trimmed == "›" || trimmed == "$" || trimmed == "%"
}

/// Strip ANSI and normalize a raw line. Returns None if empty after cleaning.
fn clean_line(line: &str) -> Option<String> {
    let clean = strip_ansi(line).replace('\r', "");
    let trimmed = clean.trim_end().to_string();
    if trimmed.trim().is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

// --- RawFilter: passes everything, strips ANSI only ---

struct RawFilter;

impl ContentFilter for RawFilter {
    fn filter_line(&self, line: &str) -> Option<String> {
        clean_line(line)
    }
    fn is_meaningful_batch(&self, lines: &[String]) -> bool {
        !lines.is_empty()
    }
    fn is_prompt(&self, line: &str) -> bool {
        is_bare_prompt(line.trim())
    }
}

// --- ShellFilter: plain shell sessions ---

struct ShellFilter;

impl ContentFilter for ShellFilter {
    fn filter_line(&self, line: &str) -> Option<String> {
        clean_line(line)
    }
    fn is_meaningful_batch(&self, lines: &[String]) -> bool {
        !lines.is_empty()
    }
    fn is_prompt(&self, line: &str) -> bool {
        let t = line.trim();
        t.ends_with('$') || t.ends_with('%') || t.ends_with('#')
            || t.starts_with("$ ") || t.starts_with("% ")
    }
}

// --- ClaudeCodeFilter: Claude Code CLI TUI ---
//
// Chrome patterns specific to Claude Code's terminal UI.
// These will change as Claude Code evolves.

struct ClaudeCodeFilter;

impl ClaudeCodeFilter {
    /// Claude Code spinner prefixes and their activity words.
    const SPINNER_CHARS: &[char] = &['·', '✻', '✶', '✽', '✢', '✳', '⏺', '•'];
    const SPINNER_WORDS: &[&str] = &[
        "Thinking",
        "Working",
        "Searching",
        "Reading",
        "Writing",
        "Analyzing",
        "Whirring",
    ];

    /// Claude Code status bar / chrome fragments.
    const CHROME_PATTERNS: &[&str] = &[
        "esc to interrupt",
        "? for shortcuts",
        "ctrl+o to expand",
        "Auto-updating",
    ];

    fn is_chrome(trimmed: &str) -> bool {
        if is_box_drawing_line(trimmed) || is_bare_prompt(trimmed) {
            return true;
        }

        // Spinner lines: "· Thinking…", "⏺ Searching for 1 pattern…"
        if let Some(first) = trimmed.chars().next() {
            if Self::SPINNER_CHARS.contains(&first) {
                let rest = trimmed[first.len_utf8()..].trim();
                if rest.ends_with('…')
                    || Self::SPINNER_WORDS
                        .iter()
                        .any(|w| rest.starts_with(w))
                {
                    return true;
                }
            }
        }

        // Status bar patterns
        if Self::CHROME_PATTERNS
            .iter()
            .any(|p| trimmed.contains(p))
        {
            return true;
        }

        false
    }
}

impl ContentFilter for ClaudeCodeFilter {
    fn filter_line(&self, line: &str) -> Option<String> {
        let trimmed = clean_line(line)?;
        if ClaudeCodeFilter::is_chrome(&trimmed) {
            None
        } else {
            Some(trimmed)
        }
    }
    fn is_meaningful_batch(&self, lines: &[String]) -> bool {
        // Require at least one line that isn't just a prompt/question
        lines.iter().any(|l| {
            let t = l.trim();
            t.len() > 2 && !t.starts_with("❯ ") && !t.starts_with("› ")
        })
    }
    fn is_prompt(&self, line: &str) -> bool {
        let t = line.trim();
        t.starts_with("❯ ") || t == "❯"
    }
}

// --- CodexFilter: OpenAI Codex CLI TUI ---
//
// Chrome patterns specific to Codex's terminal UI.
// These will change as Codex evolves.

struct CodexFilter;

impl CodexFilter {
    /// Codex spinner / activity prefixes.
    const SPINNER_CHARS: &[char] = &['·', '✻', '✶', '✽', '✢', '✳', '⏺', '•'];
    const SPINNER_WORDS: &[&str] = &[
        "Kneading",
        "Garnishing",
        "Beboppin",
        "Propagating",
        "Simmering",
        "Marinating",
        "Whirring",
        "Explored",
        "Hatching",
        "Composing",
        "Polishing",
    ];

    /// Codex status bar / chrome fragments.
    const CHROME_PATTERNS: &[&str] = &[
        "esc to interrupt",
        "? for shortcuts",
        "background terminals running",
        "/ps to view",
        "/stop to close",
        "ctrl+o to expand",
        "Explain this codebase",
        "context left",
        "gpt-5",
        "tab to queue",
    ];

    fn is_chrome(trimmed: &str) -> bool {
        if is_box_drawing_line(trimmed) || is_bare_prompt(trimmed) {
            return true;
        }

        // Spinner lines
        if let Some(first) = trimmed.chars().next() {
            if Self::SPINNER_CHARS.contains(&first) {
                let rest = trimmed[first.len_utf8()..].trim();
                if rest.ends_with('…')
                    || Self::SPINNER_WORDS
                        .iter()
                        .any(|w| rest.starts_with(w))
                {
                    return true;
                }
            }
        }

        // Status bar patterns
        if Self::CHROME_PATTERNS
            .iter()
            .any(|p| trimmed.contains(p))
        {
            return true;
        }

        false
    }
}

impl ContentFilter for CodexFilter {
    fn filter_line(&self, line: &str) -> Option<String> {
        let trimmed = clean_line(line)?;
        if CodexFilter::is_chrome(&trimmed) {
            None
        } else {
            Some(trimmed)
        }
    }
    fn is_meaningful_batch(&self, lines: &[String]) -> bool {
        // Require at least one line that isn't just a prompt/question
        lines.iter().any(|l| {
            let t = l.trim();
            t.len() > 2 && !t.starts_with("› ") && !t.starts_with("❯ ")
        })
    }
    fn is_prompt(&self, line: &str) -> bool {
        let t = line.trim();
        t.starts_with("› ") || t == "›"
    }
}

#[derive(Clone, Copy)]
enum FilterMode {
    Raw,
    Shell,
    ClaudeCode,
    Codex,
}

fn make_filter(mode: FilterMode) -> Box<dyn ContentFilter> {
    match mode {
        FilterMode::Raw => Box::new(RawFilter),
        FilterMode::Shell => Box::new(ShellFilter),
        FilterMode::ClaudeCode => Box::new(ClaudeCodeFilter),
        FilterMode::Codex => Box::new(CodexFilter),
    }
}

// ---------------------------------------------------------------------------
// FlushPolicy — when to commit accumulated lines to history (prototype)
//
// Static dispatch via enum. Each variant has its own flush logic.
// The filter provides semantic hints (is_prompt) that policies can use.
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum FlushPolicy {
    /// Flush after N lines accumulate, or after max_wait with any content.
    LineCount {
        min_lines: usize,
        max_wait: Duration,
    },

    /// Flush when content stops changing for idle_duration.
    /// Good for build logs, streaming output.
    Idle {
        idle_duration: Duration,
        max_wait: Duration,
    },

    /// Flush when a prompt line appears after content, indicating the agent
    /// finished its turn. Falls back to max_wait if no prompt is detected.
    /// Best for interactive agent sessions (Claude Code, Codex, shells).
    PromptBoundary {
        max_wait: Duration,
        min_content_lines: usize,
    },
}

impl FlushPolicy {
    fn should_flush(
        &self,
        pending: &[String],
        time_since_flush: Duration,
        time_since_last_change: Duration,
        saw_prompt: bool,
    ) -> bool {
        if pending.is_empty() {
            return false;
        }

        match self {
            FlushPolicy::LineCount { min_lines, max_wait } => {
                pending.len() >= *min_lines || time_since_flush >= *max_wait
            }
            FlushPolicy::Idle { idle_duration, max_wait } => {
                time_since_last_change >= *idle_duration || time_since_flush >= *max_wait
            }
            FlushPolicy::PromptBoundary { max_wait, min_content_lines } => {
                // Flush when prompt detected after enough content
                if saw_prompt && pending.len() >= *min_content_lines {
                    return true;
                }
                // Fallback: max wait exceeded
                time_since_flush >= *max_wait
            }
        }
    }
}

#[derive(Clone, Copy)]
enum FlushPolicyType {
    LineCount,
    Idle,
    PromptBoundary,
}

fn make_policy(policy_type: FlushPolicyType) -> FlushPolicy {
    match policy_type {
        FlushPolicyType::LineCount => FlushPolicy::LineCount {
            min_lines: 3,
            max_wait: Duration::from_secs(10),
        },
        FlushPolicyType::Idle => FlushPolicy::Idle {
            idle_duration: Duration::from_secs(3),
            max_wait: Duration::from_secs(15),
        },
        FlushPolicyType::PromptBoundary => FlushPolicy::PromptBoundary {
            max_wait: Duration::from_secs(30),
            min_content_lines: 1,
        },
    }
}

// ---------------------------------------------------------------------------
// SourceAccumulator — per-source buffered content collection (prototype)
// ---------------------------------------------------------------------------

struct SourceAccumulator {
    #[allow(dead_code)]
    name: String,
    previous: HashMap<motlie_tmux::PaneAddress, String>,
    pending_lines: Vec<String>,
    last_flush: Instant,
    last_change: Instant,
    saw_prompt_since_flush: bool,
    filter: Box<dyn ContentFilter>,
    policy: FlushPolicy,
}

impl SourceAccumulator {
    fn new(
        name: &str,
        baseline: HashMap<motlie_tmux::PaneAddress, String>,
        filter: Box<dyn ContentFilter>,
        policy: FlushPolicy,
    ) -> Self {
        let now = Instant::now();
        Self {
            name: name.to_string(),
            previous: baseline,
            pending_lines: Vec::new(),
            last_flush: now,
            last_change: now,
            saw_prompt_since_flush: false,
            filter,
            policy,
        }
    }

    fn ingest(&mut self, current: &HashMap<motlie_tmux::PaneAddress, String>) -> Option<String> {
        if current == &self.previous {
            return self.maybe_flush();
        }

        let mut pane_list: Vec<_> = current.iter().collect();
        pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

        for (addr, content) in pane_list {
            let prev_content = self.previous.get(addr).map(String::as_str).unwrap_or("");
            let new_lines = diff_new_lines(prev_content, content, &*self.filter);
            for line in &new_lines {
                if self.filter.is_prompt(line) {
                    self.saw_prompt_since_flush = true;
                }
            }
            if !new_lines.is_empty() {
                self.pending_lines.extend(new_lines);
                self.last_change = Instant::now();
            }
        }

        self.previous = current.clone();
        self.maybe_flush()
    }

    fn maybe_flush(&mut self) -> Option<String> {
        if self.pending_lines.is_empty() {
            return None;
        }

        let should = self.policy.should_flush(
            &self.pending_lines,
            self.last_flush.elapsed(),
            self.last_change.elapsed(),
            self.saw_prompt_since_flush,
        );

        if !should {
            return None;
        }

        if !self.filter.is_meaningful_batch(&self.pending_lines) {
            self.pending_lines.clear();
            self.last_flush = Instant::now();
            self.saw_prompt_since_flush = false;
            return None;
        }

        let chunk = self.pending_lines.join("\n");
        self.pending_lines.clear();
        self.last_flush = Instant::now();
        self.saw_prompt_since_flush = false;

        if chunk.trim().is_empty() {
            None
        } else {
            Some(format!("{}\n", chunk))
        }
    }

    fn flush_remaining(&mut self) -> Option<String> {
        if self.pending_lines.is_empty() {
            return None;
        }
        let chunk = self.pending_lines.join("\n");
        self.pending_lines.clear();
        self.last_flush = Instant::now();
        self.saw_prompt_since_flush = false;
        if chunk.trim().is_empty() {
            None
        } else {
            Some(format!("{}\n", chunk))
        }
    }
}

/// Extract new lines from current pane content vs previous.
/// Uses set-based diff, then applies the content filter to each line.
fn diff_new_lines(previous: &str, current: &str, filter: &dyn ContentFilter) -> Vec<String> {
    let prev_set: std::collections::HashSet<String> = previous
        .lines()
        .filter_map(|l| clean_line(l))
        .collect();

    current
        .lines()
        .filter_map(|line| {
            // First clean the raw line
            let cleaned = clean_line(line)?;
            // Skip if it was in previous content
            if prev_set.contains(&cleaned) {
                return None;
            }
            // Apply the content filter
            filter.filter_line(line)
        })
        .collect()
}

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
    render_mode: RenderMode,
    filter_mode: FilterMode,
    policy_type: FlushPolicyType,
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
    let mut render_mode = RenderMode::Interleaved;
    let mut filter_mode = FilterMode::Shell;
    let mut policy_type = FlushPolicyType::PromptBoundary;
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
            "--render-mode" => {
                i += 1;
                render_mode = match argv
                    .get(i)
                    .ok_or_else(|| anyhow!("--render-mode requires a value"))?
                    .as_str()
                {
                    "interleaved" => RenderMode::Interleaved,
                    "per-source" => RenderMode::PerSource,
                    other => {
                        return Err(anyhow!(
                            "unknown render mode '{}' (interleaved|per-source)",
                            other
                        ))
                    }
                };
            }
            "--filter" => {
                i += 1;
                filter_mode = match argv
                    .get(i)
                    .ok_or_else(|| anyhow!("--filter requires a value"))?
                    .as_str()
                {
                    "raw" => FilterMode::Raw,
                    "shell" => FilterMode::Shell,
                    "claude" => FilterMode::ClaudeCode,
                    "codex" => FilterMode::Codex,
                    other => {
                        return Err(anyhow!(
                            "unknown filter '{}' (raw|shell|claude|codex)",
                            other
                        ))
                    }
                };
            }
            "--policy" => {
                i += 1;
                policy_type = match argv
                    .get(i)
                    .ok_or_else(|| anyhow!("--policy requires a value"))?
                    .as_str()
                {
                    "lines" => FlushPolicyType::LineCount,
                    "idle" => FlushPolicyType::Idle,
                    "prompt" => FlushPolicyType::PromptBoundary,
                    other => {
                        return Err(anyhow!(
                            "unknown policy '{}' (lines|idle|prompt)",
                            other
                        ))
                    }
                };
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
        render_mode,
        filter_mode,
        policy_type,
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
            let mut history = PollHistory::new(args.max_entries, args.max_chars)
                .with_render_mode(args.render_mode);
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
            println!("Accumulator: line_threshold=3, time_threshold=5s");
            println!("Ctrl-C to stop.\n");

            let baseline_a = target_a.capture_all().await?;
            let baseline_b = target_b.capture_all().await?;
            let mut acc_a = SourceAccumulator::new(
                session_a,
                baseline_a,
                make_filter(args.filter_mode),
                make_policy(args.policy_type),
            );
            let mut acc_b = SourceAccumulator::new(
                session_b,
                baseline_b,
                make_filter(args.filter_mode),
                make_policy(args.policy_type),
            );

            let mut interval = tokio::time::interval(Duration::from_secs(1));
            let mut tick = 0u64;

            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => break,
                    _ = interval.tick() => {
                        tick += 1;

                        let current_a = target_a.capture_all().await?;
                        if let Some(chunk) = acc_a.ingest(&current_a) {
                            // Print delta as it flushes — shows what was captured
                            println!("[t={}s] {} flushed:", tick, session_a);
                            println!("{}", chunk.trim_end());
                            println!();
                            history.push_text_for_source(session_a, chunk);
                        }
                        let current_b = target_b.capture_all().await?;
                        if let Some(chunk) = acc_b.ingest(&current_b) {
                            println!("[t={}s] {} flushed:", tick, session_b);
                            println!("{}", chunk.trim_end());
                            println!();
                            history.push_text_for_source(session_b, chunk);
                        }
                    }
                }
            }

            // Flush any remaining accumulated content
            if let Some(chunk) = acc_a.flush_remaining() {
                history.push_text_for_source(session_a, chunk);
            }
            if let Some(chunk) = acc_b.flush_remaining() {
                history.push_text_for_source(session_b, chunk);
            }

            println!("\n========== FINAL ROLLING CONTEXT ==========");
            println!("{}", history.render_text());
            println!(
                "Stats: entries={}, omitted={}, chars={}",
                history.len(),
                history.omitted_entries(),
                history.rendered_chars()
            );
            Ok(())
        }
        LiveMode::Render => {
            let mut history = PollHistory::new(args.max_entries, args.max_chars)
                .with_render_mode(args.render_mode);
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
                                history.push_text_for_source(session_a, rendered);
                            }
                            previous_a = current_a;
                        }

                        let current_b = target_b.capture_all().await?;
                        if current_b != previous_b {
                            if let Some(rendered) = render_history_entry(session_b, &current_b) {
                                history.push_text_for_source(session_b, rendered);
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
        render_mode: args.render_mode,
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
