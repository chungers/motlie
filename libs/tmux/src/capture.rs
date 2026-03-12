use anyhow::Result;
use std::collections::HashMap;

use crate::discovery;
use crate::transport::{shell_escape_arg, tmux_prefix, TransportKind};
use crate::types::{
    CaptureNormalizeMode, CaptureOptions, CaptureResult, FidelityIssue, OutputFidelity,
    PaneAddress, ScrollbackQuery, TmuxSocket,
};

// ---------------------------------------------------------------------------
// Raw capture primitives (unchanged)
// ---------------------------------------------------------------------------

/// Capture visible pane content.
pub async fn capture_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<String> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} capture-pane -p -t '{}'",
        prefix,
        shell_escape_arg(target)
    );
    transport.exec(&cmd).await
}

/// Capture pane content with scrollback history.
/// `start` is negative to go into scrollback (e.g. -100 = 100 lines above visible area).
/// Captures through the end of the visible area.
pub async fn capture_pane_history(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    start: i32,
) -> Result<String> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} capture-pane -p -t '{}' -S {}",
        prefix,
        shell_escape_arg(target),
        start,
    );
    transport.exec(&cmd).await
}

/// Capture all panes in a session.
pub async fn capture_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<HashMap<PaneAddress, String>> {
    let panes = discovery::list_panes_in_session(transport, socket, session).await?;
    let mut result = HashMap::new();
    for pane in panes {
        let target = pane.address.to_tmux_target();
        let content = capture_pane(transport, socket, &target).await?;
        result.insert(pane.address, content);
    }
    Ok(result)
}

/// Sample scrollback text according to a query.
pub async fn sample_text(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    query: &ScrollbackQuery,
) -> Result<String> {
    match query {
        ScrollbackQuery::LastLines(n) => {
            let n = *n as i32;
            let content = capture_pane_history(transport, socket, target, -n).await?;
            // Trim trailing blank lines from visible area padding
            let trimmed = content.trim_end();
            Ok(trimmed.to_string())
        }
        ScrollbackQuery::Until { pattern, max_lines } => {
            let max = *max_lines as i32;
            let content =
                capture_pane_history(transport, socket, target, -max).await?;
            let lines: Vec<&str> = content.lines().collect();
            // Scan from end to find the pattern
            for (i, line) in lines.iter().enumerate().rev() {
                if pattern.is_match(line) {
                    return Ok(lines[i..].join("\n"));
                }
            }
            // Pattern not found — return all captured
            Ok(content)
        }
        ScrollbackQuery::LastLinesUntil {
            lines,
            stop_pattern,
        } => {
            let n = *lines as i32;
            let content =
                capture_pane_history(transport, socket, target, -n).await?;
            let lines: Vec<&str> = content.lines().collect();
            // Scan from end to find stop pattern
            for (i, line) in lines.iter().enumerate().rev() {
                if stop_pattern.is_match(line) {
                    return Ok(lines[i..].join("\n"));
                }
            }
            Ok(content)
        }
    }
}

// ---------------------------------------------------------------------------
// Escape-mode capture primitive (ScreenStable)
// ---------------------------------------------------------------------------

/// Capture visible pane content with ANSI/control sequences preserved (`-ep`).
pub async fn capture_pane_escape(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<String> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} capture-pane -ep -t '{}'",
        prefix,
        shell_escape_arg(target)
    );
    transport.exec(&cmd).await
}

/// Capture pane content with ANSI preservation and scrollback history.
pub async fn capture_pane_escape_history(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    start: i32,
) -> Result<String> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} capture-pane -ep -t '{}' -S {}",
        prefix,
        shell_escape_arg(target),
        start,
    );
    transport.exec(&cmd).await
}

// ---------------------------------------------------------------------------
// Normalization functions (DC20)
// ---------------------------------------------------------------------------

/// Normalize captured text for ScreenStable mode.
///
/// - Canonicalizes line endings (`\r\n` → `\n`)
/// - Trims trailing whitespace per line (width-artifact padding from tmux)
/// - Preserves ANSI/control sequences within line content
/// - Trims trailing empty lines
pub fn normalize_screen_stable(raw: &str) -> String {
    let canonical = raw.replace("\r\n", "\n");
    let lines: Vec<&str> = canonical.lines().collect();
    let trimmed: Vec<&str> = lines.iter().map(|l| l.trim_end()).collect();
    // Trim trailing empty lines
    let end = trimmed
        .iter()
        .rposition(|l| !l.is_empty())
        .map_or(0, |i| i + 1);
    if end == 0 {
        String::new()
    } else {
        let mut result = trimmed[..end].join("\n");
        result.push('\n');
        result
    }
}

/// Strip ANSI escape sequences and control characters from text.
///
/// Removes:
/// - CSI sequences: `ESC [ ... final_byte`
/// - OSC sequences: `ESC ] ... ST` (ST = `ESC \` or BEL)
/// - Simple escape sequences: `ESC <char>`
/// - Remaining C0 control characters (except `\n`, `\t`)
pub fn strip_ansi(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // ESC sequence
            match chars.peek() {
                Some('[') => {
                    // CSI sequence: consume until final byte (0x40-0x7E)
                    chars.next(); // consume '['
                    while let Some(&ch) = chars.peek() {
                        chars.next();
                        if (0x40..=0x7E).contains(&(ch as u32)) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    // OSC sequence: consume until ST (ESC \ or BEL)
                    chars.next(); // consume ']'
                    while let Some(ch) = chars.next() {
                        if ch == '\x07' {
                            // BEL terminates OSC
                            break;
                        }
                        if ch == '\x1b' {
                            if chars.peek() == Some(&'\\') {
                                chars.next(); // consume '\'
                            }
                            break;
                        }
                    }
                }
                Some(&ch) if ch == '(' || ch == ')' || ch == '*' || ch == '+' => {
                    // Character set designation: ESC ( <charset>, ESC ) <charset>, etc.
                    // These are 2-byte sequences after ESC.
                    chars.next(); // consume '(' / ')' / '*' / '+'
                    chars.next(); // consume charset designator (e.g. 'B', '0')
                }
                Some(_) => {
                    // Simple escape: consume one character
                    chars.next();
                }
                None => {}
            }
        } else if c.is_control() && c != '\n' && c != '\t' {
            // Strip other C0 control characters
        } else {
            result.push(c);
        }
    }

    result
}

/// Normalize captured text for PlainText mode.
///
/// Canonicalizes line endings, strips ANSI/control sequences,
/// trims trailing whitespace per line, trims trailing empty lines.
pub fn normalize_plain_text(raw: &str) -> String {
    let stripped = strip_ansi(raw);
    normalize_screen_stable(&stripped)
}

// ---------------------------------------------------------------------------
// Options-based capture API (DC20)
// ---------------------------------------------------------------------------

/// Perform raw tmux capture based on options (mode + history).
async fn raw_capture(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    opts: &CaptureOptions,
) -> Result<String> {
    let use_escape = opts.normalize == CaptureNormalizeMode::ScreenStable;

    match (opts.history_start, use_escape) {
        (None, false) => capture_pane(transport, socket, target).await,
        (None, true) => capture_pane_escape(transport, socket, target).await,
        (Some(start), false) => {
            capture_pane_history(transport, socket, target, start).await
        }
        (Some(start), true) => {
            capture_pane_escape_history(transport, socket, target, start).await
        }
    }
}

/// Apply normalization to raw capture output based on mode.
fn apply_normalization(raw: &str, mode: CaptureNormalizeMode) -> (String, Option<String>) {
    match mode {
        CaptureNormalizeMode::Raw => (raw.to_string(), None),
        CaptureNormalizeMode::ScreenStable => {
            let normalized = normalize_screen_stable(raw);
            (normalized, Some(raw.to_string()))
        }
        CaptureNormalizeMode::PlainText => {
            let normalized = normalize_plain_text(raw);
            (normalized, None)
        }
    }
}

/// Detect geometry changes around a capture and build fidelity metadata.
async fn detect_fidelity(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    detect_reflow: bool,
) -> (Option<crate::types::GeometrySnapshot>, OutputFidelity) {
    if !detect_reflow {
        return (None, OutputFidelity::clean());
    }

    match discovery::take_geometry_snapshot(transport, socket, target).await {
        Ok(snap) => (Some(snap), OutputFidelity::clean()),
        Err(_) => {
            // If we can't take a snapshot, proceed with clean fidelity
            (None, OutputFidelity::clean())
        }
    }
}

/// Finalize fidelity by comparing pre/post snapshots.
async fn finalize_fidelity(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    pre_snapshot: Option<crate::types::GeometrySnapshot>,
) -> OutputFidelity {
    let pre = match pre_snapshot {
        Some(s) => s,
        None => return OutputFidelity::clean(),
    };

    let post = match discovery::take_geometry_snapshot(transport, socket, target).await {
        Ok(s) => s,
        Err(_) => return OutputFidelity::clean(),
    };

    let issues = pre.compare(&post);
    if issues.is_empty() {
        OutputFidelity::clean()
    } else {
        OutputFidelity::degraded(issues)
    }
}

/// Capture pane content with options, returning a `CaptureResult`.
pub async fn capture_pane_with_options(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    opts: &CaptureOptions,
) -> Result<CaptureResult> {
    let (pre_snapshot, _) =
        detect_fidelity(transport, socket, target, opts.detect_reflow).await;

    let raw = raw_capture(transport, socket, target, opts).await?;

    let fidelity =
        finalize_fidelity(transport, socket, target, pre_snapshot).await;

    let (text, raw_text) = apply_normalization(&raw, opts.normalize);

    Ok(CaptureResult {
        text,
        raw_text,
        fidelity,
    })
}

/// Sample scrollback text with options, returning a `CaptureResult`.
///
/// When `previous_text` is provided and `opts.overlap_lines >= 2`, performs
/// overlap-aware dedup between the previous capture and the new one.
/// If overlap matching fails, the result includes `OverlapResync` in fidelity.
pub async fn sample_text_with_options(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    query: &ScrollbackQuery,
    opts: &CaptureOptions,
    previous_text: Option<&str>,
) -> Result<CaptureResult> {
    let (pre_snapshot, _) =
        detect_fidelity(transport, socket, target, opts.detect_reflow).await;

    // Build effective options with history from query
    let effective_opts = match query {
        ScrollbackQuery::LastLines(n) => CaptureOptions {
            history_start: Some(-(*n as i32)),
            normalize: opts.normalize,
            overlap_lines: opts.overlap_lines,
            detect_reflow: false, // already handled above
        },
        ScrollbackQuery::Until { max_lines, .. } => CaptureOptions {
            history_start: Some(-(*max_lines as i32)),
            normalize: opts.normalize,
            overlap_lines: opts.overlap_lines,
            detect_reflow: false,
        },
        ScrollbackQuery::LastLinesUntil { lines, .. } => CaptureOptions {
            history_start: Some(-(*lines as i32)),
            normalize: opts.normalize,
            overlap_lines: opts.overlap_lines,
            detect_reflow: false,
        },
    };

    let raw = raw_capture(transport, socket, target, &effective_opts).await?;

    let mut fidelity =
        finalize_fidelity(transport, socket, target, pre_snapshot).await;

    // Apply query-specific filtering on normalized text
    let (text, raw_text) = apply_normalization(&raw, opts.normalize);

    let filtered = match query {
        ScrollbackQuery::LastLines(_) => text.trim_end().to_string(),
        ScrollbackQuery::Until { pattern, .. } => {
            let lines: Vec<&str> = text.lines().collect();
            let mut found = None;
            for (i, line) in lines.iter().enumerate().rev() {
                if pattern.is_match(line) {
                    found = Some(i);
                    break;
                }
            }
            match found {
                Some(i) => lines[i..].join("\n"),
                None => text,
            }
        }
        ScrollbackQuery::LastLinesUntil { stop_pattern, .. } => {
            let lines: Vec<&str> = text.lines().collect();
            let mut found = None;
            for (i, line) in lines.iter().enumerate().rev() {
                if stop_pattern.is_match(line) {
                    found = Some(i);
                    break;
                }
            }
            match found {
                Some(i) => lines[i..].join("\n"),
                None => text,
            }
        }
    };

    // Apply overlap dedup if previous text and sufficient overlap lines provided
    let final_text = if let Some(prev) = previous_text {
        if opts.overlap_lines >= 2 {
            let (merged, overlap_issues) =
                overlap_deduplicate(prev, &filtered, opts.overlap_lines);
            if !overlap_issues.is_empty() {
                // Merge overlap issues into fidelity
                let mut all_issues = fidelity
                    .issues
                    .unwrap_or_default();
                all_issues.extend(overlap_issues);
                fidelity = OutputFidelity::degraded(all_issues);
            }
            merged
        } else {
            filtered
        }
    } else {
        filtered
    };

    Ok(CaptureResult {
        text: final_text,
        raw_text,
        fidelity,
    })
}

// ---------------------------------------------------------------------------
// Overlap-aware incremental sampling (DC20, Phase 1.9b)
// ---------------------------------------------------------------------------

/// Attempt to deduplicate overlapping content between a previous capture
/// and a new capture.
///
/// Returns `(merged_text, issues)`. If overlap matching succeeds, the merged
/// text contains the previous content followed by new-only lines. If matching
/// fails (ambiguous or insufficient overlap), returns the new capture with
/// an `OverlapResync` fidelity issue.
///
/// **Note**: `overlap_lines` must be `>= 2` for dedup to engage. Values of
/// 0 or 1 silently return `current` unchanged (with a `tracing::warn`).
/// This is intentional — single-line overlap is too ambiguous for reliable
/// matching.
pub fn overlap_deduplicate(
    previous: &str,
    current: &str,
    overlap_lines: usize,
) -> (String, Vec<FidelityIssue>) {
    if overlap_lines < 2 {
        if overlap_lines > 0 && !previous.is_empty() && !current.is_empty() {
            tracing::warn!(
                overlap_lines,
                "overlap_deduplicate called with overlap_lines < 2; \
                 dedup requires >= 2 lines for reliable matching, returning current unchanged"
            );
        }
        return (current.to_string(), Vec::new());
    }
    if previous.is_empty() || current.is_empty() {
        return (current.to_string(), Vec::new());
    }

    let prev_lines: Vec<&str> = previous.lines().collect();
    let curr_lines: Vec<&str> = current.lines().collect();

    if prev_lines.len() < 2 || curr_lines.len() < 2 {
        return (current.to_string(), Vec::new());
    }

    // Take the last `overlap_lines` from previous as the overlap suffix
    let overlap_count = overlap_lines.min(prev_lines.len());
    let overlap_suffix = &prev_lines[prev_lines.len() - overlap_count..];

    // Find a unique match of the overlap suffix at the start of current
    let mut match_positions = Vec::new();
    for start in 0..curr_lines.len().saturating_sub(overlap_count - 1) {
        let candidate = &curr_lines[start..start + overlap_count];
        if candidate == overlap_suffix {
            match_positions.push(start);
        }
    }

    match match_positions.len() {
        1 => {
            // Unique match found — merge
            let match_pos = match_positions[0];
            let new_start = match_pos + overlap_count;
            if new_start >= curr_lines.len() {
                // No new content beyond overlap
                (previous.to_string(), Vec::new())
            } else {
                let new_lines = &curr_lines[new_start..];
                let mut merged = previous.to_string();
                if !merged.ends_with('\n') {
                    merged.push('\n');
                }
                merged.push_str(&new_lines.join("\n"));
                (merged, Vec::new())
            }
        }
        0 => {
            // No match — resync with full current content
            (
                current.to_string(),
                vec![FidelityIssue::OverlapResync],
            )
        }
        _ => {
            // Ambiguous (multiple matches) — resync
            (
                current.to_string(),
                vec![FidelityIssue::OverlapResync],
            )
        }
    }
}

/// Capture all panes in a session with options.
pub async fn capture_session_with_options(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
    opts: &CaptureOptions,
) -> Result<HashMap<PaneAddress, CaptureResult>> {
    let panes = discovery::list_panes_in_session(transport, socket, session).await?;
    let mut result = HashMap::new();
    for pane in panes {
        let target = pane.address.to_tmux_target();
        let cr = capture_pane_with_options(transport, socket, &target, opts).await?;
        result.insert(pane.address, cr);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;
    use regex::Regex;

    #[tokio::test]
    async fn capture_pane_basic() {
        let mock = MockTransport::new()
            .with_response("capture-pane", "line1\nline2\nline3\n");
        let transport = TransportKind::Mock(mock);
        let result = capture_pane(&transport, None, "build:0.0").await.unwrap();
        assert_eq!(result, "line1\nline2\nline3\n");
    }

    #[tokio::test]
    async fn sample_text_last_lines() {
        let mock = MockTransport::new()
            .with_response("capture-pane", "line1\nline2\nline3\nline4\nline5\n");
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::LastLines(5);
        let result = sample_text(&transport, None, "build:0.0", &query)
            .await
            .unwrap();
        assert!(result.contains("line1"));
    }

    #[tokio::test]
    async fn sample_text_until_pattern() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "old stuff\n$ prompt\ncommand output\nmore output\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::Until {
            pattern: Regex::new(r"^\$ ").unwrap(),
            max_lines: 100,
        };
        let result = sample_text(&transport, None, "build:0.0", &query)
            .await
            .unwrap();
        assert!(result.starts_with("$ prompt"));
        assert!(result.contains("command output"));
    }

    #[tokio::test]
    async fn sample_text_last_lines_until() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "irrelevant\n--- marker ---\nwanted line 1\nwanted line 2\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::LastLinesUntil {
            lines: 100,
            stop_pattern: Regex::new(r"--- marker ---").unwrap(),
        };
        let result = sample_text(&transport, None, "test:0.0", &query)
            .await
            .unwrap();
        assert!(result.starts_with("--- marker ---"));
        assert!(result.contains("wanted line 1"));
    }

    // --- Normalization unit tests ---

    #[test]
    fn normalize_screen_stable_trims_trailing_spaces() {
        let raw = "hello   \nworld  \n\n\n";
        let result = normalize_screen_stable(raw);
        assert_eq!(result, "hello\nworld\n");
    }

    #[test]
    fn normalize_screen_stable_preserves_ansi() {
        let raw = "\x1b[32mgreen text\x1b[0m   \nnormal   \n";
        let result = normalize_screen_stable(raw);
        assert_eq!(result, "\x1b[32mgreen text\x1b[0m\nnormal\n");
    }

    #[test]
    fn normalize_screen_stable_canonical_line_endings() {
        let raw = "line1\r\nline2\r\n";
        let result = normalize_screen_stable(raw);
        assert_eq!(result, "line1\nline2\n");
    }

    #[test]
    fn normalize_screen_stable_empty_input() {
        assert_eq!(normalize_screen_stable(""), "");
        assert_eq!(normalize_screen_stable("\n\n\n"), "");
        assert_eq!(normalize_screen_stable("   \n  \n"), "");
    }

    #[test]
    fn strip_ansi_removes_csi() {
        let input = "\x1b[32mgreen\x1b[0m normal";
        assert_eq!(strip_ansi(input), "green normal");
    }

    #[test]
    fn strip_ansi_removes_osc_bel() {
        let input = "\x1b]0;title\x07rest";
        assert_eq!(strip_ansi(input), "rest");
    }

    #[test]
    fn strip_ansi_removes_osc_st() {
        let input = "\x1b]0;title\x1b\\rest";
        assert_eq!(strip_ansi(input), "rest");
    }

    #[test]
    fn strip_ansi_removes_simple_escape() {
        let input = "\x1b(Btext";
        assert_eq!(strip_ansi(input), "text");
    }

    #[test]
    fn strip_ansi_preserves_newlines_and_tabs() {
        let input = "line1\n\tindented\nline3";
        assert_eq!(strip_ansi(input), "line1\n\tindented\nline3");
    }

    #[test]
    fn strip_ansi_removes_control_chars() {
        let input = "hello\x01\x02world";
        assert_eq!(strip_ansi(input), "helloworld");
    }

    #[test]
    fn strip_ansi_complex_sequence() {
        let input = "\x1b[1;31mbold red\x1b[0m \x1b[48;5;234mbg\x1b[0m";
        assert_eq!(strip_ansi(input), "bold red bg");
    }

    #[test]
    fn normalize_plain_text_strips_ansi_and_trims() {
        let raw = "\x1b[32mhello\x1b[0m   \n\x1b[1mworld\x1b[0m  \n\n";
        let result = normalize_plain_text(raw);
        assert_eq!(result, "hello\nworld\n");
    }

    // --- Options-based capture tests ---

    #[tokio::test]
    async fn capture_with_options_raw() {
        let mock = MockTransport::new()
            .with_response("capture-pane", "line1\nline2\n");
        let transport = TransportKind::Mock(mock);
        let opts = CaptureOptions::default();
        let result =
            capture_pane_with_options(&transport, None, "test:0.0", &opts)
                .await
                .unwrap();
        assert_eq!(result.text, "line1\nline2\n");
        assert!(result.raw_text.is_none());
        assert!(!result.fidelity.degraded);
    }

    #[tokio::test]
    async fn capture_with_options_screen_stable() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "\x1b[32mhello\x1b[0m   \nworld  \n\n",
        );
        let transport = TransportKind::Mock(mock);
        let opts = CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable);
        let result =
            capture_pane_with_options(&transport, None, "test:0.0", &opts)
                .await
                .unwrap();
        // Normalized: trailing spaces trimmed, trailing empty lines removed
        assert_eq!(result.text, "\x1b[32mhello\x1b[0m\nworld\n");
        // raw_text preserves the original
        assert!(result.raw_text.is_some());
        assert!(result.raw_text.unwrap().contains("   \n"));
    }

    #[tokio::test]
    async fn capture_with_options_plain_text() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "\x1b[32mhello\x1b[0m   \nworld  \n\n",
        );
        let transport = TransportKind::Mock(mock);
        let opts = CaptureOptions::with_mode(CaptureNormalizeMode::PlainText);
        let result =
            capture_pane_with_options(&transport, None, "test:0.0", &opts)
                .await
                .unwrap();
        assert_eq!(result.text, "hello\nworld\n");
        // PlainText does not produce raw_text
        assert!(result.raw_text.is_none());
    }

    #[tokio::test]
    async fn capture_with_options_history() {
        let mock = MockTransport::new()
            .with_response("capture-pane", "scrollback\nvisible\n");
        let transport = TransportKind::Mock(mock);
        let opts = CaptureOptions::with_history(-100);
        let result =
            capture_pane_with_options(&transport, None, "test:0.0", &opts)
                .await
                .unwrap();
        assert!(result.text.contains("scrollback"));
    }

    #[tokio::test]
    async fn sample_text_with_options_plain_text() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "\x1b[32mold\x1b[0m\n\x1b[31m$ prompt\x1b[0m\noutput\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::Until {
            pattern: Regex::new(r"^\$ ").unwrap(),
            max_lines: 100,
        };
        let opts = CaptureOptions::with_mode(CaptureNormalizeMode::PlainText);
        let result =
            sample_text_with_options(&transport, None, "test:0.0", &query, &opts, None)
                .await
                .unwrap();
        // ANSI stripped, pattern matching works on plain text
        assert!(result.text.starts_with("$ prompt"));
        assert!(result.text.contains("output"));
        // No ANSI in result
        assert!(!result.text.contains("\x1b"));
    }

    #[test]
    fn fidelity_clean_path() {
        let cr = CaptureResult {
            text: "hello".to_string(),
            raw_text: None,
            fidelity: OutputFidelity::clean(),
        };
        assert!(!cr.fidelity.degraded);
        assert!(cr.fidelity.issues.is_none());
    }

    // --- Overlap dedup tests (Phase 1.9b) ---

    #[test]
    fn overlap_dedup_unique_match() {
        let previous = "line1\nline2\nline3\nline4\nline5";
        let current = "line4\nline5\nline6\nline7";
        let (merged, issues) = overlap_deduplicate(previous, current, 2);
        assert!(issues.is_empty());
        assert_eq!(merged, "line1\nline2\nline3\nline4\nline5\nline6\nline7");
    }

    #[test]
    fn overlap_dedup_no_match_resync() {
        let previous = "line1\nline2\nline3";
        let current = "lineA\nlineB\nlineC";
        let (merged, issues) = overlap_deduplicate(previous, current, 2);
        assert_eq!(issues, vec![FidelityIssue::OverlapResync]);
        // Falls back to current content
        assert_eq!(merged, "lineA\nlineB\nlineC");
    }

    #[test]
    fn overlap_dedup_ambiguous_resync() {
        // Repeated lines create multiple match positions → ambiguous
        let previous = "repeat\nrepeat\nrepeat";
        let current = "repeat\nrepeat\nrepeat\nnew_line";
        let (merged, issues) = overlap_deduplicate(previous, current, 2);
        // "repeat\nrepeat" appears at position 0 and position 1 → ambiguous
        assert_eq!(issues, vec![FidelityIssue::OverlapResync]);
        assert_eq!(merged, "repeat\nrepeat\nrepeat\nnew_line");
    }

    #[test]
    fn overlap_dedup_insufficient_overlap_lines() {
        // Less than 2 overlap lines → no dedup attempted
        let previous = "line1\nline2";
        let current = "line2\nline3";
        let (merged, issues) = overlap_deduplicate(previous, current, 1);
        assert!(issues.is_empty());
        assert_eq!(merged, "line2\nline3");
    }

    #[test]
    fn overlap_dedup_empty_inputs() {
        let (merged, issues) = overlap_deduplicate("", "new content", 3);
        assert!(issues.is_empty());
        assert_eq!(merged, "new content");
    }

    #[test]
    fn overlap_dedup_no_new_content() {
        let previous = "line1\nline2\nline3";
        let current = "line2\nline3";
        let (merged, issues) = overlap_deduplicate(previous, current, 2);
        assert!(issues.is_empty());
        // No new content beyond overlap
        assert_eq!(merged, "line1\nline2\nline3");
    }

    #[test]
    fn overlap_dedup_large_overlap() {
        let previous = "a\nb\nc\nd\ne";
        let current = "c\nd\ne\nf\ng";
        let (merged, issues) = overlap_deduplicate(previous, current, 3);
        assert!(issues.is_empty());
        assert_eq!(merged, "a\nb\nc\nd\ne\nf\ng");
    }

    #[tokio::test]
    async fn sample_text_with_options_overlap_dedup() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "line3\nline4\nline5\nline6\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::LastLines(4);
        let opts = CaptureOptions {
            overlap_lines: 2,
            ..Default::default()
        };
        let previous = "line1\nline2\nline3\nline4";
        let result =
            sample_text_with_options(&transport, None, "t:0.0", &query, &opts, Some(previous))
                .await
                .unwrap();
        // overlap_deduplicate should merge: previous + new unique lines
        assert!(result.text.contains("line1"));
        assert!(result.text.contains("line5"));
        assert!(result.text.contains("line6"));
        assert!(!result.fidelity.degraded);
    }

    #[tokio::test]
    async fn sample_text_with_options_overlap_resync() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "completely\ndifferent\ncontent\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::LastLines(3);
        let opts = CaptureOptions {
            overlap_lines: 2,
            ..Default::default()
        };
        let previous = "line1\nline2\nline3";
        let result =
            sample_text_with_options(&transport, None, "t:0.0", &query, &opts, Some(previous))
                .await
                .unwrap();
        // No overlap match → OverlapResync, falls back to current content
        assert!(result.fidelity.degraded);
        let issues = result.fidelity.issues.unwrap();
        assert!(issues.contains(&FidelityIssue::OverlapResync));
    }

    // --- Reflow-aware capture tests (Phase 1.9b) ---

    #[tokio::test]
    async fn capture_with_reflow_detection_clean() {
        // Pre and post snapshots are identical → clean fidelity
        let mock = MockTransport::new()
            .with_response("capture-pane", "content\n")
            .with_response("list-clients", "200\t50\tbuild\n")
            .with_response("display-message", "80\t24\t100\t2000\n");
        let transport = TransportKind::Mock(mock);
        let opts = CaptureOptions {
            detect_reflow: true,
            ..Default::default()
        };
        let result =
            capture_pane_with_options(&transport, None, "test:0.0", &opts)
                .await
                .unwrap();
        assert_eq!(result.text, "content\n");
        // MockTransport returns same response for same command prefix,
        // so pre/post snapshots match → clean fidelity
        assert!(!result.fidelity.degraded);
    }
}
