//! Content filtering for TUI pane output (DC33 Phase 4).
//!
//! Heuristic-based: detects TUI chrome by structural patterns (symbol
//! positions, line composition, segment counts) rather than dictionaries
//! of specific words. Agent-specific knowledge is limited to the prompt
//! character, which is stable and version-coupled.

use crate::capture::strip_ansi;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ContentFilter trait
// ---------------------------------------------------------------------------

/// Filters raw pane content into meaningful text for history accumulation.
///
/// Two levels of filtering:
/// - `filter_line`: per-line decision (keep or discard)
/// - `is_meaningful_batch`: per-chunk decision (worth recording?)
/// - `is_prompt`: semantic hint for flush policies
pub trait ContentFilter: Send + Sync {
    /// Filter a single line. Returns `Some(cleaned)` to keep, `None` to discard.
    fn filter_line(&self, line: &str) -> Option<String>;

    /// After collecting filtered lines, decide if the batch is worth recording.
    /// Prevents flushing spinner-only or prompt-only updates.
    fn is_meaningful_batch(&self, lines: &[String]) -> bool;

    /// Semantic hint: does this line indicate the agent is waiting for user input?
    /// Used by [`FlushPolicy::PromptBoundary`](crate::FlushPolicy) to detect turn boundaries.
    fn is_prompt(&self, _line: &str) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Structural heuristics (no dictionaries)
// ---------------------------------------------------------------------------

/// Strip ANSI escape sequences and normalize a raw line.
/// Returns `None` if the line is empty after cleaning.
pub fn clean_line(line: &str) -> Option<String> {
    let clean = strip_ansi(line).replace('\r', "");
    let trimmed = clean.trim_end().to_string();
    if trimmed.trim().is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// True if the line is composed entirely of box-drawing / separator characters.
pub fn is_box_drawing_line(trimmed: &str) -> bool {
    trimmed.len() > 3
        && trimmed
            .chars()
            .all(|c| "─━═│┃┌┐└┘├┤┬┴┼╭╮╰╯-=+|".contains(c))
}

/// True if the line is just a bare prompt character with no content.
pub fn is_bare_prompt(trimmed: &str) -> bool {
    trimmed == "❯" || trimmed == "›" || trimmed == "$" || trimmed == "%"
}

/// Spinner line: non-ASCII symbol at position 0, followed by a space, then
/// text containing an ellipsis (`…`). Catches any spinner word without
/// needing a dictionary.
///
/// Matches: `· Thinking…`, `✻ Kneading…`, `⏺ Searching for 1 pattern…`
pub fn is_spinner_line(trimmed: &str) -> bool {
    let mut chars = trimmed.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    // Must be a non-ASCII, non-alphanumeric symbol (Unicode symbols like ·✻✶⏺•)
    if first.is_ascii() || first.is_alphanumeric() {
        return false;
    }
    // Must be followed by a space
    if chars.next() != Some(' ') {
        return false;
    }
    // Rest must contain ellipsis — the universal "in progress" indicator
    let rest = &trimmed[first.len_utf8() + 1..];
    rest.contains('…')
}

/// Status bar: multiple `·`-separated or `•`-separated short segments.
///
/// Matches: `9 background terminals running · /ps to view · /stop to close`
pub fn is_status_bar(trimmed: &str) -> bool {
    let sep_count = trimmed.matches(" · ").count() + trimmed.matches(" • ").count();
    sep_count >= 2
}

/// Short affordance hint: lines under 40 chars that are mostly non-alpha,
/// or lines containing universal TUI affordance patterns.
pub fn is_affordance_hint(trimmed: &str) -> bool {
    if trimmed.len() > 40 || trimmed.len() < 3 {
        return false;
    }
    let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let total = trimmed.chars().count();
    if total > 0 && (alpha_count as f64 / total as f64) < 0.5 {
        return true;
    }
    let affordances = ["esc to", "ctrl+", "tab to", "? for"];
    affordances
        .iter()
        .any(|a| trimmed.to_lowercase().contains(a))
}

/// Context/model indicator: lines with percentage + "left"/"context"/"remaining".
pub fn is_context_indicator(trimmed: &str) -> bool {
    let lower = trimmed.to_lowercase();
    lower.contains("% left") || lower.contains("% context") || lower.contains("% remaining")
}

/// Combined heuristic: is this line TUI chrome?
pub fn is_tui_chrome(trimmed: &str) -> bool {
    is_box_drawing_line(trimmed)
        || is_bare_prompt(trimmed)
        || is_spinner_line(trimmed)
        || is_status_bar(trimmed)
        || is_affordance_hint(trimmed)
        || is_context_indicator(trimmed)
}

/// Extract new lines from current pane content vs previous.
/// Uses multiset-based diff (preserves multiplicity) then applies
/// the content filter to each line.
pub fn diff_new_lines(previous: &str, current: &str, filter: &dyn ContentFilter) -> Vec<String> {
    // Build a multiset of previous lines so repeated lines are counted correctly.
    // e.g. if previous has "done" twice, only the third "done" in current is new.
    let mut prev_counts: HashMap<String, usize> = HashMap::new();
    for line in previous.lines() {
        if let Some(cleaned) = clean_line(line) {
            *prev_counts.entry(cleaned).or_insert(0) += 1;
        }
    }

    current
        .lines()
        .filter_map(|line| {
            let cleaned = clean_line(line)?;
            // Decrement the count — if still positive, this line was in previous
            if let Some(count) = prev_counts.get_mut(&cleaned) {
                if *count > 0 {
                    *count -= 1;
                    return None;
                }
            }
            filter.filter_line(line)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Built-in filter implementations
// ---------------------------------------------------------------------------

/// Passes everything through, strips ANSI only.
///
/// Use case: debug capture, build logs, raw transcript dumps.
pub struct RawFilter;

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

/// Plain shell sessions — strips ANSI, drops empty lines.
///
/// Use case: bash/zsh prompts, CI runners, deploy scripts, test output.
pub struct ShellFilter;

impl ContentFilter for ShellFilter {
    fn filter_line(&self, line: &str) -> Option<String> {
        clean_line(line)
    }
    fn is_meaningful_batch(&self, lines: &[String]) -> bool {
        !lines.is_empty()
    }
    fn is_prompt(&self, line: &str) -> bool {
        let t = line.trim();
        t.ends_with('$')
            || t.ends_with('%')
            || t.ends_with('#')
            || t.starts_with("$ ")
            || t.starts_with("% ")
    }
}

/// Agent TUI filter — uses structural heuristics, no word dictionaries.
///
/// The only agent-specific parameter is the prompt character:
/// - `AgentTuiFilter::claude_code()` → `❯`
/// - `AgentTuiFilter::codex()` → `›`
/// - `AgentTuiFilter::new('>')` → custom prompt
///
/// Use case: Claude Code, Codex, Aider, Cursor, or any agent TUI with a
/// known prompt character.
pub struct AgentTuiFilter {
    prompt_char: char,
}

impl AgentTuiFilter {
    /// Create a filter with a custom prompt character.
    pub fn new(prompt_char: char) -> Self {
        Self { prompt_char }
    }

    /// Claude Code sessions (prompt: `❯`).
    pub fn claude_code() -> Self {
        Self { prompt_char: '❯' }
    }

    /// OpenAI Codex sessions (prompt: `›`).
    pub fn codex() -> Self {
        Self { prompt_char: '›' }
    }
}

impl ContentFilter for AgentTuiFilter {
    fn filter_line(&self, line: &str) -> Option<String> {
        let trimmed = clean_line(line)?;
        if is_tui_chrome(&trimmed) {
            return None;
        }
        Some(trimmed)
    }

    fn is_meaningful_batch(&self, lines: &[String]) -> bool {
        let prompt_prefix = format!("{} ", self.prompt_char);
        lines.iter().any(|l| {
            let t = l.trim();
            t.len() > 2
                && !t.starts_with(&prompt_prefix)
                && !t.starts_with("❯ ")
                && !t.starts_with("› ")
        })
    }

    fn is_prompt(&self, line: &str) -> bool {
        let t = line.trim();
        let prefix = format!("{} ", self.prompt_char);
        t.starts_with(&prefix) || t == self.prompt_char.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_detection_structural() {
        assert!(is_spinner_line("· Thinking…"));
        assert!(is_spinner_line("✻ Kneading…"));
        assert!(is_spinner_line("⏺ Searching for 1 pattern…"));
        assert!(is_spinner_line("• Working (3s … more)"));
        assert!(is_spinner_line("✽ Anywordhere…"));
        // Not spinners
        assert!(!is_spinner_line("Searching for pattern"));
        assert!(!is_spinner_line("A normal line"));
        assert!(!is_spinner_line("$ command"));
    }

    #[test]
    fn box_drawing_detection() {
        assert!(is_box_drawing_line("─────────────────"));
        assert!(is_box_drawing_line("━━━━━━━━━━━━━━━━━"));
        assert!(is_box_drawing_line("├──────┼──────┤"));
        assert!(!is_box_drawing_line("--- text ---"));
        assert!(!is_box_drawing_line("ab"));
    }

    #[test]
    fn status_bar_detection() {
        assert!(is_status_bar(
            "9 background terminals running · /ps to view · /stop to close"
        ));
        assert!(is_status_bar("gpt-5.4 default · 23% left · ~/projects"));
        assert!(!is_status_bar("just one · segment"));
        assert!(!is_status_bar("no segments here"));
    }

    #[test]
    fn affordance_hint_detection() {
        assert!(is_affordance_hint("esc to interrupt"));
        assert!(is_affordance_hint("? for shortcuts"));
        assert!(is_affordance_hint("ctrl+o to expand"));
        assert!(!is_affordance_hint(
            "This is a normal long sentence with many words"
        ));
    }

    #[test]
    fn context_indicator_detection() {
        assert!(is_context_indicator("23% left"));
        assert!(is_context_indicator(
            "gpt-5.4 default · 23% context remaining"
        ));
        assert!(!is_context_indicator("normal text"));
    }

    #[test]
    fn bare_prompt_detection() {
        assert!(is_bare_prompt("❯"));
        assert!(is_bare_prompt("›"));
        assert!(is_bare_prompt("$"));
        assert!(!is_bare_prompt("❯ command"));
        assert!(!is_bare_prompt("text"));
    }

    #[test]
    fn agent_tui_filter_keeps_content() {
        let f = AgentTuiFilter::codex();
        assert!(f.filter_line("Some actual content here").is_some());
        assert!(f.filter_line("❯ what is this").is_some());
        assert!(f.filter_line("· Thinking…").is_none());
        assert!(f.filter_line("──────────────").is_none());
    }

    #[test]
    fn agent_tui_filter_prompt_detection() {
        let claude = AgentTuiFilter::claude_code();
        assert!(claude.is_prompt("❯ "));
        assert!(claude.is_prompt("❯"));
        assert!(!claude.is_prompt("› something"));

        let codex = AgentTuiFilter::codex();
        assert!(codex.is_prompt("› "));
        assert!(!codex.is_prompt("❯ something"));
    }

    #[test]
    fn meaningful_batch_requires_non_prompt_content() {
        let f = AgentTuiFilter::codex();
        assert!(!f.is_meaningful_batch(&["› question".to_string()]));
        assert!(
            f.is_meaningful_batch(&["› question".to_string(), "Here is the answer".to_string(),])
        );
    }

    #[test]
    fn diff_new_lines_finds_additions() {
        let f = RawFilter;
        let prev = "line1\nline2\n";
        let curr = "line1\nline2\nline3\n";
        let new = diff_new_lines(prev, curr, &f);
        assert_eq!(new, vec!["line3"]);
    }

    #[test]
    fn diff_new_lines_applies_filter() {
        let f = AgentTuiFilter::codex();
        let prev = "content\n";
        let curr = "content\n· Thinking…\nnew answer\n";
        let new = diff_new_lines(prev, curr, &f);
        assert_eq!(new, vec!["new answer"]);
    }

    #[test]
    fn diff_new_lines_preserves_repeated_lines() {
        let f = RawFilter;
        let prev = "done\nother\n";
        let curr = "done\nother\ndone\n";
        let new = diff_new_lines(prev, curr, &f);
        // The second "done" is genuinely new — must not be dropped
        assert_eq!(new, vec!["done"]);
    }

    #[test]
    fn diff_new_lines_same_content_produces_nothing() {
        let f = RawFilter;
        let prev = "done\ndone\n";
        let curr = "done\ndone\n";
        let new = diff_new_lines(prev, curr, &f);
        assert!(new.is_empty());
    }

    #[test]
    fn shell_filter_prompt_detection() {
        let f = ShellFilter;
        assert!(f.is_prompt("$ "));
        assert!(f.is_prompt("user@host:~$ "));
        assert!(f.is_prompt("% "));
        assert!(!f.is_prompt("not a prompt"));
    }
}
