# History Subsystem — Per-Source Coherent Context

<!-- Changelog
| Date       | Who     | Summary |
|------------|---------|---------|
| 2026-05-27 | @gpt55-337-og | Documented issue #337 boundary: Fleet can build generic target-derived timeline filters/scopes, while retained history/timeline storage remains in OutputBus/HistoryHandle consumers. |
| 2026-03-25 | @claude | Initial design: coalescing, per-source rendering, per-source budgets |
-->

## Problem

`HistoryHandle` stores events in arrival order as individual entries. When
monitoring multiple panes, output from different panes interleaves based on
tmux `%output` frame timing:

```
host:session(%5)> Compiling motlie v0.1
host:session(%6)> test_login PASSED
host:session(%5)> Compiling russh v0.46
host:session(%6)> test_logout PASSED
host:session(%5)> Finished dev profile
```

An external agent reading this as LLM context cannot easily reconstruct what
happened in each pane. The source labels help, but the interleaving breaks
logical sequences. Trimming compounds the problem — dropping the oldest entry
globally can lose the beginning of one pane's output while keeping the middle
of another's.

### Root causes

1. **No coalescing**: rapid `%output` frames from the same pane become
   individual entries that get interleaved with other panes before trimming.
2. **Single timeline**: all sources share one `VecDeque`, so rendering always
   produces interleaved output.
3. **Global trimming**: oldest entries are dropped regardless of source, so a
   noisy pane can evict a quiet pane's entire context.

## Desired output

An agent wants per-source coherent sections:

```
=== build(%5) ===
$ cargo build
   Compiling motlie v0.1
   Compiling russh v0.46
   Finished dev profile

=== test(%6) ===
$ pytest
test_login PASSED
test_logout PASSED
2 passed, 0 failed
```

Each pane is a contiguous block. The agent can reason about each independently.

## Design

Three phases, each independently shippable:

### Fleet Target Filter Boundary

Issue #337 adds `ResolvedFleetTarget::sink_filter()` and
`Fleet::timeline_options_for_targets(...)` as convenience glue. These APIs only
derive `SinkFilter` values from resolved host/session targets. `HistoryHandle`
and `OutputBus` remain the owners of retained output, rendering, gaps, and
discontinuity handling; `Fleet` does not own timeline storage or higher-level
orchestration concepts.

### Phase 1: Coalesce consecutive same-source chunks

**Scope**: `HistoryHandle` background task in `sink.rs`.

When consecutive `%output` frames arrive from the same source (same pane_id),
append to the last entry's text instead of creating a new entry. This
collapses rapid same-source frames into a single entry, reducing interleaving.

```
Before: 10 rapid frames from pane A = 10 entries (interleaved with pane B)
After:  10 rapid frames from pane A = 1 entry (text concatenated)
```

The `source_changed` flag already tracks this. The change is to append
rather than push when `source_changed == false`.

Impact: fewer entries, less interleaving, no API change.

### Phase 2: Per-source render mode

**Scope**: `HistoryOptions`, `HistoryState`, `PollHistory` in `sink.rs`.

Add `RenderMode` enum to `HistoryOptions`:

```rust
pub enum RenderMode {
    /// Entries in arrival order with source labels on source transitions.
    Interleaved,
    /// Group entries by source, render each source as a labeled section.
    PerSource,
}
```

`PerSource` rendering collects entries by source key and renders each as a
contiguous section. Source sections appear in the order their first entry
arrived (insertion order).

```rust
fn render_per_source(&self) -> String {
    let mut sections: Vec<(String, Vec<&HistoryEntry>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    for entry in &self.entries {
        let key = entry.source_key();
        if let Some(&idx) = index.get(&key) {
            sections[idx].1.push(entry);
        } else {
            index.insert(key.clone(), sections.len());
            sections.push((key, vec![entry]));
        }
    }

    let mut result = String::new();
    for (source, entries) in &sections {
        result.push_str(&format!("=== {} ===\n", source));
        for entry in entries {
            // Render text only, no per-line source labels
            result.push_str(&entry.text_content());
        }
        result.push('\n');
    }
    result
}
```

Default remains `Interleaved` for backward compatibility.

`PollHistory` gets the same treatment via `push_text_for_source(source, text)`
and a matching `RenderMode` option.

### Phase 3: Per-source budgets

**Scope**: `HistoryState` internals in `sink.rs`.

Replace the single `VecDeque<HistoryEntry>` with per-source windows:

```rust
struct SourceWindow {
    entries: VecDeque<HistoryEntry>,
    rendered_chars: usize,
    omitted_entries: usize,
}

struct HistoryState {
    per_source: HashMap<String, SourceWindow>,
    source_order: Vec<String>,
    max_entries_per_source: usize,
    max_render_chars_per_source: usize,
    global_max_render_chars: usize,
    label_format: LabelFormat,
    include_omission_marker: bool,
    render_mode: RenderMode,
}
```

Each source trims independently against `max_entries_per_source` and
`max_render_chars_per_source`. A global character cap can still apply across
all sources.

This prevents a noisy pane from evicting a quiet pane's context.

`HistoryOptions` gains:

```rust
pub struct HistoryOptions {
    pub max_entries: usize,                    // per-source entry cap
    pub max_render_chars: usize,               // per-source char cap
    pub global_max_render_chars: usize,        // total char cap across sources
    pub label_format: LabelFormat,
    pub render_mode: RenderMode,
    pub include_omission_marker: bool,
}
```

For backward compatibility, `global_max_render_chars` defaults to 0 (unlimited)
and the existing `max_entries` / `max_render_chars` become per-source limits.

### Phase 4: Pluggable content filtering

**Scope**: `ContentFilter` trait + built-in impls. Prototype in example first,
fold into library once validated.

**Problem**: Raw pane content includes TUI chrome (spinners, status bars,
box-drawing separators, progress indicators) that wastes context window.
Different programs produce different kinds of noise — a plain shell needs
minimal filtering while Claude Code needs aggressive chrome removal. The
filtering strategy should be selectable per source.

**Design**: Two-level filtering via a trait:

```rust
/// Filters raw pane content into meaningful text for history accumulation.
pub trait ContentFilter: Send + Sync {
    /// Filter a single line. Returns Some(cleaned_line) to keep, None to discard.
    fn filter_line(&self, line: &str) -> Option<String>;

    /// After collecting new lines through filter_line, decide if the batch
    /// is meaningful enough to record. Prevents flushing spinner-only updates.
    fn is_meaningful_batch(&self, new_lines: &[String]) -> bool;
}
```

**Built-in implementations**:

| Strategy | Line filtering | Batch filtering | Use case |
|----------|---------------|-----------------|----------|
| `RawFilter` | Strip ANSI only | Always meaningful | Debug, raw capture |
| `ShellFilter` | Strip ANSI, drop empty lines | ≥1 non-empty line | Plain shell sessions |
| `AgentFilter` | Strip ANSI, drop spinners/status/box-drawing/progress | ≥1 content line (not just chrome) | Claude Code, Codex, agent TUIs |

**`AgentFilter` details**:

Chrome detection heuristics (lines discarded):
- Lines that are all box-drawing characters (`─━═│┃┌┐└┘` etc.)
- Lines matching known status patterns: `esc to interrupt`, `? for shortcuts`,
  `background terminals running`, `/ps to view`, spinner prefixes (`· ✻ ✶ ✽ ✢ ✳`)
- Lines that are only whitespace after ANSI stripping
- Short progress fragments (e.g. `Working (3s • esc to interrupt)`)

Batch meaningfulness:
- At least N content lines (lines that pass line filtering) must be present
- Configurable threshold (default: 1 meaningful line)

**Wiring into SourceAccumulator**:

```rust
struct SourceAccumulator<F: ContentFilter> {
    filter: F,
    // ... existing fields
}
```

Each source can have a different filter. The example CLI exposes `--filter`
with values `raw`, `shell`, `agent`. Different sessions can use different
filters if needed.

**Per-source filter assignment**:

For the common case where all sources use the same filter, the CLI flag
applies globally. For advanced use, per-source assignment could be added
later (e.g. `--filter-tmux-claude=agent --filter-tmux=shell`).

## Non-goals

- Timestamp-based ordering or merge-sort across sources. Arrival order within
  each source is sufficient.
- Cross-source correlation (e.g. "pane A's build finished, then pane B's
  tests started"). This is the agent's job, not the library's.

## Testing

- Coalescing: verify that rapid same-source frames produce fewer entries
  than individual pushes.
- Per-source render: verify that interleaved input produces grouped output
  with sections in insertion order.
- Per-source budgets: verify that a noisy source doesn't evict a quiet
  source's entries.
- Adversarial: single-source monitoring still works identically.
- Backward compat: `RenderMode::Interleaved` produces identical output
  to the pre-change implementation.

## Experiment Results

Prototyped in `history_demo.rs` (not library code). Transcripts in `/tmp/`.

### Best configuration found

```
--mode tail --render-mode per-source --filter codex --policy prompt
--chars 6000 --entries 40
```

- **CodexFilter**: strips Codex spinner words (Kneading, Garnishing, Whirring,
  Hatching, etc.), box-drawing, status bar chrome. `is_meaningful_batch` requires
  at least one non-prompt content line to prevent empty flushes.
- **PromptBoundaryPolicy**: waits for agent prompt (`›` for Codex, `❯` for
  Claude) before flushing. Falls back to 30s max_wait. Produces complete
  per-turn chunks.
- **Per-source rendering**: each session rendered as `=== session ===` section.

### Experiment progression

| Run | Lines | Turns | Filter | Policy | Issues |
|-----|-------|-------|--------|--------|--------|
| v1 | 76 | 4 | none | line(3)/time(5s) | tmux session missing, TUI chrome |
| v2 | 902 | 4 | none | line(3)/time(5s) | noisy, repeated tail excerpts |
| v3 | 62 | 4 | codex | line(3)/time(5s) | too aggressive, responses filtered |
| v4 | 1272 | 4 | codex | line(3)/time(5s) | both sessions captured, noise reduced |
| v5 | 168 | 3 | codex | prompt(30s) | excellent coherence, complete turns |
| exp1 | 219 | 6 | codex | prompt(30s) | long horizon works, some empty flushes |
| exp2 | 122 | 4 | codex+tuned | prompt(30s) | minimal noise, clean per-turn output |

### Key insights

1. **PromptBoundary >> LineCount** for agent TUI sessions. Waiting for the
   agent's prompt produces complete, coherent turns instead of fragmented
   line-at-a-time drips.
2. **Agent-specific filters are essential**. Codex and Claude Code have
   different spinner words, status patterns, and prompt characters.
3. **`is_meaningful_batch` prevents empty flushes**. Without it, the max_wait
   fallback produces question-only entries with no answer.
4. **Set-based diff + filter** works better than tail-excerpt diff for TUI
   programs that redraw the whole screen.
5. **TUI table rendering** is a remaining challenge — tables built incrementally
   produce multiple partial renders as separate entries.

### Heuristic filter experiment (exp3)

Replaced dictionary-based filters with structural heuristics:

| Heuristic | What it detects | Method |
|-----------|----------------|--------|
| `is_spinner_line` | Unicode symbol + space + ellipsis text | Structural: non-ASCII first char, space, rest contains `…` |
| `is_box_drawing_line` | All box-drawing chars | Character set check |
| `is_status_bar` | Multiple `·`/`•` separated segments | Segment count ≥ 2 |
| `is_affordance_hint` | Short UI hints | Length < 40 + low alpha ratio, or contains "esc to"/"ctrl+"/"tab to" |
| `is_context_indicator` | "N% left/context/remaining" | Substring match on percentage patterns |
| `is_bare_prompt` | Lone prompt char | Exact match |

The only agent-specific parameter is `prompt_char` (❯ for Claude, › for Codex).
Spinner word dictionaries completely eliminated.

| Run | Lines | Turns | Filter | Policy | Issues |
|-----|-------|-------|--------|--------|--------|
| exp3 | 204 | 4 | heuristic | prompt(30s) | clean, no spinner leaks, minor tool-use status leak |

Result: heuristic filters match or exceed dictionary-based quality while being
future-proof against agent vocabulary changes.

## Phase 5: Library folding plan

Move validated prototypes from `history_demo.rs` into the core library.

### New module: `filter.rs`

```rust
pub trait ContentFilter: Send + Sync {
    fn filter_line(&self, line: &str) -> Option<String>;
    fn is_meaningful_batch(&self, lines: &[String]) -> bool;
    fn is_prompt(&self, line: &str) -> bool;
}
```

Heuristic detection functions (no dictionaries):

| Function | Detection method | Stability |
|----------|-----------------|-----------|
| `is_spinner_line` | Non-ASCII symbol + space + `…` in text | High — structural, agent-independent |
| `is_box_drawing_line` | All chars in box-drawing set | High — character set is stable |
| `is_status_bar` | ≥2 `·`/`•` separated segments | High — universal TUI pattern |
| `is_affordance_hint` | Short line + low alpha ratio or "esc to"/"ctrl+"/"tab to" | Medium — stable UI conventions |
| `is_context_indicator` | "N% left/context/remaining" | Medium — common model UI pattern |
| `is_bare_prompt` | Lone prompt char (❯ › $ %) | High — fundamental |

Built-in filter types:

| Type | Config | Use case |
|------|--------|----------|
| `RawFilter` | None | Debug, raw capture, build logs. Strips ANSI, keeps everything. |
| `ShellFilter` | None | Plain shells, CI, deploy scripts. Drops empty lines. Prompt = `$`/`%`/`#`. |
| `AgentTuiFilter` | `prompt_char: char` | Any agent TUI. One parameter: the prompt character. `AgentTuiFilter::claude_code()` = `❯`, `AgentTuiFilter::codex()` = `›`. Custom agents supply their own prompt char. |

### Extend `sink.rs`: FlushPolicy + SourceAccumulator

```rust
pub enum FlushPolicy {
    LineCount { min_lines: usize, max_wait: Duration },
    Idle { idle_duration: Duration, max_wait: Duration },
    PromptBoundary { max_wait: Duration, min_content_lines: usize },
}
```

| Policy | Parameters | Use case |
|--------|------------|----------|
| `LineCount` | `min_lines=3`, `max_wait=10s` | Build output, log tailing, high-throughput streams. Regular cadence — flush every N lines or T seconds. |
| `Idle` | `idle_duration=3s`, `max_wait=15s` | CI jobs, test runners, deploy scripts. Flush when output pauses (burst ends). Captures complete test results or build phases. |
| `PromptBoundary` | `max_wait=30s`, `min_content_lines=1` | Interactive agents (Claude, Codex, shells). Waits for prompt to reappear after content. Produces complete Q&A turns. |

`SourceAccumulator`: caller-driven per-source accumulator.

```rust
pub struct SourceAccumulator {
    previous: HashMap<PaneAddress, String>,
    pending_lines: Vec<String>,
    last_flush: Instant,
    last_change: Instant,
    saw_prompt: bool,
    filter: Box<dyn ContentFilter>,
    policy: FlushPolicy,
}

impl SourceAccumulator {
    pub fn new(name, baseline, filter, policy) -> Self;
    pub fn ingest(&mut self, current: &HashMap<PaneAddress, String>) -> Option<String>;
    pub fn flush_remaining(&mut self) -> Option<String>;
}
```

`diff_new_lines(previous, current, filter)`: set-based diff with per-line
filtering. Moved to `filter.rs` or `capture.rs`.

### Recommended combinations by scenario

| Scenario | Filter | Policy | Render | Notes |
|----------|--------|--------|--------|-------|
| Multi-agent supervisor | `AgentTuiFilter` | `PromptBoundary(30s)` | `PerSource` | Validated configuration. Coherent per-turn sections. |
| Build log watcher | `ShellFilter` | `Idle(3s)` | `PerSource` | Each build target grouped. Flush between compiler phases. |
| Test runner dashboard | `ShellFilter` | `Idle(2s)` | `PerSource` | Test output by session. Flush after each test batch. |
| Live debug / raw watch | `RawFilter` | `LineCount(10)` | `Interleaved` | Rapid updates, raw content, temporal ordering. |
| Single agent deep-watch | `AgentTuiFilter` | `PromptBoundary(30s)` | `Interleaved` | One agent, temporal ordering. Cleaned-up terminal log. |
| Heterogeneous mix | Per-source: `AgentTuiFilter` for agents, `ShellFilter` for shells | `PromptBoundary` for agents, `Idle` for shells | `PerSource` | Different filters per pane type. |

### What stays in the example after folding

The example becomes ~200 lines of wiring:
- Parse CLI args
- Create filter + policy per source
- Construct `SourceAccumulator` per source from library types
- Poll loop: `target.capture_all()` → `acc.ingest()` → `history.push_text_for_source()`
- Print rolling context

All filter logic, heuristics, flush policies, and accumulation move to the library.

### Re-exports from `lib.rs`

```rust
pub use filter::{
    ContentFilter, RawFilter, ShellFilter, AgentTuiFilter,
    is_spinner_line, is_box_drawing_line, is_status_bar,
    is_tui_chrome, clean_line, diff_new_lines,
};
pub use sink::{FlushPolicy, SourceAccumulator};
// Existing: PollHistory, RenderMode, HistoryHandle, etc.
```
