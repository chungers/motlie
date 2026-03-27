# History Subsystem — Per-Source Coherent Context

<!-- Changelog
| Date       | Who     | Summary |
|------------|---------|---------|
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
