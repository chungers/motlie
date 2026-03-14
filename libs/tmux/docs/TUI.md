# TUI Reliability and Capture Fidelity

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-13 | @claude | Add §TUI Mirror Sink — detailed design for replicating tmux pane content in a ratatui frame. Two approaches analyzed: polling `capture-pane -ep` vs tmux control mode. Crate evaluation, SGR mapping, architecture, and trade-off matrix. |
| 2026-03-10 | @codex | Address PR #65 review feedback: clarify that normalized matching is shell-oriented, full-screen TUI workflows default to `Raw`, screen-buffer reconstruction is a Phase `5+` idea, and fixed geometry is best-effort unless automation is isolated. |

This document defines how to support full-screen TUIs in tmux while reducing
sensitivity to mixed client sizes, reflow, and history overflow.

## Short Answer

- Isolation is **not** the only way to run TUI automation.
- Isolation is the only way to get strong, repeatable determinism.

You can still support mixed-client environments, but the system must treat
captures as best-effort and expose degraded-confidence states.

## Why TUI Is Harder Than Line-Oriented Shell Output

Full-screen TUIs (for example `vim`, `less`, `htop`) are cursor-addressed and
frame-updated. They are not simple append-only line logs.

Key failure modes:

1. Reflow and wrapping changes when different-size clients attach.
2. Cursor/control updates are lossy when reduced to plain normalized text.
3. Scrollback truncation from `history-limit` evicts content permanently.

## Mitigation Strategy

Use a layered strategy instead of a single toggle.

### 1) Split Capture Modes

Use distinct modes:

1. `Line` mode: shell/log-like panes, normalization allowed.
2. `Tui` mode: preserve terminal semantics, avoid destructive normalization.

For `Tui` mode, default to raw/control-preserving capture paths. Normalized matching
paths are shell-oriented and should be treated as unsupported for full-screen TUIs
unless a dedicated terminal-state consumer is introduced later.

### 2) Reconstruct Screen State for TUI

For TUI fidelity, process output as a terminal stream and reconstruct a virtual
screen buffer (cells, cursor, attributes) instead of doing line-only parsing.

This keeps fidelity high even when display updates are cursor-based.

This is not part of the initial delivery plan. It requires a terminal emulation
layer (for example `vte` parsing plus a grid buffer) and is a Phase `5+`
consideration only.

### 3) Stabilize Geometry

Set automation windows to explicit size and manual sizing behavior:

```sh
tmux set-option -w -t <session:window> window-size manual
tmux resize-window -t <session:window> -x 160 -y 48
```

This reduces reflow churn but does not guarantee determinism if mixed clients
keep attaching with different sizes. Reliable determinism still requires
automation isolation (dedicated session/socket or no mixed interactive clients
during the capture/exec window).

### 4) Detect and Gate During Resize Churn

Track pane size (`#{pane_width}`, `#{pane_height}`) during command/capture
windows. If size changes mid-operation:

1. mark capture as degraded, or
2. retry in a quiet period, or
3. fail fast for strict workflows.

### 5) Increase History Capacity

Normalization cannot recover evicted scrollback. Increase history proactively:

```sh
tmux set-option -w -t <session:window> history-limit 200000
# or global for newly created windows:
tmux set-option -g history-limit 200000
```

Note: `history-limit` applies to new windows; existing windows keep their
current limit unless recreated.

### 6) Bind Locking and Execution to the Same Pane Identity

For `exec()`-style operations, resolve the effective pane id first (for
example via `display-message -p '#{pane_id}'`), then:

1. acquire lock keyed by that pane id,
2. execute against that pane id,
3. poll/capture from that same pane id.

This avoids lock-target divergence when active pane focus changes.

## Operational Policy Tiers

### Strict (Deterministic)

1. Dedicated automation session/socket.
2. No mixed interactive clients during exec/capture windows.
3. Fixed geometry + high history limit.

Use for tests, CI, or must-pass agents.

### Guarded (Practical Mixed Use)

1. Mixed clients allowed.
2. Fixed geometry attempted.
3. Resize change detection enabled.
4. Degraded/retry semantics when size churn occurs.

Use for human + automation coexistence where occasional retries are acceptable.

### Best-Effort (Shared Interactive)

1. No isolation guarantee.
2. No strict geometry/control guarantee.
3. Captures are advisory, not deterministic.

Use only when occasional fidelity loss is acceptable.

## Recommended Defaults

1. Default `capture()` behavior should remain explicit about mode.
2. Default to line-oriented normalization only for line-oriented workflows.
3. Default TUI-oriented workflows to `Raw`/terminal-state paths.
4. Always document when output is deterministic vs best-effort.

---

## TUI Mirror Sink

This section designs a **Sink** that replicates a tmux pane's visual content —
including full-screen TUI programs like `vim`, `htop`, or `top` — into a local
ratatui frame. Two approaches are analyzed: polling-based and control-mode-based.

### Problem Statement

A consumer wants to display a live replica of a remote (or local) tmux pane
inside its own terminal UI. The source pane may be running a full-screen TUI
with colors, attributes, cursor-addressed updates, and dynamic content. The
replica must be visually faithful: same characters, same colors, same text
attributes at each cell position.

### Architecture Overview

Both approaches share a common rendering pipeline:

```
tmux pane (vim, htop, shell, etc.)
    │
    ▼
[Source Layer]                ← Approach A or B (see below)
    │
    ▼
ANSI SGR Frame               ← text + SGR escape sequences, one rendered frame
    │
    ▼
[SGR Parser]                 ← extract (char, fg, bg, modifiers) per cell
    │
    ▼
StyledGrid [W × H]           ← intermediate representation
    │
    ▼
[TmuxMirror Widget]          ← maps StyledGrid → ratatui::buffer::Buffer
    │
    ▼
ratatui Terminal::draw()     ← paint to local terminal
```

The two approaches differ only in the **Source Layer** — how frames are
obtained from tmux.

---

### Approach A: Polling `capture-pane -ep`

#### How It Works

Periodically call `capture-pane -ep -t <target>` to get the rendered pane
content with ANSI SGR escape sequences. Parse the SGR sequences into a cell
grid, then render via ratatui.

```
loop {
    raw = target.capture_with_options(ScreenStable_with_escapes)
    geo = target.pane_geometry()
    grid = parse_sgr_frame(raw, geo.pane_width, geo.pane_height)
    terminal.draw(|f| f.render_widget(TmuxMirror(grid), f.area()))
    sleep(poll_interval)
}
```

#### Key Property: No VTE Required

`capture-pane -ep` returns a **pre-rendered frame** — tmux has already resolved
all cursor movement, alternate screen switching, scrolling regions, and partial
updates into a flat grid of styled characters. The output contains **only**:

- Printable characters (the cell content)
- SGR sequences (`ESC [ ... m`) for style changes
- Newlines between rows
- `SI`/`SO` for charset switching (rare)

No cursor positioning sequences (CSI A/B/C/D/H), no alternate screen commands,
no scrolling regions. This means the parser is a focused ~200-line SGR state
machine, not a full terminal emulator.

#### SGR Sequences Emitted by `capture-pane -ep`

| Category | Codes | ratatui Mapping |
|----------|-------|-----------------|
| Reset | `0` | `Style::reset()` |
| Bold | `1` | `Modifier::BOLD` |
| Dim | `2` | `Modifier::DIM` |
| Italic | `3` | `Modifier::ITALIC` |
| Underline | `4` | `Modifier::UNDERLINED` |
| Blink | `5` | `Modifier::SLOW_BLINK` |
| Reverse | `7` | `Modifier::REVERSED` |
| Hidden | `8` | `Modifier::HIDDEN` |
| Strikethrough | `9` | `Modifier::CROSSED_OUT` |
| Fg standard | `30–37` | `Color::Black..White` |
| Fg bright | `90–97` | `Color::DarkGray..LightCyan` |
| Fg 256 | `38;5;N` | `Color::Indexed(N)` |
| Fg 24-bit | `38;2;R;G;B` | `Color::Rgb(R,G,B)` |
| Fg default | `39` | `Color::Reset` |
| Bg standard | `40–47` | `Color::Black..White` |
| Bg bright | `100–107` | `Color::DarkGray..LightCyan` |
| Bg 256 | `48;5;N` | `Color::Indexed(N)` |
| Bg 24-bit | `48;2;R;G;B` | `Color::Rgb(R,G,B)` |
| Bg default | `49` | `Color::Reset` |
| Underline color | `58;5;N`, `58;2;R;G;B` | `underline_color` (feature-gated) |
| Double underline | `42`\* | No direct ratatui flag |
| Curly underline | `43`\* | No direct ratatui flag |
| Dotted underline | `44`\* | No direct ratatui flag |
| Dashed underline | `45`\* | No direct ratatui flag |
| Overline | `53` | No direct ratatui flag |

\*tmux uses non-standard codes 42–45 for extended underline styles (standard
ECMA-48 uses colon-separated `4:2`, `4:3`, etc.). These are uncommon in
practice and can be mapped to `UNDERLINED` as a fallback.

**Reset behavior**: tmux emits `SGR 0` (full reset) on any attribute transition
from set to unset, then re-applies all currently active attributes. The parser
must handle sequences like `\e[0m\e[1;38;5;196m` (reset, then bold + fg 196).

#### What Is Captured vs What Is Lost

| Aspect | Captured | Notes |
|--------|----------|-------|
| Character content | Yes | Full Unicode, wide chars |
| Foreground color | Yes | 256 + 24-bit |
| Background color | Yes | 256 + 24-bit |
| Bold/dim/italic/underline | Yes | Direct Modifier mapping |
| Blink/reverse/strikethrough | Yes | Direct Modifier mapping |
| Cursor position | **No** | Requires separate `display-message -p '#{cursor_x} #{cursor_y}'` |
| Cursor shape/visibility | **No** | Not in capture output |
| Alternate screen state | **Transparent** | capture-pane sees the currently active screen |
| Images (sixel/kitty) | **No** | tmux strips image protocols |
| Extended underline styles | **Degraded** | Non-standard codes; fallback to plain underline |

Cursor position is recoverable via a separate tmux query and can be rendered as
a highlighted cell or block cursor in the ratatui widget.

#### Impact on Current Codebase

**No refactoring required.** This approach uses existing API surface:

- `Target::capture_with_options()` with `CaptureNormalizeMode::ScreenStable`
  (but the raw `-ep` output is needed — use `CaptureResult::raw_text`)
- `Target::pane_geometry()` for dimensions
- `GeometrySnapshot` for fidelity detection

**New code needed:**

| Component | Scope | Location |
|-----------|-------|----------|
| SGR parser | ~200–300 lines | New: `libs/tmux/src/sgr.rs` or in TUI binary |
| `StyledGrid` type | ~50 lines | With SGR parser |
| `TmuxMirror` widget | ~60 lines | TUI binary (`bins/`) |
| Poll loop | ~40 lines | TUI binary |
| Cursor overlay | ~20 lines | Optional, in widget |

**Alternatively**, the `ansi-to-tui` crate (v8.0.1) converts ANSI byte streams
directly into `ratatui::text::Text` via a single `bytes.into_text()` call. This
produces line-oriented `Span` objects (not a cell grid), so it works well for
rendering into a `Paragraph` widget but requires iteration to populate a
`Buffer` cell-by-cell. It handles the full SGR mapping table above except
tmux's non-standard underline codes (42–45) and underline color (58;...).

#### Pros and Cons

| | Pro | Con |
|---|-----|-----|
| **Complexity** | Simple — no new transport, no protocol parsing | — |
| **Refactoring** | Zero — uses existing capture API | — |
| **Fidelity** | High for static/slow content | Misses sub-interval transient states |
| **Latency** | Bounded by poll interval (typ. 50–200ms) | Not real-time |
| **Overhead** | One `capture-pane` + one `display-message` per tick | N tmux commands/sec; acceptable for single pane, scales poorly to many |
| **Cursor** | Requires extra query per frame | Adds one more tmux round-trip |
| **Dependencies** | `vte` or `ansi-to-tui` (small) | — |
| **Level of effort** | ~1–2 days for a working prototype | — |

---

### Approach B: tmux Control Mode

#### How It Works

Attach to the session via `tmux -C attach -t <session>` (or `-CC` to suppress
terminal echo). tmux emits structured notifications on stdout, including
`%output` messages containing the raw byte stream for every pane in the session.
Feed that byte stream through a virtual terminal emulator (VTE) to maintain a
screen buffer, then render that buffer via ratatui.

```
[tmux -CC attach -t session]
    │  stdout: %output %5 \033[1;31mhello\033[0m\r\n
    │          %layout-change @0 ...
    │          %window-pane-changed @0 %5
    ▼
[Control Mode Parser]        ← demux by pane_id, decode octal escapes
    │
    ▼
[VTE / Terminal Emulator]    ← per-pane screen buffer (grid of cells)
    │  maintains: cursor, attributes, alternate screen, scroll regions
    ▼
[StyledGrid extraction]      ← read grid state → StyledGrid
    │
    ▼
[TmuxMirror Widget]          ← render in ratatui
```

#### Control Mode Protocol Summary

| Notification | Format | Relevance |
|--------------|--------|-----------|
| `%output` | `%output %<pane_id> <octal-encoded-data>` | Core: raw pane output |
| `%layout-change` | `%layout-change @<win_id> <layout> ...` | Geometry changed |
| `%pane-mode-changed` | `%pane-mode-changed %<pane_id>` | Copy mode enter/exit |
| `%window-pane-changed` | `%window-pane-changed @<win_id> %<pane_id>` | Active pane changed |
| `%window-add/close` | `%window-add @<win_id>` | Session structure changed |
| `%session-changed` | `%session-changed $<sid> <name>` | Session switched |
| `%pause / %continue` | Flow control | Backpressure signals |

**Key properties:**

- `%output` delivers output from **all panes** in the session, not just active.
- Data is octal-encoded: control chars and backslash → `\NNN`.
- The control client does **not affect window sizing** unless
  `refresh-client -C WxH` is sent, which would then make it count as a
  real client for size calculation. Omitting this keeps it invisible.
- Commands can be sent back via stdin; responses are wrapped in
  `%begin ... %end` / `%error` guards.

#### VTE Requirement

Unlike Approach A, the `%output` stream contains **raw terminal data** — cursor
movement, scrolling, alternate screen, partial line updates, etc. A full VTE is
required to interpret this stream into a screen buffer.

**Rust VTE crate options:**

| Crate | Version | What It Does | Maintenance | Fit |
|-------|---------|-------------|-------------|-----|
| `vte` | 0.15 | Parser only — dispatches CSI/OSC/ESC callbacks. No screen state. | Active (Alacritty team) | Too low-level alone |
| `alacritty_terminal` | 0.25 | Full terminal emulator: grid, cursor, attributes, alt screen, scrollback. `Term<T>` + `Grid<Cell>`. | Active | Best fit for faithful TUI mirroring. Heavy (~15 deps). |
| `termwiz` | 0.23 | Mid-level: `Surface` cell grid, semantic `Sgr` enum, change log. | Active (wezterm) | Good middle ground but sparse docs (34% coverage). |
| `ansi-to-tui` | 8.0 | ANSI bytes → ratatui `Text`. One-liner API. | Active | Too high-level — needs raw stream, not pre-rendered lines. |

For Approach B, `alacritty_terminal` is the practical choice. It provides:

- `Term::new(config, dims, event_proxy)` — create virtual terminal
- Feed bytes via `vte::Parser` → `Term` as `Handler`
- Read grid: `term.grid()` → iterate `Cell { c, fg, bg, flags }`
- Cell flags: `BOLD`, `DIM`, `ITALIC`, `UNDERLINE`, `INVERSE`, `STRIKEOUT`,
  `WIDE_CHAR`, etc.
- Cursor position: `term.grid().cursor.point`
- Alternate screen: handled internally

The `alacritty_terminal::Cell` → `ratatui::buffer::Cell` mapping is
straightforward: same color model (indexed + RGB), similar attribute flags.

#### Impact on Current Codebase

**Moderate refactoring required.** Control mode is a new transport/protocol
layer that does not exist today.

| Change | Scope | Details |
|--------|-------|---------|
| Control mode transport | **New module**: `src/control.rs` or `src/transport/control.rs` | Spawn `tmux -CC attach`, parse notification stream, demux by pane_id |
| Octal decoder | ~50 lines | Decode `\NNN` sequences in `%output` data |
| Per-pane VTE state | New struct | `HashMap<PaneId, alacritty_terminal::Term>` |
| Grid extraction | ~80 lines | `Term::grid()` → `StyledGrid` conversion |
| Integration with `OutputBus` | Modify `sink.rs` / Phase 2c | Control mode becomes a `TargetOutput` source alongside polling |
| `HostHandle` changes | Moderate | Need control-mode lifecycle: attach/detach per session |
| New dependency | `alacritty_terminal` | +15 transitive deps, ~2MB compile |

The existing `capture-pane` path and `exec()` continue to work over the regular
transport. Control mode is an **additional** data source, not a replacement.

#### Pros and Cons

| | Pro | Con |
|---|-----|-----|
| **Fidelity** | Highest — sees every byte, maintains full terminal state | — |
| **Latency** | Real-time — event-driven, no polling | — |
| **Cursor** | Free — VTE tracks cursor position and shape | — |
| **Multi-pane** | All panes in session delivered simultaneously | Must maintain per-pane VTE state |
| **No sizing impact** | Control client is invisible unless `refresh-client -C` sent | — |
| **Complexity** | — | New protocol parser, VTE dependency, state management |
| **Refactoring** | — | New transport layer, modify host lifecycle |
| **Dependencies** | — | `alacritty_terminal` is heavy (~15 deps) |
| **Level of effort** | — | ~1–2 weeks for a robust implementation |
| **Debugging** | — | Octal decoding, VTE state bugs, flow control (`%pause`) |
| **Integration** | Natural fit with Phase 2c `OutputBus` | Requires 2c to be in place |

---

### Comparison Matrix

| Dimension | A: Polling | B: Control Mode |
|-----------|-----------|-----------------|
| **Visual fidelity** | High (static/slow content), degraded for rapid updates | Highest — frame-perfect |
| **Cursor fidelity** | Extra query per frame | Free (VTE state) |
| **Latency** | 50–200ms (poll interval) | Sub-millisecond (event-driven) |
| **Missed updates** | Yes — transient states between polls | No — every byte seen |
| **Overhead per pane** | 1–2 tmux commands per tick | 1 persistent connection per session |
| **Multi-pane scaling** | Linear overhead (N captures/tick) | Constant (1 connection, all panes) |
| **New dependencies** | `vte` or `ansi-to-tui` (small) | `alacritty_terminal` (heavy) |
| **New code** | ~400 lines, no refactoring | ~1500+ lines, new module |
| **Level of effort** | 1–2 days | 1–2 weeks |
| **Phase dependency** | None — works with current API | Fits naturally with 2c (OutputBus) |
| **VTE required** | No | Yes |
| **Risk** | Low | Moderate (protocol edge cases, VTE bugs) |

### Recommendation

**Start with Approach A (polling).** It delivers a working TUI mirror with
minimal effort and zero refactoring, using the existing capture API. The
50–200ms latency is acceptable for most monitoring and mirroring use cases.
The visual fidelity is high — `capture-pane -ep` faithfully preserves colors
and attributes for both shell output and full-screen TUI programs.

**Evolve to Approach B (control mode) in Phase 2c or Phase 5**, when the
`OutputBus` infrastructure is in place and real-time, frame-perfect mirroring
becomes a requirement. Control mode also enables efficient multi-pane monitoring
(one connection per session vs N captures per tick), which aligns with the
Fleet/monitoring use case.

The two approaches are **not mutually exclusive**. Approach A can serve as the
initial implementation and fallback, while Approach B adds a real-time path for
sessions where low latency matters. The `TmuxMirror` widget and `StyledGrid`
are shared between both — only the source layer changes.

### Data Model (Shared)

```rust
/// A single cell in the styled grid.
struct StyledCell {
    ch: char,
    fg: ratatui::style::Color,
    bg: ratatui::style::Color,
    modifiers: ratatui::style::Modifier,
}

/// Rectangular grid of styled cells representing one pane's visible content.
struct StyledGrid {
    width: u32,
    height: u32,
    cells: Vec<StyledCell>,           // row-major, width * height
    cursor: Option<(u16, u16)>,       // (x, y) if known
}

impl StyledGrid {
    /// Parse from `capture-pane -ep` output (Approach A).
    fn from_capture_ep(raw: &str, width: u32, height: u32) -> Self { /* SGR parser */ }

    /// Extract from alacritty_terminal::Term grid (Approach B).
    fn from_vte_grid(term: &alacritty_terminal::Term<impl EventListener>) -> Self { /* grid walk */ }
}
```

### Widget (Shared)

```rust
/// ratatui Widget that renders a StyledGrid into a Buffer.
struct TmuxMirror<'a> {
    grid: &'a StyledGrid,
    show_cursor: bool,
}

impl Widget for TmuxMirror<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for y in 0..self.grid.height.min(area.height as u32) {
            for x in 0..self.grid.width.min(area.width as u32) {
                let cell = &self.grid.cells[(y * self.grid.width + x) as usize];
                if let Some(buf_cell) = buf.cell_mut(Position::new(
                    area.x + x as u16,
                    area.y + y as u16,
                )) {
                    buf_cell.set_char(cell.ch)
                        .set_fg(cell.fg)
                        .set_bg(cell.bg)
                        .set_style(Style::default().add_modifier(cell.modifiers));
                }
            }
        }
        // Optional: render cursor as inverted cell
        if self.show_cursor {
            if let Some((cx, cy)) = self.grid.cursor {
                if let Some(c) = buf.cell_mut(Position::new(
                    area.x + cx, area.y + cy,
                )) {
                    c.set_style(Style::default().add_modifier(Modifier::REVERSED));
                }
            }
        }
    }
}
```

### Where Code Lives

Per DC11, library code stays in `libs/tmux/`, application code in `bins/`.

| Component | Location | Rationale |
|-----------|----------|-----------|
| SGR parser + `StyledGrid` | `libs/tmux/src/sgr.rs` | Reusable across any consumer, not TUI-specific |
| `StyledGrid::from_vte_grid()` | `bins/` or optional feature-gated in lib | Avoids `alacritty_terminal` dep for non-TUI consumers |
| `TmuxMirror` widget | `bins/tmux-automator/` | ratatui is a binary concern |
| Control mode transport | `libs/tmux/src/control.rs` | Reusable protocol layer |
| Poll loop / render loop | `bins/tmux-automator/` | Application concern |
