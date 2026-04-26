# tmux_select Design

## Status

Draft.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-26 | @gpt55-dgx | Initial DESIGN for GitHub issue #226: local/remote tmux session selector TUI, session detail sources, monitoring mode, modal create/kill flows, accepted current-PTY attach gap, host-wide SSH integration, and SVG mock. |
| 2026-04-26 | @gpt55-dgx | Accepted PR #227 review additions from @opus47-macos-tmux: live session-list event stream via tmux control-mode notifications; focus model with `l` / `v` / `Esc` and visual focus borders; both panes scrollable with R-pane resample-backwards; bold-green motlie ASCII placeholder when MOTD absent (LT bypasses 30% cap to fit); PTY handoff non-functional requirement (no VTE-in-middle); spawn-and-wait attach with `setpgid`+`tcsetpgrp` signal hygiene; default-attach polarity with opt-in `--print-session` and opt-in `--dashboard` (re-enter on clean detach, bounded by `child.status.success()` AND list refresh AND user pick); two new accepted library gaps (`HostHandle::watch_host_events()`, `ScrollbackQuery::LinesRange`); alternatives B/C moved to appendix; testing-strategy additions; open-questions resolutions. |
| 2026-04-26 | @gpt55-dgx | Accepted PR #227 short-mode review addition from @opus47-macos-tmux: short-mode layout via `-s` flag, optimized for 32×65 terminals (mobile SSH clients, IDE terminals, tmux pop-ups). Vertical T/B split at 40:60 (T = session list, B = detail), default focus T. MOTD/motlie omitted in short mode for density. Resize keys promoted to Ctrl-modifier: `Ctrl-Up`/`Ctrl-Down` resize T/B in short mode; `Ctrl-Left`/`Ctrl-Right` resize L/R in normal mode (replacing plain `Left`/`Right`, which become reserved in main view). All other keys (`l`/`v`/`Esc`/`m`/`n`/`k`/`g`/Enter/`Ctrl-C`) and modal behavior identical across modes. |
| 2026-04-26 | @gpt55-dgx | Closed remaining PR #227 design-feedback decisions: main-view plain Left/Right stay reserved no-ops, short-mode status hints use ASCII-first compact labels, monitor history is fixed at 10,000 lines for v1, and the SVG mock now covers all required selector states. |
| 2026-04-26 | @gpt55-dgx | Addressed PR #227 re-review: added missing PLAN/API/CLI docs and pinned monitor historical fetch, kill-by-session-id, and monitored-session-close behavior. |
| 2026-04-26 | @gpt55-dgx | Addressed PR #227 round-3 cross-doc consistency feedback: aligned host events with API (`session_id`, no window-level variants), changed detail activation to `SelectedSession`, and documented stable session-id dispatch as a fourth library gap. |

## Product Scope

This is a greenfield product surface. The repository already has `motlie-tmux`
and `motlie-driver`, but `tmux_select` is a new user-facing binary with no
compatibility or migration burden. Backward compatibility for an older selector
CLI is out of scope.

The binary is intended for two use cases:

1. A host-local SSH entrypoint that replaces the login shell with a tmux session
   selector.
2. A local operator tool that can target another host by SSH URI, inspect that
   host's tmux sessions, and attach interactively to the chosen remote session.

## Problem

Users who SSH into a shared host need a fast, constrained way to choose an
existing tmux session, preview it, create a new session, kill a stale session,
and attach to the chosen session. The same selector should also work as an
operator tool for monitoring and attaching to a remote host from a different
machine.

Plain `tmux ls` followed by manual `tmux attach` is not enough because:

- it does not provide target-host context such as `/etc/motd`
- it does not preview or monitor a highlighted session before attach
- it cannot be installed as a complete host-wide selector without shell glue
- ad hoc shell glue would duplicate tmux/SSH command construction already owned
  by `motlie-tmux`

## Non-Goals

- No web UI.
- No migration path from a previous selector.
- No direct tmux command parsing or shelling in the binary for operations that
  `motlie-tmux` owns.
- No built-in session summarizer in the initial version. The `R` pane is designed
  so a summarizer can replace the sampled-content provider later.
- No policy engine for arbitrary remote target authorization. The design calls
  out where deployment policy must constrain SSH targets.

## Requirements

### Functional

- The TUI body is split into a left pane `L` and right pane `R`.
- `L` is split into upper `LT` and lower `LB`.
- `LT` displays the target host `/etc/motd`.
- `LT` height is dynamic: enough lines to show MOTD content, capped at 30% of
  the left-pane height. `LB` receives the remaining height.
- When `/etc/motd` is missing,
  empty, or unreadable, `LT` renders a built-in bold-green motlie ASCII
  placeholder followed by a `(no /etc/motd)` caption (or
  `(motd unavailable: <reason>)` on read failure). In this case `LT` height
  bypasses the 30% cap and expands to exactly fit
  `ascii_rows + caption_row + chrome` so the user always sees the full art.
  When `L_width < 48` columns or there is not enough vertical room to expand,
  fall back to a single-line `motlie · no /etc/motd` (still bold-green). The
  ASCII asset is baked into the binary as a `&'static str` (no inline ANSI
  escapes); styling is applied at render time via ratatui
  `Style { fg: Color::Green, add_modifier: Modifier::BOLD }`. Asset glyphs
  (use exactly):

  ```text
  ███╗   ███╗ ██████╗ ████████╗██╗     ██╗███████╗
  ████╗ ████║██╔═══██╗╚══██╔══╝██║     ██║██╔════╝
  ██╔████╔██║██║   ██║   ██║   ██║     ██║█████╗
  ██║╚██╔╝██║██║   ██║   ██║   ██║     ██║██╔══╝
  ██║ ╚═╝ ██║╚██████╔╝   ██║   ███████╗██║███████╗
  ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝
  ```
- `LB` lists tmux sessions on the target host and has default focus.
- `LB` and `R` are both scrollable.
  `LB` viewport scrolls automatically to keep the highlighted row visible when
  `len(sessions) > visible_rows`. A position indicator (e.g., `5/12`) is shown
  in `LB` chrome or in the status bar.
- Up and Down move the highlighted session in `LB` *when focus is `Lb`*.
  When focus is `R`, Up/Down scroll
  the `R` content one line; `PgUp`/`PgDn` page through; `Home`/`End` jump to
  top/bottom of the buffer. When focus is `Lb`, `PgUp`/`PgDn` page through the
  session list and `Home`/`End` jump to first/last session.
- `Ctrl-Left` and `Ctrl-Right` resize the `L` / `R` split in the normal main
  selector view. Plain Left and Right are reserved in the main view so arrows
  remain unambiguous for navigation and scrolling.
- `R` initially shows sampled detail for the highlighted session.
- `R` detail is supplied through a trait so future view models can summarize or
  otherwise transform session content.
- When focus is `R` in sample mode
  and the user scrolls past the top of the currently sampled buffer, the
  detail source must resample backwards: fetch additional scrollback for the
  highlighted session, prepend it to the buffer, and anchor the viewport so
  the user's scroll position stays on the same line of content (no visual
  jump). Per-page fetches must be chunked, not full-buffer rebuilds.
- When focus is `R` in monitor
  mode and the user scrolls up, auto-tail pauses; newly received history is
  appended to the buffer but the viewport stays anchored at the user's
  position. `End` (or jump-to-bottom key) re-engages auto-tail.
- Pressing `m` puts `R` into monitoring mode for the highlighted session, using
  the `motlie-tmux` monitor/history pipeline to show live updates. (Focus-
  independent: operates on the highlighted session regardless of which pane
  has focus.)
- Pressing `n` opens a centered `New Session` modal with a session-name text
  field and `Cancel` / `Ok` buttons.
- Pressing `k` opens a centered `Kill session <name>?` confirmation modal with
  `Cancel` / `Ok` buttons.
- In modal dialogs, Left and Right choose between `Cancel` and `Ok`; Enter
  exits the modal and applies `Ok` when selected. `Esc` in a modal is
  `Cancel` and closes without applying.
- Pressing `v` moves focus from
  `Lb` to `R` (no-op if already `R`). Pressing `l` moves focus from `R` to
  `Lb` (no-op if already `Lb`). Outside any modal, `Esc` is equivalent to
  `l` when focus is `R`, and is a no-op when focus is `Lb` (use `Ctrl-C` to
  exit). The currently focused pane must be visually distinguished from the
  unfocused pane via border style — a bright/colored or doubled border for
  the focused pane, dim/single for the unfocused. The status-bar focus
  indicator (below) is complementary, not a substitute.
- Pressing `g` or Enter in the main selector exits the TUI and attaches the
  current user PTY to the highlighted session. (Focus-independent: attach
  always operates on the `Lb` highlight regardless of which pane has focus.)
- The binary accepts an optional SSH URI / host target. Omitted means local host.
- For SSH targets, listing, MOTD, sampling, create, kill, monitor, and attach
  all operate against the SSH target.
- For SSH targets, attach must open an interactive SSH PTY to the target host
  and attach that remote PTY to the selected remote tmux session.
- A bottom status bar shows target host, current time, and supported keys.
  The status bar must additionally
  show the current focus (`Lb` vs `R`) and a focus-conditional key-hint set:
  when `Lb`-focused, include `v view detail`; when `R`-focused, include
  `l list`. Always-on hints (`m monitor`, `n new`, `k kill`, attach, resize,
  `Ctrl-C exit`) appear in both modes.
- The selector must keep `LB`
  consistent with the target host's tmux state without user-driven refresh,
  by subscribing at startup to a host-level event stream and reconciling on
  each event by stable session id (not name — `%session-renamed` requires
  id-based identity; `SessionInfo.id` exists). Polling
  (`list_sessions()` every N seconds) is acceptable only as a fallback when
  the control-mode link is unavailable; cadence is specified in §Data Flow →
  Live Session List below. See §Accepted motlie-tmux Library Gaps → Host
  Event Stream for the new accepted gap this requires.
- Default mode (no flag): on `g`
  or Enter, leave the TUI cleanly and spawn-and-wait attach (see §Data Flow →
  Attach). When invoked with `--print-session`, the binary instead leaves the
  TUI cleanly, prints the selected session name (and only the session name)
  followed by a newline to stdout, and exits 0; cancellation (`Ctrl-C`, no
  selection) exits non-zero with no stdout. All UI rendering, status, and
  errors go to stderr only. When invoked with `--dashboard`, the binary
  re-enters the TUI on clean child exit (see §Data Flow → Attach for the
  bounded re-entry rule). `--print-session` and `--dashboard` are mutually
  exclusive; combining them is a startup error.
- The binary accepts a short-mode
  flag `-s`. Short mode renders a compact layout optimized for 32 rows × ~65
  columns: the body splits vertically into Top (`T`, default focus, lists
  sessions) and Bottom (`B`, detail pane) at a 40:60 ratio. MOTD and the
  motlie placeholder are omitted in short mode to maximize content density.
  All command keys (`l`/`v`/`Esc`/`m`/`n`/`k`/`g`/Enter/`Ctrl-C`), modal
  behavior, focus model semantics, and detail-source trait usage are
  identical to normal mode (mapping `T` ↔ `Lb` and `B` ↔ `R`). Resize keys
  differ by mode: short mode uses `Ctrl-Up`/`Ctrl-Down` to resize `T`/`B`;
  normal mode uses `Ctrl-Left`/`Ctrl-Right` to resize `L`/`R`. Plain
  `Left`/`Right` no longer resize in main view (they remain reserved); modal
  use of `Left`/`Right` for button selection is unchanged. `-s` composes
  with `--print-session`, `--dashboard`, and SSH targets.
- The binary must use `motlie-tmux` for tmux operations and must not duplicate
  tmux command logic in the binary.

### Non-Functional

- Terminal state must be restored on normal exit, attach, error, Ctrl-C, and
  panic paths.
- Monitor handles and subscriptions must be stopped/unsubscribed when leaving
  monitoring mode, changing monitored session, killing the monitored session, or
  exiting the TUI.
- UI redraws must remain responsive while sampling or monitoring remote sessions.
- Session create/kill failures must be shown inline without corrupting the
  terminal state.
- The app must be usable as an SSH `ForceCommand` entrypoint.
- All accepted `motlie-tmux` library gaps must be implemented in the library,
  not worked around by shell command duplication in the binary.
- The attach handoff must transfer
  the user's controlling terminal directly to the attached `tmux` (or
  `ssh tmux`) process. The selector binary must not run a nested terminal
  emulator, VTE buffer, or byte-proxy between the user's PTY and the
  attached process. Concretely: before handoff the selector stops monitor
  state, leaves the alternate screen, and restores termios; for local
  targets it spawns `tmux attach-session -t <name>` as a child with
  inherited stdio (`stdin`/`stdout`/`stderr`); for SSH targets it spawns
  `ssh -t [...] tmux attach-session -t <name>` with inherited stdio. No
  `pipe()` wrapping. No re-read into the binary's TUI. See §Data Flow →
  Attach and §Accepted Library Gaps → Current PTY Attach.
- `R`-pane scroll-back fetches
  must be chunked per page, not full-buffer rebuilds, so SSH-target detail
  panes remain responsive on long-lived sessions. See §Accepted Library
  Gaps → ScrollbackQuery::LinesRange.
- Monitor-mode rolling history is bounded by line count: 10,000 lines in v1.
  On overflow, oldest lines are dropped. The bound must be documented and
  enforced to prevent unbounded memory growth on busy sessions. A configurable
  bound is a follow-up, not part of v1.

## System Design

```text
current terminal / SSH client PTY
        |
        v
tmux_select binary
        |
        +-- TargetConnection
        |      +-- local: HostHandle::local()
        |      +-- ssh:   SshConfig::parse(uri)?.connect().await?
        |
        +-- SessionStore
        |      +-- HostHandle::list_sessions()
        |      +-- HostHandle::create_session()
        |      +-- HostHandle::session_by_id(id).await?       // -> Option<Target>
        |          .ok_or(SessionVanished)?                   // race: see below
        |          .kill().await?
        |
        +-- MotdSource
        |      +-- local read /etc/motd
        |      +-- remote HostHandle::download(/etc/motd, temp)
        |
        +-- DetailPane
        |      +-- SampleDetailSource
        |      +-- MonitorDetailSource
        |
        +-- Attach
               +-- Target::attach_current_pty()  [accepted motlie-tmux gap]
```

The selector keeps all tmux state behind the connected `HostHandle`. This is the
main layering rule: UI code may decide when to list, sample, create, kill, or
attach, but it does not know how tmux commands are spelled.

## Target Model

```rust
enum TargetConnection {
    Local {
        host: motlie_tmux::HostHandle,
        label: String,
    },
    Ssh {
        uri: String,
        host: motlie_tmux::HostHandle,
        label: String,
    },
}
```

The CLI accepts:

```text
tmux_select
tmux_select ssh://user@host
tmux_select 'ssh://user@host?identity-file=/path/to/key'
```

The accepted v1 CLI target form is a positional SSH URI:
`tmux_select [ssh-uri]`. Do not add `--target` in v1; keeping a single target
form keeps help text and ForceCommand examples unambiguous. Revisit in a
follow-up only if PLAN finds positional friction with deployment tools.

Additional flags:

| Flag | Behavior |
|------|----------|
| (none) | Default. TUI → select → spawn-and-wait attach (see §Data Flow → Attach). Selector exits with the child's `ExitStatus`. |
| `--print-session` | TUI → select → leave alt-screen → print `<name>\n` to stdout → exit 0. Cancellation exits non-zero with empty stdout. All UI/diagnostics on stderr. Composable: `tmux attach -t "$(tmux_select --print-session)"`. |
| `--dashboard` | TUI → select → spawn-and-wait attach → on clean child exit (`status.success()`), re-enter the TUI; on non-zero child exit, exit with the child's status. `Ctrl-C` from the re-entered TUI exits 0 (user-initiated clean exit). See §Data Flow → Attach for the bounded re-entry rule. |
| `-s` | Short-mode layout: vertical T/B split (40:60) optimized for 32×65 terminals. MOTD omitted. Same command keys, modal behavior, focus model, and detail sources as normal mode. Resize via `Ctrl-Up`/`Ctrl-Down`. Composes with `--print-session`, `--dashboard`, and SSH targets. See §Layout → Short mode. |
| `--print-session` + `--dashboard` | Mutually exclusive — startup error. |

Polarity rationale (default attach): the binary's primary product is a session
selector that attaches; ForceCommand is the headline deployment story; users
typing `tmux_select` directly expect to enter tmux. Composable usage is opt-in
via `--print-session`.

ForceCommand-mode incompatibility: `--print-session` is incompatible with
ForceCommand mode (the user has no shell to consume the output). ForceCommand
deployments must omit the flag; the binary should warn (stderr) on a
best-effort heuristic if both are detected.

## Layout

The terminal is split into:

- body area: everything except the bottom status bar
- status bar: one terminal row

The body area is split horizontally into `L` and `R`.

`L` is split vertically:

- `LT`: MOTD, height `min(rendered_motd_lines + chrome, 30% of L height)`
  when MOTD is present. When MOTD
  is absent/empty/unreadable, `LT` height = `ascii_rows + caption + chrome`
  (bypasses the 30% cap so the motlie placeholder fully renders); narrow-
  terminal fallback collapses `LT` to a single line. See §Functional
  Requirements for the placeholder rendering rule.
- `LB`: session list, remaining height. The viewport scrolls to keep the
  highlighted row visible. A position indicator is shown.

**Focus model.** The main view has
two focus states: `Lb` (default) and `R`. Focus transitions are explicit:

- `v` → focus `R` (no-op if already `R`)
- `l` → focus `Lb` (no-op if already `Lb`)
- `Esc` outside any modal: equivalent to `l` when focus is `R`; no-op when
  focus is `Lb` (use `Ctrl-C` to exit). `Esc` inside any modal is equivalent
  to that modal's `Cancel` button.

The currently focused pane must be visually distinguished from the unfocused
pane via border style (bright/colored or doubled for focused; dim/single for
unfocused). The status-bar focus indicator (target host, time, focus, key
hints) is complementary, not a substitute.

Main-selector keymap (focus-aware):

| Key | `Lb`-focused | `R`-focused |
|-----|--------------|-------------|
| Up / Down | Move highlight; LB viewport auto-scrolls | Scroll R one line; on scroll-past-top, sample mode resamples backwards (chunked); monitor mode pins viewport (auto-tail pauses) |
| PgUp / PgDn | Page through session list | Page through R buffer |
| Home / End | First / last session | Top / bottom of buffer; `End` re-engages monitor auto-tail |
| `l` | (no-op) | Focus → `Lb` |
| `v` | Focus → `R` | (no-op) |
| `Esc` | (no-op outside modal; `Cancel` inside modal) | Focus → `Lb` (outside modal); `Cancel` inside modal |
| `Ctrl-Left` / `Ctrl-Right` | Resize `L`/`R` split (normal mode only) | Resize `L`/`R` split (normal mode only; focus-independent) |
| Left / Right | (no-op in main view; reserved) | (no-op in main view; reserved) |
| `m` | Start/switch monitoring on highlight | Same |
| `n` | Open `New Session` modal | Same |
| `k` | Open kill-confirmation modal | Same |
| Enter / `g` | Attach highlight | Attach highlight (focus-independent) |
| `Ctrl-C` | Exit selector without attach | Exit selector without attach |

Resize keys use Ctrl modifiers so plain arrows are unambiguously reserved for
navigation and scrolling. Normal mode resizes the L/R split with
`Ctrl-Left`/`Ctrl-Right`; short mode resizes the T/B split with
`Ctrl-Up`/`Ctrl-Down`.

Modal keymaps override the main keymap. In modals: Left/Right move between
`Cancel` and `Ok`; `Enter` exits and applies `Ok` if selected; `Esc` is
`Cancel`.

### Short Mode (`-s`)

Short mode is optimized for compact terminal contexts where horizontal width is
constrained: mobile SSH clients, IDE-embedded terminals, tmux pop-ups
(`display-popup`), and narrow ForceCommand deployments.

**Target dimensions:** 32 rows × ~65 columns. The layout must remain usable
at smaller sizes but is tuned for this target.

**Layout:**

- Body area: 31 rows (32 total minus 1 status-bar row).
- Body splits *vertically* into Top (`T`) and Bottom (`B`) at a 40:60 ratio
  (T ≈ 12 rows, B ≈ 19 rows for a 32-row terminal).
- `T` = session list. Equivalent to `LB` in normal mode (same scrolling,
  same position indicator, same auto-scroll-to-keep-highlight-visible
  behavior). Default focus.
- `B` = detail pane. Equivalent to `R` in normal mode (same trait-backed
  sample/monitor sources, same scroll-back-on-up, same monitor tail-pause).
- MOTD (`LT`) and the motlie placeholder are **omitted** in short mode to
  maximize content density. Status-bar focus indicator and key hints
  remain, but key hints must be terser to fit ~65 cols. Use ASCII-first
  compact labels so narrow SSH clients and IDE terminals render predictably,
  e.g., `Up/Dn pick | v detail | m mon | n new | k kill | Enter go`.

**Focus model:** Identical to normal mode, with `T` ↔ `Lb` and `B` ↔ `R`:

- Default focus is `T`.
- `v` → focus `B` (no-op if already `B`).
- `l` → focus `T` (no-op if already `T`).
- `Esc` outside modal: equivalent to `l` when focus is `B`; no-op when focus
  is `T`.
- Visual focus borders: same rule (bright/doubled for focused; dim/single
  for unfocused).

**Resize keys (mode-dependent):**

| Key | Normal mode | Short mode |
|-----|-------------|------------|
| `Ctrl-Left` / `Ctrl-Right` | Resize `L`/`R` split | (no-op; `L`/`R` not present) |
| `Ctrl-Up` / `Ctrl-Down` | (no-op; `T`/`B` not present) | Resize `T`/`B` split |
| Plain arrows (no Ctrl) | Navigation/scroll per focus-aware keymap above | Navigation/scroll per focus-aware keymap above (same — use `T`/`B` in place of `Lb`/`R`) |

**All other keys and modal behavior:** identical to normal mode (see the
focus-aware keymap above). `m`, `n`, `k`, `g`/Enter, `Ctrl-C` are
focus-independent and behave the same. Modal keymap (Left/Right for button
selection, Enter to apply, Esc to Cancel) is unchanged.

**Composition with other flags:** `-s` composes with `--print-session`,
`--dashboard`, SSH targets, and the `MOTLIE_TMUX_SELECT_BYPASS` env-var
admin bypass. ForceCommand deployments may use `-s` for tight-display
hosts (`ForceCommand /usr/local/bin/tmux_select -s --dashboard`).

## SVG Mock

The DESIGN mock source is checked in beside this document:

![tmux_select TUI mock](./tmux-select-mock.svg)

If GitHub issue rendering supports the chosen SVG embedding path, this same SVG
should be attached or linked from issue #226 after the branch is pushed.

The SVG mock includes the following panels:

1. Main selector view, `Lb`-focused.
2. Main selector view, `R`-focused.
3. Monitor mode with `R` scrolled up and auto-tail paused.
4. `New Session` modal.
5. Kill confirmation modal with title `Kill session <name>?`.
6. MOTD-absent state with bold-green motlie ASCII placeholder.
7. Short mode (`-s`) main view with focused `T`.
8. Short mode focused-`B` variant.

## R Pane Detail Source

The `R` pane should depend on a trait, not directly on sampling or monitoring
implementation details.

```rust
pub struct SelectedSession {
    pub id: String,
    pub name: String,
}

#[async_trait::async_trait]
trait SessionDetailSource {
    async fn activate(
        &mut self,
        host: &motlie_tmux::HostHandle,
        session: &SelectedSession,
    ) -> anyhow::Result<()>;

    // DetailDelta replaces the
    // bare Option<String> so monitor mode can express "append" vs "replace"
    // semantics, and so the UI can know whether to scroll the viewport.
    // Some(Append(text))  — new content arrived (monitor); append at tail.
    // Some(Replace(text)) — full re-render (sample re-fetched on highlight).
    // None                — no change since last tick.
    async fn tick(&mut self) -> anyhow::Result<Option<DetailDelta>>;

    // Resample-backwards entry
    // point. UI calls this when focus is `R` and the user scrolls past the
    // top of the currently rendered buffer. Returns lines older than
    // `before_line` (where `before_line` is an index into the source's
    // current buffer's oldest line); up to `count` lines. Empty Vec means
    // "no more history available." `SampleDetailSource` implements this via
    // `Target::sample_text(&ScrollbackQuery::LinesRange { older_than_lines,
    // count })` — see §Accepted Library Gaps.
    async fn fetch_older(
        &mut self,
        before_line: usize,
        count: usize,
    ) -> anyhow::Result<Vec<String>>;

    async fn deactivate(&mut self) -> anyhow::Result<()>;
}

pub enum DetailDelta {
    Append(String),
    Replace(String),
}
```

Initial shipped implementations:

- `SampleDetailSource`: resolves the selected session by stable id, captures
  session content, sorts panes by `(window, pane)`, omits empty panes, and
  renders text sections.
  `fetch_older` issues
  `Target::sample_text(&ScrollbackQuery::LinesRange { older_than_lines, count })`
  for paginated backwards fetch (see §Accepted Library Gaps →
  ScrollbackQuery::LinesRange).
- `MonitorDetailSource`: starts `host.watch_session()` or the equivalent
  monitor/history composition, then renders a rolling history into `R`.
  When the user scrolls up in
  monitor mode, auto-tail pauses; the source continues to receive and append
  events to its internal buffer, but the UI viewport stays anchored at the
  user's position. `fetch_older` for monitor mode falls back to a one-shot
  `Target::sample_text(&LinesRange { ... })` against the same target —
  monitor history rolls forward, so historical fetch reuses sample. When the
  10,000-line rolling monitor buffer is full and the user scrolls older than
  that buffer start, `fetch_older` must query tmux pre-monitor scrollback via
  `LinesRange` against the same target, not treat the rolling-buffer start as
  the history boundary.

Implementation should prefer static dispatch for shipped modes:

```rust
enum DetailMode {
    Sample(SampleDetailSource),
    Monitor(MonitorDetailSource),
}
```

`DetailMode` can implement `SessionDetailSource`. This preserves a trait
boundary for future summary providers without forcing dynamic dispatch into the
initial hot path.

## Data Flow

### Startup

1. Parse CLI target and flags (`--print-session`, `--dashboard`; mutually
   exclusive — error on both).
2. Connect to local or SSH target with `motlie-tmux`.
3. Load target host MOTD (or render the motlie placeholder when absent).
4. List sessions.
5. Subscribe to host-level event
   stream (see §Live Session List). On subscribe failure, fall back to
   polling; emit a status-bar indicator so the user knows refresh is degraded.
6. Initialize UI state with `LB` focused and first session highlighted.
7. Render sample detail for the highlighted session, if any.

### Live Session List

The `LB` list must stay consistent with the target host's tmux state without
user-driven refresh. Other clients may create, kill, or rename sessions;
sessions may exit unexpectedly. The selector must reconcile.

Preferred mechanism: tmux control-mode notifications. The library already
parses `%`-prefixed control-mode lines as `ControlModeMessage::Notification`
(`libs/tmux/src/monitor.rs:58–96`) but currently discards them
(`monitor.rs:337–341`). The accepted library gap (§Accepted Library
Gaps → Host Event Stream) surfaces these as a host-level
`HostEventStream` that the selector subscribes to.

Subscribe-and-reconcile loop:

1. On startup (after initial `list_sessions()`), call
   `host.watch_host_events()` and spawn a tokio task to drain it.
2. On each event:
   - `SessionsChanged` / `SessionAdded` / `SessionClosed`: re-issue
     `list_sessions()` and merge into `LB` model by stable session id (not
     name — `%session-renamed` requires id-based identity;
     `SessionInfo.id` exists in `libs/tmux/src/types.rs:66`).
     If `SessionClosed { id }` matches the currently monitored session id,
     stop the monitor and clear `R` to a placeholder or empty state until the
     user's next explicit detail/monitor action.
   - `SessionRenamed { id, old, new }`: update display name in place; preserve
     highlight.
   - `ClientDetached { session_id }` / `ClientAttached { session_id }`:
     update `attached` flag.
   - `Disconnect { reason }`: control-mode link died. Show status-bar
     indicator. Begin polling fallback (recommended cadence: 5s) until
     reconnect succeeds.
3. Reconciliation must preserve the user's highlight when possible: if the
   highlighted session id still exists, keep it highlighted; if it
   disappeared, move highlight to the next valid row (or to the previous if
   the highlighted row was the last).
4. Empty-list state (zero sessions): see §Empty Session List below.
5. Under `--dashboard`, the host-event subscription runs on a separate tokio
   task that survives the spawn-and-wait attach window. Events arriving
   during attach are buffered (bounded queue, drop-oldest on overflow); the
   buffer drains on re-entry before the first redraw.

Polling fallback semantics (when control-mode link is unavailable): re-issue
`list_sessions()` every 5s. Indicate "polling" mode in the status bar to
distinguish from event-driven freshness.

### Empty Session List

When the target host has zero tmux sessions (at startup, or after a kill under
`--dashboard` re-entry):

1. `LB` renders an inline placeholder row: `(no sessions on <host> — press n
   to create)`.
2. `R` renders nothing (or an inline hint mirroring the same `n to create`
   message).
3. Highlight is unset; `m`, `k`, `g`, `Enter` are all no-ops in this state.
4. `n` remains active and opens the New Session modal as usual.
5. ForceCommand mode treats this as the normal first-run path: the user
   creates their first session via `n`, which then becomes the highlight,
   and `Enter` attaches.

### Highlight Change

1. Up/Down (when focus is `Lb`) updates selected session index. The `LB`
   viewport scrolls to keep the highlighted row visible.
2. If `R` is in sample mode, refresh sample detail for the new session
   (replace buffer).
3. If `R` is in monitoring mode, keep monitoring the previous monitored session
   until the user presses `m` again. This avoids implicit monitor teardown when
   the user is only browsing.
4. When focus is `R`, Up/Down
   scroll the `R` content (no LB highlight movement). See §Layout keymap
   table.

### Monitoring Mode

1. Pressing `m` stops any existing monitor/detail source.
2. Start monitoring the highlighted session.
3. Subscribe to session output and render a bounded rolling history into `R`.
   Bound is line-count based: 10,000 lines in v1, oldest dropped on overflow.
4. Status bar shows the monitored session.
5. When focus is `R`, scrolling
   up pauses auto-tail; new events still append to the source's internal
   buffer but the viewport stays anchored. `End` re-engages auto-tail.
6. Killing the monitored session or exiting the TUI stops monitor state.

### New Session

1. Pressing `n` opens the modal.
2. User enters a name and selects `Ok`.
3. Call `HostHandle::create_session(name, &Default::default())`.
4. Refresh session list.
5. Highlight the created session.
6. Refresh `R` detail.

### Kill Session

1. Pressing `k` opens confirmation for the highlighted session.
2. User selects `Ok`.
3. On kill-modal-open, capture the stable session id from the highlighted
   `SessionInfo` and dispatch the kill against that id, not the display name.
   If the session was killed by another client between list and resolve,
   surface a brief inline status message ("session already gone") and let the
   host-event subscription's reconciliation refresh `LB` — do not error out.
4. Call `Target::kill()`. On error (connection dropped, permission), show
   inline error without corrupting terminal state.
5. Stop monitor state if it was monitoring that session.
6. Do not eagerly refresh; the
   host-event subscription will receive `%sessions-changed` and reconcile
   `LB` automatically. (If polling fallback is active, re-issue
   `list_sessions()` here.)
7. Move highlight to the next valid row. If the killed session was the only
   one, transition to §Empty Session List state.

### Attach

The attach handoff transfers the user's controlling terminal directly to
the spawned tmux (or `ssh tmux`) child. **No VTE-in-the-middle.**

1. Pressing Enter or `g` in the main selector (any focus) records the
   highlighted session id.
2. Stop monitor/detail state. Drop any host-event subscription draining task
   (or under `--dashboard`, leave it running on a separate tokio task with a
   bounded buffer that survives the attach window).
3. Restore raw mode and leave the alternate screen. Restore termios to
   canonical state.
4. Resolve the highlighted session id to a `Target` via the stable-id
   library path. If the session vanished between selection and resolve
   (race), show stderr message and either re-enter the TUI (under
   `--dashboard`) or exit non-zero (default).
5. **Spawn-and-wait** with inherited stdio:
   - Local target: spawn `tmux attach-session -t <name>` (using socket /
     resolved tmux binary as needed) as a child with inherited
     stdin/stdout/stderr. No `pipe()`. No proxy.
   - SSH target: spawn `ssh -t [opts] <host> tmux attach-session -t <name>`
     with inherited stdio.
   - Put the child in its own process group via `setpgid` immediately after
     fork (or via `Command::process_group(0)`). Set the foreground process
     group via `tcsetpgrp` so foreground signals (`SIGINT`, `SIGTSTP`,
     `SIGWINCH`) reach the child, not the parent.
6. Call `wait()` (parent blocks while child holds the terminal).
7. On `wait()` return, branch on flag and child exit status:

   ```text
   wait() returns
       │
       ├── --print-session ─→ (unreachable; --print-session bypasses attach)
       │
       ├── default mode ────→ exit with child.status as selector exit code.
       │                      Translate signal-terminated child to
       │                      `128 + signal` per POSIX shell convention.
       │
       └── --dashboard ─────→ if child.status.success():
                                  re-enter TUI:
                                    1. re-acquire alt-screen, raw mode
                                    2. drain buffered host events
                                    3. re-render LB (state may have changed)
                                    4. if list_sessions() refresh fails →
                                         exit with that error (bounded loop:
                                         no infinite re-entry on broken target)
                                  else (non-zero child exit):
                                    exit with child.status.
                                  Ctrl-C from re-entered TUI exits the
                                  binary with code 0 (user-initiated).
   ```

8. Tested assumption (PLAN to verify): in canonical tmux 2.x/3.x, when an
   attached session is destroyed by another client, the `tmux attach-session`
   child exits with status 0 — same as user-driven `C-b d` detach. This is
   why the `--dashboard` re-entry rule (`status.success()`) correctly
   re-enters on session-killed-elsewhere. PLAN must include a localhost
   integration test that pins this assumption against the target tmux
   version.

9. Process count footprint: under default mode, two processes are resident
   during the attach window (selector + child). Under exec-replace this
   would be one, but exec-replace forecloses recovery, observability, and
   testability — rejected. Under `--dashboard`, the same 2× count applies
   per attach cycle.

## Accepted motlie-tmux Library Gaps

### Current PTY Attach

Issue #226 accepts adding a foreground attach capability to `motlie-tmux`.

API:

```rust
pub struct AttachExit {
    pub status: std::process::ExitStatus,
}

impl Target {
    pub async fn attach_current_pty(&self) -> Result<AttachExit>;
}
```

Required semantics:

- Session targets attach that session.
- Window/pane targets should either attach their parent session and select the
  target, or return a typed unsupported-target error. The selector only needs
  session-level attach.
- Local targets run the correct tmux attach path while preserving socket and
  resolved-binary behavior.
- SSH targets open an interactive SSH PTY to the remote host and run the correct
  remote tmux attach path there.
- The API owns tmux and SSH command construction; the binary does not.
- Implementation must be
  spawn-and-wait, not exec-replace: spawn the child with inherited stdio,
  put it in its own process group via `setpgid` (e.g., via Rust's
  `std::os::unix::process::CommandExt::process_group(0)` or equivalent),
  set the foreground process group via `tcsetpgrp`, then `wait()`. Return
  the child's `ExitStatus` (translate signal-terminated to `128 + signal`).
  Rationale: spawn-and-wait preserves recovery on post-list/pre-attach
  failure (race, SSH transient, vanished target), enables per-attach
  lifecycle logging for ForceCommand fleet ops, and keeps the API
  testable with standard Rust subprocess patterns. Exec-replace is rejected.

### Host Event Stream

The selector requires a host-level event stream to keep `LB` consistent with
the target's tmux state without polling (see §Functional Requirements and
§Data Flow → Live Session List).

Today, `motlie-tmux` is per-session-only: `watch_session(name)`,
`start_monitoring_session(name)`, each opening its own
`tmux -C attach-session -t <name>` connection
(`libs/tmux/src/monitor.rs:363–377`). There is no host-scoped event API. The
control-mode parser already classifies `%`-prefixed notifications as
`ControlModeMessage::Notification` (`libs/tmux/src/monitor.rs:58–96`) but
discards them at `monitor.rs:337–341`.

API shape:

```rust
impl HostHandle {
    pub async fn watch_host_events(&self) -> Result<HostEventStream>;
}

pub enum HostEvent {
    SessionsChanged,                                  // %sessions-changed
    SessionAdded { id: String, name: String },        // derived
    SessionClosed { id: String, name: String },       // derived
    SessionRenamed { id: String, old: String, new: String },
    ClientAttached { session_id: String },
    ClientDetached { session_id: String },
    Disconnect { reason: String },                    // control-mode link died
}
```

Implementation: open a single shared `tmux -C` connection per host (e.g.,
`tmux -C new-session -d -s motlie-events` or attach to a long-lived sentinel
session), surface the already-parsed `Notification` lines as typed
`HostEvent`s. Reconnect transparently on transient drops; emit
`Disconnect { reason }` and reconnect events at the boundaries.

Recommended over the alternative (extending `OutputBus` with a host-level
variant) because the alternative couples host events to per-session monitor
lifetime — which breaks when there are no sessions (the empty-list state
must still receive `SessionAdded`).

### ScrollbackQuery::LinesRange

The `R` pane's resample-backwards behavior (see §Functional Requirements and
§R Pane Detail Source) requires a windowed scrollback fetch.

Today, `ScrollbackQuery` (`libs/tmux/src/types.rs:660–668`) supports only:

```rust
pub enum ScrollbackQuery {
    LastLines(usize),
    Until { pattern: Regex, max_lines: usize },
    LastLinesUntil { lines: usize, stop_pattern: Regex },
}
```

None supports a windowed/range fetch ("lines older than offset K, up to N
lines"). The selector could simulate it by re-issuing
`LastLines(prev_total + chunk)` and discarding overlap, but this re-fetches
the entire history each step — O(N²) bandwidth over SSH. Unacceptable for
long-lived sessions.

API shape:

```rust
pub enum ScrollbackQuery {
    LastLines(usize),
    Until { pattern: Regex, max_lines: usize },
    LastLinesUntil { lines: usize, stop_pattern: Regex },
    // new
    LinesRange { older_than_lines: usize, count: usize },
}
```

Semantics: return up to `count` lines older than `older_than_lines`, where
`older_than_lines` is anchored at the current capture tail. If the detail pane
already has 200 lines loaded and needs the previous page, it requests
`LinesRange { older_than_lines: 200, count: page_size }`. Empty result means
"no more history available." Used by `SampleDetailSource::fetch_older` and
`MonitorDetailSource::fetch_older`; monitor mode uses the same tmux capture
history anchor and does not treat the rolling monitor buffer as the source of
truth for historical fetches.

### Stable Session-Id Dispatch

The selector captures `SessionInfo.id` for destructive operations, attach, and
detail-source activation. Display names can change while a modal is open or
while the user is browsing sessions, so resolving by name is not sufficient.

API shape:

```rust
impl HostHandle {
    pub async fn session_by_id(&self, id: &str) -> Result<Option<Target>>;
}
```

The library owns id-to-target resolution so the binary does not duplicate
tmux discovery or command construction. If tmux cannot address a session by id
directly for a needed operation, the library must perform the safe lookup and
race handling internally before returning `Target`.

### Remote MOTD

The binary can use existing `HostHandle::download(remote, local, opts)`
(`libs/tmux/src/host.rs:522`) to retrieve `/etc/motd` from SSH targets into
a temporary local file. The
fallback rationale below is concrete: `download()` requires temp-file
lifecycle management (create, write, read-back, cleanup), and `/etc/motd`
files of unbounded size could waste disk. If those concerns prove
material in PLAN, the narrower library addition should be a host-level
text-file read helper:

```rust
impl HostHandle {
    pub async fn read_text_file(&self, path: &std::path::Path) -> Result<String>;
}
```

This is not accepted yet as a required gap; it is a design fallback if the
existing file-transfer API is too awkward or unsafe for MOTD.

## Host-Wide SSH Integration

The local-host deployment target is `ForceCommand`.

```text
Match Group tmux-users
    PermitTTY yes
    ForceCommand /usr/local/bin/tmux_select
```

Operational behavior:

1. `sshd` allocates the user's PTY.
2. `sshd` starts `tmux_select` instead of the login shell.
3. The selector targets the local host unless deployment passes an allowed SSH
   target argument.
4. User selects, monitors, creates, kills, or attaches.
5. On attach, the selector restores terminal state and hands the PTY to the
   selected tmux session.

Deployment policy must decide:

- which Unix users/groups are subject to the selector
- whether `SSH_ORIGINAL_COMMAND` is rejected, ignored, or used as an admin
  bypass
- whether remote target arguments are allowed in ForceCommand deployments
- how admins bypass the selector for maintenance

Recommended initial deployment policy:

- ForceCommand mode targets local host only.
- Operator-invoked CLI mode may pass an SSH URI.
- `SSH_ORIGINAL_COMMAND` is rejected with a clear message unless an explicit
  admin bypass is configured outside the binary.

Concrete admin-bypass mechanism:
the binary reads the environment variable `MOTLIE_TMUX_SELECT_BYPASS` at
startup. If unset or empty, `SSH_ORIGINAL_COMMAND` is rejected with a stderr
message and the binary exits non-zero. If set to `1` (or any non-empty
value), the binary exec's `SSH_ORIGINAL_COMMAND` via the user's login shell
(`/bin/sh -c "$SSH_ORIGINAL_COMMAND"`) and bypasses the TUI entirely.
Deployments enable this by adding `AcceptEnv MOTLIE_TMUX_SELECT_BYPASS` to
`sshd_config` for the relevant `Match Group` (or by setting the variable
via PAM/login.defs for specific users/groups). This keeps the bypass
configuration external to the binary while giving PLAN a concrete
mechanism to implement and test.

ForceCommand deployments must
NOT use `--print-session` (the user has no shell to consume stdout).
Recommended deployments:

```text
# Default: TUI selector with attach
ForceCommand /usr/local/bin/tmux_select

# Dashboard mode: re-enter selector on tmux detach (workspace UX)
ForceCommand /usr/local/bin/tmux_select --dashboard
```

## Approach (Selected)

**A. New Binary Built Directly On motlie-tmux** — adopted as the main body
of this DESIGN. Create `tmux_select` as a focused binary that uses
`motlie-tmux` APIs for all tmux and SSH operations.

Pros:

- matches issue #226 directly
- clean binary boundary
- avoids coupling selector UX to the broader driver REPL/TUI command surface
- keeps tmux logic in `motlie-tmux`
- works for ForceCommand deployment

Cons:

- needs some TUI state machinery duplicated from existing frontend patterns
- depends on four accepted library gaps
  (`Target::attach_current_pty`, `HostHandle::watch_host_events`,
  `ScrollbackQuery::LinesRange`, `HostHandle::session_by_id`)

Comparison of all three
alternatives along the four CLAUDE.md greenfield axes:

| Axis | A. New binary on motlie-tmux | B. Extend driver TUI | C. Shell-based selector |
|------|-------------------------------|-----------------------|--------------------------|
| Robustness | High — single source of truth in library; bugs caught once | Medium — inherits driver complexity; selector failures may co-fail driver workflows | Low — duplicates tmux/SSH logic; two sources of truth diverge over time |
| Correctness | High — typed library APIs | Medium — entangled with driver state | Low — string parsing of tmux output; fragile across versions |
| User experience | Best — minimal, attach-first; ForceCommand-clean | Worse — users see unrelated driver commands | Worst — no preview, no monitor, no live updates |
| Operability | Good — small binary, single-purpose, testable | Worse — larger binary, broader policy surface | Worst — ad-hoc shell glue, no clean test story |

A wins on all four axes. See Appendix A for B and C considered-but-rejected
detail.

## Dependency Choices

| Dependency | Use | Decision |
|------------|-----|----------|
| `ratatui` | layout/widgets/rendering | Use. Already used by tmux examples and driver frontend. |
| `crossterm` | terminal raw mode, alternate screen, key events | Use. Already paired with ratatui in repo. |
| `ansi-to-tui` | optional ANSI rendering for captured/monitored pane content | Defer to a follow-up. The first pass renders captured content as plain text; styling for captured panes is non-critical UX. The motlie ASCII placeholder is hand-styled via ratatui `Style` with no ANSI parsing required. |
| `async-trait` | async detail-source trait | Use if a trait object or async trait implementation is needed. Already used in repo. |
| `tempfile` | remote MOTD download target | Use if remote MOTD is implemented through `HostHandle::download()`. Already a dev dependency in parts of the repo; PLAN should decide package placement. |

## Testing Strategy

DESIGN identifies the test surfaces; PLAN must make these concrete.

- Unit tests for layout calculations:
  - MOTD height cap (present case)
  - MOTD-absent placeholder
    expansion: `LT` height = `ascii_rows + caption + chrome`, bypasses 30%
    cap; narrow-terminal fallback collapses to single line
  - status bar reservation
  - `L` / `R` resize bounds (minimum widths so neither pane collapses to 0)
  - Short mode (`-s`) layout at
    32×65 viewport: body = 31 rows; T/B split at 40:60 yields T ≈ 12 rows
    and B ≈ 19 rows; MOTD/motlie omitted; status bar present
  - Short mode `Ctrl-Up`/
    `Ctrl-Down` resize bounds (minimum heights so neither pane collapses
    to 0); normal mode `Ctrl-Left`/`Ctrl-Right` parallel
  - Plain `Left`/`Right` in main
    view is a no-op in both modes (modal use unchanged)
- Unit tests for state transitions:
  - highlight movement
  - sample vs monitor mode
  - modal button selection
  - create/kill success and error paths
  - focus toggles: `v` `Lb`→`R`,
    `l` `R`→`Lb`, `Esc` outside modal `R`→`Lb`, no-op when already focused
  - `Esc` inside modal = `Cancel`
- Style/snapshot tests:
  - motlie placeholder spans carry `Modifier::BOLD` and `Color::Green`
  - focused pane border style differs from unfocused pane border style
- Mock-backed tests through `motlie-tmux`:
  - session list rendering
  - detail source rendering
  - create session refresh and highlight
  - kill session refresh and highlight
  - host-event reconciliation:
    inject `SessionAdded`/`SessionClosed`/`SessionRenamed`/`Disconnect` events,
    assert `LB` state matches expected; reconciliation by id (rename keeps
    highlight on same id even when display name changes)
  - scrollback windowing:
    `SampleDetailSource::fetch_older` issues `LinesRange` and prepends
    correctly; viewport anchor preserved
  - monitor tail-pause: scroll-up
    pins viewport; `End` re-engages auto-tail
- Terminal smoke tests:
  - raw mode and alternate-screen restoration
  - Ctrl-C behavior
  - attach path restores terminal before handoff
  - panic-path terminal restore:
    inject a panic during the main loop; assert termios + alt-screen are
    restored via the panic hook
  - signal hygiene: child in own
    process group via `setpgid` + `tcsetpgrp`; SIGINT/SIGWINCH route to child
- Localhost integration:
  - create temporary session
  - list and sample it
  - monitor it
  - kill it
  - `--print-session` contract:
    stdout is exactly `<name>\n` on selection; empty on cancel; exit code 0
    on selection, non-zero on cancel; stderr can carry diagnostics without
    polluting stdout (assert via captured stdout in non-TTY harness)
  - `--dashboard` re-entry on
    external kill: attach to a session, kill it via `tmux kill-session -t
    <name>` from a sibling client, assert child exit status is 0
    (canonical-tmux assumption), selector re-enters TUI, killed session is
    absent from refreshed `LB`
  - `--dashboard` no-loop on
    failure: attach, force a non-zero child exit (e.g., target session
    vanished, or kill-server), assert selector exits with that status (no
    re-entry)
  - `--dashboard` no-loop on
    refresh failure: attach, detach cleanly, but make `list_sessions()`
    fail at re-entry; assert selector exits with that error
- SSH integration:
  - target an SSH URI
  - read remote MOTD
  - list remote sessions
  - monitor remote session
  - attach to remote selected session through an interactive PTY
  - `SSH_ORIGINAL_COMMAND` is
    rejected in default mode; bypassed (exec'd via shell) when
    `MOTLIE_TMUX_SELECT_BYPASS=1` and present

## Open Questions

Previously open questions that materially affect v1 are resolved below. Items
that remain speculative stay explicitly open.

### Decided

- **CLI form** — Positional SSH URI only (`tmux_select [ssh-uri]`). No
  `--target` flag in v1. Revisit if PLAN finds positional friction.
- **Modal `Esc`** — `Esc` in any modal is equivalent to `Cancel`.
- **`Esc` outside modal** — Equivalent to `l` when focus is `R`; no-op when
  focus is `Lb`.
- **Monitor follow on highlight change** — No automatic follow. Monitor only
  switches when the user explicitly presses `m` on a different highlight.
  (Unchanged from initial DESIGN; reaffirmed.)
- **`New Session` options in v1** — Defaults only (no window size / history
  flags). Future enhancement.
- **Remote targets in ForceCommand** — Local-only ForceCommand initially.
  Operator-invoked CLI mode may pass an SSH URI.
- **Main-view plain Left/Right keys** — Reserved no-ops in normal and short
  mode. Ctrl-modified arrows own resize. Modal Left/Right keeps button
  selection behavior.
- **Short-mode status hints** — ASCII-first compact labels. Unicode affordance
  glyphs can be considered later, but v1 must render predictably in narrow
  SSH clients and IDE terminals.
- **Monitor history bound** — 10,000 retained lines in v1. Oldest lines are
  dropped on overflow. Configurability is a follow-up.

### Still Open

- Optional `--exec` flag (exec-replace handoff for ops who want zero
  residency). DESIGN rejects exec-replace as default; PLAN may evaluate
  whether to add as opt-in.
- Detail-source future variants (summarizer / LLM-backed). Trait shape
  accommodates them; concrete implementations are out of scope here.

## Appendix A: Alternatives Considered (B and C)

### B. Extend motlie-tmux-driver TUI

Add selector mode to the existing driver TUI.

Pros:

- reuses existing driver frontend and monitoring state
- lower initial UI scaffolding

Cons:

- driver is command-workflow oriented, while selector is attach-first
- ForceCommand deployments would inherit unrelated commands and state
- harder to keep the UX minimal and host-policy friendly

### C. Standalone Shell-Based Selector

Build the selector with direct `tmux` and `ssh` subprocess commands in the
binary.

Pros:

- quickest prototype
- no library attach gap required before a demo

Cons:

- violates issue #226 requirement to use `motlie-tmux`
- duplicates parsing, socket, binary-resolution, SSH, and attach logic
- creates a second source of truth for tmux behavior
- weak testability and error consistency
