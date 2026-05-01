# mmux Design

## Status

Draft.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-01 | @codex | Updated issue #241 design after tmux unset research: tag deletion requires a `motlie-tmux` unset API over `set-option -u`, and the `i` modal now supports row focus with Up/Down plus `x` delete and `u` update actions for focused tag rows. |
| 2026-05-01 | @codex | Started issue #241 design for session-list rename and mmux tag management modals: list-focus-only rename on `r`, selected-session tag edit on `t`, tag info/add modal on `i`, dispatch through the motlie-tmux session tag API, and stable `(host_id, session_id)` routing. |
| 2026-04-29 | @opus47-macos-tmux | Swept stale DESIGN.md sections that still described removed contracts (`list_sessions_now()`, `SessionListing.now`, `host_clock_offset_secs`, `probe_host_clock()`, raw `SessionInfo.activity` sort): rewrote the recency-display section, transport/fan-out architecture, multi-host recency/resilience block, internal data-model snippet, refresh-loop pseudocode, and Live Session List section to match shipped behavior — `HostHandle::list_sessions()` plus binary-side `ActivityTracker`, `(host_id, session_id)` selection identity, and observer-relative sort. |
| 2026-04-29 | @opus47-macos-tmux | Removed `server_epoch` and the per-host clock-offset cache: there is no portable, side-effect-free way to read the host clock on tmux ≤ 3.6 (`run-shell 'date +%s'` corrupts the operator's attached pane on older tmux). Activity stays observer-relative; age now uses operator-side `local_now` under an explicit NTP-synced clock assumption. Wildly skewed host clocks produce mildly inaccurate "age" text but no functional regression. Net: zero new public methods on `HostHandle` for this PR. |
| 2026-04-29 | @opus47-macos-tmux | Made `mod discovery` private and dropped the `_with_socket` shims (now unused after the privatization). Removed `list_sessions_now`, `SessionListing` from the public surface. External access to tmux is via `HostHandle::*` only. |
| 2026-04-29 | @opus47-macos-tmux | Added §System Design → Clock Handling: activity is observer-relative via `ActivityTracker`; age is `local_now − session.created` under the NTP assumption. |
| 2026-04-29 | @opus47-macos-tmux | Added §System Design → Transport and Latency Architecture: documents persistent russh connection per host, channel multiplexing, parallel fan-out (`futures::join_all`), per-tick cost table, attach via separate external `ssh` subprocess for clean PTY handoff, and zero-SSH-handshakes-per-refresh property. |
| 2026-04-29 | @opus47-macos-tmux | Added Multi-host mode design (issue #235): CLI accepts multiple SSH hosts; aggregated activity-sorted session list across hosts; per-row hostname; per-host skew-free recency; MOTD hidden in multi-host; new internal `HostFleet` / `SessionRow` types; library-side fan-out left to the binary using existing `HostHandle::list_sessions_now()`. Outline of bin/lib impact added. |
| 2026-04-28 | @gpt55-dgx | Fixed ForceCommand bypass contract inconsistency: bypass requires exactly `MOTLIE_MMUX_BYPASS=1`; linked issue #232 for env-gated SSH integration tests. |
| 2026-04-28 | @gpt55-dgx | Consolidated mmux refresh from separate activity and structural pollers into one `list_sessions_now()` loop. |
| 2026-04-28 | @gpt55-dgx | Added periodic visible-row refreshes because activity-only changes do not emit host events, but must reorder the activity-sorted session list. |
| 2026-04-28 | @gpt55-dgx | Added activity-descending session-list ordering so the most recently active session appears first while selection remains stable by id. |
| 2026-04-28 | @gpt55-dgx | Changed session-list recency display to unlabeled `<active> / <age>` values with day bucketing and a right margin. |
| 2026-04-28 | @gpt55-dgx | Clarified recency clock behavior for tmux versions that expand `#{epoch}` empty: fall back to a local clock clamped to session timestamps. |
| 2026-04-28 | @gpt55-dgx | Added implemented session-list recency rows: attached marker plus right-aligned `active` and `age` values from `list_sessions_now()`; tmux alert flags remain out of scope. |
| 2026-04-28 | @gpt55-dgx | Added issue #229 library support note: session snapshots now include activity, attached client count, and server-clock listings for future recency rendering. |
| 2026-04-28 | @gpt55-dgx | Changed bottom status command hints from parenthesized mnemonics to rendered underlined shortcut letters. |
| 2026-04-28 | @gpt55-dgx | Replaced the old fixed compact-placeholder width threshold with a fit check derived from the embedded motlie glyph dimensions. |
| 2026-04-28 | @gpt55-dgx | Clarified landscape MOTD sizing: `LT` is sized from the post-split left-column area so placeholder or host MOTD content remains visible before the session list. |
| 2026-04-28 | @gpt55-dgx | Added round-3 testability clarification: MOTD loading is factored through `load_motd_from(host, path)` and regression-tested for fallback/readable cases, full/compact placeholder rendering, and portrait omission. |
| 2026-04-28 | @gpt55-dgx | Updated review cleanup reality: MOTD loading now uses bounded `HostHandle::read_text_file`, session ids are typed `SessionId`s, host events remain documented as polling-backed with control-mode notifications reserved for future use, and selector state/render/input/detail concerns are split into focused modules plus typed `StatusBanner`. |
| 2026-04-27 | @gpt55-dgx | Updated modal layout: padded content, separator above button bar, bordered New Session input, and Help build metadata before key functions. |
| 2026-04-27 | @gpt55-dgx | Reordered bottom status commands and added `l` to toggle portrait/landscape layout at runtime. |
| 2026-04-27 | @gpt55-dgx | Changed main-view pane cycling from plain Left/Right to the `p` key and updated status hints. |
| 2026-04-27 | @gpt55-dgx | Added in-memory selector UI state retention across default attach/detach re-entry. |
| 2026-04-27 | @gpt55-dgx | Split resize bounds by layout mode: landscape remains 25/75, portrait becomes 15/85. |
| 2026-04-27 | @gpt55-dgx | Added build date to Help and shortened the displayed git SHA to the last 8 characters. |
| 2026-04-27 | @gpt55-dgx | Shortened bottom status direction hints to `↑/↓ sel` and `←/→ pane`. |
| 2026-04-27 | @gpt55-dgx | Changed top status host/IP separator to `|` and reordered bottom command hints with `(h)elp` first. |
| 2026-04-27 | @gpt55-dgx | Added a top status bar for bold host/IP and right-justified time; Sessions title is now count-only. |
| 2026-04-27 | @gpt55-dgx | Changed plain Left/Right focus movement to cycle through panes, including the landscape MOTD pane. |
| 2026-04-27 | @gpt55-dgx | Renamed the selector executable and docs to `mmux`, including ForceCommand examples and mock asset references. |
| 2026-04-27 | @gpt55-dgx | Updated Sessions title format to `Sessions [n] @ <hostname>, <ip address>` and removed the `keys` label from the status bar. |
| 2026-04-27 | @gpt55-dgx | Moved host label from the status bar into the Sessions pane title. |
| 2026-04-27 | @gpt55-dgx | Replaced directional words in status hints with arrow symbols and expanded the `h` help modal with key functions. |
| 2026-04-27 | @gpt55-dgx | Changed portrait mode default T/B split from 40:60 to 30:70. |
| 2026-04-26 | @gpt55-dgx | Added an `h` About modal that shows the built-in motlie logo and the build git SHA; Enter or Esc closes it. |
| 2026-04-26 | @gpt55-dgx | Removed focus labels from the status bar because focused panes are already indicated by border styling. |
| 2026-04-26 | @gpt55-dgx | Updated status bar contract: omit layout labels from the status text and render the bar with a blue background. |
| 2026-04-26 | @gpt55-dgx | Finalized the CLI mode contract: default mode is attach-and-reenter selector behavior, and `--script` replaces `--print-session` / `--dashboard` for shell integration. |
| 2026-04-26 | @gpt55-dgx | Added `--portrait/-p` and `--landscape/-l` force flags and changed auto-detection to `columns / rows <= 4.0`, making 66x30 portrait. |
| 2026-04-26 | @gpt55-dgx | Set portrait auto-detection to the clean `columns / rows <= 2.0` rule and embedded the `/tmp/motlie-TOP-CHOICE.txt` glyph as the MOTD-absent fallback icon. |
| 2026-04-26 | @gpt55-dgx | Replaced short mode with portrait mode: `--portrait` is the explicit override, startup auto-detects portrait layout from PTY dimensions, the old `-s` flag is no longer accepted, and the MOTD fallback logo uses the requested Claude artifact ASCII art. |
| 2026-04-26 | @gpt55-dgx | Updated implemented keymap and rendering details: attach is Enter/`a`, Right/Left move focus between list and detail/monitor panes, Shift-arrow resize is documented for macOS iTerm2, sample detail preserves ANSI color, session-list refresh is polling-backed snapshot reconciliation, and narrow MOTD fallback stays graphical. |
| 2026-04-26 | @gpt55-dgx | Initial DESIGN for GitHub issue #226: local/remote tmux session selector TUI, session detail sources, monitoring mode, modal create/kill flows, accepted current-PTY attach gap, host-wide SSH integration, and SVG mock. |
| 2026-04-26 | @gpt55-dgx | Accepted PR #227 review additions from @opus47-macos-tmux: live session-list event stream via tmux control-mode notifications; focus model with `l` / `v` / `Esc` and visual focus borders; both panes scrollable with R-pane resample-backwards; bold-green motlie ASCII placeholder when MOTD absent (LT bypasses 30% cap to fit); PTY handoff non-functional requirement (no VTE-in-middle); spawn-and-wait attach with `setpgid`+`tcsetpgrp` signal hygiene; default-attach polarity with opt-in `--print-session` and opt-in `--dashboard` (re-enter on clean detach, bounded by `child.status.success()` AND list refresh AND user pick); two new accepted library gaps (`HostHandle::watch_host_events()`, `ScrollbackQuery::LinesRange`); alternatives B/C moved to appendix; testing-strategy additions; open-questions resolutions. |
| 2026-04-26 | @gpt55-dgx | Accepted PR #227 short-mode review addition from @opus47-macos-tmux: short-mode layout via `-s` flag, optimized for 32×65 terminals (mobile SSH clients, IDE terminals, tmux pop-ups). Vertical T/B split at 40:60 (T = session list, B = detail), default focus T. MOTD/motlie omitted in short mode for density. Resize keys promoted to Ctrl-modifier: `Ctrl-Up`/`Ctrl-Down` resize T/B in short mode; `Ctrl-Left`/`Ctrl-Right` resize L/R in normal mode (replacing plain `Left`/`Right`, which become reserved in main view). All other keys (`l`/`v`/`Esc`/`m`/`n`/`k`/`g`/Enter/`Ctrl-C`) and modal behavior identical across modes. |
| 2026-04-26 | @gpt55-dgx | Closed remaining PR #227 design-feedback decisions: main-view plain Left/Right stay reserved no-ops, short-mode status hints use ASCII-first compact labels, monitor history is fixed at 10,000 lines for v1, and the SVG mock now covers all required selector states. |
| 2026-04-26 | @gpt55-dgx | Addressed PR #227 re-review: added missing PLAN/API/CLI docs and pinned monitor historical fetch, kill-by-session-id, and monitored-session-close behavior. |
| 2026-04-26 | @gpt55-dgx | Addressed PR #227 round-3 cross-doc consistency feedback: aligned host events with API (`session_id`, no window-level variants), changed detail activation to `SelectedSession`, and documented stable session-id dispatch as a fourth library gap. |
| 2026-04-26 | @gpt55-dgx | Updated the MOTD-absent default placeholder art to the compact motlie glyph supplied for `/etc/motd` fallback. |
| 2026-04-26 | @gpt55-dgx | Replaced the MOTD-absent default placeholder with the full-width MOTLIE glyph supplied for `/etc/motd` fallback. |
| 2026-04-26 | @gpt55-dgx | Incorporated validation feedback: `q` exits like `Ctrl-C`, Ctrl-arrow resize accepts terminals that send extra modifiers, detail-pane scroll direction follows terminal convention and shows a scrollbar/range indicator, monitor mode strips raw ANSI/control bytes for TUI rendering, and dashboard re-enters after detach when the selected session still exists. |
| 2026-04-26 | @gpt55-dgx | Incorporated second validation feedback: monitor mode now mirrors rendered screen content through `capture_all_with_options(ScreenStable)` plus `ansi-to-tui`/VTE parsing, modified-arrow resize accepts terminal fallback sequences, and attach foreground-process-group restore ignores `SIGTTOU` to avoid stopped selector jobs after detach. |

## Product Scope

This is a greenfield product surface. The repository already has `motlie-tmux`
and `motlie-driver`, but `mmux` is a new user-facing binary with no
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
  empty, or unreadable, `LT` renders a built-in bold-green motlie glyph
  placeholder followed by a `(no /etc/motd)` caption (or
  `(motd unavailable: <reason>)` on read failure). In this case `LT` height
  bypasses the 30% cap and expands to exactly fit
  `glyph_rows + caption_row + chrome` when space allows. When `L_width < 63`
  columns or there is not enough vertical room to expand, render the compact
  built-in glyph `motlie  ══╬══` plus `(no /etc/motd)` caption (still
  bold-green), not a text-only placeholder. The glyph assets are baked into
  the binary as `&'static str` values (no inline ANSI
  escapes); styling is applied at render time via ratatui
  `Style { fg: Color::Green, add_modifier: Modifier::BOLD }`. Asset glyphs
  (use exactly):

  ```text
                   _   _ _
   _ __ ___   ___ ┃ ┃_┃ (_) ___   ╲╲ ║ ╱╱
  ┃ '▄ ` ▄ ╲ ╱ ▄ ╲┃ ▄▄┃ ┃ ┃╱ ▄ ╲  ══ ╬ ══
  ┃ ┃ ┃ ┃ ┃ ┃ (_) ┃ ┃_┃ ┃ ┃  __╱  ╱╱ ║ ╲╲
  ┃▄┃ ┃▄┃ ┃▄┃╲▄▄▄╱ ╲▄▄┃▄┃▄┃╲▄▄▄┃
  ```
- `LB` lists tmux sessions on the target host and has default focus.
- `LT`, `LB`, and `R` all participate in pane focus cycling in landscape mode.
  `LT` is focusable for visual orientation but is not scrollable in the initial
  implementation.
- `LB` and `R` are both scrollable.
  `LB` viewport scrolls automatically to keep the highlighted row visible when
  `len(sessions) > visible_rows`. A position indicator (e.g., `5/12`) is shown
  in `LB` chrome or in the status bar.
- Up and Down move the highlighted session in `LB` *when focus is `Lb`*.
  When focus is `LT`, scrolling keys are no-ops. When focus is `R`, Up/Down scroll
  the `R` content one line; `PgUp`/`PgDn` page through; `Home`/`End` jump to
  top/bottom of the buffer. When focus is `Lb`, `PgUp`/`PgDn` page through the
  session list and `Home`/`End` jump to first/last session.
- `Ctrl-Left` and `Ctrl-Right` resize the `L` / `R` split in the normal main
  selector view. The implementation also accepts terminal fallback
  modified-arrow sequences such as Alt/Shift arrows and word-left/word-right
  when a client does not report Ctrl-arrow distinctly. On macOS iTerm2, manual
  validation observed `Shift-Left` and `Shift-Right` for L/R resize. Plain
  Left/Right do not change pane focus in the main view.
- Pressing `p` cycles focus `LT -> Lb -> R -> LT` in landscape mode.
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
  `motlie-tmux` rendered screen capture (`capture_all_with_options` with
  `CaptureNormalizeMode::ScreenStable`) plus VTE/ANSI parsing to show live
  screen snapshots. This is intentionally a screen mirror, not raw tmux
  control-mode `%output` replay, because TUI programs rely on cursor movement,
  clearing, and repaint semantics. (Focus-independent: operates on the
  highlighted session regardless of which pane has focus.)
- Pressing `n` opens a centered `New Session` modal with padded content, a
  bordered session-name text field, a horizontal separator, and `Cancel` /
  `Ok` buttons in the button bar.
- Pressing `k` opens a centered `Kill session <name>?` confirmation modal with
  padded content, a horizontal separator, and `Cancel` / `Ok` buttons in the
  button bar.
- Pressing `r` opens a centered `Rename Session` modal only when the session
  list pane has focus and a session is highlighted. The modal has a text field
  labeled `Session Name`, prepopulated with the current session name, and the
  same `Cancel` / `Ok` button styling and behavior as other action modals.
  `Ok` renames only when the submitted value differs from the current name.
- Pressing `t` opens a centered `Session Tag` modal for the highlighted
  session. The modal has text fields labeled `Tag` and `Value`; `Tag = owner`
  maps to tmux user option `@mmux/owner`. When a valid tag key is entered and
  an existing value is present on the selected session, `Value` is prefilled.
  `Ok` writes through the `motlie-tmux` session tag API only when `Value` is
  non-empty.
- Pressing `i` opens a centered `Session Info` modal for the highlighted
  session. It lists all `@mmux/<tag>` options for that session sorted
  lexicographically by stripped tag key. Existing tag rows are focusable with
  Up/Down; `x` deletes the focused tag and `u` loads the focused tag into the
  update controls. The bottom of the modal shows `Key` and `Value` fields plus
  a focusable `+` control for adding or applying a tag value. `+` writes
  through the same tag API rule as the `t` modal. `Esc`, or Enter on focused
  `Cancel`, closes without writing.
- Pressing `h` opens a centered help modal with the built-in motlie logo,
  build date, current build git SHA, key-function reference text, a horizontal
  separator, and an `Ok` button. Build metadata renders below the logo and
  above the key-function reference.
- In create/kill modal dialogs, Left and Right choose between `Cancel` and
  `Ok`; Enter exits the modal and applies `Ok` when selected. `Esc` in a modal
  is `Cancel` and closes without applying. In the help modal, Enter or `Esc`
  closes the modal without changing selector state.
- Pressing `p` cycles focus through the landscape panes in this order:
  `LT -> Lb -> R -> LT`. Outside any modal, `Esc` returns focus to `Lb`
  (use `q` or `Ctrl-C` to exit). The currently
  focused pane must be visually distinguished from the unfocused panes via
  border style — a bright/colored or doubled border for the focused pane,
  dim/single for the unfocused. The status bar does not duplicate focus state.
- Pressing `l` toggles the current TUI between portrait and landscape layout.
  The toggle is runtime-only and is retained in memory across default
  attach/detach re-entry within the same `mmux` parent process.
- Pressing `a` or Enter in the main selector exits the TUI and attaches the
  current user PTY to the highlighted session. (Focus-independent: attach
  always operates on the `Lb` highlight regardless of which pane has focus.)
- Pressing `q` exits the selector without attach, equivalent to `Ctrl-C` in
  the main selector view.
- The binary accepts an optional SSH URI / host target. Omitted means local host.
- For SSH targets, listing, MOTD, sampling, create, kill, monitor, and attach
  all operate against the SSH target.
- For SSH targets, attach must open an interactive SSH PTY to the target host
  and attach that remote PTY to the selected remote tmux session.
- A top status bar shows the target host as `<hostname> | <ip address>` in bold,
  left-justified text and the current time right-justified. It uses the same
  blue background as the bottom status bar.
- The session-list pane title shows only the session count as `Sessions [n]`.
- Each session row shows the display name, a `*` attached-client marker when
  one or more tmux clients are attached, and a right-aligned
  `<active> / <age>` recency column with a small right margin. The left value
  ("active") is **observer-relative** — `local_now − activity_observed_at_local`,
  where `activity_observed_at_local` is the operator-side wall clock the last
  time the binary saw `SessionInfo.activity` advance. The right value ("age")
  is `local_now − SessionInfo.created` under an NTP-synced clock assumption.
  See §System Design → Clock Handling for the rationale. Durations use `now`,
  `m`, `h`, or `d`; day values keep at most one decimal digit.
- Session rows are sorted by `activity_observed_at_local` descending so the
  session whose activity most recently advanced is at the top. Sorting on the
  observer-side mark instead of raw host `SessionInfo.activity` keeps order
  stable across multi-host fleets even when host clocks drift. Stable
  `(host_id, session_id)` preservation keeps the current highlight on the
  same session after refresh even if the row moves. Window-level tmux
  alert/status flags such as `!`, `#`, and `~` are deferred.
- A bottom status bar shows supported keys and status text.
  Key hints must use arrow symbols instead of spelling out `up`, `down`,
  `left`, or `right`. Direction hints are `↑/↓ sel` for selection and
  `pane` for pane focus, with the shortcut letter underlined. Always-on
  command hints are ordered as `help`, `pane`, `monitor`, `enter/attach`,
  `new`, `kill`, `quit`, `layout`, then mode-specific resize. The shortcut
  letters `h`/`p`/`m`/`a`/`n`/`k`/`q`/`l` are underlined in the TUI. The
  bottom status bar must not show a `keys` label, time, host,
  focus (`list`, `detail`, `Lb`, `R`), or layout mode (`portrait`,
  `landscape`, or `normal`) and must render with a blue background.
- The selector must keep `LB`
  consistent with the target host's tmux state without user-driven refresh,
  by subscribing at startup to a host-level event stream. In the current
  implementation that stream polls `list_sessions()` once per second and
  reconciles snapshots by stable session id (not name). Direct tmux
  control-mode host notifications remain a future hardening item; see
  §Data Flow → Live Session List and §Accepted motlie-tmux Library Gaps →
  Host Event Stream.
- Default mode (no behavior flag): on `a` or Enter, leave the TUI cleanly and
  spawn-and-wait attach (see §Data Flow → Attach). Default mode re-enters the
  TUI on clean child exit, or on non-zero child exit when the selected session
  still exists (see §Data Flow → Attach for the bounded re-entry rule). When
  invoked with `--script`, the binary instead leaves the TUI cleanly, prints
  the selected session name (and only the session name) followed by a newline to
  stdout, and exits 0; cancellation (`q`/`Ctrl-C`, no selection) exits non-zero
  with no stdout. All UI rendering, status, and errors go to stderr only.
- The binary accepts explicit layout force flags: `--portrait` / `-p` and
  `--landscape` / `-l`. The flags are mutually exclusive. Portrait mode
  renders a compact layout optimized for 32 rows x ~64
  columns: the body splits vertically into Top (`T`, default focus, lists
  sessions) and Bottom (`B`, detail pane) at a 30:70 ratio. MOTD and the
  motlie placeholder are omitted in portrait mode to maximize content density.
  All command keys (`p` focus cycling, `l` layout toggle, `Esc`/`m`/`n`/`k`/`a`/Enter/`q`/`Ctrl-C`), modal
  behavior, focus model semantics, and detail-source trait usage are
  identical to normal mode (mapping `T` ↔ `Lb` and `B` ↔ `R`). Resize keys
  differ by mode: portrait mode uses `Ctrl-Up`/`Ctrl-Down` to resize `T`/`B`;
  normal mode uses `Ctrl-Left`/`Ctrl-Right` to resize `L`/`R`, with
  modified-arrow fallback sequences accepted for terminal compatibility. Plain
  `Left`/`Right` do not cycle focus in the main view;
  modal use of `Left`/`Right` for button selection is unchanged. Without a
  layout force flag, the selector calls `crossterm::terminal::size()` on the
  connecting PTY and selects portrait mode when `columns / rows <= 4.0`;
  otherwise it uses landscape layout. If the PTY size cannot be read, it
  defaults to landscape layout. The layout force flags compose with `--script`
  and SSH targets, but `l` can still toggle the layout while the TUI is
  running.
- The binary must use `motlie-tmux` for tmux operations and must not duplicate
  tmux command logic in the binary.

### Non-Functional

- Terminal state must be restored on normal exit, attach, error, `q`/`Ctrl-C`, and
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
- Monitor mode is bounded to the rendered current screen. It does not retain a
  raw control-mode transcript in the selector binary, preventing unbounded
  memory growth on busy sessions. Older detail requests use chunked tmux
  scrollback fetches via `LinesRange`.

## System Design

```text
current terminal / SSH client PTY
        |
        v
mmux binary
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
        |      +-- HostHandle::read_text_file(/etc/motd, 64 KiB)
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

### Transport and Latency Architecture

The selector polls `HostHandle::list_sessions()` once per second, per host.
The cost of that tick at the transport level matters — a naive "shell out
to `ssh user@host tmux …`" implementation would pay one SSH handshake per
host per refresh, which is visible latency on a multi-host fleet over a
slow link. The actual transport architecture avoids this by amortizing
connection setup once at startup.

#### Local target

Each `tmux <args>` call is a fresh subprocess (`sh -c "tmux …"`) that
connects to the long-lived **tmux server daemon** over its Unix socket
(`/tmp/tmux-<uid>/default` by default, or a `-L`/`-S` socket). The
subprocess is the tmux *client*; it exits after the command. Per refresh
tick: **one** subprocess per host (the chained `list-sessions \;
list-windows -a` runs in a single tmux invocation — single fork+exec).

`libs/tmux/src/transport.rs::LocalTransport::exec`.

#### SSH target

One **persistent SSH connection per host**, opened once at `connect_fleet`
time and reused for every subsequent command:

```text
mmux process
   ├── russh::client::Handle  (Arc<Mutex<...>>, opened in SshTransport::connect)
   │       │ persistent SSH session (single TCP+SSH handshake)
   │       ▼
   │   sshd on remote host
   │       └── on each exec request: open channel → `sh -c "tmux …"`
   │              → return output → close channel
```

Per command: the existing handle's `channel_open_session()` opens a fresh
**channel** on the persistent SSH connection (not a new SSH session). The
mutex on the handle is held only during channel open; reads happen unlocked,
so concurrent execs on the same connection are allowed (per
`SshTransport::exec` line comment in `libs/tmux/src/transport.rs`). On the
remote side, each channel runs a fresh `sh -c "tmux …"` subprocess via
sshd's exec.

This relies on no `~/.ssh/config` `ControlMaster` setup — the russh-based
handle multiplexes channels over a single TCP connection internally.

Keepalive (`keepalive_interval`) and idle timeout (`inactivity_timeout`) are
configurable via the SSH URI's query params.

#### Multi-host fan-out

`controller::fetch_fleet_rows` runs `HostHandle::list_sessions()` against
all hosts **in parallel** via `futures::join_all`. Each host runs on its
own persistent connection; failures are isolated per host (one host's
`Err` doesn't block the others — `join_all`, not `try_join_all`).

#### Per-tick cost summary

| Setup | Cost per refresh tick |
|---|---|
| Local (single host) | 1 subprocess fork+exec on the local box; tmux client → server Unix socket I/O |
| SSH (single host) | 1 channel-open on the existing russh connection; 1 `sh -c tmux …` exec on the remote box |
| Multi-host (n hosts) | n × per-host cost, **in parallel**. Wall-clock is bounded by the slowest host, not the sum. |

**Key property:** zero SSH handshakes per refresh tick. Handshakes are paid
once at startup; reuse forever after. On a 5-host fleet polling at 1 Hz, the
network cost over an hour is 5 SSH connections (opened once) plus 5×3600
small channel-open + exec round-trips, not 5×3600 SSH handshakes.

#### Attach is a separate path

The attach handoff (Enter/`a` on a session) does **not** go through the russh
transport. It spawns an **external `ssh -t <host> tmux attach-session -t
<name>`** subprocess with the user's stdio inherited from mmux. This satisfies
the "clean PTY handoff, no VTE-in-the-middle" non-functional requirement: tmux
needs direct control of the user's actual TTY, which means inheriting stdio
from a child process spawned by mmux, not pumping bytes through the russh
channel.

Implication: during attach there are effectively two SSH connections to the
same host — mmux's russh handle (idle, persisting) and the external `ssh`
subprocess (interactive, owning the user's TTY). When the user detaches, the
external `ssh` exits and mmux resumes polling on the existing russh handle.
The russh handle does not need to be reopened.

`libs/tmux/src/attach.rs`.

#### Attach during long-running sessions

Network drop while attached: tmux's interactive `ssh` subprocess loses its
connection and the user is dropped back to mmux. mmux's russh handle may also
have been disturbed; its keepalive will detect this and either reconnect on
next exec or surface a transport error to `StatusBanner::Error`. The user can
re-attempt selection on the next refresh tick.

#### Why this matters for the UX

- **First impression:** startup is a parallel fan-out; the user sees the
  selector populated within "slowest-host's TCP+SSH handshake" time, not
  "sum of all hosts'."
- **Tick latency:** recency display refresh is bounded by "slowest host's
  channel-open + exec" — typically tens of milliseconds on a LAN, even over
  WAN it's well under the 1 Hz interval.
- **Attach feel:** PTY handoff is instant — the external `ssh` subprocess
  inherits stdio, so there's no VTE-buffer-flush stutter.
- **Failure feel:** one host going down does not freeze the UI; the failed
  host's rows simply disappear and a banner indicates the failure. Other
  hosts continue ticking.

### Clock Handling

The recency display has two columns — `active` ("how recently did anything
happen?") and `age` ("how long has this session existed?") — and they need
**different clock semantics**.

#### Activity is observer-relative

`active` answers "is this session changing?" That's a question about *our
observation cadence*, not the host's calendar. The binary keeps a per-session
`ActivityTracker` that records, for each `(host_id, session_id)`, the
operator-side wall clock at which we last observed `session.activity` advance:

```rust
struct ActivityState {
    activity_ts: u64,           // host's session_activity at last observation
    observed_at_local: u64,     // operator's wall clock at that moment
}
```

On each refresh tick:

* If `new.activity > prev.activity_ts`: update both fields (movement seen).
* Else: leave both fields alone (no movement).

Recency is `local_now - observed_at_local`. Both endpoints are operator-side
time, so the math is naturally insensitive to host/operator clock skew. Even
on a remote host whose clock drifts, "the binary saw this advance N seconds
ago" stays correct.

**First-sight seeding.** When mmux first observes a session, we have no
previous `activity_ts` to compare against. Seeding `observed_at_local =
local_now` would falsely display "now" for a session that's been idle for
hours. Instead we seed with the reported staleness:

```text
observed_at_local = local_now - max(0, local_now - new_activity)
```

Under the NTP-synced clock assumption (see next section), `new_activity`
and `local_now` come from the same wall clock, so the first displayed
recency reflects the actual idle age at first sight. Subsequent ticks
track observer-relative movement from there.

#### Age uses operator-side `local_now` under an NTP-synced clock assumption

`age` answers "how long has this session existed?" That would ideally be
computed against the host's wall clock — `host_now - session.created` —
but reading the host clock portably across tmux versions is not possible
without side effects:

* `display-message -p '#{epoch}'` was added in tmux 3.7 and returns empty
  on older tmux.
* `run-shell 'date +%s'` is supported across versions but on tmux ≤ 3.4 it
  displays its output **into copy mode in the operator's currently-attached
  pane**, corrupting whatever session the operator is using.
* No other portable format variable returns the current wall clock.

Rather than pick between "broken on old tmux" and "corrupts the user's
session," the design assumes host clocks are NTP-synced with the operator
(typical for any production deployment) and uses the operator-side
`local_now` directly:

```text
age_seconds = local_now - session.created
```

If a host's clock is wildly off, the displayed age is wrong by the skew
amount, but every other consumer of the host's timestamps (logging,
filesystem mtimes, scheduling) is also broken — at that point the host
is misconfigured and recency text is the least of the problems. Recency
buckets (`now`, `Nm`, `Nh`, `Nd`) are coarse enough that sub-second NTP
drift is invisible.

#### Library API surface (deliberate minimalism)

The earlier session contained drafts of `discovery::probe_host_clock`,
`HostHandle::server_epoch()`, `list_sessions_now`, `SessionListing`,
and various `_with_prefix` permutations on the public surface — all
removed. The final shape:

* `motlie_tmux::HostHandle::*` is the only entry point for tmux access.
* `mod discovery` is *private* (`mod`, not `pub mod`) — no
  `motlie_tmux::discovery::*` callable from outside the crate.
* No `_with_socket` shims (all unused after the privatization).
* The `pub(crate)` `_with_prefix` helpers stay as internal plumbing for
  `host.rs` and `capture.rs`; they don't leak past the lib.
* No new `pub` methods were added on `HostHandle` for this PR. Issue
  #237's `window_activity` aggregation lives behind the existing
  `list_sessions()` method.

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
mmux
mmux ssh://user@host
mmux 'ssh://user@host?identity-file=/path/to/key'
```

The accepted v1 CLI target form is a positional SSH URI:
`mmux [ssh-uri]`. Do not add `--target` in v1; keeping a single target
form keeps help text and ForceCommand examples unambiguous. Revisit in a
follow-up only if PLAN finds positional friction with deployment tools.

Additional flags:

| Flag | Behavior |
|------|----------|
| (none) | Default. TUI → select → spawn-and-wait attach (see §Data Flow → Attach). On clean child exit, re-enter the TUI; on non-zero child exit, re-enter only if the selected session still exists, otherwise exit with the child's status. `q`/`Ctrl-C` from the re-entered TUI exits 0 (user-initiated clean exit). |
| `--script` | TUI → select → leave alt-screen → print `<name>\n` to stdout → exit 0. Cancellation exits non-zero with empty stdout. All UI/diagnostics on stderr. Composable: `tmux attach -t "$(mmux --script)"`. |
| `--portrait` / `-p` | Force portrait layout: vertical T/B split (30:70) optimized for 32x64 terminals. MOTD omitted. Same command keys, modal behavior, focus model, and detail sources as normal mode. Resize via `Ctrl-Up`/`Ctrl-Down`. Composes with `--script` and SSH targets. Without a layout force flag, layout is auto-detected from PTY dimensions. See §Layout → Portrait Mode. |
| `--landscape` / `-l` | Force landscape/normal layout: `L`/`R` split with `LT` MOTD and `LB` session list. Composes with `--script` and SSH targets. Mutually exclusive with `--portrait` / `-p`. |
| `--portrait` + `--landscape` | Mutually exclusive — startup error. |

Polarity rationale (default attach/re-enter): the binary's primary product is a
host-wide session selector that keeps users inside the selector workflow across
tmux detaches. Shell-script composition is opt-in via `--script`, which makes
stdout ownership explicit.

ForceCommand-mode incompatibility: `--script` is incompatible with ForceCommand
mode (the user has no shell to consume the output). ForceCommand deployments
must omit the flag; the binary should warn (stderr) on a best-effort heuristic
if both are detected.

## Internal State Model

Implemented selector state is split by concern:

- `HostContext`: display hostname/IP.
- `LayoutState`: mode, focus, and resize percentages.
- `MotdState`: MOTD text and fallback marker.
- `SessionListState`: live `SessionInfo` rows, target-server `now` timestamp,
  selected index, and list scroll.
- `DetailState`: rendered lines, scroll state, source, and auto-tail behavior.
- `StatusBanner`: typed loading/info/error status text for the bottom bar.

`AppState` coordinates those pieces and owns modal state. Render feedback for
detail-pane height is explicitly stored as `last_known_view_height` in
`DetailState` so input handling can compute scroll bounds on the next tick.
The main run loop is kept in `main.rs`. CLI parsing/layout detection,
terminal lifecycle, ForceCommand bypass/reject handling, target-host identity
resolution, detail sources, key handling/event refresh, and rendering live in
`cli.rs`, `terminal.rs`, `forcecommand.rs`, `target_host.rs`, `detail.rs`,
`controller.rs`, and `render.rs`; shared UI data structures live in
`model.rs`.

## Layout

The terminal is split into:

- top status bar: one terminal row
- body area: everything between the top and bottom status bars
- bottom status bar: one terminal row

The body area is split horizontally into `L` and `R`.

`L` is split vertically:

- `LT`: MOTD, height `min(rendered_motd_lines + chrome, 30% of L height)`
  when MOTD is present. When MOTD
  is absent/empty/unreadable, `LT` height = `glyph_rows + caption + chrome`
  (bypasses the 30% cap so the motlie placeholder fully renders when there is
  room). The height calculation uses the post-split `L` area and preserves
  space for `LB`; when the full placeholder does not fit, `LT` falls back to
  the compact motlie placeholder plus caption instead of disappearing. The
  compact/full decision is based on the embedded placeholder's actual line
  count and maximum line width, not a fixed terminal-width threshold. See
  §Functional Requirements for the placeholder rendering rule.
- `LB`: session list, remaining height. The viewport scrolls to keep the
  highlighted row visible. Rows render display names and attachment markers;
  stable tmux session ids are retained in state for dispatch but not shown.

**Focus model.** The landscape main view has three focus states: `LT`, `Lb`
(default), and `R`. Focus transitions are explicit:

- `p` → cycle `LT -> Lb -> R -> LT`
- `Esc` outside any modal: return focus to `Lb` (use `q` or `Ctrl-C` to
  exit). `Esc` inside any modal is equivalent to that modal's `Cancel` button.

The currently focused pane must be visually distinguished from unfocused panes
via border style (bright/colored or doubled for focused; dim/single for
unfocused). The blue top status bar shows bold host/IP at left using `|` as
the separator and
right-justified time. The blue bottom status bar shows key hints and status
text only; it does not duplicate host, time, focus, layout state, or a `keys`
label. Command shortcut letters are rendered with underline styling instead of
parenthesized mnemonics. The Sessions pane title is count-only: `Sessions [n]`.

Main-selector keymap (focus-aware):

| Key | `LT`-focused | `Lb`-focused | `R`-focused |
|-----|--------------|--------------|-------------|
| Up / Down | No-op | Move highlight; LB viewport auto-scrolls | Scroll R one line; on scroll-past-top, sample mode resamples backwards (chunked); monitor mode pins viewport (auto-tail pauses) |
| PgUp / PgDn | No-op | Page through session list | Page through R buffer |
| Home / End | No-op | First / last session | Top / bottom of buffer; `End` re-engages monitor auto-tail |
| `p` | Focus → `Lb` | Focus → `R` | Focus → `LT` |
| Left / Right | No-op | No-op | No-op |
| `Esc` | Focus → `Lb` outside modal; `Cancel` inside modal | Focus → `Lb` outside modal; `Cancel` inside modal | Focus → `Lb` outside modal; `Cancel` inside modal |
| Modified Left / Right | Resize `L`/`R` split (normal mode only; `Ctrl`, Alt, Shift, and word-arrow fallbacks accepted when terminals remap Ctrl-arrow) | Resize `L`/`R` split (normal mode only; focus-independent) | Resize `L`/`R` split (normal mode only; focus-independent) |
| `l` | Toggle portrait/landscape layout | Same | Same |
| `m` | Start/switch monitoring on highlight | Same | Same |
| `n` | Open `New Session` modal | Same | Same |
| `k` | Open kill-confirmation modal | Same | Same |
| `r` | No-op | Open rename modal for highlight | No-op |
| `t` | Open tag edit modal for highlight | Same | Same |
| `i` | Open tag info/add modal for highlight | Same | Same |
| `h` | Open help modal with logo, key functions, and build git SHA | Same | Same |
| Enter / `a` | Attach highlight | Attach highlight (focus-independent) | Attach highlight (focus-independent) |
| `q` / `Ctrl-C` | Exit selector without attach | Exit selector without attach | Exit selector without attach |

Resize keys use modified arrows so plain arrows stay available to terminals and
modal button selection while `p` owns main-view pane cycling. Normal mode advertises
`Ctrl-Left`/`Ctrl-Right` for the L/R split and also accepts common terminal
fallbacks; on macOS iTerm2 the observed fallback is `Shift-Left` /
`Shift-Right`. Portrait mode advertises `Ctrl-Up`/`Ctrl-Down` for the T/B split
and accepts the same modifier family. Resize bounds are mode-specific:
landscape L/R clamps at 25/75, while portrait T/B clamps at 15/85.

Modal keymaps override the main keymap. In modals: Left/Right move between
`Cancel` and `Ok`; `Enter` exits and applies `Ok` if selected; `Esc` is
`Cancel`.

### Portrait Mode

Portrait mode is optimized for compact terminal contexts where horizontal width is
constrained: mobile SSH clients, IDE-embedded terminals, tmux pop-ups
(`display-popup`), and narrow ForceCommand deployments.

**Target dimensions:** 32 rows x ~64 columns. The layout must remain usable
at smaller sizes but is tuned for this target.

**Layout:**

- Body area: 30 rows (32 total minus 1 top status row and 1 bottom status row).
- Body splits *vertically* into Top (`T`) and Bottom (`B`) at a 30:70 ratio
  (T ≈ 9 rows, B ≈ 21 rows for a 32-row terminal).
- `T` = session list. Equivalent to `LB` in normal mode (same scrolling,
  same position indicator, same auto-scroll-to-keep-highlight-visible
  behavior). Default focus.
- `B` = detail pane. Equivalent to `R` in normal mode (same trait-backed
  sample/monitor sources, same scroll-back-on-up, same monitor tail-pause).
- MOTD (`LT`) and the motlie placeholder are **omitted** in portrait mode to
  maximize content density. Status-bar key hints remain, but key hints must be
  terser to fit ~64 cols. Use compact symbol labels for directional keys,
  e.g., `↑/↓ sel | (h)elp | (p)ane | (m)onitor | enter/(a)ttach | (n)ew | (k)ill | (q)uit | (l)ayout`.

**Focus model:** Same semantics as normal mode, except MOTD is not present, so
`p` cycles between `T` and `B`:

- Default focus is `T`.
- `p` → cycle `T -> B -> T`.
- `Esc` outside modal returns focus to `T`.
- Visual focus borders: same rule (bright/doubled for focused; dim/single
  for unfocused).

**Resize keys (mode-dependent):**

| Key | Normal mode | Portrait mode |
|-----|-------------|------------|
| Modified Left / Right | Resize `L`/`R` split, clamped 25/75 | (no-op; `L`/`R` not present) |
| Modified Up / Down | (no-op; `T`/`B` not present) | Resize `T`/`B` split, clamped 15/85 |
| Plain arrows (no Ctrl) | Navigation/scroll per focus-aware keymap above; Left/Right no-op in main view | Navigation/scroll per focus-aware keymap above; Left/Right no-op in main view |

**All other keys and modal behavior:** identical to normal mode (see the
focus-aware keymap above). `m`, `n`, `k`, `t`, `i`, `a`/Enter, and
`q`/`Ctrl-C` are focus-independent and behave the same; `r` remains
list-focus-only (`T` in portrait). `q`/`Ctrl-C` exits without attaching. Modal
keymap (Left/Right for button selection, Enter to apply, Esc to Cancel) is
unchanged.

**Auto-detection and composition:** Without `--portrait` / `-p` or
`--landscape` / `-l`, startup reads the current PTY size through
`crossterm::terminal::size()`. It selects portrait mode when
`columns / rows <= 4.0`; 66x30, 80x24, 100x30, 160x40, and square-ish PTYs use
portrait mode. Wider PTYs use landscape mode. If the size cannot be read,
landscape mode is used. Layout force flags compose with `--script`, SSH
targets, and the `MOTLIE_MMUX_BYPASS` env-var
admin bypass. ForceCommand deployments may use explicit layout flags for fixed
display contexts (`ForceCommand /usr/local/bin/mmux --portrait`
or `ForceCommand /usr/local/bin/mmux --landscape`).

### Multi-host Mode (issue #235)

Multi-host mode is enabled implicitly when **two or more** SSH host arguments are
passed on the command line. With one host (or none) the selector remains in the
existing single-host mode unchanged.

**Activation rule.**

| `len(ssh_hosts)` | Mode |
|---|---|
| `0` | Single-host, target = localhost |
| `1` | Single-host, target = the SSH host |
| `≥ 2` | **Multi-host**, targets = all listed SSH hosts |

**Functional differences in multi-host mode:**

- Top status bar reads `mmux - multi-host mode (n)` where `n` is the host count;
  the single-host hostname/IP indicator is replaced.
- Session list rows insert a hostname column between the attached marker and the
  session name. Format becomes:

  ```
  > * <hostname-padded> <session-name>          <active> / <age>
  ```

  Hostname column width is the max label width across the configured hosts,
  truncated/elided at a sensible cap.
- Sorting remains `SessionInfo.activity` descending — but applied to the
  **merged** list of (host, session) rows across all hosts, not per-host.
- All command keys (`Up`/`Down`, `Enter`/`a` attach, `m` monitor, `n` new,
  `k` kill, `r` rename, `t` tag edit, `i` tag info/add, `Ctrl-C`/`q` exit,
  `l` toggle layout, `p` cycle panes, `Ctrl-←/→` and `Ctrl-↑/↓` resize) behave
  the same as single-host. Each applies to the highlighted row and dispatches
  against that row's host, with `r` still restricted to list-pane focus.
- Attach routes to the highlighted row's host: spawn-and-wait
  `ssh -t <host> tmux attach-session -t <name>` for SSH targets (using each
  host's `SshConfig` carried by its `HostHandle`).
- New session / kill modal dispatch to the highlighted row's host (the row
  whose host is currently selected). Default v1 policy: act on the highlighted
  row's host; no host-picker modal.
- Rename and tag modals also dispatch to the highlighted row's host and capture
  `(host_id, session_id)` when opened so refresh/reorder cannot retarget an
  in-flight modal action.
- MOTD pane is **hidden** in multi-host mode (per-host MOTD is not meaningful
  when multiple hosts coexist in the list). Layout reflows to give the entire
  left column to the session list (landscape) or the top region (portrait).
- Layout flags `-l/--landscape` and `-p/--portrait` still compose with
  multi-host. The auto-detect rule is unchanged.

**Recency math** is observer-relative for activity and operator-clock
relative for age — see §Clock Handling above. Each `SessionRow` carries
`local_now` and `activity_observed_at_local` from the binary's
`ActivityTracker`. Sorting uses `activity_observed_at_local` descending so
the row whose activity most recently advanced as observed by the binary
sits at the top. Sorting on the observer-side mark instead of raw host
`SessionInfo.activity` keeps order stable across multi-host fleets even
when host clocks drift — a host minutes ahead of others can no longer
pin its sessions to the top.

**Resilience.** If one host is unreachable at refresh time, its
`list_sessions()` errors but other hosts proceed; the failed host's rows
disappear from the list until it recovers. A status-banner indicator shows
the per-host failure count without blocking the rest of the UI.

**MotdState** is replaced by an `Option<MotdState>`-style construct in
multi-host: `None` in multi-host mode (no MOTD pane), `Some(motd)` in
single-host mode.

**Detail pane (R / B)** continues to operate on the highlighted row's session,
with the dispatch routed through `row.host_id → fleet.entries[id].handle`.
The `SessionDetailSource` trait does not change shape — it still takes
`(host: &HostHandle, session: &SelectedSession)`. The binary just resolves
`host` from the highlighted row instead of using a single global handle.

**Per-host failure semantics for the detail pane:** if the highlighted row's
host becomes unreachable, the detail pane shows a transient error string and
the row disappears on the next refresh tick; selection drops to the next valid
row (host-event reconciliation already handles this, generalized to multi-host).

#### Internal data model

```rust
// model.rs additions

pub(crate) struct HostId(pub(crate) String);  // ssh URI or "localhost"

pub(crate) struct HostEntry {
    pub(crate) id: HostId,
    pub(crate) handle: motlie_tmux::HostHandle,
    pub(crate) identity: HostIdentity,  // hostname, ip_address, label
}

pub(crate) struct HostFleet {
    pub(crate) entries: Vec<HostEntry>,
}

impl HostFleet {
    pub(crate) fn is_multi(&self) -> bool { self.entries.len() > 1 }
    pub(crate) fn host_label_width(&self) -> usize { /* max label width */ }
}

pub(crate) struct SessionRow {
    pub(crate) host_id: HostId,
    pub(crate) host_label: String,    // pre-resolved for cheap render
    pub(crate) local_now: u64,        // operator wall clock at refresh time
    pub(crate) activity_observed_at_local: u64,  // ActivityTracker mark
    pub(crate) session: motlie_tmux::SessionInfo,
}

// SessionListState becomes:
pub(crate) struct SessionListState {
    pub(crate) rows: Vec<SessionRow>,  // was: Vec<SessionInfo>
    pub(crate) selected: usize,
    pub(crate) scroll: usize,
}
```

`HostContext` is replaced by `HostFleet`; selection-by-id at the row level
must compose `(host_id, session_id)` to remain stable across rename/reorder.

**Why a binary-side `HostFleet`, not `motlie_tmux::Fleet`?** The lib's `Fleet`
(`libs/tmux/src/fleet.rs`) is the *monitoring/automation registry*. Its
load-bearing features — a shared `OutputBus` aggregating output across hosts,
per-host monitor lifecycle (`start_monitoring_session`, `start_monitoring_host`),
workstream alias→`TargetSpec` bindings, and per-session health rollup
(`HostStatus::Monitoring { sessions: Vec<SessionMonitorStatus> }`) — are
orthogonal to what the selector does. The selector needs a flat list of hosts
with per-host display metadata (label, ip) and 1 Hz fan-out polling via
`HostHandle::list_sessions()`. Using `motlie_tmux::Fleet` would require:

- Wrapping it to add display metadata on the side (the lib's `Fleet` stores
  only `HashMap<alias, HostHandle>` — no label/ip);
- Inheriting the alias-match constraint between fleet alias and
  `HostHandle::host_alias()` (`Fleet::register` rejects mismatched aliases),
  which conflicts with using the SSH URI as the stable selector id;
- Carrying the unused `OutputBus` injection side effect on host registration;
- Hand-rolling the `tokio::join_all` fan-out loop anyway, since `Fleet`
  doesn't expose a multi-host listing helper.

The binary-side `HostFleet` is ~30 LoC and fits the selector's concern
exactly without unused dependencies or ceremony. If a future need arises to
share this model across binaries, the right move is to add a new lightweight
`motlie_tmux::HostRegistry` (a vec of `HostHandle`s plus per-host metadata,
without monitoring lifecycle) that coexists with `Fleet`, rather than
overloading `Fleet`. That work is explicitly out of scope here per the
"no library changes for v1" constraint on issue #235.

#### Refresh loop (fan-out + merge)

```rust
async fn refresh_listings(
    fleet: &HostFleet,
    tracker: &mut ActivityTracker,
) -> Vec<SessionRow> {
    let futures = fleet.entries.iter().map(|entry| async move {
        let listing = entry.handle.list_sessions().await;
        (entry, listing)
    });
    let results = futures::future::join_all(futures).await;

    // Capture local_now once so all rows in this tick share the same
    // operator-side observation moment.
    let local_now = local_epoch_seconds();
    let mut rows = Vec::new();
    for (entry, listing) in results {
        match listing {
            Ok(sessions) => {
                for s in sessions {
                    let observed_at_local = tracker.observe(
                        &entry.id, s.id.as_str(), s.activity, local_now,
                    );
                    rows.push(SessionRow {
                        host_id: entry.id.clone(),
                        host_label: entry.identity.label.clone(),
                        local_now,
                        activity_observed_at_local: observed_at_local,
                        session: s,
                    });
                }
            }
            Err(_) => { /* surface to status banner; skip rows */ }
        }
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.activity_observed_at_local));
    rows
}
```

Use `join_all` (not `try_join_all`) so one host's failure doesn't drop the
others. Errors are collected per host and fed to the status banner.

The same path runs in single-host mode (`fleet.entries.len() == 1`); no
divergent code path. This satisfies the "no duplication just because of the
different views" requirement.

#### Render: row format

`render::draw_sessions` shifts to a single render path that emits the
hostname column **only when `fleet.is_multi()`**. The column width is taken
from `HostFleet::host_label_width()`. The host-label column is omitted when
`is_multi()` is false, so single-host rendering is unchanged.

#### Top status bar

`render::draw_top_status` switches on `fleet.is_multi()`:

- Single: `<hostname> | <ip>                                     <time>`
- Multi:  `mmux - multi-host mode (<n>)                            <time>`

#### Scope and impact analysis

**Library (`libs/tmux/`):** *No new public API surface required.* The existing
`HostHandle::list_sessions()`, `session_by_id()`, and
`Target::attach_current_pty()` cover everything per-host. Multi-host fan-out
is a binary-side concern (orchestration, not protocol). Optionally, the
existing `Fleet` type could grow a `list_sessions_all() -> Vec<(HostId, Result<Vec<SessionInfo>>)>`
convenience method, but this is purely sugar over `tokio::join_all` and is
not required for v1.

**Binary (`bins/mmux/`):**

| File | Change |
|---|---|
| `cli.rs` | `ssh_uri: Option<String>` → `ssh_uris: Vec<String>` (clap `num_args = 0..`). |
| `model.rs` | Add `HostId`, `HostEntry`, `HostFleet`, `SessionRow` types. Replace `HostContext` (single host) with `HostFleet`. Change `SessionListState.sessions: Vec<SessionInfo>` to `SessionListState.rows: Vec<SessionRow>`. Make `MotdState` an `Option<MotdState>` field on `AppState`. |
| `target_host.rs` | Rename / split: `connect_host(cli) → connect_fleet(cli) -> Result<HostFleet>`. Internally calls existing single-host connect for each entry. |
| `controller.rs` | `refresh_sessions` operates on `HostFleet`; uses `join_all` for fan-out; builds merged sorted `Vec<SessionRow>`. `load_motd` only called when `fleet.is_multi() == false`. New session / kill / attach paths take the highlighted `SessionRow` and dispatch via `fleet.entry(row.host_id).handle`. |
| `render.rs` | Single render path. `draw_sessions` adds optional hostname column when `fleet.is_multi()`. `draw_top_status` switches text by mode. `draw_motd` is gated on `app.motd.is_some()` (already gated in portrait — generalize to multi-host). Status hint set unchanged. |
| `detail.rs` | No shape change. Caller passes the row's `&HostHandle`. |
| `main.rs` | Calls `connect_fleet` instead of `connect_host`. |
| `forcecommand.rs` | No change (ForceCommand stays single-host; multi-host is operator-mode). |
| `tests.rs` | New tests for: multi-host CLI parsing; fleet construction; merge-and-sort across hosts; row hostname column; top-status switching; per-host failure resilience; selection-by-(host_id, session_id) preservation across reorders. |

**No abstraction changes** to: `consts.rs`, `terminal.rs`, `forcecommand.rs`,
`detail.rs` shape (only call-site changes inside `controller.rs`).

**Estimated diff** (binary only): ~+700 / −200 lines. No library changes
expected for v1.

### Session Rename and Tags (issue #241)

Session rename and tag editing are session-list affordances layered on the
existing stable-id dispatch model.

**Scope and routing**:

- `r` is active only when the list pane is focused (`LB` in landscape, `T` in
  portrait). This avoids stealing `r` from detail-pane scrolling or future
  detail-pane commands.
- `t` and `i` are selected-session actions like `m` and `k`; they operate on
  the highlighted row regardless of pane focus.
- Each modal captures `(host_id, session_id, current_session_name)` on open.
  Apply paths re-resolve `HostFleet::entry(host_id)` and
  `HostHandle::session_by_id(session_id)` before writing.
- If the target session disappears before apply, close or refresh the modal as
  appropriate and show a non-fatal status banner.

**motlie-tmux API gap**:

- tmux supports unsetting user-defined options with
  `set-option -u -t <session-id> @mmux/<key>`. For this feature, `motlie-tmux`
  must expose that operation instead of making `mmux` shell out directly.
- Add `SessionTags::unset(key) -> Result<()>` and a one-off
  `Target::unset_tag(prefix, key) -> Result<()>`.
- The library implementation should mirror the existing tag API contracts:
  session targets only, stable session-id dispatch, validated prefix/key, no
  value argument, and `set-option -u` under the existing control-layer command
  boundary.

**Rename modal**:

- State shape: `RenameSession { host_id, id, current_name, input, button }`.
- Render title `Rename Session`; render one bordered text field labeled
  `Session Name`, prepopulated with `current_name`.
- `Cancel`, `Esc`, or Enter on focused `Cancel` dismisses without action.
- `Ok` trims the submitted name using the same rule as the New Session modal.
  Empty input is rejected with a status banner. If the trimmed name equals
  `current_name`, close without calling tmux. Otherwise call `Target::rename`
  through `motlie-tmux`, then refresh sessions immediately.

**Tag edit modal (`t`)**:

- State shape: `EditSessionTag { host_id, id, name, tag, value, focus,
  button, value_dirty }`, where `focus` is modal-local (`Tag`, `Value`,
  `Button(Button)`) rather than a global cross-modal enum.
- Render title `Session Tag`; render bordered fields labeled `Tag` and
  `Value`, then the standard `Cancel` / `Ok` button row.
- Normalize the tag key by trimming surrounding whitespace. Do not prepend
  `mmux/` in state; pass the stripped key to `target.tags("mmux").await?`.
- When the tag field changes to a non-empty valid key and `value_dirty` is
  false, read `tags.read(key).await?`; if present, prepopulate `Value`.
  Invalid or missing keys leave the current value untouched except for status
  feedback.
- On `Ok`, do nothing when `Value` is empty. Otherwise call
  `tags.set(key, value).await?`; value text is not trimmed so intentional
  leading/trailing spaces survive, while the library still rejects control
  characters and overlarge values.

**Tag info/add modal (`i`)**:

- State shape: `SessionTagsInfo { host_id, id, name, tags, key_input,
  value_input, focus }`, where `focus` is modal-local (`TagRow(index)`, `Key`,
  `Value`, `Add`, `Cancel`).
- On open, call `target.tags("mmux").await?.list().await?`, sort by
  `SessionTag::key()`, and render stripped keys (for example `owner`, never
  `mmux/owner` or `@mmux/owner`). Initial focus is the first tag row when any
  tag exists, otherwise the bottom `Key` field.
- Render the key/value list in a scrollable body if it exceeds the modal's
  available height. The add row stays visible at the bottom and contains
  bordered `Key` and `Value` fields plus a focusable `+`.
- Up/Down move focus row-to-row through the visible tag list when a tag row is
  focused. If focus is in the add controls, Up returns focus to the last
  visible tag row when one exists.
- `x` on a focused tag row calls `tags.unset(key).await?`, reloads and resorts
  the tag list, and keeps the modal open. If the deleted row was the last row,
  focus moves to the previous row or to the `Key` field when the list becomes
  empty.
- `u` on a focused tag row copies that key/value into the bottom `Key` and
  `Value` fields and focuses `Value`. Pressing Enter on `+` then updates the
  same key by calling `tags.set(key, value).await?`.
- Enter on focused `+` applies the same add rule as the `t` modal:
  non-empty value only, stripped key, `mmux` namespace, motlie-tmux tag API.
  On success, refresh the modal's tag list and clear the add fields. If the key
  already existed, this is an update; otherwise it is an add.
- Enter on focused `Cancel` or `Esc` dismisses without writing.

This feature requires one small `motlie-tmux` public API addition for tag
delete. It otherwise depends on `Target::rename`,
`HostHandle::session_by_id`, and the `Target::tags("mmux")` helper added for
session metadata.

## SVG Mock

The DESIGN mock source is checked in beside this document:

![mmux TUI mock](./mmux-mock.svg)

If GitHub issue rendering supports the chosen SVG embedding path, this same SVG
should be attached or linked from issue #226 after the branch is pushed.

The SVG mock includes the following panels:

1. Main selector view, `Lb`-focused.
2. Main selector view, `R`-focused.
3. Monitor mode with `R` scrolled up and auto-tail paused.
4. `New Session` modal.
5. Kill confirmation modal with title `Kill session <name>?`.
6. MOTD-absent state with bold-green motlie glyph placeholder.
7. Portrait mode main view with focused `T`.
8. Portrait mode focused-`B` variant.

The mock is conceptual; current implementation uses `p` to cycle focus through
`LT`, `Lb`, and `R` in landscape mode.

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
    // `Target::sample_text_with_options(&ScrollbackQuery::LinesRange { ... },
    // ScreenStable, None)` — see §Accepted Library Gaps.
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
  session content with `sample_text_with_options(..., ScreenStable, ...)` so
  ANSI color/style escapes are preserved, then renders through
  `ansi-to-tui`'s VTE parser in `R`.
  `fetch_older` issues
  `Target::sample_text_with_options(&ScrollbackQuery::LinesRange { older_than_lines, count }, ScreenStable, None)`
  for paginated backwards fetch (see §Accepted Library Gaps →
  ScrollbackQuery::LinesRange).
- `MonitorDetailSource`: resolves the selected session by stable id, captures
  the rendered current screen for all panes through `capture_all_with_options`
  using `CaptureNormalizeMode::ScreenStable`, and renders ANSI with
  `ansi-to-tui`'s VTE parser in `R`. When the user scrolls up in monitor mode,
  auto-tail pauses; refreshes continue, but the UI viewport stays anchored at
  the user's position. `fetch_older` for monitor mode falls back to a one-shot
  `Target::sample_text_with_options(&LinesRange { ... }, ScreenStable, None)`
  against the same target. The
  monitor screen mirror is a current-screen source, not a rolling transcript
  source.

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

1. Parse CLI target and flags (`--portrait` / `-p`, `--landscape` / `-l`,
   `--script`; `--portrait` and `--landscape` are mutually exclusive — error
   on both).
2. Connect to local or SSH target with `motlie-tmux`.
3. Load target host MOTD (or render the motlie placeholder when absent).
4. List sessions.
5. Subscribe to the host-level event stream (see §Live Session List). The v1
   stream itself is polling-backed snapshot reconciliation; on stream failure,
   keep the current snapshot and show an inline error status.
6. Initialize UI state with `LB` focused and first session highlighted.
7. Render sample detail for the highlighted session, if any.

### Live Session List

The `LB` list must stay consistent with the target host's tmux state without
user-driven refresh. Other clients may create, kill, or rename sessions;
sessions may exit unexpectedly. The selector must reconcile.

Shipped selector mechanism: `mmux` runs one quiet `list_sessions()` refresh
per second per host. That snapshot reconciles structural session state by
stable session id, runs the `ActivityTracker` to update each row's
observer-side `activity_observed_at_local`, re-sorts rows by that mark
descending across all hosts, preserves the highlight by `(host_id,
session_id)`, and only triggers a detail re-render when the highlighted
row moved, the caller forced it, or the monitored session just closed
(see §Refresh path below). On transient refresh failure, the TUI keeps
running and reports the error in the status bar.

Issue #229 library support adds `SessionInfo.activity` and
`SessionInfo.attached_count`. Issue #237 folds `max(window_activity)` into
`SessionInfo.activity` so the field reflects "any program output OR any
client input" rather than only attached-client input. Recency math is
binary-side: see §System Design → Clock Handling for the rationale, and the
*Refresh path* immediately below for the loop shape.

Future hardening target: tmux control-mode notifications. The library already
parses `%`-prefixed control-mode lines as `ControlModeMessage::Notification`
(`libs/tmux/src/monitor.rs:58–96`) but currently discards them
(`monitor.rs:337–341`). A later implementation can wire those notifications
into either the library `HostEventStream` contract or the selector's single
snapshot refresh path without changing the selector UI model.

#### Refresh path

Single-poll reconcile loop driven by the main TUI loop:

1. On startup, call `connect_fleet(cli)` to build the `HostFleet` (rejects
   duplicate SSH URIs up-front), then run an initial fan-out
   `controller::fetch_fleet_rows(fleet, &mut tracker)` and merge into the
   `SessionListState` by stable `(host_id, session_id)` key.
2. Every second, call `refresh_sessions_quiet(...)` which re-runs the
   fan-out and updates rows + tracker state.
3. For each snapshot:
   - feed each `SessionInfo.activity` through `ActivityTracker::observe`
     to compute the row's `activity_observed_at_local`
   - sort rows by `activity_observed_at_local` descending; tie-break by
     name, session id, host id
   - preserve highlight by `(host_id, session_id)`, falling back to the
     clamped index if the session disappeared
   - if the monitored session id is absent, stop monitor mode, switch the
     detail source back to Sample, and report that the monitored session
     closed
   - call `refresh_detail` **only** when the caller forced it, the
     selection actually moved, or monitor was just closed; quiet refreshes
     that find nothing changed leave the detail pane alone (the main loop
     owns the live-monitor recapture cadence on its own 750 ms tick)
   - on per-host polling failure, surface the failed host's label in the
     status banner; other hosts continue to populate the merged list
4. Reconciliation must preserve the user's highlight when possible: if the
   highlighted `(host_id, session_id)` still exists, keep it highlighted;
   otherwise clamp to the previous index.
5. Empty-list state (zero sessions): see §Empty Session List below.
6. In default attach/re-enter mode, the active TUI loop stops before attach.
   On re-entry the selector takes a fresh fan-out snapshot before the first
   redraw.

Polling semantics: the selector re-issues `list_sessions()` every 1 s per
host. The library `HostHandle::watch_host_events()` remains available for
other consumers, but the selector TUI no longer starts it as a second
polling loop.

### Empty Session List

When the target host has zero tmux sessions (at startup, or after a kill during
selector re-entry):

1. `LB` renders an inline placeholder row: `(no sessions on <host> — press n
   to create)`.
2. `R` renders nothing (or an inline hint mirroring the same `n to create`
   message).
3. Highlight is unset; `m`, `k`, `r`, `t`, `i`, `a`, and `Enter` are all
   no-ops in this state.
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
3. Resolve the highlighted session by stable id and capture its rendered screen
   using `capture_all_with_options(CaptureNormalizeMode::ScreenStable)`.
   Render ANSI through the VTE parser so TUI screen snapshots display without
   raw escape bytes.
4. Status bar shows the monitored session.
5. When focus is `R`, scrolling
   up pauses auto-tail; refreshes still replace the source's current-screen
   buffer but the viewport stays anchored. `End` re-engages auto-tail.
6. Killing the monitored session or exiting the TUI stops monitor state.

### New Session

1. Pressing `n` opens the modal.
2. User enters a name and selects `Ok`.
   The text field is bordered, and the button bar is separated from the
   padded content area by a horizontal rule.
3. Call `HostHandle::create_session(name, &Default::default())`.
4. Refresh session list.
5. Highlight the created session.
6. Refresh `R` detail.

### Kill Session

1. Pressing `k` opens confirmation for the highlighted session.
2. User selects `Ok`.
   The confirmation text is padded away from the modal border, and the button
   bar is separated from content by a horizontal rule.
3. On kill-modal-open, capture the stable session id from the highlighted
   `SessionInfo` and dispatch the kill against that id, not the display name.
   If the session was killed by another client between list and resolve,
   surface a brief inline status message ("session already gone") and let the
   host-event subscription's reconciliation refresh `LB` — do not error out.
4. Call `Target::kill()`. On error (connection dropped, permission), show
   inline error without corrupting terminal state.
5. Stop monitor state if it was monitoring that session.
6. Refresh immediately after a successful kill for responsive feedback; the
   polling-backed host-event stream will reconcile the same state on its next
   tick as a backstop.
7. Move highlight to the next valid row. If the killed session was the only
   one, transition to §Empty Session List state.

### Rename Session

1. Pressing `r` while the session list pane is focused opens the rename modal
   for the highlighted session.
2. The modal captures the highlighted row's host id, stable session id, and
   current session name at open.
3. On `Ok`, compare the trimmed input with the captured current name. If
   unchanged, close without I/O. If changed, re-resolve the target by stable id
   and call `Target::rename`.
4. Refresh sessions immediately after a successful rename and preserve
   selection by `(host_id, session_id)`.

### Edit Session Tag

1. Pressing `t` opens the tag edit modal for the highlighted session.
2. As the user completes a valid tag key, use `Target::tags("mmux").await?`
   and `SessionTags::read(key).await?` to prepopulate `Value` when the tag
   already exists and the user has not manually edited the value field.
3. On `Ok`, do not write empty values. Otherwise call
   `SessionTags::set(key, value).await?`.
4. Refresh sessions after a successful write so any future row metadata
   derived from tags can update through the normal path.

### Session Tags Info

1. Pressing `i` opens the tag info/add modal for the highlighted session.
2. Load `Target::tags("mmux").await?.list().await?`, sort by stripped key, and
   display each key/value pair. Focus the first row if present, otherwise focus
   the bottom `Key` field.
3. Up/Down move focus through the displayed tag rows. Pressing `x` on a
   focused row calls `SessionTags::unset(key).await?`, then reloads and resorts
   the modal list.
4. Pressing `u` on a focused row copies that key/value into the bottom edit
   fields and focuses `Value`.
5. Enter on the focused `+` applies the bottom key/value fields with the same
   rules as the `t` modal. Existing keys are updated via `SessionTags::set`;
   new keys are added.
6. On successful add/update, reload the modal's list and keep the modal open
   for additional edits. `Esc` or Enter on focused `Cancel` closes the modal.

### Attach

The attach handoff transfers the user's controlling terminal directly to
the spawned tmux (or `ssh tmux`) child. **No VTE-in-the-middle.**

1. Pressing Enter or `a` in the main selector (any focus) records the
   highlighted session id.
2. Stop monitor/detail state. Drop the active host-event subscription; re-entry
   starts from a fresh session snapshot.
   The parent process also snapshots a small amount of UI state in memory:
   selected session id, selected list index, layout mode, focused pane, and
   the current layout split percentages. This state is only for the current
   `mmux` process and is not written to disk.
3. Restore raw mode and leave the alternate screen. Restore termios to
   canonical state.
4. Resolve the highlighted session id to a `Target` via the stable-id
   library path. If the session vanished between selection and resolve
   (race), show stderr message and re-enter the TUI.
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
7. On `wait()` return, branch on mode and child exit status:

   ```text
   wait() returns
       │
       ├── --script ───────→ (unreachable; --script bypasses attach)
       │
       └── default mode ───→ if child.status.success()
                                 or selected session still exists:
                                 re-enter TUI:
                                   1. re-acquire alt-screen, raw mode
                                   2. refresh list_sessions()
                                   3. start a new host-event subscription
                                   4. re-render LB (state may have changed)
                                   5. restore selected session by id, or the
                                      previous list index if the id vanished
                                   6. restore layout mode, pane focus, and
                                      split percentages
                                   7. if list_sessions() refresh fails →
                                        exit with that error (bounded loop:
                                        no infinite re-entry on broken target)
                                 else (non-zero child exit and selected
                                 session is gone):
                                   exit with child.status.
                                 q/Ctrl-C from re-entered TUI exits the
                                 binary with code 0 (user-initiated).
   ```

8. Detach status is treated conservatively. Some tmux/SSH attach paths can
   report non-zero even when the user detached and the selected session still
   exists. Default mode therefore uses child success as sufficient for
   re-entry, and for non-zero exits probes the selected session id before
   deciding whether to re-enter or propagate the child status.

9. Process count footprint: two processes are resident during the attach window
   (selector + child). Under exec-replace this would be one, but exec-replace
   forecloses recovery, observability, and testability — rejected.

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

The library exposes a host-level event stream for consumers that want typed
session-change events. The selector originally used this stream, but now keeps
`LB` consistent through its single `list_sessions()` refresh loop (see
§Data Flow → Live Session List).

Implemented v1 behavior: `watch_host_events()` is a polling-backed typed event
stream. It reconciles `list_sessions()` snapshots once per second by stable
session id and emits `SessionAdded`, `SessionClosed`, `SessionRenamed`,
`ClientAttached`, `ClientDetached`, and `Disconnect` events. It does not yet
open a host-scoped tmux control-mode notification connection.

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
    Disconnect { reason: String },                    // event source/listing failed
}
```

Future implementation target: open a single shared `tmux -C` connection per
host (e.g., `tmux -C new-session -d -s motlie-events` or attach to a
long-lived sentinel session), surface the already-parsed `Notification` lines
as typed `HostEvent`s. Reconnect transparently on transient drops; emit
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

The binary uses `HostHandle::read_text_file()` to retrieve `/etc/motd` from
local, mock, or SSH targets without embedding shell syntax in the selector.
The API requires a byte cap so a misconfigured MOTD cannot trigger an
unbounded read:

```rust
impl HostHandle {
    pub async fn read_text_file(
        &self,
        path: &std::path::Path,
        max_bytes: usize,
    ) -> Result<String>;
}
```

For v1, `mmux` calls `read_text_file(Path::new("/etc/motd"), 64 * 1024)` and
uses the embedded motlie placeholder when the read fails or returns only
whitespace. The implementation exposes the same policy internally as
`load_motd_from(host, path)` so tests can exercise missing, empty,
whitespace-only, oversized, and readable files without mutating the host's real
`/etc/motd`.

## Host-Wide SSH Integration

The local-host deployment target is `ForceCommand`.

```text
Match Group tmux-users
    PermitTTY yes
    ForceCommand /usr/local/bin/mmux
```

Operational behavior:

1. `sshd` allocates the user's PTY.
2. `sshd` starts `mmux` instead of the login shell.
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
the binary reads the environment variable `MOTLIE_MMUX_BYPASS` at
startup. If unset or empty, `SSH_ORIGINAL_COMMAND` is rejected with a stderr
message and the binary exits non-zero. If set exactly to `1`, the binary exec's
`SSH_ORIGINAL_COMMAND` via the user's login shell
(`/bin/sh -c "$SSH_ORIGINAL_COMMAND"`) and bypasses the TUI entirely. Any other
value is treated as no bypass.
Deployments enable this by adding `AcceptEnv MOTLIE_MMUX_BYPASS` to
`sshd_config` for the relevant `Match Group` (or by setting the variable
via PAM/login.defs for specific users/groups). This keeps the bypass
configuration external to the binary while giving PLAN a concrete
mechanism to implement and test.

Env-gated SSH/ForceCommand integration coverage is tracked in
[issue #232](https://github.com/chungers/motlie/issues/232).

ForceCommand deployments must
NOT use `--script` (the user has no shell to consume stdout).
Recommended deployments:

```text
# Default: TUI selector with attach/re-enter
ForceCommand /usr/local/bin/mmux
```

## Approach (Selected)

**A. New Binary Built Directly On motlie-tmux** — adopted as the main body
of this DESIGN. Create `mmux` as a focused binary that uses
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
| `ansi-to-tui` | ANSI/VTE rendering for captured/monitored pane content | Adopted for sample and monitor modes so ANSI-preserving captures render color/style without leaking escape bytes into ratatui text. |
| `async-trait` | async detail-source trait | Use if a trait object or async trait implementation is needed. Already used in repo. |
| `tempfile` | none in `mmux` | Not needed by the selector binary because MOTD reads go through `HostHandle::read_text_file()`. |

## Testing Strategy

DESIGN identifies the test surfaces; PLAN must make these concrete.

- Unit tests for layout calculations:
  - MOTD height cap (present case)
  - MOTD-absent placeholder
    expansion: `LT` height = `glyph_rows + caption + chrome`, bypasses 30%
    cap; narrow-terminal fallback still renders compact glyph art
  - status bar reservation
  - `L` / `R` resize bounds: landscape clamps at 25/75
  - Portrait mode layout at
    64x32 viewport: body = 31 rows; T/B split at 30:70 yields T ≈ 9 rows
    and B ≈ 22 rows; MOTD/motlie omitted; status bar present
  - Portrait mode modified Up/Down resize bounds: portrait clamps at 15/85
  - PTY aspect-ratio auto-detection: 64x32, 66x30, 80x24, 100x30, 160x40,
    and square-ish PTYs select portrait; 161x40 and wider-than-4.0 PTYs select
    landscape; `--portrait` forces portrait; `--landscape` forces landscape
  - `p` cycles pane focus in the main view; modal use of Left/Right for button
    selection is unchanged
  - `l` toggles layout mode and normalizes focus when switching to portrait
- Unit tests for state transitions:
  - highlight movement
  - sample vs monitor mode
  - modal button selection
  - create/kill success and error paths
  - Help modal opens on `h`, shows the logo, build date, last 8 characters of
    the build git SHA, key functions, and closes on Enter or `Esc`
  - focus cycling: `p` cycles `LT`→`Lb`→`R`→`LT`; `Esc` outside modal returns
    focus to `Lb`
  - `Esc` inside modal = `Cancel`
- Style/snapshot tests:
  - motlie glyph placeholder spans carry `Modifier::BOLD` and `Color::Green`
  - focused pane border style differs from unfocused pane border style
- Mock-backed tests through `motlie-tmux`:
  - session list rendering
  - detail source rendering, including ANSI color preservation in sample and
    monitor modes
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
  - `q`/`Ctrl-C` behavior
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
  - `--script` contract:
    stdout is exactly `<name>\n` on selection; empty on cancel; exit code 0
    on selection, non-zero on cancel; stderr can carry diagnostics without
    polluting stdout (assert via captured stdout in non-TTY harness)
  - default re-entry on clean detach:
    attach to a session, detach via `C-b d`, assert selector re-enters TUI and
    the session remains visible in refreshed `LB`
  - default re-entry on non-zero detach with surviving session:
    simulate or trigger an attach child non-zero status while
    `session_by_id()` still finds the selected session; assert selector
    re-enters TUI
  - default no-loop on
    failure: attach, force a non-zero child exit (e.g., target session
    vanished, or kill-server), assert selector exits with that status (no
    re-entry)
  - default no-loop on
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
    `MOTLIE_MMUX_BYPASS=1` and present

## Open Questions

Previously open questions that materially affect v1 are resolved below. Items
that remain speculative stay explicitly open.

### Decided

- **CLI form** — Positional SSH URI only (`mmux [ssh-uri]`). No
  `--target` flag in v1. Revisit if PLAN finds positional friction.
- **Modal `Esc`** — `Esc` in any modal is equivalent to `Cancel`.
- **`Esc` outside modal** — Return focus to the session-list pane.
- **Monitor follow on highlight change** — No automatic follow. Monitor only
  switches when the user explicitly presses `m` on a different highlight.
  (Unchanged from initial DESIGN; reaffirmed.)
- **`New Session` options in v1** — Defaults only (no window size / history
  flags). Future enhancement.
- **Remote targets in ForceCommand** — Local-only ForceCommand initially.
  Operator-invoked CLI mode may pass an SSH URI.
- **Main-view pane key** — `p` cycles landscape focus `LT -> Lb -> R -> LT`.
  Portrait mode cycles `T <-> B` because MOTD is omitted. Plain Left/Right are
  no-ops in main view, modified arrows own resize, and modal Left/Right keeps
  button selection behavior.
- **Runtime layout toggle** — `l` toggles between portrait and landscape
  layout. The selected CLI force flag controls startup only; runtime `l`
  toggles are intentionally allowed and retained only in memory for the current
  parent process.
- **Portrait-mode status hints** — ASCII-first compact labels. Unicode affordance
  glyphs can be considered later, but v1 must render predictably in narrow
  SSH clients and IDE terminals.
- **Monitor history bound** — Superseded by screen-mirror monitor mode. The
  selector keeps only the current rendered screen in monitor mode; historical
  fetch remains chunked through tmux scrollback.

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
