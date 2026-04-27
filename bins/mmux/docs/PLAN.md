# mmux Implementation Plan

## Status

Implementation plan for the `mmux` selector described in
[DESIGN.md](./DESIGN.md). The first implementation pass has landed on the
`@gpt55-dgx/session-selector-impl` branch with the selector binary, the accepted
current-PTY attach gap, windowed scrollback, stable session-id dispatch, and a
host event stream backed by stable-id snapshot reconciliation.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-27 | @gpt55-dgx | Updated status tracking for `|` host/IP separator and `(h)elp`-first bottom command hints. |
| 2026-04-27 | @gpt55-dgx | Updated status/title tracking for a top host/time status bar and count-only Sessions title. |
| 2026-04-27 | @gpt55-dgx | Updated focus/input tracking for cyclic Left/Right pane navigation, including landscape MOTD focus. |
| 2026-04-27 | @gpt55-dgx | Renamed the selector workspace package/path/binary references to `motlie-mmux`, `bins/mmux`, and `mmux`. |
| 2026-04-27 | @gpt55-dgx | Updated Sessions title tracking for count/hostname/IP format and removed the `keys` status label. |
| 2026-04-27 | @gpt55-dgx | Moved host-label tracking from the status bar to the Sessions pane title. |
| 2026-04-27 | @gpt55-dgx | Updated status hint tracking to use arrow symbols and expanded help modal coverage for key functions. |
| 2026-04-27 | @gpt55-dgx | Changed portrait mode implementation tracking from a 40:60 to a 30:70 T/B split. |
| 2026-04-26 | @gpt55-dgx | Added implementation tracking for the `h` About modal with build git SHA display and Enter/Esc close behavior. |
| 2026-04-26 | @gpt55-dgx | Removed focus labels from the status bar because focused panes are already indicated by border styling. |
| 2026-04-26 | @gpt55-dgx | Updated status bar tracking: omit layout labels from the status text and render the bar with a blue background. |
| 2026-04-26 | @gpt55-dgx | Finalized the CLI mode contract: default mode is attach-and-reenter selector behavior, and `--script` replaces `--print-session` / `--dashboard` for shell integration. |
| 2026-04-26 | @gpt55-dgx | Added `--portrait/-p` and `--landscape/-l` force flags and changed auto-detection to `columns / rows <= 4.0`, making 66x30 portrait. |
| 2026-04-26 | @gpt55-dgx | Set portrait auto-detection to `columns / rows <= 2.0`, updated layout test targets, and embedded the `/tmp/motlie-TOP-CHOICE.txt` glyph as the MOTD-absent fallback icon. |
| 2026-04-26 | @gpt55-dgx | Replaced short mode tracking with portrait mode: `--portrait`, PTY aspect-ratio auto-detection, old `-s` rejection, updated layout/test references, and the requested Claude artifact ASCII logo. |
| 2026-04-26 | @gpt55-dgx | Updated implementation tracking for validation changes: Enter/`a` attach, Left/Right focus transitions, macOS iTerm2 Shift-arrow resize documentation, ANSI-preserving sample detail, polling-backed session refresh, and compact graphical MOTD fallback. |
| 2026-04-26 | @gpt55-dgx | Addressed second manual validation feedback: monitor mode now mirrors rendered screen snapshots with ANSI/VTE parsing, modified-arrow fallback resize is tested, and attach PTY restore uses a `SIGTTOU`-safe foreground-process-group path. |
| 2026-04-26 | @gpt55-dgx | Addressed manual validation feedback: robust Ctrl-arrow resize matching, readable monitor-mode normalization, conventional detail scroll direction with scrollbar/range indicator, `q` quit key, and dashboard re-entry after detach when the selected session still exists. |
| 2026-04-26 | @gpt55-dgx | Implemented the initial selector binary and remaining library support: workspace package, CLI modes, normal/short TUI layouts, MOTD fallback art, trait-backed sample/monitor detail sources, create/kill modals, stable-id attach/kill, ForceCommand bypass/reject handling, `ScrollbackQuery::LinesRange`, host event diff stream, and docs/API/CLI updates. |
| 2026-04-26 | @gpt55-dgx | Started Phase 1.1 and 1.4 implementation: added `Target::attach_current_pty`, `AttachExit`, local/SSH attach command construction, process-group terminal handoff, signal status mapping, command/status unit tests, and `HostHandle::session_by_id`; localhost PTY smoke and rename-race tests remain open. |
| 2026-04-26 | @gpt55-dgx | Initial PLAN for issue #226 and PR #227 re-review: orders accepted `motlie-tmux` gaps before binary work, defines the selector phases, and makes the test harness concrete. |

## Scope

This is greenfield product work. There is no migration or backward
compatibility requirement for an older selector binary.

Implementation must proceed in this order:

1. Add the `motlie-tmux` capabilities the selector depends on.
2. Add the binary scaffold under `bins/mmux`.
3. Build the TUI in slices that are independently testable.
4. Add attach and deployment flows after terminal cleanup is reliable.

The binary must not duplicate tmux command construction or SSH attach logic
that belongs in `motlie-tmux`.

## Dev Harness

Use a dedicated tmux socket for local tests so developer sessions are not
modified:

```bash
export MOTLIE_MMUX_SOCKET=motlie-select-test
tmux -L "$MOTLIE_MMUX_SOCKET" start-server
```

Recommended verification commands as phases land:

```bash
cargo fmt --all
cargo check -p motlie-tmux
cargo test -p motlie-tmux
cargo check -p motlie-mmux
cargo test -p motlie-mmux
cargo clippy -p motlie-tmux -- -D warnings
cargo clippy -p motlie-mmux -- -D warnings
```

The `motlie-mmux` package now exists under `bins/mmux`.

SSH integration tests should be env-gated so normal local test runs do not
require an SSH daemon:

```bash
export MOTLIE_MMUX_SSH_URI='ssh://user@host?identity-file=/path/to/key'
cargo test -p motlie-mmux --test ssh_integration -- --ignored
```

## Phase 1: motlie-tmux Library Gaps

### 1.1 Current PTY Attach

References: [Current PTY Attach](./DESIGN.md#current-pty-attach),
[Attach](./DESIGN.md#attach), [Non-Functional](./DESIGN.md#non-functional).

- [x] 1.1a Add `Target::attach_current_pty(&self) -> Result<AttachExit>` in
  `libs/tmux` with spawn-and-wait semantics.
- [x] 1.1b Local attach spawns `tmux attach-session -t <target>` with inherited
  stdio, no pipe wrapping.
- [x] 1.1c SSH attach spawns `ssh -t ... tmux attach-session -t <target>` with
  inherited stdio, using `SshConfig` connection information owned by the
  library.
- [x] 1.1d Put the attach child in its own process group and transfer the
  controlling terminal with `tcsetpgrp`; restore the selector process group
  after `wait()`.
- [x] 1.1e Translate signal termination to `128 + signal` in `AttachExit`.
- [x] 1.1f Add unit tests for command construction and process-status mapping.
- [ ] 1.1g Add localhost smoke coverage that verifies terminal state is restored
  before and after attach.

### 1.2 Host Event Stream

References: [Host Event Stream](./DESIGN.md#host-event-stream),
[Live Session List](./DESIGN.md#live-session-list).

- [x] 1.2a Add `HostHandle::watch_host_events()` returning a typed
  `HostEventStream`.
- [ ] 1.2b Surface tmux control-mode notifications currently parsed as
  `ControlModeMessage::Notification`.
- [x] 1.2c Map session add/close/rename, client attach, client detach, and
  disconnect conditions into stable `HostEvent` variants.
- [x] 1.2d Reconcile by `SessionInfo.id`, not display name.
- [x] 1.2e Add tests for `SessionAdded`, `SessionClosed`, `SessionRenamed`, and
  client attach/detach diff behavior.

@gpt55-dgx 2026-04-26 -- Implementation note: the v1 `HostEventStream`
ships as a one-second `list_sessions()` snapshot reconciler. It preserves the
typed event API and stable-id behavior needed by the selector. True tmux
control-mode host notifications remain tracked by 1.2b as the next library
hardening item.

### 1.3 ScrollbackQuery::LinesRange

References: [ScrollbackQuery::LinesRange](./DESIGN.md#scrollbackquerylinesrange),
[R Pane Detail Source](./DESIGN.md#r-pane-detail-source).

- [x] 1.3a Add `ScrollbackQuery::LinesRange { older_than_lines, count }`.
- [x] 1.3b Implement chunked capture for the new query without rebuilding the
  whole scrollback buffer.
- [x] 1.3c Add tests for middle-page and zero-count behavior.
- [x] 1.3d Update `libs/tmux/docs/API.md` with the new query shape after code
  lands.

### 1.4 Stable Session-Id Dispatch

References: [Kill Session](./DESIGN.md#kill-session),
[Live Session List](./DESIGN.md#live-session-list).

- [x] 1.4a Ensure lifecycle operations used by the selector can dispatch
  against stable `SessionInfo.id`, not a potentially stale display name.
- [x] 1.4b If the existing `Target` internals cannot target session ids safely,
  add a small library-owned helper such as `HostHandle::session_by_id()`.
- [x] 1.4c Add stable-id unit coverage that asserts session kill dispatches by
  `SessionInfo.id`.

## Phase 2: Binary Scaffold

References: [Target Model](./DESIGN.md#target-model),
[CLI.md](./CLI.md), [API.md](./API.md).

- [x] 2.1 Add `bins/mmux/Cargo.toml` as package
  `motlie-mmux`.
- [x] 2.2 Add `bins/mmux/main.rs` as the binary entrypoint requested by
  issue #226.
- [x] 2.3 Add the package to workspace `Cargo.toml`.
- [x] 2.4 Add dependencies: `motlie-tmux`, `tokio`, `anyhow`, `clap`,
  `ratatui`, `crossterm`, and `async-trait` only if the final trait shape
  requires it.
- [x] 2.5 Implement CLI parsing for positional `ssh-uri`, `--script`,
  `--portrait` / `-p`, and `--landscape` / `-l`; reject removed
  `--print-session` / `--dashboard`, reject the old `-s` flag, and mutually
  reject both layout force flags.
- [x] 2.6 Add startup validation for mutually exclusive layout force flags.
- [x] 2.7 Add smoke tests for startup-error cases.

## Phase 3: Selector State Model

References: [Layout](./DESIGN.md#layout), [Data Flow](./DESIGN.md#data-flow),
[Empty Session List](./DESIGN.md#empty-session-list).

- [x] 3.1 Define `AppState`, `Focus`, `LayoutMode`, `Selection`, `ModalState`,
  and bounded monitor-buffer state.
- [x] 3.2 Maintain sessions keyed by stable session id with display name as
  mutable presentation data.
- [x] 3.3 Implement highlight preservation across host-event reconciliation.
- [x] 3.4 Implement empty-list state: placeholder row, `n` remains active,
  attach and kill disabled.
- [x] 3.5 Implement monitor stop when changing selected sessions, leaving the
  selector, or receiving a close event for the monitored session id.
- [x] 3.6 Add unit tests for rename preservation and monitored-session close
  reset behavior.

## Phase 4: Layout and Rendering

References: [Layout](./DESIGN.md#layout),
[Portrait Mode](./DESIGN.md#portrait-mode), [SVG Mock](./DESIGN.md#svg-mock).

- [x] 4.1 Implement normal layout: `L`/`R`, `LT`/`LB`, one-row top status bar,
  and one-row bottom command/status bar.
- [x] 4.2 Implement dynamic MOTD height cap: fit content up to 30% of left
  pane height.
- [x] 4.3 Implement absent-MOTD motlie placeholder with narrow-terminal
  fallback.
- [x] 4.4 Implement portrait mode `--portrait`: `T`/`B` split at 30:70 and
  omit MOTD.
- [x] 4.5 Implement focused/unfocused border styles.
- [x] 4.6 Implement a blue top status bar with bold left-justified
  `<hostname> | <ip address>` and right-justified time; keep the Sessions pane
  title count-only as `Sessions [n]`; keep the blue bottom status bar to
  compact arrow-symbol key hints ordered with `(h)elp` first and app status
  with no `keys`, host, time, focus, or layout-mode labels.
- [ ] 4.7 Add layout unit tests for 64x32 portrait mode, PTY auto-detection
  threshold 4.0, landscape force flag, narrow placeholder fallback, status bar
  reservation, and resize bounds.
- [ ] 4.8 Add snapshot/style tests for focused borders and motlie placeholder
  styling.

## Phase 5: Input, Focus, and Modals

References: [Functional Requirements](./DESIGN.md#functional),
[Layout](./DESIGN.md#layout).

- [x] 5.1 Implement focus transitions: cyclic Right/Left pane navigation,
  outside-modal `Esc` returning to the session list, and `q` as an exit alias
  for `Ctrl-C`.
- [x] 5.2 Implement session-list movement and scrolling for `LB`/`T`.
- [x] 5.3 Implement R/B scrolling, page movement, Home/End, and monitor
  auto-tail resume on End.
- [x] 5.4 Implement resize keys: `Ctrl-Left`/`Ctrl-Right` for normal mode and
  `Ctrl-Up`/`Ctrl-Down` for portrait mode. Accept modified-arrow and word-arrow
  fallback sequences for terminals that remap Ctrl-arrow.
- [x] 5.5 Implement `New Session` modal with text input, Cancel/Ok, Enter, and
  Esc handling.
- [x] 5.6 Implement kill confirmation modal with id captured at modal-open.
- [x] 5.7 Implement Help modal opened by `h`, showing the built-in motlie
  logo, key functions, and build git SHA with a single Ok button; Enter or Esc
  closes it.
- [ ] 5.8 Add unit tests for every key transition, modal button selection,
  modal Esc behavior, and cyclic plain Left/Right focus behavior.

## Phase 6: Detail Sources

References: [R Pane Detail Source](./DESIGN.md#r-pane-detail-source),
[Monitor Mode](./DESIGN.md#monitor-mode).

- [x] 6.1 Implement `SessionDetailSource` or equivalent closed enum wrapper.
- [x] 6.2 Implement `SampleDetailSource` using `motlie-tmux` capture/sample
  APIs with ANSI-preserving `ScreenStable` capture for colored detail output.
- [x] 6.3 Implement backwards chunk fetch through `LinesRange`.
- [x] 6.4 Implement `MonitorDetailSource` as a rendered screen mirror using
  `capture_all_with_options(CaptureNormalizeMode::ScreenStable)` and
  `ansi-to-tui`/VTE parsing for TUI-safe display.
- [x] 6.5 Keep monitor refresh bounded to the current screen; do not retain a
  raw control-mode transcript in the selector binary.
- [x] 6.6 When the user requests older detail content in monitor mode, fetch
  tmux scrollback through `LinesRange` on the same target.
- [x] 6.7 Add mock-backed tests for monitor screen capture, ANSI/VTE parsing,
  modified-arrow resize fallbacks, Left/Right focus transitions,
  ANSI-preserving sample detail, and monitored-session close behavior.

## Phase 7: Session Lifecycle Operations

References: [Create Session](./DESIGN.md#create-session),
[Kill Session](./DESIGN.md#kill-session), [Live Session List](./DESIGN.md#live-session-list).

- [x] 7.1 Implement create-session Ok path with default tmux options.
- [x] 7.2 Implement create-session duplicate/error path as inline TUI status.
- [x] 7.3 Implement kill-session Ok path by stable session id.
- [x] 7.4 Use host-event reconciliation for background changes; modal
  create/kill also refresh immediately in v1 because the shipped host-event
  stream is polling-backed.
- [x] 7.5 Refresh immediately after create/kill so modal operations update the
  first implementation without waiting for the next host-event tick.
- [ ] 7.6 Add localhost integration tests for create, list, sample, monitor,
  kill, and empty-list transition.

## Phase 8: Attach And Script Modes

References: [Attach](./DESIGN.md#attach), [CLI.md](./CLI.md).

- [x] 8.1 Implement default Enter/`a` attach path.
- [x] 8.2 Restore alternate screen and terminal raw mode before attach.
- [x] 8.3 Stop monitor state and drop the active host-event subscription before
  attach; selector re-entry starts from a fresh `list_sessions()` snapshot and
  new subscription.
- [x] 8.4 Implement `--script`: stdout exactly `<name>\n` on selection,
  non-zero exit and empty stdout on cancel.
- [x] 8.5 Implement default attach/re-enter behavior: re-enter TUI when the
  attach child exits cleanly or when detach returns non-zero but the selected
  session still exists; refresh succeeds and the user explicitly picks again.
- [x] 8.6 On pre-spawn vanished-session race, re-enter the selector.
- [ ] 8.7 Add localhost integration test pinning canonical tmux behavior:
  externally killing the attached session exits the attach child with status 0.
- [ ] 8.8 Add no-loop tests for non-zero child exit and refresh failure.
- [x] 8.9 Add a unit guard for `SIGTTOU`-safe foreground-process-group restore
  after attach detach.

## Phase 9: SSH and ForceCommand

References: [SSH ForceCommand Integration](./DESIGN.md#ssh-forcecommand-integration),
[CLI.md](./CLI.md#forcecommand).

- [x] 9.1 Implement local ForceCommand entrypoint behavior.
- [x] 9.2 Reject `SSH_ORIGINAL_COMMAND` by default with a clear stderr message.
- [x] 9.3 Implement `MOTLIE_MMUX_BYPASS` external bypass behavior.
- [x] 9.4 Implement operator-provided SSH URI target mode for list, MOTD,
  sample, create, kill, monitor, and attach.
- [x] 9.5 Document recommended `sshd_config` snippets in `CLI.md` after
  implementation confirms exact paths.
- [ ] 9.6 Add env-gated SSH integration tests for remote list, sample, monitor,
  attach command construction, and bypass handling.

## Phase 10: Final Validation and Docs

References: [Testing Strategy](./DESIGN.md#testing-strategy),
[API.md](./API.md), [CLI.md](./CLI.md).

- [x] 10.1 Update `API.md` from design target to implemented API reality.
- [x] 10.2 Update `CLI.md` from design target to implemented CLI reality.
- [ ] 10.3 Add README or examples references if runnable examples are created.
- [ ] 10.4 Run `cargo fmt --all`.
- [x] 10.5 Run `cargo clippy -p motlie-tmux -- -D warnings`.
- [x] 10.6 Run `cargo clippy -p motlie-mmux -- -D warnings`.
- [x] 10.7 Run `cargo test -p motlie-tmux`.
- [x] 10.8 Run `cargo test -p motlie-mmux`.
- [ ] 10.9 Run env-gated SSH tests where credentials are available.

@gpt55-dgx 2026-04-26 -- Validation note: `cargo fmt --all --check` fails on
pre-existing formatting drift outside this selector scope. The implementation
keeps unrelated formatting churn out of this PR; `git diff --check`, targeted
builds/tests/clippy, and `cargo build --bins --examples` passed.

## Concrete Test Matrix

| Area | Harness | Required coverage |
|------|---------|-------------------|
| Library attach | Unit + localhost smoke | command construction, process group handoff, exit status mapping, terminal restore |
| Host events | Polling-backed typed stream | add, close, rename, disconnect, one-second snapshot reconciliation |
| Scrollback range | Unit tests | first/middle/exhausted ranges, chunk size, invalid range |
| Layout | Pure unit tests | normal split, portrait mode 64x32, PTY auto-detect threshold 4.0, landscape force flag, MOTD cap, placeholder fallback, resize bounds |
| Input model | Pure unit tests | cyclic focus transitions, scrolling, attach key, modal Enter/Esc, Help modal `h` key, key functions, and build SHA display |
| Detail source | Mock `motlie-tmux` facade | sample color preservation, monitor screen capture, ANSI/VTE parse, tail pause, older-history fetch |
| Local integration | Dedicated tmux socket | create/list/sample/monitor/kill/attach/re-entry |
| SSH integration | Env-gated SSH URI | remote MOTD/list/sample/monitor/attach/bypass |
| Terminal cleanup | PTY harness | raw mode restore, alternate-screen restore, panic-path cleanup |

## Done Criteria

The implementation is ready for review when:

- all accepted `motlie-tmux` gaps are implemented in `libs/tmux`
- the selector package builds as `motlie-mmux` and produces the `mmux` binary
- default attach/re-enter, `--script`, `--portrait` / `-p`,
  `--landscape` / `-l`, local, SSH, and ForceCommand flows have targeted tests
- `DESIGN.md`, `PLAN.md`, `API.md`, and `CLI.md` are consistent with code
- `cargo fmt`, `cargo clippy -- -D warnings`, and relevant tests pass
