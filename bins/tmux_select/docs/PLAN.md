# tmux_select Implementation Plan

## Status

Draft implementation plan for the `tmux_select` selector described in
[DESIGN.md](./DESIGN.md). This PR still contains docs only; no Rust
implementation is included.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-26 | @gpt55-dgx | Initial PLAN for issue #226 and PR #227 re-review: orders accepted `motlie-tmux` gaps before binary work, defines the selector phases, and makes the test harness concrete. |

## Scope

This is greenfield product work. There is no migration or backward
compatibility requirement for an older selector binary.

Implementation must proceed in this order:

1. Add the `motlie-tmux` capabilities the selector depends on.
2. Add the binary scaffold under `bins/tmux_select`.
3. Build the TUI in slices that are independently testable.
4. Add attach and deployment flows after terminal cleanup is reliable.

The binary must not duplicate tmux command construction or SSH attach logic
that belongs in `motlie-tmux`.

## Dev Harness

Use a dedicated tmux socket for local tests so developer sessions are not
modified:

```bash
export MOTLIE_TMUX_SELECT_SOCKET=motlie-select-test
tmux -L "$MOTLIE_TMUX_SELECT_SOCKET" start-server
```

Recommended verification commands as phases land:

```bash
cargo fmt --all
cargo check -p motlie-tmux
cargo test -p motlie-tmux
cargo check -p motlie-tmux-select
cargo test -p motlie-tmux-select
cargo clippy -p motlie-tmux -- -D warnings
cargo clippy -p motlie-tmux-select -- -D warnings
```

The `motlie-tmux-select` package does not exist yet. Phase 2 creates it.

SSH integration tests should be env-gated so normal local test runs do not
require an SSH daemon:

```bash
export MOTLIE_TMUX_SELECT_SSH_URI='ssh://user@host?identity-file=/path/to/key'
cargo test -p motlie-tmux-select --test ssh_integration -- --ignored
```

## Phase 1: motlie-tmux Library Gaps

### 1.1 Current PTY Attach

References: [Current PTY Attach](./DESIGN.md#current-pty-attach),
[Attach](./DESIGN.md#attach), [Non-Functional](./DESIGN.md#non-functional).

- [ ] 1.1a Add `Target::attach_current_pty(&self) -> Result<AttachExit>` in
  `libs/tmux` with spawn-and-wait semantics.
- [ ] 1.1b Local attach spawns `tmux attach-session -t <target>` with inherited
  stdio, no pipe wrapping.
- [ ] 1.1c SSH attach spawns `ssh -t ... tmux attach-session -t <target>` with
  inherited stdio, using `SshConfig` connection information owned by the
  library.
- [ ] 1.1d Put the attach child in its own process group and transfer the
  controlling terminal with `tcsetpgrp`; restore the selector process group
  after `wait()`.
- [ ] 1.1e Translate signal termination to `128 + signal` in `AttachExit`.
- [ ] 1.1f Add unit tests for command construction and process-status mapping.
- [ ] 1.1g Add localhost smoke coverage that verifies terminal state is restored
  before and after attach.

### 1.2 Host Event Stream

References: [Host Event Stream](./DESIGN.md#host-event-stream),
[Live Session List](./DESIGN.md#live-session-list).

- [ ] 1.2a Add `HostHandle::watch_host_events()` returning a typed
  `HostEventStream`.
- [ ] 1.2b Surface tmux control-mode notifications currently parsed as
  `ControlModeMessage::Notification`.
- [ ] 1.2c Map `%sessions-changed`, session add/close/rename, client attach,
  client detach, and disconnect conditions into stable `HostEvent` variants.
- [ ] 1.2d Reconcile by `SessionInfo.id`, not display name.
- [ ] 1.2e Add tests for `SessionAdded`, `SessionClosed`, `SessionRenamed`, and
  disconnect-to-polling fallback.

### 1.3 ScrollbackQuery::LinesRange

References: [ScrollbackQuery::LinesRange](./DESIGN.md#scrollbackquerylinesrange),
[R Pane Detail Source](./DESIGN.md#r-pane-detail-source).

- [ ] 1.3a Add `ScrollbackQuery::LinesRange { older_than_lines, count }`.
- [ ] 1.3b Implement chunked capture for the new query without rebuilding the
  whole scrollback buffer.
- [ ] 1.3c Add tests for first-page, middle-page, exhausted-history, and invalid
  range behavior.
- [ ] 1.3d Update `libs/tmux/docs/API.md` with the new query shape after code
  lands.

### 1.4 Stable Session-Id Dispatch

References: [Kill Session](./DESIGN.md#kill-session),
[Live Session List](./DESIGN.md#live-session-list).

- [ ] 1.4a Ensure lifecycle operations used by the selector can dispatch
  against stable `SessionInfo.id`, not a potentially stale display name.
- [ ] 1.4b If the existing `Target` internals cannot target session ids safely,
  add a small library-owned helper such as `HostHandle::session_by_id()`.
- [ ] 1.4c Add a rename-race test: open kill confirmation, rename the session
  externally, confirm kill, and assert the original id is killed.

## Phase 2: Binary Scaffold

References: [Target Model](./DESIGN.md#target-model),
[CLI.md](./CLI.md), [API.md](./API.md).

- [ ] 2.1 Add `bins/tmux_select/Cargo.toml` as package
  `motlie-tmux-select`.
- [ ] 2.2 Add `bins/tmux_select/main.rs` as the binary entrypoint requested by
  issue #226.
- [ ] 2.3 Add the package to workspace `Cargo.toml`.
- [ ] 2.4 Add dependencies: `motlie-tmux`, `tokio`, `anyhow`, `clap`,
  `ratatui`, `crossterm`, and `async-trait` only if the final trait shape
  requires it.
- [ ] 2.5 Implement CLI parsing for positional `ssh-uri`, `--print-session`,
  `--dashboard`, and `-s`.
- [ ] 2.6 Add startup validation for mutually exclusive `--print-session` and
  `--dashboard`.
- [ ] 2.7 Add smoke tests for CLI parse success and startup-error cases.

## Phase 3: Selector State Model

References: [Layout](./DESIGN.md#layout), [Data Flow](./DESIGN.md#data-flow),
[Empty Session List](./DESIGN.md#empty-session-list).

- [ ] 3.1 Define `AppState`, `Focus`, `LayoutMode`, `Selection`, `ModalState`,
  and bounded monitor-buffer state.
- [ ] 3.2 Maintain sessions keyed by stable session id with display name as
  mutable presentation data.
- [ ] 3.3 Implement highlight preservation across host-event reconciliation.
- [ ] 3.4 Implement empty-list state: placeholder row, `n` remains active,
  attach and kill disabled.
- [ ] 3.5 Implement monitor stop and R-pane clear when `SessionClosed` matches
  the monitored session id.
- [ ] 3.6 Add unit tests for highlight movement, rename preservation,
  close-current-highlight fallback, and monitored-session close behavior.

## Phase 4: Layout and Rendering

References: [Layout](./DESIGN.md#layout),
[Short Mode](./DESIGN.md#short-mode--s), [SVG Mock](./DESIGN.md#svg-mock).

- [ ] 4.1 Implement normal layout: `L`/`R`, `LT`/`LB`, one-row status bar.
- [ ] 4.2 Implement dynamic MOTD height cap: fit content up to 30% of left
  pane height.
- [ ] 4.3 Implement absent-MOTD motlie placeholder with narrow-terminal
  fallback.
- [ ] 4.4 Implement short mode `-s`: `T`/`B` split at 40:60 and omit MOTD.
- [ ] 4.5 Implement focused/unfocused border styles.
- [ ] 4.6 Implement status bar with host, time, focus, and ASCII-first key
  hints.
- [ ] 4.7 Add layout unit tests for 32x65 short mode, narrow placeholder
  fallback, status bar reservation, and resize bounds.
- [ ] 4.8 Add snapshot/style tests for focused borders and motlie placeholder
  styling.

## Phase 5: Input, Focus, and Modals

References: [Functional Requirements](./DESIGN.md#functional),
[Layout](./DESIGN.md#layout).

- [ ] 5.1 Implement focus transitions: `v`, `l`, and outside-modal `Esc`.
- [ ] 5.2 Implement session-list movement and scrolling for `LB`/`T`.
- [ ] 5.3 Implement R/B scrolling, page movement, Home/End, and monitor
  auto-tail resume on End.
- [ ] 5.4 Implement resize keys: `Ctrl-Left`/`Ctrl-Right` for normal mode and
  `Ctrl-Up`/`Ctrl-Down` for short mode.
- [ ] 5.5 Implement `New Session` modal with text input, Cancel/Ok, Enter, and
  Esc handling.
- [ ] 5.6 Implement kill confirmation modal with id captured at modal-open.
- [ ] 5.7 Add unit tests for every key transition, modal button selection,
  modal Esc behavior, and reserved plain Left/Right no-op behavior.

## Phase 6: Detail Sources

References: [R Pane Detail Source](./DESIGN.md#r-pane-detail-source),
[Monitor Mode](./DESIGN.md#monitor-mode).

- [ ] 6.1 Implement `SessionDetailSource` or equivalent closed enum wrapper.
- [ ] 6.2 Implement `SampleDetailSource` using `motlie-tmux` capture/sample
  APIs.
- [ ] 6.3 Implement backwards chunk fetch through `LinesRange`.
- [ ] 6.4 Implement `MonitorDetailSource` using the monitor/history pipeline.
- [ ] 6.5 Enforce the 10,000-line rolling monitor history bound.
- [ ] 6.6 When monitor history scrolls beyond the rolling-buffer start, fetch
  older tmux pre-monitor scrollback through `LinesRange` on the same target.
- [ ] 6.7 Add mock-backed tests for sample replace, monitor append, tail pause,
  End resume, and older-history fetch anchoring.

## Phase 7: Session Lifecycle Operations

References: [Create Session](./DESIGN.md#create-session),
[Kill Session](./DESIGN.md#kill-session), [Live Session List](./DESIGN.md#live-session-list).

- [ ] 7.1 Implement create-session Ok path with default tmux options.
- [ ] 7.2 Implement create-session duplicate/error path as inline TUI status.
- [ ] 7.3 Implement kill-session Ok path by stable session id.
- [ ] 7.4 Do not eagerly refresh on event-stream mode; rely on host-event
  reconciliation.
- [ ] 7.5 Refresh immediately after create/kill only when polling fallback is
  active.
- [ ] 7.6 Add localhost integration tests for create, list, sample, monitor,
  kill, and empty-list transition.

## Phase 8: Attach, Print, and Dashboard Modes

References: [Attach](./DESIGN.md#attach), [CLI.md](./CLI.md).

- [ ] 8.1 Implement default Enter/`g` attach path.
- [ ] 8.2 Restore alternate screen and terminal raw mode before attach.
- [ ] 8.3 Stop monitor state and host-event subscriptions before attach.
- [ ] 8.4 Implement `--print-session`: stdout exactly `<name>\n` on selection,
  non-zero exit and empty stdout on cancel.
- [ ] 8.5 Implement `--dashboard`: re-enter TUI only when child status is
  success, refresh succeeds, and the user explicitly picks again.
- [ ] 8.6 On pre-spawn vanished-session race, re-enter under `--dashboard` or
  exit non-zero in default mode.
- [ ] 8.7 Add localhost integration test pinning canonical tmux behavior:
  externally killing the attached session exits the attach child with status 0.
- [ ] 8.8 Add no-loop tests for non-zero child exit and refresh failure.

## Phase 9: SSH and ForceCommand

References: [SSH ForceCommand Integration](./DESIGN.md#ssh-forcecommand-integration),
[CLI.md](./CLI.md#forcecommand).

- [ ] 9.1 Implement local ForceCommand entrypoint behavior.
- [ ] 9.2 Reject `SSH_ORIGINAL_COMMAND` by default with a clear stderr message.
- [ ] 9.3 Implement `MOTLIE_TMUX_SELECT_BYPASS` external bypass behavior.
- [ ] 9.4 Implement operator-provided SSH URI target mode for list, MOTD,
  sample, create, kill, monitor, and attach.
- [ ] 9.5 Document recommended `sshd_config` snippets in `CLI.md` after
  implementation confirms exact paths.
- [ ] 9.6 Add env-gated SSH integration tests for remote list, sample, monitor,
  attach command construction, and bypass handling.

## Phase 10: Final Validation and Docs

References: [Testing Strategy](./DESIGN.md#testing-strategy),
[API.md](./API.md), [CLI.md](./CLI.md).

- [ ] 10.1 Update `API.md` from design target to implemented API reality.
- [ ] 10.2 Update `CLI.md` from design target to implemented CLI reality.
- [ ] 10.3 Add README or examples references if runnable examples are created.
- [ ] 10.4 Run `cargo fmt --all`.
- [ ] 10.5 Run `cargo clippy -p motlie-tmux -- -D warnings`.
- [ ] 10.6 Run `cargo clippy -p motlie-tmux-select -- -D warnings`.
- [ ] 10.7 Run `cargo test -p motlie-tmux`.
- [ ] 10.8 Run `cargo test -p motlie-tmux-select`.
- [ ] 10.9 Run env-gated SSH tests where credentials are available.

## Concrete Test Matrix

| Area | Harness | Required coverage |
|------|---------|-------------------|
| Library attach | Unit + localhost smoke | command construction, process group handoff, exit status mapping, terminal restore |
| Host events | Mock control-mode stream | add, close, rename, disconnect, polling fallback |
| Scrollback range | Unit tests | first/middle/exhausted ranges, chunk size, invalid range |
| Layout | Pure unit tests | normal split, short mode 32x65, MOTD cap, placeholder fallback, resize bounds |
| Input model | Pure unit tests | focus transitions, scrolling, reserved arrows, modal Enter/Esc |
| Detail source | Mock `motlie-tmux` facade | sample replace, monitor append, tail pause, older-history fetch |
| Local integration | Dedicated tmux socket | create/list/sample/monitor/kill/attach/dashboard |
| SSH integration | Env-gated SSH URI | remote MOTD/list/sample/monitor/attach/bypass |
| Terminal cleanup | PTY harness | raw mode restore, alternate-screen restore, panic-path cleanup |

## Done Criteria

The implementation is ready for review when:

- all accepted `motlie-tmux` gaps are implemented in `libs/tmux`
- the selector binary builds as `motlie-tmux-select`
- default, `--print-session`, `--dashboard`, `-s`, local, SSH, and ForceCommand
  flows have targeted tests
- `DESIGN.md`, `PLAN.md`, `API.md`, and `CLI.md` are consistent with code
- `cargo fmt`, `cargo clippy -- -D warnings`, and relevant tests pass
