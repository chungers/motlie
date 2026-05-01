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
| 2026-05-01 | @codex | Updated Phase 12 after tmux unset research: add a `motlie-tmux` tag delete API (`SessionTags::unset`, `Target::unset_tag`) and expand the `i` modal with row focus, `x` delete, and `u` update flows. |
| 2026-05-01 | @codex | Started Phase 12 for issue #241: branch `feature/mmux-241-session-modals`. Plan covers list-focus-only rename on `r`, selected-session tag edit on `t`, tag info/add on `i`, modal-specific focus state, stable `(host_id, session_id)` dispatch, motlie-tmux tag API usage, and focused tests. |
| 2026-04-29 | @opus47-macos-tmux | Added Phase 11 for multi-host support (issue #235): branch `feature/mmux-multihost`. Phased work covers CLI multi-arg parsing, `HostFleet`/`HostEntry`/`SessionRow` data model, fan-out polling with per-host failure isolation, row hostname column gated on `fleet.is_multi()`, top status bar mode switch, MOTD pane suppression, attach/create/kill routing by row, and tests. No new library APIs required. |
| 2026-04-28 | @gpt55-dgx | Opened and linked issue #232 for Phase 9.6 env-gated SSH/ForceCommand integration tests; clarified exact bypass value contract. |
| 2026-04-28 | @gpt55-dgx | Consolidated mmux refresh to one `list_sessions_now()` poller for activity sorting, recency text, structural state, and monitored-session closure. |
| 2026-04-28 | @gpt55-dgx | Tracked one-second quiet visible-row refreshes so activity sorting updates without structural host events. |
| 2026-04-28 | @gpt55-dgx | Tracked activity-descending session ordering with stable-id selection preservation. |
| 2026-04-28 | @gpt55-dgx | Tracked recency formatting polish: remove labels, add day bucket, and keep a right margin. |
| 2026-04-28 | @gpt55-dgx | Added regression tracking for tmux versions that expand `#{epoch}` empty during session recency listing. |
| 2026-04-28 | @gpt55-dgx | Marked mmux session-list recency rendering complete with aligned `active`/`age` columns and no window-alert flag display. |
| 2026-04-28 | @gpt55-dgx | Marked issue #229 library support complete for `SessionInfo.activity`, `attached_count`, and `HostHandle::list_sessions_now()` skew-free recency math. |
| 2026-04-28 | @gpt55-dgx | Tracked bottom status command hints as plain words with underlined shortcut letters instead of `(x)` mnemonics. |
| 2026-04-28 | @gpt55-dgx | Updated placeholder tests so landscape requires the full motlie glyph whenever the embedded logo dimensions fit the pane. |
| 2026-04-28 | @gpt55-dgx | Added landscape MOTD pane regression coverage for placeholder and host-provided MOTD content. |
| 2026-04-28 | @gpt55-dgx | Added PR #228 round-3 regression coverage for bounded host text reads, injectable MOTD fallback loading, full/compact placeholder rendering, and portrait MOTD omission. |
| 2026-04-28 | @gpt55-dgx | Tracked PR #228 review cleanup: typed `SessionId`, bounded `read_text_file` MOTD loading, documented polling host events, decomposed selector state/status, split selector concerns into focused modules, and kept internal ids hidden from the list view. |
| 2026-04-27 | @gpt55-dgx | Tracked modal layout polish: padded content, button separators, bordered New Session input, and Help metadata placement. |
| 2026-04-27 | @gpt55-dgx | Tracked bottom status command ordering and `l` runtime layout toggling. |
| 2026-04-27 | @gpt55-dgx | Updated keymap tracking so `p` cycles panes and plain Left/Right no longer do. |
| 2026-04-27 | @gpt55-dgx | Tracked in-memory selector UI state retention across default attach/detach re-entry. |
| 2026-04-27 | @gpt55-dgx | Updated resize-bound tracking for landscape 25/75 and portrait 15/85. |
| 2026-04-27 | @gpt55-dgx | Updated Help modal tracking for build date and last-8-character git SHA display. |
| 2026-04-27 | @gpt55-dgx | Updated bottom status tracking for `↑/↓ sel` and `←/→ pane` direction hints. |
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
- [x] 1.2b Decide control-mode notification scope for v1: keep parser support
  dormant and documented for a future event-driven watcher; ship polling-backed
  snapshot reconciliation now.
- [x] 1.2c Map session add/close/rename, client attach, client detach, and
  disconnect conditions into stable `HostEvent` variants.
- [x] 1.2d Reconcile by `SessionInfo.id`, not display name.
- [x] 1.2e Add tests for `SessionAdded`, `SessionClosed`, `SessionRenamed`, and
  client attach/detach diff behavior.

@gpt55-dgx 2026-04-26 -- Implementation note: the v1 `HostEventStream`
ships as a one-second `list_sessions()` snapshot reconciler. It preserves the
typed event API and stable-id behavior needed by the selector. True tmux
control-mode host notifications remain reserved for a future event-driven
implementation under the same public event stream contract.

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

### 1.5 Review Cleanup: Host Metadata and Stable IDs

References: [Remote MOTD](./DESIGN.md#remote-motd),
[Live Session List](./DESIGN.md#live-session-list).

- [x] 1.5a Replace the broad host shell helper with bounded
  `HostHandle::read_text_file(path, max_bytes)` for `/etc/motd`.
- [x] 1.5b Introduce non-empty `SessionId` for `SessionInfo.id` and reject
  malformed empty ids at parse time.
- [x] 1.5c Remove session lifecycle fallback to display name for kill, rename,
  attach, and host-event keying.
- [x] 1.5d Shell-escape unsafe resolved tmux binary paths in generated tmux
  command prefixes without changing safe-path command text.
- [x] 1.5e Add focused tests for `HostHandle::read_text_file`: missing,
  empty, normal, oversized, unreadable local files, plus mock transport reads.

### 1.6 Session Activity and Server Clock Listing

References: [Live Session List](./DESIGN.md#live-session-list), issue #229.

- [x] 1.6a Add `SessionInfo.activity` from tmux `#{session_activity}`.
- [x] 1.6b Replace lossy attached parsing with non-lossy
  `SessionInfo.attached_count` and `SessionInfo::is_attached()`.
- [x] 1.6c Add `SessionListing { now, sessions }` and
  `HostHandle::list_sessions_now()` so remote recency math can use the tmux
  server clock instead of the local selector clock.
- [x] 1.6d Add parser and mock-backed tests for populated listings, empty
  listings, no-server fallback, malformed/missing epoch lines, and parsed
  `activity` / `attached_count`.
- [x] 1.6e Document recency math in `libs/tmux/docs/API.md`.
- [x] 1.6f Render mmux session rows with a right-aligned
  `active:<elapsed> / age:<elapsed>` column from `list_sessions_now()` and keep
  tmux window-alert flags out of the v1 list row.
- [x] 1.6g Accept empty `#{epoch}` expansion from tmux 3.4-era servers and
  fall back to a local clock clamped to listed session timestamps.
- [x] 1.6h Polish recency row formatting: remove labels, add `d` duration
  bucket with at most one decimal digit, and reserve a right-side margin.
- [x] 1.6i Sort session rows by `SessionInfo.activity` descending with stable
  name/id tie-breakers and preserve highlight by stable session id after
  refresh.
- [x] 1.6j Refresh visible session rows with `list_sessions_now()` once per
  second so activity-only changes update recency text and row order without
  waiting for add/close/rename/attach/detach events.
- [x] 1.6k Consolidate mmux polling so the same one-second
  `list_sessions_now()` snapshot drives recency text, activity ordering,
  structural state, and monitored-session closure handling.

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
- [x] 3.7 Decompose selector state into host, layout, MOTD, session-list,
  detail, and typed status concerns; split CLI parsing, terminal lifecycle,
  ForceCommand, target-host identity, detail sources, rendering, and
  input/event handling out of `main.rs`.

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
  omit MOTD; clamp portrait resize bounds at 15/85.
- [x] 4.5 Implement focused/unfocused border styles.
- [x] 4.6 Implement a blue top status bar with bold left-justified
  `<hostname> | <ip address>` and right-justified time; keep the Sessions pane
  title count-only as `Sessions [n]`; keep the blue bottom status bar to
  compact direction hints (`↑/↓ sel`, underlined `p` in `pane`), command hints
  ordered as `help`, `pane`, `monitor`, `enter/attach`, `new`, `kill`, `quit`,
  `layout`, then resize, with shortcut letters underlined and app status with
  no `keys`, host, time, focus, or layout-mode labels.
- [x] 4.7 Add layout unit tests for 64x32 portrait mode, PTY auto-detection
  threshold 4.0, landscape force flag, narrow placeholder fallback, status bar
  reservation, and resize bounds.
- [ ] 4.8 Add snapshot/style tests for focused borders and motlie placeholder
  styling.
- [x] 4.9 Add MOTD fallback regression tests for missing, empty,
  whitespace-only, oversized, and readable files through an injectable
  `load_motd_from` path; add full-logo wide rendering and portrait omission
  tests; add full-frame landscape render tests for placeholder and host MOTD
  content; assert the full embedded glyph renders whenever the landscape MOTD
  pane can fit it.

## Phase 5: Input, Focus, and Modals

References: [Functional Requirements](./DESIGN.md#functional),
[Layout](./DESIGN.md#layout).

- [x] 5.1 Implement focus transitions: `p` pane navigation, `l` runtime layout
  toggle, outside-modal `Esc` returning to the session list, and `q` as an
  exit alias for `Ctrl-C`.
- [x] 5.2 Implement session-list movement and scrolling for `LB`/`T`.
- [x] 5.3 Implement R/B scrolling, page movement, Home/End, and monitor
  auto-tail resume on End.
- [x] 5.4 Implement resize keys: `Ctrl-Left`/`Ctrl-Right` for normal mode and
  `Ctrl-Up`/`Ctrl-Down` for portrait mode. Accept modified-arrow and word-arrow
  fallback sequences for terminals that remap Ctrl-arrow.
- [x] 5.5 Implement `New Session` modal with padded content, bordered text
  input, separated Cancel/Ok button bar, Enter, and Esc handling.
- [x] 5.6 Implement kill confirmation modal with padded content, separated
  button bar, and id captured at modal-open.
- [x] 5.7 Implement Help modal opened by `h`, showing the built-in motlie
  logo, build date, last 8 characters of the build git SHA, key functions,
  and a separated single Ok button; Enter or Esc closes it.
- [ ] 5.8 Add unit tests for every key transition, modal button selection,
  modal Esc behavior, `p` pane focus behavior, and `l` layout toggle behavior.

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
  modified-arrow resize fallbacks, `p` focus transitions, `l` layout toggling,
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
- [x] 8.7 Retain selected session/list index, layout mode, pane split, and
  focus in memory across default attach/detach re-entry within one parent
  `mmux` process.
- [ ] 8.8 Add localhost integration test pinning canonical tmux behavior:
  externally killing the attached session exits the attach child with status 0.
- [ ] 8.9 Add no-loop tests for non-zero child exit and refresh failure.
- [x] 8.10 Add a unit guard for `SIGTTOU`-safe foreground-process-group restore
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
  attach command construction, and bypass handling. Tracked by
  [issue #232](https://github.com/chungers/motlie/issues/232). Include exact
  `MOTLIE_MMUX_BYPASS=1` positive coverage and negative coverage for other
  non-empty values.

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

## Phase 11: Multi-host Support (issue #235)

References: [DESIGN.md → Multi-host Mode](./DESIGN.md#multi-host-mode-issue-235),
[CLI.md → Multi-host Mode](./CLI.md#multi-host-mode-issue-235), issue #235.

Branch: `feature/mmux-multihost`. Off `feature/session-selector`.

**Library scope:** none expected for v1. Existing `HostHandle::list_sessions_now()`,
`session_by_id()`, and `Target::attach_current_pty()` cover everything per-host.
If a `Fleet`-based convenience method emerges as cleaner during implementation,
it lands as a follow-up.

### 11.1 CLI parsing

- [ ] 11.1a Change `Cli.ssh_uri: Option<String>` to `Cli.ssh_uris: Vec<String>` in
  `bins/mmux/cli.rs`. Use `clap::Args` with `num_args = 0..` for the positional.
- [ ] 11.1b Backward-compat: zero or one URI keeps existing single-host behavior.
- [ ] 11.1c Reject malformed URIs at parse time using existing `SshConfig::parse`;
  surface a single error listing all failed URIs.
- [ ] 11.1d Tests: zero, one, two, many URIs; mixed valid + invalid.

### 11.2 Connect fleet

- [ ] 11.2a Rename `connect_host(cli) -> Result<(HostHandle, HostIdentity)>`
  to `connect_fleet(cli) -> Result<HostFleet>`. Internally iterates and calls
  the existing per-host connect logic.
- [ ] 11.2b Connect concurrently via `tokio::try_join_all` to keep startup
  latency O(slowest), not O(sum).
- [ ] 11.2c Per-host connect failure surfaces in stderr but does not abort
  startup if at least one host connects (operator can still use the rest).
- [ ] 11.2d Tests: all-succeed, partial-fail, all-fail (the last exits non-zero).

### 11.3 Internal data model

- [ ] 11.3a Add `HostId(String)`, `HostEntry`, `HostFleet`, `SessionRow` types
  in `bins/mmux/model.rs`.
- [ ] 11.3b Replace `HostContext` with `HostFleet` on `AppState`.
- [ ] 11.3c Change `SessionListState.sessions: Vec<SessionInfo>` to
  `SessionListState.rows: Vec<SessionRow>`.
- [ ] 11.3d Make `MotdState` an `Option` field on `AppState`: `motd: Option<MotdState>`.
  `None` in multi-host mode.
- [ ] 11.3e Selection identity is `(HostId, SessionId)` not just `SessionId`.
  Update `RetainedUiState` to preserve both across re-entry.
- [ ] 11.3f Tests: model construction; selection preservation across reorder;
  selection drop when row's host disappears; recompose after host comes back.

### 11.4 Fan-out refresh

- [ ] 11.4a Implement `refresh_listings(fleet) -> Vec<SessionRow>` in
  `controller.rs`. Use `tokio::join_all` (not `try_join_all`) so a single host
  failure doesn't drop the others.
- [ ] 11.4b Per-host result merge: `host_id`, `host_label`, `server_now`
  carried into each `SessionRow`. Sort by `session.activity` descending across
  the merged list.
- [ ] 11.4c Per-host failure surfaces to a `StatusBanner::HostUnreachable { host_id, reason }`
  variant; banner cycles through up to 3 unreachable hosts in the status line
  if more fail.
- [ ] 11.4d Confirm 1 Hz cadence (inherited from single-host) is sufficient
  for `n` hosts; profile if needed.
- [ ] 11.4e Tests: 2-host merge sort; 3-host with one failing; activity-tie
  ordering stability; row hostname populated correctly.

### 11.5 Render

- [ ] 11.5a `draw_top_status` switches on `fleet.is_multi()`:
  - Single: existing `<hostname> | <ip>     <time>` form.
  - Multi: `mmux - multi-host mode (<n>)     <time>` form.
- [ ] 11.5b `draw_sessions` row format: hostname column inserted between
  attached marker and session name when `fleet.is_multi() == true`. Column
  width = `fleet.host_label_width()` capped at a reasonable max
  (e.g. 24 chars; longer labels truncated with `…`).
- [ ] 11.5c MOTD pane: hide entirely when `app.motd.is_none()`. Layout reflows
  the left column (landscape) or top region (portrait) to give the full area
  to sessions.
- [ ] 11.5d Status banner: render `HostUnreachable` indicator(s) without
  blocking session listing.
- [ ] 11.5e Snapshot tests for: multi-host top status; multi-host row format
  (hostname column present, padded, truncated); MOTD-pane absence in
  multi-host; layout reflow correctness.

### 11.6 Input + dispatch routing

- [ ] 11.6a Attach (`Enter` / `a`): use highlighted `SessionRow` to look up
  `fleet.entry(row.host_id).handle` and dispatch
  `handle.session_by_id(row.session.id).attach_current_pty()`.
- [ ] 11.6b Monitor (`m`): same routing — detail-source `activate` takes the
  row's host handle.
- [ ] 11.6c New session (`n`): default v1 dispatches against the highlighted
  row's host (no host-picker modal). Document so PLAN/DESIGN/CLI agree.
- [ ] 11.6d Kill session (`k`): dispatches against the highlighted row's host.
- [ ] 11.6e Tests: each command key dispatches against the right host; verify
  via mock-host call recording.

### 11.7 Resilience and recovery

- [ ] 11.7a Per-host failure does not block other hosts (covered by 11.4a).
- [ ] 11.7b Reconnect: if a host's `list_sessions_now()` succeeds again after
  prior failure, its rows reappear automatically; status banner clears its
  `HostUnreachable` entry on next refresh.
- [ ] 11.7c Selection migration: if the highlighted row's host fails, drop
  selection to the next valid row in the merged list.
- [ ] 11.7d Tests: down/up cycle; selection migration on host failure;
  selection restoration when row re-appears.

### 11.8 Documentation final-pass

- [ ] 11.8a Update `bins/mmux/docs/DESIGN.md` (Multi-host Mode section
  already drafted; tighten after implementation lands).
- [ ] 11.8b Update `bins/mmux/docs/CLI.md` (Multi-host Mode section already
  drafted).
- [ ] 11.8c Update `bins/mmux/docs/API.md` with new internal types
  (`HostFleet`, `SessionRow`, `HostId`, `HostEntry`).
- [ ] 11.8d Update `bins/mmux/docs/README.md` to remove "(planned)" qualifier.
- [ ] 11.8e Update `bins/mmux/docs/mmux-mock.svg` with multi-host panel
  variants (single-host top status + row format vs. multi-host equivalents).

### 11.9 Validation

- [ ] 11.9a `cargo fmt --all`
- [ ] 11.9b `cargo test -p motlie-mmux` (existing 50 + new ~15 tests)
- [ ] 11.9c `cargo clippy -p motlie-mmux -- -D warnings`
- [ ] 11.9d Localhost smoke: `mmux ssh://localhost ssh://localhost` (two
  connections to the same host) — verify multi-host UX activates and rows
  list both connections' sessions.
- [ ] 11.9e Two-host smoke: `mmux ssh://a ssh://b` against two real hosts —
  verify activity-sort across hosts, attach to row routes correctly, host
  failure resilience.

## Phase 12: Session Rename and Tag Modals (issue #241)

References: [DESIGN.md → Session Rename and Tags](./DESIGN.md#session-rename-and-tags-issue-241),
issue #241.

This phase is primarily binary-side UI work, plus one small `motlie-tmux`
contract addition for deleting tags. It must use `motlie-tmux` APIs:
`HostHandle::session_by_id`, `Target::rename`, `Target::tags("mmux")`, and the
new unset methods below. Do not add direct tmux shell commands to `mmux`.

### 12.0 motlie-tmux tag delete API

- [ ] 12.0a Add `SessionTags::unset(key) -> Result<()>`.
- [ ] 12.0b Add one-off `Target::unset_tag(prefix, key) -> Result<()>`.
- [ ] 12.0c Add control-layer helper using
  `set-option -u -t <stable-session-id> @<prefix>/<key>` with no value
  argument.
- [ ] 12.0d Keep existing tag contracts: session targets only, validated
  prefix/key, stable session-id dispatch, no shell pipelines.
- [ ] 12.0e Add unit tests for command construction, validation-before-exec,
  missing/non-session target behavior, and set/read/list/unset roundtrip shape.
- [ ] 12.0f Update `libs/tmux/docs/API.md`, `DESIGN.md`, and `PLAN.md` for
  tag deletion.

### 12.1 Modal model and rendering

- [ ] 12.1a Extend `ModalState` with `RenameSession`, `EditSessionTag`, and
  `SessionTagsInfo` variants that capture `(host_id, session_id)` at open.
- [ ] 12.1b Add modal-specific focus enums for multi-field dialogs rather than
  one global catch-all modal focus enum.
- [ ] 12.1c `SessionTagsInfo` focus must support existing tag rows
  (`TagRow(index)`) plus bottom controls (`Key`, `Value`, `Add`, `Cancel`).
- [ ] 12.1d Add render helpers for labeled bordered text fields that preserve
  the existing modal padding, separator, and button styling.
- [ ] 12.1e Add Tab / Shift-Tab focus movement for multi-field modals while
  preserving Left / Right button selection where buttons are focused.
- [ ] 12.1f Update Help/status key references for `r`, `t`, and `i` if the
  existing status width budget allows; otherwise keep them in Help only.

### 12.2 Rename modal (`r`)

- [ ] 12.2a In main-view input, open rename only when `Focus::List` and a
  session is selected; non-list focus is a no-op.
- [ ] 12.2b Prepopulate `Session Name` with the selected session's current
  display name.
- [ ] 12.2c On `Ok`, trim using the New Session rule. Empty input reports a
  status banner; unchanged input closes without tmux I/O.
- [ ] 12.2d On changed input, resolve the captured stable session id through
  the captured host and call `Target::rename`.
- [ ] 12.2e Refresh sessions immediately after success and preserve selection
  by `(host_id, session_id)`.
- [ ] 12.2f Tests: list-focus gating, prepopulation, unchanged no-op, changed
  rename dispatch by stable id, disappeared-session status.

### 12.3 Tag edit modal (`t`)

- [ ] 12.3a Open for the highlighted session from any pane focus; no selected
  session reports the existing "no session selected" status.
- [ ] 12.3b Render `Tag` and `Value` text fields plus standard `Cancel` /
  `Ok` buttons.
- [ ] 12.3c On valid non-empty tag key entry, prepopulate `Value` from
  `target.tags("mmux").await?.read(key).await?` when present and the value
  field has not been manually edited.
- [ ] 12.3d On `Ok`, skip writes when `Value` is empty; otherwise call
  `target.tags("mmux").await?.set(key, value).await?`.
- [ ] 12.3e Keep value text exact, without trimming, while trimming only the
  tag key. Let `motlie-tmux` enforce key/value validation.
- [ ] 12.3f Tests: existing-value prefill, no prefill when missing, empty value
  no-op, invalid key status, successful set through the tag API.

### 12.4 Tag info/add modal (`i`)

- [ ] 12.4a Open for the highlighted session from any pane focus.
- [ ] 12.4b Load `target.tags("mmux").await?.list().await?`, sort by stripped
  key lexicographically, and render keys without `mmux/` or `@mmux/` prefixes.
- [ ] 12.4c Initial focus is the first tag row when any tag exists, otherwise
  the bottom `Key` field.
- [ ] 12.4d Render bottom add controls: `Key` field, `Value` field, focusable
  `+`, and a `Cancel` button.
- [ ] 12.4e Up/Down move focus row-to-row through existing tag rows; Up from
  add controls returns to the last visible tag row when present.
- [ ] 12.4f Pressing `x` on a focused tag row calls
  `target.tags("mmux").await?.unset(key).await?`, reloads the sorted list, and
  keeps the modal open.
- [ ] 12.4g Pressing `u` on a focused tag row copies that key/value into the
  bottom `Key` and `Value` fields and focuses `Value`.
- [ ] 12.4h Enter on focused `+` applies the same non-empty-value rule as the
  `t` modal. Existing keys are updated through `set`; new keys are added.
- [ ] 12.4i After successful add/update/delete, reload and resort the displayed
  list, clear add/update fields after add/update, and keep the modal open.
- [ ] 12.4j Escape or Enter on focused `Cancel` dismisses without writing.
- [ ] 12.4k Tests: sorted display, stripped keys, row focus movement, delete
  via unset, update preload, update set, empty value no-op, Cancel/Esc dismiss.

### 12.5 Shared dispatch helpers

- [ ] 12.5a Add a small helper for resolving captured modal session targets by
  `(host_id, session_id)` to keep rename/tag apply paths consistent.
- [ ] 12.5b Add a small helper for setting an `mmux` tag from UI text that
  centralizes key trimming, empty-value no-op, and `Target::tags("mmux")`
  usage.
- [ ] 12.5c Add a small helper for deleting a focused `mmux` tag via the new
  `SessionTags::unset` API.
- [ ] 12.5d Keep these helpers binary-private.

### 12.6 Documentation and validation

- [ ] 12.6a Update `bins/mmux/docs/CLI.md` keymap and modal behavior after the
  implementation is concrete.
- [ ] 12.6b Update `bins/mmux/docs/API.md` internal `ModalState` notes after
  the implementation is concrete.
- [ ] 12.6c `cargo fmt --all`
- [ ] 12.6d `cargo test -p motlie-tmux`
- [ ] 12.6e `cargo test -p motlie-mmux`
- [ ] 12.6f `cargo clippy -p motlie-tmux -- -D warnings`
- [ ] 12.6g `cargo clippy -p motlie-mmux -- -D warnings`

## Concrete Test Matrix

| Area | Harness | Required coverage |
|------|---------|-------------------|
| Library attach | Unit + localhost smoke | command construction, process group handoff, exit status mapping, terminal restore |
| Host events | Polling-backed typed stream | add, close, rename, disconnect, one-second snapshot reconciliation |
| Scrollback range | Unit tests | first/middle/exhausted ranges, chunk size, invalid range |
| Layout | Pure unit tests | normal split, portrait mode 64x32, PTY auto-detect threshold 4.0, landscape force flag, MOTD cap, placeholder fallback, resize bounds |
| Input model | Pure unit tests | cyclic focus transitions, scrolling, attach key, modal Enter/Esc, Help modal `h` key, key functions, build date, and short build SHA display |
| Session rename/tags | Pure unit + mock tmux | `r` focus gating, rename prefill/no-op/dispatch, tag prefill, sorted stripped tag display, add/update/delete flows, empty-value no-op |
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
