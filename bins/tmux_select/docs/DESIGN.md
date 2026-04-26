# tmux_select Design

## Status

Draft.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-26 | @gpt55-dgx | Initial DESIGN for GitHub issue #226: local/remote tmux session selector TUI, session detail sources, monitoring mode, modal create/kill flows, accepted current-PTY attach gap, host-wide SSH integration, and SVG mock. |

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
- `LB` lists tmux sessions on the target host and has default focus.
- Up and Down move the highlighted session in `LB`.
- Left and Right resize the `L` / `R` split in the main selector view.
- `R` initially shows sampled detail for the highlighted session.
- `R` detail is supplied through a trait so future view models can summarize or
  otherwise transform session content.
- Pressing `m` puts `R` into monitoring mode for the highlighted session, using
  the `motlie-tmux` monitor/history pipeline to show live updates.
- Pressing `n` opens a centered `New Session` modal with a session-name text
  field and `Cancel` / `Ok` buttons.
- Pressing `k` opens a centered `Kill session <name>?` confirmation modal with
  `Cancel` / `Ok` buttons.
- In modal dialogs, Left and Right choose between `Cancel` and `Ok`; Enter
  exits the modal and applies `Ok` when selected.
- Pressing `g` or Enter in the main selector exits the TUI and attaches the
  current user PTY to the highlighted session.
- The binary accepts an optional SSH URI / host target. Omitted means local host.
- For SSH targets, listing, MOTD, sampling, create, kill, monitor, and attach
  all operate against the SSH target.
- For SSH targets, attach must open an interactive SSH PTY to the target host
  and attach that remote PTY to the selected remote tmux session.
- A bottom status bar shows target host, current time, and supported keys.
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
        |      +-- HostHandle::session(name)?.kill()
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

The final CLI shape can also accept `--target <ssh-uri>` if PLAN chooses a more
explicit flag. DESIGN only requires one accepted form and consistent help text.

## Layout

The terminal is split into:

- body area: everything except the bottom status bar
- status bar: one terminal row

The body area is split horizontally into `L` and `R`.

`L` is split vertically:

- `LT`: MOTD, height `min(rendered_motd_lines + chrome, 30% of L height)`
- `LB`: session list, remaining height

The main selector keymap is:

| Key | Action |
|-----|--------|
| Up / Down | Move highlighted session |
| Left / Right | Resize `L` / `R` split |
| `m` | Start/switch `R` monitoring mode for highlighted session |
| `n` | Open `New Session` modal |
| `k` | Open kill confirmation modal |
| Enter / `g` | Attach highlighted session |
| Ctrl-C | Exit selector without attach |

Modal keymaps override the main keymap.

## SVG Mock

The DESIGN mock source is checked in beside this document:

![tmux_select TUI mock](./tmux-select-mock.svg)

If GitHub issue rendering supports the chosen SVG embedding path, this same SVG
should be attached or linked from issue #226 after the branch is pushed.

## R Pane Detail Source

The `R` pane should depend on a trait, not directly on sampling or monitoring
implementation details.

```rust
#[async_trait::async_trait]
trait SessionDetailSource {
    async fn activate(
        &mut self,
        host: &motlie_tmux::HostHandle,
        session_name: &str,
    ) -> anyhow::Result<()>;

    async fn tick(&mut self) -> anyhow::Result<Option<String>>;

    async fn deactivate(&mut self) -> anyhow::Result<()>;
}
```

Initial shipped implementations:

- `SampleDetailSource`: resolves `host.session(name)`, captures session content,
  sorts panes by `(window, pane)`, omits empty panes, and renders text sections.
- `MonitorDetailSource`: starts `host.watch_session()` or the equivalent
  monitor/history composition, then renders a rolling history into `R`.

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

1. Parse CLI target.
2. Connect to local or SSH target with `motlie-tmux`.
3. Load target host MOTD.
4. List sessions.
5. Initialize UI state with `LB` focused and first session highlighted.
6. Render sample detail for the highlighted session, if any.

### Highlight Change

1. Up/Down updates selected session index.
2. If `R` is in sample mode, refresh sample detail for the new session.
3. If `R` is in monitoring mode, keep monitoring the previous monitored session
   until the user presses `m` again. This avoids implicit monitor teardown when
   the user is only browsing.

### Monitoring Mode

1. Pressing `m` stops any existing monitor/detail source.
2. Start monitoring the highlighted session.
3. Subscribe to session output and render a bounded rolling history into `R`.
4. Status bar shows the monitored session.
5. Killing the monitored session or exiting the TUI stops monitor state.

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
3. Resolve `host.session(name)`.
4. Call `Target::kill()`.
5. Stop monitor state if it was monitoring that session.
6. Refresh session list.
7. Move highlight to the next valid row.

### Attach

1. Pressing Enter or `g` in the main selector records the highlighted session.
2. Stop monitor/detail state.
3. Restore raw mode and leave alternate screen.
4. Resolve the highlighted session to a `Target`.
5. Call the accepted `motlie-tmux` attach API.
6. Return the attach exit status as the selector process exit result.

## Accepted motlie-tmux Library Gaps

### Current PTY Attach

Issue #226 accepts adding a foreground attach capability to `motlie-tmux`.

Candidate API:

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

### Remote MOTD

The binary can use existing `HostHandle::download()` to retrieve `/etc/motd`
from SSH targets into a temporary local file. If PLAN finds that unsuitable,
the narrower library addition should be a host-level text-file read helper:

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

## Alternatives

### A. New Binary Built Directly On motlie-tmux (Recommended)

Create `tmux_select` as a focused binary that uses `motlie-tmux` APIs for all
tmux and SSH operations.

Pros:

- matches issue #226 directly
- clean binary boundary
- avoids coupling selector UX to the broader driver REPL/TUI command surface
- keeps tmux logic in `motlie-tmux`
- works for ForceCommand deployment

Cons:

- needs some TUI state machinery duplicated from existing frontend patterns
- depends on the accepted attach API being added to `motlie-tmux`

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

## Dependency Choices

| Dependency | Use | Decision |
|------------|-----|----------|
| `ratatui` | layout/widgets/rendering | Use. Already used by tmux examples and driver frontend. |
| `crossterm` | terminal raw mode, alternate screen, key events | Use. Already paired with ratatui in repo. |
| `ansi-to-tui` | optional ANSI rendering for captured/monitored pane content | Use only if sample/monitor rendering needs styled output. Plain text is acceptable for the first pass. |
| `async-trait` | async detail-source trait | Use if a trait object or async trait implementation is needed. Already used in repo. |
| `tempfile` | remote MOTD download target | Use if remote MOTD is implemented through `HostHandle::download()`. Already a dev dependency in parts of the repo; PLAN should decide package placement. |

## Testing Strategy

DESIGN identifies the test surfaces; PLAN must make these concrete.

- Unit tests for layout calculations:
  - MOTD height cap
  - status bar reservation
  - `L` / `R` resize bounds
- Unit tests for state transitions:
  - highlight movement
  - sample vs monitor mode
  - modal button selection
  - create/kill success and error paths
- Mock-backed tests through `motlie-tmux`:
  - session list rendering
  - detail source rendering
  - create session refresh and highlight
  - kill session refresh and highlight
- Terminal smoke tests:
  - raw mode and alternate-screen restoration
  - Ctrl-C behavior
  - attach path restores terminal before handoff
- Localhost integration:
  - create temporary session
  - list and sample it
  - monitor it
  - kill it
- SSH integration:
  - target an SSH URI
  - read remote MOTD
  - list remote sessions
  - monitor remote session
  - attach to remote selected session through an interactive PTY

## Open Questions

- Should `tmux_select` use a positional SSH URI, `--target`, or both?
- Should modal `Esc` cancel, or should only `Cancel` plus Enter close the modal?
- Should monitor mode follow selection automatically or only switch on `m`? This
  design chooses only switch on `m` for predictability.
- Should `New Session` allow window size/history options in the first version?
  This design says no; use defaults initially.
- Should remote targets be allowlisted by config for ForceCommand deployments?
  This design recommends local-only ForceCommand initially.
