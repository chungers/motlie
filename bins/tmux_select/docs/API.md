# tmux_select API

## Status

Implemented API contract for the initial `tmux_select` selector and the
`motlie-tmux` support it consumes. This document reflects the code in
`bins/tmux_select/main.rs` and the new support APIs in `libs/tmux`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-27 | @gpt55-dgx | Documented moving the host label from status text into the Sessions pane title. |
| 2026-04-27 | @gpt55-dgx | Documented arrow-symbol status hints and Help modal key-function content. |
| 2026-04-27 | @gpt55-dgx | Documented portrait mode's 30:70 default T/B split. |
| 2026-04-26 | @gpt55-dgx | Documented the About modal state and build-time git SHA metadata used by the `h` key. |
| 2026-04-26 | @gpt55-dgx | Finalized the CLI mode contract: default mode is attach-and-reenter selector behavior, and `--script` replaces `--print-session` / `--dashboard` for shell integration. |
| 2026-04-26 | @gpt55-dgx | Added `--portrait/-p` and `--landscape/-l` force flags and changed auto-detection to `columns / rows <= 4.0`, making 66x30 portrait. |
| 2026-04-26 | @gpt55-dgx | Set portrait auto-detection to `columns / rows <= 2.0` and embedded the `/tmp/motlie-TOP-CHOICE.txt` glyph as the MOTD-absent fallback icon. |
| 2026-04-26 | @gpt55-dgx | Updated selector API reality for portrait mode: `LayoutMode::Portrait`, `Cli::portrait`, PTY auto-detection, and old `-s` rejection. |
| 2026-04-26 | @gpt55-dgx | Updated API notes for current selector behavior: Enter/`a` attach, Left/Right focus transitions, one-second polling-backed session refresh, and ANSI-preserving sample/detail rendering. |
| 2026-04-26 | @gpt55-dgx | Updated implementation notes for the second validation round: monitor detail now captures rendered screen snapshots with `ScreenStable` plus ANSI/VTE parsing, resize accepts modified-arrow fallbacks, and attach PTY restore is `SIGTTOU`-safe. |
| 2026-04-26 | @gpt55-dgx | Updated implementation notes for validation fixes: monitor detail uses `CaptureNormalizeMode::PlainText`, `q` exits like `Ctrl-C`, and dashboard can re-enter after detach even when tmux returns a non-zero detach status. |
| 2026-04-26 | @gpt55-dgx | Updated API reference to implemented reality: selector CLI config, trait-backed sample/monitor detail sources, stable-id create/kill/attach flows, `HostEventStream`, host shell MOTD read, and `LinesRange` scrollback. |
| 2026-04-26 | @gpt55-dgx | Mark current-PTY attach and stable session-id lookup as started in `motlie-tmux`; host event stream and windowed scrollback remain open selector dependencies. |
| 2026-04-26 | @gpt55-dgx | Initial API contract for PR #227: documents existing library dependencies, accepted `motlie-tmux` gaps, and the selector's internal API shape. |

## Existing motlie-tmux Surface Used

The selector design builds on these existing `motlie-tmux` APIs:

```rust
use motlie_tmux::{HostHandle, SshConfig};

let host = HostHandle::local();
let sessions = host.list_sessions().await?;
let maybe_target = host.session("dev").await?;
let created = host
    .create_session("new-dev", &motlie_tmux::CreateSessionOptions::default())
    .await?;
```

For SSH targets:

```rust
let host = SshConfig::parse("ssh://user@host?identity-file=/path/to/key")?
    .connect()
    .await?;
let sessions = host.list_sessions().await?;
```

For existing target operations:

```rust
let target = host.session("dev").await?.ok_or_else(|| anyhow::anyhow!("gone"))?;
let sample = target
    .sample_text_with_options(
        &motlie_tmux::ScrollbackQuery::LastLines(80),
        &motlie_tmux::CaptureOptions::with_mode(motlie_tmux::CaptureNormalizeMode::ScreenStable),
        None,
    )
    .await?;
target.kill().await?;
```

## Accepted motlie-tmux Gaps

### Current PTY Attach

Design target:

```rust
pub struct AttachExit {
    pub status: std::process::ExitStatus,
}

impl Target {
    pub async fn attach_current_pty(&self) -> motlie_tmux::Result<AttachExit>;
}
```

Required behavior:

- local targets spawn `tmux attach-session -t <target>` with inherited stdio
- SSH targets spawn interactive `ssh -t ... tmux attach-session -t <target>`
- no pipe wrapping and no terminal-byte proxy through the selector
- child has its own process group
- controlling terminal is transferred to the child and restored after wait
- signal exits are surfaced as `128 + signal`

Status: implemented in `motlie-tmux` for session targets; localhost PTY smoke
coverage is still tracked in PLAN 1.1g.

### Host Event Stream

Design target:

```rust
pub enum HostEvent {
    SessionsChanged,
    SessionAdded { id: String, name: String },
    SessionClosed { id: String, name: String },
    SessionRenamed { id: String, old: String, new: String },
    ClientAttached { session_id: String },
    ClientDetached { session_id: String },
    Disconnect { reason: String },
}

impl HostHandle {
    pub async fn watch_host_events(&self) -> motlie_tmux::Result<HostEventStream>;
}
```

The selector reconciles session state by `SessionInfo.id`, not by display name.
If `SessionClosed` matches the monitored session id, the selector stops monitor
mode and clears the detail pane until the user's next action.

Status: implemented as a typed stream backed by one-second `list_sessions()`
snapshot reconciliation. It emits stable-id add/close/rename and client
attach/detach events, plus `Disconnect` events on transient list failures.
Direct tmux control-mode host notification wiring is still tracked in
`PLAN.md` 1.2b.

### Windowed Scrollback

Design target:

```rust
pub enum ScrollbackQuery {
    LastLines(usize),
    Until { pattern: regex::Regex, max_lines: usize },
    LastLinesUntil { lines: usize, stop_pattern: regex::Regex },
    LinesRange { older_than_lines: usize, count: usize },
}
```

`LinesRange` supports chunked backwards fetch for the R pane. It is used by
both sample mode and monitor mode when the user scrolls older than the current
buffer start.

Status: implemented in `motlie-tmux` capture paths and used by
`tmux_select` PageUp detail fetches.

### Stable Session-Id Dispatch

The selector captures the highlighted `SessionInfo.id` when opening destructive
confirmation modals. If current `Target` lifecycle methods still dispatch by
display name, implementation must add a library-owned way to resolve or kill by
stable session id before the binary uses that path.

One acceptable shape:

```rust
impl HostHandle {
    pub async fn session_by_id(&self, id: &str) -> motlie_tmux::Result<Option<Target>>;
}
```

Status: implemented as `HostHandle::session_by_id()`. Session `kill()` and
`rename()` dispatch by stable session id when available.

## Selector Internal Types

The binary should keep the public API small and put most behavior behind
private modules. The following shapes document the intended boundaries.

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

```rust
enum LayoutMode {
    Normal,
    Portrait,
}

enum Focus {
    List,
    Detail,
}

enum ModalState {
    NewSession { input: String, button: Button },
    KillSession { id: String, name: String, button: Button },
    Help,
}

struct SelectedSession {
    id: String,
    name: String,
}
```

The Sessions pane title is derived from `AppState.host_label`. Status text
contains only current time and compact key hints, not the host label.

## Build Metadata

The binary embeds a build-time git SHA in a private `BUILD_GIT_SHA` constant.
`bins/tmux_select/build.rs` sets `TMUX_SELECT_GIT_SHA` from
`git rev-parse HEAD`, or uses an explicit `TMUX_SELECT_GIT_SHA` environment
override when provided. The Help modal opened by `h` renders that value below
the built-in motlie logo and key-function reference.

## Detail Source Contract

The R/B detail pane is decoupled from concrete data sources:

```rust
trait SessionDetailSource {
    async fn activate(
        &mut self,
        host: &motlie_tmux::HostHandle,
        session: &SelectedSession,
    ) -> anyhow::Result<()>;

    async fn tick(&mut self) -> anyhow::Result<Option<DetailDelta>>;

    async fn fetch_older(
        &mut self,
        before_line: usize,
        count: usize,
    ) -> anyhow::Result<Vec<String>>;

    async fn deactivate(&mut self) -> anyhow::Result<()>;
}

enum DetailDelta {
    Append(String),
    Replace(String),
}
```

The implementation uses the stable Rust private async trait directly; no
`async-trait` dependency is required. Shipped detail modes use a closed enum
over `Box<dyn ...>`:

```rust
enum DetailMode {
    Sample(SampleDetailSource),
    Monitor(MonitorDetailSource),
}
```

`SampleDetailSource` resolves the selected session by stable id and calls
`sample_text_with_options(..., CaptureNormalizeMode::ScreenStable, None)` so
normal detail mode can preserve ANSI color/style. `MonitorDetailSource`
resolves the selected session by stable id, calls
`capture_all_with_options(CaptureNormalizeMode::ScreenStable)`, and renders the
visible content through `ansi-to-tui`'s VTE parser. Both modes parse the
captured content before rendering so escape bytes do not leak into ratatui
text, while monitor mode remains a rendered screen mirror suitable for TUI
sessions instead of a raw `%output` control-mode transcript.

## Session Operations

Create:

```rust
let target = host
    .create_session(&new_session_name, &motlie_tmux::CreateSessionOptions::default())
    .await?;
```

Kill:

```rust
let selected = SelectedSession {
    id: highlighted.id.clone(),
    name: highlighted.name.clone(),
};
let target = host.session_by_id(&selected.id).await?;
if let Some(target) = target {
    target.kill().await?;
}
```

Attach:

```rust
let target = host.session_by_id(&selected.id).await?.ok_or(SessionVanished)?;
let exit = target.attach_current_pty().await?;
```

The exact `SessionVanished` error type belongs to the binary unless `motlie-tmux`
already has a suitable structured error when implementation starts.

## CLI Boundary

The API layer should expose parsed CLI config to the application loop:

```rust
struct Cli {
    ssh_uri: Option<String>,
    portrait: bool,
    landscape: bool,
    script: bool,
}
```

Validation rules:

- `--script` prints the selected session name to stdout and exits without
  attaching
- without `--script`, the selector attaches and re-enters after detach when the
  attach child succeeds or the selected session still exists
- `--portrait` / `-p` forces portrait layout
- portrait layout initializes the `T`/`B` split at 30:70
- `--landscape` / `-l` forces landscape layout
- `--portrait` and `--landscape` are mutually exclusive
- without a layout force flag, startup reads the connecting PTY dimensions and
  selects portrait when `columns / rows <= 4.0`
- the removed `--print-session` and `--dashboard` flags are not accepted in the
  finalized CLI contract
- the old `-s` flag is rejected
- target is positional SSH URI only
- omitted target means local host
- `SSH_ORIGINAL_COMMAND` is rejected unless `MOTLIE_TMUX_SELECT_BYPASS=1`
  delegates to `sh -lc "$SSH_ORIGINAL_COMMAND"` before the selector starts

## Testing Contracts

API tests must cover:

- attach command construction and exit status mapping
- host event stream notification mapping
- `LinesRange` scrollback boundaries
- session-id dispatch under rename race
- detail source append/replace/fetch older behavior
- sample color preservation, monitor screen capture, and ANSI/VTE parser
  behavior
- modified-arrow resize fallback behavior
- status hint arrow-symbol rendering
- host label rendering in the Sessions pane title, not the status line
- Help modal open/close behavior, key-function display, and build SHA display
- default attach/re-enter and no-loop conditions

Current implementation coverage:

- `cargo test -p motlie-tmux`: attach command/status including the
  `SIGTTOU`-safe restore helper, `LinesRange`, stable-id host-event diffing,
  and stable-id kill coverage.
- `cargo test -p motlie-tmux-select`: CLI mutual exclusion, stable-id
  highlight preservation, `--script` parsing, removed mode-flag rejection,
  layout force-flag parsing, `-s` rejection, PTY aspect
  auto-detection, `q` exit, Enter/`a` attach, detail scroll direction,
  modified-arrow resize fallbacks, Left/Right focus transitions, compact
  status hint rendering, sample color preservation, Help modal
  key-function/display/close behavior, monitor screen capture, ANSI/VTE
  parsing, and monitored-session-close reset.
