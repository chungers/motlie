# tmux_select API

## Status

Draft API contract for the planned selector and its required `motlie-tmux`
support. This document is intentionally explicit about current gaps: the
selector binary is not implemented yet, and only the first accepted
`motlie-tmux` gaps have started landing on this branch.

After implementation, this file must be revised into an implemented API
reference that matches code exactly.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
let sample = target.sample_text(&motlie_tmux::ScrollbackQuery::LastLines(80)).await?;
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

Status: implemented as `HostHandle::session_by_id()`. Rename-race lifecycle
coverage remains tracked in PLAN 1.4c.

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
    Short,
}

enum Focus {
    List,
    Detail,
}

struct SelectedSession {
    id: String,
    name: String,
}
```

## Detail Source Contract

The R/B detail pane is decoupled from concrete data sources:

```rust
#[async_trait::async_trait]
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

Implementation should prefer a closed enum over `Box<dyn ...>` for shipped
detail modes:

```rust
enum DetailMode {
    Sample(SampleDetailSource),
    Monitor(MonitorDetailSource),
}
```

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
struct CliConfig {
    target: Option<String>,
    short: bool,
    print_session: bool,
    dashboard: bool,
}
```

Validation rules:

- `--print-session` and `--dashboard` are mutually exclusive
- target is positional SSH URI only
- omitted target means local host

## Testing Contracts

API tests must cover:

- attach command construction and exit status mapping
- host event stream notification mapping
- `LinesRange` scrollback boundaries
- session-id dispatch under rename race
- detail source append/replace/fetch older behavior
- dashboard re-entry and no-loop conditions
