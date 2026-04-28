# mmux API

## Status

Implemented API contract for the initial `mmux` selector and the
`motlie-tmux` support it consumes. This document reflects the code in
`bins/mmux/` and the new support APIs in `libs/tmux`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-28 | @gpt55-dgx | Documented bottom status command rendering as plain labels with underlined shortcut-letter spans instead of parenthesized mnemonics. |
| 2026-04-28 | @gpt55-dgx | Replaced the fixed compact-placeholder threshold with an embedded-logo fit check so landscape panes render the full motlie glyph when it fits. |
| 2026-04-28 | @gpt55-dgx | Documented landscape render regression coverage that keeps the MOTD pane visible for both placeholder and host-provided MOTD content. |
| 2026-04-28 | @gpt55-dgx | Documented round-3 PR feedback coverage for `load_motd_from`, MOTD fallback cases, full/compact placeholder rendering, portrait MOTD omission, and local `read_text_file` edge cases. |
| 2026-04-28 | @gpt55-dgx | Documented PR #228 review cleanup: bounded `read_text_file` for MOTD, typed `SessionId`, decomposed selector state with `StatusBanner`, focused module split, and hidden internal session ids in the list view. |
| 2026-04-27 | @gpt55-dgx | Documented modal padding, button-bar separators, bordered New Session input, and Help build metadata placement. |
| 2026-04-27 | @gpt55-dgx | Documented bottom status command ordering and `l` runtime layout toggling. |
| 2026-04-27 | @gpt55-dgx | Documented `p` as the main-view pane-cycle key and updated status hints. |
| 2026-04-27 | @gpt55-dgx | Documented in-memory retained selector UI state for default attach/detach re-entry. |
| 2026-04-27 | @gpt55-dgx | Documented mode-specific resize bounds: landscape 25/75 and portrait 15/85. |
| 2026-04-27 | @gpt55-dgx | Replaced build metadata shellouts with Rust filesystem/time APIs in `build.rs`. |
| 2026-04-27 | @gpt55-dgx | Documented Help modal build date and last-8-character git SHA display. |
| 2026-04-27 | @gpt55-dgx | Documented compact bottom status direction hints `↑/↓ sel` and `←/→ pane`. |
| 2026-04-27 | @gpt55-dgx | Documented `|` host/IP separator and `(h)elp`-first bottom status command hints. |
| 2026-04-27 | @gpt55-dgx | Documented top status host/IP plus right-justified time and count-only Sessions title. |
| 2026-04-27 | @gpt55-dgx | Documented cyclic Left/Right focus behavior, including landscape MOTD focus. |
| 2026-04-27 | @gpt55-dgx | Renamed the selector binary/package docs to `mmux` / `motlie-mmux`. |
| 2026-04-27 | @gpt55-dgx | Documented Sessions title count/hostname/IP format and removal of the `keys` status label. |
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
| 2026-04-26 | @gpt55-dgx | Updated API reference to implemented reality: selector CLI config, trait-backed sample/monitor detail sources, stable-id create/kill/attach flows, `HostEventStream`, host MOTD read, and `LinesRange` scrollback. |
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
let motd = host.read_text_file(std::path::Path::new("/etc/motd"), 64 * 1024).await?;
```

`mmux` wraps that call in `load_motd_from(host, path)` so the default
`/etc/motd` path and fallback policy can be tested separately. Missing, empty,
whitespace-only, unreadable, and oversized files produce the embedded motlie
placeholder; readable content is returned with trailing whitespace trimmed.

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

The selector reconciles session state by `SessionInfo.id` (`SessionId`), not by
display name.
If `SessionClosed` matches the monitored session id, the selector stops monitor
mode and clears the detail pane until the user's next action.

Status: implemented as a typed stream backed by one-second `list_sessions()`
snapshot reconciliation. It emits stable-id add/close/rename and client
attach/detach events, plus `Disconnect` events on transient list failures.
Direct tmux control-mode host notification wiring is reserved for a future
event-driven implementation; the parser is documented as dormant plumbing.

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
`mmux` PageUp detail fetches.

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

Status: implemented as `HostHandle::session_by_id()`. `SessionInfo.id` is a
non-empty `SessionId`, so session lifecycle dispatch no longer falls back to
display name when an id is absent.

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

Implemented state is decomposed by concern: `HostContext`, `LayoutState`,
`MotdState`, `SessionListState`, `DetailState`, and `StatusBanner`. `AppState`
now coordinates those structs rather than owning one flat collection of UI,
host, selection, detail, layout, and status fields. `main.rs` contains the
entry/run loop. CLI parsing, terminal lifecycle, ForceCommand handling,
target-host identity, detail sources, key handling/event refresh, and rendering
live in `cli.rs`, `terminal.rs`, `forcecommand.rs`, `target_host.rs`,
`detail.rs`, `controller.rs`, and `render.rs`.

The top status bar is derived from the app host identity and current local
clock: `<hostname> | <ip address>` renders as bold left-justified text, and the
current time renders right-justified. The Sessions pane title is derived only
from the live session list length: `Sessions [n]`. List rows show the display
name and attached marker only; stable session ids stay internal for dispatch.
Bottom status text contains compact key hints and app status, not the host
label, current time, layout/focus labels, or a `keys` prefix. Command hints in
the bottom status start with
`(h)elp`, then `(p)ane`, `(m)onitor`, `enter/(a)ttach`, `(n)ew`, `(k)ill`,
`(q)uit`, `(l)ayout`, and the mode-specific resize hint. Direction hints render
as `↑/↓ sel` and `(p)ane`.

Resize bounds are keyed by layout mode. Normal/landscape L/R resizing keeps
both sides at least 25% wide (`25/75` through `75/25`). Portrait T/B resizing
keeps both panes at least 15% tall (`15/85` through `85/15`).
The `l` key toggles `LayoutMode` at runtime and normalizes focus if the MOTD
pane is focused while switching into portrait. Default attach/re-entry retains
that layout choice in memory inside the parent `mmux` process.

## Build Metadata

The binary embeds build metadata in private `BUILD_GIT_SHA` and `BUILD_DATE`
constants. `bins/mmux/build.rs` sets `MMUX_GIT_SHA` from an explicit
environment override or by reading `.git`/`HEAD`/refs/`packed-refs` directly
with Rust filesystem APIs. It sets `MMUX_BUILD_DATE` from an explicit
environment override or from `SystemTime` converted to a UTC `YYYY-MM-DD` date
in Rust. The Help modal opened by `h` renders the build date and only the last
8 characters of the git SHA below the built-in motlie logo and above the
key-function reference. Modal content is padded inside the outer border, and
the button bar is separated from the main content by a horizontal rule.
New Session also renders its session-name input in a bordered field.

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
- default attach/re-entry keeps selected session/list index, layout mode, pane
  split, and focused pane in memory within the parent `mmux` process; this
  state is not persisted across binary runs
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
- `SSH_ORIGINAL_COMMAND` is rejected unless `MOTLIE_MMUX_BYPASS=1`
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
- `p` key focus-cycling behavior in landscape and portrait layouts
- `l` key layout toggling and retained layout re-entry behavior
- status hint arrow-symbol rendering
- bottom status command hints with underlined shortcut-letter spans
- top status rendering for bold hostname/IP and right-justified current time
- session count rendering in the Sessions pane title without hostname/IP
- Help modal open/close behavior, key-function display, build date display,
  and last-8-character build SHA display
- MOTD fallback behavior for missing, empty, whitespace-only, oversized, and
  readable files; embedded-logo width fit for full vs. compact placeholder
  rendering; and landscape full-frame MOTD pane rendering vs. portrait-mode
  MOTD omission
- default attach/re-enter and no-loop conditions

Current implementation coverage:

- `cargo test -p motlie-tmux`: attach command/status including the
  `SIGTTOU`-safe restore helper, `LinesRange`, stable-id host-event diffing,
  stable-id kill coverage, and `read_text_file` local/mock behavior for
  missing, empty, normal, oversized, and unreadable files.
- `cargo test -p motlie-mmux`: CLI mutual exclusion, stable-id
  highlight preservation, `--script` parsing, removed mode-flag rejection,
  layout force-flag parsing, `-s` rejection, PTY aspect
  auto-detection, `q` exit, Enter/`a` attach, detail scroll direction,
  modified-arrow resize fallbacks, `p` pane focus transitions, `l` layout
  toggle behavior, compact status hint rendering, MOTD fallback/readability
  cases, full/compact placeholder rendering, landscape MOTD pane rendering,
  portrait MOTD omission, sample color preservation, Help modal
  key-function/display/close behavior, monitor screen capture, ANSI/VTE
  parsing, and monitored-session-close reset.
