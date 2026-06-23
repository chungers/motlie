# mmux API

## Status

Implemented API contract for the initial `mmux` selector and the
`motlie-tmux` support it consumes. This document reflects the code in
`bins/mmux/` and the new support APIs in `libs/tmux`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-22 | @codex-562-impl | Updated live-test follow-ups: single-space stable-id rows, `s`/`g` toggles back to activity recency, and Help key-list scrolling below the fixed logo. |
| 2026-06-22 | @codex-562-impl | Documented issue #562 reality: session rows render stable tmux ids and list-focused `/` search uses case-insensitive substring matching in current sort order. |
| 2026-06-11 | @mstream453-impl | Added `SessionSortMode::Name` for list-pane `s` sorting and Help shortcut coverage. |
| 2026-05-20 | @codex | Added `Cli.alias` and host-label override behavior so mmux can display operator-provided labels without changing host routing identity. |
| 2026-05-16 | @codex-tmux-tl | Added insertion-point cursor state for mmux modal text fields so focused Left/Right edit inside the field, with single-line and wrapped Send Keys rendering keeping the terminal cursor aligned. |
| 2026-05-03 | @codex | Added `SendKeys` modal state and documented the `s` send-keys flow through `Target::send_keys`. |
| 2026-05-02 | @codex | Changed Session Tags modal add/update/delete/check operations to stage in modal state and flush as a diff only on Ok; Cancel/Esc discard the draft. |
| 2026-05-02 | @codex | Refactored attach status setup to use `Target::status()` with `SessionStatusSnapshot` / `SessionStatusOverrides` instead of app-owned status option plumbing. |
| 2026-05-02 | @codex | Moved environment editing into the New Session modal and apply staged variables through `CreateSessionOptions::initial_environment`; removed the `e` post-creation environment shortcut. |
| 2026-05-02 | @codex | Added `HostHandle::tmux_hostname()` and changed mmux host display labels to use tmux `#{host}` while retaining SSH URI hosts as aliases. |
| 2026-05-02 | @codex | Attach now temporarily overrides session-local `status-left` to unbracketed `#{=50:session_name}` and `status-left-length` to 50, restoring prior local values after detach. |
| 2026-05-02 | @codex | Replaced multi-host `[A]` letter codes with a five-color square palette in the top legend and session rows. |
| 2026-05-02 | @codex | Made kill refresh filter the killed `(host_id, session_id)` so the row is cleared immediately even if the next tmux listing is stale. |
| 2026-05-02 | @codex | Lightened the shared status-bar blue to `#002b55` and kept attach `status-style` matched. |
| 2026-05-02 | @codex | Tightened multi-host kill dispatch: `SelectedSession` carries the captured `SessionInfo`, and kill builds a target from that selected row on the selected host. |
| 2026-05-02 | @codex | Darkened status bars and attach `status-style` to `#002b55` and changed status-bar mnemonic letters to bold coral. |
| 2026-05-02 | @codex | Added multi-host New Session host selection. |
| 2026-05-02 | @codex | Status bars now use dark blue; command shortcut letters render bold colored spans instead of underlined, and attach applies the same blue to tmux `status-style`. |
| 2026-05-02 | @codex | Restored `a` as the attach key and changed list-pane tag grouping to the `g` key with recency-ordered tag groups. |
| 2026-05-02 | @codex | mmux attach now wraps `Target::attach_current_pty()` with best-effort temporary `status-style bg=#002b55,fg=white` setup and local-style restoration after detach. |
| 2026-05-02 | @codex | Removed the `a` attach shortcut; Enter is now the only key that selects a session for attach. |
| 2026-05-02 | @codex | Defaulted the Session Tags key edit column to 30% of the edit strip when there are no tag rows. |
| 2026-05-02 | @codex | Tightened `SessionSortMode::Tag`: rows only count as tagged when they have a visible non-empty checked-tag value, and the `s` toggle selects the first row after sorting so the new top is visible. |
| 2026-05-02 | @codex | Added `SessionSortMode` and list-pane `s` toggle: default activity sort, or tag sort that groups checked-tag rows first and orders by tag value, activity, host code, and session name. |
| 2026-05-02 | @codex | Addressed PR feedback: session refresh now batch-loads selected tag metadata once per host, Session Tags Cancel focus is reachable, modal session identity is grouped, tag UI state is grouped, and render sizing lives in `render.rs`. |
| 2026-05-01 | @codex | Persisted the Session Tags checked key in the internal `@mmux/__selected-key` option, filtered it from modal tag rows, loaded it into `SessionRow`, and rendered the checked tag value as a right-aligned session-list column. |
| 2026-05-01 | @codex | Added the Session Tags modal list model: key width is longest key plus four characters, value takes the remaining width, the marker column shows a `✓` selected by `c`, the visible list is capped at five scrollable rows, and `Tab` cycles Key/Value/Cancel while the edit row submits with Enter. |
| 2026-05-01 | @codex | Documented implemented session rename and tag-management modals: `r` captures host/session id and renames through `Target::rename`; `t` manages `@mmux/` tags through the `motlie-tmux` tag API, including add/update/delete. |
| 2026-04-28 | @gpt55-dgx | Clarified exact `MOTLIE_MMUX_BYPASS=1` behavior and linked issue #232 for env-gated SSH integration coverage. |
| 2026-04-28 | @gpt55-dgx | Consolidated mmux session-list polling so one `list_sessions_now()` loop drives activity ordering and structural state. |
| 2026-04-28 | @gpt55-dgx | Documented one-second quiet visible-row refreshes for activity sorting and recency text. |
| 2026-04-28 | @gpt55-dgx | Documented activity-descending session-list ordering with stable-id selection preservation. |
| 2026-04-28 | @gpt55-dgx | Updated session-list recency rendering to unlabeled `<active> / <age>` text with day formatting and a right margin. |
| 2026-04-28 | @gpt55-dgx | Clarified that recency rendering uses `list_sessions_now()` with the library's tmux-empty-epoch fallback path. |
| 2026-04-28 | @gpt55-dgx | Documented implemented session-list recency rendering with right-aligned `active` and `age` values from `list_sessions_now()`. |
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
| 2026-04-27 | @gpt55-dgx | Documented compact bottom status direction hints `↑/↓` and `←/→ pane`. |
| 2026-04-27 | @gpt55-dgx | Documented `|` host/IP separator and `(h)elp`-first bottom status command hints. |
| 2026-04-27 | @gpt55-dgx | Documented top status host/IP plus right-justified time and count-only Sessions title. |
| 2026-04-27 | @gpt55-dgx | Documented cyclic Left/Right focus behavior, including landscape MOTD focus. |
| 2026-04-27 | @gpt55-dgx | Renamed the selector binary/package docs to `mmux` / `motlie-mmux`. |
| 2026-04-27 | @gpt55-dgx | Documented Sessions title count/hostname/IP format and removal of the `keys` status label. |
| 2026-04-27 | @gpt55-dgx | Documented moving the host label from status text into the Sessions pane title. |
| 2026-04-27 | @gpt55-dgx | Documented arrow-symbol status hints and Help modal key-function content. |
| 2026-05-03 | @codex | Documented portrait mode's 35:65 default T/B split. |
| 2026-04-26 | @gpt55-dgx | Documented the About modal state and build-time git SHA metadata used by the `h` key. |
| 2026-04-26 | @gpt55-dgx | Finalized the CLI mode contract: default mode is attach-and-reenter selector behavior, and `--script` replaces `--print-session` / `--dashboard` for shell integration. |
| 2026-04-26 | @gpt55-dgx | Added `--portrait/-p` and `--landscape/-l` force flags and changed auto-detection to `columns / rows <= 4.0`, making 66x30 portrait. |
| 2026-04-26 | @gpt55-dgx | Set portrait auto-detection to `columns / rows <= 2.0` and embedded the `/tmp/motlie-TOP-CHOICE.txt` glyph as the MOTD-absent fallback icon. |
| 2026-04-26 | @gpt55-dgx | Updated selector API reality for portrait mode: `LayoutMode::Portrait`, `Cli::portrait`, PTY auto-detection, and old `-s` rejection. |
| 2026-04-26 | @gpt55-dgx | Updated API notes for current selector behavior: Enter attach, Left/Right focus transitions, one-second polling-backed session refresh, and ANSI-preserving sample/detail rendering. |
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

Event consumers should reconcile session state by `SessionInfo.id`
(`SessionId`), not by display name.

Status: implemented as a typed stream backed by one-second `list_sessions()`
snapshot reconciliation. It emits stable-id add/close/rename and client
attach/detach events, plus `Disconnect` events on transient list failures.
Direct tmux control-mode host notification wiring is reserved for a future
event-driven implementation; the parser is documented as dormant plumbing.

`mmux` keeps this library stream available but does not start it for the TUI
session list. The selector uses one quiet `list_sessions()` refresh per
second instead; that single snapshot updates recency text, activity-descending
sort order, structural session state, and monitored-session closure handling.
Recency is computed observer-side via an `ActivityTracker` rather than from
a host-clock snapshot.

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

`LinesRange` supports chunked backwards fetch for the R/B pane when the user
scrolls older than the current buffer start.

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
    NewSession { ui: NewSessionModalUi },
    KillSession { session: SelectedSession, button: Button },
    RenameSession { session: SelectedSession, input: String, button: Button },
    SendKeys { session: SelectedSession, ui: SendKeysModalUi },
    SessionKeyValues { session: SelectedSession, ui: SessionKeyValueModalUi },
    Help { scroll: usize },
}

struct SelectedSession {
    host_id: HostId,
    host_label: String,
    info: SessionInfo,
}

impl SelectedSession {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
}

struct NewSessionModalUi {
    input: String,
    hosts: Vec<NewSessionHostChoice>,
    host_index: usize,
    env_rows: Vec<SessionKeyValueRow>,
    env_key_input: String,
    env_value_input: String,
    focus: NewSessionFocus,
    button: Button,
}

enum SendKeysFocus {
    Input,
    Ok,
    Cancel,
}

struct SendKeysModalUi {
    input: String,
    focus: SendKeysFocus,
}

struct SessionKeyValueModalUi {
    kind: SessionKeyValueKind,
    rows: Vec<SessionKeyValueRow>,
    selected_key: Option<String>,
    original_rows: Vec<SessionKeyValueRow>,
    original_selected_key: Option<String>,
    key_input: String,
    value_input: String,
    focus: SessionKeyValueFocus,
}
```

Implemented state is decomposed by concern: `HostFleet`, `LayoutState`,
`SessionListState`, `DetailState`, and `StatusBanner`. `main.rs` owns the live
`HostFleet`; `AppState` owns UI state only and receives `&HostFleet` through
rendering, key handling, refresh, and attach paths. This avoids a duplicated
host snapshot inside app state while keeping layout, selection, detail, modal,
status, and activity tracking together. `main.rs` contains the entry/run loop.
CLI parsing, terminal lifecycle, ForceCommand handling,
target-host identity, detail sources, key handling/event refresh, and rendering
live in `cli.rs`, `terminal.rs`, `forcecommand.rs`, `target_host.rs`,
`detail.rs`, `controller.rs`, and `render.rs`.

The top status bar is derived from the live `HostFleet` and current local
clock: single-host mode renders `<hostname> | <ip address>` as bold
left-justified text, while multi-host mode renders a compact host-color legend
such as `mmux ■ alpha ■ beta`. Host display names come from
`HostHandle::tmux_hostname()`, which calls tmux
`start-server ; display-message -p '#{host}'` over the configured
transport/socket. For SSH targets, mmux separately stores the URI hostname as
`HostEntry.alias` and resolves IPs from that alias. The current time renders
right-justified.
The Sessions pane title is derived only from the live session list length:
`Sessions [n]`. List rows show the display name, attached marker, optional
multi-host color-square column, and right-aligned `<active> / <age>` recency
text with a small right margin. Rows also show the stable tmux session id as
`[$id]` next to the display name with exactly one space as `<name> [$id]`
while retaining the same id for dispatch.
The attached marker is `*` when `SessionInfo::is_attached()` is true.
The list is sorted by
`activity_observed_at_local` descending — operator-side wall clock at the
moment mmux last saw the row's `session.activity` advance — by default.
When the operator presses `g` with the list focused,
`SessionSortMode::TagGroup` groups rows with visible non-empty checked-tag
values before rows without a displayed tag. Tag groups are ordered by the most
recent activity in each group, and rows within a group then sort by activity
time, host order, and session name. Empty checked-tag values sort with rows that
have no displayed tag. The `g` toggle selects the first row in the new order;
pressing `g` again restores `SessionSortMode::Activity`. Pressing `s` with the
list focused toggles `SessionSortMode::Name`: the first press sorts by session
name and selects the first row in name order, and the next press restores
`SessionSortMode::Activity`. `preserve_selection()` re-finds the highlighted row
by stable session id after refreshes. A single quiet one-second
`list_sessions()` refresh keeps the active ordering current and notices
structural session changes. Recency text is observer-relative
for the activity column (`local_now − activity_observed_at_local`) and
`local_now − session.created` for the age column under an NTP-synced
clock assumption — see `DESIGN.md` §Clock Handling for the rationale.
Durations use `now`, `m`, `h`, or `d`; day values keep at most one decimal
digit.

When the session list has focus, `/` starts quick search. `AppState` stores the
transient search query, printable characters extend it, and
`SessionListState` selects the first row whose session display name contains
the query, case-insensitively, in the current row order. Another `/`, Up, or
Down cancels search mode without moving the highlight; non-text command keys
first leave search mode before normal handling.
This is substring (`contains`) matching, not prefix matching, following the
vi/vim/less `/` convention: search is unanchored/match-anywhere, while prefix
matching is a typeahead/completion idiom. Iterating matches with `n`/`N` is a
possible future follow-up and is out of scope for #562.
Bottom status text contains compact key hints and app status, not the host
label, current time, layout/focus labels, or a `keys` prefix. Command hints in
the bottom status start with `tab ↑/↓`, then `help`, `prompt`, `attach`,
`new`, `kill`, `rename`, `group`, `layout`, `quit`, and the
mode-specific resize hint. Attach uses the `a` shortcut; command shortcut
letters are rendered bold coral in command labels.
Direction hints render as `↑/↓`.

Plain Enter while the session list has focus forces an immediate live-preview
capture for the highlighted session. The detail-pane title displays a bold
`live` label.

`r` opens `RenameSession` only when the session list has focus. The modal
captures `(host_id, session_id)` plus the current display name, prepopulates the
`Session Name` field, and dispatches changed names through
`HostHandle::session_by_id()` and `Target::rename()`.

`p` opens `SendKeys` for the highlighted session from any pane focus. The modal
keeps focus explicit (`Input`, `Ok`, `Cancel`), renders a compact text field
with `To: <session> on <host>`, wraps long input at word boundaries by growing
the field height against a fixed input width, scrolls capped input to keep the
cursor visible, sends from either focused `Ok` or non-empty
`Input` Enter, parses the submitted text with `KeySequence::parse`, resolves
the captured stable session id with `HostHandle::session_by_id()`, and
dispatches the parsed sequence exactly through `Target::send_keys`.
Ctrl-Enter uses the same parsed input dispatch, then waits 500 ms and
dispatches a second explicit `{Enter}` sequence through the same target.
Submitted input ending in `$$` is normalized by stripping that suffix and using
the same delayed Enter dispatch mode. If the suffix is the entire submitted
input, mmux skips the initial key dispatch and sends only the delayed `{Enter}`.
After a successful send, mmux forces `refresh_detail(..., true)` so the live
preview reflects the target promptly.
The modal accepts tmux key-name shorthand such as `{C-m}` for Enter because
`KeySequence` passes valid raw key names through to tmux.
Focused modal text inputs set the terminal cursor to the insertion point; the
terminal session configures that cursor as a blinking bar while mmux owns the
screen.
No new `motlie-tmux` API is required for this feature; the existing gap is only
UI policy around which key owns pane focus, handled in mmux by moving pane
cycling to Tab.

`t` opens a `SessionKeyValues` modal in tag mode for the highlighted session. Rows are loaded from
`Target::tags("mmux").await?.list().await?`, sorted lexicographically by
stripped key, rendered without `@mmux/`, filtered to hide the internal
`@mmux/__selected-key` option, and shown in a five-row scroll window. The modal
keeps row focus and bottom field focus explicit with `SessionKeyValueFocus`; `Tab`
cycles the bottom Key/Value cells, Ok, and Cancel button, `Shift-Tab` reverses
that cycle. Enter on either edit field stages a non-empty, non-reserved key/value
row in modal state; `x` stages deletion of the focused row; `m` preloads the
bottom fields. Plain `u` and `b` move row focus while a tag row is focused.
Pressing `c` on a focused tag row stages the checked key and
renders `✓` in that row's marker column. The modal keeps the original rows and
selected key alongside the draft; Enter on focused Ok diffs the draft and writes
only the resulting `SessionTags::set`, `SessionTags::unset`, and
`@mmux/__selected-key` changes. Enter on Cancel or Esc discards the draft without
tmux writes. `__selected-key` is reserved for the internal marker.

`n` opens `NewSession`, which includes the same key/value list mechanics for
initial environment variables. The modal stages variables locally, sorts them
lexicographically by key, supports `u` to preload a staged row and `x` to remove
one, and applies staged variables through
`CreateSessionOptions::initial_environment` when the session is created. These
values are passed to `tmux new-session -e` and are visible to the first shell or
command in the new session.

`fetch_fleet_rows()` enriches each `SessionRow` with the checked key/value by
batch-listing `@mmux/` options once per host refresh, resolving
`@mmux/__selected-key`, and copying the selected tag value into app state.
The progressive refresh path drains all ready `HostRefreshResult`s into a
batch before reconciling. Successful host results replace that host's rows,
failed host results contribute status diagnostics while leaving last-good rows
visible, and the merged list is sorted once per drain rather than once per host
result. If a refresh task exits without sending a result, the selector awaits
completed task handles and reports the host label in the status banner.
`session_list_line()` renders
that value in a right-aligned field after the session name. `i` is not assigned
by this feature.

Resize bounds are keyed by layout mode. Normal/landscape L/R resizing keeps
both sides at least 25% wide (`25/75` through `75/25`). Portrait T/B resizing
keeps both panes at least 15% tall (`15/85` through `85/15`).
The `l` key toggles `LayoutMode` at runtime while preserving list/detail focus.
Default attach/re-entry retains that layout choice in memory inside the parent
`mmux` process.

## Build Metadata

The binary embeds build metadata in private `BUILD_GIT_SHA` and `BUILD_DATE`
constants. `bins/mmux/build.rs` sets `MMUX_GIT_SHA` from an explicit
environment override or by reading `.git`/`HEAD`/refs/`packed-refs` directly
with Rust filesystem APIs. It sets `MMUX_BUILD_DATE` from an explicit
environment override or from `SystemTime` converted to a UTC `YYYY-MM-DD` date
in Rust. The Help modal opened by `h` renders the build date and only the last
8 characters of the git SHA below the built-in motlie logo and above the
key-function reference. The logo/build metadata are fixed at the top of the
modal; the key-function reference below them scrolls with Up/Down, PgUp/PgDn,
or `j`/`k`. Modal content is padded inside the outer border, and the button bar
is separated from the main content by a horizontal rule.
New Session renders its session-name input in a bordered field. In multi-host
mode, it also renders a Host dropdown above the session-name field and carries
the selected host id through `Ok` so create dispatches to that host.

## Detail Source Contract

The R/B detail pane is a selected-session active-pane live preview:

```rust
async fn render_live_preview(
    host: &motlie_tmux::HostHandle,
    session: &SelectedSession,
) -> anyhow::Result<String>;

async fn fetch_older_lines(
    host: &motlie_tmux::HostHandle,
    session: &SelectedSession,
    before_line: usize,
    count: usize,
) -> anyhow::Result<String>;
```

`render_live_preview` resolves the selected session by stable id and calls
`Target::capture_with_options(CaptureNormalizeMode::ScreenStable)`, which
captures the active pane for a session-level target. mmux does not use the
all-pane `capture_all_with_options()` path for default detail rendering because
the current UI has no pane selector and the detail pane is designed around one
active screen. `fetch_older_lines` uses
`sample_text_with_options(... LinesRange ..., ScreenStable, None)` for chunked
scrollback. Captured content is parsed through `ansi-to-tui` so escape bytes do
not leak into ratatui text.

## Session Operations

Create:

```rust
let target = host
    .create_session(
        &new_session_name,
        &motlie_tmux::CreateSessionOptions {
            initial_environment,
            ..Default::default()
        },
    )
    .await?;
```

In multi-host mode, the binary picks the `HostHandle` from the New Session
modal's selected host id before calling `create_session`. The staged
environment rows are converted to `SessionEnvVar` values before dispatch.

Kill:

The `KillSession` modal starts on `Cancel`; Left/Right and `Tab` /
`Shift-Tab` can move to `Ok`, and Enter dispatches only while `Ok` is active.

```rust
let selected = SelectedSession {
    host_id: row.host_id.clone(),
    host_label: row.host_label.clone(),
    info: row.session.clone(),
};
let target = host.target_for_session_info(selected.info.clone());
target.kill().await?;
refresh_sessions_excluding(
    fleet,
    app,
    true,
    (selected.host_id.clone(), selected.id().to_string()),
).await?;
```

Attach:

```rust
let target = host.session_by_id(selected.id()).await?.ok_or(SessionVanished)?;
let status = target.status().await?;
let snapshot = status.snapshot().await?;
let overrides = motlie_tmux::SessionStatusOverrides {
    style: Some(motlie_tmux::StatusStyle::new("bg=#002b55,fg=white")?),
    left: Some(motlie_tmux::StatusLeft::new("#{=50:session_name}")?),
    left_length: Some(motlie_tmux::StatusLeftLength::new(50)?),
};
status.apply(&overrides).await.ok();
let exit = target
    .attach_current_pty_with_options(&motlie_tmux::AttachOptions {
        suppress_transition_output: true,
    })
    .await;
let _ = status.restore(&snapshot).await;
let exit = exit?;
```

The exact `SessionVanished` error type belongs to the binary unless `motlie-tmux`
already has a suitable structured error when implementation starts. The
status override calls are best-effort in mmux: diagnostics are suppressed by
default and attach continues even if the remote tmux rejects setup or
restoration. Setting `MMUX_ATTACH_LOG=1` prints those diagnostics and disables
transition-output cleanup for attach debugging.

## CLI Boundary

The API layer should expose parsed CLI config to the application loop:

```rust
struct Cli {
    ssh_uris: Vec<String>,
    portrait: bool,
    landscape: bool,
    script: bool,
}
```

Validation rules:

- `--script` prints the selected session name to stdout and exits without
  attaching
- `--alias` is not accepted; SSH targets use endpoint identity labels from
  `motlie_tmux::SshConfig::endpoint_alias()`
- SSH endpoint labels include user, host, non-default port, and non-default
  tmux socket identity; they omit `identity-file`
- without `--script`, the selector attaches and re-enters after detach when the
  attach child succeeds or the selected session still exists
- default attach/re-entry keeps selected session/list index, layout mode, pane
  split, and focused pane in memory within the parent `mmux` process; this
  state is not persisted across binary runs
- `--portrait` / `-p` forces portrait layout
- portrait layout initializes the `T`/`B` split at 35:65
- `--landscape` / `-l` forces landscape layout
- `--portrait` and `--landscape` are mutually exclusive
- without a layout force flag, startup reads the connecting PTY dimensions and
  selects portrait when `columns / rows <= 4.0`
- the removed `--print-session` and `--dashboard` flags are not accepted in the
  finalized CLI contract
- the old `-s` flag is rejected
- target is positional SSH URI only
- omitted target means local host
- `SSH_ORIGINAL_COMMAND` is rejected unless `MOTLIE_MMUX_BYPASS` is exactly
  `1`; exact bypass delegates to `sh -lc "$SSH_ORIGINAL_COMMAND"` before the
  selector starts

Env-gated SSH/ForceCommand integration coverage is tracked in
[issue #232](https://github.com/chungers/motlie/issues/232).

## Testing Contracts

API tests must cover:

- attach command construction and exit status mapping
- host event stream notification mapping
- `LinesRange` scrollback boundaries
- session-id dispatch under rename race
- detail source render/fetch older behavior
- active-pane color preservation, no all-pane listing for live preview, and
  ANSI/VTE parser behavior
- modified-arrow resize fallback behavior
- `p` key focus-cycling behavior in landscape and portrait layouts
- `l` key layout toggling and retained layout re-entry behavior
- status hint arrow-symbol rendering
- bottom status command hints with bold coral shortcut-letter spans
- top status rendering for bold hostname/IP or multi-host color legend and
  right-justified current time
- session count rendering in the Sessions pane title without hostname/IP
- Help modal open/close behavior, key-function display, build date display,
  and last-8-character build SHA display
- landscape and portrait layout rendering with session-list/detail panes and
  no intro pane
- default attach/re-enter and no-loop conditions

Current implementation coverage:

- `cargo test -p motlie-tmux`: attach command/status including the
  `SIGTTOU`-safe restore helper, `LinesRange`, stable-id host-event diffing,
  stable-id kill coverage, and tmux status override snapshot/restore behavior.
- `cargo test -p motlie-mmux`: CLI mutual exclusion, stable-id
  highlight preservation, `--script` parsing, removed mode-flag rejection,
  layout force-flag parsing, `-s` rejection, PTY aspect
  auto-detection, `q` exit, `a` attach, detail scroll direction,
  modified-arrow resize fallbacks, Tab pane focus transitions, `l` layout
  toggle behavior, compact status hint rendering, landscape/portrait
  session-list/detail rendering, active-pane color preservation, Help modal
  key-function/display/close behavior, live detail capture, and ANSI/VTE
  parsing.
