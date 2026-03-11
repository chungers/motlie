# motlie-tmux API Reference

Practical usage guide for the `motlie-tmux` library. The document is organized
foundation-up: transports first, then the common API layer built on top.
Inconsistencies and surprises are called out inline where they arise, marked
with **`@claude NOTE`** for easy grep and issue creation.

**Cargo.toml dependency**:

```toml
[dependencies]
motlie-tmux = { path = "libs/tmux" }
tokio = { version = "1", features = ["full"] }
```

All examples assume an async context (`#[tokio::main]` or `#[tokio::test]`).

---

## Table of Contents

**Part I — Transport Layer**
1. [LocalTransport](#1-localtransport)
2. [SshTransport](#2-sshtransport)
3. [MockTransport](#3-mocktransport)
4. [Transport Comparison](#4-transport-comparison)

**Part II — Common API (transport-agnostic)**
5. [HostHandle](#5-hosthandle)
6. [Session Lifecycle](#6-session-lifecycle)
7. [Discovery](#7-discovery)
8. [Target and Navigation](#8-target-and-navigation)
9. [Sending Input](#9-sending-input)
10. [Capturing Output](#10-capturing-output)
11. [Structured Command Execution](#11-structured-command-execution)
12. [Advanced Capture — Modes and Fidelity](#12-advanced-capture--modes-and-fidelity)
13. [Scrollback Sampling](#13-scrollback-sampling)
14. [Geometry and Reflow Detection](#14-geometry-and-reflow-detection)
15. [History Limit Management](#15-history-limit-management)

**Part III — Reference**
16. [Normalization Utilities](#16-normalization-utilities)
17. [Type Quick Reference](#17-type-quick-reference)

---

# Part I — Transport Layer

The transport layer executes shell commands either locally or remotely.
All higher-level tmux operations (discovery, capture, control) are built
on `TransportKind::exec()`, which dispatches statically to the concrete
transport — no vtable, no dynamic dispatch (DC6).

```rust
// TransportKind is the static-dispatch enum:
pub enum TransportKind {
    Local(LocalTransport),
    Mock(MockTransport),
    Ssh(SshTransport),
}
```

Once a `HostHandle` is constructed with a transport, the same top-level
API calls are available regardless of which variant is used. Behavioral
differences exist (e.g. Mock never errors, SSH has keepalives) and are
called out per-transport below and in the comparison table.

---

## 1. LocalTransport

Executes commands via `sh -c` subprocesses on the local machine. This is
the default and simplest path.

### Quick start

```rust
use motlie_tmux::HostHandle;

// Convenience constructor — LocalTransport with 10s timeout, default socket
let host = HostHandle::local();
```

### Custom timeout

```rust
use motlie_tmux::HostHandle;
use motlie_tmux::transport::{TransportKind, LocalTransport};

let transport = LocalTransport::with_timeout(std::time::Duration::from_secs(30));
let host = HostHandle::new(TransportKind::Local(transport), None);
```

> **`@claude NOTE`**: `HostHandle::local()` hardcodes a 10s transport timeout.
> There is no builder or setter to change it — you must drop to
> `HostHandle::new()` + `LocalTransport::with_timeout()`. Consider adding
> `HostHandle::local_with_timeout()` or a builder on `HostHandle`.

### Custom tmux socket

```rust
use motlie_tmux::{HostHandle, TmuxSocket};
use motlie_tmux::transport::{TransportKind, LocalTransport};

// Named socket (tmux -L myapp)
let host = HostHandle::new(
    TransportKind::Local(LocalTransport::new()),
    Some(TmuxSocket::Name("myapp".into())),
);

// Explicit socket path (tmux -S /tmp/myapp.sock)
let host = HostHandle::new(
    TransportKind::Local(LocalTransport::new()),
    Some(TmuxSocket::Path("/tmp/myapp.sock".into())),
);
```

### Characteristics

- **Timeout**: Configurable per-transport, default 10s. Governs each individual
  `sh -c` subprocess (tmux commands like `list-sessions`, `capture-pane`, etc.).
- **Error messages**: Include stderr from the failed subprocess.
- **No server required**: tmux server starts automatically on first session creation.

---

## 2. SshTransport

Executes commands on a remote host over SSH using `russh` 0.46.
Authentication is via ssh-agent only.

### Prerequisites

- `ssh-agent` running with `SSH_AUTH_SOCK` exported
- Keys loaded: `ssh-add ~/.ssh/id_ed25519`
- Remote host in `~/.ssh/known_hosts` (for `Verify` policy)

### Connect

```rust
use motlie_tmux::{HostHandle, SshTransport, SshConfig, HostKeyPolicy};
use motlie_tmux::transport::TransportKind;

let config = SshConfig::new("server.example.com", "deploy")
    .with_port(22)                                          // default: 22
    .with_host_key_policy(HostKeyPolicy::Verify)            // default: Verify
    .with_timeout(std::time::Duration::from_secs(10))       // default: 10s
    .with_keepalive(Some(std::time::Duration::from_secs(30))); // default: 30s

let ssh = SshTransport::connect(config).await?;
let host = HostHandle::new(TransportKind::Ssh(ssh), None);

// From here, `host` is used identically to a local HostHandle.
let sessions = host.list_sessions().await?;
```

### Host key policies

```rust
// Verify (default) — host must exist in ~/.ssh/known_hosts
SshConfig::new("host", "user")
    .with_host_key_policy(HostKeyPolicy::Verify);

// Trust-on-first-use — accept + persist on first connect, reject on mismatch
SshConfig::new("host", "user")
    .with_host_key_policy(HostKeyPolicy::TrustFirstUse);

// Insecure — accept all keys (logs warning). Dev/testing only.
SshConfig::new("host", "user")
    .with_host_key_policy(HostKeyPolicy::Insecure);
```

> **`@claude NOTE`**: `TrustFirstUse` is fail-closed — if persisting the
> learned key to `~/.ssh/known_hosts` fails (e.g. file not writable), the
> connection is **rejected**. This is intentional but may surprise users
> who expect TOFU to "just work". The error is logged at `error` level
> with an actionable message ("check that ~/.ssh/known_hosts is writable").

### Connection status

```rust
// Only available on the SshTransport directly, not through HostHandle.
// You must retain a reference to the SshTransport if you need this.
if ssh.is_closed() {
    println!("SSH connection lost");
}
```

> **`@claude NOTE`**: `is_closed()` is only accessible on `SshTransport`,
> not through `HostHandle` or `TransportKind`. There is no transport-agnostic
> way to check connection health. `LocalTransport` has no equivalent (always
> "connected"). Consider adding `TransportKind::is_closed()` that returns
> `false` for Local/Mock.

### Characteristics

- **Timeout**: `SshConfig::timeout` governs connection, authentication, and
  each `exec()` call (channel open + exec + output collection are all inside
  one `tokio::time::timeout` boundary).
- **Keepalive**: `SshConfig::keepalive_interval` sends SSH keepalives.
  `None` disables. Local/Mock have no equivalent (not applicable).
- **Concurrency**: The SSH handle mutex is held only during
  `channel_open_session()`, not for the full command lifetime. Multiple
  concurrent `exec()` calls on the same connection are safe.
- **Authentication**: ssh-agent only. Key file or password auth is not supported.
  Error messages are actionable (OC3): "is SSH_AUTH_SOCK set?",
  "Add a key with: ssh-add ~/.ssh/id_ed25519".

---

## 3. MockTransport

Canned responses for unit testing. No real commands are executed.

### Setup

```rust
use motlie_tmux::HostHandle;
use motlie_tmux::transport::{TransportKind, MockTransport};

let mock = MockTransport::new()
    .with_response("list-sessions", "build\t$0\t0\t1\t2\t\n")
    .with_response("capture-pane", "hello world\n")
    .with_default("");  // fallback for unmatched commands

let host = HostHandle::new(TransportKind::Mock(mock), None);
let sessions = host.list_sessions().await?;
assert_eq!(sessions[0].name, "build");
```

### Response matching

`with_response(pattern, response)` matches when the executed command
**contains** the pattern as a substring. Multiple responses for the same
pattern are returned in FIFO order; when exhausted, the last one repeats.

```rust
let mock = MockTransport::new()
    // First call containing "list-sessions" returns this:
    .with_response("list-sessions", "sess1\t$0\t0\t0\t1\t\n")
    // Second and subsequent calls return this:
    .with_response("list-sessions", "sess1\t$0\t0\t0\t1\t\nsess2\t$1\t0\t0\t1\t\n")
    .with_default("");
```

> **`@claude NOTE`**: Pattern matching iterates the internal HashMap, so
> if a command matches multiple registered patterns, which one wins is
> **non-deterministic** (HashMap iteration order). Use specific-enough
> patterns to avoid ambiguity. For example, prefer `"list-sessions"` over
> `"list"` if you also have `"list-panes"` registered.

### Convenience helper for tests

```rust
fn mock_host(mock: MockTransport) -> HostHandle {
    HostHandle::new(TransportKind::Mock(mock), None)
}

#[tokio::test]
async fn test_session_lifecycle() {
    let mock = MockTransport::new()
        .with_default("")
        .with_response("list-sessions", "test\t$0\t0\t0\t1\t\n");
    let host = mock_host(mock);
    let target = host.create_session("test", None, None).await.unwrap();
    assert_eq!(target.session_name(), "test");
}
```

### Characteristics

- **No timeout**: Responses are instant. No timeout configuration.
- **Never errors on exec**: Always returns `Ok(response)`. To test error
  paths, you must use `LocalTransport` or construct errors at a higher level.
- **Shell channel**: `open_shell()` returns a `MockShellChannel` with empty
  data that immediately yields `Eof` on read.

> **`@claude NOTE`**: There is no way to make `MockTransport::exec()` return
> an `Err`. This means error handling paths in discovery, capture, and
> control code cannot be unit-tested with mocks alone. Consider adding
> `with_error(pattern, message)` to `MockTransport`.

---

## 4. Transport Comparison

| Aspect | Local | SSH | Mock |
|--------|-------|-----|------|
| Timeout | 10s default, configurable | 10s default, configurable | None |
| Error includes stderr | Yes | Yes | N/A (never errors) |
| `is_closed()` | Not available | Available on `SshTransport` | Not available |
| Keepalive | N/A | Configurable, default 30s | N/A |
| `open_shell()` | Spawns `sh` | PTY `xterm` 80x24 + shell | Empty data, immediate Eof |
| Auth | N/A | ssh-agent only | N/A |

> **`@claude NOTE`**: `SshTransport::open_shell()` requests a PTY at fixed
> 80x24. There is no API to specify dimensions. This may matter for
> applications that need a specific terminal size for the remote shell.

---

# Part II — Common API (transport-agnostic)

Everything below uses the same call patterns across Local, SSH, and Mock
transports. The transport is selected once at `HostHandle` construction.
Behavioral differences (e.g. Mock never errors, SSH shell channels use PTY)
are noted in the transport sections above.

---

## 5. HostHandle

`HostHandle` is the entry point for all tmux operations on a host.
It wraps `Arc<...>` and is cheaply `Clone`able — multiple handles share
the same transport and exec lock map.

```rust
use motlie_tmux::HostHandle;

// Localhost (most common)
let host = HostHandle::local();

// With explicit transport + socket (any transport)
use motlie_tmux::transport::{TransportKind, LocalTransport};
use motlie_tmux::TmuxSocket;

let host = HostHandle::new(
    TransportKind::Local(LocalTransport::new()),
    Some(TmuxSocket::Name("myapp".into())),
);

// HostHandle is Clone — both handles share state
let host2 = host.clone();
```

---

## 6. Session Lifecycle

### Create

```rust
// Minimal — detached session with default shell
let target = host.create_session("build", None, None).await?;
// Returns a Target at session level

// With named window and startup command
let target = host.create_session("dev", Some("editor"), Some("vim")).await?;
```

### Kill

```rust
target.kill().await?;
// Works at any level: kills session, window, or pane depending on target level.
```

### Rename

```rust
target.rename("new_name").await?;
```

> **`@claude NOTE`**: `rename()` mutates tmux state but does **not** update
> the `Target`'s internal address. The impact depends on target level:
>
> - **Session rename** — correctness-significant. `target_string()` uses the
>   session name (`s.name`), so after rename the old name is stale and
>   subsequent operations **fail** because tmux can't find it. You must
>   obtain a fresh `Target`:
>   ```rust
>   target.rename("new_name").await?;
>   let target = host.session("new_name").await?.unwrap(); // fresh handle
>   ```
> - **Window rename** — metadata drift only. `target_string()` uses
>   `session:index` (not the window name), so commands continue to work.
>   However, the cached `WindowInfo.name` becomes stale — callers displaying
>   window names should re-query.
>
> Consider having `rename()` return a new `Target` with the updated address,
> or making `Target` internally mutable.

> **`@claude NOTE`**: `rename()` at pane level returns `Err("cannot rename a pane")`.
> This is a tmux limitation — tmux has no `rename-pane` command. The error is
> clear, but it would be better to make this a compile-time restriction or at
> least document the asymmetry in the `Target::rename()` doc comment.

---

## 7. Discovery

All discovery methods return empty `Vec`s (not errors) when there is no
tmux server running or no entities exist.

### List sessions

```rust
let sessions = host.list_sessions().await?;
for s in &sessions {
    println!("{} (id={}, windows={}, attached={})",
        s.name, s.id, s.window_count, s.attached);
}
```

### Find a session by name

```rust
match host.session("build").await? {
    Some(t) => println!("found: {}", t.target_string()),
    None    => println!("not found"),
}
```

### Find by TargetSpec

```rust
use motlie_tmux::TargetSpec;

// Session only
let t = host.target(&TargetSpec::session("build")).await?;

// Session + window (by index or name)
let t = host.target(&TargetSpec::session("build").window(0)).await?;
let t = host.target(&TargetSpec::session("build").window_name("editor")).await?;

// Session + window + pane
let t = host.target(&TargetSpec::session("build").window(0).pane(1)).await?;

// Parse from string
let t = host.target(&TargetSpec::parse("build:0.1")?).await?;
// Returns Option<Target> — None if the entity doesn't exist.
```

> **`@claude NOTE`**: `TargetSpec::pane()` **panics** (not `Result`) if
> `.window()` was not called first. This enforces the tmux hierarchy
> (pane requires window context), but a panic is a surprising contract
> for a builder API. Consider returning `Result` instead.
> ```rust
> // This panics at runtime:
> let _ = TargetSpec::session("s").pane(0);
> // Must do:
> let _ = TargetSpec::session("s").window(0).pane(0);
> ```

### List attached clients

```rust
let clients = host.list_clients().await?;
for c in &clients {
    println!("{}x{} on '{}'", c.width, c.height, c.session);
}
// Useful for geometry/reflow detection (section 15).
```

---

## 8. Target and Navigation

`Target` is a unified handle at any hierarchy level — session, window, or pane.
It is **not `Clone`**; to share, use `HostHandle::session()` or
`HostHandle::target()` to obtain separate handles.

### Identity

```rust
target.level();          // TargetLevel::Session | Window | Pane
target.target_string();  // "build", "build:0", "build:0.1"
target.session_name();   // available at every level
target.session_info();   // Some(&SessionInfo) — session level only
target.window_info();    // Some(&WindowInfo) — window level only
target.pane_address();   // Some(&PaneAddress) — pane level only
target.address();        // &TargetAddress enum
```

### Navigate down the hierarchy

```rust
// Session → windows
let windows = target.children().await?;

// Session → window by index
let win = target.window(0).await?; // Option<Target>

// Session or window → pane by index
let pane = target.pane(0).await?;  // Option<Target>

// Pane level: children() returns empty Vec
```

> **`@claude NOTE`**: `target.pane(index)` from **session level** resolves
> via the **active window** at call time. If the active window changes
> between calls (e.g. another client switches windows), the same
> `target.pane(0)` call returns a different pane. For deterministic
> targeting, navigate explicitly: `target.window(0).pane(0)`.

### Direct pane targeting

```rust
use motlie_tmux::PaneAddress;

let addr = PaneAddress::parse("%5", "build:0.1")?;
let pane = target.pane_by_address(&addr);
// Immediate — no tmux query. Creates a Target from the address.
```

---

## 9. Sending Input

### Literal text

```rust
target.send_text("echo hello").await?;
// Types the text into the pane. Does NOT append Enter.
```

### Key sequences

```rust
use motlie_tmux::{KeySequence, SpecialKey};

// Parse with {SpecialKey} escapes
let keys = KeySequence::parse("echo hello{Enter}")?;
target.send_keys(&keys).await?;

// Builder API
let keys = KeySequence::literal("ls -la").then_enter();
target.send_keys(&keys).await?;

// Chain multiple keys
let keys = KeySequence::literal("search term")
    .then_key(SpecialKey::Enter)
    .then_key(SpecialKey::CtrlC);
target.send_keys(&keys).await?;

// Multiple special keys
let keys = KeySequence::parse("{C-c}{C-c}")?;
target.send_keys(&keys).await?;
```

**Available special keys**: `{Enter}`, `{Tab}`, `{Escape}` / `{Esc}`,
`{Up}`, `{Down}`, `{Left}`, `{Right}`, `{C-c}`, `{C-d}`, `{C-z}`,
`{C-l}`, `{Space}`, `{BSpace}`.

Arbitrary tmux key names are accepted via `{Raw}` but validated against
a shell-dangerous character blocklist (defense-in-depth — the values are
also shell-escaped before execution).

---

## 10. Capturing Output

### Visible pane content

```rust
let content = target.capture().await?;
println!("{}", content);
```

> **`@claude NOTE`**: At session or window level, `capture()` captures the
> **active pane** only — not all panes. Use `capture_all()` to capture
> every pane under the target. This is consistent with tmux behavior
> (`capture-pane -t session` targets the active pane) but may surprise
> callers who expect session-level capture to be exhaustive.

### With scrollback history

```rust
// -200 = start 200 lines above the visible area, through end of visible
let content = target.capture_with_history(-200).await?;
```

### All panes under a target

```rust
use std::collections::HashMap;
use motlie_tmux::PaneAddress;

let all: HashMap<PaneAddress, String> = target.capture_all().await?;
// Session level: all panes in all windows
// Window level: all panes in that window
// Pane level: single-entry map

for (addr, content) in &all {
    println!("--- {} ---\n{}", addr, content);
}
```

---

## 11. Structured Command Execution

`Target::exec()` runs a shell command **inside the target's tmux pane** and
returns structured output. This is different from the transport-level `exec()`
which runs raw shell commands outside tmux.

```rust
use std::time::Duration;

let result = target.exec("echo hello", Duration::from_secs(10)).await?;

if result.success() {
    println!("stdout: {}", result.stdout);    // "hello"
    println!("exit: {}", result.exit_code);   // 0
} else {
    eprintln!("failed (exit {}): {}", result.exit_code, result.stdout);
}
```

### How it works

Sends `<command> ; echo "__ML<uuid>__ $?"` to the pane via `send_keys`,
then polls scrollback (`capture-pane -ep -S -500`) until the sentinel
marker appears. Extracts stdout (lines between command echo and sentinel)
and parses the exit code.

### Concurrency

Multiple `Target` handles to the **same pane** share an exec lock (keyed by
resolved `pane_id`). Concurrent `exec()` calls serialize correctly per-pane.
Exec locks for different panes are independent.

```rust
// Session-level and pane-level targets to the same active pane share one lock:
let session_target = host.session("build").await?.unwrap();
let pane_target = session_target.pane(0).await?.unwrap();
// exec() on either serializes against the same lock for pane_id "%5"
```

### Two kinds of timeout

> **`@claude NOTE`**: `Target::exec()` has **its own timeout parameter** that
> governs sentinel polling. This is separate from the transport timeout
> (`LocalTransport::timeout` / `SshConfig::timeout`) which governs each
> individual tmux command. A 30s exec timeout with a 10s transport timeout
> means: each `capture-pane` poll can take up to 10s, and the overall
> sentinel wait can take up to 30s. These are independent knobs with no
> documentation linking them. Consider documenting the interaction or
> adding a note to `SshConfig::with_timeout()`.

### Limitations

- **Requires a shell prompt**: The pane must have a running shell that can
  accept input. `exec()` will hang (until timeout) if the pane is running
  an interactive program like `vim`, `less`, or `top`.
- **Shell detection**: Detects fish shell (uses `$status` instead of `$?`)
  via `pane_current_command`. Other exotic shells are not detected.
- **Wrap tolerance**: Works in narrow panes — the sentinel parser joins
  lines and searches in concatenated text.

---

## 12. Advanced Capture — Modes and Fidelity

### Capture normalization modes

Three modes control how captured content is processed:

```rust
use motlie_tmux::{CaptureOptions, CaptureNormalizeMode};

// Raw (default) — no transformation
let opts = CaptureOptions::default();
let result = target.capture_with_options(&opts).await?;
// result.text: tmux-rendered text
// result.raw_text: None

// ScreenStable — normalize line endings, trim width padding, preserve ANSI
let opts = CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable);
let result = target.capture_with_options(&opts).await?;
// result.text: normalized, ANSI sequences preserved
// result.raw_text: Some(original -ep capture before normalization)

// PlainText — strip all ANSI/control sequences, normalize
let opts = CaptureOptions::with_mode(CaptureNormalizeMode::PlainText);
let result = target.capture_with_options(&opts).await?;
// result.text: clean plain text
// result.raw_text: None
```

### Fidelity metadata

When `detect_reflow: true`, the library takes geometry snapshots before and
after the capture to detect instability:

```rust
use motlie_tmux::{CaptureOptions, FidelityIssue};

let opts = CaptureOptions {
    detect_reflow: true,
    ..Default::default()
};
let result = target.capture_with_options(&opts).await?;

if result.fidelity.degraded {
    for issue in result.fidelity.issues.as_ref().unwrap() {
        match issue {
            FidelityIssue::ClientResize    => { /* terminal resized during capture */ }
            FidelityIssue::PaneResize      => { /* pane dimensions changed */ }
            FidelityIssue::HistoryTruncated => { /* scrollback evicted by limit */ }
            FidelityIssue::OverlapResync    => { /* overlap dedup failed */ }
        }
    }
} else {
    // Clean — issues field is None (zero allocation on hot path)
}
```

### Bulk capture with options

```rust
use motlie_tmux::{CaptureOptions, CaptureNormalizeMode, CaptureResult, PaneAddress};
use std::collections::HashMap;

let opts = CaptureOptions::with_mode(CaptureNormalizeMode::PlainText);
let all: HashMap<PaneAddress, CaptureResult> =
    target.capture_all_with_options(&opts).await?;
```

---

## 13. Scrollback Sampling

Query-based extraction of scrollback history.

### Last N lines

```rust
use motlie_tmux::ScrollbackQuery;

let query = ScrollbackQuery::LastLines(50);
let text = target.sample_text(&query).await?;
// Last 50 lines, trailing blank lines trimmed.
```

### Scan backwards until pattern

```rust
use regex::Regex;
use motlie_tmux::ScrollbackQuery;

let query = ScrollbackQuery::Until {
    pattern: Regex::new(r"^\$ ")?,  // shell prompt
    max_lines: 500,
};
let text = target.sample_text(&query).await?;
// Returns from the last matching line to the end.
// If no match: returns all captured content.
```

### Last N lines, stop at pattern

```rust
let query = ScrollbackQuery::LastLinesUntil {
    lines: 200,
    stop_pattern: Regex::new(r"^---")?,
};
let text = target.sample_text(&query).await?;
```

### Incremental sampling with overlap dedup

For polling workflows, overlap-aware dedup merges consecutive captures
without duplicating shared content:

```rust
use motlie_tmux::{CaptureOptions, ScrollbackQuery};

let opts = CaptureOptions {
    overlap_lines: 5,
    ..Default::default()
};
let query = ScrollbackQuery::LastLines(100);

// First capture — no previous text
let first = target.sample_text_with_options(&query, &opts, None).await?;

// Subsequent capture — pass previous text for dedup
let second = target.sample_text_with_options(
    &query, &opts, Some(&first.text),
).await?;
// second.text = first.text + new-only lines (merged)
// On dedup failure: second.fidelity includes OverlapResync
```

> **`@claude NOTE`**: `overlap_lines` must be **>= 2** for dedup to activate.
> With 0 or 1, `overlap_deduplicate()` returns the current capture unchanged
> with no fidelity issues — it silently does nothing rather than warning.
> This threshold is undocumented in the function signature.

---

## 14. Geometry and Reflow Detection

Query pane geometry and detect capture-time instability from client or
pane size changes.

### Pane geometry

```rust
let geo = target.pane_geometry().await?;
println!("{}x{}, history {}/{}",
    geo.pane_width, geo.pane_height,
    geo.history_size, geo.history_limit);
```

### Geometry snapshot and comparison

```rust
let before = target.geometry_snapshot().await?;
// ... perform operations ...
let after = target.geometry_snapshot().await?;

let issues = before.compare(&after);
// Vec<FidelityIssue> — empty if stable.
// Only compares clients attached to the target's session (ignores
// clients attached to other sessions to avoid false positives).
```

---

## 15. History Limit Management

Tmux's `history-limit` controls max scrollback lines per pane.

> **`@claude NOTE`**: `history-limit` only affects panes created **after**
> the setting is applied. Existing panes retain their creation-time limit.
> This is a tmux limitation, not a library limitation, but it means
> `set_history_limit()` has no effect on already-running sessions unless
> you create new windows/panes afterward.

### Global default

```rust
// Set before creating sessions
host.set_global_history_limit(50000).await?;

// Query
let limit = host.get_global_history_limit().await?;
```

### Per-session

```rust
// Set for this session's future panes
target.set_history_limit(100000).await?;

// Query
let limit = target.get_history_limit().await?;
```

---

# Part III — Reference

## 16. Normalization Utilities

Standalone functions re-exported from the crate root, usable independently
of any transport or target:

```rust
use motlie_tmux::{
    strip_ansi, normalize_screen_stable, normalize_plain_text, overlap_deduplicate,
};

// Strip ANSI escape sequences and C0 control chars (preserves \n, \t)
let clean = strip_ansi("\x1b[32mgreen\x1b[0m text");
assert_eq!(clean, "green text");

// ScreenStable: canonical line endings, trim width-artifact trailing spaces
let norm = normalize_screen_stable("hello   \r\nworld  \n\n\n");
assert_eq!(norm, "hello\nworld\n");

// PlainText: strip ANSI then ScreenStable
let plain = normalize_plain_text("\x1b[1mbold\x1b[0m   \n");
assert_eq!(plain, "bold\n");

// Overlap dedup between consecutive captures
let (merged, issues) = overlap_deduplicate(
    "line1\nline2\nline3",  // previous
    "line2\nline3\nline4",  // current
    2,                       // overlap_lines (must be >= 2)
);
assert_eq!(merged, "line1\nline2\nline3\nline4");
assert!(issues.is_empty());
```

---

## 17. Type Quick Reference

### Core handles

| Type | Description | Clone |
|------|-------------|-------|
| `HostHandle` | Entry point — one per tmux host | Yes (Arc) |
| `Target` | Unified session/window/pane handle | No |

### Target addressing

| Type | Description |
|------|-------------|
| `TargetSpec` | Builder: `session()` → `.window()` → `.pane()`, or `parse()` |
| `TargetAddress` | Enum: `Session(SessionInfo)`, `Window(WindowInfo)`, `Pane(PaneAddress)` |
| `TargetLevel` | Enum: `Session`, `Window`, `Pane` |
| `PaneAddress` | Stable `pane_id` (%N) + display fields (session, window, pane) |

### Tmux metadata

| Type | Key fields |
|------|-----------|
| `SessionInfo` | name, id, attached, window_count, group |
| `WindowInfo` | session_name, index, name, active, pane_count |
| `PaneInfo` | address, current_command, pid, width, height, active |
| `ClientInfo` | width, height, session |

### Input

| Type | Description |
|------|-------------|
| `KeySequence` | Parsed key sequence: `parse()`, `literal()`, `then_enter()`, `then_key()` |
| `SpecialKey` | Enum: Enter, Tab, Escape, Up, Down, Left, Right, CtrlC, CtrlD, CtrlZ, CtrlL, Space, BSpace, Raw(String) |

### Capture and fidelity

| Type | Description |
|------|-------------|
| `CaptureOptions` | history_start, normalize mode, overlap_lines, detect_reflow |
| `CaptureNormalizeMode` | Enum: Raw (default), ScreenStable, PlainText |
| `CaptureResult` | text, raw_text (Option), fidelity |
| `OutputFidelity` | degraded (bool), issues (Option<Vec>) — None on clean hot path |
| `FidelityIssue` | Enum: ClientResize, PaneResize, HistoryTruncated, OverlapResync |
| `ScrollbackQuery` | Enum: LastLines(n), Until { pattern, max_lines }, LastLinesUntil { lines, stop_pattern } |
| `ExecOutput` | stdout, exit_code, success() |

### Geometry

| Type | Description |
|------|-------------|
| `GeometrySnapshot` | clients + pane geometry + session; `compare()` → Vec<FidelityIssue> |
| `PaneGeometry` | pane_width, pane_height, history_size, history_limit |

### Transport

| Type | Description |
|------|-------------|
| `TransportKind` | Enum: Local, Ssh, Mock — static dispatch |
| `LocalTransport` | Subprocess exec, configurable timeout |
| `SshTransport` | russh 0.46, ssh-agent auth; `connect()`, `is_closed()` |
| `SshConfig` | host, port, user, host_key_policy, timeout, keepalive_interval |
| `MockTransport` | Canned responses; `with_response()`, `with_default()` |
| `HostKeyPolicy` | Enum: Verify (default), TrustFirstUse, Insecure |
| `TmuxSocket` | Enum: Name(String), Path(String) |

### Shell channel (low-level, used by monitor layer — Phase 2a)

| Type | Description |
|------|-------------|
| `ShellChannelKind` | Enum: Local, Ssh, Mock — `write()`, `read()` |
| `ShellEvent` | Enum: Data(Vec<u8>), Eof |
