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

**Runnable examples** are in [`examples/`](../examples/) with full instructions
in [`examples/README.md`](../examples/README.md).

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-02 | @codex | Added `CreateSessionOptions::initial_environment` for variables that must be visible to the first pane process, and documented that `SessionEnvironment::set/unset` only affects future tmux-spawned processes. |
| 2026-05-02 | @codex | Added scoped session environment APIs: `Target::environment()`, `SessionEnvironment::{set,unset,read,list}`, public `SessionEnvVar`, and `SESSION_ENV_VAR_VALUE_MAX_BYTES`. Tags and environment variables now use scoped helper handles only; the one-off tag wrapper methods were removed from the public `Target` API. |
| 2026-05-02 | @codex | Replaced direct `Target` status option methods with `Target::status() -> SessionStatus`, plus `SessionStatusSnapshot` / `SessionStatusOverrides` for attach-time snapshot/apply/restore flows. |
| 2026-05-02 | @codex | Added narrow session-local status-left/status-style value types and scoped status APIs for temporary attach display overrides. |
| 2026-05-02 | @codex | Added `HostHandle::list_tags_for_session_infos(prefix, sessions)` to batch-read session metadata tags for a fresh session listing in one tmux command. |
| 2026-05-01 | @codex | Added `HostHandle::target_for_session_info()` so consumers enriching a fresh `list_sessions()` result can build a session `Target` without issuing a second session-discovery query. |
| 2026-05-01 | @codex | Added session metadata tag deletion: `SessionTags::unset(key)` removes a user-defined session option with tmux `set-option -u` while preserving session-only scope, stable-session-id dispatch, and prefix/key validation. |
| 2026-04-30 | @codex | Added session metadata tags via tmux user-defined session options: `Target::tags(prefix)`, scoped `SessionTags`, and public `SessionTag`. Tags are session-target only, stored as `@prefix/key`, use stable session ids for dispatch, and validate prefix/key/value bounds for poller-safe metadata. |
| 2026-04-29 | @opus47-macos-tmux | Removed `HostHandle::list_sessions_now()` and `SessionListing`. There is no portable, side-effect-free way to read the host clock across tmux versions (`run-shell` corrupts the operator's attached pane on tmux ≤ 3.4). Recency math moves to the consumer: `list_sessions()` already aggregates `window_activity` into `SessionInfo.activity`, and binaries that need observer-relative recency keep their own per-session tracker. `mod discovery` is now private — all access flows through `HostHandle::*`. |
| 2026-04-28 | @gpt55-dgx | Made `list_sessions_now()` tolerate tmux versions where `#{epoch}` expands empty by falling back to a local clock clamped to session timestamps. |
| 2026-04-28 | @gpt55-dgx | Added `SessionInfo.activity`, non-lossy `attached_count`, and `HostHandle::list_sessions_now()` / `SessionListing` for skew-free session recency math. |
| 2026-04-28 | @gpt55-dgx | Replaced the selector-oriented host shell hook note with bounded `HostHandle::read_text_file`, documented `SessionId` as the stable non-empty session id type, and clarified that host events are currently polling-backed. |
| 2026-04-26 | @gpt55-dgx | Document that current-PTY attach restores the parent foreground process group through a `SIGTTOU`-safe path so selector/dashboard callers do not remain stopped after detach. |
| 2026-04-26 | @gpt55-dgx | Document `SessionWatchOptions::normalize`, available to watch-session consumers that need to strip raw ANSI/control bytes before text rendering. |
| 2026-04-26 | @gpt55-dgx | Document selector support APIs including host metadata reads, `HostHandle::watch_host_events`, `HostEventStream`, and `ScrollbackQuery::LinesRange`. |
| 2026-04-26 | @gpt55-dgx | Document `HostHandle::session_by_id`, `AttachExit`, and `Target::attach_current_pty` added for tmux selector Phase 1.1 / 1.4. |

---

## Table of Contents

**Part I — Transport Layer**
1. [LocalTransport](#1-localtransport)
2. [SshTransport](#2-sshtransport)
   - 2a. [SshConfig URI Parsing and Connect](#2a-sshconfig-uri-parsing-and-connect)
3. [MockTransport](#3-mocktransport)
4. [Transport Comparison](#4-transport-comparison)

**Part II — Common API (transport-agnostic)**
5. [HostHandle](#5-hosthandle)
6. [Session Lifecycle](#6-session-lifecycle)
7. [Discovery](#7-discovery)
   - 7a. [Host Event Stream](#host-event-stream)
8. [Target and Navigation](#8-target-and-navigation)
   - 8a. [Current PTY Attach](#current-pty-attach)
   - 8b. [Session Tags](#session-tags)
9. [Sending Input](#9-sending-input)
10. [Capturing Output](#10-capturing-output)
11. [Structured Command Execution](#11-structured-command-execution)
   - 11a. [Tracked Execution (DC31)](#tracked-execution-dc31)
   - 11b. [Socket Isolation (DC30)](#11b-socket-isolation-dc30)
12. [Advanced Capture — Modes and Fidelity](#12-advanced-capture--modes-and-fidelity)
13. [Scrollback Sampling](#13-scrollback-sampling)
14. [Geometry and Reflow Detection](#14-geometry-and-reflow-detection)
15. [History Limit Management](#15-history-limit-management)

**Part II-b — Output Monitoring Pipeline (DC24, Track A)**
17. [Monitoring Sessions](#17-monitoring-sessions)
18. [OutputBus and Subscriptions](#18-outputbus-and-subscriptions)
19. [JoinedStream — Multi-Pane View](#19-joinedstream--multi-pane-view)
20. [Sink Pipeline](#20-sink-pipeline)

**Part II-c — External-Agent Substrate (Track B)**
21. [Predicate Filtering — filter_fn](#21-predicate-filtering--filter_fn)
22. [Rolling Transcript / History (DC28)](#22-rolling-transcript--history-dc28)
23. [Fleet — Multi-Host Coordination (DC27)](#23-fleet--multi-host-coordination-dc27)

**Part II-d — TUI (DC32)**
23b. [Split-Screen REPL TUI Mirror](#23b-split-screen-repl-tui-mirror-dc32)

**Part III — Reference**
24. [Normalization Utilities](#24-normalization-utilities)
25. [Type Quick Reference](#25-type-quick-reference)

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
differences exist (e.g. Mock errors only via `with_error()`, SSH has keepalives) and are
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

For the common "localhost, but with a different timeout" case, use the
convenience constructor:

```rust
use motlie_tmux::HostHandle;

let host = HostHandle::local_with_timeout(std::time::Duration::from_secs(30));
```

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10g)*: `HostHandle::local()` hardcodes a 10s
> transport timeout. There is no builder or setter to change it — you must
> drop to `HostHandle::new()` + `LocalTransport::with_timeout()`. Consider
> adding `HostHandle::local_with_timeout()` or a builder on `HostHandle`. **Fixed**: Added `HostHandle::local_with_timeout(Duration)` constructor.

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
Authentication is via ssh-agent by default, or via an explicit key file
(`identity-file` URI parameter or `with_identity_file()` builder).

### Prerequisites

- `ssh-agent` running with `SSH_AUTH_SOCK` exported (unless using `identity-file`)
- Keys loaded: `ssh-add ~/.ssh/id_ed25519` (unless using `identity-file`)
- Remote host in `~/.ssh/known_hosts` (for `Verify` policy)

### Connect

```rust
use motlie_tmux::{HostHandle, SshTransport, SshConfig, HostKeyPolicy};
use motlie_tmux::transport::TransportKind;

let config = SshConfig::new("server.example.com", "deploy")
    .with_port(22)                                          // default: 22
    .with_host_key_policy(HostKeyPolicy::Verify)            // default: Verify
    .with_timeout(std::time::Duration::from_secs(10))       // default: 10s
    .with_inactivity_timeout(None)                          // default: unlimited
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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10e)*: `TrustFirstUse` is fail-closed — if
> persisting the learned key to `~/.ssh/known_hosts` fails (e.g. file not
> writable), the connection is **rejected**. This is intentional but may
> surprise users who expect TOFU to "just work". The error is logged at
> `error` level with an actionable message ("check that ~/.ssh/known_hosts
> is writable"). **Fixed**: Added doc comment on `TrustFirstUse` explaining fail-closed behavior.

### Connection status

```rust
// Only available on the SshTransport directly, not through HostHandle.
// You must retain a reference to the SshTransport if you need this.
if ssh.is_closed() {
    println!("SSH connection lost");
}
```

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10d)*: `is_closed()` is only accessible on
> `SshTransport`, not through `HostHandle` or `TransportKind`. There is no
> transport-agnostic way to check connection health. `LocalTransport` has no
> equivalent (always "connected"). Consider adding
> `TransportKind::is_closed()` that returns `false` for Local/Mock. **Fixed**: Added `TransportKind::is_healthy() -> bool` (Local/Mock=true, SSH=!is_closed()).

### Characteristics

- **Exec timeout**: `SshConfig::timeout` governs connection, authentication, and
  each `exec()` call (channel open + exec + output collection are all inside
  one `tokio::time::timeout` boundary).
- **Inactivity timeout**: `SshConfig::inactivity_timeout` governs how long an
  SSH connection may sit idle before the client closes it. Default: `None`
  because monitor/stream use cases are expected to be long-lived.
- **Keepalive**: `SshConfig::keepalive_interval` sends SSH keepalives.
  `None` disables. Local/Mock have no equivalent (not applicable).
- **Concurrency**: The SSH handle mutex is held only during
  `channel_open_session()`, not for the full command lifetime. Multiple
  concurrent `exec()` calls on the same connection are safe.
- **Authentication**: ssh-agent (default) or explicit key file via `identity-file`
  (DC26). Password auth is not supported.
  Error messages are actionable (OC3): "is SSH_AUTH_SOCK set?",
  "Add a key with: ssh-add ~/.ssh/id_ed25519",
  "If the key is passphrase-protected, load it into ssh-agent instead."

---

### 2a. SshConfig URI Parsing and Connect

`SshConfig` can be constructed from an `ssh://` URI string, providing a single
entry point for host configuration. This is the recommended way to configure
connections in user-facing code.

#### URI format

```
ssh://[user[;param=value;...]@]host[:port][/socket-path][?param=value&...]
```

Parameters can appear in two locations:
- **Nassh-style** (userinfo): `ssh://deploy;timeout=30@prod`
- **Query params**: `ssh://deploy@prod?timeout=30`
- **Mixed**: `ssh://deploy;timeout=30@prod?host-key-policy=tofu`

#### Parse from URI

```rust
use motlie_tmux::SshConfig;

// Basic
let cfg = SshConfig::parse("ssh://deploy@prod-server")?;

// With port and parameters
let cfg = SshConfig::parse(
    "ssh://deploy@prod:2222?host-key-policy=tofu&timeout=30&inactivity-timeout=120"
)?;

// Nassh-style parameters in userinfo
let cfg = SshConfig::parse("ssh://deploy;host-key-policy=tofu;timeout=30@prod")?;

// Localhost (no user required)
let cfg = SshConfig::parse("ssh://localhost")?;

// IPv6
let cfg = SshConfig::parse("ssh://deploy@[::1]:2222")?;

// With tmux socket
let cfg = SshConfig::parse("ssh://deploy@host/tmp/tmux-custom.sock")?;   // socket path
let cfg = SshConfig::parse("ssh://deploy;socket-name=myserver@host")?;    // socket name

// With identity file (query-only, DC26)
let cfg = SshConfig::parse("ssh://deploy@prod?identity-file=/keys/deploy")?;

// Identity file with other params (nassh params + query identity-file)
let cfg = SshConfig::parse("ssh://deploy;timeout=30@prod?identity-file=/keys/deploy")?;

// FromStr also works
let cfg: SshConfig = "ssh://deploy@prod:2222".parse()?;
```

#### Available parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `host-key-policy` | `verify`, `tofu`, `insecure` | `verify` | SSH host key verification policy |
| `timeout` | integer seconds (> 0) | `10` | Per-command execution timeout |
| `inactivity-timeout` | integer seconds (0 = unlimited) | unlimited | SSH connection idle timeout for long-lived shells/monitors |
| `keepalive` | integer seconds (0 = off) | `30` | SSH keepalive interval |
| `socket-name` | `[A-Za-z0-9._-]+` | none | Tmux socket name (`tmux -L`) |
| `identity-file` | absolute path | none | SSH private key file (query-only, DC26) |

Socket path is specified as the URI path component (`/path/to/socket`), not as a
parameter. Socket path and `socket-name` are mutually exclusive.

`identity-file` is **query-only** — it cannot appear in nassh-style userinfo params.
When set, authentication uses the specified key file instead of ssh-agent.

#### Validation rules

- `user`, `host`, `port` are canonical URI components — they cannot appear as parameters
- Duplicate parameter keys are rejected (within the same location or across locations)
- Unknown parameter names are rejected (fail-fast)
- `timeout` must be > 0
- `identity-file` must be an absolute path and can only appear as a query param
- `with_identity_file()` returns `Err` if an identity file is already set

```rust
// These all return Err:
SshConfig::parse("ssh://user@host?port=22");           // canonical component
SshConfig::parse("ssh://user;timeout=10@host?timeout=20"); // cross-location duplicate
SshConfig::parse("ssh://user@host?unknown=value");     // unknown parameter
SshConfig::parse("ssh://user;identity-file=/key@host"); // identity-file in userinfo
SshConfig::parse("ssh://user@host?identity-file=rel/path"); // relative path

// Duplicate identity-file across parse + builder:
SshConfig::parse("ssh://user@host?identity-file=/a")?.with_identity_file("/b")?; // Err
```

#### Render to URI

```rust
use motlie_tmux::SshConfig;

let cfg = SshConfig::new("prod", "deploy")
    .with_port(2222)
    .with_host_key_policy(motlie_tmux::HostKeyPolicy::TrustFirstUse);

// Display and to_uri_string() produce canonical form
assert_eq!(cfg.to_string(), "ssh://deploy;host-key-policy=tofu@prod:2222");

// Round-trip guarantee: parse(cfg.to_string()) == cfg
// (holds for URI-safe user/host — no ;@?&=#[] characters)
let reparsed: SshConfig = cfg.to_string().parse()?;
assert_eq!(cfg, reparsed);
```

When user is non-empty, non-default parameters render as nassh-style userinfo params.
When user is empty (e.g. localhost), parameters render as query params.

`parse()` rejects URI-reserved characters (`;@?&=#[]`) in user, host, and
parameter values. Configs from `parse()` always round-trip safely.
Builder-constructed configs with reserved characters will produce URIs that
may not re-parse — use DNS-safe hostnames and POSIX usernames.

#### Connect from URI

```rust
use motlie_tmux::SshConfig;

// Localhost — automatically uses LocalTransport (no SSH)
let host = SshConfig::parse("ssh://localhost")?.connect().await?;
let sessions = host.list_sessions().await?;

// Remote — requires user, uses SshTransport
let host = SshConfig::parse("ssh://deploy@prod-server")?.connect().await?;

// With separate exec and inactivity timeouts
let host = SshConfig::parse(
    "ssh://deploy;host-key-policy=tofu@prod?timeout=30&inactivity-timeout=120"
)?
    .connect()
    .await?;

// With identity file — key-file auth, no agent needed (DC26)
let host = SshConfig::parse("ssh://deploy@prod?identity-file=/etc/deploy/id_ed25519")?
    .connect()
    .await?;

// Builder API with identity file
let host = SshConfig::new("prod", "deploy")
    .with_identity_file("/etc/deploy/id_ed25519")?
    .connect()
    .await?;
```

Transport selection is automatic:
- `localhost`, `127.0.0.1`, `::1` → `LocalTransport` (no SSH handshake)
- All other hosts → `SshTransport` via SSH (user required)

> See [`examples/uri_connect.rs`](../examples/uri_connect.rs) for a runnable version.

The `connect()` method consumes `self`. Socket configuration is propagated to
the `HostHandle` for tmux commands (`-L` or `-S` flags).

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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10b)*: Pattern matching iterates the internal
> HashMap, so if a command matches multiple registered patterns, which one
> wins is **non-deterministic** (HashMap iteration order). Use
> specific-enough patterns to avoid ambiguity. For example, prefer
> `"list-sessions"` over `"list"` if you also have `"list-panes"` registered. **Fixed**: Switched from HashMap to Vec for deterministic insertion-order matching.

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
    let target = host.create_session("test", &Default::default()).await.unwrap();
    assert_eq!(target.session_name(), "test");
}
```

### Characteristics

- **No timeout**: Responses are instant. No timeout configuration.
- **Error injection**: Use `with_error(pattern, message)` to make `exec()`
  return `Err` for commands matching the pattern. Error patterns are checked
  before response patterns. Without `with_error()`, `exec()` always returns
  `Ok(response)`.
- **Deterministic matching**: Patterns are matched in insertion order (first
  match wins), not hash order.
- **Shell channel**: `open_shell()` returns a `MockShellChannel` with empty
  data that immediately yields `Eof` on read.

```rust
use motlie_tmux::{HostHandle, transport::{MockTransport, TransportKind}};

let mock = MockTransport::new()
    .with_error("kill-session", "session not found")
    .with_response("list-sessions", "build\t$0\t0\t0\t1\t\n");
let host = HostHandle::new(TransportKind::Mock(mock), None);
let target = host.session("build").await?.unwrap();
let err = target.kill().await.unwrap_err();
assert!(err.to_string().contains("session not found"));
```

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10a)*: Added
> `MockTransport::with_error(pattern, message)` for unit-testing error
> handling paths in discovery, capture, and control code.

---

## 4. Transport Comparison

| Aspect | Local | SSH | Mock |
|--------|-------|-----|------|
| Timeout | 10s default, configurable | 10s default, configurable | None |
| Error includes stderr | Yes | Yes | Via `with_error()` (message only, no stderr) |
| `is_closed()` / `is_healthy()` | `is_healthy()` always `true` | `is_healthy()` delegates to `!is_closed()` | `is_healthy()` always `true` |
| Keepalive | N/A | Configurable, default 30s | N/A |
| `open_shell(cols, rows)` | Spawns `sh` (ignores cols/rows) | PTY `xterm` at caller-specified cols×rows | Empty data, immediate Eof (ignores cols/rows) |
| Auth | N/A | ssh-agent or identity-file (DC26) | N/A |

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10c)*: `SshTransport::open_shell()` requests a
> PTY at fixed 80x24. There is no API to specify dimensions. This may matter
> for applications that need a specific terminal size for the remote shell. **Fixed**: `open_shell()` now takes `cols: u32, rows: u32` parameters.

```rust
use motlie_tmux::transport::TransportKind;

// Assume `ssh` is a connected SshTransport (see SSH transport section above).
let transport = TransportKind::Ssh(ssh);
if transport.is_healthy() {
    let shell = transport.open_shell(120, 40).await?;
    drop(shell);
}
```

---

# Part II — Common API (transport-agnostic)

Everything below uses the same call patterns across Local, SSH, and Mock
transports. The transport is selected once at `HostHandle` construction.
Behavioral differences (e.g. Mock errors only via `with_error()`, SSH shell channels use PTY)
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

### Hostname From Tmux

```rust
let tmux_host = host.tmux_hostname().await?;
```

`tmux_hostname()` returns the value tmux reports for `#{host}` on the
configured transport/socket. It runs `start-server ; display-message -p
'#{host}'`, so it starts the tmux server if one is not already running.

---

## 6. Session Lifecycle

### Create

```rust
// Minimal — detached session with default shell
let target = host.create_session("build", &Default::default()).await?;
// Returns a Target at session level

// With named window and startup command
let opts = CreateSessionOptions {
    window_name: Some("editor".to_string()),
    command: Some("vim".to_string()),
    ..Default::default()
};
let target = host.create_session("dev", &opts).await?;

// With window size and history limit (DC22)
let opts = CreateSessionOptions {
    width: Some(200),
    height: Some(50),
    history_limit: Some(50000),
    ..Default::default()
};
let target = host.create_session("automation", &opts).await?;
// Sets -x 200 -y 50 on new-session, then set-option history-limit 50000
// on both the session (future panes) and initial pane (tmux 3.1+)

// With variables visible to the initial shell or command
let opts = CreateSessionOptions {
    initial_environment: vec![
        SessionEnvVar::new("MOTLIE", "enabled")?,
        SessionEnvVar::new("BUILD_ID", "42")?,
    ],
    ..Default::default()
};
let target = host.create_session("with-env", &opts).await?;
// Emits tmux new-session -e MOTLIE=enabled -e BUILD_ID=42 before the command,
// so the first pane process inherits those values.
// Duplicate names are emitted in Vec order; tmux applies the last value.
```

### Kill

```rust
target.kill().await?;
// Works at any level: kills session, window, or pane depending on target level.
```

> See [`examples/session_lifecycle.rs`](../examples/session_lifecycle.rs) for the full create → rename → kill flow.

### Rename

```rust
// rename() returns a new Target with the updated address.
// For session rename this is critical — the old handle has a stale name.
let target = target.rename("new_name").await?;
target.kill().await?; // uses the renamed handle
```

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10h)*: `rename()` now returns
> `Result<Target>` with the updated address. The impact by level:
>
> - **Session rename** — correctness-significant. `target_string()` uses the
>   session name, so callers **must** use the returned handle.
> - **Window rename** — metadata drift only. `target_string()` uses
>   `session:index` (not the window name), so commands continue to work.
>   However, the cached `WindowInfo.name` becomes stale — callers displaying
>   window names should re-query.
>
> Consider having `rename()` return a new `Target` with the updated address,
> or making `Target` internally mutable. **Fixed**: `Target::rename()` now returns `Result<Target>` with updated address.

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10i)*: `rename()` at pane level returns
> `Err("cannot rename a pane")`. This is a tmux limitation — tmux has no
> `rename-pane` command. The error is clear, but it would be better to make
> this a compile-time restriction or at least document the asymmetry in the
> `Target::rename()` doc comment. **Fixed**: Doc comment on `rename()` documents pane-level Err and the asymmetry.

---

## 7. Discovery

All discovery methods return empty `Vec`s (not errors) when there is no
tmux server running or no entities exist.

### List sessions

```rust
let sessions = host.list_sessions().await?;
for s in &sessions {
    println!("{} (id={}, windows={}, attached_clients={}, active={})",
        s.name, s.id, s.window_count, s.attached_count, s.is_attached());
}
```

> See [`examples/list_sessions.rs`](../examples/list_sessions.rs) for a runnable version.

### List sessions with recency data

`list_sessions()` returns rows whose `SessionInfo.activity` is the
**aggregated** activity timestamp: `max(session_activity, max(window_activity
across the session's windows))`. The chained `list-windows -a` query and
aggregation are internal to the lib; callers see a single field. See issue
#237 for why aggregation is necessary (tmux's `session_activity` only tracks
attached-client input, not program output).

```rust
let sessions = host.list_sessions().await?;
let now = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or(0);
for session in &sessions {
    let active_secs = now.saturating_sub(session.activity);
    let age_secs = now.saturating_sub(session.created);
    println!(
        "{} active={}s age={}s",
        session.name, active_secs, age_secs
    );
}
```

The lib does not ship a host-clock probe. There is no portable,
side-effect-free way to read the host's wall clock across tmux versions
(`#{epoch}` is tmux 3.7+; `run-shell 'date +%s'` corrupts the operator's
attached pane on tmux ≤ 3.4). Consumers that need true skew-free recency
should keep an observer-relative tracker — see the mmux selector for an
example. Under the typical NTP-synced clock assumption, comparing
`session.activity` against an operator-side `time(NULL)` is correct to
within sub-second drift.

### Find a session by name

```rust
match host.session("build").await? {
    Some(t) => println!("found: {}", t.target_string()),
    None    => println!("not found"),
}
```

### Find a session by stable id

```rust
let sessions = host.list_sessions().await?;
let selected_id = sessions[0].id.as_str().to_string();

match host.session_by_id(&selected_id).await? {
    Some(target) => target.kill().await?,
    None => eprintln!("session disappeared before dispatch"),
}
```

`SessionInfo.id` is a non-empty `SessionId` parsed from tmux `#{session_id}`.
`session_by_id()` is useful when a UI stores that stable id at selection time
and later needs to dispatch against the same tmux session after a display-name
rename.

### Build a target from a fresh session row

```rust
let sessions = host.list_sessions().await?;
for session in sessions {
    let target = host.target_for_session_info(session);
    let tags = target.tags("mmux").await?.list().await?;
    println!("{} tags", tags.len());
}
```

`target_for_session_info()` does not revalidate that the session still exists.
It is intended for enrichment passes that already hold a just-fetched
`SessionInfo` and want to avoid a second session-discovery query per row.

### Find by TargetSpec

```rust
use motlie_tmux::TargetSpec;

// Session only
let t = host.target(&TargetSpec::session("build")).await?;

// Session + window (by index or name)
let t = host.target(&TargetSpec::session("build").window(0)).await?;
let t = host.target(&TargetSpec::session("build").window_name("editor")).await?;

// Session + window + pane (pane() returns Result)
let t = host.target(&TargetSpec::session("build").window(0).pane(1)?).await?;

// Reuse the spec when you want to log it or pass it around first.
let spec = TargetSpec::session("build").window(0).pane(1)?;
let t = host.target(&spec).await?;

// Parse from string
let t = host.target(&TargetSpec::parse("build:0.1")?).await?;
// Returns Option<Target> — None if the entity doesn't exist.
```

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10f)*: `TargetSpec::pane()` now returns
> `Result<Self>` instead of panicking. Missing `.window()` returns an error:
> ```rust
> // Returns Err — pane requires window context:
> let _ = TargetSpec::session("s").pane(0); // Err(...)
> // Correct usage:
> let _ = TargetSpec::session("s").window(0).pane(0)?; // Ok(spec)
> ```

### List attached clients

```rust
let clients = host.list_clients().await?;
for c in &clients {
    println!("{}x{} on '{}'", c.width, c.height, c.session);
}
// Useful for geometry/reflow detection (section 15).
```

### Host text file read

```rust
use std::path::Path;

let motd = host.read_text_file(Path::new("/etc/motd"), 64 * 1024).await?;
```

`read_text_file()` reads UTF-8 host metadata through a typed, bounded API. Local
hosts use the local filesystem path directly; SSH/mock hosts use the existing
file-transfer path and then enforce the caller's byte cap before returning the
string. This keeps selector code free of shell syntax such as `cat` and
redirection while still supporting `/etc/motd`.

### Host Event Stream

```rust
use motlie_tmux::{HostEvent, HostHandle};

let mut events = host.watch_host_events().await?;
while let Some(event) = events.recv().await {
    match event {
        HostEvent::SessionsChanged => println!("session list changed"),
        HostEvent::SessionAdded { id, name } => println!("added {name} ({id})"),
        HostEvent::SessionClosed { id, name } => println!("closed {name} ({id})"),
        HostEvent::SessionRenamed { id, old, new } => {
            println!("renamed {old} -> {new} ({id})");
        }
        HostEvent::ClientAttached { session_id } => println!("client attached to {session_id}"),
        HostEvent::ClientDetached { session_id } => println!("client detached from {session_id}"),
        HostEvent::Disconnect { reason } => eprintln!("event stream degraded: {reason}"),
    }
}
```

`watch_host_events()` currently polls `list_sessions()` once per second and
derives events by diffing stable `SessionId` keys. The tmux control-mode
notification parser remains reserved for a future event-driven watcher path.

This gives selector UIs a stable event API without forcing callers to implement
their own name-vs-id reconciliation.

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

> See [`examples/target_navigate.rs`](../examples/target_navigate.rs) for a runnable hierarchy walk
> and [`examples/target_spec.rs`](../examples/target_spec.rs) for TargetSpec resolution.

### Current PTY Attach

```rust
let target = host
    .session_by_id(&selected_id)
    .await?
    .ok_or_else(|| motlie_tmux::Error::NotFound("selected session disappeared".into()))?;

let exit = target.attach_current_pty().await?;
std::process::exit(exit.shell_status());
```

`attach_current_pty()` is session-target only. Local targets spawn
`tmux attach-session -t <target>` with inherited stdio. SSH targets spawn an
interactive `ssh -t ... tmux attach-session -t <target>` command using the
`SshConfig` already owned by the `HostHandle`. The child runs in its own process
group; on Unix the current terminal foreground process group is transferred to
the child and restored after `wait()`.

The Unix restore path ignores `SIGTTOU` only around `tcsetpgrp()`. This matters
because the selector parent is briefly a background process group after the
attach child exits; without that guard, shells can leave dashboard callers in a
stopped-job state after `Ctrl-b d`.

`AttachExit::shell_status()` maps normal exits to their exit code and Unix
signal exits to `128 + signal`, which is the value CLI callers should return.

### Session Status Bar Overrides

```rust
use motlie_tmux::{
    SessionStatusOverrides, StatusLeft, StatusLeftLength, StatusStyle,
};

let status = target.status().await?;
let snapshot = status.snapshot().await?;
let overrides = SessionStatusOverrides {
    style: Some(StatusStyle::new("bg=blue,fg=white")?),
    left: Some(StatusLeft::new("#{=40:session_name}")?),
    left_length: Some(StatusLeftLength::new(40)?),
};
status.apply(&overrides).await?;

// ... attach or otherwise run with temporary status overrides ...

status.restore(&snapshot).await?;

// Low-level scoped operations are also available:
status.set_style(&StatusStyle::new("bg=black,fg=white")?).await?;
let local_style = status.read_local_style().await?;
status.unset_style().await?;
```

`Target::status()` is session-target only and captures the stable session id.
The scoped operations use tmux `set-option -t <session-id> ...` /
`set-option -u`. The read paths use `show-option -q -t <session-id> ...` and
return only session-local overrides; inherited/global values return `Ok(None)`.
`SessionStatus::snapshot()` records local `status-style`, `status-left`, and
`status-left-length`; `restore()` writes present values back and unsets absent
ones. `StatusStyle` rejects empty strings; `StatusLeft` accepts empty strings
because tmux treats an empty left format as "render nothing". `StatusLeftLength`
is validated and capped at `STATUS_LEFT_LENGTH_MAX`. tmux remains responsible
for style and format syntax.

### Session Tags

Session tags store small metadata values on tmux sessions using user-defined
session options. The API is session-target only:

```rust
let session = host
    .session("build")
    .await?
    .ok_or_else(|| motlie_tmux::Error::NotFound("build not found".into()))?;

let tags = session.tags("mmux").await?;
tags.set("owner", "david").await?;
tags.set("role", "worker").await?;
tags.unset("role").await?;

assert_eq!(
    tags.read("owner").await?,
    Some("david".to_string())
);

let all = tags.list().await?;
```

For `prefix = "mmux"` and `key = "owner"`, the tmux option is stored as
`@mmux/owner`. `SessionTag` carries the namespace prefix, key, and value with
validated private fields; use `prefix()`, `key()`, `value()`, and
`option_name()` to inspect it.

Contract:
- `tags(prefix)` validates the namespace once, captures the stable session id
  and tmux command prefix, and returns a scoped `SessionTags` helper.
- `SessionTags::set(key, value)` writes one tag.
- `SessionTags::unset(key)` removes one tag from the namespace.
- `SessionTags::read(key)` returns `Ok(Some(value))` or `Ok(None)` when missing.
- `SessionTags::list()` returns every valid tag under that namespace.
- `HostHandle::list_tags_for_session_infos(prefix, sessions)` batch-lists tags
  for a session listing in one tmux command and returns an entry for every
  provided stable session id.
- Prefixes and keys must be non-empty ASCII letters, digits, `.`, `_`, or `-`.
- Values are UTF-8 strings, may be empty, must not contain control characters,
  and are capped at 2 KiB.
- These methods return `UnsupportedTarget` for window and pane targets.

## Session Environment Variables

Session environment variables are session-target only and use tmux
`set-environment` / `show-environment` under the same stable-session-id dispatch
boundary as tags. This is a post-creation API: writes update tmux's session
environment for processes tmux starts later, such as new panes or windows. They
cannot mutate shell processes already running in existing panes. Use
`CreateSessionOptions::initial_environment` for variables that must be visible to
the first pane process created by `new-session`.

```rust
let session = host
    .session("build")
    .await?
    .ok_or_else(|| motlie_tmux::Error::NotFound("build not found".into()))?;

let env = session.environment().await?;
env.set("BUILD_ID", "42").await?;

assert_eq!(
    env.read("BUILD_ID").await?,
    Some("42".to_string())
);

let all = env.list().await?;
env.unset("BUILD_ID").await?;
```

Contract:
- `environment()` captures the stable session id and tmux command prefix, and
  returns a scoped `SessionEnvironment` helper.
- `SessionEnvironment::set(name, value)` writes one variable for future
  tmux-spawned processes.
- `SessionEnvironment::unset(name)` removes one variable for future
  tmux-spawned processes.
- `SessionEnvironment::read(name)` returns `Ok(Some(value))` or `Ok(None)` when
  missing.
- `SessionEnvironment::list()` returns valid set variables and skips tmux unset
  markers such as `-NAME`.
- Names must be ASCII environment identifiers: first byte letter or `_`, then
  letters, digits, or `_`.
- Values are UTF-8 strings, may be empty, must not contain control characters,
  and are capped at 8 KiB.
- These methods return `UnsupportedTarget` for window and pane targets.

The implementation targets the stable tmux `SessionId` held by `SessionInfo`,
not the mutable display name. Tag reads use `show-option -q` so missing and empty
values remain distinct; deletion uses `set-option -u -t <session-id>
@<prefix>/<key>`. Listing uses `show-options` and filters for the requested
namespace without shell pipelines.

### Create child windows and panes

```rust
use motlie_tmux::{CreateWindowOptions, SplitDirection, SplitPaneOptions, SplitSize};

let session = host.create_session("build", &Default::default()).await?;

let logs = session
    .new_window(&CreateWindowOptions {
        name: Some("logs".into()),
        width: Some(160),
        height: Some(40),
        ..Default::default()
    })
    .await?;
assert_eq!(logs.target_string(), "build:1");

let tail = logs
    .split_pane(&SplitPaneOptions {
        direction: SplitDirection::Horizontal,
        size: Some(SplitSize::percent(40)?),
        ..Default::default()
    })
    .await?;
assert_eq!(tail.level(), motlie_tmux::TargetLevel::Pane);
```

Level semantics:
- `new_window()` is session-only; calling it on a window/pane returns `Err`
- `split_pane()` is window/pane-only; calling it on a session returns `Err`
- splitting a window target uses that window's active pane
- splitting a pane target uses that explicit pane

> See [`examples/target_navigate.rs`](../examples/target_navigate.rs) for session-created window
> setup via `Target::new_window()`, and [`examples/repl.rs`](../examples/repl.rs) for interactive
> `new-window` / `split-pane` commands.

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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10j)*: `target.pane(index)` from **session
> level** resolves via the **active window** at call time. If the active
> window changes between calls (e.g. another client switches windows), the
> same `target.pane(0)` call returns a different pane. For deterministic
> targeting, navigate explicitly: `target.window(0).pane(0)`. **Fixed**: Doc comment on `Target::pane(index)` documents active-window drift.

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

> See [`examples/send_and_capture.rs`](../examples/send_and_capture.rs) for a runnable version.

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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10k)*: At session or window level, `capture()`
> captures the **active pane** only — not all panes. Use `capture_all()` to
> capture every pane under the target. This is consistent with tmux behavior
> (`capture-pane -t session` targets the active pane) but may surprise
> callers who expect session-level capture to be exhaustive. **Fixed**: Doc comment on `Target::capture()` clarifies active-pane-only scope.

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

> See [`examples/exec_command.rs`](../examples/exec_command.rs) for a runnable version.

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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10m)*: `Target::exec()` has **its own timeout
> parameter** that governs sentinel polling. This is separate from the
> transport timeout (`LocalTransport::timeout` / `SshConfig::timeout`) which
> governs each individual tmux command. A 30s exec timeout with a 10s
> transport timeout means: each `capture-pane` poll can take up to 10s, and
> the overall sentinel wait can take up to 30s. These are independent knobs
> with no documentation linking them. Consider documenting the interaction or
> adding a note to `SshConfig::with_timeout()`. **Fixed**: Doc comments on `Target::exec()` and `SshConfig::with_timeout()` explaining dual timeouts.

### Limitations

- **Requires a shell prompt**: The pane must have a running shell that can
  accept input. `exec()` will hang (until timeout) if the pane is running
  an interactive program like `vim`, `less`, or `top`.
- **Shell detection**: Detects fish shell (uses `$status` instead of `$?`)
  via `pane_current_command`. Other exotic shells are not detected.
- **Wrap tolerance**: Works in narrow panes — the sentinel parser joins
  lines and searches in concatenated text.

### Tracked Execution (DC31)

`Target::start_exec()` separates launch from observation. Returns an `ExecHandle`
immediately while the command runs in a background task.

```rust
use std::time::Duration;
use motlie_tmux::ExecState;

// Launch — returns immediately
let handle = target.start_exec("make build", Duration::from_secs(120)).await?;

// Non-blocking status check
match handle.status() {
    ExecState::Running => println!("still running..."),
    ExecState::Completed(out) => println!("done: exit {}", out.exit_code),
    ExecState::Unknown { reason } => eprintln!("lost track: {}", reason),
}

// Or await completion
let state = handle.wait().await?;
```

`exec()` is now implemented as `start_exec()` + `wait()` — same semantics,
same sentinel mechanism, same per-pane lock serialization.

The `Unknown` state is reached when:
- The sentinel poll times out
- A connection discontinuity occurs while the command is running

---

## 11b. Socket Isolation (DC30)

`TmuxSocket::automation(scope)` creates a dedicated automation socket with a
`motlie-` prefix, isolating automation workloads from the user's default tmux
server.

```rust
use motlie_tmux::{TmuxSocket, HostHandle, TransportKind};

// Create an isolated automation socket
let socket = TmuxSocket::automation("ci-build")?;
// => TmuxSocket::Name("motlie-ci-build")

// Build host with the automation socket
let host = HostHandle::new(
    TransportKind::Local(motlie_tmux::transport::LocalTransport::new()),
    Some(socket),
);

// Ensure the server is running (idempotent)
host.ensure_socket_server().await?;

// Sessions are isolated from the default tmux server
let session = host.create_session("build", &Default::default()).await?;
```

For SSH hosts, `SshConfig::with_automation_socket(scope)` is a convenience
builder:

```rust
let host = SshConfig::new("build-host", "deploy")
    .with_automation_socket("ci")?  // errors if socket already set
    .connect()
    .await?;
```

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

> See [`examples/stream_pane.rs`](../examples/stream_pane.rs) for continuous
> streaming with overlap-aware deduplication (`tail -f` for tmux panes).
> For event-driven (push) streaming, see `--mode monitor` and [§17-20](#17-monitoring-sessions).

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

### Windowed older range

```rust
let older = target
    .sample_text(&ScrollbackQuery::LinesRange {
        older_than_lines: 80,
        count: 40,
    })
    .await?;
```

`LinesRange` captures a bounded scrollback window older than the most recent
`older_than_lines`. It maps to tmux `capture-pane -S/-E` offsets instead of
rebuilding the full scrollback buffer, making it suitable for paged TUI
back-scroll over local and SSH transports.

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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10l)*: `overlap_lines` must be **>= 2** for
> dedup to activate. With 0 or 1, `overlap_deduplicate()` returns the
> current capture unchanged with no fidelity issues — it silently does
> nothing rather than warning. This threshold is undocumented in the
> function signature. **Fixed**: Added `tracing::warn!` when overlap_lines < 2, plus doc comment.

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

> **`@claude NOTE — RESOLVED`** *(PLAN 1.10n)*: `history-limit` only affects panes
> created **after** the setting is applied. Existing panes retain their
> creation-time limit. This is a tmux limitation, not a library limitation,
> but it means `set_history_limit()` has no effect on already-running
> sessions unless you create new windows/panes afterward. **Fixed**: Doc comments on `get_history_limit()` explaining creation-time semantics.

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

## 16. Host-Level File Transfer (DC23)

`HostHandle::upload()` and `download()` transfer files and directories between
the local machine and the host. For SSH hosts this uses SFTP; for localhost it
uses filesystem copy. See [`SFTP.md`](./SFTP.md) for the full design.

### Single file

```rust
use std::path::Path;
use motlie_tmux::TransferOptions;

let opts = TransferOptions::default(); // overwrite=true, recursive=false

// Upload a local file to the remote host
host.upload(
    Path::new("/tmp/app.tar.gz"),
    Path::new("/opt/deploy/app.tar.gz"),
    &opts,
).await?;

// Download it back
host.download(
    Path::new("/opt/deploy/app.tar.gz"),
    Path::new("/tmp/restored.tar.gz"),
    &opts,
).await?;
```

### Directory (recursive)

Directory transfer requires `recursive: true`. Placement follows `cp -r`
semantics:

- **Destination does not exist** → source is copied *as* that path
- **Destination exists as a directory** → source is copied *into* it (using the
  source basename)

```rust
let opts = TransferOptions { overwrite: true, recursive: true };

// Upload ./myapp → /remote/deploy/
// If /remote/deploy/ exists, result is /remote/deploy/myapp/...
// If /remote/deploy/ does not exist, result is /remote/deploy/...
host.upload(
    Path::new("./myapp"),
    Path::new("/remote/deploy"),
    &opts,
).await?;
```

### Directory merge

When `overwrite=true` and the destination directory already exists, the transfer
merges into the existing tree: overwrite conflicting files, create missing
entries, and preserve extras. This is *not* a destructive replace.

```rust
// Remote /opt/app/ has: config.toml, old_module/
// Local  ./app/ has:    config.toml (updated), new_module/
// After upload: /opt/app/ has config.toml (updated), old_module/, new_module/
```

### Option enforcement

```rust
// overwrite=false → error if destination already exists
let opts = TransferOptions { overwrite: false, recursive: false };
// Returns Err if /remote/file.txt already exists
host.upload(Path::new("file.txt"), Path::new("/remote/file.txt"), &opts).await;

// recursive=false → error if source is a directory
host.upload(Path::new("./my_dir"), Path::new("/remote/dir"), &opts).await;
// → Err: "source is a directory but recursive is false"
```

### Symlink rejection

Any symlink encountered in the source or destination tree causes the transfer
to fail with an error. Symlinks are never followed.

### Boundary: upload/download vs Target::exec

| Method | Scope | Mechanism | Use case |
|--------|-------|-----------|----------|
| `host.upload()` / `download()` | File transfer | SFTP / filesystem copy | Deploying artifacts, fetching logs |
| `target.exec("cmd")` | Tmux pane | Send + sentinel capture | Commands that need tmux context |

Per DC19, there is no `HostHandle::exec()` bypass — all command execution stays
within the tmux framework via `Target::exec()`. For setup/teardown around
transfers, use a tmux session:

```rust
let setup = host.create_session("deploy_setup", &Default::default()).await?;
setup.exec("mkdir -p /opt/deploy", Duration::from_secs(10)).await?;
host.upload(Path::new("./app"), Path::new("/opt/deploy"), &opts).await?;
setup.exec("systemctl restart myapp", Duration::from_secs(30)).await?;
setup.kill().await?;
```

---

# Part II-b — Output Monitoring Pipeline (DC24, Track A)

The monitoring pipeline provides **event-driven** (push) access to tmux pane
output via control mode. This is fundamentally different from the poll-based
capture/sample APIs above:

| Approach | Mechanism | Latency | Multi-pane | API |
|----------|-----------|---------|------------|-----|
| **Capture/sample** (§10-13) | Poll `capture-pane` at intervals | Interval-bound | Per-target only | `target.capture()`, `target.sample_text()` |
| **Monitor pipeline** (§17-20) | Push via `tmux -C attach` | Real-time | All panes in session | `host.start_monitoring_session()`, `OutputBus` |

> See [`examples/stream_pane.rs --mode monitor`](../examples/stream_pane.rs) for
> a runnable example comparing poll vs push streaming side by side.
>
> See [`examples/monitor_pipe.rs`](../examples/monitor_pipe.rs) for the sink-consumer
> side of Track A: `Subscription::pipe()`, `StdioSink` / `CallbackSink`, and `PipeHandle`.
>
> Socket note: monitoring uses the host's configured tmux socket too. If you connect
> with `ssh://localhost?socket-name=myserver` or a socket-path URI, control-mode
> monitoring attaches to that same server rather than the default tmux socket.

---

## 17. Monitoring Sessions

### Start monitoring a session

```rust
use motlie_tmux::{HostHandle, SshConfig};

let host = SshConfig::parse("ssh://localhost")?.connect().await?;

// Start monitoring — opens tmux control mode connection
let monitor = host.start_monitoring_session("build").await?;

// The monitor is now streaming all pane output to the OutputBus.
// Deref gives access to the underlying Target:
println!("Monitoring: {}", monitor.target_string());
assert!(monitor.is_active());
```

```rust
// Monitoring also respects named sockets / socket paths from the URI.
let host = SshConfig::parse("ssh://localhost?socket-name=myserver")?
    .connect()
    .await?;
let monitor = host.start_monitoring_session("build").await?;
monitor.shutdown().await?;
```

### Monitor lifecycle

```rust
// Stop monitoring — signals the control mode connection to close
monitor.shutdown().await?;
assert!(!monitor.is_active());
```

### Monitor all sessions

```rust
// Monitor all sessions (optional regex filter)
let mut monitors = host.start_monitoring(None).await?;
println!("{} sessions monitored", monitors.session_count());

// Or filter by pattern
let re = regex::Regex::new("^build")?;
let mut monitors = host.start_monitoring(Some(&re)).await?;

// Access individual session handles
if let Some(h) = monitors.get("build") {
    println!("build is {}", if h.is_active() { "active" } else { "stopped" });
}

// Stop one session
monitors.stop_session("build").await?;

// Shutdown all
monitors.shutdown().await?;
```

### Target-level convenience

```rust
// Session-level targets can start/stop monitoring directly
let target = host.session("build").await?.unwrap();
let monitor = target.start_monitoring().await?;

// ... use monitor ...

target.stop_monitoring()?;

// Window/pane targets return an error — monitoring is session-scoped
let pane = target.pane(0).await?.unwrap();
assert!(pane.start_monitoring().await.is_err());
```

### Host-level query and control

```rust
// List all sessions being monitored on this host
let names = host.monitored_sessions();  // Vec<String>

// Stop a specific session's monitor (fire-and-forget signal)
host.stop_monitoring_session("build")?;

// Stop all monitors on this host (fire-and-forget signal)
host.stop_monitoring();
```

**Stop vs shutdown lifecycle**: the API provides two lifecycle modes:
- **Fire-and-forget signal**: `host.stop_monitoring_session()`, `host.stop_monitoring()`,
  `target.stop_monitoring()` — sends the stop signal but does not await task completion.
  Use when you only need to initiate teardown.
- **Awaited teardown**: `SessionMonitorHandle::shutdown()`, `MonitorHandle::shutdown()` —
  signals stop and awaits task completion. Use when you need completion guarantees
  (e.g. before asserting state in tests, or before dropping resources).

---

## 18. OutputBus and Subscriptions

The `OutputBus` is a shared fan-out bus. Monitors publish `TargetOutput`
events; subscribers receive them through filtered channels.

### Subscribe with source routing

```rust
use motlie_tmux::{SinkFilter, SinkEvent};

let bus = host.output_bus();

// Subscribe to all output (no filters)
let sub_all = bus.subscribe(vec![], 64)?;

// Subscribe filtered to a specific session (exact match constructor)
let sub_build = bus.subscribe(vec![SinkFilter::for_session("build")], 64)?;

// Subscribe filtered to a specific pane (exact match constructor)
let sub_pane = bus.subscribe(vec![SinkFilter::for_pane("%5")], 64)?;

// Subscribe with combined host + session
let sub_combo = bus.subscribe(
    vec![SinkFilter::for_host_session("web-1", "build")], 64
)?;

// Regex power is still available via raw fields
let filter = SinkFilter {
    session: Some("build|deploy".to_string()),
    ..Default::default()
};
let sub_regex = bus.subscribe(vec![filter], 64)?;
```

### Consume events via receiver

```rust
let mut rx = sub_all.into_receiver();

// Poll for events
while let Some(event) = rx.recv().await {
    match event {
        SinkEvent::Data(output) => {
            println!("[{}:{}] {}",
                output.host,
                output.target_string(),
                output.content);
        }
        SinkEvent::Gap { dropped, .. } => {
            eprintln!("warning: {} events dropped (backpressure)", dropped);
        }
    }
}
```

### SinkFilter routing

All filter fields are optional regex strings. A filter matches when **all**
non-None fields match (AND logic). Multiple filters in a subscription are
OR'd — an event matches if **any** filter matches.

The `pane` filter matches against both the `pane_id` (e.g. `%5`) and the
tmux target string (e.g. `build:0.1`). Control mode only provides `pane_id`,
so pane-level filtering by `pane_id` is the canonical approach.

```rust
// Exact-match constructors (preferred for common routing):
let filter = SinkFilter::for_host("web-1");
let filter = SinkFilter::for_session("build");
let filter = SinkFilter::for_pane("%5");
let filter = SinkFilter::for_host_session("web-1", "build");

// Raw regex fields for advanced routing:
let filter = SinkFilter {
    session: Some("build|deploy".to_string()),
    window: Some("^build:0$".to_string()),
    ..Default::default()
};
```

### TargetOutput fields

```rust
// Fields available on every event:
output.host              // "localhost", "web-1", etc.
output.content           // Normalized content (per monitor mode)
output.raw_content       // Some(...) when normalization changed content
output.sequence          // Per-source sequence number (monotonic within a
                         // continuous stream segment; resets on Discontinuity)
output.fidelity          // OutputFidelity (clean for control mode)
output.timestamp         // std::time::Instant of emission

// Accessors:
output.session_name()    // Session name at any source level
output.pane_id()         // Some("%5") for pane-level sources
output.source_key()      // Canonical identity: pane_id for panes, session name for sessions
output.target_string()   // Display format: "session:window.pane" (may be synthetic for monitor output)
output.degraded()        // Shorthand for fidelity.degraded
```

### Backpressure, gaps, and discontinuity

When a subscriber can't keep up, the bus drops events and tracks the count.
Before the next `Data` delivery on that subscriber, a `Gap` event is emitted:

```rust
match event {
    SinkEvent::Gap { dropped, timestamp } => {
        // `dropped` events were lost since `timestamp`
        // The subscriber is now caught up
    }
    SinkEvent::Data(output) => { /* normal delivery */ }
    SinkEvent::Discontinuity { reason } => {
        // Upstream monitor/transport continuity was broken (DC29).
        // Distinct from Gap (subscriber backpressure).
        // Sequence numbers reset after this boundary.
        // Adapters: filter_fn always forwards, JoinedStream resets
        // source tracking, HistoryHandle records as HistoryEntry::Discontinuity.
    }
}
```

The `OutputBus` broadcasts discontinuity to all subscribers regardless of
source-routing filters (system-level signal, not content):

```rust
bus.publish_discontinuity("stream interrupted: control channel lost");
```

---

## 19. JoinedStream — Multi-Pane View

`JoinedStream` consolidates output from multiple panes into a single stream
with source attribution. It detects when the source changes between
consecutive chunks.

### Create from subscription

```rust
use motlie_tmux::LabelFormat;

let sub = bus.subscribe(vec![], 64)?;

// Bracketed labels: "[localhost:build(%0)] content"
let mut stream = sub.joined(LabelFormat::Bracketed);

// Or prompt-style: "localhost:build(%0)> content"
let mut stream = sub.joined(LabelFormat::Prompt);

// Or custom formatting
let mut stream = sub.joined(LabelFormat::Custom(|source, content| {
    format!("== {} == {}", source.minimal(), content)
}));
```

### Read chunks

```rust
while let Some(chunk) = stream.next().await {
    if chunk.source_changed {
        // New source — print separator
        println!("--- {} ---", chunk.source.minimal());
    }
    print!("{}", chunk.output.content);
}
```

### StreamChunk fields

```rust
chunk.source           // SourceLabel { host, target }
chunk.output           // TargetOutput (full event data)
chunk.source_changed   // true when source differs from previous chunk
```

### SourceLabel formatting

```rust
chunk.source.short()    // "localhost:build(%5)" (host + session + pane_id)
chunk.source.minimal()  // "build(%5)" (no host prefix)
```

### Using format() for rendering

```rust
// JoinedStream::format() always applies the configured LabelFormat
let text = stream.format(&chunk);
// With Bracketed: "[localhost:build(%5)] hello world"
// With Prompt:    "localhost:build(%5)> hello world"
```

### Multi-pane joined output (real example)

Runnable example: [`examples/joined_demo.rs`](../examples/joined_demo.rs) — creates a
2-pane session, sends `ps aux | head -5` to pane 0 and `ls -la /tmp | head -5` to pane 1,
then prints the interleaved JoinedStream.

**Bracketed format** (`--format bracketed`, default) — every line labeled:

```text
[localhost:joined_demo_12345(%5)] ps aux | head -5
[localhost:joined_demo_12345(%5)] USER         PID %CPU %MEM  ...
[localhost:joined_demo_12345(%5)] root           1  0.0  0.0  ...
[localhost:joined_demo_12345(%6)] ls -la /tmp | head -5
[localhost:joined_demo_12345(%6)] total 218368
[localhost:joined_demo_12345(%6)] drwxrwxrwt 35 root   root   ...
```

**Prompt format** (`--format prompt`) — prompt-style labels:

```text
localhost:joined_demo_12345(%5)> ps aux | head -5
localhost:joined_demo_12345(%5)> USER         PID %CPU %MEM  ...
localhost:joined_demo_12345(%6)> ls -la /tmp | head -5
localhost:joined_demo_12345(%6)> total 218368
```

**Separator format** (`--format separator`) — header only on source transitions,
using `source_changed` to insert `--- pane ---` dividers:

```text
--- localhost:joined_demo_12345(%5) ---
ps aux | head -5
USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root           1  0.0  0.0  23580 14240 ?        Ss   Mar10   2:30 /sbin/init
--- localhost:joined_demo_12345(%6) ---
ls -la /tmp | head -5
total 218368
drwxrwxrwt 35 root   root     118784 Mar 19 21:42 .
```

The key API pattern for per-line labeled rendering:

```rust
while let Some(chunk) = stream.next().await {
    let clean = strip_ansi(&chunk.output.content);
    for line in clean.lines() {
        if !line.trim().is_empty() {
            println!("[{}] {}", chunk.source.short(), line);
        }
    }
}
```

---

## 20. Sink Pipeline

Sinks are terminal consumers driven by subscriptions. The pipeline is:
`OutputBus` → `Subscription` → adapter (`.pipe()`, `.joined()`, `.into_receiver()`) → consumer.

Runnable example: [`examples/monitor_pipe.rs`](../examples/monitor_pipe.rs)

### Pipe to stdio

```rust
use motlie_tmux::{SinkKind, StdioSink, StdioFormat};

let sub = bus.subscribe(vec![], 64)?;

// Pipe events to stdout with source prefix labels
let sink = SinkKind::Stdio(StdioSink::new(StdioFormat::Prefixed));
let pipe = sub.pipe(sink);  // Spawns async task, returns PipeHandle

// PipeHandle combines subscription id + task handle:
pipe.id();            // SinkId — for bus.unsubscribe(id)
pipe.join().await?;   // Await task completion (after bus shutdown/unsubscribe)
```

### Pipe to callback

```rust
use motlie_tmux::{SinkKind, CallbackSink, SinkEvent};
use std::sync::Arc;
use std::any::Any;

let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
let state: Arc<dyn Any + Send + Sync> = events.clone();

let sink = SinkKind::Callback(CallbackSink {
    name: "collector".into(),
    state,
    on_output: |state, event| {
        if let SinkEvent::Data(output) = event {
            let events = state.downcast_ref::<std::sync::Mutex<Vec<String>>>().unwrap();
            events.lock().unwrap().push(output.content);
        }
        Ok(())
    },
    on_flush: None,
});

let sub = bus.subscribe(vec![], 64)?;
let pipe = sub.pipe(sink);
```

### StdioFormat options

| Format | Output |
|--------|--------|
| `Raw` | Content only, no labels |
| `Prefixed` | `[host] source_key \| content` (uses canonical identity) |
| `Json` | JSON object per event |

---

# Part II-c — External-Agent Substrate (Track B)

Track B provides the building blocks for external LLM/classifier loops: predicate
filtering, rolling transcript/history, and multi-host fleet coordination. These
compose on top of Track A's monitoring pipeline (OutputBus, Subscription, JoinedStream).

## 21. Predicate Filtering — filter_fn

`Subscription::filter_fn()` wraps a subscription with a consumer-owned predicate.
Only `Data` events matching the predicate are forwarded; `Gap` events always pass
through. The result is a new `Subscription` that composes with all other adapters.

```rust
let bus = host.output_bus();
let sub = bus.subscribe(vec![SinkFilter::for_session("build")], 64)?;

// Only forward events containing "ERROR"
let filtered = sub.filter_fn(|output| output.content.contains("ERROR"));

// Compose with pipe, joined, or history
let pipe = filtered.pipe(SinkKind::Stdio(StdioSink::new(StdioFormat::Prefixed)));
```

The predicate signature is `fn(&TargetOutput) -> bool` — a plain function pointer,
not a closure. This keeps the API simple and avoids lifetime complications. For
stateful filtering, use `into_receiver()` and implement the loop yourself.

## 22. Rolling Transcript / History (DC28)

`Subscription::history(opts)` creates a bounded, source-labeled rolling transcript
optimized for external LLM/classifier context windows.

### Create a history handle

```rust
use motlie_tmux::{HistoryOptions, LabelFormat};

let sub = bus.subscribe(vec![], 64)?;
let history = sub.history(HistoryOptions {
    max_entries: 500,
    max_render_chars: 50_000,  // ~12k tokens
    label_format: LabelFormat::Bracketed,
    include_omission_marker: true,
});
```

### HistoryOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_entries` | `usize` | 500 | Maximum logical entries to retain |
| `max_render_chars` | `usize` | 0 (unlimited) | Character budget; oldest entries trimmed first |
| `label_format` | `LabelFormat` | `Bracketed` | Source label format for rendering |
| `include_omission_marker` | `bool` | true | Prepend `[... N earlier entries omitted ...]` on trimming |

`HostHandle::watch_session()` also accepts `SessionWatchOptions`, which wraps a
`HistoryOptions` plus queue sizing and a monitor normalization mode:

```rust
use motlie_tmux::{CaptureNormalizeMode, HistoryOptions, SessionWatchOptions};

let watch = host
    .watch_session(
        "build",
        &SessionWatchOptions {
            queue_capacity: 256,
            normalize: CaptureNormalizeMode::PlainText,
            history: HistoryOptions::default(),
        },
    )
    .await?;
```

Use `PlainText` for ratatui/text UIs that cannot render ANSI escape sequences;
use the default `Raw` mode when consumers need the original control-mode bytes.

### Snapshot for structured access

```rust
let snap = history.snapshot().await;
for entry in &snap.entries {
    match entry {
        HistoryEntry::Output { source, text, source_changed } => {
            if *source_changed {
                println!("[{}] {}", source.short(), text);
            } else {
                println!("{}", text);
            }
        }
        HistoryEntry::Gap { dropped_events } => {
            println!("[gap: {} events dropped]", dropped_events);
        }
    }
}
println!("omitted: {}, rendered_chars: {}", snap.omitted_entries, snap.rendered_chars);
```

### Prompt-ready text for LLM context

```rust
// External agent loop
loop {
    let context = history.render_text().await;
    // Pass `context` to LLM/classifier for inference
    // Route actions back via Fleet, HostHandle, or Target
    tokio::time::sleep(Duration::from_secs(5)).await;
}
```

Sample `render_text()` output:
```
[... 37 earlier entries omitted ...]
[web-1:build(%5)] cargo build --release
Compiling motlie v0.1.0
[web-1:deploy(%8)] deploying to staging...
[gap: 3 event(s) dropped]
deploy complete
```

Runnable example: [`examples/history_demo.rs`](../examples/history_demo.rs) — creates
two panes that simulate two other agent chat traces, wraps their combined stream in a
`HistoryHandle`, and prints the rolling `render_text()` context after each turn the way
an external LLM/classifier loop would consume it.

Example flow:

```rust
let monitor = host.start_monitoring_session("history_demo").await?;
let bus = host.output_bus();
let sub = bus.subscribe(vec![SinkFilter::for_session("history_demo")], 64)?;
let history = sub.history(HistoryOptions {
    max_entries: 8,
    max_render_chars: 420,
    label_format: LabelFormat::Prompt,
    include_omission_marker: true,
});

loop {
    let context = history.render_text().await;
    // Send `context` to the external model/classifier.
    // When it decides to act, route back through Fleet / HostHandle / Target.
    println!("{}", context);
    break;
}
```

### Lifecycle

```rust
// Stop monitoring → close bus → join drains and returns final snapshot
monitor.shutdown().await?;
bus.unsubscribe(history.id())?;
let snapshot = history.join().await?;
```

## 23. Fleet — Multi-Host Coordination (DC27)

`Fleet` is a programmatic registry of `HostHandle`s with a shared `OutputBus`,
workstream bindings, and convenience routing. Fleet-level actions are wrappers
over `HostHandle` / `Target` operations, not a separate action system.

### Create and register hosts

```rust
use motlie_tmux::{Fleet, SshConfig};

let mut fleet = Fleet::new();

// Hosts must be created with the fleet alias via with_alias()
let web = SshConfig::parse("ssh://deploy@web-1")?.connect().await?;
// ^ SshConfig::connect() returns HostHandle with host_alias matching the URI host
fleet.register("web-1", web)?;

let db = SshConfig::parse("ssh://admin@db-1")?.connect().await?;
fleet.register("db-1", db)?;
```

Fleet enforces that the registration alias matches `host.host_alias()`, so output
labels and routing names stay consistent in external-agent workflows.

### Shared OutputBus

Fleet injects its shared `OutputBus` into each registered host. All monitors
publish to this single bus, enabling cross-host subscriptions:

```rust
let bus = fleet.output_bus();
let sub = bus.subscribe(vec![], 64)?;  // all hosts, all sessions
let history = sub.history(HistoryOptions::default());
```

### Monitoring lifecycle

```rust
// Monitor specific sessions
fleet.start_monitoring_session("web-1", "build").await?;
fleet.start_monitoring_session("db-1", "migration").await?;

// Or monitor all sessions on a host
fleet.start_monitoring_host("web-1").await?;

// Stop one host
fleet.stop_monitoring_host("web-1")?;

// Shutdown everything
fleet.shutdown();
```

### Workstream bindings

Workstreams map stable names to host + target combinations for alias-based routing:

```rust
use motlie_tmux::TargetSpec;

fleet.bind("ci", "web-1", TargetSpec::session("build").window(0).pane(0))?;
fleet.bind("db", "db-1", TargetSpec::session("migration"))?;

// Route actions by workstream name
fleet.send_text("ci", "cargo test\n").await?;
let output = fleet.capture("db").await?;

// Resolve to a Target for direct control
let target = fleet.target("ci").await?;
target.send_keys(&KeySequence::from_str("C-c")).await?;

// Unbind when done
fleet.unbind("ci")?;
```

### Host status

```rust
match fleet.host_status("web-1") {
    Some(HostStatus::Connected) => println!("connected, not monitoring"),
    Some(HostStatus::Monitoring { sessions }) => println!("monitoring: {:?}", sessions),
    Some(HostStatus::Error(msg)) => println!("error: {}", msg),
    None => println!("not registered"),
}
```

---

## 23b. Split-Screen REPL TUI Mirror (DC32)

The first TUI delivery is a split-screen mode inside the REPL example, not a
standalone dashboard or a new `SinkKind` variant.

**Architecture**: The TUI consumer lives in `examples/repl/tui_mirror.rs` — a
binary-local module that subscribes to the existing `OutputBus` via
`Subscription::history()` and drives a `ratatui` draw loop. No terminal
dependencies are added to `libs/tmux`.

**REPL commands** (all core REPL commands work in TUI mode):

```
repl> tui on
  → enters alternate-screen split mode
  → top mirror frame (empty until a session is watched)
  → bottom REPL frame with prompt + command history
  → status bar shows MonitorHealth (active/reconnecting/failed/stopped)

# session management
monitor agents       → bind session to mirror frame
create myapp         → create a session
kill myapp           → kill a target
targets              → list sessions with target tree

# interaction
send myapp:0 ls      → send text + Enter
keys myapp {C-c}     → send raw key sequence
capture myapp 20     → show last 20 scrollback lines

tui off              → leave alternate screen, restore plain REPL
```

**Consumer data flow**:

```
HostHandle::output_bus()
    │
    ▼
OutputBus::subscribe(filters, 64)
    │
    ▼
Subscription::history(HistoryOptions { ... })
    │
    ▼
HistoryHandle::render_text()  →  top mirror frame
```

The TUI path uses a separate `Subscription` from any existing stdout monitor.
`HistoryHandle::render_text()` returns a point-in-time snapshot on each call,
so the draw loop simply polls it periodically (~150ms).

**Running the example**:

```sh
cargo run -p motlie-tmux --example repl -- ssh://localhost
# then inside the REPL:
repl> create agents
repl> tui on
# in TUI mode:
monitor agents
# type in another terminal: tmux send-keys -t agents "hello" Enter
# watch the mirror frame update
tui off
```

---

# Part III — Reference

## 24. Normalization Utilities

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

## 25. Type Quick Reference

### Core handles

| Type | Description | Clone |
|------|-------------|-------|
| `HostHandle` | Entry point — one per tmux host | Yes (Arc) |
| `Target` | Unified session/window/pane handle | No |
| `HostEventStream` | Async stream of host-level session events | No |

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
| `SessionId` | non-empty stable tmux `#{session_id}` string, e.g. `$7` |
| `SessionInfo` | name, id (`SessionId`), created, activity (aggregated `max(session_activity, max(window_activity))` per issue #237), attached_count, window_count, group |
| `SessionTags` | prefix-scoped session metadata helper returned by `Target::tags(prefix)` |
| `SessionTag` | validated prefix, key, value for one namespaced session metadata tag |
| `SessionStatus` / `SessionStatusSnapshot` / `SessionStatusOverrides` | Scoped session-local tmux status-bar API and temporary override state |
| `StatusStyle` / `StatusLeft` / `StatusLeftLength` | Validated tmux status-bar override values used by `SessionStatus` |
| `WindowInfo` | session_name, index, name, active, pane_count |
| `PaneInfo` | address, current_command, pid, width, height, active |
| `ClientInfo` | width, height, session |
| `HostEvent` | SessionsChanged, SessionAdded, SessionClosed, SessionRenamed, ClientAttached, ClientDetached, Disconnect |

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
| `ScrollbackQuery` | Enum: LastLines(n), Until { pattern, max_lines }, LastLinesUntil { lines, stop_pattern }, LinesRange { older_than_lines, count } |
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
| `SshTransport` | russh 0.46, ssh-agent or key-file auth (DC26); `connect()`, `is_closed()` |
| `SshConfig` | host, port, user, host_key_policy, timeout, inactivity_timeout, keepalive_interval, socket; `parse()`, `to_uri_string()`, `connect()`, `Display`/`FromStr` |
| `MockTransport` | Canned responses; `with_response()`, `with_default()`, `with_file()`, `with_dir()`, `with_shell_sequence()` |
| `HostKeyPolicy` | Enum: Verify (default), TrustFirstUse, Insecure |
| `TmuxSocket` | Enum: Name(String), Path(String) |
| `TransferOptions` | overwrite (bool, default true), recursive (bool, default false) |
| `MockFsEntry` | Enum: File(Vec<u8>), Dir — in-memory mock filesystem entries |

### Monitoring pipeline (DC24, Track A)

| Type | Description |
|------|-------------|
| `SessionMonitorHandle` | Handle to one monitored session — `shutdown()`, `is_active()`, `health()`, `Deref<Target>` |
| `MonitorHandle` | Aggregate handle — `shutdown()`, `get()`, `get_by_spec()`, `stop_session()`, `active_sessions()`, `all_sessions()` |
| `MonitorHealth` | Enum: `Streaming`, `Reconnecting`, `Failed`, `Stopped` — per-session ground truth (DC29) |
| `MonitorExitReason` | Enum: `Stopped`, `ConnectionLost` — returned by `SessionMonitor::run()` |
| `OutputBus` | Fan-out bus — `subscribe()`, `publish()`, `publish_discontinuity()`, `unsubscribe()`, `shutdown()` |
| `Subscription` | Bus subscription — `.into_receiver()`, `.joined()`, `.pipe()`, `.filter_fn()`, `.history()` |
| `PipeHandle` | Lifecycle handle from `pipe()` — `id()` for bus control, `join()` for awaited teardown |
| `TargetOutput` | Output event — `source_key()` (canonical identity), `target_string()` (display), content, fidelity |
| `SinkEvent` | Enum: `Data(TargetOutput)`, `Gap { dropped, timestamp }`, `Discontinuity { reason }` |
| `SinkFilter` | Source routing — `for_session()`, `for_pane()`, `for_host()` exact constructors; raw regex fields for power |
| `SinkId` | Opaque subscription identifier |
| `SinkKind` | Enum: `Stdio(StdioSink)`, `Callback(CallbackSink)` — static dispatch |
| `CallbackSink` | User sink — name, state (`Arc<dyn Any>`), on_output, on_flush |
| `StdioSink` | Stdout writer with `StdioFormat` |
| `StdioFormat` | Enum: Raw, Prefixed, Json |
| `JoinedStream` | Multi-source view — `next()`, `format()` |
| `StreamChunk` | source (`SourceLabel`), output (`TargetOutput`), source_changed |
| `SourceLabel` | host + target — `short()`, `minimal()` |
| `LabelFormat` | Enum: Bracketed, Prompt, Custom(fn) |

### External-agent substrate (Track B)

| Type | Description |
|------|-------------|
| `HistoryHandle` | Rolling transcript handle — `snapshot()`, `render_text()`, `join()`, `id()` |
| `HistoryOptions` | Config: `max_entries`, `max_render_chars`, `label_format`, `render_mode`, `global_max_render_chars`, `include_omission_marker` |
| `HistorySnapshot` | Point-in-time snapshot — `entries`, `rendered_chars`, `omitted_entries` |
| `HistoryEntry` | Enum: `Output { source, text, source_changed }`, `Gap { dropped_events }`, `Discontinuity { reason }` |
| `Fleet` | Multi-host registry — `register()`, `host()`, `hosts()`, `output_bus()`, monitoring, workstreams, routing |
| `HostStatus` | Enum: `Connected`, `Monitoring { sessions: Vec<SessionMonitorStatus> }`, `Error(String)` |
| `SessionMonitorStatus` | Per-session status: `name`, `health: MonitorHealth` |
| `RenderMode` | Enum: `Interleaved` (default), `PerSource` — controls how `render_text()` groups entries |
| `PollHistory` | Bounded rolling text history for polling — `push_text()`, `push_text_for_source()`, `render_text()` |

### Content filtering and accumulation (DC33)

| Type | Description |
|------|-------------|
| `ContentFilter` | Trait: `filter_line()`, `is_meaningful_batch()`, `is_prompt()` — per-line TUI chrome detection |
| `RawFilter` | Strips ANSI only, keeps everything. For debug/build logs. |
| `ShellFilter` | Strips ANSI, drops empty lines. Prompt = `$`/`%`/`#`. For plain shells, CI. |
| `AgentTuiFilter` | Heuristic chrome removal (spinners, status bars, affordance hints). Parameterized by `prompt_char`. `claude_code()` = `❯`, `codex()` = `›`. |
| `FlushPolicy` | Enum: `LineCount`, `Idle`, `PromptBoundary` — controls when `SourceAccumulator` flushes |
| `SourceAccumulator` | Per-source buffered collection — `new(name, baseline, filter, policy)`, `ingest(current)`, `flush_remaining()` |

#### Content filtering usage

```rust
use motlie_tmux::{AgentTuiFilter, FlushPolicy, SourceAccumulator, PollHistory, RenderMode};
use std::time::Duration;

// Create per-source accumulators with agent-specific filters
let baseline = target.capture_all().await?;
let mut acc = SourceAccumulator::new(
    "my-session",
    baseline,
    Box::new(AgentTuiFilter::codex()),               // heuristic chrome removal
    FlushPolicy::prompt_boundary(Duration::from_secs(30), 1),  // flush on agent prompt
);

// Create a per-source rolling history
let mut history = PollHistory::new(40, 6000)
    .with_render_mode(RenderMode::PerSource);

// Polling loop:
let current = target.capture_all().await?;
if let Some(chunk) = acc.ingest(&current) {
    history.push_text_for_source("my-session", chunk);
}

// Render grouped sections:
// === my-session ===
// <coherent per-turn content>
let context = history.render_text();
```

#### Flush policies

| Policy | When to use | Example |
|--------|-------------|---------|
| `FlushPolicy::line_count(3, 10s)` | Build output, log tailing | Flush every 3 lines or 10s |
| `FlushPolicy::idle(3s, 15s)` | CI jobs, test runners | Flush when output pauses 3s |
| `FlushPolicy::prompt_boundary(30s, 1)` | Agent TUIs (Claude, Codex) | Flush when prompt appears after content |

#### Exported helper functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `clean_line` | `fn clean_line(line: &str) -> Option<String>` | Strip ANSI escape sequences and normalize a raw terminal line. Returns `None` if the line is empty/whitespace after cleaning. Used internally by all filters and `diff_new_lines`. |
| `diff_new_lines` | `fn diff_new_lines(previous: &str, current: &str, filter: &dyn ContentFilter) -> Vec<String>` | Multiset-based diff: finds lines in `current` that weren't in `previous` (preserving multiplicity), then applies the content filter. Returns filtered new lines. Core of `SourceAccumulator::ingest()`. |
| `is_tui_chrome` | `fn is_tui_chrome(trimmed: &str) -> bool` | Combined heuristic check: returns `true` if the line is TUI chrome (spinner, box-drawing, status bar, affordance hint, context indicator, or bare prompt). Used by `AgentTuiFilter` and available for custom filter implementations. |

```rust
use motlie_tmux::{clean_line, diff_new_lines, is_tui_chrome, RawFilter};

// Clean a raw terminal line
assert_eq!(clean_line("\x1b[32mhello\x1b[0m"), Some("hello".to_string()));
assert_eq!(clean_line("   \r\n"), None);

// Detect TUI chrome
assert!(is_tui_chrome("· Thinking…"));
assert!(is_tui_chrome("──────────────────"));
assert!(!is_tui_chrome("actual content here"));

// Diff two pane captures with a filter
let prev = "line1\nline2\n";
let curr = "line1\nline2\nline3\n· Thinking…\n";
let new_lines = diff_new_lines(prev, curr, &RawFilter);
assert_eq!(new_lines, vec!["line3", "· Thinking…"]); // RawFilter keeps spinners

use motlie_tmux::AgentTuiFilter;
let filtered = diff_new_lines(prev, curr, &AgentTuiFilter::codex());
assert_eq!(filtered, vec!["line3"]); // AgentTuiFilter removes spinners
```

### Shell channel (low-level, used by monitor layer — Phase 2a)

| Type | Description |
|------|-------------|
| `ShellChannelKind` | Enum: Local, Ssh, Mock — `write()`, `read()` |
| `ShellEvent` | Enum: Data(Vec<u8>), Eof |
