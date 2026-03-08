# Tmux Multi-Target Automator Design

## Status: Draft

This document describes the design for `libs/tmux`, an asynchronous, structured, multi-target
automator that monitors tmux panes over SSH and executes configurable actions in response to
output patterns. Beyond monitoring, the library provides a general-purpose remote tmux control
plane: listing sessions, capturing pane content, sending arbitrary input with proper escaping,
and managing session metadata — all across multiple hosts concurrently. The design is derived
from a working single-host prototype and specifies the decomposition, safety fixes, and
extensions required for a production library.

## Table of Contents

- [Overview](#overview)
- [Prototype Reference](#prototype-reference)
- [Architecture](#architecture)
- [Core Abstractions](#core-abstractions)
- [Module Specifications](#module-specifications)
- [Key Design Decisions](#key-design-decisions)
- [Open Concerns](#open-concerns)
- [Dependency Inventory](#dependency-inventory)
- [Implementation Phases](#implementation-phases)

---

## Overview

### Problem Statement

Interactive tmux sessions on remote hosts often reach states that require human intervention
(confirmation prompts, error recovery, continuation signals). When operating across multiple
hosts and sessions, manual monitoring does not scale.

### Solution

A library that:

1. Connects to one or more remote hosts via SSH
2. Lists and inspects tmux sessions, windows, and panes on each host
3. Captures pane content (scrollback + visible) as text on demand
4. Sends arbitrary input to panes with proper key escaping (Enter, C-c, etc.)
5. Manages session metadata (rename sessions/windows)
6. Attaches output pipes for continuous monitoring
7. Evaluates configurable trigger rules against pane output
8. Executes actions (send-keys, notify, log) when rules match
9. Reconnects automatically on failure

### Scope

- **In scope**: SSH transport, multi-host connection pool, tmux session/window/pane listing,
  pane content capture, remote input with escaping, session metadata management, pipe-based
  output monitoring, rule-based automation, structured logging, CLI binary
- **Out of scope**: Local (non-SSH) tmux, GUI, web interface, tmux session/window creation
  (may be added later as a controlled extension)

---

## Prototype Reference

The prototype is a single-file ~150-line Rust program generated via a Gemini conversation.
It demonstrates the core mechanic and is the starting point for this design. The full
prototype source, dependencies, and origin are preserved below for traceability.

**Origin**: [Gemini conversation — Rust SSH Tmux Session Interaction](https://g.co/gemini/share/e7eb11c45954)

### Prototype Cargo.toml

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
russh = "0.40"
russh-keys = "0.40"
anyhow = "1.0"
regex = "1.10"
clap = { version = "4.4", features = ["derive"] }
chrono = "0.4"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
```

### Prototype Source (`main.rs`)

```rust
use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use russh::client::{self, Handler, Msg};
use russh_keys::agent::client::AgentClient;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::{info, warn, error, instrument, span, Level};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:22")]
    host: String,
    #[arg(short, long)]
    user: String,
    #[arg(short, long)]
    filter: Option<String>,
    /// Output logs in JSON format for production monitoring
    #[arg(long)]
    json: bool,
}

struct Client;
impl Handler for Client {
    type Error = anyhow::Error;
    async fn check_server_key(self, _key: &russh_keys::key::PublicKey) -> Result<(Self, bool)> {
        Ok((self, true))
    }
}

struct TmuxAutomator {
    session: client::Handle<Client>,
    filter: Option<String>,
}

impl TmuxAutomator {
    #[instrument(skip(self), fields(host = %self.filter.as_deref().unwrap_or("all")))]
    async fn run(&mut self) -> Result<()> {
        info!("Starting introspection phase...");

        let mut channel = self.session.channel_open_session().await?;
        channel.exec(true, "tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}'").await?;

        let mut raw_list = String::new();
        while let Some(msg) = channel.wait().await {
            if let Msg::Data { ref data } = msg {
                raw_list.push_str(&String::from_utf8_lossy(data));
            }
        }

        let filter_re = self.filter.as_ref().map(|f| Regex::new(f)).transpose()?;
        let target_panes: Vec<String> = raw_list.lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .filter(|s| filter_re.as_ref().map_or(true, |re| re.is_match(s)))
            .collect();

        if target_panes.is_empty() {
            warn!("Introspection complete: No matching panes found.");
            return Ok(());
        }

        info!(pane_count = target_panes.len(), "Setting up multiplexed pipes");

        let mut shell = self.session.channel_open_session().await?;
        shell.request_pty(true, "xterm", 80, 24, 0, 0, &[]).await?;
        shell.request_shell(true).await?;

        let mut setup = String::from("set -m; ");
        for pane in &target_panes {
            let fifo = format!("/tmp/tmux_pipe_{}", pane.replace([':', '.'], "_"));
            setup.push_str(&format!(
                "[ -p {0} ] || mkfifo {0}; tmux pipe-pane -t {1} -o 'cat > {0}' & ",
                fifo, pane
            ));
        }
        setup.push_str("tail -qf /tmp/tmux_pipe_* | awk '{ print FILENAME \": \" $0 }'\n");
        shell.data(setup.as_bytes()).await?;

        let re_line = Regex::new(r"tmux_pipe_(?P<pane>[^:]+): (?P<content>.*)")?;

        // Main Processing Loop
        while let Some(msg) = shell.wait().await {
            match msg {
                Msg::Data { ref data } => {
                    let chunk = String::from_utf8_lossy(data);
                    for line in chunk.lines() {
                        if let Some(cap) = re_line.captures(line) {
                            let pane_id = &cap["pane"];
                            let content = &cap["content"];

                            if content.contains('>') {
                                info!(pane = %pane_id, "Trigger detected, sending 'continue'");
                                let tmux_id = pane_id.replace('_', ":").replace("::", ":");
                                let cmd = format!(
                                    "tmux send-keys -t {} 'continue' Enter\n", tmux_id
                                );
                                shell.data(cmd.as_bytes()).await?;
                            }
                        }
                    }
                }
                Msg::Eof => {
                    warn!("Remote end sent EOF");
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize Tracing Subscriber
    if args.json {
        tracing_subscriber::fmt().json().init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .init();
    }

    let mut retry_delay = Duration::from_secs(2);

    loop {
        let span = span!(Level::INFO, "connection_attempt", host = %args.host);
        let _enter = span.enter();

        let result = async {
            let config = Arc::new(client::Config {
                connection_timeout: Some(Duration::from_secs(10)),
                heartbeat_interval: Some(Duration::from_secs(30)),
                ..Default::default()
            });

            info!("Connecting to host...");
            let mut session = client::connect(config, args.host.clone(), Client {}).await
                .context("Failed to establish TCP/SSH transport")?;

            let mut agent = AgentClient::connect_env().await
                .context("Could not find local SSH agent")?;
            let keys = agent.request_identities().await?;

            let mut auth = false;
            for key in keys {
                if session.authenticate_pubkey(&args.user, key).await? {
                    auth = true;
                    break;
                }
            }

            if !auth {
                return Err(anyhow::anyhow!("Authentication rejected by server"));
            }
            info!("Authentication successful");

            let mut automator = TmuxAutomator {
                session,
                filter: args.filter.clone(),
            };
            automator.run().await
        }
        .await;

        if let Err(e) = result {
            error!(error = %e, "Session failure");
            tokio::time::sleep(retry_delay).await;
            retry_delay = std::cmp::min(retry_delay * 2, Duration::from_secs(60));
        } else {
            info!("Session closed gracefully.");
            break;
        }
    }

    Ok(())
}
```

### What the Prototype Does

```
main() loop with reconnect
  → SSH connect via russh + ssh-agent auth
  → Channel 1: "tmux list-panes -a -F ..." → parse + regex filter → target pane list
  → Channel 2 (PTY): for each pane, mkfifo + tmux pipe-pane → tail -qf | awk
  → Event loop: parse muxed stream, if line contains '>' → tmux send-keys 'continue'
```

### Prototype Gaps (Must Fix)

| ID | Gap | Severity | Notes |
|----|-----|----------|-------|
| P1 | Host key verification disabled (`check_server_key` always `true`) | **High** | MITM vulnerability; must verify against known_hosts |
| P2 | Pane ID encoding is lossy | **High** | `_` in session names breaks roundtrip through FIFO naming; e.g. session `my_app` pane `0.0` → FIFO `tmux_pipe_my_app_0_0` → reconstructed as `my:app:0.0` (wrong) |
| P3 | Hardcoded trigger (`contains('>')`) and response (`send-keys 'continue'`) | **High** | False-positives on shell prompts, `>` in log output; not configurable |
| P4 | No FIFO cleanup or pipe-pane detach on exit | **Medium** | Leaks `/tmp/tmux_pipe_*` files; leaves `pipe-pane` attached after crash |
| P5 | No signal handling (SIGINT/SIGTERM) | **Medium** | Cannot gracefully shut down |
| P6 | Single SSH channel for all shell work | **Medium** | FIFO setup, tail, awk, and send-keys all share one PTY; fragile parsing, no error isolation |
| P7 | Monolithic `TmuxAutomator` struct | **Medium** | Conflates connection, discovery, pipe setup, and event loop; untestable |
| P8 | Single-host only | **Medium** | Title says "multi-target" but code connects to one host |
| P9 | No error recovery in event loop | **Low** | Malformed lines or failed send-keys silently ignored |
| P10 | No timeout on introspection channel | **Low** | `tmux list-panes` hang blocks forever |

---

## Architecture

### Component Structure

```
libs/tmux/
├── src/
│   ├── lib.rs              # Public API re-exports
│   ├── config.rs           # TmuxAutomatorConfig, TriggerRule, Action
│   ├── connection.rs       # SshConnection: connect, auth, reconnect logic
│   ├── host.rs             # HostHandle: per-host facade for all tmux operations
│   ├── fleet.rs            # Fleet: multi-host pool, dispatch, aggregate status
│   ├── discovery.rs        # Session/window/pane listing, filter, PaneAddress type
│   ├── capture.rs          # Pane content capture (capture-pane) and scrollback dump
│   ├── control.rs          # Remote input: send-keys with escaping, session rename
│   ├── pipe.rs             # PipeManager: FIFO lifecycle, pipe-pane attach/detach
│   ├── monitor.rs          # OutputMonitor: stream parsing, rule evaluation, dispatch
│   ├── keys.rs             # Key literal and escape helpers (Enter, C-c, Tab, etc.)
│   └── types.rs            # PaneAddress, SessionInfo, WindowInfo, PaneInfo, shared types
├── docs/
│   └── DESIGN.md           # This document
└── Cargo.toml

bins/tmux-automator/         # (optional) CLI binary wrapping the library
├── main.rs
└── Cargo.toml
```

### Data Flow

```
  ┌───────────────────────────────────────────────────────────────┐
  │                          Fleet                                │
  │  Manages N hosts, routes commands, aggregates status          │
  └──────┬──────────────────┬──────────────────┬──────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ HostHandle  │   │ HostHandle  │   │ HostHandle  │  × N hosts
  │ (host A)    │   │ (host B)    │   │ (host C)    │
  └──────┬──────┘   └─────────────┘   └─────────────┘
         │
         │  One SSH connection, multiple capabilities:
         │
         ├── Discovery ──── list sessions, windows, panes
         │
         ├── Capture ────── dump pane content as text
         │
         ├── Control ────── send-keys, rename session/window
         │
         ├── PipeManager ── attach/detach output pipes
         │
         └── Monitor ────── stream + rule evaluation → action dispatch
```

**Two usage modes coexist on the same `HostHandle`**:

1. **On-demand operations**: List sessions, capture pane, send input, rename — called
   directly by the consumer (CLI command, MCP tool, API call). These open short-lived
   SSH exec channels and return immediately.

2. **Continuous monitoring**: Pipe setup + event loop — long-running, spawned as a
   background task. Uses dedicated PTY channel. Rule-triggered actions dispatch through
   the same `Control` module as on-demand operations.

### Concurrency Model

Each host is represented by a `HostHandle` with a single `SshConnection`. SSH multiplexes
channels over one TCP connection, so multiple operations can be in-flight concurrently.

**On-demand operations** (list, capture, send-keys, rename):
- Each call opens a short-lived SSH exec channel
- Multiple on-demand calls can run concurrently on the same host
- No long-lived state required

**Continuous monitoring** (when started):
- **Pipe setup**: Dedicated SSH exec channel (not the monitor channel)
- **Monitoring**: Dedicated PTY channel running `tail -qf`
- **Action dispatch**: Separate SSH exec channels per triggered action
  (see [DC4](#dc4-action-dispatch-channel-strategy))
- Runs as a `tokio::spawn` task; does not block on-demand operations

**Multi-host**: `Fleet` spawns per-host monitoring tasks independently. A connection failure
on host A does not affect host B. On-demand operations are dispatched to the named host's
`HostHandle` directly.

---

## Core Abstractions

This section defines the primary abstractions that structure the library. These are the
types a consumer interacts with. Module specifications (next section) describe internals.

### `Fleet` — Multi-Host Pool

The top-level entry point. Manages connections to multiple hosts and provides a uniform
interface for dispatching operations to any host by name or alias.

```rust
pub struct Fleet { /* HashMap<String, HostHandle>, shutdown_tx, config */ }

impl Fleet {
    /// Create a fleet from config. Does not connect yet.
    pub fn new(config: TmuxAutomatorConfig) -> Self;

    /// Connect to all configured hosts. Failures are per-host, not fatal.
    /// Returns a summary of which hosts connected and which failed.
    pub async fn connect_all(&mut self) -> Vec<HostStatus>;

    /// Get a handle to a specific host by name/alias.
    pub fn host(&self, name: &str) -> Option<&HostHandle>;

    /// Iterate over all connected hosts.
    pub fn hosts(&self) -> impl Iterator<Item = (&str, &HostHandle)>;

    /// Start monitoring on all connected hosts (spawns background tasks).
    pub async fn start_monitoring(&self, rules: &[TriggerRule]) -> Result<()>;

    /// Shutdown: stop monitoring, cleanup pipes, close connections.
    pub async fn shutdown(&self) -> Result<()>;
}
```

**Design rationale**: Callers should not need to manage individual host connections or
reason about which host a session lives on. `Fleet` provides the "I have N machines,
operate on them" abstraction. For single-host use, `Fleet` with one target works
identically — there is no separate single-host API.

### `HostHandle` — Per-Host Facade

All tmux operations on a single host go through `HostHandle`. This is the core
abstraction that unifies discovery, capture, control, and monitoring behind one type.

```rust
pub struct HostHandle { /* SshConnection, host config, pipe state */ }

impl HostHandle {
    // --- Discovery ---

    /// List all tmux sessions on this host.
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>>;

    /// List all windows in a session (by name or ID).
    pub async fn list_windows(&self, session: &str) -> Result<Vec<WindowInfo>>;

    /// List all panes, optionally filtered by regex.
    pub async fn list_panes(&self, filter: Option<&Regex>) -> Result<Vec<PaneInfo>>;

    // --- Capture ---

    /// Capture the visible content of a pane as text.
    /// Returns the current screen content (what a human would see).
    pub async fn capture_pane(&self, target: &PaneAddress) -> Result<String>;

    /// Capture pane content including scrollback history.
    /// `start` and `end` are line offsets (negative = scrollback).
    pub async fn capture_pane_with_history(
        &self,
        target: &PaneAddress,
        start: i32,
        end: i32,
    ) -> Result<String>;

    /// Capture all panes in a session, returned as a map of pane address → content.
    pub async fn capture_session(&self, session: &str) -> Result<HashMap<PaneAddress, String>>;

    // --- Control ---

    /// Send literal text to a pane. The text is escaped for tmux send-keys.
    /// Does NOT append Enter — caller must include it via KeySequence if desired.
    pub async fn send_text(&self, target: &PaneAddress, text: &str) -> Result<()>;

    /// Send a key sequence to a pane (supports special keys: Enter, C-c, Tab, etc.)
    pub async fn send_keys(&self, target: &PaneAddress, keys: &KeySequence) -> Result<()>;

    /// Rename a tmux session.
    pub async fn rename_session(&self, current_name: &str, new_name: &str) -> Result<()>;

    /// Rename a window within a session.
    pub async fn rename_window(
        &self,
        session: &str,
        window_index: u32,
        new_name: &str,
    ) -> Result<()>;

    // --- Monitoring (long-running) ---

    /// Start continuous output monitoring on matching panes.
    /// Returns a handle to stop monitoring later.
    pub async fn start_monitoring(
        &self,
        filter: Option<&Regex>,
        rules: &[TriggerRule],
        shutdown: watch::Receiver<bool>,
    ) -> Result<MonitorHandle>;
}
```

### `KeySequence` — Safe Input Construction

Tmux `send-keys` has specific escaping rules. Raw strings passed to `send-keys` are
interpreted as key names unless `-l` (literal) is used, but `-l` cannot send special
keys like Enter or C-c. The library must handle this correctly so callers don't need
to know tmux escaping internals.

```rust
pub struct KeySequence {
    segments: Vec<KeySegment>,
}

enum KeySegment {
    /// Literal text — sent via `send-keys -l`. Tmux will not interpret key names.
    Literal(String),
    /// Special key — sent via `send-keys` (without -l). E.g. "Enter", "C-c", "Tab".
    Special(SpecialKey),
}

pub enum SpecialKey {
    Enter,
    Tab,
    Escape,
    Up, Down, Left, Right,
    CtrlC,      // C-c
    CtrlD,      // C-d
    CtrlZ,      // C-z
    CtrlL,      // C-l (clear)
    Space,
    BSpace,     // Backspace
    /// Arbitrary tmux key name for keys not in this enum.
    Raw(String),
}

impl KeySequence {
    /// Construct from a human-friendly string with inline escapes:
    ///   "hello{Enter}"  → Literal("hello") + Special(Enter)
    ///   "yes{Enter}"    → Literal("yes") + Special(Enter)
    ///   "{C-c}"         → Special(CtrlC)
    ///   "plain text"    → Literal("plain text")
    pub fn parse(input: &str) -> Result<Self>;

    /// Build programmatically.
    pub fn literal(text: &str) -> Self;
    pub fn then_literal(self, text: &str) -> Self;
    pub fn then_key(self, key: SpecialKey) -> Self;
    pub fn then_enter(self) -> Self;

    /// Render to one or more tmux send-keys invocations.
    /// Returns the shell commands to execute.
    fn to_tmux_commands(&self, target: &str) -> Vec<String>;
}
```

**Example usage**:

```rust
// Send "continue" followed by Enter
handle.send_keys(&pane, &KeySequence::literal("continue").then_enter()).await?;

// Send Ctrl-C to interrupt, then a new command
handle.send_keys(&pane, &KeySequence::parse("{C-c}ls -la{Enter}")?).await?;

// Send text that contains special characters (properly escaped via -l)
handle.send_text(&pane, "echo 'hello > world'").await?;
```

**Why this matters**: The prototype sends `send-keys 'continue' Enter` as a raw shell
string. This works for simple cases but breaks when the text contains quotes, semicolons,
or tmux key name collisions (e.g., sending the literal word "Enter" or "Space"). The
`KeySequence` abstraction handles all escaping correctly by splitting into `-l` (literal)
and non-`-l` (special key) invocations.

### `SessionInfo` / `WindowInfo` / `PaneInfo` — Rich Discovery Types

The prototype only discovers pane addresses. The library needs richer metadata for
listing and display.

```rust
pub struct SessionInfo {
    pub name: String,
    pub id: String,             // tmux internal $N id
    pub created: u64,           // unix timestamp
    pub attached: bool,         // has a client attached
    pub window_count: u32,
    pub group: Option<String>,  // session group, if any
}

pub struct WindowInfo {
    pub session: String,
    pub index: u32,
    pub name: String,
    pub active: bool,           // is the active window in its session
    pub pane_count: u32,
    pub layout: String,         // e.g. "main-vertical"
}

pub struct PaneInfo {
    pub address: PaneAddress,   // session:window.pane
    pub title: String,          // pane title
    pub current_command: String, // running command (e.g. "vim", "bash")
    pub pid: u32,               // pane process PID
    pub width: u32,
    pub height: u32,
    pub active: bool,           // is the active pane in its window
}
```

These are populated by `tmux list-sessions -F`, `list-windows -F`, and `list-panes -F`
with appropriate format strings. The format strings are centralized in `discovery.rs` to
keep tmux version coupling in one place.

---

## Module Specifications

### `config.rs`

Defines all user-facing configuration. This is the primary integration point.

```rust
pub struct TmuxAutomatorConfig {
    pub targets: Vec<HostTarget>,
    pub rules: Vec<TriggerRule>,
    pub reconnect: ReconnectPolicy,
    pub log_json: bool,
}

pub struct HostTarget {
    pub host: String,              // "host:port"
    pub user: String,
    pub alias: Option<String>,     // short name for Fleet lookups (default: host)
    pub pane_filter: Option<String>, // regex for monitoring (None = all panes)
}

pub struct TriggerRule {
    pub name: String,              // human-readable rule name for logging
    pub pane_filter: Option<Regex>,// which panes this rule applies to (None = all)
    pub pattern: Regex,            // match against pane output lines
    pub action: Action,
    pub cooldown: Option<Duration>,// debounce repeated triggers per pane
}

pub enum Action {
    SendKeys { keys: String },
    Log { level: Level, message: String },
    // Future: Notify { channel: String }, Webhook { url: String }
}

pub struct ReconnectPolicy {
    pub initial_delay: Duration,   // default: 2s
    pub max_delay: Duration,       // default: 60s
    pub multiplier: u32,           // default: 2
}
```

**Requirement**: Config must be constructable programmatically (library use) and deserializable
from TOML/YAML (CLI use).

### `connection.rs`

Wraps `russh` SSH client lifecycle.

```rust
pub struct SshConnection { /* russh Handle<SshHandler> */ }

impl SshConnection {
    /// Connect and authenticate. Returns error if auth fails.
    pub async fn connect(target: &HostTarget, config: &SshConfig) -> Result<Self>;

    /// Open a new exec channel, run a command, return stdout as String.
    pub async fn exec(&self, command: &str) -> Result<String>;

    /// Open a PTY channel with a shell. Returns the channel for streaming.
    pub async fn open_shell(&self) -> Result<ShellChannel>;
}
```

**Must address P1**: Host key verification.

- Default: verify against `~/.ssh/known_hosts`
- Option: TOFU (trust-on-first-use) with persistent known_hosts update
- Option: `--insecure` flag to skip (prototype behavior), with a logged warning

**Must address**: SSH config should set `connection_timeout` (10s) and `heartbeat_interval` (30s)
as the prototype does, but make them configurable.

### `discovery.rs`

Session, window, and pane introspection. All tmux format strings are defined here to
centralize version-dependent coupling.

```rust
pub struct PaneAddress {
    pub session: String,
    pub window: u32,
    pub pane: u32,
}

impl PaneAddress {
    /// Canonical string form: "session:window.pane"
    pub fn to_tmux_target(&self) -> String;

    /// Safe encoding for use in file paths (must roundtrip losslessly)
    pub fn to_safe_filename(&self) -> String;

    /// Parse from tmux list-panes output
    pub fn parse(s: &str) -> Result<Self>;
}
```

**Must address P2**: The safe filename encoding. Proposal: hex-encode the session name, use
literal `-` delimiters: `{hex(session)}-{window}-{pane}`. This avoids any ambiguity regardless
of session name content.

```rust
/// List all sessions on the host.
pub async fn list_sessions(conn: &SshConnection) -> Result<Vec<SessionInfo>>;

/// List all windows in a session.
pub async fn list_windows(conn: &SshConnection, session: &str) -> Result<Vec<WindowInfo>>;

/// List all panes, optionally filtered by regex against "session:window.pane".
pub async fn list_panes(
    conn: &SshConnection,
    filter: Option<&Regex>,
) -> Result<Vec<PaneInfo>>;
```

**Must address P10**: Add a timeout (default 10s) on all exec channels for tmux commands.

**Format strings** are defined as constants, e.g.:

```rust
const LIST_SESSIONS_FMT: &str =
    "#{session_name}\t#{session_id}\t#{session_created}\t#{session_attached}\t#{session_windows}\t#{session_group}";
const LIST_PANES_FMT: &str =
    "#{session_name}:#{window_index}.#{pane_index}\t#{pane_title}\t#{pane_current_command}\t#{pane_pid}\t#{pane_width}\t#{pane_height}\t#{pane_active}";
```

### `capture.rs`

Pane content capture via `tmux capture-pane`.

```rust
/// Capture the visible content of a single pane.
/// Runs: tmux capture-pane -p -t <target>
pub async fn capture_pane(conn: &SshConnection, target: &PaneAddress) -> Result<String>;

/// Capture with scrollback history.
/// Runs: tmux capture-pane -p -t <target> -S <start> -E <end>
/// start/end are line numbers; negative values reach into scrollback buffer.
/// Example: start=-1000, end=-1 captures last 1000 lines of scrollback.
pub async fn capture_pane_history(
    conn: &SshConnection,
    target: &PaneAddress,
    start: i32,
    end: i32,
) -> Result<String>;

/// Capture all panes in a session. Calls capture_pane for each pane found via list_panes.
/// Returns a map of pane address → visible content.
pub async fn capture_session(
    conn: &SshConnection,
    session: &str,
) -> Result<HashMap<PaneAddress, String>>;
```

**Escaping**: The `-p` flag outputs to stdout (not to a buffer), which is what we need
over SSH exec channels. The `-e` flag (escape sequences) is intentionally NOT used — we
want plain text, not ANSI-encoded output.

**Use cases**:
- Dump a pane to see what state it's in before deciding to send input
- Capture all panes in a session for a holistic snapshot (e.g., debugging a multi-pane layout)
- Feed pane content to an LLM for analysis/decision-making

### `control.rs`

Remote tmux control: sending input and managing session/window metadata.

```rust
/// Send a KeySequence to a pane. Handles the split between literal text (-l)
/// and special keys (no -l) automatically.
pub async fn send_keys(
    conn: &SshConnection,
    target: &PaneAddress,
    keys: &KeySequence,
) -> Result<()>;

/// Convenience: send literal text (no special keys, no Enter appended).
/// Equivalent to: tmux send-keys -l -t <target> '<escaped_text>'
pub async fn send_text(
    conn: &SshConnection,
    target: &PaneAddress,
    text: &str,
) -> Result<()>;

/// Rename a tmux session.
/// Runs: tmux rename-session -t <current> <new>
pub async fn rename_session(
    conn: &SshConnection,
    current_name: &str,
    new_name: &str,
) -> Result<()>;

/// Rename a window.
/// Runs: tmux rename-window -t <session>:<index> <new_name>
pub async fn rename_window(
    conn: &SshConnection,
    session: &str,
    window_index: u32,
    new_name: &str,
) -> Result<()>;
```

**Shell escaping**: All arguments passed to remote tmux commands must be shell-escaped.
The `control` module must escape user-provided strings (session names, text input) to
prevent shell injection. Use single-quote wrapping with interior single-quote escaping
(`'it'\''s'` pattern), or positional argument passing where possible.

**`send_keys` dispatch logic**:

```
KeySequence [Literal("hello"), Special(Enter), Literal("world"), Special(CtrlC)]
  ↓
Command 1: tmux send-keys -l -t session:0.0 'hello'
Command 2: tmux send-keys -t session:0.0 Enter
Command 3: tmux send-keys -l -t session:0.0 'world'
Command 4: tmux send-keys -t session:0.0 C-c
```

Adjacent segments of the same type can be batched into a single command where tmux allows it.

### `keys.rs`

Key literal definitions and the `KeySequence` parser. See the `KeySequence` type in
[Core Abstractions](#keysequence--safe-input-construction) for the full API.

This module owns:
- The `KeySequence` and `KeySegment` types
- The `SpecialKey` enum with all known tmux key names
- The `{...}` inline escape parser (`"text{Enter}more{C-c}"`)
- The builder API (`KeySequence::literal("x").then_enter()`)
- Rendering to tmux shell commands

### `fleet.rs`

Multi-host management. See the `Fleet` type in [Core Abstractions](#fleet--multi-host-pool)
for the full API.

Internal responsibilities:
- Holds a `HashMap<String, HostHandle>` keyed by host alias (or `host:port` if no alias)
- Connects hosts concurrently via `tokio::JoinSet`
- Tracks per-host status: `Disconnected`, `Connecting`, `Connected`, `Monitoring`, `Error(String)`
- Routes on-demand operations (capture, send-keys, list) to the correct host
- Owns the `shutdown` watch channel; `shutdown()` signals all hosts

```rust
pub enum HostStatus {
    Disconnected,
    Connecting,
    Connected,
    Monitoring,
    Error(String),
}
```

### `host.rs`

The `HostHandle` implementation. See [Core Abstractions](#hosthandle--per-host-facade).

Internally, `HostHandle` holds an `SshConnection` and delegates to the function-level
APIs in `discovery`, `capture`, `control`, `pipe`, and `monitor`. It is the composition
root for per-host operations.

```rust
pub struct HostHandle {
    conn: SshConnection,
    config: HostTarget,
    pipe_state: Option<PipeManager>,   // None if monitoring not started
    monitor_handle: Option<MonitorHandle>,
}
```

### `pipe.rs`

FIFO lifecycle and `tmux pipe-pane` management.

```rust
pub struct PipeManager { /* tracks active pipes for cleanup */ }

impl PipeManager {
    /// Create FIFOs and attach pipe-pane for each target pane.
    /// Uses a dedicated SSH channel (not the monitor channel).
    pub async fn setup(conn: &SshConnection, panes: &[PaneAddress]) -> Result<Self>;

    /// Detach all pipe-panes and remove FIFO files.
    pub async fn cleanup(&self, conn: &SshConnection) -> Result<()>;
}

impl Drop for PipeManager {
    // Best-effort: log warning if cleanup was not called explicitly.
    // Cannot do async cleanup in Drop, so this is advisory only.
}
```

**Must address P4 and P5**: `cleanup()` must be called from the signal handler and from the
normal shutdown path.

**Cleanup sequence** (on graceful or signal-triggered shutdown):
1. `tmux pipe-pane -t <pane>` (no `-o`) for each pane — detaches the pipe
2. `rm -f /tmp/tmux_pipe_<encoded>` for each FIFO
3. Close SSH channels

### `monitor.rs`

The core event loop. Reads the multiplexed output stream and evaluates rules.

```rust
pub struct OutputMonitor { /* rules, cooldown state, shell channel */ }

impl OutputMonitor {
    pub async fn run(
        &mut self,
        conn: &SshConnection,
        panes: &[PaneAddress],
        rules: &[TriggerRule],
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()>;
}
```

**Must address P3**: Rule evaluation replaces the hardcoded `contains('>')` check.

**Must address P6**: Monitoring uses a dedicated PTY channel that only runs `tail -qf`.
Action dispatch (send-keys) uses separate exec channels or a dedicated control channel
(see [DC4](#dc4-action-dispatch-channel-strategy)).

**Must address P9**: Failed send-keys or malformed lines must be logged at `warn` level,
not silently dropped.

**Stream parsing**: The `tail -qf` + `awk` approach from the prototype works but couples
filename-based pane identification to shell text parsing. The monitor must use the safe
filename encoding from `PaneAddress::to_safe_filename()` and a regex that accounts for it.

---

## Key Design Decisions

### DC1: Pane Address Encoding

**Decision**: Hex-encode the tmux session name in FIFO filenames.

**Rationale**: Tmux session names can contain underscores, dots, colons, and other characters
that conflict with naive delimiter-based encoding. Hex encoding the session name guarantees
lossless roundtrip. Window and pane indices are numeric and safe as-is.

**Format**: `/tmp/motlie_pipe_{hex(session)}_{window}_{pane}`

**Alternative rejected**: Base64 — contains `/` and `+` which are unsafe in filenames without
further escaping.

### DC2: Host Key Verification

**Decision**: Verify against `~/.ssh/known_hosts` by default.

**Rationale**: The prototype's `check_server_key → true` is a MITM vulnerability. A library
intended for automated remote execution must not silently accept unknown hosts.

**Behavior**:
- Default: reject unknown hosts, return error with instructions to add the key
- `--trust-first-use` / config flag: accept and persist on first connect, reject on mismatch
- `--insecure` / config flag: accept all (prototype behavior), log warning on every connection

### DC3: Trigger/Action Model

**Decision**: Configurable rules with per-pane cooldown.

**Rationale**: The prototype's hardcoded `contains('>')` → `send-keys 'continue'` is a
demonstration, not a design. Real use cases include:
- Responding to `[y/N]` prompts with `y`
- Detecting error patterns and logging/alerting
- Sending different commands to different session types

**Cooldown**: Rules fire at most once per `cooldown` duration per pane. This prevents
rapid-fire send-keys when a prompt re-renders. Default cooldown: 1 second.

### DC4: Action Dispatch Channel Strategy

**Decision**: TBD — needs prototyping.

**Option A**: Open a new SSH exec channel per action (`tmux send-keys -t ...`).
- Pro: Simple, isolated, no shared state
- Con: Channel open latency (~50ms per action), may hit SSH channel limits under load

**Option B**: Maintain a persistent control shell channel, write commands to it.
- Pro: Lower latency, no per-action channel overhead
- Con: Shared channel parsing complexity (same issue as prototype P6)

**Recommendation**: Start with Option A. Measure latency. If channel open overhead is
problematic (>100ms p99 or >10 actions/sec sustained), switch to Option B with a dedicated
channel that does not share with the monitor.

### DC5: Multi-Host Concurrency

**Decision**: One `tokio::spawn` task per host, independent lifecycle.

**Rationale**: Hosts are independent. A connection failure on host A must not affect host B.
Each task runs the full lifecycle: connect → discover → pipe → monitor → cleanup.

**Shared state**: Only the rule set and shutdown signal are shared across hosts. Per-host
state (connection, panes, pipes, cooldown timers) is task-local.

### DC6: Capture-Pane vs Pipe-Pane for Content Access

**Decision**: Use `capture-pane -p` for on-demand content access, `pipe-pane` for continuous
monitoring. These are complementary, not alternatives.

**Rationale**:
- `capture-pane -p` returns the current visible content (and scrollback with `-S`/`-E`)
  as a single snapshot. It is stateless, idempotent, and requires no setup. Ideal for
  "what is this pane showing right now?" queries.
- `pipe-pane` streams ongoing output to a pipe. It captures content as it arrives, which
  `capture-pane` would miss between polling intervals. Required for trigger/response automation.

**Callers choose the mode**:
- `capture_pane()` / `capture_session()` → snapshot via `capture-pane`
- `start_monitoring()` → continuous via `pipe-pane`

### DC7: Key Escaping Strategy

**Decision**: Split `KeySequence` into literal segments (sent with `send-keys -l`) and
special key segments (sent without `-l`).

**Rationale**: Tmux `send-keys` without `-l` interprets arguments as key names. The word
"Enter" sends a newline, "Space" sends a space, "C-c" sends Ctrl-C. With `-l`, all text
is literal — "Enter" sends the five characters E-n-t-e-r. There is no single mode that
handles both correctly. The library must split input into segments and issue multiple
`send-keys` invocations.

**Shell injection prevention**: All literal text segments are single-quote-escaped before
passing to the remote shell. The `control` module is the only code path that constructs
remote shell commands from user input, making it the single point of audit for injection.

### DC8: Fleet-Level Host Addressing

**Decision**: Hosts are addressed by alias (if configured) or by `host:port` string.
All `Fleet` methods that target a specific host accept `&str` and resolve internally.

**Rationale**: Users operating across multiple machines want to say "capture pane on
`web-server`", not remember `10.0.1.42:22`. Aliases are optional — single-host use
cases just use the host string directly.

**Conflict resolution**: If two hosts share an alias, config validation fails at load time.

### DC9: Separation of Library and Binary

**Decision**: `libs/tmux` is a pure library. CLI binary lives in `bins/tmux-automator/`.

**Rationale**: Follows the existing workspace convention (`libs/mcp` + `examples/mcp/`).
The library exposes `Fleet`, `HostHandle`, `TmuxAutomatorConfig`, and all operation APIs.
The binary handles CLI parsing, config file loading, signal handling, and tracing
initialization. The library is also consumable by MCP tools or other programmatic callers.

---

## Open Concerns

These require resolution before or during implementation. They are not blockers for Phase 1
but must be addressed before the library is used in production.

### OC1: FIFO Reliability Under Load

`tmux pipe-pane` writes to a FIFO. If the reader (`tail -qf`) falls behind, the writing
pane blocks until the FIFO is drained. Under high output volume, this could stall the
monitored tmux pane.

**Mitigation options**:
- Use `tmux pipe-pane -o 'cat >> /tmp/file'` (regular file, not FIFO) and `tail -f` the file.
  Avoids blocking but requires periodic truncation/rotation.
- Accept the FIFO behavior and document it as a known limitation.
- Investigate `tmux capture-pane -p -t <pane>` polling as an alternative to pipe-pane.

### OC2: Output Interleaving

`tail -qf` on multiple files can interleave partial lines from different panes.
The `awk` prefix helps, but a long line split across reads could produce garbled attribution.

**Mitigation**: Use line-buffered output (`stdbuf -oL`) if available on the remote host.
Document as a known limitation if not.

### OC3: SSH Agent Availability

The prototype requires a running `ssh-agent` with loaded keys. If the agent is unavailable
or has no matching keys, the error message should be actionable.

**Proposal**: On auth failure, enumerate the attempted key fingerprints and suggest
`ssh-add` if no keys were found.

### OC4: Tmux Version Compatibility

`tmux pipe-pane` and `list-panes -F` format strings vary across tmux versions.
The minimum supported tmux version should be documented and tested.

**Proposal**: Require tmux >= 2.6 (pipe-pane `-o` flag). Detect version via
`tmux -V` during discovery phase and return a clear error if unsupported.

### OC5: Shell Injection in Remote Commands

The `control` and `capture` modules construct shell commands from user-provided strings
(session names, pane text, rename targets). A session named `; rm -rf /` must not result
in command execution.

**Mitigation**: All user-provided strings must be escaped before interpolation into
remote shell commands. The `control` module is the single audit point. Proposed approach:
single-quote wrapping with `'\''` escaping for interior single-quotes. This is the
standard POSIX shell escaping pattern.

**Validation**: Unit tests must cover adversarial inputs: semicolons, backticks, `$(...)`,
newlines, null bytes, and single/double quotes in session names and text input.

### OC6: Testing Strategy

The prototype is untestable because SSH and tmux are tightly coupled. The library needs:

- **Unit tests**: Rule evaluation, pane address encoding/decoding, config parsing
- **Integration tests**: Require a mock SSH server or trait-based transport abstraction
- **End-to-end tests**: Require a real SSH + tmux environment (CI with Docker?)

**Proposal**: Define a `Transport` trait in `connection.rs` that `SshConnection` implements.
Tests use a `MockTransport` that returns canned responses for `exec()` and `open_shell()`.

---

## Dependency Inventory

| Crate | Version | Purpose | Workspace? |
|-------|---------|---------|------------|
| `russh` | 0.46+ | SSH client transport | No |
| `russh-keys` | 0.46+ | SSH agent integration, known_hosts parsing | No |
| `clap` | 4.x | CLI argument parsing (binary only) | No |
| `regex` | 1.x | Pane filtering + output parsing | No |
| `tokio` | 1.x | Async runtime, signals, timers | Yes |
| `tracing` | 0.1 | Structured logging | No |
| `tracing-subscriber` | 0.3 | Log output formatting (text/JSON) | No |
| `anyhow` | 1.x | Error handling | Yes |
| `serde` | 1.x | Config deserialization | Yes |
| `toml` | 0.8 | Config file parsing (binary only) | No |

---

## Implementation Phases

Structured for incremental delivery. Each phase produces a working, testable artifact.

### Phase 1: Types, Connection, and On-Demand Operations

**Goal**: Establish the core types and single-host on-demand capabilities (list, capture,
send-keys, rename). No monitoring yet.

**Tasks**:
1. Create `libs/tmux/` workspace member with `Cargo.toml`
2. Implement `types.rs`: `PaneAddress` with safe filename encoding (fixes P2),
   `SessionInfo`, `WindowInfo`, `PaneInfo`
3. Implement `keys.rs`: `KeySequence`, `SpecialKey`, `{...}` parser, tmux command rendering
4. Implement `connection.rs`: `SshConnection` with known_hosts verification (fixes P1),
   `exec()` with configurable timeout (fixes P10), `open_shell()`
5. Implement `discovery.rs`: `list_sessions()`, `list_windows()`, `list_panes()` with
   format string constants and regex filtering
6. Implement `capture.rs`: `capture_pane()`, `capture_pane_history()`, `capture_session()`
7. Implement `control.rs`: `send_keys()`, `send_text()`, `rename_session()`, `rename_window()`
   with shell escaping (addresses OC5)
8. Implement `host.rs`: `HostHandle` wiring all of the above
9. Unit tests: `PaneAddress` roundtrip, `KeySequence` parsing + rendering, shell escaping
   with adversarial inputs

**Deliverable**: Library that connects to one host and supports all on-demand tmux operations.
A caller can list sessions, capture pane content, send input, and rename sessions.

**Acceptance criteria**:
- `PaneAddress::to_safe_filename()` roundtrips losslessly for names containing `_.:;`
- `KeySequence::parse("hello{Enter}{C-c}")` produces correct tmux commands
- `send_text()` with input `"; rm -rf /; echo "` does not execute injected commands
- `list_sessions()` returns `Vec<SessionInfo>` with all fields populated

### Phase 2: Monitoring + Pipe Management

**Goal**: Add continuous monitoring (the prototype's core loop) on top of Phase 1.

**Tasks**:
1. Implement `config.rs`: `TriggerRule`, `Action`, `ReconnectPolicy`, `TmuxAutomatorConfig`
   with serde deserialization
2. Implement `pipe.rs`: `PipeManager` with setup + cleanup (fixes P4), using safe filenames
3. Implement `monitor.rs`: event loop with configurable rules (fixes P3), warn on
   errors (fixes P9), `MonitorHandle` for stop signaling
4. Wire monitoring into `HostHandle::start_monitoring()`
5. Signal handling via `tokio::signal` for SIGINT/SIGTERM (fixes P5), wired to
   `PipeManager::cleanup()`
6. Reconnection logic with exponential backoff in `HostHandle`
7. Unit tests: rule evaluation, cooldown behavior, config deserialization

**Deliverable**: Single-host library with both on-demand operations and continuous monitoring.

**Acceptance criteria**:
- Monitoring starts, detects a pattern, and sends the configured response
- SIGINT triggers pipe-pane detach and FIFO removal on the remote host
- Reconnection resumes monitoring after a simulated SSH disconnect
- Rules with cooldown do not fire more than once per cooldown period

### Phase 3: Multi-Host Fleet + CLI

**Goal**: Multi-host support via `Fleet` and a usable CLI binary.

**Tasks**:
1. Implement `fleet.rs`: `Fleet` with concurrent connect, host lookup by alias (DC8),
   per-host status tracking, aggregate `start_monitoring()` and `shutdown()`
2. Create `bins/tmux-automator/` with `clap` CLI supporting subcommands:
   - `list-sessions [--host <alias>]` — list sessions on one or all hosts
   - `list-panes [--host <alias>] [--filter <regex>]` — list panes
   - `capture <session:window.pane> [--host <alias>] [--history <lines>]` — dump pane
   - `send <session:window.pane> <input> [--host <alias>]` — send keys
   - `rename-session <old> <new> [--host <alias>]` — rename
   - `monitor [--config <path>]` — start continuous monitoring
3. Config file support (TOML) with CLI flag overrides
4. JSON and text log output modes
5. Per-host tracing spans with host alias labels

**Deliverable**: Multi-host CLI tool and library. Single process operating across N hosts.

**Acceptance criteria**:
- `fleet.connect_all()` connects to 3 hosts concurrently; one failure does not block others
- CLI `list-sessions --host web-server` returns sessions from the aliased host
- CLI `capture myapp:0.0 --host db-server --history 500` returns scrollback content
- `monitor --config rules.toml` starts monitoring on all configured hosts

### Phase 4: Hardening + Testing

**Goal**: Production readiness.

**Tasks**:
1. `Transport` trait + `MockTransport` for unit/integration tests (OC6)
2. Tmux version detection and compatibility check (OC4)
3. Actionable SSH agent error messages (OC3)
4. FIFO-vs-file investigation and decision (OC1)
5. Line interleaving mitigation (OC2)
6. End-to-end test with Docker (SSH + tmux)
7. Document minimum tmux version, known limitations, and performance characteristics

**Deliverable**: Library with test coverage, documented limitations, and CI integration.

---

## References

### Prototype Origin

- [Gemini conversation — Rust SSH Tmux Session Interaction](https://g.co/gemini/share/e7eb11c45954)

### External Documentation

- [russh documentation](https://docs.rs/russh/latest/russh/)
- [tmux pipe-pane man page](https://man.openbsd.org/tmux#pipe-pane)
- [tmux capture-pane man page](https://man.openbsd.org/tmux#capture-pane)
- [tmux send-keys man page](https://man.openbsd.org/tmux#send-keys)

### Internal Documentation

- [motlie MCP DESIGN.md](../../../libs/mcp/docs/DESIGN.md) — reference for doc conventions
