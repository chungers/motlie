# Tmux Multi-Target Automator Design

## Status: Draft

This document describes the design for `libs/tmux`, an asynchronous, structured, multi-target
automator that monitors tmux panes over SSH or on localhost and executes configurable actions
in response to output patterns. Beyond monitoring, the library provides a general-purpose tmux
control plane: creating and terminating sessions, listing sessions, capturing pane content,
sending arbitrary input with proper escaping, and managing session metadata — across localhost
and multiple remote hosts concurrently. The design is derived from a working single-host
prototype and specifies the decomposition, safety fixes, and extensions required for a
production library.

## Table of Contents

- [Overview](#overview)
- [Prototype Reference](#prototype-reference)
- [Architecture](#architecture)
- [Core Abstractions](#core-abstractions)
- [Output Sink Pipeline](#output-sink-pipeline)
- [Module Specifications](#module-specifications)
- [Key Design Decisions](#key-design-decisions)
- [Open Concerns](#open-concerns)
- [Dependency Inventory](#dependency-inventory)
- [Implementation Phases](#implementation-phases)

---

## Overview

### Problem Statement

Interactive tmux sessions — whether on the local machine or remote hosts — often reach states
that require human intervention (confirmation prompts, error recovery, continuation signals).
When operating across multiple hosts and sessions, manual monitoring does not scale.

### Solution

A library that:

1. Operates on localhost tmux directly, or connects to remote hosts via SSH
2. Creates and terminates tmux sessions
3. Lists and inspects tmux sessions, windows, and panes on each target
4. Captures pane content (scrollback + visible) as text on demand
5. Sends arbitrary input to panes with proper key escaping (Enter, C-c, etc.)
6. Manages session metadata (rename sessions/windows)
7. Attaches output pipes for continuous monitoring
8. Evaluates configurable trigger rules against pane output
9. Executes actions (send-keys, notify, log) when rules match
10. Reconnects automatically on failure (SSH targets)

### Scope

- **In scope**: Localhost tmux (direct execution), SSH transport for remote hosts,
  multi-host connection pool, tmux session creation and termination, session/window/pane
  listing, pane content capture, remote input with escaping, session metadata management,
  pipe-based output monitoring, rule-based automation, structured logging, CLI binary
- **Out of scope**: Web UI, SSH server setup/configuration, tmux installation
- **Future**: TUI interface based on [ratatui](https://ratatui.rs/) (not in current phases)

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
| P2 | Pane ID encoding is lossy | **High** | `_` in session names breaks roundtrip through FIFO naming; resolved by using `#{pane_id}` as authoritative key (DC1) |
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
│   ├── transport.rs        # Transport trait + LocalTransport + SshTransport
│   ├── host.rs             # HostHandle: per-host facade for all tmux operations
│   ├── fleet.rs            # Fleet: multi-host pool, dispatch, aggregate status
│   ├── discovery.rs        # Session/window/pane listing, filter, PaneAddress type
│   ├── capture.rs          # Pane content capture (capture-pane) and scrollback dump
│   ├── control.rs          # Session lifecycle, send-keys with escaping, rename
│   ├── pipe.rs             # PipeManager: FIFO lifecycle, pipe-pane attach/detach
│   ├── monitor.rs          # OutputMonitor: stream parsing, rule evaluation, dispatch
│   ├── sink.rs             # OutputSink trait, SinkFilter, PaneOutput, OutputBus
│   ├── sinks/
│   │   ├── mod.rs          # Re-exports built-in sinks
│   │   └── stdio.rs        # StdioSink (default reference implementation)
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
  │  Manages N targets, routes commands, aggregates status        │
  └──────┬──────────────────┬──────────────────┬──────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ HostHandle  │   │ HostHandle  │   │ HostHandle  │  × N targets
  │ (localhost)  │   │ (host B)    │   │ (host C)    │
  │ Local       │   │ SSH         │   │ SSH         │
  └──────┬──────┘   └─────────────┘   └─────────────┘
         │
         │  Transport-agnostic (LocalTransport or SshTransport):
         │
         ├── Discovery ──── list sessions, windows, panes
         │
         ├── Capture ────── dump pane content as text
         │
         ├── Control ────── create/kill session, send-keys, rename
         │
         ├── PipeManager ── attach/detach output pipes
         │
         └── Monitor ────── stream + rule evaluation → action dispatch
```

**Two usage modes coexist on the same `HostHandle`**:

1. **On-demand operations**: Create/kill sessions, list sessions, capture pane, send input,
   rename — called directly by the consumer (CLI command, MCP tool, API call). These execute
   via the transport (local subprocess or SSH exec channel) and return immediately.

2. **Continuous monitoring**: Pipe setup + event loop — long-running, spawned as a
   background task. Uses dedicated shell/PTY channel. Rule-triggered actions dispatch through
   the same `Control` module as on-demand operations.

### Concurrency Model

Each target is represented by a `HostHandle` backed by a `Transport` implementation.
For SSH targets, the transport multiplexes channels over one TCP connection. For localhost,
the transport spawns `tokio::process::Command` subprocesses.

**On-demand operations** (create, kill, list, capture, send-keys, rename):
- Each call executes via the transport (SSH exec channel or local subprocess)
- Multiple on-demand calls can run concurrently on the same target
- No long-lived state required

**Continuous monitoring** (when started):
- **Pipe setup**: Dedicated transport exec (not the monitor channel)
- **Monitoring**: Dedicated channel — control mode session (`tmux -C attach`, DC10)
  as primary; `tail -qf` on pipe files as fallback only
- **Action dispatch**: Separate exec calls per triggered action
  (see [DC4](#dc4-action-dispatch-channel-strategy))
- Runs as a `tokio::spawn` task; does not block on-demand operations

**Multi-target**: `Fleet` spawns per-target monitoring tasks independently. A connection
failure on host A does not affect host B or localhost. On-demand operations are dispatched
to the named target's `HostHandle` directly.

---

## Core Abstractions

This section defines the primary abstractions that structure the library. These are the
types a consumer interacts with. Module specifications (next section) describe internals.

### `Fleet` — Multi-Target Pool

The top-level entry point. Manages connections to localhost and/or multiple remote hosts,
providing a uniform interface for dispatching operations to any target by name or alias.

```rust
pub struct Fleet { /* HashMap<String, HostHandle>, shutdown_tx, config */ }

impl Fleet {
    /// Create a fleet from config. Does not connect yet.
    pub fn new(config: TmuxAutomatorConfig) -> Self;

    /// Connect to all configured targets (localhost is always available).
    /// Failures are per-target, not fatal.
    /// Returns a summary of which targets connected and which failed.
    pub async fn connect_all(&mut self) -> Vec<HostStatus>;

    /// Get a handle to a specific target by name/alias.
    /// The localhost target is always addressable as "localhost" or "local".
    pub fn host(&self, name: &str) -> Option<&HostHandle>;

    /// Iterate over all connected targets.
    pub fn hosts(&self) -> impl Iterator<Item = (&str, &HostHandle)>;

    /// Start monitoring on all connected hosts (spawns background tasks).
    pub async fn start_monitoring(&self, rules: &[TriggerRule]) -> Result<()>;

    /// Shutdown: stop monitoring, cleanup pipes, close connections.
    pub async fn shutdown(&self) -> Result<()>;
}
```

**Design rationale**: Callers should not need to manage individual connections or reason
about which target a session lives on. `Fleet` provides the "I have N targets, operate
on them" abstraction. For localhost-only use, `Fleet` with one local target works
identically — there is no separate single-target API. Localhost is always available
without SSH configuration.

### `HostHandle` — Per-Host Facade

All tmux operations on a single target go through `HostHandle`. This is the core
abstraction that unifies discovery, capture, control, and monitoring behind one type.
It is transport-agnostic — the same API works for localhost and SSH targets.

```rust
pub struct HostHandle { /* Transport, host config, pipe state */ }

impl HostHandle {
    // --- Session Lifecycle ---

    /// Create a new tmux session.
    /// `window_name` and `command` are optional; if `command` is provided, it runs
    /// in the initial window instead of a default shell.
    /// Runs: tmux new-session -d -s <name> [-n <window_name>] [<command>]
    pub async fn create_session(
        &self,
        name: &str,
        window_name: Option<&str>,
        command: Option<&str>,
    ) -> Result<()>;

    /// Terminate (kill) a tmux session and all its windows/panes.
    /// Runs: tmux kill-session -t <name>
    pub async fn kill_session(&self, name: &str) -> Result<()>;

    // --- Discovery ---

    /// List all tmux sessions on this target.
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

## Output Sink Pipeline

Captured pane output must flow to multiple consumers with vastly different latency
profiles — stdio (µs), TUI (ms per frame), LLM analysis (seconds). A synchronous
pipeline would block at the speed of the slowest consumer. This section defines the
async fan-out architecture that decouples output capture from rendering/processing.

### `PaneOutput` — The Unit of Output

Every piece of captured output flows through the pipeline as a `PaneOutput`. It carries
enough context for any sink to identify the source and route actions back to it.

```rust
pub struct PaneOutput {
    pub pane_id: String,       // authoritative: "%12"
    pub pane_target: String,   // display: "session:window.pane"
    pub host: String,          // host alias (or "localhost")
    pub session: String,       // session name
    pub window: u32,           // window index
    pub content: String,       // the captured text
    pub timestamp: Instant,
}
```

`PaneOutput` is the bridge between the monitor/capture side and the sink side.
The `host`, `session`, `window`, and `pane_id` fields enable sinks to filter at
any level of the hierarchy and to route actions back to the originating entity.

### `SinkFilter` — Composable Output Targeting

A `SinkFilter` selects which output reaches a given sink. Filters target any
combination of host, session, window, or pane. Multiple filters are combined with
OR semantics — output matching **any** filter in the set is delivered to the sink.

```rust
pub struct SinkFilter {
    pub host: Option<String>,      // regex against host alias
    pub session: Option<String>,   // regex against session name
    pub window: Option<String>,    // regex against "session:window_index"
    pub pane: Option<String>,      // regex against pane_id or "session:window.pane"
}

/// Compiled form — regexes compiled once at registration time.
pub struct CompiledSinkFilter {
    pub host: Option<Regex>,
    pub session: Option<Regex>,
    pub window: Option<Regex>,
    pub pane: Option<Regex>,
}

impl CompiledSinkFilter {
    /// Returns true if the output matches ALL non-None fields in this filter.
    /// Fields that are None are wildcards (match everything).
    pub fn matches(&self, output: &PaneOutput) -> bool;
}
```

**Combining filters**: A sink registers with `Vec<SinkFilter>`. Output is delivered
if it matches **any** filter in the vec (OR across filters, AND within each filter).
An empty vec means "match all output" (the default).

**Examples**:

```rust
// Sink receives output from all panes on "db-server"
vec![SinkFilter { host: Some("db-server".into()), ..Default::default() }]

// Sink receives output from session "build" on any host, OR any session on "web-1"
vec![
    SinkFilter { session: Some("build".into()), ..Default::default() },
    SinkFilter { host: Some("web-1".into()), ..Default::default() },
]

// Sink receives output from a specific pane
vec![SinkFilter { pane: Some("%42".into()), ..Default::default() }]
```

### `OutputSink` — The Sink Trait

Every output consumer implements `OutputSink`. Each sink runs as an independent async
task with its own batching/accumulation behavior. The library provides `StdioSink` as
the default reference implementation.

```rust
#[async_trait]
pub trait OutputSink: Send + Sync + 'static {
    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str;

    /// Filters determining which output reaches this sink.
    /// Empty vec = all output (default).
    fn filters(&self) -> Vec<SinkFilter> { vec![] }

    /// Process one output event. Called from the sink's own task —
    /// never from the monitor/bus hot path.
    ///
    /// Sinks are responsible for their own batching and accumulation.
    /// A stdio sink writes immediately; an LLM sink accumulates and
    /// flushes on its own schedule. The bus does not batch on behalf
    /// of sinks.
    async fn write(&self, output: PaneOutput) -> Result<()>;

    /// Called on bus shutdown. Flush internal buffers, close resources.
    async fn flush(&self) -> Result<()> { Ok(()) }
}
```

**Key design principle**: Each sink owns its batching/accumulation strategy internally.
The bus delivers individual `PaneOutput` events; the sink decides whether to process
them immediately (stdio), buffer and render at frame rate (TUI), or accumulate and
flush on a timer/threshold (LLM). This keeps the bus simple and the sink in full
control of its own latency/throughput tradeoffs.

### `ActionHandle` — Sink-Initiated Actions

Sinks that analyze output may need to act on the source entity — send keys to the
pane, kill a session, etc. Rather than giving sinks direct access to `HostHandle`
(which would create circular dependencies), sinks receive an `ActionHandle` that
provides a scoped, async API for actions against the tmux entities they observe.

```rust
pub struct ActionHandle { /* mpsc::Sender<ActionRequest> */ }

pub struct ActionRequest {
    pub target: ActionTarget,
    pub action: SinkAction,
}

pub enum ActionTarget {
    Pane { host: String, pane_id: String },
    Session { host: String, session: String },
    Host { host: String },
}

pub enum SinkAction {
    SendKeys { keys: KeySequence },
    SendText { text: String },
    KillSession,
    RenameSession { new_name: String },
    // Extensible for future actions
}

impl ActionHandle {
    /// Send an action to the target entity. Non-blocking (queued).
    pub async fn send(&self, request: ActionRequest) -> Result<()>;

    /// Convenience: send keys to a specific pane.
    pub async fn send_keys_to_pane(
        &self,
        host: &str,
        pane_id: &str,
        keys: KeySequence,
    ) -> Result<()>;

    /// Convenience: send text to a specific pane.
    pub async fn send_text_to_pane(
        &self,
        host: &str,
        pane_id: &str,
        text: &str,
    ) -> Result<()>;
}
```

The `ActionHandle` is provided to sinks at registration time. Action requests are
routed through the existing per-host bounded dispatch queue (DC4) — the same path
used by the monitor's trigger rules. This ensures consistent ordering, backpressure,
and concurrency limits regardless of whether an action originates from a rule or a sink.

**LLM feedback loop**: An LLM sink can call `action_handle.send_keys_to_pane()` after
analyzing output. The design of the LLM sink itself (prompt engineering, approval gates,
autonomous vs supervised mode) is **out of scope** for this library. The library provides
the `OutputSink` trait and `ActionHandle` API; LLM integration is a consumer concern.

### `OutputBus` — Fan-Out Dispatcher

The `OutputBus` is the central distributor. It receives `PaneOutput` from the monitor
(or from `capture_pane()` callers) and fans out to all registered sinks.

```rust
pub struct OutputBus { /* subscribers: Vec<SinkEntry> */ }

struct SinkEntry {
    id: SinkId,
    name: String,
    tx: mpsc::Sender<PaneOutput>,
    filters: Vec<CompiledSinkFilter>,
    task: JoinHandle<()>,
}

impl OutputBus {
    pub fn new() -> Self;

    /// Register a sink. Spawns a dedicated tokio task that drives the sink.
    /// Returns a SinkId for later unsubscribe.
    /// `channel_capacity` controls the bounded channel size to this sink.
    /// Sinks that need to initiate actions should capture an `ActionHandle`
    /// at construction time — the bus does not inject one.
    pub fn subscribe(
        &mut self,
        sink: Box<dyn OutputSink>,
        channel_capacity: usize,
    ) -> SinkId;

    /// Remove a sink. Signals stop, awaits flush(), joins the task.
    pub async fn unsubscribe(&mut self, id: SinkId) -> Result<()>;

    /// Fan out an event to all matching sinks. Non-blocking.
    /// Uses try_send — if a sink's channel is full, the event is dropped
    /// for that sink only (logged at debug level). Sinks that cannot
    /// tolerate drops should use larger channel capacities.
    pub fn publish(&self, output: PaneOutput);

    /// Shutdown all sinks gracefully.
    pub async fn shutdown(&mut self) -> Result<()>;
}
```

**Backpressure**: The bus uses `try_send()` for all sinks. A full channel means the
sink is processing slower than output arrives — the bus drops for that sink only and
logs at debug level. This guarantees the bus (and therefore the monitor) never blocks.
Sinks that need lossless delivery should set a large `channel_capacity`. Sinks that
only care about recent state (TUI) should set a small capacity and accept drops.

### `StdioSink` — Default Reference Implementation

The library ships with `StdioSink` as the default, always-available sink. It serves
as the reference implementation for the `OutputSink` trait.

```rust
pub struct StdioSink {
    format: StdioFormat,
    writer: tokio::io::Stdout,
}

pub enum StdioFormat {
    /// Raw content only, no metadata prefix
    Raw,
    /// "[host] session:window.pane | content"
    Prefixed,
    /// JSON lines: {"host": "...", "pane": "...", "content": "...", "ts": ...}
    Json,
}

impl OutputSink for StdioSink {
    fn name(&self) -> &str { "stdio" }
    // filters(): default (all output)
    // write(): format and write to stdout immediately (no batching)
    // flush(): flush stdout
}
```

### Integration with Fleet and Monitor

```
  Fleet
    │
    ├── OutputBus (owned by Fleet)
    │     │
    │     ├── subscribe(StdioSink, action_handle, 1024)
    │     ├── subscribe(TuiSink, action_handle, 16)    // binary-provided
    │     └── subscribe(LlmSink, action_handle, 256)   // consumer-provided
    │
    ├── HostHandle (localhost)
    │     └── Monitor ──publish()──► OutputBus
    │
    └── HostHandle (remote)
          └── Monitor ──publish()──► OutputBus
```

**Monitor → Bus**: The monitor publishes `PaneOutput` to the bus after each output
event. This happens *in addition to* rule evaluation — rules and sinks operate
independently on the same stream.

**capture_pane() → Bus**: On-demand captures can optionally be published to the bus
via a `bus.publish()` call. This is opt-in at the call site, not automatic.

**Sink → ActionHandle → HostHandle**: A sink that decides to act (e.g., an LLM sink
that detects an error and wants to send a recovery command) submits an `ActionRequest`
through its `ActionHandle`. The request is routed to the correct `HostHandle` via the
existing per-host dispatch queue (DC4). The sink does not need to know which
`HostHandle` to use — routing is by the `host` and `pane_id` fields in `ActionTarget`.

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

pub enum HostTarget {
    Local {
        alias: Option<String>,         // default: "localhost"
        pane_filter: Option<String>,   // regex for monitoring (None = all panes)
        tmux_socket: Option<TmuxSocket>, // target a non-default tmux server
    },
    Ssh {
        host: String,                  // "host:port"
        user: String,
        alias: Option<String>,         // short name for Fleet lookups (default: host)
        pane_filter: Option<String>,   // regex for monitoring (None = all panes)
        tmux_socket: Option<TmuxSocket>, // target a non-default tmux server on remote
    },
}

/// Selects which tmux server to target on a given host.
/// Maps to tmux's `-L` (socket name) and `-S` (socket path) flags.
/// When None, the default tmux server is used.
pub enum TmuxSocket {
    /// Named socket: `tmux -L <name>` (looks in default socket dir)
    Name(String),
    /// Explicit socket path: `tmux -S <path>`
    Path(String),
}

/// Config DTO — deserialized from TOML/YAML. Patterns are strings.
pub struct TriggerRule {
    pub name: String,                  // human-readable rule name for logging
    pub pane_filter: Option<String>,   // regex string (None = all panes)
    pub pattern: String,               // regex string to match against pane output
    pub action: Action,
    pub cooldown: Option<Duration>,    // debounce repeated triggers per pane
}

/// Runtime form — compiled from TriggerRule during startup validation.
/// Compile errors include rule name and pattern for user-facing diagnostics.
pub struct CompiledRule {
    pub name: String,
    pub pane_filter: Option<Regex>,
    pub pattern: Regex,
    pub action: Action,
    pub cooldown: Option<Duration>,
}

impl TriggerRule {
    /// Compile string patterns into Regex. Returns error with rule name context.
    pub fn compile(&self) -> Result<CompiledRule>;
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

### `transport.rs`

The transport layer abstracts command execution, allowing the same tmux operations to
work on localhost (via `tokio::process::Command`) or remote hosts (via `russh` SSH channels).

```rust
#[async_trait]
pub trait Transport: Send + Sync {
    /// Execute a command and return its stdout as a String.
    /// Must respect the configured timeout.
    async fn exec(&self, command: &str) -> Result<String>;

    /// Open a persistent shell for streaming output.
    /// Used by the monitor for long-running processes (control mode session
    /// or `tail -qf` in fallback pipe mode).
    async fn open_shell(&self) -> Result<Box<dyn ShellChannel>>;
}

#[async_trait]
pub trait ShellChannel: Send {
    /// Write data to the shell's stdin.
    async fn write(&mut self, data: &[u8]) -> Result<()>;

    /// Wait for the next message from the shell.
    async fn read(&mut self) -> Option<ShellEvent>;
}

pub enum ShellEvent {
    Data(Vec<u8>),
    Eof,
}
```

**`LocalTransport`**: Executes commands via `tokio::process::Command`. No connection
setup required. `exec()` spawns a subprocess, waits for completion, returns stdout.
`open_shell()` spawns a persistent `bash` (or `sh`) process with piped stdin/stdout.

**`SshTransport`**: Wraps `russh` SSH client. `exec()` opens a short-lived SSH exec
channel. `open_shell()` opens a PTY channel with a shell.

**`MockTransport`**: For testing. Returns canned responses for `exec()` and canned
streaming data for `open_shell()`. Implements `Transport` trait directly — no separate
test infrastructure needed.

**SSH-specific concerns (SshTransport only)**:

- **Host key verification (P1)**:
  - Default: verify against `~/.ssh/known_hosts`
  - Option: TOFU (trust-on-first-use) with persistent known_hosts update
  - Option: `--insecure` / config flag to skip (prototype behavior), with a logged warning
- **Connection config**: `connection_timeout` (10s) and `heartbeat_interval` (30s)
  as the prototype does, but configurable
- **Reconnection**: Handled by `HostHandle`, not the transport itself

### `discovery.rs`

Session, window, and pane introspection. All tmux format strings are defined here to
centralize version-dependent coupling.

```rust
pub struct PaneAddress {
    pub pane_id: String,       // authoritative: "%12" — from #{pane_id}
    pub session: String,       // display: session name
    pub window: u32,           // display: window index
    pub pane: u32,             // display: pane index
}

impl PaneAddress {
    /// Canonical string form for tmux targeting: "session:window.pane"
    pub fn to_tmux_target(&self) -> String;

    /// The stable pane_id for FIFO naming and stream keying (e.g., "%12")
    pub fn id(&self) -> &str;

    /// Parse from tmux list-panes output (expects pane_id in format string)
    pub fn parse(s: &str) -> Result<Self>;
}
```

**Addresses P2**: Using `#{pane_id}` as the authoritative key (see DC1) eliminates the
need for filename encoding of session names entirely. FIFO paths (if used) are simply
`/tmp/motlie_pipe_%<id>`.

```rust
/// List all sessions on the host.
pub async fn list_sessions(transport: &dyn Transport) -> Result<Vec<SessionInfo>>;

/// List all windows in a session.
pub async fn list_windows(transport: &dyn Transport, session: &str) -> Result<Vec<WindowInfo>>;

/// List all panes, optionally filtered by regex against "session:window.pane".
pub async fn list_panes(
    transport: &dyn Transport,
    filter: Option<&Regex>,
) -> Result<Vec<PaneInfo>>;
```

**Must address P10**: Add a timeout (default 10s) on all exec channels for tmux commands.

**Format strings** are defined as constants, e.g.:

```rust
const LIST_SESSIONS_FMT: &str =
    "#{session_name}\t#{session_id}\t#{session_created}\t#{session_attached}\t#{session_windows}\t#{session_group}";
const LIST_PANES_FMT: &str =
    "#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{pane_title}\t#{pane_current_command}\t#{pane_pid}\t#{pane_width}\t#{pane_height}\t#{pane_active}";
```

### `capture.rs`

Pane content capture via `tmux capture-pane`.

```rust
/// Capture the visible content of a single pane.
/// Runs: tmux capture-pane -p -t <target>
pub async fn capture_pane(transport: &dyn Transport, target: &PaneAddress) -> Result<String>;

/// Capture with scrollback history.
/// Runs: tmux capture-pane -p -t <target> -S <start> -E <end>
/// start/end are line numbers; negative values reach into scrollback buffer.
/// Example: start=-1000, end=-1 captures last 1000 lines of scrollback.
pub async fn capture_pane_history(
    transport: &dyn Transport,
    target: &PaneAddress,
    start: i32,
    end: i32,
) -> Result<String>;

/// Capture all panes in a session. Calls capture_pane for each pane found via list_panes.
/// Returns a map of pane address → visible content.
pub async fn capture_session(
    transport: &dyn Transport,
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

Tmux control: session lifecycle, sending input, and managing session/window metadata.

```rust
// --- Session Lifecycle ---

/// Create a new detached tmux session.
/// Runs: tmux new-session -d -s <name> [-n <window_name>] [<command>]
pub async fn create_session(
    transport: &dyn Transport,
    name: &str,
    window_name: Option<&str>,
    command: Option<&str>,
) -> Result<()>;

/// Kill a tmux session and all its windows/panes.
/// Runs: tmux kill-session -t <name>
pub async fn kill_session(transport: &dyn Transport, name: &str) -> Result<()>;

// --- Input ---

/// Send a KeySequence to a pane. Handles the split between literal text (-l)
/// and special keys (no -l) automatically.
pub async fn send_keys(
    transport: &dyn Transport,
    target: &PaneAddress,
    keys: &KeySequence,
) -> Result<()>;

/// Convenience: send literal text (no special keys, no Enter appended).
/// Equivalent to: tmux send-keys -l -t <target> '<escaped_text>'
pub async fn send_text(
    transport: &dyn Transport,
    target: &PaneAddress,
    text: &str,
) -> Result<()>;

/// Rename a tmux session.
/// Runs: tmux rename-session -t <current> <new>
pub async fn rename_session(
    transport: &dyn Transport,
    current_name: &str,
    new_name: &str,
) -> Result<()>;

/// Rename a window.
/// Runs: tmux rename-window -t <session>:<index> <new_name>
pub async fn rename_window(
    transport: &dyn Transport,
    session: &str,
    window_index: u32,
    new_name: &str,
) -> Result<()>;
```

**Tmux socket selection**: All tmux commands generated by this module must prepend the
socket flag when a `TmuxSocket` is configured on the target. For example, if
`TmuxSocket::Name("myserver")` is set, `tmux list-sessions` becomes
`tmux -L myserver list-sessions`. This is handled by a shared helper that all command
builders in `control.rs`, `discovery.rs`, `capture.rs`, and `pipe.rs` use:

```rust
fn tmux_prefix(socket: Option<&TmuxSocket>) -> String {
    match socket {
        None => "tmux".to_string(),
        Some(TmuxSocket::Name(n)) => format!("tmux -L {}", shell_escape(n)),
        Some(TmuxSocket::Path(p)) => format!("tmux -S {}", shell_escape(p)),
    }
}
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

Multi-target management. See the `Fleet` type in [Core Abstractions](#fleet--multi-target-pool)
for the full API.

Internal responsibilities:
- Holds a `HashMap<String, HostHandle>` keyed by target alias (or `host:port` / `"localhost"`)
- Localhost targets are created with `LocalTransport` (no connection step)
- SSH targets are connected concurrently via `tokio::JoinSet`
- Tracks per-target status: `Disconnected`, `Connecting`, `Connected`, `Monitoring`, `Error(String)`
- Routes on-demand operations (create, kill, capture, send-keys, list) to the correct target
- Owns the `shutdown` watch channel; `shutdown()` signals all targets

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

Internally, `HostHandle` holds a `Box<dyn Transport>` and delegates to the function-level
APIs in `discovery`, `capture`, `control`, `pipe`, and `monitor`. It is the composition
root for per-target operations.

```rust
pub struct HostHandle {
    transport: Box<dyn Transport>,
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
    /// Uses a dedicated transport exec call (not the monitor channel).
    pub async fn setup(transport: &dyn Transport, panes: &[PaneAddress]) -> Result<Self>;

    /// Detach all pipe-panes and remove FIFO files.
    pub async fn cleanup(&self, transport: &dyn Transport) -> Result<()>;
}

impl Drop for PipeManager {
    // Best-effort: log warning if cleanup was not called explicitly.
    // Cannot do async cleanup in Drop, so this is advisory only.
}
```

**Must address P4**: `cleanup()` must be called on shutdown. The library exposes cleanup
as an explicit async method; it does NOT install signal handlers itself (see DC10 — signal
handling is the binary's responsibility, keeping the library embeddable in MCP/service contexts).

**Cleanup sequence** (triggered by caller via `shutdown()` or `PipeManager::cleanup()`):
1. `tmux pipe-pane -t <pane>` (no `-o`) for each pane — detaches the pipe
2. `rm -f /tmp/motlie_pipe_%<id>` for each FIFO (local) or via transport exec (remote)
3. Close transport channels

### `monitor.rs`

The core event loop. Reads the multiplexed output stream and evaluates rules.

```rust
pub struct OutputMonitor { /* rules, cooldown state, shell channel */ }

impl OutputMonitor {
    pub async fn run(
        &mut self,
        transport: &dyn Transport,
        panes: &[PaneAddress],
        rules: &[TriggerRule],
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()>;
}
```

**Must address P3**: Rule evaluation replaces the hardcoded `contains('>')` check.

**Must address P6**: Monitoring uses a dedicated channel — a control mode session
(`tmux -C attach`, see DC10) as the primary strategy, or `tail -qf` on pipe files as
fallback. Action dispatch (send-keys) uses separate exec channels routed through a
bounded queue (see [DC4](#dc4-action-dispatch-channel-strategy)).

**Must address P9**: Failed send-keys or malformed lines must be logged at `warn` level,
not silently dropped.

**Stream parsing**: With control mode (DC10), the monitor parses `%output %<pane_id> <data>`
frames — structured, unambiguous, and keyed on `#{pane_id}` per DC1. For the pipe-pane
fallback, `pipe-pane` output is prefixed with `%<pane_id>` (set at attach time), and the
monitor parses on that prefix directly — no filename decoding required.

---

## Key Design Decisions

### DC1: Pane Identity and Addressing

**Decision**: Use tmux `#{pane_id}` (e.g., `%12`) as the authoritative identifier for
FIFO naming, stream attribution, and internal keying. Retain `session:window.pane` in
`PaneAddress` as display metadata and user-facing targeting only.

**Rationale**: `#{pane_id}` is a stable, unique, tmux-assigned identifier that does not
change when sessions or windows are renamed or moved. Using it as the internal key
eliminates the need for lossy filename encoding of session names (the original P2 bug)
and simplifies stream parsing — the monitor keys on `%<id>` directly rather than
decoding hex-encoded filenames.

**FIFO format** (if FIFO strategy is used): `/tmp/motlie_pipe_%<id>` (e.g., `/tmp/motlie_pipe_%12`)

**PaneAddress** retains human-readable fields for display and `tmux send-keys -t` targeting:
```rust
pub struct PaneAddress {
    pub pane_id: String,       // authoritative: "%12" — used for FIFO names, stream keys
    pub session: String,       // display/targeting: session name
    pub window: u32,           // display/targeting: window index
    pub pane: u32,             // display/targeting: pane index
}
```

**Alternative rejected**: Hex-encoding session names in filenames — adds complexity and
is still fragile under rename. `#{pane_id}` is simpler and correct by construction.

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

**Decision**: Option A (separate exec per action) with a per-host bounded dispatch queue.

**Option A** (selected): Open a new transport exec call per action (`tmux send-keys -t ...`),
routed through a per-host bounded `tokio::sync::mpsc` channel with a configurable
concurrency semaphore.
- Pro: Simple, isolated, no shared state, ordering guaranteed by queue, backpressure
  via bounded channel capacity
- Con: For SSH, channel open latency (~50ms per action) — acceptable for typical
  automation rates. For localhost, subprocess spawn overhead is minimal.

**Option B** (rejected for v1): Persistent control shell channel.
- Pro: Lower latency
- Con: Shared channel parsing complexity (same issue as prototype P6)

**Implementation**: Each `HostHandle` owns a `tokio::sync::mpsc::Sender<ActionRequest>`
with bounded capacity (default: 64). A background task drains the queue and executes
actions via `transport.exec()`, with a `tokio::sync::Semaphore` limiting concurrent
in-flight dispatches (default: 4 for SSH, 8 for localhost). This gives ordering,
backpressure, and isolation without persistent-shell parsing.

### DC5: Multi-Host Concurrency

**Decision**: One `tokio::spawn` task per host, independent lifecycle.

**Rationale**: Hosts are independent. A connection failure on host A must not affect host B.
Each task runs the full lifecycle: connect → discover → pipe → monitor → cleanup.

**Shared state**: Only the rule set and shutdown signal are shared across hosts. Per-host
state (connection, panes, pipes, cooldown timers) is task-local.

### DC6: Local vs SSH Transport

**Decision**: A `Transport` trait abstracts command execution. Two implementations:
`LocalTransport` (localhost, subprocess-based) and `SshTransport` (remote, russh-based).

**Rationale**: The prototype assumes SSH for everything, but localhost tmux is a primary
use case (local development, CI, single-machine automation). Forcing SSH to localhost
adds unnecessary complexity (SSH server requirement, key management, latency). A trait
abstraction lets all downstream modules (`discovery`, `capture`, `control`, `pipe`,
`monitor`) be transport-agnostic.

**LocalTransport specifics**:
- `exec()`: spawns `tokio::process::Command`, captures stdout, respects timeout
- `open_shell()`: spawns a persistent shell process with piped stdin/stdout
- No connection step; always available
- FIFO paths are local filesystem paths (no remote cleanup needed)

**SshTransport specifics**:
- `exec()`: opens SSH exec channel, captures stdout
- `open_shell()`: opens PTY channel with shell
- Requires connection + auth before use
- Host key verification per DC2

**MockTransport** (for testing): Returns canned `exec()` responses and canned streaming
data from `open_shell()`. Built into the library, not behind a feature flag, so downstream
consumers can also test their integrations.

### DC7: Capture-Pane vs Stream Monitoring

**Decision**: Use `capture-pane -p` for on-demand snapshots. For continuous monitoring,
use tmux control mode as primary (DC10) and `pipe-pane` as fallback only.

**Rationale**:
- `capture-pane -p` returns the current visible content (and scrollback with `-S`/`-E`)
  as a single snapshot. It is stateless, idempotent, and requires no setup. Ideal for
  "what is this pane showing right now?" queries.
- Continuous monitoring uses control mode (`tmux -C attach`, DC10) which provides
  structured `%output %<pane_id>` framing without file lifecycle management.
- `pipe-pane` is retained as a fallback for environments where control mode is
  unavailable (see DC10 fallback section).

**Callers choose the mode**:
- `capture_pane()` / `capture_session()` → snapshot via `capture-pane`
- `start_monitoring()` → continuous via control mode (primary) or pipe-pane (fallback)

### DC8: Key Escaping Strategy

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

### DC9: Fleet-Level Host Addressing

**Decision**: Hosts are addressed by alias (if configured) or by `host:port` string.
All `Fleet` methods that target a specific host accept `&str` and resolve internally.

**Rationale**: Users operating across multiple machines want to say "capture pane on
`web-server`", not remember `10.0.1.42:22`. Aliases are optional — single-host use
cases just use the host string directly.

**Conflict resolution**: If two hosts share an alias, config validation fails at load time.

### DC10: Monitoring Strategy — Control Mode vs Pipe-Pane

**Decision**: Use tmux control mode (`tmux -C attach`) as the primary monitoring strategy
for v1. Pipe-pane with file sinks is a documented fallback for environments where control
mode is unavailable or insufficient.

**Decision matrix**:

| Criterion | Control Mode (`tmux -C`) | Pipe-Pane + FIFO/File |
|-----------|--------------------------|----------------------|
| Output framing | Structured: `%output %<pane_id> <data>` | Per-pane file/FIFO, keyed by `pane_id` |
| Pane identity | Native `%<pane_id>` in protocol | `pane_id` prefix at attach time (DC1) |
| `/tmp` artifacts | None | FIFOs or log files per pane |
| Cleanup on crash | Nothing to clean up | Orphaned FIFOs/files, dangling pipe-pane |
| Interleaving (OC2) | Not possible — framed protocol | Possible with `tail -qf` on multiple files |
| Backpressure (OC1) | Buffered by tmux control mode | FIFO blocks writer; file grows unbounded |
| Tmux version req | tmux >= 1.8 (control mode) | pipe-pane `-o`: needs version testing (OC4) |
| Multi-pane | One connection per session (see below) | One FIFO + pipe-pane per pane |
| Complexity | Lower — no file lifecycle | Higher — FIFO/file creation, rotation, cleanup |

**Session scope**: Control mode `%output` notifications are scoped to the attached session,
not host-wide. A single `tmux -C attach -t <session>` only receives output from panes in
that session. To monitor multiple sessions, `HostHandle` opens one control-mode connection
per target session and aggregates their streams internally. This is still simpler than
per-pane pipe-pane setup, and the number of sessions is typically small (single digits).

**Rationale**: Control mode eliminates OC1 (FIFO blocking), OC2 (interleaving), and most
of the pipe lifecycle complexity. It provides structured output with native `#{pane_id}`
attribution, aligning with DC1. The `%output` notifications include pane ID, eliminating
the need for filename-based identity.

**Fallback**: Pipe-pane with append-file sink (`pipe-pane -o 'cat >> file'` + `tail -f`)
is retained as an option for scenarios where control mode is insufficient (e.g., very old
tmux, or when monitoring panes across multiple tmux servers on the same host).

### DC11: Separation of Library and Binary

**Decision**: `libs/tmux` is a pure library. CLI binary lives in `bins/tmux-automator/`.

**Rationale**: Follows the existing workspace convention (`libs/mcp` + `examples/mcp/`).
The library exposes `Fleet`, `HostHandle`, `TmuxAutomatorConfig`, and all operation APIs.
The binary handles CLI parsing, config file loading, signal handling, and tracing
initialization. The library is also consumable by MCP tools or other programmatic callers.

### DC12: Output Sink Pipeline Architecture

**Decision**: Captured pane output is distributed to consumers via an async fan-out bus
(`OutputBus`) that delivers `PaneOutput` to independently-running sink tasks. Each sink
receives its own bounded channel and manages its own batching, buffering, and timing
internally. Sinks are filtered via composable `SinkFilter` structs (OR across filters,
AND within fields). Sinks may initiate actions on tmux entities via an `ActionHandle`
that routes requests through the existing per-host action dispatch queue (DC4).

**Rationale**:
- **Decoupled latency**: A slow LLM sink must never block a fast stdio sink. Per-sink
  channels with independent tasks ensure this.
- **Sink-owned batching**: The bus delivers individual `PaneOutput` events; sinks decide
  when/how to batch. A stdio sink flushes immediately. An LLM sink accumulates until a
  token budget or timeout. This keeps the bus simple and avoids a "one size fits all"
  batching policy.
- **Composable filtering**: `SinkFilter` fields (host, session, window, pane) are regex
  patterns ANDed together. Multiple filters per sink are ORed. This allows targeting
  "all panes on host-a" OR "session:build on any host" with a single sink registration.
- **Action loop**: `ActionHandle` gives sinks a way to respond to output (e.g., LLM
  decides to send keys). Actions flow through the existing bounded queue and semaphore,
  preserving ordering and backpressure guarantees from DC4.
- **LLM feedback loop out of scope**: The `ActionHandle` API is in scope; the logic that
  decides *what* action to take (LLM inference, prompt construction) is out of scope for
  `libs/tmux`. Consumers build that on top.

**Alternatives considered**:
- Bus-level batching with configurable window: Rejected — forces all sinks to the same
  cadence and complicates the bus with timer logic.
- Callback-based sinks (no channels): Rejected — a slow callback blocks the bus loop.
- Shared `Arc<Mutex<Vec<PaneOutput>>>` polling: Rejected — wastes CPU, no backpressure.

---

## Open Concerns

These require resolution before or during implementation. They are not blockers for Phase 1
but must be addressed before the library is used in production.

### OC1: FIFO Reliability Under Load

**Largely mitigated by DC10** — if control mode is adopted as the primary monitoring
strategy, FIFOs are not used and this concern does not apply.

For the pipe-pane fallback path: `tmux pipe-pane` writing to a FIFO can block the
monitored pane if the reader falls behind. Under high output volume, this stalls the
tmux pane.

**Resolution for fallback**: Default to append-file sink (`pipe-pane -o 'cat >> file'`
+ `tail -f`) rather than FIFOs. This avoids writer blocking. File rotation (truncate
when exceeding configurable size, default 10MB) prevents unbounded growth. FIFO mode
available as opt-in for latency-sensitive use cases where backpressure is acceptable.

### OC2: Output Interleaving

**Eliminated by DC10** — control mode provides framed `%output` messages with pane ID
attribution. No interleaving possible.

For the pipe-pane fallback path: `tail -qf` on multiple files can interleave partial
lines from different panes. Mitigation: use line-buffered output (`stdbuf -oL`) if
available on the remote host. Document as a known limitation of the fallback path.

### OC3: SSH Agent Availability

The prototype requires a running `ssh-agent` with loaded keys. If the agent is unavailable
or has no matching keys, the error message should be actionable.

**Proposal**: On auth failure, enumerate the attempted key fingerprints and suggest
`ssh-add` if no keys were found.

### OC4: Tmux Version Compatibility

`tmux pipe-pane`, `list-panes -F` format strings, and control mode behavior vary across
tmux versions. The minimum supported version must be determined empirically, not asserted.

**Proposal**: Runtime detection via `tmux -V` during the discovery phase. The library
validates that required features are available (control mode `%output` framing,
`capture-pane -p`, `pipe-pane -o`, `#{pane_id}` format variable) and returns a clear
error if the detected version lacks them. Phase 4 CI will build a compatibility test
matrix against tmux 2.x, 3.x, and latest to determine the actual minimum version.

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

**Proposal**: The `Transport` trait in `transport.rs` already supports this (see DC6).
`MockTransport` returns canned responses for `exec()` and `open_shell()`. It is included
in the library (not behind a feature flag) so downstream consumers can also use it.

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

### Phase 1: Types, Transport, and On-Demand Operations (Localhost)

**Goal**: Establish the core types, `LocalTransport`, and localhost on-demand capabilities
(create, kill, list, capture, send-keys, rename). No SSH or monitoring yet.

**Tasks**:
1. Create `libs/tmux/` workspace member with `Cargo.toml`
2. Implement `types.rs`: `PaneAddress` with `pane_id` (`%<id>`) as authoritative key
   and `session:window.pane` as display metadata (per DC1), `SessionInfo`, `WindowInfo`,
   `PaneInfo`
3. Implement `keys.rs`: `KeySequence`, `SpecialKey`, `{...}` parser, tmux command rendering
4. Implement `transport.rs`: `Transport` trait, `LocalTransport` (subprocess-based),
   `MockTransport` (canned responses for testing)
5. Implement `discovery.rs`: `list_sessions()`, `list_windows()`, `list_panes()` with
   format string constants and regex filtering
6. Implement `capture.rs`: `capture_pane()`, `capture_pane_history()`, `capture_session()`
7. Implement `control.rs`: `create_session()`, `kill_session()`, `send_keys()`,
   `send_text()`, `rename_session()`, `rename_window()` with shell escaping (addresses OC5)
8. Implement `host.rs`: `HostHandle` wiring all of the above
9. Unit tests: `PaneAddress` identity (pane_id as key, display target as metadata),
   `KeySequence` parsing + rendering, shell escaping with adversarial inputs,
   `MockTransport`-based tests for all operations

**Deliverable**: Library that operates on localhost tmux and supports all on-demand
operations. A caller can create/kill sessions, list sessions, capture pane content, send
input, and rename sessions — all without SSH.

**Acceptance criteria**:
- `PaneAddress` uses `pane_id` as key; `session:window.pane` roundtrips as display metadata
- `KeySequence::parse("hello{Enter}{C-c}")` produces correct tmux commands
- `send_text()` with input `"; rm -rf /; echo "` does not execute injected commands
- `create_session("test")` + `list_sessions()` shows the new session
- `kill_session("test")` + `list_sessions()` no longer shows it
- `list_sessions()` returns `Vec<SessionInfo>` with all fields populated
- All operations work via `MockTransport` in unit tests

### Phase 2a: SSH Transport + Minimal Monitoring (Vertical Slice)

**Goal**: Add `SshTransport` for remote hosts. Implement a thin monitoring vertical slice:
single-target monitor with one `SendKeys` action type and explicit shutdown API. No full
rule engine, no reconnection, no config deserialization yet.

**Tasks**:
1. Implement `SshTransport` in `transport.rs`: russh-based `exec()` and `open_shell()`,
   host key verification (fixes P1), configurable timeout (fixes P10)
2. Implement `monitor.rs`: control mode parser (`%output %<pane_id> <data>`), single
   hardcoded-pattern detection, `SendKeys` action dispatch via bounded queue (DC4),
   `MonitorHandle` with explicit `shutdown()` API
3. Implement `pipe.rs`: fallback `PipeManager` with file-sink default (fixes P4)
4. Wire monitoring into `HostHandle::start_monitoring()` for localhost and SSH
5. Unit tests: `SshTransport` via mock SSH server or `MockTransport`, monitor loop with
   canned control-mode output, action dispatch ordering

**Deliverable**: Library that monitors panes on localhost or one SSH host, detects a
pattern, and sends a response. Shutdown is explicit (caller-initiated, no signal handling
in library).

**Acceptance criteria**:
- All Phase 1 operations work identically over `SshTransport`
- Monitoring starts on localhost, detects a pattern, sends configured response
- `monitor_handle.shutdown()` cleanly stops monitoring and cleans up
- Localhost monitoring works without any SSH configuration

### Phase 2b: Full Rule Engine + Reconnection + Config

**Goal**: Layer the full configurable rule engine, cooldown, reconnection, and config
deserialization on top of the stable 2a monitoring loop.

**Tasks**:
1. Implement `config.rs`: `TriggerRule` (string patterns), `CompiledRule`, `Action`,
   `ReconnectPolicy`, `TmuxAutomatorConfig` with serde deserialization,
   `HostTarget::Local` and `HostTarget::Ssh` variants
2. Implement rule compilation: `TriggerRule::compile() -> Result<CompiledRule>` with
   user-facing error messages including rule name and pattern context
3. Expand `monitor.rs`: multi-rule evaluation, per-pane cooldown timers, `Log` action type
4. Reconnection logic with exponential backoff in `HostHandle` (SSH targets only)
5. Unit tests: rule evaluation, cooldown behavior, config deserialization, compile errors

**Deliverable**: Full monitoring with configurable rules, cooldown, and SSH reconnection.

**Acceptance criteria**:
- Config file (TOML) deserializes into `TmuxAutomatorConfig` with compiled rules
- Invalid regex in config produces error with rule name context
- Rules with cooldown do not fire more than once per cooldown period
- Reconnection resumes monitoring after a simulated SSH disconnect

### Phase 2c: Output Sink Pipeline

**Goal**: Implement the async output distribution pipeline so captured pane output can be
routed to multiple consumers (sinks) with independent latency characteristics.

**Tasks**:
1. Implement `sink.rs`: `PaneOutput` struct, `OutputSink` trait, `SinkFilter` /
   `CompiledSinkFilter`, `ActionHandle` / `ActionRequest` / `ActionTarget` / `SinkAction`
2. Implement `OutputBus`: registration API (`register(sink, filters, channel_capacity)`),
   fan-out loop that matches `PaneOutput` against compiled filters and dispatches to
   per-sink channels, graceful shutdown with `flush()` on all sinks
3. Implement `StdioSink`: default reference implementation that writes `PaneOutput` to
   stdout with `[host:pane_target]` prefix, immediate flush, no batching
4. Wire `OutputBus` into `monitor.rs`: control mode parser feeds `PaneOutput` into the
   bus alongside rule evaluation
5. Wire `ActionHandle` into `HostHandle`: sink-initiated actions route through the
   existing per-host action dispatch queue (DC4)
6. Integrate into `Fleet`: `Fleet::output_bus()` accessor, sink registration before
   `start_monitoring()`
7. Unit tests: `SinkFilter` matching logic (AND within fields, OR across filters),
   `OutputBus` fan-out to multiple mock sinks, backpressure behavior when sink channel
   is full, `ActionHandle` delivery

**Deliverable**: Library routes captured output to registered sinks asynchronously. A
`StdioSink` prints output in real-time. Custom sinks (TUI, LLM) can be registered by
consumers.

**Acceptance criteria**:
- `OutputBus` delivers `PaneOutput` to 3 registered sinks concurrently
- A slow sink (simulated 500ms delay) does not block other sinks
- `SinkFilter { host: Some("web.*"), session: None, window: None, pane: None }` matches
  all panes on hosts matching `web.*`
- Two filters ORed together match the union of their targets
- `StdioSink` produces readable output with host and pane context
- `ActionHandle::send_keys()` delivers the action through the host dispatch queue
- `OutputBus::shutdown()` calls `flush()` on all sinks before returning

### Phase 3: Multi-Target Fleet + CLI

**Goal**: Multi-target support via `Fleet` and a usable CLI binary.

**Tasks**:
1. Implement `fleet.rs`: `Fleet` with concurrent connect, target lookup by alias (DC9),
   per-target status tracking, aggregate `start_monitoring()` and `shutdown()`
2. Signal handling in the CLI binary via `tokio::signal` for SIGINT/SIGTERM (fixes P5),
   wired to `fleet.shutdown()`. The library does NOT install signal handlers (DC11).
3. Create `bins/tmux-automator/` with `clap` CLI supporting subcommands:
   - `create-session <name> [--host <alias>] [--window-name <name>] [--command <cmd>]`
   - `kill-session <name> [--host <alias>]`
   - `list-sessions [--host <alias>]` — list sessions on one or all targets
   - `list-panes [--host <alias>] [--filter <regex>]` — list panes
   - `capture <session:window.pane> [--host <alias>] [--history <lines>]` — dump pane
   - `send <session:window.pane> <input> [--host <alias>]` — send keys
   - `rename-session <old> <new> [--host <alias>]` — rename
   - `monitor [--config <path>]` — start continuous monitoring
4. Config file support (TOML) with CLI flag overrides
5. JSON and text log output modes
6. Per-target tracing spans with alias labels

**Deliverable**: Multi-target CLI tool and library. Single process operating across
localhost and N remote hosts.

**Acceptance criteria**:
- `fleet.connect_all()` connects to 3 targets concurrently; one failure does not block others
- CLI `list-sessions --host web-server` returns sessions from the aliased host
- CLI `create-session build --host localhost` creates a local session
- CLI `capture myapp:0.0 --host db-server --history 500` returns scrollback content
- CLI `kill-session build` terminates the session
- `monitor --config rules.toml` starts monitoring on all configured targets

### Phase 4: Hardening + Testing

**Goal**: Production readiness.

**Tasks**:
1. Expand `MockTransport` coverage for integration tests (OC6)
2. Tmux version detection and compatibility check (OC4)
3. Actionable SSH agent error messages (OC3)
4. FIFO-vs-file investigation and decision (OC1)
5. Line interleaving mitigation (OC2)
6. End-to-end test with Docker (SSH + tmux)
7. Document minimum tmux version, known limitations, and performance characteristics

**Deliverable**: Library with test coverage, documented limitations, and CI integration.

### Phase 5 (Future): TUI Interface

**Goal**: Terminal UI for interactive multi-target management.

**Technology**: [ratatui](https://ratatui.rs/) — a Rust library for building terminal
user interfaces.

**Not in current scope**. Listed here for planning continuity. The TUI will consume
the `Fleet` API from `libs/tmux` and provide:
- Live pane content display via `TuiSink` registered with `OutputBus` (Phase 2c)
- Session/window/pane tree navigation across targets
- Interactive send-keys input via `ActionHandle`
- Monitoring rule status and trigger history
- Host connection status dashboard

The `TuiSink` implementation lives in the binary (`bins/tmux-automator/`), not in
`libs/tmux`, consistent with DC11. It registers with the library's `OutputBus` and
manages its own rendering cadence (e.g., 60fps batching).

This phase depends on Phases 1-3 and 2c being complete and stable.

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
