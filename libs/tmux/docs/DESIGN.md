# Tmux Multi-Target Automator Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-03-08 | `Target::exec()` shell compatibility: document `$?` (POSIX) vs `$status` (fish) with shell detection. Replace `ActionTarget` enum with `TargetAddress` in `ActionRequest` for unified type consistency (DC16). | DC19, Core Abstractions, Output Sink Pipeline |
| 2026-03-08 | Address codex review round 8: clarify remaining dynamic types (`Arc<dyn Any>`, `Pin<Box<dyn Future>>`) in SinkKind docs and changelog; fix integration diagram to use `SinkKind` wrappers. | Core Abstractions, Output Sink Pipeline |
| 2026-03-08 | Address codex review round 7: `CallbackSink` uses explicit `Arc<dyn Any>` state instead of closure capture, `on_output` is synchronous; fix stale examples (`SubstringMatcher` → `MatcherKind::Substring`, `JoinedSink` uses `SinkKind`); align DC6/OC6/Phase 1 to `TransportKind` enum; fix `host.rs` module spec `session_monitors` to `RwLock`; fix "SinkKind trait" → enum; fix DC14 matcher names. | Core Abstractions, Output Sink Pipeline, Module Specs, DC6, DC14, OC6, Phase 1, Phase 2c |
| 2026-03-08 | Phase 3 CLI: noun-verb subcommand pattern (`session list`, `target capture`, `monitor start`) replacing flat hyphenated commands. `target` noun reflects unified Target type (DC16). | Phase 3 |
| 2026-03-08 | Explicit FIFO cleanup on monitoring stop: `SessionMonitorHandle::shutdown()` and `stop_monitoring_session()` call `PipeManager::cleanup()` when pipe-pane fallback is active (P4). | Core Abstractions, Module Specs, DC13 |
| 2026-03-08 | Static dispatch on hot paths: `Transport` → `TransportKind` enum, `ContentMatcher` → `MatcherKind` enum, `OutputSink` → `SinkKind` enum, `LabelFormat::Custom` → `fn` pointer. Remaining dynamic types: `CallbackSink.state` (`Arc<dyn Any>`), `on_flush` (`Pin<Box<dyn Future>>` — Rust async limitation). | Core Abstractions, Output Sink Pipeline, Module Specs, DC14 |
| 2026-03-08 | Added `Target::exec()` for structured command execution with sentinel-based capture (DC19). `ExecOutput` type with stdout and exit code. | Core Abstractions, DC19 |
| 2026-03-08 | Added workstreams to `Fleet`: `bind()`, `find()`, `workstreams()` for named (host, target) bindings. Future DC18 outlines composite workstreams composing with sinks/streams. | Core Abstractions |
| 2026-03-08 | `PaneOutput` → `TargetOutput`: source identified by `TargetAddress` instead of flat pane fields. Generalizes to session-only hosts. `SourceLabel` uses `TargetAddress`. | Output Sink Pipeline, DC12 |
| 2026-03-08 | `MonitorHandle` lookup API uses `&Target`/`&TargetSpec` instead of raw strings; `start/stop_monitoring_session` accept `&Target`. Rationale for keeping `SessionMonitorHandle` name (DC10 session-scoped constraint). | Core Abstractions, DC13 |
| 2026-03-08 | Added `TargetSpec` type with builder for `HostHandle::target()` — replaces raw string parameter | Core Abstractions, DC17 |
| 2026-03-08 | Unified `Target` type replacing `SessionHandle`/`PaneHandle`/`WindowHandle` hierarchy (DC16, DC17). `HostHandle` slimmed to host-level only. Fixed stale references throughout. | Core Abstractions, Architecture, DC16, DC17, DC13, Module Specs, Phase 1 |
| 2026-03-08 | Added session-scoped operations on `SessionMonitorHandle` via `Deref<Target=Target>` (DC16) | Core Abstractions, DC16 |
| 2026-03-08 | Added `JoinedStream` for multi-source consolidated views (DC15) | Output Sink Pipeline, DC15 |
| 2026-03-08 | Added `ContentMatcher` trait for pluggable text matching (DC14) | Core Abstractions, Output Sink Pipeline, DC14 |
| 2026-03-08 | Added `sample_text()` API with `ScrollbackQuery` for on-demand scrollback sampling | Core Abstractions |
| 2026-03-08 | Added granular monitoring lifecycle — fleet/host/session levels (DC13) | DC13 |
| 2026-03-08 | Added output sink pipeline — `OutputBus`, `SinkFilter`, `SinkKind` trait (DC12) | Output Sink Pipeline, DC12 |
| 2026-03-08 | Addressed codex review rounds 1–4: DC10 control mode fixes, ActionHandle ownership, integration diagram, version pinning | DC10, Output Sink Pipeline, DC1 |
| 2026-03-07 | Added `TmuxSocket` selector for non-default tmux servers | Core Abstractions |
| 2026-03-07 | Added localhost transport, session lifecycle, TUI roadmap (Phase 5) | Core Abstractions, Module Specs, Implementation Phases |
| 2026-03-07 | Added prototype source and Gemini link for traceability | Prototype Reference |
| 2026-03-07 | Initial design document | All |

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
6. Executes shell commands in panes and captures structured output (exit code, stdout)
7. Manages session metadata (rename sessions/windows)
8. Names logical workstreams across hosts and targets for domain-meaningful addressing
9. Attaches output pipes for continuous monitoring
10. Evaluates configurable trigger rules against pane output
11. Executes actions (send-keys, notify, log) when rules match
12. Reconnects automatically on failure (SSH targets)

### Scope

- **In scope**: Localhost tmux (direct execution), SSH transport for remote hosts,
  multi-host connection pool, tmux session creation and termination, session/window/pane
  listing, pane content capture, structured command execution (exec with exit code),
  remote input with escaping, session metadata management, named workstreams,
  output sink pipeline with pluggable content matching, pipe-based output monitoring,
  rule-based automation, structured logging, CLI binary
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
│   ├── transport.rs        # TransportKind enum + LocalTransport + SshTransport
│   ├── host.rs             # HostHandle: per-host facade for all tmux operations
│   ├── fleet.rs            # Fleet: multi-host pool, dispatch, aggregate status
│   ├── discovery.rs        # Session/window/pane listing, filter, PaneAddress type
│   ├── capture.rs          # Pane content capture (capture-pane) and scrollback dump
│   ├── control.rs          # Session lifecycle, send-keys with escaping, rename
│   ├── pipe.rs             # PipeManager: FIFO lifecycle, pipe-pane attach/detach
│   ├── monitor.rs          # OutputMonitor: stream parsing, rule evaluation, dispatch
│   ├── matcher.rs          # MatcherKind enum + built-in matchers (Regex, Substring, etc.)
│   ├── sink.rs             # SinkKind enum, SinkFilter, TargetOutput, OutputBus
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

**Two usage modes coexist across `HostHandle` and `Target`**:

1. **On-demand operations**: Host-level discovery and session creation via `HostHandle`;
   entity-level I/O (capture, send-keys, rename, kill) via `Target`. Called directly by
   the consumer (CLI command, MCP tool, API call). These execute via the transport
   (local subprocess or SSH exec channel) and return immediately.

2. **Continuous monitoring**: Pipe setup + event loop — long-running, spawned as a
   background task. Started via `HostHandle::start_monitoring()` (host-wide) or
   `Target::start_monitoring()` (single session). Uses dedicated shell/PTY channel.
   Rule-triggered actions dispatch through the same `Control` module as on-demand operations.

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

    // --- Output Bus ---

    /// Access the fleet's OutputBus for sink registration.
    /// Sinks should be registered before start_monitoring() so they
    /// receive output from the start.
    pub fn output_bus(&mut self) -> &mut OutputBus;

    // --- Monitoring (fleet-wide) ---

    /// Start monitoring on all connected hosts (spawns background tasks).
    pub async fn start_monitoring(&self, rules: &[TriggerRule]) -> Result<()>;

    /// Shutdown: stop monitoring on all hosts, cleanup pipes, close connections.
    pub async fn shutdown(&self) -> Result<()>;

    // --- Monitoring (per-host) ---

    /// Start monitoring on a single host by name/alias.
    /// The host must be connected. Monitoring sessions are discovered
    /// automatically (or filtered by the host's pane_filter config).
    pub async fn start_monitoring_host(
        &self,
        host: &str,
        rules: &[TriggerRule],
    ) -> Result<()>;

    /// Stop monitoring on a single host. Cleans up control-mode connections
    /// and pipe-pane state for this host only. Other hosts continue unaffected.
    pub async fn stop_monitoring_host(&self, host: &str) -> Result<()>;

    // --- Workstreams (named bindings) ---

    /// Bind a user-defined name to a (host, target) pair. The name must be
    /// unique within the fleet. Returns an error if the name is already bound.
    ///
    /// Workstreams give callers a stable, domain-meaningful name for a
    /// specific entity across the fleet — e.g., "build-pipeline" for
    /// the build session on web-1, or "db-migration" for a pane on db-1.
    pub fn bind(
        &mut self,
        name: &str,
        host: &HostHandle,
        target: Target,
    ) -> Result<()>;

    /// Remove a workstream binding by name.
    pub fn unbind(&mut self, name: &str) -> Result<()>;

    /// Look up a workstream by name. Returns the host alias and Target.
    pub fn find(&self, name: &str) -> Option<(&str, &HostHandle, &Target)>;

    /// List all workstream bindings.
    pub fn workstreams(&self) -> Vec<WorkstreamEntry>;
}

/// A named binding of a user-defined name to a (host, target) pair.
pub struct WorkstreamEntry {
    pub name: String,
    pub host_alias: String,
    pub target: Target,
}
```

**Design rationale**: Callers should not need to manage individual connections or reason
about which target a session lives on. `Fleet` provides the "I have N targets, operate
on them" abstraction. For localhost-only use, `Fleet` with one local target works
identically — there is no separate single-target API. Localhost is always available
without SSH configuration.

**Workstreams** give callers a domain-meaningful vocabulary that decouples intent from
infrastructure. Instead of `fleet.host("web-1")?.session("build")`, a caller says
`fleet.find("build-pipeline")` — the mapping from name to (host, target) is established
once and referenced everywhere. This is especially useful when the same logical
workstream spans config, monitoring rules, sink filters, and CLI commands.

**Usage**:

```rust
let host = fleet.host("web-1").unwrap();
let build = host.create_session("build", None, Some("cargo build")).await?;
fleet.bind("build-pipeline", host, build.clone())?;

// Later — anywhere in the codebase
let (_, _, target) = fleet.find("build-pipeline").unwrap();
target.send_text("cargo test").await?;

// List all workstreams
for ws in fleet.workstreams() {
    println!("{}: {} on {}", ws.name, ws.target.target_string(), ws.host_alias);
}
```

**Future: workstreams as sink/stream groups** (DC18). A workstream today binds a single
name to a single (host, target). A natural extension is *composite workstreams* that
group multiple targets under one name — e.g., "deploy" spans `web-1:build`, `db-1:migrate`,
and `web-1:test`. This would compose with the existing sink pipeline:

- **`SinkFilter` by workstream**: Filter output by workstream name instead of
  individual host/session/pane fields. The bus resolves the workstream to its member
  targets at filter-match time.
- **`JoinedStream` over a workstream**: `subscribe_joined()` could accept a workstream
  name, automatically creating filters for all member targets. The joined view then
  shows the multi-party conversation for that logical workflow.
- **Workstream-scoped monitoring**: Start/stop monitoring for all targets in a
  workstream with a single call.

This extension is deferred — single-target workstreams cover the immediate need.
Composite workstreams require decisions about membership lifecycle (what happens when
a target is killed?) that should be informed by real usage patterns.

### `HostHandle` — Per-Host Entry Point

The connection to a single tmux target (localhost or remote). `HostHandle` provides
host-level operations: session discovery, creation, and host-wide monitoring. It also
serves as a factory for `Target` handles (DC16).

```rust
pub struct HostHandle { /* Arc<HostHandleInner> */ }

/// Shared internals — Target holds an Arc ref to this.
/// Uses interior mutability so &self methods on HostHandle and Target
/// can mutate monitoring state under concurrent access.
struct HostHandleInner {
    transport: TransportKind,
    config: HostTarget,
    /// RwLock for concurrent read access (monitored_sessions, find) with
    /// exclusive write access (start/stop monitoring). The lock is never
    /// held across await points — lock, mutate, release, then await.
    session_monitors: RwLock<HashMap<String, SessionMonitorHandle>>,
}

impl HostHandle {
    // --- Discovery ---

    /// List all tmux sessions on this target.
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>>;

    /// Create a new tmux session. Returns a Target at session level.
    /// Runs: tmux new-session -d -s <name> [-n <window_name>] [<command>]
    pub async fn create_session(
        &self,
        name: &str,
        window_name: Option<&str>,
        command: Option<&str>,
    ) -> Result<Target>;

    /// Get a Target for an existing session by name. Queries tmux to verify
    /// the session exists. Returns None if not found.
    pub async fn session(&self, name: &str) -> Result<Option<Target>>;

    /// Get a Target at any specificity from a `TargetSpec`.
    /// Verifies the entity exists. This is the escape hatch from
    /// raw strings (CLI args, config) to typed `Target` handles.
    pub async fn target(&self, spec: &TargetSpec) -> Result<Option<Target>>;

    // --- Host-Wide Monitoring ---

    /// Start monitoring all matching sessions on this host.
    /// Opens one control-mode connection per discovered session (DC10).
    /// Shutdown is managed internally by the HostHandle (DC13).
    pub async fn start_monitoring(
        &self,
        filter: Option<&Regex>,
        rules: &[TriggerRule],
    ) -> Result<MonitorHandle>;

    /// Stop all monitoring on this host.
    pub async fn stop_monitoring(&self) -> Result<()>;

    /// Start monitoring a single session (DC13 session-level).
    /// Accepts a session-level Target. Opens one control-mode connection.
    pub async fn start_monitoring_session(
        &self,
        target: &Target,
        rules: &[TriggerRule],
    ) -> Result<SessionMonitorHandle>;

    /// Stop monitoring a single session.
    /// Calls SessionMonitorHandle::shutdown() — tears down the control-mode
    /// connection and, if pipe-pane fallback was active, runs PipeManager::cleanup()
    /// to detach pipes and remove FIFO files. Does not affect other sessions.
    pub async fn stop_monitoring_session(&self, target: &Target) -> Result<()>;

    /// List sessions currently being monitored.
    pub fn monitored_sessions(&self) -> Vec<SessionInfo>;
}
```

### `Target` — Unified Handle for Any Tmux Entity (DC16)

**Key insight**: tmux sessions, windows, and panes are all nodes in the same
hierarchy, and at the bottom every node resolves to a PTY with input and output.
Tmux itself supports targeting at any level — `send-keys -t session` resolves to
the active pane automatically. Different hosts may have different hierarchy depths:
some have just sessions with a single window/pane, others have complex multi-window
layouts.

Rather than separate `SessionHandle`, `WindowHandle`, and `PaneHandle` types — each
with duplicated I/O methods — a single `Target` type represents any node in the
hierarchy. I/O operations work at any level (tmux resolves to the active pane).
Navigation methods (`children()`, `window()`, `pane()`) narrow the target.

```rust
/// A handle to any tmux entity: session, window, or pane.
/// Lightweight — holds Arc<HostHandleInner> + address. Cheap to clone.
pub struct Target {
    inner: Arc<HostHandleInner>,
    address: TargetAddress,
}

/// The specificity of a Target in the tmux hierarchy.
pub enum TargetAddress {
    Session(SessionInfo),
    Window(WindowInfo),
    Pane(PaneAddress),
}

/// A tmux target specifier following tmux's `session:window.pane` convention.
///
/// Tmux addresses entities at three levels of specificity:
/// - Session only: `"build"` — resolves to the active window and pane
/// - Session + window: `"build:0"` or `"build:make"` — by index or name
/// - Session + window + pane: `"build:0.1"` — fully qualified
///
/// Use `TargetSpec::session("build")` and the builder methods `.window()` / `.pane()`
/// to construct a spec, or `TargetSpec::parse("build:0.1")` from a raw string.
pub struct TargetSpec {
    session: String,
    window: Option<String>,  // index or name
    pane: Option<u32>,
}

impl TargetSpec {
    /// Start building a spec targeting a session by name.
    pub fn session(name: &str) -> Self;

    /// Narrow to a window by index.
    pub fn window(self, index: u32) -> Self;

    /// Narrow to a window by name.
    pub fn window_name(self, name: &str) -> Self;

    /// Narrow to a pane by index (requires window to be set).
    pub fn pane(self, index: u32) -> Self;

    /// Parse a raw tmux target string ("session", "session:window", "session:window.pane").
    /// Returns an error if the format is invalid.
    pub fn parse(target_str: &str) -> Result<Self>;

    /// Render as a tmux target string.
    pub fn to_target_string(&self) -> String;
}

impl fmt::Display for TargetSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_target_string())
    }
}

impl Target {
    // --- Identity ---

    /// What level of the hierarchy this target represents.
    pub fn level(&self) -> TargetLevel;

    /// The tmux target string for commands (e.g., "build", "build:0", "build:0.1").
    pub fn target_string(&self) -> String;

    /// Session info (available at any level — windows and panes know their session).
    pub fn session_info(&self) -> &SessionInfo;

    /// Window info (available at window and pane level, None at session level).
    pub fn window_info(&self) -> Option<&WindowInfo>;

    /// Pane address (available at pane level only).
    pub fn pane_address(&self) -> Option<&PaneAddress>;

    // --- Navigation (narrowing) ---

    /// List children one level down.
    /// Session → windows. Window → panes. Pane → empty.
    pub async fn children(&self) -> Result<Vec<Target>>;

    /// Navigate to a specific window by index (from session level).
    /// Queries tmux to verify the window exists.
    pub async fn window(&self, index: u32) -> Result<Option<Target>>;

    /// Navigate to a specific pane by index (from session or window level).
    /// From session: resolves to session:active_window.pane.
    /// From window: resolves to window.pane.
    pub async fn pane(&self, index: u32) -> Result<Option<Target>>;

    /// Navigate to a pane by PaneAddress (from any level).
    pub fn pane_by_address(&self, address: &PaneAddress) -> Target;

    // --- I/O (works at any level — tmux resolves to active pane) ---

    /// Send literal text. Escaped for tmux send-keys.
    /// At session/window level, sends to the active pane.
    pub async fn send_text(&self, text: &str) -> Result<()>;

    /// Send a key sequence (Enter, C-c, Tab, etc.)
    pub async fn send_keys(&self, keys: &KeySequence) -> Result<()>;

    /// Execute a shell command in the target's pane and capture structured
    /// output. Sends the command with a sentinel suffix, polls scrollback
    /// until the sentinel appears (or timeout), and extracts stdout and
    /// exit code. See DC19 for mechanism and trade-offs.
    ///
    /// At session/window level, executes in the active pane.
    pub async fn exec(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<ExecOutput>;

    /// Capture the visible content of the target pane.
    /// At session/window level, captures the active pane.
    pub async fn capture(&self) -> Result<String>;

    /// Capture with scrollback history. `start` is negative for scrollback lines
    /// (e.g., -100 = 100 lines above visible area). Captures through end of visible area.
    /// Note: tmux `-E` with negative values counts from scrollback buffer start, not
    /// visible area end, making `-S -N -E -1` semantics unreliable. Using `-S` only.
    pub async fn capture_with_history(&self, start: i32) -> Result<String>;

    /// Sample recent scrollback, returned in chronological order.
    pub async fn sample_text(&self, query: &ScrollbackQuery) -> Result<String>;

    // --- Lifecycle ---

    /// Kill this entity. Session: kills session. Window: closes window.
    /// Pane: closes pane.
    pub async fn kill(&self) -> Result<()>;

    /// Rename this entity. Session: rename session. Window: rename window.
    /// Pane: not supported (returns error).
    pub async fn rename(&self, new_name: &str) -> Result<()>;

    /// Capture all panes under this target as a map.
    /// Session: all panes in all windows. Window: all panes in window.
    /// Pane: single-entry map.
    pub async fn capture_all(&self) -> Result<HashMap<PaneAddress, String>>;

    // --- Monitoring ---

    /// Start monitoring this target (session level only — DC10).
    /// Returns an error if called on a window or pane target.
    pub async fn start_monitoring(
        &self,
        rules: &[TriggerRule],
    ) -> Result<SessionMonitorHandle>;

    /// Stop monitoring this target (session level only).
    pub async fn stop_monitoring(&self) -> Result<()>;
}

pub enum TargetLevel { Session, Window, Pane }

/// Structured output from a command executed via Target::exec().
pub struct ExecOutput {
    /// Command stdout captured from scrollback between the command echo
    /// and the sentinel line. Chronological order, trailing newline stripped.
    pub stdout: String,
    /// Exit code of the command, extracted from the sentinel.
    pub exit_code: i32,
}

impl ExecOutput {
    /// True if the command exited with code 0.
    pub fn success(&self) -> bool { self.exit_code == 0 }
}
```

**Why one type instead of three**:

1. **Tmux addressing is uniform**: `send-keys -t X` works whether X is a session,
   window, or pane. The library should mirror this.
2. **Hosts are heterogeneous**: Some hosts have sessions with a single default
   window and pane. Others have complex layouts. A single type handles both without
   forcing callers to navigate through layers that may not matter.
3. **One implementation**: `send_text()`, `send_keys()`, `capture()`, etc. are
   implemented once on `Target`. They construct `tmux ... -t <target_string>` and
   execute via the transport. No delegation chains.
4. **Navigation is optional**: `target.children()` lets you drill down when needed.
   But `host.session("build")?.send_text("ls{Enter}")` works without ever listing
   windows or panes.

**Usage examples**:

```rust
let host = fleet.host("web-1")?;

// Simple: operate at session level (tmux resolves to active pane)
let build = host.create_session("build", None, Some("cargo build")).await?;
let output = build.capture().await?;           // captures active pane
build.send_keys(&KeySequence::parse("{C-c}")?).await?;  // sends to active pane

// Drill down when you need specificity
let windows = build.children().await?;         // list windows
let panes = windows[0].children().await?;      // list panes in first window
panes[1].send_text("tail -f log.txt").await?;  // target a specific pane

// Navigate directly
let pane = build.pane(2).await?;               // session → active_window.pane_2
pane.sample_text(&ScrollbackQuery::LastLines(100)).await?;

// TargetSpec builder — type-safe targeting at any depth
let spec = TargetSpec::session("build").window(0).pane(1);
let target = host.target(&spec).await?.unwrap();
target.send_text("ls").await?;

// Or parse from a raw string (CLI args, config files)
let spec = TargetSpec::parse("build:0.1")?;
let target = host.target(&spec).await?.unwrap();

// Lifecycle
build.rename("build-v2").await?;
build.kill().await?;
```

### Monitor Handles — Granular Monitoring Control

Monitoring returns handles at two granularities: per-host (`MonitorHandle`) and
per-session (`SessionMonitorHandle`). This aligns with DC10's per-session control-mode
connections and DC13's granular lifecycle requirement.

```rust
/// Handle to all monitoring on a single host.
/// Returned by HostHandle::start_monitoring().
pub struct MonitorHandle {
    /// Per-session handles, keyed by session name (String).
    /// Lookups by &Target or &TargetSpec extract the session name internally.
    sessions: HashMap<String, SessionMonitorHandle>,
}

impl MonitorHandle {
    /// Stop monitoring on all sessions for this host.
    pub async fn shutdown(&self) -> Result<()>;

    /// Stop monitoring a specific session. Accepts a session-level Target
    /// or a TargetSpec (must resolve to session level).
    pub async fn stop_session(&self, target: &Target) -> Result<()>;

    /// Get the handle for a specific monitored session by Target.
    pub fn get(&self, target: &Target) -> Option<&SessionMonitorHandle>;

    /// Get the handle for a specific monitored session by TargetSpec.
    /// Convenience for callers with a spec but no Target in hand.
    pub fn get_by_spec(&self, spec: &TargetSpec) -> Option<&SessionMonitorHandle>;

    /// List actively monitored sessions as Targets.
    pub fn active_sessions(&self) -> Vec<&Target>;
}

/// Handle to monitoring on a single session.
/// Returned by Target::start_monitoring() (session-level targets only).
/// Each handle owns one control-mode connection (DC10).
///
/// Wraps a Target — all I/O, navigation, and lifecycle operations are
/// available via Deref<Target=Target>. This means monitoring adds
/// lifecycle methods without duplicating the unified Target API (DC16).
pub struct SessionMonitorHandle {
    target: Target,    // session-level target
    /// Signals this session's monitor task to stop.
    stop_tx: watch::Sender<bool>,
    /// The monitor task's join handle. Wrapped in Option so shutdown()
    /// can take it once. None after shutdown has been called.
    task: Mutex<Option<JoinHandle<()>>>,
}

impl Deref for SessionMonitorHandle {
    type Target = Target;
    fn deref(&self) -> &Target { &self.target }
}

impl SessionMonitorHandle {
    /// Stop monitoring this session. Tears down the control-mode connection,
    /// flushes pending output to the OutputBus, and joins the monitor task.
    /// If the pipe-pane fallback path was active, also calls
    /// PipeManager::cleanup() to detach pipes and remove FIFO files (P4).
    /// Takes the JoinHandle from the internal Mutex<Option<...>> — safe to
    /// call multiple times (subsequent calls are no-ops).
    pub async fn shutdown(&self) -> Result<()>;

    /// Check if this session's monitor is still running.
    /// Returns false after shutdown() has been called.
    pub fn is_active(&self) -> bool;
}
```

**Why `SessionMonitorHandle` keeps the `Session` name**: Monitoring is fundamentally
session-scoped — DC10's control mode (`tmux -C attach -t <session>`) operates at the
session level, and `%output` events are per-session. There is no `tmux -C attach` at
window or pane granularity. Naming the type `SessionMonitorHandle` (rather than a
generic `TargetMonitorHandle`) makes this constraint visible in the type system. The
lookup API on `MonitorHandle`, however, accepts `&Target` and `&TargetSpec` for
consistency with the rest of the library — callers use the same vocabulary everywhere,
while the `Session` prefix on the handle communicates what level of entity is actually
being monitored.

**Usage**: `SessionMonitorHandle` derefs to `Target`, so all operations work directly:

```rust
let build = host.session("build").await?.unwrap();
let monitor = build.start_monitoring(&rules).await?;

// Navigate and operate — all via Deref to Target
let panes = monitor.children().await?;         // windows in monitored session
let pane = monitor.pane(0).await?.unwrap();     // specific pane
let output = pane.sample_text(&ScrollbackQuery::LastLines(50)).await?;
pane.send_keys(&KeySequence::parse("{C-c}")?).await?;

// Monitoring lifecycle
monitor.shutdown().await?;
```

### `ScrollbackQuery` — On-Demand Text Sampling

Defines how much scrollback to capture from a pane. The result is always returned in
chronological order (oldest line first), regardless of the query direction. This is the
complement to `send_text()` — send commands in, sample output back.

```rust
pub enum ScrollbackQuery {
    /// Capture the last N lines from the pane (visible + scrollback).
    /// Equivalent to `capture-pane -p -S -N`.
    LastLines(usize),

    /// Scan backwards from the current cursor position until a line matches
    /// the regex, then return everything from that match to the present.
    /// `max_lines` caps the backward scan to prevent unbounded reads.
    Until {
        pattern: Regex,
        max_lines: usize,
    },

    /// Capture the last N lines, but stop early if a line matches the regex.
    /// Returns from the match (inclusive) to the present.
    /// Useful for "give me output since the last prompt" patterns.
    LastLinesUntil {
        lines: usize,
        stop_pattern: Regex,
    },
}
```

**Implementation**: All variants use `tmux capture-pane -p -S <start> -E <end>` under
the hood. `LastLines(n)` maps directly to `-S -n`. The `Until` and `LastLinesUntil`
variants capture `max_lines` (or `lines`) of scrollback in one call, then scan the
result in reverse for the pattern match, truncating at the match point. The final
output is returned in chronological order — no reversal needed since `capture-pane`
already outputs top-to-bottom.

**Use cases**:
- `LastLines(50)` — "show me the last 50 lines" for a quick status check
- `Until { pattern: r"^\$\s*$", max_lines: 5000 }` — "everything since the last
  shell prompt" for capturing a command's full output
- `LastLinesUntil { lines: 200, stop_pattern: r"^error:" }` — "last 200 lines, but
  stop if we hit an error marker" for focused error context

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
    /// Tmux key name not in this enum (e.g., "F12", "IC", "C-\\").
    /// Validated at parse time: rejects shell-dangerous characters
    /// (spaces, semicolons, backticks, $, etc.). Shell-escaped at
    /// command construction as defense-in-depth.
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
    fn to_tmux_args(&self, target: &str) -> Vec<Vec<String>>;
}
```

**Example usage**:

```rust
// Send "continue" followed by Enter
pane.send_keys(&KeySequence::literal("continue").then_enter()).await?;

// Send Ctrl-C to interrupt, then a new command
pane.send_keys(&KeySequence::parse("{C-c}ls -la{Enter}")?).await?;

// Send text that contains special characters (properly escaped via -l)
pane.send_text("echo 'hello > world'").await?;
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
    pub session_id: String,     // tmux session $N id — ties back to SessionInfo
    pub session_name: String,   // session name (display)
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

### `TargetOutput` — The Unit of Output

Every piece of captured output flows through the pipeline as a `TargetOutput`. It
carries enough context for any sink to identify the source at any hierarchy level
and route actions back to it.

```rust
pub struct TargetOutput {
    /// The source entity. Carries the full TargetAddress so sinks can
    /// identify the source at session, window, or pane granularity.
    /// For control-mode output (DC10), this is always pane-level since
    /// `%output` includes a pane ID. For session-only hosts (single
    /// default window/pane), callers can match at session level and
    /// ignore the window/pane detail.
    pub source: TargetAddress,
    pub host: String,          // host alias (or "localhost")
    pub content: String,       // the captured text
    pub timestamp: Instant,
}

impl TargetOutput {
    /// Session name — available at any source level.
    pub fn session_name(&self) -> &str;

    /// Pane ID — available when source is pane-level, None otherwise.
    pub fn pane_id(&self) -> Option<&str>;

    /// Tmux target string for the source (e.g., "build", "build:0", "build:0.1").
    pub fn target_string(&self) -> String;
}
```

`TargetOutput` is the bridge between the monitor/capture side and the sink side.
The `source` field (a `TargetAddress`) enables sinks to filter and match at any
level of the hierarchy — session, window, or pane — and to route actions back to the
originating entity. Hosts with a single session/window/pane can be addressed at
session level without requiring callers to know the pane ID.

### `MatcherKind` — Static-Dispatch Content Matching

Content matching uses a closed enum (`MatcherKind`) rather than trait objects. All
matching variants are known at compile time — no vtable dispatch on the hot path.
Combinators use `Vec` and `Box` (heap-backed) for tree structure.
Matchers can be stateless (regex, substring) or stateful (line
counters, vocabulary detectors that accumulate across calls).

```rust
/// A content matcher. Closed enum — all variants known at compile time.
/// No vtable dispatch; combinators use Vec<MatcherKind> (heap-backed)
/// and Not uses Box<MatcherKind> (for recursive enum sizing).
#[derive(Clone)]
pub enum MatcherKind {
    /// Matches when text contains a regex pattern. Stateless.
    Regex { pattern: Regex },

    /// Matches when text contains any of the given substrings. Stateless.
    /// Faster than regex for simple keyword lists.
    Substring { needles: Vec<String>, case_insensitive: bool },

    /// Matches after accumulating N newlines across calls. Stateful.
    /// Useful for "trigger after N lines of output" patterns.
    LineCount { threshold: usize, count: usize },

    /// Matches when text contains any word from a blocklist. Stateless.
    /// Words are matched at word boundaries (not as substrings of larger words).
    WordList { words: HashSet<String>, case_insensitive: bool },

    /// Matches when ALL inner matchers match (AND).
    AllOf(Vec<MatcherKind>),

    /// Matches when ANY inner matcher matches (OR).
    AnyOf(Vec<MatcherKind>),

    /// Inverts a matcher (NOT).
    Not(Box<MatcherKind>),   // Box only for recursive enum sizing, not for dyn dispatch
}

impl MatcherKind {
    /// Human-readable name for logging (e.g., "regex:/error/i", "all-of(3)").
    pub fn name(&self) -> String;

    /// Test whether `text` matches. For stateful variants, this may update
    /// internal state and return true when a threshold is reached.
    pub fn matches(&mut self, text: &str) -> bool;

    /// Reset internal state. Called when the matcher is reused across
    /// monitoring restarts. No-op for stateless variants.
    pub fn reset(&mut self);
}
```

**Why a closed enum**: The set of matching strategies is finite and known at design
time. A `match` arm handles each variant with zero indirection. `AllOf`/`AnyOf` store
children as `Vec<MatcherKind>` (heap-backed); `Not` uses `Box<MatcherKind>` for
recursive enum sizing only (not dynamic dispatch). No vtable allocation per node.
The entire matcher tree is `Clone` — each sink gets its own copy via `clone()`.

**Statefulness contract**: The bus calls `matches()` on the filter's matcher for each
`TargetOutput`. Stateful variants (like `LineCount`) accumulate across calls for
the same sink — each sink gets its own cloned matcher instance, so state is not shared
across sinks. `reset()` is called when monitoring restarts to clear accumulated state.

### `SinkFilter` — Composable Output Targeting

A `SinkFilter` selects which output reaches a given sink. Routing fields target the
source (host, session, window, pane). An optional `content` matcher filters on the
text itself. Multiple filters are combined with OR semantics — output matching **any**
filter in the set is delivered to the sink.

```rust
pub struct SinkFilter {
    pub host: Option<String>,      // regex against host alias
    pub session: Option<String>,   // regex against session name
    pub window: Option<String>,    // regex against "session:window_index"
    pub pane: Option<String>,      // regex against pane_id or "session:window.pane"
    pub content: Option<MatcherKind>,  // optional content matching
}

/// Compiled form — routing regexes compiled once at registration time.
/// Content matcher is moved in as-is (already runtime-ready).
pub struct CompiledSinkFilter {
    pub host: Option<Regex>,
    pub session: Option<Regex>,
    pub window: Option<Regex>,
    pub pane: Option<Regex>,
    pub content: Option<MatcherKind>,
}

impl CompiledSinkFilter {
    /// Returns true if the output matches ALL non-None fields in this filter.
    /// Fields that are None are wildcards (match everything).
    /// Routing fields (host/session/window/pane) are checked first (cheap).
    /// Content matcher is checked last (potentially stateful/expensive).
    pub fn matches(&mut self, output: &TargetOutput) -> bool;
}
```

**Note**: `matches()` takes `&mut self` because content matchers may be stateful.

**Combining filters**: A sink registers with `Vec<SinkFilter>`. Output is delivered
if it matches **any** filter in the vec (OR across filters, AND within each filter's
routing + content fields). An empty vec means "match all output" (the default).

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

// Sink receives output containing "error" or "fatal" from any source
vec![SinkFilter {
    content: Some(MatcherKind::Substring {
        needles: vec!["error".into(), "fatal".into()],
        case_insensitive: true,
    }),
    ..Default::default()
}]

// Sink triggers after 100 lines of output from "build" session
vec![SinkFilter {
    session: Some("build".into()),
    content: Some(MatcherKind::LineCount { threshold: 100, count: 0 }),
    ..Default::default()
}]

// Bad-word detector on all output
vec![SinkFilter {
    content: Some(MatcherKind::WordList {
        words: ["password", "secret", "token"].into_iter().map(Into::into).collect(),
        case_insensitive: true,
    }),
    ..Default::default()
}]
```

### `SinkKind` — Static-Dispatch Output Sinks

Every output consumer is a variant of the `SinkKind` enum. The bus dispatches via
`match` with zero vtable indirection on the hot path. Each sink runs as an independent
async task with its own batching/accumulation behavior.

**Remaining dynamic types**: `CallbackSink.state` uses `Arc<dyn Any + Send + Sync>` for
type-erased consumer state. It is passed by reference to the per-event `on_output`
callback, but incurs no per-event allocation — the `Arc` is shared and `downcast_ref`
is a type-id comparison. `on_flush` returns `Pin<Box<dyn Future>>` — unavoidable until
Rust stabilizes async fn pointers; it runs once at shutdown, off the hot path.

```rust
/// Closed enum of all sink types. Static dispatch on the hot path —
/// no vtable indirection for per-event processing.
pub enum SinkKind {
    /// Writes to stdout with configurable formatting. Reference implementation.
    Stdio(StdioSink),

    /// Forwards output to a user-provided async callback function.
    /// This is the extension point for consumers (LLM, TUI, custom logging)
    /// without requiring trait objects.
    Callback(CallbackSink),
}

/// User-provided sink via callback with explicit state. Avoids trait objects
/// while allowing consumer-defined behavior with captured state.
pub struct CallbackSink {
    pub name: String,
    pub filters: Vec<SinkFilter>,
    /// Shared state passed to callbacks. Consumers put their accumulated
    /// buffers, connections, or other mutable state here.
    pub state: Arc<dyn Any + Send + Sync>,
    /// Synchronous callback for each output event. Receives shared state
    /// and the output. For I/O-heavy sinks, queue work internally and
    /// flush asynchronously via on_flush.
    pub on_output: fn(state: &Arc<dyn Any + Send + Sync>, output: TargetOutput) -> Result<()>,
    /// Called on bus shutdown. Flush internal buffers, close resources.
    /// Returns a boxed future — the only remaining async indirection,
    /// unavoidable without async fn in fn pointers (not yet stable in Rust).
    pub on_flush: Option<fn(state: &Arc<dyn Any + Send + Sync>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>>,
}

impl SinkKind {
    /// Human-readable name for logging and diagnostics.
    pub fn name(&self) -> &str;

    /// Filters determining which output reaches this sink.
    pub fn filters(&self) -> &[SinkFilter];

    /// Process one output event. Called from the sink's own task —
    /// never from the monitor/bus hot path.
    pub async fn write(&mut self, output: TargetOutput) -> Result<()>;

    /// Called on bus shutdown. Flush internal buffers, close resources.
    pub async fn flush(&mut self) -> Result<()>;
}
```

**Key design principle**: Each sink owns its batching/accumulation strategy internally.
The bus delivers individual `TargetOutput` events; the sink decides whether to process
them immediately (stdio), buffer and render at frame rate (TUI), or accumulate and
flush on a timer/threshold (LLM). This keeps the bus simple and the sink in full
control of its own latency/throughput tradeoffs.

**Extension via `CallbackSink`**: Consumers that need custom sink behavior (LLM
inference, webhook delivery, custom TUI) provide function pointers plus an explicit
`Arc<dyn Any + Send + Sync>` state field. The `on_output` callback receives `&Arc`
to access state — no closure captures needed. This keeps the bus monomorphic.
`on_flush` returns `Pin<Box<dyn Future>>` — the only remaining async indirection,
unavoidable until Rust stabilizes async fn pointers. The framework's `on_output` dispatch path is
fully synchronous with no framework-side allocation (user callbacks may allocate
internally as needed).

### `ActionHandle` — Sink-Initiated Actions

Sinks that analyze output may need to act on the source entity — send keys to the
pane, kill a session, etc. Rather than giving sinks direct access to `HostHandle`
(which would create circular dependencies), sinks receive an `ActionHandle` that
provides a scoped, async API for actions against the tmux entities they observe.

```rust
pub struct ActionHandle { /* mpsc::Sender<ActionRequest> */ }

pub struct ActionRequest {
    pub host: String,
    pub target: TargetAddress,
    pub action: SinkAction,
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

    /// Convenience: send keys to a target (any level — session, window, or pane).
    pub async fn send_keys(
        &self,
        host: &str,
        target: &TargetAddress,
        keys: KeySequence,
    ) -> Result<()>;

    /// Convenience: send text to a target (any level).
    pub async fn send_text(
        &self,
        host: &str,
        target: &TargetAddress,
        text: &str,
    ) -> Result<()>;

    /// Convenience: kill a session by name.
    pub async fn kill_session(&self, host: &str, session: &str) -> Result<()>;

    /// Convenience: rename a session.
    pub async fn rename_session(
        &self,
        host: &str,
        session: &str,
        new_name: &str,
    ) -> Result<()>;

    /// Respond to a specific output event. Extracts host and target from the
    /// TargetOutput's metadata — sinks don't need to decompose fields manually.
    /// This is the primary ergonomic entry point for reactive sinks.
    pub async fn respond(
        &self,
        output: &TargetOutput,
        action: SinkAction,
    ) -> Result<()>;
}
```

Sinks that need to initiate actions capture an `ActionHandle` at their own construction
time (not injected by `OutputBus`). Action requests are routed through the existing
per-host bounded dispatch queue (DC4) — the same path used by the monitor's trigger
rules. This ensures consistent ordering, backpressure, and concurrency limits regardless
of whether an action originates from a rule or a sink.

**LLM feedback loop**: An LLM sink can call `action_handle.respond(&output, action)` after
analyzing output — the host and target are extracted from the `TargetOutput` automatically. The design of the LLM sink itself (prompt engineering, approval gates,
autonomous vs supervised mode) is **out of scope** for this library. The library provides
the `SinkKind` enum and `ActionHandle` API; LLM integration is a consumer concern.

### `OutputBus` — Fan-Out Dispatcher

The `OutputBus` is the central distributor. It receives `TargetOutput` from the monitor
(or from `capture_pane()` callers) and fans out to all registered sinks.

```rust
pub struct OutputBus { /* subscribers: Vec<SinkEntry> */ }

struct SinkEntry {
    id: SinkId,
    name: String,
    tx: mpsc::Sender<TargetOutput>,
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
        sink: SinkKind,
        channel_capacity: usize,
    ) -> SinkId;

    /// Remove a sink. Signals stop, awaits flush(), joins the task.
    pub async fn unsubscribe(&mut self, id: SinkId) -> Result<()>;

    /// Fan out an event to all matching sinks. Non-blocking.
    /// Uses try_send — if a sink's channel is full, the event is dropped
    /// for that sink only (logged at debug level). Sinks that cannot
    /// tolerate drops should use larger channel capacities.
    pub fn publish(&self, output: TargetOutput);

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

The library ships with `StdioSink` as the default, always-available sink. It is a
variant of `SinkKind` and serves as the reference implementation.

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

// StdioSink is handled by SinkKind::Stdio variant.
// write(): format and write to stdout immediately (no batching)
// flush(): flush stdout
```

### `JoinedStream` — Multi-Source Consolidated View

Individual sinks see output from their filtered sources, but each `TargetOutput` arrives
independently. A `JoinedStream` merges multiple source streams into a single
time-ordered sequence where each chunk is attributed to its source — like a multi-party
conversation transcript. This is the natural representation for an LLM analyzing
cross-pane interactions, a log aggregator, or a TUI showing interleaved output.

```rust
/// A chunk in the joined stream. Wraps TargetOutput with a source label
/// for human-readable attribution in the consolidated view.
pub struct StreamChunk {
    pub source: SourceLabel,
    pub output: TargetOutput,
}

/// Identifies the source of a chunk in a joined stream.
/// Constructed automatically from TargetOutput metadata.
/// Works at any hierarchy level — session-only hosts produce
/// session-level labels; pane-level output gets full specificity.
pub struct SourceLabel {
    pub host: String,           // "localhost", "web-1"
    pub target: TargetAddress,  // source at whatever level is available
}

impl SourceLabel {
    /// Short form for display: "web-1:build:0.1" (pane) or "web-1:build" (session)
    pub fn short(&self) -> String;

    /// Minimal form when host is unambiguous: "build:0.1" or "build"
    pub fn minimal(&self) -> String;
}
```

**`JoinedSink`** — a sink combinator that wraps an inner `SinkKind` and presents
incoming output as a consolidated multi-source stream:

```rust
/// Wraps an inner sink, transforming the output stream into a
/// source-attributed conversation-style view.
pub struct JoinedSink {
    inner: SinkKind,
    /// Controls how source labels are formatted in the stream.
    label_format: LabelFormat,
    /// Tracks the last source to emit, so consecutive chunks from the
    /// same source can be coalesced without repeating the label.
    last_source: Option<SourceLabel>,
}

pub enum LabelFormat {
    /// "[web-1:build:0.1] output text here"
    Bracketed,
    /// "web-1:build:0.1> output text here"
    Prompt,
    /// Caller provides a format function (plain fn pointer — no heap allocation).
    Custom(fn(&SourceLabel, &str) -> String),
}

impl JoinedSink {
    pub fn new(inner: SinkKind, label_format: LabelFormat) -> Self;
}
```

When `JoinedSink::write()` receives a `TargetOutput`, it:
1. Extracts the `SourceLabel` from the output's metadata
2. If the source differs from `last_source`, emits a source header/prefix
3. Delegates to the inner sink's `write()` with the attributed content
4. Updates `last_source` for coalescing

**Coalescing**: Consecutive chunks from the same source are grouped without repeating
the source label — like a chat UI where the sender name only appears on the first
message in a burst. When a different source emits, the label appears again.

**Use cases**:

```rust
// LLM sees a consolidated conversation across 3 build panes
let llm_sink = JoinedSink::new(
    SinkKind::Callback(CallbackSink { /* LLM sink config */ }),
    LabelFormat::Bracketed,
);
// Output looks like:
//   [web-1:build:0.0] compiling crate foo...
//   [web-1:build:0.0] warning: unused variable
//   [db-1:migrate:0.0] Running migration 042...
//   [db-1:migrate:0.0] OK
//   [web-1:build:0.0] Finished dev target

// Stdio log with prompt-style labels
let log_sink = JoinedSink::new(
    SinkKind::Stdio(StdioSink::new(StdioFormat::Raw)),
    LabelFormat::Prompt,
);
// Output looks like:
//   web-1:build:0.0> compiling crate foo...
//   db-1:migrate:0.0> Running migration 042...
```

**Extracting a joined view programmatically**: For callers that want to consume
the stream as structured data (not formatted text), the `OutputBus` supports
registering a channel-based subscriber that receives `StreamChunk` directly:

```rust
impl OutputBus {
    /// Subscribe a raw channel that receives StreamChunks (TargetOutput + SourceLabel).
    /// The caller reads from the receiver directly. No SinkKind wrapper needed.
    pub fn subscribe_joined(
        &mut self,
        filters: Vec<SinkFilter>,
        channel_capacity: usize,
    ) -> (SinkId, mpsc::Receiver<StreamChunk>);
}
```

This is the lowest-level API for consumers that want to build their own rendering
on top of the joined stream (e.g., a TUI panel, a web socket feed, or test harness).

### Integration with Fleet and Monitor

```
  Fleet
    │
    ├── OutputBus (owned by Fleet)
    │     │
    │     ├── subscribe(SinkKind::Stdio(StdioSink::new(StdioFormat::Raw)), 1024)
    │     ├── subscribe(SinkKind::Callback(tui_sink), 16)        // binary-provided
    │     └── subscribe(SinkKind::Callback(llm_sink), 256)       // LLM callback sink
    │
    ├── HostHandle (localhost)
    │     └── Monitor ──publish()──► OutputBus
    │
    └── HostHandle (remote)
          └── Monitor ──publish()──► OutputBus
```

**Monitor → Bus**: The monitor publishes `TargetOutput` to the bus after each output
event. This happens *in addition to* rule evaluation — rules and sinks operate
independently on the same stream.

**capture_pane() → Bus**: On-demand captures can optionally be published to the bus
via a `bus.publish()` call. This is opt-in at the call site, not automatic.

**Sink → ActionHandle → HostHandle**: A sink that decides to act (e.g., an LLM sink
that detects an error and wants to send a recovery command) submits an `ActionRequest`
through its `ActionHandle`. The request is routed to the correct `HostHandle` via the
existing per-host dispatch queue (DC4). The sink does not need to know which
`HostHandle` to use — routing is by the `host` field and `TargetAddress` in `ActionRequest`.

---

## Module Specifications

### `matcher.rs`

The `MatcherKind` enum and its variants. See
[Core Abstractions — MatcherKind](#matcherkind--static-dispatch-content-matching) for
the full enum definition.

All variants are defined in the `MatcherKind` enum. `Regex` and `Substring` are
stateless (`reset()` is a no-op). `LineCount` is stateful and resets its counter
on `reset()`. `WordList` uses `regex::Regex` internally with `\b` word boundaries
for accurate matching. The combinators (`AllOf`, `AnyOf`, `Not`) delegate to their
children and propagate `reset()`.

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
/// This is the serde-clean layer; `pattern` is a regex string by default.
/// For non-regex matchers, use `CompiledRule::with_matcher()` directly.
pub struct TriggerRule {
    pub name: String,                  // human-readable rule name for logging
    pub pane_filter: Option<String>,   // regex string (None = all panes)
    pub pattern: String,               // regex string to match against pane output
    pub action: Action,
    pub cooldown: Option<Duration>,    // debounce repeated triggers per pane
}

/// Runtime form — compiled from TriggerRule during startup validation.
/// The `matcher` field is a `MatcherKind` enum — config-driven rules compile
/// to `MatcherKind::Regex`, but programmatic callers can use any variant
/// (WordList, LineCount, AllOf, etc.).
pub struct CompiledRule {
    pub name: String,
    pub pane_filter: Option<Regex>,
    pub matcher: MatcherKind,
    pub action: Action,
    pub cooldown: Option<Duration>,
}

impl TriggerRule {
    /// Compile string patterns into MatcherKind::Regex. Returns error with rule name context.
    pub fn compile(&self) -> Result<CompiledRule>;
}

impl CompiledRule {
    /// Construct a rule with a custom MatcherKind (bypasses config deserialization).
    /// Used by programmatic callers who want non-regex matching.
    pub fn with_matcher(
        name: impl Into<String>,
        matcher: MatcherKind,
        action: Action,
    ) -> Self;
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
/// Closed enum of transport implementations. Static dispatch — the same
/// tmux operations work on localhost or SSH with zero vtable overhead.
pub enum TransportKind {
    Local(LocalTransport),
    Ssh(SshTransport),
    Mock(MockTransport),
}

impl TransportKind {
    /// Execute a command and return its stdout as a String.
    /// Must respect the configured timeout.
    pub async fn exec(&self, command: &str) -> Result<String>;

    /// Open a persistent shell for streaming output.
    /// Used by the monitor for long-running processes (control mode session
    /// or `tail -qf` in fallback pipe mode).
    pub async fn open_shell(&self) -> Result<ShellChannelKind>;
}

/// Closed enum of shell channel implementations.
pub enum ShellChannelKind {
    Local(LocalShellChannel),
    Ssh(SshShellChannel),
    Mock(MockShellChannel),
}

impl ShellChannelKind {
    /// Write data to the shell's stdin.
    pub async fn write(&mut self, data: &[u8]) -> Result<()>;

    /// Wait for the next message from the shell.
    pub async fn read(&mut self) -> Option<ShellEvent>;
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
streaming data for `open_shell()`. No separate test infrastructure needed.

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

    /// Parse from tmux list-panes output fields.
    /// `pane_id` is the `#{pane_id}` field (e.g., "%12").
    /// `address_str` is the composite "session:window.pane" field.
    pub fn parse(pane_id: &str, address_str: &str) -> Result<Self>;
}
```

**Addresses P2**: Using `#{pane_id}` as the authoritative key (see DC1) eliminates the
need for filename encoding of session names entirely. FIFO paths (if used) are simply
`/tmp/motlie_pipe_%<id>`.

```rust
/// List all sessions on the host.
pub async fn list_sessions(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
) -> Result<Vec<SessionInfo>>;

/// List all windows in a session.
pub async fn list_windows(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<Vec<WindowInfo>>;

/// List all panes, optionally filtered by regex against "session:window.pane".
pub async fn list_panes(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
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
pub async fn capture_pane(transport: &TransportKind, target: &PaneAddress) -> Result<String>;

/// Capture with scrollback history.
/// Runs: tmux capture-pane -p -t <target> -S <start> -E <end>
/// start/end are line numbers; negative values reach into scrollback buffer.
/// Example: start=-1000, end=-1 captures last 1000 lines of scrollback.
pub async fn capture_pane_history(
    transport: &TransportKind,
    target: &PaneAddress,
    start: i32,
    end: i32,
) -> Result<String>;

/// Capture all panes in a session. Calls capture_pane for each pane found via list_panes.
/// Returns a map of pane address → visible content.
pub async fn capture_session(
    transport: &TransportKind,
    session: &str,
) -> Result<HashMap<PaneAddress, String>>;

/// Sample recent text from a pane's scrollback, returned in chronological order.
/// Delegates to capture_pane_history() internally, then applies the query's
/// pattern matching and truncation logic.
pub async fn sample_text(
    transport: &TransportKind,
    target: &PaneAddress,
    query: &ScrollbackQuery,
) -> Result<String>;
```

**`sample_text` implementation**:
1. `LastLines(n)` → calls `capture_pane_history(transport, target, -(n as i32), -1)`.
   Result is already chronological.
2. `Until { pattern, max_lines }` → calls `capture_pane_history` with
   `start = -(max_lines as i32)`, `end = -1`. Scans the result from the bottom up for
   the first line matching `pattern`. Returns from that line (inclusive) to the end.
   If no match, returns the full captured range.
3. `LastLinesUntil { lines, stop_pattern }` → same as `LastLines(lines)`, then scans
   bottom-up for `stop_pattern`. Truncates at the match point (inclusive).

In all cases the output preserves `capture-pane`'s top-to-bottom ordering — no reversal
step is needed. The pattern scan is the only post-processing.

**Escaping**: The `-p` flag outputs to stdout (not to a buffer), which is what we need
over SSH exec channels. The `-e` flag (escape sequences) is intentionally NOT used — we
want plain text, not ANSI-encoded output.

**Use cases**:
- Dump a pane to see what state it's in before deciding to send input
- Capture all panes in a session for a holistic snapshot (e.g., debugging a multi-pane layout)
- Feed pane content to an LLM for analysis/decision-making

### `control.rs`

Tmux control: session lifecycle, sending input, and managing session/window metadata.

**Note**: These are internal functions that accept raw strings extracted from typed
`TargetAddress` variants by `Target`. The public API uses `Target` methods (DC16, DC17).
Callers should use `HostHandle` and `Target` methods, not these functions directly.

```rust
// --- Session Lifecycle ---

/// Create a new detached tmux session.
/// Runs: tmux new-session -d -s <name> [-n <window_name>] [<command>]
pub async fn create_session(
    transport: &TransportKind,
    name: &str,
    window_name: Option<&str>,
    command: Option<&str>,
) -> Result<()>;

/// Kill a tmux session and all its windows/panes.
/// Runs: tmux kill-session -t <name>
pub async fn kill_session(transport: &TransportKind, name: &str) -> Result<()>;

// --- Input ---

/// Send a KeySequence to a pane. Handles the split between literal text (-l)
/// and special keys (no -l) automatically.
pub async fn send_keys(
    transport: &TransportKind,
    target: &PaneAddress,
    keys: &KeySequence,
) -> Result<()>;

/// Convenience: send literal text (no special keys, no Enter appended).
/// Equivalent to: tmux send-keys -l -t <target> '<escaped_text>'
pub async fn send_text(
    transport: &TransportKind,
    target: &PaneAddress,
    text: &str,
) -> Result<()>;

/// Rename a tmux session.
/// Runs: tmux rename-session -t <current> <new>
pub async fn rename_session(
    transport: &TransportKind,
    current_name: &str,
    new_name: &str,
) -> Result<()>;

/// Rename a window.
/// Runs: tmux rename-window -t <session>:<index> <new_name>
pub async fn rename_window(
    transport: &TransportKind,
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
- Supports per-host monitoring start/stop (`start_monitoring_host()`, `stop_monitoring_host()`)
- Per-host stop delegates to `HostHandle::stop_monitoring()`, which tears down all session monitors for that host
- Owns the `shutdown` watch channel; `shutdown()` signals all targets
- Owns the `OutputBus` (shared with HostHandles via `Arc`); exposes `output_bus()` accessor
- Maintains workstream registry: `HashMap<String, WorkstreamEntry>` for named (host, target) bindings

```rust
pub enum HostStatus {
    Disconnected,
    Connecting,
    Connected,
    /// Monitoring N sessions. Includes count for observability.
    Monitoring { sessions: usize },
    Error(String),
}
```

### `host.rs`

The `HostHandle` and `Target` implementations. See Core Abstractions for the full APIs.

`HostHandle` wraps `Arc<HostHandleInner>` which holds the transport and config.
`Target` is lightweight — holds an `Arc<HostHandleInner>` plus a `TargetAddress` enum.
Both delegate to the function-level APIs in `discovery`, `capture`, `control`, `pipe`,
and `monitor`.

```rust
struct HostHandleInner {
    transport: TransportKind,
    config: HostTarget,
    pipe_state: Option<PipeManager>,   // None if monitoring not started (fallback path)
    /// Per-session monitor handles, keyed by session name.
    /// Each entry represents one control-mode connection (DC10).
    /// RwLock allows &self methods to mutate monitoring state safely.
    session_monitors: RwLock<HashMap<String, SessionMonitorHandle>>,
}
```


### `pipe.rs`

FIFO lifecycle and `tmux pipe-pane` management.

```rust
pub struct PipeManager { /* tracks active pipes for cleanup */ }

impl PipeManager {
    /// Create FIFOs and attach pipe-pane for each target pane.
    /// Uses a dedicated transport exec call (not the monitor channel).
    pub async fn setup(transport: &TransportKind, panes: &[PaneAddress]) -> Result<Self>;

    /// Detach all pipe-panes and remove FIFO files.
    pub async fn cleanup(&self, transport: &TransportKind) -> Result<()>;
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

The core event loop. Each `SessionMonitor` owns one control-mode connection to a single
tmux session and evaluates rules against its output. `HostHandle` creates one
`SessionMonitor` per monitored session (DC10, DC13).

```rust
/// Monitors a single tmux session via control mode.
pub struct SessionMonitor { /* session name, rules, cooldown state */ }

impl SessionMonitor {
    /// Run the monitor loop for one session.
    /// Opens `tmux -C attach -t <session>` via the transport, parses
    /// `%output %<pane_id> <data>` frames, evaluates rules, and publishes
    /// TargetOutput to the OutputBus.
    /// Returns when `stop` signal is received or the connection drops.
    pub async fn run(
        &mut self,
        transport: &TransportKind,
        session: &str,
        rules: &[TriggerRule],
        bus: &OutputBus,
        stop: watch::Receiver<bool>,
    ) -> Result<()>;
}
```

**Must address P3**: Rule evaluation replaces the hardcoded `contains('>')` check.

**Must address P6**: Each `SessionMonitor` uses a dedicated control-mode connection
(`tmux -C attach -t <session>`, see DC10) as the primary strategy, or per-pane pipe
files as fallback. Action dispatch (send-keys) uses separate exec channels routed
through a bounded queue (see [DC4](#dc4-action-dispatch-channel-strategy)).

**Must address P9**: Failed send-keys or malformed lines must be logged at `warn` level,
not silently dropped.

**Stream parsing**: With control mode (DC10), the monitor parses `%output %<pane_id> <data>`
frames — structured, unambiguous, and keyed on `#{pane_id}` per DC1. For the pipe-pane
fallback, `pipe-pane` output is prefixed with `%<pane_id>` (set at attach time), and the
monitor parses on that prefix directly — no filename decoding required.

**Lifecycle**: Each `SessionMonitor::run()` is spawned as a tokio task by `HostHandle`.
The `SessionMonitorHandle` returned to the caller holds the task's `JoinHandle` and stop
channel. Stopping a session monitor is non-disruptive to other sessions (DC13).

### `sink.rs`

The output sink pipeline types. See [Output Sink Pipeline](#output-sink-pipeline) for
the full API and design rationale.

This module contains:
- `TargetOutput`: the unit of output flowing through the pipeline
- `SinkKind` enum: closed set of sink types (static dispatch, no trait objects)
- `SinkFilter` / `CompiledSinkFilter`: composable output targeting
- `MatcherKind` (re-exported from `matcher.rs`): content matching variants
- `ActionHandle` / `ActionRequest` / `SinkAction`: sink-initiated actions
- `OutputBus`: central fan-out dispatcher with per-sink bounded channels
- `SinkId`: opaque handle for unsubscribe

`OutputBus` is owned by `Fleet` and shared with all `HostHandle` instances via `Arc`.
Monitors publish `TargetOutput` to the bus; the bus fans out to per-sink tokio tasks.
Each sink task drives its own `SinkKind::write()` loop independently.

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

**Decision**: A `TransportKind` enum abstracts command execution via static dispatch.
Three variants: `Local(LocalTransport)` (localhost, subprocess-based),
`Ssh(SshTransport)` (remote, russh-based), and `Mock(MockTransport)` (testing).

**Rationale**: The prototype assumes SSH for everything, but localhost tmux is a primary
use case (local development, CI, single-machine automation). Forcing SSH to localhost
adds unnecessary complexity (SSH server requirement, key management, latency). The enum
abstraction lets all downstream modules (`discovery`, `capture`, `control`, `pipe`,
`monitor`) be transport-agnostic with zero vtable overhead.

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
| Tmux version req | Requires control mode (validated at runtime; minimum determined by CI matrix, see OC4) | pipe-pane `-o`: needs version testing (OC4) |
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
(`OutputBus`) that delivers `TargetOutput` to independently-running sink tasks. Each sink
receives its own bounded channel and manages its own batching, buffering, and timing
internally. Sinks are filtered via composable `SinkFilter` structs (OR across filters,
AND within fields). Sinks may initiate actions on tmux entities via an `ActionHandle`
that routes requests through the existing per-host action dispatch queue (DC4).

**Rationale**:
- **Decoupled latency**: A slow LLM sink must never block a fast stdio sink. Per-sink
  channels with independent tasks ensure this.
- **Sink-owned batching**: The bus delivers individual `TargetOutput` events; sinks decide
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
- Shared `Arc<Mutex<Vec<TargetOutput>>>` polling: Rejected — wastes CPU, no backpressure.

### DC13: Granular Monitoring Lifecycle

**Decision**: Monitoring is controllable at three levels: fleet-wide, per-host, and
per-session. Each level can be started and stopped independently without affecting
other active monitors at the same or higher level.

**API surface**:

| Level | Start | Stop |
|-------|-------|------|
| Fleet | `fleet.start_monitoring(rules)` | `fleet.shutdown()` |
| Host | `fleet.start_monitoring_host(host, rules)` | `fleet.stop_monitoring_host(host)` |
| Session | `host.start_monitoring_session(&target, rules)` | `host.stop_monitoring_session(&target)` |

**Rationale**: DC10 establishes that each monitored session has its own control-mode
connection (`tmux -C attach -t <session>`). These connections are independent — tearing
one down has no effect on others. The API should expose this natural granularity rather
than forcing all-or-nothing lifecycle. Use cases:

- **Dynamic session management**: A consumer creates a tmux session, monitors it for a
  task, then stops monitoring when the task completes — without disrupting monitoring
  on other sessions.
- **Selective host disconnect**: An SSH target becomes unreachable. The caller stops
  monitoring on that host while others continue.
- **Incremental rollout**: Start monitoring one session at a time for debugging before
  enabling fleet-wide monitoring.

**Implementation**: `HostHandle` tracks per-session `SessionMonitorHandle` in a
`HashMap<String, SessionMonitorHandle>`. Each handle owns its control-mode connection's
stop channel and task join handle. `MonitorHandle` (returned by `start_monitoring()`) is
a view over the session handles, not a separate entity. `stop_monitoring_session()` drops
the session's handle, which signals the stop channel, flushes pending output to the
`OutputBus`, joins the monitor task, and — if the pipe-pane fallback path was active —
calls `PipeManager::cleanup()` to detach pipes and remove FIFO files from `/tmp` (P4).
`stop_monitoring()` iterates all sessions.

**Invariant**: On-demand operations (`target.capture()`, `target.send_keys()`,
`host.list_sessions()`, etc.) are unaffected by monitoring state. A host with stopped
monitoring is still fully operational for on-demand use via `HostHandle` and `Target`.

### DC14: Trait-Based Content Matching

**Decision**: Text matching is abstracted behind the `MatcherKind` enum. Regex is
one implementation, not a privileged special case. Both `SinkFilter` (content field) and
`CompiledRule` (matcher field) use `MatcherKind`.

**Rationale**: Different use cases need different matching strategies:
- **Regex**: General-purpose pattern matching (the default for config-driven rules)
- **Substring/keyword lists**: Faster than regex for simple "does output contain X?"
- **Word-boundary blocklists**: Bad-word/secret detectors that don't false-positive on
  substrings (e.g., "token" matches "token" but not "tokenize")
- **Stateful stream matchers**: Line counters, byte accumulators, rate detectors —
  these need to track state across multiple `TargetOutput` events

Making regex the only matching mechanism would force all of these into regex patterns,
which is unnatural for stateful matchers and inefficient for simple keyword checks.

**Statefulness**: Matchers may be stateful (`matches(&mut self, ...)`). Each sink and
each compiled rule gets its own cloned `MatcherKind` instance — state is never shared.
`reset()` is called on monitoring restart. The bus evaluates routing fields
(host/session/window/pane) before calling the content matcher, so expensive matchers
are only invoked on pre-filtered output.

**Static dispatch**: All matcher variants are defined in the `MatcherKind` enum (DC14).
No vtable dispatch on the hot path. `AllOf`/`AnyOf` store children as `Vec<MatcherKind>`
(heap-backed); `Not` uses `Box<MatcherKind>` for recursive enum sizing. The entire
matcher tree is `Clone`.

**Config boundary**: `TriggerRule` (serde DTO) stores `pattern: String` which compiles
to `MatcherKind::Regex`. Programmatic callers bypass the config layer and construct
`CompiledRule::with_matcher()` or `SinkFilter { content: Some(...) }` directly with
any `MatcherKind` variant.

**Built-in matchers** (all `MatcherKind` enum variants): `Regex`, `Substring`,
`LineCount`, `WordList`, plus `AllOf`/`AnyOf`/`Not` combinators. The enum is closed —
new matcher types require adding a variant (keeping the hot path static-dispatch).

### DC15: Joined Stream — Multi-Source Consolidated View

**Decision**: Multiple filtered output streams can be joined into a single time-ordered
sequence where each chunk is attributed to its source. This is implemented as a sink
combinator (`JoinedSink`) and a raw channel API (`subscribe_joined()`), not as a
separate pipeline stage.

**Rationale**: When monitoring multiple panes across hosts, consumers often need a
unified view — an LLM analyzing cross-pane interactions, a log aggregator correlating
events, or a TUI showing interleaved output. Without joining, each sink sees isolated
streams and must implement its own source-tracking and interleaving logic.

**Design choices**:
- **Combinator, not infrastructure**: `JoinedSink<S>` wraps any `SinkKind` and adds
  source attribution. The bus itself remains simple (fan-out only). This avoids adding
  joining logic to the bus hot path.
- **Source coalescing**: Consecutive chunks from the same source are grouped without
  repeating the label. This mirrors chat UIs where the sender appears once per burst,
  reducing visual noise in high-throughput streams.
- **Two levels of API**: `JoinedSink` for formatted text output (wraps any sink),
  `subscribe_joined()` for structured `StreamChunk` data (channel-based, no sink trait).
  The structured API enables custom rendering without forcing text formatting.
- **Label customization**: `LabelFormat` supports bracketed, prompt-style, and custom
  formatters. The `SourceLabel` type provides `short()` and `minimal()` for common cases.

**Alternatives considered**:
- Bus-level stream merging: Rejected — adds complexity to the bus and forces all sinks
  into a joined view. Not all sinks want interleaved output.
- Post-hoc log parsing: Rejected — requires sinks to independently reconstruct
  source attribution from `TargetOutput` fields, duplicating logic across sinks.

### DC16: Unified Target Type

**Decision**: A single `Target` type represents any tmux entity (session, window, or
pane). `HostHandle` is slimmed to host-level concerns only — session discovery,
creation, and host-wide monitoring. All entity-level operations live on `Target`.
`SessionMonitorHandle` wraps `Target` via `Deref`.

**Rationale**: Tmux itself uses uniform addressing — `send-keys -t X` works whether
X is a session, window, or pane (tmux resolves to the active pane). A library that
mirrors this uniformity avoids three problems:

1. **God object**: A flat `HostHandle` with 20+ methods mixing host, session, and
   pane concerns is unwieldy. But decomposing into separate `SessionHandle`,
   `WindowHandle`, and `PaneHandle` types leads to duplicated I/O methods — every
   level needs `send_text()`, `capture()`, etc. because tmux supports them at every level.
2. **Heterogeneous hosts**: Not all hosts have the same hierarchy depth. Some have
   sessions with a single default window/pane; others have complex multi-window
   layouts. A single `Target` type handles both without forcing callers through
   intermediate layers that may be irrelevant.
3. **One implementation**: `send_text()`, `send_keys()`, `capture()` etc. are
   implemented once on `Target`. They construct `tmux ... -t <target_string>` and
   execute via the transport. No delegation chains, no method duplication.

**Key properties**:

| Component | Responsibility |
|-----------|---------------|
| `HostHandle` | Host-level: `list_sessions`, `create_session`, `session`, `target`, `start_monitoring`, `stop_monitoring`, `start_monitoring_session`, `stop_monitoring_session` |
| `Target` | Entity-level: I/O (`send_text`, `send_keys`, `capture`, `sample_text`), navigation (`children`, `window`, `pane`), lifecycle (`kill`, `rename`), monitoring (`start_monitoring`) |
| `SessionMonitorHandle` | Monitoring lifecycle + `Deref<Target=Target>` — adds `shutdown`, `is_active` |

- **Cheap handles**: `Target` holds `Arc<HostHandleInner>` + `TargetAddress` enum.
  Clone is cheap and safe to pass across tasks.
- **Navigation is optional**: `host.session("build")?.send_text("ls")` works without
  ever listing windows or panes. `target.children()` drills down when needed.
- **DC13 preserved**: On-demand operations work regardless of monitoring state.
  A `Target` works the same whether monitoring is active or not.

**`Deref` for monitoring**: `SessionMonitorHandle` only adds `shutdown()` and
`is_active()`. Everything else comes from `Target` via `Deref`. This means
`monitor.capture()`, `monitor.children()`, `monitor.send_text("x")` etc. all work
without any explicit delegation code.

### DC17: Type Safety via Target + TargetAddress

**Decision**: Type safety comes from the `TargetAddress` enum (`Session(SessionInfo)`,
`Window(WindowInfo)`, `Pane(PaneAddress)`) embedded in each `Target`. Discovery
methods produce `Target` values with appropriate addresses; subsequent operations
carry that typed context. Raw strings enter via `TargetSpec::parse()` +
`host.target(&spec)` as the escape hatch.

**Rationale**: Raw string parameters for session names and window indices are
error-prone — a typo silently targets the wrong entity or returns empty results.
The `Target` approach provides:
- **Provenance**: A `Target` was produced by the system (via `list_sessions()`,
  `create_session()`, `children()`, etc.), so it refers to a real entity
- **Structural scoping**: `target.children()` returns child-level `Target` values —
  no `PaneScope` enum or string-based filtering needed
- **Self-contained context**: Each `Target` carries enough info to construct tmux
  commands without external lookups. `WindowInfo` includes `session_name`; `PaneAddress`
  includes session and window indices.

**API changes from flat HostHandle baseline**:

| Before (flat HostHandle) | After (unified Target) |
|---------------------------|------------------------|
| `host.kill_session(name: &str)` | `target.kill()` |
| `host.list_windows(session: &str)` | `target.children()` (at session level) |
| `host.list_panes(filter: Option<&Regex>)` | `target.children()` (at window level) |
| `host.capture_session(session: &str)` | `target.capture_all()` |
| `host.rename_session(name, new_name)` | `target.rename(new_name)` |
| `host.send_text(&pane_addr, text)` | `target.send_text(text)` |
| `host.send_keys(&pane_addr, keys)` | `target.send_keys(keys)` |
| `host.capture_pane(&pane_addr)` | `target.capture()` |
| `host.sample_text(&pane_addr, query)` | `target.sample_text(query)` |
| `host.create_session(...) -> Result<()>` | `host.create_session(...) -> Result<Target>` |

**Escape hatch**: `TargetSpec::parse("build:0.1")` constructs a spec from a raw
string, and `host.target(&spec)` queries tmux to verify the entity exists and returns
`Option<Target>`. The builder alternative — `TargetSpec::session("build").window(0).pane(1)` —
is preferred when components are known at compile time. This is the bridge from raw
strings (CLI args, config files) to typed handles.

### DC18: Composite Workstreams (Future)

**Status**: Deferred — documented for future reference. Single-target workstreams
(`Fleet::bind()`) are in scope; composite workstreams require real usage patterns to
inform membership lifecycle decisions.

**Concept**: Extend workstreams to group multiple (host, target) pairs under one name —
e.g., "deploy" spans `web-1:build`, `db-1:migrate`, and `web-1:test`. This would
compose with existing abstractions:

- **`SinkFilter` by workstream**: Filter output by workstream name. The bus resolves
  the workstream to its member targets at filter-match time.
- **`JoinedStream` over a workstream**: `subscribe_joined()` accepts a workstream name,
  automatically creating filters for all member targets.
- **Workstream-scoped monitoring**: Start/stop monitoring for all targets in a
  workstream with a single call.

**Open questions** (to be resolved by usage):
- What happens when a member target is killed or disconnected?
- Can targets belong to multiple workstreams?
- Should workstreams be defined in config or only programmatically?

### DC19: Structured Command Execution via Target::exec()

**Decision**: `Target::exec(command, timeout)` provides structured command execution
within a tmux pane, complementing the existing fire-and-forget `send_text()`/`send_keys()`.
It returns `ExecOutput { stdout, exit_code }` by using a sentinel-based capture mechanism.
No host-level bypass (`HostHandle::exec()`) is added — the library's abstraction boundary
is tmux, and all command execution stays within the tmux framework.

**Three modes of pane interaction**:

| Method | Semantics | Output | Use case |
|--------|-----------|--------|----------|
| `send_text(text)` | Fire-and-forget PTY input | None | Interactive use: typing into vim, top, a REPL |
| `send_keys(keys)` | Fire-and-forget special keys | None | Control sequences: Enter, C-c, Tab |
| `exec(cmd, timeout)` | Command-and-capture | `ExecOutput` | Automation: build, test, deploy, diagnostics |

**Rationale**: Automation workflows need structured results — "run `make test`, did it
pass?" Today this requires `send_text("make test\n")` + manual `capture()` + hoping
the command finished + parsing output for success/failure. `exec()` encapsulates this
into a single call with clear completion semantics and an exit code.

**Sentinel mechanism**:

1. Generate a unique marker: `__MOTLIE_<uuid>__`
2. Send via `send_keys`: `<command> ; echo "__MOTLIE_<uuid>__ $?" {Enter}`
   (POSIX shells: bash, zsh, sh, dash. For fish: `; echo "__MOTLIE_<uuid>__ $status"`)
3. Poll `capture_with_history()` at intervals until the sentinel line appears (or timeout)
4. Extract everything between the command echo and the sentinel as stdout
5. Parse the exit code from `__MOTLIE_<uuid>__ <exit_code>`
6. Return `ExecOutput { stdout, exit_code }`

**Why sentinel-based**:

- **Works with POSIX shells**: The sentinel uses `echo` + `$?` — works in bash, zsh,
  sh, dash. Fish requires `$status` instead of `$?`. Shell detection precedence:
  (1) `tmux display -p '#{pane_current_command}'` to inspect the running shell,
  (2) fall back to `$SHELL` environment variable. Default is POSIX `$?` if detection
  fails. No special shell features or tmux extensions required beyond `echo`.
- **Reuses existing primitives**: `send_keys()` for input, `capture_with_history()` for
  output. No new transport capabilities or tmux features needed.
- **Non-invasive**: The sentinel echo is appended after the command via `;`. It does not
  modify the command itself. The pane's shell state is unchanged after execution.
- **Cross-pane safe**: The UUID in the sentinel ensures that concurrent `exec()` calls
  on different panes never confuse each other's output. **Same-pane concurrent `exec()`
  is not supported** — overlapping commands on one pane interleave terminal output,
  making boundary extraction ambiguous regardless of unique sentinels. `Target` holds a
  per-target `Mutex` that serializes `exec()` calls to the same pane. Callers needing
  parallel execution should use separate panes.

**Alternatives considered**:

- **Host-level `HostHandle::exec()`** (transport bypass): Rejected — runs outside tmux,
  so the command doesn't execute in the pane's environment (virtualenvs, working directory,
  shell aliases, env vars set by prior commands). The pane's shell is the ground truth for
  the workstream's state; bypassing it loses that context.
- **`tmux run-shell`**: Runs a command in tmux's server context, not in the pane's shell.
  Output appears in the pane but isn't easily captured programmatically. Doesn't inherit
  the pane's shell environment.
- **Control-mode `%output` parsing**: Control mode streams output but has no concept of
  command boundaries. There's no signal for "the command finished." Sentinel-based capture
  adds explicit boundaries.
- **Expect-style prompt detection**: Fragile — depends on knowing the exact prompt format,
  which varies across shells, hosts, and user configurations. The sentinel is universal.

**Limitations and trade-offs**:

- **Assumes a shell prompt**: `exec()` assumes the pane has an active shell waiting for
  input. Calling `exec()` on a pane running `vim` or `top` will inject the command into
  that program — not a shell. Callers should check pane state (via `capture()`) if unsure.
- **Timeout is required**: There is no reliable way to detect a hung command without a
  timeout. A command that never completes will block until timeout, then return an error.
- **No stderr separation**: Tmux scrollback captures the merged terminal output (stdout +
  stderr interleaved as they appear on the PTY). `ExecOutput.stdout` is really "terminal
  output" — true stderr separation would require shell-level redirection that `exec()` does
  not impose. The field is named `stdout` for the common case; callers needing separation
  can use `exec("cmd 2>/tmp/err", ...)` and capture stderr separately.
- **Scrollback pollution**: The sentinel line appears in the pane's scrollback. This is
  generally harmless but visible to humans viewing the pane. A future refinement could
  clear the sentinel line after capture.
- **Polling latency**: `capture_with_history()` is polled at intervals (e.g., 100ms).
  This adds up to one poll interval of latency after command completion. For automation
  use cases this is acceptable; for latency-sensitive scenarios, monitoring-based
  approaches (which stream output in real time) are more appropriate.

**Usage examples**:

```rust
let build = host.session("build").await?.unwrap();

// Run a command and check the result
let result = build.exec("cargo test", Duration::from_secs(300)).await?;
if result.success() {
    println!("Tests passed");
} else {
    println!("Tests failed (exit {}): {}", result.exit_code, result.stdout);
}

// Chain structured commands
let result = build.exec("git pull", Duration::from_secs(30)).await?;
if result.success() {
    build.exec("cargo build --release", Duration::from_secs(600)).await?;
}

// Still use send_text for interactive/fire-and-forget
build.send_text("top").await?;                    // interactive — no output needed
build.send_keys(&KeySequence::parse("{C-c}")?).await?;  // interrupt it
```

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

**Proposal**: The `TransportKind` enum in `transport.rs` already supports this (see DC6).
`TransportKind::Mock(MockTransport)` returns canned responses for `exec()` and
`open_shell()`. It is included in the library (not behind a feature flag) so downstream
consumers can also use it.

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
4. Implement `transport.rs`: `TransportKind` enum, `LocalTransport` (subprocess-based),
   `MockTransport` (canned responses for testing)
5. Implement `discovery.rs`: `list_sessions()`, `list_windows()`, `list_panes()` with
   format string constants and regex filtering
6. Implement `capture.rs`: `capture_pane()`, `capture_pane_history()`, `capture_session()`
7. Implement `control.rs`: `create_session()`, `kill_session()`, `send_keys()`,
   `send_text()`, `rename_session()`, `rename_window()` with shell escaping (addresses OC5)
8. Implement `host.rs`: `HostHandle` and `Target` wiring all of the above (DC16)
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
- `create_session("test", ...)` returns `Target` at session level with correct name and id
- `target.kill()` + `list_sessions()` no longer shows it
- `host.session("nonexistent")` returns `Ok(None)`
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
1. Implement `sink.rs`: `TargetOutput` struct, `SinkKind` enum, `SinkFilter` /
   `CompiledSinkFilter`, `ActionHandle` / `ActionRequest` / `SinkAction`
2. Implement `OutputBus`: `subscribe(sink, channel_capacity)` API,
   fan-out loop that matches `TargetOutput` against compiled filters and dispatches to
   per-sink channels, graceful shutdown with `flush()` on all sinks
3. Implement `StdioSink`: default reference implementation that writes `TargetOutput` to
   stdout with `[host:pane_target]` prefix, immediate flush, no batching
4. Wire `OutputBus` into `monitor.rs`: control mode parser feeds `TargetOutput` into the
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
- `OutputBus` delivers `TargetOutput` to 3 registered sinks concurrently
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
3. Create `bins/tmux-automator/` with `clap` CLI using noun-verb subcommand grouping:

   **`session` — session lifecycle**:
   - `session list [--host <alias>]` — list sessions on one or all hosts
   - `session create <name> [--host <alias>] [--window-name <name>] [--command <cmd>]`
   - `session kill <name> [--host <alias>]`
   - `session rename <old> <new> [--host <alias>]`

   **`target` — operations on any tmux entity (session, window, pane)**:
   - `target list [--host <alias>] [--filter <regex>]` — list panes/windows
   - `target capture <spec> [--host <alias>] [--history <lines>]` — dump content
   - `target send <spec> <input> [--host <alias>]` — send keys
   - `target exec <spec> <command> [--host <alias>]` — run command, return stdout + exit code

   **`monitor` — continuous monitoring lifecycle**:
   - `monitor start [--config <path>]` — start monitoring on all configured targets
   - `monitor status` — show active monitoring sessions

   `<spec>` follows `TargetSpec` syntax: `session`, `session:window`, or
   `session:window.pane`. The `target` noun reflects the unified `Target` type
   (DC16) — capture, send, and exec work at any tmux addressing level.
4. Config file support (TOML) with CLI flag overrides
5. JSON and text log output modes
6. Per-target tracing spans with alias labels

**Deliverable**: Multi-target CLI tool and library. Single process operating across
localhost and N remote hosts.

**Acceptance criteria**:
- `fleet.connect_all()` connects to 3 targets concurrently; one failure does not block others
- CLI `session list --host web-server` returns sessions from the aliased host
- CLI `session create build --host localhost` creates a local session
- CLI `target capture myapp:0.0 --host db-server --history 500` returns scrollback content
- CLI `target exec myapp:0.0 "make test" --host db-server` returns stdout and exit code
- CLI `session kill build` terminates the session
- `monitor start --config rules.toml` starts monitoring on all configured targets
- `monitor status` shows active sessions being monitored

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
