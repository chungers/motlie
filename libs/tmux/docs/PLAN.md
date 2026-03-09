# Tmux Multi-Target Automator — Implementation Plan

Derived from [DESIGN.md](./DESIGN.md). Each task is scoped to produce a compilable,
testable increment. Dependencies are explicit. File paths are relative to `libs/tmux/`.

---

## Phase 1: Types, Transport, and On-Demand Operations (Localhost)

Establish core types, `LocalTransport`, and all localhost on-demand operations.
No SSH, no monitoring.

### 1.0 — Workspace scaffolding

- [ ] Add `libs/tmux` to workspace `Cargo.toml` members list
- [ ] Create `libs/tmux/Cargo.toml` with initial dependencies:
  `tokio`, `anyhow`, `regex`, `tracing`, `serde`, `uuid`
- [ ] Create `src/lib.rs` with module declarations and public re-exports
- [ ] Verify `cargo check -p motlie-tmux` passes (empty lib)

### 1.1 — Shared types (`src/types.rs`)

- [ ] `PaneAddress`: `pane_id` (authoritative `%<id>`), `session`, `window`, `pane` display fields (DC1)
- [ ] `PaneAddress::to_tmux_target() -> String` (`session:window.pane`)
- [ ] `PaneAddress::id() -> &str` (returns `pane_id`)
- [ ] `PaneAddress::parse(s: &str) -> Result<Self>` from tmux format output
- [ ] `SessionInfo`, `WindowInfo`, `PaneInfo` structs with all fields from DESIGN
- [ ] `TargetAddress` enum: `Session(SessionInfo)`, `Window(WindowInfo)`, `Pane(PaneAddress)`
- [ ] `TargetLevel` enum: `Session`, `Window`, `Pane`
- [ ] `TargetSpec`: builder (`session()`, `.window()`, `.window_name()`, `.pane()`) + `parse()` + `Display`
- [ ] `TmuxSocket` enum: `Name(String)`, `Path(String)`
- [ ] `ExecOutput { stdout: String, exit_code: i32 }` + `success()` helper
- [ ] `ScrollbackQuery` enum: `LastLines`, `Until`, `LastLinesUntil`
- [ ] Unit tests: `PaneAddress` roundtrip, `TargetSpec` parse/display for all depth levels

**Depends on**: 1.0

### 1.2 — Key escaping (`src/keys.rs`)

- [ ] `SpecialKey` enum with all defined keys (Enter, Tab, CtrlC, etc.) + `Raw(String)`
- [ ] `KeySegment` enum: `Literal(String)`, `Special(SpecialKey)`
- [ ] `KeySequence { segments: Vec<KeySegment> }`
- [ ] `KeySequence::parse(input: &str) -> Result<Self>` — `{Enter}`, `{C-c}` inline escapes
- [ ] `KeySequence::literal()`, `then_literal()`, `then_key()`, `then_enter()` builder API
- [ ] `KeySequence::to_tmux_commands(target: &str) -> Vec<String>` — split into `-l` and non-`-l` invocations
- [ ] Unit tests: parse round trips, mixed literal+special sequences, edge cases
  (empty input, consecutive specials, `{` in literal text)

**Depends on**: 1.0

### 1.3 — Transport layer (`src/transport.rs`)

- [ ] `TransportKind` enum: `Local(LocalTransport)`, `Mock(MockTransport)` (SSH added in Phase 2a)
- [ ] `TransportKind::exec(&self, command: &str) -> Result<String>` — dispatch to variant
- [ ] `TransportKind::open_shell(&self) -> Result<ShellChannelKind>` — dispatch to variant
- [ ] `ShellChannelKind` enum: `Local(LocalShellChannel)`, `Mock(MockShellChannel)`
- [ ] `ShellChannelKind::write()`, `::read()` methods
- [ ] `ShellEvent` enum: `Data(Vec<u8>)`, `Eof`
- [ ] `LocalTransport`: `exec()` via `tokio::process::Command` with configurable timeout;
  `open_shell()` spawns persistent `bash`/`sh` with piped stdin/stdout
- [ ] `MockTransport`: canned `exec()` responses (map of command→output),
  canned `open_shell()` streaming data
- [ ] `tmux_prefix(socket: Option<&TmuxSocket>) -> String` — shared helper for `-L`/`-S` flags
- [ ] Unit tests: `MockTransport` returns canned data, `LocalTransport` runs `echo hello`

**Depends on**: 1.1

### 1.4 — Discovery (`src/discovery.rs`)

- [ ] Format string constants: `LIST_SESSIONS_FMT`, `LIST_WINDOWS_FMT`, `LIST_PANES_FMT`
- [ ] `list_sessions(transport, socket) -> Result<Vec<SessionInfo>>` — parse tab-delimited output
- [ ] `list_windows(transport, socket, session) -> Result<Vec<WindowInfo>>`
- [ ] `list_panes(transport, socket, filter) -> Result<Vec<PaneInfo>>` — optional regex filter
- [ ] Parsing logic: split on `\t`, map to struct fields, handle empty/unexpected fields
- [ ] Unit tests via `MockTransport`: valid output, empty sessions, malformed lines

**Depends on**: 1.3

### 1.5 — Capture (`src/capture.rs`)

- [ ] `capture_pane(transport, socket, target) -> Result<String>` — `capture-pane -p -t`
- [ ] `capture_pane_history(transport, socket, target, start, end) -> Result<String>` — `-S`/`-E`
- [ ] `capture_session(transport, socket, session) -> Result<HashMap<PaneAddress, String>>`
  — calls `list_panes` + `capture_pane` per pane
- [ ] `sample_text(transport, socket, target, query) -> Result<String>` — implements
  `LastLines`, `Until`, `LastLinesUntil` scan logic on captured output
- [ ] Unit tests: `sample_text` with each `ScrollbackQuery` variant against mock data

**Depends on**: 1.4

### 1.6 — Control (`src/control.rs`)

- [ ] Shell escape helper: single-quote wrapping with `'\''` for interior quotes (OC5)
- [ ] `create_session(transport, socket, name, window_name, command) -> Result<()>`
- [ ] `kill_session(transport, socket, name) -> Result<()>`
- [ ] `send_keys(transport, socket, target, keys) -> Result<()>` — renders `KeySequence`
  to tmux commands, executes each
- [ ] `send_text(transport, socket, target, text) -> Result<()>` — `send-keys -l`
- [ ] `rename_session(transport, socket, current, new) -> Result<()>`
- [ ] `rename_window(transport, socket, session, index, new_name) -> Result<()>`
- [ ] All functions prepend `tmux_prefix(socket)` to commands
- [ ] Unit tests: shell escaping adversarial inputs (`;`, `` ` ``, `$(...)`, newlines,
  null bytes, quotes in session names), mock-based command verification

**Depends on**: 1.3, 1.2

### 1.7 — Host handle + Target wiring (`src/host.rs`)

- [ ] `HostHandleInner` struct with `transport`, `config`, `session_monitors: RwLock<HashMap>`
- [ ] `HostHandle` wrapping `Arc<HostHandleInner>`
- [ ] `HostHandle` discovery methods: `list_sessions()`, `create_session() -> Result<Target>`,
  `session(name) -> Result<Option<Target>>`, `target(spec) -> Result<Option<Target>>`
- [ ] `Target` struct: `Arc<HostHandleInner>` + `TargetAddress`
- [ ] `Target` identity: `level()`, `target_string()`, `session_info()`, `window_info()`, `pane_address()`
- [ ] `Target` navigation: `children()`, `window(index)`, `pane(index)`, `pane_by_address()`
- [ ] `Target` I/O: `send_text()`, `send_keys()`, `capture()`, `capture_with_history()`,
  `sample_text()`, `capture_all()`
- [ ] `Target` lifecycle: `kill()`, `rename()`
- [ ] `Target::exec()` — sentinel mechanism (DC19): uuid marker, send command with sentinel,
  poll `capture_with_history()`, extract stdout + exit code, per-target `Mutex` for serialization,
  shell detection for `$?` vs `$status` (fish)
- [ ] Unit tests: `create_session` → `Target` at session level, navigation produces
  correct `TargetAddress` variants, `exec()` parses sentinel output

**Depends on**: 1.4, 1.5, 1.6

### 1.8 — Integration test (localhost)

- [ ] Integration test (behind `#[cfg(test)]` or a feature flag) that:
  - Creates a tmux session on localhost
  - Lists sessions and confirms it appears
  - Captures pane content
  - Sends text + Enter
  - Captures again and confirms output changed
  - `exec("echo hello", 10s)` returns `ExecOutput { stdout: "hello", exit_code: 0 }`
  - Renames session
  - Kills session
  - Lists sessions and confirms it is gone
- [ ] Skip if tmux not available (`which tmux` check)

**Depends on**: 1.7

---

## Phase 2a: SSH Transport + Minimal Monitoring

Add `SshTransport` and a thin monitoring vertical slice with control mode parsing.

### 2a.1 — SSH transport (`src/transport.rs` extension)

- [ ] Add `russh` + `russh-keys` dependencies to `Cargo.toml`
- [ ] `SshTransport` struct: russh `Handle`, configurable timeouts
- [ ] `SshTransport::connect(host, user, config) -> Result<Self>` — SSH connect + auth via ssh-agent
- [ ] Host key verification (DC2): `~/.ssh/known_hosts` parsing, TOFU option, insecure flag
- [ ] `SshTransport::exec()` — open exec channel, capture stdout, close channel
- [ ] `SshTransport::open_shell()` — PTY channel with shell, piped I/O
- [ ] Add `Ssh(SshTransport)` variant to `TransportKind` and `ShellChannelKind`
- [ ] Actionable error messages for SSH agent failures (OC3)
- [ ] Unit tests: mock-based (no real SSH server needed at this stage)

**Depends on**: 1.3

### 2a.2 — Control mode parser (`src/monitor.rs`)

- [ ] `SessionMonitor` struct: session name, rules, cooldown state
- [ ] Control mode stream parser: parse `%output %<pane_id> <data>` frames from
  `tmux -C attach -t <session>` output
- [ ] Handle other control mode messages gracefully (`%begin`, `%end`, `%error`, etc.)
- [ ] Rule evaluation against parsed output (initially: single compiled rule)
- [ ] Action dispatch: `SendKeys` via bounded `mpsc` channel + semaphore (DC4)
- [ ] `SessionMonitor::run()` — main loop: read from shell, parse, evaluate, dispatch
- [ ] Stop signal via `watch::Receiver<bool>`, clean shutdown on signal or connection drop
- [ ] Warn-level logging for malformed lines and failed actions (P9)
- [ ] Unit tests: control mode frame parsing, rule matching, dispatch ordering

**Depends on**: 1.7

### 2a.3 — Pipe-pane fallback (`src/pipe.rs`)

- [ ] `PipeManager` struct tracking active pipes
- [ ] `PipeManager::setup(transport, socket, panes)` — create FIFOs/files, attach `pipe-pane`
- [ ] `PipeManager::cleanup(transport, socket)` — detach pipes, remove files (P4)
- [ ] `Drop` impl: log warning if cleanup not called
- [ ] Default to append-file sink (`cat >> file` + `tail -f`), FIFO as opt-in
- [ ] Unit tests via `MockTransport`

**Depends on**: 1.3

### 2a.4 — Monitor handle wiring

- [ ] `SessionMonitorHandle`: `Target` + `stop_tx` + `task: Mutex<Option<JoinHandle>>`
- [ ] `SessionMonitorHandle::shutdown()` — signal stop, flush, join task,
  cleanup pipes if fallback active
- [ ] `SessionMonitorHandle::is_active()`
- [ ] `Deref<Target=Target>` for `SessionMonitorHandle`
- [ ] `MonitorHandle`: `HashMap<String, SessionMonitorHandle>`, `shutdown()`,
  `stop_session()`, `get()`, `get_by_spec()`, `active_sessions()`
- [ ] `HostHandle::start_monitoring()` — discovers sessions, spawns per-session monitors,
  returns `MonitorHandle`
- [ ] `HostHandle::start_monitoring_session()`, `stop_monitoring_session()`,
  `stop_monitoring()`, `monitored_sessions()` (DC13)
- [ ] `Target::start_monitoring()` — session-level only, returns `SessionMonitorHandle`
- [ ] Integration test (localhost): start monitor, send output that triggers rule,
  verify action dispatched, shutdown cleanly

**Depends on**: 2a.2, 2a.3

---

## Phase 2b: Full Rule Engine + Reconnection + Config

### 2b.1 — Configuration (`src/config.rs`)

- [ ] `TmuxAutomatorConfig`: `targets`, `rules`, `reconnect`, `log_json`
- [ ] `HostTarget` enum: `Local { alias, pane_filter, tmux_socket }`,
  `Ssh { host, user, alias, pane_filter, tmux_socket }`
- [ ] `TriggerRule`: `name`, `pane_filter`, `pattern`, `action`, `cooldown` — serde-deserializable
- [ ] `Action` enum: `SendKeys { keys }`, `Log { level, message }`
- [ ] `ReconnectPolicy`: `initial_delay`, `max_delay`, `multiplier` with defaults
- [ ] `CompiledRule`: compiled from `TriggerRule`, holds `MatcherKind` + compiled pane filter
- [ ] `TriggerRule::compile() -> Result<CompiledRule>` — error includes rule name context
- [ ] `CompiledRule::with_matcher()` — programmatic construction with any `MatcherKind`
- [ ] `serde` derive for all config types, TOML deserialization support
- [ ] Unit tests: deserialize TOML config, compile rules, invalid regex error messages

**Depends on**: 2b.2 (for `MatcherKind`)

### 2b.2 — Content matcher (`src/matcher.rs`)

- [ ] `MatcherKind` enum: `Regex`, `Substring`, `LineCount`, `WordList`,
  `AllOf(Vec)`, `AnyOf(Vec)`, `Not(Box)`
- [ ] `MatcherKind::matches(&mut self, text: &str) -> bool` for each variant
- [ ] `MatcherKind::reset(&mut self)` — clear state for restarts
- [ ] `MatcherKind::name(&self) -> String` — human-readable for logging
- [ ] `Clone` derive for full tree cloning
- [ ] `WordList` uses `\b` regex boundaries internally
- [ ] Unit tests: each variant individually, combinator nesting, stateful `LineCount`
  accumulation and reset

**Depends on**: 1.0

### 2b.3 — Expanded monitor + reconnection

- [ ] Multi-rule evaluation in `SessionMonitor::run()` with compiled rules
- [ ] Per-pane cooldown timers: `HashMap<String, HashMap<String, Instant>>` (pane_id → rule_name → last_fired)
- [ ] `Log` action type: emit structured log at configured level
- [ ] Reconnection logic in `HostHandle` (SSH targets only):
  exponential backoff per `ReconnectPolicy`, re-discover sessions on reconnect,
  resume monitoring with same rules
- [ ] Unit tests: cooldown prevents rapid re-fire, reconnect resumes monitoring

**Depends on**: 2a.4, 2b.1, 2b.2

---

## Phase 2c: Output Sink Pipeline

### 2c.1 — Sink types (`src/sink.rs`)

- [ ] `TargetOutput` struct: `source: TargetAddress`, `host`, `content`, `timestamp`
- [ ] `TargetOutput` accessors: `session_name()`, `pane_id()`, `target_string()`
- [ ] `SinkFilter`: `host`, `session`, `window`, `pane` (all optional regex strings),
  `content: Option<MatcherKind>`
- [ ] `CompiledSinkFilter`: compiled regexes + `MatcherKind`,
  `matches(&mut self, output: &TargetOutput) -> bool`
- [ ] `SinkAction` enum: `SendKeys`, `SendText`, `KillSession`, `RenameSession`
- [ ] `ActionRequest` struct: `host`, `target: TargetAddress`, `action: SinkAction`
- [ ] `ActionHandle`: wraps `mpsc::Sender<ActionRequest>`,
  provides `send()`, `send_keys()`, `send_text()`, `kill_session()`,
  `rename_session()`, `respond(output, action)`
- [ ] `SinkId` opaque type

**Depends on**: 2b.2

### 2c.2 — Sink kinds (`src/sink.rs`, `src/sinks/`)

- [ ] `SinkKind` enum: `Stdio(StdioSink)`, `Callback(CallbackSink)`
- [ ] `SinkKind::name()`, `filters()`, `write()`, `flush()` dispatch methods
- [ ] `CallbackSink`: `name`, `filters`, `state: Arc<dyn Any + Send + Sync>`,
  `on_output: fn(...)`, `on_flush: Option<fn(...)>`
- [ ] `StdioSink` in `src/sinks/stdio.rs`: `StdioFormat` enum (Raw, Prefixed, Json),
  immediate write to stdout, no batching
- [ ] `JoinedSink`: wraps inner `SinkKind`, `LabelFormat` enum (Bracketed, Prompt, Custom fn),
  source coalescing via `last_source`
- [ ] `SourceLabel` struct: `host`, `target: TargetAddress`, `short()`, `minimal()` formatters
- [ ] `StreamChunk` struct: `source: SourceLabel`, `output: TargetOutput`

**Depends on**: 2c.1

### 2c.3 — Output bus (`src/sink.rs`)

- [ ] `OutputBus::new()`
- [ ] `subscribe(sink: SinkKind, channel_capacity) -> SinkId` — spawn per-sink tokio task
- [ ] `subscribe_joined(filters, capacity) -> (SinkId, mpsc::Receiver<StreamChunk>)`
- [ ] `unsubscribe(id) -> Result<()>` — signal stop, flush, join task
- [ ] `publish(output: TargetOutput)` — fan out to all matching sinks via `try_send`,
  log drops at debug level
- [ ] `shutdown() -> Result<()>` — signal all sinks, flush, join all tasks
- [ ] `SinkEntry` internal: id, name, tx, compiled filters, task handle
- [ ] Unit tests: fan-out to 3 sinks, slow sink doesn't block others,
  filter matching (AND within / OR across), shutdown flushes

**Depends on**: 2c.2

### 2c.4 — Pipeline integration

- [ ] Wire `OutputBus` into `monitor.rs`: publish `TargetOutput` alongside rule evaluation
- [ ] Wire `ActionHandle` into `HostHandle`: sink-initiated actions route through DC4 queue
- [ ] `Fleet::output_bus()` accessor; sinks registered before `start_monitoring()`
- [ ] Integration test: monitor publishes output → `StdioSink` receives and formats,
  `CallbackSink` receives and accumulates, `ActionHandle` routes action back to target

**Depends on**: 2c.3, 2a.4

---

## Phase 3: Multi-Target Fleet + CLI

### 3.1 — Fleet (`src/fleet.rs`)

- [ ] `Fleet::new(config: TmuxAutomatorConfig)`
- [ ] `Fleet::connect_all() -> Vec<HostStatus>` — concurrent connect via `JoinSet`,
  per-target error isolation
- [ ] `Fleet::host(name) -> Option<&HostHandle>` — alias or `host:port` lookup (DC9)
- [ ] `Fleet::hosts()` iterator
- [ ] `Fleet::start_monitoring(rules)`, `shutdown()`
- [ ] `Fleet::start_monitoring_host()`, `stop_monitoring_host()` — per-host granularity
- [ ] `Fleet::output_bus()` accessor (owns `OutputBus`, shares via `Arc`)
- [ ] `HostStatus` enum: `Disconnected`, `Connecting`, `Connected`, `Monitoring { sessions }`, `Error(String)`
- [ ] Workstream registry: `bind()`, `unbind()`, `find()`, `workstreams()`
- [ ] Shutdown watch channel: `shutdown()` signals all hosts
- [ ] Unit tests: multi-host connect with one failure, alias conflict detection,
  workstream bind/find/unbind

**Depends on**: 2c.4, 2b.3

### 3.2 — `lib.rs` public API surface

- [ ] Re-export all consumer-facing types from `lib.rs`:
  `Fleet`, `HostHandle`, `Target`, `TargetSpec`, `SessionMonitorHandle`, `MonitorHandle`,
  `TmuxAutomatorConfig`, `HostTarget`, `TriggerRule`, `Action`, `ReconnectPolicy`,
  `KeySequence`, `SpecialKey`, `ScrollbackQuery`, `ExecOutput`,
  `OutputBus`, `SinkKind`, `StdioSink`, `CallbackSink`, `SinkFilter`, `MatcherKind`,
  `ActionHandle`, `TargetOutput`, `StreamChunk`, `JoinedSink`,
  `SessionInfo`, `WindowInfo`, `PaneInfo`, `PaneAddress`, `TargetAddress`, `TmuxSocket`
- [ ] Doc comments on `lib.rs` with usage example

**Depends on**: 3.1

### 3.3 — CLI binary (`bins/tmux-automator/`)

- [ ] Create `bins/tmux-automator/Cargo.toml` — depends on `motlie-tmux`, `clap`, `tokio`,
  `tracing-subscriber`, `toml`
- [ ] Add to workspace members
- [ ] `main.rs`: CLI skeleton with `clap` noun-verb subcommands
- [ ] **`session` noun**:
  - `session list [--host]`
  - `session create <name> [--host] [--window-name] [--command]`
  - `session kill <name> [--host]`
  - `session rename <old> <new> [--host]`
- [ ] **`target` noun**:
  - `target list [--host] [--filter]`
  - `target capture <spec> [--host] [--history]`
  - `target send <spec> <input> [--host]`
  - `target exec <spec> <command> [--host] [--timeout]`
- [ ] **`monitor` noun**:
  - `monitor start [--config]`
  - `monitor status`
- [ ] Config file loading: TOML → `TmuxAutomatorConfig`, CLI flag overrides
- [ ] Signal handling: `tokio::signal` for SIGINT/SIGTERM → `fleet.shutdown()` (P5)
- [ ] Tracing init: text or JSON output modes, per-target tracing spans
- [ ] Default `StdioSink` registered on `OutputBus` for monitor output

**Depends on**: 3.2

### 3.4 — End-to-end smoke test

- [ ] Manual test script or integration test:
  - Start CLI with localhost config
  - Create session, list, capture, send, exec, rename, kill
  - Start monitor with a rule, trigger it, verify action logged
  - Ctrl-C gracefully shuts down

**Depends on**: 3.3

---

## Phase 4: Hardening + Testing

### 4.1 — Tmux version compatibility (OC4)

- [ ] Runtime `tmux -V` detection at startup
- [ ] Feature matrix: validate required features against detected version
  (control mode, `capture-pane -p`, `pipe-pane -o`, `#{pane_id}`)
- [ ] Clear error messages for unsupported versions
- [ ] CI matrix: test against tmux 2.x, 3.x, latest

### 4.2 — Expanded test coverage

- [ ] Expand `MockTransport` test suite: error paths, timeouts, malformed tmux output
- [ ] Shell escaping fuzz tests or property tests (adversarial session names, text input)
- [ ] `OutputBus` stress test: high-throughput publish with slow/full sinks
- [ ] Cooldown timer accuracy tests
- [ ] Reconnection resilience test: simulated SSH drop + recover

### 4.3 — Docker-based E2E (OC6)

- [ ] Dockerfile: SSH server + tmux + test sessions
- [ ] E2E test: connect via SSH, full lifecycle (create, monitor, trigger, capture, kill)
- [ ] CI integration: run E2E tests in Docker on PR

### 4.4 — Documentation

- [ ] Document minimum tmux version and known limitations
- [ ] Document performance characteristics (latency, throughput)
- [ ] Document pipe-pane fallback path and when to use it
- [ ] Crate-level rustdoc with examples

---

## Phase 5 (Future): TUI Interface

Out of current scope. Listed for continuity.

- [ ] `TuiSink` registered with `OutputBus` (lives in `bins/tmux-automator/`)
- [ ] Ratatui-based terminal UI: live pane display, tree navigation, interactive input
- [ ] Monitoring dashboard: rule status, trigger history, host connection status
- [ ] Rendering cadence managed by sink (60fps batching)

**Depends on**: Phases 1–3, 2c stable

---

## Task Ordering Summary

```
1.0 Scaffolding
 ├── 1.1 Types
 ├── 1.2 Keys
 ├── 1.3 Transport ─── 2a.1 SSH Transport
 │    ├── 1.4 Discovery
 │    ├── 1.5 Capture ──┐
 │    ├── 1.6 Control ──┤
 │    └── 2a.3 Pipes    │
 │                      │
 │    1.7 Host+Target ◄─┘
 │     │
 │     ├── 1.8 Localhost integration test
 │     │
 │     └── 2a.2 Monitor parser
 │          │
 │          └── 2a.4 Monitor handles
 │               │
 │    2b.2 Matcher ──── 2b.1 Config
 │     │                 │
 │     │    2b.3 Full rules + reconnect ◄─┘
 │     │     │
 │     └── 2c.1 Sink types
 │          │
 │          └── 2c.2 Sink kinds
 │               │
 │               └── 2c.3 Output bus
 │                    │
 │                    └── 2c.4 Pipeline integration
 │                         │
 │                         └── 3.1 Fleet
 │                              │
 │                              └── 3.2 Public API
 │                                   │
 │                                   └── 3.3 CLI binary
 │                                        │
 │                                        └── 3.4 Smoke test
 │
 └── 4.x Hardening (parallel with Phase 3)
```

---

## Multi-Developer Parallelism

The dependency graph has three largely independent module tracks after the
shared scaffolding (1.0). Each track touches different source files, minimizing
merge conflicts. The sync points where tracks must converge are called out
explicitly.

### Parallelism Map (by time)

```
Time  Track A (data path)      Track B (input/monitor)    Track C (matching/sinks)
───── ─────────────────────── ──────────────────────────── ────────────────────────
 T0   1.0 Scaffolding ◄──────── shared gate ──────────────► 1.0 Scaffolding
      │                         │                           │
 T1   1.1 Types (types.rs)      1.2 Keys (keys.rs)          2b.2 Matcher (matcher.rs)
      │                         │                           │
 T2   1.3 Transport             │ (waits for 1.3)           2b.1 Config (config.rs)
      │  (transport.rs)         │                           │
 T3   1.4 Discovery             1.6 Control (control.rs)    2c.1 Sink types (sink.rs)
      │  (discovery.rs)         │  [needs 1.2 + 1.3]       │
 T4   1.5 Capture               │                           2c.2 Sink kinds (sinks/)
      │  (capture.rs)           │                           │
      │                         │                           2c.3 Output bus (sink.rs)
      │                         │                           │
───── ── SYNC POINT 1 ──────── ─┘                           │ (waits for 2a.4)
 T5   1.7 Host+Target (host.rs)                             │
      │  [needs 1.4 + 1.5 + 1.6]                           │
      │                                                     │
 T6   1.8 Integration test      2a.1 SSH (transport.rs)     │
      │                         2a.3 Pipes (pipe.rs)        │
      │                         │                           │
 T7   2a.2 Monitor parser       │                           │
      │  (monitor.rs)           │                           │
      │  [needs 1.7]            │                           │
      │                         │                           │
───── ── SYNC POINT 2 ──────── ─┘                           │
 T8   2a.4 Monitor handles                                  │
      │  [needs 2a.2 + 2a.3]                               │
      │                                                     │
───── ── SYNC POINT 3 ──────────────────────────────────── ─┘
 T9   2c.4 Pipeline integration [needs 2a.4 + 2c.3]
      │
      │   2b.3 Full rules + reconnect [needs 2a.4 + 2b.1 + 2b.2]
      │    │    (can run parallel with 2c.4 — different files)
      │    │
───── ── SYNC POINT 4 ─────────────────────────────────────
 T10  3.1 Fleet (fleet.rs) [needs 2c.4 + 2b.3]
      │
 T11  3.2 Public API (lib.rs)
      │
 T12  3.3 CLI binary (bins/tmux-automator/)
      │
 T13  3.4 Smoke test
```

### Dev Assignments by Team Size

#### 2 developers

| Dev | Track | Tasks (in order) |
|-----|-------|-----------------|
| **A** | Data path + wiring | 1.0 → 1.1 → 1.3 → 1.4 → 1.5 → **1.7** → 1.8 → 2a.2 → **2a.4** → **2c.4** → **3.1** → 3.2 → 3.3 |
| **B** | Input + matching + sinks | 1.2 → 2b.2 → 1.6 → 2b.1 → 2a.1 → 2a.3 → 2c.1 → 2c.2 → 2c.3 → **2b.3** → 3.4 |

Sync points: **1.7** (B's 1.6 merges with A's 1.4+1.5), **2a.4** (B's 2a.3 merges with A's 2a.2), **2c.4** (B's 2c.3 merges with A's 2a.4), **3.1** (B's 2b.3 merges with A's 2c.4).

Dev B starts `2b.2 Matcher` immediately after `1.2 Keys` — it depends only on 1.0 and touches an isolated file. This lets B build out the entire matching/config/sink stack while A drives the core data path.

#### 3 developers

| Dev | Focus area | Tasks (in order) |
|-----|-----------|-----------------|
| **A** | Data path (types → transport → discovery → capture → host wiring) | 1.0 → 1.1 → 1.3 → 1.4 → 1.5 → **1.7** → 1.8 → **2a.4** → **3.1** → 3.2 |
| **B** | Input + monitoring (keys → control → SSH → monitor → CLI) | 1.2 → 1.6 → 2a.1 → 2a.3 → 2a.2 → **2b.3** → 3.3 → 3.4 |
| **C** | Matching + sink pipeline (matcher → config → sinks → bus) | 2b.2 → 2b.1 → 2c.1 → 2c.2 → 2c.3 → **2c.4** |

Sync points: **1.7** (B's 1.6 ready), **2a.4** (B's 2a.3 + A picks up 2a.2 after 1.8), **2b.3** (C's 2b.1 ready for B), **2c.4** (A's 2a.4 ready for C), **3.1** (all three tracks converge).

Dev C is fully independent through T1–T7 — they only touch `matcher.rs`, `config.rs`, `sink.rs`, and `sinks/`. First sync with the other tracks is at 2c.4.

#### 4 developers

Split Track B into input/control (B1) and SSH/monitoring (B2):

| Dev | Focus area | Tasks |
|-----|-----------|-------|
| **A** | Data path | 1.0 → 1.1 → 1.3 → 1.4 → 1.5 → **1.7** → 1.8 |
| **B1** | Input + control | 1.2 → 1.6 → (pick up 4.2 test hardening while waiting) |
| **B2** | SSH + monitoring | 2a.1 → 2a.3 → 2a.2 → **2a.4** → **2b.3** → 3.3 → 3.4 |
| **C** | Matching + sinks | 2b.2 → 2b.1 → 2c.1 → 2c.2 → 2c.3 → **2c.4** → **3.1** → 3.2 |

B1 finishes early (1.2 + 1.6 are small) and can pivot to Phase 4 hardening tasks (4.1 tmux version compat, 4.2 test expansion, 4.3 Docker E2E) which are independent of the main feature tracks.

### File Ownership (Conflict Avoidance)

Each track primarily touches distinct files. When two tracks must modify the
same file, that's a sync point — one merges first, the other rebases.

| File | Primary owner | Touched by others at |
|------|--------------|---------------------|
| `types.rs` | Track A | — (stable after 1.1) |
| `keys.rs` | Track B | — |
| `transport.rs` | Track A (1.3), then Track B (2a.1) | Sync after 1.3 |
| `discovery.rs` | Track A | — |
| `capture.rs` | Track A | — |
| `control.rs` | Track B | — |
| `host.rs` | Track A | Track B contributes monitoring methods (2a.4) |
| `monitor.rs` | Track B | Track C wires OutputBus in (2c.4) |
| `pipe.rs` | Track B | — |
| `matcher.rs` | Track C | — |
| `config.rs` | Track C | — |
| `sink.rs` | Track C | — |
| `sinks/` | Track C | — |
| `fleet.rs` | Converge point | Needs all tracks done |
| `lib.rs` | Track A (initial), final at 3.2 | All tracks add re-exports |

### Phase 4 Parallelism

All Phase 4 tasks (4.1–4.4) are independent of each other and can be split
across developers. They can also begin as soon as their prerequisites are met:

| Task | Can start after | Parallelizable with |
|------|----------------|-------------------|
| 4.1 Tmux version compat | 1.3 (transport exists) | Everything after 1.3 |
| 4.2 Expanded test coverage | Each module's initial impl | Phase 3 work |
| 4.3 Docker E2E | 2a.1 (SSH transport exists) | Phase 3 work |
| 4.4 Documentation | 3.2 (public API finalized) | 3.3 CLI, 3.4 smoke test |

A dev who finishes their track early (e.g., B1 in the 4-dev scenario) can
immediately start Phase 4 work without blocking or being blocked.

## Conventions

- Each task should be a PR-sized unit (reviewable in isolation)
- Every task ends with `cargo check` and `cargo test` passing
- Use `MockTransport` for unit tests; real tmux for integration tests
- Integration tests gated on `which tmux` availability check
- Follow workspace patterns: `anyhow` for errors, `tracing` for logs, `serde` for config
