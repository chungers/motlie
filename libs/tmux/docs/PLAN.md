# Tmux Multi-Target Automator — Implementation Plan

Derived from [DESIGN.md](./DESIGN.md). Each task is scoped to produce a compilable,
testable increment. Dependencies are explicit. File paths are relative to `libs/tmux/`.

---

## Phase 1: Types, Transport, and On-Demand Operations (Localhost)

Establish core types, `LocalTransport`, and all localhost on-demand operations.
No SSH, no monitoring.

### 1.0 — Workspace scaffolding

- [x] Add `libs/tmux` to workspace `Cargo.toml` members list
- [x] Create `libs/tmux/Cargo.toml` with initial dependencies:
  `tokio`, `anyhow`, `regex`, `tracing`, `serde`, `uuid`
- [x] Create `src/lib.rs` with module declarations and public re-exports
- [x] Verify `cargo check -p motlie-tmux` passes (empty lib)

### 1.1 — Shared types (`src/types.rs`)

- [x] `PaneAddress`: `pane_id` (authoritative `%<id>`), `session`, `window`, `pane` display fields (DC1)
- [x] `PaneAddress::to_tmux_target() -> String` (`session:window.pane`)
- [x] `PaneAddress::id() -> &str` (returns `pane_id`)
- [x] `PaneAddress::parse(pane_id: &str, address_str: &str) -> Result<Self>` from tmux format output
- [x] `SessionInfo`, `WindowInfo`, `PaneInfo` structs with all fields from DESIGN
- [x] `TargetAddress` enum: `Session(SessionInfo)`, `Window(WindowInfo)`, `Pane(PaneAddress)`
- [x] `TargetLevel` enum: `Session`, `Window`, `Pane`
- [x] `TargetSpec`: builder (`session()`, `.window()`, `.window_name()`, `.pane()`) + `parse()` + `Display`
- [x] `TmuxSocket` enum: `Name(String)`, `Path(String)`
- [x] `HostKeyPolicy` enum: `Verify` (default, `~/.ssh/known_hosts`),
  `TrustFirstUse` (accept + persist on first connect, reject on mismatch),
  `Insecure` (accept all, log warning) — per DC2. Defined here (not in config)
  so `2a.1 SshTransport` can use it without depending on `2b.1 Config`.
- [x] `ExecOutput { stdout: String, exit_code: i32 }` + `success()` helper
- [x] `ScrollbackQuery` enum: `LastLines`, `Until`, `LastLinesUntil`
- [x] Unit tests: `PaneAddress` roundtrip, `TargetSpec` parse/display for all depth levels

**Depends on**: 1.0

### 1.2 — Key escaping (`src/keys.rs`)

- [x] `SpecialKey` enum with all defined keys (Enter, Tab, CtrlC, etc.) + `Raw(String)`
- [x] `KeySegment` enum: `Literal(String)`, `Special(SpecialKey)`
- [x] `KeySequence { segments: Vec<KeySegment> }`
- [x] `KeySequence::parse(input: &str) -> Result<Self>` — `{Enter}`, `{C-c}` inline escapes
- [x] `KeySequence::literal()`, `then_literal()`, `then_key()`, `then_enter()` builder API
- [x] `KeySequence::to_tmux_args(target: &str) -> Vec<Vec<String>>` — split into `-l` and non-`-l` invocations (each inner Vec is one send-keys argument list)
- [x] Unit tests: parse round trips, mixed literal+special sequences, edge cases
  (empty input, consecutive specials, `{` in literal text)

**Depends on**: 1.0

### 1.3 — Transport layer (`src/transport.rs`)

- [x] `TransportKind` enum: `Local(LocalTransport)`, `Mock(MockTransport)` (SSH added in Phase 2a)
- [x] `TransportKind::exec(&self, command: &str) -> Result<String>` — dispatch to variant
- [x] `TransportKind::open_shell(&self) -> Result<ShellChannelKind>` — dispatch to variant
- [x] `ShellChannelKind` enum: `Local(LocalShellChannel)`, `Mock(MockShellChannel)`
- [x] `ShellChannelKind::write()`, `::read()` methods
- [x] `ShellEvent` enum: `Data(Vec<u8>)`, `Eof`
- [x] `LocalTransport`: `exec()` via `tokio::process::Command` with configurable timeout;
  `open_shell()` spawns persistent `bash`/`sh` with piped stdin/stdout
- [x] `MockTransport`: canned `exec()` responses (map of command→output),
  canned `open_shell()` streaming data
- [x] `tmux_prefix(socket: Option<&TmuxSocket>) -> String` — shared helper for `-L`/`-S` flags
- [x] Unit tests: `MockTransport` returns canned data, `LocalTransport` runs `echo hello`

**Depends on**: 1.1

### 1.4 — Discovery (`src/discovery.rs`)

- [x] Format string constants: `LIST_SESSIONS_FMT`, `LIST_WINDOWS_FMT`, `LIST_PANES_FMT`
- [x] `list_sessions(transport, socket) -> Result<Vec<SessionInfo>>` — parse tab-delimited output
- [x] `list_windows(transport, socket, session) -> Result<Vec<WindowInfo>>`
- [x] `list_panes(transport, socket, filter) -> Result<Vec<PaneInfo>>` — optional regex filter
- [x] Parsing logic: split on `\t`, map to struct fields, handle empty/unexpected fields
- [x] Unit tests via `MockTransport`: valid output, empty sessions, malformed lines

**Depends on**: 1.3

### 1.5 — Capture (`src/capture.rs`)

- [x] `capture_pane(transport, socket, target) -> Result<String>` — `capture-pane -p -t`
- [x] `capture_pane_history(transport, socket, target, start) -> Result<String>` — `-S` (captures through visible area end)
- [x] `capture_session(transport, socket, session) -> Result<HashMap<PaneAddress, String>>`
  — calls `list_panes` + `capture_pane` per pane
- [x] `sample_text(transport, socket, target, query) -> Result<String>` — implements
  `LastLines`, `Until`, `LastLinesUntil` scan logic on captured output
- [x] Unit tests: `sample_text` with each `ScrollbackQuery` variant against mock data

**Depends on**: 1.4

### 1.6 — Control (`src/control.rs`)

- [x] Shell escape helper: single-quote wrapping with `'\''` for interior quotes (OC5)
- [x] `create_session(transport, socket, name, window_name, command) -> Result<()>`
- [x] `kill_session(transport, socket, name) -> Result<()>`
- [x] `send_keys(transport, socket, target, keys) -> Result<()>` — renders `KeySequence`
  to tmux commands, executes each
- [x] `send_text(transport, socket, target, text) -> Result<()>` — `send-keys -l`
- [x] `rename_session(transport, socket, current, new) -> Result<()>`
- [x] `rename_window(transport, socket, session, index, new_name) -> Result<()>`
- [x] All functions prepend `tmux_prefix(socket)` to commands
- [x] Unit tests: shell escaping adversarial inputs (`;`, `` ` ``, `$(...)`, newlines,
  null bytes, quotes in session names), mock-based command verification

**Depends on**: 1.3, 1.2

### 1.7 — Host handle + Target wiring (`src/host.rs`)

- [x] `HostHandleInner` struct with `transport`, `socket` (Phase 1 scope; `config` and `session_monitors: RwLock<HashMap>` added in Phase 2a.4)
- [x] `HostHandle` wrapping `Arc<HostHandleInner>`
- [x] `HostHandle` discovery methods: `list_sessions()`, `create_session() -> Result<Target>`,
  `session(name) -> Result<Option<Target>>`, `target(spec) -> Result<Option<Target>>`
- [x] `Target` struct: `Arc<HostHandleInner>` + `TargetAddress`
- [x] `Target` identity: `level()`, `target_string()`, `session_info()`, `window_info()`, `pane_address()`
- [x] `Target` navigation: `children()`, `window(index)`, `pane(index)`, `pane_by_address()`
- [x] `Target` I/O: `send_text()`, `send_keys()`, `capture()`, `capture_with_history()`,
  `sample_text()`, `capture_all()`
- [x] `Target` lifecycle: `kill()`, `rename()`
- [x] `Target::exec()` — sentinel mechanism (DC19): uuid marker, send command with sentinel,
  poll `capture_with_history()`, extract stdout + exit code, per-target `Mutex` for serialization,
  shell detection for `$?` vs `$status` (fish)
- [x] Unit tests: `create_session` → `Target` at session level, navigation produces
  correct `TargetAddress` variants, `exec()` parses sentinel output

**Depends on**: 1.4, 1.5, 1.6

### 1.8 — Integration test (localhost)

- [x] Integration test (behind `#[cfg(test)]` or a feature flag) that:
  - Creates a tmux session on localhost
  - Lists sessions and confirms it appears
  - Captures pane content
  - Sends text + Enter
  - Captures again and confirms output changed
  - `exec("echo hello", 10s)` returns `ExecOutput { stdout: "hello", exit_code: 0 }`
  - Renames session
  - Kills session
  - Lists sessions and confirms it is gone
- [x] Skip if tmux not available (`which tmux` check)

**Depends on**: 1.7

---

## Phase 2a: SSH Transport + Minimal Monitoring

Add `SshTransport` and a thin monitoring vertical slice with control mode parsing.

### 2a.1 — SSH transport (`src/transport.rs` extension)

- [ ] Add `russh` + `russh-keys` dependencies to `Cargo.toml`
- [ ] `SshTransport` struct: russh `Handle`, configurable timeouts
- [ ] `SshTransport::connect(host, user, config) -> Result<Self>` — SSH connect + auth via ssh-agent
- [ ] Host key verification (DC2): implement `HostKeyPolicy` from config —
  `Verify` (parse `~/.ssh/known_hosts`, reject unknown), `TrustFirstUse`
  (accept + persist on first connect), `Insecure` (accept all, log warning)
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
- [ ] `Target::stop_monitoring()` — session-level only, delegates to
  `HostHandle::stop_monitoring_session(&self)`; returns error if called on
  window/pane target
- [ ] Integration test (localhost): start monitor, send output that triggers rule,
  verify action dispatched, `target.stop_monitoring()` cleanly stops,
  verify on-demand operations still work after stop (DC13)

**Depends on**: 2a.2, 2a.3

---

## Phase 2b: Full Rule Engine + Reconnection + Config

### 2b.1 — Configuration (`src/config.rs`)

- [ ] `TmuxAutomatorConfig`: `targets`, `rules`, `reconnect`, `log_json`
- [ ] `HostTarget` enum: `Local { alias, pane_filter, tmux_socket }`,
  `Ssh { host, user, alias, pane_filter, tmux_socket, host_key_policy }`
  — `host_key_policy` uses `HostKeyPolicy` from `types.rs` (defined in 1.1)
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

**Depends on**: 2c.3, 2a.4, **2b.3**

> **Why serial with 2b.3**: Both 2b.3 and 2c.4 modify `monitor.rs` (rule
> evaluation / output publishing) and `host.rs` (reconnection wiring / action
> handle wiring). Running them in parallel creates merge contention on the
> same hot-path code. 2b.3 lands first (rule engine + reconnection), then
> 2c.4 layers the sink pipeline on top of the stabilized monitor.

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

**Depends on**: 2c.4 (which transitively requires 2b.3)

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
- [ ] `--host-key-policy <verify|tofu|insecure>` global CLI flag (DC2),
  overrides per-host config `host_key_policy` field
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
 │     │                 └── 2b.3 Full rules + reconnect
 │     │                      │  (modifies monitor.rs + host.rs)
 │     └── 2c.1 Sink types   │
 │          │                 │
 │          └── 2c.2 Sink kinds
 │               │            │
 │               └── 2c.3 Output bus
 │                    │       │
 │                    └───────┴── 2c.4 Pipeline integration
 │                                │  (needs 2b.3 + 2c.3, serial)
 │                                └── 3.1 Fleet
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
 T9   2b.3 Full rules + reconnect [needs 2a.4 + 2b.1 + 2b.2]
      │    (modifies monitor.rs + host.rs — must land before 2c.4)
      │
 T10  2c.4 Pipeline integration [needs 2c.3 + 2b.3]
      │    (layers sink wiring onto stabilized monitor.rs + host.rs)
      │
───── ── SYNC POINT 4 ─────────────────────────────────────
 T11  3.1 Fleet (fleet.rs) [needs 2c.4]
      │
 T12  3.2 Public API (lib.rs)
      │
 T13  3.3 CLI binary (bins/tmux-automator/)
      │
 T14  3.4 Smoke test
```

### Dev Assignments by Team Size

#### 2 developers

| Dev | Track | Tasks (in order) |
|-----|-------|-----------------|
| **A** | Data path + wiring | 1.0 → 1.1 → 1.3 → 1.4 → 1.5 → **1.7** → 1.8 → 2a.2 → **2a.4** → **2b.3** → **2c.4** → **3.1** → 3.2 → 3.3 |
| **B** | Input + matching + sinks | 1.2 → 2b.2 → 1.6 → 2b.1 → 2a.1 → 2a.3 → 2c.1 → 2c.2 → 2c.3 → 3.4 |

Sync points: **1.7** (B's 1.6 merges with A's 1.4+1.5), **2a.4** (B's 2a.3 merges with A's 2a.2), **2b.3** (B's 2b.1 ready for A), **2c.4** (B's 2c.3 ready; A's 2b.3 done — serial on `monitor.rs`/`host.rs`).

Dev B starts `2b.2 Matcher` immediately after `1.2 Keys` — it depends only on 1.0 and touches an isolated file. This lets B build out the entire matching/config/sink stack while A drives the core data path. 2b.3 and 2c.4 are serialized on A because both modify `monitor.rs` and `host.rs`.

#### 3 developers

| Dev | Focus area | Tasks (in order) |
|-----|-----------|-----------------|
| **A** | Data path (types → transport → discovery → capture → host wiring) | 1.0 → 1.1 → 1.3 → 1.4 → 1.5 → **1.7** → 1.8 → **2a.4** → **3.1** → 3.2 |
| **B** | Input + monitoring (keys → control → SSH → monitor → rules) | 1.2 → 1.6 → 2a.1 → 2a.3 → 2a.2 → **2b.3** → **2c.4** → 3.3 → 3.4 |
| **C** | Matching + sink pipeline (matcher → config → sinks → bus) | 2b.2 → 2b.1 → 2c.1 → 2c.2 → 2c.3 |

Sync points: **1.7** (B's 1.6 ready), **2a.4** (B's 2a.3 + A picks up 2a.2 after 1.8), **2b.3** (C's 2b.1 ready for B), **2c.4** (B takes this after 2b.3 — serial on `monitor.rs`/`host.rs`; needs C's 2c.3), **3.1** (all tracks converge).

Dev C is fully independent through T1–T7 — they only touch `matcher.rs`, `config.rs`, `sink.rs`, and `sinks/`. Dev B owns the `monitor.rs`/`host.rs` serialization: 2b.3 (rules + reconnection) then 2c.4 (sink wiring).

#### 4 developers

Split Track B into input/control (B1) and SSH/monitoring (B2):

| Dev | Focus area | Tasks |
|-----|-----------|-------|
| **A** | Data path | 1.0 → 1.1 → 1.3 → 1.4 → 1.5 → **1.7** → 1.8 |
| **B1** | Input + control | 1.2 → 1.6 → (pick up 4.1, 4.2 hardening while waiting) |
| **B2** | SSH + monitoring + sink wiring | 2a.1 → 2a.3 → 2a.2 → **2a.4** → **2b.3** → **2c.4** → 3.3 → 3.4 |
| **C** | Matching + sinks | 2b.2 → 2b.1 → 2c.1 → 2c.2 → 2c.3 → **3.1** → 3.2 |

B2 owns the `monitor.rs`/`host.rs` serialization: 2b.3 (rules) then 2c.4 (sink wiring) land sequentially by the same dev, eliminating merge contention. B1 finishes early (1.2 + 1.6 are small) and pivots to Phase 4 hardening tasks (4.1 tmux version compat, 4.2 test expansion, 4.3 Docker E2E).

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
| `host.rs` | Track A | Track B adds monitoring (2a.4) + reconnection (2b.3), Track C adds ActionHandle (2c.4). Serialized: 2a.4 → 2b.3 → 2c.4. |
| `monitor.rs` | Track B | Track B adds rules (2b.3), Track C wires OutputBus (2c.4). Serialized: 2a.2 → 2b.3 → 2c.4. |
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
