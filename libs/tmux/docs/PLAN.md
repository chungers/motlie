# Tmux Multi-Target Automator — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-12 | @claude | Phase 1.11 — address PR #71 R2: fix `connect(self)` in 1.11g, add socket mutual-exclusion to 1.11j test cases, fix fleet example. R1: 1.11l inspection seam. |
| 2026-03-12 | @claude | Phase 1.11 R2: address feedback — consolidate `SshUri` into `SshConfig` (no new type), support both nassh `;` and query `?` param syntax, no canonical-component duplication. 14 tasks (1.11a–n). |
| 2026-03-11 | @claude | Phase 1.10 implemented: all 14 tasks (1.10a–n) completed. Code changes: MockTransport with_error + Vec ordering, open_shell PTY params, TransportKind::is_healthy(), TargetSpec::pane() returns Result, HostHandle::local_with_timeout(), Target::rename() returns new Target. Doc comments: TrustFirstUse fail-closed, pane rename asymmetry, active-window drift, capture scope, overlap_deduplicate warn, dual timeouts, history-limit semantics. |
| 2026-03-12 | @claude | Phase 1.10 — address PR #68 R6: restore 1.10 as hard gate before 2a.2, consistent with user intent ("slotted for fixes now before the next phase"). Fixed dependency note, task ordering diagram, and notes section. |
| 2026-03-12 | @claude | Simplify PLAN: remove Multi-Developer Parallelism section (dev assignments, staffing tables, file ownership, time chart). Keep task ordering dependency graph and conventions. PLAN tracks work to do, not assignments. Per user request. |
| 2026-03-12 | @claude | Phase 1.10 — address PR #68 R5: narrow 4-dev B1 scope to non-Track-A files only, update file ownership table with 1.10 task assignments per track. |
| 2026-03-12 | @claude | Phase 1.10 — address PR #68 R4: pick implementation direction for multi-option tasks (1.10b/h/i/l), split per-task deps to exact phase granularity (1.10a/b→1.3, 1.10c/d/e→2a.1, etc.), add 1.10 staffing to all dev assignment tables. |
| 2026-03-11 | @claude | Phase 1.10 — address PR #68 R3: fix changelog timeline (round 2 had open rename issue, full validation after round 3), replace stale line-number refs with grep-stable `@claude NOTE (PLAN 1.10x)` anchors, relax dependency from hard gate to parallel-with-2a.2. |
| 2026-03-11 | @claude | Added Phase 1.10 — API Gaps and Hardening: 14 `@claude NOTE` items from API.md, validated by @codex across PR #68 rounds 2–3. Runs parallel with Phase 2a (not a hard gate) to clean up ergonomic and correctness gaps. |
| 2026-03-10 | @claude | Phase 2a.1 implemented: SshTransport with russh 0.46, ssh-agent auth, DC2 host key verification (Verify/TrustFirstUse/Insecure), SshConfig builder, exec/open_shell over SSH, SshShellChannel with PTY, actionable error messages (OC3). |
| 2026-03-10 | @claude | Address PR #66 review round 2: scope 1.9b checkboxes to match implementation — reflow detection covers capture/sample_text only (exec handled by wrap-tolerant parser), overlap provides stateless primitives (history_size tracking and wider recapture deferred to 2a.2 monitor). |
| 2026-03-10 | @claude | Phase 1.9b implemented: geometry snapshots (list_clients, query_pane_geometry), reflow detection around captures, overlap-aware incremental sampling with OverlapResync fallback, history-limit setup helpers, comprehensive tests. |
| 2026-03-10 | @claude | Phase 1.9a implemented: fidelity types, capture normalization modes (Raw/ScreenStable/PlainText), options-based capture APIs, ANSI stripping, exec() wrap-tolerant sentinel via `-ep` capture. |
| 2026-03-10 | @codex | Address PR #65 review feedback: split Phase `1.9` into `1.9a`/`1.9b`, keep default capture wrappers `Raw`, make `ExecStable` internal-only, scope history-limit work to setup-time/new panes, and move sink backpressure signaling to `SinkEvent::Gap`. |

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
- [x] `TransportKind::open_shell(&self, cols, rows) -> Result<ShellChannelKind>` — dispatch to variant
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

- [x] `HostHandleInner` struct with `transport`, `socket`, `exec_locks` (Phase 1 scope; `config` and `session_monitors: RwLock<HashMap>` added in Phase 2a.4)
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
  poll `capture_with_history()`, extract stdout + exit code, per-pane exec lock map on
  `HostHandleInner` keyed by resolved `pane_id` (all target levels resolve via `display-message`),
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

### 1.9a — Capture Fidelity Types + Explicit Modes

- [x] Define fidelity/normalization types in `types.rs`:
  `CaptureNormalizeMode` (`Raw`, `ScreenStable`, `PlainText`) and
  `CaptureOptions` (`history_start`, `overlap_lines`, `detect_reflow`), plus
  `OutputFidelity` and `CaptureResult`
- [x] Make the hot-path clean case zero-allocation in docs/code shape:
  `OutputFidelity.issues: Option<Vec<FidelityIssue>>`
- [x] Add `capture_with_options`, `sample_text_with_options`, and bulk-capture
  result APIs returning `CaptureResult`
- [x] Keep existing `capture()` / `sample_text()` / `capture_all()` wrappers as
  explicit `Raw` convenience APIs
- [x] Implement `ScreenStable` normalization with ANSI/control preservation in the
  public payload: canonical line endings, width-artifact trimming only
- [x] Keep ANSI/control stripping as explicit opt-in only (`PlainText` mode) for
  human/LLM and matcher-oriented workflows
- [x] Document and implement the mode-to-field contract:
  `text` / `content` are mode-specific public payloads, `raw_text` /
  `raw_content` are exact-capture sidecars when requested
- [x] Update `Target::exec()` polling/parser to use an internal derived parser view
  for wrap-tolerant sentinel detection without exposing a public `ExecStable` mode
- [x] Add unit tests for:
  raw-vs-screen-stable-vs-plain-text mapping, wrapped sentinel splits, bulk capture
  options, and hot-path clean fidelity metadata
- [x] Treat `1.9a` as the hard usability gate before monitor/sink work:
  `2a.2` and `2c.*` consume `1.9a` metadata/types rather than inventing parallel contracts

**Depends on**: 1.5, 1.7

### 1.9b — Mixed-Client Stabilization

- [x] Add geometry snapshot helpers using tmux metadata:
  - `list-clients -F` for attached client sizes (`client_width`, `client_height`,
    `client_session`)
  - `display-message -p` / format vars for pane state (`pane_width`, `pane_height`,
    `history_size`, `history_limit`)
- [x] Add reflow detection around capture/sampling windows:
  compare pre/post geometry snapshots via `detect_fidelity()`/`finalize_fidelity()`;
  return degraded `OutputFidelity` when client mix/geometry changes during
  `capture_pane_with_options()` or `sample_text_with_options()`.
  <!-- @claude 2026-03-10: exec() reflow detection deferred to Phase 2a. exec() already
       handles wrapping via the two-tier parse_sentinel_output() parser (fast single-line
       path + joined-text fallback) with -ep capture + ANSI strip. Surfacing geometry
       fidelity on ExecOutput requires API changes and persistent state better suited to
       the monitor layer. See PR #66 review round 2. -->
- [x] Add overlap-aware incremental sampling primitives:
  `overlap_deduplicate()` performs byte-exact overlap suffix matching with `OverlapResync`
  fallback; `sample_text_with_options()` accepts caller-supplied `previous_text` for
  dedup when `overlap_lines >= 2`.
  <!-- @claude 2026-03-10: Per-target history_size tracking and automatic wider recapture
       on overlap ambiguity deferred to Phase 2a.2 (monitor stream assembly) where
       persistent per-pane state is available. Phase 1 provides stateless library
       primitives; callers that need stateful incremental sampling (monitors, sink
       pipelines) build on these primitives with their own state. See PR #66 review
       round 2. -->
- [x] Add setup-time history-limit helpers/docs for automation windows:
  `history-limit` must be set before pane creation to affect new panes; existing
  panes keep their creation-time limit
- [x] Add unit tests for:
  mixed-client resize events, history growth/shrink behavior, large history windows,
  overlap ambiguity/resync, and non-retroactive `history-limit` behavior
- [x] Add docs/tests clarifying hard limits:
  no recovery after history eviction; deterministic mode still requires
  dedicated/fixed-geometry sessions; `resize-window` is best-effort under mixed clients

**Depends on**: 1.9a

### 1.10 — API Gaps and Hardening

Addresses 14 inconsistencies and ergonomic gaps identified during API documentation
(PR #68) and validated by @codex across PR #68 rounds 2–3. Each item references
the corresponding `@claude NOTE (PLAN 1.10x)` marker in `docs/API.md` — grep for
the task ID to locate the note (line numbers drift across edits).

#### Transport layer (`src/transport.rs`)

- [x] **1.10a** — `MockTransport`: add `with_error(pattern, message)` to allow `exec()`
  to return `Err` for specified patterns. Currently `exec()` always returns `Ok(...)`,
  preventing unit-test coverage of error handling paths in discovery, capture, and control.
  *(grep: `PLAN 1.10a` in API.md, impact: medium)*

- [x] **1.10b** — `MockTransport`: switch pattern storage from `HashMap` to `Vec` for
  deterministic ordered matching. Currently overlapping substring patterns (e.g. `"list"`
  vs `"list-sessions"`) match nondeterministically due to HashMap iteration order. With
  `Vec`, patterns are checked in registration order — first match wins.
  *(grep: `PLAN 1.10b` in API.md, impact: medium for tests)*

- [x] **1.10c** — `SshTransport::open_shell()`: add PTY size parameters. Currently
  hardcodes `request_pty("xterm", 80, 24, ...)` with no API to specify dimensions.
  Add optional width/height to `open_shell()` or an `SshShellOptions` builder.
  *(grep: `PLAN 1.10c` in API.md, impact: medium for TUIs, low for non-interactive)*

- [x] **1.10d** — `SshTransport::is_closed()`: expose transport-agnostic health probe.
  `is_closed()` exists only on `SshTransport`; neither `HostHandle` nor `TransportKind`
  exposes it. Add `TransportKind::is_healthy() -> bool` (returns `true` for Local/Mock,
  delegates to `!is_closed()` for SSH).
  *(grep: `PLAN 1.10d` in API.md, impact: low-to-medium)*

- [x] **1.10e** — `TrustFirstUse` host key policy: improve discoverability of fail-closed
  behavior. Connection is rejected if `~/.ssh/known_hosts` is not writable. Add a doc
  comment on `HostKeyPolicy::TrustFirstUse` explaining the precondition and the error
  message users will see.
  *(grep: `PLAN 1.10e` in API.md, impact: medium, mostly ergonomic)*

#### Types (`src/types.rs`)

- [x] **1.10f** — `TargetSpec::pane()`: return `Result` instead of panicking. Currently
  uses `assert!` to enforce that `.window()` was called first — builder misuse is
  process-fatal for library consumers. Change to return `Result<TargetSpec>` or a
  typed error.
  *(grep: `PLAN 1.10f` in API.md, impact: low-to-medium ergonomically)*

#### Host handle and Target (`src/host.rs`)

- [x] **1.10g** — `HostHandle::local()`: add `local_with_timeout(Duration)` convenience
  constructor. Currently hardcodes 10s transport timeout with no builder/setter — callers
  must drop to `HostHandle::new()` + `LocalTransport::with_timeout()`.
  *(grep: `PLAN 1.10g` in API.md, impact: ergonomic)*

- [x] **1.10h** — `Target::rename()`: return a new `Target` with updated address.
  After session rename, `target_string()` returns the old name and subsequent operations
  fail. Change signature to `rename(&self, new_name) -> Result<Target>` so callers get a
  fresh handle. Window rename is metadata drift only (`target_string()` uses
  `session:index`) but should also return an updated `Target` for consistency.
  *(grep: `PLAN 1.10h` in API.md, impact: correctness-significant for session, metadata for window)*

- [x] **1.10i** — `Target::rename()` at pane level: add doc comment documenting that
  pane rename returns `Err("cannot rename a pane")` because tmux has no `rename-pane`
  command. A compile-time restriction would require breaking the unified `Target` API,
  so document the asymmetry instead.
  *(grep: `PLAN 1.10i` in API.md, impact: low)*

- [x] **1.10j** — `Target::pane(index)` from session level: document active-window drift.
  Resolves via the **active window** at call time, so the same call can return different
  panes if focus changes. Add a doc comment recommending explicit navigation
  (`target.window(0).pane(0)`) for deterministic targeting.
  *(grep: `PLAN 1.10j` in API.md, impact: medium for automation)*

- [x] **1.10k** — `Target::capture()` at session/window level: clarify scope in doc
  comment. Captures the **active pane** only (consistent with tmux `capture-pane -t`),
  but the method name sounds exhaustive. Add doc comment noting this and pointing to
  `capture_all()` for all panes.
  *(grep: `PLAN 1.10k` in API.md, impact: medium)*

#### Capture (`src/capture.rs`)

- [x] **1.10l** — `overlap_deduplicate()`: log at `warn` level when `overlap_lines < 2`.
  Currently silently no-ops (returns current capture unchanged with no fidelity issues).
  A `tracing::warn!` is non-breaking and makes the threshold visible without changing
  the return type. Also add the `>= 2` requirement to the function's doc comment.
  *(grep: `PLAN 1.10l` in API.md, impact: low-to-medium)*

#### Cross-cutting documentation

- [x] **1.10m** — Document two independent timeout knobs. `Target::exec()` has its own
  sentinel polling timeout, separate from the transport timeout
  (`LocalTransport::timeout` / `SshConfig::timeout`) that governs each individual tmux
  command. Add doc comments on both `Target::exec()` and `SshConfig::with_timeout()`
  explaining the interaction and failure modes.
  *(grep: `PLAN 1.10m` in API.md, impact: medium)*

- [x] **1.10n** — Document `history-limit` creation-time semantics. `set_history_limit()`
  only affects panes created after the call; existing panes retain their creation-time
  limit. Add doc comments on `set_history_limit()` and `set_global_history_limit()`
  noting this tmux behavior.
  *(grep: `PLAN 1.10n` in API.md, impact: medium ergonomically)*

**Depends on**: Per-task, by earliest availability:
- 1.10a, 1.10b: `MockTransport` in `transport.rs` → available after **1.3**
- 1.10c: `SshTransport::open_shell()` in `transport.rs` → available after **2a.1**
- 1.10d: `TransportKind` in `transport.rs` + `SshTransport` → available after **2a.1**
- 1.10e: `HostKeyPolicy` doc comment in `types.rs` → available after **2a.1** (needs SSH context)
- 1.10f: `TargetSpec` in `types.rs` → available after **1.1**
- 1.10g: `HostHandle` in `host.rs` → available after **1.7**
- 1.10h: `Target::rename()` in `host.rs` + `control.rs` → available after **1.7**
- 1.10i, 1.10j, 1.10k: doc comments in `host.rs` → available after **1.7**
- 1.10l: `overlap_deduplicate()` in `capture.rs` → available after **1.5**
- 1.10m, 1.10n: doc comments only → no code dependency

**Gates**: Phase 2a.2 and later. All 1.10 tasks should be completed before
starting the next phase. Tasks within 1.10 can be worked in any order based
on the per-task availability above (e.g. 1.10a/b can start immediately after
1.3; 1.10c/d/e must wait for 2a.1).

### 1.11 — Unified SSH URI for `SshConfig` (`src/uri.rs`) — DC21

Extends the existing `SshConfig` type with URI parsing, rendering, and `connect()`.
No new type — `SshConfig` becomes the single entry point for host configuration.
URI logic lives in `src/uri.rs` as an `impl SshConfig` extension block.
See DESIGN.md DC21 for full specification.

#### `SshConfig` field changes (`src/transport.rs`)

- [ ] **1.11a** — Add `socket: Option<TmuxSocket>` field to `SshConfig` with
  `with_socket()` builder method. Default `None`.

- [ ] **1.11b** — Make `SshConfig` fields private, add accessor methods:
  `host()`, `user()`, `port()`, `is_localhost()`, `socket()`. Update all internal
  field access in `SshTransport`, `SshHandler` to use accessors.

#### URI parsing (`src/uri.rs` — `impl SshConfig` extension)

- [ ] **1.11c** — Create `src/uri.rs` with `SshConfig::parse(uri: &str) -> Result<Self>`:
  parse `ssh://` scheme, extract host/port from authority, user from userinfo.
  Support **both** nassh-style (`;` params in userinfo) and query-param (`?key=value&...`)
  syntax. Handle IPv6 bracket notation `[::1]`.

- [ ] **1.11d** — Canonical-component duplication rejection: `user`, `host`, `port`
  parsed exclusively from their URI positions. Reject these as `;` or `?` parameter
  names. Reject duplicate keys across userinfo and query string.

- [ ] **1.11e** — Parameter parsing and validation: `host-key-policy` → `HostKeyPolicy`,
  `timeout` → seconds → `Duration`, `keepalive` → seconds (0=off) → `Option<Duration>`,
  `socket-name` → `TmuxSocket::Name(...)`. Reject unknown parameter names (fail-fast).
  Validate ranges (timeout > 0).

- [ ] **1.11f** — `SshConfig::to_uri_string(&self) -> String`: render to canonical
  `ssh://` form (nassh-style params in userinfo). `Display` impl delegates to this.
  `FromStr` impl delegates to `parse()`. Round-trip: `parse(cfg.to_string()) == cfg`.

#### Transport selection and connect (`src/uri.rs`)

- [ ] **1.11g** — `SshConfig::connect(self) -> Result<HostHandle>`: takes ownership.
  localhost (`localhost` / `127.0.0.1` / `::1`) → `LocalTransport::with_timeout()`;
  all others → `SshTransport::connect(self)`. Extract `socket` before move, wire to
  `HostHandle::new()`. Require non-empty `user` for SSH hosts.

- [ ] **1.11h** — `SshConfig::is_localhost(&self) -> bool` helper used by `connect()`.

#### Integration

- [ ] **1.11i** — `src/lib.rs`: add `mod uri;` (private module — extends `SshConfig`
  which is already re-exported via `pub use transport::SshConfig`).

#### Tests

- [ ] **1.11j** — Unit tests for `parse()`: valid URIs with nassh params, query params,
  mixed params, socket-path. Invalid URIs: bad scheme, unknown params, malformed
  userinfo, missing host, canonical-component duplication (`?port=22`), duplicate
  keys across locations, `/socket-path` + `socket-name` mutual exclusion.
  Edge cases: IPv6, no user, port-only, empty params.

- [ ] **1.11k** — Unit tests for `to_uri_string()` and round-trip: builder-constructed
  configs render to valid URIs; parse ∘ to_string is identity for canonical forms.

- [ ] **1.11l** — Unit tests for `connect()` transport selection via
  `HostHandle::transport_kind()` accessor (added in DC21 implementation plan,
  point 4): localhost variants (`localhost`, `127.0.0.1`, `::1`) produce
  `TransportKind::Local`, other hosts produce `TransportKind::Ssh`. Verify
  config fields propagate correctly. Verify empty user rejected for SSH hosts.
  <!-- @claude 2026-03-12: rewritten per PR #71 review — original referenced
       unobservable TransportKind variant; now uses transport_kind() seam. -->

- [ ] **1.11m** — Integration test: `SshConfig::parse("ssh://localhost")?.connect()`
  produces a working `HostHandle` that can `list_sessions()`.

#### Documentation

- [ ] **1.11n** — Update `docs/API.md` with `SshConfig` URI section: parse/builder
  examples, both param syntaxes, canonical-component rules, parameter table,
  transport selection, usage patterns.

**Depends on**: Phase 2a.1 (SshTransport must exist for non-localhost connect).
Tasks 1.11a–b (field changes) can start immediately. Tasks 1.11c–f and 1.11j–k
(parse/render/tests) can proceed after 1.11a–b since they only depend on types.
Tasks 1.11g (connect) and 1.11l (connect tests) require SshTransport from 2a.1.

**Gates**: None — this phase is additive. Can run parallel with other work after
2a.1 is complete.

---

## Phase 2a: SSH Transport + Minimal Monitoring

Add `SshTransport` and a thin monitoring vertical slice with control mode parsing.

### 2a.1 — SSH transport (`src/transport.rs` extension)

- [x] Add `russh` + `russh-keys` dependencies to `Cargo.toml`
- [x] `SshTransport` struct: russh `Handle`, configurable timeouts
- [x] `SshTransport::connect(host, user, config) -> Result<Self>` — SSH connect + auth via ssh-agent
- [x] Host key verification (DC2): implement `HostKeyPolicy` from config —
  `Verify` (parse `~/.ssh/known_hosts`, reject unknown), `TrustFirstUse`
  (accept + persist on first connect), `Insecure` (accept all, log warning)
- [x] `SshTransport::exec()` — open exec channel, capture stdout, close channel
- [x] `SshTransport::open_shell()` — PTY channel with shell, piped I/O
- [x] Add `Ssh(SshTransport)` variant to `TransportKind` and `ShellChannelKind`
- [x] Actionable error messages for SSH agent failures (OC3)
- [x] Unit tests: `SshConfig` builder/defaults
  <!-- @claude 2026-03-10: Mock-based tests for TransportKind::Ssh dispatch, host-key
       policy behavior, and agent failure paths deferred — russh Handler/Handle types
       are not easily mockable without a real SSH handshake. Integration testing against
       a real SSH server (or Docker-based in 4.3) will cover these paths. See PR #67
       review. -->

**Depends on**: 1.3

### 2a.2 — Control mode parser (`src/monitor.rs`)

- [ ] `SessionMonitor` struct: session name, rules, cooldown state
- [ ] Control mode stream parser: parse `%output %<pane_id> <data>` frames from
  `tmux -C attach -t <session>` output
- [ ] Per-pane stream assembly state keyed by `pane_id`:
  partial-frame buffering, newline canonicalization, monotonic per-pane `sequence`
- [ ] Reuse `1.9a` normalization/fidelity path for monitor events:
  preserve ANSI in fidelity modes, apply the same mode-to-field contract,
  annotate degraded/reflow/history conditions in emitted metadata
- [ ] Handle other control mode messages gracefully (`%begin`, `%end`, `%error`, etc.)
- [ ] Rule evaluation against parsed output (initially: single compiled rule)
- [ ] Action dispatch: `SendKeys` via bounded `mpsc` channel + semaphore (DC4)
- [ ] `SessionMonitor::run()` — main loop: read from shell, parse, evaluate, dispatch
- [ ] Stop signal via `watch::Receiver<bool>`, clean shutdown on signal or connection drop
- [ ] Warn-level logging for malformed lines and failed actions (P9)
- [ ] Unit tests: control mode frame parsing, chunk split/reassembly, sequence monotonicity,
  rule matching, dispatch ordering

**Depends on**: 1.7, 1.9a

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

- [ ] `TargetOutput` struct: `source: TargetAddress`, `host`, canonical `content`,
  optional `raw_content`, `sequence`, `fidelity`, `timestamp`
- [ ] `SinkEvent` enum: `Data(TargetOutput)` and `Gap { dropped, timestamp }`
- [ ] `TargetOutput` accessors: `session_name()`, `pane_id()`, `target_string()`
- [ ] `OutputFidelity` / `FidelityIssue` enums shared with capture/monitor paths
- [ ] `MatcherInput` enum: `Preserve` (match delivered `content`),
  `PlainTextDerived` (derive sink-local plain-text view from `content`)
- [ ] `SinkFilter`: `host`, `session`, `window`, `pane` (all optional regex strings),
  `content: Option<MatcherKind>`, `matcher_input: MatcherInput`
- [ ] Define content-matching contract: `SinkFilter.content` matches either
  delivered `TargetOutput.content` or a sink-local plain-text derived view,
  selected by `matcher_input`
- [ ] `CompiledSinkFilter`: compiled regexes + `MatcherKind`,
  `matches(&mut self, output: &TargetOutput) -> bool`
- [ ] `SinkAction` enum: `SendKeys`, `SendText`, `KillSession`, `RenameSession`
- [ ] `ActionRequest` struct: `host`, `target: TargetAddress`, `action: SinkAction`
- [ ] `ActionHandle`: wraps `mpsc::Sender<ActionRequest>`,
  provides `send()`, `send_keys()`, `send_text()`, `kill_session()`,
  `rename_session()`, `respond(output, action)`
- [ ] `SinkId` opaque type

**Depends on**: 2b.2, 1.9a

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
- [ ] `publish(output: TargetOutput)` — fan out to all matching sinks via `try_send`
  while tracking per-sink dropped counts
- [ ] No-silent-drop contract: if drops occurred, emit `SinkEvent::Gap { dropped }`
  before the next `SinkEvent::Data(output)` on that sink route
- [ ] `shutdown() -> Result<()>` — signal all sinks, flush, join all tasks
- [ ] `SinkEntry` internal: id, name, tx, compiled filters, task handle
- [ ] Unit tests: fan-out to 3 sinks, slow sink doesn't block others,
  filter matching (AND within / OR across), gap-event emission after drops, shutdown flushes

**Depends on**: 2c.2

### 2c.4 — Pipeline integration

- [ ] Wire `OutputBus` into `monitor.rs`: publish fidelity-aware `TargetOutput`
  alongside rule evaluation
- [ ] Wire `ActionHandle` into `HostHandle`: sink-initiated actions route through DC4 queue
- [ ] `Fleet::output_bus()` accessor; sinks registered before `start_monitoring()`
- [ ] Integration test: monitor publishes output → `StdioSink` receives and formats,
  `CallbackSink` receives and accumulates, `ActionHandle` routes action back to target,
  and gap/degraded metadata is preserved end-to-end

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
 │     ├── 1.9a Capture fidelity types [needs 1.5 + 1.7]
 │     │
 │     ├── 1.9b Mixed-client stabilization [needs 1.9a]
 │     │
 │     ├── 1.10 API gaps + hardening [gates 2a.2; per-task deps within]
 │     │
 │     ├── 1.11 SshConfig URI ext (src/uri.rs) [1.11a-f after 1.1; 1.11g after 2a.1]
 │     │
 │     └── 2a.2 Monitor parser [needs 1.7 + 1.9a]
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

Notes:
- `1.9a` is the type/API gate required before `2a.2` and `2c.*`.
- `1.9b` is a follow-on stabilization pass for overlap resync, geometry detection,
  and history setup guidance.
- `1.10` gates Phase 2a.2 — all tasks must complete before next phase. Per-task
  dependencies (listed in section above) allow internal parallelism within 1.10.
- `1.11` (SshConfig URI) — field changes (1.11a–b) start immediately; parse/render
  (1.11c–f, 1.11j–k) after 1.11a–b; connect (1.11g, 1.11l) requires 2a.1. No gates.
- `2b.3` and `2c.4` both modify `monitor.rs` and `host.rs` — must land serially.
- Phase 4 tasks are independent and can start as soon as their prerequisites are met.

## Conventions

- Each task should be a PR-sized unit (reviewable in isolation)
- Every task ends with `cargo check` and `cargo test` passing
- Use `MockTransport` for unit tests; real tmux for integration tests
- Integration tests gated on `which tmux` availability check
- Follow workspace patterns: `anyhow` for errors, `tracing` for logs, `serde` for config
