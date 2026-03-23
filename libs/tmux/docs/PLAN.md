# Tmux Multi-Target Automator — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-22 | @claude | Implement Phase 5.1 and 5.2: split-screen TUI REPL mode (`tui on`/`tui off`) with binary-local `tui_mirror` consumer using `HistoryHandle` for bounded mirror frame. Restructured `examples/repl.rs` → `examples/repl/main.rs` + `examples/repl/tui_mirror.rs`. Added `ratatui`/`crossterm` dev-dependencies. |
| 2026-03-21 | @claude | Implement 4.2g (DC30 socket isolation) and 4.2h (DC31 tracked execution). DC30: `TmuxSocket::automation()`, `SshConfig::with_automation_socket()`, `HostHandle::ensure_socket_server()`. DC31: `ExecId`/`ExecState`/`ExecHandle` types, `Target::start_exec()`, `exec()` refactored onto tracked substrate, `active_execs` discontinuity wiring. 378 unit tests (+11 new), 2 new integration tests. |
| 2026-03-21 | @codex | Address PR #96 review feedback — clarify 4.2h wording from "blocking" to "await-to-completion", require slice-3 test parity with the existing `Target::exec()` sentinel/scrollback/lock behavior before refactoring it onto tracked execution, and align the product-driven notes with shipped SSH identity-file support. |
| 2026-03-21 | @codex | Execution-planning follow-up for 4.2g/4.2h: document impact radius by module and require 4.2h to land in staged slices (`ExecId`/state, `start_exec`, `exec()` layering, discontinuity semantics/tests) so tracked execution remains reviewable. |
| 2026-03-21 | @codex | Product-driven robustness follow-up from [`docs/PRODUCT.md`](../../../docs/PRODUCT.md): add concrete plan items for 4.2g dedicated socket-isolation ergonomics and 4.2h tracked command execution. Prioritize socket isolation first as the higher-value robustness feature; keep tracked execution as a complementary pane-scoped execution primitive. |
| 2026-03-21 | @claude | 4.2c enhancement: full capture-pane snapshot on reconnect — after "stream resumed" discontinuity, list panes in session and publish each pane's visible content as `TargetOutput` so downstream consumers get real screen state re-anchoring (not just a marker). |
| 2026-03-20 | @claude | Implement Phase 4.2 (DC29 streaming resilience): 4.2a reconnect supervision with exponential backoff in host.rs, 4.2b `SinkEvent::Discontinuity` threaded through all adapters, 4.2c fresh snapshot anchoring (discontinuity markers on reconnect), 4.2d per-session `MonitorHealth` as Fleet ground truth, 4.2e stress tests (bus throughput, history determinism, MockTransport multi-phase), 4.2f adversarial shell-escape property tests. 355 tests (+14 new). |
| 2026-03-20 | @codex | Refine Phase 4.2 per PR #94 review: reframe reconnect resync as fresh snapshot anchoring (not replay), add adapter-propagation tasks for discontinuity, specify missing-session/topology-change handling, and require per-session monitor health as Fleet ground truth. |
| 2026-03-20 | @codex | Expand Phase 4 into explicit streaming-resilience work: reconnect supervision, upstream discontinuity modeling, fresh snapshot anchoring after reconnect, Fleet health state, and failure-injection coverage. This becomes the next active hardening priority for long-lived external-agent workflows. |
| 2026-03-20 | @claude | Implement Track B (2b.1, 2b.2, 2b.3): `HistoryHandle`/`HistoryOptions`/`HistorySnapshot`/`HistoryEntry` in sink.rs, `Subscription::filter_fn()` predicate adapter, `Fleet` module with host registry/workstream routing/monitoring lifecycle. 339 tests pass (50 new). |
| 2026-03-20 | @codex | DC28 follow-up — specify 2b.1 transcript/history as a bounded rolling snapshot layer built on `JoinedStream`, optimized for external LLM/classifier context windows. Add concrete `HistoryHandle`/`HistorySnapshot`/`HistoryEntry` direction and explicit trimming semantics. |
| 2026-03-20 | @codex | Directional simplification — active post-Track-A plan now prioritizes transcript/history adapters, simplified Fleet coordination, and external-agent workflows. Built-in matcher/rule/reactor/config direction moved to historical context section. |
| 2026-03-20 | @claude | Phase 1.15 implemented (1.15a–g): SshConfig `identity_file` field + fallible builder, `authenticate_key_file()` auth dispatch, query-only URI parse/render with `QUERY_ONLY_PARAMS`, 20 new unit tests (318 total), 1 integration test (23 total), API.md/README.md/repl.rs docs updated. SSH key-file integration tests deferred to Phase 4.3 CI infra. |
| 2026-03-20 | @claude | Phase 1.15 R1 — address PR #89 review: `identity-file` is query-only (not nassh userinfo), `with_identity_file()` is fallible (returns `Err` on duplicate), add query-only parse rejection tests, duplicate-source builder tests, mixed-param rendering tests. |
| 2026-03-20 | @claude | Add Phase 1.15 — SSH identity-file authentication (DC26). Extends SshConfig/URI with `identity-file` parameter, adds `authenticate_key_file()` auth path. 7 tasks (1.15a–g). Slotted after 1.14, depends on 1.11 (URI infra) and 2a.1 (SSH transport). |
| 2026-03-19 | @codex | Phase 1.14 correction — `SplitSize::Percent` maps to `tmux split-window -l <n>%` in tmux 3.4, not `-p`. Keep `--percent` as the public API/example surface but emit `-l 40%` in the control layer. |
| 2026-03-19 | @codex | Implement Phase 1.14 target-creation symmetry: add typed window/pane creation APIs, localhost coverage, and example/API updates including REPL commands. |
| 2026-03-19 | @codex | Add Phase 1.14 — hierarchy creation symmetry on `Target`: first-class `new_window()` / `split_pane()` with typed option structs, deterministic typed returns, tests, and API/example updates. Slotted as the next API-generalization task after 1.13. |
| 2026-03-19 | @codex | Make `HostHandle::transport_kind()` a test-only `#[cfg(test)]` seam. It is only used by DC21 localhost transport-selection unit tests; scoping it to tests removes the standing dead-code warning without changing behavior. |
| 2026-03-18 | @claude | PR #83 R3 — reconcile docs with implementation: OutputBus sync `&self` signatures, `PipeHandle` replaces bare `JoinHandle` in DESIGN/PLAN, `source_key()` added to TargetOutput in DESIGN, `SourceLabel` pane_id format (`build(%5)`), `format()` always labels, `StdioSink` Prefixed uses `source_key`, API.md format examples updated. |
| 2026-03-18 | @claude | PR #83 R2 design/API improvements: (1) fix start-monitoring error-path — registration after session resolution, (2) split `source_key()` (identity) from `target_string()` (display), (3) `SinkFilter::for_session/for_pane/for_host/for_host_session` exact-match constructors, (4) `PipeHandle` wraps subscription id + task JoinHandle, (5) format() layering documented (JoinedStream = consumer, StdioSink = terminal), (6) stop vs shutdown lifecycle distinction in all docs. 280 tests (+6 new). |
| 2026-03-18 | @claude | PR #83 R1 fixes: (1) UTF-8 octal decoder rewritten to collect bytes before decoding — multi-byte chars now correct, (2) pane identity uses pane_id as canonical identity for source comparison, display, and filter matching — distinct panes no longer merge, (3) OutputBus lifecycle docs narrowed to match fire-and-forget implementation. 274 tests (+6 new). |
| 2026-03-18 | @claude | Track A deferred items completed: (1) fidelity normalization in monitor via CaptureNormalizeMode, (2) MonitorHandle.stop_session()/get_by_spec(), (3) HostHandle.stop_monitoring_session()/stop_monitoring()/monitored_sessions() with per-host signal tracking, (4) Target::start_monitoring()/stop_monitoring() session-level convenience, (5) live localhost integration test (localhost_monitor_pipeline). 268 lib + 19 integration tests. |
| 2026-03-18 | @claude | Track A implemented (2a.2a, 2c.1a, 2c.2a, 2c.3, 2a.4a, 2c.4a): control mode parser, sink types, SinkKind/Subscription/JoinedStream, OutputBus, monitor handle wiring, pipeline integration. 262 tests. Deferred items noted inline: fidelity normalization in monitor, Target-level monitoring methods, live tmux integration test. |
| 2026-03-17 | @claude | DC24 R2 — Address PR #82 review round 2: add Track B static-dispatch guardrail note (closed `MatcherKind` enum, no `Box<dyn Matcher>`). |
| 2026-03-17 | @claude | DC24 R1 — Address PR #82 review: unify subscription API (`subscribe() -> Subscription` with adapters). Rules/reactors as subscription consumers, not monitor-internal (removes 2a.2b). Replace 2c.1b with 2b.4 (subscription matching + reaction adapters) and 2c.4b with 2b.5 (action handle wiring). One matching system via `.filter()`/`.react()`. Track B now: 2b.2 → 2b.1 → 2b.4 → 2b.5 → 2b.3. |
| 2026-03-17 | @claude | DC24 — Restructure Phase 2 into Track A (streaming + combining) and Track B (matching + reactors). Split 2a.2 → 2a.2a (parse only) + 2a.2b (rules + dispatch). Split 2a.4 → 2a.4a (streaming handles). Split 2c.1 → 2c.1a (routing only) + 2c.1b (content matching + actions). Split 2c.2 → 2c.2a (sinks + JoinedStream combinator, no JoinedSink). Split 2c.4 → 2c.4a (streaming) + 2c.4b (action wiring). Remove `subscribe_joined` from OutputBus. Update task ordering diagram and linear execution sequence. |
| 2026-03-17 | @claude | Address PR #80 R2 — remove `2a.3 Pipes` from task ordering diagram (already descoped in checklist). |
| 2026-03-16 | @claude | Remove Phase 2a.3 (pipe-pane fallback) — out of scope. tmux 3.1+ baseline (established in DC22) guarantees control mode availability; pipe-pane fallback is dead weight. Removed fallback references from 2a.4, 4.1, 4.4. Updated DESIGN.md DC10 with out-of-scope note. |
| 2026-03-15 | @claude | Phase 1.13 complete (1.13a–i): API.md section 16 (file transfer + exec boundary table), `upload`/`download` REPL commands with `--recursive`, cross-link to SFTP.md. |
| 2026-03-15 | @claude | Phase 1.13 implementation complete (1.13a–h): SSH SFTP via `russh-sftp`, all unit + integration tests passing (223 unit, 17 integration). Added `HostHandle::exec()` public API for ad-hoc shell commands. Remaining: 1.13i (docs/examples). |
| 2026-03-15 | @claude | Phase 1.13 implementation: completed 1.13a–d, 1.13f–h (types, transport surface, local/mock impl, host wiring, unit + integration tests). SSH SFTP impl (1.13e) and SSH integration tests stubbed for follow-up. |
| 2026-03-15 | @codex | Address PR #78 final doc follow-up: lock symlink rejection, metadata non-preservation, and `Result<()>` return shape into Phase 1.13 tasks and test coverage. |
| 2026-03-14 | @codex | Address PR #78 re-review: make remote path parameters `&Path`, define `cp -r` style directory placement semantics, and add explicit copy-into vs copy-as test tasks. |
| 2026-03-14 | @codex | Address PR #78 review: localhost SFTP integration tests run unconditionally (no tmux gate), directory overwrite semantics are explicit merge semantics, and Phase 1.13 is split into smaller incremental tasks. |
| 2026-03-14 | @codex | Refined Phase 1.13 per user decisions: greenfield/breaking changes accepted, API uses `upload` / `download`, overwrite semantics configurable, directory transfer included now, file-only/v1 phasing removed. |
| 2026-03-14 | @codex | Added Phase 1.13 — host-level SFTP file transfer. Slots after 1.12 as an additive transport/host feature depending on 1.3, 1.7, and 2a.1. No hard gate for monitoring phases. |
| 2026-03-14 | @claude | Phase 1.12 — `CreateSessionOptions` for window size and history limit (DC22). 7 tasks (1.12a–g). Migration/backwards compatibility explicitly out of scope per user direction. |
| 2026-03-13 | @claude | Phase 1.11 — address PR #71 R4: scope `transport_kind()` to `pub(crate)` in 1.11l, add within-location duplicate rejection to 1.11d/1.11j, split 1.11m (localhost-only integration) from new 1.11o (SSH integration with env requirements). |
| 2026-03-13 | @claude | Phase 1.11 — address PR #71 R3: scope 1.11l unit tests to localhost + error paths only, SSH transport verification in 1.11m integration. |
| 2026-03-12 | @claude | Phase 1.11 — address PR #71 R2: fix `connect(self)` in 1.11g, add socket mutual-exclusion to 1.11j test cases, fix fleet example. R1: 1.11l inspection seam. |
| 2026-03-12 | @claude | Phase 1.11 R2: address feedback — consolidate `SshUri` into `SshConfig` (no new type), support both nassh `;` and query `?` param syntax, no canonical-component duplication. 14 tasks (1.11a–n). |
| 2026-03-12 | @claude | Phase 1.11 implemented: all 15 tasks (1.11a–o) completed. DC21 — Unified SSH URI for SshConfig. New `src/uri.rs` with `parse()`/`to_uri_string()`/`connect()`/`Display`/`FromStr`. SshConfig fields privatized with accessors, `socket` field added, `transport_kind()` pub(crate) accessor on HostHandle. 45+ unit tests, 2 integration tests (localhost + env-gated SSH). API.md updated. |
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

Derived from [DESIGN.md](./DESIGN.md). Product prioritization context for competitive
positioning and robustness tradeoffs lives in [`docs/PRODUCT.md`](../../../docs/PRODUCT.md).
Each task is scoped to produce a compilable, testable increment. Dependencies are explicit.
File paths are relative to `libs/tmux/`.

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

- [x] **1.11a** — Add `socket: Option<TmuxSocket>` field to `SshConfig` with
  `with_socket()` builder method. Default `None`.

- [x] **1.11b** — Make `SshConfig` fields private, add accessor methods:
  `host()`, `user()`, `port()`, `host_key_policy()`, `timeout()`,
  `keepalive_interval()`, `socket()`, `is_localhost()`. Update all internal
  field access in `SshTransport`, `SshHandler` (same module — direct access ok).

#### URI parsing (`src/uri.rs` — `impl SshConfig` extension)

- [x] **1.11c** — Create `src/uri.rs` with `SshConfig::parse(uri: &str) -> Result<Self>`:
  parse `ssh://` scheme, extract host/port from authority, user from userinfo.
  Support **both** nassh-style (`;` params in userinfo) and query-param (`?key=value&...`)
  syntax. Handle IPv6 bracket notation `[::1]`.

- [x] **1.11d** — Canonical-component duplication rejection: `user`, `host`, `port`
  parsed exclusively from their URI positions. Reject these as `;` or `?` parameter
  names. Reject duplicate keys across userinfo and query string. Reject repeated
  keys within a single location (e.g. `;timeout=10;timeout=20`).
  <!-- @claude 2026-03-13: within-location duplicates added per PR #71 R4. -->

- [x] **1.11e** — Parameter parsing and validation: `host-key-policy` → `HostKeyPolicy`,
  `timeout` → seconds → `Duration`, `keepalive` → seconds (0=off) → `Option<Duration>`,
  `socket-name` → `TmuxSocket::Name(...)`. Reject unknown parameter names (fail-fast).
  Validate ranges (timeout > 0).

- [x] **1.11f** — `SshConfig::to_uri_string(&self) -> String`: render to canonical
  `ssh://` form (nassh-style params in userinfo when user non-empty, query params
  when user empty). `Display` impl delegates to this.
  `FromStr` impl delegates to `parse()`. Round-trip: `parse(cfg.to_string()) == cfg`.

#### Transport selection and connect (`src/uri.rs`)

- [x] **1.11g** — `SshConfig::connect(self) -> Result<HostHandle>`: takes ownership.
  localhost (`localhost` / `127.0.0.1` / `::1`) → `LocalTransport::with_timeout()`;
  all others → `SshTransport::connect(self)`. Extract `socket` before move, wire to
  `HostHandle::new()`. Require non-empty `user` for SSH hosts.

- [x] **1.11h** — `SshConfig::is_localhost(&self) -> bool` helper used by `connect()`.

#### Integration

- [x] **1.11i** — `src/lib.rs`: add `mod uri;` (private module — extends `SshConfig`
  which is already re-exported via `pub use transport::SshConfig`).

#### Tests

- [x] **1.11j** — Unit tests for `parse()`: valid URIs with nassh params, query params,
  mixed params, socket-path. Invalid URIs: bad scheme, unknown params, malformed
  userinfo, missing host, canonical-component duplication (`?port=22`), duplicate
  keys across locations, duplicate keys within a single location
  (`;timeout=10;timeout=20`, `?timeout=10&timeout=20`), `/socket-path` +
  `socket-name` mutual exclusion. Edge cases: IPv6, no user, port-only, empty params.
  <!-- @claude 2026-03-13: within-location duplicate tests added per PR #71 R4. -->

- [x] **1.11k** — Unit tests for `to_uri_string()` and round-trip: builder-constructed
  configs render to valid URIs; parse ∘ to_string is identity for canonical forms.

- [x] **1.11l** — Unit tests for `connect()` localhost selection via
  test-only `HostHandle::transport_kind()` (`pub(crate)` + `#[cfg(test)]`;
  not public API).
  Localhost variants (`localhost`, `127.0.0.1`, `::1`) produce
  `TransportKind::Local`. Verify empty user rejected for SSH hosts (error
  path — no handshake needed). Config field propagation for localhost
  (timeout, socket).
  <!-- @claude 2026-03-13: scoped to localhost + error paths per PR #71 R3 —
       SSH branch requires real handshake, not unit-testable.
       @claude 2026-03-13: transport_kind() scoped to pub(crate) per PR #71 R4. -->

- [x] **1.11m** — Integration test (localhost only):
  `SshConfig::parse("ssh://localhost")?.connect()` produces a working `HostHandle`
  that can `list_sessions()`. Verify config field propagation (timeout, socket)
  through to the connected handle. No network or SSH server required.
  <!-- @claude 2026-03-13: scoped to localhost per PR #71 R4 — SSH split to 1.11o. -->

#### SSH integration (environment-gated)

- [x] **1.11o** — Integration test (SSH transport selection): connect to a real SSH
  host via `SshConfig::parse("ssh://user@host")?.connect()`, verify `list_sessions()`
  succeeds. Gated by env var `MOTLIE_SSH_TEST_HOST` (format: `user@host[:port]`).
  Skipped in CI when the var is unset. Prerequisites: reachable SSH server with
  key-based auth via ssh-agent. Pass criteria: `connect()` returns a `HostHandle`,
  `list_sessions()` returns `Ok(...)`.
  <!-- @claude 2026-03-13: split from 1.11m per PR #71 R4 — reviewer flagged SSH
       path needs explicit environment requirements and pass criteria. -->

#### Documentation

- [x] **1.11n** — Update `docs/API.md` with `SshConfig` URI section: parse/builder
  examples, both param syntaxes, canonical-component rules, parameter table,
  transport selection, usage patterns.

**Depends on**: Phase 2a.1 (SshTransport must exist for non-localhost connect).
Tasks 1.11a–b (field changes) can start immediately. Tasks 1.11c–f and 1.11j–k
(parse/render/tests) can proceed after 1.11a–b since they only depend on types.
Tasks 1.11g (connect) and 1.11l (connect tests) require SshTransport from 2a.1.
Task 1.11o (SSH integration) requires a real SSH server and is env-gated.

**Gates**: None — this phase is additive. Can run parallel with other work after
2a.1 is complete.

### 1.12 — Session Creation Options — DC22

Add `CreateSessionOptions` to `create_session()` for window size and history limit.
See DESIGN.md DC22.

#### Types (`src/types.rs`)

- [x] **1.12a** — Add `CreateSessionOptions` struct with `window_name`, `command`,
  `width`, `height`, `history_limit` fields (all `Option`). Derive `Default`.

#### Control (`src/control.rs`)

- [x] **1.12b** — Update `control::create_session()` to accept `&CreateSessionOptions`.
  Append `-x W -y H` to `new-session` command when `width`/`height` are set.
  Append `-n <name>` when `window_name` is set. Append `<command>` when set.

- [x] **1.12c** — When `history_limit` is set, issue two additional commands after
  `new-session`:
  `set-option -t <name> history-limit <N>` (per-session, covers future panes) and
  `set-option -p -t <name> history-limit <N>` (per-pane, tmux 3.1+, covers initial pane).

#### Host handle (`src/host.rs`)

- [x] **1.12d** — Update `HostHandle::create_session()` signature from
  `(name, window_name, command)` to `(name, &CreateSessionOptions)`. Wire through
  to `control::create_session()`.

#### Tests

- [x] **1.12e** — Unit tests: `CreateSessionOptions::default()` produces same command
  as current behavior (no `-x`, `-y`, no `set-option`). Options with `width`/`height`
  produce `-x W -y H`. Options with `history_limit` produce both `set-option` commands.

- [x] **1.12f** — Update existing callers (`session_lifecycle.rs`, `send_and_capture.rs`,
  `repl.rs`, unit/integration tests) to use `CreateSessionOptions::default()`.

#### Documentation

- [x] **1.12g** — Update `docs/API.md` with `CreateSessionOptions` usage examples.

**Depends on**: 1.7 (HostHandle and Target must exist).

**Gates**: None — additive change. Can run parallel with other work.

---

## Phase 1.13: Host-Level SFTP File Transfer

Add transport/host-level SFTP-backed upload/download to complement transport
`exec()`. This is greenfield work: breaking changes are acceptable and no migration
or compatibility layer is required. Scope includes files and directories; `Target`
is not extended.

### 1.13a — Transfer options type (`src/types.rs`)

- [x] Add public `TransferOptions { overwrite: bool, recursive: bool }` with `Default`
- [x] Document in rustdoc that directory overwrite with `overwrite=true` uses merge semantics

### 1.13b — Transport surface (`src/transport.rs`)

- [x] Add `TransportKind::upload(&self, local_path: &Path, remote_path: &Path, opts) -> Result<()>`
- [x] Add `TransportKind::download(&self, remote_path: &Path, local_path: &Path, opts) -> Result<()>`
- [x] Document `cp -r` style directory placement semantics: existing destination directory
  means copy into it; missing destination path means copy as that path
- [x] Lock the public return shape to `Result<()>` initially; no transfer summary type yet
- [x] Keep failure modes non-panicking; validation and I/O errors return `Err`

### 1.13c — Local transport implementation (`src/transport.rs`)

- [x] Implement `LocalTransport::upload()` for regular files
- [x] Implement `LocalTransport::download()` symmetrically
- [x] Implement recursive directory copy when `opts.recursive == true`
- [x] Return error for directory sources when `opts.recursive == false`
- [x] Return error when destination exists and `opts.overwrite == false`
- [x] Reject symlinks encountered during upload/download rather than following them
- [x] For directory destinations with `opts.overwrite == true`, merge into the existing
  tree: overwrite conflicting files, create missing entries, preserve extras
- [x] Wrap each top-level transfer in the existing local transport timeout

### 1.13d — Mock transport implementation (`src/transport.rs`)

- [x] Add an in-memory filesystem/tree model for `MockTransport`
- [x] Support file and directory upload/download semantics
- [x] Mirror the same directory merge semantics as the local implementation
- [x] Mirror the same symlink rejection semantics as the local implementation
- [x] Add deterministic transfer error injection for tests
- [x] Keep command mocking behavior unchanged for existing `exec()` tests

### 1.13e — SSH SFTP implementation (`src/transport.rs`)

- [x] Add explicit `russh-sftp` dependency to `Cargo.toml`
- [x] Implement `SshTransport::upload()` using SFTP for regular files
- [x] Implement `SshTransport::download()` using SFTP for regular files
- [x] Implement recursive directory upload/download over SFTP
- [x] Enforce `overwrite=false` and `recursive=false` semantics consistently
- [x] Reject symlinks encountered during SFTP traversal rather than following them
- [x] For directory destinations with `opts.overwrite == true`, mirror the same merge
  semantics as local/mock implementations
- [x] Bound each top-level SFTP transfer by `SshConfig::timeout`
- [x] Open a fresh SFTP channel per top-level transfer; no shared client cache initially

### 1.13f — HostHandle, public exports, and call-site wiring (`src/host.rs`, `src/lib.rs`, callers`)

- [x] Add `HostHandle::upload(&Path, &Path, ...)` / `HostHandle::download(&Path, &Path, ...)`
- [x] Re-export `TransferOptions` and any new public transfer types from `lib.rs`
- [x] Update callers/tests/examples to use `upload(...)` / `download(...)`
- [x] Do not add file transfer methods on `Target`

### 1.13g — Unit tests

- [x] `MockTransport` file upload/download round-trip
- [x] `MockTransport` directory upload/download round-trip
- [x] `MockTransport` directory copy-into vs copy-as behavior for existing vs missing destination roots
- [x] `MockTransport` directory merge semantics with `overwrite=true`
- [x] `MockTransport` symlink rejection path
- [x] `overwrite=false` conflict path
- [x] `recursive=false` directory rejection path
- [x] `TransportKind` dispatch tests for `upload()` / `download()`

### 1.13h — Integration tests

- [x] Localhost file upload/download round-trip with exact byte verification.
  Run unconditionally; no `tmux` availability gate is relevant for host-level file transfer.
- [x] Localhost directory upload/download round-trip for a nested tree
- [x] Localhost directory copy-into vs copy-as behavior for existing vs missing destination roots
- [x] Localhost directory merge behavior with `overwrite=true`
- [x] Localhost overwrite=false and recursive=false error paths
- [x] Localhost symlink rejection path
- [x] SSH file upload/download round-trip using the existing `MOTLIE_SSH_TEST_HOST`
  env gate (no new env var)
- [x] SSH directory upload/download round-trip using the existing
  `MOTLIE_SSH_TEST_HOST` env gate
- [x] SSH directory copy-into vs copy-as behavior for existing vs missing destination
  roots using the existing `MOTLIE_SSH_TEST_HOST` env gate
- [x] SSH directory merge behavior with `overwrite=true` using the existing
  `MOTLIE_SSH_TEST_HOST` env gate
- [x] SSH overwrite=false and recursive=false error paths using the existing
  `MOTLIE_SSH_TEST_HOST` env gate
- [x] SSH symlink rejection path using the existing `MOTLIE_SSH_TEST_HOST` env gate

### 1.13i — Documentation and behavior verification

- [x] Update `docs/API.md` with host-level upload/download examples and boundary notes:
  transport `exec()` vs host upload/download vs pane `Target::exec()`
- [x] Add `upload` and `download` commands to `examples/repl.rs` with `--recursive` flag
- [x] Cross-link implementation docs to [`SFTP.md`](./SFTP.md)

**Depends on**: 1.3, 1.7, 2a.1

**Gates**: None — additive. Can run parallel with 2a.2 and 2a.3. If 2a.4 starts in
parallel, serialize `host.rs` edits to avoid merge contention.

---

## Phase 1.14: Target Hierarchy Creation Symmetry

Add first-class window and pane creation to `Target` so the hierarchy API is
structurally symmetric: session root creation on `HostHandle`, child creation on
`Target`, and `kill()` as the inverse lifecycle operation at every level. This is
the next planned API task to remove the current `target.exec("tmux ...")`
workaround from examples and consumer code.

### 1.14a — Creation option types (`src/types.rs`)

- [x] Add public `CreateWindowOptions` with `Default`:
  `name`, `command`, `width`, `height`, `start_directory`
- [x] Add public `SplitDirection` enum: `Horizontal`, `Vertical`
- [x] Add public `SplitSize` enum: `Cells(u16)`, `Percent(u8)`
- [x] Add checked `SplitSize::percent(value: u8) -> Result<Self>` constructor so
  invalid percentages are rejected at call-site time rather than first failing
  in the control layer
- [x] Add public `SplitPaneOptions`:
  `direction`, `size`, `command`, `start_directory`
- [x] Rustdoc the level semantics and option-to-tmux flag mapping

### 1.14b — Control-layer tmux wrappers (`src/control.rs`)

- [x] Add `new_window(transport, socket, session, opts) -> Result<WindowInfo>`
- [x] Add `split_pane(transport, socket, target, opts) -> Result<PaneAddress>`
- [x] Use `tmux new-window -P -F ...` to capture the created window identity in the
  same command that performs the mutation
- [x] Use `tmux split-window -P -F ...` to capture the created pane identity in the
  same command that performs the mutation
- [x] Shell-escape all user-provided fields (`name`, `command`, `start_directory`)
- [x] Reject non-UTF-8 `start_directory` paths with `Err` when converting `PathBuf`
  into tmux command arguments
- [x] Keep defensive execution-time validation for split percentages even though
  `SplitSize::percent(...)` is the preferred construction path
  <!-- 2026-03-19 @codex -- tmux 3.4 uses `split-window -l <n>%` for percentage
       splits; the public API stays `SplitSize::Percent` / `--percent`, but the
       emitted control command must use `-l 40%`, not a nonexistent `-p`. -->

### 1.14c — `Target` API wiring (`src/host.rs`, `src/lib.rs`)

- [x] Add `Target::new_window(&CreateWindowOptions) -> Result<Target>`
- [x] Add `Target::split_pane(&SplitPaneOptions) -> Result<Target>`
- [x] `new_window()` is session-level only; window/pane calls return structured `Err`
- [x] `split_pane()` is window/pane-only; session calls return structured `Err`
- [x] Window target split semantics: split the active pane in that window
- [x] Pane target split semantics: split the addressed pane explicitly
- [x] Re-export `CreateWindowOptions`, `SplitDirection`, `SplitSize`,
  `SplitPaneOptions` from `lib.rs`

### 1.14d — Unit tests

- [x] `control.rs` tests for `new-window` command generation and printed-result parsing
- [x] `control.rs` tests for `split-window` command generation, direction, and size flags
- [x] `Target` level-gating tests (`new_window` rejects window/pane, `split_pane`
  rejects session)
- [x] Returned `Target` tests: created window carries `WindowInfo`, created pane
  carries `PaneAddress`
- [x] Invalid option tests: out-of-range split percentages, malformed parsed output

### 1.14e — Integration tests

- [x] Localhost round-trip: create session → create window → split pane → verify
  `children()` reflects the new hierarchy
- [x] Verify returned window/pane targets are immediately usable for `send_text()`,
  `capture()`, and `kill()`
- [x] Verify window-level split uses the active pane in that window
- [x] Verify pane-level split targets the explicit pane
- [ ] Verify rename/kill symmetry on newly created window/pane targets
  <!-- 2026-03-19 @codex -- Kill symmetry is covered in the localhost round-trip. Rename on
       freshly created window/pane targets is still an open follow-up; pane rename is a tmux
       non-feature, so only window rename remains worth adding if we want explicit coverage. -->

### 1.14f — Documentation and behavior verification

- [x] Update `docs/API.md` with first-class `new_window()` / `split_pane()` examples
- [x] Update `examples/target_navigate.rs` to stop using `exec("tmux new-window ...")`
  for demo setup
- [x] Rewrite the `examples/README.md` "Future" section that currently documents the
  `exec("tmux new-window ...")` workaround; replace it with the new hierarchy
  creation path and any remaining follow-up scope
- [x] Decide whether `examples/repl.rs` should grow explicit `new-window` /
  `split-pane` commands in the same round or as immediate follow-up; document the choice

**Depends on**: 1.6, 1.7, 1.12

**Gates**: None — additive. Prioritize immediately before Track B to keep the typed
Target hierarchy complete before layering more consumer APIs on top.

---

## Phase 1.15: SSH Identity File Authentication

Extends the SSH URI and `SshConfig` with an `identity-file` parameter for explicit
private-key authentication (DC26). No new dependencies — uses `russh_keys::load_secret_key()`
and `handle.authenticate_publickey()` already in the crate's dependency tree.

**Depends on**: 1.11 (URI parsing infrastructure), 2a.1 (SSH transport / `authenticate_agent`)

### 1.15a — `SshConfig` field and builder (`src/transport.rs`)

- [x] Add `identity_file: Option<PathBuf>` field to `SshConfig` struct
- [x] Add `with_identity_file(self, path: impl Into<PathBuf>) -> Result<Self>` — fallible
  builder that returns `Err` if `identity_file` is already `Some` (prevents silent
  overwrite when combining URI parse with programmatic config)
- [x] Add `identity_file(&self) -> Option<&Path>` accessor
- [x] Update `PartialEq`/`Debug` derives to include the new field
- [x] Default: `None` (existing agent auth behavior unchanged)

### 1.15b — Key file authentication (`src/transport.rs`)

- [x] Add `authenticate_key_file(handle, config, key_path) -> Result<()>` private method
  on `SshTransport`
- [x] Use `russh_keys::load_secret_key(path, None)` to load the key — no passphrase in v1
- [x] Use `handle.authenticate_publickey(user, Arc::new(key_pair))` for auth
- [x] Actionable error messages:
  - Key file not found / unreadable → suggest checking path and permissions
  - Encrypted key (passphrase required) → suggest loading into ssh-agent instead
  - Key rejected by server → include path, host, port in error
- [x] Update `SshTransport::connect()` to dispatch:
  `if identity_file.is_some() → authenticate_key_file() else → authenticate_agent()`

### 1.15c — URI parsing and rendering (`src/uri.rs`)

- [x] Add `"identity-file"` to `KNOWN_PARAMS`
- [x] Add `"identity-file"` to a new `QUERY_ONLY_PARAMS` list (or equivalent guard)
- [x] In `parse()`, reject `identity-file` if it appears in userinfo params:
  `"identity-file is a query-only parameter"` — absolute paths are a poor fit for the
  userinfo/authority/path split (DC26 rationale)
- [x] Add parse match arm for query-param `identity-file`:
  - Validate path is absolute (reject relative paths with clear error)
  - Reject empty value
  - Set `config.identity_file`
- [x] Add render logic in `to_uri_string()`:
  - Always emit `identity-file` as a query param, even when user is non-empty
    (other params go to nassh userinfo in that case)
- [x] Round-trip: absolute POSIX paths should round-trip safely (no URI-reserved chars)

### 1.15d — Unit tests (`src/uri.rs`, `src/transport.rs`)

- [x] Parse tests:
  - `ssh://deploy@host?identity-file=/path/to/key` — query style, accepted
  - `ssh://deploy;identity-file=/path/to/key@host` — nassh style, **rejected**
    ("identity-file is a query-only parameter")
  - Round-trip for identity-file URIs (with user, without user)
  - Reject relative path: `ssh://deploy@host?identity-file=relative/key` → error
  - Reject empty path: `ssh://deploy@host?identity-file=` → error
  - Mixed params: `ssh://deploy;timeout=30@host?identity-file=/path` — nassh timeout
    + query identity-file, accepted
- [x] Builder tests:
  - `SshConfig::new(...).with_identity_file("/path")` — accessor returns `Some`
  - Default config — `identity_file()` returns `None`
  - Duplicate-source error: `parse("...?identity-file=/a")?.with_identity_file("/b")`
    → `Err` with message identifying both paths
  - Double builder call: `.with_identity_file("/a")?.with_identity_file("/b")` → `Err`
- [x] `to_uri_string()` with identity-file set:
  - With user: identity-file in query, other params in nassh userinfo
  - Without user: identity-file in query alongside other query params
- [x] Localhost with identity-file: parses OK, connect ignores it (LocalTransport)

### 1.15e — Integration tests (`tests/integration.rs`)

- [x] Localhost with `identity-file` set: connects via LocalTransport (identity-file
  silently ignored), can list sessions — verifies no regression
- [ ] SSH key-file auth test (env-gated, requires test SSH server with known key):
  - Connect with valid key file → auth succeeds
  - Connect with wrong key file → auth fails with actionable error
  - Connect with nonexistent key file → load fails with actionable error
  <!-- @claude 2026-03-20: SSH key-file auth integration tests deferred — requires a test
       SSH server with a known authorized key. Will be added when CI SSH test infra is
       available (Phase 4.3). Localhost test verifies no regression. -->

### 1.15f — Documentation updates

- [x] Update `docs/API.md` — add identity-file examples to the URI / SshConfig section
- [x] Update `examples/README.md` — mention `identity-file` param in Prerequisites
  for key-file workflows
- [x] Update `examples/repl.rs` help text — note that URIs accept `identity-file`

### 1.15g — Example program (`examples/uri_connect.rs`)

- [x] Ensure `uri_connect` example works with identity-file URIs (it already accepts
  any URI — just verify and document in README expected output)
- [x] Add identity-file usage example to `examples/README.md` uri_connect section:
  ```sh
  ./target/debug/examples/uri_connect 'ssh://deploy@prod?identity-file=/path/to/key'
  ```

**Gates**: None — additive. Does not change behavior for any existing URI or code path.

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

### 2a.3 — ~~Pipe-pane fallback (`src/pipe.rs`)~~ — OUT OF SCOPE

<!-- @claude 2026-03-16: Removed. tmux 3.1+ baseline (established in DC22, per user direction)
     guarantees control mode availability. Pipe-pane fallback adds complexity (FIFO lifecycle,
     P4 cleanup, OC1 backpressure, OC2 interleaving) with no benefit when 3.1+ is the floor.
     Control mode is the only monitoring path. See DESIGN.md DC10 for matching note. -->

---

<!-- @claude 2026-03-17: Phase 2 restructured into Track A (streaming + combining) and
     Track B (matching + reactors) per DC24 analysis. 2026-03-20 @codex: active post-Track-A
     path now shifts again toward transcript/history adapters and simplified Fleet
     coordination for external agents. The older matcher/reactor track is preserved below
     as historical context rather than active execution order. -->

## Track A: Streaming + Stream Combining

Deliver end-to-end real-time monitoring and multi-source stream combining before the
rule engine or reactor machinery. After Track A, the system can: monitor sessions via
control mode, fan-out output to multiple consumers via OutputBus, and combine streams
via JoinedStream — with no matcher or action dispatch dependency.

### 2a.2a — Control mode stream parser (`src/monitor.rs`) — parse only

- [x] `SessionMonitor` struct: session name, stream assembly state (no rules, no cooldowns)
- [x] Control mode stream parser: parse `%output %<pane_id> <data>` frames from
  `tmux -C attach -t <session>` output
- [x] Per-pane stream assembly state keyed by `pane_id`:
  partial-frame buffering, newline canonicalization, monotonic per-pane `sequence`
- [x] Reuse `1.9a` normalization/fidelity path for monitor events:
  preserve ANSI in fidelity modes, apply the same mode-to-field contract,
  annotate degraded/reflow/history conditions in emitted metadata
  <!-- @claude 2026-03-18: Implemented — SessionMonitor accepts CaptureNormalizeMode
       (Raw/ScreenStable/PlainText). process_output() applies normalization, sets
       raw_content when content differs. Fidelity remains clean() for control mode
       (no geometry/reflow concern). Unit tests verify all three modes. -->
- [x] Handle other control mode messages gracefully (`%begin`, `%end`, `%error`, etc.)
- [x] `SessionMonitor::run()` — main loop: read from shell, parse, emit `TargetOutput`
  (no rule evaluation, no action dispatch — streaming only)
- [x] Stop signal via `watch::Receiver<bool>`, clean shutdown on signal or connection drop
- [x] Warn-level logging for malformed lines (P9)
- [x] Unit tests: control mode frame parsing, chunk split/reassembly, sequence monotonicity

**Depends on**: 1.7 ✓, 1.9a ✓, 1.10 ✓

### 2a.4a — Monitor handle wiring (streaming only)

- [x] `SessionMonitorHandle`: `Target` + `stop_tx` + `task: Mutex<Option<JoinHandle>>`
- [x] `SessionMonitorHandle::shutdown()` — signal stop, flush, join task
- [x] `SessionMonitorHandle::is_active()`
- [x] `Deref<Target=Target>` for `SessionMonitorHandle`
- [x] `MonitorHandle`: `HashMap<String, SessionMonitorHandle>`, `shutdown()`,
  `stop_session()`, `get()`, `get_by_spec()`, `active_sessions()`
  <!-- @claude 2026-03-18: All methods implemented. stop_session() removes and shuts
       down by name. get_by_spec() resolves via TargetSpec::session_name(). -->
- [x] `HostHandle::start_monitoring()` — discovers sessions, spawns per-session monitors,
  returns `MonitorHandle` (no rules parameter — monitors all output)
- [x] `HostHandle::start_monitoring_session()`
- [x] `stop_monitoring_session()`, `stop_monitoring()`, `monitored_sessions()` (DC13)
  <!-- @claude 2026-03-18: Implemented. Per-host tracking via monitor_signals map in
       HostHandleInner (watch::Sender keyed by session name). Stop signals fire the
       watch channel; tasks self-deregister on exit. -->
- [x] `Target::start_monitoring()` — session-level only, returns `SessionMonitorHandle`
- [x] `Target::stop_monitoring()` — session-level only, delegates to
  `HostHandle::stop_monitoring_session(&self)`; returns error if called on
  window/pane target
  <!-- @claude 2026-03-18: Implemented. Both methods gate on session level and
       reconstruct HostHandle from inner Arc for delegation. Integration test
       verifies session-only gate and round-trip start/stop. -->
- [x] Integration test (localhost): start monitor, send text to pane, verify
  `TargetOutput` emitted, `target.stop_monitoring()` cleanly stops,
  verify on-demand operations still work after stop (DC13)
  <!-- @claude 2026-03-18: Implemented as localhost_monitor_pipeline in
       tests/integration.rs. Full round-trip: create session → start monitoring →
       subscribe → send text → verify TargetOutput received → stop_monitoring() →
       verify capture still works → verify Target::start/stop_monitoring() →
       verify session-level-only gate on pane targets → cleanup. -->

**Depends on**: 2a.2a ✓

### 2c.1a — Sink types (`src/sink.rs`) — no content matching

- [x] `TargetOutput` struct: `source: TargetAddress`, `host`, canonical `content`,
  optional `raw_content`, `sequence`, `fidelity`, `timestamp`
- [x] `SinkEvent` enum: `Data(TargetOutput)` and `Gap { dropped, timestamp }`
- [x] `TargetOutput` accessors: `session_name()`, `pane_id()`, `target_string()`
- [x] `OutputFidelity` / `FidelityIssue` enums shared with capture/monitor paths
- [x] `SinkFilter`: `host`, `session`, `window`, `pane` (all optional regex strings)
  — routing only, no `content` / `MatcherKind` / `MatcherInput` fields yet
- [x] `CompiledSinkFilter`: compiled regexes,
  `matches(&self, output: &TargetOutput) -> bool`
- [x] `SinkId` opaque type

**Depends on**: 1.9a ✓

### 2c.2a — Sink kinds + Subscription + JoinedStream (`src/sink.rs`, `src/sinks/`)

- [x] `SinkKind` enum: `Stdio(StdioSink)`, `Callback(CallbackSink)` — no `JoinedSink` variant (DC24)
- [x] `SinkKind::name()`, `write()`, `flush()` dispatch methods
- [x] `CallbackSink`: `name`, `state: Arc<dyn Any + Send + Sync>`,
  `on_output: fn(...)`, `on_flush: Option<fn(...)>`
- [x] `StdioSink` in `src/sinks/stdio.rs`: `StdioFormat` enum (Raw, Prefixed, Json),
  immediate write to stdout, no batching
- [x] `Subscription` type wrapping `(SinkId, mpsc::Receiver<SinkEvent>)` with
  Track A adapters: `.id()`, `.into_receiver()`, `.joined(LabelFormat)`, `.pipe(SinkKind)` (DC24)
- [x] `JoinedStream`: returned by `subscription.joined()`, produces `StreamChunk`s
  with `source_changed` flag + source coalescing (DC24, revised DC15)
- [x] `JoinedStream::next()` and `JoinedStream::format()` methods
- [x] `SourceLabel` struct: `host`, `target: TargetAddress`, `short()`, `minimal()` formatters
- [x] `StreamChunk` struct: `source: SourceLabel`, `output: TargetOutput`, `source_changed: bool`
- [x] `LabelFormat` enum: `Bracketed`, `Prompt`, `Custom(fn)`
- [x] Unit tests: JoinedStream coalescing + source_changed flag, label formatting,
  gap passthrough, Subscription::pipe spawns task, Subscription::into_receiver works

**Depends on**: 2c.1a ✓

### 2c.3 — Output bus (`src/sink.rs`)

- [x] `OutputBus::new()`
- [x] `subscribe(filters, capacity) -> Subscription` — single subscription primitive;
  all consumer composition layered on `Subscription` adapters (DC24)
- [x] `unsubscribe(id) -> Result<()>` — drops sender, closing channel; piped tasks drain and exit on their own (caller holds `PipeHandle` for join semantics)
- [x] `publish(output: TargetOutput)` — fan out to all matching subscribers via `try_send`
  while tracking per-subscriber dropped counts
- [x] No-silent-drop contract: if drops occurred, emit `SinkEvent::Gap { dropped }`
  before the next `SinkEvent::Data(output)` on that subscriber route
- [x] `shutdown()` — drops all senders, closing channels; piped tasks drain and exit on their own (caller holds `PipeHandle` for join semantics)
- [x] `SubEntry` internal: id, name, tx, compiled filters
- [x] Unit tests: fan-out to 3 subscribers, slow subscriber doesn't block others,
  source-routing filter matching, gap-event emission after drops, shutdown flushes

**Depends on**: 2c.2a ✓

### 2c.4a — Pipeline integration (streaming only)

- [x] Wire `OutputBus` into `monitor.rs`: `SessionMonitor` publishes `TargetOutput`
  to the bus (no rule evaluation, no action dispatch)
- [x] `HostHandle` owns or shares `OutputBus`; monitoring tasks publish through it
- [x] Integration test: monitor publishes output via mock shell → bus → subscription
  <!-- @claude 2026-03-18: Mock-based pipeline test + live localhost_monitor_pipeline
       integration test both implemented. -->

**Depends on**: 2c.3 ✓, 2a.4a ✓

---

## Track B: History + External-Agent Ergonomics

Build on the completed streaming substrate from Track A by making transcript/history
consumption and routed control pleasant for external LLM/classifier workflows.

### 2b.1 — Transcript/history adapters (`src/sink.rs` extension)

- [x] Add `Subscription::history(opts) -> HistoryHandle`
- [x] Implement `HistoryOptions` with:
  - `max_entries`
  - `max_render_chars`
  - `label_format`
  - `include_omission_marker`
- [x] Build history on top of `JoinedStream` source-coalescing logic (replicates tracking inline to also capture Gap events)
- [x] Define `HistorySnapshot` and `HistoryEntry` (`Output`, `Gap`)
- [x] `HistoryHandle::snapshot()` — structured access for custom formatters
- [x] `HistoryHandle::render_text()` — prompt-ready rolling transcript for LLM/classifier context
- [x] Oldest-first trimming across the **global merged transcript**, not per-source windows
- [x] Omission marker support when older entries are trimmed
- [x] Unit tests:
  - source attribution and coalescing (`history_source_coalescing`)
  - oldest-first trimming by entry count (`history_trims_oldest_by_entry_count`)
  - rendered-char budget trimming (`history_trims_by_render_char_budget`)
  - explicit gap propagation into history (`history_gap_propagation`)
  - omission marker rendering (`history_render_text_with_omission`)
  - rolling `render_text()` behavior under sustained multi-source output (`history_rolling_under_sustained_output`)

**Depends on**: 2c.2a ✓, 2c.4a ✓

### 2b.2 — Consumer predicate helpers (`src/sink.rs` extension)

- [x] Add lightweight predicate-based filtering helpers that do not require a built-in
  matcher DSL — `Subscription::filter_fn(predicate: fn(&TargetOutput) -> bool)`
- [x] Keep selection logic consumer-owned and composable over `TargetOutput`
- [x] Unit tests: predicate pass/block behavior (`filter_fn_passes_matching_events`),
  gap forwarding (`filter_fn_always_forwards_gaps`), composition with pipe (`filter_fn_composes_with_pipe`)

**Depends on**: 2c.2a ✓

### 2b.3 — Fleet coordination + routing (`src/fleet.rs`)

- [x] `Fleet::new()` — empty programmatic registry
- [x] Programmatic host registration (`register()`) with alias support and conflict detection
- [x] `Fleet::host(name) -> Option<&HostHandle>` — alias lookup (DC9)
- [x] `Fleet::hosts()` iterator
- [x] `Fleet::output_bus()` accessor (owns `OutputBus`, shares via `Arc`)
- [x] Aggregate monitoring lifecycle: `start_monitoring_session()`, `start_monitoring_host()`,
  `stop_monitoring_host()`, `shutdown()`
- [x] Workstream registry: `bind()`, `unbind()`, `find()`, `workstreams()`
- [x] Convenience routed actions: `send_text`, `send_keys`, `capture`, `target`
- [x] `HostStatus` enum: `Connected`, `Monitoring { sessions }`, `Error(String)`
- [x] Delegation contract documented in module-level rustdoc
- [x] Unit tests: alias conflict detection (`fleet_alias_conflict_detection`),
  workstream bind/find/unbind (`fleet_workstream_bind_find_unbind`),
  shared bus (`fleet_output_bus_is_shared`), shutdown (`fleet_shutdown_clears_state`)

**Depends on**: 2c.4a ✓

### 2b.4 — Public API + examples for external-agent workflows

- [ ] Re-export the active consumer-facing types from `lib.rs`:
  `Fleet`, `HostHandle`, `Target`, `TargetSpec`, `SessionMonitorHandle`, `MonitorHandle`,
  `KeySequence`, `SpecialKey`, `ScrollbackQuery`, `ExecOutput`,
  `OutputBus`, `Subscription`, `SinkKind`, `StdioSink`, `CallbackSink`, `SinkFilter`,
  `TargetOutput`, `StreamChunk`, `JoinedStream`,
  transcript/history types, `SessionInfo`, `WindowInfo`, `PaneInfo`, `PaneAddress`,
  `TargetAddress`, `TmuxSocket`
- [ ] Doc comments on `lib.rs` with external-agent-oriented usage example
- [ ] API/examples sweep: monitor → transcript/history → routed control

**Depends on**: 2b.1, 2b.2, 2b.3

### 2b.5 — CLI / smoke coverage for the simplified direction

- [ ] Keep CLI/examples focused on connect, monitor, transcript inspection, capture,
  send, exec, and routing rather than built-in rule execution
- [ ] Signal handling: `tokio::signal` for SIGINT/SIGTERM → `fleet.shutdown()` (P5)
- [ ] Tracing init: text or JSON output modes, per-target tracing spans
- [ ] End-to-end smoke test:
  - Start CLI/examples with localhost
  - Create session, list, capture, send, exec, rename, kill
  - Start monitor, build transcript/history, verify routed follow-up action
  - Ctrl-C gracefully shuts down

**Depends on**: 2b.4

---

## Historical Context: Deferred Built-In Automation Track

Preserved for continuity, but **not** on the active execution path after 2026-03-20:

- Built-in matcher DSL (`MatcherKind`)
- Config DTOs (`TmuxAutomatorConfig`, `TriggerRule`, `ReconnectPolicy`)
- Subscription-side `.react()` action dispatch and `ActionHandle`
- Reconnecting internal rule processors
- Config-driven `Fleet::new(config)` / `monitor start --config` automator flow

If Motlie later grows a first-class built-in automation product direction, re-evaluate
those ideas against the then-current stream/history/Fleet APIs instead of reinstating
this older track wholesale.

Note: SSH reconnect / long-lived host reliability is still an active infrastructure concern.
It remains in Phase 4 hardening and any Fleet reliability follow-on work; only the
config-driven automator coupling was deferred.

---

## Phase 4: Hardening + Testing

### 4.1 — Tmux version compatibility (OC4)

- [ ] Runtime `tmux -V` detection at startup
- [ ] Feature matrix: validate required features against detected version
  (control mode, `capture-pane -p`, `set-option -p`, `#{pane_id}`)
- [ ] Clear error messages for unsupported versions (minimum: tmux 3.1)
- [ ] CI matrix: test against tmux 3.1, 3.x latest, latest

### 4.2 — Streaming resilience + failure semantics

- [x] **4.2a — Reconnecting monitor supervision**
  - wrap session monitor lifecycle so unexpected control-mode EOF does not permanently
    kill monitoring on the first failure
  - add bounded retry/backoff policy for SSH-backed monitors
  - support localhost control-mode reattach after tmux server/client interruption
  - preserve explicit caller-driven stop/shutdown semantics (no reconnect after
    intentional stop)
  <!-- @claude 2026-03-20 — Implemented in host.rs: supervision loop with exponential backoff (1s–30s, max 5 retries), MonitorExitReason-based dispatch, session-existence verification on reconnect, intentional-stop detection via tokio::select!. -->
- [x] **4.2b — Upstream discontinuity artifact**
  - introduce `SinkEvent::Discontinuity` (or equivalent) distinct from `SinkEvent::Gap`
    for upstream monitor
    interruption/resume/resync
  - document invariant: `Gap` means subscriber backpressure only; discontinuity means
    monitor/transport continuity was broken
  - thread the new artifact through:
    - raw subscriptions
    - `filter_fn()` (always forwarded)
    - `pipe()` / terminal sinks
    - `HistoryEntry::Discontinuity`
    - `JoinedStream` source-reset semantics
  <!-- @claude 2026-03-20 — Implemented in sink.rs: SinkEvent::Discontinuity variant, HistoryEntry::Discontinuity, filter_fn always-forward, JoinedStream source-reset, OutputBus::publish_discontinuity(), pipe/stdio/integration match arms. 6 unit tests. -->
- [x] **4.2c — Fresh snapshot anchoring after reconnect**
  - after successful reconnect, capture a bounded **current-state snapshot** instead of
    pretending missed output was recovered
  - record snapshot outcome as an explicit transcript/system entry
  - define limits for snapshot scope so recovery is predictable and does not explode
    memory/history budgets
  - test/document invariant: reconnect snapshot re-anchors the stream but does **not**
    replay output lost during the outage
  <!-- @claude 2026-03-21 — Implemented: emits "stream resumed" discontinuity, then lists panes in session and publishes each pane's visible content as TargetOutput for real screen-state re-anchoring. -->
- [x] **4.2d — Fleet/host streaming health**
  - introduce per-session monitor health as ground truth:
    `streaming | reconnecting | failed | stopped`
  - derive host/Fleet health from per-session state (counts or worst-of), rather than a
    single flattened host status
  - ensure alias/workstream routing remains stable across reconnect attempts
  - specify behavior when reconnect succeeds but the monitored session no longer exists
    (tmux restart / external kill)
  - specify behavior when pane topology changes across outage and pane-id-filtered
    subscriptions no longer match
  - expose enough state for future TUI/dashboard surfaces without inventing a second
    monitoring model
  <!-- @claude 2026-03-20 — Implemented: MonitorHealth enum in monitor.rs, SessionMonitorHandle.health() accessor, SessionMonitorStatus in fleet.rs, host_status() reports per-session health. 3 unit tests. -->
- [x] **4.2e — Stress + failure injection coverage**
  - expand `MockTransport` test suite: error paths, timeouts, malformed tmux output
  - `OutputBus` stress test: high-throughput publish with slow/full sinks
  - transcript/history determinism tests under bursty multi-source output and explicit
    discontinuity/resync markers
  - Fleet routing resilience test: simulated SSH drop + recover while preserving
    alias/workstream lookup semantics
  - integration tests that intentionally kill/restart tmux server or break control-mode
    shell during active monitoring
  <!-- @claude 2026-03-20 — Implemented: MockTransport.with_shell_sequence() for multi-phase reconnect testing (transport.rs), OutputBus stress test with slow subscriber + Gap verification (sink.rs), history determinism test with multi-source + discontinuity interleaving (sink.rs), bus discontinuity under backpressure test (sink.rs). Fleet routing resilience and live kill/restart integration tests deferred to 4.3 Docker-based E2E. -->
- [x] **4.2f — Shell/input hardening**
  - shell escaping fuzz tests or property tests (adversarial session names, text input)
  <!-- @claude 2026-03-20 — Implemented: adversarial input test (17 cases: injection, unicode, control chars, long strings) and round-trip identity property test (10 cases) in transport.rs. -->

- [x] **4.2g — Dedicated socket-isolation ergonomics**
  - add deterministic helper for dedicated automation socket naming on top of `TmuxSocket`
  - add fallible `SshConfig` convenience builder for automation socket selection that errors
    instead of silently overwriting existing socket intent
  - add `HostHandle::ensure_socket_server()` / equivalent bootstrap helper that explicitly
    runs `tmux start-server` against the configured socket
  - document dedicated socket usage as the preferred operational isolation path for robust
    automation / external-agent workflows
  - add localhost integration coverage showing dedicated socket isolation from the default
    tmux server
  - add env-gated SSH integration coverage for socket-scoped startup if practical; otherwise
    defer the SSH-specific operational path to 4.3 Docker E2E
  - expected impact radius:
    `src/types.rs`, `src/transport.rs`, `src/uri.rs`, `src/host.rs`, examples/docs, and
    socket-focused localhost/SSH integration tests
  - implementation note: keep this additive/localized; it should not require OutputBus,
    monitor, Fleet, or history refactors
  <!-- @claude 2026-03-21 — Implemented: TmuxSocket::automation(scope), SshConfig::with_automation_socket(), HostHandle::ensure_socket_server(), unit tests (8 types.rs + 3 transport.rs + 2 host.rs), integration test (localhost_automation_socket_lifecycle), API.md sections. -->

- [x] **4.2h — Tracked command execution**
  - add typed command identity and state:
    `ExecId`, `ExecHandle`, `ExecState`
  - add `Target::start_exec(...) -> ExecHandle`
  - add `ExecHandle::status()` / `wait()` with explicit `Unknown { reason }` outcome when
    continuity breaks before completion can be proven
  - keep `Target::exec()` as the await-to-completion convenience wrapper; layer it on the tracked
    execution substrate where practical
  - store execution state in-process on the host/target side only; no persistent job store
  - preserve same-pane concurrency restrictions from DC19
  - add unit tests for:
    - running → completed transitions
    - discontinuity → unknown transitions
    - same-pane lock sharing across target levels
  - add localhost integration coverage for long-running execution + later status retrieval
  - expected impact radius:
    primarily `src/types.rs`, `src/host.rs`, docs/examples/tests, with possible targeted
    monitor/discontinuity integration if execution truthfulness needs explicit stream-state hooks
  - implementation staging:
    1. add `ExecId` / `ExecHandle` / `ExecState` plus in-memory state model
    2. add `Target::start_exec(...)` and polling/status/wait behavior
    3. refactor `Target::exec()` to layer on the tracked substrate
    4. finalize discontinuity/unknown semantics and integration coverage
  - slice-3 guardrail: before shipping the `Target::exec()` refactor, prove parity with the
    current sentinel mechanism:
    marker injection, scrollback polling/retry behavior, and same-pane exec lock semantics
    must remain covered by tests rather than assumed from the refactor shape
  - implementation note: keep the first slice narrow and reviewable; avoid coupling initial
    tracked execution state to Fleet/history unless a concrete correctness need appears
  <!-- @claude 2026-03-21 — Implemented all 4 slices: (1) ExecId/ExecState/ExecHandle types, (2) Target::start_exec() with background tokio task, pane lock inside task, sentinel polling, (3) exec() refactored onto start_exec()+wait(), (4) active_execs tracking + notify_exec_discontinuity() wired into monitor supervision ConnectionLost path. Parity guardrail: all 6 sentinel parse tests + 3 exec lock tests pass unchanged. 378 unit tests, 2 new integration tests. -->

### 4.3 — Docker-based E2E (OC6)

- [ ] Dockerfile: SSH server + tmux + test sessions
- [ ] E2E test: connect via SSH, full lifecycle (create, monitor, build transcript/history,
  route follow-up control action, capture, kill)
- [ ] CI integration: run E2E tests in Docker on PR

### 4.4 — Documentation

- [ ] Document minimum tmux version (3.1+) and known limitations
- [ ] Document performance characteristics (latency, throughput)
- [ ] Crate-level rustdoc with examples

---

## Phase 5: TUI Interface

- [x] **5.1 — Split-screen REPL shell**
  <!-- @claude 2026-03-22: implemented in examples/repl/main.rs -->
  - `tui on` / `tui off` commands added to `examples/repl`
  - alternate-screen mode with vertical split: top mirror frame, bottom REPL frame
  - preserves current non-TUI REPL behavior when the mode is off
  - terminal state restored on exit, error, and Ctrl-C (panic hook)
- [x] **5.2 — Binary-local mirror consumer**
  <!-- @claude 2026-03-22: implemented in examples/repl/tui_mirror.rs -->
  - `tui_mirror` module beside the REPL, not a new `SinkKind` variant
  - consumes `OutputBus` via `Subscription` + `HistoryHandle` for bounded top-frame state
  - surfaces idle/active/waiting states in the mirror frame
  - draw cadence and terminal ownership local to the binary
  - `ratatui` + `crossterm` added as dev-dependencies only
- [ ] **5.3 — Watch binding semantics**
  - in plain mode, `monitor <session>` keeps current stdout-oriented streaming
  - in TUI mode, `monitor <session>` binds the watched session to the top mirror frame
  - render an explicit idle placeholder before any session is watched
- [ ] **5.4 — Docs and tutorial coverage**
  - document `tui on`, `monitor <session>`, and `tui off` in `examples/README.md`
  - add API/example references for the split-screen workflow
  - include expected output/screenshot guidance for manual verification
- [ ] **5.5 — Full terminal-state mirror follow-on**
  - evaluate whether the transcript/history-oriented mirror is sufficient
  - if not, add the deeper cursor-addressed mirror path from `TUI.md`
  - do not block 5.1–5.4 on this deeper path

**Depends on**: Track A / 2c stable plus Fleet coordination work from Track B

Implementation note:
- 5.1–5.4 are the intended first delivery.
- 5.5 is explicitly later; the REPL split-screen mode should ship first.

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
 │                      │
 │    1.7 Host+Target ◄─┘
 │     │
 │     ├── 1.8 Localhost integration test
 │     ├── 1.9a Capture fidelity types [needs 1.5 + 1.7]
 │     ├── 1.9b Mixed-client stabilization [needs 1.9a]
 │     ├── 1.10 API gaps + hardening [gates 2a.2a]
 │     ├── 1.11 SshConfig URI ext (src/uri.rs)
 │     ├── 1.13 SFTP file transfer [needs 1.3 + 1.7 + 2a.1]
 │     ├── 1.14 Target hierarchy creation symmetry [needs 1.6 + 1.7 + 1.12]
 │     │
 │  TRACK A — Streaming + Combining (priority)
 │  ──────────────────────────────────────────
 │     └── 2a.2a Stream parser (parse only)
 │          └── 2a.4a Monitor handles (streaming only)
 │               │
 │     2c.1a Sink types (routing only, no content matching)
 │       └── 2c.2a Sink kinds + JoinedStream combinator (DC24)
 │            └── 2c.3 Output bus
 │                 └──── 2c.4a Pipeline integration (streaming only)
 │                        │     [needs 2c.3 + 2a.4a]
 │                        │
 │  TRACK B — History + External-Agent Ergonomics
 │  ─────────────────────────────────────────────
 │     2b.1 Transcript/history adapters [needs 2c.2a + 2c.4a]
 │      ├── 2b.2 Consumer predicate helpers [needs 2c.2a]
 │      └── 2b.3 Fleet coordination + routing [needs 2c.4a]
 │             └── 2b.4 Public API + examples [needs 2b.1 + 2b.2 + 2b.3]
 │                  └── 2b.5 CLI / smoke coverage [needs 2b.4]
 │
 └── 4.x Hardening (parallel with Track B)
      ├── 4.2 Streaming resilience [complete]
      ├── 4.2g Socket-isolation ergonomics
      └── 4.2h Tracked command execution
```

**Linear execution order (single-threaded)**:
```
2a.2a → 2a.4a → 2c.1a → 2c.2a → 2c.3 → 2c.4a → 2b.1 → 2b.2 → 2b.3 → 2b.4 → 2b.5
```

Notes:
- `1.14` and `1.15` are complete. Track B established the transcript/history/Fleet
  substrate; the next active implementation priorities are the product-driven robustness
  follow-ons after completed Phase `4.2`: **4.2g dedicated socket isolation ergonomics**
  first, then **4.2h tracked command execution**.
- After Track A (6 tasks ending at 2c.4a), the system has end-to-end streaming:
  monitor → parse control mode → fan-out via OutputBus → Subscription adapters →
  JoinedStream combining. No matcher or reactor dependency.
- The active post-Track-A direction prioritizes transcript/history construction,
  simplified Fleet coordination, and external-agent workflows.
- [`docs/PRODUCT.md`](../../../docs/PRODUCT.md) now informs the next robustness work:
  socket isolation is the higher-priority operational reliability feature; tracked
  command execution follows as a pane-scoped execution-state improvement.
- Completed **4.2 streaming resilience** established reconnect supervision, explicit
  discontinuity semantics, fresh snapshot anchoring, and Fleet health visibility for
  long-lived agent loops.
- The older matcher/rule/reactor/config direction is preserved in the historical
  context section, not in the active dependency chain.
- `1.10` gates Track A start (2a.2a).
- `OutputBus::subscribe()` returns `Subscription` — the single composable seam.
  All consumer composition (joining, history construction, piping, lightweight
  predicates) is layered on adapters (DC24).
- Phase 4 tasks are independent and can start as soon as their prerequisites are met.

## Conventions

- Each task should be a PR-sized unit (reviewable in isolation)
- Every task ends with `cargo check` and `cargo test` passing
- Use `MockTransport` for unit tests; real tmux for integration tests
- Integration tests gated on `which tmux` availability check
- Follow workspace patterns: `anyhow` for errors, `tracing` for logs, `serde` for config
