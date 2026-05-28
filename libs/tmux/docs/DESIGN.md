# Tmux Multi-Target Automator Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-05-28 | @codex: Add the host-alias integration cleanup from feature/mstream: `SshConfig::connect_with_alias()` creates `HostHandle`s with caller-owned Fleet aliases, `Fleet::unregister()` removes host, monitor, and target-alias bookkeeping together, timeline lifecycle remains on `OutputBus` rather than duplicated on Fleet, and historical workstream/short-name aliases were removed in favor of explicit target-alias APIs. | Fleet, DC21 |
| 2026-05-27 | @gpt55-337-og: Issue #337 Fleet API follow-up — add cross-host `FleetTargetSpec`, target aliases, fleet-wide session inventory with generic tag prefixing, batch tag writes/removals, idempotent target monitoring, and timeline filter/scope helpers. Keep application/workflow business concepts outside `libs/tmux`. | Fleet |
| 2026-05-02 | @codex: Added initial session environment values to `CreateSessionOptions` and documented that post-creation `SessionEnvironment` writes only affect future tmux-spawned processes. | HostHandle, Target, control.rs, types.rs |
| 2026-05-02 | @codex: Reworked session status-bar control into `Target::status() -> SessionStatus`, with snapshot/apply/restore helpers and validated `StatusLeftLength`. | Target, SessionStatus |
| 2026-05-02 | @codex: DC34 follow-up for mmux PR feedback — add a host-level batch session-tag read API so selector refreshes can enrich a fresh session listing without one round trip per session. | Target, DC34 |
| 2026-05-01 | @codex: DC34 follow-up for issue #241 — add planned session tag deletion API using tmux `set-option -u`: scoped `SessionTags::unset(key)`. | Target, DC34 |
| 2026-04-30 | @codex: DC34 — session metadata tags on `Target` via tmux user-defined session options. Add scoped `SessionTags` and validated self-describing `SessionTag` with strict session-only scope, stable-id dispatch, namespace/key validation, and small-value bounds. | Target, DC34 |
| 2026-04-28 | @gpt55-dgx: PR #228 selector cleanup — document the implemented non-empty `SessionId` wrapper for `SessionInfo.id` so stable id dispatch cannot silently fall back to names. | Discovery Types |
| 2026-04-09 | @claude: Note anyhow→thiserror migration in dependency table and prototype sections. Library now uses typed `Error` enum via `thiserror`; `anyhow` retained as dev-dependency only. Prototype code snippets are pre-migration and preserved as historical context. | Dependencies, Prototype |
| 2026-03-25 | @claude: DC33 — per-source coherent history rendering. Coalesce same-source chunks, add `RenderMode::PerSource`, per-source budgets. See [`docs/HISTORY.md`](./HISTORY.md). | DC28, DC33, History |
| 2026-03-22 | @claude: Update Phase 5 section to reflect shipped DC32 split-screen REPL mode — replace "not in current scope" / generic `TuiSink` with shipped 5.1+5.2 status and binary-local consumer description. | Phase 5, DC32 |
| 2026-03-22 | @codex: Add DC32 for the first TUI delivery: a split-screen REPL mirror mode with `tui on` / `tui off`. Keep the first mirror consumer binary-local instead of adding a core `SinkKind::Tui`, and stage full terminal-state mirroring after the transcript/history-oriented REPL cut. | DC32, Phase 5, TUI cross-reference |
| 2026-03-21 | @claude: Update DC31 `ExecHandle` contract to match shipped API — `status()` is sync/infallible, `wait()` consumes self. | DC31 |
| 2026-03-21 | @codex: Address PR #96 review feedback — clarify DC31 exit-code semantics, narrow `exec()` wording from "blocking" to "await-to-completion", and tighten product/design wording around competitive evidence and SSH ergonomics. | DC31, DC19, PRODUCT cross-reference |
| 2026-03-21 | @codex: Product-driven follow-up from [`docs/PRODUCT.md`](../../../docs/PRODUCT.md): add DC30 (socket-isolation ergonomics) and DC31 (tracked command execution) as concrete robustness features. Prioritize dedicated automation sockets first, tracked command execution second. | DC30, DC31, DC19, Phase 4 |
| 2026-03-20 | @codex: Refine DC29 per PR #94 review — resync is a fresh snapshot after reconnect, not replay. Specify missing-session/topology-change behavior, adapter propagation (`SinkEvent::Discontinuity`, `HistoryEntry::Discontinuity`, `filter_fn` forwarding, `JoinedStream` source reset), and per-session monitor health as the ground truth for Fleet aggregation. | DC29, Phase 4 |
| 2026-03-20 | @codex: Add DC29 — long-lived streaming resilience. Separate upstream stream discontinuity from subscriber backpressure gaps, require reconnect supervision + fresh snapshot anchoring after reconnect, and make the hardening direction explicit for external-agent/Fleet workflows. | DC29, DC28, Phase 4 |
| 2026-03-20 | @claude: Update Fleet and HistoryHandle API blocks to match shipped implementation (PR #92 R2). Fleet: `register()` with alias enforcement + bus injection, `TargetSpec`-based `bind()`, async `find()`, sync `shutdown()`. HistoryHandle: async `snapshot()` / `render_text()` / `join()`, `id()` accessor. Fix `rendered_chars()` to measure actual rendered string for Custom label format. | Fleet, Subscription, DC27, DC28 |
| 2026-03-20 | @codex: DC28 — specify transcript/history adapter as a bounded rolling snapshot layer for LLM/classifier context windows. Define artifact, bounds, relationship to `JoinedStream`, and consumer interaction model. | Subscription, DC28, Phase 2b |
| 2026-03-20 | @codex: Directional simplification — active design now treats `libs/tmux` as tmux stream/history/control substrate for an external LLM/classifier or other policy engine. Simplify Fleet toward coordination/aggregation/routing. Move matcher/rule/reactor/config automation direction to appendix as historical context. | Overview, Fleet, DC24, DC12, DC13, DC14, Phase 2b/2c, Phase 3, Appendix |
| 2026-03-20 | @claude: DC26 R1 — address PR #89 review: constrain `identity-file` to query-only (not nassh userinfo), make `with_identity_file()` fallible to prevent URI+builder silent overwrite, update DC21 `SshConfig` contract block with new field/methods and query-only exception note. | DC26, DC21, OC3 |
| 2026-03-20 | @claude: DC26 — SSH identity-file URI parameter for explicit key-file authentication. Extends DC21 URI/SshConfig with `identity-file` param; adds `authenticate_key_file()` alongside existing agent auth. Addresses OC3 limitation for CI/agentless workflows. | DC26, DC21, OC3 |
| 2026-03-19 | @codex: DC25 implementation note — correct split-percentage mapping for tmux 3.4: `SplitSize::Percent` maps to `split-window -l <n>%`, not a nonexistent `-p` flag. | DC25 |
| 2026-03-19 | @codex: Add DC25 — first-class window/pane creation on `Target` to restore hierarchy symmetry. Document `new_window()` / `split_pane()` requirements, typed option structs, tmux `-P -F` return strategy, and why `exec(\"tmux ...\")` is not sufficient. | Target, DC25 |
| 2026-03-19 | @codex: Narrow `HostHandle::transport_kind()` to a test-only `#[cfg(test)]` seam. The accessor exists solely for DC21 localhost transport-selection tests and should not live in non-test builds. | DC21 |
| 2026-03-18 | @claude: PR #83 R3 — reconcile DESIGN/PLAN/API docs with current implementation: OutputBus sync `&self` signatures, `PipeHandle` replaces bare `JoinHandle`, `source_key()` + `target_string()` dual accessors, `SourceLabel` pane_id format (`build(%5)`), `format()` always labels (not only on transitions), `StdioSink` Prefixed uses `source_key`. | OutputBus, Subscription, TargetOutput, SourceLabel, JoinedStream, StdioSink |
| 2026-03-18 | @claude: PR #83 R1 — narrowed OutputBus shutdown/unsubscribe lifecycle docs to match fire-and-forget impl (piped tasks drain on their own; caller holds PipeHandle for join semantics). | OutputBus, SinkKind |
| 2026-03-17 | @claude: DC24 R3 — address PR #82 review round 3: remove `filters` from `CallbackSink` and `filters()` from `SinkKind` — routing is `subscribe(filters)` only, sinks are terminal consumers. Update DC12 rationale, ActionHandle section, JoinedStream intro. | SinkKind, CallbackSink, DC12, ActionHandle, JoinedStream |
| 2026-03-17 | @claude: DC24 R2 — address PR #82 review round 2: fix stale sections still describing old monitor-side rule evaluation and direct sink registration. Simplify `SinkFilter` to routing-only (remove `content`/`MatcherKind`/`MatcherInput`). Rewrite integration diagram to show `subscribe(filters) -> Subscription` + adapters. Remove `rules` param from `SessionMonitor::run()`. Update Phase 2b/2c narratives. | SinkFilter, Integration, monitor.rs, Phase 2b/2c |
| 2026-03-17 | @claude: DC24 R1 — address PR #82 review: unify subscription API into `subscribe() -> Subscription` with composable adapters (`.joined()`, `.pipe()`, `.filter()`, `.react()`). Rules/reactors are subscription consumers, not monitor-internal. `StreamChunk` gains `source_changed` flag. Eliminate dual-matching (DC14 updated). | DC24, DC15, DC14, OutputBus, Subscription, JoinedStream |
| 2026-03-17 | @claude: DC24 — evaluate Unix-pipe pipeline model vs pub/sub OutputBus. Retain pub/sub for fan-out; extract `JoinedStream` from `SinkKind` into independent consumer-side stream combinator. Remove `subscribe_joined()` from OutputBus. Revise DC15 accordingly. Restructure Phase 2 PLAN into Track A (streaming + combining) / Track B (matching + reactors). | DC24, DC15, JoinedStream, OutputBus, Phase 2 |
| 2026-03-17 | @claude: Address PR #80 R2 — fix DC10 decision sentence to say "sole" not "primary with fallback". | DC10 |
| 2026-03-17 | @claude: Address PR #80 R1 — remove all live pipe-pane/FIFO/fallback references from active architecture sections (overview, API signatures, module specs, DC1, DC6, DC7, DC13, OC1–OC4, Phase 2a/4). Remaining references are in historical context only (prototype, problem table, DC10 comparison table, struck-out sections). | Overview, Core Abstractions, Module Specs, DC1, DC6, DC7, DC13, OC1, OC2, OC4, Phase 2a, Phase 4 |
| 2026-03-16 | @claude: Mark pipe-pane fallback (DC10) as out of scope. tmux 3.1+ baseline (DC22) guarantees control mode; pipe-pane/PipeManager/FIFO machinery adds complexity with no benefit. Historical references retained as context. | DC10 |
| 2026-03-15 | @codex: Address PR #78 final doc follow-up by locking DC23 follow-on decisions: reject symlinks initially, do not preserve metadata, and keep the first public API at `Result<()>`. | DC23 |
| 2026-03-14 | @codex: Address PR #78 re-review by clarifying DC23 path typing (`&Path` for local and remote endpoints) and `cp -r` style directory placement semantics (existing directory => copy into; missing path => copy as). | DC23 |
| 2026-03-14 | @codex: Address PR #78 review by clarifying DC23 test policy (localhost SFTP tests are not tmux-gated) and directory overwrite semantics (`overwrite=true` merges into an existing destination tree rather than replacing it). | DC23 |
| 2026-03-14 | @codex: Refine DC23 per user decisions — greenfield/breaking changes accepted, API uses `upload` / `download`, overwrite semantics configurable, directory transfer included now. | DC23 |
| 2026-03-14 | @codex: Add DC23 summary for host-level SFTP file transfer, scoped as a transport/host capability (not a tmux-target capability), with companion deep-dive doc `SFTP.md`. | Overview, DC23, References |
| 2026-03-14 | @claude: DC22 — `CreateSessionOptions` for window size and history limit on session creation. Option (b): per-session + per-pane `set-option` after create (tmux 3.1+). Migration/backwards compatibility explicitly out of scope per user direction. | HostHandle, control.rs, DC22 |
| 2026-03-13 | @claude: DC21 R5 — address PR #71 R4: scope `transport_kind()` to `pub(crate)` (not public API), define reject semantics for duplicate params within same location. | DC21 |
| 2026-03-13 | @claude: DC21 R4 — address PR #71 R3: wrap raw transports in `TransportKind::Local/Ssh` in pseudocode (match `HostHandle::new` constructor). | DC21 |
| 2026-03-12 | @claude: DC21 R3 — address PR #71 R2: fix `connect(self)` in API signature block, add `/socket-path` vs `socket-name` mutual-exclusion rule, fix fleet example to use URI string instead of undeclared `HostHandle` `Display`. R1 fixes: `connect()` ownership (`&self`→`self`), `HostHandle::transport_kind()` inspection seam. | DC21 |
| 2026-03-12 | @claude: DC21 R2 — address feedback: consolidate `SshUri` into `SshConfig` (no new type), support both nassh `;` and query `?` param syntax, no canonical-component duplication (port/user/host only from URI structure). `SshConfig` gains `parse()`, `connect()`, `Display`/`FromStr`. URI parsing in `src/uri.rs` as `impl SshConfig` extension. | DC21 |
| 2026-03-12 | @claude: DC21 — Unified SSH URI for Host Addressing. Single `ssh://` URI scheme for all hosts (localhost and remote), transport auto-selection (localhost→Local, else→SSH). | DC21 |
| 2026-03-11 | @claude: Sync DESIGN.md signatures with Phase 1.10 implementation — `TargetSpec::pane()` returns `Result<Self>`, `Target::rename()` returns `Result<Target>`, `TransportKind::open_shell()` takes `cols, rows` params. Per PR #69 review. | TargetSpec, Target, TransportKind |
| 2026-03-10 | @claude: Add inline note in DC20 geometry/reflow contract clarifying that Phase 1.9b implements snapshot comparison for capture/sample_text only; exec() deferred to Phase 2a. See PR #66 review round 2. | DC20 |
| 2026-03-10 | @codex: address PR #65 review feedback by tightening the capture/sink contract: preserve-fidelity payloads by default, make TUI workflows explicitly `Raw`, move exec stability to an internal parser view, replace synthetic `TargetOutput.kind=Gap` with `SinkEvent::Gap`, scope history-limit handling to setup-time/new panes, and split Phase `1.9` into `1.9a`/`1.9b`. | ScrollbackQuery, Output Sink Pipeline, capture.rs, DC20, Implementation Phases |
| 2026-03-10 | Prioritize consumer usability for mixed-client fidelity: slot implementation as `1.9` (capture/result fidelity) → `2a.2` (monitor stream assembly) → `2c.1/2c.3/2c.4` (sink metadata + no-silent-drop delivery). | DC20, TargetOutput, OutputBus, monitor.rs, Implementation Phases |
| 2026-03-10 | Revise mixed-client capture design for TUI fidelity: no ANSI stripping in screen-stability paths, geometry/reflow detection via tmux client/pane metadata, dynamic history-limit floor management, and overlap-aware sampling. | ScrollbackQuery, capture.rs, DC20 |
| 2026-03-10 | Add explicit reference to companion `TUI.md` for TUI-specific reliability/fidelity policy. Keep implementation planning in `PLAN.md` unchanged to avoid blocking core delivery. | Overview, DC20, Phase 5, References |
| 2026-03-10 | Add mixed-client screen-size resilience proposal: capture/output normalization and fixed-size automation guidance for `capture()`, `sample_text()`, and `exec()` reliability under client resize/reflow. | DC20, DC19, Open Concerns, Implementation Phases |
| 2026-03-08 | `Target::exec()` shell compatibility: document `$?` (POSIX) vs `$status` (fish) with shell detection. Replace `ActionTarget` enum with `TargetAddress` in `ActionRequest` for unified type consistency (DC16). | DC19, Core Abstractions, Output Sink Pipeline |
| 2026-03-08 | Address codex review round 8: clarify remaining dynamic types (`Arc<dyn Any>`, `Pin<Box<dyn Future>>`) in SinkKind docs and changelog; fix integration diagram to use `SinkKind` wrappers. | Core Abstractions, Output Sink Pipeline |
| 2026-03-08 | Address codex review round 7: `CallbackSink` uses explicit `Arc<dyn Any>` state instead of closure capture, `on_output` is synchronous; fix stale examples (`SubstringMatcher` → `MatcherKind::Substring`, `JoinedSink` uses `SinkKind`); align DC6/OC6/Phase 1 to `TransportKind` enum; fix `host.rs` module spec `session_monitors` to `RwLock`; fix "SinkKind trait" → enum; fix DC14 matcher names. | Core Abstractions, Output Sink Pipeline, Module Specs, DC6, DC14, OC6, Phase 1, Phase 2c |
| 2026-03-08 | Phase 3 CLI: noun-verb subcommand pattern (`session list`, `target capture`, `monitor start`) replacing flat hyphenated commands. `target` noun reflects unified Target type (DC16). | Phase 3 |
| 2026-03-08 | Explicit FIFO cleanup on monitoring stop: `SessionMonitorHandle::shutdown()` and `stop_monitoring_session()` call `PipeManager::cleanup()` when pipe-pane fallback is active (P4). | Core Abstractions, Module Specs, DC13 |
| 2026-03-08 | Static dispatch on hot paths: `Transport` → `TransportKind` enum, `ContentMatcher` → `MatcherKind` enum, `OutputSink` → `SinkKind` enum, `LabelFormat::Custom` → `fn` pointer. Remaining dynamic types: `CallbackSink.state` (`Arc<dyn Any>`), `on_flush` (`Pin<Box<dyn Future>>` — Rust async limitation). | Core Abstractions, Output Sink Pipeline, Module Specs, DC14 |
| 2026-03-08 | Added `Target::exec()` for structured command execution with sentinel-based capture (DC19). `ExecOutput` type with stdout and exit code. | Core Abstractions, DC19 |
| 2026-03-08 | Added the original `Fleet` named-target APIs: `bind()`, `find()`, `workstreams()` for named (host, target) bindings. These names were later replaced by target-alias APIs so application terms remain outside `libs/tmux`. | Core Abstractions |
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

Product prioritization notes live in [`docs/PRODUCT.md`](../../../docs/PRODUCT.md). The current
product comparison against `tmux-mcp-rs` does not change the tmux-only scope, but it does
clarify two robustness-oriented follow-ons for the foundation:

- dedicated socket-isolation ergonomics
- tracked command execution as a complement to await-to-completion `Target::exec()`

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
8. Names target aliases across hosts and targets for domain-meaningful addressing
9. Attaches output pipes for continuous monitoring
10. Combines multi-source output into conversation/transcript-friendly history
11. Exposes composable stream/history adapters for external analysis
12. Routes control actions back to the correct host/target when an external agent decides to act

### Scope

- **In scope**: Localhost tmux (direct execution), SSH transport for remote hosts,
  multi-host connection pool, tmux session creation and termination, session/window/pane
  listing, pane content capture, structured command execution (exec with exit code),
  remote input with escaping, host-level file transfer (local filesystem / SSH SFTP),
  session metadata management, target aliases, stream/history-oriented output
  sink pipeline, control-mode monitoring, external-agent-friendly stream composition,
  structured logging, CLI binary
- **Out of scope**: Web UI, SSH server setup/configuration, tmux installation
- **Out of active design path**: Built-in matcher DSL, declarative trigger-rule engine,
  internal reactor/action pipeline, and reconnecting rule processors. These are preserved
  in the appendix as historical context, not deleted from project memory.
- **Deferred but still active infrastructure concern**: SSH reconnection / long-lived
  host reliability remains in scope as transport/Fleet hardening. It is no longer
  coupled to a built-in rule engine or config-driven automator design.
- **Future**: TUI interface based on [ratatui](https://ratatui.rs/) (not in current phases)
  with reliability and capture-fidelity guidance in [`TUI.md`](./TUI.md). SFTP
  design deep dive lives in [`SFTP.md`](./SFTP.md).

---

## Prototype Reference

The prototype is a single-file ~150-line Rust program generated via a Gemini conversation.
It demonstrates the core mechanic and is the starting point for this design. The full
prototype source, dependencies, and origin are preserved below for traceability.

**Origin**: [Gemini conversation — Rust SSH Tmux Session Interaction](https://g.co/gemini/share/e7eb11c45954)

### Prototype Cargo.toml

<!-- @claude 2026-04-09: Prototype code below predates the anyhow→thiserror migration.
     The library now uses a typed Error enum (thiserror); anyhow is dev-dependency only. -->

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
russh = "0.40"
russh-keys = "0.40"
anyhow = "1.0"  # migrated to thiserror in PR #145
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
│   ├── config.rs           # Historical appendix direction only — built-in automation config is deferred
│   ├── transport.rs        # TransportKind enum + LocalTransport + SshTransport
│   ├── host.rs             # HostHandle: per-host facade for all tmux operations
│   ├── fleet.rs            # Fleet: multi-host registry, aggregation, target-alias routing
│   ├── discovery.rs        # Session/window/pane listing, filter, PaneAddress type
│   ├── capture.rs          # Pane content capture (capture-pane) and scrollback dump
│   ├── control.rs          # Session lifecycle, send-keys with escaping, rename
│   ├── pipe.rs             # OUT OF SCOPE — pipe-pane fallback removed (tmux 3.1+ baseline)
│   ├── monitor.rs          # OutputMonitor: stream parsing, publish to OutputBus
│   ├── matcher.rs          # Historical appendix direction only — matcher DSL is deferred
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
         └── Monitor ────── stream parsing → publish to OutputBus
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
- **Monitoring**: Dedicated channel — control mode session (`tmux -C attach`, DC10)
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

The top-level coordination layer. Manages connections to localhost and/or multiple remote
hosts, provides stable names for targets, aggregates monitoring streams, and routes
control actions back to the right host/target when an external agent decides to act.

<!-- @claude 2026-03-20 — updated to match shipped API (PR #92 R2). -->

```rust
pub struct Fleet { /* host registry, shared output bus, target aliases */ }

pub struct FleetTargetSpec { /* host alias + TargetSpec */ }

pub struct ResolvedFleetTarget {
    pub spec: FleetTargetSpec,
    pub host: HostHandle,
    pub target: Target,
}

pub struct FleetSnapshotOptions {
    pub hosts: Option<Vec<String>>,
    pub tag_prefixes: Vec<String>,
}

pub struct FleetSessionSnapshot {
    pub host_alias: String,
    pub session: SessionInfo,
    pub target: FleetTargetSpec,
    pub tags: BTreeMap<String, Vec<SessionTag>>,
}

impl Fleet {
    /// Create an empty fleet with a fresh shared OutputBus.
    pub fn new() -> Self;

    /// Register a host by alias. The fleet alias must match host.host_alias()
    /// so output labels and routing names stay consistent. Injects the fleet's
    /// shared OutputBus into the host (via inject_output_bus) so all monitors
    /// publish to a single aggregation bus.
    pub fn register(&mut self, alias: &str, host: HostHandle) -> Result<()>;

    /// Remove a host plus Fleet-owned monitor and target-alias bookkeeping.
    pub fn unregister(&mut self, alias: &str) -> Result<HostHandle>;

    /// Look up a host by alias.
    pub fn host(&self, name: &str) -> Option<&HostHandle>;

    /// Iterate over all registered (alias, HostHandle) pairs.
    pub fn hosts(&self) -> impl Iterator<Item = (&str, &HostHandle)>;

    /// The shared OutputBus aggregating output from all registered hosts.
    pub fn output_bus(&self) -> Arc<OutputBus>;

    /// Host status: Connected, Monitoring { sessions }, Error(String).
    pub fn host_status(&self, alias: &str) -> Option<HostStatus>;

    /// Resolve a cross-host target spec.
    pub async fn resolve_target(&self, spec: &FleetTargetSpec) -> Result<Option<ResolvedFleetTarget>>;
    pub async fn require_target(&self, spec: &FleetTargetSpec) -> Result<ResolvedFleetTarget>;
    pub async fn resolve_targets<I>(&self, specs: I) -> Result<Vec<ResolvedFleetTarget>>
    where
        I: IntoIterator<Item = FleetTargetSpec>;

    // --- Monitoring lifecycle ---

    /// Start monitoring a specific session on a host.
    pub async fn start_monitoring_session(&mut self, alias: &str, session: &str) -> Result<()>;

    /// Start monitoring all sessions on a host.
    pub async fn start_monitoring_host(&mut self, alias: &str) -> Result<()>;

    /// Start/stop monitoring the session containing a target.
    /// Repeated calls are idempotent.
    pub async fn start_monitoring_target(&mut self, spec: &FleetTargetSpec) -> Result<()>;
    pub fn stop_monitoring_target(&mut self, spec: &FleetTargetSpec) -> Result<()>;
    pub async fn ensure_monitoring_session(&mut self, spec: &FleetTargetSpec) -> Result<SessionMonitorStatus>;
    pub async fn ensure_monitoring_sessions<I>(&mut self, specs: I) -> Result<Vec<SessionMonitorStatus>>
    where
        I: IntoIterator<Item = FleetTargetSpec>;
    pub fn stop_monitoring_session_target(&mut self, spec: &FleetTargetSpec) -> Result<()>;
    pub fn monitor_status_for_target(&self, spec: &FleetTargetSpec) -> Option<SessionMonitorStatus>;

    /// Stop all monitoring on a host.
    pub fn stop_monitoring_host(&mut self, alias: &str) -> Result<()>;

    /// Shutdown: stop all monitoring, close the bus.
    pub fn shutdown(&mut self);

    // --- Inventory ---

    pub async fn list_sessions_by_host(&self) -> Result<BTreeMap<String, Vec<SessionInfo>>>;
    pub async fn list_sessions_with_tags(&self, tag_prefix: &str) -> Result<Vec<FleetSessionInfo>>;
    pub async fn snapshot_sessions(&self, opts: FleetSnapshotOptions) -> Result<Vec<FleetSessionSnapshot>>;

    // --- Target aliases (named bindings) ---

    pub fn bind_target_alias(&mut self, name: &str, target: FleetTargetSpec) -> Result<()>;
    pub fn unbind_target_alias(&mut self, name: &str) -> Result<()>;
    pub async fn resolve_target_alias(&self, name: &str) -> Result<Option<ResolvedFleetTarget>>;
    pub async fn require_target_alias(&self, name: &str) -> Result<ResolvedFleetTarget>;
    pub fn target_aliases(&self) -> impl Iterator<Item = &str>;

    // --- Timeline helpers ---

    pub fn timeline_options_for_targets(
        &self,
        targets: &[ResolvedFleetTarget],
        base: TimelineOptions,
    ) -> TimelineOptions;

    /// Convenience routing helpers for external agents.
    pub async fn send_text(&self, name: &str, text: &str) -> Result<()>;
    pub async fn send_keys(&self, name: &str, keys: &KeySequence) -> Result<()>;
    pub async fn capture(&self, name: &str) -> Result<String>;
    pub async fn target(&self, name: &str) -> Result<Target>;
    pub async fn send_text_to(&self, spec: &FleetTargetSpec, text: &str) -> Result<()>;
    pub async fn capture_target(&self, spec: &FleetTargetSpec) -> Result<String>;
}
```

**Design rationale**: The active design assumes policy lives outside Motlie. An external
LLM/classifier or other controller consumes monitoring output, decides what to do, and
then uses Motlie to act. `Fleet` therefore focuses on three coordination jobs:

1. **Connection registry**: keep `HostHandle`s by stable alias.
2. **Stream aggregation**: expose one cross-host `OutputBus` / subscription seam.
3. **Action routing**: resolve a target alias or `FleetTargetSpec` back to the
   correct `HostHandle` and `Target`.

**Target aliases** give callers a stable vocabulary that decouples caller intent from
tmux addressing. Instead of repeatedly resolving `fleet.host("web-1")?.target(...)`, a
caller can bind `"primary"` to `FleetTargetSpec::new("web-1", TargetSpec::session("build"))?`.
Fleet registration requires `alias == host.host_alias()` so monitor labels and routing
names are consistent; `SshConfig::connect_with_alias(alias)` is the canonical way for
higher-level tools to use stable routing names like `"local"` or `"amd2"` when the
transport URI host differs.
The only naming layer in `Fleet` is the target-alias registry; domain-specific workflow
concepts stay outside `libs/tmux`.

**Session inventory + tags** provide generic discovery for higher-level selectors and
dashboards. `Fleet::list_sessions_with_tags(prefix)` enumerates all registered hosts,
batch-reads tags under a caller-owned prefix on each host, and returns `FleetSessionInfo`
records. The library validates prefix/key/value syntax but does not interpret tag names
or values.

**Timeline helpers** are intentionally narrow. `ResolvedFleetTarget::sink_filter()` and
`Fleet::timeline_options_for_targets()` produce generic bus filters for each target's
host/session. Storage, retention, rendering, and marker persistence remain owned by the
existing output/history pipeline or external consumers.

**Usage**:

```rust
let host = fleet.host("web-1").unwrap();
let opts = CreateSessionOptions { command: Some("cargo build".to_string()), ..Default::default() };
host.create_session("build", &opts).await?;
let build = FleetTargetSpec::session("web-1", "build")?;
fleet.bind_target_alias("build-pipeline", build.clone())?;

// Later — anywhere in the codebase
let target = fleet.target("build-pipeline").await?;
target.send_text("cargo test").await?;

// List all target aliases
for alias in fleet.target_aliases() {
    println!("{alias}");
}
```

**Future: target alias groups**. Fleet aliases intentionally bind one name to one
target today. A possible future extension is grouping multiple targets under one
alias, e.g. "deploy" spanning `web-1:build`, `db-1:migrate`, and `web-1:test`.
That grouping belongs above the current generic target-alias API and is deferred
until real usage clarifies membership lifecycle and monitoring semantics.

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
    /// Per-pane exec locks keyed by stable identity (DC19).
    exec_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
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
    /// Runs: tmux new-session -d -s <name> [-n <window_name>] [-x W -y H] [<command>]
    /// If `opts.history_limit` is set, also runs set-option -t and set-option -p (DC22).
    pub async fn create_session(
        &self,
        name: &str,
        opts: &CreateSessionOptions,
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
    ) -> Result<MonitorHandle>;

    /// Stop all monitoring on this host.
    pub async fn stop_monitoring(&self) -> Result<()>;

    /// Start monitoring a single session (DC13 session-level).
    /// Accepts a session-level Target. Opens one control-mode connection and
    /// does not return until the monitor has parsed its first `%output` frame,
    /// so callers do not need sleep-based readiness workarounds.
    pub async fn start_monitoring_session(
        &self,
        target: &Target,
    ) -> Result<SessionMonitorHandle>;

    /// Stop monitoring a single session.
    /// Calls SessionMonitorHandle::shutdown() — tears down the control-mode
    /// connection, flushes pending output, and joins the monitor task.
    /// Does not affect other sessions.
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
    /// Returns `Err` if `.window()` was not called first.
    pub fn pane(self, index: u32) -> Result<Self>;

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

    /// Full session info (only available at session level).
    /// For cross-level session name access, use `session_name()`.
    pub fn session_info(&self) -> Option<&SessionInfo>;

    /// Full window info (only available at window level).
    /// Pane targets carry window index via `pane_address().window` but not
    /// full `WindowInfo`.
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

    // --- Creation (hierarchy growth) ---

    /// Create a new window in this session and return it as a Target.
    /// Session-level only — returns an error for window/pane targets.
    pub async fn new_window(
        &self,
        opts: &CreateWindowOptions,
    ) -> Result<Target>;

    /// Split a pane and return the newly created pane Target.
    /// Window-level: splits the active pane in that window.
    /// Pane-level: splits this pane explicitly.
    /// Session-level: returns an error instead of implicitly targeting the active pane.
    pub async fn split_pane(
        &self,
        opts: &SplitPaneOptions,
    ) -> Result<Target>;

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

    /// Capture with explicit fidelity options and metadata.
    /// Preferred for automation that needs to detect degraded output.
    pub async fn capture_result(
        &self,
        options: &CaptureOptions,
    ) -> Result<CaptureResult>;

    /// Capture with scrollback history. `start` is negative for scrollback lines
    /// (e.g., -100 = 100 lines above visible area). Captures through end of visible area.
    /// Note: tmux `-E` with negative values counts from scrollback buffer start, not
    /// visible area end, making `-S -N -E -1` semantics unreliable. Using `-S` only.
    pub async fn capture_with_history(&self, start: i32) -> Result<String>;

    /// Sample recent scrollback, returned in chronological order.
    pub async fn sample_text(&self, query: &ScrollbackQuery) -> Result<String>;

    /// Sample recent scrollback with explicit fidelity options and metadata.
    pub async fn sample_text_result(
        &self,
        query: &ScrollbackQuery,
        options: &CaptureOptions,
    ) -> Result<CaptureResult>;

    // --- Lifecycle ---

    /// Kill this entity. Session: kills session. Window: closes window.
    /// Pane: closes pane.
    pub async fn kill(&self) -> Result<()>;

    /// Rename this entity and return a new `Target` with updated address.
    /// Session: rename session (must use returned handle). Window: rename window.
    /// Pane: not supported (returns error).
    pub async fn rename(&self, new_name: &str) -> Result<Target>;

    /// Capture all panes under this target as a map.
    /// Session: all panes in all windows. Window: all panes in window.
    /// Pane: single-entry map.
    pub async fn capture_all(&self) -> Result<HashMap<PaneAddress, String>>;

    /// Capture all panes under this target with explicit fidelity options and metadata.
    pub async fn capture_all_result(
        &self,
        options: &CaptureOptions,
    ) -> Result<HashMap<PaneAddress, CaptureResult>>;

    // --- Monitoring ---

    /// Start monitoring this target (session level only — DC10).
    /// Returns an error if called on a window or pane target.
    pub async fn start_monitoring(&self) -> Result<SessionMonitorHandle>;

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
let opts = CreateSessionOptions { command: Some("cargo build".to_string()), ..Default::default() };
let build = host.create_session("build", &opts).await?;
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
let spec = TargetSpec::session("build").window(0).pane(1)?;
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
the hood, with optional `-e` in fidelity-preserving modes. `LastLines(n)` maps
directly to `-S -n`. The `Until` and `LastLinesUntil` variants capture `max_lines`
(or `lines`) of scrollback in one call, then scan the result in reverse for the
pattern match, truncating at the match point. The final output is returned in
chronological order — no reversal needed since `capture-pane` already outputs
top-to-bottom.

**Use cases**:
- `LastLines(50)` — "show me the last 50 lines" for a quick status check
- `Until { pattern: r"^\$\s*$", max_lines: 5000 }` — "everything since the last
  shell prompt" for capturing a command's full output
- `LastLinesUntil { lines: 200, stop_pattern: r"^error:" }` — "last 200 lines, but
  stop if we hit an error marker" for focused error context

**Overlap-aware incremental sampling (Phase `1.9b`)**: For long-running panes,
`sample_text()` may later support incremental polling with overlap to avoid gaps at
chunk boundaries, but this is intentionally not part of the `1.9a` gate. The
algorithm is:

1. Track `history_size` and pane geometry per target between polls.
2. Compute `delta_lines = max(0, current_history_size - previous_history_size)`.
3. Capture `delta_lines + overlap_lines` from the tail (bounded by the requested window).
4. Attempt a unique, byte-exact suffix/prefix match after newline canonicalization only.
   Require at least two complete overlapping lines; otherwise skip de-duplication.
5. If repeated lines make the match ambiguous, or no qualifying match exists, emit
   `OverlapResync` and fall back to a wider recapture window instead of guessing.
6. If history shrinks or pane geometry changed, mark the result degraded and resync from
   a wider recapture window.

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
    /// Returns argument vectors — each inner `Vec<String>` is one send-keys
    /// argument list (e.g. `["send-keys", "-l", "-t", target, text]`).
    /// `control::send_keys` assembles these into full shell command strings.
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
    pub id: SessionId,          // non-empty tmux internal $N id
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

`SessionId` is parsed from tmux `#{session_id}` and rejects empty values at the
discovery boundary. Callers that need a stable dispatch key use
`session.id.as_str()`; falling back to display names for id-based dispatch is
not part of the contract.

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
pub enum FidelityIssue {
    ClientResize,
    PaneResize,
    HistoryTruncated,
    OverlapResync,
}

pub struct OutputFidelity {
    pub degraded: bool,
    /// `None` is the hot-path clean case: no heap allocation for issue storage.
    pub issues: Option<Vec<FidelityIssue>>,
}

pub enum SinkEvent {
    Data(TargetOutput),
    /// Per-sink backpressure marker emitted before the next successfully delivered
    /// data event. This does not consume a source sequence number.
    Gap {
        dropped: usize,
        timestamp: Instant,
    },
}

pub struct TargetOutput {
    /// The source entity. Carries the full TargetAddress so sinks can
    /// identify the source at session, window, or pane granularity.
    /// For control-mode output (DC10), this is always pane-level since
    /// `%output` includes a pane ID. For session-only hosts (single
    /// default window/pane), callers can match at session level and
    /// ignore the window/pane detail.
    pub source: TargetAddress,
    pub host: String,          // host alias (or "localhost")
    /// Canonical delivery view for the selected mode. `ScreenStable` preserves
    /// ANSI/control sequences (via `-ep`); `Raw` returns tmux-rendered text
    /// (no escape sequences, via `-p`); `PlainText` returns normalized text.
    pub content: String,
    /// Exact tmux capture before mode-specific normalization, when requested.
    pub raw_content: Option<String>,
    /// Monotonic per-source sequence for gap detection and ordering checks.
    pub sequence: u64,
    pub fidelity: OutputFidelity,
    pub timestamp: Instant,
}

impl TargetOutput {
    /// Session name — available at any source level.
    pub fn session_name(&self) -> &str;

    /// Pane ID — available when source is pane-level, None otherwise.
    pub fn pane_id(&self) -> Option<&str>;

    /// Canonical identity for bus routing, filter matching, and coalescing.
    /// Returns pane_id for pane-level sources, session name for sessions.
    pub fn source_key(&self) -> String;

    /// Display format: tmux target string (e.g., "build", "build:0", "build:0.1").
    /// May be synthetic for monitor-originated output (window/pane indices are 0).
    pub fn target_string(&self) -> String;

    /// True when fidelity was degraded (reflow/history/drop/resync events).
    pub fn degraded(&self) -> bool;
}
```

`TargetOutput` is the bridge between the monitor/capture side and the sink side.
The `source` field (a `TargetAddress`) enables sinks to filter and match at any
level of the hierarchy — session, window, or pane — and to route actions back to the
originating entity. Hosts with a single session/window/pane can be addressed at
session level without requiring callers to know the pane ID.

Usability contract for downstream consumers (stdio, LLM, classifier, triggers):
1. `content` is the canonical delivery stream for the selected mode.
2. `raw_content` is an optional fidelity sidecar containing the exact tmux capture.
3. Source-side degradation is explicit in `fidelity`; sink-route drops are explicit in
   `SinkEvent::Gap` and are never folded into source metadata.

### Historical Note — `MatcherKind` / Built-In Matching

The earlier Track B direction introduced a first-class `MatcherKind` DSL plus
subscription-side `.filter()` / `.react()` adapters for in-library automation.
That design is no longer on the active path. The active direction treats
`libs/tmux` as a tmux stream/history/control substrate for external policy engines,
so matching and action selection are expected to live in the external consumer.

The matcher/rule/reactor design is preserved in [Appendix A](#appendix-a--historical-automation-direction)
for historical context and possible future revival, but it should not drive the
active API or implementation plan.

### `SinkFilter` — Source Routing

<!-- @claude 2026-03-17: Simplified to routing-only per DC24. Content matching
     moved out of the bus layer. 2026-03-20 @codex: active path now favors
     transcript/history adapters and consumer-owned predicates over a built-in matcher DSL. -->

A `SinkFilter` selects which output reaches a given `Subscription` by source identity.
Routing fields target host, session, window, and pane. Content matching is **not** part
of `SinkFilter` — the active path keeps the bus layer fast and stateless, and leaves
transcript construction, consumer predicates, and policy evaluation above the bus.

Multiple filters are combined with OR semantics — output matching **any** filter in the
set is delivered.

```rust
pub struct SinkFilter {
    pub host: Option<String>,      // regex against host alias
    pub session: Option<String>,   // regex against session name
    pub window: Option<String>,    // regex against "session:window_index"
    pub pane: Option<String>,      // regex against pane_id or "session:window.pane"
}

/// Compiled form — routing regexes compiled once at subscribe() time.
pub struct CompiledSinkFilter {
    pub host: Option<Regex>,
    pub session: Option<Regex>,
    pub window: Option<Regex>,
    pub pane: Option<Regex>,
}

impl CompiledSinkFilter {
    /// Returns true if the output matches ALL non-None fields in this filter.
    /// Fields that are None are wildcards (match everything).
    pub fn matches(&self, output: &TargetOutput) -> bool;
}
```

**Note**: `matches()` takes `&self` — routing is stateless.

**Combining filters**: `subscribe()` accepts `Vec<SinkFilter>`. Output is delivered
if it matches **any** filter in the vec (OR across filters, AND within each filter's
routing fields). An empty vec means "match all output" (the default).

**Examples**:

```rust
// Subscription receives output from all panes on "db-server"
vec![SinkFilter { host: Some("db-server".into()), ..Default::default() }]

// Subscription receives output from session "build" on any host, OR any session on "web-1"
vec![
    SinkFilter { session: Some("build".into()), ..Default::default() },
    SinkFilter { host: Some("web-1".into()), ..Default::default() },
]

// Subscription receives output from a specific pane
vec![SinkFilter { pane: Some("%42".into()), ..Default::default() }]
```

Content-based filtering examples use `Subscription` adapters — see
[DC24 Subscription composable seam](#dc24-subscription-composable-seam).

### `SinkKind` — Static-Dispatch Terminal Consumers

<!-- @claude 2026-03-17: Sinks are terminal consumers only — no routing. Routing is
     owned by subscribe(filters) -> Subscription. Sinks are attached via .pipe(). -->

Every output consumer is a variant of the `SinkKind` enum. Sinks are **terminal
consumers** — they process events but do not own routing filters. Routing is the
responsibility of `OutputBus::subscribe(filters, capacity) -> Subscription`; sinks
are attached to a subscription via `.pipe(SinkKind)`. The bus dispatches via `match`
with zero vtable indirection on the hot path.

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
    /// Shared state passed to callbacks. Consumers put their accumulated
    /// buffers, connections, or other mutable state here.
    pub state: Arc<dyn Any + Send + Sync>,
    /// Synchronous callback for each output event. Receives shared state
    /// and the output. For I/O-heavy sinks, queue work internally and
    /// flush asynchronously via on_flush.
    pub on_output: fn(state: &Arc<dyn Any + Send + Sync>, event: SinkEvent) -> Result<()>,
    /// Called on bus shutdown. Flush internal buffers, close resources.
    /// Returns a boxed future — the only remaining async indirection,
    /// unavoidable without async fn in fn pointers (not yet stable in Rust).
    pub on_flush: Option<fn(state: &Arc<dyn Any + Send + Sync>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>>,
}

impl SinkKind {
    /// Human-readable name for logging and diagnostics.
    pub fn name(&self) -> &str;

    /// Process one sink event. Called from the sink's own task —
    /// never from the monitor/bus hot path.
    pub async fn write(&mut self, event: SinkEvent) -> Result<()>;

    /// Called on bus shutdown. Flush internal buffers, close resources.
    pub async fn flush(&mut self) -> Result<()>;
}
```

**Key design principle**: Each sink owns its batching/accumulation strategy internally.
The subscription's `.pipe()` adapter forwards `SinkEvent`s to the sink; the sink
decides whether to process them immediately (stdio), buffer and render at frame rate
(TUI), or accumulate and flush on a timer/threshold (LLM). Routing is upstream in
`subscribe(filters)` — the sink sees only the events its subscription selected.

**Extension via `CallbackSink`**: Consumers that need custom sink behavior (LLM
inference, webhook delivery, custom TUI) provide function pointers plus an explicit
`Arc<dyn Any + Send + Sync>` state field. The `on_output` callback receives `&Arc`
to access state — no closure captures needed. This keeps the bus monomorphic.
`on_flush` returns `Pin<Box<dyn Future>>` — the only remaining async indirection,
unavoidable until Rust stabilizes async fn pointers. The framework's `on_output` dispatch path is
fully synchronous with no framework-side allocation (user callbacks may allocate
internally as needed).

### Historical Note — In-Library Reaction Handles

The earlier automation direction introduced `ActionHandle`, `ActionRequest`, and
subscription-side `.react()` so library-managed consumers could dispatch actions back
into tmux directly. That is now deferred. The active design assumes an external
LLM/classifier or other policy engine consumes stream/history output, decides what to
do, and then calls Motlie’s normal control APIs (`Fleet`, `HostHandle`, `Target`) to
route actions back to the correct host/session/pane.

The historical reaction-handle design is preserved in
[Appendix A](#appendix-a--historical-automation-direction) for context, but it should
not drive the active API or implementation plan.

### `OutputBus` — Fan-Out Dispatcher

The `OutputBus` is the central distributor. It receives `TargetOutput` from the monitor
(or from `capture_pane()` callers) and fans out to all registered subscribers.

<!-- @claude 2026-03-17: PR #82 R1 — unified subscription API. One primitive:
     subscribe(filters, capacity) -> Subscription. Adapters layered above.
     Replaces the prior inconsistency of subscribe(sink), subscribe_channel(filters),
     and subscribe_joined(filters) as separate bus methods. -->

```rust
pub struct OutputBus { /* subscribers: Vec<SubEntry> */ }

struct SubEntry {
    id: SinkId,
    name: String,
    tx: mpsc::Sender<SinkEvent>,
    filters: Vec<CompiledSinkFilter>,
    dropped_since_last_send: usize,
}

impl OutputBus {
    pub fn new() -> Self;

    /// The single subscription primitive. Returns a `Subscription` that
    /// receives source-routed `SinkEvent`s. All consumer composition
    /// (joining, filtering, piping to sinks, reacting) is layered on
    /// `Subscription` — the bus only handles fan-out and source routing.
    pub fn subscribe(
        &self,
        filters: Vec<SinkFilter>,
        channel_capacity: usize,
    ) -> Result<Subscription>;

    /// Remove a subscription by id. Drops the sender, closing the channel.
    /// Piped tasks drain remaining events and exit on their own — this method
    /// does **not** await the task. Callers hold `PipeHandle` for join semantics.
    pub fn unsubscribe(&self, id: SinkId) -> Result<()>;

    /// Fan out an event to all matching subscribers. Non-blocking.
    /// Uses try_send for the hot path. If a subscriber channel is full,
    /// the bus increments `dropped_since_last_send` and does NOT silently
    /// lose observability: on the next successful send, emit
    /// `SinkEvent::Gap { dropped }` before the next `SinkEvent::Data(output)`.
    pub fn publish(&self, output: TargetOutput);

    /// Drop all senders, closing all receiver channels.
    /// Piped tasks drain remaining events and exit on their own — this method
    /// does **not** track or await those tasks. Callers hold `PipeHandle`
    /// from `pipe()` for flush-and-join semantics.
    pub fn shutdown(&self);
}
```

### `Subscription` — Composable Consumer Seam

A `Subscription` wraps the `mpsc::Receiver<SinkEvent>` from the bus and provides
composable adapter methods. The bus handles source routing (host/session/pane
filtering); everything above that — content matching, joining, transforming,
reacting — is expressed as `Subscription` adapters. This is the composability
borrowed from Unix-pipe thinking (DC24) applied at the consumer level, not the
bus level.

```rust
pub struct Subscription {
    id: SinkId,
    rx: mpsc::Receiver<SinkEvent>,
}

impl Subscription {
    /// Access the subscription id for later unsubscribe.
    pub fn id(&self) -> SinkId;

    /// Raw receiver access. Consumes the Subscription.
    /// For consumers that want full control (TUI, test harness).
    pub fn into_receiver(self) -> mpsc::Receiver<SinkEvent>;

    // --- Track A adapters (available immediately) ---

    /// Labeled, coalesced multi-source view. Consumes the Subscription.
    /// Returns a JoinedStream that produces StreamChunks.
    pub fn joined(self, label_format: LabelFormat) -> JoinedStream;

    /// Fire-and-forget: spawn a tokio task that drives the given sink.
    /// Consumes the Subscription. Returns PipeHandle combining subscription
    /// id (for bus control) and task JoinHandle (for awaited teardown).
    pub fn pipe(self, sink: SinkKind) -> PipeHandle;

    // --- Active next-wave adapters ---

    /// Build a bounded rolling transcript/history view optimized for
    /// external-agent context windows. Consumes the Subscription and
    /// spawns an internal accumulation task.
    pub fn history(self, opts: HistoryOptions) -> HistoryHandle;

    /// Consumer-owned predicate filtering for lightweight selection without
    /// introducing a built-in matcher DSL.
    pub fn filter_fn(self, predicate: fn(&TargetOutput) -> bool) -> Subscription;
}
```

**Design principle**: The bus gives you a stream of source-routed events. Everything
else is layered above. This eliminates the need for multiple `subscribe_*` bus methods
and makes new consumer patterns (joining, transcript/history construction, lightweight
predicate filtering, piping) orthogonal to the bus itself.

**Transcript/history note**: The active direction needs a first-class transcript layer
because the most common consumer is a rolling LLM/classifier loop with a bounded context
window. The library should therefore own accumulation, source labeling, trimming, and
gap markers — and let the external agent own policy.

```rust
/// Controls how a rolling transcript/history buffer is bounded and rendered.
pub struct HistoryOptions {
    /// Keep at most this many logical entries (source bursts or gap markers).
    pub max_entries: usize,
    /// Keep rendered text under this many characters by trimming oldest entries first.
    /// Character count is used rather than token count because tokenizer choice is model-specific.
    pub max_render_chars: usize,
    /// Label style reused from JoinedStream formatting.
    pub label_format: LabelFormat,
    /// Include an omission marker when older history has been trimmed.
    pub include_omission_marker: bool,
}

/// Rolling, bounded transcript accumulator built from a Subscription.
/// Internally consumes JoinedStream-style source coalescing.
pub struct HistoryHandle { /* shared state + background task */ }

pub struct HistorySnapshot {
    pub entries: Vec<HistoryEntry>,
    pub rendered_chars: usize,
    pub omitted_entries: usize,
}

pub enum HistoryEntry {
    /// One coalesced burst from a single source, ready for transcript rendering.
    Output {
        source: SourceLabel,
        text: String,
        source_changed: bool,
    },
    /// Explicit loss marker derived from SinkEvent::Gap.
    Gap {
        dropped_events: usize,
    },
}

// @claude 2026-03-20 — updated to match shipped async API (PR #92 R2).
impl HistoryHandle {
    /// The subscription id for bus control.
    pub fn id(&self) -> SinkId;

    /// Return the current bounded transcript state for custom formatting.
    pub async fn snapshot(&self) -> HistorySnapshot;

    /// Render a prompt-ready transcript string for external LLM/classifier context.
    /// This is the primary ergonomic API for rolling-context consumers.
    pub async fn render_text(&self) -> String;

    /// Await the background accumulator task to completion and return
    /// the final snapshot, guaranteed to include all buffered events.
    pub async fn join(self) -> Result<HistorySnapshot>;
}
```

The active intended use is:
1. monitor and subscribe once
2. build a `HistoryHandle`
3. periodically call `render_text()` to obtain the latest bounded context window
4. send that text to the external LLM/classifier
5. route any resulting action back through `Fleet`, `HostHandle`, or `Target`

This keeps the common agent loop simple while avoiding any built-in decision engine.

**Backpressure**: The bus remains non-blocking via `try_send()`, with loss visibility.
When a subscriber falls behind, dropped events are counted and surfaced via explicit
`SinkEvent::Gap` markers before the next delivered `Data` event.
This preserves monitor liveness while keeping per-subscriber loss separate from
source-side fidelity metadata.

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
    /// "[host] source_key | content" — uses canonical source identity.
    Prefixed,
    /// JSON lines with both canonical key and display target.
    Json,
}

// StdioSink is handled by SinkKind::Stdio variant.
// write(): format and write to stdout immediately (no batching)
// flush(): flush stdout
```

### `JoinedStream` — Multi-Source Consolidated View

Individual subscriptions see output from their filtered sources, but each `TargetOutput` arrives
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
    /// Short form for display: "web-1:build(%5)" (pane) or "web-1:build" (session).
    /// Pane-level sources use `pane_id` as canonical identity.
    pub fn short(&self) -> String;

    /// Minimal form when host is unambiguous: "build(%5)" or "build"
    pub fn minimal(&self) -> String;
}
```

### `JoinedStream` — Multi-Source Consolidated View

<!-- @claude 2026-03-17: Revised from JoinedSink (SinkKind wrapper) to JoinedStream
     (Subscription adapter). See DC24 for the Unix-pipe analysis that motivated this. -->

`JoinedStream` is returned by `Subscription::joined()`. It merges multiple source
streams into a single time-ordered sequence where each chunk carries source
attribution and a `source_changed` flag for coalescing.

```rust
/// Returned by Subscription::joined(). Produces StreamChunks
/// with source attribution and coalescing state.
pub struct JoinedStream {
    rx: mpsc::Receiver<SinkEvent>,
    label_format: LabelFormat,
    last_source: Option<SourceLabel>,
}

pub struct StreamChunk {
    pub source: SourceLabel,
    pub output: TargetOutput,
    /// True when the source differs from the previous chunk —
    /// signals the consumer to emit a source header/separator.
    pub source_changed: bool,
}

pub enum LabelFormat {
    /// "[web-1:build(%5)] output text here"
    Bracketed,
    /// "web-1:build(%5)> output text here"
    Prompt,
    /// Caller provides a format function (plain fn pointer — no heap allocation).
    Custom(fn(&SourceLabel, &str) -> String),
}

impl JoinedStream {
    /// Receive the next labeled chunk. Returns None when the
    /// subscription closes. Coalescing is expressed via
    /// `StreamChunk::source_changed`.
    pub async fn next(&mut self) -> Option<StreamChunk>;

    /// Format a StreamChunk as a string using this stream's LabelFormat.
    /// Always applies the configured label prefix. Consumer-side convenience
    /// for interactive multi-pane views — terminal sinks (StdioSink) own
    /// their own presentation formatting at a different layer.
    pub fn format(&self, chunk: &StreamChunk) -> String;
}
```

**Coalescing**: Consecutive chunks from the same source have `source_changed: false`.
When a different source emits, the new chunk has `source_changed: true`. Consumers
decide how to render transitions — `format()` always applies the label prefix;
consumers can use `source_changed` to add separators or headers between sources.

**Use cases via Subscription adapter**:

```rust
// Labeled stdio output from build panes across hosts
let sub = bus.subscribe(build_filters, 64);
let mut joined = sub.joined(LabelFormat::Bracketed);
tokio::spawn(async move {
    while let Some(chunk) = joined.next().await {
        print!("{}", joined.format(&chunk));
    }
});
// Output:
//   [web-1:build(%0)] compiling crate foo...
//   [web-1:build(%0)] warning: unused variable
//   [db-1:migrate(%3)] Running migration 042...
//   [web-1:build(%0)] Finished dev target

// Structured data for LLM consumption
let sub2 = bus.subscribe(all_filters, 64);
let mut joined2 = sub2.joined(LabelFormat::Bracketed);
tokio::spawn(async move {
    while let Some(chunk) = joined2.next().await {
        llm_ingest(&chunk.source, &chunk.output);
    }
});

// Raw receiver for custom consumption
let sub3 = bus.subscribe(filters, 64);
let rx = sub3.into_receiver();

// Fire-and-forget sink
let sub4 = bus.subscribe(log_filters, 64);
sub4.pipe(SinkKind::Stdio(StdioSink::new(StdioFormat::Json)));
```

### Integration with Fleet and Monitor

<!-- @claude 2026-03-17: Rewritten to reflect DC24 subscribe(filters) -> Subscription
     model. Monitor is stream-only; rules are subscription consumers. -->

```
  Fleet
    │
    ├── OutputBus (owned by Fleet)
    │     │
    │     ├── subscribe(all_filters, 1024) -> Subscription
    │     │     ├── .joined("…")              // combined multi-pane view
    │     │     ├── .pipe(SinkKind::Stdio(…))  // pipe to stdio sink
    │     │     └── .history(…)                // transcript / conversation view
    │     │
    │     └── subscribe(db_filters, 256) -> Subscription
    │           └── .pipe(SinkKind::Callback(llm_sink))
    │
    ├── HostHandle (localhost)
    │     └── Monitor ──publish()──► OutputBus
    │
    └── HostHandle (remote)
          └── Monitor ──publish()──► OutputBus
```

**Monitor → Bus**: The monitor parses control-mode `%output` frames and publishes
`TargetOutput` to the bus. The monitor does **not** evaluate policy — it is purely a
stream parser.

**capture_pane() → Bus**: On-demand captures can optionally be published to the bus
via a `bus.publish()` call. This is opt-in at the call site, not automatic.

**Subscription → External policy → Fleet / HostHandle / Target**: Consumers turn
subscription output into transcript/history, analyze it externally (LLM/classifier or
other code), and then call Motlie’s normal control APIs back through `Fleet`,
`HostHandle`, or `Target`. Routing remains explicit and uses the same host/target
identities already present in `TargetOutput`.

---

## Module Specifications

### `matcher.rs`

Historical appendix direction only. The built-in matcher DSL is not on the active
implementation path. If revived later, `matcher.rs` will house that logic; see
[Appendix A](#appendix-a--historical-automation-direction).

### `config.rs`

Historical appendix direction only. The earlier built-in automation configuration
(`TmuxAutomatorConfig`, `TriggerRule`, `ReconnectPolicy`, etc.) is deferred in favor
of a programmatic `Fleet`/`HostHandle`/`Target` workflow plus external policy engines.
See [Appendix A](#appendix-a--historical-automation-direction) for the preserved shape.

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
    /// `cols` and `rows` set PTY dimensions for SSH (ignored by Local/Mock).
    /// Used by the monitor for long-running processes (control mode session).
    pub async fn open_shell(&self, cols: u32, rows: u32) -> Result<ShellChannelKind>;
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

    /// The stable pane_id for stream keying (e.g., "%12")
    pub fn id(&self) -> &str;

    /// Parse from tmux list-panes output fields.
    /// `pane_id` is the `#{pane_id}` field (e.g., "%12").
    /// `address_str` is the composite "session:window.pane" field.
    pub fn parse(pane_id: &str, address_str: &str) -> Result<Self>;
}
```

**Addresses P2**: Using `#{pane_id}` as the authoritative key (see DC1) eliminates the
need for filename encoding of session names entirely.

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
pub async fn capture_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<String>;

/// Capture with explicit fidelity options and metadata.
pub async fn capture_pane_with_options(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    options: &CaptureOptions,
) -> Result<CaptureResult>;

/// Capture with scrollback history. `start` is negative for scrollback lines
/// (e.g., -100 = 100 lines above visible area). Captures through end of visible area.
/// Runs: tmux capture-pane -p -t <target> -S <start>
/// Note: tmux `-E` with negative values counts from scrollback buffer start, not
/// visible area end, making `-S -N -E -1` semantics unreliable. Using `-S` only.
pub async fn capture_pane_history(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    start: i32,
) -> Result<String>;

/// Capture all panes in a session. Calls capture_pane for each pane found via list_panes.
/// Returns a map of pane address → visible content.
pub async fn capture_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<HashMap<PaneAddress, String>>;

/// Capture all panes in a session with explicit fidelity options and metadata.
/// Applies the same capture mode independently to each pane.
pub async fn capture_session_with_options(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
    options: &CaptureOptions,
) -> Result<HashMap<PaneAddress, CaptureResult>>;

/// Sample recent text from a pane's scrollback, returned in chronological order.
/// Delegates to capture_pane_history() internally, then applies the query's
/// pattern matching and truncation logic.
pub async fn sample_text(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    query: &ScrollbackQuery,
) -> Result<String>;

/// Sample with explicit fidelity options and metadata.
pub async fn sample_text_with_options(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    query: &ScrollbackQuery,
    options: &CaptureOptions,
) -> Result<CaptureResult>;

pub struct CaptureResult {
    pub text: String,               // primary view for the selected mode
    pub raw_text: Option<String>,   // exact tmux capture before normalization
    pub fidelity: OutputFidelity,   // degraded/reflow/history annotations
}
```

**`sample_text` implementation**:
1. `LastLines(n)` → calls `capture_pane_history(transport, socket, target, -(n as i32))`.
   Captures from scrollback through visible area end. Result is trimmed of trailing blank lines.
2. `Until { pattern, max_lines }` → calls `capture_pane_history` with
   `start = -(max_lines as i32)`. Scans the result from the bottom up for
   the first line matching `pattern`. Returns from that line (inclusive) to the end.
   If no match, returns the full captured range.
3. `LastLinesUntil { lines, stop_pattern }` → same as `LastLines(lines)`, then scans
   bottom-up for `stop_pattern`. Returns from match point (inclusive) to the end.

In all cases the output preserves `capture-pane`'s top-to-bottom ordering — no reversal
step is needed. The pattern scan is the only post-processing.

**Escaping/format mode**: The `-p` flag outputs to stdout (not to a buffer), which is
what we need over SSH exec channels. The `-e` flag is mode-dependent:
- `Raw` and `PlainText` use `capture-pane -p`.
- `ScreenStable` uses `capture-pane -ep` so the public payload can preserve terminal
  control/ANSI while an optional `raw_text` retains the exact capture.
- `exec()` may derive its own internal parser view from `ScreenStable` capture, but
  that derived view is not exposed as a public capture mode.

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

/// Create a new detached tmux session (DC22).
/// Runs: tmux new-session -d -s <name> [-n <window_name>] [-x W -y H] [<command>]
/// If history_limit set, also runs set-option -t and set-option -p.
/// Rolls back (kills session) if post-create set-option fails.
pub async fn create_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
    opts: &CreateSessionOptions,
) -> Result<()>;

/// Kill a tmux session and all its windows/panes.
/// Runs: tmux kill-session -t <name>
pub async fn kill_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
) -> Result<()>;

/// Kill a tmux window.
pub async fn kill_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<()>;

/// Kill a tmux pane.
pub async fn kill_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<()>;

// --- Input ---

/// Send a KeySequence to a target. Handles the split between literal text (-l)
/// and special keys (no -l) automatically. All values are shell-escaped.
pub async fn send_keys(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    keys: &KeySequence,
) -> Result<()>;

/// Convenience: send literal text (no special keys, no Enter appended).
/// Equivalent to: tmux send-keys -l -t <target> '<escaped_text>'
pub async fn send_text(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    text: &str,
) -> Result<()>;

/// Rename a tmux session.
/// Runs: tmux rename-session -t <current> <new>
pub async fn rename_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    current_name: &str,
    new_name: &str,
) -> Result<()>;

/// Rename a window.
/// Runs: tmux rename-window -t <session>:<index> <new_name>
pub async fn rename_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
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
- Maintains target alias registry: `HashMap<String, TargetAliasEntry>` for named cross-host target bindings

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

`HostHandle` wraps `Arc<HostHandleInner>` which holds the transport and socket.
`Target` is lightweight — holds an `Arc<HostHandleInner>` plus a `TargetAddress` enum.
Both delegate to the function-level APIs in `discovery`, `capture`, and `control`.

```rust
// Phase 1 — on-demand operations only (no monitoring, no config)
struct HostHandleInner {
    transport: TransportKind,
    socket: Option<TmuxSocket>,
    /// Per-pane exec locks keyed by stable identity (DC19).
    exec_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

// Phase 2+ additions (pipe.rs, monitor.rs):
// struct HostHandleInner {
//     transport: TransportKind,
//     socket: Option<TmuxSocket>,
//     exec_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
//     config: HostTarget,
//     session_monitors: RwLock<HashMap<String, SessionMonitorHandle>>,
// }
```


### ~~`pipe.rs`~~ — OUT OF SCOPE

<!-- @claude 2026-03-16: Entire pipe.rs module is out of scope. tmux 3.1+ baseline (DC22)
     guarantees control mode. See DC10 out-of-scope note for rationale. Original spec retained
     below as historical context only. -->

~~FIFO lifecycle and `tmux pipe-pane` management.~~

### `monitor.rs`

The core event loop. Each `SessionMonitor` owns one control-mode connection to a single
tmux session and publishes parsed output into `OutputBus`. `HostHandle` creates one
`SessionMonitor` per monitored session (DC10, DC13).

```rust
/// Monitors a single tmux session via control mode.
pub struct SessionMonitor { /* session name, stream assembly state */ }

impl SessionMonitor {
    /// Run the monitor loop for one session.
    /// Opens `tmux -C attach -t <session>` via the transport, parses
    /// `%output %<pane_id> <data>` frames and publishes TargetOutput to the
    /// OutputBus. The monitor does not evaluate rules — rule evaluation is a
    /// consumer concern via Subscription adapters (DC24).
    /// Returns when `stop` signal is received or the connection drops.
    pub async fn run(
        &mut self,
        transport: &TransportKind,
        session: &str,
        bus: &OutputBus,
        stop: watch::Receiver<bool>,
    ) -> Result<()>;
}
```

<!-- @claude 2026-03-17: Removed rules parameter and P3 (rule evaluation). Rules are
     subscription consumers, not monitor concerns. See DC24. -->

**Must address P6**: Each `SessionMonitor` uses a dedicated control-mode connection
(`tmux -C attach -t <session>`, see DC10). Action dispatch (send-keys) uses separate
exec channels routed through a bounded queue
(see [DC4](#dc4-action-dispatch-channel-strategy)).

**Must address P9**: Failed send-keys or malformed lines must be logged at `warn` level,
not silently dropped.

**Stream parsing**: With control mode (DC10), the monitor parses `%output %<pane_id> <data>`
frames — structured, unambiguous, and keyed on `#{pane_id}` per DC1.

The parser maintains per-pane assembly state to make the stream consumer-friendly:
1. Canonicalize line endings and accumulate partial frame fragments.
2. Apply the same source-side normalization/fidelity rules used by capture paths.
3. Track per-pane `sequence` counters in emitted `TargetOutput`.
4. Attach fidelity metadata (`OutputFidelity`) for reflow/resize/history instability.
5. Let `OutputBus` emit explicit `SinkEvent::Gap` markers for sink-route backpressure.

**Lifecycle**: Each `SessionMonitor::run()` is spawned as a tokio task by `HostHandle`.
The `SessionMonitorHandle` returned to the caller holds the task's `JoinHandle` and stop
channel. `HostHandle::start_monitoring_session()` blocks until the monitor has parsed
its first `%output` frame, establishing a concrete readiness contract for callers and
eliminating sleep-based startup guesses in examples/consumers. Stopping a session
monitor is non-disruptive to other sessions (DC13).

### `sink.rs`

The output sink pipeline types. See [Output Sink Pipeline](#output-sink-pipeline) for
the full API and design rationale.

This module contains:
- `TargetOutput`: the unit of output flowing through the pipeline
- `SinkKind` enum: closed set of sink types (static dispatch, no trait objects)
- `SinkFilter` / `CompiledSinkFilter`: composable output targeting
- `OutputBus`: central fan-out dispatcher with per-sink bounded channels
- `SinkId`: opaque handle for unsubscribe
- transcript/history-oriented adapters layered above `Subscription`

Historical appendix only:
- `MatcherKind`: built-in matcher DSL (deferred)
- `ActionHandle` / `ActionRequest` / `SinkAction`: in-library reaction path (deferred)

`OutputBus` is owned by `Fleet` and shared with all `HostHandle` instances via `Arc`.
Monitors publish `TargetOutput` to the bus; the bus wraps them in `SinkEvent::Data`
and fans out to per-sink tokio tasks.
Each sink task drives its own `SinkKind::write()` loop independently.

---

## Key Design Decisions

### DC1: Pane Identity and Addressing

**Decision**: Use tmux `#{pane_id}` (e.g., `%12`) as the authoritative identifier for
stream attribution and internal keying. Retain `session:window.pane` in
`PaneAddress` as display metadata and user-facing targeting only.

**Rationale**: `#{pane_id}` is a stable, unique, tmux-assigned identifier that does not
change when sessions or windows are renamed or moved. Using it as the internal key
eliminates the need for lossy filename encoding of session names (the original P2 bug)
and simplifies stream parsing — the monitor keys on `%<id>` directly rather than
decoding hex-encoded filenames.

**PaneAddress** retains human-readable fields for display and `tmux send-keys -t` targeting:
```rust
pub struct PaneAddress {
    pub pane_id: String,       // authoritative: "%12" — used for stream keys
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

### DC3: Trigger/Action Model (Historical)

**Status**: Deferred. Preserved for historical context in
[Appendix A](#appendix-a--historical-automation-direction).

The earlier direction introduced configurable rules with per-pane cooldown. That is
no longer an active design goal. Policy now lives outside the library; Motlie focuses
on stream/history delivery plus routed control primitives.

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

**Shared state**: Only registry/binding metadata and shutdown signals are shared across
hosts. Per-host state (connection, panes, monitoring tasks, routed control queues) is
task-local.

### DC6: Local vs SSH Transport

**Decision**: A `TransportKind` enum abstracts command execution via static dispatch.
Three variants: `Local(LocalTransport)` (localhost, subprocess-based),
`Ssh(SshTransport)` (remote, russh-based), and `Mock(MockTransport)` (testing).

**Rationale**: The prototype assumes SSH for everything, but localhost tmux is a primary
use case (local development, CI, single-machine automation). Forcing SSH to localhost
adds unnecessary complexity (SSH server requirement, key management, latency). The enum
abstraction lets all downstream modules (`discovery`, `capture`, `control`,
`monitor`) be transport-agnostic with zero vtable overhead.

**LocalTransport specifics**:
- `exec()`: spawns `tokio::process::Command`, captures stdout, respects timeout
- `open_shell()`: spawns a persistent shell process with piped stdin/stdout
- No connection step; always available

**SshTransport specifics**:
- `exec()`: opens SSH exec channel, captures stdout
- `open_shell()`: opens PTY channel with shell
- Requires connection + auth before use
- Host key verification per DC2

**MockTransport** (for testing): Returns canned `exec()` responses and canned streaming
data from `open_shell()`. Built into the library, not behind a feature flag, so downstream
consumers can also test their integrations.

### DC22: Session Creation Options — Window Size and History Limit

**Problem**: `create_session()` runs `tmux new-session -d -s <name>` with no window
size or history limit control. Detached sessions default to 80x24 (or the server
default). History limit inherits the global `history-limit` option (tmux default:
2000 lines). Callers that need larger scrollback or specific geometry must issue
separate `set-option` calls manually and account for tmux's creation-time semantics.

**Decision**: Introduce `CreateSessionOptions` to bundle optional parameters, and
apply window size + history limit atomically during session creation.

**Approach (Option B)**: After `new-session -d -s <name> -x W -y H`, issue:
1. `set-option -t <name> history-limit <N>` — per-session, covers future windows/panes
2. `set-option -p -t <name> history-limit <N>` — per-pane (tmux 3.1+), covers the
   initial pane that `new-session` already created

If either `set-option` fails (e.g. tmux < 3.1 lacks `-p`), the implementation
rolls back by killing the just-created session to avoid leaked state.

This avoids the race condition of mutating/restoring the global `history-limit`
(Option A) and requires no minimum-version gate beyond tmux 3.1 (released 2020-06-25,
widely available).

**API changes**:

```rust
/// Options for session creation. All fields optional; defaults to tmux server defaults.
#[derive(Debug, Clone, Default)]
pub struct CreateSessionOptions {
    pub window_name: Option<String>,
    pub command: Option<String>,
    pub width: Option<u16>,
    pub height: Option<u16>,
    pub history_limit: Option<u32>,
    pub initial_environment: Vec<SessionEnvVar>,
}
```

`initial_environment` is emitted to `tmux new-session -e` in vector order. If a
caller supplies duplicate variable names, tmux applies the last emitted value.

`HostHandle::create_session` signature changes from:

```rust
pub async fn create_session(
    &self, name: &str, window_name: Option<&str>, command: Option<&str>,
) -> Result<Target>;
```

to:

```rust
pub async fn create_session(
    &self, name: &str, opts: &CreateSessionOptions,
) -> Result<Target>;
```

`control::create_session` (internal) changes similarly. The generated tmux commands:

```
tmux new-session -d -s <name> [-n <window_name>] [-x <W> -y <H>] [-e KEY=VALUE ...] [<command>]
tmux set-option -t <name> history-limit <N>        # if history_limit set
tmux set-option -p -t <name> history-limit <N>     # if history_limit set (tmux 3.1+)
```

### DC21: Unified SSH URI for Host Addressing

**Decision**: All hosts — including localhost — are addressed through a single SSH URI
scheme. The existing `SshConfig` type is extended with URI parsing, rendering, and a
`connect()` method that auto-selects the transport. No new wrapper type — `SshConfig`
becomes the single entry point for both programmatic and string-based host configuration.

**Requirement**: External consumers use `SshConfig` to specify a host and obtain a
`HostHandle`. The URI supports both nassh-style (`;`-delimited userinfo params) and
standard query-param (`?key=value&...`) syntax. `LocalTransport` and `SshTransport`
remain internal implementation details — `connect()` selects automatically.

**Non-goal**: Migration of existing `HostHandle::local()` / `HostHandle::new()` callers.
Those remain available as crate-internal API for testing and advanced use cases.

#### URI Format

```
ssh://[user[;param=value...]@]host[:port][/socket-path][?param=value&...]
```

Two parameter locations are supported:

- **Userinfo params** (nassh style): `ssh://user;timeout=30@host`
- **Query params** (standard style): `ssh://user@host?timeout=30`
- **Mixed**: `ssh://user;timeout=30@host?host-key-policy=tofu` (allowed)

**Examples**:
```
ssh://deploy@prod-server                              # defaults: port 22, verify
ssh://deploy;host-key-policy=tofu@prod-server         # nassh style
ssh://deploy@prod-server?host-key-policy=tofu         # query param style
ssh://root@10.0.0.5:2222?timeout=30                   # port in authority, timeout in query
ssh://user@localhost                                   # local transport (no SSH)
ssh://localhost                                        # local, no user (user ignored)
ssh://user@localhost/tmp/tmux-custom.sock              # local + socket path
ssh://deploy;host-key-policy=insecure@dev-box          # insecure policy
ssh://deploy@long-running?keepalive=0                  # disable keepalives
ssh://user;socket-name=myserver@host                   # named tmux socket
```

The `/socket-path` component maps to `TmuxSocket::Path(...)`. If omitted, the default
tmux socket is used. To use a named socket instead of a path, use the `socket-name`
parameter: `ssh://user;socket-name=myserver@host`.

**Mutual exclusion**: `/socket-path` and `socket-name` both map to the same
`Option<TmuxSocket>` field. If both are present in a URI, parsing fails:

```
ssh://user@host/tmp/tmux.sock?socket-name=other   # ERROR — socket-path and socket-name
                                                   #   are mutually exclusive
```

<!-- @claude 2026-03-12: added per PR #71 R2 — reviewer flagged undefined behavior
     when both socket specifiers are present. Reject at parse time. -->

#### No Canonical-Component Duplication

Components with a dedicated position in the URI syntax — `user`, `host`, `port` — are
parsed exclusively from their canonical location. They **cannot** appear as `;` or `?`
parameters. This eliminates ambiguity and conflicting values:

```
ssh://root@10.0.0.5:2222            # OK — port in authority
ssh://root@10.0.0.5?port=2222       # ERROR — port is a canonical component
ssh://root;port=2222@10.0.0.5       # ERROR — port is a canonical component
ssh://root;user=other@10.0.0.5      # ERROR — user is a canonical component
```

If the same non-canonical parameter appears in **both** userinfo and query string,
parsing fails. Likewise, repeated keys **within** a single location are rejected —
each non-canonical parameter name may appear at most once across the entire URI:

```
ssh://user;timeout=30@host?timeout=30   # ERROR — duplicate parameter (cross-location)
ssh://user;timeout=10;timeout=20@host   # ERROR — duplicate parameter (within userinfo)
ssh://user@host?timeout=10&timeout=20   # ERROR — duplicate parameter (within query)
```

<!-- @claude 2026-03-13: added per PR #71 R4 — reviewer flagged that within-location
     duplicate handling was unspecified. Reject at parse time for deterministic
     behavior and round-trip guarantees. -->

#### Parameters

Parameter names align with OpenSSH `ssh_config(5)` where an equivalent exists:

| Parameter | ssh_config Equivalent | Maps to | Default |
|---|---|---|---|
| `host-key-policy` | `StrictHostKeyChecking` | `HostKeyPolicy` | `verify` |
| `timeout` | `ConnectTimeout` | `SshConfig::timeout` (seconds, per-command exec/connect timeout) | `10` |
| `inactivity-timeout` | — | `SshConfig::inactivity_timeout` (seconds, 0=unlimited) | unlimited |
| `keepalive` | `ServerAliveInterval` | `SshConfig::keepalive_interval` (seconds, 0=off) | `30` |
| `socket-name` | — (tmux-specific) | `TmuxSocket::Name(...)` | none |
| `identity-file` | `IdentityFile` | `SshConfig::identity_file` (absolute path) | none (agent auth) |

<!-- @claude 2026-03-20: identity-file added by DC26. See DC26 for full design.
     @claude 2026-03-20: PR #89 R1 — identity-file is query-only per reviewer feedback.
     Absolute paths are a poor fit for userinfo/authority/path split. While the older
     transport params (host-key-policy, timeout, keepalive, socket-name) support both
     nassh `;` and query `?` placement, identity-file is restricted to query params only.
     Parsing rejects identity-file in userinfo; rendering always emits it as a query param. -->

**`host-key-policy` values**: `verify` (default, maps to `HostKeyPolicy::Verify`),
`tofu` (maps to `TrustFirstUse`), `insecure` (maps to `Insecure`).

Unknown parameters are rejected at parse time (fail-fast, not silently ignored).

**Canonical-only components** (never valid as parameters):

| Component | Parsed from | Notes |
|---|---|---|
| `user` | Userinfo (before `;` or `@`) | Optional for localhost (ignored); required for SSH |
| `host` | Authority | Required |
| `port` | Authority (`:port`) | Default 22 |

#### Consolidated `SshConfig` Type

Instead of a separate `SshUri` wrapper, `SshConfig` itself gains URI parsing,
rendering, and `connect()`. This eliminates type duplication — one type, one builder,
one set of fields:

```rust
/// SSH/host connection configuration.
///
/// Constructed via builder (`SshConfig::new()`) or parsed from an SSH URI
/// string (`SshConfig::parse()`). Supports both nassh-style (`;` in userinfo)
/// and query-param (`?key=value`) syntax.
#[derive(Debug, Clone)]
pub struct SshConfig {
    host: String,
    port: u16,
    user: String,
    host_key_policy: HostKeyPolicy,
    timeout: Duration,
    inactivity_timeout: Option<Duration>,
    keepalive_interval: Option<Duration>,
    socket: Option<TmuxSocket>,
    identity_file: Option<PathBuf>,     // DC26
}

impl SshConfig {
    // --- Builder (existing, extended) ---

    pub fn new(host: impl Into<String>, user: impl Into<String>) -> Self;
    pub fn with_port(self, port: u16) -> Self;
    pub fn with_host_key_policy(self, policy: HostKeyPolicy) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn with_inactivity_timeout(self, timeout: Option<Duration>) -> Self;
    pub fn with_keepalive(self, interval: Option<Duration>) -> Self;
    pub fn with_socket(self, socket: TmuxSocket) -> Self;  // NEW

    /// Set an explicit SSH identity (private key) file for authentication (DC26).
    ///
    /// Returns `Err` if the config already has an identity file set (e.g., from
    /// a parsed URI). This prevents silent overwrites when combining URI parsing
    /// with programmatic configuration.
    pub fn with_identity_file(self, path: impl Into<PathBuf>) -> Result<Self>;  // DC26

    // --- URI parsing (NEW) ---

    /// Parse from an `ssh://` URI string.
    ///
    /// Accepts nassh-style params (`;` in userinfo), query params (`?`),
    /// or both. Rejects unknown params and canonical-component duplication.
    /// For localhost URIs, `user` defaults to empty (ignored by LocalTransport).
    ///
    /// Note: `identity-file` is accepted as a **query param only** — it is
    /// rejected in nassh-style userinfo (see DC26 for rationale).
    pub fn parse(uri: &str) -> Result<Self>;

    /// Render to canonical URI form (nassh-style params in userinfo).
    ///
    /// `identity-file` is always rendered as a query param, even when other
    /// params use nassh-style userinfo placement (DC26: query-only constraint).
    pub fn to_uri_string(&self) -> String;

    // --- Connect (NEW) ---

    /// Connect and return a HostHandle.
    ///
    /// Transport selection:
    /// - `localhost`, `127.0.0.1`, `::1` → LocalTransport (no SSH)
    /// - All other hosts → SshTransport via SSH
    ///
    /// `user` is ignored for localhost connections (LocalTransport runs as
    /// the current OS user). For SSH hosts, `user` is required — `connect()`
    /// returns an error if empty.
    pub async fn connect(self) -> Result<HostHandle>;

    // --- Accessors ---

    pub fn host(&self) -> &str;
    pub fn user(&self) -> &str;
    pub fn port(&self) -> u16;
    pub fn is_localhost(&self) -> bool;
    pub fn socket(&self) -> Option<&TmuxSocket>;
    pub fn identity_file(&self) -> Option<&Path>;           // DC26
}

impl fmt::Display for SshConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uri_string())
    }
}

impl FromStr for SshConfig {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> { Self::parse(s) }
}
```

**Usage — basic parse and connect**:
```rust
use motlie_tmux::{SshConfig, HostKeyPolicy, TmuxSocket};
use std::time::Duration;

// Parse from string (config files, CLI args)
let host = SshConfig::parse("ssh://deploy@prod:2222?host-key-policy=tofu")?
    .connect().await?;
let sessions = host.list_sessions().await?;

// nassh-style params — same result, different syntax
let host = SshConfig::parse("ssh://deploy;host-key-policy=tofu@prod:2222")?
    .connect().await?;

// Build programmatically (unchanged from current API)
let host = SshConfig::new("10.0.0.5", "deploy")
    .with_port(2222)
    .with_host_key_policy(HostKeyPolicy::TrustFirstUse)
    .connect()
    .await?;

// Localhost — no SSH overhead, user optional
let host = SshConfig::parse("ssh://localhost")?.connect().await?;
let host = SshConfig::parse("ssh://user@localhost")?.connect().await?;
```

**Usage — config-file driven fleet**:
```rust
// Read URIs from a TOML config, connect in parallel.
let uris = vec![
    "ssh://deploy@web-1",
    "ssh://deploy@web-2",
    "ssh://deploy;host-key-policy=tofu@db-1:2222",
];
let hosts: Vec<HostHandle> = futures::future::try_join_all(
    uris.iter().map(|u| async { SshConfig::parse(u)?.connect().await })
).await?;
for (uri, host) in uris.iter().zip(&hosts) {
    let sessions = host.list_sessions().await?;
    println!("{}: {} sessions", uri, sessions.len());
}
```

**Usage — `FromStr` integration**:
```rust
// FromStr works with clap, config parsers, or plain .parse()
let uri = "ssh://deploy@prod:2222";
let config: SshConfig = uri.parse()?;  // FromStr delegates to SshConfig::parse()
let host = config.connect().await?;
```

**Usage — builder with socket selection**:
```rust
// Target a non-default tmux server on the remote host.
let host = SshConfig::new("10.0.0.5", "deploy")
    .with_port(2222)
    .with_host_key_policy(HostKeyPolicy::TrustFirstUse)
    .with_socket(TmuxSocket::Name("deploy-server".into()))
    .connect()
    .await?;
```

**Usage — round-trip serialization**:
```rust
// Parse, modify, serialize back for storage.
let config = SshConfig::parse("ssh://deploy@prod:2222")?
    .with_host_key_policy(HostKeyPolicy::TrustFirstUse);
let uri_string = config.to_string(); // "ssh://deploy;host-key-policy=tofu@prod:2222"
save_to_config_file(&uri_string);
```

**Usage — localhost development**:
```rust
// Identical to HostHandle::local() but with the uniform URI API.
let host = SshConfig::parse("ssh://localhost")?.connect().await?;
let opts = CreateSessionOptions { window_name: Some("main".to_string()), ..Default::default() };
host.create_session("dev", &opts).await?;
```

#### Transport Selection Logic

`SshConfig::connect()` selects the transport:

```
fn is_localhost(&self) -> bool:
    host == "localhost" || host == "127.0.0.1" || host == "::1"

async fn connect(self) -> Result<HostHandle>:
    if self.is_localhost():
        transport = TransportKind::Local(LocalTransport::with_timeout(self.timeout))
        return HostHandle::new(transport, self.socket)
    if self.user.is_empty():
        return Err("user is required for SSH connections")
    let socket = self.socket.clone()
    transport = TransportKind::Ssh(SshTransport::connect(self).await?)
    return HostHandle::new(transport, socket)
```

<!-- @claude 2026-03-12: PR #71 R1 fix — `connect(self)` takes ownership, consistent
     with `SshTransport::connect(SshConfig)`. No hidden clone. Caller `.clone().connect()`
     for fleet reuse. Socket extracted before self is moved into SshTransport. -->

`connect(self)` takes ownership, consistent with `SshTransport::connect(SshConfig)`.
For fleet patterns where the caller reuses a config, clone explicitly:
`cfg.clone().connect().await?`. Socket is extracted before `self` moves into
`SshTransport`. For localhost, `user` is silently ignored (no SSH handshake occurs).

#### Refactor Plan

The change extends the existing `SshConfig` rather than adding a new type:

1. **`src/transport.rs`**: Add `socket: Option<TmuxSocket>` field to `SshConfig`,
   add `with_socket()` builder method, make fields private with accessor methods.
   Internal usage in `SshTransport` and `SshHandler` switches from `config.host` to
   `config.host()`. All changes are within the same file.

2. **New file `src/uri.rs`**: Contains `impl SshConfig` extension block with `parse()`,
   `to_uri_string()`, `connect()`, `Display`, `FromStr` impls, and all URI-related
   tests. Rust allows `impl` blocks for a type in any module within the same crate —
   this keeps URI logic separated from transport logic while consolidating on one type.

3. **`src/lib.rs`**: Add `mod uri;` (private — no public types to export, just extends
   `SshConfig` which is already re-exported).

4. **`src/host.rs`**: Add test-only `#[cfg(test)] pub(crate) fn transport_kind(&self)
   -> &TransportKind` accessor to `HostHandle`. Not part of the public API or
   non-test build surface, so the transport split stays internal per DC21's
   requirement while still enabling localhost transport-selection unit tests
   within the crate to assert behavior without relying on side effects.
   `HostHandle::local()`, `HostHandle::new()`,
   `HostHandle::local_with_timeout()` remain as crate-internal convenience constructors
   for testing and advanced use cases.
   <!-- @claude 2026-03-13: scoped to pub(crate) per PR #71 R4 — reviewer correctly
        flagged that a pub accessor leaks the transport abstraction DC21 says should
        stay internal. -->

### DC26: SSH Identity File Authentication

**Problem**: `SshTransport::connect()` authenticates exclusively via `ssh-agent`
(`authenticate_agent()`). If no agent is running or no keys are loaded, connection
fails with an actionable error (OC3). However, there is no mechanism to specify an
explicit private key file. This is a usability gap for:

1. **CI/CD pipelines** — headless runners with deploy keys, no agent daemon.
2. **Multi-identity hosts** — deterministic key selection per host (e.g., separate
   deploy keys for prod vs staging).
3. **Containers** — Docker images with baked-in deploy keys, no agent forwarding.
4. **Agentless workflows** — environments where `ssh-agent` is unavailable or
   undesirable.

**Decision**: Add `identity-file` as a **query-only** URI parameter and `SshConfig`
field. When set, authentication uses the specified key file directly via
`russh_keys::load_secret_key()` and `handle.authenticate_publickey()` — both already
in the dependency tree (`russh` 0.46, `russh-keys` 0.46) but unused. No new
dependencies. No OpenSSH binaries or C FFI.

**Auth strategy dispatch** in `SshTransport::connect()`:

```
if config.identity_file is Some(path):
    authenticate_key_file(handle, config, path)
else:
    authenticate_agent(handle, config)       // current default, unchanged
```

When `identity-file` is specified, the agent is not consulted. This is intentional —
if a user specifies an explicit key, they want deterministic selection, not a silent
fallback to agent keys that may or may not work.

**URI parameter**: `identity-file` (**query-only**)

Unlike the older transport params (`host-key-policy`, `timeout`, `keepalive`,
`socket-name`) which accept both nassh-style `;` userinfo and `?` query placement,
`identity-file` is restricted to query params only. Rationale: it is a client-side
auth input rather than part of the remote authority shape, and absolute file paths
are a poor fit for the userinfo/authority/path URI split. Parsing rejects
`identity-file` in userinfo; rendering always emits it as a query param.

```
ssh://deploy@prod?identity-file=/home/deploy/.ssh/id_ed25519
ssh://deploy@prod:2222?host-key-policy=tofu&identity-file=/etc/deploy/key
```

The following is **rejected** at parse time:
```
ssh://deploy;identity-file=/path/to/key@prod   # ERROR — identity-file is query-only
```

**Parameter table update** (extends DC21 §Parameters):

| Parameter | ssh_config Equivalent | Maps to | Default | Placement |
|---|---|---|---|---|
| `identity-file` | `IdentityFile` | `SshConfig::identity_file` (absolute path) | none (use agent) | query only |

**SshConfig struct change**:

```rust
pub struct SshConfig {
    // ... existing fields ...
    identity_file: Option<PathBuf>,   // NEW (DC26)
}

impl SshConfig {
    /// Set an explicit SSH identity file for key-file authentication.
    ///
    /// Returns `Err` if the config already has an identity file set (e.g.,
    /// from a parsed URI). This prevents silent overwrites when combining
    /// URI parsing with programmatic configuration.
    pub fn with_identity_file(self, path: impl Into<PathBuf>) -> Result<Self>;
    pub fn identity_file(&self) -> Option<&Path>;
}
```

**Duplicate-source error**: If a config is parsed from a URI containing
`?identity-file=...` and the caller then also calls `.with_identity_file(...)`,
`with_identity_file()` returns `Err` rather than silently overwriting the
parsed value. This mirrors the fail-fast principle used elsewhere in DC21
(canonical-component duplication, cross-location duplicate params). The error
message identifies both the existing and attempted paths so the caller can
diagnose the conflict:

```rust
// OK — builder-only
SshConfig::new("prod", "deploy")
    .with_identity_file("/keys/prod")?;

// OK — URI-only
SshConfig::parse("ssh://deploy@prod?identity-file=/keys/prod")?;

// ERROR — duplicate identity-file specification
SshConfig::parse("ssh://deploy@prod?identity-file=/keys/prod")?
    .with_identity_file("/keys/other")?;
// => Err("identity-file already set to '/keys/prod'; cannot overwrite with '/keys/other'")
```

**New auth method** (`transport.rs`):

```rust
async fn authenticate_key_file(
    handle: &mut russh::client::Handle<SshHandler>,
    config: &SshConfig,
    key_path: &Path,
) -> Result<()> {
    let key_pair = russh_keys::load_secret_key(key_path, None)
        .map_err(|e| anyhow!(
            "Failed to load SSH key '{}': {}. \
             If the key is passphrase-protected, load it into ssh-agent instead.",
            key_path.display(), e
        ))?;

    let accepted = handle
        .authenticate_publickey(&config.user, Arc::new(key_pair))
        .await
        .map_err(|e| anyhow!("SSH pubkey auth failed: {}", e))?;

    if !accepted {
        return Err(anyhow!(
            "SSH key '{}' was rejected by {}:{}",
            key_path.display(), config.host, config.port
        ));
    }
    Ok(())
}
```

**URI parse/render**:

- `KNOWN_PARAMS` adds `"identity-file"`.
- `parse()`: if `identity-file` appears in userinfo params, reject with error
  ("identity-file is query-only"). If in query params, validate absolute path,
  set `config.identity_file`.
- `to_uri_string()`: always emits `identity-file` as a query param, even when
  user is non-empty (other params go to nassh userinfo in that case).
- Round-trip: absolute POSIX paths contain no URI-reserved characters (`; @ ? & = # [ ]`)
  under normal conditions, so round-trip is safe without percent-encoding.

**Design constraints**:

| Decision | Rationale |
|----------|-----------|
| Query-only in URI | Client-side auth input, not part of remote authority. Absolute paths are a poor fit for userinfo/authority/path split. |
| Absolute paths only | Relative paths are ambiguous (relative to what CWD?). Mirrors how `IdentityFile` works in practice when fully qualified. |
| Single key file, not a list | Keeps URI clean. Multi-key → use agent. YAGNI for v1. |
| No passphrase support in v1 | `load_secret_key(path, None)` — encrypted keys fail with clear error suggesting agent. Avoids blocking I/O / TTY interaction in async context. Passphrases never appear in URIs (security: URIs in logs, env vars, process lists). |
| No `~` expansion | `~` is a shell construct, not a path literal. Users pass expanded paths. Avoids footgun. |
| No agent fallback when identity-file is set | Explicit key = deterministic. Silent fallback defeats the purpose. |
| Localhost ignores identity-file | `LocalTransport` does not SSH. If `identity-file` is set on a localhost URI, it is silently ignored (same as `user`). |
| Fallible `with_identity_file()` | Prevents silent overwrite when combining URI parse with builder. URI-specified + programmatic = error, not last-write-wins. |

**Desired usage**:

```rust
// CI deploy key — query-param URI, no agent needed
let host = SshConfig::parse("ssh://deploy@prod?identity-file=/etc/deploy/id_ed25519")?
    .connect().await?;

// Builder API — programmatic
let host = SshConfig::new("prod", "deploy")
    .with_identity_file("/etc/deploy/id_ed25519")?
    .connect().await?;

// Fleet with per-host keys (query-param style)
let hosts = vec![
    "ssh://deploy@prod-1?identity-file=/keys/prod",
    "ssh://deploy@staging-1?identity-file=/keys/staging",
    "ssh://deploy@dev-box",  // agent auth (default)
];
```

**Relationship to OC3**: OC3 identified ssh-agent availability as a concern and proposed
actionable error messages. DC26 goes further — it provides an alternative auth path that
eliminates the agent dependency entirely for workflows that can provide a key file.
`authenticate_agent()` retains its OC3 error messages unchanged; `authenticate_key_file()`
provides its own actionable messages for key-load and rejection failures.

### DC23: Host-Level File Transfer via SFTP

**Decision**: Add file transfer as a **transport/host capability**, not as a tmux-target
capability. The public API should live on `HostHandle` and dispatch through
`TransportKind`, with SSH-backed hosts using **SFTP** over the existing `russh`
connection. Use **`upload` / `download`** naming in the public API. Support files
and directories now. `Target` is intentionally not extended for file transfer.

**Rationale**:
- File transfer addresses the host filesystem, not a tmux session/window/pane hierarchy.
- The existing split already distinguishes transport-level host execution from pane-level
  `Target::exec()`. SFTP complements the former.
- SFTP is a better fit than the SCP protocol for the current architecture: binary-safe,
  structured, and implementable over the in-process SSH client without shelling out to
  external `scp`.
- This is greenfield work, so breaking changes are acceptable and no migration layer is needed.

**Initial scope**:
- Host-level `upload` / `download` operations
- Strongly typed path parameters for both local and remote endpoints
- Support for regular files and directories
- Transport-agnostic surface across `Local`, `Mock`, and `Ssh`
- SSH implementation uses SFTP
- Directory placement follows `cp -r` semantics: existing destination directory means
  copy into it; missing destination path means copy as that path
- Configurable overwrite semantics (`overwrite=false` returns error on existing destination;
  for directories, `overwrite=true` merges into an existing destination tree rather than
  replacing it)
- Reject symlinks initially rather than following them
- Do not preserve file metadata as part of the transfer contract
- Public transfer methods return `Result<()>` initially
- No required migration/backwards-compatibility layer

**Deep dive**: See [`SFTP.md`](./SFTP.md) for the design and implementation outline.

### DC24: Stream Pipeline Architecture — Pub/Sub vs Unix Pipes

**Decision**: Retain the pub/sub `OutputBus` architecture for fan-out, but extract
`JoinedStream` from `SinkKind` into an independent consumer-side stream combinator.
Adopt Unix-pipe-style composition at the subscription layer for transforms/history,
while keeping policy and action selection outside the library.

**Context**: The original Track B direction leaned toward a built-in matcher/rule/reactor
engine. After Track A landed and the dominant use case became clearer, the design shifted:
the common consumer is an external LLM/classifier that combines output into conversation
history, makes a decision out of process, and then uses Motlie again to route control
actions (`send_text`, `send_keys`, `capture`, etc.) back to the same session or pane.
That makes Motlie primarily a **stream/history/control substrate**, not a policy engine.

**Evaluation**:

| Criterion | Pub/Sub (OutputBus) | Unix Pipe Model |
|-----------|--------------------|--------------------|
| Fan-out (primary use case) | Native — publish to N subscribers | Requires explicit `tee` at every branch |
| Fan-in / joining | Combinator on subscriber side | Source-level `join()` combinator |
| Runtime dynamism | `subscribe()`/`unsubscribe()` at any time | Pipeline topology is structural — adding consumers requires rebuilding |
| Backpressure | Per-subscriber channels, independent | Propagates upstream through chain — slow stage blocks source |
| Rust implementation | `SinkKind` enum, `mpsc` channels — idiomatic | `Stream` trait composition, `Pin`, type-erased chains — complex |
| Transform/history chains | Consumer-side adapters | First-class composable stages |
| Complexity | One struct, publish/subscribe | Stage traits, chain builders, tee/join combinators, per-stage lifecycle |

**Conclusion**: Full pipe adoption is still unnecessary — the actual patterns (fan-out to
multiple consumers, fan-in from multiple hosts, runtime subscribe/unsubscribe, per-consumer
backpressure tolerance) are exactly what pub/sub handles well. But linear transforms and
history construction *are* important at the consumer boundary. The right split is:
pub/sub for routing/fan-out, subscription adapters for transforms/history, and an
external policy engine for decisions.

**What we borrowed**: The analysis revealed two composability improvements that adopt
pipe-style thinking without abandoning pub/sub:

1. **`Subscription` as the composable seam**: The bus exposes one primitive —
   `subscribe(filters, capacity) -> Subscription`. All consumer-side composition
   (joining, mapping, predicate filtering, history building, piping) is expressed as
   `Subscription` adapter methods. This is the Unix-pipe ergonomic pattern applied at
   the consumer boundary: the bus does fan-out and source routing; adapters chain above it.

2. **`JoinedStream` as an adapter, not a sink variant**: Joining is a consumer
   concern (like Unix `paste`), not a sink property. `subscription.joined(format)`
   returns a `JoinedStream` that produces `StreamChunk`s with `source_changed` flags
   for coalescing. Composes freely with any downstream consumer.

3. **History-oriented adapters over rule engines**: The primary downstream artifact is
   no longer “rule matched” but “conversation/history is ready for analysis.” The active
   path therefore prioritizes transcript-friendly adapters and bounded history windows over
   a built-in matcher DSL. Matcher/rule/reactor designs are preserved in the appendix as
   historical context rather than deleted.

**Layering**:
- **Bus**: Source routing (host/session/pane), fan-out, backpressure/gap tracking
- **Subscription adapters**: Joining, plain-text/history transforms, consumer-side predicates
- **Terminal consumers**: `SinkKind` (stdio, callback), custom code via `into_receiver()`
- **External policy**: LLM/classifier or other consumer decides whether and how to act,
  then uses Motlie’s normal control APIs to route actions back through `HostHandle`,
  `Target`, or `Fleet`

**Impact**: DC15 remains the active multi-source view. OutputBus stays simplified to one
`subscribe()` primitive. The active next work shifts from matcher/reactor machinery to
history/transcript composition and Fleet coordination for external agents.

### DC28: Rolling Transcript / History for External LLM Context

**Decision**: The transcript/history layer is a bounded in-memory rolling snapshot built
on top of `JoinedStream`. `subscription.history(opts)` returns a `HistoryHandle` that
accumulates source-labeled output bursts and explicit gap markers, and exposes two
consumer surfaces:
- `snapshot()` for structured access
- `render_text()` for prompt-ready rolling context

**Why this artifact**:
- The dominant consumer is an external LLM/classifier loop with a finite context window.
- That loop usually wants the **latest bounded transcript now**, not a second async
  stream that it must separately buffer and trim.
- A bounded snapshot handle keeps Motlie responsible for the tmux-specific mechanics
  (source labels, coalescing, gap markers, omission markers) while leaving model choice,
  prompt framing, and decision-making outside the library.

**Relationship to `JoinedStream`**:
- History is built **on top of** `JoinedStream`, not in parallel with it.
- `JoinedStream` already solves source attribution and `source_changed` coalescing.
- `HistoryHandle` adds accumulation, trimming, omission markers, and prompt-oriented
  rendering on top of those semantics.

**Bounding strategy**:
- The primary window is **global merged history**, not per-source history, because LLM
  context windows are global.
- History is bounded by:
  1. logical entry count (`max_entries`)
  2. rendered character budget (`max_render_chars`)
- When over budget, trim the **oldest entries first**.
- If trimming occurred and `include_omission_marker` is true, prepend a single omission
  marker like `[... 37 earlier entries omitted ...]` in `render_text()`.
- Character budget is used in-library because token budgets are model-specific and belong
  to the external consumer.

**Consumer interaction model**:
- Primary path: polling/snapshot.
- External agent loop:
  1. `let history = sub.history(opts);`
  2. periodically `let prompt_context = history.render_text();`
  3. pass `prompt_context` to the model/classifier
  4. route actions back with `Fleet`, `HostHandle`, or `Target`
- Structured consumers can call `snapshot()` and format entries themselves.

**Why not an async `HistoryStream` as the primary artifact**:
- It would force every external consumer to reimplement rolling buffering and trimming.
- That duplicates the exact prompt-window management problem this layer should solve.
- Snapshot access composes well with periodic model inference and agent turns.

**Why this supports the rolling-context use case cleanly**:
- The same `HistoryHandle` can live for the lifetime of an agent session.
- `render_text()` always returns the most recent bounded, source-labeled context.
- Gap markers and omission markers make uncertainty explicit instead of silently
  pretending the transcript is complete.
- External policy stays simple: “get latest context, infer, act.”

### DC33: Per-Source Coherent History Rendering

**Decision**: Multi-pane history rendering switches from a single interleaved
timeline to per-source coherent sections. See [`docs/HISTORY.md`](./HISTORY.md)
for full design.

**Problem**: The DC28 interleaved timeline garbles multi-pane output — an LLM
reading the context cannot reconstruct what happened in each pane because
entries are arbitrarily interleaved by tmux `%output` frame timing.

**Solution** (three phases):
1. **Coalesce** consecutive same-source `%output` chunks into single entries
   (reduces interleaving noise, no API change).
2. **`RenderMode::PerSource`** groups entries by source and renders each as a
   labeled section. Default stays `Interleaved` for backward compat.
3. **Per-source budgets** — each source gets independent `max_entries` /
   `max_render_chars` so a noisy pane cannot evict a quiet pane's context.

**Impact on DC28**: The `HistoryOptions` struct gains a `render_mode` field.
`HistoryHandle::render_text()` dispatches on it. `PollHistory` gains a
matching `push_text_for_source()` + `RenderMode` option. Existing callers
using `RenderMode::Interleaved` (the default) see no behavior change.

### DC29: Streaming Resilience, Discontinuity, and Resync

**Decision**: Long-lived monitor reliability is now an explicit design goal, not just a
generic hardening follow-on. The streaming substrate must distinguish:
- **subscriber-local loss**: backpressure on one `Subscription` route (`Gap`)
- **upstream monitor discontinuity**: control-mode shell drop, SSH reconnect, tmux server
  restart, or resync after monitor recovery

The active design requires reconnect supervision and explicit discontinuity artifacts so
external-agent workflows can reason about uncertainty instead of silently treating the
transcript as complete.

**Why this matters now**:
- The dominant consumer is a long-lived LLM/classifier loop over rolling history.
- That consumer can tolerate bounded incompleteness if it is made explicit.
- It cannot safely reason from a transcript that silently skipped output because a
  control-mode monitor died and later restarted.
- `Gap` already models subscriber-local drop; overloading it for transport or monitor
  outages would blur two different failure classes.

**Required semantics**:
1. **Reconnect supervision**
   - A monitored session should not permanently die on the first unexpected EOF from the
     control-mode shell.
   - SSH-backed monitoring must attempt bounded reconnect with backoff.
   - Localhost monitoring should also support reattach/restart semantics when the tmux
     server or the control-mode client dies unexpectedly.
2. **Explicit discontinuity signaling**
   - Upstream monitor interruptions must surface as an explicit stream/history artifact,
     distinct from `Gap`.
   - The artifact must be consumable by:
     - raw subscription receivers
     - transcript/history rendering
     - future TUI status surfaces
3. **Fresh snapshot after reconnect**
   - After a successful reconnect, the monitor must capture a bounded **current-state
     snapshot** to re-anchor the transcript rather than silently resuming from “now”.
   - This is **not** a replay or recovery of missed output. Any output produced during
     the outage and no longer present in tmux history is permanently lost.
   - The snapshot result should be reflected in history/output as a synthetic system
     entry so consumers know continuity was broken and the transcript is now anchored
     to a fresh visible-state snapshot.
4. **Health visibility**
   - `Fleet` / host-level coordination must be able to observe that a host or monitor is
     degraded, reconnecting, resumed, or permanently failed.
   - Long-lived consumers should not need to infer health only from absence of output.

**Artifact direction**:
- Keep `SinkEvent::Gap` for subscriber backpressure only.
- Introduce `SinkEvent::Discontinuity` (or an equivalent dedicated system event) in the
  streaming path.
- `HistoryEntry` gains a corresponding `Discontinuity` variant.
- `HistoryHandle::render_text()` should include prompt-visible lines such as:
  - `[stream interrupted: ssh control channel lost for web-1:build]`
  - `[stream resumed: reattached after reconnect]`
  - `[stream snapshot: captured current screen state after reconnect; intermediate output may be missing]`

**Adapter propagation contract**:
- **Raw subscription receivers**: receive `SinkEvent::Discontinuity` directly.
- **`filter_fn()`**: always forwards discontinuity events, same as `Gap`. Predicates apply
  only to `TargetOutput`, not to system continuity signals.
- **`pipe()`**: forwards discontinuity transparently to the terminal sink/callback.
- **`HistoryHandle`**: records discontinuity as `HistoryEntry::Discontinuity`; it counts
  against normal `max_entries` / `max_render_chars` budgets because it is part of the
  truthfulness contract of the transcript.
- **`JoinedStream`**: does not synthesize a fake data chunk for discontinuity. Instead it
  resets source-coalescing state so the next real output chunk is treated as
  `source_changed = true`. Consumers that need explicit discontinuity markers should use
  raw subscriptions or `HistoryHandle`.

**Session identity and topology semantics**:
- **tmux server restart / session killed externally**:
  - if reconnect succeeds but the monitored session no longer exists, emit a terminal
    discontinuity/failure marker and transition that session monitor to permanent
    failed/stopped state
  - Fleet target aliases remain as names, but subsequent target resolution / routed actions may fail
    until the caller recreates or rebinds the target
- **Pane topology change during outage**:
  - if the session still exists but pane IDs changed, the monitor resumes at the session
    level and emits a discontinuity + fresh snapshot marker
  - subscriptions filtered to old pane IDs naturally stop matching removed panes; callers
    must resubscribe if they require the new pane identities

**Health model**:
- Per-session monitor health is the ground truth:
  - `streaming`
  - `reconnecting`
  - `failed`
  - `stopped`
- Host/Fleet health is derived from per-session state rather than flattened into one
  coarse status. An aggregate such as counts or worst-of severity is acceptable, but the
  per-session view must remain inspectable for targeted recovery decisions.

**Relationship to external-agent workflows**:
- This design keeps Motlie responsible for transport truthfulness and bounded recovery.
- The external agent stays simple:
  1. read rolling context
  2. see explicit interruption/resume/resync markers
  3. decide whether to continue, recapture more context, or take corrective action
- That is cleaner than pushing reconnect detection and transcript repair into every
  LLM/classifier consumer.

**Relationship to TUI**:
- TUI remains fully supported.
- The same discontinuity/resync artifacts that feed `HistoryHandle` can drive TUI status
  badges, transcript banners, or degraded-connection indicators without requiring a
  separate monitoring model.

### DC30: Dedicated Socket Isolation Ergonomics

**Decision**: Motlie should treat dedicated tmux sockets as a first-class robustness tool for
automation, not merely as an advanced optional parameter. The existing `TmuxSocket` selector
remains the core transport primitive, but the library should add higher-level helpers that make
isolated automation sockets easy to construct, validate, and operate.

This decision is driven by the product comparison captured in
[`docs/PRODUCT.md`](../../../docs/PRODUCT.md): for a tmux-over-SSH foundation, socket isolation
improves correctness and determinism more directly than many broader tmux command-surface
expansions.

**Why this matters**:
- Dedicated sockets reduce interference from unrelated human tmux activity.
- They make monitor/capture/exec behavior more deterministic by narrowing the operational scope.
- They provide a clean boundary for long-lived external-agent or automation workflows without
  requiring complex runtime coordination.
- They improve debuggability: "this automation socket" becomes a concrete operational unit.

**Required semantics**:
1. **Stable, explicit isolation**
   - Socket selection must remain explicit and inspectable.
   - Motlie must not silently fall back from a dedicated automation socket to the default tmux
     socket when the isolated socket is missing or not running.
2. **Ergonomic construction**
   - Callers should not have to hand-roll every socket name string.
   - Provide a safe, deterministic helper for creating automation-oriented socket names from a
     human scope string.
3. **Bootstrap / readiness**
   - A socket-scoped host should be able to ensure the tmux server for that socket exists before
     higher-level operations start.
   - This should be available for localhost and SSH-backed hosts through the same `HostHandle`
     abstraction.
4. **No ambiguity with existing explicit socket configuration**
   - If the caller already specified a concrete socket, convenience helpers must return an error
     rather than silently overwrite or merge socket intent.

**Proposed API direction**:

```rust
impl TmuxSocket {
    pub fn automation(scope: &str) -> Result<Self>;
}

impl SshConfig {
    pub fn with_automation_socket(self, scope: &str) -> Result<Self>;
}

impl HostHandle {
    pub async fn ensure_socket_server(&self) -> Result<()>;
}
```

**Behavioral contract**:
- `TmuxSocket::automation(scope)` creates a deterministic, validated named socket intended for
  dedicated automation use (for example `motlie-<scope>`). The exact normalization algorithm is
  library-defined, ASCII-safe, and documented.
- `with_automation_socket(scope)` is a convenience wrapper over `with_socket(...)`, but it is
  fallible if a socket is already configured.
- `ensure_socket_server()` explicitly runs `tmux start-server` against the configured socket.
- None of these APIs manage session lifecycle automatically; they only improve socket-scoped
  operational isolation.

**Why not stop at raw `TmuxSocket`?**
- Raw socket selectors are necessary but not sufficient. The product gap is operational
  ergonomics: callers should be guided toward isolation by default for robust automation flows.
- "Pass `socket-name` manually everywhere" is too low-level to function as a product answer.

### DC31: Tracked Command Execution

**Decision**: Add tracked command execution as a first-class `Target` capability that
complements, but does not replace, `Target::exec()`. The tracked form should expose an explicit
execution handle and execution state so callers can distinguish:

- command launched
- command still running
- command completed with structured result
- command outcome became unknown because continuity was broken

This is a robustness feature, not a workflow engine. It exists to make long-running or delayed
command observation more truthful in the face of reconnects, slow commands, and polling loops.

**Why this matters**:
- `Target::exec()` is a good await-to-completion convenience API, but it collapses launch + wait + parse
  into one operation.
- Long-lived automation and external-agent loops benefit from separating "start command" from
  "observe command outcome later".
- If a monitor/transport discontinuity happens while a command is in flight, the right answer is
  often not "timeout" but "result unknown".

**Required semantics**:
1. **Pane-scoped execution remains the boundary**
   - No host-level bypass (`HostHandle::exec()`) is added.
   - All command execution still happens inside the target pane's shell context.
2. **Tracked lifecycle**
   - Starting an execution returns a typed handle / ID.
   - Callers can inspect status or await completion later.
3. **Unknown outcome on broken continuity**
   - If command completion can no longer be proven because continuity was broken before the
     sentinel/result boundary was observed, the tracked command must transition to an explicit
     unknown state rather than pretending success, timeout, or clean completion.
4. **Blocking `exec()` remains**
   - `Target::exec()` stays as the simple await-to-completion convenience wrapper, layered on the tracked
     execution substrate where appropriate.

**Proposed API direction**:

```rust
pub struct ExecId(uuid::Uuid);

pub enum ExecState {
    Running,
    Completed(ExecOutput),
    Unknown { reason: String },
}

pub struct ExecHandle {
    // opaque
}

impl Target {
    pub async fn start_exec(
        &self,
        command: &str,
        timeout: std::time::Duration,
    ) -> Result<ExecHandle>;
}

impl ExecHandle {
    pub fn id(&self) -> ExecId;
    pub fn status(&self) -> ExecState;        // sync, infallible snapshot
    pub async fn wait(self) -> Result<ExecState>;  // consumes handle
}
```

**Behavioral contract**:
- The tracked execution state is process-local and in-memory. It is not intended as a
  cross-process or persistent job store.
- `Completed(ExecOutput)` covers all exit codes, including non-zero exits. A command that
  exits with `1` is still `Completed(...)`; `ExecOutput.exit_code` carries the result.
  Non-zero exit codes are not `Unknown`.
- Same-pane concurrency restrictions from DC19 still apply.
- On monitor/transport discontinuity before completion is proven, the tracked command transitions
  to `Unknown { reason }`.
- `Target::exec()` may internally call `start_exec(...).wait()` and require `Completed(...)` to
  preserve the existing simple API.

**Why this is not redundant with DC19**
- DC19 is about await-to-completion command-and-capture convenience.
- DC31 is about explicit execution state and truthfulness over time.
- Both belong on `Target`, and both preserve the tmux-pane abstraction boundary.

### DC32: Split-Screen REPL TUI Mirror First, Full Dashboard Later

**Decision**: The first TUI-facing delivery should be a split-screen REPL mode,
not a standalone dashboard and not a new core `SinkKind::Tui` variant.

The immediate product cut is:
- top frame mirrors a watched remote session
- bottom frame preserves the existing REPL prompt and command history
- `tui on` enters the split-screen mode
- `tui off` stops the active mirror and returns to plain REPL mode

**Why this shape first**:
- It proves the user-facing value of TUI output without forcing a full navigation
  or dashboard design immediately.
- It fits the existing examples/repl workflow, so the feature is easy to demo and test.
- The dominant near-term use case is line-oriented shells, logs, and agent/chat
  traces, which are already well served by the stream/history substrate.

**Why this should not be a core `SinkKind::Tui` yet**:
- Terminal UI ownership (alternate screen, raw mode, cursor, resize, draw cadence)
  is a binary concern, not a core library concern.
- Extending `SinkKind` with a TUI variant would drag local-terminal lifecycle and
  `ratatui`-style dependencies into `libs/tmux` prematurely.
- The current `OutputBus` / `Subscription` / `HistoryHandle` surface already gives
  a binary-local consumer enough structure to render the watched stream.

**Recommended architecture**:
1. `libs/tmux` remains unchanged at the sink-enum layer.
2. The REPL or a small helper module beside it creates a binary-local `TuiMirrorSink`
   / `ReplTuiMirror` consumer.
3. That consumer subscribes to the existing bus, uses transcript/history semantics
   for bounded top-frame state, and drives a `ratatui` draw loop locally.

**Command semantics**:
- `tui on`
  - enter alternate-screen split mode
  - show top mirror frame + bottom REPL frame
  - no watched session initially; render placeholder state
- `monitor <session>`
  - in plain REPL mode: preserve current stdout-stream behavior
  - in TUI mode: bind or switch the watched session for the top frame
- `tui off`
  - drop the watched-session subscription
  - leave alternate-screen/raw mode
  - restore plain REPL

**Delivery staging**:
- **Phase 5a**: transcript/history-oriented mirror for shell/chat/agent traces
- **Phase 5b**: full terminal-state mirror for cursor-addressed TUIs (`vim`, `htop`)

This keeps the first implementation aligned with the current external-agent and
conversation-history direction while leaving room for a deeper full-fidelity TUI path.

### DC27: Fleet Routing Convenience vs Direct Target Use

**Decision**: Keep both levels. `Target` remains the canonical direct-control handle,
while `Fleet` offers convenience routing for workflows that reason in target aliases
rather than holding a resolved `Target` the whole time.

**Rationale**:
- External agents often consume output first, then decide later what to do. In that
  delay, the most convenient stable handle is often a target alias or host alias
  rather than a previously retained `Target`.
- `Fleet` already owns the registry/binding layer, so convenience methods like
  `send_text(name, ...)`, `send_keys(name, ...)`, and `capture(name)` are natural
  wrappers around `find(name)` plus normal `Target` operations.
- This does not replace direct `Target` usage. Callers that already hold a `Target`
  should keep using it directly.

**Contract**:
- `Fleet`-level routing helpers are conveniences, not privileged APIs.
- They resolve a binding or alias, then delegate to the same underlying
  `HostHandle` / `Target` control path used by direct callers.
- Target alias resolution remains explicit and inspectable through
  `bind_target_alias()`, `resolve_target_alias()`, and `target_aliases()`.
  Historical `bind()`, `find()`, and `workstreams()` names were removed; caller
  concepts such as workstreams stay outside `libs/tmux`.

### DC7: Capture-Pane vs Stream Monitoring

**Decision**: Use `capture-pane -p` for on-demand snapshots, with optional `-e` when
fidelity mode requires escape-sequence preservation. For continuous monitoring, use
tmux control mode (DC10).

**Rationale**:
- `capture-pane -p` returns the current visible content (and scrollback with `-S`/`-E`)
  as a single snapshot. `-e` may be added for ANSI-preserving capture. This is
  stateless, idempotent, and requires no setup. Ideal for "what is this pane showing
  right now?" queries.
- Continuous monitoring uses control mode (`tmux -C attach`, DC10) which provides
  structured `%output %<pane_id>` framing without file lifecycle management.

**Callers choose the mode**:
- `capture_pane()` / `capture_session()` → snapshot via `capture-pane`
- `start_monitoring()` → continuous via control mode

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

**Decision**: Use tmux control mode (`tmux -C attach`) as the sole monitoring strategy.
The tmux 3.1+ baseline (DC22) guarantees control mode availability.

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

**~~Fallback~~**: ~~Pipe-pane with append-file sink (`pipe-pane -o 'cat >> file'` + `tail -f`)
is retained as an option for scenarios where control mode is insufficient (e.g., very old
tmux, or when monitoring panes across multiple tmux servers on the same host).~~

<!-- @claude 2026-03-16: PIPE-PANE FALLBACK IS OUT OF SCOPE.
     DC22 established tmux 3.1+ as the minimum version (per user direction: "migration/backwards
     compatibility explicitly out of scope"). tmux 3.1 (released 2020-06-25) fully supports
     control mode. With 3.1+ as the floor, control mode is guaranteed available and the pipe-pane
     fallback adds only complexity (FIFO lifecycle, P4 cleanup, OC1 backpressure, OC2
     interleaving) with no benefit. All pipe-pane/PipeManager/FIFO references in this document
     are historical context from early design iterations and should not be implemented.
     Corresponding PLAN.md Phase 2a.3 has been removed. -->

### DC11: Separation of Library and Binary

**Decision**: `libs/tmux` is a pure library. CLI binary lives in `bins/tmux-automator/`.

**Rationale**: Follows the existing workspace convention (`libs/mcp` + `examples/mcp/`).
The library exposes `Fleet`, `HostHandle`, `Target`, `OutputBus`, `Subscription`, and
all operation APIs. The binary handles CLI parsing, signal handling, and tracing
initialization. The library is also consumable by MCP tools, LLM agents, or other
programmatic callers.

### DC12: Output Sink Pipeline Architecture

**Decision**: Captured pane output is distributed to consumers via an async fan-out bus
(`OutputBus`) that delivers `TargetOutput` to subscriptions. Each subscription receives
its own bounded channel; terminal consumers (sinks) are attached via `.pipe(SinkKind)`.
Routing is owned by `subscribe(filters, capacity) -> Subscription` — sinks are terminal
consumers with no routing logic. Consumer-side composition is layered as adapters above
`Subscription`; the active path prioritizes joining, transcript/history construction,
and custom consumer predicates rather than a built-in matcher/reactor DSL.

**Rationale**:
- **Decoupled latency**: A slow LLM consumer must never block a fast stdio sink. Per-
  subscription channels with independent tasks ensure this.
- **Sink-owned batching**: The subscription's `.pipe()` adapter forwards `SinkEvent`s;
  sinks decide when/how to batch. A stdio sink flushes immediately. An LLM callback
  accumulates until a token budget or timeout. This keeps the bus simple.
- **Composable routing**: `SinkFilter` fields (host, session, window, pane) are regex
  patterns ANDed together. Multiple filters per subscription are ORed. This allows
  targeting "all panes on host-a" OR "session:build on any host" with a single subscribe.
- **Single routing layer**: Routing lives in `subscribe(filters)` only — sinks do not
  own filters. This eliminates ambiguity about where routing is configured.
- **External-agent-first**: The primary consumer is expected to be an external
  LLM/classifier or other policy engine. It consumes stream/history output, decides
  what to do, and then calls Motlie’s normal control APIs directly.
- **No privileged reaction path**: The library should not force all consumers through
  an internal rule engine when the common case is external analysis.

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
| Fleet | `fleet.start_monitoring()` | `fleet.shutdown()` |
| Host | `fleet.start_monitoring_host(host)` | `fleet.stop_monitoring_host(host)` |
| Session | `host.start_monitoring_session(&target)` | `host.stop_monitoring_session(&target)` |

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
`OutputBus`, and joins the monitor task. `stop_monitoring()` iterates all sessions.

**Invariant**: On-demand operations (`target.capture()`, `target.send_keys()`,
`host.list_sessions()`, etc.) are unaffected by monitoring state. A host with stopped
monitoring is still fully operational for on-demand use via `HostHandle` and `Target`.

### DC14: Historical Matcher DSL Direction

**Status**: Deferred. Preserved for historical context in
[Appendix A](#appendix-a--historical-automation-direction).

The previous direction introduced a built-in `MatcherKind` enum and subscription-side
`.filter()` / `.react()` adapters so Motlie could host its own automation logic. That
is no longer the active design goal. The common case is now external analysis over
conversation/transcript history, followed by ordinary Motlie control calls back into
tmux. As a result, matching remains an optional future layer rather than an active
centerpiece of the library.

### DC15: Joined Stream — Multi-Source Consolidated View

<!-- @claude 2026-03-17: DC24 revised DC15. JoinedSink was a SinkKind wrapper; now
     JoinedStream is a Subscription adapter returned by subscription.joined().
     See DC24 for the Unix-pipe analysis that motivated the change. -->

**Decision**: Multiple filtered output streams can be joined into a single time-ordered
sequence where each chunk is attributed to its source. This is implemented as a
`Subscription` adapter — `subscription.joined(label_format)` returns a `JoinedStream`
that produces `StreamChunk`s with `source_changed` flags for coalescing.

**Rationale**: When monitoring multiple panes across hosts, consumers often need a
unified view — an LLM analyzing cross-pane interactions, a log aggregator correlating
events, or a TUI showing interleaved output. Without joining, each sink sees isolated
streams and must implement its own source-tracking and interleaving logic.

**Design choices**:
- **Subscription adapter, not sink variant**: `subscription.joined(format)` returns a
  `JoinedStream`. It composes freely with any downstream consumer — `StdioSink`,
  `CallbackSink`, TUI channel, or custom rendering — without being a `SinkKind` variant.
  The bus stays simple (fan-out + source routing); joining is opt-in at the consumer site.
  (Revised from original `JoinedSink`-as-`SinkKind` design; see DC24 for rationale.)
- **Source coalescing**: Consecutive chunks from the same source are grouped without
  repeating the label. This mirrors chat UIs where the sender appears once per burst,
  reducing visual noise in high-throughput streams.
- **Label customization**: `LabelFormat` supports bracketed, prompt-style, and custom
  formatters. The `SourceLabel` type provides `short()` and `minimal()` for common cases.

**Alternatives considered**:
- ~~`JoinedSink` as `SinkKind` wrapper~~: Original design had joining as a sink
  combinator wrapping an inner `SinkKind`. Rejected during DC24 analysis — coupling
  joining to the sink enum limits composability and requires a special
  `subscribe_joined()` bus API. The combinator pattern composes better as an
  independent stream adapter.
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
| `Target` | Entity-level: session metadata (`tags`, `set_tag`, `read_tag`, `list_tags`), creation (`new_window`, `split_pane`), I/O (`send_text`, `send_keys`, `capture`, `sample_text`), navigation (`children`, `window`, `pane`), lifecycle (`kill`, `rename`), monitoring (`start_monitoring`) |
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

### DC34: Session Metadata Tags

**Decision**: Session metadata is represented as tmux user-defined session options
behind a small `Target` API:

```rust
pub struct SessionTag {
    /* private validated fields: prefix, key, value */
}

pub struct SessionTags<'a> {
    /* transport, tmux prefix, stable session id, validated tag prefix */
}

pub struct StatusStyle(String);
pub struct StatusLeft(String);
pub struct StatusLeftLength(u32);

pub struct SessionStatus<'a> {
    /* transport, tmux prefix, stable session id */
}

pub struct SessionStatusSnapshot {
    pub style: Option<StatusStyle>,
    pub left: Option<StatusLeft>,
    pub left_length: Option<StatusLeftLength>,
}

pub struct SessionStatusOverrides {
    pub style: Option<StatusStyle>,
    pub left: Option<StatusLeft>,
    pub left_length: Option<StatusLeftLength>,
}

impl Target {
    pub async fn tags(&self, prefix: &str) -> Result<SessionTags<'_>>;
    pub async fn status(&self) -> Result<SessionStatus<'_>>;
}

impl SessionTags<'_> {
    pub fn prefix(&self) -> &str;
    pub async fn set(&self, key: &str, value: &str) -> Result<()>;
    pub async fn read(&self, key: &str) -> Result<Option<String>>;
    pub async fn list(&self) -> Result<Vec<SessionTag>>;
    pub async fn unset(&self, key: &str) -> Result<()>;
}

impl SessionStatus<'_> {
    pub async fn snapshot(&self) -> Result<SessionStatusSnapshot>;
    pub async fn apply(&self, overrides: &SessionStatusOverrides) -> Result<()>;
    pub async fn restore(&self, snapshot: &SessionStatusSnapshot) -> Result<()>;
    pub async fn set_style(&self, style: &StatusStyle) -> Result<()>;
    pub async fn unset_style(&self) -> Result<()>;
    pub async fn read_local_style(&self) -> Result<Option<StatusStyle>>;
    pub async fn set_left(&self, left: &StatusLeft) -> Result<()>;
    pub async fn unset_left(&self) -> Result<()>;
    pub async fn read_local_left(&self) -> Result<Option<StatusLeft>>;
    pub async fn set_left_length(&self, length: StatusLeftLength) -> Result<()>;
    pub async fn unset_left_length(&self) -> Result<()>;
    pub async fn read_local_left_length(&self) -> Result<Option<StatusLeftLength>>;
}
```

`HostHandle::list_tags_for_session_infos(prefix, sessions)` is the batch read
companion for callers that already have a session listing and need to enrich it
without one round trip per session.

For `prefix = "mmux"` and `key = "role"`, the underlying tmux option is
`@mmux/role`. `SessionTag` carries the namespace prefix as well as the
unprefixed key so listed tags are self-describing and round-trippable.

**Scope**:
- Session targets only. Window and pane targets return `UnsupportedTarget`.
- Prefix and key are stable ASCII components: letters, digits, `.`, `_`, `-`.
- Values are UTF-8 strings, may be empty, reject control characters, and are
  capped at 2 KiB.
- Dispatch uses the stable `SessionId` from `SessionInfo`, not the display name.
- `Target::tags(prefix)` validates the prefix and captures the command prefix and
  stable session id once. There are no direct one-shot tag methods on `Target`;
  all tag work goes through the `SessionTags` scope.
- `HostHandle::list_tags_for_session_infos(prefix, sessions)` returns tags by
  stable `SessionId` and includes empty vectors for sessions without matching
  tags.
- `SessionTags::unset(key)` removes the session-local user option with tmux
  `set-option -u`; it does not encode deletion as an empty string.
- `Target::status()` captures the same session identity and command prefix for
  built-in status-bar options. `SessionStatus::snapshot/apply/restore` keep
  attach-time temporary chrome semantics in the library. `StatusStyle` rejects
  empty values; `StatusLeft` allows empty values to mean "render no left status
  text"; `StatusLeftLength` is a fallible bounded numeric type. Reads return
  only session-local overrides, not inherited/global values.
- The helper constructor is async because resolving the command prefix can lazily
  probe the tmux binary on the underlying transport before it is cached.

**Rationale**: tmux user-defined options are the closest native mechanism to
session metadata: they live in tmux state, survive for the session lifetime, and
can be written by processes inside the session. Keeping this API session-only
avoids a single method whose meaning changes across session/window/pane scopes.

**Command boundary**: The implementation uses direct tmux option commands through
the existing control module (`set-option`, `show-option -q`, `show-options`) and
does not run shell pipelines such as `grep`. Deletion uses `set-option -u -t
<stable-session-id> @<prefix>/<key>` with no value argument. A persistent
control-mode command client was not introduced for this slice because
`tmux -C attach-session` creates an attached client and would perturb
`attached_count`/client state for metadata polling. If the library later grows a
non-attaching command channel, these helpers can move under it without changing
the public contract.

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
`Option<Target>`. The builder alternative — `TargetSpec::session("build").window(0).pane(1)?` —
is preferred when components are known at compile time. This is the bridge from raw
strings (CLI args, config files) to typed handles.

### DC18: Target Alias Groups (Future)

**Status**: Deferred — documented for future reference. Current Fleet aliases
bind one name to one cross-host target. Multi-target grouping is out of scope for
`libs/tmux` until consumers prove the lifecycle and monitoring requirements.

**Concept**: A higher layer could group multiple `(host, target)` pairs under one
name, e.g. "deploy" spans `web-1:build`, `db-1:migrate`, and `web-1:test`, then
derive ordinary `SinkFilter` values from the member targets.

**Open questions** (to be resolved by usage):
- What happens when a member target is killed or disconnected?
- Can targets belong to multiple groups?
- Should groups be defined in config or only programmatically?

### DC25: Hierarchy Creation Symmetry on `Target`

**Problem**: The current API is asymmetric. `Target` already supports navigation,
I/O, lifecycle, and monitoring across session/window/pane levels, but only sessions
can be created through the typed API. Windows and panes still require callers to
shell out through `Target::exec("tmux new-window ...")` or external setup scripts.
That weakens the abstraction in three ways:

1. It leaks raw tmux command syntax into consumer code.
2. It returns untyped shell output instead of a typed `Target` for the new entity.
3. It breaks the symmetry of the hierarchy API: `kill()` works at every level, but
   the inverse creation path is only first-class at the root.

**Decision**: Add first-class window and pane creation to `Target`:

- `Target::new_window(&CreateWindowOptions) -> Result<Target>`
- `Target::split_pane(&SplitPaneOptions) -> Result<Target>`

`HostHandle::create_session()` remains the root-level creation API because sessions
have no tmux parent target. Child creation moves onto `Target`, where the hierarchy
context already exists.

**Level semantics**:

- `new_window()` is **session-level only**. Creating a window is a child-of-session
  operation; calling it on a window/pane target returns `Err`.
- `split_pane()` is allowed on **window** and **pane** targets.
  - Window target: split the active pane in that window.
  - Pane target: split that explicit pane.
- `split_pane()` on a session target returns `Err` rather than silently using the
  active window/pane. This keeps hierarchy growth explicit and avoids “active target”
  ambiguity in automation.

**API shape**:

```rust
#[derive(Debug, Clone, Default)]
pub struct CreateWindowOptions {
    pub name: Option<String>,
    pub command: Option<String>,
    pub width: Option<u16>,
    pub height: Option<u16>,
    pub start_directory: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
pub enum SplitDirection {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy)]
pub enum SplitSize {
    Cells(u16),
    Percent(u8),
}

impl SplitSize {
    /// Preferred constructor for percentage sizing. Rejects values above 100
    /// at construction time rather than deferring failure to command execution.
    pub fn percent(value: u8) -> Result<Self>;
}

#[derive(Debug, Clone)]
pub struct SplitPaneOptions {
    pub direction: SplitDirection,
    pub size: Option<SplitSize>,
    pub command: Option<String>,
    pub start_directory: Option<PathBuf>,
}

impl Target {
    pub async fn new_window(&self, opts: &CreateWindowOptions) -> Result<Target>;
    pub async fn split_pane(&self, opts: &SplitPaneOptions) -> Result<Target>;
}
```

The option structs intentionally mirror tmux’s `new-window` / `split-window`
surface while staying typed and shell-safe.

`SplitSize::Percent(u8)` remains the compact stored representation, but callers
should use a checked constructor such as `SplitSize::percent(50)?` so invalid
percentages are rejected at the API boundary rather than deep in the control layer.
Execution should still validate defensively.

At the tmux CLI layer, percentage splits map to `split-window -l <n>%`. tmux 3.4
does not provide a dedicated `split-window -p` percentage flag, so the control
wrapper should append `%` to the `-l` size value when `SplitSize::Percent` is used.

`start_directory` uses `PathBuf` for type safety. When the control layer builds the
tmux command, non-UTF-8 paths are rejected with `Err` rather than lossy conversion.

**Return strategy**: Creation APIs must return the created entity directly, not a
best-effort re-query of “whatever looks newest.” The implementation should use tmux’s
printing mode:

- `new-window -P -F ...`
- `split-window -P -F ...`

This lets the control layer capture the exact created window/pane identity in the
same command that performs the mutation, then construct the returned `Target`
deterministically.

**Why not keep using `exec("tmux ...")`?**

- `exec()` is a pane command runner governed by DC19, not a control-plane API.
- It forces callers to manually compose tmux command strings and parse success
  indirectly from shell output.
- It cannot return typed `WindowInfo` / `PaneAddress` without layering more parsing
  logic into consumers.
- It makes examples and higher-level tools tutorialize the workaround instead of the
  actual library abstraction.

**Why not add one generic `create(target_spec)` API?**

Rejected for now. A single “create hierarchy from spec” entry point conflates three
different operations with different parents and return shapes:

- root creation (`create_session`)
- child window creation (`new_window`)
- pane split (`split_pane`)

The direct primitives are clearer, compose naturally, and still leave room for a
higher-level convenience API later if real usage patterns justify it.

**Desired usage**:

```rust
let session = host.create_session("build", &Default::default()).await?;

let logs = session
    .new_window(&CreateWindowOptions {
        name: Some("logs".into()),
        command: Some("tail -f /var/log/app.log".into()),
        ..Default::default()
    })
    .await?;

let editor = logs
    .split_pane(&SplitPaneOptions {
        direction: SplitDirection::Vertical,
        size: Some(SplitSize::percent(50)?),
        command: Some("vim".into()),
        start_directory: None,
    })
    .await?;

editor.send_text(":w").await?;
```

**Rationale**: This restores symmetry to the hierarchy API:

- `HostHandle::create_session()` creates the root node
- `Target::new_window()` creates a window child
- `Target::split_pane()` creates a pane child
- `Target::kill()` remains the inverse lifecycle operation at every level

That makes `Target` a complete hierarchy abstraction rather than “operations on
preexisting entities plus a session-only root constructor.”

### DC19: Structured Command Execution via Target::exec()

**Decision**: `Target::exec(command, timeout)` provides structured command execution
within a tmux pane, complementing the existing fire-and-forget `send_text()`/`send_keys()`.
It returns `ExecOutput { stdout, exit_code }` by using a sentinel-based capture mechanism.
No host-level bypass (`HostHandle::exec()`) is added — the library's abstraction boundary
is tmux, and all command execution stays within the tmux framework.

**Relationship to DC31**: `exec()` remains the await-to-completion convenience form. A future tracked
execution API may layer underneath it, but it does not change this decision's core boundary:
execution stays pane-scoped and tmux-contextual.

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
  making boundary extraction ambiguous regardless of unique sentinels. `HostHandleInner`
  holds a per-pane exec lock map keyed by resolved `pane_id`. All target levels
  (session, window, pane) resolve their effective pane via
  `display-message -p '#{pane_id}'` before lock acquisition, so a session-level handle
  and a pane-level handle targeting the same active pane share the same lock.
  Callers needing parallel execution should use separate panes.

**Alternatives considered**:

- **Host-level `HostHandle::exec()`** (transport bypass): Rejected — runs outside tmux,
  so the command doesn't execute in the pane's environment (virtualenvs, working directory,
  shell aliases, env vars set by prior commands). The pane's shell is the ground truth for
  the target's state; bypassing it loses that context.
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

### DC20: Capture Normalization for Mixed-Client Screen Sizes

**Decision**: Add an explicit mixed-client fidelity layer for `capture()`,
`sample_text()`, `capture_session()` / `capture_all()`, and `exec()`. Default
on-demand capture remains `Raw` because the library cannot infer whether a pane is a
line-oriented shell or a full-screen TUI. Shell-oriented callers opt into
`ScreenStable` or `PlainText`; TUI workflows stay `Raw` unless a future terminal-state
path is implemented.

**Design boundary**: This section defines the API/behavioral contract impact on core
capture and exec paths. TUI-specific robustness policy, operational tiers, and
mitigations are documented in the companion [`TUI.md`](./TUI.md). The phased
implementation plan remains in `PLAN.md` and is intentionally not expanded here.

**Implementation slotting (usability-first)**:
1. Phase `1.9a`: capture/sample types, explicit modes, and metadata on localhost.
2. Phase `1.9b`: overlap resync, geometry detection, and history/setup helpers.
3. Phase `2a.2`: monitor stream assembly and normalization parity with capture paths.
4. Phase `2c.1` + `2c.3`: fidelity metadata in sink payloads and no-silent-drop gap signaling.
5. Phase `2c.4`: integrate monitor + sink contracts end-to-end.

**Why this is needed**: tmux pane content is width-dependent. When clients with
different screen sizes attach/detach, wrapping and visual layout can change. This
can make line-oriented matching and sentinel parsing brittle even when the underlying
command behavior is correct.

**Important limits**:
- Reflow/normalization is best-effort. It cannot perfectly reconstruct pre-wrap output
  after tmux has already reflowed or overwritten display state.
- No algorithm can recover content already evicted by tmux history limits.
- For strict determinism, automation should run in a dedicated session/pane with a
  fixed size, not a human-shared pane.
- Full-screen/cursor-addressed TUI panes are outside normalized matching defaults; use
  `Raw` unless a higher-cost terminal-state consumer is explicitly selected.

**Normalization and fidelity contract**:
- `Raw` uses `capture-pane -p` and returns tmux's default rendered text with no added
  normalization.
- `ScreenStable` uses `capture-pane -ep`, canonicalizes line endings, trims only
  width-artifact-safe trailing padding, and preserves ANSI/control sequences in the
  public `text` / `content` payload.
- `PlainText` is explicit opt-in for line-oriented consumers that want ANSI/control
  stripped in the public payload.
- `raw_text` / `raw_content`, when requested, preserve the exact tmux capture before
  mode-specific normalization.
- Keep ordering unchanged (top-to-bottom).
- `exec()` uses an internal derived parser view for wrap-tolerant sentinel detection
  without changing the public capture mode contract.

**Geometry/reflow detection contract**:
- Query attached client geometries with `tmux list-clients -F` (`client_width`,
  `client_height`, `client_session`).
- Query pane/window geometry and scrollback counters with `display-message -p` and
  format vars (`pane_width`, `pane_height`, `history_size`, `history_limit`).
- Compare pre/post snapshots around `capture()` / `sample_text()` / `exec()` polling.
  <!-- @claude 2026-03-10: Phase 1.9b implements snapshot comparison for capture/sample_text
       only. exec() uses a two-tier wrap-tolerant parser (parse_sentinel_output) with -ep
       capture + ANSI strip, which handles sentinel detection under reflow without geometry
       snapshots. Geometry-based fidelity for exec() is deferred to Phase 2a where persistent
       monitor state can surface it on ExecOutput. See PR #66. -->
- If client set, pane geometry, or history counters indicate instability, mark the
  result degraded and optionally retry. `resize-window` remains best-effort only when
  mixed interactive clients attach to the same session.

**History-limit management contract**:
- `history-limit` is a tmux window/global option, not a per-client setting.
- tmux applies `history-limit` only to windows/panes created after the option is set;
  existing panes retain their creation-time limit.
- For automation-dedicated sessions/windows, provide setup-time guidance or helpers to
  set `history-limit` before creating the pane.
- For existing panes, rely on `capture-pane -S` range management and explicit fidelity
  metadata (`HistoryTruncated`) rather than promising runtime resize of history.
- Do not automatically lower or restore `history-limit`; callers that mutate
  automation-dedicated windows own any cleanup.

**Proposed API surface** (additive):

```rust
pub enum CaptureNormalizeMode {
    Raw,         // no transformation
    ScreenStable,// newline + width-artifact trimming, ANSI/control preserved
    PlainText,   // explicit opt-in ANSI/control stripping for human/LLM text workflows
}

pub struct CaptureOptions {
    pub history_start: Option<i32>,
    pub normalize: CaptureNormalizeMode,
    pub overlap_lines: usize,
    pub detect_reflow: bool,
}

pub struct CaptureResult {
    pub text: String,
    pub raw_text: Option<String>,
    pub fidelity: OutputFidelity,
}
```

**Mode to field mapping**:

| Mode | tmux flags | `text` / `content` | `raw_text` / `raw_content` | Intended use |
|------|------------|--------------------|-----------------------------|--------------|
| `Raw` | `-p` | tmux-rendered text, no added normalization | `None` | Full-fidelity default, safest when pane type is unknown |
| `ScreenStable` | `-ep` | ANSI/control-preserving normalized stream | exact `-ep` capture before normalization | Shell/log-like panes that need reflow-aware stability without losing terminal data |
| `PlainText` | `-p` | plain-text normalized stream | `None` | Line-oriented matching, human/LLM summarization |

`exec()` uses an internal parser view derived from `ScreenStable` capture for sentinel
matching. That parser view is not a public `CaptureNormalizeMode`.

**Execution model**:
- `capture()` defaults to `Raw`.
- `sample_text()` defaults to `Raw`.
- `capture_session()` / `capture_all()` default to `Raw` on each pane.
- `exec()` derives its internal sentinel parser view from `ScreenStable` capture.
- ANSI/control stripping only occurs in explicit `PlainText` mode or sink-local matcher
  views.
- `capture_result()` / `sample_text_result()` / `capture_session_with_options()` are
  the metadata-bearing APIs.
- `capture()` / `sample_text()` / `capture_session()` remain convenience wrappers that
  return only `.text`.

**Operational guidance**:
- Reliable automation requires a dedicated tmux session/socket, or at minimum no mixed
  interactive clients during the command/capture window.
- `window-size manual` plus `resize-window -x/-y` reduce churn but do not guarantee
  determinism when differently sized interactive clients attach later.

---

## Open Concerns

These require resolution before or during implementation. They are not blockers for Phase 1
but must be addressed before the library is used in production.

### OC1: ~~FIFO Reliability Under Load~~ (OUT OF SCOPE)

<!-- @claude 2026-03-17 — pipe-pane fallback removed from scope (DC10 + DC22 tmux 3.1+ baseline).
     Control mode is the sole monitoring strategy; FIFOs are never used. -->
**Resolved by DC10 + DC22** — control mode is the sole monitoring strategy (tmux 3.1+
baseline guarantees availability). FIFOs are not used and this concern does not apply.

### OC2: ~~Output Interleaving~~ (OUT OF SCOPE)

<!-- @claude 2026-03-17 — pipe-pane fallback removed from scope (DC10 + DC22 tmux 3.1+ baseline).
     Control mode framing eliminates interleaving; no fallback path exists. -->
**Eliminated by DC10** — control mode provides framed `%output` messages with pane ID
attribution. No interleaving possible. No fallback path exists to reintroduce this risk.

### OC3: SSH Agent Availability

The prototype requires a running `ssh-agent` with loaded keys. If the agent is unavailable
or has no matching keys, the error message should be actionable.

**Proposal**: On auth failure, enumerate the attempted key fingerprints and suggest
`ssh-add` if no keys were found.

<!-- @claude 2026-03-20: Actionable agent error messages implemented in Phase 2a.1.
     DC26 extends this further by providing an alternative auth path (`identity-file`)
     that eliminates the agent dependency entirely for CI/agentless workflows. -->

### OC4: Tmux Version Compatibility

<!-- @claude 2026-03-17 — Updated to reflect tmux 3.1+ baseline (DC22). Removed pipe-pane
     from feature list, committed to 3.1 minimum, aligned CI matrix with PLAN. -->
`list-panes -F` format strings and control mode behavior vary across tmux versions.
The minimum supported version is **tmux 3.1** (DC22), which guarantees control mode
availability.

**Proposal**: Runtime detection via `tmux -V` during the discovery phase. The library
validates that required features are available (control mode `%output` framing,
`capture-pane -p`, `#{pane_id}` format variable) and returns a clear error if the
detected version is below 3.1. Phase 4 CI will test against tmux 3.1, 3.x latest,
and latest to verify compatibility.

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
| `thiserror` | 2.x | Typed error enum (`Error`) | Yes |
| `anyhow` | 1.x | Dev-dependency only (examples, tests) | No |
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

### Phase 1.9a: Capture Fidelity Types + Explicit Modes (Usability Gate)

**Goal**: Make capture/sampling output reliably usable for downstream consumers before
monitoring/sink work layers on top.

**Tasks**:
1. Implement `CaptureNormalizeMode`, `CaptureOptions`, and `CaptureResult`
2. Add `capture_*_with_options` / `sample_*_with_options` APIs with fidelity metadata
3. Keep `capture()` / `sample_text()` / `capture_session()` as explicit `Raw` wrappers
4. Implement `ScreenStable` and `PlainText` normalization contracts
5. Implement the internal `exec()` parser view for wrap-tolerant sentinel detection

**Deliverable**: On-demand capture APIs return mode-specific text plus machine-readable
fidelity signals (`OutputFidelity`) for degraded/reflow cases.

**Acceptance criteria**:
- Public mode/field semantics are explicit and unambiguous
- `capture()` / `sample_text()` / bulk capture wrappers preserve `Raw` semantics
- `exec()` sentinel parsing is more wrap-tolerant without exposing a new public mode

### Phase 1.9b: Mixed-Client Stabilization

**Goal**: Add best-effort resync and operational guidance for shared-session workflows.

**Tasks**:
1. Implement geometry/history instability detection
2. Implement overlap-aware resync for incremental sampling
3. Add setup-time helpers/guidance for automation windows with sufficient `history-limit`
4. Surface `HistoryTruncated` / `OverlapResync` metadata in tests and docs

**Acceptance criteria**:
- Mixed-client resize churn is detectable and surfaced in metadata
- Ambiguous overlap falls back to resync instead of heuristic merging
- Existing-pane `history-limit` limits are documented and tested as non-retroactive

### Phase 2a: SSH Transport + Minimal Monitoring (Vertical Slice)

**Goal**: Add `SshTransport` for remote hosts. Implement a thin monitoring vertical slice:
single-target monitor with one `SendKeys` action type and explicit shutdown API. No full
rule engine, no reconnection, no config deserialization yet.

**Tasks**:
1. Implement `SshTransport` in `transport.rs`: russh-based `exec()` and `open_shell()`,
   host key verification (fixes P1), configurable timeout (fixes P10)
2. Implement `monitor.rs`: control mode parser (`%output %<pane_id> <data>`), per-pane
   chunk assembly, fidelity metadata propagation into `TargetOutput`, single
   hardcoded-pattern detection, `SendKeys` action dispatch via
   bounded queue (DC4), `MonitorHandle` with explicit `shutdown()` API
3. Wire monitoring into `HostHandle::start_monitoring()` for localhost and SSH
5. Unit tests: `SshTransport` via mock SSH server or `MockTransport`, monitor loop with
   canned control-mode output, action dispatch ordering

**Deliverable**: Library that monitors panes on localhost or one SSH host, publishes
deterministic `TargetOutput` into the bus, and supports explicit shutdown. Library policy
logic remains out of scope.

**Acceptance criteria**:
- All Phase 1 operations work identically over `SshTransport`
- Monitoring starts on localhost and publishes deterministic output into `OutputBus`
- `monitor_handle.shutdown()` cleanly stops monitoring and cleans up
- Localhost monitoring works without any SSH configuration
- Monitor output includes deterministic source sequencing and fidelity annotations

### Phase 2b: Transcript / History Adapters

**Goal**: Build the consumer-side adapters that turn source-routed stream output into
conversation/transcript-friendly artifacts for external LLM/classifier analysis.

**Tasks**:
1. Extend `Subscription` with transcript/history-oriented adapters (`joined()`,
   plain-text helpers, bounded history windows)
2. Add lightweight consumer-owned predicate helpers where useful, without introducing
   a built-in matcher DSL as an active dependency
3. Define transcript/history data structures with stable source labels and clear
   chunk/turn boundaries for downstream analysis
4. Add focused examples and API snippets showing external-agent consumption patterns
5. Unit/integration tests: source labeling, coalescing, bounded history behavior,
   gap visibility, transcript determinism

**Deliverable**: Library produces transcript-friendly history views from live tmux output
without embedding policy or decision logic.

**Acceptance criteria**:
- A subscription can be turned into a bounded multi-source transcript/history stream
- Source attribution remains stable and explicit across hosts/sessions/panes
- Gap/backpressure markers remain visible to downstream consumers
- Examples demonstrate an external consumer building history and acting separately

### Phase 2c: Fleet Coordination + Routed Control

**Goal**: Make `Fleet` the coordination layer for multi-host monitoring, aggregation,
target-alias lookup, and routed control actions.

**Tasks**:
1. Implement `fleet.rs` as a programmatic registry: connect hosts, assign aliases,
   expose `host()`, `hosts()`, `output_bus()`, and target alias bindings
2. Aggregate monitoring lifecycle across hosts with `start_monitoring()`,
   `start_monitoring_host()`, `stop_monitoring_host()`, and `shutdown()`
3. Add convenience routing helpers (`send_text`, `send_keys`, `capture`, `target`)
   so external agents can act through Fleet without reimplementing lookup logic
4. Preserve explicit per-host status tracking and per-target error isolation
5. Unit/integration tests: multi-host connect with one failure, alias conflict
   detection, target alias bind/find/unbind, routed action correctness

**Deliverable**: Multi-host registry and routing layer that pairs naturally with
external policy engines.

**Acceptance criteria**:
- `Fleet` can connect to multiple hosts concurrently with per-host isolation
- `Fleet::output_bus()` exposes unified stream aggregation for all monitored hosts
- `Fleet` can route control actions back to the correct bound host/target
- Target alias bindings make stable target lookup possible without config DSLs

### Phase 3: Agent-Facing API + CLI / Examples

**Goal**: Expose the simplified registry/stream/history/control model cleanly in the
public API and supporting tools.

**Tasks**:
1. Re-export the active consumer-facing surface from `lib.rs`:
   `Fleet`, `HostHandle`, `Target`, `OutputBus`, `Subscription`, `SinkKind`,
   `JoinedStream`, transcript/history types, and existing control/capture APIs
2. Keep the CLI/examples focused on connection, monitoring, capture, send/exec, and
   transcript inspection rather than built-in rule execution
3. Provide agent-oriented examples that show:
   - connect/register hosts
   - monitor and build transcript/history
   - route follow-up actions back through `Fleet` / `Target`
4. Add smoke tests and docs that verify the external-agent workflow end to end

**Deliverable**: Public API, examples, and optional CLI surfaces that tutorialize the
external-agent workflow instead of a built-in automator.

**Acceptance criteria**:
- Public docs present `Fleet` as registry/aggregation/routing rather than rule runner
- Examples cover monitor → transcript/history → external decision → routed control
- `stream_pane` distinguishes raw monitor output (`--mode monitor`) from rendered
  TUI watching (`--mode render`) so stream/history semantics stay explicit
- CLI/examples remain useful without a config-driven rule engine

### Phase 4: Hardening + Testing

**Goal**: Production readiness for long-lived streaming, external-agent loops, and
multi-host/Fleet operation.

**Tasks**:
1. Expand `MockTransport` coverage for integration tests (OC6)
2. Tmux version detection and compatibility check (OC4)
3. Actionable SSH agent error messages (OC3)
4. ~~FIFO-vs-file investigation and decision (OC1)~~ — OUT OF SCOPE (resolved by DC10 + DC22)
5. ~~Line interleaving mitigation (OC2)~~ — OUT OF SCOPE (resolved by DC10 + DC22)
6. Reconnecting monitor supervision with bounded retry/backoff
7. Explicit stream discontinuity artifacts distinct from subscriber `Gap`
8. Fresh snapshot anchoring after reconnect, reflected in transcript/history
9. Fleet/host health visibility for reconnecting/degraded/failed streaming state
10. End-to-end test with Docker (SSH + tmux)
11. Document minimum tmux version, known limitations, and performance characteristics

**Deliverable**: Library with test coverage, documented limitations, and CI integration.

### Phase 5 (Future): TUI Interface

**Goal**: Terminal UI for interactive multi-target management.

**Technology**: [ratatui](https://ratatui.rs/) — a Rust library for building terminal
user interfaces.

**Status**: Phase 5.1 and 5.2 shipped (DC32). The first TUI delivery is a
split-screen REPL mode, not a standalone dashboard:
- `tui on` / `tui off` commands in `examples/repl`
- top-frame mirror of a watched remote session via `HistoryHandle`
- bottom-frame REPL prompt and command history
- transcript/history-oriented rendering (Phase 5a)

The TUI mirror consumer lives in the example layer (`examples/repl/tui_mirror.rs`),
not in `libs/tmux`, consistent with DC11 and DC32. It consumes the library's
`OutputBus`/`Subscription`/`HistoryHandle` surface and manages its own rendering
cadence locally. `ratatui` and `crossterm` are dev-dependencies only.

Follow-on TUI work can grow toward:
- standalone multi-pane/dashboard surfaces
- session/window/pane tree navigation across targets
- interactive send-keys input via `Fleet`, `HostHandle`, or `Target`
- full terminal-state mirroring for cursor-addressed TUIs (Phase 5b)

If later multiple binaries need the same TUI lifecycle, the consumer can be
promoted to a reusable helper crate or feature-gated module.

For TUI fidelity/reliability constraints under mixed-client attachment and resizing,
see [`TUI.md`](./TUI.md).

---

## Appendix A — Historical Automation Direction

The following material is preserved for historical context. It represents the
pre-2026-03-20 direction where Motlie hosted more of the automation/policy loop
internally. That direction is not the active plan, but it may still inform future
reconsideration if built-in automation becomes a primary product goal.

### A.1 Matcher DSL (`MatcherKind`)

The historical design introduced a closed, statically dispatched `MatcherKind` enum
with variants such as `Regex`, `Substring`, `LineCount`, `WordList`, `AllOf`,
`AnyOf`, and `Not`. The intent was to let Motlie host content matching directly in the
stream pipeline via subscription adapters like `.filter()` and `.react()`.

Why it was deferred:
- the dominant use case shifted to external LLM/classifier analysis over transcript/history
- introducing a built-in matcher DSL pushed the library toward owning policy decisions
- transcript/history ergonomics became more valuable than in-library trigger semantics

### A.2 In-Library Reaction Path (`ActionHandle`)

The historical design introduced `ActionHandle`, `ActionRequest`, and `SinkAction` so
consumers could detect output and enqueue actions back into tmux without leaving the
library. This paired naturally with `.react()` and cooldown logic.

Why it was deferred:
- external agents can already use `Fleet`, `HostHandle`, and `Target` directly
- an internal reaction path makes Motlie look like a rule engine rather than a substrate
- the same control primitives remain available without a separate reaction API

### A.3 Built-In Automation Config

The historical direction also introduced `TmuxAutomatorConfig`, `HostTarget`,
`TriggerRule`, `CompiledRule`, `Action`, and `ReconnectPolicy` as a declarative config
surface for built-in automation and config-driven CLI workflows.

Why it was deferred:
- the simplified active path is programmatic and agent-oriented, not config-first
- external policy engines want streaming/history plus routed control, not Motlie-owned
  declarative rule evaluation
- config/reconnect complexity is easier to justify once a concrete built-in automation
  product direction exists

### A.4 Historical Fleet / CLI Shape

The older Phase 3 direction treated `Fleet` as the core of a config-driven automator:
`Fleet::new(config)`, `connect_all()`, `start_monitoring(rules)`, plus a CLI centered on
config files and built-in rule execution.

Why it was deferred:
- the active direction instead treats `Fleet` as a registry, aggregator, and routing layer
- examples and CLI value are clearer when focused on connect/monitor/history/control
- external agents can supply policy without requiring Motlie to own the automation loop

If this historical path is revived later, it should be re-evaluated against the then-current
stream/history/Fleet APIs rather than reinstated wholesale.

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
- [tmux TUI.md](./TUI.md) — TUI-specific reliability and capture-fidelity policy
- [tmux SFTP.md](./SFTP.md) — host-level file transfer design deep dive
