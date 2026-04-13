# motlie-driver Design

## Status

Implemented feasibility slice on `main`:
- `libs/driver` exists and builds
- `CommandEngine<C, S>` is the generic execution core
- `CommandSet<C>` is the typed command-family contract
- `driver::commands::tmux` is the first real adapter
- shared tmux REPL/TUI frontends live in `driver::tmux_frontend`
- top-level app assembly lives in `bins/tmux/driver`

This document describes the code that exists today. Future adapters such as
VMM/VNET/VFS are design targets only and are not scaffolded as code on `main`.

## Crate Boundaries

### `libs/driver`

Owns:
- command parsing and dispatch
- completion analysis and merge
- driver-local history buffers
- frontend-neutral command output/effects
- tmux adapter and tmux-specific shared frontends
- terminal-side utilities such as asciicast recording

Does not own:
- tmux lifecycle semantics
- VM/network/filesystem business logic
- remote process/resource discovery beyond what adapters call through public APIs

### Resource crates

Resource crates continue to own their public lifecycle APIs.

Examples:
- `motlie-tmux` owns host/session/target/monitor operations
- future `motlie-vmm` should own `prepare/boot/ready/exec/shutdown`
- future `motlie-vnet` should own network backend lifecycle

`motlie-driver` consumes those APIs. It does not redefine them.

### Top-level applications

Applications assemble:
- a concrete context
- a concrete `CommandSet`
- one or more frontends
- feature gates for which adapters/frontends are compiled

Current real application:
- `bins/tmux/driver`

## Feature Topology

Current `motlie-driver` features:

- `repl`
  - enables `reedline` integration and the tmux line REPL path
- `tui`
  - enables `ratatui + crossterm` integration and the tmux split-screen UI
- `commands-tmux`
  - enables the tmux command adapter and shared tmux frontends
- `term-vt100`
  - reserved for future terminal-emulation work
- `term-shadow`
  - reserved for future terminal-shadow work

Important current constraint:
- future adapter feature names for VMM/VNET/VFS are documented work items, not implemented modules on `main`

## Core Runtime Model

The generic driver surface is intentionally small:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEffect {
    ExitShell,
    EnterTui,
    ExitTui,
}

#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub lines: Vec<String>,
    pub effects: Vec<CommandEffect>,
}

#[async_trait::async_trait]
pub trait CommandSet<C>: Sized {
    type CompletionContext: Send + 'static;

    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self>;
    fn completion_context(context: &C) -> Self::CompletionContext;
    fn help(topic: &[String]) -> Option<String> { None }
    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> { Vec::new() }
    async fn execute(self, context: &mut C) -> DriverResult<CommandOutput>;
}

pub struct CommandEngine<C, S> {
    /* context + typed command family */
}
```

`CommandEngine` owns:
- the mutable session context `C`
- `run_line()` and `run_argv()`
- built-in `help`
- built-in `quit` / `exit`
- static + dynamic completion merge

It does not own:
- editor loops
- TUI rendering
- subsystem business logic

## clap Integration

`clap` is the source of truth for static command structure.

The driver uses it for:
- command parsing
- subcommand discovery
- flag completion
- help rendering fallback

The driver adds:
- root-name prefixing for REPL-style input
- completion analysis for the current command path / active arg
- built-in command handling before command-family dispatch

## Completion Architecture

Completion has two layers.

### Static completion

Derived from the `clap::Command` tree:
- subcommands
- long flags
- short flags

### Dynamic completion

Provided by the adapter through:
- `CommandSet::completion_context(&C)`
- `CommandSet::complete(request, &CompletionContext)`

This split matters:
- execution owns `&mut C`
- completion needs a sync, read-only snapshot

The associated `CompletionContext` type on `CommandSet` is the actual mechanism used in code
to solve that borrowing boundary.

## Help Architecture

`help` is engine-owned.

Its content is composed from the same command family that drives parsing:

1. `CommandEngine` checks whether the input starts with `help`
2. it asks `CommandSet::help(topic)` for an adapter-specific rich help topic
3. if no rich topic exists, it renders `clap` help from the composed command tree

That means:
- disabled command families contribute no help
- rich help stays near the adapter that owns the command semantics
- static command help never drifts away from the `clap` schema

Current real example:
- `driver::commands::tmux::help()` provides richer topics for `stream`, `monitor`,
  `mirror`, `capture`, `history`, and `tui`

## tmux Vertical Slice

The first full slice is tmux-only.

### Shared state

`TmuxState` owns:
- connected `HostHandle`
- discovered session names
- discovered target names
- owned session names created by this driver session
- at most one active watch
- at most one active stream
- current mirror snapshot
- driver-local retained mirror history

This is intentionally a narrow slice:
- one active watch/stream at a time
- no attach/import persistence yet
- no generalized resource-mode metadata yet

### Shared frontends

The tmux product frontends are shared library code:
- `driver::tmux_frontend::run_tmux_repl(...)`
- `driver::tmux_frontend::run_tmux_tui(...)`

That code is consumed by:
- `bins/tmux/driver`
- `libs/driver/examples/tmux_repl.rs`
- `libs/driver/examples/tmux_tui.rs`

This replaced the earlier duplication between:
- bin-local `plain.rs` / `ui.rs`
- example-local `common/tmux_plain.rs` / `common/tmux_ui.rs`

## History Buffer Design

`driver::history` is generic and driver-owned.

It is not tmux-specific and not terminal-specific.

Current types:

```rust
pub struct HistoryRecord<T> {
    pub seq: u64,
    pub at: SystemTime,
    pub item: T,
}

pub struct HistoryPage<T> {
    pub items: Vec<HistoryRecord<T>>,
    pub next_after: Option<u64>,
    pub oldest_available: Option<u64>,
    pub newest_available: Option<u64>,
}

pub struct HistoryBuffer<T> { /* bounded FIFO */ }
```

Current real use:
- tmux watch/stream/capture/history commands retain local mirror history
- plain REPL follow mode pages forward over that retained history
- `mirror history --after ... --limit ...` exposes the retained local history

Motivation:
- the retained buffer is a better primitive than periodic stdout dumps
- it can later support RPC/admin pagination
- it can later support teeing driver-owned tracing/log events into another buffer

## Asciicast Recording

`driver::term::asciicast` is a driver-owned recorder.

Current real behavior:
- the tmux driver binary can record REPL and TUI command-flow events
- plain REPL live-follow output is recorded
- startup TUI mode records entry/exit and command/output flow

Current limitation:
- this is not a full alternate-screen frame recorder
- it does not yet serialize every rendered TUI frame as a visual replay stream
- it is currently best understood as a driver-owned terminal/session artifact format

That is why the implementation includes an explicit comment that the event timestamps are
delta-based and are not aiming for strict asciinema compatibility.

## Top-level tmux driver binary

`bins/tmux/driver` exists outside `motlie-tmux` on purpose:
- `motlie-driver` depends on `motlie-tmux`
- placing the driver binary in the `motlie-tmux` package would create a package cycle

The binary is intentionally assembly-only:
- parse top-level args
- connect `TmuxState`
- choose REPL or TUI
- optionally create an asciicast recorder

## What Is Deliberately Not Implemented Yet

Not implemented on `main`:
- VMM/VNET/VFS command adapters
- generic resource-mode enums like `Owned | Imported | Ephemeral`
- generic remote-proxy lifecycle support
- generic shared TUI abstraction for non-tmux consumers
- full-screen frame-accurate TUI capture

Those remain valid future directions, but they are not part of the implemented contract of
this PR and should not be documented as if they already exist.
